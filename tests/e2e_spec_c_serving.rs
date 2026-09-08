//! Spec C e2e: streamable-http serving surface driven over real sockets against the real `kb`
//! binary — probe endpoints, bearer auth, body-limit, the startup safety interlock, graceful
//! shutdown, and (best-effort) the per-IP rate-limit.
#![cfg(feature = "e2e")]

#[path = "e2e/harness.rs"]
mod harness;

use harness::*;

fn corpus() -> Corpus {
    Corpus::with_files(&[("a.md", "# Doc\n\nsome indexed content for the serving tests.\n")])
}

#[test]
fn probe_endpoints_are_open_and_correct() {
    let c = corpus();
    let state = state_dir();
    let server = ServerBuilder::new()
        .root(c.root_arg("docs"))
        .state_dir(state.path())
        .start();

    let health = http_get(server.base(), "/health", &[]);
    assert_eq!(health.status, 200);
    assert!(health.body.contains("ok"), "health body: {:?}", health.body);

    let ready = http_get(server.base(), "/ready", &[]);
    assert_eq!(ready.status, 200);
    assert!(ready.body.contains("ready"), "ready body: {:?}", ready.body);

    let metrics = http_get(server.base(), "/metrics", &[]);
    assert_eq!(metrics.status, 200);
    assert!(
        metrics.body.contains("glossa_up"),
        "metrics missing glossa_up: {:?}",
        metrics.body
    );
}

#[test]
fn bearer_auth_gates_the_mcp_endpoint() {
    let c = corpus();
    let state = state_dir();
    let server = ServerBuilder::new()
        .root(c.root_arg("docs"))
        .state_dir(state.path())
        .auth_token("secret-e2e")
        .start();

    // Probe endpoints stay open even with auth enabled.
    assert_eq!(http_get(server.base(), "/health", &[]).status, 200);

    let init_body = br#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"e2e","version":"0"}}}"#;

    // No Authorization header -> 401.
    let unauth = http_post(
        server.base(),
        "/mcp",
        &[
            ("Accept", MCP_ACCEPT),
            ("Content-Type", "application/json"),
        ],
        init_body,
    );
    assert_eq!(unauth.status, 401, "expected 401, body: {:?}", unauth.body);

    // With the right bearer token -> initialize succeeds and returns a session id.
    let ok = http_post(
        server.base(),
        "/mcp",
        &[
            ("Accept", MCP_ACCEPT),
            ("Content-Type", "application/json"),
            ("Authorization", "Bearer secret-e2e"),
        ],
        init_body,
    );
    assert!(
        ok.status == 200 || ok.status == 202,
        "authorized initialize status: {} body: {:?}",
        ok.status,
        ok.body
    );
    assert!(
        ok.header("mcp-session-id").is_some(),
        "authorized initialize carried no Mcp-Session-Id: {ok:?}"
    );

    // And the high-level client works end-to-end through auth.
    let mcp = McpClient::connect_with_auth(server.base(), "secret-e2e");
    let _ = mcp.search_text("content");
}

#[test]
fn body_limit_rejects_oversized_post() {
    let c = corpus();
    let state = state_dir();
    let server = ServerBuilder::new()
        .root(c.root_arg("docs"))
        .state_dir(state.path())
        .env("GLOSSA_MCP_MAX_BODY_BYTES", "1024")
        .start();

    // A >1KiB body must be rejected with 413 before it reaches the MCP service.
    let big = vec![b'x'; 4096];
    let resp = http_post(
        server.base(),
        "/mcp",
        &[
            ("Accept", MCP_ACCEPT),
            ("Content-Type", "application/json"),
        ],
        &big,
    );
    assert_eq!(resp.status, 413, "expected 413 Payload Too Large, got {}", resp.status);
}

#[test]
fn startup_interlock_refuses_nonloopback_without_auth() {
    // assert_cmd (no long-lived server): a non-loopback bind with no token/tls/insecure must refuse
    // to start, non-zero exit, with a clear stderr message.
    let c = corpus();
    let state = state_dir();
    let port = free_port();
    assert_cmd::Command::cargo_bin("kb")
        .unwrap()
        .arg("mcp")
        .arg("--transport")
        .arg("streamable-http")
        .arg("--bind")
        .arg(format!("0.0.0.0:{port}"))
        .arg("--root")
        .arg(c.root_arg("docs"))
        .arg("--state-dir")
        .arg(state.path())
        .assert()
        .failure()
        .stderr(predicates::str::contains("refusing to serve"));
}

#[cfg(unix)]
#[test]
fn sigterm_shuts_down_gracefully_with_exit_zero() {
    let c = corpus();
    let state = state_dir();
    let mut server = ServerBuilder::new()
        .root(c.root_arg("docs"))
        .state_dir(state.path())
        .start();

    // Prove it is live, then send SIGTERM and require a clean exit within a few seconds.
    assert_eq!(http_get(server.base(), "/health", &[]).status, 200);
    let pid = server.pid() as i32;
    // SAFETY: sending SIGTERM to our own child process.
    unsafe {
        libc::kill(pid, libc::SIGTERM);
    }
    let status = server
        .wait_for_exit(std::time::Duration::from_secs(5))
        .expect("server should exit within 5s of SIGTERM");
    assert!(status.success(), "graceful shutdown should exit 0, got {status}");
}

#[test]
fn rate_limit_sheds_excess_mcp_requests() {
    // Best-effort but deterministic: per-IP token bucket at 1/s (burst 1) => the first /mcp request
    // in a tight burst passes, the rest get 429. From loopback the peer IP is the bucket key.
    let c = corpus();
    let state = state_dir();
    let server = ServerBuilder::new()
        .root(c.root_arg("docs"))
        .state_dir(state.path())
        .env("GLOSSA_MCP_RATE_LIMIT_PER_SEC", "1")
        .start();

    let init_body = br#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"e2e","version":"0"}}}"#;
    let mut saw_429 = false;
    for _ in 0..12 {
        if let Ok(r) = try_request(
            server.base(),
            "POST",
            "/mcp",
            &[
                ("Accept", MCP_ACCEPT),
                ("Content-Type", "application/json"),
            ],
            Some(init_body),
        ) {
            if r.status == 429 {
                saw_429 = true;
                break;
            }
        }
    }
    assert!(
        saw_429,
        "expected at least one 429 from the per-IP rate limit.\nstderr:\n{}",
        server.stderr()
    );
}
