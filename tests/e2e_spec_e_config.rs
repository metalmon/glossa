//! Spec E e2e: the declarative TOML deployment config (`--config`) drives a real `kb` server —
//! the file alone can stand up the server, a CLI flag overrides the file per-setting, and a secret
//! placed in the file is rejected at load time.
#![cfg(feature = "e2e")]

#[path = "e2e/harness.rs"]
mod harness;

use harness::*;

/// Build a `glossa.toml` body. Paths use single-quoted TOML literal strings so Windows
/// backslashes are not treated as escapes.
fn config_toml(roots: &[String], state_dir: &std::path::Path, bind: &str) -> String {
    let roots_list = roots
        .iter()
        .map(|r| format!("'{r}'"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "[corpus]\nroots = [{roots_list}]\nstate_dir = '{}'\n\n[server]\ntransport = \"streamable-http\"\nbind = \"{bind}\"\n",
        state_dir.display()
    )
}

fn write_config(body: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("glossa.toml");
    std::fs::write(&p, body).unwrap();
    (dir, p)
}

#[test]
fn config_file_alone_starts_and_serves() {
    let c = Corpus::with_files(&[("a.md", "# Doc\n\ncontent with a configword marker.\n")]);
    let state = state_dir();
    let port = free_port();
    let body = config_toml(
        &[c.root_arg("docs")],
        state.path(),
        &format!("127.0.0.1:{port}"),
    );
    let (_cfgdir, cfgpath) = write_config(&body);

    // Index the corpus first (same config file, so it targets the SAME roots + state_dir the server
    // will use) so the server's first search is against an already-built index — no startup-indexing
    // race, and no need to retry the same query (which would hit the session's identical-query dedup).
    assert_cmd::Command::cargo_bin("kb")
        .unwrap()
        .arg("index")
        .arg("--config")
        .arg(&cfgpath)
        .assert()
        .success();

    // No flags beyond --config: roots, state_dir, transport and bind all come from the file.
    let mut cmd = kb_command();
    cmd.arg("mcp").arg("--config").arg(&cfgpath);
    let base = format!("http://127.0.0.1:{port}");
    let probe = base.clone();
    let server = spawn_kb(
        cmd,
        base,
        port,
        Vec::new(),
        move || matches!(try_request(&probe, "GET", "/ready", &[], None), Ok(r) if r.status == 200),
    );

    let mcp = McpClient::connect(server.base());
    assert!(
        mcp.search_text("configword").contains("configword"),
        "config-driven server should serve a search.\nstderr:\n{}",
        server.stderr()
    );
    // Keep fixtures alive until the server is dropped.
    drop(server);
    drop(c);
    drop(state);
}

#[test]
fn cli_bind_overrides_config_bind() {
    let c = Corpus::with_files(&[("a.md", "# Doc\n\nprecedence content.\n")]);
    let state = state_dir();
    let file_port = free_port(); // what the config file asks for
    let flag_port = free_port(); // what --bind forces (must win)
    let body = config_toml(
        &[c.root_arg("docs")],
        state.path(),
        &format!("127.0.0.1:{file_port}"),
    );
    let (_cfgdir, cfgpath) = write_config(&body);

    // ServerBuilder passes --bind flag_port; also add --config. The flag must win.
    let builder = ServerBuilder::new()
        .root(c.root_arg("docs"))
        .state_dir(state.path())
        .arg("--config")
        .arg(cfgpath.display().to_string());
    let cmd = builder.command(flag_port);
    let base = format!("http://127.0.0.1:{flag_port}");
    let probe = base.clone();
    let server = spawn_kb(
        cmd,
        base,
        flag_port,
        Vec::new(),
        move || matches!(try_request(&probe, "GET", "/ready", &[], None), Ok(r) if r.status == 200),
    );

    // The flag port is up (start already gated it); the file port must NOT be listening.
    assert_eq!(http_get(server.base(), "/ready", &[]).status, 200);
    let file_base = format!("http://127.0.0.1:{file_port}");
    assert!(
        try_request(&file_base, "GET", "/ready", &[], None).is_err(),
        "the config-file bind port {file_port} must not be listening when --bind overrides it"
    );
    drop(server);
    drop(c);
    drop(state);
}

#[test]
fn secret_in_config_is_rejected() {
    // A token in the file must be refused at load time (before serving), with a targeted error.
    let c = Corpus::with_files(&[("a.md", "# Doc\n\nx\n")]);
    let state = state_dir();
    let port = free_port();
    let mut body = config_toml(
        &[c.root_arg("docs")],
        state.path(),
        &format!("127.0.0.1:{port}"),
    );
    body.push_str("auth_token = \"deadbeef\"\n");
    let (_cfgdir, cfgpath) = write_config(&body);

    assert_cmd::Command::cargo_bin("kb")
        .unwrap()
        .arg("mcp")
        .arg("--config")
        .arg(&cfgpath)
        .assert()
        .failure()
        .stderr(predicates::str::contains("token"));
}
