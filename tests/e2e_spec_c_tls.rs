//! Spec C TLS/mTLS e2e: native TLS termination on the streamable-http listener, driven with a real
//! blocking rustls client against the real `kb` binary built with `--features tls`.
#![cfg(all(feature = "e2e", feature = "tls"))]

#[path = "e2e/harness.rs"]
mod harness;

use harness::tls::{https_get, make_ca, make_leaf, TestCa};
use harness::*;
use std::path::PathBuf;

/// Write `cert`/`key` (and optionally a client-CA) into `dir`, returning their paths.
fn write_tls_files(
    dir: &std::path::Path,
    cert_pem: &str,
    key_pem: &str,
    client_ca_pem: Option<&str>,
) -> (PathBuf, PathBuf, Option<PathBuf>) {
    let cert = dir.join("server.pem");
    let key = dir.join("server.key");
    std::fs::write(&cert, cert_pem).unwrap();
    std::fs::write(&key, key_pem).unwrap();
    let ca = client_ca_pem.map(|pem| {
        let p = dir.join("client-ca.pem");
        std::fs::write(&p, pem).unwrap();
        p
    });
    (cert, key, ca)
}

/// Start a TLS `kb` server. When `client_ca_pem` is `Some`, mTLS is enabled and the readiness probe
/// presents `ready_identity` (a leaf signed by the trusted CA). Returns the handle + port.
fn start_tls_server(
    ca: &TestCa,
    client_ca_pem: Option<&str>,
    ready_identity: Option<(String, String)>,
) -> (ServerHandle, u16) {
    let tlsdir = tempfile::tempdir().unwrap();
    let (server_cert, server_key) = make_leaf(ca, "127.0.0.1");
    let (cert_path, key_path, ca_path) =
        write_tls_files(tlsdir.path(), &server_cert, &server_key, client_ca_pem);

    let c = Corpus::with_files(&[("a.md", "# Doc\n\ntls served content.\n")]);
    let state = state_dir();
    let port = free_port();

    let builder = ServerBuilder::new().root(c.root_arg("docs")).state_dir(state.path());
    let mut cmd = builder.command(port);
    cmd.arg("--tls-cert").arg(&cert_path).arg("--tls-key").arg(&key_path);
    if let Some(p) = &ca_path {
        cmd.arg("--tls-client-ca").arg(p);
    }

    let ca_pem = ca.cert_pem.clone();
    let base = format!("https://127.0.0.1:{port}");
    // Keep the cert dir, corpus and state dir alive for the life of the server.
    let keep = vec![tlsdir, c.into_dir(), state];
    let server = spawn_kb(cmd, base, port, keep, move || {
        let identity = ready_identity
            .as_ref()
            .map(|(crt, k)| (crt.as_str(), k.as_str()));
        matches!(
            https_get("127.0.0.1", port, &ca_pem, identity, "/ready"),
            Ok(r) if r.status == 200
        )
    });
    (server, port)
}

#[test]
fn tls_serves_health_when_trusting_the_ca() {
    let ca = make_ca();
    let (server, port) = start_tls_server(&ca, None, None);

    let resp = https_get("127.0.0.1", port, &ca.cert_pem, None, "/health")
        .expect("HTTPS GET /health trusting the CA should succeed");
    assert_eq!(resp.status, 200);
    assert!(resp.body.contains("ok"), "health body: {:?}", resp.body);
    drop(server);
}

#[test]
fn mtls_requires_a_client_certificate() {
    let ca = make_ca();
    // The readiness probe (and the positive assertion) present a client leaf signed by the CA.
    let (client_cert, client_key) = make_leaf(&ca, "glossa-e2e-client");
    let (server, port) = start_tls_server(
        &ca,
        Some(&ca.cert_pem),
        Some((client_cert.clone(), client_key.clone())),
    );

    // No client certificate -> handshake/connection must fail.
    let no_cert = https_get("127.0.0.1", port, &ca.cert_pem, None, "/health");
    assert!(
        no_cert.is_err(),
        "a client presenting no certificate must be rejected under mTLS, got: {no_cert:?}"
    );

    // A client presenting a leaf signed by the trusted CA -> 200.
    let with_cert = https_get(
        "127.0.0.1",
        port,
        &ca.cert_pem,
        Some((&client_cert, &client_key)),
        "/health",
    )
    .expect("a client cert signed by the trusted CA should be accepted");
    assert_eq!(with_cert.status, 200);
    assert!(with_cert.body.contains("ok"), "health body: {:?}", with_cert.body);
    drop(server);
}
