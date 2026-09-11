//! Reusable end-to-end test harness: spawns the REAL `kb` binary and drives it over REAL sockets.
//!
//! This file is NOT an auto-run test binary — it lives in a subdirectory so cargo won't compile it
//! on its own. Each e2e test file pulls it in with:
//!
//! ```ignore
//! #[path = "e2e/harness.rs"]
//! mod harness;
//! ```
//!
//! Everything here is dependency-free beyond what the workspace already carries: the plaintext HTTP
//! client is a minimal raw HTTP/1.1 speaker over `std::net::TcpStream`; the TLS client (behind the
//! `tls` cargo feature) is a blocking `rustls` client. `serde_json` is a normal dependency and
//! `tempfile`/`assert_cmd` are dev-dependencies — no new crates are introduced.
#![allow(dead_code)] // not every e2e file uses every helper

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use tempfile::TempDir;

/// How long `spawn_*` waits for `GET /ready` to go green before declaring the server dead.
const READY_TIMEOUT: Duration = Duration::from_secs(20);
/// Socket read timeout backstop (the client always sends `Connection: close`, so a well-behaved
/// server closes and reads terminate at EOF; this only guards a misbehaving/hung server).
const IO_TIMEOUT: Duration = Duration::from_secs(15);

// ---------------------------------------------------------------------------------------------
// Ports, binary path, fixtures
// ---------------------------------------------------------------------------------------------

/// Grab a currently-free loopback TCP port by binding `:0`, reading the assigned port, and dropping
/// the listener. Inherently racy (another process could grab it before the server binds), but the
/// window is tiny and a lost race surfaces as a readiness-timeout panic with the child's stderr.
pub fn free_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    l.local_addr().expect("local_addr").port()
}

/// Absolute path to the freshly-built `kb` binary for this test run.
pub fn kb_bin() -> PathBuf {
    assert_cmd::cargo::cargo_bin("kb")
}

/// A bare `Command` for the `kb` binary, ready for the caller to append args to.
pub fn kb_command() -> Command {
    Command::new(kb_bin())
}

/// A corpus directory (backed by a `TempDir`) seeded with synthetic files.
pub struct Corpus {
    dir: TempDir,
}

impl Corpus {
    /// Create a corpus with the given `(relative_path, contents)` files.
    pub fn with_files(files: &[(&str, &str)]) -> Corpus {
        let dir = tempfile::tempdir().expect("create corpus tempdir");
        let c = Corpus { dir };
        for (rel, body) in files {
            c.add_file(rel, body);
        }
        c
    }

    /// An empty corpus directory.
    pub fn empty() -> Corpus {
        Corpus {
            dir: tempfile::tempdir().expect("create corpus tempdir"),
        }
    }

    /// Write (or overwrite) `rel` with `body`, creating parent dirs as needed.
    pub fn add_file(&self, rel: &str, body: &str) {
        let p = self.dir.path().join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).expect("create corpus subdir");
        }
        std::fs::write(&p, body).expect("write corpus file");
    }

    pub fn path(&self) -> &Path {
        self.dir.path()
    }

    /// Consume the corpus, yielding its backing `TempDir` (e.g. to hand to a `ServerHandle` via
    /// [`ServerBuilder::keep`] / `spawn_kb`'s `keep`, so the corpus outlives the server).
    pub fn into_dir(self) -> TempDir {
        self.dir
    }

    /// A `--root` argument for this corpus: `LABEL=PATH`.
    pub fn root_arg(&self, label: &str) -> String {
        format!("{label}={}", self.dir.path().display())
    }
}

/// A dedicated `.glossa` state directory (see `--state-dir`), kept separate from any corpus root.
pub fn state_dir() -> TempDir {
    tempfile::tempdir().expect("create state tempdir")
}

/// Push `dir`'s mtime strictly into the future so the freshen dir-mtime gate reliably re-scans it,
/// rather than racing the coarse-granularity Windows FS clock (the same reason the in-tree unit
/// tests use `filetime`). Call after adding a file to a served corpus.
pub fn bump_dir_mtime(dir: &Path) {
    let future = std::time::SystemTime::now() + Duration::from_secs(10);
    let ft = filetime::FileTime::from_system_time(future);
    filetime::set_file_mtime(dir, ft).expect("bump dir mtime");
}

// ---------------------------------------------------------------------------------------------
// Minimal raw HTTP/1.1 client (plaintext)
// ---------------------------------------------------------------------------------------------

/// A parsed HTTP response.
#[derive(Debug, Clone)]
pub struct HttpResp {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl HttpResp {
    /// Case-insensitive header lookup.
    pub fn header(&self, name: &str) -> Option<String> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.clone())
    }
}

fn host_port(base: &str) -> String {
    base.strip_prefix("http://")
        .or_else(|| base.strip_prefix("https://"))
        .unwrap_or(base)
        .to_string()
}

fn build_request(
    host: &str,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&[u8]>,
) -> Vec<u8> {
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n");
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    if let Some(b) = body {
        req.push_str(&format!("Content-Length: {}\r\n", b.len()));
    }
    req.push_str("\r\n");
    let mut out = req.into_bytes();
    if let Some(b) = body {
        out.extend_from_slice(b);
    }
    out
}

/// Read to EOF, tolerating a read-timeout / unexpected-EOF as "the body ended" (best-effort).
fn read_to_end_lenient<R: Read>(r: &mut R) -> Vec<u8> {
    let mut raw = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        match r.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => raw.extend_from_slice(&buf[..n]),
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock
                        | std::io::ErrorKind::TimedOut
                        | std::io::ErrorKind::UnexpectedEof
                ) =>
            {
                break
            }
            Err(_) => break,
        }
    }
    raw
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Decode a `Transfer-Encoding: chunked` body.
fn dechunk(mut data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let Some(eol) = find_subslice(data, b"\r\n") else {
            break;
        };
        let size_line = String::from_utf8_lossy(&data[..eol]);
        let size_hex = size_line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_hex, 16).unwrap_or(0);
        data = &data[eol + 2..];
        if size == 0 {
            break;
        }
        if data.len() < size {
            out.extend_from_slice(data);
            break;
        }
        out.extend_from_slice(&data[..size]);
        // Skip the chunk's trailing CRLF.
        data = if data.len() >= size + 2 {
            &data[size + 2..]
        } else {
            &[]
        };
    }
    out
}

fn parse_response(raw: &[u8]) -> HttpResp {
    let split = find_subslice(raw, b"\r\n\r\n").unwrap_or(raw.len());
    let head = String::from_utf8_lossy(&raw[..split]);
    let body_bytes = if split + 4 <= raw.len() {
        &raw[split + 4..]
    } else {
        &[][..]
    };

    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or("");
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(0);

    let mut headers = Vec::new();
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }

    let chunked = headers.iter().any(|(k, v)| {
        k.eq_ignore_ascii_case("transfer-encoding") && v.to_ascii_lowercase().contains("chunked")
    });
    let content_length = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.parse::<usize>().ok());

    let body = if chunked {
        dechunk(body_bytes)
    } else if let Some(len) = content_length {
        body_bytes[..len.min(body_bytes.len())].to_vec()
    } else {
        body_bytes.to_vec()
    };

    HttpResp {
        status,
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    }
}

/// Perform one plaintext HTTP request, returning `Err` on any connection/IO failure (used by the
/// readiness poll, which must distinguish "not up yet" from "up").
pub fn try_request(
    base: &str,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&[u8]>,
) -> Result<HttpResp, String> {
    let hp = host_port(base);
    let mut stream = TcpStream::connect(&hp).map_err(|e| format!("connect {hp}: {e}"))?;
    stream.set_read_timeout(Some(IO_TIMEOUT)).ok();
    stream.set_write_timeout(Some(IO_TIMEOUT)).ok();
    let req = build_request(&hp, method, path, headers, body);
    stream.write_all(&req).map_err(|e| format!("write: {e}"))?;
    stream.flush().ok();
    let raw = read_to_end_lenient(&mut stream);
    if raw.is_empty() {
        return Err("empty response".to_string());
    }
    Ok(parse_response(&raw))
}

/// `GET`, panicking on IO failure (for use once the server is known-ready).
pub fn http_get(base: &str, path: &str, headers: &[(&str, &str)]) -> HttpResp {
    try_request(base, "GET", path, headers, None).expect("http GET")
}

/// `POST`, panicking on IO failure.
pub fn http_post(base: &str, path: &str, headers: &[(&str, &str)], body: &[u8]) -> HttpResp {
    try_request(base, "POST", path, headers, Some(body)).expect("http POST")
}

/// Concatenate the payloads of `data:` lines from an SSE body. If `body` carries no `data:` line
/// (i.e. it is already a plain JSON response), it is returned unchanged.
pub fn sse_data(body: &str) -> String {
    let mut out = String::new();
    let mut saw = false;
    for line in body.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            saw = true;
            out.push_str(rest.trim_start());
        }
    }
    if saw {
        out
    } else {
        body.to_string()
    }
}

// ---------------------------------------------------------------------------------------------
// MCP client (rmcp streamable-http) over the plaintext transport
// ---------------------------------------------------------------------------------------------

/// The two `Accept` values rmcp's streamable-http endpoint requires on every `/mcp` POST.
pub const MCP_ACCEPT: &str = "application/json, text/event-stream";

/// A minimal MCP client: `initialize` (capturing the session id) + `initialized`, then `tools/call`.
pub struct McpClient {
    base: String,
    session: String,
    bearer: Option<String>,
}

impl McpClient {
    /// Handshake with no auth.
    pub fn connect(base: &str) -> McpClient {
        Self::connect_inner(base, None)
    }

    /// Handshake presenting `Authorization: Bearer <token>`.
    pub fn connect_with_auth(base: &str, token: &str) -> McpClient {
        Self::connect_inner(base, Some(token.to_string()))
    }

    fn connect_inner(base: &str, bearer: Option<String>) -> McpClient {
        let auth_hdr = bearer.as_ref().map(|t| format!("Bearer {t}"));
        let mut headers: Vec<(&str, &str)> =
            vec![("Accept", MCP_ACCEPT), ("Content-Type", "application/json")];
        if let Some(a) = auth_hdr.as_deref() {
            headers.push(("Authorization", a));
        }
        let init = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "e2e", "version": "0"}
            }
        });
        let resp = http_post(base, "/mcp", &headers, init.to_string().as_bytes());
        assert!(
            resp.status == 200 || resp.status == 202,
            "initialize failed: status={} body={}",
            resp.status,
            resp.body
        );
        let session = resp.header("mcp-session-id").unwrap_or_else(|| {
            panic!("initialize response carried no Mcp-Session-Id header: {resp:?}")
        });

        // The `initialized` notification, echoing the session header.
        let mut note_headers = headers.clone();
        let sid = session.clone();
        note_headers.push(("Mcp-Session-Id", sid.as_str()));
        let note = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
        let _ = http_post(base, "/mcp", &note_headers, note.to_string().as_bytes());

        McpClient {
            base: base.to_string(),
            session,
            bearer,
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session
    }

    /// Call a tool and return the JSON-RPC `result` object.
    pub fn call_tool(&self, name: &str, args: Value) -> Value {
        let auth_hdr = self.bearer.as_ref().map(|t| format!("Bearer {t}"));
        let mut headers: Vec<(&str, &str)> = vec![
            ("Accept", MCP_ACCEPT),
            ("Content-Type", "application/json"),
            ("Mcp-Session-Id", self.session.as_str()),
        ];
        if let Some(a) = auth_hdr.as_deref() {
            headers.push(("Authorization", a));
        }
        let body = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": name, "arguments": args}
        });
        let resp = http_post(&self.base, "/mcp", &headers, body.to_string().as_bytes());
        assert_eq!(
            resp.status, 200,
            "tools/call {name} status: body={}",
            resp.body
        );
        let payload = sse_data(&resp.body);
        let v: Value = serde_json::from_str(&payload)
            .unwrap_or_else(|e| panic!("tools/call {name} response not JSON ({e}): {payload}"));
        v.get("result")
            .cloned()
            .unwrap_or_else(|| panic!("tools/call {name} carried no result: {v}"))
    }

    /// Convenience: run the `search` tool and return the first text content block.
    pub fn search_text(&self, query: &str) -> String {
        let result = self.call_tool("search", json!({"query": query}));
        result
            .get("content")
            .and_then(|c| c.get(0))
            .and_then(|b| b.get("text"))
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string()
    }
}

// ---------------------------------------------------------------------------------------------
// Server process lifecycle
// ---------------------------------------------------------------------------------------------

/// A running `kb` server child, its base URL, and a background-drained copy of its stderr. Dropping
/// this kills the child.
pub struct ServerHandle {
    child: Child,
    base: String,
    port: u16,
    stderr: Arc<Mutex<String>>,
    _keep: Vec<TempDir>,
}

impl ServerHandle {
    pub fn base(&self) -> &str {
        &self.base
    }
    pub fn port(&self) -> u16 {
        self.port
    }
    pub fn pid(&self) -> u32 {
        self.child.id()
    }
    /// Everything the child has written to stderr so far.
    pub fn stderr(&self) -> String {
        self.stderr.lock().unwrap().clone()
    }
    /// Wait up to `dur` for the child to exit on its own, returning its status if it did.
    pub fn wait_for_exit(&mut self, dur: Duration) -> Option<ExitStatus> {
        let deadline = Instant::now() + dur;
        loop {
            match self.child.try_wait().expect("try_wait") {
                Some(status) => return Some(status),
                None => {
                    if Instant::now() >= deadline {
                        return None;
                    }
                    thread::sleep(Duration::from_millis(50));
                }
            }
        }
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn drain_stderr(child: &mut Child) -> Arc<Mutex<String>> {
    let buf = Arc::new(Mutex::new(String::new()));
    if let Some(err) = child.stderr.take() {
        let b = buf.clone();
        thread::spawn(move || {
            let mut rd = BufReader::new(err);
            let mut line = String::new();
            loop {
                line.clear();
                match rd.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => b.lock().unwrap().push_str(&line),
                    Err(_) => break,
                }
            }
        });
    }
    buf
}

/// Spawn `cmd` (stdout discarded, stderr drained), then poll `ready` every 100ms until it returns
/// `true` or [`READY_TIMEOUT`] elapses. Panics — including the captured stderr — if the child exits
/// early or never becomes ready. `keep` holds any `TempDir`s that must outlive the server.
pub fn spawn_kb<F: Fn() -> bool>(
    mut cmd: Command,
    base: String,
    port: u16,
    keep: Vec<TempDir>,
    ready: F,
) -> ServerHandle {
    cmd.stdout(Stdio::null()).stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn kb");
    let stderr = drain_stderr(&mut child);

    let deadline = Instant::now() + READY_TIMEOUT;
    loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            panic!(
                "kb exited early ({status}) before becoming ready.\n--- stderr ---\n{}",
                stderr.lock().unwrap()
            );
        }
        if ready() {
            break;
        }
        if Instant::now() >= deadline {
            let s = stderr.lock().unwrap().clone();
            let _ = child.kill();
            panic!("kb did not become ready within {READY_TIMEOUT:?}.\n--- stderr ---\n{s}");
        }
        thread::sleep(Duration::from_millis(100));
    }

    ServerHandle {
        child,
        base,
        port,
        stderr,
        _keep: keep,
    }
}

/// Fluent builder for a plaintext streamable-http `kb` server.
#[derive(Default)]
pub struct ServerBuilder {
    roots: Vec<String>,
    state_dir: Option<PathBuf>,
    auth_token: Option<String>,
    envs: Vec<(String, String)>,
    extra_args: Vec<String>,
    keep: Vec<TempDir>,
}

impl ServerBuilder {
    pub fn new() -> ServerBuilder {
        ServerBuilder::default()
    }

    /// Add one `--root LABEL=PATH` (or bare `PATH`).
    pub fn root(mut self, spec: impl Into<String>) -> Self {
        self.roots.push(spec.into());
        self
    }

    pub fn state_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.state_dir = Some(dir.into());
        self
    }

    pub fn auth_token(mut self, token: impl Into<String>) -> Self {
        self.auth_token = Some(token.into());
        self
    }

    pub fn env(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.envs.push((k.into(), v.into()));
        self
    }

    pub fn arg(mut self, a: impl Into<String>) -> Self {
        self.extra_args.push(a.into());
        self
    }

    /// Hand a `TempDir` to the server handle so it outlives the process.
    pub fn keep(mut self, dir: TempDir) -> Self {
        self.keep.push(dir);
        self
    }

    /// Build the full `kb mcp --transport streamable-http --bind 127.0.0.1:<port> ...` command,
    /// WITHOUT the transport-specific readiness wiring (used both by [`start`](Self::start) and by
    /// the TLS e2e which adds `--tls-*` flags and polls over HTTPS).
    pub fn command(&self, port: u16) -> Command {
        let mut cmd = kb_command();
        cmd.arg("mcp")
            .arg("--transport")
            .arg("streamable-http")
            .arg("--bind")
            .arg(format!("127.0.0.1:{port}"));
        for r in &self.roots {
            cmd.arg("--root").arg(r);
        }
        if let Some(sd) = &self.state_dir {
            cmd.arg("--state-dir").arg(sd);
        }
        if let Some(t) = &self.auth_token {
            cmd.arg("--auth-token").arg(t);
        }
        for a in &self.extra_args {
            cmd.arg(a);
        }
        for (k, v) in &self.envs {
            cmd.env(k, v);
        }
        cmd
    }

    /// Allocate a port, spawn the server, and gate on `GET /ready` == 200.
    pub fn start(self) -> ServerHandle {
        let port = free_port();
        let cmd = self.command(port);
        let base = format!("http://127.0.0.1:{port}");
        let probe = base.clone();
        spawn_kb(
            cmd,
            base,
            port,
            self.keep,
            move || matches!(try_request(&probe, "GET", "/ready", &[], None), Ok(r) if r.status == 200),
        )
    }
}

// ---------------------------------------------------------------------------------------------
// TLS client + cert factory (behind the `tls` feature)
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "tls")]
pub mod tls {
    use super::*;
    use rcgen::{
        BasicConstraints, CertificateParams, DistinguishedName, DnType, IsCa, KeyPair, SanType,
    };
    use rustls::pki_types::ServerName;
    use rustls::{ClientConfig, ClientConnection, RootCertStore, StreamOwned};

    /// A self-signed CA (PEM + the rcgen material needed to sign leaves).
    pub struct TestCa {
        pub cert_pem: String,
        cert: rcgen::Certificate,
        key: KeyPair,
    }

    /// Generate a self-signed CA — mirrors `src/tls.rs`'s `mod tests` recipe.
    pub fn make_ca() -> TestCa {
        let key = KeyPair::generate().unwrap();
        let mut params = CertificateParams::new(Vec::<String>::new()).unwrap();
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let mut dn = DistinguishedName::new();
        dn.push(DnType::CommonName, "glossa-e2e-ca");
        params.distinguished_name = dn;
        let cert = params.self_signed(&key).unwrap();
        let cert_pem = cert.pem();
        TestCa {
            cert_pem,
            cert,
            key,
        }
    }

    /// Generate a leaf cert for `127.0.0.1`/`localhost` signed by `ca`, returning `(cert_pem,
    /// key_pem)`. Usable as either a server identity or (for mTLS) a client identity.
    pub fn make_leaf(ca: &TestCa, common_name: &str) -> (String, String) {
        let key = KeyPair::generate().unwrap();
        let mut params =
            CertificateParams::new(vec!["127.0.0.1".to_string(), "localhost".to_string()]).unwrap();
        let mut dn = DistinguishedName::new();
        dn.push(DnType::CommonName, common_name);
        params.distinguished_name = dn;
        params
            .subject_alt_names
            .push(SanType::IpAddress("127.0.0.1".parse().unwrap()));
        let cert = params.signed_by(&key, &ca.cert, &ca.key).unwrap();
        (cert.pem(), key.serialize_pem())
    }

    /// Install a process-default rustls crypto provider once (idempotent). rustls 0.23's
    /// `ClientConfig::builder()` needs one; installing twice is a harmless `Err` we ignore.
    fn ensure_provider() {
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        });
    }

    fn certs_from_pem(pem: &str) -> Vec<rustls::pki_types::CertificateDer<'static>> {
        rustls_pemfile::certs(&mut BufReader::new(pem.as_bytes()))
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    }

    /// A blocking HTTPS GET over rustls. Trusts `ca_pem`; optionally presents a client identity
    /// (for mTLS). Returns `Err` on any connect/handshake/IO failure — the mTLS "no client cert is
    /// rejected" test relies on this.
    pub fn https_get(
        host: &str,
        port: u16,
        ca_pem: &str,
        client_identity: Option<(&str, &str)>,
        path: &str,
    ) -> Result<HttpResp, String> {
        ensure_provider();
        let mut roots = RootCertStore::empty();
        for c in certs_from_pem(ca_pem) {
            roots.add(c).map_err(|e| format!("add CA: {e}"))?;
        }
        let builder = ClientConfig::builder().with_root_certificates(roots);
        let config = match client_identity {
            Some((cert_pem, key_pem)) => {
                let certs = certs_from_pem(cert_pem);
                let key = rustls_pemfile::private_key(&mut BufReader::new(key_pem.as_bytes()))
                    .map_err(|e| format!("parse client key: {e}"))?
                    .ok_or_else(|| "no client key".to_string())?;
                builder
                    .with_client_auth_cert(certs, key)
                    .map_err(|e| format!("client auth cert: {e}"))?
            }
            None => builder.with_no_client_auth(),
        };
        let server_name =
            ServerName::try_from(host.to_string()).map_err(|e| format!("server name: {e}"))?;
        let conn = ClientConnection::new(Arc::new(config), server_name)
            .map_err(|e| format!("client connection: {e}"))?;
        let sock = TcpStream::connect((host, port)).map_err(|e| format!("connect: {e}"))?;
        sock.set_read_timeout(Some(IO_TIMEOUT)).ok();
        sock.set_write_timeout(Some(IO_TIMEOUT)).ok();
        let mut tls = StreamOwned::new(conn, sock);
        let req = build_request(&format!("{host}:{port}"), "GET", path, &[], None);
        tls.write_all(&req).map_err(|e| format!("tls write: {e}"))?;
        tls.flush().ok();
        let raw = read_to_end_lenient(&mut tls);
        if raw.is_empty() {
            return Err("empty TLS response".to_string());
        }
        Ok(parse_response(&raw))
    }
}
