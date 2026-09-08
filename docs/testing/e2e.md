# End-to-end test harness

The e2e suite spawns the **real `kb` binary** and drives it over **real sockets** — raw HTTP/1.1
(and, under the `tls` feature, real TLS) against a live `kb mcp --transport streamable-http` server.
It exercises the prod-hardening features (Spec A multi-root + state-dir, Spec B freshness, Spec C
serving/auth/limits/TLS/shutdown, Spec E config file) the way an operator or agent actually hits
them, not through in-process shims.

## Layout

| File | Covers |
|------|--------|
| `tests/e2e/harness.rs` | Shared harness (not an auto-run test — lives in a subdir, pulled in per file) |
| `tests/e2e_spec_a_multiroot.rs` | Two labeled roots + separated `--state-dir`; index placement + multi-root retrieval |
| `tests/e2e_spec_b_freshness.rs` | On-read freshen picks up a new file; retry-tuning knobs accepted |
| `tests/e2e_spec_c_serving.rs` | `/health` `/ready` `/metrics`, bearer auth, body-limit, startup interlock, graceful shutdown, rate-limit |
| `tests/e2e_spec_c_tls.rs` | Native TLS + mTLS with a blocking rustls client |
| `tests/e2e_spec_e_config.rs` | TOML `--config` drives the server; flag-over-file precedence; secret-in-file rejection |

Each test file starts with `#![cfg(feature = "e2e")]` (the TLS file with
`#![cfg(all(feature = "e2e", feature = "tls"))]`) and includes the harness via:

```rust
#[path = "e2e/harness.rs"]
mod harness;
use harness::*;
```

## Feature gating

The suite is behind the **`e2e`** cargo feature (TLS additionally behind **`tls`**), so a normal
`cargo test` compiles these files to nothing and is completely unaffected — no new dependencies, no
sockets, no spawned binary. The harness uses only `std`, `serde_json` (a normal dependency),
`tempfile`/`assert_cmd`/`predicates`/`filetime` (dev-dependencies), and — under `tls` —
`rustls`/`rustls-pemfile`/`rcgen` (already present for the `tls` feature).

## Running

Build/run **foreground** with the target dir on a fast local disk:

```sh
export CARGO_TARGET_DIR=C:/glossa-target        # any local path

# All plaintext e2e:
cargo test -p glossa --features e2e

# One plaintext file:
cargo test -p glossa --features e2e --test e2e_spec_c_serving

# The TLS/mTLS file (needs both features):
cargo test -p glossa --features e2e,tls --test e2e_spec_c_tls
```

Every server binds a fresh loopback port (`127.0.0.1:0`-allocated) and is killed on `Drop`, so the
tests are self-contained and parallel-safe. `cargo test` builds the `kb` binary the tests locate via
`assert_cmd::cargo::cargo_bin("kb")`, so the binary always matches the features you passed.

## Writing a new e2e test

```rust
#![cfg(feature = "e2e")]

#[path = "e2e/harness.rs"]
mod harness;
use harness::*;

#[test]
fn my_case() {
    // A synthetic corpus + a separate state dir.
    let corpus = Corpus::with_files(&[("a.md", "# Doc\n\nfindme content.\n")]);
    let state = state_dir();

    // Fluent server builder -> ServerHandle. start() allocates a port, spawns the real `kb`,
    // and blocks until `GET /ready` == 200 (panics with captured stderr on timeout / early exit).
    let server = ServerBuilder::new()
        .root(corpus.root_arg("docs"))   // "docs=<path>"
        .state_dir(state.path())
        .start();

    // Raw probe endpoints:
    assert_eq!(http_get(server.base(), "/health", &[]).status, 200);

    // MCP over streamable-http: initialize (captures Mcp-Session-Id) + initialized, then tools/call.
    let mcp = McpClient::connect(server.base());
    assert!(mcp.search_text("findme").contains("findme"));

    // The server is killed when `server` drops at end of scope.
}
```

Useful primitives (all in `harness`):

- `free_port() -> u16`, `kb_bin() -> PathBuf`, `kb_command() -> Command`.
- `Corpus::with_files(&[(rel, body)])` / `Corpus::empty()` / `.add_file(rel, body)` / `.path()` /
  `.root_arg(label)` / `.into_dir()`; `state_dir() -> TempDir`; `bump_dir_mtime(&Path)`.
- HTTP: `http_get`, `http_post`, `try_request` (returns `Result`, used by the readiness poll),
  `sse_data`, and the `HttpResp { status, headers, body }` struct with `.header(name)`.
- `McpClient::connect(base)` / `connect_with_auth(base, token)`; `.call_tool(name, json)` returns the
  JSON-RPC `result`; `.search_text(query)` returns the first text content block; `.session_id()`.
- `ServerBuilder`: `.root(..)`, `.state_dir(..)`, `.auth_token(..)`, `.env(k, v)`, `.arg(..)`,
  `.keep(TempDir)`, `.command(port)` (build the `Command` yourself), `.start()`.
- `ServerHandle`: `.base()`, `.port()`, `.pid()`, `.stderr()`, `.wait_for_exit(dur)`; kills on `Drop`.
- Generic spawn: `spawn_kb(cmd, base, port, keep, ready_fn)` — used by the `--config` and TLS files
  that build their own `Command` and poll readiness their own way.
- TLS (feature `tls`, in `harness::tls`): `make_ca()`, `make_leaf(&ca, cn)`,
  `https_get(host, port, ca_pem, client_identity, path)` — a blocking rustls client.

## Knobs, readiness, shutdown

- **Corpus / state**: `--root LABEL=PATH` (repeatable) + `--state-dir`; index/graph state lands only
  under the state dir, never in a corpus root.
- **Auth**: `.auth_token("…")` sets `--auth-token`; `/mcp` then requires `Authorization: Bearer …`
  (probe endpoints stay open).
- **Env limits**: `.env("GLOSSA_MCP_MAX_BODY_BYTES", "1024")`,
  `.env("GLOSSA_MCP_RATE_LIMIT_PER_SEC", "1")`, `.env("GLOSSA_MIN_RESCAN_MS", "50")`,
  `.env("GLOSSA_READ_RETRIES", "5")`, etc.
- **Config file**: build a `glossa.toml` and pass `--config` via `.arg("--config").arg(path)` (or a
  hand-built `Command` + `spawn_kb`). CLI flags override the file per setting.
- **TLS**: pass `--tls-cert`/`--tls-key` (and `--tls-client-ca` for mTLS) on a `command(port)` built
  from `ServerBuilder`, then `spawn_kb` with an `https_get` readiness closure.
- **Readiness**: the primary gate is polling `GET /ready` until `200` (plaintext) or an `https_get`
  `/ready` for TLS; `spawn_kb`/`start` also fail fast (panic with stderr) if the child exits early.
- **Shutdown**: `ServerHandle` kills the child on `Drop`. The graceful-shutdown test is `#[cfg(unix)]`
  (sends `SIGTERM` via `libc::kill` and asserts exit 0); on Windows the harness just kills the child.

## Spec B honesty note

The fault-injection seam (`read_fault`) that Spec B's *unit* tests use to force transient
corpus-read failures is `#[cfg(test)]`, in-process only, and is **unreachable from an
externally-spawned `kb` binary**. The e2e file therefore covers only externally-observable behavior:
an on-read freshen picking up a newly-added file, and the retry-tuning knobs being accepted while the
server keeps serving. It does not (and cannot) assert fault-injected retry/serve-stale behavior over
the socket. One observed nuance is documented in the test itself: the freshen indexes a new file
server-side (the `/metrics` chunk gauge climbs) and the *first* search after the file appears
observes it, but a search reader already built by an earlier query keeps serving its snapshot — so
the test drives the "before" state via `/metrics` rather than a search.
