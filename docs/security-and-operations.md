# Security & Operations

How to run the glossa MCP server securely and observe it in production. This covers the
**network transport** (`kb mcp --transport streamable-http`); the local **stdio** transport is a
subprocess owned by its parent and needs none of this.

This is the security/observability-focused view. For deployment mechanics — topology, the
`--config` file, native TLS build/run instructions, systemd unit and Windows service setup — see
**[deploy/mcp-server.md](deploy/mcp-server.md)**, which this doc links to rather than duplicates.

## Threat model in one line

The streamable-http server binds `127.0.0.1` by default. For any network exposure it expects
either an auth token, TLS (native or via a reverse-proxy gateway), or both — and a startup safety
interlock (below) refuses to come up at all if none of those are in place, so a forgotten
token/proxy fails closed instead of silently serving the open internet.

## Safety interlock

On every `--transport streamable-http` start, the server refuses to come up when **all** of these
hold: the bind address is **non-loopback**, **no** `--auth-token`/`GLOSSA_MCP_TOKEN` is set, and
**no** TLS is active. Fix it by setting a token, enabling TLS, or explicitly overriding with
`--insecure=true` (also `GLOSSA_MCP_INSECURE=true`) — logged as a loud warning plus a
`glossa::audit` `insecure_serve` event. `--insecure` takes a value; `--insecure` alone (no `=true`)
is a parse error, not "enabled." Full detail: [Safety interlock](deploy/mcp-server.md#safety-interlock).

## Authentication

A shared **bearer token** guards the `/mcp` endpoint — an interim integration key ahead of full
identity (OIDC/IdP) integration.

```
kb mcp <corpus> --transport streamable-http --bind 0.0.0.0:8080 --auth-token <TOKEN>
# or: GLOSSA_MCP_TOKEN=<TOKEN> kb mcp <corpus> --transport streamable-http …
```

- Every `/mcp` request must send `Authorization: Bearer <TOKEN>`; anything else gets **401**.
- `/health`, `/ready`, `/metrics` are **never** guarded, so liveness/readiness probes and metric
  scrapes keep working without the token.
- The token is compared in **constant time**. It is never echoed in `--help` or error output.
- **Env/flag only, never the config file.** The token is deliberately not a usable
  `--config`-file setting: a `token`/`auth_token` key under `[server]` makes the whole file fail to
  load (fail closed, not silently ignored). Set it via `--auth-token`, `GLOSSA_MCP_TOKEN`, or a
  systemd `EnvironmentFile` at mode `0600` — see
  [Deployment config file](deploy/mcp-server.md#deployment-config-file---config).
- Unset → the endpoint is unauthenticated (only safe on loopback, or behind a gateway/mTLS that
  authenticates). Ignored for `--transport stdio`.

**Optional mTLS** (`--features tls` build): pass `--tls-client-ca <ca.pem>` to require and verify a
client certificate on every connection, in addition to or instead of the bearer token. See
[Native TLS](deploy/mcp-server.md#native-tls---features-tls-build) for setup and caveats — note in
particular that **TLS alone is not authentication**: a non-loopback TLS listener with no token and
no `--tls-client-ca` passes the safety interlock but leaves every tool callable by anyone who can
reach the port.

## Network hardening

- **Bind address:** `--bind` (env `GLOSSA_MCP_BIND`, default `127.0.0.1:8080`).
- **DNS-rebind guard:** the `Host` header is checked; loopback is allowed by default. For a gateway
  or public host, pass each expected host with `--allowed-host` (repeatable).
- **TLS — two supported paths:**
  - **Reverse proxy (default build):** the default build has no TLS crypto surface (smaller
    binary, no extra CVE exposure); terminate TLS/OAuth2/OIDC/mTLS/rate-limiting at a reverse proxy
    or API gateway in front, and the server speaks plain HTTP on its bind address.
  - **Native TLS (`--features tls` build):** `kb mcp` can terminate HTTPS (and optional mTLS)
    itself via `--tls-cert`/`--tls-key`/`--tls-client-ca`. Cert/key are hot-reloaded on SIGHUP or
    mtime change with no dropped connections. A reverse proxy is still recommended for
    high-exposure, no-proxy deployments — most notably because the per-IP **rate-limit guard
    degrades to a single shared global bucket** over native TLS (no `ConnectInfo` to key on)
    unless a trusted proxy sets a forwarded-for header. The connection cap
    (`GLOSSA_MCP_MAX_CONNECTIONS`) *does* apply over native TLS, but with TLS-specific semantics:
    it bounds **established (post-handshake) sessions** and **sheds** a connection that arrives
    when the cap is full (closes it) rather than queueing it — unlike the plaintext listener,
    which caps at accept and lets the kernel backlog absorb the excess. A separate, always-on
    pre-auth bound (`GLOSSA_MCP_MAX_HANDSHAKES`, default `256`, `--features tls` only) caps
    concurrent in-flight TLS handshakes so a slow-handshake flood cannot exhaust file descriptors;
    worst-case open descriptors are bounded by `MAX_HANDSHAKES + MAX_CONNECTIONS`. Full caveats:
    [Native TLS](deploy/mcp-server.md#native-tls---features-tls-build).

## Serving hardening

Applies to the `/mcp` endpoint regardless of TLS mode:

- **Request timeout:** `GLOSSA_MCP_REQUEST_TIMEOUT_SECS`, default `120`.
- **Body size limit:** `GLOSSA_MCP_MAX_BODY_BYTES`, default `4000000` (4 MB).
- **Overload guards (opt-in, off by default):**
  - `GLOSSA_MCP_MAX_CONCURRENCY=<n>` — caps in-flight `/mcp` requests; the `(n+1)`-th is shed with
    `503`.
  - `GLOSSA_MCP_RATE_LIMIT_PER_SEC=<n>` — per-IP token-bucket rate limit.
  - `GLOSSA_MCP_MAX_CONNECTIONS=<n>` — caps concurrently open connections. On the plaintext
    listener this caps at accept; over native TLS it caps established (post-handshake) sessions
    and sheds on full (see the Native TLS note above).

  These exist for a native-TLS-without-proxy deployment or any case where the process itself
  should shed load; a proxy/load-balancer deployment normally delegates this to the proxy. Combining
  the connection cap with the rate limiter (or running native TLS with the rate limiter) degrades
  per-IP keying to a single global bucket for any request without a trusted
  `X-Forwarded-For`/`X-Real-Ip`/`Forwarded` header — it never fails closed, but per-IP fairness is
  lost until a trusted proxy supplies that header. Details:
  [Overload guards](deploy/mcp-server.md#overload-guards-opt-in).

## Session idle timeout (opt-in)

A streamable-http session that makes no request for longer than a threshold is refused on its next
request, so the client re-initializes.

```
kb mcp … --session-idle-secs 900          # 15 min; env GLOSSA_MCP_SESSION_IDLE_SECS
```

- **Opt-in:** `0` (default) disables it.
- On expiry the next request gets **404** — the streamable-http signal a spec-compliant client
  answers by re-running the `initialize` handshake (cheap; the KB holds no per-session state, so no
  work is lost). An in-flight request is rejected *before* execution, so it is safely replayable.
- A background reaper prunes abandoned sessions from the activity map.
- Enable it per deployment where an idle-session policy is required; leave it off elsewhere so
  clients that do not expect session expiry are never surprised.

## Reliability & lifecycle

- **Panic barrier.** Every tool call is dispatched through a panic barrier: a handler panic is
  caught, the client gets a generic internal-error response (the panic payload is never surfaced),
  and the `glossa_tool_panics_total` counter increments — the process stays up and keeps serving
  other requests.
- **systemd watchdog (`Type=notify`).** `WatchdogSec=` arms a keepalive ping at half that interval,
  gated on the async runtime actually being scheduled (liveness), not on `/ready` — a long blocking
  index/generalize pass does not by itself starve the ping. `READY=1` is sent right after the
  listener binds, before the initial index warm-up runs. `STOPPING=1` is sent on graceful shutdown
  (SIGTERM/Ctrl-C). Requires `NotifyAccess=main` in the unit. Details:
  [Panic barrier + systemd watchdog](deploy/mcp-server.md#panic-barrier--systemd-watchdog).
- **`glossa_index_warm`.** Prometheus gauge (0/1) for "the first freshen pass has completed" — a
  different signal from `/ready` (which only checks the local index/graph files are openable).
  See [Two initial-index flows](deploy/mcp-server.md#two-initial-index-flows).
- **Log-level reload without a restart.** The tracing filter reloads from a one-line control file
  at `<state-dir>/.glossa/loglevel` (SIGHUP applies it immediately on unix; a ~5s mtime poll covers
  every platform including Windows). `systemctl reload` (SIGHUP) also forces an immediate freshen
  and, in a `--features tls` build, re-reads the TLS cert/key/client-CA files — it does **not**
  reload `--bind`/`--transport`/`--state-dir`/`--auth-token`/`--allowed-host`. Details:
  [Log-level reload](deploy/mcp-server.md#log-level-reload-without-a-restart).
- **Ontology auto-reload.** `<state-dir>/.glossa/ontology.toml` is cached in memory and re-parsed
  automatically whenever its mtime advances — no signal or restart needed. The mtime and the parsed
  ontology are published together as one atomic snapshot, so a concurrent read never observes a
  newer mtime paired with a stale parse (or vice versa).

## Deployment config file

`--config <path>` (also `GLOSSA_CONFIG`) points at a TOML file holding the base settings for one
role, instead of spelling out every flag/env var in the launching unit/script. Precedence, per
individual setting: **CLI flag > env var > config file > built-in default**. The file format
rejects unknown sections/keys at load time (fail closed on a typo). **Secrets stay env/flag-only:**
a `token`/`auth_token` key under `[server]` makes the file fail to load rather than being silently
ignored — see [Authentication](#authentication) above. No wholesale hot reload: a structural change
(roots, `state_dir`, `bind`, `transport`, TLS paths) needs a process restart.

Full operator guide and precedence table:
[Deployment config file (`--config`)](deploy/mcp-server.md#deployment-config-file---config).
Annotated example covering every section: [glossa.toml](deploy/glossa.toml).

## systemd unit

[`deploy/glossa-mcp.service`](deploy/glossa-mcp.service) is the canonical hardened production unit:
`Type=notify` with the watchdog wired up, `ExecReload` (SIGHUP) for log-level/TLS/freshen reload,
`RequiresMountsFor` for a network corpus root, the bearer token isolated in a mode-`0600`
`EnvironmentFile`, and filesystem sandboxing (`ProtectSystem=strict`, `NoNewPrivileges`, …). Copy it
per (base, profile, port) rather than writing a unit from scratch.

## Observability

### Health & readiness

- `GET /health` → `200 ok` (liveness: the process is up).
- `GET /ready` → `200 ready` / `503 not ready` (the index + graph are openable).

### Metrics (Prometheus)

`GET /metrics` returns Prometheus text-exposition. Scrape it directly, or bridge to OpenTelemetry
with an OTel Collector `prometheus` receiver (no native OTLP exporter is built in, by design).

| Metric | Type | Meaning |
| --- | --- | --- |
| `glossa_up` | gauge | 1 if serving |
| `glossa_index_chunks` | gauge | indexed chunks |
| `glossa_graph_nodes` / `glossa_graph_edges` | gauge | knowledge-graph size |
| `glossa_graph_dirty` | gauge | derived layer stale (1) / fresh (0) |
| `glossa_indexing` | gauge | a freshen is in progress |
| `glossa_index_warm` | gauge | the initial index freshen has completed (see above) |
| `glossa_tool_panics_total` | counter | tool-handler panics caught by the panic barrier |
| `glossa_http_requests_total` | counter | HTTP requests received |
| `glossa_http_responses_total{class}` | counter | responses by status class (`2xx`…`5xx`) |
| `glossa_http_requests_in_flight` | gauge | requests currently served |
| `glossa_http_request_duration_seconds` | histogram | request latency |
| `glossa_mcp_auth_rejected_total` | counter | `/mcp` requests rejected (missing/invalid token) |

`/health`, `/ready`, `/metrics` are exempt from `--auth-token`.

### Logs

- Structured logs go to **stderr** (stdout is the stdio JSON-RPC channel and never carries logs).
- Level via `RUST_LOG` (default `info`); reloadable at runtime — see
  [Log-level reload](#reliability--lifecycle) above.
- `GLOSSA_LOG_FORMAT=json` emits **one JSON object per line** for a SIEM / log pipeline; the default
  is human-readable.

### Security audit events

Dedicated events are emitted on the `glossa::audit` tracing target (filter with
`RUST_LOG=glossa::audit=info`, or a SIEM rule on `"target":"glossa::audit"`). Under
`GLOSSA_LOG_FORMAT=json` each is one JSON object with a stable schema:

| Field | Meaning |
| --- | --- |
| `category` | `auth`, `access`, `session` |
| `action` | e.g. `bearer_reject`, `insecure_serve`, `tool_invoke`, `idle_expired` |
| `outcome` | `denied`, `invoked`, `override`, … |
| `source` | client IP for network events, else `-` |
| `object` | the route or tool acted on |

Recorded today: bearer-token rejections, the safety-interlock `--insecure` override, idle-session
expiries, and every write/admin tool invocation (`graph_upsert`, `graph_delete`, `graph_build`,
`note`, `del`). The acting **subject** (a per-user principal) stays coarse until identity
integration lands — see the scorecard.

## Indexing hygiene

- On a corpus with no ignore file, `kb index` seeds a default `.ignore` that **whitelists** the file
  types glossa can extract (documents, images, common text/code). Installers, archives and temp
  files are not read. Edit the file to tune it; an existing `.ignore`/`.gitignore` is never touched.
- A corrupt/unreadable file is logged, skipped, and listed in the end-of-run error summary — it
  never aborts the index.

## Enterprise-readiness scorecard

An honest snapshot. glossa is **production-hardened for a controlled on-prem deployment**, either
behind a gateway or self-terminating TLS; full enterprise identity/authorization is on the roadmap.

| Capability | Status | Notes |
| --- | --- | --- |
| Startup safety interlock | ✅ | Refuses non-loopback + no-auth + no-TLS unless `--insecure=true` |
| Network auth (shared token) | ✅ | Bearer token on `/mcp`; interim integration key |
| mTLS (client certs) | ✅ | Opt-in via `--tls-client-ca`, `--features tls` build |
| Identity / SSO (OIDC/IdP) | 🔲 Roadmap | No per-user principals yet |
| Authorization / RBAC | 🔲 Roadmap | No per-user/per-tool access control yet |
| Per-user data isolation | 🔲 Roadmap | Single shared corpus |
| TLS | ✅ Native (opt-in) or ⚠️ External | Reverse-proxy by default; native TLS behind `--features tls` — see its caveats above |
| Request timeout / body limit | ✅ | Defaults 120s / 4 MB |
| Overload guards (concurrency/rate/connection) | ✅ Opt-in | Off by default; see caveats above |
| Panic barrier | ✅ | A handler panic never takes the process down |
| Audit logging | ⚠️ Partial | Auth + write-tool events; subject coarse until identity |
| Metrics (Prometheus) | ✅ | Request + index/graph metrics; OTel via Collector |
| Structured logs (SIEM) | ✅ | JSON logs to stderr; runtime log-level reload |
| Health / readiness probes | ✅ | `/health`, `/ready` |
| Session idle timeout | ✅ | Opt-in |
| Graceful shutdown / service host | ✅ | Signal + systemd watchdog + Windows service |

**Bottom line:** ready to deploy securely on-prem — behind a TLS/auth gateway, or self-terminating
TLS with a token/mTLS — with monitoring and audit. Corporate identity, RBAC and per-user isolation
are the remaining gap and are on the roadmap.
