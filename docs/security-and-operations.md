# Security & Operations

How to run the glossa MCP server securely and observe it in production. This covers the
**network transport** (`kb mcp --transport streamable-http`); the local **stdio** transport is a
subprocess owned by its parent and needs none of this.

## Threat model in one line

The streamable-http server binds `127.0.0.1` by default and expects a TLS/auth gateway in front for
any network exposure. Everything below either hardens that edge or makes the server observable.

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
- The token is compared in constant time. It is never echoed in `--help` or error output.
- Unset → the endpoint is unauthenticated (only safe on loopback or behind a gateway that
  authenticates). Ignored for `--transport stdio`.

## Network hardening

- **Bind address:** `--bind` (env `GLOSSA_MCP_BIND`, default `127.0.0.1:8080`).
- **DNS-rebind guard:** the `Host` header is checked; loopback is allowed by default. For a gateway
  or public host, pass each expected host with `--allowed-host` (repeatable).
- **TLS:** terminate TLS at a reverse proxy / gateway in front of the server (the server speaks
  plain HTTP on its bind address).

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
| `glossa_http_requests_total` | counter | HTTP requests received |
| `glossa_http_responses_total{class}` | counter | responses by status class (`2xx`…`5xx`) |
| `glossa_http_requests_in_flight` | gauge | requests currently served |
| `glossa_http_request_duration_seconds` | histogram | request latency |
| `glossa_mcp_auth_rejected_total` | counter | `/mcp` requests rejected (missing/invalid token) |

### Logs

- Structured logs go to **stderr** (stdout is the stdio JSON-RPC channel and never carries logs).
- Level via `RUST_LOG` (default `info`).
- `GLOSSA_LOG_FORMAT=json` emits **one JSON object per line** for a SIEM / log pipeline; the default
  is human-readable.

### Security audit events

Dedicated events are emitted on the `glossa::audit` tracing target (filter with
`RUST_LOG=glossa::audit=info`, or a SIEM rule on `"target":"glossa::audit"`). Under
`GLOSSA_LOG_FORMAT=json` each is one JSON object with a stable schema:

| Field | Meaning |
| --- | --- |
| `category` | `auth`, `access`, `session` |
| `action` | e.g. `bearer_reject`, `tool_invoke`, `idle_expired` |
| `outcome` | `denied`, `invoked`, … |
| `source` | client IP for network events, else `-` |
| `object` | the route or tool acted on |

Recorded today: bearer-token rejections, idle-session expiries, and every write/admin tool
invocation (`graph_upsert`, `graph_delete`, `graph_build`, `note`, `del`). The acting **subject**
(a per-user principal) stays coarse until identity integration lands — see the scorecard.

## Indexing hygiene

- On a corpus with no ignore file, `kb index` seeds a default `.ignore` that **whitelists** the file
  types glossa can extract (documents, images, common text/code). Installers, archives and temp
  files are not read. Edit the file to tune it; an existing `.ignore`/`.gitignore` is never touched.
- A corrupt/unreadable file is logged, skipped, and listed in the end-of-run error summary — it
  never aborts the index.

## Enterprise-readiness scorecard

An honest snapshot. glossa is **production-hardened for a controlled on-prem deployment behind a
gateway**; full enterprise identity/authorization is on the roadmap.

| Capability | Status | Notes |
| --- | --- | --- |
| Network auth (shared token) | ✅ | Bearer token on `/mcp`; interim integration key |
| Identity / SSO (OIDC/IdP) | 🔲 Roadmap | No per-user principals yet |
| Authorization / RBAC | 🔲 Roadmap | No per-user/per-tool access control yet |
| Per-user data isolation | 🔲 Roadmap | Single shared corpus |
| TLS | ⚠️ External | Terminate at a gateway (no native TLS) |
| Audit logging | ⚠️ Partial | Auth + write-tool events; subject coarse until identity |
| Metrics (Prometheus) | ✅ | Request + index/graph metrics; OTel via Collector |
| Structured logs (SIEM) | ✅ | JSON logs to stderr |
| Health / readiness probes | ✅ | `/health`, `/ready` |
| Session idle timeout | ✅ | Opt-in |
| Graceful shutdown / service host | ✅ | Signal + Windows service |

**Bottom line:** ready to deploy securely on-prem behind a TLS/auth gateway with monitoring and
audit. Corporate identity, RBAC and per-user isolation are the remaining gap and are on the roadmap.
