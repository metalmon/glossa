# Deploying the glossa MCP server

Back to [MCP guide](../mcp.md). Service install from release: [service.md](service.md).

The `kb mcp` server speaks MCP (JSON-RPC 2.0, protocol `2025-06-18`) over two transports:

- **stdio** — local subprocess (IDE / desktop clients co-located with the binary).
- **streamable-http** — network endpoint at `<bind>/mcp` (the prod transport).

**Default build:** HTTP-plaintext, meant to sit behind a **reverse proxy / API gateway** that
terminates TLS, OAuth2/OIDC/mTLS and rate-limiting. **`--features tls` build:** the binary can
terminate TLS (and optional mTLS) itself — see [Native TLS](#native-tls---features-tls-build)
below; a reverse proxy is still recommended for high-exposure, no-proxy deployments (see that
section's caveats). Either way, a **startup safety interlock** (below) refuses to serve a
non-loopback bind with no authentication at all, so a forgotten token/proxy fails closed instead of
silently serving the open internet.

For a local streamable-http quickstart, [`scripts/start-mcp-http.sh`](../../scripts/start-mcp-http.sh) /
[`scripts/start-mcp-http.ps1`](../../scripts/start-mcp-http.ps1) start the server against a corpus and print
ready-to-paste Cursor `mcpServers` JSON.

## Topology: one process per (base × profile)

The corpus root (a positional arg) and the `--profile` decide what a process is. Run several
processes for one base, each with its own port:

```
:: indexer/editor — the ONLY writer; runs the generalize loop; exposes write + admin tools
kb.exe mcp C:\kb\base1 --profile editor --transport streamable-http --bind 127.0.0.1:8801 --allowed-host gw.internal

:: reader pool — read tools only; stays fresh via ensure_fresh; no generalize loop
kb.exe mcp C:\kb\base1 --profile reader --transport streamable-http --bind 127.0.0.1:8802 --allowed-host gw.internal
kb.exe mcp C:\kb\base1 --profile reader --transport streamable-http --bind 127.0.0.1:8803 --allowed-host gw.internal
```

- **Freshness is on every instance** — readers serve up-to-date results (the cooperative tantivy
  writer lock makes concurrent `ensure_fresh` safe). The profile gates **tools**, not freshness.
- **Constraint:** instances sharing one `.glossa` must be on the **same host** (the writer lock is a
  local file lock; unreliable over SMB/NFS). For another host, give it its own index copy.
- **Multiple editors** are fine — the heavy generalize pass is serialized across them by
  `.glossa/generalize.lock`. Notebook writes are serialized by `.glossa/notebook.lock`.

### Two (or more) bases

Each base is an independent set of processes with its own ports; bases may live on different hosts:

```
kb.exe mcp C:\kb\base2 --profile editor --transport streamable-http --bind 127.0.0.1:8811 ...
kb.exe mcp C:\kb\base2 --profile reader --transport streamable-http --bind 127.0.0.1:8812 ...
```

Gateway routes by prefix: `/base1/*` → base1 reader pool (writes/admin → :8801),
`/base2/*` → base2 pool, etc.

## Config

All knobs are CLI flags, with env fallback (flag overrides env). Below that, per setting, is a
third, lowest-priority source: a [deployment config file](#deployment-config-file---config).

| Flag | Env | Default |
|---|---|---|
| `<path>` (positional) | — | nearest indexed root / cwd |
| `--transport stdio\|streamable-http` | `GLOSSA_MCP_TRANSPORT` | `stdio` |
| `--bind <addr>` | `GLOSSA_MCP_BIND` | `127.0.0.1:8080` |
| `--profile reader\|editor\|full` | — | `editor` |
| `--allowed-host <h>` (repeatable) | — | loopback only |
| `--auth-token <TOKEN>` (bearer on `/mcp`; 401 on miss) | `GLOSSA_MCP_TOKEN` | none (unauthenticated) |
| `--insecure <true\|false>` (override the safety interlock; **takes a value**, not a bare flag — see below) | `GLOSSA_MCP_INSECURE` | `false` |
| `--tls-cert <path>` (`tls` feature build) | `GLOSSA_TLS_CERT` | none (plaintext) |
| `--tls-key <path>` (`tls` feature build) | `GLOSSA_TLS_KEY` | none |
| `--tls-client-ca <path>` (opt-in mTLS, `tls` feature build) | `GLOSSA_TLS_CLIENT_CA` | none (no client-cert requirement) |
| per-handshake TLS timeout, seconds (`tls` feature build) | `GLOSSA_MCP_TLS_HANDSHAKE_TIMEOUT_SECS` | `10` |
| max concurrent in-flight TLS handshakes (`tls` feature build) | `GLOSSA_MCP_MAX_HANDSHAKES` | `256` |
| `--session-idle-secs <N>` (opt-in idle timeout; 404→re-init) | `GLOSSA_MCP_SESSION_IDLE_SECS` | `0` (disabled) |
| `--service-name <name>` (Windows SCM) | `GLOSSA_SERVICE_NAME` | `glossa` |
| request timeout on `/mcp` | `GLOSSA_MCP_REQUEST_TIMEOUT_SECS` | `120` |
| max request body size on `/mcp`, bytes | `GLOSSA_MCP_MAX_BODY_BYTES` | `4000000` |
| max concurrent in-flight `/mcp` requests (opt-in overload guard) | `GLOSSA_MCP_MAX_CONCURRENCY` | unset (off) |
| per-IP rate limit, requests/sec (opt-in overload guard) | `GLOSSA_MCP_RATE_LIMIT_PER_SEC` | unset (off) |
| max concurrent open connections (opt-in overload guard) | `GLOSSA_MCP_MAX_CONNECTIONS` | unset (off) |
| `RUST_LOG` (log level) | `RUST_LOG` | `info,tantivy=warn,pdf_oxide=error` |
| log format (`json` → one JSON object per line) | `GLOSSA_LOG_FORMAT` | text |

See **[security-and-operations.md](../security-and-operations.md)** for the auth, idle-timeout,
metrics, JSON-logs, and audit-event details plus an enterprise-readiness scorecard.

## Deployment config file (`--config`)

`--config <path>` (also `GLOSSA_CONFIG`) points at a TOML file holding the BASE settings for one
role — an alternative to spelling out every flag/env var in the unit/script that launches `kb`.
An annotated example covering every section is [`glossa.toml`](glossa.toml).

**One role, one file, both subcommands.** `kb index` and `kb mcp` both read `[corpus]` from the
same file (roots + `state_dir`); `kb mcp` additionally reads `[server]`/`[tls]`/`[limits]`/
`[retrieval]`/`[logging]`. Point the same `--config`/`GLOSSA_CONFIG` at both the provisioning
`kb index` run and the long-lived `kb mcp` process for one role, instead of keeping two copies of
the corpus roots in sync. Running `kb index` against a file that also carries serving-only
sections is not an error — those sections are simply inert for that subcommand (logged at
`debug`, never a warning or failure), since a serving section being present doesn't mean it's
wrong, only unused by this particular subcommand.

**Precedence, per individual setting** (not per file/section — a file can supply `bind` while an
env var supplies `transport`, and each resolves independently):

| Source | Wins over |
|---|---|
| CLI flag (e.g. `--bind`) | env var, config file, built-in default |
| Env var (e.g. `GLOSSA_MCP_BIND`) | config file, built-in default |
| Config file (`[server].bind`) | built-in default only |
| Built-in default | (nothing — last resort) |

A setting left unset in the file, the flag, and the env var falls back to the same built-in
default `kb mcp --help` already documents (the [Config](#config) table above) — the file changes
*where a value can come from*, never what an unset value defaults to.

**Secrets are env/flag-only.** The bearer token is deliberately not a usable config-file
setting: a `token` or `auth_token` key under `[server]` makes the file fail to load at all
(fail closed, not "ignored") — the error names the offending key and points at `--auth-token`/
`GLOSSA_MCP_TOKEN`. Set the token the same way you would without a config file (flag, or an env
var sourced from a systemd `EnvironmentFile` at mode `0600`); never write it into a config file
that might end up checked in or world-readable.

**Strictness.** The file format rejects anything it doesn't recognize: an unknown top-level
section or an unknown/typo'd key inside a known section fails the file to load, rather than
silently ignoring it. A `[tls]` section is only valid in a binary built with `--features tls` —
in a default (non-`tls`) build, a populated (or even empty) `[tls]` section also fails to load,
with an error explaining the feature gate; remove the section (and terminate TLS at a reverse
proxy instead) or rebuild with the feature.

**No wholesale hot reload.** The file is read once, at process start (or once per one-shot `kb
index` run) — there is no mechanism that watches it and re-applies structural changes while `kb
mcp` keeps running. This is consistent with the existing [SIGHUP reload](#log-level-reload-without-a-restart)
scope, which only ever covered the log-level control file, a forced freshen, and (in a
`--features tls` build) re-reading the TLS cert/key/client-CA bytes off the same paths — never a
config *file* reload. Changing anything structural in this file — `[corpus]` roots or
`state_dir`, `[server]` `bind`/`transport`, `[tls]` paths — needs a process restart
(`systemctl restart`, or stop/start the service) to take effect; editing the file alone, or
sending SIGHUP, does not pick up those changes.

See [`glossa-mcp.service`](glossa-mcp.service) for a production unit whose `ExecStart` passes
`--config /etc/glossa/roleA.toml` (with the bearer token still isolated in `EnvironmentFile`, per
the secret rule above).

## Safety interlock

On every `--transport streamable-http` start, the server refuses to come up at all when **all** of
these hold: the bind address is **non-loopback**, **no** `--auth-token`/`GLOSSA_MCP_TOKEN` is set,
and **no** TLS is active (no `tls`-feature cert/key configured) — that combination would otherwise
serve an unauthenticated MCP endpoint to the network. Fix it by setting a token, enabling TLS
(native or via a proxy that also sets `--allowed-host` and effectively keeps the bind loopback), or
explicitly overriding with `--insecure=true` (also `GLOSSA_MCP_INSECURE=true`) — logged as a loud
warning plus a `glossa::audit` `insecure_serve` event, since it means "serve to the network with no
auth." **`--insecure` takes a value** (`--insecure=true` / `--insecure=false`); it is not a bare
flag, so `--insecure` alone is a clap parse error, not "enabled."

## Native TLS (`--features tls` build)

The default release build has no TLS crypto surface (smaller binary, no extra CVE exposure) and
expects a reverse proxy in front. Build with `--features tls` to let `kb mcp` terminate HTTPS
itself:

```
kb mcp <corpus> --transport streamable-http --bind 0.0.0.0:8443 \
  --tls-cert /etc/glossa/tls/server.pem --tls-key /etc/glossa/tls/server.key
# optional mTLS: require + verify client certs against a CA
kb mcp <corpus> --transport streamable-http --bind 0.0.0.0:8443 \
  --tls-cert server.pem --tls-key server.key --tls-client-ca client-ca.pem
```

- Cert/key (and client-CA) are re-read **without dropping the listener or any established
  connection** on SIGHUP or when the file's mtime changes (~5s poll) — a `certbot`/ACME renewal
  that rewrites the same path is picked up automatically; a bad/partial file on reload keeps the
  previous cert serving and logs a warning instead of taking the listener down.
- **Operator caveats specific to native TLS, not present with a reverse proxy in front:**
  - The TLS handshake is performed **off the accept path**: each connection's handshake runs in
    its own spawned task under a per-handshake timeout
    (`GLOSSA_MCP_TLS_HANDSHAKE_TIMEOUT_SECS`, default `10`), so a single slow or stalled (pre-auth)
    handshake can only ever tie up its own task/slot — it does not block acceptance of new TCP
    connections or other in-flight handshakes.
  - Concurrent **in-flight (not-yet-completed) handshakes** are separately bounded by
    `GLOSSA_MCP_MAX_HANDSHAKES` (default `256`), independent of and applied regardless of
    `GLOSSA_MCP_MAX_CONNECTIONS` — this is what protects a native-TLS deployment that leaves the
    connection cap unset (the common case) from a pre-auth connection flood exhausting file
    descriptors. Once the bound is reached, a new pre-auth connection is dropped immediately
    (logged at `warn`) rather than queued.
  - The listener-level connection cap (`GLOSSA_MCP_MAX_CONNECTIONS`) **does** apply over TLS too,
    but — unlike the plaintext build, which caps at raw accept — it caps concurrent
    **established** (post-handshake) sessions: the permit is acquired only once a connection's TLS
    handshake has already succeeded. This is deliberate: capping at raw accept (as the plaintext
    listener does) would let a stalled pre-auth handshake occupy the cap for its whole timeout
    window and lock out legitimate clients entirely. A pre-auth flood can only ever exhaust the
    handshake bound above, never this cap. The acquire is also **non-blocking**: a connection that
    finishes its handshake while the cap is already full is **shed (closed) immediately, not
    queued** — unlike the plaintext listener, which queues excess in the bounded OS accept
    backlog. This is likewise deliberate: queuing here while still holding a pre-auth handshake
    permit would let enough long-lived (e.g. SSE) sessions parked on a full cap exhaust the
    handshake bound too, starving new clients at the pre-auth gate — the connection cap and the
    handshake bound are independent budgets in both directions, never the other's queue.
  - The concurrency-limit (`GLOSSA_MCP_MAX_CONCURRENCY`) and rate-limit
    (`GLOSSA_MCP_RATE_LIMIT_PER_SEC`) guards **do** still apply over TLS (they sit in the axum
    middleware stack, above the listener) — but per-IP **keying** for the rate limiter degrades to
    a **single shared global bucket** over native TLS: `serve_tls` serves the plain `app` with no
    `into_make_service_with_connect_info`, so there's no `ConnectInfo` to key on. That's the same
    limitation as combining `GLOSSA_MCP_MAX_CONNECTIONS` with the rate limiter (see
    [Overload guards](#overload-guards-opt-in) below) — a trusted proxy that sets a forwarded-for
    header restores real per-IP limiting; otherwise the limit still bounds total load, just not
    per-caller.
  - **TLS is not authentication.** A non-loopback bind with TLS active but no
    `--auth-token`/`GLOSSA_MCP_TOKEN` and no client-certificate requirement passes the startup
    interlock (TLS satisfies its no-plaintext requirement) but leaves the endpoint
    **unauthenticated** — anyone who can reach the port can call every tool. Set a token or require
    mTLS (`--tls-client-ca`) for any non-loopback TLS deployment.
  - Net effect: for a high-exposure, no-reverse-proxy TLS deployment, still put a reverse proxy (or
    a network-level connection limiter) in front — native TLS here is meant for a trusted or
    already-bounded network path, not as a full substitute for a hardened edge.

## Overload guards (opt-in)

All three are **off by default** — a proxy/load-balancer deployment delegates this to the proxy;
these exist for a native-TLS-without-proxy deployment or any case where you want the process itself
to shed load:

- `GLOSSA_MCP_MAX_CONCURRENCY=<n>` — caps in-flight `/mcp` requests; the `(n+1)`-th concurrent
  request is shed with `503` instead of queuing unbounded.
- `GLOSSA_MCP_RATE_LIMIT_PER_SEC=<n>` — per-IP token-bucket rate limit on `/mcp`.
- `GLOSSA_MCP_MAX_CONNECTIONS=<n>` — caps concurrently **open** connections at the listener; the
  `(n+1)`-th connection's `accept()` simply doesn't resolve until one closes (the OS backlog
  absorbs the excess instead of unbounded per-connection state).

**Known interaction:** enabling `GLOSSA_MCP_MAX_CONNECTIONS` together with
`GLOSSA_MCP_RATE_LIMIT_PER_SEC` — **or** running a native-TLS build (`--features tls`) with
`GLOSSA_MCP_RATE_LIMIT_PER_SEC` set — degrades per-IP rate-limiting to a **single shared global
bucket** for any request that doesn't carry a trusted `X-Forwarded-For`/`X-Real-Ip`/`Forwarded`
header. Neither the connection-cap listener nor `serve_tls` supply axum's `ConnectInfo`, so the
rate limiter's direct-socket IP fallback has nothing to read in either case. It never fails closed
(no request is rejected outright over this), but per-IP fairness is lost until you set a trusted
forwarded-for header (e.g. from a proxy in front). A startup warning is logged whenever either
combination is in effect.

## Panic barrier + systemd watchdog

Every tool call is dispatched through a panic barrier: if a tool handler panics, the panic is
caught, the client gets a generic internal-error response (the panic payload/message is never
surfaced), a `warn!` is logged, and the `glossa_tool_panics_total` counter increments — the process
itself stays up and keeps serving other requests. Use that counter to alert on a tool that's
panicking repeatedly even though the server as a whole looks healthy.

Under `Type=notify` (the example unit below), the systemd watchdog is armed automatically whenever
`WatchdogSec=` is set in the unit — the server pings systemd at half that interval. The ping is
gated on the **async runtime actually being scheduled** (liveness), not on the `/ready` check — a
long-running blocking index/generalize pass does not by itself starve the ping and trigger a false
restart; only a genuinely wedged tokio runtime does. `READY=1` is sent right after the TCP listener
binds, before the initial index warm-up runs, so `TimeoutStartSec` does not need to accommodate a
slow corpus (see [Two initial-index flows](#two-initial-index-flows) below). `STOPPING=1` is sent on
graceful shutdown (SIGTERM/Ctrl-C).

`Type=notify` only works if the unit also sets `NotifyAccess=main` — systemd's own default
(`NotifyAccess=none`) silently ignores every sd_notify message, which would make the watchdog,
READY and STOPPING signaling all no-ops. The example unit sets this; if you write your own from
scratch, don't drop it.

## Log-level reload without a restart

The tracing filter is reloadable at runtime from a one-line control file at
`<state-dir>/.glossa/loglevel` — write a bare level (`debug`) or a full `RUST_LOG`-style directive
and it takes effect without restarting the process:

```
echo debug > /var/lib/glossa/state/.glossa/loglevel
systemctl reload glossa-roleA-editor      # unix: SIGHUP applies it immediately
# or just wait ~5s — a background poll picks up the file's mtime change on every platform,
# including Windows, where SIGHUP isn't available
```

A missing file, an empty file, or an unparseable directive is a no-op (logged, never crashes on an
operator typo) — the previous filter keeps applying. `systemctl reload` (SIGHUP) also forces an
immediate freshen and, in a `tls`-feature build, re-reads the TLS cert/key/client-CA files; it does
**not** reload `--bind`/`--transport`/`--state-dir`/`--auth-token`/`--allowed-host` — those need
`systemctl restart`.

## Ontology auto-reload

The ontology (`<state-dir>/.glossa/ontology.toml`) is cached in memory and re-parsed automatically
whenever its **mtime** advances — no signal, reload, or restart needed. Edit the file (e.g. via
`kb ontology`) and the next tool call that reads the ontology picks up the change; concurrent reads
never observe a torn state (a newer mtime paired with a stale parse, or vice versa), since the
mtime and the parsed ontology are published together as one atomic snapshot.

## Two initial-index flows

`kb mcp` never requires a separate indexing step before it can serve — but the *first* time it
starts against a corpus, you get to choose between two behaviors:

1. **Lazy warm-up (default, no extra step).** `kb mcp` starts, binds, and reports `READY=1`
   immediately, then kicks off the first index/graph freshen in the background. Reads served during
   this window use whatever's already in the index (empty, on a brand-new corpus) until the freshen
   catches up. Simplest — nothing to schedule — but the first requests after a cold start on a
   large or slow (network-mounted) corpus may see an incomplete index.
2. **Explicit provisioning (`kb index` before/at install).** Run `kb index <corpus>` (or the
   commented `ExecStartPre` line in the example unit) once, so the index/graph are already built
   before `kb mcp` ever starts serving. Slower `systemctl start` (or install step), but the very
   first request after start sees a complete index.

Either way, **`glossa_index_warm`** (Prometheus gauge, 0 or 1) is the signal for "the first freshen
pass has completed" — it is a **different** signal from `/ready` (which only checks that the local
index/graph files are openable, and returns `ready` almost immediately even on a cold, empty
index). On a large corpus behind a slow network mount, `glossa_index_warm` can stay at `0` for a
long time — potentially minutes to hours — after the process reports `READY=1` to systemd and
starts accepting connections. Don't gate external traffic solely on `glossa_index_warm`; use it as
an operator/alerting signal for "still catching up," not a startup readiness gate.

## Ops endpoints (streamable-http)

- `GET /health` — liveness (200 `ok`).
- `GET /ready` — readiness: index + graph openable (200 `ready`, else 503). This is a **local
  file-open check**, not "the corpus is fully indexed" — see
  [Two initial-index flows](#two-initial-index-flows) above for that distinction.
- `GET /metrics` — Prometheus. Index/graph gauges (`glossa_up`, `glossa_index_chunks`,
  `glossa_graph_nodes`, `glossa_graph_edges`, `glossa_graph_dirty`, `glossa_indexing`,
  `glossa_index_warm` — see above), the panic-barrier counter (`glossa_tool_panics_total`),
  network-read resilience gauges (`glossa_index_transient_failures_last_pass`,
  `glossa_index_permanent_skips` counter, `glossa_index_empty_mount_holds` — see
  [Network corpus resilience](#network-corpus-resilience) below) plus HTTP request metrics
  (`glossa_http_requests_total`, `glossa_http_responses_total{class}`,
  `glossa_http_requests_in_flight`, `glossa_http_request_duration_seconds` histogram,
  `glossa_mcp_auth_rejected_total`). `/health`, `/ready`, `/metrics` are exempt from `--auth-token`.

Logs go to **stderr** (stdout is the stdio JSON-RPC channel); `GLOSSA_LOG_FORMAT=json` makes them
one JSON object per line. Each HTTP request is traced (method/path/status/latency). Security **audit
events** (auth rejections, idle-session expiries, write-tool invocations) are emitted on the
`glossa::audit` tracing target — see [security-and-operations.md](../security-and-operations.md).

## Network corpus resilience

When the corpus root sits on a network mount (SMB/NFS), individual reads can fail transiently —
share hiccups, a file mid-copy, a brief attribute-cache stall — without the mount itself going
away. The indexer/freshen paths retry transient failures and refuse to treat a dropped mount as a
mass deletion; this section covers the operator-facing knobs and the two behaviors most likely to
look like a bug: the freshness limitation and the empty-mount hold.

### Transient vs. permanent read failures

A read/stat failure is classified transient (retried) or permanent (skipped, logged, retried again
next pass) by error kind — `TimedOut`, `ConnectionReset`, `ConnectionAborted`, `WouldBlock`,
`Interrupted`, and (unix only, via `raw_os_error()`) `ESTALE`/`ETIMEDOUT`/`ECONNRESET`/
`ECONNABORTED`/`EHOSTUNREACH`/`ENETUNREACH`/`ENETDOWN`/`EIO` are transient; `NotFound` (a real
delete), `PermissionDenied`, and `InvalidData` are permanent. A transient failure is retried up to
`GLOSSA_READ_RETRIES` times with exponential backoff starting at `GLOSSA_READ_RETRY_BACKOFF_MS`
(doubling each attempt). A file that still fails after the retry budget is skipped for this pass
only — it keeps its last-known indexed content rather than being deleted from the index — and is
re-attempted on the next freshen/`kb index` run.

### Freshness limitation and `kb index --force`

Change detection has two tiers, and both have a blind spot worth knowing about:

- **Per-file signature** (`kb index`, and any directory a freshen pass actually re-walks) compares
  whole-second mtime + size. An in-place file replacement that lands the **same size** at the
  **same whole-second mtime** as the previous version is indistinguishable from "unchanged" and
  will not be picked up.
- **Directory-mtime gate** (used by every read tool's `ensure_fresh`/freshen — `search`,
  `glossary`, `resolve`, `graph_query`, etc.) only re-walks a directory when its own mtime changes,
  which happens on add/remove/rename of an entry, **not** on an in-place content edit. So a pure
  in-place edit is invisible to those tools until *something else* (an add/remove/rename) touches
  the same directory and triggers a rescan. The `read` tool is the exception — it reads the file's
  current content straight off disk, not from the index, so it always sees the latest bytes.

Both cases are rare in practice (most editors and sync tools bump mtime and/or size on save), but
when the served result looks stale and neither has caught up: **`kb index --force`** does a full,
unscoped rebuild that re-stats every file regardless of the dirsig/manifest state, and is the
escape hatch for both of these plus a stuck empty-mount hold (below).

### Empty-mount guard

If a root's walk succeeds (no I/O error) but returns **zero files** for a root that previously had
files, that's treated as a possibly-unmounted share rather than a corpus-side mass delete: the
root's previously-indexed content is held at its last-known state instead of being erased, a
warning is logged (`root walk returned zero files but was previously populated; holding stale
index`), and `glossa_index_empty_mount_holds` (see [Ops endpoints](#ops-endpoints-streamable-http))
goes non-zero. The hold clears automatically once the mount comes back and the walk finds files
again. If the root was genuinely emptied on purpose, `kb index --force` resets the manifest before
scanning (no "previously populated" state to compare against), so the empty state sticks.

### Scoped rescan

A freshen pass only re-walks the directories whose own mtime changed since the last pass (per the
directory-mtime gate above), not the whole corpus — this keeps a live MCP query's `ensure_fresh`
call sublinear on a large network corpus (10k+ files) instead of re-stating everything on every
tool call. `kb index` (no `--force`) is the CLI-equivalent incremental pass; `kb index --force`
is the only unscoped full walk.

### Env knobs

| Env | Default | Meaning |
|---|---|---|
| `GLOSSA_READ_RETRIES` | `3` | Max retry attempts for a transient corpus-read failure. |
| `GLOSSA_READ_RETRY_BACKOFF_MS` | `200` | Base backoff before the first retry; doubles each subsequent attempt. |
| `GLOSSA_FRESHEN_DEADLINE_MS` | `3000` | Wall-clock budget for the scoped rescan behind a blocking MCP tool call. Directories not reached within the budget are left unsettled and picked up on the next call — the tool call returns promptly with the current (possibly slightly stale) index rather than blocking on a slow network walk. Only bounds the MCP freshen path; `kb index` is unbounded. |
| `GLOSSA_MIN_RESCAN_MS` | `2000` | Minimum spacing between freshen stat-walks. A read tool called again within this window of the previous freshen serves the current index unchanged instead of re-walking — absorbs SMB/NFS attribute-cache lag and caps per-query walk cost on a hot path where freshen runs on nearly every tool call. |

### Separated corpus / state-dir mode

`--state-dir <path>` (env `GLOSSA_STATE_DIR`) puts `.glossa` (index, graph, locks) on local disk,
separate from one or more `--root [LABEL=]PATH` corpus roots — the recommended layout when the
corpus itself is the network mount, since the writer lock and index/graph commits are unreliable
over SMB/NFS. Most subcommands (`kb index`, `kb search`, `kb read`, `kb graph query`/`doctor`/
`export`, `kb mcp`, …) resolve `--root`/`--state-dir` consistently. Two do not, and pointing them
at the wrong path silently splits your state:

- **`kb graph import <file> <path>` / `kb graph prune <path> -t <type>`** — `<path>` here is a
  separate required positional argument opened directly as the graph's state directory; it does
  **not** go through `--root`/`--state-dir` resolution. In separated/multi-root mode, pass the
  **state-dir** as `<path>`, not the network corpus root — passing the corpus root creates/opens a
  second, disconnected `<corpus-root>/.glossa/graph.sqlite` instead of the one `kb index`/`kb
  search`/`kb mcp` actually use.
- **`kb search --scan`** (the raw ripgrep-style regex mode that bypasses the index) walks only the
  **primary** root — the first `--root` flag, or the positional path. With multiple `--root`
  flags, files under secondary roots are not scanned by `--scan`. Plain `kb search` (BM25) covers
  every configured root; either restrict `--scan` to a single-root corpus or run it once per root.

## Graceful shutdown

One signal stops the loop, drains the listener, and tears down sessions together:
- **Linux / containers:** SIGTERM (`systemctl stop`, `docker stop`) or Ctrl-C.
- **Windows:** Ctrl-C (console) or the SCM Stop/Shutdown control (service).

## Running as a service

### Linux (systemd) — native, foreground binary

`kb mcp ... --transport streamable-http` runs in the foreground; systemd supervises it. One unit
per (base, profile, port).

**Minimal unit** (loopback-only, no watchdog, plaintext behind a proxy):

```ini
# /etc/systemd/system/glossa-base1-editor.service
[Unit]
Description=glossa MCP (base1, editor)
After=network.target

[Service]
ExecStart=/opt/glossa/kb mcp /srv/kb/base1 --profile editor --transport streamable-http --bind 127.0.0.1:8801 --allowed-host gw.internal
Environment=RUST_LOG=info,tantivy=warn
Restart=on-failure
# SIGTERM (the default KillSignal) triggers graceful shutdown
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

```
systemctl daemon-reload && systemctl enable --now glossa-base1-editor
```

**Hardened production unit** — `Type=notify` with the systemd watchdog, `ExecReload` (SIGHUP)
log-level/TLS/freshen reload, `RequiresMountsFor` for a network corpus root, and the sandboxing
directives (`ProtectSystem=strict`, `NoNewPrivileges`, etc.): see
[`glossa-mcp.service`](glossa-mcp.service) for a fully-commented example — copy it, rename per
(base, profile, port), and fill in your paths/bind address/role name. Generate the bearer token for
its `EnvironmentFile` with:

```
openssl rand -hex 32
```

and write it as `GLOSSA_MCP_TOKEN=<value>` into the `EnvironmentFile` path referenced by the unit
(e.g. `/etc/glossa/roleA.env`), `chmod 0600` and owned by the service user — **never** in the unit
file or a checked-in config: unit files are readable via `systemctl cat`, so `EnvironmentFile` is
the only place a secret belongs.

### Windows — native service (SCM)

The binary integrates with the Service Control Manager (`--windows-service`, set in the binPath; not
for manual use). Create one service per (base, profile, port). Prefer the install script or
`New-Service` in PowerShell (elevated) — `sc.exe create` is easy to mis-quote from PowerShell and
can leave the service **Disabled** (error 1058 on start).

```powershell
$ServiceName = "glossa-base1-editor"
$KbExe = "C:\kb\kb.exe"
$CorpusPath = "C:\kb\base1"
$BinaryPathName = "`"$KbExe`" mcp `"$CorpusPath`" --profile editor --transport streamable-http --bind 127.0.0.1:8801 --allowed-host gw.internal --windows-service --service-name $ServiceName"

New-Service -Name $ServiceName -BinaryPathName $BinaryPathName `
  -DisplayName $ServiceName -Description "glossa MCP (base1, editor)" -StartupType Automatic
Start-Service $ServiceName
# ...
Stop-Service $ServiceName    # SCM Stop → graceful shutdown
sc.exe delete $ServiceName
```

Or use `deploy/windows/install-service.ps1` (downloads the release binary and registers the service).

`cmd.exe` alternative (note escaped inner quotes):

```cmd
sc.exe create glossa-base1-editor binPath= "\"C:\kb\kb.exe\" mcp \"C:\kb\base1\" --profile editor --transport streamable-http --bind 127.0.0.1:8801 --allowed-host gw.internal --windows-service --service-name glossa-base1-editor" start= auto
sc.exe description glossa-base1-editor "glossa MCP (base1, editor)"
sc.exe start glossa-base1-editor
```

Notes:
- Do **not** run `kb mcp ... --windows-service` from an interactive console — only the SCM should launch that flag.
- Pass `--service-name` matching the SCM service name (install scripts set this automatically).
- Set `--allowed-host` to the host clients use in the `Host` header (e.g. `127.0.0.1` when binding loopback by IP).
- Set the service log-on account and grant it read access to the corpus + read/write to `.glossa`.
- Run readers as separate services on their own ports (`glossa-base1-reader-8802`, …).
- Chain commands in PowerShell with `;`, not `&&` (Windows PowerShell 5.x does not support `&&`).

## Production

For service install (release binary, all platforms): [service.md](service.md).

For multi-process topology, TLS termination, systemd details, and Windows SCM: [mcp-server.md](mcp-server.md).

## Install binary

Download from [GitHub Releases](https://github.com/metalmon/glossa/releases) — see [install.md](../install.md). Contributors build with `cargo build --release`.
