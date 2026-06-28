# MCP Server — Enterprise Prod Deployment Prep

**Decisions (2026-06-28):**
- **Transports:** keep **stdio**, add **Streamable HTTP** (the modern MCP remote transport). No SSE.
- **Auth/TLS:** terminated by a **reverse proxy / API gateway** (TLS + OAuth2/OIDC/mTLS) — OUT of our binary.
- **Scaling:** **multiple instances, each with a different `Profile`** — one writer/indexer (`editor`/`full`) + N read-only (`reader`).
- **Platform:** native **Windows binary** (run as a Windows Service, behind the gateway).

## The key correctness insight: freshness is cooperative; the PROFILE gates tools, not freshness
Requirement: **every instance must serve fresh data** — the first client after a file change sees the
update. So readers MUST stay fresh; we do NOT disable their freshness.

This is safe because `ensure_fresh` is already **cooperative**, not owner-bound:
- It does a lock-free `stat` pre-scan; only when there IS a delta does it take the tantivy **writer
  lock**. If another instance holds it (already indexing), this instance no-ops — and its tantivy
  reader auto-reloads on the other's commit (default `OnCommitWithDelay`).
- So "exactly one writer at a time" is enforced by the **lock**, not by the profile. **All instances
  (readers included) call `ensure_fresh`** → the first client after an update triggers the reindex (or
  another instance already did), and everyone converges. Requirement met.

Therefore:
- **`Profile` gates TOOLS only** — readers expose read tools, never `graph_upsert`/`reindex`/`purge`.
  It does NOT gate freshness.
- **All profiles run `ensure_fresh`** (startup reconcile + throttled per-tool `freshen()`). Cheap on
  the common no-op path; cooperative + safe under concurrency.
- **The heavy `generalize` pass is a cross-process singleton.** Multiple editor instances are
  expected. The debounced loop runs only on `Profile != Reader` instances, AND each pass is guarded by
  a **`.glossa/generalize.lock` advisory try-lock**: acquire → run → release; if held by another
  editor, skip this round (the graph is shared, so the holder's pass refreshes the derived layer for
  everyone). Readers READ the derived layer (community/centrality, SIMILAR/closure edges) from the
  shared `graph.sqlite`. Note: `ensure_fresh` needs NO extra lock — tantivy's writer lock already
  serializes the index write (and the graph writes inside that same critical section) across editors;
  only `generalize` (which doesn't open the tantivy writer) needs its own lock.

Constraints this imposes:
- All instances sharing one `.glossa` dir **must be on the same host** (tantivy's writer lock is a
  local file lock; over SMB/NFS it is unreliable). For multi-host: give each host its own writable
  index copy (each freshens cooperatively within its host), or a single indexer publishes a read-only
  snapshot the readers mount (then readers can't self-index → freshness latency = the indexer's
  cadence; only pick this if cross-host RO is a hard requirement).

## Phases

### Phase 1 — Streamable HTTP transport (core)
- Add rmcp feature `transport-streamable-http-server` (+ `axum`) in `Cargo.toml`.
- `kb mcp` gains `--transport stdio|streamable-http` (default `stdio`) and `--bind <addr>`
  (default `127.0.0.1:8080`), also from env (`GLOSSA_MCP_TRANSPORT`, `GLOSSA_MCP_BIND`).
- Wire `StreamableHttpService` + `LocalSessionManager` into an axum router; share ONE `GlossaServer`
  (it is `Clone` + `Arc`-state) across sessions for the single-corpus case.
- Keep the stdio path unchanged.

### Phase 2 — Generalize-loop singleton (correctness for multi-instance)
- ALL profiles keep `ensure_fresh` (startup + per-tool `freshen()`) — readers stay fresh (cooperative
  lock makes concurrent freshen safe). Do NOT gate freshness on profile.
- Spawn the debounced `maintenance_loop` (generalize) ONLY when `Profile != Reader`.
- **Multiple editor instances are expected**, so the generalize pass MUST be guarded by a
  `.glossa/generalize.lock` advisory **try-lock** (cross-process; use `fs4`/`fs2` or `LockFileEx` on
  Windows): acquire before the pass, release after; if already held, skip the round (the holder
  refreshes the shared graph for all). SQLite already prevents corruption — the lock avoids wasted N×
  passes and concurrent churn.
- `ensure_fresh` needs no extra lock (tantivy writer lock already serializes it across editors).
- Test: a reader instance serves fresh results after a file change (calls ensure_fresh) but never
  spawns the generalize loop.

### Phase 3 — Ops surface (health, metrics, logging)
- HTTP listener also serves `GET /health` (liveness) and `GET /ready` (readiness: index openable).
- Structured logging via `tracing` (levels, request-id, per-tool latency) replacing/augmenting the
  eval-only `TraceLog`.
- Optional `GET /metrics` (Prometheus): request counts, latencies, errors, index freshness,
  generalize runs.

### Phase 4 — Lifecycle & config (reliability)
- **Graceful shutdown:** on Ctrl-C / Windows service stop → cancel the maintenance loop, drain
  in-flight requests, flush. (None today.)
- Config precedence: flags > env > file; document all knobs (root, transport, bind, profile, log).

### Phase 5 — Windows packaging
- Release build (`cargo build --release`); document running behind the gateway.
- Optional **Windows Service** mode (`windows-service` crate, or wrap with NSSM/Task Scheduler):
  install/start/stop, service stop → graceful shutdown.
- Recommended topology: gateway (IIS ARR / nginx / Envoy) terminates TLS + OAuth → routes to the
  reader instances; admin/enrichment routed to the single editor/indexer instance.

## Hardening (mostly via profile + gateway)
- Destructive tools (`purge`, `reindex`, `graph_upsert/delete/update/generalize`) only on the
  editor/full instance — readers expose read tools only (already gated by `Profile`; verify).
- Rate-limit, max request size, request timeouts: at the gateway (keep the binary lean).
- Path-traversal: `source_path` validation already guards graph writes; keep.

## Out of scope (delegated to the gateway / platform)
- TLS termination, OAuth2/OIDC/mTLS, rate limiting, WAF — all at the gateway.
- Horizontal autoscaling of writers (only readers scale; the writer is singular by design).

## Status
- [ ] Phase 1 — Streamable HTTP transport
- [ ] Phase 2 — profile-gated freshness
- [ ] Phase 3 — health/metrics/logging
- [ ] Phase 4 — graceful shutdown + config
- [ ] Phase 5 — Windows service + packaging

NOTE: Phases 1/2/4 touch `src/main.rs` + `src/mcp.rs`, which the second agent is actively editing —
coordinate / land after their in-flight `glossary`/`neighbors` work is committed.
