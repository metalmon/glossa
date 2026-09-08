# Troubleshooting

Practical answers to problems you'll actually hit, each grounded in the behavior that causes it.
For the full flag/env surface referenced below, see [cli-reference.md](cli-reference.md) and
[configuration.md](configuration.md).

## Server refuses to start: "refusing to serve MCP on non-loopback bind..."

This is the **startup safety interlock**: `kb mcp --transport streamable-http` refuses to come up
when *all* of these hold — the bind address is non-loopback, no `--auth-token`/`GLOSSA_MCP_TOKEN`
is set, and no TLS is active. It exists so a forgotten token/proxy fails closed instead of silently
serving the open internet. Fix it one of three ways:

- Set a bearer token: `--auth-token <TOKEN>` or `GLOSSA_MCP_TOKEN=<TOKEN>`.
- Enable TLS (native `--tls-cert`/`--tls-key` in a `--features tls` build, or terminate TLS at a
  reverse proxy in front of a loopback bind).
- Explicitly override with `--insecure=true` (also `GLOSSA_MCP_INSECURE=true`) — logs a loud
  warning and a `glossa::audit` `insecure_serve` event. Note `--insecure` **takes a value**; the
  bare flag with no `=true` is a parse error, not "enabled."

Binding to `127.0.0.1` (the default) never triggers this — the interlock only fires on a
non-loopback bind. Details: [security-and-operations.md § Safety interlock](security-and-operations.md#safety-interlock).

## 401 on `/mcp`

Every `/mcp` request must send `Authorization: Bearer <TOKEN>` once `--auth-token`/
`GLOSSA_MCP_TOKEN` is set; anything else (missing header, wrong token) gets **401**. `/health`,
`/ready`, and `/metrics` are never guarded, so probes and scrapes keep working without the token.
See [security-and-operations.md § Authentication](security-and-operations.md#authentication).

## Results look stale after files changed

Every read path (`search`/`read`/`grep`/`glob`, every MCP tool) calls `ensure_fresh` first, which
stat-scans the corpus and incrementally reindexes changed files — but scans are throttled to at
most once per `GLOSSA_MIN_RESCAN_MS` (default `2000` ms) so a hot query loop doesn't re-walk the
filesystem on every call. If you edited a file and query again within that window, you may see the
old text; wait out the window or lower `GLOSSA_MIN_RESCAN_MS`. If a manual `kb index --force` is
needed for a genuinely stuck state, run it — this is also the only pass that rebuilds the
answer-grounding DF sidecar (`.glossa/df`).

## Corpus is on a network mount

Keep `--state-dir` (or `GLOSSA_STATE_DIR`) pointed at **local** disk even when one or more corpus
roots are on a network share (SMB/NFS). glossa's index/graph writer relies on file locks and WAL
journaling that are unreliable over network filesystems; `.glossa/` living locally avoids
corruption and lock contention. If multiple MCP instances share one `.glossa/`, they must be on the
**same host** for the same reason. See
[security-and-operations.md](security-and-operations.md) and
[deploy/mcp-server.md § Topology](deploy/mcp-server.md#topology-one-process-per-base--profile).

## A `.docx`/`.xlsx`/`.pptx` extracts poorly or comes back empty

Known extraction limits, not a bug to work around with a different flag:

- Charts are extracted as **data** (series/categories/values as a text table), never rendered as an
  image — for a faithful visual rendering, use a PDF source instead (PDF pages are rasterized for
  vision).
- Legacy binary `.doc`/`.xls`/`.ppt` charts (OLE/BIFF) are not extracted; OOXML and ODF charts are.
- `.ppt` embedded images are not extracted (`.doc`/`.xls` raster images are); vector metafiles
  (EMF/WMF/PICT) are never rasterized.
- ODF cell-range charts referencing another sheet/document, multi-column category ranges, or a
  merged-cell region inside the referenced range are best-effort and may under-resolve — the chart
  is skipped with a log warning, extraction never crashes.

Full list: [architecture.md § Known extraction limitations](architecture.md#known-extraction-limitations).

## `[tls]` section rejected / TLS flags have no effect

The **default build has no TLS crypto surface** (smaller binary, no extra CVE exposure) — it
expects a reverse proxy in front for TLS. A `--config` file with a `[tls]` section (even an empty
one) fails to load on a non-`tls` build, and `--tls-cert`/`--tls-key`/`--tls-client-ca` are only
compiled in behind `cargo build --features tls`. Either rebuild with `--features tls` to terminate
TLS natively, or remove the `[tls]` section and terminate TLS at a reverse proxy instead. See
[deploy/mcp-server.md § Native TLS](deploy/mcp-server.md#native-tls---features-tls-build).

## "No results" from `search`/`grep`/`glossary`

- Confirm the path you're querying is actually an indexed root — a subdirectory works (glossa finds
  the nearest `.glossa/`), but a sibling directory outside any root won't.
- Confirm you've indexed at least once (`kb index <path>`) — a brand-new corpus with no `.glossa/`
  yet has nothing to search until the first index (or the first MCP `ensure_fresh` call).
- For `graph glossary`/`glossary`: an empty result means no reasoning nodes match yet — the
  reasoning layer is agent/`kbx`-authored, not automatic, so a freshly indexed corpus with no
  ontology work done has a structural graph but no reasoning nodes. See
  [graph-lifecycle.md](graph-lifecycle.md).
- `--scope`/`scope` and `--glob`/`-g` are **ANDed** — an overly narrow combination of both silently
  returns nothing rather than erroring.

## `constraint_solve`/`graph_build` tool missing from the MCP tool list

Both are only registered when the binary is built with `cargo build --features constraint`; the
default build omits them (a leaner default tool set). Rebuild with the feature if you need CSP
validation over `.csp` limit tables. See [constraint-tables-compiler.md](constraint-tables-compiler.md).

## See also

- [cli-reference.md](cli-reference.md) — every flag
- [configuration.md](configuration.md) — every environment variable and config-file key
- [security-and-operations.md](security-and-operations.md) — the full hardening/observability picture
- [graph-and-ontology.md § Graph doctor](graph-and-ontology.md#graph-doctor) — diagnosing a graph
  that looks wrong (ungrounded/stale/incomplete/dangling nodes)
