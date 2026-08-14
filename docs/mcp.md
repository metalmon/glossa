# MCP server

glossa exposes its knowledge-base tools via the [Model Context Protocol](https://modelcontextprotocol.io/). Run:

```bash
kb mcp ./my-corpus --profile reader
```

Default transport is **stdio** (JSON-RPC on stdin/stdout). For network deployment see [deploy/mcp-server.md](deploy/mcp-server.md). Install without building: [install.md](install.md).

## Profiles

Profiles control which tools are visible. They are **not** RBAC — all instances can refresh the index; profiles only hide write/admin tools from the tool list.

| Profile | Typical use | Disabled tools |
|---------|-------------|----------------|
| `reader` | Answer agents | `index`, `graph_*`, `purge`, `note`, `del` |
| `editor` | Index + graph + notebook write | `purge` |
| `full` | Admin | (none) |

Reader keeps **notebook read** (`ls` to list notes; note content is read with the regular `read` tool) to inspect specialist notes.

`--no-graph` hides all graph and index tools (search + read only) for eval control arms.

Image output is **off by default** and opt-in via `--vision` (or `GLOSSA_VISION=1`). Without it, `read` strips `page_image` and `include_images` from its schema and never returns `Content::image` (`get_source_file` is unaffected). This is off by default because a figure-heavy page's base64 image payload can overflow the stdio JSON-RPC frame and drop the connection; enabling `--vision` is safe on `--transport streamable-http`. When enabled, all images are served as JPEG (embedded PNGs re-encoded, real JPEGs passed through). `-N` / `--noimage` is a deprecated no-op kept for compatibility (images are already off unless `--vision` is set).

`resolve`, `get_ontology`, and `constraint_solve` are available in every profile. `constraint_solve` and `graph_build` are only registered when the binary is built with `--features constraint` — without it, these tools are absent from the tool list.

## Tools

| Tool | reader | editor | full | Purpose |
|------|:------:|:------:|:----:|---------|
| `search` | ✓ | ✓ | ✓ | BM25 keyword search; returns `[#n] path · snippet` |
| `grep` | ✓ | ✓ | ✓ | Regex/literal over extracted text |
| `glob` | ✓ | ✓ | ✓ | List documents by path glob |
| `read` | ✓ | ✓ | ✓ | Read chunk `#n`, graph node evidence, or a notebook note (path from `ls`). Image output is off unless the server was started with `--vision`; then `page_image: true` returns PDF page `n` as a 200 DPI JPEG for vision models, and embedded/`<img>` images are returned alongside the text (all as JPEG). Without `--vision`, the image params are removed from the schema and responses are text only |
| `get_source_file` | ✓ | ✓ | ✓ | Deliver the original source file behind a citation (`path`, PDF page `n`) as an embedded resource blob for the client to preview/download — for source attribution, not reading. Whole file when ≤ cap (default 10 MB); a larger PDF returns just the cited page as its own PDF |
| `glossary` | ✓ | ✓ | ✓ | Resolve concept → reasoning chain + anchors |
| `related` | ✓ | ✓ | ✓ | SIMILAR / COMMUNITY siblings after glossary |
| `neighbors` | ✓ | ✓ | ✓ | A node's direct structural edges (typed, 1-hop, with direction) |
| `path` | ✓ | ✓ | ✓ | Shortest connection (chain of edges) between two nodes |
| `resolve` | ✓ | ✓ | ✓ | Entity resolution by name |
| `get_ontology` | ✓ | ✓ | ✓ | Knowledge-base ontology as JSON: parameters, constraints, relations, graph-building patterns |
| `constraint_solve` | ✓* | ✓* | ✓* | CSP solver over the constraint graph (`validate` / `infer` / `check`); only available with `--features constraint` |
| `ls` | ✓ | ✓ | ✓ | List notebook notes (agent workspace); read note content with `read` |
| `note` | | ✓ | ✓ | Create/replace — or with `append: true` extend — a notebook note (`doc`, `file`, `content`; `.csp` = validated limit table) |
| `del` | | ✓ | ✓ | Delete a notebook note |
| `index` | | ✓ | ✓ | Incremental index; `force: true` for a full rebuild, `path: <doc>` to reindex just one document |
| `graph_upsert` | | ✓ | ✓ | Create/update reasoning nodes and edges |
| `graph_build` | | ✓* | ✓* | Compile `.csp` limit tables into constraint graph; only available with `--features constraint` |
| `graph_delete` | | ✓ | ✓ | Remove nodes/edges by label |
| `graph_update` | | ✓ | ✓ | Rename or retype a node in place |
| `graph_generalize` | | ✓ | ✓ | Recompute derived layer (non-destructive; no longer reports ungrounded) |
| `graph_doctor` | | ✓ | ✓ | Report ungrounded, stale, and incomplete nodes (report-only; pruning is CLI-only via `kb graph doctor`) |
| `graph_stats` | | ✓ | ✓ | Node/edge counts and community overview |
| `purge` | | | ✓ | Delete entire `.glossa/` |

Source of truth: [`src/mcp.rs`](../src/mcp.rs).

\* Only available when built with `cargo build --features constraint`. Without it, these tools are absent from the tool list.

## Typical agent workflow

1. **`search`** or **`grep`** — find relevant chunks (`[#n]` in results).
2. **`read(path, n)`** — open full chunk text (with `--vision`, embedded office images are returned as JPEG vision content; `page_image: true` renders a PDF page as a JPEG for hard-to-parse tables/layout).
3. **`glossary("concept")`** — jump to reasoning graph; get cause → resolution chain with `read` anchors.
4. **`related(node_id)`** — alternate cases (SIMILAR, COMMUNITY) when the first chain is close but wrong.
5. **`neighbors(node_id)`** / **`path(from, to)`** — inspect a node's direct structural edges, or the shortest chain connecting two nodes.
6. **`graph_upsert`** (editor) — add validated reasoning nodes; response shows what was written.

### `graph_upsert` response

Responses are human-readable for the model:

- **`Written:`** — node ids and resolved edges persisted
- **`Merged:`** — duplicate labels merged into existing nodes
- **`REJECTED — nothing written`** — validation failed (ontology, missing chunk, bad endpoints); fix and retry

Reference endpoints by **node id** (e.g. `sym:...`) or by label. Do not paste ids into `label` fields.

## Transports

### stdio (local)

```bash
kb mcp ./my-corpus --profile editor --transport stdio
```

Use with subprocess-based MCP clients (Claude Desktop, some IDE integrations).

### streamable-http (network)

```bash
kb mcp ./my-corpus --profile reader \
  --transport streamable-http \
  --bind 127.0.0.1:8080 \
  --allowed-host localhost
```

Endpoint: `http://127.0.0.1:8080/mcp`

Environment fallbacks: `GLOSSA_MCP_TRANSPORT`, `GLOSSA_MCP_BIND`.

Quickstart helpers: [`scripts/start-mcp-http.sh`](../scripts/start-mcp-http.sh) / [`scripts/start-mcp-http.ps1`](../scripts/start-mcp-http.ps1) start a streamable-http server against a corpus and print ready-to-paste Cursor `mcpServers` JSON.

Ops endpoints: `/health`, `/ready`, `/metrics` (Prometheus). Bearer auth (`--auth-token`), idle-session timeout (`--session-idle-secs`), JSON logs (`GLOSSA_LOG_FORMAT=json`), and audit events: [security-and-operations.md](security-and-operations.md). Deployment topology: [deploy/mcp-server.md](deploy/mcp-server.md).

## IDE configuration

Install `kb` first: [install.md](install.md). Use your corpus folder path in the examples below.

### Cursor (stdio)

```json
{
  "mcpServers": {
    "glossa": {
      "command": "/usr/local/bin/kb",
      "args": ["mcp", "/path/to/my-documents", "--profile", "reader", "--transport", "stdio"]
    }
  }
}
```

On Windows use the full path to `kb.exe`. Place in `.cursor/mcp.json` (project) or user MCP settings.

### Cursor (HTTP)

If glossa runs as a local HTTP service ([deploy/service.md](deploy/service.md)):

```json
{
  "mcpServers": {
    "glossa-reader": {
      "url": "http://127.0.0.1:8080/mcp"
    }
  }
}
```

Match `--bind` and `--allowed-host` on the server.

See [connect-to-agents.md](connect-to-agents.md) for Claude Desktop and other clients.

## ZeroClaw

→ [integrations/zeroclaw.md](integrations/zeroclaw.md)

## Freshness and maintenance

Every read tool calls `ensure_fresh` (throttled) so new files on disk appear without a manual `index`. Editor instances run a debounced **`graph_generalize`** maintenance loop after index changes, guarded by `.glossa/generalize.lock` across processes. Notebook writes (`note`, `del`) use `.glossa/notebook.lock` the same way.

Index (re)builds are serialized across processes by `.glossa/index.lock`: `index` (incremental, `force`, or single-`path`) and the `ensure_fresh` background scan hold it for the whole rebuild. If another process already holds it, the call skips with a no-op stat instead of racing it — clearing and reopening `.glossa/index` concurrently would otherwise fail on Windows ("Access is denied"). The index is cooperative, so whoever wins the lock leaves it correct.

## Regenerate external tool schemas

After changing MCP tools:

```bash
just tools
```

Writes schemas to `eval/tensorzero/config/tools/` from the live router definitions. Equivalent CLI: `kb mcp dump-tz-tools -d eval/tensorzero/config` — regenerates TensorZero tool config from the live MCP tool definitions (one source of truth).

## Production

For multi-process topology, TLS termination, systemd, and Windows SCM: [deploy/mcp-server.md](deploy/mcp-server.md).
