# MCP server

glossa exposes its knowledge-base tools via the [Model Context Protocol](https://modelcontextprotocol.io/). Run:

```bash
kb mcp ./my-corpus --profile reader
```

Default transport is **stdio** (JSON-RPC on stdin/stdout). For network deployment see [deploy/mcp-server.md](deploy/mcp-server.md).

## Profiles

Profiles control which tools are visible. They are **not** RBAC — all instances can refresh the index; profiles only hide write/admin tools from the tool list.

| Profile | Typical use | Disabled tools |
|---------|-------------|----------------|
| `reader` | Answer agents | `index`, `reindex`, `graph_upsert`, `graph_delete`, `graph_update`, `graph_generalize`, `graph_stats`, `purge` |
| `editor` | Index + graph editing | `purge` |
| `full` | Admin | (none) |

`--no-graph` hides all graph and index tools (search + read only) for eval control arms.

`resolve` is available in every profile.

## Tools (15)

| Tool | reader | editor | full | Purpose |
|------|:------:|:------:|:----:|---------|
| `search` | ✓ | ✓ | ✓ | BM25 keyword search; returns `[#n] path · snippet` |
| `grep` | ✓ | ✓ | ✓ | Regex/literal over extracted text |
| `glob` | ✓ | ✓ | ✓ | List documents by path glob |
| `read` | ✓ | ✓ | ✓ | Read chunk `#n` or graph node evidence |
| `glossary` | ✓ | ✓ | ✓ | Resolve concept → reasoning chain + anchors |
| `neighbors` | ✓ | ✓ | ✓ | SIMILAR / COMMUNITY siblings after glossary |
| `resolve` | ✓ | ✓ | ✓ | Entity resolution by name |
| `index` | | ✓ | ✓ | Incremental index |
| `reindex` | | ✓ | ✓ | Full rebuild |
| `graph_upsert` | | ✓ | ✓ | Create/update reasoning nodes and edges |
| `graph_delete` | | ✓ | ✓ | Remove nodes/edges by label |
| `graph_update` | | ✓ | ✓ | Rename or retype a node in place |
| `graph_generalize` | | ✓ | ✓ | Recompute derived layer (non-destructive) |
| `graph_stats` | | ✓ | ✓ | Node/edge counts and community overview |
| `purge` | | | ✓ | Delete entire `.glossa/` |

Source of truth: [`src/mcp.rs`](../src/mcp.rs).

## Typical agent workflow

1. **`search`** or **`grep`** — find relevant chunks (`[#n]` in results).
2. **`read(path, n)`** — open full chunk text (embedded office images returned as vision content when supported).
3. **`glossary("concept")`** — jump to reasoning graph; get cause → resolution chain with `read` anchors.
4. **`neighbors(node_id)`** — alternate cases (SIMILAR, COMMUNITY) when the first chain is close but wrong.
5. **`graph_upsert`** (editor) — add validated reasoning nodes; response shows what was written.

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

Ops endpoints: `/health`, `/ready`, `/metrics` (Prometheus). Details in [deploy/mcp-server.md](deploy/mcp-server.md).

## IDE configuration

### Cursor (HTTP)

If glossa runs as a local HTTP server:

```json
{
  "mcpServers": {
    "glossa-reader": {
      "url": "http://127.0.0.1:8080/mcp"
    }
  }
}
```

Place in `.cursor/mcp.json` (project) or user MCP settings. Match `--bind` and `--allowed-host` to your setup.

### stdio subprocess

Configure your client to spawn:

```
/path/to/kb mcp /path/to/corpus --profile reader --transport stdio
```

Working directory should allow resolving the corpus path.

## Freshness and maintenance

Every read tool calls `ensure_fresh` (throttled) so new files on disk appear without a manual `index`. Editor instances run a debounced **`graph_generalize`** maintenance loop after index changes, guarded by `.glossa/generalize.lock` across processes.

## Regenerate external tool schemas

After changing MCP tools:

```bash
just tools
```

Writes schemas to `eval/tensorzero/config/tools/` from the live router definitions.

## Production

For multi-process topology, TLS termination, systemd, and Windows SCM: [deploy/mcp-server.md](deploy/mcp-server.md).
