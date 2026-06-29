<p align="center">
  <img src="docs/assets/logo.svg" alt="glossa" width="200"/>
</p>

<p align="center">
  Offline, file-first knowledge base with a reasoning graph and MCP server for LLM agents.<br/>
  One Rust binary (<code>kb</code>); no external services required for core operation.
</p>

<p align="center">
  <a href="docs/getting-started.md">Getting started</a> ·
  <a href="docs/mcp.md">MCP tools</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="LICENSE">MIT</a>
</p>

glossa indexes documents on disk, serves ranked search and ripgrep-style tools, and maintains a provenance-stamped knowledge graph that agents can query and extend.

## Why glossa

- **Native corpora** — PDF and Office (Word, Excel, PowerPoint) indexed in place; no markdown conversion step.
- **File-first graph** — files stay authoritative; `.glossa/` is a rebuildable overlay with provenance-stamped reasoning nodes.
- **Agent retrieval loop** — BM25 `search`, exact `grep`, chunk `read`, then `glossary` → `neighbors` over solved-case chains.
- **Production MCP** — one offline binary; stdio for local IDEs, streamable HTTP with health/metrics for deploy.

## Features

| Feature | What it does |
|---------|----------------|
| **Native document ingest** | Drop PDF/Office files into the corpus folder — glossa extracts text and chunks them (`p.N` for PDF, sections for Office). Embedded images in `read` for vision-capable agents. No LibreOffice, pandoc, or separate ETL. |
| **Auto-indexing** | `ensure_fresh` before MCP/CLI reads — new or changed files are indexed automatically. Cheap stat-scan when nothing changed; safe across concurrent reader/editor instances. |
| **Auto-generalize** | Editor MCP runs a debounced `graph_generalize` after index changes — SIMILAR links, communities, centrality without agent action. Cross-process lock on `.glossa/generalize.lock`. |
| **Graph without embeddings** | Reasoning types and relations from `ontology.toml`; derived layer (closure, SIMILAR, communities) is deterministic — no vector DB, no model calls. |
| **Gitignore-aware indexing** | Skips paths matched by `.gitignore` / `.ignore` (ripgrep-style). Use `-u` / `--no-ignore` when you need everything. |
| **Profiles, not RBAC** | `reader` / `editor` / `full` hide write tools from the model; every profile still serves fresh data. |
| **Production HTTP MCP** | `--transport streamable-http` at `<bind>/mcp`; `/health`, `/ready`, `/metrics` (Prometheus). Multi-instance reader pools; TLS/auth at the gateway. See [deploy guide](docs/deploy/mcp-server.md). |

Details: [architecture.md](docs/architecture.md), [mcp.md](docs/mcp.md).

## Quickstart

```bash
cargo build --release
./target/release/kb index ./my-corpus
cd ./my-corpus
./target/release/kb search "connection timeout"
./target/release/kb read manual.md
./target/release/kb mcp --profile reader
```

On Windows, the binary is `target\release\kb.exe`. Optional: install [just](https://github.com/casey/just) and run `just build`.

## Architecture

```mermaid
flowchart LR
  files[Corpus files]
  index[Tantivy BM25 index]
  graph[SQLite reasoning graph]
  mcp[MCP server kb mcp]
  agent[LLM agent]

  files --> index
  files --> graph
  agent --> mcp
  mcp --> index
  mcp --> graph
```

- **Structural layer** (auto during index): Document → Section → CONTAINS, MENTIONS, chunk navigation.
- **Reasoning layer** (agent via `graph_upsert`): Symptom → Cause → Resolution and domain-specific types from `ontology.toml`.
- **Derived layer** (`graph generalize`): transitive closure, SIMILAR links, communities, centrality.

See [docs/architecture.md](docs/architecture.md) for details.

## MCP profiles

| Profile | Purpose | Write tools |
|---------|---------|-------------|
| `reader` | Query-only agents | Hidden: index, graph_upsert, graph_generalize, purge, … |
| `editor` | Index + graph editing | All except `purge` |
| `full` | Admin | All tools including `purge` |

Tool reference: [docs/mcp.md](docs/mcp.md). Production deployment: [docs/deploy/mcp-server.md](docs/deploy/mcp-server.md).

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/getting-started.md](docs/getting-started.md) | Install, index, search, read |
| [docs/architecture.md](docs/architecture.md) | Index, extraction, graph layers |
| [docs/mcp.md](docs/mcp.md) | MCP tools and IDE setup |
| [docs/graph-and-ontology.md](docs/graph-and-ontology.md) | Ontology, enrich, generalize |
| [docs/eval-and-training.md](docs/eval-and-training.md) | kb-eval, kb-train, just pipeline |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Backlog and direction |
| [docs/benchmarks.md](docs/benchmarks.md) | Eval run history |

Full index: [docs/README.md](docs/README.md).

## Development

```bash
just build
cargo test -p glossa --release
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgments

glossa is built with excellent Rust libraries:

| Project | Role |
|---------|------|
| [Tantivy](https://github.com/quickwit-oss/tantivy) | BM25 full-text index |
| [oxidize-pdf](https://github.com/bzsanti/oxidizePdf) | PDF text extraction |
| [office_oxide](https://github.com/anthonyjoeseph/office_oxide) | Word, Excel, PowerPoint extraction |

Also: ripgrep ecosystem (`regex`, `globset`, `ignore`), [rusqlite](https://github.com/rusqlite/rusqlite) (graph storage), [rmcp](https://github.com/modelcontextprotocol/rust-sdk) (MCP server).

## License

MIT — see [LICENSE](LICENSE).
