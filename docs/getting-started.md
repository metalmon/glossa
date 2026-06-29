# Getting started

This guide is for **developers** building glossa from source. **Operators** should start with [install.md](install.md) (GitHub Release, no Rust) and [connect-to-agents.md](connect-to-agents.md).

For MCP agent integration, see [mcp.md](mcp.md).

## Build

```bash
cargo build --release
```

The binary is `target/release/kb` (or `target\release\kb.exe` on Windows). With [just](https://github.com/casey/just) installed:

```bash
just build
```

## Corpus layout

Point glossa at a directory containing your documents. On first index, glossa creates:

```
my-corpus/
  document.pdf
  manual.md
  .glossa/           # created by glossa (git-ignore this)
    index/           # Tantivy BM25 index
    graph.sqlite     # reasoning graph
    manifest.json    # file change tracking
    ontology.toml    # optional domain overlay (see below)
```

Deleting `.glossa/` loses only derived state; re-index rebuilds everything from files.

### Auto-indexing

After the first `kb index`, you usually **don't need to re-run it**. When you use MCP (`search`, `read`, `grep`, …) or CLI ranked search, glossa stat-scans the corpus and incrementally indexes changed files. Drop a new PDF into the folder — the next agent query sees it.

Manual index is still available:

```bash
kb index          # incremental
kb reindex        # full rebuild
```

The MCP server also reconciles on startup. See [mcp.md § Freshness](mcp.md#freshness-and-maintenance).

## Index

```bash
kb index ./my-corpus
```

Incremental: changed files are re-extracted; removed files drop from the index and graph. Full rebuild:

```bash
kb reindex ./my-corpus
```

Indexing respects `.gitignore` by default. Use `kb search --no-ignore` when you need to include ignored paths in scans.

## Search

Run commands from the corpus directory (or any subdirectory — glossa finds the nearest `.glossa/` root).

**Ranked keyword search** (BM25, morphology-aware stemming):

```bash
cd ./my-corpus
kb search "connection timeout"
kb search "maxTsdr" -l 20
kb search "print queue" -g '*.pdf'
```

Output is ripgrep-compatible by default (`path:location: snippet`). In a terminal, format auto-switches to numbered lines; force with `-f pretty` or `-f rg`.

**Literal / regex scan** over extracted text (exact tokens, error codes, version strings):

```bash
kb grep maxTsdr
kb grep -F "5.7.2" -g '*.pdf'
```

**Slow regex scan** over raw files (bypasses index, not stemmed):

```bash
kb search "pattern" --scan
```

### When to use which

| Tool | CLI | Use when |
|------|-----|----------|
| `search` | `kb search` | Natural-language or keyword lookup; fuzzy, ranked |
| `grep` | `kb grep` | Exact token, code, version, regex over indexed text |
| `glob` | `kb glob '*.pdf'` | Discover documents by path pattern |

## Read

Open a hit from search (by result number) or by path:

```bash
kb search "timeout"
kb read 1          # first hit from last search
kb read manual.md
kb read manual.md "p.3"   # PDF page
```

## Graph (optional)

Inspect the knowledge graph from the CLI:

```bash
kb graph stats
kb graph glossary "connection loss"
kb graph near sym:abc123
kb graph generalize
```

For a support-domain ontology overlay, copy the reference file:

```bash
mkdir -p ./my-corpus/.glossa
cp eval/ontology-support.toml ./my-corpus/.glossa/ontology.toml
```

Then re-index or run enrich (see [graph-and-ontology.md](graph-and-ontology.md)).

## MCP server (local)

```bash
kb mcp ./my-corpus --profile reader --transport stdio
```

For Cursor or other HTTP clients, see [mcp.md](mcp.md) and [deploy/mcp-server.md](deploy/mcp-server.md).

## Next steps

- [connect-to-agents.md](connect-to-agents.md) — Claude, Cursor, ZeroClaw
- [install.md](install.md) — release install (no build)
- [architecture.md](architecture.md) — how indexing and the graph fit together
- [mcp.md](mcp.md) — full tool list for agents
- [graph-and-ontology.md](graph-and-ontology.md) — reasoning graph workflow
