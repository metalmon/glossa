# glossa — roadmap & backlog

Status as of 2026-06-24: Milestones 1–5 + real `graph_upsert` are merged to `master`.
Pure-Rust single offline binary (`kb`); ~54 tests green; no C compiled on shipping targets.

## What works today
- Extraction: md, docx/doc/xlsx/xls/pptx/ppt (office_oxide), pdf (pdf-extract).
- Search: ripgrep-syntax scan (default) **and** BM25 ranked (`--rank`) with RU/EN stemming.
- Knowledge graph (redb): provenance-stamped nodes/edges, `ontology.toml` validation, bounded
  traversal (`neighbors`/`path`), deterministic auto layer-1 (Document/Section/CONTAINS) during index.
- MCP server (`kb mcp --profile reader|editor|full`): `search`, `read` (text + embedded images as
  vision content), `glossary`, `neighbors`, `index`, `reindex`, `resolve`, `graph_upsert` (validated,
  provenance-stamped), `purge`. gitignore-aware indexing.

## Technical backlog (carry-forward, non-blocking)
- **Graph crash-atomicity**: fold one file's node/edge writes into a single redb write txn so a
  mid-file crash can't leave a partial graph the manifest then skips as "unchanged". (`reindex`
  recovers today.)
- **`--expand`**: glossary query expansion — needs the layer-2 `Term`/co-occurrence layer (not built).
- **HTTP/streamable transport** for the MCP server (`transport-streamable-http-server`) — stdio only today.
- **PDF page-level locations + PDF image extraction** (needs a page-aware pure-Rust PDF lib; office
  images only today).
- `type_of` in `upsert` swallows `get_node` errors via `.ok()` (fail-closed) — propagate.
- `read_region` returns empty string for unknown extensions silently; `join("\n")` may double newlines.
- Bounded-traversal depth semantics doc-comments (`neighbors`=hops vs `path`=node-count).
- Secondary indexes for `resolve`/`delete_by_source` (currently O(n) scans).
- Cosmetic: `Mcp` enum-variant vs match-arm ordering.

## Product roadmap (staged)

### Stage 1 — local bring-up (IN PROGRESS)
- Build the release binary; connect glossa as an MCP server to the controlling agent (Claude Code).
- Smoke-test on a real knowledge base (`kb-test/`, git-ignored): `index` → `search`/`--rank` → `graph`.
- Hand the operator CLI commands to try in the terminal.

### Stage 2 — RAG benchmarks
- Run established RAG benchmarks; capture our metrics.
- Two model tiers: (a) the smart controlling agent (Claude), (b) a weak local model
  (Qwen3.5-4B via LM Studio) — compare retrieval/answer quality and graph-build quality.

### Stage 3 — multi-hop benchmarks
- First **without graph saturation** (code-built layers 1–2 only), driven by the smart agent.
- Then **after saturation** (the agent has built layers 3–4 via `graph_upsert`) — measure the lift.

### Stage 4 — agent skills
- Ship companion skills teaching agents to use glossa (search → read → build-graph → answer).
- Pattern: a **base ontology + base skill**, refined per domain (legal/medical/eng/…).

### Stage 5 — testing toolkit + heatmap (domain refinement loop)
- The test base carries example **questions + correct answers**.
- Per-step **regex matchers** that check whether the correct answer is found at each retrieval/hop step.
- **Trace + reverse-trace** of the search path; render the steps on a **heatmap** so the operator can
  see, visually, how retrieval is doing across questions.
- Mine **common patterns** across traces → write the **domain skills** from them.
- Deliver this as a convenient in-repo toolkit + a detailed manual.

## Notes
- Everything stays pure Rust / offline / single binary. The graph is a disposable overlay over files
  (File-First) — deleting `.glossa/` loses nothing; it rebuilds from the index.
- Profiles are tool-visibility gating, not RBAC: `reader` (query), `editor` (+ build/index), `full` (+ admin).
