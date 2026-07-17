# Changelog

All notable changes to glossa are documented here. Release tags ship the **`kb`** binary only; `kb-eval` / `kb-train` are built from source.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.2.0] — 2026-07-17

### Added

- **PDF stack on `pdf_oxide`**: native text extraction, merged-table-cell densification, embedded images in `read`, and `read(page_image: true)` — rendered page PNG (200 DPI) for vision models.
- **Office IR chunking**: Word/Excel/PowerPoint indexed through `office_oxide` DocumentIR — heading/section-aware chunks, merged-cell densify, captions glued to tables.
- **Constraint tables compiler** (`constraint` cargo feature): notebook `.csp` tables compile into a typed constraint graph via `graph_build`; new `glossa-constraint` CSP solver crate; `constraint_solve` and `get_ontology` MCP tools.
- **`graph_stats` doc-owned inventory**: ontology-free listing of a document's nodes and outgoing edges.
- **Eval harness**: SOP-driven constraint eval (`kb-eval-constraint`), GEPA prompt optimization for the constraint pipeline, `kb-train apply-sop-slices` (anchor-driven GEPA slice splicing), and a corpus-agnostic example SOP pack in `eval/sops/example/`.
- **Streamable-HTTP helpers**: `scripts/start-mcp-http.sh` / `.ps1` launchers.

### Changed

- **English-only public surface**: all comments, prompts, and docs translated; corpus-specific assets are local-only; multilingual (ru+en) search remains a supported engine feature.
- **Docs**: MCP tool table corrected (removed stale `cat`/`sed`, added `get_ontology`/`constraint_solve`/`page_image`); documented `kb graph dump/import/prune/path/node` and `kb mcp dump-tz-tools`.

### Removed

- Python helper scripts: GEPA slice applier ported to Rust; one-off graph-path and DICL experiments dropped.

## [0.1.1] — 2026-07-17

### Fixed

- **MCP profiles**: `#[tool_handler]` now uses the instance router so `reader` / `editor` / `full` actually hide write tools from `tools/list` (bare handler was rebuilding a fresh router every call).

## [1.2.0] — 2026-06-28

### Added

- **Quad GEPA**: optimize prod `answer_hotpot` prompt against search, grep, glob, and read micro-tasks via TensorZero (`functions.search`, `grep`, `glob`, `read`, `gepa_reflect`).
- **export-tz**: emit `grep.jsonl` and `glob.jsonl`; synthetic grep/glob rows when episodes lack those tool calls; improved gold path canonicalization.
- **Pareto GEPA**: `pareto_size`, frequency-weighted parent selection, full-val final candidate pick (canonical acceptance after minibatch improve).
- **justfile**: Windows `.exe` binaries for `kb-eval` / `kb-train`; `gepa-apply` fix; default judge in `just eval`; `gepa-reset` / `eval-reset` recipes.
- TZ micro-task templates: `grep/system.minijinja`, `glob/system.minijinja`.

### Changed

- **Eval harness**: TensorZero backend skips per-question corpus wipe/reindex; glossa-train JSON accepts missing `context` / `supporting_facts`.
- **Prod agent prompt**: glob-first retrieval protocol in `answer_hotpot/system.minijinja`.
- **Docs**: rewritten [eval-and-training.md](docs/eval-and-training.md) for quad GEPA and current just recipes.

### Fixed

- GEPA scoring via TZ gateway (IPv4 localhost, episode id skew, tool-call parsing).
- Stale GNU `kb-eval` / `kb-train` artifacts on Windows bypassing fresh builds.
- `gepa-reset` now clears `grep` and `glob` inference history.

## [1.1.0] — 2025-06-XX

- TensorZero eval integration, kb-train enrich, initial GEPA search/read path, justfile dev pipeline.

## [1.0.0]

- Initial public release: file-first index, graph, MCP server, BM25 search, grep, glob, read.

[Unreleased]: https://github.com/metalmon/glossa/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/metalmon/glossa/compare/v1.2.0...v0.1.1
[1.2.0]: https://github.com/metalmon/glossa/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/metalmon/glossa/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/metalmon/glossa/releases/tag/v1.0.0
