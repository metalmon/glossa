# Changelog

All notable changes to glossa are documented here. Release tags ship the **`kb`** binary only; `kb-eval` / `kb-train` are built from source.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.2.6] — 2026-07-26

### Fixed

- **Section node ids unified to the 1-based ordinal**: structural `Section` nodes are now keyed by their ordinal (`<path>#<n>` — the same `#n` that `read`/`grep`/`search` show), not by the chunk's heading text. Previously `build_section` derived the id from `chunk.location`, which is a heading breadcrumb for most chunks but a bare number for others, so section ids were a mix of `foo.docx#18` and `foo.docx#**Содержание**`. Meanwhile `resolve_section_ref` and `neighbors` always address a section as `<path>#<ordinal>`; when a chunk's stored id was heading-based, an agent's `graph_upsert` edge anchored to `<path>#n` resolved to a section id that did not exist as a node, its endpoint type came back empty, and ontology validation rejected the edge with a misleading `relation '…' endpoints ->X not allowed`. Now every section is `<path>#<ordinal>`, so `path#n` endpoints resolve whether or not the chunk has a heading (PDF page refs like `spec.pdf#4` included). The heading is preserved as the node **label** and as the hierarchy breadcrumb (`CHILD`/`PARENT` still built from `location`), so `glossary`/`search`/`node_ref` are unaffected. Existing corpora need a `reindex` to migrate their section ids.

## [0.2.5] — 2026-07-22

### Added

- **`--noimage` / `-N` flag**: disable all image output in MCP tools — `read` strips `page_image` and `include_images` from its schema and disables image content in responses. `get_source_file` is unaffected. Also settable via `GLOSSA_NO_IMAGE=1` env var.
- **HTML image extraction**: `read` now extracts images referenced by `<img>` tags in `.htm`/`.html` files and returns them as `Content::image` — same as PDF embedded images and DOCX media. Relative paths are resolved relative to the HTML file's directory. Remote URLs (`http://`, `https://`, `ftp://`) and inline `data:` URIs are skipped.

### Changed

- **Image extraction limit removed**: `read` now extracts *all* embedded images by default (previously capped at 4). The `max` parameter was hardcoded and not user-facing; the limit is gone.
- **Shared read logic**: `read` handler delegates to `read_common()` — image/no-image is a single flag controlled by schema stripping, not duplicated handlers.
- **`--noimage` enforces override**: `no_image` flag now stored in `GlossaServer` and forces both `page_image` and `include_images` to `false` in the `read` handler (previously only hid the parameters from the tool schema).
- **Path normalization (SRP)**: all path separator normalization (`/` ↔ `\`, collapsing double-escaped backslashes, host-OS conversion) is now a single `normalize_path()` function called from `resolve_path()`. Removed `deserialize_str_loose` serde helpers — normalization happens at resolve time, not deserialization time.

### Removed

- **`constraint_solve` and `graph_build` tools hidden when feature disabled**: these tools are now excluded from the tool list when the binary is built without `--features constraint` (previously they were always visible and returned a runtime error).

### Fixed

- **Cross-process index lock**: `index_dir` now holds an advisory `.glossa/index.lock` for the whole (re)build. When two processes (editor instances, the MCP server, a CLI `reindex`) index the same base at once, one used to clear `.glossa/index` while the other opened it — an "Access is denied" race on Windows. The lock serializes the rebuild: the first holder proceeds, the rest skip with a no-op stat (the index is cooperative, so whoever wins leaves it correct). RAII guard releases on function exit.
- **Transient Windows write retry**: tantivy index writes (`write_chunks`, `delete_path`, `index_dir`) now retry past a transient "Access is denied (os error 5)" — a just-created or mmap'd index file briefly held by Windows Defender or a lingering reader handle. A short bounded backoff clears it; only permission-denied IO errors retry, real failures propagate immediately. Fixes intermittent Windows CI test failures.

## [0.2.2] — 2026-07-19

### Added

- **`kb cat <file>`**: print a document's whole extracted text straight from disk — a `cat` that understands PDF, Word, Excel, and PowerPoint. No index, no `.glossa`; deliberately file-only (unlike `read`, which also resolves result numbers, graph nodes, and notebook notes).
- **CLI-for-agents positioning**: README now leads with the dual **CLI or MCP** story, plus a new [cli-for-agents.md](docs/cli-for-agents.md) guide — shell out to `kb cat` / `kb grep` / `kb read` as the `grep`/`cat` an agent lacks for Office & PDF, or run the MCP server for a persistent graph-backed corpus.

## [0.2.1] — 2026-07-18

### Fixed

- **MCP tool args tolerate string-encoded primitives**: LLM/MCP clients often JSON-encode numbers and booleans as strings (`"n": "8"`, `"page_image": "true"`), which strict deserialization rejected (`invalid type: string "8", expected u32`). Every primitive field across the tool arg structs (`search`, `read`, `get_source_file`, `neighbors`, `grep`, …) now accepts both native and string forms; the advertised schema stays `integer`/`boolean`.

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
