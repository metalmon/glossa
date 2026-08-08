# Changelog

All notable changes to glossa are documented here. Release tags ship the **`kb`** binary only; `kb-eval` / `kb-train` are built from source.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.3.1] — 2026-08-09

### Added

- **`kb graph import` now MERGES by default** (was: always replace). Import upserts a graph JSON into the existing graph, keeping everything already there — so you can top up a graph or accumulate several files without losing prior content. Node identity is normalized `label` + type, so re-importing the same file is idempotent (the graph does not grow) and imports from different sources converge. The previous behavior is available with `--mode replace` (prune the file's exported types first, then upsert — file = source of truth for those types). One sharp edge, by design: an incoming file that reuses an existing node's id with a different label overwrites that node in place (last-writer-by-id), since the store key is the id.

### Fixed

- **Concurrent agent graph writes are now safe**: `graph_upsert` takes a cross-process lock, so multiple MCP clients / agents writing the same `graph.sqlite` no longer race, corrupt, or drop writes.
- **Index freshness**: the directory signature never advances while a file inside is mid-write, so a racing indexer can't record a half-written file as up-to-date.
- **Path lookups** tolerate mangled/rewritten document paths, so a citation still resolves instead of missing.

### Changed

- Path clamping uses `clamp` instead of a hand-rolled `min`/`max` chain (internal cleanup, no behavior change).

### Tooling (kb-eval / kb-train — built from source, not shipped in the `kb` binary)

- Graph-transfer eval foundations: a MuSiQue loader with alias-aware EM/F1 scoring, and the OpenAI reader backend now exposes the graph tools (`glossary`/`related`/`neighbors`/`path`) for a graph-ON arm alongside a `--no-graph` flat baseline.

## [0.3.0] — 2026-08-06

### Added

- **Baked ontology presets + `kb ontology` CLI**: 26 ready-made task ontologies ship inside the binary — Tier 1 conformance/compliance shapes (`compliance`, `tender`, `contract`, `certification`, `qa-inspection`, `audit`, `reg-change`, `data-privacy`, `access-governance`, `hr-compliance`, `risk-register`, `fmea`, `policy`) and Tier 2 operational shapes (`support`, `sop`, `faq`, `traceability`, `vendor`, `product-catalog`, `customer-journey`, `okr`, `project-schedule`, `decision-log`, `timeline`, `competency`, `org-roles`). Pick one at first index with `kb index --ontology <name>` (materializes `.glossa/ontology.toml`, then indexes) or manage them with `kb ontology list` / `show` / `init` / `suggest` — offline fuzzy + free-text matching, aliases, and typo hints, no model call. Each preset is a thin reasoning skeleton with exactly one grounded terminal (no separate Evidence/citation-proxy node). The on-disk file is always the source of truth; `--force` to overwrite an existing one.
- **Valid-time (temporality Phase 1)**: a reasoning node can carry a validity interval (`valid_from`/`valid_to`, any ISO-8601 granularity, raw expression preserved), stored in an authored `node_validity` side table that survives `generalize`/reindex. A per-entity `requires_validity` flag (mirror of `requires_grounding`) makes `graph_upsert` reject an untimed node of that type. Graph reads take `--as-of <date>` (CLI `glossary`/`ls`/`near`/`node`/`dump`) and `as_of` (MCP `glossary`/`neighbors`/`related`), hiding facts outside their window; `read`/`node` report per-node status (current/future/expired/superseded) and a `SUPERSEDES` relation chains revisions. World time ≠ document time throughout — validity is when a fact holds, provenance is when it was recorded.
- **Graph doctor (`kb graph doctor` + MCP `graph_doctor`)**: one consolidated health report over three doubts — `ungrounded` (a `requires_grounding` node lost its live `MENTIONS`), **`stale`** (the node's source file signature drifted since it was grounded), and `incomplete` (off the reasoning spine). The CLI offers targeted `--prune-ungrounded` / `--prune-incomplete` (destructive); the MCP tool is report-only; stale is never pruned — it is re-grounded. Reader-facing `read`/`glossary`/`neighbors` now show an inline `⚠ stale` marker so an answering agent can de-prioritize a drifted fact and re-read the source.
- **Mandatory grounding for reasoning nodes**: an ontology entity may set `requires_grounding = true` (e.g. `Resolution`). `graph_upsert` then rejects the whole call if such a node has no `MENTIONS` edge to a source section — in the batch or already in the graph — with an actionable message (document-agnostic: the `MENTIONS` may cite any indexed document). `graph generalize` treats a required node that lost a *live* `MENTIONS` (none, or a dangling target after its source was removed) as a **separate `ungrounded` bucket** from off-spine `prune_candidates`: it reports them (in `graph_generalize` output, for re-grounding) and prunes them only under the new CLI flag `--prune-ungrounded` (MCP stays non-destructive). Grounding is transitive — only the spine's grounding node needs a direct `MENTIONS`.
- **`get_ontology` advertises grounding/validity**: the exported ontology now carries per-entity `requires_grounding` / `requires_validity` booleans plus a validity convention note, so an enricher prepares `MENTIONS` and valid-time up front instead of learning by `graph_upsert` rejection.
- **`kb graph dump -f html` — self-contained offline graph explorer**: emits a single HTML file with the graph library (Cytoscape.js) and data embedded, so it pulls nothing at runtime and works fully offline. Search-first UX: a glossary search returns matching nodes; selecting one opens a focused local view (the node centred with its typed relations, similar nodes, and `MENTIONS` sources) that you traverse by clicking. Density is label-aware — nodes pack tightly and spread only where their texts would overlap — and text stays at a readable size (the view never shrinks it). Colours/legend are derived from the data; light/dark theme follows the system with a manual toggle; UI is English, switching to Russian on a `ru` locale; responsive for mobile. Viewer refinements: infinite-scroll glossary search, dark theme, a mini-stat header, and temporal display.

### Changed

- **`generalize` is derived-layer only**: `graph generalize` / `graph_generalize` now recompute just the derived layer (closure, SIMILAR, communities, centrality). All hygiene — ungrounded/incomplete detection and pruning — moved to `graph doctor`; `graph_generalize` no longer reports `ungrounded_candidates` and the `--prune-*` flags moved off `generalize`.

### Removed

- **Dead PDF table-detection fallback**: the structured PDF table path (`extract_tables` → markdown) was gated behind an empty-text branch, so it fired only on text-less pages (where the detector has nothing to work with) and never on real tables. It is removed along with `src/extract/pdf_table.rs`; PDF extraction is flat layout-text as before (no behavior change). Merged-cell densification remains on the Office path; a real PDF-table feature can revive the merge later.

### Fixed

- **Grounding `file_sig` resolved against the corpus root**: the producer (`graph_upsert`) and the consumers now resolve `source_path` against the same base (the corpus root, not the CWD), so staleness detection works under a normal off-CWD MCP deployment.
- **`requires_validity` on dedup-merge**: valid-time is remapped through `apply_upsert`'s merge, so a node deduped onto a canonical id keeps its interval.
- **Traceability preset closure**: the requirement→test shortcut closure and certification wording were corrected.
- **Glob backslash normalization**: `\` is normalized to `/` in glob patterns so slash handling is idempotent across platforms.
- **Stored path keys normalized to forward slash**: the index and graph now key a document by its forward-slash relative path, decided in one place (`rel_key`). A lone backslash is an escape character that gets mangled through MCP tool args, so mixed-separator keys could split the same document across the index and graph; forward slashes are JSON/transport-safe and Windows-native, so every displayed path round-trips. Also drops a redundant `@owner` suffix in glossary output.
- **Node search index rebuilds on content drift, not just document count**: the glossary/resolve BM25 node index was rebuilt only when the document count changed, so same-count edits (alias adds, path migrations) were missed and glossary/resolve went stale. It now rebuilds when the underlying content drifts.

## [0.2.7] - 2026-08-04

### Added

- **`neighbors` MCP tool (structural, typed 1-hop edges)**: lists a node's direct outgoing/incoming edges (`edge_types` filter, `direction: out|in|both`), each rendered with its real direction (`--REL-->` / `<--REL--`) and a `read path #n` anchor. This is the FACTUAL graph-structure tool — for fuzzy "similar cases" use `related`.
- **`path` MCP tool (shortest connection between two nodes)**: undirected BFS between two node refs (or chunk `path`+`n` pairs), rendered as a chain of hops with each edge's real direction and a `read path #n` anchor per hop. `max_depth` defaults to 6, capped at 12.
- **`--vision` flag (image output opt-in)**: image output in the `read` tool — embedded figures and `page_image` — is served only when the server is started with `--vision` (env `GLOSSA_VISION`). When enabled, all images go out as **JPEG**: embedded PNGs are decoded and re-encoded, PDF-embedded/rasterized JPEGs pass through unchanged. JPEG is far smaller on the wire than base64-PNG, so no size cap is needed. Safe on `--transport streamable-http`.

### Changed

- **`neighbors` renamed to `related`**: the old fuzzy "similar/community cases after glossary" tool is now called `related` — its behavior is unchanged, only the name. `neighbors` is now the new structural tool described above. Tool descriptions, the eval harness (routing/caching/CLI dispatch), the generated TensorZero tool config, and the `answer_hotpot` prompt were all updated to match.
- **Image output is now OFF by default**: previously `read` returned embedded images unless `--noimage` was set. A figure-heavy page (e.g. an HTML manual page with many embedded PNGs) base64-encoded every image into a single stdio JSON-RPC frame, which could exceed the transport frame limit and drop the connection. Images are now opt-in via `--vision` (see above). `-N` / `--noimage` is kept as a hidden, deprecated no-op that still forces images off.
- **Public examples and fixtures neutralized**: sample data in prompts, test fixtures, the eval/integration fixtures, and the changelog were switched to English and generic fieldbus examples, so the public surface carries no domain-specific sample content. Internal only — no behavior change.

### Fixed

- **Structural `neighbors` no longer emits a self-loop**: a node that carried a structural edge to itself surfaced as its own neighbour in both directions; the self-loop is now de-duplicated out of the `neighbors` result.

## [0.2.6] — 2026-07-26

### Fixed

- **`note` tool description no longer pushes `.csp` for ordinary notes**: the `note`/`file`/`content` schema descriptions led with the `.csp` limit-table format, biasing agents to reach for it (tab-separated, header-validated) even for free-form notes. They now lead with free-form (any extension, e.g. `.md`) and mark `.csp` as the niche constraint-graph format. Description-only change, no behavior change.
- **Notebook notes for documents in subdirectories**: `note`/`ls`/`read`/`del` now work for any indexed document, not only those at the corpus root. The notebook mirrors the corpus tree under `.glossa/notes/<document path>/<file>`, but two places assumed the mirror was a single path segment: (1) `list_note_paths` built its `ls(doc)` filter from `canonical_document_path`, which returns the host-OS separator (backslash on Windows), and compared it with `starts_with` against listed paths that `walk_notes` had normalized to `/` — so for a document like `work/spec.docx` the filter never matched and `ls(doc)` silently returned nothing; (2) `resolve_note_by_path` split the notebook path at the *first* `/` and required that segment to carry a file extension, so a nested document's mirror (`work/spec.docx`) was truncated to `work` and `read`/`del` failed. Both are fixed: the mirror is normalized to `/` on write and in the list filter, and the document/file boundary is now resolved against the index (the longest `/`-prefix that is an indexed document is the mirror, the remainder is the note file). This also supports nested note filenames. Root-only corpora were unaffected, which is why it went unnoticed.
- **Section node ids unified to the 1-based ordinal**: structural `Section` nodes are now keyed by their ordinal (`<path>#<n>` — the same `#n` that `read`/`grep`/`search` show), not by the chunk's heading text. Previously `build_section` derived the id from `chunk.location`, which is a heading breadcrumb for most chunks but a bare number for others, so section ids were a mix of `foo.docx#18` and `foo.docx#**Contents**`. Meanwhile `resolve_section_ref` and `neighbors` always address a section as `<path>#<ordinal>`; when a chunk's stored id was heading-based, an agent's `graph_upsert` edge anchored to `<path>#n` resolved to a section id that did not exist as a node, its endpoint type came back empty, and ontology validation rejected the edge with a misleading `relation '…' endpoints ->X not allowed`. Now every section is `<path>#<ordinal>`, so `path#n` endpoints resolve whether or not the chunk has a heading (PDF page refs like `spec.pdf#4` included). The heading is preserved as the node **label** and as the hierarchy breadcrumb (`CHILD`/`PARENT` still built from `location`), so `glossary`/`search`/`node_ref` are unaffected. Existing corpora need a `reindex` to migrate their section ids.
- **Document edge endpoints normalize path separators**: an edge endpoint that names a document (e.g. `DEPENDS_ON` doc→doc) now resolves through the same path normalization as section refs (`canonical_document_path`), so an agent's `docs/foo.docx` matches the Document node stored with the host separator (`docs\foo.docx`). Previously `resolve_endpoint_label` did an exact-id `get_node` only; a forward-slash document path missed and fell through to the fuzzy label match, silently grabbing a token-overlapping reasoning node. Labels that legitimately contain a slash (e.g. a reference `ISO/TR 10013`) are unaffected — they resolve to no indexed document and fall through to label matching.

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

[Unreleased]: https://github.com/metalmon/glossa/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/metalmon/glossa/compare/v0.2.7...v0.3.0
[0.1.1]: https://github.com/metalmon/glossa/compare/v1.2.0...v0.1.1
[1.2.0]: https://github.com/metalmon/glossa/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/metalmon/glossa/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/metalmon/glossa/releases/tag/v1.0.0
