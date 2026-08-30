# Changelog

All notable changes to glossa are documented here. Release tags ship the **`kb`** binary only; `kb-eval` / `kb-train` are built from source.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.4.0] — 2026-08-30

### Added

- **Multi-API transport.** Per-endpoint `api = "openai" | "anthropic" | "openai_responses"` (including native Anthropic Messages) routes through one unified `ChatTransport`, so a stage can call whichever provider's native API fits without a shim.
- **Rate-limit resilience + fallback chains.** Opt-in per-endpoint retry/backoff and throttling (`rpm`, `max_inflight`, `retry`, `backoff_ms`), plus an ordered `[[<stage>.fallback]]` chain so a stage degrades to a backup endpoint instead of failing outright.
- **Per-endpoint temperature.** `Option<f64>` per endpoint; omitted falls back to the provider default, and `KB_EVAL_TEMP` overrides it at run time.
- **Parallel kbx pipeline workers.** `[tuning] jobs_build/reason/train/distil/eval` runs cases concurrently across the pipeline, with a race-free per-case trace.
- **`user_sim` dialogue gate.** An opt-in patient simulated-user turn deflects a non-answer back into the reader loop instead of accepting it, for both eval and train.
- **Fine-tuning dataset export.** Collects SFT and DPO pairs (Unsloth-ready) from the reasoning graph; see [docs/finetuning-datasets.md](docs/finetuning-datasets.md).
- **Anti-loop retrieval signals.** Neutral plateau/repeat/streak markers surface per-session in the MCP server and in the eval/train reader, with novelty-gated result trimming to curb context bloat; plateaus are also paired into judge-labeled DPO examples during export.

### Changed

- **Evidence-grounded judge.** The judge now grades an answer against the retrieved source evidence, not only against the gold string.
- **`kbx train` progress.** A visible iteration-based progress bar and honest metric names replace the earlier opaque counters.

## [0.3.4] — 2026-08-21

### Fixed

- **Silent cross-type node conflation in `graph_upsert`.** A node's identity is now its `(label, type)`, not its label alone. Two nodes that share a label but differ in type (which the ontology already promises are distinct) stayed distinct instead of the second silently collapsing onto the first's id — a general silent-corruption hole in the core graph, and the deterministic cause of the constraint compiler's failures. Edge-endpoint resolution is type-first: when a relation fixes an endpoint's type, the edge lands on the correct-typed node.
- **Ambiguous edge endpoints under a non-strict ontology.** When a relation fixes neither endpoint type and a bare label matches two differently-typed nodes, `graph_upsert` no longer silently picks one. It drops the edge with an actionable message naming the candidate type-qualified ids, and accepts an explicit `Type:label` qualifier (e.g. `Symptom:cache`) — or a node id — to address the endpoint precisely.
- **Recovery for graphs built before this release:** the fix is forward-only — no data is migrated. A graph already conflated under the old behavior self-heals on a corpus **re-index** (`kb index`), which regenerates a clean graph from the source documents.

### Added

- **`--source-file` flag** (env `GLOSSA_SOURCE_FILE`). `get_source_file` — delivery of the original source file behind a citation — is now **off by default** and opt-in, mirroring how image output is opt-in via `--vision`. Many clients can't consume the returned file resource, so the tool is withheld unless a client that uses it opts in.

### Changed

- **`graph_query` renamed to `sql`.** The read-only SQL-over-the-reasoning-graph tool is now `sql` (one short word); behavior is unchanged. Call it with an empty query to get the schema.
- **`sql` is now available to the `reader` profile.** It is read-only and useful for rankings/extremes, so answer agents get it (previously editor/full only).
- **Reader profile decluttered.** `neighbors`, `related`, `resolve`, and `constraint_solve` are withheld from `reader` (editor/full keep them). A weak answer model chooses better from fewer tools, and the value of those low-level navigation tools is already surfaced in `glossary`'s composed section.
- **`glossary`'s composed section now ranks by graph connectivity, not just lexical overlap.** It fuses the lexical alias-join with a Personalized-PageRank walk over the graph (parameter-free reciprocal-rank fusion), so a multi-hop answer that shares no words with the question — which a plain lexical rank can't float — now surfaces. `glossary` gained an optional `query` (the full question) to rank the neighborhood by what the question actually needs.

## [0.3.3] — 2026-08-15

### Added

- **Bearer-token auth for the network endpoint.** `kb mcp --transport streamable-http --auth-token <TOKEN>` (or env `GLOSSA_MCP_TOKEN`) requires `Authorization: Bearer <TOKEN>` on every `/mcp` request — anything else gets `401`. `/health`, `/ready` and `/metrics` stay open for probes. Unset → unauthenticated (the loopback default); ignored for `--transport stdio`. The token compare is constant-time. An interim integration-key guard ahead of full OIDC/IdP.
- **HTTP request metrics.** `/metrics` now also exposes request-rate/latency: `glossa_http_requests_total`, `glossa_http_responses_total{class}`, `glossa_http_requests_in_flight`, a `glossa_http_request_duration_seconds` histogram, and `glossa_mcp_auth_rejected_total`, next to the existing index/graph gauges.
- **JSON logs.** `GLOSSA_LOG_FORMAT=json` emits one JSON object per log line (for a SIEM / log pipeline); default stays human-readable. Both go to stderr.
- **Security audit events.** Dedicated structured events on the `glossa::audit` tracing target (JSON-filterable): auth rejections (`bearer_reject`, with source IP) and every write/admin tool (`graph_upsert`, `graph_delete`, `graph_build`, `note`, `del`). Schema: `category / action / outcome / source / object`.
- **Opt-in idle-session timeout.** `kb mcp … --session-idle-secs <N>` (env `GLOSSA_MCP_SESSION_IDLE_SECS`, default `0` = disabled) refuses a streamable-http session idle longer than `N` seconds with `404` on its next request, so a spec-compliant client re-initializes (a cheap handshake; the KB holds no per-session state).
- **`kb --version`.**
- **Default `.ignore` whitelist.** `kb index` seeds a `.ignore` (gitignore-whitelist idiom) on a corpus that has none, so a first index only reads types glossa can actually extract — installers, archives and temp files are no longer slurped as text. Editable; never clobbers an existing `.ignore`/`.gitignore`.
- **Index error summary.** `kb index` prints `N skipped(errors)` in its stats line and an end-of-run list of files that failed to extract, so corrupt files aren't lost in terminal scroll.
- **Root reporting.** Every command prints the resolved `root:` and warns on the nested-corpus / deleted-`.glossa`-walks-up traps (a deleted corpus `.glossa` otherwise silently reindexes into an ancestor).
- **`docs/security-and-operations.md`** — hardening, observability, and an enterprise-readiness scorecard for the streamable-http server.

### Changed

- **Upgraded the MCP SDK `rmcp` 1.8 → 3.1** (a one-line source change: `Content` → `ContentBlock`). Unblocks the MCP Apps (`ext apps`) direction; all tests and the streamable-http handshake/tool-call path verified.
- **Reindex reads the tree once, not up to three times** (`ensure_fresh` → `index_dir` no longer re-walks) — restores earlier-version speed on large corpora.
- **Images are not read at index time** — they are indexed by filename only, so a large/scanned image is never slurped into memory just to be dropped.
- **Heading-only markdown is indexed by its title** instead of producing no searchable chunk.

### Fixed

- **Nested/deleted `.glossa` root-resolution traps** that let the CLI and MCP server drift onto different indexes (split-brain).
- **A corrupt or unreadable file no longer aborts the whole index** (e.g. a `.doc` with a bad CFB header) — it is logged, skipped, and reported in the error summary.
- **Content finalized after a mid-copy freshen** is now picked up (the dir-mtime gate held a just-written dir for one more pass).

## [0.3.2] — 2026-08-14

### Added

- **OpenDocument extraction (`.odt` / `.ods` / `.odp`).** glossa now reads OpenDocument text, spreadsheets, and presentations directly — a built-in `content.xml` parser builds the same `DocumentIR` the Office path uses, so odt/ods/odp get the same heading-scoped chunking and GFM tables (with merged-cell densify, `number-columns/rows-repeated` expansion and used-range clamping) as docx/xlsx/pptx. Headings are recognized both as structural `<text:h>` and as styled paragraphs (`Heading_20_N`, as LibreOffice/converted docs emit); ODS becomes section-per-sheet, ODP section-per-slide.
- **Chart data extraction (Office + ODF).** Charts are no longer invisible: their underlying data (series, categories, values, title) is extracted into a searchable GFM table — one chunk per chart. OOXML reads the `charts/chartN.xml` cache; ODF reads the chart object's embedded local `<table:table>`, or, when the chart references sheet cells instead, resolves the `cell-range-address` ranges against the document's own sheet (A1 addressing with merged-cell / repeat handling, quoted sheet names, and bounded allocation). Charts are extracted as **data**, not rendered images.
- **Embedded images delivered to vision agents.** With `--vision`, `read(page_image: true)` now returns embedded images from more sources: zip-based Office/ODF media (`word|xl|ppt/media/` and ODF `Pictures/`) and raster pictures embedded in legacy binary `.doc` / `.xls`. Legacy OLE image extraction is wrapped in a panic guard so a malformed file degrades to "no images" instead of aborting the read.

### Notes

- **Out of scope (documented):** charts are data-not-rendered (no bundled renderer; use a PDF source for a rendered visual), legacy binary `.doc/.xls/.ppt` charts are not extracted, and `.ppt` embedded images / vector metafiles (EMF/WMF/PICT) are not delivered. See `docs/architecture.md` → "Known extraction limitations". A design direction for reading engineering drawings / CAD (three layers + native render + MCP Apps) is in `docs/cad-drawing-reading-design.md`.

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
