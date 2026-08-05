# glossa — roadmap and backlog

Status as of **2026-08-06**. Version **0.2.7** (tag `v0.2.7`).

For what ships today, see [README.md](../README.md) and [architecture.md](architecture.md). This file tracks performance notes, technical debt, and direction.

For the reasoning graph’s inference-method direction (abduction / deduction / induction and beyond), see [graph-reasoning-directions.md](graph-reasoning-directions.md).

Legend used below: **Shipped** = in a release today; **Partial** = exists but incomplete vs the goal; **Open** = not built.

> **Post-v0.2.7, on `master` (committed, pending push/release):** baked ontology presets + `kb ontology` CLI, mandatory grounding (`requires_grounding`), valid-time Phase 1, graph doctor (ungrounded/stale/incomplete), and the offline HTML graph explorer. These are marked **Shipped** below but are not yet in a tagged release.

---

## Shipped

- **Extraction:** md (heading-scoped), Office (office_oxide), PDF (pdf_oxide, per-page `p.N`), images (filename label), text/json/yaml/xml/html/csv/source via streaming; gitignore-aware walk; per-file skip on errors; HTML `<img>` tag extraction (v0.2.5).
- **Search:** BM25 ranked search (multilingual stemming), ripgrep-style `grep`, path `glob`, optional raw `--scan`.
- **Graph:** SQLite store, provenance-stamped nodes/edges, configurable `ontology.toml` with `id_prefix`, structural layer on index.
- **Derived layer:** `graph generalize` — closure, SIMILAR, communities, centrality; debounced auto-generalize on editor MCP after index changes.
- **MCP:** 22 tools (24 with `--features constraint`), profiles `reader` | `editor` | `full`, stdio + **streamable-http**, `/health` `/ready` `/metrics`; background `ensure_fresh` on read tools.
- **CLI:** `kb search|grep|glob|read|index|graph …|mcp` — scripting-first, not a TUI.
- **Source-file delivery:** `get_source_file` returns the original document for a hit; docx is auto-converted to PDF for viewing (`src/convert/docx_pdf.rs`).
- **Graph UX:** `graph_stats`, SIMILAR + COMMUNITY in `related`, structural `neighbors` (typed 1-hop edges) and `path` (shortest connection between two nodes), formatted `graph_upsert` responses (Written / Merged / REJECTED).
- **Ontology presets** *(post-v0.2.7)*: 26 baked task ontologies (Tier 1 conformance / Tier 2 operational), `kb ontology list/show/init/suggest` + `kb index --ontology <name>`; thin reasoning skeletons — one grounded terminal per preset, no Evidence node.
- **Mandatory grounding + valid-time** *(post-v0.2.7)*: `requires_grounding` / `requires_validity` enforced at `graph_upsert` and advertised by `get_ontology`; valid-time Phase 1 — `--as-of` / `as_of`, `SUPERSEDES`, per-node status (current/future/expired).
- **Graph doctor** *(post-v0.2.7)*: `kb graph doctor` + MCP `graph_doctor` — ungrounded / stale (source `file_sig` drift) / incomplete report and targeted prune; `generalize` is derived-layer only; inline `⚠ stale` marker on reads.
- **HTML graph explorer** *(post-v0.2.7)*: `kb graph dump -f html` — one self-contained offline interactive explorer (search → focused local view, light/dark, mobile-friendly).
- **Eval harness:** `kb-eval`, `kb-train enrich`, TensorZero backend, TZ episode export, initial GEPA (search + read micro-tasks).
- **Dev pipeline:** `justfile` recipes; Windows-friendly eval tooling.
- **Quad GEPA:** optimize prod `answer_hotpot` prompt against search, grep, glob, and read via TensorZero micro-functions + `gepa_reflect`; Pareto parent selection and full-val final pick.
- **export-tz:** four jsonl streams (`search`, `grep`, `glob`, `read`); synthetic grep/glob rows when episodes lack those tool calls; `TrainCase.source` gold join when present.
- **Constraint graph (CSP):** solver in `constraint/src/solver.rs` (validate/infer/check, ~1218 lines), `constraint_solve` MCP tool — behind `--features constraint`.
- **Image output opt-in:** off by default, enabled with `--vision` (`GLOSSA_VISION`); all images served as JPEG (embedded PNGs re-encoded, PDF-embedded/raster JPEGs passed through), so a figure-heavy page no longer overflows the stdio frame. `-N` / `--noimage` kept as a deprecated no-op (v0.2.7; the original `--noimage` toggle shipped in v0.2.5).

See [eval-and-training.md](eval-and-training.md) for the dev pipeline and [benchmarks.md](benchmarks.md) for HotpotQA numbers.

---

## Performance

### Shipped

- **`grep` trigram prefilter:** indexed `body_trigrams` field (char 3-grams); Cox-style plan from the regex, Tantivy candidate lookup, then line-by-line confirmation with `regex`. Falls back to full chunk scan when the pattern has no selective trigrams. Details: [architecture.md](architecture.md).
- **Tantivy mmap:** search index segments mapped for read-heavy queries.
- **Streaming index walk:** gitignore-aware single-file pipeline; extract errors logged and skipped.
- **PDF resilience:** malformed PDFs caught with `catch_unwind`; concurrent single-file extract covered by tests.

### Open

- **Large corpora:** indexing is sequential (one file at a time); `--scan` without index is heavy. Opportunities: parallel traversal/extract, fewer syscalls, mmap reads of source files, persistent file list between runs (see [fff](https://github.com/dmtrKovalenko/fff)).
- **Parallel indexing:** not enabled — needs safe multi-threaded extract end-to-end (PDF/Office temp dirs, writer locking).

---

## Technical backlog

### Retrieval and extraction

| Item | Status | Notes |
|------|--------|-------|
| Markdown heading-scoped chunks | **Shipped** | `chunk_markdown` / `A > B` locations |
| HTML / CSV / text streaming | **Partial** | Basic `html`, `csv`/`tsv` (100 rows/chunk), encoding sniff + binary skip in `text`; HTML image extraction (`<img>` tags) shipped in v0.2.5 |
| Image files (png, …) | **Partial** | Index: filename/folder label chunk only. `read` serves standalone image files as raw bytes to vision (`read.rs`) and can rasterize any PDF page — behind `--vision` |
| Image-only / scanned PDFs | **Partial** | Parseable scans now get one empty stub **per physical page** (`pad_pdf_page_stubs`, `pdf.rs`); the `(no-text)` filename fallback fires only for *unparseable* PDFs. Still no OCR |
| Indexing UX | **Partial** | `+ path` per file on index; no bar/counters/ETA |
| Format sniffing (content, not extension) | **Open** | Routing by suffix; mislabeled `.doc`/RTF etc. hit wrong parser |
| Table fidelity (xlsx/docx/pdf) | **Partial** | Office **done** — `expand_merged_tables` densifies merged col/row spans, multi-line cells collapsed, emitted as GFM (`office_table.rs` / `office_chunk.rs`). PDF: **flat layout-text only** — the structured table detector was removed (unreliable: mis-classified multi-column prose as tables, grids low-quality; A/B on kb-gost). Tables come through as flattened rows |
| Vision for image-only pages | **Partial** | `render_pdf_page` rasterizes any page to JPEG@200DPI and serves embedded scans byte-for-byte, wired into `read` via the `page_image` arg behind `--vision` (`read.rs` / `tools.rs`). **Manual** — not auto-triggered for detected image-only pages |
| Structured JSON chunks | **Open** | `.json` indexed as plain text windows |
| Heading-aware HTML | **Open** | Tag strip + line windower only |
| Row-level CSV | **Open** | 100 rows/chunk today |
| rtf, epub, eml/msg | **Open** | No extractors |

### Graph

| Item | Status | Notes |
|------|--------|-------|
| Ontology strict validation | **Partial** | `validate_edge` / entity checks at upsert; **edge** errors now tailored (point to `get_ontology → relations.X` + `edge_validation_hint`), **node** errors still generic |
| `resolve` / `delete_by_source` | **Partial** | Label index for resolve; `delete_by_source` scans `source_path` O(n) |
| Support ontology overlay | **Shipped** | [eval/ontology-support.toml](../eval/ontology-support.toml) — Symptom/Cause/Task spine, strict mode |
| Baked ontology presets + `kb ontology` CLI | **Shipped** *(post-v0.2.7)* | 26 presets (Tier 1/2), list/show/init/suggest, `index --ontology`; thin skeletons, one grounded terminal; [ontology-presets.md](ontology-presets.md) |
| Mandatory grounding (`requires_grounding`) | **Shipped** *(post-v0.2.7)* | Enforced at `graph_upsert`; `get_ontology` advertises it; `generalize` / `graph doctor` surface ungrounded + `--prune-ungrounded` |
| Graph doctor: ungrounded / stale / incomplete | **Shipped** *(post-v0.2.7)* | `kb graph doctor` + MCP `graph_doctor`; `stale` = source `file_sig` drift; prune flags CLI-only; inline `⚠ stale` |
| Valid-time (Phase 1, as-of) | **Shipped** *(post-v0.2.7)* | `node_validity` side table, `requires_validity`, `--as-of`/`as_of`, `SUPERSEDES`, status; see Temporality below |
| Offline HTML graph explorer | **Shipped** *(post-v0.2.7)* | `kb graph dump -f html`, self-contained; infinite-scroll search, dark theme, temporal display |
| Crash atomicity per file | **Open** | Chunk graph writes autocommit; no one-txn-per-file |
| Cross-process `graph_upsert` lock | **Open** | Advisory `.glossa/graph.lock` (fs4), like `generalize.lock` / `notebook.lock`; today only in-process Mutex on SQLite |
| Glossary `--expand` | **Open** | Term/co-occurrence layer not built; `CO_OCCURS` declared, no lexical indexer |
| Induction/deduction ontology | **Open** | Environment/Heuristic/INDICATES/APPLIES_TO; dual build vs answer agents; see [graph-reasoning-directions.md](graph-reasoning-directions.md) |
| Tailored ontology error messages | **Partial** | `edge_validation_hint()` in `ontology.rs` covers constraint fields; domain-specific messages (e.g. Task → CAUSED_BY → Cause) not yet added |

### Temporality (valid-time → bitemporal)

Give reasoning facts a time dimension so the conformance engine can answer
*"was this valid / known on date X"* and trace every verdict to the interval it
held in. Built phase by phase to a professional (bitemporal, SQL:2011-informed)
level. Design lives in `docs/superpowers/specs/` (per-phase spec → plan).

**Grounding parallel:** temporality is grounded in the ontology the same way
`MENTIONS` grounding is — a per-entity `requires_validity` flag (mirror of
`requires_grounding`) makes `graph_upsert` reject a node of that type that
carries no valid-time. Presets mark the types that must be timed
(`hr-compliance` Record, `data-privacy` DataAsset, `contract` Obligation,
`reg-change` Requirement, `certification` Test).

Core rule throughout: **world time ≠ document time** — `valid_*` is when a fact
holds in the world; `provenance` (source doc / section, `created_at`) is when it
was recorded.

| Phase | Scope | Status |
|-------|-------|--------|
| **1 — Valid-time core (as-of)** | authored `node_validity` side table (1:1, survives `generalize`), `valid_from`/`valid_to` on `graph_upsert`, `requires_validity` enforcement, `--as-of <date>` filter on `glossary`/`neighbors`/`ls`/`dump`/`read`, per-node status (current/future/expired), `SUPERSEDES` relation, ISO-8601 storage keeping the raw expression | **Shipped** (Phase 1) |
| **2 — Transaction-time / bitemporal (as-at)** | `known_from` (seeded from `created_at`) / `known_to`, logical retraction for agent temporal nodes, `--as-at <T>`, combined bitemporal query | **Open** |
| **3 — Temporal reasoning & hygiene** | Allen interval relations; `generalize` surfacing (expired-but-referenced, coverage gaps/overlaps, supersession-chain consistency); temporal conditions in `constraint_solve` (valid-as-of, one-valid-at-a-time, no-gap coverage) | **Open** |
| **4 — Uncertainty & edges** | EDTF (`~`/`?`/unspecified) parsed to bounds (raw stored since Phase 1), strict/lenient query modes, edge-level validity | **Open** |

### Constraint graph (CSP)

| Item | Status | Notes |
|------|--------|-------|
| CSP solver in `kb` | **Shipped** | `constraint/src/solver.rs` (validate/infer/check, ~1218 lines); behind `--features constraint` |
| `constraint_solve` MCP tool | **Shipped** | Registered when built with `--features constraint`; reads graph subgraph → solver Problem |
| Structured table tools (`table_add_row`, `table_get`, …) | **Open** | Agents build `.csp` tables as free-form `note` TSV today, which is error-prone on wide/joint tables. Give models first-class ops — add/edit/delete a row, read a table back structured, and pull a table out of a source document — so complex tables are manipulated as data, not hand-formatted text. At minimum: build these tools and add eval coverage that they behave. |

**Planned behavior:** agent models constraints via **`graph_upsert`** (`Field` → `CONSTRAINED_BY` → Range/Enum/Regex/…); **`constraint_solve`** reads that subgraph only (no table extraction from the index).

| Mode | What it does |
|------|----------------|
| **validate** | given values — do they satisfy all constraints? |
| **infer** | what values are still allowed per field? |
| **check** | is the constraint model itself consistent? |

**Open work:** cross-field formulas/conditionals, operator CLI, standards mini-corpus eval, solver scaling for large enums.

### Decision-tree walker (guarded-branch traversal)

Deterministic traversal of a **decision tree** stored in the graph — for logic where
the next step depends on the previous answer (triage/diagnosis, applicability
determination, approval/escalation routing). Complements flat `constraint_solve`
(which is single-shot and forbids `Conditional` nesting) by moving the multi-hop choice
**into code**: a fixed-signature walker `decision_walk(root, facts) → {outcome |
need_field | path+citations}` walks the tree, evaluating each fork's guard by reusing the
existing per-node evaluator (`build_constraint` + solver `validate`). Lazy/resumable —
stops and names the missing fact instead of guessing. Aligned with the thin-skeleton
principle: per-node guard stays flat; depth comes from edge recursion + deterministic
traversal, not a complex graph fed to a weak model. Design (with preset sketch):
`docs/superpowers/specs/2026-08-06-decision-tree-walker-design.md`.

| Item | Status | Notes |
|------|--------|-------|
| `decision-tree` preset (guarded-branch) | **Open** | New Tier-1 preset; bumps preset count 26→27 |
| `decision_walk` MCP tool + walk fn | **Open** | `walk`/`signature`/`check`; reuses constraint adapter; cycle-guard, bounded depth |
| Match policy (`strict` / `ordered`) + `check` linter | **Open** | Exclusive-with-static-check vs first-match+default; `check` flags overlap/gap/unreachable/type-mismatch |
| Per-hop vs terminal-only grounding | **Open** | Both via `requires_grounding` flag; walker surfaces whatever `MENTIONS` are on the path |
| Enrich authoring of trees | **Open** | Hardest part — extracting branching from a regulation; SOP/prompt pattern + `check` as guardrail |

### MCP and product

| Item | Status | Notes |
|------|--------|-------|
| MCP server + tools | **Shipped** | **22** `#[tool]` fns default, **24** with `--features constraint` (added since "15": graph_doctor / graph_stats / graph_update / graph_generalize / get_source_file / …); stdio + streamable-http, profiles, traces, auto-generalize |
| `kb` CLI (search/grep/glob/read/graph) | **Partial** | Operator commands exist; no TUI/REPL, completion, or rich progress |
| Install/deploy scripts | **Partial** | GitHub Releases + [install.md](install.md) + [deploy/](../deploy/) ansible/service — not apt/Homebrew/winget |
| Human-friendly operator UX | **Open** | Progress bars, shell completion, browse/maintenance TUI |
| Parallel indexing | **Open** | See Performance → Open |
| Layer-2 term glossary | **Open** | Query expansion from co-occurrence / Term layer |
| Package managers | **Open** | apt, Homebrew, winget publishing |

### Eval harness

| Item | Status | Notes |
|------|--------|-------|
| Hotpot distractor runs | **Shipped** | Logged in [benchmarks.md](benchmarks.md) (50q slices) |
| `prep-fullwiki` | **Shipped** | CLI + shard builder in `kb-eval prep-fullwiki` |
| `export-tz` quad jsonl + GEPA | **Shipped** | v1.2.0 — search, grep, glob, read micro-tasks only |
| Constraint GEPA (5 pools: discover / materialize / compile / coverage / validate) | **Partial** | 5-pool loop, TZ functions, and apply path shipped; graph_build / graph_stats / constraint_solve scorers still use text-recall proxies. See [constraint-gepa.md](constraint-gepa.md). |
| GEPA graph micro-tasks (`glossary`, `related`) | **Open** | Extend prompt optimization to graph-first retrieval: export episodes → jsonl, TZ micro-functions, scored like search/read (symptom → chain hit, related → alternate case / gold chunk). Needed so GEPA tunes the prod prompt's graph protocol, not only flat retrieval. |
| `--no-graph` control arm | **Shipped** | `kb-eval run --no-graph`, MCP `--no-graph` |
| Gold join / `case_id` | **Partial** | TZ sets `case_id`; export joins by id or question; OpenAI backend has no tags; enrich sets `case_id` |
| Whole-run timeout | **Partial** | `kb-eval run --timeout-secs`; not per-round in tool loop |
| Fullwiki benchmark run | **Open** | Prep + run path + fullwiki recall scoring all wired (`prep`/`run`/`score`); no logged EM/F1/Recall@k series yet |
| Graph on/off A/B series | **Open** | `--no-graph` exists; no formal logged comparison on Hotpot |
| Per-round wall-clock budget | **Open** | OpenAI backend tool loop |
| `MAX_ROUNDS` / read truncation CLI | **Open** | Hardcoded at 50 in backends |
| 2WikiMultihopQA / MuSiQue | **Open** | Not wired |

---

## Product tracks

### Track A — public benchmarks

Measure the **engine** on standard QA sets (HotpotQA, 2WikiMultihopQA, MuSiQue): EM/F1, retrieval Recall@k, graph on/off.

| Milestone | Status |
|-----------|--------|
| Hotpot distractor 50q (Qwen vs Claude reader) | **Done** — EM ~0.68–0.80; see [benchmarks.md](benchmarks.md) |
| Larger N (200–500) stable estimate | **Open** |
| fullwiki Recall@k | **Open** — prep shipped, run not logged |
| Graph A/B (graph off vs on) | **Partial** — `--no-graph` shipped; benchmark series not logged |
| 2Wiki / MuSiQue | **Open** |

Caveat: English Wikipedia does not stress office/PDF, legacy encodings, or offline deployment.

### Track B — domain refinement

Curate domain Q/A with gold source spans; retrieval via span match; LLM judge; groundedness checks.

| Milestone | Status |
|-----------|--------|
| `kb-train enrich` + support ontology | **Shipped** |
| glossa-train JSON format + export-tz | **Shipped** |
| GEPA over `glossary` + `related` | **Open** | Quad GEPA covers flat tools only; add graph micro-tasks so optimized prompt learns when/how to use reasoning chains and COMMUNITY/SIMILAR hops |
| Fixed domain mini-corpus + regression CI | **Open** |
| Domain skills / ontology overlays from patterns | **Open** |

### Track C — standards and constraint validation

Normative corpora (ISO, IEC, internal standards, datasheets): retrieval → constraint graph → deterministic validate/infer.

| Milestone | Status |
|-----------|--------|
| CSP solver + `constraint_solve` in `kb` | **Shipped** — v0.2.5, behind `--features constraint` |
| Constraint ontology overlay | **Shipped** *(post-v0.2.7)* — `compliance` / `qa-inspection` / `certification` presets ship the Field→CONSTRAINED_BY→Range/Enum/… constraint shape |
| Standards mini-corpus eval | **Open** |

Complements Track B (answer quality) with **deterministic compliance** — e.g. "are these voltage and temperature ratings within the datasheet limits?"

---

## Ordering

1. Track B — domain mini-corpus + regression CI.
2. Track A — fullwiki run + graph on/off A/B series (+ larger N).
3. Track C — CSP in `kb`.
4. Extraction quality — format sniffing, table fidelity.

---

## Principles

- Pure Rust, offline, single binary on shipping targets (`kb` on release tags; eval tooling from source).
- File-first: delete `.glossa/`, re-index — corpus files are authoritative.
- Domain rules in `ontology.toml`, not hardcoded in Rust.
- Profiles gate tool visibility, not data access or freshness.

See [benchmarks.md](benchmarks.md) for eval numbers and [eval-and-training.md](eval-and-training.md) for the dev pipeline.
