# glossa — roadmap and backlog

Status as of **2026-07-02**. Version **1.2.0** (tag `v1.2.0`; `master` may be one commit ahead with eval test fixes).

For what ships today, see [README.md](../README.md) and [architecture.md](architecture.md). This file tracks performance notes, technical debt, and direction.

Legend used below: **Shipped** = in a release today; **Partial** = exists but incomplete vs the goal; **Open** = not built.

---

## Shipped in v1.0

- **Extraction:** md (heading-scoped), Office (office_oxide), PDF (oxidize-pdf, per-page `p.N`), images (filename label), text/json/yaml/xml/html/csv/source via streaming; gitignore-aware walk; per-file skip on errors.
- **Search:** BM25 ranked search (multilingual stemming), ripgrep-style `grep`, path `glob`, optional raw `--scan`.
- **Graph:** SQLite store, provenance-stamped nodes/edges, configurable `ontology.toml` with `id_prefix`, structural layer on index.
- **Derived layer:** `graph generalize` — closure, SIMILAR, communities, centrality; debounced auto-generalize on editor MCP after index changes.
- **MCP:** 15 tools, profiles `reader` | `editor` | `full`, stdio + **streamable-http**, `/health` `/ready` `/metrics`; background `ensure_fresh` on read tools.
- **CLI:** `kb search|grep|glob|read|index|reindex|graph …|mcp` — scripting-first, not a TUI.
- **Graph UX:** `graph_stats`, SIMILAR + COMMUNITY in `neighbors`, formatted `graph_upsert` responses (Written / Merged / REJECTED).

## Shipped in v1.1.0

- **Eval harness:** `kb-eval`, `kb-train enrich`, TensorZero backend, TZ episode export, initial GEPA (search + read micro-tasks).
- **Dev pipeline:** `justfile` recipes; Windows-friendly eval tooling.

## Shipped in v1.2.0

- **Quad GEPA:** optimize prod `answer_hotpot` prompt against search, grep, glob, and read via TensorZero micro-functions + `gepa_reflect`; Pareto parent selection and full-val final pick. **Graph tools (`glossary`, `neighbors`) not in the scoring loop yet** — see Eval harness.
- **export-tz:** four jsonl streams (`search`, `grep`, `glob`, `read`); synthetic grep/glob rows when episodes lack those tool calls; `TrainCase.source` gold join when present.
- **Eval harness:** TensorZero `rebuild_corpus_each_question=false`; glossa-train JSON without mandatory `context`; tagged eval runs (`just eval … run-tag`); `case_id` tags on TensorZero episodes.
- **Prod prompt:** refined retrieval fallback in `answer_hotpot/system.minijinja` (glob → scoped search/grep → read).
- **justfile:** Windows `.exe` for `kb-eval` / `kb-train`; `gepa-reset` / `eval-reset`; default judge in `just eval`.
- **Docs:** [eval-and-training.md](eval-and-training.md) playbook, [CHANGELOG.md](../CHANGELOG.md).

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
| HTML / CSV / text streaming | **Partial** | Basic `html`, `csv`/`tsv` (100 rows/chunk), encoding sniff + binary skip in `text` |
| Image files (png, …) | **Partial** | Filename/folder label chunk; vision at `read` time for embedded office images, not scanned PDF pages |
| Image-only / scanned PDFs | **Partial** | One `(no-text)` filename chunk when no text layer — not per-page, no OCR |
| Indexing UX | **Partial** | `+ path` per file on reindex; no bar/counters/ETA |
| Format sniffing (content, not extension) | **Open** | Routing by suffix; mislabeled `.doc`/RTF etc. hit wrong parser |
| Table fidelity (xlsx/docx/pdf) | **Open** | Office → markdown tables break on multi-line cells; PDF is flat layout text |
| Vision for image-only pages | **Open** | Render/read page images like office embeds in `read` |
| Structured JSON chunks | **Open** | `.json` indexed as plain text windows |
| Heading-aware HTML | **Open** | Tag strip + line windower only |
| Row-level CSV | **Open** | 100 rows/chunk today |
| rtf, epub, eml/msg | **Open** | No extractors |

### Graph

| Item | Status | Notes |
|------|--------|-------|
| Ontology strict validation | **Partial** | `validate_edge` / entity checks at upsert; generic error strings |
| `resolve` / `delete_by_source` | **Partial** | Label index for resolve; `delete_by_source` scans `source_path` O(n) |
| Support ontology overlay | **Shipped** | [eval/ontology-support.toml](../eval/ontology-support.toml) — Symptom/Cause/Task spine, strict mode |
| Crash atomicity per file | **Open** | Chunk graph writes autocommit; no one-txn-per-file |
| Glossary `--expand` | **Open** | Term/co-occurrence layer not built; `CO_OCCURS` declared, no lexical indexer |
| Induction/deduction ontology | **Open** | Environment/Heuristic/INDICATES/APPLIES_TO; dual build vs answer agents |
| Tailored ontology error messages | **Open** | e.g. explain Task → CAUSED_BY → Cause is invalid |

### Constraint graph (CSP)

| Item | Status | Notes |
|------|--------|-------|
| CSP solver in `kb` | **Open** | Not in main binary; eval TZ config has forward-looking `constraint_validate` / tool stubs only |
| `constraint_solve` MCP tool | **Open** | Planned — see Track C |

**Planned behavior:** agent models constraints via **`graph_upsert`** (`Field` → `CONSTRAINED_BY` → Range/Enum/Regex/…); **`constraint_solve`** reads that subgraph only (no table extraction from the index).

| Mode | What it does |
|------|----------------|
| **validate** | given values — do they satisfy all constraints? |
| **infer** | what values are still allowed per field? |
| **check** | is the constraint model itself consistent? |

**Open work:** ship solver + constraint ontology in `kb`, cross-field formulas/conditionals, operator CLI, standards mini-corpus eval, solver scaling for large enums.

### MCP and product

| Item | Status | Notes |
|------|--------|-------|
| MCP server + 15 tools | **Shipped** | stdio + streamable-http, profiles, traces, auto-generalize |
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
| GEPA graph micro-tasks (`glossary`, `neighbors`) | **Open** | Extend prompt optimization to graph-first retrieval: export episodes → jsonl, TZ micro-functions, scored like search/read (symptom → chain hit, neighbors → related case / gold chunk). Needed so GEPA tunes the prod prompt's graph protocol, not only flat retrieval. |
| `--no-graph` control arm | **Shipped** | `kb-eval run --no-graph`, MCP `--no-graph` |
| Gold join / `case_id` | **Partial** | TZ sets `case_id`; export joins by id or question; OpenAI backend has no tags; enrich sets `case_id` |
| Whole-run timeout | **Partial** | `kb-eval run --timeout-secs`; not per-round in tool loop |
| Fullwiki benchmark run | **Open** | Prep exists; no logged fullwiki EM/F1/Recall@k series |
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
| GEPA over `glossary` + `neighbors` | **Open** | Quad GEPA covers flat tools only; add graph micro-tasks so optimized prompt learns when/how to use reasoning chains and COMMUNITY/SIMILAR hops |
| Fixed domain mini-corpus + regression CI | **Open** |
| Domain skills / ontology overlays from patterns | **Open** |

### Track C — standards and constraint validation

Normative corpora (ISO, IEC, internal standards, datasheets): retrieval → constraint graph → deterministic validate/infer.

| Milestone | Status |
|-----------|--------|
| CSP solver + `constraint_solve` in `kb` | **Open** |
| Constraint ontology overlay | **Open** |
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
