# glossa — roadmap and backlog

Status as of **2026-06-28**. Version **0.1.0** on `master`.

For what ships today, see [README.md](../README.md) and [architecture.md](architecture.md). This file tracks performance notes, technical debt, and direction.

## Shipped in v0.1

- **Extraction:** md, Office (office_oxide), PDF (oxidize-pdf, per-page `p.N`), text/json/yaml/xml/html/csv/source; streaming pipeline; gitignore-aware walk.
- **Search:** BM25 ranked search (RU/EN stemming), ripgrep-style `grep`, path `glob`, optional raw `--scan`.
- **Graph:** SQLite store, provenance-stamped nodes/edges, configurable `ontology.toml` with `id_prefix`, structural layer on index.
- **Derived layer:** `graph generalize` — closure, SIMILAR, communities, centrality; MCP maintenance loop on editor profiles.
- **MCP:** 15 tools, profiles `reader` | `editor` | `full`, stdio + **streamable-http**, `/health` `/ready` `/metrics`.
- **Graph UX:** `graph_stats`, COMMUNITY neighbors, formatted `graph_upsert` responses (Written / Merged / REJECTED).
- **Eval:** `kb-eval`, `kb-train enrich`, TensorZero integration, GEPA select optimization; release `justfile` recipes.

## Performance

- **Large corpora:** indexing and unindexed regex `--scan` are heavy. Opportunities: parallel traversal, fewer syscalls, mmap reads, persistent file list (see [fff](https://github.com/dmtrKovalenko/fff) for traversal ideas).
- **PDF extraction:** parallel indexing blocked by process-global panic-hook usage in PDF path — remove or guard before multi-threaded extract.

## Technical backlog

### Retrieval and extraction

- **Table fidelity:** multi-line cells break markdown tables in xlsx/docx; PDF tables are flat text only — column reconstruction is high value, hard.
- **Image-only PDF pages:** filename-indexed when no text; vision-read of rendered page images (like embedded office images in `read`).
- **Extractors:** structured JSON chunks, heading-aware HTML, row-level CSV, rtf, epub, eml/msg.
- **Indexing UX:** per-file progress on slow bases.

### Graph

- **Crash atomicity:** single SQLite transaction per file's graph writes.
- **Glossary expansion (`--expand`):** needs Term/co-occurrence layer (not built).
- **Induction/deduction ontology:** richer reasoning types (Environment, Term, Heuristic; INDICATES, APPLIES_TO, …) and dual agents (build_graph vs answer).
- **Resolve/delete_by_source:** secondary indexes (O(n) scans today).
- **Ontology-aware edge errors:** clearer messages when endpoint types violate relation rules (e.g. Task → CAUSED_BY → Cause).

### MCP and product

- **Parallel indexing** behind feature flag once PDF hook is fixed.
- **Trigram grep accelerator** (research done; not integrated).
- **Layer-2 term glossary** for query expansion.

### Eval harness

- Per-round wall-clock budget in OpenAI backend tool loop.
- Expose `MAX_ROUNDS` / read truncation as CLI flags.
- **Fullwiki** HotpotQA (hard retrieval regime) — not yet run.

## Product tracks

### Track A — public benchmarks

Measure the **engine** (agent + tools) on standard QA sets (HotpotQA, 2WikiMultihopQA, MuSiQue). Official EM/F1 against gold answers. Multihop A/B: graph off vs on.

Caveat: English Wikipedia text does not stress office/PDF, Russian, or offline deployment — expect dense-embedding RAG to win on pure semantic paraphrase; glossa's bet is multihop + graph + offline.

### Track B — domain refinement

Curate domain Q/A with gold source spans; retrieval via span match; answer via LLM judge; groundedness/citation checks. Traces in the eval harness (MCP client), not the server. Patterns → domain skills and ontology overlays.

## Ordering

1. OSS docs and stable v0.1 operator path (this release).
2. Track A fullwiki + graph A/B tooling.
3. Track B domain corpus + regression CI on a fixed mini-set.

## Principles

- Pure Rust, offline, single binary on shipping targets.
- File-first: delete `.glossa/`, re-index — corpus files are authoritative.
- Domain rules in `ontology.toml`, not hardcoded in Rust.
- Profiles gate tool visibility, not data access or freshness.

See [benchmarks.md](benchmarks.md) for eval numbers and [eval-and-training.md](eval-and-training.md) for the dev pipeline.
