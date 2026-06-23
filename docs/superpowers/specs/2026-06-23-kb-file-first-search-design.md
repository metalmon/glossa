# glossa — File-First Knowledge-Base Search (design)

- **Date:** 2026-06-23
- **Status:** Draft (awaiting review)
- **Working name:** `glossa` (provisional, easy to rename)
- **License/positioning:** Open-source on the author's GitHub (metalmon); also used for commercial engagements.

> This is a design/spec document. No implementation has started. The terminal step
> of brainstorming is to hand this to the planning phase (`writing-plans`), not to write code.

---

## 1. Problem & vision

Provide **File-First search over a knowledge base** of mixed documents (Markdown,
DOCX/DOC, PDF, XLSX/XLS, PPTX/PPT) where:

- **Files are the source of truth.** The index is only a pointer. Every search result
  is `path + location (heading / sheet!row / page) + snippet`.
- **An AI agent does the deep reading.** The tool returns a chunk + source metadata; the
  connected agent then opens the source via `read` (including images, returned as
  MCP image content blocks for the agent's vision) to answer in full fidelity.
- **The graph is a glossary**, not a code graph: terms / headings / entities → where they
  appear, for navigation and query expansion.
- **Fully offline.** No data leaves the machine; no network calls from the server. Strong
  selling point for privacy/compliance-sensitive (e.g. RU enterprise) deployments.

This is deliberately **not** "another RAG server." The defensible angles are: offline +
agent-native File-First + good Russian out of the box + single binary.

### Why build (honest take on existing solutions)

The components and turnkey products exist (Recoll/Xapian, Apache Tika + Solr/ES,
Meilisearch/Typesense/Quickwit, LlamaIndex/Haystack, AnythingLLM/Khoj/Danswer,
Microsoft MarkItDown + markitdown-mcp). "Search over docs" is commodity. We therefore:

- **Reuse, not reinvent**, the hard commoditized parts (document extraction, BM25 index).
- **Invest original work** only in the differentiators: the MCP/agentic File-First layer,
  the offline story, Russian/multilingual quality, the optional agent-built knowledge graph,
  and single-binary packaging.

If this were a one-off internal tool, buying-vs-building would favor existing tools. As an
OSS product + commercial base, building is justified **only** as the differentiated package
above.

### Prior art & influences

- **ugrep** (https://ugrep.com/) — a strong **File-First** validator: it searches files
  directly (results are `file:line`, no derived source of truth) and treats its optional
  `ugrep-indexer` as an *accelerator over files* — exactly our "index accelerates, files are
  truth" model. We borrow its **query ergonomics** (boolean AND/OR/NOT, fuzzy/approximate
  matching, an optional interactive CLI mode). We **deliberately diverge** on two points:
  (1) ugrep extracts office/PDF text via *external filters* (pdftotext/pandoc / `ug+`) — we use
  *native Rust crates* for a zero-dependency single binary; (2) ugrep is grep (regex scan, no
  ranking/stemming) — we are index-first with BM25 + multilingual stemming for KB-quality
  ranked retrieval.

## 2. Goals / non-goals

**Goals (v1)**
- Index a directory tree of `.md/.docx/.doc/.xlsx/.xls/.pptx/.ppt` (PDF text in v1, see phasing).
- Lexical full-text search (BM25) with multilingual stemming, auto language detection.
- Return `path + location + highlighted snippet`, ranked.
- `read` returning text + embedded images (modern office formats) as MCP image blocks.
- Lightweight auto glossary graph (headings + co-occurring terms) for navigation + query
  expansion.
- Incremental re-index (mtime + content hash); optional file watch.
- MCP server **and** mirrored CLI. Single self-contained binary.

**Non-goals (v1)**
- Semantic / vector search (architecture stays ready for it; not implemented).
- OCR / image understanding inside the server (delegated to the agent's vision).
- PDF page-level locations and PDF image extraction (phase 2).
- Image extraction from **legacy** OLE2 formats (.doc/.xls/.ppt) — text only in v1 (phase 2+).
- LLM-based entity/relation extraction inside the server (delegated to the agent; phase 2/3).

## 3. Architecture overview

```
                +-------------------- glossa (Rust, single binary) --------------------+
   files  --->  | discover/walk -> extract -> chunk -> index (tantivy)                 |
   (KB dir)     |                                  \-> glossary graph (SQLite)         |
                |                                                                       |
   agent  <---> | MCP server  (search, read, glossary, index, reindex)                 |
   (MCP only)   | CLI         (kb search | read | index)                               |
                +-----------------------------------------------------------------------+
```

The **MCP server is the filesystem boundary**: it runs locally with disk access; the agent
needs no filesystem access and receives text + images strictly through MCP results.

## 4. Components

### 4.1 Extraction (native crates, no WASM)
The indexer walks the KB directory and reads files directly:

| Format | Library | Notes / location granularity |
|---|---|---|
| md | native parser | split by headings; location = heading path |
| docx, doc, pptx, ppt | `office_oxide` (behind `Extractor` trait) | text/markdown for all four **including legacy OLE2**; sections by heading. Proven in our Zeroclaw office-tools PR |
| xlsx, xls, ods | `calamine` | per-sheet + cell coords; snippet carries `Sheet!Row` (legacy + modern) |
| pdf | `pdf-extract` | v1: whole-doc text (`extract_text_by_pages` exists); page-level + images via `lopdf` in phase 2 |
| images | `zip` (deflate-only) | pull embedded media (`word|xl|ppt/media/*`) from modern office files (office_oxide drops images) |
| encoding | `encoding_rs` | Cyrillic / legacy encodings (windows-1251, KOI8-R) |

All extractors sit behind a single `Extractor` trait, so any backend (notably the young
`office_oxide`) is swappable per format without touching chunking/index — the maturity risk is
contained, not designed-in.

Failure isolation: a corrupt / encrypted / scanned / unsupported file is **skipped with a
recorded reason** in the index status; indexing never aborts.

### 4.2 Chunking
Structural chunks: heading section (md/docx/doc/pptx/ppt), sheet (xlsx/xls), page (pdf, phase 2).
Each chunk becomes one index document with `{path, location, file_type, text, offsets, project}`.

### 4.3 Index (search)
- **tantivy** — BM25, fast, single embedded index directory; highlighted snippets.
- **Multilingual stemming** via a custom tokenizer `MultiLangStemmer`:
  - `lingua` (feature-gated to the configured languages, for offline short-text accuracy)
    detects the language of each chunk/text.
  - If the language is in the configured set **and** Snowball-supported (~20 langs: ru, en,
    de, fr, es, it, pt, nl, fi, sv, no, da, hu, ro, tr, el, ar, ta, …) → apply that stemmer.
  - Otherwise → fallback: Unicode tokenization + lowercase, no stemming.
  - Same detection path applied at query time.
- **Default behavior:** auto-detect across **all** supported languages; configurable.
- **Query syntax = ripgrep (`rg`), 1:1.** Match dialect and core flags are ripgrep's,
  implemented with the **same `regex` crate ripgrep uses** — the real dialect, not an
  approximation. Tool/CLI descriptions can simply say *"use ripgrep syntax"* and even weak
  models get it right.
  - Surface: regex pattern; `-i`/smart-case, `-w` (word), `-F` (fixed string), `-e` (multiple
    patterns), `-g`/`--glob` (path filter), `-t`/`--type` (file type), context `-A`/`-B`/`-C`.
  - **Default semantics = ripgrep** (literal/regex match over document text) → File-First,
    exact, predictable.
- **Index as accelerator.** tantivy (+ a trigram term index) narrows candidate chunks so we
  don't scan the whole corpus; the `regex` engine then confirms matches — the "index
  accelerates, files are truth" model (cf. ugrep-indexer). The index also stores chunk text
  for snippets; canonical content stays in the file.
- **KB modes are explicit flags** (opt-in, so default rg behavior stays pure):
  - `--rank` → BM25 relevance ranking instead of file order,
  - `--stem` → multilingual stemming (`MultiLangStemmer`: `lingua` detection → Snowball),
  - `--expand` → glossary query expansion (related terms).

```toml
[search]
# rg semantics are the default; these tune the opt-in KB modes
auto_detect = true
languages = []            # empty = all supported; or a whitelist like ["ru","en"]
default_language = "ru"   # fallback when detection is low-confidence
expand_with_glossary = false   # enable per-query with --expand
```

### 4.4 Glossary graph

> Full node/edge ontology, provenance model, and the domain-schema mechanism: see §11.
> Build model, File-First invariants, temporality, traversal, and storage choice: see §12.

- Storage: **`redb`** — a pure-Rust embedded store (see §12; supersedes the earlier SQLite mention to keep the build C-free).
- **v1 (auto, lightweight):** nodes = documents, headings, auto-extracted terms/entities;
  edges = `appears_in`, `co_occurs`. Built during indexing. Used for:
  - navigation (term → documents/sections), and
  - **query expansion** (blend related terms into the BM25 query — `search` option, default on).
- Honest scope note: for unstructured prose, an auto co-occurrence glossary is a **navigation
  aid + mild recall boost**, not a killer feature. We do not oversell it in v1.
- **Phase 2/3 (agent-built knowledge graph):** because the product is agent-native, the
  connected agent can extract entities/relations (GraphRAG-style) **and cross-document links**
  (references / supersedes / relates-to / contradicts) by reading documents via
  `read` and **writing them back** through the `graph` MCP tool. This yields graph-of-text
  quality **without bundling an LLM and without server-side network** — the in-the-loop agent
  is the extractor. This is the primary differentiator and where commercial value is added.

### 4.5 Images over MCP (no agent filesystem access)
- MCP tool results can carry image content blocks (`type: image`, base64 `data`, `mimeType`).
  A vision-capable agent sees them directly in the tool response.
- `read(path, range?, include_images?)` returns the region's text **plus** image blocks.
- Extraction: modern office formats store media inside the ZIP container (`word/media/*`,
  `xl/media/*`, `ppt/media/*`) — we pull them directly (note: `office_oxide` silently drops
  images, so media is read from the zip ourselves). → **v1**.
- Legacy OLE2 formats (.doc/.xls/.ppt) embed media in compound-binary streams → image
  extraction is **phase 2+**; v1 returns their text only.
- PDF image extraction (XObjects) needs a richer PDF lib (`pdfium`/`lopdf`) → **phase 2**,
  together with PDF page-level locations.
- Safeguards: cap image count/size per response; optional downscale to protect agent context.

### 4.6 Interfaces

**MCP tools** (single-word names)
- `search(query, limit?, ...rg_flags)` → `[{path, location, snippet, score}]` — `query` uses
  **ripgrep syntax**; flags mirror rg (`-i`/`-w`/`-F`/`-g`/`-t`/`-A`/`-B`/`-C`) plus KB flags
  `--rank`/`--stem`/`--expand`
- `read(path, range?, include_images?)` → text/markdown + optional image blocks
- `glossary(term)` → related terms + where it appears
- `index(dir)` / `reindex()` → (incremental: mtime + content hash; optional watch)
- (phase 2) `resolve(name)` → fuzzy/alias lookup over existing entities (for entity resolution)
- (phase 2) `graph(nodes, edges)` → agent-built knowledge graph upsert (validated against `ontology.toml`)

**CLI (mirrors MCP)**
- `kb index <dir>` · `kb search "<query>" [--type pdf] [--limit N]` · `kb read <path> [--range ..]`

### 4.7 Incremental indexing
Track per-file `(path, mtime, size, content_hash)`. On reindex, only changed/added/removed
files are processed. Optional filesystem watcher for live updates.

## 5. Storage layout
```
<kb>/.glossa/
  index/            # tantivy index
  graph.redb        # pure-Rust embedded store: graph nodes + edges (see §12)
  manifest.json     # per-file mtime+size signatures for incremental (index + graph)
  config.toml
  ontology.toml     # domain schema: allowed entity/relation types + props (see §11)
```

## 6. Errors & observability
- Per-file failure isolation with a queryable status (counts, skipped files + reasons).
- Clear, non-fatal messages for encrypted/scanned/oversized/unsupported files.

## 7. Testing
- `cargo test`: per-extractor fixtures (tiny md/docx/doc/xlsx/xls/pptx/ppt/pdf samples).
- Russian stemming tests (e.g. «договор» matches «договоров/договорам»).
- Multilingual detection/fallback tests.
- End-to-end: index → search → assert path/location/snippet.
- Glossary query-expansion test.
- Image extraction test (modern office media → image block).

## 8. Phasing
- **v1:** md/docx/doc/xlsx/xls/pptx/ppt; lexical BM25 + multilingual stemming; lightweight auto
  glossary; modern-office image return; MCP + CLI; incremental index.
- **Phase 2:** PDF page-level locations + PDF image extraction; legacy-format image extraction;
  filesystem watcher hardening; **agent-built knowledge graph** via the `graph` tool — the
  agent reads documents and writes entities/relations **and cross-document links**
  (references / supersedes / relates-to / contradicts), turning the flat corpus into a
  navigable graph of connected documents.
- **Phase 3 — companion skills:** agent playbooks/skills for working with glossa (search →
  read → build-graph → answer workflows), analogous to the `office-documents` skill in
  zeroclaw-skills. Ships as a small skill pack so agents use the tool well out of the box.
- **Phase 4 (optional):** semantic / hybrid (pluggable embedding provider — local multilingual
  model or external API — with BM25+vector reranking); deeper GraphRAG-style retrieval over
  the agent-built graph; **optional RDF/SHACL export** for semantic-web interop (SHACL shapes
  generated from `ontology.toml` — see §11).

## 9. Dependencies

Recommended stack (versions from mid-2026 recon — pin exact patch at integration):

| Crate | Ver | Role |
|---|---|---|
| `tantivy` | 0.26 | embedded BM25 index, tokenizers, fuzzy/phrase/boolean, snippets |
| `rust-stemmers` | 1.2 | Snowball stemmers (18 langs incl. ru/en); used by tantivy |
| `lingua` | 1.8 | offline language detection (feature-gated to enabled langs) |
| `calamine` | 0.35 | xls/xlsx/xlsb/ods extraction (per-sheet/cell) |
| `office_oxide` | 0.1.x (pinned) | text for docx/doc/pptx/ppt incl. **legacy**; behind `Extractor` trait, swappable |
| `zip` | 8.6 | embedded image extraction from office files — **deflate-only** (`default-features=false`) |
| `encoding_rs` | 0.8 | Cyrillic/legacy encodings |
| `pdf-extract` | 0.10 | PDF text (v1); `lopdf` for page-level + images (phase 2) |
| `rmcp` | 1.8 | official MCP SDK — **supports image content blocks**; stdio + HTTP/SSE |
| `redb` | — | pure-Rust embedded store for the graph (NOT rusqlite — bundled SQLite pulls C via cc; see §12) |
| `regex` | — | **the** ripgrep regex engine → 1:1 rg syntax (see §4.3) |
| `tokio` | — | async runtime |

Single static binary; the server makes no network calls. (Optional embedding providers in
phase 4 may add network — opt-in.) `office_oxide` is young (v0.1.x) — used behind the
`Extractor` trait, version-pinned, fork/vendor if needed (it is the only crate covering legacy
.doc/.ppt).

## 10. Open questions / risks
- Mixed-language documents detect by dominant language (acceptable for File-First).
- Auto co-occurrence glossary can be noisy; keep weights conservative and keep expansion
  toggleable (`expand`).
- Legacy OLE2 parsing quality (.doc/.ppt especially) varies; verify `office_oxide` coverage,
  keep failures non-fatal.
- `office_oxide` does not expose per-element locations; heading-based sectioning is our
  granularity for docx/doc/pptx/ppt.
- MCP image content blocks confirmed supported by `rmcp` 1.8 (resolved).
- `office_oxide` is young (v0.1.x) — **mitigation:** behind an `Extractor` trait (swappable per
  format), version-pinned, fork/vendor if needed. It is the only crate covering **legacy**
  .doc/.ppt and is proven in our Zeroclaw office-tools PR, so legacy stays in v1.
- `zip` default features pull in C bzip2/zstd — must build deflate-only to stay pure-Rust.
- `lingua` full models are large — must feature-gate to the enabled languages.
- `pdf-extract` is text-only (no images/scanned/OCR); page-level + images need `lopdf` (phase 2).
- Avoid `pdfium-render`/`mupdf` for PDF (native libs break single-binary; mupdf is AGPL).
- Repository name (`glossa`) is provisional.

## 11. Knowledge-graph ontology (detail)

Layered so search works without the upper layers; **provenance** and **source-anchoring**
apply throughout. Files remain the source of truth — the graph only points at them.

**Cross-cutting rules**
- Every node/edge carries: `origin` ∈ {auto-structural, auto-lexical, agent, curated},
  `confidence` (0–1), `created_by` (agent/run), `created_at`, `evidence[]` (chunk/section refs).
- Nodes/edges anchor to source (`path` + `range`); the graph never stores canonical content.
- Stable IDs across reindex: `Document = hash(path)`, `Section = doc + heading-path`,
  `Term = lemma`, `Entity = canonical_name (+ aliases)`.

**Layer 1 — Structural (v1, auto, deterministic)**
- Nodes: `Document`, `Section` (hierarchical), `Sheet`/`Page`/`Slide`.
- Edges: `CONTAINS`, `NEXT`/`PREV`.

**Layer 2 — Glossary (v1, auto; SKOS semantics)**
- Node: `Term` (lemma, surface_forms[], doc_freq, score).
- Edges: `MENTIONS` (Section→Term, freq/positions), `CO_OCCURS` (Term↔Term, weighted),
  `DEFINED_IN` (Term→Section); SKOS `broader`/`narrower`/`related`.

**Layer 3 — Entities & topics (phase 2, agent-built)**
- Nodes: `Entity` (type from domain schema, canonical_name, aliases[]),
  `Topic`/`Community` (summary).
- Edges: `ABOUT` (Document/Section→Entity/Topic); domain `Entity↔Entity` relations.

**Layer 4 — Cross-document links (phase 2, agent-built)**
- `Document↔Document`: `REFERENCES`, `SUPERSEDES`/`SUPERSEDED_BY`, `RELATES_TO`,
  `CONTRADICTS`, `DUPLICATES`, `VERSION_OF`, `DERIVED_FROM`.

**Schema model — fixed core + domain config**
- Fixed core = layers 1–2 + the provenance fields (always present).
- Entity/relation types are an **open, typed vocabulary** declared per deployment in
  `ontology.toml`. `graph` upserts are validated against it; `strict` mode rejects (or warns
  on) unknown types. Ships with a generic default schema; verticals (legal/medical/eng)
  override it. This is the portability lever across commercial engagements.

**Entity resolution — server-assisted, agent decides**
- The server exposes `resolve(name)` (fuzzy + alias lookup over existing canonical names).
- The agent checks before creating and writes `canonical_name` + `aliases[]`.
- The server never auto-merges (avoids false merges like «Акме Юр» vs «Акме Мед»).

**`ontology.toml` sketch**
```toml
[entities.Person]
props = ["full_name", "role"]

[entities.Organization]
props = ["legal_name", "inn"]

[relations.AUTHORED_BY]
from = ["Document"]
to   = ["Person", "Organization"]

[validation]
strict = true   # reject upserts with types not declared above
```

**Validation — native now, SHACL-mappable, optional SHACL export later**
- v1: a small **native validator** checks every `graph` upsert against `ontology.toml`.
  No RDF stack in the core (no oxigraph) — keeps the lean single-binary, offline ethos.
- The constraint vocabulary is intentionally a **subset that maps 1:1 to SHACL Core**, so
  nothing has to be re-modelled later:
  | `ontology.toml` | SHACL Core |
  |---|---|
  | entity/relation `type` | `sh:class` / `sh:targetClass` |
  | required + cardinality | `sh:minCount` / `sh:maxCount` |
  | property datatype | `sh:datatype` |
  | allowed value set | `sh:in` |
  | `strict = true` | `sh:closed true` |
- **Phase 3 (optional interop):** export the graph as RDF and validate with an embedded/external
  SHACL engine (rudof / pySHACL); SHACL shapes are **generated from `ontology.toml`**. This
  aligns with the SKOS glossary layer and serves users who need semantic-web interoperability —
  without forcing RDF/SHACL into the core.

> Concrete default + legal-vertical `ontology.toml` examples: see
> [`2026-06-23-ontology-examples.md`](./2026-06-23-ontology-examples.md).

## 12. Knowledge graph — build model, invariants, temporality, traversal, storage (M4+)

Binding for Milestones 4–5. The graph's value over the directory tree is **cross-document
links, entities, and term connections that span folders** — a graph, not a tree. We do NOT
model the folder tree (the path already encodes it; search has `path_filter`); only
intra-document structure + terms + entities + relations.

### 12.1 Build model — code vs model, by layer
- **Layers 1–2 (structural + glossary): built by CODE**, deterministically, during `kb index`
  (Document/Section nodes + `CONTAINS`; optionally terms + `MENTIONS`/`CO_OCCURS`). No model.
- **Layers 3–4 (entities + cross-document links): built by the connected AGENT** (an LLM via
  MCP), **not** a model embedded in the server. The Rust server stays 100% code/pure-Rust; it
  provides storage, ontology validation, traversal, and the `graph`/`resolve`/`read` tools the
  agent calls.
- **Weak-model behaviour:** schema-bounded per-document extraction is forgiving (weak models
  cope); cross-document reasoning is hard and unreliable on weak models. **Nothing breaks** if
  the agent is weak — lexical/BM25 search + code-built layers 1–2 work without the semantic
  graph (graceful degradation). High-stakes edges (CONTRADICTS/SUPERSEDES) are confidence-scored
  **candidates with evidence**, never asserted as truth.

### 12.2 File-First invariants (HARD — implementation must enforce)
1. Graph is **derived & disposable, never authoritative** — holds no content, only pointers +
   assertions + confidence. Deleting the graph loses nothing; it rebuilds from files.
2. **No fact without provenance**: every node/edge carries `source_path` + `range` +
   `file_sig` (source signature at extraction) + `origin` + `confidence` + `created_at`. If you
   can't cite a span, it's not in the graph.
3. **File-level incrementality via provenance ownership**: reindex of file F = drop all graph
   elements whose provenance is in F, then re-derive/re-extract F only. Elements into F's
   entities from other files survive. The whole corpus is never re-extracted.
4. **Don't model the catalog**, and don't model what search already answers.
5. **Extraction is on-demand / incremental / opt-in** — not a mandatory full-corpus pass.
6. **Staleness is surfaced, never hidden**: results carry recency + confidence; elements whose
   `file_sig` no longer matches the file are marked STALE (or dropped), never served as fresh.
7. **Auto layer (co-occurrence) stays conservative + toggleable** (noise control).
8. **High-stakes edges need confirmation** (human/strong-model), stored as evidence-bearing
   candidates.
9. **No ungrounded global reasoning**: every synthesized claim must trace to spans (avoids the
   GraphRAG "community-summary drift" failure).

### 12.3 Temporality — two clocks, never conflated
- **Graph freshness vs files**: enforced by `file_sig` (invariant 2/6).
- **World validity**: domain facts like `valid_from`/`valid_until` or `SUPERSEDES` are **edge
  properties**, distinct from graph freshness.

### 12.4 Traversal — bounded primitives, not a query engine
- Provide `neighbors(node, edge_types?, depth)` and `path(from, to, max_depth)` as **bounded
  BFS/DFS in code**, returning paths **with provenance**. No Cypher/SPARQL engine in v1.
- **Multi-hop**: the agent walks these primitives + `read`, **verifying each hop against the
  source span** — File-First defuses the compounding-error problem of pure-graph multi-hop.
  Constrain by depth + a confidence threshold.

### 12.5 Storage — pure-Rust embedded (`redb`), NOT SQLite
- The graph lives in **`redb`** (pure-Rust embedded ACID store) at `<dir>/.glossa/graph.redb`.
- **This supersedes the earlier SQLite/`rusqlite` mention.** `rusqlite`'s `bundled` feature
  compiles SQLite's C via `cc`, reintroducing the C/build dependency we deliberately removed in
  M3 (zstd-sys). `redb` keeps the pure-Rust + single-static-binary + offline property. The store
  is disposable/rebuildable from files (invariant 1).
