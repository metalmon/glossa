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
| docx, pptx | `zip` (deflate-only) + `quick-xml` | unzip → parse `word/document.xml` / slide `<a:t>`; sections by heading. (Recon: `office_oxide` v0.1.x too immature to be a core dep.) |
| doc, ppt (legacy OLE2) | `office_oxide` (experimental) or deferred | text only, coarse; no mature pure-Rust legacy parser — may slip to phase 2 if unstable |
| xlsx, xls, ods | `calamine` | per-sheet; snippet carries `Sheet!Row` (reads legacy + modern) |
| pdf | `pdf-extract` | v1: whole-doc text (`extract_text_by_pages` exists); page-level + images via `lopdf` in phase 2 |
| encoding | `encoding_rs` | Cyrillic / legacy encodings (windows-1251, KOI8-R) |

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
- **Query ergonomics** (influenced by ugrep, all native to tantivy):
  - boolean operators (AND/OR/NOT) and phrase queries,
  - fuzzy / approximate matching (Levenshtein term queries) for typo tolerance,
  - optional interactive query mode in the CLI (`kb search -i`).

```toml
[search]
auto_detect = true
languages = []            # empty = all supported; or a whitelist like ["ru","en"]
default_language = "ru"   # fallback when detection is low-confidence
expand_with_glossary = true
```

The index stores chunk text for snippet generation; the canonical content remains the file.

### 4.4 Glossary graph

> Full node/edge ontology, provenance model, and the domain-schema mechanism: see §11.

- Storage: **SQLite** (`rusqlite`, bundled — keeps single-binary).
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
- `search(query, limit?, file_type?, path_filter?, expand?)` → `[{path, location, snippet, score}]`
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
  glossary.db       # SQLite: nodes, edges, file manifest (mtime/hash), index status
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
| `zip` | 8.6 | docx/xlsx/pptx containers — **deflate-only** (`default-features=false`) |
| `quick-xml` | — | docx/pptx XML text extraction |
| `encoding_rs` | 0.8 | Cyrillic/legacy encodings |
| `pdf-extract` | 0.10 | PDF text (v1); `lopdf` for page-level + images (phase 2) |
| `rmcp` | 1.8 | official MCP SDK — **supports image content blocks**; stdio + HTTP/SSE |
| `rusqlite` | — | glossary graph (bundled SQLite) |
| `regex` | — | ripgrep-compatible query engine (see §4.3 — pending syntax decision) |
| `tokio` | — | async runtime |

Single static binary; the server makes no network calls. (Optional embedding providers in
phase 4 may add network — opt-in.) `office_oxide` intentionally **not** a core dep (immature).

## 10. Open questions / risks
- Mixed-language documents detect by dominant language (acceptable for File-First).
- Auto co-occurrence glossary can be noisy; keep weights conservative and keep expansion
  toggleable (`expand`).
- Legacy OLE2 parsing quality (.doc/.ppt especially) varies; verify `office_oxide` coverage,
  keep failures non-fatal.
- `office_oxide` does not expose per-element locations; heading-based sectioning is our
  granularity for docx/doc/pptx/ppt.
- MCP image content blocks confirmed supported by `rmcp` 1.8 (resolved).
- **Legacy OLE2 (.doc/.ppt)** has no mature pure-Rust parser — `office_oxide` is experimental;
  legacy support may slip to phase 2 (modern docx/pptx/xls/xlsx are solid via zip+quick-xml/calamine).
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
