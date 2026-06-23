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
| docx, doc, pptx, ppt | `office_oxide` | markdown/text → sections by heading (legacy = coarser, text only) |
| xlsx, xls | `calamine` | per-sheet; snippet carries `Sheet!Row` (calamine reads both) |
| pdf | `pdf-extract` | v1: whole-document text (no page metadata) |
| encoding | `encoding_rs` | Cyrillic / legacy encodings |

Failure isolation: a corrupt / encrypted / scanned / unsupported file is **skipped with a
recorded reason** in the index status; indexing never aborts.

### 4.2 Chunking
Structural chunks: heading section (md/docx/doc/pptx/ppt), sheet (xlsx/xls), page (pdf, phase 2).
Each chunk becomes one index document with `{path, location, file_type, text, offsets, project}`.

### 4.3 Index (search)
- **tantivy** — BM25, fast, single embedded index directory; highlighted snippets.
- **Multilingual stemming** via a custom tokenizer `MultiLangStemmer`:
  - `whatlang` detects the language of each chunk/text.
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
- Storage: **SQLite** (`rusqlite`, bundled — keeps single-binary).
- **v1 (auto, lightweight):** nodes = documents, headings, auto-extracted terms/entities;
  edges = `appears_in`, `co_occurs`. Built during indexing. Used for:
  - navigation (term → documents/sections), and
  - **query expansion** (blend related terms into the BM25 query — `search` option, default on).
- Honest scope note: for unstructured prose, an auto co-occurrence glossary is a **navigation
  aid + mild recall boost**, not a killer feature. We do not oversell it in v1.
- **Phase 2/3 (agent-built knowledge graph):** because the product is agent-native, the
  connected agent can extract entities/relations (GraphRAG-style) by reading documents via
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
- (phase 2/3) `graph(nodes, edges)` → agent-built knowledge graph upsert

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
  filesystem watcher hardening; agent-built knowledge-graph (`graph`) tool.
- **Phase 3 (optional):** semantic / hybrid (pluggable embedding provider — local multilingual
  model or external API — with BM25+vector reranking); deeper GraphRAG-style retrieval.

## 9. Dependencies
`tantivy`, `office_oxide`, `calamine`, `pdf-extract`, `whatlang`, `rusqlite` (bundled),
`zip`, `encoding_rs`, an MCP server crate, async runtime (`tokio`). Single binary; the server
makes no network calls. (Optional embedding providers in phase 3 may add network — opt-in.)

## 10. Open questions / risks
- Mixed-language documents detect by dominant language (acceptable for File-First).
- Auto co-occurrence glossary can be noisy; keep weights conservative and keep expansion
  toggleable (`expand`).
- Legacy OLE2 parsing quality (.doc/.ppt especially) varies; verify `office_oxide` coverage,
  keep failures non-fatal.
- `office_oxide` does not expose per-element locations; heading-based sectioning is our
  granularity for docx/doc/pptx/ppt.
- Confirm the MCP server crate's support for returning image content blocks; otherwise a thin
  custom MCP layer.
- Repository name (`glossa`) is provisional.
