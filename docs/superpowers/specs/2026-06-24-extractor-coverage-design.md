# Extractor coverage expansion — design

**Status:** approved (brainstorm 2026-06-24)
**Goal:** stop silently dropping readable files. Add a catch-all UTF-8/legacy-text extractor with
real encoding detection, plus dedicated CSV/TSV and HTML extractors. Keep glossa pure-Rust / offline /
C-free.

## Problem

The extractor registry (`walk::extractors()`) dispatches by file extension; a file whose extension
matches no extractor is **silently skipped**. Today only md/markdown, pdf, and office
(docx/doc/xlsx/xls/pptx/ppt) are indexed. `.txt`, `.json`, `.csv`, `.html`, `.yaml`, `.xml`, `.log`,
source code, and any unknown-but-readable file are lost — a violation of the File-First promise
"never drop a readable file". Additionally, the markdown path uses `String::from_utf8_lossy`, which
mangles non-UTF-8 (e.g. Windows-1251 Russian) files.

## Scope (v1)

In: generic catch-all text extractor **with encoding detection**, CSV/TSV extractor, HTML extractor,
a shared decode helper (also adopted by the markdown extractor), a per-file size guard, and a
dispatch refactor. Out (backlog, see end): structured JSON, heading-aware HTML, HTML meta-charset,
CSV via the `csv` crate with `col: val` rendering, rtf/epub/eml, streaming for files over the cap.

## Architecture

### Dispatch (refactor `walk.rs` + `extract.rs`)

`collect_chunks` (indexing) and `read::read_region` both duplicate "pick extractor by extension, else
skip". Extract one shared helper:

```
extract_path(path, bytes) -> anyhow::Result<Vec<Chunk>>:
  1. specific extractor whose file_types() contains the lowercased extension
     (md / office / pdf / csv / html)  -> ex.extract(path, bytes)
  2. else  text::decode(bytes):
        Some(utf8_text) -> text::chunk_text(path, &utf8_text, ext_or("txt"))
        None (binary)   -> Ok(vec![])   // not readable -> skip, no error
```

Both call sites use it; specific extractors always win over the fallback (priority by extension).
This also removes the current duplicated dispatch loops.

### Encoding + binary sniff (`extract::text::decode`) — the core

```
decode(bytes) -> Option<String>:
  1. BOM: UTF-8 BOM -> strip + decode UTF-8;
          UTF-16 LE/BE BOM -> decode via encoding_rs.
  2. else strict UTF-8 (encoding_rs, no replacement) -> if valid, use it.
  3. else chardetng detects the charset (Firefox's detector) -> encoding_rs decodes it
     (covers Windows-1251, KOI8-R, ISO-8859-*, etc.).
  4. binary sniff on the decoded string: a NUL char present, OR the share of
     U+FFFD replacement + non-tab/newline control chars exceeds 10% of chars
     -> return None (treat as binary, skip).
  5. else Some(text).
```

New deps: **`chardetng`** + **`encoding_rs`**, both pure-Rust. After adding, verify
`cargo tree -p glossa -i cc` is empty (C-free invariant). CSV and HTML are hand-rolled — no extra deps.

The markdown extractor switches from `from_utf8_lossy` to `text::decode(bytes).unwrap_or_default()`
so Windows-1251 `.md` files are read correctly (a NUL-containing "markdown" file decoding to None
yields no chunks, which is correct — it was binary).

### Extractors

- **catch-all text** (`extract::text::TextFallback`, no fixed `file_types` — reached only via the
  fallback branch). `file_type` of each chunk = the file's extension (or `"txt"` if none).
  `chunk_text(path, text, file_type)`: window the text so large files stay searchable and chunks stay
  a sensible retrieval/snippet unit — **a new chunk every 100 lines OR 4000 chars, whichever comes
  first**. `location` = `""` for a single-window file, `part.1`, `part.2`, … (1-based) when it splits.
  Covers txt/json/yaml/xml/toml/ini/log/source/unknown.

- **CSV/TSV** (`extract::csv_tsv`, `file_types = ["csv", "tsv"]`). Decode via `text::decode`. Treat the
  first line as the header. Emit **header + 100 data rows per chunk** (the header line is repeated at
  the top of each chunk for context); `location` = `rows A-B` (1-based data-row range). Rows are kept
  as their raw text lines (TSV split on tab is irrelevant — we index the line text). A huge table
  therefore produces many chunks and is indexed **in full**, never truncated. Proper quoted-field
  parsing (`csv` crate, `col: val`) is backlog.

- **HTML** (`extract::html`, `file_types = ["html", "htm"]`). Decode via `text::decode`. A hand-rolled
  stripper: drop the contents of `<script>` and `<style>` elements, remove all `<...>` tags, decode
  the common entities (`&amp; &lt; &gt; &nbsp; &quot; &#39;`), collapse runs of blank lines, then feed
  the result through `chunk_text`. No `scraper`/`html5ever` dependency. Heading-aware sectioning
  (h1–h6) and `<meta charset>` honoring are backlog.

### Size guard (huge knowledge bases)

Knowledge bases legitimately contain very large files (big CSV/Excel exports, 700-page PDFs, long
logs). The design must index them **fully via chunking, never silently truncate**. The only hard
limit is an OOM backstop: in `collect_chunks`, before `fs::read`, check the entry size and **skip
files larger than 256 MiB with an `eprintln!` notice** (no silent cap). 700-page PDFs and typical
big tables are far below this. Streaming extraction for files over the cap (read line-windows without
loading the whole file) is backlog. The cap applies to all extractors uniformly.

## Components / files

- `src/extract.rs` — add `pub mod text; pub mod csv_tsv; pub mod html;` and `pub fn extract_path(...)`.
- `src/extract/text.rs` — `decode(&[u8]) -> Option<String>`, `chunk_text(path, &str, file_type) -> Vec<Chunk>`, `TextFallback`.
- `src/extract/csv_tsv.rs` — `CsvTsvExtractor`.
- `src/extract/html.rs` — `HtmlExtractor` + `strip_html(&str) -> String`.
- `src/extract/markdown.rs` — swap `from_utf8_lossy` for `text::decode`.
- `src/walk.rs` — register csv/html in `extractors()`; add the 256 MiB guard; route the no-match case through `extract_path`'s fallback.
- `src/read.rs` — use `extract_path` so `read` of a `.txt`/`.csv`/`.html` works.
- `Cargo.toml` — add `chardetng`, `encoding_rs`.

## Error handling

- Unreadable/oversized files: logged via `eprintln!`, indexing continues (existing pattern).
- Binary files: `decode` returns `None` → no chunks, no error (correct — nothing to index).
- A decode that yields empty string → no chunks.

## Testing (TDD)

- `decode`: valid UTF-8; UTF-8 BOM; UTF-16 LE BOM; UTF-16 BE BOM; **Windows-1251 Russian bytes → correct
  Cyrillic text**; PNG header / bytes containing NUL → `None`; mostly-control-char blob → `None`.
- `chunk_text`: small file → 1 chunk, `location == ""`; file > window → multiple `part.N` chunks
  covering all text.
- `csv_tsv`: header repeated per chunk; row grouping + `rows A-B` location; `.tsv` handled; a >100-row
  file → multiple chunks, all rows present.
- `html`: tags stripped; `<script>`/`<style>` contents removed; entities decoded; resulting text chunked.
- `walk` integration (via `index_dir` or `collect_chunks`): a dir with `.txt`, `.json`, `.rs`, and a
  `.png` → first three indexed, `.png` skipped; oversized file skipped with notice.
- `read_region`: reading a `.txt` returns its text via the fallback.
- C-free: documented check `cargo tree -p glossa -i cc` is empty (CI/manual).

## Backlog (deferred)

Structured JSON (`key: value` per line); heading-aware HTML (h1–h6 → sections) and `<meta charset>`;
CSV via the `csv` crate with proper quoted fields and `col: val` rendering; rtf, epub (zip+html),
email (eml/msg); streaming extraction for files over the 256 MiB cap; `<title>`-aware HTML location.
