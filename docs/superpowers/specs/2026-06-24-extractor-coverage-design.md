# Extractor coverage expansion — design

**Status:** approved (brainstorm 2026-06-24)
**Goal:** stop silently dropping readable files. Add a catch-all text extractor with real encoding
detection, plus dedicated CSV/TSV and HTML extractors, on a **fully streaming** extract→index
pipeline so arbitrarily large files (huge tables, logs) index in constant memory with **no size
limit**. Keep glossa pure-Rust / offline / C-free.

## Problem

The extractor registry (`walk::extractors()`) dispatches by file extension; a file whose extension
matches no extractor is **silently skipped**. Today only md/markdown, pdf, and office
(docx/doc/xlsx/xls/pptx/ppt) are indexed. `.txt`, `.json`, `.csv`, `.html`, `.yaml`, `.xml`, `.log`,
source code, and any unknown-but-readable file are lost — a violation of File-First "never drop a
readable file". Markdown also uses `String::from_utf8_lossy`, which mangles non-UTF-8 (e.g.
Windows-1251 Russian) files. And `collect_chunks` materializes **all** chunks of all files into one
`Vec<Chunk>` before indexing, so a single huge file would blow memory.

## Scope (v1)

In: streaming extract→index pipeline (chunk sink); catch-all text extractor with encoding detection;
CSV/TSV and HTML extractors; a shared decode/streaming helper (also adopted by markdown); dispatch
refactor; no size limit. Out (backlog, see end): structured JSON, heading-aware HTML, HTML
meta-charset, CSV via the `csv` crate with `col: val`, rtf/epub/eml.

## Architecture

### Streaming pipeline (the core change)

A chunk **sink** replaces the materialized `Vec<Chunk>` between extraction and indexing:

```
type ChunkSink<'a> = &'a mut dyn FnMut(Chunk);

walk::stream_chunks(root, glob, respect_ignore, sink):
    for each file under root (gitignore-aware, skip .glossa):
        extract::extract_file(path, sink)        // pushes that file's chunks into sink

index::index_dir consumes via a sink that adds each Chunk to the index as it arrives
    -> extra memory is O(one chunk), independent of file size. No cap.
```

`collect_chunks(root, …) -> Vec<Chunk>` is kept as a thin wrapper (a sink that pushes into a Vec) for
`read` and tests. `index_dir` uses the streaming sink, not `collect_chunks`.

### Per-file dispatch (`extract::extract_file(path, sink)`)

```
ext = lowercased extension
1. specific binary/doc extractor (md/markdown, office…, pdf): these formats need the whole file and
   are inherently bounded -> read bytes, run Extractor::extract, push each chunk to sink.
   (markdown switches from from_utf8_lossy to text::decode.)
2. csv/tsv | html | (anything else = fallback text): STREAM from the path (below), pushing chunks to
   the sink as windows fill. A file the sniff deems binary yields nothing (skip, no error).
```

Specific extractors win by extension; everything else streams as text.

### Encoding detection + binary sniff (`extract::text`)

Detect once from a prefix, then stream-decode the rest:

```
prefix = first 64 KiB of the file
1. BOM in prefix: UTF-8 BOM -> UTF-8;  UTF-16 LE/BE BOM -> that encoding.
2. else if prefix is valid strict UTF-8 -> UTF-8.
3. else chardetng over the prefix -> detected Encoding (Windows-1251, KOI8-R, ISO-8859-*, …).
binary sniff on the prefix: a NUL byte, OR >10% non-tab/newline control bytes -> file is binary,
    return None (skip the whole file).
otherwise -> Some(encoding).

Then decode the full file incrementally with encoding_rs::Decoder for that encoding, feeding decoded
text to the active extractor's windowing, which emits chunks into the sink. Raw bytes and decoded
text are never both held whole.
```

New deps: **`chardetng`** + **`encoding_rs`**, both pure-Rust. After adding, verify
`cargo tree -p glossa -i cc` is empty (C-free invariant). CSV and HTML are hand-rolled — no extra deps.

### Extractors (all streaming, emit into the sink)

- **catch-all text** (`file_type` = extension or `"txt"`): window the decoded stream — a new chunk
  every **100 lines OR 4000 chars, whichever first**. `location` = `""` for a single-window file,
  else `part.1`, `part.2`, … (1-based). Covers txt/json/yaml/xml/toml/ini/log/source/unknown.

- **CSV/TSV** (`file_types = ["csv", "tsv"]`): first line = header; emit **header + 100 data rows per
  chunk** (header repeated atop each chunk for context); `location` = `rows A-B` (1-based data rows).
  Rows kept as raw line text. A huge table → many chunks, indexed **in full**, never truncated.
  Quoted-field parsing via the `csv` crate (`col: val`) is backlog.

- **HTML** (`file_types = ["html", "htm"]`): a streaming state machine over the decoded char stream —
  drop `<script>`/`<style>` contents, remove `<…>` tags, decode common entities
  (`&amp; &lt; &gt; &nbsp; &quot; &#39;`), collapse blank-line runs — feeding the same windowing as
  text. No `scraper`/`html5ever`. Heading-aware sectioning and `<meta charset>` are backlog.

## Components / files

- `src/extract.rs` — `pub mod text; pub mod csv_tsv; pub mod html;` and `pub fn extract_file(path, sink)`.
- `src/extract/text.rs` — prefix detection (`detect(&[u8]) -> Option<&'static Encoding>`, None=binary),
  `stream_text(path, file_type, sink)`, window logic.
- `src/extract/csv_tsv.rs` — `stream_csv_tsv(path, sink)` (+ a thin `Extractor` adapter is unnecessary; dispatch calls it directly).
- `src/extract/html.rs` — `stream_html(path, sink)` + `strip_html` state machine.
- `src/extract/markdown.rs` — swap `from_utf8_lossy` for `text::decode`.
- `src/walk.rs` — `stream_chunks(root, glob, respect, sink)`; `collect_chunks` becomes a Vec-collecting
  wrapper; dispatch through `extract_file`. Remove any size cap.
- `src/index/store.rs` — `index_dir` adds chunks via the streaming sink, not a collected Vec.
- `src/read.rs` — read a single file via `extract_file` with a collecting sink (location-filtered).
- `Cargo.toml` — add `chardetng`, `encoding_rs`.

## Error handling

- Unreadable files: logged via `eprintln!`, walk continues (existing pattern).
- Binary files: prefix sniff returns None → no chunks, no error.
- Empty/whitespace-only decoded content → no chunks.

## Testing (TDD)

- `detect`/decode: valid UTF-8; UTF-8 BOM; UTF-16 LE BOM; UTF-16 BE BOM; **Windows-1251 Russian bytes
  → correct Cyrillic**; PNG header / NUL bytes → None; mostly-control blob → None.
- `stream_text`: small file → 1 chunk `location==""`; file past one window → multiple `part.N` chunks
  whose concatenation covers all input.
- `csv_tsv`: header repeated per chunk; `rows A-B` ranges; `.tsv`; a >100-row file → multiple chunks,
  every row present.
- `html`: tags stripped; `<script>`/`<style>` removed; entities decoded; output chunked.
- streaming pipeline: `stream_chunks` invokes the sink once per chunk (count == produced chunks); a
  generated multi-window file is fully covered; the sink sees chunks incrementally (no full Vec
  required by the producer).
- `walk` integration (`index_dir`/`collect_chunks`): dir with `.txt`, `.json`, `.rs`, `.png` → first
  three indexed, `.png` skipped.
- `read_region`: reading a `.txt`/`.csv` returns text via the streaming fallback.
- C-free: documented check `cargo tree -p glossa -i cc` empty (CI/manual).

## Backlog (deferred)

Structured JSON (`key: value` per line); heading-aware HTML (h1–h6 → sections) and `<meta charset>`;
CSV via the `csv` crate with proper quoted fields and `col: val`; rtf, epub (zip+html), email
(eml/msg); `<title>`-aware HTML location; parallel streaming extraction (blocked on the PdfExtractor
panic-hook race noted in ROADMAP).
