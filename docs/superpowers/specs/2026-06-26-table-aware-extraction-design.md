# Table-Aware Document Extraction — Design

**Date:** 2026-06-26
**Status:** Approved (approach confirmed by user)
**Branch:** `feat/pdf-extract-layout`

## Goal

Extract **proper tables** (GitHub-flavored markdown `| a | b |`) from PDF and DOCX/XLSX into the search index, and fix the residual PDF text-quality problems (1-line-per-page, glued words, image-region artifacts) — all pure-Rust, C-free.

## Background

glossa indexes documents by extracting text into per-page/section `Chunk`s (`{doc_path, location, file_type, text}`). The model addresses chunks by `read(path, n)` where `n` maps to `ord` (page number `p.N` for PDF). The PDF extractor was producing one giant line per page with glued words (`548Важно!`) and image garbage; tables came through as unreadable space-runs. This broke grep (whole-page lines = 258K-token blobs) and made tabular data unsearchable.

### Investigation findings (two research passes)

- **DOCX/XLSX** uses a *different* crate, **`office_oxide` v0.1.2** (not oxidize-pdf), via `Document::to_markdown()`. That path **already emits proper GFM pipe tables** — the crate's own tests assert `"| H1 | H2 |"`, `"| --- | --- |"`. Word `w:tbl` and Excel grids both render as tables. **No code change needed** — only a regression-locking test.
- **PDF** uses **`oxidize-pdf` v2.16.6**. The just-committed change (`3a398b4`) switched the extractor to `extract_text_with_options(preserve_layout: true, …)`, fixing spacing/newlines/artifacts — but tables stay as space-aligned text. oxidize-pdf 2.16.6 **does** have real table recognition (ruling-line + alignment/borderless heuristics) exposed via `partition()` → `Element::Table` → `table_to_markdown()`. Each `Element` carries `.page: u32`, so per-page chunking (`p.N`) is preserved.

## Design

### PDF: `src/extract/pdf.rs` — switch to `partition()` per page

Replace the per-page `extract_text_with_options` loop with a `partition()`-based path that renders each page's typed elements (including detected tables) to markdown.

**Core flow** (inside the existing `std::panic::catch_unwind` guard, on the `PdfDocument`):

```rust
use oxidize_pdf::pipeline::{Element, ElementMarkdownExporter, ExportConfig};
use std::collections::BTreeMap;

// 1. Partition into typed elements (each Element has .page: u32 and an
//    Element::Table(t) variant with t.rows: Vec<Vec<String>>).
let elements = doc.partition()?;            // ParseResult<Vec<Element>>

if !elements.is_empty() {
    // 2. Group elements by page, preserving document order within a page.
    let mut by_page: BTreeMap<u32, Vec<Element>> = BTreeMap::new();
    for el in elements {
        by_page.entry(el.page).or_default().push(el);
    }
    // 3. Render each page to markdown; tables become GFM pipe tables.
    let exporter = ElementMarkdownExporter::new(ExportConfig::default());
    for (page, els) in by_page {
        let md = exporter.export(&els);
        if md.trim().is_empty() { continue; }
        out.push(Chunk {
            doc_path: path.to_path_buf(),
            location: format!("p.{}", page_label(page)), // 1-based, see below
            file_type: "pdf".into(),
            text: md,
        });
    }
}
```

**Page numbering (`page_label`):** the current extractor emits 1-based `p.{i+1}`. `partition()`'s `Element.page` must be normalized to the same 1-based convention so `read(path, n)` addressing is unchanged. The implementer **must verify empirically** whether `Element.page` is 0- or 1-based (the single-page `sample.pdf` fixture must produce exactly `p.1`) and apply `+1` iff it is 0-based. Lock this with the fixture test.

**Fallback chain (robustness — indexing must NEVER drop a document's text):**
1. `partition()` returns ≥1 element → per-page markdown chunks (above).
2. Else (no structural elements detected, or `partition()` returns `Err`) → fall back to the **existing `extract_text_with_options` layout path** (the `3a398b4` logic — keep it as the fallback; its `ExtractionOptions` still apply), one chunk per page.
3. Else (no text at all) → the existing **filename-stem** fallback chunk (`location: "(no-text)"`).

The whole body stays inside `catch_unwind`; a panicking PDF is caught and falls through to the filename-stem chunk, exactly as today. `ParseOptions::lenient()` reader unchanged.

**Why keep `3a398b4`:** it is no longer the primary path but remains the layer-2 fallback for PDFs where `partition()` finds no structure but raw text exists. Not wasted.

### DOCX/XLSX: `src/extract/office.rs` — no code change, add a test

`office.rs` already calls `office_oxide::Document::to_markdown()`, which emits GFM tables. Add a regression test proving a table-bearing office doc yields a pipe table in a chunk. If the existing `tests/fixtures/sample.docx` already contains a table, assert against it; otherwise add `tests/fixtures/sample_table.docx` (a minimal Word doc with a 2×2 table) and assert a chunk contains `"| --- |"` (or a `|`-delimited header/separator row).

## API reference (oxidize-pdf 2.16.6, verified against crate source)

- `PdfDocument::partition(&self) -> ParseResult<Vec<Element>>` — runs `preserve_layout + reconstruct_paragraphs` internally, then structure/table detection.
- `enum Element { Title, Paragraph, Table(TableElementData), Header, Footer, ListItem, Image, CodeBlock, KeyValue }`; field `Element.page: u32`; `Element::Table(t)` where `t.rows: Vec<Vec<String>>`.
- `ElementMarkdownExporter::new(config: ExportConfig).export(&[Element]) -> String` — `Element::Table` → `table_to_markdown(&t.rows)` (GFM `| a | b |` + `| --- |`). `ExportConfig::default()`.
- Import path: `oxidize_pdf::pipeline::{Element, ElementMarkdownExporter, ExportConfig}` (implementer to confirm exact re-export path; may be `oxidize_pdf::pipeline::export::…`).

## Testing

- **PDF — fixture still works:** `sample.pdf` → one chunk `p.1` containing `"glossa sample"` (existing assertion preserved).
- **PDF — page numbering:** `sample.pdf` (1 page) → location exactly `"p.1"`.
- **PDF — table detection:** a ruled-table PDF fixture → a chunk whose markdown contains a `|`-delimited table row (`"| --- |"` or a pipe row). Primary approach: generate the fixture programmatically via oxidize-pdf's writer (`AdvancedTableBuilder`) in a test helper (no committed binary, pure-Rust); if the writer→reader round-trip does not reliably trigger detection, commit a small synthetic `tests/fixtures/table.pdf` instead (implementer's call, flag which was used).
- **PDF — no-text fallback:** the existing `unparseable_pdf_is_indexed_by_filename_not_dropped` test stays green.
- **DOCX — table:** as above.
- Full `cargo test -p glossa` green.

## Global constraints

- **Pure-Rust, C-free** (`cargo tree -p glossa -i cc` empty): no new C deps; oxidize-pdf + office_oxide already present, no new crates.
- **File-First**; indexing must never abort or silently drop a document's text (3-layer fallback).
- **Addressing contract preserved:** `read(path, n)` maps to `p.N`; the 1-based page convention is unchanged.
- **MCP/eval unaffected:** extraction is upstream of `glossa::tools::*`; the shared tool layer needs no change. Chunk text simply becomes higher-quality markdown.
- TDD; keep the panic-guard and `ParseOptions::lenient()` reader.

## Out of scope (follow-ups)

- `PartitionConfig` / `TableDetectionConfig` tuning on real RU PDFs — start with defaults; tune later if detection underperforms.
- Caching read-images in the index (separate deferred Minor).
- Re-running the eval (post-merge, separate step).

## Risks

- `partition()` is heavier than flat extraction → slower indexing (one-time cost; acceptable for quality).
- Table detection is **heuristic**: clean ruled/aligned tables convert well; merged/multiline-cell tables degrade to aligned text (graceful, not a regression vs today).
