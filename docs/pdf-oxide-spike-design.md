> **Superseded** (2026-07-14): Spike complete — glossa migrated to `pdf_oxide` for PDF text, page render, and embeds. See `docs/superpowers/specs/2026-07-14-pdf-oxide-migration-design.md`.

# PDF library spike: oxidize-pdf vs pdf_oxide

**Date:** 2026-07-14  
**Status:** Approved — ready to run  
**Corpus:** `kb-gost` (path resolved at run time; typically next to or under the workspace)

## Context

`read` should optionally return a **raster of a PDF page** for vision models (`page_image` flag: image only, PDF-only, default off). Current `oxidize-pdf` (locked `2.16.6`) extracts text and **embedded** images but does **not** rasterize pages to PNG.

Candidate pure-Rust renderer: `pdf_oxide` (`rendering` feature, tiny-skia). Before choosing dual-stack vs full extract migration vs external render, run a measured spike on real ГОСТ PDFs.

## Goal

On 8–10 files from `kb-gost`, compare:

1. **Text** — same page via current glossa oxidize path vs `pdf_oxide` extract  
2. **Raster** — `pdf_oxide` PNG @ 150 DPI  

Decide:

| Outcome | Next step |
|---------|-----------|
| pdf_oxide text ≥ oxidize on table pages; PNG readable | Consider extract migrate **or** dual |
| pdf_oxide text worse / breaks numbers; PNG OK | **Dual:** oxidize text + pdf_oxide render only |
| PNG poor (Cyrillic/tables) | External render (`pdftoppm` / pdfium); keep oxidize extract |

## Corpus selection

From `kb-gost`, pick ~8–10 PDFs:

- ≥3 dense tables  
- ≥2 multi-column / messy layout  
- 1–2 scan / little text  
- ≥1 simple text PDF  

Record chosen paths in `SUMMARY.md`.

## Protocol (per file, per interesting page N)

1. Extract text with oxidize (same options as `PdfExtractor` in `src/extract/pdf.rs`).  
2. Extract text with pdf_oxide for page N−1 (0-based).  
3. Write `tmp/pdf-spike/<stem>/pN.oxidize.txt` and `pN.pdf_oxide.txt`.  
4. Render `pN.png` via pdf_oxide @ 150 DPI.  
5. Score text 1–5 and PNG 1–5 (human); note whether key numerals survive (e.g. table values).

## Deliverables

- Spike binary or `examples/pdf_spike.rs` (dev-only; not production MCP wiring)  
- `tmp/pdf-spike/**` artifacts (gitignored if under `tmp/`)  
- `tmp/pdf-spike/SUMMARY.md` — scores + recommendation  

## Out of scope

- MCP `page_image` flag / production `Cargo.toml` default deps for render  
- Office formats  
- Changing production extract path before SUMMARY verdict  

## Follow-up

After SUMMARY: write feature design for `page_image` using the chosen stack; then implementation plan.
