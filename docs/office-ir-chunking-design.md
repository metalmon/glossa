# Office IR extraction, merge expand, and IR-level chunking

**Date:** 2026-07-14  
**Status:** Draft for review  
**Context:** Docx tables are accurate via `office_oxide`, but headingless Office docs become one huge markdown blob; PDF standardization loses table fidelity. We keep Office formats and fix structure in our pipeline.

## Goal

For **all** Office formats (`docx`/`doc`/`xlsx`/`xls`/`pptx`/`ppt`):

1. Parse via `office_oxide` **DocumentIR** (not `to_markdown()` as the primary path).
2. **Expand merged cells** (repeat cell value horizontally and vertically into a dense grid).
3. **Chunk at IR level** with universal, format-agnostic rules (no GOST/bold heuristics).
4. Emit the same `Chunk { doc_path, location, file_type, text }` as today; agent refs stay `path#N`.

Out of scope: PDF pipeline changes; inventing “pages” for docx; render docx→PDF for paging.

## Current state

- `OfficeExtractor` calls `Document::to_markdown()` then `chunk_markdown` (ATX headings only).
- Headingless docs → one chunk → weak MENTIONS / Marka / speed targeting.
- `office_oxide` has public `Document::to_ir() → DocumentIR` with `Element::{Heading, Paragraph, Table, …}` and table `col_span` / `row_span`.
- Library does **not** densify merges: IR skips vMerge continue cells; markdown ignores spans. No expand option.

## Architecture

```
bytes → Document::from_reader → to_ir()
         → expand_merged_tables(IR)     // in-place / new IR
         → chunk_ir(path, IR, file_type) // Vec<Chunk>
              └── each chunk: render selected elements → GFM markdown text
```

`.md` stays on `chunk_markdown` (optional later: same size-fallback on blank lines). PDF unchanged (`p.N`).

### Modules

| Unit | Responsibility |
|------|----------------|
| `extract/office.rs` | Open document, `to_ir()`, call expand + chunk; no markdown-first path |
| `extract/office_table.rs` (new) | Dense-grid merge expand; table → GFM pipes |
| `extract/office_chunk.rs` (new) | IR walk, split rules, per-chunk markdown render |
| `extract/chunk.rs` | Unchanged for markdown/PDF consumers (unless we add blank-line fallback later) |

Depend only on public `office_oxide::{Document, DocumentIR, ir::*}`.

## 1. Merge expand (H + V)

**Input:** `ir::Table` with `col_span` / `row_span` (continue cells already omitted by office_oxide).  
**Output:** rectangular grid; every logical cell has `col_span = 1`, `row_span = 1`; merge origin text **copied** into all covered positions.

Algorithm:

1. Infer column count from max occupied grid width (sum of `col_span` per row, accounting for active vertical spans).
2. Allocate `rows × cols` of optional cell content.
3. Place each IR cell at the next free grid slot; fill the `row_span × col_span` rectangle with clones of that cell’s rendered plain/inline content.
4. Rebuild `Table` with one cell per grid slot.

Apply to every `Element::Table` in every section (and nested tables inside cells, if any — recurse).

**XLSX:** IR today often has `col_span`/`row_span` = 1; if merge regions are not in IR, expand is a no-op until upstream exposes them. Still run the same code path. Do not special-case formats.

**Tests:** synthetic IR tables — horizontal span, vertical span, 2×2 block; assert dense pipe markdown repeats values.

## 2. IR chunking

Walk sections in order. Flatten to a sequence of **block elements** (skip empty decorative breaks used only as split hints — see below).

### Hard splits (always flush buffer)

- Boundary between IR `Section`s when there is more than one (Excel sheets, PPTX slides, multi-sectPr Word).
- `Element::Heading` — flush before the heading; heading starts the next chunk’s location path (same spirit as `chunk_markdown`).

### Soft splits (only if buffer size ≥ threshold)

- **Empty paragraph:** `Paragraph` whose inline text trims to empty (and no meaningful children). Consumed as a delimiter, not kept as body.
- Else, if still over threshold: boundary **before** the next element, subject to **glue**.

Empty paragraphs are **not** hard splits before the threshold (avoids exploding spacing into hundreds of tiny chunks).

### Glue (never split here)

- Do **not** split between a non-empty paragraph/heading and an immediately following `Table` (keeps captions with tables).
- Do **not** split between a `Table` and an immediately following non-empty paragraph (keeps notes / «Примечание» under the table).
- Do **not** split inside a `Table` (table is one element after expand).
- Glue applies to **at most one** paragraph on each side of the table (caption before, note after). Further paragraphs are normal soft/hard split candidates.

### Size threshold

- Default **4000 characters** of rendered text in the buffer (align with `extract/text.rs` `MAX_CHARS`).
- Measured on the markdown that would be emitted for buffered elements (prefer rendered markdown length for consistency with indexed body).
- Constant in one place (`office_chunk.rs`); not user-facing config in v1.

### Location strings

- Prefer heading path (`A > B`) when headings were seen in this section, matching markdown behavior.
- Else if `Section.title` is set (sheet/slide name), use that (and optionally `title > part` if size-split inside a sheet).
- Else empty location (ordinal `#N` from index remains the stable agent handle), same as headingless markdown today.

### Rendering

Per chunk, render its elements to GFM:

- Headings → `#`…  
- Paragraphs → text (+ inline bold/italic as today)  
- Tables → pipe tables from **expanded** grid  
- Lists → markdown lists  

Reuse patterns from office_oxide’s markdown rendering where practical, but **our** table renderer must use the expanded grid (do not call library `to_markdown()` for the whole doc).

## 3. Error handling

- Parse failures: same as today (`anyhow` with path).
- Empty IR / no text: prefer zero chunks if no body (match sparse fixtures).
- Oversized single element (huge table): emit as one chunk even above threshold (cannot split without breaking glue/atomicity). Acceptable; researcher already greps with context.

## 4. Testing

| Case | Expect |
|------|--------|
| Fixture `sample.docx` / `sample_table.docx` | Still extract; table has pipes |
| Synthetic merge H/V/block | Repeated values in every covered cell |
| Headingless IR with empty paras + size over threshold | ≥2 chunks; splits on empty para after threshold |
| Paragraph then Table (and Table then note paragraph) under size pressure | Same chunk (glue both sides) |
| Multi-section IR (two sheet titles) | ≥2 chunks; locations carry titles |
| Existing office unit tests | Updated for IR path; no regress on markers |

Eval follow-up (manual / existing qwen35 harness, not blocking unit-test merge): docx Marka 0/3→3/3 without losing grain/T/H table values.

## 5. Alternatives considered

| Approach | Why not |
|----------|---------|
| Size-fallback only on markdown | No merge expand; empty-para weaker than IR elements; captions harder to glue |
| PDF as universal page unit | Table extraction quality collapses (grain 0/25) |
| Always split on every empty paragraph | Explodes into tiny chunks from spacing paras |
| Bold/ГОСТ heading heuristics | Format-specific, brittle |

**Chosen:** IR + merge expand + hard section/heading splits + soft empty-para (post-threshold) + caption glue.

## Success criteria

1. Office extractor never calls whole-doc `to_markdown()` for indexing.
2. Merged cells appear densified in chunk text (H and V repeat).
3. Headingless multi-thousand-char Office docs produce multiple chunks without GOST-specific rules.
4. Table captions immediately preceding tables, and one note paragraph immediately following, are not stranded across a size split.
5. Unit tests cover expand + chunk rules; existing fixtures still pass.
