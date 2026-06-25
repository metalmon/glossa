# Table-Aware Document Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract real GFM tables (`| a | b |`) from PDF (via oxidize-pdf `partition()`) and lock the already-working DOCX/XLSX table path with a test.

**Architecture:** Rewrite the PDF extractor to partition each document into typed elements (paragraphs, headings, **tables**), group by page, and render per-page markdown — preserving the `p.N` chunk-addressing contract. A 3-layer fallback (partition → layout-text → filename stem) guarantees no document's text is ever dropped. DOCX/XLSX already render tables via `office_oxide::to_markdown()`; add only a regression test.

**Tech Stack:** Rust, `oxidize-pdf` 2.16.6 (`pipeline::{Element, ElementMarkdownExporter, ExportConfig}`, `parser::PdfDocument`), `office_oxide` 0.1.2.

## Global Constraints

- **Pure-Rust, C-free**: `cargo tree -p glossa -i cc` MUST be empty. No new crates — oxidize-pdf and office_oxide are already deps.
- **File-First**; indexing MUST NEVER abort or silently drop a document's text. Keep the `std::panic::catch_unwind` guard and `ParseOptions::lenient()` reader.
- **Addressing contract preserved**: PDF chunks use `location = "p.N"`, 1-based, first page = `p.1`. `read(path, n)` depends on this — do not change the convention.
- **MCP/eval unaffected**: extraction is upstream of `glossa::tools::*`; no tool-layer change. Chunk text just becomes higher-quality markdown.
- TDD. Frequent commits.

---

### Task 1: PDF extractor — partition() per-page markdown with tables + fallback chain

**Files:**
- Modify: `src/extract/pdf.rs` (replace the `extract_text_with_options` loop with the partition path; keep `extract_text_with_options` as the layer-2 fallback)
- Test: `src/extract/pdf.rs` (existing tests must stay green; add a page-numbering assertion)

**Interfaces:**
- Consumes (oxidize-pdf 2.16.6, verify exact re-export paths against the crate before writing):
  - `oxidize_pdf::parser::{PdfDocument, PdfReader, ParseOptions}` (already imported)
  - `oxidize_pdf::text::ExtractionOptions` (already imported)
  - `oxidize_pdf::pipeline::{Element, ElementMarkdownExporter, ExportConfig}` — NEW import. `PdfDocument::partition(&self) -> ParseResult<Vec<Element>>`; `Element.page: u32`; `Element::Table(t)` with `t.rows: Vec<Vec<String>>`; `ElementMarkdownExporter::new(ExportConfig::default()).export(&[Element]) -> String`.
- Produces: `PdfExtractor::extract` returns `Vec<Chunk>` with `location = "p.N"` (unchanged contract), now containing markdown (tables as pipe tables).

**PAGE NUMBERING — resolve before implementing:** Determine whether `Element.page` from `partition()` is 0-based or 1-based. Check the `do_partition_pages` source in the oxidize-pdf crate AND let the `sample.pdf` fixture test (single page → must be `"p.1"`) be the hard gate. Define `fn page_label(page: u32) -> u32` returning `page` if 1-based or `page + 1` if 0-based. The test below pins it.

- [ ] **Step 1: Update the `extracts_text_from_pdf_fixture` test to also pin a table-free page renders cleanly**

The existing test already asserts `chunks.len() == 1`, `location == "p.1"`, and `text.contains("glossa sample")`. Keep it exactly as-is — it now doubles as the page-numbering gate (a single-page PDF must yield `p.1` through the partition path). No edit needed unless `sample.pdf`'s marker text changes after the rewrite; if the partition exporter alters spacing, relax to `contains("glossa")` only if necessary (flag if so).

- [ ] **Step 2: Run the existing tests to capture the baseline (they pass against `3a398b4`)**

Run: `cargo test -p glossa extract::pdf`
Expected: `extracts_text_from_pdf_fixture` and `unparseable_pdf_is_indexed_by_filename_not_dropped` PASS (pre-change baseline).

- [ ] **Step 3: Rewrite `PdfExtractor::extract` to the partition-first path with a 3-layer fallback**

Replace the body of `extract` (lines 12–74) with:

```rust
fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>> {
    use oxidize_pdf::parser::{ParseOptions, PdfDocument, PdfReader};
    use oxidize_pdf::pipeline::{Element, ElementMarkdownExporter, ExportConfig};
    use oxidize_pdf::text::ExtractionOptions;
    use std::collections::BTreeMap;

    // Any PDF parser can panic on a malformed file; catch it so indexing never aborts.
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let owned = bytes.to_vec();
    let path_buf = path.to_path_buf();
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || -> Vec<Chunk> {
        // lenient() enables xref recovery — parses damaged-but-valid PDFs that strict mode rejects.
        let reader = match PdfReader::new_with_options(std::io::Cursor::new(owned), ParseOptions::lenient()) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        let doc = PdfDocument::new(reader);

        // Layer 1: partition into typed elements (paragraphs, headings, TABLES) and
        // render per-page markdown. Tables become GFM pipe tables.
        if let Ok(elements) = doc.partition() {
            if !elements.is_empty() {
                let mut by_page: BTreeMap<u32, Vec<Element>> = BTreeMap::new();
                for el in elements {
                    by_page.entry(el.page).or_default().push(el);
                }
                let exporter = ElementMarkdownExporter::new(ExportConfig::default());
                let mut out = Vec::new();
                for (page, els) in by_page {
                    let md = exporter.export(&els);
                    if md.trim().is_empty() {
                        continue;
                    }
                    out.push(Chunk {
                        doc_path: path_buf.clone(),
                        location: format!("p.{}", page_label(page)),
                        file_type: "pdf".into(),
                        text: md,
                    });
                }
                if !out.is_empty() {
                    return out;
                }
            }
        }

        // Layer 2: layout-text fallback (preserve_layout reconstructs spaces+newlines)
        // for PDFs where partition finds no structure but raw text exists.
        let opts = ExtractionOptions {
            preserve_layout: true,
            space_threshold: 0.3,
            newline_threshold: 10.0,
            merge_hyphenated: true,
            reconstruct_paragraphs: true,
            detect_columns: true,
            include_artifacts: false,
            ..Default::default()
        };
        if let Ok(pages) = doc.extract_text_with_options(opts) {
            let mut out = Vec::new();
            for (i, page) in pages.iter().enumerate() {
                if page.text.trim().is_empty() {
                    continue;
                }
                out.push(Chunk {
                    doc_path: path_buf.clone(),
                    location: format!("p.{}", i + 1),
                    file_type: "pdf".into(),
                    text: page.text.clone(),
                });
            }
            return out;
        }
        Vec::new()
    }));
    std::panic::set_hook(prev);

    let out = caught.unwrap_or_default();
    if !out.is_empty() {
        return Ok(out);
    }

    // Layer 3: no extractable text (scanned / image-only) or unparseable: NEVER drop the
    // document — index it by filename so it's findable by name.
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("document")
        .to_string();
    eprintln!("  · no text layer, indexed by filename: {}", path.display());
    Ok(vec![Chunk {
        doc_path: path.to_path_buf(),
        location: "(no-text)".into(),
        file_type: "pdf".into(),
        text: name,
    }])
}
```

Add the `page_label` free function in the same module (above the `impl` or below it):

```rust
/// Map oxidize-pdf's `Element.page` to glossa's 1-based `p.N` convention.
/// VERIFY the base against the crate's `do_partition_pages` and the `sample.pdf`
/// fixture test (single page → "p.1"). If `Element.page` is 0-based, return `page + 1`.
fn page_label(page: u32) -> u32 {
    page + 1 // ADJUST: use `page` if partition is already 1-based; sample.pdf test must yield p.1
}
```

- [ ] **Step 4: Run the tests; adjust `page_label` until `sample.pdf → p.1`**

Run: `cargo test -p glossa extract::pdf`
Expected: both tests PASS. If `extracts_text_from_pdf_fixture` reports `location == "p.2"` (or `"p.0"`), flip `page_label` (drop or add the `+1`) and re-run until it is `"p.1"`. If the marker assertion fails because the exporter changed spacing, inspect the actual text and relax to `contains("glossa")` only if genuinely required (flag in the report).

- [ ] **Step 5: Verify C-free invariant holds**

Run: `cargo tree -p glossa -i cc`
Expected: empty output (no `cc` in glossa's tree). No new deps were added.

- [ ] **Step 6: Commit**

```bash
git add src/extract/pdf.rs
git commit -m "feat(pdf): table-aware extraction via partition() (GFM tables, per-page)"
```

---

### Task 2: PDF table-detection regression test

**Files:**
- Create: `tests/fixtures/table.pdf` (a small PDF containing one ruled/aligned table — see Step 1)
- Test: `src/extract/pdf.rs` (new test `extracts_table_as_markdown`)

**Interfaces:**
- Consumes: `PdfExtractor::extract` (Task 1).
- Produces: a committed `tests/fixtures/table.pdf` fixture + a test asserting a chunk's text contains a GFM pipe-table row.

- [ ] **Step 1: Produce `tests/fixtures/table.pdf`**

Generate a small PDF with a single visible **2-column, 2-row ruled table** (e.g. header `Параметр | Значение`, one data row). Generation method is the implementer's choice — it is a one-time committed binary, not a code dependency. Preferred: oxidize-pdf's own table writer (`AdvancedTableBuilder` / `Document` + `Page`, pure-Rust) in a throwaway helper, then commit the output bytes. Acceptable alternative: any tool that emits a standards-compliant PDF with a ruled table. The table must have visible ruling lines or clear column alignment so the detector recognizes it. Keep it tiny (one page).

- [ ] **Step 2: Write the failing test**

```rust
#[test]
fn extracts_table_as_markdown() {
    let bytes = include_bytes!("../../tests/fixtures/table.pdf");
    let chunks = PdfExtractor.extract(Path::new("table.pdf"), bytes).unwrap();
    let joined = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n");
    assert!(
        joined.contains('|') && joined.contains("---"),
        "expected a GFM pipe table (| … | and a --- separator row), got:\n{joined}"
    );
}
```

- [ ] **Step 3: Run the test**

Run: `cargo test -p glossa extract::pdf::tests::extracts_table_as_markdown`
Expected: PASS. If the table is NOT detected (no pipe row), the heuristic missed this layout — make the table more clearly ruled/aligned and regenerate `table.pdf`, OR tune via `partition_with(PartitionConfig { … })` in Task 1's Layer 1 (flag the change). If after reasonable effort detection is unreliable for the generated fixture, commit a hand-made ruled-table PDF that the detector recognizes, and note which fixture source was used in the report.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/table.pdf src/extract/pdf.rs
git commit -m "test(pdf): assert ruled tables extract as GFM markdown"
```

---

### Task 3: DOCX/XLSX table regression test

**Files:**
- Maybe create: `tests/fixtures/sample_table.docx` (only if the existing `sample.docx` has no table — see Step 1)
- Test: `src/extract/office.rs` (new test `extracts_table_as_markdown`)

**Interfaces:**
- Consumes: `OfficeExtractor::extract` (unchanged — `office_oxide::Document::to_markdown()` already emits pipe tables).
- Produces: a test asserting a table-bearing office doc yields a GFM pipe table.

- [ ] **Step 1: Determine the fixture**

Check whether the existing `tests/fixtures/sample.docx` contains a Word table (`w:tbl`) — unzip it and inspect `word/document.xml`, or simply run the extractor and look for a `|` row. If it already has a table, assert against `sample.docx` and skip creating a new fixture. Otherwise create `tests/fixtures/sample_table.docx`: a minimal Word document containing a 2×2 table (header `H1 | H2`, one data row). A `.docx` is a zip of XML; the `zip` crate is already a dep, but any tool may produce this one-time committed fixture.

- [ ] **Step 2: Write the failing test**

```rust
#[test]
fn extracts_table_as_markdown() {
    // Use sample.docx if it already contains a table; otherwise sample_table.docx.
    let bytes = include_bytes!("../../tests/fixtures/sample_table.docx");
    let chunks = OfficeExtractor
        .extract(Path::new("sample_table.docx"), bytes)
        .unwrap();
    let joined = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n");
    assert!(
        joined.contains('|') && joined.contains("---"),
        "expected a GFM pipe table from the docx table, got:\n{joined}"
    );
}
```

(If asserting against the existing `sample.docx`, change the `include_bytes!` path and the `extract` path argument accordingly.)

- [ ] **Step 3: Run the test**

Run: `cargo test -p glossa extract::office::tests::extracts_table_as_markdown`
Expected: PASS (office_oxide renders the table as a pipe table).

- [ ] **Step 4: Run the full glossa suite**

Run: `cargo test -p glossa`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/ src/extract/office.rs
git commit -m "test(office): assert docx tables extract as GFM markdown"
```

---

## Self-Review

**Spec coverage:** PDF table extraction (Task 1+2), DOCX/XLSX table fidelity (Task 3), fallback chain + page-numbering preservation (Task 1), C-free gate (Task 1 Step 5), no-text fallback regression (existing test, Task 1). All spec sections covered.

**Placeholder scan:** `page_label` carries an explicit ADJUST instruction gated by a concrete test — not a placeholder but a resolve-before-merge with a hard gate. Fixture-generation method is deliberately flexible (one-time committed binary) with a precise assertion. No TODO/TBD.

**Type consistency:** `Element`, `ElementMarkdownExporter`, `ExportConfig`, `partition()`, `Element.page`, `TableElementData.rows` match the verified crate API. `Chunk` fields (`doc_path`, `location`, `file_type`, `text`) match the existing struct. `page_label(page: u32) -> u32` used consistently.
