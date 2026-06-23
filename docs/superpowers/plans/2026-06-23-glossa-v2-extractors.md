# glossa v2 — Milestone 2: Office + PDF extractors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `kb search` work over Office documents (docx/doc/xlsx/xls/pptx/ppt) and PDFs, by adding two extractors behind the existing `Extractor` trait, and clear the Milestone 1 review backlog.

**Architecture:** `office_oxide` parses all six Office formats from in-memory bytes; we render each to Markdown (`to_markdown()`) and reuse Milestone 1's heading-splitter to produce heading-scoped chunks. `pdf-extract` yields PDF page text, joined into one chunk. Both register in `walk::extractors()` so the walker and CLI pick them up unchanged. No new search/index logic.

**Tech Stack:** Rust; `office_oxide = "0.1"` (verified against the author's Zeroclaw `office-tools` plugin — `Document::from_reader(Cursor, DocumentFormat)` + `to_markdown()`), `pdf-extract = "0.10"` (`extract_text_from_mem_by_pages(&[u8]) -> Result<Vec<String>>`). Both pure Rust / offline.

## Global Constraints

- Pure Rust, **single static binary**, **fully offline** — no network, no native/system libs.
- New deps for this milestone: ONLY `office_oxide = "0.1"` and `pdf-extract = "0.10"`. No `calamine`, no writer crates (`printpdf`/`docx-rs`/`rust_xlsxwriter` are NOT used — fixtures are real committed sample files).
- All extractors implement the existing `extract::Extractor` trait; the office backend is isolated so it stays swappable (office_oxide is young, v0.1.x).
- Reuse Milestone 1's heading-splitter — do not reimplement chunking.
- Search syntax stays ripgrep 1:1 (unchanged); files remain the source of truth.
- TDD: failing test first; frequent commits; DRY; YAGNI.

## Fixture prerequisite (read before Task 3)

Tasks 3–5 need **real, small** sample files committed under `tests/fixtures/`, each containing the ASCII marker text `glossa sample` (and, where the format has headings, a heading named `Sample`):

- `tests/fixtures/sample.docx` (required for Task 3)
- `tests/fixtures/sample.pdf` (required for Task 4, must be a *text* PDF, not a scan)

These are provided by the project owner (real files from any office suite). A subagent cannot fabricate valid binary Office/PDF files; if a required fixture is absent, the task is BLOCKED pending the file. Other Office formats (doc/xlsx/xls/pptx/ppt) share the identical code path and can get fixtures + smoke tests later without code changes.

## Out of scope (deferred to later milestones)

- Embedded **image extraction** (zip media) — deferred to the `read`/MCP milestone, where it has a consumer (no consumer exists yet; adding it now is YAGNI).
- Per-page PDF locations, per-sheet `Sheet!Row` precision — deferred (v1 = whole-document text for PDF; spreadsheets render via `to_markdown`).

---

### Task 1: Clear Milestone 1 review backlog

**Files:**
- Modify: `src/extract/markdown.rs` (empty-title heading fix + tests)
- Modify: `src/query.rs` (add an `ignore_case` test)
- Modify: `src/walk.rs` (log directory-traversal errors to stderr)
- Test: inline in the above files

**Interfaces:**
- Consumes: existing `parse_atx_heading`, `QueryOpts`/`compile`, `collect_chunks`.
- Produces: no signature changes — behavior fixes + new tests only.

- [ ] **Step 1: Write failing test — bare `## ` is not a heading**

In `src/extract/markdown.rs` `mod tests`, add:
```rust
    #[test]
    fn empty_title_heading_is_body_not_heading() {
        // A hash run with no title text must not create an empty location segment.
        let md = "# A\n## \nbody\n";
        let chunks = MarkdownExtractor
            .extract(Path::new("d.md"), md.as_bytes())
            .unwrap();
        // "## " is treated as body, so everything stays under "A".
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].location, "A");
        assert!(!chunks[0].location.contains(" > "));
    }
```

- [ ] **Step 2: Run it — verify it fails**

Run: `cargo test --lib empty_title_heading_is_body_not_heading`
Expected: FAIL — current `parse_atx_heading("## ")` returns `Some((2, ""))`, producing location `"A > "`.

- [ ] **Step 3: Fix `parse_atx_heading`**

In `src/extract/markdown.rs`, replace the tail of `parse_atx_heading`:
```rust
    let rest = &t[hashes..];
    // CommonMark requires a space (or EOL) after the # run.
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }
    let title = rest.trim();
    if title.is_empty() {
        return None; // a hashes-only line ("##", "## ") is body, not a heading
    }
    Some((hashes, title.to_string()))
```

- [ ] **Step 4: Add a heading-level-jump regression test (documents current behavior)**

In the same `mod tests`:
```rust
    #[test]
    fn heading_level_jump_keeps_deterministic_path() {
        // h1 -> h3 skips the h2 slot; location is "A > C" (no panic, deterministic).
        let md = "# A\n### C\nbody\n";
        let chunks = MarkdownExtractor
            .extract(Path::new("d.md"), md.as_bytes())
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].location, "A > C");
    }
```

- [ ] **Step 5: Add isolated `ignore_case` test in query.rs**

In `src/query.rs` `mod tests`:
```rust
    #[test]
    fn ignore_case_forces_case_insensitive_even_with_uppercase_pattern() {
        let re = compile("Cat", &QueryOpts { ignore_case: true, ..Default::default() }).unwrap();
        assert!(re.is_match("cat"));
        assert!(re.is_match("CAT"));
    }
```

- [ ] **Step 6: Log WalkDir traversal errors in walk.rs**

In `src/walk.rs`, replace:
```rust
    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
```
with:
```rust
    for entry in WalkDir::new(root) {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skip (walk error): {e}");
                continue;
            }
        };
```
(The existing loop body and its closing brace are unchanged.)

- [ ] **Step 7: Run the full suite**

Run: `cargo test`
Expected: PASS — all prior tests plus the 3 new ones; output pristine.

- [ ] **Step 8: Commit**

```bash
git add src/extract/markdown.rs src/query.rs src/walk.rs
git commit -m "fix: clear M1 review backlog (empty-title heading, ignore_case test, walk error logging)"
```

---

### Task 2: Extract reusable Markdown chunker

**Files:**
- Create: `src/extract/chunk.rs`
- Modify: `src/extract.rs` (add `pub mod chunk;`)
- Modify: `src/extract/markdown.rs` (delegate to `chunk::chunk_markdown`)
- Test: `src/extract/chunk.rs` (inline)

**Interfaces:**
- Consumes: `model::Chunk`.
- Produces: `extract::chunk::chunk_markdown(path: &Path, text: &str, file_type: &str) -> Vec<Chunk>`
  and `extract::chunk::parse_atx_heading(line: &str) -> Option<(usize, String)>` (moved here, made `pub`).
  `MarkdownExtractor::extract` becomes a thin wrapper that calls `chunk_markdown(path, &text, "md")`.

- [ ] **Step 1: Write the failing test**

Create `src/extract/chunk.rs`:
```rust
use crate::model::Chunk;
use std::path::Path;

pub fn parse_atx_heading(line: &str) -> Option<(usize, String)> {
    let t = line.trim_start();
    if !t.starts_with('#') {
        return None;
    }
    let hashes = t.chars().take_while(|c| *c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = &t[hashes..];
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }
    let title = rest.trim();
    if title.is_empty() {
        return None;
    }
    Some((hashes, title.to_string()))
}

fn push_chunk(
    path: &Path,
    heading_path: &[String],
    file_type: &str,
    buf: &mut String,
    out: &mut Vec<Chunk>,
) {
    if buf.trim().is_empty() {
        buf.clear();
        return;
    }
    out.push(Chunk {
        doc_path: path.to_path_buf(),
        location: heading_path.join(" > "),
        file_type: file_type.to_string(),
        text: std::mem::take(buf),
    });
}

/// Split Markdown (or Markdown rendered from another format) into heading-scoped chunks.
pub fn chunk_markdown(path: &Path, text: &str, file_type: &str) -> Vec<Chunk> {
    let mut out = Vec::new();
    let mut heading_path: Vec<String> = Vec::new();
    let mut buf = String::new();

    for line in text.lines() {
        if let Some((level, title)) = parse_atx_heading(line) {
            push_chunk(path, &heading_path, file_type, &mut buf, &mut out);
            heading_path.truncate(level.saturating_sub(1));
            heading_path.push(title);
        } else {
            buf.push_str(line);
            buf.push('\n');
        }
    }
    push_chunk(path, &heading_path, file_type, &mut buf, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_type_is_propagated_into_chunks() {
        let chunks = chunk_markdown(Path::new("x.docx"), "# H\nbody\n", "docx");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].file_type, "docx");
        assert_eq!(chunks[0].location, "H");
    }
}
```

- [ ] **Step 2: Run it — verify it fails**

Run: `cargo test`
Expected: FAIL — `extract::chunk` module not declared.

- [ ] **Step 3: Declare the module and delegate from Markdown**

In `src/extract.rs`, add near the other `pub mod` lines:
```rust
pub mod chunk;
```

Replace the body of `src/extract/markdown.rs` with the delegating version (drop the local `parse_atx_heading`/`push_chunk`/loop, keep the struct + the markdown tests):
```rust
use crate::extract::chunk::chunk_markdown;
use crate::extract::Extractor;
use crate::model::Chunk;
use std::path::Path;

pub struct MarkdownExtractor;

impl Extractor for MarkdownExtractor {
    fn file_types(&self) -> &'static [&'static str] {
        &["md", "markdown"]
    }

    fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>> {
        let text = String::from_utf8_lossy(bytes);
        Ok(chunk_markdown(path, &text, "md"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_by_headings_with_location_path() {
        let md = "# A\nintro\n## B\nbody b\n# C\nbody c\n";
        let chunks = MarkdownExtractor.extract(Path::new("d.md"), md.as_bytes()).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].location, "A");
        assert_eq!(chunks[0].text.trim(), "intro");
        assert_eq!(chunks[1].location, "A > B");
        assert_eq!(chunks[2].location, "C");
    }

    #[test]
    fn hash_without_space_is_not_a_heading() {
        let md = "#nothashtag is body\n";
        let chunks = MarkdownExtractor.extract(Path::new("d.md"), md.as_bytes()).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].location, "");
        assert!(chunks[0].text.contains("#nothashtag"));
    }

    #[test]
    fn empty_title_heading_is_body_not_heading() {
        let md = "# A\n## \nbody\n";
        let chunks = MarkdownExtractor.extract(Path::new("d.md"), md.as_bytes()).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].location, "A");
        assert!(!chunks[0].location.contains(" > "));
    }

    #[test]
    fn heading_level_jump_keeps_deterministic_path() {
        let md = "# A\n### C\nbody\n";
        let chunks = MarkdownExtractor.extract(Path::new("d.md"), md.as_bytes()).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].location, "A > C");
    }
}
```
(The empty-title fix and jump test from Task 1 now live in `chunk.rs`'s `parse_atx_heading` and these tests; the behavior is identical, so all assertions still hold.)

- [ ] **Step 4: Run the full suite**

Run: `cargo test`
Expected: PASS — markdown tests behave identically through the shared function; `chunk` test passes.

- [ ] **Step 5: Commit**

```bash
git add src/extract.rs src/extract/chunk.rs src/extract/markdown.rs
git commit -m "refactor: extract reusable chunk_markdown for non-markdown sources"
```

---

### Task 3: Office extractor (office_oxide, all six formats)

**Files:**
- Modify: `Cargo.toml` (add `office_oxide = "0.1"`)
- Create: `src/extract/office.rs`
- Modify: `src/extract.rs` (add `pub mod office;`)
- Add (project owner): `tests/fixtures/sample.docx` — a real docx whose body contains `glossa sample`
- Test: `src/extract/office.rs` (inline, using `include_bytes!` of the fixture)

**Interfaces:**
- Consumes: `extract::Extractor`, `extract::chunk::chunk_markdown`, `model::Chunk`, office_oxide.
- Produces: `extract::office::OfficeExtractor` implementing `Extractor` for
  `["docx", "doc", "xlsx", "xls", "pptx", "ppt"]`.

- [ ] **Step 0: Confirm the fixture exists**

Verify `tests/fixtures/sample.docx` is present and contains the text `glossa sample`.
If missing, STOP and report BLOCKED — request the fixture file (a subagent cannot fabricate a valid docx).

- [ ] **Step 1: Write the failing test**

Create `src/extract/office.rs`:
```rust
use crate::extract::chunk::chunk_markdown;
use crate::extract::Extractor;
use crate::model::Chunk;
use anyhow::anyhow;
use office_oxide::{Document, DocumentFormat};
use std::io::Cursor;
use std::path::Path;

pub struct OfficeExtractor;

fn format_for(ext: &str) -> Option<DocumentFormat> {
    match ext {
        "docx" => Some(DocumentFormat::Docx),
        "doc" => Some(DocumentFormat::Doc),
        "xlsx" => Some(DocumentFormat::Xlsx),
        "xls" => Some(DocumentFormat::Xls),
        "pptx" => Some(DocumentFormat::Pptx),
        "ppt" => Some(DocumentFormat::Ppt),
        _ => None,
    }
}

impl Extractor for OfficeExtractor {
    fn file_types(&self) -> &'static [&'static str] {
        &["docx", "doc", "xlsx", "xls", "pptx", "ppt"]
    }

    fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        let fmt = format_for(&ext).ok_or_else(|| anyhow!("unsupported office extension: {ext}"))?;
        let doc = Document::from_reader(Cursor::new(bytes.to_vec()), fmt)
            .map_err(|e| anyhow!("office parse failed for {}: {e}", path.display()))?;
        let md = doc.to_markdown();
        Ok(chunk_markdown(path, &md, &ext))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_text_from_docx_fixture() {
        let bytes = include_bytes!("../../tests/fixtures/sample.docx");
        let chunks = OfficeExtractor
            .extract(Path::new("sample.docx"), bytes)
            .unwrap();
        let joined = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n");
        assert!(
            joined.contains("glossa sample"),
            "expected fixture marker text, got: {joined}"
        );
        assert!(chunks.iter().all(|c| c.file_type == "docx"));
    }

    #[test]
    fn unsupported_extension_errors() {
        let err = OfficeExtractor.extract(Path::new("x.rtf"), b"junk").unwrap_err();
        assert!(err.to_string().contains("unsupported office extension"));
    }
}
```

- [ ] **Step 2: Run it — verify it fails**

Run: `cargo test --lib office`
Expected: FAIL — `office_oxide` dep + `extract::office` module not present (won't compile).

- [ ] **Step 3: Add the dependency and declare the module**

In `Cargo.toml` under `[dependencies]`:
```toml
office_oxide = "0.1"
```
In `src/extract.rs`:
```rust
pub mod office;
```

- [ ] **Step 4: Run it — verify it passes**

Run: `cargo test --lib office`
Expected: PASS (2 tests). If `to_markdown()` produces no headings for the docx, the single chunk still contains the marker text — the assertion only checks `contains`.

- [ ] **Step 5: Run the full suite**

Run: `cargo test`
Expected: PASS, output pristine (office_oxide must not print warnings to the test output; if it does, note it).

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock src/extract.rs src/extract/office.rs tests/fixtures/sample.docx
git commit -m "feat: office extractor via office_oxide (docx/doc/xlsx/xls/pptx/ppt)"
```

---

### Task 4: PDF extractor (pdf-extract)

**Files:**
- Modify: `Cargo.toml` (add `pdf-extract = "0.10"`)
- Create: `src/extract/pdf.rs`
- Modify: `src/extract.rs` (add `pub mod pdf;`)
- Add (project owner): `tests/fixtures/sample.pdf` — a real *text* PDF containing `glossa sample`
- Test: `src/extract/pdf.rs` (inline)

**Interfaces:**
- Consumes: `extract::Extractor`, `model::Chunk`, pdf-extract.
- Produces: `extract::pdf::PdfExtractor` implementing `Extractor` for `["pdf"]`; one whole-document
  chunk with `location = ""` and `file_type = "pdf"`.

- [ ] **Step 0: Confirm the fixture exists**

Verify `tests/fixtures/sample.pdf` exists and is a text PDF (not a scan) containing `glossa sample`.
If missing, STOP and report BLOCKED — request the fixture.

- [ ] **Step 1: Write the failing test**

Create `src/extract/pdf.rs`:
```rust
use crate::extract::Extractor;
use crate::model::Chunk;
use anyhow::anyhow;
use std::path::Path;

pub struct PdfExtractor;

impl Extractor for PdfExtractor {
    fn file_types(&self) -> &'static [&'static str] {
        &["pdf"]
    }

    fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>> {
        let pages = pdf_extract::extract_text_from_mem_by_pages(bytes)
            .map_err(|e| anyhow!("pdf parse failed for {}: {e}", path.display()))?;
        let text = pages.join("\n");
        Ok(vec![Chunk {
            doc_path: path.to_path_buf(),
            location: String::new(),
            file_type: "pdf".into(),
            text,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_text_from_pdf_fixture() {
        let bytes = include_bytes!("../../tests/fixtures/sample.pdf");
        let chunks = PdfExtractor.extract(Path::new("sample.pdf"), bytes).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].file_type, "pdf");
        assert!(
            chunks[0].text.contains("glossa sample"),
            "expected fixture marker text, got: {}",
            chunks[0].text
        );
    }
}
```

- [ ] **Step 2: Run it — verify it fails**

Run: `cargo test --lib pdf`
Expected: FAIL — `pdf-extract` dep + `extract::pdf` module not present.

- [ ] **Step 3: Add the dependency and declare the module**

In `Cargo.toml` under `[dependencies]`:
```toml
pdf-extract = "0.10"
```
In `src/extract.rs`:
```rust
pub mod pdf;
```

- [ ] **Step 4: Run it — verify it passes**

Run: `cargo test --lib pdf`
Expected: PASS (1 test). Note: `pdf-extract` may log font warnings; if they appear in test output, record as a concern (not a failure).

- [ ] **Step 5: Run the full suite**

Run: `cargo test`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock src/extract.rs src/extract/pdf.rs tests/fixtures/sample.pdf
git commit -m "feat: pdf extractor via pdf-extract (whole-document text)"
```

---

### Task 5: Register extractors in the walker + cross-format integration test

**Files:**
- Modify: `src/walk.rs` (extend `extractors()`)
- Test: `tests/formats_it.rs` (integration test over `tests/fixtures/`)

**Interfaces:**
- Consumes: `extract::markdown::MarkdownExtractor`, `extract::office::OfficeExtractor`,
  `extract::pdf::PdfExtractor`, `walk::collect_chunks`.
- Produces: `walk::extractors()` now returns all three; the walker/CLI dispatch office + pdf files
  automatically (no CLI changes needed).

- [ ] **Step 1: Write the failing test**

Create `tests/formats_it.rs`:
```rust
use glossa::walk::collect_chunks;
use std::path::Path;

#[test]
fn collects_chunks_across_office_and_pdf_fixtures() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let chunks = collect_chunks(&fixtures, None).unwrap();
    assert!(chunks.iter().any(|c| c.file_type == "docx"), "no docx chunks");
    assert!(chunks.iter().any(|c| c.file_type == "pdf"), "no pdf chunks");
    assert!(
        chunks.iter().any(|c| c.text.contains("glossa sample")),
        "marker text not found in any chunk"
    );
}
```

- [ ] **Step 2: Run it — verify it fails**

Run: `cargo test --test formats_it`
Expected: FAIL — `extractors()` only returns `MarkdownExtractor`, so docx/pdf files are skipped.

- [ ] **Step 3: Extend `extractors()`**

In `src/walk.rs`, update imports and the function:
```rust
use crate::extract::markdown::MarkdownExtractor;
use crate::extract::office::OfficeExtractor;
use crate::extract::pdf::PdfExtractor;
use crate::extract::Extractor;
```
```rust
pub fn extractors() -> Vec<Box<dyn Extractor>> {
    vec![
        Box::new(MarkdownExtractor),
        Box::new(OfficeExtractor),
        Box::new(PdfExtractor),
    ]
}
```

- [ ] **Step 4: Run it — verify it passes**

Run: `cargo test --test formats_it`
Expected: PASS.

- [ ] **Step 5: Run the full suite + a manual CLI smoke**

Run: `cargo test`
Expected: PASS (all suites).
Run: `cargo run --bin kb -- search "glossa sample" tests/fixtures`
Expected: prints hits for `sample.docx` and `sample.pdf` as `path:location:line: snippet`.

- [ ] **Step 6: Commit**

```bash
git add src/walk.rs tests/formats_it.rs
git commit -m "feat: register office + pdf extractors in the walker"
```

---

## Self-Review

**Spec coverage (Milestone 2 slice):**
- Office formats incl. legacy via `office_oxide` behind `Extractor` trait → Task 3. ✓
- PDF text via `pdf-extract` → Task 4. ✓
- Reuse heading-splitter (no reimplementation) → Task 2 (`chunk_markdown`). ✓
- Walker/CLI pick up new formats with no search changes → Task 5. ✓
- M1 review backlog cleared → Task 1. ✓
- Deferred (stated): image extraction, per-page PDF, per-sheet precision. ✓

**Placeholder scan:** none — every code/test step has complete code and exact commands. The only non-code prerequisites are the two real fixture files (Task 3/4 Step 0), called out explicitly because binary samples cannot be authored in a plan.

**Type consistency:** `Extractor` trait (`file_types`/`extract`) matches Milestone 1 verbatim; `chunk_markdown(&Path, &str, &str) -> Vec<Chunk>` is defined in Task 2 and consumed identically in Tasks 2 & 3; `Chunk` fields unchanged; `office_oxide` API (`Document::from_reader`, `DocumentFormat`, `to_markdown`) matches the verified Zeroclaw usage; `pdf_extract::extract_text_from_mem_by_pages(&[u8]) -> Result<Vec<String>>` per recon. ✓

**Risk note:** `office_oxide` is v0.1.x; it is isolated in `src/extract/office.rs` behind the trait, so a future swap touches one file. If `to_markdown()` for spreadsheets proves too coarse, a dedicated `calamine`-based spreadsheet extractor can replace the xls/xlsx branch later without changing the trait or the walker.
