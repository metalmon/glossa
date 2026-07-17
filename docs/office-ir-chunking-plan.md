# Office IR Chunking Implementation Plan

**Goal:** Index all Office formats via `office_oxide` DocumentIR: densify merged cells (H+V value repeat), chunk at IR level (heading/section hard splits, post-threshold empty-paragraph soft splits, caption/note glue around tables), emit GFM `Chunk`s without whole-doc `to_markdown()`.

**Architecture:** `OfficeExtractor` opens with `Document::from_reader` → `to_ir()` → `expand_merged_tables` → `chunk_ir`. New modules `office_table.rs` (expand + pipe render) and `office_chunk.rs` (split rules + element markdown). Threshold 4000 chars (same as `extract/text.rs` `MAX_CHARS`).

**Tech Stack:** Rust, `office_oxide` 0.1 (`Document`, `DocumentIR`, `office_oxide::ir::*`), existing `crate::model::Chunk`, `cargo test`.

**Spec:** [docs/office-ir-chunking-design.md](office-ir-chunking-design.md)

---

## File map

| File | Role |
|------|------|
| Create `src/extract/office_table.rs` | `expand_table`, `expand_merged_tables`, `table_to_markdown`, cell plain-text helpers |
| Create `src/extract/office_chunk.rs` | `CHUNK_CHAR_THRESHOLD`, `chunk_ir`, element→markdown, split/glue logic |
| Modify `src/extract/office.rs` | Wire IR path; keep fixture tests |
| Modify `src/extract.rs` | `pub mod office_table; pub mod office_chunk;` |
| Unchanged | `chunk.rs`, PDF, markdown extractors |

---

### Task 1: Merge expand — failing tests + densify

**Files:**
- Create: `src/extract/office_table.rs`
- Modify: `src/extract.rs` (add `pub mod office_table;`)

- [ ] **Step 1: Add module stub and failing tests**

In `src/extract.rs`, after `pub mod office;`, add:

```rust
pub mod office_chunk;
pub mod office_table;
```

(Create empty `office_chunk.rs` with `// placeholder` so the crate compiles, or only add `office_table` until Task 2 — prefer adding both stubs now.)

Create `src/extract/office_table.rs`:

```rust
use office_oxide::ir::{
    DocumentIR, Element, InlineContent, Paragraph, Table, TableCell, TableRow, TextSpan,
};

/// Expand every table in the IR so merged spans become a dense grid
/// with the origin value repeated in each covered cell.
pub fn expand_merged_tables(ir: &mut DocumentIR) {
    let _ = ir;
    todo!("expand_merged_tables")
}

pub fn expand_table(table: &Table) -> Table {
    let _ = table;
    todo!("expand_table")
}

/// GFM pipe table from an already-expanded (or span=1) table.
pub fn table_to_markdown(table: &Table) -> String {
    let _ = table;
    todo!("table_to_markdown")
}

fn cell_plain(cell: &TableCell) -> String {
    let mut out = String::new();
    for el in &cell.content {
        match el {
            Element::Paragraph(p) => {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(&inline_plain(&p.content));
            }
            Element::Heading(h) => {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(&inline_plain(&h.content));
            }
            _ => {}
        }
    }
    out.trim().replace('\n', " ")
}

fn inline_plain(content: &[InlineContent]) -> String {
    let mut s = String::new();
    for c in content {
        match c {
            InlineContent::Text(t) => s.push_str(&t.text),
            InlineContent::LineBreak => s.push(' '),
            _ => {}
        }
    }
    s
}

fn text_cell(s: &str) -> TableCell {
    TableCell {
        content: vec![Element::Paragraph(Paragraph {
            content: vec![InlineContent::Text(TextSpan {
                text: s.to_string(),
                ..Default::default()
            })],
            ..Default::default()
        })],
        col_span: 1,
        row_span: 1,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_horizontal_span_repeats_value() {
        // | A (colspan 2) |   | B |
        let table = Table {
            rows: vec![TableRow {
                cells: vec![
                    TableCell {
                        content: text_cell("A").content,
                        col_span: 2,
                        row_span: 1,
                        ..Default::default()
                    },
                    text_cell("B"),
                ],
                ..Default::default()
            }],
            ..Default::default()
        };
        let expanded = expand_table(&table);
        assert_eq!(expanded.rows.len(), 1);
        assert_eq!(expanded.rows[0].cells.len(), 3);
        assert_eq!(cell_plain(&expanded.rows[0].cells[0]), "A");
        assert_eq!(cell_plain(&expanded.rows[0].cells[1]), "A");
        assert_eq!(cell_plain(&expanded.rows[0].cells[2]), "B");
        assert!(expanded.rows[0]
            .cells
            .iter()
            .all(|c| c.col_span == 1 && c.row_span == 1));
    }

    #[test]
    fn expand_vertical_span_repeats_value() {
        // col0: X rowspan 2; col1: Y / Z
        let table = Table {
            rows: vec![
                TableRow {
                    cells: vec![
                        TableCell {
                            content: text_cell("X").content,
                            col_span: 1,
                            row_span: 2,
                            ..Default::default()
                        },
                        text_cell("Y"),
                    ],
                    ..Default::default()
                },
                TableRow {
                    // office_oxide omits vMerge continue — only Z
                    cells: vec![text_cell("Z")],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let expanded = expand_table(&table);
        assert_eq!(expanded.rows.len(), 2);
        assert_eq!(expanded.rows[0].cells.len(), 2);
        assert_eq!(expanded.rows[1].cells.len(), 2);
        assert_eq!(cell_plain(&expanded.rows[0].cells[0]), "X");
        assert_eq!(cell_plain(&expanded.rows[1].cells[0]), "X");
        assert_eq!(cell_plain(&expanded.rows[0].cells[1]), "Y");
        assert_eq!(cell_plain(&expanded.rows[1].cells[1]), "Z");
    }

    #[test]
    fn expand_2x2_block_repeats_value() {
        let table = Table {
            rows: vec![
                TableRow {
                    cells: vec![
                        TableCell {
                            content: text_cell("M").content,
                            col_span: 2,
                            row_span: 2,
                            ..Default::default()
                        },
                        text_cell("R1"),
                    ],
                    ..Default::default()
                },
                TableRow {
                    cells: vec![text_cell("R2")],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        // Grid width: colspan2 + R1 = 3 cols; row2 has continue omitted for M + R2
        let expanded = expand_table(&table);
        assert_eq!(expanded.rows[0].cells.len(), 3);
        assert_eq!(expanded.rows[1].cells.len(), 3);
        assert_eq!(cell_plain(&expanded.rows[0].cells[0]), "M");
        assert_eq!(cell_plain(&expanded.rows[0].cells[1]), "M");
        assert_eq!(cell_plain(&expanded.rows[1].cells[0]), "M");
        assert_eq!(cell_plain(&expanded.rows[1].cells[1]), "M");
        assert_eq!(cell_plain(&expanded.rows[0].cells[2]), "R1");
        assert_eq!(cell_plain(&expanded.rows[1].cells[2]), "R2");
    }

    #[test]
    fn table_to_markdown_pipes_expanded_grid() {
        let table = expand_table(&Table {
            rows: vec![TableRow {
                cells: vec![
                    TableCell {
                        content: text_cell("A").content,
                        col_span: 2,
                        row_span: 1,
                        ..Default::default()
                    },
                ],
                is_header: true,
                ..Default::default()
            }],
            ..Default::default()
        });
        let md = table_to_markdown(&table);
        assert!(md.contains("| A | A |"), "got:\n{md}");
        assert!(md.contains("---"), "got:\n{md}");
    }
}
```

Also create `src/extract/office_chunk.rs`:

```rust
// Task 2 will implement chunk_ir here.
```

- [ ] **Step 2: Run tests — expect FAIL**

```powershell
cargo test --lib extract::office_table -- --nocapture
```

Expected: compile error or panic on `todo!`.

- [ ] **Step 3: Implement `expand_table` and `table_to_markdown`**

Algorithm for `expand_table`:

1. Determine `n_rows = table.rows.len()`.
2. Compute column count: walk rows with a `Vec` of remaining vertical-occupancy per column (from prior rowspan). For each row, start `col = 0`; while occupancy[col] > 0, skip; place each cell at next free col, advance by `col_span`, record rowspan into occupancy for following rows. `n_cols = max col reached`.
3. Allocate `grid: Vec<Vec<Option<TableCell>>>` size `n_rows × n_cols`, all `None`.
4. Second pass: same cursor logic; for each source cell, clone into every `(r..r+row_span, c..c+col_span)` with `col_span=1`, `row_span=1`.
5. Build output rows; empty slots → empty `text_cell("")`.

`table_to_markdown`: for each row, `| cell | … |`; after first row emit `| --- |` separators; use `cell_plain`; pad short rows to `n_cols`.

`expand_merged_tables`: for each section, map elements; for `Element::Table(t)` replace with `Element::Table(expand_table(t))`; recurse into cell content if nested tables appear (call expand on any `Table` inside cell `content` before densifying, or densify outer only — nested is rare; recurse `expand_element` for completeness).

```rust
pub fn expand_merged_tables(ir: &mut DocumentIR) {
    for section in &mut ir.sections {
        for el in &mut section.elements {
            expand_element(el);
        }
    }
}

fn expand_element(el: &mut Element) {
    match el {
        Element::Table(t) => {
            for row in &mut t.rows {
                for cell in &mut row.cells {
                    for child in &mut cell.content {
                        expand_element(child);
                    }
                }
            }
            *t = expand_table(t);
        }
        Element::List(list) => {
            for item in &mut list.items {
                for child in &mut item.content {
                    expand_element(child);
                }
                if let Some(nested) = &mut item.nested {
                    for child_item in &mut nested.items {
                        for child in &mut child_item.content {
                            expand_element(child);
                        }
                    }
                }
            }
        }
        _ => {}
    }
}
```

- [ ] **Step 4: Run tests — expect PASS**

```powershell
cargo test --lib extract::office_table -- --nocapture
```

Expected: all four tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/extract/office_table.rs src/extract/office_chunk.rs src/extract.rs
git commit -m "feat(extract): densify office IR merged table cells"
```

---

### Task 2: Element markdown helpers

**Files:**
- Modify: `src/extract/office_chunk.rs`
- Modify: `src/extract/office_table.rs` (export `cell_plain` / `inline` if needed — keep render in `office_chunk`, call `table_to_markdown`)

- [ ] **Step 1: Write failing tests for heading/paragraph/list render**

Append to `office_chunk.rs`:

```rust
use crate::extract::office_table::table_to_markdown;
use crate::model::Chunk;
use office_oxide::ir::{
    DocumentIR, Element, Heading, InlineContent, List, ListItem, Paragraph, Section, Table,
    TextSpan,
};
use std::path::Path;

pub const CHUNK_CHAR_THRESHOLD: usize = 4000;

pub fn chunk_ir(path: &Path, ir: &DocumentIR, file_type: &str) -> Vec<Chunk> {
    let _ = (path, ir, file_type);
    todo!("chunk_ir")
}

pub(crate) fn render_elements(elements: &[Element]) -> String {
    let _ = elements;
    todo!("render_elements")
}

fn inline_md(content: &[InlineContent]) -> String {
    let mut out = String::new();
    for c in content {
        match c {
            InlineContent::Text(t) => {
                let mut s = t.text.clone();
                if t.bold && t.italic {
                    s = format!("***{s}***");
                } else if t.bold {
                    s = format!("**{s}**");
                } else if t.italic {
                    s = format!("*{s}*");
                }
                out.push_str(&s);
            }
            InlineContent::LineBreak => out.push('\n'),
            _ => {}
        }
    }
    out
}

fn text_para(s: &str) -> Element {
    Element::Paragraph(Paragraph {
        content: vec![InlineContent::Text(TextSpan {
            text: s.to_string(),
            ..Default::default()
        })],
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_heading_and_paragraph() {
        let els = vec![
            Element::Heading(Heading {
                level: 1,
                content: vec![InlineContent::Text(TextSpan {
                    text: "Title".into(),
                    ..Default::default()
                })],
                ..Default::default()
            }),
            text_para("Body"),
        ];
        let md = render_elements(&els);
        assert!(md.contains("# Title"), "{md}");
        assert!(md.contains("Body"), "{md}");
    }
}
```

- [ ] **Step 2: Run test — expect FAIL**

```powershell
cargo test --lib extract::office_chunk::tests::render_heading_and_paragraph -- --nocapture
```

- [ ] **Step 3: Implement `render_elements`**

```rust
pub(crate) fn render_elements(elements: &[Element]) -> String {
    let mut parts = Vec::new();
    for el in elements {
        let s = match el {
            Element::Heading(h) => {
                let level = (h.level as usize).clamp(1, 6);
                format!("{} {}", "#".repeat(level), inline_md(&h.content))
            }
            Element::Paragraph(p) => inline_md(&p.content),
            Element::Table(t) => table_to_markdown(t),
            Element::List(list) => render_list(list, 0),
            Element::ThematicBreak => "---".to_string(),
            Element::CodeBlock(cb) => {
                let lang = cb.language.as_deref().unwrap_or("");
                format!("```{lang}\n{}\n```", cb.content)
            }
            _ => String::new(),
        };
        if !s.trim().is_empty() {
            parts.push(s);
        }
    }
    parts.join("\n\n")
}

fn render_list(list: &List, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    let mut lines = Vec::new();
    for (i, item) in list.items.iter().enumerate() {
        let body = item
            .content
            .iter()
            .map(|e| match e {
                Element::Paragraph(p) => inline_md(&p.content),
                other => render_elements(std::slice::from_ref(other)),
            })
            .collect::<Vec<_>>()
            .join(" ");
        let marker = if list.ordered {
            format!("{}. ", i + 1)
        } else {
            "- ".into()
        };
        lines.push(format!("{pad}{marker}{body}"));
        if let Some(nested) = &item.nested {
            lines.push(render_list(nested, indent + 1));
        }
    }
    lines.join("\n")
}
```

- [ ] **Step 4: PASS + commit**

```powershell
cargo test --lib extract::office_chunk::tests::render_heading_and_paragraph -- --nocapture
git add src/extract/office_chunk.rs
git commit -m "feat(extract): render office IR elements to GFM"
```

---

### Task 3: `chunk_ir` — headings, sections, empty location

**Files:**
- Modify: `src/extract/office_chunk.rs`

- [ ] **Step 1: Failing tests**

```rust
#[test]
fn chunk_splits_on_heading() {
    let ir = DocumentIR {
        sections: vec![Section {
            elements: vec![
                text_para("intro"),
                Element::Heading(Heading {
                    level: 1,
                    content: vec![InlineContent::Text(TextSpan {
                        text: "H1".into(),
                        ..Default::default()
                    })],
                    ..Default::default()
                }),
                text_para("under"),
            ],
            ..Default::default()
        }],
        ..Default::default()
    };
    let chunks = chunk_ir(Path::new("a.docx"), &ir, "docx");
    assert!(chunks.len() >= 2, "{:?}", chunks.len());
    assert_eq!(chunks[0].text.contains("intro"), true);
    assert_eq!(chunks.last().unwrap().location, "H1");
    assert!(chunks.last().unwrap().text.contains("under"));
}

#[test]
fn chunk_splits_on_section_boundary() {
    let ir = DocumentIR {
        sections: vec![
            Section {
                title: Some("Sheet1".into()),
                elements: vec![text_para("a")],
                ..Default::default()
            },
            Section {
                title: Some("Sheet2".into()),
                elements: vec![text_para("b")],
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let chunks = chunk_ir(Path::new("a.xlsx"), &ir, "xlsx");
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].location, "Sheet1");
    assert_eq!(chunks[1].location, "Sheet2");
}
```

- [ ] **Step 2: FAIL then implement core `chunk_ir` loop**

Sketch (full logic refined in Task 4 for size/glue):

```rust
pub fn chunk_ir(path: &Path, ir: &DocumentIR, file_type: &str) -> Vec<Chunk> {
    let mut out = Vec::new();
    let multi_section = ir.sections.len() > 1;

    for section in &ir.sections {
        let section_title = section.title.as_deref();
        let mut heading_path: Vec<String> = Vec::new();
        let mut buf: Vec<Element> = Vec::new();

        let flush = |buf: &mut Vec<Element>,
                     heading_path: &[String],
                     section_title: Option<&str>,
                     out: &mut Vec<Chunk>| {
            if buf.is_empty() {
                return;
            }
            let text = render_elements(buf);
            buf.clear();
            if text.trim().is_empty() {
                return;
            }
            let location = if !heading_path.is_empty() {
                heading_path.join(" > ")
            } else {
                section_title.unwrap_or("").to_string()
            };
            out.push(Chunk {
                doc_path: path.to_path_buf(),
                location,
                file_type: file_type.to_string(),
                text,
            });
        };

        // If multi-section, flush between sections is automatic (new buf each section).
        let _ = multi_section;

        for el in &section.elements {
            if let Element::Heading(h) = el {
                flush(&mut buf, &heading_path, section_title, &mut out);
                let title = inline_md(&h.content);
                let level = h.level as usize;
                heading_path.truncate(level.saturating_sub(1));
                heading_path.push(title);
                buf.push(el.clone());
                continue;
            }
            // Task 4: empty-para soft split, size, glue
            buf.push(el.clone());
        }
        flush(&mut buf, &heading_path, section_title, &mut out);
    }
    out
}
```

Note: include the heading element in the chunk body (so `# H1` appears in text), matching mental model of markdown chunker which drops the heading line from body but stores location — **match `chunk_markdown`**: it does NOT put the `#` line into `text`, only location. Spec says “Headings → `#`…” under Rendering — prefer including heading line in chunk text for readability OR match markdown. **Decision locked:** include `# Title` as first line of the chunk after a heading split (better for grep); location still `Title`. Tests above allow either; assert location and body content.

- [ ] **Step 3: PASS + commit**

```powershell
cargo test --lib extract::office_chunk -- --nocapture
git add src/extract/office_chunk.rs
git commit -m "feat(extract): chunk office IR by heading and section"
```

---

### Task 4: Size threshold, empty-paragraph soft split, table glue

**Files:**
- Modify: `src/extract/office_chunk.rs`

- [ ] **Step 1: Failing tests**

Use a **test-only threshold** via `chunk_ir_with_threshold(path, ir, file_type, threshold)` so tests stay small. Production `chunk_ir` calls it with `CHUNK_CHAR_THRESHOLD`.

```rust
pub fn chunk_ir(path: &Path, ir: &DocumentIR, file_type: &str) -> Vec<Chunk> {
    chunk_ir_with_threshold(path, ir, file_type, CHUNK_CHAR_THRESHOLD)
}

pub fn chunk_ir_with_threshold(
    path: &Path,
    ir: &DocumentIR,
    file_type: &str,
    threshold: usize,
) -> Vec<Chunk> {
    // ...
}

fn is_empty_paragraph(el: &Element) -> bool {
    match el {
        Element::Paragraph(p) => inline_md(&p.content).trim().is_empty(),
        _ => false,
    }
}

fn is_table(el: &Element) -> bool {
    matches!(el, Element::Table(_))
}

fn is_nonempty_text_block(el: &Element) -> bool {
    match el {
        Element::Paragraph(p) => !inline_md(&p.content).trim().is_empty(),
        Element::Heading(_) => true,
        _ => false,
    }
}
```

Tests:

```rust
#[test]
fn soft_split_on_empty_para_only_after_threshold() {
    let mut els = Vec::new();
    els.push(text_para(&"x".repeat(50)));
    els.push(text_para("")); // empty — below threshold, must NOT split
    els.push(text_para("still-same"));
    let ir = DocumentIR {
        sections: vec![Section {
            elements: els,
            ..Default::default()
        }],
        ..Default::default()
    };
    let chunks = chunk_ir_with_threshold(Path::new("a.docx"), &ir, "docx", 10_000);
    assert_eq!(chunks.len(), 1);

    let mut els = Vec::new();
    els.push(text_para(&"a".repeat(30)));
    els.push(text_para("")); // over threshold → split
    els.push(text_para("b".repeat(5)));
    let ir = DocumentIR {
        sections: vec![Section {
            elements: els,
            ..Default::default()
        }],
        ..Default::default()
    };
    let chunks = chunk_ir_with_threshold(Path::new("a.docx"), &ir, "docx", 20);
    assert!(chunks.len() >= 2, "got {}", chunks.len());
}

#[test]
fn glue_caption_and_note_around_table() {
    use office_oxide::ir::{Table, TableCell, TableRow};
    let table = Element::Table(Table {
        rows: vec![TableRow {
            cells: vec![TableCell {
                content: vec![text_para("1")],
                ..Default::default()
            }],
            ..Default::default()
        }],
        ..Default::default()
    });
    // Buffer already large; next is caption then table then note — must stay one chunk
    let ir = DocumentIR {
        sections: vec![Section {
            elements: vec![
                text_para(&"p".repeat(40)),
                text_para("Table 1 — caption"),
                table,
                text_para("Note under"),
                text_para("after-glue-can-split"),
            ],
            ..Default::default()
        }],
        ..Default::default()
    };
    let chunks = chunk_ir_with_threshold(Path::new("a.docx"), &ir, "docx", 30);
    let joined_first = &chunks[0].text;
    assert!(joined_first.contains("Table 1 — caption"), "{joined_first}");
    assert!(joined_first.contains("| 1 |") || joined_first.contains("1"), "{joined_first}");
    assert!(joined_first.contains("Note under"), "{joined_first}");
    // The paragraph after the glued note may be in chunk 0 or 1 depending on size;
    // assert note+caption+table co-occur in some single chunk:
    assert!(chunks.iter().any(|c| {
        c.text.contains("Table 1 — caption")
            && c.text.contains("Note under")
            && (c.text.contains('|') || c.text.contains('1'))
    }));
}
```

- [ ] **Step 2: Implement split/glue rules**

When considering whether to start a new chunk before `el`:

1. If `is_empty_paragraph(el)` and `render_elements(&buf).chars().count() >= threshold`: flush buf; **do not** push empty para; continue.
2. Else if `render_elements(&buf).chars().count() >= threshold`:
   - Let `last = buf.last()`;
   - If `last` is nonempty text/heading and `el` is table → **do not** split (caption glue).
   - If `last` is table and `el` is nonempty paragraph → **do not** split (note glue), but only if we have not already appended a note after this table (track `glued_note_after_table: bool`, set true when pushing that para, clear when pushing anything else after).
   - Else: flush, then push `el`.
3. Else: push `el`.

Hard heading split still runs first (before soft logic): flush, update path, push heading.

After soft flush on empty para, clear glue flags.

Oversized single table: if buf empty and el is table, push it even if over threshold after; never split inside table.

- [ ] **Step 3: PASS**

```powershell
cargo test --lib extract::office_chunk -- --nocapture
```

- [ ] **Step 4: Commit**

```powershell
git add src/extract/office_chunk.rs
git commit -m "feat(extract): office IR size split, empty-para, table glue"
```

---

### Task 5: Wire `OfficeExtractor` to IR pipeline

**Files:**
- Modify: `src/extract/office.rs`

- [ ] **Step 1: Switch extract path**

Replace markdown path:

```rust
use crate::extract::office_chunk::chunk_ir;
use crate::extract::office_table::expand_merged_tables;
use crate::extract::Extractor;
use crate::model::Chunk;
use anyhow::anyhow;
use office_oxide::{Document, DocumentFormat};
use std::io::Cursor;
use std::path::Path;

// ... format_for unchanged ...

fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let fmt = format_for(&ext).ok_or_else(|| anyhow!("unsupported office extension: {ext}"))?;
    let doc = Document::from_reader(Cursor::new(bytes.to_vec()), fmt)
        .map_err(|e| anyhow!("office parse failed for {}: {e}", path.display()))?;
    let mut ir = doc.to_ir();
    expand_merged_tables(&mut ir);
    Ok(chunk_ir(path, &ir, &ext))
}
```

Remove `use crate::extract::chunk::chunk_markdown` and any `to_markdown()` call.

- [ ] **Step 2: Run existing office fixture tests**

```powershell
cargo test --lib extract::office -- --nocapture
```

Expected: `extracts_text_from_docx_fixture`, `extracts_table_as_markdown`, `unsupported_extension_errors` PASS. If table fixture fails on pipe format, adjust `table_to_markdown` whitespace to still satisfy `contains('|') && contains("---")`.

- [ ] **Step 3: Commit**

```powershell
git add src/extract/office.rs
git commit -m "feat(extract): office indexing via IR expand and chunk"
```

---

### Task 6: Docs touch-up + full lib test

**Files:**
- Modify: `docs/architecture.md` (Office row — one sentence: IR + merge expand + IR chunking)
- Modify: `docs/office-ir-chunking-design.md` — set **Status:** Implemented (or leave until eval)

- [ ] **Step 1: Update architecture Office line**

In `docs/architecture.md` table, change Office cell to mention DocumentIR, merge densify, IR-level chunks (not whole-doc markdown).

- [ ] **Step 2: Full test**

```powershell
cargo test --lib -- --nocapture
```

Expected: all lib tests pass.

- [ ] **Step 3: Commit**

```powershell
git add docs/architecture.md docs/office-ir-chunking-design.md
git commit -m "docs: note office IR extraction path"
```

---

### Task 7 (optional follow-up, not blocking): Eval harness

Manual / existing small-model docx eval: target-parameter retrieval 0/3→3/3; dependent table values intact. No code in this plan — operator run after merge.

---

## Spec coverage check

| Spec requirement | Task |
|------------------|------|
| All office via `to_ir()` | 5 |
| Merge expand H+V | 1 |
| IR chunking, no corpus-specific heuristics | 3–4 |
| Empty para soft split post-threshold | 4 |
| Caption + note glue | 4 |
| Threshold 4000 | 4 (`CHUNK_CHAR_THRESHOLD`) |
| Section / heading hard splits | 3 |
| GFM chunk text | 2 |
| No whole-doc `to_markdown` | 5 |
| Fixture + unit tests | 1, 3–5 |
| PDF / md unchanged | (no tasks touch them) |

## Placeholder scan

None intentional. `todo!` only in red→green steps before implementation in the same task.
