use crate::extract::office_chunk::chunk_ir;
use crate::extract::office_table::expand_merged_tables;
use crate::extract::Extractor;
use crate::model::Chunk;
use anyhow::{anyhow, Context};
use office_oxide::ir::{
    DocumentIR, Element, Heading, InlineContent, Paragraph, Section, Table, TableCell, TableRow,
    TextSpan,
};
use quick_xml::events::Event;
use quick_xml::Reader;
use std::io::{Cursor, Read};
use std::path::Path;

pub struct OdfExtractor;

/// Cap on `number-columns-repeated` / `number-rows-repeated` / `number-columns-spanned`
/// expansion. Real LibreOffice output tops out around ~1024; this is well above that
/// but still bounds the allocation a malformed/adversarial file can trigger (glossa
/// indexes arbitrary user files).
const MAX_REPEAT: usize = 4096;

/// Cap on the total number of cells a single table may materialize (summed
/// across all rows, after row/column repeat expansion). `MAX_REPEAT` bounds a
/// single dimension (one row's column-repeat, or one row's repeat count); this
/// bounds the PRODUCT across many rows, so an adversarial/malformed file can't
/// combine many largish-but-under-cap repeats into an unbounded allocation.
const MAX_TABLE_CELLS: usize = 1_000_000;

/// Parse a `number-*` repeat/span attribute, defaulting to 1 and clamping to
/// `MAX_REPEAT`. Clamping is logged (never silent) so a truncated expansion is
/// discoverable rather than looking like a parser bug.
fn bounded_repeat(raw: Option<String>, attr_name: &str, ext: &str) -> usize {
    let n = raw.and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).max(1);
    if n > MAX_REPEAT {
        tracing::warn!(
            "odf {attr_name}={n} exceeds cap {MAX_REPEAT} in .{ext} file; clamping"
        );
        MAX_REPEAT
    } else {
        n
    }
}

fn read_content_xml(bytes: &[u8], path: &Path) -> anyhow::Result<String> {
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes))
        .with_context(|| format!("odf not a zip: {}", path.display()))?;
    let mut f = zip
        .by_name("content.xml")
        .with_context(|| format!("odf missing content.xml: {}", path.display()))?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    Ok(s)
}

fn local(name: &[u8]) -> &[u8] {
    match name.iter().position(|&b| b == b':') {
        Some(i) => &name[i + 1..],
        None => name,
    }
}

fn attr<'a>(e: &'a quick_xml::events::BytesStart, want: &[u8]) -> Option<String> {
    e.attributes().flatten().find_map(|a| {
        if local(a.key.as_ref()) == want {
            std::str::from_utf8(&a.value).ok().map(str::to_string)
        } else {
            None
        }
    })
}

/// Map an ODF paragraph style name to a heading level, if it is a heading style.
/// LibreOffice/converted docs use `Heading_20_N` (encoded space); some use
/// `Heading N`; some (no separator at all) use `Heading1`..`Heading6`.
fn heading_level_from_style(style: &str) -> Option<u8> {
    let norm = style.replace("_20_", " ");
    let rest = norm
        .strip_prefix("Heading ")
        .or_else(|| norm.strip_prefix("Heading"))?;
    rest.trim().parse::<u8>().ok().filter(|n| (1..=6).contains(n))
}

fn para(text: String) -> Element {
    Element::Paragraph(Paragraph {
        content: vec![InlineContent::Text(TextSpan { text, ..Default::default() })],
        ..Default::default()
    })
}

fn heading(level: u8, text: String) -> Element {
    Element::Heading(Heading {
        level,
        content: vec![InlineContent::Text(TextSpan { text, ..Default::default() })],
        ..Default::default()
    })
}

/// Push `repeat` copies of `cell`. Empty single-column cells are deferred into
/// `empty_run` (so trailing empties can be clamped by the caller); a non-empty
/// cell first flushes any pending empties.
fn push_cells(out: &mut Vec<TableCell>, empty_run: &mut usize, cell: TableCell, is_empty: bool, repeat: usize) {
    if is_empty && cell.col_span <= 1 {
        *empty_run += repeat;
        return;
    }
    for _ in 0..*empty_run {
        out.push(TableCell { content: vec![para(String::new())], ..Default::default() });
    }
    *empty_run = 0;
    for _ in 0..repeat {
        out.push(cell.clone());
    }
}

/// A row is "empty" (deferrable/clampable, like `number-rows-repeated` trailing
/// blank rows) when every cell's text content is blank.
fn row_is_empty(cells: &[TableCell]) -> bool {
    cells.iter().all(|c| {
        c.content.iter().all(|el| match el {
            Element::Paragraph(p) => p.content.iter().all(|ic| match ic {
                InlineContent::Text(t) => t.text.trim().is_empty(),
                _ => false,
            }),
            _ => false,
        })
    })
}

fn parse_to_ir(xml: &str, ext: &str) -> anyhow::Result<DocumentIR> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(false);
    // Deliver every self-closing element (<table:table-cell .../>, <text:tab/>,
    // <text:line-break/>, ...) as a Start immediately followed by an End, instead
    // of a separate Event::Empty. This gives self-closing and "open ... close"
    // elements ONE shared code path: an interior self-closing empty table-cell
    // now flows through the exact same Start/End handling (and push_cells'
    // empty-run deferral) as a real `<table:table-cell></table:table-cell>`,
    // instead of being silently skipped and misaligning every later column.
    reader.config_mut().expand_empty_elements = true;

    let mut sections: Vec<Section> = Vec::new(); // ODS: one per <table:table>; ODT: single, pushed at EOF
    let mut elements: Vec<Element> = Vec::new();
    let mut buf = String::new();          // active paragraph/heading text
    let mut pending_heading: Option<u8> = None; // Some(level) while inside a heading-ish block
    let mut in_table = false; // derived: table_depth > 0
    let mut table_depth: usize = 0; // nesting depth of <table:table>; only 0<->1 starts/emits
    let mut current_sheet: Option<String> = None; // ODS: table:name of the outermost table in progress
    let mut rows: Vec<TableRow> = Vec::new();
    let mut cur_cells: Vec<TableCell> = Vec::new();
    let mut cell_span: usize = 1;
    let mut cell_row_span: usize = 1; // number-rows-spanned, stashed at cell Start
    let mut cell_repeat: usize = 1; // number-columns-repeated, stashed at cell Start
    let mut empty_run: usize = 0; // deferred trailing-empty cells (ODS repeats; harmless for ODT)
    let mut row_repeat: usize = 1; // number-rows-repeated, stashed at row Start
    let mut pending_empty_rows: usize = 0; // deferred trailing-empty rows (ODS only; clamped like empty_run)
    let mut table_cell_count: usize = 0; // running total of cells materialized in the current outer table
    let mut table_cell_budget_warned = false; // emit the MAX_TABLE_CELLS warning at most once per table
    // Did this row contain a REAL <table:table-cell>? Deliberately excludes
    // <table:covered-table-cell> — a covered-only row (the continuation of a
    // vertical merge) must stay a genuine 0-cell TableRow, since expand_table
    // (office_table.rs) fills those grid slots purely from the origin cell's
    // row_span; padding a phantom blank cell here would widen the whole table.
    let mut row_has_cell = false;
    let mut ev = Vec::new();

    loop {
        match reader.read_event_into(&mut ev) {
            Err(e) => { tracing::warn!("odf xml error ({ext}): {e}"); break; }
            Ok(Event::Eof) => break,
            Ok(Event::Text(t)) => {
                if let Ok(decoded) = t.decode() {
                    match quick_xml::escape::unescape(&decoded) {
                        Ok(s) => buf.push_str(&s),
                        Err(_) => buf.push_str(&decoded),
                    }
                }
            }
            Ok(Event::Start(e)) => match local(e.name().as_ref()) {
                b"h" => {
                    if in_table {
                        // Same rule as <text:p>: a heading inside a cell just adds
                        // more cell text, it must not wipe out prior cell content.
                        if !buf.is_empty() {
                            buf.push(' ');
                        }
                    } else {
                        buf.clear();
                    }
                    pending_heading = Some(
                        attr(&e, b"outline-level").and_then(|v| v.parse().ok()).unwrap_or(1),
                    );
                }
                b"p" => {
                    if in_table {
                        // Table cells may hold multiple <text:p> — accumulate across
                        // all of them (the table-cell Start already reset buf), with
                        // a space separator so words don't run together.
                        if !buf.is_empty() {
                            buf.push(' ');
                        }
                    } else {
                        buf.clear();
                    }
                    pending_heading = attr(&e, b"style-name")
                        .as_deref()
                        .and_then(heading_level_from_style);
                }
                b"tab" | b"line-break" => buf.push(' '),
                b"s" => {
                    // <text:s text:c="N"/> is a run of N spaces (default 1 when
                    // the attribute is absent). Cap it small so a malformed/
                    // adversarial count can't push an unbounded String.
                    let count = attr(&e, b"c")
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(1)
                        .max(1)
                        .min(1000);
                    for _ in 0..count {
                        buf.push(' ');
                    }
                }
                b"table" => {
                    // Nested <table:table> (a table inside a cell) must NOT reset the
                    // outer table's state — only the outermost table's own open/close
                    // (depth 0<->1) starts/emits. Deeper opens/closes are no-ops here;
                    // their rows/cells fold into the same `rows`/`cur_cells` state as
                    // the outer table (lossy but not corrupting — no wiped outer rows,
                    // no cells leaking to document level, no panic).
                    table_depth += 1;
                    if table_depth == 1 {
                        rows.clear();
                        pending_empty_rows = 0;
                        table_cell_count = 0;
                        table_cell_budget_warned = false;
                        if ext == "ods" {
                            current_sheet = attr(&e, b"name");
                        }
                    }
                    in_table = table_depth > 0;
                }
                b"table-row" => {
                    cur_cells = Vec::new();
                    empty_run = 0;
                    row_has_cell = false;
                    row_repeat = bounded_repeat(
                        attr(&e, b"number-rows-repeated"),
                        "number-rows-repeated",
                        ext,
                    );
                }
                b"table-cell" => {
                    buf.clear();
                    row_has_cell = true;
                    cell_span = bounded_repeat(
                        attr(&e, b"number-columns-spanned"),
                        "number-columns-spanned",
                        ext,
                    );
                    cell_row_span = bounded_repeat(
                        attr(&e, b"number-rows-spanned"),
                        "number-rows-spanned",
                        ext,
                    );
                    cell_repeat = bounded_repeat(
                        attr(&e, b"number-columns-repeated"),
                        "number-columns-repeated",
                        ext,
                    );
                }
                _ => {}
            },
            // No Event::Empty arm: `expand_empty_elements = true` (set above) means
            // every self-closing element (<table:table-cell/>, <text:tab/>, ...)
            // arrives as Start immediately followed by End instead — one shared
            // code path, handled by the Start/End arms above/below.
            Ok(Event::End(e)) => match local(e.name().as_ref()) {
                b"h" | b"p" => {
                    if !in_table {
                        let text = buf.trim().to_string();
                        buf.clear();
                        if !text.is_empty() {
                            match pending_heading.take() {
                                Some(level) => elements.push(heading(level, text)),
                                None => elements.push(para(text)),
                            }
                        } else {
                            pending_heading = None;
                        }
                    }
                }
                b"table-cell" => {
                    let text = buf.trim().to_string();
                    buf.clear();
                    let repeat = cell_repeat;
                    let is_empty = text.is_empty();
                    let cell = TableCell {
                        content: vec![para(text)],
                        col_span: cell_span as u32,
                        row_span: cell_row_span as u32,
                        ..Default::default()
                    };
                    push_cells(&mut cur_cells, &mut empty_run, cell, is_empty, repeat);
                    cell_span = 1;
                    cell_row_span = 1;
                    cell_repeat = 1;
                }
                b"covered-table-cell" => { /* span already consumed the column; skip */ }
                b"table-row" => {
                    empty_run = 0; // drop trailing empty cells within the row (clamp)
                    let mut cells = std::mem::take(&mut cur_cells);
                    if ext == "ods" && row_is_empty(&cells) {
                        // Defer: number-rows-repeated (or a bare empty row) may just be
                        // trailing padding — only materialize it if content follows.
                        // ODS only: number-rows-repeated is a spreadsheet-only attribute,
                        // and ODT tables should never silently drop a real row.
                        pending_empty_rows += row_repeat;
                    } else {
                        if cells.is_empty() && row_has_cell {
                            // The row did contain real <table:table-cell> elements, but
                            // column-level clamping (all-empty single-column cells
                            // deferred into `empty_run`) dropped every one of them —
                            // keep the row as one valid blank GFM cell rather than a
                            // zero-cell row. row_has_cell is NOT set by covered-table-cell,
                            // so a covered-only vertical-merge continuation row stays a
                            // genuine 0-cell TableRow (expand_table in office_table.rs
                            // fills those grid slots purely from the origin's row_span).
                            cells.push(TableCell { content: vec![para(String::new())], ..Default::default() });
                        }
                        // Budget guard: MAX_REPEAT already bounds a single row/column
                        // repeat attribute, but many rows each near that cap — or a
                        // long chain of deferred-empty padding rows flushed here —
                        // can still multiply into an unbounded allocation. Every
                        // materialized row, real OR 0-cell padding, costs at least 1
                        // against the table's cell budget: padding rows can't slip
                        // the cap just because they have 0 cells of their own (each
                        // flush previously got its own uncounted allowance).
                        let row_cost = cells.len().max(1);
                        let mut over_budget = false;
                        for _ in 0..pending_empty_rows {
                            if table_cell_count >= MAX_TABLE_CELLS {
                                over_budget = true;
                                break;
                            }
                            table_cell_count += 1;
                            rows.push(TableRow::default());
                        }
                        pending_empty_rows = 0;
                        if !over_budget {
                            for _ in 0..row_repeat {
                                if table_cell_count.saturating_add(row_cost) > MAX_TABLE_CELLS {
                                    over_budget = true;
                                    break;
                                }
                                table_cell_count += row_cost;
                                rows.push(TableRow { cells: cells.clone(), ..Default::default() });
                            }
                        }
                        if over_budget && !table_cell_budget_warned {
                            tracing::warn!(
                                "odf .{ext}: table exceeds {MAX_TABLE_CELLS}-cell budget, dropping further rows"
                            );
                            table_cell_budget_warned = true;
                        }
                    }
                    row_repeat = 1;
                }
                b"table" => {
                    // Only the outermost table's close (depth 1->0) emits an Element.
                    // A nested table's close just steps back up a level; its rows/cells
                    // already folded into `rows` above (see the Start arm's comment).
                    table_depth = table_depth.saturating_sub(1);
                    in_table = table_depth > 0;
                    if table_depth == 0 {
                        pending_empty_rows = 0; // drop trailing empty rows (clamp)
                        let table = Element::Table(Table { rows: std::mem::take(&mut rows), ..Default::default() });
                        if ext == "ods" {
                            sections.push(Section { title: current_sheet.take(), elements: vec![table], ..Default::default() });
                        } else {
                            elements.push(table);
                        }
                    }
                }
                _ => {}
            },
            _ => {}
        }
        ev.clear();
    }

    // Flush a pending in-progress table (truncated/malformed XML — EOF or an XML
    // error hit mid-table) instead of silently dropping accumulated rows.
    if in_table && (!rows.is_empty() || !cur_cells.is_empty()) {
        if !cur_cells.is_empty() {
            rows.push(TableRow { cells: std::mem::take(&mut cur_cells), ..Default::default() });
        }
        tracing::warn!(
            "odf .{ext}: unterminated <table:table>, flushing {} pending row(s)",
            rows.len()
        );
        let table = Element::Table(Table { rows: std::mem::take(&mut rows), ..Default::default() });
        if ext == "ods" {
            sections.push(Section { title: current_sheet.take(), elements: vec![table], ..Default::default() });
        } else {
            elements.push(table);
        }
    } else if !in_table {
        // Flush a pending trailing paragraph/heading that never saw its closing tag.
        let text = buf.trim().to_string();
        if !text.is_empty() {
            match pending_heading.take() {
                Some(level) => elements.push(heading(level, text)),
                None => elements.push(para(text)),
            }
        }
    }

    if ext != "ods" {
        // ODT: single section, heading-scoped chunking happens downstream in chunk_ir.
        sections.push(Section { elements, ..Default::default() });
    }
    Ok(DocumentIR { sections, ..Default::default() })
}

impl Extractor for OdfExtractor {
    fn file_types(&self) -> &'static [&'static str] {
        &["odt", "ods"]
    }

    fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        let xml = read_content_xml(bytes, path)?;
        let mut ir = parse_to_ir(&xml, &ext)?;
        expand_merged_tables(&mut ir);
        let chunks = chunk_ir(path, &ir, &ext);
        if chunks.is_empty() {
            return Err(anyhow!("odf produced no chunks for {}", path.display()));
        }
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ODT: &[u8] = include_bytes!("../../tests/fixtures/sample.odt");

    fn extract(bytes: &[u8], name: &str) -> Vec<Chunk> {
        OdfExtractor.extract(Path::new(name), bytes).unwrap()
    }

    /// Build a minimal in-memory ODF zip (just `content.xml`) for tests that
    /// need to go through the full `Extractor::extract` path (read_content_xml
    /// -> parse_to_ir -> expand_merged_tables -> chunk_ir), not just parse_to_ir.
    fn make_odf_zip(content_xml: &str) -> Vec<u8> {
        use std::io::Write as _;
        let mut buf = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zw = zip::ZipWriter::new(cursor);
            let opts = zip::write::SimpleFileOptions::default();
            zw.start_file("content.xml", opts).unwrap();
            zw.write_all(content_xml.as_bytes()).unwrap();
            zw.finish().unwrap();
        }
        buf
    }

    /// Flatten a TableCell's paragraph text runs into one string.
    fn cell_text(cell: &TableCell) -> String {
        cell.content
            .iter()
            .map(|el| match el {
                Element::Paragraph(p) => p
                    .content
                    .iter()
                    .map(|c| match c {
                        InlineContent::Text(t) => t.text.clone(),
                        _ => String::new(),
                    })
                    .collect::<String>(),
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn find_table(ir: &DocumentIR) -> &Table {
        ir.sections[0]
            .elements
            .iter()
            .find_map(|el| match el {
                Element::Table(t) => Some(t),
                _ => None,
            })
            .expect("expected a table element in the parsed IR")
    }

    #[test]
    fn odt_marker_and_both_heading_conventions() {
        let chunks = extract(ODT, "sample.odt");
        let all_text: String = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n");
        assert!(all_text.contains("glossa sample"), "marker missing: {all_text}");
        assert!(chunks.iter().all(|c| c.file_type == "odt"));
        // structural <text:h>
        assert!(chunks.iter().any(|c| c.location.contains("Alpha Section")),
            "structural heading not a location: {:?}", chunks.iter().map(|c| &c.location).collect::<Vec<_>>());
        // styled paragraph Heading_20_1 — the real-world quirk
        assert!(chunks.iter().any(|c| c.location.contains("Beta Section")),
            "styled-paragraph heading not recognised: {:?}", chunks.iter().map(|c| &c.location).collect::<Vec<_>>());
    }

    #[test]
    fn odt_table_renders_gfm_with_spanned_cell() {
        let chunks = extract(ODT, "sample.odt");
        let t: String = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n");
        assert!(t.contains("| h1 | h2 |") && t.contains("---"), "GFM header row missing:\n{t}");
        // number-columns-spanned=2 → expand_merged_tables duplicates the origin value
        // into the covered cell (see office_table.rs densified_cell/expand_table).
        assert!(t.contains("| wide | wide |"), "spanned cell not expanded to 2 columns:\n{t}");
    }

    #[test]
    fn odt_table_cell_multi_paragraph_keeps_all_text() {
        let xml = r#"<?xml version='1.0' encoding='UTF-8'?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
<office:body><office:text>
<table:table>
<table:table-row>
<table:table-cell><text:p>foo</text:p><text:p>bar</text:p></table:table-cell>
</table:table-row>
</table:table>
</office:text></office:body>
</office:document-content>"#;

        let ir = parse_to_ir(xml, "odt").unwrap();
        let table = ir.sections[0]
            .elements
            .iter()
            .find_map(|el| match el {
                Element::Table(t) => Some(t),
                _ => None,
            })
            .expect("expected a table element in the parsed IR");

        let cell_text: String = table.rows[0].cells[0]
            .content
            .iter()
            .map(|el| match el {
                Element::Paragraph(p) => p
                    .content
                    .iter()
                    .map(|c| match c {
                        InlineContent::Text(t) => t.text.clone(),
                        _ => String::new(),
                    })
                    .collect::<String>(),
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join(" ");

        assert!(cell_text.contains("foo"), "first paragraph lost, cell text: {cell_text:?}");
        assert!(cell_text.contains("bar"), "second paragraph lost, cell text: {cell_text:?}");
    }

    const ODS: &[u8] = include_bytes!("../../tests/fixtures/sample.ods");

    #[test]
    fn ods_one_chunk_per_sheet_with_names() {
        let chunks = extract(ODS, "sample.ods");
        assert!(chunks.iter().all(|c| c.file_type == "ods"));
        let locs: Vec<&str> = chunks.iter().map(|c| c.location.as_str()).collect();
        assert!(locs.iter().any(|l| l.contains("Sheet1")), "locs: {locs:?}");
        assert!(locs.iter().any(|l| l.contains("Data")), "locs: {locs:?}");
    }

    #[test]
    fn ods_repeats_expand_and_clamp() {
        let chunks = extract(ODS, "sample.ods");
        let data = chunks.iter().find(|c| c.location.contains("Data")).unwrap();
        // number-columns-repeated=3 on "dup" → three dup columns between x and y
        assert!(data.text.contains("| x | dup | dup | dup | y |"),
            "repeated non-empty cell not expanded:\n{}", data.text);
        let sheet1 = chunks.iter().find(|c| c.location.contains("Sheet1")).unwrap();
        // trailing repeated empty cells (repeated=6) must be clamped, not 6+ empty columns
        let widest = sheet1.text.lines().map(|l| l.matches('|').count()).max().unwrap_or(0);
        assert!(widest <= 4, "trailing empties not clamped (cols≈{}):\n{}", widest, sheet1.text);
    }

    #[test]
    fn ods_row_repeated_expands_and_trailing_empty_clamped() {
        let xml = r#"<?xml version='1.0' encoding='UTF-8'?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
<office:body><office:spreadsheet>
<table:table table:name="Sheet1">
<table:table-row table:number-rows-repeated="3">
<table:table-cell><text:p>rowval</text:p></table:table-cell>
</table:table-row>
<table:table-row table:number-rows-repeated="1000">
<table:table-cell/>
</table:table-row>
</table:table>
</office:spreadsheet></office:body>
</office:document-content>"#;

        let ir = parse_to_ir(xml, "ods").unwrap();
        let table = ir.sections[0]
            .elements
            .iter()
            .find_map(|el| match el {
                Element::Table(t) => Some(t),
                _ => None,
            })
            .expect("expected a table element in the parsed IR");

        // number-rows-repeated=3 on a non-empty row → 3 identical rows; the
        // trailing number-rows-repeated=1000 EMPTY row must be clamped away
        // entirely (not materialized as 1000 extra blank rows).
        assert_eq!(table.rows.len(), 3, "expected 3 expanded rows (empty trailing row not clamped), got {}", table.rows.len());
        for row in &table.rows {
            let text: String = row.cells[0]
                .content
                .iter()
                .map(|el| match el {
                    Element::Paragraph(p) => p
                        .content
                        .iter()
                        .map(|c| match c {
                            InlineContent::Text(t) => t.text.clone(),
                            _ => String::new(),
                        })
                        .collect::<String>(),
                    _ => String::new(),
                })
                .collect::<String>();
            assert_eq!(text, "rowval", "expanded row content mismatch");
        }
    }

    #[test]
    fn odt_unterminated_table_flushed_on_eof() {
        // Truncated file: one full row (own open+close tags) inside an open
        // <table:table> that never gets a closing </table:table>, and the rest
        // of the document (office:text/body/document-content) is also missing.
        let xml = r#"<?xml version='1.0' encoding='UTF-8'?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
<office:body><office:text>
<table:table>
<table:table-row>
<table:table-cell><text:p>truncated</text:p></table:table-cell>
</table:table-row>"#;

        let ir = parse_to_ir(xml, "odt").unwrap();
        let table = ir.sections[0]
            .elements
            .iter()
            .find_map(|el| match el {
                Element::Table(t) => Some(t),
                _ => None,
            })
            .expect("unterminated table should still be flushed into the IR, not dropped");

        let cell_text: String = table.rows[0].cells[0]
            .content
            .iter()
            .map(|el| match el {
                Element::Paragraph(p) => p
                    .content
                    .iter()
                    .map(|c| match c {
                        InlineContent::Text(t) => t.text.clone(),
                        _ => String::new(),
                    })
                    .collect::<String>(),
                _ => String::new(),
            })
            .collect::<String>();

        assert!(cell_text.contains("truncated"), "unterminated table row lost, cell text: {cell_text:?}");
    }

    #[test]
    fn odt_table_cell_heading_does_not_wipe_prior_text() {
        let xml = r#"<?xml version='1.0' encoding='UTF-8'?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
<office:body><office:text>
<table:table>
<table:table-row>
<table:table-cell><text:p>starttext</text:p><text:h text:outline-level="1">headtext</text:h></table:table-cell>
</table:table-row>
</table:table>
</office:text></office:body>
</office:document-content>"#;

        let ir = parse_to_ir(xml, "odt").unwrap();
        let table = ir.sections[0]
            .elements
            .iter()
            .find_map(|el| match el {
                Element::Table(t) => Some(t),
                _ => None,
            })
            .expect("expected a table element in the parsed IR");

        let cell_text: String = table.rows[0].cells[0]
            .content
            .iter()
            .map(|el| match el {
                Element::Paragraph(p) => p
                    .content
                    .iter()
                    .map(|c| match c {
                        InlineContent::Text(t) => t.text.clone(),
                        _ => String::new(),
                    })
                    .collect::<String>(),
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join(" ");

        assert!(cell_text.contains("starttext"), "text before the in-cell heading was wiped: {cell_text:?}");
        assert!(cell_text.contains("headtext"), "in-cell heading text lost: {cell_text:?}");
    }

    #[test]
    fn odt_trailing_empty_row_kept_with_one_blank_cell() {
        // FIX 3: ODT rows are never deferred/dropped as "trailing empty" (that
        // clamp is ODS-only, since number-rows-repeated is spreadsheet-only).
        // FIX 4: a row that did have a <table:table-cell> but ended up with zero
        // cells (column-level empty_run clamp) must render as one blank cell,
        // not a zero-cell row.
        let xml = r#"<?xml version='1.0' encoding='UTF-8'?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
<office:body><office:text>
<table:table>
<table:table-row>
<table:table-cell><text:p>real</text:p></table:table-cell>
</table:table-row>
<table:table-row>
<table:table-cell><text:p></text:p></table:table-cell>
</table:table-row>
</table:table>
</office:text></office:body>
</office:document-content>"#;

        let ir = parse_to_ir(xml, "odt").unwrap();
        let table = ir.sections[0]
            .elements
            .iter()
            .find_map(|el| match el {
                Element::Table(t) => Some(t),
                _ => None,
            })
            .expect("expected a table element in the parsed IR");

        assert_eq!(table.rows.len(), 2, "ODT trailing empty row must not be dropped/deferred (FIX 3)");
        assert_eq!(
            table.rows[1].cells.len(),
            1,
            "all-empty row collapsed to zero cells instead of one blank cell (FIX 4)"
        );
    }

    #[test]
    fn ods_number_columns_repeated_clamped_to_max_repeat() {
        let xml = r#"<?xml version='1.0' encoding='UTF-8'?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
<office:body><office:spreadsheet>
<table:table table:name="Huge">
<table:table-row>
<table:table-cell office:value-type="string" table:number-columns-repeated="100000"><text:p>dup</text:p></table:table-cell>
</table:table-row>
</table:table>
</office:spreadsheet></office:body>
</office:document-content>"#;

        let ir = parse_to_ir(xml, "ods").unwrap();
        let table = ir.sections[0]
            .elements
            .iter()
            .find_map(|el| match el {
                Element::Table(t) => Some(t),
                _ => None,
            })
            .expect("expected a table element in the parsed IR");

        assert_eq!(
            table.rows[0].cells.len(),
            MAX_REPEAT,
            "number-columns-repeated=100000 should clamp to MAX_REPEAT ({MAX_REPEAT}), got {}",
            table.rows[0].cells.len()
        );
    }

    #[test]
    fn ods_number_rows_spanned_sets_cell_row_span() {
        let xml = r#"<?xml version='1.0' encoding='UTF-8'?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
<office:body><office:spreadsheet>
<table:table table:name="Merged">
<table:table-row>
<table:table-cell table:number-rows-spanned="3"><text:p>tall</text:p></table:table-cell>
</table:table-row>
</table:table>
</office:spreadsheet></office:body>
</office:document-content>"#;

        let ir = parse_to_ir(xml, "ods").unwrap();
        let table = ir.sections[0]
            .elements
            .iter()
            .find_map(|el| match el {
                Element::Table(t) => Some(t),
                _ => None,
            })
            .expect("expected a table element in the parsed IR");

        assert_eq!(table.rows[0].cells[0].row_span, 3, "number-rows-spanned not carried onto TableCell.row_span");
    }

    #[test]
    fn odt_vertical_merge_covered_row_stays_zero_cells_no_grid_widen() {
        // Regression test (FIX 4 x FIX 6 interaction): row1 has ONE real cell
        // with number-rows-spanned="2"; row2 is a covered-only continuation
        // row (<table:covered-table-cell/>, no real <table:table-cell>). FIX
        // 4's blank-cell padding must NOT fire for row2 — otherwise
        // expand_merged_tables treats the phantom cell as a real placement and
        // widens the WHOLE table to 2 columns instead of the correct 1.
        let xml = r#"<?xml version='1.0' encoding='UTF-8'?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
<office:body><office:text>
<table:table>
<table:table-row>
<table:table-cell table:number-rows-spanned="2"><text:p>tall</text:p></table:table-cell>
</table:table-row>
<table:table-row>
<table:covered-table-cell/>
</table:table-row>
</table:table>
</office:text></office:body>
</office:document-content>"#;

        let mut ir = parse_to_ir(xml, "odt").unwrap();
        let table_before = ir.sections[0]
            .elements
            .iter()
            .find_map(|el| match el {
                Element::Table(t) => Some(t),
                _ => None,
            })
            .expect("expected a table element in the parsed IR");
        // Pre-expand: the covered-only row must stay a genuine 0-cell row —
        // no phantom blank cell padded in for it.
        assert_eq!(
            table_before.rows[1].cells.len(),
            0,
            "covered-only continuation row should stay 0-cell before expand_merged_tables, got {} cells",
            table_before.rows[1].cells.len()
        );

        expand_merged_tables(&mut ir);
        let table_after = ir.sections[0]
            .elements
            .iter()
            .find_map(|el| match el {
                Element::Table(t) => Some(t),
                _ => None,
            })
            .expect("expected a table element after expand_merged_tables");

        assert_eq!(table_after.rows.len(), 2, "both rows should survive expand_merged_tables");
        for (i, row) in table_after.rows.iter().enumerate() {
            assert_eq!(
                row.cells.len(),
                1,
                "row {i} should stay 1 column wide after expand_merged_tables, got {} cells (table widened by phantom cell)",
                row.cells.len()
            );
        }
    }

    // FIX A (CRITICAL): a self-closing interior empty <table:table-cell/> used to
    // fall to `_ => {}` in the old Event::Empty arm and vanish entirely (never
    // reaching push_cells), silently misaligning every later column. Fixed via
    // `expand_empty_elements = true` — self-closing cells now flow through the
    // same Start/End path as explicit `<table:table-cell></table:table-cell>`.

    #[test]
    fn odt_interior_self_closing_empty_cell_preserves_column_count() {
        let xml = r#"<?xml version='1.0' encoding='UTF-8'?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
<office:body><office:text>
<table:table>
<table:table-row>
<table:table-cell><text:p>A</text:p></table:table-cell>
<table:table-cell/>
<table:table-cell><text:p>B</text:p></table:table-cell>
</table:table-row>
</table:table>
</office:text></office:body>
</office:document-content>"#;

        let ir = parse_to_ir(xml, "odt").unwrap();
        let table = find_table(&ir);

        assert_eq!(
            table.rows[0].cells.len(),
            3,
            "interior self-closing empty cell dropped a column, got {} cells",
            table.rows[0].cells.len()
        );
        assert_eq!(cell_text(&table.rows[0].cells[0]), "A");
        assert_eq!(cell_text(&table.rows[0].cells[1]), "", "interior empty cell should stay a real (empty) cell");
        assert_eq!(cell_text(&table.rows[0].cells[2]), "B", "B landed under the wrong column after the dropped cell");
    }

    #[test]
    fn ods_interior_self_closing_empty_cell_preserves_column_count() {
        let xml = r#"<?xml version='1.0' encoding='UTF-8'?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
<office:body><office:spreadsheet>
<table:table table:name="Sheet1">
<table:table-row>
<table:table-cell><text:p>A</text:p></table:table-cell>
<table:table-cell/>
<table:table-cell><text:p>B</text:p></table:table-cell>
</table:table-row>
</table:table>
</office:spreadsheet></office:body>
</office:document-content>"#;

        let ir = parse_to_ir(xml, "ods").unwrap();
        let table = find_table(&ir);

        assert_eq!(
            table.rows[0].cells.len(),
            3,
            "interior self-closing empty cell dropped a column, got {} cells",
            table.rows[0].cells.len()
        );
        assert_eq!(cell_text(&table.rows[0].cells[0]), "A");
        assert_eq!(cell_text(&table.rows[0].cells[1]), "", "interior empty cell should stay a real (empty) cell");
        assert_eq!(cell_text(&table.rows[0].cells[2]), "B", "B landed under the wrong column after the dropped cell");
    }

    #[test]
    fn odt_nested_table_no_panic_outer_cell_text_survives() {
        // FIX B (IMPORTANT): a bare `bool in_table` meant an inner <table:table>
        // (nested inside an outer table's cell) ran `rows.clear()` on Start and
        // `in_table = false` on End — wiping the outer table's already-closed
        // rows and leaking the outer table's remaining cells out to document
        // level. Fixed with a `table_depth` counter: only the 0->1 transition
        // resets/starts, only the 1->0 transition emits. Inner rows/cells
        // folding into the outer table's row list is an accepted trade-off; what
        // must not happen is a wiped table, leaked cells, or a panic.
        let xml = r#"<?xml version='1.0' encoding='UTF-8'?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
<office:body><office:text>
<table:table>
<table:table-row>
<table:table-cell>
<table:table>
<table:table-row>
<table:table-cell><text:p>innertext</text:p></table:table-cell>
</table:table-row>
</table:table>
<text:p>outertext</text:p>
</table:table-cell>
</table:table-row>
</table:table>
</office:text></office:body>
</office:document-content>"#;

        let ir = parse_to_ir(xml, "odt").unwrap(); // must not panic

        let table = find_table(&ir);
        let all_text: String = table
            .rows
            .iter()
            .flat_map(|r| r.cells.iter())
            .map(cell_text)
            .collect::<Vec<_>>()
            .join(" ");

        assert!(all_text.contains("outertext"), "outer cell's own text lost after nested table: {all_text:?}");
        // No document-level leak: everything folded into the single table element.
        assert!(
            ir.sections[0].elements.iter().all(|el| matches!(el, Element::Table(_))),
            "outer table cells leaked out to document level: {:?}",
            ir.sections[0].elements
        );
    }

    #[test]
    fn odt_line_break_inserts_space_not_concatenation() {
        // FIX C (MINOR): <text:line-break/> was unhandled, so "a<text:line-break/>b"
        // rendered as the run-together "ab" instead of "a b".
        let xml = r#"<?xml version='1.0' encoding='UTF-8'?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
<office:body><office:text>
<text:p>a<text:line-break/>b</text:p>
</office:text></office:body>
</office:document-content>"#;

        let ir = parse_to_ir(xml, "odt").unwrap();
        let text: String = ir.sections[0]
            .elements
            .iter()
            .map(|el| match el {
                Element::Paragraph(p) => p
                    .content
                    .iter()
                    .map(|c| match c {
                        InlineContent::Text(t) => t.text.clone(),
                        _ => String::new(),
                    })
                    .collect::<String>(),
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join(" ");

        assert!(text.contains("a b"), "line-break didn't insert a separating space: {text:?}");
        assert!(!text.contains("ab"), "line-break text ran together: {text:?}");
    }

    #[test]
    fn heading_style_no_separator_form_recognised() {
        // FIX E (MINOR): some producers write `Heading1`..`Heading6` with no
        // separator at all (vs. `Heading_20_N` or `Heading N`).
        assert_eq!(heading_level_from_style("Heading2"), Some(2));
        assert_eq!(heading_level_from_style("Heading 2"), Some(2), "space-separated form regressed");
        assert_eq!(heading_level_from_style("Heading_20_2"), Some(2), "encoded-space form regressed");
        assert_eq!(heading_level_from_style("HeadingSomethingElse"), None, "non-numeric suffix must not false-positive");
    }

    #[test]
    fn ods_merge_origin_row_span_exceeds_available_rows_no_panic() {
        // FIX P (CRITICAL) regression, odf-level: a merge origin's row_span can
        // end up exceeding the number of rows actually materialized once its
        // covered continuation row is dropped by the ODS-only trailing-empty-
        // row clamp (FIX 3, an earlier wave — deliberately left as-is; a
        // covered-only row is "empty" and, if trailing, gets clamped away).
        // Before FIX P this reached expand_table with row_span=3 on a 1-row
        // table and panicked (grid[r+dr] index out of bounds). Runs through
        // the FULL Extractor::extract path (real zip, expand_merged_tables,
        // chunk_ir) so the whole pipeline is exercised, not just parse_to_ir.
        let content_xml = r#"<?xml version='1.0' encoding='UTF-8'?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
<office:body><office:spreadsheet>
<table:table table:name="Sheet1">
<table:table-row>
<table:table-cell table:number-rows-spanned="3"><text:p>tall</text:p></table:table-cell>
</table:table-row>
<table:table-row>
<table:covered-table-cell/>
</table:table-row>
</table:table>
</office:spreadsheet></office:body>
</office:document-content>"#;

        let bytes = make_odf_zip(content_xml);
        let chunks = OdfExtractor.extract(Path::new("merge.ods"), &bytes).unwrap(); // must not panic
        assert!(!chunks.is_empty());
        let all_text: String = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n");
        assert!(all_text.contains("tall"), "merge origin text lost: {all_text}");
    }

    #[test]
    fn odt_text_s_repeat_count_expands_to_multiple_spaces() {
        // FIX R (MINOR): <text:s text:c="N"/> was treated as a single space,
        // ignoring its repeat count.
        let xml = r#"<?xml version='1.0' encoding='UTF-8'?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
<office:body><office:text>
<text:p>a<text:s text:c="3"/>b</text:p>
</office:text></office:body>
</office:document-content>"#;

        let ir = parse_to_ir(xml, "odt").unwrap();
        let text: String = ir.sections[0]
            .elements
            .iter()
            .map(|el| match el {
                Element::Paragraph(p) => p
                    .content
                    .iter()
                    .map(|c| match c {
                        InlineContent::Text(t) => t.text.clone(),
                        _ => String::new(),
                    })
                    .collect::<String>(),
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join(" ");

        assert!(text.contains("a   b"), "text:s repeat count not expanded to 3 spaces: {text:?}");
    }
}
