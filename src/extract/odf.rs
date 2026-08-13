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
/// LibreOffice/converted docs use `Heading_20_N` (encoded space); some use `Heading N`.
fn heading_level_from_style(style: &str) -> Option<u8> {
    let norm = style.replace("_20_", " ");
    let rest = norm.strip_prefix("Heading ")?;
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

    let mut sections: Vec<Section> = Vec::new(); // ODS: one per <table:table>; ODT: single, pushed at EOF
    let mut elements: Vec<Element> = Vec::new();
    let mut buf = String::new();          // active paragraph/heading text
    let mut pending_heading: Option<u8> = None; // Some(level) while inside a heading-ish block
    let mut in_table = false;
    let mut current_sheet: Option<String> = None; // ODS: table:name of the table in progress
    let mut rows: Vec<TableRow> = Vec::new();
    let mut cur_cells: Vec<TableCell> = Vec::new();
    let mut cell_span: usize = 1;
    let mut cell_row_span: usize = 1; // number-rows-spanned, stashed at cell Start
    let mut cell_repeat: usize = 1; // number-columns-repeated, stashed at cell Start
    let mut empty_run: usize = 0; // deferred trailing-empty cells (ODS repeats; harmless for ODT)
    let mut row_repeat: usize = 1; // number-rows-repeated, stashed at row Start
    let mut pending_empty_rows: usize = 0; // deferred trailing-empty rows (ODS only; clamped like empty_run)
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
                b"table" => {
                    in_table = true;
                    rows.clear();
                    pending_empty_rows = 0;
                    if ext == "ods" {
                        current_sheet = attr(&e, b"name");
                    }
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
            Ok(Event::Empty(e)) => match local(e.name().as_ref()) {
                b"tab" | b"s" => buf.push(' '),
                _ => {}
            },
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
                        for _ in 0..pending_empty_rows {
                            rows.push(TableRow::default());
                        }
                        pending_empty_rows = 0;
                        for _ in 0..row_repeat {
                            rows.push(TableRow { cells: cells.clone(), ..Default::default() });
                        }
                    }
                    row_repeat = 1;
                }
                b"table" => {
                    in_table = false;
                    pending_empty_rows = 0; // drop trailing empty rows (clamp)
                    let table = Element::Table(Table { rows: std::mem::take(&mut rows), ..Default::default() });
                    if ext == "ods" {
                        sections.push(Section { title: current_sheet.take(), elements: vec![table], ..Default::default() });
                    } else {
                        elements.push(table);
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
}
