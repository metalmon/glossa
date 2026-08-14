//! Extract ODF chart data (`Object N/content.xml` embedded chart sub-documents)
//! into searchable chunks. Mirrors `ooxml_chart.rs`, but the ODF chart shape is
//! different: a `<chart:chart>` sub-document carries its data as a LOCAL
//! `<table:table>` already in display shape (header row, then category/value
//! rows) — this is how LibreOffice/odfpy write charts. No re-pivoting needed;
//! the rows are rendered verbatim as a GFM table.
use crate::extract::office_table::table_to_markdown;
use crate::model::Chunk;
use office_oxide::ir::{Element, InlineContent, Paragraph, Table, TableCell, TableRow, TextSpan};
use quick_xml::events::Event;
use quick_xml::Reader;
use std::io::{Cursor, Read};
use std::path::Path;

#[derive(Debug, Default, PartialEq)]
pub(crate) struct OdfChartData {
    pub title: Option<String>,
    pub kind: String,
    pub rows: Vec<Vec<String>>,
    /// `<chart:categories table:cell-range-address="...">` — only populated
    /// when the chart has no local table (ref-only chart).
    pub categories_ref: Option<String>,
    /// One entry per `<chart:series>`, in document order.
    pub series_refs: Vec<SeriesRef>,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct SeriesRef {
    /// `chart:label-cell-address` — a single-cell ref for the series name.
    pub label_ref: Option<String>,
    /// `chart:values-cell-range-address` — the data range for this series.
    pub values_ref: Option<String>,
}

fn local(name: &[u8]) -> &[u8] {
    match name.iter().position(|&b| b == b':') {
        Some(i) => &name[i + 1..],
        None => name,
    }
}

fn attr(e: &quick_xml::events::BytesStart, want: &[u8]) -> Option<String> {
    e.attributes().flatten().find_map(|a| {
        if local(a.key.as_ref()) == want {
            std::str::from_utf8(&a.value).ok().map(str::to_string)
        } else {
            None
        }
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Field {
    None,
    Title,
}

/// Parse a chart sub-document's `content.xml`. Infallible: malformed XML just
/// yields whatever was parsed before the error (or the default, empty value).
pub(crate) fn parse_odf_chart_xml(xml: &str) -> OdfChartData {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    reader.config_mut().expand_empty_elements = true;

    let mut cd = OdfChartData::default();
    let mut field = Field::None;
    let mut in_axis = false;
    let mut in_table = false;
    let mut table_done = false;
    let mut in_cell = false;
    let mut buf = String::new();
    let mut cur_row: Vec<String> = Vec::new();
    let mut ev = Vec::new();

    loop {
        match reader.read_event_into(&mut ev) {
            Err(_) => break, // malformed → return what we have
            Ok(Event::Eof) => break,
            Ok(Event::Start(e)) => match local(e.name().as_ref()) {
                // <chart:chart chart:class="chart:bar">. The wrapper <office:chart>
                // also has local name "chart" but carries no class attribute, so it
                // is a harmless no-op here (kind stays empty until the real tag).
                b"chart" if cd.kind.is_empty() => {
                    if let Some(class) = attr(&e, b"class") {
                        cd.kind = class.strip_prefix("chart:").unwrap_or(&class).to_string();
                    }
                }
                // axis titles (<chart:axis><chart:title>) are NOT the chart title;
                // suppress field capture for the whole axis subtree.
                b"axis" => in_axis = true,
                b"title" if !in_axis => field = Field::Title,
                // the chart's own local data table; not nested inside another table.
                // A chart part has one local data table; ignore any extras (keep the first).
                b"table" if !in_table && !table_done => {
                    in_table = true;
                }
                b"table-row" if in_table => cur_row = Vec::new(),
                b"table-cell" if in_table => {
                    in_cell = true;
                    buf.clear();
                }
                // a cell may hold multiple <text:p>; join them with a space.
                b"p" if in_cell && !buf.is_empty() => buf.push(' '),
                // ref-only chart: series/category refs to sheet cells, captured
                // regardless of the local-table state (a chart has one or the other).
                b"series" => {
                    cd.series_refs.push(SeriesRef {
                        label_ref: attr(&e, b"label-cell-address"),
                        values_ref: attr(&e, b"values-cell-range-address"),
                    });
                }
                b"categories" if cd.categories_ref.is_none() => {
                    cd.categories_ref = attr(&e, b"cell-range-address");
                }
                _ => {}
            },
            Ok(Event::End(e)) => match local(e.name().as_ref()) {
                b"axis" => in_axis = false,
                b"title" => field = Field::None,
                b"table-cell" => {
                    if in_table {
                        cur_row.push(buf.trim().to_string());
                    }
                    buf.clear();
                    in_cell = false;
                }
                b"table-row" => {
                    if in_table {
                        cd.rows.push(std::mem::take(&mut cur_row));
                    }
                }
                b"table" => {
                    in_table = false;
                    table_done = true;
                }
                _ => {}
            },
            Ok(Event::Text(t)) => {
                let s = t
                    .decode()
                    .ok()
                    .and_then(|d| quick_xml::escape::unescape(&d).ok().map(|u| u.into_owned()))
                    .unwrap_or_default();
                if s.is_empty() {
                    continue;
                }
                if field == Field::Title {
                    let s2 = s.trim();
                    // first non-empty title wins (ignore any later text)
                    if !s2.is_empty() && cd.title.is_none() {
                        cd.title = Some(s2.to_string());
                    }
                }
                if in_cell {
                    buf.push_str(&s);
                }
            }
            _ => {}
        }
        ev.clear();
    }
    cd
}

fn cell(text: &str) -> TableCell {
    TableCell {
        content: vec![Element::Paragraph(Paragraph {
            content: vec![InlineContent::Text(TextSpan { text: text.to_string(), ..Default::default() })],
            ..Default::default()
        })],
        ..Default::default()
    }
}

/// The local table's rows are already pivoted (header + category/value rows) —
/// render verbatim, one IR row per parsed row, no re-pivoting.
fn rows_to_table(rows: &[Vec<String>]) -> Table {
    let table_rows = rows
        .iter()
        .map(|r| TableRow { cells: r.iter().map(|c| cell(c)).collect(), ..Default::default() })
        .collect();
    Table { rows: table_rows, ..Default::default() }
}

fn odf_ext(ext: &str) -> bool {
    matches!(ext, "odt" | "ods" | "odp")
}

// ---------------------------------------------------------------------------
// Ref-only chart resolution: the chart part carries no local `<table:table>`,
// only cell-range refs into the container's own sheet data (LibreOffice/Excel
// "link to spreadsheet" chart form). Resolve those refs against a grid built
// from the container's top-level content.xml, then reuse the same
// rows -> ir::Table -> markdown path as the local-table case.
// ---------------------------------------------------------------------------

/// Cap on a single ODF cell-range span (rows or columns) and on repeat-attribute
/// expansion, mirroring the ODF text extractor's bound: real files never need
/// more, and this keeps a malformed/adversarial range or repeat count from
/// materializing an unbounded Vec.
const MAX_REPEAT: usize = 4096;
/// Cap on the total number of cells materialized while reading the container's
/// sheet grid (summed across all rows), independent of any single dimension.
const MAX_GRID_CELLS: usize = 200_000;

type SheetGrid = std::collections::HashMap<String, Vec<Vec<String>>>;

/// A parsed `[Sheet.]$COL$ROW` address (1-based col/row). `sheet` is `""` when
/// the component omitted it (the caller is expected to have inherited one).
#[derive(Debug, Clone, PartialEq)]
struct CellAddr {
    sheet: String,
    col: usize,
    row: usize,
}

fn col_letters_to_index(s: &str) -> Option<usize> {
    if s.is_empty() || !s.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    let mut n: usize = 0;
    for c in s.chars() {
        let d = (c.to_ascii_uppercase() as u8 - b'A' + 1) as usize;
        n = n * 26 + d;
    }
    Some(n)
}

/// Parse one `[Sheet.]$COL$ROW` component. `inherit_sheet` fills in a sheet
/// name when the component omits its own (e.g. the second half of a range).
fn parse_cell_addr(part: &str, inherit_sheet: &str) -> Option<CellAddr> {
    let part = part.trim();
    let (sheet_part, cell_part) = match part.split_once('.') {
        Some((s, c)) => (s, c),
        None => ("", part),
    };
    let sheet = if sheet_part.is_empty() { inherit_sheet.to_string() } else { sheet_part.to_string() };
    let stripped: String = cell_part.chars().filter(|&c| c != '$').collect();
    let split_at = stripped.find(|c: char| c.is_ascii_digit())?;
    let (col_s, row_s) = stripped.split_at(split_at);
    if col_s.is_empty() || row_s.is_empty() {
        return None;
    }
    let col = col_letters_to_index(col_s)?;
    let row: usize = row_s.parse().ok()?;
    Some(CellAddr { sheet, col, row })
}

/// A resolved cell range: 1-based, inclusive, `c1<=c2` / `r1<=r2` (normalized).
#[derive(Debug, Clone, PartialEq)]
struct CellRange {
    sheet: String,
    c1: usize,
    r1: usize,
    c2: usize,
    r2: usize,
}

/// Parse an ODF cell-range address, e.g. `Sheet1.$B$2:.$B$4` (range, second
/// half inherits the sheet) or `Sheet1.$B$1` (single cell). Infallible:
/// anything unparseable yields `None` so the caller can skip that chart.
fn parse_cell_range(addr: &str) -> Option<CellRange> {
    let addr = addr.trim();
    if addr.is_empty() {
        return None;
    }
    let mut parts = addr.splitn(2, ':');
    let first = parts.next()?;
    let a1 = parse_cell_addr(first, "")?;
    match parts.next() {
        Some(second) => {
            let a2 = parse_cell_addr(second, &a1.sheet)?;
            Some(CellRange {
                sheet: a1.sheet,
                c1: a1.col.min(a2.col),
                r1: a1.row.min(a2.row),
                c2: a1.col.max(a2.col),
                r2: a1.row.max(a2.row),
            })
        }
        None => Some(CellRange { sheet: a1.sheet, c1: a1.col, r1: a1.row, c2: a1.col, r2: a1.row }),
    }
}

fn grid_cell(grid: &[Vec<String>], row: usize, col: usize) -> Option<String> {
    if row == 0 || col == 0 {
        return None;
    }
    grid.get(row - 1)?.get(col - 1).cloned()
}

/// Flatten a (typically single-column or single-row) range into an ordered
/// list of cell texts. Missing/out-of-grid cells become `""` rather than
/// shrinking the list, so category/value alignment by index still holds.
/// Span is clamped to `MAX_REPEAT` per dimension to bound a malformed range.
fn grid_range_values(grid: &[Vec<String>], range: &CellRange) -> Vec<String> {
    let r2 = range.r1 + (range.r2 - range.r1).min(MAX_REPEAT - 1);
    let c2 = range.c1 + (range.c2 - range.c1).min(MAX_REPEAT - 1);
    let mut out = Vec::new();
    if range.c1 == c2 {
        for row in range.r1..=r2 {
            out.push(grid_cell(grid, row, range.c1).unwrap_or_default());
        }
    } else if range.r1 == r2 {
        for col in range.c1..=c2 {
            out.push(grid_cell(grid, range.r1, col).unwrap_or_default());
        }
    } else {
        for row in range.r1..=r2 {
            for col in range.c1..=c2 {
                out.push(grid_cell(grid, row, col).unwrap_or_default());
            }
        }
    }
    out
}

/// Resolve a single-cell ref (e.g. a series' `label-cell-address`) against
/// the sheet grid. `None` on any unparseable ref / unknown sheet / empty cell.
fn resolve_single_cell(addr: &str, grids: &SheetGrid) -> Option<String> {
    let r = parse_cell_range(addr)?;
    let grid = grids.get(&r.sheet)?;
    grid_cell(grid, r.r1, r.c1).filter(|s| !s.is_empty())
}

/// Resolve a ref-only chart's series/category refs against the container's
/// sheet grid into the same `rows` shape the local-table path produces:
/// header `["", series1, series2, ...]`, then one row per category.
/// Best-effort/infallible: `None` means "give up on this chart" (bad sheet,
/// bad range, or no series), never a panic.
fn resolve_ref_chart(cd: &OdfChartData, grids: &SheetGrid) -> Option<Vec<Vec<String>>> {
    let cat_range = cd.categories_ref.as_deref().and_then(parse_cell_range)?;
    let cat_grid = grids.get(&cat_range.sheet)?;
    let categories = grid_range_values(cat_grid, &cat_range);
    if categories.is_empty() {
        return None;
    }

    let mut header = vec![String::new()];
    let mut series_values: Vec<Vec<String>> = Vec::new();
    for (i, s) in cd.series_refs.iter().enumerate() {
        let Some(values_range) = s.values_ref.as_deref().and_then(parse_cell_range) else { continue };
        let Some(values_grid) = grids.get(&values_range.sheet) else { continue };
        let values = grid_range_values(values_grid, &values_range);
        let label = s
            .label_ref
            .as_deref()
            .and_then(|a| resolve_single_cell(a, grids))
            .unwrap_or_else(|| format!("Series {}", i + 1));
        header.push(label);
        series_values.push(values);
    }
    if series_values.is_empty() {
        return None;
    }

    let mut rows = vec![header];
    for (i, cat) in categories.iter().enumerate() {
        let mut row = vec![cat.clone()];
        for sv in &series_values {
            row.push(sv.get(i).cloned().unwrap_or_default());
        }
        rows.push(row);
    }
    Some(rows)
}

/// Parse the container's top-level `content.xml` (`office:spreadsheet >
/// table:table`) into a dense per-sheet grid, `sheet_name -> [row][col]`,
/// expanding `table:number-columns-repeated` / `table:number-rows-repeated`
/// so ODF A1-style addressing lines up. Infallible: malformed XML yields
/// whatever sheets were parsed before the error.
fn parse_sheet_grid(xml: &str) -> SheetGrid {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    reader.config_mut().expand_empty_elements = true;

    let mut grids: SheetGrid = SheetGrid::new();
    let mut in_table = false;
    let mut table_name: Option<String> = None;
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut cur_row: Vec<String> = Vec::new();
    let mut row_repeat: usize = 1;
    let mut col_repeat: usize = 1;
    let mut in_cell = false;
    let mut value_attr: Option<String> = None;
    let mut buf = String::new();
    let mut total_cells: usize = 0;
    let mut ev = Vec::new();

    loop {
        if total_cells >= MAX_GRID_CELLS {
            break; // budget exhausted — return what we have (best-effort)
        }
        match reader.read_event_into(&mut ev) {
            Err(_) => break,
            Ok(Event::Eof) => break,
            Ok(Event::Start(e)) => match local(e.name().as_ref()) {
                b"table" if !in_table => {
                    in_table = true;
                    table_name = attr(&e, b"name");
                    rows = Vec::new();
                }
                b"table-row" if in_table => {
                    cur_row = Vec::new();
                    row_repeat = attr(&e, b"number-rows-repeated")
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(1)
                        .clamp(1, MAX_REPEAT);
                }
                b"table-cell" if in_table => {
                    in_cell = true;
                    buf.clear();
                    value_attr = attr(&e, b"value");
                    col_repeat = attr(&e, b"number-columns-repeated")
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(1)
                        .clamp(1, MAX_REPEAT);
                }
                b"p" if in_cell && !buf.is_empty() => buf.push(' '),
                _ => {}
            },
            Ok(Event::End(e)) => match local(e.name().as_ref()) {
                b"table-cell" => {
                    if in_table {
                        let trimmed = buf.trim();
                        let text = if trimmed.is_empty() {
                            value_attr.clone().unwrap_or_default()
                        } else {
                            trimmed.to_string()
                        };
                        for _ in 0..col_repeat {
                            cur_row.push(text.clone());
                        }
                    }
                    buf.clear();
                    value_attr = None;
                    in_cell = false;
                    col_repeat = 1;
                }
                b"table-row" => {
                    if in_table {
                        for _ in 0..row_repeat {
                            total_cells += cur_row.len();
                            rows.push(cur_row.clone());
                            if total_cells >= MAX_GRID_CELLS {
                                break;
                            }
                        }
                        cur_row = Vec::new();
                    }
                    row_repeat = 1;
                }
                b"table" => {
                    if in_table {
                        if let Some(name) = table_name.take() {
                            grids.insert(name, std::mem::take(&mut rows));
                        }
                    }
                    in_table = false;
                }
                _ => {}
            },
            Ok(Event::Text(t)) => {
                if in_cell {
                    let s = t
                        .decode()
                        .ok()
                        .and_then(|d| quick_xml::escape::unescape(&d).ok().map(|u| u.into_owned()))
                        .unwrap_or_default();
                    buf.push_str(&s);
                }
            }
            _ => {}
        }
        ev.clear();
    }
    grids
}

/// Read and parse the container's own top-level `content.xml` (the sheet
/// data a ref-only chart's cell ranges point into) — distinct from the
/// `Object*/content.xml` chart sub-documents. Infallible: any read/parse
/// failure yields an empty grid, and callers treat that as "resolution
/// failed" for whatever ref-only charts needed it.
fn load_sheet_grid(zip: &mut zip::ZipArchive<Cursor<&[u8]>>, path: &Path) -> SheetGrid {
    let mut xml = String::new();
    let read_ok = zip
        .by_name("content.xml")
        .ok()
        .and_then(|mut f| f.read_to_string(&mut xml).ok())
        .is_some();
    if !read_ok {
        tracing::warn!(
            "odf container content.xml unreadable in {} (needed to resolve a ref-only chart)",
            path.display()
        );
        return SheetGrid::new();
    }
    parse_sheet_grid(&xml)
}

/// Scan the ODF zip for `Object*/content.xml` chart sub-documents and render
/// each into a searchable chunk (title + GFM data table). Infallible by
/// design: any problem (bad zip, unreadable part, no local data table) skips
/// that chart with a warning rather than failing the whole file's extraction.
pub fn extract_odf_charts(path: &Path, bytes: &[u8], ext: &str) -> Vec<Chunk> {
    if !odf_ext(ext) {
        return Vec::new();
    }
    let Ok(mut zip) = zip::ZipArchive::new(Cursor::new(bytes)) else {
        return Vec::new();
    };
    // early-out: only pay for charts if any Object*/content.xml entries exist
    let names: Vec<String> = zip
        .file_names()
        .filter(|n| n.starts_with("Object") && n.ends_with("/content.xml"))
        .map(|s| s.to_string())
        .collect();
    if names.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    // Lazily read + parse the container's own top-level content.xml (sheet
    // data) at most once, and only if some chart actually turns out to be
    // ref-only. Most files with charts use the local-table form and never
    // touch this.
    let mut sheet_grid: Option<SheetGrid> = None;
    for name in &names {
        let mut xml = String::new();
        let read_ok = zip
            .by_name(name)
            .ok()
            .and_then(|mut f| f.read_to_string(&mut xml).ok())
            .is_some();
        if !read_ok {
            tracing::warn!("odf chart part {name} unreadable in {}", path.display());
            continue;
        }
        // Object N may be any embedded OLE object, not necessarily a chart
        // (e.g. an embedded spreadsheet); only chart sub-documents qualify.
        // Fast pre-filter assuming the conventional `chart:` prefix; the streaming
        // parser below is prefix-agnostic (matches on local name) and is the source of truth.
        if !xml.contains("chart:chart") {
            continue;
        }
        let cd = parse_odf_chart_xml(&xml);
        let rows = if !cd.rows.is_empty() {
            cd.rows.clone()
        } else if cd.categories_ref.is_some() && !cd.series_refs.is_empty() {
            // ref-only chart: no local table, but it has cell-range refs into
            // the container's own sheet data — resolve those.
            let grid = sheet_grid.get_or_insert_with(|| load_sheet_grid(&mut zip, path));
            match resolve_ref_chart(&cd, grid) {
                Some(r) => r,
                None => {
                    tracing::warn!(
                        "odf chart {name} in {} is ref-only but its cell-range refs did not resolve \
                         (unknown sheet / bad range) — skipping",
                        path.display()
                    );
                    continue;
                }
            }
        } else {
            tracing::warn!(
                "odf chart {name} in {} had no local data table and no cell-range refs (unsupported)",
                path.display()
            );
            continue;
        };
        let table = rows_to_table(&rows);
        let body = table_to_markdown(&table);
        // number by emitted charts (out.len()), not scan index, so a chart
        // following a skipped non-chart Object doesn't get an inflated number.
        let title = cd.title.clone().unwrap_or_else(|| format!("Chart {}", out.len() + 1));
        let header = if cd.kind.is_empty() {
            format!("Chart: {}", title)
        } else {
            format!("Chart: {} ({})", title, cd.kind)
        };
        let text = format!("{}\n\n{}", header, body);
        out.push(Chunk {
            doc_path: path.to_path_buf(),
            location: title,
            file_type: ext.to_string(),
            text,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::odf::OdfExtractor;
    use crate::extract::Extractor;

    const CHART_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:chart="urn:oasis:names:tc:opendocument:xmlns:chart:1.0"
 xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
<office:body><office:chart>
<chart:chart chart:class="chart:bar">
<chart:plot-area>
 <chart:axis chart:dimension="x"><chart:title><text:p>Quarter</text:p></chart:title></chart:axis>
</chart:plot-area>
<chart:title><text:p>Sales by quarter</text:p></chart:title>
<table:table table:name="local-table">
<table:table-row>
 <table:table-cell><text:p></text:p></table:table-cell>
 <table:table-cell><text:p>Series 1</text:p></table:table-cell>
 <table:table-cell><text:p>Series 2</text:p></table:table-cell>
</table:table-row>
<table:table-row>
 <table:table-cell><text:p>Q1</text:p></table:table-cell>
 <table:table-cell><text:p>4.3</text:p></table:table-cell>
 <table:table-cell><text:p>2.4</text:p></table:table-cell>
</table:table-row>
</table:table>
</chart:chart>
</office:chart></office:body></office:document-content>"#;

    #[test]
    fn parses_title_kind_rows_and_ignores_axis_title() {
        let cd = parse_odf_chart_xml(CHART_XML);
        assert_eq!(cd.title.as_deref(), Some("Sales by quarter"), "chart title mismatch: {cd:?}");
        assert_ne!(cd.title.as_deref(), Some("Quarter"), "axis title must not be adopted as chart title");
        assert_eq!(cd.kind, "bar");
        assert_eq!(cd.rows.len(), 2, "expected header + one data row: {cd:?}");
        assert_eq!(cd.rows[0], vec!["".to_string(), "Series 1".to_string(), "Series 2".to_string()]);
        assert_eq!(cd.rows[1], vec!["Q1".to_string(), "4.3".to_string(), "2.4".to_string()]);
    }

    #[test]
    fn empty_on_junk() {
        let cd = parse_odf_chart_xml("not xml at all <<<");
        assert!(cd.rows.is_empty());
        assert!(cd.title.is_none());
    }

    fn charts(fixture: &str) -> Vec<Chunk> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(fixture);
        let bytes = std::fs::read(&path).unwrap();
        let ext = fixture.rsplit('.').next().unwrap();
        super::extract_odf_charts(&path, &bytes, ext)
    }

    #[test]
    fn ods_chart_yields_data_table_chunk() {
        let cs = charts("sample_chart.ods");
        assert!(!cs.is_empty(), "expected a chart chunk");
        let t = cs.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n");
        assert!(t.starts_with("Chart:"), "chart header missing:\n{t}");
        assert!(t.contains("Sales by quarter"), "chart title missing:\n{t}");
        assert!(t.contains("Series 1") && t.contains("4.3"), "series/value data missing:\n{t}");
        assert!(t.contains('|') && t.contains("---"), "GFM table missing:\n{t}");
        assert!(cs.iter().all(|c| c.file_type == "ods"));
    }

    /// End-to-end through OdfExtractor: the injected `Object 1/content.xml`
    /// chart sub-document must not break ODS text extraction, and the chart
    /// chunk must be appended alongside the regular sheet chunks.
    #[test]
    fn odf_extractor_appends_chart_chunk() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sample_chart.ods");
        let bytes = std::fs::read(&path).unwrap();
        let chunks = OdfExtractor.extract(&path, &bytes).unwrap();
        assert!(
            chunks.iter().any(|c| c.text.starts_with("Chart:")),
            "expected a chart chunk, got: {chunks:?}"
        );
        let chart = chunks.iter().find(|c| c.text.starts_with("Chart:")).unwrap();
        assert!(chart.text.contains("Sales by quarter"));
        assert!(chart.text.contains("Series 1"));
        assert!(chart.text.contains("4.3"));
        assert!(chart.text.contains("---"), "expected a GFM table:\n{}", chart.text);
    }

    /// A zip with an `Object 1/content.xml` part containing garbage/truncated
    /// XML must never panic; extract_odf_charts should just skip it.
    #[test]
    fn garbage_chart_part_no_panic() {
        use std::io::Write as _;
        let mut buf = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zw = zip::ZipWriter::new(cursor);
            let opts = zip::write::SimpleFileOptions::default();
            zw.start_file("Object 1/content.xml", opts).unwrap();
            zw.write_all(b"<chart:chart not xml at all <<< \x00\x01\x02 garbage").unwrap();
            zw.finish().unwrap();
        }
        let cs = extract_odf_charts(Path::new("garbage.ods"), &buf, "ods");
        assert!(cs.is_empty(), "garbage chart part should yield no chunks, got: {cs:?}");
    }

    /// Fallback chart numbering must count EMITTED charts, not the scan-loop
    /// index: `Object 1` is a non-chart part (skipped at the `chart:chart`
    /// gate) and `Object 2` is an untitled chart, which must still be "Chart 1".
    #[test]
    fn fallback_title_numbers_by_emitted_charts_not_scan_index() {
        use std::io::Write as _;
        let mut buf = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zw = zip::ZipWriter::new(cursor);
            let opts = zip::write::SimpleFileOptions::default();
            zw.start_file("Object 1/content.xml", opts).unwrap();
            zw.write_all(
                br#"<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0">
<office:body><office:spreadsheet/></office:body></office:document-content>"#,
            )
            .unwrap();
            zw.start_file("Object 2/content.xml", opts).unwrap();
            zw.write_all(
                br#"<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:chart="urn:oasis:names:tc:opendocument:xmlns:chart:1.0"
 xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
<office:body><office:chart>
<chart:chart chart:class="chart:bar">
<table:table table:name="local-table">
<table:table-row>
 <table:table-cell><text:p></text:p></table:table-cell>
 <table:table-cell><text:p>Series 1</text:p></table:table-cell>
</table:table-row>
<table:table-row>
 <table:table-cell><text:p>Q1</text:p></table:table-cell>
 <table:table-cell><text:p>4.3</text:p></table:table-cell>
</table:table-row>
</table:table>
</chart:chart>
</office:chart></office:body></office:document-content>"#,
            )
            .unwrap();
            zw.finish().unwrap();
        }
        let cs = extract_odf_charts(Path::new("mixed.ods"), &buf, "ods");
        assert_eq!(cs.len(), 1, "expected exactly one chart chunk, got: {cs:?}");
        assert_eq!(cs[0].location, "Chart 1", "untitled chart after a skipped object must be Chart 1: {cs:?}");
    }

    /// A truncated/malformed zip entirely must not panic either.
    #[test]
    fn not_a_zip_no_panic() {
        let cs = extract_odf_charts(Path::new("junk.ods"), b"not a zip at all", "ods");
        assert!(cs.is_empty());
    }

    /// Non-ODF extensions are always a no-op, even with chart-shaped bytes.
    #[test]
    fn non_odf_ext_returns_empty() {
        let bytes = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sample_chart.ods"),
        )
        .unwrap();
        let cs = extract_odf_charts(Path::new("x.docx"), &bytes, "docx");
        assert!(cs.is_empty());
    }

    // -----------------------------------------------------------------
    // Ref-only chart resolution (cell-range refs, no local data table).
    // -----------------------------------------------------------------

    #[test]
    fn parse_cell_range_handles_range_single_cell_and_missing_sheet() {
        // Range: second half omits the sheet ("." with no name) and must
        // inherit the first half's sheet.
        let r = parse_cell_range("Sheet1.$B$2:.$B$4").expect("range should parse");
        assert_eq!(r.sheet, "Sheet1");
        assert_eq!((r.c1, r.r1, r.c2, r.r2), (2, 2, 2, 4), "B col=2, rows 2..4: {r:?}");

        // Single cell, no ':'.
        let single = parse_cell_range("Sheet1.$B$1").expect("single cell should parse");
        assert_eq!(single.sheet, "Sheet1");
        assert_eq!((single.c1, single.r1, single.c2, single.r2), (2, 1, 2, 1));

        // Column past 'Z' to make sure letter->index isn't hardcoded to one char.
        let aa = parse_cell_range("Sheet1.$AA$1").expect("two-letter column should parse");
        assert_eq!(aa.c1, 27, "AA should be column 27: {aa:?}");

        assert!(parse_cell_range("").is_none());
        assert!(parse_cell_range("garbage").is_none());
    }

    const GRID_XML: &str = r#"<?xml version='1.0' encoding='UTF-8'?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
<office:body><office:spreadsheet>
<table:table table:name="Sheet1">
<table:table-row>
<table:table-cell><text:p>a</text:p></table:table-cell>
<table:table-cell table:number-columns-repeated="3"><text:p>dup</text:p></table:table-cell>
<table:table-cell><text:p>z</text:p></table:table-cell>
</table:table-row>
</table:table>
</office:spreadsheet></office:body></office:document-content>"#;

    #[test]
    fn sheet_grid_reader_expands_number_columns_repeated() {
        let grid = parse_sheet_grid(GRID_XML);
        let sheet = grid.get("Sheet1").expect("Sheet1 present in grid");
        assert_eq!(sheet.len(), 1, "expected one row: {sheet:?}");
        let row = &sheet[0];
        // a, dup, dup, dup, z: the repeat=3 cell must expand to 3 positions
        // so the trailing "z" lands at column E (5), not column C (3).
        assert_eq!(row, &vec!["a", "dup", "dup", "dup", "z"], "repeat expansion mismatch: {row:?}");
        assert_eq!(grid_cell(sheet, 1, 1).as_deref(), Some("a"));
        assert_eq!(grid_cell(sheet, 1, 5).as_deref(), Some("z"), "z must be at col 5 after expansion");
    }

    /// The committed ref-only fixture: `Object 1/content.xml` has a
    /// `<chart:chart>` with no local `<table:table>`, only series/category
    /// cell-range refs into the container's own `Sheet1`. This must resolve
    /// to the same rows shape (and thus the same GFM table) the local-table
    /// path produces — the data lives at Sheet1 A1:B4: header `Cat|Series 1`,
    /// then Q1/4.3, Q2/2.5, Q3/3.5.
    #[test]
    fn ref_only_chart_resolves_against_sheet_cells() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sample_chart_ref.ods");
        let bytes = std::fs::read(&path).unwrap();
        let cs = extract_odf_charts(&path, &bytes, "ods");
        assert!(!cs.is_empty(), "expected a resolved ref-only chart chunk");
        let t = cs.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n");
        assert!(
            t.starts_with("Chart: Sales by quarter (bar)"),
            "chart header mismatch:\n{t}"
        );
        assert!(t.contains('|') && t.contains("---"), "GFM table missing:\n{t}");
        assert!(t.contains("Series 1"), "resolved series label missing:\n{t}");
        assert!(t.contains("Q1") && t.contains("Q2") && t.contains("Q3"), "category labels missing:\n{t}");
        assert!(
            t.contains("4.3") && t.contains("2.5") && t.contains("3.5"),
            "resolved values missing:\n{t}"
        );
    }

    /// Same fixture, through the full `OdfExtractor` path (not just the
    /// chart-scan helper), matching how `sample_chart.ods` is covered above.
    #[test]
    fn odf_extractor_resolves_ref_only_chart_end_to_end() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sample_chart_ref.ods");
        let bytes = std::fs::read(&path).unwrap();
        let chunks = OdfExtractor.extract(&path, &bytes).unwrap();
        let chart = chunks
            .iter()
            .find(|c| c.text.starts_with("Chart:"))
            .unwrap_or_else(|| panic!("expected a chart chunk, got: {chunks:?}"));
        assert!(chart.text.starts_with("Chart: Sales by quarter (bar)"), "{}", chart.text);
        assert!(chart.text.contains("Series 1"));
        assert!(chart.text.contains("Q1") && chart.text.contains("Q2") && chart.text.contains("Q3"));
        assert!(chart.text.contains("4.3") && chart.text.contains("2.5") && chart.text.contains("3.5"));
        assert!(chart.text.contains("---"), "expected a GFM table:\n{}", chart.text);
    }

    /// A ref-only chart whose series/categories point at a sheet that does
    /// not exist in the container must be skipped (best-effort, no panic),
    /// while a sibling chart with a local table is still emitted normally.
    #[test]
    fn ref_only_chart_unknown_sheet_skipped_other_chart_kept() {
        use std::io::Write as _;
        let mut buf = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zw = zip::ZipWriter::new(cursor);
            let opts = zip::write::SimpleFileOptions::default();

            // Container's own sheet data: only "Sheet1" exists.
            zw.start_file("content.xml", opts).unwrap();
            zw.write_all(
                br#"<?xml version='1.0' encoding='UTF-8'?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
<office:body><office:spreadsheet>
<table:table table:name="Sheet1">
<table:table-row><table:table-cell><text:p>Q1</text:p></table:table-cell><table:table-cell><text:p>1</text:p></table:table-cell></table:table-row>
</table:table>
</office:spreadsheet></office:body></office:document-content>"#,
            )
            .unwrap();

            // Object 1: ref-only chart pointing at a sheet that doesn't exist.
            zw.start_file("Object 1/content.xml", opts).unwrap();
            zw.write_all(
                br#"<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:chart="urn:oasis:names:tc:opendocument:xmlns:chart:1.0"
 xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
<office:body><office:chart>
<chart:chart chart:class="chart:bar">
<chart:title><text:p>Ghost</text:p></chart:title>
<chart:plot-area>
<chart:series chart:values-cell-range-address="NoSuchSheet.$B$2:.$B$4" chart:label-cell-address="NoSuchSheet.$B$1">
<chart:categories table:cell-range-address="NoSuchSheet.$A$2:.$A$4"/>
</chart:series>
</chart:plot-area>
</chart:chart>
</office:chart></office:body></office:document-content>"#,
            )
            .unwrap();

            // Object 2: an ordinary local-table chart — must still come through.
            zw.start_file("Object 2/content.xml", opts).unwrap();
            zw.write_all(CHART_XML.as_bytes()).unwrap();

            zw.finish().unwrap();
        }
        let cs = extract_odf_charts(Path::new("mixed_ref.ods"), &buf, "ods");
        assert_eq!(cs.len(), 1, "unresolvable ref chart must be skipped, good chart kept: {cs:?}");
        assert!(
            cs[0].text.contains("Sales by quarter"),
            "expected the surviving local-table chart: {cs:?}"
        );
    }

    /// A ref-only chart whose values-cell-range-address is unparseable
    /// garbage must be skipped without panicking (categories still parse
    /// fine — only the series ref is bad).
    #[test]
    fn ref_only_chart_bad_range_string_no_panic() {
        use std::io::Write as _;
        let mut buf = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zw = zip::ZipWriter::new(cursor);
            let opts = zip::write::SimpleFileOptions::default();
            zw.start_file("content.xml", opts).unwrap();
            zw.write_all(
                br#"<?xml version='1.0' encoding='UTF-8'?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
<office:body><office:spreadsheet>
<table:table table:name="Sheet1">
<table:table-row><table:table-cell><text:p>Q1</text:p></table:table-cell><table:table-cell><text:p>1</text:p></table:table-cell></table:table-row>
</table:table>
</office:spreadsheet></office:body></office:document-content>"#,
            )
            .unwrap();
            zw.start_file("Object 1/content.xml", opts).unwrap();
            zw.write_all(
                br#"<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:chart="urn:oasis:names:tc:opendocument:xmlns:chart:1.0"
 xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
<office:body><office:chart>
<chart:chart chart:class="chart:bar">
<chart:plot-area>
<chart:series chart:values-cell-range-address="not a range" chart:label-cell-address="also not a range">
<chart:categories table:cell-range-address="Sheet1.$A$2:.$A$4"/>
</chart:series>
</chart:plot-area>
</chart:chart>
</office:chart></office:body></office:document-content>"#,
            )
            .unwrap();
            zw.finish().unwrap();
        }
        let cs = extract_odf_charts(Path::new("bad_range.ods"), &buf, "ods");
        assert!(cs.is_empty(), "bad series range must skip the chart, not panic: {cs:?}");
    }
}
