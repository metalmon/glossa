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
        if cd.rows.is_empty() {
            // ref-only chart (data lives in referenced sheet cells, not
            // embedded locally) — cell-range resolution is out of scope.
            tracing::warn!(
                "odf chart {name} in {} had no local data table (ref-only chart, unsupported)",
                path.display()
            );
            continue;
        }
        let table = rows_to_table(&cd.rows);
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
}
