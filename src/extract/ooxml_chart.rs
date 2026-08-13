//! Extract OOXML chart data (word/xl/ppt `charts/chartN.xml`) into searchable
//! chunks. Charts are neither embedded images nor part of office_oxide's IR;
//! their data lives in the chart part's cache. Type-agnostic: bar/line/pie/…
//! all use the same `c:ser → c:cat/c:val` shape.
use crate::extract::office_table::table_to_markdown;
use crate::model::Chunk;
use office_oxide::ir::{Element, InlineContent, Paragraph, Table, TableCell, TableRow, TextSpan};
use quick_xml::events::Event;
use quick_xml::Reader;
use std::io::{Cursor, Read};
use std::path::Path;

#[derive(Debug, Default, PartialEq)]
pub(crate) struct Series {
    pub name: Option<String>,
    pub cats: Vec<String>,
    pub vals: Vec<String>,
}

#[derive(Debug, Default, PartialEq)]
pub(crate) struct ChartData {
    pub title: Option<String>,
    pub kind: String,
    pub series: Vec<Series>,
}

fn local(name: &[u8]) -> &[u8] {
    match name.iter().position(|&b| b == b':') {
        Some(i) => &name[i + 1..],
        None => name,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Field {
    None,
    Tx,
    Cat,
    Val,
    Title,
}

pub(crate) fn parse_chart_xml(xml: &str) -> ChartData {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut cd = ChartData::default();
    let mut field = Field::None;
    let mut in_ser = false;
    let mut in_f = false;
    let mut in_plot_area = false;
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Err(_) => break, // malformed → return what we have
            Ok(Event::Eof) => break,
            Ok(Event::Start(e)) => match local(e.name().as_ref()) {
                // first plot-type tag (…Chart) sets the kind
                t if cd.kind.is_empty() && t.ends_with(b"Chart") => {
                    cd.kind = String::from_utf8_lossy(t).into_owned();
                }
                b"plotArea" => in_plot_area = true,
                // the real chart title is a direct child of <c:chart>, before
                // <c:plotArea>; axis titles (<c:catAx>/<c:valAx>) live inside
                // plotArea and must not be mistaken for the chart title.
                b"title" if !in_ser && !in_plot_area => field = Field::Title,
                b"ser" => {
                    in_ser = true;
                    cd.series.push(Series::default());
                }
                b"tx" if in_ser => field = Field::Tx,
                b"cat" if in_ser => field = Field::Cat,
                b"val" if in_ser => field = Field::Val,
                b"f" => in_f = true,
                _ => {}
            },
            Ok(Event::End(e)) => match local(e.name().as_ref()) {
                b"plotArea" => in_plot_area = false,
                b"title" => field = Field::None,
                b"ser" => in_ser = false,
                b"tx" | b"cat" | b"val" => field = Field::None,
                b"f" => in_f = false,
                _ => {}
            },
            // text content: `c:v` values and `a:t` title runs both arrive as Text
            Ok(Event::Text(t)) => {
                if in_f {
                    continue;
                }
                let s = t
                    .decode()
                    .ok()
                    .and_then(|d| quick_xml::escape::unescape(&d).ok().map(|u| u.into_owned()))
                    .unwrap_or_default();
                let s = s.trim();
                if s.is_empty() {
                    continue;
                }
                match field {
                    Field::Title => {
                        // first non-empty title wins (ignore later axis titles)
                        if cd.title.is_none() {
                            cd.title = Some(s.to_string());
                        }
                    }
                    Field::Tx => {
                        if let Some(ser) = cd.series.last_mut() {
                            ser.name = Some(s.to_string());
                        }
                    }
                    Field::Cat => {
                        if let Some(ser) = cd.series.last_mut() {
                            ser.cats.push(s.to_string());
                        }
                    }
                    Field::Val => {
                        if let Some(ser) = cd.series.last_mut() {
                            ser.vals.push(s.to_string());
                        }
                    }
                    Field::None => {}
                }
            }
            _ => {}
        }
        buf.clear();
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

/// Pivot ChartData into an IR table: header ["" , series names...], one body
/// row per category index with the value from each series at that index.
fn chart_to_table(cd: &ChartData) -> Table {
    let n_rows = cd.series.iter().map(|s| s.vals.len()).max().unwrap_or(0);
    // category labels: take the longest cats list among series (they usually match)
    let cats = cd.series.iter().map(|s| &s.cats).max_by_key(|c| c.len());
    let mut rows = Vec::new();

    let mut header = vec![cell("")];
    for (i, s) in cd.series.iter().enumerate() {
        let name = s.name.clone().unwrap_or_else(|| format!("Series {}", i + 1));
        header.push(cell(&name));
    }
    rows.push(TableRow { cells: header, ..Default::default() });

    for r in 0..n_rows {
        let label = cats.and_then(|c| c.get(r)).cloned().unwrap_or_else(|| (r + 1).to_string());
        let mut cells = vec![cell(&label)];
        for s in &cd.series {
            cells.push(cell(s.vals.get(r).map(String::as_str).unwrap_or("")));
        }
        rows.push(TableRow { cells, ..Default::default() });
    }
    Table { rows, ..Default::default() }
}

fn chart_names_for(ext: &str) -> bool {
    matches!(ext, "docx" | "xlsx" | "pptx")
}

/// Scan the OOXML zip for `*/charts/chart*.xml` parts and render each into a
/// searchable chunk (title + GFM data table). Infallible by design: any
/// problem (bad zip, unreadable part, empty parse) skips that chart with a
/// warning rather than failing the whole file's extraction.
pub fn extract_charts(path: &Path, bytes: &[u8], ext: &str) -> Vec<Chunk> {
    if !chart_names_for(ext) {
        return Vec::new();
    }
    let Ok(mut zip) = zip::ZipArchive::new(Cursor::new(bytes)) else {
        return Vec::new();
    };
    // early-out: only pay for charts if any exist
    let parts: Vec<String> = zip
        .file_names()
        .filter(|n| {
            let n = n.to_ascii_lowercase();
            n.contains("/charts/chart") && n.ends_with(".xml") && !n.contains("colors") && !n.contains("style")
        })
        .map(|s| s.to_string())
        .collect();
    if parts.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (i, name) in parts.iter().enumerate() {
        let mut xml = String::new();
        let read_ok = zip
            .by_name(name)
            .ok()
            .and_then(|mut f| f.read_to_string(&mut xml).ok())
            .is_some();
        if !read_ok {
            tracing::warn!("chart part {name} unreadable in {}", path.display());
            continue;
        }
        let cd = parse_chart_xml(&xml);
        if cd.series.is_empty() {
            tracing::warn!("chart {name} in {} had no series data", path.display());
            continue;
        }
        let table = chart_to_table(&cd);
        let body = table_to_markdown(&table);
        let title = cd.title.clone().unwrap_or_else(|| format!("Chart {}", i + 1));
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

    const BAR: &str = r#"<?xml version="1.0"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<c:chart>
 <c:title><c:tx><c:rich><a:p><a:r><a:t>Sales</a:t></a:r></a:p></c:rich></c:tx></c:title>
 <c:plotArea><c:barChart>
  <c:ser>
   <c:tx><c:strRef><c:strCache><c:pt idx="0"><c:v>Series 1</c:v></c:pt></c:strCache></c:strRef></c:tx>
   <c:cat><c:strRef><c:strCache>
     <c:pt idx="0"><c:v>Q1</c:v></c:pt><c:pt idx="1"><c:v>Q2</c:v></c:pt>
   </c:strCache></c:strRef></c:cat>
   <c:val><c:numRef><c:numCache>
     <c:pt idx="0"><c:v>4.3</c:v></c:pt><c:pt idx="1"><c:v>2.5</c:v></c:pt>
   </c:numCache></c:numRef></c:val>
  </c:ser>
  <c:ser>
   <c:tx><c:v>Series 2</c:v></c:tx>
   <c:cat><c:strRef><c:strCache>
     <c:pt idx="0"><c:v>Q1</c:v></c:pt><c:pt idx="1"><c:v>Q2</c:v></c:pt>
   </c:strCache></c:strRef></c:cat>
   <c:val><c:numRef><c:numCache>
     <c:pt idx="0"><c:v>2.4</c:v></c:pt><c:pt idx="1"><c:v>4.4</c:v></c:pt>
   </c:numCache></c:numRef></c:val>
  </c:ser>
 </c:barChart></c:plotArea>
</c:chart></c:chartSpace>"#;

    #[test]
    fn parses_title_kind_series() {
        let cd = parse_chart_xml(BAR);
        assert_eq!(cd.title.as_deref(), Some("Sales"));
        assert_eq!(cd.kind, "barChart");
        assert_eq!(cd.series.len(), 2);
        assert_eq!(cd.series[0].name.as_deref(), Some("Series 1"));
        assert_eq!(cd.series[0].cats, vec!["Q1", "Q2"]);
        assert_eq!(cd.series[0].vals, vec!["4.3", "2.5"]);
        assert_eq!(cd.series[1].name.as_deref(), Some("Series 2"));
        assert_eq!(cd.series[1].vals, vec!["2.4", "4.4"]);
    }

    #[test]
    fn empty_on_junk() {
        let cd = parse_chart_xml("not xml at all <<<");
        assert!(cd.series.is_empty());
    }

    #[test]
    fn excludes_formula_ref_text() {
        const WITH_F: &str = r#"<?xml version="1.0"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart">
<c:chart><c:plotArea><c:barChart>
 <c:ser>
  <c:cat><c:strRef>
    <c:f>Sheet1!$A$1</c:f>
    <c:strCache><c:pt idx="0"><c:v>Q1</c:v></c:pt></c:strCache>
  </c:strRef></c:cat>
  <c:val><c:numRef>
    <c:f>Sheet1!$B$1</c:f>
    <c:numCache><c:pt idx="0"><c:v>4.3</c:v></c:pt></c:numCache>
  </c:numRef></c:val>
 </c:ser>
</c:barChart></c:plotArea></c:chart></c:chartSpace>"#;
        let cd = parse_chart_xml(WITH_F);
        assert_eq!(cd.series.len(), 1);
        assert_eq!(cd.series[0].cats, vec!["Q1"]);
        assert!(!cd.series[0].cats.contains(&"Sheet1!$A$1".to_string()));
        assert_eq!(cd.series[0].vals, vec!["4.3"]);
        assert!(!cd.series[0].vals.contains(&"Sheet1!$B$1".to_string()));
    }

    #[test]
    fn axis_title_is_not_mistaken_for_chart_title() {
        // No chart-level <c:title> (direct child of <c:chart>, before plotArea);
        // only an axis title inside <c:plotArea>/<c:catAx>. Must NOT be adopted
        // as the chart title.
        const AXIS_ONLY: &str = r#"<?xml version="1.0"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<c:chart>
 <c:plotArea><c:barChart>
  <c:ser>
   <c:cat><c:strRef><c:strCache><c:pt idx="0"><c:v>Q1</c:v></c:pt></c:strCache></c:strRef></c:cat>
   <c:val><c:numRef><c:numCache><c:pt idx="0"><c:v>4.3</c:v></c:pt></c:numCache></c:numRef></c:val>
  </c:ser>
  <c:catAx>
   <c:title><c:tx><c:rich><a:p><a:r><a:t>Quarter</a:t></a:r></a:p></c:rich></c:tx></c:title>
  </c:catAx>
 </c:barChart></c:plotArea>
</c:chart></c:chartSpace>"#;
        let cd = parse_chart_xml(AXIS_ONLY);
        assert_eq!(cd.title, None, "axis title must not become the chart title: {cd:?}");
    }

    use crate::model::Chunk;
    use std::path::Path;

    fn charts(fixture: &str) -> Vec<Chunk> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(fixture);
        let bytes = std::fs::read(&path).unwrap();
        let ext = fixture.rsplit('.').next().unwrap();
        super::extract_charts(&path, &bytes, ext)
    }

    #[test]
    fn docx_chart_yields_data_table_chunk() {
        let cs = charts("sample_chart.docx");
        assert!(!cs.is_empty(), "expected a chart chunk");
        let t = cs.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n");
        assert!(t.contains("Chart:") && t.contains("Sales by quarter"), "chart header/title missing:\n{t}");
        assert!(t.contains('|') && t.contains("---"), "GFM table missing:\n{t}");
        // English cached data from the synthetic fixture
        assert!(t.contains("Series 1") && t.contains("Series 2"), "series names missing:\n{t}");
        assert!(t.contains("Q1") && t.contains("4.3") && t.contains("2.4"), "categories/values missing:\n{t}");
        // the c:f formula ref (Sheet1!$B$2:$B$4) must NOT leak into the table
        assert!(!t.contains("Sheet1!"), "formula ref leaked into chart table:\n{t}");
        assert!(cs.iter().all(|c| c.file_type == "docx"));
    }

    /// A zip with a `word/charts/chart1.xml` part containing garbage bytes
    /// must never panic; extract_charts should just skip it (empty result).
    #[test]
    fn garbage_chart_part_no_panic() {
        use std::io::Write as _;
        let mut buf = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zw = zip::ZipWriter::new(cursor);
            let opts = zip::write::SimpleFileOptions::default();
            zw.start_file("word/charts/chart1.xml", opts).unwrap();
            zw.write_all(b"not xml at all <<< \x00\x01\x02 garbage").unwrap();
            zw.finish().unwrap();
        }
        let cs = extract_charts(Path::new("garbage.docx"), &buf, "docx");
        assert!(cs.is_empty(), "garbage chart part should yield no chunks, got: {cs:?}");
    }

    /// When no plot-type (…Chart) tag was captured, `cd.kind` is empty; the
    /// rendered header must omit the empty parens rather than reading
    /// "Chart: Chart 1 ()".
    #[test]
    fn no_empty_parens_when_kind_absent() {
        use std::io::Write as _;
        const NO_KIND: &str = r#"<?xml version="1.0"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart">
<c:chart><c:plotArea>
 <c:ser>
  <c:cat><c:strRef><c:strCache><c:pt idx="0"><c:v>Q1</c:v></c:pt></c:strCache></c:strRef></c:cat>
  <c:val><c:numRef><c:numCache><c:pt idx="0"><c:v>4.3</c:v></c:pt></c:numCache></c:numRef></c:val>
 </c:ser>
</c:plotArea></c:chart></c:chartSpace>"#;
        let mut buf = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zw = zip::ZipWriter::new(cursor);
            let opts = zip::write::SimpleFileOptions::default();
            zw.start_file("word/charts/chart1.xml", opts).unwrap();
            zw.write_all(NO_KIND.as_bytes()).unwrap();
            zw.finish().unwrap();
        }
        let cs = extract_charts(Path::new("nokind.docx"), &buf, "docx");
        assert_eq!(cs.len(), 1, "expected one chart chunk: {cs:?}");
        assert!(!cs[0].text.contains("()"), "empty kind parens leaked:\n{}", cs[0].text);
        assert!(
            cs[0].text.starts_with("Chart: Chart 1\n\n"),
            "expected kind-less header, got:\n{}",
            cs[0].text
        );
    }
}
