//! Extract OOXML chart data (word/xl/ppt `charts/chartN.xml`) into searchable
//! chunks. Charts are neither embedded images nor part of office_oxide's IR;
//! their data lives in the chart part's cache. Type-agnostic: bar/line/pie/…
//! all use the same `c:ser → c:cat/c:val` shape.
//!
//! `ChartData`/`parse_chart_xml` are not yet called from `office.rs` — that
//! wiring (zip scanning + `extract_charts` + IR-Table rendering) is a
//! follow-up task. Allow dead_code until then.
#![allow(dead_code)]

use quick_xml::events::Event;
use quick_xml::Reader;

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
                b"title" if !in_ser => field = Field::Title,
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
}
