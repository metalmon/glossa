use crate::extract::office_chunk::chunk_ir;
use crate::extract::office_table::expand_merged_tables;
use crate::extract::Extractor;
use crate::model::Chunk;
use anyhow::{anyhow, Context};
#[allow(unused_imports)]
use office_oxide::ir::{
    DocumentIR, Element, Heading, InlineContent, Paragraph, Section, Table, TableCell, TableRow,
    TextSpan,
};
use quick_xml::events::Event;
use quick_xml::Reader;
use std::io::{Cursor, Read};
use std::path::Path;

pub struct OdfExtractor;

fn read_content_xml(bytes: &[u8], path: &Path) -> anyhow::Result<String> {
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes.to_vec()))
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

fn parse_to_ir(xml: &str, ext: &str) -> anyhow::Result<DocumentIR> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(false);

    let mut elements: Vec<Element> = Vec::new();
    let mut buf = String::new();          // active paragraph/heading text
    let mut pending_heading: Option<u8> = None; // Some(level) while inside a heading-ish block
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
                    buf.clear();
                    pending_heading = Some(
                        attr(&e, b"outline-level").and_then(|v| v.parse().ok()).unwrap_or(1),
                    );
                }
                b"p" => {
                    buf.clear();
                    pending_heading = attr(&e, b"style-name")
                        .as_deref()
                        .and_then(heading_level_from_style);
                }
                _ => {}
            },
            Ok(Event::Empty(e)) => match local(e.name().as_ref()) {
                b"tab" | b"s" => buf.push(' '),
                _ => {}
            },
            Ok(Event::End(e)) => match local(e.name().as_ref()) {
                b"h" | b"p" => {
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
                _ => {}
            },
            _ => {}
        }
        ev.clear();
    }

    let _ = ext; // ODS branch (Task 3) will switch on `office:spreadsheet`
    Ok(DocumentIR {
        sections: vec![Section { elements, ..Default::default() }],
        ..Default::default()
    })
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
}
