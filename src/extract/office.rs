use crate::extract::office_chunk::chunk_ir;
use crate::extract::office_table::expand_merged_tables;
use crate::extract::Extractor;
use crate::model::Chunk;
use anyhow::anyhow;
use office_oxide::{Document, DocumentFormat};
use std::collections::HashSet;
use std::io::Cursor;
use std::path::Path;

pub struct OfficeExtractor;

fn format_for(ext: &str) -> Option<DocumentFormat> {
    match ext {
        "docx" => Some(DocumentFormat::Docx),
        "doc" => Some(DocumentFormat::Doc),
        // BIFF12 is a different encoding of the same model, so it rides the Xlsx format — which is
        // how office_oxide's own `from_extension` maps it.
        "xlsx" | "xlsb" => Some(DocumentFormat::Xlsx),
        "xls" => Some(DocumentFormat::Xls),
        "pptx" => Some(DocumentFormat::Pptx),
        "ppt" => Some(DocumentFormat::Ppt),
        _ => None,
    }
}

/// Does this IR's last section match the synthetic chart section `office_oxide` appends for
/// xlsx (`convert_xlsx.rs`, around line 380) whenever its own `doc.chart_text` is non-empty?
/// Detected structurally, not by title alone — a user may legitimately name a real worksheet
/// "Charts" — so this requires ALL of: it is the last section, its title is exactly "Charts",
/// and its elements are the strict alternation `Heading(level 3, "Chart 1")`, `Paragraph`,
/// `Heading(level 3, "Chart 2")`, `Paragraph`, ... starting at 1, with nothing else mixed in.
/// Returns the chart count when it matches (used only to describe the section in a warning —
/// the actual drop/keep decision compares chart-PART NAMES, see `xlsx_chart_part_names` and its
/// caller below, because an equal count does not mean the same charts were found), `None` when
/// the section does not match.
fn synthetic_xlsx_chart_section_chart_count(ir: &office_oxide::ir::DocumentIR) -> Option<usize> {
    use office_oxide::ir::{Element, InlineContent};

    let section = ir.sections.last()?;
    if section.title.as_deref() != Some("Charts") {
        return None;
    }
    let elements = &section.elements;
    if elements.is_empty() || elements.len() % 2 != 0 {
        return None;
    }
    let mut count = 0usize;
    for pair in elements.chunks(2) {
        count += 1;
        let Element::Heading(h) = &pair[0] else {
            return None;
        };
        if h.level != 3 {
            return None;
        }
        let heading_text: String = h
            .content
            .iter()
            .filter_map(|c| match c {
                InlineContent::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect();
        if heading_text != format!("Chart {count}") {
            return None;
        }
        if !matches!(&pair[1], Element::Paragraph(_)) {
            return None;
        }
    }
    Some(count)
}

/// The xlsx chart-part names `office_oxide`'s own xlsx reader would see (`xlsx/mod.rs`:
/// `n.starts_with("xl/charts/chart") && n.ends_with(".xml")`, scanning every zip entry). We
/// cannot read which of those it actually kept — its accept rule lives inside
/// `extract_chart_text` and is internal — so this stands in for "every chart part present in
/// the package" and is deliberately broader than our own `ooxml_chart` glob (which also
/// excludes `colors`/`style` parts): the synthetic section is only dropped when our own pass
/// covers this whole set, never when we merely match its size.
fn xlsx_chart_part_names(bytes: &[u8]) -> Vec<String> {
    let Ok(zip) = zip::ZipArchive::new(Cursor::new(bytes)) else {
        return Vec::new();
    };
    zip.file_names()
        .filter(|n| n.starts_with("xl/charts/chart") && n.ends_with(".xml"))
        .map(|s| s.to_string())
        .collect()
}

impl Extractor for OfficeExtractor {
    fn file_types(&self) -> &'static [&'static str] {
        &["docx", "doc", "xlsx", "xlsb", "xls", "pptx", "ppt"]
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
        let mut ir = doc.to_ir();
        expand_merged_tables(&mut ir);
        let (chart_chunks, covered_chart_parts) =
            crate::extract::ooxml_chart::extract_charts_with_coverage(path, bytes, &ext);
        if ext == "xlsx" {
            if let Some(synthetic_chart_count) = synthetic_xlsx_chart_section_chart_count(&ir) {
                let present_parts = xlsx_chart_part_names(bytes);
                let covered: HashSet<&str> =
                    covered_chart_parts.iter().map(String::as_str).collect();
                let missing: Vec<&String> = present_parts
                    .iter()
                    .filter(|p| !covered.contains(p.as_str()))
                    .collect();
                // Vacuously true on an empty `present_parts` is the wrong default here: that
                // would mean our own zip scan found nothing even though `office_oxide` clearly
                // built a chart section from *something*, so treat "found zero present parts"
                // as "not covered" rather than as trivially satisfied.
                if !present_parts.is_empty() && missing.is_empty() {
                    ir.sections.pop();
                } else {
                    tracing::warn!(
                        "xlsx {}: office_oxide's synthetic chart section names {synthetic_chart_count} chart(s), but our chart pass covered only {}/{} chart part(s) in the package (missing: {missing:?}) — keeping both sections rather than risk losing chart text",
                        path.display(),
                        present_parts.len() - missing.len(),
                        present_parts.len()
                    );
                }
            }
        }
        let mut chunks = chunk_ir(path, &ir, &ext);
        chunks.extend(chart_chunks);
        if chunks.is_empty() {
            // "Parsed fine, produced nothing" is the failure that hides. An error is listed in the
            // run summary and retried on the next pass; `Ok(vec![])` enters the manifest as
            // indexed-and-empty WITH its signature recorded, so the document is never looked at
            // again — and nothing in the output distinguishes it from a document that genuinely
            // has no text. `rtf.rs` and `odf.rs` have refused this since they were written; this
            // reader is the one that did not, and it is the one that has silently under-extracted
            // before (the `w:pBdr` scanner desync that ran the reader to EOF).
            //
            // The cost, taken knowingly: an error is a permanent skip, and `index_file` buffers
            // before it deletes, so a document EDITED down to no text keeps its last indexed text
            // under its new signature until a `--force`. That is the better of the two wrongs —
            // zero text from an office reader is far more often the reader's failure than the
            // author's intent — and it is what `odf.rs` and `rtf.rs` have always done. The real
            // repair is a verdict the caller can tell apart ("parsed, genuinely empty" vs "could
            // not read"), which belongs with the extractor registry, not in this branch.
            return Err(anyhow!(
                "office produced no chunks for {} — the file parsed but yielded no text",
                path.display()
            ));
        }
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_office_reader_declines_rtf_bytes_rather_than_half_reading_them() {
        // `extract_file`'s mislabeled-extension rescue rests on this: a `.doc` that is really RTF
        // must either fail or produce nothing, so the rescue can see it declined. The dangerous
        // third outcome is a reader that "succeeds" with garbage chunks — then the file is indexed
        // as mangled text and nothing downstream can tell. That is the docx under-extraction class
        // we have already been bitten by, so it is pinned here rather than assumed.
        let r = OfficeExtractor.extract(
            Path::new("mislabelled.doc"),
            br"{\rtf1\ansi\pard Real text lives here.\par}",
        );
        match r {
            Err(_) => {}
            Ok(cs) => assert!(
                cs.is_empty(),
                "the office reader claimed to read RTF, so the rescue will never fire: {:?}",
                cs.iter().map(|c| &c.text).collect::<Vec<_>>()
            ),
        }
    }

    /// A docx that opens cleanly but carries no text must be a verdict, not `Ok(vec![])`.
    /// Built by repacking the real fixture with an empty body, so everything else about the
    /// package stays a genuine docx and the only thing under test is the empty result.
    #[test]
    fn a_document_that_opens_but_yields_no_text_is_an_error_not_an_empty_success() {
        use std::io::{Read as _, Write as _};
        const EMPTY_BODY: &[u8] = br#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:sectPr/></w:body></w:document>"#;
        let mut zr = zip::ZipArchive::new(Cursor::new(
            include_bytes!("../../tests/fixtures/sample.docx").to_vec(),
        ))
        .unwrap();
        let names: Vec<String> = zr
            .file_names()
            .filter(|n| !n.ends_with('/'))
            .map(|s| s.to_string())
            .collect();
        let mut buf = Vec::new();
        {
            let mut zw = zip::ZipWriter::new(Cursor::new(&mut buf));
            let opts = zip::write::SimpleFileOptions::default();
            for name in names {
                let mut bytes = Vec::new();
                zr.by_name(&name).unwrap().read_to_end(&mut bytes).unwrap();
                zw.start_file(&name, opts).unwrap();
                zw.write_all(if name == "word/document.xml" {
                    EMPTY_BODY
                } else {
                    bytes.as_slice()
                })
                .unwrap();
            }
            zw.finish().unwrap();
        }
        let e = OfficeExtractor
            .extract(Path::new("blank.docx"), &buf)
            .unwrap_err()
            .to_string();
        assert!(e.contains("no chunks"), "{e}");
    }

    #[test]
    fn extracts_text_from_docx_fixture() {
        let bytes = include_bytes!("../../tests/fixtures/sample.docx");
        let chunks = OfficeExtractor
            .extract(Path::new("sample.docx"), bytes)
            .unwrap();
        let joined = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            joined.contains("glossa sample"),
            "expected fixture marker text, got: {joined}"
        );
        assert!(chunks.iter().all(|c| c.file_type == "docx"));
    }

    #[test]
    fn extracts_text_from_xlsb_fixture() {
        // `.xlsb` is a BIFF12 binary workbook — a different encoding of the same model as `.xlsx`,
        // which `office_oxide` reads since 0.1.12. Before that it was not in `file_types()`, so a
        // `.xlsb` in a corpus produced NOTHING, silently: no extractor claimed it and no error said
        // so. The fixture is synthetic and English-only (see tests/fixtures/README if adding more).
        let bytes = include_bytes!("../../tests/fixtures/sample.xlsb");
        let chunks = OfficeExtractor
            .extract(Path::new("sample.xlsb"), bytes)
            .unwrap();
        let joined = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            joined.contains("glossa sample"),
            "expected fixture marker text, got: {joined}"
        );
        assert!(
            joined.contains("inlet pressure") && joined.contains("97"),
            "expected the sheet's cells, got: {joined}"
        );
        assert!(chunks.iter().all(|c| c.file_type == "xlsb"));
    }

    #[test]
    fn claims_xlsb_as_a_supported_type() {
        // `file_types()` is the gate: a type absent here is never routed to this extractor, so the
        // mapping below and this list have to agree.
        assert!(OfficeExtractor.file_types().contains(&"xlsb"));
        assert_eq!(format_for("xlsb"), Some(DocumentFormat::Xlsx));
    }

    #[test]
    fn unsupported_extension_errors() {
        let err = OfficeExtractor
            .extract(Path::new("x.rtf"), b"junk")
            .unwrap_err();
        assert!(err.to_string().contains("unsupported office extension"));
    }

    #[test]
    fn extracts_table_as_markdown() {
        let bytes = include_bytes!("../../tests/fixtures/sample_table.docx");
        let chunks = OfficeExtractor
            .extract(Path::new("sample_table.docx"), bytes)
            .unwrap();
        let joined = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            joined.contains('|') && joined.contains("---"),
            "expected a GFM pipe table from the docx table, got:\n{joined}"
        );
    }

    /// End-to-end through OfficeExtractor: office_oxide must open the
    /// synthetic fixture (an injected, undeclared chart part must be inert
    /// to the text path) AND the wiring in `extract` must append the chart
    /// chunk after the doc's text chunks.
    #[test]
    fn office_extractor_appends_chart_chunk() {
        let bytes = include_bytes!("../../tests/fixtures/sample_chart.docx");
        let chunks = OfficeExtractor
            .extract(Path::new("sample_chart.docx"), bytes)
            .unwrap();
        assert!(
            chunks.iter().any(|c| !c.text.starts_with("Chart:")),
            "expected at least one non-chart (text) chunk, got: {chunks:?}"
        );
        assert!(
            chunks.iter().any(|c| c.text.starts_with("Chart:")),
            "expected a chart chunk appended, got: {chunks:?}"
        );
    }

    /// For xlsx, `office_oxide` appends its own synthetic "Charts" section (a flat blob per
    /// chart) on top of our well-formed `ooxml_chart::extract_charts` chunk — the same numbers
    /// reaching the index twice. When our pass covers every chart part present in the package
    /// (as it does here — one chart, with series data our pass reads fine), the fix drops the
    /// synthetic section, leaving the chart's numbers indexed exactly once, in our chunk, with
    /// the sheet's own text untouched.
    #[test]
    fn xlsx_chart_text_is_not_duplicated() {
        let bytes = include_bytes!("../../tests/fixtures/sample_chart.xlsx");
        let chunks = OfficeExtractor
            .extract(Path::new("sample_chart.xlsx"), bytes)
            .unwrap();
        let joined = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        // "4.4" is a cached value from the chart's second series and appears nowhere on the
        // sheet itself — the only way it reaches the index is through chart extraction, so it
        // is the clean signal for "did the same chart data land twice".
        assert_eq!(
            joined.matches("4.4").count(),
            1,
            "chart value must be indexed exactly once, got:\n{joined}"
        );

        // Our well-formed chunk is the one that survived.
        assert!(
            chunks
                .iter()
                .any(|c| c.text.starts_with("Chart: Sales by quarter")),
            "expected the structured chart chunk to survive, got: {chunks:?}"
        );

        // The fix did not eat a real section: the sheet's own cell text is still there.
        assert!(
            joined.contains("glossa sample"),
            "expected the sheet's own marker text, got:\n{joined}"
        );
    }

    /// The data-loss guard: when `office_oxide`'s synthetic section names a chart part our own
    /// pass did not cover — here, a chart with only a `<c:title>` and no `<c:ser>` at all, so
    /// our pass finds no series and skips it — the fix must keep BOTH sections rather than drop
    /// the one `office_oxide` built from it, or that chart's title would reach the index
    /// nowhere at all. Built by repacking the validated `sample_chart.xlsx` fixture with its
    /// chart part swapped for a title-only one, so everything else (the sheet, the rels chain)
    /// stays the already-exercised fixture.
    #[test]
    fn xlsx_keeps_both_sections_when_our_pass_misses_a_present_chart_part() {
        use std::io::{Read as _, Write as _};
        const TITLE_ONLY_CHART: &[u8] = br#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<c:chart><c:title><c:tx><c:rich><a:p><a:r><a:t>Sales by quarter</a:t></a:r></a:p></c:rich></c:tx></c:title>
<c:plotArea><c:barChart></c:barChart></c:plotArea></c:chart></c:chartSpace>"#;

        let mut zr = zip::ZipArchive::new(Cursor::new(
            include_bytes!("../../tests/fixtures/sample_chart.xlsx").to_vec(),
        ))
        .unwrap();
        let names: Vec<String> = zr
            .file_names()
            .filter(|n| !n.ends_with('/'))
            .map(|s| s.to_string())
            .collect();
        let mut buf = Vec::new();
        {
            let mut zw = zip::ZipWriter::new(Cursor::new(&mut buf));
            let opts = zip::write::SimpleFileOptions::default();
            for name in names {
                let mut bytes = Vec::new();
                zr.by_name(&name).unwrap().read_to_end(&mut bytes).unwrap();
                zw.start_file(&name, opts).unwrap();
                zw.write_all(if name == "xl/charts/chart1.xml" {
                    TITLE_ONLY_CHART
                } else {
                    bytes.as_slice()
                })
                .unwrap();
            }
            zw.finish().unwrap();
        }

        let chunks = OfficeExtractor
            .extract(Path::new("sample_chart_title_only.xlsx"), &buf)
            .unwrap();
        let joined = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        // Our own pass found no series in this chart, so it produced no "Chart: ..." chunk —
        // the synthetic section is the only place this chart's title could reach the index.
        assert!(
            !chunks.iter().any(|c| c.text.starts_with("Chart:")),
            "our chart pass should not have covered a title-only chart, got: {chunks:?}"
        );
        assert!(
            joined.contains("Title: Sales by quarter"),
            "expected office_oxide's synthetic section to survive (our pass did not cover its \
             only chart part), got:\n{joined}"
        );

        // The fix did not touch the sheet's own text.
        assert!(joined.contains("glossa sample"), "got:\n{joined}");
    }

    /// The synthetic-section predicate is structural, not title-based: a real worksheet that a
    /// user happens to name "Charts" — holding an ordinary table, not the heading/paragraph
    /// alternation `office_oxide` emits — must not be mistaken for the synthetic section.
    #[test]
    fn a_real_sheet_named_charts_is_not_mistaken_for_the_synthetic_section() {
        use office_oxide::ir::{
            DocumentIR, Element, Paragraph, Section, Table, TableCell, TableRow,
        };

        let ir = DocumentIR {
            sections: vec![Section {
                title: Some("Charts".to_string()),
                elements: vec![Element::Table(Table {
                    rows: vec![TableRow {
                        cells: vec![TableCell {
                            content: vec![Element::Paragraph(Paragraph::default())],
                            ..Default::default()
                        }],
                        ..Default::default()
                    }],
                    ..Default::default()
                })],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(synthetic_xlsx_chart_section_chart_count(&ir), None);
    }

    /// The positive case: office_oxide's actual shape (two charts) is recognized and counted.
    #[test]
    fn the_synthetic_charts_section_shape_is_recognized_and_counted() {
        use office_oxide::ir::{
            DocumentIR, Element, Heading, InlineContent, Paragraph, Section, TextSpan,
        };

        let chart_heading = |n: usize| {
            Element::Heading(Heading {
                level: 3,
                content: vec![InlineContent::Text(TextSpan {
                    text: format!("Chart {n}"),
                    ..Default::default()
                })],
                ..Default::default()
            })
        };
        let ir = DocumentIR {
            sections: vec![Section {
                title: Some("Charts".to_string()),
                elements: vec![
                    chart_heading(1),
                    Element::Paragraph(Paragraph::default()),
                    chart_heading(2),
                    Element::Paragraph(Paragraph::default()),
                ],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(synthetic_xlsx_chart_section_chart_count(&ir), Some(2));
    }
}
