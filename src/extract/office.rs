use crate::extract::office_chunk::chunk_ir;
use crate::extract::office_table::expand_merged_tables;
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
        let mut ir = doc.to_ir();
        expand_merged_tables(&mut ir);
        let mut chunks = chunk_ir(path, &ir, &ext);
        chunks.extend(crate::extract::ooxml_chart::extract_charts(
            path, bytes, &ext,
        ));
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
}
