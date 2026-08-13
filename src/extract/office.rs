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
        chunks.extend(crate::extract::ooxml_chart::extract_charts(path, bytes, &ext));
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
