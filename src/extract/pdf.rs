use crate::extract::Extractor;
use crate::model::Chunk;
use anyhow::anyhow;
use std::path::Path;

pub struct PdfExtractor;

impl Extractor for PdfExtractor {
    fn file_types(&self) -> &'static [&'static str] {
        &["pdf"]
    }

    fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>> {
        let pages = pdf_extract::extract_text_from_mem_by_pages(bytes)
            .map_err(|e| anyhow!("pdf parse failed for {}: {e}", path.display()))?;
        let text = pages.join("\n");
        Ok(vec![Chunk {
            doc_path: path.to_path_buf(),
            location: String::new(),
            file_type: "pdf".into(),
            text,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_text_from_pdf_fixture() {
        let bytes = include_bytes!("../../tests/fixtures/sample.pdf");
        let chunks = PdfExtractor.extract(Path::new("sample.pdf"), bytes).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].file_type, "pdf");
        assert!(
            chunks[0].text.contains("glossa sample"),
            "expected fixture marker text, got: {}",
            chunks[0].text
        );
    }
}
