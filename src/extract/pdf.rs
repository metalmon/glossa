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
        // pdf-extract can panic on malformed PDFs; catch it so indexing never aborts.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pdf_extract::extract_text_from_mem_by_pages(bytes)
        }));
        std::panic::set_hook(prev);

        let pages = match caught {
            Ok(Ok(p)) => p,
            Ok(Err(e)) => {
                return Err(anyhow!("pdf parse failed for {}: {e}", path.display()))
            }
            Err(_) => {
                return Err(anyhow!("pdf parser panicked for {} (skipped)", path.display()))
            }
        };
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
    fn malformed_pdf_does_not_panic_returns_err() {
        // Bytes with a PDF header but garbage body — must not abort the process.
        let bytes = b"%PDF-1.4\ngarbage not a real pdf";
        let r = PdfExtractor.extract(Path::new("bad.pdf"), bytes);
        assert!(r.is_err(), "malformed pdf should be an Err, not a panic/empty");
    }

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
