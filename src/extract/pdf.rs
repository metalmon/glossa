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
        use oxidize_pdf::parser::{PdfDocument, PdfReader};

        // Any PDF parser can panic on a malformed file; catch it so indexing never aborts.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let owned = bytes.to_vec();
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let reader = PdfReader::new(std::io::Cursor::new(owned))?;
            let doc = PdfDocument::new(reader);
            doc.extract_text()
        }));
        std::panic::set_hook(prev);

        let pages = match caught {
            Ok(Ok(p)) => p,
            Ok(Err(e)) => return Err(anyhow!("pdf parse failed for {}: {e}", path.display())),
            Err(_) => return Err(anyhow!("pdf parser panicked for {} (skipped)", path.display())),
        };

        // One chunk per page; skip pages with no text layer (blank / scanned-image-only).
        // location = "p.N" (1-based page number).
        let mut out = Vec::new();
        for (i, page) in pages.iter().enumerate() {
            if page.text.trim().is_empty() {
                continue;
            }
            out.push(Chunk {
                doc_path: path.to_path_buf(),
                location: format!("p.{}", i + 1),
                file_type: "pdf".into(),
                text: page.text.clone(),
            });
        }
        if out.is_empty() {
            return Err(anyhow!(
                "no extractable text in {} (image-only or unparseable)",
                path.display()
            ));
        }
        Ok(out)
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
        assert_eq!(chunks.len(), 1, "single-page fixture → one page chunk");
        assert_eq!(chunks[0].file_type, "pdf");
        assert_eq!(chunks[0].location, "p.1");
        assert!(
            chunks[0].text.contains("glossa sample"),
            "expected fixture marker text, got: {}",
            chunks[0].text
        );
    }
}
