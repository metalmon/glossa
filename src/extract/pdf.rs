use crate::extract::Extractor;
use crate::model::Chunk;
use std::path::Path;

pub struct PdfExtractor;

impl Extractor for PdfExtractor {
    fn file_types(&self) -> &'static [&'static str] {
        &["pdf"]
    }

    fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>> {
        use oxidize_pdf::parser::{PdfDocument, PdfReader, ParseOptions};
        use oxidize_pdf::text::ExtractionOptions;

        // Any PDF parser can panic on a malformed file; catch it so indexing never aborts.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let owned = bytes.to_vec();
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            // lenient() enables xref recovery — parses damaged-but-valid PDFs that strict mode rejects.
            let reader = PdfReader::new_with_options(std::io::Cursor::new(owned), ParseOptions::lenient())?;
            let doc = PdfDocument::new(reader);
            let opts = ExtractionOptions {
                preserve_layout: true,       // emit newlines + reconstruct spaces from glyph positions
                space_threshold: 0.3,        // horizontal gap > k·char-width → insert a space
                newline_threshold: 10.0,     // baseline (y) drop → newline
                merge_hyphenated: true,
                reconstruct_paragraphs: true,
                detect_columns: true,        // RU technical PDFs are often multi-column
                include_artifacts: false,    // drop headers/footers/watermarks
                ..Default::default()
            };
            doc.extract_text_with_options(opts)
        }));
        std::panic::set_hook(prev);

        // Per-page text chunks; skip pages with no text layer. location = "p.N" (1-based).
        let pages = match caught {
            Ok(Ok(p)) => p,
            // Parse error or panic: no pages — fall through to the filename fallback below.
            Ok(Err(_)) | Err(_) => Vec::new(),
        };
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
        if !out.is_empty() {
            return Ok(out);
        }

        // No extractable text (scanned / image-only) or unparseable: NEVER drop the document —
        // index it by filename so it's findable by name; the agent can vision-read the image later.
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("document")
            .to_string();
        eprintln!("  · no text layer, indexed by filename: {}", path.display());
        Ok(vec![Chunk {
            doc_path: path.to_path_buf(),
            location: "(no-text)".into(),
            file_type: "pdf".into(),
            text: name,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unparseable_pdf_is_indexed_by_filename_not_dropped() {
        // Garbage body: must NOT panic and must NOT be dropped — indexed by filename instead.
        let bytes = b"%PDF-1.4\ngarbage not a real pdf";
        let chunks = PdfExtractor.extract(Path::new("bad.pdf"), bytes).unwrap();
        assert_eq!(chunks.len(), 1, "one filename fallback chunk");
        assert_eq!(chunks[0].location, "(no-text)");
        assert_eq!(chunks[0].file_type, "pdf");
        assert!(
            chunks[0].text.contains("bad"),
            "fallback chunk text should be the filename stem, got: {}",
            chunks[0].text
        );
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
