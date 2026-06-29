use crate::extract::Extractor;
use crate::model::Chunk;
use std::path::Path;

pub struct PdfExtractor;

impl Extractor for PdfExtractor {
    fn file_types(&self) -> &'static [&'static str] {
        &["pdf"]
    }

    fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>> {
        use oxidize_pdf::parser::{ParseOptions, PdfDocument, PdfReader};
        use oxidize_pdf::pipeline::{Element, ElementMarkdownExporter, ExportConfig};
        use oxidize_pdf::text::ExtractionOptions;
        use std::collections::BTreeMap;

        // Any PDF parser can panic on a malformed file; catch it so indexing never aborts.
        let owned = bytes.to_vec();
        let path_buf = path.to_path_buf();
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || -> Vec<Chunk> {
            // lenient() enables xref recovery — parses damaged-but-valid PDFs that strict mode rejects.
            let reader = match PdfReader::new_with_options(std::io::Cursor::new(owned), ParseOptions::lenient()) {
                Ok(r) => r,
                Err(_) => return Vec::new(),
            };
            let doc = PdfDocument::new(reader);

            // Layer 1 (primary): plain layout-text reconstruction (preserve_layout rebuilds
            // spaces+newlines). oxidize-pdf's table PARTITION — the old primary — mis-detects
            // multi-column RU prose as tables and mangles words ("Выгр у зка", false pipe cells);
            // the flat layout text is clean for BOTH prose and tables, so it goes first.
            let opts = ExtractionOptions {
                preserve_layout: true,
                space_threshold: 0.3,        // horizontal gap > k·char-width → insert a space
                newline_threshold: 10.0,     // baseline (y) drop → newline
                merge_hyphenated: true,
                reconstruct_paragraphs: true,
                detect_columns: true,        // RU technical PDFs are often multi-column
                include_artifacts: false,    // drop headers/footers/watermarks
                ..Default::default()
            };
            if let Ok(pages) = doc.extract_text_with_options(opts) {
                let mut out = Vec::new();
                for (i, page) in pages.iter().enumerate() {
                    if page.text.trim().is_empty() {
                        continue;
                    }
                    out.push(Chunk {
                        doc_path: path_buf.clone(),
                        location: format!("p.{}", i + 1),
                        file_type: "pdf".into(),
                        text: page.text.clone(),
                    });
                }
                if !out.is_empty() {
                    return out;
                }
            }

            // Layer 2 (fallback): structural partition → per-page markdown (GFM tables), used ONLY
            // when layout-text found nothing — a PDF that exposes structure but no flat text stream.
            if let Ok(elements) = doc.partition() {
                if !elements.is_empty() {
                    let mut by_page: BTreeMap<u32, Vec<Element>> = BTreeMap::new();
                    for el in elements {
                        by_page.entry(el.page()).or_default().push(el);
                    }
                    let exporter = ElementMarkdownExporter::new(ExportConfig::default());
                    let mut out = Vec::new();
                    for (page, els) in by_page {
                        let md = exporter.export(&els);
                        if md.trim().is_empty() {
                            continue;
                        }
                        out.push(Chunk {
                            doc_path: path_buf.clone(),
                            location: format!("p.{}", page_label(page)),
                            file_type: "pdf".into(),
                            text: md,
                        });
                    }
                    if !out.is_empty() {
                        return out;
                    }
                }
            }
            Vec::new()
        }));

        let out = caught.unwrap_or_default();
        if !out.is_empty() {
            return Ok(out);
        }

        // Layer 3: no extractable text (scanned / image-only) or unparseable: NEVER drop the
        // document — index it by filename so it's findable by name.
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

/// Map oxidize-pdf's `Element.page()` to glossa's 1-based `p.N` convention.
/// `do_partition_pages` tags each element with its 0-based `page_idx` (from
/// `pages.iter().enumerate()`), so add 1 to reach glossa's 1-based pages.
fn page_label(page: u32) -> u32 {
    page + 1
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

    #[test]
    fn extracts_table_content_as_flat_text() {
        let bytes = include_bytes!("../../tests/fixtures/table.pdf");
        let chunks = PdfExtractor.extract(Path::new("table.pdf"), bytes).unwrap();
        // Layout-text is the primary path now: a table is flattened to readable rows on p.1 (its
        // cell VALUES are preserved). The markdown-table partition is a fallback because oxidize-pdf
        // mis-detects multi-column prose as tables and mangles the words. p.1 also locks the
        // 1-based `p.N` page mapping the read contract rests on.
        assert_eq!(chunks[0].location, "p.1");
        let joined = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n");
        for cell in ["Parametr", "Znachenie", "Tsvet", "Siniy"] {
            assert!(joined.contains(cell), "table cell '{cell}' missing from:\n{joined}");
        }
    }

    #[test]
    fn concurrent_pdf_extract_does_not_panic() {
        use std::thread;
        let bytes = include_bytes!("../../tests/fixtures/sample.pdf");
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let b = bytes.to_vec();
                thread::spawn(move || {
                    PdfExtractor.extract(Path::new("sample.pdf"), &b).unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }
}
