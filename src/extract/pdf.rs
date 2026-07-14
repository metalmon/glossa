use crate::extract::Extractor;
use crate::extract::pdf_table::{expand_table, table_to_markdown};
use crate::model::Chunk;
use std::path::Path;

pub struct PdfExtractor;

impl Extractor for PdfExtractor {
    fn file_types(&self) -> &'static [&'static str] {
        &["pdf"]
    }

    fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>> {
        use pdf_oxide::PdfDocument;

        // Any PDF parser can panic on a malformed file; catch it so indexing never aborts.
        let owned = bytes.to_vec();
        let path_buf = path.to_path_buf();
        let caught =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || -> Vec<Chunk> {
                let doc = match PdfDocument::from_bytes(owned) {
                    Ok(doc) => doc,
                    Err(_) => return Vec::new(),
                };
                let page_count = doc.page_count().unwrap_or(0) as u32;
                if page_count == 0 {
                    return Vec::new();
                }

                let mut out = Vec::with_capacity(page_count as usize);
                let mut any_text = false;
                for i in 0..page_count as usize {
                    let text = doc.extract_text(i).unwrap_or_default();
                    if !text.trim().is_empty() {
                        any_text = true;
                    }
                    out.push(Chunk {
                        doc_path: path_buf.clone(),
                        location: format!("p.{}", i + 1),
                        file_type: "pdf".into(),
                        text,
                    });
                }

                // If the text layer is empty on every page, preserve structural tables as markdown.
                if !any_text {
                    let mut filled = false;
                    for i in 0..page_count as usize {
                        let markdown = doc
                            .extract_tables(i)
                            .unwrap_or_default()
                            .iter()
                            .map(|table| table_to_markdown(&expand_table(table)))
                            .filter(|text| !text.trim().is_empty())
                            .collect::<Vec<_>>()
                            .join("\n\n");
                        if !markdown.trim().is_empty() {
                            out[i].text = markdown;
                            filled = true;
                        }
                    }
                    if !filled {
                        return Vec::new();
                    }
                }

                pad_pdf_page_stubs(&mut out, &path_buf, page_count);
                out
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

/// Ensure every physical page `1..=page_count` has a chunk (blank pages get empty body).
fn pad_pdf_page_stubs(out: &mut Vec<Chunk>, doc_path: &Path, page_count: u32) {
    if page_count == 0 {
        return;
    }
    let have: std::collections::HashSet<String> = out.iter().map(|c| c.location.clone()).collect();
    for p in 1..=page_count {
        let loc = format!("p.{p}");
        if !have.contains(&loc) {
            out.push(Chunk {
                doc_path: doc_path.to_path_buf(),
                location: loc,
                file_type: "pdf".into(),
                text: String::new(),
            });
        }
    }
    out.sort_by(|a, b| {
        let ord = |loc: &str| {
            loc.strip_prefix("p.")
                .and_then(|n| n.parse::<u32>().ok())
                .unwrap_or(0)
        };
        ord(&a.location).cmp(&ord(&b.location))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn pad_pdf_page_stubs_fills_gaps() {
        let path = PathBuf::from("d.pdf");
        let mut out = vec![
            Chunk {
                doc_path: path.clone(),
                location: "p.1".into(),
                file_type: "pdf".into(),
                text: "a".into(),
            },
            Chunk {
                doc_path: path.clone(),
                location: "p.3".into(),
                file_type: "pdf".into(),
                text: "c".into(),
            },
        ];
        pad_pdf_page_stubs(&mut out, &path, 3);
        assert_eq!(out.len(), 3);
        assert_eq!(out[1].location, "p.2");
        assert!(out[1].text.is_empty());
    }

    #[test]
    fn indexes_blank_pdf_page_as_empty_chunk() {
        let bytes = include_bytes!("../../tests/fixtures/three-page-blank-middle.pdf");
        let chunks = PdfExtractor
            .extract(Path::new("three-page-blank-middle.pdf"), bytes)
            .unwrap();
        assert_eq!(
            chunks.len(),
            3,
            "expected one chunk per physical page, got: {chunks:?}"
        );
        assert_eq!(chunks[0].location, "p.1");
        assert_eq!(chunks[1].location, "p.2");
        assert!(
            chunks[1].text.trim().is_empty(),
            "blank middle page must be indexed with empty body"
        );
        assert_eq!(chunks[2].location, "p.3");
        assert!(chunks[0].text.contains("page one"));
        assert!(chunks[2].text.contains("page three"));
    }

    /// Regression: real GOST PDFs have physically blank separator pages (p.4).
    #[test]
    fn gost_57978_extract_includes_blank_page_four() {
        let path = std::path::Path::new("kb-gost/gost_r_57978-2017.pdf");
        if !path.exists() {
            return;
        }
        let bytes = std::fs::read(path).unwrap();
        let chunks = PdfExtractor.extract(path, &bytes).unwrap();
        assert_eq!(
            chunks.len(),
            21,
            "expected 21 physical pages, got {}: {:?}",
            chunks.len(),
            chunks.iter().map(|c| &c.location).collect::<Vec<_>>()
        );
        let p4 = chunks
            .iter()
            .find(|c| c.location == "p.4")
            .expect("missing p.4 chunk");
        assert!(
            p4.text.trim().is_empty(),
            "p.4 must be empty, got {:?}",
            p4.text
        );
    }

    #[test]
    fn gost_57978_reindex_puts_page_four_in_index() {
        use crate::index::store::{index_dir, DocIndex};
        let kb = std::path::Path::new("kb-gost");
        let pdf = kb.join("gost_r_57978-2017.pdf");
        if !pdf.exists() {
            return;
        }
        index_dir(kb, true).unwrap();
        let idx = DocIndex::open_or_create(kb).unwrap();
        let hit = idx.read_chunk_by_ord("gost_r_57978-2017.pdf", 4).unwrap();
        assert!(hit.is_some(), "ord #4 must exist in index after reindex");
        assert!(hit.unwrap().body.trim().is_empty());
    }

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
        let chunks = PdfExtractor
            .extract(Path::new("sample.pdf"), bytes)
            .unwrap();
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
        let joined = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        for cell in ["Parametr", "Znachenie", "Tsvet", "Siniy"] {
            assert!(
                joined.contains(cell),
                "table cell '{cell}' missing from:\n{joined}"
            );
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
