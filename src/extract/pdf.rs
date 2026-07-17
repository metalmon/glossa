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
                for i in 0..page_count as usize {
                    let mut text = doc.extract_text(i).unwrap_or_default();
                    if text.trim().is_empty() {
                        text = doc
                            .extract_tables(i)
                            .unwrap_or_default()
                            .iter()
                            .map(|table| table_to_markdown(&expand_table(table)))
                            .filter(|markdown| !markdown.trim().is_empty())
                            .collect::<Vec<_>>()
                            .join("\n\n");
                    }
                    out.push(Chunk {
                        doc_path: path_buf.clone(),
                        location: format!("p.{}", i + 1),
                        file_type: "pdf".into(),
                        text,
                    });
                }

                pad_pdf_page_stubs(&mut out, &path_buf, page_count);
                out
            }));

        let out = caught.unwrap_or_default();
        if !out.is_empty() {
            return Ok(out);
        }

        // Layer 3: only unparseable PDFs (open failure, zero pages, or a caught panic) reach this
        // filename fallback. Parseable scan-only PDFs keep one empty p.N chunk per physical page.
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

    fn blank_pdf(page_count: usize) -> Vec<u8> {
        let mut objects = vec![
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            format!(
                "<< /Type /Pages /Kids [{}] /Count {page_count} >>",
                (0..page_count)
                    .map(|i| format!("{} 0 R", i + 3))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
        ];
        objects.extend((0..page_count).map(|_| {
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources <<>> >>"
                .to_string()
        }));

        let mut bytes = b"%PDF-1.4\n".to_vec();
        let mut offsets = vec![0usize];
        for (i, object) in objects.iter().enumerate() {
            offsets.push(bytes.len());
            bytes.extend_from_slice(format!("{} 0 obj\n{object}\nendobj\n", i + 1).as_bytes());
        }
        let xref = bytes.len();
        bytes.extend_from_slice(format!("xref\n0 {}\n", objects.len() + 1).as_bytes());
        bytes.extend_from_slice(b"0000000000 65535 f \n");
        for offset in offsets.iter().skip(1) {
            bytes.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
        }
        bytes.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
                objects.len() + 1
            )
            .as_bytes(),
        );
        bytes
    }

    #[test]
    fn scan_only_pdf_keeps_one_empty_chunk_per_page() {
        // A parseable scan has physical pages but no text layer. Those page stubs are required
        // so page_image can address every page; only unparseable PDFs use the filename fallback.
        let chunks = PdfExtractor
            .extract(Path::new("three-page-scan.pdf"), &blank_pdf(3))
            .unwrap();

        assert_eq!(chunks.len(), 3, "expected physical page stubs: {chunks:?}");
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.location, format!("p.{}", i + 1));
            assert!(chunk.text.is_empty(), "blank page must stay empty: {chunk:?}");
        }
    }

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

    /// Regression: real-world standards PDFs have physically blank separator pages (p.4).
    #[test]
    fn local_sample_extract_includes_blank_page_four() {
        let path = std::path::Path::new("kb-local/sample_spec.pdf");
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
    fn local_sample_reindex_puts_page_four_in_index() {
        use crate::index::store::{index_dir, DocIndex};
        let kb = std::path::Path::new("kb-local");
        let pdf = kb.join("sample_spec.pdf");
        if !pdf.exists() {
            return;
        }
        index_dir(kb, true).unwrap();
        let idx = DocIndex::open_or_create(kb).unwrap();
        let hit = idx.read_chunk_by_ord("sample_spec.pdf", 4).unwrap();
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
        // cell VALUES are preserved). The markdown-table partition is a fallback when pdf_oxide
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
