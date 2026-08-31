use crate::model::Chunk;
use std::path::Path;

pub trait Extractor {
    /// Lower-case file extensions this extractor handles (e.g. `["md"]`).
    fn file_types(&self) -> &'static [&'static str];
    /// Extract a file's raw bytes into heading/section-scoped chunks.
    fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>>;
    /// Whether `extract` actually reads `bytes`. Extractors that key only off the PATH — images are
    /// indexed by filename, no pixels read — return `false` so `extract_file` skips reading the file
    /// body entirely. No point slurping a multi-MB (or huge scanned) image into memory to drop it.
    fn needs_bytes(&self) -> bool {
        true
    }
}

pub mod chunk;
pub mod csv_tsv;
pub mod html;
pub mod image;
pub mod links;
pub mod markdown;
pub mod odf;
pub mod odf_chart;
pub mod office;
pub mod office_chunk;
pub mod office_table;
pub mod ooxml_chart;
pub mod pdf;
pub mod text;

/// Extract one file's chunks into `sink`. Whole-file binary/doc formats (md/office/pdf) are read
/// fully; csv/tsv/html and any other readable file stream from the path (constant memory).
pub fn extract_file(path: &Path, sink: &mut dyn FnMut(Chunk)) -> anyhow::Result<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    for ex in crate::walk::extractors() {
        if ex.file_types().contains(&ext.as_str()) {
            // Skip reading the file body for extractors that don't use it (images → name-only).
            let bytes = if ex.needs_bytes() {
                std::fs::read(path)?
            } else {
                Vec::new()
            };
            for c in ex.extract(path, &bytes)? {
                sink(c);
            }
            return Ok(());
        }
    }
    match ext.as_str() {
        "csv" | "tsv" => csv_tsv::stream(path, &ext, sink),
        "html" | "htm" => html::stream(path, &ext, sink),
        other => {
            let ft = if other.is_empty() { "txt" } else { other };
            text::stream_text(path, ft, sink)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_file_does_not_read_an_image_body() {
        // A .png path that does NOT exist on disk: if extract_file tried to read the body it would
        // error. Because ImageExtractor::needs_bytes() is false, the read is skipped and we still get
        // the name-based chunk — proving the (potentially huge) image body is never slurped.
        let mut chunks = Vec::new();
        extract_file(Path::new("no/such/Diagrams/bus_segment.png"), &mut |c| {
            chunks.push(c)
        })
        .expect("image extraction must not read (or need) the file body");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].file_type, "png");
        assert!(chunks[0].text.contains("bus") && chunks[0].text.contains("segment"));
    }
}
