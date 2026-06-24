use crate::model::Chunk;
use std::path::Path;

pub trait Extractor {
    /// Lower-case file extensions this extractor handles (e.g. `["md"]`).
    fn file_types(&self) -> &'static [&'static str];
    /// Extract a file's raw bytes into heading/section-scoped chunks.
    fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>>;
}

pub mod chunk;
pub mod csv_tsv;
pub mod html;
pub mod markdown;
pub mod office;
pub mod pdf;
pub mod text;

/// Extract one file's chunks into `sink`. Whole-file binary/doc formats (md/office/pdf) are read
/// fully; csv/tsv/html and any other readable file stream from the path (constant memory).
pub fn extract_file(path: &Path, sink: &mut dyn FnMut(Chunk)) -> anyhow::Result<()> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    for ex in crate::walk::extractors() {
        if ex.file_types().contains(&ext.as_str()) {
            let bytes = std::fs::read(path)?;
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
