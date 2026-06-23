use crate::model::Chunk;
use std::path::Path;

pub trait Extractor {
    /// Lower-case file extensions this extractor handles (e.g. `["md"]`).
    fn file_types(&self) -> &'static [&'static str];
    /// Extract a file's raw bytes into heading/section-scoped chunks.
    fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>>;
}
