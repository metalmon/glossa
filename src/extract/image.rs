use crate::extract::Extractor;
use crate::model::Chunk;
use std::path::Path;

pub struct ImageExtractor;

/// Build a searchable label from the parent folder name + file stem, with separators turned into
/// spaces (so an image is findable by name/topic via search/glob). No pixels are read at index time.
fn label_for(path: &Path) -> String {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let parent = path.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()).unwrap_or("");
    let raw = format!("{parent} {stem}");
    raw.chars().map(|c| if c == '_' || c == '-' || c == '.' { ' ' } else { c }).collect::<String>()
        .split_whitespace().collect::<Vec<_>>().join(" ")
}

impl Extractor for ImageExtractor {
    fn file_types(&self) -> &'static [&'static str] {
        &["png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff"]
    }

    fn extract(&self, path: &Path, _bytes: &[u8]) -> anyhow::Result<Vec<Chunk>> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("img").to_lowercase();
        Ok(vec![Chunk {
            doc_path: path.to_path_buf(),
            location: "(image)".into(),
            file_type: ext,
            text: label_for(path),
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes_image_by_folder_and_stem() {
        let chunks = ImageExtractor
            .extract(Path::new("kb/Схемы/profibus_сегмент-2.png"), b"\x89PNG")
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].file_type, "png");
        assert_eq!(chunks[0].location, "(image)");
        assert!(chunks[0].text.contains("Схемы") && chunks[0].text.contains("profibus") && chunks[0].text.contains("сегмент"));
    }
}
