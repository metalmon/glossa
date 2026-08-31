use crate::extract::chunk::chunk_markdown;
use crate::extract::Extractor;
use crate::model::Chunk;
use std::path::Path;

pub struct MarkdownExtractor;

impl Extractor for MarkdownExtractor {
    fn file_types(&self) -> &'static [&'static str] {
        &["md", "markdown"]
    }

    fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>> {
        let text = crate::extract::text::decode_all(bytes).unwrap_or_default();
        Ok(chunk_markdown(path, &text, "md"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_by_headings_with_location_path() {
        let md = "# A\nintro\n## B\nbody b\n# C\nbody c\n";
        let chunks = MarkdownExtractor
            .extract(Path::new("d.md"), md.as_bytes())
            .unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].location, "A");
        assert_eq!(chunks[0].text.trim(), "intro");
        assert_eq!(chunks[1].location, "A > B");
        assert_eq!(chunks[1].text.trim(), "body b");
        assert_eq!(chunks[2].location, "C");
    }

    #[test]
    fn hash_without_space_is_not_a_heading() {
        let md = "#nothashtag is body\n";
        let chunks = MarkdownExtractor
            .extract(Path::new("d.md"), md.as_bytes())
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].location, "");
        assert!(chunks[0].text.contains("#nothashtag"));
    }

    #[test]
    fn empty_title_heading_is_body_not_heading() {
        let md = "# A\n## \nbody\n";
        let chunks = MarkdownExtractor
            .extract(Path::new("d.md"), md.as_bytes())
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].location, "A");
        assert!(!chunks[0].location.contains(" > "));
    }

    #[test]
    fn heading_level_jump_keeps_deterministic_path() {
        let md = "# A\n### C\nbody\n";
        let chunks = MarkdownExtractor
            .extract(Path::new("d.md"), md.as_bytes())
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].location, "A > C");
    }

    #[test]
    fn decodes_non_utf8_markdown() {
        // "# café" header in Windows-1252 (0x23 0x20 'c' 'a' 'f' then 0xE9 for é).
        let mut bytes = vec![b'#', b' ', b'c', b'a', b'f', 0xE9];
        bytes.push(b'\n');
        let chunks = MarkdownExtractor
            .extract(Path::new("latin.md"), &bytes)
            .unwrap();
        // Header-only file: still indexed as one chunk carrying the heading text, so a title-only
        // doc stays findable by its title instead of vanishing from search.
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("caf"));
        // Now with a body line.
        let mut b2 = bytes.clone();
        b2.extend_from_slice("body\n".as_bytes());
        let c2 = MarkdownExtractor
            .extract(Path::new("latin.md"), &b2)
            .unwrap();
        assert_eq!(c2.len(), 1);
        assert_eq!(c2[0].location, "café");
    }
}
