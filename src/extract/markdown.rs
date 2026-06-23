use crate::extract::Extractor;
use crate::model::Chunk;
use std::path::Path;

pub struct MarkdownExtractor;

fn parse_atx_heading(line: &str) -> Option<(usize, String)> {
    let t = line.trim_start();
    if !t.starts_with('#') {
        return None;
    }
    let hashes = t.chars().take_while(|c| *c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = &t[hashes..];
    // CommonMark requires a space (or EOL) after the # run.
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }
    Some((hashes, rest.trim().to_string()))
}

fn push_chunk(path: &Path, heading_path: &[String], buf: &mut String, out: &mut Vec<Chunk>) {
    if buf.trim().is_empty() {
        buf.clear();
        return;
    }
    out.push(Chunk {
        doc_path: path.to_path_buf(),
        location: heading_path.join(" > "),
        file_type: "md".into(),
        text: std::mem::take(buf),
    });
}

impl Extractor for MarkdownExtractor {
    fn file_types(&self) -> &'static [&'static str] {
        &["md", "markdown"]
    }

    fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>> {
        let text = String::from_utf8_lossy(bytes);
        let mut out = Vec::new();
        let mut heading_path: Vec<String> = Vec::new();
        let mut buf = String::new();

        for line in text.lines() {
            if let Some((level, title)) = parse_atx_heading(line) {
                push_chunk(path, &heading_path, &mut buf, &mut out);
                heading_path.truncate(level.saturating_sub(1));
                heading_path.push(title);
            } else {
                buf.push_str(line);
                buf.push('\n');
            }
        }
        push_chunk(path, &heading_path, &mut buf, &mut out);
        Ok(out)
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
}
