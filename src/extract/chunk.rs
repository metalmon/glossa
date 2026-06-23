use crate::model::Chunk;
use std::path::Path;

pub fn parse_atx_heading(line: &str) -> Option<(usize, String)> {
    let t = line.trim_start();
    if !t.starts_with('#') {
        return None;
    }
    let hashes = t.chars().take_while(|c| *c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = &t[hashes..];
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }
    let title = rest.trim();
    if title.is_empty() {
        return None;
    }
    Some((hashes, title.to_string()))
}

fn push_chunk(
    path: &Path,
    heading_path: &[String],
    file_type: &str,
    buf: &mut String,
    out: &mut Vec<Chunk>,
) {
    if buf.trim().is_empty() {
        buf.clear();
        return;
    }
    out.push(Chunk {
        doc_path: path.to_path_buf(),
        location: heading_path.join(" > "),
        file_type: file_type.to_string(),
        text: std::mem::take(buf),
    });
}

/// Split Markdown (or Markdown rendered from another format) into heading-scoped chunks.
pub fn chunk_markdown(path: &Path, text: &str, file_type: &str) -> Vec<Chunk> {
    let mut out = Vec::new();
    let mut heading_path: Vec<String> = Vec::new();
    let mut buf = String::new();

    for line in text.lines() {
        if let Some((level, title)) = parse_atx_heading(line) {
            push_chunk(path, &heading_path, file_type, &mut buf, &mut out);
            heading_path.truncate(level.saturating_sub(1));
            heading_path.push(title);
        } else {
            buf.push_str(line);
            buf.push('\n');
        }
    }
    push_chunk(path, &heading_path, file_type, &mut buf, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_type_is_propagated_into_chunks() {
        let chunks = chunk_markdown(Path::new("x.docx"), "# H\nbody\n", "docx");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].file_type, "docx");
        assert_eq!(chunks[0].location, "H");
    }
}
