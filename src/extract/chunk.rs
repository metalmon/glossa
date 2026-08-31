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
    // Strip inline Markdown emphasis/code markers so `# **Title**` yields a clean
    // location "Title"; otherwise the `*`/`` ` `` leak into the section id and its
    // display (e.g. an edge target shown as `path#**SECTION TITLE**`).
    let title = rest.trim().replace(['*', '`'], "");
    let title = title.trim();
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
    let mut all_headings: Vec<String> = Vec::new();
    let mut buf = String::new();

    for line in text.lines() {
        if let Some((level, title)) = parse_atx_heading(line) {
            push_chunk(path, &heading_path, file_type, &mut buf, &mut out);
            heading_path.truncate(level.saturating_sub(1));
            heading_path.push(title.clone());
            all_headings.push(title);
        } else {
            buf.push_str(line);
            buf.push('\n');
        }
    }
    push_chunk(path, &heading_path, file_type, &mut buf, &mut out);
    // A document made up ENTIRELY of headings (a title-only stub, no body text) would otherwise
    // produce zero chunks and vanish from search — the manifest records the file as indexed, yet
    // nothing is findable. Index the heading text as a single chunk so the doc stays searchable by
    // its title. Only fires when there is no body chunk at all; docs with any body are unaffected.
    if out.is_empty() && !all_headings.is_empty() {
        out.push(Chunk {
            doc_path: path.to_path_buf(),
            location: heading_path.join(" > "),
            file_type: file_type.to_string(),
            text: all_headings.join("\n"),
        });
    }
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

    #[test]
    fn heading_markdown_emphasis_is_stripped_from_location() {
        let chunks = chunk_markdown(
            Path::new("x.docx"),
            "# **SECTION** > `code`\nbody\n",
            "docx",
        );
        assert_eq!(chunks[0].location, "SECTION > code");
    }

    #[test]
    fn heading_only_doc_is_indexed_by_its_title() {
        // A title-only stub (all headings, no body) must still produce a searchable chunk whose text
        // carries the heading words — otherwise the doc is tracked in the manifest but invisible.
        let chunks = chunk_markdown(Path::new("stub.md"), "# Alpha doc\n", "md");
        assert_eq!(chunks.len(), 1, "heading-only doc yields one chunk");
        assert!(chunks[0].text.contains("Alpha") && chunks[0].text.contains("doc"));
        // Every heading is captured, even across multiple heading-only sections.
        let multi = chunk_markdown(Path::new("stub.md"), "# One\n## Two\n### Three\n", "md");
        assert_eq!(multi.len(), 1);
        assert!(
            multi[0].text.contains("One")
                && multi[0].text.contains("Two")
                && multi[0].text.contains("Three")
        );
    }

    #[test]
    fn doc_with_body_is_unaffected_by_heading_only_fallback() {
        // The fallback must NOT change chunking for a normal doc that has body text.
        let chunks = chunk_markdown(Path::new("x.md"), "# A\nintro\n## B\nbody b\n", "md");
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text.trim(), "intro");
        assert_eq!(chunks[1].text.trim(), "body b");
    }
}
