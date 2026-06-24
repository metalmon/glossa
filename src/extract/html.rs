use crate::extract::text::{decode_all, Windower};
use crate::model::Chunk;
use std::path::Path;

/// Strip tags from HTML: drop <script>/<style> bodies, remove all tags, decode common entities,
/// collapse runs of blank lines.
pub fn strip_html(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let lower = input.to_lowercase();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            // Skip <script>...</script> and <style>...</style> bodies entirely.
            for (tag, end) in [("<script", "</script>"), ("<style", "</style>")] {
                if lower[i..].starts_with(tag) {
                    if let Some(rel) = lower[i..].find(end) {
                        i += rel + end.len();
                    } else {
                        i = bytes.len();
                    }
                    out.push(' ');
                    break;
                }
            }
            // Skip a normal tag <...>.
            if i < bytes.len() && bytes[i] == b'<' {
                if let Some(rel) = input[i..].find('>') {
                    i += rel + 1;
                    out.push(' ');
                    continue;
                } else {
                    break;
                }
            }
            continue;
        }
        out.push(input[i..].chars().next().unwrap());
        i += input[i..].chars().next().unwrap().len_utf8();
    }
    let decoded = out
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ");
    // Collapse blank-line runs.
    let mut result = String::with_capacity(decoded.len());
    let mut blanks = 0;
    for line in decoded.lines() {
        if line.trim().is_empty() {
            blanks += 1;
            if blanks > 1 {
                continue;
            }
            result.push('\n');
        } else {
            blanks = 0;
            result.push_str(line.trim());
            result.push('\n');
        }
    }
    result
}

/// Read an HTML file, strip it, and window the text into chunks.
pub fn stream(path: &Path, file_type: &str, sink: &mut dyn FnMut(Chunk)) -> anyhow::Result<()> {
    let bytes = std::fs::read(path)?;
    let Some(text) = decode_all(&bytes) else { return Ok(()) };
    let stripped = strip_html(&text);
    let mut win = Windower::new(path, file_type);
    for line in stripped.lines() {
        win.push_line(line, sink);
    }
    win.finish(sink);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_tags_scripts_styles_and_entities() {
        let html = "<html><head><style>p{color:red}</style></head>\
                    <body><script>alert(1)</script><h1>Title</h1><p>Hello &amp; bye</p></body></html>";
        let s = strip_html(html);
        assert!(s.contains("Title"));
        assert!(s.contains("Hello & bye"));
        assert!(!s.contains("color:red"));
        assert!(!s.contains("alert"));
        assert!(!s.contains('<'));
    }

    #[test]
    fn stream_produces_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("d.html");
        std::fs::write(&p, b"<h1>Hi</h1><p>body text here</p>").unwrap();
        let mut out = Vec::new();
        stream(&p, "html", &mut |c| out.push(c)).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].text.contains("body text here"));
    }
}
