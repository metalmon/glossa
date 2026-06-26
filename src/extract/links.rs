//! Deterministic extraction of explicit document links (markdown + html), for building
//! cross-document REFERENCES graph edges. No model.

use regex::Regex;
use std::sync::OnceLock;

/// Markdown `[text](target)` and html `href="target"` targets, excluding external URLs
/// (http/https/mailto, protocol-relative `//`) and pure anchors (`#...`). A trailing
/// `#anchor` on an otherwise-local target is stripped.
pub fn extract_links(text: &str) -> Vec<String> {
    static MD: OnceLock<Regex> = OnceLock::new();
    static HTML: OnceLock<Regex> = OnceLock::new();
    // Capture an optional leading `!` so image embeds `![alt](img.png)` can be skipped — they
    // are not document references.
    let md = MD.get_or_init(|| Regex::new(r"(!?)\[[^\]]*\]\(([^)\s]+)\)").unwrap());
    let html = HTML.get_or_init(|| Regex::new(r#"(?i)href\s*=\s*["']([^"']+)["']"#).unwrap());
    let mut out = Vec::new();
    for caps in md.captures_iter(text) {
        if caps.get(1).is_some_and(|b| !b.as_str().is_empty()) {
            continue; // image embed, not a link
        }
        if let Some(m) = caps.get(2) { push_if_local(m.as_str(), &mut out); }
    }
    for caps in html.captures_iter(text) {
        if let Some(m) = caps.get(1) { push_if_local(m.as_str(), &mut out); }
    }
    out
}

fn push_if_local(target: &str, out: &mut Vec<String>) {
    let t = target.trim();
    let lower = t.to_ascii_lowercase();
    if t.is_empty()
        || t.starts_with('#')
        || lower.starts_with("http://")
        || lower.starts_with("https://")
        || lower.starts_with("mailto:")
        || t.starts_with("//")
    {
        return;
    }
    let path = t.split('#').next().unwrap_or(t);
    if !path.is_empty() {
        out.push(path.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_markdown_and_html_local_links_only() {
        let text = "see [B](b.md) and <a href=\"sub/c.html\">c</a> and \
                    [ext](https://x.com) and [anchor](#sec) and [d](d.md#part)";
        let links = extract_links(text);
        assert!(links.contains(&"b.md".to_string()));
        assert!(links.contains(&"sub/c.html".to_string()));
        assert!(links.contains(&"d.md".to_string()), "trailing #anchor stripped");
        assert!(!links.iter().any(|l| l.contains("x.com")), "external excluded");
        assert!(!links.iter().any(|l| l.starts_with('#')), "anchor excluded");
    }

    #[test]
    fn image_embeds_are_not_links() {
        let links = extract_links("![diagram](pic.png) but [real](doc.md)");
        assert!(links.contains(&"doc.md".to_string()), "real link kept");
        assert!(!links.iter().any(|l| l.contains("pic.png")), "image embed excluded");
    }
}
