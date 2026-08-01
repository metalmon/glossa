//! GEPA anchor helpers for per-step prompts in `SOP.md`.

use anyhow::{Context, Result};
use std::path::Path;

pub const GEPA_SLICE_TAGS: &[&str] =
    &["DISCOVER", "MATERIALIZE", "COMPILE", "COVERAGE", "VALIDATE"];

/// Extract optimizable text between `{# GEPA:TAG_START #}` … `{# GEPA:TAG_END #}`.
pub fn extract_gepa_slice(md: &str, tag: &str) -> Result<String> {
    let start = format!("{{# GEPA:{tag}_START #}}");
    let end = format!("{{# GEPA:{tag}_END #}}");
    let start_idx = md
        .find(&start)
        .with_context(|| format!("GEPA anchor {start} not found"))?;
    let after_start = start_idx + start.len();
    let end_idx = md[after_start..]
        .find(&end)
        .with_context(|| format!("GEPA anchor {end} not found after {start}"))?;
    Ok(md[after_start..after_start + end_idx].trim().to_string())
}

/// Replace the slice between GEPA anchors; preserves anchor markers.
pub fn apply_gepa_slice(md: &str, tag: &str, body: &str) -> Result<String> {
    let start = format!("{{# GEPA:{tag}_START #}}");
    let end = format!("{{# GEPA:{tag}_END #}}");
    let start_idx = md
        .find(&start)
        .with_context(|| format!("GEPA anchor {start} not found"))?;
    let after_start = start_idx + start.len();
    let end_idx = md[after_start..]
        .find(&end)
        .with_context(|| format!("GEPA anchor {end} not found after {start}"))?;
    let mut out = String::new();
    out.push_str(&md[..after_start]);
    out.push('\n');
    out.push_str(body.trim());
    out.push('\n');
    out.push_str(&md[after_start + end_idx..]);
    Ok(out)
}

pub fn load_sop_md(sop_dir: &Path) -> Result<String> {
    let path = sop_dir.join("SOP.md");
    std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))
}

pub fn load_all_gepa_seeds(sop_dir: &Path) -> Result<Vec<(String, String)>> {
    let md = load_sop_md(sop_dir)?;
    GEPA_SLICE_TAGS
        .iter()
        .map(|tag| extract_gepa_slice(&md, tag).map(|body| (tag.to_string(), body)))
        .collect()
}

/// Remove GEPA anchor marker lines before showing step body to the agent.
pub fn strip_gepa_anchor_lines(text: &str) -> String {
    text.lines()
        .filter(|line| !is_gepa_anchor_line(line))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn is_gepa_anchor_line(line: &str) -> bool {
    let t = line.trim();
    t.starts_with("{# GEPA:") && t.ends_with("#}")
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"## Steps

1. **Discover**

   {# GEPA:DISCOVER_START #}
   Search with grep first.
   {# GEPA:DISCOVER_END #}

   - tools: grep
"#;

    #[test]
    fn extract_and_apply_roundtrip() {
        let body = extract_gepa_slice(FIXTURE, "DISCOVER").unwrap();
        assert_eq!(body, "Search with grep first.");

        let updated = apply_gepa_slice(FIXTURE, "DISCOVER", "Prefer search then grep.").unwrap();
        assert!(updated.contains("Prefer search then grep."));
        assert!(!updated.contains("Search with grep first."));
    }

    #[test]
    fn strip_gepa_anchor_lines_removes_markers_only() {
        let text = "{# GEPA:DISCOVER_START #}\nDo work.\n{# GEPA:DISCOVER_END #}";
        assert_eq!(strip_gepa_anchor_lines(text), "Do work.");
    }
}
