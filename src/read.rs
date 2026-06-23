use crate::walk::extractors;
use anyhow::Context;
use std::path::Path;

/// Read a document's text, optionally narrowed to a `location` (heading/sheet/page),
/// matched as a case-insensitive substring of the chunk's `location`.
pub fn read_region(path: &Path, location: Option<&str>) -> anyhow::Result<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut chunks = Vec::new();
    for ex in extractors() {
        if ex.file_types().contains(&ext.as_str()) {
            chunks = ex.extract(path, &bytes)?;
            break;
        }
    }
    let selected: Vec<&str> = match location {
        Some(loc) => {
            let needle = loc.to_lowercase();
            let matched: Vec<&str> = chunks
                .iter()
                .filter(|c| c.location.to_lowercase().contains(&needle))
                .map(|c| c.text.as_str())
                .collect();
            if matched.is_empty() {
                chunks.iter().map(|c| c.text.as_str()).collect()
            } else {
                matched
            }
        }
        None => chunks.iter().map(|c| c.text.as_str()).collect(),
    };
    Ok(selected.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_whole_then_narrows_by_location() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("d.md");
        std::fs::write(&p, b"# Intro\nalpha\n## Body\nbeta\n").unwrap();

        let whole = read_region(&p, None).unwrap();
        assert!(whole.contains("alpha") && whole.contains("beta"));

        let body = read_region(&p, Some("body")).unwrap();
        assert!(body.contains("beta") && !body.contains("alpha"));
    }

    #[test]
    fn unmatched_location_falls_back_to_whole_doc() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("d.md");
        std::fs::write(&p, b"# Intro\nalpha\n").unwrap();
        let out = read_region(&p, Some("nonexistent")).unwrap();
        assert!(out.contains("alpha"));
    }
}
