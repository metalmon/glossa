use crate::extract::markdown::MarkdownExtractor;
use crate::extract::Extractor;
use crate::model::Chunk;
use globset::Glob;
use std::path::Path;
use walkdir::WalkDir;

pub fn extractors() -> Vec<Box<dyn Extractor>> {
    vec![Box::new(MarkdownExtractor)]
}

pub fn collect_chunks(root: &Path, glob: Option<&str>) -> anyhow::Result<Vec<Chunk>> {
    let matcher = match glob {
        Some(g) => Some(Glob::new(g)?.compile_matcher()),
        None => None,
    };
    let exts = extractors();
    let mut all = Vec::new();

    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if let Some(m) = &matcher {
            if !m.is_match(path) {
                continue;
            }
        }
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        for ex in &exts {
            if ex.file_types().contains(&ext.as_str()) {
                match std::fs::read(path) {
                    Ok(bytes) => match ex.extract(path, &bytes) {
                        Ok(mut cs) => all.append(&mut cs),
                        Err(e) => eprintln!("skip {}: {}", path.display(), e),
                    },
                    Err(e) => eprintln!("skip {}: {}", path.display(), e),
                }
                break;
            }
        }
    }
    Ok(all)
}
