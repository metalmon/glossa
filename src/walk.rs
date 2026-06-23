use crate::extract::markdown::MarkdownExtractor;
use crate::extract::office::OfficeExtractor;
use crate::extract::pdf::PdfExtractor;
use crate::extract::Extractor;
use crate::model::Chunk;
use globset::Glob;
use ignore::WalkBuilder;
use std::path::Path;

pub fn extractors() -> Vec<Box<dyn Extractor>> {
    vec![
        Box::new(MarkdownExtractor),
        Box::new(OfficeExtractor),
        Box::new(PdfExtractor),
    ]
}

pub fn collect_chunks(
    root: &Path,
    glob: Option<&str>,
    respect_ignore: bool,
) -> anyhow::Result<Vec<Chunk>> {
    let matcher = match glob {
        Some(g) => Some(Glob::new(g)?.compile_matcher()),
        None => None,
    };
    let exts = extractors();
    let mut all = Vec::new();

    let mut wb = WalkBuilder::new(root);
    wb.standard_filters(respect_ignore); // gitignore/.ignore/hidden/parents
    // Apply .gitignore even outside a git repo (e.g. in tests with no .git dir).
    wb.require_git(!respect_ignore);
    // Always skip our own store, even when respect_ignore is false.
    wb.filter_entry(|e| e.file_name() != ".glossa");

    for result in wb.build() {
        let entry = match result {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skip (walk error): {e}");
                continue;
            }
        };
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
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
