use crate::extract::image::ImageExtractor;
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
        Box::new(ImageExtractor),
        Box::new(MarkdownExtractor),
        Box::new(OfficeExtractor),
        Box::new(PdfExtractor),
    ]
}

/// Enumerate indexable files under `root` (gitignore-aware, skipping `.glossa`), calling `visit`
/// for each file path.
pub fn walk_files(
    root: &Path,
    glob: Option<&str>,
    respect_ignore: bool,
    visit: &mut dyn FnMut(&Path) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let matcher = match glob {
        Some(g) => Some(Glob::new(g)?.compile_matcher()),
        None => None,
    };
    let mut wb = WalkBuilder::new(root);
    wb.standard_filters(respect_ignore);
    wb.require_git(!respect_ignore);
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
        if let Err(e) = visit(path) {
            eprintln!("skip {}: {}", path.display(), e);
        }
    }
    Ok(())
}

/// Collect all chunks under `root` into a Vec (thin wrapper over the streaming pipeline; for `read`
/// and tests — `index_dir` streams instead).
pub fn collect_chunks(root: &Path, glob: Option<&str>, respect_ignore: bool) -> anyhow::Result<Vec<Chunk>> {
    let mut all = Vec::new();
    walk_files(root, glob, respect_ignore, &mut |path| {
        crate::extract::extract_file(path, &mut |c| all.push(c))
    })?;
    Ok(all)
}

#[cfg(test)]
mod cover_tests {
    use super::*;

    #[test]
    fn collect_indexes_text_json_code_skips_binary() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"plain text alpha").unwrap();
        std::fs::write(dir.path().join("b.json"), br#"{"key":"jsonvalue"}"#).unwrap();
        std::fs::write(dir.path().join("c.rs"), b"fn beta() {}").unwrap();
        std::fs::write(dir.path().join("d.png"), [0x89, b'P', 0x00, 0x01]).unwrap();
        let chunks = collect_chunks(dir.path(), None, false).unwrap();
        let joined: String = chunks.iter().map(|c| c.text.clone()).collect::<Vec<_>>().join("\n");
        assert!(joined.contains("alpha"));
        assert!(joined.contains("jsonvalue"));
        assert!(joined.contains("beta"));
        // the .png is indexed by name via ImageExtractor
        assert!(chunks.iter().any(|c| c.file_type == "png"));
    }
}
