use crate::extract::image::ImageExtractor;
use crate::extract::markdown::MarkdownExtractor;
use crate::extract::office::OfficeExtractor;
use crate::extract::pdf::PdfExtractor;
use crate::extract::Extractor;
use crate::model::Chunk;
use ignore::WalkBuilder;
use std::path::Path;

pub fn extractors() -> Vec<Box<dyn Extractor>> {
    vec![
        Box::new(ImageExtractor),
        Box::new(MarkdownExtractor),
        Box::new(crate::extract::odf::OdfExtractor),
        Box::new(OfficeExtractor),
        Box::new(PdfExtractor),
    ]
}

/// Well-known OS/editor junk that must never be indexed as corpus content: Windows thumbnail
/// caches and folder settings, macOS metadata, and MS Office temp/lock files. Matched by name
/// (case-insensitive) so they are dropped in any directory, keeping the corpus signature clean.
pub(crate) fn is_junk_file(name: &std::ffi::OsStr) -> bool {
    let n = name.to_string_lossy();
    let lower = n.to_ascii_lowercase();
    matches!(lower.as_str(), "thumbs.db" | "desktop.ini" | ".ds_store")
        || n.starts_with("~$") // MS Office temp/owner-lock files
        || n.starts_with("._") // macOS AppleDouble resource forks
}

/// Enumerate indexable files under `root` (gitignore-aware, skipping `.glossa` and OS/editor junk),
/// calling `visit` for each file path.
pub fn walk_files(
    root: &Path,
    glob: Option<&str>,
    respect_ignore: bool,
    visit: &mut dyn FnMut(&Path) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let matcher = match glob {
        Some(g) => Some(crate::glob::compile_glob(g)?),
        None => None,
    };
    let mut wb = WalkBuilder::new(root);
    wb.standard_filters(respect_ignore);
    wb.require_git(!respect_ignore);
    wb.filter_entry(|e| e.file_name() != ".glossa" && !is_junk_file(e.file_name()));
    for result in wb.build() {
        let entry = match result {
            Ok(e) => e,
            Err(e) => {
                // An error ON THE ROOT means nothing below it was ever readable, so continuing
                // produces an empty pass that reports success. That is how a corpus on an
                // unreachable network share indexed zero documents and still exited 0 — the
                // operator saw "added: 0" and had no reason to look further. A per-entry error
                // deeper in the tree stays a skip: one unreadable file is not a failed corpus.
                if e.path() == Some(root) {
                    anyhow::bail!(
                        "corpus root is not readable: {} ({e}) — check the path, and for a network \
                         share that it is mounted and reachable by this user (a service runs as \
                         its own account, which may not see your mapped drives)",
                        root.display()
                    );
                }
                eprintln!("skip (walk error): {e}");
                continue;
            }
        };
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.path();
        if let Some(m) = &matcher {
            if !crate::glob::path_matches_fs(m, path) {
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
pub fn collect_chunks(
    root: &Path,
    glob: Option<&str>,
    respect_ignore: bool,
) -> anyhow::Result<Vec<Chunk>> {
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
    fn collect_indexes_text_json_code_and_images() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"plain text alpha").unwrap();
        std::fs::write(dir.path().join("b.json"), br#"{"key":"jsonvalue"}"#).unwrap();
        std::fs::write(dir.path().join("c.rs"), b"fn beta() {}").unwrap();
        std::fs::write(dir.path().join("d.png"), [0x89, b'P', 0x00, 0x01]).unwrap();
        let chunks = collect_chunks(dir.path(), None, false).unwrap();
        let joined: String = chunks
            .iter()
            .map(|c| c.text.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("alpha"));
        assert!(joined.contains("jsonvalue"));
        assert!(joined.contains("beta"));
        // the .png is indexed by name via ImageExtractor
        assert!(chunks.iter().any(|c| c.file_type == "png"));
    }

    /// An unreadable ROOT must fail, not report an empty success. This is the shape a corpus on an
    /// unreachable network share takes: the walk yields one error for the root, zero files, and the
    /// caller printed "added: 0" and exited 0 — so a mis-mounted share looked like an indexed
    /// corpus until someone searched it. A missing directory stands in for the share here; the
    /// failure path is identical and it needs no network.
    #[test]
    fn an_unreadable_root_is_an_error_not_an_empty_pass() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("no-such-corpus");

        let mut seen = 0usize;
        let err = walk_files(&missing, None, false, &mut |_| {
            seen += 1;
            Ok(())
        })
        .expect_err("an unreadable root must not look like an empty corpus");

        assert_eq!(seen, 0, "nothing could have been visited");
        let msg = err.to_string();
        assert!(
            msg.contains("corpus root is not readable"),
            "the error must name the cause, got: {msg}"
        );
        assert!(
            msg.contains("network share"),
            "the message must point at the common cause, got: {msg}"
        );
    }

    #[test]
    fn skips_os_and_office_junk_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("real.txt"), b"genuine corpus content").unwrap();
        // junk that must never be indexed
        std::fs::write(dir.path().join("Thumbs.db"), b"thumbnail cache junk").unwrap();
        std::fs::write(dir.path().join("desktop.ini"), b"[.ShellClassInfo] junk").unwrap();
        std::fs::write(dir.path().join(".DS_Store"), b"mac junk").unwrap();
        std::fs::write(dir.path().join("~$report.docx"), b"office lock junk").unwrap();
        std::fs::write(dir.path().join("._real.txt"), b"appledouble junk").unwrap();

        let mut seen: Vec<String> = Vec::new();
        walk_files(dir.path(), None, false, &mut |p| {
            seen.push(p.file_name().unwrap().to_string_lossy().into_owned());
            Ok(())
        })
        .unwrap();

        assert!(
            seen.contains(&"real.txt".to_string()),
            "real file must index"
        );
        for junk in [
            "Thumbs.db",
            "desktop.ini",
            ".DS_Store",
            "~$report.docx",
            "._real.txt",
        ] {
            assert!(!seen.contains(&junk.to_string()), "junk indexed: {junk}");
        }
    }
}
