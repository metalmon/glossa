//! Table-phase coverage: how many `.csp` limit tables the agent authored for a document.

use std::path::Path;

use crate::notebook::{mirror_dir_for_doc, notes_root};

fn csp_files(dir: &Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let mut files: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e.eq_ignore_ascii_case("csp")))
        .collect();
    files.sort();
    Ok(files)
}

pub fn count_csp_files(root: &Path, doc: &str) -> usize {
    let tables_dir = notes_root(root).join(mirror_dir_for_doc(doc));
    if !tables_dir.is_dir() {
        return 0;
    }
    csp_files(&tables_dir).map(|f| f.len()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_only_csp_files_and_zero_without_a_notes_dir() {
        let dir = tempfile::tempdir().unwrap();
        let doc = "kb/test.pdf";
        assert_eq!(count_csp_files(dir.path(), "no/such.pdf"), 0);
        let mirror = notes_root(dir.path()).join(mirror_dir_for_doc(doc));
        std::fs::create_dir_all(&mirror).unwrap();
        std::fs::write(mirror.join("height.csp"), "h\tD\n0,6\t125\n").unwrap();
        std::fs::write(mirror.join("voltage.CSP"), "Voltage\n220\n").unwrap();
        std::fs::write(mirror.join("notes.md"), "not a table").unwrap();
        assert_eq!(count_csp_files(dir.path(), doc), 2);
    }
}
