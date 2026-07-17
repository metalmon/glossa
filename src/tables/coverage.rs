//! Table-phase coverage: what the agent's `.csp` limit tables contain
//! (per-column value sets), for data-driven comparison against a reference.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::notebook::{mirror_dir_for_doc, notes_root};
use crate::tables::csp::{parse_csp, CspTable};

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

/// Column header → union of that column's cell values, across every `.csp` in
/// the document's notes mirror. Files that fail to parse are skipped (a
/// malformed table contributes nothing rather than aborting the whole scan);
/// columns with the same header in different files merge. Empty cells are
/// dropped.
pub fn csp_column_values(
    root: &Path,
    doc: &str,
) -> anyhow::Result<BTreeMap<String, BTreeSet<String>>> {
    let tables_dir = notes_root(root).join(mirror_dir_for_doc(doc));
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    if !tables_dir.is_dir() {
        return Ok(out);
    }
    for path in csp_files(&tables_dir)? {
        let text = std::fs::read_to_string(&path)?;
        let Ok(rel) = parse_csp(&text) else {
            continue;
        };
        for (i, header) in rel.headers.iter().enumerate() {
            if header.is_empty() {
                continue;
            }
            let col = out.entry(header.clone()).or_default();
            for row in &rel.rows {
                if let Some(cell) = row.get(i) {
                    if !cell.is_empty() {
                        col.insert(cell.clone());
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Per-file agent tables: `(filename, parsed table)` for every `.csp` in the
/// document's notes mirror. Files that fail to parse are skipped. Unlike
/// `csp_column_values` (which unions columns across files), this keeps each file
/// as its own table — needed to match a multi-column table against a reference
/// relation without collapsing combination rows.
pub fn csp_tables_per_file(
    root: &Path,
    doc: &str,
) -> anyhow::Result<Vec<(String, CspTable)>> {
    let tables_dir = notes_root(root).join(mirror_dir_for_doc(doc));
    let mut out = Vec::new();
    if !tables_dir.is_dir() {
        return Ok(out);
    }
    for path in csp_files(&tables_dir)? {
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let text = std::fs::read_to_string(&path)?;
        let Ok(rel) = parse_csp(&text) else {
            continue;
        };
        out.push((name, rel));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_values_union_across_files() {
        let dir = tempfile::tempdir().unwrap();
        let doc = "kb/test.pdf";
        let mirror = notes_root(dir.path()).join(mirror_dir_for_doc(doc));
        std::fs::create_dir_all(&mirror).unwrap();
        std::fs::write(mirror.join("a.csp"), "Width\tNote\n10\tok\n20\t\n").unwrap();
        std::fs::write(mirror.join("b.csp"), "Height\n5\n7\n").unwrap();
        std::fs::write(mirror.join("c.csp"), "Width\n30\n").unwrap();
        let cols = csp_column_values(dir.path(), doc).unwrap();
        let width: Vec<&str> = cols["Width"].iter().map(String::as_str).collect();
        assert_eq!(width, vec!["10", "20", "30"]);
        let height: Vec<&str> = cols["Height"].iter().map(String::as_str).collect();
        assert_eq!(height, vec!["5", "7"]);
        // Empty cell dropped, not stored as "".
        assert_eq!(
            cols["Note"].iter().collect::<Vec<_>>(),
            vec![&"ok".to_string()]
        );
        assert_eq!(count_csp_files(dir.path(), doc), 3);
    }

    #[test]
    fn malformed_csp_skipped_not_fatal() {
        let dir = tempfile::tempdir().unwrap();
        let doc = "kb/test.pdf";
        let mirror = notes_root(dir.path()).join(mirror_dir_for_doc(doc));
        std::fs::create_dir_all(&mirror).unwrap();
        std::fs::write(mirror.join("bad.csp"), "a\tb\n1\t2\t3\t4\n").unwrap();
        std::fs::write(mirror.join("good.csp"), "Width\n10\n").unwrap();
        let cols = csp_column_values(dir.path(), doc).unwrap();
        assert!(cols.contains_key("Width"));
        assert!(!cols.contains_key("a"));
    }

    #[test]
    fn empty_when_no_notes_dir() {
        let dir = tempfile::tempdir().unwrap();
        assert!(csp_column_values(dir.path(), "no/such.pdf")
            .unwrap()
            .is_empty());
        assert_eq!(count_csp_files(dir.path(), "no/such.pdf"), 0);
    }

    #[test]
    fn tables_per_file_keeps_files_separate() {
        let dir = tempfile::tempdir().unwrap();
        let doc = "kb/test.pdf";
        let mirror = notes_root(dir.path()).join(mirror_dir_for_doc(doc));
        std::fs::create_dir_all(&mirror).unwrap();
        // A multi-column combination table and a flat table.
        std::fs::write(mirror.join("height.csp"), "h\tD\n0,6\t125\n0,6\t150\n").unwrap();
        std::fs::write(mirror.join("voltage.csp"), "Voltage\n220\n380\n").unwrap();
        let tables = csp_tables_per_file(dir.path(), doc).unwrap();
        assert_eq!(tables.len(), 2);
        let height = tables.iter().find(|(f, _)| f == "height.csp").unwrap();
        assert_eq!(height.1.headers, vec!["h", "D"]);
        assert_eq!(height.1.rows.len(), 2);
        assert_eq!(height.1.rows[0], vec!["0,6", "125"]);
    }
}
