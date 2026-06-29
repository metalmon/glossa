//! Path-mask navigation over the knowledge base: ripgrep `-g` glob semantics via `globset`
//! (shared by the `glob` tool, grep's `-g` filter, and `search` scope). File-First: reads
//! stored index paths only.

use crate::index::store::DocIndex;
use globset::{GlobBuilder, GlobMatcher};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::Path;

/// Discovery aliases: agents often send `**/*` to list everything.
pub fn normalize_pattern(pattern: &str) -> &str {
    match pattern.trim() {
        "" | "**" | "**/*" | "**/**" => "**",
        other => other,
    }
}

/// Compile a ripgrep `-g` glob (not `--iglob`: case-sensitive).
pub fn compile_glob(pattern: &str) -> anyhow::Result<GlobMatcher> {
    GlobBuilder::new(normalize_pattern(pattern))
        .literal_separator(false)
        .backslash_escape(true)
        .build()
        .map(|g| g.compile_matcher())
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Indexed paths on Windows use `\`; agent globs typically use `/`.
pub fn normalize_path_for_glob(path: &str) -> Cow<'_, str> {
    if path.contains('\\') {
        Cow::Owned(path.replace('\\', "/"))
    } else {
        Cow::Borrowed(path)
    }
}

/// Match a stored corpus-relative path against a compiled glob.
pub fn path_matches(matcher: &GlobMatcher, stored_path: &str) -> bool {
    matcher.is_match(normalize_path_for_glob(stored_path).as_ref())
}

/// Match a filesystem path (walk/index boundary).
pub fn path_matches_fs(matcher: &GlobMatcher, path: &Path) -> bool {
    matcher.is_match(path)
}

/// List the DISTINCT document paths whose path matches `pattern`, each with its highest chunk
/// number (≈ page/section count, the `n`-range for `read(path, n)`). Sorted by path.
pub fn glob_docs(idx: &DocIndex, pattern: &str) -> anyhow::Result<Vec<(String, u64)>> {
    let matcher = compile_glob(pattern)?;
    let mut by_path: BTreeMap<String, u64> = BTreeMap::new();
    idx.iter_chunks(|path, ord, _ft, _body| {
        if path_matches(&matcher, path) {
            let e = by_path.entry(path.to_string()).or_insert(0);
            if ord > *e {
                *e = ord;
            }
        }
    })?;
    Ok(by_path.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::store::DocIndex;
    use crate::model::Chunk;
    use std::path::PathBuf;

    fn idx_with(chunks: &[(&str, &str, &str)]) -> (tempfile::TempDir, DocIndex) {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let cs: Vec<Chunk> = chunks
            .iter()
            .map(|(p, loc, t)| Chunk {
                doc_path: PathBuf::from(p),
                location: (*loc).into(),
                file_type: "pdf".into(),
                text: (*t).into(),
            })
            .collect();
        idx.write_chunks(&cs).unwrap();
        (dir, idx)
    }

    fn idx_with_types(chunks: &[(&str, &str, &str, &str)]) -> (tempfile::TempDir, DocIndex) {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let cs: Vec<Chunk> = chunks
            .iter()
            .map(|(p, loc, ft, t)| Chunk {
                doc_path: PathBuf::from(p),
                location: (*loc).into(),
                file_type: (*ft).into(),
                text: (*t).into(),
            })
            .collect();
        idx.write_chunks(&cs).unwrap();
        (dir, idx)
    }

    #[test]
    fn glob_docs_lists_distinct_matching_paths_with_counts() {
        let (_d, idx) = idx_with(&[
            ("kb\\Руководство АБАК.pdf", "p.1", "a"),
            ("kb\\Руководство АБАК.pdf", "p.2", "b"),
            ("kb\\Safety Manual.pdf", "p.1", "c"),
            ("kb\\Прочее.md", "S1", "d"),
        ]);
        let pdfs = glob_docs(&idx, "*.pdf").unwrap();
        // BTreeMap sorts by UTF-8 byte order: 'S' (0x53) < 'Р' (0xD0), so Safety sorts first.
        assert_eq!(
            pdfs,
            vec![
                ("kb\\Safety Manual.pdf".to_string(), 1),
                ("kb\\Руководство АБАК.pdf".to_string(), 2),
            ]
        );
        let abak = glob_docs(&idx, "*АБАК*").unwrap();
        assert_eq!(abak, vec![("kb\\Руководство АБАК.pdf".to_string(), 2)]);
        assert!(glob_docs(&idx, "*nomatch*").unwrap().is_empty());
    }

    #[test]
    fn glob_rg_recursive_on_windows_paths() {
        let (_d, idx) = idx_with(&[
            ("top.pdf", "p.1", "a"),
            ("dir\\nested.pdf", "p.1", "b"),
            ("dir\\sub\\deep.htm", "S1", "c"),
        ]);
        let all = glob_docs(&idx, "**/*").unwrap();
        assert_eq!(all.len(), 3);
        let pdfs = glob_docs(&idx, "**/*.pdf").unwrap();
        assert_eq!(pdfs.len(), 2);
        assert!(pdfs.iter().any(|(p, _)| p == "top.pdf"));
        assert!(pdfs.iter().any(|(p, _)| p == "dir\\nested.pdf"));
    }

    #[test]
    fn glob_rg_brace_expansion() {
        let (_d, idx) = idx_with_types(&[
            ("a.pdf", "p.1", "pdf", "x"),
            ("b.htm", "S1", "htm", "x"),
            ("c.md", "S1", "md", "x"),
        ]);
        let docs = glob_docs(&idx, "*.{pdf,htm}").unwrap();
        assert_eq!(docs.len(), 2);
        assert!(docs.iter().any(|(p, _)| p == "a.pdf"));
        assert!(docs.iter().any(|(p, _)| p == "b.htm"));
    }

    #[test]
    fn glob_discovery_alias_lists_all() {
        let (_d, idx) = idx_with(&[
            ("one.pdf", "p.1", "a"),
            ("two\\three.pdf", "p.1", "b"),
        ]);
        assert_eq!(glob_docs(&idx, "").unwrap().len(), 2);
        assert_eq!(glob_docs(&idx, "**/*").unwrap().len(), 2);
    }

    #[test]
    fn normalize_pattern_aliases() {
        assert_eq!(normalize_pattern(""), "**");
        assert_eq!(normalize_pattern("  **/*  "), "**");
    }
}
