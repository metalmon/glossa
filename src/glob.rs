//! Path-mask navigation over the knowledge base: a shared shell-glob→regex translator (also used
//! by grep's -g filter) and `glob_docs`, which lists the distinct documents whose path matches a
//! mask, with each document's chunk count. File-First: it reads the index's stored paths only.

use crate::index::store::DocIndex;
use std::collections::BTreeMap;

/// Translate a shell glob (`*`, `?`) into an anchored regex over the whole string.
pub fn glob_to_regex(glob: &str) -> Result<regex::Regex, regex::Error> {
    let mut re = String::from("^");
    for ch in glob.chars() {
        match ch {
            '*' => re.push_str(".*"),
            '?' => re.push('.'),
            c => re.push_str(&regex::escape(&c.to_string())),
        }
    }
    re.push('$');
    regex::RegexBuilder::new(&re).case_insensitive(cfg!(target_os = "windows")).build()
}

/// List the DISTINCT document paths whose path matches `pattern`, each with its highest chunk
/// number (≈ page/section count, the `n`-range for `read(path, n)`). Sorted by path.
pub fn glob_docs(idx: &DocIndex, pattern: &str) -> anyhow::Result<Vec<(String, u64)>> {
    let re = glob_to_regex(pattern)?;
    let mut by_path: BTreeMap<String, u64> = BTreeMap::new();
    idx.iter_chunks(|path, ord, _ft, _body| {
        if re.is_match(path) {
            let e = by_path.entry(path.to_string()).or_insert(0);
            if ord > *e { *e = ord; }
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
        let cs: Vec<Chunk> = chunks.iter().map(|(p, loc, t)| Chunk {
            doc_path: PathBuf::from(p), location: (*loc).into(), file_type: "pdf".into(), text: (*t).into(),
        }).collect();
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
        assert_eq!(pdfs, vec![
            ("kb\\Safety Manual.pdf".to_string(), 1),
            ("kb\\Руководство АБАК.pdf".to_string(), 2),
        ]); // distinct paths, max ord as count, sorted by path; .md excluded
        let abak = glob_docs(&idx, "*АБАК*").unwrap();
        assert_eq!(abak, vec![("kb\\Руководство АБАК.pdf".to_string(), 2)]);
        assert!(glob_docs(&idx, "*nomatch*").unwrap().is_empty());
    }
}
