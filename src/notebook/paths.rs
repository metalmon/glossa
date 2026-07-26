use std::path::{Path, PathBuf};

use crate::index::store::DocIndex;

pub struct NotePath {
    /// Notebook path returned to agents (`<document>/<rel>`).
    pub rel_path: String,
    pub abs_path: PathBuf,
}

pub struct NoteEntry {
    pub rel_path: String,
    pub size: u64,
}

/// Notebook storage root: `<corpus>/.glossa/notes/` — inside `.glossa` so the corpus
/// indexer's walker (which skips `.glossa`) can never index the agent's own notes.
pub fn notes_root(root: &Path) -> PathBuf {
    root.join(".glossa").join("notes")
}

/// Mirror directory name under the notes root — full indexed document path (with extension).
pub fn mirror_dir_for_doc(doc_path: &str) -> String {
    doc_path.trim().replace('\\', "/")
}

/// Normalize a note `file` within a mirror.
pub fn normalize_note_file(file: &str) -> anyhow::Result<String> {
    let mut file = file.trim().replace('\\', "/");
    while file.starts_with("./") {
        file = file[2..].to_string();
    }
    if file.is_empty() {
        anyhow::bail!("file must not be empty");
    }
    if Path::new(&file).is_absolute() {
        anyhow::bail!("file must be relative");
    }
    for comp in file.split('/') {
        if comp.is_empty() || comp == "." {
            continue;
        }
        if comp == ".." {
            anyhow::bail!("file must not contain ..");
        }
    }
    Ok(file.trim_matches('/').to_string())
}

fn reject_corpus_only_path(rel: &str) -> anyhow::Result<()> {
    if !rel.contains('/') {
        anyhow::bail!(
            "{rel} is an indexed document path — use read(path, n); notebook paths come from ls (document/file)"
        );
    }
    Ok(())
}

fn storage_abs(root: &Path, rel: &str) -> anyhow::Result<PathBuf> {
    let rel = rel.trim().trim_matches('/');
    if rel.is_empty() {
        anyhow::bail!("path must not be empty");
    }
    reject_corpus_only_path(rel)?;
    let storage_root = notes_root(root);
    let mut out = storage_root.clone();
    for comp in rel.split('/') {
        if comp.is_empty() || comp == "." {
            continue;
        }
        if comp == ".." {
            anyhow::bail!("path must not contain ..");
        }
        out.push(comp);
    }
    if !out.starts_with(&storage_root) {
        anyhow::bail!("path must stay under notes storage");
    }
    Ok(out)
}

fn canonical_document(idx: &DocIndex, doc: &str) -> anyhow::Result<String> {
    let stripped = crate::index::store::strip_section_anchor(doc.trim());
    idx.canonical_document_path(&stripped)
        .ok_or_else(|| anyhow::anyhow!("unknown document '{stripped}'; copy path from grep/read"))
}

fn mirror_for_document(idx: &DocIndex, doc: &str) -> anyhow::Result<String> {
    // `canonical_document_path` returns the indexed path with host-OS separators
    // (backslash on Windows). Notebook `rel_path`s are always `/`-joined and
    // `walk_notes` normalizes listed paths to `/`, so the mirror MUST be `/`-based
    // too — otherwise the `ls(doc)` filter (`rel.starts_with("{mirror}/")`) never
    // matches a note whose document lives in a subdirectory (e.g. `work/x.docx`).
    // A root-only corpus has no separator in the mirror, which is why this went
    // unnoticed until documents were nested under folders.
    Ok(canonical_document(idx, doc)?.replace('\\', "/"))
}

/// Split a notebook `rel` (`<document>/<file>`) into (mirror, file) using the index
/// as the source of truth. The document may span several `/`-segments (it lives in a
/// corpus subdirectory) and the note file may itself be nested, so we cannot just cut
/// at the first `/`. Instead we try each `/`-prefix from longest to shortest and take
/// the longest one that is an indexed document — the remainder is the note file.
fn split_notebook_path<'a>(idx: &DocIndex, rel: &'a str) -> anyhow::Result<(&'a str, &'a str)> {
    let mut boundaries: Vec<usize> = rel.match_indices('/').map(|(i, _)| i).collect();
    boundaries.reverse();
    for b in boundaries {
        let candidate = &rel[..b];
        if idx.canonical_document_path(candidate).is_some() {
            return Ok((candidate, &rel[b + 1..]));
        }
    }
    anyhow::bail!(
        "notebook path must be <indexed document>/<file>; no prefix of '{rel}' is an indexed document — copy the path from ls"
    )
}

/// Resolve `note(doc, file)` → storage path with KB validation.
pub fn resolve_note_by_document(
    root: &Path,
    idx: &DocIndex,
    doc: &str,
    file: &str,
) -> anyhow::Result<NotePath> {
    let mirror = mirror_for_document(idx, doc)?;
    let rel_name = normalize_note_file(file)?;
    let rel_path = format!("{mirror}/{rel_name}");
    let abs_path = storage_abs(root, &rel_path)?;
    Ok(NotePath { rel_path, abs_path })
}

/// Resolve notebook `path` from `ls` (validates document prefix is indexed).
pub fn resolve_note_by_path(root: &Path, idx: &DocIndex, path: &str) -> anyhow::Result<NotePath> {
    let mut rel = path.trim().replace('\\', "/");
    rel = crate::index::store::strip_section_anchor(&rel);
    while rel.starts_with("./") {
        rel = rel[2..].to_string();
    }
    if let Some(stripped) = rel.strip_prefix(".glossa/notes/") {
        rel = stripped.to_string();
    }
    rel = rel.trim_matches('/').to_string();
    if rel.is_empty() {
        anyhow::bail!("path must not be empty");
    }
    // Validates that some prefix of `rel` is an indexed document (guards against a
    // hallucinated path) and locates the document/file boundary.
    let _ = split_notebook_path(idx, &rel)?;
    let abs_path = storage_abs(root, &rel)?;
    Ok(NotePath {
        rel_path: rel,
        abs_path,
    })
}

/// List notebook files under the notes root.
pub fn list_note_paths(
    root: &Path,
    idx: &DocIndex,
    doc: Option<&str>,
) -> anyhow::Result<Vec<NoteEntry>> {
    let storage_root = notes_root(root);
    if !storage_root.is_dir() {
        return Ok(Vec::new());
    }
    let filter_mirror = doc.map(|d| mirror_for_document(idx, d)).transpose()?;
    let mut out = Vec::new();
    walk_notes(
        &storage_root,
        &storage_root,
        filter_mirror.as_deref(),
        &mut out,
    )?;
    out.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    Ok(out)
}

fn walk_notes(
    storage_root: &Path,
    dir: &Path,
    filter_mirror: Option<&str>,
    out: &mut Vec<NoteEntry>,
) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_notes(storage_root, &path, filter_mirror, out)?;
            continue;
        }
        let rel = path
            .strip_prefix(storage_root)
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .unwrap_or_default();
        if rel.is_empty() {
            continue;
        }
        if let Some(m) = filter_mirror {
            if !rel.starts_with(&format!("{m}/")) {
                continue;
            }
        }
        let size = entry.metadata()?.len();
        out.push(NoteEntry {
            rel_path: rel,
            size,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirror_keeps_extension() {
        assert_eq!(
            mirror_dir_for_doc("spec_2019-rev3.pdf"),
            "spec_2019-rev3.pdf"
        );
        assert_ne!(
            mirror_dir_for_doc("doc.pdf"),
            mirror_dir_for_doc("doc.docx")
        );
    }

    #[test]
    fn normalize_keeps_file_in_place() {
        assert_eq!(normalize_note_file("limits.csp").unwrap(), "limits.csp");
        assert_eq!(normalize_note_file("./notes.txt").unwrap(), "notes.txt");
        assert!(normalize_note_file("../x.csp").is_err());
    }

    /// Regression: a document nested in a subdirectory (e.g. `work/x.docx`) must
    /// round-trip through note → ls(doc). Before the mirror was `/`-normalized,
    /// `canonical_document_path` returned a backslash mirror on Windows and the
    /// `ls(doc)` filter silently matched nothing (root-only corpora hid the bug).
    #[test]
    fn note_roundtrips_for_subdirectory_document() {
        use crate::model::Chunk;
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[Chunk {
            doc_path: PathBuf::from("work/sub doc.docx"),
            location: "p.1".into(),
            file_type: "docx".into(),
            text: "x".into(),
        }])
        .unwrap();
        let root = dir.path();

        // write a note bound to the nested document
        let np = resolve_note_by_document(root, &idx, "work/sub doc.docx", "n.md").unwrap();
        assert_eq!(np.rel_path, "work/sub doc.docx/n.md", "rel_path must be /-joined");
        std::fs::create_dir_all(np.abs_path.parent().unwrap()).unwrap();
        std::fs::write(&np.abs_path, b"hi").unwrap();

        // ls WITH the doc filter must find it (the regression)
        let filtered = list_note_paths(root, &idx, Some("work/sub doc.docx")).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].rel_path, "work/sub doc.docx/n.md");

        // ls without filter, and the ls-path resolver, agree
        assert_eq!(list_note_paths(root, &idx, None).unwrap().len(), 1);
        let by_path = resolve_note_by_path(root, &idx, "work/sub doc.docx/n.md").unwrap();
        assert_eq!(by_path.abs_path, np.abs_path);
    }
}
