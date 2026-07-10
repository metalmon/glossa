//! Agent notebook: per-document notes under `.glossa/notes/<document>/`.

mod lock;
mod paths;

pub use lock::{with_notebook_write_lock, LOCK_BUSY_MSG};
pub use paths::{
    list_note_paths, mirror_dir_for_doc, normalize_note_file, notes_root, resolve_note_by_document,
    resolve_note_by_path, NoteEntry, NotePath,
};

use std::path::Path;

use crate::index::store::DocIndex;
use crate::tables::csp::{
    dedupe_csp_rows, format_csp, join_csp_fields, merge_append_rows, parse_csp, parse_csp_row,
    CspTable, CSP_DELIMITER,
};

fn format_csp_columns(headers: &[String]) -> String {
    join_csp_fields(headers)
}

fn csp_dup_ignored_suffix(dup_ignored: usize) -> String {
    if dup_ignored == 0 {
        String::new()
    } else {
        format!(
            " ({dup_ignored} duplicate row{} ignored)",
            if dup_ignored == 1 { "" } else { "s" }
        )
    }
}

fn csp_dup_removed_suffix(dup_removed: usize) -> String {
    if dup_removed == 0 {
        String::new()
    } else {
        format!(
            " ({dup_removed} duplicate row{} removed)",
            if dup_removed == 1 { "" } else { "s" }
        )
    }
}

fn header_looks_like_commentary(h: &str) -> bool {
    h.chars().count() > 35
        || h.contains(" - ")
        || h.contains(" — ")
        || h.contains("из пун")
        || h.contains("из табл")
        || h.contains('(')
}

fn cell_looks_like_prose(cell: &str) -> bool {
    let n = cell.chars().count();
    if n > 45 {
        return true;
    }
    cell.contains('(')
        || cell.contains('—')
        || cell.contains(" - ")
        || cell.contains(" для ")
        || cell.contains(" из ")
        || cell.contains(" и мельче")
        || cell.contains(" и крупнее")
        || cell.contains('±')
        || cell.contains("±")
}

/// Soft observations after a successful `.csp` write (at most two lines).
pub fn csp_tutor_hints(table: &CspTable) -> Vec<String> {
    let mut hints = Vec::new();

    if table
        .headers
        .iter()
        .any(|h| header_looks_like_commentary(h))
    {
        hints.push(
            "column headers look long — source tables usually name the parameter alone \
             (e.g. «Наружный диаметр», «Связка»)"
                .into(),
        );
    }

    let mut total = 0usize;
    let mut prose = 0usize;
    let mut max_len = 0usize;
    for row in &table.rows {
        for cell in row {
            if cell.is_empty() {
                continue;
            }
            total += 1;
            max_len = max_len.max(cell.chars().count());
            if cell_looks_like_prose(cell) {
                prose += 1;
            }
        }
    }

    if total > 0 {
        let prose_ratio = prose as f64 / total as f64;
        if prose_ratio >= 0.25 || (prose >= 2 && max_len > 25) {
            hints.push(
                "many cells read like sentences; a copied source grid usually has one code \
                 or number per cell"
                    .into(),
            );
        }
    }

    hints.truncate(2);
    hints
}

fn normalize_csp_content(content: &str) -> anyhow::Result<(String, usize)> {
    let mut table = parse_csp(content)?;
    for (i, h) in table.headers.iter().enumerate() {
        if h.is_empty() {
            anyhow::bail!("empty header cell in column {}", i + 1);
        }
    }
    let dup_removed = dedupe_csp_rows(&mut table);
    Ok((format_csp(&table), dup_removed))
}

/// Create, fully replace, or (with `append`) extend a note bound to an indexed document.
///
/// A `.csp` file (a limit table: tab-separated rows, first line = headers) is
/// validated on write and the reply echoes exactly what the compiler will see
/// — parsed column headers and data-row count — so a malformed table is
/// rejected at write time instead of failing silently at compile time.
/// Replacing an existing file says so explicitly (the previous content is gone).
pub fn note(
    root: &Path,
    idx: &DocIndex,
    doc: &str,
    file: &str,
    content: &str,
    append: bool,
) -> String {
    match with_notebook_write_lock(root, || write_note(root, idx, doc, file, content, append)) {
        Ok(o) => {
            let mut msg = match (&o.mode, &o.table) {
                (WriteMode::Appended { added, dup_ignored }, Some(t)) => format!(
                    "Appended {added} row{} to {} — columns: [{}]; now {} data row{}{}",
                    if *added == 1 { "" } else { "s" },
                    o.rel_path,
                    format_csp_columns(&t.headers),
                    t.rows,
                    if t.rows == 1 { "" } else { "s" },
                    csp_dup_ignored_suffix(*dup_ignored)
                ),
                (WriteMode::Appended { .. }, None) => {
                    format!("Appended — {} now {} lines", o.rel_path, o.lines)
                }
                (_, Some(t)) => {
                    let dup_removed = match &o.mode {
                        WriteMode::Created { dup_removed }
                        | WriteMode::Replaced { dup_removed, .. } => *dup_removed,
                        _ => 0,
                    };
                    format!(
                        "Noted {} — columns: [{}]; {} data row{}{}",
                        o.rel_path,
                        format_csp_columns(&t.headers),
                        t.rows,
                        if t.rows == 1 { "" } else { "s" },
                        csp_dup_removed_suffix(dup_removed)
                    )
                }
                (_, None) => format!("Noted {} — {} lines", o.rel_path, o.lines),
            };
            if let WriteMode::Replaced { prev, .. } = &o.mode {
                msg.push_str(&format!(
                    "; replaced previous version (was {prev}) — pass append=true to add instead"
                ));
            }
            for hint in &o.tutor_hints {
                msg.push_str(&format!("\n  · {hint}"));
            }
            msg
        }
        Err(e) => format!("REJECTED — {e}"),
    }
}

/// Read a note by notebook path (from `ls`).
pub fn cat(root: &Path, idx: &DocIndex, path: &str) -> String {
    match read_note(root, idx, path) {
        Ok((rel, body)) => format!("{rel}\n{body}"),
        Err(e) => format!("REJECTED — {e}"),
    }
}

/// List notes, optionally filtered by document.
pub fn ls(root: &Path, idx: &DocIndex, doc: Option<&str>) -> String {
    match list_note_paths(root, idx, doc) {
        Ok(entries) => {
            if entries.is_empty() {
                return "(no notes — use note(doc, file, content) to create one)".into();
            }
            entries
                .into_iter()
                .map(|e| format!("{}  ({})", e.rel_path, e.size))
                .collect::<Vec<_>>()
                .join("\n")
        }
        Err(e) => format!("REJECTED — {e}"),
    }
}

/// Delete a note by notebook path.
pub fn del(root: &Path, idx: &DocIndex, path: &str) -> String {
    match with_notebook_write_lock(root, || delete_note(root, idx, path)) {
        Ok(rel) => format!("Deleted {rel}"),
        Err(e) => format!("REJECTED — {e}"),
    }
}

/// Literal substring replace inside a note (`all` = every occurrence).
pub fn sed(root: &Path, idx: &DocIndex, path: &str, old: &str, new: &str, all: bool) -> String {
    match with_notebook_write_lock(root, || patch_note(root, idx, path, old, new, all)) {
        Ok((rel, n)) => format!(
            "Patched {rel} ({n} replacement{})",
            if n == 1 { "" } else { "s" }
        ),
        Err(e) => format!("REJECTED — {e}"),
    }
}

pub struct TableEcho {
    pub headers: Vec<String>,
    pub rows: usize,
}

pub enum WriteMode {
    Created {
        dup_removed: usize,
    },
    /// Overwrote an existing file; `prev` describes what was lost ("21 data rows" / "5 lines").
    Replaced {
        prev: String,
        dup_removed: usize,
    },
    /// Extended an existing file; `added` counts new unique data rows (`.csp`) or lines.
    Appended {
        added: usize,
        dup_ignored: usize,
    },
}

pub struct WriteOutcome {
    pub rel_path: String,
    pub lines: usize,
    /// Present for `.csp` files: what the compiler will see (headers + row count).
    pub table: Option<TableEcho>,
    /// Optional soft observations on table shape (`.csp` only).
    pub tutor_hints: Vec<String>,
    pub mode: WriteMode,
}

/// Validate `.csp` content as the compiler will read it; rejects before anything is written.
fn validate_csp(content: &str) -> anyhow::Result<TableEcho> {
    let rel = parse_csp(content)?;
    for (i, h) in rel.headers.iter().enumerate() {
        if h.is_empty() {
            anyhow::bail!("empty header cell in column {}", i + 1);
        }
    }
    Ok(TableEcho {
        headers: rel.headers,
        rows: rel.rows.len(),
    })
}

fn write_note(
    root: &Path,
    idx: &DocIndex,
    doc: &str,
    file: &str,
    content: &str,
    append: bool,
) -> anyhow::Result<WriteOutcome> {
    let NotePath { rel_path, abs_path } = resolve_note_by_document(root, idx, doc, file)?;
    let is_csp = rel_path.to_ascii_lowercase().ends_with(".csp");
    let existing = if abs_path.is_file() {
        Some(std::fs::read_to_string(&abs_path)?)
    } else {
        None
    };

    let (full, mode) = match (&existing, append) {
        // append onto an existing file
        (Some(old), true) => {
            if is_csp {
                let old_rel = parse_csp(old)?;
                let new_body = match content.lines().next() {
                    Some(first) if split_cells(first) == old_rel.headers => {
                        content.split_once('\n').map(|(_, rest)| rest).unwrap_or("")
                    }
                    _ => content,
                };
                let (merged, stats) = merge_append_rows(&old_rel, new_body)?;
                (
                    format_csp(&merged),
                    WriteMode::Appended {
                        added: stats.added,
                        dup_ignored: stats.dup_ignored,
                    },
                )
            } else {
                let mut full = old.clone();
                if !full.is_empty() && !full.ends_with('\n') {
                    full.push('\n');
                }
                full.push_str(content);
                let added = content.lines().count();
                (
                    full,
                    WriteMode::Appended {
                        added,
                        dup_ignored: 0,
                    },
                )
            }
        }
        // append requested but the file does not exist yet → plain create
        (None, true) | (None, false) => {
            if is_csp {
                let (full, dup_removed) = normalize_csp_content(content)?;
                (full, WriteMode::Created { dup_removed })
            } else {
                (content.to_string(), WriteMode::Created { dup_removed: 0 })
            }
        }
        // overwrite an existing file: allowed, but the echo must say what was lost
        (Some(old), false) => {
            let prev = if is_csp {
                match parse_csp(old) {
                    Ok(r) => format!(
                        "{} data row{}",
                        r.rows.len(),
                        if r.rows.len() == 1 { "" } else { "s" }
                    ),
                    Err(_) => format!("{} lines", old.lines().count()),
                }
            } else {
                format!("{} lines", old.lines().count())
            };
            if is_csp {
                let (full, dup_removed) = normalize_csp_content(content)?;
                (full, WriteMode::Replaced { prev, dup_removed })
            } else {
                (
                    content.to_string(),
                    WriteMode::Replaced {
                        prev,
                        dup_removed: 0,
                    },
                )
            }
        }
    };

    let (table, tutor_hints) = if is_csp {
        let parsed = parse_csp(&full)?;
        let echo = validate_csp(&full)?;
        let hints = csp_tutor_hints(&parsed);
        (Some(echo), hints)
    } else {
        (None, Vec::new())
    };
    if let Some(parent) = abs_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&abs_path, &full)?;
    let lines = full.lines().count();
    Ok(WriteOutcome {
        rel_path,
        lines,
        table,
        tutor_hints,
        mode,
    })
}

/// Split one CSP header line into cells, mirroring how the table parser reads it.
fn split_cells(line: &str) -> Vec<String> {
    parse_csp_row(line.trim_end_matches('\r'), CSP_DELIMITER)
}

pub fn read_note(root: &Path, idx: &DocIndex, path: &str) -> anyhow::Result<(String, String)> {
    let NotePath { rel_path, abs_path } = resolve_note_by_path(root, idx, path)?;
    if !abs_path.is_file() {
        anyhow::bail!("no such note at {rel_path}; use note(doc, file, content) to create");
    }
    let body = std::fs::read_to_string(&abs_path)?;
    Ok((rel_path, body))
}

fn delete_note(root: &Path, idx: &DocIndex, path: &str) -> anyhow::Result<String> {
    let NotePath { rel_path, abs_path } = resolve_note_by_path(root, idx, path)?;
    if !abs_path.is_file() {
        anyhow::bail!("no such note at {rel_path}");
    }
    std::fs::remove_file(&abs_path)?;
    Ok(rel_path)
}

fn patch_note(
    root: &Path,
    idx: &DocIndex,
    path: &str,
    old: &str,
    new: &str,
    all: bool,
) -> anyhow::Result<(String, usize)> {
    let (rel_path, body) = read_note(root, idx, path)?;
    if old.is_empty() {
        anyhow::bail!("old must not be empty");
    }
    let (patched, n) = if all {
        let n = body.matches(old).count();
        (body.replace(old, new), n)
    } else if let Some(pos) = body.find(old) {
        let mut out = String::with_capacity(body.len() - old.len() + new.len());
        out.push_str(&body[..pos]);
        out.push_str(new);
        out.push_str(&body[pos + old.len()..]);
        (out, 1)
    } else {
        (body, 0)
    };
    if n == 0 {
        anyhow::bail!("pattern not found in {rel_path}; cat(path) the note first");
    }
    // A .csp must stay compiler-readable after the patch, same contract as note().
    if rel_path.to_ascii_lowercase().ends_with(".csp") {
        validate_csp(&patched)?;
    }
    let NotePath { abs_path, .. } = resolve_note_by_path(root, idx, path)?;
    std::fs::write(&abs_path, &patched)?;
    Ok((rel_path, n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Chunk;
    use fs4::fs_std::FileExt;

    const TAB: &str = "\t";

    fn idx_with_doc(dir: &std::path::Path, doc: &str) -> DocIndex {
        let idx = DocIndex::open_or_create(dir).unwrap();
        idx.write_chunks(&[Chunk {
            doc_path: doc.into(),
            location: String::new(),
            file_type: "pdf".into(),
            text: "stub".into(),
        }])
        .unwrap();
        idx
    }

    #[test]
    fn note_then_ls_cat_sed_del() {
        let dir = tempfile::tempdir().unwrap();
        let idx = idx_with_doc(dir.path(), "doc.pdf");
        note(
            dir.path(),
            &idx,
            "doc.pdf",
            "parameters.md",
            "A\nB\n",
            false,
        );
        let listing = ls(dir.path(), &idx, Some("doc.pdf"));
        assert!(listing.contains("doc.pdf/parameters.md"));
        let body = cat(dir.path(), &idx, "doc.pdf/parameters.md");
        assert!(body.contains("A\nB"));
        let msg = sed(dir.path(), &idx, "doc.pdf/parameters.md", "B", "C", false);
        assert!(msg.contains("Patched"));
        assert!(cat(dir.path(), &idx, "doc.pdf/parameters.md").contains("C"));
        del(dir.path(), &idx, "doc.pdf/parameters.md");
        assert!(cat(dir.path(), &idx, "doc.pdf/parameters.md").starts_with("REJECTED"));
    }

    #[test]
    fn note_strips_document_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let idx = idx_with_doc(dir.path(), "doc.pdf");
        let msg = note(dir.path(), &idx, "doc.pdf#3", "parameters.md", "x\n", false);
        assert!(msg.contains("doc.pdf/parameters.md"));
    }

    #[test]
    fn csp_stays_in_mirror_root_and_echoes_columns() {
        let dir = tempfile::tempdir().unwrap();
        let idx = idx_with_doc(dir.path(), "doc.pdf");
        let msg = note(
            dir.path(),
            &idx,
            "doc.pdf",
            "t.csp",
            &format!("h{TAB}v\n1{TAB}2\n3{TAB}4\n"),
            false,
        );
        assert!(
            dir.path().join(".glossa/notes/doc.pdf/t.csp").is_file(),
            "{msg}"
        );
        assert!(msg.contains(&format!("columns: [h{TAB}v]")), "{msg}");
        assert!(msg.contains("2 data rows"), "{msg}");
    }

    #[test]
    fn malformed_csp_rejected_before_write() {
        let dir = tempfile::tempdir().unwrap();
        let idx = idx_with_doc(dir.path(), "doc.pdf");
        let msg = note(
            dir.path(),
            &idx,
            "doc.pdf",
            "t.csp",
            &format!("h{TAB}{TAB}v\n1{TAB}2{TAB}3\n"),
            false,
        );
        assert!(msg.contains("REJECTED"), "{msg}");
        assert!(msg.contains("empty header cell in column 2"), "{msg}");
        assert!(!dir.path().join(".glossa/notes/doc.pdf/t.csp").exists());
    }

    #[test]
    fn plain_csv_is_freeform() {
        let dir = tempfile::tempdir().unwrap();
        let idx = idx_with_doc(dir.path(), "doc.pdf");
        let msg = note(
            dir.path(),
            &idx,
            "doc.pdf",
            "scratch.csv",
            "just;some\nrows\n",
            false,
        );
        assert!(msg.contains("2 lines"), "{msg}");
        assert!(dir
            .path()
            .join(".glossa/notes/doc.pdf/scratch.csv")
            .is_file());
    }

    #[test]
    fn sed_keeps_csp_valid() {
        let dir = tempfile::tempdir().unwrap();
        let idx = idx_with_doc(dir.path(), "doc.pdf");
        note(
            dir.path(),
            &idx,
            "doc.pdf",
            "t.csp",
            &format!("h{TAB}v\n1{TAB}2\n"),
            false,
        );
        let msg = sed(
            dir.path(),
            &idx,
            "doc.pdf/t.csp",
            &format!("1{TAB}2"),
            &format!("1{TAB}2{TAB}3"),
            false,
        );
        assert!(msg.contains("REJECTED"), "{msg}");
        assert!(cat(dir.path(), &idx, "doc.pdf/t.csp").contains(&format!("1{TAB}2")));
    }

    #[test]
    fn distinct_mirrors_for_same_stem_different_ext() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            Chunk {
                doc_path: "doc.pdf".into(),
                location: String::new(),
                file_type: "pdf".into(),
                text: "a".into(),
            },
            Chunk {
                doc_path: "doc.docx".into(),
                location: String::new(),
                file_type: "docx".into(),
                text: "b".into(),
            },
        ])
        .unwrap();
        note(dir.path(), &idx, "doc.pdf", "parameters.md", "pdf\n", false);
        note(
            dir.path(),
            &idx,
            "doc.docx",
            "parameters.md",
            "docx\n",
            false,
        );
        assert!(dir
            .path()
            .join(".glossa/notes/doc.pdf/parameters.md")
            .is_file());
        assert!(dir
            .path()
            .join(".glossa/notes/doc.docx/parameters.md")
            .is_file());
        assert_eq!(
            cat(dir.path(), &idx, "doc.pdf/parameters.md"),
            "doc.pdf/parameters.md\npdf\n"
        );
    }

    #[test]
    fn note_rejected_when_lock_held() {
        let dir = tempfile::tempdir().unwrap();
        let idx = idx_with_doc(dir.path(), "doc.pdf");
        let lock_path = dir.path().join(".glossa").join("notebook.lock");
        std::fs::create_dir_all(lock_path.parent().unwrap()).unwrap();
        let holder = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&lock_path)
            .unwrap();
        assert!(FileExt::try_lock_exclusive(&holder).unwrap());
        let msg = note(dir.path(), &idx, "doc.pdf", "parameters.md", "x\n", false);
        assert!(
            msg.contains("REJECTED") && msg.contains(LOCK_BUSY_MSG),
            "{msg}"
        );
        FileExt::unlock(&holder).unwrap();
    }

    #[test]
    fn csp_append_adds_rows_under_existing_header() {
        let dir = tempfile::tempdir().unwrap();
        let idx = idx_with_doc(dir.path(), "doc.pdf");
        note(
            dir.path(),
            &idx,
            "doc.pdf",
            "t.csp",
            &format!("h{TAB}v\n1{TAB}2\n"),
            false,
        );
        // without the header line: rows land under the existing header
        let msg = note(
            dir.path(),
            &idx,
            "doc.pdf",
            "t.csp",
            &format!("3{TAB}4\n5{TAB}6\n"),
            true,
        );
        assert!(msg.contains("Appended 2 rows"), "{msg}");
        assert!(msg.contains("now 3 data rows"), "{msg}");
        // with a repeated header line: the duplicate header is dropped
        let msg = note(
            dir.path(),
            &idx,
            "doc.pdf",
            "t.csp",
            &format!("h{TAB}v\n7{TAB}8\n"),
            true,
        );
        assert!(msg.contains("Appended 1 row"), "{msg}");
        assert!(msg.contains("now 4 data rows"), "{msg}");
        let body = cat(dir.path(), &idx, "doc.pdf/t.csp");
        assert_eq!(
            body.matches(&format!("h{TAB}v")).count(),
            1,
            "single header: {body}"
        );
        assert!(body.contains(&format!("7{TAB}8")), "{body}");
    }

    #[test]
    fn csp_append_ignores_duplicate_rows() {
        let dir = tempfile::tempdir().unwrap();
        let idx = idx_with_doc(dir.path(), "doc.pdf");
        note(
            dir.path(),
            &idx,
            "doc.pdf",
            "t.csp",
            &format!("h{TAB}v\n1{TAB}2\n3{TAB}4\n"),
            false,
        );
        let msg = note(
            dir.path(),
            &idx,
            "doc.pdf",
            "t.csp",
            &format!("1{TAB}2\n3{TAB}4\n"),
            true,
        );
        assert!(msg.contains("Appended 0 rows"), "{msg}");
        assert!(msg.contains("now 2 data rows"), "{msg}");
        assert!(msg.contains("2 duplicate rows ignored"), "{msg}");
        let body = cat(dir.path(), &idx, "doc.pdf/t.csp");
        assert_eq!(body.matches(&format!("1{TAB}2")).count(), 1, "{body}");
    }

    #[test]
    fn csp_replace_dedupes_internal_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let idx = idx_with_doc(dir.path(), "doc.pdf");
        let msg = note(
            dir.path(),
            &idx,
            "doc.pdf",
            "t.csp",
            &format!("h{TAB}v\n1{TAB}2\n1{TAB}2\n3{TAB}4\n"),
            false,
        );
        assert!(msg.contains("2 data rows"), "{msg}");
        assert!(msg.contains("1 duplicate row removed"), "{msg}");
    }

    #[test]
    fn csp_append_rejects_ragged_rows_without_writing() {
        let dir = tempfile::tempdir().unwrap();
        let idx = idx_with_doc(dir.path(), "doc.pdf");
        note(
            dir.path(),
            &idx,
            "doc.pdf",
            "t.csp",
            &format!("h{TAB}v\n1{TAB}2\n"),
            false,
        );
        let msg = note(
            dir.path(),
            &idx,
            "doc.pdf",
            "t.csp",
            &format!("3{TAB}4{TAB}5\n"),
            true,
        );
        assert!(msg.contains("REJECTED"), "{msg}");
        // the file keeps its pre-append content
        assert!(cat(dir.path(), &idx, "doc.pdf/t.csp").contains(&format!("1{TAB}2")));
        assert!(!cat(dir.path(), &idx, "doc.pdf/t.csp").contains(&format!("3{TAB}4{TAB}5")));
    }

    #[test]
    fn plain_append_concatenates() {
        let dir = tempfile::tempdir().unwrap();
        let idx = idx_with_doc(dir.path(), "doc.pdf");
        note(dir.path(), &idx, "doc.pdf", "parameters.md", "A\nB", false);
        let msg = note(dir.path(), &idx, "doc.pdf", "parameters.md", "C\n", true);
        assert!(msg.contains("Appended"), "{msg}");
        assert!(msg.contains("now 3 lines"), "{msg}");
        let body = cat(dir.path(), &idx, "doc.pdf/parameters.md");
        assert!(body.contains("A\nB\nC"), "{body}");
    }

    #[test]
    fn append_to_missing_file_creates_it() {
        let dir = tempfile::tempdir().unwrap();
        let idx = idx_with_doc(dir.path(), "doc.pdf");
        let msg = note(
            dir.path(),
            &idx,
            "doc.pdf",
            "t.csp",
            &format!("h{TAB}v\n1{TAB}2\n"),
            true,
        );
        assert!(msg.starts_with("Noted"), "{msg}");
        assert!(msg.contains("1 data row"), "{msg}");
    }

    #[test]
    fn overwrite_echo_warns_about_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let idx = idx_with_doc(dir.path(), "doc.pdf");
        note(
            dir.path(),
            &idx,
            "doc.pdf",
            "t.csp",
            &format!("h{TAB}v\n1{TAB}2\n3{TAB}4\n"),
            false,
        );
        let msg = note(
            dir.path(),
            &idx,
            "doc.pdf",
            "t.csp",
            &format!("h{TAB}v\n9{TAB}9\n"),
            false,
        );
        assert!(
            msg.contains("replaced previous version (was 2 data rows)"),
            "{msg}"
        );
        note(dir.path(), &idx, "doc.pdf", "s.md", "a\nb\nc\n", false);
        let msg = note(dir.path(), &idx, "doc.pdf", "s.md", "x\n", false);
        assert!(
            msg.contains("replaced previous version (was 3 lines)"),
            "{msg}"
        );
    }

    #[test]
    fn csp_tutor_hints_clean_grid_is_silent() {
        let table = parse_csp(&format!("Связка{TAB}Зернистость\nBF{TAB}F24\nR{TAB}F30\n")).unwrap();
        assert!(csp_tutor_hints(&table).is_empty());
    }

    #[test]
    fn csp_tutor_hints_prose_cells_and_long_header() {
        let table = parse_csp(
            "Связка (bond type) - из пункта 4.4\n\
             B (бакелитовая)\n\
             BF (с упрочняющими)\n",
        )
        .unwrap();
        let hints = csp_tutor_hints(&table);
        assert!(
            hints.iter().any(|h| h.contains("headers look long")),
            "{hints:?}"
        );
        assert!(hints.iter().any(|h| h.contains("sentences")), "{hints:?}");
    }

    #[test]
    fn note_echoes_csp_tutor_hints() {
        let dir = tempfile::tempdir().unwrap();
        let idx = idx_with_doc(dir.path(), "doc.pdf");
        let msg = note(
            dir.path(),
            &idx,
            "doc.pdf",
            "bond.csp",
            "Связка (bond type) - из пункта 4.4\n\
             B (бакелитовая)\n",
            false,
        );
        assert!(msg.contains("  · "), "{msg}");
        assert!(msg.contains("headers look long"), "{msg}");
    }
}
