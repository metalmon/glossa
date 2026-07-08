use std::collections::BTreeSet;
#[cfg(feature = "constraint")]
use std::fs;
#[cfg(feature = "constraint")]
use std::path::Path;

#[cfg(feature = "constraint")]
use crate::graph::store::normalize_label;

/// Column delimiter for agent `.csp` limit tables (tab — natural for copied grids; `;` stays free in cells).
pub const CSP_DELIMITER: char = '\t';

/// Join row cells for serialization / synthetic headers.
pub fn join_csp_fields(fields: &[String]) -> String {
    fields.join("\t")
}

/// Parsed `.csp` table: header row + data rows (same width as header).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CspTable {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

/// Trim; keep display form for aliases (decimal comma preserved).
pub fn normalize_cell(s: &str) -> String {
    s.trim().to_string()
}

/// Parse one line of delimiter-separated CSP (RFC-style quoted fields).
pub fn parse_csp_row(line: &str, delimiter: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if !in_quotes => in_quotes = true,
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                }
            }
            d if d == delimiter && !in_quotes => {
                out.push(normalize_cell(&field));
                field.clear();
            }
            _ => field.push(c),
        }
    }
    out.push(normalize_cell(&field));
    out
}

/// Parse full `.csp` text (header + rows). Skips empty lines.
pub fn parse_csp(text: &str) -> anyhow::Result<CspTable> {
    parse_csp_with_delimiter(text, CSP_DELIMITER)
}

pub fn parse_csp_with_delimiter(text: &str, delimiter: char) -> anyhow::Result<CspTable> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty()).peekable();
    let header = match lines.next() {
        Some(h) => parse_csp_row(h, delimiter),
        None => anyhow::bail!("empty CSP table"),
    };
    if header.is_empty() || header.iter().all(|c| c.is_empty()) {
        anyhow::bail!("CSP header is empty");
    }
    let width = header.len();
    let mut rows = Vec::new();
    for (i, line) in lines.enumerate() {
        let mut row = parse_csp_row(line, delimiter);
        // A trailing delimiter (or several) produces empty cells past the header width — an easy
        // slip that carries no data. Forgive it instead of rejecting the whole table.
        while row.len() > width && row.last().is_some_and(|c| c.is_empty()) {
            row.pop();
        }
        if row.len() < width {
            row.resize(width, String::new());
        } else if row.len() > width {
            anyhow::bail!(
                "row {} has {} cells for {} headers (a valid row has exactly {} {}):\n{}",
                i + 2,
                row.len(),
                width,
                width - 1,
                delimiter_name(delimiter),
                misalignment_layout(&header, &row)
            );
        }
        rows.push(row);
    }
    Ok(CspTable {
        headers: header,
        rows,
    })
}

/// Pair each cell of an over-wide row with its header (`header=cell | … | EXTRA=cell`), so
/// the writer sees WHERE the row slipped instead of having to count delimiters by hand.
fn misalignment_layout(headers: &[String], row: &[String]) -> String {
    let show = |c: &str| {
        if c.is_empty() {
            "(empty)".to_string()
        } else {
            c.to_string()
        }
    };
    row.iter()
        .enumerate()
        .map(|(i, cell)| match headers.get(i) {
            Some(h) => format!("{h}={}", show(cell)),
            None => format!("EXTRA={}", show(cell)),
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

fn delimiter_name(d: char) -> &'static str {
    if d == '\t' {
        "tab characters"
    } else {
        "column delimiters"
    }
}

/// Load and union all `*.csp` files in `dir`.
///
/// Two merge modes (same as eval table coverage):
/// - **Same headers** (normalized): row shard — append data rows (one relation table split across files).
/// - **Different headers**: column union — widen the schema; each file's rows map sparsely (independent
///   parameters in separate `.csp` files; row counts need not match).
///
/// When `note_dir_label` is set (document owner path), errors cite `label/file.csp` for agents.
#[cfg(feature = "constraint")]
pub fn load_csp_dir(
    dir: &Path,
    delimiter: char,
    note_dir_label: Option<&str>,
) -> anyhow::Result<CspTable> {
    let mut files: Vec<_> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("csp")))
        .collect();
    files.sort();
    if files.is_empty() {
        anyhow::bail!("no .csp files in {}", dir.display());
    }
    let note_path = |path: &Path| -> String {
        let file = path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
        match note_dir_label {
            Some(doc) => format!("{doc}/{file}"),
            None => path.display().to_string(),
        }
    };
    let mut merged: Option<CspTable> = None;
    for path in files {
        let text =
            fs::read_to_string(&path).map_err(|e| anyhow::anyhow!("{}: {e}", note_path(&path)))?;
        let rel = parse_csp_with_delimiter(&text, delimiter)
            .map_err(|e| anyhow::anyhow!("{}: {e}", note_path(&path)))?;
        merged = Some(match merged {
            None => rel,
            Some(m) => merge_csp_tables(m, rel),
        });
    }
    let mut rel = merged.unwrap();
    dedupe_rows(&mut rel);
    Ok(rel)
}

/// Same normalized header row → append rows; otherwise union columns with sparse row mapping.
#[cfg(feature = "constraint")]
fn merge_csp_tables(left: CspTable, right: CspTable) -> CspTable {
    if normalize_headers(&left.headers) == normalize_headers(&right.headers) {
        let mut out = left;
        out.rows.extend(right.rows);
        return out;
    }
    union_columns_sparse(left, right)
}

#[cfg(feature = "constraint")]
fn union_columns_sparse(left: CspTable, right: CspTable) -> CspTable {
    let mut headers = left.headers.clone();
    for h in &right.headers {
        if column_index(&headers, h).is_none() {
            headers.push(h.clone());
        }
    }
    let mut rows: Vec<Vec<String>> = left
        .rows
        .iter()
        .map(|r| remap_row(&left.headers, r, &headers))
        .collect();
    for r in &right.rows {
        rows.push(remap_row(&right.headers, r, &headers));
    }
    CspTable { headers, rows }
}

#[cfg(feature = "constraint")]
fn remap_row(src_headers: &[String], src_row: &[String], dst_headers: &[String]) -> Vec<String> {
    let mut out = vec![String::new(); dst_headers.len()];
    for (i, h) in src_headers.iter().enumerate() {
        let Some(j) = column_index(dst_headers, h) else {
            continue;
        };
        if let Some(cell) = src_row.get(i) {
            out[j] = cell.clone();
        }
    }
    out
}

#[cfg(feature = "constraint")]
fn normalize_headers(h: &[String]) -> Vec<String> {
    h.iter().map(|s| normalize_label(s)).collect()
}

fn dedupe_rows(rel: &mut CspTable) {
    let mut seen = BTreeSet::new();
    rel.rows.retain(|row| seen.insert(row.clone()));
}

/// Remove duplicate data rows; returns how many were dropped (later copies).
pub fn dedupe_csp_rows(table: &mut CspTable) -> usize {
    let before = table.rows.len();
    dedupe_rows(table);
    before - table.rows.len()
}

pub struct CspAppendStats {
    pub added: usize,
    pub dup_ignored: usize,
}

/// Append parsed data rows from `new_body` onto `old`, skipping duplicates.
pub fn merge_append_rows(
    old: &CspTable,
    new_body: &str,
) -> anyhow::Result<(CspTable, CspAppendStats)> {
    let new_rows = if new_body.trim().is_empty() {
        Vec::new()
    } else {
        let synthetic = format!("{}\n{}", join_csp_fields(&old.headers), new_body.trim());
        parse_csp(&synthetic)?.rows
    };
    let mut merged = old.clone();
    let mut known: BTreeSet<Vec<String>> = merged.rows.iter().cloned().collect();
    let mut seen_in_chunk = BTreeSet::new();
    let mut added = 0usize;
    let mut dup_ignored = 0usize;
    for row in new_rows {
        if known.contains(&row) {
            dup_ignored += 1;
        } else if !seen_in_chunk.insert(row.clone()) {
            dup_ignored += 1;
        } else {
            known.insert(row.clone());
            merged.rows.push(row);
            added += 1;
        }
    }
    Ok((merged, CspAppendStats { added, dup_ignored }))
}

/// Serialize a table back to `.csp` text (tab-separated).
pub fn format_csp(table: &CspTable) -> String {
    let mut out = join_csp_fields(&table.headers);
    out.push('\n');
    for row in &table.rows {
        out.push_str(&join_csp_fields(row));
        out.push('\n');
    }
    out
}

#[cfg(feature = "constraint")]
pub fn column_index(headers: &[String], name: &str) -> Option<usize> {
    let n = normalize_label(name);
    headers.iter().position(|h| normalize_label(h) == n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tab_and_quoted_comma() {
        let row = parse_csp_row("a\t\"22,23\"\tc", CSP_DELIMITER);
        assert_eq!(row, vec!["a", "22,23", "c"]);
    }

    #[test]
    fn parse_full_csp() {
        let text = "Тип\tD\n41\t50\n42\t63\n";
        let r = parse_csp(text).unwrap();
        assert_eq!(r.headers, vec!["Тип", "D"]);
        assert_eq!(r.rows.len(), 2);
        assert_eq!(r.rows[0][1], "50");
    }

    #[test]
    fn trailing_empty_cells_are_forgiven() {
        let r = parse_csp("a\tb\n1\t2\t\t\n").unwrap();
        assert_eq!(r.rows, vec![vec!["1".to_string(), "2".to_string()]]);
        let r = parse_csp("a\tb\tc\n1\t\t3\n").unwrap();
        assert_eq!(
            r.rows,
            vec![vec!["1".to_string(), String::new(), "3".to_string()]]
        );
    }

    #[test]
    fn overwide_row_error_shows_header_cell_layout() {
        let err = parse_csp("Тип\tD\tСкорость\n41\t\t50\t80\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("row 2 has 4 cells for 3 headers"), "{err}");
        assert!(err.contains("exactly 2 tab characters"), "{err}");
        assert!(err.contains("Тип=41"), "{err}");
        assert!(err.contains("D=(empty)"), "{err}");
        assert!(err.contains("Скорость=50"), "{err}");
        assert!(err.contains("EXTRA=80"), "{err}");
    }

    #[test]
    fn semicolon_in_cell_value_is_preserved() {
        let r = parse_csp("D\n115; 125; 150\n").unwrap();
        assert_eq!(r.rows[0][0], "115; 125; 150");
    }

    #[cfg(feature = "constraint")]
    mod constraint_merge {
        use super::*;

        #[test]
        fn load_csp_dir_error_uses_notebook_path() {
            let dir = tempfile::tempdir().unwrap();
            let doc = "gost.pdf";
            std::fs::write(dir.path().join("bad.csp"), "a\tb\n1\t2\t3\n").unwrap();
            let err = load_csp_dir(dir.path(), CSP_DELIMITER, Some(doc))
                .unwrap_err()
                .to_string();
            assert!(err.contains("gost.pdf/bad.csp"), "{err}");
        }

        #[test]
        fn load_csp_dir_same_headers_row_shard() {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join("a.csp"), "D\tT\n50\t2\n").unwrap();
            std::fs::write(dir.path().join("b.csp"), "D\tT\n63\t3\n").unwrap();
            let t = load_csp_dir(dir.path(), CSP_DELIMITER, None).unwrap();
            assert_eq!(t.headers, vec!["D", "T"]);
            assert_eq!(t.rows.len(), 2);
            assert_eq!(t.rows[0], vec!["50", "2"]);
            assert_eq!(t.rows[1], vec!["63", "3"]);
        }

        #[test]
        fn load_csp_dir_different_headers_column_union_sparse() {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join("D.csp"), "D\n50\n63\n").unwrap();
            std::fs::write(dir.path().join("F.csp"), "F\nF16\nF24\n").unwrap();
            let t = load_csp_dir(dir.path(), CSP_DELIMITER, None).unwrap();
            assert_eq!(t.headers, vec!["D", "F"]);
            assert_eq!(t.rows.len(), 4);
            assert_eq!(t.rows[0], vec!["50", ""]);
            assert_eq!(t.rows[1], vec!["63", ""]);
            assert_eq!(t.rows[2], vec!["", "F16"]);
            assert_eq!(t.rows[3], vec!["", "F24"]);
        }

        #[test]
        fn load_csp_dir_shared_column_unions_values() {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join("a.csp"), "Width\n10\n20\n").unwrap();
            std::fs::write(dir.path().join("b.csp"), "Width\n30\n").unwrap();
            let t = load_csp_dir(dir.path(), CSP_DELIMITER, None).unwrap();
            assert_eq!(t.headers, vec!["Width"]);
            assert_eq!(t.rows.len(), 3);
        }

        #[test]
        fn merge_csp_tables_partial_header_overlap() {
            let left = parse_csp("D\tT\n50\t2\n").unwrap();
            let right = parse_csp("D\tH\n63\t10\n").unwrap();
            let t = merge_csp_tables(left, right);
            assert_eq!(t.headers, vec!["D", "T", "H"]);
            assert_eq!(t.rows[0], vec!["50", "2", ""]);
            assert_eq!(t.rows[1], vec!["63", "", "10"]);
        }
    }
}
