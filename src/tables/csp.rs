use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

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

/// Load and union all `*.csp` files in `dir` (identical headers required per file; rows merged).
pub fn load_csp_dir(dir: &Path, delimiter: char) -> anyhow::Result<CspTable> {
    let mut files: Vec<_> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("csp")))
        .collect();
    files.sort();
    if files.is_empty() {
        anyhow::bail!("no .csp files in {}", dir.display());
    }
    let mut merged: Option<CspTable> = None;
    for path in files {
        let text = fs::read_to_string(&path)?;
        let rel = parse_csp_with_delimiter(&text, delimiter)?;
        merged = Some(match merged {
            None => rel,
            Some(mut m) => {
                if normalize_headers(&m.headers) != normalize_headers(&rel.headers) {
                    anyhow::bail!(
                        "header mismatch in {} (expected {:?}, got {:?})",
                        path.display(),
                        m.headers,
                        rel.headers
                    );
                }
                m.rows.extend(rel.rows);
                m
            }
        });
    }
    let mut rel = merged.unwrap();
    dedupe_rows(&mut rel);
    Ok(rel)
}

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
pub fn merge_append_rows(old: &CspTable, new_body: &str) -> anyhow::Result<(CspTable, CspAppendStats)> {
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
    Ok((
        merged,
        CspAppendStats {
            added,
            dup_ignored,
        },
    ))
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

    #[test]
    fn merge_append_skips_duplicate_rows() {
        let old = parse_csp("h\tv\n1\t2\n3\t4\n").unwrap();
        let (merged, stats) = merge_append_rows(&old, "1\t2\n3\t4\n5\t6\n").unwrap();
        assert_eq!(stats.added, 1);
        assert_eq!(stats.dup_ignored, 2);
        assert_eq!(merged.rows.len(), 3);
    }
}
