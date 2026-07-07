use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use crate::graph::store::normalize_label;

/// Parsed table: header row + data rows (same width as header).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relation {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

/// Trim; keep display form for aliases (decimal comma preserved).
pub fn normalize_cell(s: &str) -> String {
    s.trim().to_string()
}

/// Parse one line of semicolon-separated CSV (RFC-style quoted fields).
pub fn parse_semicolon_row(line: &str) -> Vec<String> {
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
            ';' if !in_quotes => {
                out.push(normalize_cell(&field));
                field.clear();
            }
            _ => field.push(c),
        }
    }
    out.push(normalize_cell(&field));
    out
}

/// Parse full CSV text (header + rows). Skips empty lines.
pub fn parse_semicolon_csv(text: &str) -> anyhow::Result<Relation> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty()).peekable();
    let header = match lines.next() {
        Some(h) => parse_semicolon_row(h),
        None => anyhow::bail!("empty CSV"),
    };
    if header.is_empty() || header.iter().all(|c| c.is_empty()) {
        anyhow::bail!("CSV header is empty");
    }
    let width = header.len();
    let mut rows = Vec::new();
    for (i, line) in lines.enumerate() {
        let mut row = parse_semicolon_row(line);
        // A trailing `;` (or several) produces empty cells past the header width — an easy
        // slip that carries no data. Forgive it instead of rejecting the whole table.
        while row.len() > width && row.last().is_some_and(|c| c.is_empty()) {
            row.pop();
        }
        if row.len() < width {
            row.resize(width, String::new());
        } else if row.len() > width {
            anyhow::bail!(
                "row {} has {} cells for {} headers (a valid row has exactly {} ';'):\n{}",
                i + 2,
                row.len(),
                width,
                width - 1,
                misalignment_layout(&header, &row)
            );
        }
        rows.push(row);
    }
    Ok(Relation {
        headers: header,
        rows,
    })
}

/// Pair each cell of an over-wide row with its header (`header=cell | … | EXTRA=cell`), so
/// the writer sees WHERE the row slipped instead of having to count `;` by hand.
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

/// Load and union all `*.csp` files in `dir` (identical headers required per file; rows merged).
pub fn load_relation_dir(dir: &Path, delimiter: char) -> anyhow::Result<Relation> {
    if delimiter != ';' {
        anyhow::bail!("only semicolon delimiter is supported today (got {delimiter})");
    }
    let mut files: Vec<_> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("csp")))
        .collect();
    files.sort();
    if files.is_empty() {
        anyhow::bail!("no .csp files in {}", dir.display());
    }
    let mut merged: Option<Relation> = None;
    for path in files {
        let text = fs::read_to_string(&path)?;
        let rel = parse_semicolon_csv(&text)?;
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

fn dedupe_rows(rel: &mut Relation) {
    let mut seen = BTreeSet::new();
    rel.rows.retain(|row| seen.insert(row.clone()));
}

pub fn column_index(headers: &[String], name: &str) -> Option<usize> {
    let n = normalize_label(name);
    headers.iter().position(|h| normalize_label(h) == n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_semicolon_and_quoted_comma() {
        let row = parse_semicolon_row(r#"a;"22,23";c"#);
        assert_eq!(row, vec!["a", "22,23", "c"]);
    }

    #[test]
    fn parse_full_csv() {
        let csv = "Тип;D\n41;50\n42;63\n";
        let r = parse_semicolon_csv(csv).unwrap();
        assert_eq!(r.headers, vec!["Тип", "D"]);
        assert_eq!(r.rows.len(), 2);
        assert_eq!(r.rows[0][1], "50");
    }

    #[test]
    fn trailing_empty_cells_are_forgiven() {
        // "1;2;;" → 4 cells, but the extras are empty → trimmed back to header width
        let r = parse_semicolon_csv("a;b\n1;2;;\n").unwrap();
        assert_eq!(r.rows, vec![vec!["1".to_string(), "2".to_string()]]);
        // an empty cell WITHIN the width is kept as a normal empty value
        let r = parse_semicolon_csv("a;b;c\n1;;3\n").unwrap();
        assert_eq!(
            r.rows,
            vec![vec!["1".to_string(), String::new(), "3".to_string()]]
        );
    }

    #[test]
    fn overwide_row_error_shows_header_cell_layout() {
        let err = parse_semicolon_csv("Тип;D;Скорость\n41;;50;80\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("row 2 has 4 cells for 3 headers"), "{err}");
        assert!(err.contains("exactly 2 ';'"), "{err}");
        assert!(err.contains("Тип=41"), "{err}");
        assert!(err.contains("D=(empty)"), "{err}");
        assert!(err.contains("Скорость=50"), "{err}");
        assert!(err.contains("EXTRA=80"), "{err}");
    }
}
