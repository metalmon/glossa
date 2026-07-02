//! Convert reference validation tables (xlsx) to the eval's JSON format.
//!
//! MDM exports carry a UID header row (GUIDs / numeric ids) above the
//! human-readable one, plus UID-valued reference columns. Those are export
//! artifacts of the source system — they are CUT here so the JSON is clean:
//! every column is keyed by its human-readable name, every value is a datum.
//!
//! Files whose stem starts with `_` are metadata by convention and are still
//! converted, but the eval loader ignores them.

use calamine::{open_workbook, Reader, Xlsx};
use serde_json::{Map, Value};
use std::path::PathBuf;

/// A GUID (8-4-4-4-12 hex form).
fn is_guid(s: &str) -> bool {
    let t = s.trim();
    t.len() == 36 && t.chars().all(|c| c.is_ascii_hexdigit() || c == '-') && t.matches('-').count() == 4
}

/// A prod-MDM identifier in a HEADER position: a GUID or a bare numeric id.
/// (Numeric ids only count for headers — numeric DATA is legitimate values.)
fn is_mdm_id(s: &str) -> bool {
    let t = s.trim();
    !t.is_empty() && (t.chars().all(|c| c.is_ascii_digit()) || is_guid(t))
}

fn cell_text(c: &calamine::Data) -> String {
    match c {
        calamine::Data::Empty => String::new(),
        calamine::Data::String(s) => s.to_string(),
        calamine::Data::Float(f) => {
            if *f == f.floor() && f.is_finite() {
                format!("{}", *f as i64)
            } else {
                format!("{f}")
            }
        }
        calamine::Data::Int(i) => format!("{i}"),
        calamine::Data::Bool(b) => format!("{b}"),
        calamine::Data::DateTime(d) => d.to_string(),
        calamine::Data::Error(e) => format!("{e:?}"),
        _ => format!("{c:?}"),
    }
}

fn typed_value(s: &str) -> Value {
    if s.is_empty() {
        Value::Null
    } else if let Ok(n) = s.parse::<f64>() {
        if n == n.floor() && n.is_finite() {
            Value::Number(serde_json::Number::from(n as i64))
        } else {
            Value::Number(serde_json::Number::from_f64(n).unwrap())
        }
    } else {
        Value::String(s.to_string())
    }
}

fn main() -> anyhow::Result<()> {
    let in_dir = std::env::args().nth(1).unwrap_or_else(|| "kb-val-gost".into());
    let out_dir = std::env::args().nth(2).unwrap_or_else(|| "kb-val-gost".into());

    let out = PathBuf::from(&out_dir);
    std::fs::create_dir_all(&out)?;

    let entries = std::fs::read_dir(&in_dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(true, |e| e != "xlsx") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let out_path = out.join(format!("{stem}.json"));

        let mut wb: Xlsx<_> = open_workbook(&path)?;
        let sheet_names = wb.sheet_names().to_vec();

        let mut tables = Vec::new();
        for sname in &sheet_names {
            let Ok(ws) = wb.worksheet_range(sname) else { continue };
            let raw: Vec<Vec<String>> = ws
                .rows()
                .map(|r| r.iter().map(cell_text).collect())
                .collect();
            if raw.is_empty() {
                continue;
            }

            // MDM exports put a UID row above the human-readable header row.
            // If row 0 contains ids, the real headers are on row 1.
            let uid_header_row = raw[0].iter().any(|s| is_mdm_id(s));
            let (headers, data_start) = if uid_header_row && raw.len() >= 2 {
                (raw[1].clone(), 2)
            } else {
                (raw[0].clone(), 1)
            };
            let data_rows = &raw[data_start.min(raw.len())..];

            // Keep a column when it has a human-readable name, is not itself a
            // UID column, and its data is not just UID references. On duplicate
            // names keep the first survivor.
            let mut seen_names: std::collections::BTreeSet<String> = Default::default();
            let keep: Vec<usize> = headers
                .iter()
                .enumerate()
                .filter(|(ci, h)| {
                    let h = h.trim();
                    if h.is_empty() || is_mdm_id(h) {
                        return false;
                    }
                    let mut data = data_rows
                        .iter()
                        .filter_map(|r| r.get(*ci))
                        .filter(|v| !v.trim().is_empty())
                        .peekable();
                    if data.peek().is_some() && data.all(|v| is_guid(v)) {
                        return false; // UID-reference column (GUID values only —
                                      // bare numbers are real data, e.g. type "41")
                    }
                    seen_names.insert(h.to_string())
                })
                .map(|(ci, _)| ci)
                .collect();

            let rows: Vec<Map<String, Value>> = data_rows
                .iter()
                .map(|r| {
                    keep.iter()
                        .map(|&ci| {
                            (
                                headers[ci].trim().to_string(),
                                r.get(ci).map(|s| typed_value(s)).unwrap_or(Value::Null),
                            )
                        })
                        .collect()
                })
                .filter(|m: &Map<String, Value>| m.values().any(|v| !v.is_null()))
                .collect();

            tables.push(serde_json::json!({
                "sheet": sname,
                "rows": rows
            }));
        }

        let output = serde_json::json!({ "file": stem, "tables": tables });
        let bytes = serde_json::to_vec_pretty(&output)?;
        std::fs::write(&out_path, bytes)?;
        eprintln!(
            "{} → {} ({} sheets, {} rows total)",
            path.display(),
            out_path.display(),
            tables.len(),
            tables.iter().map(|t| t["rows"].as_array().map_or(0, |a| a.len())).sum::<usize>()
        );
    }
    Ok(())
}
