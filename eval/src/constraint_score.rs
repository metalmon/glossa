//! Score agent `.csp` content against reference value domains (pre-graph).

use glossa::tables::csp::parse_csp;
use std::collections::BTreeSet;
use std::path::Path;

pub fn norm_value(s: &str) -> String {
    let stripped = match s.rfind('[') {
        Some(p) if s.trim_end().ends_with(']') => s[..p].trim_end().to_string(),
        _ => s.to_string(),
    };
    let stripped = strip_trailing_unit_if_numeric(stripped.trim_end_matches([';', ',', ' ']));
    glossa_constraint::solver::canon_scalar(&stripped)
}

fn strip_trailing_unit_if_numeric(s: &str) -> String {
    let s = s.trim();
    if let Some((left, right)) = s.rsplit_once(|c: char| c.is_whitespace()) {
        let left = left.trim_end();
        let right = right.trim();
        if is_unit_token(right) && looks_numeric(left) {
            return left.to_string();
        }
    }
    let unit_start = s
        .char_indices()
        .rev()
        .take_while(|(_, c)| c.is_alphabetic() || *c == '/' || *c == '°')
        .map(|(i, _)| i)
        .last();
    if let Some(i) = unit_start {
        if i > 0 {
            let (num, unit) = s.split_at(i);
            if is_unit_token(unit) && looks_numeric(num) {
                return num.to_string();
            }
        }
    }
    s.to_string()
}

fn is_unit_token(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphabetic() || c == '/' || c == '°')
}

fn looks_numeric(s: &str) -> bool {
    !s.is_empty() && s.replace(',', ".").parse::<f64>().is_ok()
}

pub fn domain_covers(agent_dom: &BTreeSet<String>, ref_val: &str) -> bool {
    let rv = norm_value(ref_val);
    agent_dom.contains(&rv)
        || agent_dom
            .iter()
            .any(|a| glossa_constraint::enum_alias_matches(a, &rv))
        || glossa_constraint::values_cover(agent_dom.iter().map(|s| s.as_str()), &rv)
}

/// Union of all non-empty cell values from parsed `.csp` text (all columns).
pub fn csp_text_values(csp: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let Ok(rel) = parse_csp(csp) else {
        return out;
    };
    for row in &rel.rows {
        for cell in row {
            if !cell.is_empty() {
                out.insert(norm_value(cell));
            }
        }
    }
    out
}

/// Fraction of `gold_values` covered by `csp` cell union.
pub fn value_recall(csp: &str, gold_values: &[String]) -> f64 {
    let gold: BTreeSet<String> = gold_values.iter().map(|v| norm_value(v)).collect();
    if gold.is_empty() {
        return 1.0;
    }
    let agent = csp_text_values(csp);
    let hit = gold.iter().filter(|v| domain_covers(&agent, v)).count();
    hit as f64 / gold.len() as f64
}

pub fn param_hit(csp: &str, gold_values: &[String], threshold: f64) -> bool {
    value_recall(csp, gold_values) >= threshold
}

/// Write `csp` into a temp notes mirror and reuse glossa column scan (optional helper).
pub fn write_csp_to_notes(root: &Path, doc: &str, filename: &str, csp: &str) -> anyhow::Result<()> {
    use glossa::notebook::{mirror_dir_for_doc, notes_root};
    let dir = notes_root(root).join(mirror_dir_for_doc(doc));
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join(filename), csp)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_recall_full_and_partial() {
        let csp = "Тип\n41\n42\n";
        let gold = vec!["41".into(), "42".into(), "99".into()];
        let r = value_recall(csp, &gold);
        assert!((r - 2.0 / 3.0).abs() < 1e-9);
        assert!(param_hit(csp, &gold, 0.5));
        assert!(!param_hit(csp, &gold, 0.8));
    }

    #[test]
    fn value_recall_empty_csp() {
        assert_eq!(value_recall("not\ttsv\n", &["1".into()]), 0.0);
    }

    #[test]
    fn value_recall_ignores_unit_suffix_mismatch() {
        // Agent wrote marking-style "63 м/с"; gold has bare "63".
        let csp = "V\n63 м/с\n80 м/с\n";
        let gold = vec!["63".into(), "80".into()];
        assert!((value_recall(csp, &gold) - 1.0).abs() < 1e-9);
        assert_eq!(norm_value("63 м/с"), norm_value("63"));
    }
}
