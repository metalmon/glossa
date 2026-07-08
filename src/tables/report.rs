//! Human-readable compile output for agents (`graph_build` tool / MCP).

use crate::tables::compile::CompileReport;

const FIX_HINT: &str = "fix: cat(path) → sed/note → graph_build again";

/// Format `tables_to_graph` result for the `graph_build` agent tool.
pub fn format_graph_build_output(result: anyhow::Result<CompileReport>) -> String {
    match result {
        Ok(report) => {
            let mut out = String::from("graph_build OK\n");
            for line in &report.lines {
                out.push_str(line);
                out.push('\n');
            }
            out
        }
        Err(e) => format_graph_build_error(&e),
    }
}

fn format_graph_build_error(e: &anyhow::Error) -> String {
    let mut out = String::from("graph_build FAILED\n");
    let msg = format!("{e:#}");
    for line in msg.lines() {
        out.push_str("  ");
        out.push_str(line);
        out.push('\n');
    }
    if let Some(field) = extract_field_name(&msg) {
        out.push_str("  at field: ");
        out.push_str(&field);
        out.push('\n');
    }
    out.push_str("  ");
    out.push_str(FIX_HINT);
    out.push('\n');
    out
}

fn extract_field_name(msg: &str) -> Option<String> {
    for prefix in ["Field \"", "Field «"] {
        if let Some(rest) = msg.split(prefix).nth(1) {
            let end = rest.find('"').or_else(|| rest.find('»'))?;
            let name = rest[..end].trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_banner_includes_report_lines() {
        let out = format_graph_build_output(Ok(CompileReport {
            lines: vec!["Field «X» → enum".into()],
        }));
        assert!(out.starts_with("graph_build OK\n"));
        assert!(out.contains("Field «X»"));
    }

    #[test]
    fn failed_banner_includes_fix_hint_and_field() {
        let err = anyhow::anyhow!(
            "Field \"Связка\" requires capability formula_cross_field (planned v2)"
        );
        let out = format_graph_build_error(&err);
        assert!(out.starts_with("graph_build FAILED\n"));
        assert!(out.contains("at field: Связка"));
        assert!(out.contains(FIX_HINT));
    }
}
