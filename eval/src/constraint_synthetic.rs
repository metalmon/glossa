use crate::export_tz::GrepExample;
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterializeExample {
    pub episode_id: String,
    pub doc: String,
    pub parameter: String,
    pub workbook_excerpt: String,
    pub gold_csp: String,
    pub gold_values: Vec<String>,
    #[serde(default)]
    pub synthetic: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileFixExample {
    pub episode_id: String,
    pub doc: String,
    pub parameter: String,
    pub broken_csp: String,
    pub compiler_error: String,
    pub gold_csp: String,
    pub gold_values: Vec<String>,
    #[serde(default)]
    pub synthetic: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageExample {
    pub episode_id: String,
    pub doc: String,
    pub parameter: String,
    pub graph_stats_report: String,
    pub broken_csp: String,
    pub gold_csp: String,
    pub gold_values: Vec<String>,
    #[serde(default)]
    pub synthetic: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidateExample {
    pub episode_id: String,
    pub doc: String,
    pub parameter: String,
    pub solve_error: String,
    pub broken_csp: String,
    pub gold_csp: String,
    pub gold_values: Vec<String>,
    #[serde(default)]
    pub synthetic: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct ResearchGoldExample {
    episode_id: String,
    question: String,
    gold: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct KbValGostFile {
    tables: Vec<KbValTable>,
}

#[derive(Debug, Deserialize)]
struct KbValTable {
    rows: Vec<serde_json::Map<String, serde_json::Value>>,
}

pub fn load_gold_param(path: &Path) -> anyhow::Result<(String, Vec<String>)> {
    let text = std::fs::read_to_string(path)?;
    let v: KbValGostFile = serde_json::from_str(&text)?;
    let table = v.tables.first().context("no tables")?;
    let first_row = table.rows.first().context("no rows")?;
    let (key, _) = first_row.iter().next().context("empty row")?;
    let param = key.clone();
    let mut values = Vec::new();
    for row in &table.rows {
        if let Some(val) = row.get(&param) {
            values.push(json_scalar_to_string(val));
        }
    }
    Ok((param, values))
}

fn json_scalar_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

pub fn gold_csp_tsv(param: &str, values: &[String]) -> String {
    let mut lines = vec![param.to_string()];
    lines.extend(values.iter().cloned());
    lines.join("\n") + "\n"
}

pub fn oracle_workbook_excerpt(param: &str, values: &[String]) -> String {
    format!(
        "## Параметры\n| Параметр | independent | данные |\n| {param} | да | {} |\n",
        values.join(", ")
    )
}

pub fn materialize_examples_from_dir(
    val_dir: &Path,
    doc: &str,
) -> anyhow::Result<Vec<MaterializeExample>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(val_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with('_'))
        {
            continue;
        }
        let (param, values) = load_gold_param(&path)?;
        let stem = path.file_stem().unwrap().to_string_lossy();
        out.push(MaterializeExample {
            episode_id: format!("synthetic-{doc}-{stem}"),
            doc: doc.into(),
            parameter: param.clone(),
            workbook_excerpt: oracle_workbook_excerpt(&param, &values),
            gold_csp: gold_csp_tsv(&param, &values),
            gold_values: values,
            synthetic: true,
        });
    }
    out.sort_by(|a, b| a.parameter.cmp(&b.parameter));
    Ok(out)
}

fn broken_csp_first_thirty_percent(gold_csp: &str) -> String {
    let mut lines = gold_csp.lines();
    let Some(header) = lines.next() else {
        return String::new();
    };
    let value_rows: Vec<&str> = lines.collect();
    let keep = if value_rows.is_empty() {
        0
    } else {
        ((value_rows.len() as f64) * 0.30).floor().max(1.0) as usize
    };
    let mut out = vec![header.to_string()];
    out.extend(value_rows.into_iter().take(keep).map(str::to_string));
    out.join("\n") + "\n"
}

pub fn compile_fix_examples_from_materialize(
    examples: &[MaterializeExample],
) -> Vec<CompileFixExample> {
    examples
        .iter()
        .map(|ex| CompileFixExample {
            episode_id: ex.episode_id.clone(),
            doc: ex.doc.clone(),
            parameter: ex.parameter.clone(),
            broken_csp: broken_csp_first_thirty_percent(&ex.gold_csp),
            compiler_error: format!(
                "graph_build FAILED: incomplete domain for parameter \"{}\"",
                ex.parameter
            ),
            gold_csp: ex.gold_csp.clone(),
            gold_values: ex.gold_values.clone(),
            synthetic: true,
        })
        .collect()
}

pub fn coverage_examples_from_materialize(examples: &[MaterializeExample]) -> Vec<CoverageExample> {
    examples
        .iter()
        .map(|ex| {
            let graph_stats_report = format!(
                "owned({}): 1 nodes\nprm:{}  [Param]  {} → {}#1\n",
                ex.doc, ex.parameter, ex.parameter, ex.doc,
            );
            CoverageExample {
                episode_id: ex.episode_id.clone(),
                doc: ex.doc.clone(),
                parameter: ex.parameter.clone(),
                graph_stats_report,
                broken_csp: broken_csp_first_thirty_percent(&ex.gold_csp),
                gold_csp: ex.gold_csp.clone(),
                gold_values: ex.gold_values.clone(),
                synthetic: true,
            }
        })
        .collect()
}

pub fn validate_examples_from_materialize(examples: &[MaterializeExample]) -> Vec<ValidateExample> {
    examples
        .iter()
        .map(|ex| ValidateExample {
            episode_id: ex.episode_id.clone(),
            doc: ex.doc.clone(),
            parameter: ex.parameter.clone(),
            solve_error: format!(
                "constraint_solve REJECTED: marking invalid for parameter \"{}\"",
                ex.parameter
            ),
            broken_csp: broken_csp_first_thirty_percent(&ex.gold_csp),
            gold_csp: ex.gold_csp.clone(),
            gold_values: ex.gold_values.clone(),
            synthetic: true,
        })
        .collect()
}

pub fn load_research_grep_examples(path: &Path) -> anyhow::Result<Vec<GrepExample>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read research gold {}", path.display()))?;
    let rows: Vec<ResearchGoldExample> =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    Ok(rows
        .into_iter()
        .map(|row| GrepExample {
            episode_id: row.episode_id,
            case_id: None,
            question: row.question.clone(),
            grep_pattern: row.question,
            gold: row.gold,
            hit: true,
            rank: None,
            synthetic: true,
        })
        .collect())
}

pub fn write_materialize_jsonl(path: &Path, examples: &[MaterializeExample]) -> anyhow::Result<()> {
    let mut f = std::fs::File::create(path)?;
    for ex in examples {
        writeln!(f, "{}", serde_json::to_string(ex)?)?;
    }
    Ok(())
}

pub fn write_compile_fix_jsonl(path: &Path, examples: &[CompileFixExample]) -> anyhow::Result<()> {
    let mut f = std::fs::File::create(path)?;
    for ex in examples {
        writeln!(f, "{}", serde_json::to_string(ex)?)?;
    }
    Ok(())
}

pub fn write_research_jsonl(path: &Path, examples: &[GrepExample]) -> anyhow::Result<()> {
    let mut f = std::fs::File::create(path)?;
    for ex in examples {
        writeln!(f, "{}", serde_json::to_string(ex)?)?;
    }
    Ok(())
}

pub fn write_discover_jsonl(path: &Path, examples: &[GrepExample]) -> anyhow::Result<()> {
    write_research_jsonl(path, examples)
}

pub fn write_compile_jsonl(path: &Path, examples: &[CompileFixExample]) -> anyhow::Result<()> {
    write_compile_fix_jsonl(path, examples)
}

pub fn write_coverage_jsonl(path: &Path, examples: &[CoverageExample]) -> anyhow::Result<()> {
    let mut f = std::fs::File::create(path)?;
    for ex in examples {
        writeln!(f, "{}", serde_json::to_string(ex)?)?;
    }
    Ok(())
}

pub fn write_validate_jsonl(path: &Path, examples: &[ValidateExample]) -> anyhow::Result<()> {
    let mut f = std::fs::File::create(path)?;
    for ex in examples {
        writeln!(f, "{}", serde_json::to_string(ex)?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn materialize_example() -> MaterializeExample {
        MaterializeExample {
            episode_id: "synthetic-doc-param".to_string(),
            doc: "doc.pdf".to_string(),
            parameter: "Марка".to_string(),
            workbook_excerpt: "workbook".to_string(),
            gold_csp: "Марка\nA\nB\nC\nD\n".to_string(),
            gold_values: vec!["A".into(), "B".into(), "C".into(), "D".into()],
            synthetic: true,
        }
    }

    #[test]
    fn load_gold_param_from_fixture() {
        let path = Path::new("kb-val-gost/Тип.json");
        if !path.exists() {
            return; // skip if kb-val-gost not present locally
        }
        let (param, values) = load_gold_param(path).unwrap();
        assert_eq!(param, "Обозначение типа");
        assert!(values.contains(&"41".to_string()) || values.contains(&"42".to_string()));
    }

    #[test]
    fn compile_fix_synthetic_keeps_header_and_first_thirty_percent_rows() {
        let examples = compile_fix_examples_from_materialize(&[materialize_example()]);

        assert_eq!(examples.len(), 1);
        let ex = &examples[0];
        assert_eq!(ex.episode_id, "synthetic-doc-param");
        assert_eq!(ex.doc, "doc.pdf");
        assert_eq!(ex.parameter, "Марка");
        assert_eq!(ex.broken_csp, "Марка\nA\n");
        assert_eq!(
            ex.compiler_error,
            "graph_build FAILED: incomplete domain for parameter \"Марка\""
        );
        assert_eq!(ex.gold_csp, "Марка\nA\nB\nC\nD\n");
        assert_eq!(ex.gold_values, vec!["A", "B", "C", "D"]);
        assert!(ex.synthetic);
    }
}
