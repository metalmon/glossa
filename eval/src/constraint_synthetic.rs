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

pub fn materialize_examples_from_dir(val_dir: &Path, doc: &str) -> anyhow::Result<Vec<MaterializeExample>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(val_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        if path.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with('_')) {
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

pub fn write_materialize_jsonl(path: &Path, examples: &[MaterializeExample]) -> anyhow::Result<()> {
    let mut f = std::fs::File::create(path)?;
    for ex in examples {
        writeln!(f, "{}", serde_json::to_string(ex)?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
