//! Export constraint GEPA supervision datasets from `constraint_validate` TensorZero episodes.

use crate::constraint_synthetic::{
    coverage_examples_from_materialize, gold_csp_tsv, load_gold_param, CompileFixExample,
    CoverageExample, MaterializeExample, ValidateExample, validate_examples_from_materialize,
};
use crate::export_tz::{GrepExample, ReadExample, ReadPick};
use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

const READ_RESEARCH_JSONL: &str = "research_read.jsonl";

pub struct ExportTzConstraintConfig {
    pub clickhouse_url: String,
    pub work: PathBuf,
    pub out: PathBuf,
    pub function: String,
    pub run: Option<String>,
    pub val_dir: PathBuf,
}

#[derive(Debug, Default)]
pub struct ExportTzConstraintStats {
    pub episodes_total: u64,
    pub materialize_rows: u64,
    pub research_rows: u64,
    pub research_read_rows: u64,
    pub compile_fix_rows: u64,
    pub skipped_parse: u64,
}

#[derive(Debug, Clone)]
pub struct GoldParam {
    pub parameter: String,
    pub gold_csp: String,
    pub gold_values: Vec<String>,
}

#[derive(Debug, Default)]
pub struct ConstraintEpisodeRows {
    pub materialize: Vec<MaterializeExample>,
    pub research: Vec<GrepExample>,
    pub research_reads: Vec<ReadExample>,
    pub compile_fix: Vec<CompileFixExample>,
}

#[derive(Clone)]
struct ToolEvent {
    name: String,
    args: Value,
    result: String,
}

#[derive(Clone)]
struct PendingCall {
    name: String,
    args: Value,
}

#[derive(Deserialize)]
struct ChEpisodeRow {
    episode_id: String,
    input: String,
}

fn parse_tool_args(args: &Value) -> Value {
    match args {
        Value::Object(_) => args.clone(),
        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
        _ => json!({}),
    }
}

fn message_text(msg: &Value) -> String {
    match msg.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        Some(other) => other.as_str().unwrap_or("").to_string(),
        None => String::new(),
    }
}

fn episode_question(messages: &[Value]) -> String {
    for msg in messages {
        if msg.get("role").and_then(|r| r.as_str()) == Some("user") {
            let text = message_text(msg);
            if !text.trim().is_empty() {
                return text.trim().to_string();
            }
        }
    }
    String::new()
}

fn parse_tool_events(messages: &[Value]) -> Vec<ToolEvent> {
    let mut events = Vec::new();
    let mut pending: HashMap<String, PendingCall> = HashMap::new();

    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let content = msg.get("content").and_then(|c| c.as_array());
        if role == "assistant" {
            if let Some(blocks) = content {
                for block in blocks {
                    if block.get("type").and_then(|t| t.as_str()) != Some("tool_call") {
                        continue;
                    }
                    let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("");
                    let id = block
                        .get("id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("")
                        .to_string();
                    pending.insert(
                        id,
                        PendingCall {
                            name: name.to_string(),
                            args: parse_tool_args(block.get("arguments").unwrap_or(&Value::Null)),
                        },
                    );
                }
            }
        } else if role == "user" {
            if let Some(blocks) = content {
                for block in blocks {
                    if block.get("type").and_then(|t| t.as_str()) != Some("tool_result") {
                        continue;
                    }
                    let id = block
                        .get("id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("")
                        .to_string();
                    let Some(call) = pending.remove(&id) else {
                        continue;
                    };
                    let result = block
                        .get("result")
                        .and_then(|r| r.as_str())
                        .unwrap_or("")
                        .to_string();
                    events.push(ToolEvent {
                        name: call.name,
                        args: call.args,
                        result,
                    });
                }
            }
        }
    }
    events
}

fn normalize_key(s: &str) -> String {
    s.replace('\\', "/")
        .rsplit('/')
        .next()
        .unwrap_or(s)
        .trim_end_matches(".csp")
        .trim_end_matches(".json")
        .replace('_', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn csp_filename(args: &Value) -> Option<String> {
    let file = args
        .get("file")
        .or_else(|| args.get("path"))
        .and_then(|p| p.as_str())?;
    if file.replace('\\', "/").to_lowercase().ends_with(".csp") {
        Some(file.to_string())
    } else {
        None
    }
}

fn file_stem_name(path: &str) -> String {
    let name = path.replace('\\', "/");
    let name = name.rsplit('/').next().unwrap_or(&name);
    name.trim_end_matches(".csp").to_string()
}

fn csp_header(content: &str) -> Option<String> {
    content
        .lines()
        .next()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
}

fn lookup_gold<'a>(
    args: &Value,
    file: &str,
    content: &str,
    gold: &'a HashMap<String, GoldParam>,
) -> Option<&'a GoldParam> {
    let mut candidates = Vec::new();
    if let Some(p) = args.get("parameter").and_then(|p| p.as_str()) {
        candidates.push(p.to_string());
    }
    if let Some(header) = csp_header(content) {
        candidates.push(header);
    }
    candidates.push(file_stem_name(file));

    candidates.iter().find_map(|candidate| {
        gold.get(candidate)
            .or_else(|| gold.get(&normalize_key(candidate)))
    })
}

fn gold_hit(text: &str, values: &[String]) -> bool {
    values.iter().any(|v| !v.is_empty() && text.contains(v))
}

fn read_path_with_ord(args: &Value) -> String {
    let path = args.get("path").and_then(|p| p.as_str()).unwrap_or("");
    let n = args
        .get("n")
        .and_then(|n| n.as_u64().or_else(|| n.as_i64().map(|i| i as u64)))
        .unwrap_or(0);
    if n > 0 {
        format!("{path}#{n}")
    } else {
        path.to_string()
    }
}

fn sed_path(args: &Value) -> Option<String> {
    csp_filename(args).or_else(|| {
        args.get("path")
            .and_then(|p| p.as_str())
            .filter(|p| p.replace('\\', "/").to_lowercase().ends_with(".csp"))
            .map(ToString::to_string)
    })
}

fn apply_sed(content: &str, args: &Value) -> String {
    let old = args.get("old").and_then(|v| v.as_str()).unwrap_or("");
    let new = args.get("new").and_then(|v| v.as_str()).unwrap_or("");
    if old.is_empty() {
        return content.to_string();
    }
    if args.get("all").and_then(|v| v.as_bool()).unwrap_or(false) {
        content.replace(old, new)
    } else {
        content.replacen(old, new, 1)
    }
}

pub fn examples_from_messages(
    episode_id: &str,
    messages: &[Value],
    gold: &HashMap<String, GoldParam>,
) -> ConstraintEpisodeRows {
    let question = episode_question(messages);
    let mut rows = ConstraintEpisodeRows::default();
    let mut workbook_excerpt = String::new();
    let mut current_csp: HashMap<String, String> = HashMap::new();
    let mut pending_compile_error: Option<String> = None;
    let mut materialize_started = false;

    for ev in parse_tool_events(messages) {
        match ev.name.as_str() {
            "grep" if !materialize_started => {
                let pattern = ev
                    .args
                    .get("pattern")
                    .and_then(|p| p.as_str())
                    .unwrap_or("")
                    .to_string();
                let values: Vec<String> = gold
                    .values()
                    .flat_map(|g| g.gold_values.iter().cloned())
                    .collect();
                rows.research.push(GrepExample {
                    episode_id: episode_id.to_string(),
                    case_id: None,
                    question: question.clone(),
                    grep_pattern: pattern,
                    gold: values.clone(),
                    hit: gold_hit(&ev.result, &values),
                    rank: None,
                    synthetic: false,
                });
            }
            "read" if !materialize_started => {
                let path = ev
                    .args
                    .get("path")
                    .and_then(|p| p.as_str())
                    .unwrap_or("")
                    .to_string();
                let n = ev
                    .args
                    .get("n")
                    .and_then(|n| n.as_u64().or_else(|| n.as_i64().map(|i| i as u64)))
                    .unwrap_or(0);
                let values: Vec<String> = gold
                    .values()
                    .flat_map(|g| g.gold_values.iter().cloned())
                    .collect();
                rows.research_reads.push(ReadExample {
                    episode_id: episode_id.to_string(),
                    case_id: None,
                    question: question.clone(),
                    search_query: read_path_with_ord(&ev.args),
                    hits: Vec::new(),
                    gold: values.clone(),
                    model_read: Some(ReadPick { path, n }),
                    hit: gold_hit(&ev.result, &values),
                    prefill_source: "read".to_string(),
                    grep_result: ev.result,
                });
            }
            "note" => {
                let file = ev.args.get("file").and_then(|f| f.as_str()).unwrap_or("");
                let content = ev
                    .args
                    .get("content")
                    .and_then(|c| c.as_str())
                    .unwrap_or("")
                    .to_string();
                if file.eq_ignore_ascii_case("workbook.md") {
                    workbook_excerpt = content;
                    continue;
                }
                let Some(csp_file) = csp_filename(&ev.args) else {
                    continue;
                };
                materialize_started = true;
                let prior = current_csp.get(&csp_file).cloned().unwrap_or_default();
                if let (Some(err), Some(gold_param)) = (
                    pending_compile_error.take(),
                    lookup_gold(&ev.args, &csp_file, &content, gold),
                ) {
                    rows.compile_fix.push(CompileFixExample {
                        episode_id: episode_id.to_string(),
                        doc: ev
                            .args
                            .get("doc")
                            .and_then(|d| d.as_str())
                            .unwrap_or("")
                            .to_string(),
                        parameter: gold_param.parameter.clone(),
                        broken_csp: if prior.is_empty() {
                            content.clone()
                        } else {
                            prior
                        },
                        compiler_error: err,
                        gold_csp: gold_param.gold_csp.clone(),
                        gold_values: gold_param.gold_values.clone(),
                        synthetic: false,
                    });
                } else if let Some(gold_param) = lookup_gold(&ev.args, &csp_file, &content, gold) {
                    rows.materialize.push(MaterializeExample {
                        episode_id: episode_id.to_string(),
                        doc: ev
                            .args
                            .get("doc")
                            .and_then(|d| d.as_str())
                            .unwrap_or("")
                            .to_string(),
                        parameter: gold_param.parameter.clone(),
                        workbook_excerpt: workbook_excerpt.clone(),
                        gold_csp: gold_param.gold_csp.clone(),
                        gold_values: gold_param.gold_values.clone(),
                        synthetic: false,
                    });
                }
                current_csp.insert(csp_file, content);
            }
            "sed" => {
                let Some(path) = sed_path(&ev.args) else {
                    continue;
                };
                let prior = current_csp.get(&path).cloned().unwrap_or_default();
                if let Some(err) = pending_compile_error.take() {
                    let lookup_content = if prior.is_empty() {
                        ev.args
                            .get("old")
                            .and_then(|old| old.as_str())
                            .unwrap_or("")
                            .to_string()
                    } else {
                        prior.clone()
                    };
                    if let Some(gold_param) = lookup_gold(&ev.args, &path, &lookup_content, gold) {
                        rows.compile_fix.push(CompileFixExample {
                            episode_id: episode_id.to_string(),
                            doc: String::new(),
                            parameter: gold_param.parameter.clone(),
                            broken_csp: lookup_content,
                            compiler_error: err,
                            gold_csp: gold_param.gold_csp.clone(),
                            gold_values: gold_param.gold_values.clone(),
                            synthetic: false,
                        });
                    }
                }
                current_csp.insert(path, apply_sed(&prior, &ev.args));
            }
            "graph_build" if ev.result.contains("FAILED") => {
                pending_compile_error = Some(ev.result);
            }
            _ => {}
        }
    }

    rows
}

fn load_gold_params(val_dir: &Path) -> Result<HashMap<String, GoldParam>> {
    let mut out = HashMap::new();
    for entry in
        std::fs::read_dir(val_dir).with_context(|| format!("read {}", val_dir.display()))?
    {
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
        let (parameter, values) =
            load_gold_param(&path).with_context(|| format!("load {}", path.display()))?;
        let gold = GoldParam {
            parameter: parameter.clone(),
            gold_csp: gold_csp_tsv(&parameter, &values),
            gold_values: values,
        };
        out.insert(normalize_key(&parameter), gold.clone());
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            out.insert(normalize_key(stem), gold);
        }
    }
    Ok(out)
}

fn sql_quote(s: &str) -> String {
    s.replace('\'', "''")
}

fn ch_query_episodes(
    ch_url: &str,
    function_name: &str,
    run_tag: Option<&str>,
) -> Result<Vec<(String, String)>> {
    let function_name = sql_quote(function_name);
    let run_filter = match run_tag {
        Some(r) if !r.is_empty() => format!("AND tags['run'] = '{}'", sql_quote(r)),
        _ => String::new(),
    };
    let sql = format!(
        "SELECT episode_id, argMax(input, timestamp) AS input \
         FROM tensorzero.ChatInference \
         WHERE function_name = '{function_name}' {run_filter} \
         GROUP BY episode_id \
         FORMAT JSONEachRow"
    );
    let resp = ureq::post(ch_url)
        .timeout(std::time::Duration::from_secs(120))
        .send_string(&sql)
        .map_err(|e| anyhow::anyhow!("clickhouse query failed: {e}"))?;
    let body = resp.into_string().context("read clickhouse response")?;
    let mut out = Vec::new();
    for line in body.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row: ChEpisodeRow =
            serde_json::from_str(line).context("parse clickhouse JSONEachRow")?;
        out.push((row.episode_id, row.input));
    }
    Ok(out)
}

fn write_jsonl<T: serde::Serialize>(path: &Path, rows: &[T]) -> Result<()> {
    let part = path.with_extension("jsonl.part");
    let mut f =
        std::fs::File::create(&part).with_context(|| format!("create {}", part.display()))?;
    for row in rows {
        writeln!(f, "{}", serde_json::to_string(row)?)?;
    }
    f.flush().ok();
    drop(f);
    std::fs::rename(&part, path)
        .with_context(|| format!("rename {} -> {}", part.display(), path.display()))?;
    Ok(())
}

pub fn run(cfg: ExportTzConstraintConfig) -> Result<ExportTzConstraintStats> {
    let _ = &cfg.work;
    let gold = load_gold_params(&cfg.val_dir)?;
    std::fs::create_dir_all(&cfg.out).with_context(|| format!("create {}", cfg.out.display()))?;
    let episodes = ch_query_episodes(&cfg.clickhouse_url, &cfg.function, cfg.run.as_deref())?;
    let mut stats = ExportTzConstraintStats {
        episodes_total: episodes.len() as u64,
        ..Default::default()
    };

    let mut materialize = Vec::new();
    let mut research = Vec::new();
    let mut research_reads = Vec::new();
    let mut compile_fix = Vec::new();

    for (episode_id, input_json) in episodes {
        let input: Value = match serde_json::from_str(&input_json) {
            Ok(input) => input,
            Err(_) => {
                stats.skipped_parse += 1;
                continue;
            }
        };
        let messages = input
            .get("messages")
            .and_then(|m| m.as_array())
            .cloned()
            .unwrap_or_default();
        let rows = examples_from_messages(&episode_id, &messages, &gold);
        materialize.extend(rows.materialize);
        research.extend(rows.research);
        research_reads.extend(rows.research_reads);
        compile_fix.extend(rows.compile_fix);
    }

    stats.materialize_rows = materialize.len() as u64;
    stats.research_rows = research.len() as u64;
    stats.research_read_rows = research_reads.len() as u64;
    stats.compile_fix_rows = compile_fix.len() as u64;

    write_jsonl(&cfg.out.join("materialize.jsonl"), &materialize)?;
    write_jsonl(&cfg.out.join("discover.jsonl"), &research)?;
    write_jsonl(&cfg.out.join("research.jsonl"), &research)?;
    write_jsonl(&cfg.out.join(READ_RESEARCH_JSONL), &research_reads)?;
    write_jsonl(&cfg.out.join("compile.jsonl"), &compile_fix)?;
    write_jsonl(&cfg.out.join("compile_fix.jsonl"), &compile_fix)?;
    let coverage: Vec<CoverageExample> = coverage_examples_from_materialize(&materialize);
    let validate: Vec<ValidateExample> = validate_examples_from_materialize(&materialize);
    write_jsonl(&cfg.out.join("coverage.jsonl"), &coverage)?;
    write_jsonl(&cfg.out.join("validate.jsonl"), &validate)?;

    println!(
        "export-tz-constraint: episodes={} skipped_parse={} materialize={} discover={} research_read={} compile={} coverage={} validate={} -> {}",
        stats.episodes_total,
        stats.skipped_parse,
        stats.materialize_rows,
        stats.research_rows,
        stats.research_read_rows,
        stats.compile_fix_rows,
        coverage.len(),
        validate.len(),
        cfg.out.display()
    );
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::collections::HashMap;

    #[test]
    fn extracts_constraint_examples_from_transcript() {
        let messages: Vec<Value> = serde_json::from_str(
            r#"[
              {"role":"assistant","content":[{"type":"tool_call","id":"g1","name":"grep","arguments":{"pattern":"Марка шлифматериала","context":20}}]},
              {"role":"user","content":[{"type":"tool_result","id":"g1","name":"grep","result":"doc.pdf:#7: 14А, 15А"}]},
              {"role":"assistant","content":[{"type":"tool_call","id":"n1","name":"note","arguments":{"doc":"gost_r_57978-2017.pdf","file":"workbook.md","content":"Марка шлифматериала: 14А, 15А"}}]},
              {"role":"user","content":[{"type":"tool_result","id":"n1","name":"note","result":"wrote workbook.md"}]},
              {"role":"assistant","content":[{"type":"tool_call","id":"r1","name":"read","arguments":{"path":"doc.pdf","n":7}}]},
              {"role":"user","content":[{"type":"tool_result","id":"r1","name":"read","result":"Марка шлифматериала"}]},
              {"role":"assistant","content":[{"type":"tool_call","id":"n2","name":"note","arguments":{"doc":"gost_r_57978-2017.pdf","file":"Марка_шлифматериала.csp","content":"Марка шлифматериала\n14А\n15А\n"}}]},
              {"role":"user","content":[{"type":"tool_result","id":"n2","name":"note","result":"parsed 1 column, 2 rows"}]},
              {"role":"assistant","content":[{"type":"tool_call","id":"b1","name":"graph_build","arguments":{"doc":"gost_r_57978-2017.pdf"}}]},
              {"role":"user","content":[{"type":"tool_result","id":"b1","name":"graph_build","result":"graph_build FAILED: Марка_шлифматериала.csp line 3"}]},
              {"role":"assistant","content":[{"type":"tool_call","id":"n3","name":"note","arguments":{"doc":"gost_r_57978-2017.pdf","file":"Марка_шлифматериала.csp","content":"Марка шлифматериала\n14А\n15А\n"}}]},
              {"role":"user","content":[{"type":"tool_result","id":"n3","name":"note","result":"parsed 1 column, 2 rows"}]}
            ]"#,
        )
        .unwrap();
        let mut gold = HashMap::new();
        gold.insert(
            "Марка шлифматериала".to_string(),
            GoldParam {
                parameter: "Марка шлифматериала".to_string(),
                gold_csp: "Марка шлифматериала\n14А\n15А\n".to_string(),
                gold_values: vec!["14А".into(), "15А".into()],
            },
        );

        let rows = examples_from_messages("ep1", &messages, &gold);

        assert_eq!(rows.materialize.len(), 1);
        assert_eq!(rows.materialize[0].doc, "gost_r_57978-2017.pdf");
        assert_eq!(rows.materialize[0].parameter, "Марка шлифматериала");
        assert_eq!(
            rows.materialize[0].workbook_excerpt,
            "Марка шлифматериала: 14А, 15А"
        );
        assert_eq!(rows.materialize[0].gold_values, vec!["14А", "15А"]);
        assert!(!rows.materialize[0].synthetic);

        assert_eq!(rows.research.len(), 1);
        assert_eq!(rows.research[0].grep_pattern, "Марка шлифматериала");
        assert!(rows.research[0].hit);
        assert_eq!(rows.research_reads.len(), 1);
        assert_eq!(rows.research_reads[0].prefill_source, "read");
        assert_eq!(rows.research_reads[0].search_query, "doc.pdf#7");

        assert_eq!(rows.compile_fix.len(), 1);
        assert_eq!(
            rows.compile_fix[0].broken_csp,
            "Марка шлифматериала\n14А\n15А\n"
        );
        assert_eq!(
            rows.compile_fix[0].compiler_error,
            "graph_build FAILED: Марка_шлифматериала.csp line 3"
        );
        assert_eq!(
            rows.compile_fix[0].gold_csp,
            "Марка шлифматериала\n14А\n15А\n"
        );
    }
}
