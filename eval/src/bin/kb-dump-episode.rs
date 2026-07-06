//! Dump a TensorZero agent episode from ClickHouse as plain text.
//!
//! Standalone utility bin (like `convert-xlsx`): self-contained, no kb_eval imports.
//! MCP `list_inferences` returns one giant JSON blob; this writes a turn-by-turn log.

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::path::PathBuf;

const DEFAULT_CH_URL: &str = "http://localhost:8123/?user=chuser&password=chpassword";

#[derive(Parser)]
#[command(
    name = "kb-dump-episode",
    about = "Dump a TZ agent episode from ClickHouse as a readable transcript"
)]
struct Cli {
    #[arg(long, conflicts_with_all = ["run", "latest"])]
    episode_id: Option<String>,

    #[arg(long, conflicts_with = "episode_id")]
    run: Option<String>,

    #[arg(long, conflicts_with = "episode_id")]
    latest: bool,

    #[arg(long, default_value = "constraint_validate")]
    function: String,

    #[arg(long)]
    clickhouse: Option<String>,

    #[arg(long, short)]
    out: Option<PathBuf>,

    #[arg(long, default_value_t = 8000)]
    truncate: usize,

    #[arg(long)]
    full: bool,
}

#[derive(Debug, Clone)]
struct DumpConfig {
    clickhouse_url: String,
    episode_id: Option<String>,
    run_tag: Option<String>,
    function_name: String,
    latest: bool,
    truncate: usize,
    out: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChInferenceRow {
    id: String,
    episode_id: String,
    function_name: String,
    variant_name: String,
    input: String,
    output: String,
    tags: std::collections::HashMap<String, String>,
}

fn ch_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\'', "''")
}

fn ch_query(ch_url: &str, sql: &str) -> Result<String> {
    let resp = ureq::post(ch_url)
        .timeout(std::time::Duration::from_secs(120))
        .send_string(sql)
        .map_err(|e| anyhow::anyhow!("clickhouse query failed: {e}"))?;
    let mut body = String::new();
    resp.into_reader()
        .read_to_string(&mut body)
        .context("read clickhouse response")?;
    Ok(body)
}

fn ch_query_rows(ch_url: &str, sql: &str) -> Result<Vec<ChInferenceRow>> {
    let body = ch_query(ch_url, sql)?;
    body.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| serde_json::from_str(line).with_context(|| format!("parse row: {line:.120}")))
        .collect()
}

fn resolve_episode_id(cfg: &DumpConfig) -> Result<String> {
    if let Some(id) = cfg.episode_id.as_deref().filter(|s| !s.is_empty()) {
        return Ok(id.to_string());
    }
    let fn_name = ch_escape(&cfg.function_name);
    let sql = if let Some(run) = cfg.run_tag.as_deref().filter(|s| !s.is_empty()) {
        let run = ch_escape(run);
        format!(
            "SELECT episode_id FROM tensorzero.ChatInference \
             WHERE function_name = '{fn_name}' AND tags['run'] = '{run}' \
             GROUP BY episode_id ORDER BY max(timestamp) DESC LIMIT 1 \
             FORMAT TabSeparatedRaw"
        )
    } else if cfg.latest {
        format!(
            "SELECT episode_id FROM tensorzero.ChatInference \
             WHERE function_name = '{fn_name}' \
             ORDER BY timestamp DESC LIMIT 1 \
             FORMAT TabSeparatedRaw"
        )
    } else {
        anyhow::bail!("specify --episode-id, --run, or --latest");
    };
    let body = ch_query(&cfg.clickhouse_url, &sql)?.trim().to_string();
    if body.is_empty() {
        anyhow::bail!("no episode found (function={}, run={:?})", cfg.function_name, cfg.run_tag);
    }
    Ok(body)
}

fn parse_messages(input_json: &str) -> Result<Vec<Value>> {
    let v: Value = serde_json::from_str(input_json).context("parse inference input JSON")?;
    Ok(v
        .get("messages")
        .and_then(|m| m.as_array())
        .cloned()
        .unwrap_or_default())
}

fn parse_output_blocks(output_json: &str) -> Result<Vec<Value>> {
    let v: Value = serde_json::from_str(output_json).context("parse inference output JSON")?;
    match v {
        Value::Array(blocks) => Ok(blocks),
        Value::Object(o) => Ok(o
            .get("content")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default()),
        Value::String(s) if !s.is_empty() => {
            serde_json::from_str(&s).or(Ok(vec![json!({"type":"text","text":s})]))
        }
        _ => Ok(vec![]),
    }
}

fn block_text(block: &Value) -> String {
    block.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string()
}

fn fmt_args(args: &Value) -> String {
    match args {
        Value::Object(_) => serde_json::to_string(args).unwrap_or_else(|_| "{}".into()),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn maybe_truncate(s: &str, limit: usize) -> String {
    if limit == 0 || s.len() <= limit {
        return s.to_string();
    }
    let mut end = limit.min(s.len());
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n… [{}/{} chars truncated — use --full]",
        &s[..end],
        s.len() - end,
        s.len()
    )
}

fn tool_result_name(block: &Value) -> &str {
    block
        .get("name")
        .or_else(|| block.get("tool_name"))
        .and_then(|n| n.as_str())
        .unwrap_or("tool")
}

fn format_content_blocks(blocks: &[Value], truncate: usize, indent: &str) -> String {
    let mut out = String::new();
    for block in blocks {
        let typ = block.get("type").and_then(|t| t.as_str()).unwrap_or("?");
        match typ {
            "text" => {
                let text = block_text(block);
                if !text.trim().is_empty() {
                    out.push_str(indent);
                    out.push_str(text.trim_end());
                    out.push('\n');
                }
            }
            "tool_call" => {
                let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("?");
                let args = block
                    .get("arguments")
                    .filter(|v| !v.is_null())
                    .or_else(|| block.get("raw_arguments"))
                    .unwrap_or(&Value::Null);
                out.push_str(indent);
                out.push_str(&format!("→ {name} {}\n", fmt_args(args)));
            }
            "tool_result" => {
                let name = tool_result_name(block);
                let body = block
                    .get("result")
                    .or_else(|| block.get("content"))
                    .and_then(|r| r.as_str())
                    .unwrap_or("");
                out.push_str(indent);
                out.push_str(&format!("← {name}\n"));
                out.push_str(indent);
                out.push_str(&maybe_truncate(body, truncate));
                if !body.ends_with('\n') {
                    out.push('\n');
                }
            }
            other => {
                out.push_str(indent);
                out.push_str(&format!("[{other}] {block}\n"));
            }
        }
    }
    out
}

fn format_message(msg: &Value, truncate: usize) -> String {
    let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("?");
    let mut out = format!("[{role}]\n");
    match msg.get("content") {
        Some(Value::String(s)) => {
            out.push_str(&maybe_truncate(s, truncate));
            out.push('\n');
        }
        Some(Value::Array(blocks)) => {
            out.push_str(&format_content_blocks(blocks, truncate, "  "));
        }
        Some(other) => {
            out.push_str(&format!("  {other}\n"));
        }
        None => {}
    }
    out
}

fn format_turn(
    turn: usize,
    total: usize,
    row: &ChInferenceRow,
    prev_msg_len: usize,
    messages: &[Value],
    truncate: usize,
) -> Result<String> {
    const SEP: &str = "================================================================================\n";
    let mut out = format!(
        "\n{SEP}TURN {turn}/{total}  inference={}  variant={}\n{SEP}",
        row.id, row.variant_name
    );
    if turn == 1 {
        for msg in messages {
            out.push_str(&format_message(msg, truncate));
        }
    } else {
        for msg in messages.get(prev_msg_len..).unwrap_or(&[]) {
            out.push_str(&format_message(msg, truncate));
        }
    }
    let blocks = parse_output_blocks(&row.output)?;
    if !blocks.is_empty() {
        out.push_str("[assistant]\n");
        out.push_str(&format_content_blocks(&blocks, truncate, "  "));
    }
    Ok(out)
}

fn format_episode(rows: &[ChInferenceRow], truncate: usize) -> Result<String> {
    if rows.is_empty() {
        return Ok("(empty episode)\n".into());
    }
    let head = &rows[0];
    let run = head.tags.get("run").map(|s| s.as_str()).unwrap_or("-");
    let mut out = format!(
        "EPISODE {}\nfunction={}  variant={}  run={}\ninferences={}\n",
        head.episode_id,
        head.function_name,
        head.variant_name,
        run,
        rows.len(),
    );
    let mut prev_len = 0usize;
    for (i, row) in rows.iter().enumerate() {
        let messages = parse_messages(&row.input)?;
        out.push_str(&format_turn(i + 1, rows.len(), row, prev_len, &messages, truncate)?);
        prev_len = messages.len();
    }
    Ok(out)
}

fn run_dump(cfg: DumpConfig) -> Result<String> {
    let episode_id = resolve_episode_id(&cfg)?;
    let eid = ch_escape(&episode_id);
    let sql = format!(
        "SELECT id, episode_id, function_name, variant_name, input, output, tags \
         FROM tensorzero.ChatInference \
         WHERE episode_id = '{eid}' \
         ORDER BY timestamp ASC \
         FORMAT JSONEachRow"
    );
    let rows = ch_query_rows(&cfg.clickhouse_url, &sql)?;
    if rows.is_empty() {
        anyhow::bail!("episode {episode_id} has no inferences in ClickHouse");
    }
    let text = format_episode(&rows, cfg.truncate)?;
    if let Some(path) = &cfg.out {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        let mut f = std::fs::File::create(path)
            .with_context(|| format!("create {}", path.display()))?;
        f.write_all(text.as_bytes())
            .with_context(|| format!("write {}", path.display()))?;
        eprintln!(
            "dump-episode: {} → {} ({} inferences, {} bytes)",
            episode_id,
            path.display(),
            rows.len(),
            text.len(),
        );
    } else {
        eprintln!("dump-episode: {episode_id} ({} inferences)", rows.len());
    }
    Ok(text)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let text = run_dump(DumpConfig {
        clickhouse_url: cli.clickhouse.unwrap_or_else(|| DEFAULT_CH_URL.into()),
        episode_id: cli.episode_id,
        run_tag: cli.run,
        function_name: cli.function,
        latest: cli.latest,
        truncate: if cli.full { 0 } else { cli.truncate },
        out: cli.out.clone(),
    })?;
    if cli.out.is_none() {
        print!("{text}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_limits_tool_result() {
        let body = "x".repeat(100);
        let s = maybe_truncate(&body, 20);
        assert!(s.contains("truncated"));
        assert!(s.len() < 100);
    }
}
