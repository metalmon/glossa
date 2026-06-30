//! Export GEPA supervision datasets from TensorZero `answer_hotpot` episodes in ClickHouse.
//!
//! One eval question = one episode. We parse the cumulative `input.messages` transcript,
//! keep only `search` / `read` tool events, and label query/read hits against gold chunks
//! from the case registry (`source: path#loc`) with optional graph `MENTIONS` fallback.

use anyhow::{Context, Result};
use glossa::graph::ontology::Ontology;
use glossa::graph::store::GraphStore;
use glossa::index::multilang::TermAnalyzer;
use glossa::index::store::DocIndex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

const DEFAULT_CH_URL: &str = "http://localhost:8123/?user=chuser&password=chpassword";

/// One training case from `{_id, question, answer, source?}` JSON arrays.
#[derive(Debug, Clone, Deserialize)]
pub struct TrainCase {
    #[serde(rename = "_id")]
    pub id: String,
    pub question: String,
    pub answer: String,
    #[serde(default)]
    pub source: Option<String>,
}

/// Parsed search hit from a tool_result text block (`[#n] path · label · snippet`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedHit {
    pub ord: u64,
    pub path: String,
    pub label: String,
}

/// One row in `query.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryExample {
    pub episode_id: String,
    pub case_id: Option<String>,
    pub question: String,
    pub search_query: String,
    pub gold: Vec<String>,
    pub hit: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
}

/// One row in `read.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReadExample {
    pub episode_id: String,
    pub case_id: Option<String>,
    pub question: String,
    pub search_query: String,
    pub hits: Vec<Value>,
    pub gold: Vec<String>,
    pub model_read: Option<ReadPick>,
    pub hit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReadPick {
    pub path: String,
    pub n: u64,
}

/// Export statistics printed after a run.
#[derive(Debug, Default)]
pub struct ExportStats {
    pub episodes_total: u64,
    pub episodes_skipped_no_gold: u64,
    pub episodes_skipped_no_question: u64,
    /// Episodes matched via `tags.case_id` (not question text).
    pub episodes_joined_by_id: u64,
    pub query_rows: u64,
    pub read_rows: u64,
    pub query_hits: u64,
    pub read_hits: u64,
}

pub struct ExportConfig {
    pub clickhouse_url: String,
    pub function_name: String,
    pub run_tag: Option<String>,
    pub train_files: Vec<PathBuf>,
    pub work: PathBuf,
    pub out: PathBuf,
    pub top_k: usize,
}

/// Case registry: join episodes by `tags.case_id` or normalized question text.
#[derive(Debug, Default)]
pub struct CaseRegistry {
    pub by_question: HashMap<String, TrainCase>,
    pub by_id: HashMap<String, TrainCase>,
}

impl CaseRegistry {
    /// Resolve a case: `case_id` tag first, then question text.
    pub fn resolve(&self, case_id: Option<&str>, question: &str) -> Option<&TrainCase> {
        if let Some(id) = case_id.filter(|s| !s.is_empty()) {
            if let Some(c) = self.by_id.get(id) {
                return Some(c);
            }
        }
        self.by_question.get(&normalize_question(question))
    }
}

/// Merge multiple train JSON files into id + question lookups.
pub fn load_case_registry(paths: &[PathBuf]) -> Result<CaseRegistry> {
    let mut reg = CaseRegistry::default();
    for path in paths {
        let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let cases: Vec<TrainCase> = serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
        for case in cases {
            reg.by_id.insert(case.id.clone(), case.clone());
            reg.by_question.insert(normalize_question(&case.question), case);
        }
    }
    Ok(reg)
}

/// Normalize question text for registry join (collapse whitespace, trim).
pub fn normalize_question(q: &str) -> String {
    q.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Extract question from the first user message (`Question: …`).
pub fn extract_question(messages: &[Value]) -> Option<String> {
    for msg in messages {
        if msg.get("role").and_then(|r| r.as_str()) != Some("user") {
            continue;
        }
        let text = message_text(msg);
        if !text.contains("Question:") && !text.contains("Question：") {
            continue;
        }
        let q = text
            .split_once("Question:")
            .or_else(|| text.split_once("Question："))
            .map(|(_, rest)| rest.trim())
            .filter(|q| !q.is_empty())?;
        return Some(q.to_string());
    }
    None
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

/// Parse `[#n] path · label · snippet` lines from a search tool_result body.
pub fn parse_search_hits(text: &str) -> Vec<ParsedHit> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if !line.starts_with("[#") {
            continue;
        }
        let Some(rest) = line.strip_prefix("[#") else { continue };
        let Some((ord_s, rest)) = rest.split_once(']') else { continue };
        let Ok(ord) = ord_s.trim().parse::<u64>() else { continue };
        let rest = rest.trim();
        let Some((path, label)) = rest.split_once('·') else { continue };
        out.push(ParsedHit {
            ord,
            path: path.trim().to_string(),
            label: label.trim().to_string(),
        });
    }
    out
}

fn parse_tool_args(args: &Value) -> Value {
    match args {
        Value::Object(_) => args.clone(),
        Value::String(s) => serde_json::from_str(s).unwrap_or(json!({})),
        _ => json!({}),
    }
}

/// Walk cumulative TZ messages and extract search/read events (tool_call + paired tool_result).
pub fn parse_search_read_events(messages: &[Value]) -> Vec<EpisodeEvent> {
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
                    if name != "search" && name != "read" {
                        continue;
                    }
                    let id = block
                        .get("id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("")
                        .to_string();
                    let args = parse_tool_args(block.get("arguments").unwrap_or(&Value::Null));
                    pending.insert(
                        id.clone(),
                        PendingCall {
                            name: name.to_string(),
                            args,
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
                    let id = block.get("id").and_then(|i| i.as_str()).unwrap_or("").to_string();
                    let Some(call) = pending.remove(&id) else { continue };
                    let result = block
                        .get("result")
                        .and_then(|r| r.as_str())
                        .unwrap_or("")
                        .to_string();
                    if call.name == "search" {
                        let query = call.args.get("query").and_then(|q| q.as_str()).unwrap_or("").to_string();
                        events.push(EpisodeEvent::Search {
                            query,
                            result_text: result,
                        });
                    } else if call.name == "read" {
                        let path = call.args.get("path").and_then(|p| p.as_str()).unwrap_or("").to_string();
                        let n = call
                            .args
                            .get("n")
                            .and_then(|n| n.as_u64().or_else(|| n.as_i64().map(|i| i as u64)))
                            .unwrap_or(0);
                        events.push(EpisodeEvent::Read { path, n });
                    }
                }
            }
        }
    }
    events
}

#[derive(Clone)]
struct PendingCall {
    name: String,
    args: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EpisodeEvent {
    Search { query: String, result_text: String },
    Read { path: String, n: u64 },
}

fn parse_gold_source(source: &str) -> Option<(String, String)> {
    let (path, loc) = source.split_once('#')?;
    let path = path.trim_end_matches(':').trim();
    let loc = loc.trim();
    if path.is_empty() || loc.is_empty() {
        return None;
    }
    Some((path.to_string(), loc.to_string()))
}

/// Gold chunk must exist in the index under a canonical path with a non-empty location.
fn validate_gold_pair(idx: &DocIndex, path: &str, loc: &str) -> Option<(String, String)> {
    let loc = loc.trim();
    if loc.is_empty() {
        return None;
    }
    let path = idx.canonical_document_path(path)?;
    idx.read_chunk(&path, loc).ok().flatten()?;
    Some((path, loc.to_string()))
}

fn canonicalize_hit(idx: &DocIndex, hit: ParsedHit) -> Option<ParsedHit> {
    let path = idx.canonical_document_path(&hit.path)?;
    Some(ParsedHit { path, ..hit })
}

/// Resolve gold `(path, location)` pairs for a case.
pub fn gold_for_case(
    case: &TrainCase,
    graph: &GraphStore,
    ont: &Ontology,
    idx: &DocIndex,
) -> Vec<(String, String)> {
    if let Some(source) = &case.source {
        if let Some((p, l)) = parse_gold_source(source) {
            if let Some(g) = validate_gold_pair(idx, &p, &l) {
                return vec![g];
            }
        }
    }
    let by_label = gold_from_graph_by_question(graph, ont, idx, &case.question);
    if !by_label.is_empty() {
        return by_label;
    }
    gold_from_graph_relevance(&case.question, graph, idx, ont)
}

fn gold_from_graph_relevance(
    question: &str,
    graph: &GraphStore,
    idx: &DocIndex,
    ont: &Ontology,
) -> Vec<(String, String)> {
    let structural: HashSet<String> = ont.structural().into_iter().collect();
    let analyzer = TermAnalyzer::new();
    let mut q_terms = HashSet::new();
    {
        let mut s = std::collections::BTreeSet::new();
        analyzer.terms(question, &mut s);
        q_terms.extend(s);
    }
    if q_terms.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(usize, (String, String))> = Vec::new();
    let nodes = graph.all_nodes().unwrap_or_default();
    for node in nodes {
        if structural.contains(&node.node_type) {
            continue;
        }
        for e in graph.outgoing(&node.id).unwrap_or_default() {
            if e.edge_type != glossa::graph::MENTIONS {
                continue;
            }
            let Some((p, l)) = e.to.split_once('#') else { continue };
            let Some((canon, loc)) = validate_gold_pair(idx, p, l) else { continue };
            let Some(text) = idx.read_chunk(&canon, &loc).ok().flatten() else { continue };
            let mut chunk_terms = std::collections::BTreeSet::new();
            analyzer.terms(&text, &mut chunk_terms);
            let shared = q_terms.iter().filter(|t| chunk_terms.contains(*t)).count();
            if shared >= 2 {
                scored.push((shared, (canon, loc)));
            }
        }
    }
    scored.sort_by(|a, b| b.0.cmp(&a.0));
    scored.dedup_by(|a, b| a.1 == b.1);
    scored.into_iter().take(3).map(|(_, g)| g).collect()
}

fn gold_from_graph_by_question(
    graph: &GraphStore,
    ont: &Ontology,
    idx: &DocIndex,
    question: &str,
) -> Vec<(String, String)> {
    let structural: HashSet<String> = ont.structural().into_iter().collect();
    let nq = normalize_question(question);
    let mut gold = Vec::new();
    let nodes = graph.all_nodes().unwrap_or_default();
    for node in nodes {
        if structural.contains(&node.node_type) {
            continue;
        }
        if normalize_question(&node.label) != nq {
            continue;
        }
        for e in graph.outgoing(&node.id).unwrap_or_default() {
            if e.edge_type == glossa::graph::MENTIONS {
                if let Some((p, l)) = e.to.split_once('#') {
                    if let Some(pair) = validate_gold_pair(idx, p, l) {
                        gold.push(pair);
                    }
                }
            }
        }
    }
    gold.sort();
    gold.dedup();
    gold
}

fn gold_strings(chunks: &[(String, String)]) -> Vec<String> {
    chunks.iter().map(|(p, l)| format!("{p}#{l}")).collect()
}

fn hit_matches_gold(idx: &DocIndex, hit: &ParsedHit, gold: &[(String, String)]) -> bool {
    for (gp, gl) in gold {
        if hit.path != *gp {
            continue;
        }
        if let Ok(Some(loc)) = idx.location_for_ord(&hit.path, hit.ord) {
            if loc == *gl {
                return true;
            }
        }
    }
    false
}

fn read_matches_gold(idx: &DocIndex, path: &str, n: u64, gold: &[(String, String)]) -> bool {
    for (gp, gl) in gold {
        if path != *gp {
            continue;
        }
        if let Ok(Some(loc)) = idx.location_for_ord(path, n) {
            if loc == *gl {
                return true;
            }
        }
    }
    false
}

fn rank_of_gold(hits: &[ParsedHit], idx: &DocIndex, gold: &[(String, String)]) -> Option<usize> {
    for (i, h) in hits.iter().enumerate() {
        if hit_matches_gold(idx, h, gold) {
            return Some(i + 1);
        }
    }
    None
}

fn hits_to_json(idx: &DocIndex, hits: &[ParsedHit]) -> Vec<Value> {
    hits.iter()
        .map(|h| {
            let location = idx
                .location_for_ord(&h.path, h.ord)
                .ok()
                .flatten()
                .unwrap_or_default();
            let file_type = if location.starts_with("p.") {
                "pdf"
            } else {
                ""
            };
            json!({
                "ord": h.ord,
                "path": h.path,
                "location": location,
                "file_type": file_type,
                "snippet": "",
            })
        })
        .collect()
}

/// Build query/read examples from one episode transcript.
pub fn examples_from_episode(
    episode_id: &str,
    messages: &[Value],
    case: &TrainCase,
    idx: &DocIndex,
    graph: &GraphStore,
    ont: &Ontology,
    top_k: usize,
) -> (Vec<QueryExample>, Vec<ReadExample>) {
    let question = match extract_question(messages) {
        Some(q) => q,
        None => return (vec![], vec![]),
    };
    let gold_pairs = gold_for_case(case, graph, ont, idx);
    if gold_pairs.is_empty() {
        return (vec![], vec![]);
    }
    let gold = gold_strings(&gold_pairs);
    let case_id = Some(case.id.clone());

    let mut queries = Vec::new();
    let mut reads = Vec::new();
    let mut last_search: Option<(String, String, Vec<ParsedHit>)> = None;

    for ev in parse_search_read_events(messages) {
        match ev {
            EpisodeEvent::Search { query, result_text } => {
                let hits: Vec<ParsedHit> = parse_search_hits(&result_text)
                    .into_iter()
                    .filter_map(|h| canonicalize_hit(idx, h))
                    .collect();
                let top = hits.iter().take(top_k).cloned().collect::<Vec<_>>();
                let rank = rank_of_gold(&top, idx, &gold_pairs);
                let hit = rank.is_some();
                queries.push(QueryExample {
                    episode_id: episode_id.to_string(),
                    case_id: case_id.clone(),
                    question: question.clone(),
                    search_query: query.clone(),
                    gold: gold.clone(),
                    hit,
                    rank,
                });
                last_search = Some((query, result_text, hits));
            }
            EpisodeEvent::Read { path, n } => {
                let Some(ctx) = last_search.as_ref() else {
                    continue;
                };
                let path = idx
                    .canonical_document_path(&path)
                    .unwrap_or(path);
                let read_hit = read_matches_gold(idx, &path, n, &gold_pairs);
                reads.push(ReadExample {
                    episode_id: episode_id.to_string(),
                    case_id: case_id.clone(),
                    question: question.clone(),
                    search_query: ctx.0.clone(),
                    hits: hits_to_json(idx, &ctx.2),
                    gold: gold.clone(),
                    model_read: Some(ReadPick { path, n }),
                    hit: read_hit,
                });
            }
        }
    }
    (queries, reads)
}

#[derive(Deserialize)]
struct ChEpisodeRow {
    episode_id: String,
    input: String,
    case_id: String,
}

fn ch_query_episodes(
    ch_url: &str,
    function_name: &str,
    run_tag: Option<&str>,
) -> Result<Vec<(String, String, String)>> {
    let run_filter = match run_tag {
        Some(r) if !r.is_empty() => format!("AND tags['run'] = '{r}'"),
        _ => String::new(),
    };
    let sql = format!(
        "SELECT episode_id, argMax(input, timestamp) AS input, \
         argMax(tags['case_id'], timestamp) AS case_id \
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
        let row: ChEpisodeRow = serde_json::from_str(line).context("parse clickhouse JSONEachRow")?;
        out.push((row.episode_id, row.input, row.case_id));
    }
    Ok(out)
}

/// Full export: ClickHouse episodes → `query.jsonl` + `read.jsonl`.
pub fn run_export(cfg: ExportConfig) -> Result<ExportStats> {
    let registry = load_case_registry(&cfg.train_files)?;
    let idx = DocIndex::open_or_create(&cfg.work).context("open index")?;
    let graph = GraphStore::open(&cfg.work).context("open graph")?;
    let ont = Ontology::load_or_default(&cfg.work);

    std::fs::create_dir_all(&cfg.out).with_context(|| format!("create {}", cfg.out.display()))?;
    let query_path = cfg.out.join("query.jsonl");
    let read_path = cfg.out.join("read.jsonl");
    let mut query_f = std::fs::File::create(&query_path)?;
    let mut read_f = std::fs::File::create(&read_path)?;

    let episodes = ch_query_episodes(&cfg.clickhouse_url, &cfg.function_name, cfg.run_tag.as_deref())?;
    let mut stats = ExportStats {
        episodes_total: episodes.len() as u64,
        ..Default::default()
    };

    use std::io::Write;
    for (episode_id, input_json, tag_case_id) in episodes {
        let input: Value = serde_json::from_str(&input_json).context("parse episode input")?;
        let messages = input
            .get("messages")
            .and_then(|m| m.as_array())
            .cloned()
            .unwrap_or_default();

        let question = match extract_question(&messages) {
            Some(q) => q,
            None => {
                stats.episodes_skipped_no_question += 1;
                continue;
            }
        };
        let tag_id = tag_case_id.trim();
        let joined_by_id = !tag_id.is_empty() && registry.by_id.contains_key(tag_id);
        let case = match registry.resolve(if tag_id.is_empty() { None } else { Some(tag_id) }, &question) {
            Some(c) => c.clone(),
            None => {
                stats.episodes_skipped_no_gold += 1;
                continue;
            }
        };
        if joined_by_id {
            stats.episodes_joined_by_id += 1;
        }

        let gold_pairs = gold_for_case(&case, &graph, &ont, &idx);
        if gold_pairs.is_empty() {
            stats.episodes_skipped_no_gold += 1;
            continue;
        }

        let (queries, reads) = examples_from_episode(
            &episode_id,
            &messages,
            &case,
            &idx,
            &graph,
            &ont,
            cfg.top_k,
        );

        if queries.is_empty() && reads.is_empty() {
            stats.episodes_skipped_no_gold += 1;
            continue;
        }

        for q in &queries {
            if q.hit {
                stats.query_hits += 1;
            }
            writeln!(query_f, "{}", serde_json::to_string(q)?)?;
            stats.query_rows += 1;
        }
        for r in &reads {
            if r.hit {
                stats.read_hits += 1;
            }
            writeln!(read_f, "{}", serde_json::to_string(r)?)?;
            stats.read_rows += 1;
        }
    }

    query_f.flush().ok();
    read_f.flush().ok();

    println!(
        "export-tz: episodes={} skipped_no_q={} skipped_no_gold={} joined_by_id={} \
         query={} (hit={}) read={} (hit={}) -> {} / {}",
        stats.episodes_total,
        stats.episodes_skipped_no_question,
        stats.episodes_skipped_no_gold,
        stats.episodes_joined_by_id,
        stats.query_rows,
        stats.query_hits,
        stats.read_rows,
        stats.read_hits,
        query_path.display(),
        read_path.display(),
    );
    if stats.episodes_total > 0 && stats.episodes_joined_by_id == 0 {
        eprintln!(
            "export-tz warn: joined_by_id=0 — episodes lack tags.case_id; \
             re-run eval with current kb-eval (run=…) for reliable gold join"
        );
    }
    if stats.query_rows == 0 && stats.read_rows == 0 {
        anyhow::bail!("export produced no query/read rows — check episodes, registry, and gold");
    }
    Ok(stats)
}

pub fn default_clickhouse_url() -> String {
    std::env::var("CLICKHOUSE_URL").unwrap_or_else(|_| DEFAULT_CH_URL.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_TRANSCRIPT: &str = r#"[
      {"role":"user","content":[{"type":"text","text":"Question: What is the CPU clock speed?"}]},
      {"role":"assistant","content":[{"type":"tool_call","id":"s1","name":"search","arguments":{"query":"CPU clock MHz"}}]},
      {"role":"user","content":[{"type":"tool_result","id":"s1","name":"search","result":"[#3] kb-test/doc.htm · Intro · CPU runs at 1000 MHz\n[#5] kb-test/other.htm · Other · unrelated"}]},
      {"role":"assistant","content":[{"type":"tool_call","id":"r1","name":"read","arguments":{"path":"kb-test/doc.htm","n":3}}]},
      {"role":"user","content":[{"type":"tool_result","id":"r1","name":"read","result":"CPU runs at 1000 MHz"}]}
    ]"#;

    #[test]
    fn extract_question_strips_prefix() {
        let msgs: Vec<Value> = serde_json::from_str(SAMPLE_TRANSCRIPT).unwrap();
        assert_eq!(
            extract_question(&msgs).as_deref(),
            Some("What is the CPU clock speed?")
        );
    }

    #[test]
    fn parse_search_hits_reads_ord_and_path() {
        let hits = parse_search_hits("[#3] kb-test/doc.htm · Intro · snippet\n[#5] other · x · y");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].ord, 3);
        assert_eq!(hits[0].path, "kb-test/doc.htm");
    }

    #[test]
    fn parse_events_search_then_read() {
        let msgs: Vec<Value> = serde_json::from_str(SAMPLE_TRANSCRIPT).unwrap();
        let ev = parse_search_read_events(&msgs);
        assert_eq!(ev.len(), 2);
        assert!(matches!(&ev[0], EpisodeEvent::Search { query, .. } if query == "CPU clock MHz"));
        assert!(matches!(&ev[1], EpisodeEvent::Read { path, n: 3, .. } if path == "kb-test/doc.htm"));
    }

    #[test]
    fn registry_resolves_by_case_id_before_question() {
        let mut reg = CaseRegistry::default();
        let case = TrainCase {
            id: "syn-0001".into(),
            question: "What is the clock speed?".into(),
            answer: "1000 MHz".into(),
            source: Some("kb-test/doc.htm:#2".into()),
        };
        reg.by_id.insert(case.id.clone(), case.clone());
        reg.by_question.insert(normalize_question(&case.question), case);
        assert!(reg.resolve(Some("syn-0001"), "totally different text").is_some());
        assert!(reg.resolve(None, "What is the clock speed?").is_some());
    }

    #[test]
    fn parse_gold_source_strips_trailing_colon() {
        let (p, l) = super::parse_gold_source("kb-test/doc.htm:#2").unwrap();
        assert_eq!(p, "kb-test/doc.htm");
        assert_eq!(l, "2");
    }

    #[test]
    fn validate_gold_pair_skips_empty_location_and_unknown_chunks() {
        use glossa::index::store::DocIndex;
        use glossa::model::Chunk;

        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[Chunk {
            doc_path: "d.pdf".into(),
            location: "p.1".into(),
            file_type: "pdf".into(),
            text: "x".into(),
        }])
        .unwrap();

        assert!(super::validate_gold_pair(&idx, "d.pdf", "").is_none());
        assert!(super::validate_gold_pair(&idx, "d.pdf", "p.99").is_none());
        let g = super::validate_gold_pair(&idx, "d.pdf", "p.1").unwrap();
        assert_eq!(g, ("d.pdf".to_string(), "p.1".to_string()));
    }

    #[test]
    fn export_canonicalizes_absolute_paths_for_hit_scoring() {
        use glossa::graph::ontology::Ontology;
        use glossa::graph::store::GraphStore;
        use glossa::index::store::DocIndex;
        use glossa::model::Chunk;

        let dir = tempfile::tempdir().unwrap();
        let work = dir.path();
        let idx = DocIndex::open_or_create(work).unwrap();
        idx.write_chunks(&[Chunk {
            doc_path: "kb-test/doc.pdf".into(),
            location: "p.3".into(),
            file_type: "pdf".into(),
            text: "CPU runs at 1000 MHz".into(),
        }])
        .unwrap();
        let graph = GraphStore::open(work).unwrap();
        let ont = Ontology::load_or_default(work);

        let abs_path = work.join("kb-test").join("doc.pdf");
        let abs_path_s = abs_path.to_string_lossy();
        let search_result = format!("[#3] {abs_path_s} · pdf · CPU runs at 1000 MHz");
        let msgs = vec![
            json!({"role":"user","content":[{"type":"text","text":"Question: What is the CPU clock speed?"}]}),
            json!({"role":"assistant","content":[{"type":"tool_call","id":"s1","name":"search","arguments":{"query":"CPU clock MHz"}}]}),
            json!({"role":"user","content":[{"type":"tool_result","id":"s1","name":"search","result":search_result}]}),
            json!({"role":"assistant","content":[{"type":"tool_call","id":"r1","name":"read","arguments":{"path":abs_path_s.as_ref(),"n":3}}]}),
            json!({"role":"user","content":[{"type":"tool_result","id":"r1","name":"read","result":"CPU runs at 1000 MHz"}]}),
        ];
        let case = TrainCase {
            id: "t1".into(),
            question: "What is the CPU clock speed?".into(),
            answer: "1000 MHz".into(),
            source: Some("kb-test/doc.pdf#p.3".into()),
        };

        let (queries, reads) = examples_from_episode("ep1", &msgs, &case, &idx, &graph, &ont, 10);
        assert_eq!(queries.len(), 1);
        assert!(queries[0].hit, "expected query hit after path canonicalization");
        assert_eq!(queries[0].gold, vec!["kb-test/doc.pdf#p.3"]);
        assert_eq!(reads.len(), 1);
        assert!(reads[0].hit);
        assert_eq!(reads[0].model_read.as_ref().unwrap().path, "kb-test/doc.pdf");
        assert_eq!(reads[0].hits[0]["path"], "kb-test/doc.pdf");
        assert_eq!(reads[0].hits[0]["location"], "p.3");
    }
}
