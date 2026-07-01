//! Export GEPA supervision datasets from TensorZero `answer_hotpot` episodes in ClickHouse.
//!
//! One eval question = one episode. We parse the cumulative `input.messages` transcript,
//! keep `search` / `grep` / `glob` / `read` tool events, and label hits against gold chunks
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
    pub snippet: String,
}

const SNIPPET_HYDRATE_MAX: usize = 200;

/// One row in `search.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchExample {
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
    /// `search` (default) or `grep` — which tool prefilled the read context.
    #[serde(default = "default_prefill_source")]
    pub prefill_source: String,
    /// Raw grep output when `prefill_source == "grep"`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub grep_result: String,
}

fn default_prefill_source() -> String {
    "search".into()
}

/// One row in `grep.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GrepExample {
    pub episode_id: String,
    pub case_id: Option<String>,
    pub question: String,
    pub grep_pattern: String,
    pub gold: Vec<String>,
    pub hit: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
    /// Oracle-derived row from registry gold (not from an eval episode).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub synthetic: bool,
}

/// One row in `glob.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GlobExample {
    pub episode_id: String,
    pub case_id: Option<String>,
    pub question: String,
    pub glob_pattern: String,
    pub gold: Vec<String>,
    pub hit: bool,
    /// Oracle-derived row from registry gold (not from an eval episode).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub synthetic: bool,
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
    pub search_rows: u64,
    pub grep_rows: u64,
    pub glob_rows: u64,
    pub read_rows: u64,
    pub search_hits: u64,
    pub grep_hits: u64,
    pub glob_hits: u64,
    pub read_hits: u64,
    pub read_skipped_no_gold_in_prefill: u64,
    pub read_snippets_hydrated: u64,
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

/// Split `path · label · snippet` on middle dots (snippet may contain `·`).
fn split_search_hit_body(body: &str) -> Option<(String, String, String)> {
    let parts: Vec<&str> = body.split('·').map(str::trim).collect();
    if parts.len() < 2 {
        return None;
    }
    let path = parts[0].to_string();
    let label = parts[1].to_string();
    let snippet = if parts.len() > 2 {
        parts[2..].join(" · ")
    } else {
        String::new()
    };
    Some((path, label, snippet))
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
        let Some((path, label, snippet)) = split_search_hit_body(rest.trim()) else {
            continue;
        };
        out.push(ParsedHit {
            ord,
            path,
            label,
            snippet,
        });
    }
    out
}

/// Parse `path:#ord: line` rows from a grep tool_result body.
pub fn parse_grep_hits(text: &str) -> Vec<ParsedHit> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let Some((path, rest)) = line.split_once(":#") else { continue };
        let Some((ord_s, snippet)) = rest.split_once(": ") else { continue };
        let Ok(ord) = ord_s.trim().parse::<u64>() else { continue };
        out.push(ParsedHit {
            ord,
            path: path.trim().to_string(),
            label: String::new(),
            snippet: snippet.trim().to_string(),
        });
    }
    out
}

/// Parse `path  (N chunks)` lines from a glob tool_result body.
pub fn parse_glob_paths(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('(') {
                return None;
            }
            let path = line.split("  (").next()?.trim();
            if path.is_empty() {
                None
            } else {
                Some(path.to_string())
            }
        })
        .collect()
}

fn normalize_export_path(p: &str) -> String {
    p.replace('\\', "/")
}

fn glob_lists_gold(paths: &[String], idx: &DocIndex, gold: &[(String, String)]) -> bool {
    for (gp, _) in gold {
        for p in paths {
            let canon = idx.canonical_document_path(p).unwrap_or_else(|| p.clone());
            if normalize_export_path(&canon) == normalize_export_path(gp) {
                return true;
            }
        }
    }
    false
}

fn parse_tool_args(args: &Value) -> Value {
    match args {
        Value::Object(_) => args.clone(),
        Value::String(s) => serde_json::from_str(s).unwrap_or(json!({})),
        _ => json!({}),
    }
}

/// Walk cumulative TZ messages and extract search/grep/glob/read events (tool_call + paired tool_result).
pub fn parse_retrieval_events(messages: &[Value]) -> Vec<EpisodeEvent> {
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
                    if !matches!(name, "search" | "grep" | "glob" | "read") {
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
                    match call.name.as_str() {
                        "search" => {
                            let query = call.args.get("query").and_then(|q| q.as_str()).unwrap_or("").to_string();
                            events.push(EpisodeEvent::Search {
                                query,
                                result_text: result,
                            });
                        }
                        "grep" => {
                            let pattern = call.args.get("pattern").and_then(|p| p.as_str()).unwrap_or("").to_string();
                            events.push(EpisodeEvent::Grep {
                                pattern,
                                result_text: result,
                            });
                        }
                        "glob" => {
                            let pattern = call.args.get("pattern").and_then(|p| p.as_str()).unwrap_or("").to_string();
                            events.push(EpisodeEvent::Glob {
                                pattern,
                                result_text: result,
                            });
                        }
                        "read" => {
                            let path = call.args.get("path").and_then(|p| p.as_str()).unwrap_or("").to_string();
                            let n = call
                                .args
                                .get("n")
                                .and_then(|n| n.as_u64().or_else(|| n.as_i64().map(|i| i as u64)))
                                .unwrap_or(0);
                            events.push(EpisodeEvent::Read { path, n });
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    events
}

/// Back-compat alias — same as [`parse_retrieval_events`].
pub fn parse_search_read_events(messages: &[Value]) -> Vec<EpisodeEvent> {
    parse_retrieval_events(messages)
        .into_iter()
        .filter(|e| matches!(e, EpisodeEvent::Search { .. } | EpisodeEvent::Read { .. }))
        .collect()
}

#[derive(Clone)]
struct PendingCall {
    name: String,
    args: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EpisodeEvent {
    Search { query: String, result_text: String },
    Grep { pattern: String, result_text: String },
    Glob { pattern: String, result_text: String },
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

/// Index: normalized question label → gold chunks from graph MENTIONS edges.
fn build_question_gold_index(
    graph: &GraphStore,
    ont: &Ontology,
    idx: &DocIndex,
) -> HashMap<String, Vec<(String, String)>> {
    let structural: HashSet<String> = ont.structural().into_iter().collect();
    let mut by_question: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for node in graph.all_nodes().unwrap_or_default() {
        if structural.contains(&node.node_type) {
            continue;
        }
        let nq = normalize_question(&node.label);
        let mut gold = Vec::new();
        for e in graph.outgoing(&node.id).unwrap_or_default() {
            if e.edge_type == glossa::graph::MENTIONS {
                if let Some((p, l)) = e.to.split_once('#') {
                    if let Some(pair) = validate_gold_pair(idx, p, l) {
                        gold.push(pair);
                    }
                }
            }
        }
        if gold.is_empty() {
            continue;
        }
        gold.sort();
        gold.dedup();
        let entry = by_question.entry(nq).or_default();
        for pair in gold {
            if !entry.contains(&pair) {
                entry.push(pair);
            }
        }
    }
    for pairs in by_question.values_mut() {
        pairs.sort();
        pairs.dedup();
    }
    by_question
}

/// Resolve gold `(path, location)` pairs for a case.
pub fn gold_for_case(
    case: &TrainCase,
    graph: &GraphStore,
    ont: &Ontology,
    idx: &DocIndex,
) -> Vec<(String, String)> {
    gold_for_case_indexed(case, graph, ont, idx, None)
}

fn gold_for_case_indexed(
    case: &TrainCase,
    graph: &GraphStore,
    ont: &Ontology,
    idx: &DocIndex,
    question_gold: Option<&HashMap<String, Vec<(String, String)>>>,
) -> Vec<(String, String)> {
    if let Some(source) = &case.source {
        if let Some((p, l)) = parse_gold_source(source) {
            if let Some(g) = validate_gold_pair(idx, &p, &l) {
                return vec![g];
            }
        }
    }
    let nq = normalize_question(&case.question);
    if let Some(index) = question_gold {
        if let Some(g) = index.get(&nq) {
            if !g.is_empty() {
                return g.clone();
            }
        }
    } else {
        let by_label = gold_from_graph_by_question(graph, ont, idx, &case.question);
        if !by_label.is_empty() {
            return by_label;
        }
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

fn hydrate_snippet(idx: &DocIndex, path: &str, ord: u64, parsed: &str, hydrated: &mut u64) -> String {
    if !parsed.is_empty() {
        return parsed.to_string();
    }
    if let Ok(Some(loc)) = idx.location_for_ord(path, ord) {
        if let Ok(Some(text)) = idx.read_chunk(path, &loc) {
            if !text.is_empty() {
                *hydrated += 1;
                return text.chars().take(SNIPPET_HYDRATE_MAX).collect();
            }
        }
    }
    String::new()
}

fn hits_to_json(idx: &DocIndex, hits: &[ParsedHit], snippets_hydrated: &mut u64) -> Vec<Value> {
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
            let snippet = hydrate_snippet(idx, &h.path, h.ord, &h.snippet, snippets_hydrated);
            json!({
                "ord": h.ord,
                "path": h.path,
                "location": location,
                "file_type": file_type,
                "snippet": snippet,
            })
        })
        .collect()
}

fn prefill_contains_gold(
    idx: &DocIndex,
    prefill: &LastPrefill,
    gold_pairs: &[(String, String)],
) -> bool {
    match prefill {
        LastPrefill::Search { hits, .. } => hits
            .iter()
            .cloned()
            .filter_map(|h| canonicalize_hit(idx, h))
            .any(|h| hit_matches_gold(idx, &h, gold_pairs)),
        LastPrefill::Grep { result_text, .. } => parse_grep_hits(result_text)
            .into_iter()
            .filter_map(|h| canonicalize_hit(idx, h))
            .any(|h| hit_matches_gold(idx, &h, gold_pairs)),
    }
}

enum LastPrefill {
    Search {
        query: String,
        hits: Vec<ParsedHit>,
    },
    Grep {
        pattern: String,
        result_text: String,
    },
}

/// Build search/grep/glob/read examples from one episode transcript.
pub fn examples_from_episode(
    episode_id: &str,
    messages: &[Value],
    case: &TrainCase,
    gold_pairs: &[(String, String)],
    idx: &DocIndex,
    top_k: usize,
    read_skipped_no_gold_in_prefill: &mut u64,
    read_snippets_hydrated: &mut u64,
) -> (Vec<SearchExample>, Vec<GrepExample>, Vec<GlobExample>, Vec<ReadExample>) {
    let question = match extract_question(messages) {
        Some(q) => q,
        None => return (vec![], vec![], vec![], vec![]),
    };
    if gold_pairs.is_empty() {
        return (vec![], vec![], vec![], vec![]);
    }
    let gold = gold_strings(&gold_pairs);
    let case_id = Some(case.id.clone());

    let mut searches = Vec::new();
    let mut greps = Vec::new();
    let mut globs = Vec::new();
    let mut reads = Vec::new();
    let mut last: Option<LastPrefill> = None;

    for ev in parse_retrieval_events(messages) {
        match ev {
            EpisodeEvent::Search { query, result_text } => {
                let hits: Vec<ParsedHit> = parse_search_hits(&result_text)
                    .into_iter()
                    .filter_map(|h| canonicalize_hit(idx, h))
                    .collect();
                let top = hits.iter().take(top_k).cloned().collect::<Vec<_>>();
                let rank = rank_of_gold(&top, idx, &gold_pairs);
                let hit = rank.is_some();
                searches.push(SearchExample {
                    episode_id: episode_id.to_string(),
                    case_id: case_id.clone(),
                    question: question.clone(),
                    search_query: query.clone(),
                    gold: gold.clone(),
                    hit,
                    rank,
                });
                last = Some(LastPrefill::Search {
                    query,
                    hits,
                });
            }
            EpisodeEvent::Grep { pattern, result_text } => {
                let hits: Vec<ParsedHit> = parse_grep_hits(&result_text)
                    .into_iter()
                    .filter_map(|h| canonicalize_hit(idx, h))
                    .collect();
                let top = hits.iter().take(top_k).cloned().collect::<Vec<_>>();
                let rank = rank_of_gold(&top, idx, &gold_pairs);
                let hit = rank.is_some();
                greps.push(GrepExample {
                    episode_id: episode_id.to_string(),
                    case_id: case_id.clone(),
                    question: question.clone(),
                    grep_pattern: pattern.clone(),
                    gold: gold.clone(),
                    hit,
                    rank,
                    synthetic: false,
                });
                last = Some(LastPrefill::Grep {
                    pattern,
                    result_text,
                });
            }
            EpisodeEvent::Glob { pattern, result_text } => {
                let paths: Vec<String> = parse_glob_paths(&result_text)
                    .into_iter()
                    .filter_map(|p| idx.canonical_document_path(&p))
                    .collect();
                let hit = glob_lists_gold(&paths, idx, &gold_pairs);
                globs.push(GlobExample {
                    episode_id: episode_id.to_string(),
                    case_id: case_id.clone(),
                    question: question.clone(),
                    glob_pattern: pattern,
                    gold: gold.clone(),
                    hit,
                    synthetic: false,
                });
            }
            EpisodeEvent::Read { path, n } => {
                let Some(prefill) = last.as_ref() else { continue };
                if !prefill_contains_gold(idx, prefill, &gold_pairs) {
                    *read_skipped_no_gold_in_prefill += 1;
                    continue;
                }
                let path = idx.canonical_document_path(&path).unwrap_or(path);
                let read_hit = read_matches_gold(idx, &path, n, &gold_pairs);
                match prefill {
                    LastPrefill::Search { query, hits } => {
                        reads.push(ReadExample {
                            episode_id: episode_id.to_string(),
                            case_id: case_id.clone(),
                            question: question.clone(),
                            search_query: query.clone(),
                            hits: hits_to_json(idx, hits, read_snippets_hydrated),
                            gold: gold.clone(),
                            model_read: Some(ReadPick { path, n }),
                            hit: read_hit,
                            prefill_source: "search".into(),
                            grep_result: String::new(),
                        });
                    }
                    LastPrefill::Grep { pattern, result_text } => {
                        reads.push(ReadExample {
                            episode_id: episode_id.to_string(),
                            case_id: case_id.clone(),
                            question: question.clone(),
                            search_query: pattern.clone(),
                            hits: Vec::new(),
                            gold: gold.clone(),
                            model_read: Some(ReadPick { path, n }),
                            hit: read_hit,
                            prefill_source: "grep".into(),
                            grep_result: result_text.clone(),
                        });
                    }
                }
            }
        }
    }
    (searches, greps, globs, reads)
}

fn pick_grep_token(text: &str) -> Option<String> {
    for word in text.split_whitespace() {
        let w: String = word
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '.' || *c == '-')
            .collect();
        if w.len() >= 4 && w.chars().any(|c| c.is_ascii_digit()) {
            return Some(w);
        }
    }
    text.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
                .collect::<String>()
        })
        .filter(|w| w.len() >= 5)
        .max_by_key(|w| w.len())
}

fn glob_pattern_for_gold(path: &str) -> String {
    let name = path
        .rsplit('/')
        .next()
        .or_else(|| path.rsplit('\\').next())
        .unwrap_or(path);
    format!("*{name}*")
}

/// Synthetic grep rows when episodes contain none (gold-derived).
pub fn synthetic_grep(
    cases: &[TrainCase],
    case_gold: &HashMap<String, Vec<(String, String)>>,
    idx: &DocIndex,
) -> Vec<GrepExample> {
    let mut greps = Vec::new();
    for case in cases {
        let gold_pairs = match case_gold.get(&case.id) {
            Some(g) if !g.is_empty() => g,
            _ => continue,
        };
        let gold = gold_strings(&gold_pairs);
        let (path, loc) = &gold_pairs[0];
        let Some(text) = idx.read_chunk(path, loc).ok().flatten() else {
            continue;
        };
        let Some(token) = pick_grep_token(&text) else {
            continue;
        };
        let hits = glossa::grep::grep(idx, &token, &glossa::grep::GrepOpts::default()).unwrap_or_default();
        let parsed: Vec<ParsedHit> = hits
            .iter()
            .map(|h| ParsedHit {
                ord: h.ord,
                path: h.path.clone(),
                label: String::new(),
                snippet: h.line.clone(),
            })
            .collect();
        let hit = parsed.iter().any(|h| hit_matches_gold(idx, h, &gold_pairs));
        greps.push(GrepExample {
            episode_id: format!("synthetic-grep-{}", case.id),
            case_id: Some(case.id.clone()),
            question: case.question.clone(),
            grep_pattern: token,
            gold,
            hit,
            rank: None,
            synthetic: true,
        });
    }
    greps
}

/// Synthetic glob rows when episodes contain none (gold-derived).
pub fn synthetic_glob(
    cases: &[TrainCase],
    case_gold: &HashMap<String, Vec<(String, String)>>,
    idx: &DocIndex,
) -> Vec<GlobExample> {
    let mut globs = Vec::new();
    for case in cases {
        let gold_pairs = match case_gold.get(&case.id) {
            Some(g) if !g.is_empty() => g,
            _ => continue,
        };
        let (path, _) = &gold_pairs[0];
        let pattern = glob_pattern_for_gold(path);
        let glob_text = glossa::tools::glob(idx, &pattern, &glossa::trace::TraceLog::disabled());
        let paths: Vec<String> = parse_glob_paths(&glob_text)
            .into_iter()
            .filter_map(|p| idx.canonical_document_path(&p))
            .collect();
        let hit = glob_lists_gold(&paths, idx, &gold_pairs);
        globs.push(GlobExample {
            episode_id: format!("synthetic-glob-{}", case.id),
            case_id: Some(case.id.clone()),
            question: case.question.clone(),
            glob_pattern: pattern,
            gold: gold_strings(&gold_pairs),
            hit,
            synthetic: true,
        });
    }
    globs
}

/// Synthetic grep/glob rows when episodes contain none (gold-derived).
pub fn synthetic_grep_glob(
    cases: &[TrainCase],
    idx: &DocIndex,
    graph: &GraphStore,
    ont: &Ontology,
) -> (Vec<GrepExample>, Vec<GlobExample>) {
    let question_gold = build_question_gold_index(graph, ont, idx);
    let case_gold: HashMap<String, Vec<(String, String)>> = cases
        .iter()
        .map(|c| {
            (
                c.id.clone(),
                gold_for_case_indexed(c, graph, ont, idx, Some(&question_gold)),
            )
        })
        .collect();
    (
        synthetic_grep(cases, &case_gold, idx),
        synthetic_glob(cases, &case_gold, idx),
    )
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

/// Full export: ClickHouse episodes → `search.jsonl` + `read.jsonl`.
pub fn run_export(cfg: ExportConfig) -> Result<ExportStats> {
    use std::io::Write;
    use std::time::Instant;

    let t0 = Instant::now();
    let registry = load_case_registry(&cfg.train_files)?;
    let idx = DocIndex::open_or_create(&cfg.work).context("open index")?;
    let graph = GraphStore::open(&cfg.work).context("open graph")?;
    let ont = Ontology::load_or_default(&cfg.work);
    let t_index = Instant::now();
    let question_gold = build_question_gold_index(&graph, &ont, &idx);
    eprintln!(
        "export-tz: gold index {} questions in {:.1}s",
        question_gold.len(),
        t_index.elapsed().as_secs_f64(),
    );
    let mut case_gold: HashMap<String, Vec<(String, String)>> = HashMap::new();
    let t_cases = Instant::now();
    for case in registry.by_id.values() {
        case_gold.insert(
            case.id.clone(),
            gold_for_case_indexed(case, &graph, &ont, &idx, Some(&question_gold)),
        );
    }
    eprintln!(
        "export-tz: registry gold for {} cases in {:.1}s",
        case_gold.len(),
        t_cases.elapsed().as_secs_f64(),
    );

    std::fs::create_dir_all(&cfg.out).with_context(|| format!("create {}", cfg.out.display()))?;
    let search_path = cfg.out.join("search.jsonl");
    let grep_path = cfg.out.join("grep.jsonl");
    let glob_path = cfg.out.join("glob.jsonl");
    let read_path = cfg.out.join("read.jsonl");
    let search_part = cfg.out.join("search.jsonl.part");
    let grep_part = cfg.out.join("grep.jsonl.part");
    let glob_part = cfg.out.join("glob.jsonl.part");
    let read_part = cfg.out.join("read.jsonl.part");
    let mut search_f = std::fs::File::create(&search_part)?;
    let mut grep_f = std::fs::File::create(&grep_part)?;
    let mut glob_f = std::fs::File::create(&glob_part)?;
    let mut read_f = std::fs::File::create(&read_part)?;

    let episodes = ch_query_episodes(&cfg.clickhouse_url, &cfg.function_name, cfg.run_tag.as_deref())?;
    eprintln!(
        "export-tz: fetched {} episodes from ClickHouse in {:.1}s",
        episodes.len(),
        t0.elapsed().as_secs_f64(),
    );
    let mut stats = ExportStats {
        episodes_total: episodes.len() as u64,
        ..Default::default()
    };

    let loop_start = Instant::now();
    let mut processed = 0u64;
    for (episode_id, input_json, tag_case_id) in episodes {
        processed += 1;
        if processed % 25 == 0 || processed == stats.episodes_total {
            eprintln!(
                "export-tz: episode {processed}/{} ({:.1}s)",
                stats.episodes_total,
                loop_start.elapsed().as_secs_f64(),
            );
        }
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

        let gold_pairs = case_gold.get(&case.id).cloned().unwrap_or_default();
        if gold_pairs.is_empty() {
            stats.episodes_skipped_no_gold += 1;
            continue;
        }

        let (searches, greps, globs, reads) = examples_from_episode(
            &episode_id,
            &messages,
            &case,
            &gold_pairs,
            &idx,
            cfg.top_k,
            &mut stats.read_skipped_no_gold_in_prefill,
            &mut stats.read_snippets_hydrated,
        );

        if searches.is_empty() && greps.is_empty() && globs.is_empty() && reads.is_empty() {
            stats.episodes_skipped_no_gold += 1;
            continue;
        }

        for s in &searches {
            if s.hit {
                stats.search_hits += 1;
            }
            writeln!(search_f, "{}", serde_json::to_string(s)?)?;
            stats.search_rows += 1;
        }
        for g in &greps {
            if g.hit {
                stats.grep_hits += 1;
            }
            writeln!(grep_f, "{}", serde_json::to_string(g)?)?;
            stats.grep_rows += 1;
        }
        for g in &globs {
            if g.hit {
                stats.glob_hits += 1;
            }
            writeln!(glob_f, "{}", serde_json::to_string(g)?)?;
            stats.glob_rows += 1;
        }
        for r in &reads {
            if r.hit {
                stats.read_hits += 1;
            }
            writeln!(read_f, "{}", serde_json::to_string(r)?)?;
            stats.read_rows += 1;
        }
    }

    if stats.grep_rows == 0 || stats.glob_rows == 0 {
        let all_cases: Vec<TrainCase> = registry.by_id.values().cloned().collect();
        if stats.grep_rows == 0 {
            eprintln!(
                "export-tz: no grep rows in episodes — synthesizing from {} registry cases…",
                all_cases.len(),
            );
            let syn_start = Instant::now();
            for g in synthetic_grep(&all_cases, &case_gold, &idx) {
                if g.hit {
                    stats.grep_hits += 1;
                }
                writeln!(grep_f, "{}", serde_json::to_string(&g)?)?;
                stats.grep_rows += 1;
            }
            eprintln!(
                "export-tz: synthetic grep {} rows in {:.1}s",
                stats.grep_rows,
                syn_start.elapsed().as_secs_f64(),
            );
        }
        if stats.glob_rows == 0 {
            eprintln!(
                "export-tz: no glob rows in episodes — synthesizing from {} registry cases…",
                all_cases.len(),
            );
            let syn_start = Instant::now();
            for g in synthetic_glob(&all_cases, &case_gold, &idx) {
                if g.hit {
                    stats.glob_hits += 1;
                }
                writeln!(glob_f, "{}", serde_json::to_string(&g)?)?;
                stats.glob_rows += 1;
            }
            eprintln!(
                "export-tz: synthetic glob {} rows in {:.1}s",
                stats.glob_rows,
                syn_start.elapsed().as_secs_f64(),
            );
        }
    }

    search_f.flush().ok();
    grep_f.flush().ok();
    glob_f.flush().ok();
    read_f.flush().ok();
    drop(search_f);
    drop(grep_f);
    drop(glob_f);
    drop(read_f);
    for (part, final_path) in [
        (&search_part, &search_path),
        (&grep_part, &grep_path),
        (&glob_part, &glob_path),
        (&read_part, &read_path),
    ] {
        std::fs::rename(part, final_path).with_context(|| {
            format!("rename {} -> {}", part.display(), final_path.display())
        })?;
    }

    println!(
        "export-tz: episodes={} skipped_no_q={} skipped_no_gold={} joined_by_id={} \
         search={} (hit={}) grep={} (hit={}) glob={} (hit={}) read={} (hit={}) \
         read_skipped_no_gold_in_prefill={} read_snippets_hydrated={} -> {} / {} / {} / {} ({:.1}s total)",
        stats.episodes_total,
        stats.episodes_skipped_no_question,
        stats.episodes_skipped_no_gold,
        stats.episodes_joined_by_id,
        stats.search_rows,
        stats.search_hits,
        stats.grep_rows,
        stats.grep_hits,
        stats.glob_rows,
        stats.glob_hits,
        stats.read_rows,
        stats.read_hits,
        stats.read_skipped_no_gold_in_prefill,
        stats.read_snippets_hydrated,
        search_path.display(),
        grep_path.display(),
        glob_path.display(),
        read_path.display(),
        t0.elapsed().as_secs_f64(),
    );
    if stats.episodes_total > 0 && stats.episodes_joined_by_id == 0 {
        eprintln!(
            "export-tz warn: joined_by_id=0 — episodes lack tags.case_id; \
             re-run eval with current kb-eval (run=…) for reliable gold join"
        );
    }
    if stats.search_rows == 0 && stats.grep_rows == 0 && stats.glob_rows == 0 && stats.read_rows == 0 {
        anyhow::bail!("export produced no supervision rows — check episodes, registry, and gold");
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
    fn parse_search_hits_reads_ord_path_and_snippet() {
        let hits = parse_search_hits(
            "[#3] kb-test/doc.htm · Intro · CPU runs at 1000 MHz\n[#5] other · x · y",
        );
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].ord, 3);
        assert_eq!(hits[0].path, "kb-test/doc.htm");
        assert_eq!(hits[0].label, "Intro");
        assert_eq!(hits[0].snippet, "CPU runs at 1000 MHz");
    }

    #[test]
    fn parse_grep_hits_reads_snippet_line() {
        let hits = parse_grep_hits("kb-test/doc.pdf:#3: CPU MHz\nother:#1: x");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].snippet, "CPU MHz");
    }

    #[test]
    fn hits_to_json_preserves_parsed_snippet() {
        use glossa::index::store::DocIndex;
        use glossa::model::Chunk;

        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[Chunk {
            doc_path: "d.pdf".into(),
            location: "p.1".into(),
            file_type: "pdf".into(),
            text: "body".into(),
        }])
        .unwrap();
        let hits = vec![ParsedHit {
            ord: 1,
            path: "d.pdf".into(),
            label: "pdf".into(),
            snippet: "visible snippet".into(),
        }];
        let mut hydrated = 0u64;
        let json = hits_to_json(&idx, &hits, &mut hydrated);
        assert_eq!(json[0]["snippet"], "visible snippet");
        assert_eq!(hydrated, 0);
    }

    #[test]
    fn read_export_skips_when_gold_not_in_prefill() {
        use glossa::graph::ontology::Ontology;
        use glossa::graph::store::GraphStore;
        use glossa::index::store::DocIndex;
        use glossa::model::Chunk;

        let dir = tempfile::tempdir().unwrap();
        let work = dir.path();
        let idx = DocIndex::open_or_create(work).unwrap();
        idx.write_chunks(&[
            Chunk {
                doc_path: "d.pdf".into(),
                location: "p.1".into(),
                file_type: "pdf".into(),
                text: "near".into(),
            },
            Chunk {
                doc_path: "d.pdf".into(),
                location: "p.99".into(),
                file_type: "pdf".into(),
                text: "gold".into(),
            },
        ])
        .unwrap();
        let graph = GraphStore::open(work).unwrap();
        let ont = Ontology::load_or_default(work);
        let msgs = vec![
            json!({"role":"user","content":[{"type":"text","text":"Question: q?"}]}),
            json!({"role":"assistant","content":[{"type":"tool_call","id":"s1","name":"search","arguments":{"query":"q"}}]}),
            json!({"role":"user","content":[{"type":"tool_result","id":"s1","name":"search","result":"[#1] d.pdf · pdf · near text"}]}),
            json!({"role":"assistant","content":[{"type":"tool_call","id":"r1","name":"read","arguments":{"path":"d.pdf","n":1}}]}),
            json!({"role":"user","content":[{"type":"tool_result","id":"r1","name":"read","result":"near"}]}),
        ];
        let case = TrainCase {
            id: "t".into(),
            question: "q?".into(),
            answer: "gold".into(),
            source: Some("d.pdf#p.99".into()),
        };
        let gold = gold_for_case(&case, &graph, &ont, &idx);
        let mut skipped = 0u64;
        let mut hydrated = 0u64;
        let (_, _, _, reads) = examples_from_episode(
            "ep",
            &msgs,
            &case,
            &gold,
            &idx,
            10,
            &mut skipped,
            &mut hydrated,
        );
        assert_eq!(reads.len(), 0);
        assert_eq!(skipped, 1);
    }

    #[test]
    fn read_export_keeps_solvable_prefill_with_snippet() {
        use glossa::graph::ontology::Ontology;
        use glossa::graph::store::GraphStore;
        use glossa::index::store::DocIndex;
        use glossa::model::Chunk;

        let dir = tempfile::tempdir().unwrap();
        let work = dir.path();
        let idx = DocIndex::open_or_create(work).unwrap();
        idx.write_chunks(&[Chunk {
            doc_path: "d.pdf".into(),
            location: "p.3".into(),
            file_type: "pdf".into(),
            text: "answer".into(),
        }])
        .unwrap();
        let graph = GraphStore::open(work).unwrap();
        let ont = Ontology::load_or_default(work);
        let msgs = vec![
            json!({"role":"user","content":[{"type":"text","text":"Question: q?"}]}),
            json!({"role":"assistant","content":[{"type":"tool_call","id":"s1","name":"search","arguments":{"query":"q"}}]}),
            json!({"role":"user","content":[{"type":"tool_result","id":"s1","name":"search","result":"[#3] d.pdf · pdf · answer text"}]}),
            json!({"role":"assistant","content":[{"type":"tool_call","id":"r1","name":"read","arguments":{"path":"d.pdf","n":3}}]}),
            json!({"role":"user","content":[{"type":"tool_result","id":"r1","name":"read","result":"answer"}]}),
        ];
        let case = TrainCase {
            id: "t".into(),
            question: "q?".into(),
            answer: "answer".into(),
            source: Some("d.pdf#p.3".into()),
        };
        let gold = gold_for_case(&case, &graph, &ont, &idx);
        let mut skipped = 0u64;
        let mut hydrated = 0u64;
        let (_, _, _, reads) = examples_from_episode(
            "ep",
            &msgs,
            &case,
            &gold,
            &idx,
            10,
            &mut skipped,
            &mut hydrated,
        );
        assert_eq!(reads.len(), 1);
        assert_eq!(skipped, 0);
        assert_eq!(reads[0].hits[0]["snippet"], "answer text");
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

        let mut skipped = 0u64;
        let mut hydrated = 0u64;
        let (searches, _greps, _globs, reads) = examples_from_episode(
            "ep1",
            &msgs,
            &case,
            &gold_for_case(&case, &graph, &ont, &idx),
            &idx,
            10,
            &mut skipped,
            &mut hydrated,
        );
        assert_eq!(searches.len(), 1);
        assert!(searches[0].hit, "expected search hit after path canonicalization");
        assert_eq!(searches[0].gold, vec!["kb-test/doc.pdf#p.3"]);
        assert_eq!(reads.len(), 1);
        assert!(reads[0].hit);
        assert_eq!(reads[0].model_read.as_ref().unwrap().path, "kb-test/doc.pdf");
        assert_eq!(reads[0].hits[0]["path"], "kb-test/doc.pdf");
        assert_eq!(reads[0].hits[0]["location"], "p.3");
        assert_eq!(reads[0].hits[0]["snippet"], "CPU runs at 1000 MHz");
    }

    #[test]
    fn parse_grep_hits_reads_ord_path_and_snippet() {
        let hits = parse_grep_hits("kb-test/doc.pdf:#3: CPU MHz\nother:#1: x");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].ord, 3);
        assert_eq!(hits[0].path, "kb-test/doc.pdf");
        assert_eq!(hits[0].snippet, "CPU MHz");
    }

    #[test]
    fn parse_glob_paths_reads_document_lines() {
        let paths = parse_glob_paths("kb-test/doc.pdf  (12 chunks)\nother.htm  (3 chunks)");
        assert_eq!(paths, vec!["kb-test/doc.pdf", "other.htm"]);
    }

    #[test]
    fn parse_retrieval_events_includes_grep_and_glob() {
        let msgs = vec![
            json!({"role":"assistant","content":[{"type":"tool_call","id":"g1","name":"grep","arguments":{"pattern":"maxTsdr"}}]}),
            json!({"role":"user","content":[{"type":"tool_result","id":"g1","name":"grep","result":"a.pdf:#1: maxTsdr"}]}),
            json!({"role":"assistant","content":[{"type":"tool_call","id":"l1","name":"glob","arguments":{"pattern":"*.pdf"}}]}),
            json!({"role":"user","content":[{"type":"tool_result","id":"l1","name":"glob","result":"a.pdf  (2 chunks)"}]}),
        ];
        let ev = parse_retrieval_events(&msgs);
        assert!(matches!(&ev[0], EpisodeEvent::Grep { pattern, .. } if pattern == "maxTsdr"));
        assert!(matches!(&ev[1], EpisodeEvent::Glob { pattern, .. } if pattern == "*.pdf"));
    }
}
