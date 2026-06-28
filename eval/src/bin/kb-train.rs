//! kb-train — stream the LIVE reasoning graph and optimize retrieval prompts against it as a silver
//! answer-key. v1: tails the graph (SQLite WAL — reads concurrently with the enricher writing it),
//! and for each reasoning node builds the two retrieval sub-tasks from its `MENTIONS` gold chunk:
//!   - query/recall: does `search(label)` surface the gold chunk in top-k?  (objective, no model)
//!   - select:       given the hits, does the model pick the gold chunk?    (one model call, exact-match)
//! It prints rolling baselines and runs until the graph stops growing. The GEPA-style mutate loop
//! (improve the select prompt from failures via a cloud mutator) is layered on next.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use glossa::graph::ontology::Ontology;
use glossa::graph::store::GraphStore;
use glossa::index::store::{DocIndex, RankedHit};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser)]
#[command(name = "kb-train", about = "Build & learn: enrich the reasoning graph from solved cases, and optimize retrieval prompts against it")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Enrich the reasoning graph from solved training cases (reverse-trace Q→A into nodes/edges).
    Enrich {
        /// Path to the training JSON (array of {_id, question, answer}).
        #[arg(long)]
        train: PathBuf,
        /// Corpus root (index + graph live here).
        #[arg(long, default_value = "kb-test")]
        work: PathBuf,
        /// Enrich the first N cases (0 = all).
        #[arg(long, default_value_t = 0)]
        limit: usize,
        /// TensorZero gateway base URL.
        #[arg(long, default_value = "http://localhost:3000")]
        tensorzero_endpoint: String,
        /// TensorZero function name.
        #[arg(long, default_value = "enrich")]
        tensorzero_function: String,
    },
    /// Dump the gold-supervision retrieval datasets from the graph (GPU-free): for each gated node,
    /// `search(label)` and write a query example (question → gold) to `query.jsonl` and, when gold is
    /// in the hits, a select example (question + hits → gold ord) to `select.jsonl`.
    Dump {
        /// Corpus root (live graph + index).
        #[arg(long, default_value = "kb-test")]
        work: PathBuf,
        /// Directory to write `query.jsonl` / `select.jsonl` into.
        #[arg(long, default_value = ".")]
        out: PathBuf,
        /// search top-k.
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// seconds to wait before re-polling for new nodes (tail mode).
        #[arg(long, default_value_t = 20)]
        poll_secs: u64,
        /// stop after this many consecutive idle polls.
        #[arg(long, default_value_t = 10)]
        idle_stop: u32,
        /// process the current nodes once and exit (no tailing) — the debug default.
        #[arg(long)]
        once: bool,
    },
    /// Optimize retrieval against the LIVE graph: tail reasoning nodes, score recall@k + select.
    Optimize {
        /// Corpus root (live graph + index). Read concurrently with `enrich` (SQLite WAL).
        #[arg(long, default_value = "kb-test")]
        work: PathBuf,
        /// TZ gateway base URL (OpenAI-compatible endpoint used for the select call).
        #[arg(long, default_value = "http://localhost:3000")]
        gateway: String,
        /// Model id for the select step (gateway OpenAI-compat).
        #[arg(long, default_value = "tensorzero::model_name::qwen")]
        model: String,
        /// search top-k.
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// seconds to wait before re-polling the graph for new nodes.
        #[arg(long, default_value_t = 20)]
        poll_secs: u64,
        /// stop after this many consecutive idle polls (no new nodes).
        #[arg(long, default_value_t = 10)]
        idle_stop: u32,
        /// process the current nodes once and exit (no tailing).
        #[arg(long)]
        once: bool,
    },
}

/// First run of ASCII digits in a string (e.g. "[#45]" / "45" / "chunk 45" -> 45).
fn first_int(s: &str) -> Option<u64> {
    let digits: String = s.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// The current select-prompt being evaluated (v1 baseline; v2 will let the mutator evolve it).
const SELECT_PROMPT: &str = "You are given a support question and numbered search results from a knowledge base. Exactly one result contains the answer. Reply with ONLY that result's number (the integer shown after `#`). No words, no punctuation.";

fn hits_block(hits: &[RankedHit]) -> String {
    hits.iter()
        .map(|h| {
            let label = if h.location.starts_with("p.") { h.file_type.as_str() } else { h.location.as_str() };
            format!("[#{}] {} · {} · {}", h.ord, h.path, label, h.snippet)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Ask the model to pick the chunk number that answers the question, given the hits. Returns the
/// chosen ord (None on parse/transport failure — counted as a miss).
fn select_call(gateway: &str, model: &str, question: &str, hits_text: &str) -> Option<u64> {
    let url = format!("{}/openai/v1/chat/completions", gateway.trim_end_matches('/'));
    let body = json!({
        "model": model, "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SELECT_PROMPT},
            {"role": "user", "content": format!("Question: {question}\n\nResults:\n{hits_text}\n\nWhich result number contains the answer?")}
        ]
    });
    let resp = ureq::post(&url).timeout(Duration::from_secs(120)).set("Content-Type", "application/json").send_string(&body.to_string());
    let text = resp.ok()?.into_string().ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
    first_int(content)
}

/// Quality gate for a training example: is the chunk a node MENTIONS actually ABOUT that node?
/// Term overlap on the shared tantivy analyzer — at least half the label's distinct (stemmed) terms
/// must appear in the chunk text. Catches the enricher linking a node to the wrong page, so bad
/// (concept → chunk) pairs never reach the optimizer. Cheap, deterministic, no LLM.
fn evidence_is_relevant(analyzer: &glossa::index::multilang::TermAnalyzer, label: &str, chunk_text: &str) -> bool {
    let mut label_terms = HashSet::new();
    {
        let mut s = std::collections::BTreeSet::new();
        analyzer.terms(label, &mut s);
        label_terms.extend(s);
    }
    if label_terms.is_empty() {
        return false;
    }
    let mut chunk_terms = std::collections::BTreeSet::new();
    analyzer.terms(chunk_text, &mut chunk_terms);
    let shared = label_terms.iter().filter(|t| chunk_terms.contains(*t)).count();
    shared * 2 >= label_terms.len()
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Enrich { train, work, limit, tensorzero_endpoint, tensorzero_function } => {
            kb_eval::enrich::run_enrich(&train, &work, limit, &tensorzero_endpoint, &tensorzero_function)
        }
        Cmd::Dump { work, out, k, poll_secs, idle_stop, once } => {
            run_dump(work, out, k, poll_secs, idle_stop, once)
        }
        Cmd::Optimize { work, gateway, model, k, poll_secs, idle_stop, once } => {
            run_optimize(work, gateway, model, k, poll_secs, idle_stop, once)
        }
    }
}

/// Render the hits a select example must choose among as structured JSON — the SAME fields
/// `hits_block` renders, so GEPA can re-render under any evolved prompt without re-searching.
fn hits_json(hits: &[RankedHit]) -> Vec<Value> {
    hits.iter()
        .map(|h| json!({
            "ord": h.ord, "path": h.path, "location": h.location,
            "file_type": h.file_type, "snippet": h.snippet,
        }))
        .collect()
}

/// Collect the gold-supervision retrieval datasets from the live graph — GPU-free, no model.
/// For each non-structural reasoning node with a `MENTIONS` anchor that passes the relevance gate:
///   - `query.jsonl`:  {question, gold:[path#loc]}            — the recall target for the query stage
///   - `select.jsonl`: {question, hits:[...], gold_ords:[n]}  — written only when gold is in the hits
#[allow(clippy::too_many_arguments)]
fn run_dump(work: PathBuf, out: PathBuf, k: usize, poll_secs: u64, idle_stop: u32, once: bool) -> Result<()> {
    use std::io::Write as _;

    let idx = DocIndex::open_or_create(&work).context("open index")?;
    let graph = GraphStore::open(&work).context("open graph")?;
    let ont = Ontology::load_or_default(&work);
    let structural: HashSet<String> = ont.structural().into_iter().collect();
    let analyzer = glossa::index::multilang::TermAnalyzer::new();

    std::fs::create_dir_all(&out).with_context(|| format!("create {}", out.display()))?;
    let query_path = out.join("query.jsonl");
    let select_path = out.join("select.jsonl");
    let mut query_f = std::fs::File::create(&query_path).with_context(|| format!("create {}", query_path.display()))?;
    let mut select_f = std::fs::File::create(&select_path).with_context(|| format!("create {}", select_path.display()))?;
    println!(
        "kb-train dump: {} (mentions='{}', k={}) -> {} / {}",
        work.display(), glossa::graph::MENTIONS, k, query_path.display(), select_path.display(),
    );

    let mut seen: HashSet<String> = HashSet::new();
    let (mut nodes_kept, mut dropped_no_gold, mut dropped_gate, mut query_written, mut select_written, mut recall_hit) =
        (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let mut idle = 0u32;

    loop {
        // Re-query the live graph each poll — WAL means we see the enricher's newly-committed nodes.
        let nodes = graph.all_nodes().context("all_nodes")?;
        let mut fresh = 0u64;
        for node in &nodes {
            if structural.contains(&node.node_type) || !seen.insert(node.id.clone()) {
                continue;
            }
            fresh += 1;
            // gold = the (path, location) of every section this node MENTIONS.
            let mut gold: Vec<(String, String)> = Vec::new();
            for e in graph.outgoing(&node.id).unwrap_or_default() {
                if e.edge_type == glossa::graph::MENTIONS {
                    if let Some((p, l)) = e.to.split_once('#') {
                        gold.push((p.to_string(), l.to_string()));
                    }
                }
            }
            if gold.is_empty() {
                dropped_no_gold += 1;
                continue; // a node with no evidence anchor can't be a retrieval target
            }
            // Gate: keep only the MENTIONS chunks actually ABOUT the node's label. The enricher can
            // link a node to the wrong page; those (concept → chunk) pairs would poison the optimizer.
            let relevant: Vec<(String, String)> = gold
                .into_iter()
                .filter(|(p, l)| {
                    idx.read_chunk(p, l)
                        .ok()
                        .flatten()
                        .map(|text| evidence_is_relevant(&analyzer, &node.label, &text))
                        .unwrap_or(false)
                })
                .collect();
            if relevant.is_empty() {
                dropped_gate += 1;
                continue;
            }
            nodes_kept += 1;

            // query example: question -> the gold locations a good query must surface.
            let gold_locs: Vec<String> = relevant.iter().map(|(p, l)| format!("{p}#{l}")).collect();
            writeln!(query_f, "{}", json!({"question": node.label, "gold": gold_locs}))?;
            query_written += 1;

            // select example: only meaningful when the gold chunk is among the hits to choose from.
            let hits = idx.search_filtered(&node.label, k, None, None).unwrap_or_default();
            let gold_set: HashSet<(String, String)> = relevant.into_iter().collect();
            let gold_ords: Vec<u64> =
                hits.iter().filter(|h| gold_set.contains(&(h.path.clone(), h.location.clone()))).map(|h| h.ord).collect();
            if !gold_ords.is_empty() {
                recall_hit += 1;
                writeln!(select_f, "{}", json!({
                    "question": node.label, "hits": hits_json(&hits), "gold_ords": gold_ords,
                }))?;
                select_written += 1;
            }
        }
        if once {
            break;
        }
        idle = if fresh == 0 { idle + 1 } else { 0 };
        println!("[poll] {} reasoning nodes seen (+{fresh}); idle={idle}/{idle_stop}", seen.len());
        if idle >= idle_stop {
            println!("graph stopped growing — done");
            break;
        }
        std::thread::sleep(Duration::from_secs(poll_secs));
    }

    query_f.flush().ok();
    select_f.flush().ok();
    println!(
        "DONE: kept {nodes_kept} nodes (dropped {dropped_no_gold} no-gold, {dropped_gate} gate)  \
         query={query_written}  select={select_written}  baseline recall@{k}={:.3} ({recall_hit}/{nodes_kept})",
        recall_hit as f64 / nodes_kept.max(1) as f64,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_optimize(work: PathBuf, gateway: String, model: String, k: usize, poll_secs: u64, idle_stop: u32, once: bool) -> Result<()> {
    let idx = DocIndex::open_or_create(&work).context("open index")?;
    let graph = GraphStore::open(&work).context("open graph")?;
    let ont = Ontology::load_or_default(&work);
    let structural: HashSet<String> = ont.structural().into_iter().collect();
    println!("kb-train: tailing {} (mentions='{}', k={}, model={})", work.display(), glossa::graph::MENTIONS, k, model);

    let mut seen: HashSet<String> = HashSet::new();
    let (mut n, mut recall_hit, mut select_total, mut select_ok) = (0u64, 0u64, 0u64, 0u64);
    let mut idle = 0u32;

    loop {
        // Re-query the live graph each poll — WAL means we see the enricher's newly-committed nodes.
        let nodes = graph.all_nodes().context("all_nodes")?;
        let mut fresh = 0u64;
        for node in &nodes {
            if structural.contains(&node.node_type) || !seen.insert(node.id.clone()) {
                continue;
            }
            fresh += 1;
            // gold = the (path, location) of every section this node MENTIONS.
            let mut gold: HashSet<(String, String)> = HashSet::new();
            for e in graph.outgoing(&node.id).unwrap_or_default() {
                if e.edge_type == glossa::graph::MENTIONS {
                    if let Some((p, l)) = e.to.split_once('#') {
                        gold.insert((p.to_string(), l.to_string()));
                    }
                }
            }
            if gold.is_empty() {
                continue; // a node with no evidence anchor can't be a retrieval target
            }
            n += 1;
            let hits = idx.search_filtered(&node.label, k, None, None).unwrap_or_default();
            // gold ords actually present in the top-k hits
            let gold_ords: HashSet<u64> =
                hits.iter().filter(|h| gold.contains(&(h.path.clone(), h.location.clone()))).map(|h| h.ord).collect();
            if !gold_ords.is_empty() {
                recall_hit += 1;
                // select is only meaningful when the gold chunk is in the hits to choose from
                select_total += 1;
                if let Some(pick) = select_call(&gateway, &model, &node.label, &hits_block(&hits)) {
                    if gold_ords.contains(&pick) {
                        select_ok += 1;
                    }
                }
            }
            if n % 5 == 0 {
                println!(
                    "[{n}] recall@{}={:.2}  select_acc={:.2} ({select_ok}/{select_total})",
                    k,
                    recall_hit as f64 / n as f64,
                    if select_total > 0 { select_ok as f64 / select_total as f64 } else { 0.0 },
                );
            }
        }
        if once {
            break;
        }
        idle = if fresh == 0 { idle + 1 } else { 0 };
        println!("[poll] {} reasoning nodes seen (+{fresh}); idle={idle}/{}", seen.len(), idle_stop);
        if idle >= idle_stop {
            println!("graph stopped growing — done");
            break;
        }
        std::thread::sleep(Duration::from_secs(poll_secs));
    }

    println!(
        "DONE: {n} cases  recall@{}={:.3}  select_acc={:.3} ({select_ok}/{select_total})",
        k,
        recall_hit as f64 / n.max(1) as f64,
        select_ok as f64 / select_total.max(1) as f64,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_keeps_relevant_drops_wrong_chunk() {
        let a = glossa::index::multilang::TermAnalyzer::new();
        // chunk really about the label → kept
        assert!(evidence_is_relevant(&a, "Изменить maxTsdr в 3000",
            "Установите параметр maxTsdr равным 3000 Tbit и нажмите recalculate"));
        // chunk about something unrelated → dropped (enricher linked the wrong page)
        assert!(!evidence_is_relevant(&a, "Изменить maxTsdr в 3000",
            "Калибровка аналоговых каналов выполняется через меню настройки прибора"));
    }

    #[test]
    fn hits_json_carries_the_select_choice_fields() {
        let hits = vec![RankedHit {
            path: "doc.pdf".into(), location: "p.7".into(), file_type: "pdf".into(),
            ord: 7, snippet: "the answer".into(), score: 1.0,
        }];
        let j = hits_json(&hits);
        assert_eq!(j.len(), 1);
        // the model selects by `ord`; path/location identify it as gold; snippet is what it reads.
        assert_eq!(j[0]["ord"], 7);
        assert_eq!(j[0]["path"], "doc.pdf");
        assert_eq!(j[0]["location"], "p.7");
        assert_eq!(j[0]["snippet"], "the answer");
        // score is internal ranking noise — it must NOT leak into the model-facing example.
        assert!(j[0].get("score").is_none());
    }
}
