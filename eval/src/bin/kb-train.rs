//! kb-train — stream the LIVE reasoning graph and optimize retrieval prompts against it as a silver
//! answer-key. v1: tails the graph (SQLite WAL — reads concurrently with the enricher writing it),
//! and for each reasoning node builds the two retrieval sub-tasks from its `MENTIONS` gold chunk:
//!   - query/recall: does `search(label)` surface the gold chunk in top-k?  (objective, no model)
//!   - select:       given the hits, does the model pick the gold chunk?    (one model call, exact-match)
//! It prints rolling baselines and runs until the graph stops growing. The GEPA-style mutate loop
//! (improve the select prompt from failures via a cloud mutator) is layered on next.

use anyhow::{Context, Result};
use clap::Parser;
use glossa::graph::ontology::Ontology;
use glossa::graph::store::GraphStore;
use glossa::index::store::{DocIndex, RankedHit};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::time::Duration;

#[derive(Parser)]
#[command(name = "kb-train", about = "Stream the reasoning graph; optimize retrieval against its MENTIONS gold chunks")]
struct Cli {
    /// Corpus root (live graph + index). Read concurrently with the enricher (SQLite WAL).
    #[arg(long, default_value = "kb-test")]
    work: std::path::PathBuf,
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    let idx = DocIndex::open_or_create(&cli.work).context("open index")?;
    let graph = GraphStore::open(&cli.work).context("open graph")?;
    let ont = Ontology::load_or_default(&cli.work);
    let structural: HashSet<String> = ont.structural().into_iter().collect();
    println!("kb-train: tailing {} (mentions='{}', k={}, model={})", cli.work.display(), glossa::graph::MENTIONS, cli.k, cli.model);

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
            let hits = idx.search_filtered(&node.label, cli.k, None, None).unwrap_or_default();
            // gold ords actually present in the top-k hits
            let gold_ords: HashSet<u64> =
                hits.iter().filter(|h| gold.contains(&(h.path.clone(), h.location.clone()))).map(|h| h.ord).collect();
            if !gold_ords.is_empty() {
                recall_hit += 1;
                // select is only meaningful when the gold chunk is in the hits to choose from
                select_total += 1;
                if let Some(pick) = select_call(&cli.gateway, &cli.model, &node.label, &hits_block(&hits)) {
                    if gold_ords.contains(&pick) {
                        select_ok += 1;
                    }
                }
            }
            if n % 5 == 0 {
                println!(
                    "[{n}] recall@{}={:.2}  select_acc={:.2} ({select_ok}/{select_total})",
                    cli.k,
                    recall_hit as f64 / n as f64,
                    if select_total > 0 { select_ok as f64 / select_total as f64 } else { 0.0 },
                );
            }
        }
        if cli.once {
            break;
        }
        idle = if fresh == 0 { idle + 1 } else { 0 };
        println!("[poll] {} reasoning nodes seen (+{fresh}); idle={idle}/{}", seen.len(), cli.idle_stop);
        if idle >= cli.idle_stop {
            println!("graph stopped growing — done");
            break;
        }
        std::thread::sleep(Duration::from_secs(cli.poll_secs));
    }

    println!(
        "DONE: {n} cases  recall@{}={:.3}  select_acc={:.3} ({select_ok}/{select_total})",
        cli.k,
        recall_hit as f64 / n.max(1) as f64,
        select_ok as f64 / select_total.max(1) as f64,
    );
    Ok(())
}
