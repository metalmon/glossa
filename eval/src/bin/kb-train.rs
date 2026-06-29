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
    /// GEPA-optimize the `select` prompt against a dumped `select.jsonl`: reflect on the failures of
    /// the local model with a cloud mutator LM, keep prompts that beat their parent (Pareto frontier).
    /// All inferences and run metrics are logged under the TZ `select` / `gepa_reflect` functions.
    Optimize {
        /// The select dataset produced by `kb-train dump`.
        #[arg(long, default_value = "select.jsonl")]
        select: PathBuf,
        /// File to write the best prompt found into.
        #[arg(long, default_value = "select.prompt.txt")]
        out: PathBuf,
        /// TensorZero gateway base URL.
        #[arg(long, default_value = "http://localhost:3000")]
        gateway: String,
        /// TZ function for per-case select calls (visible in the UI metrics dashboard).
        #[arg(long, default_value = "select")]
        function: String,
        /// TZ variant for this run — filter by variant in the UI; add new variants to `tensorzero.toml`.
        #[arg(long, default_value = "baseline")]
        variant: String,
        /// TZ function for reflection/mutator calls.
        #[arg(long, default_value = "gepa_reflect")]
        reflect_function: String,
        /// Run label → TZ tag `run=<value>` (for ClickHouse); auto-generated if omitted.
        #[arg(long)]
        run: Option<String>,
        /// Extra TZ tags on inferences + feedback (repeatable KEY=VALUE).
        #[arg(long = "tag", value_name = "KEY=VALUE")]
        tag: Vec<String>,
        /// Fraction of examples held out for validation.
        #[arg(long, default_value_t = 0.3)]
        val_frac: f64,
        /// Number of reflection iterations.
        #[arg(long, default_value_t = 6)]
        budget: usize,
        /// How many failures to show the mutator per reflection.
        #[arg(long, default_value_t = 4)]
        minibatch: usize,
    },
}

/// The seed select-prompt GEPA starts its reflection from (also the production select baseline).
const SELECT_PROMPT: &str = "You are given a support question and numbered search results from a knowledge base. Exactly one result contains the answer. Reply with ONLY that result's number (the integer shown after `#`). No words, no punctuation.";

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
        Cmd::Optimize {
            select,
            out,
            gateway,
            function,
            variant,
            reflect_function,
            run,
            tag,
            val_frac,
            budget,
            minibatch,
        } => run_optimize(
            select,
            out,
            gateway,
            function,
            variant,
            reflect_function,
            run,
            tag,
            val_frac,
            budget,
            minibatch,
        ),
    }
}

/// Load `select.jsonl` (one JSON example per line) and GEPA-optimize the select prompt against it.
#[allow(clippy::too_many_arguments)]
fn run_optimize(
    select: PathBuf,
    out: PathBuf,
    gateway: String,
    function: String,
    variant: String,
    reflect_function: String,
    run: Option<String>,
    tag: Vec<String>,
    val_frac: f64,
    budget: usize,
    minibatch: usize,
) -> Result<()> {
    let text = std::fs::read_to_string(&select).with_context(|| format!("read {}", select.display()))?;
    let examples: Vec<kb_eval::gepa::SelectExample> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()
        .context("parse select.jsonl")?;

    let run_label = run.unwrap_or_else(kb_eval::gepa::default_run_tag);
    let mut tags = serde_json::Map::new();
    tags.insert("run".into(), run_label.clone().into());
    tags.insert("budget".into(), budget.to_string().into());
    tags.insert("minibatch".into(), minibatch.to_string().into());
    for t in &tag {
        if let Some((k, v)) = t.split_once('=') {
            tags.insert(k.to_string(), v.into());
        }
    }

    let result = kb_eval::gepa::run(
        kb_eval::gepa::GepaConfig {
            gateway,
            function,
            variant,
            reflect_function,
            episode_id: kb_eval::tz::backdated_episode_id(30),
            tags: Value::Object(tags),
            val_frac,
            budget,
            minibatch,
            seed_prompt: SELECT_PROMPT.to_string(),
        },
        examples,
    )?;
    std::fs::write(&out, &result.prompt).with_context(|| format!("write {}", out.display()))?;
    println!("wrote best prompt -> {} (run={run_label}, episode={})", out.display(), result.episode_id);
    Ok(())
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
