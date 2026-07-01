//! kb-train — enrich the reasoning graph, export TZ episode supervision, GEPA-optimize retrieval.

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
#[command(name = "kb-train", about = "Build & learn: enrich graph, export TZ episodes, GEPA optimize")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Enrich the reasoning graph from solved training cases.
    Enrich {
        #[arg(long)]
        train: PathBuf,
        #[arg(long, default_value = "kb-test")]
        work: PathBuf,
        #[arg(long, default_value_t = 0)]
        limit: usize,
        #[arg(long, default_value = "http://127.0.0.1:3000")]
        tensorzero_endpoint: String,
        #[arg(long, default_value = "enrich")]
        tensorzero_function: String,
    },
    /// Export search/read datasets from TZ ClickHouse episodes (primary GEPA source).
    ExportTz {
        #[arg(long, default_value = "kb-test")]
        work: PathBuf,
        #[arg(long, default_value = "gepa-out")]
        out: PathBuf,
        #[arg(long, action = clap::ArgAction::Append)]
        train: Vec<PathBuf>,
        #[arg(long, default_value = "answer_hotpot")]
        function: String,
        #[arg(long)]
        run: Option<String>,
        #[arg(long)]
        clickhouse: Option<String>,
        #[arg(long, default_value_t = 10)]
        k: usize,
    },
    /// Legacy graph dump (auxiliary gold index only — not primary training source).
    Dump {
        #[arg(long, default_value = "kb-test")]
        work: PathBuf,
        #[arg(long, default_value = ".")]
        out: PathBuf,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long, default_value_t = 20)]
        poll_secs: u64,
        #[arg(long, default_value_t = 10)]
        idle_stop: u32,
        #[arg(long)]
        once: bool,
    },
    /// GEPA-optimize the prod answer_hotpot system prompt (quad search + grep + glob + read scoring).
    Optimize {
        #[arg(long, default_value = "gepa-out/search.jsonl")]
        search: PathBuf,
        #[arg(long, default_value = "gepa-out/grep.jsonl")]
        grep: PathBuf,
        #[arg(long, default_value = "gepa-out/glob.jsonl")]
        glob: PathBuf,
        #[arg(long, default_value = "gepa-out/read.jsonl")]
        read: PathBuf,
        #[arg(long, default_value = "gepa-out/answer_hotpot.prompt.txt")]
        out: PathBuf,
        #[arg(long, default_value = "eval/tensorzero/config/answer_hotpot/system.minijinja")]
        seed: PathBuf,
        #[arg(long, default_value = "kb-test")]
        work: PathBuf,
        #[arg(long, default_value = "http://127.0.0.1:3000")]
        gateway: String,
        #[arg(long, default_value = "search")]
        search_function: String,
        #[arg(long, default_value = "read")]
        read_function: String,
        #[arg(long, default_value = "grep")]
        grep_function: String,
        #[arg(long, default_value = "glob")]
        glob_function: String,
        #[arg(long, default_value = "baseline")]
        variant: String,
        #[arg(long, default_value = "gepa_reflect")]
        reflect_function: String,
        #[arg(long)]
        run: Option<String>,
        #[arg(long = "tag", value_name = "KEY=VALUE")]
        tag: Vec<String>,
        #[arg(long, default_value_t = 0.3)]
        val_frac: f64,
        #[arg(long, default_value_t = 6)]
        budget: usize,
        #[arg(long, default_value_t = 12)]
        minibatch: usize,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long, default_value_t = 0.25)]
        w_search: f64,
        #[arg(long, default_value_t = 0.25)]
        w_grep: f64,
        #[arg(long, default_value_t = 0.10)]
        w_glob: f64,
        #[arg(long, default_value_t = 0.25)]
        w_read: f64,
        /// RNG seed for minibatch sampling (default: hash of run tag).
        #[arg(long)]
        rng_seed: Option<u64>,
        /// Cap on D_pareto instances sampled from val (parent selection + pool scores).
        #[arg(long, default_value_t = 20)]
        pareto_size: usize,
        /// Parent selection: pareto (default) or current_best.
        #[arg(long, default_value = "pareto")]
        candidate_selection: String,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Enrich { train, work, limit, tensorzero_endpoint, tensorzero_function } => {
            kb_eval::enrich::run_enrich(&train, &work, limit, &tensorzero_endpoint, &tensorzero_function)
        }
        Cmd::ExportTz { work, out, train, function, run, clickhouse, k } => {
            let train_files = if train.is_empty() {
                vec![PathBuf::from("kb-val/derived/synthetic-train.json")]
            } else {
                train
            };
            kb_eval::export_tz::run_export(kb_eval::export_tz::ExportConfig {
                clickhouse_url: clickhouse.unwrap_or_else(kb_eval::export_tz::default_clickhouse_url),
                function_name: function,
                run_tag: run,
                train_files,
                work,
                out,
                top_k: k,
            })
            .map(|_| ())
        }
        Cmd::Dump { work, out, k, poll_secs, idle_stop, once } => {
            run_dump(work, out, k, poll_secs, idle_stop, once)
        }
        Cmd::Optimize {
            search,
            grep,
            glob,
            read,
            out,
            seed,
            work,
            gateway,
            search_function,
            read_function,
            grep_function,
            glob_function,
            variant,
            reflect_function,
            run,
            tag,
            val_frac,
            budget,
            minibatch,
            k,
            w_search,
            w_grep,
            w_glob,
            w_read,
            rng_seed,
            pareto_size,
            candidate_selection,
        } => run_optimize(
            search,
            grep,
            glob,
            read,
            out,
            seed,
            work,
            gateway,
            search_function,
            read_function,
            grep_function,
            glob_function,
            variant,
            reflect_function,
            run,
            tag,
            val_frac,
            budget,
            minibatch,
            k,
            w_search,
            w_grep,
            w_glob,
            w_read,
            rng_seed,
            pareto_size,
            candidate_selection,
        ),
    }
}

fn load_jsonl<T: serde::de::DeserializeOwned>(path: &PathBuf) -> Result<Vec<T>> {
    use std::io::{BufRead, BufReader};
    let f = std::fs::File::open(path).with_context(|| format!("read {}", path.display()))?;
    let reader = BufReader::new(f);
    let mut out = Vec::new();
    for (n, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "read {} line {} (invalid UTF-8 — re-run `just export-tz`; file may be corrupted by concurrent export)",
                path.display(),
                n + 1,
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(&line).with_context(|| {
            format!("parse {} line {}", path.display(), n + 1)
        })?);
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn run_optimize(
    search: PathBuf,
    grep: PathBuf,
    glob: PathBuf,
    read: PathBuf,
    out: PathBuf,
    seed: PathBuf,
    work: PathBuf,
    gateway: String,
    search_function: String,
    read_function: String,
    grep_function: String,
    glob_function: String,
    variant: String,
    reflect_function: String,
    run: Option<String>,
    tag: Vec<String>,
    val_frac: f64,
    budget: usize,
    minibatch: usize,
    top_k: usize,
    w_search: f64,
    w_grep: f64,
    w_glob: f64,
    w_read: f64,
    rng_seed: Option<u64>,
    pareto_size: usize,
    candidate_selection: String,
) -> Result<()> {
    let searches: Vec<kb_eval::gepa::SearchExample> = if search.exists() {
        load_jsonl(&search)?
    } else {
        Vec::new()
    };
    let greps: Vec<kb_eval::export_tz::GrepExample> = if grep.exists() {
        load_jsonl::<kb_eval::export_tz::GrepExample>(&grep)?
            .into_iter()
            .filter(|g| !g.synthetic)
            .collect()
    } else {
        Vec::new()
    };
    let globs: Vec<kb_eval::export_tz::GlobExample> = if glob.exists() {
        load_jsonl::<kb_eval::export_tz::GlobExample>(&glob)?
            .into_iter()
            .filter(|g| !g.synthetic)
            .collect()
    } else {
        Vec::new()
    };
    let reads: Vec<kb_eval::export_tz::ReadExample> = if read.exists() {
        load_jsonl(&read)?
    } else {
        Vec::new()
    };

    let seed_prompt = kb_eval::gepa::load_seed_prompt(&seed)?;
    let run_label = run.unwrap_or_else(kb_eval::gepa::default_run_tag);
    let rng_seed = rng_seed.unwrap_or_else(|| kb_eval::gepa::hash_run_seed(&run_label));
    let candidate_selection = candidate_selection
        .parse::<kb_eval::gepa::CandidateSelection>()
        .with_context(|| format!("parse candidate_selection {candidate_selection:?}"))?;
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
            search_function,
            read_function,
            grep_function,
            glob_function,
            variant,
            reflect_function,
            episode_id: kb_eval::tz::backdated_episode_id(30),
            tags: Value::Object(tags),
            val_frac,
            budget,
            minibatch,
            seed_prompt,
            work,
            top_k,
            w_search,
            w_grep,
            w_glob,
            w_read,
            seed: rng_seed,
            pareto_size,
            candidate_selection,
        },
        searches,
        greps,
        globs,
        reads,
    )?;
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out, &result.prompt).with_context(|| format!("write {}", out.display()))?;
    println!(
        "wrote best prompt -> {} (run={run_label}, episode={}, search_acc={:.3}, grep_acc={:.3}, glob_acc={:.3}, read_acc={:.3})",
        out.display(),
        result.episode_id,
        result.search_acc,
        result.grep_acc,
        result.glob_acc,
        result.read_acc,
    );
    Ok(())
}

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

fn hits_json(hits: &[RankedHit]) -> Vec<Value> {
    hits.iter()
        .map(|h| {
            json!({
                "ord": h.ord, "path": h.path, "location": h.location,
                "file_type": h.file_type, "snippet": h.snippet,
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_dump(work: PathBuf, out: PathBuf, k: usize, poll_secs: u64, idle_stop: u32, once: bool) -> Result<()> {
    use std::io::Write as _;

    eprintln!("note: `dump` is legacy — prefer `export-tz` from ClickHouse episodes");

    let idx = DocIndex::open_or_create(&work).context("open index")?;
    let graph = GraphStore::open(&work).context("open graph")?;
    let ont = Ontology::load_or_default(&work);
    let structural: HashSet<String> = ont.structural().into_iter().collect();
    let analyzer = glossa::index::multilang::TermAnalyzer::new();

    std::fs::create_dir_all(&out).with_context(|| format!("create {}", out.display()))?;
    let search_path = out.join("search.jsonl");
    let select_path = out.join("select.jsonl");
    let mut search_f = std::fs::File::create(&search_path)?;
    let mut select_f = std::fs::File::create(&select_path)?;
    println!(
        "kb-train dump (legacy): {} -> {} / {}",
        work.display(),
        search_path.display(),
        select_path.display(),
    );

    let mut seen: HashSet<String> = HashSet::new();
    let (mut nodes_kept, mut search_written, mut select_written, mut recall_hit) = (0u64, 0u64, 0u64, 0u64);
    let mut idle = 0u32;

    loop {
        let nodes = graph.all_nodes().context("all_nodes")?;
        let mut fresh = 0u64;
        for node in &nodes {
            if structural.contains(&node.node_type) || !seen.insert(node.id.clone()) {
                continue;
            }
            fresh += 1;
            let mut gold: Vec<(String, String)> = Vec::new();
            for e in graph.outgoing(&node.id).unwrap_or_default() {
                if e.edge_type == glossa::graph::MENTIONS {
                    if let Some((p, l)) = e.to.split_once('#') {
                        gold.push((p.to_string(), l.to_string()));
                    }
                }
            }
            if gold.is_empty() {
                continue;
            }
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
                continue;
            }
            nodes_kept += 1;
            let gold_locs: Vec<String> = relevant.iter().map(|(p, l)| format!("{p}#{l}")).collect();
            writeln!(search_f, "{}", json!({"question": node.label, "gold": gold_locs}))?;
            search_written += 1;
            let hits = idx.search_filtered(&node.label, k, None, None).unwrap_or_default();
            let gold_set: HashSet<(String, String)> = relevant.into_iter().collect();
            let gold_ords: Vec<u64> =
                hits.iter().filter(|h| gold_set.contains(&(h.path.clone(), h.location.clone()))).map(|h| h.ord).collect();
            if !gold_ords.is_empty() {
                recall_hit += 1;
                writeln!(
                    select_f,
                    "{}",
                    json!({"question": node.label, "hits": hits_json(&hits), "gold_ords": gold_ords})
                )?;
                select_written += 1;
            }
        }
        if once {
            break;
        }
        idle = if fresh == 0 { idle + 1 } else { 0 };
        if idle >= idle_stop {
            break;
        }
        std::thread::sleep(Duration::from_secs(poll_secs));
    }
    search_f.flush().ok();
    select_f.flush().ok();
    println!(
        "DONE (legacy dump): kept {nodes_kept} nodes, search={search_written}, select={select_written}, recall@{k}={:.3}",
        recall_hit as f64 / nodes_kept.max(1) as f64,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hits_json_carries_ord_and_path() {
        let hits = vec![RankedHit {
            path: "doc.pdf".into(),
            location: "p.7".into(),
            file_type: "pdf".into(),
            ord: 7,
            snippet: "the answer".into(),
            score: 1.0,
        }];
        let j = hits_json(&hits);
        assert_eq!(j[0]["ord"], 7);
        assert!(j[0].get("score").is_none());
    }
}
