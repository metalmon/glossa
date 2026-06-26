//! Enricher runner: for each solved training case, drive the `enrich` TZ function
//! to reverse-trace the answer into reasoning-graph nodes/edges.
//!
//! Design: we build a case-local exec closure that handles `graph_upsert` in-process
//! (parse → ontology-validate → apply_upsert) and delegates every other tool to
//! `glossa_tools::exec`. This keeps the shared exec signature untouched.

use anyhow::Context;
use glossa::graph::agent::{apply_upsert, EdgeSpec, NodeSpec};
use glossa::graph::ontology::Ontology;
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;
use glossa::trace::TraceLog;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use crate::backend::glossa_tools;
use crate::backend::tensorzero::{run_episode, TzTurn};

// High cap — identical to tensorzero.rs MAX_ROUNDS (which is private).
const MAX_ROUNDS: usize = 50;

#[derive(Debug, Deserialize)]
struct TrainCase {
    #[serde(rename = "_id")]
    _id: String,
    question: String,
    answer: String,
}

pub fn run_enrich(
    train: &Path,
    work: &Path,
    limit: usize,
    endpoint: &str,
    function_name: &str,
) -> anyhow::Result<()> {
    let raw = std::fs::read_to_string(train).context("read train JSON")?;
    let mut cases: Vec<TrainCase> =
        serde_json::from_str(&raw).context("parse train JSON")?;

    if limit > 0 && cases.len() > limit {
        cases.truncate(limit);
    }

    let url = format!("{}/inference", endpoint.trim_end_matches('/'));
    let function_name = function_name.to_string();
    let work_buf: PathBuf = work.to_path_buf();

    for (i, case) in cases.iter().enumerate() {
        println!(
            "[{}/{}] enriching: {}",
            i + 1,
            cases.len(),
            &case._id
        );

        // Clone per iteration so the move-captured exec closure doesn't consume the outer buf.
        let work_iter = work_buf.clone();
        let idx = DocIndex::open_or_create(&work_iter)?;
        let graph = GraphStore::open(&work_iter)
            .context("open graph store")?;
        let trace = TraceLog::to_dir(&work_iter);

        // Shared atomic counters so the exec closure (called concurrently for
        // parallel tool calls) can accumulate upsert counts safely.
        let nodes_total = Arc::new(AtomicUsize::new(0));
        let edges_total = Arc::new(AtomicUsize::new(0));
        let nc = Arc::clone(&nodes_total);
        let ec = Arc::clone(&edges_total);

        // Enrich-specific exec: handles graph_upsert locally, delegates the rest.
        let exec = move |name: &str, args: &Value| -> (String, Vec<String>, Vec<glossa::read::DocImage>) {
            if name == "graph_upsert" {
                let nodes: Vec<NodeSpec> = serde_json::from_value(
                    args.get("nodes").cloned().unwrap_or(json!([])),
                )
                .unwrap_or_default();
                let edges: Vec<EdgeSpec> = serde_json::from_value(
                    args.get("edges").cloned().unwrap_or(json!([])),
                )
                .unwrap_or_default();
                let ont = Ontology::load_or_default(&work_iter);
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                match apply_upsert(&graph, &ont, nodes, edges, now) {
                    Ok((n, e)) => {
                        nc.fetch_add(n, Ordering::Relaxed);
                        ec.fetch_add(e, Ordering::Relaxed);
                        (format!("upserted {n} nodes, {e} edges"), vec![], vec![])
                    }
                    Err(err) => (err.to_string(), vec![], vec![]),
                }
            } else {
                glossa_tools::exec(name, args, &idx, Some(&graph), &trace)
            }
        };

        // Per-case episode id (no backdating — telemetry only, no feedback loop here).
        let eid = uuid::Uuid::now_v7().to_string();
        let fn_clone = function_name.clone();
        let url_clone = url.clone();

        let chat = move |messages: &[Value], _episode_id: Option<&str>| -> anyhow::Result<TzTurn> {
            let body = json!({
                "function_name": fn_clone,
                "input": { "messages": messages },
                "episode_id": eid
            });
            let payload = serde_json::to_string(&body)?;
            let resp = ureq::post(&url_clone)
                .set("Content-Type", "application/json")
                .send_string(&payload)
                .map_err(|e| anyhow::anyhow!("tensorzero /inference: {e}"))?;
            let text = resp.into_string().context("read /inference response")?;
            let v: Value = serde_json::from_str(&text).context("parse /inference json")?;
            if let Some(err) = v.get("error") {
                anyhow::bail!("tensorzero error: {err}");
            }
            let episode_id = v
                .get("episode_id")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let content = v
                .get("content")
                .and_then(|c| c.as_array())
                .cloned()
                .unwrap_or_default();
            Ok(TzTurn { content, episode_id })
        };

        let user = format!(
            "Question: {}\nKnown correct answer: {}\nBuild the reusable reasoning graph for this case.",
            case.question, case.answer
        );

        let _outcome = run_episode(chat, &user, exec, MAX_ROUNDS)
            .with_context(|| format!("run_episode for {}", case._id))?;

        println!(
            "  → nodes={} edges={}",
            nodes_total.load(Ordering::Relaxed),
            edges_total.load(Ordering::Relaxed),
        );
    }

    Ok(())
}
