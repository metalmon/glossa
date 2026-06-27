//! Enricher runner: for each solved training case, drive the `enrich` TZ function
//! to reverse-trace the answer into reasoning-graph nodes/edges.
//!
//! Design: we build a case-local exec closure that handles `graph_upsert` in-process
//! (parse → ontology-validate → apply_upsert) and delegates every other tool to
//! `glossa_tools::exec`. This keeps the shared exec signature untouched.

use anyhow::Context;
use glossa::graph::agent::{EdgeRef, NodeUpdate};
use glossa::graph::ops::{UpsertEdge, UpsertNode};
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
        let errors_total = Arc::new(AtomicUsize::new(0)); // upserts rejected by the strict ontology
        let rounds_total = Arc::new(AtomicUsize::new(0)); // /inference calls (episode depth)
        let nc = Arc::clone(&nodes_total);
        let ec = Arc::clone(&edges_total);
        let erc = Arc::clone(&errors_total);
        let rc = Arc::clone(&rounds_total);

        // Enrich-specific exec: handles graph_upsert locally, delegates the rest.
        let exec = move |name: &str, args: &Value| -> (String, Vec<String>, Vec<glossa::read::DocImage>) {
            if name == "graph_upsert" {
                // Parse element-wise so one malformed node/edge doesn't silently drop the rest:
                // a blanket `unwrap_or_default()` would yield an empty vec on any error and report a
                // fake "upserted 0" success, losing the model's valid items without telling it.
                let mut parse_errs: Vec<String> = Vec::new();
                let mut nodes: Vec<UpsertNode> = Vec::new();
                for (i, n) in args.get("nodes").and_then(|v| v.as_array()).cloned().unwrap_or_default().iter().enumerate() {
                    match serde_json::from_value::<UpsertNode>(n.clone()) {
                        Ok(un) => nodes.push(un),
                        Err(e) => parse_errs.push(format!("node[{i}]: {e}")),
                    }
                }
                let mut edges: Vec<UpsertEdge> = Vec::new();
                for (i, e) in args.get("edges").and_then(|v| v.as_array()).cloned().unwrap_or_default().iter().enumerate() {
                    match serde_json::from_value::<UpsertEdge>(e.clone()) {
                        Ok(ue) => edges.push(ue),
                        Err(err) => parse_errs.push(format!("edge[{i}]: {err}")),
                    }
                }
                if !parse_errs.is_empty() {
                    erc.fetch_add(1, Ordering::Relaxed);
                    let msg = format!(
                        "graph_upsert REJECTED — malformed input, fix and resend:\n- {}",
                        parse_errs.join("\n- ")
                    );
                    eprintln!("    \u{2717} {}", msg.replace('\n', "; "));
                    return (msg, vec![], vec![]);
                }
                let ont = Ontology::load_or_default(&work_iter);
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let out = glossa::graph::ops::graph_upsert(&idx, &graph, &ont, nodes, edges, now);
                for l in &out.dump {
                    eprintln!("    {l}");
                }
                if out.rejected {
                    erc.fetch_add(1, Ordering::Relaxed);
                } else {
                    nc.fetch_add(out.nodes, Ordering::Relaxed);
                    ec.fetch_add(out.edges, Ordering::Relaxed);
                }
                (out.message, vec![], vec![])
            } else if name == "graph_delete" {
                #[derive(serde::Deserialize)]
                struct DeleteEdgeArg {
                    from: String,
                    edge_type: String,
                    to: String,
                }
                let node_labels: Vec<String> = serde_json::from_value(
                    args.get("nodes").cloned().unwrap_or(json!([])),
                )
                .unwrap_or_default();
                let edge_args: Vec<DeleteEdgeArg> = serde_json::from_value(
                    args.get("edges").cloned().unwrap_or(json!([])),
                )
                .unwrap_or_default();
                let edge_refs: Vec<EdgeRef> = edge_args
                    .into_iter()
                    .map(|e| EdgeRef { from: e.from, edge_type: e.edge_type, to: e.to })
                    .collect();
                let msg = glossa::graph::ops::graph_delete(&graph, node_labels, edge_refs);
                (msg, vec![], vec![])
            } else if name == "graph_update" {
                #[derive(serde::Deserialize)]
                struct UpdateNodeArg {
                    label: String,
                    new_label: Option<String>,
                    new_type: Option<String>,
                }
                let update_args: Vec<UpdateNodeArg> = serde_json::from_value(
                    args.get("nodes").cloned().unwrap_or(json!([])),
                )
                .unwrap_or_default();
                let ups: Vec<NodeUpdate> = update_args
                    .into_iter()
                    .map(|u| NodeUpdate { label: u.label, new_label: u.new_label, new_type: u.new_type })
                    .collect();
                let msg = glossa::graph::ops::graph_update(&graph, ups);
                (msg, vec![], vec![])
            } else {
                glossa_tools::exec(name, args, &idx, Some(&graph), &trace)
            }
        };

        // Per-case episode id (groups all the case's inferences into one TZ episode).
        let eid = uuid::Uuid::now_v7().to_string();
        let eid_fb = eid.clone(); // kept for posting per-case feedback after the episode
        let fn_clone = function_name.clone();
        let url_clone = url.clone();

        let chat = move |messages: &[Value], _episode_id: Option<&str>| -> anyhow::Result<TzTurn> {
            rc.fetch_add(1, Ordering::Relaxed);
            let body = json!({
                "function_name": fn_clone,
                "input": { "messages": messages },
                "episode_id": eid
            });
            let payload = serde_json::to_string(&body)?;
            // Retry transient gateway failures (5xx, timeouts, dropped connections — e.g. the
            // gateway being restarted by another task) so one blip doesn't kill a long enrich pass.
            let mut attempt = 0u32;
            let resp = loop {
                match ureq::post(&url_clone)
                    .timeout(std::time::Duration::from_secs(180))
                    .set("Content-Type", "application/json")
                    .send_string(&payload)
                {
                    Ok(r) => break r,
                    Err(e) => {
                        let retryable = match &e {
                            ureq::Error::Status(code, _) => *code >= 500,
                            ureq::Error::Transport(_) => true,
                        };
                        attempt += 1;
                        if retryable && attempt <= 4 {
                            std::thread::sleep(std::time::Duration::from_millis(800 * u64::from(attempt)));
                            continue;
                        }
                        return Err(anyhow::anyhow!("tensorzero /inference: {e}"));
                    }
                }
            };
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

        // Best-effort per case: a single case failing must not abort the whole enrich pass.
        match run_episode(chat, &user, exec, MAX_ROUNDS) {
            Ok(_) => println!(
                "  → nodes={} edges={}",
                nodes_total.load(Ordering::Relaxed),
                edges_total.load(Ordering::Relaxed),
            ),
            Err(e) => eprintln!("  ! {} failed (skipped): {e:#}", case._id),
        }

        // Per-case telemetry → TZ /feedback (grouped by the case's episode), so enricher
        // productivity + conformance is visible in the UI. Best-effort; never fail on feedback.
        let fb_url = format!("{}/feedback", endpoint.trim_end_matches('/'));
        let post_fb = |metric: &str, value: f64| {
            let body = json!({ "episode_id": eid_fb, "metric_name": metric, "value": value });
            let _ = ureq::post(&fb_url)
                .timeout(std::time::Duration::from_secs(30))
                .set("Content-Type", "application/json")
                .send_string(&serde_json::to_string(&body).unwrap_or_default());
        };
        post_fb("enrich_nodes", nodes_total.load(Ordering::Relaxed) as f64);
        post_fb("enrich_edges", edges_total.load(Ordering::Relaxed) as f64);
        post_fb("enrich_errors", errors_total.load(Ordering::Relaxed) as f64);
        post_fb("enrich_rounds", rounds_total.load(Ordering::Relaxed) as f64);
    }

    Ok(())
}
