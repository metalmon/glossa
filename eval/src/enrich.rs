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

/// Resolve a section reference of the form `<path>#<n>` (a chunk number the agent has from a
/// read/search result) into the real Section node id `<path>#<location>`. Anything else (a
/// reasoning-node slug like `sym:…`, or an already-resolved `<path>#<location>`) passes through.
fn resolve_section_ref(idx: &DocIndex, s: &str) -> String {
    if let Some(pos) = s.rfind('#') {
        if let Ok(n) = s[pos + 1..].parse::<u64>() {
            let path = &s[..pos];
            if let Ok(Some(loc)) = idx.location_for_ord(path, n) {
                return glossa::graph::build::section_id(path, &loc);
            }
        }
    }
    s.to_string()
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
                let nodes: Vec<NodeSpec> = serde_json::from_value(
                    args.get("nodes").cloned().unwrap_or(json!([])),
                )
                .unwrap_or_default();
                let mut edges: Vec<EdgeSpec> = serde_json::from_value(
                    args.get("edges").cloned().unwrap_or(json!([])),
                )
                .unwrap_or_default();
                // Resolve section refs the agent gives as `<path>#<n>` (the (path,#n) it has from
                // reads) into the real Section node id `<path>#<location>` — it cannot reconstruct
                // the heading breadcrumb itself. Reasoning-node ids (sym:/res:/…) pass through.
                for e in &mut edges {
                    e.from = resolve_section_ref(&idx, &e.from);
                    e.to = resolve_section_ref(&idx, &e.to);
                }
                // Dump what the model proposed so the controller can eyeball graph quality.
                for nd in &nodes {
                    eprintln!("    node {} [{}] {}", nd.id, nd.node_type, nd.label);
                }
                for e in &edges {
                    eprintln!("    edge {} -{}-> {}", e.from, e.edge_type, e.to);
                }
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
                    Err(err) => {
                        erc.fetch_add(1, Ordering::Relaxed);
                        (err.to_string(), vec![], vec![])
                    }
                }
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
