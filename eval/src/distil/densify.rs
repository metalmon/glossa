//! Task 2 — per-document densify pass: the STRONG `[distil]` model walks a document's chunks,
//! sees what the reasoning graph already has for each chunk (via [`existing_for_chunk`]), and
//! adds whatever is missing — grounded terminals the section states AND/OR the query-side
//! reasoning that leads to them — via `graph_upsert`, stamped `origin = "distil"`.
//!
//! Modelled closely on `build::extract::extract_doc`'s orchestrated, bounded-round coverage walk
//! (same `run_agent_loop` / `lmstudio_chat` shape, same per-round `on_progress` callback, same
//! `doc_chunk_ords` coverage-loop), with four differences (see the 2026-08-28 densification plan,
//! Task 2):
//!
//! 1. The strong model comes from `lab.distil` (not `lab.model`) — mirrors
//!    `distil::gen::generate_one`'s clear error when `[distil]` is unset.
//! 2. The `graph_upsert` tool is the SAME dual-kind pair `reason::chain_one_seed` uses —
//!    `reason::seed::SEED_GRAPH_UPSERT_DESC` + `build::extract::extract_tools_schema` — so the
//!    model may write BOTH `requires_grounding` terminals AND ungrounded query-side nodes.
//!    `filter_grounding_only` (build's terminals-only rule) is deliberately NOT applied here.
//! 3. Before each round, [`existing_for_chunk`] gathers the reasoning nodes already grounded to
//!    the round's starting section and lists their `type + label` in the user message, so the
//!    model adds only what is missing instead of re-deriving facts already in the graph.
//! 4. Writes go through `ops::graph_upsert(..., "distil")` — the Task-1 origin param — so every
//!    node/edge this pass produces is attributable to the densify stage, distinct from `"agent"`
//!    (build) and `"agent"` (reason; both currently share that origin string).
//!
//! `densify_doc`'s full agent loop needs a live endpoint, so it is exercised by a smoke run, not
//! a unit test; [`existing_for_chunk`] and the `graph_upsert` write body ([`densify_write`]) are
//! unit-testable and covered below without a live model (see `extract_doc`'s doc comment for the
//! same split).

use crate::backend::glossa_tools;
use crate::backend::openai::run_agent_loop;
use crate::backend::transport::openai::agent_chat_full;
use crate::build::extract::{
    doc_chunk_ords, extract_tools_schema, parse_and_filter_upsert, upserted_node_ids,
};
use crate::lab::LabConfig;
use crate::parallel::GraphWriter;
use crate::reason::schema_graph_block;
use crate::reason::seed::SEED_GRAPH_UPSERT_DESC;
use crate::workspace::KbxPaths;
use anyhow::anyhow;
use glossa::graph::ontology::Ontology;
use glossa::graph::ops;
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;
use glossa::read::DocImage;
use glossa::trace::TraceLog;
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::time::Duration;

/// Fallback cap on tool-call rounds for one document's densify pass, used when neither a CLI flag
/// nor `lab.toml`'s `[tuning] max_rounds` overrides it — mirrors `extract_doc`'s
/// `DEFAULT_MAX_ROUNDS`: generous for several reads plus multiple `graph_upsert` fan-out writes,
/// still bounded against a stuck model. The resolved value (CLI > lab > this default; see
/// `crate::lab::resolve`) is threaded in by the caller as `densify_doc`'s `max_rounds` parameter.
pub(crate) const DEFAULT_MAX_ROUNDS: usize = 30;

/// Sampling temperature for the densify pass (via `KB_EVAL_TEMP`, read by `lmstudio_chat`) —
/// same default `extract_doc`'s callers use for the build harvest (`BuildOpts::build_temp`
/// defaults to 0.8); `densify_doc`'s interface has no caller-supplied temperature knob, so a
/// sane constant stands in for the `build_temp` arg. `pub(crate)` so the caller
/// (`distil::run::run_densify_at`) can set `KB_EVAL_TEMP` from it ONCE, before the worker pool is
/// spawned, instead of `densify_doc` writing it per-worker.
pub(crate) const DENSIFY_TEMP: f64 = 0.8;

/// How much of one document's densify pass wrote: nodes upserted, non-`MENTIONS` edges upserted,
/// and how many of those edges were `MENTIONS` groundings (mirrors `reason::ReasonStats`'s shape,
/// which — unlike `build::extract::ExtractStats` — already separates edges from groundings; this
/// pass, like reason, may write both).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DensifyStats {
    pub nodes: usize,
    pub edges: usize,
    pub grounded: usize,
}

/// `(node_type, label)` of every reasoning node whose `MENTIONS` edge points at the Section node
/// for `<doc_path>#<ord>` — i.e. what the graph already has grounded to this one chunk. Resolves
/// `doc_path` through the index's canonical path (mirrors `ops::resolve_section_ref`'s own
/// lookup) before building the deterministic Section id (`graph::build::section_id`), then reads
/// `g.incoming` on that id and keeps only `MENTIONS` edges — same walk `src/tools/mod.rs`'s
/// `glossary` uses to surface reasoning nodes grounded to a matched section. Empty when the
/// chunk has nothing grounded to it yet (including when `doc_path` doesn't resolve at all).
pub(crate) fn existing_for_chunk(
    g: &GraphStore,
    idx: &DocIndex,
    doc_path: &str,
    ord: u64,
) -> Vec<(String, String)> {
    let path = idx
        .canonical_document_path(doc_path)
        .unwrap_or_else(|| doc_path.to_string());
    let sec_id = glossa::graph::build::section_id(&path, &ord.to_string());
    g.incoming(&sec_id)
        .unwrap_or_default()
        .into_iter()
        .filter(|e| e.edge_type == glossa::graph::MENTIONS)
        .filter_map(|e| g.get_node(&e.from).ok().flatten())
        .map(|n| (n.node_type, n.label))
        .collect()
}

/// Render [`existing_for_chunk`]'s output as the "what's already here" block injected into the
/// round's user message. Pure text formatting, split out so the message-building is unit-testable
/// without a graph.
fn existing_block(existing: &[(String, String)]) -> String {
    if existing.is_empty() {
        "(nothing grounded to this section yet)".to_string()
    } else {
        existing
            .iter()
            .map(|(t, l)| format!("- [{t}] {l}"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// The `graph_upsert` exec arm's write body, factored out so it is testable without a live model
/// or `run_agent_loop`: parse + partial-apply type-filter via [`parse_and_filter_upsert`] (NOT
/// `filter_grounding_only` — densify's whole point is to also write ungrounded query-side nodes),
/// then write through `writer.upsert` — which serializes this worker against every other worker
/// in the pool AND reuses the core file-lock (see `parallel::GraphWriter`) — stamped with the
/// Task-1 `"distil"` origin. Returns the raw `ops::UpsertOutcome` (for stats accounting at the
/// call site) plus the ids `run_agent_loop`'s unproductive-streak novelty tracker should see.
/// Errors (lock-busy/poisoned) propagate to the caller, which reports them to the model like any
/// other tool error rather than panicking one worker's whole document pass.
pub(crate) fn densify_write(
    idx: &DocIndex,
    writer: &GraphWriter,
    ont: &Ontology,
    args: &Value,
    now: u64,
) -> anyhow::Result<(ops::UpsertOutcome, Vec<String>, Vec<String>)> {
    let (nodes, edges, notes) = parse_and_filter_upsert(args, ont);
    let out = writer.upsert(idx, ont, nodes, edges, now, "distil")?;
    let ids = upserted_node_ids(&out);
    Ok((out, ids, notes))
}

/// Run the agentic densify pass over ONE document: the `lab.distil` strong model walks the
/// document in bounded coverage rounds (same shape as `extract_doc`), sees per-round what the
/// graph already has grounded to the round's starting section ([`existing_for_chunk`]), and adds
/// via `graph_upsert` whatever is missing — grounded terminals and/or query-side reasoning,
/// unlike `extract_doc`'s terminals-only harvest. Errors clearly if `[distil]` is unset.
/// `max_rounds` bounds EACH round's agent loop (resolved CLI > lab.toml `[tuning]` >
/// `DEFAULT_MAX_ROUNDS` by the caller — see `distil::run::run_densify_at`).
///
/// `writer` and `idx` are shared across every worker in the pool (opened ONCE by
/// `distil::run::run_densify_at`, not per-document): `idx` reads go straight through tantivy's
/// `IndexReader` (safe for concurrent readers, no lock needed); `writer.store()` reads serialize
/// on `GraphStore`'s own connection mutex against any in-flight write (correctness is fine either
/// way — a read mutates nothing; the pool's parallelism win is on the LLM round-trips, not DB
/// access). Every write funnels through `writer.upsert(..)`, which serializes the N worker
/// threads in-process AND reuses the core file-lock so a concurrent `glossa` MCP writer can never
/// interleave with an eval worker (mirrors `build::extract::extract_doc`/
/// `reason::seed::chain_one_seed`'s `GraphWriter` wiring).
#[allow(clippy::too_many_arguments)] // shared pool wiring; signature kept explicit
pub fn densify_doc(
    paths: &KbxPaths,
    ont: &Ontology,
    lab: &LabConfig,
    distil_md: &str,
    doc_path: &str,
    chunks_per_round: usize,
    max_rounds: usize,
    writer: &GraphWriter,
    idx: &DocIndex,
    // Called once per coverage round with the number of chunk ordinals newly covered that round
    // (mirrors `extract_doc`'s live-progress callback).
    mut on_progress: impl FnMut(u64),
) -> anyhow::Result<DensifyStats> {
    let distil_ep = lab
        .distil
        .as_ref()
        .ok_or_else(|| anyhow!("kbx distil needs a [distil] endpoint in lab.toml"))?;

    // lmstudio_chat reads the sampling temperature from KB_EVAL_TEMP — the caller
    // (`run_densify_at`) sets it ONCE, before the worker pool is spawned, so this fn only ever
    // reads it (concurrent `set_var` from N worker threads is UB; `env::var` is not).

    // `root` is still needed for the search/grep exec arm below (`glossa_tools::exec`) — reads
    // and writes themselves go through the shared `idx`/`writer`, not a fresh handle.
    let root = paths.root.as_path();
    let trace = TraceLog::disabled();
    let spec = glossa::tools::ChainSpec::from_ontology(ont);

    let system = format!("{}\n\n{distil_md}", schema_graph_block(ont));
    // The SAME dual-kind graph_upsert pair `reason::chain_one_seed` uses (search/read/grep +
    // graph_upsert, label-based, ontology-permitting) — allows query-side writes that build's
    // `BUILD_GRAPH_UPSERT_DESC`/`build_tools_schema` pair does not.
    let tools = extract_tools_schema(SEED_GRAPH_UPSERT_DESC);

    let endpoint = distil_ep.endpoint.clone();
    let model = distil_ep.model.clone();
    let api_key = distil_ep.resolve_key();
    let timeout = Duration::from_secs(distil_ep.timeout_secs);

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let ords = doc_chunk_ords(idx, doc_path)?;

    let stats = RefCell::new(DensifyStats::default());
    let reads = RefCell::new(Vec::<u64>::new());

    // Coverage loop: one bounded round per pass, starting at the first uncovered ordinal, until
    // every existing chunk ordinal has been visited — identical shape to `extract_doc`.
    let mut covered: BTreeSet<u64> = BTreeSet::new();
    while let Some(&start) = ords.iter().find(|o| !covered.contains(o)) {
        reads.borrow_mut().clear();

        let existing = existing_for_chunk(writer.store(), idx, doc_path, start);
        let user = format!(
            "Densify this document. Start at section {start}: call read(path=\"{doc_path}\", \
             n={start}) first.\n\nReasoning already grounded to this section:\n{}\n\nAdd only \
             what is missing: grounded terminals this section states, and/or the query-side \
             reasoning that leads to them, per the system prompt.",
            existing_block(&existing)
        );
        let messages = vec![
            json!({ "role": "system", "content": system.clone() }),
            json!({ "role": "user", "content": user }),
        ];

        // Full-response one-shot; resampling is applied provider-neutrally by the agent loop
        // (`backend::resample::call_with_resample`).
        let chat = |messages: &[Value]| {
            agent_chat_full(
                &endpoint,
                &model,
                api_key.as_deref(),
                &tools,
                messages,
                timeout,
            )
        };

        let exec = |name: &str, args: &Value| -> (String, Vec<String>, Vec<DocImage>) {
            if std::env::var("KB_EVAL_DUMP_TOOLS").is_ok() {
                eprintln!("[TOOL] {name} {args}");
            }
            if name == "graph_upsert" {
                let (out, ids, notes) = match densify_write(idx, writer, ont, args, now) {
                    Ok(v) => v,
                    // Lock-busy / poisoned-lock timeout: report to the model like any other tool
                    // error rather than panicking one worker's whole document pass.
                    Err(e) => return (format!("graph_upsert failed: {e}"), Vec::new(), Vec::new()),
                };
                if !out.rejected {
                    let mut s = stats.borrow_mut();
                    s.nodes += out.nodes;
                    // Split written edges into MENTIONS (grounded) vs everything else, mirroring
                    // how `extract_doc`/`chain_one_seed` count mentions from the outcome's dump.
                    let mentions_marker = format!("-{}->", glossa::graph::MENTIONS);
                    let grounded_this = out
                        .dump
                        .iter()
                        .filter(|l| l.starts_with("edge ") && l.contains(&mentions_marker))
                        .count();
                    s.grounded += grounded_this;
                    s.edges += out.edges.saturating_sub(grounded_this);
                }
                let message = if notes.is_empty() {
                    out.message
                } else {
                    format!("{}\n{}", notes.join("\n"), out.message)
                };
                (message, ids, Vec::new())
            } else if name == "read" {
                // Bounded per-round document read — the coverage-driving tool, same as
                // `extract_doc`'s exec arm.
                let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(0);
                if reads.borrow().len() >= chunks_per_round {
                    return (
                        "(no more sections for this round — now call graph_upsert for every \
                         missing node you found, then stop)"
                            .into(),
                        vec![],
                        Vec::new(),
                    );
                }
                match idx.read_chunk_by_ord(doc_path, n) {
                    Ok(Some(c)) => {
                        reads.borrow_mut().push(n);
                        (c.body, vec![], Vec::new())
                    }
                    Ok(None) => ("(no more sections)".into(), vec![], Vec::new()),
                    Err(e) => (format!("(read error: {e})"), vec![], Vec::new()),
                }
            } else {
                // search/grep — unbounded exploration for existing context, delegated to the
                // shared registry (same as `reason::chain_one_seed`/`distil::gen::generate_one`).
                let (body, ids, _images) =
                    glossa_tools::exec(name, args, root, idx, Some(writer.store()), &spec, &trace);
                (body, ids, Vec::new())
            }
        };

        let on_repeat = |name: &str, _args: &Value| {
            format!(
                "(dup {name}) you already called this — try a different tool, a different query, \
                 or move on to graph_upsert"
            )
        };

        run_agent_loop(chat, messages, exec, on_repeat, max_rounds, None)?;

        // Advance coverage monotonically (identical to `extract_doc`): `start` is unconditionally
        // marked covered after its round, plus whatever ordinals the round actually read.
        let before = covered.len();
        covered.insert(start);
        covered.extend(reads.borrow().iter().copied());
        on_progress((covered.len() - before) as u64);
    }

    Ok(stats.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glossa::graph::store::{Edge, GraphStore, Node, Provenance};
    use glossa::model::Chunk;

    /// Test ontology with BOTH a `requires_grounding` terminal type (`Fact`) and a plain
    /// query-side type (`Task`) that does not — so a write can validly contain one of each,
    /// exercising densify's "allow query-side, not just grounded terminals" behaviour.
    const DUAL_ONT: &str = r#"
[entities.Fact]
requires_grounding = true
[entities.Task]
[relations.LEADS_TO]
from = ["Task"]
to = ["Fact"]
[validation]
strict = true
"#;

    fn dual_ontology() -> Ontology {
        Ontology::parse(DUAL_ONT).expect("dual test ontology parses")
    }

    fn prov() -> Provenance {
        Provenance {
            source_path: "d.md".into(),
            range: None,
            file_sig: None,
            origin: "test".into(),
            confidence: 0.9,
            created_at: 1,
        }
    }

    fn write_doc_chunk(idx: &DocIndex, path: &str, location: &str) {
        idx.write_chunks(&[Chunk {
            doc_path: path.into(),
            location: location.into(),
            file_type: "md".into(),
            text: "stub content".into(),
        }])
        .unwrap();
    }

    // --- existing_for_chunk ---------------------------------------------------------------

    /// A `Fact` node MENTIONS-ing `docA.md#1` is returned as its `(type, label)` for that chunk;
    /// a bare, ungrounded `docB.md#1` (no node mentions it) returns empty.
    #[test]
    fn existing_for_chunk_returns_mentioning_nodes_for_the_right_chunk_only() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        write_doc_chunk(&idx, "docA.md", "S1");
        write_doc_chunk(&idx, "docB.md", "S1");

        g.put_node(&Node {
            id: "fact:a1".into(),
            node_type: "Fact".into(),
            label: "docA fact one".into(),
            aliases: vec![],
            prov: prov(),
        })
        .unwrap();
        g.put_edge(&Edge {
            from: "fact:a1".into(),
            to: "docA.md#1".into(),
            edge_type: glossa::graph::MENTIONS.to_string(),
            prov: prov(),
        })
        .unwrap();

        let a = existing_for_chunk(&g, &idx, "docA.md", 1);
        assert_eq!(a, vec![("Fact".to_string(), "docA fact one".to_string())]);

        let b = existing_for_chunk(&g, &idx, "docB.md", 1);
        assert!(
            b.is_empty(),
            "bare chunk with no MENTIONS must return empty: {b:?}"
        );
    }

    /// A doc_path that isn't indexed at all (no canonical resolution) must not panic — it just
    /// falls back to the raw string and finds nothing grounded.
    #[test]
    fn existing_for_chunk_empty_for_unindexed_doc() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(existing_for_chunk(&g, &idx, "nowhere.md", 1).is_empty());
    }

    #[test]
    fn existing_block_formats_empty_and_nonempty() {
        assert!(existing_block(&[]).contains("nothing grounded"));
        let block = existing_block(&[("Fact".to_string(), "x".to_string())]);
        assert!(block.contains("[Fact] x"), "{block}");
    }

    // --- densify_write: allows query-side + stamps distil origin --------------------------

    /// The core behavioural difference from build's harvest: a batch with BOTH a grounded
    /// `Fact` terminal and an ungrounded query-side `Task` node writes BOTH (build's
    /// `filter_grounding_only` would drop the `Task`), and every written node carries
    /// `prov.origin == "distil"` (the Task-1 origin param), not `"agent"`.
    #[test]
    fn densify_write_keeps_query_side_node_and_stamps_distil_origin() {
        let dir = tempfile::tempdir().unwrap();
        let ont = dual_ontology();
        let g = std::sync::Arc::new(GraphStore::open(dir.path()).unwrap());
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        write_doc_chunk(&idx, "d.md", "S1");
        let writer = GraphWriter::new(std::sync::Arc::clone(&g), dir.path().to_path_buf());

        let args = serde_json::json!({
            "nodes": [
                {"node_type": "Fact", "label": "grounded fact", "source_path": "d.md#1"},
                {"node_type": "Task", "label": "ungrounded task"}
            ],
            "edges": [
                {"from": "ungrounded task", "to": "grounded fact", "edge_type": "LEADS_TO", "source_path": "d.md#1"}
            ]
        });

        let (out, ids, notes) = densify_write(&idx, &writer, &ont, &args, 1_000).unwrap();
        assert!(!out.rejected, "batch must not be rejected: {}", out.message);
        assert!(notes.is_empty(), "no node should be filtered: {notes:?}");
        assert_eq!(
            out.nodes, 2,
            "both the grounded terminal AND the query-side node must write"
        );
        assert_eq!(
            ids.len(),
            2,
            "both written node ids must surface for the novelty tracker: {ids:?}"
        );

        let fact_id = ops::id_for(&ont, "Fact", "grounded fact");
        let task_id = ops::id_for(&ont, "Task", "ungrounded task");

        let fact_node = g
            .get_node(&fact_id)
            .unwrap()
            .expect("grounded Fact must be in the graph");
        assert_eq!(fact_node.prov.origin, "distil");

        let task_node = g
            .get_node(&task_id)
            .unwrap()
            .expect("query-side Task must be in the graph, NOT dropped like build would");
        assert_eq!(task_node.prov.origin, "distil");

        // The LEADS_TO edge from the query-side node to the terminal must also have landed.
        let out_edges = g.outgoing(&task_id).unwrap();
        assert!(
            out_edges
                .iter()
                .any(|e| e.edge_type == "LEADS_TO" && e.to == fact_id),
            "query-side -> terminal edge must write: {out_edges:?}"
        );
    }

    /// `densify_write` must reuse the canonical `ops::graph_upsert` auto-grounding, exactly like
    /// `extract_doc`/`chain_one_seed`: a `Fact` given only a `source_path` (no explicit MENTIONS)
    /// still gets grounded automatically.
    #[test]
    fn densify_write_auto_grounds_terminal_from_source_path() {
        let dir = tempfile::tempdir().unwrap();
        let ont = dual_ontology();
        let g = std::sync::Arc::new(GraphStore::open(dir.path()).unwrap());
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        write_doc_chunk(&idx, "d.md", "S1");
        let writer = GraphWriter::new(std::sync::Arc::clone(&g), dir.path().to_path_buf());

        let args = serde_json::json!({
            "nodes": [{"node_type": "Fact", "label": "auto grounded", "source_path": "d.md#1"}],
            "edges": []
        });
        let (out, _ids, _notes) = densify_write(&idx, &writer, &ont, &args, 1_000).unwrap();
        assert!(!out.rejected, "{}", out.message);
        assert_eq!(
            out.edges, 1,
            "auto-derived MENTIONS edge must be written: {}",
            out.message
        );

        let id = ops::id_for(&ont, "Fact", "auto grounded");
        let mentions: Vec<_> = g
            .outgoing(&id)
            .unwrap()
            .into_iter()
            .filter(|e| e.edge_type == glossa::graph::MENTIONS)
            .collect();
        assert_eq!(mentions.len(), 1, "{mentions:?}");
        assert_eq!(mentions[0].to, "d.md#1");
    }
}
