//! Phase-2 (`kbx reason` seed-from-grounded) engine: read a grounded terminal's FULL source,
//! backward-synthesize the query-side reasoning layer (fan-out), and write it via the canonical
//! write triad. Sibling to `chain.rs` (gold-anchored, migrating to `kbx train`).

use crate::backend::glossa_tools;
use crate::backend::openai::{lmstudio_chat, run_agent_loop};
use crate::build::extract::{extract_tools_schema, parse_and_filter_upsert, upserted_node_ids};
use crate::distil::Seed;
use crate::lab::LabConfig;
use crate::reason::schema_graph_block;
use crate::reason::ReasonStats;
use crate::workspace::KbxPaths;
use anyhow::anyhow;
use glossa::graph::ontology::Ontology;
use glossa::graph::ops;
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;
use glossa::trace::TraceLog;
use serde_json::{json, Value};
use std::time::Duration;

/// High cap on tool-call rounds for one seed's backward-synth pass — matches `chain_one_gold`'s
/// `MAX_ROUNDS` (generous for several reads + multiple fan-out `graph_upsert` writes, still bounded).
const MAX_ROUNDS: usize = 30;

/// `graph_upsert` description for phase-2 seed-from-grounded synthesis. Ontology-general (no domain
/// type names): the model is handed a grounded TERMINAL and told to synthesize the query-side
/// predecessors backward along the ontology's relations, fan-out. Grounding stays governed by each
/// type's `requires_grounding` (soft-recommended otherwise) — same discipline as `chain_one_gold`.
const SEED_GRAPH_UPSERT_DESC: &str =
    "Write the query-side reasoning nodes/edges that lead BACKWARD to this already-grounded \
     terminal node. Using the ontology's schema-graph above, synthesize the predecessor nodes \
     (the intermediate and entry types whose relations point toward this terminal's type) and \
     connect them with the ontology's declared relations, respecting each relation's from/to \
     types. Let the source text decide how much there is: where it supports more than one distinct \
     problem or task this terminal answers, give each its own predecessor; where it supports just \
     one, one is right; and where the source does not describe any query-side situation for this \
     terminal at all, writing nothing is the correct, honest outcome — never add a predecessor to \
     reach a count or because the schema-graph would allow it. Each node needs a `node_type` from \
     the ontology's declared entity types (any other value drops just that node), a `label` phrased \
     as a user would express it, and `aliases` listing alternate phrasings. Reference an edge \
     endpoint either by the exact LABEL of a node you create in this same call, or by the node ID \
     of a node that already exists in the graph. In particular, the grounded terminal you were \
     given ALREADY exists — do NOT create another node for it; point your chaining edges at it by \
     ITS id (the `res:`/type-prefixed id shown for the terminal). To ground a node, set its \
     `source_path` to the copy-ready `path#n` token \
     exactly as a search/read/grep result showed it — the same token you would pass to `read`. A \
     node you are NOT grounding — any query-side or entry node whose type is not marked \
     `[requires_grounding]` — must carry NO `source_path` at all: omit the field entirely, and \
     never invent or borrow a path to fill it; an ungrounded entry node is the correct, expected \
     outcome. Emit a node only when the source genuinely supports it as a real reasoning step; do \
     not invent filler.";

/// Build the per-seed user message: introduce the grounded terminal (id/type/label), give the model
/// its FULL grounded source text, and instruct backward fan-out synthesis bounded by `fanout_max`.
/// Pure (no I/O) so it is unit-testable. Ontology-general — names no concrete types.
pub(crate) fn build_seed_user_message(seed: &Seed, source_text: &str, fanout_max: usize) -> String {
    let body = if source_text.is_empty() {
        "(no grounded source text found for this terminal)".to_string()
    } else {
        source_text.to_string()
    };
    format!(
        "Grounded terminal node: {} [{}] \"{}\"\nIts grounded source text:\n{}\n\nSynthesize the \
         query-side reasoning layer that leads to this terminal: walk the ontology's schema-graph \
         BACKWARD, emitting the predecessor nodes and relations a new user question would traverse \
         to reach it. This terminal ALREADY exists as node id `{}` — do NOT create another node for \
         it; attach your chain to it by pointing your chaining edge(s) TO that id. Let the source \
         text set the shape: where it describes more than one distinct situation a user could \
         arrive from, give each its own path (at most {} per step); where it describes one, one is \
         enough; where it does not describe the query side at all — first check whether the source \
         even matches this terminal — it is correct to write nothing rather than invent. Call \
         `graph_upsert` for each node and edge the source genuinely supports.",
        seed.id, seed.node_type, seed.label, body, seed.id, fanout_max
    )
}

/// The grounded source text for EVERY outgoing `MENTIONS` edge of `seed_id`, each target chunk
/// read and concatenated in sorted-ref order (deduped), separated by blank lines. Empty string if
/// the seed has no groundings or none resolve to readable chunks. Unlike
/// `distil::gen::seed_source_text` (first edge only), phase-2 needs the whole resolution — it may
/// span several chunks and the query-side (problem, aliases, task) must be inferred from all of them.
pub(crate) fn seed_source_text_all(
    root: &std::path::Path,
    idx: &DocIndex,
    g: &GraphStore,
    seed_id: &str,
) -> String {
    let mut targets: Vec<String> = g
        .outgoing(seed_id)
        .unwrap_or_default()
        .into_iter()
        .filter(|e| e.edge_type == glossa::graph::MENTIONS)
        .map(|e| e.to)
        .collect();
    // Sort by (path, numeric chunk n), NOT the raw `<path>#n` string — lexicographic order would
    // put `d.md#10` before `d.md#2`, scrambling multi-chunk resolutions out of section order.
    targets.sort_by_key(|t| match t.rsplit_once('#') {
        Some((p, n)) => (p.to_string(), n.parse::<u64>().unwrap_or(0)),
        None => (t.clone(), 0),
    });
    targets.dedup();

    let trace = TraceLog::disabled();
    let mut out = String::new();
    for target in targets {
        let (path, n) = match target.rsplit_once('#') {
            Some((p, n)) => (p.to_string(), n.parse::<u64>().unwrap_or(0)),
            None => (target, 0),
        };
        let (text, _images) = glossa_tools::run_read(root, idx, None, &path, n, false, &trace);
        if !text.is_empty() {
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            out.push_str(&text);
        }
    }
    out
}

/// Run the agentic backward-synth pass for ONE grounded seed terminal: the `lab.[reason]` model
/// (Task 1) reads the corpus (unscoped readers) and calls `graph_upsert` to synthesize the
/// query-side predecessor nodes/relations that lead to the seed, fan-out to `fanout_max`. Writes
/// through the CANONICAL triad — `extract_tools_schema(SEED_GRAPH_UPSERT_DESC)` ->
/// `parse_and_filter_upsert` (ontology-permitting, partial-apply) -> `ops::graph_upsert`
/// (auto-grounds from a `<path>#n` source_path). Returns nodes/edges/groundings written; errors if
/// no reason/distil endpoint is configured.
pub fn chain_one_seed(
    paths: &KbxPaths,
    ont: &Ontology,
    lab: &LabConfig,
    reason_md: &str,
    seed: &Seed,
    fanout_max: usize,
) -> anyhow::Result<ReasonStats> {
    let ep = lab
        .reason_endpoint()
        .ok_or_else(|| anyhow!("kbx reason needs a [reason] (or [distil]) endpoint in lab.toml"))?;

    let root = paths.root.as_path();
    let g = GraphStore::open(root)?;
    let idx = DocIndex::open_or_create(root)?;
    let trace = TraceLog::disabled();
    let spec = glossa::tools::ChainSpec::from_ontology(ont);

    let source_text = seed_source_text_all(root, &idx, &g, &seed.id);
    let system = format!("{}\n\n{reason_md}", schema_graph_block(ont));
    let user = build_seed_user_message(seed, &source_text, fanout_max);
    let messages = vec![
        json!({ "role": "system", "content": system }),
        json!({ "role": "user", "content": user }),
    ];

    let endpoint = ep.endpoint.clone();
    let model = ep.model.clone();
    let api_key = ep.resolve_key();
    let timeout = Duration::from_secs(ep.timeout_secs);
    let tools = extract_tools_schema(SEED_GRAPH_UPSERT_DESC);

    let chat = |messages: &[Value]| {
        lmstudio_chat(&endpoint, &model, api_key.as_deref(), &tools, messages, timeout)
    };

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut stats = ReasonStats::default();

    let exec = |name: &str, args: &Value| -> (String, Vec<String>, Vec<glossa::read::DocImage>) {
        if name == "graph_upsert" {
            let (nodes, edges, notes) = parse_and_filter_upsert(args, ont);
            let out = ops::graph_upsert(&idx, &g, ont, nodes, edges, now);
            if !out.rejected {
                stats.nodes += out.nodes;
                stats.edges += out.edges;
                let mentions_marker = format!("-{}->", glossa::graph::MENTIONS);
                stats.grounded += out
                    .dump
                    .iter()
                    .filter(|l| l.starts_with("edge ") && l.contains(&mentions_marker))
                    .count();
            }
            let ids = upserted_node_ids(&out);
            let message = if notes.is_empty() {
                out.message
            } else {
                format!("{}\n{}", notes.join("\n"), out.message)
            };
            (message, ids, Vec::new())
        } else {
            let (body, ids, _images) =
                glossa_tools::exec(name, args, root, &idx, Some(&g), &spec, &trace);
            (body, ids, Vec::new())
        }
    };

    let on_repeat = |name: &str, _args: &Value| {
        format!(
            "(dup {name}) you already called this — try a different tool, a different query, \
             or move on to graph_upsert"
        )
    };

    run_agent_loop(chat, messages, exec, on_repeat, MAX_ROUNDS)?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glossa::graph::store::{Edge, GraphStore, Node, Provenance};
    use glossa::index::store::DocIndex;

    #[test]
    fn build_seed_user_message_names_seed_source_and_fanout() {
        let seed = crate::distil::Seed {
            id: "res-1".into(), node_type: "Resolution".into(), label: "do the thing".into() };
        let msg = build_seed_user_message(&seed, "grounded body text", 3);
        assert!(msg.contains("res-1") && msg.contains("Resolution") && msg.contains("do the thing"),
            "seed identity must appear: {msg}");
        assert!(msg.contains("grounded body text"), "source text must appear: {msg}");
        assert!(msg.contains('3'), "fan-out cap must appear: {msg}");
    }

    #[test]
    fn build_seed_user_message_handles_empty_source() {
        let seed = crate::distil::Seed {
            id: "res-2".into(), node_type: "Resolution".into(), label: "x".into() };
        let msg = build_seed_user_message(&seed, "", 2);
        assert!(msg.to_lowercase().contains("no grounded source"),
            "empty source must be flagged, not left blank: {msg}");
    }

    fn prov() -> Provenance {
        Provenance { source_path: "d.md".into(), range: None, file_sig: None,
            origin: "test".into(), confidence: 0.9, created_at: 1 }
    }

    #[test]
    fn seed_source_text_all_concatenates_every_grounding_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            glossa::model::Chunk { doc_path: "d.md".into(), location: "S1".into(),
                file_type: "md".into(), text: "FIRST chunk body".into() },
            glossa::model::Chunk { doc_path: "d.md".into(), location: "S2".into(),
                file_type: "md".into(), text: "SECOND chunk body".into() },
        ]).unwrap();

        g.put_node(&Node { id: "res-1".into(), node_type: "Resolution".into(),
            label: "r".into(), aliases: vec![], prov: prov() }).unwrap();
        for n in [1u64, 2] {
            g.put_edge(&Edge { from: "res-1".into(), to: format!("d.md#{n}"),
                edge_type: glossa::graph::MENTIONS.to_string(), prov: prov() }).unwrap();
        }

        let text = seed_source_text_all(dir.path(), &idx, &g, "res-1");
        assert!(text.contains("FIRST chunk body"), "missing chunk 1: {text}");
        assert!(text.contains("SECOND chunk body"), "missing chunk 2: {text}");
    }

    /// Guards the fix for the numeric-sort bug: groundings at chunk `#2` and `#10` (both real,
    /// distinct ordinals — the doc has 10 chunks) must concatenate in true section order (2 before
    /// 10). A lexicographic `Vec<String>::sort()` on the raw `<path>#n` targets would instead put
    /// `#10` before `#2` (`"...#10" < "...#2"` as strings), scrambling multi-chunk resolutions out
    /// of order.
    #[test]
    fn seed_source_text_all_orders_chunks_numerically_not_lexicographically() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let chunks: Vec<glossa::model::Chunk> = (1..=10)
            .map(|i| glossa::model::Chunk {
                doc_path: "d.md".into(),
                location: format!("S{i}"),
                file_type: "md".into(),
                text: if i == 2 {
                    "CHUNK TWO body".to_string()
                } else if i == 10 {
                    "CHUNK TEN body".to_string()
                } else {
                    format!("filler chunk {i}")
                },
            })
            .collect();
        idx.write_chunks(&chunks).unwrap();

        g.put_node(&Node { id: "res-1".into(), node_type: "Resolution".into(),
            label: "r".into(), aliases: vec![], prov: prov() }).unwrap();
        for n in [10u64, 2] {
            g.put_edge(&Edge { from: "res-1".into(), to: format!("d.md#{n}"),
                edge_type: glossa::graph::MENTIONS.to_string(), prov: prov() }).unwrap();
        }

        let text = seed_source_text_all(dir.path(), &idx, &g, "res-1");
        let pos2 = text.find("CHUNK TWO body").expect("chunk 2 must appear");
        let pos10 = text.find("CHUNK TEN body").expect("chunk 10 must appear");
        assert!(pos2 < pos10, "chunk #2 must come before chunk #10 in true section order: {text}");
    }

    #[test]
    fn seed_source_text_all_is_empty_when_ungrounded() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        g.put_node(&Node { id: "res-x".into(), node_type: "Resolution".into(),
            label: "r".into(), aliases: vec![], prov: prov() }).unwrap();
        assert_eq!(seed_source_text_all(dir.path(), &idx, &g, "res-x"), "");
    }
}
