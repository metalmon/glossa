//! Phase-2 (`kbx reason` seed-from-grounded) engine: read a grounded terminal's FULL source,
//! backward-synthesize the query-side reasoning layer (fan-out), and write it via the canonical
//! write triad.

use crate::backend::glossa_tools;
use crate::backend::openai::{lmstudio_chat, run_agent_loop};
use crate::build::extract::{extract_tools_schema, parse_and_filter_upsert, upserted_node_ids};
use crate::distil::Seed;
use crate::lab::LabConfig;
use crate::reason::schema_graph_block;
use crate::workspace::KbxPaths;
use anyhow::anyhow;
use glossa::graph::ontology::Ontology;
use glossa::graph::ops;
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;
use glossa::trace::TraceLog;
use serde_json::{json, Value};
use std::time::Duration;

/// High cap on tool-call rounds for one seed's backward-synth pass — generous for several reads
/// plus multiple fan-out `graph_upsert` writes, still bounded against a stuck model.
const MAX_ROUNDS: usize = 30;

/// How much of one seed's typed reasoning layer a `chain_one_seed` pass wrote: nodes and edges
/// upserted, and how many of those edges were `MENTIONS` groundings.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ReasonStats {
    pub nodes: usize,
    pub edges: usize,
    pub grounded: usize,
}

/// Thin, generic `graph_upsert` tool description for phase-2. The behavioural guidance lives in the
/// system prompt (`reason.md`, loaded from disk and editable without a rebuild) and the field-level
/// contracts (source_path = `path#n`/omit-for-ungrounded; edge endpoints by label or existing node
/// id) live in the tool's parameter schema (`graph_upsert_tool_value`). This one line only names
/// the tool's job so nothing behavioural is baked into the binary.
pub(crate) const SEED_GRAPH_UPSERT_DESC: &str =
    "Write the query-side reasoning nodes and edges for this terminal, as instructed in the system \
     prompt.";

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
        "Grounded terminal node: {} [{}] \"{}\"\nIts grounded source text:\n{}\n\nBuild the \
         query-side reasoning layer that leads to this terminal, following the system prompt. \
         Branch to at most {} predecessor(s) per step.",
        seed.id, seed.node_type, seed.label, body, fanout_max
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
        // `run_read` now leads each chunk with its own `── path#n ──` header (the copy-ready ref),
        // so the concatenated source already carries the token the model reuses as source_path.
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
            let out = ops::graph_upsert(&idx, &g, ont, nodes, edges, now, "agent");
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
