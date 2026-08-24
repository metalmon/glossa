//! Task 3 — `chain_one_gold`: the backward-chain engine driving `kbx reason`.
//!
//! Reuses the SAME agent-loop substrate as `build::extract::extract_doc` (`run_agent_loop` +
//! `lmstudio_chat`) but swaps the model (`lab.distil`, a strong endpoint), the system prompt
//! (the ontology's schema-graph block + `reason.md`), and the `graph_upsert` tool's
//! description: reason is NOT pinned to `Fact` — it advertises the corpus's own ontology entity
//! types and lets the model type each node per the ontology's schema-graph, backward-chaining
//! from a gold answer to a reified question entry node.
//!
//! Routed through the CANONICAL write path (single-write-source plan, Task 3): the SAME
//! `extract_tools_schema` / `parse_and_filter_upsert` / `ops::graph_upsert` triad
//! `build::extract::extract_doc` uses (Task 2) — label-based `ops::UpsertNode`/`ops::UpsertEdge`
//! (no agent-assigned `id`), a partial-apply type filter via `Ontology::validate_node` (which
//! permits any DECLARED ontology entity type under a strict ontology, not just `Fact` — see
//! `parse_and_filter_upsert`'s doc comment in `build/extract.rs`), then `ops::graph_upsert`
//! itself, which auto-derives a MENTIONS edge from a node's `<doc>#n` source_path (Task 1). THIS
//! is what fixes the reason grounding gap: a typed node (e.g. `Symptom`) carrying a source_path
//! is now grounded automatically, which it was NOT under the old `agent::apply_upsert` path (no
//! such auto-derive existed there).
//!
//! Corpus tools (`search`/`read`/`grep`) are left doc-UNSCOPED here (unlike `extract_doc`'s
//! `scope_to_doc`): a gold `(Q, A)` pair may be grounded anywhere in the corpus, not one document.

use crate::backend::glossa_tools;
use crate::backend::openai::{lmstudio_chat, run_agent_loop};
use crate::build::extract::{extract_tools_schema, parse_and_filter_upsert, upserted_node_ids};
use crate::reason::schema_graph_block;
use crate::lab::LabConfig;
use crate::workspace::KbxPaths;
use anyhow::anyhow;
use glossa::graph::ontology::Ontology;
use glossa::graph::ops;
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;
use glossa::trace::TraceLog;
use serde_json::{json, Value};
use std::time::Duration;

/// High cap on tool-call rounds for one gold's backward-chain pass — mirrors `extract_doc`'s
/// `MAX_ROUNDS`: generous enough for several read/grep calls plus multiple `graph_upsert` writes
/// while a chain is traced back to the question, still bounded against a stuck model.
const MAX_ROUNDS: usize = 30;

/// How much of one gold's typed reasoning chain a `chain_one_gold` pass wrote: nodes and edges
/// upserted, and how many of those edges were `MENTIONS` groundings.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ReasonStats {
    pub nodes: usize,
    pub edges: usize,
    pub grounded: usize,
}

/// The `graph_upsert` tool description for `kbx reason`: unlike `extract_doc`'s Fact-only text,
/// this tells the model it may use any of the ontology's declared entity types (per the
/// schema-graph block already in the system prompt) and to honor each type's
/// `requires_grounding`/`requires_validity` flags — nothing Fact-specific, nothing hardcoded to
/// a domain's type names. LABEL-based (matches `extract_tools_schema`'s canonical shape): nodes
/// have no `id`, a bad `node_type` drops just that node (not the whole call), and edges reference
/// endpoints by label or a `<path>#<n>` section ref.
const REASON_GRAPH_UPSERT_DESC: &str =
    "Write reasoning nodes/edges for this gold's backward-anchored chain. Each node needs a \
     `node_type` matching one of the ontology's declared entity types listed above (any other \
     value drops just that node, not the whole call), a `label`, and `aliases` listing the \
     entities it mentions. Reference edge endpoints by LABEL (nodes have no id here), or as a \
     document section `<path>#<n>`. Ground every node whose type is marked `[requires_grounding]` \
     with a MENTIONS edge from it to the section you read it from (`to`: `<path>#n`) — or simply \
     give the node itself a `source_path` of `<path>#n` and grounding is derived automatically; \
     the question's reified entry node is usually left UNgrounded (the corpus holds general \
     knowledge, not this specific question) unless its type requires grounding. For a node whose \
     type is marked `[requires_validity]`, set `valid_from`/`valid_to` from what the corpus states \
     or implies. Connect nodes with edges typed as one of the ontology's declared relations, \
     respecting each relation's declared from/to types.";

/// Run the agentic backward-chain pass for ONE gold `(q, a)`: a strong model (the `lab.distil`
/// endpoint) reads the corpus (unscoped `search`/`read`/`grep`) and calls `graph_upsert` to
/// ground the answer as a terminal node, backward-chain the intermediate typed nodes/relations,
/// and reify the question as an entry node — per the corpus's own ontology schema-graph
/// (`schema_graph_block`). Each `graph_upsert` call is parsed/filtered
/// (`parse_and_filter_upsert`, ontology-permitting — NOT `Fact`-pinned, partial-apply on a
/// bad type) then written through the CANONICAL `ops::graph_upsert` (the same write path the
/// MCP server uses: resolves edges by label, auto-grounds from a `<path>#n` source_path,
/// provenance-stamped). Returns how many nodes/edges/groundings were written; errors clearly if
/// `lab.distil` is unset.
pub fn chain_one_gold(
    paths: &KbxPaths,
    ont: &Ontology,
    lab: &LabConfig,
    reason_md: &str,
    q: &str,
    a: &str,
) -> anyhow::Result<ReasonStats> {
    let distil_ep = lab
        .distil
        .as_ref()
        .ok_or_else(|| anyhow!("kbx reason needs a [distil] endpoint in lab.toml"))?;

    let root = paths.root.as_path();
    let g = GraphStore::open(root)?;
    let idx = DocIndex::open_or_create(root)?;
    let trace = TraceLog::disabled();
    let spec = glossa::tools::ChainSpec::from_ontology(ont);

    let system = format!("{}\n\n{reason_md}", schema_graph_block(ont));
    let user = format!(
        "Question: {q}\nAnswer: {a}\n\nGround the answer as a terminal typed node, backward-chain \
         the grounded intermediate nodes/relations that lead to it, and reify the question as an \
         entry node, per the ontology's schema-graph above."
    );
    let messages = vec![
        json!({ "role": "system", "content": system }),
        json!({ "role": "user", "content": user }),
    ];

    let endpoint = distil_ep.endpoint.clone();
    let model = distil_ep.model.clone();
    let api_key = distil_ep.resolve_key();
    let timeout = Duration::from_secs(distil_ep.timeout_secs);
    let tools = extract_tools_schema(REASON_GRAPH_UPSERT_DESC);

    let chat = |messages: &[Value]| {
        lmstudio_chat(
            &endpoint,
            &model,
            api_key.as_deref(),
            &tools,
            messages,
            timeout,
        )
    };

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut stats = ReasonStats::default();

    let exec = |name: &str, args: &Value| -> (String, Vec<String>) {
        if name == "graph_upsert" {
            // parse_and_filter_upsert: canonical label-based parse (ops::parse_upsert_payload)
            // plus a partial-apply type filter — a node whose node_type the ontology doesn't
            // declare is dropped with a reason; every well-typed sibling (any of the ontology's
            // declared entity types, not just Fact — reason is NOT Fact-pinned) still reaches
            // ops::graph_upsert, the SAME write path the MCP server uses.
            let (nodes, edges, notes) = parse_and_filter_upsert(args, ont);
            let out = ops::graph_upsert(&idx, &g, ont, nodes, edges, now);
            if !out.rejected {
                stats.nodes += out.nodes;
                stats.edges += out.edges;
                // Groundings actually written (explicit MENTIONS the model sent, plus any
                // ops::graph_upsert auto-derived from a node's `<path>#n` source_path per Task 1)
                // — read back from the outcome's dump, not the pre-write request.
                let mentions_marker = format!("-{}->", glossa::graph::MENTIONS);
                stats.grounded += out
                    .dump
                    .iter()
                    .filter(|l| l.starts_with("edge ") && l.contains(&mentions_marker))
                    .count();
            }
            // Surfaced ids for the unproductive-streak novelty tracker (see extract_doc's exec
            // arm for the full rationale): the ids of nodes this call actually wrote or merged
            // into.
            let ids = upserted_node_ids(&out);
            // Feed drop reasons back to the model: parse_and_filter_upsert's own notes (parse
            // fixups + type-dropped nodes) first, then ops::graph_upsert's formatted response.
            let message = if notes.is_empty() {
                out.message
            } else {
                format!("{}\n{}", notes.join("\n"), out.message)
            };
            (message, ids)
        } else {
            let (body, ids, _images) =
                glossa_tools::exec(name, args, root, &idx, None, &spec, &trace);
            (body, ids)
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

    /// Strict typed ontology declaring one non-`Fact` entity type — guards that reason is not
    /// `Fact`-pinned: `parse_and_filter_upsert` (the canonical ops-path filter, reused from
    /// `build::extract`) must accept a declared ontology type here and drop an undeclared one.
    const TYPED_ONT: &str = r#"
[entities.Symptom]
props = []
[relations.LEADS_TO]
from = ["Symptom"]
to = ["Symptom"]
[validation]
strict = true
"#;

    fn typed_ontology() -> Ontology {
        Ontology::parse(TYPED_ONT).expect("typed test ontology parses")
    }

    /// Index a stub document so `source_path`/section refs against `path` resolve — mirrors
    /// `build::extract`'s own test helper of the same purpose.
    fn write_doc(idx: &DocIndex, path: &str) {
        idx.write_chunks(&[glossa::model::Chunk {
            doc_path: path.into(),
            location: "S1".into(),
            file_type: "md".into(),
            text: "stub content".into(),
        }])
        .unwrap();
    }

    /// The reroute's whole point (single-write-source plan, Task 3): a declared non-`Fact` typed
    /// node (`Symptom`) carrying a `<doc>#n` source_path, upserted via the reason path
    /// (`parse_and_filter_upsert` -> `ops::graph_upsert`), writes AND is grounded automatically —
    /// a MENTIONS edge appears from it, with no explicit MENTIONS sent. This is the auto-derive
    /// (Task 1) reached through the reroute; it did NOT happen under the old `agent::apply_upsert`
    /// path (no such auto-derive existed there), which was the reason grounding gap.
    #[test]
    fn reason_writes_typed_node_and_grounds_from_source_path() {
        let dir = tempfile::tempdir().unwrap();
        let ont = typed_ontology();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        write_doc(&idx, "d.md");

        let call = serde_json::json!({
            "nodes": [{"node_type": "Symptom", "label": "x", "source_path": "d.md#1"}],
            "edges": []
        });
        let (nodes, edges, notes) = parse_and_filter_upsert(&call, &ont);
        assert!(notes.is_empty(), "no node should be filtered: {notes:?}");
        let out = ops::graph_upsert(&idx, &g, &ont, nodes, edges, 1);
        assert!(!out.rejected, "{}", out.message);
        assert_eq!(out.nodes, 1, "the Symptom node must be written, not dropped");
        assert_eq!(
            out.edges, 1,
            "the auto-derived MENTIONS edge must be written: {}",
            out.message
        );

        let id = ops::id_for(&ont, "Symptom", "x");
        let mentions: Vec<_> = g
            .outgoing(&id)
            .unwrap()
            .into_iter()
            .filter(|e| e.edge_type == glossa::graph::MENTIONS)
            .collect();
        assert_eq!(mentions.len(), 1, "grounded from source_path: {mentions:?}");
        assert_eq!(mentions[0].to, "d.md#1");
    }

    /// An undeclared type is DROPPED with a reason naming it (partial-apply), not rejecting the
    /// whole call — matches `build::extract`'s partial-apply behavior on the same shared filter.
    #[test]
    fn reason_drops_undeclared_type_with_reason() {
        let ont = typed_ontology();
        let call = serde_json::json!({
            "nodes": [{"node_type": "Bogus", "label": "x", "source_path": "d.md#1"}],
            "edges": []
        });
        let (nodes, _edges, notes) = parse_and_filter_upsert(&call, &ont);
        assert!(nodes.is_empty(), "the undeclared-type node must be dropped");
        assert!(
            notes.iter().any(|n| n.contains("Bogus")),
            "the drop reason must name the offending type: {notes:?}"
        );
    }
}
