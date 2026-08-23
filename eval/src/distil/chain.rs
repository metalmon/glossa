//! Task 3 — `chain_one_gold`: the backward-chain engine driving `kbx distil`.
//!
//! Reuses the SAME agent-loop substrate as `build::extract::extract_doc` (`run_agent_loop` +
//! `lmstudio_chat`) but swaps the model (`lab.distil`, a strong endpoint), the system prompt
//! (the ontology's schema-graph block + `distil.md`), and the `graph_upsert` tool's
//! description: distil is NOT pinned to `Fact` — it advertises the corpus's own ontology entity
//! types and lets the model type each node per the ontology's schema-graph, backward-chaining
//! from a gold answer to a reified question entry node.
//!
//! `parse_and_validate_upsert` already permits any declared ontology `entity_type` under a
//! strict ontology — `Ontology::validate_node` accepts `CORE_NODES`, the flat `Fact` type, AND
//! any declared `entity_types`/`constraint_types`. The `Fact`-only restriction `extract_doc` uses
//! lives solely in ITS prompt text (and, before this task, in `build_tools_schema`'s hardcoded
//! tool description), never in the shared parse/validate helper. So distil reuses
//! `parse_and_validate_upsert` and `build_tools_schema` unchanged in shape, passing only a
//! different description string (see `build_tools_schema`'s doc comment).
//!
//! Corpus tools (`search`/`read`/`grep`) are left doc-UNSCOPED here (unlike `extract_doc`'s
//! `scope_to_doc`): a gold `(Q, A)` pair may be grounded anywhere in the corpus, not one document.

use crate::backend::glossa_tools;
use crate::backend::openai::{lmstudio_chat, run_agent_loop};
use crate::build::extract::{build_tools_schema, parse_and_validate_upsert};
use crate::distil::schema_graph_block;
use crate::lab::LabConfig;
use crate::workspace::KbxPaths;
use anyhow::anyhow;
use glossa::graph::agent::apply_upsert;
use glossa::graph::ontology::Ontology;
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
pub struct DistilStats {
    pub nodes: usize,
    pub edges: usize,
    pub grounded: usize,
}

/// The `graph_upsert` tool description for `kbx distil`: unlike `extract_doc`'s Fact-only text,
/// this tells the model it may use any of the ontology's declared entity types (per the
/// schema-graph block already in the system prompt) and to honor each type's
/// `requires_grounding`/`requires_validity` flags — nothing Fact-specific, nothing hardcoded to
/// a domain's type names.
const DISTIL_GRAPH_UPSERT_DESC: &str =
    "Write reasoning nodes/edges for this gold's backward-anchored chain. Each node needs a \
     unique `id` (your choice), a `node_type` matching one of the ontology's declared entity \
     types listed above (any other value rejects the WHOLE call), a `label`, and `aliases` \
     listing the entities it mentions. Ground every node whose type is marked \
     `[requires_grounding]` with a MENTIONS edge from it to the section you read it from (`to`: \
     `<path>#n`); the question's reified entry node is usually left UNgrounded (the corpus holds \
     general knowledge, not this specific question) unless its type requires grounding. For a \
     node whose type is marked `[requires_validity]`, set `valid_from`/`valid_to` from what the \
     corpus states or implies. Connect nodes with edges typed as one of the ontology's declared \
     relations, respecting each relation's declared from/to types.";

/// Run the agentic backward-chain pass for ONE gold `(q, a)`: a strong model (the `lab.distil`
/// endpoint) reads the corpus (unscoped `search`/`read`/`grep`) and calls `graph_upsert` to
/// ground the answer as a terminal node, backward-chain the intermediate typed nodes/relations,
/// and reify the question as an entry node — per the corpus's own ontology schema-graph
/// (`schema_graph_block`). Each `graph_upsert` call is parsed/validated
/// (`parse_and_validate_upsert`, ontology-permitting — NOT `Fact`-pinned) then applied
/// (`apply_upsert`, temporality-aware, provenance-stamped). Returns how many nodes/edges/
/// groundings were written; errors clearly if `lab.distil` is unset.
pub fn chain_one_gold(
    paths: &KbxPaths,
    ont: &Ontology,
    lab: &LabConfig,
    distil_md: &str,
    q: &str,
    a: &str,
) -> anyhow::Result<DistilStats> {
    let distil_ep = lab
        .distil
        .as_ref()
        .ok_or_else(|| anyhow!("kbx distil needs a [distil] endpoint in lab.toml"))?;

    let root = paths.root.as_path();
    let g = GraphStore::open(root)?;
    let idx = DocIndex::open_or_create(root)?;
    let trace = TraceLog::disabled();
    let spec = glossa::tools::ChainSpec::from_ontology(ont);

    let system = format!("{}\n\n{distil_md}", schema_graph_block(ont));
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
    let tools = build_tools_schema(DISTIL_GRAPH_UPSERT_DESC);

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
    let mut stats = DistilStats::default();

    let exec = |name: &str, args: &Value| -> (String, Vec<String>) {
        if name == "graph_upsert" {
            match parse_and_validate_upsert(args, ont) {
                Ok((nodes, edges)) => {
                    let ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
                    let grounded = edges
                        .iter()
                        .filter(|e| e.edge_type == glossa::graph::MENTIONS)
                        .count();
                    match apply_upsert(&g, ont, nodes, edges, now, root) {
                        Ok(r) => {
                            stats.nodes += r.nodes_written;
                            stats.edges += r.edges_written;
                            stats.grounded += grounded;
                            (
                                format!(
                                    "upserted {} node(s), {} edge(s)",
                                    r.nodes_written, r.edges_written
                                ),
                                ids,
                            )
                        }
                        Err(e) => (format!("graph_upsert REJECTED: {e}"), ids),
                    }
                }
                Err(e) => (format!("graph_upsert REJECTED: {e}"), Vec::new()),
            }
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

    /// Strict typed ontology declaring one non-`Fact` entity type — guards that distil is not
    /// `Fact`-pinned: `parse_and_validate_upsert` must accept a declared ontology type here and
    /// reject an undeclared one, exactly as it does for `Fact` under `build::extract`'s tests.
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

    #[test]
    fn parse_upsert_accepts_declared_non_fact_type_under_strict_ontology() {
        let ont = typed_ontology();
        let call = serde_json::json!({
            "nodes": [{"id": "s1", "node_type": "Symptom", "label": "x", "source_path": "d.md"}],
            "edges": []
        });
        let (nodes, _edges) = parse_and_validate_upsert(&call, &ont).unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].node_type, "Symptom");
    }

    #[test]
    fn parse_upsert_rejects_undeclared_type_under_strict_ontology() {
        let ont = typed_ontology();
        let call = serde_json::json!({
            "nodes": [{"id": "s1", "node_type": "Bogus", "label": "x", "source_path": "d.md"}],
            "edges": []
        });
        let err = parse_and_validate_upsert(&call, &ont).unwrap_err();
        assert!(err.to_string().contains("Bogus"));
    }
}
