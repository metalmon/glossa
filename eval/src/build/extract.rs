//! Task 5 — agentic STATED fact-extraction over a single document.
//!
//! Reuses `backend::openai::run_agent_loop` for the tool-calling loop (the SAME substrate the
//! eval reader drives) with a build-specific tool set: doc-scoped `search`/`read`/`grep` plus a
//! `graph_upsert` tool. `graph_upsert` parses the model's call into `NodeSpec`/`EdgeSpec`
//! (`parse_and_validate_upsert`) and rejects the WHOLE call if any node's type is not declared
//! by the ontology, naming the bad type so the model can fix and resend — then writes through
//! `glossa::graph::agent::apply_upsert` (strict, provenance-stamped, dedup-by-label).
//!
//! `parse_and_validate_upsert` is the one piece testable without a live model; `extract_doc`
//! drives the full loop and is exercised by a live-endpoint smoke run, not a unit test.

use crate::backend::glossa_tools;
use crate::backend::openai::{lmstudio_chat, run_agent_loop};
use crate::lab::LabConfig;
use anyhow::anyhow;
use glossa::graph::agent::{apply_upsert, EdgeSpec, NodeSpec};
use glossa::graph::ontology::Ontology;
use glossa::graph::store::GraphStore;
use glossa::grep::path_to_glob;
use glossa::index::store::DocIndex;
use glossa::trace::TraceLog;
use serde_json::{json, Value};
use std::path::Path;
use std::time::Duration;

/// High cap on tool-call rounds for one document's extraction pass — generous because a real
/// document may need several read/grep calls before every STATED node is grounded, but still
/// bounded so a stuck model can't loop forever.
const MAX_ROUNDS: usize = 30;

/// How much of one document's STATED reasoning graph an `extract_doc` pass wrote.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ExtractStats {
    pub nodes: usize,
    pub mentions: usize,
}

/// Parse a model `graph_upsert` tool-call JSON (`{"nodes":[...], "edges":[...]}`, each item
/// shaped like [`NodeSpec`]/[`EdgeSpec`] — the agent assigns its own `id`, unlike the label-only
/// `UpsertNode`/`UpsertEdge` the MCP-facing `graph_upsert` tool uses) into specs ready for
/// [`apply_upsert`]. Rejects the WHOLE call — before anything is written — if any node's
/// `node_type` is not declared by `ont` (strict), naming the offending type in the error so the
/// model can fix and resend instead of silently losing the bad item.
pub fn parse_and_validate_upsert(
    call: &Value,
    ont: &Ontology,
) -> anyhow::Result<(Vec<NodeSpec>, Vec<EdgeSpec>)> {
    let nodes: Vec<NodeSpec> = match call.get("nodes") {
        Some(v) => {
            serde_json::from_value(v.clone()).map_err(|e| anyhow!("invalid nodes[]: {e}"))?
        }
        None => Vec::new(),
    };
    let edges: Vec<EdgeSpec> = match call.get("edges") {
        Some(v) => {
            serde_json::from_value(v.clone()).map_err(|e| anyhow!("invalid edges[]: {e}"))?
        }
        None => Vec::new(),
    };
    for n in &nodes {
        ont.validate_node(&n.node_type).map_err(|e| anyhow!(e))?;
    }
    Ok((nodes, edges))
}

/// OpenAI-function tool schema for the build agent: `search`/`read`/`grep` descriptors come
/// straight from the shared registry (`glossa::tools::registry`) so their name/description/
/// schema can't drift from the eval reader's; `graph_upsert` is build-specific (it takes
/// `NodeSpec`/`EdgeSpec` — agent-assigned ids — not the MCP surface's label-only upsert).
fn build_tools_schema() -> Value {
    let mut tools: Vec<Value> = glossa::tools::registry::registry()
        .into_iter()
        .filter(|d| matches!(d.name, "search" | "read" | "grep"))
        .map(|d| {
            json!({
                "type": "function",
                "function": {
                    "name": d.name,
                    "description": d.description,
                    "parameters": d.params_schema,
                }
            })
        })
        .collect();
    tools.push(json!({
        "type": "function",
        "function": {
            "name": "graph_upsert",
            "description": "Write reasoning nodes/edges this document STATES. Each node needs a unique `id` (your choice), a `node_type` the ontology allows (see the system prompt — an out-of-ontology type rejects the WHOLE call), and a `label`. Ground every node with a MENTIONS edge from it to the section you read it from (`to`: `<path>#n`).",
            "parameters": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "node_type": { "type": "string" },
                                "label": { "type": "string" },
                                "aliases": { "type": "array", "items": { "type": "string" } },
                                "source_path": { "type": "string" },
                                "range": { "type": ["string", "null"] },
                                "confidence": { "type": ["number", "null"] }
                            },
                            "required": ["id", "node_type", "label", "source_path"]
                        }
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from": { "type": "string" },
                                "to": { "type": "string" },
                                "edge_type": { "type": "string" },
                                "source_path": { "type": "string" },
                                "range": { "type": ["string", "null"] },
                                "confidence": { "type": ["number", "null"] }
                            },
                            "required": ["from", "to", "edge_type", "source_path"]
                        }
                    }
                },
                "required": []
            }
        }
    }));
    Value::Array(tools)
}

/// Force a `search`/`grep`/`read` call's path scope onto `doc_path`, overriding whatever the
/// model passed — a STATED-extraction pass over ONE document must never wander into another
/// document's text.
fn scope_to_doc(name: &str, args: &Value, doc_path: &str) -> Value {
    let mut a = args.clone();
    match name {
        "search" => a["glob"] = json!(path_to_glob(doc_path)),
        "grep" | "read" => a["path"] = json!(doc_path),
        _ => {}
    }
    a
}

/// Run the agentic STATED-extraction loop over ONE document: the model reads what it needs
/// (`search`/`read`/`grep`, all doc-scoped) and calls `graph_upsert` to ground reasoning nodes
/// it finds explicitly stated in the text. `builder_md` is the system prompt verbatim, with the
/// ontology's allowed node types appended; the user turn asks for STATED-only nodes grounded by
/// a MENTIONS edge to `<doc_path>#n`. Returns how many nodes/MENTIONS edges were written.
pub fn extract_doc(
    root: &Path,
    lab: &LabConfig,
    builder_md: &str,
    ontology: &Ontology,
    doc_path: &str,
) -> anyhow::Result<ExtractStats> {
    let g = GraphStore::open(root)?;
    let idx = DocIndex::open_or_create(root)?;
    let trace = TraceLog::disabled();
    let spec = glossa::tools::ChainSpec::from_ontology(ontology);

    let allowed: Vec<String> = ontology.entity_types().iter().cloned().collect();
    let system = format!(
        "{builder_md}\n\nAllowed node types (ontology): {}",
        allowed.join(", ")
    );
    let user = format!(
        "Extract the reasoning nodes this document STATES. Read what you need; ground each \
         node with a MENTIONS edge to its `{doc_path}#n`. Emit only node types the ontology \
         allows.\n\nDocument: {doc_path}"
    );
    let messages = vec![
        json!({ "role": "system", "content": system }),
        json!({ "role": "user", "content": user }),
    ];

    let endpoint = lab.model.endpoint.clone();
    let model = lab.model.model.clone();
    let api_key = lab.model.resolve_key();
    let timeout = Duration::from_secs(lab.model.timeout_secs);
    let tools = build_tools_schema();

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
    let mut stats = ExtractStats::default();

    let exec = |name: &str, args: &Value| -> (String, Vec<String>) {
        if name == "graph_upsert" {
            match parse_and_validate_upsert(args, ontology) {
                Ok((nodes, edges)) => {
                    // Surfaced ids for the unproductive-streak novelty tracker: the requested node
                    // ids. A batch of distinct nodes must register as "new" progress even though
                    // graph_upsert's own reply is a short confirmation/rejection string with no
                    // `path#ord` anchors of its own (`extract_node_ids` would find nothing in it).
                    // Without this, 3+ consecutive distinct graph_upsert calls — very plausible
                    // (several facts from one read, or retries after a rejection) — would each
                    // surface zero ids and, at UNPRODUCTIVE_STREAK_K, have their real result
                    // (confirmation OR rejection text) replaced by the generic steer.
                    let ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
                    let mentions = edges
                        .iter()
                        .filter(|e| e.edge_type == glossa::graph::MENTIONS)
                        .count();
                    match apply_upsert(&g, ontology, nodes, edges, now, root) {
                        Ok(r) => {
                            stats.nodes += r.nodes_written;
                            stats.mentions += mentions;
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
            let scoped = scope_to_doc(name, args, doc_path);
            let (body, ids, _images) =
                glossa_tools::exec(name, &scoped, root, &idx, None, &spec, &trace);
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

    const FLAT_ONT: &str = r#"
[entities.Fact]
props = []
[relations.LEADS_TO]
from = ["Fact"]
to = ["Fact"]
[validation]
strict = true
"#;

    /// Minimal flat test ontology: one entity type (`Fact`) + one relation (`LEADS_TO`),
    /// strict — stands in for the brief's `Ontology::flat_default()` (no such constructor
    /// exists; `Ontology::parse` is the real one). `MENTIONS` needs no declaration — it's a
    /// CORE_EDGE, always allowed regardless of the ontology.
    fn flat_ontology() -> Ontology {
        Ontology::parse(FLAT_ONT).expect("flat test ontology parses")
    }

    /// Regression test for the ids-fix: mirrors `extract_doc`'s `graph_upsert` exec arm
    /// (parse -> validate -> apply_upsert, returning the requested node ids as the surfaced
    /// ids) directly against `run_agent_loop`, with no live model. Before the fix this arm
    /// always returned `Vec::new()` for ids, so N+1 consecutive DISTINCT `graph_upsert` calls
    /// (very plausible — several facts from one read) each looked "unproductive" to the
    /// streak detector; by the (K+1)th call its real "upserted ..." confirmation would have
    /// been replaced by the generic steer. Mirrors
    /// `openai::tests::loop_unproductive_streak_never_fires_when_calls_are_productive`.
    #[test]
    fn loop_distinct_graph_upsert_calls_never_trip_unproductive_streak() {
        use crate::backend::openai::{run_agent_loop, UNPRODUCTIVE_STREAK_K};
        use glossa::graph::store::GraphStore;
        use std::cell::RefCell;

        let dir = tempfile::tempdir().unwrap();
        let ont = flat_ontology();
        let g = GraphStore::open(dir.path()).unwrap();

        // Same shape as extract_doc's graph_upsert arm: parse -> validate -> apply_upsert,
        // returning the requested node ids as the surfaced ids (the fix under test).
        let exec = |_name: &str, args: &Value| -> (String, Vec<String>) {
            match parse_and_validate_upsert(args, &ont) {
                Ok((nodes, edges)) => {
                    let ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
                    match apply_upsert(&g, &ont, nodes, edges, 1, dir.path()) {
                        Ok(r) => (format!("upserted {} node(s)", r.nodes_written), ids),
                        Err(e) => (format!("graph_upsert REJECTED: {e}"), ids),
                    }
                }
                Err(e) => (format!("graph_upsert REJECTED: {e}"), Vec::new()),
            }
        };

        let n = UNPRODUCTIVE_STREAK_K + 1; // enough distinct calls to have tripped the old bug
        let round = RefCell::new(0usize);
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            if let Some(last_tool) = msgs.iter().rev().find(|m| m["role"] == "tool") {
                let c = last_tool["content"].as_str().unwrap_or("");
                assert!(
                    !c.to_lowercase().contains("no new information"),
                    "round {}: steer must not fire on distinct graph_upsert writes, got: {c:?}",
                    *r
                );
                assert!(
                    c.contains("upserted"),
                    "round {}: the real graph_upsert confirmation must reach the model, got: {c:?}",
                    *r
                );
            }
            if *r > n {
                return Ok(json!({ "role": "assistant", "content": "ANSWER: done" }));
            }
            let i = *r - 1;
            Ok(json!({
                "role": "assistant", "content": "writing",
                "tool_calls": [{
                    "id": format!("c{}", *r),
                    "function": {
                        "name": "graph_upsert",
                        "arguments": json!({
                            "nodes": [{
                                "id": format!("f{i}"),
                                "node_type": "Fact",
                                "label": format!("fact {i}"),
                                "source_path": "d.md"
                            }],
                            "edges": []
                        }).to_string()
                    }
                }]
            }))
        };

        let nudge = |name: &str, _args: &Value| format!("(dup {name}) you already called this");
        let out = run_agent_loop(chat, vec![], exec, nudge, n + 2).unwrap();
        assert_eq!(out, "ANSWER: done");
    }

    #[test]
    fn parse_upsert_rejects_out_of_ontology_type() {
        let ont = flat_ontology();
        let call = serde_json::json!({
            "nodes":[{"id":"f1","node_type":"Bogus","label":"x","source_path":"d.md"}],
            "edges":[]
        });
        let err = parse_and_validate_upsert(&call, &ont).unwrap_err();
        assert!(err.to_string().contains("Bogus"));
    }

    #[test]
    fn parse_upsert_accepts_ontology_type_and_grounding_edge() {
        let ont = flat_ontology();
        let call = serde_json::json!({
            "nodes":[{"id":"f1","node_type":"Fact","label":"x","source_path":"d.md"}],
            "edges":[{"from":"f1","to":"d.md#1","edge_type":"MENTIONS","source_path":"d.md"}]
        });
        let (nodes, edges) = parse_and_validate_upsert(&call, &ont).unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].node_type, "Fact");
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].edge_type, "MENTIONS");
    }

    #[test]
    fn parse_upsert_defaults_missing_arrays_to_empty() {
        let ont = flat_ontology();
        let (nodes, edges) = parse_and_validate_upsert(&serde_json::json!({}), &ont).unwrap();
        assert!(nodes.is_empty());
        assert!(edges.is_empty());
    }
}
