//! Task 5 — agentic STATED fact-extraction over a single document.
//!
//! Reuses `backend::openai::run_agent_loop` for the tool-calling loop (the SAME substrate the
//! eval reader drives) with a build-specific tool set: doc-scoped `search`/`read`/`grep` plus a
//! `graph_upsert` tool. `graph_upsert` parses the model's call via the CANONICAL
//! `glossa::graph::ops::parse_upsert_payload` (label-based `UpsertNode`/`UpsertEdge` — no
//! agent-assigned ids) into label-referenced upserts, drops any node whose type the ontology
//! doesn't declare (partial — siblings still write; extraction is pinned to `Fact`, so this only
//! fires on a model mistake), and writes the rest through `glossa::graph::ops::graph_upsert` — the
//! SAME write path the MCP server uses (validates, resolves edge endpoints by label, auto-grounds
//! from a `<path>#n` source_path, provenance-stamped, dedup-by-label).
//!
//! `extract_doc` drives the full agent loop and is exercised by a live-endpoint smoke run, not a
//! unit test; the `graph_upsert` exec arm's parse/filter/write behavior is unit-testable and
//! covered below without a live model.

use crate::backend::glossa_tools;
use crate::backend::openai::{lmstudio_chat, run_agent_loop};
use crate::lab::LabConfig;
use crate::reason::schema_graph_block;
use glossa::graph::ontology::Ontology;
use glossa::graph::ops;
use glossa::graph::store::GraphStore;
use glossa::grep::path_to_glob;
use glossa::index::store::DocIndex;
use glossa::read::DocImage;
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

/// Parse a model `graph_upsert` tool-call JSON via the canonical
/// [`ops::parse_upsert_payload`], then drop (partial — not reject-the-whole-call) any node whose
/// `node_type` `ont` doesn't declare, naming the offending type in a reason string so the model
/// can fix and resend just that item. Extraction is pinned to a single flat type (`Fact`, always
/// declared — see `extract_doc`), so this only fires when the model emits some other type by
/// mistake; every well-typed sibling node (and every edge — `ops::graph_upsert` resolves edges by
/// label, so an edge naming a dropped node simply fails to resolve on its own, with its own
/// actionable reason) still reaches the write path.
///
/// Returns `(nodes, edges, notes)`: `notes` carries `ops::parse_upsert_payload`'s own parse notes
/// (tolerant node/edge-shape fixups) plus one line per dropped-for-type node.
pub(crate) fn parse_and_filter_upsert(
    call: &Value,
    ont: &Ontology,
) -> (Vec<ops::UpsertNode>, Vec<ops::UpsertEdge>, Vec<String>) {
    let (mut nodes, edges, mut notes) = ops::parse_upsert_payload(call);
    nodes.retain(|n| match ont.validate_node(&n.node_type) {
        Ok(()) => true,
        Err(e) => {
            notes.push(format!("node \"{}\" dropped: {e}", n.label));
            false
        }
    });
    (nodes, edges, notes)
}

/// Harvest keeps ONLY `requires_grounding` node types (Symptom/Cause/Task are query/chain side,
/// not harvested from documents). Non-grounding types are dropped with a reason.
pub(crate) fn filter_grounding_only(
    nodes: Vec<ops::UpsertNode>,
    ont: &Ontology,
) -> (Vec<ops::UpsertNode>, Vec<String>) {
    let mut kept = Vec::new();
    let mut notes = Vec::new();
    for n in nodes {
        if ont.requires_grounding(&n.node_type) {
            kept.push(n);
        } else {
            notes.push(format!("dropped node type `{}` — harvest creates only grounding-required types", n.node_type));
        }
    }
    (kept, notes)
}

/// Node ids `graph_upsert` actually wrote or merged into, read back from an
/// [`ops::UpsertOutcome`] — for the unproductive-streak novelty tracker (see `extract_doc`'s exec
/// closure). `dump` lines for a written node are `"node <id> [<type>] <label>..."`; `merged` holds
/// `(requested_id, canonical_id)` pairs for nodes that deduped into an existing one. Empty when
/// the call was rejected (nothing was written) — `dump`/`merged` are only populated on success.
pub(crate) fn upserted_node_ids(out: &ops::UpsertOutcome) -> Vec<String> {
    if out.rejected {
        return Vec::new();
    }
    let mut ids: Vec<String> = out
        .dump
        .iter()
        .filter_map(|line| line.strip_prefix("node "))
        .filter_map(|rest| rest.split_whitespace().next())
        .map(|s| s.to_string())
        .collect();
    ids.extend(out.merged.iter().map(|(_, canonical)| canonical.clone()));
    ids
}

/// The `graph_upsert` tool description `extract_doc` (`kbx build`) uses: pinned to `Fact`, and
/// LABEL-based (mirrors the canonical `ops::UpsertNode`/`ops::UpsertEdge` — no agent-assigned
/// `id`; a bad `node_type` drops just that node, not the whole call). Kept as a named constant so
/// the call site can pass it explicitly to [`extract_tools_schema`].
#[allow(dead_code)] // superseded by BUILD_GRAPH_UPSERT_DESC (ontology-typed harvest spike); kept for the ontology-free path
const FACT_ONLY_GRAPH_UPSERT_DESC: &str = "Write reasoning nodes/edges this document STATES. \
     Each node needs `node_type` (MUST be `Fact` — any other value drops just that node, not the \
     whole call), a `label`, an indexed `source_path`, and `aliases` listing the entities it \
     mentions. Reference edge endpoints by LABEL (nodes have no id here), or as a document \
     section `<path>#<n>`. Ground every Fact with a MENTIONS edge from it to the section you \
     read it from (`to`: `<path>#n`) — or simply give the node itself a `source_path` of \
     `<path>#n` and grounding is derived automatically.";

/// The `graph_upsert` tool description for the ontology-TYPED build harvest (spike): unlike the
/// Fact-pinned `FACT_ONLY_GRAPH_UPSERT_DESC`, the model may use ANY of the ontology's declared
/// entity types (listed in the schema-graph block in the system prompt) and grounds each node to
/// the section it read it from. LABEL-based, partial-apply on a bad type — same as reason's.
const BUILD_GRAPH_UPSERT_DESC: &str =
    "Write the reasoning nodes this document STATES. Each node needs a `node_type` matching one \
     of the ontology's declared entity types listed above (any other value drops just that node, \
     not the whole call), a `label` — the node as ONE short phrase, with concrete values left in \
     the source, not copied into the label — and `aliases` listing the entities it mentions. \
     Reference edge endpoints by LABEL (nodes have no id here), or as a document section \
     `<path>#<n>`. Ground every node whose type is marked `[requires_grounding]` with a MENTIONS \
     edge from it to the section you read it from (`to`: `<path>#n`) — or simply give the node a \
     `source_path` of `<path>#n` and grounding is derived automatically. Connect nodes with edges \
     typed as one of the ontology's declared relations, respecting each relation's from/to types.";

/// The `graph_upsert` function-schema shape (reusable across all tool schemas that need it).
/// Returns the OpenAI-function tool object with name/description/parameters set per the caller's
/// description. DRY helper for `extract_tools_schema` and `build_tools_schema`.
fn graph_upsert_tool_value(desc: &str) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "graph_upsert",
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "node_type": { "type": "string" },
                                "label": { "type": "string" },
                                "aliases": { "type": "array", "items": { "type": "string" } },
                                "source_path": { "type": "string" },
                                "valid_from": { "type": ["string", "null"], "description": "Start of this fact's validity interval, if the document states or implies one (any ISO-8601 granularity, e.g. \"2020\", \"2020-06\", \"2020-06-15\")." },
                                "valid_to": { "type": ["string", "null"], "description": "End of this fact's validity interval, if the document states or implies one (same granularity as valid_from)." }
                            },
                            "required": ["node_type", "label", "source_path"]
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
                                "source_path": { "type": "string" }
                            },
                            "required": ["from", "to", "edge_type", "source_path"]
                        }
                    }
                },
                "required": []
            }
        }
    })
}

/// OpenAI-function tool schema for `extract_doc`'s (and `kbx reason`'s) graph-writing agent — the
/// CANONICAL label-based shape: `graph_upsert`'s `nodes[]`/`edges[]` mirror [`ops::UpsertNode`]/
/// [`ops::UpsertEdge`] exactly (no agent-assigned `id`; edges reference endpoints by `label` or a
/// `<path>#<n>` section ref). `search`/`read`/`grep` descriptors come straight from the shared
/// registry (`glossa::tools::registry`).
pub(crate) fn extract_tools_schema(graph_upsert_description: &str) -> Value {
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
    tools.push(graph_upsert_tool_value(graph_upsert_description));
    Value::Array(tools)
}

/// Build-path tool schema: only `read` for document access plus `graph_upsert` for writing the
/// reasoning graph. No `search` or `grep` — the builder operates on a single document at a time.
pub(crate) fn build_tools_schema(graph_upsert_description: &str) -> Value {
    let mut tools: Vec<Value> = glossa::tools::registry::registry()
        .into_iter()
        .filter(|d| d.name == "read")
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
    tools.push(graph_upsert_tool_value(graph_upsert_description));
    Value::Array(tools)
}

/// Gate a tool call's returned images by the `--vision` flag: `Vec::new()` (discarded) when
/// `vision` is off — preserving today's text-only extraction byte-for-byte — or `images`
/// unchanged when on, so `extract_doc`'s agent loop can feed them to the model (see
/// `run_agent_loop`'s vision-message threading in `backend::openai`). Only `read` ever returns a
/// non-empty `images`; gating here (rather than per-tool) keeps the call site a one-liner.
fn gate_images(vision: bool, images: Vec<DocImage>) -> Vec<DocImage> {
    if vision {
        images
    } else {
        Vec::new()
    }
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
///
/// `vision`: `kbx build --vision` (default OFF). When on, images `read` returns (page rasters /
/// embedded figures — see `glossa::read::DocImage`) are fed to the extraction model as vision
/// input via `run_agent_loop`'s shared image-threading mechanism, so scanned/image-only content
/// can still yield grounded facts. When off, `gate_images` discards them and this pass is
/// byte-identical to the pre-vision text-only extraction.
pub fn extract_doc(
    root: &Path,
    lab: &LabConfig,
    builder_md: &str,
    ontology: &Ontology,
    doc_path: &str,
    vision: bool,
) -> anyhow::Result<ExtractStats> {
    let g = GraphStore::open(root)?;
    let idx = DocIndex::open_or_create(root)?;
    let trace = TraceLog::disabled();
    let spec = glossa::tools::ChainSpec::from_ontology(ontology);

    // Ontology-TYPED harvest (spike): drop the `Fact` pin — the model reads the doc and creates
    // nodes typed per the corpus's OWN ontology (schema-graph block for the type list/relations +
    // per-type descriptions carried in `builder_md`, mirroring reason's `schema_graph_block +
    // reason_md`), or nothing. `parse_and_filter_upsert` already permits any declared type.
    let system = format!("{}\n\n{builder_md}", schema_graph_block(ontology));
    let user = format!(
        "Read this document. Wherever a section STATES or PRESCRIBES something that matches one of \
         the node types above, create it with `graph_upsert`, grounded to that section's \
         `{doc_path}#n`. A section that states no such node — a heading, boilerplate, or a bare \
         list of values — gets nothing. Do not invent nodes the text does not state.\n\n\
         Document: {doc_path}"
    );
    let messages = vec![
        json!({ "role": "system", "content": system }),
        json!({ "role": "user", "content": user }),
    ];

    let endpoint = lab.model.endpoint.clone();
    let model = lab.model.model.clone();
    let api_key = lab.model.resolve_key();
    let timeout = Duration::from_secs(lab.model.timeout_secs);
    let tools = extract_tools_schema(BUILD_GRAPH_UPSERT_DESC);

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

    let exec = |name: &str, args: &Value| -> (String, Vec<String>, Vec<DocImage>) {
        if name == "graph_upsert" {
            // parse_and_filter_upsert: canonical label-based parse (ops::parse_upsert_payload)
            // plus a partial-apply type filter — a node whose node_type the ontology doesn't
            // declare (extraction is pinned to Fact, so this only fires on a model mistake) is
            // dropped with a reason; every well-typed sibling still reaches ops::graph_upsert,
            // the SAME write path the MCP server uses.
            let (nodes, edges, notes) = parse_and_filter_upsert(args, ontology);
            let out = ops::graph_upsert(&idx, &g, ontology, nodes, edges, now);
            if !out.rejected {
                stats.nodes += out.nodes;
                // Count MENTIONS edges actually written (explicit ones the model sent, plus any
                // ops::graph_upsert auto-derived from a node's `<path>#n` source_path) — read back
                // from the outcome's dump rather than the pre-write request, since a requested
                // edge can still be dropped (e.g. an unresolvable endpoint).
                let mentions_marker = format!("-{}->", glossa::graph::MENTIONS);
                stats.mentions += out
                    .dump
                    .iter()
                    .filter(|l| l.starts_with("edge ") && l.contains(&mentions_marker))
                    .count();
            }
            // Surfaced ids for the unproductive-streak novelty tracker: the ids of nodes this call
            // actually wrote or merged into (see `upserted_node_ids`). A batch of distinct nodes
            // must register as "new" progress even though graph_upsert's own reply is a short
            // confirmation/rejection string with no `path#ord` anchors of its own (`extract_node_ids`
            // would find nothing in it). Without this, 3+ consecutive distinct graph_upsert calls —
            // very plausible (several facts from one read, or retries after a rejection) — would
            // each surface zero ids and, at UNPRODUCTIVE_STREAK_K, have their real result
            // (confirmation OR rejection text) replaced by the generic steer.
            let ids = upserted_node_ids(&out);
            // Feed drop reasons back to the model: parse_and_filter_upsert's own notes (parse
            // fixups + type-dropped nodes) first, then ops::graph_upsert's formatted response
            // (which already lists its own dropped/merged items — mirrors format_upsert_response).
            let message = if notes.is_empty() {
                out.message
            } else {
                format!("{}\n{}", notes.join("\n"), out.message)
            };
            (message, ids, Vec::new())
        } else {
            let scoped = scope_to_doc(name, args, doc_path);
            let (body, ids, images) =
                glossa_tools::exec(name, &scoped, root, &idx, None, &spec, &trace);
            (body, ids, gate_images(vision, images))
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

    /// Index a stub document so `source_path`/section refs against `path` resolve — mirrors
    /// `glossa::graph::ops`'s own test helper of the same purpose.
    fn write_doc(idx: &DocIndex, path: &str) {
        idx.write_chunks(&[glossa::model::Chunk {
            doc_path: path.into(),
            location: "S1".into(),
            file_type: "md".into(),
            text: "stub content".into(),
        }])
        .unwrap();
    }

    /// Step 1 of the task: a valid `Fact` node writes through the canonical ops path AND is
    /// grounded automatically from its `<path>#n` source_path (no explicit MENTIONS edge sent) —
    /// `ops::graph_upsert`'s Task-1 auto-derive, now reached via `extract_doc`'s reroute.
    #[test]
    fn extract_writes_fact_and_grounds_from_source_path() {
        use glossa::graph::store::GraphStore;

        let dir = tempfile::tempdir().unwrap();
        let ont = flat_ontology();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        write_doc(&idx, "d.md");

        let call = serde_json::json!({
            "nodes": [{"node_type": "Fact", "label": "x", "source_path": "d.md#1"}],
            "edges": []
        });
        let (nodes, edges, notes) = parse_and_filter_upsert(&call, &ont);
        assert!(notes.is_empty(), "no node should be filtered: {notes:?}");
        let out = ops::graph_upsert(&idx, &g, &ont, nodes, edges, 1);
        assert!(!out.rejected, "{}", out.message);
        assert_eq!(out.nodes, 1);
        assert_eq!(
            out.edges, 1,
            "the auto-derived MENTIONS edge must be written: {}",
            out.message
        );

        let id = ops::id_for(&ont, "Fact", "x");
        let mentions: Vec<_> = g
            .outgoing(&id)
            .unwrap()
            .into_iter()
            .filter(|e| e.edge_type == glossa::graph::MENTIONS)
            .collect();
        assert_eq!(mentions.len(), 1, "grounded from source_path: {mentions:?}");
        assert_eq!(mentions[0].to, "d.md#1");
    }

    /// Step 1 of the task: an undeclared-type node is DROPPED with a reason naming the bad type,
    /// while a valid `Fact` sibling in the SAME batch still writes — partial-apply, not the old
    /// all-or-nothing `parse_and_validate_upsert` behavior (which rejected the whole call).
    #[test]
    fn extract_drops_undeclared_type_keeps_valid_fact_sibling() {
        use glossa::graph::store::GraphStore;

        let dir = tempfile::tempdir().unwrap();
        let ont = flat_ontology();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        write_doc(&idx, "d.md");

        let call = serde_json::json!({
            "nodes": [
                {"node_type": "Fact", "label": "good fact", "source_path": "d.md#1"},
                {"node_type": "Bogus", "label": "bad node", "source_path": "d.md#1"}
            ],
            "edges": []
        });
        let (nodes, edges, notes) = parse_and_filter_upsert(&call, &ont);
        assert_eq!(nodes.len(), 1, "only the Fact node survives the type filter");
        assert_eq!(nodes[0].label, "good fact");
        assert!(
            notes.iter().any(|n| n.contains("Bogus")),
            "the drop reason must name the offending type: {notes:?}"
        );

        let out = ops::graph_upsert(&idx, &g, &ont, nodes, edges, 1);
        assert!(
            !out.rejected,
            "the valid Fact sibling must still write: {}",
            out.message
        );
        assert_eq!(out.nodes, 1);
        let id = ops::id_for(&ont, "Fact", "good fact");
        assert!(
            g.get_node(&id).unwrap().is_some(),
            "the good Fact must actually be in the graph"
        );
    }

    /// Regression test for the ids-fix: mirrors `extract_doc`'s `graph_upsert` exec arm (parse +
    /// filter -> `ops::graph_upsert`, returning the written node ids as the surfaced ids)
    /// directly against `run_agent_loop`, with no live model. Before the fix (and before the
    /// reroute) this arm could return `Vec::new()` for ids, so N+1 consecutive DISTINCT
    /// `graph_upsert` calls (very plausible — several facts from one read) each looked
    /// "unproductive" to the streak detector; by the (K+1)th call its real "upserted ..."
    /// confirmation would have been replaced by the generic steer. Mirrors
    /// `openai::tests::loop_unproductive_streak_never_fires_when_calls_are_productive`.
    #[test]
    fn loop_distinct_graph_upsert_calls_never_trip_unproductive_streak() {
        use crate::backend::openai::{run_agent_loop, UNPRODUCTIVE_STREAK_K};
        use glossa::graph::store::GraphStore;
        use std::cell::RefCell;

        let dir = tempfile::tempdir().unwrap();
        let ont = flat_ontology();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        write_doc(&idx, "d.md");

        // Same shape as extract_doc's graph_upsert arm: parse + filter -> ops::graph_upsert,
        // returning the written node ids as the surfaced ids (the fix under test).
        let exec = |_name: &str, args: &Value| -> (String, Vec<String>, Vec<DocImage>) {
            let (nodes, edges, _notes) = parse_and_filter_upsert(args, &ont);
            let out = ops::graph_upsert(&idx, &g, &ont, nodes, edges, 1);
            let ids = upserted_node_ids(&out);
            (out.message, ids, Vec::new())
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

    /// The pin's whole point: extraction is fixed to `Fact` regardless of what the corpus's
    /// ontology declares. Here the ontology is a STRICT typed one that doesn't even declare
    /// `Fact` (only `Symptom`) — `Fact` still validates because Task 1 made it always-permitted
    /// by `Ontology::validate_node`, independent of `strict` or the declared entity set. Through
    /// the NEW ops path now (`parse_and_filter_upsert`'s type filter uses the same
    /// `Ontology::validate_node` call): the Fact node must survive the filter, not be dropped.
    #[test]
    fn extract_accepts_fact_under_strict_typed_ontology() {
        let ont = Ontology::parse("[entities.Symptom]\n[validation]\nstrict=true\n").unwrap();
        let call = serde_json::json!({
            "nodes":[{"node_type":"Fact","label":"x","source_path":"d.md"}],"edges":[]
        });
        let (nodes, _edges, notes) = parse_and_filter_upsert(&call, &ont);
        assert_eq!(nodes.len(), 1, "Fact must survive the type filter: {notes:?}");
        assert!(notes.is_empty(), "{notes:?}");
    }

    // --- harvest filter: keeps only grounding-required types --------

    /// Test helper: construct a minimal `ops::UpsertNode` for testing.
    fn mk_node(ty: &str, label: &str, src: &str) -> ops::UpsertNode {
        ops::UpsertNode {
            node_type: ty.into(),
            label: label.into(),
            source_path: src.into(),
            aliases: vec![],
            valid_from: None,
            valid_to: None,
        }
    }

    /// Task 3: harvest keeps ONLY `requires_grounding` node types; non-grounding types
    /// are dropped with a reason naming the type.
    #[test]
    fn harvest_keeps_only_grounding_types() {
        let ont = Ontology::parse(r#"
[entities.Res]
requires_grounding = true
[entities.Sym]
[validation]
strict = true
"#).unwrap();
        let nodes = vec![
            mk_node("Res", "r1", "doc#0"),
            mk_node("Sym", "s1", "doc#1"),
        ];
        let (kept, notes) = filter_grounding_only(nodes, &ont);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].node_type, "Res");
        assert!(notes.iter().any(|n| n.contains("Sym")), "dropped-type note missing");
    }

    // --- vision: `--vision` gates images out of (or into) `extract_doc`'s exec closure ---------

    fn stub_image() -> DocImage {
        DocImage {
            mime: "image/jpeg".to_string(),
            bytes: vec![1, 2, 3],
        }
    }

    /// `--vision` off (the default): a tool call's images are discarded — same as today, before
    /// this task, when `extract_doc`'s exec arm always dropped `glossa_tools::exec`'s `_images`.
    #[test]
    fn gate_images_vision_off_discards() {
        let out = gate_images(false, vec![stub_image()]);
        assert!(out.is_empty(), "vision off must discard images: {out:?}");
    }

    /// `--vision` on: images pass through unchanged, for `run_agent_loop` to thread into a vision
    /// user message (see `openai::vision_user_message`).
    #[test]
    fn gate_images_vision_on_passes_through() {
        let img = stub_image();
        let out = gate_images(true, vec![img.clone()]);
        assert_eq!(out, vec![img], "vision on must pass images through unchanged");
    }

    /// Task 2: build-path tool schema has only `read` + `graph_upsert`, no search/grep.
    #[test]
    fn build_tools_schema_has_no_search_or_grep() {
        let v = build_tools_schema("desc");
        let names: Vec<String> = v.as_array().unwrap().iter()
            .map(|t| t["function"]["name"].as_str().unwrap().to_string()).collect();
        assert!(names.contains(&"read".to_string()));
        assert!(names.contains(&"graph_upsert".to_string()));
        assert!(!names.contains(&"search".to_string()), "search must be off on build path");
        assert!(!names.contains(&"grep".to_string()), "grep must be off on build path");
    }
}
