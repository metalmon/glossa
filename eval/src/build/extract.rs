//! Task 5 — agentic STATED fact-extraction over a single document.
//!
//! `extract_doc` drives an orchestrated, bounded-round builder: each round starts at the first
//! uncovered chunk ordinal and reads a document's sections sequentially with `read` +
//! `graph_upsert` ONLY (no `search`/`grep` — the pass is a coverage walk, not a free-form search).
//! `graph_upsert` parses the model's call via the CANONICAL
//! `glossa::graph::ops::parse_upsert_payload` (label-based `UpsertNode`/`UpsertEdge` — no
//! agent-assigned ids) into label-referenced upserts, drops any node whose type the ontology
//! doesn't declare (partial — siblings still write), keeps ONLY the ontology's
//! `requires_grounding` node types via `filter_grounding_only` (ontology-typed harvest, not pinned
//! to a single `Fact` type), and writes the rest through `glossa::graph::ops::graph_upsert` — the
//! SAME write path the MCP server uses (validates, resolves edge endpoints by label, auto-grounds
//! from a `<path>#n` source_path, provenance-stamped, dedup-by-label).
//!
//! `extract_doc` drives the full agent loop and is exercised by a live-endpoint smoke run, not a
//! unit test; the `graph_upsert` exec arm's parse/filter/write behavior is unit-testable and
//! covered below without a live model.

use crate::backend::openai::{lmstudio_chat, run_agent_loop};
use crate::lab::LabConfig;
use crate::reason::grounding_schema_block;
use glossa::graph::ontology::Ontology;
use glossa::graph::ops;
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;
use glossa::read::DocImage;
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::BTreeSet;
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
/// can fix and resend just that item. The build harvest is ontology-typed (the model may use any
/// of the ontology's declared entity types, per `BUILD_GRAPH_UPSERT_DESC`), so this only fires on
/// a genuine model mistake (an undeclared type); every well-typed sibling node (and every edge —
/// `ops::graph_upsert` resolves edges by label, so an edge naming a dropped node simply fails to
/// resolve on its own, with its own actionable reason) still reaches the write path. The narrower
/// grounding-required-types-only rule is enforced separately by `filter_grounding_only`.
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
     `source_path` of `<path>#n` and grounding is derived automatically.";

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

/// Ordered list of a document's existing chunk ordinals (chunks are sparse — follow `.next`).
fn doc_chunk_ords(idx: &DocIndex, doc: &str) -> anyhow::Result<Vec<u64>> {
    let mut ords = Vec::new();
    // find the first existing ord at or after 0
    let mut cur = match idx.read_chunk_by_ord(doc, 0)? {
        Some(c) => {
            ords.push(0);
            c.next
        }
        None => {
            // scan forward a bounded window for the first chunk if 0 is absent
            let mut first = None;
            for n in 1..1000u64 {
                if let Some(c) = idx.read_chunk_by_ord(doc, n)? {
                    first = Some((n, c));
                    break;
                }
            }
            match first {
                Some((n, c)) => {
                    ords.push(n);
                    c.next
                }
                None => return Ok(ords),
            }
        }
    };
    while let Some(n) = cur {
        match idx.read_chunk_by_ord(doc, n)? {
            Some(c) => {
                ords.push(n);
                cur = c.next;
            }
            None => break,
        }
    }
    Ok(ords)
}

/// Run the orchestrated, bounded-round STATED-extraction pass over ONE document. Instead of a
/// single open-ended agent loop, the builder walks the document in SEQUENTIAL coverage rounds: a
/// round starts at the first not-yet-covered chunk ordinal and asks the model to `read` up to
/// `chunks_per_round` sections, then harvest — via `graph_upsert` — every grounded node those
/// sections STATE. `read` is the only navigation tool (no `search`/`grep` — the builder is pinned
/// to this one document); the round's read-count is bounded so the model can't wander, and the
/// coverage loop repeats until every ordinal has been visited.
///
/// Harvest keeps ONLY `requires_grounding` node types (see `filter_grounding_only`) — the
/// document side of the ontology (query/chain-side types are produced by `kbx reason`, not here).
/// `builder_md` is the system prompt verbatim, prefixed with `grounding_schema_block` (the
/// grounding-required node types + descriptions). Returns how many nodes/MENTIONS edges were
/// written across all rounds.
///
/// `build_temp` sets the sampling temperature (via `KB_EVAL_TEMP`, read by `lmstudio_chat`).
/// `vision`: `kbx build --vision` (default OFF) — `read` here surfaces text only, so images are
/// gated out via `gate_images`; the flag is threaded through for parity with the vision path.
pub fn extract_doc(
    root: &Path,
    lab: &LabConfig,
    builder_md: &str,
    ontology: &Ontology,
    doc_path: &str,
    build_temp: f64,
    chunks_per_round: usize,
    vision: bool,
) -> anyhow::Result<ExtractStats> {
    // lmstudio_chat reads the sampling temperature from KB_EVAL_TEMP; set it once for this pass.
    std::env::set_var("KB_EVAL_TEMP", build_temp.to_string());

    let g = GraphStore::open(root)?;
    let idx = DocIndex::open_or_create(root)?;

    // Grounding-required node types + descriptions prepended to the builder prompt (the schema the
    // harvest may create; `filter_grounding_only` enforces it on write).
    let system = format!("{}\n\n{builder_md}", grounding_schema_block(ontology));
    let tools = build_tools_schema(BUILD_GRAPH_UPSERT_DESC);

    let endpoint = lab.model.endpoint.clone();
    let model = lab.model.model.clone();
    let api_key = lab.model.resolve_key();
    let timeout = Duration::from_secs(lab.model.timeout_secs);

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let ords = doc_chunk_ords(&idx, doc_path)?;

    // `stats` accumulates across rounds; `reads` tracks the ordinals a single round actually read
    // (cleared per round). Both live in RefCell so the FnMut `exec` closure can mutate them without
    // conflicting with `chat`/`on_repeat`'s borrows inside `run_agent_loop`.
    let stats = RefCell::new(ExtractStats::default());
    let reads = RefCell::new(Vec::<u64>::new());

    // Coverage loop: one bounded round per pass, starting at the first uncovered ordinal, until
    // every existing chunk ordinal has been visited.
    let mut covered: BTreeSet<u64> = BTreeSet::new();
    while let Some(&start) = ords.iter().find(|o| !covered.contains(o)) {
        reads.borrow_mut().clear();

        let user = format!(
            "Extract grounded nodes from this document. Start at section {start}: call \
             read(path=\"{doc_path}\", n={start}) first."
        );
        let messages = vec![
            json!({ "role": "system", "content": system.clone() }),
            json!({ "role": "user", "content": user }),
        ];

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

        let exec = |name: &str, args: &Value| -> (String, Vec<String>, Vec<DocImage>) {
            if std::env::var("KB_EVAL_DUMP_TOOLS").is_ok() {
                eprintln!("[TOOL] {name} {args}");
            }
            if name == "graph_upsert" {
                // Canonical label-based parse (ops::parse_upsert_payload) + partial-apply type
                // filter, then keep ONLY grounding-required types (the document-harvest side of the
                // ontology), then write through the SAME ops::graph_upsert path the MCP server uses.
                let (nodes, edges, _notes) = parse_and_filter_upsert(args, ontology);
                let (nodes, _drop_notes) = filter_grounding_only(nodes, ontology);
                let out = ops::graph_upsert(&idx, &g, ontology, nodes, edges, now);
                if !out.rejected {
                    let mut s = stats.borrow_mut();
                    s.nodes += out.nodes;
                    // Count MENTIONS edges actually written (explicit + auto-derived from a node's
                    // `<path>#n` source_path) — read back from the outcome's dump, not the request.
                    let mentions_marker = format!("-{}->", glossa::graph::MENTIONS);
                    s.mentions += out
                        .dump
                        .iter()
                        .filter(|l| l.starts_with("edge ") && l.contains(&mentions_marker))
                        .count();
                }
                // Surfaced ids for the unproductive-streak novelty tracker: the ids of nodes this
                // call actually wrote or merged into (see `upserted_node_ids`), so a batch of
                // distinct writes registers as progress.
                let ids = upserted_node_ids(&out);
                (out.message, ids, Vec::new())
            } else {
                // `read` — the only navigation tool. Bound the round to `chunks_per_round` reads,
                // then steer the model to harvest and stop.
                let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(0);
                if reads.borrow().len() >= chunks_per_round {
                    return (
                        "(no more sections for this round — now call graph_upsert for every \
                         prescribed node you read, then stop)"
                            .into(),
                        vec![],
                        gate_images(vision, vec![]),
                    );
                }
                match idx.read_chunk_by_ord(doc_path, n) {
                    Ok(Some(c)) => {
                        reads.borrow_mut().push(n);
                        (c.body, vec![], gate_images(vision, vec![]))
                    }
                    Ok(None) => ("(no more sections)".into(), vec![], gate_images(vision, vec![])),
                    Err(e) => (format!("(read error: {e})"), vec![], gate_images(vision, vec![])),
                }
            }
        };

        let on_repeat = |name: &str, _args: &Value| {
            format!(
                "(dup {name}) you already called this — try a different tool, a different query, \
                 or move on to graph_upsert"
            )
        };

        run_agent_loop(chat, messages, exec, on_repeat, MAX_ROUNDS)?;

        // Advance coverage monotonically: `start` is unconditionally marked covered after its
        // round (regardless of what the round actually read), plus whatever ordinals the round
        // read. This guarantees strict progress even if the model reads zero sections, or reads
        // only already-covered ordinals but never `start` itself — either way the coverage loop
        // can't spin forever on the same `start`.
        covered.insert(start);
        covered.extend(reads.borrow().iter().copied());
    }

    Ok(stats.into_inner())
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

    // --- doc_chunk_ords: ordered enumeration of a document's existing (sparse) chunk ordinals ---

    /// Task 4: `write_chunks` assigns 1-based ordinals (see `index::store::write_chunks`'s
    /// `(i + 1) as u64`) — ord 0 never exists. Mirrors the fixture-building style of
    /// `backend::glossa_tools`'s tests (`DocIndex::open_or_create` + `write_chunks`).
    #[test]
    fn doc_chunk_ords_returns_ordinals_in_order() {
        use glossa::model::Chunk;
        use std::path::PathBuf;

        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            Chunk {
                doc_path: PathBuf::from("d.md"),
                location: "A".into(),
                file_type: "md".into(),
                text: "alpha".into(),
            },
            Chunk {
                doc_path: PathBuf::from("d.md"),
                location: "B".into(),
                file_type: "md".into(),
                text: "bravo".into(),
            },
            Chunk {
                doc_path: PathBuf::from("d.md"),
                location: "C".into(),
                file_type: "md".into(),
                text: "charlie".into(),
            },
        ])
        .unwrap();

        assert_eq!(doc_chunk_ords(&idx, "d.md").unwrap(), vec![1, 2, 3]);
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
