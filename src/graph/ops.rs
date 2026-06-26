//! Shared write operations for the reasoning graph.
//!
//! Both the MCP server (`src/mcp.rs`) and the kb-eval enricher (`eval/src/enrich.rs`)
//! funnel through these functions so validation and resolution behaviour is identical.

use crate::graph::agent::{apply_delete, apply_update, apply_upsert, EdgeRef, EdgeSpec, NodeSpec, NodeUpdate};
use crate::graph::ontology::Ontology;
use crate::graph::store::GraphStore;
use crate::index::store::DocIndex;

// ── label helpers ─────────────────────────────────────────────────────────────

/// Resolve a section reference of the form `<path>#<n>` (a chunk number the agent has from a
/// read/search result) into the real Section node id `<path>#<location>`. Reasoning-node slugs
/// (`sym:…`) and already-resolved `<path>#<location>` refs (non-numeric suffix) pass through.
/// A numeric chunk ord that does NOT exist in the document is REJECTED with an actionable message
/// — so the model re-anchors to a real chunk instead of writing a dangling edge.
fn resolve_section_ref(idx: &DocIndex, s: &str) -> Result<String, String> {
    if let Some(pos) = s.rfind('#') {
        let suffix = &s[pos + 1..];
        if let Ok(n) = suffix.parse::<u64>() {
            let path = &s[..pos];
            return match idx.location_for_ord(path, n) {
                Ok(Some(loc)) => Ok(crate::graph::build::section_id(path, &loc)),
                Ok(None) => Err(format!(
                    "chunk #{n} does not exist in {path}; take the chunk number from a search/grep/read on THIS document — never reuse a number from another file"
                )),
                Err(e) => Err(format!("could not resolve chunk #{n} in {path}: {e}")),
            };
        }
    }
    Ok(s.to_string())
}

/// Label discipline (language-agnostic): a label should be a concise, broad class — a
/// generalisable problem/fix pattern, not the specifics of one case. The label's *language* is a
/// deployment concern stated in the agent prompt, not enforced here. Returns the list of
/// violations for `id`/`label`, empty when the label is fine.
fn check_label(id: &str, label: &str) -> Vec<String> {
    let mut v = Vec::new();
    if label.split_whitespace().count() > 12 {
        v.push(format!(
            "node '{id}': label has >12 words — generalise to a broad problem class, not the case specifics: \"{label}\""
        ));
    }
    v
}

// ── public API ────────────────────────────────────────────────────────────────

/// Outcome of a `graph_upsert` operation.
pub struct UpsertOutcome {
    /// Success summary or rejection text the model should act on.
    pub message: String,
    pub nodes: usize,
    pub edges: usize,
    /// True when the call was rejected (nothing was written).
    pub rejected: bool,
    /// Human-readable dump lines for logging:
    /// `"node <id> [<type>] <label>"` and `"edge <from> -<type>-> <to>"` (resolved).
    pub dump: Vec<String>,
}

/// Validate, resolve, and apply a graph upsert.
///
/// Behaviour is identical for the MCP server and the kb-eval enricher:
/// 1. `check_label` for each node.
/// 2. Build `batch_ids` from this call's nodes.
/// 3. For each edge, resolve `from`/`to` via `resolve_section_ref`; existence-check
///    each endpoint against `batch_ids` or the live graph.
/// 4. Build dump lines.
/// 5. If any errors: return a rejection `UpsertOutcome` (nothing written).
/// 6. Otherwise call `apply_upsert` and return the result.
pub fn graph_upsert(
    idx: &DocIndex,
    g: &GraphStore,
    ont: &Ontology,
    nodes: Vec<NodeSpec>,
    mut edges: Vec<EdgeSpec>,
    now: u64,
) -> UpsertOutcome {
    let mut errs: Vec<String> = Vec::new();

    // (1) label checks
    for nd in &nodes {
        errs.extend(check_label(&nd.id, &nd.label));
    }

    // (2) batch ids — an edge may reference a node from this same call before it is committed
    let batch_ids: std::collections::HashSet<String> =
        nodes.iter().map(|n| n.id.clone()).collect();

    // (3) resolve section refs + existence-check endpoints
    for e in &mut edges {
        let (of, ot, oet) = (e.from.clone(), e.to.clone(), e.edge_type.clone());
        let mut edge_ok = true;
        match resolve_section_ref(idx, &e.from) {
            Ok(r) => e.from = r,
            Err(m) => {
                errs.push(format!("edge {of} -{oet}-> {ot}: {m}"));
                edge_ok = false;
            }
        }
        match resolve_section_ref(idx, &e.to) {
            Ok(r) => e.to = r,
            Err(m) => {
                errs.push(format!("edge {of} -{oet}-> {ot}: {m}"));
                edge_ok = false;
            }
        }
        if edge_ok {
            for (role, id) in [("from", e.from.clone()), ("to", e.to.clone())] {
                let exists = batch_ids.contains(&id)
                    || g.get_node(&id).ok().flatten().is_some();
                if !exists {
                    errs.push(format!(
                        "edge {of} -{oet}-> {ot}: {role} endpoint '{id}' is not a known node — create that node (add it to nodes[]) before referencing it"
                    ));
                }
            }
        }
    }

    // (4) build dump lines (resolved state, for the caller to log)
    let mut dump = Vec::new();
    for nd in &nodes {
        dump.push(format!("node {} [{}] {}", nd.id, nd.node_type, nd.label));
    }
    for e in &edges {
        dump.push(format!("edge {} -{}-> {}", e.from, e.edge_type, e.to));
    }

    // (5) rejection path — nothing is written
    if !errs.is_empty() {
        return UpsertOutcome {
            message: format!(
                "graph_upsert REJECTED — nothing was written. Fix these and resend:\n- {}",
                errs.join("\n- ")
            ),
            nodes: 0,
            edges: 0,
            rejected: true,
            dump,
        };
    }

    // (6) apply
    match apply_upsert(g, ont, nodes, edges, now) {
        Ok((n, e)) => UpsertOutcome {
            message: format!("upserted {n} nodes, {e} edges"),
            nodes: n,
            edges: e,
            rejected: false,
            dump,
        },
        Err(e) => UpsertOutcome {
            message: e.to_string(),
            nodes: 0,
            edges: 0,
            rejected: true,
            dump,
        },
    }
}

/// Delete reasoning nodes and/or edges by label.
/// Returns a human-readable result string.
pub fn graph_delete(g: &GraphStore, node_labels: Vec<String>, edges: Vec<EdgeRef>) -> String {
    match apply_delete(g, node_labels, edges) {
        Ok(n) => format!("deleted {n} graph entries"),
        Err(e) => format!("graph_delete error: {e}"),
    }
}

/// Edit existing reasoning nodes in place — rename label and/or change type — by current label.
/// The node id and all its edges are preserved. Returns a human-readable result string.
pub fn graph_update(g: &GraphStore, nodes: Vec<NodeUpdate>) -> String {
    match apply_update(g, nodes) {
        Ok(n) => format!("updated {n} nodes"),
        Err(e) => format!("graph_update error: {e}"),
    }
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::agent::NodeUpdate;

    const DEDUP_ONT: &str = r#"
[entities.Symptom]
props = []
[entities.Resolution]
props = []
[relations.RESOLVED_BY]
from = ["Symptom"]
to = ["Resolution"]
[validation]
strict = true
"#;

    fn node(id: &str, ty: &str, label: &str, src: &str) -> NodeSpec {
        NodeSpec {
            id: id.into(),
            node_type: ty.into(),
            label: label.into(),
            aliases: vec![],
            source_path: src.into(),
            range: None,
            confidence: None,
        }
    }

    fn edge_spec(from: &str, to: &str, edge_type: &str, src: &str) -> EdgeSpec {
        EdgeSpec {
            from: from.into(),
            to: to.into(),
            edge_type: edge_type.into(),
            source_path: src.into(),
            range: None,
            confidence: None,
        }
    }

    /// Happy path: Symptom + Resolution + RESOLVED_BY in one batch — accepted and written.
    #[test]
    fn happy_path_symptom_resolution_resolved_by() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();

        let nodes = vec![
            node("sym:loss-1", "Symptom", "Потеря связи", "case1.docx"),
            node("res:restart-1", "Resolution", "Перезагрузка модуля", "case1.docx"),
        ];
        let edges = vec![edge_spec("sym:loss-1", "res:restart-1", "RESOLVED_BY", "case1.docx")];

        let out = graph_upsert(&idx, &g, &ont, nodes, edges, 1_000_000);
        assert!(!out.rejected, "should not be rejected: {}", out.message);
        assert_eq!(out.nodes, 2);
        assert_eq!(out.edges, 1);
        assert!(out.message.contains("upserted 2 nodes, 1 edges"), "{}", out.message);
    }

    /// graph_update renames a node in place while preserving its id and all outgoing edges.
    #[test]
    fn update_renames_node_keeps_id_and_edges() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();

        // Upsert: Symptom + Resolution + RESOLVED_BY edge.
        let nodes = vec![
            node("sym:loss-1", "Symptom", "Потеря связи", "case1.docx"),
            node("res:restart-1", "Resolution", "Перезагрузка модуля", "case1.docx"),
        ];
        let edges = vec![edge_spec("sym:loss-1", "res:restart-1", "RESOLVED_BY", "case1.docx")];
        graph_upsert(&idx, &g, &ont, nodes, edges, 1);

        // Rename the Symptom node.
        let ups = vec![NodeUpdate {
            label: "Потеря связи".into(),
            new_label: Some("Нестабильная связь".into()),
            new_type: None,
        }];
        let result = graph_update(&g, ups);
        assert_eq!(result, "updated 1 nodes", "unexpected result: {result}");

        // (a) a node with the new label exists
        let id = g.find_by_label("Нестабильная связь").unwrap();
        assert!(id.is_some(), "new label not found");
        let id = id.unwrap();

        // (b) the id is the same as before
        assert_eq!(id, "sym:loss-1", "id changed after rename");

        // (c) the RESOLVED_BY edge still exists
        let out = g.outgoing(&id).unwrap();
        assert!(
            out.iter().any(|e| e.edge_type == "RESOLVED_BY" && e.to == "res:restart-1"),
            "RESOLVED_BY edge lost after rename"
        );
    }

    /// Rejection: an edge whose `to` endpoint (`res:`) is neither in the batch nor in the graph.
    #[test]
    fn rejects_edge_to_unknown_node() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();

        // Only the Symptom node — Resolution is missing from the batch and from the graph.
        let nodes = vec![node("sym:loss-1", "Symptom", "Потеря связи", "case1.docx")];
        let edges = vec![edge_spec(
            "sym:loss-1",
            "res:restart-1",
            "RESOLVED_BY",
            "case1.docx",
        )];

        let out = graph_upsert(&idx, &g, &ont, nodes, edges, 1_000_000);
        assert!(out.rejected, "should be rejected");
        assert!(
            out.message.contains("not a known node"),
            "message should mention 'not a known node': {}",
            out.message
        );
        assert_eq!(out.nodes, 0);
        assert_eq!(out.edges, 0);
    }
}
