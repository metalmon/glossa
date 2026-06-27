//! Shared write operations for the reasoning graph.
//!
//! Both the MCP server (`src/mcp.rs`) and the kb-eval enricher (`eval/src/enrich.rs`)
//! funnel through these functions so validation and resolution behaviour is identical.

use crate::graph::agent::{apply_delete, apply_update, apply_upsert, EdgeRef, EdgeSpec, NodeSpec, NodeUpdate};
use crate::graph::ontology::Ontology;
use crate::graph::store::{GraphStore, normalize_label};
use crate::index::store::DocIndex;

// ── label-based input types ───────────────────────────────────────────────────

/// A reasoning node to create/update, identified by label only (no id).
/// The system derives the canonical id from `node_type` + `label` and deduplicates by label.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct UpsertNode {
    pub node_type: String,
    pub label: String,
    pub source_path: String,
    #[serde(default)]
    pub aliases: Vec<String>,
}

/// A directed edge identified by node labels (or section refs) at both endpoints.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct UpsertEdge {
    pub from: String,
    pub edge_type: String,
    pub to: String,
    pub source_path: String,
}

/// Derive the canonical node id from its type and label.
/// Abbrevs: Symptom→"sym", Cause→"cau", Resolution→"res", Task→"tsk", else lowercased type.
/// The label is normalised (lowercase, collapsed whitespace) and spaces replaced with "-".
pub fn id_for(node_type: &str, label: &str) -> String {
    let slug = normalize_label(label).replace(' ', "-");
    let abbrev = match node_type {
        "Symptom" => "sym".to_string(),
        "Cause" => "cau".to_string(),
        "Resolution" => "res".to_string(),
        "Task" => "tsk".to_string(),
        other => other.to_lowercase(),
    };
    format!("{abbrev}:{slug}")
}

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

/// Resolve an edge endpoint label to a node id: exact normalized-label match first, then a fuzzy
/// morphology match against existing reasoning nodes (the small model often paraphrases its own
/// label — a truncation or wording variant). Returns None when nothing matches.
fn resolve_endpoint_label(
    g: &GraphStore,
    label_to_id: &std::collections::HashMap<String, String>,
    label: &str,
) -> Option<String> {
    if let Some(id) = label_to_id.get(&normalize_label(label)) {
        return Some(id.clone());
    }
    // Exact match against EXISTING nodes via the label_norm index (replaces the old prebuilt
    // all_nodes map — same unfiltered "first exact" semantics, but O(log N) instead of O(N)).
    if let Some(id) = g.ids_by_label_norm(label).ok()?.into_iter().next() {
        return Some(id);
    }
    const STRUCTURAL: &[&str] = &["Document", "Section", "Term", "Topic"];
    let ids = g.resolve(label).ok()?;
    ids.into_iter()
        .filter_map(|id| g.get_node(&id).ok().flatten())
        .filter(|n| !STRUCTURAL.contains(&n.node_type.as_str()))
        // all morphology matches contain the query's terms; the SHORTEST label is the closest
        // superset of the model's (often truncated) reference.
        .min_by_key(|n| n.label.split_whitespace().count())
        .map(|n| n.id)
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
/// The caller provides node LABELS only (no ids). The system derives canonical ids via
/// `id_for(node_type, label)` and deduplicates by label. Edges reference nodes by their label
/// (or a section as `<path>#<n>`).
///
/// Steps:
/// 1. `check_label` per node.
/// 2. Build `label_to_id` map: input nodes first (win), then existing graph nodes (`or_insert`).
/// 3. Build `Vec<NodeSpec>` using `id_for`.
/// 4. Resolve each `UpsertEdge` endpoint: section ref → resolved id; otherwise label → id via map.
/// 5. Build dump lines.
/// 6. If any errors: return rejection (nothing written), keeping the hint block.
/// 7. Otherwise call `apply_upsert` and return.
pub fn graph_upsert(
    idx: &DocIndex,
    g: &GraphStore,
    ont: &Ontology,
    nodes: Vec<UpsertNode>,
    edges: Vec<UpsertEdge>,
    now: u64,
) -> UpsertOutcome {
    let mut errs: Vec<String> = Vec::new();

    // (1) label checks
    for nd in &nodes {
        let id = id_for(&nd.node_type, &nd.label);
        errs.extend(check_label(&id, &nd.label));
    }

    // (2) label_to_id: ONLY the input (batch) nodes — they win over existing graph nodes. Existing
    // nodes are no longer loaded wholesale (that was O(N) per upsert); an edge endpoint naming an
    // existing node is resolved on demand via the label_norm index in `resolve_endpoint_label`.
    let mut label_to_id: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for nd in &nodes {
        label_to_id.insert(normalize_label(&nd.label), id_for(&nd.node_type, &nd.label));
    }

    // (3) build NodeSpec list
    let nodespecs: Vec<NodeSpec> = nodes
        .iter()
        .map(|nd| NodeSpec {
            id: id_for(&nd.node_type, &nd.label),
            node_type: nd.node_type.clone(),
            label: nd.label.clone(),
            aliases: nd.aliases.clone(),
            source_path: nd.source_path.clone(),
            range: None,
            confidence: None,
        })
        .collect();

    // batch ids — an edge may reference a node from this same call before it is committed
    let batch_ids: std::collections::HashSet<String> =
        nodespecs.iter().map(|n| n.id.clone()).collect();

    // (4) resolve edges
    let mut edgespecs: Vec<EdgeSpec> = Vec::new();
    for ue in &edges {
        let (of, ot, oet) = (ue.from.clone(), ue.to.clone(), ue.edge_type.clone());
        let mut from_resolved: Option<String> = None;
        let mut to_resolved: Option<String> = None;
        let mut edge_ok = true;

        // resolve from endpoint
        match resolve_section_ref(idx, &ue.from) {
            Err(m) => {
                errs.push(format!("edge {of} -{oet}-> {ot}: {m}"));
                edge_ok = false;
            }
            Ok(v) if v != ue.from => {
                // numeric section ref resolved to section id
                from_resolved = Some(v);
            }
            Ok(_) => {
                // treat as node label (exact, then fuzzy morphology fallback)
                match resolve_endpoint_label(g, &label_to_id, &ue.from) {
                    Some(id) => from_resolved = Some(id),
                    None => {
                        errs.push(format!(
                            "edge {of} -{oet}-> {ot}: from endpoint label \"{of}\" has no node — add a node with that label"
                        ));
                        edge_ok = false;
                    }
                }
            }
        }

        // resolve to endpoint
        match resolve_section_ref(idx, &ue.to) {
            Err(m) => {
                errs.push(format!("edge {of} -{oet}-> {ot}: {m}"));
                edge_ok = false;
            }
            Ok(v) if v != ue.to => {
                // numeric section ref resolved to section id
                to_resolved = Some(v);
            }
            Ok(_) => {
                // treat as node label (exact, then fuzzy morphology fallback)
                match resolve_endpoint_label(g, &label_to_id, &ue.to) {
                    Some(id) => to_resolved = Some(id),
                    None => {
                        errs.push(format!(
                            "edge {of} -{oet}-> {ot}: to endpoint label \"{ot}\" has no node — add a node with that label"
                        ));
                        edge_ok = false;
                    }
                }
            }
        }

        if edge_ok {
            let from_id = from_resolved.unwrap();
            let to_id = to_resolved.unwrap();

            // post-resolution existence check
            let mut exists_ok = true;
            for (role, id) in [("from", from_id.clone()), ("to", to_id.clone())] {
                let exists =
                    batch_ids.contains(&id) || g.get_node(&id).ok().flatten().is_some();
                if !exists {
                    errs.push(format!(
                        "edge {of} -{oet}-> {ot}: {role} endpoint '{id}' is not a known node — create that node (add it to nodes[]) before referencing it"
                    ));
                    exists_ok = false;
                }
            }

            if exists_ok {
                edgespecs.push(EdgeSpec {
                    from: from_id,
                    to: to_id,
                    edge_type: ue.edge_type.clone(),
                    source_path: ue.source_path.clone(),
                    range: None,
                    confidence: None,
                });
            }
        }
    }

    // (5) build dump lines (resolved state, for the caller to log)
    let mut dump = Vec::new();
    for nd in &nodespecs {
        dump.push(format!("node {} [{}] {}", nd.id, nd.node_type, nd.label));
    }
    for e in &edgespecs {
        dump.push(format!("edge {} -{}-> {}", e.from, e.edge_type, e.to));
    }

    // (6) rejection path — nothing is written
    if !errs.is_empty() {
        // Help the model recover from id confusion (it often references a node it created under a
        // DIFFERENT id, then loops on the rejection): list the reasoning nodes that already exist so
        // it can reference the right id instead of an invented one. Structural nodes are excluded.
        const STRUCTURAL: &[&str] = &["Document", "Section", "Term", "Topic"];
        let existing: Vec<String> = g
            .all_nodes()
            .unwrap_or_default()
            .into_iter()
            .filter(|n| !STRUCTURAL.contains(&n.node_type.as_str()))
            .map(|n| format!("{} [{}] {}", n.id, n.node_type, n.label))
            .take(25)
            .collect();
        let hint = if existing.is_empty() {
            String::new()
        } else {
            format!(
                "\nExisting reasoning nodes — reference one of THESE ids, do not invent a new id for the same concept:\n- {}",
                existing.join("\n- ")
            )
        };
        return UpsertOutcome {
            message: format!(
                "graph_upsert REJECTED — nothing was written. Fix these and resend:\n- {}{}",
                errs.join("\n- "),
                hint
            ),
            nodes: 0,
            edges: 0,
            rejected: true,
            dump,
        };
    }

    // (7) apply
    match apply_upsert(g, ont, nodespecs, edgespecs, now) {
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

    fn unode(node_type: &str, label: &str, src: &str) -> UpsertNode {
        UpsertNode {
            node_type: node_type.into(),
            label: label.into(),
            source_path: src.into(),
            aliases: vec![],
        }
    }

    fn uedge(from: &str, edge_type: &str, to: &str, src: &str) -> UpsertEdge {
        UpsertEdge {
            from: from.into(),
            edge_type: edge_type.into(),
            to: to.into(),
            source_path: src.into(),
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
            unode("Symptom", "Потеря связи", "case1.docx"),
            unode("Resolution", "Перезагрузка модуля", "case1.docx"),
        ];
        let edges = vec![uedge("Потеря связи", "RESOLVED_BY", "Перезагрузка модуля", "case1.docx")];

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
            unode("Symptom", "Потеря связи", "case1.docx"),
            unode("Resolution", "Перезагрузка модуля", "case1.docx"),
        ];
        let edges =
            vec![uedge("Потеря связи", "RESOLVED_BY", "Перезагрузка модуля", "case1.docx")];
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

        // (b) the id is the label-derived id for the original label
        let expected_id = id_for("Symptom", "Потеря связи");
        assert_eq!(id, expected_id, "id should be the label-derived id");

        // (c) the RESOLVED_BY edge still exists
        let res_id = id_for("Resolution", "Перезагрузка модуля");
        let out = g.outgoing(&id).unwrap();
        assert!(
            out.iter().any(|e| e.edge_type == "RESOLVED_BY" && e.to == res_id),
            "RESOLVED_BY edge lost after rename"
        );
    }

    /// Rejection: an edge whose `to` endpoint label is neither in the batch nor in the graph.
    #[test]
    fn rejects_edge_to_unknown_node() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();

        // Only the Symptom node — Resolution label is missing from batch and graph.
        let nodes = vec![unode("Symptom", "Потеря связи", "case1.docx")];
        let edges =
            vec![uedge("Потеря связи", "RESOLVED_BY", "Перезагрузка модуля", "case1.docx")];

        let out = graph_upsert(&idx, &g, &ont, nodes, edges, 1_000_000);
        assert!(out.rejected, "should be rejected");
        assert!(
            out.message.contains("has no node"),
            "message should mention 'has no node': {}",
            out.message
        );
        assert_eq!(out.nodes, 0);
        assert_eq!(out.edges, 0);
    }

    /// Label-based upsert with no ids: ids are derived from labels, RESOLVED_BY edge is wired.
    #[test]
    fn label_based_upsert_no_ids() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();

        let nodes = vec![
            unode("Symptom", "Потеря связи", "case1.docx"),
            unode("Resolution", "Перезапуск", "case1.docx"),
        ];
        let edges = vec![uedge("Потеря связи", "RESOLVED_BY", "Перезапуск", "case1.docx")];

        let out = graph_upsert(&idx, &g, &ont, nodes, edges, 1_000_000);
        assert!(!out.rejected, "should not be rejected: {}", out.message);
        assert_eq!(out.nodes, 2);
        assert_eq!(out.edges, 1);

        // Assert the RESOLVED_BY edge connects the label-derived ids.
        let sym_id = id_for("Symptom", "Потеря связи");
        let res_id = id_for("Resolution", "Перезапуск");
        let outgoing = g.outgoing(&sym_id).unwrap();
        assert!(
            outgoing.iter().any(|e| e.edge_type == "RESOLVED_BY" && e.to == res_id),
            "RESOLVED_BY edge not found from {sym_id} to {res_id}: {outgoing:?}"
        );
    }

    /// Edge whose `to` label has no corresponding node is rejected; message names the label.
    #[test]
    fn edge_to_undefined_label_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();

        let nodes = vec![unode("Symptom", "Потеря связи", "case1.docx")];
        let edges = vec![uedge("Потеря связи", "RESOLVED_BY", "Неизвестный узел", "case1.docx")];

        let out = graph_upsert(&idx, &g, &ont, nodes, edges, 1_000_000);
        assert!(out.rejected, "should be rejected");
        assert!(
            out.message.contains("Неизвестный узел"),
            "message should name the undefined label: {}",
            out.message
        );
    }

    /// The model paraphrases its own label across calls — a truncated reference must resolve
    /// fuzzily (morphology) to the existing node instead of being rejected.
    #[test]
    fn edge_label_resolves_fuzzily_to_paraphrase() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();

        let out1 = graph_upsert(
            &idx, &g, &ont,
            vec![
                unode("Symptom", "Потеря связи Profibus", "c.docx"),
                unode("Resolution", "Изменение параметра maxTsdr и перезапуск службы", "c.docx"),
            ],
            vec![uedge("Потеря связи Profibus", "RESOLVED_BY", "Изменение параметра maxTsdr и перезапуск службы", "c.docx")],
            1,
        );
        assert!(!out1.rejected, "{}", out1.message);

        // Later edge references the Resolution by a TRUNCATED label.
        let out2 = graph_upsert(
            &idx, &g, &ont,
            vec![],
            vec![uedge("Потеря связи Profibus", "RESOLVED_BY", "Изменение параметра maxTsdr", "c.docx")],
            2,
        );
        assert!(!out2.rejected, "truncated label must resolve fuzzily: {}", out2.message);
    }
}
