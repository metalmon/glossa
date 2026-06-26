use crate::graph::ontology::Ontology;
use crate::graph::store::{normalize_label, Edge, GraphStore, Node, Provenance};
use crate::index::manifest::FileSig;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct NodeSpec {
    pub id: String,
    pub node_type: String,
    pub label: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub source_path: String,
    #[serde(default)]
    pub range: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct EdgeSpec {
    pub from: String,
    pub to: String,
    pub edge_type: String,
    pub source_path: String,
    #[serde(default)]
    pub range: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
}

fn stat_sig(path: &str) -> Option<FileSig> {
    let md = std::fs::metadata(path).ok()?;
    let mtime_secs = md
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    Some(FileSig { mtime_secs, size: md.len() })
}

/// Convert agent-supplied specs into provenance-stamped graph elements
/// (origin="agent", file_sig from the current source file, created_at=`now`),
/// validate against the ontology, and upsert. Deduplicates nodes by normalized
/// label+type so the graph converges across cases. Returns (#nodes written, #edges).
pub fn apply_upsert(
    g: &GraphStore,
    ont: &Ontology,
    nodes: Vec<NodeSpec>,
    edges: Vec<EdgeSpec>,
    now: u64,
) -> anyhow::Result<(usize, usize)> {
    use std::collections::HashMap;

    let prov = |source_path: &str, range: Option<String>, confidence: Option<f32>| Provenance {
        source_path: source_path.to_string(),
        range,
        file_sig: stat_sig(source_path),
        origin: "agent".into(),
        confidence: confidence.unwrap_or(0.8),
        created_at: now,
    };

    // Build canonical-id map: nodes with same normalized label+type converge to one id.
    let mut canonical: HashMap<String, String> = HashMap::new();
    for n in &nodes {
        let mut canon = n.id.clone();
        // 1. Check earlier nodes in this batch first.
        'batch: for earlier in &nodes {
            if earlier.id == n.id {
                break 'batch;
            }
            if earlier.node_type == n.node_type
                && normalize_label(&earlier.label) == normalize_label(&n.label)
            {
                canon = canonical.get(&earlier.id).cloned().unwrap_or_else(|| earlier.id.clone());
                break 'batch;
            }
        }
        // 2. If still unmerged, check the persistent graph.
        if canon == n.id {
            if let Some(existing_id) = g.find_by_label_type(&n.label, &n.node_type)? {
                if existing_id != n.id {
                    canon = existing_id;
                }
            }
        }
        canonical.insert(n.id.clone(), canon);
    }

    // Only create nodes whose canonical id is their own id (genuinely new ones).
    let model_nodes: Vec<Node> = nodes
        .iter()
        .filter(|n| canonical.get(&n.id).map(|c| c == &n.id).unwrap_or(true))
        .map(|n| {
            let p = prov(&n.source_path, n.range.clone(), n.confidence);
            Node {
                id: n.id.clone(),
                node_type: n.node_type.clone(),
                label: n.label.clone(),
                aliases: n.aliases.clone(),
                prov: p,
            }
        })
        .collect();

    // Rewrite edges: remap from/to through canonical map; unknown ids pass through unchanged.
    let model_edges: Vec<Edge> = edges
        .into_iter()
        .map(|e| {
            let from = canonical.get(&e.from).cloned().unwrap_or(e.from);
            let to = canonical.get(&e.to).cloned().unwrap_or(e.to);
            let p = prov(&e.source_path, e.range.clone(), e.confidence);
            Edge { from, to, edge_type: e.edge_type, prov: p }
        })
        .collect();

    g.upsert(ont, &model_nodes, &model_edges)?;
    Ok((model_nodes.len(), model_edges.len()))
}

/// Label-based reference to an edge (from/to are human-readable labels, not internal ids).
pub struct EdgeRef {
    pub from: String,
    pub edge_type: String,
    pub to: String,
}

/// Delete nodes and/or edges by label. Node deletion also removes all attached edges.
/// Returns total entries (nodes + edges) removed.
pub fn apply_delete(
    g: &GraphStore,
    node_labels: Vec<String>,
    edges: Vec<EdgeRef>,
) -> anyhow::Result<usize> {
    let mut total = 0;

    // Delete every node whose normalized label matches any of the given labels.
    for label in &node_labels {
        let norm = normalize_label(label);
        let matching: Vec<String> = g
            .all_nodes()?
            .into_iter()
            .filter(|n| normalize_label(&n.label) == norm)
            .map(|n| n.id)
            .collect();
        for id in matching {
            total += g.delete_node(&id)?;
        }
    }

    // Delete individual edges identified by label-pairs.
    for er in &edges {
        let from_norm = normalize_label(&er.from);
        let to_norm = normalize_label(&er.to);
        let all = g.all_nodes()?;
        let from_id = all.iter().find(|n| normalize_label(&n.label) == from_norm).map(|n| n.id.clone());
        let to_id = all.iter().find(|n| normalize_label(&n.label) == to_norm).map(|n| n.id.clone());
        if let (Some(f), Some(t)) = (from_id, to_id) {
            total += g.delete_edge(&f, &er.edge_type, &t)?;
        }
    }

    Ok(total)
}

/// Spec for an in-place node edit: change label and/or type while keeping the id and all edges.
pub struct NodeUpdate {
    pub label: String,
    pub new_label: Option<String>,
    pub new_type: Option<String>,
}

/// Rename and/or retype nodes in place, identified by their current label.
/// Skips nodes whose label is not found. Returns the total number of rows updated.
pub fn apply_update(g: &GraphStore, nodes: Vec<NodeUpdate>) -> anyhow::Result<usize> {
    let mut total = 0;
    for u in nodes {
        if let Some(id) = g.find_by_label(&u.label)? {
            total += g.update_node(&id, u.new_label.as_deref(), u.new_type.as_deref())?;
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONT: &str = r#"
[entities.Organization]
props = ["name"]
[relations.PARTY_TO]
from = ["Organization"]
to = ["Document"]
[validation]
strict = true
"#;

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
        NodeSpec { id: id.into(), node_type: ty.into(), label: label.into(), aliases: vec![], source_path: src.into(), range: None, confidence: None }
    }

    fn edge_spec(from: &str, to: &str, edge_type: &str, src: &str) -> EdgeSpec {
        EdgeSpec { from: from.into(), to: to.into(), edge_type: edge_type.into(), source_path: src.into(), range: None, confidence: None }
    }

    #[test]
    fn applies_validated_agent_nodes_with_provenance() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(ONT).unwrap();
        let nodes = vec![
            node("org:acme", "Organization", "Acme", "contract.docx"),
            node("contract.docx", "Document", "contract.docx", "contract.docx"),
        ];
        let edges = vec![EdgeSpec {
            from: "org:acme".into(), to: "contract.docx".into(), edge_type: "PARTY_TO".into(),
            source_path: "contract.docx".into(), range: None, confidence: Some(0.9),
        }];
        let (n, e) = apply_upsert(&g, &ont, nodes, edges, 123).unwrap();
        assert_eq!((n, e), (2, 1));
        assert_eq!(g.node_count().unwrap(), 2);
        let org = g.get_node("org:acme").unwrap().unwrap();
        assert_eq!(org.prov.origin, "agent");
        assert_eq!(org.prov.created_at, 123);
    }

    #[test]
    fn rejects_undeclared_type_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(ONT).unwrap();
        let nodes = vec![node("x", "Alien", "x", "d.docx")];
        assert!(apply_upsert(&g, &ont, nodes, vec![], 1).is_err());
        assert_eq!(g.node_count().unwrap(), 0);
    }

    #[test]
    fn dedup_merges_same_label_nodes() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();

        // First upsert: a Symptom + Resolution + RESOLVED_BY
        let nodes1 = vec![
            node("sym:fibas-loss-1", "Symptom", "Профибас потеря связи", "case1.docx"),
            node("res:restart-1", "Resolution", "Перезагрузка модуля", "case1.docx"),
        ];
        let edges1 = vec![edge_spec("sym:fibas-loss-1", "res:restart-1", "RESOLVED_BY", "case1.docx")];
        apply_upsert(&g, &ont, nodes1, edges1, 1).unwrap();

        // Second upsert: DIFFERENT id but SAME label (different case + extra space) → dedup
        let nodes2 = vec![
            node("sym:fibas-loss-2", "Symptom", "профибас  потеря связи", "case2.docx"),
            node("res:check-cable-2", "Resolution", "Проверка кабеля", "case2.docx"),
        ];
        let edges2 = vec![edge_spec("sym:fibas-loss-2", "res:check-cable-2", "RESOLVED_BY", "case2.docx")];
        apply_upsert(&g, &ont, nodes2, edges2, 2).unwrap();

        // Only 1 Symptom node (deduped — first id wins)
        let all = g.all_nodes().unwrap();
        let symptoms: Vec<_> = all.iter().filter(|n| n.node_type == "Symptom").collect();
        assert_eq!(symptoms.len(), 1, "expected exactly 1 Symptom after dedup");
        let symptom_id = &symptoms[0].id;
        assert_eq!(symptom_id, "sym:fibas-loss-1");

        // The second RESOLVED_BY edge must have been rewritten to originate from the first Symptom's id
        let out = g.outgoing(symptom_id).unwrap();
        assert_eq!(out.len(), 2, "expected 2 outgoing edges from the deduplicated Symptom");
        let has_check_cable = out.iter().any(|e| e.to == "res:check-cable-2" && e.edge_type == "RESOLVED_BY");
        assert!(has_check_cable, "second RESOLVED_BY edge should point from first symptom id to res:check-cable-2");
    }

    #[test]
    fn apply_delete_removes_node_and_edges() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();

        // Build a small Symptom→Resolution graph
        let nodes = vec![
            node("sym:test", "Symptom", "Тестовый симптом", "test.docx"),
            node("res:test", "Resolution", "Тестовое решение", "test.docx"),
        ];
        let edges = vec![edge_spec("sym:test", "res:test", "RESOLVED_BY", "test.docx")];
        apply_upsert(&g, &ont, nodes, edges, 1).unwrap();

        assert_eq!(g.node_count().unwrap(), 2);
        assert_eq!(g.edge_count().unwrap(), 1);

        // Delete the Symptom by label
        apply_delete(&g, vec!["Тестовый симптом".into()], vec![]).unwrap();

        // Symptom node is gone
        let all = g.all_nodes().unwrap();
        assert!(all.iter().all(|n| n.node_type != "Symptom"), "Symptom node should be deleted");

        // Its RESOLVED_BY edge is also gone
        let out = g.outgoing("sym:test").unwrap();
        assert!(out.is_empty(), "Edges attached to deleted Symptom should be removed");
    }
}
