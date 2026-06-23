use crate::graph::ontology::Ontology;
use crate::graph::store::{Edge, GraphStore, Node, Provenance};
use crate::index::manifest::FileSig;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
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

#[derive(Debug, Clone, Deserialize, JsonSchema)]
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
/// validate against the ontology, and upsert. Returns (#nodes, #edges).
pub fn apply_upsert(
    g: &GraphStore,
    ont: &Ontology,
    nodes: Vec<NodeSpec>,
    edges: Vec<EdgeSpec>,
    now: u64,
) -> anyhow::Result<(usize, usize)> {
    let prov = |source_path: &str, range: Option<String>, confidence: Option<f32>| Provenance {
        source_path: source_path.to_string(),
        range,
        file_sig: stat_sig(source_path),
        origin: "agent".into(),
        confidence: confidence.unwrap_or(0.8),
        created_at: now,
    };
    let model_nodes: Vec<Node> = nodes
        .into_iter()
        .map(|n| {
            let p = prov(&n.source_path, n.range.clone(), n.confidence);
            Node { id: n.id, node_type: n.node_type, label: n.label, aliases: n.aliases, prov: p }
        })
        .collect();
    let model_edges: Vec<Edge> = edges
        .into_iter()
        .map(|e| {
            let p = prov(&e.source_path, e.range.clone(), e.confidence);
            Edge { from: e.from, to: e.to, edge_type: e.edge_type, prov: p }
        })
        .collect();
    g.upsert(ont, &model_nodes, &model_edges)?;
    Ok((model_nodes.len(), model_edges.len()))
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

    fn node(id: &str, ty: &str, label: &str, src: &str) -> NodeSpec {
        NodeSpec { id: id.into(), node_type: ty.into(), label: label.into(), aliases: vec![], source_path: src.into(), range: None, confidence: None }
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
}
