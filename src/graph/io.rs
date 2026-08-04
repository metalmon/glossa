use crate::graph::agent::{apply_upsert, EdgeSpec, NodeSpec};
use crate::graph::ontology::Ontology;
use crate::graph::store::GraphStore;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashSet};

const STRUCTURAL: &[&str] = &["Document", "Section", "Term", "Topic"];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphExport {
    pub exported_types: Vec<String>,
    pub nodes: Vec<NodeSpec>,
    pub edges: Vec<EdgeSpec>,
}

pub fn collect(g: &GraphStore, type_filter: Option<&str>) -> anyhow::Result<GraphExport> {
    let mut all = g.all_nodes()?;
    all.retain(|n| {
        !STRUCTURAL.contains(&n.node_type.as_str()) && type_filter.is_none_or(|t| t == n.node_type)
    });

    let mut type_set: BTreeSet<String> = BTreeSet::new();
    let mut nodes: Vec<NodeSpec> = Vec::new();
    let mut edges: Vec<EdgeSpec> = Vec::new();

    let mut seen: std::collections::HashSet<(String, String, String)> =
        std::collections::HashSet::new();
    for n in &all {
        type_set.insert(n.node_type.clone());
        nodes.push(NodeSpec {
            id: n.id.clone(),
            node_type: n.node_type.clone(),
            label: n.label.clone(),
            aliases: n.aliases.clone(),
            source_path: n.prov.source_path.clone(),
            range: n.prov.range.clone(),
            confidence: Some(n.prov.confidence),
        });
        // Both OUTBOUND and INBOUND edges touching this node, deduped — so an edge whose other
        // endpoint is a non-exported node survives a replace-layer round-trip (prune+reimport).
        for e in g.outgoing(&n.id)?.into_iter().chain(g.incoming(&n.id)?) {
            let key = (e.from.clone(), e.edge_type.clone(), e.to.clone());
            if seen.insert(key) {
                edges.push(EdgeSpec {
                    from: e.from,
                    to: e.to,
                    edge_type: e.edge_type,
                    source_path: e.prov.source_path,
                    range: e.prov.range,
                    confidence: Some(e.prov.confidence),
                });
            }
        }
    }

    nodes.sort_by(|a, b| a.id.cmp(&b.id));
    edges.sort_by(|a, b| {
        a.from
            .cmp(&b.from)
            .then(a.to.cmp(&b.to))
            .then(a.edge_type.cmp(&b.edge_type))
    });

    Ok(GraphExport {
        exported_types: type_set.into_iter().collect(),
        nodes,
        edges,
    })
}

pub fn to_json(e: &GraphExport) -> anyhow::Result<String> {
    Ok(serde_json::to_string_pretty(e)?)
}

pub fn from_json(s: &str) -> anyhow::Result<GraphExport> {
    Ok(serde_json::from_str(s)?)
}

fn escape_dot(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', " ")
}

fn type_color(node_type: &str) -> &'static str {
    match node_type {
        "Symptom" => "#ffd6d6",
        "Cause" => "#fff2c8",
        "Resolution" => "#d6f5d6",
        "Task" => "#d6e4ff",
        _ => "#eeeeee",
    }
}

pub fn to_dot(e: &GraphExport) -> String {
    let mut out = String::new();
    out.push_str(
        "digraph kb {\n  rankdir=LR;\n  node [shape=box, style=filled, fontname=\"sans\"];\n",
    );
    for n in &e.nodes {
        let label = format!("{}\\n{}", escape_dot(&n.node_type), escape_dot(&n.label));
        let color = type_color(&n.node_type);
        out.push_str(&format!(
            "  \"{}\" [label=\"{}\", fillcolor=\"{}\"];\n",
            escape_dot(&n.id),
            label,
            color
        ));
    }
    for ed in &e.edges {
        out.push_str(&format!(
            "  \"{}\" -> \"{}\" [label=\"{}\"];\n",
            escape_dot(&ed.from),
            escape_dot(&ed.to),
            escape_dot(&ed.edge_type)
        ));
    }
    out.push_str("}\n");
    out
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

pub fn to_graphml(e: &GraphExport) -> String {
    let node_ids: HashSet<&str> = e.nodes.iter().map(|n| n.id.as_str()).collect();
    let stubs: BTreeSet<&str> = e
        .edges
        .iter()
        .flat_map(|ed| [ed.from.as_str(), ed.to.as_str()])
        .filter(|id| !node_ids.contains(*id))
        .collect();

    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str("<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\">\n");
    out.push_str("  <key id=\"label\" for=\"node\" attr.name=\"label\" attr.type=\"string\"/>\n");
    out.push_str("  <key id=\"type\" for=\"node\" attr.name=\"type\" attr.type=\"string\"/>\n");
    out.push_str(
        "  <key id=\"source_path\" for=\"node\" attr.name=\"source_path\" attr.type=\"string\"/>\n",
    );
    out.push_str(
        "  <key id=\"etype\" for=\"edge\" attr.name=\"edge_type\" attr.type=\"string\"/>\n",
    );
    out.push_str("  <graph id=\"G\" edgedefault=\"directed\">\n");

    for n in &e.nodes {
        out.push_str(&format!("    <node id=\"{}\">\n", xml_escape(&n.id)));
        out.push_str(&format!(
            "      <data key=\"label\">{}</data>\n",
            xml_escape(&n.label)
        ));
        out.push_str(&format!(
            "      <data key=\"type\">{}</data>\n",
            xml_escape(&n.node_type)
        ));
        out.push_str(&format!(
            "      <data key=\"source_path\">{}</data>\n",
            xml_escape(&n.source_path)
        ));
        out.push_str("    </node>\n");
    }

    for id in &stubs {
        out.push_str(&format!("    <node id=\"{}\">\n", xml_escape(id)));
        out.push_str(&format!(
            "      <data key=\"label\">{}</data>\n",
            xml_escape(id)
        ));
        out.push_str("      <data key=\"type\">Section</data>\n");
        out.push_str("    </node>\n");
    }

    for (i, ed) in e.edges.iter().enumerate() {
        out.push_str(&format!(
            "    <edge id=\"e{}\" source=\"{}\" target=\"{}\">\n",
            i,
            xml_escape(&ed.from),
            xml_escape(&ed.to)
        ));
        out.push_str(&format!(
            "      <data key=\"etype\">{}</data>\n",
            xml_escape(&ed.edge_type)
        ));
        out.push_str("    </edge>\n");
    }

    out.push_str("  </graph>\n");
    out.push_str("</graphml>\n");
    out
}

/// Derived similarity links the generalize pass writes. Excluded from graph
/// traversal in the viewer (they are dense and noisy); the viewer navigates the
/// real typed relations. Kept in the payload so a future mode could use them.
const SOFT_EDGE_TYPES: &[&str] = &["SIMILAR"];

/// Render the export as a single self-contained, offline interactive HTML page.
///
/// The page is a search-first, node-by-node graph explorer: a glossary search
/// returns matching nodes, and selecting one shows a concentric ego view (the
/// node in the center with its typed connections), which you traverse by
/// clicking. The graph is drawn with Cytoscape.js, whose minified source is
/// embedded in the binary via `include_str!` alongside the template — nothing
/// is read from disk or the network at runtime, so it works fully offline. Node
/// positions are computed in the browser (small ego networks), so no layout is
/// baked here. Type colors and the legend are derived from the data.
pub fn to_html(g: &GraphStore, e: &GraphExport) -> String {
    use std::collections::{BTreeSet, HashMap};

    let idx: HashMap<&str, usize> = e
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id.as_str(), i))
        .collect();
    let base = e.nodes.len();

    // Non-exported edge endpoints are sources (the structural Section/Document
    // nodes referenced via MENTIONS etc.). Give them indices after the exported
    // nodes so the viewer can show them as `source` nodes you can open. collect()
    // only returns edges touching an exported node, so at most one end is a stub.
    let mut stub_pos: HashMap<&str, usize> = HashMap::new();
    let mut stub_order: Vec<&str> = Vec::new();
    for ed in &e.edges {
        for endp in [ed.from.as_str(), ed.to.as_str()] {
            if !idx.contains_key(endp) && !stub_pos.contains_key(endp) {
                stub_pos.insert(endp, base + stub_order.len());
                stub_order.push(endp);
            }
        }
    }

    // Resolve edges to combined indices; degree (for glossary ranking) counts
    // hard relations between two exported nodes; stubs get a reference count.
    let resolve = |id: &str| -> usize { idx.get(id).copied().unwrap_or_else(|| stub_pos[id]) };
    let mut render_edges: Vec<(usize, usize, &str)> = Vec::with_capacity(e.edges.len());
    let mut deg = vec![0u32; base];
    let mut stub_deg = vec![0u32; stub_order.len()];
    for ed in &e.edges {
        let (s, t) = (resolve(&ed.from), resolve(&ed.to));
        render_edges.push((s, t, ed.edge_type.as_str()));
        if !SOFT_EDGE_TYPES.contains(&ed.edge_type.as_str()) && s < base && t < base && s != t {
            deg[s] += 1;
            deg[t] += 1;
        }
        if s >= base {
            stub_deg[s - base] += 1;
        }
        if t >= base {
            stub_deg[t - base] += 1;
        }
    }

    // Look up real labels for the source stubs, prefixing each with its document
    // so it reads "file.pdf · p.734", not a bare "p.734". All of these
    // (Document, Section, …) are provenance, so they collapse to one `Source`
    // type — a single legend entry instead of three near-synonyms.
    const SOURCE_TYPE: &str = "Source";
    let stub_labels: Vec<String> = stub_order
        .iter()
        .map(|id| match g.get_node(id) {
            Ok(Some(n)) => {
                let doc = n.prov.source_path;
                if doc.is_empty() || n.label.contains(doc.as_str()) {
                    n.label
                } else {
                    let base = doc.rsplit(['/', '\\']).next().unwrap_or(doc.as_str());
                    format!("{base} · {}", n.label)
                }
            }
            _ => id.to_string(),
        })
        .collect();

    let mut node_types: BTreeSet<&str> = e.nodes.iter().map(|n| n.node_type.as_str()).collect();
    if !stub_labels.is_empty() {
        node_types.insert(SOURCE_TYPE);
    }
    let edge_types: BTreeSet<&str> = render_edges.iter().map(|(_, _, k)| *k).collect();

    let mut nodes_json: Vec<serde_json::Value> = e
        .nodes
        .iter()
        .enumerate()
        .map(|(i, nd)| serde_json::json!({"l": nd.label, "t": nd.node_type, "d": deg[i]}))
        .collect();
    for (i, label) in stub_labels.iter().enumerate() {
        nodes_json.push(serde_json::json!({"l": label, "t": SOURCE_TYPE, "d": stub_deg[i], "s": 1}));
    }
    let edges_json: Vec<serde_json::Value> = render_edges
        .iter()
        .map(|&(s, t, k)| serde_json::json!([s, t, k]))
        .collect();

    let payload = serde_json::json!({
        "nodes": nodes_json,
        "edges": edges_json,
        "types": node_types.iter().copied().collect::<Vec<_>>(),
        "edgeTypes": edge_types.iter().copied().collect::<Vec<_>>(),
        "softEdges": SOFT_EDGE_TYPES,
    });
    // `</` inside a <script> would close the tag early; neutralize it.
    let data = serde_json::to_string(&payload)
        .unwrap_or_else(|_| "{}".into())
        .replace("</", "<\\/");
    // Guard against a stray `</script` sequence inside the vendored library.
    let lib = include_str!("vendor/cytoscape.min.js").replace("</script", "<\\/script");

    include_str!("viewer.html.tmpl")
        .replace("__GRAPH_DATA__", &data)
        .replace("/*__CYTOSCAPE_LIB__*/", &lib)
}

pub fn import_replace_layer(
    g: &GraphStore,
    ont: &Ontology,
    e: GraphExport,
    now: u64,
) -> anyhow::Result<(usize, usize, usize)> {
    let mut pruned = 0usize;
    for t in &e.exported_types {
        pruned += g.delete_by_type(t)?;
    }
    let r = apply_upsert(g, ont, e.nodes, e.edges, now)?;
    Ok((pruned, r.nodes_written, r.edges_written))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::agent::apply_upsert;
    use crate::graph::ontology::Ontology;
    use crate::graph::store::{GraphStore, Node, Provenance};

    fn make_export() -> GraphExport {
        GraphExport {
            exported_types: vec!["Symptom".into(), "Resolution".into()],
            nodes: vec![
                NodeSpec {
                    id: "s1".into(),
                    node_type: "Symptom".into(),
                    label: "Label A".into(),
                    aliases: vec![],
                    source_path: "f.md".into(),
                    range: None,
                    confidence: Some(0.9),
                },
                NodeSpec {
                    id: "r1".into(),
                    node_type: "Resolution".into(),
                    label: "Fix it".into(),
                    aliases: vec![],
                    source_path: "f.md".into(),
                    range: None,
                    confidence: Some(0.8),
                },
            ],
            edges: vec![EdgeSpec {
                from: "s1".into(),
                to: "r1".into(),
                edge_type: "RESOLVED_BY".into(),
                source_path: "f.md".into(),
                range: None,
                confidence: Some(0.85),
            }],
        }
    }

    #[test]
    fn roundtrip_json() {
        let export = make_export();
        let json = to_json(&export).unwrap();
        let back = from_json(&json).unwrap();
        assert_eq!(back.exported_types.len(), 2);
        assert_eq!(back.nodes.len(), 2);
        assert_eq!(back.edges.len(), 1);
        assert_eq!(back.nodes[0].label, "Label A");
    }

    fn empty_store() -> GraphStore {
        GraphStore::open(tempfile::tempdir().unwrap().path()).unwrap()
    }

    #[test]
    fn html_is_self_contained_and_embeds_data() {
        let html = to_html(&empty_store(), &make_export());
        // Both template placeholders must be filled in.
        assert!(!html.contains("__GRAPH_DATA__"), "data placeholder not substituted");
        assert!(
            !html.contains("/*__CYTOSCAPE_LIB__*/"),
            "library placeholder not substituted"
        );
        // The graph library is embedded (offline), not loaded from a CDN.
        assert!(html.contains("Cytoscape"), "vendored library not embedded");
        assert!(
            !html.contains("<script src=") && !html.contains("<link "),
            "viewer must not load any external resource"
        );
        // Graph data (labels + types) is baked in.
        assert!(html.contains("Label A"), "node label missing from payload");
        assert!(html.contains("RESOLVED_BY"), "edge type missing from payload");
        assert!(html.contains("Symptom"), "node type missing from payload");
    }

    #[test]
    fn html_handles_empty_graph() {
        let empty = GraphExport {
            exported_types: vec![],
            nodes: vec![],
            edges: vec![],
        };
        let html = to_html(&empty_store(), &empty);
        assert!(!html.contains("__GRAPH_DATA__"));
        assert!(html.contains("Cytoscape"));
    }

    const ONT: &str = r#"
[entities.Symptom]
props = []
[entities.Resolution]
props = []
[relations.RESOLVED_BY]
from = ["Symptom"]
to = ["Resolution"]
"#;

    #[test]
    fn collect_excludes_structural_and_roundtrips_import() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(ONT).unwrap();

        let nodes = vec![
            NodeSpec {
                id: "s1".into(),
                node_type: "Symptom".into(),
                label: "Engine noise".into(),
                aliases: vec![],
                source_path: "doc.md".into(),
                range: None,
                confidence: None,
            },
            NodeSpec {
                id: "r1".into(),
                node_type: "Resolution".into(),
                label: "Replace bearing".into(),
                aliases: vec![],
                source_path: "doc.md".into(),
                range: None,
                confidence: None,
            },
        ];
        let edges = vec![EdgeSpec {
            from: "s1".into(),
            to: "r1".into(),
            edge_type: "RESOLVED_BY".into(),
            source_path: "doc.md".into(),
            range: None,
            confidence: None,
        }];
        apply_upsert(&g, &ont, nodes, edges, 1).unwrap();

        // Also insert a structural Document node directly
        g.put_node(&Node {
            id: "doc.md".into(),
            node_type: "Document".into(),
            label: "doc.md".into(),
            aliases: vec![],
            prov: Provenance {
                source_path: "doc.md".into(),
                range: None,
                file_sig: None,
                origin: "auto-structural".into(),
                confidence: 1.0,
                created_at: 1,
            },
        })
        .unwrap();

        let export = collect(&g, None).unwrap();
        // Document must be excluded
        assert!(!export.nodes.iter().any(|n| n.node_type == "Document"));
        assert_eq!(export.nodes.len(), 2);
        assert_eq!(export.edges.len(), 1);
        assert!(export.exported_types.contains(&"Symptom".to_string()));
        assert!(export.exported_types.contains(&"Resolution".to_string()));

        // import_replace_layer into a fresh store
        let dir2 = tempfile::tempdir().unwrap();
        let g2 = GraphStore::open(dir2.path()).unwrap();
        let (pruned, n, ed) = import_replace_layer(&g2, &ont, export, 2).unwrap();
        assert_eq!(pruned, 0); // nothing to prune in a fresh store
        assert_eq!(n, 2);
        assert_eq!(ed, 1);
    }
}
