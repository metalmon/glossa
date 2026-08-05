//! Consolidated graph-health diagnosis: the three doubts (ungrounded / stale /
//! incomplete) + opt-in prune. Reuses the generalize hygiene primitives; owns
//! no derived-layer logic.

use crate::graph::generalize::hygiene;
use crate::graph::ontology::Ontology;
use crate::graph::store::GraphStore;
use crate::index::manifest::FileSig;
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug, Clone)]
pub enum Reason {
    Ungrounded,
    Stale {
        stored: Option<FileSig>,
        current: Option<FileSig>,
    },
    Incomplete,
}

#[derive(Debug, Clone)]
pub struct DoubtfulNode {
    pub id: String,
    pub node_type: String,
    pub label: String,
    pub source_path: String,
    pub reason: Reason,
}

#[derive(Debug, Default)]
pub struct DoctorReport {
    pub ungrounded: Vec<DoubtfulNode>,
    pub stale: Vec<DoubtfulNode>,
    pub incomplete: Vec<DoubtfulNode>,
    pub unverifiable: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct PruneOpts {
    pub incomplete: bool,
    pub ungrounded: bool,
}

/// Run the three hygiene checks (ungrounded / incomplete / stale) over the live graph and return
/// a report — never mutates the store. The edge `Triple` list and the ontology-derived
/// grounding/spine/structural sets are built exactly as `generalize::apply::Opts::from_ontology` +
/// the hygiene block in `generalize::apply::generalize` do, so the ungrounded/incomplete buckets
/// here are identical to what a `kb graph generalize` pass would compute.
pub fn doctor(g: &GraphStore, ont: &Ontology, root: &Path) -> anyhow::Result<DoctorReport> {
    let nodes = g.all_nodes()?; // Vec<Node> with full Provenance
    let edges = g.all_edges()?; // for hygiene fns

    // Build the lightweight inputs the hygiene fns take (mirror apply.rs:133-166).
    let id_types: Vec<(String, String)> = nodes
        .iter()
        .map(|n| (n.id.clone(), n.node_type.clone()))
        .collect();
    let triples: Vec<(String, String, String)> = edges
        .into_iter()
        .map(|e| (e.from, e.edge_type, e.to))
        .collect();

    let grounding_types: HashSet<String> = ont
        .entity_types()
        .iter()
        .filter(|t| ont.requires_grounding(t))
        .cloned()
        .collect();
    let spines = ont.spines();
    let spine_types = ont.spine_types();
    let structural: HashSet<String> = ont.structural().into_iter().collect();

    let ungrounded_ids = hygiene::ungrounded_nodes(&id_types, &triples, &grounding_types);
    let incomplete_ids = hygiene::incomplete_nodes(
        &id_types,
        &triples,
        &spines,
        &spine_types,
        &structural,
    );

    let stale_input: Vec<(String, String, Option<FileSig>)> = nodes
        .iter()
        .map(|n| (n.id.clone(), n.prov.source_path.clone(), n.prov.file_sig))
        .collect();
    let stale_ids = hygiene::stale_nodes(root, &stale_input);

    // index nodes by id for detail lookup
    let by_id: std::collections::HashMap<&str, &crate::graph::store::Node> =
        nodes.iter().map(|n| (n.id.as_str(), n)).collect();
    let mk = |id: &str, reason: Reason| {
        by_id.get(id).map(|n| DoubtfulNode {
            id: n.id.clone(),
            node_type: n.node_type.clone(),
            label: n.label.clone(),
            source_path: n.prov.source_path.clone(),
            reason,
        })
    };

    let mut rep = DoctorReport::default();
    for id in &ungrounded_ids {
        if let Some(d) = mk(id, Reason::Ungrounded) {
            rep.ungrounded.push(d);
        }
    }
    for id in &incomplete_ids {
        if let Some(d) = mk(id, Reason::Incomplete) {
            rep.incomplete.push(d);
        }
    }
    for id in &stale_ids {
        if let Some(n) = by_id.get(id.as_str()) {
            let current = crate::index::store::file_sig(&root.join(&n.prov.source_path)).ok();
            rep.stale.push(DoubtfulNode {
                id: n.id.clone(),
                node_type: n.node_type.clone(),
                label: n.label.clone(),
                source_path: n.prov.source_path.clone(),
                reason: Reason::Stale {
                    stored: n.prov.file_sig,
                    current,
                },
            });
        }
    }
    // count grounded nodes we could not verify (file_sig None on an agent-authored node)
    rep.unverifiable = nodes
        .iter()
        .filter(|n| n.prov.origin == "agent" && n.prov.file_sig.is_none())
        .count();
    Ok(rep)
}

/// Delete the incomplete and/or ungrounded nodes from `report` (per `opts`). Returns
/// `(incomplete_pruned, ungrounded_pruned)`. `report.stale` is intentionally NEVER pruned — a
/// stale node is doubtful, not doomed, and deleting it would destroy reasoning work over a
/// transient source edit.
pub fn prune(
    g: &GraphStore,
    report: &DoctorReport,
    opts: &PruneOpts,
) -> anyhow::Result<(usize, usize)> {
    let mut inc = 0;
    let mut ung = 0;
    if opts.incomplete && !report.incomplete.is_empty() {
        let ids: Vec<String> = report.incomplete.iter().map(|d| d.id.clone()).collect();
        inc = g.delete_nodes(&ids)?;
    }
    if opts.ungrounded && !report.ungrounded.is_empty() {
        let ids: Vec<String> = report.ungrounded.iter().map(|d| d.id.clone()).collect();
        ung = g.delete_nodes(&ids)?;
    }
    // stale is intentionally NEVER pruned
    Ok((inc, ung))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ontology::Ontology;
    use crate::graph::store::{Edge, GraphStore, Node, Provenance};
    use crate::index::store::file_sig;

    const ONT: &str = r#"
[entities.Symptom]
props = ["name"]
[entities.Cause]
props = ["name"]
[entities.Resolution]
props = ["name"]
requires_grounding = true
[entities.Section]
props = ["name"]
[relations.CAUSED_BY]
from = ["Symptom"]
to = ["Cause"]
[relations.RESOLVED_BY]
from = ["Cause", "Symptom"]
to = ["Resolution"]
[relations.MENTIONS]
from = ["Symptom", "Resolution"]
to = ["Section"]
[validation]
strict = false
[reasoning]
spines = [{ anchor = "Symptom", relations = ["CAUSED_BY", "RESOLVED_BY"] }]
"#;

    fn prov(source_path: &str, file_sig: Option<crate::index::manifest::FileSig>) -> Provenance {
        Provenance {
            source_path: source_path.into(),
            range: None,
            file_sig,
            origin: "agent".into(),
            confidence: 0.9,
            created_at: 1,
        }
    }
    fn node(id: &str, ty: &str, label: &str, p: Provenance) -> Node {
        Node {
            id: id.into(),
            node_type: ty.into(),
            label: label.into(),
            aliases: vec![],
            prov: p,
        }
    }
    fn edge(f: &str, ty: &str, t: &str, p: Provenance) -> Edge {
        Edge {
            from: f.into(),
            to: t.into(),
            edge_type: ty.into(),
            prov: p,
        }
    }

    #[test]
    fn doctor_reports_three_buckets() {
        // Build a graph with: a grounded node whose source drifted (stale),
        // a requires_grounding node with no MENTIONS (ungrounded), and an
        // off-spine node (incomplete). Reuse the store/ontology test scaffolding
        // from generalize/hygiene tests.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let g = GraphStore::open(root).unwrap();
        let ont = Ontology::parse(ONT).unwrap();

        // ── Chain A: complete + grounded, but its source file drifted → stale. ──
        let doc_a = root.join("doc_a.md");
        std::fs::write(&doc_a, b"v1").unwrap();
        let sig0 = file_sig(&doc_a).unwrap();

        // ── Chain B: complete + on-spine, but the grounding-required node has no
        // live MENTIONS → ungrounded. ──
        // ── Chain C: an isolated Cause node, on no complete spine → incomplete. ──
        let nodes = vec![
            node("sym:a", "Symptom", "A", prov("doc_a.md", None)),
            node("cau:a", "Cause", "A cause", prov("doc_a.md", None)),
            node("res:a", "Resolution", "A res", prov("doc_a.md", Some(sig0))),
            node("sec:a", "Section", "A sec", prov("doc_a.md", None)),
            node("sym:b", "Symptom", "B", prov("doc_b.md", None)),
            node("cau:b", "Cause", "B cause", prov("doc_b.md", None)),
            node("res:b", "Resolution", "B res", prov("doc_b.md", None)),
            node("cau:orphan", "Cause", "Orphan", prov("doc_c.md", None)),
        ];
        let edges = vec![
            edge("sym:a", "CAUSED_BY", "cau:a", prov("doc_a.md", None)),
            edge("cau:a", "RESOLVED_BY", "res:a", prov("doc_a.md", None)),
            edge("res:a", "MENTIONS", "sec:a", prov("doc_a.md", None)),
            edge("sym:b", "CAUSED_BY", "cau:b", prov("doc_b.md", None)),
            edge("cau:b", "RESOLVED_BY", "res:b", prov("doc_b.md", None)),
        ];
        g.upsert(&ont, &nodes, &edges).unwrap();

        // Now let the source drift so res:a's stored sig no longer matches disk.
        std::fs::write(&doc_a, b"v2-longer").unwrap();

        let rep = doctor(&g, &ont, root).unwrap();
        assert_eq!(rep.stale.len(), 1, "res:a's source drifted");
        assert_eq!(rep.ungrounded.len(), 1, "res:b has no live MENTIONS");
        assert_eq!(rep.incomplete.len(), 1, "cau:orphan is on no complete spine");
        assert_eq!(rep.stale[0].id, "res:a");
        assert_eq!(rep.ungrounded[0].id, "res:b");
        assert_eq!(rep.incomplete[0].id, "cau:orphan");

        // prune removes incomplete + ungrounded, NOT stale
        let (inc, ung) = prune(
            &g,
            &rep,
            &PruneOpts {
                incomplete: true,
                ungrounded: true,
            },
        )
        .unwrap();
        assert_eq!((inc, ung), (1, 1));
        assert!(
            g.get_node(&rep.stale[0].id).unwrap().is_some(),
            "stale node must survive prune"
        );
        assert!(g.get_node("cau:orphan").unwrap().is_none());
        assert!(g.get_node("res:b").unwrap().is_none());
    }
}
