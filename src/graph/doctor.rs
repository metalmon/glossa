//! Consolidated graph-health diagnosis: the four doubts (ungrounded / stale /
//! incomplete / dangling) + opt-in prune. Reuses the generalize hygiene primitives; `dangling` is
//! the one derived-layer doubt owned here (structural reachability over the other three).

use crate::graph::generalize::hygiene;
use crate::graph::ontology::{Ontology, RelationRole};
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
    /// A query-side (non-`requires_grounding`) node that reaches no live grounded terminal by
    /// walking forward along the ontology's chaining relations — its answer's source is gone.
    Dangling,
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
    /// Query-side nodes (type NOT `requires_grounding`) that reach no live grounded terminal —
    /// derived/structural staleness. UNLIKE `stale`, `dangling` IS prunable — opt-in via
    /// `prune_dangling` (MCP) / `--prune-dangling` (CLI), same "last resort" policy as
    /// `ungrounded`. Because a whole-layer dangling flood is the signature of an ontology
    /// mismatch rather than genuine per-node rot, `dangling_prune_risk` gates the delete: an
    /// agent (MCP) can never mass-prune, only a human can force it (CLI `--force`).
    pub dangling: Vec<DoubtfulNode>,
    /// Count of `requires_grounding` nodes present in the graph, not ungrounded, not stale —
    /// i.e. how many live terminals the current ontology recognizes. Computed once here and
    /// reused by `dangling_prune_risk` so the mass-wipe check never recomputes (and can't drift
    /// from) the same derivation `doctor()` used to flag `dangling` in the first place.
    pub live_terminal_count: usize,
    pub unverifiable: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct PruneOpts {
    pub incomplete: bool,
    pub ungrounded: bool,
    /// Opt-in, like `ungrounded`: last resort, prefer restoring the terminal. By default a
    /// dangling node's terminal may come back live (source restored, or re-grounded), so this
    /// stays off unless explicitly requested.
    pub dangling: bool,
}

/// Run the four hygiene checks (ungrounded / incomplete / stale / dangling) over the live graph
/// and return a report — never mutates the store. The edge `Triple` list and the ontology-derived
/// grounding/spine/structural sets are built exactly as `generalize::apply::Opts::from_ontology` +
/// the hygiene block in `generalize::apply::generalize` do, so the ungrounded/incomplete buckets
/// here are identical to what a `kb graph generalize` pass would compute. `dangling` is a fourth,
/// DERIVED doubt: a query-side node whose chain (walked forward over `RelationRole::Chaining`
/// edges) reaches no live grounded terminal — e.g. its terminal's source document was deleted, so
/// the terminal itself went ungrounded/stale but the query-side nodes leading to it have no
/// `file_sig` of their own and would otherwise look fresh forever.
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
    let incomplete_ids =
        hygiene::incomplete_nodes(&id_types, &triples, &spines, &spine_types, &structural);

    let stale_input: Vec<(String, String, Option<FileSig>)> = nodes
        .iter()
        .map(|n| (n.id.clone(), n.prov.source_path.clone(), n.prov.file_sig))
        .collect();
    let stale_ids = hygiene::stale_nodes(root, &stale_input);

    // ── Derived (structural) staleness: a query-side node is dangling if it reaches no LIVE
    // grounded terminal — one whose type requires_grounding, is present, and is not itself
    // ungrounded or stale. Chaining edges are the ontology's reasoning hops (role==Chaining);
    // MENTIONS/SIMILAR/structural (role==Grounding) are excluded — same distinction traverse::reach
    // uses to decide what advances the chain. ──
    let ungrounded_set: HashSet<&str> = ungrounded_ids.iter().map(String::as_str).collect();
    let stale_set: HashSet<&str> = stale_ids.iter().map(String::as_str).collect();
    let live_terminal_ids: HashSet<String> = id_types
        .iter()
        .filter(|(id, ty)| {
            grounding_types.contains(ty)
                && !ungrounded_set.contains(id.as_str())
                && !stale_set.contains(id.as_str())
        })
        .map(|(id, _)| id.clone())
        .collect();
    let chaining_edges: Vec<(String, String)> = triples
        .iter()
        .filter(|(_, et, _)| ont.relation_role(et) == RelationRole::Chaining)
        .map(|(f, _, t)| (f.clone(), t.clone()))
        .collect();
    let dangling_ids = hygiene::dangling_nodes(
        &id_types,
        &chaining_edges,
        &grounding_types,
        &structural,
        &live_terminal_ids,
    );

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

    let mut rep = DoctorReport {
        live_terminal_count: live_terminal_ids.len(),
        ..Default::default()
    };
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
    for id in &dangling_ids {
        if let Some(d) = mk(id, Reason::Dangling) {
            rep.dangling.push(d);
        }
    }
    // Count nodes we SHOULD be able to verify but can't: authored (agent- or distil-origin), of
    // a type the ontology declares `requires_grounding`, with no stored file_sig. Excludes
    // ungrounded-by-design reasoning node types (requires_grounding == false) — those never
    // carry a file_sig and are not a doubt, just not source-verifiable by design.
    rep.unverifiable = nodes
        .iter()
        .filter(|n| {
            matches!(n.prov.origin.as_str(), "agent" | "distil")
                && n.prov.file_sig.is_none()
                && ont.requires_grounding(&n.node_type)
        })
        .count();
    Ok(rep)
}

/// Delete the incomplete, ungrounded and/or dangling nodes from `report` (per `opts`). Returns
/// `(incomplete_pruned, ungrounded_pruned, dangling_pruned)`. `report.stale` is intentionally
/// NEVER pruned — a stale node's source may be re-synced, at which point it is fresh again;
/// deleting it would destroy reasoning work over what may be a transient state. `dangling` is
/// prunable, but opt-in only (like `ungrounded`: last resort, prefer restoring the terminal) —
/// by default a dangling node's terminal may come back live (source restored, or a new MENTIONS
/// re-grounding it).
pub fn prune(
    g: &GraphStore,
    report: &DoctorReport,
    opts: &PruneOpts,
) -> anyhow::Result<(usize, usize, usize)> {
    let mut inc = 0;
    let mut ung = 0;
    let mut dang = 0;
    if opts.incomplete && !report.incomplete.is_empty() {
        let ids: Vec<String> = report.incomplete.iter().map(|d| d.id.clone()).collect();
        inc = g.delete_nodes(&ids)?;
    }
    if opts.ungrounded && !report.ungrounded.is_empty() {
        let ids: Vec<String> = report.ungrounded.iter().map(|d| d.id.clone()).collect();
        ung = g.delete_nodes(&ids)?;
    }
    if opts.dangling && !report.dangling.is_empty() {
        let ids: Vec<String> = report.dangling.iter().map(|d| d.id.clone()).collect();
        dang = g.delete_nodes(&ids)?;
    }
    // stale is intentionally NEVER pruned
    Ok((inc, ung, dang))
}

/// Returns `Some(reason)` when pruning the `dangling` bucket would be a mass-wipe — the signal of
/// an ontology mismatch (e.g. a missing/changed `ontology.toml`) rather than genuine per-node rot.
/// `None` = safe to prune. Two triggers:
///   1. the graph has non-structural nodes but the ontology recognizes NO live grounded terminal
///      (`report.live_terminal_count == 0`) — every non-structural node is trivially "dangling";
///   2. dangling nodes exceed ~50% of the non-structural (reasoning) layer.
///
/// This only gates the DELETE — `doctor()` keeps reporting `dangling` regardless.
pub fn dangling_prune_risk(
    report: &DoctorReport,
    g: &GraphStore,
    ont: &Ontology,
) -> Option<String> {
    if report.dangling.is_empty() {
        return None;
    }
    let structural: HashSet<String> = ont.structural().into_iter().collect();
    let non_structural = g
        .all_nodes()
        .map(|nodes| {
            nodes
                .iter()
                .filter(|n| !structural.contains(&n.node_type))
                .count()
        })
        .unwrap_or(0);
    if non_structural > 0 && report.live_terminal_count == 0 {
        return Some(
            "the ontology recognizes no live grounded terminal in this graph — likely a missing \
             or mismatched .glossa/ontology.toml; refusing to prune the whole reasoning layer"
                .to_string(),
        );
    }
    if report.dangling.len() * 2 > non_structural {
        return Some(format!(
            "dangling ({}) is over half the reasoning layer ({non_structural}) — refusing a mass delete",
            report.dangling.len()
        ));
    }
    None
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
role = "grounding"
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
        assert_eq!(
            rep.incomplete.len(),
            1,
            "cau:orphan is on no complete spine"
        );
        assert_eq!(rep.stale[0].id, "res:a");
        assert_eq!(rep.ungrounded[0].id, "res:b");
        assert_eq!(rep.incomplete[0].id, "cau:orphan");

        // prune removes incomplete + ungrounded, NOT stale
        let (inc, ung, dang) = prune(
            &g,
            &rep,
            &PruneOpts {
                incomplete: true,
                ungrounded: true,
                dangling: false,
            },
        )
        .unwrap();
        assert_eq!((inc, ung, dang), (1, 1, 0));
        assert!(
            g.get_node(&rep.stale[0].id).unwrap().is_some(),
            "stale node must survive prune"
        );
        assert!(g.get_node("cau:orphan").unwrap().is_none());
        assert!(g.get_node("res:b").unwrap().is_none());
    }

    #[test]
    fn unverifiable_counts_distil_origin_same_as_agent() {
        // Regression guard: `unverifiable` must key on origin IN ('agent', 'distil'), not just
        // "agent" — a distil-origin node (kbx distil densification writer) that requires
        // grounding but has no stored file_sig must be counted too, or it silently escapes the
        // doubt it should raise. A "curated" origin (not authored by either writer) must NOT be
        // swept in.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let g = GraphStore::open(root).unwrap();
        let ont = Ontology::parse(ONT).unwrap();

        let ungrounded_prov = |origin: &str| Provenance {
            source_path: "docA.md".into(),
            range: None,
            file_sig: None,
            origin: origin.into(),
            confidence: 0.9,
            created_at: 1,
        };
        let nodes = vec![
            node(
                "res:agent",
                "Resolution",
                "Agent res",
                ungrounded_prov("agent"),
            ),
            node(
                "res:distil",
                "Resolution",
                "Distil res",
                ungrounded_prov("distil"),
            ),
            node(
                "res:curated",
                "Resolution",
                "Curated res",
                ungrounded_prov("curated"),
            ),
        ];
        g.upsert(&ont, &nodes, &[]).unwrap();

        let rep = doctor(&g, &ont, root).unwrap();
        assert_eq!(
            rep.unverifiable, 2,
            "agent- and distil-origin ungrounded Resolution nodes must both count; curated must not"
        );
    }

    #[test]
    fn doctor_flags_query_side_node_dangling_when_its_terminal_is_stale() {
        // Symptom -CAUSED_BY-> Cause -RESOLVED_BY-> Resolution(grounded), then the Resolution's
        // source drifts → it lands in `stale`, and both query-side nodes leading to it — which
        // have no file_sig of their own and so could never go stale directly — must be flagged
        // `dangling`: their chain now reaches no LIVE grounded terminal.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let g = GraphStore::open(root).unwrap();
        let ont = Ontology::parse(ONT).unwrap();

        let doc = root.join("doc.md");
        std::fs::write(&doc, b"v1").unwrap();
        let sig0 = file_sig(&doc).unwrap();

        let nodes = vec![
            node("sym:1", "Symptom", "S", prov("doc.md", None)),
            node("cau:1", "Cause", "C", prov("doc.md", None)),
            node("res:1", "Resolution", "R", prov("doc.md", Some(sig0))),
            node("sec:1", "Section", "Sec", prov("doc.md", None)),
        ];
        let edges = vec![
            edge("sym:1", "CAUSED_BY", "cau:1", prov("doc.md", None)),
            edge("cau:1", "RESOLVED_BY", "res:1", prov("doc.md", None)),
            edge("res:1", "MENTIONS", "sec:1", prov("doc.md", None)),
        ];
        g.upsert(&ont, &nodes, &edges).unwrap();

        // The Resolution's source drifts → it goes stale.
        std::fs::write(&doc, b"v2-longer").unwrap();

        let rep = doctor(&g, &ont, root).unwrap();
        assert_eq!(
            rep.stale.iter().map(|d| d.id.as_str()).collect::<Vec<_>>(),
            vec!["res:1"]
        );
        let dangling: Vec<&str> = rep.dangling.iter().map(|d| d.id.as_str()).collect();
        assert!(
            dangling.contains(&"sym:1"),
            "Symptom must dangle: its only terminal went stale"
        );
        assert!(
            dangling.contains(&"cau:1"),
            "Cause must dangle: its only terminal went stale"
        );
    }

    #[test]
    fn prune_dangling_is_opt_in_and_stale_survives_regardless() {
        // Same shape as `doctor_flags_query_side_node_dangling_when_its_terminal_is_stale`:
        // sym:1 -CAUSED_BY-> cau:1 -RESOLVED_BY-> res:1(stale). sym:1/cau:1 are dangling.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let g = GraphStore::open(root).unwrap();
        let ont = Ontology::parse(ONT).unwrap();

        let doc = root.join("doc.md");
        std::fs::write(&doc, b"v1").unwrap();
        let sig0 = file_sig(&doc).unwrap();

        let nodes = vec![
            node("sym:1", "Symptom", "S", prov("doc.md", None)),
            node("cau:1", "Cause", "C", prov("doc.md", None)),
            node("res:1", "Resolution", "R", prov("doc.md", Some(sig0))),
            node("sec:1", "Section", "Sec", prov("doc.md", None)),
        ];
        let edges = vec![
            edge("sym:1", "CAUSED_BY", "cau:1", prov("doc.md", None)),
            edge("cau:1", "RESOLVED_BY", "res:1", prov("doc.md", None)),
            edge("res:1", "MENTIONS", "sec:1", prov("doc.md", None)),
        ];
        g.upsert(&ont, &nodes, &edges).unwrap();

        // The Resolution's source drifts → it goes stale, and sym:1/cau:1 go dangling.
        std::fs::write(&doc, b"v2-longer").unwrap();

        let rep = doctor(&g, &ont, root).unwrap();
        assert_eq!(rep.stale.len(), 1);
        assert_eq!(rep.dangling.len(), 2);

        // dangling=false leaves both dangling nodes (and stale) untouched.
        let (inc0, ung0, dang0) = prune(
            &g,
            &rep,
            &PruneOpts {
                incomplete: false,
                ungrounded: false,
                dangling: false,
            },
        )
        .unwrap();
        assert_eq!((inc0, ung0, dang0), (0, 0, 0));
        assert!(g.get_node("sym:1").unwrap().is_some());
        assert!(g.get_node("cau:1").unwrap().is_some());
        assert!(
            g.get_node("res:1").unwrap().is_some(),
            "stale node must survive prune regardless of the dangling flag"
        );

        // dangling=true deletes exactly the dangling nodes; stale still survives.
        let (inc1, ung1, dang1) = prune(
            &g,
            &rep,
            &PruneOpts {
                incomplete: false,
                ungrounded: false,
                dangling: true,
            },
        )
        .unwrap();
        assert_eq!((inc1, ung1, dang1), (0, 0, 2));
        assert!(g.get_node("sym:1").unwrap().is_none());
        assert!(g.get_node("cau:1").unwrap().is_none());
        assert!(
            g.get_node("res:1").unwrap().is_some(),
            "stale node must survive prune even when dangling=true"
        );
    }

    #[test]
    fn doctor_does_not_flag_query_side_node_when_terminal_is_live() {
        // Same chain shape, but the Resolution's source has NOT drifted and it has a live
        // MENTIONS — the chain reaches a live terminal, so Symptom/Cause must NOT be dangling.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let g = GraphStore::open(root).unwrap();
        let ont = Ontology::parse(ONT).unwrap();

        let doc = root.join("doc.md");
        std::fs::write(&doc, b"v1").unwrap();
        let sig0 = file_sig(&doc).unwrap();

        let nodes = vec![
            node("sym:1", "Symptom", "S", prov("doc.md", None)),
            node("cau:1", "Cause", "C", prov("doc.md", None)),
            node("res:1", "Resolution", "R", prov("doc.md", Some(sig0))),
            node("sec:1", "Section", "Sec", prov("doc.md", None)),
        ];
        let edges = vec![
            edge("sym:1", "CAUSED_BY", "cau:1", prov("doc.md", None)),
            edge("cau:1", "RESOLVED_BY", "res:1", prov("doc.md", None)),
            edge("res:1", "MENTIONS", "sec:1", prov("doc.md", None)),
        ];
        g.upsert(&ont, &nodes, &edges).unwrap();
        // No drift, live MENTIONS: res:1 stays a live terminal.

        let rep = doctor(&g, &ont, root).unwrap();
        assert!(rep.stale.is_empty());
        assert!(rep.ungrounded.is_empty());
        let dangling: Vec<&str> = rep.dangling.iter().map(|d| d.id.as_str()).collect();
        assert!(!dangling.contains(&"sym:1"));
        assert!(!dangling.contains(&"cau:1"));
    }

    #[test]
    fn doctor_flags_query_side_node_with_no_path_to_any_terminal() {
        // An orphan Symptom with no outgoing chaining edge at all reaches no terminal (live or
        // otherwise) → dangling, independent of the stale/ungrounded machinery.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let g = GraphStore::open(root).unwrap();
        let ont = Ontology::parse(ONT).unwrap();

        std::fs::write(root.join("doc.md"), b"v1").unwrap();
        let nodes = vec![node(
            "sym:orphan",
            "Symptom",
            "Orphan",
            prov("doc.md", None),
        )];
        g.upsert(&ont, &nodes, &[]).unwrap();

        let rep = doctor(&g, &ont, root).unwrap();
        let dangling: Vec<&str> = rep.dangling.iter().map(|d| d.id.as_str()).collect();
        assert!(dangling.contains(&"sym:orphan"));
    }

    #[test]
    fn dangling_prune_risk_none_when_no_dangling() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let g = GraphStore::open(root).unwrap();
        let ont = Ontology::parse(ONT).unwrap();
        let rep = DoctorReport::default(); // dangling empty by construction
        assert!(dangling_prune_risk(&rep, &g, &ont).is_none());
    }

    #[test]
    fn dangling_prune_risk_flags_ontology_mismatch_zero_live_terminals() {
        // Same shape as `doctor_flags_query_side_node_dangling_when_its_terminal_is_stale`: the
        // only grounded terminal (res:1) went stale, so `live_terminal_count == 0` while
        // non-structural nodes (sym:1/cau:1/res:1) are present — the ontology-mismatch trigger.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let g = GraphStore::open(root).unwrap();
        let ont = Ontology::parse(ONT).unwrap();

        let doc = root.join("doc.md");
        std::fs::write(&doc, b"v1").unwrap();
        let sig0 = file_sig(&doc).unwrap();

        let nodes = vec![
            node("sym:1", "Symptom", "S", prov("doc.md", None)),
            node("cau:1", "Cause", "C", prov("doc.md", None)),
            node("res:1", "Resolution", "R", prov("doc.md", Some(sig0))),
            node("sec:1", "Section", "Sec", prov("doc.md", None)),
        ];
        let edges = vec![
            edge("sym:1", "CAUSED_BY", "cau:1", prov("doc.md", None)),
            edge("cau:1", "RESOLVED_BY", "res:1", prov("doc.md", None)),
            edge("res:1", "MENTIONS", "sec:1", prov("doc.md", None)),
        ];
        g.upsert(&ont, &nodes, &edges).unwrap();
        std::fs::write(&doc, b"v2-longer").unwrap(); // res:1 -> stale -> zero live terminals

        let rep = doctor(&g, &ont, root).unwrap();
        assert_eq!(rep.live_terminal_count, 0);
        assert!(!rep.dangling.is_empty());
        let risk = dangling_prune_risk(&rep, &g, &ont);
        assert!(risk.is_some(), "zero live terminals must refuse the prune");
        assert!(risk.unwrap().contains("no live grounded terminal"));
    }

    #[test]
    fn dangling_prune_risk_flags_majority_dangling_even_with_a_live_terminal() {
        // One healthy chain (sym:a/cau:a/res:a, res:a live+grounded) plus four orphan Symptoms
        // with no outgoing edge at all. non_structural = 3 + 4 = 7, dangling = 4 > 7/2 — the
        // over-half-the-layer trigger, independent of the zero-live-terminal one.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let g = GraphStore::open(root).unwrap();
        let ont = Ontology::parse(ONT).unwrap();

        std::fs::write(root.join("doc.md"), b"v1").unwrap();
        let mut nodes = vec![
            node("sym:a", "Symptom", "A", prov("doc.md", None)),
            node("cau:a", "Cause", "A cause", prov("doc.md", None)),
            node("res:a", "Resolution", "A res", prov("doc.md", None)),
            node("sec:a", "Section", "A sec", prov("doc.md", None)),
        ];
        for i in 0..4 {
            nodes.push(node(
                &format!("sym:orphan{i}"),
                "Symptom",
                "Orphan",
                prov("doc.md", None),
            ));
        }
        let edges = vec![
            edge("sym:a", "CAUSED_BY", "cau:a", prov("doc.md", None)),
            edge("cau:a", "RESOLVED_BY", "res:a", prov("doc.md", None)),
            edge("res:a", "MENTIONS", "sec:a", prov("doc.md", None)),
        ];
        g.upsert(&ont, &nodes, &edges).unwrap();

        let rep = doctor(&g, &ont, root).unwrap();
        assert!(rep.live_terminal_count > 0, "res:a must be a live terminal");
        assert_eq!(
            rep.dangling.len(),
            4,
            "the four orphan Symptoms must dangle"
        );
        let risk = dangling_prune_risk(&rep, &g, &ont);
        assert!(risk.is_some(), "4-of-7 dangling must refuse the prune");
        assert!(risk.unwrap().contains("over half"));
    }

    #[test]
    fn dangling_prune_risk_none_for_small_dangling_fraction_with_live_terminal() {
        // Same healthy chain, but only ONE orphan Symptom: non_structural = 3 + 1 = 4,
        // dangling = 1, not over half, and a live terminal exists -> safe to prune.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let g = GraphStore::open(root).unwrap();
        let ont = Ontology::parse(ONT).unwrap();

        std::fs::write(root.join("doc.md"), b"v1").unwrap();
        let nodes = vec![
            node("sym:a", "Symptom", "A", prov("doc.md", None)),
            node("cau:a", "Cause", "A cause", prov("doc.md", None)),
            node("res:a", "Resolution", "A res", prov("doc.md", None)),
            node("sec:a", "Section", "A sec", prov("doc.md", None)),
            node("sym:orphan", "Symptom", "Orphan", prov("doc.md", None)),
        ];
        let edges = vec![
            edge("sym:a", "CAUSED_BY", "cau:a", prov("doc.md", None)),
            edge("cau:a", "RESOLVED_BY", "res:a", prov("doc.md", None)),
            edge("res:a", "MENTIONS", "sec:a", prov("doc.md", None)),
        ];
        g.upsert(&ont, &nodes, &edges).unwrap();

        let rep = doctor(&g, &ont, root).unwrap();
        assert!(rep.live_terminal_count > 0);
        assert_eq!(rep.dangling.len(), 1, "only the orphan Symptom must dangle");
        assert!(
            dangling_prune_risk(&rep, &g, &ont).is_none(),
            "a single dangling node in an otherwise-healthy graph must be safe to prune"
        );
    }
}
