//! Consolidated graph-health diagnosis: the four doubts (ungrounded / stale /
//! incomplete / dangling) + opt-in prune. Reuses the generalize hygiene primitives; `dangling` is
//! the one derived-layer doubt owned here (structural reachability over the other three).

use crate::graph::generalize::hygiene;
use crate::graph::ontology::{Ontology, RelationRole};
use crate::graph::store::GraphStore;
use crate::index::manifest::FileSig;
use std::collections::HashSet;

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
    /// derived/structural staleness. Prunable opt-in via `prune_dangling` (MCP) /
    /// `--prune-dangling` (CLI), same "last resort" policy as `ungrounded`. Because a whole-layer
    /// dangling flood is the signature of an ontology
    /// mismatch rather than genuine per-node rot, `dangling_prune_risk` gates the delete: an
    /// agent (MCP) can never mass-prune, only a human can force it (CLI `--force`).
    pub dangling: Vec<DoubtfulNode>,
    /// Count of `requires_grounding` nodes present in the graph, not ungrounded, not stale —
    /// i.e. how many live terminals the current ontology recognizes. Computed once here and
    /// reused by `dangling_prune_risk` so the mass-wipe check never recomputes (and can't drift
    /// from) the same derivation `doctor()` used to flag `dangling` in the first place.
    pub live_terminal_count: usize,
    pub unverifiable: usize,
    /// `Some(reason)` when the `incomplete` check could not run against this ontology — currently
    /// only "no `[reasoning] spines` declared", which makes `incomplete_nodes` a structural no-op.
    /// Lets the formatter distinguish "0 incomplete nodes" (check ran, clean) from "check disabled",
    /// so a bare `0` can't read as a false all-clear on a spine-less / terminal-as-sink ontology.
    pub incomplete_disabled: Option<&'static str>,
    /// `Some(reason)` when the `dangling` check is inapplicable — no declared node type is
    /// query-side (every entity type is `requires_grounding` or structural), so `dangling_nodes`
    /// has no candidate and can never fire. Same false-all-clear guard as `incomplete_disabled`.
    pub dangling_inapplicable: Option<&'static str>,
}

#[derive(Debug, Clone, Copy)]
pub struct PruneOpts {
    pub incomplete: bool,
    pub ungrounded: bool,
    /// Opt-in, like `ungrounded`: last resort, prefer restoring the terminal. By default a
    /// dangling node's terminal may come back live (source restored, or re-grounded), so this
    /// stays off unless explicitly requested.
    pub dangling: bool,
    /// Opt-in: also delete `stale` nodes (the backing source drifted since the node was recorded).
    /// Off by default — a stale node's source may be re-synced (or the doc rebuilt), refreshing it
    /// in place; prune only once you've decided the drifted content is gone for good (typically a
    /// cleanup pass after a rebuild).
    pub stale: bool,
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
pub fn doctor(
    g: &GraphStore,
    ont: &Ontology,
    roots: &[crate::root::Root],
) -> anyhow::Result<DoctorReport> {
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
    let stale_ids = hygiene::stale_nodes(roots, &stale_input);

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

    // Two doubts are structurally inert on some ontologies — record WHY so the formatter can print
    // `n/a` instead of a bare `0` that reads as a false all-clear. `incomplete_nodes` no-ops without
    // spines; `dangling_nodes` has no candidate when every declared type is a grounded terminal or
    // structural substrate (the "terminal-as-sink" shape, e.g. an all-`requires_grounding` ontology).
    let incomplete_disabled = spines
        .is_empty()
        .then_some("no [reasoning] spines declared in ontology");
    let has_query_side_type = ont
        .entity_types()
        .iter()
        .any(|t| !grounding_types.contains(t) && !structural.contains(t));
    let dangling_inapplicable = (!has_query_side_type)
        .then_some("no query-side node types (every type is a grounded terminal or structural)");

    let mut rep = DoctorReport {
        live_terminal_count: live_terminal_ids.len(),
        incomplete_disabled,
        dangling_inapplicable,
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
            let current = crate::index::store::file_sig(&crate::index::store::doc_file_in(
                roots,
                &n.prov.source_path,
            ))
            .ok();
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

/// Delete the incomplete, ungrounded, dangling and/or stale nodes from `report` (per `opts`).
/// Returns `(incomplete_pruned, ungrounded_pruned, dangling_pruned, stale_pruned)`. Every bucket
/// is opt-in and off by default (report-only): each delete is a last resort — prefer re-grounding
/// (`ungrounded`), restoring the terminal (`dangling`), or re-syncing / rebuilding the source
/// (`stale`, whose file may drift back into agreement). `dangling`'s extra mass-wipe guard lives
/// in `dangling_prune_risk`, checked by the callers before they set `opts.dangling`.
pub fn prune(
    g: &GraphStore,
    report: &DoctorReport,
    opts: &PruneOpts,
) -> anyhow::Result<(usize, usize, usize, usize)> {
    let mut inc = 0;
    let mut ung = 0;
    let mut dang = 0;
    let mut stale = 0;
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
    if opts.stale && !report.stale.is_empty() {
        let ids: Vec<String> = report.stale.iter().map(|d| d.id.clone()).collect();
        stale = g.delete_nodes(&ids)?;
    }
    Ok((inc, ung, dang, stale))
}

/// Returns `Some(reason)` when pruning the `dangling` bucket would be a mass-wipe — the signal of
/// an ontology mismatch (e.g. a missing/changed `ontology.toml`) rather than genuine per-node rot.
/// `None` = safe to prune. Three triggers:
///   1. the graph has non-structural nodes but the ontology recognizes NO live grounded terminal
///      (`report.live_terminal_count == 0`) — every non-structural node is trivially "dangling";
///   2. dangling nodes exceed ~50% of the non-structural (reasoning) layer;
///   3. a large dangling set (≥ `NEVER_BUILT_FLOOD`) that is mostly "never-built" — anchors with no
///      outgoing chaining edge at all — i.e. a mid-construction graph (`kbx reason` unfinished) or a
///      just-changed ontology, not per-node rot.
///
/// This only gates the DELETE — `doctor()` keeps reporting `dangling` regardless. `--force` overrides.
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
    // 3. A large dangling set that is mostly "never-built" — query-side anchors with no outgoing
    //    reasoning (chaining) edge at all — is the signature of a graph `kbx reason` hasn't finished
    //    wiring, or whose ontology shape just changed, NOT per-node rot ("the answer's source is
    //    gone"). Deleting here discards anchors reason is about to connect. A few orphan anchors in
    //    a built graph are normal prunable junk, so require a non-trivial absolute count first so
    //    only a genuine flood trips this.
    const NEVER_BUILT_FLOOD: usize = 16;
    if report.dangling.len() >= NEVER_BUILT_FLOOD {
        let chaining_srcs: HashSet<String> = g
            .all_edges()
            .map(|edges| {
                edges
                    .into_iter()
                    .filter(|e| ont.relation_role(&e.edge_type) == RelationRole::Chaining)
                    .map(|e| e.from)
                    .collect()
            })
            .unwrap_or_default();
        let never_built = report
            .dangling
            .iter()
            .filter(|d| !chaining_srcs.contains(&d.id))
            .count();
        if never_built * 2 > report.dangling.len() {
            return Some(format!(
                "{never_built} of {} dangling nodes were never chained to a terminal (no outgoing \
                 reasoning edge) — the graph looks mid-construction or its ontology just changed, \
                 not rotted; run `kbx reason` / rebuild before pruning, or --force to override",
                report.dangling.len()
            ));
        }
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
    /// Back-compat single empty-label root, for tests that only ever had one.
    fn single_root(root: &std::path::Path) -> Vec<crate::root::Root> {
        vec![crate::root::Root {
            label: String::new(),
            path: root.to_path_buf(),
        }]
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

        let rep = doctor(&g, &ont, &single_root(root)).unwrap();
        // ONT declares spines and has query-side types (Symptom/Cause) → both checks are LIVE.
        assert!(
            rep.incomplete_disabled.is_none(),
            "ONT declares [reasoning] spines"
        );
        assert!(
            rep.dangling_inapplicable.is_none(),
            "ONT has query-side Symptom/Cause types"
        );
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

        // prune removes incomplete + ungrounded; stale survives WITHOUT --prune-stale.
        let (inc, ung, dang, stale) = prune(
            &g,
            &rep,
            &PruneOpts {
                incomplete: true,
                ungrounded: true,
                dangling: false,
                stale: false,
            },
        )
        .unwrap();
        assert_eq!((inc, ung, dang, stale), (1, 1, 0, 0));
        assert!(
            g.get_node(&rep.stale[0].id).unwrap().is_some(),
            "stale node survives prune unless --prune-stale is set"
        );
        assert!(g.get_node("cau:orphan").unwrap().is_none());
        assert!(g.get_node("res:b").unwrap().is_none());

        // Opt-in: --prune-stale deletes the stale bucket.
        let (_, _, _, stale2) = prune(
            &g,
            &rep,
            &PruneOpts {
                incomplete: false,
                ungrounded: false,
                dangling: false,
                stale: true,
            },
        )
        .unwrap();
        assert_eq!(stale2, 1);
        assert!(
            g.get_node(&rep.stale[0].id).unwrap().is_none(),
            "stale node is deleted when --prune-stale is set"
        );
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

        let rep = doctor(&g, &ont, &single_root(root)).unwrap();
        assert_eq!(
            rep.unverifiable, 2,
            "agent- and distil-origin ungrounded Resolution nodes must both count; curated must not"
        );
    }

    #[test]
    fn doctor_marks_incomplete_and_dangling_disabled_on_spineless_all_grounding_ontology() {
        // A "terminal-as-sink" ontology: every entity type is a grounded terminal and NO
        // [reasoning] spines are declared. `incomplete` (spine-based) and `dangling`
        // (query-side-based) are then BOTH structurally inert — doctor must mark them
        // disabled/inapplicable so the formatter prints `n/a`, not a false-all-clear `0`.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let g = GraphStore::open(root).unwrap();
        let ont = Ontology::parse(
            r#"
[entities.Entity]
requires_grounding = true
[entities.Fact]
requires_grounding = true
[relations.HAS_FACT]
from = ["Entity"]
to = ["Fact"]
role = "chaining"
[validation]
strict = false
"#,
        )
        .unwrap();
        std::fs::write(root.join("d.md"), b"v1").unwrap();
        g.upsert(
            &ont,
            &[node("ent:1", "Entity", "E", prov("d.md", None))],
            &[],
        )
        .unwrap();

        let rep = doctor(&g, &ont, &single_root(root)).unwrap();
        assert!(
            rep.incomplete_disabled.is_some(),
            "no [reasoning] spines → incomplete check disabled"
        );
        assert!(
            rep.dangling_inapplicable.is_some(),
            "every type is a grounded terminal → dangling check inapplicable"
        );
        // And the shared formatter renders `n/a`, not `0`.
        let text = crate::graph::ops::fmt_doctor_report(&rep);
        assert!(text.contains("incomplete: n/a"), "got:\n{text}");
        assert!(text.contains("dangling: n/a"), "got:\n{text}");
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

        let rep = doctor(&g, &ont, &single_root(root)).unwrap();
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

        let rep = doctor(&g, &ont, &single_root(root)).unwrap();
        assert_eq!(rep.stale.len(), 1);
        assert_eq!(rep.dangling.len(), 2);

        // dangling=false leaves both dangling nodes (and stale) untouched.
        let (inc0, ung0, dang0, stale0) = prune(
            &g,
            &rep,
            &PruneOpts {
                incomplete: false,
                ungrounded: false,
                dangling: false,
                stale: false,
            },
        )
        .unwrap();
        assert_eq!((inc0, ung0, dang0, stale0), (0, 0, 0, 0));
        assert!(g.get_node("sym:1").unwrap().is_some());
        assert!(g.get_node("cau:1").unwrap().is_some());
        assert!(
            g.get_node("res:1").unwrap().is_some(),
            "stale node survives prune unless --prune-stale is set"
        );

        // dangling=true deletes exactly the dangling nodes; stale still survives (no --prune-stale).
        let (inc1, ung1, dang1, stale1) = prune(
            &g,
            &rep,
            &PruneOpts {
                incomplete: false,
                ungrounded: false,
                dangling: true,
                stale: false,
            },
        )
        .unwrap();
        assert_eq!((inc1, ung1, dang1, stale1), (0, 0, 2, 0));
        assert!(g.get_node("sym:1").unwrap().is_none());
        assert!(g.get_node("cau:1").unwrap().is_none());
        assert!(
            g.get_node("res:1").unwrap().is_some(),
            "stale node survives even when dangling=true (needs its own --prune-stale)"
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

        let rep = doctor(&g, &ont, &single_root(root)).unwrap();
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

        let rep = doctor(&g, &ont, &single_root(root)).unwrap();
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

        let rep = doctor(&g, &ont, &single_root(root)).unwrap();
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

        let rep = doctor(&g, &ont, &single_root(root)).unwrap();
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

        let rep = doctor(&g, &ont, &single_root(root)).unwrap();
        assert!(rep.live_terminal_count > 0);
        assert_eq!(rep.dangling.len(), 1, "only the orphan Symptom must dangle");
        assert!(
            dangling_prune_risk(&rep, &g, &ont).is_none(),
            "a single dangling node in an otherwise-healthy graph must be safe to prune"
        );
    }

    #[test]
    fn dangling_prune_risk_refuses_never_built_flood_below_the_over_half_line() {
        // Mid-construction / just-changed-ontology graph: many query-side anchors have NO outgoing
        // reasoning edge yet (reason hasn't wired them) → they dangle, but they are a MINORITY of
        // the whole reasoning layer (plenty of live terminals), so neither the zero-live-terminal
        // nor the over-half trigger fires. Without the never-built trigger, `--prune-dangling` would
        // silently delete anchors `kbx reason` is about to connect. The guard must refuse (→ --force).
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let g = GraphStore::open(root).unwrap();
        let ont = Ontology::parse(ONT).unwrap();

        let doc = root.join("doc.md");
        std::fs::write(&doc, b"v1").unwrap();
        let sig0 = file_sig(&doc).unwrap();

        let mut nodes = vec![node("sec:1", "Section", "Sec", prov("doc.md", None))];
        let mut edges = vec![];
        // 21 live, grounded Resolution terminals (live MENTIONS + fresh sig) — the reasoning layer
        // is mostly healthy terminals, so the 20 dangling orphans stay well under half of it.
        for i in 0..21 {
            let id = format!("res:{i}");
            nodes.push(node(&id, "Resolution", "R", prov("doc.md", Some(sig0))));
            edges.push(edge(&id, "MENTIONS", "sec:1", prov("doc.md", None)));
        }
        // 20 never-built orphan Symptoms: no outgoing chaining edge → dangling, but NOT rot.
        for i in 0..20 {
            nodes.push(node(
                &format!("sym:orphan{i}"),
                "Symptom",
                "Orphan",
                prov("doc.md", None),
            ));
        }
        g.upsert(&ont, &nodes, &edges).unwrap();

        let rep = doctor(&g, &ont, &single_root(root)).unwrap();
        assert_eq!(rep.dangling.len(), 20, "the 20 orphan Symptoms dangle");
        assert!(
            rep.live_terminal_count >= 21,
            "21 live Resolution terminals"
        );
        let risk = dangling_prune_risk(&rep, &g, &ont);
        assert!(risk.is_some(), "a never-built flood must refuse the prune");
        let msg = risk.unwrap();
        assert!(
            msg.contains("never chained") || msg.contains("mid-construction"),
            "expected the never-built reason, got: {msg}"
        );
        assert!(
            !msg.contains("over half"),
            "must be the never-built trigger, not over-half: {msg}"
        );
    }
}
