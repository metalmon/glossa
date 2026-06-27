//! Degenerate-chain prune — remove reasoning nodes that do NOT lie on a complete instance of any
//! ontology reasoning `spine` (and auxiliaries left orphaned by that removal), BEFORE the
//! generalization pass derives edges/communities/centrality over them.
//!
//! Fully generic: the spines (each an anchor node type + ordered relations), the reasoning
//! ("spine") types, and the structural types all come from the ontology — no domain literals.
//! Multiple spines coexist: a causal `Symptom -CAUSED_BY-> Cause -RESOLVED_BY-> Resolution` and an
//! informational `Task -RESOLVED_BY-> Resolution` are BOTH complete, so how-to/reference cases
//! survive while a Cause-less `Symptom -RESOLVED_BY-> Resolution` (which matches no spine's anchor)
//! is still pruned — it adds nothing over plain search.

use super::Triple;
use crate::graph::ontology::Spine;
use std::collections::{HashMap, HashSet, VecDeque};

/// Ids to delete: non-structural nodes not on a complete instance of ANY `spine`, and not
/// transitively attached to one. `nodes` = `(id, node_type)`; `edges` = `(from, edge_type, to)`.
/// Returns a sorted, deterministic list. No spines → empty (no-op).
pub fn incomplete_nodes(
    nodes: &[(String, String)],
    edges: &[Triple],
    spines: &[Spine],
    spine_types: &HashSet<String>,
    structural: &HashSet<String>,
) -> Vec<String> {
    if spines.is_empty() {
        return Vec::new();
    }
    let type_of: HashMap<&str, &str> =
        nodes.iter().map(|(id, ty)| (id.as_str(), ty.as_str())).collect();

    // ── Core survivors: union, over every declared spine, of the nodes on a complete typed chain. ──
    let mut survivors: HashSet<&str> = HashSet::new();
    for spine in spines {
        survivors.extend(survivors_for_spine(spine, edges, &type_of));
    }

    // ── Keep set: BFS from core survivors over non-structural edges, never ENTERING a doomed
    // spine-type node (a bridge about to be deleted must not rescue what hangs off it). ──
    let is_structural = |id: &str| type_of.get(id).is_some_and(|t| structural.contains(*t));
    let is_doomed = |id: &str| {
        type_of.get(id).is_some_and(|t| spine_types.contains(*t)) && !survivors.contains(id)
    };
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for (f, _et, t) in edges {
        if is_structural(f) || is_structural(t) {
            continue;
        }
        adj.entry(f).or_default().push(t);
        adj.entry(t).or_default().push(f);
    }
    let mut keep: HashSet<&str> = survivors.clone();
    let mut queue: VecDeque<&str> = survivors.iter().copied().collect();
    while let Some(n) = queue.pop_front() {
        for &m in adj.get(n).map(Vec::as_slice).unwrap_or(&[]) {
            if keep.contains(m) || is_doomed(m) {
                continue;
            }
            keep.insert(m);
            queue.push_back(m);
        }
    }

    // ── Delete = non-structural nodes not kept. ──
    let mut del: Vec<String> = nodes
        .iter()
        .filter(|(id, ty)| !structural.contains(ty) && !keep.contains(id.as_str()))
        .map(|(id, _)| id.clone())
        .collect();
    del.sort();
    del
}

/// Nodes on a complete instance of ONE typed spine: the start must have the spine's `anchor` type
/// AND reach the end through the relation sequence. Returned ids borrow `edges`.
fn survivors_for_spine<'e>(
    spine: &Spine,
    edges: &'e [Triple],
    type_of: &HashMap<&str, &str>,
) -> HashSet<&'e str> {
    let rels = &spine.relations;
    let k = rels.len();
    let mut out: HashSet<&str> = HashSet::new();
    if k == 0 {
        return out;
    }
    // Backward: can_finish[i] = nodes that can complete the suffix rels[i..].
    let mut can_finish: Vec<HashSet<&str>> = vec![HashSet::new(); k];
    for i in (0..k).rev() {
        let rel = &rels[i];
        for (f, et, t) in edges {
            if et != rel {
                continue;
            }
            let m_ok = i + 1 == k || can_finish[i + 1].contains(t.as_str());
            if m_ok {
                can_finish[i].insert(f.as_str());
            }
        }
    }
    // Forward: valid starts = nodes of the anchor type that can finish the whole sequence.
    let mut at_prev: HashSet<&str> = can_finish[0]
        .iter()
        .copied()
        .filter(|n| type_of.get(n).is_some_and(|t| *t == spine.anchor))
        .collect();
    out.extend(at_prev.iter().copied());
    for i in 0..k {
        let rel = &rels[i];
        let mut at_next: HashSet<&str> = HashSet::new();
        for (f, et, t) in edges {
            if et != rel || !at_prev.contains(f.as_str()) {
                continue;
            }
            let m_ok = i + 1 == k || can_finish[i + 1].contains(t.as_str());
            if m_ok {
                at_next.insert(t.as_str());
            }
        }
        out.extend(at_next.iter().copied());
        at_prev = at_next;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(items: &[&str]) -> HashSet<String> {
        items.iter().map(|x| x.to_string()).collect()
    }
    fn n(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
    }
    fn e(triples: &[(&str, &str, &str)]) -> Vec<Triple> {
        triples.iter().map(|(a, b, c)| (a.to_string(), b.to_string(), c.to_string())).collect()
    }
    fn sp(anchor: &str, rels: &[&str]) -> Spine {
        Spine { anchor: anchor.into(), relations: rels.iter().map(|r| r.to_string()).collect() }
    }
    // The support overlay: a causal spine and an informational (Task) spine.
    fn spines() -> Vec<Spine> {
        vec![sp("Symptom", &["CAUSED_BY", "RESOLVED_BY"]), sp("Task", &["RESOLVED_BY"])]
    }
    // RESOLVED_BY.from = [Symptom, Cause, Task] → all four are spine types.
    fn spine_types() -> HashSet<String> {
        s(&["Symptom", "Cause", "Resolution", "Task"])
    }
    fn structural() -> HashSet<String> {
        s(&["Document", "Section", "Term", "Topic"])
    }

    #[test]
    fn complete_chain_survives() {
        let nodes = n(&[("S", "Symptom"), ("C", "Cause"), ("R", "Resolution")]);
        let edges = e(&[("S", "CAUSED_BY", "C"), ("C", "RESOLVED_BY", "R")]);
        assert!(incomplete_nodes(&nodes, &edges, &spines(), &spine_types(), &structural()).is_empty());
    }

    #[test]
    fn task_resolution_survives() {
        // the informational shape: Task -RESOLVED_BY-> Resolution is complete on the Task spine
        let nodes = n(&[("T", "Task"), ("R", "Resolution")]);
        let edges = e(&[("T", "RESOLVED_BY", "R")]);
        assert!(incomplete_nodes(&nodes, &edges, &spines(), &spine_types(), &structural()).is_empty());
    }

    #[test]
    fn symptom_resolution_no_cause_pruned() {
        // a Cause-less Symptom→Resolution matches NEITHER spine (Symptom needs CAUSED_BY first;
        // the Task spine's anchor is Task, not Symptom) → both pruned.
        let nodes = n(&[("S", "Symptom"), ("R", "Resolution")]);
        let edges = e(&[("S", "RESOLVED_BY", "R")]);
        assert_eq!(
            incomplete_nodes(&nodes, &edges, &spines(), &spine_types(), &structural()),
            vec!["R".to_string(), "S".to_string()]
        );
    }

    #[test]
    fn symptom_cause_no_resolution_pruned() {
        let nodes = n(&[("S", "Symptom"), ("C", "Cause")]);
        let edges = e(&[("S", "CAUSED_BY", "C")]);
        assert_eq!(
            incomplete_nodes(&nodes, &edges, &spines(), &spine_types(), &structural()),
            vec!["C".to_string(), "S".to_string()]
        );
    }

    #[test]
    fn orphans_and_isolated_pruned() {
        let nodes = n(&[("C", "Cause"), ("R", "Resolution"), ("S", "Symptom"), ("T", "Task")]);
        let edges = e(&[]); // all isolated
        assert_eq!(
            incomplete_nodes(&nodes, &edges, &spines(), &spine_types(), &structural()),
            vec!["C".to_string(), "R".to_string(), "S".to_string(), "T".to_string()]
        );
    }

    #[test]
    fn auxiliary_on_survivor_kept() {
        let nodes = n(&[
            ("S", "Symptom"),
            ("C", "Cause"),
            ("R", "Resolution"),
            ("P", "Parameter"),
            ("M", "Module"),
        ]);
        let edges = e(&[
            ("S", "CAUSED_BY", "C"),
            ("C", "RESOLVED_BY", "R"),
            ("R", "SETS", "P"),
            ("P", "OF", "M"),
        ]);
        assert!(incomplete_nodes(&nodes, &edges, &spines(), &spine_types(), &structural()).is_empty());
    }

    #[test]
    fn auxiliary_behind_doomed_pruned() {
        // S→R (no Cause → both doomed); R -SETS-> P. P hangs only off a doomed node → pruned.
        let nodes = n(&[("S", "Symptom"), ("R", "Resolution"), ("P", "Parameter")]);
        let edges = e(&[("S", "RESOLVED_BY", "R"), ("R", "SETS", "P")]);
        assert_eq!(
            incomplete_nodes(&nodes, &edges, &spines(), &spine_types(), &structural()),
            vec!["P".to_string(), "R".to_string(), "S".to_string()]
        );
    }

    #[test]
    fn structural_never_pruned() {
        let nodes = n(&[("S", "Symptom"), ("SEC", "Section")]);
        let edges = e(&[("S", "MENTIONS", "SEC")]);
        assert_eq!(
            incomplete_nodes(&nodes, &edges, &spines(), &spine_types(), &structural()),
            vec!["S".to_string()]
        );
    }

    #[test]
    fn lone_task_without_resolution_pruned() {
        // a Task whose only edge is MENTIONS→Section is not on the Task spine → pruned; Section kept
        let nodes = n(&[("T", "Task"), ("SEC", "Section")]);
        let edges = e(&[("T", "MENTIONS", "SEC")]);
        assert_eq!(
            incomplete_nodes(&nodes, &edges, &spines(), &spine_types(), &structural()),
            vec!["T".to_string()]
        );
    }

    #[test]
    fn no_spines_is_noop() {
        let nodes = n(&[("S", "Symptom")]);
        let edges = e(&[]);
        assert!(incomplete_nodes(&nodes, &edges, &[], &spine_types(), &structural()).is_empty());
    }
}
