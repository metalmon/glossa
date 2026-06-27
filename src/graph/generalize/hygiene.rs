//! Degenerate-chain prune — remove reasoning nodes that do NOT lie on a complete ontology
//! `spine` instance (and auxiliaries left orphaned by that removal), BEFORE the generalization
//! pass derives edges/communities/centrality over them.
//!
//! Fully generic: the spine relations, the reasoning ("spine") types, and the structural types
//! all come from the ontology (`spine()`, `spine_types()`, `structural()`) — no domain literals.
//! A Cause-less `Symptom→Resolution` is degenerate because it adds nothing over plain search;
//! the graph's value is the intermediate hop.

use super::Triple;
use std::collections::{HashMap, HashSet, VecDeque};

/// Ids to delete: non-structural nodes not on a complete `spine` chain and not transitively
/// attached to one. `nodes` = `(id, node_type)`; `edges` = `(from, edge_type, to)`. Returns a
/// sorted, deterministic list. Empty `spine` → empty (no-op).
pub fn incomplete_nodes(
    nodes: &[(String, String)],
    edges: &[Triple],
    spine: &[String],
    spine_types: &HashSet<String>,
    structural: &HashSet<String>,
) -> Vec<String> {
    if spine.is_empty() {
        return Vec::new();
    }
    let k = spine.len();
    let type_of: HashMap<&str, &str> =
        nodes.iter().map(|(id, ty)| (id.as_str(), ty.as_str())).collect();

    // ── Core survivors: every node on a complete chain v0 -r0-> v1 … -r_{k-1}-> vk. ──
    // Backward: can_finish[i] = nodes that can complete the suffix r_i..r_{k-1}.
    let mut can_finish: Vec<HashSet<&str>> = vec![HashSet::new(); k];
    for i in (0..k).rev() {
        let rel = &spine[i];
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
    // Forward: walk valid starts (can_finish[0]) through the spine, collecting every visited node.
    let mut survivors: HashSet<&str> = HashSet::new();
    let mut at_prev: HashSet<&str> = can_finish[0].clone();
    survivors.extend(at_prev.iter().copied());
    for i in 0..k {
        let rel = &spine[i];
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
        survivors.extend(at_next.iter().copied());
        at_prev = at_next;
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
    fn spine() -> Vec<String> {
        vec!["CAUSED_BY".into(), "RESOLVED_BY".into()]
    }
    fn spine_types() -> HashSet<String> {
        s(&["Symptom", "Cause", "Resolution"])
    }
    fn structural() -> HashSet<String> {
        s(&["Document", "Section", "Term", "Topic"])
    }

    #[test]
    fn complete_chain_survives() {
        let nodes = n(&[("S", "Symptom"), ("C", "Cause"), ("R", "Resolution")]);
        let edges = e(&[("S", "CAUSED_BY", "C"), ("C", "RESOLVED_BY", "R")]);
        assert!(incomplete_nodes(&nodes, &edges, &spine(), &spine_types(), &structural()).is_empty());
    }

    #[test]
    fn symptom_resolution_no_cause_pruned() {
        // the ticket-6 case: Symptom straight to Resolution, no Cause hop
        let nodes = n(&[("S", "Symptom"), ("R", "Resolution")]);
        let edges = e(&[("S", "RESOLVED_BY", "R")]);
        assert_eq!(
            incomplete_nodes(&nodes, &edges, &spine(), &spine_types(), &structural()),
            vec!["R".to_string(), "S".to_string()]
        );
    }

    #[test]
    fn symptom_cause_no_resolution_pruned() {
        let nodes = n(&[("S", "Symptom"), ("C", "Cause")]);
        let edges = e(&[("S", "CAUSED_BY", "C")]);
        assert_eq!(
            incomplete_nodes(&nodes, &edges, &spine(), &spine_types(), &structural()),
            vec!["C".to_string(), "S".to_string()]
        );
    }

    #[test]
    fn orphans_and_isolated_pruned() {
        let nodes = n(&[("C", "Cause"), ("R", "Resolution"), ("S", "Symptom")]);
        let edges = e(&[]); // all isolated
        assert_eq!(
            incomplete_nodes(&nodes, &edges, &spine(), &spine_types(), &structural()),
            vec!["C".to_string(), "R".to_string(), "S".to_string()]
        );
    }

    #[test]
    fn auxiliary_on_survivor_kept() {
        // S→C→R complete; R -SETS-> P(Parameter) -OF-> M(Module): both auxiliaries survive
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
        assert!(incomplete_nodes(&nodes, &edges, &spine(), &spine_types(), &structural()).is_empty());
    }

    #[test]
    fn auxiliary_behind_doomed_pruned() {
        // S→R (no Cause → both doomed); R -SETS-> P. P hangs only off a doomed node → pruned.
        let nodes =
            n(&[("S", "Symptom"), ("R", "Resolution"), ("P", "Parameter")]);
        let edges = e(&[("S", "RESOLVED_BY", "R"), ("R", "SETS", "P")]);
        assert_eq!(
            incomplete_nodes(&nodes, &edges, &spine(), &spine_types(), &structural()),
            vec!["P".to_string(), "R".to_string(), "S".to_string()]
        );
    }

    #[test]
    fn structural_never_pruned() {
        // a doomed Symptom MENTIONS a Section — the Section is structural, untouched
        let nodes = n(&[("S", "Symptom"), ("SEC", "Section")]);
        let edges = e(&[("S", "MENTIONS", "SEC")]);
        assert_eq!(
            incomplete_nodes(&nodes, &edges, &spine(), &spine_types(), &structural()),
            vec!["S".to_string()]
        );
    }

    #[test]
    fn task_culled_for_free() {
        // Task has no spine relation; its only edge is MENTIONS→Section → never reachable → pruned
        let nodes = n(&[("T", "Task"), ("SEC", "Section")]);
        let edges = e(&[("T", "MENTIONS", "SEC")]);
        assert_eq!(
            incomplete_nodes(&nodes, &edges, &spine(), &spine_types(), &structural()),
            vec!["T".to_string()]
        );
    }

    #[test]
    fn empty_spine_is_noop() {
        let nodes = n(&[("S", "Symptom")]);
        let edges = e(&[]);
        assert!(incomplete_nodes(&nodes, &edges, &[], &spine_types(), &structural()).is_empty());
    }
}
