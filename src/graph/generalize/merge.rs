//! Tier-1 — Near-duplicate MERGE grouping. The similarity / shared-evidence passes emit candidate
//! pairs; this collapses them (union-find) into clusters of near-dup nodes. The integration layer
//! then picks a canonical node per group (e.g. the shortest label), keeps the others' labels as
//! `aliases`, and reattaches edges — so routing and edge references become unambiguous.

use std::collections::HashMap;

fn find(parent: &mut [usize], x: usize) -> usize {
    let mut r = x;
    while parent[r] != r {
        r = parent[r];
    }
    let mut c = x;
    while parent[c] != r {
        let next = parent[c];
        parent[c] = r;
        c = next;
    }
    r
}

/// Union-find over `similar_pairs` → groups of ≥2 node ids (singletons dropped). Pairs referencing
/// ids absent from `node_ids` are ignored. Deterministic: each group sorted, groups sorted.
pub fn merge_groups(node_ids: &[String], similar_pairs: &[(String, String)]) -> Vec<Vec<String>> {
    let mut ids: Vec<&String> = node_ids.iter().collect();
    ids.sort();
    ids.dedup();
    let index: HashMap<&String, usize> = ids.iter().enumerate().map(|(i, s)| (*s, i)).collect();
    let mut parent: Vec<usize> = (0..ids.len()).collect();
    for (a, b) in similar_pairs {
        if let (Some(&ia), Some(&ib)) = (index.get(a), index.get(b)) {
            let (ra, rb) = (find(&mut parent, ia), find(&mut parent, ib));
            if ra != rb {
                parent[ra] = rb;
            }
        }
    }
    let mut groups: HashMap<usize, Vec<String>> = HashMap::new();
    for (i, s) in ids.iter().enumerate() {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push((*s).clone());
    }
    let mut out: Vec<Vec<String>> = groups.into_values().filter(|g| g.len() >= 2).collect();
    for g in out.iter_mut() {
        g.sort();
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn s(x: &str) -> String {
        x.into()
    }

    #[test]
    fn union_find_groups_transitive_pairs_drops_singletons() {
        let nodes = vec![s("a"), s("b"), s("c"), s("d")];
        let pairs = vec![(s("a"), s("b")), (s("b"), s("c"))];
        // a-b-c collapse into one group; d stays a singleton → dropped
        assert_eq!(merge_groups(&nodes, &pairs), vec![vec![s("a"), s("b"), s("c")]]);
    }

    #[test]
    fn disjoint_pairs_form_separate_groups() {
        let nodes = vec![s("a"), s("b"), s("c"), s("d")];
        let pairs = vec![(s("a"), s("b")), (s("c"), s("d"))];
        assert_eq!(
            merge_groups(&nodes, &pairs),
            vec![vec![s("a"), s("b")], vec![s("c"), s("d")]]
        );
    }

    #[test]
    fn unknown_ids_ignored() {
        let nodes = vec![s("a"), s("b")];
        let pairs = vec![(s("a"), s("ghost"))];
        assert!(merge_groups(&nodes, &pairs).is_empty());
    }
}
