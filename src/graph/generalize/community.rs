//! #6 — Community detection via undirected connected components (union-find).
//! The simplest, fully-deterministic member of the family; Louvain / label-propagation can later
//! replace the body behind the same `node_id -> community_id` output.

use super::Triple;
use std::collections::HashMap;

fn find(parent: &mut [usize], x: usize) -> usize {
    let mut r = x;
    while parent[r] != r {
        r = parent[r];
    }
    // path compression
    let mut c = x;
    while parent[c] != r {
        let next = parent[c];
        parent[c] = r;
        c = next;
    }
    r
}

/// Assign each node a dense 0-based community id by undirected connected components. Isolated nodes
/// each get their own id. Deterministic: component ids are assigned in sorted node order. Edges
/// referencing unknown ids are ignored.
pub fn connected_components(node_ids: &[String], edges: &[Triple]) -> HashMap<String, usize> {
    let mut ids: Vec<&String> = node_ids.iter().collect();
    ids.sort();
    ids.dedup();
    let index: HashMap<&String, usize> = ids.iter().enumerate().map(|(i, s)| (*s, i)).collect();
    let mut parent: Vec<usize> = (0..ids.len()).collect();
    for (f, _t, to) in edges {
        if let (Some(&a), Some(&b)) = (index.get(f), index.get(to)) {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                parent[ra] = rb;
            }
        }
    }
    let mut root_to_comm: HashMap<usize, usize> = HashMap::new();
    let mut out = HashMap::new();
    for (i, s) in ids.iter().enumerate() {
        let r = find(&mut parent, i);
        let next = root_to_comm.len();
        let comm = *root_to_comm.entry(r).or_insert(next);
        out.insert((*s).clone(), comm);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn s(x: &str) -> String {
        x.into()
    }
    fn t(a: &str, b: &str) -> Triple {
        (a.into(), "REL".into(), b.into())
    }

    #[test]
    fn two_clusters_get_distinct_ids_isolated_its_own() {
        let nodes = vec![s("a"), s("b"), s("c"), s("d"), s("lonely")];
        let edges = vec![t("a", "b"), t("c", "d")];
        let comm = connected_components(&nodes, &edges);
        assert_eq!(comm["a"], comm["b"]);
        assert_eq!(comm["c"], comm["d"]);
        assert_ne!(comm["a"], comm["c"]);
        assert_ne!(comm["lonely"], comm["a"]);
        assert_ne!(comm["lonely"], comm["c"]);
    }

    #[test]
    fn transitive_chain_is_one_component() {
        let nodes = vec![s("a"), s("b"), s("c")];
        let edges = vec![t("a", "b"), t("b", "c")];
        let comm = connected_components(&nodes, &edges);
        assert_eq!(comm["a"], comm["b"]);
        assert_eq!(comm["b"], comm["c"]);
    }
}
