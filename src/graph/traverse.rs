use crate::graph::store::{Edge, GraphStore};
use std::collections::{HashSet, VecDeque};

fn type_match(e: &Edge, edge_types: Option<&[String]>) -> bool {
    match edge_types {
        None => true,
        Some(types) => types.iter().any(|t| t == &e.edge_type),
    }
}

/// One step of a `path_between` result.
#[derive(Debug, Clone, PartialEq)]
pub struct Hop {
    /// The node reached at this step.
    pub node: String,
    /// How we reached it from the previous hop: `(edge_type, forward)`. `forward=true` means the
    /// edge points prev->node (we followed an outgoing edge); `false` means node->prev (incoming).
    /// `None` only for the first hop (the `from` node).
    pub via: Option<(String, bool)>,
}

/// Undirected shortest path from `from` to `to`: every edge is walked in both directions for
/// connectivity, but each hop records the edge's real direction. Returns the hop chain (including
/// `from` with `via=None`), or `None` if `to` is unreachable within `max_depth` edges.
pub fn path_between(
    g: &GraphStore,
    from: &str,
    to: &str,
    max_depth: usize,
) -> anyhow::Result<Option<Vec<Hop>>> {
    if from == to {
        return Ok(Some(vec![Hop { node: from.to_string(), via: None }]));
    }
    let mut visited: HashSet<String> = HashSet::from([from.to_string()]);
    let mut q: VecDeque<Vec<Hop>> =
        VecDeque::from([vec![Hop { node: from.to_string(), via: None }]]);
    while let Some(p) = q.pop_front() {
        // p.len()-1 == edges walked so far; stop expanding once we'd exceed max_depth edges.
        if p.len() > max_depth {
            continue;
        }
        let last = p.last().unwrap().node.clone();
        let mut steps: Vec<(String, String, bool)> = Vec::new();
        for e in g.outgoing(&last)? {
            steps.push((e.to, e.edge_type, true));
        }
        for e in g.incoming(&last)? {
            steps.push((e.from, e.edge_type, false));
        }
        for (next, et, fwd) in steps {
            if next == to {
                let mut found = p.clone();
                found.push(Hop { node: next, via: Some((et, fwd)) });
                return Ok(Some(found));
            }
            if visited.insert(next.clone()) {
                let mut np = p.clone();
                np.push(Hop { node: next, via: Some((et, fwd)) });
                q.push_back(np);
            }
        }
    }
    Ok(None)
}

pub fn neighbors(
    g: &GraphStore,
    from: &str,
    edge_types: Option<&[String]>,
    depth: usize,
) -> anyhow::Result<Vec<String>> {
    let mut visited: HashSet<String> = HashSet::from([from.to_string()]);
    let mut frontier: VecDeque<(String, usize)> = VecDeque::from([(from.to_string(), 0)]);
    let mut out = Vec::new();
    while let Some((node, d)) = frontier.pop_front() {
        if d >= depth {
            continue;
        }
        for e in g.outgoing(&node)? {
            if !type_match(&e, edge_types) {
                continue;
            }
            if visited.insert(e.to.clone()) {
                out.push(e.to.clone());
                frontier.push_back((e.to, d + 1));
            }
        }
    }
    Ok(out)
}

pub fn path(
    g: &GraphStore,
    from: &str,
    to: &str,
    max_depth: usize,
) -> anyhow::Result<Option<Vec<String>>> {
    if from == to {
        return Ok(Some(vec![from.to_string()]));
    }
    let mut visited: HashSet<String> = HashSet::from([from.to_string()]);
    let mut q: VecDeque<Vec<String>> = VecDeque::from([vec![from.to_string()]]);
    while let Some(p) = q.pop_front() {
        if p.len() > max_depth {
            continue;
        }
        let last = p.last().unwrap().clone();
        for e in g.outgoing(&last)? {
            if e.to == to {
                let mut found = p.clone();
                found.push(e.to);
                return Ok(Some(found));
            }
            if visited.insert(e.to.clone()) {
                let mut np = p.clone();
                np.push(e.to);
                q.push_back(np);
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::store::{Edge, GraphStore, Node, Provenance};

    fn prov() -> Provenance {
        Provenance {
            source_path: "s".into(),
            range: None,
            file_sig: None,
            origin: "agent".into(),
            confidence: 1.0,
            created_at: 0,
        }
    }
    fn node(g: &GraphStore, id: &str) {
        g.put_node(&Node {
            id: id.into(),
            node_type: "Entity".into(),
            label: id.into(),
            aliases: vec![],
            prov: prov(),
        })
        .unwrap();
    }
    fn edge(g: &GraphStore, from: &str, to: &str, ty: &str) {
        g.put_edge(&Edge {
            from: from.into(),
            to: to.into(),
            edge_type: ty.into(),
            prov: prov(),
        })
        .unwrap();
    }

    #[test]
    fn neighbors_respects_depth_and_type() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        for id in ["a", "b", "c", "d"] {
            node(&g, id);
        }
        edge(&g, "a", "b", "REL");
        edge(&g, "b", "c", "REL");
        edge(&g, "a", "d", "OTHER");

        let d1 = neighbors(&g, "a", None, 1).unwrap();
        assert!(
            d1.contains(&"b".to_string())
                && d1.contains(&"d".to_string())
                && !d1.contains(&"c".to_string())
        );

        let d2 = neighbors(&g, "a", None, 2).unwrap();
        assert!(d2.contains(&"c".to_string()));

        let only_rel = neighbors(&g, "a", Some(&["REL".to_string()]), 1).unwrap();
        assert_eq!(only_rel, vec!["b".to_string()]);
    }

    #[test]
    fn path_finds_chain() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        for id in ["a", "b", "c"] {
            node(&g, id);
        }
        edge(&g, "a", "b", "REL");
        edge(&g, "b", "c", "REL");
        assert_eq!(
            path(&g, "a", "c", 5).unwrap(),
            Some(vec!["a".into(), "b".into(), "c".into()])
        );
        assert_eq!(path(&g, "a", "z", 5).unwrap(), None);
    }

    #[test]
    fn path_between_is_undirected_and_records_direction() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        for id in ["a", "b", "c"] {
            node(&g, id);
        }
        // Only forward edges a->b->c exist.
        edge(&g, "a", "b", "REL");
        edge(&g, "b", "c", "REL");

        // Forward reachable, hops carry forward=true.
        let fwd = path_between(&g, "a", "c", 5).unwrap().unwrap();
        let nodes: Vec<&str> = fwd.iter().map(|h| h.node.as_str()).collect();
        assert_eq!(nodes, vec!["a", "b", "c"]);
        assert_eq!(fwd[0].via, None);
        assert_eq!(fwd[1].via, Some(("REL".to_string(), true)));
        assert_eq!(fwd[2].via, Some(("REL".to_string(), true)));

        // Reverse direction is found too (undirected), with forward=false.
        let rev = path_between(&g, "c", "a", 5).unwrap().unwrap();
        let rnodes: Vec<&str> = rev.iter().map(|h| h.node.as_str()).collect();
        assert_eq!(rnodes, vec!["c", "b", "a"]);
        assert_eq!(rev[1].via, Some(("REL".to_string(), false)));

        // Unreachable within depth.
        node(&g, "z");
        assert_eq!(path_between(&g, "a", "z", 5).unwrap(), None);
        // Same node.
        let same = path_between(&g, "a", "a", 5).unwrap().unwrap();
        assert_eq!(same.len(), 1);
        assert_eq!(same[0].node, "a");
    }
}
