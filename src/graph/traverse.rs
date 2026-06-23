use crate::graph::store::{Edge, GraphStore};
use std::collections::{HashSet, VecDeque};

fn type_match(e: &Edge, edge_types: Option<&[String]>) -> bool {
    match edge_types {
        None => true,
        Some(types) => types.iter().any(|t| t == &e.edge_type),
    }
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
        Provenance { source_path: "s".into(), range: None, file_sig: None, origin: "agent".into(), confidence: 1.0, created_at: 0 }
    }
    fn node(g: &GraphStore, id: &str) {
        g.put_node(&Node { id: id.into(), node_type: "Entity".into(), label: id.into(), aliases: vec![], prov: prov() }).unwrap();
    }
    fn edge(g: &GraphStore, from: &str, to: &str, ty: &str) {
        g.put_edge(&Edge { from: from.into(), to: to.into(), edge_type: ty.into(), prov: prov() }).unwrap();
    }

    #[test]
    fn neighbors_respects_depth_and_type() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        for id in ["a", "b", "c", "d"] { node(&g, id); }
        edge(&g, "a", "b", "REL");
        edge(&g, "b", "c", "REL");
        edge(&g, "a", "d", "OTHER");

        let d1 = neighbors(&g, "a", None, 1).unwrap();
        assert!(d1.contains(&"b".to_string()) && d1.contains(&"d".to_string()) && !d1.contains(&"c".to_string()));

        let d2 = neighbors(&g, "a", None, 2).unwrap();
        assert!(d2.contains(&"c".to_string()));

        let only_rel = neighbors(&g, "a", Some(&["REL".to_string()]), 1).unwrap();
        assert_eq!(only_rel, vec!["b".to_string()]);
    }

    #[test]
    fn path_finds_chain() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        for id in ["a", "b", "c"] { node(&g, id); }
        edge(&g, "a", "b", "REL");
        edge(&g, "b", "c", "REL");
        assert_eq!(path(&g, "a", "c", 5).unwrap(), Some(vec!["a".into(), "b".into(), "c".into()]));
        assert_eq!(path(&g, "a", "z", 5).unwrap(), None);
    }
}
