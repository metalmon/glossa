//! Ontology-agnostic Personalized PageRank (random walk with restart) over the reasoning graph.
//!
//! The graph is treated as a generic weighted digraph: `edge_type` and `node_type` are NEVER read.
//! Mass starts on the question's lexical seed nodes and flows across edges, ranking every node by
//! CONNECTIVITY to the question's neighborhood. This floats the terminal node of a multi-hop chain
//! even when it shares no tokens with the question — the thing lexical ranking cannot do — and it
//! disperses high-degree hubs for free (a PageRank property), so no df/hop caps are needed.

use crate::graph::store::GraphStore;
use std::collections::HashMap;

/// A symmetric transition structure built once from the graph's nodes + edges. Ontology-blind:
/// every stored edge becomes a bidirectional transition of equal weight (confidence-weighting is a
/// future knob, deliberately not wired here).
pub struct Transition {
    ids: Vec<String>,            // idx -> node id
    idx: HashMap<String, usize>, // node id -> idx
    adj: Vec<Vec<usize>>,        // undirected adjacency (idx -> neighbor idxs, with multiplicity)
}

impl Transition {
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    fn index_of(&self, id: &str) -> Option<usize> {
        self.idx.get(id).copied()
    }
    /// Node ids in walk order (for persistence).
    pub fn ids(&self) -> &[String] {
        &self.ids
    }
    /// Undirected adjacency in walk order (for persistence).
    pub fn adj(&self) -> &[Vec<usize>] {
        &self.adj
    }
    /// Reassemble from persisted `(ids, adj)`; the id->index map is derived, not stored.
    pub fn from_parts(ids: Vec<String>, adj: Vec<Vec<usize>>) -> Transition {
        let idx = ids.iter().cloned().enumerate().map(|(i, s)| (s, i)).collect();
        Transition { ids, idx, adj }
    }
}

/// Load every node + edge into a symmetric adjacency. Edges whose endpoints aren't both present, and
/// self-loops, are dropped. No ontology is consulted.
pub fn build_transition(g: &GraphStore) -> anyhow::Result<Transition> {
    let nodes = g.all_nodes()?;
    let mut idx: HashMap<String, usize> = HashMap::with_capacity(nodes.len());
    let mut ids: Vec<String> = Vec::with_capacity(nodes.len());
    for n in &nodes {
        idx.entry(n.id.clone()).or_insert_with(|| {
            ids.push(n.id.clone());
            ids.len() - 1
        });
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); ids.len()];
    for e in g.all_edges()? {
        if let (Some(&a), Some(&b)) = (idx.get(&e.from), idx.get(&e.to)) {
            if a != b {
                adj[a].push(b);
                adj[b].push(a); // symmetric — a multi-hop answer may need to walk "backward"
            }
        }
    }
    Ok(Transition { ids, idx, adj })
}

/// Random walk with restart to a stationary distribution. `seeds` maps node id -> unnormalized mass
/// and is normalized to the restart vector `p`. `alpha` is the restart probability. Returns
/// (node id, stationary score) sorted descending, skipping zero-mass nodes. Dangling nodes (no
/// edges) teleport their mass back to `p` so total mass is conserved every iteration.
pub fn ppr(
    trans: &Transition,
    seeds: &HashMap<String, f32>,
    alpha: f32,
    max_iter: usize,
    eps: f32,
) -> Vec<(String, f32)> {
    let n = trans.len();
    if n == 0 {
        return Vec::new();
    }
    // Restart vector p: normalized seed mass.
    let mut p = vec![0.0f32; n];
    let mut sum = 0.0f32;
    for (id, &m) in seeds {
        if m > 0.0 {
            if let Some(i) = trans.index_of(id) {
                p[i] += m;
                sum += m;
            }
        }
    }
    if sum <= 0.0 {
        return Vec::new(); // no seed landed on a real node
    }
    for v in p.iter_mut() {
        *v /= sum;
    }

    let mut r = p.clone();
    for _ in 0..max_iter {
        let mut next = vec![0.0f32; n];
        let mut dangling = 0.0f32;
        for i in 0..n {
            let deg = trans.adj[i].len();
            if deg == 0 {
                dangling += r[i];
            } else {
                let share = r[i] / deg as f32;
                for &j in &trans.adj[i] {
                    next[j] += share;
                }
            }
        }
        // r' = (1-alpha)*(M r) + [alpha + (1-alpha)*dangling]*p  — teleport keeps sum(r)=1.
        let teleport = alpha + (1.0 - alpha) * dangling;
        let mut delta = 0.0f32;
        for i in 0..n {
            let v = (1.0 - alpha) * next[i] + teleport * p[i];
            delta += (v - r[i]).abs();
            next[i] = v;
        }
        r = next;
        if delta < eps {
            break;
        }
    }

    let mut out: Vec<(String, f32)> = r
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0.0)
        .map(|(i, &v)| (trans.ids[i].clone(), v))
        .collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::store::{Edge, GraphStore, Node, Provenance};

    fn prov() -> Provenance {
        Provenance {
            source_path: "s.md".into(),
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
            node_type: "Fact".into(),
            label: id.into(),
            aliases: vec![],
            prov: prov(),
        })
        .unwrap();
    }
    fn link(g: &GraphStore, a: &str, b: &str) {
        g.put_edge(&Edge { from: a.into(), to: b.into(), edge_type: "LEADS_TO".into(), prov: prov() })
            .unwrap();
    }
    fn seed(ids: &[(&str, f32)]) -> HashMap<String, f32> {
        ids.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }
    fn rank_of(out: &[(String, f32)], id: &str) -> Option<usize> {
        out.iter().position(|(i, _)| i == id)
    }

    #[test]
    fn connected_terminal_outranks_a_disconnected_node() {
        // seed -> bridge -> terminal; `far` is an isolated node. The terminal shares no seed mass
        // directly, but connectivity carries mass to it; the disconnected node stays at zero.
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        for x in ["seed", "bridge", "terminal", "far"] {
            node(&g, x);
        }
        link(&g, "seed", "bridge");
        link(&g, "bridge", "terminal");
        let t = build_transition(&g).unwrap();
        let out = ppr(&t, &seed(&[("seed", 1.0)]), 0.15, 50, 1e-6);
        assert!(rank_of(&out, "terminal").is_some(), "terminal reached: {out:?}");
        assert!(rank_of(&out, "far").is_none(), "disconnected node gets no mass: {out:?}");
    }

    #[test]
    fn multi_path_answer_outranks_hub_leaf() {
        // `answer` is reachable from the seed two ways (direct and via the hub); `noise*` hang off
        // the hub only. Even though the hub has high degree, the answer's extra path floats it above
        // any single hub leaf — the property the df-cap hack approximated.
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        node(&g, "seed");
        node(&g, "hub");
        node(&g, "answer");
        for i in 0..10 {
            node(&g, &format!("noise{i}"));
            link(&g, "hub", &format!("noise{i}"));
        }
        link(&g, "seed", "hub");
        link(&g, "seed", "answer");
        link(&g, "hub", "answer");
        let t = build_transition(&g).unwrap();
        let out = ppr(&t, &seed(&[("seed", 1.0)]), 0.15, 50, 1e-6);
        let ra = rank_of(&out, "answer").unwrap();
        let rn = rank_of(&out, "noise0").unwrap();
        assert!(ra < rn, "answer outranks a hub leaf: {out:?}");
    }

    #[test]
    fn shared_neighbor_of_two_seeds_outranks_single_seed_neighbor() {
        // Cone-intersection intuition, soft: `both` neighbors both seeds; `one` neighbors only s1.
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        for x in ["s1", "s2", "both", "one"] {
            node(&g, x);
        }
        link(&g, "s1", "both");
        link(&g, "s2", "both");
        link(&g, "s1", "one");
        let t = build_transition(&g).unwrap();
        let out = ppr(&t, &seed(&[("s1", 1.0), ("s2", 1.0)]), 0.15, 50, 1e-6);
        assert!(
            rank_of(&out, "both").unwrap() < rank_of(&out, "one").unwrap(),
            "intersection node ranks above the single-seed neighbor: {out:?}"
        );
    }

    #[test]
    fn stationary_distribution_is_normalized() {
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        for x in ["a", "b", "c"] {
            node(&g, x);
        }
        link(&g, "a", "b");
        link(&g, "b", "c");
        let t = build_transition(&g).unwrap();
        let out = ppr(&t, &seed(&[("a", 1.0)]), 0.15, 100, 1e-9);
        let total: f32 = out.iter().map(|(_, v)| v).sum();
        assert!((total - 1.0).abs() < 1e-3, "mass conserved to ~1.0, got {total}");
    }

    #[test]
    fn empty_seed_yields_nothing() {
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        node(&g, "a");
        let t = build_transition(&g).unwrap();
        assert!(ppr(&t, &seed(&[("nonexistent", 1.0)]), 0.15, 50, 1e-6).is_empty());
    }
}
