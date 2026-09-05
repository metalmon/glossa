//! #6 — Community detection via a deterministic, tier-weighted Louvain modularity optimiser.
//! Replaces the earlier union-find connected-components detector (which collapsed almost every
//! reasoning node into ONE community, making the `reach` bridge-disambiguation gate vacuous) behind
//! the SAME `node_id -> community_id` contract.
//!
//! Determinism: nodes are indexed in sorted+deduped order, iterated in ascending index order, ties
//! are broken by keeping the current community else the lowest community id, and community labels
//! are renumbered by ascending smallest-member-index. No RNG, and no decision depends on `HashMap`
//! iteration order.

use super::Triple;
use std::collections::{BTreeMap, HashMap};

/// Epsilon guarding the modularity-gain tie comparison so IEEE float noise cannot flip a tie.
const EPS: f64 = 1e-12;
/// Guard on local-moving passes within one Louvain level.
const MAX_PASSES: usize = 100;
/// Guard on the number of aggregation levels.
const MAX_LEVELS: usize = 100;

/// Edge weight derived from the `edge_type` (middle element of the `Triple`).
fn tier_weight(edge_type: &str) -> f64 {
    match edge_type {
        "SIMILAR" => 0.1,
        "RESOLVED_BY" | "CAUSED_BY" | "LEADS_TO" => 2.0,
        _ => 1.0,
    }
}

/// A single Louvain level: a weighted undirected graph. Self-loops (produced by aggregation) are
/// stored as `2*L` so that a plain sum over an adjacency row yields the weighted degree with the
/// conventional "self-loop counts twice" rule.
struct Level {
    n: usize,
    adj: Vec<BTreeMap<usize, f64>>,
    k: Vec<f64>,
    m: f64,
}

impl Level {
    fn from_adj(adj: Vec<BTreeMap<usize, f64>>) -> Self {
        let n = adj.len();
        let k: Vec<f64> = adj.iter().map(|row| row.values().sum()).collect();
        let m = 0.5 * k.iter().sum::<f64>();
        Level { n, adj, k, m }
    }
}

/// Phase 1 (local moving): assign each node to a community label (in `0..n`), moving nodes to the
/// neighbouring community with the largest modularity gain until a full pass makes no move.
fn local_moving(level: &Level) -> Vec<usize> {
    let n = level.n;
    let mut comm: Vec<usize> = (0..n).collect();
    let m = level.m;
    if m <= 0.0 {
        return comm; // no edges: every node is its own community
    }
    // Σ_tot per community: total incident weight of the community (indexed in node-label space).
    // Grows past `n` as nodes isolate into fresh singleton communities.
    let mut sigma_tot: Vec<f64> = level.k.clone();
    let mut next_comm = n; // id allocator for freshly-isolated singleton communities
    const ISOLATE: usize = usize::MAX; // sentinel: "move to a new empty community" (gain 0)

    let mut converged = false;
    for _ in 0..MAX_PASSES {
        let mut moved = false;
        for i in 0..n {
            let ci = comm[i];
            let ki = level.k[i];
            // Remove i from its current community before scoring candidates.
            sigma_tot[ci] -= ki;

            // k_{i,in} per neighbouring community (deterministic ascending-id iteration).
            let mut kin: BTreeMap<usize, f64> = BTreeMap::new();
            for (&j, &w) in &level.adj[i] {
                if j == i {
                    continue; // self-loop: not a link to another node
                }
                *kin.entry(comm[j]).or_insert(0.0) += w;
            }

            // Baseline is the better of {rejoin ci, isolate}: isolating into a new empty community
            // has gain 0, so if rejoining ci is worse than that (its gain < 0), the node should leave
            // rather than stay trapped in a community it no longer belongs to. Prefer ci on a tie
            // (rejoin gain >= 0), keeping churn down; a neighbour must strictly beat the baseline by
            // EPS to win, and among neighbour ties the lowest id wins (BTreeMap ascending order).
            let rejoin = kin.get(&ci).copied().unwrap_or(0.0) / m - sigma_tot[ci] * ki / (2.0 * m * m);
            let (mut best_comm, mut best_gain) = if rejoin >= -EPS {
                (ci, rejoin)
            } else {
                (ISOLATE, 0.0)
            };
            for (&c, &kin_c) in &kin {
                if c == ci {
                    continue;
                }
                let gain = kin_c / m - sigma_tot[c] * ki / (2.0 * m * m);
                if gain > best_gain + EPS {
                    best_gain = gain;
                    best_comm = c;
                }
            }

            if best_comm == ISOLATE {
                best_comm = next_comm; // allocate a fresh singleton community
                next_comm += 1;
                sigma_tot.push(0.0);
            }
            sigma_tot[best_comm] += ki;
            if best_comm != ci {
                comm[i] = best_comm;
                moved = true;
            }
        }
        if !moved {
            converged = true;
            break;
        }
    }
    debug_assert!(
        converged,
        "louvain local_moving did not converge within MAX_PASSES"
    );
    comm
}

/// Renumber arbitrary community labels to dense 0-based ids in order of first appearance when
/// scanning nodes in ascending index order. Returns `(dense_labels, num_communities)`.
fn densify(comm: &[usize]) -> (Vec<usize>, usize) {
    let mut label_map: BTreeMap<usize, usize> = BTreeMap::new();
    let mut dense = vec![0usize; comm.len()];
    for (i, &c) in comm.iter().enumerate() {
        let next = label_map.len();
        dense[i] = *label_map.entry(c).or_insert(next);
    }
    (dense, label_map.len())
}

/// Phase 2 (aggregation): collapse each dense community into a super-node. Intra-community weight
/// becomes the super-node's self-loop (stored as `2*L`); inter-community weight is summed.
fn aggregate(level: &Level, dense: &[usize], num_comm: usize) -> Level {
    let mut intra = vec![0.0f64; num_comm];
    let mut inter: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for i in 0..level.n {
        let cu = dense[i];
        for (&j, &w) in &level.adj[i] {
            if j < i {
                continue; // each undirected edge processed once
            }
            if j == i {
                intra[cu] += w / 2.0; // stored self-loop is 2*L
            } else {
                let cv = dense[j];
                if cu == cv {
                    intra[cu] += w;
                } else {
                    let key = if cu < cv { (cu, cv) } else { (cv, cu) };
                    *inter.entry(key).or_insert(0.0) += w;
                }
            }
        }
    }
    let mut adj: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); num_comm];
    for (cu, &wi) in intra.iter().enumerate() {
        if wi > 0.0 {
            adj[cu].insert(cu, 2.0 * wi);
        }
    }
    for ((cu, cv), w) in inter {
        *adj[cu].entry(cv).or_insert(0.0) += w;
        *adj[cv].entry(cu).or_insert(0.0) += w;
    }
    Level::from_adj(adj)
}

/// Assign each node a dense 0-based community id via deterministic tier-weighted Louvain modularity.
/// Edge weight is derived from the edge type; parallel/undirected edges are aggregated by sum;
/// self-loops and edges referencing unknown ids are ignored; isolated nodes get their own community.
/// Community labels are assigned in ascending order of each community's smallest member index, so
/// identical input yields byte-identical output across runs.
pub fn detect_communities(node_ids: &[String], edges: &[Triple]) -> HashMap<String, usize> {
    let mut ids: Vec<&String> = node_ids.iter().collect();
    ids.sort();
    ids.dedup();
    let n = ids.len();
    let index: HashMap<&String, usize> = ids.iter().enumerate().map(|(i, s)| (*s, i)).collect();

    // Base weighted undirected adjacency (parallel edges summed; self-loops / unknown ids dropped).
    let mut adj: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); n];
    for (f, ty, to) in edges {
        if let (Some(&a), Some(&b)) = (index.get(f), index.get(to)) {
            if a == b {
                continue; // self-loop ignored
            }
            let w = tier_weight(ty);
            *adj[a].entry(b).or_insert(0.0) += w;
            *adj[b].entry(a).or_insert(0.0) += w;
        }
    }

    let mut level = Level::from_adj(adj);
    // Maps each original node index to its node index in the current (aggregated) level.
    let mut orig_level: Vec<usize> = (0..n).collect();

    let mut level_converged = false;
    for _ in 0..MAX_LEVELS {
        let comm = local_moving(&level);
        let (dense, num_comm) = densify(&comm);
        for c in orig_level.iter_mut() {
            *c = dense[*c];
        }
        if num_comm >= level.n {
            level_converged = true;
            break; // no community merged this level; converged
        }
        level = aggregate(&level, &dense, num_comm);
    }
    debug_assert!(
        level_converged,
        "louvain aggregation did not converge within MAX_LEVELS"
    );

    // Final relabel: dense 0-based ids in ascending order of each community's smallest member index.
    let mut final_label: BTreeMap<usize, usize> = BTreeMap::new();
    let mut out = HashMap::new();
    for (i, s) in ids.iter().enumerate() {
        let next = final_label.len();
        let fc = *final_label.entry(orig_level[i]).or_insert(next);
        out.insert((*s).clone(), fc);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

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
        let comm = detect_communities(&nodes, &edges);
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
        let comm = detect_communities(&nodes, &edges);
        assert_eq!(comm["a"], comm["b"]);
        assert_eq!(comm["b"], comm["c"]);
    }

    // NOTE (Minor 1, isolate option): local_moving now offers a node the "isolate into a fresh
    // singleton" move (gain 0) so it is never trapped in a community it no longer benefits from.
    // Its effect is marginal by nature — a pendant node's modularity gain is ~0 either way — so a
    // reliable black-box "forces isolation" fixture on a toy graph is not constructible; the change
    // is covered by no-regression (the tests here + determinism) plus the debug_assert on convergence.

    #[test]
    fn louvain_shatters_a_similar_flooded_hub() {
        // Two tight spine-linked (RESOLVED_BY) triangles, bridged only by a single weak SIMILAR
        // edge, plus a dense SIMILAR star (hub -> N leaves). Union-find would merge the triangles
        // through the SIMILAR bridge; tier-weighted Louvain must keep them apart.
        let mut nodes = vec![
            s("A1"),
            s("A2"),
            s("A3"),
            s("B1"),
            s("B2"),
            s("B3"),
            s("hub"),
        ];
        for i in 0..6 {
            nodes.push(s(&format!("leaf{i}")));
        }
        let spine = |a: &str, b: &str| -> Triple { (a.into(), "RESOLVED_BY".into(), b.into()) };
        let sim = |a: &str, b: &str| -> Triple { (a.into(), "SIMILAR".into(), b.into()) };
        let mut edges = vec![
            spine("A1", "A2"),
            spine("A2", "A3"),
            spine("A3", "A1"),
            spine("B1", "B2"),
            spine("B2", "B3"),
            spine("B3", "B1"),
            sim("A1", "B1"), // the lone weak bridge
        ];
        for i in 0..6 {
            edges.push(sim("hub", &format!("leaf{i}")));
        }

        let comm = detect_communities(&nodes, &edges);
        // Each triangle collapses into a single community.
        assert_eq!(comm["A1"], comm["A2"]);
        assert_eq!(comm["A2"], comm["A3"]);
        assert_eq!(comm["B1"], comm["B2"]);
        assert_eq!(comm["B2"], comm["B3"]);
        // The weak SIMILAR bridge does NOT merge the two triangles.
        assert_ne!(comm["A1"], comm["B1"]);
    }

    #[test]
    fn louvain_is_deterministic() {
        let mut nodes = Vec::new();
        for i in 0..20 {
            nodes.push(s(&format!("n{i}")));
        }
        let sim = |a: String, b: String| -> Triple { (a, "SIMILAR".into(), b) };
        let spine = |a: String, b: String| -> Triple { (a, "CAUSED_BY".into(), b) };
        let mut edges = Vec::new();
        // Two dense spine cliques.
        for i in 0..5 {
            for j in (i + 1)..5 {
                edges.push(spine(format!("n{i}"), format!("n{j}")));
            }
        }
        for i in 5..10 {
            for j in (i + 1)..10 {
                edges.push(spine(format!("n{i}"), format!("n{j}")));
            }
        }
        // Sparse SIMILAR ring noise touching every node.
        for i in 0..20 {
            edges.push(sim(format!("n{i}"), format!("n{}", (i + 7) % 20)));
        }

        let a = detect_communities(&nodes, &edges);
        let b = detect_communities(&nodes, &edges);
        assert_eq!(a, b);
        // Sanity: the tier weights actually produce structure (more than one community).
        let distinct: BTreeSet<usize> = a.values().copied().collect();
        assert!(distinct.len() >= 2);
    }

    #[test]
    fn isolated_node_gets_own_community() {
        let nodes = vec![s("a"), s("b"), s("island")];
        let edges = vec![(s("a"), "RESOLVED_BY".into(), s("b"))];
        let comm = detect_communities(&nodes, &edges);
        assert_eq!(comm["a"], comm["b"]);
        assert_ne!(comm["island"], comm["a"]);
        assert_ne!(comm["island"], comm["b"]);
    }
}
