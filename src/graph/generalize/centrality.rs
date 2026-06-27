//! #7 — Node centrality: degree and PageRank. Hubs (e.g. a Resolution that fixes many Symptoms)
//! score high and can be surfaced first in retrieval.

use super::Triple;
use std::collections::HashMap;

/// Total (in + out) degree per node; self-loops and unknown ids ignored.
pub fn degree(node_ids: &[String], edges: &[Triple]) -> HashMap<String, usize> {
    let mut deg: HashMap<String, usize> = node_ids.iter().map(|s| (s.clone(), 0)).collect();
    for (f, _t, to) in edges {
        if f == to {
            continue;
        }
        if let Some(d) = deg.get_mut(f) {
            *d += 1;
        }
        if let Some(d) = deg.get_mut(to) {
            *d += 1;
        }
    }
    deg
}

/// PageRank over the directed graph (`damping` ~0.85, `iters` ~20-30). Dangling mass is redistributed
/// uniformly; scores sum ≈ 1. Multi-edges between the same pair are collapsed.
pub fn pagerank(
    node_ids: &[String],
    edges: &[Triple],
    damping: f64,
    iters: usize,
) -> HashMap<String, f64> {
    let n = node_ids.len();
    if n == 0 {
        return HashMap::new();
    }
    let idx: HashMap<&String, usize> = node_ids.iter().enumerate().map(|(i, s)| (s, i)).collect();
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (f, _t, to) in edges {
        if f == to {
            continue;
        }
        if let (Some(&a), Some(&b)) = (idx.get(f), idx.get(to)) {
            out[a].push(b);
        }
    }
    for o in out.iter_mut() {
        o.sort_unstable();
        o.dedup();
    }
    let mut rank = vec![1.0 / n as f64; n];
    let base = (1.0 - damping) / n as f64;
    for _ in 0..iters {
        let mut next = vec![base; n];
        let mut dangling = 0.0;
        for (a, outs) in out.iter().enumerate() {
            if outs.is_empty() {
                dangling += rank[a];
            } else {
                let share = damping * rank[a] / outs.len() as f64;
                for &b in outs {
                    next[b] += share;
                }
            }
        }
        let spread = damping * dangling / n as f64;
        for v in next.iter_mut() {
            *v += spread;
        }
        rank = next;
    }
    node_ids.iter().enumerate().map(|(i, s)| (s.clone(), rank[i])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn s(x: &str) -> String {
        x.into()
    }
    fn e(a: &str, b: &str) -> Triple {
        (a.into(), "E".into(), b.into())
    }

    #[test]
    fn degree_counts_in_and_out() {
        let nodes = vec![s("h"), s("a"), s("b")];
        let edges = vec![e("a", "h"), e("b", "h")];
        let d = degree(&nodes, &edges);
        assert_eq!(d["h"], 2);
        assert_eq!(d["a"], 1);
        assert_eq!(d["b"], 1);
    }

    #[test]
    fn pagerank_hub_scores_highest() {
        let nodes = vec![s("h"), s("a"), s("b"), s("c")];
        let edges = vec![e("a", "h"), e("b", "h"), e("c", "h")];
        let pr = pagerank(&nodes, &edges, 0.85, 50);
        assert!(pr["h"] > pr["a"], "hub {} should exceed leaf {}", pr["h"], pr["a"]);
        let sum: f64 = pr.values().sum();
        assert!((sum - 1.0).abs() < 1e-6, "ranks should sum to ~1, got {sum}");
    }
}
