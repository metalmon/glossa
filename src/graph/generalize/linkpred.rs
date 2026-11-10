//! #4 — Structural link prediction: nodes sharing many graph neighbours are likely related
//! (e.g. two Symptoms sharing a Resolution). Jaccard and Adamic-Adar over the undirected neighbour
//! sets. Pure topology, no text. Already-adjacent pairs are excluded.

use super::{undirected_adjacency, Triple};

/// Candidate similar pairs `(id_a, id_b, score)` (id_a < id_b) by Jaccard of neighbour sets, with
/// `score >= threshold` (0..1). Sorted by score desc then id.
///
/// `pairable` is the set of node ids allowed to FORM a pair — in practice the reasoning nodes, with
/// the ontology's structural types (`Document`, `Section`, …) left out. Neighbourhoods are still
/// built from every edge, because that is where the signal lives: two reasoning nodes look alike
/// precisely because they point at the same structural nodes.
///
/// The parameter is required rather than optional on purpose: pairing is the expensive half of this
/// function, and a caller that wants every node in it has to say so. It used to receive every node
/// with an edge — one `Section` per chunk on a document corpus — which cost 11.8 s at 10k nodes and
/// 289 s at 37k to emit 4 to 64 section-to-section edges, the very ones retrieval de-weights.
///
/// **Candidates come from shared neighbours, not from every pair.** A pair with no common neighbour
/// scores zero, and the naive loop computed the intersection of every pair just to discard those —
/// so this walks each node's neighbourhood instead and takes its pairable members as candidates.
/// Exactly the same pairs, with the ones that cannot qualify never visited. The cost is
/// `sum over nodes of (pairable degree)^2` rather than `(pairable nodes)^2`: on a sparse graph far
/// less, and on a graph with one hub that every reasoning node points at, no better — that case is
/// a measurement away, not a guess, and `similar_threshold` does not bound it.
pub fn jaccard_pairs(
    edges: &[Triple],
    threshold: f64,
    pairable: &std::collections::HashSet<String>,
) -> Vec<(String, String, f64)> {
    let adj = undirected_adjacency(edges);
    let mut candidates: std::collections::HashSet<(&str, &str)> = std::collections::HashSet::new();
    for members in adj.values() {
        let shared: Vec<&str> = members
            .iter()
            .map(String::as_str)
            .filter(|id| pairable.contains(*id))
            .collect();
        // `members` is a BTreeSet, so `shared` is already in id order and (a, b) comes out with
        // a < b — the orientation the naive loop produced from its sorted node list.
        for i in 0..shared.len() {
            for j in (i + 1)..shared.len() {
                candidates.insert((shared[i], shared[j]));
            }
        }
    }
    let mut out = Vec::new();
    for (a, b) in candidates {
        let (Some(na), Some(nb)) = (adj.get(a), adj.get(b)) else {
            continue;
        };
        if na.contains(b) {
            continue; // already linked
        }
        // At least 1 by construction: the pair was seen inside some node's neighbourhood.
        let inter = na.intersection(nb).count();
        let union = na.union(nb).count();
        let score = inter as f64 / union as f64;
        if score >= threshold {
            out.push((a.to_string(), b.to_string(), score));
        }
    }
    out.sort_by(|x, y| y.2.total_cmp(&x.2).then(x.0.cmp(&y.0)).then(x.1.cmp(&y.1)));
    out
}

/// Adamic-Adar variant: each shared neighbour contributes `1/ln(deg)`, so rare shared neighbours
/// weigh more. Returns pairs with `score >= threshold`, sorted by score desc.
pub fn adamic_adar_pairs(edges: &[Triple], threshold: f64) -> Vec<(String, String, f64)> {
    let adj = undirected_adjacency(edges);
    let mut nodes: Vec<&String> = adj.keys().collect();
    nodes.sort();
    let mut out = Vec::new();
    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            let (a, b) = (nodes[i], nodes[j]);
            let na = &adj[a];
            let nb = &adj[b];
            if na.contains(b.as_str()) {
                continue;
            }
            let mut score = 0.0;
            for common in na.intersection(nb) {
                let deg = adj.get(common).map(|s| s.len()).unwrap_or(0);
                if deg > 1 {
                    score += 1.0 / (deg as f64).ln();
                }
            }
            if score >= threshold {
                out.push((a.to_string(), b.to_string(), score));
            }
        }
    }
    out.sort_by(|x, y| y.2.total_cmp(&x.2).then(x.0.cmp(&y.0)).then(x.1.cmp(&y.1)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn e(a: &str, b: &str) -> Triple {
        (a.into(), "E".into(), b.into())
    }
    fn ids(v: &[&str]) -> std::collections::HashSet<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn shared_neighbours_yield_high_jaccard() {
        // a and b both connect to h1,h2; not linked to each other → Jaccard 1.0
        let edges = vec![e("a", "h1"), e("a", "h2"), e("b", "h1"), e("b", "h2")];
        let pairs = jaccard_pairs(&edges, 0.5, &ids(&["a", "b"]));
        let ab = pairs.iter().find(|(a, b, _)| a == "a" && b == "b");
        assert!(ab.is_some(), "expected (a,b) pair, got {pairs:?}");
        assert!((ab.unwrap().2 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn already_linked_pairs_excluded() {
        let edges = vec![e("a", "b"), e("a", "h1"), e("b", "h1")];
        // a,b are directly linked → must not appear despite the shared neighbour h1
        let pairs = jaccard_pairs(&edges, 0.0, &ids(&["a", "b"]));
        assert!(!pairs.iter().any(|(a, b, _)| a == "a" && b == "b"));
    }

    #[test]
    fn a_node_outside_the_pairable_set_never_forms_a_pair() {
        // h1 and h2 share both a and b as neighbours, so they would pair with each other — but a
        // hub that is not pairable (a Section, say) must not. Its neighbourhood still counts: that
        // is what makes a and b look alike.
        let edges = vec![e("a", "h1"), e("a", "h2"), e("b", "h1"), e("b", "h2")];
        let pairs = jaccard_pairs(&edges, 0.5, &ids(&["a", "b"]));
        assert!(pairs.iter().any(|(x, y, _)| x == "a" && y == "b"));
        assert!(
            !pairs.iter().any(|(x, y, _)| x == "h1" || y == "h1"),
            "a non-pairable hub paired anyway: {pairs:?}"
        );
    }

    /// The loop this function replaced, kept as the thing the fast path is checked against. An
    /// optimisation that changes which pairs come out is not an optimisation.
    fn naive_jaccard_pairs(
        edges: &[Triple],
        threshold: f64,
        pairable: &std::collections::HashSet<String>,
    ) -> Vec<(String, String, f64)> {
        let adj = undirected_adjacency(edges);
        let mut nodes: Vec<&String> = adj.keys().filter(|id| pairable.contains(*id)).collect();
        nodes.sort();
        let mut out = Vec::new();
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let (a, b) = (nodes[i], nodes[j]);
                let (na, nb) = (&adj[a], &adj[b]);
                if na.contains(b.as_str()) {
                    continue;
                }
                let inter = na.intersection(nb).count();
                if inter == 0 {
                    continue;
                }
                let union = na.union(nb).count();
                let score = inter as f64 / union as f64;
                if score >= threshold {
                    out.push((a.to_string(), b.to_string(), score));
                }
            }
        }
        out.sort_by(|x, y| y.2.total_cmp(&x.2).then(x.0.cmp(&y.0)).then(x.1.cmp(&y.1)));
        out
    }

    #[test]
    fn candidate_generation_agrees_with_the_naive_loop() {
        // A deterministic pseudo-random bipartite-ish graph: reasoning nodes pointing at shared
        // hubs, plus some direct links between reasoning nodes (which must stay excluded), at
        // several thresholds. If the two ever disagree, the fast path is wrong, not faster.
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut edges: Vec<Triple> = Vec::new();
        for r in 0..40usize {
            for _ in 0..(1 + (next() % 5)) {
                let h = next() % 12;
                edges.push((format!("r{r}"), "MENTIONS".into(), format!("h{h}")));
            }
            if next() % 4 == 0 {
                let other = next() as usize % 40;
                edges.push((format!("r{r}"), "RELATED".into(), format!("r{other}")));
            }
        }
        let pairable: std::collections::HashSet<String> =
            (0..40).map(|r| format!("r{r}")).collect();
        for threshold in [0.0, 0.2, 0.5, 0.8, 1.0] {
            let fast = jaccard_pairs(&edges, threshold, &pairable);
            let naive = naive_jaccard_pairs(&edges, threshold, &pairable);
            assert_eq!(
                fast.len(),
                naive.len(),
                "pair count differs at threshold {threshold}"
            );
            for (f, n) in fast.iter().zip(naive.iter()) {
                assert_eq!((&f.0, &f.1), (&n.0, &n.1), "at threshold {threshold}");
                assert!((f.2 - n.2).abs() < 1e-12, "score differs: {f:?} vs {n:?}");
            }
        }
        // The fixture has to be non-trivial or the agreement proves nothing.
        assert!(
            !jaccard_pairs(&edges, 0.2, &pairable).is_empty(),
            "fixture produced no pairs at all"
        );
    }

    #[test]
    fn a_pair_sharing_no_neighbour_is_never_visited_and_never_emitted() {
        // Two nodes with disjoint neighbourhoods: Jaccard 0, and the candidate walk never sees
        // them. Asserted at threshold 0.0, where "0 >= threshold" would otherwise let them in.
        let edges = vec![e("a", "h1"), e("b", "h2")];
        let pairs = jaccard_pairs(&edges, 0.0, &ids(&["a", "b"]));
        assert!(pairs.is_empty(), "{pairs:?}");
    }

    #[test]
    fn an_empty_pairable_set_does_no_work_and_finds_nothing() {
        let edges = vec![e("a", "h1"), e("a", "h2"), e("b", "h1"), e("b", "h2")];
        assert!(jaccard_pairs(&edges, 0.0, &ids(&[])).is_empty());
    }

    #[test]
    fn adamic_adar_scores_shared_neighbours() {
        let edges = vec![e("a", "h1"), e("a", "h2"), e("b", "h1"), e("b", "h2")];
        let pairs = adamic_adar_pairs(&edges, 0.1);
        assert!(pairs
            .iter()
            .any(|(a, b, s)| a == "a" && b == "b" && *s > 0.0));
    }
}
