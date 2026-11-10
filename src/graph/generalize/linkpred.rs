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
/// The parameter is required rather than optional on purpose. This loop is **O(P^2)** in the
/// pairable set, and it used to run over every node that had an edge — which on a document corpus
/// means one `Section` per chunk. Measured: 11.8 s at 10k nodes, 67 s at 19k and 289 s at 37k, to
/// emit 4 to 64 `SIMILAR` edges between sections, the very edges retrieval then has to de-weight.
/// A caller that passes everything is asking for that again, and now has to say so.
pub fn jaccard_pairs(
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
            let na = &adj[a];
            let nb = &adj[b];
            if na.contains(b.as_str()) {
                continue; // already linked
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
