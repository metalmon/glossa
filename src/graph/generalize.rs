//! Deterministic, embeddings-free graph generalization.
//!
//! Implements the techniques from
//! `docs/superpowers/plans/2026-06-27-graph-generalization-no-embeddings.md`, each as a PURE
//! function over the graph's ids/edges (no DB, no model, no embeddings) in its own file, so it can
//! be unit-tested in isolation and later wired into a single `kb graph generalize` post-enrichment
//! pass. NONE of this is hooked into the live pipeline yet — that is the integration step (map
//! `store::Edge` ↔ these triples, stamp derived edges `origin = "auto-generalized"`).
//!
//! Coverage vs the plan: #2 BM25-over-labels (approximated embeddings-free by stemmed-token Jaccard
//! in `similarity::label_jaccard`), #3 shared-evidence (`similarity::shared_evidence`), #4 structural
//! link-prediction (`linkpred`), #6 communities (`community`), #7 centrality (`centrality`),
//! #8 transitive closure (`closure`), plus near-dup MERGE grouping (`merge`). #5 synonym dictionary
//! (needs curation) and a true tantivy-BM25 label index are left for the integration step.

pub mod apply;
pub mod centrality;
pub mod closure;
pub mod community;
pub mod linkpred;
pub mod merge;
pub mod similarity;

/// A directed edge as `(from_id, edge_type, to_id)` — the decoupled shape these passes operate on,
/// independent of `store::Edge`'s provenance fields.
pub type Triple = (String, String, String);

/// Undirected adjacency (id → set of neighbour ids) from directed edges; self-loops dropped.
/// Shared by the topology passes (community detection, link prediction).
pub(crate) fn undirected_adjacency(
    edges: &[Triple],
) -> std::collections::HashMap<String, std::collections::BTreeSet<String>> {
    use std::collections::{BTreeSet, HashMap};
    let mut adj: HashMap<String, BTreeSet<String>> = HashMap::new();
    for (f, _t, to) in edges {
        if f == to {
            continue;
        }
        adj.entry(f.clone()).or_default().insert(to.clone());
        adj.entry(to.clone()).or_default().insert(f.clone());
    }
    adj
}
