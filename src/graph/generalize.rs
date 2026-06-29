//! Deterministic, embeddings-free graph generalization.
//!
//! Pure functions over graph ids/edges (no DB, no model, no embeddings), each in its own module
//! under this directory. Wired into `kb graph generalize`, MCP `graph_generalize`, and the editor
//! maintenance loop. Derived edges are stamped `origin = "auto-generalized"`.
//!
//! Techniques: stemmed label Jaccard + shared evidence (SIMILAR), link prediction, community
//! detection, PageRank centrality, ontology-defined transitive closure, optional near-dup merge.
//! See [docs/architecture.md](../../docs/architecture.md) § Derived layer.

pub mod apply;
pub mod centrality;
pub mod closure;
pub mod community;
pub mod hygiene;
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
