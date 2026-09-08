//! A shared, swappable snapshot of the corpus's retrieval state.
//!
//! `GraphHandle` bundles the three things a retrieval tool call reads — the graph store, the document
//! index, and the (pre-warmed, out-of-core) CSR transition — into ONE snapshot. `GlossaServer` holds
//! it behind an `ArcSwapOption` so every concurrent request loads the SAME snapshot lock-free (no
//! per-call re-open), and a freshness rebuild swaps in a new snapshot without blocking in-flight
//! readers (RCU: a reader keeps its `Arc` until it is done).
//!
//! Freshness: the bundled components are individually self-freshening, so a held handle serves
//! current data with the SAME semantics as re-opening per call, minus the re-open cost —
//! - the `GraphStore`'s SQLite reads are always live (WAL, and any writer commits on its own conn);
//! - its PPR transition and CSR caches self-invalidate on the DB file signature / content signature;
//! - the `DocIndex` tantivy reader auto-reloads on commit (its default `OnCommitWithDelay` policy).
//!
//! `csr` is a convenience snapshot pre-warmed at build time; `graph.csr()` remains the authoritative
//! (self-invalidating) accessor that `compose_ppr` uses on the hot path.

use crate::graph::csr::CsrTransition;
use crate::graph::store::GraphStore;
use crate::index::store::DocIndex;
use std::path::Path;
use std::sync::Arc;

/// One consistent, shareable snapshot of a corpus's retrieval state. See the module docs for the
/// sharing and freshness contract.
pub struct GraphHandle {
    pub graph: GraphStore,
    pub idx: DocIndex,
    /// The out-of-core CSR transition, pre-warmed at build time so the first request doesn't pay the
    /// build. Authoritative (self-invalidating) accessor is `graph.csr()`.
    pub csr: Arc<CsrTransition>,
}

impl GraphHandle {
    /// Open all components across `roots` (corpus content) with state (index + graph store) rooted
    /// at `state_base`, and pre-warm the CSR (mmaps it, building once on a miss). Called once at
    /// startup (lazily, on first use) and again on a freshness swap.
    pub fn open_at(roots: &[crate::root::Root], state_base: &Path) -> anyhow::Result<GraphHandle> {
        let idx = DocIndex::open_or_create_at(roots, state_base)?;
        let graph = GraphStore::open(state_base)?;
        let csr = graph.csr()?;
        Ok(GraphHandle { graph, idx, csr })
    }

    /// Back-compat: a single positional root that is also the state base (corpus == state dir).
    pub fn open(root: &Path) -> anyhow::Result<GraphHandle> {
        Self::open_at(
            &[crate::root::Root {
                label: String::new(),
                path: root.to_path_buf(),
            }],
            root,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_open_at_uses_state_base_for_graph_and_index() {
        use crate::root::Root;
        let corpus = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(corpus.path().join("a.md"), "content").unwrap();
        let roots = [Root {
            label: String::new(),
            path: corpus.path().into(),
        }];
        let h = GraphHandle::open_at(&roots, state.path()).unwrap();
        assert!(state.path().join(".glossa").join("graph.sqlite").exists());
        assert!(!corpus.path().join(".glossa").exists());
        assert!(h.graph.all_nodes().is_ok());
    }

    #[test]
    fn arcswap_handle_serves_reads_and_swaps() {
        let dir = tempfile::tempdir().unwrap();
        // A minimal corpus+graph fixture: opening creates the .glossa store + index.
        let cell = arc_swap::ArcSwap::from_pointee(GraphHandle::open(dir.path()).unwrap());
        let before = cell.load_full(); // a reader holds this Arc
                                       // Simulate a corpus change → rebuild + swap.
        cell.store(std::sync::Arc::new(GraphHandle::open(dir.path()).unwrap()));
        let after = cell.load_full();
        assert!(
            !std::sync::Arc::ptr_eq(&before, &after),
            "swap installs a new handle"
        );
        // The in-flight `before` Arc is still valid (RCU): its graph still answers.
        assert!(before.graph.all_nodes().is_ok());
    }
}
