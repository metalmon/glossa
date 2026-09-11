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
//! - the `DocIndex` tantivy reader picks up the latest commit — but its background auto-reload
//!   (`OnCommitWithDelay`) fires on tantivy's own filesystem watcher, which is unreliable on the
//!   network folders this daemon targets. So the search index lives behind an [`ArcSwap`] here and
//!   the server SWAPS in a FRESHLY-OPENED `DocIndex` on a freshen that indexed a change (see
//!   [`GraphHandle::refresh_idx`]) rather than trusting that watcher. Reads take an O(1) atomic
//!   [`GraphHandle::idx`] load; a swap never disrupts an in-flight query (RCU: the old
//!   `Arc<DocIndex>` stays alive until its last reader drops).
//!
//! `csr` is a convenience snapshot pre-warmed at build time; `graph.csr()` remains the authoritative
//! (self-invalidating) accessor that `compose_ppr` uses on the hot path.

use crate::graph::csr::CsrTransition;
use crate::graph::store::GraphStore;
use crate::index::store::DocIndex;
use arc_swap::ArcSwap;
use std::path::Path;
use std::sync::Arc;

/// One consistent, shareable snapshot of a corpus's retrieval state. See the module docs for the
/// sharing and freshness contract.
pub struct GraphHandle {
    pub graph: GraphStore,
    /// The full-text search index, behind an `ArcSwap` so a freshen can install a freshly-opened
    /// `DocIndex` (reflecting newly committed segments) without rebuilding the whole handle and
    /// without disrupting in-flight queries. Read it via [`GraphHandle::idx`] (an O(1) load); never
    /// depend on a single long-lived reader auto-reloading — see the module docs.
    idx: ArcSwap<DocIndex>,
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
        Ok(GraphHandle {
            graph,
            idx: ArcSwap::from_pointee(idx),
            csr,
        })
    }

    /// The current search-index snapshot. O(1) atomic load; the returned `Arc` keeps THIS snapshot
    /// alive for the whole query even if a concurrent [`refresh_idx`](Self::refresh_idx) swaps in a
    /// newer one. Every read path (search/read/glossary/…) goes through here.
    pub fn idx(&self) -> Arc<DocIndex> {
        self.idx.load_full()
    }

    /// Open a FRESH `DocIndex` on the same roots/state-base (so it reflects segments committed by the
    /// freshen's separate `Index`) and swap it in. Replaces the old, fragile "nudge the long-lived
    /// reader via `reload()`" path: reopening is watcher-independent and goes through the transient-FS
    /// retry in `open_or_create_at`, and the `Result` is returned so a failed refresh is observable
    /// instead of silently swallowed. Cheap enough to run once per min-rescan window on a change.
    pub fn refresh_idx(&self) -> anyhow::Result<()> {
        let cur = self.idx.load();
        let fresh = DocIndex::open_or_create_at(&cur.roots, &cur.state_base)?;
        self.idx.store(Arc::new(fresh));
        Ok(())
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
    fn refresh_idx_swaps_in_a_fresh_index_that_sees_a_new_file() {
        use crate::model::Chunk;
        use std::path::PathBuf;
        let dir = tempfile::tempdir().unwrap();
        let h = GraphHandle::open(dir.path()).unwrap();
        let mk = |path: &str, text: &str| Chunk {
            doc_path: PathBuf::from(path),
            location: "S".into(),
            file_type: "md".into(),
            text: text.into(),
        };
        // Seed the corpus through the handle's own index and confirm it serves it.
        h.idx()
            .write_chunks(&[mk("a.md", "alpha content")])
            .unwrap();
        h.refresh_idx().unwrap();
        assert!(
            !h.idx().search("alpha", 10).unwrap().is_empty(),
            "handle serves the seeded file"
        );
        let before = h.idx(); // an in-flight reader holds this snapshot

        // A new file is committed through a SEPARATE index instance on the same state (as freshen does).
        let external = DocIndex::open_or_create(dir.path()).unwrap();
        external
            .write_chunks(&[
                mk("a.md", "alpha content"),
                mk("b.md", "beta latecomer content"),
            ])
            .unwrap();

        // refresh_idx must swap in a fresh reader that sees it — and it RETURNS the result (no
        // longer swallowed).
        h.refresh_idx().unwrap();
        assert!(
            h.idx()
                .search("latecomer", 10)
                .unwrap()
                .iter()
                .any(|hit| hit.path.contains("b.md")),
            "after refresh_idx the handle serves the externally committed file"
        );
        // The pre-swap snapshot the in-flight reader holds is still valid (RCU).
        assert!(!before.search("alpha", 10).unwrap().is_empty());
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
