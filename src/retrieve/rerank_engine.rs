//! Adapters: the glossa-side `Reranker` trait implemented for glossa-nli's in-process cross-encoder
//! engines (ORT and burn/wgpu). Mirrors `src/gate/nli_engine.rs` (keeping the trait impls here, not
//! in glossa-nli, is what avoids the glossa <-> glossa-nli dependency cycle — orphan rule: the trait
//! is glossa's, the types are glossa-nli's). Compiled whenever ANY reranker engine feature is on —
//! see `src/retrieve/mod.rs`'s module cfg.
#![cfg(any(feature = "nli", feature = "nli-dynamic", feature = "nli-burn"))]

use crate::retrieve::rerank::Reranker;

#[cfg(all(
    any(feature = "nli", feature = "nli-dynamic"),
    not(feature = "nli-burn")
))]
impl Reranker for glossa_nli::InProcessReranker {
    fn rerank(&self, query: &str, passages: &[&str]) -> anyhow::Result<Vec<f32>> {
        glossa_nli::InProcessReranker::rerank(self, query, passages)
    }
}

#[cfg(feature = "nli-burn")]
impl Reranker for glossa_nli::InProcessBurnReranker {
    fn rerank(&self, query: &str, passages: &[&str]) -> anyhow::Result<Vec<f32>> {
        glossa_nli::InProcessBurnReranker::rerank(self, query, passages)
    }
}
