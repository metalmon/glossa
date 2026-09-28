//! Adapter: the glossa-side `Reranker` trait implemented for glossa-nli's in-process ONNX
//! cross-encoder engine. Mirrors `src/gate/nli_engine.rs` (keeping the trait impl here, not in
//! glossa-nli, is what avoids the glossa <-> glossa-nli dependency cycle — orphan rule: the trait
//! is glossa's, the type is glossa-nli's). Compiled only when an ORT engine feature is on and the
//! burn engine (which has no reranker type) is not.
#![cfg(all(
    any(feature = "nli", feature = "nli-dynamic"),
    not(feature = "nli-burn")
))]

use crate::retrieve::rerank::Reranker;

impl Reranker for glossa_nli::InProcessReranker {
    fn rerank(&self, query: &str, passages: &[&str]) -> anyhow::Result<Vec<f32>> {
        glossa_nli::InProcessReranker::rerank(self, query, passages)
    }
}
