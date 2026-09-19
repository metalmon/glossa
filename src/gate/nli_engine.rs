//! Adapters implementing glossa's `NliScorer` trait for glossa-nli's engine types. Keeping the
//! trait impls HERE (not in glossa-nli) is what avoids the glossa <-> glossa-nli dependency cycle,
//! and it's the only crate that can (orphan rule: the trait is glossa's, the types are glossa-nli's).
//! Exactly one engine is compiled per build (`nli`/`nli-dynamic` = ORT, XOR `nli-burn` = burn/wgpu).
use crate::gate::nli::NliScorer;

#[cfg(any(feature = "nli", feature = "nli-dynamic"))]
impl NliScorer for glossa_nli::InProcessNli {
    fn entail(&self, premise: &str, hypotheses: &[&str]) -> anyhow::Result<Vec<f32>> {
        glossa_nli::InProcessNli::entail(self, premise, hypotheses)
    }
}

#[cfg(feature = "nli-burn")]
impl NliScorer for glossa_nli::InProcessBurnNli {
    fn entail(&self, premise: &str, hypotheses: &[&str]) -> anyhow::Result<Vec<f32>> {
        glossa_nli::InProcessBurnNli::entail(self, premise, hypotheses)
    }
}
