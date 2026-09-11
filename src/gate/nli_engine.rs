//! `#[cfg(feature = "nli")]` adapter: implement glossa's `NliScorer` trait for the standalone
//! `glossa_nli::InProcessNli`. Mirrors `src/constraint_adapter.rs`. Keeping the trait impl HERE (not
//! in glossa-nli) is what avoids the glossa <-> glossa-nli dependency cycle.
use crate::gate::nli::NliScorer;

impl NliScorer for glossa_nli::InProcessNli {
    fn entail(&self, premise: &str, hypotheses: &[&str]) -> anyhow::Result<Vec<f32>> {
        glossa_nli::InProcessNli::entail(self, premise, hypotheses)
    }
}
