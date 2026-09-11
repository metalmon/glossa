//! In-process NLI scorer over an ONNX 3-way NLI model (ort, CPU). Standalone crate — it does NOT
//! depend on `glossa`; `glossa` owns the `NliScorer` trait and implements it for `InProcessNli`
//! (see glossa's `src/gate/nli_engine.rs`), mirroring the glossa-constraint adapter pattern.

// Exactly one inference engine per build.
#[cfg(all(feature = "nli-ort", feature = "nli-burn-wgpu"))]
compile_error!("enable exactly ONE of `nli-ort` / `nli-burn-wgpu`, not both");

use std::path::Path;

/// An in-process NLI scorer. Constructed from a local `model_dir` (`model.onnx` + `tokenizer.json`)
/// and the entailment class index. Task 2 fills in the real ort session + tokenizer + windowing;
/// this stub keeps the crate + glossa's `nli` feature compiling.
pub struct InProcessNli {
    _model_dir: std::path::PathBuf,
    _entail_index: usize,
}

impl InProcessNli {
    /// Load the model + tokenizer from `model_dir`. `entail_index` is the softmax index of the
    /// entailment class (from the model's `id2label`). Task 2 makes this real.
    pub fn load(model_dir: &Path, entail_index: usize) -> anyhow::Result<Self> {
        Ok(Self {
            _model_dir: model_dir.to_path_buf(),
            _entail_index: entail_index,
        })
    }

    /// P(entail) of each hypothesis against `premise`. Stub returns an error until Task 2 wires ort.
    pub fn entail(&self, _premise: &str, _hypotheses: &[&str]) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("glossa-nli InProcessNli::entail not implemented yet (Task 2)")
    }
}
