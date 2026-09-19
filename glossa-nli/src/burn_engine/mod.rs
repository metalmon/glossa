//! Pure-Rust burn/wgpu NLI engine (Plan 4): a hand-written RuBERT-base forward on the GPU via
//! Vulkan/SPIR-V (or an NdArray CPU backend for tests). Implements the shared
//! [`crate::harness::RawForward`] so it reuses the exact windowing/batching/pooling of the ORT
//! engine — cross-engine parity by construction.

mod model;

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use tokenizers::Tokenizer;

use crate::harness::{self, RawForward, DEFAULT_MAX_SEQ_LEN};
use model::{BertNliConfig, BertNliModel};

// Exactly one burn backend per build: the GPU (Vulkan/SPIR-V) production backend, or the NdArray
// CPU backend used by CI parity tests (no GPU runner).
#[cfg(not(any(feature = "nli-burn-vulkan", feature = "nli-burn-cpu")))]
compile_error!(
    "engine `nli-burn-wgpu` needs a backend: enable `nli-burn-vulkan` (GPU, production) or \
     `nli-burn-cpu` (NdArray, tests)"
);

/// The concrete burn backend for this build. Vulkan wins if both are (accidentally) enabled.
#[cfg(feature = "nli-burn-vulkan")]
pub type BurnBackend = burn::backend::Vulkan;
#[cfg(all(feature = "nli-burn-cpu", not(feature = "nli-burn-vulkan")))]
pub type BurnBackend = burn::backend::NdArray;

type Dev = burn::tensor::Device<BurnBackend>;

/// In-process NLI scorer backed by the hand-written burn BERT-base model. Holds the loaded model,
/// its device, the tokenizer, and the shared windowing/batching config.
pub struct InProcessBurnNli {
    model: BertNliModel<BurnBackend>,
    device: Dev,
    tokenizer: Tokenizer,
    max_seq_len: usize,
    batch_budget_tokens: usize,
    entail_index: usize,
}

/// Resolve the burn weights file inside `model_dir`: `model.safetensors` if present, else the lone
/// `*.safetensors`, else `Err`. Weights are the RuBERT NLI parameters in PyTorch orientation
/// ([out,in] Linear weights, HF names); the `PyTorchToBurnAdapter` transposes them for burn.
fn resolve_weights_file(model_dir: &Path) -> Result<PathBuf> {
    let canonical = model_dir.join("model.safetensors");
    if canonical.is_file() {
        return Ok(canonical);
    }
    let mut candidates = Vec::new();
    for entry in std::fs::read_dir(model_dir)
        .map_err(|e| anyhow!("reading model dir ({}): {e}", model_dir.display()))?
    {
        let path = entry.map_err(|e| anyhow!("reading model dir entry: {e}"))?.path();
        let is_st = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
        if is_st && path.is_file() {
            candidates.push(path);
        }
    }
    match candidates.len() {
        0 => anyhow::bail!("no .safetensors weights file in {}", model_dir.display()),
        1 => Ok(candidates.remove(0)),
        _ => {
            candidates.sort();
            anyhow::bail!(
                "multiple .safetensors candidates, none named model.safetensors in {} — rename \
                 the one you want to model.safetensors",
                model_dir.display()
            )
        }
    }
}

impl InProcessBurnNli {
    /// Load the tokenizer + weights from `model_dir` (`tokenizer.json` + a `.safetensors` weights
    /// file). `entail_index` is the softmax index of the entailment class. Fail-open: returns `Err`
    /// on any missing/garbage file or incomplete weight set; never panics.
    pub fn load(model_dir: &Path, entail_index: usize) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("tokenizer load ({}): {e}", tokenizer_path.display()))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow!("tokenizer truncation config: {e}"))?;
        tokenizer.with_padding(None);

        let device = Dev::default();
        let mut m = BertNliConfig::new().init::<BurnBackend>(&device);

        let weights = resolve_weights_file(model_dir)?;
        let mut store = SafetensorsStore::from_file(
            weights.to_str().ok_or_else(|| anyhow!("non-utf8 weights path"))?,
        )
        .with_from_adapter(PyTorchToBurnAdapter)
        .with_key_remapping(r"\.self\.", ".self_attention.")
        .with_key_remapping("LayerNorm", "layer_norm")
        .allow_partial(true);
        let result = m
            .load_from(&mut store)
            .map_err(|e| anyhow!("burn weight load ({}): {e}", weights.display()))?;
        if !result.missing.is_empty() {
            anyhow::bail!(
                "burn weight load incomplete: {} tensor(s) missing from {} (first: {:?})",
                result.missing.len(),
                weights.display(),
                result.missing.iter().take(3).collect::<Vec<_>>()
            );
        }

        Ok(Self {
            model: m,
            device,
            tokenizer,
            max_seq_len: DEFAULT_MAX_SEQ_LEN,
            batch_budget_tokens: harness::parse_batch_budget_tokens(),
            entail_index,
        })
    }

    /// P(entail) of each hypothesis against `premise`. Delegates windowing/batching/pooling to the
    /// shared [`harness::entail`]; this engine only supplies the raw forward (`RawForward` below).
    pub fn entail(&self, premise: &str, hypotheses: &[&str]) -> Result<Vec<f32>> {
        harness::entail(
            self,
            &self.tokenizer,
            self.max_seq_len,
            self.batch_budget_tokens,
            self.entail_index,
            premise,
            hypotheses,
        )
    }
}

impl RawForward for InProcessBurnNli {
    fn forward_logits(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        token_type_ids: &[i64],
        n: usize,
        seq: usize,
    ) -> Result<Vec<f32>> {
        let mk = |v: &[i64]| {
            Tensor::<BurnBackend, 2, Int>::from_data(TensorData::new(v.to_vec(), [n, seq]), &self.device)
        };
        let logits = self
            .model
            .forward(mk(input_ids), mk(attention_mask), mk(token_type_ids));
        logits
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow!("burn logits to_vec: {e:?}"))
    }
}
