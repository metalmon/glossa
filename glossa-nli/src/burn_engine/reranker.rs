//! In-process XLM-RoBERTa cross-encoder reranker on the burn backend. Mirrors [`super::InProcessBurnNli`]:
//! it loads the tokenizer + safetensors weights (with the XLM-R -> burn key remap) and builds the
//! [`RobertaRerankerModel`] from the model's `config.json`; `rerank()` delegates the
//! tokenization/truncation/batching to the shared [`crate::harness::rerank`], and this engine only
//! supplies the raw cross-encoder forward via [`RerankForward`], so ORT and burn cannot diverge in
//! anything but the forward itself.

use std::path::Path;

use anyhow::{anyhow, Result};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use tokenizers::Tokenizer;

use super::reranker_model::{RobertaRerankerConfig, RobertaRerankerModel};
use super::{resolve_weights_file, BurnBackend, Dev};
use crate::harness::{self, RerankForward, DEFAULT_MAX_SEQ_LEN};

/// Read a required unsigned-integer field from the HF `config.json` object.
fn cfg_usize(v: &serde_json::Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| anyhow!("config.json missing/invalid `{key}` (expected unsigned int)"))
}

/// In-process reranker backed by the hand-written burn XLM-RoBERTa cross-encoder. Holds the loaded
/// model, its device, the tokenizer, and the shared truncation/batching config.
pub struct InProcessBurnReranker {
    model: RobertaRerankerModel<BurnBackend>,
    device: Dev,
    tokenizer: Tokenizer,
    max_seq_len: usize,
    batch_budget_tokens: usize,
}

impl InProcessBurnReranker {
    /// Load the tokenizer (`tokenizer.json`), model config (`config.json`), and weights (a
    /// `.safetensors` file) from `model_dir`. Fail-open: returns `Err` on any missing/garbage file
    /// or incomplete weight set; never panics.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("tokenizer load ({}): {e}", tokenizer_path.display()))?;
        // The harness owns truncation (longest-first, so the query survives) and padding.
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow!("tokenizer truncation config: {e}"))?;
        tokenizer.with_padding(None);

        let config_path = model_dir.join("config.json");
        let config_bytes = std::fs::read(&config_path)
            .map_err(|e| anyhow!("reading config ({}): {e}", config_path.display()))?;
        let hf: serde_json::Value = serde_json::from_slice(&config_bytes)
            .map_err(|e| anyhow!("parsing config ({}): {e}", config_path.display()))?;
        let layer_norm_eps = hf
            .get("layer_norm_eps")
            .and_then(|x| x.as_f64())
            .ok_or_else(|| anyhow!("config.json missing/invalid `layer_norm_eps`"))?;
        // The reranker is a single-logit relevance head; take num_labels from id2label, default 1.
        let num_labels = hf
            .get("id2label")
            .and_then(|x| x.as_object())
            .map(|m| m.len())
            .unwrap_or(1)
            .max(1);
        let cfg = RobertaRerankerConfig::new(
            cfg_usize(&hf, "vocab_size")?,
            cfg_usize(&hf, "hidden_size")?,
            cfg_usize(&hf, "num_hidden_layers")?,
            cfg_usize(&hf, "num_attention_heads")?,
            cfg_usize(&hf, "intermediate_size")?,
            cfg_usize(&hf, "max_position_embeddings")?,
            cfg_usize(&hf, "pad_token_id")?,
            layer_norm_eps,
            num_labels,
        );

        let device = Dev::default();
        let mut model = cfg.init::<BurnBackend>(&device);

        let weights = resolve_weights_file(model_dir)?;
        // XLM-R safetensors names -> burn module tree:
        //   `roberta.` prefix stripped (embeddings/encoder sit at the model root; `classifier.*`
        //   is already un-prefixed), `.self.` -> `.self_attention.`, `LayerNorm` -> `layer_norm`.
        let mut store = SafetensorsStore::from_file(
            weights
                .to_str()
                .ok_or_else(|| anyhow!("non-utf8 weights path"))?,
        )
        .with_from_adapter(PyTorchToBurnAdapter)
        .with_key_remapping(r"^roberta\.", "")
        .with_key_remapping(r"\.self\.", ".self_attention.")
        .with_key_remapping("LayerNorm", "layer_norm")
        .allow_partial(true);
        let result = model
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
            model,
            device,
            tokenizer,
            max_seq_len: DEFAULT_MAX_SEQ_LEN,
            batch_budget_tokens: harness::parse_batch_budget_tokens(),
        })
    }

    /// One relevance score per passage (higher = more relevant), in `passages` order. Delegates the
    /// tokenization/truncation/batching to the shared [`harness::rerank`]; this engine only supplies
    /// the raw forward ([`RerankForward`] below).
    pub fn rerank(&self, query: &str, passages: &[&str]) -> Result<Vec<f32>> {
        harness::rerank(
            self,
            &self.tokenizer,
            self.max_seq_len,
            self.batch_budget_tokens,
            query,
            passages,
        )
    }
}

impl RerankForward for InProcessBurnReranker {
    fn forward_logits(&self, ids: &[i64], mask: &[i64], n: usize, seq: usize) -> Result<Vec<f32>> {
        let mk = |v: &[i64]| {
            Tensor::<BurnBackend, 2, Int>::from_data(
                TensorData::new(v.to_vec(), [n, seq]),
                &self.device,
            )
        };
        self.model
            .forward(mk(ids), mk(mask))
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow!("burn rerank logits to_vec: {e:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden-parity + ordering (env-gated on `RERANK_SAFETENSORS_DIR`, so CI without the staged
    /// 1.5GB model stays green). Loads the real bge-reranker-v2-m3 en-ru weights, runs the burn
    /// forward on the fixed generic pair, and checks both the relevance order and the absolute
    /// logits against the verified torch/ONNX export.
    #[test]
    fn burn_reranker_matches_golden_and_orders() {
        let Ok(dir) = std::env::var("RERANK_SAFETENSORS_DIR") else {
            return;
        };
        let rr = InProcessBurnReranker::load(Path::new(&dir)).unwrap();
        let q = "What is the capital of France?";
        let s = rr
            .rerank(
                q,
                &[
                    "Paris is the capital of France.",
                    "Bananas are a yellow fruit.",
                ],
            )
            .unwrap();
        assert_eq!(s.len(), 2);
        assert!(s[0] > s[1], "relevant {} > irrelevant {}", s[0], s[1]);
        // Golden reference from the verified torch/ONNX export (parity 7.6e-6):
        assert!(
            (s[0] - 8.7818).abs() < 0.2,
            "relevant {} vs golden 8.7818",
            s[0]
        );
        assert!(
            (s[1] - (-11.0175)).abs() < 0.2,
            "irrelevant {} vs golden -11.0175",
            s[1]
        );
    }
}
