//! In-process XLM-RoBERTa cross-encoder reranker on the burn backend. Mirrors [`super::InProcessBurnNli`]:
//! it loads the tokenizer + safetensors weights (with the XLM-R -> burn key remap) and builds the
//! [`RobertaRerankerModel`] from the model's `config.json`; `rerank()` delegates the
//! tokenization/truncation/batching to the shared [`crate::harness::rerank`], and this engine only
//! supplies the raw cross-encoder forward via [`RerankForward`], so ORT and burn cannot diverge in
//! anything but the forward itself.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, Result};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use tokenizers::{Tokenizer, TruncationParams, TruncationStrategy};

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

/// The loaded model + tokenizer for one `model_dir`, shared by every handle pointing at it.
/// Mirrors the ORT engine's `RerankInner`/`RERANK_CACHE`. Without it a burn build reloaded the
/// weights from disk on EVERY query, because `resolve_reranker` runs inside `retrieve()`.
struct BurnRerankInner {
    model: RobertaRerankerModel<BurnBackend>,
    device: Dev,
    tokenizer: Tokenizer,
    max_seq_len: usize,
    batch_budget_tokens: usize,
}

/// In-process reranker backed by the hand-written burn XLM-RoBERTa cross-encoder. A cheap handle
/// onto a [`BurnRerankInner`] shared with every other handle for the same model dir.
pub struct InProcessBurnReranker {
    inner: Arc<BurnRerankInner>,
}

/// Keyed by canonicalized model dir alone: the burn backend selects its own device, so the ORT
/// key's provider / device-id / memory-limit components have no analogue here.
type BurnRerankCache = OnceLock<Mutex<HashMap<PathBuf, Arc<BurnRerankInner>>>>;
static BURN_RERANK_CACHE: BurnRerankCache = OnceLock::new();

impl InProcessBurnReranker {
    /// Load the tokenizer (`tokenizer.json`), model config (`config.json`), and weights (a
    /// `.safetensors` file) from `model_dir`, reusing an already-loaded model for the same dir.
    /// Fail-open: returns `Err` on any missing/garbage file or incomplete weight set; never panics.
    ///
    /// Caching brings the same contract the ORT engines already have: an edited config or weights
    /// file takes effect on the next process start, not on the next query.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let key = model_dir
            .canonicalize()
            .unwrap_or_else(|_| model_dir.to_path_buf());
        let cache = BURN_RERANK_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(inner) = cache
            .lock()
            .map_err(|_| anyhow!("burn rerank cache mutex poisoned"))?
            .get(&key)
        {
            return Ok(Self {
                inner: Arc::clone(inner),
            });
        }
        let built = Arc::new(Self::build_inner(model_dir)?);
        let mut guard = cache
            .lock()
            .map_err(|_| anyhow!("burn rerank cache mutex poisoned"))?;
        // `or_insert`, not `insert`: two threads that both missed the read above must end up
        // sharing ONE instance, with the loser's build dropped.
        let inner = Arc::clone(guard.entry(key).or_insert(built));
        Ok(Self { inner })
    }

    fn build_inner(model_dir: &Path) -> Result<BurnRerankInner> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("tokenizer load ({}): {e}", tokenizer_path.display()))?;
        // Truncate the (query, passage) pair longest-first (so the query survives) to the model's
        // max length — the SAME tokenizer-level truncation the ORT engine uses (`InProcessReranker`),
        // so both engines score identically on overlength inputs (a flat post-tokenize clamp would
        // trim only-second and diverge). The harness's `ids.truncate` is then a redundant safety
        // clamp. Padding is built by the harness from the attention mask, so leave it off here.
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: DEFAULT_MAX_SEQ_LEN,
                strategy: TruncationStrategy::LongestFirst,
                ..Default::default()
            }))
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

        Ok(BurnRerankInner {
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
            &self.inner.tokenizer,
            self.inner.max_seq_len,
            self.inner.batch_budget_tokens,
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
                &self.inner.device,
            )
        };
        self.inner
            .model
            .forward(mk(ids), mk(mask))
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow!("burn rerank logits to_vec: {e:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The staged weights dir the other burn tests are gated on. CI has no 1.5 GB model, so these
    /// skip there and the compile is what CI proves.
    fn staged_model_dir() -> Option<PathBuf> {
        std::env::var("RERANK_SAFETENSORS_DIR")
            .ok()
            .map(PathBuf::from)
    }

    /// The defect: `resolve_reranker` runs inside `retrieve()`, so before the cache every search
    /// rebuilt tokenizer, model and weights from disk.
    #[test]
    fn two_loads_of_one_dir_share_one_instance() {
        let Some(dir) = staged_model_dir() else {
            return;
        };
        let a = InProcessBurnReranker::load(&dir).unwrap();
        let b = InProcessBurnReranker::load(&dir).unwrap();
        assert!(
            Arc::ptr_eq(&a.inner, &b.inner),
            "a second load must reuse the cached model, not rebuild it"
        );
    }

    /// The race the cache itself introduces: whoever loses the insert must adopt the winner's
    /// instance rather than keep a second copy of a 1.5 GB model.
    #[test]
    fn concurrent_loads_converge_on_one_instance() {
        let Some(dir) = staged_model_dir() else {
            return;
        };
        let other = dir.clone();
        let h = std::thread::spawn(move || InProcessBurnReranker::load(&other).unwrap());
        let a = InProcessBurnReranker::load(&dir).unwrap();
        let b = h.join().unwrap();
        assert!(Arc::ptr_eq(&a.inner, &b.inner));
    }

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
