//! In-process NLI scorer over an ONNX 3-way NLI model (ort, CPU). Standalone crate — it does NOT
//! depend on `glossa`; `glossa` owns the `NliScorer` trait and implements it for `InProcessNli`
//! (see glossa's `src/gate/nli_engine.rs`), mirroring the glossa-constraint adapter pattern.
//!
//! Fail-open (hard requirement): `load()` and `entail()` return `anyhow::Result` and may return
//! `Err` on any failure (missing file, tokenizer error, session error, degenerate output). Neither
//! ever panics or calls `.unwrap()`/`.expect()` on fallible IO/inference — the caller maps
//! `Err`/`None` to AC-only scoring.

// Exactly one inference engine per build.
#[cfg(all(feature = "nli-ort", feature = "nli-burn-wgpu"))]
compile_error!("enable exactly ONE of `nli-ort` / `nli-burn-wgpu`, not both");

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::Tokenizer;

/// Fallback max sequence length (tokens) used for premise windowing when the model/tokenizer
/// don't expose one. BERT-family NLI models are near-universally trained at 512; Task 6 should
/// confirm this against the real export's position-embedding size if it differs.
const DEFAULT_MAX_SEQ_LEN: usize = 512;

/// The loaded model + tokenizer for one `model_dir`, shared across all `InProcessNli` handles
/// that point at the same directory (see `MODEL_CACHE`).
struct Inner {
    tokenizer: Tokenizer,
    // rc.13 `Session::run` takes `&mut self` (ONNX Runtime session internals are not thread
    // safe — see the `ort` docs on `Session::run`), so a session shared behind `&self` needs
    // interior mutability. One `Mutex` per process-global model instance serializes inference,
    // which matches the "one session per process" norm from spec §2.1a.
    session: Mutex<Session>,
    max_seq_len: usize,
}

/// Process-global load-once cache, keyed by canonicalized `model_dir`. Constructing two
/// `InProcessNli` for the same directory reuses one `Inner` (one tokenizer, one ONNX session)
/// instead of building a second one.
static MODEL_CACHE: OnceLock<Mutex<HashMap<PathBuf, Arc<Inner>>>> = OnceLock::new();

/// An in-process NLI scorer. Constructed from a local `model_dir` (`model.onnx` +
/// `tokenizer.json`) and the entailment class index (from the model's `id2label`, config-pinned
/// by the caller — see spec §2a.2).
pub struct InProcessNli {
    inner: Arc<Inner>,
    entail_index: usize,
}

impl InProcessNli {
    /// Load the model + tokenizer from `model_dir`. `entail_index` is the softmax index of the
    /// entailment class. Reuses a cached `Inner` for the same (canonicalized) `model_dir` rather
    /// than building a second ONNX session.
    pub fn load(model_dir: &Path, entail_index: usize) -> anyhow::Result<Self> {
        let cache_key = model_dir
            .canonicalize()
            .unwrap_or_else(|_| model_dir.to_path_buf());
        let cache = MODEL_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

        if let Some(inner) = cache
            .lock()
            .map_err(|_| anyhow::anyhow!("nli model cache mutex poisoned"))?
            .get(&cache_key)
        {
            return Ok(Self {
                inner: Arc::clone(inner),
                entail_index,
            });
        }

        // Build outside the lock (tokenizer + session load can be slow); then reconcile with the
        // cache. If another caller raced us to the same dir, keep whichever landed first so the
        // cache never ends up holding two sessions for one model_dir.
        let built = Arc::new(Self::build_inner(model_dir)?);
        let mut guard = cache
            .lock()
            .map_err(|_| anyhow::anyhow!("nli model cache mutex poisoned"))?;
        let inner = Arc::clone(guard.entry(cache_key).or_insert(built));
        Ok(Self {
            inner,
            entail_index,
        })
    }

    fn build_inner(model_dir: &Path) -> anyhow::Result<Inner> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            // `tokenizers::Result`'s error is a boxed trait object, not guaranteed
            // `std::error::Error + Send + Sync` across versions — map explicitly.
            .map_err(|e| anyhow::anyhow!("tokenizer load ({}): {e}", tokenizer_path.display()))?;
        // Premise windowing (below) does its own truncation; disable any truncation/padding
        // baked into tokenizer.json so it can't silently clip a window out from under us.
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("tokenizer truncation config: {e}"))?;
        tokenizer.with_padding(None);

        let model_path = model_dir.join("model.onnx");
        // ort's `SessionBuilder`-typed error (`ort::Error<SessionBuilder>`) carries a
        // `NonNull<OrtSessionOptions>` and is therefore NOT `Send + Sync`, so `?` cannot convert it
        // into `anyhow::Error` (whose `From<E>` requires `E: Send + Sync + 'static`). Map every
        // builder step through `Display` first, exactly as `commit_from_file` already does.
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("onnx session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("onnx session optimization level: {e}"))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("onnx session intra-threads: {e}"))?
            .commit_from_file(&model_path)
            .map_err(|e| anyhow::anyhow!("onnx session load ({}): {e}", model_path.display()))?;

        Ok(Inner {
            tokenizer,
            session: Mutex::new(session),
            max_seq_len: DEFAULT_MAX_SEQ_LEN,
        })
    }

    /// P(entail) of each hypothesis against `premise`, one `f32` in `[0, 1]` per hypothesis.
    pub fn entail(&self, premise: &str, hypotheses: &[&str]) -> anyhow::Result<Vec<f32>> {
        if hypotheses.is_empty() {
            return Ok(Vec::new());
        }
        if self.entail_index >= 3 {
            anyhow::bail!(
                "entail_index {} out of range for 3-way NLI logits",
                self.entail_index
            );
        }
        hypotheses
            .iter()
            .map(|hypothesis| self.entail_one(premise, hypothesis))
            .collect()
    }

    fn entail_one(&self, premise: &str, hypothesis: &str) -> anyhow::Result<f32> {
        let windows = self.premise_windows(premise, hypothesis)?;
        // MAX-pool across windows (spec §3.3): a hypothesis is entailed if ANY window entails it.
        let mut best = 0.0f32;
        for window in &windows {
            let score = self.score_pair(window, hypothesis)?;
            if score > best {
                best = score;
            }
        }
        Ok(best)
    }

    /// Split `premise` into overlapping token windows so each `(window, hypothesis)` pair fits
    /// within `max_seq_len` (spec §3.3). Returns the whole premise as a single window when it
    /// already fits.
    fn premise_windows<'p>(
        &self,
        premise: &'p str,
        hypothesis: &str,
    ) -> anyhow::Result<Vec<&'p str>> {
        let max_seq_len = self.inner.max_seq_len;

        let hyp_len = self
            .inner
            .tokenizer
            .encode(hypothesis, false)
            .map_err(|e| anyhow::anyhow!("hypothesis tokenize: {e}"))?
            .get_ids()
            .len();

        // Probe the pair post-processor's special-token overhead without guessing token names —
        // works whether the export is BERT-style (`[CLS] A [SEP] B [SEP]`, overhead 3) or
        // RoBERTa-style (`<s> A </s></s> B </s>`, overhead 4).
        let probe_len = self
            .inner
            .tokenizer
            .encode(("", hypothesis), true)
            .map_err(|e| anyhow::anyhow!("pair-overhead probe tokenize: {e}"))?
            .get_ids()
            .len();
        let overhead = probe_len.saturating_sub(hyp_len);

        let budget = max_seq_len.saturating_sub(hyp_len).saturating_sub(overhead);
        if budget == 0 {
            anyhow::bail!(
                "hypothesis ({hyp_len} tokens) + special tokens ({overhead}) already fill \
                 max_seq_len ({max_seq_len}); no room for any premise window"
            );
        }

        let premise_enc = self
            .inner
            .tokenizer
            .encode(premise, false)
            .map_err(|e| anyhow::anyhow!("premise tokenize: {e}"))?;
        let ids = premise_enc.get_ids();
        if ids.is_empty() || ids.len() <= budget {
            return Ok(vec![premise]);
        }

        // Overlapping windows, stride ~75% of budget, so a fact sitting near a window boundary is
        // still fully inside at least one window.
        let offsets = premise_enc.get_offsets();
        let stride = (budget * 3 / 4).max(1);
        let mut windows = Vec::new();
        let mut start = 0usize;
        loop {
            let end = (start + budget).min(ids.len());
            let byte_start = offsets[start].0;
            let byte_end = offsets[end - 1].1;
            // A normalizing tokenizer can emit offsets that don't land on a UTF-8 char boundary
            // (e.g. after character substitution during normalization); `&premise[a..b]` would
            // panic on those. Clamp down to the nearest valid boundaries via a checked `.get(..)`
            // instead, and skip the window if clamping collapses it to empty — `entail_one`'s
            // max-pool already treats zero windows as "no entailment signal" (score stays 0.0),
            // the same fallback path as an already-empty premise.
            let safe_start = floor_char_boundary(premise, byte_start);
            let safe_end = floor_char_boundary(premise, byte_end);
            if safe_start < safe_end {
                if let Some(window) = premise.get(safe_start..safe_end) {
                    windows.push(window);
                }
            }
            if end == ids.len() {
                break;
            }
            start += stride;
        }
        Ok(windows)
    }

    /// Tokenize `(window_text, hypothesis)` as a pair, run the ONNX session, and return
    /// P(entail) = softmax(logits)[entail_index].
    fn score_pair(&self, window_text: &str, hypothesis: &str) -> anyhow::Result<f32> {
        let encoding = self
            .inner
            .tokenizer
            .encode((window_text, hypothesis), true)
            .map_err(|e| anyhow::anyhow!("pair tokenize: {e}"))?;

        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mut attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let mut token_type_ids: Vec<i64> =
            encoding.get_type_ids().iter().map(|&x| x as i64).collect();

        let max_seq_len = self.inner.max_seq_len;
        if input_ids.len() > max_seq_len {
            // The windowing budget above is an estimate (probed via a "" + hypothesis pair); clip
            // defensively rather than let a rare mis-estimate crash the whole hypothesis.
            input_ids.truncate(max_seq_len);
            attention_mask.truncate(max_seq_len);
            token_type_ids.truncate(max_seq_len);
        }
        let seq_len = input_ids.len();

        let input_ids_tensor = Tensor::from_array(([1usize, seq_len], input_ids))?;
        let attention_mask_tensor = Tensor::from_array(([1usize, seq_len], attention_mask))?;
        let token_type_ids_tensor = Tensor::from_array(([1usize, seq_len], token_type_ids))?;

        let mut session = self
            .inner
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("nli onnx session mutex poisoned"))?;
        // Conventional HF/optimum export input names for BERT-family sequence-classification ONNX
        // graphs. Task 6 MUST confirm these against the real export (see task-2 report).
        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ])?;

        // Index by position (not name) so the output naming in the real export doesn't matter —
        // these classification heads have exactly one output (the logits). `SessionOutputs` has
        // no `get(usize)` (only `get(&str)`); `.values().next()` is the checked equivalent of
        // position-0 access — unlike `outputs[0]` (whose `Index<usize>` impl panics when the
        // graph emits zero outputs), this returns `None` instead of unwinding.
        let out0 = outputs
            .values()
            .next()
            .ok_or_else(|| anyhow::anyhow!("onnx session returned no outputs"))?;
        let (_shape, data) = out0
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("logits extraction: {e}"))?;
        if data.len() != 3 {
            anyhow::bail!(
                "expected 3-way NLI logits, got {} value(s) from the model output",
                data.len()
            );
        }
        let probs = softmax3([data[0], data[1], data[2]]);
        Ok(probs[self.entail_index])
    }
}

/// Round `idx` down to the nearest UTF-8 char boundary of `s` (clamped to `s.len()`). Used to
/// make tokenizer byte offsets safe to slice with even if a normalizing tokenizer produced an
/// offset that lands mid-char.
fn floor_char_boundary(s: &str, idx: usize) -> usize {
    let mut idx = idx.min(s.len());
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Numerically stable softmax over exactly 3 logits.
fn softmax3(logits: [f32; 3]) -> [f32; 3] {
    let max = logits[0].max(logits[1]).max(logits[2]);
    let exps = [
        (logits[0] - max).exp(),
        (logits[1] - max).exp(),
        (logits[2] - max).exp(),
    ];
    let sum: f32 = exps.iter().sum();
    [exps[0] / sum, exps[1] / sum, exps[2] / sum]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs real inference only against a model dir supplied via `GLOSSA_NLI_TEST_MODEL`
    /// (`model.onnx` + `tokenizer.json`). Skips (no-ops) when unset so CI stays green without the
    /// ONNX model artifact — but the test still COMPILES under `cargo clippy --workspace
    /// --all-targets`, which is what catches an `ort`/`tokenizers` API break across rc bumps.
    #[test]
    fn obvious_entailment_outranks_contradiction() {
        let Ok(model_dir) = std::env::var("GLOSSA_NLI_TEST_MODEL") else {
            return;
        };
        // The entailment softmax index is model-specific and config-pinned in real use (spec
        // §2a.2); a literal is fine here since this test targets one known model dir.
        const ENTAIL_IDX: usize = 0;

        let nli = InProcessNli::load(Path::new(&model_dir), ENTAIL_IDX)
            .expect("model load should succeed against a real GLOSSA_NLI_TEST_MODEL dir");
        let scores = nli
            .entail(
                "A dog is sleeping on the couch.",
                &["An animal is resting.", "The room is empty."],
            )
            .expect("entail should succeed against a real model");
        assert_eq!(scores.len(), 2);
        assert!(
            scores[0] > scores[1],
            "expected the entailment score to beat the contradiction score, got {scores:?}"
        );
    }
}
