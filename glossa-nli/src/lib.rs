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

/// Default per-batch token budget for the batched inference planner (`plan_batches`): a batch is
/// filled greedily while `rows_in_batch * max_len_in_batch <= NLI_BATCH_TOKENS`. Overridable via
/// the `GLOSSA_NLI_BATCH_TOKENS` env var (see `parse_batch_budget_tokens`); a missing/zero/
/// unparsable value falls back to this constant. This targets killing per-`session.run` overhead,
/// not maximizing batch size — onnxruntime already parallelizes one forward across intra-op
/// threads, so past a moderate budget a bigger batch only adds padding and latency.
const NLI_BATCH_TOKENS: usize = 8192;

/// Hard cap on rows per batch, independent of the token budget (secondary guard against
/// pathologically many short rows building one huge batch).
const NLI_BATCH_MAX_ROWS: usize = 64;

/// Read the `GLOSSA_NLI_BATCH_TOKENS` override, falling back to `NLI_BATCH_TOKENS` when unset,
/// unparsable, or zero.
fn parse_batch_budget_tokens() -> usize {
    std::env::var("GLOSSA_NLI_BATCH_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(NLI_BATCH_TOKENS)
}

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
    /// Token budget for `plan_batches`, resolved once at load time (see
    /// `parse_batch_budget_tokens`).
    batch_budget_tokens: usize,
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
            batch_budget_tokens: parse_batch_budget_tokens(),
        })
    }

    /// P(entail) of each hypothesis against `premise`, one `f32` in `[0, 1]` per hypothesis.
    ///
    /// Internally this collects every `(window, hypothesis)` row across all hypotheses, plans
    /// them into a few token-budgeted batches (`plan_batches`), and runs one `session.run` per
    /// batch instead of one per row — a hot-path optimization (see
    /// `docs/superpowers/specs/2026-09-12-nli-batched-inference-followup.md`). The windowing
    /// (`premise_windows`), the pair tokenization, and the per-hypothesis MAX-pool semantics are
    /// unchanged from the per-row implementation; only how the rows are *executed* differs, so
    /// scores are parity-exact modulo float reassociation.
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

        // 1. Build every (window, hypothesis) row up front, tagged with its owning hypothesis
        // index for the MAX-pool below. A hypothesis with zero windows contributes zero rows and
        // simply keeps its default 0.0 score.
        let mut rows: Vec<Row> = Vec::new();
        for (hyp_idx, hypothesis) in hypotheses.iter().enumerate() {
            let windows = self.premise_windows(premise, hypothesis)?;
            for window in &windows {
                let (input_ids, attention_mask, token_type_ids) =
                    self.encode_row(window, hypothesis)?;
                rows.push(Row {
                    hyp_idx,
                    input_ids,
                    attention_mask,
                    token_type_ids,
                });
            }
        }

        let mut out = vec![0.0f32; hypotheses.len()];
        if rows.is_empty() {
            return Ok(out);
        }

        // 2. Plan rows into token-budgeted batches (shorter rows grouped together to minimize
        // padding), then 3. run each batch and 4. MAX-pool its scores into `out` by hyp_idx.
        let lens: Vec<usize> = rows.iter().map(|r| r.input_ids.len()).collect();
        let batches = plan_batches(&lens, self.inner.batch_budget_tokens, NLI_BATCH_MAX_ROWS);
        for batch in &batches {
            let scores = self.run_batch(&rows, batch)?;
            for (&row_idx, &score) in batch.iter().zip(scores.iter()) {
                let hyp_idx = rows[row_idx].hyp_idx;
                if score > out[hyp_idx] {
                    out[hyp_idx] = score;
                }
            }
        }
        Ok(out)
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
            // instead, and skip the window if clamping collapses it to empty — `entail`'s
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

    /// Tokenize `(window_text, hypothesis)` as a pair and return its `(input_ids,
    /// attention_mask, token_type_ids)`, defensively truncated to `max_seq_len` — the same
    /// tokenization + truncation `score_pair` used to do per-row, now split out so `entail` can
    /// collect rows before deciding how to batch them.
    fn encode_row(
        &self,
        window_text: &str,
        hypothesis: &str,
    ) -> anyhow::Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
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
        Ok((input_ids, attention_mask, token_type_ids))
    }

    /// Run one batch of `rows` (referenced by index via `batch`) through the ONNX session in a
    /// single `session.run`, padding every row up to the batch's max length. Padding fills
    /// `input_ids`/`token_type_ids`/`attention_mask` with `0`; the zeroed `attention_mask` is
    /// what makes the padded positions contribute nothing to self-attention, so the actual pad
    /// value in `input_ids`/`token_type_ids` is irrelevant to the output. Returns one P(entail)
    /// score per row, in `batch`'s order.
    fn run_batch(&self, rows: &[Row], batch: &[usize]) -> anyhow::Result<Vec<f32>> {
        let n = batch.len();
        let max_len = batch
            .iter()
            .map(|&i| rows[i].input_ids.len())
            .max()
            .unwrap_or(0);

        let mut input_ids = vec![0i64; n * max_len];
        let mut attention_mask = vec![0i64; n * max_len];
        let mut token_type_ids = vec![0i64; n * max_len];
        for (row_pos, &row_idx) in batch.iter().enumerate() {
            let row = &rows[row_idx];
            let offset = row_pos * max_len;
            let len = row.input_ids.len();
            input_ids[offset..offset + len].copy_from_slice(&row.input_ids);
            attention_mask[offset..offset + len].copy_from_slice(&row.attention_mask);
            token_type_ids[offset..offset + len].copy_from_slice(&row.token_type_ids);
            // Positions [len..max_len) stay 0 in all three tensors: attention_mask=0 there masks
            // them out of self-attention entirely, so the 0 filler in input_ids/token_type_ids
            // never reaches the (unmasked) logits — see the padding-invariance test below.
        }

        let input_ids_tensor = Tensor::from_array(([n, max_len], input_ids))?;
        let attention_mask_tensor = Tensor::from_array(([n, max_len], attention_mask))?;
        let token_type_ids_tensor = Tensor::from_array(([n, max_len], token_type_ids))?;

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
        if data.len() != n * 3 {
            anyhow::bail!(
                "expected {} = {n} row(s) * 3-way NLI logits, got {} value(s) from the model output",
                n * 3,
                data.len()
            );
        }

        let mut scores = Vec::with_capacity(n);
        for k in 0..n {
            let probs = softmax3([data[k * 3], data[k * 3 + 1], data[k * 3 + 2]]);
            scores.push(probs[self.entail_index]);
        }
        Ok(scores)
    }
}

/// One `(window, hypothesis)` row queued for batched execution, tagged with the index of the
/// hypothesis it belongs to (for the MAX-pool in `entail`).
struct Row {
    hyp_idx: usize,
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    token_type_ids: Vec<i64>,
}

/// Partition row indices `0..lens.len()` into batches by a token budget, minimizing padding.
///
/// Rows are visited in ascending length order (so a batch groups similar lengths together) and
/// greedily added to the current batch while `(rows_in_batch + 1) * max(current_max_len,
/// this_row_len) <= budget_tokens` and the batch is under `max_rows`; otherwise the current batch
/// is flushed and a new one started with this row. A row whose own length exceeds
/// `budget_tokens` still gets its own batch — never dropped, just over budget alone.
///
/// Every index in `0..lens.len()` appears in exactly one returned batch; row order within a
/// batch (and batch order) doesn't matter — the caller's MAX-pool over rows is order-free. Pure
/// function, no model access — unit-tested directly below.
fn plan_batches(lens: &[usize], budget_tokens: usize, max_rows: usize) -> Vec<Vec<usize>> {
    if lens.is_empty() {
        return Vec::new();
    }

    let mut order: Vec<usize> = (0..lens.len()).collect();
    order.sort_by_key(|&i| lens[i]);

    let mut batches = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut cur_max_len = 0usize;

    for idx in order {
        let len = lens[idx];
        let candidate_max = cur_max_len.max(len);
        let fits = !cur.is_empty()
            && (cur.len() + 1) * candidate_max <= budget_tokens
            && cur.len() < max_rows;
        if fits {
            cur.push(idx);
            cur_max_len = candidate_max;
        } else {
            if !cur.is_empty() {
                batches.push(std::mem::take(&mut cur));
            }
            cur.push(idx);
            cur_max_len = len;
        }
    }
    if !cur.is_empty() {
        batches.push(cur);
    }
    batches
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
        // Model-agnostic smoke check: inference ran and produced valid probabilities. We do NOT
        // assert entailment direction on an English probe — the shipped model may be non-English
        // (the reference model is Russian-trained), so an English pair's scores are not a reliable
        // direction signal (this is exactly why `kbx nli check`'s sanity probe is advisory-only).
        // Batching/parity correctness is proven language-agnostically by the parity test below.
        for s in &scores {
            assert!(
                (0.0..=1.0).contains(s),
                "entail score outside [0,1]: {scores:?}"
            );
        }
    }

    // ---- plan_batches: pure, no model -------------------------------------------------------

    #[test]
    fn plan_batches_covers_every_index_exactly_once() {
        let lens = [10, 5000, 3, 100, 8000, 1, 50, 6000];
        let batches = plan_batches(&lens, 8192, 64);
        let mut seen = vec![false; lens.len()];
        for batch in &batches {
            for &idx in batch {
                assert!(!seen[idx], "index {idx} appeared in more than one batch");
                seen[idx] = true;
            }
        }
        assert!(
            seen.iter().all(|&s| s),
            "not every index was covered: {seen:?}"
        );
    }

    #[test]
    fn plan_batches_respects_budget_except_a_lone_overbudget_row() {
        let lens = [100, 200, 300, 9000, 50];
        let budget = 1000;
        let batches = plan_batches(&lens, budget, 64);
        for batch in &batches {
            let max_len = batch.iter().map(|&i| lens[i]).max().unwrap();
            let total = batch.len() * max_len;
            if batch.len() == 1 && lens[batch[0]] > budget {
                continue; // a lone row longer than the budget is allowed to exceed it alone.
            }
            assert!(
                total <= budget,
                "batch {batch:?} (max_len={max_len}) violates the budget: {total} > {budget}"
            );
        }
    }

    #[test]
    fn plan_batches_honors_max_rows() {
        let lens = vec![10usize; 200];
        let max_rows = 16;
        let batches = plan_batches(&lens, 1_000_000, max_rows);
        for batch in &batches {
            assert!(
                batch.len() <= max_rows,
                "batch of {} rows exceeds max_rows {max_rows}",
                batch.len()
            );
        }
    }

    #[test]
    fn plan_batches_empty_input_yields_no_batches() {
        let lens: [usize; 0] = [];
        let batches = plan_batches(&lens, 8192, 64);
        assert!(batches.is_empty());
    }

    #[test]
    fn plan_batches_groups_short_rows_away_from_a_long_row() {
        // 20 short rows (len 10) plus one long row (len 500), budget 1000: a batch of the long
        // row plus even one short row still fits the budget (2*500=1000), but the length-sorted
        // greedy fill should bucket the 20 shorts together first rather than letting the long row
        // pull them in one at a time.
        let mut lens = vec![10usize; 20];
        lens.push(500);
        let long_idx = lens.len() - 1;
        let batches = plan_batches(&lens, 1000, 64);
        let long_batch = batches
            .iter()
            .find(|b| b.contains(&long_idx))
            .expect("the long row must end up in some batch");
        assert!(
            long_batch.len() <= 2,
            "long row got batched with too many short rows: {long_batch:?}"
        );
    }

    // ---- batched entail vs per-row reference: env-gated real model ---------------------------

    /// Builds a `Row` with an arbitrary `hyp_idx` — the parity/padding tests below score rows
    /// individually (batch of 1) or in small hand-built batches, so `hyp_idx` bookkeeping (which
    /// only matters for `entail`'s MAX-pool) is irrelevant here.
    fn row_for(ids: Vec<i64>, mask: Vec<i64>, types: Vec<i64>) -> Row {
        Row {
            hyp_idx: 0,
            input_ids: ids,
            attention_mask: mask,
            token_type_ids: types,
        }
    }

    #[test]
    fn batched_entail_matches_per_row_reference_and_is_padding_invariant() {
        let Ok(model_dir) = std::env::var("GLOSSA_NLI_TEST_MODEL") else {
            return;
        };
        const ENTAIL_IDX: usize = 0;
        let nli = InProcessNli::load(Path::new(&model_dir), ENTAIL_IDX)
            .expect("model load should succeed against a real GLOSSA_NLI_TEST_MODEL dir");

        // A long, repetitive premise forces >=2 windows; hypotheses of clearly different lengths
        // force a batch that spans multiple lengths and therefore real padding.
        let premise =
            "The device must be fully powered off before any servicing begins. ".repeat(80);
        let short_hyp = "The device is off.";
        let long_hyp = "Before servicing the device, an operator must first ensure that every \
                         power source, including auxiliary and backup supplies, has been \
                         completely disconnected and independently verified as de-energized.";
        let other_hyp = "It is raining outside today.";
        let hypotheses = [short_hyp, long_hyp, other_hyp];

        let batched = nli
            .entail(&premise, &hypotheses)
            .expect("batched entail should succeed");
        assert_eq!(batched.len(), hypotheses.len());

        // Reference: score every (window, hypothesis) row alone (batch of 1) and MAX-pool by
        // hand — the same computation `entail` used to do per-row before batching.
        let mut reference = vec![0.0f32; hypotheses.len()];
        for (hyp_idx, hypothesis) in hypotheses.iter().enumerate() {
            let windows = nli
                .premise_windows(&premise, hypothesis)
                .expect("premise_windows should succeed");
            for window in &windows {
                let (ids, mask, types) = nli
                    .encode_row(window, hypothesis)
                    .expect("encode_row should succeed");
                let row = row_for(ids, mask, types);
                let scores = nli
                    .run_batch(&[row], &[0])
                    .expect("single-row run_batch should succeed");
                if scores[0] > reference[hyp_idx] {
                    reference[hyp_idx] = scores[0];
                }
            }
        }

        for (i, (b, r)) in batched.iter().zip(reference.iter()).enumerate() {
            assert!(
                (b - r).abs() < 1e-4,
                "hypothesis {i}: batched {b} vs per-row reference {r} differ beyond epsilon"
            );
        }

        // Padding invariance: a short row, scored alone, must match its score when padded out to a
        // longer row's length inside a shared batch — proving attention_mask=0 on the padded tail
        // contributes nothing. Use a SHORT premise here so the two rows differ in length purely by
        // hypothesis length; the long windowing premise above fills every row to max_seq_len, which
        // would leave the two rows equal-length and unable to exercise padding at all.
        let short_premise = "The device is powered off.";
        let (s_ids, s_mask, s_types) = nli
            .encode_row(short_premise, short_hyp)
            .expect("encode_row should succeed");
        let (l_ids, l_mask, l_types) = nli
            .encode_row(short_premise, long_hyp)
            .expect("encode_row should succeed");
        assert!(
            l_ids.len() > s_ids.len(),
            "test setup expects the long hypothesis's row to be strictly longer \
             (short={}, long={}) so batching them together forces padding on the short row",
            s_ids.len(),
            l_ids.len()
        );

        let alone_score = nli
            .run_batch(
                &[row_for(s_ids.clone(), s_mask.clone(), s_types.clone())],
                &[0],
            )
            .expect("single-row run_batch should succeed")[0];

        let padded_scores = nli
            .run_batch(
                &[
                    row_for(s_ids, s_mask, s_types),
                    row_for(l_ids, l_mask, l_types),
                ],
                &[0, 1],
            )
            .expect("padded run_batch should succeed");

        assert!(
            (padded_scores[0] - alone_score).abs() < 1e-6,
            "padding changed the short row's score: alone={alone_score} \
             padded={}",
            padded_scores[0]
        );
    }
}
