//! Engine-agnostic NLI scoring harness shared by every inference engine (ORT, burn/wgpu).
//!
//! The only engine-specific step is the raw transformer forward pass, expressed as the
//! [`RawForward`] trait: given a padded batch of token rows, return the raw 3-way logits. Everything
//! else — premise windowing, pair tokenization, token-budget batching, softmax, the `entail_index`
//! pick, and the per-hypothesis MAX-pool — lives here and is driven by the same code for every
//! engine, so two engines pointed at the same model cannot diverge in anything but the forward
//! itself. This is what makes cross-engine parity hold by construction (spec §2.1/§4).

use tokenizers::Tokenizer;

/// Fallback max sequence length (tokens) used for premise windowing when the model/tokenizer don't
/// expose one. BERT-family NLI models are near-universally trained at 512.
pub const DEFAULT_MAX_SEQ_LEN: usize = 512;

/// Default per-batch token budget for the batched planner (`plan_batches`): a batch is filled
/// greedily while `rows_in_batch * max_len_in_batch <= NLI_BATCH_TOKENS`. Overridable via
/// `GLOSSA_NLI_BATCH_TOKENS` (see [`parse_batch_budget_tokens`]).
///
/// Default = one max-length row per batch, i.e. effectively sequential — deliberate for the CPU
/// execution provider (a bigger tensor is a net loss on CPU; the k-fold win is a GPU property).
/// Raise the env var (e.g. `16384`) for a GPU build. The batching path is parity-tested at any
/// budget; only the default is tuned to not regress CPU.
pub const NLI_BATCH_TOKENS: usize = DEFAULT_MAX_SEQ_LEN;

/// Hard cap on rows per batch, independent of the token budget (guards against pathologically many
/// short rows building one huge batch).
pub const NLI_BATCH_MAX_ROWS: usize = 64;

/// Read the `GLOSSA_NLI_BATCH_TOKENS` override, falling back to [`NLI_BATCH_TOKENS`] when unset,
/// unparsable, or zero.
pub fn parse_batch_budget_tokens() -> usize {
    std::env::var("GLOSSA_NLI_BATCH_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(NLI_BATCH_TOKENS)
}

/// The raw transformer forward — the ONLY engine-specific step. Given a padded batch of `n` rows,
/// each `seq` tokens long (row-major flat arrays of length `n * seq`; padded positions carry
/// `attention_mask == 0`), return the raw 3-way NLI logits, row-major, length `n * 3`. No softmax:
/// the harness owns softmax and the `entail_index` pick. Fail-open: return `Err`, never panic.
pub trait RawForward {
    fn forward_logits(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        token_type_ids: &[i64],
        n: usize,
        seq: usize,
    ) -> anyhow::Result<Vec<f32>>;
}

/// One `(window, hypothesis)` row queued for batched execution, tagged with the index of the
/// hypothesis it belongs to (for the MAX-pool in [`entail`]).
pub struct Row {
    pub hyp_idx: usize,
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub token_type_ids: Vec<i64>,
}

/// P(entail) of each hypothesis against `premise`, one `f32` in `[0, 1]` per hypothesis, using
/// `fwd` for the raw forward. Builds every `(window, hypothesis)` row, plans them into
/// token-budgeted batches, runs each batch through `fwd`, softmaxes, picks `entail_index`, and
/// MAX-pools per hypothesis. Engine-agnostic: ORT and burn share this exact code.
pub fn entail(
    fwd: &dyn RawForward,
    tokenizer: &Tokenizer,
    max_seq_len: usize,
    batch_budget_tokens: usize,
    entail_index: usize,
    premise: &str,
    hypotheses: &[&str],
) -> anyhow::Result<Vec<f32>> {
    if hypotheses.is_empty() {
        return Ok(Vec::new());
    }
    if entail_index >= 3 {
        anyhow::bail!("entail_index {entail_index} out of range for 3-way NLI logits");
    }

    let mut rows: Vec<Row> = Vec::new();
    for (hyp_idx, hypothesis) in hypotheses.iter().enumerate() {
        let windows = premise_windows(tokenizer, max_seq_len, premise, hypothesis)?;
        for window in &windows {
            let (input_ids, attention_mask, token_type_ids) =
                encode_row(tokenizer, max_seq_len, window, hypothesis)?;
            rows.push(Row { hyp_idx, input_ids, attention_mask, token_type_ids });
        }
    }

    let mut out = vec![0.0f32; hypotheses.len()];
    if rows.is_empty() {
        return Ok(out);
    }

    let lens: Vec<usize> = rows.iter().map(|r| r.input_ids.len()).collect();
    let batches = plan_batches(&lens, batch_budget_tokens, NLI_BATCH_MAX_ROWS);
    for batch in &batches {
        let scores = run_batch(fwd, &rows, batch, entail_index)?;
        for (&row_idx, &score) in batch.iter().zip(scores.iter()) {
            let hyp_idx = rows[row_idx].hyp_idx;
            if score > out[hyp_idx] {
                out[hyp_idx] = score;
            }
        }
    }
    Ok(out)
}

/// Pad one batch of `rows` (referenced by index) to the batch's max length, run it through `fwd`,
/// softmax each row's logits, and return the `entail_index` probability per row (in `batch` order).
/// Padding fills all three tensors with `0`; the zeroed `attention_mask` masks padded positions out
/// of self-attention, so the pad filler never reaches the logits.
pub fn run_batch(
    fwd: &dyn RawForward,
    rows: &[Row],
    batch: &[usize],
    entail_index: usize,
) -> anyhow::Result<Vec<f32>> {
    let n = batch.len();
    let seq = batch.iter().map(|&i| rows[i].input_ids.len()).max().unwrap_or(0);

    let mut input_ids = vec![0i64; n * seq];
    let mut attention_mask = vec![0i64; n * seq];
    let mut token_type_ids = vec![0i64; n * seq];
    for (row_pos, &row_idx) in batch.iter().enumerate() {
        let row = &rows[row_idx];
        let offset = row_pos * seq;
        let len = row.input_ids.len();
        input_ids[offset..offset + len].copy_from_slice(&row.input_ids);
        attention_mask[offset..offset + len].copy_from_slice(&row.attention_mask);
        token_type_ids[offset..offset + len].copy_from_slice(&row.token_type_ids);
    }

    let logits = fwd.forward_logits(&input_ids, &attention_mask, &token_type_ids, n, seq)?;
    if logits.len() != n * 3 {
        anyhow::bail!(
            "expected {} = {n} row(s) * 3-way NLI logits, got {} value(s) from the forward",
            n * 3,
            logits.len()
        );
    }

    let mut scores = Vec::with_capacity(n);
    for k in 0..n {
        let probs = softmax3([logits[k * 3], logits[k * 3 + 1], logits[k * 3 + 2]]);
        scores.push(probs[entail_index]);
    }
    Ok(scores)
}

/// Split `premise` into overlapping token windows so each `(window, hypothesis)` pair fits within
/// `max_seq_len`. Returns the whole premise as a single window when it already fits.
pub fn premise_windows<'p>(
    tokenizer: &Tokenizer,
    max_seq_len: usize,
    premise: &'p str,
    hypothesis: &str,
) -> anyhow::Result<Vec<&'p str>> {
    let hyp_len = tokenizer
        .encode(hypothesis, false)
        .map_err(|e| anyhow::anyhow!("hypothesis tokenize: {e}"))?
        .get_ids()
        .len();

    // Probe the pair post-processor's special-token overhead without guessing token names — works
    // whether the export is BERT-style (`[CLS] A [SEP] B [SEP]`) or RoBERTa-style.
    let probe_len = tokenizer
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

    let premise_enc = tokenizer
        .encode(premise, false)
        .map_err(|e| anyhow::anyhow!("premise tokenize: {e}"))?;
    let ids = premise_enc.get_ids();
    if ids.is_empty() || ids.len() <= budget {
        return Ok(vec![premise]);
    }

    // Overlapping windows, stride ~75% of budget, so a fact near a window boundary is still fully
    // inside at least one window.
    let offsets = premise_enc.get_offsets();
    let stride = (budget * 3 / 4).max(1);
    let mut windows = Vec::new();
    let mut start = 0usize;
    loop {
        let end = (start + budget).min(ids.len());
        let byte_start = offsets[start].0;
        let byte_end = offsets[end - 1].1;
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

/// Tokenize `(window_text, hypothesis)` as a pair and return `(input_ids, attention_mask,
/// token_type_ids)`, defensively truncated to `max_seq_len`.
pub fn encode_row(
    tokenizer: &Tokenizer,
    max_seq_len: usize,
    window_text: &str,
    hypothesis: &str,
) -> anyhow::Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
    let encoding = tokenizer
        .encode((window_text, hypothesis), true)
        .map_err(|e| anyhow::anyhow!("pair tokenize: {e}"))?;

    let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let mut attention_mask: Vec<i64> =
        encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
    let mut token_type_ids: Vec<i64> =
        encoding.get_type_ids().iter().map(|&x| x as i64).collect();

    if input_ids.len() > max_seq_len {
        input_ids.truncate(max_seq_len);
        attention_mask.truncate(max_seq_len);
        token_type_ids.truncate(max_seq_len);
    }
    Ok((input_ids, attention_mask, token_type_ids))
}

/// Partition row indices `0..lens.len()` into batches by a token budget, minimizing padding. Rows
/// are visited in ascending length order and greedily added while
/// `(rows_in_batch + 1) * max_len <= budget_tokens` and under `max_rows`; a row longer than the
/// budget gets its own over-budget batch. Every index appears in exactly one batch. Pure.
pub fn plan_batches(lens: &[usize], budget_tokens: usize, max_rows: usize) -> Vec<Vec<usize>> {
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

/// Round `idx` down to the nearest UTF-8 char boundary of `s` (clamped to `s.len()`).
pub fn floor_char_boundary(s: &str, idx: usize) -> usize {
    let mut idx = idx.min(s.len());
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Numerically stable softmax over exactly 3 logits.
pub fn softmax3(logits: [f32; 3]) -> [f32; 3] {
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
        assert!(seen.iter().all(|&s| s), "not every index was covered: {seen:?}");
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
                continue;
            }
            assert!(total <= budget, "batch {batch:?} violates budget: {total} > {budget}");
        }
    }

    #[test]
    fn plan_batches_honors_max_rows() {
        let lens = vec![10usize; 200];
        let max_rows = 16;
        let batches = plan_batches(&lens, 1_000_000, max_rows);
        for batch in &batches {
            assert!(batch.len() <= max_rows, "batch of {} rows exceeds {max_rows}", batch.len());
        }
    }

    #[test]
    fn plan_batches_empty_input_yields_no_batches() {
        let lens: [usize; 0] = [];
        assert!(plan_batches(&lens, 8192, 64).is_empty());
    }

    #[test]
    fn plan_batches_groups_short_rows_away_from_a_long_row() {
        let mut lens = vec![10usize; 20];
        lens.push(500);
        let long_idx = lens.len() - 1;
        let batches = plan_batches(&lens, 1000, 64);
        let long_batch = batches.iter().find(|b| b.contains(&long_idx)).expect("long row batched");
        assert!(long_batch.len() <= 2, "long row batched with too many shorts: {long_batch:?}");
    }

    #[test]
    fn softmax3_sums_to_one_and_is_shift_invariant() {
        let a = softmax3([1.0, 2.0, 3.0]);
        let b = softmax3([101.0, 102.0, 103.0]);
        assert!((a.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "softmax not shift-invariant: {a:?} vs {b:?}");
        }
    }
}
