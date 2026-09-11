//! The gate<->scorer seam (spec 2.1). `NliScorer` is a THIN entailment primitive: P(entail) of each
//! hypothesis (an answer claim) against a premise (a cited chunk). The scorer OWNS everything needing
//! the model tokenizer (windowing an over-length premise + max-pool over windows — Plan 2). The gate
//! owns claim-split, cross-chunk max, and DfTable mean_filt (Task 4) — all tokenizer-free.

use crate::gate::config::VerifyConfig;
use crate::gate::df::DfTable;

/// P(entail) of each hypothesis against `premise`. See spec 2.1.
pub trait NliScorer {
    /// P(entail) in [0,1] of each hypothesis against `premise`, batched — one score per hypothesis,
    /// already max-pooled over any internal windowing. The returned vector's length MUST equal
    /// `hypotheses.len()`.
    fn entail(&self, premise: &str, hypotheses: &[&str]) -> anyhow::Result<Vec<f32>>;
}

/// Deterministic test double: returns a fixed score vector, truncated to the hypothesis count (NOT
/// padded), so length-mismatch handling downstream (Task 4) is testable.
#[cfg(test)]
pub struct MockNli {
    scores: Vec<f32>,
}

#[cfg(test)]
impl MockNli {
    pub fn new(scores: Vec<f32>) -> Self {
        Self { scores }
    }
}

#[cfg(test)]
impl NliScorer for MockNli {
    fn entail(&self, _premise: &str, hypotheses: &[&str]) -> anyhow::Result<Vec<f32>> {
        Ok(self.scores.iter().copied().take(hypotheses.len()).collect())
    }
}

/// Test double that returns the i-th configured score vector on the i-th `entail` call — i.e. one
/// pre-set per-claim score vector PER CHUNK, in call order. Used by Task 4 to drive cross-chunk max.
#[cfg(test)]
pub struct SeqMock {
    per_call: std::cell::RefCell<std::collections::VecDeque<Vec<f32>>>,
}

#[cfg(test)]
impl SeqMock {
    pub fn new(per_call: Vec<Vec<f32>>) -> Self {
        Self { per_call: std::cell::RefCell::new(per_call.into()) }
    }
}

#[cfg(test)]
impl NliScorer for SeqMock {
    fn entail(&self, _premise: &str, _hypotheses: &[&str]) -> anyhow::Result<Vec<f32>> {
        Ok(self.per_call.borrow_mut().pop_front().unwrap_or_default())
    }
}

/// Gate-side NLI aggregation (spec 3): split the answer into claims, score each claim against each
/// cited chunk via the scorer, take the cross-chunk MAX per claim, keep only claims carrying a rare
/// token (AC's `is_rare` — `mean_filt`), and return the mean of survivors. `None` when nothing
/// survives, or on any scorer contract violation (non-finite / wrong length) — the caller then falls
/// back to the AC-only decision (spec 4). Never panics.
pub fn nli_score(
    answer: &str,
    chunk_texts: &[String],
    df: &DfTable,
    cfg: &VerifyConfig,
    scorer: &dyn NliScorer,
) -> Option<f32> {
    let claims = split_claims(answer);
    if claims.is_empty() || chunk_texts.is_empty() {
        return None;
    }
    let refs: Vec<&str> = claims.iter().map(|s| s.as_str()).collect();
    let mut per_claim_max = vec![f32::NEG_INFINITY; claims.len()];
    for chunk in chunk_texts {
        let scores = scorer.entail(chunk, &refs).ok()?;
        if scores.len() != claims.len() || scores.iter().any(|s| !s.is_finite()) {
            return None; // spec 2a.3 contract violation => fail-open, not Some(NaN)/panic
        }
        for (m, s) in per_claim_max.iter_mut().zip(scores) {
            if s > *m {
                *m = s;
            }
        }
    }
    // mean_filt: keep only claims carrying >=1 rare token.
    let kept: Vec<f32> = claims
        .iter()
        .zip(&per_claim_max)
        .filter(|(claim, _)| {
            crate::gate::token::tokenize(claim)
                .iter()
                .any(|t| df.is_rare(t, cfg.rare_df_frac))
        })
        .map(|(_, m)| *m)
        .collect();
    if kept.is_empty() {
        return None;
    }
    Some(kept.iter().sum::<f32>() / kept.len() as f32)
}

/// Deterministic, language-agnostic sentence segmentation (spec 3 step 1 / 2a.4). No dictionary, no
/// model. Quality feeds `mean_filt`; keep it deterministic and fixture-tested.
fn split_claims(answer: &str) -> Vec<String> {
    answer
        .split(['.', '!', '?', '\n'])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{nli_score, MockNli, NliScorer, SeqMock};
    use crate::gate::config::{VerifyConfig, VerifyMode};

    #[test]
    fn mock_returns_fixed_scores_per_hypothesis() {
        let m = MockNli::new(vec![0.9, 0.1]);
        let got = m.entail("premise chunk text", &["claim one", "claim two"]).unwrap();
        assert_eq!(got, vec![0.9, 0.1]);
    }

    #[test]
    fn seqmock_returns_next_vec_per_call() {
        let s = SeqMock::new(vec![vec![0.9, 0.2], vec![0.3, 0.8]]);
        assert_eq!(s.entail("a", &["x", "y"]).unwrap(), vec![0.9, 0.2]);
        assert_eq!(s.entail("b", &["x", "y"]).unwrap(), vec![0.3, 0.8]);
    }

    fn test_cfg() -> VerifyConfig {
        VerifyConfig {
            enabled: true,
            rare_df_frac: 0.03,
            min_answer_tokens: 10,
            threshold_single: None,
            threshold_multi: None,
            mode: VerifyMode::Nli,
            nli_threshold_single: Some(0.5),
            nli_threshold_multi: Some(0.5),
        }
    }

    #[test]
    fn mean_of_rare_token_claims_over_chunk_max() {
        // df empty ⇒ every token is rare (df==0). Two claims, two chunks.
        // chunk A scores [0.9,0.2], chunk B [0.3,0.8] ⇒ per-claim max [0.9,0.8] ⇒ mean 0.85.
        let df = crate::gate::df::DfTable::new();
        let cfg = test_cfg();
        let scorer = SeqMock::new(vec![vec![0.9, 0.2], vec![0.3, 0.8]]);
        let got = nli_score(
            "Widget zeta emits fault. Reset code kappa clears it.",
            &["chunk A".to_string(), "chunk B".to_string()],
            &df, &cfg, &scorer,
        );
        assert!((got.unwrap() - 0.85).abs() < 1e-5);
    }

    #[test]
    fn no_rare_token_claims_yields_none() {
        // All claim tokens are common (high df on every chunk) ⇒ zero survivors ⇒ None.
        let mut df = crate::gate::df::DfTable::new();
        for _ in 0..100 {
            df.add_chunk(&["the".to_string(), "and".to_string(), "for".to_string()]);
        }
        let cfg = test_cfg();
        let scorer = SeqMock::new(vec![vec![0.9]]);
        assert_eq!(nli_score("The and for.", &["c".to_string()], &df, &cfg, &scorer), None);
    }

    #[test]
    fn non_finite_or_wrong_length_scorer_output_is_none_not_panic() {
        let df = crate::gate::df::DfTable::new();
        let cfg = test_cfg();
        let nan = SeqMock::new(vec![vec![f32::NAN, 0.5]]);
        assert_eq!(nli_score("Alpha claim. Beta claim.", &["c".to_string()], &df, &cfg, &nan), None);
        let short = SeqMock::new(vec![vec![0.9]]); // 1 score for 2 claims
        assert_eq!(nli_score("Alpha claim. Beta claim.", &["c".to_string()], &df, &cfg, &short), None);
    }
}
