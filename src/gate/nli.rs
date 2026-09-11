//! The gate<->scorer seam (spec 2.1). `NliScorer` is a THIN entailment primitive: P(entail) of each
//! hypothesis (an answer claim) against a premise (a cited chunk). The scorer OWNS everything needing
//! the model tokenizer (windowing an over-length premise + max-pool over windows — Plan 2). The gate
//! owns claim-split, cross-chunk max, and DfTable mean_filt (Task 4) — all tokenizer-free.

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

#[cfg(test)]
mod tests {
    use super::{MockNli, NliScorer, SeqMock};

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
}
