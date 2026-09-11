use crate::gate::{df::DfTable, token::tokenize};
use std::collections::HashSet;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Bucket { Single, Multi }
impl Bucket {
    pub fn of(n_chunks: usize) -> Bucket {
        if n_chunks <= 1 { Bucket::Single } else { Bucket::Multi }
    }
}

#[derive(Debug, Clone)]
pub struct GateScore {
    pub grounding: f32,
    pub rare_total: usize,
    pub rare_ungrounded: usize,
    pub ungrounded_tokens: Vec<String>,
    pub bucket: Bucket,
}

/// Fraction of the answer's rare tokens present in the union of chunk texts.
pub fn score(answer: &str, chunks: &[String], df: &DfTable, rare_df_frac: f32) -> GateScore {
    let chunk_tokens: HashSet<String> = chunks.iter().flat_map(|c| tokenize(c)).collect();
    let mut rare: Vec<String> = tokenize(answer).into_iter().filter(|t| df.is_rare(t, rare_df_frac)).collect();
    rare.sort(); rare.dedup();
    let rare_total = rare.len();
    let ungrounded: Vec<String> = rare.into_iter().filter(|t| !chunk_tokens.contains(t)).collect();
    let grounding = if rare_total == 0 { 1.0 } else { 1.0 - ungrounded.len() as f32 / rare_total as f32 };
    GateScore { grounding, rare_total, rare_ungrounded: ungrounded.len(), ungrounded_tokens: ungrounded, bucket: Bucket::of(chunks.len()) }
}

use crate::gate::config::{VerifyConfig, VerifyMode};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Decision { Serve, Abstain }

pub struct GateOutcome {
    pub decision: Decision,
    pub score: GateScore,
    pub threshold: Option<f32>,
    pub reason: String,
    pub nli: Option<f32>,
}

/// Serve/abstain decision for one answer: guards on `min_answer_tokens` first (an empty or
/// too-short answer never has enough rare tokens to trust a grounding ratio), then compares the
/// score's `grounding` against the bucket's calibrated threshold. Uncalibrated (`threshold` unset
/// for the answer's bucket) fails closed to `Abstain` — no threshold means no basis to serve.
pub fn decide(score: GateScore, cfg: &VerifyConfig, answer_tokens: usize) -> GateOutcome {
    if answer_tokens < cfg.min_answer_tokens {
        return GateOutcome { threshold: None, reason: "empty or too-short answer".into(), decision: Decision::Abstain, score, nli: None };
    }
    let threshold = cfg.threshold(score.bucket);
    match threshold {
        Some(t) if score.grounding > t => GateOutcome { decision: Decision::Serve, threshold: Some(t), reason: "grounded".into(), score, nli: None },
        Some(t) => {
            let reason = if score.ungrounded_tokens.is_empty() { "below threshold".into() }
                         else { format!("ungrounded specifics: {}", score.ungrounded_tokens.join(", ")) };
            GateOutcome { decision: Decision::Abstain, threshold: Some(t), reason, score, nli: None }
        }
        None => GateOutcome { decision: Decision::Abstain, threshold: None, reason: "uncalibrated".into(), score, nli: None },
    }
}

/// Mode-aware decision (spec 4): compute the AC-only outcome first, then combine with the optional
/// NLI score per `cfg.mode`. `nli == None` (off / not ready / no rare-token claim / scorer failure)
/// always falls back to the AC-only outcome — NLI never turns a would-be serve into abstain-all.
pub fn decide_modes(ac: GateScore, nli: Option<f32>, cfg: &VerifyConfig, answer_tokens: usize) -> GateOutcome {
    let base = decide(ac, cfg, answer_tokens); // existing AC-only outcome (thresholds, empty guard, etc.)
    let bucket = base.score.bucket;
    let ac_serves = matches!(base.decision, Decision::Serve);
    let nli_serves = |s: f32| cfg.nli_threshold(bucket).map(|t| s > t).unwrap_or(false);
    let (decision, reason) = match (cfg.mode, nli) {
        (VerifyMode::Ac, _) | (_, None) => (base.decision, base.reason.clone()),
        (VerifyMode::Nli, Some(s)) => {
            if nli_serves(s) { (Decision::Serve, "grounded".to_string()) }
            else { (Decision::Abstain, "unentailed".to_string()) }
        }
        (VerifyMode::Combined, Some(s)) => {
            if ac_serves && nli_serves(s) { (Decision::Serve, "grounded".to_string()) }
            else if ac_serves { (Decision::Abstain, "unentailed".to_string()) }
            else { (base.decision, base.reason.clone()) }
        }
    };
    GateOutcome { decision, reason, nli, ..base }
}

#[cfg(test)]
mod tests {
    use super::{score, Bucket};
    use crate::gate::{df::DfTable, token::tokenize};
    fn df_of(pages: &[&str]) -> DfTable {
        let mut t = DfTable::new();
        for p in pages { t.add_chunk(&tokenize(p)); }
        t
    }
    #[test]
    fn grounded_specifics_score_high_invented_low() {
        // corpus: the code appears on 1 of 4 pages ⇒ rare
        let df = df_of(&["pp.19.00.00.00 license", "general text", "more text", "configuration"]);
        let chunk = "this mode needs a pp.19.00.00.00 license".to_string();
        let grounded = score("needs a pp.19.00.00.00 license", std::slice::from_ref(&chunk), &df, 0.5);
        assert!(grounded.grounding > 0.9, "rare token present on chunk: {}", (grounded.grounding, &grounded.ungrounded_tokens).0);
        let invented = score("install gsd-driver step7", &[chunk], &df, 0.5);
        assert!(invented.grounding < 0.5, "invented codes absent from chunk");
        assert!(invented.ungrounded_tokens.iter().any(|t| t == "gsd-driver" || t == "step7"));
    }
    #[test]
    fn no_rare_tokens_scores_one() {
        let df = df_of(&["just a plain answer without codes", "another text"]);
        let s = score("just a plain answer without codes", &["any chunk".to_string()], &df, 0.03);
        assert_eq!(s.grounding, 1.0);
        assert_eq!(s.rare_total, 0);
    }
    #[test]
    fn bucket_by_chunk_count() {
        assert_eq!(Bucket::of(1), Bucket::Single);
        assert_eq!(Bucket::of(3), Bucket::Multi);
    }

    #[test]
    fn decide_guards_and_thresholds() {
        use super::{decide, score, Decision};
        use crate::gate::{
            config::{VerifyConfig, VerifyMode},
            df::DfTable,
            token::tokenize,
        };
        let cfg = VerifyConfig {
            enabled: true,
            rare_df_frac: 0.5,
            min_answer_tokens: 10,
            threshold_single: Some(0.8),
            threshold_multi: Some(0.9),
            mode: VerifyMode::Ac,
            nli_threshold_single: None,
            nli_threshold_multi: None,
        };
        let mut df = DfTable::new(); df.add_chunk(&tokenize("pp.19.00.00.00 text"));
        let answer = "needs a pp.19.00.00.00 license plus ten more english words to pass the guard threshold";
        // cited chunk echoes the answer's own words, so every rare token is grounded
        let s = score(answer, &[answer.to_string()], &df, 0.5);
        // long-enough, single chunk, grounding 1.0 > 0.8 ⇒ serve
        assert_eq!(decide(s, &cfg, 20).decision, Decision::Serve);
        let mut df2 = DfTable::new(); df2.add_chunk(&tokenize("nothing here"));
        let s2 = score("invented code zzz.999", &["nothing here".into()], &df2, 0.5);
        assert_eq!(decide(s2, &cfg, 20).decision, Decision::Abstain);       // ungrounded ⇒ below 0.8
        let s3 = score("short", &["x".into()], &df2, 0.5);
        assert_eq!(decide(s3, &cfg, 2).decision, Decision::Abstain);        // guard: < min_answer_tokens
    }

    /// A `GateScore` with the given AC grounding/bucket and zero rare-token counts (irrelevant to
    /// `decide_modes` once `grounding` and `bucket` are fixed — `reason` on the ungrounded-abstain
    /// path is the only other consumer, and these tests only exercise the serve/nli-abstain paths).
    fn gscore(grounding: f32, bucket: Bucket) -> GateScore {
        GateScore { grounding, bucket, rare_total: 0, rare_ungrounded: 0, ungrounded_tokens: vec![] }
    }

    /// A `VerifyConfig` for `decide_modes` mode tests: `min_answer_tokens: 1` so a 20-token test
    /// answer clears `decide`'s empty/too-short guard (`answer_tokens < min_answer_tokens`) and the
    /// AC-only outcome underneath reaches the threshold comparison rather than aborting early.
    fn cfg_for(mode: crate::gate::config::VerifyMode, ac_thr: f32, nli_thr: f32) -> crate::gate::config::VerifyConfig {
        crate::gate::config::VerifyConfig {
            enabled: true,
            rare_df_frac: 0.03,
            min_answer_tokens: 1,
            threshold_single: Some(ac_thr),
            threshold_multi: Some(ac_thr),
            mode,
            nli_threshold_single: Some(nli_thr),
            nli_threshold_multi: Some(nli_thr),
        }
    }

    #[test]
    fn combined_requires_both_signals() {
        use super::{decide_modes, Decision};
        use crate::gate::config::VerifyMode;
        let cfg = cfg_for(VerifyMode::Combined, /*ac*/ 0.3, /*nli*/ 0.5);
        let ac = gscore(0.9, Bucket::Single); // AC grounding above ac threshold ⇒ AC serves
        assert_eq!(decide_modes(ac.clone(), Some(0.8), &cfg, 20).decision, Decision::Serve);
        assert_eq!(decide_modes(ac.clone(), Some(0.4), &cfg, 20).decision, Decision::Abstain); // NLI fails
        assert_eq!(decide_modes(ac, None, &cfg, 20).decision, Decision::Serve); // None ⇒ AC-only serves
    }

    #[test]
    fn nli_mode_abstain_reason_is_unentailed() {
        use super::{decide_modes, Decision};
        use crate::gate::config::VerifyMode;
        let cfg = cfg_for(VerifyMode::Nli, 0.3, 0.5);
        let ac = gscore(0.9, Bucket::Single);
        let o = decide_modes(ac, Some(0.1), &cfg, 20);
        assert_eq!(o.decision, Decision::Abstain);
        assert_eq!(o.reason, "unentailed");
    }
}
