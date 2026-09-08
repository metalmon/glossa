//! On-disk checkpoint for GEPA train runs: enables crash-resilient resumption.
//!
//! Stores the candidate pool, best-so-far Pareto front, iteration state, and RNG position.
//! Atomic writes via temp-then-rename prevent partial corruption on crash.
//! Fingerprinting guards against config mismatches: a checkpoint is resumed only if its hash
//! of the run config matches the current run, or if --resume forces it anyway (caller validates
//! that intent is sound).

pub const CHECKPOINT_VERSION: u32 = 1;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq)]
pub struct GepaCheckpoint {
    pub version: u32,
    pub fingerprint: String,
    pub pool: Vec<crate::gepa_graph::Candidate>,
    pub best_pareto_so_far: f64,
    pub baseline_score: f64,
    pub val_ids: Vec<String>,
    pub pareto_ids: Vec<String>,
    pub blocked_train: Vec<String>,
    pub iter: usize,
    pub metric_calls: usize,
    pub rng_seed: [u8; 32],
    pub rng_word_pos: u128,
    pub final_val: Option<Vec<Option<f64>>>,
}

pub enum ResumeDecision {
    Fresh,
    Resume(Box<GepaCheckpoint>),
}

#[allow(clippy::too_many_arguments)]
pub fn fingerprint(
    seed_prompt: &str,
    model: &str,
    endpoint: &str,
    max_metric_calls: usize,
    max_candidates: usize,
    minibatch: usize,
    pareto_size: usize,
    val_frac: f64,
    seed: u64,
    rollout_samples: usize,
    judge: bool,
    credit_abstention: bool,
    fp_gate: bool,
    question_ids_sorted: &[String],
) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    seed_prompt.hash(&mut h);
    model.hash(&mut h);
    endpoint.hash(&mut h);
    max_metric_calls.hash(&mut h);
    max_candidates.hash(&mut h);
    minibatch.hash(&mut h);
    pareto_size.hash(&mut h);
    val_frac.to_bits().hash(&mut h);
    seed.hash(&mut h);
    rollout_samples.hash(&mut h);
    judge.hash(&mut h);
    credit_abstention.hash(&mut h);
    fp_gate.hash(&mut h);
    question_ids_sorted.len().hash(&mut h);
    for id in question_ids_sorted {
        id.hash(&mut h);
    }
    format!("{:016x}", h.finish())
}

fn tmp_path(path: &std::path::Path) -> std::path::PathBuf {
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "gepa.checkpoint.json".to_string());
    path.with_file_name(format!("{name}.tmp"))
}

pub fn save(path: &std::path::Path, ckpt: &GepaCheckpoint) -> anyhow::Result<()> {
    use anyhow::Context;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("create {}", dir.display()))?;
    }
    let tmp = tmp_path(path);
    let bytes =
        serde_json::to_vec_pretty(ckpt).context("serialize checkpoint")?;
    std::fs::write(&tmp, &bytes)
        .with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| {
        format!("rename {} -> {}", tmp.display(), path.display())
    })?;
    Ok(())
}

pub fn load(path: &std::path::Path) -> anyhow::Result<Option<GepaCheckpoint>> {
    use anyhow::Context;
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(path)
        .with_context(|| format!("read {}", path.display()))?;
    let ckpt: GepaCheckpoint = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse {}", path.display()))?;
    Ok(Some(ckpt))
}

pub fn delete(path: &std::path::Path) -> anyhow::Result<()> {
    use anyhow::Context;
    if path.exists() {
        std::fs::remove_file(path)
            .with_context(|| format!("remove {}", path.display()))?;
    }
    Ok(())
}

pub fn decide_resume(
    on_disk: Option<GepaCheckpoint>,
    current_fp: &str,
    resume: bool,
    force: bool,
) -> anyhow::Result<ResumeDecision> {
    if force {
        return Ok(ResumeDecision::Fresh);
    }
    if resume {
        return match on_disk {
            Some(c) => Ok(ResumeDecision::Resume(Box::new(c))),
            None => anyhow::bail!(
                "--resume: no checkpoint to resume (nothing on disk)"
            ),
        };
    }
    match on_disk {
        None => Ok(ResumeDecision::Fresh),
        Some(c) if c.fingerprint == current_fp => {
            Ok(ResumeDecision::Resume(Box::new(c)))
        }
        Some(c) => anyhow::bail!(
            "GEPA checkpoint on disk was written for a different run \
             (fingerprint {} != current {}). Pass --force to discard it and start fresh, \
             or --resume to resume it anyway.",
            c.fingerprint, current_fp
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gepa_graph::Candidate;

    fn sample_ckpt() -> GepaCheckpoint {
        GepaCheckpoint {
            version: CHECKPOINT_VERSION,
            fingerprint: "deadbeefdeadbeef".to_string(),
            pool: vec![
                Candidate {
                    prompt: "seed".into(),
                    score_val: vec![1.0, 0.0, 0.5],
                },
                Candidate {
                    prompt: "child".into(),
                    score_val: vec![1.0, 1.0, 0.5],
                },
            ],
            best_pareto_so_far: 0.833,
            baseline_score: 0.5,
            val_ids: vec!["a".into(), "b".into(), "c".into()],
            pareto_ids: vec!["a".into(), "c".into()],
            blocked_train: vec!["z".into()],
            iter: 7,
            metric_calls: 123,
            rng_seed: [9u8; 32],
            rng_word_pos: 340_282_366_920_938_463_463u128,
            final_val: Some(vec![Some(0.6), None]),
        }
    }

    #[test]
    fn checkpoint_roundtrip_preserves_all_fields() {
        let c = sample_ckpt();
        let json = serde_json::to_vec_pretty(&c).unwrap();
        let back: GepaCheckpoint = serde_json::from_slice(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn atomic_save_leaves_no_tmp_and_prior_intact() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gepa.checkpoint.json");
        save(&path, &sample_ckpt()).unwrap();
        assert!(path.exists());
        assert!(!tmp_path(&path).exists(), "no .tmp left after a clean save");
        // A second save overwrites atomically and still parses.
        let mut c2 = sample_ckpt();
        c2.iter = 8;
        save(&path, &c2).unwrap();
        assert_eq!(load(&path).unwrap().unwrap().iter, 8);
    }

    #[test]
    fn load_missing_is_none_and_delete_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gepa.checkpoint.json");
        assert!(load(&path).unwrap().is_none());
        delete(&path).unwrap(); // no error on absent file
        save(&path, &sample_ckpt()).unwrap();
        delete(&path).unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn fingerprint_changes_on_seed_prompt_and_is_order_stable() {
        let ids = vec!["a".to_string(), "b".to_string()];
        let base = fingerprint(
            "seed", "m", "e", 100, 6, 3, 12, 0.2, 1, 1, true, false, false,
            &ids,
        );
        let diff = fingerprint(
            "SEED2", "m", "e", 100, 6, 3, 12, 0.2, 1, 1, true, false, false,
            &ids,
        );
        assert_ne!(base, diff);
        // Same SORTED ids in the same order => identical (caller sorts before hashing).
        let same = fingerprint(
            "seed", "m", "e", 100, 6, 3, 12, 0.2, 1, 1, true, false, false,
            &ids,
        );
        assert_eq!(base, same);
    }

    #[test]
    fn decide_resume_matrix() {
        let mut c = sample_ckpt();
        c.fingerprint = "FP".to_string();
        // force => Fresh regardless of disk
        assert!(matches!(
            decide_resume(Some(c.clone()), "FP", false, true).unwrap(),
            ResumeDecision::Fresh
        ));
        // auto + match => Resume
        assert!(matches!(
            decide_resume(Some(c.clone()), "FP", false, false).unwrap(),
            ResumeDecision::Resume(_)
        ));
        // auto + mismatch => error
        assert!(decide_resume(Some(c.clone()), "OTHER", false, false).is_err());
        // auto + none => Fresh
        assert!(matches!(
            decide_resume(None, "FP", false, false).unwrap(),
            ResumeDecision::Fresh
        ));
        // --resume + mismatch => Resume anyway
        assert!(matches!(
            decide_resume(Some(c.clone()), "OTHER", true, false).unwrap(),
            ResumeDecision::Resume(_)
        ));
        // --resume + none => error
        assert!(decide_resume(None, "FP", true, false).is_err());
    }
}
