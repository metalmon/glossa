//! `VerifyConfig`: resolved answer-grounding-gate tuning knobs (see `crate::gate`).
//! Precedence env > ontology > default, mirroring `graph::ppr::sim_weight`.

use std::path::Path;

use crate::gate::score::Bucket;
use crate::graph::ontology::Ontology;

/// How the AC (lexical/anomaly) and NLI (entailment) verdicts combine into a gate decision.
/// Default `Ac` preserves today's behaviour exactly: a deployment that sets nothing is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VerifyMode {
    #[default]
    Ac,
    Nli,
    Combined,
}

impl VerifyMode {
    /// Parse from the `[verify] mode` string; anything unrecognised is the safe default `Ac`.
    pub fn parse(s: &str) -> VerifyMode {
        match s {
            "nli" => VerifyMode::Nli,
            "combined" => VerifyMode::Combined,
            _ => VerifyMode::Ac,
        }
    }
}

pub struct VerifyConfig {
    pub enabled: bool,
    pub rare_df_frac: f32,
    pub min_answer_tokens: usize,
    pub threshold_single: Option<f32>,
    pub threshold_multi: Option<f32>,
    pub mode: VerifyMode,
    pub nli_threshold_single: Option<f32>,
    pub nli_threshold_multi: Option<f32>,
    pub combined_single: Option<CombinedStats>,
    pub combined_multi: Option<CombinedStats>,
}

/// Calibrated z-score consensus stats for `combined` mode (spec §4 rev.5): AC and NLI are each
/// standardized by their own calibrated per-bucket mean/std, summed, and compared against a
/// calibrated z-threshold. Written by calibration (Task CZ-2); this side only reads and applies
/// them. `threshold` is a z-score, not a raw [0,1] score, so it may legitimately be negative.
#[derive(Clone, Copy, Debug)]
pub struct CombinedStats {
    pub mean_ac: f32,
    pub std_ac: f32,
    pub mean_nli: f32,
    pub std_nli: f32,
    pub threshold: f32,
}

impl CombinedStats {
    /// Sum of the two standardized scores. A zero calibrated std means that signal never varied
    /// during calibration (or wasn't observed) — its z-contribution is defined as 0 rather than
    /// dividing by zero, so it neither serves nor blocks on its own.
    pub fn z(&self, ac: f32, nli: f32) -> f32 {
        let zac = if self.std_ac == 0.0 {
            0.0
        } else {
            (ac - self.mean_ac) / self.std_ac
        };
        let znli = if self.std_nli == 0.0 {
            0.0
        } else {
            (nli - self.mean_nli) / self.std_nli
        };
        zac + znli
    }

    pub fn serves(&self, ac: f32, nli: f32) -> bool {
        self.z(ac, nli) > self.threshold
    }
}

fn env_f32(key: &str) -> Option<f32> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}
fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}
fn env_bool(key: &str) -> Option<bool> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}
fn env_string(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

impl VerifyConfig {
    /// `glossa_dir` is the corpus `.glossa` dir; ontology loads from its parent.
    pub fn resolve(glossa_dir: &Path) -> VerifyConfig {
        let ont = glossa_dir.parent().map(Ontology::load_or_default);
        let og = |f: fn(&Ontology) -> Option<f32>| ont.as_ref().and_then(f);
        VerifyConfig {
            enabled: env_bool("GLOSSA_VERIFY_ENABLED")
                .or_else(|| ont.as_ref().and_then(|o| o.verify_enabled()))
                .unwrap_or(false),
            rare_df_frac: env_f32("GLOSSA_VERIFY_RARE_DF_FRAC")
                .or_else(|| og(Ontology::verify_rare_df_frac))
                .unwrap_or(0.03),
            min_answer_tokens: env_usize("GLOSSA_VERIFY_MIN_ANSWER_TOKENS")
                .or_else(|| ont.as_ref().and_then(|o| o.verify_min_answer_tokens()))
                .unwrap_or(10),
            mode: env_string("GLOSSA_VERIFY_MODE")
                .map(|s| VerifyMode::parse(&s))
                .or_else(|| {
                    ont.as_ref()
                        .and_then(|o| o.verify_mode())
                        .map(|s| VerifyMode::parse(&s))
                })
                .unwrap_or_default(),
            // AC thresholds: env, else ontology verify.ac.threshold, else legacy ontology
            // verify.threshold.
            threshold_single: env_f32("GLOSSA_VERIFY_THRESHOLD_SINGLE")
                .or_else(|| og(Ontology::verify_ac_threshold_single))
                .or_else(|| og(Ontology::verify_threshold_single)),
            threshold_multi: env_f32("GLOSSA_VERIFY_THRESHOLD_MULTI")
                .or_else(|| og(Ontology::verify_ac_threshold_multi))
                .or_else(|| og(Ontology::verify_threshold_multi)),
            // NLI thresholds: env, else ontology verify.nli.threshold.
            nli_threshold_single: env_f32("GLOSSA_VERIFY_NLI_THRESHOLD_SINGLE")
                .or_else(|| og(Ontology::verify_nli_threshold_single)),
            nli_threshold_multi: env_f32("GLOSSA_VERIFY_NLI_THRESHOLD_MULTI")
                .or_else(|| og(Ontology::verify_nli_threshold_multi)),
            // Combined z-score consensus stats: ontology ONLY, no env override — these are
            // calibrated (Task CZ-2's output), not a knob a deployment hand-sets.
            combined_single: ont.as_ref().and_then(|o| o.verify_combined_single()),
            combined_multi: ont.as_ref().and_then(|o| o.verify_combined_multi()),
        }
    }

    pub fn threshold(&self, b: Bucket) -> Option<f32> {
        match b {
            Bucket::Single => self.threshold_single,
            Bucket::Multi => self.threshold_multi,
        }
    }

    pub fn is_calibrated(&self) -> bool {
        self.threshold_single.is_some() && self.threshold_multi.is_some()
    }

    pub fn nli_threshold(&self, b: Bucket) -> Option<f32> {
        match b {
            Bucket::Single => self.nli_threshold_single,
            Bucket::Multi => self.nli_threshold_multi,
        }
    }

    /// NLI-side readiness (mirror of AC's `enabled && is_calibrated`, spec 2.5): consult NLI only
    /// when the mode asks for it AND both nli thresholds are set. `mode == Ac` is never ready.
    pub fn is_nli_ready(&self) -> bool {
        self.mode != VerifyMode::Ac
            && self.nli_threshold_single.is_some()
            && self.nli_threshold_multi.is_some()
    }

    /// Calibrated combined-mode stats for the given bucket, or `None` when uncalibrated.
    pub fn combined_stats(&self, b: Bucket) -> Option<CombinedStats> {
        match b {
            Bucket::Single => self.combined_single,
            Bucket::Multi => self.combined_multi,
        }
    }

    /// Combined-mode readiness: the mode asks for it AND both buckets are calibrated. Mirrors
    /// `is_nli_ready`'s shape; `decide_modes` also fails open per-call via `combined_stats(..)`
    /// being `None`, so this is for callers (e.g. `gate::verify_outcome_with_scorer`) that need a
    /// single up-front readiness check.
    pub fn is_combined_ready(&self) -> bool {
        self.mode == VerifyMode::Combined
            && self.combined_single.is_some()
            && self.combined_multi.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::{VerifyConfig, VerifyMode};
    use crate::gate::score::Bucket;

    #[test]
    fn defaults_when_no_ontology() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_VERIFY_RARE_DF_FRAC");
        let dir = tempfile::tempdir().unwrap(); // no .glossa/ontology.toml
        let c = VerifyConfig::resolve(dir.path());
        assert!(!c.enabled);
        assert!((c.rare_df_frac - 0.03).abs() < 1e-6);
        assert_eq!(c.min_answer_tokens, 10);
        assert!(!c.is_calibrated());
        assert_eq!(c.threshold(Bucket::Single), None);
        assert!(matches!(c.mode, VerifyMode::Ac));
        assert!(!c.is_nli_ready());
    }

    #[test]
    fn env_overrides_default() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::set_var("GLOSSA_VERIFY_RARE_DF_FRAC", "0.10");
        let dir = tempfile::tempdir().unwrap();
        let c = VerifyConfig::resolve(dir.path());
        assert!((c.rare_df_frac - 0.10).abs() < 1e-6);
        std::env::remove_var("GLOSSA_VERIFY_RARE_DF_FRAC");
    }

    #[test]
    fn mode_defaults_to_ac_and_ac_alias_precedence() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_VERIFY_MODE");
        std::env::remove_var("GLOSSA_VERIFY_THRESHOLD_SINGLE");
        std::env::remove_var("GLOSSA_VERIFY_THRESHOLD_MULTI");
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "[verify]\nenabled=true\n[verify.threshold]\nsingle=0.8\nmulti=0.9\n",
        )
        .unwrap();
        let c = VerifyConfig::resolve(&g);
        assert!(matches!(c.mode, VerifyMode::Ac));
        assert_eq!(c.threshold(Bucket::Single), Some(0.8)); // legacy alias honored when ac.threshold absent
        assert!(!c.is_nli_ready());
    }

    #[test]
    fn nli_ready_requires_mode_and_nli_threshold() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_VERIFY_MODE");
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "[verify]\nenabled=true\nmode=\"nli\"\n\
             [verify.ac.threshold]\nsingle=0.1\nmulti=0.2\n\
             [verify.nli.threshold]\nsingle=0.5\nmulti=0.6\n",
        )
        .unwrap();
        let c = VerifyConfig::resolve(&g);
        assert!(matches!(c.mode, VerifyMode::Nli));
        assert_eq!(c.threshold(Bucket::Single), Some(0.1)); // ac.threshold wins over legacy
        assert_eq!(c.nli_threshold(Bucket::Single), Some(0.5));
        assert!(c.is_nli_ready());
    }

    #[test]
    fn ac_threshold_wins_over_legacy_when_both_present() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_VERIFY_MODE");
        std::env::remove_var("GLOSSA_VERIFY_THRESHOLD_SINGLE");
        std::env::remove_var("GLOSSA_VERIFY_THRESHOLD_MULTI");
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "[verify]\nenabled=true\n\
             [verify.ac.threshold]\nsingle=0.1\nmulti=0.2\n\
             [verify.threshold]\nsingle=0.8\nmulti=0.9\n",
        )
        .unwrap();
        let c = VerifyConfig::resolve(&g);
        assert_eq!(c.threshold(Bucket::Single), Some(0.1)); // ac.threshold wins, NOT legacy 0.8
        assert_eq!(c.threshold(Bucket::Multi), Some(0.2)); // ac.threshold wins, NOT legacy 0.9
    }

    #[test]
    fn combined_mode_round_trips_from_ontology() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_VERIFY_MODE");
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "[verify]\nenabled=true\nmode=\"combined\"\n",
        )
        .unwrap();
        assert!(matches!(
            VerifyConfig::resolve(&g).mode,
            VerifyMode::Combined
        ));
    }

    #[test]
    fn combined_stats_round_trip_from_ontology() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_VERIFY_MODE");
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "[verify]\nenabled=true\nmode=\"combined\"\n\
             [verify.combined.single]\nmean_ac=0.5\nstd_ac=0.1\nmean_nli=0.6\nstd_nli=0.2\nthreshold=0.1\n\
             [verify.combined.multi]\nmean_ac=0.4\nstd_ac=0.15\nmean_nli=0.55\nstd_nli=0.25\nthreshold=-0.2\n",
        )
        .unwrap();
        let c = VerifyConfig::resolve(&g);
        assert!(c.is_combined_ready());
        let single = c.combined_stats(Bucket::Single).unwrap();
        assert!((single.mean_ac - 0.5).abs() < 1e-6);
        assert!((single.std_nli - 0.2).abs() < 1e-6);
        assert!((single.threshold - 0.1).abs() < 1e-6);
        let multi = c.combined_stats(Bucket::Multi).unwrap();
        assert!((multi.mean_nli - 0.55).abs() < 1e-6);
        assert!((multi.threshold - (-0.2)).abs() < 1e-6);
    }
}
