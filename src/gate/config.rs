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
}

#[cfg(test)]
mod tests {
    use super::{VerifyConfig, VerifyMode};
    use crate::gate::score::Bucket;

    #[test]
    fn defaults_when_no_ontology() {
        let _env = crate::TEST_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        let _env = crate::TEST_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("GLOSSA_VERIFY_RARE_DF_FRAC", "0.10");
        let dir = tempfile::tempdir().unwrap();
        let c = VerifyConfig::resolve(dir.path());
        assert!((c.rare_df_frac - 0.10).abs() < 1e-6);
        std::env::remove_var("GLOSSA_VERIFY_RARE_DF_FRAC");
    }

    #[test]
    fn mode_defaults_to_ac_and_ac_alias_precedence() {
        let _env = crate::TEST_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        let _env = crate::TEST_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
}
