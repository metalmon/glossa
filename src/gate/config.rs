//! `VerifyConfig`: resolved answer-grounding-gate tuning knobs (see `crate::gate`).
//! Precedence env > ontology > default, mirroring `graph::ppr::sim_weight`.

use std::path::Path;

use crate::gate::score::Bucket;
use crate::graph::ontology::Ontology;

pub struct VerifyConfig {
    pub enabled: bool,
    pub rare_df_frac: f32,
    pub min_answer_tokens: usize,
    pub threshold_single: Option<f32>,
    pub threshold_multi: Option<f32>,
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
            threshold_single: env_f32("GLOSSA_VERIFY_THRESHOLD_SINGLE")
                .or_else(|| og(Ontology::verify_threshold_single)),
            threshold_multi: env_f32("GLOSSA_VERIFY_THRESHOLD_MULTI")
                .or_else(|| og(Ontology::verify_threshold_multi)),
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
}

#[cfg(test)]
mod tests {
    use super::VerifyConfig;
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
}
