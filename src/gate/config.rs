//! `VerifyConfig`: resolved answer-grounding-gate tuning knobs (see `crate::gate`).
//! Precedence env > ontology > default, mirroring `graph::ppr::sim_weight`.

use std::path::{Path, PathBuf};

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
    /// Runtime NLI scorer selection from `[verify.nli]` (Plan 2 Task 3): `"in_process"` (built) |
    /// `"http"` (Task 4, not built yet). `None` ⇒ `gate::resolve_scorer` returns `None` (AC-only).
    pub scorer: Option<String>,
    /// Filesystem path to the exported NLI model directory (in-process scorer only). `None` ⇒
    /// `resolve_scorer` cannot build an `InProcessNli` and fails open to `None`.
    pub model_dir: Option<PathBuf>,
    /// Softmax index of the entailment class; `0` is the `cointegrated/rubert-base-cased-nli-threeway`
    /// convention (`id2label[0]=entailment`); Task 6 confirms against the real export.
    pub entail_index: usize,
    /// Ordered execution-provider preference list (Plan 3 Task 1), e.g. `["cuda", "cpu"]`. Consumed
    /// by a later task's EP registration — this task only resolves and normalizes it. Always
    /// non-empty after `resolve`: an unset/empty/all-unknown list defaults to `["cpu"]`, which the
    /// runtime treats exactly as today (CPU-only, no behavior change).
    pub execution_providers: Vec<String>,
    /// Which GPU (device id) the CUDA/DirectML/ROCm EP binds to. `None` (unset) ⇒ each EP uses its
    /// own default (device 0), i.e. exactly today's behavior. Sourced from env
    /// `GLOSSA_VERIFY_NLI_EP_DEVICE` (i32; non-integer ignored with a warning) else ontology
    /// `[verify.nli].ep_device`. CoreML ignores it (no device-id concept).
    pub execution_provider_device: Option<i32>,
    /// GPU arena memory cap in megabytes for the NLI EP. `None` (unset) ⇒ no memory options set, i.e.
    /// exactly today's behavior. `Some(mb)` caps CUDA's arena (`with_memory_limit`, converted to
    /// bytes) + same-as-requested arena growth + disables the session memory-pattern optimizer, so
    /// NLI can share a GPU with an LLM. ROCm gets only the arena-growth change; DirectML/CoreML
    /// expose no memory option in this ort version and ignore it. Sourced from env
    /// `GLOSSA_VERIFY_NLI_EP_MEM_LIMIT_MB` else ontology `[verify.nli].ep_mem_limit_mb`.
    pub execution_provider_mem_limit_mb: Option<usize>,
}

/// Execution providers the runtime actually knows how to register — exactly the EPs `glossa-nli`
/// has a Cargo feature + dispatch arm for (`nli-cuda`, `nli-directml`, `nli-coreml`, `nli-rocm`)
/// plus the implicit `"cpu"` fallback. Anything else is dropped with a warning rather than
/// erroring — fail-open, since a bad/typo'd EP name should never block the gate from resolving.
const KNOWN_EXECUTION_PROVIDERS: &[&str] = &["cpu", "cuda", "directml", "coreml", "rocm"];

/// Lowercase + trim each entry, drop anything outside [`KNOWN_EXECUTION_PROVIDERS`] (warning, not
/// error), and default to `["cpu"]` when the result is empty — whether because the input was empty
/// or because every entry was unknown.
fn normalize_eps(raw: impl Iterator<Item = String>) -> Vec<String> {
    let normalized: Vec<String> = raw
        .map(|s| s.trim().to_lowercase())
        .filter(|s| {
            let known = KNOWN_EXECUTION_PROVIDERS.contains(&s.as_str());
            if !known {
                tracing::warn!(execution_provider = %s, "unknown [verify.nli].execution_providers entry dropped");
            }
            known
        })
        .collect();
    if normalized.is_empty() {
        vec!["cpu".to_string()]
    } else {
        normalized
    }
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
/// Parse an env var as `i32`. Absent/empty ⇒ `None`; a set-but-non-integer value is ignored with a
/// warning (fail-open — a typo'd device id should never take the gate down, it just falls back to
/// the default device).
fn env_i32(key: &str) -> Option<i32> {
    let raw = std::env::var(key).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.parse::<i32>() {
        Ok(n) => Some(n),
        Err(_) => {
            tracing::warn!(key, value = %raw, "non-integer NLI EP device id ignored");
            None
        }
    }
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
            // Runtime NLI scorer selection: env, else ontology verify.nli.{scorer,model_dir,
            // entail_index}. Unset scorer/model_dir ⇒ `gate::resolve_scorer` returns `None`
            // (AC-only, fail-open) — see that function's doc comment.
            scorer: env_string("GLOSSA_VERIFY_NLI_SCORER").or_else(|| {
                ont.as_ref()
                    .and_then(|o| o.verify_nli_scorer())
                    .map(str::to_string)
            }),
            model_dir: env_string("GLOSSA_VERIFY_NLI_MODEL_DIR")
                .or_else(|| {
                    ont.as_ref()
                        .and_then(|o| o.verify_nli_model_dir())
                        .map(str::to_string)
                })
                .map(PathBuf::from),
            entail_index: env_usize("GLOSSA_VERIFY_NLI_ENTAIL_INDEX")
                .or_else(|| ont.as_ref().and_then(|o| o.verify_nli_entail_index()))
                .unwrap_or(0),
            // Ordered EP preference list: env (comma-separated) wins wholesale over ontology, else
            // ontology's list, else empty — normalize_eps then defaults empty/all-unknown to
            // ["cpu"], so an unset deployment resolves to today's CPU-only behavior unchanged.
            execution_providers: normalize_eps(
                match env_string("GLOSSA_NLI_EP") {
                    Some(v) => v.split(',').map(str::to_string).collect::<Vec<_>>(),
                    None => ont
                        .as_ref()
                        .map(|o| o.verify_nli_execution_providers().to_vec())
                        .unwrap_or_default(),
                }
                .into_iter(),
            ),
            // GPU device id: env wins, else ontology `[verify.nli].ep_device`, else None (default
            // device — today's behavior). Non-integer env value is dropped by `env_i32`.
            execution_provider_device: env_i32("GLOSSA_VERIFY_NLI_EP_DEVICE")
                .or_else(|| ont.as_ref().and_then(|o| o.verify_nli_ep_device())),
            // GPU memory cap (MB): env wins, else ontology `[verify.nli].ep_mem_limit_mb`, else None
            // (no memory options — today's behavior). Non-integer env value is dropped by env_usize.
            execution_provider_mem_limit_mb: env_usize("GLOSSA_VERIFY_NLI_EP_MEM_LIMIT_MB")
                .or_else(|| ont.as_ref().and_then(|o| o.verify_nli_ep_mem_limit_mb())),
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

    #[test]
    fn nli_scorer_config_round_trips_from_ontology() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_VERIFY_NLI_SCORER");
        std::env::remove_var("GLOSSA_VERIFY_NLI_MODEL_DIR");
        std::env::remove_var("GLOSSA_VERIFY_NLI_ENTAIL_INDEX");
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "[verify]\nenabled=true\n\
             [verify.nli]\nscorer=\"in_process\"\nmodel_dir=\"/x\"\nentail_index=2\n",
        )
        .unwrap();
        let c = VerifyConfig::resolve(&g);
        assert_eq!(c.scorer.as_deref(), Some("in_process"));
        assert_eq!(c.model_dir, Some(std::path::PathBuf::from("/x")));
        assert_eq!(c.entail_index, 2);
    }

    #[test]
    fn nli_scorer_config_defaults_when_absent() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_VERIFY_NLI_SCORER");
        std::env::remove_var("GLOSSA_VERIFY_NLI_MODEL_DIR");
        std::env::remove_var("GLOSSA_VERIFY_NLI_ENTAIL_INDEX");
        let dir = tempfile::tempdir().unwrap(); // no .glossa/ontology.toml
        let c = VerifyConfig::resolve(dir.path());
        assert_eq!(c.scorer, None);
        assert_eq!(c.model_dir, None);
        assert_eq!(c.entail_index, 0);
    }

    #[test]
    fn execution_providers_round_trips_from_ontology() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_NLI_EP");
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "[verify]\nenabled=true\n\
             [verify.nli]\nexecution_providers=[\"cuda\",\"cpu\"]\n",
        )
        .unwrap();
        let c = VerifyConfig::resolve(&g);
        assert_eq!(
            c.execution_providers,
            vec!["cuda".to_string(), "cpu".to_string()]
        );
    }

    #[test]
    fn ep_device_round_trips_from_ontology() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_VERIFY_NLI_EP_DEVICE");
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "[verify]\nenabled=true\n[verify.nli]\nep_device=1\n",
        )
        .unwrap();
        let c = VerifyConfig::resolve(&g);
        assert_eq!(c.execution_provider_device, Some(1));
    }

    #[test]
    fn ep_device_none_when_absent() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_VERIFY_NLI_EP_DEVICE");
        let dir = tempfile::tempdir().unwrap(); // no .glossa/ontology.toml
        let c = VerifyConfig::resolve(dir.path());
        assert_eq!(c.execution_provider_device, None);
    }

    #[test]
    fn ep_mem_limit_mb_round_trips_from_ontology() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_VERIFY_NLI_EP_MEM_LIMIT_MB");
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "[verify]\nenabled=true\n[verify.nli]\nep_mem_limit_mb=512\n",
        )
        .unwrap();
        let c = VerifyConfig::resolve(&g);
        assert_eq!(c.execution_provider_mem_limit_mb, Some(512));
    }

    #[test]
    fn ep_mem_limit_mb_none_when_absent() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_VERIFY_NLI_EP_MEM_LIMIT_MB");
        let dir = tempfile::tempdir().unwrap(); // no .glossa/ontology.toml
        let c = VerifyConfig::resolve(dir.path());
        assert_eq!(c.execution_provider_mem_limit_mb, None);
    }

    #[test]
    fn execution_providers_defaults_to_cpu_when_absent() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_NLI_EP");
        let dir = tempfile::tempdir().unwrap(); // no .glossa/ontology.toml
        let c = VerifyConfig::resolve(dir.path());
        assert_eq!(c.execution_providers, vec!["cpu".to_string()]);
    }

    #[test]
    fn execution_providers_env_overrides_ontology() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "[verify]\nenabled=true\n[verify.nli]\nexecution_providers=[\"cuda\"]\n",
        )
        .unwrap();
        std::env::set_var("GLOSSA_NLI_EP", "directml,cpu");
        let c = VerifyConfig::resolve(&g);
        std::env::remove_var("GLOSSA_NLI_EP");
        assert_eq!(
            c.execution_providers,
            vec!["directml".to_string(), "cpu".to_string()]
        );
    }

    #[test]
    fn execution_providers_unknown_entries_dropped() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_NLI_EP");
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "[verify]\nenabled=true\n[verify.nli]\nexecution_providers=[\"bogus\",\"cpu\"]\n",
        )
        .unwrap();
        let c = VerifyConfig::resolve(&g);
        assert_eq!(c.execution_providers, vec!["cpu".to_string()]);
    }

    #[test]
    fn execution_providers_all_unknown_falls_back_to_cpu() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_NLI_EP");
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "[verify]\nenabled=true\n[verify.nli]\nexecution_providers=[\"bogus\"]\n",
        )
        .unwrap();
        let c = VerifyConfig::resolve(&g);
        assert_eq!(c.execution_providers, vec!["cpu".to_string()]);
    }

    #[test]
    fn execution_providers_mixed_case_normalized_lowercase() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_NLI_EP");
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "[verify]\nenabled=true\n[verify.nli]\nexecution_providers=[\"CUDA\"]\n",
        )
        .unwrap();
        let c = VerifyConfig::resolve(&g);
        assert_eq!(c.execution_providers, vec!["cuda".to_string()]);
    }
}
