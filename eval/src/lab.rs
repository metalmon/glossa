//! `lab.toml` — the file-first `kbx` eval toolkit's endpoint config (model/judge/reflect/distil
//! endpoints, default filenames, timeout). The corpus is NOT configured here: it comes from
//! kb-style PATH resolution (`glossa::root::resolve_root` via `kb_eval::workspace::resolve`),
//! exactly like `kb` itself.

use std::path::Path;

fn d120() -> u64 {
    120
}

/// Which wire-format chat API an [`Endpoint`] speaks. Defaults to `OpenAiChat` for back-compat
/// with existing `lab.toml` files that predate this field. Each variant has its own
/// `ChatTransport` impl (see `backend::transport::transport_for`): `OpenAiChat` ->
/// `transport::openai::OpenAiTransport`, `Anthropic` -> `transport::anthropic::AnthropicTransport`,
/// `OpenAiResponses` -> `transport::responses::ResponsesTransport`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiKind {
    /// OpenAI-compatible `/v1/chat/completions`. Accepts both `"openai"` (back-compat alias) and
    /// `"openai_chat"` in `lab.toml`.
    #[default]
    #[serde(rename = "openai", alias = "openai_chat")]
    OpenAiChat,
    /// Anthropic Messages API (`/v1/messages`).
    #[serde(rename = "anthropic")]
    Anthropic,
    /// OpenAI Responses API (`/v1/responses`).
    #[serde(rename = "openai_responses")]
    OpenAiResponses,
}

#[derive(serde::Deserialize, Clone)]
pub struct Endpoint {
    pub endpoint: String,
    pub model: String,
    /// Inline literal API key. Prefer `api_key_env` to keep secrets out of the file.
    #[serde(default)]
    pub api_key: String,
    /// Name of an env var to read the API key from (optional).
    #[serde(default)]
    pub api_key_env: String,
    /// Per-endpoint request timeout, in seconds.
    #[serde(default = "d120")]
    pub timeout_secs: u64,
    /// Which chat API this endpoint speaks. Defaults to `OpenAiChat` when absent, so existing
    /// `lab.toml` files parse unchanged.
    #[serde(default)]
    pub api: ApiKind,
}

impl Endpoint {
    /// Resolve the API key for this endpoint: an inline `api_key` wins; otherwise
    /// fall back to reading `api_key_env` from the environment; otherwise `None`.
    pub fn resolve_key(&self) -> Option<String> {
        if !self.api_key.is_empty() {
            return Some(self.api_key.clone());
        }
        if !self.api_key_env.is_empty() {
            if let Ok(v) = std::env::var(&self.api_key_env) {
                return Some(v);
            }
        }
        None
    }
}

#[derive(serde::Deserialize, Clone)]
pub struct LabConfig {
    pub model: Endpoint,
    #[serde(default)]
    pub judge: Option<Endpoint>,
    /// Harvest endpoint for `kbx build`'s extract stage. Its own model, separate from the eval
    /// reader's `[model]`. Falls back to `[model]` when unset (see `build_endpoint`), so existing
    /// `lab.toml` files without a `[build]` section keep today's behavior exactly.
    #[serde(default)]
    pub build: Option<Endpoint>,
    /// Endpoint used to reflect on/rewrite prompts (`kbx train`, a later plan).
    #[serde(default)]
    pub reflect: Option<Endpoint>,
    /// Strong-model endpoint for `kbx distil` (graph densification). Also the fallback for
    /// `kbx reason` when `[reason]` is unset.
    #[serde(default)]
    pub distil: Option<Endpoint>,
    /// Phase-2 (`kbx reason`, seed-from-grounded backward synthesis) endpoint. Its own model,
    /// separate from `build`'s harvest model and `distil`'s densify model. Falls back to `distil`
    /// when unset (see `reason_endpoint`), for continuity while lab.toml is migrated.
    #[serde(default)]
    pub reason: Option<Endpoint>,
    /// Endpoint for `kbx build`'s stage-3 bridge judge (a binary "is this a real cross-doc link?"
    /// per candidate pair). A cheap/fast model suffices — falls back to `model` when unset. Point it
    /// at a small local model to keep a large candidate set from costing a strong-model call each.
    #[serde(default)]
    pub bridge: Option<Endpoint>,
    /// Per-workspace overrides for the reason/build/distil agent-loop tuning knobs. Absent
    /// section == every field unset == identical behavior to today (see [`Tuning`]).
    #[serde(default)]
    pub tuning: Tuning,
}

/// Per-workspace overrides for the reason/build/distil agent-loop tuning knobs — `fanout_max`
/// (reason's predecessor branching cap), `max_rounds` (agent-loop round cap, shared by reason/
/// build/distil), `chunks_per_round` (sections read per coverage round, build/distil), and the
/// four per-stage `jobs_*` worker-pool sizes (`jobs_build`/`jobs_reason`/`jobs_train`/
/// `jobs_distil`). Every field is optional: an absent `[tuning]` section (or an absent field
/// within it) means "no override here", so [`resolve`] falls through to the CLI flag (if any) and
/// finally the built-in default — this section is purely additive over today's
/// CLI-flag-or-hardcoded-const behavior.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct Tuning {
    #[serde(default)]
    pub fanout_max: Option<usize>,
    #[serde(default)]
    pub max_rounds: Option<usize>,
    #[serde(default)]
    pub chunks_per_round: Option<usize>,
    /// Worker-pool size for `kbx build`'s extract stage only (judge stays sequential). `None`
    /// defers to a `--jobs` CLI flag, then the built-in default (3) — resolved via [`resolve`]
    /// `.max(1)` at the call site.
    #[serde(default)]
    pub jobs_build: Option<usize>,
    /// Worker-pool size for `kbx reason`'s seed workers. Same precedence as `jobs_build`.
    #[serde(default)]
    pub jobs_reason: Option<usize>,
    /// Worker-pool size for `kbx train`'s read-only rollouts. Same precedence as `jobs_build`.
    #[serde(default)]
    pub jobs_train: Option<usize>,
    /// Worker-pool size for `kbx distil`'s densify workers. Same precedence as `jobs_build`.
    #[serde(default)]
    pub jobs_distil: Option<usize>,
    /// Worker-pool size for `kbx eval`'s per-case reader+judge loop. Same precedence as
    /// `jobs_build`.
    #[serde(default)]
    pub jobs_eval: Option<usize>,
}

/// The precedence every kbx pipeline's tuning knob resolves through: an explicit CLI flag wins,
/// then a `lab.toml` `[tuning]` value, then the built-in default. A plain `Option::or`/
/// `unwrap_or` chain, named so every call site reads the same, self-documenting way.
pub fn resolve<T>(cli: Option<T>, lab: Option<T>, default: T) -> T {
    cli.or(lab).unwrap_or(default)
}

impl LabConfig {
    /// Read and parse `<workspace>/lab.toml`.
    pub fn load(workspace: &Path) -> anyhow::Result<Self> {
        Self::load_at(&workspace.join("lab.toml"))
    }

    /// Read and parse an explicit `lab.toml` path.
    pub fn load_at(lab_path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(lab_path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", lab_path.display()))?;
        let config: LabConfig = toml::from_str(&text)
            .map_err(|e| anyhow::anyhow!("parsing {}: {e}", lab_path.display()))?;
        Ok(config)
    }

    /// The endpoint `kbx reason` (phase-2) uses: `[reason]` if set, else `[distil]` as fallback.
    pub fn reason_endpoint(&self) -> Option<&Endpoint> {
        self.reason.as_ref().or(self.distil.as_ref())
    }

    /// The endpoint `kbx build`'s extract stage uses: `[build]` if set, else `[model]` (the default
    /// endpoint). Mirrors `reason_endpoint`'s override-or-fallback shape.
    pub fn build_endpoint(&self) -> &Endpoint {
        self.build.as_ref().unwrap_or(&self.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lab_config_parses_and_defaults() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("lab.toml"),
            "[model]\nendpoint=\"http://x\"\nmodel=\"m\"\n",
        )
        .unwrap();
        let c = LabConfig::load(dir.path()).unwrap();
        assert!(c.judge.is_none());
        assert!(c.reflect.is_none());
        assert!(c.distil.is_none());
        assert_eq!(c.model.timeout_secs, 120); // d120 default
    }

    #[test]
    fn lab_loads_without_corpus_and_reads_distil() {
        let toml = r#"
            [model]
            endpoint = "http://x"
            model = "m"
            [distil]
            endpoint = "http://y"
            model = "big"
        "#;
        let lab: LabConfig = toml::from_str(toml).unwrap();
        assert!(lab.distil.is_some());
        assert_eq!(lab.distil.as_ref().unwrap().model, "big");
    }

    #[test]
    fn load_at_reads_an_explicit_path() {
        let dir = tempfile::tempdir().unwrap();
        let lab_path = dir.path().join("custom-lab.toml");
        std::fs::write(&lab_path, "[model]\nendpoint=\"http://x\"\nmodel=\"m\"\n").unwrap();
        let c = LabConfig::load_at(&lab_path).unwrap();
        assert_eq!(c.model.model, "m");
    }

    #[test]
    fn reason_endpoint_prefers_reason_then_falls_back_to_distil() {
        // [reason] present -> used
        let toml = r#"
            [model]
            endpoint = "http://x"
            model = "m"
            [distil]
            endpoint = "http://d"
            model = "big"
            [reason]
            endpoint = "http://r"
            model = "phase2"
        "#;
        let lab: LabConfig = toml::from_str(toml).unwrap();
        assert_eq!(lab.reason_endpoint().unwrap().model, "phase2");

        // [reason] absent, [distil] present -> falls back to distil
        let toml2 = r#"
            [model]
            endpoint = "http://x"
            model = "m"
            [distil]
            endpoint = "http://d"
            model = "big"
        "#;
        let lab2: LabConfig = toml::from_str(toml2).unwrap();
        assert_eq!(lab2.reason_endpoint().unwrap().model, "big");

        // neither -> None
        let toml3 = "[model]\nendpoint=\"http://x\"\nmodel=\"m\"\n";
        let lab3: LabConfig = toml::from_str(toml3).unwrap();
        assert!(lab3.reason_endpoint().is_none());
    }

    #[test]
    fn tuning_parses_all_three_fields_when_present() {
        let toml = r#"
            [model]
            endpoint = "http://x"
            model = "m"
            [tuning]
            fanout_max = 5
            max_rounds = 40
            chunks_per_round = 4
        "#;
        let lab: LabConfig = toml::from_str(toml).unwrap();
        assert_eq!(lab.tuning.fanout_max, Some(5));
        assert_eq!(lab.tuning.max_rounds, Some(40));
        assert_eq!(lab.tuning.chunks_per_round, Some(4));
    }

    #[test]
    fn tuning_defaults_to_all_none_when_section_absent() {
        let toml = "[model]\nendpoint=\"http://x\"\nmodel=\"m\"\n";
        let lab: LabConfig = toml::from_str(toml).unwrap();
        assert_eq!(lab.tuning.fanout_max, None);
        assert_eq!(lab.tuning.max_rounds, None);
        assert_eq!(lab.tuning.chunks_per_round, None);
    }

    #[test]
    fn tuning_section_with_a_subset_of_fields_leaves_the_rest_none() {
        let toml = r#"
            [model]
            endpoint = "http://x"
            model = "m"
            [tuning]
            max_rounds = 40
        "#;
        let lab: LabConfig = toml::from_str(toml).unwrap();
        assert_eq!(lab.tuning.fanout_max, None);
        assert_eq!(lab.tuning.max_rounds, Some(40));
        assert_eq!(lab.tuning.chunks_per_round, None);
    }

    #[test]
    fn build_endpoint_prefers_build_then_falls_back_to_model() {
        // [build] present -> used
        let toml = r#"
            [model]
            endpoint = "http://x"
            model = "reader"
            [build]
            endpoint = "http://b"
            model = "harvest"
        "#;
        let lab: LabConfig = toml::from_str(toml).unwrap();
        assert_eq!(lab.build_endpoint().model, "harvest");

        // [build] absent -> falls back to [model]
        let toml2 = "[model]\nendpoint=\"http://x\"\nmodel=\"reader\"\n";
        let lab2: LabConfig = toml::from_str(toml2).unwrap();
        assert_eq!(lab2.build_endpoint().model, "reader");
    }

    #[test]
    fn tuning_parses_jobs_eval_and_defaults_to_none() {
        let toml = r#"
            [model]
            endpoint = "http://x"
            model = "m"
            [tuning]
            jobs_eval = 7
        "#;
        let lab: LabConfig = toml::from_str(toml).unwrap();
        assert_eq!(lab.tuning.jobs_eval, Some(7));

        let toml2 = "[model]\nendpoint=\"http://x\"\nmodel=\"m\"\n";
        let lab2: LabConfig = toml::from_str(toml2).unwrap();
        assert_eq!(lab2.tuning.jobs_eval, None);
    }

    #[test]
    fn tuning_parses_all_four_jobs_fields_when_present() {
        let toml = r#"
            [model]
            endpoint = "http://x"
            model = "m"
            [tuning]
            jobs_build = 2
            jobs_reason = 4
            jobs_train = 5
            jobs_distil = 6
        "#;
        let lab: LabConfig = toml::from_str(toml).unwrap();
        assert_eq!(lab.tuning.jobs_build, Some(2));
        assert_eq!(lab.tuning.jobs_reason, Some(4));
        assert_eq!(lab.tuning.jobs_train, Some(5));
        assert_eq!(lab.tuning.jobs_distil, Some(6));
    }

    #[test]
    fn tuning_jobs_fields_default_to_none_when_section_absent() {
        let toml = "[model]\nendpoint=\"http://x\"\nmodel=\"m\"\n";
        let lab: LabConfig = toml::from_str(toml).unwrap();
        assert_eq!(lab.tuning.jobs_build, None);
        assert_eq!(lab.tuning.jobs_reason, None);
        assert_eq!(lab.tuning.jobs_train, None);
        assert_eq!(lab.tuning.jobs_distil, None);
    }

    /// The precedence every stage's `jobs` resolution follows: CLI > lab.toml `[tuning]
    /// jobs_<stage>` > built-in default (3), then `.max(1)` so `--jobs 0` can never spawn zero
    /// workers (falls through to 1, the same as a single-threaded inline run).
    #[test]
    fn jobs_resolve_precedence_cli_over_lab_over_default_and_clamps_zero_to_one() {
        assert_eq!(resolve(Some(5), Some(2), 3).max(1), 5);
        assert_eq!(resolve(None, Some(2), 3).max(1), 2);
        assert_eq!(resolve(None, None, 3).max(1), 3);
        assert_eq!(resolve(Some(0), Some(2), 3).max(1), 1);
        assert_eq!(resolve(None, Some(0), 3).max(1), 1);
    }

    #[test]
    fn resolve_prefers_cli_over_lab_over_default() {
        assert_eq!(resolve(Some(9), Some(5), 3), 9, "CLI must win outright");
        assert_eq!(resolve(None, Some(5), 3), 5, "lab must win when CLI is unset");
        assert_eq!(resolve(None::<usize>, None, 3), 3, "default when both are unset");
    }

    #[test]
    fn endpoint_without_api_defaults_to_openai_chat() {
        let toml = r#"
            [model]
            endpoint = "http://x"
            model = "m"
        "#;
        let lab: LabConfig = toml::from_str(toml).unwrap();
        assert_eq!(lab.model.api, ApiKind::OpenAiChat);
    }

    #[test]
    fn endpoint_api_anthropic_parses() {
        let toml = r#"
            [model]
            endpoint = "http://x"
            model = "m"
            api = "anthropic"
        "#;
        let lab: LabConfig = toml::from_str(toml).unwrap();
        assert_eq!(lab.model.api, ApiKind::Anthropic);
    }

    #[test]
    fn endpoint_api_openai_alias_parses_to_openai_chat() {
        let toml = r#"
            [model]
            endpoint = "http://x"
            model = "m"
            api = "openai"
        "#;
        let lab: LabConfig = toml::from_str(toml).unwrap();
        assert_eq!(lab.model.api, ApiKind::OpenAiChat);
    }

    #[test]
    fn endpoint_api_openai_responses_parses() {
        let toml = r#"
            [model]
            endpoint = "http://x"
            model = "m"
            api = "openai_responses"
        "#;
        let lab: LabConfig = toml::from_str(toml).unwrap();
        assert_eq!(lab.model.api, ApiKind::OpenAiResponses);
    }
}
