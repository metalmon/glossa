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
/// `OpenAiResponses` -> `transport::responses::ResponsesTransport`, `Tensorzero` ->
/// `transport::tensorzero::TzTransport`.
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
    /// Native TensorZero gateway API (`/inference` + `/feedback`) — NOT the OpenAI-compatible
    /// shim TensorZero also exposes. Groups every turn of one question into ONE TZ episode (see
    /// `crate::episode`) and lets `kbx eval` post the judge verdict as episode feedback. Requires
    /// `Endpoint::function_name`.
    #[serde(rename = "tensorzero")]
    Tensorzero,
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
    /// Optional per-endpoint sampling temperature. `None` (the default, and when the key is absent
    /// from `lab.toml`) means "don't send a `temperature` field at all" — the provider/model uses
    /// its own default. Overridable at every call site by the `KB_EVAL_TEMP` env var (see
    /// [`Endpoint::resolve_temperature`]).
    #[serde(default)]
    pub temperature: Option<f64>,
    /// Opt-in per-endpoint rate-limit + retry policy (`backend::resilience`). `None` (absent from
    /// `lab.toml`, the default) reproduces today's behavior EXACTLY: no throttle, and the transport
    /// retries with the historical `4 attempts / 400ms*attempt` linear backoff (see
    /// [`crate::backend::resilience::RetryPolicy`]). When present, `rpm`/`max_inflight` throttle the
    /// endpoint and `retry`/`backoff_ms` override the retry constants.
    #[serde(default)]
    pub rate_limit: Option<RateLimit>,
    /// Opt-in fallback chain: on a HARD failure of this endpoint (retries exhausted / 5xx / timeout /
    /// connect error) the resilience layer advances through these endpoints in order, each with its
    /// OWN `api`/`rate_limit`. FLAT chain — a fallback's own `fallback` field is IGNORED (never
    /// recursed) to bound the chain. Empty (the default, and absent from `lab.toml`) means no
    /// fallback, byte-identical to today.
    #[serde(default)]
    pub fallback: Vec<Endpoint>,
    /// TensorZero function name to call at `/inference` (e.g. `"answer_hotpot"`). REQUIRED when
    /// `api = "tensorzero"` — `TzTransport::call` errors clearly if it's missing; ignored by every
    /// other `ApiKind`.
    #[serde(default)]
    pub function_name: Option<String>,
    /// TensorZero feedback metric name for the judge's graded score (`Verdict` -> 1.0/0.5/0.0).
    /// Defaults to `"judge"` (see [`Endpoint::feedback_score_metric`]) when absent, so an existing
    /// `lab.toml` need not declare it to match a TZ config that already uses that name.
    #[serde(default)]
    pub feedback_score_metric: Option<String>,
    /// TensorZero feedback metric name for the boolean correctness flag (`verdict == Correct`).
    /// Defaults to `"correct"` (see [`Endpoint::feedback_bool_metric`]) when absent.
    #[serde(default)]
    pub feedback_bool_metric: Option<String>,
}

/// Opt-in per-endpoint rate-limit + retry policy. Every field is optional so a partial `[<stage>]`
/// `[<stage>.rate_limit]` table is valid and unspecified knobs keep their historical defaults. A
/// wholly-absent `rate_limit` (the common case) means no throttling and the historical retry
/// constants — see [`Endpoint::rate_limit`].
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct RateLimit {
    /// Max requests per minute for this endpoint (a min-interval throttle of `60_000/rpm` ms between
    /// request starts). `None` = no rate throttle.
    #[serde(default)]
    pub rpm: Option<u32>,
    /// Max concurrent in-flight requests to this endpoint (a blocking semaphore). `None` = uncapped.
    #[serde(default)]
    pub max_inflight: Option<u32>,
    /// Total retry attempts on transient failure. `None` = the historical default of `4`.
    #[serde(default)]
    pub retry: Option<u32>,
    /// Linear backoff base in ms: attempt `n` sleeps `backoff_ms * n`. `None` = the historical `400`.
    #[serde(default)]
    pub backoff_ms: Option<u64>,
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

    /// Resolve the sampling temperature to send for this endpoint, uniform across reader/build/
    /// judge: the `KB_EVAL_TEMP` env override wins (parsed to `Some`), otherwise this endpoint's
    /// own `temperature`, otherwise `None`. `None` means the caller OMITS the `temperature` field
    /// entirely so the provider/model applies its own default — there is no hardcoded fallback.
    pub fn resolve_temperature(&self) -> Option<f64> {
        if let Ok(v) = std::env::var("KB_EVAL_TEMP") {
            if let Ok(t) = v.parse::<f64>() {
                return Some(t);
            }
        }
        self.temperature
    }

    /// The TZ feedback metric name for the graded judge score: `feedback_score_metric` if set,
    /// else `"judge"`. Only consulted on the `Tensorzero` API kind.
    pub fn feedback_score_metric(&self) -> String {
        self.feedback_score_metric
            .clone()
            .unwrap_or_else(|| "judge".to_string())
    }

    /// The TZ feedback metric name for the boolean correctness flag: `feedback_bool_metric` if
    /// set, else `"correct"`. Only consulted on the `Tensorzero` API kind.
    pub fn feedback_bool_metric(&self) -> String {
        self.feedback_bool_metric
            .clone()
            .unwrap_or_else(|| "correct".to_string())
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
    /// Optional endpoint for the reader's "simulated user" dialogue gate (see
    /// `backend::user_sim`). When present, a text-only assistant turn is no longer accepted as the
    /// final answer outright: this endpoint role-plays a patient user who deflects a non-answer
    /// (restated question / "let me think") back into the loop and only signals DONE once the
    /// assistant has actually answered. ABSENT (the default) reproduces today's behavior EXACTLY —
    /// the first text-only turn is the answer. Reuses `Endpoint::temperature`, so
    /// `[user_sim].temperature` works via `resolve_temperature()`.
    #[serde(default)]
    pub user_sim: Option<Endpoint>,
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
    /// `kbx train` GEPA minibatch-cache mode. `None`/`false` (default) = canonical GEPA (fresh
    /// minibatch + parent re-roll every proposal). `true` = frozen per-candidate minibatch — an
    /// opt-in ONLY for a weak, high-variance reader (e.g. a 4B) where canonical's per-proposal noise
    /// drowns the accept signal. See `gepa_graph::GepaGraphConfig::minibatch_cache`; env
    /// `GEPA_MINIBATCH_CACHE=1/0` overrides.
    #[serde(default)]
    pub gepa_minibatch_cache: Option<bool>,
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
        assert_eq!(
            resolve(None, Some(5), 3),
            5,
            "lab must win when CLI is unset"
        );
        assert_eq!(
            resolve(None::<usize>, None, 3),
            3,
            "default when both are unset"
        );
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
    fn endpoint_temperature_parses_when_present_and_is_none_when_absent() {
        // Present -> Some(value).
        let toml = r#"
            [model]
            endpoint = "http://x"
            model = "m"
            temperature = 0.3
        "#;
        let lab: LabConfig = toml::from_str(toml).unwrap();
        assert_eq!(lab.model.temperature, Some(0.3));

        // Absent -> None (existing lab.toml files parse unchanged, field omitted from the request).
        let toml2 = "[model]\nendpoint=\"http://x\"\nmodel=\"m\"\n";
        let lab2: LabConfig = toml::from_str(toml2).unwrap();
        assert_eq!(lab2.model.temperature, None);
    }

    #[test]
    fn user_sim_endpoint_parses_when_present_and_is_none_when_absent() {
        // Present -> Some(endpoint), and its own temperature is read like any other endpoint.
        let toml = r#"
            [model]
            endpoint = "http://x"
            model = "m"
            [user_sim]
            endpoint = "http://sim"
            model = "sim-model"
            temperature = 0.7
        "#;
        let lab: LabConfig = toml::from_str(toml).unwrap();
        assert!(lab.user_sim.is_some());
        let us = lab.user_sim.as_ref().unwrap();
        assert_eq!(us.model, "sim-model");
        assert_eq!(us.temperature, Some(0.7));

        // Absent -> None (existing lab.toml files parse unchanged; today's behavior preserved).
        let toml2 = "[model]\nendpoint=\"http://x\"\nmodel=\"m\"\n";
        let lab2: LabConfig = toml::from_str(toml2).unwrap();
        assert!(lab2.user_sim.is_none());
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

    #[test]
    fn endpoint_api_tensorzero_parses_with_function_name() {
        let toml = r#"
            [model]
            endpoint = "http://localhost:3000"
            model = "m"
            api = "tensorzero"
            function_name = "answer_hotpot"
        "#;
        let lab: LabConfig = toml::from_str(toml).unwrap();
        assert_eq!(lab.model.api, ApiKind::Tensorzero);
        assert_eq!(lab.model.function_name.as_deref(), Some("answer_hotpot"));
    }

    #[test]
    fn feedback_metric_names_default_when_absent_and_honor_overrides() {
        let toml = r#"
            [model]
            endpoint = "http://x"
            model = "m"
            api = "tensorzero"
        "#;
        let lab: LabConfig = toml::from_str(toml).unwrap();
        assert_eq!(lab.model.function_name, None);
        assert_eq!(lab.model.feedback_score_metric(), "judge");
        assert_eq!(lab.model.feedback_bool_metric(), "correct");

        let toml2 = r#"
            [model]
            endpoint = "http://x"
            model = "m"
            api = "tensorzero"
            function_name = "f"
            feedback_score_metric = "my_score"
            feedback_bool_metric = "my_bool"
        "#;
        let lab2: LabConfig = toml::from_str(toml2).unwrap();
        assert_eq!(lab2.model.feedback_score_metric(), "my_score");
        assert_eq!(lab2.model.feedback_bool_metric(), "my_bool");
    }
}
