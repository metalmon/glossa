//! `kbx train` engine: GEPA-optimizes the graph-reader SYSTEM prompt (the workspace's
//! `answer.md`) file-first — rollouts run against `lab.toml`'s `[model]` endpoint (the same
//! in-process agent loop `kbx run`/the eval backend drives), reflection is delegated to the
//! plain OpenAI-compatible `[reflect]` endpoint via a bare `chat_once` call (deliberately NOT
//! TensorZero — `kbx` has no gateway dependency). The winner is always written under
//! `runs/<tag>/answer.md` (+ a short `report.md`); it's copied back onto the workspace's own
//! `answer.md` only when it strictly beats the seed prompt's EM (`should_apply`), backing up the
//! prior file to `answer.md.bak` first. `--no-apply` disables that copy-back unconditionally, for
//! a dry-run/inspect-only pass.
//!
//! The CLI subcommand that parses flags into `TrainArgs` and calls `run_train` is a separate,
//! later task — this module is pure engine glue.

use crate::backend::openai::{reset_resamples, reset_tokens, StatusTicker};
use crate::gepa::{self, CandidateSelection};
use crate::gepa_graph::{self, GepaGraphConfig};
use crate::lab::LabConfig;
use crate::workspace;
use anyhow::Context;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::IsTerminal;
use std::path::PathBuf;
use std::time::Duration;

/// Fallback worker-pool size for concurrent read-only rollouts when neither `--jobs` nor
/// `lab.toml`'s `[tuning] jobs_train` overrides it. Same single-source-of-truth rationale as
/// `build::DEFAULT_CHUNKS_PER_ROUND`/`reason::run::DEFAULT_FANOUT_MAX`.
const DEFAULT_JOBS: usize = 3;

/// Auto-budget preset: sizes the DSPy-style metric-call budget and candidate cap RELATIVE to the
/// dataset (N answerable golds), so the same flag scales from a tiny smoke-test to a heavy sweep
/// without the caller hand-tuning `--max-metric-calls`. `mult` × N = total reader rollouts allowed
/// for the search; `max_candidates` caps the pool (and thus the final full-val pass).
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum AutoPreset {
    /// Smallest useful sweep (smoke-test): 6×N rollouts, ≤4 candidates.
    Tiny,
    /// Balanced default: 13×N rollouts, ≤6 candidates.
    Light,
    /// Deeper search: 17×N rollouts, ≤12 candidates.
    Medium,
    /// Heaviest sweep: 20×N rollouts, ≤18 candidates.
    Heavy,
}

impl AutoPreset {
    /// Metric-call multiplier: total allowed rollouts = `mult * n`.
    fn mult(self) -> usize {
        match self {
            AutoPreset::Tiny => 6,
            AutoPreset::Light => 13,
            AutoPreset::Medium => 17,
            AutoPreset::Heavy => 20,
        }
    }

    /// Candidate-pool cap for this preset.
    fn max_candidates(self) -> usize {
        match self {
            AutoPreset::Tiny => 4,
            AutoPreset::Light => 6,
            AutoPreset::Medium => 12,
            AutoPreset::Heavy => 18,
        }
    }
}

/// Default candidate-pool cap when a RAW budget (`--max-metric-calls`/`--max-full-evals`) is given
/// without an `--auto` preset to imply one.
const DEFAULT_RAW_MAX_CANDIDATES: usize = 12;

/// Resolved DSPy-style budget: hard ceiling on reader rollouts (`max_metric_calls`) plus the pool
/// cap (`max_candidates`) and the `--auto` preset it derived from (`None` when a raw budget was
/// given), for the AUTO-PARAMS print.
struct AutoBudget {
    max_metric_calls: usize,
    max_candidates: usize,
    preset: Option<AutoPreset>,
}

/// Derive the budget from the dataset size `n` (answerable golds) and the mutually-exclusive budget
/// knobs. Exactly one of `{auto, max_metric_calls, max_full_evals}` should be `Some`; the caller
/// enforces exclusivity and defaults `auto = Light` when none is set. A raw `--max-metric-calls m`
/// is used verbatim; `--max-full-evals k` means `k` full passes over the N golds (`k * n`).
fn auto_budget(
    n: usize,
    auto: Option<AutoPreset>,
    max_metric_calls: Option<usize>,
    max_full_evals: Option<usize>,
) -> AutoBudget {
    if let Some(m) = max_metric_calls {
        return AutoBudget {
            max_metric_calls: m.max(1),
            max_candidates: DEFAULT_RAW_MAX_CANDIDATES,
            preset: None,
        };
    }
    if let Some(k) = max_full_evals {
        return AutoBudget {
            max_metric_calls: k.saturating_mul(n).max(1),
            max_candidates: DEFAULT_RAW_MAX_CANDIDATES,
            preset: None,
        };
    }
    let preset = auto.unwrap_or(AutoPreset::Light);
    AutoBudget {
        max_metric_calls: preset.mult().saturating_mul(n).max(1),
        max_candidates: preset.max_candidates(),
        preset: Some(preset),
    }
}

/// CLI-facing knobs for `kbx train`. The three budget knobs (`auto`/`max_metric_calls`/
/// `max_full_evals`) are mutually exclusive; `minibatch`/`val_frac`/`pareto_size` are `Option`
/// (defaulted in `run_train`) so "not passed" is distinguishable from an explicit value.
pub struct TrainArgs {
    /// Dataset-relative budget preset (`tiny|light|medium|heavy`). Mutually exclusive with the two
    /// raw-budget knobs; when all three are `None`, `run_train` defaults to `light`.
    pub auto: Option<AutoPreset>,
    /// Raw hard ceiling on total reader rollouts (metric calls) — used verbatim.
    pub max_metric_calls: Option<usize>,
    /// Budget expressed as K full passes over the N answerable golds (`max_metric_calls = k * n`).
    pub max_full_evals: Option<usize>,
    pub minibatch: Option<usize>,
    pub val_frac: Option<f64>,
    pub pareto_size: Option<usize>,
    pub candidate_selection: String,
    pub dataset: Option<PathBuf>,
    pub prompt: Option<PathBuf>,
    pub reflect_prompt: Option<PathBuf>,
    pub tag: Option<String>,
    pub rng_seed: Option<u64>,
    pub no_apply: bool,
    pub no_progress: bool,
    /// Worker-pool size for concurrent read-only rollouts. `None` defers to `lab.toml`'s
    /// `[tuning] jobs_train`, then `DEFAULT_JOBS` (3) — resolved in `run_train`.
    pub jobs: Option<usize>,
    /// Selection metric: `"judge"` (default, graded LLM judge: Correct=1.0/Partial=0.5/Wrong=0.0)
    /// or `"exact"` (exact-match EM — byte-for-byte the pre-metric behavior).
    pub metric: String,
}

/// Apply-gate: copy the winning prompt back onto the workspace `answer.md` only when GEPA's
/// full-set gate passed (`GepaGraphResult::applied` — the winner scored >= the seed on the FULL
/// question set, not just the val split GEPA selected on, so a val-overfit winner that regresses is
/// refused), and never when `--no-apply` was passed (a dry-run: still writes `runs/<tag>/answer.md`,
/// never touches the workspace file).
pub(crate) fn should_apply(applied: bool, no_apply: bool) -> bool {
    !no_apply && applied
}

/// Run one `kbx train` pass: resolve the workspace, load `lab.toml`, GEPA-optimize `answer.md`
/// against `dataset.toml` using `[model]` for rollouts and `[reflect]` for reflection, write the
/// winner under `runs/<tag>/`, and apply it to the workspace when it earns its keep.
pub fn run_train(path: Option<PathBuf>, args: TrainArgs) -> anyhow::Result<()> {
    let paths = workspace::resolve(path);
    let lab = LabConfig::load_at(&paths.lab)?;
    // Worker-pool size for `gepa_graph::score_questions`'s concurrent read-only rollouts
    // (CLI `--jobs` > `[tuning] jobs_train` > DEFAULT_JOBS), clamped to at least 1.
    let jobs = crate::lab::resolve(args.jobs, lab.tuning.jobs_train, DEFAULT_JOBS).max(1);
    let reflect_ep = lab
        .reflect
        .clone()
        .context("kbx train needs a [reflect] endpoint in lab.toml")?;

    let prompt_path = args.prompt.clone().unwrap_or_else(|| paths.answer.clone());
    let seed = std::fs::read_to_string(&prompt_path)
        .with_context(|| format!("read seed prompt {}", prompt_path.display()))?;

    let dataset_path = args
        .dataset
        .clone()
        .unwrap_or_else(|| paths.dataset.clone());
    let dataset_text = std::fs::read_to_string(&dataset_path)
        .with_context(|| format!("read dataset {}", dataset_path.display()))?;
    let dataset = crate::dataset_toml::parse_dataset_toml(&dataset_text)?;
    // Filter out `answerable = false` (out-of-corpus) golds before they reach GEPA — the whole
    // `dataset` vec below flows into `gepa_graph::run`, which derives the train/val split,
    // minibatches, rollouts, and Pareto set from it, so filtering here covers every GEPA site.
    // Absent field => all `true` => nothing dropped (unchanged behavior).
    let before_answerable = dataset.len();
    let dataset: Vec<crate::dataset::Question> =
        dataset.into_iter().filter(|q| q.answerable).collect();
    let excluded_unanswerable = before_answerable - dataset.len();
    if excluded_unanswerable > 0 {
        println!(
            "kbx train: excluded {excluded_unanswerable} unanswerable (answerable=false) — not optimized against"
        );
    }

    let reflect_md_path = args
        .reflect_prompt
        .clone()
        .unwrap_or_else(|| paths.reflect.clone());
    let reflect_md = std::fs::read_to_string(&reflect_md_path)
        .with_context(|| format!("read reflector prompt {}", reflect_md_path.display()))?;

    let tag = args.tag.clone().unwrap_or_else(gepa::default_run_tag);
    let rng_seed = args.rng_seed.unwrap_or_else(|| gepa::hash_run_seed(&tag));
    let candidate_selection = args
        .candidate_selection
        .parse::<CandidateSelection>()
        .with_context(|| format!("parse candidate_selection {:?}", args.candidate_selection))?;

    // Selection metric: "judge" (default, graded LLM judge) or "exact" (exact-match EM). Judge
    // needs both a [judge] endpoint in lab.toml and the workspace judge.md prompt; exact needs
    // neither and reproduces the pre-metric GEPA path exactly (cfg.judge == None).
    let judge_cfg = match args.metric.as_str() {
        "exact" => None,
        "judge" => {
            let ep = lab
                .judge
                .clone()
                .context("--metric judge needs a [judge] endpoint in lab.toml")?;
            let md = std::fs::read_to_string(&paths.judge)
                .with_context(|| format!("read judge prompt {}", paths.judge.display()))?;
            Some(gepa_graph::JudgeCfg { ep, md })
        }
        other => anyhow::bail!("unknown --metric {other:?} (expected \"exact\" or \"judge\")"),
    };

    // Simulated-user dialogue gate (opt-in): load the persona prompt only when `[user_sim]` is
    // configured, so train rollouts optimize the prod prompt under the same dialogue dynamics eval
    // uses. Absent -> `None` -> today's behavior (a text-only rollout turn ends the loop).
    let user_sim_prompt = if lab.user_sim.is_some() {
        Some(
            std::fs::read_to_string(&paths.user_sim)
                .with_context(|| format!("read user_sim prompt {}", paths.user_sim.display()))?,
        )
    } else {
        None
    };

    // --- dataset analysis + DSPy-style auto-budget --------------------------------------------
    // N = answerable golds GEPA optimizes against (unanswerable already filtered above). The
    // budget is sized RELATIVE to N so one `--auto` preset scales across dataset sizes.
    let n = dataset.len();
    // hop_type breakdown (empty -> "(untyped)"), alphabetical for a deterministic printed line.
    let mut hop_counts: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for q in &dataset {
        let key = if q.hop_type.is_empty() {
            "(untyped)".to_string()
        } else {
            q.hop_type.clone()
        };
        *hop_counts.entry(key).or_default() += 1;
    }
    let hop_breakdown = hop_counts
        .iter()
        .map(|(k, c)| format!("{k} {c}"))
        .collect::<Vec<_>>()
        .join(" · ");

    // The three budget knobs are mutually exclusive; when none is set, default to `auto = light`.
    let n_budget_knobs = [
        args.auto.is_some(),
        args.max_metric_calls.is_some(),
        args.max_full_evals.is_some(),
    ]
    .iter()
    .filter(|&&set| set)
    .count();
    anyhow::ensure!(
        n_budget_knobs <= 1,
        "--auto, --max-metric-calls and --max-full-evals are mutually exclusive (pass at most one)"
    );
    let budget = auto_budget(n, args.auto, args.max_metric_calls, args.max_full_evals);
    // Minibatch/val_frac/pareto default here (GEPA/DSPy guidance: small minibatch, small val so
    // train stays large). Pareto defaults to a small constant — the true val size is only known
    // inside `gepa_graph::run` after the split, so it can't be derived here.
    let minibatch = args.minibatch.unwrap_or(3);
    let val_frac = args.val_frac.unwrap_or(0.2);
    let pareto_size = args.pareto_size.unwrap_or(12);
    // Minibatch-cache mode: lab.toml `[tuning] gepa_minibatch_cache` (default false = canonical
    // fresh-per-proposal GEPA); env `GEPA_MINIBATCH_CACHE` overrides inside `gepa_graph::run`.
    let minibatch_cache = lab.tuning.gepa_minibatch_cache.unwrap_or(false);
    // Rollout-averaging: K rollouts per question, averaged, to cut a weak reader's variance in the
    // accept/Pareto/apply-gate decisions. lab.toml `[tuning] gepa_rollout_samples`; None/1 = a
    // single rollout (today's behavior). Floored at 1 so a stray 0 can't disable scoring.
    let rollout_samples = lab.tuning.gepa_rollout_samples.unwrap_or(1).max(1);

    let model_ep = lab.model.clone();
    let model_key = model_ep.resolve_key();
    let cfg = GepaGraphConfig {
        endpoint: model_ep.endpoint.clone(),
        model: model_ep.model.clone(),
        api_key: model_key,
        val_frac,
        max_metric_calls: budget.max_metric_calls,
        max_candidates: budget.max_candidates,
        minibatch,
        seed_prompt: seed,
        work: paths.root.clone(),
        seed: rng_seed,
        pareto_size,
        candidate_selection,
        minibatch_cache,
        rollout_samples,
        jobs,
        judge: judge_cfg,
        user_sim: lab.user_sim.clone(),
        user_sim_prompt,
    };

    // Reflect via the plain `[reflect]` endpoint: system = reflect.md, user = GEPA's instruction.
    let reflect = |instruction: &str| -> anyhow::Result<String> {
        let msgs = [
            serde_json::json!({"role": "system", "content": reflect_md}),
            serde_json::json!({"role": "user", "content": instruction}),
        ];
        let msg = crate::backend::openai::chat_once_resampled(&reflect_ep, &msgs)?;
        let text = msg
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        // Take text after the last marker if present, else the whole reply, THEN check
        // emptiness — checking before extraction would let an all-preamble reply with only a
        // bare trailing marker pass the guard while still yielding an empty child prompt.
        let child = gepa::after_marker(&text, "=== NEW SYSTEM PROMPT ===");
        anyhow::ensure!(!child.trim().is_empty(), "reflector returned empty output");
        Ok(child)
    };

    // Visible progress bar for the long GEPA run (mirrors `kbx run`/build/reason): one bar owned
    // here, driven by `gepa_graph::run` (length = max_metric_calls, position = rollouts spent).
    // Hidden on a non-TTY or under `--no-progress`, exactly like `run_eval`.
    let show_progress =
        !args.no_progress && std::io::stdout().is_terminal() && std::io::stderr().is_terminal();
    let pb = if show_progress {
        // Length 0 at creation: `gepa_graph::run` sets the real length (= max_metric_calls) once it
        // starts, so the bar tracks metric-call spend end-to-end (`[spent/max_metric_calls]`).
        // Seeding a count here would flash a misleading total before that first `set_length`.
        let pb = ProgressBar::new(0);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.white} {prefix} [{pos}/{len}] {wide_bar:.white} {elapsed_precise}{msg}",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );
        pb.enable_steady_tick(Duration::from_millis(90));
        pb
    } else {
        ProgressBar::hidden()
    };
    // Zero the shared token/resample counters before the run so the ticker's `{msg}` reflects only
    // this run, then start the background ticker (ETA + tokens/resamples in `{msg}`). The static
    // stage word `training` prefixes the bar; `gepa_graph::run` extends it per iteration.
    reset_tokens();
    reset_resamples();
    pb.set_prefix("training");
    let ticker = StatusTicker::start(&pb);

    // Dataset analysis + resolved auto-params, above the live bar (pb.println is a plain println on
    // a hidden bar). Kept value-free (counts/params only — no gold/corpus text).
    pb.println(format!("ANALYSIS: N={n} · {hop_breakdown}"));
    pb.println(format!(
        "AUTO-PARAMS: max_metric_calls={} max_candidates={} minibatch={minibatch} val_frac={val_frac} pareto={pareto_size} (auto={})",
        budget.max_metric_calls,
        budget.max_candidates,
        match budget.preset {
            Some(p) => format!("{p:?}").to_lowercase(),
            None => "off".to_string(),
        },
    ));
    if n < 30 {
        pb.println(format!(
            "WARNING: small dataset (N={n} < 30) — GEPA may overfit and the winner may not generalize"
        ));
    }

    let result = gepa_graph::run(cfg, dataset, &reflect, &pb)?;

    drop(ticker);
    pb.finish_and_clear();

    let run_dir = paths.runs.join(&tag);
    std::fs::create_dir_all(&run_dir)
        .with_context(|| format!("create run dir {}", run_dir.display()))?;
    let winner_path = run_dir.join("answer.md");
    std::fs::write(&winner_path, &result.prompt)
        .with_context(|| format!("write {}", winner_path.display()))?;

    let apply = should_apply(result.applied, args.no_apply);
    let report = format!(
        "# kbx train — {tag}\n\n\
         seed score (baseline, full val): {baseline_score:.3}\n\
         best score (winner, full val):   {best_score:.3}\n\
         candidates explored: {candidates}\n\
         applied to workspace answer.md: {apply}\n",
        tag = tag,
        baseline_score = result.baseline_score,
        best_score = result.best_score,
        candidates = result.candidates,
        apply = apply,
    );
    let report_path = run_dir.join("report.md");
    std::fs::write(&report_path, &report)
        .with_context(|| format!("write {}", report_path.display()))?;

    if apply {
        if paths.answer.exists() {
            let mut backup = paths.answer.clone().into_os_string();
            backup.push(".bak");
            let backup = PathBuf::from(backup);
            std::fs::copy(&paths.answer, &backup).with_context(|| {
                format!("backup {} -> {}", paths.answer.display(), backup.display())
            })?;
        }
        std::fs::write(&paths.answer, &result.prompt)
            .with_context(|| format!("write {}", paths.answer.display()))?;
    }

    println!(
        "kbx train {tag}: seed_score={:.3} best_score={:.3} candidates={} winner={} applied={}",
        result.baseline_score,
        result.best_score,
        result.candidates,
        winner_path.display(),
        apply,
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_apply_gates_on_full_set_verdict_and_not_when_disabled() {
        assert!(should_apply(true, false)); // full-set gate passed -> apply
        assert!(!should_apply(false, false)); // winner regressed on full set -> keep seed
        assert!(!should_apply(true, true)); // --no-apply -> never write answer.md
    }

    /// `jobs`' own precedence mirrors every other stage's tuning knob: CLI > lab.toml `[tuning]
    /// jobs_train` > `DEFAULT_JOBS` (3), then `.max(1)` so `--jobs 0` never spawns zero workers.
    #[test]
    fn jobs_resolves_cli_over_lab_over_default_and_clamps_zero_to_one() {
        use crate::lab::resolve;
        assert_eq!(DEFAULT_JOBS, 3);
        assert_eq!(resolve(Some(5), Some(2), DEFAULT_JOBS).max(1), 5);
        assert_eq!(resolve(None, Some(2), DEFAULT_JOBS).max(1), 2);
        assert_eq!(resolve(None, None, DEFAULT_JOBS).max(1), DEFAULT_JOBS);
        assert_eq!(resolve(Some(0), Some(2), DEFAULT_JOBS).max(1), 1);
    }

    /// Auto-budget: presets scale the metric-call ceiling with N and imply a candidate cap; the two
    /// raw knobs override (verbatim `m`; `k * n` full passes) and fall back to the raw candidate cap.
    #[test]
    fn auto_budget_scales_with_n_and_raw_knobs_override() {
        // Preset default (light) = 13×N rollouts, ≤6 candidates.
        let b = auto_budget(40, Some(AutoPreset::Light), None, None);
        assert_eq!(b.max_metric_calls, 13 * 40);
        assert_eq!(b.max_candidates, 6);
        assert_eq!(b.preset, Some(AutoPreset::Light));

        // tiny/medium/heavy multipliers + candidate caps.
        assert_eq!(
            auto_budget(10, Some(AutoPreset::Tiny), None, None).max_metric_calls,
            60
        );
        assert_eq!(
            auto_budget(10, Some(AutoPreset::Medium), None, None).max_candidates,
            12
        );
        assert_eq!(
            auto_budget(10, Some(AutoPreset::Heavy), None, None).max_metric_calls,
            200
        );

        // Raw max-metric-calls used verbatim, raw candidate cap.
        let m = auto_budget(40, None, Some(250), None);
        assert_eq!(m.max_metric_calls, 250);
        assert_eq!(m.max_candidates, DEFAULT_RAW_MAX_CANDIDATES);
        assert_eq!(m.preset, None);

        // Full-evals = k passes over N.
        let f = auto_budget(40, None, None, Some(5));
        assert_eq!(f.max_metric_calls, 5 * 40);
        assert_eq!(f.max_candidates, DEFAULT_RAW_MAX_CANDIDATES);
    }
}
