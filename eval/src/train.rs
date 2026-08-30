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

/// CLI-facing knobs for `kbx train` (parsed by the later CLI-wiring task; this struct is its
/// target shape).
pub struct TrainArgs {
    pub budget: usize,
    pub minibatch: usize,
    pub val_frac: f64,
    pub pareto_size: usize,
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

/// Apply-gate: copy the winning prompt back onto the workspace `answer.md` only on a STRICT EM
/// improvement over the seed prompt's own full-val EM, and never when `--no-apply` was passed
/// (a dry-run: still writes `runs/<tag>/answer.md`, never touches the workspace file).
pub(crate) fn should_apply(seed_score: f64, best_score: f64, no_apply: bool) -> bool {
    !no_apply && best_score > seed_score
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
        .with_context(|| {
            format!(
                "parse candidate_selection {:?}",
                args.candidate_selection
            )
        })?;

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

    let model_ep = lab.model.clone();
    let model_key = model_ep.resolve_key();
    let cfg = GepaGraphConfig {
        endpoint: model_ep.endpoint.clone(),
        model: model_ep.model.clone(),
        api_key: model_key,
        val_frac: args.val_frac,
        budget: args.budget,
        minibatch: args.minibatch,
        seed_prompt: seed,
        work: paths.root.clone(),
        seed: rng_seed,
        pareto_size: args.pareto_size,
        candidate_selection,
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
        let key = reflect_ep.resolve_key();
        let msg = crate::backend::openai::chat_once(
            &reflect_ep.endpoint,
            &reflect_ep.model,
            &msgs,
            key.as_deref(),
            reflect_ep.timeout_secs,
            reflect_ep.resolve_temperature(),
        )?;
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

    // Visible progress bar for the long rollout-scoring passes (mirrors `kbx run`/build/reason):
    // one bar owned here, driven per scoring pass by `gepa_graph::score_questions`. Hidden on a
    // non-TTY or under `--no-progress`, exactly like `run_eval`.
    let show_progress = !args.no_progress
        && std::io::stdout().is_terminal()
        && std::io::stderr().is_terminal();
    let pb = if show_progress {
        let pb = ProgressBar::new(dataset.len() as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.white} {prefix} [{pos}/{len}] {bar:40.white} {elapsed_precise}{msg}",
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

    let result = gepa_graph::run(cfg, dataset, &reflect, &pb)?;

    drop(ticker);
    pb.finish_and_clear();

    let run_dir = paths.runs.join(&tag);
    std::fs::create_dir_all(&run_dir)
        .with_context(|| format!("create run dir {}", run_dir.display()))?;
    let winner_path = run_dir.join("answer.md");
    std::fs::write(&winner_path, &result.prompt)
        .with_context(|| format!("write {}", winner_path.display()))?;

    let apply = should_apply(result.baseline_score, result.best_score, args.no_apply);
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
                format!(
                    "backup {} -> {}",
                    paths.answer.display(),
                    backup.display()
                )
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
    fn should_apply_only_on_strict_improvement_and_not_when_disabled() {
        assert!(should_apply(0.5, 0.6, false));
        assert!(!should_apply(0.5, 0.5, false)); // no improvement -> keep seed
        assert!(!should_apply(0.5, 0.4, false)); // regression -> keep seed
        assert!(!should_apply(0.5, 0.9, true)); // --no-apply -> never write answer.md
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
}
