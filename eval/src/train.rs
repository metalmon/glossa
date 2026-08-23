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

use crate::gepa::{self, CandidateSelection};
use crate::gepa_graph::{self, GepaGraphConfig};
use crate::lab::LabConfig;
use crate::workspace;
use anyhow::Context;
use std::path::PathBuf;

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
}

/// Apply-gate: copy the winning prompt back onto the workspace `answer.md` only on a STRICT EM
/// improvement over the seed prompt's own full-val EM, and never when `--no-apply` was passed
/// (a dry-run: still writes `runs/<tag>/answer.md`, never touches the workspace file).
pub(crate) fn should_apply(seed_em: f64, best_em: f64, no_apply: bool) -> bool {
    !no_apply && best_em > seed_em
}

/// Run one `kbx train` pass: resolve the workspace, load `lab.toml`, GEPA-optimize `answer.md`
/// against `dataset.toml` using `[model]` for rollouts and `[reflect]` for reflection, write the
/// winner under `runs/<tag>/`, and apply it to the workspace when it earns its keep.
pub fn run_train(path: Option<PathBuf>, args: TrainArgs) -> anyhow::Result<()> {
    let paths = workspace::resolve(path);
    let lab = LabConfig::load_at(&paths.lab)?;
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

    let result = gepa_graph::run(cfg, dataset, &reflect)?;

    let run_dir = paths.runs.join(&tag);
    std::fs::create_dir_all(&run_dir)
        .with_context(|| format!("create run dir {}", run_dir.display()))?;
    let winner_path = run_dir.join("answer.md");
    std::fs::write(&winner_path, &result.prompt)
        .with_context(|| format!("write {}", winner_path.display()))?;

    let apply = should_apply(result.baseline_em, result.best_em, args.no_apply);
    let report = format!(
        "# kbx train — {tag}\n\n\
         seed EM (baseline, full val): {baseline_em:.3}\n\
         best EM (winner, full val):   {best_em:.3}\n\
         candidates explored: {candidates}\n\
         applied to workspace answer.md: {apply}\n",
        tag = tag,
        baseline_em = result.baseline_em,
        best_em = result.best_em,
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
        "kbx train {tag}: seed_em={:.3} best_em={:.3} candidates={} winner={} applied={}",
        result.baseline_em,
        result.best_em,
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
}
