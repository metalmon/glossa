//! `kbx distil` pipeline orchestrator (Task 4): resolve the workspace, load the gold dataset,
//! run `chain_one_gold` once per gold `(question, answer)` pair, checkpoint per-gold for
//! `--resume`, then `finalize` (hygiene/doctor + node-index rebuild) — mirrors `run_build`'s
//! shape (`crate::build::run_build`) so the two pipelines stay recognizably siblings.
//!
//! `--mode split` wires the contamination guard from the design doc's "two frames": a
//! deterministic train/test split of the gold ids, holding the test ids OUT of this run's
//! `chain_one_gold` loop entirely and recording them to a run file under `paths.runs/distil/` so
//! a later held-out-eval task (Task 6) can read them back. `--mode kb` (the default) processes
//! every gold — the "domain-KB from all solved cases" frame, where question-contamination is a
//! non-issue.

use crate::checkpoint::Checkpoint;
use crate::dataset::Question;
use crate::dataset_toml::parse_dataset_toml;
use crate::distil::chain_one_gold;
use crate::lab::LabConfig;
use crate::workspace::{self, KbxPaths};
use anyhow::{bail, Context, Result};
use glossa::graph::ontology::Ontology;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::IsTerminal;
use std::path::PathBuf;

/// Deterministic held-out fraction of gold ids under `--mode split`. Not yet CLI-configurable
/// (the brief's `DistilArgs` carries no `test_frac` field) — a fixed, documented default until a
/// later task exposes it.
const DEFAULT_TEST_FRAC: f64 = 0.2;

/// CLI-level options for `kbx distil`, folded from the `kbx` binary's clap fields (mirrors
/// `crate::build::BuildOpts`'s shape).
#[derive(Debug, Clone)]
pub struct DistilArgs {
    /// Override the workspace's default gold dataset (`paths.dataset`).
    pub gold: Option<PathBuf>,
    /// `"split"` (train-only, holds out a deterministic test fraction — never ingested) or
    /// `"kb"` (process every gold; default).
    pub mode: String,
    /// Only process the first N (post-holdout, in sorted-id order) gold cases.
    pub limit: Option<usize>,
    /// Clear this run's checkpoint first — a true full rebuild of the typed layer's gold marks.
    pub force: bool,
    /// Skip gold ids already recorded done in the distil checkpoint.
    pub resume: bool,
    /// Never draw the progress bar, even on a TTY.
    pub no_progress: bool,
}

/// indicatif progress bar over `len` units — hidden when `no_progress` is set or stdout/stderr
/// isn't a TTY (mirrors `build::progress_bar`/`kbx eval`'s `show_progress` gate).
fn progress_bar(len: usize, no_progress: bool) -> ProgressBar {
    let show = !no_progress && std::io::stdout().is_terminal() && std::io::stderr().is_terminal();
    if !show {
        return ProgressBar::hidden();
    }
    let pb = ProgressBar::new(len as u64);
    pb.set_style(
        ProgressStyle::with_template("{msg} [{pos}/{len}] {bar:40.cyan/blue} {elapsed_precise}")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    pb
}

/// Clear a distil run's persistent checkpoint (`run_dir/done/`) — `--force`'s full-rebuild half,
/// mirroring `build::clear_checkpoint`. A no-op when the checkpoint dir doesn't exist yet.
fn clear_checkpoint(run_dir: &std::path::Path) -> Result<()> {
    let done_dir = run_dir.join("done");
    if done_dir.exists() {
        std::fs::remove_dir_all(&done_dir)
            .with_context(|| format!("clearing checkpoint {}", done_dir.display()))?;
    }
    Ok(())
}

/// The checkpoint unit id for one gold case — `distil:{id}`, matching `run_build`'s
/// `extract:{doc}` naming convention.
fn unit_id(gold_id: &str) -> String {
    format!("distil:{gold_id}")
}

/// A gold id is skipped this pass iff `--resume` is set AND the checkpoint already recorded it
/// done. Pure and model-free — the piece Step 1's TDD unit exercises directly.
pub fn should_skip(cp: &Checkpoint, gold_id: &str, resume: bool) -> bool {
    resume && cp.is_done(&unit_id(gold_id))
}

/// Deterministically split `questions` into (kept, held_out) by sorted id, holding out the last
/// `frac` fraction — same split-by-id shape as `crate::gepa::split_by_episode`, kept local here
/// since that helper is generic over an externally-owned `episode_id` closure and gepa's own
/// item types, whereas this just needs the ids themselves recorded to a run file.
fn split_gold_by_id(questions: Vec<Question>, frac: f64) -> (Vec<Question>, Vec<String>) {
    let mut ids: Vec<String> = questions.iter().map(|q| q.id.clone()).collect();
    ids.sort();
    let n_held = if ids.len() <= 1 {
        0
    } else {
        ((ids.len() as f64 * frac).round() as usize).clamp(1, ids.len() - 1)
    };
    let held: std::collections::HashSet<String> = ids.iter().rev().take(n_held).cloned().collect();
    let mut held_sorted: Vec<String> = held.iter().cloned().collect();
    held_sorted.sort();
    let kept: Vec<Question> = questions.into_iter().filter(|q| !held.contains(&q.id)).collect();
    (kept, held_sorted)
}

/// Orchestrate the `kbx distil` pipeline over the corpus at `path` (kb-style PATH resolution via
/// `workspace::resolve`): load `lab.toml` + ontology + `distil.md` + gold dataset, ensure the
/// corpus is indexed, run `chain_one_gold` once per (non-held-out, non-`--limit`-excluded,
/// non-`--resume`-skipped) gold case, checkpoint each as it completes, then `finalize`.
pub fn run_distil(path: Option<PathBuf>, args: DistilArgs) -> Result<()> {
    let paths = workspace::resolve(path);
    run_distil_at(paths, args)
}

/// `run_distil`'s body, taking already-resolved `KbxPaths` — split out so tests (and any future
/// caller that already has `paths`, e.g. after scaffolding a temp workspace) don't need to fight
/// `workspace::resolve`'s PATH-walking discovery.
fn run_distil_at(paths: KbxPaths, args: DistilArgs) -> Result<()> {
    if args.mode != "split" && args.mode != "kb" {
        bail!("kbx distil --mode must be \"split\" or \"kb\", got {:?}", args.mode);
    }

    let lab = LabConfig::load_at(&paths.lab)
        .with_context(|| format!("loading {}", paths.lab.display()))?;
    let ontology = Ontology::load_or_default(&paths.root);

    // Ensure the corpus is indexed (structural Document/Section nodes + chunks) — mirrors
    // `run_build`'s own first step; a no-op if already indexed.
    glossa::index::store::index_dir(&paths.root, false).context("indexing corpus")?;

    let distil_md = std::fs::read_to_string(&paths.distil)
        .with_context(|| format!("reading {}", paths.distil.display()))?;

    let gold_path = args.gold.clone().unwrap_or_else(|| paths.dataset.clone());
    let gold_text = std::fs::read_to_string(&gold_path)
        .with_context(|| format!("reading gold dataset {}", gold_path.display()))?;
    let questions = parse_dataset_toml(&gold_text)
        .with_context(|| format!("parsing gold dataset {}", gold_path.display()))?;

    let run_dir = paths.runs.join("distil");
    let mut questions = questions;
    if args.mode == "split" {
        let (kept, held_out) = split_gold_by_id(questions, DEFAULT_TEST_FRAC);
        questions = kept;
        std::fs::create_dir_all(&run_dir)
            .with_context(|| format!("creating {}", run_dir.display()))?;
        let holdout_path = run_dir.join("holdout.txt");
        std::fs::write(&holdout_path, held_out.join("\n"))
            .with_context(|| format!("writing {}", holdout_path.display()))?;
        println!(
            "distil: --mode split held out {} gold id(s) (recorded to {})",
            held_out.len(),
            holdout_path.display()
        );
    }
    // Deterministic processing order regardless of dataset.toml's on-disk case order.
    questions.sort_by(|a, b| a.id.cmp(&b.id));

    if let Some(n) = args.limit {
        questions.truncate(n);
    }

    if args.force {
        clear_checkpoint(&run_dir).context("clearing distil checkpoint for --force")?;
    }
    let cp = Checkpoint::open(&run_dir).context("open distil checkpoint")?;

    let pb = progress_bar(questions.len(), args.no_progress);
    pb.set_message("distil");
    let mut total_nodes = 0usize;
    let mut total_edges = 0usize;
    let mut total_grounded = 0usize;
    let mut processed = 0usize;
    for q in &questions {
        if should_skip(&cp, &q.id, args.resume) {
            pb.inc(1);
            continue;
        }
        let stats = chain_one_gold(&paths, &ontology, &lab, &distil_md, &q.question, &q.answer)
            .with_context(|| format!("distilling gold {}", q.id))?;
        total_nodes += stats.nodes;
        total_edges += stats.edges;
        total_grounded += stats.grounded;
        processed += 1;
        cp.mark(&unit_id(&q.id), "done")
            .with_context(|| format!("marking gold {} done", q.id))?;
        println!(
            "distil {}: {} node(s), {} edge(s), {} grounded",
            q.id, stats.nodes, stats.edges, stats.grounded
        );
        pb.inc(1);
    }
    pb.finish_and_clear();
    println!(
        "distil: {} gold case(s) processed, {} node(s), {} edge(s), {} grounded",
        processed, total_nodes, total_edges, total_grounded
    );

    let summary = crate::build::finalize(&paths.root).context("finalizing distil")?;
    println!("{summary}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Step 1's TDD unit (per the brief): a gold id already recorded done in the checkpoint is
    /// skipped under `--resume`; a fresh id is not. Pure, tempdir-only — no model involved.
    #[test]
    fn should_skip_marks_done_id_under_resume_and_not_a_fresh_id() {
        let dir = tempfile::tempdir().unwrap();
        let cp = Checkpoint::open(&dir.path().join("runs").join("distil")).unwrap();
        cp.mark(&unit_id("gold-1"), "done").unwrap();

        assert!(
            should_skip(&cp, "gold-1", true),
            "an already-done gold id must be skipped under --resume"
        );
        assert!(
            !should_skip(&cp, "gold-2", true),
            "a fresh gold id must not be skipped even under --resume"
        );
        assert!(
            !should_skip(&cp, "gold-1", false),
            "a done gold id must NOT be skipped when --resume isn't set"
        );
    }

    #[test]
    fn split_gold_by_id_holds_out_a_deterministic_fraction_and_never_returns_a_held_out_kept_id() {
        let questions: Vec<Question> = (0..10)
            .map(|i| Question {
                id: format!("q{i:02}"),
                question: "q".into(),
                answer: "a".into(),
                ..Default::default()
            })
            .collect();

        let (kept1, held1) = split_gold_by_id(questions.clone(), 0.2);
        let (kept2, held2) = split_gold_by_id(questions.clone(), 0.2);

        assert_eq!(held1, held2, "split must be deterministic across calls");
        assert_eq!(held1.len(), 2, "20% of 10 ids => 2 held out");
        let kept_ids: std::collections::HashSet<&str> =
            kept1.iter().map(|q| q.id.as_str()).collect();
        for h in &held1 {
            assert!(
                !kept_ids.contains(h.as_str()),
                "a held-out id must never also appear in kept: {h}"
            );
        }
        assert_eq!(kept1.len() + held1.len(), questions.len());
        assert_eq!(kept2.len(), kept1.len());
    }
}
