//! `kbx reason` pipeline orchestrator (phase-2): resolve the workspace, pull the grounded SEED
//! pool from the graph (`distil::seed_pool`), run `chain_one_seed` once per grounded terminal to
//! backward-synthesize the query-side reasoning layer, checkpoint per-seed for `--resume`, then
//! `finalize` (hygiene/doctor + node-index rebuild) — mirrors `run_build`'s shape
//! (`crate::build::run_build`) so the two pipelines stay recognizably siblings.

use crate::backend::openai::{
    cache_is_estimated, reset_resamples, reset_tokens, token_summary, StatusTicker,
};
use crate::checkpoint::Checkpoint;
use crate::distil::Seed;
use crate::lab::LabConfig;
use crate::parallel::{run_units_parallel, GraphWriter};
use crate::workspace::{self, KbxPaths};
use anyhow::{bail, Context, Result};
use glossa::graph::ontology::{Ontology, RelationRole};
use glossa::index::store::DocIndex;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashSet;
use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// Soft cap on predecessors synthesized per backward step when `--fanout-max` isn't given —
/// the single source of truth `run_reason_at` applies at its real call site AND the unit test
/// below asserts against, so drift between the two can't happen silently.
const DEFAULT_FANOUT_MAX: usize = 3;

/// Fallback worker-pool size for the seed loop when neither `--jobs` nor `lab.toml`'s `[tuning]
/// jobs_reason` overrides it. Same single-source-of-truth rationale as `DEFAULT_FANOUT_MAX`.
const DEFAULT_JOBS: usize = 3;

/// CLI-level options for `kbx reason`, folded from the `kbx` binary's clap fields (mirrors
/// `crate::build::BuildOpts`'s shape).
#[derive(Debug, Clone)]
pub struct ReasonArgs {
    /// Restrict seeds to this node_type (default: the ontology's `requires_grounding` types).
    pub seed_type: Option<String>,
    /// Soft cap on predecessors synthesized per backward step (default 3).
    pub fanout_max: Option<usize>,
    /// Agent-loop round cap for one seed's backward-synth pass (default 30). Resolved
    /// CLI > `lab.toml` `[tuning] max_rounds` > `reason::seed::DEFAULT_MAX_ROUNDS`.
    pub max_rounds: Option<usize>,
    /// Only process the first N (in seed-pool order) grounded seeds.
    pub limit: Option<usize>,
    /// Clear this run's checkpoint first — a true full rebuild of the typed layer's seed marks.
    pub force: bool,
    /// Skip seeds already recorded done in the reason checkpoint.
    pub resume: bool,
    /// Never draw the progress bar, even on a TTY.
    pub no_progress: bool,
    /// Worker-pool size for the seed loop. `None` defers to `lab.toml`'s `[tuning] jobs_reason`,
    /// then `DEFAULT_JOBS` (3) — resolved in `run_reason_at`. Stored for now; the worker pool
    /// itself lands in a later task (see `parallel::run_units_parallel`).
    pub jobs: Option<usize>,
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
        ProgressStyle::with_template(
            "{spinner:.white} {prefix} [{pos}/{len}] {bar:40.white} {elapsed_precise}{msg}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    // Animates the spinner and redraws the bar on a timer even between `pb.inc`/`set_message`
    // calls — a no-op on a hidden bar.
    pb.enable_steady_tick(Duration::from_millis(90));
    pb
}

/// Clear a reason run's persistent checkpoint (`run_dir/done/`) — `--force`'s full-rebuild half,
/// mirroring `build::clear_checkpoint`. A no-op when the checkpoint dir doesn't exist yet.
fn clear_checkpoint(run_dir: &std::path::Path) -> Result<()> {
    let done_dir = run_dir.join("done");
    if done_dir.exists() {
        std::fs::remove_dir_all(&done_dir)
            .with_context(|| format!("clearing checkpoint {}", done_dir.display()))?;
    }
    Ok(())
}

/// The checkpoint unit id for one seed — `reason:{id}`, matching `run_build`'s
/// `extract:{doc}` naming convention.
fn unit_id(seed_id: &str) -> String {
    format!("reason:{seed_id}")
}

/// A seed id is skipped this pass iff `--resume` is set AND the checkpoint already recorded it
/// done. Pure and model-free — the piece Step 1's TDD unit exercises directly.
pub fn should_skip(cp: &Checkpoint, seed_id: &str, resume: bool) -> bool {
    resume && cp.is_done(&unit_id(seed_id))
}

/// Grounded seeds that still need a query-side chain built: with `force`, every seed (today's
/// full-rebuild behavior, unchanged); otherwise only the "lonely"/chainless ones — those whose id
/// is NOT in `chained_ids` (ids that are the destination of a chaining-relation edge, i.e. already
/// have a query-side chain from a prior `kbx reason` run). Pure and graph-free so it's unit-tested
/// without a `GraphStore`; order is preserved (seed-pool's deterministic id sort survives).
pub fn chainless_seeds(seeds: Vec<Seed>, chained_ids: &HashSet<String>, force: bool) -> Vec<Seed> {
    if force {
        return seeds;
    }
    seeds
        .into_iter()
        .filter(|s| !chained_ids.contains(&s.id))
        .collect()
}

/// Orchestrate the `kbx reason` pipeline over the corpus at `path` (kb-style PATH resolution via
/// `workspace::resolve`): load `lab.toml` + ontology + `reason.md`, pull the grounded seed pool,
/// run `chain_one_seed` once per (non-`--limit`-excluded, non-`--resume`-skipped) seed,
/// checkpoint each as it completes, then `finalize`.
pub fn run_reason(path: Option<PathBuf>, args: ReasonArgs) -> Result<()> {
    let paths = workspace::resolve(path);
    run_reason_at(paths, args)
}

/// `run_reason`'s body, taking already-resolved `KbxPaths` — split out so tests (and any future
/// caller that already has `paths`, e.g. after scaffolding a temp workspace) don't need to fight
/// `workspace::resolve`'s PATH-walking discovery.
fn run_reason_at(paths: KbxPaths, args: ReasonArgs) -> Result<()> {
    let lab = LabConfig::load_at(&paths.lab)
        .with_context(|| format!("loading {}", paths.lab.display()))?;
    let ontology = Ontology::load_or_default(&paths.root);

    // Ensure the corpus is indexed (structural nodes + chunks) — no-op if already indexed.
    glossa::index::store::index_dir(&paths.root, false).context("indexing corpus")?;

    let reason_md = std::fs::read_to_string(&paths.reason)
        .with_context(|| format!("reading {}", paths.reason.display()))?;

    // Phase-2 seeds from grounded terminal nodes (reuses distil's seed pool), NOT a gold dataset.
    // Opened ONCE for the whole run: every worker in the pool shares this SAME `GraphStore`
    // (reads via `GraphWriter::store()`, writes via `GraphWriter::upsert`) instead of each seed
    // opening its own handle against `graph.sqlite`.
    let g = Arc::new(glossa::graph::store::GraphStore::open(&paths.root)?);
    let seeds = crate::distil::seed_pool(&g, &ontology, args.seed_type.as_deref())?;
    if seeds.is_empty() {
        bail!(
            "kbx reason: no grounded seed nodes found (need a node of a grounding-required type \
             carrying a MENTIONS edge) — run `kbx build` first"
        );
    }
    let total_seeds = seeds.len();
    // By default, skip grounded terminals that already have a query-side chain — mirrors
    // `kbx build`'s delta-incremental NEW-∪-CHANGED behavior instead of reprocessing the whole
    // seed pool every run. A terminal already has a chain iff some edge's `to` is this seed's id
    // AND that edge's relation is `Chaining` per the ontology (not hardcoded per edge_type — see
    // `RelationRole`). `--force` skips this filter entirely (today's full-rebuild behavior).
    // Single-threaded and order-preserving, run BEFORE the worker pool so the pool only ever sees
    // seeds that must actually be chained.
    let chained_ids: HashSet<String> = if args.force {
        HashSet::new()
    } else {
        g.all_edges()?
            .into_iter()
            .filter(|e| ontology.relation_role(&e.edge_type) == RelationRole::Chaining)
            .map(|e| e.to)
            .collect()
    };
    let seeds = chainless_seeds(seeds, &chained_ids, args.force);
    if args.force {
        println!("reason: {total_seeds} grounded terminal(s), --force set: rebuilding all");
    } else {
        let skipped = total_seeds - seeds.len();
        println!(
            "reason: {total_seeds} grounded terminal(s), {skipped} already have a query-side \
             chain (skipped; use --force to rebuild), {} to process",
            seeds.len()
        );
    }
    // The worker pool shares this ONE `GraphStore` (kept alive above for the chainless scan) via
    // the `GraphWriter`; the doc index is opened once and shared read-only.
    let idx = DocIndex::open_or_create(&paths.root).context("opening doc index")?;
    let writer = GraphWriter::new(Arc::clone(&g), paths.root.clone());

    let fanout_max =
        crate::lab::resolve(args.fanout_max, lab.tuning.fanout_max, DEFAULT_FANOUT_MAX);
    let max_rounds = crate::lab::resolve(
        args.max_rounds,
        lab.tuning.max_rounds,
        crate::reason::seed::DEFAULT_MAX_ROUNDS,
    );
    let jobs = crate::lab::resolve(args.jobs, lab.tuning.jobs_reason, DEFAULT_JOBS).max(1);
    let run_dir = paths.runs.join("reason");
    if args.force {
        clear_checkpoint(&run_dir).context("clearing reason checkpoint for --force")?;
    }
    let cp = Checkpoint::open(&run_dir).context("open reason checkpoint")?;

    let mut seeds = seeds;
    if let Some(n) = args.limit {
        seeds.truncate(n);
    }

    reset_tokens();
    reset_resamples();

    let pb = progress_bar(seeds.len(), args.no_progress);
    // One static stage word at the front for the whole run — never switches. The ticker owns only
    // `{msg}` (ETA + tokens/resamples), redrawn on its own timer.
    pb.set_prefix("reasoning");
    let ticker = StatusTicker::start(&pb);

    // Pre-filter (single-threaded, order-preserving): a seed already recorded done under
    // `--resume` is skipped and accounted for in the bar right here — identical to what the old
    // sequential loop did inline — BEFORE any seed is handed to the worker pool. The pool then
    // only ever sees seeds that must actually run, so `should_skip`'s checkpoint read never
    // needs to be reasoned about under concurrency.
    let mut to_run = Vec::with_capacity(seeds.len());
    for seed in seeds {
        if should_skip(&cp, &seed.id, args.resume) {
            pb.inc(1);
            continue;
        }
        to_run.push(seed);
    }

    let results = run_units_parallel(
        to_run,
        jobs,
        &pb,
        |_seed| 1,
        |seed| {
            let stats = crate::reason::chain_one_seed(
                &paths, &ontology, &lab, &reason_md, seed, fanout_max, max_rounds, &writer, &idx,
            )
            .with_context(|| format!("synthesizing query-side for seed {}", seed.id))?;
            cp.mark(&unit_id(&seed.id), "done")
                .with_context(|| format!("marking seed {} done", seed.id))?;
            pb.println(format!(
                "reason {}: {} node(s), {} edge(s), {} grounded",
                seed.id, stats.nodes, stats.edges, stats.grounded
            ));
            Ok(stats)
        },
    )?;

    let processed = results.len();
    let (mut total_nodes, mut total_edges, mut total_grounded) = (0usize, 0usize, 0usize);
    for stats in &results {
        total_nodes += stats.nodes;
        total_edges += stats.edges;
        total_grounded += stats.grounded;
    }

    drop(ticker); // stop before finish_and_clear so it can't redraw a message onto a cleared bar
    pb.finish_and_clear();
    println!(
        "reason: {} seed(s) processed, {} node(s), {} edge(s), {} grounded",
        processed, total_nodes, total_edges, total_grounded
    );
    let footnote = if cache_is_estimated() {
        " (cache estimated from prompt re-send)"
    } else {
        ""
    };
    println!("tokens: {}{footnote}", token_summary());

    let summary = crate::build::finalize(&paths.root).context("finalizing reason")?;
    println!("{summary}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glossa::graph::store::{Edge, GraphStore, Node, Provenance};

    fn seed(id: &str) -> Seed {
        Seed {
            id: id.to_string(),
            node_type: "Fact".to_string(),
            label: id.to_string(),
        }
    }

    #[test]
    fn chainless_seeds_force_returns_all_regardless_of_chained_ids() {
        let seeds = vec![seed("a"), seed("b")];
        let chained: HashSet<String> = ["a".to_string(), "b".to_string()].into_iter().collect();
        let out = chainless_seeds(seeds.clone(), &chained, true);
        assert_eq!(
            out, seeds,
            "--force must reproduce today's full-rebuild: every seed kept"
        );
    }

    #[test]
    fn chainless_seeds_drops_chained_keeps_chainless_preserving_order() {
        let seeds = vec![seed("a"), seed("b"), seed("c")];
        let chained: HashSet<String> = ["b".to_string()].into_iter().collect();
        let out = chainless_seeds(seeds, &chained, false);
        let ids: Vec<&str> = out.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["a", "c"],
            "chained seed 'b' must be skipped; order preserved"
        );
    }

    #[test]
    fn chainless_seeds_no_chained_ids_keeps_everything() {
        let seeds = vec![seed("a"), seed("b")];
        let out = chainless_seeds(seeds.clone(), &HashSet::new(), false);
        assert_eq!(out, seeds);
    }

    fn prov() -> Provenance {
        Provenance {
            source_path: "doc.md".into(),
            range: None,
            file_sig: None,
            origin: "test".into(),
            confidence: 0.9,
            created_at: 1,
        }
    }

    /// Fixture-graph check that the chained_ids computation itself (relation_role ==
    /// Chaining, filtered from `all_edges`) matches what `run_reason_at` wires up: a terminal
    /// with an incoming chaining-role edge (RESOLVED_BY, undeclared -> defaults to Chaining per
    /// `RelationRole`) is chained; one with only a grounding-role edge (MENTIONS, a CORE_EDGE ->
    /// forced to Grounding) is chainless.
    #[test]
    fn chained_ids_from_graph_matches_chaining_role_not_grounding() {
        let dir = tempfile::tempdir().unwrap();
        let ont = Ontology::load_or_default(dir.path());
        let g = GraphStore::open(dir.path()).unwrap();

        for id in ["has-chain", "lonely"] {
            g.put_node(&Node {
                id: id.into(),
                node_type: "Fact".into(),
                label: id.into(),
                aliases: Vec::new(),
                prov: prov(),
            })
            .unwrap();
        }
        // "has-chain" is the destination of a chaining-role edge -> already has a query-side chain.
        g.put_edge(&Edge {
            from: "step-1".into(),
            to: "has-chain".into(),
            edge_type: "RESOLVED_BY".into(),
            prov: prov(),
        })
        .unwrap();
        // "lonely" only has a grounding-role MENTIONS edge -> NOT a chain, stays chainless.
        g.put_edge(&Edge {
            from: "lonely".into(),
            to: "doc.md#1".into(),
            edge_type: glossa::graph::MENTIONS.to_string(),
            prov: prov(),
        })
        .unwrap();

        let chained_ids: HashSet<String> = g
            .all_edges()
            .unwrap()
            .into_iter()
            .filter(|e| ont.relation_role(&e.edge_type) == RelationRole::Chaining)
            .map(|e| e.to)
            .collect();

        let seeds = vec![seed("has-chain"), seed("lonely")];
        let out = chainless_seeds(seeds, &chained_ids, false);
        let ids: Vec<&str> = out.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["lonely"],
            "only the chainless terminal must remain: {ids:?}"
        );
    }

    /// Step 1's TDD unit (per the brief): a seed id already recorded done in the checkpoint is
    /// skipped under `--resume`; a fresh id is not. Pure, tempdir-only — no model involved.
    #[test]
    fn should_skip_marks_done_id_under_resume_and_not_a_fresh_id() {
        let dir = tempfile::tempdir().unwrap();
        let cp = Checkpoint::open(&dir.path().join("runs").join("reason")).unwrap();
        cp.mark(&unit_id("seed-1"), "done").unwrap();

        assert!(
            should_skip(&cp, "seed-1", true),
            "an already-done seed id must be skipped under --resume"
        );
        assert!(
            !should_skip(&cp, "seed-2", true),
            "a fresh seed id must not be skipped even under --resume"
        );
        assert!(
            !should_skip(&cp, "seed-1", false),
            "a done seed id must NOT be skipped when --resume isn't set"
        );
    }

    /// Documents the default the seed-loop applies; guards against silent drift. Ties to the SAME
    /// `DEFAULT_FANOUT_MAX` constant `run_reason_at` uses at its real `unwrap_or` call site, so
    /// changing the real default (without also updating this test) fails the test — unlike
    /// asserting against a literal `3` hardcoded only here, which would track nothing.
    #[test]
    fn default_fanout_is_three_when_unset() {
        assert_eq!(DEFAULT_FANOUT_MAX, 3);

        let args = ReasonArgs {
            seed_type: None,
            fanout_max: None,
            max_rounds: None,
            limit: None,
            force: false,
            resume: false,
            no_progress: true,
            jobs: None,
        };
        assert_eq!(
            args.fanout_max.unwrap_or(DEFAULT_FANOUT_MAX),
            DEFAULT_FANOUT_MAX
        );
    }

    /// `max_rounds`' own default mirrors `fanout_max`'s: `reason::seed::DEFAULT_MAX_ROUNDS` (30)
    /// when neither a CLI flag nor `lab.toml` overrides it.
    #[test]
    fn default_max_rounds_is_thirty_when_unset() {
        assert_eq!(crate::reason::seed::DEFAULT_MAX_ROUNDS, 30);
        let args = ReasonArgs {
            seed_type: None,
            fanout_max: None,
            max_rounds: None,
            limit: None,
            force: false,
            resume: false,
            no_progress: true,
            jobs: None,
        };
        assert_eq!(
            crate::lab::resolve(
                args.max_rounds,
                None,
                crate::reason::seed::DEFAULT_MAX_ROUNDS
            ),
            30
        );
    }

    /// Precedence at the exact call sites `run_reason_at` uses: a CLI flag wins over a lab.toml
    /// `[tuning]` value, which wins over the built-in default — for both `fanout_max` and
    /// `max_rounds`.
    #[test]
    fn tuning_precedence_cli_over_lab_over_default() {
        use crate::lab::resolve;
        assert_eq!(resolve(Some(7), Some(5), DEFAULT_FANOUT_MAX), 7);
        assert_eq!(resolve(None, Some(5), DEFAULT_FANOUT_MAX), 5);
        assert_eq!(resolve(None, None, DEFAULT_FANOUT_MAX), DEFAULT_FANOUT_MAX);

        assert_eq!(
            resolve(Some(50), Some(40), crate::reason::seed::DEFAULT_MAX_ROUNDS),
            50
        );
        assert_eq!(
            resolve(None, Some(40), crate::reason::seed::DEFAULT_MAX_ROUNDS),
            40
        );
        assert_eq!(
            resolve(None, None, crate::reason::seed::DEFAULT_MAX_ROUNDS),
            crate::reason::seed::DEFAULT_MAX_ROUNDS
        );
    }

    /// `jobs`' own precedence mirrors `fanout_max`/`max_rounds`': CLI > lab.toml `[tuning]
    /// jobs_reason` > `DEFAULT_JOBS` (3), then `.max(1)` so `--jobs 0` never spawns zero workers.
    #[test]
    fn jobs_resolves_cli_over_lab_over_default_and_clamps_zero_to_one() {
        use crate::lab::resolve;
        assert_eq!(DEFAULT_JOBS, 3);
        assert_eq!(resolve(Some(7), Some(5), DEFAULT_JOBS).max(1), 7);
        assert_eq!(resolve(None, Some(5), DEFAULT_JOBS).max(1), 5);
        assert_eq!(resolve(None, None, DEFAULT_JOBS).max(1), DEFAULT_JOBS);
        assert_eq!(resolve(Some(0), Some(5), DEFAULT_JOBS).max(1), 1);
    }
}
