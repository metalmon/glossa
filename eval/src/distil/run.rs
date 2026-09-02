//! `kbx distil` pipeline orchestrator: resolve the workspace, build the seed pool from grounded
//! non-structural graph nodes, run `gen::generate_one` (generate + verify-gate) once per attempt,
//! and write the kept synthetic golds as `[[case]]` rows to `--out` — the SAME `dataset.toml`
//! shape `dataset_toml::parse_dataset_toml` reads back, so `kbx reason --gold <out>` and
//! `kbx eval --dataset <out>` consume it unchanged.
//!
//! Read-only on the graph: this module never calls `graph_upsert`. The only write anywhere in
//! `kbx distil` is the `--out` dataset file.

use crate::backend::openai::{
    cache_is_estimated, reset_resamples, reset_tokens, token_summary, StatusTicker,
};
use crate::checkpoint::Checkpoint;
use crate::distil::densify::{densify_doc, DensifyStats};
use crate::distil::gen::{generate_one, GenOutcome, Seed};
use crate::lab::LabConfig;
use crate::parallel::{run_units_parallel, GraphWriter};
use crate::workspace::{self, KbxPaths};
use anyhow::{bail, Context, Result};
use glossa::graph::ontology::Ontology;
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// Fallback for `chunks_per_round` when neither a CLI flag nor `lab.toml`'s `[tuning]
/// chunks_per_round` overrides it — mirrors `build::DEFAULT_CHUNKS_PER_ROUND`'s rationale (single
/// source of truth `run_densify_at` resolves against AND this module's own test asserts against).
const DEFAULT_CHUNKS_PER_ROUND: usize = 3;

/// Fallback worker-pool size for the densify doc loop when neither `--jobs` nor `lab.toml`'s
/// `[tuning] jobs_distil` overrides it. Same single-source-of-truth rationale as
/// `DEFAULT_CHUNKS_PER_ROUND`.
const DEFAULT_JOBS: usize = 3;

/// CLI-level options for `kbx distil`, folded from the `kbx` binary's clap fields (mirrors
/// `reason::ReasonArgs`'s shape).
#[derive(Debug, Clone)]
pub struct DistilArgs {
    /// Number of synthetic golds to ATTEMPT — the gate may drop some; kept/dropped are reported.
    /// `None` when `--target` is used instead. When both are `None`, defaults to 1 attempt.
    pub count: Option<usize>,
    /// Keep generating until this many golds are KEPT, bounded by `max_attempts`. Takes precedence
    /// over `count` when set.
    pub target: Option<usize>,
    /// Attempt ceiling when `target` is set (default: `target * 4`, floored at `target`).
    pub max_attempts: Option<usize>,
    /// Dataset TOML to write (default `<kbx>/dataset.synthetic.toml`). Always overwritten whole.
    pub out: Option<PathBuf>,
    /// Restrict seeds to this node_type (default: the ontology's grounding-required types, or
    /// every non-structural declared type when none are marked `requires_grounding`).
    pub seed_type: Option<String>,
    /// Never draw the progress bar, even on a TTY.
    pub no_progress: bool,
    // --- `kbx distil densify` orchestrator fields (Task 3 of the densification plan; `kbx`'s
    // CLI wiring for these lands in Task 4) ---
    /// Restrict `run_densify` to a single document (its structural-graph `Document` node id, i.e.
    /// its corpus-relative path) — mirrors `BuildOpts::doc`/`ReasonArgs`' single-unit narrowing.
    pub doc: Option<String>,
    /// Clear this run's `distil:{doc}` checkpoint first — a true full rebuild of the densify pass
    /// (mirrors `BuildOpts::force`/`ReasonArgs::force`).
    pub force: bool,
    /// Skip documents already recorded done in the densify checkpoint (mirrors
    /// `BuildOpts::resume`/`ReasonArgs::resume`).
    pub resume: bool,
    /// Number of chunks folded into a single densify round — threaded straight into
    /// `densify_doc` (mirrors `BuildOpts::chunks_per_round`). `None` defers to `lab.toml`'s
    /// `[tuning] chunks_per_round`, then the built-in default (3) — resolved in `run_densify_at`.
    pub chunks_per_round: Option<usize>,
    /// Agent-loop round cap for the densify pass. `None` defers to `lab.toml`'s `[tuning]
    /// max_rounds`, then `distil::densify::DEFAULT_MAX_ROUNDS` (30) — resolved in `run_densify_at`.
    pub max_rounds: Option<usize>,
    /// Worker-pool size for the densify doc loop. `None` defers to `lab.toml`'s `[tuning]
    /// jobs_distil`, then `DEFAULT_JOBS` (3) — resolved in `run_densify_at`. Stored for now; the
    /// worker pool itself lands in a later task (see `parallel::run_units_parallel`). Densify
    /// mode only.
    pub jobs: Option<usize>,
    /// When set, `distil::run` runs the synthetic (question, answer) gold generator (the former
    /// default `kbx distil` behavior) instead of densify, writing the kept golds to this file —
    /// it supplies the gold path `run_distil_at` writes to (see [`run`]). `None` selects densify,
    /// the new default.
    pub emit_golds: Option<PathBuf>,
    /// When set, `distil::run` runs the CHAIN-driven alias enricher (`aliases::enrich_aliases_at`)
    /// instead of densify — enriches alias-poor reasoning nodes so `glossary`/`resolve` match how
    /// users phrase questions. Lower precedence than `emit_golds` (see [`distil_mode`]).
    pub aliases_only: bool,
    /// A reasoning node is "alias-poor" (eligible for `--aliases-only` enrichment) when its alias
    /// count is strictly below this. Default 3 (wired in `kbx.rs`). Alias mode only.
    pub min_aliases: usize,
    /// Golds mode only: skip terminals that are ALREADY well-covered. A terminal whose count of
    /// direct incoming chaining-role edges (each is one existing chain arriving at it — the cheap
    /// metric `gen::incoming_chain_count` computes) is `>= N` is dropped from the seed pool, so
    /// generation spreads to under-covered terminals and avoids near-duplicate golds. `None`
    /// (default) = no filter.
    pub max_chains: Option<usize>,
}

/// Which pipeline `distil::run` dispatches to — decided purely from `DistilArgs.emit_golds`, kept
/// as a standalone enum/function so the decision is unit-testable without touching either
/// model-driven body (`run_distil`/`run_densify`) it selects between.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// `--emit-golds <file>` was given: run the synthetic (question, answer) gold generator,
    /// writing kept golds to that file.
    Golds,
    /// `--aliases-only` was given (and not `--emit-golds`): run the CHAIN-driven alias enricher.
    AliasesOnly,
    /// The default when neither `--emit-golds` nor `--aliases-only` is given: densify the graph
    /// with the strong model.
    Densify,
}

/// Pure selector behind `distil::run`: `emit_golds` picks the gold generator; else `aliases_only`
/// picks the alias enricher; else densify. See [`Mode`].
pub fn distil_mode(args: &DistilArgs) -> Mode {
    if args.emit_golds.is_some() {
        Mode::Golds
    } else if args.aliases_only {
        Mode::AliasesOnly
    } else {
        Mode::Densify
    }
}

/// `kbx distil`'s single CLI entry point (mirrors `run_distil`/`run_densify`'s thin
/// `workspace::resolve`-then-dispatch shape one level up): `--emit-golds <file>` runs the
/// synthetic gold generator, writing to that file — otherwise densify runs, the new default.
///
/// The gold path is byte-for-byte the pre-existing `run_distil` behavior: `emit_golds` simply
/// supplies the `out` path the existing `write_dataset_toml`/seed-pool/`generate_one` plumbing
/// already writes to, overriding any `--out` also given (both name the same output file; a
/// caller wanting a different name once golds mode is selected should pass it via `--emit-golds`,
/// not `--out`).
pub fn run(path: Option<PathBuf>, mut args: DistilArgs) -> Result<()> {
    match distil_mode(&args) {
        Mode::Golds => {
            if let Some(emit) = args.emit_golds.clone() {
                args.out = Some(emit);
            }
            run_distil(path, args)
        }
        Mode::AliasesOnly => {
            let paths = workspace::resolve(path);
            crate::distil::aliases::enrich_aliases_at(paths, &args)
        }
        Mode::Densify => run_densify(path, &args),
    }
}

/// indicatif progress bar over `len` units — hidden when `no_progress` is set or stdout/stderr
/// isn't a TTY (mirrors `reason::progress_bar`, including its white/ETA template).
fn progress_bar(len: usize, no_progress: bool) -> ProgressBar {
    let show = !no_progress && std::io::stdout().is_terminal() && std::io::stderr().is_terminal();
    if !show {
        return ProgressBar::hidden();
    }
    let pb = ProgressBar::new(len as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.white} {prefix} [{pos}/{len}] {wide_bar:.white} {elapsed_precise}{msg}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    // Animates the spinner and redraws the bar on a timer even between `pb.inc`/`set_message`
    // calls — a no-op on a hidden bar.
    pb.enable_steady_tick(Duration::from_millis(90));
    pb
}

/// The node types eligible to seed from. An explicit `--seed-type` restricts to exactly that one
/// type (the caller's call — no further filtering). Otherwise: the ontology's types marked
/// `requires_grounding` (its "KNOWLEDGE" types, per the design doc), excluding anything the
/// ontology also declares structural; if none are marked, every declared type that isn't
/// structural. Structural (Document/Section) types are excluded either way as a safety net — the
/// seed pool itself does the real MENTIONS-groundedness filtering, but a structural node is never
/// eligible even if it happened to satisfy that.
pub fn eligible_seed_types(ont: &Ontology, seed_type: Option<&str>) -> BTreeSet<String> {
    if let Some(t) = seed_type {
        return std::iter::once(t.to_string()).collect();
    }
    let structural: HashSet<String> = ont.structural().into_iter().collect();
    let grounded: BTreeSet<String> = ont
        .entity_types()
        .iter()
        .filter(|t| ont.requires_grounding(t) && !structural.contains(*t))
        .cloned()
        .collect();
    if !grounded.is_empty() {
        return grounded;
    }
    ont.entity_types()
        .iter()
        .filter(|t| !structural.contains(*t))
        .cloned()
        .collect()
}

/// The seed pool: every node of an eligible type (see [`eligible_seed_types`]) carrying at least
/// one outgoing `MENTIONS` edge (grounded), sorted deterministically by id — so `--count` attempts
/// are reproducible run-to-run for the same graph, no RNG/wallclock involved.
pub fn seed_pool(g: &GraphStore, ont: &Ontology, seed_type: Option<&str>) -> Result<Vec<Seed>> {
    let types = eligible_seed_types(ont, seed_type);
    let mut seeds: Vec<Seed> = g
        .all_nodes()?
        .into_iter()
        .filter(|n| types.contains(&n.node_type))
        .filter(|n| {
            g.outgoing(&n.id)
                .map(|edges| edges.iter().any(|e| e.edge_type == glossa::graph::MENTIONS))
                .unwrap_or(false)
        })
        .map(|n| Seed {
            id: n.id,
            node_type: n.node_type,
            label: n.label,
        })
        .collect();
    seeds.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(seeds)
}

/// Drop already-well-covered terminals from a seed pool for `--max-chains N`: a seed whose direct
/// incoming chaining-edge count (`gen::incoming_chain_count`, ontology-general) is `>= N` is
/// excluded; a seed with fewer is kept. `None` returns the pool untouched. Order-preserving and
/// model-free, so `--max-chains` is unit-testable without a live model. See
/// [`DistilArgs::max_chains`].
pub fn filter_by_max_chains(
    seeds: Vec<Seed>,
    g: &GraphStore,
    spec: &glossa::tools::ChainSpec,
    max_chains: Option<usize>,
) -> Vec<Seed> {
    let Some(n) = max_chains else {
        return seeds;
    };
    seeds
        .into_iter()
        .filter(|s| crate::distil::gen::incoming_chain_count(g, spec, &s.id) < n)
        .collect()
}

/// One kept synthetic gold, in the exact `[[case]]` shape `dataset_toml::parse_dataset_toml`
/// reads back (`id`/`question`/`answer`/`hop_type`; `aliases`/`tags` are optional there and
/// simply omitted here — they default to empty on read-back).
#[derive(Debug, Serialize)]
struct OutCase {
    id: String,
    question: String,
    answer: String,
    /// Objective (code-B) retrieval verdict for this gold — `"lexical"`/`"multihop"`. Read back by
    /// `dataset_toml::parse_dataset_toml`'s `hop_type` field for the by-question-type report slice.
    hop_type: String,
}

#[derive(Debug, Serialize)]
struct OutFile {
    case: Vec<OutCase>,
}

/// Serialize `kept` as `[[case]]` blocks and write them to `out_path` (created/truncated).
fn write_dataset_toml(out_path: &std::path::Path, kept: &[OutCase]) -> Result<()> {
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let file = OutFile {
        case: kept
            .iter()
            .map(|c| OutCase {
                id: c.id.clone(),
                question: c.question.clone(),
                answer: c.answer.clone(),
                hop_type: c.hop_type.clone(),
            })
            .collect(),
    };
    let text = toml::to_string_pretty(&file).context("serializing synthetic dataset.toml")?;
    std::fs::write(out_path, text).with_context(|| format!("writing {}", out_path.display()))?;
    Ok(())
}

/// Orchestrate `kbx distil` over the corpus at `path` (kb-style PATH resolution via
/// `workspace::resolve`): load `lab.toml` + ontology + `distil_golds.md` (the gold-gen mode's own
/// prompt file, separate from densify's `distil.md`), build the seed pool, attempt `args.count`
/// generate+gate passes (`gen::generate_one`), and write the kept golds to `args.out` (default
/// `<kbx>/dataset.synthetic.toml`).
pub fn run_distil(path: Option<PathBuf>, args: DistilArgs) -> Result<()> {
    let paths = workspace::resolve(path);
    run_distil_at(paths, args)
}

/// `run_distil`'s body, taking already-resolved `KbxPaths` — split out so tests can exercise it
/// without fighting `workspace::resolve`'s PATH-walking discovery (mirrors `reason::run_reason_at`).
fn run_distil_at(paths: KbxPaths, args: DistilArgs) -> Result<()> {
    let lab = LabConfig::load_at(&paths.lab)
        .with_context(|| format!("loading {}", paths.lab.display()))?;
    let ontology = Ontology::load_or_default(&paths.root);

    // Ensure the corpus is indexed — mirrors `run_distil`'s own first step; a no-op if already
    // indexed. Needed so `read` calls the generator makes resolve real chunks.
    glossa::index::store::index_dir(&paths.root, false).context("indexing corpus")?;

    let distil_golds_md = std::fs::read_to_string(&paths.distil_golds)
        .with_context(|| format!("reading {}", paths.distil_golds.display()))?;

    let g = GraphStore::open(&paths.root)?;
    // Shared read-only index for the code-B (retrieval) hop_type probe on kept golds — opened once,
    // reused every attempt. Distinct from `generate_one`'s own per-call index (it opens its own).
    let idx = DocIndex::open_or_create(&paths.root)?;
    let spec = glossa::tools::ChainSpec::from_ontology(&ontology);
    let seeds = seed_pool(&g, &ontology, args.seed_type.as_deref())?;
    // `--max-chains N`: drop terminals already fed by >= N incoming chaining chains (well-covered),
    // so generation spreads to under-covered terminals. No-op when unset.
    let seeds = filter_by_max_chains(seeds, &g, &spec, args.max_chains);
    if seeds.is_empty() {
        bail!(
            "kbx distil: no grounded seed nodes found (need a node of an eligible type carrying \
             an outgoing MENTIONS edge; --max-chains may have filtered the pool) — build the \
             graph first (`kbx build`)"
        );
    }

    let out_path = args
        .out
        .clone()
        .unwrap_or_else(|| paths.kbx_dir.join("dataset.synthetic.toml"));

    // Attempt budget: `--target N` keeps generating until N are KEPT, bounded by `max_attempts`
    // (default N*4, floored at N) so a stubborn gate can't loop forever; otherwise a fixed `--count`
    // attempts (default 1). Both share the sorted-by-id pool cycled by attempt index — no
    // RNG/wallclock, so a given (graph, budget) is reproducible.
    let target = args.target;
    let cap = match target {
        Some(t) => args.max_attempts.unwrap_or(t.saturating_mul(4)).max(t),
        None => args.count.unwrap_or(1),
    };

    reset_tokens();
    reset_resamples();
    let pb = progress_bar(cap, args.no_progress);
    // One static stage word at the front for the whole run; the ticker owns only `{msg}` (ETA +
    // tokens/resamples).
    pb.set_prefix("distilling");
    let ticker = StatusTicker::start(&pb);

    let mut kept: Vec<OutCase> = Vec::new();
    let mut n_dropped = 0usize;
    let mut attempts = 0usize;

    while attempts < cap {
        if target.is_some_and(|t| kept.len() >= t) {
            break;
        }
        let i = attempts;
        let seed = &seeds[i % seeds.len()];
        match generate_one(&paths, &ontology, &lab, &distil_golds_md, seed)
            .with_context(|| format!("distil attempt {i} (seed {})", seed.id))?
        {
            GenOutcome::Kept(p) => {
                // Objective (code-B) hop_type: the authoritative label for the emitted gold —
                // pure lexical retrieval, graph-independent (see `gen::code_b_hop_type`). The
                // model's own `p.hop_type` is advisory only; log one line when they disagree.
                let hop_type = crate::distil::gen::code_b_hop_type(&idx, &g, &seed.id, &p.question);
                if !p.hop_type.is_empty() && p.hop_type != hop_type {
                    pb.println(format!(
                        "hop_type: model said {}, retrieval says {} (using {})",
                        p.hop_type, hop_type, hop_type
                    ));
                }
                println!("distil {i}: kept \"{}\" (seed {})", p.question, seed.id);
                // Sequential id over KEPT golds (no gaps from dropped attempts).
                kept.push(OutCase {
                    id: format!("synth-{}", kept.len()),
                    question: p.question,
                    answer: p.answer,
                    hop_type,
                });
            }
            GenOutcome::Dropped(reason) => {
                println!(
                    "distil {i}: dropped ({}) (seed {})",
                    reason.describe(),
                    seed.id
                );
                n_dropped += 1;
            }
        }
        attempts += 1;
        pb.inc(1);
    }
    drop(ticker); // stop before finish_and_clear so it can't redraw a message onto a cleared bar
    pb.finish_and_clear();

    write_dataset_toml(&out_path, &kept)?;

    println!(
        "distil: {} attempted, {} kept, {} dropped -> {}",
        attempts,
        kept.len(),
        n_dropped,
        out_path.display()
    );
    let footnote = if cache_is_estimated() {
        " (cache estimated from prompt re-send)"
    } else {
        ""
    };
    println!("tokens: {}{footnote}", token_summary());
    if let Some(t) = target {
        if kept.len() < t {
            println!(
                "distil: target {} NOT reached — kept {} in {} attempts (cap {}); raise --max-attempts or loosen seeds",
                t,
                kept.len(),
                attempts,
                cap
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// `kbx distil densify` — the whole-corpus, checkpointed/resumable orchestrator for `densify_doc`
// (Task 3 of the 2026-08-28 densification plan). Mirrors `run_build`'s extract stage
// (`crate::build::run_build`, the `if run_extract_stage { ... }` block) in shape: enumerate docs
// -> chunk-weighted progress bar (`crate::build::extract_total_chunks`/`extract_doc_weight`,
// widened to `pub(crate)` for this reuse) -> per-doc loop with a `distil:{doc}` checkpoint mark
// and per-round `on_progress` top-up -> `finish_and_clear` -> the SAME finalize
// (`crate::build::finalize`) `run_reason` runs at the end of `kbx reason`.
// ---------------------------------------------------------------------------------------------

/// The checkpoint unit id for one densify doc pass — `distil:{doc}`, matching `run_build`'s
/// `extract:{doc}` / `run_reason`'s `reason:{seed_id}` naming convention.
pub fn distil_unit_id(doc: &str) -> String {
    format!("distil:{doc}")
}

/// Narrow an enumerated doc list to exactly `doc` when set, preserving order — the `--doc`
/// selection `run_densify_at` applies right after enumeration (mirrors `BuildOpts::doc`'s
/// narrowing in `run_build`'s extract stage). Pure and model-free, extracted so `--doc` selection
/// is unit-testable without a live model.
pub fn select_docs(mut docs: Vec<String>, doc: Option<&str>) -> Vec<String> {
    if let Some(d) = doc {
        docs.retain(|p| p == d);
    }
    docs
}

/// A doc is skipped this densify pass iff `--resume` is set AND the checkpoint already recorded
/// its `distil:{doc}` mark done. Pure and model-free — mirrors `reason::should_skip` exactly, one
/// level down (doc id instead of seed id).
pub fn should_skip_densify(cp: &Checkpoint, doc: &str, resume: bool) -> bool {
    resume && cp.is_done(&distil_unit_id(doc))
}

/// Clear a densify run's persistent checkpoint (`run_dir/done/`) — `--force`'s full-rebuild half,
/// mirroring `build::clear_checkpoint`/`reason`'s own local copy of the same helper. A no-op when
/// the checkpoint dir doesn't exist yet (first-ever densify run).
fn clear_densify_checkpoint(run_dir: &std::path::Path) -> Result<()> {
    let done_dir = run_dir.join("done");
    if done_dir.exists() {
        std::fs::remove_dir_all(&done_dir)
            .with_context(|| format!("clearing checkpoint {}", done_dir.display()))?;
    }
    Ok(())
}

/// Orchestrate `kbx distil densify` over the corpus at `path` (kb-style PATH resolution via
/// `workspace::resolve`, mirroring `run_reason`/`run_distil`'s thin entry shape).
pub fn run_densify(path: Option<PathBuf>, args: &DistilArgs) -> Result<()> {
    let paths = workspace::resolve(path);
    run_densify_at(paths, args)
}

/// `run_densify`'s body, taking already-resolved `KbxPaths` — split out so tests can exercise it
/// without fighting `workspace::resolve`'s PATH-walking discovery (mirrors
/// `run_distil_at`/`run_reason_at`).
fn run_densify_at(paths: KbxPaths, args: &DistilArgs) -> Result<()> {
    let lab = LabConfig::load_at(&paths.lab)
        .with_context(|| format!("loading {}", paths.lab.display()))?;
    let ontology = Ontology::load_or_default(&paths.root);

    // Ensure the corpus is indexed (structural nodes + chunks) — no-op if already indexed, same
    // first step every `kbx` pipeline entry takes.
    glossa::index::store::index_dir(&paths.root, false).context("indexing corpus")?;

    let distil_md = std::fs::read_to_string(&paths.distil)
        .with_context(|| format!("reading {}", paths.distil.display()))?;

    // Precedence: CLI flag > lab.toml `[tuning]` > built-in default (see `crate::lab::resolve`).
    let chunks_per_round = crate::lab::resolve(
        args.chunks_per_round,
        lab.tuning.chunks_per_round,
        DEFAULT_CHUNKS_PER_ROUND,
    );
    let max_rounds = crate::lab::resolve(
        args.max_rounds,
        lab.tuning.max_rounds,
        crate::distil::densify::DEFAULT_MAX_ROUNDS,
    );
    // Worker-pool size for the densify doc loop: CLI > lab.toml `[tuning] jobs_distil` > 3,
    // clamped to at least 1 — drives the per-doc densify loop below via `run_units_parallel`.
    let jobs = crate::lab::resolve(args.jobs, lab.tuning.jobs_distil, DEFAULT_JOBS).max(1);

    // A fresh, short-lived handle: enumerate then drop before densify_doc opens its own
    // per-document GraphStore connection — same reasoning as `run_build`'s extract stage.
    let mut docs = {
        let g = GraphStore::open(&paths.root).context("open graph store to enumerate docs")?;
        crate::build::enumerate_docs(&g)?
    };
    docs = select_docs(docs, args.doc.as_deref());
    if docs.is_empty() {
        bail!(
            "kbx distil densify: no document matched (corpus empty, or --doc names a document \
             that isn't indexed)"
        );
    }

    // Weight the bar by chunk, not by document — identical rationale/mechanism to `run_build`'s
    // extract stage (see `extract_doc_weight`'s doc comment): a huge document is otherwise one
    // tick and the bar/ETA lie on a mixed-size corpus.
    let idx = DocIndex::open_or_create(&paths.root).context("open doc index for densify")?;
    let mut chunk_counts: HashMap<String, usize> = HashMap::new();
    idx.iter_chunks(|path, _ord, _kind, _text| {
        *chunk_counts.entry(path.to_string()).or_default() += 1;
    })
    .context("counting chunks per doc for densify-bar weights")?;
    let total_chunks = crate::build::extract_total_chunks(&docs, &chunk_counts);

    // Densify shares ONE `GraphStore` (wrapped in a `GraphWriter`) across every worker in the
    // pool, instead of each document opening its own connection against `graph.sqlite` — mirrors
    // `build::run_build`'s/`reason::run_reason_at`'s `GraphWriter` wiring (Tasks 4/5). `idx`
    // (opened above for the chunk-weight count) is reused as-is: reads go straight through it
    // (tantivy's `IndexReader` is safe for concurrent readers, no lock needed — unlike
    // `GraphStore`'s reads, which funnel through its own connection mutex; see
    // `GraphWriter::store`'s doc comment).
    let g = Arc::new(GraphStore::open(&paths.root).context("open graph store for densify")?);
    let writer = GraphWriter::new(Arc::clone(&g), paths.root.clone());

    // Densify state lives under its own stable `runs/distil/` dir — one corpus has exactly one
    // in-progress densify pass to resume/checkpoint, same convention as `runs/build/`/`runs/reason/`.
    let run_dir = paths.runs.join("distil");
    if args.force {
        clear_densify_checkpoint(&run_dir).context("clearing distil checkpoint for --force")?;
    }
    let cp = Checkpoint::open(&run_dir).context("open distil checkpoint")?;

    reset_tokens();
    reset_resamples();

    let pb = progress_bar(total_chunks.max(1), args.no_progress);
    // One static stage word at the front for the whole densify pass; the ticker owns only `{msg}`
    // (ETA + tokens/resamples).
    pb.set_prefix("distilling");
    let ticker = StatusTicker::start(&pb);

    // Pre-filter (single-threaded, order-preserving): a doc already recorded done under
    // `--resume` is skipped and accounted for in the bar right here — identical to what the old
    // sequential loop did inline — BEFORE any doc is handed to the worker pool. The pool then
    // only ever sees docs that must actually run, so `should_skip_densify`'s checkpoint read
    // never needs to be reasoned about under concurrency (mirrors `run_build`'s/`run_reason_at`'s
    // pre-filter).
    let mut to_run: Vec<String> = Vec::with_capacity(docs.len());
    for doc in &docs {
        let w = crate::build::extract_doc_weight(doc, &chunk_counts) as u64;
        if should_skip_densify(&cp, doc, args.resume) {
            pb.inc(w);
            continue;
        }
        to_run.push(doc.clone());
    }

    // `lmstudio_chat` reads the sampling temperature from `KB_EVAL_TEMP`; set it ONCE here, before
    // the worker pool is spawned below — every worker's `densify_doc` call uses the SAME constant
    // `DENSIFY_TEMP` for the whole run, so a single write covers them all. Setting it per-worker
    // (the old placement, inside `densify_doc`) would race N threads on a process-global env var,
    // which is UB even though every writer agrees on the value.
    std::env::set_var(
        "KB_EVAL_TEMP",
        crate::distil::densify::DENSIFY_TEMP.to_string(),
    );

    // Each doc's progress is driven entirely by `densify_doc`'s own per-round callback plus the
    // gap top-up below, so `run_units_parallel`'s own post-work `pb.inc(weight)` must be a no-op
    // here (`weight` always 0) — mirrors `run_build`'s extract loop. `--jobs 1` runs this same
    // closure inline, in `docs` order, byte-for-byte the old sequential loop.
    let results = run_units_parallel(
        to_run,
        jobs,
        &pb,
        |_doc| 0u64,
        |doc: &String| -> Result<(String, DensifyStats)> {
            let w = crate::build::extract_doc_weight(doc, &chunk_counts) as u64;
            // Advance the bar WITHIN the doc via densify_doc's per-round callback (real chunks
            // covered), same as `run_build`'s extract stage: `covered` tracks what the callback
            // advanced so the per-doc total can be reconciled to this doc's weight below.
            let mut covered = 0u64;
            let stats = densify_doc(
                &paths,
                &ontology,
                &lab,
                &distil_md,
                doc,
                chunks_per_round,
                max_rounds,
                &writer,
                &idx,
                |n| {
                    covered += n;
                    pb.inc(n);
                },
            )
            .with_context(|| format!("densifying {doc}"))?;
            cp.mark(&distil_unit_id(doc), "done")
                .with_context(|| format!("marking {doc} done"))?;
            // A live indicatif bar owns the terminal: any per-doc message while it's live MUST go
            // through `pb.println`, never a bare `println!` (the bug this task's brief calls out
            // by name) — only the post-`finish_and_clear` summary below may use `println!`
            // directly.
            pb.println(format!(
                "distil {doc}: {} node(s), {} edge(s), {} grounded",
                stats.nodes, stats.edges, stats.grounded
            ));
            // Top up any gap so the bar stays aligned to this doc's weight, same as `run_build`.
            if covered < w {
                pb.inc(w - covered);
            }
            Ok((doc.clone(), stats))
        },
    )?;

    let mut total = DensifyStats::default();
    let mut docs_done: Vec<String> = Vec::new();
    for (doc, stats) in results {
        total.nodes += stats.nodes;
        total.edges += stats.edges;
        total.grounded += stats.grounded;
        docs_done.push(doc);
    }

    drop(ticker); // stop before finish_and_clear so it can't redraw a message onto a cleared bar
    pb.finish_and_clear();
    println!(
        "distil: {} doc(s) densified, {} chunk(s), {} node(s), {} edge(s), {} grounded",
        docs_done.len(),
        total_chunks,
        total.nodes,
        total.edges,
        total.grounded
    );
    let footnote = if cache_is_estimated() {
        " (cache estimated from prompt re-send)"
    } else {
        ""
    };
    println!("tokens: {}{footnote}", token_summary());

    // Same hygiene/generalize/node-index finalize `run_reason` runs at the end of `kbx reason`,
    // so the derived layer + node index are refreshed after densify's writes too.
    let summary = crate::build::finalize(&paths.root).context("finalizing distil")?;
    println!("{summary}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glossa::graph::store::{Edge, Node, Provenance};

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

    const GROUNDING_ONT: &str = r#"
[entities.Fact]
requires_grounding = true
[entities.Document]
[entities.Section]

[relations.LEADS_TO]
from = ["Fact"]
to = ["Fact"]
"#;

    const UNGROUNDED_ONT: &str = r#"
[entities.Fact]
[entities.Document]
[entities.Section]

[relations.LEADS_TO]
from = ["Fact"]
to = ["Fact"]
"#;

    #[test]
    fn eligible_seed_types_prefers_grounding_required_types() {
        let ont = Ontology::parse(GROUNDING_ONT).unwrap();
        let types = eligible_seed_types(&ont, None);
        assert!(types.contains("Fact"));
        assert!(
            !types.contains("Document"),
            "structural type must never be eligible"
        );
        assert!(
            !types.contains("Section"),
            "structural type must never be eligible"
        );
    }

    #[test]
    fn eligible_seed_types_falls_back_to_all_non_structural_when_none_require_grounding() {
        let ont = Ontology::parse(UNGROUNDED_ONT).unwrap();
        let types = eligible_seed_types(&ont, None);
        assert!(types.contains("Fact"));
        assert!(!types.contains("Document"));
        assert!(!types.contains("Section"));
    }

    #[test]
    fn eligible_seed_types_explicit_seed_type_wins_outright() {
        let ont = Ontology::parse(GROUNDING_ONT).unwrap();
        let types = eligible_seed_types(&ont, Some("Section"));
        assert_eq!(types, std::iter::once("Section".to_string()).collect());
    }

    #[test]
    fn seed_pool_excludes_structural_and_requires_a_mentions_edge() {
        let dir = tempfile::tempdir().unwrap();
        let ont = Ontology::parse(GROUNDING_ONT).unwrap();
        let g = GraphStore::open(dir.path()).unwrap();

        // A grounded Fact: eligible.
        g.put_node(&Node {
            id: "fact-grounded".into(),
            node_type: "Fact".into(),
            label: "grounded fact".into(),
            aliases: Vec::new(),
            prov: prov(),
        })
        .unwrap();
        g.put_edge(&Edge {
            from: "fact-grounded".into(),
            to: "doc.md#1".into(),
            edge_type: glossa::graph::MENTIONS.to_string(),
            prov: prov(),
        })
        .unwrap();

        // An UNgrounded Fact (no MENTIONS edge): excluded.
        g.put_node(&Node {
            id: "fact-ungrounded".into(),
            node_type: "Fact".into(),
            label: "ungrounded fact".into(),
            aliases: Vec::new(),
            prov: prov(),
        })
        .unwrap();

        // A structural Document node, even if (hypothetically) grounded: excluded by type.
        g.put_node(&Node {
            id: "doc.md".into(),
            node_type: "Document".into(),
            label: "doc.md".into(),
            aliases: Vec::new(),
            prov: prov(),
        })
        .unwrap();
        g.put_edge(&Edge {
            from: "doc.md".into(),
            to: "doc.md#1".into(),
            edge_type: glossa::graph::MENTIONS.to_string(),
            prov: prov(),
        })
        .unwrap();

        let seeds = seed_pool(&g, &ont, None).unwrap();
        let ids: Vec<&str> = seeds.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["fact-grounded"],
            "seed pool must be exactly the grounded Fact: {ids:?}"
        );
    }

    #[test]
    fn seed_pool_is_sorted_deterministically_by_id() {
        let dir = tempfile::tempdir().unwrap();
        let ont = Ontology::parse(GROUNDING_ONT).unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        for id in ["fact-c", "fact-a", "fact-b"] {
            g.put_node(&Node {
                id: id.into(),
                node_type: "Fact".into(),
                label: id.into(),
                aliases: Vec::new(),
                prov: prov(),
            })
            .unwrap();
            g.put_edge(&Edge {
                from: id.into(),
                to: "doc.md#1".into(),
                edge_type: glossa::graph::MENTIONS.to_string(),
                prov: prov(),
            })
            .unwrap();
        }
        let seeds = seed_pool(&g, &ont, None).unwrap();
        let ids: Vec<&str> = seeds.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["fact-a", "fact-b", "fact-c"]);
    }

    /// Ontology WITH a reasoning spine so `ChainSpec::spine_rels` (hence `incoming_chain_count`)
    /// is non-empty — the `--max-chains` filter is meaningless without a chaining relation set.
    const CHAIN_SPINE_ONT: &str = r#"
[entities.Fact]
requires_grounding = true
[entities.Document]
[entities.Section]

[relations.LEADS_TO]
from = ["Fact"]
to = ["Fact"]
role = "chaining"

[reasoning]
[[reasoning.spines]]
anchor = "Fact"
relations = ["LEADS_TO"]
"#;

    /// `--max-chains N` excludes a terminal fed by `>= N` incoming chaining chains and keeps one
    /// below N; `None` leaves the pool untouched.
    #[test]
    fn filter_by_max_chains_excludes_well_covered_terminals() {
        use glossa::graph::store::{Edge, Node};
        let dir = tempfile::tempdir().unwrap();
        let ont = Ontology::parse(CHAIN_SPINE_ONT).unwrap();
        let spec = glossa::tools::ChainSpec::from_ontology(&ont);
        let g = GraphStore::open(dir.path()).unwrap();

        // Nodes: `rich` (2 incoming chaining chains), `lean` (1), plus their predecessors.
        for id in ["rich", "lean", "p1", "p2", "p3"] {
            g.put_node(&Node {
                id: id.into(),
                node_type: "Fact".into(),
                label: id.into(),
                aliases: vec![],
                prov: prov(),
            })
            .unwrap();
        }
        for from in ["p1", "p2"] {
            g.put_edge(&Edge {
                from: from.into(),
                to: "rich".into(),
                edge_type: "LEADS_TO".into(),
                prov: prov(),
            })
            .unwrap();
        }
        g.put_edge(&Edge {
            from: "p3".into(),
            to: "lean".into(),
            edge_type: "LEADS_TO".into(),
            prov: prov(),
        })
        .unwrap();

        let seeds = vec![
            Seed {
                id: "rich".into(),
                node_type: "Fact".into(),
                label: "rich".into(),
            },
            Seed {
                id: "lean".into(),
                node_type: "Fact".into(),
                label: "lean".into(),
            },
        ];

        // N=2: `rich` (2 >= 2) excluded, `lean` (1 < 2) kept.
        let kept = filter_by_max_chains(seeds.clone(), &g, &spec, Some(2));
        let ids: Vec<&str> = kept.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["lean"], "well-covered terminal excluded: {ids:?}");

        // None => untouched.
        assert_eq!(filter_by_max_chains(seeds, &g, &spec, None).len(), 2);
    }

    #[test]
    fn write_dataset_toml_round_trips_through_the_real_parser() {
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("dataset.synthetic.toml");
        let kept = vec![
            OutCase {
                id: "synth-0".into(),
                question: "what follows the seed?".into(),
                answer: "the terminal fact".into(),
                hop_type: "multihop".into(),
            },
            OutCase {
                id: "synth-1".into(),
                question: "second question?".into(),
                answer: "second answer".into(),
                hop_type: "lexical".into(),
            },
        ];
        write_dataset_toml(&out_path, &kept).unwrap();

        let text = std::fs::read_to_string(&out_path).unwrap();
        let parsed = crate::dataset_toml::parse_dataset_toml(&text).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].id, "synth-0");
        assert_eq!(parsed[0].question, "what follows the seed?");
        assert_eq!(parsed[0].answer, "the terminal fact");
        assert_eq!(parsed[0].hop_type, "multihop", "hop_type round-trips");
        assert_eq!(parsed[1].id, "synth-1");
        assert_eq!(parsed[1].hop_type, "lexical");
    }

    // ---- densify orchestrator (Task 3): doc selection + checkpoint, model-free ----------------

    #[test]
    fn distil_unit_id_formats_distil_prefix() {
        assert_eq!(distil_unit_id("a.md"), "distil:a.md");
    }

    /// `--doc` narrows an enumerated set to exactly that one doc; unset leaves it untouched.
    /// Mirrors `plan_extract`/`should_skip`'s pure, model-free unit-test style in `build`/`reason`.
    #[test]
    fn select_docs_filters_to_exactly_the_named_doc() {
        let docs = vec!["a.md".to_string(), "b.md".to_string(), "c.md".to_string()];

        assert_eq!(
            select_docs(docs.clone(), Some("b.md")),
            vec!["b.md".to_string()]
        );
        assert_eq!(select_docs(docs.clone(), None), docs);
    }

    /// `--doc` naming a document absent from the enumerated set yields an empty selection, not an
    /// error — `run_densify_at` turns that into a clear bail! rather than silently no-op-ing.
    #[test]
    fn select_docs_absent_doc_yields_empty() {
        let docs = vec!["a.md".to_string()];
        assert!(select_docs(docs, Some("nonexistent.md")).is_empty());
    }

    /// Step 1's TDD unit (per the brief, mirroring `reason::should_skip`'s own test): a doc id
    /// already recorded done in the checkpoint is skipped under `--resume`; a fresh id is not; and
    /// a done id is NOT skipped when `--resume` isn't set.
    #[test]
    fn should_skip_densify_marks_done_doc_under_resume_and_not_a_fresh_doc() {
        let dir = tempfile::tempdir().unwrap();
        let cp = Checkpoint::open(&dir.path().join("runs").join("distil")).unwrap();
        cp.mark(&distil_unit_id("a.md"), "done").unwrap();

        assert!(
            should_skip_densify(&cp, "a.md", true),
            "an already-done doc must be skipped under --resume"
        );
        assert!(
            !should_skip_densify(&cp, "b.md", true),
            "a fresh doc must not be skipped even under --resume"
        );
        assert!(
            !should_skip_densify(&cp, "a.md", false),
            "a done doc must NOT be skipped when --resume isn't set"
        );
    }

    /// `--force`'s checkpoint-clearing half: a doc marked done in a prior densify run must report
    /// `!is_done` again after `clear_densify_checkpoint`, mirroring `build::force_clears_checkpoint`.
    #[test]
    fn force_clears_densify_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let run_dir = dir.path().join("runs").join("distil");
        let cp = Checkpoint::open(&run_dir).unwrap();
        cp.mark(&distil_unit_id("a.md"), "done").unwrap();
        assert!(cp.is_done(&distil_unit_id("a.md")));

        clear_densify_checkpoint(&run_dir).unwrap();

        let cp2 = Checkpoint::open(&run_dir).unwrap();
        assert!(!cp2.is_done(&distil_unit_id("a.md")));
    }

    /// `clear_densify_checkpoint` must be a no-op (not an error) when the run has never
    /// checkpointed anything yet — the first-ever `--force` densify run on a fresh corpus.
    #[test]
    fn clear_densify_checkpoint_noop_when_absent() {
        let dir = tempfile::tempdir().unwrap();
        let run_dir = dir.path().join("runs").join("distil");
        assert!(!run_dir.exists());
        clear_densify_checkpoint(&run_dir).unwrap();
    }

    /// `DistilArgs`' new densify fields (Task 3) default to a sane full-corpus, non-resumed pass
    /// when constructed directly (as `kbx.rs`'s CLI wiring will do in Task 4).
    #[test]
    fn distil_args_densify_fields_are_plain_and_settable() {
        let args = DistilArgs {
            count: None,
            target: None,
            max_attempts: None,
            out: None,
            seed_type: None,
            no_progress: true,
            doc: Some("a.md".to_string()),
            force: true,
            resume: false,
            chunks_per_round: Some(3),
            max_rounds: None,
            jobs: None,
            emit_golds: None,
            aliases_only: false,
            min_aliases: 3,
            max_chains: None,
        };
        assert_eq!(args.doc.as_deref(), Some("a.md"));
        assert!(args.force);
        assert_eq!(args.chunks_per_round, Some(3));
    }

    /// Precedence resolver's contract at `run_densify_at`'s exact call sites, for both knobs this
    /// pass resolves: CLI wins, then `lab.toml`'s `[tuning]`, then the built-in default.
    #[test]
    fn chunks_per_round_and_max_rounds_resolve_cli_over_lab_over_default() {
        use crate::lab::resolve;
        assert_eq!(DEFAULT_CHUNKS_PER_ROUND, 3);
        assert_eq!(resolve(Some(9), Some(6), DEFAULT_CHUNKS_PER_ROUND), 9);
        assert_eq!(resolve(None, Some(6), DEFAULT_CHUNKS_PER_ROUND), 6);
        assert_eq!(
            resolve(None, None, DEFAULT_CHUNKS_PER_ROUND),
            DEFAULT_CHUNKS_PER_ROUND
        );

        assert_eq!(crate::distil::densify::DEFAULT_MAX_ROUNDS, 30);
        assert_eq!(
            resolve(
                Some(50),
                Some(40),
                crate::distil::densify::DEFAULT_MAX_ROUNDS
            ),
            50
        );
        assert_eq!(
            resolve(None, Some(40), crate::distil::densify::DEFAULT_MAX_ROUNDS),
            40
        );
        assert_eq!(
            resolve(None, None, crate::distil::densify::DEFAULT_MAX_ROUNDS),
            crate::distil::densify::DEFAULT_MAX_ROUNDS
        );
    }

    /// `jobs`' own precedence mirrors `chunks_per_round`/`max_rounds`': CLI > lab.toml `[tuning]
    /// jobs_distil` > `DEFAULT_JOBS` (3), then `.max(1)` so `--jobs 0` never spawns zero workers.
    #[test]
    fn jobs_resolves_cli_over_lab_over_default_and_clamps_zero_to_one() {
        use crate::lab::resolve;
        assert_eq!(DEFAULT_JOBS, 3);
        assert_eq!(resolve(Some(9), Some(6), DEFAULT_JOBS).max(1), 9);
        assert_eq!(resolve(None, Some(6), DEFAULT_JOBS).max(1), 6);
        assert_eq!(resolve(None, None, DEFAULT_JOBS).max(1), DEFAULT_JOBS);
        assert_eq!(resolve(Some(0), Some(6), DEFAULT_JOBS).max(1), 1);
    }

    // ---- `distil::run` mode dispatch (Task 4): pure selector, no model involved -----------------

    /// Bare constructor for `DistilArgs` in dispatch tests — every field explicit so a future field
    /// addition fails loudly here instead of silently defaulting in a test.
    fn args_with_emit_golds(emit_golds: Option<PathBuf>) -> DistilArgs {
        DistilArgs {
            count: None,
            target: None,
            max_attempts: None,
            out: None,
            seed_type: None,
            no_progress: true,
            doc: None,
            force: false,
            resume: false,
            chunks_per_round: Some(3),
            max_rounds: None,
            jobs: None,
            emit_golds,
            aliases_only: false,
            min_aliases: 3,
            max_chains: None,
        }
    }

    #[test]
    fn distil_mode_selects_golds_when_emit_golds_is_set() {
        let args = args_with_emit_golds(Some(PathBuf::from("out.toml")));
        assert_eq!(distil_mode(&args), Mode::Golds);
    }

    #[test]
    fn distil_mode_selects_densify_when_emit_golds_is_absent() {
        let args = args_with_emit_golds(None);
        assert_eq!(distil_mode(&args), Mode::Densify);
    }

    #[test]
    fn distil_mode_selects_aliases_only_when_flag_set_without_emit_golds() {
        let mut args = args_with_emit_golds(None);
        args.aliases_only = true;
        assert_eq!(distil_mode(&args), Mode::AliasesOnly);

        // emit_golds still WINS over aliases_only (higher precedence).
        let mut both = args_with_emit_golds(Some(PathBuf::from("out.toml")));
        both.aliases_only = true;
        assert_eq!(distil_mode(&both), Mode::Golds);
    }
}
