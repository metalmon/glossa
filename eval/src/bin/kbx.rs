//! `kbx` — the file-first eval toolkit CLI: `init` scaffolds a workspace (lab.toml + editable
//! answer/judge prompts + a starter dataset.toml + runs/), `eval` runs a dataset.toml against a
//! corpus with the OpenAI-compatible agent backend, scores EM/F1, optionally judges each case,
//! and writes a `runs/<tag>/report.md` (+ per-case trace files) via `kb_eval::report::write_run`.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use kb_eval::backend::openai::{
    cache_is_estimated, reset_resamples, reset_tokens, token_summary, OpenAiBackend, StatusTicker,
};
use kb_eval::backend::AgentBackend;
use kb_eval::build::{run_build, BuildOpts, BuildStage};
use kb_eval::dataset::Question;
use kb_eval::dataset_ops;
use kb_eval::dataset_toml::parse_dataset_toml;
use kb_eval::distil::{self, DistilArgs};
use kb_eval::finetune::{
    export_dpo, export_sft, load_trajectories_for_tag, to_jsonl, write_trajectories, SftShape,
    TrajectoryRecord,
};
use kb_eval::judge::{judge, Judgement, Verdict};
use kb_eval::lab::{self, LabConfig};
use kb_eval::parallel::run_units_parallel;
use kb_eval::reason::{self, ReasonArgs};
use kb_eval::report::{
    lexical_text, load_cases, summary_text, write_answers_csv, write_case, write_run, AnswerRow,
    CaseResult, RunMeta,
};
use kb_eval::scaffold::scaffold_init;
use kb_eval::score::{relaxed_match_any, token_f1_any};
use kb_eval::train::{self, TrainArgs};
use kb_eval::workspace::{self, KbxPaths};
use std::collections::HashSet;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Parser)]
#[command(name = "kbx", about = "File-first glossa eval toolkit")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

/// `export --format` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum ExportFormat {
    /// Supervised fine-tuning demonstrations from Correct trajectories.
    Sft,
    /// Preference pairs (Correct vs Wrong) for DPO.
    Dpo,
}

/// `export --shape` values (SFT only).
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum ExportShape {
    /// `{"messages":[...]}` for Unsloth `apply_chat_template` (default).
    Messages,
    /// `{"conversations":[...]}` for `standardize_sharegpt`.
    Sharegpt,
}

/// `export --dpo-focus` values (DPO only). Currently a single variant — kept as an enum (rather
/// than a bare bool flag) so a future focus mode has a natural slot without a breaking CLI change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum DpoFocus {
    /// Emit ONLY pairs whose rejected trajectory spiraled past a "gain has plateaued" signal
    /// (kept searching instead of answering); skip questions lacking such a rejected candidate.
    Plateau,
}

#[derive(Subcommand)]
enum Cmd {
    /// Scaffold a fresh eval workspace at `<path>/.glossa/kbx/`: lab.toml + prompts + dataset.toml + runs/.
    Init {
        /// Corpus root (kb-style PATH resolution: explicit if given, else discovered from the
        /// current directory upward, else the current directory).
        path: Option<PathBuf>,
        /// Overwrite existing template files instead of skipping them.
        #[arg(long)]
        force: bool,
    },
    /// Run a corpus's `.glossa/kbx/dataset.toml` against it and write a run report.
    Eval {
        /// Corpus root (kb-style PATH resolution: explicit if given, else discovered from the
        /// current directory upward, else the current directory).
        path: Option<PathBuf>,
        /// Run tag (report dir name under runs/). Default: slug(root)-slug(dataset).
        #[arg(long)]
        tag: Option<String>,
        /// Override the workspace's default `dataset.toml`.
        #[arg(long)]
        dataset: Option<PathBuf>,
        /// Override the workspace's default `answer.md` (the answer-agent system prompt file).
        #[arg(long)]
        prompt: Option<PathBuf>,
        /// Override the workspace's default `judge.md`.
        #[arg(long)]
        judge: Option<PathBuf>,
        /// Only run the first N cases (after --tag-filter).
        #[arg(long)]
        limit: Option<usize>,
        /// Only run cases whose `tags` include this value.
        #[arg(long = "tag-filter")]
        tag_filter: Option<String>,
        /// Skip LLM judging even if lab.toml has a [judge] endpoint configured.
        #[arg(long = "no-judge")]
        no_judge: bool,
        /// Skip cases whose id already has a persisted result under runs/<tag>/cases/, then merge
        /// the old + newly-run cases into the final report.
        #[arg(long)]
        resume: bool,
        /// Never draw the progress bar, even on a TTY.
        #[arg(long = "no-progress")]
        no_progress: bool,
        /// Worker-pool size for the per-case reader+judge loop (default 3). Falls back to
        /// `lab.toml`'s `[tuning] jobs_eval`, then the built-in default, when unset. `0` clamps to
        /// 1 (never zero workers). `1` reproduces the sequential run exactly.
        #[arg(long)]
        jobs: Option<usize>,
        /// Record each case's full chat trajectory (system+user, every tool round, final answer)
        /// to `runs/<tag>/trajectories.jsonl`, joining the judge verdict as the reward — the raw
        /// material for `kbx export`. OFF by default: no file, no overhead, byte-identical.
        #[arg(long)]
        capture: bool,
        /// With `--capture`, run each case this many times so several varied trajectories per
        /// question accumulate (the reader is stochastic at temp>0) — needed for DPO pairs.
        /// Ignored without `--capture`; the reported EM/F1/verdict still come from the first sample.
        #[arg(long, default_value_t = 1)]
        samples: usize,
        /// Predict-only: run the agent over the dataset's questions WITHOUT scoring — skips the
        /// judge, keeps `answerable=false` cases (there are no golds to be un/answerable), and
        /// tolerates empty gold answers. Use for a fresh question list you only want answered. The
        /// end-of-run graded-quality summary is omitted (nothing to score against).
        #[arg(long = "no-gold")]
        no_gold: bool,
        /// Also write a flat question->answer CSV (UTF-8 BOM, Excel-friendly) for handing to the
        /// customer to grade — always with a trailing blank `quality` column. With golds it also
        /// carries `gold`+`verdict` columns; under `--no-gold` just `id,question,answer,quality`. A
        /// relative path resolves under `runs/<tag>/` (never the indexed corpus); absolute is used
        /// as given.
        #[arg(long)]
        answers: Option<PathBuf>,
    },
    /// Post-process captured `runs/<tag>/trajectories.jsonl` into an Unsloth-ready fine-tuning
    /// dataset: SFT (`messages`/`sharegpt`) from Correct trajectories, or DPO
    /// (`prompt`/`chosen`/`rejected`) pairing a Correct against a Wrong trajectory per question.
    /// Pure deterministic post-process — no network.
    Export {
        /// Corpus root (kb-style PATH resolution) whose `runs/` holds the captured trajectories.
        path: Option<PathBuf>,
        /// Run tag(s) to read `runs/<tag>/trajectories.jsonl` from (comma-separated for several).
        #[arg(long)]
        from: String,
        /// Dataset kind to emit.
        #[arg(long, value_enum)]
        format: ExportFormat,
        /// Output JSONL file to write.
        #[arg(long)]
        out: PathBuf,
        /// SFT output shape: `messages` (default, for `apply_chat_template`) or `sharegpt`
        /// (`conversations`, for `standardize_sharegpt`). Ignored for DPO.
        #[arg(long, value_enum, default_value = "messages")]
        shape: ExportShape,
        /// SFT only: also keep Partial-verdict trajectories as demonstrations (default: Correct only).
        #[arg(long = "include-partial")]
        include_partial: bool,
        /// DPO only: max pairs emitted per question (default 1 = best Correct vs a Wrong).
        #[arg(long = "max-pairs", default_value_t = 1)]
        max_pairs: usize,
        /// DPO only: restrict to plateau-contrastive pairs — rejected must have searched past a
        /// "gain has plateaued" signal instead of answering. Questions lacking such a rejected
        /// candidate are skipped (and counted). Without this flag, DPO pairing still BIASES the
        /// rejected choice toward a spiraled trajectory when several Wrong ones exist, but falls
        /// back to any Wrong so today's behavior is preserved when nothing plateaued.
        #[arg(long = "dpo-focus", value_enum)]
        dpo_focus: Option<DpoFocus>,
        /// SFT only: stable-partition kept trajectories so ones containing a plateau turn come
        /// first (a light up-weight of "reacted to the signal and still answered correctly"
        /// demonstrations). Default off = today's input order.
        #[arg(long = "sft-prefer-signal")]
        sft_prefer_signal: bool,
    },
    /// Build a corpus's reasoning graph: extract -> candidates -> judge -> finalize.
    Build {
        /// Corpus root (kb-style PATH resolution: explicit if given, else discovered from the
        /// current directory upward, else the current directory).
        path: Option<PathBuf>,
        /// Which stage(s) of the pipeline to run.
        #[arg(long, value_enum, default_value = "all")]
        stage: BuildStage,
        /// Restrict extraction to a single document (its corpus-relative path).
        #[arg(long)]
        doc: Option<String>,
        /// Only extract the first N enumerated documents.
        #[arg(long)]
        limit: Option<usize>,
        /// Bypass the incremental delta (which by default extracts only new/changed docs) for a
        /// full rebuild: extract every document and re-judge every candidate pair.
        #[arg(long)]
        force: bool,
        /// Skip units already recorded done in the build checkpoint.
        #[arg(long)]
        resume: bool,
        /// Never draw the progress bar, even on a TTY.
        #[arg(long = "no-progress")]
        no_progress: bool,
        /// Optional size guard (stage 3): cap the facts fed to one entity-group bridge-judge call.
        /// DEFAULT 0 = NO CAP — judge every group whole and trust the model to return no links for a
        /// generic/co-mention-only entity (that is the whole point of the group judge); the model's
        /// large context fits realistic groups. Set a positive N only as an escape hatch for a
        /// pathologically huge group, where it judges the first N (deterministic) and logs the drop.
        #[arg(long = "bridge-max-facts", default_value_t = 0)]
        bridge_max_facts: usize,
        /// Feed images the `read` tool returns (page rasters / embedded figures) to the
        /// extraction model as vision input, so scanned/image-only content can still yield
        /// grounded facts. OFF by default: the text-only extraction path stays byte-identical to
        /// today, and images are large on the wire.
        #[arg(long = "vision", env = "GLOSSA_VISION")]
        vision: bool,
        /// Sampling temperature for the extract-stage model call.
        #[arg(long = "build-temp", default_value_t = 0.8)]
        build_temp: f64,
        /// Number of chunks folded into a single extract-stage model call (default 3). Falls back
        /// to `lab.toml`'s `[tuning] chunks_per_round`, then the built-in default, when unset.
        #[arg(long = "chunks-per-round")]
        chunks_per_round: Option<usize>,
        /// Agent-loop round cap for the extract stage (default 30). Falls back to `lab.toml`'s
        /// `[tuning] max_rounds`, then the built-in default, when unset.
        #[arg(long = "max-rounds")]
        max_rounds: Option<usize>,
        /// Worker-pool size for the extract stage only (default 3); judge stays sequential
        /// (it writes directly via `apply_upsert`, not through the pool). Falls back to
        /// `lab.toml`'s `[tuning] jobs_build`, then the built-in default, when unset. `0` clamps
        /// to 1 (never zero workers).
        #[arg(long)]
        jobs: Option<usize>,
    },
    /// GEPA-optimize a corpus's `answer.md` (the answer-agent system prompt) against its
    /// `dataset.toml`, applying the winner back onto the workspace only when it strictly beats
    /// the seed prompt's full-val EM.
    Train {
        /// Corpus root (kb-style PATH resolution: explicit if given, else discovered from the
        /// current directory upward, else the current directory).
        path: Option<PathBuf>,
        /// Dataset-relative budget preset (tiny|light|medium|heavy): sizes the metric-call ceiling
        /// as mult×N (N = answerable golds) and the candidate cap. Mutually exclusive with
        /// --max-metric-calls / --max-full-evals; defaults to `light` when none is passed.
        #[arg(long, value_enum)]
        auto: Option<train::AutoPreset>,
        /// Raw hard ceiling on total reader rollouts (metric calls) for the search — used verbatim.
        /// Mutually exclusive with --auto / --max-full-evals.
        #[arg(long = "max-metric-calls")]
        max_metric_calls: Option<usize>,
        /// Budget as K full passes over the N answerable golds (max_metric_calls = K×N). Mutually
        /// exclusive with --auto / --max-metric-calls.
        #[arg(long = "max-full-evals")]
        max_full_evals: Option<usize>,
        /// Minibatch size for per-candidate rollouts (default 3).
        #[arg(long)]
        minibatch: Option<usize>,
        /// Fraction of the dataset held out as the full-validation split (default 0.2).
        #[arg(long = "val-frac")]
        val_frac: Option<f64>,
        /// Max size of the Pareto frontier retained across candidates (default 12).
        #[arg(long = "pareto-size")]
        pareto_size: Option<usize>,
        /// Candidate-selection strategy (e.g. "pareto").
        #[arg(long = "candidate-selection", default_value = "pareto")]
        candidate_selection: String,
        /// Selection metric: "judge" (the default; graded LLM judge scoring
        /// Correct=1.0/Partial=0.5/Wrong=0.0 via lab.toml's [judge] endpoint + judge.md) or
        /// "exact" (exact-match EM; use for datasets with short factoid answers).
        #[arg(long, default_value = "judge")]
        metric: String,
        /// Override the workspace's default `dataset.toml`.
        #[arg(long)]
        dataset: Option<PathBuf>,
        /// Override the workspace's default `answer.md` (the seed prompt to optimize).
        #[arg(long)]
        prompt: Option<PathBuf>,
        /// Override the workspace's default `reflect.md` (the reflector's system prompt).
        #[arg(long = "reflect-prompt")]
        reflect_prompt: Option<PathBuf>,
        /// Run tag (report dir name under runs/). Default: a generated tag.
        #[arg(long)]
        tag: Option<String>,
        /// Seed the run's RNG explicitly (default: derived from the tag).
        #[arg(long = "rng-seed")]
        rng_seed: Option<u64>,
        /// Never copy the winning prompt back onto the workspace's `answer.md` — dry-run/inspect
        /// only, still writes `runs/<tag>/answer.md`.
        #[arg(long = "no-apply")]
        no_apply: bool,
        /// Never draw the progress bar, even on a TTY.
        #[arg(long = "no-progress")]
        no_progress: bool,
        /// Worker-pool size for concurrent read-only rollouts (default 3). Falls back to
        /// `lab.toml`'s `[tuning] jobs_train`, then the built-in default, when unset. `0` clamps
        /// to 1 (never zero workers).
        #[arg(long)]
        jobs: Option<usize>,
    },
    /// Phase-2 of graph construction: backward query-side synthesis (one `chain_one_seed` pass per
    /// grounded terminal, fan-out), checkpointed for `--resume`, then finalize.
    Reason {
        /// Corpus root (kb-style PATH resolution: explicit if given, else discovered from the
        /// current directory upward, else the current directory).
        path: Option<PathBuf>,
        /// Restrict seeds to this node_type (default: the ontology's grounding-required types).
        #[arg(long = "seed-type")]
        seed_type: Option<String>,
        /// Soft cap on predecessors synthesized per backward step (default 3).
        #[arg(long = "fanout-max")]
        fanout_max: Option<usize>,
        /// Agent-loop round cap for one seed's backward-synth pass (default 30). Falls back to
        /// `lab.toml`'s `[tuning] max_rounds`, then the built-in default, when unset.
        #[arg(long = "max-rounds")]
        max_rounds: Option<usize>,
        /// Only process the first N (in seed-pool order) grounded seeds.
        #[arg(long)]
        limit: Option<usize>,
        /// Clear this run's checkpoint first — a true full rebuild of the typed layer's seed marks.
        #[arg(long)]
        force: bool,
        /// Skip seeds already recorded done in the reason checkpoint.
        #[arg(long)]
        resume: bool,
        /// Never draw the progress bar, even on a TTY.
        #[arg(long = "no-progress")]
        no_progress: bool,
        /// Worker-pool size for seed workers (default 3). Falls back to `lab.toml`'s `[tuning]
        /// jobs_reason`, then the built-in default, when unset. `0` clamps to 1 (never zero
        /// workers).
        #[arg(long)]
        jobs: Option<usize>,
    },
    /// Densify the graph with the strong model (`lab.distil`), adding the reasoning the weak
    /// build+reason pass missed — the default. `--emit-golds <file>` instead runs the synthetic
    /// (question, answer) gold generator (the former default): seed from a grounded node, explore
    /// the corpus read-only, propose one gated gold per attempt, and write the kept ones to that
    /// file.
    Distil {
        /// Corpus root (kb-style PATH resolution: explicit if given, else discovered from the
        /// current directory upward, else the current directory).
        path: Option<PathBuf>,
        /// Run the synthetic gold generator instead of densify, writing kept `(question, answer)`
        /// golds to this dataset TOML — the same shape `kbx eval --dataset`/`kbx reason --gold`
        /// read back. Absent: densify runs (the default).
        #[arg(long = "emit-golds")]
        emit_golds: Option<PathBuf>,
        /// Restrict densify to a single document (its corpus-relative path). Densify mode only.
        #[arg(long)]
        doc: Option<String>,
        /// Clear this run's densify checkpoint first — a full rebuild of the densify pass.
        /// Densify mode only.
        #[arg(long)]
        force: bool,
        /// Skip documents already recorded done in the densify checkpoint. Densify mode only.
        #[arg(long)]
        resume: bool,
        /// Number of chunks folded into a single densify round (default 3). Falls back to
        /// `lab.toml`'s `[tuning] chunks_per_round`, then the built-in default, when unset.
        /// Densify mode only.
        #[arg(long = "chunks-per-round")]
        chunks_per_round: Option<usize>,
        /// Agent-loop round cap for the densify pass (default 30). Falls back to `lab.toml`'s
        /// `[tuning] max_rounds`, then the built-in default, when unset. Densify mode only.
        #[arg(long = "max-rounds")]
        max_rounds: Option<usize>,
        /// Number of synthetic golds to ATTEMPT (the gate may drop some). Use this OR `--target`.
        /// Golds mode only (`--emit-golds`).
        #[arg(long)]
        count: Option<usize>,
        /// Keep generating until this many golds are KEPT (accumulate to a target), instead of a
        /// fixed attempt count. Bounded by `--max-attempts` so a stubborn gate can't loop forever.
        /// Golds mode only.
        #[arg(long)]
        target: Option<usize>,
        /// Attempt ceiling when `--target` is set (default: target x 4). Stops and reports the
        /// shortfall honestly if the target isn't reached within this many attempts. Golds mode
        /// only.
        #[arg(long = "max-attempts")]
        max_attempts: Option<usize>,
        /// Dataset TOML to write (default `<kbx>/dataset.synthetic.toml`); overridden by
        /// `--emit-golds` when both are given. Always overwritten. Golds mode only.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Restrict seeds to this node_type (default: the ontology's grounding-required types,
        /// or every non-structural declared type when none are marked `requires_grounding`).
        /// Golds mode only.
        #[arg(long = "seed-type")]
        seed_type: Option<String>,
        /// Never draw the progress bar, even on a TTY.
        #[arg(long = "no-progress")]
        no_progress: bool,
        /// Worker-pool size for the densify pass (default 3). Falls back to `lab.toml`'s
        /// `[tuning] jobs_distil`, then the built-in default, when unset. `0` clamps to 1 (never
        /// zero workers). Densify mode only.
        #[arg(long)]
        jobs: Option<usize>,
        /// Run the CHAIN-driven alias enricher instead of densify: add search aliases to
        /// alias-poor reasoning nodes so `glossary`/`resolve` match how users phrase questions.
        /// Ignored when `--emit-golds` is also given (golds mode wins).
        #[arg(long = "aliases-only")]
        aliases_only: bool,
        /// A reasoning node is "alias-poor" (enriched by `--aliases-only`) when it has fewer than
        /// this many aliases (default 3). Alias mode only.
        #[arg(long = "min-aliases", default_value_t = 3)]
        min_aliases: usize,
        /// Skip already-well-covered terminals: a terminal fed by this many or more existing
        /// incoming chaining chains is dropped from the seed pool, spreading generation to
        /// under-covered terminals and avoiding near-duplicate golds. Unset = no filter. Golds
        /// mode only (`--emit-golds`).
        #[arg(long = "max-chains")]
        max_chains: Option<usize>,
    },
    /// Operate on `dataset.toml`-shape files (stat / merge / validate / dedup / sample). All
    /// subcommands read+write through the single case parser, round-tripping every `[[case]]`
    /// field.
    Dataset {
        #[command(subcommand)]
        cmd: DatasetCmd,
    },
}

/// `kbx dataset` subcommands — pure file ops on the `[[case]]` dataset format (logic lives in
/// `kb_eval::dataset_ops`; this layer only parses args and prints).
#[derive(Subcommand)]
enum DatasetCmd {
    /// Print counts/breakdowns for a dataset file (read-only): hop_type, answerable, needs_graph,
    /// alias coverage, duplicate/blank counts, and question/answer length min/median/max.
    Stat {
        /// Dataset TOML to inspect.
        file: PathBuf,
    },
    /// Append `--from`'s cases into `--into`: dedup by normalized question, re-id colliding ids,
    /// back `--into` up to `<into>.bak`, then write the merged set.
    Merge {
        /// Source dataset whose cases are appended.
        #[arg(long)]
        from: PathBuf,
        /// Destination dataset, merged in place (backed up to `<into>.bak` first).
        #[arg(long)]
        into: PathBuf,
    },
    /// Check a dataset (non-empty q/a, valid hop_type, unique ids); exit non-zero on any issue.
    Validate {
        /// Dataset TOML to validate.
        file: PathBuf,
    },
    /// Remove normalized-duplicate questions (keep first), backing up to `<file>.bak` first.
    Dedup {
        /// Dataset TOML to dedup in place.
        file: PathBuf,
    },
    /// Print N cases chosen with a seeded RNG (reproducible).
    Sample {
        /// Dataset TOML to sample from.
        file: PathBuf,
        /// Number of cases to print (>= total prints all).
        #[arg(short = 'n')]
        n: usize,
        /// RNG seed for a reproducible sample (default 0).
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Init { path, force } => {
            let root = workspace::resolve(path).root;
            let paths = scaffold_init(&root, force)?;
            println!("initialized kbx workspace at {}", paths.kbx_dir.display());
            Ok(())
        }
        Cmd::Eval {
            path,
            tag,
            dataset,
            prompt,
            judge: judge_path,
            limit,
            tag_filter,
            no_judge,
            resume,
            no_progress,
            jobs,
            capture,
            samples,
            no_gold,
            answers,
        } => run_eval(EvalArgs {
            path,
            tag,
            dataset,
            prompt,
            judge: judge_path,
            limit,
            tag_filter,
            no_judge,
            resume,
            no_progress,
            jobs,
            capture,
            samples,
            no_gold,
            answers,
        }),
        Cmd::Export {
            path,
            from,
            format,
            out,
            shape,
            include_partial,
            max_pairs,
            dpo_focus,
            sft_prefer_signal,
        } => run_export(ExportArgs {
            path,
            from,
            format,
            out,
            shape,
            include_partial,
            max_pairs,
            dpo_focus,
            sft_prefer_signal,
        }),
        Cmd::Build {
            path,
            stage,
            doc,
            limit,
            force,
            resume,
            no_progress,
            bridge_max_facts,
            vision,
            build_temp,
            chunks_per_round,
            max_rounds,
            jobs,
        } => {
            let paths = workspace::resolve(path);
            let report = run_build(
                paths,
                BuildOpts {
                    stage,
                    doc,
                    limit,
                    force,
                    resume,
                    no_progress,
                    bridge_max_facts,
                    vision,
                    build_temp,
                    chunks_per_round,
                    max_rounds,
                    jobs,
                },
            )?;
            println!(
                "build report: {} doc(s) extracted, {} group(s) judged",
                report.docs_extracted.len(),
                report.groups_judged
            );
            Ok(())
        }
        Cmd::Train {
            path,
            auto,
            max_metric_calls,
            max_full_evals,
            minibatch,
            val_frac,
            pareto_size,
            candidate_selection,
            metric,
            dataset,
            prompt,
            reflect_prompt,
            tag,
            rng_seed,
            no_apply,
            no_progress,
            jobs,
        } => train::run_train(
            path,
            TrainArgs {
                auto,
                max_metric_calls,
                max_full_evals,
                minibatch,
                val_frac,
                pareto_size,
                candidate_selection,
                dataset,
                prompt,
                reflect_prompt,
                tag,
                rng_seed,
                no_apply,
                no_progress,
                jobs,
                metric,
            },
        ),
        Cmd::Reason {
            path,
            seed_type,
            fanout_max,
            max_rounds,
            limit,
            force,
            resume,
            no_progress,
            jobs,
        } => reason::run_reason(
            path,
            ReasonArgs {
                seed_type,
                fanout_max,
                max_rounds,
                limit,
                force,
                resume,
                no_progress,
                jobs,
            },
        ),
        Cmd::Distil {
            path,
            emit_golds,
            doc,
            force,
            resume,
            chunks_per_round,
            max_rounds,
            count,
            target,
            max_attempts,
            out,
            seed_type,
            no_progress,
            jobs,
            aliases_only,
            min_aliases,
            max_chains,
        } => distil::run(
            path,
            DistilArgs {
                count,
                target,
                max_attempts,
                out,
                seed_type,
                no_progress,
                doc,
                force,
                resume,
                chunks_per_round,
                max_rounds,
                emit_golds,
                jobs,
                aliases_only,
                min_aliases,
                max_chains,
            },
        ),
        Cmd::Dataset { cmd } => run_dataset(cmd),
    }
}

struct EvalArgs {
    path: Option<PathBuf>,
    tag: Option<String>,
    dataset: Option<PathBuf>,
    prompt: Option<PathBuf>,
    judge: Option<PathBuf>,
    limit: Option<usize>,
    tag_filter: Option<String>,
    no_judge: bool,
    resume: bool,
    no_progress: bool,
    jobs: Option<usize>,
    /// Capture full chat trajectories to `runs/<tag>/trajectories.jsonl`.
    capture: bool,
    /// With `--capture`, samples per case (varied trajectories for DPO). Default 1.
    samples: usize,
    /// Predict-only: run the agent without scoring (no judge, keep unanswerable, tolerate empty gold).
    no_gold: bool,
    /// Optional path for a question->answer CSV deliverable (relative resolves under runs/<tag>/).
    answers: Option<PathBuf>,
}

struct ExportArgs {
    path: Option<PathBuf>,
    from: String,
    format: ExportFormat,
    out: PathBuf,
    shape: ExportShape,
    include_partial: bool,
    max_pairs: usize,
    dpo_focus: Option<DpoFocus>,
    sft_prefer_signal: bool,
}

/// The concrete files/dirs an `eval` run reads from and writes to, after folding the workspace's
/// `KbxPaths` defaults with any `--dataset`/`--prompt`/`--judge` CLI overrides. `root` doubles as
/// the corpus dir passed to the agent backend and recorded in `RunMeta`.
struct EvalPaths {
    root: PathBuf,
    dataset: PathBuf,
    prompt: PathBuf,
    judge_prompt: PathBuf,
    runs: PathBuf,
}

/// Fold `KbxPaths` (the `.glossa/kbx/` layout) with CLI overrides into the paths an `eval` run
/// actually uses. `root` (the corpus) is never overridable — it IS the kb-style PATH the whole
/// workspace resolved against.
fn resolve_eval_paths(
    paths: &KbxPaths,
    dataset: Option<PathBuf>,
    prompt: Option<PathBuf>,
    judge: Option<PathBuf>,
) -> EvalPaths {
    EvalPaths {
        root: paths.root.clone(),
        dataset: dataset.unwrap_or_else(|| paths.dataset.clone()),
        prompt: prompt.unwrap_or_else(|| paths.answer.clone()),
        judge_prompt: judge.unwrap_or_else(|| paths.judge.clone()),
        runs: paths.runs.clone(),
    }
}

/// Fallback worker-pool size for `kbx eval`'s per-case loop when neither `--jobs` nor `lab.toml`'s
/// `[tuning] jobs_eval` overrides it — matches the other stages' `DEFAULT_JOBS` (3).
const DEFAULT_JOBS_EVAL: usize = 3;

/// Graded score a `Verdict` maps to for the TensorZero feedback metric (`Correct=1.0/Partial=0.5/
/// Wrong=0.0` — same scale `backend::tensorzero::TensorZeroBackend::judge_score` already uses).
/// `Unscored` never reaches this function (the feedback post is gated on `verdict != Unscored` at
/// the call site), so it's given the conservative 0.0 rather than treated as a real case.
fn verdict_score(v: Verdict) -> f32 {
    match v {
        Verdict::Correct => 1.0,
        Verdict::Partial => 0.5,
        Verdict::Wrong | Verdict::Unscored => 0.0,
    }
}

fn run_eval(args: EvalArgs) -> Result<()> {
    let kbx_paths = workspace::resolve(args.path);
    let lab = LabConfig::load_at(&kbx_paths.lab)
        .with_context(|| format!("loading {}", kbx_paths.lab.display()))?;
    let paths = resolve_eval_paths(&kbx_paths, args.dataset, args.prompt, args.judge);

    let answer_md = std::fs::read_to_string(&paths.prompt)
        .with_context(|| format!("reading prompt {}", paths.prompt.display()))?;
    let dataset_text = std::fs::read_to_string(&paths.dataset)
        .with_context(|| format!("reading dataset {}", paths.dataset.display()))?;
    let mut cases = parse_dataset_toml(&dataset_text)?;

    // id -> (question, gold answer) for the whole parsed dataset, built BEFORE any slicing so the
    // `--answers` CSV can join a question + gold onto every result (including `--resume`d cases
    // whose persisted `CaseResult` doesn't carry the question text).
    let qmeta: std::collections::HashMap<String, (String, String)> = cases
        .iter()
        .map(|q| (q.id.clone(), (q.question.clone(), q.answer.clone())))
        .collect();

    // Drop cases marked `answerable = false` (out-of-corpus golds) BEFORE any resume/tag/limit
    // slicing so they never enter scoring — they'd only cap the achievable metric. Absent field
    // => all `true` => nothing dropped (unchanged behavior). Skipped under `--no-gold`: there are
    // no golds to be un/answerable, so every question is run.
    if !args.no_gold {
        let before_answerable = cases.len();
        cases.retain(|q| q.answerable);
        let excluded_unanswerable = before_answerable - cases.len();
        if excluded_unanswerable > 0 {
            println!(
                "excluded {excluded_unanswerable} unanswerable (answerable=false) — not scored"
            );
        }
    }

    if let Some(t) = &args.tag_filter {
        cases.retain(|q| q.tags.iter().any(|x| x == t));
    }
    if let Some(n) = args.limit {
        cases.truncate(n);
    }

    let tag = args
        .tag
        .clone()
        .unwrap_or_else(|| default_tag(&paths.root, &paths.dataset));
    let runs_dir = paths.runs.clone();
    let cases_dir = runs_dir.join(&tag).join("cases");

    // --resume: whatever already has a persisted `<id>.json` under cases_dir is done — skip it and
    // only run the rest. The full report at the end still merges old + new (see below).
    let previously_done = if args.resume {
        load_cases(&cases_dir)
            .with_context(|| format!("loading prior cases from {}", cases_dir.display()))?
    } else {
        Vec::new()
    };
    if args.resume {
        let done_ids: HashSet<&str> = previously_done.iter().map(|c| c.id.as_str()).collect();
        cases.retain(|q| !done_ids.contains(q.id.as_str()));
    }

    // Simulated-user dialogue gate (opt-in): load the persona prompt only when `[user_sim]` is
    // configured. Absent -> `None` -> the reader keeps today's behavior (a text-only turn is the
    // final answer). Read once here; each per-case backend clones it.
    let user_sim_prompt = if lab.user_sim.is_some() {
        Some(
            std::fs::read_to_string(&kbx_paths.user_sim).with_context(|| {
                format!("reading user_sim prompt {}", kbx_paths.user_sim.display())
            })?,
        )
    } else {
        None
    };

    let use_judge = !args.no_judge && !args.no_gold && lab.judge.is_some();
    let judge_md = if use_judge {
        Some(
            std::fs::read_to_string(&paths.judge_prompt).with_context(|| {
                format!("reading judge prompt {}", paths.judge_prompt.display())
            })?,
        )
    } else {
        None
    };

    // Corpus index shared (read-only) across judge calls so the evidence-grounded judge can load
    // the source chunks named by each case's `source` refs. Opened only when judging; best-effort
    // (`None` on failure) — a case with no `source`, or an unavailable index, falls back to the
    // gold-only judge prompt, byte-identical to before.
    let judge_idx = if use_judge {
        glossa::index::store::DocIndex::open_or_create(&paths.root).ok()
    } else {
        None
    };

    let api_key = lab.model.resolve_key();
    let timeout = Duration::from_secs(lab.model.timeout_secs);

    // indicatif draws to stderr by default; also check stdout since some shells redirect one but
    // not the other and either being non-interactive is a good signal this run isn't at a console.
    let show_progress =
        !args.no_progress && std::io::stdout().is_terminal() && std::io::stderr().is_terminal();
    let pb = if show_progress {
        let pb = ProgressBar::new(cases.len() as u64);
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
    } else {
        ProgressBar::hidden()
    };

    reset_tokens();
    reset_resamples();
    // One static stage word at the front for the whole run — never switches. The ticker owns only
    // `{msg}` (ETA + tokens/resamples), redrawn on its own timer.
    pb.set_prefix("evaluating");
    let ticker = StatusTicker::start(&pb);

    // Worker-pool size for the per-case reader+judge loop: CLI > lab.toml `[tuning] jobs_eval` > 3,
    // clamped to at least 1. Eval scoring is READ-ONLY (graph reads are WAL-safe; the judge is a
    // stateless cloud call), so cases parallelize like `train`'s rollouts. `--jobs 1` runs the
    // closure inline in `cases` order — byte-for-byte the old sequential behavior. Each worker
    // builds its OWN `OpenAiBackend` (never shared across threads) and retrieves its own trace file
    // via `glossa::trace::last_trace_path()` (a thread-local set by `TraceLog::to_dir` on the SAME
    // thread inside `answer()`), so concurrent cases never collide on trace attribution.
    let jobs = lab::resolve(args.jobs, lab.tuning.jobs_eval, DEFAULT_JOBS_EVAL).max(1);
    // `--capture` accumulates one `TrajectoryRecord` per sampled episode across all workers into
    // this shared buffer (a `Mutex`, like the pool's own results), flushed to
    // `runs/<tag>/trajectories.jsonl` after the pool drains. Without `--capture` it stays empty and
    // the per-case path is byte-identical to before (the reader runs exactly once, via `answer`).
    let n_samples = if args.capture { args.samples.max(1) } else { 1 };
    let trajectories: Mutex<Vec<TrajectoryRecord>> = Mutex::new(Vec::new());
    let results = run_units_parallel(
        cases,
        jobs,
        &pb,
        |_q| 1,
        |q| {
            let backend = OpenAiBackend {
                endpoint: lab.model.endpoint.clone(),
                model: lab.model.model.clone(),
                api_key: api_key.clone(),
                timeout,
                use_graph: true,
                system_prompt: Some(answer_md.clone()),
                temperature: lab.model.temperature,
                user_sim: lab.user_sim.clone(),
                user_sim_prompt: user_sim_prompt.clone(),
                rate_limit: lab.model.rate_limit.clone(),
                fallback: lab.model.fallback.clone(),
                api: lab.model.api,
                function_name: lab.model.function_name.clone(),
                feedback_score_metric: lab.model.feedback_score_metric.clone(),
                feedback_bool_metric: lab.model.feedback_bool_metric.clone(),
            };

            // One reader+judge sample. `capture=false` drives the byte-identical non-capturing reader
            // (`answer`); `capture=true` drives `answer_capturing`, recording the full trajectory into
            // `episode`. Returns everything the CaseResult and a `TrajectoryRecord` both need.
            type Sample = (
                String,                                        // answer
                Vec<String>,                                   // tools (deduped names)
                String,                                        // transcript
                f32,                                           // em
                f32,                                           // f1
                Verdict,                                       // verdict (reward)
                String,                                        // reason
                String,                                        // judge_raw
                kb_eval::backend::agent_loop::CapturedEpisode, // trajectory (empty unless captured)
                bool,                                          // errored (reader endpoint failure)
            );
            let run_sample = |capture: bool| -> Sample {
                // Reset THIS worker thread's TZ episode grouping before the reader runs, so a stale
                // episode id left over from a PRIOR case on the same thread (or the previous sample
                // of THIS case) never leaks into this rollout — mirrors `TraceLog::to_dir`'s own
                // per-call thread-local reset, just done explicitly here since episode-setting lives
                // inside `TzTransport::call`, not the backend constructor. A no-op off TensorZero
                // (nothing ever calls `kb_eval::episode::set`, so `current()` stays `None`).
                kb_eval::episode::reset();
                let mut episode = kb_eval::backend::agent_loop::CapturedEpisode::default();
                // `errored` = the reader's ENDPOINT/TRANSPORT failed (500, context-length overflow,
                // …) so it never produced an answer. This is NOT a wrong answer: the case is marked
                // errored, the judge is skipped, and the report EXCLUDES it from the graded-quality
                // denominator (see `report::tally`) instead of scoring it 0.0. A reader that returns
                // wrong/empty TEXT is `Ok` here (errored=false) and scored normally.
                let (answer, errored) = if capture {
                    match backend.answer_capturing(&paths.root, q, Some(&mut episode)) {
                        Ok(a) => (a, false),
                        Err(e) => {
                            pb.println(format!("case {}: agent error: {e}", q.id));
                            (format!("(error: {e})"), true)
                        }
                    }
                } else {
                    match backend.answer(&paths.root, q) {
                        Ok(a) => (a, false),
                        Err(e) => {
                            pb.println(format!("case {}: agent error: {e}", q.id));
                            (format!("(error: {e})"), true)
                        }
                    }
                };

                // This sample's exact trace file — the one `answer*()` created on THIS worker thread —
                // read straight from the thread-local rather than diffing the trace dir (which races
                // under concurrency). Best-effort: empty when tracing was disabled / no file recorded.
                let (tools, transcript) = match glossa::trace::last_trace_path() {
                    Some(p) => parse_trace_file(&p),
                    None => (Vec::new(), String::new()),
                };

                let golds = gold_forms(q);
                // Endpoint-errored rollouts produced no answer — no EM/F1 sample (0.0) and the
                // report excludes them from those denominators too.
                let em = if !errored && relaxed_match_any(&answer, &golds) {
                    1.0
                } else {
                    0.0
                };
                let f1 = if errored {
                    0.0
                } else {
                    token_f1_any(&answer, &golds)
                };

                // On an endpoint error, skip the judge entirely (there is no answer to grade) and
                // record an Unscored verdict; the case is flagged `errored` so the report surfaces it
                // as an excluded count rather than a scored 0.0.
                let (verdict, reason, judge_raw) = if errored {
                    (
                        Verdict::Unscored,
                        "reader endpoint error (excluded from scoring)".to_string(),
                        String::new(),
                    )
                } else {
                    match (&judge_md, &lab.judge) {
                        (Some(jmd), Some(jep)) => match judge(
                            jep,
                            jmd,
                            &q.question,
                            &q.answer,
                            &answer,
                            &q.source,
                            judge_idx.as_ref(),
                        ) {
                            Ok(Judgement {
                                verdict,
                                reason,
                                raw,
                            }) => (verdict, reason, raw),
                            Err(e) => (
                                Verdict::Unscored,
                                format!("judge error: {e}"),
                                String::new(),
                            ),
                        },
                        _ => (Verdict::Unscored, String::new(), String::new()),
                    }
                };

                // TensorZero episode feedback: post the judge verdict on the episode this rollout
                // grouped into. `episode::current()` is `Some` ONLY when the reader's transport is
                // `TzTransport` (every other transport never calls `episode::set`) AND a judge
                // actually scored this rollout (`Unscored` — no judge configured, or the judge call
                // itself errored — posts nothing rather than a misleading 0.0). Off-TZ this whole
                // block is inert: `current()` is always `None`, so `post_feedback` is never reached.
                if verdict != Verdict::Unscored {
                    if let Some(eid) = kb_eval::episode::current() {
                        let score_metric = lab.model.feedback_score_metric();
                        let bool_metric = lab.model.feedback_bool_metric();
                        backend.post_feedback(
                            &eid,
                            &[
                                (
                                    score_metric.as_str(),
                                    serde_json::json!(verdict_score(verdict)),
                                ),
                                (
                                    bool_metric.as_str(),
                                    serde_json::json!(verdict == Verdict::Correct),
                                ),
                            ],
                        );
                    }
                }

                (
                    answer, tools, transcript, em, f1, verdict, reason, judge_raw, episode, errored,
                )
            };

            // Sample 0 produces the reported CaseResult (and its trajectory when capturing).
            let (answer, tools, transcript, em, f1, verdict, reason, judge_raw, episode, errored) =
                run_sample(args.capture);

            // Capture: record sample 0's trajectory + any additional samples (varied outcomes → DPO).
            if args.capture {
                let push =
                    |ans: &str, ep: &kb_eval::backend::agent_loop::CapturedEpisode, v: Verdict| {
                        trajectories
                            .lock()
                            .expect("trajectories mutex poisoned")
                            .push(TrajectoryRecord {
                                id: q.id.clone(),
                                question: q.question.clone(),
                                model: lab.model.model.clone(),
                                tools: ep.tools.clone().unwrap_or(serde_json::Value::Null),
                                messages: ep.messages.clone(),
                                answer: ans.to_string(),
                                verdict: v,
                                hop_type: q.hop_type.clone(),
                            });
                    };
                push(&answer, &episode, verdict);
                for _ in 1..n_samples {
                    let s = run_sample(true);
                    push(&s.0, &s.8, s.5);
                }
            }

            // The graded JUDGE verdict is the real signal (EM is ~0 on paragraph answers); lead with
            // the question's `hop_type` when present. `em`/`f1` still flow into `CaseResult` for the
            // end-of-run report's secondary numbers.
            let type_tag = if q.hop_type.is_empty() {
                String::new()
            } else {
                format!(" [{}]", q.hop_type)
            };
            pb.println(format!("case {}{type_tag}: {verdict:?}", q.id));

            let r = CaseResult {
                id: q.id.clone(),
                verdict,
                reason,
                f1,
                em,
                tools,
                answer,
                transcript,
                judge_raw,
                hop_type: q.hop_type.clone(),
                needs_graph: q.needs_graph.clone(),
                errored,
            };
            write_case(&cases_dir, &r)
                .with_context(|| format!("persisting case {} to {}", r.id, cases_dir.display()))?;
            Ok(r)
        },
    )?;
    drop(ticker); // stop before finish_and_clear so it can't redraw a message onto a cleared bar
    pb.finish_and_clear();

    // Flush captured trajectories (reward-joined) to `runs/<tag>/trajectories.jsonl` for
    // `kbx export`. No-op without `--capture` (the buffer is empty).
    if args.capture {
        let recs = trajectories
            .into_inner()
            .expect("trajectories mutex poisoned");
        let tpath = write_trajectories(&runs_dir.join(&tag), &recs)?;
        println!(
            "captured {} trajectories -> {}",
            recs.len(),
            tpath.display()
        );
    }

    // Merge prior (resumed) cases with the ones just run. Dedup by id — a case just re-run wins
    // over its stale, previously-persisted copy.
    let mut all_results = previously_done;
    let new_ids: HashSet<&str> = results.iter().map(|r| r.id.as_str()).collect();
    all_results.retain(|c| !new_ids.contains(c.id.as_str()));
    all_results.extend(results);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string();
    let meta = RunMeta {
        model: lab.model.model.clone(),
        judge: lab
            .judge
            .as_ref()
            .map(|j| j.model.clone())
            .unwrap_or_default(),
        corpus: paths.root.display().to_string(),
        n: all_results.len(),
        timestamp,
    };
    let report_path = write_run(&runs_dir, &tag, &meta, &all_results)?;
    if args.no_gold {
        // Predict-only: no golds to score against, so the graded/lexical summaries are meaningless.
        let answered = all_results
            .iter()
            .filter(|r| !r.answer.trim().is_empty())
            .count();
        let errored = all_results.iter().filter(|r| r.errored).count();
        println!(
            "predict-only: {} question(s), {answered} answered, {errored} endpoint-errored",
            all_results.len()
        );
    } else {
        println!("{}", summary_text(&all_results));
        println!("{}", lexical_text(&all_results));
    }

    // `--answers`: flat question->answer CSV deliverable. A relative path lands under runs/<tag>/
    // (inside .glossa, never the indexed corpus); an absolute path is used verbatim.
    if let Some(ap) = &args.answers {
        let out_path = if ap.is_absolute() {
            ap.clone()
        } else {
            runs_dir.join(&tag).join(ap)
        };
        let rows: Vec<AnswerRow> = all_results
            .iter()
            .map(|r| {
                let (question, gold) = qmeta.get(&r.id).cloned().unwrap_or_default();
                AnswerRow {
                    id: r.id.clone(),
                    question,
                    answer: r.answer.clone(),
                    gold,
                    verdict: if use_judge {
                        format!("{:?}", r.verdict)
                    } else {
                        String::new()
                    },
                }
            })
            .collect();
        let written = write_answers_csv(&out_path, &rows, !args.no_gold)?;
        println!("wrote answers -> {}", written.display());
    }

    let footnote = if cache_is_estimated() {
        " (cache estimated from prompt re-send)"
    } else {
        ""
    };
    println!("tokens: {}{footnote}", token_summary());
    println!("wrote {}", report_path.display());
    Ok(())
}

/// `kbx export`: post-process captured trajectories into an Unsloth-ready JSONL dataset.
/// Pure, deterministic, network-free — it only reads `runs/<tag>/trajectories.jsonl` for each tag
/// in `--from`, filters/pairs by the joined judge verdict, and writes `--out`.
fn run_export(args: ExportArgs) -> Result<()> {
    let runs_dir = workspace::resolve(args.path).runs;
    let tags: Vec<String> = args
        .from
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let mut records: Vec<TrajectoryRecord> = Vec::new();
    for tag in &tags {
        records.extend(load_trajectories_for_tag(&runs_dir, tag).with_context(|| {
            format!(
                "loading trajectories for tag '{tag}' under {}",
                runs_dir.display()
            )
        })?);
    }

    let (rows, summary) = match args.format {
        ExportFormat::Sft => {
            let shape = match args.shape {
                ExportShape::Messages => SftShape::Messages,
                ExportShape::Sharegpt => SftShape::Sharegpt,
            };
            let out = export_sft(
                &records,
                shape,
                args.include_partial,
                args.sft_prefer_signal,
            );
            let summary = format!(
                "SFT: {} lines from {} trajectories ({} tag(s)), {} signal-reaction",
                out.rows.len(),
                records.len(),
                tags.len(),
                out.signal_reaction
            );
            (out.rows, summary)
        }
        ExportFormat::Dpo => {
            let focus_plateau = matches!(args.dpo_focus, Some(DpoFocus::Plateau));
            let out = export_dpo(&records, args.max_pairs.max(1), focus_plateau);
            let skip_reason = if focus_plateau {
                "lacked both classes or no spiraled-past-plateau rejected"
            } else {
                "lacked both classes"
            };
            let summary = format!(
                "DPO: {} pairs ({} plateau-contrastive), {} question(s) skipped ({}), from {} trajectories",
                out.pairs.len(),
                out.plateau_contrastive,
                out.questions_skipped,
                skip_reason,
                records.len()
            );
            (out.pairs, summary)
        }
    };

    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating output dir {}", parent.display()))?;
        }
    }
    std::fs::write(&args.out, to_jsonl(&rows))
        .with_context(|| format!("writing {}", args.out.display()))?;
    println!("{summary}");
    println!("wrote {}", args.out.display());
    Ok(())
}

/// `kbx dataset` dispatch: thin arg-parse -> `dataset_ops` -> print. `validate` returns an `Err`
/// (non-zero exit) when the file has any issue, so it doubles as a CI/pre-merge gate.
fn run_dataset(cmd: DatasetCmd) -> Result<()> {
    match cmd {
        DatasetCmd::Stat { file } => {
            let cases = dataset_ops::load_cases(&file)?;
            let s = dataset_ops::compute_stat(&cases);
            let pct = |n: usize| {
                if s.total == 0 {
                    0.0
                } else {
                    100.0 * n as f64 / s.total as f64
                }
            };
            println!("cases: {}", s.total);
            println!(
                "hop_type: lexical {} ({:.0}%), multihop {} ({:.0}%), untyped {} ({:.0}%)",
                s.lexical,
                pct(s.lexical),
                s.multihop,
                pct(s.multihop),
                s.untyped,
                pct(s.untyped)
            );
            println!(
                "answerable: {} ({:.0}%), unanswerable: {}",
                s.answerable,
                pct(s.answerable),
                s.unanswerable
            );
            let ng = s
                .needs_graph
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(", ");
            println!("needs_graph: {ng}");
            println!(
                "aliases: {} with, {} without",
                s.with_aliases, s.without_aliases
            );
            println!(
                "duplicates: {} question(s), {} answer(s)",
                s.dup_questions, s.dup_answers
            );
            println!("blank question/answer: {}", s.blank);
            println!(
                "question chars: min {} / median {} / max {}",
                s.q_len.min, s.q_len.median, s.q_len.max
            );
            println!(
                "answer chars: min {} / median {} / max {}",
                s.a_len.min, s.a_len.median, s.a_len.max
            );
            Ok(())
        }
        DatasetCmd::Merge { from, into } => {
            let summary = dataset_ops::merge_files(&from, &into)?;
            println!(
                "merge: from={}, added={}, skipped_dup={}, total={} (backed up {} -> {})",
                summary.from,
                summary.added,
                summary.skipped_dup,
                summary.total,
                into.display(),
                dataset_ops::backup_path(&into).display()
            );
            Ok(())
        }
        DatasetCmd::Validate { file } => {
            let cases = dataset_ops::load_cases(&file)?;
            let issues = dataset_ops::validate_cases(&cases);
            if issues.is_empty() {
                println!("ok: {} cases", cases.len());
                Ok(())
            } else {
                for i in &issues {
                    println!("case {}: {}", i.id, i.problem);
                }
                anyhow::bail!("{} issue(s) found in {}", issues.len(), file.display())
            }
        }
        DatasetCmd::Dedup { file } => {
            let (removed, total) = dataset_ops::dedup_file(&file)?;
            println!(
                "dedup: removed {removed}, {total} remain (backed up {} -> {})",
                file.display(),
                dataset_ops::backup_path(&file).display()
            );
            Ok(())
        }
        DatasetCmd::Sample { file, n, seed } => {
            let cases = dataset_ops::load_cases(&file)?;
            let chosen = dataset_ops::sample_cases(&cases, n, seed);
            for c in &chosen {
                let hop = if c.hop_type.is_empty() {
                    "(untyped)"
                } else {
                    c.hop_type.as_str()
                };
                println!("{} [{}] {}", c.id, hop, c.question);
                println!("    -> {}", truncate_chars(&c.answer, 100));
            }
            Ok(())
        }
    }
}

/// Truncate `s` to at most `max` chars, appending an ellipsis when cut — keeps `sample`'s answer
/// column readable without wrapping the terminal on a paragraph-length gold.
fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let head: String = s.chars().take(max).collect();
    format!("{head}…")
}

/// Gold answer forms accepted for scoring: the primary `answer` plus any `answer_aliases` —
/// mirrors the `golds` vector `run::eval_one` builds for the same reason (MuSiQue/SQuAD-style
/// alias-aware EM/F1).
fn gold_forms(q: &Question) -> Vec<String> {
    std::iter::once(q.answer.clone())
        .chain(q.answer_aliases.iter().cloned())
        .collect()
}

/// Parse a single case's trace file (the exact path `TraceLog::to_dir` created on this case's
/// worker thread, retrieved via `glossa::trace::last_trace_path`) into (deduped tool names in
/// first-seen order, raw JSONL trace text). Best-effort: an empty pair when the file is missing or
/// unreadable (e.g. the corpus has tracing disabled) — the run stays functional, it just carries no
/// tool/transcript detail for that case. Attributing by the known path (not a directory diff) is
/// race-free under the parallel case pool.
fn parse_trace_file(path: &Path) -> (Vec<String>, String) {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let mut tools = Vec::new();
    let mut seen = HashSet::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(t) = v.get("tool").and_then(|t| t.as_str()) {
                if seen.insert(t.to_string()) {
                    tools.push(t.to_string());
                }
            }
        }
    }
    (tools, text)
}

/// Stable slug: `slug(root basename)-slug(dataset stem)` — no clock in it, so repeat runs against
/// the same corpus/dataset land in the same `runs/<tag>/` unless the caller passes `--tag` (the
/// run's own timestamp still lives in `RunMeta`/the report body).
fn default_tag(root: &Path, dataset: &Path) -> String {
    let root_name = root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("workspace");
    let ds = dataset
        .file_stem()
        .and_then(|n| n.to_str())
        .unwrap_or("dataset");
    format!("{}-{}", slugify(root_name), slugify(ds))
}

fn slugify(s: &str) -> String {
    let mut out = String::new();
    let mut last_dash = false;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_collapses_non_alnum_runs() {
        assert_eq!(slugify("My Workspace!!"), "my-workspace");
        assert_eq!(slugify("dataset.toml"), "dataset-toml");
        assert_eq!(slugify("__lead_trail__"), "lead-trail");
    }

    #[test]
    fn default_tag_combines_workspace_and_dataset_stem() {
        let ws = Path::new("/a/b/My Lab");
        let ds = Path::new("/a/b/My Lab/cases.toml");
        assert_eq!(default_tag(ws, ds), "my-lab-cases");
    }

    #[test]
    fn gold_forms_includes_primary_and_aliases() {
        let q = Question {
            id: "q1".into(),
            question: "?".into(),
            answer: "Bob".into(),
            answer_aliases: vec!["Robert".into()],
            ..Default::default()
        };
        assert_eq!(
            gold_forms(&q),
            vec!["Bob".to_string(), "Robert".to_string()]
        );
    }

    /// `kbx eval`'s path resolver: given a scaffolded `.glossa/kbx/` workspace, the corpus is
    /// `paths.root` (the workspace root itself, not the kbx subdir) and the dataset defaults to
    /// the one under `.glossa/kbx/` — unless a CLI override is given.
    #[test]
    fn eval_path_resolver_uses_root_as_corpus_and_kbx_dataset() {
        let dir = tempfile::tempdir().unwrap();
        let scaffolded = scaffold_init(dir.path(), false).unwrap();

        let kbx_paths = workspace::resolve(Some(dir.path().to_path_buf()));
        let resolved = resolve_eval_paths(&kbx_paths, None, None, None);

        assert_eq!(resolved.root, dir.path());
        assert_eq!(resolved.dataset, scaffolded.dataset);
        assert!(resolved
            .dataset
            .starts_with(dir.path().join(".glossa").join("kbx")));
        assert_eq!(resolved.prompt, scaffolded.answer);
        assert_eq!(resolved.judge_prompt, scaffolded.judge);
        assert_eq!(resolved.runs, scaffolded.runs);
    }

    #[test]
    fn eval_path_resolver_honors_cli_overrides() {
        let dir = tempfile::tempdir().unwrap();
        scaffold_init(dir.path(), false).unwrap();
        let kbx_paths = workspace::resolve(Some(dir.path().to_path_buf()));

        let custom_dataset = dir.path().join("custom-dataset.toml");
        let resolved = resolve_eval_paths(&kbx_paths, Some(custom_dataset.clone()), None, None);

        assert_eq!(resolved.dataset, custom_dataset);
        assert_eq!(resolved.root, dir.path());
    }

    #[test]
    fn parse_trace_file_is_empty_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nope.jsonl");
        let (tools, transcript) = parse_trace_file(&missing);
        assert!(tools.is_empty());
        assert!(transcript.is_empty());
    }

    #[test]
    fn parse_trace_file_dedups_tools_first_seen_order() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("200-1-0.jsonl");
        std::fs::write(
            &file,
            concat!(
                "{\"ts_ms\":2,\"tool\":\"read\",\"args\":{},\"result\":{}}\n",
                "{\"ts_ms\":3,\"tool\":\"read\",\"args\":{},\"result\":{}}\n",
                "{\"ts_ms\":4,\"tool\":\"search\",\"args\":{},\"result\":[]}\n",
            ),
        )
        .unwrap();

        let (tools, transcript) = parse_trace_file(&file);
        assert_eq!(tools, vec!["read".to_string(), "search".to_string()]);
        assert!(transcript.contains("read") && transcript.contains("search"));
    }

    #[test]
    fn eval_cmd_parses_jobs_flag() {
        let cli = Cli::try_parse_from(["kbx", "eval", "--jobs", "5"]).unwrap();
        match cli.cmd {
            Cmd::Eval { jobs, .. } => assert_eq!(jobs, Some(5)),
            _ => panic!("expected Cmd::Eval"),
        }
        let cli = Cli::try_parse_from(["kbx", "eval"]).unwrap();
        match cli.cmd {
            Cmd::Eval { jobs, .. } => assert!(jobs.is_none()),
            _ => panic!("expected Cmd::Eval"),
        }
    }

    #[test]
    fn build_cmd_defaults_stage_to_all() {
        let cli = Cli::try_parse_from(["kbx", "build"]).unwrap();
        match cli.cmd {
            Cmd::Build {
                stage,
                doc,
                limit,
                force,
                resume,
                no_progress,
                bridge_max_facts: _,
                vision,
                build_temp,
                chunks_per_round,
                max_rounds,
                jobs,
                path,
            } => {
                assert_eq!(stage, BuildStage::All);
                assert!(doc.is_none());
                assert!(limit.is_none());
                assert!(!force);
                assert!(!resume);
                assert!(!no_progress);
                assert!(!vision, "--vision must default OFF");
                assert!(
                    (build_temp - 0.8).abs() < f64::EPSILON,
                    "build_temp default should be 0.8, got {build_temp}"
                );
                assert!(
                    chunks_per_round.is_none(),
                    "unset --chunks-per-round must be None (resolved CLI>lab>default in run_build), not a clap-level 3"
                );
                assert!(
                    max_rounds.is_none(),
                    "unset --max-rounds must be None (resolved CLI>lab>default in run_build)"
                );
                assert!(
                    jobs.is_none(),
                    "unset --jobs must be None (resolved CLI>lab>default in run_build), not a clap-level 3"
                );
                assert!(path.is_none());
            }
            _ => panic!("expected Cmd::Build"),
        }
    }

    #[test]
    fn build_cmd_parses_stage_and_flags() {
        let cli = Cli::try_parse_from([
            "kbx",
            "build",
            "/corpus",
            "--stage",
            "judge",
            "--doc",
            "a.md",
            "--limit",
            "3",
            "--force",
            "--resume",
            "--no-progress",
            "--vision",
            "--build-temp",
            "0.5",
            "--chunks-per-round",
            "7",
            "--max-rounds",
            "50",
            "--jobs",
            "8",
        ])
        .unwrap();
        match cli.cmd {
            Cmd::Build {
                path,
                stage,
                doc,
                limit,
                force,
                resume,
                no_progress,
                bridge_max_facts: _,
                vision,
                build_temp,
                chunks_per_round,
                max_rounds,
                jobs,
            } => {
                assert_eq!(path, Some(PathBuf::from("/corpus")));
                assert_eq!(stage, BuildStage::Judge);
                assert_eq!(doc, Some("a.md".to_string()));
                assert_eq!(limit, Some(3));
                assert!(force);
                assert!(resume);
                assert!(no_progress);
                assert!(vision);
                assert!(
                    (build_temp - 0.5).abs() < f64::EPSILON,
                    "build_temp should parse to 0.5, got {build_temp}"
                );
                assert_eq!(chunks_per_round, Some(7));
                assert_eq!(max_rounds, Some(50));
                assert_eq!(jobs, Some(8));
            }
            _ => panic!("expected Cmd::Build"),
        }
    }

    /// `--jobs 0` must parse as `Some(0)` at the clap level (unclamped) — the `.max(1)` clamp
    /// happens at each stage's `resolve(...)` call site (see `lab::resolve` tests), not in clap.
    #[test]
    fn build_cmd_parses_jobs_zero_unclamped() {
        let cli = Cli::try_parse_from(["kbx", "build", "--jobs", "0"]).unwrap();
        match cli.cmd {
            Cmd::Build { jobs, .. } => assert_eq!(jobs, Some(0)),
            _ => panic!("expected Cmd::Build"),
        }
    }

    #[test]
    fn reason_cmd_parses_jobs_flag() {
        let cli = Cli::try_parse_from(["kbx", "reason", "--jobs", "5"]).unwrap();
        match cli.cmd {
            Cmd::Reason { jobs, .. } => assert_eq!(jobs, Some(5)),
            _ => panic!("expected Cmd::Reason"),
        }
        let cli = Cli::try_parse_from(["kbx", "reason"]).unwrap();
        match cli.cmd {
            Cmd::Reason { jobs, .. } => assert!(jobs.is_none()),
            _ => panic!("expected Cmd::Reason"),
        }
    }

    #[test]
    fn train_cmd_parses_jobs_flag() {
        let cli = Cli::try_parse_from(["kbx", "train", "--jobs", "5"]).unwrap();
        match cli.cmd {
            Cmd::Train { jobs, .. } => assert_eq!(jobs, Some(5)),
            _ => panic!("expected Cmd::Train"),
        }
        let cli = Cli::try_parse_from(["kbx", "train"]).unwrap();
        match cli.cmd {
            Cmd::Train { jobs, .. } => assert!(jobs.is_none()),
            _ => panic!("expected Cmd::Train"),
        }
    }

    #[test]
    fn distil_cmd_parses_aliases_only_flag() {
        // Flag present → aliases_only true; min_aliases defaults to 3 and is overridable.
        let cli = Cli::try_parse_from(["kbx", "distil", "--aliases-only"]).unwrap();
        match cli.cmd {
            Cmd::Distil {
                aliases_only,
                min_aliases,
                ..
            } => {
                assert!(aliases_only);
                assert_eq!(min_aliases, 3, "min_aliases defaults to 3");
            }
            _ => panic!("expected Cmd::Distil"),
        }

        let cli = Cli::try_parse_from(["kbx", "distil", "--min-aliases", "5"]).unwrap();
        match cli.cmd {
            Cmd::Distil {
                aliases_only,
                min_aliases,
                ..
            } => {
                assert!(!aliases_only, "flag absent → false");
                assert_eq!(min_aliases, 5);
            }
            _ => panic!("expected Cmd::Distil"),
        }
    }

    #[test]
    fn distil_cmd_parses_jobs_flag() {
        let cli = Cli::try_parse_from(["kbx", "distil", "--jobs", "5"]).unwrap();
        match cli.cmd {
            Cmd::Distil { jobs, .. } => assert_eq!(jobs, Some(5)),
            _ => panic!("expected Cmd::Distil"),
        }
        let cli = Cli::try_parse_from(["kbx", "distil"]).unwrap();
        match cli.cmd {
            Cmd::Distil { jobs, .. } => assert!(jobs.is_none()),
            _ => panic!("expected Cmd::Distil"),
        }
    }

    #[test]
    fn eval_capture_defaults_off_and_samples_one() {
        let cli = Cli::try_parse_from(["kbx", "eval"]).unwrap();
        match cli.cmd {
            Cmd::Eval {
                capture, samples, ..
            } => {
                assert!(!capture, "--capture must default OFF (non-breaking)");
                assert_eq!(samples, 1, "--samples must default to 1");
            }
            _ => panic!("expected Cmd::Eval"),
        }
        let cli = Cli::try_parse_from(["kbx", "eval", "--capture", "--samples", "4"]).unwrap();
        match cli.cmd {
            Cmd::Eval {
                capture, samples, ..
            } => {
                assert!(capture);
                assert_eq!(samples, 4);
            }
            _ => panic!("expected Cmd::Eval"),
        }
    }

    #[test]
    fn dataset_subcommands_parse() {
        let cli = Cli::try_parse_from(["kbx", "dataset", "stat", "d.toml"]).unwrap();
        match cli.cmd {
            Cmd::Dataset {
                cmd: DatasetCmd::Stat { file },
            } => assert_eq!(file, PathBuf::from("d.toml")),
            _ => panic!("expected dataset stat"),
        }

        let cli = Cli::try_parse_from([
            "kbx", "dataset", "merge", "--from", "s.toml", "--into", "d.toml",
        ])
        .unwrap();
        match cli.cmd {
            Cmd::Dataset {
                cmd: DatasetCmd::Merge { from, into },
            } => {
                assert_eq!(from, PathBuf::from("s.toml"));
                assert_eq!(into, PathBuf::from("d.toml"));
            }
            _ => panic!("expected dataset merge"),
        }

        // sample: -n required, --seed defaults to 0.
        let cli = Cli::try_parse_from(["kbx", "dataset", "sample", "d.toml", "-n", "5"]).unwrap();
        match cli.cmd {
            Cmd::Dataset {
                cmd: DatasetCmd::Sample { file, n, seed },
            } => {
                assert_eq!(file, PathBuf::from("d.toml"));
                assert_eq!(n, 5);
                assert_eq!(seed, 0, "--seed defaults to 0 for reproducibility");
            }
            _ => panic!("expected dataset sample"),
        }
    }

    #[test]
    fn truncate_chars_cuts_and_marks() {
        assert_eq!(truncate_chars("short", 100), "short");
        assert_eq!(truncate_chars("abcdef", 3), "abc…");
    }

    #[test]
    fn export_dataset_parses_sft_defaults() {
        let cli = Cli::try_parse_from([
            "kbx", "export", "--from", "tagA", "--format", "sft", "--out", "d.jsonl",
        ])
        .unwrap();
        match cli.cmd {
            Cmd::Export {
                from,
                format,
                out,
                shape,
                include_partial,
                max_pairs,
                ..
            } => {
                assert_eq!(from, "tagA");
                assert_eq!(format, ExportFormat::Sft);
                assert_eq!(out, PathBuf::from("d.jsonl"));
                assert_eq!(
                    shape,
                    ExportShape::Messages,
                    "shape must default to messages"
                );
                assert!(!include_partial);
                assert_eq!(max_pairs, 1);
            }
            _ => panic!("expected Cmd::Export"),
        }
    }

    #[test]
    fn export_dataset_parses_dpo_and_sharegpt() {
        let cli = Cli::try_parse_from([
            "kbx",
            "export",
            "--from",
            "a,b",
            "--format",
            "dpo",
            "--out",
            "o.jsonl",
            "--max-pairs",
            "3",
        ])
        .unwrap();
        match cli.cmd {
            Cmd::Export {
                from,
                format,
                max_pairs,
                ..
            } => {
                assert_eq!(from, "a,b");
                assert_eq!(format, ExportFormat::Dpo);
                assert_eq!(max_pairs, 3);
            }
            _ => panic!("expected Cmd::ExportDataset"),
        }
        let cli = Cli::try_parse_from([
            "kbx",
            "export",
            "--from",
            "t",
            "--format",
            "sft",
            "--out",
            "o.jsonl",
            "--shape",
            "sharegpt",
            "--include-partial",
        ])
        .unwrap();
        match cli.cmd {
            Cmd::Export {
                shape,
                include_partial,
                ..
            } => {
                assert_eq!(shape, ExportShape::Sharegpt);
                assert!(include_partial);
            }
            _ => panic!("expected Cmd::ExportDataset"),
        }
    }
}
