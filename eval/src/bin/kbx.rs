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
use kb_eval::dataset_toml::parse_dataset_toml;
use kb_eval::distil::{self, DistilArgs};
use kb_eval::judge::{judge, Judgement, Verdict};
use kb_eval::lab::LabConfig;
use kb_eval::reason::{self, ReasonArgs};
use kb_eval::report::{load_cases, summary_text, write_case, write_run, CaseResult, RunMeta};
use kb_eval::scaffold::scaffold_init;
use kb_eval::score::{relaxed_match_any, token_f1_any};
use kb_eval::train::{self, TrainArgs};
use kb_eval::workspace::{self, KbxPaths};
use std::collections::HashSet;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Parser)]
#[command(name = "kbx", about = "File-first glossa eval toolkit")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
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
    },
    /// GEPA-optimize a corpus's `answer.md` (the answer-agent system prompt) against its
    /// `dataset.toml`, applying the winner back onto the workspace only when it strictly beats
    /// the seed prompt's full-val EM.
    Train {
        /// Corpus root (kb-style PATH resolution: explicit if given, else discovered from the
        /// current directory upward, else the current directory).
        path: Option<PathBuf>,
        /// Number of GEPA candidates to explore.
        #[arg(long, default_value_t = 12)]
        budget: usize,
        /// Minibatch size for per-candidate rollouts.
        #[arg(long, default_value_t = 6)]
        minibatch: usize,
        /// Fraction of the dataset held out as the full-validation split.
        #[arg(long = "val-frac", default_value_t = 0.3)]
        val_frac: f64,
        /// Max size of the Pareto frontier retained across candidates.
        #[arg(long = "pareto-size", default_value_t = 12)]
        pareto_size: usize,
        /// Candidate-selection strategy (e.g. "pareto").
        #[arg(long = "candidate-selection", default_value = "pareto")]
        candidate_selection: String,
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
            budget,
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
        } => train::run_train(
            path,
            TrainArgs {
                budget,
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
            },
        ),
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

    let use_judge = !args.no_judge && lab.judge.is_some();
    let judge_md = if use_judge {
        Some(
            std::fs::read_to_string(&paths.judge_prompt)
                .with_context(|| format!("reading judge prompt {}", paths.judge_prompt.display()))?,
        )
    } else {
        None
    };

    let api_key = lab.model.resolve_key();
    let timeout = Duration::from_secs(lab.model.timeout_secs);

    // indicatif draws to stderr by default; also check stdout since some shells redirect one but
    // not the other and either being non-interactive is a good signal this run isn't at a console.
    let show_progress = !args.no_progress
        && std::io::stdout().is_terminal()
        && std::io::stderr().is_terminal();
    let pb = if show_progress {
        let pb = ProgressBar::new(cases.len() as u64);
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
    } else {
        ProgressBar::hidden()
    };

    reset_tokens();
    reset_resamples();
    // One static stage word at the front for the whole run — never switches. The ticker owns only
    // `{msg}` (ETA + tokens/resamples), redrawn on its own timer.
    pb.set_prefix("evaluating");
    let ticker = StatusTicker::start(&pb);

    let mut results = Vec::with_capacity(cases.len());
    for q in &cases {
        let before = list_trace_files(&paths.root);

        let backend = OpenAiBackend {
            endpoint: lab.model.endpoint.clone(),
            model: lab.model.model.clone(),
            api_key: api_key.clone(),
            timeout,
            use_graph: true,
            system_prompt: Some(answer_md.clone()),
        };
        let answer = match backend.answer(&paths.root, q) {
            Ok(a) => a,
            Err(e) => {
                pb.println(format!("case {}: agent error: {e}", q.id));
                format!("(error: {e})")
            }
        };

        let (tools, transcript) = read_new_trace(&paths.root, &before);

        let golds = gold_forms(q);
        let em = if relaxed_match_any(&answer, &golds) {
            1.0
        } else {
            0.0
        };
        let f1 = token_f1_any(&answer, &golds);

        let (verdict, reason, judge_raw) = match (&judge_md, &lab.judge) {
            (Some(jmd), Some(jep)) => match judge(jep, jmd, &q.question, &q.answer, &answer) {
                Ok(Judgement {
                    verdict,
                    reason,
                    raw,
                }) => (verdict, reason, raw),
                Err(e) => (Verdict::Unscored, format!("judge error: {e}"), String::new()),
            },
            _ => (Verdict::Unscored, String::new(), String::new()),
        };

        pb.println(format!("case {}: em={em:.2} f1={f1:.2} verdict={verdict:?}", q.id));

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
        };
        write_case(&cases_dir, &r)
            .with_context(|| format!("persisting case {} to {}", r.id, cases_dir.display()))?;
        results.push(r);
        pb.inc(1);
    }
    drop(ticker); // stop before finish_and_clear so it can't redraw a message onto a cleared bar
    pb.finish_and_clear();

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
    println!("{}", summary_text(&all_results));
    let footnote = if cache_is_estimated() {
        " (cache estimated from prompt re-send)"
    } else {
        ""
    };
    println!("tokens: {}{footnote}", token_summary());
    println!("wrote {}", report_path.display());
    Ok(())
}

/// Gold answer forms accepted for scoring: the primary `answer` plus any `answer_aliases` —
/// mirrors the `golds` vector `run::eval_one` builds for the same reason (MuSiQue/SQuAD-style
/// alias-aware EM/F1).
fn gold_forms(q: &Question) -> Vec<String> {
    std::iter::once(q.answer.clone())
        .chain(q.answer_aliases.iter().cloned())
        .collect()
}

/// Snapshot of trace-file names currently under `<corpus>/.glossa/traces`, taken before an
/// `answer()` call so the file it writes (`TraceLog::to_dir` names it `<ts_ms>-<pid>.jsonl`) can
/// be told apart from traces left by earlier cases in the same run.
fn list_trace_files(corpus: &Path) -> HashSet<String> {
    let dir = corpus.join(".glossa").join("traces");
    let mut out = HashSet::new();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for ent in rd.flatten() {
            if let Some(name) = ent.file_name().to_str() {
                out.insert(name.to_string());
            }
        }
    }
    out
}

/// Diff the trace dir against the pre-call snapshot to find the file this one `answer()` call
/// wrote, then return (deduped tool names in first-seen order, raw JSONL trace text). Best-effort:
/// an empty pair when no new trace file turns up (e.g. the corpus has tracing disabled) — the run
/// stays functional, it just carries no tool/transcript detail for that case.
fn read_new_trace(corpus: &Path, before: &HashSet<String>) -> (Vec<String>, String) {
    let dir = corpus.join(".glossa").join("traces");
    let new_file = match std::fs::read_dir(&dir) {
        Ok(rd) => rd.flatten().find_map(|e| {
            let name = e.file_name().to_str()?.to_string();
            if before.contains(&name) {
                None
            } else {
                Some(e.path())
            }
        }),
        Err(_) => None,
    };
    let Some(path) = new_file else {
        return (Vec::new(), String::new());
    };
    let text = std::fs::read_to_string(&path).unwrap_or_default();
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
        assert_eq!(gold_forms(&q), vec!["Bob".to_string(), "Robert".to_string()]);
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
    fn read_new_trace_is_empty_when_no_trace_dir() {
        let dir = tempfile::tempdir().unwrap();
        let before = list_trace_files(dir.path());
        let (tools, transcript) = read_new_trace(dir.path(), &before);
        assert!(tools.is_empty());
        assert!(transcript.is_empty());
    }

    #[test]
    fn read_new_trace_picks_the_file_written_after_the_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let traces = dir.path().join(".glossa").join("traces");
        std::fs::create_dir_all(&traces).unwrap();
        std::fs::write(
            traces.join("100-1.jsonl"),
            "{\"ts_ms\":1,\"tool\":\"search\",\"args\":{},\"result\":[]}\n",
        )
        .unwrap();

        let before = list_trace_files(dir.path());
        std::fs::write(
            traces.join("200-1.jsonl"),
            concat!(
                "{\"ts_ms\":2,\"tool\":\"read\",\"args\":{},\"result\":{}}\n",
                "{\"ts_ms\":3,\"tool\":\"read\",\"args\":{},\"result\":{}}\n",
                "{\"ts_ms\":4,\"tool\":\"search\",\"args\":{},\"result\":[]}\n",
            ),
        )
        .unwrap();

        let (tools, transcript) = read_new_trace(dir.path(), &before);
        assert_eq!(tools, vec!["read".to_string(), "search".to_string()]);
        assert!(transcript.contains("read") && transcript.contains("search"));
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
            }
            _ => panic!("expected Cmd::Build"),
        }
    }
}
