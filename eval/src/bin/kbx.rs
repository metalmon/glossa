//! `kbx` — the file-first eval toolkit CLI: `init` scaffolds a workspace (lab.toml + editable
//! answer/judge prompts + a starter dataset.toml + runs/), `eval` runs a dataset.toml against a
//! corpus with the OpenAI-compatible agent backend, scores EM/F1, optionally judges each case,
//! and writes a `runs/<tag>/report.md` (+ per-case trace files) via `kb_eval::report::write_run`.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use kb_eval::backend::openai::OpenAiBackend;
use kb_eval::backend::AgentBackend;
use kb_eval::dataset::Question;
use kb_eval::dataset_toml::parse_dataset_toml;
use kb_eval::judge::{judge, Judgement, Verdict};
use kb_eval::lab::LabConfig;
use kb_eval::report::{load_cases, summary_text, write_case, write_run, CaseResult, RunMeta};
use kb_eval::scaffold::scaffold_init;
use kb_eval::score::{relaxed_match_any, token_f1_any};
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
    /// Scaffold a fresh eval workspace: lab.toml + answer.md/judge.md + dataset.toml + runs/.
    Init {
        /// Workspace directory to create/populate (created if missing).
        dir: PathBuf,
        /// Overwrite existing template files instead of refusing.
        #[arg(long)]
        force: bool,
    },
    /// Run a workspace's dataset.toml against its corpus and write a run report.
    Eval {
        /// Workspace directory (must contain lab.toml, or all pieces given via flags).
        workspace: PathBuf,
        /// Run tag (report dir name under runs/). Default: slug(workspace)-slug(dataset).
        #[arg(long)]
        tag: Option<String>,
        /// Override lab.toml's `corpus`.
        #[arg(long)]
        corpus: Option<PathBuf>,
        /// Override lab.toml's `defaults.dataset`.
        #[arg(long)]
        dataset: Option<PathBuf>,
        /// Override lab.toml's `defaults.prompt` (the answer-agent system prompt file).
        #[arg(long)]
        prompt: Option<PathBuf>,
        /// Override lab.toml's `defaults.judge_prompt`.
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
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Init { dir, force } => {
            scaffold_init(&dir, force)?;
            println!("initialized kbx workspace at {}", dir.display());
            Ok(())
        }
        Cmd::Eval {
            workspace,
            tag,
            corpus,
            dataset,
            prompt,
            judge: judge_path,
            limit,
            tag_filter,
            no_judge,
            resume,
            no_progress,
        } => run_eval(EvalArgs {
            workspace,
            tag,
            corpus,
            dataset,
            prompt,
            judge: judge_path,
            limit,
            tag_filter,
            no_judge,
            resume,
            no_progress,
        }),
    }
}

struct EvalArgs {
    workspace: PathBuf,
    tag: Option<String>,
    corpus: Option<PathBuf>,
    dataset: Option<PathBuf>,
    prompt: Option<PathBuf>,
    judge: Option<PathBuf>,
    limit: Option<usize>,
    tag_filter: Option<String>,
    no_judge: bool,
    resume: bool,
    no_progress: bool,
}

fn run_eval(args: EvalArgs) -> Result<()> {
    let workspace = args.workspace;
    let lab = LabConfig::load(&workspace)
        .with_context(|| format!("loading lab.toml under {}", workspace.display()))?;
    let mut paths = lab.resolve(&workspace);
    if let Some(c) = args.corpus {
        paths.corpus = c;
    }
    if let Some(d) = args.dataset {
        paths.dataset = d;
    }
    if let Some(p) = args.prompt {
        paths.prompt = p;
    }
    if let Some(j) = args.judge {
        paths.judge_prompt = j;
    }

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
        .unwrap_or_else(|| default_tag(&workspace, &paths.dataset));
    let runs_dir = workspace.join("runs");
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
            ProgressStyle::with_template("{msg} [{pos}/{len}] {bar:40.cyan/blue} {elapsed_precise}")
                .unwrap_or_else(|_| ProgressStyle::default_bar()),
        );
        pb
    } else {
        ProgressBar::hidden()
    };

    let mut results = Vec::with_capacity(cases.len());
    for q in &cases {
        pb.set_message(q.id.clone());
        let before = list_trace_files(&paths.corpus);

        let backend = OpenAiBackend {
            endpoint: lab.model.endpoint.clone(),
            model: lab.model.model.clone(),
            api_key: api_key.clone(),
            timeout,
            use_graph: true,
            system_prompt: Some(answer_md.clone()),
        };
        let answer = match backend.answer(&paths.corpus, q) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("case {}: agent error: {e}", q.id);
                format!("(error: {e})")
            }
        };

        let (tools, transcript) = read_new_trace(&paths.corpus, &before);

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

        println!("case {}: em={em:.2} f1={f1:.2} verdict={verdict:?}", q.id);

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
        corpus: paths.corpus.display().to_string(),
        n: all_results.len(),
        timestamp,
    };
    let report_path = write_run(&runs_dir, &tag, &meta, &all_results)?;
    println!("{}", summary_text(&all_results));
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

/// Stable slug: `slug(workspace basename)-slug(dataset stem)` — no clock in it, so repeat runs
/// against the same workspace/dataset land in the same `runs/<tag>/` unless the caller passes
/// `--tag` (the run's own timestamp still lives in `RunMeta`/the report body).
fn default_tag(workspace: &Path, dataset: &Path) -> String {
    let ws = workspace
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("workspace");
    let ds = dataset
        .file_stem()
        .and_then(|n| n.to_str())
        .unwrap_or("dataset");
    format!("{}-{}", slugify(ws), slugify(ds))
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
}
