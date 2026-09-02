//! kb-train — enrich the reasoning graph, export TZ episode supervision, GEPA-optimize retrieval.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use glossa::graph::ontology::Ontology;
use glossa::graph::store::GraphStore;
use glossa::index::store::{DocIndex, RankedHit};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Parser)]
#[command(
    name = "kb-train",
    about = "Build & learn: enrich graph, export TZ episodes, GEPA optimize"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Cmd {
    /// Enrich the reasoning graph from solved training cases.
    Enrich {
        #[arg(long)]
        train: PathBuf,
        #[arg(long, default_value = "kb-test")]
        work: PathBuf,
        #[arg(long, default_value_t = 0)]
        limit: usize,
        #[arg(long, default_value = "http://127.0.0.1:3000")]
        tensorzero_endpoint: String,
        #[arg(long, default_value = "enrich")]
        tensorzero_function: String,
    },
    /// Export search/read datasets from TZ ClickHouse episodes (primary GEPA source).
    ExportTz {
        #[arg(long, default_value = "kb-test")]
        work: PathBuf,
        #[arg(long, default_value = "gepa-out")]
        out: PathBuf,
        #[arg(long, action = clap::ArgAction::Append)]
        train: Vec<PathBuf>,
        #[arg(long, default_value = "answer_hotpot")]
        function: String,
        #[arg(long)]
        run: Option<String>,
        #[arg(long)]
        clickhouse: Option<String>,
        #[arg(long, default_value_t = 10)]
        k: usize,
    },
    /// Export constraint materialize/research/compile-fix datasets from TZ ClickHouse episodes.
    #[cfg(feature = "constraint")]
    ExportTzConstraint {
        #[arg(long, default_value = "kb-test")]
        work: PathBuf,
        #[arg(long, default_value = "gepa-constraint-out")]
        out: PathBuf,
        #[arg(long, default_value = "constraint_validate")]
        function: String,
        #[arg(long)]
        run: Option<String>,
        #[arg(long, default_value = "kb-val")]
        val_dir: PathBuf,
        #[arg(long)]
        clickhouse: Option<String>,
    },
    /// Legacy graph dump (auxiliary gold index only — not primary training source).
    Dump {
        #[arg(long, default_value = "kb-test")]
        work: PathBuf,
        #[arg(long, default_value = ".")]
        out: PathBuf,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long, default_value_t = 20)]
        poll_secs: u64,
        #[arg(long, default_value_t = 10)]
        idle_stop: u32,
        #[arg(long)]
        once: bool,
    },
    /// Bootstrap constraint GEPA materialize.jsonl from local gold tables.
    #[cfg(feature = "constraint")]
    SyntheticConstraint {
        #[arg(long, default_value = "kb-val")]
        val_dir: PathBuf,
        #[arg(long, default_value = "doc.pdf")]
        doc: String,
        #[arg(long, default_value = "eval/fixtures/constraint-research-gold.json")]
        research_gold: PathBuf,
        #[arg(long, default_value = "gepa-constraint-out")]
        out: PathBuf,
    },
    /// GEPA-optimize the prod answer_hotpot system prompt (quad search + grep + glob + read scoring).
    Optimize {
        #[arg(long, default_value = "gepa-out/search.jsonl")]
        search: PathBuf,
        #[arg(long, default_value = "gepa-out/grep.jsonl")]
        grep: PathBuf,
        #[arg(long, default_value = "gepa-out/glob.jsonl")]
        glob: PathBuf,
        #[arg(long, default_value = "gepa-out/read.jsonl")]
        read: PathBuf,
        #[arg(long, default_value = "gepa-out/answer_hotpot.prompt.txt")]
        out: PathBuf,
        #[arg(
            long,
            default_value = "eval/tensorzero/config/answer_hotpot/system.minijinja"
        )]
        seed: PathBuf,
        #[arg(long, default_value = "kb-test")]
        work: PathBuf,
        #[arg(long, default_value = "http://127.0.0.1:3000")]
        gateway: String,
        #[arg(long, default_value = "search")]
        search_function: String,
        #[arg(long, default_value = "read")]
        read_function: String,
        #[arg(long, default_value = "grep")]
        grep_function: String,
        #[arg(long, default_value = "glob")]
        glob_function: String,
        #[arg(long, default_value = "baseline")]
        variant: String,
        #[arg(long, default_value = "gepa_reflect")]
        reflect_function: String,
        #[arg(long)]
        run: Option<String>,
        #[arg(long = "tag", value_name = "KEY=VALUE")]
        tag: Vec<String>,
        #[arg(long, default_value_t = 0.3)]
        val_frac: f64,
        #[arg(long, default_value_t = 6)]
        budget: usize,
        #[arg(long, default_value_t = 12)]
        minibatch: usize,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long, default_value_t = 0.25)]
        w_search: f64,
        #[arg(long, default_value_t = 0.25)]
        w_grep: f64,
        #[arg(long, default_value_t = 0.10)]
        w_glob: f64,
        #[arg(long, default_value_t = 0.25)]
        w_read: f64,
        /// RNG seed for minibatch sampling (default: hash of run tag).
        #[arg(long)]
        rng_seed: Option<u64>,
        /// Cap on D_pareto instances sampled from val (parent selection + pool scores).
        #[arg(long, default_value_t = 20)]
        pareto_size: usize,
        /// Parent selection: pareto (default) or current_best.
        #[arg(long, default_value = "pareto")]
        candidate_selection: String,
    },
    /// GEPA-optimize constraint agent prompt slices (5 pools: discover → validate).
    #[cfg(feature = "constraint")]
    OptimizeConstraint {
        #[arg(long, default_value = "gepa-constraint-out/discover.jsonl")]
        discover: PathBuf,
        #[arg(long, default_value = "gepa-constraint-out/materialize.jsonl")]
        materialize: PathBuf,
        #[arg(long, default_value = "gepa-constraint-out/compile.jsonl")]
        compile: PathBuf,
        #[arg(long, default_value = "gepa-constraint-out/coverage.jsonl")]
        coverage: PathBuf,
        #[arg(long, default_value = "gepa-constraint-out/validate.jsonl")]
        validate: PathBuf,
        #[arg(long, default_value = "gepa-constraint-out")]
        out_dir: PathBuf,
        #[arg(
            long,
            default_value = "gepa-constraint-out/constraint_materialize.prompt.txt"
        )]
        out: PathBuf,
        #[arg(long, default_value = "eval/sops/example")]
        sop_dir: PathBuf,
        #[arg(long, default_value = "kb-test")]
        work: PathBuf,
        #[arg(long, default_value = "http://127.0.0.1:3000")]
        gateway: String,
        #[arg(long, default_value = "cdiscover")]
        discover_function: String,
        #[arg(long, default_value = "cmaterialize")]
        materialize_function: String,
        #[arg(long, default_value = "ccompile")]
        compile_function: String,
        #[arg(long, default_value = "ccoverage")]
        coverage_function: String,
        #[arg(long, default_value = "cvalidate")]
        validate_function: String,
        #[arg(long, default_value = "gepa_reflect")]
        reflect_function: String,
        #[arg(long, default_value = "baseline")]
        variant: String,
        #[arg(long)]
        run: Option<String>,
        #[arg(long = "tag", value_name = "KEY=VALUE")]
        tag: Vec<String>,
        #[arg(long, default_value_t = 0.3)]
        val_frac: f64,
        #[arg(long, default_value_t = 6)]
        budget: usize,
        #[arg(long, default_value_t = 8)]
        minibatch: usize,
        #[arg(long, default_value_t = 0.5)]
        hit_threshold: f64,
        #[arg(long)]
        rng_seed: Option<u64>,
        #[arg(long, default_value_t = 20)]
        pareto_size: usize,
        #[arg(long, default_value_t = 0.25)]
        w_discover: f64,
        #[arg(long, default_value_t = 0.30)]
        w_materialize: f64,
        #[arg(long, default_value_t = 0.15)]
        w_compile: f64,
        #[arg(long, default_value_t = 0.20)]
        w_coverage: f64,
        #[arg(long, default_value_t = 0.10)]
        w_validate: f64,
    },
    /// GEPA-optimize the GRAPH reader system prompt for multi-hop Exact-Match (LM Studio rollouts).
    OptimizeGraph {
        #[arg(long)]
        questions: PathBuf,
        #[arg(long)]
        work: PathBuf,
        #[arg(long, default_value = "http://localhost:1234/v1/chat/completions")]
        endpoint: String,
        #[arg(long, default_value = "qwen3.5-4b")]
        model: String,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long, default_value = "http://localhost:3100")]
        gateway: String,
        #[arg(long, default_value = "gepa_reflect")]
        reflect_function: String,
        #[arg(long, default_value = "baseline")]
        variant: String,
        #[arg(long, default_value = "gepa-out/graph_prompt.txt")]
        out: PathBuf,
        #[arg(long)]
        seed_prompt: Option<PathBuf>,
        #[arg(long)]
        run: Option<String>,
        #[arg(long = "tag", value_name = "KEY=VALUE")]
        tag: Vec<String>,
        #[arg(long, default_value_t = 0.3)]
        val_frac: f64,
        #[arg(long, default_value_t = 12)]
        budget: usize,
        #[arg(long, default_value_t = 6)]
        minibatch: usize,
        #[arg(long)]
        rng_seed: Option<u64>,
        #[arg(long, default_value_t = 12)]
        pareto_size: usize,
        #[arg(long, default_value = "pareto")]
        candidate_selection: String,
    },
    /// Splice GEPA-optimized prompt slices into SOP.md between `{# GEPA:<TAG>_START/_END #}` anchors.
    ApplySopSlices {
        /// SOP pack directory containing SOP.md.
        #[arg(long, default_value = "eval/sops/example")]
        sop_dir: PathBuf,
        /// Directory with `constraint_<tag>.prompt.txt` slice files.
        #[arg(long, default_value = "gepa-constraint-out")]
        slices: PathBuf,
    },
}

/// Replace each anchored SOP.md region with the matching slice file, keeping the anchors.
/// Anchors present in the SOP but without a slice file are skipped with a note; at least
/// one region must be spliced. The previous SOP.md is kept as SOP.md.bak.
fn apply_sop_slices(sop_dir: &Path, slices: &Path) -> Result<()> {
    let sop_md = sop_dir.join("SOP.md");
    let text =
        std::fs::read_to_string(&sop_md).with_context(|| format!("read {}", sop_md.display()))?;
    let mut tags: Vec<String> = Vec::new();
    let mut rest = text.as_str();
    while let Some(i) = rest.find("{# GEPA:") {
        let after = &rest[i + "{# GEPA:".len()..];
        match after.find("_START #}") {
            Some(j)
                if after[..j]
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_') =>
            {
                if !tags.contains(&after[..j].to_string()) {
                    tags.push(after[..j].to_string());
                }
                rest = &after[j..];
            }
            _ => rest = after,
        }
    }
    anyhow::ensure!(
        !tags.is_empty(),
        "no {{# GEPA:<TAG>_START #}} anchors in {}",
        sop_md.display()
    );
    let mut out = text.clone();
    let mut applied = 0usize;
    for tag in &tags {
        let slice = slices.join(format!("constraint_{}.prompt.txt", tag.to_lowercase()));
        if !slice.exists() {
            eprintln!("skip {tag}: no {}", slice.display());
            continue;
        }
        let body =
            std::fs::read_to_string(&slice).with_context(|| format!("read {}", slice.display()))?;
        let start = format!("{{# GEPA:{tag}_START #}}");
        let end = format!("{{# GEPA:{tag}_END #}}");
        let s = out
            .find(&start)
            .with_context(|| format!("anchor {start} missing"))?;
        let e = out
            .find(&end)
            .with_context(|| format!("anchor {end} missing"))?;
        anyhow::ensure!(e > s, "anchor {end} precedes {start}");
        out.replace_range(s + start.len()..e, &format!("\n{}\n", body.trim()));
        applied += 1;
    }
    anyhow::ensure!(
        applied > 0,
        "no slice files in {} matched anchors ({})",
        slices.display(),
        tags.join(", ")
    );
    let bak = sop_md.with_extension("md.bak");
    std::fs::write(&bak, &text).with_context(|| format!("write {}", bak.display()))?;
    std::fs::write(&sop_md, &out).with_context(|| format!("write {}", sop_md.display()))?;
    println!(
        "applied {applied} GEPA slice(s) to {} (backup {})",
        sop_md.display(),
        bak.display()
    );
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::ApplySopSlices { sop_dir, slices } => apply_sop_slices(&sop_dir, &slices),
        Cmd::OptimizeGraph {
            questions,
            work,
            endpoint,
            model,
            api_key,
            gateway,
            reflect_function,
            variant,
            out,
            seed_prompt,
            run,
            tag,
            val_frac,
            budget,
            minibatch,
            rng_seed,
            pareto_size,
            candidate_selection,
        } => run_optimize_graph(
            questions,
            work,
            endpoint,
            model,
            api_key,
            gateway,
            reflect_function,
            variant,
            out,
            seed_prompt,
            run,
            tag,
            val_frac,
            budget,
            minibatch,
            rng_seed,
            pareto_size,
            candidate_selection,
        ),
        Cmd::Enrich {
            train,
            work,
            limit,
            tensorzero_endpoint,
            tensorzero_function,
        } => kb_eval::enrich::run_enrich(
            &train,
            &work,
            limit,
            &tensorzero_endpoint,
            &tensorzero_function,
        ),
        Cmd::ExportTz {
            work,
            out,
            train,
            function,
            run,
            clickhouse,
            k,
        } => {
            let train_files = if train.is_empty() {
                vec![PathBuf::from("kb-val/derived/synthetic-train.json")]
            } else {
                train
            };
            kb_eval::export_tz::run_export(kb_eval::export_tz::ExportConfig {
                clickhouse_url: clickhouse
                    .unwrap_or_else(kb_eval::export_tz::default_clickhouse_url),
                function_name: function,
                run_tag: run,
                train_files,
                work,
                out,
                top_k: k,
            })
            .map(|_| ())
        }
        #[cfg(feature = "constraint")]
        Cmd::ExportTzConstraint {
            work,
            out,
            function,
            run,
            val_dir,
            clickhouse,
        } => kb_eval::export_tz_constraint::run(
            kb_eval::export_tz_constraint::ExportTzConstraintConfig {
                clickhouse_url: clickhouse
                    .unwrap_or_else(kb_eval::export_tz::default_clickhouse_url),
                work,
                out,
                function,
                run,
                val_dir,
            },
        )
        .map(|_| ()),
        Cmd::Dump {
            work,
            out,
            k,
            poll_secs,
            idle_stop,
            once,
        } => run_dump(work, out, k, poll_secs, idle_stop, once),
        #[cfg(feature = "constraint")]
        Cmd::SyntheticConstraint {
            val_dir,
            doc,
            research_gold,
            out,
        } => {
            std::fs::create_dir_all(&out)?;
            let examples =
                kb_eval::constraint_synthetic::materialize_examples_from_dir(&val_dir, &doc)?;
            let materialize_path = out.join("materialize.jsonl");
            kb_eval::constraint_synthetic::write_materialize_jsonl(&materialize_path, &examples)?;
            let compile =
                kb_eval::constraint_synthetic::compile_fix_examples_from_materialize(&examples);
            let compile_path = out.join("compile.jsonl");
            kb_eval::constraint_synthetic::write_compile_jsonl(&compile_path, &compile)?;
            let coverage =
                kb_eval::constraint_synthetic::coverage_examples_from_materialize(&examples);
            let coverage_path = out.join("coverage.jsonl");
            kb_eval::constraint_synthetic::write_coverage_jsonl(&coverage_path, &coverage)?;
            let validate =
                kb_eval::constraint_synthetic::validate_examples_from_materialize(&examples);
            let validate_path = out.join("validate.jsonl");
            kb_eval::constraint_synthetic::write_validate_jsonl(&validate_path, &validate)?;
            let discover = if research_gold.exists() {
                kb_eval::constraint_synthetic::load_research_grep_examples(&research_gold)?
            } else {
                eprintln!(
                    "research gold fixture {} not found (local-only asset) — writing empty discover.jsonl",
                    research_gold.display()
                );
                Vec::new()
            };
            let discover_path = out.join("discover.jsonl");
            kb_eval::constraint_synthetic::write_discover_jsonl(&discover_path, &discover)?;
            println!(
                "wrote materialize={} -> {}, compile={} -> {}, coverage={} -> {}, validate={} -> {}, discover={} -> {}",
                examples.len(),
                materialize_path.display(),
                compile.len(),
                compile_path.display(),
                coverage.len(),
                coverage_path.display(),
                validate.len(),
                validate_path.display(),
                discover.len(),
                discover_path.display()
            );
            Ok(())
        }
        Cmd::Optimize {
            search,
            grep,
            glob,
            read,
            out,
            seed,
            work,
            gateway,
            search_function,
            read_function,
            grep_function,
            glob_function,
            variant,
            reflect_function,
            run,
            tag,
            val_frac,
            budget,
            minibatch,
            k,
            w_search,
            w_grep,
            w_glob,
            w_read,
            rng_seed,
            pareto_size,
            candidate_selection,
        } => run_optimize(
            search,
            grep,
            glob,
            read,
            out,
            seed,
            work,
            gateway,
            search_function,
            read_function,
            grep_function,
            glob_function,
            variant,
            reflect_function,
            run,
            tag,
            val_frac,
            budget,
            minibatch,
            k,
            w_search,
            w_grep,
            w_glob,
            w_read,
            rng_seed,
            pareto_size,
            candidate_selection,
        ),
        #[cfg(feature = "constraint")]
        Cmd::OptimizeConstraint {
            discover,
            materialize,
            compile,
            coverage,
            validate,
            out_dir,
            out,
            sop_dir,
            work,
            gateway,
            discover_function,
            materialize_function,
            compile_function,
            coverage_function,
            validate_function,
            reflect_function,
            variant,
            run,
            tag,
            val_frac,
            budget,
            minibatch,
            hit_threshold,
            rng_seed,
            pareto_size,
            w_discover,
            w_materialize,
            w_compile,
            w_coverage,
            w_validate,
        } => run_optimize_constraint(
            discover,
            materialize,
            compile,
            coverage,
            validate,
            out_dir,
            out,
            sop_dir,
            work,
            gateway,
            discover_function,
            materialize_function,
            compile_function,
            coverage_function,
            validate_function,
            reflect_function,
            variant,
            run,
            tag,
            val_frac,
            budget,
            minibatch,
            hit_threshold,
            rng_seed,
            pareto_size,
            w_discover,
            w_materialize,
            w_compile,
            w_coverage,
            w_validate,
        ),
    }
}

fn load_jsonl<T: serde::de::DeserializeOwned>(path: &PathBuf) -> Result<Vec<T>> {
    use std::io::{BufRead, BufReader};
    let f = std::fs::File::open(path).with_context(|| format!("read {}", path.display()))?;
    let reader = BufReader::new(f);
    let mut out = Vec::new();
    for (n, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "read {} line {} (invalid UTF-8 — re-run `just export-tz`; file may be corrupted by concurrent export)",
                path.display(),
                n + 1,
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(
            serde_json::from_str(&line)
                .with_context(|| format!("parse {} line {}", path.display(), n + 1))?,
        );
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn run_optimize(
    search: PathBuf,
    grep: PathBuf,
    glob: PathBuf,
    read: PathBuf,
    out: PathBuf,
    seed: PathBuf,
    work: PathBuf,
    gateway: String,
    search_function: String,
    read_function: String,
    grep_function: String,
    glob_function: String,
    variant: String,
    reflect_function: String,
    run: Option<String>,
    tag: Vec<String>,
    val_frac: f64,
    budget: usize,
    minibatch: usize,
    top_k: usize,
    w_search: f64,
    w_grep: f64,
    w_glob: f64,
    w_read: f64,
    rng_seed: Option<u64>,
    pareto_size: usize,
    candidate_selection: String,
) -> Result<()> {
    let searches: Vec<kb_eval::gepa::SearchExample> = if search.exists() {
        load_jsonl(&search)?
    } else {
        Vec::new()
    };
    let greps: Vec<kb_eval::export_tz::GrepExample> = if grep.exists() {
        load_jsonl::<kb_eval::export_tz::GrepExample>(&grep)?
            .into_iter()
            .filter(|g| !g.synthetic)
            .collect()
    } else {
        Vec::new()
    };
    let globs: Vec<kb_eval::export_tz::GlobExample> = if glob.exists() {
        load_jsonl::<kb_eval::export_tz::GlobExample>(&glob)?
            .into_iter()
            .filter(|g| !g.synthetic)
            .collect()
    } else {
        Vec::new()
    };
    let reads: Vec<kb_eval::export_tz::ReadExample> = if read.exists() {
        load_jsonl(&read)?
    } else {
        Vec::new()
    };

    let seed_prompt = kb_eval::gepa::load_seed_prompt(&seed)?;
    let run_label = run.unwrap_or_else(kb_eval::gepa::default_run_tag);
    let rng_seed = rng_seed.unwrap_or_else(|| kb_eval::gepa::hash_run_seed(&run_label));
    let candidate_selection = candidate_selection
        .parse::<kb_eval::gepa::CandidateSelection>()
        .with_context(|| format!("parse candidate_selection {candidate_selection:?}"))?;
    let mut tags = serde_json::Map::new();
    tags.insert("run".into(), run_label.clone().into());
    tags.insert("budget".into(), budget.to_string().into());
    tags.insert("minibatch".into(), minibatch.to_string().into());
    tags.insert("w_search".into(), w_search.to_string().into());
    tags.insert("w_grep".into(), w_grep.to_string().into());
    tags.insert("w_glob".into(), w_glob.to_string().into());
    tags.insert("w_read".into(), w_read.to_string().into());
    for t in &tag {
        if let Some((k, v)) = t.split_once('=') {
            tags.insert(k.to_string(), v.into());
        }
    }

    let result = kb_eval::gepa::run(
        kb_eval::gepa::GepaConfig {
            gateway,
            search_function,
            read_function,
            grep_function,
            glob_function,
            variant,
            reflect_function,
            episode_id: kb_eval::tz::backdated_episode_id(30),
            tags: Value::Object(tags),
            val_frac,
            budget,
            minibatch,
            seed_prompt,
            work,
            top_k,
            w_search,
            w_grep,
            w_glob,
            w_read,
            seed: rng_seed,
            pareto_size,
            candidate_selection,
        },
        searches,
        greps,
        globs,
        reads,
    )?;
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out, &result.prompt).with_context(|| format!("write {}", out.display()))?;
    println!(
        "wrote best prompt -> {} (run={run_label}, episode={}, search_acc={:.3}, grep_acc={:.3}, glob_acc={:.3}, read_acc={:.3})",
        out.display(),
        result.episode_id,
        result.search_acc,
        result.grep_acc,
        result.glob_acc,
        result.read_acc,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_optimize_graph(
    questions: PathBuf,
    work: PathBuf,
    endpoint: String,
    model: String,
    api_key: Option<String>,
    gateway: String,
    reflect_function: String,
    variant: String,
    out: PathBuf,
    seed_prompt: Option<PathBuf>,
    run: Option<String>,
    tag: Vec<String>,
    val_frac: f64,
    budget: usize,
    minibatch: usize,
    rng_seed: Option<u64>,
    pareto_size: usize,
    candidate_selection: String,
) -> Result<()> {
    let text = std::fs::read_to_string(&questions)
        .with_context(|| format!("read {}", questions.display()))?;
    let qs = kb_eval::dataset::parse_questions(&text)?;
    let seed = match seed_prompt {
        Some(p) => kb_eval::gepa::load_seed_prompt(&p)?,
        None => kb_eval::backend::prompt::system_prompt(true).to_string(),
    };
    let run_label = run.unwrap_or_else(kb_eval::gepa::default_run_tag);
    let rng_seed = rng_seed.unwrap_or_else(|| kb_eval::gepa::hash_run_seed(&run_label));
    let candidate_selection = candidate_selection
        .parse::<kb_eval::gepa::CandidateSelection>()
        .with_context(|| format!("parse candidate_selection {candidate_selection:?}"))?;
    let mut tags = serde_json::Map::new();
    tags.insert("run".into(), run_label.clone().into());
    tags.insert("budget".into(), budget.to_string().into());
    tags.insert("minibatch".into(), minibatch.to_string().into());
    for t in &tag {
        if let Some((k, v)) = t.split_once('=') {
            tags.insert(k.to_string(), v.into());
        }
    }

    let episode_id = kb_eval::tz::backdated_episode_id(30);
    let tags = Value::Object(tags);
    let reflect = |instruction: &str| -> Result<String> {
        println!(
            "graph reflect payload: {} chars (~{} tok est)",
            instruction.len(),
            instruction.len() / 4,
        );
        let turn = kb_eval::tz::infer(
            &gateway,
            &reflect_function,
            &episode_id,
            &[json!({"role": "user", "content": instruction})],
            &tags,
            Duration::from_secs(180),
            // Honors the CLI `--variant` flag. The old (deleted) `gepa_graph::reflect` hardcoded
            // `Some("baseline")` and ignored `cfg.variant`, so this flag was previously dead;
            // its default value is "baseline", so default behavior is unchanged.
            Some(variant.as_str()),
            None,
            None,
        )
        .context("gepa_reflect inference failed")?;
        let text = turn.text().trim().to_string();
        anyhow::ensure!(!text.is_empty(), "gepa_reflect returned an empty prompt");
        anyhow::ensure!(
            !kb_eval::gepa::output_likely_truncated(&text, turn.finish_reason.as_deref()),
            "gepa_reflect output truncated (finish_reason={:?})",
            turn.finish_reason,
        );
        Ok(text)
    };

    // `gepa_graph::run` moved from an iteration budget to a DSPy-style metric-call ceiling + a
    // candidate-pool cap. This legacy research binary keeps its `--budget` (iteration) flag; map it
    // loosely: `budget` candidates max, with a finite rollout ceiling (~a handful of minibatch/
    // pareto passes per iteration) so a run that never accepts still terminates. `kbx train` is the
    // maintained front-end with the real dataset-derived auto-budget.
    let legacy_max_candidates = budget.max(1);
    let legacy_max_metric_calls = budget
        .saturating_mul(minibatch.saturating_add(pareto_size).saturating_mul(4))
        .max(minibatch.max(1));
    let result = kb_eval::gepa_graph::run(
        kb_eval::gepa_graph::GepaGraphConfig {
            endpoint,
            model,
            api_key,
            val_frac,
            max_metric_calls: legacy_max_metric_calls,
            max_candidates: legacy_max_candidates,
            minibatch,
            seed_prompt: seed,
            work,
            seed: rng_seed,
            pareto_size,
            candidate_selection,
            // Legacy research binary has no lab.toml; use the canonical-GEPA default (no cache).
            minibatch_cache: false,
            // Legacy research binary has no `--jobs` flag; keep its rollouts sequential (no
            // behaviour change).
            jobs: 1,
            // Legacy research binary has no `--metric` flag; keep exact-EM (no behaviour change).
            judge: None,
            // Legacy research binary has no `[user_sim]` wiring; gate off (no behaviour change).
            user_sim: None,
            user_sim_prompt: None,
        },
        qs,
        &reflect,
        // Legacy research binary has no progress bar; a hidden one is a no-op sink (behaviour
        // byte-identical to the pre-bar `ProgressBar::hidden()` `score_questions` used internally).
        &indicatif::ProgressBar::hidden(),
    )?;
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out, &result.prompt).with_context(|| format!("write {}", out.display()))?;
    println!(
        "wrote best graph prompt -> {} (run={run_label}, baseline_score={:.3}, best_score={:.3}, candidates={})",
        out.display(),
        result.baseline_score,
        result.best_score,
        result.candidates,
    );
    Ok(())
}

#[cfg(feature = "constraint")]
fn load_sop_gepa_seed(sop_dir: &Path, out: &Path, tag: &str) -> Result<String> {
    if out.exists() {
        let content = std::fs::read_to_string(out)
            .with_context(|| format!("read prior prompt {}", out.display()))?;
        if !content.trim().is_empty() {
            return Ok(content);
        }
    }
    let md = kb_eval::constraint_gepa_sop::load_sop_md(sop_dir)?;
    kb_eval::constraint_gepa_sop::extract_gepa_slice(&md, tag)
        .with_context(|| format!("extract GEPA:{tag} seed from SOP.md"))
}

#[cfg(feature = "constraint")]
#[allow(clippy::too_many_arguments)]
fn run_optimize_constraint(
    discover: PathBuf,
    materialize: PathBuf,
    compile: PathBuf,
    coverage: PathBuf,
    validate: PathBuf,
    out_dir: PathBuf,
    out: PathBuf,
    sop_dir: PathBuf,
    work: PathBuf,
    gateway: String,
    discover_function: String,
    materialize_function: String,
    compile_function: String,
    coverage_function: String,
    validate_function: String,
    reflect_function: String,
    variant: String,
    run: Option<String>,
    tag: Vec<String>,
    val_frac: f64,
    budget: usize,
    minibatch: usize,
    hit_threshold: f64,
    rng_seed: Option<u64>,
    pareto_size: usize,
    w_discover: f64,
    w_materialize: f64,
    w_compile: f64,
    w_coverage: f64,
    w_validate: f64,
) -> Result<()> {
    let discover_examples = if discover.exists() {
        kb_eval::gepa_constraint::load_discover_jsonl(&discover)?
    } else {
        Vec::new()
    };
    let materialize_examples = kb_eval::gepa_constraint::load_materialize_jsonl(&materialize)?;
    let compile_examples = if compile.exists() {
        kb_eval::gepa_constraint::load_compile_jsonl(&compile)?
    } else {
        Vec::new()
    };
    let coverage_examples = if coverage.exists() {
        kb_eval::gepa_constraint::load_coverage_jsonl(&coverage)?
    } else {
        Vec::new()
    };
    let validate_examples = if validate.exists() {
        kb_eval::gepa_constraint::load_validate_jsonl(&validate)?
    } else {
        Vec::new()
    };
    let discover_out = out_dir.join("constraint_discover.prompt.txt");
    let materialize_out = out;
    let compile_out = out_dir.join("constraint_compile.prompt.txt");
    let coverage_out = out_dir.join("constraint_coverage.prompt.txt");
    let validate_out = out_dir.join("constraint_validate.prompt.txt");
    let seed_discover_prompt = load_sop_gepa_seed(&sop_dir, &discover_out, "DISCOVER")?;
    let seed_materialize_prompt = load_sop_gepa_seed(&sop_dir, &materialize_out, "MATERIALIZE")?;
    let seed_compile_prompt = load_sop_gepa_seed(&sop_dir, &compile_out, "COMPILE")?;
    let seed_coverage_prompt = load_sop_gepa_seed(&sop_dir, &coverage_out, "COVERAGE")?;
    let seed_validate_prompt = load_sop_gepa_seed(&sop_dir, &validate_out, "VALIDATE")?;
    let run_label = run.clone().unwrap_or_else(kb_eval::gepa::default_run_tag);
    let rng_seed = rng_seed.unwrap_or_else(|| kb_eval::gepa::hash_run_seed(&run_label));
    let mut tags = serde_json::Map::new();
    if let Some(ref r) = run {
        tags.insert("run".into(), r.clone().into());
    }
    for t in &tag {
        if let Some((k, v)) = t.split_once('=') {
            tags.insert(k.to_string(), v.into());
        }
    }

    let result = kb_eval::gepa_constraint::run(
        kb_eval::gepa_constraint::GepaConstraintConfig {
            gateway,
            discover_function,
            materialize_function,
            compile_function,
            coverage_function,
            validate_function,
            reflect_function,
            variant,
            episode_id: kb_eval::tz::backdated_episode_id(30),
            tags: Value::Object(tags),
            val_frac,
            budget,
            minibatch,
            seed_discover_prompt,
            seed_materialize_prompt,
            seed_compile_prompt,
            seed_coverage_prompt,
            seed_validate_prompt,
            hit_threshold,
            seed: rng_seed,
            pareto_size,
            w_discover,
            w_materialize,
            w_compile,
            w_coverage,
            w_validate,
            work,
        },
        discover_examples,
        materialize_examples,
        compile_examples,
        coverage_examples,
        validate_examples,
    )?;
    std::fs::create_dir_all(&out_dir).with_context(|| format!("create {}", out_dir.display()))?;
    if let Some(parent) = materialize_out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&discover_out, &result.discover_prompt)
        .with_context(|| format!("write {}", discover_out.display()))?;
    std::fs::write(&materialize_out, &result.materialize_prompt)
        .with_context(|| format!("write {}", materialize_out.display()))?;
    std::fs::write(&compile_out, &result.compile_prompt)
        .with_context(|| format!("write {}", compile_out.display()))?;
    std::fs::write(&coverage_out, &result.coverage_prompt)
        .with_context(|| format!("write {}", coverage_out.display()))?;
    std::fs::write(&validate_out, &result.validate_prompt)
        .with_context(|| format!("write {}", validate_out.display()))?;
    println!(
        "wrote best prompts -> {}, {}, {}, {}, {} (run={run_label}, episode={}, baseline_acc={:.3}, best_acc={:.3}, discover_acc={:.3}, materialize_acc={:.3}, compile_acc={:.3}, coverage_acc={:.3}, validate_acc={:.3}, candidates={})",
        discover_out.display(),
        materialize_out.display(),
        compile_out.display(),
        coverage_out.display(),
        validate_out.display(),
        result.episode_id,
        result.baseline_acc,
        result.best_acc,
        result.discover_acc,
        result.materialize_acc,
        result.compile_acc,
        result.coverage_acc,
        result.validate_acc,
        result.candidates,
    );
    Ok(())
}

fn evidence_is_relevant(
    analyzer: &glossa::index::multilang::TermAnalyzer,
    label: &str,
    chunk_text: &str,
) -> bool {
    let mut label_terms = HashSet::new();
    {
        let mut s = std::collections::BTreeSet::new();
        analyzer.terms(label, &mut s);
        label_terms.extend(s);
    }
    if label_terms.is_empty() {
        return false;
    }
    let mut chunk_terms = std::collections::BTreeSet::new();
    analyzer.terms(chunk_text, &mut chunk_terms);
    let shared = label_terms
        .iter()
        .filter(|t| chunk_terms.contains(*t))
        .count();
    shared * 2 >= label_terms.len()
}

fn hits_json(hits: &[RankedHit]) -> Vec<Value> {
    hits.iter()
        .map(|h| {
            json!({
                "ord": h.ord, "path": h.path, "location": h.location,
                "file_type": h.file_type, "snippet": h.snippet,
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_dump(
    work: PathBuf,
    out: PathBuf,
    k: usize,
    poll_secs: u64,
    idle_stop: u32,
    once: bool,
) -> Result<()> {
    use std::io::Write as _;

    eprintln!("note: `dump` is legacy — prefer `export-tz` from ClickHouse episodes");

    let idx = DocIndex::open_or_create(&work).context("open index")?;
    let graph = GraphStore::open(&work).context("open graph")?;
    let ont = Ontology::load_or_default(&work);
    let structural: HashSet<String> = ont.structural().into_iter().collect();
    let analyzer = glossa::index::multilang::TermAnalyzer::new();

    std::fs::create_dir_all(&out).with_context(|| format!("create {}", out.display()))?;
    let search_path = out.join("search.jsonl");
    let select_path = out.join("select.jsonl");
    let mut search_f = std::fs::File::create(&search_path)?;
    let mut select_f = std::fs::File::create(&select_path)?;
    println!(
        "kb-train dump (legacy): {} -> {} / {}",
        work.display(),
        search_path.display(),
        select_path.display(),
    );

    let mut seen: HashSet<String> = HashSet::new();
    let (mut nodes_kept, mut search_written, mut select_written, mut recall_hit) =
        (0u64, 0u64, 0u64, 0u64);
    let mut idle = 0u32;

    loop {
        let nodes = graph.all_nodes().context("all_nodes")?;
        let mut fresh = 0u64;
        for node in &nodes {
            if structural.contains(&node.node_type) || !seen.insert(node.id.clone()) {
                continue;
            }
            fresh += 1;
            let mut gold: Vec<(String, String)> = Vec::new();
            for e in graph.outgoing(&node.id).unwrap_or_default() {
                if e.edge_type == glossa::graph::MENTIONS {
                    if let Some((p, l)) = e.to.split_once('#') {
                        gold.push((p.to_string(), l.to_string()));
                    }
                }
            }
            if gold.is_empty() {
                continue;
            }
            let relevant: Vec<(String, String)> = gold
                .into_iter()
                .filter(|(p, l)| {
                    idx.read_chunk(p, l)
                        .ok()
                        .flatten()
                        .map(|text| evidence_is_relevant(&analyzer, &node.label, &text))
                        .unwrap_or(false)
                })
                .collect();
            if relevant.is_empty() {
                continue;
            }
            nodes_kept += 1;
            let gold_locs: Vec<String> = relevant.iter().map(|(p, l)| format!("{p}#{l}")).collect();
            writeln!(
                search_f,
                "{}",
                json!({"question": node.label, "gold": gold_locs})
            )?;
            search_written += 1;
            let hits = idx
                .search_filtered(&node.label, k, None, None, None)
                .unwrap_or_default();
            let gold_set: HashSet<(String, String)> = relevant.into_iter().collect();
            let gold_ords: Vec<u64> = hits
                .iter()
                .filter(|h| gold_set.contains(&(h.path.clone(), h.location.clone())))
                .map(|h| h.ord)
                .collect();
            if !gold_ords.is_empty() {
                recall_hit += 1;
                writeln!(
                    select_f,
                    "{}",
                    json!({"question": node.label, "hits": hits_json(&hits), "gold_ords": gold_ords})
                )?;
                select_written += 1;
            }
        }
        if once {
            break;
        }
        idle = if fresh == 0 { idle + 1 } else { 0 };
        if idle >= idle_stop {
            break;
        }
        std::thread::sleep(Duration::from_secs(poll_secs));
    }
    search_f.flush().ok();
    select_f.flush().ok();
    println!(
        "DONE (legacy dump): kept {nodes_kept} nodes, search={search_written}, select={select_written}, recall@{k}={:.3}",
        recall_hit as f64 / nodes_kept.max(1) as f64,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hits_json_carries_ord_and_path() {
        let hits = vec![RankedHit {
            path: "doc.pdf".into(),
            location: "p.7".into(),
            file_type: "pdf".into(),
            ord: 7,
            snippet: "the answer".into(),
            score: 1.0,
        }];
        let j = hits_json(&hits);
        assert_eq!(j[0]["ord"], 7);
        assert!(j[0].get("score").is_none());
    }
}
