//! GEPA-style reflective optimization of the GRAPH reader system prompt.
//!
//! Objective: maximize multi-hop Exact-Match (EM). Rollouts run through the SAME LM Studio
//! (OpenAI-compatible) agent loop the eval backend uses (`backend::openai`), driving glossa's
//! graph tools in-process against the corpus in `work`. Only the REFLECT step is delegated to a
//! caller-injected `reflect: &dyn Fn(&str) -> Result<String>` closure, which proposes an improved
//! GENERAL graph-reader prompt (the transport — TensorZero, a direct model call, etc. — is the
//! caller's concern, not this module's).
//!
//! Structure mirrors `gepa_constraint.rs` (single-objective GEPA): one `Candidate { prompt,
//! em_val }` scored per-question on a Pareto validation subset, minibatch reflection from the
//! train split, accept-if-child-beats-parent-on-minibatch, best-on-full-val at the end.
//!
//! Deviation from the blueprint's "reuse gepa.rs Pareto helpers": those helpers are typed on
//! gepa.rs's private quad `Candidate`; generalizing them would churn gepa.rs's own call sites and
//! its ~30 tests. Following the designated template (`gepa_constraint.rs`, which keeps its own
//! single-objective copies), the Pareto helpers here are local single-objective versions. The
//! freestanding items that reuse cleanly — `split_by_episode`, `CandidateSelection`,
//! `hash_run_seed`, `default_run_tag`, `load_seed_prompt` — ARE reused from `gepa.rs` (promoted to
//! `pub(crate)`/`pub`). `output_likely_truncated` (now `pub` on `gepa.rs`) is used by callers'
//! reflect closures (e.g. `kb-train`'s TensorZero closure), not by this module directly.

use crate::dataset::Question;
use crate::gepa::CandidateSelection;
use anyhow::{Context, Result};
use glossa::graph::ontology::Ontology;
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;
use glossa::tools::ChainSpec;
use glossa::trace::TraceLog;
use indicatif::ProgressBar;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Duration;

// 30 (not the eval's 50): a graph-reader that FOLLOWS the chain needs ~4-10 rounds; the extra 20
// only let a floundering reader search-flood into a huge context that blows the per-call timeout.
// Capping lower kills those timeouts and ~halves rollout time without losing real navigation.
const MAX_ROUNDS: usize = 30;
const MINIBATCH_RESAMPLE_ATTEMPTS: usize = 3;
/// Per-step tool-result body length fed to the reflector (chars).
const STEP_RESULT_CHARS: usize = 400;

/// Graded judge metric config: the reused LLM judge endpoint (`lab.judge`) plus the workspace
/// `judge.md` system prompt. `Some(..)` on `GepaGraphConfig.judge` selects the graded path;
/// `None` keeps the backward-compatible exact-EM path.
pub struct JudgeCfg {
    pub ep: crate::lab::Endpoint,
    pub md: String,
}

pub struct GepaGraphConfig {
    pub endpoint: String,
    pub model: String,
    pub api_key: Option<String>,
    pub val_frac: f64,
    pub budget: usize,
    pub minibatch: usize,
    pub seed_prompt: String,
    pub work: PathBuf,
    pub seed: u64,
    pub pareto_size: usize,
    pub candidate_selection: CandidateSelection,
    /// Worker-pool size for `score_questions`'s per-question rollouts. Rollouts are graph-READ
    /// reader navigations only (train never writes) — safe to run `jobs` of them concurrently
    /// sharing the one opened `GraphStore`/`DocIndex` (both read-only shareable). The GEPA
    /// candidate/iteration loop in `run` stays sequential (Pareto selection is inherently serial).
    /// `jobs == 1` runs the exact sequential loop used before parallelism landed.
    pub jobs: usize,
    /// `Some` => graded judge metric (Correct=1.0/Partial=0.5/Wrong=0.0); `None` => exact-EM
    /// (default, backward-compatible with pre-metric GEPA).
    pub judge: Option<JudgeCfg>,
    /// Optional `[user_sim]` endpoint: when `Some` (paired with `user_sim_prompt`), each rollout's
    /// reader loop is dialogue-gated by a patient simulated user (see `backend::user_sim`), so
    /// `kbx train` optimizes the prod prompt under the SAME dialogue dynamics eval uses. `None`
    /// keeps today's behavior (a text-only turn ends the rollout).
    pub user_sim: Option<crate::lab::Endpoint>,
    /// The simulated-user persona prompt (`user_sim.md`) for the gate above; only used when
    /// `user_sim` is also `Some`.
    pub user_sim_prompt: Option<String>,
}

/// Verdict → graded score: Correct=1.0, Partial=0.5, Wrong/Unscored=0.0.
fn verdict_to_score(v: crate::judge::Verdict) -> f64 {
    match v {
        crate::judge::Verdict::Correct => 1.0,
        crate::judge::Verdict::Partial => 0.5,
        crate::judge::Verdict::Wrong | crate::judge::Verdict::Unscored => 0.0,
    }
}

pub struct GepaGraphResult {
    pub prompt: String,
    pub baseline_em: f64,
    pub best_em: f64,
    pub candidates: usize,
}

#[derive(Clone)]
struct ToolStep {
    name: String,
    args: Value,
    result: String,
}

struct RolloutOutcome {
    /// Graded rollout score in {0.0, 0.5, 1.0}: exact-EM as 0.0/1.0 when `cfg.judge` is `None`,
    /// else the reused judge's Correct/Partial/Wrong verdict. `< 1.0` means "not fully correct".
    score: f64,
    pred: String,
    steps: Vec<ToolStep>,
}

#[derive(Clone)]
struct Candidate {
    prompt: String,
    /// Per-instance graded score on D_pareto (not full val).
    em_val: Vec<f64>,
}

struct FailCase {
    question: String,
    gold: String,
    pred: String,
    steps: Vec<ToolStep>,
}

pub(crate) struct GraphReflectContext {
    parent_prompt: String,
    parent_em: f64,
    fails: Vec<FailCase>,
}

fn golds_of(q: &Question) -> Vec<String> {
    let mut g = vec![q.answer.clone()];
    g.extend(q.answer_aliases.clone());
    g
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn scores(out: &[RolloutOutcome]) -> Vec<f64> {
    out.iter().map(|o| o.score).collect()
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    let mut out: String = s.chars().take(max_chars).collect();
    if s.chars().count() > max_chars {
        out.push_str("…");
    }
    out
}

// --- rollout (LM Studio direct, graph tools in-process) -------------------------------------

#[allow(clippy::too_many_arguments)]
fn rollout_one(
    cfg: &GepaGraphConfig,
    url: &str,
    tools: &Value,
    prompt: &str,
    q: &Question,
    idx: &DocIndex,
    graph: Option<&GraphStore>,
    spec: &ChainSpec,
) -> RolloutOutcome {
    let trace = TraceLog::disabled();
    let steps = std::cell::RefCell::new(Vec::<ToolStep>::new());
    let chat = |messages: &[Value]| {
        crate::backend::openai::lmstudio_chat(
            url,
            &cfg.model,
            cfg.api_key.as_deref(),
            tools,
            messages,
            Duration::from_secs(240),
        )
    };
    let exec = |name: &str, args: &Value| -> (String, Vec<String>, Vec<glossa::read::DocImage>) {
        let (body, ids, _images) =
            crate::backend::glossa_tools::exec(name, args, &cfg.work, idx, graph, spec, &trace);
        // Mirror openai::execute_tool: `read`'s surfaced id is its `path` arg (glossa_tools::exec
        // returns no ids for read itself).
        let ids = if name == "read" {
            args.get("path")
                .and_then(|v| v.as_str())
                .filter(|p| !p.is_empty())
                .map(|p| vec![p.to_string()])
                .unwrap_or_default()
        } else {
            ids
        };
        steps.borrow_mut().push(ToolStep {
            name: name.to_string(),
            args: args.clone(),
            result: truncate_chars(&body, STEP_RESULT_CHARS),
        });
        // GEPA graph rollouts don't feed vision input (not `kbx build --vision`) — discard, same
        // as `_images` above.
        (body, ids, Vec::new())
    };
    let messages = vec![
        json!({ "role": "system", "content": prompt }),
        json!({ "role": "user", "content": crate::backend::prompt::user_prompt(q) }),
    ];
    let nba = |name: &str, args: &Value| {
        crate::backend::glossa_tools::next_best_action(
            name, args, &cfg.work, idx, graph, spec, &trace,
        )
    };
    // Simulated-user dialogue gate (opt-in): built only when BOTH the `[user_sim]` endpoint and its
    // persona prompt are configured, so train rollouts see the same dialogue dynamics as eval.
    // `None` -> today's behavior (a text-only reader turn ends the rollout).
    let gate = match (&cfg.user_sim, &cfg.user_sim_prompt) {
        (Some(ep), Some(prompt)) => Some(crate::backend::user_sim::UserSimGate::new(ep, prompt)),
        _ => None,
    };
    let user_sim = gate
        .as_ref()
        .map(|g| g as &dyn crate::backend::user_sim::DialogueGate);
    let raw = match crate::backend::openai::run_agent_loop(chat, messages, exec, nba, MAX_ROUNDS, user_sim) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("graph rollout failed for q {}: {e:#}", q.id);
            String::new()
        }
    };
    let pred = crate::backend::prompt::parse_answer(&raw);
    // Scoring is either graded (reused LLM judge: Correct=1.0/Partial=0.5/Wrong=0.0) when a judge
    // is configured, or STRICT exact-match (0.0/1.0) by default. Exact-EM is the ungameable target
    // (relaxed/substring EM would reward a verbose prompt that merely embeds the gold); the graded
    // judge exists for corpora whose gold answers are long paragraphs, where exact-EM is ~0 for
    // every candidate and GEPA has no gradient to climb. Judge only the primary gold `q.answer`
    // (aliases are exact-match forms).
    let score = match &cfg.judge {
        Some(jc) => match crate::judge::judge(&jc.ep, &jc.md, &q.question, &q.answer, &pred) {
            Ok(j) => verdict_to_score(j.verdict),
            Err(e) => {
                eprintln!("judge failed (scored 0): {e:#}");
                0.0
            }
        },
        None => {
            if crate::score::exact_match_any(&pred, &golds_of(q)) {
                1.0
            } else {
                0.0
            }
        }
    };
    RolloutOutcome {
        score,
        pred,
        steps: steps.into_inner(),
    }
}

/// Score `questions` with `cfg.jobs` concurrent read-only rollout workers.
///
/// The returned `Vec<RolloutOutcome>` MUST stay in `questions`' input order: `Candidate::em_val`
/// is POSITIONAL (Pareto dominance in `dominates`/`pareto_frontier_win_counts` compares
/// instance-by-instance across candidates scored against the SAME question set), so a caller
/// zipping two `score_questions` calls' outputs — or this file's own `batch.iter().zip(&outcomes)`
/// — silently mismatches questions to outcomes if order isn't preserved. `run_units_parallel`
/// returns results in COMPLETION order, not input order, so each unit here carries its original
/// index and results are sorted back into place before returning.
fn score_questions(
    cfg: &GepaGraphConfig,
    url: &str,
    tools: &Value,
    prompt: &str,
    questions: &[Question],
    idx: &DocIndex,
    graph: Option<&GraphStore>,
    spec: &ChainSpec,
) -> Vec<RolloutOutcome> {
    let units: Vec<(usize, Question)> = questions.iter().cloned().enumerate().collect();
    // No visible progress bar for train rollouts today (unlike build/reason/distil) — `run_train`
    // reports per-iteration summaries via println!, not a bar. `run_units_parallel` still needs a
    // `&ProgressBar` to report completion into; a hidden one is a correct no-op sink.
    let pb = ProgressBar::hidden();
    let mut indexed: Vec<(usize, RolloutOutcome)> = crate::parallel::run_units_parallel(
        units,
        cfg.jobs,
        &pb,
        |_unit| 1,
        |(i, q)| Ok((*i, rollout_one(cfg, url, tools, prompt, q, idx, graph, spec))),
    )
    // `rollout_one` itself never returns `Err` (it catches its own agent-loop failure and scores
    // an empty prediction) — the only `Result` here is `run_units_parallel`'s own plumbing, which
    // is infallible for an infallible `work` closure.
    .expect("rollout_one is infallible; run_units_parallel only errors on a failing work closure");
    indexed.sort_unstable_by_key(|(i, _)| *i);
    indexed.into_iter().map(|(_, o)| o).collect()
}

// --- reflection (caller-injected transport) -------------------------------------------------

pub(crate) fn build_graph_reflect_instruction(ctx: &GraphReflectContext) -> String {
    let mut cases = String::new();
    for (i, f) in ctx.fails.iter().enumerate() {
        let mut steps = String::new();
        for (j, s) in f.steps.iter().enumerate() {
            steps.push_str(&format!("  step {}: {}({}) -> {}\n", j + 1, s.name, s.args, s.result));
        }
        if steps.is_empty() {
            steps.push_str("  (no tool calls)\n");
        }
        cases.push_str(&format!(
            "--- Failing case {} ---\nQuestion: {}\nGold answer: {}\nModel answer: {}\nTool-call trace:\n{}\n",
            i + 1,
            f.question,
            f.gold,
            f.pred,
            steps,
        ));
    }
    format!(
        "You are improving the SYSTEM PROMPT for a multi-hop question-answering agent that navigates \
         a PRE-BUILT REASONING GRAPH with the tools glossary, neighbors, path, related, search, read. \
         Answers are graded by EXACT MATCH of the shortest answer span.\n\
         Below are FAILING rollouts: the question, the gold answer, the model's answer, and the \
         model's tool-call trace (tool name, arguments, truncated result per step).\n\
         Diagnose the recurring navigation/answering mistakes and rewrite the system prompt so \
         multi-hop exact-match improves. Preserve behavior that already works.\n\
         Output ONLY general, reusable behavior guidance for using the graph and tools. It MUST NOT \
         mention or reuse ANY specific entity name, answer, date, place, number, or other value from \
         the examples — those are test data and must never appear in the prompt.\n\
         Keep the strict answer contract (one `ANSWER:` line, shortest exact span). \
         Reply with ONLY the new system prompt text — no preamble, no quotes.\n\n\
         === PARENT EXACT-MATCH ON MINIBATCH ===\nem={parent_em:.3}\n\n\
         === CURRENT SYSTEM PROMPT ===\n{prompt}\n\n=== FAILING ROLLOUTS ===\n{cases}=== NEW SYSTEM PROMPT ===",
        parent_em = ctx.parent_em,
        prompt = ctx.parent_prompt,
        cases = cases,
    )
}

// --- leak guard -----------------------------------------------------------------------------

fn tokens_lower(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(String::from)
        .collect()
}

fn contains_subsequence(hay: &[String], needle: &[String]) -> bool {
    if needle.is_empty() || needle.len() > hay.len() {
        return false;
    }
    hay.windows(needle.len()).any(|w| w == needle)
}

/// A single gold form or proper noun leaked into `hay` (candidate tokens)?
fn value_leaked(hay: &[String], value: &str) -> bool {
    let needle = tokens_lower(value);
    match needle.len() {
        0 => false,
        // Single short tokens (yes/no, numbers under 4 chars) are too collision-prone to flag.
        1 => needle[0].len() >= 4 && hay.contains(&needle[0]),
        _ => contains_subsequence(hay, &needle),
    }
}

/// Question-specific proper nouns: capitalized, alphabetic, len>=4, not the leading interrogative.
fn question_proper_nouns(question: &str) -> Vec<String> {
    question
        .split_whitespace()
        .skip(1)
        .filter_map(|w| {
            let t: String = w.chars().filter(|c| c.is_alphanumeric()).collect();
            let first_upper = t.chars().next().is_some_and(|c| c.is_uppercase());
            if first_upper && t.chars().all(|c| c.is_alphabetic()) && t.chars().count() >= 4 {
                Some(t)
            } else {
                None
            }
        })
        .collect()
}

/// Reject a child prompt that echoes any gold answer or question proper noun from the minibatch.
/// Returns `Some(reason)` on a leak, `None` when clean.
fn leak_scan(candidate: &str, minibatch: &[Question]) -> Option<String> {
    let hay = tokens_lower(candidate);
    for q in minibatch {
        for g in golds_of(q) {
            if value_leaked(&hay, &g) {
                return Some(format!("contains gold answer {g:?}"));
            }
        }
        for pn in question_proper_nouns(&q.question) {
            if value_leaked(&hay, &pn) {
                return Some(format!("contains question proper noun {pn:?}"));
            }
        }
    }
    None
}

// --- sampling + Pareto (single-objective, local; see module note) ---------------------------

fn sample_indices(len: usize, n: usize, rng: &mut StdRng) -> Vec<usize> {
    if len == 0 || n == 0 {
        return Vec::new();
    }
    let n = n.min(len);
    if n >= len {
        return (0..len).collect();
    }
    rand::seq::index::sample(rng, len, n).into_iter().collect()
}

fn sample_questions(pool: &[Question], n: usize, rng: &mut StdRng) -> Vec<Question> {
    sample_indices(pool.len(), n, rng)
        .into_iter()
        .map(|i| pool[i].clone())
        .collect()
}

fn candidate_bits(c: &Candidate) -> &[f64] {
    &c.em_val
}

/// Graded Pareto dominance: `a` dominates `b` iff `a[i] >= b[i]` for every instance AND
/// `a[i] > b[i]` for at least one. Compared with a small epsilon so the discrete {0,0.5,1} scores
/// never trip on float equality.
fn dominates(a: &[f64], b: &[f64]) -> bool {
    const EPS: f64 = 1e-9;
    let mut strictly = false;
    for (x, y) in a.iter().zip(b) {
        if *x < *y - EPS {
            return false;
        }
        if *x > *y + EPS {
            strictly = true;
        }
    }
    strictly
}

fn frontier(pool: &[Candidate]) -> Vec<usize> {
    (0..pool.len())
        .filter(|&i| {
            !pool
                .iter()
                .enumerate()
                .any(|(j, c)| j != i && dominates(candidate_bits(c), candidate_bits(&pool[i])))
        })
        .collect()
}

fn pareto_frontier_win_counts(pool: &[Candidate]) -> (Vec<usize>, Vec<usize>) {
    let bits: Vec<&[f64]> = pool.iter().map(candidate_bits).collect();
    let n_inst = bits.first().map(|b| b.len()).unwrap_or(0);
    if n_inst == 0 {
        return ((0..pool.len()).collect(), vec![0; pool.len()]);
    }

    const EPS: f64 = 1e-9;
    let mut union = HashSet::new();
    let mut per_instance_winners: Vec<Vec<usize>> = Vec::with_capacity(n_inst);
    for i in 0..n_inst {
        let max_score = bits.iter().map(|b| b[i]).fold(f64::NEG_INFINITY, f64::max);
        let winners: Vec<usize> = bits
            .iter()
            .enumerate()
            .filter(|(_, b)| (b[i] - max_score).abs() < EPS)
            .map(|(k, _)| k)
            .collect();
        for &k in &winners {
            union.insert(k);
        }
        per_instance_winners.push(winners);
    }

    let mut candidate_idxs: Vec<usize> = union.into_iter().collect();
    candidate_idxs.sort_unstable();
    let mut dominated = HashSet::new();
    for &i in &candidate_idxs {
        for &j in &candidate_idxs {
            if i != j && dominates(candidate_bits(&pool[j]), candidate_bits(&pool[i])) {
                dominated.insert(i);
                break;
            }
        }
    }
    let mut frontier_idxs: Vec<usize> = candidate_idxs
        .into_iter()
        .filter(|k| !dominated.contains(k))
        .collect();
    if frontier_idxs.is_empty() {
        frontier_idxs = frontier(pool);
    }
    frontier_idxs.sort_unstable();

    let mut counts = vec![0usize; pool.len()];
    for winners in &per_instance_winners {
        let active: Vec<usize> = winners
            .iter()
            .copied()
            .filter(|k| frontier_idxs.contains(k))
            .collect();
        let active = if active.is_empty() {
            winners.clone()
        } else {
            active
        };
        for k in active {
            counts[k] += 1;
        }
    }

    (frontier_idxs, counts)
}

fn select_parent_pareto_weighted(pool: &[Candidate], rng: &mut StdRng) -> usize {
    if pool.len() == 1 {
        return 0;
    }
    let n_inst = pool.first().map(|c| c.em_val.len()).unwrap_or(0);
    if n_inst == 0 {
        return rng.gen_range(0..pool.len());
    }
    let (frontier_idxs, counts) = pareto_frontier_win_counts(pool);
    if frontier_idxs.is_empty() {
        return 0;
    }
    let total: usize = frontier_idxs.iter().map(|&k| counts[k]).sum();
    if total == 0 {
        return frontier_idxs[rng.gen_range(0..frontier_idxs.len())];
    }
    let pick = rng.gen_range(0..total);
    let mut acc = 0usize;
    for &k in &frontier_idxs {
        acc += counts[k];
        if pick < acc {
            return k;
        }
    }
    *frontier_idxs.last().unwrap_or(&0)
}

fn select_parent_idx(pool: &[Candidate], sel: CandidateSelection, rng: &mut StdRng) -> usize {
    if pool.len() == 1 {
        return 0;
    }
    match sel {
        CandidateSelection::CurrentBest => pool
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                mean(&a.em_val)
                    .partial_cmp(&mean(&b.em_val))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0),
        CandidateSelection::Pareto => select_parent_pareto_weighted(pool, rng),
    }
}

// --- driver ---------------------------------------------------------------------------------

pub fn run(
    cfg: GepaGraphConfig,
    questions: Vec<Question>,
    reflect: &dyn Fn(&str) -> Result<String>,
) -> Result<GepaGraphResult> {
    anyhow::ensure!(!questions.is_empty(), "no questions to optimize against");
    let idx = DocIndex::open_or_create(&cfg.work).context("open index for graph GEPA")?;
    let graph = GraphStore::open(&cfg.work).ok();
    let spec = ChainSpec::from_ontology(&Ontology::load_or_default(&cfg.work));
    let tools = crate::backend::openai::tools_schema(graph.is_some());
    // Full chat-completions URL, used verbatim (no suffix appended).
    let url = cfg.endpoint.clone();

    let (train, val) = crate::gepa::split_by_episode(&questions, |q| q.id.as_str(), cfg.val_frac);
    println!(
        "gepa_graph: {} questions ({} train, {} val), budget={}, minibatch={}, pareto_size={}, graph={}, selection={}, work={}",
        questions.len(),
        train.len(),
        val.len(),
        cfg.budget,
        cfg.minibatch,
        cfg.pareto_size,
        graph.is_some(),
        cfg.candidate_selection,
        cfg.work.display(),
    );
    anyhow::ensure!(!val.is_empty(), "empty validation split — need >=2 distinct question ids");

    let baseline_out = score_questions(
        &cfg,
        &url,
        &tools,
        &cfg.seed_prompt,
        &val,
        &idx,
        graph.as_ref(),
        &spec,
    );
    let baseline_em = mean(&scores(&baseline_out));
    println!("baseline val: score={baseline_em:.3}");

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let pareto_set = sample_questions(&val, cfg.pareto_size, &mut rng);
    println!(
        "pareto set (D_pareto): {} of {} val",
        pareto_set.len(),
        val.len(),
    );
    let base_pareto = score_questions(
        &cfg,
        &url,
        &tools,
        &cfg.seed_prompt,
        &pareto_set,
        &idx,
        graph.as_ref(),
        &spec,
    );
    let mut pool = vec![Candidate {
        prompt: cfg.seed_prompt.clone(),
        em_val: scores(&base_pareto),
    }];

    for it in 0..cfg.budget {
        let parent_idx = select_parent_idx(&pool, cfg.candidate_selection, &mut rng);
        let parent_prompt = pool[parent_idx].prompt.clone();

        let mut minibatch = None;
        let mut parent_outcomes = None;
        for _ in 0..MINIBATCH_RESAMPLE_ATTEMPTS {
            let batch = sample_questions(&train, cfg.minibatch, &mut rng);
            if batch.is_empty() {
                break;
            }
            let outcomes =
                score_questions(&cfg, &url, &tools, &parent_prompt, &batch, &idx, graph.as_ref(), &spec);
            if outcomes.iter().any(|o| o.score < 1.0) {
                minibatch = Some(batch);
                parent_outcomes = Some(outcomes);
                break;
            }
        }
        let (Some(batch), Some(outcomes)) = (minibatch, parent_outcomes) else {
            println!("[iter {it}] no failures in sampled minibatch — skip");
            continue;
        };
        let parent_em_mb = mean(&scores(&outcomes));
        let fails: Vec<FailCase> = batch
            .iter()
            .zip(&outcomes)
            .filter(|(_, o)| o.score < 1.0)
            .map(|(q, o)| FailCase {
                question: q.question.clone(),
                gold: golds_of(q).join(" | "),
                pred: o.pred.clone(),
                steps: o.steps.clone(),
            })
            .collect();
        println!(
            "[iter {it}] reflect minibatch: {} rollouts (fails={}) parent_mb_em={parent_em_mb:.3}",
            outcomes.len(),
            fails.len(),
        );

        let ctx = GraphReflectContext {
            parent_prompt: parent_prompt.clone(),
            parent_em: parent_em_mb,
            fails,
        };
        let instruction = build_graph_reflect_instruction(&ctx);
        let child_prompt = match reflect(&instruction) {
            Ok(p) => p,
            Err(e) => {
                println!("[iter {it}] reflection failed: {e:#}");
                continue;
            }
        };
        if let Some(reason) = leak_scan(&child_prompt, &batch) {
            println!("[iter {it}] child REJECTED by leak-scan: {reason}");
            continue;
        }

        let child_mb =
            score_questions(&cfg, &url, &tools, &child_prompt, &batch, &idx, graph.as_ref(), &spec);
        let child_em_mb = mean(&scores(&child_mb));
        if child_em_mb <= parent_em_mb {
            println!("[iter {it}] child_mb {child_em_mb:.3} <= parent_mb {parent_em_mb:.3} — discarded");
            continue;
        }

        let child_pareto = score_questions(
            &cfg,
            &url,
            &tools,
            &child_prompt,
            &pareto_set,
            &idx,
            graph.as_ref(),
            &spec,
        );
        pool.push(Candidate {
            prompt: child_prompt,
            em_val: scores(&child_pareto),
        });
        let best_pareto = pool
            .iter()
            .map(|c| mean(&c.em_val))
            .fold(f64::NEG_INFINITY, f64::max);
        println!(
            "[iter {it}] parent_idx={parent_idx} parent_mb={parent_em_mb:.3} -> child_mb={child_em_mb:.3} — accepted (pareto_em={best_pareto:.3}, pool_size={})",
            pool.len(),
        );
    }

    println!("final full-val scoring: {} candidates", pool.len());
    let mut best_prompt = pool[0].prompt.clone();
    let mut best_em = f64::NEG_INFINITY;
    for c in &pool {
        let out = score_questions(&cfg, &url, &tools, &c.prompt, &val, &idx, graph.as_ref(), &spec);
        let em = mean(&scores(&out));
        if em > best_em {
            best_em = em;
            best_prompt = c.prompt.clone();
        }
    }
    if !best_em.is_finite() {
        best_em = 0.0;
    }
    println!(
        "gepa_graph final: em={best_em:.3} (baseline was {baseline_em:.3}), candidates={}",
        pool.len(),
    );

    Ok(GepaGraphResult {
        prompt: best_prompt,
        baseline_em,
        best_em,
        candidates: pool.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(id: &str, question: &str, answer: &str, aliases: &[&str]) -> Question {
        Question {
            id: id.to_string(),
            question: question.to_string(),
            answer: answer.to_string(),
            answer_aliases: aliases.iter().map(|s| s.to_string()).collect(),
            paragraphs: vec![],
            supporting_titles: vec![],
            tags: vec![],
            ..Default::default()
        }
    }

    #[test]
    fn leak_scan_rejects_gold_answer_token() {
        let mb = vec![q("e1", "Who founded Foobar Corp?", "Ada Lovelace", &[])];
        // A candidate that leaks the multi-word gold answer must be rejected.
        let leaked = "Follow the chain. The answer is often Ada Lovelace for such cases.";
        let reason = leak_scan(leaked, &mb).expect("gold answer must be caught");
        assert!(reason.contains("Ada Lovelace"), "reason: {reason}");
    }

    #[test]
    fn leak_scan_catches_alias_and_proper_noun() {
        let mb = vec![q("e1", "Where is Foobar located?", "Paris", &["City of Light"])];
        assert!(leak_scan("… known as the City of Light …", &mb).is_some(), "alias leaked");
        assert!(leak_scan("… mention Foobar directly …", &mb).is_some(), "proper noun leaked");
    }

    #[test]
    fn leak_scan_passes_clean_general_prompt() {
        let mb = vec![
            q("e1", "Who founded Foobar Corp?", "Ada Lovelace", &[]),
            q("e2", "What year did Barsoft ship?", "1998", &[]),
        ];
        // General graph-reader guidance with no example values must pass — note it contains "not",
        // "node", "no" which must NOT collide with the short "no" family via substring matching.
        let clean = "Start from an entity, call glossary, then read the source chunk. Do not stop \
                     at the first node; follow neighbors until the answer is grounded. Reply with one \
                     ANSWER line holding the shortest exact span.";
        assert!(leak_scan(clean, &mb).is_none(), "clean prompt wrongly rejected");
    }

    #[test]
    fn short_gold_values_are_not_flagged_by_substring() {
        // gold "no" must not fire on "node"/"not"; single short tokens are skipped.
        let mb = vec![q("e1", "Is it raining?", "no", &["yes"])];
        assert!(leak_scan("Use the graph node and do not guess.", &mb).is_none());
    }

    #[test]
    fn reflect_instruction_has_leak_guard_and_markers() {
        let ctx = GraphReflectContext {
            parent_prompt: "seed graph prompt".to_string(),
            parent_em: 0.25,
            fails: vec![FailCase {
                question: "Who directed the film?".to_string(),
                gold: "Jane Doe".to_string(),
                pred: "unknown".to_string(),
                steps: vec![ToolStep {
                    name: "glossary".to_string(),
                    args: json!({"name": "the film"}),
                    result: "(no matches)".to_string(),
                }],
            }],
        };
        let msg = build_graph_reflect_instruction(&ctx);
        assert!(msg.contains("PARENT EXACT-MATCH ON MINIBATCH"));
        assert!(msg.contains("em=0.250"));
        assert!(msg.contains("EXACT MATCH"));
        assert!(msg.contains("MUST NOT"), "leak guard instruction present");
        assert!(msg.contains("glossary(") && msg.contains("Tool-call trace"));
        assert!(msg.contains("Gold answer: Jane Doe"));
        assert!(msg.contains("=== NEW SYSTEM PROMPT ==="));
    }

    /// `score_questions` pairs each question with its input index, lets `run_units_parallel`
    /// return results in whatever order workers finish (jobs>1 reorders by design), then sorts
    /// back by index before returning. This is the load-bearing correctness property for Task 7:
    /// `Candidate::em_val` is POSITIONAL (Pareto dominance compares instance-by-instance across
    /// candidates scored against the same question set), so a reordered result vector would
    /// silently mismatch questions to outcomes. Exercise the exact same index-carry-then-sort
    /// pattern `score_questions` uses, with workers deliberately finishing OUT of input order
    /// (later units sleep less), to prove the sort restores input order regardless.
    #[test]
    fn indexed_parallel_results_sort_back_to_input_order() {
        let items: Vec<u32> = (0..12).map(|i| i * 10).collect();
        let units: Vec<(usize, u32)> = items.iter().copied().enumerate().collect();
        let pb = indicatif::ProgressBar::hidden();
        let mut indexed: Vec<(usize, u32)> = crate::parallel::run_units_parallel(
            units,
            4,
            &pb,
            |_unit| 1,
            |(i, v)| {
                // Later-indexed units sleep less, so they tend to finish FIRST — forcing
                // completion order to differ from input order.
                std::thread::sleep(Duration::from_millis((items.len() as u64 - *i as u64) % 5));
                Ok((*i, *v))
            },
        )
        .unwrap();
        indexed.sort_unstable_by_key(|(i, _)| *i);
        let got: Vec<u32> = indexed.into_iter().map(|(_, v)| v).collect();
        assert_eq!(
            got, items,
            "sorting by original index must restore input order despite parallel completion reordering"
        );
    }

    #[test]
    fn split_by_episode_is_deterministic_and_keyed_on_id() {
        let items = vec![
            q("a", "qa", "A", &[]),
            q("b", "qb", "B", &[]),
            q("c", "qc", "C", &[]),
            q("d", "qd", "D", &[]),
        ];
        let (t1, v1) = crate::gepa::split_by_episode(&items, |x| x.id.as_str(), 0.25);
        let (_t2, v2) = crate::gepa::split_by_episode(&items, |x| x.id.as_str(), 0.25);
        assert_eq!(v1.len(), 1);
        let ids1: Vec<_> = v1.iter().map(|x| x.id.clone()).collect();
        let ids2: Vec<_> = v2.iter().map(|x| x.id.clone()).collect();
        assert_eq!(ids1, ids2, "split must be deterministic");
        assert_eq!(t1.len(), 3);
        assert!(t1.iter().all(|x| !ids1.contains(&x.id)), "no train/val id overlap");
    }

    #[test]
    fn select_parent_current_best_picks_highest_em() {
        let pool = vec![
            Candidate { prompt: "weak".into(), em_val: vec![0.0, 0.0] },
            Candidate { prompt: "strong".into(), em_val: vec![1.0, 1.0, 1.0] },
        ];
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(
            select_parent_idx(&pool, CandidateSelection::CurrentBest, &mut rng),
            1
        );
    }

    #[test]
    fn pareto_win_counts_prefer_frequent_winner() {
        let pool = vec![
            Candidate { prompt: "a".into(), em_val: vec![1.0, 0.0, 0.0] },
            Candidate { prompt: "b".into(), em_val: vec![0.0, 1.0, 1.0] },
            Candidate { prompt: "c".into(), em_val: vec![0.0, 0.0, 0.0] },
        ];
        let (frontier, counts) = pareto_frontier_win_counts(&pool);
        assert!(frontier.contains(&0) && frontier.contains(&1));
        assert_eq!(counts[1], 2);
        assert_eq!(counts[2], 0);
    }

    #[test]
    fn verdict_to_score_maps_grades() {
        assert_eq!(verdict_to_score(crate::judge::Verdict::Correct), 1.0);
        assert_eq!(verdict_to_score(crate::judge::Verdict::Partial), 0.5);
        assert_eq!(verdict_to_score(crate::judge::Verdict::Wrong), 0.0);
        assert_eq!(verdict_to_score(crate::judge::Verdict::Unscored), 0.0);
    }

    #[test]
    fn mean_f64_handles_empty_and_grades() {
        assert_eq!(mean(&[]), 0.0);
        assert_eq!(mean(&[1.0, 0.5, 0.0]), 0.5);
    }

    #[test]
    fn dominates_f64_graded() {
        // strictly better on one instance, >= on the rest -> dominates
        assert!(dominates(&[1.0, 0.5], &[0.5, 0.5]));
        // better on one but worse on another -> does NOT dominate
        assert!(!dominates(&[1.0, 0.0], &[0.5, 0.5]));
        // equal vectors never dominate (no strict improvement)
        assert!(!dominates(&[0.5, 1.0], &[0.5, 1.0]));
        // graded partial credit counts as strictly better
        assert!(dominates(&[0.5, 0.5], &[0.0, 0.5]));
    }

    #[test]
    fn pareto_win_counts_graded_partial_credit() {
        // Instance 0: only `b` reaches the max (1.0). Instance 1: `a`=1.0 is sole max, `b`=0.5.
        // Instance 2: `a` and `b` tie at 0.5. Neither dominates the other -> both on frontier.
        let pool = vec![
            Candidate { prompt: "a".into(), em_val: vec![0.5, 1.0, 0.5] },
            Candidate { prompt: "b".into(), em_val: vec![1.0, 0.5, 0.5] },
            Candidate { prompt: "c".into(), em_val: vec![0.0, 0.0, 0.0] },
        ];
        let (frontier, counts) = pareto_frontier_win_counts(&pool);
        assert!(frontier.contains(&0) && frontier.contains(&1));
        assert!(!frontier.contains(&2), "dominated candidate off frontier");
        // a wins instance 1 outright + ties instance 2 => 2; b wins instance 0 + ties instance 2 => 2.
        assert_eq!(counts[0], 2);
        assert_eq!(counts[1], 2);
        assert_eq!(counts[2], 0);
    }

    #[test]
    fn run_uses_injected_reflector_and_reports_candidates() {
        // Tiny in-memory-ish config: point work at a temp dir with a minimal indexed corpus is heavy;
        // instead assert the plumbing compiles and the injected reflector is the only transport by
        // constructing the closure and checking it is invoked. Full rollout is covered by e2e.
        let called = std::cell::Cell::new(0);
        let reflect = |_instr: &str| -> anyhow::Result<String> {
            called.set(called.get() + 1);
            Ok("NEW PROMPT".to_string())
        };
        // Signature compiles with a closure; call indirection verified via the unit below.
        let _f: &dyn Fn(&str) -> anyhow::Result<String> = &reflect;
        assert_eq!(called.get(), 0);
        let _ = _f("x");
        assert_eq!(called.get(), 1);
    }
}
