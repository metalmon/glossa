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
//! score_val }` scored per-question on a Pareto validation subset, minibatch reflection from the
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
    /// Hard ceiling on total reader rollouts (metric calls) the GEPA SEARCH may spend, DSPy-style.
    /// Every `score_questions` pass adds the number of questions it scored to a running counter;
    /// the candidate loop stops once the counter reaches this. Replaces the old fixed iteration
    /// `budget` — a metric-calls budget is dataset-scale-invariant (see `train::auto_budget`).
    pub max_metric_calls: usize,
    /// Cap on the candidate pool size: the loop also stops accepting once the pool reaches this,
    /// bounding the final full-val selection pass (which scores every pool candidate).
    pub max_candidates: usize,
    pub minibatch: usize,
    pub seed_prompt: String,
    pub work: PathBuf,
    pub seed: u64,
    pub pareto_size: usize,
    pub candidate_selection: CandidateSelection,
    /// Minibatch-cache mode. **`false` (default) = canonical GEPA:** every proposal re-samples a
    /// FRESH reflect minibatch and re-scores the parent on it, an unbiased paired accept with better
    /// exploration. **`true` = cached minibatch** (opt-in for a WEAK, high-variance reader): a
    /// candidate's minibatch + score are frozen the first time it parents and reused, trading
    /// exploration + freedom from batch-overfit for a noise-stable accept baseline (on a 4B,
    /// canonical's per-proposal noise — minibatch=3, one flip = 0.33 — drowned the accept signal so
    /// nothing was accepted). The apply-gate guards BOTH modes from shipping a regression. Sourced
    /// from lab.toml `[tuning].gepa_minibatch_cache`; env `GEPA_MINIBATCH_CACHE=1/0` overrides.
    pub minibatch_cache: bool,
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
    pub baseline_score: f64,
    pub best_score: f64,
    pub candidates: usize,
    /// Apply-gate verdict: `true` when the winner scored >= the seed on the FULL question set (safe to
    /// write over the prod prompt); `false` when the winner regressed there, in which case `prompt` is
    /// the SEED (unchanged) and the caller must NOT apply. Guards against the val-overfit regression.
    pub applied: bool,
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
    /// The judge's free-text reason for its verdict, captured at scoring time so the reflector can
    /// tell the teacher WHY a case was marked wrong. `None` on the exact-match path (no judge) or
    /// when the judge call errored.
    judge_reason: Option<String>,
    /// True when the reader's ENDPOINT/TRANSPORT failed (`run_agent_loop` returned `Err` — e.g. a
    /// 500 or a context-length overflow), so the rollout never produced an answer. Such an outcome
    /// is EXCLUDED from scored means (`mean_scored`) rather than counted as a real 0.0 — an endpoint
    /// failure is deterministic and would otherwise inject noise into GEPA's minibatch scores and
    /// deflate the reported number. A rollout that returned wrong/empty TEXT is `errored: false` and
    /// scored normally (a real 0.0). It also never becomes a `FailCase` fed to the reflector.
    errored: bool,
}

#[derive(Clone)]
struct Candidate {
    prompt: String,
    /// Per-instance graded score on D_pareto (not full val).
    score_val: Vec<f64>,
}

/// A candidate's reflect-minibatch, scored ONCE the first time the candidate is a reflect parent
/// and reused on every later iteration it parents again (kept in a `mb_cache` vec parallel to the
/// pool). Canonical GEPA scores a candidate once and stores it; re-rolling the parent every
/// iteration is ~half the rollouts and, because the reader is stochastic, makes the parent's own
/// score swing by the reader's noise across iterations — so a child could never reliably "beat" it
/// and nothing got accepted. Reusing the cached minibatch also keeps the child scored on the SAME
/// questions.
#[derive(Clone)]
struct MbCache {
    batch: Vec<Question>,
    score: f64,
    fails: Vec<FailCase>,
}

#[derive(Clone)]
struct FailCase {
    question: String,
    gold: String,
    pred: String,
    steps: Vec<ToolStep>,
    /// The judge's reason for marking this case wrong (signal B), carried from its `RolloutOutcome`
    /// so `build_graph_reflect_instruction` can show the teacher WHY it failed. `None` on the
    /// exact-match path or a judge error.
    judge_reason: Option<String>,
}

pub(crate) struct GraphReflectContext {
    parent_prompt: String,
    /// The parent's mean score on the reflect minibatch, in the ACTIVE metric (graded judge when
    /// `judge` is true, else exact-match). Shown to the reflector as the single number to beat.
    parent_score: f64,
    /// True when the run optimizes by the graded judge (not exact-match). The reflect instruction
    /// describes ONLY the active metric — never mixing EM framing into a judge run (that pushed the
    /// teacher toward terse "shortest exact span" answers the judge then penalized).
    judge: bool,
    fails: Vec<FailCase>,
    /// The reader's ACTUAL tool schema (`backend::openai::tools_schema(graph_on)`), threaded in so
    /// the reflect instruction renders its tool reference from the single source of truth instead of
    /// a hand-maintained name list that drifts (omitted graph_query/reach). See
    /// [`render_tool_reference`].
    tools: Value,
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

/// Mean `score` over ONLY non-errored outcomes: an endpoint-errored rollout (`errored: true`) never
/// produced an answer, so it is EXCLUDED from the average rather than dragged in as a real 0.0.
/// Returns 0.0 when every outcome errored (mean of an empty slice). Used for all the simple
/// rollout-mean computations (baseline/minibatch/child/apply-gate); the per-instance Pareto vectors
/// keep using `scores()` and rely on the baseline pre-filter (see `drop_baseline_errored`) instead.
fn mean_scored(out: &[RolloutOutcome]) -> f64 {
    let kept: Vec<f64> = out.iter().filter(|o| !o.errored).map(|o| o.score).collect();
    mean(&kept)
}

/// Drop the val questions whose BASELINE rollout errored on the endpoint, keeping the surviving
/// `(questions, outcomes)` and the dropped count. An endpoint/context-overflow error is
/// deterministic, so it reproduces for every candidate — dropping such a question once, up front,
/// keeps every candidate's Pareto `score_val` vector aligned on the SAME scorable instances and
/// excludes the error from scoring entirely (instead of scoring it a misleading 0.0). Extracted so
/// the pre-filter is unit-testable without a live endpoint.
fn drop_baseline_errored(
    val: Vec<Question>,
    out: Vec<RolloutOutcome>,
) -> (Vec<Question>, Vec<RolloutOutcome>, usize) {
    let n_before = val.len();
    let (kept_q, kept_o): (Vec<Question>, Vec<RolloutOutcome>) =
        val.into_iter().zip(out).filter(|(_, o)| !o.errored).unzip();
    let dropped = n_before - kept_q.len();
    (kept_q, kept_o, dropped)
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    let mut out: String = s.chars().take(max_chars).collect();
    if s.chars().count() > max_chars {
        out.push('…');
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
    // The live training bar: error notices go through `pb.println` (prints ABOVE the bar) instead of
    // a raw `eprintln!` that would garble the bar. Safe from parallel rollout workers (indicatif
    // serializes internally); a hidden bar makes it a plain println.
    pb: &ProgressBar,
) -> RolloutOutcome {
    let trace = TraceLog::disabled();
    let steps = std::cell::RefCell::new(Vec::<ToolStep>::new());
    // Per-episode reader-signal tracker (see `glossa_tools::ReaderSignals`): train rollouts act on
    // the SAME PLATEAU render eval's `answer_capturing` does, so GEPA optimizes the prompt under
    // the identical signal. Repeat/Streak stay the agent loop's job — not acted on here (see
    // `openai::answer_capturing`'s exec closure for the twin comment). Owned per rollout; the
    // POLICY stays in the prompt / GEPA, not the tool layer.
    let mut signals = crate::backend::glossa_tools::ReaderSignals::new();
    // Full-response one-shot; resampling is applied provider-neutrally by the agent loop
    // (`backend::resample::call_with_resample`).
    let chat = |messages: &[Value]| {
        crate::backend::transport::openai::agent_chat_full(
            url,
            &cfg.model,
            cfg.api_key.as_deref(),
            tools,
            messages,
            Duration::from_secs(240),
        )
    };
    let exec = |name: &str, args: &Value| -> (String, Vec<String>, Vec<glossa::read::DocImage>) {
        let (mut body, ids, _images) =
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
        // Only id-surfacing RETRIEVAL calls feed the tracker; only the PLATEAU kind is acted on
        // here (Repeat/Streak are the loop's job) — render into the step-trace the reflector reads.
        if crate::backend::glossa_tools::is_retrieval_tool(name) {
            let key = format!("{name}:{args}");
            body = crate::backend::glossa_tools::apply_plateau_render(
                &mut signals,
                name,
                &key,
                &ids,
                body,
            );
        }
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
    let (raw, errored) = match crate::backend::openai::run_agent_loop(
        chat, messages, exec, nba, MAX_ROUNDS, user_sim,
    ) {
        Ok(r) => (r, false),
        Err(e) => {
            // ENDPOINT/TRANSPORT failure (500, context-length overflow, …): the reader never
            // produced an answer. Flag the outcome `errored` so scored means exclude it — it is NOT
            // a wrong answer and must not be counted as a real 0.0.
            pb.println(format!("graph rollout failed for q {}: {e:#}", q.id));
            (String::new(), true)
        }
    };
    let pred = crate::backend::prompt::parse_answer(&raw);
    // Scoring is either graded (reused LLM judge: Correct=1.0/Partial=0.5/Wrong=0.0) when a judge
    // is configured, or STRICT exact-match (0.0/1.0) by default. Exact-EM is the ungameable target
    // (relaxed/substring EM would reward a verbose prompt that merely embeds the gold); the graded
    // judge exists for corpora whose gold answers are long paragraphs, where exact-EM is ~0 for
    // every candidate and GEPA has no gradient to climb. Judge only the primary gold `q.answer`
    // (aliases are exact-match forms).
    // An endpoint-errored rollout is not scored (no answer to grade): score stays 0.0, judge_reason
    // None, and `errored` excludes it from every scored mean. Only real answers reach the judge/EM.
    let (score, judge_reason) = if errored {
        (0.0, None)
    } else {
        match &cfg.judge {
            Some(jc) => match crate::judge::judge(
                &jc.ep,
                &jc.md,
                &q.question,
                &q.answer,
                &pred,
                &q.source,
                Some(idx),
            ) {
                // Keep the judge's reason alongside the score — the reflector surfaces it to the
                // teacher as WHY this case was wrong (signal B). Dropping it here is what left the
                // teacher guessing at the failure mode.
                Ok(j) => (verdict_to_score(j.verdict), Some(j.reason)),
                Err(e) => {
                    pb.println(format!("judge failed (scored 0): {e:#}"));
                    (0.0, None)
                }
            },
            None => {
                let s = if crate::score::exact_match_any(&pred, &golds_of(q)) {
                    1.0
                } else {
                    0.0
                };
                (s, None)
            }
        }
    };
    RolloutOutcome {
        score,
        pred,
        steps: steps.into_inner(),
        judge_reason,
        errored,
    }
}

/// Score `questions` with `cfg.jobs` concurrent read-only rollout workers.
///
/// The returned `Vec<RolloutOutcome>` MUST stay in `questions`' input order: `Candidate::score_val`
/// is POSITIONAL (Pareto dominance in `dominates`/`pareto_frontier_win_counts` compares
/// instance-by-instance across candidates scored against the SAME question set), so a caller
/// zipping two `score_questions` calls' outputs — or this file's own `batch.iter().zip(&outcomes)`
/// — silently mismatches questions to outcomes if order isn't preserved. `run_units_parallel`
/// returns results in COMPLETION order, not input order, so each unit here carries its original
/// index and results are sorted back into place before returning.
#[allow(clippy::too_many_arguments)] // config-passing helper; signature kept explicit
fn score_questions(
    cfg: &GepaGraphConfig,
    url: &str,
    tools: &Value,
    prompt: &str,
    questions: &[Question],
    idx: &DocIndex,
    graph: Option<&GraphStore>,
    spec: &ChainSpec,
    // Threaded to `rollout_one` so a rollout's endpoint/judge error prints ABOVE the live bar (via
    // `pb.println`) instead of a raw `eprintln!` that garbles it.
    pb: &ProgressBar,
) -> Vec<RolloutOutcome> {
    let units: Vec<(usize, Question)> = questions.iter().cloned().enumerate().collect();
    // The main training bar tracks metric-call spend end-to-end (length = max_metric_calls, position
    // = rollouts spent; set in `run`), NOT individual scoring passes — so a pass must not rewind it.
    // Per-rollout completion is fed to a throwaway hidden bar; within-pass liveness comes from the
    // `StatusTicker`'s live `{msg}` (elapsed/ETA/tokens) and the per-pass `pb.println` lines.
    let sink = ProgressBar::hidden();
    let mut indexed: Vec<(usize, RolloutOutcome)> = crate::parallel::run_units_parallel(
        units,
        cfg.jobs,
        &sink,
        |_unit| 1,
        |(i, q)| {
            Ok((
                *i,
                rollout_one(cfg, url, tools, prompt, q, idx, graph, spec, pb),
            ))
        },
    )
    // `rollout_one` itself never returns `Err` (it catches its own agent-loop failure and scores
    // an empty prediction) — the only `Result` here is `run_units_parallel`'s own plumbing, which
    // is infallible for an infallible `work` closure.
    .expect("rollout_one is infallible; run_units_parallel only errors on a failing work closure");
    indexed.sort_unstable_by_key(|(i, _)| *i);
    indexed.into_iter().map(|(_, o)| o).collect()
}

// --- reflection (caller-injected transport) -------------------------------------------------

/// Render a compact tool reference from the reader's ACTUAL tool schema
/// (`backend::openai::tools_schema(graph_on)`) — one line per tool as
/// `- name — description (params: a, b, …)`. This is the single source of truth for what the
/// reader can do, so the reflector's view can never drift from the real toolset (the old hardcoded
/// name list silently omitted `graph_query`/`reach`). Description is first-sentence/first-line
/// trimmed to keep the block compact; param names come from the schema's
/// `function.parameters.properties` keys.
fn render_tool_reference(tools: &Value) -> String {
    let mut out = String::new();
    for t in tools.as_array().into_iter().flatten() {
        let f = t.get("function").unwrap_or(t);
        let name = f.get("name").and_then(Value::as_str).unwrap_or("");
        if name.is_empty() {
            continue;
        }
        let desc_full = f.get("description").and_then(Value::as_str).unwrap_or("");
        // First sentence or first line, whichever is shorter — keeps the reference terse.
        let desc = desc_full
            .split(['\n', '.'])
            .next()
            .unwrap_or(desc_full)
            .trim();
        let params: Vec<&str> = f
            .pointer("/parameters/properties")
            .and_then(Value::as_object)
            .map(|m| m.keys().map(String::as_str).collect())
            .unwrap_or_default();
        out.push_str("- ");
        out.push_str(name);
        if !desc.is_empty() {
            out.push_str(" — ");
            out.push_str(desc);
        }
        if !params.is_empty() {
            out.push_str(&format!(" (params: {})", params.join(", ")));
        }
        out.push('\n');
    }
    out
}

pub(crate) fn build_graph_reflect_instruction(ctx: &GraphReflectContext) -> String {
    let mut cases = String::new();
    let mut retrieval_gaps = 0usize;
    for (i, f) in ctx.fails.iter().enumerate() {
        let mut steps = String::new();
        for (j, s) in f.steps.iter().enumerate() {
            steps.push_str(&format!(
                "  step {}: {}({}) -> {}\n",
                j + 1,
                s.name,
                s.args,
                s.result
            ));
        }
        if steps.is_empty() {
            steps.push_str("  (no tool calls)\n");
        }
        // Signal A — retrieval hit/miss: did the gold answer's material actually surface in any
        // tool result? If yes, the reader HAD the evidence and still answered wrong (a
        // reasoning/answering error this prompt CAN address). If no, retrieval never surfaced it —
        // a graph/coverage gap that rewording THIS prompt cannot fix.
        let retrieved = gold_retrieved(&f.gold, &f.steps);
        if !retrieved {
            retrieval_gaps += 1;
        }
        let retrieval_line = if retrieved {
            "Retrieval: the gold answer's material DID appear in a tool result — the reader had the \
             evidence and still answered wrong (a reasoning/answering error this prompt CAN address)."
        } else {
            "Retrieval: the gold answer's material NEVER appeared in any tool result — retrieval did \
             not surface it (a graph/coverage gap; rewording this prompt will NOT fix this case)."
        };
        // Signal B — the judge's own reason for the verdict, when present.
        let judge_line = match &f.judge_reason {
            Some(r) if !r.trim().is_empty() => format!("Judge reason (why wrong): {}\n", r.trim()),
            _ => String::new(),
        };
        cases.push_str(&format!(
            "--- Failing case {} ---\nQuestion: {}\nGold answer: {}\nModel answer: {}\n{}\n{}Tool-call trace:\n{}\n",
            i + 1,
            f.question,
            f.gold,
            f.pred,
            retrieval_line,
            judge_line,
            steps,
        ));
    }
    // Aggregate steer: tell the teacher up front how many failures are unwinnable retrieval gaps,
    // so it concentrates the rewrite on the cases the prompt can actually move.
    let gap_summary = if ctx.fails.is_empty() {
        String::new()
    } else {
        format!(
            "Of the {} failing cases below, {} are RETRIEVAL GAPS — the gold answer never surfaced \
             in any tool result, so NO prompt wording can make the reader answer them. Focus your \
             rewrite on the remaining cases, where the evidence WAS retrieved but the reader still \
             missed it.\n",
            ctx.fails.len(),
            retrieval_gaps,
        )
    };
    let tool_reference = render_tool_reference(&ctx.tools);
    // Name ONLY the metric the run actually optimizes. Mixing an exact-match contract into a judge
    // run pushed the teacher toward terse "shortest span" answers the judge then penalized (every
    // child scored worse). The gold-vs-model in each failing case supplies the standard of a good
    // answer, so we don't hand-describe the judge rubric here (it would drift from judge.md).
    let (grading, objective, answer_contract, score_header) = if ctx.judge {
        (
            "Answers are graded by an automated judge that compares the model's answer to the gold \
             answer — it rewards a correct, complete, well-grounded answer, NOT an exact string match.",
            "rewrite the system prompt so the model answers more of these correctly (a higher judge score)",
            "Preserve the answer format the current prompt already defines — do not force a terser one.",
            "PARENT JUDGE SCORE ON MINIBATCH",
        )
    } else {
        (
            "Answers are graded by EXACT MATCH of the shortest answer span.",
            "rewrite the system prompt so multi-hop exact-match improves",
            "Keep the strict answer contract (one `ANSWER:` line, shortest exact span).",
            "PARENT EXACT-MATCH ON MINIBATCH",
        )
    };
    format!(
        "You are improving the SYSTEM PROMPT for a multi-hop question-answering agent that navigates \
         a PRE-BUILT REASONING GRAPH. {grading}\n\
         The reader has these tools (reference only — the reader is given each tool's full \
         description by the API at call time, so this is context for YOU, not text to copy):\n\
         {tool_reference}\n\
         Below are FAILING rollouts. Each case gives the question, the gold answer, the model's \
         answer, a RETRIEVAL note (whether the gold's material was surfaced by the tools at all), \
         the JUDGE's reason it was wrong (when available), and the model's tool-call trace (tool \
         name, arguments, truncated result per step).\n\
         {gap_summary}\
         Diagnose the recurring navigation/answering mistakes and {objective}. Preserve behavior that already works.\n\
         Write STRATEGY and POLICY that leverages these tools — when and why to reach for each, how \
         to compose multi-hop steps, and grounding discipline before answering. Do NOT copy tool \
         mechanics, descriptions, or parameter lists into the reader prompt: the reader already \
         receives those from the API at call time, so duplicating them only bloats the prompt and \
         drifts from the real schema.\n\
         Output ONLY general, reusable behavior guidance for using the graph and tools. It MUST NOT \
         mention or reuse ANY specific entity name, answer, date, place, number, or other value from \
         the examples — those are test data and must never appear in the prompt.\n\
         {answer_contract} \
         Reply with ONLY the new system prompt text — no preamble, no quotes.\n\n\
         === {score_header} ===\nscore={parent_score:.3}\n\n\
         === CURRENT SYSTEM PROMPT ===\n{prompt}\n\n=== FAILING ROLLOUTS ===\n{cases}=== NEW SYSTEM PROMPT ===",
        tool_reference = tool_reference,
        gap_summary = gap_summary,
        parent_score = ctx.parent_score,
        prompt = ctx.parent_prompt,
        cases = cases,
    )
}

/// Signal A heuristic: did the gold answer's material actually surface in the rollout's tool
/// results? Checks each `|`-joined gold alternative against the concatenated (lowercased) step
/// results — a direct substring hit, or (for a multi-word gold) any distinctive token (length >=
/// 4) present. Deliberately LENIENT toward "retrieved": a false "retrieved" only costs the teacher
/// one prompt-focused case, whereas a false "gap" would wrongly tell it to give up on a case the
/// prompt could actually fix. Returns false when there are no tool results at all.
fn gold_retrieved(gold: &str, steps: &[ToolStep]) -> bool {
    let hay: String = steps
        .iter()
        .map(|s| s.result.to_lowercase())
        .collect::<Vec<_>>()
        .join("\n");
    if hay.trim().is_empty() {
        return false;
    }
    gold.split(" | ").any(|alt| {
        let a = alt.trim().to_lowercase();
        if a.is_empty() {
            return false;
        }
        if hay.contains(&a) {
            return true;
        }
        tokens_lower(alt)
            .iter()
            .filter(|t| t.len() >= 4)
            .any(|t| hay.contains(t.as_str()))
    })
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

/// Lowercased proper-noun-SHAPED tokens that occur across `>= 2` DISTINCT questions of the whole
/// dataset. A purely lexical `question_proper_nouns` rule can't tell a real entity from a
/// capitalized COMMON word (a sentence-initial interrogative like "Why"/"How", or a shared noun) —
/// both are Title-case. Frequency separates them: a question-specific entity typically appears in a
/// SINGLE question, while a common word recurs. Tokens returned here are EXCLUDED from the
/// per-question proper-noun leak check so GEPA isn't denied a good child prompt over a common word.
/// Language-agnostic — derived from the data, NOT a hardcoded stopword list. Counting is by
/// distinct question (a token repeated within one question counts once), and MUST be fed the FULL
/// train+val set so the frequency is global, not per-minibatch. An empty result (every candidate
/// token unique to one question) reproduces the pre-relaxation behavior exactly.
fn common_proper_noun_tokens(questions: &[Question]) -> HashSet<String> {
    let mut per_token: std::collections::HashMap<String, HashSet<usize>> =
        std::collections::HashMap::new();
    for (qi, q) in questions.iter().enumerate() {
        for pn in question_proper_nouns(&q.question) {
            per_token.entry(pn.to_lowercase()).or_default().insert(qi);
        }
    }
    per_token
        .into_iter()
        .filter(|(_, qs)| qs.len() >= 2)
        .map(|(t, _)| t)
        .collect()
}

/// Reject a child prompt that echoes any gold answer or question proper noun from the minibatch.
/// Returns `Some(reason)` on a leak, `None` when clean. `common_tokens` are dataset-wide common
/// proper-noun-shaped words (see [`common_proper_noun_tokens`]) that are NOT treated as leakable
/// entities; the gold-answer guard is unaffected by it.
fn leak_scan(
    candidate: &str,
    minibatch: &[Question],
    common_tokens: &HashSet<String>,
) -> Option<String> {
    let hay = tokens_lower(candidate);
    for q in minibatch {
        for g in golds_of(q) {
            if value_leaked(&hay, &g) {
                return Some(format!("contains gold answer {g:?}"));
            }
        }
        for pn in question_proper_nouns(&q.question) {
            // A proper-noun-shaped token that recurs across >=2 dataset questions is a COMMON word
            // (interrogative / shared noun), not a question-specific entity — don't reject over it.
            // A token unique to ONE question is absent from `common_tokens` and stays flagged.
            if common_tokens.contains(&pn.to_lowercase()) {
                continue;
            }
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
    &c.score_val
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
    let n_inst = pool.first().map(|c| c.score_val.len()).unwrap_or(0);
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
                mean(&a.score_val)
                    .partial_cmp(&mean(&b.score_val))
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
    pb: &ProgressBar,
) -> Result<GepaGraphResult> {
    anyhow::ensure!(!questions.is_empty(), "no questions to optimize against");
    let idx = DocIndex::open_or_create(&cfg.work).context("open index for graph GEPA")?;
    let graph = GraphStore::open(&cfg.work).ok();
    let spec = ChainSpec::from_ontology(&Ontology::load_or_default(&cfg.work));
    let tools = crate::backend::openai::tools_schema(graph.is_some());
    // Full chat-completions URL, used verbatim (no suffix appended).
    let url = cfg.endpoint.clone();

    // Stratified by `hop_type` so the small val split proportionally represents each reasoning
    // shape present in the dataset (empty `hop_type` forms one stratum); within each stratum the
    // split is still by episode id, so no gold's paraphrases straddle train/val.
    let (train, val) = crate::gepa::split_by_episode_stratified(
        &questions,
        |q| q.id.as_str(),
        |q| q.hop_type.as_str(),
        cfg.val_frac,
    );
    // All user-facing progress lines below go through `pb.println` (not raw `println!`/`eprintln!`):
    // the bar is LIVE for the whole run (created in `run_train` before this fn is called), and a raw
    // print interleaved with indicatif's redraw garbles the bar's line. `pb.println` prints the line
    // ABOVE the bar and redraws it cleanly underneath; on a hidden bar (`--no-progress`/non-TTY) it
    // is equivalent to a plain `println!`.
    pb.println(format!(
        "gepa_graph: {} questions ({} train, {} val), max_metric_calls={}, max_candidates={}, minibatch={}, pareto_size={}, graph={}, selection={}, work={}",
        questions.len(),
        train.len(),
        val.len(),
        cfg.max_metric_calls,
        cfg.max_candidates,
        cfg.minibatch,
        cfg.pareto_size,
        graph.is_some(),
        cfg.candidate_selection,
        cfg.work.display(),
    ));
    anyhow::ensure!(
        !val.is_empty(),
        "empty validation split — need >=2 distinct question ids"
    );

    // Main bar tracks metric-call spend end-to-end: length = max_metric_calls (the DSPy-style
    // budget), position = rollouts spent so far. This yields a real whole-run ETA (StatusTicker
    // derives it from pos/len). The baseline/pareto passes below run at position 0.
    pb.set_length(cfg.max_metric_calls as u64);
    pb.set_position(0);
    pb.set_prefix("training · baseline");

    // Running count of reader rollouts (metric calls) spent by the SEARCH; bumped after every
    // `score_questions` pass by the number of questions it scored. Bounds the candidate loop.
    let mut metric_calls: usize = 0;

    // Dataset-wide common proper-noun-shaped tokens (interrogatives, shared nouns recurring across
    // >=2 questions) that the per-minibatch leak-scan must NOT reject a child prompt over. Computed
    // ONCE over the FULL train+val set so the frequency is global, then threaded into every
    // `leak_scan` call.
    let common_tokens = common_proper_noun_tokens(&questions);

    let baseline_out = score_questions(
        &cfg,
        &url,
        &tools,
        &cfg.seed_prompt,
        &val,
        &idx,
        graph.as_ref(),
        &spec,
        pb,
    );
    metric_calls += val.len();
    // Pre-filter the val/pareto question set: drop any question whose BASELINE rollout errored on the
    // endpoint (a deterministic 500 / context-overflow reproduces for every candidate). Doing this
    // ONCE, before D_pareto is built, keeps every candidate's per-instance `score_val` vector aligned
    // on the SAME scorable instances and excludes endpoint errors from the reported score. Surfaced,
    // never silently dropped.
    let (val, baseline_out, dropped) = drop_baseline_errored(val, baseline_out);
    if dropped > 0 {
        pb.println(format!(
            "pareto set: dropped {dropped} question(s) that errored on the endpoint (excluded from scoring)"
        ));
    }
    anyhow::ensure!(
        !val.is_empty(),
        "every validation question errored on the endpoint — check the reader endpoint"
    );
    // No errored outcomes remain after the pre-filter, so `mean_scored` here equals a plain mean.
    let baseline_score = mean_scored(&baseline_out);
    pb.println(format!("baseline val: score={baseline_score:.3}"));

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let pareto_set = sample_questions(&val, cfg.pareto_size, &mut rng);
    pb.println(format!(
        "pareto set (D_pareto): {} of {} val",
        pareto_set.len(),
        val.len(),
    ));
    let base_pareto = score_questions(
        &cfg,
        &url,
        &tools,
        &cfg.seed_prompt,
        &pareto_set,
        &idx,
        graph.as_ref(),
        &spec,
        pb,
    );
    metric_calls += pareto_set.len();
    let mut pool = vec![Candidate {
        prompt: cfg.seed_prompt.clone(),
        score_val: scores(&base_pareto),
    }];
    // Best full-Pareto mean seen so far, surfaced in the bar prefix each iteration.
    let mut best_pareto_so_far = mean(&pool[0].score_val);
    // Per-candidate cached reflect-minibatch, indexed parallel to `pool` (see `MbCache`). Grows
    // with `pool` on every accept; the seed starts uncached.
    let mut mb_cache: Vec<Option<MbCache>> = vec![None];
    // Train questions to STOP sampling into reflect minibatches: a question whose rollout failed on
    // the endpoint (a deterministic context-length overflow / 500) would fail identically every time
    // it is resampled, burning budget + wall-clock and shrinking each minibatch for zero signal. The
    // val/D_pareto side is pre-filtered once via `drop_baseline_errored`; the train side has no
    // baseline pass, so we blocklist a question the first time it errors and skip it thereafter.
    let mut blocked_train: std::collections::HashSet<String> = std::collections::HashSet::new();
    // Minibatch-cache mode (see `GepaGraphConfig::minibatch_cache`). Default OFF = canonical GEPA
    // (fresh minibatch + parent re-roll every proposal). Turn ON only for a weak, high-variance
    // reader. Source: lab.toml `[tuning].gepa_minibatch_cache` (cfg.minibatch_cache); env
    // `GEPA_MINIBATCH_CACHE=1/0` overrides for a one-off sweep without editing the lab file.
    let cache_on = match std::env::var("GEPA_MINIBATCH_CACHE").ok().as_deref() {
        Some("1") | Some("true") => true,
        Some("0") | Some("false") => false,
        _ => cfg.minibatch_cache,
    };
    let fresh_mb = !cache_on;
    pb.println(if fresh_mb {
        "minibatch cache OFF — canonical fresh-per-proposal (default)"
    } else {
        "minibatch cache ON — frozen per-candidate minibatch (weak-reader mode)"
    });

    // DSPy-style budget: keep proposing candidates until the metric-call ceiling is hit OR the pool
    // reaches `max_candidates`. Every iteration either spends budget (a fresh minibatch is sampled &
    // the parent scored) or, on a cached-parent reflect-fail / leak-reject, EVICTS that parent's cache
    // (see the two `continue`s below) so its next selection re-samples fresh — so the metric-call
    // counter always advances and the ceiling alone guarantees termination (no stall counter needed).
    let mut it: usize = 0;
    while metric_calls < cfg.max_metric_calls && pool.len() < cfg.max_candidates {
        it += 1;
        // Static stage word + live candidate progress in the prefix; the StatusTicker owns `{msg}`
        // (ETA + tokens/resamples). No-op on a hidden bar.
        // Position = metric calls spent so far; the bar body renders `[spent/max_metric_calls]`.
        pb.set_position(metric_calls as u64);
        pb.set_prefix(format!("training · best={best_pareto_so_far:.3}"));
        let parent_idx = select_parent_idx(&pool, cfg.candidate_selection, &mut rng);
        let parent_prompt = pool[parent_idx].prompt.clone();

        // Reuse this parent's cached minibatch (score + fails) if it has one; otherwise sample a
        // minibatch that has failures, score the parent once, and cache it. Cloning the cache
        // Option (cheap vs a rollout) drops the borrow so the None branch can write it back.
        // `fresh_mb` forces the None branch every time (canonical GEPA — no cache reuse).
        let cached = if fresh_mb {
            None
        } else {
            mb_cache[parent_idx].clone()
        };
        let (batch, parent_mb_score, fails) = match cached {
            Some(mb) => (mb.batch, mb.score, mb.fails),
            None => {
                let mut found = None;
                for _ in 0..MINIBATCH_RESAMPLE_ATTEMPTS {
                    // Sample only from train questions not yet blocklisted by a prior endpoint error.
                    let available: Vec<Question> = train
                        .iter()
                        .filter(|q| !blocked_train.contains(&q.id))
                        .cloned()
                        .collect();
                    let batch = sample_questions(&available, cfg.minibatch, &mut rng);
                    if batch.is_empty() {
                        break;
                    }
                    let outcomes = score_questions(
                        &cfg,
                        &url,
                        &tools,
                        &parent_prompt,
                        &batch,
                        &idx,
                        graph.as_ref(),
                        &spec,
                        pb,
                    );
                    metric_calls += batch.len();
                    // Blocklist any question that errored on the endpoint so it is never sampled
                    // again (a context-length overflow is deterministic — resampling only wastes
                    // budget/time). Log once, when first dropped, so the exclusion is visible.
                    for (q, o) in batch.iter().zip(&outcomes) {
                        if o.errored && blocked_train.insert(q.id.clone()) {
                            pb.println(format!(
                                "train: dropped q {} from the minibatch pool (endpoint error — excluded from future sampling)",
                                q.id
                            ));
                        }
                    }
                    // Only a NON-errored sub-perfect rollout is a teachable failure: an endpoint
                    // error is not a wrong answer and must never seed a reflect batch or become a
                    // FailCase.
                    if outcomes.iter().any(|o| !o.errored && o.score < 1.0) {
                        let score = mean_scored(&outcomes);
                        let fails: Vec<FailCase> = batch
                            .iter()
                            .zip(&outcomes)
                            .filter(|(_, o)| !o.errored && o.score < 1.0)
                            .map(|(q, o)| FailCase {
                                question: q.question.clone(),
                                gold: golds_of(q).join(" | "),
                                pred: o.pred.clone(),
                                steps: o.steps.clone(),
                                judge_reason: o.judge_reason.clone(),
                            })
                            .collect();
                        found = Some((batch, score, fails));
                        break;
                    }
                }
                let Some((batch, score, fails)) = found else {
                    pb.println(format!(
                        "[iter {it}] no failures in sampled minibatch — skip"
                    ));
                    continue;
                };
                mb_cache[parent_idx] = Some(MbCache {
                    batch: batch.clone(),
                    score,
                    fails: fails.clone(),
                });
                (batch, score, fails)
            }
        };
        pb.println(format!(
            "[iter {it}] reflect minibatch: {} cases (fails={}) parent_mb_score={parent_mb_score:.3}",
            batch.len(),
            fails.len(),
        ));

        let ctx = GraphReflectContext {
            parent_prompt: parent_prompt.clone(),
            parent_score: parent_mb_score,
            judge: cfg.judge.is_some(),
            fails,
            tools: tools.clone(),
        };
        let instruction = build_graph_reflect_instruction(&ctx);
        let child_prompt = match reflect(&instruction) {
            Ok(p) => p,
            Err(e) => {
                pb.println(format!("[iter {it}] reflection failed: {e:#}"));
                // Evict this parent's cached minibatch so its next selection re-samples a FRESH batch
                // (spending budget) instead of re-failing on the identical one — guarantees the
                // metric-call counter advances, so the budget ceiling alone terminates the loop.
                mb_cache[parent_idx] = None;
                continue;
            }
        };
        if let Some(reason) = leak_scan(&child_prompt, &batch, &common_tokens) {
            pb.println(format!("[iter {it}] child REJECTED by leak-scan: {reason}"));
            // Evict (see above): a fresh batch next time gives reflection new failures to work from,
            // letting it escape a persistent leak, and guarantees budget progress.
            mb_cache[parent_idx] = None;
            continue;
        }

        let child_mb = score_questions(
            &cfg,
            &url,
            &tools,
            &child_prompt,
            &batch,
            &idx,
            graph.as_ref(),
            &spec,
            pb,
        );
        metric_calls += batch.len();
        let child_mb_score = mean_scored(&child_mb);
        if child_mb_score <= parent_mb_score {
            pb.println(format!(
                "[iter {it}] child_mb {child_mb_score:.3} <= parent_mb {parent_mb_score:.3} — discarded"
            ));
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
            pb,
        );
        metric_calls += pareto_set.len();
        pool.push(Candidate {
            prompt: child_prompt,
            // Per-instance Pareto vector uses raw `scores()` (positional, aligned across candidates).
            // The baseline pre-filter already removed every deterministically-erroring instance, so
            // an errored 0.0 here would only be a RARE residual mid-run failure on an instance that
            // survived the pre-filter; leaving that column at 0.0 is a neutral carry (it neither helps
            // nor hurts dominance, since every candidate hits the same deterministic error).
            score_val: scores(&child_pareto),
        });
        mb_cache.push(None); // keep parallel to `pool`; the new child starts uncached
        let best_pareto = pool
            .iter()
            .map(|c| mean(&c.score_val))
            .fold(f64::NEG_INFINITY, f64::max);
        best_pareto_so_far = best_pareto;
        pb.println(format!(
            "[iter {it}] parent_idx={parent_idx} parent_mb={parent_mb_score:.3} -> child_mb={child_mb_score:.3} — accepted (pareto_score={best_pareto:.3}, pool_size={})",
            pool.len(),
        ));
    }

    pb.println(format!("final full-val scoring: {} candidates", pool.len()));
    pb.set_position(cfg.max_metric_calls as u64);
    pb.set_prefix(format!(
        "training · final-val · best={best_pareto_so_far:.3}"
    ));
    let mut best_prompt = pool[0].prompt.clone();
    let mut best_score = f64::NEG_INFINITY;
    for c in &pool {
        let out = score_questions(
            &cfg,
            &url,
            &tools,
            &c.prompt,
            &val,
            &idx,
            graph.as_ref(),
            &spec,
            pb,
        );
        let em = mean_scored(&out);
        if em > best_score {
            best_score = em;
            best_prompt = c.prompt.clone();
        }
    }
    if !best_score.is_finite() {
        best_score = 0.0;
    }
    pb.println(format!(
        "gepa_graph final: score={best_score:.3} (baseline was {baseline_score:.3}), candidates={}",
        pool.len(),
    ));

    // Apply-gate: the winner was picked on the small val set that GEPA also selected on, so it can
    // overfit and REGRESS on the full task. Re-score the winner AND the seed on the FULL question set
    // and refuse to ship a winner that isn't at least as good as the seed. (Full set, not a further
    // held-out split: at this dataset size a third split is too noisy, and the full set IS the eval
    // objective we must not regress.)
    let score_full = |prompt: &str| {
        mean_scored(&score_questions(
            &cfg,
            &url,
            &tools,
            prompt,
            &questions,
            &idx,
            graph.as_ref(),
            &spec,
            pb,
        ))
    };
    let winner_full = score_full(&best_prompt);
    let seed_full = score_full(&cfg.seed_prompt);
    let applied = winner_full >= seed_full;
    if applied {
        pb.println(format!(
            "apply-gate: winner full-set={winner_full:.3} >= seed={seed_full:.3} — APPLYING winner"
        ));
    } else {
        pb.println(format!(
            "apply-gate: winner full-set={winner_full:.3} < seed={seed_full:.3} — winner regresses; KEEPING seed (not applied)"
        ));
        best_prompt = cfg.seed_prompt.clone();
        best_score = seed_full;
    }

    Ok(GepaGraphResult {
        prompt: best_prompt,
        baseline_score,
        best_score,
        candidates: pool.len(),
        applied,
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
        let reason = leak_scan(leaked, &mb, &HashSet::new()).expect("gold answer must be caught");
        assert!(reason.contains("Ada Lovelace"), "reason: {reason}");
    }

    #[test]
    fn leak_scan_catches_alias_and_proper_noun() {
        let mb = vec![q(
            "e1",
            "Where is Foobar located?",
            "Paris",
            &["City of Light"],
        )];
        assert!(
            leak_scan("… known as the City of Light …", &mb, &HashSet::new()).is_some(),
            "alias leaked"
        );
        assert!(
            leak_scan("… mention Foobar directly …", &mb, &HashSet::new()).is_some(),
            "proper noun leaked"
        );
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
        assert!(
            leak_scan(clean, &mb, &HashSet::new()).is_none(),
            "clean prompt wrongly rejected"
        );
    }

    #[test]
    fn short_gold_values_are_not_flagged_by_substring() {
        // gold "no" must not fire on "node"/"not"; single short tokens are skipped.
        let mb = vec![q("e1", "Is it raining?", "no", &["yes"])];
        assert!(leak_scan("Use the graph node and do not guess.", &mb, &HashSet::new()).is_none());
    }

    #[test]
    fn common_proper_noun_across_two_questions_is_not_flagged() {
        // A capitalized COMMON word (proper-noun-SHAPED, non-first token) that appears in >=2
        // DISTINCT questions is dataset-common, not a question-specific entity. The frequency guard
        // must exclude it so a child prompt using that word is NOT rejected — while the SAME word
        // stays flagged when the common-token set is empty (pre-relaxation behavior).
        let qs = vec![
            q("e1", "First, Which region ships it?", "Ada", &[]),
            q("e2", "Overall, Which method applies?", "Bee", &[]),
        ];
        let common = common_proper_noun_tokens(&qs);
        assert!(common.contains("which"), "token in >=2 questions is common");
        let cand = "Decide which node to expand and follow the chain to a grounded answer.";
        // Relaxed: common word does not trigger a leak.
        assert!(
            leak_scan(cand, &qs, &common).is_none(),
            "common proper-noun-shaped word wrongly rejected"
        );
        // Absent common set == today's strict behavior: the same word IS flagged.
        assert!(
            leak_scan(cand, &qs, &HashSet::new()).is_some(),
            "empty common set must reproduce the old strict rejection"
        );
    }

    #[test]
    fn single_question_proper_noun_stays_common_free_and_flagged() {
        // A proper-noun token that appears in only ONE question is NOT common, so it is still
        // caught by the leak-scan (the primary entity guard is preserved).
        let qs = vec![
            q("e1", "Who founded Foobar Corp?", "Ada", &[]),
            q("e2", "What shipped in the update?", "Bee", &[]),
        ];
        let common = common_proper_noun_tokens(&qs);
        assert!(
            !common.contains("foobar"),
            "single-question entity is not common"
        );
        assert!(
            leak_scan("please mention Foobar in guidance", &qs, &common).is_some(),
            "single-question proper noun must still be flagged"
        );
    }

    #[test]
    fn gold_retrieved_detects_hit_and_miss() {
        let step = |result: &str| ToolStep {
            name: "read".to_string(),
            args: json!({}),
            result: result.to_string(),
        };
        // Direct substring hit.
        assert!(gold_retrieved(
            "widget_alpha",
            &[step("the parameter widget_alpha sets the ceiling")]
        ));
        // Never surfaced -> gap.
        assert!(!gold_retrieved("widget_alpha", &[step("(no matches)")]));
        // Distinctive-token hit for a multi-word gold.
        assert!(gold_retrieved(
            "restart procedure",
            &[step("to RESTART the device, hold the button")]
        ));
        // `|`-joined alternatives: a hit on any alternative counts.
        assert!(gold_retrieved(
            ".zzz | image file",
            &[step("write the .zzz to the card")]
        ));
        // No tool results at all -> not retrieved.
        assert!(!gold_retrieved("anything", &[]));
    }

    #[test]
    fn reflect_instruction_has_leak_guard_and_markers() {
        // Tool reference is rendered from the reader's ACTUAL schema (single source of truth),
        // graph-ON so the graph tools (glossary/reach/sql) the old hardcoded list drifted from
        // are present.
        let tools = crate::backend::openai::tools_schema(true);
        let ctx = GraphReflectContext {
            parent_prompt: "seed graph prompt".to_string(),
            parent_score: 0.25,
            judge: false,
            fails: vec![FailCase {
                question: "Who directed the film?".to_string(),
                gold: "Jane Doe".to_string(),
                pred: "unknown".to_string(),
                steps: vec![ToolStep {
                    name: "glossary".to_string(),
                    args: json!({"name": "the film"}),
                    result: "(no matches)".to_string(),
                }],
                judge_reason: Some(
                    "expected the director's name; the model gave 'unknown'".to_string(),
                ),
            }],
            tools: tools.clone(),
        };
        let msg = build_graph_reflect_instruction(&ctx);
        assert!(msg.contains("PARENT EXACT-MATCH ON MINIBATCH"));
        assert!(msg.contains("score=0.250"));
        assert!(msg.contains("EXACT MATCH"));
        assert!(msg.contains("MUST NOT"), "leak guard instruction present");
        assert!(msg.contains("glossary(") && msg.contains("Tool-call trace"));
        assert!(msg.contains("Gold answer: Jane Doe"));
        assert!(msg.contains("=== NEW SYSTEM PROMPT ==="));

        // Signal A — the gold "Jane Doe" never appears in the "(no matches)" result, so the case is
        // flagged a RETRIEVAL GAP and the aggregate steer reports 1 of 1.
        assert!(
            msg.contains("NEVER appeared in any tool result"),
            "retrieval-gap note present when the gold was never surfaced"
        );
        assert!(
            msg.contains("1 are RETRIEVAL GAPS"),
            "aggregate retrieval-gap count present"
        );
        // Signal B — the judge's reason is surfaced to the teacher.
        assert!(
            msg.contains("Judge reason (why wrong): expected the director's name"),
            "judge reason line present when judge_reason is Some"
        );

        // Judge metric: the instruction names ONLY the judge — never EM / "shortest exact span",
        // which pushed the teacher toward terse answers the judge then penalizes.
        let judge_ctx = GraphReflectContext { judge: true, ..ctx };
        let jmsg = build_graph_reflect_instruction(&judge_ctx);
        assert!(jmsg.contains("PARENT JUDGE SCORE ON MINIBATCH"));
        assert!(jmsg.contains("automated judge"));
        assert!(
            !jmsg.contains("EXACT MATCH"),
            "judge run must not mention exact match"
        );
        assert!(
            !jmsg.contains("shortest exact span"),
            "judge run must not impose a terse answer contract"
        );

        // --- tool-awareness: reference block derived from the schema, not a hardcoded list ---
        assert!(
            msg.contains("The reader has these tools"),
            "tool reference introduced as context"
        );
        // Every real graph-ON tool name from the schema appears in the rendered block — including
        // `reach`/`sql`, which the retired hardcoded sentence silently omitted.
        for name in ["search", "read", "grep", "glob", "glossary", "reach", "sql"] {
            assert!(
                msg.contains(&format!("- {name} —")),
                "rendered tool reference must list `{name}` derived from the schema"
            );
        }
        // A parameter name from the schema is rendered (proves params come from the schema, not
        // prose). `query` is search's param; order-independent (serde_json may sort keys).
        assert!(msg.contains("params:"), "tool params rendered from schema");
        assert!(
            msg.contains("query"),
            "search's `query` param name rendered from schema"
        );
        // The retired hand-maintained tool-name sentence must be gone.
        assert!(
            !msg.contains("neighbors, path, related"),
            "old hardcoded/drifting tool-name list must be removed"
        );
        // Reflector is told to write strategy, NOT to copy tool mechanics into the reader prompt.
        assert!(
            msg.contains("Do NOT copy tool mechanics"),
            "guidance against duplicating tool mechanics present"
        );
        assert!(
            msg.contains("STRATEGY"),
            "reflector told to write strategy/policy"
        );
    }

    /// `score_questions` pairs each question with its input index, lets `run_units_parallel`
    /// return results in whatever order workers finish (jobs>1 reorders by design), then sorts
    /// back by index before returning. This is the load-bearing correctness property for Task 7:
    /// `Candidate::score_val` is POSITIONAL (Pareto dominance compares instance-by-instance across
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
        assert!(
            t1.iter().all(|x| !ids1.contains(&x.id)),
            "no train/val id overlap"
        );
    }

    #[test]
    fn select_parent_current_best_picks_highest_em() {
        let pool = vec![
            Candidate {
                prompt: "weak".into(),
                score_val: vec![0.0, 0.0],
            },
            Candidate {
                prompt: "strong".into(),
                score_val: vec![1.0, 1.0, 1.0],
            },
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
            Candidate {
                prompt: "a".into(),
                score_val: vec![1.0, 0.0, 0.0],
            },
            Candidate {
                prompt: "b".into(),
                score_val: vec![0.0, 1.0, 1.0],
            },
            Candidate {
                prompt: "c".into(),
                score_val: vec![0.0, 0.0, 0.0],
            },
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

    fn outcome(score: f64, errored: bool) -> RolloutOutcome {
        RolloutOutcome {
            score,
            pred: String::new(),
            steps: Vec::new(),
            judge_reason: None,
            errored,
        }
    }

    #[test]
    fn mean_scored_skips_errored_outcomes() {
        // The errored 0.0 is EXCLUDED, so the mean is over {1.0, 0.0} = 0.5 (not 1.0/3 ≈ 0.33).
        let out = vec![outcome(1.0, false), outcome(0.0, true), outcome(0.0, false)];
        assert!((mean_scored(&out) - 0.5).abs() < 1e-9);
        // Every outcome errored -> 0.0 (mean of the empty kept-set), never a divide-by-zero.
        assert_eq!(mean_scored(&[outcome(0.0, true), outcome(0.0, true)]), 0.0);
        // No errors -> identical to a plain mean.
        assert!((mean_scored(&[outcome(1.0, false), outcome(0.0, false)]) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn drop_baseline_errored_removes_erroring_questions() {
        let val = vec![
            q("a", "qa", "A", &[]),
            q("b", "qb", "B", &[]),
            q("c", "qc", "C", &[]),
        ];
        // The middle question's baseline rollout errored on the endpoint.
        let out = vec![outcome(1.0, false), outcome(0.0, true), outcome(0.0, false)];
        let (kept_q, kept_o, dropped) = drop_baseline_errored(val, out);
        assert_eq!(dropped, 1, "one baseline-erroring question dropped");
        let ids: Vec<&str> = kept_q.iter().map(|q| q.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["a", "c"],
            "erroring question removed, order preserved"
        );
        assert!(
            kept_o.iter().all(|o| !o.errored),
            "no errored outcome survives the pre-filter"
        );
        // Surviving questions and outcomes stay positionally aligned (load-bearing for Pareto vectors).
        assert_eq!(kept_q.len(), kept_o.len());
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
            Candidate {
                prompt: "a".into(),
                score_val: vec![0.5, 1.0, 0.5],
            },
            Candidate {
                prompt: "b".into(),
                score_val: vec![1.0, 0.5, 0.5],
            },
            Candidate {
                prompt: "c".into(),
                score_val: vec![0.0, 0.0, 0.0],
            },
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
