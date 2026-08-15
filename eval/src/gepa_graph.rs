//! GEPA-style reflective optimization of the GRAPH reader system prompt.
//!
//! Objective: maximize multi-hop Exact-Match (EM). Rollouts run through the SAME LM Studio
//! (OpenAI-compatible) agent loop the eval backend uses (`backend::openai`), driving glossa's
//! graph tools in-process against the corpus in `work`. Only the REFLECT step goes to the
//! TensorZero gateway (`gepa_reflect`), which proposes an improved GENERAL graph-reader prompt.
//!
//! Structure mirrors `gepa_constraint.rs` (single-objective GEPA): one `Candidate { prompt,
//! em_val }` scored per-question on a Pareto validation subset, minibatch reflection from the
//! train split, accept-if-child-beats-parent-on-minibatch, best-on-full-val at the end.
//!
//! Deviation from the blueprint's "reuse gepa.rs Pareto helpers": those helpers are typed on
//! gepa.rs's private quad `Candidate`; generalizing them would churn gepa.rs's own call sites and
//! its ~30 tests. Following the designated template (`gepa_constraint.rs`, which keeps its own
//! single-objective copies), the Pareto helpers here are local single-objective versions. The
//! freestanding items that reuse cleanly — `split_by_episode`, `output_likely_truncated`,
//! `CandidateSelection`, `hash_run_seed`, `default_run_tag`, `load_seed_prompt` — ARE reused from
//! `gepa.rs` (promoted to `pub(crate)`).

use crate::dataset::Question;
use crate::gepa::CandidateSelection;
use anyhow::{Context, Result};
use glossa::graph::ontology::Ontology;
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;
use glossa::tools::ChainSpec;
use glossa::trace::TraceLog;
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

pub struct GepaGraphConfig {
    pub endpoint: String,
    pub model: String,
    pub api_key: Option<String>,
    pub gateway: String,
    pub reflect_function: String,
    pub variant: String,
    pub episode_id: String,
    pub tags: Value,
    pub val_frac: f64,
    pub budget: usize,
    pub minibatch: usize,
    pub seed_prompt: String,
    pub work: PathBuf,
    pub seed: u64,
    pub pareto_size: usize,
    pub candidate_selection: CandidateSelection,
}

pub struct GepaGraphResult {
    pub prompt: String,
    pub baseline_em: f64,
    pub best_em: f64,
    pub candidates: usize,
    pub episode_id: String,
}

#[derive(Clone)]
struct ToolStep {
    name: String,
    args: Value,
    result: String,
}

struct RolloutOutcome {
    em: bool,
    pred: String,
    steps: Vec<ToolStep>,
}

#[derive(Clone)]
struct Candidate {
    prompt: String,
    /// Per-instance EM hit/miss on D_pareto (not full val).
    em_val: Vec<bool>,
}

struct FailCase {
    question: String,
    gold: String,
    pred: String,
    steps: Vec<ToolStep>,
}

struct GraphReflectContext {
    parent_prompt: String,
    parent_em: f64,
    fails: Vec<FailCase>,
}

fn golds_of(q: &Question) -> Vec<String> {
    let mut g = vec![q.answer.clone()];
    g.extend(q.answer_aliases.clone());
    g
}

fn mean_bool(b: &[bool]) -> f64 {
    if b.is_empty() {
        return 0.0;
    }
    b.iter().filter(|x| **x).count() as f64 / b.len() as f64
}

fn em_bools(out: &[RolloutOutcome]) -> Vec<bool> {
    out.iter().map(|o| o.em).collect()
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
    let exec = |name: &str, args: &Value| -> String {
        let body =
            crate::backend::glossa_tools::exec(name, args, &cfg.work, idx, graph, spec, &trace).0;
        steps.borrow_mut().push(ToolStep {
            name: name.to_string(),
            args: args.clone(),
            result: truncate_chars(&body, STEP_RESULT_CHARS),
        });
        body
    };
    let messages = vec![
        json!({ "role": "system", "content": prompt }),
        json!({ "role": "user", "content": crate::backend::prompt::user_prompt(q) }),
    ];
    let raw = match crate::backend::openai::run_agent_loop(chat, messages, exec, MAX_ROUNDS) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("graph rollout failed for q {}: {e:#}", q.id);
            String::new()
        }
    };
    let pred = crate::backend::prompt::parse_answer(&raw);
    let em = crate::score::exact_match_any(&pred, &golds_of(q));
    RolloutOutcome {
        em,
        pred,
        steps: steps.into_inner(),
    }
}

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
    questions
        .iter()
        .map(|q| rollout_one(cfg, url, tools, prompt, q, idx, graph, spec))
        .collect()
}

// --- reflection (TZ gateway, unchanged mechanism) -------------------------------------------

fn build_graph_reflect_instruction(ctx: &GraphReflectContext) -> String {
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

fn reflect(cfg: &GepaGraphConfig, ctx: &GraphReflectContext) -> Result<String> {
    let instruction = build_graph_reflect_instruction(ctx);
    println!(
        "graph reflect payload: {} chars (~{} tok est)",
        instruction.len(),
        instruction.len() / 4,
    );
    let turn = crate::tz::infer(
        &cfg.gateway,
        &cfg.reflect_function,
        &cfg.episode_id,
        &[json!({"role": "user", "content": instruction})],
        &cfg.tags,
        Duration::from_secs(180),
        Some("baseline"),
        None,
        None,
    )
    .context("gepa_reflect inference failed")?;
    let out = turn.text().trim().to_string();
    if out.is_empty() {
        anyhow::bail!("gepa_reflect returned an empty prompt");
    }
    if crate::gepa::output_likely_truncated(&out, turn.finish_reason.as_deref()) {
        anyhow::bail!(
            "gepa_reflect output truncated (finish_reason={:?})",
            turn.finish_reason,
        );
    }
    Ok(out)
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

fn candidate_bits(c: &Candidate) -> &[bool] {
    &c.em_val
}

fn dominates(a: &[bool], b: &[bool]) -> bool {
    let mut strictly = false;
    for (x, y) in a.iter().zip(b) {
        if !x && *y {
            return false;
        }
        if *x && !y {
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
    let bits: Vec<&[bool]> = pool.iter().map(candidate_bits).collect();
    let n_inst = bits.first().map(|b| b.len()).unwrap_or(0);
    if n_inst == 0 {
        return ((0..pool.len()).collect(), vec![0; pool.len()]);
    }

    let mut union = HashSet::new();
    let mut per_instance_winners: Vec<Vec<usize>> = Vec::with_capacity(n_inst);
    for i in 0..n_inst {
        let max_score = bits.iter().map(|b| b[i] as u8).max().unwrap_or(0);
        let winners: Vec<usize> = bits
            .iter()
            .enumerate()
            .filter(|(_, b)| b[i] as u8 == max_score)
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
                mean_bool(&a.em_val)
                    .partial_cmp(&mean_bool(&b.em_val))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0),
        CandidateSelection::Pareto => select_parent_pareto_weighted(pool, rng),
    }
}

// --- driver ---------------------------------------------------------------------------------

pub fn run(cfg: GepaGraphConfig, questions: Vec<Question>) -> Result<GepaGraphResult> {
    anyhow::ensure!(!questions.is_empty(), "no questions to optimize against");
    let idx = DocIndex::open_or_create(&cfg.work).context("open index for graph GEPA")?;
    let graph = GraphStore::open(&cfg.work).ok();
    let spec = ChainSpec::from_ontology(&Ontology::load_or_default(&cfg.work));
    let tools = crate::backend::openai::tools_schema(graph.is_some());
    let url = format!("{}/v1/chat/completions", cfg.endpoint.trim_end_matches('/'));

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
    let baseline_em = mean_bool(&em_bools(&baseline_out));
    println!("baseline val: em={baseline_em:.3}");

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
        em_val: em_bools(&base_pareto),
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
            if outcomes.iter().any(|o| !o.em) {
                minibatch = Some(batch);
                parent_outcomes = Some(outcomes);
                break;
            }
        }
        let (Some(batch), Some(outcomes)) = (minibatch, parent_outcomes) else {
            println!("[iter {it}] no failures in sampled minibatch — skip");
            continue;
        };
        let parent_em_mb = mean_bool(&em_bools(&outcomes));
        let fails: Vec<FailCase> = batch
            .iter()
            .zip(&outcomes)
            .filter(|(_, o)| !o.em)
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
        let child_prompt = match reflect(&cfg, &ctx) {
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
        let child_em_mb = mean_bool(&em_bools(&child_mb));
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
            em_val: em_bools(&child_pareto),
        });
        let best_pareto = pool
            .iter()
            .map(|c| mean_bool(&c.em_val))
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
        let em = mean_bool(&em_bools(&out));
        if em > best_em {
            best_em = em;
            best_prompt = c.prompt.clone();
        }
    }
    if !best_em.is_finite() {
        best_em = 0.0;
    }
    println!(
        "gepa_graph final: episode={} em={best_em:.3} (baseline was {baseline_em:.3}), candidates={}",
        cfg.episode_id,
        pool.len(),
    );

    Ok(GepaGraphResult {
        prompt: best_prompt,
        baseline_em,
        best_em,
        candidates: pool.len(),
        episode_id: cfg.episode_id,
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
            Candidate { prompt: "weak".into(), em_val: vec![false, false] },
            Candidate { prompt: "strong".into(), em_val: vec![true, true, true] },
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
            Candidate { prompt: "a".into(), em_val: vec![true, false, false] },
            Candidate { prompt: "b".into(), em_val: vec![false, true, true] },
            Candidate { prompt: "c".into(), em_val: vec![false, false, false] },
        ];
        let (frontier, counts) = pareto_frontier_win_counts(&pool);
        assert!(frontier.contains(&0) && frontier.contains(&1));
        assert_eq!(counts[1], 2);
        assert_eq!(counts[2], 0);
    }
}
