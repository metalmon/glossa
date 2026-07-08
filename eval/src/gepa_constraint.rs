//! GEPA-style reflective optimization for constraint `.csp` materialization.

pub use crate::constraint_synthetic::{CompileFixExample, MaterializeExample};
use crate::export_tz::GrepExample;
use anyhow::{Context, Result};
use glossa::grep::GrepOpts;
use glossa::index::store::DocIndex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Duration;

const MINIBATCH_RESAMPLE_ATTEMPTS: usize = 3;

pub struct GepaConstraintConfig {
    pub gateway: String,
    pub research_function: String,
    pub materialize_function: String,
    pub compile_fix_function: String,
    pub reflect_function: String,
    pub variant: String,
    pub episode_id: String,
    pub tags: Value,
    pub val_frac: f64,
    pub budget: usize,
    pub minibatch: usize,
    pub seed_research_prompt: String,
    pub seed_materialize_prompt: String,
    pub seed_compile_fix_prompt: String,
    pub hit_threshold: f64,
    pub seed: u64,
    pub pareto_size: usize,
    pub w_research: f64,
    pub w_materialize: f64,
    pub w_compile_fix: f64,
    pub work: PathBuf,
}

pub struct GepaConstraintRunResult {
    pub prompt: String,
    pub research_prompt: String,
    pub materialize_prompt: String,
    pub compile_fix_prompt: String,
    pub baseline_acc: f64,
    pub best_acc: f64,
    pub research_acc: f64,
    pub materialize_acc: f64,
    pub compile_fix_acc: f64,
    pub candidates: usize,
    pub episode_id: String,
}

#[derive(Clone)]
struct MaterializeOutcome {
    ok: bool,
    model_csp: String,
    recall: f64,
}

#[derive(Clone)]
struct ResearchOutcome {
    ok: bool,
    model_pattern: Option<String>,
    top_k: String,
}

#[derive(Clone)]
struct MaterializeTrace {
    ok: bool,
    ex: MaterializeExample,
    model_csp: String,
    recall: f64,
}

struct ReflectContext {
    parent_prompt: String,
    parent_acc: f64,
    traces: Vec<MaterializeTrace>,
}

#[derive(Clone)]
struct Candidate {
    prompt: String,
    val_bools: Vec<bool>,
}

pub fn load_materialize_jsonl(path: &Path) -> Result<Vec<MaterializeExample>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("open materialize jsonl {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("read line {} from {}", i + 1, path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let ex: MaterializeExample = serde_json::from_str(&line)
            .with_context(|| format!("parse materialize example line {}", i + 1))?;
        out.push(ex);
    }
    Ok(out)
}

pub fn load_compile_fix_jsonl(path: &Path) -> Result<Vec<CompileFixExample>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("open compile-fix jsonl {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("read line {} from {}", i + 1, path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let ex: CompileFixExample = serde_json::from_str(&line)
            .with_context(|| format!("parse compile-fix example line {}", i + 1))?;
        out.push(ex);
    }
    Ok(out)
}

pub fn load_research_jsonl(path: &Path) -> Result<Vec<GrepExample>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("open research jsonl {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("read line {} from {}", i + 1, path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let ex: GrepExample = serde_json::from_str(&line)
            .with_context(|| format!("parse research line {}", i + 1))?;
        out.push(ex);
    }
    Ok(out)
}

fn hit_threshold(cfg: &GepaConstraintConfig) -> f64 {
    if cfg.hit_threshold.is_finite() && cfg.hit_threshold > 0.0 {
        cfg.hit_threshold
    } else {
        0.5
    }
}

fn materialize_user_message(ex: &MaterializeExample) -> String {
    format!(
        "Документ: {}\nПараметр: {}\n\n=== WORKBOOK ===\n{}\n\n=== CSP ===\n",
        ex.doc, ex.parameter, ex.workbook_excerpt
    )
}

fn score_materialize_one(
    cfg: &GepaConstraintConfig,
    prompt: &str,
    ex: &MaterializeExample,
) -> Result<MaterializeOutcome> {
    let msg = materialize_user_message(ex);
    let turn = crate::tz::infer(
        &cfg.gateway,
        &cfg.materialize_function,
        &cfg.episode_id,
        &[json!({"role": "user", "content": msg})],
        &cfg.tags,
        Duration::from_secs(120),
        Some(&cfg.variant),
        Some(prompt),
        None,
    )?;
    let model_csp = turn.text().trim().to_string();
    let recall = crate::constraint_score::value_recall(&model_csp, &ex.gold_values);
    let ok = crate::constraint_score::param_hit(&model_csp, &ex.gold_values, hit_threshold(cfg));
    Ok(MaterializeOutcome {
        ok,
        model_csp,
        recall,
    })
}

fn score_materialize(
    cfg: &GepaConstraintConfig,
    prompt: &str,
    examples: &[MaterializeExample],
) -> Result<Vec<MaterializeOutcome>> {
    examples
        .iter()
        .map(|ex| score_materialize_one(cfg, prompt, ex))
        .collect()
}

fn compile_fix_user_message(ex: &CompileFixExample) -> String {
    format!(
        "Документ: {}\nПараметр: {}\n\n=== BROKEN CSP ===\n{}\n\n=== COMPILER ERROR ===\n{}\n\n=== FIXED CSP ===\n",
        ex.doc, ex.parameter, ex.broken_csp, ex.compiler_error
    )
}

fn score_compile_fix_one(
    cfg: &GepaConstraintConfig,
    prompt: &str,
    ex: &CompileFixExample,
) -> Result<MaterializeOutcome> {
    let msg = compile_fix_user_message(ex);
    let turn = crate::tz::infer(
        &cfg.gateway,
        &cfg.compile_fix_function,
        &cfg.episode_id,
        &[json!({"role": "user", "content": msg})],
        &cfg.tags,
        Duration::from_secs(120),
        Some(&cfg.variant),
        Some(prompt),
        None,
    )?;
    let model_csp = turn.text().trim().to_string();
    let recall = crate::constraint_score::value_recall(&model_csp, &ex.gold_values);
    let ok = crate::constraint_score::param_hit(&model_csp, &ex.gold_values, hit_threshold(cfg));
    Ok(MaterializeOutcome {
        ok,
        model_csp,
        recall,
    })
}

fn score_compile_fix(
    cfg: &GepaConstraintConfig,
    prompt: &str,
    examples: &[CompileFixExample],
) -> Result<Vec<MaterializeOutcome>> {
    examples
        .iter()
        .map(|ex| score_compile_fix_one(cfg, prompt, ex))
        .collect()
}

fn tool_call_args(block: &Value) -> Value {
    if let Some(args) = block.get("arguments").filter(|v| !v.is_null()) {
        if let Some(s) = args.as_str() {
            return serde_json::from_str(s).unwrap_or_else(|_| json!({}));
        }
        return args.clone();
    }
    if let Some(raw) = block.get("raw_arguments") {
        if let Some(s) = raw.as_str() {
            return serde_json::from_str(s).unwrap_or_else(|_| json!({}));
        }
        return raw.clone();
    }
    json!({})
}

fn first_grep_call(content: &[Value]) -> Option<String> {
    content
        .iter()
        .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_call"))
        .filter(|b| b.get("name").and_then(|n| n.as_str()) == Some("grep"))
        .find_map(|b| {
            let args = tool_call_args(b);
            args.get("pattern")
                .and_then(|p| p.as_str())
                .map(str::to_string)
        })
}

fn parse_gold_loc(s: &str) -> Option<(String, String)> {
    let (path, loc) = s.split_once('#')?;
    let path = path.trim_end_matches(':').trim();
    let loc = loc.trim();
    if path.is_empty() || loc.is_empty() {
        return None;
    }
    Some((path.to_string(), loc.to_string()))
}

fn gold_pairs(gold: &[String]) -> Vec<(String, String)> {
    gold.iter().filter_map(|g| parse_gold_loc(g)).collect()
}

fn normalize_path(p: &str) -> String {
    p.replace('\\', "/")
}

fn grep_top_k(idx: &DocIndex, pattern: &str, k: usize) -> Vec<glossa::grep::GrepHit> {
    glossa::grep::grep(idx, pattern, &GrepOpts::default())
        .unwrap_or_default()
        .into_iter()
        .take(k)
        .collect()
}

fn render_grep_hits(hits: &[glossa::grep::GrepHit]) -> String {
    hits.iter()
        .map(glossa::grep::GrepHit::display_line)
        .collect::<Vec<_>>()
        .join("\n")
}

fn grep_hit_hits(
    hits: &[glossa::grep::GrepHit],
    gold: &[(String, String)],
    idx: &DocIndex,
) -> bool {
    for h in hits {
        for (gp, gl) in gold {
            if normalize_path(&h.path) != normalize_path(gp) {
                continue;
            }
            if let Ok(Some(loc)) = idx.location_for_ord(&h.path, h.ord) {
                if loc == *gl {
                    return true;
                }
            }
        }
    }
    false
}

fn score_research_one(
    cfg: &GepaConstraintConfig,
    prompt: &str,
    ex: &GrepExample,
    idx: &DocIndex,
) -> ResearchOutcome {
    let messages = [json!({"role": "user", "content": format!("Question: {}", ex.question)})];
    let turn = match crate::tz::infer(
        &cfg.gateway,
        &cfg.research_function,
        &cfg.episode_id,
        &messages,
        &cfg.tags,
        Duration::from_secs(120),
        Some(&cfg.variant),
        Some(prompt),
        None,
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("constraint research inference failed: {e:#}");
            return ResearchOutcome {
                ok: false,
                model_pattern: None,
                top_k: String::new(),
            };
        }
    };
    let Some(pattern) = first_grep_call(&turn.content) else {
        return ResearchOutcome {
            ok: false,
            model_pattern: None,
            top_k: String::new(),
        };
    };
    let hits = grep_top_k(idx, &pattern, 10);
    let gold = gold_pairs(&ex.gold);
    ResearchOutcome {
        ok: grep_hit_hits(&hits, &gold, idx),
        model_pattern: Some(pattern),
        top_k: render_grep_hits(&hits),
    }
}

fn score_research(
    cfg: &GepaConstraintConfig,
    prompt: &str,
    examples: &[GrepExample],
    idx: &DocIndex,
) -> Vec<ResearchOutcome> {
    examples
        .iter()
        .map(|ex| score_research_one(cfg, prompt, ex, idx))
        .collect()
}

fn acc(scores: &[bool]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    scores.iter().filter(|b| **b).count() as f64 / scores.len() as f64
}

fn outcome_acc(scores: &[MaterializeOutcome]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    scores.iter().filter(|o| o.ok).count() as f64 / scores.len() as f64
}

fn research_acc(scores: &[ResearchOutcome]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    scores.iter().filter(|o| o.ok).count() as f64 / scores.len() as f64
}

fn research_to_bools(scores: &[ResearchOutcome]) -> Vec<bool> {
    scores.iter().map(|o| o.ok).collect()
}

fn outcomes_to_bools(scores: &[MaterializeOutcome]) -> Vec<bool> {
    scores.iter().map(|o| o.ok).collect()
}

fn combined_acc_from_pools(
    have_research: bool,
    have_materialize: bool,
    have_compile_fix: bool,
    research: f64,
    materialize: f64,
    compile_fix: f64,
    cfg: &GepaConstraintConfig,
) -> f64 {
    let wr = if have_research {
        cfg.w_research.max(0.0)
    } else {
        0.0
    };
    let wm = if have_materialize {
        cfg.w_materialize.max(0.0)
    } else {
        0.0
    };
    let wf = if have_compile_fix {
        cfg.w_compile_fix.max(0.0)
    } else {
        0.0
    };
    let w = wr + wm + wf;
    if w <= 0.0 {
        return 0.0;
    }
    (wr * research + wm * materialize + wf * compile_fix) / w
}

fn traces_from_outcomes(
    examples: &[MaterializeExample],
    outcomes: &[MaterializeOutcome],
) -> Vec<MaterializeTrace> {
    examples
        .iter()
        .cloned()
        .zip(outcomes.iter().cloned())
        .map(|(ex, o)| MaterializeTrace {
            ok: o.ok,
            ex,
            model_csp: o.model_csp,
            recall: o.recall,
        })
        .collect()
}

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

fn sample_examples(
    examples: &[MaterializeExample],
    n: usize,
    rng: &mut StdRng,
) -> Vec<MaterializeExample> {
    sample_indices(examples.len(), n, rng)
        .into_iter()
        .map(|i| examples[i].clone())
        .collect()
}

fn sample_pareto_set(
    examples: &[MaterializeExample],
    pareto_size: usize,
    rng: &mut StdRng,
) -> Vec<MaterializeExample> {
    sample_examples(examples, pareto_size, rng)
}

/// Split examples by episode_id so train/val don't leak the same source case.
fn split_by_episode<T: Clone>(
    items: &[T],
    episode_id: impl Fn(&T) -> &str,
    val_frac: f64,
) -> (Vec<T>, Vec<T>) {
    let mut episodes: Vec<String> = items.iter().map(|x| episode_id(x).to_string()).collect();
    episodes.sort();
    episodes.dedup();
    let n_val = if episodes.len() <= 1 {
        0
    } else {
        ((episodes.len() as f64 * val_frac).round() as usize).clamp(1, episodes.len() - 1)
    };
    let val_eps: HashSet<String> = episodes.into_iter().rev().take(n_val).collect();
    let mut train = Vec::new();
    let mut val = Vec::new();
    for item in items {
        if val_eps.contains(episode_id(item)) {
            val.push(item.clone());
        } else {
            train.push(item.clone());
        }
    }
    (train, val)
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    let mut out: String = s.chars().take(max_chars).collect();
    if s.chars().count() > max_chars {
        out.push_str("\n...");
    }
    out
}

fn format_trace_case(i: usize, trace: &MaterializeTrace) -> String {
    let status = if trace.ok { "OK" } else { "FAIL" };
    format!(
        "--- Materialize trace {} ({status}) ---\n\
         Parameter: {}\n\
         Gold values: {}\n\
         Model .csp snippet:\n{}\n\
         Recall: {:.3}\n\n",
        i + 1,
        trace.ex.parameter,
        trace.ex.gold_values.join(", "),
        truncate_chars(&trace.model_csp, 500),
        trace.recall,
    )
}

fn build_reflect_instruction(ctx: &ReflectContext) -> String {
    let mut cases = String::new();
    for (i, trace) in ctx.traces.iter().enumerate() {
        cases.push_str(&format_trace_case(i, trace));
    }
    format!(
        "You are improving the SYSTEM PROMPT for a constraint-table agent that materializes workbook excerpts into .csp tables.\n\
         Diagnose recurring mistakes in the materialize traces below and rewrite the system prompt so value recall improves.\n\
         Do NOT mention specific cases or gold values in the new prompt. Preserve instructions that already work.\n\
         Reply with ONLY the new system prompt text — no preamble, no quotes.\n\n\
         === PARENT SCORES ON MINIBATCH ===\n\
         materialize={parent_acc:.3}\n\n\
         === CURRENT SYSTEM PROMPT ===\n{prompt}\n\n=== MATERIALIZE TRACES ===\n{cases}=== NEW SYSTEM PROMPT ===",
        parent_acc = ctx.parent_acc,
        prompt = ctx.parent_prompt,
        cases = cases,
    )
}

fn output_likely_truncated(_text: &str, finish_reason: Option<&str>) -> bool {
    finish_reason
        .is_some_and(|r| r.eq_ignore_ascii_case("length") || r.eq_ignore_ascii_case("max_tokens"))
}

fn reflect(cfg: &GepaConstraintConfig, ctx: &ReflectContext) -> Result<String> {
    let instruction = build_reflect_instruction(ctx);
    println!(
        "constraint reflect payload: {} chars (~{} tok est)",
        instruction.len(),
        instruction.len() / 4,
    );
    let messages = [json!({"role": "user", "content": instruction})];
    let turn = crate::tz::infer(
        &cfg.gateway,
        &cfg.reflect_function,
        &cfg.episode_id,
        &messages,
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
    if output_likely_truncated(&out, turn.finish_reason.as_deref()) {
        anyhow::bail!(
            "gepa_reflect output truncated (finish_reason={:?})",
            turn.finish_reason,
        );
    }
    Ok(out)
}

fn candidate_bits(c: &Candidate) -> &[bool] {
    &c.val_bools
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
    let n_inst = pool.first().map(|c| c.val_bools.len()).unwrap_or(0);
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

fn feedback_tags(cfg: &GepaConstraintConfig, stage: &str) -> Value {
    let mut m = cfg.tags.as_object().cloned().unwrap_or_default();
    m.insert("stage".into(), stage.into());
    if let Some(n) = stage.strip_prefix("iter_") {
        m.insert("iter".into(), n.into());
    }
    Value::Object(m)
}

fn post_materialize_feedback(cfg: &GepaConstraintConfig, stage: &str, metric: &str, value: f64) {
    let tags = feedback_tags(cfg, stage);
    crate::tz::post_feedback(&cfg.gateway, &cfg.episode_id, metric, json!(value), &tags);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PromptSlice {
    Research,
    Materialize,
    CompileFix,
}

#[derive(Clone)]
struct TripleCandidate {
    research_prompt: String,
    materialize_prompt: String,
    compile_fix_prompt: String,
    r_val: Vec<bool>,
    m_val: Vec<bool>,
    f_val: Vec<bool>,
}

fn candidate_triple_bits(c: &TripleCandidate) -> Vec<bool> {
    c.r_val
        .iter()
        .chain(&c.m_val)
        .chain(&c.f_val)
        .copied()
        .collect()
}

fn triple_candidate_combined(c: &TripleCandidate, cfg: &GepaConstraintConfig) -> f64 {
    combined_acc_from_pools(
        !c.r_val.is_empty(),
        !c.m_val.is_empty(),
        !c.f_val.is_empty(),
        acc(&c.r_val),
        acc(&c.m_val),
        acc(&c.f_val),
        cfg,
    )
}

fn select_triple_parent(pool: &[TripleCandidate], rng: &mut StdRng) -> usize {
    if pool.len() <= 1 {
        return 0;
    }
    let candidates: Vec<Candidate> = pool
        .iter()
        .enumerate()
        .map(|(i, c)| Candidate {
            prompt: i.to_string(),
            val_bools: candidate_triple_bits(c),
        })
        .collect();
    select_parent_pareto_weighted(&candidates, rng)
}

fn sample_slice<T: Clone>(items: &[T], n: usize, rng: &mut StdRng) -> Vec<T> {
    sample_indices(items.len(), n, rng)
        .into_iter()
        .map(|i| items[i].clone())
        .collect()
}

fn weighted_slots(minibatch: usize, have: [bool; 3], weights: [f64; 3]) -> [usize; 3] {
    let active = have.iter().filter(|&&h| h).count();
    if minibatch == 0 || active == 0 {
        return [0; 3];
    }
    if minibatch < active {
        let mut out = [0; 3];
        let mut left = minibatch;
        for (i, h) in have.iter().enumerate() {
            if *h && left > 0 {
                out[i] = 1;
                left -= 1;
            }
        }
        return out;
    }
    let mut out = [0; 3];
    let mut total_w = 0.0;
    for i in 0..3 {
        if have[i] {
            out[i] = 1;
            total_w += weights[i].max(0.0);
        }
    }
    let remaining = minibatch - active;
    if remaining == 0 {
        return out;
    }
    if total_w <= 0.0 {
        return weighted_slots(minibatch, have, [1.0; 3]);
    }
    let mut frac = Vec::new();
    let mut assigned = 0usize;
    for i in 0..3 {
        if !have[i] {
            continue;
        }
        let share = remaining as f64 * weights[i].max(0.0) / total_w;
        let base = share.floor() as usize;
        out[i] += base;
        assigned += base;
        frac.push((i, share - base as f64));
    }
    let mut left = remaining.saturating_sub(assigned);
    frac.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (i, _) in frac {
        if left == 0 {
            break;
        }
        out[i] += 1;
        left -= 1;
    }
    out
}

#[derive(Clone)]
struct TripleBatch {
    research: Vec<GrepExample>,
    materialize: Vec<MaterializeExample>,
    compile_fix: Vec<CompileFixExample>,
}

fn sample_triple_batch(
    cfg: &GepaConstraintConfig,
    research: &[GrepExample],
    materialize: &[MaterializeExample],
    compile_fix: &[CompileFixExample],
    rng: &mut StdRng,
) -> TripleBatch {
    let slots = weighted_slots(
        cfg.minibatch,
        [
            !research.is_empty(),
            !materialize.is_empty(),
            !compile_fix.is_empty(),
        ],
        [cfg.w_research, cfg.w_materialize, cfg.w_compile_fix],
    );
    TripleBatch {
        research: sample_slice(research, slots[0], rng),
        materialize: sample_slice(materialize, slots[1], rng),
        compile_fix: sample_slice(compile_fix, slots[2], rng),
    }
}

fn target_from_failures(r_fail: usize, m_fail: usize, f_fail: usize) -> Option<PromptSlice> {
    if r_fail == 0 && m_fail == 0 && f_fail == 0 {
        return None;
    }
    if r_fail >= m_fail && r_fail >= f_fail {
        Some(PromptSlice::Research)
    } else if m_fail >= f_fail {
        Some(PromptSlice::Materialize)
    } else {
        Some(PromptSlice::CompileFix)
    }
}

fn reflect_triple(
    cfg: &GepaConstraintConfig,
    target: PromptSlice,
    current_prompt: &str,
    parent_scores: (f64, f64, f64, f64),
    traces: &str,
) -> Result<String> {
    let target_name = match target {
        PromptSlice::Research => "research",
        PromptSlice::Materialize => "materialize",
        PromptSlice::CompileFix => "compile_fix",
    };
    let instruction = format!(
        "You are improving one SYSTEM PROMPT slice for a constraint-table GEPA optimizer.\n\
         Target slice to mutate: {target_name}\n\
         Slices are: research (find source chunks with grep/read), materialize (write .csp from workbook), compile_fix (repair broken .csp from compiler errors).\n\
         Diagnose FAIL traces and rewrite ONLY the target slice prompt. Preserve behavior that works. Do not mention exact gold values or case IDs.\n\
         Reply with ONLY the new prompt text.\n\n\
         === PARENT SCORES ON MINIBATCH ===\n\
         research={:.3} materialize={:.3} compile_fix={:.3} combined={:.3}\n\n\
         === CURRENT TARGET PROMPT ===\n{}\n\n=== TRACES ===\n{}=== NEW TARGET PROMPT ===",
        parent_scores.0,
        parent_scores.1,
        parent_scores.2,
        parent_scores.3,
        current_prompt,
        traces,
    );
    println!(
        "constraint reflect target={target_name} payload: {} chars (~{} tok est)",
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
    if output_likely_truncated(&out, turn.finish_reason.as_deref()) {
        anyhow::bail!(
            "gepa_reflect output truncated (finish_reason={:?})",
            turn.finish_reason,
        );
    }
    Ok(out)
}

fn format_triple_traces(
    research: &[GrepExample],
    r_out: &[ResearchOutcome],
    materialize: &[MaterializeExample],
    m_out: &[MaterializeOutcome],
    compile_fix: &[CompileFixExample],
    f_out: &[MaterializeOutcome],
) -> String {
    let mut out = String::new();
    for (i, (ex, score)) in research.iter().zip(r_out).enumerate() {
        let status = if score.ok { "OK" } else { "FAIL" };
        out.push_str(&format!(
            "--- Research trace {} ({status}) ---\nQuestion: {}\nGold chunks: {}\nModel grep: {:?}\nTop hits:\n{}\n\n",
            i + 1,
            ex.question,
            ex.gold.join(", "),
            score.model_pattern,
            truncate_chars(&score.top_k, 500),
        ));
    }
    for (i, (ex, score)) in materialize.iter().zip(m_out).enumerate() {
        let status = if score.ok { "OK" } else { "FAIL" };
        out.push_str(&format_trace_case(
            i,
            &MaterializeTrace {
                ok: score.ok,
                ex: ex.clone(),
                model_csp: score.model_csp.clone(),
                recall: score.recall,
            },
        ));
        out.push_str(&format!("Trace type: materialize ({status})\n\n"));
    }
    for (i, (ex, score)) in compile_fix.iter().zip(f_out).enumerate() {
        let status = if score.ok { "OK" } else { "FAIL" };
        out.push_str(&format!(
            "--- Compile-fix trace {} ({status}) ---\nParameter: {}\nCompiler error: {}\nBroken .csp:\n{}\nGold values: {}\nModel .csp snippet:\n{}\nRecall: {:.3}\n\n",
            i + 1,
            ex.parameter,
            ex.compiler_error,
            truncate_chars(&ex.broken_csp, 300),
            ex.gold_values.join(", "),
            truncate_chars(&score.model_csp, 500),
            score.recall,
        ));
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn score_triple(
    cfg: &GepaConstraintConfig,
    candidate: &TripleCandidate,
    research: &[GrepExample],
    materialize: &[MaterializeExample],
    compile_fix: &[CompileFixExample],
    idx: &DocIndex,
) -> Result<(
    Vec<ResearchOutcome>,
    Vec<MaterializeOutcome>,
    Vec<MaterializeOutcome>,
)> {
    let r = score_research(cfg, &candidate.research_prompt, research, idx);
    let m = score_materialize(cfg, &candidate.materialize_prompt, materialize)?;
    let f = score_compile_fix(cfg, &candidate.compile_fix_prompt, compile_fix)?;
    Ok((r, m, f))
}

pub fn run(
    cfg: GepaConstraintConfig,
    research_grep: Vec<GrepExample>,
    materialize: Vec<MaterializeExample>,
    compile_fix: Vec<CompileFixExample>,
) -> Result<GepaConstraintRunResult> {
    anyhow::ensure!(
        !research_grep.is_empty() || !materialize.is_empty() || !compile_fix.is_empty(),
        "no research/materialize/compile-fix examples to optimize against"
    );
    let idx = DocIndex::open_or_create(&cfg.work).context("open index for constraint research")?;
    if !research_grep.is_empty() {
        crate::tz::ensure_function(&cfg.gateway, &cfg.research_function, Some(&cfg.variant))
            .context("GEPA constraint research function")?;
    }
    if !materialize.is_empty() {
        crate::tz::ensure_function(&cfg.gateway, &cfg.materialize_function, Some(&cfg.variant))
            .context("GEPA constraint materialize function")?;
    }
    if !compile_fix.is_empty() {
        crate::tz::ensure_function(&cfg.gateway, &cfg.compile_fix_function, Some(&cfg.variant))
            .context("GEPA constraint compile-fix function")?;
    }

    let (train_r, val_r) = split_by_episode(&research_grep, |ex| &ex.episode_id, cfg.val_frac);
    let (train_m, val_m) = split_by_episode(&materialize, |ex| &ex.episode_id, cfg.val_frac);
    let (train_f, val_f) = split_by_episode(&compile_fix, |ex| &ex.episode_id, cfg.val_frac);
    println!(
        "gepa_constraint: research {} ({} train, {} val), materialize {} ({} train, {} val), compile_fix {} ({} train, {} val), budget={}, minibatch={}, work={}",
        research_grep.len(),
        train_r.len(),
        val_r.len(),
        materialize.len(),
        train_m.len(),
        val_m.len(),
        compile_fix.len(),
        train_f.len(),
        val_f.len(),
        cfg.budget,
        cfg.minibatch,
        cfg.work.display(),
    );

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let base = TripleCandidate {
        research_prompt: cfg.seed_research_prompt.clone(),
        materialize_prompt: cfg.seed_materialize_prompt.clone(),
        compile_fix_prompt: cfg.seed_compile_fix_prompt.clone(),
        r_val: Vec::new(),
        m_val: Vec::new(),
        f_val: Vec::new(),
    };
    let (base_r, base_m, base_f) = score_triple(&cfg, &base, &val_r, &val_m, &val_f, &idx)?;
    let base_r_acc = research_acc(&base_r);
    let base_m_acc = outcome_acc(&base_m);
    let base_f_acc = outcome_acc(&base_f);
    let baseline_acc = combined_acc_from_pools(
        !base_r.is_empty(),
        !base_m.is_empty(),
        !base_f.is_empty(),
        base_r_acc,
        base_m_acc,
        base_f_acc,
        &cfg,
    );
    post_materialize_feedback(&cfg, "baseline", "gepa_c_baseline_research", base_r_acc);
    post_materialize_feedback(&cfg, "baseline", "gepa_c_baseline_materialize", base_m_acc);
    post_materialize_feedback(&cfg, "baseline", "gepa_c_baseline_compile_fix", base_f_acc);
    post_materialize_feedback(&cfg, "baseline", "gepa_c_combined_acc", baseline_acc);

    let pr = sample_slice(&val_r, cfg.pareto_size, &mut rng);
    let pm = sample_slice(&val_m, cfg.pareto_size, &mut rng);
    let pf = sample_slice(&val_f, cfg.pareto_size, &mut rng);
    let (base_pr, base_pm, base_pf) = score_triple(&cfg, &base, &pr, &pm, &pf, &idx)?;
    let mut pool = vec![TripleCandidate {
        r_val: research_to_bools(&base_pr),
        m_val: outcomes_to_bools(&base_pm),
        f_val: outcomes_to_bools(&base_pf),
        ..base
    }];

    for it in 0..cfg.budget {
        let parent_idx = select_triple_parent(&pool, &mut rng);
        let parent = pool[parent_idx].clone();
        let mut selected = None;
        for _ in 0..MINIBATCH_RESAMPLE_ATTEMPTS {
            let batch = sample_triple_batch(&cfg, &train_r, &train_m, &train_f, &mut rng);
            let (r_out, m_out, f_out) = score_triple(
                &cfg,
                &parent,
                &batch.research,
                &batch.materialize,
                &batch.compile_fix,
                &idx,
            )?;
            let r_fail = r_out.iter().filter(|o| !o.ok).count();
            let m_fail = m_out.iter().filter(|o| !o.ok).count();
            let f_fail = f_out.iter().filter(|o| !o.ok).count();
            if let Some(target) = target_from_failures(r_fail, m_fail, f_fail) {
                selected = Some((batch, r_out, m_out, f_out, target));
                break;
            }
        }
        let Some((batch, r_out, m_out, f_out, target)) = selected else {
            println!("[iter {it}] no failures in sampled triple minibatch — skip");
            continue;
        };
        let parent_r = research_acc(&r_out);
        let parent_m = outcome_acc(&m_out);
        let parent_f = outcome_acc(&f_out);
        let parent_c = combined_acc_from_pools(
            !r_out.is_empty(),
            !m_out.is_empty(),
            !f_out.is_empty(),
            parent_r,
            parent_m,
            parent_f,
            &cfg,
        );
        let traces = format_triple_traces(
            &batch.research,
            &r_out,
            &batch.materialize,
            &m_out,
            &batch.compile_fix,
            &f_out,
        );
        let current_prompt = match target {
            PromptSlice::Research => &parent.research_prompt,
            PromptSlice::Materialize => &parent.materialize_prompt,
            PromptSlice::CompileFix => &parent.compile_fix_prompt,
        };
        let child_slice = match reflect_triple(
            &cfg,
            target,
            current_prompt,
            (parent_r, parent_m, parent_f, parent_c),
            &traces,
        ) {
            Ok(prompt) => prompt,
            Err(e) => {
                println!("[iter {it}] reflection failed: {e:#}");
                continue;
            }
        };
        let mut child = parent.clone();
        match target {
            PromptSlice::Research => child.research_prompt = child_slice,
            PromptSlice::Materialize => child.materialize_prompt = child_slice,
            PromptSlice::CompileFix => child.compile_fix_prompt = child_slice,
        }
        let (child_r_out, child_m_out, child_f_out) = score_triple(
            &cfg,
            &child,
            &batch.research,
            &batch.materialize,
            &batch.compile_fix,
            &idx,
        )?;
        let child_c = combined_acc_from_pools(
            !child_r_out.is_empty(),
            !child_m_out.is_empty(),
            !child_f_out.is_empty(),
            research_acc(&child_r_out),
            outcome_acc(&child_m_out),
            outcome_acc(&child_f_out),
            &cfg,
        );
        if child_c <= parent_c {
            println!("[iter {it}] child_mb {child_c:.3} <= parent_mb {parent_c:.3} — discarded");
            continue;
        }
        let (child_pr, child_pm, child_pf) = score_triple(&cfg, &child, &pr, &pm, &pf, &idx)?;
        child.r_val = research_to_bools(&child_pr);
        child.m_val = outcomes_to_bools(&child_pm);
        child.f_val = outcomes_to_bools(&child_pf);
        pool.push(child);
        let best = pool
            .iter()
            .map(|c| triple_candidate_combined(c, &cfg))
            .fold(f64::NEG_INFINITY, f64::max);
        println!(
            "[iter {it}] accepted target={target:?} parent_mb={parent_c:.3} child_mb={child_c:.3} pareto_combined={best:.3} pool_size={}",
            pool.len()
        );
        post_materialize_feedback(&cfg, &format!("iter_{it}"), "gepa_c_combined_acc", best);
    }

    let mut best = pool[0].clone();
    let mut best_acc = f64::NEG_INFINITY;
    let mut best_r_acc = 0.0;
    let mut best_m_acc = 0.0;
    let mut best_f_acc = 0.0;
    for candidate in &pool {
        let (r, m, f) = score_triple(&cfg, candidate, &val_r, &val_m, &val_f, &idx)?;
        let r_acc = research_acc(&r);
        let m_acc = outcome_acc(&m);
        let f_acc = outcome_acc(&f);
        let combined = combined_acc_from_pools(
            !r.is_empty(),
            !m.is_empty(),
            !f.is_empty(),
            r_acc,
            m_acc,
            f_acc,
            &cfg,
        );
        if combined > best_acc {
            best_acc = combined;
            best = candidate.clone();
            best_r_acc = r_acc;
            best_m_acc = m_acc;
            best_f_acc = f_acc;
        }
    }
    if !best_acc.is_finite() {
        best_acc = 0.0;
    }
    post_materialize_feedback(&cfg, "final", "gepa_c_final_research", best_r_acc);
    post_materialize_feedback(&cfg, "final", "gepa_c_final_materialize", best_m_acc);
    post_materialize_feedback(&cfg, "final", "gepa_c_final_compile_fix", best_f_acc);
    post_materialize_feedback(&cfg, "final", "gepa_c_combined_acc", best_acc);
    println!(
        "TZ final: episode={} research={best_r_acc:.3} materialize={best_m_acc:.3} compile_fix={best_f_acc:.3} combined={best_acc:.3} (baseline was {baseline_acc:.3})",
        cfg.episode_id,
    );
    Ok(GepaConstraintRunResult {
        prompt: best.materialize_prompt.clone(),
        research_prompt: best.research_prompt,
        materialize_prompt: best.materialize_prompt,
        compile_fix_prompt: best.compile_fix_prompt,
        baseline_acc,
        best_acc,
        research_acc: best_r_acc,
        materialize_acc: best_m_acc,
        compile_fix_acc: best_f_acc,
        candidates: pool.len(),
        episode_id: cfg.episode_id,
    })
}

pub fn run_materialize(
    cfg: GepaConstraintConfig,
    examples: Vec<MaterializeExample>,
) -> Result<GepaConstraintRunResult> {
    anyhow::ensure!(
        !examples.is_empty(),
        "no materialize examples to optimize against"
    );
    crate::tz::ensure_function(&cfg.gateway, &cfg.materialize_function, Some(&cfg.variant))
        .context("GEPA constraint materialize function")?;

    let (train, val) = split_by_episode(&examples, |ex| &ex.episode_id, cfg.val_frac);
    println!(
        "gepa_constraint: materialize {} ({} train, {} val), budget={}, minibatch={}, pareto_size={}",
        examples.len(),
        train.len(),
        val.len(),
        cfg.budget,
        cfg.minibatch,
        cfg.pareto_size,
    );

    let baseline_out = score_materialize(&cfg, &cfg.seed_materialize_prompt, &val)
        .context("score baseline materialize on validation set")?;
    let baseline_acc = outcome_acc(&baseline_out);
    println!("baseline val: materialize={baseline_acc:.3}");
    post_materialize_feedback(
        &cfg,
        "baseline",
        "gepa_c_baseline_materialize",
        baseline_acc,
    );

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let pareto_set = sample_pareto_set(&val, cfg.pareto_size, &mut rng);
    println!(
        "pareto set (D_pareto): materialize={} (of {} val)",
        pareto_set.len(),
        val.len(),
    );

    let base_pareto = score_materialize(&cfg, &cfg.seed_materialize_prompt, &pareto_set)
        .context("score baseline materialize on pareto set")?;
    let mut pool = vec![Candidate {
        prompt: cfg.seed_materialize_prompt.clone(),
        val_bools: outcomes_to_bools(&base_pareto),
    }];

    for it in 0..cfg.budget {
        let parent_idx = select_parent_pareto_weighted(&pool, &mut rng);
        let parent_prompt = pool[parent_idx].prompt.clone();

        let mut minibatch = None;
        let mut parent_traces = None;
        let mut parent_acc = 0.0;
        for _ in 0..MINIBATCH_RESAMPLE_ATTEMPTS {
            let batch = sample_examples(&train, cfg.minibatch, &mut rng);
            if batch.is_empty() {
                break;
            }
            let outcomes = score_materialize(&cfg, &parent_prompt, &batch)
                .with_context(|| format!("score parent minibatch at iter {it}"))?;
            let traces = traces_from_outcomes(&batch, &outcomes);
            let failures = traces.iter().filter(|t| !t.ok).count();
            if failures > 0 {
                parent_acc = outcome_acc(&outcomes);
                parent_traces = Some(traces);
                minibatch = Some(batch);
                break;
            }
        }

        let Some(batch) = minibatch else {
            println!("[iter {it}] no failures in sampled minibatch — skip");
            continue;
        };
        let traces = parent_traces.expect("traces with minibatch");
        let failures = traces.iter().filter(|t| !t.ok).count();
        println!(
            "[iter {it}] reflect minibatch: {} traces (fails={failures}) parent_mb={parent_acc:.3}",
            traces.len(),
        );

        let ctx = ReflectContext {
            parent_prompt: parent_prompt.clone(),
            parent_acc,
            traces,
        };
        let child_prompt = match reflect(&cfg, &ctx) {
            Ok(prompt) => prompt,
            Err(e) => {
                println!("[iter {it}] reflection failed: {e:#}");
                continue;
            }
        };

        let child_mb = score_materialize(&cfg, &child_prompt, &batch)
            .with_context(|| format!("score child minibatch at iter {it}"))?;
        let child_acc = outcome_acc(&child_mb);
        if child_acc <= parent_acc {
            println!(
                "[iter {it}] child_mb {child_acc:.3} <= parent_mb {parent_acc:.3} — discarded"
            );
            continue;
        }

        let child_pareto = score_materialize(&cfg, &child_prompt, &pareto_set)
            .with_context(|| format!("score child pareto set at iter {it}"))?;
        pool.push(Candidate {
            prompt: child_prompt,
            val_bools: outcomes_to_bools(&child_pareto),
        });
        let best_pareto = pool
            .iter()
            .map(|c| acc(&c.val_bools))
            .fold(f64::NEG_INFINITY, f64::max);
        println!(
            "[iter {it}] parent_idx={parent_idx} parent_mb={parent_acc:.3} -> child_mb={child_acc:.3} — accepted (pareto_materialize={best_pareto:.3}, pool_size={})",
            pool.len(),
        );
        post_materialize_feedback(
            &cfg,
            &format!("iter_{it}"),
            "gepa_c_iter_materialize",
            best_pareto,
        );
        post_materialize_feedback(
            &cfg,
            &format!("iter_{it}"),
            "gepa_c_combined_acc",
            best_pareto,
        );
    }

    println!("final full-val scoring: {} candidates", pool.len());
    let mut best_prompt = pool[0].prompt.clone();
    let mut best_acc = f64::NEG_INFINITY;
    for candidate in &pool {
        let outcomes = score_materialize(&cfg, &candidate.prompt, &val)
            .context("score final materialize candidate on validation set")?;
        let candidate_acc = outcome_acc(&outcomes);
        if candidate_acc > best_acc {
            best_acc = candidate_acc;
            best_prompt = candidate.prompt.clone();
        }
    }
    if !best_acc.is_finite() {
        best_acc = 0.0;
    }

    post_materialize_feedback(&cfg, "final", "gepa_c_final_materialize", best_acc);
    post_materialize_feedback(&cfg, "final", "gepa_c_combined_acc", best_acc);
    println!(
        "TZ final: episode={} materialize={best_acc:.3} (baseline was {baseline_acc:.3})",
        cfg.episode_id,
    );

    Ok(GepaConstraintRunResult {
        prompt: best_prompt.clone(),
        research_prompt: cfg.seed_research_prompt,
        materialize_prompt: best_prompt,
        compile_fix_prompt: cfg.seed_compile_fix_prompt,
        baseline_acc,
        best_acc,
        research_acc: 0.0,
        materialize_acc: best_acc,
        compile_fix_acc: 0.0,
        candidates: pool.len(),
        episode_id: cfg.episode_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn materialize_example(episode_id: &str, parameter: &str) -> MaterializeExample {
        MaterializeExample {
            episode_id: episode_id.to_string(),
            doc: "doc.pdf".to_string(),
            parameter: parameter.to_string(),
            workbook_excerpt: format!("workbook excerpt for {parameter}"),
            gold_csp: format!("{parameter}\n41\n42\n"),
            gold_values: vec!["41".to_string(), "42".to_string()],
            synthetic: true,
        }
    }

    fn test_cfg() -> GepaConstraintConfig {
        GepaConstraintConfig {
            gateway: "http://127.0.0.1:3000".to_string(),
            research_function: "cresearch".to_string(),
            materialize_function: "cmaterialize".to_string(),
            compile_fix_function: "ccompile_fix".to_string(),
            reflect_function: "gepa_reflect".to_string(),
            variant: "baseline".to_string(),
            episode_id: "ep".to_string(),
            tags: json!({}),
            val_frac: 0.3,
            budget: 1,
            minibatch: 3,
            seed_research_prompt: "research prompt".to_string(),
            seed_materialize_prompt: "materialize prompt".to_string(),
            seed_compile_fix_prompt: "compile fix prompt".to_string(),
            hit_threshold: 0.5,
            seed: 7,
            pareto_size: 6,
            w_research: 0.35,
            w_materialize: 0.45,
            w_compile_fix: 0.20,
            work: "kb-test".into(),
        }
    }

    #[test]
    fn combined_acc_renormalizes_weights_for_non_empty_constraint_pools() {
        let cfg = test_cfg();

        let score = combined_acc_from_pools(true, true, false, 0.0, 1.0, 0.25, &cfg);

        assert!((score - (0.45 / (0.35 + 0.45))).abs() < 1e-9);
    }

    #[test]
    fn split_by_episode_holdout() {
        let examples = vec![
            materialize_example("e1", "A"),
            materialize_example("e2", "B"),
            materialize_example("e2", "C"),
            materialize_example("e3", "D"),
        ];

        let (train, val) = split_by_episode(&examples, |ex| &ex.episode_id, 0.34);

        assert_eq!(val.len(), 1);
        assert_eq!(val[0].episode_id, "e3");
        assert_eq!(train.len(), 3);
        assert!(train.iter().all(|ex| ex.episode_id != "e3"));
    }

    #[test]
    fn reflect_instruction_mentions_recall() {
        let ctx = ReflectContext {
            parent_prompt: "seed prompt".to_string(),
            parent_acc: 0.25,
            traces: vec![MaterializeTrace {
                ok: false,
                ex: materialize_example("e1", "Обозначение типа"),
                model_csp: "Обозначение типа\n41\n".to_string(),
                recall: 0.5,
            }],
        };

        let msg = build_reflect_instruction(&ctx);

        assert!(msg.contains("PARENT SCORES ON MINIBATCH"));
        assert!(msg.contains("materialize=0.250"));
        assert!(msg.contains("Обозначение типа"));
        assert!(msg.contains("Gold values: 41, 42"));
        assert!(msg.contains("Recall: 0.500"));
        assert!(msg.contains("(FAIL)"));
        assert!(msg.contains("SYSTEM PROMPT"));
        assert!(msg.contains(".csp"));
    }
}
