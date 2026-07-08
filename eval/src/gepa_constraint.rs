//! GEPA-style reflective optimization for constraint `.csp` materialization.

pub use crate::constraint_synthetic::MaterializeExample;
use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Duration;

const MINIBATCH_RESAMPLE_ATTEMPTS: usize = 3;

pub struct GepaConstraintConfig {
    pub gateway: String,
    pub materialize_function: String,
    pub reflect_function: String,
    pub variant: String,
    pub episode_id: String,
    pub tags: Value,
    pub val_frac: f64,
    pub budget: usize,
    pub minibatch: usize,
    pub seed_prompt: String,
    pub hit_threshold: f64,
    pub seed: u64,
    pub pareto_size: usize,
}

pub struct GepaConstraintRunResult {
    pub prompt: String,
    pub baseline_acc: f64,
    pub best_acc: f64,
    pub materialize_acc: f64,
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

fn outcomes_to_bools(scores: &[MaterializeOutcome]) -> Vec<bool> {
    scores.iter().map(|o| o.ok).collect()
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

    let baseline_out = score_materialize(&cfg, &cfg.seed_prompt, &val)
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

    let base_pareto = score_materialize(&cfg, &cfg.seed_prompt, &pareto_set)
        .context("score baseline materialize on pareto set")?;
    let mut pool = vec![Candidate {
        prompt: cfg.seed_prompt.clone(),
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
        prompt: best_prompt,
        baseline_acc,
        best_acc,
        materialize_acc: best_acc,
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
