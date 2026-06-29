//! GEPA-style reflective optimization of the `select` prompt against the dumped `select.jsonl`.
//!
//! The select sub-task: given a question and the numbered search hits (one of which is gold), the
//! local model must reply with the gold chunk's `#ord`. GEPA improves the *system prompt* that drives
//! that pick by reflection: run the current prompt, collect failures, hand them to a strong (cloud)
//! mutator LM that proposes a better prompt, keep it only if it beats its parent. A Pareto frontier
//! over per-validation-example scores preserves prompt diversity so we don't collapse to one local
//! optimum.
//!
//! Select calls go through the TensorZero `select` function (`POST /inference`) so they appear in the
//! UI/ClickHouse; run-level metrics are posted as episode feedback at the end.

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::{Duration, SystemTime};

/// One select example from `select.jsonl`: the question, the hits to choose among (each carrying its
/// `#ord`), and the gold ord(s) that count as a correct pick.
#[derive(Deserialize, Clone)]
pub struct SelectExample {
    pub question: String,
    pub hits: Vec<Value>,
    pub gold_ords: Vec<u64>,
}

/// Knobs for one optimization run.
pub struct GepaConfig {
    /// TensorZero gateway base URL.
    pub gateway: String,
    /// TZ function for per-case select calls (logged under this name in the UI).
    pub function: String,
    /// TZ variant — filter runs in the UI by variant; add new variants to `tensorzero.toml`.
    pub variant: String,
    /// TZ function for reflection/mutator calls.
    pub reflect_function: String,
    /// One episode id for the whole GEPA run (all select + reflect inferences + feedback).
    pub episode_id: String,
    /// Flat `{key: value}` tags on inferences and feedback (e.g. `run`, `budget`).
    pub tags: Value,
    /// Fraction of examples held out for validation (the rest are the reflection train pool).
    pub val_frac: f64,
    /// Number of reflection iterations (each = 1 mutator call + a minibatch re-score).
    pub budget: usize,
    /// How many failures to show the mutator per reflection.
    pub minibatch: usize,
    /// The starting select prompt to improve.
    pub seed_prompt: String,
}

/// Outcome of one GEPA run (also posted to TZ `/feedback`).
pub struct GepaRunResult {
    pub prompt: String,
    pub baseline_acc: f64,
    pub best_acc: f64,
    pub candidates: usize,
    pub episode_id: String,
}

/// First run of ASCII digits in a string (e.g. "[#45]" / "45" / "chunk 45" -> 45).
fn first_int(s: &str) -> Option<u64> {
    let digits: String = s.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// Render the hits a select example must choose among — the same one-line-per-hit shape the eval/MCP
/// search renders, reconstructed from the dumped JSON so the model sees production-identical results.
fn render_hits(hits: &[Value]) -> String {
    hits.iter()
        .map(|h| {
            let ord = h["ord"].as_u64().unwrap_or(0);
            let path = h["path"].as_str().unwrap_or("");
            let location = h["location"].as_str().unwrap_or("");
            let file_type = h["file_type"].as_str().unwrap_or("");
            let snippet = h["snippet"].as_str().unwrap_or("");
            let label = if location.starts_with("p.") { file_type } else { location };
            format!("[#{ord}] {path} · {label} · {snippet}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn select_user_content(ex: &SelectExample) -> String {
    format!(
        "Question: {}\n\nResults:\n{}\n\nWhich result number contains the answer?",
        ex.question,
        render_hits(&ex.hits),
    )
}

/// Run one select call via TZ `select`: dynamic system prompt + user turn. Returns chosen ord.
fn select_pick(cfg: &GepaConfig, prompt: &str, ex: &SelectExample) -> Option<u64> {
    let messages = [json!({"role": "user", "content": select_user_content(ex)})];
    let turn = crate::tz::infer(
        &cfg.gateway,
        &cfg.function,
        &cfg.episode_id,
        &messages,
        &cfg.tags,
        Duration::from_secs(120),
        Some(&cfg.variant),
        Some(prompt),
    )
    .inspect_err(|e| eprintln!("select inference failed: {e:#}"))
    .ok()?;
    first_int(&turn.text())
}

/// Score a prompt over a set of examples: per-example correctness (the pick is one of the gold ords).
fn score(cfg: &GepaConfig, prompt: &str, examples: &[SelectExample]) -> Vec<bool> {
    examples
        .iter()
        .map(|ex| select_pick(cfg, prompt, ex).map(|p| ex.gold_ords.contains(&p)).unwrap_or(false))
        .collect()
}

fn acc(scores: &[bool]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    scores.iter().filter(|b| **b).count() as f64 / scores.len() as f64
}

/// Reflection: hand the mutator the current prompt and a batch of failures and ask for an improved prompt.
fn reflect(cfg: &GepaConfig, prompt: &str, failures: &[(&SelectExample, Option<u64>)]) -> Result<String> {
    let mut cases = String::new();
    for (i, (ex, pick)) in failures.iter().enumerate() {
        cases.push_str(&format!(
            "--- Failure {} ---\nQuestion: {}\nResults:\n{}\nGold answer: #{}\nModel picked: {}\n\n",
            i + 1,
            ex.question,
            render_hits(&ex.hits),
            ex.gold_ords.iter().map(|o| o.to_string()).collect::<Vec<_>>().join(" or #"),
            pick.map(|p| format!("#{p}")).unwrap_or_else(|| "(no valid number)".into()),
        ));
    }
    let instruction = format!(
        "You are improving the SYSTEM PROMPT for a retrieval model. Given a question and numbered \
         search results (each prefixed `[#N]`), the model must reply with ONLY the number of the \
         result that answers the question. Below is the current system prompt and cases where it \
         picked the wrong number. Diagnose the recurring mistake and rewrite the system prompt so the \
         model picks correctly. Keep it concise and general (do NOT mention these specific cases). \
         Reply with ONLY the new system prompt text — no preamble, no quotes, no explanation.\n\n\
         === CURRENT SYSTEM PROMPT ===\n{prompt}\n\n=== FAILURES ===\n{cases}=== NEW SYSTEM PROMPT ==="
    );
    let messages = [json!({"role": "user", "content": instruction})];
    let turn = crate::tz::infer(
        &cfg.gateway,
        &cfg.reflect_function,
        &cfg.episode_id,
        &messages,
        &cfg.tags,
        Duration::from_secs(180),
        Some(&cfg.variant),
        None,
    )
    .context("gepa_reflect inference failed")?;
    let out = turn.text().trim().to_string();
    if out.is_empty() {
        anyhow::bail!("gepa_reflect returned an empty prompt");
    }
    Ok(out)
}

/// A prompt variant and how it scored on each validation example (the vector drives Pareto dominance).
struct Candidate {
    prompt: String,
    val: Vec<bool>,
}

/// `a` Pareto-dominates `b` iff a is ≥ on every validation example and strictly better on at least one.
fn dominates(a: &[bool], b: &[bool]) -> bool {
    let mut strictly = false;
    for (x, y) in a.iter().zip(b) {
        if !x && *y {
            return false;
        }
        if *x && !*y {
            strictly = true;
        }
    }
    strictly
}

/// Indices of candidates on the Pareto frontier (not dominated by any other candidate).
fn frontier(pool: &[Candidate]) -> Vec<usize> {
    (0..pool.len())
        .filter(|&i| !pool.iter().enumerate().any(|(j, c)| j != i && dominates(&c.val, &pool[i].val)))
        .collect()
}

fn post_run_feedback(cfg: &GepaConfig, result: &GepaRunResult, n_train: usize, n_val: usize) {
    let ep = &result.episode_id;
    let tags = &cfg.tags;
    crate::tz::post_feedback(&cfg.gateway, ep, "select_baseline_acc", json!(result.baseline_acc), tags);
    crate::tz::post_feedback(&cfg.gateway, ep, "select_acc", json!(result.best_acc), tags);
    crate::tz::post_feedback(&cfg.gateway, ep, "gepa_candidates", json!(result.candidates as f64), tags);
    crate::tz::post_feedback(&cfg.gateway, ep, "gepa_examples_train", json!(n_train as f64), tags);
    crate::tz::post_feedback(&cfg.gateway, ep, "gepa_examples_val", json!(n_val as f64), tags);
    println!(
        "TZ feedback: episode={} function={} variant={} select_acc={:.3} baseline={:.3}",
        ep, cfg.function, cfg.variant, result.best_acc, result.baseline_acc,
    );
}

/// Run the GEPA optimization loop; post run metrics to TZ and return the best prompt found.
pub fn run(cfg: GepaConfig, examples: Vec<SelectExample>) -> Result<GepaRunResult> {
    anyhow::ensure!(!examples.is_empty(), "no select examples to optimize against");
    let n_val = ((examples.len() as f64 * cfg.val_frac).round() as usize).clamp(1, examples.len() - 1);
    let split = examples.len() - n_val;
    let (train, val) = examples.split_at(split);
    println!(
        "gepa: {} examples ({} train, {} val), budget={}, minibatch={}, function={}, variant={}, episode={}",
        examples.len(),
        train.len(),
        val.len(),
        cfg.budget,
        cfg.minibatch,
        cfg.function,
        cfg.variant,
        cfg.episode_id,
    );

    let base_val = score(&cfg, &cfg.seed_prompt, val);
    let baseline_acc = acc(&base_val);
    println!(
        "baseline select_acc(val)={:.3} ({}/{})",
        baseline_acc,
        base_val.iter().filter(|b| **b).count(),
        val.len()
    );
    let mut pool = vec![Candidate {
        prompt: cfg.seed_prompt.clone(),
        val: base_val,
    }];

    for it in 0..cfg.budget {
        let front = frontier(&pool);
        let parent_idx = front[it % front.len()];
        let parent_prompt = pool[parent_idx].prompt.clone();

        let train_scores = score(&cfg, &parent_prompt, train);
        let failures: Vec<(&SelectExample, Option<u64>)> = train
            .iter()
            .zip(&train_scores)
            .filter(|(_, ok)| !**ok)
            .take(cfg.minibatch)
            .map(|(ex, _)| (ex, select_pick(&cfg, &parent_prompt, ex)))
            .collect();
        if failures.is_empty() {
            println!("[iter {it}] parent has no train failures — perfect on train, stopping reflection");
            break;
        }

        let child_prompt = match reflect(&cfg, &parent_prompt, &failures) {
            Ok(p) => p,
            Err(e) => {
                println!("[iter {it}] reflection failed: {e:#} — skipping");
                continue;
            }
        };
        let mb: Vec<SelectExample> = failures.iter().map(|(ex, _)| (*ex).clone()).collect();
        let child_mb = score(&cfg, &child_prompt, &mb);
        if acc(&child_mb) <= 0.0 {
            println!("[iter {it}] child fixed 0/{} minibatch failures — discarded", mb.len());
            continue;
        }
        let child_val = score(&cfg, &child_prompt, val);
        println!(
            "[iter {it}] parent#{parent_idx} acc(val)={:.3} -> child acc(val)={:.3} (fixed {}/{} minibatch)",
            acc(&pool[parent_idx].val),
            acc(&child_val),
            child_mb.iter().filter(|b| **b).count(),
            mb.len()
        );
        pool.push(Candidate {
            prompt: child_prompt,
            val: child_val,
        });
    }

    let best = pool
        .iter()
        .max_by(|a, b| acc(&a.val).partial_cmp(&acc(&b.val)).unwrap())
        .unwrap();
    let result = GepaRunResult {
        prompt: best.prompt.clone(),
        baseline_acc,
        best_acc: acc(&best.val),
        candidates: pool.len(),
        episode_id: cfg.episode_id.clone(),
    };
    println!(
        "DONE: best select_acc(val)={:.3} ({} candidates explored)",
        result.best_acc,
        result.candidates
    );
    post_run_feedback(&cfg, &result, train.len(), val.len());
    Ok(result)
}

/// Default run label for TZ tags when the caller does not supply one.
pub fn default_run_tag() -> String {
    let secs = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("gepa-{secs}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_hits_matches_search_line_shape() {
        let hits = vec![
            json!({"ord": 7, "path": "doc.pdf", "location": "p.7", "file_type": "pdf", "snippet": "the answer"}),
            json!({"ord": 3, "path": "n.md", "location": "Intro", "file_type": "md", "snippet": "hello"}),
        ];
        let out = render_hits(&hits);
        assert_eq!(out, "[#7] doc.pdf · pdf · the answer\n[#3] n.md · Intro · hello");
    }

    #[test]
    fn dominance_is_strict_pareto() {
        assert!(dominates(&[true, true], &[true, false]));
        assert!(!dominates(&[true, false], &[false, true]));
        assert!(!dominates(&[true, true], &[true, true]));
        assert!(!dominates(&[false, false], &[true, true]));
    }

    #[test]
    fn frontier_keeps_nondominated_only() {
        let pool = vec![
            Candidate { prompt: "a".into(), val: vec![true, false] },
            Candidate { prompt: "b".into(), val: vec![false, true] },
            Candidate { prompt: "c".into(), val: vec![false, false] },
        ];
        let f = frontier(&pool);
        assert_eq!(f, vec![0, 1]);
    }
}
