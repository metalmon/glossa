//! GEPA-style reflective optimization of the `select` prompt against the dumped `select.jsonl`.
//!
//! The select sub-task: given a question and the numbered search hits (one of which is gold), the
//! local model must reply with the gold chunk's `#ord`. GEPA improves the *system prompt* that drives
//! that pick by reflection: run the current prompt, collect failures, hand them to a strong (cloud)
//! mutator LM that proposes a better prompt, keep it only if it beats its parent. A Pareto frontier
//! over per-validation-example scores preserves prompt diversity so we don't collapse to one local
//! optimum.
//!
//! Two endpoints: the per-case `select` calls hit the LOCAL gateway (qwen via LM Studio — free, high
//! volume); the reflection calls hit a cloud OpenAI-compatible endpoint (OpenRouter — tiny volume).

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::Duration;

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
    /// Local gateway (OpenAI-compat) for the per-case select calls.
    pub gateway: String,
    /// Model id for the select step.
    pub model: String,
    /// Cloud OpenAI-compatible endpoint for the reflection/mutator LM.
    pub mutator_endpoint: String,
    /// Reflection model id (e.g. `deepseek/deepseek-r1`).
    pub mutator_model: String,
    /// Bearer key for the mutator endpoint (None for keyless).
    pub mutator_key: Option<String>,
    /// Fraction of examples held out for validation (the rest are the reflection train pool).
    pub val_frac: f64,
    /// Number of reflection iterations (each = 1 mutator call + a minibatch re-score).
    pub budget: usize,
    /// How many failures to show the mutator per reflection.
    pub minibatch: usize,
    /// The starting select prompt to improve.
    pub seed_prompt: String,
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

/// Run one select call: ask the local model to pick the chunk number. Returns the chosen ord
/// (None on parse/transport failure — counted as a miss).
fn select_pick(cfg: &GepaConfig, prompt: &str, ex: &SelectExample) -> Option<u64> {
    let url = format!("{}/openai/v1/chat/completions", cfg.gateway.trim_end_matches('/'));
    let body = json!({
        "model": cfg.model, "temperature": 0.0,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": format!(
                "Question: {}\n\nResults:\n{}\n\nWhich result number contains the answer?",
                ex.question, render_hits(&ex.hits))}
        ]
    });
    let resp = ureq::post(&url).timeout(Duration::from_secs(120)).set("Content-Type", "application/json").send_string(&body.to_string());
    let text = resp.ok()?.into_string().ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    first_int(v["choices"][0]["message"]["content"].as_str().unwrap_or(""))
}

/// Score a prompt over a set of examples: per-example correctness (the pick is one of the gold ords).
fn score(cfg: &GepaConfig, prompt: &str, examples: &[SelectExample]) -> Vec<bool> {
    examples
        .iter()
        .map(|ex| select_pick(cfg, prompt, ex).map(|p| ex.gold_ords.contains(&p)).unwrap_or(false))
        .collect()
}

fn acc(scores: &[bool]) -> f64 {
    if scores.is_empty() { return 0.0; }
    scores.iter().filter(|b| **b).count() as f64 / scores.len() as f64
}

/// Reflection: hand the mutator the current prompt and a batch of failures (question, the numbered
/// hits, the gold ord, and what the model wrongly picked) and ask for an improved system prompt.
fn reflect(cfg: &GepaConfig, prompt: &str, failures: &[(&SelectExample, Option<u64>)]) -> Result<String> {
    let mut cases = String::new();
    for (i, (ex, pick)) in failures.iter().enumerate() {
        cases.push_str(&format!(
            "--- Failure {} ---\nQuestion: {}\nResults:\n{}\nGold answer: #{}\nModel picked: {}\n\n",
            i + 1, ex.question, render_hits(&ex.hits),
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
    let url = format!("{}/chat/completions", cfg.mutator_endpoint.trim_end_matches('/'));
    let body = json!({
        "model": cfg.mutator_model, "temperature": 1.0,
        "messages": [{"role": "user", "content": instruction}]
    });
    let mut req = ureq::post(&url).timeout(Duration::from_secs(300)).set("Content-Type", "application/json");
    if let Some(k) = &cfg.mutator_key {
        req = req.set("Authorization", &format!("Bearer {k}"));
    }
    let text = req.send_string(&body.to_string()).context("mutator request")?.into_string()?;
    let v: Value = serde_json::from_str(&text).context("mutator response not JSON")?;
    let out = v["choices"][0]["message"]["content"].as_str().unwrap_or("").trim().to_string();
    if out.is_empty() {
        anyhow::bail!("mutator returned an empty prompt (response: {text})");
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
            return false; // a worse somewhere → cannot dominate
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

/// Run the GEPA optimization loop and return the best prompt found (highest validation accuracy).
pub fn run(cfg: GepaConfig, examples: Vec<SelectExample>) -> Result<String> {
    anyhow::ensure!(!examples.is_empty(), "no select examples to optimize against");
    // Deterministic split: trailing `val_frac` is validation, the rest is the reflection train pool.
    let n_val = ((examples.len() as f64 * cfg.val_frac).round() as usize).clamp(1, examples.len() - 1);
    let split = examples.len() - n_val;
    let (train, val) = examples.split_at(split);
    println!("gepa: {} examples ({} train, {} val), budget={}, minibatch={}, mutator={}",
        examples.len(), train.len(), val.len(), cfg.budget, cfg.minibatch, cfg.mutator_model);

    let base_val = score(&cfg, &cfg.seed_prompt, val);
    println!("baseline select_acc(val)={:.3} ({}/{})", acc(&base_val), base_val.iter().filter(|b| **b).count(), val.len());
    let mut pool = vec![Candidate { prompt: cfg.seed_prompt.clone(), val: base_val }];

    for it in 0..cfg.budget {
        // Parent = rotate through the current frontier (keeps prompt diversity, no RNG needed).
        let front = frontier(&pool);
        let parent_idx = front[it % front.len()];
        let parent_prompt = pool[parent_idx].prompt.clone();

        // Find this parent's failures on the train pool to reflect on.
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
            Err(e) => { println!("[iter {it}] reflection failed: {e:#} — skipping"); continue; }
        };
        // Cheap gate: the child must beat its parent on the reflected minibatch before we pay for val.
        let mb: Vec<SelectExample> = failures.iter().map(|(ex, _)| (*ex).clone()).collect();
        let child_mb = score(&cfg, &child_prompt, &mb);
        if acc(&child_mb) <= 0.0 {
            println!("[iter {it}] child fixed 0/{} minibatch failures — discarded", mb.len());
            continue;
        }
        let child_val = score(&cfg, &child_prompt, val);
        println!("[iter {it}] parent#{parent_idx} acc(val)={:.3} -> child acc(val)={:.3} (fixed {}/{} minibatch)",
            acc(&pool[parent_idx].val), acc(&child_val), child_mb.iter().filter(|b| **b).count(), mb.len());
        pool.push(Candidate { prompt: child_prompt, val: child_val });
    }

    let best = pool.iter().max_by(|a, b| acc(&a.val).partial_cmp(&acc(&b.val)).unwrap()).unwrap();
    println!("DONE: best select_acc(val)={:.3} ({} candidates explored)", acc(&best.val), pool.len());
    Ok(best.prompt.clone())
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
        // paged location (p.N) renders the file type as the label; a heading renders the heading.
        assert_eq!(out, "[#7] doc.pdf · pdf · the answer\n[#3] n.md · Intro · hello");
    }

    #[test]
    fn dominance_is_strict_pareto() {
        assert!(dominates(&[true, true], &[true, false])); // ≥ everywhere, better once
        assert!(!dominates(&[true, false], &[false, true])); // trade-off → neither dominates
        assert!(!dominates(&[true, true], &[true, true])); // equal → not strict
        assert!(!dominates(&[false, false], &[true, true]));
    }

    #[test]
    fn frontier_keeps_nondominated_only() {
        let pool = vec![
            Candidate { prompt: "a".into(), val: vec![true, false] },
            Candidate { prompt: "b".into(), val: vec![false, true] },
            Candidate { prompt: "c".into(), val: vec![false, false] }, // dominated by both a and b
        ];
        let f = frontier(&pool);
        assert_eq!(f, vec![0, 1]);
    }
}
