//! GEPA-style reflective optimization of the prod `answer_hotpot` system prompt.
//!
//! Dual sub-tasks scored deterministically:
//! - **search**: TZ `functions.search` → model emits `search(query)` → gold chunk in top-k?
//! - **read**: TZ `functions.read` → after prefilled hits, model emits `read(path,n)` → matches gold?

use anyhow::{Context, Result};
use crate::export_tz::{ReadExample, ReadPick};
use glossa::index::store::DocIndex;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;
use std::time::{Duration, SystemTime};

/// One query example from `query.jsonl`.
#[derive(Deserialize, Clone)]
pub struct QueryExample {
    pub episode_id: String,
    pub question: String,
    pub search_query: String,
    pub gold: Vec<String>,
    pub hit: bool,
}

/// Knobs for one optimization run.
pub struct GepaConfig {
    pub gateway: String,
    pub search_function: String,
    pub read_function: String,
    pub variant: String,
    pub reflect_function: String,
    pub episode_id: String,
    pub tags: Value,
    pub val_frac: f64,
    pub budget: usize,
    pub minibatch: usize,
    pub seed_prompt: String,
    pub work: std::path::PathBuf,
    pub top_k: usize,
    pub w_query: f64,
    pub w_read: f64,
}

pub struct GepaRunResult {
    pub prompt: String,
    pub baseline_acc: f64,
    pub best_acc: f64,
    pub query_acc: f64,
    pub read_acc: f64,
    pub candidates: usize,
    pub episode_id: String,
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

fn normalize_path(p: &str) -> String {
    p.replace('\\', "/")
}

fn gold_pairs(gold: &[String]) -> Vec<(String, String)> {
    gold.iter().filter_map(|g| parse_gold_loc(g)).collect()
}

fn tool_calls(content: &[Value]) -> Vec<(String, Value)> {
    content
        .iter()
        .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_call"))
        .map(|b| {
            let name = b.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
            let args = match b.get("arguments") {
                Some(Value::Object(_)) => b.get("arguments").cloned().unwrap_or(json!({})),
                Some(Value::String(s)) => serde_json::from_str(s).unwrap_or(json!({})),
                _ => json!({}),
            };
            (name, args)
        })
        .collect()
}

fn first_search_call(content: &[Value]) -> Option<String> {
    tool_calls(content)
        .into_iter()
        .find(|(n, _)| n == "search")
        .and_then(|(_, args)| args.get("query").and_then(|q| q.as_str()).map(str::to_string))
}

fn first_read_call(content: &[Value]) -> Option<ReadPick> {
    tool_calls(content)
        .into_iter()
        .find(|(n, _)| n == "read")
        .and_then(|(_, args)| {
            let path = args.get("path").and_then(|p| p.as_str())?.to_string();
            let n = args
                .get("n")
                .and_then(|n| n.as_u64().or_else(|| n.as_i64().map(|i| i as u64)))?;
            Some(ReadPick { path, n })
        })
}

fn render_hits(hits: &[Value]) -> String {
    hits.iter()
        .map(|h| {
            let ord = h["ord"].as_u64().unwrap_or(0);
            let path = h["path"].as_str().unwrap_or("");
            let location = h["location"].as_str().unwrap_or("");
            let file_type = h["file_type"].as_str().unwrap_or("");
            let snippet = h["snippet"].as_str().unwrap_or("");
            let label = if location.starts_with("p.") || location == "pdf" {
                file_type
            } else if !location.is_empty() {
                location
            } else {
                file_type
            };
            format!("[#{ord}] {path} · {label} · {snippet}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn search_top_k(idx: &DocIndex, query: &str, k: usize) -> Vec<glossa::index::store::RankedHit> {
    idx.search_filtered(query, k, None, None).unwrap_or_default()
}

fn render_ranked_hits(hits: &[glossa::index::store::RankedHit]) -> String {
    hits.iter().map(glossa::index::store::RankedHit::display_line).collect::<Vec<_>>().join("\n")
}

fn query_hit_hits(hits: &[glossa::index::store::RankedHit], gold: &[(String, String)]) -> bool {
    for h in hits {
        for (gp, gl) in gold {
            if normalize_path(&h.path) == normalize_path(gp) && h.location == *gl {
                return true;
            }
        }
    }
    false
}

fn read_hit(idx: &DocIndex, pick: &ReadPick, gold: &[(String, String)]) -> bool {
    for (gp, gl) in gold {
        if normalize_path(&pick.path) != normalize_path(gp) {
            continue;
        }
        if let Ok(Some(loc)) = idx.location_for_ord(&pick.path, pick.n) {
            if loc == *gl {
                return true;
            }
        }
    }
    false
}

fn user_question_msg(question: &str) -> Value {
    json!({"role": "user", "content": format!("Question: {question}")})
}

fn prefilled_read_messages(ex: &ReadExample) -> Vec<Value> {
    vec![
        user_question_msg(&ex.question),
        json!({
            "role": "assistant",
            "content": [{
                "type": "tool_call",
                "id": "prefill-search",
                "name": "search",
                "arguments": {"query": ex.search_query}
            }]
        }),
        json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "id": "prefill-search",
                "name": "search",
                "result": render_hits(&ex.hits)
            }]
        }),
    ]
}

fn infer_prompt(
    cfg: &GepaConfig,
    function: &str,
    prompt: &str,
    messages: &[Value],
) -> Result<crate::tz::InferenceTurn> {
    crate::tz::infer(
        &cfg.gateway,
        function,
        &cfg.episode_id,
        messages,
        &cfg.tags,
        Duration::from_secs(120),
        Some(&cfg.variant),
        Some(prompt),
    )
}

struct QueryOutcome {
    ok: bool,
    model_search: Option<String>,
    top_k: String,
}

struct ReadOutcome {
    ok: bool,
    model_read: Option<ReadPick>,
    resolved: Option<String>,
}

fn score_query_one(cfg: &GepaConfig, prompt: &str, ex: &QueryExample, idx: &DocIndex) -> QueryOutcome {
    let messages = [user_question_msg(&ex.question)];
    let turn = match infer_prompt(cfg, &cfg.search_function, prompt, &messages) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("query inference failed: {e:#}");
            return QueryOutcome {
                ok: false,
                model_search: None,
                top_k: String::new(),
            };
        }
    };
    let Some(query) = first_search_call(&turn.content) else {
        return QueryOutcome {
            ok: false,
            model_search: None,
            top_k: String::new(),
        };
    };
    let hits = search_top_k(idx, &query, cfg.top_k);
    let gold = gold_pairs(&ex.gold);
    QueryOutcome {
        ok: query_hit_hits(&hits, &gold),
        model_search: Some(query),
        top_k: render_ranked_hits(&hits),
    }
}

fn score_read_one(cfg: &GepaConfig, prompt: &str, ex: &ReadExample, idx: &DocIndex) -> ReadOutcome {
    let messages = prefilled_read_messages(ex);
    let turn = match infer_prompt(cfg, &cfg.read_function, prompt, &messages) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("read inference failed: {e:#}");
            return ReadOutcome {
                ok: false,
                model_read: None,
                resolved: None,
            };
        }
    };
    let Some(pick) = first_read_call(&turn.content) else {
        return ReadOutcome {
            ok: false,
            model_read: None,
            resolved: None,
        };
    };
    let resolved = idx.location_for_ord(&pick.path, pick.n).ok().flatten();
    ReadOutcome {
        ok: read_hit(idx, &pick, &gold_pairs(&ex.gold)),
        model_read: Some(pick),
        resolved,
    }
}

fn score_dual(
    cfg: &GepaConfig,
    prompt: &str,
    queries: &[QueryExample],
    reads: &[ReadExample],
    idx: &DocIndex,
) -> (Vec<QueryOutcome>, Vec<ReadOutcome>) {
    let q_scores = queries.iter().map(|ex| score_query_one(cfg, prompt, ex, idx)).collect();
    let r_scores = reads.iter().map(|ex| score_read_one(cfg, prompt, ex, idx)).collect();
    (q_scores, r_scores)
}

fn query_acc(q: &[QueryOutcome]) -> f64 {
    if q.is_empty() {
        return 0.0;
    }
    q.iter().filter(|o| o.ok).count() as f64 / q.len() as f64
}

fn read_acc(r: &[ReadOutcome]) -> f64 {
    if r.is_empty() {
        return 0.0;
    }
    r.iter().filter(|o| o.ok).count() as f64 / r.len() as f64
}

fn outcomes_to_bools(q: &[QueryOutcome], r: &[ReadOutcome]) -> (Vec<bool>, Vec<bool>) {
    (q.iter().map(|o| o.ok).collect(), r.iter().map(|o| o.ok).collect())
}

fn acc(scores: &[bool]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    scores.iter().filter(|b| **b).count() as f64 / scores.len() as f64
}

fn combined_acc(q: f64, r: f64, cfg: &GepaConfig) -> f64 {
    let wq = cfg.w_query;
    let wr = cfg.w_read;
    if wq + wr <= 0.0 {
        return (q + r) / 2.0;
    }
    (wq * q + wr * r) / (wq + wr)
}

#[derive(Clone)]
enum FailureCase {
    Query {
        ex: QueryExample,
        model_search: Option<String>,
        top_k: String,
    },
    Read {
        ex: ReadExample,
        model_read: Option<ReadPick>,
        resolved: Option<String>,
    },
}

/// Split minibatch between query and read failures when both sub-tasks have train rows.
fn failure_budget(minibatch: usize, have_query: bool, have_read: bool) -> (usize, usize) {
    if minibatch == 0 {
        return (0, 0);
    }
    if !have_query {
        return (0, minibatch);
    }
    if !have_read {
        return (minibatch, 0);
    }
    let q = (minibatch + 1) / 2;
    (q, minibatch - q)
}

/// Interleave query/read failures so the mutator sees both sub-tasks when possible.
fn pick_balanced_failures(
    q: Vec<FailureCase>,
    r: Vec<FailureCase>,
    minibatch: usize,
) -> Vec<FailureCase> {
    let (want_q, want_r) = failure_budget(minibatch, !q.is_empty(), !r.is_empty());
    let mut out = Vec::with_capacity(minibatch.min(q.len() + r.len()));
    let mut qi = 0usize;
    let mut ri = 0usize;
    let mut picked_q = 0usize;
    let mut picked_r = 0usize;
    while out.len() < minibatch {
        let need_q = picked_q < want_q && qi < q.len();
        let need_r = picked_r < want_r && ri < r.len();
        if !need_q && !need_r {
            break;
        }
        let take_q = need_q && (!need_r || picked_q <= picked_r);
        if take_q {
            out.push(q[qi].clone());
            qi += 1;
            picked_q += 1;
        } else {
            out.push(r[ri].clone());
            ri += 1;
            picked_r += 1;
        }
    }
    while out.len() < minibatch && qi < q.len() {
        out.push(q[qi].clone());
        qi += 1;
    }
    while out.len() < minibatch && ri < r.len() {
        out.push(r[ri].clone());
        ri += 1;
    }
    out
}

/// Score train examples until enough balanced failures are collected (early stop).
fn collect_train_failures(
    cfg: &GepaConfig,
    prompt: &str,
    train_q: &[QueryExample],
    train_r: &[ReadExample],
    idx: &DocIndex,
) -> Vec<FailureCase> {
    let (want_q, want_r) = failure_budget(cfg.minibatch, !train_q.is_empty(), !train_r.is_empty());
    let mut q_failures = Vec::new();
    let mut r_failures = Vec::new();
    let mut q_i = 0usize;
    let mut r_i = 0usize;

    while q_i < train_q.len() {
        if q_failures.len() >= want_q && (want_r == 0 || r_failures.len() >= want_r) {
            break;
        }
        let ex = &train_q[q_i];
        q_i += 1;
        let outcome = score_query_one(cfg, prompt, ex, idx);
        if !outcome.ok {
            q_failures.push(FailureCase::Query {
                ex: ex.clone(),
                model_search: outcome.model_search,
                top_k: outcome.top_k,
            });
        }
    }

    while r_i < train_r.len() {
        if r_failures.len() >= want_r && q_failures.len() >= want_q {
            break;
        }
        let ex = &train_r[r_i];
        r_i += 1;
        let outcome = score_read_one(cfg, prompt, ex, idx);
        if !outcome.ok {
            r_failures.push(FailureCase::Read {
                ex: ex.clone(),
                model_read: outcome.model_read,
                resolved: outcome.resolved,
            });
        }
    }

    let mut picked = pick_balanced_failures(q_failures, r_failures, cfg.minibatch);
    if picked.len() >= cfg.minibatch {
        return picked;
    }

    while picked.len() < cfg.minibatch && q_i < train_q.len() {
        let ex = &train_q[q_i];
        q_i += 1;
        let outcome = score_query_one(cfg, prompt, ex, idx);
        if !outcome.ok {
            picked.push(FailureCase::Query {
                ex: ex.clone(),
                model_search: outcome.model_search,
                top_k: outcome.top_k,
            });
        }
    }
    while picked.len() < cfg.minibatch && r_i < train_r.len() {
        let ex = &train_r[r_i];
        r_i += 1;
        let outcome = score_read_one(cfg, prompt, ex, idx);
        if !outcome.ok {
            picked.push(FailureCase::Read {
                ex: ex.clone(),
                model_read: outcome.model_read,
                resolved: outcome.resolved,
            });
        }
    }
    picked.truncate(cfg.minibatch);
    picked
}

fn validate_dataset(queries: &[QueryExample], reads: &[ReadExample]) {
    let mut episodes: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for q in queries {
        episodes.insert(&q.episode_id);
    }
    for r in reads {
        episodes.insert(&r.episode_id);
    }
    if queries.is_empty() {
        eprintln!("gepa warn: no query examples — only read sub-task will be optimized");
    }
    if reads.is_empty() {
        eprintln!("gepa warn: no read examples — only query sub-task will be optimized");
    }
    if episodes.len() < 8 {
        eprintln!(
            "gepa warn: only {} unique episodes — val split may be empty or noisy; run eval on more cases",
            episodes.len()
        );
    }
}

fn format_model_search(model_search: Option<&str>, top_k: &str, k: usize) -> String {
    match model_search {
        Some(q) => {
            let results = if top_k.is_empty() {
                "(empty)".to_string()
            } else {
                top_k.to_string()
            };
            format!("Model output: search({q:?})\nTop-{k} for that query:\n{results}")
        }
        None => "Model output: no search(query) tool call".to_string(),
    }
}

fn format_model_read(model_read: Option<&ReadPick>, resolved: Option<&str>) -> String {
    match model_read {
        Some(p) => {
            let loc = resolved.map(|l| format!(" → {l}")).unwrap_or_default();
            format!("Model output: read({:?}, {}){loc}", p.path, p.n)
        }
        None => "Model output: no read(path,n) tool call".to_string(),
    }
}

fn reflect(cfg: &GepaConfig, prompt: &str, failures: &[FailureCase]) -> Result<String> {
    let mut cases = String::new();
    for (i, f) in failures.iter().enumerate() {
        match f {
            FailureCase::Query {
                ex,
                model_search,
                top_k,
            } => {
                cases.push_str(&format!(
                    "--- Query failure {} ---\nQuestion: {}\nGold chunks: {}\n{}\n\
                     Gold was not in top-{} for the model's search query.\n\n",
                    i + 1,
                    ex.question,
                    ex.gold.join(", "),
                    format_model_search(model_search.as_deref(), top_k, cfg.top_k),
                    cfg.top_k,
                ));
            }
            FailureCase::Read {
                ex,
                model_read,
                resolved,
            } => {
                cases.push_str(&format!(
                    "--- Read failure {} ---\nQuestion: {}\nPrefilled search query: {}\n\
                     Prefilled results:\n{}\nGold chunks: {}\n{}\n\
                     The model's read did not match gold.\n\n",
                    i + 1,
                    ex.question,
                    ex.search_query,
                    render_hits(&ex.hits),
                    ex.gold.join(", "),
                    format_model_read(model_read.as_ref(), resolved.as_deref()),
                ));
            }
        }
    }
    let instruction = format!(
        "You are improving the SYSTEM PROMPT for a knowledge-base agent that uses search and read tools.\n\
         Failures fall into two types:\n\
         1) QUERY — the model's search(query) did not retrieve the gold document chunk in top results.\n\
         2) READ — given search results, the model's read(path,n) did not open the gold chunk.\n\
         Diagnose recurring mistakes and rewrite the system prompt so both retrieval stages improve.\n\
         Keep it concise and general (do NOT mention specific cases). Preserve tool-use instructions.\n\
         Reply with ONLY the new system prompt text — no preamble, no quotes.\n\n\
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
        Some("baseline"),
        None,
    )
    .context("gepa_reflect inference failed")?;
    let out = turn.text().trim().to_string();
    if out.is_empty() {
        anyhow::bail!("gepa_reflect returned an empty prompt");
    }
    Ok(out)
}

struct Candidate {
    prompt: String,
    q_val: Vec<bool>,
    r_val: Vec<bool>,
}

fn dominates(a_q: &[bool], a_r: &[bool], b_q: &[bool], b_r: &[bool]) -> bool {
    let a: Vec<bool> = a_q.iter().chain(a_r.iter()).copied().collect();
    let b: Vec<bool> = b_q.iter().chain(b_r.iter()).copied().collect();
    let mut strictly = false;
    for (x, y) in a.iter().zip(&b) {
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
            !pool.iter().enumerate().any(|(j, c)| {
                j != i && dominates(&c.q_val, &c.r_val, &pool[i].q_val, &pool[i].r_val)
            })
        })
        .collect()
}

fn feedback_tags(cfg: &GepaConfig, stage: &str) -> Value {
    let mut m = cfg
        .tags
        .as_object()
        .cloned()
        .unwrap_or_default();
    m.insert("stage".into(), stage.into());
    if let Some(n) = stage.strip_prefix("iter_") {
        m.insert("iter".into(), n.into());
    }
    Value::Object(m)
}

fn post_gepa_triplet(
    gw: &str,
    episode_id: &str,
    prefix: &str,
    query_acc: f64,
    read_acc: f64,
    combined: f64,
    tags: &Value,
) {
    crate::tz::post_feedback(gw, episode_id, &format!("{prefix}_query"), json!(query_acc), tags);
    crate::tz::post_feedback(gw, episode_id, &format!("{prefix}_read"), json!(read_acc), tags);
    crate::tz::post_feedback(gw, episode_id, &format!("{prefix}_combined"), json!(combined), tags);
}

fn post_baseline_feedback(cfg: &GepaConfig, query_acc: f64, read_acc: f64, combined: f64) {
    let tags = feedback_tags(cfg, "baseline");
    post_gepa_triplet(
        &cfg.gateway,
        &cfg.episode_id,
        "gepa_baseline",
        query_acc,
        read_acc,
        combined,
        &tags,
    );
    println!(
        "TZ baseline: query={query_acc:.3} read={read_acc:.3} combined={combined:.3}"
    );
}

fn post_iter_feedback(cfg: &GepaConfig, iter: usize, query_acc: f64, read_acc: f64, combined: f64, candidates: usize) {
    let stage = format!("iter_{iter}");
    let tags = feedback_tags(cfg, &stage);
    post_gepa_triplet(
        &cfg.gateway,
        &cfg.episode_id,
        "gepa_iter",
        query_acc,
        read_acc,
        combined,
        &tags,
    );
    crate::tz::post_feedback(
        &cfg.gateway,
        &cfg.episode_id,
        "gepa_iter_candidates",
        json!(candidates as f64),
        &tags,
    );
    println!(
        "TZ iter {iter}: query={query_acc:.3} read={read_acc:.3} combined={combined:.3} candidates={candidates}"
    );
}

fn post_final_feedback(
    cfg: &GepaConfig,
    result: &GepaRunResult,
    n_train_q: usize,
    n_train_r: usize,
    n_val_q: usize,
    n_val_r: usize,
) {
    let ep = &result.episode_id;
    let tags = feedback_tags(cfg, "final");
    post_gepa_triplet(
        &cfg.gateway,
        ep,
        "gepa_final",
        result.query_acc,
        result.read_acc,
        result.best_acc,
        &tags,
    );
    // Single optimize target: best val combined after the run (posted once).
    crate::tz::post_feedback(&cfg.gateway, ep, "gepa_combined_acc", json!(result.best_acc), &tags);
    crate::tz::post_feedback(
        &cfg.gateway,
        ep,
        "gepa_candidates",
        json!(result.candidates as f64),
        &tags,
    );
    crate::tz::post_feedback(
        &cfg.gateway,
        ep,
        "gepa_examples_train",
        json!((n_train_q + n_train_r) as f64),
        &tags,
    );
    crate::tz::post_feedback(
        &cfg.gateway,
        ep,
        "gepa_examples_val",
        json!((n_val_q + n_val_r) as f64),
        &tags,
    );
    println!(
        "TZ final: episode={ep} query={:.3} read={:.3} combined={:.3} (baseline was {:.3})",
        result.query_acc,
        result.read_acc,
        result.best_acc,
        result.baseline_acc,
    );
}

fn best_pool_combined(pool: &[Candidate], cfg: &GepaConfig) -> (f64, f64, f64) {
    let best = pool
        .iter()
        .max_by(|a, b| {
            let ca = combined_acc(acc(&a.q_val), acc(&a.r_val), cfg);
            let cb = combined_acc(acc(&b.q_val), acc(&b.r_val), cfg);
            ca.partial_cmp(&cb).unwrap()
        })
        .expect("non-empty pool");
    (
        acc(&best.q_val),
        acc(&best.r_val),
        combined_acc(acc(&best.q_val), acc(&best.r_val), cfg),
    )
}

/// Load prod seed prompt from `answer_hotpot/system.minijinja`.
pub fn load_seed_prompt(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("read seed prompt {}", path.display()))
}

/// Split examples by episode_id so train/val don't leak the same question.
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
    let val_eps: std::collections::HashSet<String> = episodes.into_iter().rev().take(n_val).collect();
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

pub fn run(cfg: GepaConfig, queries: Vec<QueryExample>, reads: Vec<ReadExample>) -> Result<GepaRunResult> {
    anyhow::ensure!(
        !queries.is_empty() || !reads.is_empty(),
        "no query/read examples to optimize against"
    );
    let idx = DocIndex::open_or_create(&cfg.work).context("open index for GEPA scoring")?;
    validate_dataset(&queries, &reads);
    crate::tz::ensure_function(&cfg.gateway, &cfg.search_function, Some(&cfg.variant))
        .context("GEPA search function")?;
    crate::tz::ensure_function(&cfg.gateway, &cfg.read_function, Some(&cfg.variant))
        .context("GEPA read function")?;

    let (train_q, val_q) = split_by_episode(&queries, |q| &q.episode_id, cfg.val_frac);
    let (train_r, val_r) = split_by_episode(&reads, |r| &r.episode_id, cfg.val_frac);

    println!(
        "gepa: query {} ({} train, {} val), read {} ({} train, {} val), budget={}, work={}",
        queries.len(),
        train_q.len(),
        val_q.len(),
        reads.len(),
        train_r.len(),
        val_r.len(),
        cfg.budget,
        cfg.work.display(),
    );

    let (base_q, base_r) = score_dual(&cfg, &cfg.seed_prompt, &val_q, &val_r, &idx);
    let baseline_acc = combined_acc(query_acc(&base_q), read_acc(&base_r), &cfg);
    let base_q_acc = query_acc(&base_q);
    let base_r_acc = read_acc(&base_r);
    let (base_q_bools, base_r_bools) = outcomes_to_bools(&base_q, &base_r);
    println!(
        "baseline val: query={:.3} read={:.3} combined={:.3}",
        base_q_acc, base_r_acc, baseline_acc,
    );
    post_baseline_feedback(&cfg, base_q_acc, base_r_acc, baseline_acc);
    {
        let tags = feedback_tags(&cfg, "start");
        crate::tz::post_feedback(
            &cfg.gateway,
            &cfg.episode_id,
            "gepa_examples_train",
            json!((train_q.len() + train_r.len()) as f64),
            &tags,
        );
        crate::tz::post_feedback(
            &cfg.gateway,
            &cfg.episode_id,
            "gepa_examples_val",
            json!((val_q.len() + val_r.len()) as f64),
            &tags,
        );
    }

    let mut pool = vec![Candidate {
        prompt: cfg.seed_prompt.clone(),
        q_val: base_q_bools,
        r_val: base_r_bools,
    }];

    for it in 0..cfg.budget {
        let front = frontier(&pool);
        let parent_idx = front[it % front.len()];
        let parent_prompt = pool[parent_idx].prompt.clone();

        let failures = collect_train_failures(&cfg, &parent_prompt, &train_q, &train_r, &idx);
        let n_q_fail = failures.iter().filter(|f| matches!(f, FailureCase::Query { .. })).count();
        let n_r_fail = failures.iter().filter(|f| matches!(f, FailureCase::Read { .. })).count();
        println!(
            "[iter {it}] reflect minibatch: {} failures (query={n_q_fail} read={n_r_fail})",
            failures.len(),
        );

        if failures.is_empty() {
            println!("[iter {it}] no train failures — stopping");
            break;
        }

        let child_prompt = match reflect(&cfg, &parent_prompt, &failures) {
            Ok(p) => p,
            Err(e) => {
                println!("[iter {it}] reflection failed: {e:#}");
                continue;
            }
        };

        let mb_q: Vec<QueryExample> = failures
            .iter()
            .filter_map(|f| match f {
                FailureCase::Query { ex, .. } => Some(ex.clone()),
                _ => None,
            })
            .collect();
        let mb_r: Vec<ReadExample> = failures
            .iter()
            .filter_map(|f| match f {
                FailureCase::Read { ex, .. } => Some(ex.clone()),
                _ => None,
            })
            .collect();
        let (mb_qs, mb_rs) = score_dual(&cfg, &child_prompt, &mb_q, &mb_r, &idx);
        if query_acc(&mb_qs) + read_acc(&mb_rs) <= 0.0 {
            println!("[iter {it}] child fixed 0 minibatch failures — discarded");
            continue;
        }

        let (child_q_val, child_r_val) = score_dual(&cfg, &child_prompt, &val_q, &val_r, &idx);
        let (child_q_bools, child_r_bools) = outcomes_to_bools(&child_q_val, &child_r_val);
        let parent_combined = combined_acc(acc(&pool[parent_idx].q_val), acc(&pool[parent_idx].r_val), &cfg);
        let child_combined = combined_acc(query_acc(&child_q_val), read_acc(&child_r_val), &cfg);
        println!(
            "[iter {it}] parent combined={:.3} -> child combined={:.3} (query {:.3} read {:.3})",
            parent_combined,
            child_combined,
            query_acc(&child_q_val),
            read_acc(&child_r_val),
        );
        pool.push(Candidate {
            prompt: child_prompt,
            q_val: child_q_bools,
            r_val: child_r_bools,
        });
        let (q, r, combined) = best_pool_combined(&pool, &cfg);
        post_iter_feedback(&cfg, it, q, r, combined, pool.len());
    }

    let best = pool
        .iter()
        .max_by(|a, b| {
            let ca = combined_acc(acc(&a.q_val), acc(&a.r_val), &cfg);
            let cb = combined_acc(acc(&b.q_val), acc(&b.r_val), &cfg);
            ca.partial_cmp(&cb).unwrap()
        })
        .unwrap();

    let result = GepaRunResult {
        prompt: best.prompt.clone(),
        baseline_acc,
        best_acc: combined_acc(acc(&best.q_val), acc(&best.r_val), &cfg),
        query_acc: acc(&best.q_val),
        read_acc: acc(&best.r_val),
        candidates: pool.len(),
        episode_id: cfg.episode_id.clone(),
    };
    post_final_feedback(&cfg, &result, train_q.len(), train_r.len(), val_q.len(), val_r.len());
    Ok(result)
}

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
    fn first_read_call_parses_tool_block() {
        let content = vec![json!({
            "type": "tool_call", "name": "read",
            "arguments": {"path": "doc.htm", "n": 3}
        })];
        let pick = first_read_call(&content).unwrap();
        assert_eq!(pick.path, "doc.htm");
        assert_eq!(pick.n, 3);
    }

    #[test]
    fn first_search_call_extracts_query() {
        let content = vec![json!({
            "type": "tool_call", "name": "search",
            "arguments": {"query": "CPU MHz"}
        })];
        assert_eq!(first_search_call(&content).as_deref(), Some("CPU MHz"));
    }

    #[test]
    fn reflect_feedback_includes_model_outputs() {
        let search = format_model_search(Some("CPU MHz"), "[#1] a.pdf · p.1 · x", 8);
        assert!(search.contains("search(\"CPU MHz\")"));
        assert!(search.contains("[#1]"));

        let none = format_model_search(None, "", 8);
        assert!(none.contains("no search"));

        let read = format_model_read(
            Some(&ReadPick {
                path: "doc.htm".into(),
                n: 3,
            }),
            Some("h1"),
        );
        assert!(read.contains("read(\"doc.htm\", 3)"));
        assert!(read.contains("→ h1"));

        let no_read = format_model_read(None, None);
        assert!(no_read.contains("no read"));
    }

    #[test]
    fn failure_budget_splits_minibatch() {
        assert_eq!(failure_budget(10, true, true), (5, 5));
        assert_eq!(failure_budget(9, true, true), (5, 4));
        assert_eq!(failure_budget(4, true, false), (4, 0));
        assert_eq!(failure_budget(4, false, true), (0, 4));
    }

    #[test]
    fn pick_balanced_interleaves_query_and_read() {
        let q: Vec<FailureCase> = (0..3)
            .map(|i| FailureCase::Query {
                ex: QueryExample {
                    episode_id: format!("q{i}"),
                    question: format!("q{i}"),
                    search_query: String::new(),
                    gold: vec![],
                    hit: false,
                },
                model_search: None,
                top_k: String::new(),
            })
            .collect();
        let r: Vec<FailureCase> = (0..3)
            .map(|i| FailureCase::Read {
                ex: ReadExample {
                    episode_id: format!("r{i}"),
                    case_id: None,
                    question: format!("r{i}"),
                    search_query: String::new(),
                    hits: vec![],
                    gold: vec![],
                    model_read: None,
                    hit: false,
                },
                model_read: None,
                resolved: None,
            })
            .collect();
        let picked = pick_balanced_failures(q, r, 4);
        assert_eq!(picked.len(), 4);
        assert!(matches!(&picked[0], FailureCase::Query { .. }));
        assert!(matches!(&picked[1], FailureCase::Read { .. }));
        assert!(matches!(&picked[2], FailureCase::Query { .. }));
        assert!(matches!(&picked[3], FailureCase::Read { .. }));
    }

    #[test]
    fn normalize_question_used_for_join() {
        use crate::export_tz::normalize_question;
        assert_eq!(normalize_question("  a  b "), "a b");
    }
}
