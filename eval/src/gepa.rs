//! GEPA-style reflective optimization of the prod `answer_hotpot` system prompt.
//!
//! Quad sub-tasks scored deterministically:
//! - **search**: TZ `functions.search` → model emits `search(query)` → gold chunk in top-k?
//! - **grep**: TZ `functions.grep` → model emits `grep(pattern)` → gold chunk in grep hits?
//! - **glob**: TZ `functions.glob` → model emits `glob(pattern)` → gold document path listed?
//! - **read**: TZ `functions.read` → after prefilled search/grep hits, model emits `read(path,n)` → matches gold?

use anyhow::{Context, Result};
pub use crate::export_tz::SearchExample;
use crate::export_tz::{GlobExample, GrepExample, ReadExample, ReadPick};
use glossa::grep::GrepOpts;
use glossa::index::store::DocIndex;
use glossa::trace::TraceLog;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::time::{Duration, SystemTime};

/// Knobs for one optimization run.
pub struct GepaConfig {
    pub gateway: String,
    pub search_function: String,
    pub read_function: String,
    pub grep_function: String,
    pub glob_function: String,
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
    pub w_search: f64,
    pub w_grep: f64,
    pub w_glob: f64,
    pub w_read: f64,
    /// RNG seed for minibatch sampling (deterministic per run + iter offset).
    pub seed: u64,
    /// Max instances in D_pareto (sampled from val) for parent selection and pool scores.
    pub pareto_size: usize,
    pub candidate_selection: CandidateSelection,
}

/// How to pick the parent prompt each GEPA iteration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CandidateSelection {
    /// Frequency-weighted sampling from per-instance Pareto winners (paper Algorithm 2).
    #[default]
    Pareto,
    /// Always mutate the candidate with best combined score on D_pareto.
    CurrentBest,
}

impl fmt::Display for CandidateSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pareto => write!(f, "pareto"),
            Self::CurrentBest => write!(f, "current_best"),
        }
    }
}

impl FromStr for CandidateSelection {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().replace('-', "_").as_str() {
            "pareto" => Ok(Self::Pareto),
            "current_best" | "currentbest" | "best" => Ok(Self::CurrentBest),
            other => anyhow::bail!("unknown candidate_selection {other:?} (use pareto or current_best)"),
        }
    }
}

/// Stratified sample from val pools used as D_pareto.
struct ParetoSet {
    search: Vec<SearchExample>,
    grep: Vec<GrepExample>,
    glob: Vec<GlobExample>,
    read: Vec<ReadExample>,
}

impl ParetoSet {
    fn len(&self) -> usize {
        self.search.len() + self.grep.len() + self.glob.len() + self.read.len()
    }
}

pub struct GepaRunResult {
    pub prompt: String,
    pub baseline_acc: f64,
    pub best_acc: f64,
    pub search_acc: f64,
    pub grep_acc: f64,
    pub glob_acc: f64,
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

fn parse_n(v: &Value) -> Option<u64> {
    if let Some(n) = v.as_u64() {
        return Some(n);
    }
    if let Some(n) = v.as_i64().filter(|&i| i >= 0) {
        return Some(n as u64);
    }
    let digits: String = v.as_str()?.chars().filter(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
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

fn tool_calls(content: &[Value]) -> Vec<(String, Value)> {
    content
        .iter()
        .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_call"))
        .map(|b| {
            let name = b.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
            (name, tool_call_args(b))
        })
        .collect()
}

fn read_pick_from_args(args: &Value) -> Option<ReadPick> {
    let path_v = args.get("path")?;
    let n_v = args.get("n")?;
    if let (Some(path), Some(n)) = (path_v.as_str(), parse_n(n_v)) {
        return Some(ReadPick {
            path: path.to_string(),
            n,
        });
    }
    // Model sometimes swaps path and n (TZ schema rejects; raw_arguments may still carry this).
    if let (Some(n), Some(path)) = (
        parse_n(path_v),
        n_v.as_str().filter(|p| !p.is_empty()),
    ) {
        return Some(ReadPick {
            path: path.to_string(),
            n,
        });
    }
    None
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
        .and_then(|(_, args)| read_pick_from_args(&args))
}

fn first_grep_call(content: &[Value]) -> Option<String> {
    tool_calls(content)
        .into_iter()
        .find(|(n, _)| n == "grep")
        .and_then(|(_, args)| args.get("pattern").and_then(|p| p.as_str()).map(str::to_string))
}

fn first_glob_call(content: &[Value]) -> Option<String> {
    tool_calls(content)
        .into_iter()
        .find(|(n, _)| n == "glob")
        .and_then(|(_, args)| args.get("pattern").and_then(|p| p.as_str()).map(str::to_string))
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

fn grep_top_k(idx: &DocIndex, pattern: &str, k: usize) -> Vec<glossa::grep::GrepHit> {
    glossa::grep::grep(idx, pattern, &GrepOpts::default())
        .unwrap_or_default()
        .into_iter()
        .take(k)
        .collect()
}

fn glob_paths(idx: &DocIndex, pattern: &str) -> Vec<String> {
    let text = glossa::tools::glob(idx, pattern, &TraceLog::disabled());
    crate::export_tz::parse_glob_paths(&text)
        .into_iter()
        .filter_map(|p| idx.canonical_document_path(&p))
        .collect()
}

fn render_grep_hits(hits: &[glossa::grep::GrepHit]) -> String {
    hits.iter()
        .map(glossa::grep::GrepHit::display_line)
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_ranked_hits(hits: &[glossa::index::store::RankedHit]) -> String {
    hits.iter().map(glossa::index::store::RankedHit::display_line).collect::<Vec<_>>().join("\n")
}

fn search_hit_hits(hits: &[glossa::index::store::RankedHit], gold: &[(String, String)]) -> bool {
    for h in hits {
        for (gp, gl) in gold {
            if normalize_path(&h.path) == normalize_path(gp) && h.location == *gl {
                return true;
            }
        }
    }
    false
}

fn grep_hit_hits(hits: &[glossa::grep::GrepHit], gold: &[(String, String)], idx: &DocIndex) -> bool {
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

fn glob_hit_paths(paths: &[String], gold: &[(String, String)]) -> bool {
    for gp in gold.iter().map(|(p, _)| p) {
        for p in paths {
            if normalize_path(p) == normalize_path(gp) {
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
    if ex.prefill_source == "grep" {
        return vec![
            user_question_msg(&ex.question),
            json!({
                "role": "assistant",
                "content": [{
                    "type": "tool_call",
                    "id": "prefill-grep",
                    "name": "grep",
                    "arguments": {"pattern": ex.search_query}
                }]
            }),
            json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "id": "prefill-grep",
                    "name": "grep",
                    "result": ex.grep_result
                }]
            }),
        ];
    }
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

struct SearchOutcome {
    ok: bool,
    model_search: Option<String>,
    top_k: String,
}

struct ReadOutcome {
    ok: bool,
    model_read: Option<ReadPick>,
    resolved: Option<String>,
}

struct GrepOutcome {
    ok: bool,
    model_pattern: Option<String>,
    top_k: String,
}

struct GlobOutcome {
    ok: bool,
    model_pattern: Option<String>,
    listing: String,
}

fn score_search_one(cfg: &GepaConfig, prompt: &str, ex: &SearchExample, idx: &DocIndex) -> SearchOutcome {
    let messages = [user_question_msg(&ex.question)];
    let turn = match infer_prompt(cfg, &cfg.search_function, prompt, &messages) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("search inference failed: {e:#}");
            return SearchOutcome {
                ok: false,
                model_search: None,
                top_k: String::new(),
            };
        }
    };
    let Some(query) = first_search_call(&turn.content) else {
        return SearchOutcome {
            ok: false,
            model_search: None,
            top_k: String::new(),
        };
    };
    let hits = search_top_k(idx, &query, cfg.top_k);
    let gold = gold_pairs(&ex.gold);
    SearchOutcome {
        ok: search_hit_hits(&hits, &gold),
        model_search: Some(query),
        top_k: render_ranked_hits(&hits),
    }
}

fn score_grep_one(cfg: &GepaConfig, prompt: &str, ex: &GrepExample, idx: &DocIndex) -> GrepOutcome {
    let messages = [user_question_msg(&ex.question)];
    let turn = match infer_prompt(cfg, &cfg.grep_function, prompt, &messages) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("grep inference failed: {e:#}");
            return GrepOutcome {
                ok: false,
                model_pattern: None,
                top_k: String::new(),
            };
        }
    };
    let Some(pattern) = first_grep_call(&turn.content) else {
        return GrepOutcome {
            ok: false,
            model_pattern: None,
            top_k: String::new(),
        };
    };
    let hits = grep_top_k(idx, &pattern, cfg.top_k);
    let gold = gold_pairs(&ex.gold);
    GrepOutcome {
        ok: grep_hit_hits(&hits, &gold, idx),
        model_pattern: Some(pattern),
        top_k: render_grep_hits(&hits),
    }
}

fn score_glob_one(cfg: &GepaConfig, prompt: &str, ex: &GlobExample, idx: &DocIndex) -> GlobOutcome {
    let messages = [user_question_msg(&ex.question)];
    let turn = match infer_prompt(cfg, &cfg.glob_function, prompt, &messages) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("glob inference failed: {e:#}");
            return GlobOutcome {
                ok: false,
                model_pattern: None,
                listing: String::new(),
            };
        }
    };
    let Some(pattern) = first_glob_call(&turn.content) else {
        return GlobOutcome {
            ok: false,
            model_pattern: None,
            listing: String::new(),
        };
    };
    let paths = glob_paths(idx, &pattern);
    let gold = gold_pairs(&ex.gold);
    GlobOutcome {
        ok: glob_hit_paths(&paths, &gold),
        model_pattern: Some(pattern),
        listing: paths.join("\n"),
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

fn score_quad(
    cfg: &GepaConfig,
    prompt: &str,
    searches: &[SearchExample],
    greps: &[GrepExample],
    globs: &[GlobExample],
    reads: &[ReadExample],
    idx: &DocIndex,
) -> (Vec<SearchOutcome>, Vec<GrepOutcome>, Vec<GlobOutcome>, Vec<ReadOutcome>) {
    let s_scores = searches.iter().map(|ex| score_search_one(cfg, prompt, ex, idx)).collect();
    let g_scores = greps.iter().map(|ex| score_grep_one(cfg, prompt, ex, idx)).collect();
    let l_scores = globs.iter().map(|ex| score_glob_one(cfg, prompt, ex, idx)).collect();
    let r_scores = reads.iter().map(|ex| score_read_one(cfg, prompt, ex, idx)).collect();
    (s_scores, g_scores, l_scores, r_scores)
}


fn search_acc(scores: &[SearchOutcome]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    scores.iter().filter(|o| o.ok).count() as f64 / scores.len() as f64
}

fn grep_acc(g: &[GrepOutcome]) -> f64 {
    if g.is_empty() {
        return 0.0;
    }
    g.iter().filter(|o| o.ok).count() as f64 / g.len() as f64
}

fn glob_acc(l: &[GlobOutcome]) -> f64 {
    if l.is_empty() {
        return 0.0;
    }
    l.iter().filter(|o| o.ok).count() as f64 / l.len() as f64
}

fn read_acc(r: &[ReadOutcome]) -> f64 {
    if r.is_empty() {
        return 0.0;
    }
    r.iter().filter(|o| o.ok).count() as f64 / r.len() as f64
}

fn outcomes_to_bools(
    s: &[SearchOutcome],
    g: &[GrepOutcome],
    l: &[GlobOutcome],
    r: &[ReadOutcome],
) -> (Vec<bool>, Vec<bool>, Vec<bool>, Vec<bool>) {
    (
        s.iter().map(|o| o.ok).collect(),
        g.iter().map(|o| o.ok).collect(),
        l.iter().map(|o| o.ok).collect(),
        r.iter().map(|o| o.ok).collect(),
    )
}

fn acc(scores: &[bool]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    scores.iter().filter(|b| **b).count() as f64 / scores.len() as f64
}

fn weighted_quad_acc(s: f64, g: f64, l: f64, r: f64, ws: f64, wg: f64, wl: f64, wr: f64, if_zero: f64) -> f64 {
    let w = ws + wg + wl + wr;
    if w <= 0.0 {
        return if_zero;
    }
    (ws * s + wg * g + wl * l + wr * r) / w
}

fn combined_acc_from_pools(
    have_s: bool,
    have_g: bool,
    have_l: bool,
    have_r: bool,
    s: f64,
    g: f64,
    l: f64,
    r: f64,
    cfg: &GepaConfig,
) -> f64 {
    weighted_quad_acc(
        s,
        g,
        l,
        r,
        if have_s { cfg.w_search } else { 0.0 },
        if have_g { cfg.w_grep } else { 0.0 },
        if have_l { cfg.w_glob } else { 0.0 },
        if have_r { cfg.w_read } else { 0.0 },
        0.0,
    )
}

const REJECTED_HISTORY_CAP: usize = 5;
const MINIBATCH_RESAMPLE_ATTEMPTS: usize = 3;

#[derive(Clone)]
struct Minibatch {
    s: Vec<SearchExample>,
    g: Vec<GrepExample>,
    l: Vec<GlobExample>,
    r: Vec<ReadExample>,
}

#[derive(Clone)]
enum MinibatchTrace {
    Search {
        ok: bool,
        ex: SearchExample,
        model_search: Option<String>,
        top_k: String,
    },
    Grep {
        ok: bool,
        ex: GrepExample,
        model_pattern: Option<String>,
        top_k: String,
    },
    Glob {
        ok: bool,
        ex: GlobExample,
        model_pattern: Option<String>,
        listing: String,
    },
    Read {
        ok: bool,
        ex: ReadExample,
        model_read: Option<ReadPick>,
        resolved: Option<String>,
    },
}

impl MinibatchTrace {
    fn ok(&self) -> bool {
        match self {
            MinibatchTrace::Search { ok, .. }
            | MinibatchTrace::Grep { ok, .. }
            | MinibatchTrace::Glob { ok, .. }
            | MinibatchTrace::Read { ok, .. } => *ok,
        }
    }
}

struct ReflectContext {
    parent_prompt: String,
    parent_search_acc: f64,
    parent_grep_acc: f64,
    parent_glob_acc: f64,
    parent_read_acc: f64,
    parent_combined: f64,
    traces: Vec<MinibatchTrace>,
    rejected: Vec<RejectedMutation>,
}

#[derive(Clone)]
struct RejectedMutation {
    iter: usize,
    reason: &'static str,
    detail: String,
    prompt_preview: Option<String>,
}

/// Split minibatch evenly across non-empty train pools.
fn failure_budget_equal(minibatch: usize, have: [bool; 4]) -> [usize; 4] {
    let n_active = have.iter().filter(|&&h| h).count();
    if minibatch == 0 || n_active == 0 {
        return [0; 4];
    }
    let base = minibatch / n_active;
    let mut extra = minibatch % n_active;
    let mut out = [0usize; 4];
    for (i, h) in have.iter().enumerate() {
        if *h {
            out[i] = base + if extra > 0 { extra -= 1; 1 } else { 0 };
        }
    }
    out
}

/// Split minibatch across active pools proportional to GEPA weights (floor=1 each when possible).
fn failure_budget_weighted(minibatch: usize, have: [bool; 4], weights: [f64; 4]) -> [usize; 4] {
    let n_active = have.iter().filter(|&&h| h).count();
    if minibatch == 0 || n_active == 0 {
        return [0; 4];
    }
    if minibatch < n_active {
        return failure_budget_equal(minibatch, have);
    }

    let mut out = [0usize; 4];
    let mut w_active = 0.0;
    for i in 0..4 {
        if have[i] {
            out[i] = 1;
            w_active += weights[i].max(0.0);
        }
    }
    let remaining = minibatch - n_active;
    if remaining == 0 {
        return out;
    }
    if w_active <= 0.0 {
        return failure_budget_equal(minibatch, have);
    }

    let mut fracs: Vec<(usize, f64)> = Vec::new();
    let mut assigned = 0usize;
    for i in 0..4 {
        if !have[i] {
            continue;
        }
        let share = remaining as f64 * weights[i].max(0.0) / w_active;
        let base = share.floor() as usize;
        out[i] += base;
        assigned += base;
        fracs.push((i, share - base as f64));
    }
    let mut leftover = remaining.saturating_sub(assigned);
    fracs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (i, _) in fracs {
        if leftover == 0 {
            break;
        }
        out[i] += 1;
        leftover -= 1;
    }
    debug_assert_eq!(out.iter().sum::<usize>(), minibatch);
    out
}

#[cfg(test)]
fn failure_budget_legacy(minibatch: usize, have_search: bool, have_read: bool) -> (usize, usize) {
    let b = failure_budget_equal(minibatch, [have_search, false, false, have_read]);
    (b[0], b[3])
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

fn sample_minibatch(
    cfg: &GepaConfig,
    train_s: &[SearchExample],
    train_g: &[GrepExample],
    train_l: &[GlobExample],
    train_r: &[ReadExample],
    rng: &mut StdRng,
) -> Minibatch {
    let have = [
        !train_s.is_empty(),
        !train_g.is_empty(),
        !train_l.is_empty(),
        !train_r.is_empty(),
    ];
    let want = failure_budget_weighted(
        cfg.minibatch,
        have,
        [cfg.w_search, cfg.w_grep, cfg.w_glob, cfg.w_read],
    );
    Minibatch {
        s: sample_indices(train_s.len(), want[0], rng)
            .into_iter()
            .map(|i| train_s[i].clone())
            .collect(),
        g: sample_indices(train_g.len(), want[1], rng)
            .into_iter()
            .map(|i| train_g[i].clone())
            .collect(),
        l: sample_indices(train_l.len(), want[2], rng)
            .into_iter()
            .map(|i| train_l[i].clone())
            .collect(),
        r: sample_indices(train_r.len(), want[3], rng)
            .into_iter()
            .map(|i| train_r[i].clone())
            .collect(),
    }
}

fn sample_pareto_set(
    val_s: &[SearchExample],
    val_g: &[GrepExample],
    val_l: &[GlobExample],
    val_r: &[ReadExample],
    pareto_size: usize,
    weights: [f64; 4],
    rng: &mut StdRng,
) -> ParetoSet {
    let have = [
        !val_s.is_empty(),
        !val_g.is_empty(),
        !val_l.is_empty(),
        !val_r.is_empty(),
    ];
    let total_avail = val_s.len() + val_g.len() + val_l.len() + val_r.len();
    if total_avail == 0 || pareto_size == 0 {
        return ParetoSet {
            search: Vec::new(),
            grep: Vec::new(),
            glob: Vec::new(),
            read: Vec::new(),
        };
    }
    let cap = pareto_size.min(total_avail);
    let mut want = failure_budget_weighted(cap, have, weights);
    want[0] = want[0].min(val_s.len());
    want[1] = want[1].min(val_g.len());
    want[2] = want[2].min(val_l.len());
    want[3] = want[3].min(val_r.len());
    ParetoSet {
        search: sample_indices(val_s.len(), want[0], rng)
            .into_iter()
            .map(|i| val_s[i].clone())
            .collect(),
        grep: sample_indices(val_g.len(), want[1], rng)
            .into_iter()
            .map(|i| val_g[i].clone())
            .collect(),
        glob: sample_indices(val_l.len(), want[2], rng)
            .into_iter()
            .map(|i| val_l[i].clone())
            .collect(),
        read: sample_indices(val_r.len(), want[3], rng)
            .into_iter()
            .map(|i| val_r[i].clone())
            .collect(),
    }
}

fn score_minibatch_traces(
    cfg: &GepaConfig,
    prompt: &str,
    mb: &Minibatch,
    idx: &DocIndex,
) -> (Vec<MinibatchTrace>, f64, f64, f64, f64, f64) {
    let (s_out, g_out, l_out, r_out) =
        score_quad(cfg, prompt, &mb.s, &mb.g, &mb.l, &mb.r, idx);
    let ps = search_acc(&s_out);
    let pg = grep_acc(&g_out);
    let pl = glob_acc(&l_out);
    let pr = read_acc(&r_out);
    let pc = combined_acc_from_pools(
        !mb.s.is_empty(),
        !mb.g.is_empty(),
        !mb.l.is_empty(),
        !mb.r.is_empty(),
        ps,
        pg,
        pl,
        pr,
        cfg,
    );
    let mut traces = Vec::with_capacity(s_out.len() + g_out.len() + l_out.len() + r_out.len());
    for (ex, o) in mb.s.iter().zip(s_out) {
        traces.push(MinibatchTrace::Search {
            ok: o.ok,
            ex: ex.clone(),
            model_search: o.model_search,
            top_k: o.top_k,
        });
    }
    for (ex, o) in mb.g.iter().zip(g_out) {
        traces.push(MinibatchTrace::Grep {
            ok: o.ok,
            ex: ex.clone(),
            model_pattern: o.model_pattern,
            top_k: o.top_k,
        });
    }
    for (ex, o) in mb.l.iter().zip(l_out) {
        traces.push(MinibatchTrace::Glob {
            ok: o.ok,
            ex: ex.clone(),
            model_pattern: o.model_pattern,
            listing: o.listing,
        });
    }
    for (ex, o) in mb.r.iter().zip(r_out) {
        traces.push(MinibatchTrace::Read {
            ok: o.ok,
            ex: ex.clone(),
            model_read: o.model_read,
            resolved: o.resolved,
        });
    }
    (traces, ps, pg, pl, pr, pc)
}

fn combined_from_outcomes(
    s: &[SearchOutcome],
    g: &[GrepOutcome],
    l: &[GlobOutcome],
    r: &[ReadOutcome],
    cfg: &GepaConfig,
) -> f64 {
    combined_acc_from_pools(
        !s.is_empty(),
        !g.is_empty(),
        !l.is_empty(),
        !r.is_empty(),
        search_acc(s),
        grep_acc(g),
        glob_acc(l),
        read_acc(r),
        cfg,
    )
}

#[cfg(test)]
fn child_beats_parent_on_minibatch(
    parent_s: &[SearchOutcome],
    parent_g: &[GrepOutcome],
    parent_l: &[GlobOutcome],
    parent_r: &[ReadOutcome],
    child_s: &[SearchOutcome],
    child_g: &[GrepOutcome],
    child_l: &[GlobOutcome],
    child_r: &[ReadOutcome],
    cfg: &GepaConfig,
) -> bool {
    let p = combined_from_outcomes(parent_s, parent_g, parent_l, parent_r, cfg);
    let c = combined_from_outcomes(child_s, child_g, child_l, child_r, cfg);
    c > p
}

fn push_rejected(rejected: &mut Vec<RejectedMutation>, item: RejectedMutation) {
    rejected.push(item);
    if rejected.len() > REJECTED_HISTORY_CAP {
        rejected.remove(0);
    }
}

fn format_rejected_section(rejected: &[RejectedMutation]) -> String {
    if rejected.is_empty() {
        return String::new();
    }
    let mut s = String::from("=== REJECTED MUTATIONS (do not repeat) ===\n");
    for r in rejected {
        s.push_str(&format!("iter {}: {} — {}", r.iter, r.reason, r.detail));
        if let Some(p) = &r.prompt_preview {
            s.push_str(&format!("; preview: {p:?}"));
        }
        s.push('\n');
    }
    s.push('\n');
    s
}

fn format_trace_case(cfg: &GepaConfig, i: usize, trace: &MinibatchTrace) -> String {
    let status = if trace.ok() { "OK" } else { "FAIL" };
    match trace {
        MinibatchTrace::Search {
            ex,
            model_search,
            top_k,
            ..
        } => {
            let detail = if trace.ok() {
                format!(
                    "Gold chunk was in top-{} for the model's search query.\n{}",
                    cfg.top_k,
                    format_model_search(model_search.as_deref(), top_k, cfg.top_k),
                )
            } else {
                format!(
                    "Gold was not in top-{} for the model's search query.\n{}",
                    cfg.top_k,
                    format_model_search(model_search.as_deref(), top_k, cfg.top_k),
                )
            };
            format!(
                "--- Search trace {} ({status}) ---\nQuestion: {}\nGold chunks: {}\n{detail}\n\n",
                i + 1,
                ex.question,
                ex.gold.join(", "),
            )
        }
        MinibatchTrace::Grep {
            ex,
            model_pattern,
            top_k,
            ..
        } => {
            let detail = if trace.ok() {
                format!(
                    "Gold chunk was in top-{} grep hits.\n{}",
                    cfg.top_k,
                    format_model_grep(model_pattern.as_deref(), top_k, cfg.top_k),
                )
            } else {
                format!(
                    "Gold was not in top-{} grep hits.\n{}",
                    cfg.top_k,
                    format_model_grep(model_pattern.as_deref(), top_k, cfg.top_k),
                )
            };
            format!(
                "--- Grep trace {} ({status}) ---\nQuestion: {}\nGold chunks: {}\n{detail}\n\n",
                i + 1,
                ex.question,
                ex.gold.join(", "),
            )
        }
        MinibatchTrace::Glob {
            ex,
            model_pattern,
            listing,
            ..
        } => {
            let detail = if trace.ok() {
                "Gold document path appeared in the glob listing."
            } else {
                "Gold document path was not in the glob listing."
            };
            format!(
                "--- Glob trace {} ({status}) ---\nQuestion: {}\nGold chunks: {}\n{}\n{detail}\n\n",
                i + 1,
                ex.question,
                ex.gold.join(", "),
                format_model_glob(model_pattern.as_deref(), listing),
            )
        }
        MinibatchTrace::Read {
            ex,
            model_read,
            resolved,
            ..
        } => {
            let outcome = if trace.ok() {
                "The model's read matched gold."
            } else {
                "The model's read did not match gold."
            };
            let prefill_body = if ex.prefill_source == "grep" {
                ex.grep_result.clone()
            } else {
                render_hits(&ex.hits)
            };
            let prefill_label = if ex.prefill_source == "grep" {
                "grep pattern"
            } else {
                "search query"
            };
            format!(
                "--- Read trace {} ({status}) ---\nQuestion: {}\nPrefilled from: {}\n\
                 Prefill context ({}): {}\nGold chunks: {}\n{}\n{outcome}\n\n",
                i + 1,
                ex.question,
                ex.prefill_source,
                prefill_label,
                prefill_body,
                ex.gold.join(", "),
                format_model_read(model_read.as_ref(), resolved.as_deref()),
            )
        }
    }
}

fn build_reflect_instruction(ctx: &ReflectContext, cfg: &GepaConfig) -> String {
    let mut cases = String::new();
    for (i, t) in ctx.traces.iter().enumerate() {
        cases.push_str(&format_trace_case(cfg, i, t));
    }
    let rejected = format_rejected_section(&ctx.rejected);
    format!(
        "You are improving the SYSTEM PROMPT for a knowledge-base agent that uses search, grep, glob, and read tools.\n\
         Traces fall into four types:\n\
         1) SEARCH — search(query) should retrieve the gold chunk in top results.\n\
         2) GREP — grep(pattern) should find the gold chunk for exact tokens/codes.\n\
         3) GLOB — glob(pattern) should list the gold document path.\n\
         4) READ — given prefilled search or grep results, read(path,n) should open the gold chunk.\n\
         Diagnose recurring mistakes and rewrite the system prompt so all retrieval stages improve.\n\
         Do NOT break patterns that already work (OK traces). Fix FAIL traces.\n\
         Keep it concise and general (do NOT mention specific cases). Preserve tool-use instructions.\n\
         Reply with ONLY the new system prompt text — no preamble, no quotes.\n\n\
         === PARENT SCORES ON MINIBATCH ===\n\
         search={parent_s:.3} grep={parent_g:.3} glob={parent_l:.3} read={parent_r:.3} combined={parent_c:.3}\n\n\
         {rejected}\
         === CURRENT SYSTEM PROMPT ===\n{prompt}\n\n=== TRACES ===\n{cases}=== NEW SYSTEM PROMPT ===",
        parent_s = ctx.parent_search_acc,
        parent_g = ctx.parent_grep_acc,
        parent_l = ctx.parent_glob_acc,
        parent_r = ctx.parent_read_acc,
        parent_c = ctx.parent_combined,
        rejected = rejected,
        prompt = ctx.parent_prompt,
        cases = cases,
    )
}

fn output_likely_truncated(_text: &str, finish_reason: Option<&str>) -> bool {
    finish_reason.is_some_and(|r| {
        r.eq_ignore_ascii_case("length") || r.eq_ignore_ascii_case("max_tokens")
    })
}

fn reflect(cfg: &GepaConfig, ctx: &ReflectContext) -> Result<String> {
    let instruction = build_reflect_instruction(ctx, cfg);
    let est_tokens = instruction.len() / 4;
    println!(
        "reflect payload: {} chars (~{est_tokens} tok est)",
        instruction.len(),
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
    if output_likely_truncated(&out, turn.finish_reason.as_deref()) {
        anyhow::bail!(
            "gepa_reflect output truncated (finish_reason={:?})",
            turn.finish_reason,
        );
    }
    Ok(out)
}

/// Deterministic seed from a run tag string.
pub fn hash_run_seed(run: &str) -> u64 {
    let mut h = DefaultHasher::new();
    run.hash(&mut h);
    h.finish()
}

struct Candidate {
    prompt: String,
    /// Per-instance hit/miss on D_pareto (not full val).
    s_val: Vec<bool>,
    g_val: Vec<bool>,
    l_val: Vec<bool>,
    r_val: Vec<bool>,
}

fn validate_dataset(
    searches: &[SearchExample],
    greps: &[GrepExample],
    globs: &[GlobExample],
    reads: &[ReadExample],
) {
    let mut episodes: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for s in searches {
        episodes.insert(&s.episode_id);
    }
    for g in greps {
        episodes.insert(&g.episode_id);
    }
    for g in globs {
        episodes.insert(&g.episode_id);
    }
    for r in reads {
        episodes.insert(&r.episode_id);
    }
    if searches.is_empty() {
        eprintln!("gepa warn: no search examples");
    }
    if greps.is_empty() {
        eprintln!("gepa warn: no grep examples");
    }
    if globs.is_empty() {
        eprintln!("gepa warn: no glob examples");
    }
    if reads.is_empty() {
        eprintln!("gepa warn: no read examples");
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
            format!("Model output: search({q:?})\nTop-{k} for that search:\n{results}")
        }
        None => "Model output: no search(query) tool call".to_string(),
    }
}

fn format_model_grep(model_pattern: Option<&str>, top_k: &str, k: usize) -> String {
    match model_pattern {
        Some(p) => {
            let results = if top_k.is_empty() {
                "(empty)".to_string()
            } else {
                top_k.to_string()
            };
            format!("Model output: grep({p:?})\nTop-{k} for that pattern:\n{results}")
        }
        None => "Model output: no grep(pattern) tool call".to_string(),
    }
}

fn format_model_glob(model_pattern: Option<&str>, listing: &str) -> String {
    match model_pattern {
        Some(p) => {
            let body = if listing.is_empty() {
                "(empty)".to_string()
            } else {
                listing.to_string()
            };
            format!("Model output: glob({p:?})\nListing:\n{body}")
        }
        None => "Model output: no glob(pattern) tool call".to_string(),
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

fn dominates(a_q: &[bool], a_g: &[bool], a_l: &[bool], a_r: &[bool], b_q: &[bool], b_g: &[bool], b_l: &[bool], b_r: &[bool]) -> bool {
    let a: Vec<bool> = a_q.iter().chain(a_g).chain(a_l).chain(a_r).copied().collect();
    let b: Vec<bool> = b_q.iter().chain(b_g).chain(b_l).chain(b_r).copied().collect();
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
                j != i
                    && dominates(
                        &c.s_val,
                        &c.g_val,
                        &c.l_val,
                        &c.r_val,
                        &pool[i].s_val,
                        &pool[i].g_val,
                        &pool[i].l_val,
                        &pool[i].r_val,
                    )
            })
        })
        .collect()
}

fn candidate_pareto_bits(c: &Candidate) -> Vec<bool> {
    c.s_val
        .iter()
        .chain(&c.g_val)
        .chain(&c.l_val)
        .chain(&c.r_val)
        .copied()
        .collect()
}

fn select_parent_idx(pool: &[Candidate], cfg: &GepaConfig, rng: &mut StdRng) -> usize {
    if pool.len() == 1 {
        return 0;
    }
    match cfg.candidate_selection {
        CandidateSelection::CurrentBest => pool
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                candidate_combined(a, cfg)
                    .partial_cmp(&candidate_combined(b, cfg))
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0),
        CandidateSelection::Pareto => select_parent_pareto_weighted(pool, rng),
    }
}

fn pareto_frontier_win_counts(pool: &[Candidate]) -> (Vec<usize>, Vec<usize>) {
    let bits: Vec<Vec<bool>> = pool.iter().map(candidate_pareto_bits).collect();
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

    let mut c_vec: Vec<usize> = union.into_iter().collect();
    c_vec.sort_unstable();
    let mut dominated = HashSet::new();
    for &i in &c_vec {
        for &j in &c_vec {
            if i != j {
                let a = &pool[j];
                let b = &pool[i];
                if dominates(
                    &a.s_val,
                    &a.g_val,
                    &a.l_val,
                    &a.r_val,
                    &b.s_val,
                    &b.g_val,
                    &b.l_val,
                    &b.r_val,
                ) {
                    dominated.insert(i);
                    break;
                }
            }
        }
    }
    let mut frontier_idxs: Vec<usize> = c_vec.into_iter().filter(|k| !dominated.contains(k)).collect();
    if frontier_idxs.is_empty() {
        frontier_idxs = frontier(pool);
    }
    frontier_idxs.sort_unstable();

    let mut counts = vec![0usize; pool.len()];
    for winners in &per_instance_winners {
        let hat: Vec<usize> = winners
            .iter()
            .copied()
            .filter(|k| frontier_idxs.contains(k))
            .collect();
        let active = if hat.is_empty() { winners.clone() } else { hat };
        for k in active {
            counts[k] += 1;
        }
    }

    (frontier_idxs, counts)
}

fn select_parent_pareto_weighted(pool: &[Candidate], rng: &mut StdRng) -> usize {
    let bits: Vec<Vec<bool>> = pool.iter().map(candidate_pareto_bits).collect();
    let n_inst = bits.first().map(|b| b.len()).unwrap_or(0);
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
    let mut acc = 0;
    for &k in &frontier_idxs {
        acc += counts[k];
        if pick < acc {
            return k;
        }
    }
    *frontier_idxs.last().unwrap_or(&0)
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

fn post_gepa_metrics(
    gw: &str,
    episode_id: &str,
    prefix: &str,
    search_acc: f64,
    grep_acc: f64,
    glob_acc: f64,
    read_acc: f64,
    combined: f64,
    tags: &Value,
) {
    crate::tz::post_feedback(gw, episode_id, &format!("{prefix}_search"), json!(search_acc), tags);
    crate::tz::post_feedback(gw, episode_id, &format!("{prefix}_grep"), json!(grep_acc), tags);
    crate::tz::post_feedback(gw, episode_id, &format!("{prefix}_glob"), json!(glob_acc), tags);
    crate::tz::post_feedback(gw, episode_id, &format!("{prefix}_read"), json!(read_acc), tags);
    crate::tz::post_feedback(gw, episode_id, &format!("{prefix}_combined"), json!(combined), tags);
}

fn post_baseline_feedback(
    cfg: &GepaConfig,
    search_acc: f64,
    grep_acc: f64,
    glob_acc: f64,
    read_acc: f64,
    combined: f64,
) {
    let tags = feedback_tags(cfg, "baseline");
    post_gepa_metrics(
        &cfg.gateway,
        &cfg.episode_id,
        "gepa_baseline",
        search_acc,
        grep_acc,
        glob_acc,
        read_acc,
        combined,
        &tags,
    );
}

fn post_iter_feedback(
    cfg: &GepaConfig,
    iter: usize,
    search_acc: f64,
    grep_acc: f64,
    glob_acc: f64,
    read_acc: f64,
    combined: f64,
    candidates: usize,
) {
    let stage = format!("iter_{iter}");
    let tags = feedback_tags(cfg, &stage);
    post_gepa_metrics(
        &cfg.gateway,
        &cfg.episode_id,
        "gepa_iter",
        search_acc,
        grep_acc,
        glob_acc,
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
        "TZ iter {iter}: search={search_acc:.3} grep={grep_acc:.3} glob={glob_acc:.3} read={read_acc:.3} combined={combined:.3} candidates={candidates}"
    );
}

fn post_final_feedback(
    cfg: &GepaConfig,
    result: &GepaRunResult,
    n_train: usize,
    n_val: usize,
) {
    let ep = &result.episode_id;
    let tags = feedback_tags(cfg, "final");
    post_gepa_metrics(
        &cfg.gateway,
        ep,
        "gepa_final",
        result.search_acc,
        result.grep_acc,
        result.glob_acc,
        result.read_acc,
        result.best_acc,
        &tags,
    );
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
        json!(n_train as f64),
        &tags,
    );
    crate::tz::post_feedback(
        &cfg.gateway,
        ep,
        "gepa_examples_val",
        json!(n_val as f64),
        &tags,
    );
    println!(
        "TZ final: episode={ep} search={:.3} grep={:.3} glob={:.3} read={:.3} combined={:.3} (baseline was {:.3})",
        result.search_acc,
        result.grep_acc,
        result.glob_acc,
        result.read_acc,
        result.best_acc,
        result.baseline_acc,
    );
}

fn candidate_combined(c: &Candidate, cfg: &GepaConfig) -> f64 {
    combined_acc_from_pools(
        !c.s_val.is_empty(),
        !c.g_val.is_empty(),
        !c.l_val.is_empty(),
        !c.r_val.is_empty(),
        acc(&c.s_val),
        acc(&c.g_val),
        acc(&c.l_val),
        acc(&c.r_val),
        cfg,
    )
}

fn best_pool_combined(pool: &[Candidate], cfg: &GepaConfig) -> (f64, f64, f64, f64, f64) {
    let best = pool
        .iter()
        .max_by(|a, b| candidate_combined(a, &cfg).partial_cmp(&candidate_combined(b, &cfg)).unwrap())
        .expect("non-empty pool");
    (
        acc(&best.s_val),
        acc(&best.g_val),
        acc(&best.l_val),
        acc(&best.r_val),
        candidate_combined(best, cfg),
    )
}

#[cfg(test)]
fn outcomes_from_traces(
    traces: &[MinibatchTrace],
) -> (Vec<SearchOutcome>, Vec<GrepOutcome>, Vec<GlobOutcome>, Vec<ReadOutcome>) {
    let mut s = Vec::new();
    let mut g = Vec::new();
    let mut l = Vec::new();
    let mut r = Vec::new();
    for t in traces {
        match t {
            MinibatchTrace::Search {
                ok,
                model_search,
                top_k,
                ..
            } => s.push(SearchOutcome {
                ok: *ok,
                model_search: model_search.clone(),
                top_k: top_k.clone(),
            }),
            MinibatchTrace::Grep {
                ok,
                model_pattern,
                top_k,
                ..
            } => g.push(GrepOutcome {
                ok: *ok,
                model_pattern: model_pattern.clone(),
                top_k: top_k.clone(),
            }),
            MinibatchTrace::Glob {
                ok,
                model_pattern,
                listing,
                ..
            } => l.push(GlobOutcome {
                ok: *ok,
                model_pattern: model_pattern.clone(),
                listing: listing.clone(),
            }),
            MinibatchTrace::Read {
                ok,
                model_read,
                resolved,
                ..
            } => r.push(ReadOutcome {
                ok: *ok,
                model_read: model_read.clone(),
                resolved: resolved.clone(),
            }),
        }
    }
    (s, g, l, r)
}

#[cfg(test)]
fn passes_minibatch_gate(child_mb: f64, parent_mb: f64) -> bool {
    child_mb > parent_mb
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

pub fn run(
    cfg: GepaConfig,
    searches: Vec<SearchExample>,
    greps: Vec<GrepExample>,
    globs: Vec<GlobExample>,
    reads: Vec<ReadExample>,
) -> Result<GepaRunResult> {
    anyhow::ensure!(
        !searches.is_empty() || !greps.is_empty() || !globs.is_empty() || !reads.is_empty(),
        "no search/grep/glob/read examples to optimize against"
    );
    let idx = DocIndex::open_or_create(&cfg.work).context("open index for GEPA scoring")?;
    validate_dataset(&searches, &greps, &globs, &reads);
    crate::tz::ensure_function(&cfg.gateway, &cfg.search_function, Some(&cfg.variant))
        .context("GEPA search function")?;
    crate::tz::ensure_function(&cfg.gateway, &cfg.read_function, Some(&cfg.variant))
        .context("GEPA read function")?;
    crate::tz::ensure_function(&cfg.gateway, &cfg.grep_function, Some(&cfg.variant))
        .context("GEPA grep function")?;
    crate::tz::ensure_function(&cfg.gateway, &cfg.glob_function, Some(&cfg.variant))
        .context("GEPA glob function")?;

    let (train_s, val_s) = split_by_episode(&searches, |s| &s.episode_id, cfg.val_frac);
    let (train_g, val_g) = split_by_episode(&greps, |g| &g.episode_id, cfg.val_frac);
    let (train_l, val_l) = split_by_episode(&globs, |g| &g.episode_id, cfg.val_frac);
    let (train_r, val_r) = split_by_episode(&reads, |r| &r.episode_id, cfg.val_frac);

    println!(
        "gepa: search {} ({} train, {} val), grep {} ({} train, {} val), glob {} ({} train, {} val), read {} ({} train, {} val), budget={}, work={}",
        searches.len(),
        train_s.len(),
        val_s.len(),
        greps.len(),
        train_g.len(),
        val_g.len(),
        globs.len(),
        train_l.len(),
        val_l.len(),
        reads.len(),
        train_r.len(),
        val_r.len(),
        cfg.budget,
        cfg.work.display(),
    );

    let have = [
        !train_s.is_empty(),
        !train_g.is_empty(),
        !train_l.is_empty(),
        !train_r.is_empty(),
    ];
    let slots = failure_budget_weighted(
        cfg.minibatch,
        have,
        [cfg.w_search, cfg.w_grep, cfg.w_glob, cfg.w_read],
    );
    if have.iter().any(|&h| h) {
        println!(
            "minibatch slots (weighted): search={} grep={} glob={} read={} (total {})",
            slots[0],
            slots[1],
            slots[2],
            slots[3],
            slots.iter().sum::<usize>(),
        );
    }

    let (base_s, base_g, base_l, base_r) =
        score_quad(&cfg, &cfg.seed_prompt, &val_s, &val_g, &val_l, &val_r, &idx);
    let baseline_acc = combined_from_outcomes(&base_s, &base_g, &base_l, &base_r, &cfg);
    let base_s_acc = search_acc(&base_s);
    let base_g_acc = grep_acc(&base_g);
    let base_l_acc = glob_acc(&base_l);
    let base_r_acc = read_acc(&base_r);
    println!(
        "baseline val: search={:.3} grep={:.3} glob={:.3} read={:.3} combined={:.3}",
        base_s_acc, base_g_acc, base_l_acc, base_r_acc, baseline_acc,
    );
    post_baseline_feedback(&cfg, base_s_acc, base_g_acc, base_l_acc, base_r_acc, baseline_acc);
    {
        let tags = feedback_tags(&cfg, "start");
        let n_train = train_s.len() + train_g.len() + train_l.len() + train_r.len();
        let n_val = val_s.len() + val_g.len() + val_l.len() + val_r.len();
        crate::tz::post_feedback(
            &cfg.gateway,
            &cfg.episode_id,
            "gepa_examples_train",
            json!(n_train as f64),
            &tags,
        );
        crate::tz::post_feedback(
            &cfg.gateway,
            &cfg.episode_id,
            "gepa_examples_val",
            json!(n_val as f64),
            &tags,
        );
    }

    let mut rejected: Vec<RejectedMutation> = Vec::new();
    let mut rng = StdRng::seed_from_u64(cfg.seed);

    let pareto_set = sample_pareto_set(
        &val_s,
        &val_g,
        &val_l,
        &val_r,
        cfg.pareto_size,
        [cfg.w_search, cfg.w_grep, cfg.w_glob, cfg.w_read],
        &mut rng,
    );
    println!(
        "pareto set (D_pareto): search={} grep={} glob={} read={} (total {}) selection={}",
        pareto_set.search.len(),
        pareto_set.grep.len(),
        pareto_set.glob.len(),
        pareto_set.read.len(),
        pareto_set.len(),
        cfg.candidate_selection,
    );

    let (base_p_s, base_p_g, base_p_l, base_p_r) = score_quad(
        &cfg,
        &cfg.seed_prompt,
        &pareto_set.search,
        &pareto_set.grep,
        &pareto_set.glob,
        &pareto_set.read,
        &idx,
    );
    let (base_p_s_bools, base_p_g_bools, base_p_l_bools, base_p_r_bools) =
        outcomes_to_bools(&base_p_s, &base_p_g, &base_p_l, &base_p_r);

    let mut pool = vec![Candidate {
        prompt: cfg.seed_prompt.clone(),
        s_val: base_p_s_bools,
        g_val: base_p_g_bools,
        l_val: base_p_l_bools,
        r_val: base_p_r_bools,
    }];

    for it in 0..cfg.budget {
        let parent_idx = select_parent_idx(&pool, &cfg, &mut rng);
        let parent_prompt = pool[parent_idx].prompt.clone();

        let mut mb = None;
        let mut traces = None;
        let mut parent_mb_scores = None;
        for _ in 0..MINIBATCH_RESAMPLE_ATTEMPTS {
            let batch = sample_minibatch(&cfg, &train_s, &train_g, &train_l, &train_r, &mut rng);
            let (t, ps, pg, pl, pr, pc) = score_minibatch_traces(&cfg, &parent_prompt, &batch, &idx);
            let n_fail = t.iter().filter(|x| !x.ok()).count();
            if n_fail > 0 {
                mb = Some(batch);
                traces = Some(t);
                parent_mb_scores = Some((ps, pg, pl, pr, pc));
                break;
            }
        }

        let Some(mb) = mb else {
            println!("[iter {it}] no failures in sampled minibatch — skip");
            continue;
        };
        let traces = traces.expect("traces with mb");
        let (parent_mb_s, parent_mb_g, parent_mb_l, parent_mb_r, parent_mb_c) =
            parent_mb_scores.expect("scores with mb");
        let n_s = traces
            .iter()
            .filter(|t| matches!(t, MinibatchTrace::Search { .. }))
            .count();
        let n_g = traces
            .iter()
            .filter(|t| matches!(t, MinibatchTrace::Grep { .. }))
            .count();
        let n_l = traces
            .iter()
            .filter(|t| matches!(t, MinibatchTrace::Glob { .. }))
            .count();
        let n_r = traces.len() - n_s - n_g - n_l;
        let n_fail = traces.iter().filter(|t| !t.ok()).count();
        println!(
            "[iter {it}] reflect minibatch: {} traces (search={n_s} grep={n_g} glob={n_l} read={n_r} fails={n_fail}) parent_mb={parent_mb_c:.3}",
            traces.len(),
        );

        let ctx = ReflectContext {
            parent_prompt: parent_prompt.clone(),
            parent_search_acc: parent_mb_s,
            parent_grep_acc: parent_mb_g,
            parent_glob_acc: parent_mb_l,
            parent_read_acc: parent_mb_r,
            parent_combined: parent_mb_c,
            traces: traces.clone(),
            rejected: rejected.clone(),
        };

        let child_prompt = match reflect(&cfg, &ctx) {
            Ok(p) => p,
            Err(e) => {
                println!("[iter {it}] reflection failed: {e:#}");
                push_rejected(
                    &mut rejected,
                    RejectedMutation {
                        iter: it,
                        reason: "reflect_failed",
                        detail: format!("{e:#}"),
                        prompt_preview: None,
                    },
                );
                continue;
            }
        };

        let (child_s_out, child_g_out, child_l_out, child_r_out) =
            score_quad(&cfg, &child_prompt, &mb.s, &mb.g, &mb.l, &mb.r, &idx);
        let child_mb_c = combined_from_outcomes(
            &child_s_out,
            &child_g_out,
            &child_l_out,
            &child_r_out,
            &cfg,
        );
        if child_mb_c <= parent_mb_c {
            println!(
                "[iter {it}] child_mb {child_mb_c:.3} <= parent_mb {parent_mb_c:.3} — discarded (minibatch_regressed)"
            );
            push_rejected(
                &mut rejected,
                RejectedMutation {
                    iter: it,
                    reason: "minibatch_regressed",
                    detail: format!("parent_mb={parent_mb_c:.3} child_mb={child_mb_c:.3}"),
                    prompt_preview: Some(child_prompt.clone()),
                },
            );
            continue;
        }

        let (child_p_s, child_p_g, child_p_l, child_p_r) = score_quad(
            &cfg,
            &child_prompt,
            &pareto_set.search,
            &pareto_set.grep,
            &pareto_set.glob,
            &pareto_set.read,
            &idx,
        );
        let (child_s_bools, child_g_bools, child_l_bools, child_r_bools) =
            outcomes_to_bools(&child_p_s, &child_p_g, &child_p_l, &child_p_r);
        let child_pareto_c = combined_from_outcomes(
            &child_p_s,
            &child_p_g,
            &child_p_l,
            &child_p_r,
            &cfg,
        );

        println!(
            "[iter {it}] parent_idx={parent_idx} parent_mb={parent_mb_c:.3} -> child_mb={child_mb_c:.3} — accepted (pareto_combined={child_pareto_c:.3}, pool_size={})",
            pool.len() + 1,
        );
        pool.push(Candidate {
            prompt: child_prompt,
            s_val: child_s_bools,
            g_val: child_g_bools,
            l_val: child_l_bools,
            r_val: child_r_bools,
        });
        let (s, g, l, r, combined) = best_pool_combined(&pool, &cfg);
        post_iter_feedback(&cfg, it, s, g, l, r, combined, pool.len());
    }

    println!("final full-val scoring: {} candidates", pool.len());
    let mut best_prompt = pool[0].prompt.clone();
    let mut best_acc = f64::NEG_INFINITY;
    let mut best_s_acc = 0.0;
    let mut best_g_acc = 0.0;
    let mut best_l_acc = 0.0;
    let mut best_r_acc = 0.0;
    for c in &pool {
        let (s, g, l, r) =
            score_quad(&cfg, &c.prompt, &val_s, &val_g, &val_l, &val_r, &idx);
        let combined = combined_from_outcomes(&s, &g, &l, &r, &cfg);
        if combined > best_acc {
            best_acc = combined;
            best_prompt = c.prompt.clone();
            best_s_acc = search_acc(&s);
            best_g_acc = grep_acc(&g);
            best_l_acc = glob_acc(&l);
            best_r_acc = read_acc(&r);
        }
    }

    let result = GepaRunResult {
        prompt: best_prompt,
        baseline_acc,
        best_acc,
        search_acc: best_s_acc,
        grep_acc: best_g_acc,
        glob_acc: best_l_acc,
        read_acc: best_r_acc,
        candidates: pool.len(),
        episode_id: cfg.episode_id.clone(),
    };
    let n_train = train_s.len() + train_g.len() + train_l.len() + train_r.len();
    let n_val = val_s.len() + val_g.len() + val_l.len() + val_r.len();
    post_final_feedback(&cfg, &result, n_train, n_val);
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
    use std::path::PathBuf;

    fn test_cfg() -> GepaConfig {
        GepaConfig {
            gateway: String::new(),
            search_function: String::new(),
            read_function: String::new(),
            grep_function: String::new(),
            glob_function: String::new(),
            variant: String::new(),
            reflect_function: String::new(),
            episode_id: String::new(),
            tags: json!({}),
            val_frac: 0.3,
            budget: 1,
            minibatch: 4,
            seed_prompt: String::new(),
            work: PathBuf::from("."),
            top_k: 8,
            w_search: 0.25,
            w_grep: 0.25,
            w_glob: 0.25,
            w_read: 0.25,
            seed: 42,
            pareto_size: 20,
            candidate_selection: CandidateSelection::Pareto,
        }
    }

    fn test_candidate(prompt: &str, hits: &[bool]) -> Candidate {
        Candidate {
            prompt: prompt.into(),
            s_val: hits.to_vec(),
            g_val: Vec::new(),
            l_val: Vec::new(),
            r_val: Vec::new(),
        }
    }

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
    fn first_read_call_parses_string_n_and_raw_arguments() {
        let content = vec![json!({
            "type": "tool_call", "name": "read",
            "arguments": {"path": "doc.htm", "n": "[#5]"}
        })];
        assert_eq!(first_read_call(&content).unwrap().n, 5);

        let raw = vec![json!({
            "type": "tool_call", "id": "r1", "name": "read",
            "arguments": Value::Null,
            "raw_arguments": "{\"path\":\"a.pdf\",\"n\":\"p.7\"}"
        })];
        let pick = first_read_call(&raw).unwrap();
        assert_eq!(pick.path, "a.pdf");
        assert_eq!(pick.n, 7);
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
    fn reflect_instruction_includes_scores_and_ok_fail() {
        let cfg = test_cfg();
        let ctx = ReflectContext {
            parent_prompt: "seed prompt".into(),
            parent_search_acc: 0.5,
            parent_grep_acc: 0.0,
            parent_glob_acc: 0.0,
            parent_read_acc: 0.0,
            parent_combined: 0.125,
            traces: vec![
                MinibatchTrace::Search {
                    ok: true,
                    ex: SearchExample {
                        episode_id: "e1".into(),
                        case_id: None,
                        question: "q ok".into(),
                        search_query: String::new(),
                        gold: vec![],
                        hit: true,
                        rank: None,
                    },
                    model_search: Some("term".into()),
                    top_k: "[#1] a".into(),
                },
                MinibatchTrace::Read {
                    ok: false,
                    ex: ReadExample {
                        episode_id: "e2".into(),
                        case_id: None,
                        question: "q fail".into(),
                        search_query: "sq".into(),
                        hits: vec![],
                        gold: vec![],
                        model_read: None,
                        hit: false,
                        prefill_source: "search".into(),
                        grep_result: String::new(),
                    },
                    model_read: None,
                    resolved: None,
                },
            ],
            rejected: vec![RejectedMutation {
                iter: 0,
                reason: "minibatch_regressed",
                detail: "parent_mb=0.40 child_mb=0.20".into(),
                prompt_preview: Some("bad prompt".into()),
            }],
        };
        let msg = build_reflect_instruction(&ctx, &cfg);
        assert!(msg.contains("PARENT SCORES ON MINIBATCH"));
        assert!(msg.contains("search=0.500"));
        assert!(msg.contains("grep=0.000"));
        assert!(msg.contains("(OK)"));
        assert!(msg.contains("(FAIL)"));
        assert!(msg.contains("REJECTED MUTATIONS"));
    }

    #[test]
    fn sample_minibatch_respects_balance_and_seed() {
        let train_s: Vec<SearchExample> = (0..20)
            .map(|i| SearchExample {
                episode_id: format!("q{i}"),
                case_id: None,
                question: format!("q{i}"),
                search_query: String::new(),
                gold: vec![],
                hit: false,
                rank: None,
            })
            .collect();
        let train_g: Vec<GrepExample> = (0..20)
            .map(|i| GrepExample {
                episode_id: format!("g{i}"),
                case_id: None,
                question: format!("g{i}"),
                grep_pattern: String::new(),
                gold: vec![],
                hit: false,
                rank: None,
                synthetic: false,
            })
            .collect();
        let train_l: Vec<GlobExample> = (0..20)
            .map(|i| GlobExample {
                episode_id: format!("l{i}"),
                case_id: None,
                question: format!("l{i}"),
                glob_pattern: String::new(),
                gold: vec![],
                hit: false,
                synthetic: false,
            })
            .collect();
        let train_r: Vec<ReadExample> = (0..20)
            .map(|i| ReadExample {
                episode_id: format!("r{i}"),
                case_id: None,
                question: format!("r{i}"),
                search_query: String::new(),
                hits: vec![],
                gold: vec![],
                model_read: None,
                hit: false,
                prefill_source: "search".into(),
                grep_result: String::new(),
            })
            .collect();
        let mut cfg = test_cfg();
        cfg.minibatch = 8;
        cfg.w_search = 0.25;
        cfg.w_grep = 0.25;
        cfg.w_glob = 0.25;
        cfg.w_read = 0.25;
        let mut rng1 = StdRng::seed_from_u64(99);
        let mb1 = sample_minibatch(&cfg, &train_s, &train_g, &train_l, &train_r, &mut rng1);
        assert_eq!(mb1.s.len(), 2);
        assert_eq!(mb1.g.len(), 2);
        assert_eq!(mb1.l.len(), 2);
        assert_eq!(mb1.r.len(), 2);
        let mut rng2 = StdRng::seed_from_u64(99);
        let mb2 = sample_minibatch(&cfg, &train_s, &train_g, &train_l, &train_r, &mut rng2);
        assert_eq!(mb1.s[0].episode_id, mb2.s[0].episode_id);
    }

    #[test]
    fn child_beats_parent_gate() {
        let cfg = test_cfg();
        let empty_g = Vec::<GrepOutcome>::new();
        let empty_l = Vec::<GlobOutcome>::new();
        let parent_s = vec![SearchOutcome { ok: false, model_search: None, top_k: String::new() }];
        let parent_r = vec![ReadOutcome { ok: false, model_read: None, resolved: None }];
        let child_s = vec![SearchOutcome { ok: true, model_search: None, top_k: String::new() }];
        let child_r = vec![ReadOutcome { ok: false, model_read: None, resolved: None }];
        assert!(child_beats_parent_on_minibatch(
            &parent_s, &empty_g, &empty_l, &parent_r, &child_s, &empty_g, &empty_l, &child_r, &cfg
        ));
        assert!(!child_beats_parent_on_minibatch(
            &parent_s, &empty_g, &empty_l, &parent_r, &parent_s, &empty_g, &empty_l, &child_r, &cfg
        ));
    }

    #[test]
    fn combined_acc_four_way() {
        let cfg = test_cfg();
        assert!((combined_acc_from_pools(true, true, true, true, 1.0, 0.0, 0.0, 0.0, &cfg) - 0.25).abs() < 1e-9);
        assert!((combined_acc_from_pools(true, false, false, true, 1.0, 0.0, 0.0, 1.0, &cfg) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rejected_history_cap() {
        let mut rejected = Vec::new();
        for i in 0..7 {
            push_rejected(
                &mut rejected,
                RejectedMutation {
                    iter: i,
                    reason: "test",
                    detail: format!("d{i}"),
                    prompt_preview: None,
                },
            );
        }
        assert_eq!(rejected.len(), REJECTED_HISTORY_CAP);
        assert_eq!(rejected[0].iter, 2);
    }

    #[test]
    fn output_truncation_detects_finish_reason() {
        assert!(output_likely_truncated("x", Some("length")));
        assert!(output_likely_truncated("x", Some("max_tokens")));
        assert!(!output_likely_truncated("short", Some("stop")));
        assert!(!output_likely_truncated(&"x".repeat(4000), Some("stop")));
        assert!(!output_likely_truncated(&"x".repeat(4000), None));
    }

    #[test]
    fn outcomes_from_traces_roundtrip() {
        let traces = vec![
            MinibatchTrace::Search {
                ok: true,
                ex: SearchExample {
                    episode_id: "e1".into(),
                    case_id: None,
                    question: "q".into(),
                    search_query: String::new(),
                    gold: vec!["a.pdf#p.1".into()],
                    hit: true,
                    rank: Some(1),
                },
                model_search: Some("terms".into()),
                top_k: "hits".into(),
            },
            MinibatchTrace::Read {
                ok: false,
                ex: ReadExample {
                    episode_id: "e2".into(),
                    case_id: None,
                    question: "q2".into(),
                    search_query: String::new(),
                    hits: vec![],
                    gold: vec!["b.pdf#p.2".into()],
                    model_read: None,
                    hit: false,
                    prefill_source: "search".into(),
                    grep_result: String::new(),
                },
                model_read: None,
                resolved: None,
            },
        ];
        let (s, _g, _l, r) = outcomes_from_traces(&traces);
        assert_eq!(s.len(), 1);
        assert!(s[0].ok);
        assert_eq!(s[0].model_search.as_deref(), Some("terms"));
        assert_eq!(r.len(), 1);
        assert!(!r[0].ok);
    }

    #[test]
    fn acceptance_gates() {
        assert!(passes_minibatch_gate(0.45, 0.275));
        assert!(!passes_minibatch_gate(0.10, 0.10));
        assert!(!passes_minibatch_gate(0.05, 0.10));
    }

    #[test]
    fn sample_pareto_set_respects_cap_and_weights() {
        let mk_search = |i: usize| SearchExample {
            episode_id: format!("e{i}"),
            case_id: None,
            question: format!("q{i}"),
            search_query: String::new(),
            gold: vec![],
            hit: false,
            rank: None,
        };
        let val_s: Vec<_> = (0..30).map(mk_search).collect();
        let mut rng = StdRng::seed_from_u64(7);
        let set = sample_pareto_set(
            &val_s,
            &[],
            &[],
            &[],
            20,
            [0.35, 0.15, 0.10, 0.40],
            &mut rng,
        );
        assert_eq!(set.len(), 20);
        assert_eq!(set.search.len(), 20);
    }

    #[test]
    fn select_parent_current_best() {
        let cfg = test_cfg();
        let pool = vec![
            test_candidate("weak", &[false, false]),
            test_candidate("strong", &[true, true, true]),
        ];
        let mut rng = StdRng::seed_from_u64(1);
        let mut cfg_best = cfg;
        cfg_best.candidate_selection = CandidateSelection::CurrentBest;
        assert_eq!(select_parent_idx(&pool, &cfg_best, &mut rng), 1);
    }

    #[test]
    fn pareto_parent_win_counts_prefer_frequent_winner() {
        let pool = vec![
            test_candidate("a", &[true, false, false]),
            test_candidate("b", &[false, true, true]),
            test_candidate("c", &[false, false, false]),
        ];
        let (frontier, counts) = pareto_frontier_win_counts(&pool);
        assert!(frontier.contains(&0));
        assert!(frontier.contains(&1));
        assert_eq!(counts[0], 1);
        assert_eq!(counts[1], 2);
        assert_eq!(counts[2], 0);
    }

    #[test]
    fn select_parent_weighted_prefers_frequent_winner() {
        let pool = vec![
            test_candidate("a", &[true, false, false]),
            test_candidate("b", &[false, true, true]),
            test_candidate("c", &[false, false, false]),
        ];
        let cfg = test_cfg();
        let mut rng = StdRng::seed_from_u64(99);
        let mut picks = [0usize; 3];
        for _ in 0..600 {
            let k = select_parent_idx(&pool, &cfg, &mut rng);
            picks[k] += 1;
        }
        assert!(picks[1] > picks[0]);
        assert!(picks[1] > picks[2]);
    }

    #[test]
    fn failure_budget_equal_splits_minibatch() {
        assert_eq!(failure_budget_equal(10, [true, false, false, true]), [5, 0, 0, 5]);
        assert_eq!(failure_budget_equal(9, [true, false, false, true]), [5, 0, 0, 4]);
        assert_eq!(failure_budget_equal(4, [true, false, false, false]), [4, 0, 0, 0]);
        assert_eq!(failure_budget_equal(4, [false, false, false, true]), [0, 0, 0, 4]);
        assert_eq!(failure_budget_legacy(4, true, true), (2, 2));
    }

    #[test]
    fn failure_budget_weighted_read_focus() {
        let have = [true, true, true, true];
        let w = [0.35, 0.15, 0.10, 0.40];
        let slots = failure_budget_weighted(12, have, w);
        assert_eq!(slots.iter().sum::<usize>(), 12);
        assert!(slots[3] >= slots[2]);
        assert!(slots.iter().all(|&n| n >= 1));
        assert_eq!(slots, [4, 2, 2, 4]);
    }

    #[test]
    fn failure_budget_weighted_falls_back_when_minibatch_too_small() {
        let have = [true, true, true, true];
        let w = [0.35, 0.15, 0.10, 0.40];
        assert_eq!(
            failure_budget_weighted(3, have, w),
            failure_budget_equal(3, have),
        );
    }

    #[test]
    fn failure_budget_weighted_equal_weights_matches_equal() {
        let have = [true, true, true, true];
        let w = [0.25; 4];
        for mb in [8, 12, 16] {
            assert_eq!(
                failure_budget_weighted(mb, have, w),
                failure_budget_equal(mb, have),
            );
        }
    }

    #[test]
    fn sample_minibatch_uses_weights() {
        let train_s: Vec<SearchExample> = (0..20)
            .map(|i| SearchExample {
                episode_id: format!("q{i}"),
                case_id: None,
                question: format!("q{i}"),
                search_query: String::new(),
                gold: vec![],
                hit: false,
                rank: None,
            })
            .collect();
        let train_g: Vec<GrepExample> = (0..20)
            .map(|i| GrepExample {
                episode_id: format!("g{i}"),
                case_id: None,
                question: format!("g{i}"),
                grep_pattern: String::new(),
                gold: vec![],
                hit: false,
                rank: None,
                synthetic: false,
            })
            .collect();
        let train_l: Vec<GlobExample> = (0..20)
            .map(|i| GlobExample {
                episode_id: format!("l{i}"),
                case_id: None,
                question: format!("l{i}"),
                glob_pattern: String::new(),
                gold: vec![],
                hit: false,
                synthetic: false,
            })
            .collect();
        let train_r: Vec<ReadExample> = (0..20)
            .map(|i| ReadExample {
                episode_id: format!("r{i}"),
                case_id: None,
                question: format!("r{i}"),
                search_query: String::new(),
                hits: vec![],
                gold: vec![],
                model_read: None,
                hit: false,
                prefill_source: "search".into(),
                grep_result: String::new(),
            })
            .collect();
        let mut cfg = test_cfg();
        cfg.minibatch = 12;
        cfg.w_search = 0.35;
        cfg.w_grep = 0.15;
        cfg.w_glob = 0.10;
        cfg.w_read = 0.40;
        let mb = sample_minibatch(
            &cfg,
            &train_s,
            &train_g,
            &train_l,
            &train_r,
            &mut StdRng::seed_from_u64(1),
        );
        assert_eq!(mb.s.len(), 4);
        assert_eq!(mb.g.len(), 2);
        assert_eq!(mb.l.len(), 2);
        assert_eq!(mb.r.len(), 4);
    }

    #[test]
    fn normalize_question_used_for_join() {
        use crate::export_tz::normalize_question;
        assert_eq!(normalize_question("  a  b "), "a b");
    }
}
