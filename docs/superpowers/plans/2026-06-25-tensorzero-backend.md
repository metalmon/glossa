# TensorZero Backend Implementation Plan (Plan 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `tensorzero` eval backend so `kb-eval run --fullwiki <corpus> --backend tensorzero` drives Qwen through the TensorZero gateway (banking an optimizable Track-B dataset) while glossa search/read run in-process and em/f1/retrieved feedback is posted per episode.

**Architecture:** `TensorZeroBackend` calls TZ's native `/inference` for function `answer_hotpot` (prompt + tools live in TZ config), parsing TZ content blocks; on a `tool_call` block it executes the tool against glossa in-process (shared with the openai backend), feeds the result back as a `tool_result` content block in the same `episode_id`, and loops to a final text answer; then it POSTs `/feedback` (`em`, `f1`, `retrieved`) for the episode. Recall@k is still computed in `run.rs` from the trace (unchanged).

**Tech Stack:** Rust (dev-only `eval` crate), existing `ureq` (no TLS — gateway is local http), `glossa` index/read/trace, the TZ stack from `eval/tensorzero/`.

## Global Constraints

- Plan 2 only. The TZ infra (`eval/tensorzero/` compose + config) already exists and is running — do NOT modify it here; this plan is the harness-side code.
- The `eval` crate is dev-only; no C-free constraint.
- The agent must use ONLY glossa's tools (the harness executes them) — TZ orchestrates the LLM turns and logs; it does not call glossa.
- `openai`/`cli`/`mock` backends and the distractor + fullwiki paths must keep working unchanged.
- The prompt lives in TZ config (`answer_hotpot` variant), NOT in this code — the request sends only the question.
- TDD: the episode loop is tested with an injectable transport (no live gateway/model needed). Run `cargo test -p kb-eval` (binary-only crate — no `--lib`).

---

### Task 1: Shared glossa tool execution + the injectable episode loop

**Files:**
- Create: `eval/src/backend/glossa_tools.rs`
- Modify: `eval/src/backend/openai.rs` (use the shared helper)
- Modify: `eval/src/backend/mod.rs` (declare `pub mod glossa_tools;`)
- Create: `eval/src/backend/tensorzero.rs` (the episode loop + content parsing; HTTP added in Task 2)

**Interfaces:**
- Produces: `glossa_tools::run_search(work, query, limit, trace) -> (String, Vec<String>)` (body for the model, + surfaced titles), `glossa_tools::run_read(work, path, location, trace) -> String`, `glossa_tools::exec(name, args, work, trace) -> (String, Vec<String>)`.
- Produces: `tensorzero::run_episode(chat, user_question, exec, max_rounds) -> EpisodeOutcome` with `pub struct EpisodeOutcome { pub answer: String, pub episode_id: Option<String>, pub surfaced_titles: Vec<String> }`.

- [ ] **Step 1: Declare the modules**

In `eval/src/backend/mod.rs` add (near the other `pub mod` lines):
```rust
pub mod glossa_tools;
pub mod tensorzero;
```

- [ ] **Step 2: Create the shared glossa tool executor**

Create `eval/src/backend/glossa_tools.rs`:
```rust
use glossa::trace::TraceLog;
use serde_json::{json, Value};
use std::path::Path;

const READ_CHARS_CAP: usize = 4000;

/// Run a BM25 search against the corpus index; return (model-facing text, surfaced titles).
pub fn run_search(work: &Path, query: &str, limit: usize, trace: &TraceLog) -> (String, Vec<String>) {
    let idx = match glossa::index::store::DocIndex::open_or_create(work) {
        Ok(i) => i,
        Err(e) => return (format!("search error: {e}"), Vec::new()),
    };
    match idx.search(query, limit.max(1)) {
        Ok(hits) => {
            let trace_hits: Vec<Value> = hits
                .iter()
                .map(|h| json!({ "path": h.path, "location": h.location, "score": h.score }))
                .collect();
            trace.log("search", json!({ "query": query }), json!(trace_hits));
            let titles: Vec<String> = hits.iter().map(|h| h.location.clone()).collect();
            if hits.is_empty() {
                return ("(no results)".to_string(), titles);
            }
            let body = hits
                .iter()
                .map(|h| format!("{}:{}: {}  [{:.3}]", h.path, h.location, h.snippet, h.score))
                .collect::<Vec<_>>()
                .join("\n");
            (body, titles)
        }
        Err(e) => (format!("search error: {e}"), Vec::new()),
    }
}

/// Read a document (optionally a location); truncated to fit small-model context.
pub fn run_read(work: &Path, path: &str, location: Option<&str>, trace: &TraceLog) -> String {
    let _ = work; // path is absolute in search results
    match glossa::read::read_region(Path::new(path), location) {
        Ok(text) => {
            trace.log("read", json!({ "path": path, "location": location }), json!({ "path": path }));
            if text.chars().count() > READ_CHARS_CAP {
                text.chars().take(READ_CHARS_CAP).collect::<String>() + "\n…(truncated)"
            } else {
                text
            }
        }
        Err(e) => format!("read error: {e}"),
    }
}

/// Dispatch a tool by name. Returns (result string for the model, titles surfaced by a search).
pub fn exec(name: &str, args: &Value, work: &Path, trace: &TraceLog) -> (String, Vec<String>) {
    match name {
        "search" => {
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
            run_search(work, query, limit, trace)
        }
        "read" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let location = args.get("location").and_then(|v| v.as_str());
            (run_read(work, path, location, trace), Vec::new())
        }
        other => (format!("unknown tool: {other}"), Vec::new()),
    }
}
```

- [ ] **Step 3: Point the openai backend at the shared executor**

In `eval/src/backend/openai.rs`, replace the body of its private `execute_tool` (the `match name { "search" => …, "read" => …, … }`) so it delegates and discards the titles:
```rust
fn execute_tool(name: &str, args: &Value, work: &Path, trace: &TraceLog) -> String {
    crate::backend::glossa_tools::exec(name, args, work, trace).0
}
```
Remove now-unused imports in `openai.rs` if the compiler flags them (e.g. `DocIndex`, `read_region`, `READ_CHARS_CAP` if they were only used by the old `execute_tool`). Keep everything else. Run `cargo test -p kb-eval openai` — the existing openai loop tests must still pass.

- [ ] **Step 4: Write the failing test + the episode loop**

Create `eval/src/backend/tensorzero.rs`:
```rust
use serde_json::{json, Value};

pub struct EpisodeOutcome {
    pub answer: String,
    pub episode_id: Option<String>,
    pub surfaced_titles: Vec<String>,
}

/// A single TensorZero `/inference` turn result.
pub struct TzTurn {
    pub content: Vec<Value>, // TZ content blocks: {type:"tool_call"|"text"|"thought", ...}
    pub episode_id: String,
}

/// Drive a TensorZero episode to a final text answer.
///
/// `chat(messages, episode_id)` performs one `/inference` call (episode_id is None on the first turn,
/// then the id returned by the first turn). When the returned content has `tool_call` blocks, each is
/// executed via `exec(name, args)` and fed back as a `tool_result` content block; otherwise the
/// concatenated `text` blocks are the answer.
pub fn run_episode<C, X>(
    mut chat: C,
    user_question: &str,
    mut exec: X,
    max_rounds: usize,
) -> anyhow::Result<EpisodeOutcome>
where
    C: FnMut(&[Value], Option<&str>) -> anyhow::Result<TzTurn>,
    X: FnMut(&str, &Value) -> (String, Vec<String>),
{
    let mut messages: Vec<Value> = vec![json!({ "role": "user", "content": user_question })];
    let mut episode_id: Option<String> = None;
    let mut surfaced_titles: Vec<String> = Vec::new();

    for _ in 0..max_rounds {
        let turn = chat(&messages, episode_id.as_deref())?;
        episode_id = Some(turn.episode_id.clone());

        let tool_calls: Vec<&Value> = turn
            .content
            .iter()
            .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_call"))
            .collect();

        if tool_calls.is_empty() {
            let answer: String = turn
                .content
                .iter()
                .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("");
            return Ok(EpisodeOutcome { answer, episode_id, surfaced_titles });
        }

        for call in &tool_calls {
            let id = call.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let name = call.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let args = call.get("arguments").cloned().unwrap_or_else(|| json!({}));
            // echo the assistant tool_call, then the tool_result (result MUST be a string)
            messages.push(json!({
                "role": "assistant",
                "content": [{ "type": "tool_call", "id": id, "name": name, "arguments": args }]
            }));
            let (result, titles) = exec(name, &args);
            surfaced_titles.extend(titles);
            messages.push(json!({
                "role": "user",
                "content": [{ "type": "tool_result", "id": id, "name": name, "result": result }]
            }));
        }
    }
    // Out of rounds: best-effort empty answer (the report still scores it).
    Ok(EpisodeOutcome { answer: String::new(), episode_id, surfaced_titles })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn episode_executes_tool_then_answers() {
        let round = RefCell::new(0usize);
        let chat = |_msgs: &[Value], _eid: Option<&str>| {
            let mut r = round.borrow_mut();
            *r += 1;
            if *r == 1 {
                Ok(TzTurn {
                    content: vec![json!({ "type": "tool_call", "id": "c1", "name": "search", "arguments": { "query": "corliss" } })],
                    episode_id: "ep1".into(),
                })
            } else {
                Ok(TzTurn {
                    content: vec![json!({ "type": "text", "text": "ANSWER: Chief of Protocol" })],
                    episode_id: "ep1".into(),
                })
            }
        };
        let exec = |name: &str, args: &Value| {
            assert_eq!(name, "search");
            assert_eq!(args["query"], "corliss");
            ("Meet_Corliss_Archer.md:Meet Corliss Archer: …  [9.0]".to_string(),
             vec!["Meet Corliss Archer".to_string()])
        };
        let out = run_episode(chat, "Question: ...", exec, 8).unwrap();
        assert_eq!(out.answer, "ANSWER: Chief of Protocol");
        assert_eq!(out.episode_id.as_deref(), Some("ep1"));
        assert_eq!(out.surfaced_titles, vec!["Meet Corliss Archer".to_string()]);
    }

    #[test]
    fn episode_returns_direct_answer() {
        let chat = |_: &[Value], _: Option<&str>| Ok(TzTurn {
            content: vec![json!({ "type": "text", "text": "ANSWER: yes" })],
            episode_id: "e".into(),
        });
        let exec = |_: &str, _: &Value| (String::new(), Vec::new());
        let out = run_episode(chat, "q", exec, 4).unwrap();
        assert_eq!(out.answer, "ANSWER: yes");
    }
}
```

- [ ] **Step 5: Run the tests**

Run: `cargo test -p kb-eval tensorzero` then `cargo test -p kb-eval openai` then full `cargo test -p kb-eval`.
Expected: the two new episode-loop tests pass; the openai loop tests still pass after the shared-executor refactor; whole crate green.

- [ ] **Step 6: Commit**
```bash
git add eval/src/backend/glossa_tools.rs eval/src/backend/openai.rs eval/src/backend/mod.rs eval/src/backend/tensorzero.rs
git commit -m "feat(eval): shared glossa tool exec + TensorZero episode loop (injectable, tested)"
```

---

### Task 2: `TensorZeroBackend` (HTTP /inference + /feedback) + wiring

**Files:**
- Modify: `eval/src/backend/tensorzero.rs` (add `TensorZeroBackend` impl + feedback)
- Modify: `eval/src/main.rs` (`BackendKind::Tensorzero`, `--tensorzero-endpoint` flag, construction)

**Interfaces:**
- Consumes: `run_episode`, `glossa_tools::exec`, `crate::score::{exact_match, token_f1, normalize}`, `glossa::trace::TraceLog`, `AgentBackend`.
- Produces: `pub struct TensorZeroBackend { pub endpoint: String, pub function: String, pub timeout: Duration }` implementing `AgentBackend`.

- [ ] **Step 1: Add the backend impl to `eval/src/backend/tensorzero.rs`**

Append:
```rust
use super::{prompt, AgentBackend};
use crate::dataset::Question;
use anyhow::{anyhow, bail, Context};
use glossa::trace::TraceLog;
use std::path::Path;
use std::time::Duration;

const MAX_ROUNDS: usize = 8;

/// Drives Qwen through the TensorZero gateway (function `answer_hotpot`), executing glossa tools
/// in-process, then posts em/f1/retrieved feedback for the episode. The prompt lives in TZ config.
pub struct TensorZeroBackend {
    pub endpoint: String, // e.g. http://localhost:3000
    pub function: String, // e.g. answer_hotpot
    pub timeout: Duration,
}

impl AgentBackend for TensorZeroBackend {
    fn needs_corpus(&self) -> bool {
        true
    }

    fn answer(&self, work: &Path, q: &Question) -> anyhow::Result<String> {
        let url = format!("{}/inference", self.endpoint.trim_end_matches('/'));
        let function = self.function.clone();
        let timeout = self.timeout;
        let chat = |messages: &[Value], episode_id: Option<&str>| -> anyhow::Result<TzTurn> {
            let mut body = json!({ "function_name": function, "input": { "messages": messages } });
            if let Some(eid) = episode_id {
                body["episode_id"] = json!(eid);
            }
            let resp = ureq::post(&url)
                .timeout(timeout)
                .set("Content-Type", "application/json")
                .send_string(&serde_json::to_string(&body)?)
                .map_err(|e| anyhow!("tensorzero /inference failed: {e}"))?;
            let text = resp.into_string().context("read /inference response")?;
            let v: Value = serde_json::from_str(&text).context("parse /inference json")?;
            if let Some(err) = v.get("error") {
                bail!("tensorzero error: {err}");
            }
            let episode_id = v.get("episode_id").and_then(|x| x.as_str()).unwrap_or("").to_string();
            let content = v.get("content").and_then(|c| c.as_array()).cloned().unwrap_or_default();
            Ok(TzTurn { content, episode_id })
        };

        let trace = TraceLog::to_dir(work);
        let exec = |name: &str, args: &Value| crate::backend::glossa_tools::exec(name, args, work, &trace);

        let user = prompt::user_prompt(q);
        let outcome = run_episode(chat, &user, exec, MAX_ROUNDS)?;
        let pred = prompt::parse_answer(&outcome.answer);

        // Post Track-B feedback for this episode (best-effort; never fail the run on feedback error).
        if let Some(eid) = &outcome.episode_id {
            let em = crate::score::exact_match(&pred, &q.answer);
            let f1 = crate::score::token_f1(&pred, &q.answer);
            let retrieved = retrieved_any(&outcome.surfaced_titles, &q.supporting_titles);
            self.feedback(eid, "em", json!(em));
            self.feedback(eid, "f1", json!(f1));
            self.feedback(eid, "retrieved", json!(retrieved));
        }
        Ok(pred)
    }
}

impl TensorZeroBackend {
    fn feedback(&self, episode_id: &str, metric: &str, value: Value) {
        let url = format!("{}/feedback", self.endpoint.trim_end_matches('/'));
        let body = json!({ "episode_id": episode_id, "metric_name": metric, "value": value });
        let _ = ureq::post(&url)
            .timeout(self.timeout)
            .set("Content-Type", "application/json")
            .send_string(&serde_json::to_string(&body).unwrap_or_default());
    }
}

/// True if any gold supporting title appears among the titles the agent's searches surfaced.
fn retrieved_any(surfaced: &[String], gold: &[String]) -> bool {
    if gold.is_empty() {
        return true;
    }
    let surf: Vec<String> = surfaced.iter().map(|s| crate::score::normalize(s)).collect();
    gold.iter().any(|g| surf.contains(&crate::score::normalize(g)))
}

#[cfg(test)]
mod retrieved_tests {
    use super::*;

    #[test]
    fn retrieved_matches_normalized_title() {
        assert!(retrieved_any(&["The Beatles".into()], &["the beatles".into()]));
        assert!(!retrieved_any(&["X".into()], &["Y".into()]));
        assert!(retrieved_any(&[], &["anything".into()])); // empty gold -> trivially true
    }
}
```
NOTE: `crate::score::normalize` must be `pub` (it already is, used by Plan 1). If `prompt::user_prompt`/`prompt::parse_answer` are not `pub`, make them `pub` (they are used by the openai backend already, so they are at least `pub(crate)` — confirm and use the same visibility).

- [ ] **Step 2: Run the backend tests**

Run: `cargo test -p kb-eval tensorzero`
Expected: episode-loop tests (Task 1) + `retrieved_matches_normalized_title` pass.

- [ ] **Step 3: Wire the backend into `main.rs`**

Add to `enum BackendKind`: `Tensorzero` (so it becomes `{ Mock, Cli, Openai, Tensorzero }`).
Add to the `Run` variant fields:
```rust
        /// TensorZero gateway base URL (for `--backend tensorzero`).
        #[arg(long, default_value = "http://localhost:3000")]
        tensorzero_endpoint: String,
        /// TensorZero function name to call.
        #[arg(long, default_value = "answer_hotpot")]
        tensorzero_function: String,
```
Add `tensorzero_endpoint, tensorzero_function` to the `Cmd::Run { … }` destructuring, and a match arm in the backend construction:
```rust
                BackendKind::Tensorzero => Box::new(backend::tensorzero::TensorZeroBackend {
                    endpoint: tensorzero_endpoint,
                    function: tensorzero_function,
                    timeout,
                }),
```

- [ ] **Step 4: Verify the whole crate + binary**

Run: `cargo test -p kb-eval` (all green) and `cargo build -p kb-eval --release`.
Run: `kb-eval run --help` shows `--backend` accepting `tensorzero` and the `--tensorzero-endpoint` flag.

- [ ] **Step 5: Commit**
```bash
git add eval/src/backend/tensorzero.rs eval/src/main.rs
git commit -m "feat(eval): tensorzero backend — /inference episode loop + em/f1/retrieved feedback"
```

---

## Notes for the implementer

- The TZ gateway must be up (`docker compose up -d` in `eval/tensorzero/`) only for a *live* run — NOT for the tests (the episode loop is tested with an injectable `chat`). Do not start Docker during implementation.
- After both tasks, a live smoke is operational (needs the gateway + LM Studio + an indexed corpus):
  `kb-eval run --dataset eval-data\hotpot_dev_distractor_v1.json --backend tensorzero --limit 2 --work eval-corpus --kb-bin target\release\kb.exe` — then check the TZ UI (`:4000`) Observability for the episodes + feedback. (Operational, not part of this plan's tests.)
- `--fullwiki` composes with `--backend tensorzero` automatically (the fullwiki gating in `run.rs` is backend-agnostic): `kb-eval run --fullwiki wiki-corpus --backend tensorzero …`.
