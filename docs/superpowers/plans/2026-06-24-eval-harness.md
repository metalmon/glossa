# glossa agent-eval harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `kb-eval` harness that drives an LLM (Claude / Qwen / mock) to answer HotpotQA-distractor questions using glossa's real MCP tools, captures glossa's tool-call trace, and scores answer EM/F1 + retrieval-recall.

**Architecture:** Convert the repo to a cargo workspace: the product crate `glossa` (lib + `kb`) gains two small MCP-server features (JSONL trace logging, `--no-graph`); a new dev crate `eval` (`kb-eval`) orchestrates — it builds a per-question corpus, drives a model **that is itself the MCP client** of glossa (Claude via `claude -p`, Qwen via LM Studio, or a deterministic `mock`), reads the trace glossa wrote, and scores.

**Tech Stack:** Rust; `serde`/`serde_json`; `ureq` (HTTP to LM Studio, no-TLS/localhost); subprocess for `kb` and `claude`. Plotting/heatmap deferred.

## Global Constraints

- Product crate `glossa` stays pure-Rust, offline, single-binary; its deps do NOT change except the new trace module (std + serde only). `cargo tree -p glossa -i cc` stays empty.
- The harness drives the **real shipped path**: the model is a **direct MCP client** of glossa; the harness does NOT mediate tool calls. Trace comes from the glossa MCP **server** (JSONL).
- Harness HTTP uses `ureq = { version = "2", default-features = false }` (no TLS → no `ring`/`cc`; LM Studio is `http://localhost`).
- Runs are **sequential**; trace correlation is by time window `[t_send, t_recv]` around each backend call.
- A single failing question never aborts a run (recorded as a `failed` row, scored 0).
- TDD throughout; the `mock` backend gives a deterministic end-to-end test with no live model/network.
- Deferred (do NOT build): PNG heatmap, LLM-judge, Track B, fullwiki/large-corpus graph A/B, dataset auto-download, TensorZero.

## File Structure

- `Cargo.toml` (root) — gains `[workspace] members = ["eval"]` (the root `glossa` package stays the root member).
- `src/trace.rs` (new, in `glossa` lib) — `TraceEntry` + `TraceLog`; declared in `src/lib.rs`.
- `src/mcp.rs` (modify) — tools call `TraceLog` when enabled; `GlossaServer::new` gains `trace`/`no_graph`.
- `src/main.rs` (modify) — `kb mcp` gains `--trace` and `--no-graph`.
- `eval/Cargo.toml` (new) — `kb-eval` package.
- `eval/src/main.rs` — CLI (`kb-eval run …`).
- `eval/src/dataset.rs` — HotpotQA parse + `sanitize_title`.
- `eval/src/score.rs` — normalize / EM / token-F1 / retrieval-recall (pure).
- `eval/src/trace_read.rs` — read+window glossa traces; extract seen files.
- `eval/src/corpus.rs` — write per-question corpus + `kb index` subprocess.
- `eval/src/backend/mod.rs` + `prompt.rs` + `mock.rs` + `claude.rs` + `qwen.rs`.
- `eval/src/run.rs` — orchestration + report.

---

### Task 1: glossa — `glossa::trace` module (TraceEntry + TraceLog)

**Files:**
- Create: `src/trace.rs`
- Modify: `src/lib.rs` (add `pub mod trace;`)
- Test: `src/trace.rs` (inline)

**Interfaces:**
- Produces:
  - `trace::TraceEntry { ts_ms: u64, tool: String, args: serde_json::Value, result: serde_json::Value }` (derive Serialize, Deserialize, Debug, Clone, PartialEq)
  - `trace::now_ms() -> u64`
  - `trace::TraceLog` with `TraceLog::disabled() -> TraceLog`, `TraceLog::to_dir(root: &Path) -> TraceLog` (file `root/.glossa/traces/<ts_ms>-<pid>.jsonl`), and `fn log(&self, tool: &str, args: serde_json::Value, result: serde_json::Value)` (no-op when disabled; best-effort append).

- [ ] **Step 1: Write the failing test**

Create `src/trace.rs`:
```rust
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TraceEntry {
    pub ts_ms: u64,
    pub tool: String,
    pub args: serde_json::Value,
    pub result: serde_json::Value,
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Append-only JSONL tool-call log. `disabled()` is a no-op; `to_dir()` writes one line per call.
#[derive(Clone)]
pub struct TraceLog {
    path: Option<PathBuf>,
}

impl TraceLog {
    pub fn disabled() -> TraceLog {
        TraceLog { path: None }
    }

    pub fn to_dir(root: &Path) -> TraceLog {
        let dir = root.join(".glossa").join("traces");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join(format!("{}-{}.jsonl", now_ms(), std::process::id()));
        TraceLog { path: Some(file) }
    }

    pub fn log(&self, tool: &str, args: serde_json::Value, result: serde_json::Value) {
        let Some(p) = &self.path else { return };
        let entry = TraceEntry { ts_ms: now_ms(), tool: tool.to_string(), args, result };
        if let Ok(line) = serde_json::to_string(&entry) {
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
                let _ = writeln!(f, "{line}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        TraceLog::disabled().log("search", serde_json::json!({"q":"x"}), serde_json::json!([]));
        assert!(!dir.path().join(".glossa").join("traces").exists());
    }

    #[test]
    fn enabled_appends_parseable_lines() {
        let dir = tempfile::tempdir().unwrap();
        let log = TraceLog::to_dir(dir.path());
        log.log("search", serde_json::json!({"query":"поверка"}), serde_json::json!([{"path":"a.md","location":"p.1","score":1.0}]));
        log.log("read", serde_json::json!({"path":"a.md"}), serde_json::json!({"path":"a.md","location":"p.1"}));

        let tdir = dir.path().join(".glossa").join("traces");
        let file = std::fs::read_dir(&tdir).unwrap().next().unwrap().unwrap().path();
        let body = std::fs::read_to_string(file).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2);
        let e0: TraceEntry = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(e0.tool, "search");
        assert_eq!(e0.args["query"], "поверка");
    }
}
```

- [ ] **Step 2: Run → fail → declare module → pass**

Run `cargo test --lib trace` (RED — module not declared), add `pub mod trace;` to `src/lib.rs`, re-run → PASS, then `cargo test`.

- [ ] **Step 3: Commit**

```bash
git add src/trace.rs src/lib.rs
git commit -m "feat(glossa): trace module — JSONL tool-call log (TraceEntry/TraceLog)"
```

---

### Task 2: glossa — wire trace + `--no-graph`/`--trace` into the MCP server

**Files:**
- Modify: `src/mcp.rs` (GlossaServer holds a `TraceLog`; tools log; `new` gains `trace`/`no_graph`)
- Modify: `src/main.rs` (`Cmd::Mcp` gains `--trace`, `--no-graph`)
- Test: `src/mcp.rs` (inline — profile/no-graph gating)

**Interfaces:**
- Consumes: `glossa::trace::TraceLog`.
- Produces: `GlossaServer::new(root: PathBuf, profile: Profile, trace: bool, no_graph: bool) -> Self`.

- [ ] **Step 1: Update `GlossaServer`** (`src/mcp.rs`)

Change the struct + constructor and add the no-graph tool set:
```rust
#[derive(Clone)]
pub struct GlossaServer {
    root: PathBuf,
    tool_router: ToolRouter<Self>,
    trace: crate::trace::TraceLog,
}

const GRAPH_TOOLS: &[&str] = &["glossary", "neighbors", "graph_upsert", "resolve", "index", "reindex", "purge"];

impl GlossaServer {
    pub fn new(root: PathBuf, profile: Profile, trace: bool, no_graph: bool) -> Self {
        let mut router = Self::tool_router();
        if profile == Profile::Reader {
            for t in EDITOR_TOOLS.iter().chain(FULL_TOOLS) {
                router.disable_route(*t);
            }
        } else if profile == Profile::Editor {
            for t in FULL_TOOLS {
                router.disable_route(*t);
            }
        }
        if no_graph {
            for t in GRAPH_TOOLS {
                router.disable_route(*t);
            }
        }
        let trace = if trace { crate::trace::TraceLog::to_dir(&root) } else { crate::trace::TraceLog::disabled() };
        Self { root, tool_router: router, trace }
    }

    #[cfg(test)]
    pub fn enabled_tools(&self) -> Vec<String> {
        self.tool_router.list_all().iter().map(|t| t.name.to_string()).collect()
    }
}
```
(Keep `EDITOR_TOOLS`/`FULL_TOOLS`. `GRAPH_TOOLS` is new.)

- [ ] **Step 2: Log from the `search` and `read` tools** (`src/mcp.rs`)

In the `search` tool, after `hits` are computed and before returning, log them (match the hit field names actually in scope — scan hits have `doc_path/location/line`, ranked hits `path/location/score`):
```rust
        let trace_hits: Vec<serde_json::Value> = hits.iter().map(|h| serde_json::json!({
            "path": h.doc_path.display().to_string(), "location": h.location, "line": h.line
        })).collect();
        self.trace.log("search", serde_json::json!({"query": a.query}), serde_json::json!(trace_hits));
```
In the `read` tool, after resolving the text and before returning:
```rust
        self.trace.log("read", serde_json::json!({"path": a.path, "location": a.location}), serde_json::json!({"path": a.path}));
```
(If `search` has separate scan/ranked branches, log in each with the fields present there.)

- [ ] **Step 3: Update the gating test** (`src/mcp.rs`)

Adjust `GlossaServer::new` calls to the new signature and assert `--no-graph`:
```rust
    #[test]
    fn profile_gates_tool_visibility() {
        let root = std::path::PathBuf::from(".");
        let reader = GlossaServer::new(root.clone(), Profile::Reader, false, false).enabled_tools();
        assert!(reader.contains(&"search".to_string()) && reader.contains(&"read".to_string()));
        assert!(!reader.contains(&"index".to_string()) && !reader.contains(&"graph_upsert".to_string()) && !reader.contains(&"purge".to_string()));

        let editor = GlossaServer::new(root.clone(), Profile::Editor, false, false).enabled_tools();
        assert!(editor.contains(&"index".to_string()) && editor.contains(&"resolve".to_string()));
        assert!(!editor.contains(&"purge".to_string()));

        let full = GlossaServer::new(root.clone(), Profile::Full, false, false).enabled_tools();
        assert!(full.contains(&"purge".to_string()));

        let ng = GlossaServer::new(root, Profile::Editor, false, true).enabled_tools();
        assert!(ng.contains(&"search".to_string()) && ng.contains(&"read".to_string()));
        assert!(!ng.contains(&"neighbors".to_string()) && !ng.contains(&"graph_upsert".to_string()) && !ng.contains(&"index".to_string()));
    }
```

- [ ] **Step 4: CLI flags** (`src/main.rs`)

In `Cmd::Mcp`, add `--trace`/`--no-graph` and pass them:
```rust
    Mcp {
        path: Option<PathBuf>,
        #[arg(long, default_value = "editor")]
        profile: String,
        /// Log every tool call to <root>/.glossa/traces/*.jsonl (for the eval harness).
        #[arg(long)]
        trace: bool,
        /// Expose only search + read (graph/index/admin tools hidden) — eval control arm.
        #[arg(long = "no-graph")]
        no_graph: bool,
    },
```
```rust
        Cmd::Mcp { path, profile, trace, no_graph } => {
            let path = glossa::root::resolve_root(path);
            use rmcp::{transport::stdio, ServiceExt};
            let server = glossa::mcp::GlossaServer::new(path, glossa::mcp::Profile::parse(&profile), trace, no_graph);
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async move {
                let service = server.serve(stdio()).await?;
                let _ = service.waiting().await;
                Ok::<(), anyhow::Error>(())
            })?;
            Ok(())
        }
```

- [ ] **Step 5: Run + C-hygiene + commit**

`cargo test`; `cargo tree -i cc` (empty). Commit:
```bash
git add src/mcp.rs src/main.rs
git commit -m "feat(glossa): kb mcp --trace (JSONL tool log) + --no-graph (search/read only)"
```

---

### Task 3: workspace + `kb-eval` crate skeleton

**Files:**
- Modify: `Cargo.toml` (root — add `[workspace]`)
- Create: `eval/Cargo.toml`, `eval/src/main.rs`
- Test: `eval/src/main.rs` (inline trivial)

**Interfaces:**
- Produces: a building `kb-eval` binary with clap CLI `run --dataset <path> --backend <mock|qwen|claude> [--limit N] [--lmstudio-url URL] [--kb-bin PATH] [--work DIR]`.

- [ ] **Step 1: Root workspace** (`Cargo.toml`)

Append to the root `Cargo.toml` (leave existing `[package]`/`[dependencies]` intact):
```toml
[workspace]
members = ["eval"]
```

- [ ] **Step 2: Create `eval/Cargo.toml`**
```toml
[package]
name = "kb-eval"
version = "0.0.1"
edition = "2021"

[[bin]]
name = "kb-eval"
path = "src/main.rs"

[dependencies]
glossa = { path = ".." }
anyhow = "1"
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
ureq = { version = "2", default-features = false }

[dev-dependencies]
tempfile = "3"
assert_cmd = "2"
predicates = "3"
```

- [ ] **Step 3: Create `eval/src/main.rs`**
```rust
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "kb-eval", about = "glossa agent-eval harness")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run a benchmark and score it.
    Run {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long, value_enum)]
        backend: BackendKind,
        #[arg(long, default_value_t = 0)]
        limit: usize, // 0 = all
        #[arg(long, default_value = "http://localhost:1234")]
        lmstudio_url: String,
        #[arg(long, default_value = "kb")]
        kb_bin: String,
        #[arg(long, default_value = "eval-corpus")]
        work: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendKind { Mock, Qwen, Claude }

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Run { .. } => {
            println!("kb-eval: not yet implemented");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_compiles() {
        assert!(true);
    }
}
```

- [ ] **Step 4: Build the workspace + commit**

`cargo build` (builds `kb` + `kb-eval`); `cargo test`. Confirm the product stays clean: `cargo tree -p glossa -i cc` → empty.
```bash
git add Cargo.toml eval/Cargo.toml eval/src/main.rs
git commit -m "build: cargo workspace + kb-eval crate skeleton"
```

---

### Task 4: `eval` — dataset (HotpotQA-distractor parse)

**Files:**
- Create: `eval/src/dataset.rs`
- Modify: `eval/src/main.rs` (add `mod dataset;`)
- Test: `eval/src/dataset.rs` (inline)

**Interfaces:**
- Produces:
  - `dataset::Paragraph { title: String, sentences: Vec<String> }`
  - `dataset::Question { id: String, question: String, answer: String, paragraphs: Vec<Paragraph>, supporting_titles: Vec<String> }`
  - `dataset::parse_hotpot(json: &str) -> anyhow::Result<Vec<Question>>`
  - `dataset::sanitize_title(title: &str) -> String`

- [ ] **Step 1: Write the failing test**

Create `eval/src/dataset.rs`:
```rust
use anyhow::Context;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq)]
pub struct Paragraph {
    pub title: String,
    pub sentences: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub id: String,
    pub question: String,
    pub answer: String,
    pub paragraphs: Vec<Paragraph>,
    pub supporting_titles: Vec<String>,
}

#[derive(Deserialize)]
struct RawItem {
    #[serde(rename = "_id")]
    id: String,
    question: String,
    answer: String,
    context: Vec<(String, Vec<String>)>,
    supporting_facts: Vec<(String, i64)>,
}

pub fn sanitize_title(title: &str) -> String {
    let mut s: String = title
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' { c } else { '_' })
        .collect();
    s = s.trim().replace(' ', "_");
    if s.is_empty() {
        s.push_str("untitled");
    }
    s
}

pub fn parse_hotpot(json: &str) -> anyhow::Result<Vec<Question>> {
    let raw: Vec<RawItem> = serde_json::from_str(json).context("parse hotpot json")?;
    Ok(raw
        .into_iter()
        .map(|r| {
            let mut titles: Vec<String> = r.supporting_facts.into_iter().map(|(t, _)| t).collect();
            titles.sort();
            titles.dedup();
            Question {
                id: r.id,
                question: r.question,
                answer: r.answer,
                paragraphs: r.context.into_iter().map(|(title, sentences)| Paragraph { title, sentences }).collect(),
                supporting_titles: titles,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"[
      {"_id":"q1","question":"Who?","answer":"Bob",
       "context":[["Alice",["s1.","s2."]],["Bob Page",["b1."]]],
       "supporting_facts":[["Bob Page",0],["Bob Page",0],["Alice",1]]}
    ]"#;

    #[test]
    fn parses_questions_and_dedups_supporting_titles() {
        let qs = parse_hotpot(SAMPLE).unwrap();
        assert_eq!(qs.len(), 1);
        assert_eq!(qs[0].answer, "Bob");
        assert_eq!(qs[0].paragraphs.len(), 2);
        assert_eq!(qs[0].supporting_titles, vec!["Alice".to_string(), "Bob Page".to_string()]);
    }

    #[test]
    fn sanitize_title_is_fs_safe() {
        assert_eq!(sanitize_title("Bob Page"), "Bob_Page");
        assert_eq!(sanitize_title("A/B: C?"), "A_B__C_");
    }
}
```

- [ ] **Step 2: Run → fail → add `mod dataset;` to main.rs → pass** (`cargo test -p kb-eval dataset`).

- [ ] **Step 3: Commit**
```bash
git add eval/src/dataset.rs eval/src/main.rs
git commit -m "feat(eval): HotpotQA-distractor dataset parsing"
```

---

### Task 5: `eval` — scoring (normalize / EM / token-F1 / retrieval-recall)

**Files:**
- Create: `eval/src/score.rs`
- Modify: `eval/src/main.rs` (add `mod score;`)
- Test: `eval/src/score.rs` (inline)

**Interfaces:**
- Consumes: `dataset::sanitize_title`.
- Produces: `score::{normalize(&str)->String, exact_match(&str,&str)->bool, token_f1(&str,&str)->f32, retrieval_recall(seen_files: &[String], supporting_titles: &[String]) -> f32}`

- [ ] **Step 1: Write the failing test**

Create `eval/src/score.rs`:
```rust
use crate::dataset::sanitize_title;

/// HotpotQA answer normalization: lowercase, drop articles, drop punctuation, collapse whitespace.
pub fn normalize(s: &str) -> String {
    let lower = s.to_lowercase();
    let no_punct: String = lower.chars().map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' }).collect();
    no_punct
        .split_whitespace()
        .filter(|w| !matches!(*w, "a" | "an" | "the"))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn exact_match(pred: &str, gold: &str) -> bool {
    normalize(pred) == normalize(gold)
}

pub fn token_f1(pred: &str, gold: &str) -> f32 {
    let p: Vec<String> = normalize(pred).split_whitespace().map(|s| s.to_string()).collect();
    let g: Vec<String> = normalize(gold).split_whitespace().map(|s| s.to_string()).collect();
    if p.is_empty() || g.is_empty() {
        return if p.is_empty() && g.is_empty() { 1.0 } else { 0.0 };
    }
    let mut shared = 0usize;
    let mut gleft = g.clone();
    for tok in &p {
        if let Some(pos) = gleft.iter().position(|x| x == tok) {
            shared += 1;
            gleft.remove(pos);
        }
    }
    if shared == 0 {
        return 0.0;
    }
    let precision = shared as f32 / p.len() as f32;
    let recall = shared as f32 / g.len() as f32;
    2.0 * precision * recall / (precision + recall)
}

/// Fraction of gold supporting paragraphs whose file appeared in the trace's seen files,
/// matched by sanitized-title filename substring.
pub fn retrieval_recall(seen_files: &[String], supporting_titles: &[String]) -> f32 {
    if supporting_titles.is_empty() {
        return 1.0;
    }
    let hit = supporting_titles
        .iter()
        .filter(|t| {
            let stem = sanitize_title(t);
            seen_files.iter().any(|f| f.contains(&stem))
        })
        .count();
    hit as f32 / supporting_titles.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_drops_articles_punct_case() {
        assert_eq!(normalize("The Big-Apple!"), "big apple");
    }

    #[test]
    fn em_and_f1() {
        assert!(exact_match("the cat", "Cat"));
        assert!(!exact_match("cat", "dog"));
        // pred "quick brown" vs gold "the quick brown fox": shared=2, P=2/2=1.0, R=2/3=0.667, F1=0.8
        assert!((token_f1("quick brown", "the quick brown fox") - 0.8).abs() < 1e-3);
        assert_eq!(token_f1("cat", "dog"), 0.0);
    }

    #[test]
    fn retrieval_recall_matches_by_sanitized_title() {
        let seen = vec!["eval-corpus/Bob_Page.md".to_string()];
        assert!((retrieval_recall(&seen, &["Bob Page".into(), "Alice".into()]) - 0.5).abs() < 1e-6);
        assert_eq!(retrieval_recall(&seen, &["Bob Page".into()]), 1.0);
        assert_eq!(retrieval_recall(&[], &["Bob Page".into()]), 0.0);
    }
}
```

- [ ] **Step 2: Run → fail → add `mod score;` → pass** (`cargo test -p kb-eval score`). If a hand-computed F1 differs, recompute precisely and assert the exact value — do not weaken the test.

- [ ] **Step 3: Commit**
```bash
git add eval/src/score.rs eval/src/main.rs
git commit -m "feat(eval): HotpotQA scoring — normalize/EM/token-F1 + retrieval-recall"
```

---

### Task 6: `eval` — trace reader (window + seen files)

**Files:**
- Create: `eval/src/trace_read.rs`
- Modify: `eval/src/main.rs` (add `mod trace_read;`)
- Test: `eval/src/trace_read.rs` (inline)

**Interfaces:**
- Consumes: `glossa::trace::TraceEntry`.
- Produces:
  - `trace_read::read_window(traces_dir: &Path, t0_ms: u64, t1_ms: u64) -> anyhow::Result<Vec<glossa::trace::TraceEntry>>`
  - `trace_read::seen_files(entries: &[glossa::trace::TraceEntry]) -> Vec<String>`

- [ ] **Step 1: Write the failing test**

Create `eval/src/trace_read.rs`:
```rust
use glossa::trace::TraceEntry;
use std::path::Path;

pub fn read_window(traces_dir: &Path, t0_ms: u64, t1_ms: u64) -> anyhow::Result<Vec<TraceEntry>> {
    let mut out = Vec::new();
    let rd = match std::fs::read_dir(traces_dir) {
        Ok(rd) => rd,
        Err(_) => return Ok(out),
    };
    for ent in rd {
        let path = ent?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        for line in std::fs::read_to_string(&path)?.lines() {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(e) = serde_json::from_str::<TraceEntry>(line) {
                if e.ts_ms >= t0_ms && e.ts_ms <= t1_ms {
                    out.push(e);
                }
            }
        }
    }
    out.sort_by_key(|e| e.ts_ms);
    Ok(out)
}

/// Collect every `path` mentioned in search-result arrays and read results.
pub fn seen_files(entries: &[TraceEntry]) -> Vec<String> {
    let mut out = Vec::new();
    for e in entries {
        match &e.result {
            serde_json::Value::Array(arr) => {
                for v in arr {
                    if let Some(p) = v.get("path").and_then(|p| p.as_str()) {
                        out.push(p.to_string());
                    }
                }
            }
            serde_json::Value::Object(o) => {
                if let Some(p) = o.get("path").and_then(|p| p.as_str()) {
                    out.push(p.to_string());
                }
            }
            _ => {}
        }
    }
    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_by_time_and_extracts_paths() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.jsonl");
        std::fs::write(&p, concat!(
            r#"{"ts_ms":100,"tool":"search","args":{},"result":[{"path":"Bob_Page.md","location":"p.1"}]}"#, "\n",
            r#"{"ts_ms":500,"tool":"read","args":{},"result":{"path":"Alice.md"}}"#, "\n",
            r#"{"ts_ms":999,"tool":"search","args":{},"result":[{"path":"Late.md"}]}"#, "\n",
        )).unwrap();

        let win = read_window(dir.path(), 50, 600).unwrap();
        assert_eq!(win.len(), 2);
        let files = seen_files(&win);
        assert_eq!(files, vec!["Alice.md".to_string(), "Bob_Page.md".to_string()]);
    }

    #[test]
    fn missing_dir_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_window(&dir.path().join("nope"), 0, u64::MAX).unwrap().is_empty());
    }
}
```

- [ ] **Step 2: Run → fail → add `mod trace_read;` → pass** (`cargo test -p kb-eval trace_read`).

- [ ] **Step 3: Commit**
```bash
git add eval/src/trace_read.rs eval/src/main.rs
git commit -m "feat(eval): trace reader — time-window + seen-file extraction"
```

---

### Task 7: `eval` — corpus builder + prompt/answer helpers

**Files:**
- Create: `eval/src/corpus.rs`, `eval/src/backend/mod.rs`, `eval/src/backend/prompt.rs`
- Modify: `eval/src/main.rs` (add `mod corpus;`, `mod backend;`)
- Test: `eval/src/corpus.rs` + `eval/src/backend/prompt.rs` (inline)

**Interfaces:**
- Consumes: `dataset::{Question, sanitize_title}`.
- Produces:
  - `corpus::write_corpus(work: &Path, q: &dataset::Question) -> anyhow::Result<()>`
  - `corpus::index(work: &Path, kb_bin: &str) -> anyhow::Result<()>`
  - `backend::prompt::build_prompt(q: &dataset::Question) -> String`
  - `backend::prompt::parse_answer(model_output: &str) -> String`

- [ ] **Step 1: Write the failing tests**

Create `eval/src/corpus.rs`:
```rust
use crate::dataset::{sanitize_title, Question};
use anyhow::{bail, Context};
use std::path::Path;
use std::process::Command;

pub fn write_corpus(work: &Path, q: &Question) -> anyhow::Result<()> {
    if work.exists() {
        for ent in std::fs::read_dir(work)? {
            let p = ent?.path();
            if p.extension().and_then(|e| e.to_str()) == Some("md") {
                let _ = std::fs::remove_file(p);
            }
        }
        let _ = std::fs::remove_dir_all(work.join(".glossa"));
    } else {
        std::fs::create_dir_all(work)?;
    }
    for para in &q.paragraphs {
        let file = work.join(format!("{}.md", sanitize_title(&para.title)));
        let mut body = format!("# {}\n", para.title);
        for s in &para.sentences {
            body.push_str(s);
            body.push('\n');
        }
        std::fs::write(&file, body).with_context(|| format!("write {file:?}"))?;
    }
    Ok(())
}

pub fn index(work: &Path, kb_bin: &str) -> anyhow::Result<()> {
    let status = Command::new(kb_bin)
        .arg("index")
        .arg(work)
        .status()
        .with_context(|| format!("spawn {kb_bin} index"))?;
    if !status.success() {
        bail!("kb index failed for {work:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Paragraph;

    fn q() -> Question {
        Question {
            id: "q1".into(), question: "?".into(), answer: "a".into(),
            paragraphs: vec![Paragraph { title: "Bob Page".into(), sentences: vec!["b1.".into(), "b2.".into()] }],
            supporting_titles: vec!["Bob Page".into()],
        }
    }

    #[test]
    fn write_corpus_writes_md_and_clears_prior() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("stale.md"), b"old").unwrap();
        write_corpus(dir.path(), &q()).unwrap();
        assert!(!dir.path().join("stale.md").exists());
        let body = std::fs::read_to_string(dir.path().join("Bob_Page.md")).unwrap();
        assert!(body.contains("# Bob Page") && body.contains("b1.") && body.contains("b2."));
    }
}
```

Create `eval/src/backend/mod.rs`:
```rust
pub mod prompt;
```

Create `eval/src/backend/prompt.rs`:
```rust
use crate::dataset::Question;

pub fn build_prompt(q: &Question) -> String {
    format!(
        "You are answering a question using a document search tool (glossa MCP: `search`, `read`).\n\
         Search the indexed corpus, read what you need, then answer.\n\
         Output ONLY your final answer on a single line beginning with `ANSWER:`.\n\n\
         Question: {}",
        q.question
    )
}

/// Extract the answer after the last `ANSWER:` marker; if absent, the trimmed whole output.
pub fn parse_answer(model_output: &str) -> String {
    if let Some(idx) = model_output.rfind("ANSWER:") {
        model_output[idx + "ANSWER:".len()..].trim().lines().next().unwrap_or("").trim().to_string()
    } else {
        model_output.trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Question;

    #[test]
    fn parse_answer_takes_after_marker() {
        assert_eq!(parse_answer("thinking...\nANSWER: Bob Page\n"), "Bob Page");
        assert_eq!(parse_answer("ANSWER:  42 "), "42");
        assert_eq!(parse_answer("no marker here"), "no marker here");
    }

    #[test]
    fn build_prompt_includes_question_and_marker() {
        let q = Question { id: "x".into(), question: "Who?".into(), answer: "".into(), paragraphs: vec![], supporting_titles: vec![] };
        let p = build_prompt(&q);
        assert!(p.contains("Who?") && p.contains("ANSWER:"));
    }
}
```

- [ ] **Step 2: Run → fail → add `mod corpus;` and `mod backend;` → pass** (`cargo test -p kb-eval`). The `index` subprocess fn is exercised live / via the run integration test, not unit-tested here.

- [ ] **Step 3: Commit**
```bash
git add eval/src/corpus.rs eval/src/backend/ eval/src/main.rs
git commit -m "feat(eval): corpus builder + prompt/answer helpers"
```

---

### Task 8: `eval` — backends (mock, claude, qwen)

**Files:**
- Create: `eval/src/backend/mock.rs`, `eval/src/backend/claude.rs`, `eval/src/backend/qwen.rs`
- Modify: `eval/src/backend/mod.rs` (trait + submodules)
- Test: `eval/src/backend/mock.rs` (inline)

**Interfaces:**
- Consumes: `dataset::Question`, `backend::prompt`.
- Produces:
  - `backend::AgentBackend` trait: `fn needs_corpus(&self) -> bool;` `fn answer(&self, work: &Path, q: &Question) -> anyhow::Result<String>`.
  - `backend::mock::MockBackend { canned: HashMap<String,String> }` (needs_corpus=false).
  - `backend::claude::ClaudeBackend { kb_bin, profile, no_graph }` (needs_corpus=true).
  - `backend::qwen::QwenBackend { url, model }` (needs_corpus=true).

- [ ] **Step 1: Trait + mock (tested), claude, qwen**

Replace `eval/src/backend/mod.rs`:
```rust
pub mod prompt;
pub mod mock;
pub mod claude;
pub mod qwen;

use crate::dataset::Question;
use std::path::Path;

pub trait AgentBackend {
    fn needs_corpus(&self) -> bool;
    fn answer(&self, work: &Path, q: &Question) -> anyhow::Result<String>;
}
```

Create `eval/src/backend/mock.rs`:
```rust
use super::AgentBackend;
use crate::dataset::Question;
use std::collections::HashMap;
use std::path::Path;

pub struct MockBackend {
    pub canned: HashMap<String, String>,
}

impl AgentBackend for MockBackend {
    fn needs_corpus(&self) -> bool {
        false
    }
    fn answer(&self, _work: &Path, q: &Question) -> anyhow::Result<String> {
        Ok(self.canned.get(&q.id).cloned().unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Question;

    #[test]
    fn mock_returns_canned() {
        let mut canned = HashMap::new();
        canned.insert("q1".to_string(), "Bob Page".to_string());
        let b = MockBackend { canned };
        let q = Question { id: "q1".into(), question: "?".into(), answer: "".into(), paragraphs: vec![], supporting_titles: vec![] };
        assert_eq!(b.answer(Path::new("."), &q).unwrap(), "Bob Page");
        assert!(!b.needs_corpus());
    }
}
```

Create `eval/src/backend/claude.rs`:
```rust
use super::{prompt, AgentBackend};
use crate::dataset::Question;
use anyhow::Context;
use std::path::Path;
use std::process::Command;

/// Drives `claude -p` as a headless agent that is itself the glossa MCP client.
pub struct ClaudeBackend {
    pub kb_bin: String,
    pub profile: String,
    pub no_graph: bool,
}

impl AgentBackend for ClaudeBackend {
    fn needs_corpus(&self) -> bool {
        true
    }
    fn answer(&self, work: &Path, q: &Question) -> anyhow::Result<String> {
        let mut args = vec!["mcp".to_string(), "--profile".to_string(), self.profile.clone(), "--trace".to_string()];
        if self.no_graph {
            args.push("--no-graph".to_string());
        }
        args.push(work.display().to_string());
        let cfg = serde_json::json!({ "mcpServers": { "glossa": { "command": self.kb_bin, "args": args } } });
        let cfg_path = work.join(".claude-mcp.json");
        std::fs::write(&cfg_path, serde_json::to_string(&cfg)?)?;

        // NOTE: claude CLI flags are best-effort; verify against the installed version.
        let out = Command::new("claude")
            .arg("-p")
            .arg(prompt::build_prompt(q))
            .arg("--mcp-config")
            .arg(&cfg_path)
            .arg("--permission-mode")
            .arg("bypassPermissions")
            .output()
            .context("spawn claude -p")?;
        Ok(prompt::parse_answer(&String::from_utf8_lossy(&out.stdout)))
    }
}
```

Create `eval/src/backend/qwen.rs`:
```rust
use super::{prompt, AgentBackend};
use crate::dataset::Question;
use anyhow::{anyhow, Context};
use std::path::Path;

/// Drives a local model via LM Studio's OpenAI-compatible chat API. The operator configures LM Studio
/// (once) with the glossa MCP server so the model itself calls search/read.
pub struct QwenBackend {
    pub url: String,
    pub model: String,
}

impl AgentBackend for QwenBackend {
    fn needs_corpus(&self) -> bool {
        true
    }
    fn answer(&self, _work: &Path, q: &Question) -> anyhow::Result<String> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{ "role": "user", "content": prompt::build_prompt(q) }],
            "temperature": 0.0
        });
        let resp = ureq::post(&format!("{}/v1/chat/completions", self.url))
            .send_json(body)
            .map_err(|e| anyhow!("lmstudio request failed: {e}"))?;
        let v: serde_json::Value = resp.into_json().context("parse lmstudio json")?;
        let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
        Ok(prompt::parse_answer(content))
    }
}
```

- [ ] **Step 2: Run → fail → pass** (`cargo test -p kb-eval backend`; mock test is the gate). Then `cargo test`.

- [ ] **Step 3: Commit**
```bash
git add eval/src/backend/
git commit -m "feat(eval): agent backends — mock (tested), claude (-p), qwen (LM Studio)"
```

---

### Task 9: `eval` — run orchestration + report + end-to-end mock test

**Files:**
- Create: `eval/src/run.rs`
- Modify: `eval/src/main.rs` (add `mod run;`; wire `Cmd::Run`)
- Test: `eval/tests/mock_e2e.rs` (integration)

**Interfaces:**
- Produces:
  - `run::Row`, `run::Report` (both `Serialize`)
  - `run::run_eval(dataset_path: &Path, backend: &dyn backend::AgentBackend, backend_name: &str, limit: usize, kb_bin: &str, work: &Path) -> anyhow::Result<Report>`

- [ ] **Step 1: Write `run.rs`**
```rust
use crate::backend::AgentBackend;
use crate::{corpus, dataset, score, trace_read};
use glossa::trace::now_ms;
use serde::Serialize;
use std::path::Path;

#[derive(Serialize)]
pub struct Row {
    pub id: String,
    pub question: String,
    pub gold: String,
    pub pred: String,
    pub em: bool,
    pub f1: f32,
    pub retrieval_recall: f32,
    pub failed: Option<String>,
}

#[derive(Serialize)]
pub struct Report {
    pub backend: String,
    pub rows: Vec<Row>,
    pub em_mean: f32,
    pub f1_mean: f32,
    pub recall_mean: f32,
}

pub fn run_eval(
    dataset_path: &Path,
    backend: &dyn AgentBackend,
    backend_name: &str,
    limit: usize,
    kb_bin: &str,
    work: &Path,
) -> anyhow::Result<Report> {
    let json = std::fs::read_to_string(dataset_path)?;
    let mut questions = dataset::parse_hotpot(&json)?;
    if limit > 0 && questions.len() > limit {
        questions.truncate(limit);
    }
    let rows: Vec<Row> = questions.iter().map(|q| eval_one(backend, q, kb_bin, work)).collect();
    let n = rows.len().max(1) as f32;
    let em_mean = rows.iter().filter(|r| r.em).count() as f32 / n;
    let f1_mean = rows.iter().map(|r| r.f1).sum::<f32>() / n;
    let recall_mean = rows.iter().map(|r| r.retrieval_recall).sum::<f32>() / n;
    Ok(Report { backend: backend_name.to_string(), rows, em_mean, f1_mean, recall_mean })
}

fn eval_one(backend: &dyn AgentBackend, q: &dataset::Question, kb_bin: &str, work: &Path) -> Row {
    let base = Row {
        id: q.id.clone(), question: q.question.clone(), gold: q.answer.clone(),
        pred: String::new(), em: false, f1: 0.0, retrieval_recall: 0.0, failed: None,
    };
    if backend.needs_corpus() {
        if let Err(e) = corpus::write_corpus(work, q).and_then(|_| corpus::index(work, kb_bin)) {
            return Row { failed: Some(format!("corpus: {e}")), ..base };
        }
    }
    let t0 = now_ms();
    let pred = match backend.answer(work, q) {
        Ok(p) => p,
        Err(e) => return Row { failed: Some(format!("backend: {e}")), ..base },
    };
    let t1 = now_ms();
    let recall = if backend.needs_corpus() {
        let dir = work.join(".glossa").join("traces");
        let entries = trace_read::read_window(&dir, t0, t1).unwrap_or_default();
        score::retrieval_recall(&trace_read::seen_files(&entries), &q.supporting_titles)
    } else {
        0.0
    };
    Row {
        em: score::exact_match(&pred, &q.answer),
        f1: score::token_f1(&pred, &q.answer),
        retrieval_recall: recall,
        pred,
        ..base
    }
}
```

- [ ] **Step 2: Wire `Cmd::Run`** (`eval/src/main.rs`)

Declare the remaining modules (`mod run; mod corpus; mod score; mod trace_read; mod dataset; mod backend;` — those not already present) and implement the arm:
```rust
        Cmd::Run { dataset, backend, limit, lmstudio_url, kb_bin, work } => {
            use backend::AgentBackend;
            let be: Box<dyn AgentBackend> = match backend {
                BackendKind::Mock => Box::new(backend::mock::MockBackend { canned: std::collections::HashMap::new() }),
                BackendKind::Qwen => Box::new(backend::qwen::QwenBackend { url: lmstudio_url, model: "local-model".to_string() }),
                BackendKind::Claude => Box::new(backend::claude::ClaudeBackend { kb_bin: kb_bin.clone(), profile: "editor".to_string(), no_graph: false }),
            };
            let name = format!("{backend:?}").to_lowercase();
            let report = run::run_eval(&dataset, be.as_ref(), &name, limit, &kb_bin, &work)?;
            let json_path = format!("eval-{}-{}.json", report.backend, glossa::trace::now_ms());
            std::fs::write(&json_path, serde_json::to_string_pretty(&report)?)?;
            println!(
                "backend={} questions={} EM={:.3} F1={:.3} retrieval_recall={:.3}\nwrote {}",
                report.backend, report.rows.len(), report.em_mean, report.f1_mean, report.recall_mean, json_path
            );
            Ok(())
        }
```

- [ ] **Step 3: End-to-end mock integration test** (`eval/tests/mock_e2e.rs`)
```rust
use assert_cmd::Command;
use predicates::prelude::PredicateBooleanExt;
use predicates::str::contains;
use std::fs;

#[test]
fn mock_run_scores_and_reports() {
    let dir = tempfile::tempdir().unwrap();
    let ds = dir.path().join("ds.json");
    fs::write(&ds, r#"[
      {"_id":"q1","question":"Who?","answer":"Bob Page","context":[["Bob Page",["b1."]]],"supporting_facts":[["Bob Page",0]]},
      {"_id":"q2","question":"What?","answer":"42","context":[["N",["n1."]]],"supporting_facts":[["N",0]]}
    ]"#).unwrap();

    Command::cargo_bin("kb-eval").unwrap()
        .current_dir(dir.path())
        .args(["run", "--dataset", ds.to_str().unwrap(), "--backend", "mock"])
        .assert()
        .success()
        .stdout(contains("backend=mock").and(contains("questions=2")).and(contains("EM=")));

    let wrote = fs::read_dir(dir.path()).unwrap()
        .filter_map(|e| e.ok())
        .any(|e| e.file_name().to_string_lossy().starts_with("eval-mock-"));
    assert!(wrote, "a report JSON should be written");
}
```

- [ ] **Step 4: Run → fail → pass + full suite**

`cargo test -p kb-eval --test mock_e2e`, then `cargo test` (workspace green). `cargo tree -p glossa -i cc` → empty.

- [ ] **Step 5: Commit**
```bash
git add eval/src/run.rs eval/src/main.rs eval/tests/mock_e2e.rs
git commit -m "feat(eval): run orchestration + report + mock end-to-end test"
```

---

## Self-Review

**Spec coverage:** workspace (Task 3); `--trace` JSONL + `--no-graph` (Tasks 1–2); dataset (4); scorer (5); trace reader (6); corpus (7); backends incl. mock-tested + claude/qwen (7 prompt, 8); run/report + deterministic mock e2e (9); sequential time-window correlation (9 `eval_one`); pure-Rust harness via `ureq` no-TLS + subprocess (3). Deferred items appear in no task. ✓

**Placeholder scan:** none — complete code + commands in every step. Claude CLI flags flagged "verify against installed version"; the claude backend is operator-run and is not the CI gate (mock e2e + unit tests are).

**Type consistency:** `dataset::{Question,Paragraph,sanitize_title}` (T4) used by T5/T7/T9; `glossa::trace::{TraceEntry,TraceLog,now_ms}` (T1) used by T2/T6/T9; `backend::AgentBackend{needs_corpus,answer}` (T8) implemented by mock/claude/qwen, consumed by T9; `score::{normalize,exact_match,token_f1,retrieval_recall}` (T5) used by T9; `corpus::{write_corpus,index}` (T7) used by T9; `trace_read::{read_window,seen_files}` (T6) used by T9; `prompt::{build_prompt,parse_answer}` (T7) used by T8; CLI `BackendKind` derives `Debug` (T3) for the report name (T9).
