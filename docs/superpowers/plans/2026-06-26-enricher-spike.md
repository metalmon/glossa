# Enricher Spike (Task 0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire up a `graph_upsert`-capable enricher agent (qwen3.5-4B via TensorZero) that reverse-traces solved support cases into reasoning-graph edges, then verify the whole thing compiles.

**Architecture:** A new `enrich`-specific exec closure in `eval/src/enrich.rs` handles `"graph_upsert"` locally (parse → validate → `apply_upsert`) and delegates every other tool name to the existing `glossa_tools::exec`. This keeps the shared `exec` signature untouched. The `kb-eval enrich` subcommand wires it into `run_episode` via the same TZ HTTP pattern as `TensorZeroBackend::answer`. Steps 5–6 (running the spike + eyeballing the graph) are the controller's job; this plan stops at `cargo build`.

**Tech Stack:** Rust (edition 2021), `glossa` crate, `ureq` v2, `serde_json`, `uuid` v1 (v7), `clap` v4, TensorZero TOML config.

## Global Constraints

- Branch: `feat/semantic-enricher` — do NOT switch or create branches.
- Working directory: `E:\glossa`.
- Do NOT change the signature of `glossa_tools::exec`.
- Do NOT restart the TZ gateway, do NOT run `kb-eval enrich`.
- Build target: `cargo build -p kb-eval --release` must produce zero errors.
- Commit message exactly: `feat(eval): enricher runner + graph_upsert tool + enrich function (spike code)`.
- Report written to: `E:\glossa\.superpowers\sdd\task-0-report.md`.

---

## Verified shapes (read from source before coding)

```
// src/graph/agent.rs
pub struct NodeSpec {
    pub id: String,
    pub node_type: String,
    pub label: String,
    pub aliases: Vec<String>,          // #[serde(default)]
    pub source_path: String,
    pub range: Option<String>,         // #[serde(default)]
    pub confidence: Option<f32>,       // #[serde(default)]
}

pub struct EdgeSpec {
    pub from: String,
    pub to: String,
    pub edge_type: String,
    pub source_path: String,
    pub range: Option<String>,         // #[serde(default)]
    pub confidence: Option<f32>,       // #[serde(default)]
}

pub fn apply_upsert(
    g: &GraphStore,
    ont: &Ontology,
    nodes: Vec<NodeSpec>,
    edges: Vec<EdgeSpec>,
    now: u64,
) -> anyhow::Result<(usize, usize)>    // (#nodes_upserted, #edges_upserted)

// src/graph/ontology.rs
pub fn load_or_default(root: &std::path::Path) -> Ontology

// eval/src/backend/tensorzero.rs  (pub items we reuse)
pub struct TzTurn { pub content: Vec<Value>, pub episode_id: String }
pub struct EpisodeOutcome { pub answer: String, pub episode_id: Option<String>, pub surfaced_titles: Vec<String> }
pub fn run_episode<C, X>(chat: C, user_question: &str, exec: X, max_rounds: usize) -> anyhow::Result<EpisodeOutcome>
  where C: FnMut(&[Value], Option<&str>) -> anyhow::Result<TzTurn>,
        X: Fn(&str, &Value) -> (String, Vec<String>, Vec<glossa::read::DocImage>) + Sync
// NOTE: backdated_episode_id and MAX_ROUNDS are private — define locally in enrich.rs
```

Train JSON field note: the actual `kb-val/derived/train.json` uses `"answer"`, not `"gold"`. The Serde struct below uses `answer`.

---

## File map

| File | Action | Responsibility |
|------|--------|----------------|
| `E:\glossa\kb-test\ontology.toml` | **Create** (copy) | Ontology loaded by `load_or_default` during enrichment |
| `E:\glossa\eval\tensorzero\config\tools\graph_upsert.json` | **Create** | TZ tool schema for `graph_upsert` |
| `E:\glossa\eval\tensorzero\config\tensorzero.toml` | **Modify** | Add `[tools.graph_upsert]`, `[functions.enrich]`, `[functions.enrich.variants.baseline]` |
| `E:\glossa\eval\tensorzero\config\enrich\system.minijinja` | **Create** | System prompt directing qwen to reverse-trace and build graph nodes |
| `E:\glossa\eval\src\enrich.rs` | **Create** | `run_enrich()`: enrich-specific exec closure + episode loop |
| `E:\glossa\eval\src\main.rs` | **Modify** | Add `mod enrich;` + `Cmd::Enrich` variant + match arm |
| `E:\glossa\.superpowers\sdd\task-0-report.md` | **Create** | Report for controller (Step 5+ command, build result, concerns) |

---

## Task 1: Deploy ontology

**Files:**
- Create: `E:\glossa\kb-test\ontology.toml` (copy of `E:\glossa\eval\ontology-support.toml`)

- [ ] **Step 1: Copy the ontology file**

```powershell
Copy-Item "E:\glossa\eval\ontology-support.toml" "E:\glossa\kb-test\ontology.toml"
```

- [ ] **Step 2: Verify the copy landed**

```powershell
Get-Item "E:\glossa\kb-test\ontology.toml"
```

Expected: file exists, non-zero size.

- [ ] **Step 3: Commit**

```powershell
git -C E:\glossa add kb-test/ontology.toml
git -C E:\glossa commit -m "feat(eval): deploy support ontology to kb-test corpus root"
```

---

## Task 2: `graph_upsert` tool schema + TZ config additions

**Files:**
- Create: `E:\glossa\eval\tensorzero\config\tools\graph_upsert.json`
- Modify: `E:\glossa\eval\tensorzero\config\tensorzero.toml`
- Create: `E:\glossa\eval\tensorzero\config\enrich\system.minijinja`

### Step 1 — Write `graph_upsert.json`

The schema mirrors `GraphUpsertArgs` in `src/mcp.rs`. NodeSpec omits `range`/`confidence` (optional, model rarely sets them). EdgeSpec likewise.

- [ ] **Step 1: Create `E:\glossa\eval\tensorzero\config\tools\graph_upsert.json`**

```json
{
  "type": "object",
  "properties": {
    "nodes": {
      "type": "array",
      "description": "Nodes to upsert into the reasoning graph.",
      "items": {
        "type": "object",
        "required": ["id", "node_type", "label", "source_path"],
        "properties": {
          "id":          { "type": "string", "description": "Stable slug, e.g. sym:profibus-link-loss" },
          "node_type":   { "type": "string", "description": "One of: Symptom, Cause, Resolution, Task" },
          "label":       { "type": "string", "description": "Short human label — broad problem/fix class, NOT the literal answer text" },
          "aliases":     { "type": "array", "items": { "type": "string" }, "description": "Alternative names / synonyms" },
          "source_path": { "type": "string", "description": "Document path (relative to corpus root)" }
        }
      }
    },
    "edges": {
      "type": "array",
      "description": "Edges to upsert. Allowed types: CAUSED_BY, RESOLVED_BY, MENTIONS.",
      "items": {
        "type": "object",
        "required": ["from", "to", "edge_type", "source_path"],
        "properties": {
          "from":        { "type": "string" },
          "to":          { "type": "string" },
          "edge_type":   { "type": "string", "description": "One of: CAUSED_BY, RESOLVED_BY, MENTIONS" },
          "source_path": { "type": "string" }
        }
      }
    }
  },
  "required": ["nodes", "edges"]
}
```

- [ ] **Step 2: Create enrich prompt directory**

```powershell
New-Item -ItemType Directory -Force "E:\glossa\eval\tensorzero\config\enrich"
```

- [ ] **Step 3: Create `E:\glossa\eval\tensorzero\config\enrich\system.minijinja`**

```
You build a reusable reasoning graph from a SOLVED support case.

You are given a question and its KNOWN correct answer. Your job is NOT to answer the question — it is already answered. Your job is to record the GENERAL reasoning pattern so a FUTURE similar question can be routed to the right document sections.

## Workflow

1. Use `search`, `grep`, `read` to locate the document section(s) the known answer comes from.
2. Abstract BROADLY — not this specific machine or site, but the class of problem:
   - A `Symptom` node: the broad problem class (e.g. "PROFIBUS DP periodic link loss / watchdog timeout"), NOT "Сгущение №3 ШВВП ×3".
   - A `Resolution` node: the fix pattern (e.g. "increase maxTsdr, recalculate watchdog"), NOT the specific value "3000 tbit".
   - Optionally a `Cause` node: the root cause class.
3. Call `graph_upsert` with:
   - The Symptom, Resolution (and optional Cause) nodes.
   - Edges: `Symptom →CAUSED_BY→ Cause`, `Cause →RESOLVED_BY→ Resolution` (or `Symptom →RESOLVED_BY→ Resolution`).
   - `MENTIONS` edges from each Symptom/Cause/Resolution node to the Section node at the supporting document path (use `source_path = "<path>"` — the structural Section node already exists from indexing).
   - Node ids: short stable slugs, e.g. `sym:profibus-link-loss`, `res:increase-maxtsdr`.

## Hard rules

- **NEVER** put the literal answer text in a label — store routing, not the answer.
- **NEVER** put case-specific details (machine name, site, unit number) in id or label.
- Allowed node types: `Symptom`, `Cause`, `Resolution`, `Task`.
- Allowed edge types: `CAUSED_BY`, `RESOLVED_BY`, `MENTIONS`.
- `graph_upsert` will reject invalid types — read the error and retry with corrected types.

When done, confirm how many nodes and edges were upserted.
```

- [ ] **Step 4: Add `[tools.graph_upsert]`, `[functions.enrich]` to `tensorzero.toml`**

Open `E:\glossa\eval\tensorzero\config\tensorzero.toml`. Locate the line:

```
# <<< GENERATED TOOLS
```

Insert the following block **after** that line (i.e., between `# <<< GENERATED TOOLS` and `# ── Feedback metrics`):

```toml
# ── Enricher tool (hand-written for spike; generator integration is Task 1) ──
[tools.graph_upsert]
description = "Upsert reasoning-graph nodes and edges validated against the support ontology. Returns 'upserted N nodes, M edges' on success, or a validation error (so you can self-correct). Allowed node types: Symptom, Cause, Resolution, Task. Allowed edge types: CAUSED_BY, RESOLVED_BY, MENTIONS."
parameters = "tools/graph_upsert.json"

# ── Enricher function: builds the reasoning graph from a solved train case ──
[functions.enrich]
type = "chat"
tools = ["search", "read", "grep", "graph_upsert"]
parallel_tool_calls = true

[functions.enrich.variants.baseline]
type = "chat_completion"
model = "qwen"
system_template = "enrich/system.minijinja"
max_tokens = 2048
temperature = 0.8
top_p = 0.95
extra_body = [
  { pointer = "/top_k", value = 40 },
  { pointer = "/min_p", value = 0.1 },
  { pointer = "/repetition_penalty", value = 1.1 },
]
```

- [ ] **Step 5: Commit config additions**

```powershell
git -C E:\glossa add eval/tensorzero/config/tools/graph_upsert.json
git -C E:\glossa add eval/tensorzero/config/enrich/system.minijinja
git -C E:\glossa add eval/tensorzero/config/tensorzero.toml
git -C E:\glossa commit -m "feat(eval): add graph_upsert tool schema + enrich TZ function config"
```

---

## Task 3: `eval/src/enrich.rs` — enrich-specific exec closure + runner

**Files:**
- Create: `E:\glossa\eval\src\enrich.rs`

The key design:
- Build an enrich-specific closure that handles `"graph_upsert"` locally, delegates everything else to `glossa_tools::exec`.
- `glossa_tools::exec` signature is **not changed**: `exec(name, args, idx, graph, trace)`.
- `MAX_ROUNDS` and `backdated_episode_id` are private in `tensorzero.rs`; define them locally.
- Use `Arc<AtomicUsize>` to accumulate upserted counts across parallel tool calls (the exec closure may be called from multiple threads by `run_episode`).

- [ ] **Step 1: Create `E:\glossa\eval\src\enrich.rs`**

```rust
//! Enricher runner: for each solved training case, drive the `enrich` TZ function
//! to reverse-trace the answer into reasoning-graph nodes/edges.
//!
//! Design: we build a case-local exec closure that handles `graph_upsert` in-process
//! (parse → ontology-validate → apply_upsert) and delegates every other tool to
//! `glossa_tools::exec`. This keeps the shared exec signature untouched.

use anyhow::Context;
use glossa::graph::agent::{apply_upsert, EdgeSpec, NodeSpec};
use glossa::graph::ontology::Ontology;
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;
use glossa::trace::TraceLog;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;

use crate::backend::glossa_tools;
use crate::backend::tensorzero::{run_episode, EpisodeOutcome, TzTurn};

/// Cap on tool-call rounds per episode (mirrors the private constant in tensorzero.rs).
const MAX_ROUNDS: usize = 50;

/// One entry from the training JSON.
/// The file uses `"answer"` for the known correct answer (not `"gold"`).
#[derive(Debug, Deserialize)]
struct TrainCase {
    #[serde(rename = "_id")]
    id: String,
    question: String,
    answer: String,
}

/// Drive the enricher for up to `limit` cases from `train_path`.
/// Opens the index + graph at `work` once per case.
pub fn run_enrich(
    train_path: &Path,
    work: &Path,
    limit: usize,
    tz_endpoint: &str,
    tz_function: &str,
) -> anyhow::Result<()> {
    let data = std::fs::read_to_string(train_path)
        .with_context(|| format!("read train json: {}", train_path.display()))?;
    let mut cases: Vec<TrainCase> =
        serde_json::from_str(&data).context("parse train json")?;
    if limit > 0 {
        cases.truncate(limit);
    }
    println!("enriching {} cases from {}", cases.len(), train_path.display());

    for case in &cases {
        let result = enrich_case(case, work, tz_endpoint, tz_function);
        match result {
            Ok((n, e)) => println!("[{}] upserted {} nodes, {} edges", case.id, n, e),
            Err(err) => eprintln!("[{}] ERROR: {err:#}", case.id),
        }
    }
    Ok(())
}

/// Run one enrichment episode and return (#nodes_upserted, #edges_upserted).
fn enrich_case(
    case: &TrainCase,
    work: &Path,
    tz_endpoint: &str,
    tz_function: &str,
) -> anyhow::Result<(usize, usize)> {
    // Open index and graph fresh per case (cheap; avoids cross-case state).
    let idx = DocIndex::open_or_create(work).context("open DocIndex")?;
    let graph = GraphStore::open(work).context("open GraphStore")?;
    let trace = TraceLog::disabled();
    let work_buf: PathBuf = work.to_path_buf();

    // Atomic counters shared between the exec closure and the caller.
    // Arc because run_episode may invoke exec from multiple threads in parallel.
    let nodes_count = Arc::new(AtomicUsize::new(0));
    let edges_count = Arc::new(AtomicUsize::new(0));
    let nc = Arc::clone(&nodes_count);
    let ec = Arc::clone(&edges_count);

    // Enrich-specific exec: handles graph_upsert locally, delegates the rest.
    let exec = move |name: &str, args: &Value| -> (String, Vec<String>, Vec<glossa::read::DocImage>) {
        if name == "graph_upsert" {
            let nodes: Vec<NodeSpec> = serde_json::from_value(
                args.get("nodes").cloned().unwrap_or(json!([])),
            )
            .unwrap_or_default();
            let edges: Vec<EdgeSpec> = serde_json::from_value(
                args.get("edges").cloned().unwrap_or(json!([])),
            )
            .unwrap_or_default();
            let ont = Ontology::load_or_default(&work_buf);
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            match apply_upsert(&graph, &ont, nodes, edges, now) {
                Ok((n, e)) => {
                    nc.fetch_add(n, Ordering::Relaxed);
                    ec.fetch_add(e, Ordering::Relaxed);
                    (format!("upserted {n} nodes, {e} edges"), vec![], vec![])
                }
                Err(err) => (err.to_string(), vec![], vec![]),
            }
        } else {
            glossa_tools::exec(name, args, &idx, Some(&graph), &trace)
        }
    };

    // Chat closure: one HTTP call to TZ /inference per turn.
    // Use a per-case UUID v7 as episode_id so all turns group in TZ telemetry.
    let eid = uuid::Uuid::now_v7().to_string();
    let url = format!("{}/inference", tz_endpoint.trim_end_matches('/'));
    let function = tz_function.to_string();
    let chat = move |messages: &[Value], _episode_id: Option<&str>| -> anyhow::Result<TzTurn> {
        let body = json!({
            "function_name": function,
            "input": { "messages": messages },
            "episode_id": eid,
        });
        let text = ureq::post(&url)
            .set("Content-Type", "application/json")
            .timeout(Duration::from_secs(300))
            .send_string(&serde_json::to_string(&body)?)
            .context("TZ /inference request")?
            .into_string()
            .context("read TZ response")?;
        let v: Value = serde_json::from_str(&text).context("parse TZ JSON")?;
        if let Some(err) = v.get("error") {
            anyhow::bail!("tensorzero error: {err}");
        }
        let content = v
            .get("content")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();
        let episode_id = v
            .get("episode_id")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        Ok(TzTurn { content, episode_id })
    };

    let user_prompt = format!(
        "Question: {}\nKnown correct answer: {}\nBuild the reusable reasoning graph for this case.",
        case.question, case.answer
    );

    let _outcome: EpisodeOutcome = run_episode(chat, &user_prompt, exec, MAX_ROUNDS)?;

    Ok((
        nodes_count.load(Ordering::Relaxed),
        edges_count.load(Ordering::Relaxed),
    ))
}
```

- [ ] **Step 2: Verify the file was written**

```powershell
Get-Item "E:\glossa\eval\src\enrich.rs"
```

Expected: file exists, size > 2 KB.

---

## Task 4: Add `kb-eval enrich` subcommand to `main.rs`

**Files:**
- Modify: `E:\glossa\eval\src\main.rs`

There are two edits: (a) add `mod enrich;` near the top, (b) add `Enrich` variant to `Cmd` enum + match arm.

- [ ] **Step 1: Add `mod enrich;` after existing mod declarations**

In `E:\glossa\eval\src\main.rs`, find the block of `mod` declarations (lines ~6–12):

```rust
mod backend;
mod corpus;
mod dataset;
mod prep;
mod run;
mod score;
mod trace_read;
```

Replace with:

```rust
mod backend;
mod corpus;
mod dataset;
mod enrich;
mod prep;
mod run;
mod score;
mod trace_read;
```

- [ ] **Step 2: Add `Enrich` variant to the `Cmd` enum**

Find the last variant of the `Cmd` enum (before the closing `}`). The existing last variant looks like:

```rust
    },
}
```

Add the `Enrich` variant before the closing `}` of the enum. Locate the line that ends the last existing variant (e.g., after `max_shards: Option<usize>,` and its closing `},`):

```rust
    /// Enrich the knowledge-graph from solved training cases via the `enrich` TZ function.
    Enrich {
        /// Path to the training JSON (array of {_id, question, answer}).
        #[arg(long)]
        train: std::path::PathBuf,
        /// Corpus root (index + graph live here).
        #[arg(long, default_value = "kb-test")]
        work: std::path::PathBuf,
        /// Enrich the first N cases (0 = all).
        #[arg(long, default_value_t = 4)]
        limit: usize,
        /// TensorZero gateway base URL.
        #[arg(long, default_value = "http://localhost:3000")]
        tensorzero_endpoint: String,
        /// TensorZero function name.
        #[arg(long, default_value = "enrich")]
        tensorzero_function: String,
    },
```

- [ ] **Step 3: Add `Cmd::Enrich` match arm**

In the `fn main()` match block, after the last existing arm (before `}` closing the `match`), add:

```rust
        Cmd::Enrich { train, work, limit, tensorzero_endpoint, tensorzero_function } => {
            enrich::run_enrich(&train, &work, limit, &tensorzero_endpoint, &tensorzero_function)?;
        }
```

- [ ] **Step 4: Commit source additions**

```powershell
git -C E:\glossa add eval/src/enrich.rs eval/src/main.rs
git -C E:\glossa commit -m "feat(eval): add enrich.rs runner + kb-eval enrich subcommand"
```

---

## Task 5: Build and report

**Files:**
- Create: `E:\glossa\.superpowers\sdd\task-0-report.md`

- [ ] **Step 1: Stop any running kb-eval process (should be none)**

```powershell
# Check for stray kb-eval processes
Get-Process -Name "kb-eval" -ErrorAction SilentlyContinue
# If any: Stop-Process -Name "kb-eval" -Force
```

- [ ] **Step 2: Run the build**

```powershell
cargo build -p kb-eval --release 2>&1 | Tee-Object -FilePath E:\glossa\.superpowers\sdd\build-output.txt
```

Expected last line: `Finished release [optimized] target(s) in ...`

If the build fails, diagnose from the output. Common issues:
- Missing `pub` on `TzTurn`/`EpisodeOutcome`/`run_episode` in `tensorzero.rs` → add `pub` there.
- `GraphStore` not `Sync` → wrap in `Arc<Mutex<...>>` in `enrich.rs`.
- `ureq` timeout method name wrong → check `ureq` v2 API (`ureq::post(...).timeout(Duration)` is correct for v2.x).

- [ ] **Step 3: Write task-0-report.md**

Create `E:\glossa\.superpowers\sdd\task-0-report.md` with content based on actual build output:

```markdown
# Task 0 Spike — Implementation Report

## Status
[PASS / FAIL — fill in from build output]

## Commit hash
[`git -C E:\glossa rev-parse HEAD` output]

## Build result
`cargo build -p kb-eval --release` — [Finished / error summary]

## Files changed
- `kb-test/ontology.toml` — deployed support ontology (copy of eval/ontology-support.toml)
- `eval/tensorzero/config/tools/graph_upsert.json` — TZ tool schema
- `eval/tensorzero/config/tensorzero.toml` — added [tools.graph_upsert], [functions.enrich], [functions.enrich.variants.baseline]
- `eval/tensorzero/config/enrich/system.minijinja` — enricher system prompt
- `eval/src/enrich.rs` — run_enrich() + enrich-specific exec closure
- `eval/src/main.rs` — Cmd::Enrich subcommand

## Verified shapes
apply_upsert(g: &GraphStore, ont: &Ontology, nodes: Vec<NodeSpec>, edges: Vec<EdgeSpec>, now: u64) -> anyhow::Result<(usize, usize)>

NodeSpec { id, node_type, label, aliases: Vec<String>, source_path, range: Option<String>, confidence: Option<f32> }
EdgeSpec { from, to, edge_type, source_path, range: Option<String>, confidence: Option<f32> }

Ontology::load_or_default(root: &std::path::Path) -> Ontology

## Enrich prompt (eval/tensorzero/config/enrich/system.minijinja)
Reverse-trace SOLVED case → abstract broad Symptom/Cause/Resolution nodes → MENTIONS to Section nodes.
Hard rules: never put literal answer text in labels; allowed types Symptom/Cause/Resolution/Task; allowed edges CAUSED_BY/RESOLVED_BY/MENTIONS.

## Step 5 command (for controller)
```
# 1. Restart TZ gateway to load the new enrich function:
docker compose -f E:/glossa/eval/tensorzero/docker-compose.yml restart gateway

# 2. Run enricher on first 4 cases:
E:/glossa/target/release/kb-eval enrich \
  --train E:/glossa/kb-val/derived/train.json \
  --work E:/glossa/kb-test \
  --limit 4 \
  --tensorzero-endpoint http://localhost:3000 \
  --tensorzero-function enrich
```

## Concerns / watch points
- Train JSON uses field `"answer"` (not `"gold"`) — the `TrainCase` struct reads `answer`.
- `backdated_episode_id` is private in tensorzero.rs; enrich.rs uses `uuid::Uuid::now_v7()` directly (no backdating — telemetry grouping only, no functional impact).
- `MAX_ROUNDS` is a private const (50); redefined locally in enrich.rs.
- graph_upsert is NOT in the `# >>> GENERATED TOOLS` block; it was placed after `# <<< GENERATED TOOLS` so the generator won't clobber it.
- If qwen3.5-4B leaks answer text into node labels, the spec calls for STOP and prompt rethink. Eyeball the graph after Step 5.
```

- [ ] **Step 4: Final commit**

```powershell
git -C E:\glossa add .superpowers/sdd/task-0-report.md
git -C E:\glossa commit -m "feat(eval): enricher runner + graph_upsert tool + enrich function (spike code)"
```

---

## Self-review

**Spec coverage:**
- [x] Step 1: ontology deploy → Task 1
- [x] Step 2: graph_upsert arm in exec (in-process, not changing shared exec) → Task 3 `enrich.rs`
- [x] Step 3: TZ function + tool config → Task 2
- [x] Step 4: `kb-eval enrich` subcommand → Tasks 3+4
- [x] Build only (no gateway restart, no run) → Task 5 Step 2
- [x] Commit with exact message → Task 5 Step 4
- [x] Report to task-0-report.md → Task 5 Step 3
- [x] enrich prompt: broad abstraction, no answer text, MENTIONS to Section, allowed types/edges → Task 2 Step 3

**Placeholder scan:** No TBD/TODO/placeholder text in code blocks. All signatures and field names match verified source.

**Type consistency:**
- `NodeSpec`/`EdgeSpec` fields used in `enrich.rs` match `src/graph/agent.rs` exactly.
- `apply_upsert` 5-arg signature used correctly.
- `run_episode` generic bounds satisfied: `exec` is `Fn(&str, &Value) -> (String, Vec<String>, Vec<DocImage>) + Sync` (the `Arc<AtomicUsize>` captures are `Sync`; `GraphStore`, `DocIndex`, `TraceLog` are `Sync`).
- `TzTurn { content: Vec<Value>, episode_id: String }` — fields used correctly.
