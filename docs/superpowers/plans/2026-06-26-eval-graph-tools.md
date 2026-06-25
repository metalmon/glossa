# Eval Graph Tools (glossary + neighbors) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Give the kb-eval agent the full **Reader-profile** MCP toolset by adding the two missing read-only graph tools — `glossary` and `neighbors` — wired through the SAME shared `glossa::tools` functions the MCP server uses (eval ≡ prod by construction).

**Architecture:** Add `glossa::tools::{glossary, neighbors}` (shared model-facing render, like `search`/`grep`/`glob`). MCP's `glossary`/`neighbors` tools call them; the eval's `exec` calls them too, given a `GraphStore` opened at the eval's working root. Register both tools in the TensorZero config + the `answer_hotpot` tool list + system prompt.

**Tech Stack:** Rust, `glossa::graph::store::GraphStore` (`resolve`), `glossa::graph::traverse::neighbors`, TensorZero tool config (JSON schema + toml).

**Excluded by design (NOT in scope):** the mutation/admin tools `index`, `reindex`, `purge`, `graph_upsert`, and `resolve` (a duplicate of `glossary`). They are outside the Reader profile the production answering agent runs, and in the eval they would let the agent rebuild/wipe its own index mid-run.

## Global Constraints

- **Pure-Rust, C-free** (`cargo tree -p glossa -i cc` empty): no new deps; `GraphStore`/`traverse::neighbors`/`resolve` already exist.
- **Single source of truth:** MCP and eval MUST render glossary/neighbors via the SAME `glossa::tools::*` functions — no reimplementation in the eval.
- **Empty-result sentinels** consistent with the existing tools: glossary → `"(no matches)"`, neighbors → `"(no neighbors)"`.
- While the MCP host holds `kb.exe`, build the eval with `cargo build -p kb-eval --release` (not the whole workspace).
- TDD. Frequent commits.

---

### Task 1: `glossa::tools::{glossary, neighbors}` shared render + MCP parity

**Files:**
- Modify: `src/tools.rs` (add two functions + tests)
- Modify: `src/mcp.rs` (glossary/neighbors tools call the shared functions)

**Interfaces:**
- Consumes: `glossa::graph::store::GraphStore::resolve(&self, name: &str) -> anyhow::Result<Vec<String>>`; `glossa::graph::traverse::neighbors(g: &GraphStore, from: &str, edge_types: Option<&[String]>, depth: usize) -> anyhow::Result<Vec<String>>`.
- Produces: `glossa::tools::glossary(g: &GraphStore, name: &str, trace: &TraceLog) -> String`; `glossa::tools::neighbors(g: &GraphStore, node_id: &str, depth: usize, trace: &TraceLog) -> String`.

- [ ] **Step 1: Add the two functions to `src/tools.rs`** (append after `glob`, matching the existing match/trace/sentinel style)

```rust
/// Resolve a name/term to graph node ids (the "glossary" lookup). Model text only.
pub fn glossary(g: &crate::graph::store::GraphStore, name: &str, trace: &TraceLog) -> String {
    match g.resolve(name) {
        Ok(ids) => {
            trace.log("glossary", json!({ "name": name }), json!({ "ids": ids.len() }));
            if ids.is_empty() { "(no matches)".to_string() } else { ids.join("\n") }
        }
        Err(e) => format!("glossary error: {e}"),
    }
}

/// Graph neighbors reachable from a node id, up to `depth` hops. Model text only.
pub fn neighbors(g: &crate::graph::store::GraphStore, node_id: &str, depth: usize, trace: &TraceLog) -> String {
    match crate::graph::traverse::neighbors(g, node_id, None, depth) {
        Ok(ids) => {
            trace.log("neighbors", json!({ "node_id": node_id, "depth": depth }), json!({ "ids": ids.len() }));
            if ids.is_empty() { "(no neighbors)".to_string() } else { ids.join("\n") }
        }
        Err(e) => format!("neighbors error: {e}"),
    }
}
```

- [ ] **Step 2: Add tests to `src/tools.rs`** (build a tiny graph via `upsert`, mirroring the pattern in `src/graph/store.rs` test `upsert_validates_and_resolve_finds_by_alias` near line 266). The test must: upsert an Org node `acme` with alias `"ACME"` + a Document node + an edge, open/resolve, and assert `glossary(&g, "ACME", &disabled)` contains the node id, and `glossary(&g, "nonesuch", &disabled) == "(no matches)"`; and `neighbors(&g, "<from-id>", 1, &disabled)` contains the connected node id, `neighbors(&g, "isolated", 1, &disabled) == "(no neighbors)"`. Use `TraceLog::disabled()` and a `tempfile::tempdir()`. Verify the exact `Node`/`Edge`/`Ontology` construction against `src/graph/store.rs` so it compiles (provenance + declared types are validated by `upsert`).

- [ ] **Step 3: Run the tests**

Run: `cargo test -p glossa tools::`
Expected: the new glossary/neighbors tests PASS.

- [ ] **Step 4: Rewire `src/mcp.rs` to call the shared functions**

Replace the body of the `glossary` tool (mcp.rs:167-171) so it delegates:
```rust
async fn glossary(&self, Parameters(a): Parameters<NameArg>) -> Result<CallToolResult, McpError> {
    let g = GraphStore::open(&self.root).map_err(internal)?;
    Ok(CallToolResult::success(vec![Content::text(crate::tools::glossary(&g, &a.name, &self.trace))]))
}
```
Replace the body of the `neighbors` tool (mcp.rs:174-178):
```rust
async fn neighbors(&self, Parameters(a): Parameters<NeighborsArgs>) -> Result<CallToolResult, McpError> {
    let g = GraphStore::open(&self.root).map_err(internal)?;
    Ok(CallToolResult::success(vec![Content::text(crate::tools::neighbors(&g, &a.node_id, a.depth.unwrap_or(1), &self.trace))]))
}
```

- [ ] **Step 5: Build + C-free gate**

Run: `cargo test -p glossa tools::` (green) and `cargo tree -p glossa -i cc` (empty).

- [ ] **Step 6: Commit**

```bash
git add src/tools.rs src/mcp.rs
git commit -m "feat(tools): shared glossary + neighbors render (MCP parity)"
```

---

### Task 2: Wire glossary + neighbors into the eval + TensorZero config

**Files:**
- Modify: `eval/src/backend/glossa_tools.rs` (exec gains a graph param + two arms; update test call sites)
- Modify: `eval/src/backend/tensorzero.rs:165-166` (open the graph, pass it to exec)
- Create: `eval/tensorzero/config/tools/glossary.json`, `eval/tensorzero/config/tools/neighbors.json`
- Modify: `eval/tensorzero/config/tensorzero.toml` (`[tools.glossary]`, `[tools.neighbors]`, `answer_hotpot` tools list)
- Modify: `eval/tensorzero/config/answer_hotpot/system.minijinja` (mention the two tools)

**Interfaces:**
- Consumes: `glossa::tools::{glossary, neighbors}` (Task 1); `glossa::graph::store::GraphStore::open(dir: &Path) -> anyhow::Result<GraphStore>`.
- Produces: `exec(name, args, idx, graph, trace)` with `graph: Option<&glossa::graph::store::GraphStore>`.

- [ ] **Step 1: Change `exec` signature + add arms in `glossa_tools.rs`**

Signature becomes:
```rust
pub fn exec(name: &str, args: &Value, idx: &DocIndex, graph: Option<&glossa::graph::store::GraphStore>, trace: &TraceLog) -> (String, Vec<String>, Vec<glossa::read::DocImage>) {
```
Add two arms before the `other =>` fallback:
```rust
        "glossary" => {
            let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let body = match graph { Some(g) => glossa::tools::glossary(g, name, trace), None => "(graph unavailable)".to_string() };
            (body, Vec::new(), Vec::new())
        }
        "neighbors" => {
            let node_id = args.get("node_id").and_then(|v| v.as_str()).unwrap_or("");
            let depth = args.get("depth").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
            let body = match graph { Some(g) => glossa::tools::neighbors(g, node_id, depth, trace), None => "(graph unavailable)".to_string() };
            (body, Vec::new(), Vec::new())
        }
```

- [ ] **Step 2: Update existing `exec(...)` test call sites in `glossa_tools.rs`** — every `exec("…", &json!(…), &idx, &trace)` becomes `exec("…", &json!(…), &idx, None, &trace)`. Then add one test that builds a graph via `glossa::index::store::index_dir(dir, true)` on a temp dir containing a small `.md`, opens `GraphStore::open(dir)`, and asserts `exec("glossary", &json!({"name": "<known label>"}), &idx, Some(&g), &trace).0` is non-empty OR (simpler, robust) asserts `exec("glossary", &json!({"name":"zzz-nomatch"}), &idx, Some(&g), &trace).0 == "(no matches)"` and that with `graph = None` it returns `"(graph unavailable)"`. Pick assertions that are stable regardless of node-label specifics.

- [ ] **Step 3: Open the graph in `tensorzero.rs` and pass it** (lines 165-166)

```rust
        let idx = glossa::index::store::DocIndex::open_or_create(work)?;
        let graph = glossa::graph::store::GraphStore::open(work).ok();
        let exec = |name: &str, args: &Value| crate::backend::glossa_tools::exec(name, args, &idx, graph.as_ref(), &trace);
```

- [ ] **Step 4: Run the eval crate tests**

Run: `cargo test -p kb-eval backend::glossa_tools`
Expected: PASS (updated call sites compile; new arms covered).

- [ ] **Step 5: Add the TensorZero tool schemas**

`eval/tensorzero/config/tools/glossary.json`:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "description": "Resolve a term/name to knowledge-graph node ids.",
  "properties": {
    "name": { "type": "string", "description": "a term, label, or alias to resolve to node ids" }
  },
  "required": ["name"],
  "additionalProperties": false
}
```
`eval/tensorzero/config/tools/neighbors.json`:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "description": "List graph nodes connected to a node id.",
  "properties": {
    "node_id": { "type": "string", "description": "a graph node id (e.g. from glossary or a search result)" },
    "depth": { "type": "integer", "description": "how many hops to expand (default 1)" }
  },
  "required": ["node_id"],
  "additionalProperties": false
}
```

- [ ] **Step 6: Register the tools + extend the answer_hotpot tool list in `tensorzero.toml`**

After the `[tools.glob]` block (line ~145) add:
```toml
[tools.glossary]
description = "Resolve a term/name to knowledge-graph node ids whose label/alias matches it."
parameters = "tools/glossary.json"

[tools.neighbors]
description = "List knowledge-graph nodes connected to a node id (the graph links documents to their sections)."
parameters = "tools/neighbors.json"
```
Change the `answer_hotpot` tool list (line 109):
```toml
tools = ["search", "read", "grep", "glob", "glossary", "neighbors"]
```

- [ ] **Step 7: Mention the two tools in the system prompt** (`answer_hotpot/system.minijinja`) — add one bullet (do NOT mandate "always start with glossary"; the graph currently links only documents↔sections, so forcing it would waste turns):
```
- glossary(name): resolve a term/name to graph node ids; neighbors(node_id): list nodes linked to a node id (the graph connects documents and their sections).
```

- [ ] **Step 8: Build the eval binary (MCP holds kb.exe — build only kb-eval)**

Run: `cargo build -p kb-eval --release`
Expected: compiles clean.

- [ ] **Step 9: Commit**

```bash
git add eval/src/backend/glossa_tools.rs eval/src/backend/tensorzero.rs eval/tensorzero/config/
git commit -m "feat(eval): expose glossary + neighbors tools (Reader-profile parity)"
```

## Self-Review

**Coverage:** shared render (T1) + MCP parity (T1) + eval exec arms (T2) + graph threading (T2) + TZ schemas/registration/prompt (T2). The excluded mutation tools are documented in Global Constraints. **Types:** `glossary(&GraphStore, &str, &TraceLog) -> String`, `neighbors(&GraphStore, &str, usize, &TraceLog) -> String`, `exec(.., Option<&GraphStore>, ..)` consistent across tasks. **Placeholders:** the only judgement call (exact `upsert`/`index_dir` test construction) is bounded by "verify against `src/graph/store.rs`".
