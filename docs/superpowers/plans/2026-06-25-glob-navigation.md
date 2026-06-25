# glob Navigation + Consistent Scope Filters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-evident `glob(pattern)` tool that lists knowledge-base documents by path mask, and give `search` the same `-g`/`-t` scope filters `grep` already has — so search/grep/glob form a consistent, models-already-know-it toolset.

**Architecture:** A shared `glob_to_regex` (moved out of grep into `src/glob.rs`) backs both the new `glob_docs(idx, pattern)` (distinct document paths + chunk count, over the index's stored paths — File-First, no FS access) and a new `DocIndex::search_filtered` (BM25 then post-filter by path-glob + file_type). Exposed via MCP tools, `kb` CLI, and the kb-eval harness.

**Tech Stack:** Rust, tantivy 0.26, `regex` (already a dep). Pure-Rust.

**Scope:** Extends the tool-contract redesign (`docs/superpowers/specs/2026-06-25-tool-contract-redesign-design.md`) and reuses Plan B's grep code (merged on master). The Layer-2 term glossary is a SEPARATE future feature — out of scope.

## Global Constraints

- **C-free:** `cargo tree -p glossa -i cc` empty. No new deps (reuse `regex`).
- **File-First:** glob/filters operate over the index's stored paths/fields (disposable accelerator); no filesystem access.
- **Self-evident tools:** `glob` needs no prompt explanation — models know `glob`. Tool/param descriptions carry the contract.
- **Consistent addressing:** search/grep emit `path:#n`; `glob` emits document paths the agent reads via `read(path, n)`. All three tools accept `-g`/`-t` scope filters.
- **DRY:** reuse `glob_to_regex`, `DocIndex::iter_chunks`, `DocIndex::search`, `RankedHit`. Do not reinvent.
- **TDD:** failing test first. `cargo test -p glossa` / `cargo test -p kb-eval`.

---

### Task 1: Shared `glob_to_regex` + `glob_docs` (new `src/glob.rs`)

**Files:**
- Create: `src/glob.rs`
- Modify: `src/grep.rs` (drop the private `glob_to_regex`, import the shared one)
- Modify: `src/lib.rs` (add `pub mod glob;`)

**Interfaces:**
- Produces: `pub fn glob_to_regex(glob: &str) -> Result<regex::Regex, regex::Error>`; `pub fn glob_docs(idx: &crate::index::store::DocIndex, pattern: &str) -> anyhow::Result<Vec<(String, u64)>>` (distinct path, max ord), sorted by path.

- [ ] **Step 1: Write the failing test** (in the new `src/glob.rs`, a `#[cfg(test)] mod tests`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::store::DocIndex;
    use crate::model::Chunk;
    use std::path::PathBuf;

    fn idx_with(chunks: &[(&str, &str, &str)]) -> (tempfile::TempDir, DocIndex) {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let cs: Vec<Chunk> = chunks.iter().map(|(p, loc, t)| Chunk {
            doc_path: PathBuf::from(p), location: (*loc).into(), file_type: "pdf".into(), text: (*t).into(),
        }).collect();
        idx.write_chunks(&cs).unwrap();
        (dir, idx)
    }

    #[test]
    fn glob_docs_lists_distinct_matching_paths_with_counts() {
        let (_d, idx) = idx_with(&[
            ("kb\\Руководство АБАК.pdf", "p.1", "a"),
            ("kb\\Руководство АБАК.pdf", "p.2", "b"),
            ("kb\\Safety Manual.pdf", "p.1", "c"),
            ("kb\\Прочее.md", "S1", "d"),
        ]);
        let pdfs = glob_docs(&idx, "*.pdf").unwrap();
        assert_eq!(pdfs, vec![
            ("kb\\Руководство АБАК.pdf".to_string(), 2),
            ("kb\\Safety Manual.pdf".to_string(), 1),
        ]); // distinct paths, max ord as count, sorted; .md excluded
        let abak = glob_docs(&idx, "*АБАК*").unwrap();
        assert_eq!(abak, vec![("kb\\Руководство АБАК.pdf".to_string(), 2)]);
        assert!(glob_docs(&idx, "*nomatch*").unwrap().is_empty());
    }
}
```

(Note: `Chunk.location "S1"` for the md → `chunk_ord` gives seq 1; pdf `p.N` → ord N. The pdf doc has ords 1,2 → max 2.)

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p glossa glob_docs_lists 2>&1 | tail -5`
Expected: FAIL — module `glob` not found.

- [ ] **Step 3: Implement** — create `src/glob.rs`

```rust
//! Path-mask navigation over the knowledge base: a shared shell-glob→regex translator (also used
//! by grep's -g filter) and `glob_docs`, which lists the distinct documents whose path matches a
//! mask, with each document's chunk count. File-First: it reads the index's stored paths only.

use crate::index::store::DocIndex;
use std::collections::BTreeMap;

/// Translate a shell glob (`*`, `?`) into an anchored regex over the whole string.
pub fn glob_to_regex(glob: &str) -> Result<regex::Regex, regex::Error> {
    let mut re = String::from("^");
    for ch in glob.chars() {
        match ch {
            '*' => re.push_str(".*"),
            '?' => re.push('.'),
            c => re.push_str(&regex::escape(&c.to_string())),
        }
    }
    re.push('$');
    regex::Regex::new(&re)
}

/// List the DISTINCT document paths whose path matches `pattern`, each with its highest chunk
/// number (≈ page/section count, the `n`-range for `read(path, n)`). Sorted by path.
pub fn glob_docs(idx: &DocIndex, pattern: &str) -> anyhow::Result<Vec<(String, u64)>> {
    let re = glob_to_regex(pattern)?;
    let mut by_path: BTreeMap<String, u64> = BTreeMap::new();
    idx.iter_chunks(|path, ord, _ft, _body| {
        if re.is_match(path) {
            let e = by_path.entry(path.to_string()).or_insert(0);
            if ord > *e { *e = ord; }
        }
    })?;
    Ok(by_path.into_iter().collect())
}
```

Add `pub mod glob;` to `src/lib.rs`.

In `src/grep.rs`, delete the local `fn glob_to_regex(...)` and add `use crate::glob::glob_to_regex;` at the top (next to the other `use`s). Leave grep's call sites unchanged.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p glossa glob 2>&1 | tail -6`
Expected: PASS (glob_docs test + grep tests still green — grep now imports the shared glob_to_regex).

- [ ] **Step 5: Commit**

```bash
git add src/glob.rs src/grep.rs src/lib.rs
git commit -m "feat(glob): shared glob_to_regex + glob_docs (list documents by mask)"
```

---

### Task 2: `DocIndex::search_filtered` (path-glob + file_type scope)

**Files:**
- Modify: `src/index/store.rs` (add `search_filtered`)

**Interfaces:**
- Consumes: `DocIndex::search`, `crate::glob::glob_to_regex`, `RankedHit`.
- Produces: `pub fn search_filtered(&self, query: &str, limit: usize, glob: Option<&str>, file_type: Option<&str>) -> anyhow::Result<Vec<RankedHit>>`.

- [ ] **Step 1: Write the failing test** (append to `mod search_tests` in `src/index/store.rs`)

```rust
    #[test]
    fn search_filtered_scopes_by_glob_and_type() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            Chunk { doc_path: PathBuf::from("a/АБАК.pdf"), location: "p.1".into(), file_type: "pdf".into(), text: "горячая замена цпу".into() },
            Chunk { doc_path: PathBuf::from("b/Other.pdf"), location: "p.1".into(), file_type: "pdf".into(), text: "горячая замена цпу".into() },
            Chunk { doc_path: PathBuf::from("c/Notes.md"),  location: "S1".into(),  file_type: "md".into(),  text: "горячая замена цпу".into() },
        ]).unwrap();

        let all = idx.search_filtered("замена", 10, None, None).unwrap();
        assert_eq!(all.len(), 3);
        // glob scopes to the matching path only
        let abak = idx.search_filtered("замена", 10, Some("*АБАК*"), None).unwrap();
        assert_eq!(abak.len(), 1);
        assert!(abak[0].path.contains("АБАК"));
        // file_type scopes to md only
        let md = idx.search_filtered("замена", 10, None, Some("md")).unwrap();
        assert_eq!(md.len(), 1);
        assert_eq!(md[0].file_type, "md");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p glossa search_filtered_scopes 2>&1 | tail -5`
Expected: FAIL — no method `search_filtered`.

- [ ] **Step 3: Implement** (add inside an `impl DocIndex` block in `src/index/store.rs`)

```rust
    /// BM25 search scoped by an optional path glob and/or exact file_type. The filters are applied
    /// AFTER ranking, so a generous candidate pool is fetched when filtering to still fill `limit`.
    /// Reuses `search` (unfiltered) so ranking semantics stay identical.
    pub fn search_filtered(
        &self,
        query: &str,
        limit: usize,
        glob: Option<&str>,
        file_type: Option<&str>,
    ) -> anyhow::Result<Vec<RankedHit>> {
        if glob.is_none() && file_type.is_none() {
            return self.search(query, limit);
        }
        let glob_re = match glob {
            Some(g) => Some(crate::glob::glob_to_regex(g)?),
            None => None,
        };
        let pool = limit.saturating_mul(20).clamp(limit.max(1), 2000);
        let hits = self.search(query, pool)?;
        let filtered: Vec<RankedHit> = hits
            .into_iter()
            .filter(|h| file_type.is_none_or(|ft| h.file_type == ft))
            .filter(|h| glob_re.as_ref().is_none_or(|re| re.is_match(&h.path)))
            .take(limit)
            .collect();
        Ok(filtered)
    }
```

(If `is_none_or` is unavailable on the toolchain, use `file_type.map_or(true, |ft| h.file_type == ft)` and `glob_re.as_ref().map_or(true, |re| re.is_match(&h.path))`.)

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p glossa search_filtered_scopes 2>&1 | tail -5`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/index/store.rs
git commit -m "feat(search): search_filtered — path-glob + file_type scope"
```

---

### Task 3: MCP `glob` tool + `search` scope args + CLI

**Files:**
- Modify: `src/mcp.rs` (`GlobArgs` + `glob` tool; `SearchArgs` gains `glob`/`file_type`; `search` uses `search_filtered`)
- Modify: `src/main.rs` (`kb glob` subcommand; `kb search` gains `-g`/`-t`)

**Interfaces:**
- Consumes: `crate::glob::glob_docs`, `DocIndex::search_filtered`.

- [ ] **Step 1: Write the failing test** (append to `src/mcp.rs` `mod tests`)

```rust
    #[tokio::test]
    async fn glob_tool_lists_documents() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("АБАК.md"), b"# A\nраз\n# B\nдва\n").unwrap();
        std::fs::write(dir.path().join("Other.md"), b"# A\nраз\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let srv = GlossaServer::new(dir.path().to_path_buf(), Profile::Editor, false, false);
        let out = format!("{:?}", srv.glob(Parameters(GlobArgs { pattern: "*АБАК*".into() })).await.unwrap());
        assert!(out.contains("АБАК"), "lists the matching doc: {out}");
        assert!(!out.contains("Other"), "excludes non-matching: {out}");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p glossa glob_tool_lists 2>&1 | tail -6`
Expected: FAIL — no `GlobArgs`/`glob` tool.

- [ ] **Step 3: Implement**

Add `GlobArgs` near the other `*Args` in `src/mcp.rs`:

```rust
#[derive(Debug, Deserialize, JsonSchema)]
struct GlobArgs {
    #[schemars(description = "shell glob over document paths, e.g. *.pdf or *Safety*")]
    pattern: String,
}
```

Add the tool inside `#[tool_router] impl GlossaServer`:

```rust
    #[tool(description = "List knowledge-base documents whose path matches a shell glob (e.g. `*.pdf`, `*Safety*`, `*АБАК*`). Returns one `path  (N chunks)` per line — use it to discover what documents exist or find a file by name, then `read(path, n)` or scope a `search`/`grep` to it. N is the document's last chunk number (page/section count).")]
    async fn glob(&self, Parameters(a): Parameters<GlobArgs>) -> Result<CallToolResult, McpError> {
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let docs = crate::glob::glob_docs(&idx, &a.pattern).map_err(internal)?;
        self.trace.log("glob", serde_json::json!({"pattern": a.pattern}), serde_json::json!({"docs": docs.len()}));
        let body = if docs.is_empty() {
            "(no documents match)".to_string()
        } else {
            docs.iter().map(|(p, n)| format!("{p}  ({n} chunks)")).collect::<Vec<_>>().join("\n")
        };
        Ok(CallToolResult::success(vec![Content::text(body)]))
    }
```

Extend `SearchArgs` with optional scope filters:

```rust
    #[serde(default)]
    #[schemars(description = "only documents whose path matches this glob, e.g. *.pdf or *АБАК* (-g)")]
    glob: Option<String>,
    #[serde(default)]
    #[schemars(description = "only this file type, e.g. pdf (-t)")]
    file_type: Option<String>,
```

Change `search`'s body to use the filtered query:

```rust
        let hits = idx.search_filtered(&a.query, a.limit.unwrap_or(50), a.glob.as_deref(), a.file_type.as_deref()).map_err(internal)?;
```

(Also add the same `glob`/`file_type` schemars line to the `search` description so the model knows the scope filters exist: append " Scope with optional glob/file_type filters." to its `#[tool(description=...)]`.)

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p glossa glob_tool_lists 2>&1 | tail -6`
Expected: PASS.

- [ ] **Step 5: Add the CLI** (in `src/main.rs`)

Add a `Glob` subcommand to the `Cmd` enum (mirror `Grep`):

```rust
    /// List documents whose path matches a shell glob.
    Glob {
        /// glob pattern, e.g. *.pdf or *Safety*
        pattern: String,
        /// knowledge-base directory
        path: std::path::PathBuf,
    },
```

And its match arm:

```rust
        Cmd::Glob { pattern, path } => {
            let idx = glossa::index::store::DocIndex::open_or_create(&path)?;
            for (p, n) in glossa::glob::glob_docs(&idx, &pattern)? {
                println!("{p}  ({n} chunks)");
            }
            Ok(())
        }
```

Add `-g`/`-t` to the existing `Cmd::Search` variant + route its handler through `search_filtered` (find the `Search` variant and its arm; add `#[arg(short='g', long)] glob: Option<String>` and `#[arg(short='t', long="type")] file_type: Option<String>`, and call `idx.search_filtered(&query, limit, glob.as_deref(), file_type.as_deref())` in place of `idx.search(...)`).

- [ ] **Step 6: Build + smoke test**

Run: `cargo build --release 2>&1 | tail -1 && ./target/release/kb.exe glob "*АБАК*" kb-test 2>&1 | head -3`
Expected: `Finished`; lists АБАК documents with chunk counts.

(If `kb.exe` is locked by an external MCP host, build `cargo build -p glossa --release --bin kb` is still blocked — note it and run the smoke test after the lock is released; the unit test already covers correctness.)

- [ ] **Step 7: Commit**

```bash
git add src/mcp.rs src/main.rs
git commit -m "feat(glob): MCP glob tool + search -g/-t scope + kb glob CLI"
```

---

### Task 4: Mirror in the kb-eval harness

**Files:**
- Modify: `eval/src/backend/glossa_tools.rs` (`glob` arm; `run_search` reads `glob`/`file_type`)
- Create: `eval/tensorzero/config/tools/glob.json`
- Modify: `eval/tensorzero/config/tools/search.json` (add `glob`/`file_type`)
- Modify: `eval/tensorzero/config/tensorzero.toml` (`[tools.glob]` + add to `answer_hotpot.tools`)
- Modify: `eval/tensorzero/config/answer_hotpot/system.minijinja` (one line)

**Interfaces:**
- Consumes: `glossa::glob::glob_docs`, `DocIndex::search_filtered`.

- [ ] **Step 1: Write the failing test** (append to `eval/src/backend/glossa_tools.rs` `mod tests`)

```rust
    #[test]
    fn glob_and_scoped_search_via_exec() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            Chunk { doc_path: PathBuf::from("АБАК.pdf"), location: "p.1".into(), file_type: "pdf".into(), text: "горячая замена".into() },
            Chunk { doc_path: PathBuf::from("Other.pdf"), location: "p.1".into(), file_type: "pdf".into(), text: "горячая замена".into() },
        ]).unwrap();
        let trace = TraceLog::disabled();
        let g = exec("glob", &json!({"pattern": "*АБАК*"}), &idx, &trace).0;
        assert!(g.contains("АБАК") && !g.contains("Other"), "glob: {g}");
        let s = exec("search", &json!({"query": "замена", "glob": "*АБАК*"}), &idx, &trace).0;
        assert!(s.contains("АБАК") && !s.contains("Other"), "scoped search: {s}");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p kb-eval glob_and_scoped_search 2>&1 | tail -6`
Expected: FAIL — no `glob` arm; `run_search` ignores `glob`.

- [ ] **Step 3: Implement** (in `eval/src/backend/glossa_tools.rs`)

Change `run_search` to accept + apply the filters (read from args in `exec`). Replace the whole `run_search` function with:

```rust
/// Run a BM25 search (optionally scoped by path glob / file_type); model-facing numbered text + titles.
pub fn run_search(idx: &DocIndex, query: &str, limit: usize, glob: Option<&str>, file_type: Option<&str>, trace: &TraceLog) -> (String, Vec<String>) {
    match idx.search_filtered(query, limit.max(1), glob, file_type) {
        Ok(hits) => {
            let trace_hits: Vec<Value> = hits.iter().map(|h| json!({ "path": h.path, "location": h.location, "score": h.score })).collect();
            trace.log("search", json!({ "query": query }), json!(trace_hits));
            let titles: Vec<String> = hits.iter().map(|h| h.location.clone()).collect();
            if hits.is_empty() {
                return ("(no results)".to_string(), titles);
            }
            let body = hits.iter().map(|h| h.display_line()).collect::<Vec<_>>().join("\n");
            (body, titles)
        }
        Err(e) => (format!("search error: {e}"), Vec::new()),
    }
}
```

Add `run_glob` + the two `exec` arms:

```rust
/// List documents matching a shell glob; one `path  (N chunks)` per line.
pub fn run_glob(idx: &DocIndex, pattern: &str, trace: &TraceLog) -> (String, Vec<String>) {
    match glossa::glob::glob_docs(idx, pattern) {
        Ok(docs) => {
            trace.log("glob", json!({ "pattern": pattern }), json!({ "docs": docs.len() }));
            let titles: Vec<String> = docs.iter().map(|(p, _)| p.clone()).collect();
            let body = if docs.is_empty() {
                "(no documents match)".to_string()
            } else {
                docs.iter().map(|(p, n)| format!("{p}  ({n} chunks)")).collect::<Vec<_>>().join("\n")
            };
            (body, titles)
        }
        Err(e) => (format!("glob error: {e}"), Vec::new()),
    }
}
```

In `exec`, update the `"search"` arm to pass the filters and add a `"glob"` arm:

```rust
        "search" => {
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
            let glob = args.get("glob").and_then(|v| v.as_str());
            let file_type = args.get("file_type").and_then(|v| v.as_str());
            run_search(idx, query, limit, glob, file_type, trace)
        }
        "glob" => {
            let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
            run_glob(idx, pattern, trace)
        }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p kb-eval glob_and_scoped_search 2>&1 | tail -6`
Expected: PASS.

- [ ] **Step 5: Config + schemas + prompt**

Create `eval/tensorzero/config/tools/glob.json`:

```json
{
  "type": "object",
  "properties": {
    "pattern": { "type": "string", "description": "shell glob over document paths, e.g. *.pdf or *Safety* or *АБАК*" }
  },
  "required": ["pattern"]
}
```

Add to `eval/tensorzero/config/tools/search.json` `properties` (keep `query` required):

```json
    "glob": { "type": "string", "description": "scope to documents whose path matches this glob, e.g. *.pdf or *АБАК* (-g)" },
    "file_type": { "type": "string", "description": "scope to this file type, e.g. pdf (-t)" }
```

In `eval/tensorzero/config/tensorzero.toml`: add `glob` to the function tools list and declare the tool:

```toml
tools = ["search", "read", "grep", "glob"]
```
```toml
[tools.glob]
description = "List knowledge-base documents whose path matches a shell glob (e.g. *.pdf, *Safety*, *АБАК*). Returns `path  (N chunks)` per line — discover what documents exist or find a file by name, then read(path, n) or scope a search/grep to it."
parameters = "tools/glob.json"
```

In `eval/tensorzero/config/answer_hotpot/system.minijinja`, after the grep line add:

```
- glob(pattern): list documents by path mask (e.g. *Safety*, *АБАК*.pdf) to discover/find files; then read(path, n) or scope search/grep with the glob filter.
```

- [ ] **Step 6: Run to verify it passes**

Run: `cargo test -p kb-eval 2>&1 | grep "test result" | tail -2`
Expected: `ok.`

- [ ] **Step 7: Commit**

```bash
git add eval/src/backend/glossa_tools.rs eval/tensorzero/config/tools/glob.json eval/tensorzero/config/tools/search.json eval/tensorzero/config/tensorzero.toml eval/tensorzero/config/answer_hotpot/system.minijinja
git commit -m "feat(eval): mirror glob tool + search scope filters"
```

---

### Task 5: C-free invariant + full-suite gate

**Files:** none (verification only)

- [ ] **Step 1: C-free**

Run: `cargo tree -p glossa -i cc 2>&1 | tail -2`
Expected: `warning: nothing to print.`

- [ ] **Step 2: Full suites**

Run: `cargo test -p glossa 2>&1 | grep "test result" | head -1 && cargo test -p kb-eval 2>&1 | grep "test result" | tail -2`
Expected: all `ok.`

- [ ] **Step 3: Release build**

Run: `cargo build --release 2>&1 | tail -1`
Expected: `Finished` (if `kb.exe` is locked by an external MCP host, build `cargo build -p kb-eval --release` and note the `kb.exe` rebuild is pending lock release).

- [ ] **Step 4: Commit (if incidental fixes were needed)**

```bash
git add -A
git commit -m "chore: verify C-free + full suite green for glob navigation"
```

---

## Notes for the implementer

- **No reindex needed** — glob/filters read existing stored `path`/`file_type`/`ord` fields.
- **Reuse, don't reinvent:** `glob_to_regex` and `iter_chunks` already exist; `search_filtered` wraps `search`. The grep `-g` filter keeps working via the moved-but-identical `glob_to_regex`.
- **Out of scope:** Layer-2 term glossary; `**`/`{a,b}`/char-class globs (basic `*`/`?` only).
