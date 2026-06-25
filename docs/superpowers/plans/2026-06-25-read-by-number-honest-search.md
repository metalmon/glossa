# Read-by-Number + Honest Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the agent address `read` by a single typed integer (the chunk's canonical number) instead of a free-form `location` string, and make `search` advertise itself honestly — killing the `location:"страница 21"` mis-address bug by construction.

**Architecture:** Add a per-document canonical number `ord` to each indexed chunk (page number for PDFs, 1-based section/sheet index otherwise). `read(path, n)` resolves a chunk by `(path, ord=n)` from the index (File-First accelerator), returns its stored body plus `n-1`/`n+1` neighbor hints. `search` numbers each hit `[#n]` and drops the stale "ripgrep" labels. Mirrored in the `kb-eval` harness so the eval measures the prod contract.

**Tech Stack:** Rust, tantivy 0.26 (BM25 index), rmcp (MCP), serde_json. Pure-Rust only.

**Scope:** This is **Plan A** of Step 1 in `docs/superpowers/specs/2026-06-25-tool-contract-redesign-design.md`. The new `grep` tool is **Plan B** (separate); the trigram accelerator is **Step 2**. Do NOT build grep or trigrams here.

## Global Constraints

- **C-free invariant:** `cargo tree -p glossa -i cc` must stay empty. No new C-linked deps.
- **File-First:** files are the source of truth; the index is a disposable accelerator. `read` serves the chunk body from the index; `read_region(path, location)` stays as the CLI/human surface (do not remove it).
- **Self-evident contract:** `read`'s number is a JSON-schema `integer`; the model must not need prompt text to use it. No parameter may admit a second reading.
- **`ord` definition (one number per chunk):** page number for PDF chunks (parsed from the existing `p.N` location); 1-based section/sheet index for every other format. Never a synthetic ordinal competing with a page number.
- **No "ripgrep" in the `search` contract:** the implementation is BM25 keywords; every "ripgrep syntax" claim must go.
- **TDD:** every task writes a failing test first. Run `cargo test -p glossa` (core) / `cargo test -p kb-eval` (eval).

---

### Task 1: Add `ord` to the index schema and assign it on write

**Files:**
- Modify: `src/index/store.rs` (schema `build_schema`, `Fields`, `RankedHit`, `search`, `write_chunks`, `index_dir`)

**Interfaces:**
- Produces: `Fields.ord: tantivy::schema::Field`; `RankedHit.ord: u64`; `pub fn chunk_ord(file_type: &str, location: &str, seq: u64) -> u64`.

- [ ] **Step 1: Write the failing test** (append to `mod search_tests` in `src/index/store.rs`)

```rust
    #[test]
    fn chunk_ord_uses_page_for_pdf_else_sequence() {
        assert_eq!(chunk_ord("pdf", "p.21", 5), 21);
        assert_eq!(chunk_ord("pdf", "p.350", 1), 350);
        assert_eq!(chunk_ord("md", "Introduction", 3), 3); // non-pdf -> sequence
        assert_eq!(chunk_ord("pdf", "weird", 7), 7);        // unparseable page -> sequence fallback
    }

    #[test]
    fn search_hit_carries_ord() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            Chunk { doc_path: PathBuf::from("d.pdf"), location: "p.7".into(), file_type: "pdf".into(), text: "горячая замена цпу".into() },
        ]).unwrap();
        let hits = idx.search("замена", 10).unwrap();
        assert_eq!(hits[0].ord, 7);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p glossa chunk_ord_uses_page search_hit_carries_ord 2>&1 | tail -5`
Expected: FAIL — `chunk_ord` not found / `RankedHit` has no field `ord`.

- [ ] **Step 3: Implement**

In `build_schema` (after the `file_type` field):

```rust
    let ord = sb.add_u64_field("ord", INDEXED | STORED);
```

Add `INDEXED` to the schema import line:

```rust
use tantivy::schema::{Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, INDEXED, STORED, STRING};
```

Extend `Fields` and its constructor return:

```rust
pub struct Fields {
    pub body: Field,
    pub path: Field,
    pub location: Field,
    pub file_type: Field,
    pub ord: Field,
}
```
```rust
    (sb.build(), Fields { body, path, location, file_type, ord })
```

Add the helper (top-level fn in `src/index/store.rs`):

```rust
/// The chunk's single canonical number within its document: the page number for PDFs
/// (parsed from the `p.N` location), otherwise the 1-based sequence position `seq`.
pub fn chunk_ord(file_type: &str, location: &str, seq: u64) -> u64 {
    if file_type == "pdf" {
        if let Some(n) = location.strip_prefix("p.").and_then(|d| d.parse::<u64>().ok()) {
            return n;
        }
    }
    seq
}
```

Add `ord` to `RankedHit`:

```rust
pub struct RankedHit {
    pub path: String,
    pub location: String,
    pub file_type: String,
    pub ord: u64,
    pub snippet: String,
    pub score: f32,
}
```

In `search`, read the stored `ord` (after `let snippet = ...`):

```rust
            let ord = d.get_first(self.fields.ord).and_then(|v| v.as_u64()).unwrap_or(0);
            hits.push(RankedHit {
                path: get(self.fields.path),
                location: get(self.fields.location),
                file_type: get(self.fields.file_type),
                ord,
                snippet,
                score,
            });
```

In `write_chunks`, assign a 1-based per-call sequence:

```rust
    pub fn write_chunks(&self, chunks: &[Chunk]) -> anyhow::Result<()> {
        let mut writer = self.index.writer(50_000_000)?;
        for (i, c) in chunks.iter().enumerate() {
            let ord = chunk_ord(&c.file_type, &c.location, (i + 1) as u64);
            writer.add_document(doc!(
                self.fields.body => c.text.clone(),
                self.fields.path => c.doc_path.to_string_lossy().to_string(),
                self.fields.location => c.location.clone(),
                self.fields.file_type => c.file_type.clone(),
                self.fields.ord => ord,
            ))?;
        }
        writer.commit()?;
        self.reader.reload()?;
        Ok(())
    }
```

In `index_dir`, assign a per-file sequence inside the walk closure. Replace the `extract_file` block (store.rs ~214-226) with:

```rust
        let mut seq = 0u64;
        crate::extract::extract_file(path, &mut |c| {
            if !doc_written {
                let _ = crate::graph::build::build_document(&graph, &path_str, sig);
                doc_written = true;
            }
            seq += 1;
            let ord = crate::index::store::chunk_ord(&c.file_type, &c.location, seq);
            let _ = writer.add_document(doc!(
                idx.fields.body => c.text.clone(),
                idx.fields.path => path_str.clone(),
                idx.fields.location => c.location.clone(),
                idx.fields.file_type => c.file_type.clone(),
                idx.fields.ord => ord,
            ));
            let _ = crate::graph::build::build_section(&graph, &c, sig);
        })?;
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p glossa 2>&1 | tail -5`
Expected: PASS (all glossa tests; the two new ones included).

- [ ] **Step 5: Commit**

```bash
git add src/index/store.rs
git commit -m "feat(index): add per-document canonical chunk number (ord)"
```

---

### Task 2: `DocIndex::read_chunk_by_ord(path, n)` → body + neighbors

**Files:**
- Modify: `src/index/store.rs` (new method + `ChunkRead` struct, in the `impl DocIndex` block that holds `read_chunk`)

**Interfaces:**
- Consumes: `Fields.ord` (Task 1).
- Produces: `pub struct ChunkRead { pub body: String, pub prev: Option<u64>, pub next: Option<u64> }`; `pub fn read_chunk_by_ord(&self, path: &str, n: u64) -> anyhow::Result<Option<ChunkRead>>`.

- [ ] **Step 1: Write the failing test** (append to `mod search_tests`)

```rust
    #[test]
    fn read_chunk_by_ord_returns_body_and_neighbors() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let sec = |loc: &str, t: &str| Chunk {
            doc_path: PathBuf::from("d.md"), location: loc.into(), file_type: "md".into(), text: t.into(),
        };
        idx.write_chunks(&[sec("A", "alpha"), sec("B", "bravo"), sec("C", "charlie")]).unwrap();

        let mid = idx.read_chunk_by_ord("d.md", 2).unwrap().unwrap();
        assert_eq!(mid.body, "bravo");
        assert_eq!(mid.prev, Some(1));
        assert_eq!(mid.next, Some(3));

        let first = idx.read_chunk_by_ord("d.md", 1).unwrap().unwrap();
        assert_eq!(first.prev, None);
        assert_eq!(first.next, Some(2));

        assert!(idx.read_chunk_by_ord("d.md", 99).unwrap().is_none());
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p glossa read_chunk_by_ord 2>&1 | tail -5`
Expected: FAIL — method not found.

- [ ] **Step 3: Implement** (add inside the `impl DocIndex` block containing `read_chunk`)

```rust
/// A chunk read by its canonical number, with the numbers of its in-document neighbors.
pub struct ChunkRead {
    pub body: String,
    pub prev: Option<u64>,
    pub next: Option<u64>,
}

impl DocIndex {
    /// Fetch a chunk's stored body by exact (path, ord). Reports whether ord-1 / ord+1 exist in
    /// the same document, so the caller can offer "next/previous chunk" navigation. None if no
    /// chunk with that (path, ord) is indexed.
    pub fn read_chunk_by_ord(&self, path: &str, n: u64) -> anyhow::Result<Option<ChunkRead>> {
        let body = match self.ord_body(path, n)? {
            Some(b) => b,
            None => return Ok(None),
        };
        let prev = if n > 1 && self.ord_body(path, n - 1)?.is_some() { Some(n - 1) } else { None };
        let next = if self.ord_body(path, n + 1)?.is_some() { Some(n + 1) } else { None };
        Ok(Some(ChunkRead { body, prev, next }))
    }

    fn ord_body(&self, path: &str, n: u64) -> anyhow::Result<Option<String>> {
        use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
        let searcher = self.reader.searcher();
        let clauses: Vec<(Occur, Box<dyn Query>)> = vec![
            (Occur::Must, Box::new(TermQuery::new(
                tantivy::Term::from_field_text(self.fields.path, path), IndexRecordOption::Basic))),
            (Occur::Must, Box::new(TermQuery::new(
                tantivy::Term::from_field_u64(self.fields.ord, n), IndexRecordOption::Basic))),
        ];
        let top = searcher.search(&BooleanQuery::new(clauses), &TopDocs::with_limit(1).order_by_score())?;
        match top.first() {
            Some((_score, addr)) => {
                let d: TantivyDocument = searcher.doc(*addr)?;
                Ok(Some(d.get_first(self.fields.body).and_then(|v| v.as_str()).unwrap_or("").to_string()))
            }
            None => Ok(None),
        }
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p glossa read_chunk_by_ord 2>&1 | tail -5`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/index/store.rs
git commit -m "feat(index): read_chunk_by_ord — body + in-document neighbors"
```

---

### Task 3: MCP `read(path, n)` typed integer + neighbor footer

**Files:**
- Modify: `src/mcp.rs` (`ReadArgs`, `read` tool)

**Interfaces:**
- Consumes: `DocIndex::read_chunk_by_ord` (Task 2), `GlossaServer.root`.

- [ ] **Step 1: Write the failing test** (append to `src/mcp.rs`'s `#[cfg(test)] mod tests`)

```rust
    // `mod tests` already does `use super::*`, so `Parameters`, `GlossaServer`, `Profile`,
    // `ReadArgs`, and `index_dir` are in scope from mcp.rs's own imports.
    #[tokio::test]
    async fn read_by_number_returns_body_and_footer() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("d.md"), b"# A\nalpha\n# B\nbravo\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let srv = GlossaServer::new(dir.path().to_path_buf(), Profile::Editor, false, false);
        let path = dir.path().join("d.md").to_string_lossy().to_string();

        let out = srv.read(Parameters(ReadArgs { path, n: 1, include_images: Some(false) })).await.unwrap();
        let text = format!("{:?}", out);
        assert!(text.contains("alpha"), "body present: {text}");
        assert!(text.contains("#2") || text.contains("next"), "footer offers next: {text}");
    }
```

(Note: the section title line may be part of the chunk; assert on the body word `alpha`, which is in section 1.)

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p glossa read_by_number_returns_body 2>&1 | tail -8`
Expected: FAIL — `ReadArgs` has no field `n` / no `location`.

- [ ] **Step 3: Implement**

Replace `ReadArgs`:

```rust
#[derive(Debug, Deserialize, JsonSchema)]
struct ReadArgs {
    #[schemars(description = "document path, exactly as shown in a search result")]
    path: String,
    #[schemars(description = "chunk number to read, exactly as shown in `[#n]` in a search result (page number for PDFs)")]
    n: u32,
    #[serde(default)]
    #[schemars(description = "include embedded images (default true)")]
    include_images: Option<bool>,
}
```

Replace the `read` tool body:

```rust
    #[tool(description = "Read a document chunk by its number `n` (the `[#n]` shown in search results; for PDFs this is the page number). Returns the chunk text plus the numbers of the previous/next chunks for context expansion.")]
    async fn read(&self, Parameters(a): Parameters<ReadArgs>) -> Result<CallToolResult, McpError> {
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let path = std::path::PathBuf::from(&a.path);
        let chunk = idx.read_chunk_by_ord(&a.path, a.n as u64).map_err(internal)?
            .ok_or_else(|| McpError::invalid_params(format!("no chunk #{} in {}", a.n, a.path), None))?;
        self.trace.log("read", serde_json::json!({"path": a.path, "n": a.n}), serde_json::json!({"path": a.path}));
        let footer = match (chunk.prev, chunk.next) {
            (Some(p), Some(n)) => format!("\n\n‹ prev #{p} · next #{n} ›"),
            (None, Some(n)) => format!("\n\n‹ start of document · next #{n} ›"),
            (Some(p), None) => format!("\n\n‹ prev #{p} · end of document ›"),
            (None, None) => String::new(),
        };
        let mut content = vec![Content::text(format!("{}{}", chunk.body, footer))];
        if a.include_images.unwrap_or(true) {
            for img in extract_images(&path, 8).map_err(internal)? {
                let b64 = base64::engine::general_purpose::STANDARD.encode(&img.bytes);
                content.push(Content::image(b64, img.mime));
            }
        }
        Ok(CallToolResult::success(content))
    }
```

(Keep the `read_region` import/function — it is still used by the CLI in `src/main.rs`. If the compiler warns it is unused in `mcp.rs`, remove only the `mcp.rs` `use` of it, not the function.)

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p glossa read_by_number_returns_body 2>&1 | tail -8`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mcp.rs
git commit -m "feat(mcp): read by chunk number n with prev/next footer"
```

---

### Task 4: Honest, numbered `search` (MCP + shared formatter)

**Files:**
- Modify: `src/index/store.rs` (add `RankedHit::display_line`)
- Modify: `src/mcp.rs` (`SearchArgs` description, `search` output, server instructions line ~202)
- Modify: `src/main.rs` (CLI search output uses the shared formatter)

**Interfaces:**
- Consumes: `RankedHit.ord` (Task 1).
- Produces: `impl RankedHit { pub fn display_line(&self) -> String }`.

- [ ] **Step 1: Write the failing test** (append to `mod search_tests` in `src/index/store.rs`)

```rust
    #[test]
    fn display_line_is_numbered_with_nonnumeric_label() {
        let pdf = RankedHit { path: "d.pdf".into(), location: "p.350".into(), file_type: "pdf".into(), ord: 350, snippet: "горячая замена".into(), score: 17.7 };
        let line = pdf.display_line();
        assert!(line.starts_with("[#350] "), "numbered key: {line}");
        assert!(line.contains("pdf"), "non-numeric label for pdf: {line}");
        assert!(!line.contains("p.350"), "no competing page number: {line}");

        let md = RankedHit { path: "d.md".into(), location: "Введение".into(), file_type: "md".into(), ord: 2, snippet: "текст".into(), score: 3.0 };
        assert!(md.display_line().starts_with("[#2] "));
        assert!(md.display_line().contains("Введение"));
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p glossa display_line_is_numbered 2>&1 | tail -5`
Expected: FAIL — no method `display_line`.

- [ ] **Step 3: Implement**

Add to `src/index/store.rs` (after the `RankedHit` struct):

```rust
impl RankedHit {
    /// One search-result line carrying exactly one number — the read key `[#ord]` — and a
    /// non-numeric label (the heading text, or the file type for paged formats whose location is
    /// itself a number) so nothing competes with the read key.
    pub fn display_line(&self) -> String {
        let label = if self.location.starts_with("p.") { self.file_type.as_str() } else { self.location.as_str() };
        format!("[#{}] {} · {} · {}  [{:.3}]", self.ord, self.path, label, self.snippet, self.score)
    }
}
```

In `src/mcp.rs`, fix the `SearchArgs` query description:

```rust
    #[schemars(description = "natural-language keywords (Russian or English; morphology-aware, BM25-ranked) — NOT a regex")]
    query: String,
```

In `src/mcp.rs` `search`, replace the `body` builder:

```rust
        let body = hits.iter().map(|h| h.display_line()).collect::<Vec<_>>().join("\n");
```

In `src/mcp.rs`, fix the server instructions (the line containing "Use ripgrep syntax for `search`", ~202):

```rust
        info.instructions = Some("glossa File-First knowledge-base search. `search` takes BM25 keywords (morphology-aware), returns numbered hits `[#n]`; `read` opens chunk number `n`.".into());
```

In `src/main.rs`, find the CLI search print loop and replace the per-hit format string with `h.display_line()` (the CLI currently formats hits inline — use the shared method for parity).

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p glossa 2>&1 | tail -5`
Expected: PASS. Then grep to confirm the lie is gone:
Run: `grep -rni "ripgrep" src/mcp.rs` → Expected: no matches.

- [ ] **Step 5: Commit**

```bash
git add src/index/store.rs src/mcp.rs src/main.rs
git commit -m "feat(search): numbered [#n] output; drop stale ripgrep labels"
```

---

### Task 5: Mirror the contract in the kb-eval harness

**Files:**
- Modify: `eval/src/backend/glossa_tools.rs` (`run_search` numbered, `run_read` by ord, `exec` parses integer `n`)
- Modify: `eval/tensorzero/config/tools/read.json` (schema → `{path, n:integer}`)
- Modify: `eval/tensorzero/config/answer_hotpot/system.minijinja` (describe `read(path, n)` + numbered search)

**Interfaces:**
- Consumes: `DocIndex::read_chunk_by_ord`, `RankedHit::display_line` (Tasks 2, 4).

- [ ] **Step 1: Write the failing test** (append to a new `#[cfg(test)] mod tests` in `eval/src/backend/glossa_tools.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use glossa::index::store::DocIndex;
    use glossa::model::Chunk;
    use glossa::trace::TraceLog;
    use std::path::PathBuf;

    #[test]
    fn read_accepts_integer_or_digit_string_and_returns_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            Chunk { doc_path: PathBuf::from("d.pdf"), location: "p.7".into(), file_type: "pdf".into(), text: "седьмая страница".into() },
        ]).unwrap();
        let trace = TraceLog::disabled();

        // integer n
        let out = exec("read", &json!({"path": "d.pdf", "n": 7}), &idx, &trace).0;
        assert!(out.contains("седьмая"), "got: {out}");
        // stray string "p.7" -> digit-strip fallback -> 7
        let out2 = exec("read", &json!({"path": "d.pdf", "n": "p.7"}), &idx, &trace).0;
        assert!(out2.contains("седьмая"), "digit-strip fallback: {out2}");
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p kb-eval read_accepts_integer_or_digit 2>&1 | tail -8`
Expected: FAIL — `run_read` still takes `location`, no digit-strip.

- [ ] **Step 3: Implement**

In `eval/src/backend/glossa_tools.rs`, replace `run_search`'s body builder to use the numbered formatter, and keep logging `location` in the trace (recall@k depends on it):

```rust
            let body = hits.iter().map(|h| h.display_line()).collect::<Vec<_>>().join("\n");
```

Add a digit-strip helper and replace `run_read` to read by `(path, n)` with a footer. `parse_n`
accepts a JSON integer or strips a stray string to its digits (`"p.7"` → 7):

```rust
/// Parse the model's `n` argument: a JSON integer, or any string we strip to its digits
/// (e.g. "p.7" -> 7). None if no digits are present.
fn parse_n(v: &Value) -> Option<u64> {
    if let Some(n) = v.as_u64() { return Some(n); }
    let s: String = v.as_str()?.chars().filter(|c| c.is_ascii_digit()).collect();
    s.parse::<u64>().ok()
}

/// Read a chunk by (path, number n) from the index; truncated to fit small-model context.
pub fn run_read(idx: &DocIndex, path: &str, n: u64, trace: &TraceLog) -> String {
    match idx.read_chunk_by_ord(path, n) {
        Ok(Some(c)) => {
            trace.log("read", json!({ "path": path, "n": n }), json!({ "path": path }));
            let footer = match (c.prev, c.next) {
                (Some(p), Some(nx)) => format!("\n‹ prev #{p} · next #{nx} ›"),
                (None, Some(nx)) => format!("\n‹ start · next #{nx} ›"),
                (Some(p), None) => format!("\n‹ prev #{p} · end ›"),
                (None, None) => String::new(),
            };
            cap_read(c.body) + &footer
        }
        Ok(None) => format!("no chunk #{n} in {path}"),
        Err(e) => format!("read error: {e}"),
    }
}
```

Update `exec`'s `read` arm to parse `path` + integer `n`:

```rust
        "read" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let n = args.get("n").and_then(parse_n).unwrap_or(0);
            (run_read(idx, path, n, trace), Vec::new())
        }
```

(Delete the now-unused `run_read(idx, path, location)` signature and the old `read_chunk(location)` call path. If `glossa::...::read_chunk` becomes unused crate-wide, leave it — Plan B's grep does not need it; removing it is out of scope.)

Update `eval/tensorzero/config/tools/read.json`:

```json
{
  "type": "object",
  "properties": {
    "path": { "type": "string", "description": "document path, exactly as shown in a search result" },
    "n": { "type": "integer", "description": "chunk number to read, exactly as shown in [#n] in a search result (page number for PDFs)" }
  },
  "required": ["path", "n"]
}
```

Update `eval/tensorzero/config/answer_hotpot/system.minijinja` lines 1-3:

```
You answer a question using glossa, a document-search tool with two tools:
- search(query): BM25 keyword search (morphology-aware). Returns numbered hits `[#n] path · label · snippet`.
- read(path, n): read chunk number n (the [#n] from a search hit; for PDFs n is the page). It also shows prev/next chunk numbers — read n+1 to see more context.
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p kb-eval 2>&1 | tail -6`
Expected: PASS (the new test + existing eval tests).

- [ ] **Step 5: Commit**

```bash
git add eval/src/backend/glossa_tools.rs eval/tensorzero/config/tools/read.json eval/tensorzero/config/answer_hotpot/system.minijinja
git commit -m "feat(eval): mirror read-by-number + numbered search contract"
```

---

### Task 6: C-free invariant + full-suite gate

**Files:** none (verification only)

- [ ] **Step 1: Confirm C-free**

Run: `cargo tree -p glossa -i cc 2>&1 | tail -2`
Expected: `warning: nothing to print.` (cc absent).

- [ ] **Step 2: Full core + eval suites**

Run: `cargo test -p glossa 2>&1 | tail -3 && cargo test -p kb-eval 2>&1 | tail -3`
Expected: both `test result: ok.`

- [ ] **Step 3: Build release binaries**

Run: `cargo build --release 2>&1 | tail -2`
Expected: `Finished`.

- [ ] **Step 4: Commit (if any incidental fixes were needed)**

```bash
git add -A
git commit -m "chore: verify C-free + full suite green for read-by-number"
```

---

## Notes for the implementer

- **Re-index required after merge.** `ord` is a new field; existing indexes lack it. The user will rebuild kb-test (`kb reindex`) before the next eval run. Do not attempt to migrate old indexes.
- **Validation after Plan A** (manual, by the controller, not a task): rebuild kb-test, run one `kb-eval` real-domain pass, confirm reads no longer error and `judge` is no longer depressed by failed reads.
- **`grep` is Plan B** — do not add it here. The trigram accelerator is Step 2.
