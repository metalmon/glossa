# Tool Unification (shared render module + images) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the eval harness call the *same* tool-rendering code as the MCP server (one source of truth, no drift) and have `read` return images to the vision model in both surfaces.

**Architecture:** A new `src/tools.rs` (glossa core) renders each agent tool's model-facing output exactly once (`search`/`grep`/`glob` → text + hits; `read` → text + images, no cap, one unified footer). `src/mcp.rs` and `eval/src/backend/glossa_tools.rs` both call it. The eval's episode loop appends an image user-message after a read's tool_result (the TZ `tool_result.result` is a string, so images can't embed there). An image-path spike gates the work.

**Tech Stack:** Rust, tantivy, rmcp (MCP), TensorZero (eval), LM Studio qwen3.5-4b (vision-capable).

**Scope:** Refactor — move formatting into a shared module; reuse all existing engine code (`search_filtered`, `grep`, `glob_docs`, `read_chunk_by_ord`, `extract_images`). No new engine logic. The PDF-extractor fix (grep-bloat root) is a SEPARATE future feature.

## Global Constraints

- **C-free:** `cargo tree -p glossa -i cc` empty. No new deps.
- **Single source of truth:** the `glossa::tools::*` render functions are the ONLY place tool output is formatted. MCP and eval MUST produce byte-identical model-facing text — a parity test enforces this.
- **File-First:** read body comes from the index (`read_chunk_by_ord`); images from the file via `extract_images` (gracefully absent if the file is gone).
- **read returns the FULL chunk** — no truncation (drop `cap_read`).
- **TDD:** failing test first. `cargo test -p glossa` / `cargo test -p kb-eval`.

---

### Task 1: SPIKE — verify the TZ → LM Studio image path

**Goal:** Confirm a TensorZero image content block in a user message reaches LM Studio's `qwen3.5-4b` and is interpreted as an image. **This gates the rest** — if it fails, STOP and report; the image delivery needs rethinking.

**Files:** none committed (throwaway probe). Document the working format in the report.

- [ ] **Step 1: Make a tiny known test image**

Run (creates a 2×2 solid-red PNG, base64 to a var):
```bash
cd /e/glossa
python -c "import base64,struct,zlib;
def png(rgb):
 raw=b''.join(b'\x00'+bytes(rgb)*2 for _ in range(2))
 def chunk(t,d):
  c=t+d; return struct.pack('>I',len(d))+c+struct.pack('>I',zlib.crc32(c)&0xffffffff)
 sig=b'\x89PNG\r\n\x1a\n'
 ihdr=struct.pack('>IIBBBBB',2,2,8,2,0,0,0)
 idat=zlib.compress(raw)
 return sig+chunk(b'IHDR',ihdr)+chunk(b'IDAT',idat)+chunk(b'IEND',b'')
print(base64.b64encode(png((255,0,0))).decode())" > /tmp/img_b64.txt
echo "b64 len: $(wc -c < /tmp/img_b64.txt)"
```

- [ ] **Step 2: Probe the gateway with candidate TZ image formats**

The eval calls `POST /inference` with `function_name: answer_hotpot`, `input.messages`. Send a user message that mixes text + an image block and a question the answer can only come from the image ("what color is the square?"). Try the TZ image content-block shapes in order until one returns "red":

Format A (inline base64):
```bash
B64=$(cat /tmp/img_b64.txt)
curl -s http://localhost:3000/inference -H "Content-Type: application/json" -d "{\"function_name\":\"answer_hotpot\",\"input\":{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What color is the square in the image? One word.\"},{\"type\":\"image\",\"mime_type\":\"image/png\",\"data\":\"$B64\"}]}]}}" | head -c 600
```

Format B (image.url data URI), if A fails:
```bash
curl -s http://localhost:3000/inference -H "Content-Type: application/json" -d "{\"function_name\":\"answer_hotpot\",\"input\":{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What color is the square? One word.\"},{\"type\":\"image\",\"image\":{\"url\":\"data:image/png;base64,$B64\"}}]}]}}" | head -c 600
```

(If both error on schema, consult the TZ version's content-block docs via context7 `tensorzero` and try the documented image shape. The gateway runs `tensorzero-2026.6.0`.)

- [ ] **Step 3: Judge the spike**

Expected PASS: the model's text answer contains "red" (it saw the image). Note WHICH format string worked (the exact JSON shape) — Task 4 reuses it verbatim.
Expected FAIL: schema error on every format, or the model answers about color it cannot know / refuses → **STOP, report BLOCKED**: the eval cannot deliver images to this model/gateway; the unification proceeds text-only and images stay MCP-only (revert that part of the plan).

- [ ] **Step 4: Record the result**

Write the working image-content-block JSON shape (or the BLOCKED finding) to `.superpowers/sdd/task-1-report.md`. No commit (probe only).

---

### Task 2: `glossa::tools::{search, grep, glob}` — shared text render

**Files:**
- Create: `src/tools.rs`
- Modify: `src/lib.rs` (add `pub mod tools;`)
- Modify: `src/mcp.rs` (`search`/`grep`/`glob` call the shared fns)
- Modify: `eval/src/backend/glossa_tools.rs` (`run_search`/`run_grep`/`run_glob` call the shared fns)
- Test: a parity test in `src/tools.rs`

**Interfaces:**
- Consumes: `DocIndex::search_filtered`, `crate::grep::{grep, GrepOpts}`, `crate::glob::glob_docs`, `RankedHit::display_line`, `GrepHit::display_line`, `crate::trace::TraceLog`.
- Produces (in `glossa::tools`):
  - `pub fn search(idx: &DocIndex, query: &str, limit: usize, glob: Option<&str>, file_type: Option<&str>, trace: &TraceLog) -> (String, Vec<RankedHit>)`
  - `pub fn grep(idx: &DocIndex, pattern: &str, opts: &GrepOpts, trace: &TraceLog) -> String`
  - `pub fn glob(idx: &DocIndex, pattern: &str, trace: &TraceLog) -> String`

- [ ] **Step 1: Write the failing parity test** (in the new `src/tools.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::store::DocIndex;
    use crate::model::Chunk;
    use crate::trace::TraceLog;
    use std::path::PathBuf;

    fn idx() -> (tempfile::TempDir, DocIndex) {
        let d = tempfile::tempdir().unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        i.write_chunks(&[
            Chunk { doc_path: PathBuf::from("АБАК.pdf"), location: "p.7".into(), file_type: "pdf".into(), text: "параметр maxTsdr равен 3000".into() },
        ]).unwrap();
        (d, i)
    }

    #[test]
    fn search_renders_numbered_or_empty() {
        let (_d, i) = idx();
        let t = TraceLog::disabled();
        let (body, hits) = search(&i, "maxTsdr", 10, None, None, &t);
        assert_eq!(hits.len(), 1);
        assert!(body.starts_with("[#7] ") && body.contains("maxTsdr"));
        let (empty, _) = search(&i, "nonexistentzzz", 10, None, None, &t);
        assert_eq!(empty, "(no results)");
    }

    #[test]
    fn grep_and_glob_render() {
        let (_d, i) = idx();
        let t = TraceLog::disabled();
        assert!(grep(&i, "maxTsdr", &crate::grep::GrepOpts::default(), &t).contains(":#7:"));
        assert_eq!(grep(&i, "nomatchzzz", &crate::grep::GrepOpts::default(), &t), "(no matches)");
        assert!(glob(&i, "*АБАК*", &t).contains("АБАК.pdf  (7 chunks)"));
        assert_eq!(glob(&i, "*nomatch*", &t), "(no documents match)");
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p glossa tools:: 2>&1 | tail -5`
Expected: FAIL — module `tools` not found.

- [ ] **Step 3: Implement** — create `src/tools.rs`

```rust
//! Single source of truth for the agent tools' model-facing output. Both the MCP server
//! (src/mcp.rs) and the kb-eval harness call these so prod and eval render identically.

use crate::grep::GrepOpts;
use crate::index::store::{DocIndex, RankedHit};
use crate::trace::TraceLog;
use serde_json::json;

/// BM25 search (optionally scoped). Returns (model text, hits for the caller's scoring).
pub fn search(idx: &DocIndex, query: &str, limit: usize, glob: Option<&str>, file_type: Option<&str>, trace: &TraceLog) -> (String, Vec<RankedHit>) {
    match idx.search_filtered(query, limit.max(1), glob, file_type) {
        Ok(hits) => {
            let th: Vec<_> = hits.iter().map(|h| json!({"path": h.path, "location": h.location, "score": h.score})).collect();
            trace.log("search", json!({"query": query}), json!(th));
            let body = if hits.is_empty() { "(no results)".to_string() }
                       else { hits.iter().map(|h| h.display_line()).collect::<Vec<_>>().join("\n") };
            (body, hits)
        }
        Err(e) => (format!("search error: {e}"), Vec::new()),
    }
}

/// ripgrep-style literal/regex search; model text only.
pub fn grep(idx: &DocIndex, pattern: &str, opts: &GrepOpts, trace: &TraceLog) -> String {
    match crate::grep::grep(idx, pattern, opts) {
        Ok(hits) => {
            trace.log("grep", json!({"pattern": pattern}), json!({"hits": hits.len()}));
            if hits.is_empty() { "(no matches)".to_string() }
            else { hits.iter().map(|h| h.display_line()).collect::<Vec<_>>().join("\n") }
        }
        Err(e) => format!("grep error: {e}"),
    }
}

/// List documents by path mask; model text only.
pub fn glob(idx: &DocIndex, pattern: &str, trace: &TraceLog) -> String {
    match crate::glob::glob_docs(idx, pattern) {
        Ok(docs) => {
            trace.log("glob", json!({"pattern": pattern}), json!({"docs": docs.len()}));
            if docs.is_empty() { "(no documents match)".to_string() }
            else { docs.iter().map(|(p, n)| format!("{p}  ({n} chunks)")).collect::<Vec<_>>().join("\n") }
        }
        Err(e) => format!("glob error: {e}"),
    }
}
```

Add `pub mod tools;` to `src/lib.rs`.

In `src/mcp.rs`, replace the bodies of the `search`, `grep`, `glob` tools to delegate. Example for `search` (mirror for grep/glob):

```rust
    async fn search(&self, Parameters(a): Parameters<SearchArgs>) -> Result<CallToolResult, McpError> {
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let (body, _hits) = crate::tools::search(&idx, &a.query, a.limit.unwrap_or(50), a.glob.as_deref(), a.file_type.as_deref(), &self.trace);
        Ok(CallToolResult::success(vec![Content::text(body)]))
    }
```
```rust
    async fn grep(&self, Parameters(a): Parameters<GrepArgs>) -> Result<CallToolResult, McpError> {
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let opts = crate::grep::GrepOpts { ignore_case: a.ignore_case.unwrap_or(false), fixed: a.fixed.unwrap_or(false), word: a.word.unwrap_or(false), glob: a.glob, file_type: a.file_type };
        Ok(CallToolResult::success(vec![Content::text(crate::tools::grep(&idx, &a.pattern, &opts, &self.trace))]))
    }
```
```rust
    async fn glob(&self, Parameters(a): Parameters<GlobArgs>) -> Result<CallToolResult, McpError> {
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(crate::tools::glob(&idx, &a.pattern, &self.trace))]))
    }
```

In `eval/src/backend/glossa_tools.rs`, replace `run_search`/`run_grep`/`run_glob` to delegate (returning the `(text, titles)` shape the eval's `exec` expects — titles from the hits):

```rust
pub fn run_search(idx: &DocIndex, query: &str, limit: usize, glob: Option<&str>, file_type: Option<&str>, trace: &TraceLog) -> (String, Vec<String>) {
    let (body, hits) = glossa::tools::search(idx, query, limit, glob, file_type, trace);
    (body, hits.iter().map(|h| h.location.clone()).collect())
}
pub fn run_grep(idx: &DocIndex, pattern: &str, opts: glossa::grep::GrepOpts, trace: &TraceLog) -> (String, Vec<String>) {
    (glossa::tools::grep(idx, pattern, &opts, trace), Vec::new())
}
pub fn run_glob(idx: &DocIndex, pattern: &str, trace: &TraceLog) -> (String, Vec<String>) {
    let body = glossa::tools::glob(idx, pattern, trace);
    (body, Vec::new())
}
```

(Keep the `exec` arms calling these unchanged: `run_search` already passes `glob`/`file_type`; `run_grep` still takes `opts` by value and forwards `&opts` to `glossa::tools::grep`; `run_glob` is unchanged.)

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p glossa 2>&1 | grep "test result" | head -1 && cargo test -p kb-eval 2>&1 | grep "test result" | head -1`
Expected: both `ok.` Then a quick byte-parity sanity:
Run: `grep -rn "display_line\|(no results)\|(no matches)\|(no documents" src/mcp.rs eval/src/backend/glossa_tools.rs` → Expected: NO inline formatting left (only calls into `crate::tools`/`glossa::tools`).

- [ ] **Step 5: Commit**

```bash
git add src/tools.rs src/lib.rs src/mcp.rs eval/src/backend/glossa_tools.rs
git commit -m "refactor(tools): shared search/grep/glob render for MCP + eval"
```

---

### Task 3: `glossa::tools::read` — text + images, no cap, one footer

**Files:**
- Modify: `src/tools.rs` (add `read` + `ReadOut`)
- Modify: `src/mcp.rs` (`read` delegates; keeps image wrapping)
- Modify: `eval/src/backend/glossa_tools.rs` (`run_read` delegates; drop `cap_read`)

**Interfaces:**
- Consumes: `DocIndex::read_chunk_by_ord`, `crate::read::{extract_images, DocImage}`.
- Produces: `pub struct ReadOut { pub text: String, pub images: Vec<crate::read::DocImage> }`; `pub fn read(idx: &DocIndex, path: &str, n: u64, trace: &TraceLog) -> ReadOut`.

- [ ] **Step 1: Write the failing test** (in `src/tools.rs` tests)

```rust
    #[test]
    fn read_returns_full_body_and_unified_footer() {
        let d = tempfile::tempdir().unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        let big = "Я".repeat(5000); // > old 4000-char cap
        i.write_chunks(&[
            Chunk { doc_path: PathBuf::from("d.md"), location: "S1".into(), file_type: "md".into(), text: big.clone() },
            Chunk { doc_path: PathBuf::from("d.md"), location: "S2".into(), file_type: "md".into(), text: "second".into() },
        ]).unwrap();
        let t = TraceLog::disabled();
        let out = read(&i, "d.md", 1, &t);
        assert!(out.text.contains(&big), "full body, no cap");                 // not truncated
        assert!(out.text.contains("next #2") && out.text.contains("end of document") == false);
        assert!(out.text.contains("‹ start of document · next #2 ›"));        // unified footer (MCP wording)
        assert_eq!(read(&i, "d.md", 99, &t).text, "no chunk #99 in d.md");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p glossa read_returns_full_body 2>&1 | tail -5`
Expected: FAIL — no `tools::read`.

- [ ] **Step 3: Implement** (add to `src/tools.rs`)

```rust
/// A read result: the chunk text (with prev/next footer) plus the file's images for vision models.
pub struct ReadOut {
    pub text: String,
    pub images: Vec<crate::read::DocImage>,
}

/// Read chunk `n` of `path`: full stored body + a unified prev/next footer, plus extracted images
/// (empty if the source file is absent — body still comes from the index). No truncation.
pub fn read(idx: &DocIndex, path: &str, n: u64, trace: &TraceLog) -> ReadOut {
    let chunk = match idx.read_chunk_by_ord(path, n) {
        Ok(Some(c)) => c,
        Ok(None) => return ReadOut { text: format!("no chunk #{n} in {path}"), images: Vec::new() },
        Err(e) => return ReadOut { text: format!("read error: {e}"), images: Vec::new() },
    };
    trace.log("read", json!({"path": path, "n": n}), json!({"path": path}));
    let footer = match (chunk.prev, chunk.next) {
        (Some(p), Some(nx)) => format!("\n\n‹ prev #{p} · next #{nx} ›"),
        (None, Some(nx)) => format!("\n\n‹ start of document · next #{nx} ›"),
        (Some(p), None) => format!("\n\n‹ prev #{p} · end of document ›"),
        (None, None) => String::new(),
    };
    let images = crate::read::extract_images(std::path::Path::new(path), 8).unwrap_or_default();
    ReadOut { text: format!("{}{}", chunk.body, footer), images }
}
```

In `src/mcp.rs`, replace the `read` body to delegate (keeping image wrapping; drop the inline footer + read_chunk_by_ord):

```rust
    async fn read(&self, Parameters(a): Parameters<ReadArgs>) -> Result<CallToolResult, McpError> {
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let out = crate::tools::read(&idx, &a.path, a.n as u64, &self.trace);
        let mut content = vec![Content::text(out.text)];
        if a.include_images.unwrap_or(true) {
            for img in out.images {
                let b64 = base64::engine::general_purpose::STANDARD.encode(&img.bytes);
                content.push(Content::image(b64, img.mime));
            }
        }
        Ok(CallToolResult::success(content))
    }
```

In `eval/src/backend/glossa_tools.rs`, replace `run_read` to delegate and DELETE `cap_read`. `run_read` now also surfaces images to the caller (Task 4 uses them), so change its return to carry images:

```rust
/// Read a chunk: full text + the chunk's images (for the vision model, delivered by the backend).
pub fn run_read(idx: &DocIndex, path: &str, n: u64, trace: &TraceLog) -> (String, Vec<glossa::read::DocImage>) {
    let out = glossa::tools::read(idx, path, n, trace);
    (out.text, out.images)
}
```

Update the `exec` "read" arm: it currently returns `(String, Vec<String>)`. Change `exec` to return the read's images too — make `exec` return `(String, Vec<String>, Vec<glossa::read::DocImage>)`, with non-read tools returning an empty image vec. (Task 4 consumes the images; the openai backend, which also calls `exec`, ignores the third field.) Update the `read` arm:

```rust
        "read" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let n = args.get("n").and_then(parse_n).unwrap_or(0);
            let (text, imgs) = run_read(idx, path, n, trace);
            (text, Vec::new(), imgs)
        }
```
…and the other arms return `(body, titles, Vec::new())`. (No `openai.rs` change needed: its `execute_tool` already takes `exec(...).0`, which is still the text string under a 3-tuple.)

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p glossa read_returns_full_body 2>&1 | tail -5 && cargo test -p glossa 2>&1 | grep "test result" | head -1`
Expected: PASS; full glossa suite green.

- [ ] **Step 5: Commit**

```bash
git add src/tools.rs src/mcp.rs eval/src/backend/glossa_tools.rs
git commit -m "refactor(tools): shared read (full body, unified footer, images)"
```

---

### Task 4: Deliver read images to the eval's vision model

**Files:**
- Modify: `eval/src/backend/tensorzero.rs` (`run_episode` appends an image user-message after a read tool_result)

**Interfaces:**
- Consumes: `glossa_tools::exec` now returns `(String, Vec<String>, Vec<glossa::read::DocImage>)`; the WORKING TZ image content-block shape from Task 1.

- [ ] **Step 1: Write the failing test** (append to `eval/src/backend/tensorzero.rs` tests)

A unit test on the message-building helper. First factor the image-message construction into a testable fn `image_user_message(images: &[glossa::read::DocImage]) -> Option<Value>` (returns a user message with TZ image content blocks, or None if empty), then:

```rust
    #[test]
    fn image_user_message_uses_working_tz_shape() {
        let imgs = vec![glossa::read::DocImage { mime: "image/png".into(), bytes: vec![1,2,3] }];
        let m = image_user_message(&imgs).unwrap();
        assert_eq!(m["role"], "user");
        let blocks = m["content"].as_array().unwrap();
        // exactly one image block, in the shape Task 1's spike proved works:
        assert!(blocks.iter().any(|b| b["type"] == "image"));
        assert!(image_user_message(&[]).is_none());
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p kb-eval image_user_message 2>&1 | tail -5`
Expected: FAIL — no `image_user_message`.

- [ ] **Step 3: Implement** — using the EXACT JSON shape Task 1 proved (the example below assumes Format A `{type:image, mime_type, data}`; if the spike found Format B, use that shape instead):

```rust
/// Build a user message carrying the read's images as TZ image content blocks (vision input),
/// or None when there are no images. Uses the content-block shape verified by the Task-1 spike.
fn image_user_message(images: &[glossa::read::DocImage]) -> Option<Value> {
    if images.is_empty() { return None; }
    use base64::Engine as _;
    let mut content = vec![json!({"type": "text", "text": "(images from the chunk you just read)"})];
    for img in images {
        let b64 = base64::engine::general_purpose::STANDARD.encode(&img.bytes);
        content.push(json!({"type": "image", "mime_type": img.mime, "data": b64}));
    }
    Some(json!({"role": "user", "content": content}))
}
```

In `run_episode`, after pushing the read's `tool_result` message, push the image message when present. Adapt the tool-call loop to receive `exec`'s third return value and, for the call, append:

```rust
            let (result, titles, images) = exec(name, &args);
            surfaced_titles.extend(titles);
            messages.push(json!({ "role": "user", "content": [{ "type": "tool_result", "id": id, "name": name, "result": result }] }));
            if let Some(img_msg) = image_user_message(&images) {
                messages.push(img_msg);
            }
```

(The `exec` closure in `answer()` must now forward three values; update its type and the `run_episode` signature's `X` bound accordingly.)

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p kb-eval 2>&1 | grep "test result" | head -2`
Expected: `ok.`

- [ ] **Step 5: Commit**

```bash
git add eval/src/backend/tensorzero.rs
git commit -m "feat(eval): deliver read images to the vision model as a user message"
```

---

### Task 5: C-free invariant + full-suite gate

**Files:** none (verification only)

- [ ] **Step 1: C-free**

Run: `cargo tree -p glossa -i cc 2>&1 | tail -2`
Expected: `warning: nothing to print.`

- [ ] **Step 2: Full suites + no inline formatting remains**

Run: `cargo test -p glossa 2>&1 | grep "test result" | head -1 && cargo test -p kb-eval 2>&1 | grep "test result" | head -1`
Expected: both `ok.`
Run: `grep -rn "cap_read" eval/ src/` → Expected: no matches (the divergent cap is gone).

- [ ] **Step 3: Release build** (defer if `kb.exe`/`kb-eval.exe` are locked — build `-p kb-eval`)

Run: `cargo build -p kb-eval --release 2>&1 | tail -1`
Expected: `Finished`.

- [ ] **Step 4: Commit (if incidental fixes)**

```bash
git add -A
git commit -m "chore: verify C-free + suites green for tool unification"
```

---

## Notes for the implementer

- **Task 1 gates everything** — if the image path can't be proven, stop and report; Tasks 3–4's image parts are then dropped (text unification still proceeds).
- **Parity is the point:** after Task 2, the only place tool text is formatted is `glossa::tools`. Grep the two files to confirm no inline formatting survived.
- **Reuse only:** no new engine code — `search_filtered`/`grep`/`glob_docs`/`read_chunk_by_ord`/`extract_images` already exist.
- **Out of scope:** the PDF-extractor fix (the grep-bloat/garbage root) is the next, separate feature.
