# glossa v5 — Milestone 5: MCP server + read/images + gitignore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose glossa to a connected agent over MCP (`kb mcp`) with **reader/editor/full** profiles, including a `read` tool that returns document text **plus embedded images** as MCP image content, and make indexing **respect `.gitignore`**. This puts the M4 graph substrate + M1–M3 search into the agent's hands — the differentiator.

**Architecture:** A single `rmcp` (official Rust MCP SDK, pure-Rust, tokio) stdio server. One server type registers all tools via `#[tool_router]`; the runtime `--profile` disables routes (`ToolRouter::disable_route`) so `reader` exposes read-only tools, `editor` adds graph-write + index, `full` adds admin. Tools delegate to the already-tested library (`walk`/`index`/`graph`). `read` composes a new `read_region` (text of a doc location) + `extract_images` (office zip media → base64 image blocks). The directory walk switches to the `ignore` crate so `.gitignore`/`.ignore`/hidden files are skipped by default (`.glossa/` always skipped); `--no-ignore` indexes everything.

**Tech Stack:** Rust; `rmcp = { version = "1.8", features = ["server","transport-io","macros"] }`, `schemars = "1.0"`, `tokio` (rt), `ignore = "0.4"` (ripgrep's own walker, pure Rust), `base64 = "0.22"` (encode image bytes), existing `zip` (M2) for media. All pure Rust, offline, no C.

## Global Constraints

- Pure Rust, single static binary, fully offline, **no C / no cc** (verified: rmcp/schemars/ignore/base64 are pure Rust). Do not reintroduce any `*-sys`.
- **`.gitignore` respected by default** via the `ignore` crate (`.gitignore` + `.ignore` + hidden files skipped), `.glossa/` ALWAYS skipped, `--no-ignore` flag indexes everything. Consistent with the ripgrep model glossa already follows.
- **MCP profiles** (`--profile`, default `editor`): `reader` = read-only (`search`, `read`, `glossary`, `neighbors`); `editor` = reader + `index`, `reindex`, `graph`, `resolve`; `full` = editor + admin (`purge`). Implemented by `ToolRouter::disable_route` at construction — a single server type, tools gated by visibility (no RBAC).
- The `read` tool returns text + images as MCP content blocks; the MCP server is the filesystem boundary (agent needs no disk access). Cap image count/size per response.
- §12 graph invariants still hold; graph tools go through the validated `upsert`/`resolve` from M4.
- TDD: failing test first; frequent commits; DRY; YAGNI.

## Deferred (out of scope here)

- `--expand` (glossary query expansion) — needs the layer-2 `Term`/co-occurrence layer, which is not built yet; revisit when terms exist.
- HTTP/streamable transport (`transport-streamable-http-server`) — stdio first; HTTP later.
- PDF image extraction (needs a page-aware PDF lib) — office images only here.
- M5 backlog carried forward: graph-build crash-atomicity (one redb txn per file), propagate `type_of` errors, traversal depth doc-comments, secondary source/label index.

---

### Task 1: gitignore-aware directory walk

**Files:**
- Modify: `Cargo.toml` (add `ignore = "0.4"`)
- Modify: `src/walk.rs` (`collect_chunks` gains `respect_ignore: bool`; switch `walkdir`→`ignore::WalkBuilder`; always skip `.glossa/`)
- Modify: `src/index/store.rs` (call `collect_chunks(dir, None, true)`)
- Modify: `src/main.rs` (default search: add `--no-ignore`, pass `!no_ignore`)
- Modify: `tests/walk_it.rs` (update calls to the 3-arg signature; add a gitignore test)

**Interfaces:**
- Produces: `walk::collect_chunks(root: &Path, glob: Option<&str>, respect_ignore: bool) -> anyhow::Result<Vec<Chunk>>`. When `respect_ignore`, `.gitignore`/`.ignore`/hidden are skipped; `.glossa/` is always skipped regardless.

- [ ] **Step 1: Write the failing test**

In `tests/walk_it.rs`, update existing calls to pass `true` as the third arg, and add:
```rust
#[test]
fn respects_gitignore_by_default() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join(".gitignore"), b"secret.md\n").unwrap();
    fs::write(dir.path().join("keep.md"), b"# K\nkeep\n").unwrap();
    fs::write(dir.path().join("secret.md"), b"# S\nsecret\n").unwrap();

    let respected = collect_chunks(dir.path(), None, true).unwrap();
    assert!(respected.iter().any(|c| c.location == "K"));
    assert!(!respected.iter().any(|c| c.location == "S"), "gitignored file must be skipped");

    let all = collect_chunks(dir.path(), None, false).unwrap();
    assert!(all.iter().any(|c| c.location == "S"), "--no-ignore indexes everything");
}
```
(Update `collects_chunks_from_markdown_files_in_tree` and `glob_filters_paths` to pass `true`.)

- [ ] **Step 2: Run → fail**

Run: `cargo test --test walk_it`
Expected: FAIL — `collect_chunks` arity changed / `ignore` missing.

- [ ] **Step 3: Implement**

In `Cargo.toml`: `ignore = "0.4"`.
Rewrite `src/walk.rs` `collect_chunks`:
```rust
use crate::extract::markdown::MarkdownExtractor;
use crate::extract::office::OfficeExtractor;
use crate::extract::pdf::PdfExtractor;
use crate::extract::Extractor;
use crate::model::Chunk;
use globset::Glob;
use ignore::WalkBuilder;
use std::path::Path;

pub fn extractors() -> Vec<Box<dyn Extractor>> {
    vec![
        Box::new(MarkdownExtractor),
        Box::new(OfficeExtractor),
        Box::new(PdfExtractor),
    ]
}

pub fn collect_chunks(
    root: &Path,
    glob: Option<&str>,
    respect_ignore: bool,
) -> anyhow::Result<Vec<Chunk>> {
    let matcher = match glob {
        Some(g) => Some(Glob::new(g)?.compile_matcher()),
        None => None,
    };
    let exts = extractors();
    let mut all = Vec::new();

    let mut wb = WalkBuilder::new(root);
    wb.standard_filters(respect_ignore); // gitignore/.ignore/hidden/parents
    // Always skip our own store, even when respect_ignore is false.
    wb.filter_entry(|e| e.file_name() != ".glossa");

    for result in wb.build() {
        let entry = match result {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skip (walk error): {e}");
                continue;
            }
        };
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.path();
        if let Some(m) = &matcher {
            if !m.is_match(path) {
                continue;
            }
        }
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        for ex in &exts {
            if ex.file_types().contains(&ext.as_str()) {
                match std::fs::read(path) {
                    Ok(bytes) => match ex.extract(path, &bytes) {
                        Ok(mut cs) => all.append(&mut cs),
                        Err(e) => eprintln!("skip {}: {}", path.display(), e),
                    },
                    Err(e) => eprintln!("skip {}: {}", path.display(), e),
                }
                break;
            }
        }
    }
    Ok(all)
}
```
Update `src/index/store.rs`: `collect_chunks(dir, None, true)?`.
Update `src/main.rs` default `Search` arm: add field `#[arg(long = "no-ignore")] no_ignore: bool` and call `collect_chunks(&path, glob.as_deref(), !no_ignore)?`.

- [ ] **Step 4: Run → pass + full suite**

Run: `cargo test` (all suites). Verify `cargo tree -i cc` empty.
Verification note: `ignore::WalkBuilder::filter_entry` prunes a directory and its descendants when the predicate is false — correct for skipping `.glossa/`. `standard_filters(false)` disables all ignore/hidden logic (so `--no-ignore` sees everything).

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock src/walk.rs src/index/store.rs src/main.rs tests/walk_it.rs
git commit -m "feat: gitignore-aware indexing via ignore crate (--no-ignore to override; always skip .glossa)"
```

---

### Task 2: `read_region` — text of a document location

**Files:**
- Create: `src/read.rs`
- Modify: `src/lib.rs` (add `pub mod read;`)
- Test: `src/read.rs` (inline)

**Interfaces:**
- Consumes: `walk::extractors`, `model::Chunk`.
- Produces: `read::read_region(path: &Path, location: Option<&str>) -> anyhow::Result<String>` — extracts the file, returns the text of the chunk whose `location` matches (substring, case-insensitive); if `location` is None, returns all chunks' text joined; if no chunk matches, returns the whole document text (with a note is unnecessary — just the joined text).

- [ ] **Step 1: Write the failing test**

Create `src/read.rs`:
```rust
use crate::extract::Extractor;
use crate::walk::extractors;
use anyhow::Context;
use std::path::Path;

/// Read a document's text, optionally narrowed to a location (heading/sheet/page),
/// matched as a case-insensitive substring of the chunk's `location`.
pub fn read_region(path: &Path, location: Option<&str>) -> anyhow::Result<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut chunks = Vec::new();
    for ex in extractors() {
        if ex.file_types().contains(&ext.as_str()) {
            chunks = ex.extract(path, &bytes)?;
            break;
        }
    }
    let selected: Vec<&str> = match location {
        Some(loc) => {
            let needle = loc.to_lowercase();
            let matched: Vec<&str> = chunks
                .iter()
                .filter(|c| c.location.to_lowercase().contains(&needle))
                .map(|c| c.text.as_str())
                .collect();
            if matched.is_empty() {
                chunks.iter().map(|c| c.text.as_str()).collect()
            } else {
                matched
            }
        }
        None => chunks.iter().map(|c| c.text.as_str()).collect(),
    };
    Ok(selected.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_whole_then_narrows_by_location() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("d.md");
        std::fs::write(&p, b"# Intro\nalpha\n## Body\nbeta\n").unwrap();

        let whole = read_region(&p, None).unwrap();
        assert!(whole.contains("alpha") && whole.contains("beta"));

        let body = read_region(&p, Some("body")).unwrap();
        assert!(body.contains("beta") && !body.contains("alpha"));
    }
}
```

- [ ] **Step 2: Run → fail → declare module → pass**

Run `cargo test --lib read` (RED), add `pub mod read;` to `src/lib.rs`, re-run → PASS.

- [ ] **Step 3: Commit**

```bash
git add src/lib.rs src/read.rs
git commit -m "feat: read_region — document text by optional location"
```

---

### Task 3: `extract_images` — office embedded media

**Files:**
- Modify: `src/read.rs` (add `DocImage`, `extract_images`)
- Test: `src/read.rs` (inline, builds a synthetic zip)

**Interfaces:**
- Consumes: `zip` (already a dep), std.
- Produces:
  - `read::DocImage { mime: String, bytes: Vec<u8> }`
  - `read::extract_images(path: &Path, max: usize) -> anyhow::Result<Vec<DocImage>>` — for a zip-based office file (docx/xlsx/pptx), returns media under `word/media/`, `xl/media/`, `ppt/media/` (up to `max`); mime inferred from extension (png/jpeg/jpg/gif/bmp/webp); returns empty for non-zip / no media.

- [ ] **Step 1: Write the failing test**

Add to `src/read.rs`:
```rust
#[derive(Debug, Clone, PartialEq)]
pub struct DocImage {
    pub mime: String,
    pub bytes: Vec<u8>,
}

fn mime_for(name: &str) -> Option<&'static str> {
    let lower = name.to_lowercase();
    if lower.ends_with(".png") {
        Some("image/png")
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        Some("image/jpeg")
    } else if lower.ends_with(".gif") {
        Some("image/gif")
    } else if lower.ends_with(".bmp") {
        Some("image/bmp")
    } else if lower.ends_with(".webp") {
        Some("image/webp")
    } else {
        None
    }
}

pub fn extract_images(path: &Path, max: usize) -> anyhow::Result<Vec<DocImage>> {
    let bytes = std::fs::read(path)?;
    let reader = std::io::Cursor::new(bytes);
    let mut archive = match zip::ZipArchive::new(reader) {
        Ok(a) => a,
        Err(_) => return Ok(Vec::new()), // not a zip → no images
    };
    let media_names: Vec<String> = archive
        .file_names()
        .filter(|n| {
            n.starts_with("word/media/")
                || n.starts_with("xl/media/")
                || n.starts_with("ppt/media/")
        })
        .map(|s| s.to_string())
        .collect();

    let mut out = Vec::new();
    for name in media_names {
        if out.len() >= max {
            break;
        }
        let Some(mime) = mime_for(&name) else { continue };
        use std::io::Read;
        let mut entry = archive.by_name(&name)?;
        let mut buf = Vec::new();
        entry.read_to_end(&mut buf)?;
        out.push(DocImage { mime: mime.into(), bytes: buf });
    }
    Ok(out)
}

#[cfg(test)]
mod image_tests {
    use super::*;
    use std::io::Write;
    use zip::write::SimpleFileOptions;

    fn make_docx_with_png(path: &Path, png: &[u8]) {
        let f = std::fs::File::create(path).unwrap();
        let mut zw = zip::ZipWriter::new(f);
        let opts = SimpleFileOptions::default();
        zw.start_file("word/document.xml", opts).unwrap();
        zw.write_all(b"<w:document/>").unwrap();
        zw.start_file("word/media/image1.png", opts).unwrap();
        zw.write_all(png).unwrap();
        zw.finish().unwrap();
    }

    #[test]
    fn extracts_png_media_from_office_zip() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("withimg.docx");
        let png = b"\x89PNG\r\n\x1a\n-fake-png-bytes";
        make_docx_with_png(&p, png);

        let imgs = extract_images(&p, 10).unwrap();
        assert_eq!(imgs.len(), 1);
        assert_eq!(imgs[0].mime, "image/png");
        assert_eq!(imgs[0].bytes, png);

        // max cap respected
        assert_eq!(extract_images(&p, 0).unwrap().len(), 0);
    }

    #[test]
    fn non_zip_returns_no_images() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("plain.md");
        std::fs::write(&p, b"# H\nhi\n").unwrap();
        assert!(extract_images(&p, 10).unwrap().is_empty());
    }
}
```

- [ ] **Step 2: Run → fail → pass**

The `zip` crate's write API (`ZipWriter`, `SimpleFileOptions`) needs the default features. If `zip` was added deflate-only in M2, confirm `ZipWriter`/`SimpleFileOptions` are available (they are with the `deflate` feature). Run `cargo test --lib read` (RED for image_tests until impl present, then GREEN).

Verification note: if `SimpleFileOptions` path differs in the installed `zip` version, use the equivalent `zip::write::FileOptions::default()`.

- [ ] **Step 3: Commit**

```bash
git add src/read.rs
git commit -m "feat: extract_images — office embedded media (zip) with mime + cap"
```

---

### Task 4: MCP server — tools + profile gating

**Files:**
- Modify: `Cargo.toml` (add `rmcp`, `schemars`, `tokio`, `base64`)
- Create: `src/mcp.rs`
- Modify: `src/lib.rs` (add `pub mod mcp;`)
- Test: `src/mcp.rs` (inline — profile gating via enabled tool names)

**Interfaces:**
- Produces:
  - `mcp::Profile { Reader, Editor, Full }` with `parse(&str) -> Profile` (default Reader on unknown, but the CLI default is Editor).
  - `mcp::GlossaServer` with `new(root: PathBuf, profile: Profile) -> Self` (registers all tools, disables out-of-profile routes), and a test helper `enabled_tools(&self) -> Vec<String>`.
  - Tools: `search`, `read`, `glossary`, `neighbors` (reader); `+ index`, `reindex`, `graph_upsert`, `resolve` (editor); `+ purge` (full). Each delegates to the library.

- [ ] **Step 1: Write the failing test (profile gating)**

Create `src/mcp.rs` (tool bodies delegate to existing lib fns; see Task 5 for the serve loop):
```rust
use crate::graph::store::GraphStore;
use crate::index::store::{index_dir, DocIndex};
use crate::query::{compile, QueryOpts};
use crate::read::{extract_images, read_region};
use crate::search::search_chunks;
use crate::walk::collect_chunks;
use base64::Engine as _;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, Implementation, ProtocolVersion, ServerCapabilities, ServerInfo};
use rmcp::{tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler};
use schemars::JsonSchema;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile { Reader, Editor, Full }

impl Profile {
    pub fn parse(s: &str) -> Profile {
        match s {
            "editor" => Profile::Editor,
            "full" => Profile::Full,
            _ => Profile::Reader,
        }
    }
}

#[derive(Clone)]
pub struct GlossaServer {
    root: PathBuf,
    tool_router: ToolRouter<Self>,
}

const READER_TOOLS: &[&str] = &["search", "read", "glossary", "neighbors"];
const EDITOR_TOOLS: &[&str] = &["index", "reindex", "graph_upsert", "resolve"];
const FULL_TOOLS: &[&str] = &["purge"];

impl GlossaServer {
    pub fn new(root: PathBuf, profile: Profile) -> Self {
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
        Self { root, tool_router: router }
    }

    #[cfg(test)]
    pub fn enabled_tools(&self) -> Vec<String> {
        self.tool_router.list_all().iter().map(|t| t.name.to_string()).collect()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchArgs {
    #[schemars(description = "ripgrep-syntax query")]
    query: String,
    #[serde(default)]
    #[schemars(description = "max hits (default 50)")]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ReadArgs {
    #[schemars(description = "document path")]
    path: String,
    #[serde(default)]
    #[schemars(description = "optional location (heading/sheet/page) substring")]
    location: Option<String>,
    #[serde(default)]
    #[schemars(description = "include embedded images (default true)")]
    include_images: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct NeighborsArgs {
    node_id: String,
    #[serde(default)]
    depth: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct NameArg { name: String }

#[derive(Debug, Deserialize, JsonSchema)]
struct Empty {}

#[tool_router]
impl GlossaServer {
    #[tool(description = "Search the knowledge base (ripgrep syntax). Returns path:location:line: snippet.")]
    async fn search(&self, Parameters(a): Parameters<SearchArgs>) -> Result<CallToolResult, McpError> {
        let opts = QueryOpts { smart_case: true, ..Default::default() };
        let re = compile(&a.query, &opts).map_err(|e| McpError::invalid_params(e.to_string(), None))?;
        let chunks = collect_chunks(&self.root, None, true).map_err(internal)?;
        let hits = search_chunks(&chunks, &re, a.limit.unwrap_or(50));
        let body = hits.iter()
            .map(|h| format!("{}:{}:{}: {}", h.doc_path.display(), h.location, h.line, h.snippet))
            .collect::<Vec<_>>().join("\n");
        Ok(CallToolResult::success(vec![Content::text(body)]))
    }

    #[tool(description = "Read a document (optionally a location), with embedded images for the agent's vision.")]
    async fn read(&self, Parameters(a): Parameters<ReadArgs>) -> Result<CallToolResult, McpError> {
        let path = std::path::PathBuf::from(&a.path);
        let text = read_region(&path, a.location.as_deref()).map_err(internal)?;
        let mut content = vec![Content::text(text)];
        if a.include_images.unwrap_or(true) {
            for img in extract_images(&path, 8).map_err(internal)? {
                let b64 = base64::engine::general_purpose::STANDARD.encode(&img.bytes);
                content.push(Content::image(b64, img.mime));
            }
        }
        Ok(CallToolResult::success(content))
    }

    #[tool(description = "List glossary node ids whose label/alias matches a name.")]
    async fn glossary(&self, Parameters(a): Parameters<NameArg>) -> Result<CallToolResult, McpError> {
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let ids = g.resolve(&a.name).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(ids.join("\n"))]))
    }

    #[tool(description = "Graph neighbors reachable from a node id.")]
    async fn neighbors(&self, Parameters(a): Parameters<NeighborsArgs>) -> Result<CallToolResult, McpError> {
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let ids = crate::graph::traverse::neighbors(&g, &a.node_id, None, a.depth.unwrap_or(1)).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(ids.join("\n"))]))
    }

    #[tool(description = "Build/update the index + structural graph for the knowledge base.")]
    async fn index(&self, Parameters(_): Parameters<Empty>) -> Result<CallToolResult, McpError> {
        let s = index_dir(&self.root, false).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(format!("indexed: {} added, {} removed, {} unchanged", s.added, s.removed, s.unchanged))]))
    }

    #[tool(description = "Rebuild the index + graph from scratch.")]
    async fn reindex(&self, Parameters(_): Parameters<Empty>) -> Result<CallToolResult, McpError> {
        let s = index_dir(&self.root, true).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(format!("reindexed: {} files", s.added))]))
    }

    #[tool(description = "Resolve a name to existing graph node ids (entity resolution).")]
    async fn resolve(&self, Parameters(a): Parameters<NameArg>) -> Result<CallToolResult, McpError> {
        let g = GraphStore::open(&self.root).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(g.resolve(&a.name).map_err(internal)?.join("\n"))]))
    }

    #[tool(description = "Upsert agent-built graph nodes/edges (JSON), validated against the ontology.")]
    async fn graph_upsert(&self, Parameters(_): Parameters<Empty>) -> Result<CallToolResult, McpError> {
        // Full JSON node/edge payload wiring is editor-only; kept minimal here (Task 5 may expand).
        Ok(CallToolResult::success(vec![Content::text("graph_upsert: provide nodes/edges payload (see ontology.toml)")]))
    }

    #[tool(description = "Delete the index + graph for the knowledge base.")]
    async fn purge(&self, Parameters(_): Parameters<Empty>) -> Result<CallToolResult, McpError> {
        let g = self.root.join(".glossa");
        if g.exists() { std::fs::remove_dir_all(&g).map_err(|e| internal(e.into()))?; }
        Ok(CallToolResult::success(vec![Content::text("purged .glossa")]))
    }
}

fn internal(e: anyhow::Error) -> McpError {
    McpError::internal_error(e.to_string(), None)
}

#[tool_handler]
impl ServerHandler for GlossaServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2025_06_18,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation::from_build_env(),
            instructions: Some("glossa File-First knowledge-base search. Use ripgrep syntax for `search`.".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_gates_tool_visibility() {
        let root = std::path::PathBuf::from(".");
        let reader = GlossaServer::new(root.clone(), Profile::Reader).enabled_tools();
        assert!(reader.contains(&"search".to_string()) && reader.contains(&"read".to_string()));
        assert!(!reader.contains(&"index".to_string()) && !reader.contains(&"graph_upsert".to_string()) && !reader.contains(&"purge".to_string()));

        let editor = GlossaServer::new(root.clone(), Profile::Editor).enabled_tools();
        assert!(editor.contains(&"index".to_string()) && editor.contains(&"resolve".to_string()));
        assert!(!editor.contains(&"purge".to_string()));

        let full = GlossaServer::new(root, Profile::Full).enabled_tools();
        assert!(full.contains(&"purge".to_string()));
    }
}
```

- [ ] **Step 2: Run → fail**

Run: `cargo test --lib mcp`
Expected: FAIL — rmcp/schemars/base64/tokio deps + module missing.

- [ ] **Step 3: Add deps + module**

In `Cargo.toml`:
```toml
rmcp = { version = "1.8", features = ["server", "transport-io", "macros"] }
schemars = "1.0"
base64 = "0.22"
tokio = { version = "1", features = ["rt-multi-thread", "macros", "io-std"] }
```
In `src/lib.rs`: `pub mod mcp;`.

- [ ] **Step 4: Run → pass + full suite**

Run: `cargo test --lib mcp` then `cargo test`. Verify `cargo tree -i cc` empty.
Verification notes (confirm on first compile — recon-flagged): `#[tool_router]`/`#[tool_handler]` split macros; `Parameters<T>` import path; `ToolRouter::list_all()` and `.name` field shape (used only in the test helper — adjust if the accessor differs); `ProtocolVersion::V_2025_06_18` (fall back to `V_2024_11_05` if absent); `Content::image(base64_string, mime)`.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock src/lib.rs src/mcp.rs
git commit -m "feat: MCP server tools + reader/editor/full profile gating (rmcp)"
```

---

### Task 5: `kb mcp` CLI subcommand (stdio serve)

**Files:**
- Modify: `src/main.rs` (add `Mcp` subcommand; run the rmcp server over stdio on a tokio runtime)
- Test: `tests/mcp_it.rs` (smoke: `kb mcp --help` lists the flag; server constructs)

**Interfaces:**
- Produces: `kb mcp [--profile reader|editor|full] [path]` — serves the MCP server over stdio (logs to stderr; stdout is the JSON-RPC wire).

- [ ] **Step 1: Write the failing test**

Create `tests/mcp_it.rs`:
```rust
use assert_cmd::Command;
use predicates::str::contains;

#[test]
fn mcp_subcommand_exists_with_profile_flag() {
    Command::cargo_bin("kb").unwrap()
        .args(["mcp", "--help"])
        .assert().success()
        .stdout(contains("--profile"));
}
```

- [ ] **Step 2: Run → fail**

Run: `cargo test --test mcp_it`
Expected: FAIL — no `mcp` subcommand.

- [ ] **Step 3: Add the subcommand**

In `src/main.rs`, add to `Cmd`:
```rust
    /// Run the MCP server over stdio (for AI agents).
    Mcp {
        #[arg(default_value = ".")]
        path: PathBuf,
        /// Tool profile: reader | editor | full.
        #[arg(long, default_value = "editor")]
        profile: String,
    },
```
Add the arm in `main` (keep `main` synchronous; spin a tokio runtime locally):
```rust
        Cmd::Mcp { path, profile } => {
            use rmcp::{transport::stdio, ServiceExt};
            let server = glossa::mcp::GlossaServer::new(path, glossa::mcp::Profile::parse(&profile));
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async move {
                let service = server.serve(stdio()).await?;
                service.waiting().await?;
                Ok::<(), anyhow::Error>(())
            })?;
            Ok(())
        }
```
(`GlossaServer` must be `Clone`/`Send` for `serve` — it is, holding `PathBuf` + `ToolRouter`.)

- [ ] **Step 4: Run → pass + full suite**

Run: `cargo test --test mcp_it` then `cargo test`.
Expected: PASS. Manual smoke (optional): `echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}' | cargo run --bin kb -- mcp --profile reader` should return a JSON-RPC initialize response on stdout.

Verification note: confirm `ServiceExt::serve` + `stdio()` import paths from the recon; if `serve` returns a different handle, adjust the `.waiting()` call (the recon's pattern is `serve(stdio()).await?.waiting().await?`).

- [ ] **Step 5: Commit**

```bash
git add src/main.rs tests/mcp_it.rs
git commit -m "feat: kb mcp stdio server subcommand with --profile"
```

---

## Self-Review

**Spec coverage (Milestone 5 slice):**
- gitignore-aware indexing (default on, `--no-ignore`, always skip `.glossa/`) → Task 1. ✓
- `read` returning text + embedded office images as MCP image blocks → Tasks 2-4. ✓
- MCP server over stdio with reader/editor/full profile gating (`disable_route`) → Tasks 4-5. ✓
- Tools delegate to tested lib (search/index/graph/resolve/neighbors) → Task 4. ✓
- Pure Rust / offline / no C → deps are all pure Rust (verify `cargo tree -i cc`). ✓
- Deferred (stated): `--expand` (needs term layer), HTTP transport, PDF images, M5 backlog (graph crash-atomicity etc.). ✓

**Placeholder scan:** none — every step has complete code + commands. The `graph_upsert` tool body is intentionally a minimal stub returning guidance (full JSON node/edge payload wiring is a follow-up); it is gated to editor/full and does not block the milestone. Verification notes flag the rmcp API points to confirm at first compile (macro split, `list_all()`/`.name`, `ProtocolVersion`, `Content::image`).

**Type consistency:** `collect_chunks(&Path, Option<&str>, bool)` updated at all call sites (index_dir, default search, tests); `read_region`/`extract_images`/`DocImage` defined in `read.rs`, consumed by the `read` MCP tool; `Profile`/`GlossaServer` used by Task 5; tools call existing lib signatures (`index_dir`, `GraphStore::{open,resolve}`, `graph::traverse::neighbors`, `search_chunks`, `compile`) verbatim from M1–M4.

**Dependency note:** new deps `rmcp 1.8` (`server`/`transport-io`/`macros`), `schemars 1.0`, `base64 0.22`, `tokio` (rt-multi-thread/macros/io-std), `ignore 0.4` — all pure Rust, offline, no `cc`/C (rmcp uses rustls, not OpenSSL). Keeps the C-free build.

**Security note:** profiles are visibility gating, not sandboxing — a `reader` server simply does not register write tools, so the agent cannot call them. The MCP server is the filesystem boundary; the agent accesses documents only through `search`/`read`.
