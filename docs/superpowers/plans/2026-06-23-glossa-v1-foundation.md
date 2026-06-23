# glossa v1 — Foundation (Markdown → chunk → ripgrep search → CLI) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working `kb search` CLI that walks a directory, extracts Markdown into heading-scoped chunks, and matches them with **ripgrep-compatible** syntax, printing `path:location:line: snippet`.

**Architecture:** A single Rust crate `glossa` (lib + `kb` bin). Pure-logic core — `Extractor` trait, a Markdown extractor, a directory walker, a ripgrep→`regex` query compiler, and a line matcher. No index/DB/network yet; this milestone proves the File-First + ripgrep-syntax contract end-to-end on `.md` using only stable crates.

**Tech Stack:** Rust (edition 2021), `clap` v4 (CLI), `walkdir`, `regex` (the same engine ripgrep uses → 1:1 syntax), `globset` (path globs), `anyhow`; `tempfile` for tests.

## Global Constraints

- Pure Rust, **single static binary**, **fully offline** — no network calls, no native/system libs.
- Search syntax is **ripgrep, 1:1**, implemented via the `regex` crate (not an approximation).
- Files are the source of truth; results are always `path + location + snippet`.
- TDD: every behavior gets a failing test first. Frequent commits. DRY. YAGNI.

## v1 Milestone Roadmap (this plan = Milestone 1)

1. **Foundation (this plan):** md extraction + chunking + ripgrep search + CLI.
2. **Extractors:** `office_oxide` (docx/doc/pptx/ppt incl. legacy) + `calamine` (xls/xlsx/ods) + `pdf-extract` (pdf), all behind the `Extractor` trait; embedded-image extraction via `zip`.
3. **Index + language:** `tantivy` as accelerator + `MultiLangStemmer` (`lingua` detect → `rust-stemmers`); `--rank`/`--stem` flags; persistent `kb index`/`reindex` (mtime+hash manifest).
4. **Glossary + read:** SQLite glossary graph (headings + co-occurrence) + `--expand`; `read` with image content.
5. **MCP server:** `rmcp` exposing `search`/`read`/`glossary`/`index`/`reindex` (image content blocks).

Each milestone is its own plan, written when its predecessor is merged.

---

### Task 1: Project scaffold

**Files:**
- Create: `Cargo.toml`
- Create: `src/lib.rs`
- Create: `src/main.rs`
- Test: `src/lib.rs` (inline `#[cfg(test)]`)

**Interfaces:**
- Consumes: nothing.
- Produces: crate `glossa` (lib) + bin `kb`; empty module tree to be filled by later tasks.

- [ ] **Step 1: Write the failing test**

In `src/lib.rs`:
```rust
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        assert!(!version().is_empty());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test`
Expected: FAIL — no `Cargo.toml` / package not yet defined.

- [ ] **Step 3: Write minimal implementation**

Create `Cargo.toml`:
```toml
[package]
name = "glossa"
version = "0.0.1"
edition = "2021"

[lib]
name = "glossa"
path = "src/lib.rs"

[[bin]]
name = "kb"
path = "src/main.rs"

[dependencies]
anyhow = "1"
clap = { version = "4", features = ["derive"] }
walkdir = "2"
regex = "1"
globset = "0.4"

[dev-dependencies]
tempfile = "3"
```

Create `src/main.rs`:
```rust
fn main() -> anyhow::Result<()> {
    println!("glossa {}", glossa::version());
    Ok(())
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS (1 test). Also `cargo build` succeeds.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock src/lib.rs src/main.rs
git commit -m "feat: scaffold glossa crate (lib + kb bin)"
```

---

### Task 2: Core model — `Chunk` and `Extractor`

**Files:**
- Create: `src/model.rs`
- Create: `src/extract.rs`
- Modify: `src/lib.rs` (add `pub mod model; pub mod extract;`)
- Test: `src/model.rs` (inline)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `model::Chunk { doc_path: PathBuf, location: String, file_type: String, text: String }`
  - `extract::Extractor` trait: `fn file_types(&self) -> &'static [&'static str];`
    `fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>>;`

- [ ] **Step 1: Write the failing test**

In `src/model.rs`:
```rust
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub doc_path: PathBuf,
    pub location: String,
    pub file_type: String,
    pub text: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_constructs_and_compares() {
        let a = Chunk {
            doc_path: PathBuf::from("a.md"),
            location: "Intro".into(),
            file_type: "md".into(),
            text: "hello".into(),
        };
        let b = a.clone();
        assert_eq!(a, b);
    }
}
```

In `src/extract.rs`:
```rust
use crate::model::Chunk;
use std::path::Path;

pub trait Extractor {
    /// Lower-case file extensions this extractor handles (e.g. `["md"]`).
    fn file_types(&self) -> &'static [&'static str];
    /// Extract a file's raw bytes into heading/section-scoped chunks.
    fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>>;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test`
Expected: FAIL — `model`/`extract` modules not declared in `lib.rs`.

- [ ] **Step 3: Write minimal implementation**

In `src/lib.rs`, add above the `version` fn:
```rust
pub mod model;
pub mod extract;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/model.rs src/extract.rs
git commit -m "feat: add Chunk model and Extractor trait"
```

---

### Task 3: Markdown extractor (heading-scoped chunks)

**Files:**
- Create: `src/extract/markdown.rs`
- Modify: `src/extract.rs` (add `pub mod markdown;`)
- Test: `src/extract/markdown.rs` (inline)

**Interfaces:**
- Consumes: `model::Chunk`, `extract::Extractor`.
- Produces: `extract::markdown::MarkdownExtractor` (unit struct) implementing `Extractor`;
  emits one `Chunk` per heading section; `location` = heading path joined by `" > "`.

- [ ] **Step 1: Write the failing test**

In `src/extract/markdown.rs`:
```rust
use crate::extract::Extractor;
use crate::model::Chunk;
use std::path::Path;

pub struct MarkdownExtractor;

fn parse_atx_heading(line: &str) -> Option<(usize, String)> {
    let t = line.trim_start();
    if !t.starts_with('#') {
        return None;
    }
    let hashes = t.chars().take_while(|c| *c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = &t[hashes..];
    // CommonMark requires a space (or EOL) after the # run.
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }
    Some((hashes, rest.trim().to_string()))
}

fn push_chunk(path: &Path, heading_path: &[String], buf: &mut String, out: &mut Vec<Chunk>) {
    if buf.trim().is_empty() {
        buf.clear();
        return;
    }
    out.push(Chunk {
        doc_path: path.to_path_buf(),
        location: heading_path.join(" > "),
        file_type: "md".into(),
        text: std::mem::take(buf),
    });
}

impl Extractor for MarkdownExtractor {
    fn file_types(&self) -> &'static [&'static str] {
        &["md", "markdown"]
    }

    fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>> {
        let text = String::from_utf8_lossy(bytes);
        let mut out = Vec::new();
        let mut heading_path: Vec<String> = Vec::new();
        let mut buf = String::new();

        for line in text.lines() {
            if let Some((level, title)) = parse_atx_heading(line) {
                push_chunk(path, &heading_path, &mut buf, &mut out);
                heading_path.truncate(level.saturating_sub(1));
                heading_path.push(title);
            } else {
                buf.push_str(line);
                buf.push('\n');
            }
        }
        push_chunk(path, &heading_path, &mut buf, &mut out);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_by_headings_with_location_path() {
        let md = "# A\nintro\n## B\nbody b\n# C\nbody c\n";
        let chunks = MarkdownExtractor
            .extract(Path::new("d.md"), md.as_bytes())
            .unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].location, "A");
        assert_eq!(chunks[0].text.trim(), "intro");
        assert_eq!(chunks[1].location, "A > B");
        assert_eq!(chunks[1].text.trim(), "body b");
        assert_eq!(chunks[2].location, "C");
    }

    #[test]
    fn hash_without_space_is_not_a_heading() {
        let md = "#nothashtag is body\n";
        let chunks = MarkdownExtractor
            .extract(Path::new("d.md"), md.as_bytes())
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].location, "");
        assert!(chunks[0].text.contains("#nothashtag"));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test`
Expected: FAIL — `extract::markdown` module not declared.

- [ ] **Step 3: Write minimal implementation**

In `src/extract.rs`, add at the bottom:
```rust
pub mod markdown;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test markdown`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/extract.rs src/extract/markdown.rs
git commit -m "feat: markdown extractor with heading-scoped chunks"
```

---

### Task 4: ripgrep → `regex` query compiler

**Files:**
- Create: `src/query.rs`
- Modify: `src/lib.rs` (add `pub mod query;`)
- Test: `src/query.rs` (inline)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `query::QueryOpts { ignore_case: bool, smart_case: bool, word: bool, fixed: bool }` (derives `Default, Clone`)
  - `query::compile(pattern: &str, opts: &QueryOpts) -> anyhow::Result<regex::Regex>`

- [ ] **Step 1: Write the failing test**

In `src/query.rs`:
```rust
use regex::{Regex, RegexBuilder};

#[derive(Debug, Default, Clone)]
pub struct QueryOpts {
    pub ignore_case: bool,
    pub smart_case: bool,
    pub word: bool,
    pub fixed: bool,
}

pub fn compile(pattern: &str, opts: &QueryOpts) -> anyhow::Result<Regex> {
    let mut pat = if opts.fixed {
        regex::escape(pattern)
    } else {
        pattern.to_string()
    };
    if opts.word {
        pat = format!(r"\b(?:{})\b", pat);
    }
    let ci = opts.ignore_case
        || (opts.smart_case && !pattern.chars().any(|c| c.is_uppercase()));
    let re = RegexBuilder::new(&pat).case_insensitive(ci).build()?;
    Ok(re)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_strings_escape_metacharacters() {
        let re = compile("a.b", &QueryOpts { fixed: true, ..Default::default() }).unwrap();
        assert!(re.is_match("a.b"));
        assert!(!re.is_match("axb"));
    }

    #[test]
    fn word_boundaries_restrict_matches() {
        let re = compile("cat", &QueryOpts { word: true, ..Default::default() }).unwrap();
        assert!(re.is_match("the cat sat"));
        assert!(!re.is_match("category"));
    }

    #[test]
    fn smart_case_is_insensitive_for_lowercase_pattern() {
        let re = compile("cat", &QueryOpts { smart_case: true, ..Default::default() }).unwrap();
        assert!(re.is_match("Cat"));
    }

    #[test]
    fn smart_case_is_sensitive_when_pattern_has_uppercase() {
        let re = compile("Cat", &QueryOpts { smart_case: true, ..Default::default() }).unwrap();
        assert!(!re.is_match("cat"));
        assert!(re.is_match("Cat"));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test`
Expected: FAIL — `query` module not declared.

- [ ] **Step 3: Write minimal implementation**

In `src/lib.rs`, add to the module list:
```rust
pub mod query;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test query`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/query.rs
git commit -m "feat: ripgrep-compatible query compiler over regex crate"
```

---

### Task 5: Line matcher over chunks

**Files:**
- Create: `src/search.rs`
- Modify: `src/lib.rs` (add `pub mod search;`)
- Test: `src/search.rs` (inline)

**Interfaces:**
- Consumes: `model::Chunk`, `regex::Regex`.
- Produces:
  - `search::Hit { doc_path: PathBuf, location: String, line: usize, snippet: String }` (derives `Debug, PartialEq, Eq`)
  - `search::search_chunks(chunks: &[Chunk], re: &Regex, limit: usize) -> Vec<Hit>`
    (1-based `line` within the chunk text; stops at `limit`).

- [ ] **Step 1: Write the failing test**

In `src/search.rs`:
```rust
use crate::model::Chunk;
use regex::Regex;
use std::path::PathBuf;

#[derive(Debug, PartialEq, Eq)]
pub struct Hit {
    pub doc_path: PathBuf,
    pub location: String,
    pub line: usize,
    pub snippet: String,
}

pub fn search_chunks(chunks: &[Chunk], re: &Regex, limit: usize) -> Vec<Hit> {
    let mut hits = Vec::new();
    for c in chunks {
        for (i, line) in c.text.lines().enumerate() {
            if re.is_match(line) {
                hits.push(Hit {
                    doc_path: c.doc_path.clone(),
                    location: c.location.clone(),
                    line: i + 1,
                    snippet: line.trim().to_string(),
                });
                if hits.len() >= limit {
                    return hits;
                }
            }
        }
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(text: &str) -> Chunk {
        Chunk {
            doc_path: PathBuf::from("d.md"),
            location: "S".into(),
            file_type: "md".into(),
            text: text.into(),
        }
    }

    #[test]
    fn finds_matching_line_with_number_and_snippet() {
        let chunks = vec![chunk("alpha\nbeta cat\ngamma\n")];
        let re = Regex::new("cat").unwrap();
        let hits = search_chunks(&chunks, &re, 100);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].line, 2);
        assert_eq!(hits[0].snippet, "beta cat");
        assert_eq!(hits[0].location, "S");
    }

    #[test]
    fn respects_limit() {
        let chunks = vec![chunk("cat\ncat\ncat\n")];
        let re = Regex::new("cat").unwrap();
        let hits = search_chunks(&chunks, &re, 2);
        assert_eq!(hits.len(), 2);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test`
Expected: FAIL — `search` module not declared.

- [ ] **Step 3: Write minimal implementation**

In `src/lib.rs`, add to the module list:
```rust
pub mod search;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test search`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/search.rs
git commit -m "feat: line matcher producing path/location/line/snippet hits"
```

---

### Task 6: Directory walker with extractor dispatch + glob filter

**Files:**
- Create: `src/walk.rs`
- Modify: `src/lib.rs` (add `pub mod walk;`)
- Test: `tests/walk_it.rs` (integration test using `tempfile`)

**Interfaces:**
- Consumes: `extract::Extractor`, `extract::markdown::MarkdownExtractor`, `model::Chunk`.
- Produces:
  - `walk::extractors() -> Vec<Box<dyn Extractor>>` (Markdown only in this milestone)
  - `walk::collect_chunks(root: &Path, glob: Option<&str>) -> anyhow::Result<Vec<Chunk>>`
    (recursively reads supported files, dispatching by lower-cased extension; per-file
    extraction errors are logged to stderr and skipped, never fatal).

- [ ] **Step 1: Write the failing test**

In `tests/walk_it.rs`:
```rust
use glossa::walk::collect_chunks;
use std::fs;

#[test]
fn collects_chunks_from_markdown_files_in_tree() {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir(dir.path().join("sub")).unwrap();
    fs::write(dir.path().join("a.md"), b"# Title\nhello world\n").unwrap();
    fs::write(dir.path().join("sub/b.md"), b"# Other\nbye\n").unwrap();
    fs::write(dir.path().join("ignore.txt"), b"not indexed\n").unwrap();

    let chunks = collect_chunks(dir.path(), None).unwrap();
    assert_eq!(chunks.len(), 2);
    assert!(chunks.iter().any(|c| c.location == "Title"));
    assert!(chunks.iter().any(|c| c.location == "Other"));
}

#[test]
fn glob_filters_paths() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("keep.md"), b"# K\nkeep\n").unwrap();
    fs::write(dir.path().join("skip.md"), b"# S\nskip\n").unwrap();

    let chunks = collect_chunks(dir.path(), Some("**/keep.md")).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].location, "K");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test walk_it`
Expected: FAIL — `walk` module not declared / `collect_chunks` missing.

- [ ] **Step 3: Write minimal implementation**

Create `src/walk.rs`:
```rust
use crate::extract::markdown::MarkdownExtractor;
use crate::extract::Extractor;
use crate::model::Chunk;
use globset::Glob;
use std::path::Path;
use walkdir::WalkDir;

pub fn extractors() -> Vec<Box<dyn Extractor>> {
    vec![Box::new(MarkdownExtractor)]
}

pub fn collect_chunks(root: &Path, glob: Option<&str>) -> anyhow::Result<Vec<Chunk>> {
    let matcher = match glob {
        Some(g) => Some(Glob::new(g)?.compile_matcher()),
        None => None,
    };
    let exts = extractors();
    let mut all = Vec::new();

    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
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

In `src/lib.rs`, add to the module list:
```rust
pub mod walk;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --test walk_it`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/walk.rs tests/walk_it.rs
git commit -m "feat: directory walker with extractor dispatch and glob filter"
```

---

### Task 7: CLI wiring (`kb search`)

**Files:**
- Modify: `src/main.rs`
- Test: `tests/cli_it.rs` (integration test invoking the built binary)
- Modify: `Cargo.toml` (add `assert_cmd` + `predicates` dev-deps)

**Interfaces:**
- Consumes: `query::{compile, QueryOpts}`, `walk::collect_chunks`, `search::search_chunks`.
- Produces: `kb search <pattern> [path] [-i] [-w] [-F] [-g GLOB] [--limit N]`,
  printing one line per hit as `path:location:line: snippet`.

- [ ] **Step 1: Write the failing test**

In `tests/cli_it.rs`:
```rust
use assert_cmd::Command;
use predicates::str::contains;
use std::fs;

#[test]
fn kb_search_prints_matching_lines() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), b"# Intro\nthe cat sat\n").unwrap();

    let mut cmd = Command::cargo_bin("kb").unwrap();
    cmd.args(["search", "cat", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(contains("Intro").and(contains("the cat sat")));
}

#[test]
fn kb_search_word_flag_excludes_substring() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), b"# H\ncategory only\n").unwrap();

    let mut cmd = Command::cargo_bin("kb").unwrap();
    cmd.args(["search", "cat", "-w", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicates::str::is_empty());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test cli_it`
Expected: FAIL — `assert_cmd`/`predicates` missing and `main` has no `search` subcommand.

- [ ] **Step 3: Write minimal implementation**

In `Cargo.toml` under `[dev-dependencies]` add:
```toml
assert_cmd = "2"
predicates = "3"
```

Replace `src/main.rs`:
```rust
use clap::{Parser, Subcommand};
use glossa::query::{compile, QueryOpts};
use glossa::search::search_chunks;
use glossa::walk::collect_chunks;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "kb", about = "File-First knowledge-base search (ripgrep syntax)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Search files. PATTERN uses ripgrep syntax.
    Search {
        /// Search pattern (ripgrep regex syntax).
        pattern: String,
        /// Directory to search.
        #[arg(default_value = ".")]
        path: PathBuf,
        /// Case-insensitive (rg -i).
        #[arg(short = 'i', long = "ignore-case")]
        ignore_case: bool,
        /// Match whole words (rg -w).
        #[arg(short = 'w', long = "word-regexp")]
        word: bool,
        /// Treat pattern as a literal string (rg -F).
        #[arg(short = 'F', long = "fixed-strings")]
        fixed: bool,
        /// Only search paths matching GLOB (rg -g).
        #[arg(short = 'g', long = "glob")]
        glob: Option<String>,
        /// Max number of hits.
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Search { pattern, path, ignore_case, word, fixed, glob, limit } => {
            let opts = QueryOpts {
                ignore_case,
                smart_case: !ignore_case, // rg smart-case default
                word,
                fixed,
            };
            let re = compile(&pattern, &opts)?;
            let chunks = collect_chunks(&path, glob.as_deref())?;
            for h in search_chunks(&chunks, &re, limit) {
                println!("{}:{}:{}: {}", h.doc_path.display(), h.location, h.line, h.snippet);
            }
            Ok(())
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --test cli_it`
Expected: PASS (2 tests). Then `cargo test` — all suites green.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock src/main.rs tests/cli_it.rs
git commit -m "feat: kb search CLI with ripgrep-style flags"
```

---

## Self-Review

**Spec coverage (this milestone's slice):**
- File-First (`path + location + snippet`) → Tasks 5, 7. ✓
- ripgrep 1:1 via `regex` crate → Task 4 (`compile`) + Task 7 flags. ✓
- Structural chunking (heading sections) → Task 3. ✓
- Per-file failure isolation (skip + log, never abort) → Task 6. ✓
- Pure Rust / offline / single binary → Cargo.toml deps (Task 1) are all pure-Rust. ✓
- Deferred to later milestones (intentionally not in this plan): office/pdf/xlsx extractors (M2), tantivy index + stemming + `--rank`/`--stem` (M3), glossary + `--expand` + `read`/images (M4), MCP server (M5), persistent `kb index`/incremental (M3). ✓ (tracked in roadmap)

**Placeholder scan:** none — every code/test step contains complete code and exact run commands.

**Type consistency:** `Chunk` fields (`doc_path`/`location`/`file_type`/`text`) are used identically in Tasks 3, 5, 6. `QueryOpts` fields match between Tasks 4 and 7. `search_chunks(&[Chunk], &Regex, usize)` and `collect_chunks(&Path, Option<&str>)` signatures are consistent across producer/consumer tasks. ✓
