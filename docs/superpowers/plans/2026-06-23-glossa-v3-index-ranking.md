# glossa v3 — Milestone 3: Persistent index + multilingual ranked search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent on-disk full-text index with multilingual (RU/EN) stemming, so `kb index` builds/updates an index and `kb search --rank` returns BM25-ranked results with snippets. The default (no `--rank`) ripgrep scan from M1/M2 is unchanged.

**Architecture:** A `tantivy` index lives at `<dir>/.glossa/index`. Documents are the same `Chunk`s the extractors already produce (path/location/file_type/text). The `body` field uses a custom `multilang` tokenizer that lowercases then stems each token with the Snowball stemmer for the language `lingua` detects per text. `kb index` walks the corpus (reusing `walk::collect_chunks`), diffs a small `manifest.json` (mtime+size) to reindex only changed files, and deletes a file's stale docs by its path term before re-adding. `kb search --rank` opens the index, runs a BM25 query, and renders snippets.

**Tech Stack:** Rust; `tantivy = "0.26"` (no `stemmer` feature — we stem via `rust-stemmers` directly), `rust-stemmers = "1.2"`, `serde`/`serde_json` (manifest). Language detection defaults to a **zero-dependency script heuristic** (Cyrillic→Russian, else English — RU/EN are different scripts, so this is free and ~exact). `lingua` is an **optional** dependency behind the `lingua` cargo feature, for distinguishing same-script languages. All pure Rust, offline.

## Global Constraints

- Pure Rust, single static binary, fully offline (no network, no native/system libs). Default builds embed NO language-detection models — detection is a script heuristic. `lingua` (statistical n-gram models, CPU-only, no GPU; full set ≈ hundreds of MB embedded) is opt-in via the `lingua` feature and, when enabled, MUST stay `default-features = false` with only the needed language features.
- Index location: `<dir>/.glossa/index/` (tantivy `MmapDirectory`). Manifest: `<dir>/.glossa/manifest.json`.
- The `body` field is indexed with the `multilang` tokenizer AND stored (snippets require the field to be STORED). `path`/`location`/`file_type` are `STRING | STORED` (exact-match; `path` must be STRING so `delete_term` works).
- `--rank` = indexed BM25 search (intrinsically stemmed). The default search path (ripgrep scan over files) is unchanged; do not modify `search.rs`/`query.rs`/the default CLI path.
- Change-signal for incremental reindex is **mtime + size** (a content hash is a future hardening — note, not required here).
- tantivy 0.26 notes (verified): `Document` is a trait → use `TantivyDocument`; `searcher.doc::<TantivyDocument>(addr)?`; stored values via `doc.get_first(field).and_then(|v| v.as_str())`; `Index::create_in_dir` errors `TantivyError::IndexAlreadyExists`; register the tokenizer on the index instance before writing or searching.
- TDD: failing test first; frequent commits; DRY; YAGNI.

## Deferred (explicitly out of scope)

- Regex-path **accelerator** (trigram index to prune the default scan) — an optimization with unclear payoff at typical KB sizes; revisit only if scans get slow.
- A separate `--stem` flag — redundant: the ranked index is always stemmed. `--rank` covers it.
- Glossary query expansion (`--expand`), `read`, images, MCP — later milestones.

---

### Task 1: `multilang` tokenizer (lingua detect → rust-stemmers)

**Files:**
- Modify: `Cargo.toml` (add `tantivy`, `rust-stemmers`, `lingua`)
- Create: `src/index.rs` (module root with `pub mod multilang;`)
- Create: `src/index/multilang.rs`
- Modify: `src/lib.rs` (add `pub mod index;`)
- Test: `src/index/multilang.rs` (inline)

**Interfaces:**
- Produces:
  - `index::multilang::DetectFn = Arc<dyn Fn(&str) -> rust_stemmers::Algorithm + Send + Sync>`
  - `index::multilang::script_detector() -> DetectFn` (default; Cyrillic→Russian, else English)
  - `index::multilang::default_detector() -> DetectFn` (returns the script detector)
  - `#[cfg(feature = "lingua")] index::multilang::lingua_detector() -> DetectFn` (opt-in)
  - `index::multilang::multilang_analyzer(detect: DetectFn) -> tantivy::tokenizer::TextAnalyzer`
  - `index::multilang::MultiLangStemmer` (a `tantivy::tokenizer::TokenFilter`)

- [ ] **Step 1: Write the failing test**

Create `src/index/multilang.rs`:
```rust
use rust_stemmers::{Algorithm, Stemmer as RsStemmer};
use std::sync::Arc;
use tantivy::tokenizer::{
    LowerCaser, SimpleTokenizer, TextAnalyzer, Token, TokenFilter, TokenStream, Tokenizer,
};

/// Maps a text to the stemming algorithm to use for it.
/// Cloneable, thread-safe — required because tantivy tokenizers must be `Clone + Send + Sync + 'static`.
pub type DetectFn = Arc<dyn Fn(&str) -> Algorithm + Send + Sync>;

/// Default, zero-dependency detector. RU and EN live in different scripts, so a
/// single Cyrillic char is a reliable, free signal for Russian; everything else → English.
pub fn script_detector() -> DetectFn {
    Arc::new(|text: &str| {
        if text.chars().any(|c| ('\u{0400}'..='\u{04FF}').contains(&c)) {
            Algorithm::Russian
        } else {
            Algorithm::English
        }
    })
}

/// The detector used by default (script-based; embeds no models, pulls no extra deps).
pub fn default_detector() -> DetectFn {
    script_detector()
}

/// Optional `lingua`-backed detector, for distinguishing same-script languages.
/// Build with `--features lingua`.
#[cfg(feature = "lingua")]
pub fn lingua_detector() -> DetectFn {
    use lingua::{Language, LanguageDetectorBuilder};
    let detector =
        LanguageDetectorBuilder::from_languages(&[Language::English, Language::Russian]).build();
    Arc::new(move |text: &str| match detector.detect_language_of(text) {
        Some(Language::Russian) => Algorithm::Russian,
        _ => Algorithm::English,
    })
}

/// A tantivy TokenFilter that picks a stemming algorithm per text (via `detect`),
/// then stems every token. Tokens must already be lower-cased (chain LowerCaser first).
#[derive(Clone)]
pub struct MultiLangStemmer {
    detect: DetectFn,
}

impl MultiLangStemmer {
    pub fn new(detect: DetectFn) -> Self {
        Self { detect }
    }
}

#[derive(Clone)]
pub struct MultiLangTokenizer<T> {
    inner: T,
    detect: DetectFn,
}

pub struct MultiLangStream<T> {
    inner: T,
    stemmer: RsStemmer,
    buf: String,
}

impl TokenFilter for MultiLangStemmer {
    type Tokenizer<T: Tokenizer> = MultiLangTokenizer<T>;
    fn transform<T: Tokenizer>(self, inner: T) -> MultiLangTokenizer<T> {
        MultiLangTokenizer {
            inner,
            detect: self.detect,
        }
    }
}

impl<T: Tokenizer> Tokenizer for MultiLangTokenizer<T> {
    type TokenStream<'a> = MultiLangStream<T::TokenStream<'a>>;
    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        let algo = (self.detect)(text);
        MultiLangStream {
            inner: self.inner.token_stream(text),
            stemmer: RsStemmer::create(algo),
            buf: String::new(),
        }
    }
}

impl<T: TokenStream> TokenStream for MultiLangStream<T> {
    fn advance(&mut self) -> bool {
        if !self.inner.advance() {
            return false;
        }
        let token = self.inner.token_mut();
        let stemmed = self.stemmer.stem(&token.text);
        if stemmed != token.text.as_str() {
            self.buf.clear();
            self.buf.push_str(&stemmed);
            token.text.clear();
            token.text.push_str(&self.buf);
        }
        true
    }
    fn token(&self) -> &Token {
        self.inner.token()
    }
    fn token_mut(&mut self) -> &mut Token {
        self.inner.token_mut()
    }
}

/// Compose the full analyzer: split → lowercase → multilingual stem.
pub fn multilang_analyzer(detect: DetectFn) -> TextAnalyzer {
    TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(LowerCaser)
        .filter(MultiLangStemmer::new(detect))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(analyzer: &mut TextAnalyzer, text: &str) -> Vec<String> {
        let mut ts = analyzer.token_stream(text);
        let mut out = Vec::new();
        while ts.advance() {
            out.push(ts.token().text.clone());
        }
        out
    }

    #[test]
    fn russian_inflections_stem_to_same_root() {
        let mut a = multilang_analyzer(default_detector());
        let one = tokens(&mut a, "договор");
        let many = tokens(&mut a, "договоры договоров договорам");
        // All Russian inflections share the stem of "договор".
        let root = &one[0];
        assert!(
            many.iter().all(|t| t == root),
            "expected all forms to stem to {root:?}, got {many:?}"
        );
    }

    #[test]
    fn english_inflections_stem_to_same_root() {
        let mut a = multilang_analyzer(default_detector());
        let toks = tokens(&mut a, "running runs runner");
        assert_eq!(toks[0], "run");
        assert_eq!(toks[1], "run");
    }
}
```

- [ ] **Step 2: Run it — verify it fails**

Run: `cargo test --lib multilang`
Expected: FAIL — deps + module not present (won't compile).

- [ ] **Step 3: Add deps and declare the modules**

In `Cargo.toml` under `[dependencies]`:
```toml
tantivy = "0.26"
rust-stemmers = "1.2"
lingua = { version = "1.8", default-features = false, features = ["russian", "english"], optional = true }
```
And a features section:
```toml
[features]
default = []
lingua = ["dep:lingua"]
```
Create `src/index.rs`:
```rust
pub mod multilang;
```
In `src/lib.rs`, add to the module list:
```rust
pub mod index;
```

- [ ] **Step 4: Run it — verify it passes**

Run: `cargo test --lib multilang`
Expected: PASS (2 tests). First build fetches+compiles tantivy/lingua — slow, expected.

The default build pulls NO `lingua` — the script detector is used. Tests run on the default build.
Verification notes (confirm on first compile, adjust the single line if the crate differs):
- The GAT trait shapes (`type TokenStream<'a>`, `type Tokenizer<T: Tokenizer>`) are tantivy 0.26.
- `RsStemmer::create(Algorithm)` + `.stem(&str) -> Cow<str>` (input already lowercased by `LowerCaser`).
- The `#[cfg(feature = "lingua")] lingua_detector()` only compiles under `cargo build --features lingua`; if those `lingua` language feature names differ on this version, fix them there (it does not affect the default build or the tests).

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock src/lib.rs src/index.rs src/index/multilang.rs
git commit -m "feat: multilang tokenizer (lingua detect + rust-stemmers)"
```

---

### Task 2: Index schema + open-or-create

**Files:**
- Create: `src/index/store.rs`
- Modify: `src/index.rs` (add `pub mod store;`)
- Test: `src/index/store.rs` (inline, using `tempfile`)

**Interfaces:**
- Consumes: `index::multilang::{default_detector, multilang_analyzer}`, tantivy.
- Produces:
  - `index::store::Fields { body: Field, path: Field, location: Field, file_type: Field }`
  - `index::store::DocIndex { pub index: tantivy::Index, pub fields: Fields }`
  - `index::store::DocIndex::open_or_create(dir: &Path) -> anyhow::Result<DocIndex>`
    (index dir = `dir/.glossa/index`; registers the `multilang` tokenizer on the index)

- [ ] **Step 1: Write the failing test**

Create `src/index/store.rs`:
```rust
use crate::index::multilang::{default_detector, multilang_analyzer};
use anyhow::Context;
use std::path::Path;
use tantivy::schema::{Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, STORED, STRING};
use tantivy::{Index, TantivyError};

#[derive(Clone, Copy)]
pub struct Fields {
    pub body: Field,
    pub path: Field,
    pub location: Field,
    pub file_type: Field,
}

pub struct DocIndex {
    pub index: Index,
    pub fields: Fields,
}

fn build_schema() -> (Schema, Fields) {
    let mut sb = Schema::builder();
    let body_opts = TextOptions::default().set_stored().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer("multilang")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions),
    );
    let body = sb.add_text_field("body", body_opts);
    let path = sb.add_text_field("path", STRING | STORED);
    let location = sb.add_text_field("location", STRING | STORED);
    let file_type = sb.add_text_field("file_type", STRING | STORED);
    (sb.build(), Fields { body, path, location, file_type })
}

fn index_dir(dir: &Path) -> std::path::PathBuf {
    dir.join(".glossa").join("index")
}

impl DocIndex {
    pub fn open_or_create(dir: &Path) -> anyhow::Result<DocIndex> {
        let (schema, fields) = build_schema();
        let idx_path = index_dir(dir);
        std::fs::create_dir_all(&idx_path).with_context(|| format!("create {idx_path:?}"))?;
        let index = match Index::create_in_dir(&idx_path, schema.clone()) {
            Ok(i) => i,
            Err(TantivyError::IndexAlreadyExists) => Index::open_in_dir(&idx_path)?,
            Err(e) => return Err(e.into()),
        };
        index
            .tokenizers()
            .register("multilang", multilang_analyzer(default_detector()));
        Ok(DocIndex { index, fields })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_then_reopens_index() {
        let dir = tempfile::tempdir().unwrap();
        let a = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(a.index.schema().get_field("body").is_ok());
        drop(a);
        // Second call must open the existing index, not error.
        let b = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(b.index.schema().get_field("path").is_ok());
    }
}
```

- [ ] **Step 2: Run it — verify it fails**

Run: `cargo test --lib store`
Expected: FAIL — `index::store` module not declared.

- [ ] **Step 3: Declare the module**

In `src/index.rs`:
```rust
pub mod store;
```

- [ ] **Step 4: Run it — verify it passes**

Run: `cargo test --lib store`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add src/index.rs src/index/store.rs
git commit -m "feat: tantivy schema + open-or-create index with multilang tokenizer"
```

---

### Task 3: Write chunks + BM25 ranked search

**Files:**
- Modify: `src/index/store.rs` (add `write_chunks`, `search`, `RankedHit`)
- Test: `src/index/store.rs` (inline)

**Interfaces:**
- Consumes: `model::Chunk`, the `DocIndex`/`Fields` from Task 2, tantivy query/collector/snippet APIs.
- Produces:
  - `index::store::RankedHit { path: String, location: String, file_type: String, snippet: String, score: f32 }`
  - `DocIndex::write_chunks(&self, chunks: &[Chunk]) -> anyhow::Result<()>` (adds + commits)
  - `DocIndex::search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<RankedHit>>` (BM25, snippet from stored body)

- [ ] **Step 1: Write the failing test**

Append to `src/index/store.rs` (add imports at top, items below the impl):
```rust
use crate::model::Chunk;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::snippet::SnippetGenerator;
use tantivy::{TantivyDocument, doc};

#[derive(Debug, Clone, PartialEq)]
pub struct RankedHit {
    pub path: String,
    pub location: String,
    pub file_type: String,
    pub snippet: String,
    pub score: f32,
}

impl DocIndex {
    pub fn write_chunks(&self, chunks: &[Chunk]) -> anyhow::Result<()> {
        let mut writer = self.index.writer(50_000_000)?;
        for c in chunks {
            writer.add_document(doc!(
                self.fields.body => c.text.clone(),
                self.fields.path => c.doc_path.to_string_lossy().to_string(),
                self.fields.location => c.location.clone(),
                self.fields.file_type => c.file_type.clone(),
            ))?;
        }
        writer.commit()?;
        Ok(())
    }

    pub fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<RankedHit>> {
        let reader = self.index.reader()?;
        let searcher = reader.searcher();
        let parser = QueryParser::for_index(&self.index, vec![self.fields.body]);
        let parsed = parser.parse_query(query)?;
        let top = searcher.search(&parsed, &TopDocs::with_limit(limit))?;

        let snippet_gen = SnippetGenerator::create(&searcher, &*parsed, self.fields.body)?;

        let mut hits = Vec::with_capacity(top.len());
        for (score, addr) in top {
            let d: TantivyDocument = searcher.doc(addr)?;
            let get = |f| d.get_first(f).and_then(|v| v.as_str()).unwrap_or("").to_string();
            let snippet = snippet_gen.snippet_from_doc(&d).fragment().to_string();
            hits.push(RankedHit {
                path: get(self.fields.path),
                location: get(self.fields.location),
                file_type: get(self.fields.file_type),
                snippet,
                score,
            });
        }
        Ok(hits)
    }
}

#[cfg(test)]
mod search_tests {
    use super::*;
    use std::path::PathBuf;

    fn chunk(path: &str, text: &str) -> Chunk {
        Chunk {
            doc_path: PathBuf::from(path),
            location: "S".into(),
            file_type: "md".into(),
            text: text.into(),
        }
    }

    #[test]
    fn ranked_search_finds_russian_by_inflected_query() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            chunk("a.md", "Подписаны договоры на поставку"),
            chunk("b.md", "unrelated english content"),
        ])
        .unwrap();

        // Query uses a different inflection ("договор") than the doc ("договоры").
        let hits = idx.search("договор", 10).unwrap();
        assert!(!hits.is_empty(), "stemmed query should match inflected doc");
        assert_eq!(hits[0].path, "a.md");
        assert!(hits[0].score > 0.0);
        assert!(!hits[0].snippet.is_empty());
    }
}
```

- [ ] **Step 2: Run it — verify it fails**

Run: `cargo test --lib search_tests`
Expected: FAIL — `write_chunks`/`search`/`RankedHit` not yet defined (before you paste Step 1) — once pasted it should compile; if you wrote the test first per TDD, the RED is the missing impl.

- [ ] **Step 3: (implementation is in Step 1's appended items)**

Confirm the impl block and `RankedHit` are present as shown.

- [ ] **Step 4: Run it — verify it passes**

Run: `cargo test --lib search_tests`
Expected: PASS (1 test). Then `cargo test --lib` (all lib tests green).

Verification notes (confirm/adjust on first compile):
- `searcher.doc(addr)` may need a turbofish: `searcher.doc::<TantivyDocument>(addr)?` — both forms documented; use whichever compiles.
- `v.as_str()` is the accessor on the stored value returned by `get_first`.
- `snippet.fragment()` returns the plain-text window (use `.to_html()` instead if you want `<b>` highlights).

- [ ] **Step 5: Commit**

```bash
git add src/index/store.rs
git commit -m "feat: write chunks to index + BM25 ranked search with snippets"
```

---

### Task 4: Incremental indexing with a manifest

**Files:**
- Modify: `Cargo.toml` (add `serde`, `serde_json`)
- Create: `src/index/manifest.rs`
- Modify: `src/index.rs` (add `pub mod manifest;`)
- Modify: `src/index/store.rs` (add `delete_path` + `index_dir`/`reindex_dir` entry points)
- Test: `src/index/manifest.rs` (inline) + `src/index/store.rs` (inline)

**Interfaces:**
- Consumes: `walk::collect_chunks`, `DocIndex`, std fs metadata.
- Produces:
  - `index::manifest::Manifest` (`BTreeMap<String, FileSig>`, `FileSig { mtime_secs: u64, size: u64 }`) with `load(dir)`, `save(dir)`, and `pub fn changed(&self, path: &str, sig: FileSig) -> bool`.
  - `index::store::DocIndex::delete_path(&self, path: &str) -> anyhow::Result<()>`
  - `index::store::index_dir(dir: &Path, force: bool) -> anyhow::Result<IndexStats>` where
    `IndexStats { added: usize, removed: usize, unchanged: usize }`. `force = true` rebuilds from scratch.

- [ ] **Step 1: Write the failing test (manifest)**

Create `src/index/manifest.rs`:
```rust
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileSig {
    pub mtime_secs: u64,
    pub size: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Manifest {
    pub files: BTreeMap<String, FileSig>,
}

fn manifest_path(dir: &Path) -> std::path::PathBuf {
    dir.join(".glossa").join("manifest.json")
}

impl Manifest {
    pub fn load(dir: &Path) -> Manifest {
        let p = manifest_path(dir);
        match std::fs::read_to_string(&p) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
            Err(_) => Manifest::default(),
        }
    }

    pub fn save(&self, dir: &Path) -> anyhow::Result<()> {
        let p = manifest_path(dir);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let s = serde_json::to_string_pretty(self).context("serialize manifest")?;
        std::fs::write(&p, s).with_context(|| format!("write {p:?}"))?;
        Ok(())
    }

    /// True if the path is new or its signature differs from the recorded one.
    pub fn changed(&self, path: &str, sig: FileSig) -> bool {
        self.files.get(path) != Some(&sig)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_and_detects_change() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = Manifest::default();
        m.files.insert("a.md".into(), FileSig { mtime_secs: 10, size: 20 });
        m.save(dir.path()).unwrap();

        let loaded = Manifest::load(dir.path());
        assert_eq!(loaded.files.get("a.md"), Some(&FileSig { mtime_secs: 10, size: 20 }));
        assert!(!loaded.changed("a.md", FileSig { mtime_secs: 10, size: 20 }));
        assert!(loaded.changed("a.md", FileSig { mtime_secs: 11, size: 20 }));
        assert!(loaded.changed("new.md", FileSig { mtime_secs: 1, size: 1 }));
    }
}
```

- [ ] **Step 2: Run it — verify it fails, then add deps + module**

Run: `cargo test --lib manifest`
Expected: FAIL — deps/module missing.

In `Cargo.toml`:
```toml
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```
In `src/index.rs`:
```rust
pub mod manifest;
```
Re-run `cargo test --lib manifest` → PASS.

- [ ] **Step 3: Write the failing test (incremental index_dir)**

Append to `src/index/store.rs`:
```rust
use crate::index::manifest::{FileSig, Manifest};
use crate::walk::collect_chunks;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct IndexStats {
    pub added: usize,
    pub removed: usize,
    pub unchanged: usize,
}

fn file_sig(path: &Path) -> anyhow::Result<FileSig> {
    let md = std::fs::metadata(path)?;
    let mtime_secs = md
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Ok(FileSig { mtime_secs, size: md.len() })
}

impl DocIndex {
    pub fn delete_path(&self, path: &str) -> anyhow::Result<()> {
        let mut writer = self.index.writer(50_000_000)?;
        writer.delete_term(tantivy::Term::from_field_text(self.fields.path, path));
        writer.commit()?;
        Ok(())
    }
}

/// Walk `dir`, (re)index changed files, drop removed files, update the manifest.
/// `force = true` ignores the manifest and rebuilds every file.
pub fn index_dir(dir: &Path, force: bool) -> anyhow::Result<IndexStats> {
    let idx = DocIndex::open_or_create(dir)?;
    let mut manifest = if force { Manifest::default() } else { Manifest::load(dir) };
    let chunks = collect_chunks(dir, None)?;

    // Group chunks by file path and capture the current signature per file.
    use std::collections::BTreeMap;
    let mut by_path: BTreeMap<String, Vec<crate::model::Chunk>> = BTreeMap::new();
    for c in chunks {
        by_path
            .entry(c.doc_path.to_string_lossy().to_string())
            .or_default()
            .push(c);
    }

    let mut stats = IndexStats::default();
    let mut next = Manifest::default();
    for (path, file_chunks) in &by_path {
        let sig = match file_sig(Path::new(path)) {
            Ok(s) => s,
            Err(_) => continue,
        };
        next.files.insert(path.clone(), sig);
        if !force && !manifest.changed(path, sig) {
            stats.unchanged += 1;
            continue;
        }
        idx.delete_path(path)?; // drop any stale docs for this file first
        idx.write_chunks(file_chunks)?;
        stats.added += 1;
    }

    // Files in the old manifest but no longer present → delete their docs.
    for old_path in manifest.files.keys() {
        if !next.files.contains_key(old_path) {
            idx.delete_path(old_path)?;
            stats.removed += 1;
        }
    }

    next.save(dir)?;
    Ok(stats)
}

#[cfg(test)]
mod incremental_tests {
    use super::*;
    use std::fs;

    #[test]
    fn reindex_picks_up_changes_and_skips_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.md"), b"# T\nдоговоры поставка\n").unwrap();

        let s1 = index_dir(dir.path(), false).unwrap();
        assert_eq!(s1.added, 1);

        // Re-run with no changes → unchanged, nothing re-added.
        let s2 = index_dir(dir.path(), false).unwrap();
        assert_eq!(s2.unchanged, 1);
        assert_eq!(s2.added, 0);

        // Search still works against the built index.
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let hits = idx.search("договор", 10).unwrap();
        assert!(hits.iter().any(|h| h.path.ends_with("a.md")));
    }
}
```

- [ ] **Step 4: Run it — verify it passes**

Run: `cargo test --lib incremental_tests` then `cargo test --lib`
Expected: PASS. (Note: `index_dir` opens a fresh writer per changed file via `delete_path`/`write_chunks`; acceptable for v1 — a single-writer batch is a later optimization.)

Verification note: confirm `tantivy::Term` import path (the 0.26 changelog mentions a Term/IndexingTerm split; `Term::from_field_text` is the documented constructor — adjust the import if the compiler points elsewhere).

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock src/index.rs src/index/manifest.rs src/index/store.rs
git commit -m "feat: incremental index_dir with mtime+size manifest and delete-by-path"
```

---

### Task 5: CLI — `kb index`, `kb reindex`, `kb search --rank`

**Files:**
- Modify: `src/main.rs`
- Test: `tests/rank_it.rs` (integration, invokes the built binary)

**Interfaces:**
- Consumes: `index::store::{index_dir, DocIndex}`; existing default search path (unchanged).
- Produces: CLI subcommands `index`/`reindex`, and a `--rank` flag on `search` that routes to `DocIndex::search`. Ranked output line format: `path:location: snippet  [score]`.

- [ ] **Step 1: Write the failing test**

Create `tests/rank_it.rs`:
```rust
use assert_cmd::Command;
use predicates::str::contains;
use std::fs;

#[test]
fn index_then_ranked_search_finds_russian_inflection() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), "# T\nПодписаны договоры на поставку\n").unwrap();

    // Build the index.
    Command::cargo_bin("kb")
        .unwrap()
        .args(["index", dir.path().to_str().unwrap()])
        .assert()
        .success();

    // Ranked search with a different inflection than the document.
    Command::cargo_bin("kb")
        .unwrap()
        .args(["search", "--rank", "договор", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(contains("a.md"));
}
```

- [ ] **Step 2: Run it — verify it fails**

Run: `cargo test --test rank_it`
Expected: FAIL — no `index` subcommand and `search` has no `--rank`.

- [ ] **Step 3: Extend the CLI**

In `src/main.rs`, add to the `Cmd` enum and `main` match. Add the `rank` flag to `Search`, and two new subcommands:
```rust
    /// Build or update the on-disk index for ranked search.
    Index {
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Rebuild the index from scratch.
    Reindex {
        #[arg(default_value = ".")]
        path: PathBuf,
    },
```
Add to the `Search` variant's fields:
```rust
        /// Use the on-disk index for BM25-ranked, stemmed search (run `kb index` first).
        #[arg(long)]
        rank: bool,
```
In `main`, handle the new arms and the `rank` branch (place the `rank` check at the start of the existing `Search` arm):
```rust
        Cmd::Index { path } => {
            let stats = glossa::index::store::index_dir(&path, false)?;
            println!("indexed: {} added, {} removed, {} unchanged", stats.added, stats.removed, stats.unchanged);
            Ok(())
        }
        Cmd::Reindex { path } => {
            let stats = glossa::index::store::index_dir(&path, true)?;
            println!("reindexed: {} files", stats.added);
            Ok(())
        }
        Cmd::Search { pattern, path, ignore_case, word, fixed, glob, limit, rank } => {
            if rank {
                let idx = glossa::index::store::DocIndex::open_or_create(&path)?;
                for h in idx.search(&pattern, limit)? {
                    println!("{}:{}: {}  [{:.3}]", h.path, h.location, h.snippet, h.score);
                }
                return Ok(());
            }
            // ... existing default (ripgrep scan) path unchanged ...
        }
```
(Keep the entire existing default `Search` body after the `if rank { ... return }` guard. Only add the `rank` field and the guard — do not change the default scan behavior.)

- [ ] **Step 4: Run it — verify it passes**

Run: `cargo test --test rank_it`
Expected: PASS. Then `cargo test` — all suites green.

- [ ] **Step 5: Manual smoke**

Run:
```bash
cargo run --bin kb -- index tests/fixtures
cargo run --bin kb -- search --rank "glossa" tests/fixtures
```
Expected: index reports added files; ranked search prints the docx/pdf/md hits with scores.

- [ ] **Step 6: Commit**

```bash
git add src/main.rs tests/rank_it.rs
git commit -m "feat: kb index/reindex + kb search --rank (BM25 stemmed)"
```

---

## Self-Review

**Spec coverage (Milestone 3 slice):**
- Persistent tantivy index at `<dir>/.glossa/index` → Tasks 2-4. ✓
- Multilingual stemming (RU/EN, lingua-detected) → Task 1 (multilang tokenizer). ✓
- BM25 ranked search with snippets → Task 3. ✓
- `kb index`/`reindex` incremental (mtime+size manifest, delete-by-path) → Tasks 4-5. ✓
- `kb search --rank` routes to the index; default scan unchanged → Task 5. ✓
- Deferred (stated): regex accelerator, `--stem` (redundant), `--expand`, read/images/MCP. ✓

**Placeholder scan:** none — every step has complete code + exact commands. The "verification notes" flag the few API points the recon could not 100% pin (turbofish on `searcher.doc`, `Term` import path, lingua feature names); each is a one-line confirm-at-compile, not a missing implementation.

**Type consistency:** `Fields`/`DocIndex` defined in Task 2 and used in Tasks 3-4; `Chunk` fields match M1; `RankedHit`/`IndexStats`/`FileSig`/`Manifest` defined once and consumed consistently; `index_dir(&Path, bool) -> Result<IndexStats>` signature matches its CLI callers in Task 5; `multilang_analyzer`/`default_detector` signatures match between Task 1 and Task 2.

**Dependency note:** new always-on deps are `tantivy 0.26`, `rust-stemmers 1.2`, `serde`/`serde_json` — all pure Rust, offline. Language detection defaults to the in-house script heuristic (no model data, no extra dep). `lingua 1.8` is OPTIONAL behind the `lingua` feature (off by default) — when enabled it must stay `default-features = false` with only the needed language features, since its full model set is large. `tantivy` is used WITHOUT its `stemmer` feature (we stem via `rust-stemmers` directly through the custom filter). No GPU, no native/system libs anywhere.
