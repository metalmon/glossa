use crate::index::manifest::{FileSig, Manifest};
use crate::index::multilang::{default_detector, multilang_analyzer};
use crate::model::Chunk;
use anyhow::Context;
use ignore::WalkBuilder;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::Value;
use tantivy::schema::{
    Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, INDEXED, STORED, STRING,
};
use tantivy::snippet::SnippetGenerator;
use tantivy::tokenizer::{LowerCaser, NgramTokenizer, TextAnalyzer};
use tantivy::{doc, Index, IndexReader, TantivyDocument, TantivyError};

/// Bump when the tantivy schema changes (triggers index-only rebuild via manifest migration).
pub const INDEX_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Copy)]
pub struct Fields {
    pub body: Field,
    pub body_trigrams: Field,
    pub path: Field,
    pub location: Field,
    pub file_type: Field,
    pub ord: Field,
}

pub struct DocIndex {
    pub index: Index,
    pub fields: Fields,
    /// Corpus read anchors (abs, canonicalized paths). A stored doc key is `<label>/<relpath>`
    /// (empty label ⇒ bare relpath, back-compat) resolved against the matching root's `path` —
    /// see `doc_file`/`doc_key`. Multiple roots let one index span several corpus directories.
    pub roots: Vec<crate::root::Root>,
    /// Base directory for on-disk state (`.glossa/index`, manifest, etc); abs, canonicalized.
    /// Defaults to the sole corpus root for the back-compat co-located layout.
    pub state_base: PathBuf,
    /// Long-lived reader, reused across every search/read_chunk. Building a reader reopens the
    /// segments, so doing it once per call (rather than per query) is what made repeated tool
    /// calls on a shared index pay an open cost each time. Refreshed after writes via reload().
    reader: IndexReader,
}

fn build_schema() -> Schema {
    let mut sb = Schema::builder();
    let body_opts = TextOptions::default().set_stored().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer("multilang")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions),
    );
    sb.add_text_field("body", body_opts);
    let trigram_opts = TextOptions::default().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer("trigram3")
            .set_index_option(IndexRecordOption::Basic),
    );
    sb.add_text_field("body_trigrams", trigram_opts);
    sb.add_text_field("path", STRING | STORED);
    sb.add_text_field("location", STRING | STORED);
    sb.add_text_field("file_type", STRING | STORED);
    sb.add_u64_field("ord", INDEXED | STORED);
    sb.build()
}

fn fields_from_schema(schema: &Schema) -> anyhow::Result<Fields> {
    Ok(Fields {
        body: schema.get_field("body")?,
        body_trigrams: schema.get_field("body_trigrams")?,
        path: schema.get_field("path")?,
        location: schema.get_field("location")?,
        file_type: schema.get_field("file_type")?,
        ord: schema.get_field("ord")?,
    })
}

fn register_tokenizers(index: &Index) {
    index
        .tokenizers()
        .register("multilang", multilang_analyzer(default_detector()));
    let trigram =
        TextAnalyzer::builder(NgramTokenizer::new(3, 3, false).expect("trigram tokenizer"))
            .filter(LowerCaser)
            .build();
    index.tokenizers().register("trigram3", trigram);
}

fn index_dir_path(dir: &Path) -> PathBuf {
    dir.join(".glossa").join("index")
}

/// The absolute corpus root — the single anchor for relative doc keys. Canonicalized so that `.`,
/// `kb-test` and `E:\…\kb-test` all resolve to the same root (the `\\?\` verbatim prefix Windows
/// adds is stripped so `root.join(rel)` stays a clean path).
pub fn abs_root(dir: &Path) -> PathBuf {
    match std::fs::canonicalize(dir) {
        Ok(p) => {
            let s = p.to_string_lossy();
            PathBuf::from(
                s.strip_prefix(r"\\?\")
                    .map(str::to_string)
                    .unwrap_or_else(|| s.into_owned()),
            )
        }
        Err(_) => dir.to_path_buf(),
    }
}

/// Like `abs_root`, but for an arbitrary path rather than the corpus root, and `None` (not a
/// same-path fallback) when canonicalization fails — the caller needs to know the path didn't
/// resolve, not receive an un-canonicalized stand-in it could accidentally match against.
fn canonicalize_stripped(path: &Path) -> Option<PathBuf> {
    let p = std::fs::canonicalize(path).ok()?;
    let s = p.to_string_lossy();
    Some(PathBuf::from(
        s.strip_prefix(r"\\?\")
            .map(str::to_string)
            .unwrap_or_else(|| s.into_owned()),
    ))
}

/// A document's canonical key: its path RELATIVE to the corpus root. This is the ONE form stored in
/// the index and the graph; `DocIndex::doc_file` turns it back into a real file path. Built once,
/// at the walk boundary, so the stored key never depends on how the corpus was addressed.
pub fn rel_key(root: &Path, abs: &Path) -> String {
    // Canonical key uses forward slashes. `/` is JSON/transport-safe — a lone `\` is an escape
    // character and gets dropped/mangled through tool args and MCP — while Windows accepts `/` in
    // paths natively. This is the ONE place the stored separator is decided, so the index and the
    // graph stay uniform (no mixed `docs/sub\file` keys) and every displayed path round-trips cleanly.
    abs.strip_prefix(root)
        .unwrap_or(abs)
        .to_string_lossy()
        .replace('\\', "/")
}

/// The stored key for a walked file: `<label>/<relpath>`, or bare `<relpath>` when `label` is
/// empty (the back-compat single-root form). `root_abs` is the corpus root `abs` was found under.
pub fn doc_key(label: &str, root_abs: &Path, abs: &Path) -> String {
    let rel = rel_key(root_abs, abs);
    if label.is_empty() {
        rel
    } else {
        format!("{label}/{rel}")
    }
}

/// The inverse of `doc_key`: turn a stored doc key back into a real file path, given the set of
/// configured `roots`. The single place the label->root lookup is implemented — `DocIndex::doc_file`
/// delegates here, and any caller that needs to resolve a key without an open `DocIndex` (staleness
/// re-stat, a single-file reindex) can call it directly.
///
/// Back-compat: exactly one empty-label root ⇒ bare relpath (byte-identical to the old
/// `root.join`). Otherwise the key's `label/` prefix picks the matching root; an unresolvable label
/// (a root removed between runs) falls back to the primary root, never a corpus write path.
pub fn doc_file_in(roots: &[crate::root::Root], key: &str) -> PathBuf {
    if roots.len() == 1 && roots[0].label.is_empty() {
        return roots[0].path.join(key);
    }
    if let Some((label, rel)) = key.split_once('/') {
        if let Some(r) = roots.iter().find(|r| r.label == label) {
            return r.path.join(rel);
        }
    }
    roots
        .first()
        .map(|r| r.path.join(key))
        .unwrap_or_else(|| PathBuf::from(key))
}

impl DocIndex {
    /// Primitive: open the index under `state_base/.glossa/index`, resolving doc keys against
    /// `roots`. `roots` are canonicalized here so every stored/looked-up key is anchored to the
    /// same absolute form regardless of how the caller addressed them.
    pub fn open_or_create_at(
        roots: &[crate::root::Root],
        state_base: &Path,
    ) -> anyhow::Result<DocIndex> {
        let schema = build_schema();
        let idx_path = index_dir_path(state_base);
        std::fs::create_dir_all(&idx_path).with_context(|| format!("create {idx_path:?}"))?;
        let index = match Index::create_in_dir(&idx_path, schema.clone()) {
            Ok(i) => i,
            // Open reads `meta.json` (`atomic_read` + serde parse); on Windows that read can hit a
            // transient lock / partial read while another pass is mid temp+rename. Drive it through
            // the bounded transient-FS retry instead of failing the whole open.
            Err(TantivyError::IndexAlreadyExists) => {
                with_writer_retry(|| Index::open_in_dir(&idx_path))?
            }
            Err(e) => return Err(e.into()),
        };
        register_tokenizers(&index);
        let fields = fields_from_schema(&index.schema())?;
        let reader = index.reader()?;
        let roots = roots
            .iter()
            .map(|r| crate::root::Root {
                label: r.label.clone(),
                path: abs_root(&r.path),
            })
            .collect();
        Ok(DocIndex {
            index,
            fields,
            roots,
            state_base: abs_root(state_base),
            reader,
        })
    }

    /// Back-compat wrapper: single empty-label root == state_base == `dir` (co-located, unchanged).
    pub fn open_or_create(dir: &Path) -> anyhow::Result<DocIndex> {
        Self::open_or_create_at(
            &[crate::root::Root {
                label: String::new(),
                path: dir.to_path_buf(),
            }],
            dir,
        )
    }

    /// The primary corpus root — the first configured root, or `state_base` if somehow there are
    /// none (defensive; every real `DocIndex` has at least one root). Used by callers that still
    /// walk/anchor a single tree; multi-root-aware iteration is Task 5.
    pub fn primary_root(&self) -> &Path {
        self.roots
            .first()
            .map(|r| r.path.as_path())
            .unwrap_or(&self.state_base)
    }

    /// Turn a stored doc key (`<label>/<relpath>`, or bare `<relpath>` for the back-compat
    /// single empty-label root) back into the real file path. The single place the system maps a
    /// key to a file — no path joining is scattered elsewhere.
    pub fn doc_file(&self, key: &str) -> PathBuf {
        doc_file_in(&self.roots, key)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RankedHit {
    pub path: String,
    pub location: String,
    pub file_type: String,
    pub ord: u64,
    pub snippet: String,
    pub score: f32,
}

impl RankedHit {
    /// One search-result line leading with the canonical copy-ready chunk reference `path#ord`,
    /// followed by a non-numeric label (the heading text, or the file type for paged formats whose
    /// location is itself a number) so nothing competes with the read key.
    pub fn display_line(&self) -> String {
        let label = if self.location.starts_with("p.") {
            self.file_type.as_str()
        } else {
            self.location.as_str()
        };
        format!("{}#{} · {} · {}", self.path, self.ord, label, self.snippet)
    }
}

impl DocIndex {
    pub fn write_chunks(&self, chunks: &[Chunk]) -> anyhow::Result<()> {
        // Delete existing docs for every distinct path so re-calling is idempotent.
        let mut distinct_paths: Vec<String> = chunks
            .iter()
            .map(|c| c.doc_path.to_string_lossy().to_string())
            .collect();
        distinct_paths.sort();
        distinct_paths.dedup();
        // The whole write (open writer → delete → add → commit) runs under a transient-IO retry:
        // on Windows a just-created index file is briefly locked (Defender scan, lingering reader)
        // and opening the writer can fail with "Access is denied (os error 5)". The transaction is
        // idempotent (delete-by-path then re-add), so retrying the whole thing is safe.
        with_writer_retry(|| {
            let mut writer = self.index.writer(50_000_000)?;
            for path_str in &distinct_paths {
                writer.delete_term(tantivy::Term::from_field_text(self.fields.path, path_str));
            }
            for (i, c) in chunks.iter().enumerate() {
                let ord = chunk_ord(&c.file_type, &c.location, (i + 1) as u64);
                // A heading-less chunk has no location, which would make its section id a
                // bare "<path>#" — meaningless and, to the agent, indistinguishable from a
                // broken empty path. Fall back to the chunk's ordinal so every section has a
                // real id ("<path>#<ord>") that matches how the agent references it (#n) and
                // resolves the same way in the section node, resolve_section_ref, and read.
                let location = if c.location.is_empty() {
                    ord.to_string()
                } else {
                    c.location.clone()
                };
                writer.add_document(doc!(
                    self.fields.body => c.text.clone(),
                    self.fields.body_trigrams => c.text.clone(),
                    self.fields.path => c.doc_path.to_string_lossy().to_string(),
                    self.fields.location => location,
                    self.fields.file_type => c.file_type.clone(),
                    self.fields.ord => ord,
                ))?;
            }
            writer.commit()?;
            Ok(())
        })?;
        self.reader.reload()?;
        Ok(())
    }

    pub fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<RankedHit>> {
        let searcher = self.reader.searcher();
        let parser = QueryParser::for_index(&self.index, vec![self.fields.body]);
        let parsed = parser.parse_query(query)?;
        let top = searcher.search(&parsed, &TopDocs::with_limit(limit).order_by_score())?;

        let snippet_gen = SnippetGenerator::create(&searcher, &*parsed, self.fields.body)?;

        let mut hits = Vec::with_capacity(top.len());
        for (score, addr) in top {
            let d: TantivyDocument = searcher.doc(addr)?;
            let get = |f: tantivy::schema::Field| -> String {
                d.get_first(f)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            };
            let snippet = snippet_gen.snippet_from_doc(&d).fragment().to_string();
            let ord = d
                .get_first(self.fields.ord)
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            hits.push(RankedHit {
                path: get(self.fields.path),
                location: get(self.fields.location),
                file_type: get(self.fields.file_type),
                ord,
                snippet,
                score,
            });
        }
        Ok(hits)
    }

    /// BM25 search scoped by an optional path glob and/or exact file_type. The filters are applied
    /// AFTER ranking, so a generous candidate pool is fetched when filtering to still fill `limit`.
    /// Reuses `search` (unfiltered) so ranking semantics stay identical.
    ///
    /// `scope` (when `Some`) is a SEPARATE, ANDed filter — the friendly "restrict to one document"
    /// counterpart to `glob` (the raw ripgrep `-g` glob `search` has always taken): a bare document
    /// path compiles to `**/<p>` via [`crate::grep::path_to_glob`] (also stripping a trailing `#n`);
    /// a value already carrying glob metacharacters passes through unchanged. When both `glob` and
    /// `scope` are set, a hit must match BOTH. `None` scope = no extra filtering (identical to
    /// omitting it).
    pub fn search_filtered(
        &self,
        query: &str,
        limit: usize,
        glob: Option<&str>,
        file_type: Option<&str>,
        scope: Option<&str>,
    ) -> anyhow::Result<Vec<RankedHit>> {
        if glob.is_none() && file_type.is_none() && scope.is_none() {
            return self.search(query, limit);
        }
        let glob_m = match glob {
            Some(g) => Some(crate::glob::compile_glob(g)?),
            None => None,
        };
        let scope_m = match scope {
            Some(s) => Some(crate::glob::compile_glob(&crate::grep::path_to_glob(s))?),
            None => None,
        };
        let pool = limit.saturating_mul(20).min(2000).max(limit);
        let hits = self.search(query, pool)?;
        let filtered: Vec<RankedHit> = hits
            .into_iter()
            .filter(|h| file_type.is_none_or(|ft| h.file_type == ft))
            .filter(|h| {
                glob_m
                    .as_ref()
                    .is_none_or(|m| crate::glob::path_matches(m, &h.path))
            })
            .filter(|h| {
                scope_m
                    .as_ref()
                    .is_none_or(|m| crate::glob::path_matches(m, &h.path))
            })
            .take(limit)
            .collect();
        Ok(filtered)
    }

    /// Fetch a single chunk's stored body by exact path + location (an index lookup, no source
    /// re-parse). Returns `None` when no chunk matches, so callers can fall back to reading the
    /// file. This keeps `read` cheap on large bases where a single PDF may be hundreds of pages.
    pub fn read_chunk(&self, path: &str, location: &str) -> anyhow::Result<Option<String>> {
        use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
        let searcher = self.reader.searcher();
        let clauses: Vec<(Occur, Box<dyn Query>)> = vec![
            (
                Occur::Must,
                Box::new(TermQuery::new(
                    tantivy::Term::from_field_text(self.fields.path, path),
                    IndexRecordOption::Basic,
                )),
            ),
            (
                Occur::Must,
                Box::new(TermQuery::new(
                    tantivy::Term::from_field_text(self.fields.location, location),
                    IndexRecordOption::Basic,
                )),
            ),
        ];
        let top = searcher.search(
            &BooleanQuery::new(clauses),
            &TopDocs::with_limit(1).order_by_score(),
        )?;
        match top.first() {
            Some((_score, addr)) => {
                let d: TantivyDocument = searcher.doc(*addr)?;
                let body = d
                    .get_first(self.fields.body)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                Ok(Some(body))
            }
            None => Ok(None),
        }
    }
}

/// A chunk read by its canonical number, with the numbers of its in-document neighbors.
pub struct ChunkRead {
    pub body: String,
    pub prev: Option<u64>,
    pub next: Option<u64>,
}

impl DocIndex {
    /// Fetch a chunk's stored body by exact (path, ord). Reports the nearest preceding and
    /// following ords that exist for the same document, so the caller can offer "next/previous
    /// chunk" navigation even when pages are non-contiguous (e.g. blank PDF pages produce no
    /// chunk). Scans up to 50 steps in each direction. None if no chunk with that (path, ord)
    /// is indexed.
    pub fn read_chunk_by_ord(&self, path: &str, n: u64) -> anyhow::Result<Option<ChunkRead>> {
        let body = match self.ord_body(path, n)? {
            Some(b) => b,
            None => return Ok(None),
        };
        // Scan backward up to 50 steps to find the nearest existing predecessor.
        let prev = {
            let mut found = None;
            let lo = n.saturating_sub(50);
            for k in (lo..n).rev() {
                if k == 0 {
                    break;
                }
                if self.ord_body(path, k)?.is_some() {
                    found = Some(k);
                    break;
                }
            }
            found
        };
        // Scan forward up to 50 steps to find the nearest existing successor.
        let next = {
            let mut found = None;
            for k in (n + 1)..=(n + 50) {
                if self.ord_body(path, k)?.is_some() {
                    found = Some(k);
                    break;
                }
            }
            found
        };
        Ok(Some(ChunkRead { body, prev, next }))
    }

    fn ord_body(&self, path: &str, n: u64) -> anyhow::Result<Option<String>> {
        use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
        let searcher = self.reader.searcher();
        let clauses: Vec<(Occur, Box<dyn Query>)> = vec![
            (
                Occur::Must,
                Box::new(TermQuery::new(
                    tantivy::Term::from_field_text(self.fields.path, path),
                    IndexRecordOption::Basic,
                )),
            ),
            (
                Occur::Must,
                Box::new(TermQuery::new(
                    tantivy::Term::from_field_u64(self.fields.ord, n),
                    IndexRecordOption::Basic,
                )),
            ),
        ];
        let top = searcher.search(
            &BooleanQuery::new(clauses),
            &TopDocs::with_limit(1).order_by_score(),
        )?;
        match top.first() {
            Some((_score, addr)) => {
                let d: TantivyDocument = searcher.doc(*addr)?;
                Ok(Some(
                    d.get_first(self.fields.body)
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                ))
            }
            None => Ok(None),
        }
    }

    /// Resolve chunk number `n` to the `location` string stored in the index for `path`.
    /// Mirrors `ord_body` but returns the location field instead of the body.
    /// Returns `None` when no chunk with that (path, ord) pair is indexed.
    pub fn location_for_ord(&self, path: &str, n: u64) -> anyhow::Result<Option<String>> {
        use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
        let searcher = self.reader.searcher();
        let clauses: Vec<(Occur, Box<dyn Query>)> = vec![
            (
                Occur::Must,
                Box::new(TermQuery::new(
                    tantivy::Term::from_field_text(self.fields.path, path),
                    IndexRecordOption::Basic,
                )),
            ),
            (
                Occur::Must,
                Box::new(TermQuery::new(
                    tantivy::Term::from_field_u64(self.fields.ord, n),
                    IndexRecordOption::Basic,
                )),
            ),
        ];
        let top = searcher.search(
            &BooleanQuery::new(clauses),
            &TopDocs::with_limit(1).order_by_score(),
        )?;
        match top.first() {
            Some((_score, addr)) => {
                let d: TantivyDocument = searcher.doc(*addr)?;
                Ok(Some(
                    d.get_first(self.fields.location)
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                ))
            }
            None => Ok(None),
        }
    }

    /// Return `true` iff the index contains at least one chunk whose `path` field equals `path`.
    /// Used by `graph_upsert` to reject hallucinated `source_path` values.
    pub fn has_document(&self, path: &str) -> anyhow::Result<bool> {
        use tantivy::query::TermQuery;
        let searcher = self.reader.searcher();
        let q = TermQuery::new(
            tantivy::Term::from_field_text(self.fields.path, path),
            IndexRecordOption::Basic,
        );
        let top = searcher.search(&q, &TopDocs::with_limit(1).order_by_score())?;
        Ok(!top.is_empty())
    }

    /// The stored `file_type` of the first chunk indexed under `path`, or `None` if no chunk has
    /// that exact path. Tells a notebook note chunk (`file_type == "note"`) from a corpus document,
    /// and reads the type of a note's owner document. Mirrors `last_chunk_ord`'s single-term lookup.
    pub fn file_type_of(&self, path: &str) -> anyhow::Result<Option<String>> {
        use tantivy::query::TermQuery;
        let searcher = self.reader.searcher();
        let q = TermQuery::new(
            tantivy::Term::from_field_text(self.fields.path, path),
            IndexRecordOption::Basic,
        );
        let top = searcher.search(&q, &TopDocs::with_limit(1).order_by_score())?;
        match top.first() {
            Some((_score, addr)) => {
                let d: TantivyDocument = searcher.doc(*addr)?;
                Ok(d.get_first(self.fields.file_type)
                    .and_then(|v| v.as_str())
                    .map(str::to_string))
            }
            None => Ok(None),
        }
    }

    /// The corpus document a notebook note belongs to: the longest `/`-prefix of `rel` that is
    /// itself an indexed, non-note document (the note's mirror directory). `None` for corpus
    /// paths, paths with no indexed prefix, or paths whose only indexed prefix is another note.
    /// Equivalent to the index-authority rule in `crate::notebook::paths::split_notebook_path`.
    pub fn note_owner(&self, rel: &str) -> anyhow::Result<Option<String>> {
        for (i, _) in rel.match_indices('/').rev() {
            let candidate = &rel[..i];
            if self.has_document(candidate)? && self.file_type_of(candidate)? != Some("note".into())
            {
                return Ok(Some(candidate.to_string()));
            }
        }
        Ok(None)
    }

    /// Resolve a `location` string to the chunk number (`ord`) stored in the index for `path`.
    /// Mirrors `read_chunk` (path+location BooleanQuery) but returns the `ord` field.
    /// Returns `None` when no chunk matches that (path, location) pair.
    pub fn ord_for_location(&self, path: &str, location: &str) -> anyhow::Result<Option<u64>> {
        use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
        let searcher = self.reader.searcher();
        let clauses: Vec<(Occur, Box<dyn Query>)> = vec![
            (
                Occur::Must,
                Box::new(TermQuery::new(
                    tantivy::Term::from_field_text(self.fields.path, path),
                    IndexRecordOption::Basic,
                )),
            ),
            (
                Occur::Must,
                Box::new(TermQuery::new(
                    tantivy::Term::from_field_text(self.fields.location, location),
                    IndexRecordOption::Basic,
                )),
            ),
        ];
        let top = searcher.search(
            &BooleanQuery::new(clauses),
            &TopDocs::with_limit(1).order_by_score(),
        )?;
        match top.first() {
            Some((_score, addr)) => {
                let d: TantivyDocument = searcher.doc(*addr)?;
                Ok(d.get_first(self.fields.ord).and_then(|v| v.as_u64()))
            }
            None => Ok(None),
        }
    }

    /// The largest chunk number (`ord`) indexed for `path`, or `None` if no chunk exists for that
    /// exact path. Lets a failed `read` report the document's valid range instead of a dead end.
    pub fn last_chunk_ord(&self, path: &str) -> anyhow::Result<Option<u64>> {
        use tantivy::collector::DocSetCollector;
        use tantivy::query::TermQuery;
        let searcher = self.reader.searcher();
        let q = TermQuery::new(
            tantivy::Term::from_field_text(self.fields.path, path),
            IndexRecordOption::Basic,
        );
        let mut max: Option<u64> = None;
        for addr in searcher.search(&q, &DocSetCollector)? {
            let d: TantivyDocument = searcher.doc(addr)?;
            if let Some(ord) = d.get_first(self.fields.ord).and_then(|v| v.as_u64()) {
                max = Some(max.map_or(ord, |m| m.max(ord)));
            }
        }
        Ok(max)
    }

    /// Agent-supplied path → canonical index key. Fast exact hit, else [`Self::resolve_path`].
    pub fn canonical_document_path(&self, input: &str) -> Option<String> {
        // Read/section tools print a chunk as `path #ord · label`; when the model
        // reuses that path in graph_upsert it arrives as `path#ord`. Strip the
        // trailing `#<ord>` anchor so a real document still resolves — the exact and
        // tolerant matches below keep the hallucination guard (a stripped path that
        // is not indexed still returns None).
        let input = strip_section_anchor(input);
        let input = input.as_str();
        if self.has_document(input).unwrap_or(false) {
            return Some(input.to_string());
        }
        if let Some(p) = self.resolve_path(input).ok().flatten() {
            return Some(p);
        }
        // `glossary` prints an ungrounded reasoning node's owner document as `@<path>` (the
        // node has no chunk anchor, so the source path is its only pointer). The model copies
        // that sigil verbatim into read/graph_upsert. Retry once with a leading `@` stripped —
        // but only as a fallback, AFTER the exact/tolerant matches above, so a document whose
        // real path genuinely begins with `@` (e.g. `@types/foo.md`) still resolves first.
        if let Some(stripped) = input.strip_prefix('@') {
            if self.has_document(stripped).unwrap_or(false) {
                return Some(stripped.to_string());
            }
            return self.resolve_path(stripped).ok().flatten();
        }
        None
    }

    /// Resolve a possibly-mangled `input` path to the real indexed path by collapsing runs of
    /// whitespace and underscores (the model routinely turns a document's double space into a
    /// single one, or swaps spaces for underscores, when copying a path) and stripping spurious
    /// leading path segments (e.g. a corpus-folder prefix the model prepends even though search
    /// results omit it). Returns the exact path only when exactly one indexed document matches —
    /// never guesses between ambiguous candidates.
    pub fn resolve_path(&self, input: &str) -> anyhow::Result<Option<String>> {
        /// Normalize path separators: collapse runs of `/` or `\` into one,
        /// replace with host OS separator.
        fn normalize_path(s: &str) -> String {
            let mut out = String::with_capacity(s.len());
            let mut chars = s.chars().peekable();
            while let Some(c) = chars.next() {
                if c == '\\' || c == '/' {
                    // Skip any following separators (collapse runs)
                    while chars.peek() == Some(&'\\') || chars.peek() == Some(&'/') {
                        chars.next();
                    }
                    // Emit the host separator
                    #[cfg(windows)]
                    out.push('\\');
                    #[cfg(not(windows))]
                    out.push('/');
                } else {
                    out.push(c);
                }
            }
            out
        }
        fn norm(s: &str) -> String {
            // Normalize separators first, then collapse runs of whitespace AND underscores
            // to a single space. A corpus that mixes spaced and underscored filenames (e.g. a
            // spaced PDF sitting next to underscore-named HTML/PNG siblings) leads the model to
            // "regularize" one style into the other when copying a path, so treat `_` and space
            // as equivalent. Applied to both the input and every stored path, so the fold is
            // symmetric; `match_normalized` still resolves only when exactly one document
            // matches, so this never guesses between two names that differ only by `_` vs space.
            let normalized = normalize_path(s);
            let mut out = String::with_capacity(normalized.len());
            let mut prev_ws = false;
            for c in normalized.chars() {
                if c.is_whitespace() || c == '_' {
                    if !prev_ws && !out.is_empty() {
                        out.push(' ');
                    }
                    prev_ws = true;
                } else {
                    out.push(c);
                    prev_ws = false;
                }
            }
            out.trim().to_string()
        }
        fn match_normalized(
            idx: &DocIndex,
            target: &str,
            norm: &impl Fn(&str) -> String,
        ) -> anyhow::Result<Option<String>> {
            let mut seen = std::collections::HashSet::new();
            let mut matches: Vec<String> = Vec::new();
            idx.iter_chunks(|path, _ord, _ft, _body| {
                if seen.insert(path.to_string()) && norm(path) == target {
                    matches.push(path.to_string());
                }
            })?;
            Ok(if matches.len() == 1 {
                matches.pop()
            } else {
                None
            })
        }
        if let Some(p) = match_normalized(self, &norm(input), &norm)? {
            return Ok(Some(p));
        }
        let segments: Vec<&str> = input.split(['\\', '/']).filter(|s| !s.is_empty()).collect();
        for i in 1..segments.len() {
            let stripped = segments[i..].join("\\");
            if let Some(p) = match_normalized(self, &norm(&stripped), &norm)? {
                return Ok(Some(p));
            }
        }
        Ok(None)
    }
}

/// Strip a trailing section anchor (`#<ord>`, optionally followed by ` · label`) that
/// the read/section tools append when printing a chunk, so `doc.docx#3` collapses to
/// `doc.docx`. Only a `#` immediately followed by a digit counts as an anchor — a `#`
/// that is part of the real filename (e.g. `C#_notes.md`) is left untouched.
pub(crate) fn strip_section_anchor(input: &str) -> String {
    match input.rfind('#') {
        Some(h) if input[h + 1..].starts_with(|c: char| c.is_ascii_digit()) => {
            input[..h].trim_end().to_string()
        }
        _ => input.to_string(),
    }
}

/// The chunk's single canonical number within its document: the page number for PDFs
/// (parsed from the `p.N` location), otherwise the 1-based sequence position `seq`.
pub fn chunk_ord(file_type: &str, location: &str, seq: u64) -> u64 {
    if file_type == "pdf" {
        if let Some(n) = location
            .strip_prefix("p.")
            .and_then(|d| d.parse::<u64>().ok())
        {
            return n;
        }
    }
    seq
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct IndexStats {
    pub added: usize,
    pub removed: usize,
    pub unchanged: usize,
    /// Files that were reached but failed to extract (corrupt/unreadable), as `(path, error)`. The
    /// pass continues past them; the CLI prints this list at the end so they aren't lost in scroll.
    pub errors: Vec<(String, String)>,
    /// Files whose read failed transiently (per `is_transient`) even after `with_read_retry`
    /// exhausted its retries — NOT counted in `errors`/`unchanged`: the manifest sig is reverted
    /// (old file) or dropped (new file) so the file is retried on the next pass, and its dir's
    /// dirsig is held back (see `index_dir_at_locked`'s `unsettled_extra`).
    pub transient_failures: usize,
    /// Files whose read failed permanently (bad CFB/PDF, `InvalidData`, `PermissionDenied`,
    /// `NotFound` mid-read) — the current sig stays in `next.files`, so the file is treated as
    /// unchanged and NOT retried next pass (matches today's bad-.doc skip). A duplicate of the same
    /// path also lands in `errors` (for the CLI's end-of-run error summary).
    pub permanent_skips: usize,
    /// Roots currently held back by the empty-mount guard (Task 8's `delta.empty_mount_roots`),
    /// i.e. a previously-populated root whose walk this pass came back suspiciously empty. A gauge,
    /// not a cumulative count — reflects this pass's snapshot so callers (`GlossaServer::freshen_now`)
    /// can overwrite rather than accumulate.
    pub empty_mount_holds: usize,
}

pub fn file_sig(path: &Path) -> anyhow::Result<FileSig> {
    #[cfg(test)]
    if let Some(e) = read_fault::take_stat(path) {
        return Err(anyhow::Error::from(e).context("file_sig (injected)"));
    }
    let md = std::fs::metadata(path)?;
    let mtime_secs = md
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Ok(FileSig {
        mtime_secs,
        size: md.len(),
    })
}

/// Test-only fault-injection seam for the two read primitives `index_file_into` uses: `file_sig`
/// (stat) and `extract_file`'s body read. Separate STAT/READ maps so arming one does not trip the
/// other within a single `index_file_into` call (which stats then reads). This is the single
/// injection mechanism reused by the network-read-resilience test suite — no other seam is
/// introduced anywhere else.
#[cfg(test)]
pub(crate) mod read_fault {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::io;
    use std::path::{Path, PathBuf};
    #[derive(Clone)]
    struct Armed {
        remaining: u32,
        kind: io::ErrorKind,
        errno: Option<i32>,
    }
    thread_local! {
        static STAT: RefCell<HashMap<PathBuf, Armed>> = RefCell::new(HashMap::new());
        static READ: RefCell<HashMap<PathBuf, Armed>> = RefCell::new(HashMap::new());
    }
    fn take(
        map: &'static std::thread::LocalKey<RefCell<HashMap<PathBuf, Armed>>>,
        abs: &Path,
    ) -> Option<io::Error> {
        map.with(|m| {
            let mut m = m.borrow_mut();
            let a = m.get_mut(abs)?;
            if a.remaining == 0 {
                return None;
            }
            a.remaining -= 1;
            Some(match a.errno {
                Some(e) => io::Error::from_raw_os_error(e),
                None => io::Error::from(a.kind),
            })
        })
    }
    /// Arm the next `k` STATs of `abs` to fail (consulted in `file_sig`).
    pub fn arm_stat(abs: &Path, k: u32, kind: io::ErrorKind, errno: Option<i32>) {
        STAT.with(|m| {
            m.borrow_mut().insert(
                abs.to_path_buf(),
                Armed {
                    remaining: k,
                    kind,
                    errno,
                },
            );
        });
    }
    /// Arm the next `k` body READs of `abs` to fail (consulted in `extract_file`).
    pub fn arm_read(abs: &Path, k: u32, kind: io::ErrorKind, errno: Option<i32>) {
        READ.with(|m| {
            m.borrow_mut().insert(
                abs.to_path_buf(),
                Armed {
                    remaining: k,
                    kind,
                    errno,
                },
            );
        });
    }
    pub fn take_stat(abs: &Path) -> Option<io::Error> {
        take(&STAT, abs)
    }
    pub fn take_read(abs: &Path) -> Option<io::Error> {
        take(&READ, abs)
    }
    /// Clear all armed faults (call at end of each test — the maps are thread-local + reused).
    pub fn clear() {
        STAT.with(|m| m.borrow_mut().clear());
        READ.with(|m| m.borrow_mut().clear());
    }
}

/// Test-only seam for `scan_scoped_delta_at`'s per-dir loop: forces the SAME "stop between dirs"
/// behavior the real `deadline` produces, but keyed off a dir COUNT instead of wall-clock time.
/// A real-Instant deadline can't deterministically land "after dir 1, before dir 2" in a test —
/// both are near-instant operations on a tiny temp dir, so any window tight enough to split them
/// is also tight enough to flake under CI scheduling jitter. This makes a genuine partial
/// truncation (some dirs actually, fully scanned; others held back) reproducible without a race,
/// for exercising per-root logic (e.g. the empty-mount guard) that depends on WHICH dirs were
/// scanned vs held, not on why.
#[cfg(test)]
pub(crate) mod deadline_fault {
    use std::cell::Cell;
    thread_local! {
        static AFTER: Cell<Option<u32>> = const { Cell::new(None) };
    }
    /// Arm: the loop must stop once `n` dirs have already been fully scanned this pass.
    pub fn arm_after(n: u32) {
        AFTER.with(|c| c.set(Some(n)));
    }
    pub fn clear() {
        AFTER.with(|c| c.set(None));
    }
    /// Checked at the same between-iterations point as the real deadline. `scanned_so_far` is the
    /// count of dirs this pass has already fully scanned before the current iteration.
    pub(crate) fn should_break(scanned_so_far: u32) -> bool {
        AFTER.with(|c| matches!(c.get(), Some(n) if scanned_so_far >= n))
    }
}

/// Reindex ONE notebook note (an external in-place content edit picked up on read), assuming the
/// caller holds `index.lock`. A note is a single `"note"` chunk with no graph edges, so this is an
/// idempotent `write_chunks` (delete-by-path → add → commit) plus a `manifest.notes` sig update.
/// `None` if the note file is gone/unreadable — the manifest is left untouched (deletion changes the
/// notes dir mtime and is handled by the freshen notes pass, which drops the chunk).
pub fn reindex_note_locked(dir: &Path, rel: &str) -> anyhow::Result<Option<FileSig>> {
    let abs = dir.join(".glossa").join("notes").join(rel);
    let body = match std::fs::read_to_string(&abs) {
        Ok(b) => b,
        Err(_) => return Ok(None),
    };
    let idx = DocIndex::open_or_create(dir)?;
    idx.write_chunks(&[crate::model::Chunk {
        doc_path: std::path::PathBuf::from(rel),
        location: "note".into(),
        file_type: "note".into(),
        text: body,
    }])?;
    let sig = file_sig(&abs)?;
    let mut m = Manifest::load(dir);
    m.notes.insert(rel.to_string(), sig);
    m.save(dir)?;
    Ok(Some(sig))
}

impl DocIndex {
    /// Visit every stored chunk: `f(path, ord, file_type, body)`. Used by grep's full scan.
    pub fn iter_chunks(&self, mut f: impl FnMut(&str, u64, &str, &str)) -> anyhow::Result<()> {
        use tantivy::collector::DocSetCollector;
        use tantivy::query::AllQuery;
        let searcher = self.reader.searcher();
        let docs = searcher.search(&AllQuery, &DocSetCollector)?;
        for addr in docs {
            let d: TantivyDocument = searcher.doc(addr)?;
            let s = |fld| d.get_first(fld).and_then(|v| v.as_str()).unwrap_or("");
            let ord = d
                .get_first(self.fields.ord)
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            f(
                s(self.fields.path),
                ord,
                s(self.fields.file_type),
                s(self.fields.body),
            );
        }
        Ok(())
    }

    /// Visit chunks whose indexed char-trigrams satisfy an AND of `grams` (grep prefilter).
    pub fn iter_chunks_trigram_candidates(
        &self,
        grams: &[String],
        mut f: impl FnMut(&str, u64, &str, &str),
    ) -> anyhow::Result<()> {
        if grams.is_empty() {
            return self.iter_chunks(f);
        }
        use tantivy::collector::DocSetCollector;
        use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
        let clauses: Vec<(Occur, Box<dyn Query>)> = grams
            .iter()
            .map(|g| {
                (
                    Occur::Must,
                    Box::new(TermQuery::new(
                        tantivy::Term::from_field_text(self.fields.body_trigrams, g),
                        IndexRecordOption::Basic,
                    )) as Box<dyn Query>,
                )
            })
            .collect();
        let query = BooleanQuery::new(clauses);
        let searcher = self.reader.searcher();
        let docs = searcher.search(&query, &DocSetCollector)?;
        for addr in docs {
            let d: TantivyDocument = searcher.doc(addr)?;
            let s = |fld| d.get_first(fld).and_then(|v| v.as_str()).unwrap_or("");
            let ord = d
                .get_first(self.fields.ord)
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            f(
                s(self.fields.path),
                ord,
                s(self.fields.file_type),
                s(self.fields.body),
            );
        }
        Ok(())
    }

    pub fn delete_path(&self, path: &str) -> anyhow::Result<()> {
        with_writer_retry(|| {
            let mut writer = self.index.writer::<TantivyDocument>(50_000_000)?;
            writer.delete_term(tantivy::Term::from_field_text(self.fields.path, path));
            writer.commit()?;
            Ok(())
        })?;
        self.reader.reload()?;
        Ok(())
    }
}

/// Filesystem delta vs a saved `Manifest`, computed with `stat` only (no extraction, no writer
/// lock, no commit). `changed` = new-or-modified files; `removed` = manifest files no longer on
/// disk; `next` = the fresh manifest for the current tree. This is the cheap hot-path gate that
/// lets callers (CLI commands, the MCP server) decide whether any real indexing work is needed.
#[derive(Debug, Default)]
pub struct Delta {
    pub changed: Vec<String>,
    pub removed: Vec<String>,
    /// Notebook notes (path relative to `.glossa/notes`) that are new or modified.
    pub notes_changed: Vec<String>,
    /// Notebook notes in the saved manifest that are gone from disk.
    pub notes_removed: Vec<String>,
    pub next: Manifest,
    /// Every corpus doc key seen this scan (both changed and unchanged), mapped to the absolute
    /// path it was walked at. Populated by `scan_delta_at` so a multi-root indexing pass never
    /// needs to re-derive a key's root: with >1 root, `some_root.join(key)` alone is wrong (the
    /// key may carry a DIFFERENT root's label), so the walk records the pairing once, here.
    pub abs_paths: BTreeMap<String, PathBuf>,
    /// Doc keys whose `file_sig` stat failed TRANSIENTLY this scan (per `is_transient`) — the file
    /// is still present in `next.files` (carried forward from the OLD manifest sig, not dropped as
    /// a delete). Callers (`index_dir_at_locked`/`reindex_dirs_at_locked`) map each key's parent dir
    /// into their unsettled set so a blip doesn't advance that dir's dirsig and get skipped forever.
    pub stat_failed: Vec<String>,
    /// Labels of roots held back this pass because a previously-populated root's walk came back
    /// suspiciously empty (Task 8): the walk succeeded (no io error) but found zero files, despite
    /// the manifest showing prior files under that root. Purely observational — `d.next.files`
    /// already carries the held-back docs forward, so nothing downstream needs to special-case this
    /// field except logging/metrics and the dirsig-advance sites (hold back that root's dirs so the
    /// next freshen re-attempts the walk instead of settling on the empty snapshot).
    pub empty_mount_roots: Vec<String>,
    /// Changed/added corpus dir keys NOT scanned this pass because a caller-supplied `deadline`
    /// (Task 10, D1) elapsed mid-walk — the scoped rescan stops between dirs, not mid-dir. Every
    /// key here must be excluded from the "scanned" set used to decide what gets carried forward
    /// (its manifest files are untouched, so they carry forward as-is) AND held back from the
    /// dirsig advance, so the NEXT freshen re-diffs and finishes exactly these dirs. Serve-stale:
    /// never partially settles a dir it didn't actually get to.
    pub deadline_held_dirs: Vec<String>,
}

/// Per-directory mtime map: `c:{rel}` for corpus dirs (gitignore-aware, skips `.glossa`), `n:{rel}`
/// for `.glossa/notes` dirs; values are nanosecond mtimes. Add/remove/rename of a file bumps its
/// parent dir's entry; a new/removed subdir adds/drops an entry. In-place content edits don't change it.
pub fn dir_mtime_map(dir: &Path) -> anyhow::Result<BTreeMap<String, u128>> {
    dir_mtime_map_at(
        &[crate::root::Root {
            label: String::new(),
            path: dir.to_path_buf(),
        }],
        dir,
    )
}

/// Multi-root primitive: one `collect_dir_mtimes` pass per root, each keyed with a per-root prefix
/// (`c:{label}:` for a labeled root, plain `c:` for the back-compat empty label) so identical
/// relpaths in two different roots never collide in the combined map. Notes live once, under
/// `state_base/.glossa/notes` — they are not per-root.
pub fn dir_mtime_map_at(
    roots: &[crate::root::Root],
    state_base: &Path,
) -> anyhow::Result<BTreeMap<String, u128>> {
    let mut map: BTreeMap<String, u128> = BTreeMap::new();
    for r in roots {
        let root = abs_root(&r.path);
        let prefix = if r.label.is_empty() {
            "c".to_string()
        } else {
            format!("c:{}", r.label)
        };
        collect_dir_mtimes(&root, true, &prefix, &mut map);
    }
    let notes_root = state_base.join(".glossa").join("notes");
    if notes_root.is_dir() {
        collect_dir_mtimes(&notes_root, false, "n", &mut map);
    }
    Ok(map)
}

/// Stable hash of every directory's mtime under the corpus (skipping `.glossa`) plus every
/// directory under `.glossa/notes`. Adding/removing/renaming a file bumps its parent dir's mtime
/// (and new/removed dirs change the set), so the hash changes; an in-place content edit does not.
/// Uses nanosecond mtime so a same-second add is still detected. O(dirs), not O(files).
pub fn dir_mtime_signature(dir: &Path) -> anyhow::Result<u64> {
    let map = dir_mtime_map(dir)?;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for (k, v) in &map {
        k.hash(&mut h);
        v.hash(&mut h);
    }
    Ok(h.finish())
}

/// Walk `base` with the same `WalkBuilder` configuration `walk_files` (`src/walk.rs`) uses for
/// indexing — gitignore/hidden-aware when `respect_ignore`, skipping `.glossa`, not following
/// symlinks — so the signature tracks exactly what gets indexed (no `.git` churn, no symlink
/// cycles) and stays consistent between the two callers (corpus pass vs. `.glossa/notes` pass,
/// which mirrors how `scan_notes_delta` walks notes with `respect_ignore=false`). Records each
/// directory entry's path (relative to `base`, prefixed with `key_prefix` to keep corpus and notes
/// trees distinct in the combined map) and its mtime in nanoseconds.
fn collect_dir_mtimes(
    base: &Path,
    respect_ignore: bool,
    key_prefix: &str,
    map: &mut BTreeMap<String, u128>,
) {
    let mut wb = WalkBuilder::new(base);
    wb.standard_filters(respect_ignore);
    wb.require_git(!respect_ignore);
    wb.filter_entry(|e| e.file_name() != ".glossa");
    for result in wb.build() {
        let entry = match result {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let path = entry.path();
        let nanos = std::fs::metadata(path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let rel = path
            .strip_prefix(base)
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .unwrap_or_default();
        let key = format!("{key_prefix}:{rel}");
        map.insert(key, nanos);
    }
}

/// True iff a `/`-prefix of `rel` is an indexed corpus document in this pass. `files` contains only
/// corpus documents (never notes), so a hit means the note's owner is present.
#[cfg(feature = "notebook")]
fn owner_in(files: &BTreeMap<String, FileSig>, rel: &str) -> bool {
    let mut cut = rel.rfind('/');
    while let Some(i) = cut {
        if files.contains_key(&rel[..i]) {
            return true;
        }
        cut = rel[..i].rfind('/');
    }
    false
}

pub fn scan_delta(dir: &Path, manifest: &Manifest) -> anyhow::Result<Delta> {
    scan_delta_at(
        &[crate::root::Root {
            label: String::new(),
            path: dir.to_path_buf(),
        }],
        dir,
        manifest,
    )
}

/// Multi-root primitive: walk EVERY root, keying each file with `doc_key(&r.label, root_abs, p)`
/// so identical relpaths across roots never collide, and accumulating into ONE `Delta`. Notes
/// (single tree, not per-root) live under `state_base/.glossa/notes`.
pub fn scan_delta_at(
    roots: &[crate::root::Root],
    state_base: &Path,
    manifest: &Manifest,
) -> anyhow::Result<Delta> {
    let mut d = Delta::default();
    let mut stat_failed: Vec<String> = Vec::new();
    for r in roots {
        let root_abs = abs_root(&r.path);
        let before = d.next.files.len();
        let had_prior = manifest.files.keys().any(|k| {
            let (label, _) = split_doc_key(k, roots);
            label == r.label
        });
        crate::walk::walk_files(&root_abs, None, true, &mut |path| {
            let key = doc_key(&r.label, &root_abs, path);
            let sig = match file_sig(path) {
                Ok(s) => s,
                Err(e) => {
                    // A transient stat failure must NOT drop the file (that reads as a delete).
                    // Carry the old manifest sig (and this pass's abs path, so a retry can still
                    // resolve it) forward; flag the labeled key so its dir's dirsig is not advanced
                    // (next freshen re-scans). A permanent stat error (NotFound) falls through to
                    // the removed-loop as today.
                    if is_transient(&e) {
                        if let Some(&old) = manifest.files.get(&key) {
                            d.next.files.insert(key.clone(), old);
                            d.abs_paths.insert(key.clone(), path.to_path_buf());
                        }
                        stat_failed.push(key);
                    }
                    return Ok(());
                }
            };
            if manifest.changed(&key, sig) {
                d.changed.push(key.clone());
            }
            d.next.files.insert(key.clone(), sig);
            d.abs_paths.insert(key, path.to_path_buf());
            Ok(())
        })?;
        let found_this_root = d.next.files.len() - before;
        if had_prior && found_this_root == 0 {
            // Suspicious: this root previously had files, the walk succeeded (no io error), but
            // found none. Treat as a possibly-unmounted share, NOT a corpus-side mass delete: carry
            // every previously-known file under this root's label forward into next.files at its
            // old signature (skips the removed-loop entirely for this root) rather than letting the
            // normal `manifest.files - d.next.files` diff below erase them. `kb index --force`
            // resets the manifest before scanning (`had_prior` is then always false), so an operator
            // who has genuinely emptied a root on purpose can still make the empty state stick.
            for (k, sig) in &manifest.files {
                let (label, _) = split_doc_key(k, roots);
                if label == r.label {
                    d.next.files.insert(k.clone(), *sig);
                    // No abs_path available (nothing was walked) — Task 5/6's loops only touch
                    // `delta.changed`, and this root contributed nothing to `changed`, so this is
                    // safe.
                }
            }
            d.empty_mount_roots.push(r.label.clone());
            tracing::warn!(root = %r.label, "root walk returned zero files but was previously populated; holding stale index (possible unmounted network share)");
        }
    }
    d.removed = manifest
        .files
        .keys()
        .filter(|k| !d.next.files.contains_key(*k))
        .cloned()
        .collect();
    d.stat_failed = stat_failed;
    #[cfg(feature = "notebook")]
    scan_notes_delta(state_base, manifest, &mut d)?;
    Ok(d)
}

/// Notes half of `scan_delta`: walk `.glossa/notes` and diff each note's signature against
/// `manifest.notes`. Walking the notes root directly (with `respect_ignore=false`) is safe:
/// the `.glossa` skip in `walk_files` only matches entries *named* `.glossa`, and `false`
/// disables hidden/gitignore filtering that would otherwise hide `.glossa` itself.
#[cfg(feature = "notebook")]
fn scan_notes_delta(dir: &Path, manifest: &Manifest, d: &mut Delta) -> anyhow::Result<()> {
    let notes_root = dir.join(".glossa").join("notes");
    if !notes_root.is_dir() {
        // The whole notes tree is gone → every saved note is removed.
        d.notes_removed = manifest.notes.keys().cloned().collect();
        return Ok(());
    }
    crate::walk::walk_files(&notes_root, None, false, &mut |path| {
        let rel = path
            .strip_prefix(&notes_root)
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .unwrap_or_default();
        if rel.is_empty() {
            return Ok(());
        }
        if !owner_in(&d.next.files, &rel) {
            return Ok(()); // orphan: not a live note → falls into notes_removed, chunk dropped; file kept
        }
        let sig = match file_sig(path) {
            Ok(s) => s,
            Err(_) => return Ok(()),
        };
        if manifest.notes.get(&rel) != Some(&sig) {
            d.notes_changed.push(rel.clone());
        }
        d.next.notes.insert(rel, sig);
        Ok(())
    })?;
    d.notes_removed = manifest
        .notes
        .keys()
        .filter(|k| !d.next.notes.contains_key(*k))
        .cloned()
        .collect();
    Ok(())
}

/// Notes-root-relative paths of every note file whose owner document is not in the corpus.
#[cfg(feature = "notebook")]
pub fn orphan_notes(dir: &Path) -> anyhow::Result<Vec<String>> {
    orphan_notes_at(
        &[crate::root::Root {
            label: String::new(),
            path: dir.to_path_buf(),
        }],
        dir,
    )
}

/// Multi-root primitive behind `orphan_notes`: the owner-document check must see files from EVERY
/// root, not just `state_base` — `scan_delta(state_base, ...)` alone walks `state_base` as if it
/// were the sole corpus root, finds none of the actual corpus files there in separated state-dir
/// mode, and flags EVERY note as orphaned (a non-dry-run `kb prune` would then delete every note
/// file). Same class of bug as `ensure_fresh` vs `ensure_fresh_at`.
#[cfg(feature = "notebook")]
pub fn orphan_notes_at(
    roots: &[crate::root::Root],
    state_base: &Path,
) -> anyhow::Result<Vec<String>> {
    let manifest = Manifest::load(state_base);
    let delta = scan_delta_at(roots, state_base, &manifest)?;
    let files = &delta.next.files;
    let notes_root = state_base.join(".glossa").join("notes");
    let mut orphans = Vec::new();
    if notes_root.is_dir() {
        crate::walk::walk_files(&notes_root, None, false, &mut |path| {
            let rel = path
                .strip_prefix(&notes_root)
                .map(|p| p.to_string_lossy().replace('\\', "/"))
                .unwrap_or_default();
            if !rel.is_empty() && !owner_in(files, &rel) {
                orphans.push(rel);
            }
            Ok(())
        })?;
    }
    Ok(orphans)
}

/// Bring the on-disk index/graph up to date with the filesystem — cheap and concurrency-safe.
/// A lock-free `stat` pre-scan short-circuits the common "nothing changed" case with zero writes.
/// Only when a delta exists does it call `index_dir` (which takes the tantivy writer lock + commits).
/// If another process already holds that lock (it is indexing the same delta), this returns a no-op
/// stat instead of erroring: freshness is COOPERATIVE — coordinated by the writer lock, not owned by
/// any one process — so both the CLI and a long-lived MCP server can keep the index fresh safely.
pub fn ensure_fresh(dir: &Path) -> anyhow::Result<IndexStats> {
    ensure_fresh_at(
        &[crate::root::Root {
            label: String::new(),
            path: dir.to_path_buf(),
        }],
        dir,
    )
}

/// Multi-root primitive behind `ensure_fresh`: the pre-scan (`scan_delta_at`) and the eventual
/// reindex (`index_dir_at`) both walk EVERY root, keyed by label — the single-root delegate above
/// must NOT be used once `roots` and `state_base` can diverge (state-dir separation), because a
/// scan rooted at `state_base` alone sees none of the actual corpus files there and misreads every
/// already-indexed doc as deleted, DESTRUCTIVELY dropping it on the next reindex.
pub fn ensure_fresh_at(
    roots: &[crate::root::Root],
    state_base: &Path,
) -> anyhow::Result<IndexStats> {
    let manifest = Manifest::load(state_base);
    let delta = scan_delta_at(roots, state_base, &manifest)?;
    if delta.changed.is_empty()
        && delta.removed.is_empty()
        && delta.notes_changed.is_empty()
        && delta.notes_removed.is_empty()
        && delta.empty_mount_roots.is_empty()
    {
        return Ok(IndexStats {
            added: 0,
            removed: 0,
            unchanged: delta.next.files.len() + delta.next.notes.len(),
            ..Default::default()
        });
    }
    // An empty-mount hold (Task 8) falls through here too, even though this pre-scan's own delta
    // shows nothing `changed`/`removed`: `index_dir_at`'s locked pass is what actually persists the
    // dirsig hold-back (see `hold_back_empty_mount_roots`), and skipping it would leave the stale
    // snapshot to be silently recomputed (and re-logged) on every call without ever sticking.
    // Something changed → take the lock and index. index_dir_at recomputes the delta FRESH inside
    // the lock (this pre-scan is lock-free and only decides whether to bother): reusing this delta
    // across the lock boundary would be unsound — a concurrent indexer could land between, and our
    // stale file set would then delete what it just added.
    match index_dir_at(roots, state_base, false) {
        Ok(s) => Ok(s),
        Err(e) if is_lock_busy(&e) => Ok(IndexStats::default()),
        Err(e) => Err(e),
    }
}

/// True if `e` is a tantivy writer-lock contention error (another process is indexing right now).
fn is_lock_busy(e: &anyhow::Error) -> bool {
    matches!(
        e.downcast_ref::<TantivyError>(),
        Some(TantivyError::LockFailure(_, _))
    )
}

/// Retry a tantivy index-write/open past a transient Windows filesystem error. A file that was just
/// created (or is still mmap'd by an open reader) can be briefly held by the OS — Windows Defender
/// scanning it, a lingering handle, an in-progress `meta.json` temp+rename — so opening the index or
/// writer, or committing, occasionally fails with a transient IO/lock error. A short BOUNDED backoff
/// (capped attempts) clears it; on other platforms the first attempt succeeds. `is_transient_fs`
/// decides what counts as transient; a structural error propagates immediately, and even a
/// persistently-transient-looking error propagates once the attempt cap is hit (never an infinite loop).
fn with_writer_retry<T>(mut op: impl FnMut() -> tantivy::Result<T>) -> tantivy::Result<T> {
    let mut attempt = 0u32;
    loop {
        match op() {
            Err(e) if attempt < 4 && is_transient_fs(&e) => {
                attempt += 1;
                std::thread::sleep(std::time::Duration::from_millis(u64::from(40 * attempt)));
            }
            other => return other,
        }
    }
}

/// `GLOSSA_READ_RETRIES` (default 3) — max retry attempts for a transient corpus-read failure.
fn read_retries() -> u32 {
    std::env::var("GLOSSA_READ_RETRIES")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(3)
}

/// `GLOSSA_READ_RETRY_BACKOFF_MS` (default 200) — base backoff; doubles each attempt.
fn read_retry_backoff_ms() -> u64 {
    std::env::var("GLOSSA_READ_RETRY_BACKOFF_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(200)
}

/// `GLOSSA_FRESHEN_DEADLINE_MS` (default 3000) — wall budget for the awaited freshen walk+reindex
/// (Task 10, D1). Bounds `reindex_dirs_at_locked`'s scoped per-dir walk on the latency-bounded
/// `freshen_blocking_at` path only; `kb index` (`index_dir_at_locked`) is unbounded (`deadline:
/// None`) and always uses the full `read_retries()` budget, never this one.
fn freshen_deadline_ms() -> u64 {
    std::env::var("GLOSSA_FRESHEN_DEADLINE_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        // Test builds default to an effectively-unbounded wall budget. The freshen unit tests assert
        // that a pass COMPLETES (file indexed, sig recorded) — not that it times out — but under a
        // loaded `-j4` box the real 3s budget can elapse mid-pass from CPU starvation alone, dropping
        // the pass into serve-stale and flaking those completion assertions (~1/12). That is a test
        // harness race, not a behavior under test: the two tests that DO exercise a budget drive it
        // explicitly via the `timeout` (lock spin) / `deadline` (reindex) params and never touch this
        // env default. Production keeps the 3s default unchanged.
        .unwrap_or(if cfg!(test) { 600_000 } else { 3000 })
}

/// `GLOSSA_MIN_RESCAN_MS` (default 2000) — minimum spacing between freshen stat-walks (Task 10,
/// D2). Absorbs SMB/NFS attribute-cache lag and caps per-query walk cost on a hot read path where
/// `freshen_now` is awaited on nearly every tool call. Gated in `GlossaServer::freshen_now`
/// (mcp.rs) via `last_freshen_ms`, not here — this just centralizes the env-or-default read.
pub fn min_rescan_ms() -> u64 {
    std::env::var("GLOSSA_MIN_RESCAN_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(2000)
}

/// Retry `op` up to `max_retries` times when it fails with an `is_transient` error, sleeping an
/// exponentially-growing backoff between attempts. A permanent failure returns immediately (0 retries).
/// Runs under `spawn_blocking` (freshen) or the CLI thread, so a blocking sleep is acceptable.
fn with_read_retry<T>(
    max_retries: u32,
    backoff_base_ms: u64,
    mut op: impl FnMut() -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let mut attempt = 0u32;
    loop {
        match op() {
            Err(e) if attempt < max_retries && is_transient(&e) => {
                attempt += 1;
                let backoff = backoff_base_ms.saturating_mul(1u64 << (attempt - 1).min(20));
                std::thread::sleep(std::time::Duration::from_millis(backoff));
            }
            other => return other,
        }
    }
}

/// True for a transient Windows filesystem error surfaced during an index open/commit — safe to
/// retry a bounded number of times. Windows briefly holds a just-written or mmap'd index file
/// (Defender scan, a lingering reader handle, an in-progress atomic temp+rename of `meta.json`), and
/// that contention surfaces through several tantivy frames, not just `PermissionDenied`:
///   - `IoError` of ANY `io::Error` kind — Windows reports a locked / partially-read file as
///     `PermissionDenied`, `Other`, or a raw OS error, so we retry every kind (not just os error 5).
///   - `OpenDirectoryError` / `OpenReadError` / `OpenWriteError` — the directory abstraction wraps the
///     underlying io error before it reaches `TantivyError::IoError`; `meta.json` is read via
///     `atomic_read`, whose failure comes back as `OpenReadError`.
///   - `LockFailure` — the writer lockfile was momentarily held by a racing pass.
///
/// This widening is the hypothesized fix for the intermittent Windows tantivy `meta.json`/commit
/// flakes (`reindex_dirs_matches_full_across_scenarios`,
/// `freshen_blocking_picks_up_new_file_and_is_noop_when_fresh`); the exact error frame is to be
/// confirmed by a CI stress lane. It stays a BOUNDED retry (see `with_writer_retry`'s attempt cap),
/// so a genuinely permanent failure still propagates after a few wasted attempts and can never loop.
/// A structural error (`FieldNotFound`, `SchemaError`, `DataCorruption`, …) is NOT transient and
/// propagates on the first attempt — retrying can't fix it.
fn is_transient_fs(e: &TantivyError) -> bool {
    matches!(
        e,
        TantivyError::IoError(_)
            | TantivyError::OpenDirectoryError(_)
            | TantivyError::OpenReadError(_)
            | TantivyError::OpenWriteError(_)
            | TantivyError::LockFailure(_, _)
    )
}

/// True iff `err`'s chain carries an `io::Error` with a network-transient signature (safe to retry).
/// Permanent (returns false): NotFound (real delete), PermissionDenied, InvalidData, and any
/// extractor/parse error that carries no io::Error (bad CFB/PDF). Transient errno set is matched via
/// `raw_os_error()` because ESTALE(116)/EIO(5) map to `ErrorKind::Uncategorized` — but ONLY on unix:
/// `raw_os_error()` numbers are platform-specific, and on Windows os-error 5 is `ERROR_ACCESS_DENIED`
/// (a genuinely permanent failure, already caught above via `PermissionDenied`), not POSIX EIO. Gating
/// the errno table to unix keeps that collision from misclassifying a real Windows access-denied as
/// transient; on non-unix, only the cross-platform named `ErrorKind`s above apply.
pub fn is_transient(err: &anyhow::Error) -> bool {
    use std::io::ErrorKind;
    for cause in err.chain() {
        let Some(io) = cause.downcast_ref::<std::io::Error>() else {
            continue;
        };
        match io.kind() {
            ErrorKind::NotFound | ErrorKind::PermissionDenied | ErrorKind::InvalidData => {
                return false;
            }
            ErrorKind::TimedOut
            | ErrorKind::ConnectionReset
            | ErrorKind::ConnectionAborted
            | ErrorKind::WouldBlock
            | ErrorKind::Interrupted => return true,
            _ => {}
        }
        // ESTALE 116, ETIMEDOUT 110, ECONNRESET 104, ECONNABORTED 103, EHOSTUNREACH 113,
        // ENETUNREACH 101, ENETDOWN 100, EIO 5 (EIO mid-read over a flaky mount → transient).
        // unix-only: these are POSIX errno numbers and do not carry the same meaning on Windows.
        #[cfg(unix)]
        if let Some(errno) = io.raw_os_error() {
            if matches!(errno, 116 | 110 | 104 | 103 | 113 | 101 | 100 | 5) {
                return true;
            }
        }
    }
    false
}

/// The persisted per-directory map the index was last built from (`.glossa/dirsig`, JSON), or `None`
/// if absent or unparsable (a legacy u64 file parses as `None` → next pass reindexes and rewrites it).
pub fn read_dirsig(dir: &Path) -> Option<BTreeMap<String, u128>> {
    std::fs::read_to_string(dir.join(".glossa").join("dirsig"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

/// Persist the directory map atomically (temp + rename), so a concurrent reader never sees a
/// half-written value. Best-effort: failure to record it just means the next read re-scans.
fn write_dirsig(dir: &Path, map: &BTreeMap<String, u128>) {
    let glossa = dir.join(".glossa");
    if std::fs::create_dir_all(&glossa).is_err() {
        return;
    }
    let Ok(s) = serde_json::to_string(map) else {
        return;
    };
    // Atomic + retry (temp + rename past transient Windows Access-denied). Best-effort.
    let _ = crate::index::manifest::atomic_write(&glossa.join("dirsig"), s.as_bytes());
}

/// Index one file into an already-open writer + graph: drops the file's old chunks and
/// auto-graph-by-source, extracts chunks, writes each chunk + builds the structural graph
/// (Document/Section nodes, sequential/hierarchy edges), and appends the file's outgoing links
/// to `links`. Returns the file's `FileSig`, or `None` if the file is unreadable.
///
/// Reference (cross-document) link *resolution* is NOT done here — it stays in the callers,
/// since it needs the full document set.
pub fn index_file_into(
    idx: &DocIndex,
    graph: &crate::graph::store::GraphStore,
    writer: &tantivy::IndexWriter,
    // The PRECOMPUTED, already-labeled stored key (`<label>/<relpath>`, or bare `<relpath>` for the
    // back-compat empty-label root) — resolved ONCE by the caller (the walk) via
    // `doc_key(&r.label, root_abs, abs_path)`. Stored verbatim; never recomputed here. This makes
    // this function the single write-boundary for the key (no downstream re-derivation), which is
    // what keeps multi-root keys collision-free.
    doc_key: &str,
    abs_path: &Path,
    links: &mut Vec<(String, String)>,
    // `Some` ONLY on the full-coverage (`--force`) rebuild, which accumulates the grounding DF
    // table over EVERY document. Incremental/single-file/scoped passes pass `None`: a DfTable has
    // no per-source decrement, so merging just the changed docs into the existing sidecar would
    // double-count modified files and never subtract removed ones, inflating df/n_chunks over a
    // long-lived server's uptime. The sidecar is therefore refreshed only by `kb index --force`.
    mut df: Option<&mut crate::gate::df::DfTable>,
) -> anyhow::Result<Option<FileSig>> {
    // A stat error propagates (rather than being swallowed to `Ok(None)`): the caller's
    // `with_read_retry` + `is_transient` classify it (Tasks 5/6), and a genuine delete surfaces as
    // `NotFound` — permanent — for the removed-loop to handle, same as any other stat.
    let sig = file_sig(abs_path)?;

    // 1. Extract into a buffer FIRST — this is where the network read (extract.rs's
    //    `std::fs::read`) can fail. Nothing in the index/graph is mutated until extraction fully
    //    succeeds, so a transient read failure (surfaced here via `?`) leaves the prior doc + auto
    //    edges completely intact for this pass — no delete has happened yet.
    let mut chunks: Vec<Chunk> = Vec::new();
    crate::extract::extract_file(abs_path, &mut |mut c: Chunk| {
        // Canonicalize the chunk to the (already-labeled) doc key here, at the single indexing
        // boundary, so the index, the structural graph and section ids all speak ONE form.
        c.doc_path = PathBuf::from(doc_key);
        chunks.push(c);
    })?;

    // 2. Extraction succeeded — now (and only now) drop the old doc and rebuild it from the
    //    buffered chunks. Everything below is unchanged from the prior per-chunk sink body, just
    //    replayed over `chunks` instead of running inside the extract callback.
    writer.delete_term(tantivy::Term::from_field_text(idx.fields.path, doc_key));
    graph.delete_auto_by_source(doc_key)?;
    let mut doc_written = false;
    let mut seq = 0u64;
    let mut prev_sec: Option<String> = None;
    let mut file_links: Vec<String> = Vec::new();
    let mut seen: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    // Index/graph write errors are intentionally not propagated here: one bad chunk must not
    // abort the whole run (matches the prior per-file behavior). The file is still recorded
    // in the manifest; a failed write is corrected on the next `reindex`.
    for mut c in chunks {
        if !doc_written {
            let _ = crate::graph::build::build_document(graph, doc_key, sig);
            doc_written = true;
        }
        seq += 1;
        let ord = crate::index::store::chunk_ord(&c.file_type, &c.location, seq);
        // Section ids are always the ordinal now (see build_section), so `path#n`
        // from resolve_section_ref/neighbors always matches. A heading-less chunk
        // still has an empty location, which reads back as a blank node label /
        // index field — fall back to the ordinal so it shows something. This only
        // affects the label/location field, not the (already ordinal) section id.
        if c.location.is_empty() {
            c.location = ord.to_string();
        }
        // DF accumulation (full-rebuild only — see the `df` param doc) reuses the same chunk body
        // text being written to the tantivy `body` field below — one tokenize pass per chunk, no
        // separate corpus walk (rides the existing per-chunk indexing loop's progress).
        if let Some(df) = df.as_deref_mut() {
            df.add_chunk(&crate::gate::token::tokenize(&c.text));
        }
        let _ = writer.add_document(doc!(
            idx.fields.body => c.text.clone(),
            idx.fields.body_trigrams => c.text.clone(),
            idx.fields.path => doc_key.to_string(),
            idx.fields.location => c.location.clone(),
            idx.fields.file_type => c.file_type.clone(),
            idx.fields.ord => ord,
        ));
        let _ = crate::graph::build::build_section(graph, &c, ord, sig);
        let cur_id = crate::graph::build::section_id(doc_key, &ord.to_string());
        if let Some(prev) = prev_sec.as_deref() {
            let _ = crate::graph::build::link_sequential(graph, prev, &cur_id, sig, doc_key);
        }
        if let Some(parent) = crate::graph::build::nearest_ancestor(&seen, &c.location) {
            let _ = crate::graph::build::link_parent(graph, &cur_id, &parent, sig, doc_key);
        }
        file_links.extend(crate::extract::links::extract_links(&c.text));
        seen.insert(c.location.clone(), cur_id.clone());
        prev_sec = Some(cur_id);
    }
    for t in file_links {
        links.push((doc_key.to_string(), t));
    }
    Ok(Some(sig))
}

/// Back-compat wrapper: single empty-label root == state_base == `dir` (co-located, unchanged).
/// `force = true` ignores the manifest and rebuilds every file.
pub fn index_dir(dir: &Path, force: bool) -> anyhow::Result<IndexStats> {
    index_dir_at(
        &[crate::root::Root {
            label: String::new(),
            path: dir.to_path_buf(),
        }],
        dir,
        force,
    )
}

/// Primitive: walk every root, write index/graph/manifest/dirsig under `state_base/.glossa`.
/// `force = true` ignores the manifest and rebuilds every file.
/// Streams each chunk directly into the tantivy writer + graph (constant memory).
pub fn index_dir_at(
    roots: &[crate::root::Root],
    state_base: &Path,
    force: bool,
) -> anyhow::Result<IndexStats> {
    // Cross-process guard: this clears and rewrites `.glossa/index`, which collides with another
    // process doing the same on Windows ("Access is denied"). Hold `.glossa/index.lock` for the
    // whole rebuild; if another process already holds it, skip with a default (no-op) stat rather
    // than racing it — the index is cooperative, so whoever wins leaves it correct. The guard
    // drops at function exit, releasing the lock (RAII).
    let _lock = match crate::index::lock::try_index_lock(state_base) {
        Some(guard) => guard,
        None => {
            eprintln!("index_dir: another process is indexing, skipping");
            return Ok(IndexStats::default());
        }
    };
    index_dir_at_locked(roots, state_base, force)
}

/// Split a stored doc key into `(label, relpath)` — the inverse of `doc_key`. Mirrors
/// `DocIndex::doc_file`'s matching rule: a single empty-label root is always bare relpath;
/// otherwise the key's `label/` prefix (matched against a configured root) picks it apart.
/// An unmatched label (a stale key from a since-removed root) falls back to treating the whole
/// key as a bare relpath — the same defensive fallback `doc_file` uses.
fn split_doc_key<'a>(key: &'a str, roots: &[crate::root::Root]) -> (&'a str, &'a str) {
    if roots.len() == 1 && roots[0].label.is_empty() {
        return ("", key);
    }
    if let Some((label, rel)) = key.split_once('/') {
        if roots.iter().any(|r| r.label == label) {
            return (label, rel);
        }
    }
    ("", key)
}

/// Resolve collected cross-document links against the current document set, creating `REFERENCES`
/// edges for those that resolve to a real Document node, and returning the `(src, raw_target)` pairs
/// that still don't resolve (target not an indexed doc). Mirrors the resolution `index_dir_locked`
/// did inline; shared with the scoped pass. Multi-root aware: a relative link target is resolved
/// WITHIN the same root as its source (a relative link never crosses roots) — the source's root is
/// found from its stored key's label prefix (`split_doc_key`), and the resolved target is re-keyed
/// with that SAME label via `doc_key`, so cross-root REFERENCES edges are never fabricated.
fn resolve_reference_links(
    roots: &[crate::root::Root],
    graph: &crate::graph::store::GraphStore,
    files: &std::collections::BTreeMap<String, FileSig>,
    links: &[(String, String)],
) -> Vec<(String, String)> {
    // Nothing to resolve → skip touching the filesystem at all (the common case on a scoped
    // pass over a large corpus, where most dirs carry no cross-doc links).
    if links.is_empty() || roots.is_empty() {
        return links.to_vec();
    }
    // Canonicalize only the LINK TARGETS (O(links)), not every corpus document (O(files)) — a
    // scoped reindex pass over a handful of changed dirs must not pay an all-corpus filesystem
    // stat just because a link happened to be present. `canonicalize_stripped` mirrors `abs_root`'s
    // `\\?\`-stripping exactly (unlike `abs_root` itself, it returns `None` on failure rather than
    // falling back to the un-canonicalized path) so the result is in the SAME form as `root_abs`
    // and `rel_key`'s `strip_prefix` actually matches instead of silently falling through to the
    // absolute path.
    let mut unresolved = Vec::new();
    for (src, raw_target) in links {
        let (label, rel) = split_doc_key(src, roots);
        let r = roots.iter().find(|r| r.label == label).unwrap_or(&roots[0]);
        let root_abs = abs_root(&r.path);
        let src_dir = Path::new(rel).parent().unwrap_or_else(|| Path::new(""));
        let dst = canonicalize_stripped(&r.path.join(src_dir).join(raw_target))
            .map(|canon| doc_key(label, &root_abs, &canon))
            .filter(|d| files.contains_key(d));
        match dst {
            Some(dst) if &dst != src && matches!(graph.get_node(&dst), Ok(Some(_))) => {
                let sig = files.get(src).copied().unwrap_or(FileSig {
                    mtime_secs: 0,
                    size: 0,
                });
                let _ = crate::graph::build::link_reference(graph, src, &dst, sig);
            }
            _ => unresolved.push((src.clone(), raw_target.clone())),
        }
    }
    unresolved
}

/// Back-compat delegate: single empty-label root, `state_base = dir`. Assumes `index.lock` is
/// already held by the caller (see `index_dir_at_locked`).
pub fn index_dir_locked(dir: &Path, force: bool) -> anyhow::Result<IndexStats> {
    index_dir_at_locked(
        &[crate::root::Root {
            label: String::new(),
            path: dir.to_path_buf(),
        }],
        dir,
        force,
    )
}

/// Body of `index_dir_at` assuming `index.lock` is already held by the caller. Records
/// `.glossa/dirsig` = the current directory signature on every return, so a reader's
/// `freshen_blocking` can tell the index reflects the current tree even when the delta was empty
/// (e.g. a temp file came and went).
pub fn index_dir_at_locked(
    roots: &[crate::root::Root],
    state_base: &Path,
    force: bool,
) -> anyhow::Result<IndexStats> {
    // force = true rebuilds the DOCUMENT-DERIVED layer from scratch: clear the tantivy index and
    // the auto-* graph (structural + lexical) so stale entries (deleted files, or docs previously
    // indexed under a different path form) cannot linger. The agent/curated reasoning graph in
    // graph.sqlite is PRESERVED — a reindex must not destroy hand/agent-built knowledge.
    let mut manifest = if force {
        Manifest::default()
    } else {
        Manifest::load(state_base)
    };
    let schema_migrate = manifest.index_schema_version < INDEX_SCHEMA_VERSION;
    if force {
        let _ = std::fs::remove_dir_all(state_base.join(".glossa").join("index"));
    } else if schema_migrate {
        let _ = std::fs::remove_dir_all(state_base.join(".glossa").join("index"));
        manifest.files.clear();
        manifest.notes.clear();
    }
    let idx = DocIndex::open_or_create_at(roots, state_base)?;
    let graph = crate::graph::store::GraphStore::open(state_base)?;
    if force {
        graph.delete_auto()?;
    }
    let df_path = crate::gate::df::DfTable::sidecar_path(&state_base.join(".glossa"));
    // The grounding DF sidecar is (re)built ONLY on the full-coverage `--force` rebuild, which
    // re-extracts EVERY document (delta.changed = all files, since the manifest was reset above) so
    // the table accumulates over the whole corpus. Incremental passes leave `df = None` and never
    // touch the sidecar: DfTable has no per-source decrement, so merging only the changed docs would
    // double-count edits and never subtract removals, inflating df/n_chunks over uptime. A stale
    // sidecar is corrected by `kb index --force` (see the note in that command's help).
    let mut df = if force {
        Some(crate::gate::df::DfTable::new())
    } else {
        None
    };

    // The delta drives everything below in a single tree walk (over ALL roots): the hot-path gate
    // skips all work when nothing changed, the walk-free indexing loop re-extracts just
    // `delta.changed`, and the notes pass reuses the same `notes_changed`/`notes_removed`. Under
    // `force`/schema-migrate the manifest was reset, so every file/note shows up as changed and is
    // re-added.
    let mut delta = scan_delta_at(roots, state_base, &manifest)?;
    if !force
        && delta.changed.is_empty()
        && delta.removed.is_empty()
        && delta.notes_changed.is_empty()
        && delta.notes_removed.is_empty()
        && delta.empty_mount_roots.is_empty()
    {
        write_dirsig(
            state_base,
            &dir_mtime_map_at(roots, state_base).unwrap_or_default(),
        );
        return Ok(IndexStats {
            added: 0,
            removed: 0,
            unchanged: delta.next.files.len() + delta.next.notes.len(),
            ..Default::default()
        });
    }
    // An empty-mount hold (Task 8) must NOT take the fast no-op path above: that path writes the
    // dirsig straight from the CURRENT (empty) dir-mtime map, which would permanently mask the held
    // root from ever re-triggering a rescan once the share comes back. Falling through to the full
    // pass below still finds nothing to (re)extract, but its dirsig-advance step calls
    // `hold_back_empty_mount_roots`, which reverts the held root's dir keys to their OLD `stored`
    // values so the next freshen keeps re-diffing (and re-attempting) it instead of settling.

    let mut writer = with_writer_retry(|| idx.index.writer(50_000_000))?;
    let mut stats = IndexStats::default();
    let mut next = Manifest::default();

    let mut links: Vec<(String, String)> = Vec::new();
    // Files we actually (re)indexed this pass — re-stat'd at the end so a file that changed while
    // we were indexing it holds back its dir's dirsig entry (see `unsettled_dirs_at`).
    let mut indexed: Vec<(String, FileSig)> = Vec::new();
    eprintln!("indexing files under {} root(s)...", idx.roots.len());
    // Drive indexing off the delta we already computed — do NOT walk the tree again just to re-stat
    // every file. `delta.next.files` is the full current file set (with signatures) that
    // `scan_delta_at` produced in a single walk, so it becomes `next.files` directly; `delta.changed`
    // is exactly the subset needing (re)extraction (under `force`/schema-migrate the manifest was
    // reset, so every file is "changed" and re-extracted — same as before). This removes one of the
    // redundant full walks per `kb index`/`ensure_fresh` pass (the "reindex is slower than early
    // versions" report).
    next.files = std::mem::take(&mut delta.next.files);
    // Dirs held back from this pass's dirsig advance for a reason OTHER than "the indexed file's
    // sig moved under us" (that's `unsettled_dirs_at`, computed after the loop from `indexed`):
    // a stat that failed transiently this scan (Task 4), or a read that failed transiently even
    // after retrying (below) — either way the file wasn't settled at a known-good sig this pass,
    // so its dir must be re-diffed next freshen rather than assumed fresh.
    let mut unsettled_extra: std::collections::HashSet<String> = std::collections::HashSet::new();
    for key in &delta.stat_failed {
        unsettled_extra.insert(parent_dir_key_at(key, &idx.roots));
    }
    let retries = read_retries();
    let backoff = read_retry_backoff_ms();
    for doc_key_str in &delta.changed {
        let Some(&sig) = next.files.get(doc_key_str) else {
            continue;
        };
        let Some(abs) = delta.abs_paths.get(doc_key_str) else {
            continue;
        };
        eprintln!("  + {doc_key_str}");
        let res = with_read_retry(retries, backoff, || {
            index_file_into(
                &idx,
                &graph,
                &writer,
                doc_key_str,
                abs,
                &mut links,
                df.as_mut(),
            )
        });
        match res {
            Ok(_) => {
                indexed.push((doc_key_str.clone(), sig));
                stats.added += 1;
            }
            Err(e) if is_transient(&e) => {
                // Transient even after exhausting retries: do NOT persist the new sig. Revert to
                // the OLD sig if the file was in the prior manifest (preserves the prior doc +
                // guarantees a "changed" verdict next pass); if it's brand-new, remove it entirely
                // (stays "new" next pass). Either way the doc is not erased and the file is
                // retried — mirrors `scan_delta_at`'s transient-stat handling above.
                match manifest.files.get(doc_key_str) {
                    Some(&old) => {
                        next.files.insert(doc_key_str.clone(), old);
                    }
                    None => {
                        next.files.remove(doc_key_str);
                    }
                }
                unsettled_extra.insert(parent_dir_key_at(doc_key_str, &idx.roots));
                stats.transient_failures += 1;
                eprintln!("retry exhausted {doc_key_str}: {e}");
                tracing::warn!(path = %doc_key_str, error = %e, "transient read failure; will retry next pass");
            }
            Err(e) => {
                // Permanent (bad CFB/PDF, InvalidData, PermissionDenied, NotFound): a single
                // unreadable/corrupt file must NOT abort the whole index — log and skip it. The
                // current sig is already in `next.files`, so a later pass treats it as unchanged
                // and does not retry it every time (matches today's bad-.doc skip).
                eprintln!("skip {}: {e}", abs.display());
                stats.permanent_skips += 1;
                stats.errors.push((doc_key_str.clone(), e.to_string()));
                tracing::warn!(path = %doc_key_str, error = %e, "permanent read failure; skipping (not retried)");
            }
        }
    }
    stats.unchanged = next
        .files
        .len()
        .saturating_sub(stats.added + stats.errors.len() + stats.transient_failures);

    for old_path in manifest.files.keys() {
        if !next.files.contains_key(old_path) {
            writer.delete_term(tantivy::Term::from_field_text(
                idx.fields.path,
                old_path.as_str(),
            ));
            graph.delete_auto_by_source(old_path)?;
            // Also drop auto edges OTHER docs authored pointing AT this now-removed doc (e.g. a
            // REFERENCES edge whose source_path is the referencing doc, not this one), so a
            // deleted document doesn't leave a dangling edge behind (mirrors reindex_dirs_locked).
            graph.delete_auto_by_target(old_path)?;
            stats.removed += 1;
        }
    }
    // Cross-document REFERENCES: resolve collected link targets against indexed documents.
    // Links carried over from the prior manifest (sources this pass left `unchanged`, so they were
    // never reindexed and never re-collected into `links`) are retried too, so a dangling link gets
    // picked up the moment its target appears even if the source itself never changes again. Drop
    // carry-overs whose source no longer exists so a deleted doc can't leave a phantom edge.
    let fresh_srcs: std::collections::HashSet<String> =
        links.iter().map(|(src, _)| src.clone()).collect();
    for (src, target) in &manifest.unresolved_links {
        if next.files.contains_key(src) && !fresh_srcs.contains(src) {
            links.push((src.clone(), target.clone()));
        }
    }
    // Links that still don't resolve (target not an indexed doc yet) ride along in the manifest so
    // a later pass that adds the target can pick the edge back up.
    next.unresolved_links = resolve_reference_links(&idx.roots, &graph, &next.files, &links);
    // Notebook notes: index every changed note as a single `"note"` chunk (delete-by-path is
    // idempotent, so a note that was also written through in-process re-adds cleanly), and drop
    // removed notes. Notes never create graph nodes (§6 of the spec) — only search chunks.
    #[cfg(feature = "notebook")]
    {
        let notes_root = state_base.join(".glossa").join("notes");
        for rel in &delta.notes_changed {
            let body = match std::fs::read_to_string(notes_root.join(rel)) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("index notes: skip {rel}: {e}");
                    delta.next.notes.remove(rel);
                    continue;
                }
            };
            writer.delete_term(tantivy::Term::from_field_text(idx.fields.path, rel));
            let _ = writer.add_document(doc!(
                idx.fields.body => body.clone(),
                idx.fields.body_trigrams => body.clone(),
                idx.fields.path => rel.clone(),
                idx.fields.location => "note",
                idx.fields.file_type => "note",
                idx.fields.ord => 1u64,
            ));
            stats.added += 1;
        }
        for rel in &delta.notes_removed {
            writer.delete_term(tantivy::Term::from_field_text(idx.fields.path, rel));
            stats.removed += 1;
        }
        next.notes = std::mem::take(&mut delta.next.notes);
    }
    // Commit under the bounded transient-FS retry — the commit writes+renames `meta.json`, the same
    // frame that flakes on Windows (Defender / a racing reader briefly holds it). Idempotent: the
    // writer's pending ops stay staged, so re-committing after a transient failure is safe.
    with_writer_retry(|| writer.commit())?;
    next.index_schema_version = INDEX_SCHEMA_VERSION;
    next.save(state_base)?;
    // Advance dirsig, holding back any dir whose file changed mid-pass so a full index can't poison
    // the dir-mtime gate either (mirrors reindex_dirs_locked).
    let cur = dir_mtime_map_at(roots, state_base).unwrap_or_default();
    let stored = read_dirsig(state_base).unwrap_or_default();
    let mut unsettled = unsettled_dirs_at(&idx.roots, &indexed, &delta.abs_paths);
    unsettled.extend(unsettled_extra); // transient read + transient stat dirs hold back their dirsig
    hold_back_empty_mount_roots(
        &idx.roots,
        &delta.empty_mount_roots,
        &cur,
        &stored,
        &mut unsettled,
    );
    write_dirsig(state_base, &settled_dirsig(&cur, &stored, &unsettled));
    // Sidecar serialization is a distinct, potentially slow phase on a large vocabulary — flag it
    // explicitly rather than let it pass silently inside the (already-finished) index loop above.
    // Only the full `--force` rebuild produced a DF table (`Some`); incremental passes leave the
    // existing sidecar untouched (see the `df` binding above).
    if let Some(df) = &df {
        eprintln!(
            "writing DF sidecar: {} tokens over {} chunks...",
            df.len(),
            df.n_chunks
        );
        df.save(&df_path)?;
        eprintln!("DF sidecar written ({} tokens)", df.len());
    }
    stats.empty_mount_holds = delta.empty_mount_roots.len();
    Ok(stats)
}

/// Reindex ONE corpus document, assuming the caller already holds `index.lock`. Drops the file's old
/// chunks + auto-graph-by-source and rebuilds them, resolves its outgoing references against the
/// current document set, commits, and records the new signature in the manifest. `None` if the file
/// is gone/unreadable (manifest is left unchanged so a later pass can drop it).
pub fn index_one_file_locked(dir: &Path, rel: &str) -> anyhow::Result<Option<FileSig>> {
    index_one_file_locked_at(
        &[crate::root::Root {
            label: String::new(),
            path: dir.to_path_buf(),
        }],
        dir,
        rel,
    )
}

/// Multi-root primitive behind `index_one_file_locked`: reindex ONE document, given by its stored
/// doc key `rel` (bare relpath for the back-compat single empty-label root, `<label>/<relpath>`
/// otherwise), resolving it to a real file via `doc_file_in(roots, rel)` — so a key addressing a
/// SECONDARY root reindexes from that root's disk, not `roots[0]`. State (index/graph/manifest)
/// lives under `state_base`, exactly like `index_dir_at`. Used by `kb index --file` and the MCP
/// `read`/`index` tools' lazy single-file reindex.
pub fn index_one_file_locked_at(
    roots: &[crate::root::Root],
    state_base: &Path,
    rel: &str,
) -> anyhow::Result<Option<FileSig>> {
    let idx = DocIndex::open_or_create_at(roots, state_base)?;
    let graph = crate::graph::store::GraphStore::open(state_base)?;
    let abs = idx.doc_file(rel);
    let mut writer = with_writer_retry(|| idx.index.writer(50_000_000))?;
    let mut links: Vec<(String, String)> = Vec::new();
    // Single-file reindex is an incremental path: it does NOT touch the grounding DF sidecar (no
    // per-source decrement — see index_file_into's `df` param). Only `kb index --force` rebuilds it.
    let sig = match index_file_into(&idx, &graph, &writer, rel, &abs, &mut links, None)? {
        Some(s) => s,
        None => return Ok(None),
    };
    // Resolve this file's outgoing references against the current document set, via the shared
    // O(links) helper (mirrors index_dir_locked / reindex_dirs_locked — this used to be its own
    // inline by_canon-over-ALL-manifest.files block, the same O(files) shape Finding 1 fixed
    // elsewhere; a single-file reindex must not pay an all-corpus filesystem stat either).
    // `links` only ever carries entries with src == rel (index_file_into only reindexed that one
    // file), so seed the lookup map with THIS file's freshly-computed post-edit `sig` before
    // resolving — otherwise the helper would read `rel`'s stale pre-edit (or altogether missing)
    // signature out of the manifest as it stood before this call, and stamp the REFERENCES edge's
    // provenance with that instead of the edit that just happened.
    let mut manifest = Manifest::load(state_base);
    manifest.files.insert(rel.to_string(), sig);
    let _ = resolve_reference_links(&idx.roots, &graph, &manifest.files, &links);
    // Bounded transient-FS retry around the `meta.json`-writing commit (see index_dir_locked).
    with_writer_retry(|| writer.commit())?;
    let mut m = Manifest::load(state_base);
    m.files.insert(rel.to_string(), sig);
    m.save(state_base)?;
    Ok(Some(sig))
}

/// The `c:`-prefixed corpus dir key (as produced by `dir_mtime_map`) containing manifest file key
/// `rel_file`: `"c:"` for a root file, else `"c:{parent}"`. Manifest file keys (`rel_key`) use the
/// OS-native separator (`\` on Windows), but `dir_mtime_map`'s dir keys are always forward-slash
/// (`collect_dir_mtimes` normalizes them) — normalize here first so the two key spaces compare
/// equal on every platform.
fn parent_dir_key(rel_file: &str) -> String {
    let normalized = rel_file.replace('\\', "/");
    match normalized.rfind('/') {
        Some(i) => format!("c:{}", &normalized[..i]),
        None => "c:".to_string(),
    }
}

/// Wall-clock seconds since the epoch, for freshness windows. On a clock error it returns 0, which
/// (via the `now >= mtime` guard at the call site) means "hold nothing" — the safe default.
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A file (re)indexed within this many seconds of a freshen is treated as possibly still mid-copy,
/// so its directory is held unsettled for one more pass (see the call in `reindex_dirs_locked`). Old
/// corpus files sit far outside the window, so a normal freshen still settles immediately.
///
/// Sized to comfortably exceed two things: the finalize latency of a real copy (a large file over a
/// network share can take several seconds to land its full content after first appearing) AND the
/// wall time of the freshen pass itself. The window is measured from the file's mtime to `now` at
/// the END of the pass, both floored to whole seconds — so if the window is only a hair longer than
/// a slow tantivy commit, a file written just before the pass can already read as "aged out" by the
/// time we check, and the mid-copy hold silently fails to fire. That is exactly what flaked
/// `freshen_gate_misses_content_finalized_after_a_settled_index` on the loaded CI windows runner. A
/// generous window costs only a few extra dir re-stats for files touched in the last few seconds
/// (freshens are throttled), so steady state stays cheap.
const FRESH_WINDOW_SECS: u64 = 10;

/// Given files just indexed under `root` with the signature we recorded for each, re-stat every one
/// and return the set of `c:`-prefixed parent DIR KEYS (as produced by `dir_mtime_map`) whose
/// content is NOT settled: the file's current on-disk signature no longer matches what we indexed
/// (it changed while we were indexing it) or it can no longer be stat'd. Such a directory must not
/// have its `.glossa/dirsig` entry advanced, or the change would be lost — a later content write
/// does not re-bump the directory mtime, so the dir-mtime gate would never re-trigger. O(indexed
/// files), one stat each.
//
// Test-only single-root reference implementation: production uses the multi-root
// `unsettled_dirs_at` (see above); the plain single-root form is exercised only by the unit tests
// below, so it is gated to test builds to avoid a dead-code warning in production builds.
#[cfg(test)]
fn unsettled_dirs(root: &Path, indexed: &[(String, FileSig)]) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    for (rel, indexed_sig) in indexed {
        match file_sig(&root.join(rel)) {
            Ok(cur) if &cur == indexed_sig => {} // settled: on disk matches what we indexed
            _ => {
                out.insert(parent_dir_key(rel)); // changed under us, or vanished → hold back
            }
        }
    }
    out
}

/// `"c:{label}:{parent}"` for a labeled key, or plain `"c:{parent}"` for the back-compat bare
/// (empty-label) key — matches `dir_mtime_map_at`'s per-root prefix, so a multi-root dirsig diff
/// and this hold-back set speak the same key space.
fn parent_dir_key_at(doc_key_str: &str, roots: &[crate::root::Root]) -> String {
    let (label, rel) = split_doc_key(doc_key_str, roots);
    let base = parent_dir_key(rel); // "c:" or "c:{parent}"
    if label.is_empty() {
        base
    } else {
        format!("c:{label}:{}", &base[2..])
    }
}

/// Inverse of `parent_dir_key_at`: resolve a `c:`-prefixed corpus dir key (as produced by
/// `dir_mtime_map_at`/`diff_corpus_dirs`) back to the directory's absolute path and its root's label.
/// Returns `None` for a key that doesn't parse against `roots` (defensive — the caller falls back to
/// a full walk rather than silently skip a directory it can't resolve).
fn resolve_dir_key(key: &str, roots: &[crate::root::Root]) -> Option<(PathBuf, String)> {
    let rest = key.strip_prefix("c:")?;
    if let Some((label, rel)) = rest.split_once(':') {
        if let Some(r) = roots.iter().find(|r| r.label == label) {
            return Some((abs_root(&r.path).join(rel), label.to_string()));
        }
        return None; // labeled form but no matching root — don't guess
    }
    let r = roots.iter().find(|r| r.label.is_empty())?;
    Some((abs_root(&r.path).join(rest), String::new()))
}

/// Depth-1 scan of the immediate files in one directory (subdirectories are tracked by their OWN
/// `dir_mtime_map_at` entry and, if they changed, appear as their own key in `changed`/`added` — so a
/// depth-1 listing per key is sufficient). Uses `ignore::WalkBuilder` capped at `max_depth(1)` rather
/// than a bare `std::fs::read_dir` — with the SAME `standard_filters`/`require_git(false)`/junk-file
/// filtering `walk::walk_files` (and `collect_dir_mtimes`) use for the full-corpus walk, so a
/// gitignored or OS/editor-junk file in a re-scanned dir is excluded here exactly as it would be by a
/// full walk (a bare `read_dir` would silently index it — a real regression this closed). Applies the
/// SAME per-file classification `scan_delta_at` does (changed vs. unchanged, transient stat →
/// `stat_failed` + carried-forward sig) into the shared `Delta` accumulator. Reuses
/// `file_sig`/`is_transient`/`doc_key` — no new classification logic.
fn scan_dir_delta_into(
    dir_abs: &Path,
    label: &str,
    root_abs: &Path,
    manifest: &Manifest,
    d: &mut Delta,
) -> anyhow::Result<()> {
    #[cfg(test)]
    rescan_probe::record(dir_abs);
    let mut wb = WalkBuilder::new(dir_abs);
    wb.standard_filters(true);
    wb.require_git(false);
    wb.max_depth(Some(1));
    wb.filter_entry(|e| e.file_name() != ".glossa" && !crate::walk::is_junk_file(e.file_name()));
    for result in wb.build() {
        let entry = match result {
            Ok(e) => e,
            Err(_) => continue, // vanished/unreadable dir or entry — falls out of next.files naturally
        };
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.path();
        let key = doc_key(label, root_abs, path);
        match file_sig(path) {
            Ok(sig) => {
                if manifest.changed(&key, sig) {
                    d.changed.push(key.clone());
                }
                d.next.files.insert(key.clone(), sig);
                d.abs_paths.insert(key, path.to_path_buf());
            }
            Err(e) if is_transient(&e) => {
                if let Some(&old) = manifest.files.get(&key) {
                    d.next.files.insert(key.clone(), old);
                    d.abs_paths.insert(key.clone(), path.to_path_buf());
                }
                d.stat_failed.push(key);
            }
            Err(_) => {} // permanent stat error (NotFound) — falls out of next.files, reads as removed
        }
    }
    Ok(())
}

/// Test-only spy recording every directory `scan_dir_delta_into` actually walked this pass — the
/// network-scale proof that a scoped rescan touches ONLY the changed/added dirs, not the whole
/// corpus. `take()` drains it so consecutive freshen passes in one test don't accumulate stale entries.
#[cfg(test)]
pub(crate) mod rescan_probe {
    use std::cell::RefCell;
    use std::path::{Path, PathBuf};
    thread_local! { static WALKED: RefCell<Vec<PathBuf>> = const { RefCell::new(Vec::new()) }; }
    pub fn record(dir: &Path) {
        WALKED.with(|w| w.borrow_mut().push(dir.to_path_buf()));
    }
    pub fn take() -> Vec<PathBuf> {
        WALKED.with(|w| std::mem::take(&mut *w.borrow_mut()))
    }
}

/// Scoped counterpart to `scan_delta_at`: given the dir-key diff the caller already computed
/// (`changed`/`added`/`removed`, from `diff_corpus_dirs`), re-stat ONLY those corpus dirs — via
/// `resolve_dir_key` + `scan_dir_delta_into` — and carry every other manifest file forward untouched,
/// instead of re-walking every root. Restores the pre-Plan-A scoped best case (see
/// `reindex_dirs_at_locked`'s doc) while keeping multi-root correctness. Falls back to a full
/// `scan_delta_at` walk in two SAFE cases: the diff is itself empty (first run / migration, where
/// `stored` was empty and every corpus dir already shows up in `added` — scoping would just re-walk
/// everything anyway) or a dir key fails to resolve against `roots` (a stale label, a malformed key)
/// — a scoping bug must never silently under-scan.
fn scan_scoped_delta_at(
    roots: &[crate::root::Root],
    state_base: &Path,
    manifest: &Manifest,
    changed: &[String],
    added: &[String],
    removed: &[String],
    deadline: Option<std::time::Instant>,
) -> anyhow::Result<Delta> {
    if changed.is_empty() && added.is_empty() && removed.is_empty() {
        return scan_delta_at(roots, state_base, manifest);
    }
    let mut d = Delta::default();
    // Dir keys actually walked this pass (a subset of `changed`+`added` when `deadline` cuts the
    // loop short) — drives both the carry-forward decision below AND `d.deadline_held_dirs`. Task
    // 10 (D1): the check sits BETWEEN iterations (before resolving/scanning the next dir), never
    // mid-dir, so a dir is either fully scanned or not touched at all this pass.
    let mut scanned_keys: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for key in changed.iter().chain(added.iter()) {
        if let Some(dl) = deadline {
            if std::time::Instant::now() >= dl {
                break; // wall budget spent — serve-stale: leave the rest for the next freshen
            }
        }
        #[cfg(test)]
        if deadline_fault::should_break(scanned_keys.len() as u32) {
            break; // test seam: deterministic stand-in for a deadline expiring mid-loop
        }
        let Some((dir_abs, label)) = resolve_dir_key(key, roots) else {
            return scan_delta_at(roots, state_base, manifest); // unresolvable key — full walk, for safety
        };
        let root_abs = roots
            .iter()
            .find(|r| r.label == label)
            .map(|r| abs_root(&r.path))
            .unwrap_or_else(|| abs_root(&roots[0].path));
        scan_dir_delta_into(&dir_abs, &label, &root_abs, manifest, &mut d)?;
        scanned_keys.insert(key.as_str());
    }
    for k in changed.iter().chain(added.iter()) {
        if !scanned_keys.contains(k.as_str()) {
            d.deadline_held_dirs.push(k.clone());
        }
    }
    // Fix round 2: root labels ACTUALLY deadline-held this pass — derived from the genuinely
    // unscanned `changed`/`added` keys above, BEFORE `removed` gets folded into
    // `d.deadline_held_dirs` below. A root whose label is NOT in this set had every one of its
    // own changed/added dirs actually, fully scanned this pass, so a fresh "came back empty"
    // result for THAT root is trustworthy on its own merits — regardless of whether some OTHER
    // root in a multi-root corpus was held back. Round 1 skipped the empty-mount guard for EVERY
    // root whenever ANY root was truncated, which reopened the exact Task 8 hole for a root that
    // this pass fully, successfully scanned and found suspiciously empty (e.g. one root
    // empty-mounts while a different, slower/lock-contended root is what actually burns the
    // deadline): the guard never ran for the first root, so its docs were dropped for real. The
    // guard below is now evaluated per-root using this set instead of a single pass-wide flag.
    let held_labels: std::collections::HashSet<String> = d
        .deadline_held_dirs
        .iter()
        .filter_map(|k| resolve_dir_key(k, roots).map(|(_, label)| label))
        .collect();
    // Data-safety fix: a pass the deadline cut short must never let an UNCONFIRMED dir-vanish
    // inference turn into a doc removal. `removed` was classified by the CALLER from a plain
    // dirsig-vs-mtime diff taken BEFORE this walk ever ran — it names a dir key that disappeared
    // from the mtime map, not one this pass actually listed. Trusting it below is exactly how the
    // empty-mount guard a few lines down gets defeated: a transiently-unmounted root's subdirs
    // land in `removed` while the root's OWN dir key sits unscanned in `deadline_held_dirs` (the
    // very peer-lock contention that burns the freshen deadline can also be why the mtime diff
    // looks like a mass dir-vanish) — the root's directly-owned files then survive untouched
    // (their dir was never scanned, never `removed`), so the guard's `survivors == 0` check never
    // fires, and the `removed`-list drop below would otherwise silently wipe every doc under the
    // vanished subdirectory. Once ANY dir was held back by the deadline this pass, hold `removed`
    // dirs back too (for the NEXT, complete pass to confirm) instead of acting on them now.
    let truncated = !d.deadline_held_dirs.is_empty();
    if truncated {
        d.deadline_held_dirs.extend(removed.iter().cloned());
    }
    // Carry forward every manifest file NOT under a dir this pass actually scanned, untouched — the
    // scoped walk above only re-stats the dirs it got to (all of `changed`/`added`, unless a
    // deadline cut it short), so the rest of the corpus (the vast majority, at scale, or every dir
    // a deadline held back) keeps its previously recorded signature. A file whose parent dir WAS
    // scanned but didn't turn up in `d.next.files` is genuinely gone (the depth-1 walk just proved
    // it), not merely "unknown yet" — it must NOT be carried forward, or a file removed from a
    // still-existing (`changed`, not `removed`) dir would incorrectly survive. A dir in `removed`
    // is a DIFFERENT case (see above): only trust it as a confirmed drop when this pass wasn't
    // itself deadline-truncated.
    let scanned = scanned_keys;
    for (k, sig) in &manifest.files {
        if d.next.files.contains_key(k) {
            continue; // already (re)scanned above
        }
        let pk = parent_dir_key_at(k, roots);
        if scanned.contains(pk.as_str()) {
            continue; // its dir was actually scanned this pass (and came back without it) — drop it
        }
        if !truncated && removed.contains(&pk) {
            continue; // dir confirmed gone (this pass wasn't truncated) — drop it
        }
        d.next.files.insert(k.clone(), *sig);
    }
    // Task 8, fix round 1 / fix round 2: root-level empty-mount guard for the SCOPED path,
    // evaluated PER ROOT. Skipped only for a root whose OWN dirs were deadline-held this pass
    // (`held_labels`): nothing was dropped for THAT root based on an unconfirmed inference, so
    // there is nothing here for its guard to catch, and evaluating it against that root's
    // partial view could itself misfire. Every OTHER root — fully, actually scanned this pass —
    // still gets the guard regardless of whether some unrelated root was held back; skipping it
    // pass-wide (fix round 1's bug) would let a different root's genuine empty-mount slip through
    // whenever ANY root in the same multi-root corpus happened to be deadline-held. A transient
    // mounted-but-empty share does NOT generally show up here as "the diff was empty" (the fallback
    // above only fires then) — a share that goes empty while still mounted bumps the ROOT's OWN
    // `c:`/`c:{label}:` dir key into `changed` (its listing lost entries) while every subdir under it
    // lands in `removed`, so the scoped walk above and the carry-forward loop just ran and correctly
    // (from their narrower, per-dir view) dropped every file under that root. The only signal that
    // catches this at the ROOT level — as opposed to "some dirs churned" — is: did this pass end up
    // removing EVERY manifest doc previously known under a given label? A PARTIAL removal (some of
    // the root's docs still present in `d.next.files`) is a legitimate, ordinary deletion and must
    // drop normally; only a COMPLETE wipe of a previously-populated root, with the root directory
    // still physically present (mount up, just empty — a genuinely vanished/renamed root path is a
    // different failure mode and is not held), is treated as suspicious.
    for r in roots {
        if held_labels.contains(&r.label) {
            continue; // this root's own dirs were deadline-held this pass — not evaluated yet
        }
        let had_prior = manifest.files.keys().any(|k| {
            let (label, _) = split_doc_key(k, roots);
            label == r.label
        });
        if !had_prior {
            continue;
        }
        let survivors = manifest
            .files
            .keys()
            .filter(|k| {
                let (label, _) = split_doc_key(k, roots);
                label == r.label
            })
            .filter(|k| d.next.files.contains_key(*k))
            .count();
        if survivors == 0 && abs_root(&r.path).is_dir() {
            for (k, sig) in &manifest.files {
                let (label, _) = split_doc_key(k, roots);
                if label == r.label {
                    d.next.files.insert(k.clone(), *sig);
                }
            }
            d.empty_mount_roots.push(r.label.clone());
            tracing::warn!(root = %r.label, "scoped rescan would remove every doc under a previously-populated root; holding stale index (possible unmounted network share)");
        }
    }
    d.removed = manifest
        .files
        .keys()
        .filter(|k| !d.next.files.contains_key(*k))
        .cloned()
        .collect();
    // Notes aren't keyed into the corpus (`c:`) dir-diff at all (`diff_corpus_dirs` only classifies
    // `c:` keys), so the scoped corpus walk above never touches them — keep the notes half of
    // `scan_delta_at` unconditional here too (cheap: it's a single small `.glossa/notes` walk, not
    // the corpus-scale cost this function exists to avoid), or a note edit would go unindexed.
    #[cfg(feature = "notebook")]
    scan_notes_delta(state_base, manifest, &mut d)?;
    Ok(d)
}

/// Multi-root form of `unsettled_dirs`: each indexed key's abs path comes from the scan's
/// `abs_paths` map (a single `root.join(key)` is wrong once >1 root is in play, since the key may
/// carry a DIFFERENT root's label) rather than being re-derived from a single root.
fn unsettled_dirs_at(
    roots: &[crate::root::Root],
    indexed: &[(String, FileSig)],
    abs_paths: &BTreeMap<String, PathBuf>,
) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    for (key, indexed_sig) in indexed {
        let cur = abs_paths.get(key).and_then(|p| file_sig(p).ok());
        match cur {
            Some(c) if &c == indexed_sig => {} // settled: on disk matches what we indexed
            _ => {
                out.insert(parent_dir_key_at(key, roots)); // changed under us, or vanished → hold back
            }
        }
    }
    out
}

/// Build the dir-signature map to persist to `.glossa/dirsig`: start from the freshly observed
/// `cur` map, but for every directory flagged `unsettled` fall back to its previously `stored`
/// value (so the next freshen re-diffs and retries it), or drop the key entirely when it has no
/// prior entry (a newly-added dir → re-classifies as `added` next pass). Keeps the invariant
/// "dirsig[d] == mtime(d) ⟹ every file in d is indexed at its current signature".
fn settled_dirsig(
    cur: &BTreeMap<String, u128>,
    stored: &BTreeMap<String, u128>,
    unsettled: &std::collections::HashSet<String>,
) -> BTreeMap<String, u128> {
    let mut out = cur.clone();
    for d in unsettled {
        match stored.get(d) {
            Some(&old) => {
                out.insert(d.clone(), old); // retry next freshen against the old signature
            }
            None => {
                out.remove(d); // newly-added dir not captured → re-classify as `added`
            }
        }
    }
    out
}

/// Task 8: extend `unsettled` with EVERY corpus dir key belonging to a root flagged in
/// `empty_mount_roots` — the empty-mount guard's walk found nothing under that root this pass, so
/// there are no individual changed/indexed dirs to enumerate (unlike `unsettled_dirs_at`, which
/// hold back dirs it can name). Instead, every `c:`/`c:{label}:`-prefixed key resolving (via
/// `resolve_dir_key`) to a flagged label is held back wholesale, in both `cur` and `stored` (a
/// vanished root may be missing its keys from one map or the other), so the next freshen fully
/// re-attempts the whole root rather than settling on the empty snapshot.
fn hold_back_empty_mount_roots(
    roots: &[crate::root::Root],
    empty_mount_roots: &[String],
    cur: &BTreeMap<String, u128>,
    stored: &BTreeMap<String, u128>,
    unsettled: &mut std::collections::HashSet<String>,
) {
    if empty_mount_roots.is_empty() {
        return;
    }
    for key in cur.keys().chain(stored.keys()) {
        if let Some((_, label)) = resolve_dir_key(key, roots) {
            if empty_mount_roots.contains(&label) {
                unsettled.insert(key.clone());
            }
        }
    }
}

/// Back-compat delegate: single empty-label root, `state_base = dir`. The dir-key diff params
/// (`cur_map`/`changed`/`added`/`removed`/`notes_touched`) that drove the old scoped depth-1
/// per-dir rescan are no longer consulted by `reindex_dirs_at_locked` (see its doc) — kept only so
/// existing callers (the freshen entry points, and their tests) keep compiling unchanged.
pub fn reindex_dirs_locked(
    dir: &Path,
    cur_map: &BTreeMap<String, u128>,
    changed: &[String],
    added: &[String],
    removed: &[String],
    notes_touched: bool,
) -> anyhow::Result<IndexStats> {
    reindex_dirs_at_locked(
        &[crate::root::Root {
            label: String::new(),
            path: dir.to_path_buf(),
        }],
        dir,
        cur_map,
        changed,
        added,
        removed,
        notes_touched,
        None, // kb index-equivalent: unbounded walk
        read_retries(),
    )
}

/// Multi-root reindex, assuming `index.lock` is already held — the primitive behind the live MCP
/// freshen path. Was previously `idx.primary_root()`-only: files added under a SECONDARY root were
/// never picked up during serving, and dir keys for a secondary root were computed wrong. Fixed by
/// mirroring `index_dir_at_locked`'s shape exactly: one accumulated `Delta` over EVERY root, built by
/// `scan_scoped_delta_at`, `doc_key`-based keys throughout, and the `_at` dirsig helpers
/// (`dir_mtime_map_at`/`parent_dir_key_at`/`unsettled_dirs_at`).
///
/// Restores the pre-Plan-A SCOPED depth-1-per-changed-dir rescan (the point of having a distinct
/// `reindex_dirs_locked` at all): `changed`/`added`/`removed` — the caller's dir-key diff against the
/// persisted dirsig — drive `scan_scoped_delta_at` to re-stat only those dirs instead of every root,
/// falling back to a full walk when the diff is empty (first run/migration) or a key won't resolve
/// (defensive). At 10k-50k files over a network, re-stating every root on every awaited freshen would
/// defeat the point of the dir-mtime gate; the scoped rescan keeps the freshen pass sublinear in the
/// common case (a handful of dirs touched) while still converging to full correctness (see
/// `scan_scoped_delta_at`'s doc for the exact fallback rules).
// Scoped-freshen signature mirrors `reindex_dirs_locked` (incl. the reserved `_cur_map` /
// `_notes_touched` slots); bundling into a params struct is churn for no readability gain.
#[allow(clippy::too_many_arguments)]
pub fn reindex_dirs_at_locked(
    roots: &[crate::root::Root],
    state_base: &Path,
    _cur_map: &BTreeMap<String, u128>,
    changed: &[String],
    added: &[String],
    removed: &[String],
    _notes_touched: bool,
    // Task 10 (D1): `None` = kb index-equivalent, unbounded walk (no caller currently passes
    // `None` except the back-compat `reindex_dirs_locked` delegate and direct tests); `Some(_)` =
    // the freshen serve-stale path — the scoped per-dir walk in `scan_scoped_delta_at` stops
    // between dirs once the deadline passes, holding back whatever it didn't get to.
    deadline: Option<std::time::Instant>,
    // `read_retries()` for CLI-equivalent (unbounded) callers, `read_retries().min(1)` for the
    // latency-bounded freshen path — the retry split (D1): full retries would multiply the wall
    // cost a slow network read already threatens the deadline with.
    max_retries: u32,
) -> anyhow::Result<IndexStats> {
    let idx = DocIndex::open_or_create_at(roots, state_base)?;
    let graph = crate::graph::store::GraphStore::open(state_base)?;
    let mut writer = with_writer_retry(|| idx.index.writer(50_000_000))?;
    let manifest = Manifest::load(state_base);
    let mut stats = IndexStats::default();
    // Scoped freshen is an incremental path: it does NOT touch the grounding DF sidecar (no
    // per-source decrement — see index_file_into's `df` param). Only `kb index --force` rebuilds it.
    // Files we actually (re)indexed this pass, with the signature we captured — re-stat'd at the
    // end so a file that changed WHILE we were indexing it holds back its dir's dirsig entry.
    let mut indexed: Vec<(String, FileSig)> = Vec::new();
    let mut links: Vec<(String, String)> = Vec::new();

    // Scoped walk over just the changed/added dirs (falls back to a full walk when unsafe to scope —
    // see `scan_scoped_delta_at`); the rest of this function mirrors `index_dir_at_locked`'s indexing
    // loop exactly regardless of which walk produced `delta`.
    let mut delta = scan_scoped_delta_at(
        roots, state_base, &manifest, changed, added, removed, deadline,
    )?;
    let mut next = Manifest::default();
    next.files = std::mem::take(&mut delta.next.files);
    // Dirs held back from this pass's dirsig advance for a reason OTHER than "the indexed file's
    // sig moved under us" (that's `unsettled_dirs_at`, computed after the loop from `indexed`): a
    // stat that failed transiently this scan, or a read that failed transiently even after
    // retrying (below) — either way the file wasn't settled at a known-good sig this pass, so its
    // dir must be re-diffed next freshen rather than assumed fresh (mirrors `index_dir_at_locked`).
    let mut unsettled_extra: std::collections::HashSet<String> = std::collections::HashSet::new();
    for key in &delta.stat_failed {
        unsettled_extra.insert(parent_dir_key_at(key, &idx.roots));
    }
    // Task 10 (D1) serve-stale: every dir the deadline kept `scan_scoped_delta_at` from reaching
    // must hold back its dirsig too, exactly like a transient stat failure — so the NEXT freshen
    // re-diffs and finishes it, and this pass never partially settles a dir it didn't scan.
    for key in &delta.deadline_held_dirs {
        unsettled_extra.insert(key.clone());
    }
    let backoff = read_retry_backoff_ms();
    for doc_key_str in &delta.changed {
        let Some(&sig) = next.files.get(doc_key_str) else {
            continue;
        };
        let Some(abs) = delta.abs_paths.get(doc_key_str) else {
            continue;
        };
        // Skip (don't abort the freshen on) a single corrupt/unreadable file — mirrors the
        // resilience of the `kb index` walk.
        let res = with_read_retry(max_retries, backoff, || {
            // Freshen never touches the DF sidecar (df: None) — only `kb index --force` rebuilds it.
            index_file_into(&idx, &graph, &writer, doc_key_str, abs, &mut links, None)
        });
        match res {
            Ok(_) => {
                indexed.push((doc_key_str.clone(), sig));
                stats.added += 1;
            }
            Err(e) if is_transient(&e) => {
                // Same revert rule as index_dir_at_locked (Task 5): next.files was ALREADY
                // prefilled with the new sig by scan_delta_at, so on transient we must revert/
                // remove it or the removed-loop and the next pass's diff both mis-read this file.
                match manifest.files.get(doc_key_str) {
                    Some(&old) => {
                        next.files.insert(doc_key_str.clone(), old);
                    }
                    None => {
                        next.files.remove(doc_key_str);
                    }
                }
                unsettled_extra.insert(parent_dir_key_at(doc_key_str, &idx.roots));
                stats.transient_failures += 1;
                tracing::warn!(path = %doc_key_str, error = %e, "transient read failure on freshen; will retry");
            }
            Err(e) => {
                // Permanent (bad CFB/PDF, InvalidData, PermissionDenied, NotFound): the current sig
                // is already in `next.files`, so a later pass treats it as unchanged and does not
                // retry it every time (matches today's bad-.doc skip).
                eprintln!("skip {}: {e}", abs.display());
                stats.permanent_skips += 1;
                stats.errors.push((doc_key_str.clone(), e.to_string()));
                tracing::warn!(path = %doc_key_str, error = %e, "permanent read failure on freshen; skipping (not retried)");
            }
        }
    }
    stats.unchanged = next
        .files
        .len()
        .saturating_sub(stats.added + stats.errors.len() + stats.transient_failures);

    // Drop removed files (clean edges BOTH directions): `delta.removed` is already exactly the
    // manifest keys no longer present in the fresh walk — no bespoke rescan/removed-dir-key set
    // needed once the walk is full rather than scoped.
    for k in &delta.removed {
        writer.delete_term(tantivy::Term::from_field_text(idx.fields.path, k.as_str()));
        graph.delete_auto_by_source(k)?;
        graph.delete_auto_by_target(k)?;
        stats.removed += 1;
    }

    // Notes: `scan_delta_at` already computed `notes_changed`/`notes_removed` unconditionally
    // (like `index_dir_at_locked`'s notes pass), so there is no need for the old `notes_touched`
    // gate — reuse the exact notes-index/drop block `index_dir_at_locked` uses.
    #[cfg(feature = "notebook")]
    {
        let notes_root = state_base.join(".glossa").join("notes");
        for rel in &delta.notes_changed {
            let body = match std::fs::read_to_string(notes_root.join(rel)) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("index notes: skip {rel}: {e}");
                    delta.next.notes.remove(rel);
                    continue;
                }
            };
            writer.delete_term(tantivy::Term::from_field_text(idx.fields.path, rel));
            let _ = writer.add_document(doc!(
                idx.fields.body => body.clone(),
                idx.fields.body_trigrams => body.clone(),
                idx.fields.path => rel.clone(),
                idx.fields.location => "note",
                idx.fields.file_type => "note",
                idx.fields.ord => 1u64,
            ));
            stats.added += 1;
        }
        for rel in &delta.notes_removed {
            writer.delete_term(tantivy::Term::from_field_text(idx.fields.path, rel));
            stats.removed += 1;
        }
        next.notes = std::mem::take(&mut delta.next.notes);
    }

    // References: resolve freshly-collected links plus carried-over `unresolved_links` (dropping
    // any whose source no longer exists — its edges were already dropped above) against the
    // now-current document set; store what's still unresolved back into the manifest.
    let fresh_srcs: std::collections::HashSet<String> =
        links.iter().map(|(src, _)| src.clone()).collect();
    let mut all = links;
    for (src, target) in &manifest.unresolved_links {
        if next.files.contains_key(src) && !fresh_srcs.contains(src) {
            all.push((src.clone(), target.clone()));
        }
    }
    next.unresolved_links = resolve_reference_links(&idx.roots, &graph, &next.files, &all);

    // Bounded transient-FS retry around the `meta.json`-writing commit (see index_dir_at_locked).
    with_writer_retry(|| writer.commit())?;
    // Carry the schema version forward as-is: this is not a schema-migrating path (that's
    // `index_dir_at_locked`'s job), so resetting it to `Manifest::default()`'s value would make
    // the NEXT `kb index` think a migration is still pending and force a full rebuild.
    next.index_schema_version = manifest.index_schema_version;
    next.save(state_base)?;
    // Advance dirsig, but hold back any dir whose file changed under us (still being written): the
    // lock is held, so the on-disk dirsig is still the value we diffed against (`stored`).
    let stored = read_dirsig(state_base).unwrap_or_default();
    let cur = dir_mtime_map_at(roots, state_base).unwrap_or_default();
    let mut unsettled = unsettled_dirs_at(&idx.roots, &indexed, &delta.abs_paths);
    unsettled.extend(unsettled_extra); // spec R-B3: reverted-sig / transient-stat dirs must re-scan
                                       // Also hold back a dir whose file we just indexed was written within the last few seconds: it
                                       // may still be mid-copy. A finalize write landing AFTER this pass does NOT re-bump the dir mtime,
                                       // so without this the dir-mtime gate would never re-open and the finalized content would stay
                                       // invisible to search/grep (regression: `freshen_gate_misses_content_finalized_after_a_settled_
                                       // index`). This is the freshen (MCP) path only — the CLI `scan_delta_at` path already re-stats
                                       // every file. The hold self-clears once the file ages past the window, so steady state stays fast.
    let now = now_secs();
    for (key, sig) in &indexed {
        if now >= sig.mtime_secs && now - sig.mtime_secs < FRESH_WINDOW_SECS {
            unsettled.insert(parent_dir_key_at(key, &idx.roots));
        }
    }
    hold_back_empty_mount_roots(
        &idx.roots,
        &delta.empty_mount_roots,
        &cur,
        &stored,
        &mut unsettled,
    );
    write_dirsig(state_base, &settled_dirsig(&cur, &stored, &unsettled));
    stats.empty_mount_holds = delta.empty_mount_roots.len();
    Ok(stats)
}

/// Classify a dir-map diff into changed/added/removed CORPUS (`c:`) dir keys.
fn diff_corpus_dirs(
    before: &BTreeMap<String, u128>,
    after: &BTreeMap<String, u128>,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let is_c = |k: &String| k.starts_with("c:");
    let changed = after
        .iter()
        .filter(|(k, v)| is_c(k) && before.get(*k).is_some_and(|bv| bv != *v))
        .map(|(k, _)| k.clone())
        .collect();
    let added = after
        .keys()
        .filter(|k| is_c(k) && !before.contains_key(*k))
        .cloned()
        .collect();
    let removed = before
        .keys()
        .filter(|k| is_c(k) && !after.contains_key(*k))
        .cloned()
        .collect();
    (changed, added, removed)
}

/// True iff the `n:`-prefixed (notebook notes) sub-map differs between two dir-mtime maps —
/// the production rule for whether `reindex_dirs_locked`'s notes pass needs to run because a note
/// itself moved (as opposed to Finding 2's owner-removed case, which the caller ORs in separately).
fn notes_submap_changed(before: &BTreeMap<String, u128>, after: &BTreeMap<String, u128>) -> bool {
    let notes_of = |m: &BTreeMap<String, u128>| {
        m.iter()
            .filter(|(k, _)| k.starts_with("n:"))
            .map(|(k, v)| (k.clone(), *v))
            .collect::<BTreeMap<_, _>>()
    };
    notes_of(before) != notes_of(after)
}

/// Back-compat delegate: single empty-label root, `state_base = dir` — byte-identical to the old
/// single-root behavior. See [`freshen_blocking_at`] for the multi-root primitive.
pub fn freshen_blocking(dir: &Path, timeout: std::time::Duration) -> anyhow::Result<IndexStats> {
    freshen_blocking_at(
        &[crate::root::Root {
            label: String::new(),
            path: dir.to_path_buf(),
        }],
        dir,
        timeout,
    )
}

/// Bring the on-disk index up to date with EVERY root, synchronously, without ever hanging — the
/// multi-root primitive behind the live MCP freshen path (`GlossaServer::freshen_now`). Fast path:
/// if `.glossa/dirsig` already equals the current combined dir-map, return immediately. Else take
/// `index.lock` (held under `state_base`) and reindex across all roots via
/// [`reindex_dirs_at_locked`]; if another process holds the lock, poll until either the persisted
/// map reaches what we observed (the peer indexed it) or `timeout` elapses (serve current).
pub fn freshen_blocking_at(
    roots: &[crate::root::Root],
    state_base: &Path,
    timeout: std::time::Duration,
) -> anyhow::Result<IndexStats> {
    let cur = dir_mtime_map_at(roots, state_base)?;
    if read_dirsig(state_base).as_ref() == Some(&cur) {
        return Ok(IndexStats::default());
    }
    // Task 10 (D1): the wall budget for the awaited walk+reindex itself (checked mid-walk inside
    // `scan_scoped_delta_at`'s per-dir loop) — separate from `lock_deadline` below, which only
    // bounds how long we spin waiting for a PEER to release `index.lock`.
    let deadline =
        std::time::Instant::now() + std::time::Duration::from_millis(freshen_deadline_ms());
    let lock_deadline = std::time::Instant::now() + timeout;
    loop {
        if let Some(_guard) = crate::index::lock::try_index_lock(state_base) {
            // We own the lock: compute the dir-map diff against the last-indexed state and
            // reindex only the changed/added/removed corpus dirs (scoped, sublinear pass).
            // On first run / migration `stored` is empty, so every dir is `added` and the
            // scoped pass reindexes everything -- equivalent to a full pass, self-healing.
            let stored = read_dirsig(state_base).unwrap_or_default();
            let (changed, added, removed) = diff_corpus_dirs(&stored, &cur);
            let notes_touched = notes_submap_changed(&stored, &cur);
            return reindex_dirs_at_locked(
                roots,
                state_base,
                &cur,
                &changed,
                &added,
                &removed,
                notes_touched,
                Some(deadline),
                read_retries().min(1), // 0-1 retry on the latency-bounded freshen path (D1 retry split)
            );
        }
        // Another process is indexing. If it already reached our observed state, we are fresh.
        if read_dirsig(state_base).as_ref() == Some(&cur) {
            return Ok(IndexStats::default());
        }
        if std::time::Instant::now() >= lock_deadline {
            return Ok(IndexStats::default());
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

#[cfg(test)]
mod incremental_tests {
    use super::*;
    use std::fs;

    #[test]
    fn index_dir_writes_dirsig_matching_current_tree() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nhello\n").unwrap();
        index_dir(dir.path(), false).unwrap();
        assert_eq!(
            read_dirsig(dir.path()),
            Some(dir_mtime_map(dir.path()).unwrap()),
            "after indexing, the persisted map equals the current directory map"
        );
    }

    #[test]
    fn dirsig_map_roundtrips_and_migrates_from_u64() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nhello\n").unwrap();
        index_dir(dir.path(), false).unwrap();
        // After indexing, dirsig holds the current map.
        assert_eq!(
            read_dirsig(dir.path()),
            Some(dir_mtime_map(dir.path()).unwrap())
        );
        // A legacy u64 dirsig file is treated as "no snapshot" (migration).
        std::fs::write(dir.path().join(".glossa").join("dirsig"), b"12345").unwrap();
        assert_eq!(read_dirsig(dir.path()), None, "legacy u64 dirsig -> None");
    }

    #[test]
    fn index_dir_builds_structural_graph() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# Intro\nhello\n## Body\nworld\n").unwrap();
        index_dir(dir.path(), false).unwrap();
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        assert!(g.node_count().unwrap() >= 2); // Document + at least one Section
        let intro = g.resolve("Intro").unwrap();
        assert!(!intro.is_empty());
    }

    #[test]
    fn index_dir_still_builds_chunks_and_graph_after_refactor() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("a.md"),
            b"# Intro\nhello alpha\n## Body\nworld beta\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("b.md"), b"# B\nsee [a](a.md)\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(idx
            .search("alpha", 10)
            .unwrap()
            .iter()
            .any(|h| h.path.ends_with("a.md")));
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        assert!(
            g.node_count().unwrap() >= 2,
            "Document + Section nodes built"
        );
        // cross-doc reference b.md -> a.md resolved to a REFERENCES edge (REFERENCES links
        // Document node to Document node, not section to section — see link_reference).
        // Asserts the SPECIFIC target, not just non-emptiness: `neighbors` with
        // `edge_types: None` also picks up the unconditional CONTAINS edge from
        // Document("b.md") to its own Section("b.md#1"), which would make a bare
        // `!is_empty()` pass even if reference resolution were broken.
        assert!(
            crate::graph::traverse::neighbors(&g, "b.md", None, 1)
                .unwrap()
                .contains(&"a.md".to_string()),
            "cross-doc reference b.md -> a.md resolved to a REFERENCES edge (not just the self CONTAINS edge)"
        );
    }

    #[test]
    fn index_dir_skips_malformed_pdf_and_continues() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("ok.md"), b"# T\nhello world\n").unwrap();
        std::fs::write(dir.path().join("bad.pdf"), b"%PDF-1.4\nnot a real pdf").unwrap();
        // Must complete (not panic); the md is indexed.
        let stats = index_dir(dir.path(), false).unwrap();
        assert!(stats.added >= 1);
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let hits = idx.search("hello", 10).unwrap();
        assert!(hits.iter().any(|h| h.path.ends_with("ok.md")));
    }

    #[test]
    fn index_dir_skips_corrupt_office_doc_and_continues() {
        // A .doc whose extractor HARD-errors (bad CFB header) must not abort the whole index — the
        // walk-free loop of Fix C once propagated this with `?`, killing the run with no stats. The
        // good doc must still index and stats must be returned. (Malformed PDFs soft-fail, so only a
        // hard-error extractor like office exercises this path.)
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("ok.md"), b"# T\nhello world\n").unwrap();
        std::fs::write(
            dir.path().join("bad.doc"),
            b"this is not a real CFB .doc file",
        )
        .unwrap();
        let stats = index_dir(dir.path(), false).expect("a corrupt .doc must not abort the index");
        assert!(
            stats.added >= 1,
            "the good doc is indexed despite the corrupt one"
        );
        // The failure is collected (not lost): reported to the CLI as an end-of-run error summary.
        assert_eq!(
            stats.errors.len(),
            1,
            "the corrupt .doc is recorded as an error"
        );
        assert!(
            stats.errors[0].0.contains("bad.doc"),
            "error names the offending file"
        );
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(idx
            .search("hello", 10)
            .unwrap()
            .iter()
            .any(|h| h.path.ends_with("ok.md")));
        // The MCP freshen path must be resilient too: add another corrupt doc, freshen, don't abort.
        std::fs::write(dir.path().join("bad2.doc"), b"garbage two").unwrap();
        std::fs::write(dir.path().join("good2.md"), b"# U\nbravo term\n").unwrap();
        freshen_blocking(dir.path(), std::time::Duration::from_secs(3))
            .expect("a corrupt .doc must not abort freshen");
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(idx
            .search("bravo", 10)
            .unwrap()
            .iter()
            .any(|h| h.path.ends_with("good2.md")));
    }

    #[test]
    fn index_dir_builds_sequential_and_hierarchy_edges() {
        use crate::graph::build::section_id;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("a.md"),
            b"# A\nintro\n## B\nbody b\n## C\nbody c\n",
        )
        .unwrap();
        index_dir(dir.path(), true).unwrap();
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        let p = "a.md".to_string(); // canonical key: corpus-root-relative
                                    // Section ids are the 1-based ordinal now (heading "A"/"A > B"/"A > C" stays the
                                    // node label; hierarchy is still built from the heading breadcrumb).
        let a = section_id(&p, "1");
        let ab = section_id(&p, "2");
        let ac = section_id(&p, "3");
        // sequential: A -> A>B -> A>C reachable from A's section via outgoing edges
        let na = crate::graph::traverse::neighbors(&g, &a, None, 1).unwrap();
        assert!(
            na.contains(&ab),
            "A neighbors include next/child A>B: {na:?}"
        );
        // hierarchy: A>B's parent A is reachable
        let nab = crate::graph::traverse::neighbors(&g, &ab, None, 1).unwrap();
        assert!(nab.contains(&a), "A>B neighbors include parent A: {nab:?}");
        assert!(
            nab.contains(&ac),
            "A>B neighbors include next sibling A>C: {nab:?}"
        );
    }

    #[test]
    fn index_dir_builds_cross_document_references() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("a.md"),
            b"# A\nsee [the manual](b.md) and [ext](https://x.com)\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("b.md"), b"# B\ncontent\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        let a = "a.md".to_string(); // canonical keys: corpus-root-relative
        let b = "b.md".to_string();
        let na = crate::graph::traverse::neighbors(&g, &a, None, 1).unwrap();
        assert!(na.contains(&b), "a.md REFERENCES b.md: {na:?}");
        assert!(
            !na.iter().any(|n| n.contains("x.com")),
            "external URL is not a REFERENCES edge: {na:?}"
        );
    }

    #[test]
    fn index_dir_persists_and_resolves_unresolved_links() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nsee [b](b.md)\n").unwrap(); // b.md missing
        index_dir(dir.path(), true).unwrap();
        let m = Manifest::load(dir.path());
        assert!(
            m.unresolved_links
                .iter()
                .any(|(s, t)| s == "a.md" && t == "b.md"),
            "dangling link recorded"
        );
        // Add b.md; a full reindex resolves the edge and clears the unresolved entry.
        std::fs::write(dir.path().join("b.md"), b"# B\ncontent\n").unwrap();
        index_dir(dir.path(), false).unwrap();
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        assert!(
            crate::graph::traverse::neighbors(&g, "a.md", None, 1)
                .unwrap()
                .contains(&"b.md".to_string()),
            "a->b resolved"
        );
        assert!(
            Manifest::load(dir.path()).unresolved_links.is_empty(),
            "unresolved cleared"
        );
    }

    #[test]
    fn reindex_force_purges_removed_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nalphaword\n").unwrap();
        std::fs::write(dir.path().join("b.md"), b"# B\nbravoword\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        // Delete a.md, then force-reindex: the stale doc must be purged (not just left behind).
        std::fs::remove_file(dir.path().join("a.md")).unwrap();
        index_dir(dir.path(), true).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(
            idx.search("alphaword", 10).unwrap().is_empty(),
            "removed file purged on force reindex"
        );
        assert!(
            !idx.search("bravoword", 10).unwrap().is_empty(),
            "kept file still indexed"
        );
    }

    #[test]
    fn reindex_preserves_agent_graph_drops_auto() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nalphaword\n").unwrap();
        index_dir(dir.path(), true).unwrap();

        // An agent-built reasoning node (origin = "agent"), as the enricher/specialist would add.
        {
            let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
            g.put_node(&crate::graph::store::Node {
                id: "sym:test".into(),
                node_type: "Symptom".into(),
                label: "test symptom".into(),
                aliases: vec![],
                prov: crate::graph::store::Provenance {
                    source_path: "a.md".into(),
                    range: None,
                    file_sig: None,
                    origin: "agent".into(),
                    confidence: 0.9,
                    created_at: 1,
                },
            })
            .unwrap();
        }

        // Reindex rebuilds the auto-* structure but must PRESERVE the agent node.
        index_dir(dir.path(), true).unwrap();
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        assert!(
            g.get_node("sym:test").unwrap().is_some(),
            "agent node survives reindex"
        );
        let autos = g
            .all_nodes()
            .unwrap()
            .into_iter()
            .filter(|n| n.prov.origin == "auto-structural")
            .count();
        assert!(autos > 0, "structural (auto) layer was rebuilt");
    }

    #[test]
    fn reindex_picks_up_changes_and_skips_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.md"), "# T\ncontracts delivery\n").unwrap();

        let s1 = index_dir(dir.path(), false).unwrap();
        assert_eq!(s1.added, 1);

        let s2 = index_dir(dir.path(), false).unwrap();
        assert_eq!(s2.unchanged, 1);
        assert_eq!(s2.added, 0);

        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let hits = idx.search("contract", 10).unwrap();
        assert!(hits.iter().any(|h| h.path.ends_with("a.md")));
    }

    #[test]
    fn scan_delta_reports_changed_added_removed() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.md"), "# A\nalpha\n").unwrap();
        fs::write(dir.path().join("b.md"), "# B\nbravo\n").unwrap();
        index_dir(dir.path(), false).unwrap();
        let manifest = Manifest::load(dir.path());

        // clean tree → empty delta (the cheap hot-path gate)
        let d0 = scan_delta(dir.path(), &manifest).unwrap();
        assert!(
            d0.changed.is_empty() && d0.removed.is_empty(),
            "clean: {d0:?}"
        );

        // modify a (size differs), add c, remove b
        fs::write(dir.path().join("a.md"), "# A\nalpha changed longer\n").unwrap();
        fs::write(dir.path().join("c.md"), "# C\ncharlie\n").unwrap();
        fs::remove_file(dir.path().join("b.md")).unwrap();
        let d = scan_delta(dir.path(), &manifest).unwrap();
        assert!(
            d.changed.iter().any(|p| p.ends_with("a.md")),
            "a changed: {:?}",
            d.changed
        );
        assert!(
            d.changed.iter().any(|p| p.ends_with("c.md")),
            "c added: {:?}",
            d.changed
        );
        assert!(
            d.removed.iter().any(|p| p.ends_with("b.md")),
            "b removed: {:?}",
            d.removed
        );
    }

    /// Task-4 regression guard: a transient `stat` failure on a file that IS still present must not
    /// be misread by `scan_delta_at` as a delete. Before the fix, `file_sig`'s `Err` branch dropped
    /// the key from `d.next.files` unconditionally, so the removed-loop (keys in the old manifest but
    /// absent from `d.next.files`) picked it up as a genuine removal. `read_fault::arm_stat` injects
    /// a transient error (`is_transient` classifies it true) on the next `file_sig(abs)` call. Uses
    /// the cross-platform `TimedOut` kind (no errno) rather than an ESTALE errno code: `is_transient`'s
    /// errno table is unix-only (see its doc comment), so an errno-keyed fault would silently fail to
    /// classify as transient on Windows, per `index_file_into_preserves_prior_doc_when_read_fails`'s
    /// same choice for the analogous read-fault test.
    #[test]
    fn scan_delta_at_transient_stat_preserves_doc_and_marks_stat_failed() {
        use crate::root::Root;
        let a = tempfile::tempdir().unwrap();
        std::fs::write(a.path().join("keep.md"), "body").unwrap();
        let state = tempfile::tempdir().unwrap();
        let roots = [Root {
            label: String::new(),
            path: a.path().into(),
        }];
        index_dir_at(&roots, state.path(), true).unwrap(); // seed manifest
        let manifest = Manifest::load(state.path());
        let old = *manifest.files.get("keep.md").unwrap();
        let abs = abs_root(a.path()).join("keep.md"); // same abs form the walk yields
        read_fault::arm_stat(&abs, 1, std::io::ErrorKind::TimedOut, None); // transient, cross-platform
        let d = scan_delta_at(&roots, state.path(), &manifest).unwrap();
        read_fault::clear();
        assert!(d.stat_failed.iter().any(|k| k == "keep.md"));
        assert!(
            !d.removed.iter().any(|k| k == "keep.md"),
            "transient stat must not read as delete"
        );
        assert_eq!(
            d.next.files.get("keep.md"),
            Some(&old),
            "old sig carried forward"
        );
    }

    /// Task 8: a root that was previously populated but this pass's walk succeeds (no io error) and
    /// finds ZERO files must be treated as a possibly-unmounted share, not a corpus-side mass delete.
    /// A DIFFERENT, genuinely empty tempdir stands in for "the same root looks empty this pass"
    /// (simulates a transient unmount/empty-remount without needing an OS-level mount operation) —
    /// `scan_delta_at` never looks at the path's identity, only what the walk finds under it.
    #[test]
    fn scan_delta_at_empty_mount_holds_stale_docs() {
        use crate::root::Root;
        let a = tempfile::tempdir().unwrap();
        std::fs::write(a.path().join("one.md"), "alpha").unwrap();
        std::fs::write(a.path().join("two.md"), "bravo").unwrap();
        let state = tempfile::tempdir().unwrap();
        let roots = [Root {
            label: String::new(),
            path: a.path().into(),
        }];
        index_dir_at(&roots, state.path(), true).unwrap(); // seed manifest with 2 entries
        let manifest = Manifest::load(state.path());
        assert_eq!(manifest.files.len(), 2);

        let empty = tempfile::tempdir().unwrap();
        let empty_roots = [Root {
            label: String::new(),
            path: empty.path().into(),
        }];
        let d = scan_delta_at(&empty_roots, state.path(), &manifest).unwrap();
        assert!(
            d.removed.is_empty(),
            "held back, not mass-removed: {:?}",
            d.removed
        );
        assert_eq!(d.next.files.len(), 2, "both stale sigs carried forward");
        assert_eq!(d.empty_mount_roots, vec![String::new()]);
    }

    /// Task 8: a brand-new (never-populated) manifest against a genuinely empty root is NOT the
    /// empty-mount failure mode — a fresh corpus with nothing indexed yet must not be flagged.
    #[test]
    fn scan_delta_at_first_run_not_flagged_empty_mount() {
        use crate::root::Root;
        let empty = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        let roots = [Root {
            label: String::new(),
            path: empty.path().into(),
        }];
        let manifest = Manifest::default(); // had_prior is false for every root
        let d = scan_delta_at(&roots, state.path(), &manifest).unwrap();
        assert!(
            d.empty_mount_roots.is_empty(),
            "never-populated corpus must not be flagged: {:?}",
            d.empty_mount_roots
        );
    }

    /// Task 8: `--force` is the escape hatch for a genuinely emptied corpus root. After the guard
    /// holds a root's stale docs back (root really did go empty, no `--force`), a forced rebuild
    /// resets the manifest before scanning — `had_prior` is then false, so the empty state is
    /// allowed to stick rather than being held forever.
    #[test]
    fn index_dir_at_force_clears_empty_mount_hold() {
        use crate::root::Root;
        let a = tempfile::tempdir().unwrap();
        std::fs::write(a.path().join("one.md"), "alpha").unwrap();
        std::fs::write(a.path().join("two.md"), "bravo").unwrap();
        let state = tempfile::tempdir().unwrap();
        let roots = [Root {
            label: String::new(),
            path: a.path().into(),
        }];
        index_dir_at(&roots, state.path(), true).unwrap(); // seed manifest with 2 entries

        // The root genuinely empties out (share drops its content / goes empty) — a normal
        // (non-forced) pass must hold the stale docs, not remove them.
        std::fs::remove_file(a.path().join("one.md")).unwrap();
        std::fs::remove_file(a.path().join("two.md")).unwrap();
        let stats = index_dir_at(&roots, state.path(), false).unwrap();
        assert_eq!(stats.removed, 0, "held back, not removed: {stats:?}");
        let manifest = Manifest::load(state.path());
        assert_eq!(manifest.files.len(), 2, "stale docs still held");

        // Now force: the operator has genuinely emptied this root on purpose.
        index_dir_at(&roots, state.path(), true).unwrap();
        let manifest = Manifest::load(state.path());
        assert!(
            manifest.files.is_empty(),
            "force makes the genuinely-empty state stick: {:?}",
            manifest.files
        );
    }

    /// Task 5: a transient READ failure that survives `with_read_retry`'s full attempt budget must
    /// not silently drop the file from the manifest — the sig reverts to the OLD (pre-change) value,
    /// not the freshly-observed one, so the doc stays searchable at its prior content and the next
    /// pass still sees it as "changed" and retries it. Arming `read_retries() + 1` consecutive
    /// failures exhausts the retry budget (1 initial attempt + `read_retries()` retries) so the
    /// error reaching the loop is the final, still-transient one.
    #[test]
    fn index_dir_at_locked_transient_reverts_to_old_sig_and_preserves_doc() {
        use crate::root::Root;
        let dir = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nalpha term\n").unwrap();
        let roots = [Root {
            label: String::new(),
            path: dir.path().into(),
        }];
        index_dir_at(&roots, state.path(), true).unwrap();
        let old = *Manifest::load(state.path()).files.get("a.md").unwrap();

        // A different SIZE (not just mtime) guarantees `manifest.changed` sees this as modified
        // regardless of the mtime's whole-second granularity.
        std::fs::write(dir.path().join("a.md"), b"# A\nalpha term changed\n").unwrap();
        let abs = abs_root(dir.path()).join("a.md");
        read_fault::arm_read(&abs, read_retries() + 1, std::io::ErrorKind::TimedOut, None);

        let stats = index_dir_at(&roots, state.path(), false).unwrap();
        read_fault::clear();

        assert_eq!(
            stats.transient_failures, 1,
            "the retry-exhausted read counts as a transient failure"
        );
        assert_eq!(stats.removed, 0, "the file is not treated as removed");
        let manifest = Manifest::load(state.path());
        assert_eq!(
            manifest.files.get("a.md"),
            Some(&old),
            "reverted to the OLD sig, not the freshly-observed (changed) one"
        );

        let idx = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert!(
            idx.search("alpha", 10)
                .unwrap()
                .iter()
                .any(|h| h.path.ends_with("a.md")),
            "the prior doc must still be searchable"
        );
    }

    /// Task 5: a transient READ failure on a BRAND-NEW file (no entry in the prior manifest) must
    /// leave no sig behind at all — the manifest should read as if the file was never observed, so
    /// the next pass classifies it as `added` (not `changed`) once its read succeeds.
    #[test]
    fn index_dir_at_locked_transient_new_file_absent_from_manifest() {
        use crate::root::Root;
        let dir = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nalpha term\n").unwrap();
        let roots = [Root {
            label: String::new(),
            path: dir.path().into(),
        }];
        index_dir_at(&roots, state.path(), true).unwrap(); // seed a stable manifest

        std::fs::write(dir.path().join("b.md"), b"# B\nbravo term\n").unwrap();
        let abs = abs_root(dir.path()).join("b.md");
        read_fault::arm_read(&abs, read_retries() + 1, std::io::ErrorKind::TimedOut, None);

        let stats = index_dir_at(&roots, state.path(), false).unwrap();
        read_fault::clear();

        assert_eq!(stats.transient_failures, 1);
        let manifest = Manifest::load(state.path());
        assert!(
            !manifest.files.contains_key("b.md"),
            "a brand-new file that failed transiently must not be recorded in the manifest"
        );
    }

    /// Task 5: a PERMANENT failure (bad CFB header — no `io::Error` anywhere in the chain, so
    /// `is_transient` is false) keeps the CURRENT sig in the manifest — the file is treated as
    /// settled/unchanged and is never re-attempted, matching the existing bad-.doc behavior
    /// (`index_dir_skips_corrupt_office_doc_and_continues`).
    #[test]
    fn index_dir_at_locked_permanent_keeps_current_sig_not_retried() {
        use crate::root::Root;
        let dir = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("ok.md"), b"# T\nhello world\n").unwrap();
        std::fs::write(
            dir.path().join("bad.doc"),
            b"this is not a real CFB .doc file",
        )
        .unwrap();
        let roots = [Root {
            label: String::new(),
            path: dir.path().into(),
        }];

        let s1 = index_dir_at(&roots, state.path(), false).unwrap();
        assert_eq!(
            s1.permanent_skips, 1,
            "the corrupt .doc is a permanent skip"
        );
        assert!(
            Manifest::load(state.path()).files.contains_key("bad.doc"),
            "its sig IS recorded, so it's treated as settled"
        );

        // Change an UNRELATED file so this pass actually re-runs the indexing loop (not just the
        // empty-delta hot-path early return) — bad.doc's unchanged sig must keep it out of
        // `delta.changed` entirely, so it's never re-attempted.
        std::fs::write(dir.path().join("ok.md"), b"# T\nhello world again\n").unwrap();
        let s2 = index_dir_at(&roots, state.path(), false).unwrap();
        assert_eq!(
            s2.permanent_skips, 0,
            "an unchanged permanently-bad file is not re-attempted next pass"
        );
    }

    /// Task 11 (integration): a transient READ failure that recovers WITHIN `with_read_retry`'s
    /// budget (`K < read_retries()` armed failures, so the call that finally succeeds is still
    /// inside the loop) must be fully absorbed by a single `index_dir_at` pass — the file ends up
    /// indexed with its NEW content, `transient_failures` stays 0, and nothing is lost. Contrasts
    /// with `index_dir_at_locked_transient_reverts_to_old_sig_and_preserves_doc` (Task 5), which
    /// arms `read_retries() + 1` failures to exhaust the budget instead.
    #[test]
    fn transient_k_times_then_indexed() {
        use crate::root::Root;
        let dir = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("x.md"), b"# X\noriginal term\n").unwrap();
        let roots = [Root {
            label: String::new(),
            path: dir.path().into(),
        }];
        index_dir_at(&roots, state.path(), true).unwrap(); // seed manifest

        std::fs::write(dir.path().join("x.md"), b"# X\nrecovered term here\n").unwrap();
        let abs = abs_root(dir.path()).join("x.md");
        let k = read_retries().saturating_sub(1).max(1); // K < read_retries()
        read_fault::arm_read(&abs, k, std::io::ErrorKind::TimedOut, None);

        let stats = index_dir_at(&roots, state.path(), false).unwrap();
        read_fault::clear();

        assert_eq!(
            stats.transient_failures, 0,
            "retries absorbed it within the pass: {stats:?}"
        );
        assert_eq!(stats.removed, 0);
        let idx = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert!(
            idx.search("recovered", 10)
                .unwrap()
                .iter()
                .any(|h| h.path.ends_with("x.md")),
            "the file is indexed with its new content, nothing lost"
        );
    }

    /// Task 11 (integration): regression guard for the bad-`.doc` skip, exercised end-to-end
    /// across two REAL passes (the first a full `index_dir_at`, the second the production
    /// `freshen_blocking_at` path) instead of two `index_dir_at` calls — a genuinely-corrupt file
    /// (no `io::Error` in its chain, so `is_transient` is false) is counted exactly once as a
    /// `permanent_skips`, and a later freshen — even via the scoped path — never re-attempts it.
    #[test]
    fn permanent_corrupt_indexed_once_not_retried() {
        use crate::root::Root;
        let dir = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("ok.md"), b"# T\nhello world\n").unwrap();
        std::fs::write(
            dir.path().join("bad.doc"),
            b"this is not a real CFB .doc file",
        )
        .unwrap();
        let roots = [Root {
            label: String::new(),
            path: dir.path().into(),
        }];

        let s1 = index_dir_at(&roots, state.path(), true).unwrap();
        assert_eq!(
            s1.permanent_skips, 1,
            "the corrupt file is exactly one permanent skip: {s1:?}"
        );

        // Touch an unrelated file (and bump the root's own mtime so the SCOPED freshen path
        // actually re-walks this dir instead of short-circuiting on an unchanged dirsig) so the
        // second pass runs the real indexing loop rather than the empty-delta hot path.
        std::fs::write(dir.path().join("ok.md"), b"# T\nhello world again\n").unwrap();
        filetime::set_file_mtime(
            dir.path(),
            filetime::FileTime::from_system_time(
                std::time::SystemTime::now() + std::time::Duration::from_secs(30),
            ),
        )
        .unwrap();
        let s2 =
            freshen_blocking_at(&roots, state.path(), std::time::Duration::from_secs(3)).unwrap();
        assert_eq!(
            s2.permanent_skips, 0,
            "the second freshen does not re-attempt the already-settled bad file: {s2:?}"
        );

        let idx = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert!(
            idx.search("again", 10)
                .unwrap()
                .iter()
                .any(|h| h.path.ends_with("ok.md")),
            "the unrelated file's update is indexed normally"
        );
    }

    /// Task 11 (integration): a transient STAT failure during a freshen must not just preserve the
    /// doc this pass (Task 4) — it must leave the file's DIR unsettled so the change it masked is
    /// picked up on the very next freshen, with no fault armed. Uses the cross-platform `TimedOut`
    /// kind (no errno): `is_transient`'s errno table is unix-only (see its doc comment), so an
    /// errno-keyed fault would silently fail to classify as transient on Windows — the same choice
    /// every other seam call in this file makes.
    #[test]
    fn transient_stat_marks_dir_unsettled_and_rescans() {
        use crate::root::Root;
        let dir = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("x.md"), b"# X\noriginal term\n").unwrap();
        let roots = [Root {
            label: String::new(),
            path: dir.path().into(),
        }];
        index_dir_at(&roots, state.path(), true).unwrap(); // seed manifest + dirsig

        std::fs::write(dir.path().join("x.md"), b"# X\nrevised term now\n").unwrap();
        // A content-only edit doesn't bump the parent dir's own mtime (that only tracks
        // add/remove/rename of entries) — force it deterministically so the scoped freshen path
        // actually re-walks this dir instead of hitting the unchanged-dirsig fast path, same
        // technique `reindex_scoped_rescan_skips_unrelated_dir` uses for a coarse-granularity FS
        // clock.
        filetime::set_file_mtime(
            dir.path(),
            filetime::FileTime::from_system_time(
                std::time::SystemTime::now() + std::time::Duration::from_secs(30),
            ),
        )
        .unwrap();
        let abs = abs_root(dir.path()).join("x.md");
        read_fault::arm_stat(&abs, 1, std::io::ErrorKind::TimedOut, None);

        let _s1 =
            freshen_blocking_at(&roots, state.path(), std::time::Duration::from_secs(3)).unwrap();
        read_fault::clear();
        let idx1 = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert!(
            idx1.search("revised", 10).unwrap().is_empty(),
            "the transient stat this pass must not surface the new content yet"
        );

        // Next freshen, no fault armed: the dir was left unsettled (dirsig not advanced), so it
        // re-scans and recovers the masked change.
        let _s2 =
            freshen_blocking_at(&roots, state.path(), std::time::Duration::from_secs(3)).unwrap();
        let idx2 = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert!(
            idx2.search("revised", 10)
                .unwrap()
                .iter()
                .any(|h| h.path.ends_with("x.md")),
            "the follow-up freshen recovers the change"
        );
    }

    /// Task 11 (integration): the Task 8 empty-mount hold end-to-end — a root's walk comes back
    /// empty while the root itself stays put, docs stay searchable through the hold (not
    /// mass-removed), and once the root regains its populated form, a follow-up freshen clears the
    /// hold and re-syncs normally (nothing left dangling, nothing genuinely removed).
    #[test]
    fn empty_mount_then_remount_recovers() {
        use crate::root::Root;
        let root = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        let one = b"alpha content one\n";
        let two = b"bravo content two\n";
        std::fs::write(root.path().join("one.md"), one).unwrap();
        std::fs::write(root.path().join("two.md"), two).unwrap();
        let roots = vec![Root {
            label: String::new(),
            path: root.path().into(),
        }];
        index_dir_at(&roots, state.path(), true).unwrap(); // seed manifest with 2 entries

        // The mount goes empty (share drops out) while the root directory itself stays put — bumps
        // only the root's own dir mtime, same shape `reindex_scoped_rescan_holds_root_on_empty_mount`
        // exercises via the scoped freshen path.
        std::fs::remove_file(root.path().join("one.md")).unwrap();
        std::fs::remove_file(root.path().join("two.md")).unwrap();
        filetime::set_file_mtime(
            root.path(),
            filetime::FileTime::from_system_time(
                std::time::SystemTime::now() + std::time::Duration::from_secs(30),
            ),
        )
        .unwrap();

        let s1 =
            freshen_blocking_at(&roots, state.path(), std::time::Duration::from_secs(3)).unwrap();
        assert!(s1.empty_mount_holds > 0, "the hold must register: {s1:?}");
        assert_eq!(s1.removed, 0, "held back, not mass-removed: {s1:?}");

        let idx1 = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert!(
            idx1.search("alpha", 10)
                .unwrap()
                .iter()
                .any(|h| h.path.ends_with("one.md")),
            "held doc stays searchable through the hold"
        );
        assert!(
            idx1.search("bravo", 10)
                .unwrap()
                .iter()
                .any(|h| h.path.ends_with("two.md")),
            "held doc stays searchable through the hold"
        );

        // The mount comes back — the root regains its populated form.
        std::fs::write(root.path().join("one.md"), one).unwrap();
        std::fs::write(root.path().join("two.md"), two).unwrap();
        filetime::set_file_mtime(
            root.path(),
            filetime::FileTime::from_system_time(
                std::time::SystemTime::now() + std::time::Duration::from_secs(60),
            ),
        )
        .unwrap();

        let s2 =
            freshen_blocking_at(&roots, state.path(), std::time::Duration::from_secs(3)).unwrap();
        assert_eq!(
            s2.empty_mount_holds, 0,
            "the hold clears once the root is populated again: {s2:?}"
        );
        assert_eq!(s2.removed, 0, "nothing genuinely removed: {s2:?}");

        let manifest = Manifest::load(state.path());
        assert_eq!(
            manifest.files.len(),
            2,
            "both docs re-synced normally: {:?}",
            manifest.files
        );
        let idx2 = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert!(idx2
            .search("alpha", 10)
            .unwrap()
            .iter()
            .any(|h| h.path.ends_with("one.md")));
        assert!(idx2
            .search("bravo", 10)
            .unwrap()
            .iter()
            .any(|h| h.path.ends_with("two.md")));
    }

    #[test]
    fn ensure_fresh_skips_schema_migration_when_nothing_changed() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# Hi\nhello world\n").unwrap();
        let sig = file_sig(dir.path().join("a.md").as_path()).unwrap();
        let mut old = Manifest::default();
        old.files.insert("a.md".into(), sig);
        old.index_schema_version = 1;
        old.save(dir.path()).unwrap();

        let s = ensure_fresh(dir.path()).unwrap();
        assert_eq!(s.added, 0);
        assert_eq!(s.removed, 0);
        assert_eq!(Manifest::load(dir.path()).index_schema_version, 1);
    }

    #[test]
    fn ensure_fresh_noop_then_picks_up_change() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.md"), "# T\ncontracts\n").unwrap();
        let s1 = ensure_fresh(dir.path()).unwrap();
        assert_eq!(s1.added, 1);

        // nothing changed → cheap no-op, no writes
        let s2 = ensure_fresh(dir.path()).unwrap();
        assert_eq!(s2.added, 0);
        assert_eq!(s2.removed, 0);

        // change on disk → picked up automatically, searchable
        fs::write(
            dir.path().join("a.md"),
            "# T\ncontracts delivery regulation\n",
        )
        .unwrap();
        let s3 = ensure_fresh(dir.path()).unwrap();
        assert_eq!(s3.added, 1);
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(idx
            .search("regulation", 10)
            .unwrap()
            .iter()
            .any(|h| h.path.ends_with("a.md")));
    }

    #[test]
    fn index_dir_skips_when_index_lock_held() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nalpha\n").unwrap();

        // Stand in for another process that is already indexing this base.
        let held = crate::index::lock::try_index_lock(dir.path());
        assert!(held.is_some(), "the test itself holds the index lock");

        // Under contention index_dir must SKIP: no work reported, nothing written.
        let stats = index_dir(dir.path(), false).unwrap();
        assert_eq!(
            stats,
            IndexStats::default(),
            "contended index_dir returns default stats"
        );
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(
            idx.search("alpha", 10).unwrap().is_empty(),
            "no document indexed while the lock was held"
        );

        // Once the holder releases, the lock is free and indexing proceeds normally.
        drop(held);
        let stats2 = index_dir(dir.path(), false).unwrap();
        assert_eq!(stats2.added, 1, "released: index_dir indexes the file");
        let idx2 = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(
            !idx2.search("alpha", 10).unwrap().is_empty(),
            "document searchable after the lock is released"
        );
    }

    #[test]
    fn index_dir_indexes_loose_images() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("Diagrams")).unwrap();
        std::fs::write(dir.path().join("Diagrams").join("bus.png"), b"\x89PNG\r\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(
            idx.search("bus", 10)
                .unwrap()
                .iter()
                .any(|h| h.path.ends_with("bus.png")),
            "loose image is searchable by name"
        );
    }

    #[test]
    fn dir_mtime_map_keys_and_changes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nx\n").unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub").join("b.md"), b"# B\ny\n").unwrap();
        let m1 = dir_mtime_map(dir.path()).unwrap();
        assert!(m1.contains_key("c:"), "root dir present as c:");
        assert!(m1.contains_key("c:sub"), "subdir present as c:sub");
        // Adding a file bumps its parent dir's mtime entry.
        std::thread::sleep(std::time::Duration::from_millis(10));
        std::fs::write(dir.path().join("sub").join("c.md"), b"# C\nz\n").unwrap();
        let m2 = dir_mtime_map(dir.path()).unwrap();
        assert_ne!(m1.get("c:sub"), m2.get("c:sub"), "sub mtime changed");
        assert_eq!(
            m1.get("c:"),
            m2.get("c:"),
            "root unchanged (no direct child added)"
        );
        // The aggregate signature wrapper still works and is stable across calls.
        let s = dir_mtime_signature(dir.path()).unwrap();
        assert_eq!(
            s,
            dir_mtime_signature(dir.path()).unwrap(),
            "aggregate signature stable"
        );
    }

    #[test]
    fn dir_signature_changes_on_file_add_and_ignores_pure_read() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nx\n").unwrap();
        let s1 = dir_mtime_signature(dir.path()).unwrap();
        assert_eq!(
            s1,
            dir_mtime_signature(dir.path()).unwrap(),
            "signature is stable when nothing changes"
        );

        std::thread::sleep(std::time::Duration::from_millis(10));
        std::fs::write(dir.path().join("b.md"), b"# B\ny\n").unwrap(); // same dir, new file
        let s2 = dir_mtime_signature(dir.path()).unwrap();
        assert_ne!(
            s1, s2,
            "adding a file in an existing dir must change the signature"
        );

        std::thread::sleep(std::time::Duration::from_millis(10));
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub").join("c.md"), b"# C\nz\n").unwrap();
        assert_ne!(
            s2,
            dir_mtime_signature(dir.path()).unwrap(),
            "new subdir + file changes the signature"
        );
    }

    #[test]
    fn dir_signature_ignores_hidden_dir_but_tracks_real_changes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nx\n").unwrap();
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".git").join("head"), b"placeholder\n").unwrap();
        let s1 = dir_mtime_signature(dir.path()).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(10));
        std::fs::write(dir.path().join(".git").join("index"), b"placeholder\n").unwrap();
        assert_eq!(
            s1,
            dir_mtime_signature(dir.path()).unwrap(),
            "a hidden dir like .git must not perturb the signature"
        );

        std::thread::sleep(std::time::Duration::from_millis(10));
        std::fs::write(dir.path().join("b.md"), b"# B\ny\n").unwrap();
        assert_ne!(
            s1,
            dir_mtime_signature(dir.path()).unwrap(),
            "a real corpus file add still changes the signature"
        );
    }

    #[test]
    fn index_one_file_updates_only_that_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nalpha original\n").unwrap();
        std::fs::write(dir.path().join("b.md"), b"# B\nbravo stays\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        // Edit a.md in place.
        std::fs::write(dir.path().join("a.md"), b"# A\nalpha rewritten sentinel\n").unwrap();
        let _lock = crate::index::lock::try_index_lock(dir.path()).unwrap();
        index_one_file_locked(dir.path(), "a.md").unwrap();
        drop(_lock);
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(
            idx.search("sentinel", 10)
                .unwrap()
                .iter()
                .any(|h| h.path.ends_with("a.md")),
            "new text indexed"
        );
        assert!(
            idx.search("original", 10).unwrap().is_empty(),
            "old text dropped"
        );
        assert!(
            idx.search("bravo", 10)
                .unwrap()
                .iter()
                .any(|h| h.path.ends_with("b.md")),
            "other file untouched"
        );
        assert_eq!(
            Manifest::load(dir.path()).files.get("a.md"),
            Some(&file_sig(&dir.path().join("a.md")).unwrap())
        );
    }

    #[test]
    fn index_one_file_resolves_reference_with_fresh_provenance_sig() {
        // The by_canon/link_reference block is the risky part of index_one_file_locked: it re-stamps
        // this file's REFERENCES edge(s) with a FileSig for provenance. That signature must be the
        // freshly-computed post-edit one (what index_file_into just returned), not a stale copy read
        // from the manifest loaded before this call updated it — otherwise provenance silently
        // diverges from what a full reindex would record for the same edit.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nsee [b](b.md)\n").unwrap();
        std::fs::write(dir.path().join("b.md"), b"# B\nbravo\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        // Edit a.md, keeping the link to b.md, so index_one_file_locked must re-resolve it. The
        // edited body is a different length than the original, so `size` alone guarantees a
        // different FileSig regardless of mtime_secs resolution.
        std::fs::write(
            dir.path().join("a.md"),
            b"# A\nsentinel edit\nsee [b](b.md)\n",
        )
        .unwrap();
        let _lock = crate::index::lock::try_index_lock(dir.path()).unwrap();
        index_one_file_locked(dir.path(), "a.md").unwrap();
        drop(_lock);

        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        assert!(
            crate::graph::traverse::neighbors(&g, "a.md", None, 1)
                .unwrap()
                .contains(&"b.md".to_string()),
            "REFERENCES edge a.md -> b.md re-resolved after single-file reindex"
        );

        let fresh_sig = file_sig(&dir.path().join("a.md")).unwrap();
        let edge = g
            .all_edges()
            .unwrap()
            .into_iter()
            .find(|e| e.edge_type == "REFERENCES" && e.from == "a.md" && e.to == "b.md")
            .expect("REFERENCES edge a.md -> b.md exists");
        assert_eq!(
            edge.prov.file_sig,
            Some(fresh_sig),
            "REFERENCES edge provenance must carry a.md's post-edit signature, not a stale pre-edit one"
        );
    }

    #[test]
    fn index_one_file_matches_full_reindex_for_that_file() {
        // Golden: after the same edit, index_one_file_locked(X) yields the same searchable state for X
        // as a full index_dir(force). (Chunk-level equivalence via search; graph auto-edges by source;
        // a cross-file link from a.md to b.md exercises the by_canon/link_reference resolution path too.)
        let build = |single: bool| -> (Vec<String>, u64, Vec<(String, String)>) {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(
                dir.path().join("a.md"),
                b"# A\none\nsee [b](b.md)\n## S2\ntwo\n",
            )
            .unwrap();
            std::fs::write(dir.path().join("b.md"), b"# B\nbravo\n").unwrap();
            index_dir(dir.path(), true).unwrap();
            std::fs::write(
                dir.path().join("a.md"),
                b"# A\none edited\nsee [b](b.md)\n## S2\ntwo\n### S3\nthree\n",
            )
            .unwrap();
            if single {
                let _l = crate::index::lock::try_index_lock(dir.path()).unwrap();
                index_one_file_locked(dir.path(), "a.md").unwrap();
            } else {
                index_dir(dir.path(), false).unwrap();
            }
            let idx = DocIndex::open_or_create(dir.path()).unwrap();
            let mut hits: Vec<String> = idx
                .search("edited three", 20)
                .unwrap()
                .into_iter()
                .map(|h| h.path)
                .collect();
            hits.sort();
            let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
            let mut refs: Vec<(String, String)> = g
                .all_edges()
                .unwrap()
                .into_iter()
                .filter(|e| e.edge_type == "REFERENCES")
                .map(|e| (e.from, e.to))
                .collect();
            refs.sort();
            (hits, g.node_count().unwrap(), refs)
        };
        assert_eq!(
            build(true),
            build(false),
            "single-file reindex equals full delta reindex for the edited file, including its REFERENCES edges"
        );
    }

    #[test]
    fn reindex_dirs_matches_full_across_scenarios() {
        // Golden: for each mutation, a scoped pass over the affected dirs yields the same searchable
        // state + graph (nodes + auto edges) as a full index_dir(force) on the same final disk state.
        let mutate = |p: &std::path::Path| {
            std::fs::write(p.join("a.md"), b"# A\nalpha [c](sub/c.md)\n").unwrap();
            std::fs::create_dir_all(p.join("sub")).unwrap();
            std::fs::write(p.join("sub").join("c.md"), b"# C\ngamma\n").unwrap();
        };
        let scenario = |apply: &dyn Fn(&std::path::Path),
                        scoped: bool|
         -> (Vec<String>, u64, Vec<(String, String)>) {
            let dir = tempfile::tempdir().unwrap();
            mutate(dir.path());
            index_dir(dir.path(), true).unwrap();
            let before = dir_mtime_map(dir.path()).unwrap();
            apply(dir.path());
            let after = dir_mtime_map(dir.path()).unwrap();
            if scoped {
                // Production classification: the same `diff_corpus_dirs` + `notes_submap_changed`
                // calls `freshen_blocking` makes, so this golden test guards the real
                // scoped-vs-full equivalence, not a local approximation of it.
                let (changed, added, removed) = diff_corpus_dirs(&before, &after);
                let notes = notes_submap_changed(&before, &after);
                let _l = crate::index::lock::try_index_lock(dir.path()).unwrap();
                reindex_dirs_locked(dir.path(), &after, &changed, &added, &removed, notes).unwrap();
            } else {
                index_dir(dir.path(), false).unwrap();
            }
            let idx = DocIndex::open_or_create(dir.path()).unwrap();
            let mut hits: Vec<String> = idx
                .search("alpha gamma delta", 50)
                .unwrap()
                .into_iter()
                .map(|h| h.path)
                .collect();
            hits.sort();
            let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
            let mut refs: Vec<(String, String)> = g
                .all_edges()
                .unwrap()
                .into_iter()
                .filter(|e| e.edge_type == "REFERENCES")
                .map(|e| (e.from.clone(), e.to.clone()))
                .collect();
            refs.sort();
            (hits, g.node_count().unwrap(), refs)
        };
        // add a new file in a subdir
        let add = |p: &std::path::Path| {
            std::fs::write(p.join("sub").join("d.md"), b"# D\ndelta\n").unwrap()
        };
        assert_eq!(scenario(&add, true), scenario(&add, false), "add");
        // edit a file
        let edit = |p: &std::path::Path| {
            std::fs::write(p.join("sub").join("c.md"), b"# C\ngamma edited delta\n").unwrap()
        };
        assert_eq!(scenario(&edit, true), scenario(&edit, false), "edit");
        // remove a file
        let rm = |p: &std::path::Path| std::fs::remove_file(p.join("sub").join("c.md")).unwrap();
        assert_eq!(scenario(&rm, true), scenario(&rm, false), "remove-file");
        // remove a dir
        let rmdir = |p: &std::path::Path| std::fs::remove_dir_all(p.join("sub")).unwrap();
        assert_eq!(
            scenario(&rmdir, true),
            scenario(&rmdir, false),
            "remove-dir"
        );
        // delete the referenced doc (sub/c.md) — a.md's edge to it must be gone in BOTH
        // (covered by remove-file above, since a.md -> sub/c.md).
    }

    #[test]
    fn reindex_dirs_respects_gitignore_without_a_git_repo() {
        // Regression: the scoped per-dir walk in `reindex_dirs_locked` must set `require_git(false)`
        // like the reference walkers (`walk::walk_files`, `collect_dir_mtimes`) do — `require_git`
        // defaults to `true` in the `ignore` crate, so `.gitignore` is honored only inside a git
        // repo. No `.git` dir is created here on purpose: the bug only shows on a non-git corpus.
        let build = |scoped: bool| -> (Vec<String>, u64, Vec<(String, String)>) {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join(".gitignore"), b"ignored.md\n").unwrap();
            std::fs::write(dir.path().join("a.md"), b"# A\nkeep\n").unwrap();
            std::fs::write(dir.path().join("ignored.md"), b"# X\nshould not appear\n").unwrap();
            index_dir(dir.path(), true).unwrap();
            let before = dir_mtime_map(dir.path()).unwrap();
            // Touch the root dir (which also contains ignored.md) so the scoped pass rescans it.
            std::fs::write(dir.path().join("b.md"), b"# B\nother\n").unwrap();
            let after = dir_mtime_map(dir.path()).unwrap();
            if scoped {
                let (changed, added, removed) = diff_corpus_dirs(&before, &after);
                let notes = notes_submap_changed(&before, &after);
                let _l = crate::index::lock::try_index_lock(dir.path()).unwrap();
                reindex_dirs_locked(dir.path(), &after, &changed, &added, &removed, notes).unwrap();
            } else {
                index_dir(dir.path(), false).unwrap();
            }
            let idx = DocIndex::open_or_create(dir.path()).unwrap();
            assert!(
                idx.search("appear", 10).unwrap().is_empty(),
                "gitignored file must not be indexed even without a .git dir (scoped={scoped})"
            );
            assert!(
                !Manifest::load(dir.path()).files.contains_key("ignored.md"),
                "gitignored file must not be recorded in the manifest (scoped={scoped})"
            );
            let mut hits: Vec<String> = idx
                .search("keep other", 50)
                .unwrap()
                .into_iter()
                .map(|h| h.path)
                .collect();
            hits.sort();
            let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
            let mut refs: Vec<(String, String)> = g
                .all_edges()
                .unwrap()
                .into_iter()
                .filter(|e| e.edge_type == "REFERENCES")
                .map(|e| (e.from.clone(), e.to.clone()))
                .collect();
            refs.sort();
            (hits, g.node_count().unwrap(), refs)
        };
        assert_eq!(
            build(true),
            build(false),
            "scoped matches full: gitignored file excluded from both"
        );
    }

    /// Finding-4 (buffer-then-swap): a transient read failure during `extract_file` must NOT lose
    /// the prior doc. Before the fix, `index_file_into` deleted the old chunks/auto-edges BEFORE
    /// extracting, so a failed read this pass left the doc gone once the caller committed. Arming
    /// the injected read fault (`read_fault::arm_read`) reproduces the failure deterministically —
    /// no real network/flaky-fs needed.
    #[test]
    fn index_file_into_preserves_prior_doc_when_read_fails() {
        // Two headings so a.md produces at least one auto-structural NODE (Document/Section) AND
        // one auto-structural EDGE (the sequential link between the two sections) — the graph
        // assertion below needs a real edge to discriminate the bug, not just a node.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("a.md"),
            b"# Intro\nhello alpha\n## Body\nworld beta\n",
        )
        .unwrap();
        index_dir(dir.path(), true).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(
            idx.search("alpha", 10)
                .unwrap()
                .iter()
                .any(|h| h.path.ends_with("a.md")),
            "sanity: original doc is indexed before the injected failure"
        );

        let graph = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        // Snapshot the GRAPH state for a.md's auto layer before the injected failure. This is the
        // layer the pre-fix code ACTUALLY loses data on: `graph.delete_auto_by_source` runs a
        // plain autocommit sqlite DELETE (no transaction wrapper), so pre-fix it deletes a.md's
        // auto nodes/edges immediately — permanently, independent of whatever the tantivy
        // `IndexWriter` has staged (tantivy deletes aren't durable without a `commit()`, so a
        // tantivy-only assertion can't tell buggy from fixed).
        let auto_snapshot = |g: &crate::graph::store::GraphStore| -> (usize, usize) {
            let nodes = g
                .all_nodes()
                .unwrap()
                .into_iter()
                .filter(|n| n.prov.source_path == "a.md" && n.prov.origin.starts_with("auto-"))
                .count();
            let edges = g
                .all_edges()
                .unwrap()
                .into_iter()
                .filter(|e| e.prov.source_path == "a.md" && e.prov.origin.starts_with("auto-"))
                .count();
            (nodes, edges)
        };
        let before = auto_snapshot(&graph);
        assert!(before.0 > 0, "sanity: a.md produced auto graph node(s)");
        assert!(
            before.1 > 0,
            "sanity: a.md produced at least one auto graph edge (sequential section link)"
        );

        let mut writer = with_writer_retry(|| idx.index.writer(50_000_000)).unwrap();
        let abs = idx.doc_file("a.md");
        read_fault::arm_read(&abs, 1, std::io::ErrorKind::TimedOut, None);
        let mut links: Vec<(String, String)> = Vec::new();
        let result = index_file_into(&idx, &graph, &writer, "a.md", &abs, &mut links, None);
        assert!(
            result.is_err(),
            "a read failure must surface as Err, not a silent None"
        );
        read_fault::clear();

        // The graph's auto nodes/edges for a.md must be UNCHANGED: pre-fix, delete_auto_by_source
        // committed (autocommit sqlite) BEFORE the failed read, so this goes RED against pre-fix
        // (before.1 nodes/edges gone, after == (0, 0)) and GREEN with buffer-then-swap.
        let after = auto_snapshot(&graph);
        assert_eq!(
            before, after,
            "a.md's auto graph nodes/edges must survive a read failure this pass"
        );

        // tantivy side too: commit (a no-op here since nothing was ever staged) and confirm via a
        // fresh reader that the prior doc is still searchable.
        with_writer_retry(|| writer.commit()).unwrap();
        drop(writer);
        let idx2 = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(
            idx2.search("alpha", 10)
                .unwrap()
                .iter()
                .any(|h| h.path.ends_with("a.md")),
            "prior doc must survive a read failure this pass"
        );
    }

    /// Finding-3 regression guard: `index_file_into` stores the labeled `doc_key` it was PASSED,
    /// never a bare relpath re-derived internally. Deviation from the task brief: the brief's
    /// snippet checks ordinal `0`, but `index_file_into`'s per-chunk ordinal starts at `1` (see
    /// `chunk_ord`/`read_chunk_by_ord_returns_body_and_neighbors`) — ordinal `0` never exists for
    /// any key, labeled or bare, so it can't discriminate the bug this test guards against. Using
    /// the real first ordinal (`1`) keeps the same intent while actually being able to fail.
    #[test]
    fn index_file_into_persists_labeled_doc_key() {
        use crate::root::Root;
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(a.path().join("x.md"), b"# X\nhello\n").unwrap();
        let roots = vec![
            Root {
                label: "docs".into(),
                path: a.path().into(),
            },
            Root {
                label: "specs".into(),
                path: b.path().into(),
            },
        ];
        let idx = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        let graph = crate::graph::store::GraphStore::open(state.path()).unwrap();
        let mut writer = with_writer_retry(|| idx.index.writer(50_000_000)).unwrap();
        let mut links: Vec<(String, String)> = Vec::new();
        let abs = idx.doc_file("docs/x.md");
        index_file_into(&idx, &graph, &writer, "docs/x.md", &abs, &mut links, None).unwrap();
        with_writer_retry(|| writer.commit()).unwrap();

        let idx2 = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert!(
            idx2.read_chunk_by_ord("docs/x.md", 1).unwrap().is_some(),
            "the labeled doc_key is persisted verbatim"
        );
        assert!(
            idx2.read_chunk_by_ord("x.md", 1).unwrap().is_none(),
            "no bare relpath was recomputed internally"
        );
    }
}

#[cfg(test)]
mod search_tests {
    use super::*;
    use std::fs;
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
    fn ranked_search_finds_by_inflected_query() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            chunk("a.md", "Delivery contracts have been signed"),
            chunk("b.md", "unrelated other content"),
        ])
        .unwrap();

        // Query uses a different inflection ("contract") than the doc ("contracts").
        let hits = idx.search("contract", 10).unwrap();
        assert!(!hits.is_empty(), "stemmed query should match inflected doc");
        assert_eq!(hits[0].path, "a.md");
        assert!(hits[0].score > 0.0);
        assert!(!hits[0].snippet.is_empty());
    }

    /// The core of the search-reader staleness bug, made deterministic. A long-lived shared reader
    /// picks up a commit made through a SEPARATE `Index` (as the server's freshen does) only via
    /// tantivy's background filesystem watcher — which is dead on the network folders this daemon
    /// targets. We emulate that here by forcing the shared reader to `ReloadPolicy::Manual` (no
    /// watcher). The reader is then STALE for a new file (the bug), and RE-OPENING the index — what
    /// `GraphHandle::refresh_idx` does on a freshen — is what surfaces it. (A bare `reload()` also
    /// surfaces it, but its error was swallowed and it was skipped when the local delta was 0; the
    /// fix reopens through the transient-FS-retry path and propagates the error.)
    #[test]
    fn shared_reader_is_stale_until_reopen_when_watcher_is_dead() {
        use tantivy::ReloadPolicy;
        let dir = tempfile::tempdir().unwrap();
        let mut shared = DocIndex::open_or_create(dir.path()).unwrap();
        // Defeat the background watcher — the network-share reality.
        shared.reader = shared
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .unwrap();
        shared
            .write_chunks(&[chunk("a.md", "alpha content")])
            .unwrap();
        assert!(
            !shared.search("alpha", 10).unwrap().is_empty(),
            "sanity: the pre-existing file is visible"
        );

        // A NEW file is committed through a SEPARATE index instance (mirrors freshen's own `Index`).
        let external = DocIndex::open_or_create(dir.path()).unwrap();
        external
            .write_chunks(&[
                chunk("a.md", "alpha content"),
                chunk("b.md", "beta latecomer content"),
            ])
            .unwrap();

        // The bug: with no watcher and no explicit refresh, the shared reader misses it.
        assert!(
            shared.search("latecomer", 10).unwrap().is_empty(),
            "regression: a dead-watcher shared reader must be stale before an explicit refresh"
        );

        // The fix: re-opening the index (what GraphHandle::refresh_idx does) surfaces the commit.
        let reopened = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(
            reopened
                .search("latecomer", 10)
                .unwrap()
                .iter()
                .any(|h| h.path.contains("b.md")),
            "a freshly re-opened index must see the externally committed file"
        );
    }

    #[test]
    fn chunk_ord_uses_page_for_pdf_else_sequence() {
        assert_eq!(chunk_ord("pdf", "p.21", 5), 21);
        assert_eq!(chunk_ord("pdf", "p.350", 1), 350);
        assert_eq!(chunk_ord("md", "Introduction", 3), 3); // non-pdf -> sequence
        assert_eq!(chunk_ord("pdf", "weird", 7), 7); // unparseable page -> sequence fallback
    }

    #[test]
    fn search_hit_carries_ord() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[Chunk {
            doc_path: PathBuf::from("d.pdf"),
            location: "p.7".into(),
            file_type: "pdf".into(),
            text: "hot cpu swap".into(),
        }])
        .unwrap();
        let hits = idx.search("swap", 10).unwrap();
        assert_eq!(hits[0].ord, 7);
    }

    #[test]
    fn read_chunk_fetches_body_by_path_and_location() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let page = |loc: &str, text: &str| Chunk {
            doc_path: PathBuf::from("doc.pdf"),
            location: loc.into(),
            file_type: "pdf".into(),
            text: text.into(),
        };
        idx.write_chunks(&[
            page("p.1", "first page body"),
            page("p.2", "second page body"),
        ])
        .unwrap();

        // Exact path+location returns that chunk's stored body — no file re-parse.
        assert_eq!(
            idx.read_chunk("doc.pdf", "p.2").unwrap().as_deref(),
            Some("second page body")
        );
        // Unknown location -> None, so the caller falls back to reading the file.
        assert_eq!(idx.read_chunk("doc.pdf", "p.99").unwrap(), None);
    }

    #[test]
    fn display_line_is_numbered_with_nonnumeric_label() {
        let pdf = RankedHit {
            path: "d.pdf".into(),
            location: "p.350".into(),
            file_type: "pdf".into(),
            ord: 350,
            snippet: "hot swap".into(),
            score: 17.7,
        };
        let line = pdf.display_line();
        assert!(line.starts_with("d.pdf#350"), "copy-ready key: {line}");
        assert!(line.contains("pdf"), "non-numeric label for pdf: {line}");
        assert!(!line.contains("p.350"), "no competing page number: {line}");

        let md = RankedHit {
            path: "d.md".into(),
            location: "Introduction".into(),
            file_type: "md".into(),
            ord: 2,
            snippet: "text".into(),
            score: 3.0,
        };
        assert!(md.display_line().starts_with("d.md#2"));
        assert!(md.display_line().contains("Introduction"));
        assert!(
            !md.display_line().contains("· md ·"),
            "file_type must not leak as label in non-paged line: {}",
            md.display_line()
        );
    }

    #[test]
    fn read_chunk_by_ord_returns_body_and_neighbors() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let sec = |loc: &str, t: &str| Chunk {
            doc_path: PathBuf::from("d.md"),
            location: loc.into(),
            file_type: "md".into(),
            text: t.into(),
        };
        idx.write_chunks(&[sec("A", "alpha"), sec("B", "bravo"), sec("C", "charlie")])
            .unwrap();

        let mid = idx.read_chunk_by_ord("d.md", 2).unwrap().unwrap();
        assert_eq!(mid.body, "bravo");
        assert_eq!(mid.prev, Some(1));
        assert_eq!(mid.next, Some(3));

        let first = idx.read_chunk_by_ord("d.md", 1).unwrap().unwrap();
        assert_eq!(first.prev, None);
        assert_eq!(first.next, Some(2));

        assert!(idx.read_chunk_by_ord("d.md", 99).unwrap().is_none());
    }

    #[test]
    fn resolve_path_strips_spurious_leading_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[Chunk {
            doc_path: PathBuf::from("Archive DB\\doc.pdf"),
            location: "p.1".into(),
            file_type: "pdf".into(),
            text: "x".into(),
        }])
        .unwrap();
        assert_eq!(
            idx.resolve_path("kb-manual\\Archive DB\\doc.pdf")
                .unwrap()
                .as_deref(),
            Some("Archive DB\\doc.pdf")
        );
    }

    #[test]
    fn canonical_document_path_exact_and_tolerant() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[Chunk {
            doc_path: PathBuf::from("real.pdf"),
            location: "p.1".into(),
            file_type: "pdf".into(),
            text: "x".into(),
        }])
        .unwrap();
        assert_eq!(
            idx.canonical_document_path("real.pdf").as_deref(),
            Some("real.pdf")
        );
        assert_eq!(
            idx.canonical_document_path("kb-manual\\real.pdf")
                .as_deref(),
            Some("real.pdf")
        );
        assert!(idx.canonical_document_path("missing.pdf").is_none());
        // Section-anchored paths from read/section tool output (`path #ord · label`)
        // reach graph_upsert as `real.pdf#1`; strip the trailing `#<ord>` anchor so a
        // genuinely indexed document still resolves (the model copies the anchored path
        // exactly as the tool prints it). Hallucination guard is preserved below.
        assert_eq!(
            idx.canonical_document_path("real.pdf#1").as_deref(),
            Some("real.pdf")
        );
        assert_eq!(
            idx.canonical_document_path("real.pdf#12").as_deref(),
            Some("real.pdf")
        );
        assert_eq!(
            idx.canonical_document_path("kb-manual\\real.pdf#3")
                .as_deref(),
            Some("real.pdf")
        );
        assert_eq!(
            idx.canonical_document_path("real.pdf  #3").as_deref(),
            Some("real.pdf")
        );
        // Stripping the anchor does not resolve a document that is not indexed.
        assert!(idx.canonical_document_path("missing.pdf#2").is_none());
    }

    #[test]
    fn canonical_document_path_strips_glossary_owner_sigil() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            Chunk {
                doc_path: PathBuf::from("dir/manual.pdf"),
                location: "p.1".into(),
                file_type: "pdf".into(),
                text: "x".into(),
            },
            // A document whose real path legitimately begins with `@` must still win exactly.
            Chunk {
                doc_path: PathBuf::from("@types/react.md"),
                location: "S1".into(),
                file_type: "md".into(),
                text: "y".into(),
            },
        ])
        .unwrap();
        // glossary prints an ungrounded node's owner as `@<path>`; read must strip the sigil.
        assert_eq!(
            idx.canonical_document_path("@dir/manual.pdf").as_deref(),
            Some("dir/manual.pdf")
        );
        // Combined with a trailing section anchor (`@<path>#<ord>`), both are stripped.
        assert_eq!(
            idx.canonical_document_path("@dir/manual.pdf#350")
                .as_deref(),
            Some("dir/manual.pdf")
        );
        // A real path starting with `@` resolves to itself — the strip is a fallback only.
        assert_eq!(
            idx.canonical_document_path("@types/react.md").as_deref(),
            Some("@types/react.md")
        );
        // Stripping the sigil off a non-indexed path still returns None (hallucination guard).
        assert!(idx.canonical_document_path("@dir/missing.pdf").is_none());
    }

    #[test]
    fn canonical_document_path_folds_underscores_and_spaces() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        // Real filename is spaced (a PDF), but the corpus is full of underscore-named siblings,
        // so the model copies the path back with every space turned into an underscore.
        idx.write_chunks(&[Chunk {
            doc_path: PathBuf::from("Setup guide ACME PLC_v_1.pdf"),
            location: "p.1".into(),
            file_type: "pdf".into(),
            text: "x".into(),
        }])
        .unwrap();
        assert_eq!(
            idx.canonical_document_path("Setup_guide_ACME_PLC_v_1.pdf")
                .as_deref(),
            Some("Setup guide ACME PLC_v_1.pdf")
        );
        // The fold does not turn a genuinely different name into a false match.
        assert!(idx
            .canonical_document_path("Completely_different_document.pdf")
            .is_none());
    }

    #[test]
    fn iter_chunks_visits_every_stored_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            Chunk {
                doc_path: PathBuf::from("a.md"),
                location: "S1".into(),
                file_type: "md".into(),
                text: "alpha".into(),
            },
            Chunk {
                doc_path: PathBuf::from("a.md"),
                location: "S2".into(),
                file_type: "md".into(),
                text: "beta".into(),
            },
        ])
        .unwrap();
        let mut seen: Vec<(u64, String)> = Vec::new();
        idx.iter_chunks(|_path, ord, _ft, body| seen.push((ord, body.to_string())))
            .unwrap();
        seen.sort();
        assert_eq!(
            seen,
            vec![(1, "alpha".to_string()), (2, "beta".to_string())]
        );
    }

    #[test]
    fn location_ord_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let i = DocIndex::open_or_create(dir.path()).unwrap();
        i.write_chunks(&[crate::model::Chunk {
            doc_path: "d.md".into(),
            location: "A > B".into(),
            file_type: "md".into(),
            text: "x".into(),
        }])
        .unwrap();
        let n = i.ord_for_location("d.md", "A > B").unwrap().unwrap();
        assert_eq!(
            i.location_for_ord("d.md", n).unwrap().as_deref(),
            Some("A > B")
        );
        assert_eq!(i.ord_for_location("d.md", "missing").unwrap(), None);
    }

    #[test]
    fn search_filtered_scopes_by_glob_and_type() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            Chunk {
                doc_path: PathBuf::from("a/MODULE.pdf"),
                location: "p.1".into(),
                file_type: "pdf".into(),
                text: "hot cpu swap".into(),
            },
            Chunk {
                doc_path: PathBuf::from("b/Other.pdf"),
                location: "p.1".into(),
                file_type: "pdf".into(),
                text: "hot cpu swap".into(),
            },
            Chunk {
                doc_path: PathBuf::from("c/Notes.md"),
                location: "S1".into(),
                file_type: "md".into(),
                text: "hot cpu swap".into(),
            },
        ])
        .unwrap();

        let all = idx.search_filtered("swap", 10, None, None, None).unwrap();
        assert_eq!(all.len(), 3);
        // glob scopes to the matching path only
        let manual = idx
            .search_filtered("swap", 10, Some("*MODULE*"), None, None)
            .unwrap();
        assert_eq!(manual.len(), 1);
        assert!(manual[0].path.contains("MODULE"));
        // file_type scopes to md only
        let md = idx
            .search_filtered("swap", 10, None, Some("md"), None)
            .unwrap();
        assert_eq!(md.len(), 1);
        assert_eq!(md[0].file_type, "md");
        // recursive glob on nested paths
        idx.write_chunks(&[Chunk {
            doc_path: PathBuf::from("nested\\inner.pdf"),
            location: "p.1".into(),
            file_type: "pdf".into(),
            text: "hot cpu swap".into(),
        }])
        .unwrap();
        let rec = idx
            .search_filtered("swap", 10, Some("**/*.pdf"), None, None)
            .unwrap();
        assert!(rec.len() >= 2);
        assert!(rec.iter().any(|h| h.path.contains("inner.pdf")));
    }

    #[test]
    fn search_filtered_scope_ands_with_glob_and_narrows_to_one_document() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            Chunk {
                doc_path: PathBuf::from("docA.md"),
                location: "S1".into(),
                file_type: "md".into(),
                text: "hot cpu swap".into(),
            },
            Chunk {
                doc_path: PathBuf::from("docB.md"),
                location: "S1".into(),
                file_type: "md".into(),
                text: "hot cpu swap".into(),
            },
        ])
        .unwrap();

        // No scope: unchanged baseline — both documents hit.
        let all = idx.search_filtered("swap", 10, None, None, None).unwrap();
        assert_eq!(all.len(), 2);

        // scope=docA.md: only that document's hit remains.
        let scoped = idx
            .search_filtered("swap", 10, None, None, Some("docA.md"))
            .unwrap();
        assert_eq!(scoped.len(), 1);
        assert_eq!(scoped[0].path, "docA.md");

        // glob AND scope must both match — a glob that excludes docA.md leaves nothing, even
        // though scope alone would have matched it.
        let anded_out = idx
            .search_filtered("swap", 10, Some("**/docB.md"), None, Some("docA.md"))
            .unwrap();
        assert!(
            anded_out.is_empty(),
            "glob and scope must be ANDed: {anded_out:?}"
        );

        // A non-matching scope glob returns nothing.
        let none = idx
            .search_filtered("swap", 10, None, None, Some("nomatch*"))
            .unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn write_chunks_is_idempotent_no_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        // Write first version of d.md
        idx.write_chunks(&[chunk("d.md", "original content alpha")])
            .unwrap();
        // Overwrite with a different chunk for the same path
        idx.write_chunks(&[chunk("d.md", "updated content beta")])
            .unwrap();
        // Search should find the new content and NOT return duplicates
        let hits = idx.search("content", 10).unwrap();
        let d_hits: Vec<_> = hits.iter().filter(|h| h.path == "d.md").collect();
        assert_eq!(
            d_hits.len(),
            1,
            "expected exactly one hit for d.md, got {}",
            d_hits.len()
        );
        assert!(
            d_hits[0].snippet.contains("beta") || d_hits[0].snippet.contains("updated"),
            "expected updated content, got: {}",
            d_hits[0].snippet
        );
    }

    #[test]
    fn file_type_of_and_note_owner() {
        use crate::model::Chunk;
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            Chunk {
                doc_path: "doc.pdf".into(),
                location: "p.1".into(),
                file_type: "pdf".into(),
                text: "stub".into(),
            },
            Chunk {
                doc_path: "doc.pdf/limits.csp".into(),
                location: "note".into(),
                file_type: "note".into(),
                text: "D\n50".into(),
            },
        ])
        .unwrap();
        assert_eq!(idx.file_type_of("doc.pdf").unwrap().as_deref(), Some("pdf"));
        assert_eq!(
            idx.file_type_of("doc.pdf/limits.csp").unwrap().as_deref(),
            Some("note")
        );
        assert_eq!(idx.file_type_of("missing").unwrap(), None);
        assert_eq!(
            idx.note_owner("doc.pdf/limits.csp").unwrap().as_deref(),
            Some("doc.pdf")
        );
        // A corpus path (no `/` to split, or the path itself) has no note owner.
        assert_eq!(idx.note_owner("doc.pdf").unwrap(), None);
        // A nested note file: the longest indexed prefix wins.
        idx.write_chunks(&[Chunk {
            doc_path: "doc.pdf/sub/values.csp".into(),
            location: "note".into(),
            file_type: "note".into(),
            text: "x".into(),
        }])
        .unwrap();
        assert_eq!(
            idx.note_owner("doc.pdf/sub/values.csp").unwrap().as_deref(),
            Some("doc.pdf")
        );
    }

    #[cfg(feature = "notebook")]
    #[test]
    fn scan_delta_picks_up_and_drops_notes() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("doc.md"), b"# Doc\nplaceholder\n").unwrap();

        let manifest = Manifest::default();
        let d = scan_delta(dir.path(), &manifest).unwrap();
        assert!(!d.changed.is_empty(), "corpus changed: {d:?}");
        assert!(d.notes_changed.is_empty(), "no notes yet: {d:?}");

        // A note created outside `note()` (e.g. an external editor) shows up as notes_changed.
        let note_dir = dir.path().join(".glossa/notes/doc.md");
        fs::create_dir_all(&note_dir).unwrap();
        fs::write(note_dir.join("limits.csp"), b"D\n50\n63\n").unwrap();
        let d = scan_delta(dir.path(), &manifest).unwrap();
        assert!(
            d.notes_changed.contains(&"doc.md/limits.csp".to_string()),
            "{d:?}"
        );
        assert!(d.next.notes.contains_key("doc.md/limits.csp"), "{d:?}");

        // A saved note that disappears from disk lands in notes_removed.
        let mut m = Manifest::default();
        m.notes.insert(
            "doc.md/limits.csp".into(),
            file_sig(note_dir.join("limits.csp").as_path()).unwrap(),
        );
        fs::remove_file(note_dir.join("limits.csp")).unwrap();
        let d = scan_delta(dir.path(), &m).unwrap();
        assert!(
            d.notes_removed.contains(&"doc.md/limits.csp".to_string()),
            "{d:?}"
        );
    }

    #[cfg(feature = "notebook")]
    #[test]
    fn orphan_notes_lists_only_owner_less_notes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("live.md"), b"# Live\nbody\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let notes = dir.path().join(".glossa").join("notes");
        std::fs::create_dir_all(notes.join("live.md")).unwrap();
        std::fs::write(notes.join("live.md").join("n.csp"), b"kept").unwrap();
        std::fs::create_dir_all(notes.join("gone.md")).unwrap();
        std::fs::write(notes.join("gone.md").join("o.csp"), b"orphan").unwrap();

        let mut got = orphan_notes(dir.path()).unwrap();
        got.sort();
        assert_eq!(got, vec!["gone.md/o.csp".to_string()]);
    }

    #[cfg(feature = "notebook")]
    #[test]
    fn orphan_notes_at_does_not_flag_a_note_owned_by_a_secondary_root_doc() {
        // Regression: `orphan_notes` (single-root) walks `state_base` for corpus files. In
        // separated state-dir mode `state_base` holds none of the real corpus files, so it saw
        // an empty file set and flagged EVERY note as orphaned — a non-dry-run `kb prune` would
        // then delete every note. `orphan_notes_at` must see files from every root.
        use crate::root::Root;
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(a.path().join("live.md"), b"# Live\nbody\n").unwrap();
        std::fs::write(b.path().join("live.md"), b"# Live B\nbody\n").unwrap();
        let roots = vec![
            Root {
                label: "A".into(),
                path: a.path().into(),
            },
            Root {
                label: "B".into(),
                path: b.path().into(),
            },
        ];
        index_dir_at(&roots, state.path(), true).unwrap();

        // A note whose owner doc lives under the SECONDARY root ("B/live.md").
        let notes = state.path().join(".glossa").join("notes");
        std::fs::create_dir_all(notes.join("B").join("live.md")).unwrap();
        std::fs::write(notes.join("B").join("live.md").join("n.csp"), b"kept").unwrap();
        // A genuinely orphaned note, for contrast.
        std::fs::create_dir_all(notes.join("B").join("gone.md")).unwrap();
        std::fs::write(notes.join("B").join("gone.md").join("o.csp"), b"orphan").unwrap();

        let mut got = orphan_notes_at(&roots, state.path()).unwrap();
        got.sort();
        assert_eq!(
            got,
            vec!["B/gone.md/o.csp".to_string()],
            "the note owned by B/live.md (a real, live secondary-root doc) must NOT be flagged \
             orphan: {got:?}"
        );
    }

    #[cfg(feature = "notebook")]
    #[test]
    fn note_is_dropped_when_owner_removed_and_restored_when_owner_returns() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("doc.md"), b"# Doc\nbody\n").unwrap();
        // An unrelated second doc keeps the root from going to ZERO files when `doc.md` is removed
        // below — Task 8's empty-mount guard (`scan_delta_at`) holds a previously-populated root's
        // docs when a pass finds nothing at all under it, which would otherwise mask this test's
        // targeted single-doc removal as a suspicious whole-root wipe.
        fs::write(dir.path().join("keepalive.md"), b"# Keepalive\nstays put\n").unwrap();
        index_dir(dir.path(), true).unwrap();

        // A note mirrored under .glossa/notes/doc.md/limits.csp, created and committed while
        // the owner still exists — so it is a genuinely live, persisted note, not just a
        // one-off delta.
        let notes_dir = dir.path().join(".glossa").join("notes").join("doc.md");
        fs::create_dir_all(&notes_dir).unwrap();
        fs::write(notes_dir.join("limits.csp"), b"sentinel note text").unwrap();
        index_dir(dir.path(), false).unwrap();

        let m = crate::index::manifest::Manifest::load(dir.path());
        assert!(
            m.notes.contains_key("doc.md/limits.csp"),
            "note committed into the persisted manifest"
        );
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(
            idx.search("sentinel", 10)
                .unwrap()
                .iter()
                .any(|h| h.path == "doc.md/limits.csp"),
            "note is searchable while owner exists"
        );

        // Remove the owner document; the note file stays on disk. Reindexing must drop the
        // now-orphaned chunk from the live (persisted, previously-committed) index.
        fs::remove_file(dir.path().join("doc.md")).unwrap();
        index_dir(dir.path(), false).unwrap();

        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(
            !idx.search("sentinel", 10)
                .unwrap()
                .iter()
                .any(|h| h.path == "doc.md/limits.csp"),
            "orphan chunk removed from the index"
        );
        assert!(
            notes_dir.join("limits.csp").is_file(),
            "orphan file stays on disk (reversible)"
        );
        let m2 = crate::index::manifest::Manifest::load(dir.path());
        assert!(
            !m2.notes.contains_key("doc.md/limits.csp"),
            "orphan not tracked as a live note in the persisted manifest"
        );

        // Restore the owner → note is picked up and re-indexed again.
        fs::write(dir.path().join("doc.md"), b"# Doc\nbody\n").unwrap();
        index_dir(dir.path(), false).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(
            idx.search("sentinel", 10)
                .unwrap()
                .iter()
                .any(|h| h.path == "doc.md/limits.csp"),
            "note re-indexed once owner returns"
        );
    }

    #[cfg(feature = "notebook")]
    #[test]
    fn reindex_dirs_locked_purges_orphaned_note_on_owner_removal() {
        // Regression for Finding 2: deleting an owner document bumps only its `c:` corpus dir —
        // the `.glossa/notes/<doc>` tree is untouched — so `notes_touched` computed from the
        // dir-map diff alone stays false. A SCOPED `reindex_dirs_locked` pass must still purge
        // the now-orphaned note's chunk, matching what a full `index_dir` already does (see
        // `note_is_dropped_when_owner_removed_and_restored_when_owner_returns` above).
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("doc.md"), b"# Doc\nbody\n").unwrap();
        // An unrelated second doc keeps the root from going to ZERO files when `doc.md` is removed
        // below — Task 8's empty-mount guard (root-level check in `scan_scoped_delta_at`) holds a
        // previously-populated root's docs when a pass would remove EVERY doc under it, which would
        // otherwise mask this test's targeted single-doc removal as a suspicious whole-root wipe.
        fs::write(dir.path().join("keepalive.md"), b"# Keepalive\nstays put\n").unwrap();
        index_dir(dir.path(), true).unwrap();

        let notes_dir = dir.path().join(".glossa").join("notes").join("doc.md");
        fs::create_dir_all(&notes_dir).unwrap();
        fs::write(notes_dir.join("limits.csp"), b"sentinel note text").unwrap();
        index_dir(dir.path(), false).unwrap();

        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(
            idx.search("sentinel", 10)
                .unwrap()
                .iter()
                .any(|h| h.path == "doc.md/limits.csp"),
            "note is searchable while owner exists"
        );

        // Remove the owner document, then run the same scoped-reindex path `freshen_blocking`
        // would run: diff the dir-mtime maps and hand the classification to `reindex_dirs_locked`.
        let before = dir_mtime_map(dir.path()).unwrap();
        fs::remove_file(dir.path().join("doc.md")).unwrap();
        let after = dir_mtime_map(dir.path()).unwrap();
        let (changed, added, removed) = diff_corpus_dirs(&before, &after);
        let notes_touched = notes_submap_changed(&before, &after);
        assert!(
            !notes_touched,
            "removing the owner doc must NOT bump the notes dir (that's the bug this test guards against)"
        );
        {
            let _l = crate::index::lock::try_index_lock(dir.path()).unwrap();
            reindex_dirs_locked(
                dir.path(),
                &after,
                &changed,
                &added,
                &removed,
                notes_touched,
            )
            .unwrap();
        }

        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(
            !idx.search("sentinel", 10)
                .unwrap()
                .iter()
                .any(|h| h.path == "doc.md/limits.csp"),
            "orphan chunk removed from the index after a SCOPED reindex, matching a full index_dir"
        );
        let m = crate::index::manifest::Manifest::load(dir.path());
        assert!(
            !m.notes.contains_key("doc.md/limits.csp"),
            "orphan not tracked as a live note in the persisted manifest after a scoped reindex"
        );
    }

    #[cfg(feature = "notebook")]
    #[test]
    fn index_dir_indexes_removes_and_reindexes_notes() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("doc.md"), b"# Doc\nplaceholder\n").unwrap();
        index_dir(dir.path(), true).unwrap();

        let note_dir = dir.path().join(".glossa/notes/doc.md");
        fs::create_dir_all(&note_dir).unwrap();
        fs::write(note_dir.join("limits.csp"), b"D\n50\n63\n").unwrap();
        index_dir(dir.path(), false).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(
            idx.search("63", 10)
                .unwrap()
                .iter()
                .any(|h| h.path == "doc.md/limits.csp"),
            "note indexed by delta"
        );

        // Deleting the note file removes its chunk.
        fs::remove_file(note_dir.join("limits.csp")).unwrap();
        index_dir(dir.path(), false).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(
            !idx.search("63", 10)
                .unwrap()
                .iter()
                .any(|h| h.path == "doc.md/limits.csp"),
            "note chunk removed"
        );

        // Force reindex keeps notes present on disk.
        fs::write(note_dir.join("limits.csp"), b"D\n50\n63\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(
            idx.search("63", 10)
                .unwrap()
                .iter()
                .any(|h| h.path == "doc.md/limits.csp"),
            "force reindex preserves notes"
        );
    }

    #[test]
    #[cfg(feature = "notebook")]
    fn reindex_note_locked_picks_up_an_edit() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("doc.md"), b"# Doc\nbody\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let notes = dir.path().join(".glossa").join("notes").join("doc.md");
        std::fs::create_dir_all(&notes).unwrap();
        std::fs::write(notes.join("n.csp"), b"oldtoken").unwrap();
        // First index of the note.
        {
            let _l = crate::index::lock::try_index_lock(dir.path()).unwrap();
            reindex_note_locked(dir.path(), "doc.md/n.csp").unwrap();
        }
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(idx
            .search("oldtoken", 10)
            .unwrap()
            .iter()
            .any(|h| h.path == "doc.md/n.csp"));
        // Edit the note file on disk, reindex → new content searchable, old gone, manifest updated.
        std::fs::write(notes.join("n.csp"), b"newtoken sentinel").unwrap();
        {
            let _l = crate::index::lock::try_index_lock(dir.path()).unwrap();
            reindex_note_locked(dir.path(), "doc.md/n.csp").unwrap();
        }
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(
            idx.search("sentinel", 10)
                .unwrap()
                .iter()
                .any(|h| h.path == "doc.md/n.csp"),
            "new content searchable"
        );
        assert!(
            idx.search("oldtoken", 10).unwrap().is_empty(),
            "old content gone"
        );
        assert_eq!(
            crate::index::manifest::Manifest::load(dir.path())
                .notes
                .get("doc.md/n.csp"),
            Some(&file_sig(&notes.join("n.csp")).unwrap()),
            "manifest.notes sig updated"
        );
        // Gone file → Ok(None), manifest untouched.
        std::fs::remove_file(notes.join("n.csp")).unwrap();
        {
            let _l = crate::index::lock::try_index_lock(dir.path()).unwrap();
            assert!(reindex_note_locked(dir.path(), "doc.md/n.csp")
                .unwrap()
                .is_none());
        }
        assert!(
            crate::index::manifest::Manifest::load(dir.path())
                .notes
                .contains_key("doc.md/n.csp"),
            "manifest.notes untouched on missing file (deletion handled by freshen)"
        );
    }
}

#[cfg(test)]
mod retry_tests {
    use super::*;
    use std::cell::Cell;

    fn perm_denied() -> TantivyError {
        // Windows surfaces "Access is denied. (os error 5)" as ErrorKind::PermissionDenied.
        std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "Access is denied. (os error 5)",
        )
        .into()
    }

    #[test]
    fn with_writer_retry_recovers_from_transient_permission_denied() {
        let calls = Cell::new(0u32);
        let out: tantivy::Result<u32> = with_writer_retry(|| {
            let n = calls.get() + 1;
            calls.set(n);
            if n < 3 {
                Err(perm_denied())
            } else {
                Ok(n)
            }
        });
        assert_eq!(
            out.unwrap(),
            3,
            "succeeds after the transient failures clear"
        );
        assert_eq!(calls.get(), 3, "retried both permission-denied failures");
    }

    #[test]
    fn with_writer_retry_recovers_from_non_permission_transient_io() {
        // Windows surfaces a locked / partially-read `meta.json` through IO kinds other than
        // PermissionDenied (Other, or a raw OS error). The widened predicate retries every IO kind,
        // so such a transient failure must also clear rather than propagate on the first attempt.
        let calls = Cell::new(0u32);
        let out: tantivy::Result<u32> = with_writer_retry(|| {
            let n = calls.get() + 1;
            calls.set(n);
            if n < 3 {
                Err(std::io::Error::other("sharing violation").into())
            } else {
                Ok(n)
            }
        });
        assert_eq!(
            out.unwrap(),
            3,
            "a non-permission transient IO error is retried"
        );
        assert_eq!(calls.get(), 3, "retried both non-permission IO failures");
    }

    #[test]
    fn with_writer_retry_propagates_non_transient_error() {
        // A structural error (not IO/lock/open) can never be fixed by retrying — it must propagate
        // on the FIRST attempt, so the bounded retry can never mask a real permanent failure.
        let calls = Cell::new(0u32);
        let out: tantivy::Result<u32> = with_writer_retry(|| {
            calls.set(calls.get() + 1);
            Err(TantivyError::SchemaError("bad schema".to_string()))
        });
        assert!(
            out.is_err(),
            "a non-transient (structural) error propagates"
        );
        assert_eq!(calls.get(), 1, "a non-transient error is not retried");
    }

    fn transient_io_err() -> anyhow::Error {
        anyhow::Error::from(std::io::Error::from(std::io::ErrorKind::TimedOut))
    }

    fn permanent_io_err() -> anyhow::Error {
        anyhow::Error::from(std::io::Error::from(std::io::ErrorKind::InvalidData))
    }

    #[test]
    fn with_read_retry_succeeds_after_k_transient() {
        let calls = Cell::new(0u32);
        let k = 2u32; // fails twice, succeeds on the 3rd call
        let out: anyhow::Result<u32> = with_read_retry(3, 0, || {
            let n = calls.get() + 1;
            calls.set(n);
            if n <= k {
                Err(transient_io_err())
            } else {
                Ok(n)
            }
        });
        assert_eq!(
            out.unwrap(),
            k + 1,
            "succeeds once the transient failures clear"
        );
        assert_eq!(calls.get(), k + 1, "retried exactly the transient failures");
    }

    #[test]
    fn with_read_retry_does_not_retry_permanent() {
        let calls = Cell::new(0u32);
        let out: anyhow::Result<u32> = with_read_retry(3, 0, || {
            calls.set(calls.get() + 1);
            Err(permanent_io_err())
        });
        assert!(out.is_err(), "a permanent error propagates");
        assert_eq!(calls.get(), 1, "a permanent error is not retried");
    }

    #[test]
    fn with_read_retry_exhausts_and_returns_last_transient() {
        let calls = Cell::new(0u32);
        let max_retries = 3u32;
        let out: anyhow::Result<u32> = with_read_retry(max_retries, 0, || {
            calls.set(calls.get() + 1);
            Err(transient_io_err())
        });
        assert!(out.is_err(), "exhausted retries still return an error");
        assert!(
            is_transient(&out.unwrap_err()),
            "the final error is still the transient one, not something else"
        );
        assert_eq!(
            calls.get(),
            max_retries + 1,
            "one initial attempt plus max_retries retries"
        );
    }

    #[test]
    fn with_read_retry_zero_retries_calls_once() {
        let calls = Cell::new(0u32);
        let out: anyhow::Result<u32> = with_read_retry(0, 0, || {
            calls.set(calls.get() + 1);
            Err(transient_io_err())
        });
        assert!(out.is_err(), "no retries means the first error propagates");
        assert_eq!(calls.get(), 1, "called exactly once when max_retries is 0");
    }
}

#[cfg(test)]
mod classify_tests {
    use super::*;

    #[test]
    fn is_transient_classifies_errno_table() {
        // ErrorKind-level transient signatures.
        assert!(is_transient(&anyhow::Error::from(std::io::Error::from(
            std::io::ErrorKind::TimedOut
        ))));
        // Errno-level transient signatures (ESTALE/EIO map to `Uncategorized` in current Rust, so
        // these are ONLY reachable via `raw_os_error()`, not `kind()`). `raw_os_error()` numbers are
        // platform-specific — the classifier's errno table is unix-only (see `is_transient` doc
        // comment) — so these assertions are unix-only too. Raw code 5 in particular is POSIX EIO on
        // unix but Win32 `ERROR_ACCESS_DENIED` on Windows (the SAME numeric code as a genuine
        // permission failure, which must stay permanent), so its Windows counterpart asserts `false`
        // instead of skipping the case outright.
        #[cfg(unix)]
        assert!(is_transient(&anyhow::Error::from(
            std::io::Error::from_raw_os_error(116) // ESTALE
        )));
        #[cfg(unix)]
        assert!(is_transient(&anyhow::Error::from(
            std::io::Error::from_raw_os_error(5) // EIO
        )));
        #[cfg(unix)]
        assert!(is_transient(&anyhow::Error::from(
            std::io::Error::from_raw_os_error(101) // ENETUNREACH
        )));
        #[cfg(windows)]
        assert!(!is_transient(&anyhow::Error::from(
            std::io::Error::from_raw_os_error(5) // ERROR_ACCESS_DENIED, not EIO, on this platform
        )));
        // Permanent io::Error kinds never retry.
        assert!(!is_transient(&anyhow::Error::from(std::io::Error::from(
            std::io::ErrorKind::NotFound
        ))));
        assert!(!is_transient(&anyhow::Error::from(std::io::Error::from(
            std::io::ErrorKind::PermissionDenied
        ))));
        assert!(!is_transient(&anyhow::Error::from(std::io::Error::from(
            std::io::ErrorKind::InvalidData
        ))));
        // A non-io (parse/extractor) error carries no io::Error anywhere in its chain → permanent.
        assert!(!is_transient(&anyhow::anyhow!("bad CFB header")));
        // Wrapped chain: `.context()` must not hide the io::Error from `.chain()`. Uses a named
        // (cross-platform) `ErrorKind` rather than a raw errno — this case is about chain-walking,
        // not platform-specific errno decoding.
        let wrapped =
            anyhow::Error::from(std::io::Error::from(std::io::ErrorKind::ConnectionReset))
                .context("reading X");
        assert!(is_transient(&wrapped));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-guard for the failure-counter fields on `IndexStats` (Task 9 observability): the real
    /// behavioral assertions that populate them live in the Task 5/6 transient/permanent tests.
    #[test]
    fn index_stats_has_failure_counters() {
        let stats = IndexStats::default();
        assert_eq!(stats.transient_failures, 0);
        assert_eq!(stats.permanent_skips, 0);
        assert_eq!(stats.empty_mount_holds, 0);
    }

    /// Set `dir`'s mtime to a fixed point in the future so the freshen dir-mtime gate sees a value
    /// STRICTLY AFTER any previously recorded one — deterministic where a `thread::sleep` + wall-clock
    /// bump flakes on a coarse-granularity Windows FS (the new file may not distinctly re-bump the
    /// parent-dir mtime, so `dir_mtime_map == dirsig` still holds and the gate never re-opens). Only
    /// the directory mtime is set; the new file's own mtime — and thus the FRESH_WINDOW mid-copy
    /// guard, which keys on the FILE signature — is left untouched.
    fn bump_dir_mtime_future(dir: &Path) {
        let future = filetime::FileTime::from_system_time(
            std::time::SystemTime::now() + std::time::Duration::from_secs(30),
        );
        filetime::set_file_mtime(dir, future).expect("set dir mtime");
    }

    #[test]
    fn creates_then_reopens_index() {
        let dir = tempfile::tempdir().unwrap();
        let a = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(a.index.schema().get_field("body").is_ok());
        assert!(a.index.schema().get_field("body_trigrams").is_ok());
        drop(a);
        let b = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(b.index.schema().get_field("path").is_ok());
    }

    #[test]
    fn iter_chunks_trigram_candidates_and_semantics() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let mk = |p: &str, ord: u64, t: &str| crate::model::Chunk {
            doc_path: p.into(),
            location: format!("S{ord}"),
            file_type: "md".into(),
            text: t.into(),
        };
        idx.write_chunks(&[
            mk("a.md", 1, "device registration completed"),
            mk("b.md", 1, "unrelated text"),
        ])
        .unwrap();
        let mut paths = Vec::new();
        idx.iter_chunks_trigram_candidates(&["reg".to_string()], |path, _ord, _ft, _body| {
            paths.push(path.to_string());
        })
        .unwrap();
        assert_eq!(paths, vec!["a.md".to_string()]);
    }

    #[test]
    fn index_schema_migration_rebuilds() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# Hi\nhello world\n").unwrap();
        let mut old = Manifest::default();
        old.files.insert(
            "a.md".into(),
            FileSig {
                mtime_secs: 1,
                size: 10,
            },
        );
        old.index_schema_version = 1;
        old.save(dir.path()).unwrap();
        index_dir(dir.path(), false).unwrap();
        let loaded = Manifest::load(dir.path());
        assert_eq!(loaded.index_schema_version, INDEX_SCHEMA_VERSION);
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(idx.index.schema().get_field("body_trigrams").is_ok());
        let hits = idx.search("hello", 5).unwrap();
        assert!(!hits.is_empty());
    }

    #[test]
    fn unsettled_dirs_flags_a_file_that_changed_after_indexing() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"one").unwrap();
        let indexed_sig = file_sig(&dir.path().join("a.md")).unwrap();
        // A settled file — current sig equals what we indexed — is NOT flagged.
        assert!(unsettled_dirs(dir.path(), &[("a.md".to_string(), indexed_sig)]).is_empty());
        // The file grows after we recorded `indexed_sig` (mid-write / partial copy). Its dir is
        // now unsettled, so its dirsig entry must be held back.
        std::thread::sleep(std::time::Duration::from_millis(10));
        std::fs::write(dir.path().join("a.md"), b"one two three four").unwrap();
        let u = unsettled_dirs(dir.path(), &[("a.md".to_string(), indexed_sig)]);
        assert!(u.contains("c:"), "root dir flagged unsettled: {u:?}");
        // A file in a subdir flags its own dir key, not the root.
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub").join("b.md"), b"x").unwrap();
        let sub_sig = file_sig(&dir.path().join("sub").join("b.md")).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        std::fs::write(dir.path().join("sub").join("b.md"), b"xyz longer").unwrap();
        let u2 = unsettled_dirs(
            dir.path(),
            &[(format!("sub{}b.md", std::path::MAIN_SEPARATOR), sub_sig)],
        );
        assert!(
            u2.contains("c:sub") && !u2.contains("c:"),
            "only sub flagged: {u2:?}"
        );
    }

    #[test]
    fn settled_dirsig_holds_back_unsettled_dirs() {
        let mut cur = BTreeMap::new();
        cur.insert("c:".to_string(), 100u128);
        cur.insert("c:sub".to_string(), 200u128);
        cur.insert("c:new".to_string(), 300u128); // a directory added this pass
        let mut stored = BTreeMap::new();
        stored.insert("c:".to_string(), 50u128);
        stored.insert("c:sub".to_string(), 150u128);
        let unsettled: std::collections::HashSet<String> =
            ["c:sub".to_string(), "c:new".to_string()]
                .into_iter()
                .collect();
        let out = settled_dirsig(&cur, &stored, &unsettled);
        assert_eq!(out.get("c:"), Some(&100), "settled dir advances to current");
        assert_eq!(
            out.get("c:sub"),
            Some(&150),
            "unsettled dir keeps its stored value"
        );
        assert_eq!(
            out.get("c:new"),
            None,
            "unsettled newly-added dir is dropped"
        );
    }

    #[test]
    fn freshen_blocking_picks_up_new_file_and_is_noop_when_fresh() {
        use std::time::Duration;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nhello\n").unwrap();
        freshen_blocking(dir.path(), Duration::from_secs(3)).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(idx
            .search("hello", 10)
            .unwrap()
            .iter()
            .any(|h| h.path.ends_with("a.md")));

        // Nothing changed → no work, map already matches.
        let cur = dir_mtime_map(dir.path()).unwrap();
        freshen_blocking(dir.path(), Duration::from_secs(3)).unwrap();
        assert_eq!(read_dirsig(dir.path()), Some(cur));

        // Add a file → next freshen makes it searchable. Bump the dir mtime deterministically
        // (strictly after the settled state) so the gate reliably re-opens on a coarse-granularity FS.
        std::fs::write(dir.path().join("b.md"), b"# B\nworld\n").unwrap();
        bump_dir_mtime_future(dir.path());
        freshen_blocking(dir.path(), Duration::from_secs(3)).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(idx
            .search("world", 10)
            .unwrap()
            .iter()
            .any(|h| h.path.ends_with("b.md")));
    }

    // Regression: a file copied into the corpus can be seen by a freshen mid-copy — its PARTIAL
    // content indexed and its directory settled. When the copy finishes, the rest lands as an
    // in-place write, which does NOT re-bump the parent dir's mtime, so the `dir_mtime_map == dirsig`
    // gate would stay shut and the finalized content never reindex. Fixed by holding a dir unsettled
    // for one more pass when a file we just indexed is younger than FRESH_WINDOW_SECS (see
    // reindex_dirs_locked), forcing the next freshen to re-stat it. (877a9d6's re-stat only caught a
    // change DURING the pass, not one landing AFTER it.) The CLI `scan_delta` path re-stats every
    // file and never had this gap.
    #[test]
    fn freshen_gate_misses_content_finalized_after_a_settled_index() {
        use std::time::Duration;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nhello\n").unwrap();
        freshen_blocking(dir.path(), Duration::from_secs(3)).unwrap();

        // Mid-copy: a freshen sees b.md with only PARTIAL content, indexes it, and settles the dir.
        std::thread::sleep(Duration::from_millis(150));
        std::fs::write(dir.path().join("b.md"), b"partialonly").unwrap();
        freshen_blocking(dir.path(), Duration::from_secs(3)).unwrap();

        // The copy completes: b.md's real content lands (in-place content write — no dir-mtime bump).
        // Sleep so the finalized write gets a distinctly later mtime than the partial one — the gate's
        // re-stat compares timestamps, and a coarse-resolution FS (the CI windows runner) otherwise
        // can't tell the two writes apart, which flaked this test.
        std::thread::sleep(Duration::from_millis(150));
        std::fs::write(dir.path().join("b.md"), b"# B\nworldfinished\n").unwrap();

        // A freshen must still pick up the finalized content.
        freshen_blocking(dir.path(), Duration::from_secs(3)).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(
            idx.search("worldfinished", 10)
                .unwrap()
                .iter()
                .any(|h| h.path.ends_with("b.md")),
            "content finalized after the dir was settled (mid-copy) must still be indexed — the \
             dir-mtime gate misses it because a content write does not re-bump the dir mtime"
        );
    }

    #[test]
    fn freshen_blocking_scoped_picks_up_a_new_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nalpha\n").unwrap();
        freshen_blocking(dir.path(), std::time::Duration::from_secs(3)).unwrap();
        // Add a file, then freshen — it must become searchable (via the scoped path). Set the dir
        // mtime deterministically past the settled state instead of racing the FS clock with a sleep.
        std::fs::write(dir.path().join("b.md"), b"# B\nbravo\n").unwrap();
        bump_dir_mtime_future(dir.path());
        freshen_blocking(dir.path(), std::time::Duration::from_secs(3)).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(idx
            .search("bravo", 10)
            .unwrap()
            .iter()
            .any(|h| h.path.ends_with("b.md")));
        // b.md was just written, so the pass that indexed it holds its dir unsettled for one more
        // freshen (mid-copy guard). A confirming freshen — which re-stats b.md, finds it stable, and
        // reindexes nothing — settles the dir, and dirsig then matches the current tree.
        freshen_blocking(dir.path(), std::time::Duration::from_secs(3)).unwrap();
        assert_eq!(
            read_dirsig(dir.path()),
            Some(dir_mtime_map(dir.path()).unwrap())
        );
    }

    /// Task 6: the SAME three-outcome handling Task 5 gave `index_dir_at_locked` (CLI path), now on
    /// `reindex_dirs_at_locked` (the PRIMARY prod freshen path, reached via `freshen_blocking`). A
    /// transient READ failure that survives `with_read_retry`'s full budget must revert `next.files`
    /// to the OLD sig (not the freshly-observed one) — the prior doc stays searchable — AND hold the
    /// file's dir back at its STORED (pre-pass) dirsig value rather than the just-observed `cur`, so
    /// `freshen_blocking`'s fast path (`read_dirsig == cur -> return`) sees a mismatch next time and
    /// re-diffs instead of silently treating the dir as settled forever (spec R-B3, the BLOCKER).
    #[test]
    fn reindex_dirs_at_locked_transient_holds_dir_and_preserves_doc() {
        use std::time::Duration;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nalpha term\n").unwrap();
        freshen_blocking(dir.path(), Duration::from_secs(3)).unwrap();
        let old = *Manifest::load(dir.path()).files.get("a.md").unwrap();
        let stored_before = read_dirsig(dir.path()).unwrap();

        // A different SIZE (not just mtime) guarantees the diff sees this as modified regardless of
        // the mtime's whole-second granularity. In-place content edits don't bump the parent dir's
        // real mtime (see `dir_mtime_map`'s doc comment), so bump it ourselves to force
        // `freshen_blocking` off its fast path and into a real `reindex_dirs_at_locked` pass.
        std::fs::write(dir.path().join("a.md"), b"# A\nalpha term changed\n").unwrap();
        bump_dir_mtime_future(dir.path());
        let abs = abs_root(dir.path()).join("a.md");
        read_fault::arm_read(&abs, read_retries() + 1, std::io::ErrorKind::TimedOut, None);

        let stats = freshen_blocking(dir.path(), Duration::from_secs(3)).unwrap();
        read_fault::clear();

        assert_eq!(
            stats.transient_failures, 1,
            "the retry-exhausted read counts as a transient failure"
        );
        let manifest = Manifest::load(dir.path());
        assert_eq!(
            manifest.files.get("a.md"),
            Some(&old),
            "reverted to the OLD sig, not the freshly-observed (changed) one"
        );

        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(
            idx.search("alpha", 10)
                .unwrap()
                .iter()
                .any(|h| h.path.ends_with("a.md")),
            "the prior doc must still be searchable"
        );

        let dirsig = read_dirsig(dir.path()).unwrap();
        assert_eq!(
            dirsig.get("c:"),
            stored_before.get("c:"),
            "held at the STORED (old) dirsig value, not the just-observed `cur` — so the next \
             freshen's fast path sees a mismatch and re-diffs"
        );
    }

    /// Task 6 regression proof: once the transient fault clears, the NEXT `freshen_blocking` pass
    /// must actually re-diff the held-back dir (not stay stuck forever) and successfully index the
    /// file this time — proving the revert-then-hold-dirsig loop above is a HOLD, not a permanent
    /// wedge. This is the end-to-end path the review flagged as untested on Task 5's CLI twin.
    #[test]
    fn reindex_dirs_at_locked_transient_then_recovers_on_next_freshen() {
        use std::time::Duration;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nalpha term\n").unwrap();
        freshen_blocking(dir.path(), Duration::from_secs(3)).unwrap();

        std::fs::write(dir.path().join("a.md"), b"# A\nalpha term changed\n").unwrap();
        bump_dir_mtime_future(dir.path());
        let abs = abs_root(dir.path()).join("a.md");
        read_fault::arm_read(&abs, read_retries() + 1, std::io::ErrorKind::TimedOut, None);
        let s1 = freshen_blocking(dir.path(), Duration::from_secs(3)).unwrap();
        read_fault::clear();
        assert_eq!(
            s1.transient_failures, 1,
            "sanity: the fault fires on the first pass"
        );

        // Fault cleared. `cur` (the future-bumped dir mtime) still doesn't match the held-back
        // `stored` dirsig, so this pass must re-diff the dir (not short-circuit on the fast path)
        // and pick the file's real (unfaulted) read back up.
        let s2 = freshen_blocking(dir.path(), Duration::from_secs(3)).unwrap();
        assert_eq!(s2.transient_failures, 0, "the fault is cleared this pass");
        assert_eq!(s2.added, 1, "the file is re-indexed once the fault clears");

        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(
            idx.search("changed", 10)
                .unwrap()
                .iter()
                .any(|h| h.path.ends_with("a.md")),
            "the updated content is now searchable"
        );

        // A freshly-(re)indexed file holds its dir unsettled for one more pass (the FRESH_WINDOW
        // mid-copy guard, same as `freshen_blocking_scoped_picks_up_a_new_file`) — a confirming
        // freshen settles it, proving the dirsig genuinely advances rather than staying wedged.
        freshen_blocking(dir.path(), Duration::from_secs(3)).unwrap();
        assert_eq!(
            read_dirsig(dir.path()),
            Some(dir_mtime_map(dir.path()).unwrap()),
            "dirsig advances to the current tree now that the dir has settled"
        );
    }

    /// Task 6: a PERMANENT failure (bad CFB header, no `io::Error` in the chain) keeps the CURRENT
    /// sig in `next.files` — the file is treated as settled/unchanged and never retried, matching
    /// `index_dir_at_locked`'s (Task 5) behavior on the same fixture, now on the freshen path.
    #[test]
    fn reindex_dirs_at_locked_permanent_records_sig_and_is_not_retried() {
        use std::time::Duration;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("ok.md"), b"# T\nhello world\n").unwrap();
        std::fs::write(
            dir.path().join("bad.doc"),
            b"this is not a real CFB .doc file",
        )
        .unwrap();

        let s1 = freshen_blocking(dir.path(), Duration::from_secs(3)).unwrap();
        assert_eq!(
            s1.permanent_skips, 1,
            "the corrupt .doc is a permanent skip"
        );
        assert!(
            Manifest::load(dir.path()).files.contains_key("bad.doc"),
            "its sig IS recorded, so it's treated as settled"
        );

        // Change an UNRELATED file so this pass actually re-runs the indexing loop — bad.doc's
        // unchanged sig must keep it out of `delta.changed` entirely, so it's never re-attempted.
        std::fs::write(dir.path().join("ok.md"), b"# T\nhello world again\n").unwrap();
        bump_dir_mtime_future(dir.path());
        let s2 = freshen_blocking(dir.path(), Duration::from_secs(3)).unwrap();
        assert_eq!(
            s2.permanent_skips, 0,
            "an unchanged permanently-bad file is not re-attempted next pass"
        );
    }

    /// Task 10 (D1) — the headline network-scale fix: an already-elapsed `deadline` must stop the
    /// scoped per-dir walk BETWEEN dirs (not mid-dir, not never), serve the current index (0 dirs
    /// indexed this pass), and hold back every dir it didn't reach so the next (unbounded) pass
    /// resumes exactly there instead of losing or half-applying the change.
    #[test]
    fn reindex_deadline_serves_stale_and_holds_remaining_dirs() {
        use crate::root::Root;
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(a.path().join("a.md"), "alpha body").unwrap();
        std::fs::write(b.path().join("b.md"), "beta body").unwrap();
        let roots = vec![
            Root {
                label: "A".into(),
                path: a.path().into(),
            },
            Root {
                label: "B".into(),
                path: b.path().into(),
            },
        ];
        index_dir_at(&roots, state.path(), true).unwrap();
        let stored = read_dirsig(state.path()).unwrap();

        // Change BOTH roots so their dir keys both land in `changed` — a single-dir change would
        // never prove "remaining dirs held back", only "the one dir held back".
        std::fs::write(a.path().join("a.md"), "alpha body changed, now much longer").unwrap();
        std::fs::write(b.path().join("b.md"), "beta body changed, now much longer").unwrap();
        bump_dir_mtime_future(a.path());
        bump_dir_mtime_future(b.path());

        let cur = dir_mtime_map_at(&roots, state.path()).unwrap();
        let (changed, added, removed) = diff_corpus_dirs(&stored, &cur);
        assert_eq!(
            changed.len(),
            2,
            "sanity: both roots' dirs changed: {changed:?}"
        );

        // Already-elapsed deadline: the loop's per-dir check must fire before the FIRST dir is
        // even resolved/scanned.
        let deadline = Some(std::time::Instant::now());
        let stats = reindex_dirs_at_locked(
            &roots,
            state.path(),
            &cur,
            &changed,
            &added,
            &removed,
            false,
            deadline,
            1,
        )
        .unwrap();
        assert_eq!(
            stats.added, 0,
            "deadline already elapsed before the walk started — nothing indexed this pass: {stats:?}"
        );

        let dirsig = read_dirsig(state.path()).unwrap();
        assert_eq!(
            dirsig.get("c:A:"),
            stored.get("c:A:"),
            "root A held back at the STORED value, not the just-observed one"
        );
        assert_eq!(
            dirsig.get("c:B:"),
            stored.get("c:B:"),
            "root B held back at the STORED value, not the just-observed one"
        );

        // The manifest content is UNCHANGED (serve-stale: never partially-deletes) — both old
        // bodies are still what's on record and still searchable.
        let idx = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert!(
            idx.search("alpha body", 10)
                .unwrap()
                .iter()
                .any(|h| h.path.ends_with("A/a.md")),
            "the OLD content is still searchable — the timed-out pass didn't delete it"
        );

        // A follow-up UNBOUNDED pass (as the maintenance loop's next freshen would run) resumes
        // and finishes both dirs the first pass held back.
        let stats2 = reindex_dirs_at_locked(
            &roots,
            state.path(),
            &cur,
            &changed,
            &added,
            &removed,
            false,
            None,
            3,
        )
        .unwrap();
        assert_eq!(
            stats2.added, 2,
            "the follow-up unbounded pass finishes both previously-held-back dirs: {stats2:?}"
        );
        let idx2 = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert!(
            idx2.search("changed", 10)
                .unwrap()
                .iter()
                .any(|h| h.path.ends_with("A/a.md")),
            "root A's change is now indexed"
        );
        assert!(
            idx2.search("changed", 10)
                .unwrap()
                .iter()
                .any(|h| h.path.ends_with("B/b.md")),
            "root B's change is now indexed"
        );
    }

    /// Data-safety regression: an ALREADY-ELAPSED deadline (Task 10) must never let an
    /// UNCONFIRMED dir-vanish inference (the `removed` list, computed from a plain
    /// dirsig-vs-mtime diff BEFORE this pass's walk ever ran) turn into an actual doc/graph
    /// removal. Reproduces the exact interaction `freshen_blocking_at` can hit in production: a
    /// peer holds `index.lock` long enough to burn the whole wall budget, so by the time we
    /// acquire it the deadline is already in the past and the scoped per-dir loop in
    /// `scan_scoped_delta_at` breaks on its FIRST iteration — the root's own dir key (in
    /// `changed`, since a transient unmount empties its listing) goes straight into
    /// `deadline_held_dirs` UNSCANNED. Meanwhile a subdirectory that vanished in the same breath
    /// lands in `removed` (its own dir key disappeared from the mtime map entirely). Because the
    /// root ALSO has a file directly under it, that file survives untouched (its dir was never
    /// scanned, never in `removed`) — so the Task 8 empty-mount guard's `survivors == 0` check
    /// never fires, and without the fix the `removed`-list inference alone would silently drop
    /// every doc under the vanished subdirectory.
    #[test]
    fn deadline_truncated_pass_holds_removed_subdir_docs_instead_of_dropping_them() {
        use crate::root::Root;
        let root = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("root_file.md"), "root level content\n").unwrap();
        std::fs::create_dir(root.path().join("sub")).unwrap();
        std::fs::write(
            root.path().join("sub").join("sub_file.md"),
            "sub level content\n",
        )
        .unwrap();
        let roots = vec![Root {
            label: String::new(),
            path: root.path().into(),
        }];
        index_dir_at(&roots, state.path(), true).unwrap(); // seed manifest with both docs
        let stored = read_dirsig(state.path()).unwrap();
        assert_eq!(
            Manifest::load(state.path()).files.len(),
            2,
            "sanity: both docs indexed"
        );

        // The subdir transiently vanishes (unmount-style): its own dir key drops out of the
        // mtime map entirely (-> `removed`), while the ROOT's own listing loses an entry (-> its
        // key in `changed`) but the root directory itself stays put.
        std::fs::remove_dir_all(root.path().join("sub")).unwrap();
        bump_dir_mtime_future(root.path());

        let cur = dir_mtime_map_at(&roots, state.path()).unwrap();
        let (changed, added, removed) = diff_corpus_dirs(&stored, &cur);
        assert_eq!(
            changed,
            vec!["c:".to_string()],
            "sanity: only the root's own key changed"
        );
        assert_eq!(
            removed,
            vec!["c:sub".to_string()],
            "sanity: the subdir's key vanished entirely"
        );

        // Already-elapsed deadline: the scoped loop's per-dir check fires before the FIRST (only)
        // changed dir is even resolved — nothing gets scanned this pass.
        let deadline = Some(std::time::Instant::now());
        let stats = reindex_dirs_at_locked(
            &roots,
            state.path(),
            &cur,
            &changed,
            &added,
            &removed,
            false,
            deadline,
            1,
        )
        .unwrap();
        assert_eq!(
            stats.removed, 0,
            "a deadline-truncated pass must drop NOTHING, even a subdir the pre-scan diff called \
             `removed`: {stats:?}"
        );

        let manifest = Manifest::load(state.path());
        assert_eq!(
            manifest.files.len(),
            2,
            "both docs survive the truncated pass, including the one under the 'removed' \
             subdir: {:?}",
            manifest.files
        );
        let idx = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert!(
            idx.search("sub level content", 10)
                .unwrap()
                .iter()
                .any(|h| h.path == "sub/sub_file.md"),
            "the held-back subdir doc is still searchable after the truncated pass"
        );

        // A follow-up UNBOUNDED (complete) pass, once given the chance to actually scan, performs
        // the real removal via the empty-mount-guarded path (this is a genuine deletion, not an
        // empty mount: the root still has `root_file.md` either way).
        let stats2 = reindex_dirs_at_locked(
            &roots,
            state.path(),
            &cur,
            &changed,
            &added,
            &removed,
            false,
            None,
            1,
        )
        .unwrap();
        assert_eq!(
            stats2.removed, 1,
            "the complete follow-up pass performs the real removal: {stats2:?}"
        );
        let manifest2 = Manifest::load(state.path());
        assert_eq!(
            manifest2.files.len(),
            1,
            "only the genuinely-removed subdir doc is gone: {:?}",
            manifest2.files
        );
        assert!(manifest2.files.contains_key("root_file.md"));
    }

    /// Fix round 2 regression: the empty-mount guard must be evaluated PER ROOT, not once for
    /// the whole pass. Round 1 skipped the guard globally whenever ANY dir was deadline-held,
    /// which reintroduced the exact Task 8 loss for a DIFFERENT root that this pass fully,
    /// successfully scanned and found suspiciously empty: multi-share corpus, root A empty-mounts
    /// (scanned to completion this pass), root B is slow/lock-contended and gets deadline-held —
    /// under round 1, A's guard was skipped just because B was held, and A's docs were dropped
    /// for real. Uses the `deadline_fault` test seam (see its doc) instead of a real-time race to
    /// make "A fully scanned, B held" deterministic — a genuine wall-clock deadline can't
    /// reliably land between two near-instant dir scans.
    #[test]
    fn deadline_held_root_does_not_suppress_a_different_roots_empty_mount_guard() {
        use crate::root::Root;
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(a.path().join("a1.md"), "alpha one\n").unwrap();
        std::fs::write(a.path().join("a2.md"), "alpha two\n").unwrap();
        std::fs::write(b.path().join("b1.md"), "beta body\n").unwrap();
        let roots = vec![
            Root {
                label: "A".into(),
                path: a.path().into(),
            },
            Root {
                label: "B".into(),
                path: b.path().into(),
            },
        ];
        index_dir_at(&roots, state.path(), true).unwrap(); // seed manifest with all 3 docs
        let stored = read_dirsig(state.path()).unwrap();
        assert_eq!(
            Manifest::load(state.path()).files.len(),
            3,
            "sanity: all 3 docs indexed"
        );

        // Root A goes empty while still mounted (bumps ONLY its own dir key into `changed`, both
        // files vanish). Root B has an ordinary content change (also bumps its own dir key into
        // `changed`) — unrelated to empty-mount, just here to be the OTHER root that gets held.
        std::fs::remove_file(a.path().join("a1.md")).unwrap();
        std::fs::remove_file(a.path().join("a2.md")).unwrap();
        bump_dir_mtime_future(a.path());
        std::fs::write(
            b.path().join("b1.md"),
            "beta body changed, now much longer\n",
        )
        .unwrap();
        bump_dir_mtime_future(b.path());

        let cur = dir_mtime_map_at(&roots, state.path()).unwrap();
        let (changed, added, removed) = diff_corpus_dirs(&stored, &cur);
        assert_eq!(
            changed,
            vec!["c:A:".to_string(), "c:B:".to_string()],
            "sanity: both roots' dirs changed, A sorts first: {changed:?}"
        );
        assert!(
            removed.is_empty(),
            "sanity: no dir vanished entirely: {removed:?}"
        );

        // Force the per-dir loop to stop after root A's dir is fully scanned but before it
        // reaches root B — deterministic stand-in for "B is slow/lock-held and burns the
        // deadline", without racing a real Instant against two near-instant dir scans.
        deadline_fault::arm_after(1);
        let deadline = Some(std::time::Instant::now() + std::time::Duration::from_secs(3600));
        let stats = reindex_dirs_at_locked(
            &roots,
            state.path(),
            &cur,
            &changed,
            &added,
            &removed,
            false,
            deadline,
            1,
        )
        .unwrap();
        deadline_fault::clear();

        assert_eq!(
            stats.removed, 0,
            "root A's docs must be HELD (empty-mount), not dropped, even though root B was \
             separately deadline-held this same pass: {stats:?}"
        );
        assert_eq!(
            stats.empty_mount_holds, 1,
            "root A is flagged as an empty-mount hold: {stats:?}"
        );

        let manifest = Manifest::load(state.path());
        assert_eq!(
            manifest.files.len(),
            3,
            "all 3 docs survive: A's 2 held by the empty-mount guard, B's 1 held by the \
             deadline: {:?}",
            manifest.files
        );
        assert!(
            manifest.files.contains_key("A/a1.md") && manifest.files.contains_key("A/a2.md"),
            "root A's docs specifically survive, not just the count: {:?}",
            manifest.files
        );
    }

    #[test]
    fn freshen_blocking_degrades_when_lock_held_until_deadline() {
        use std::time::Duration;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nx\n").unwrap();
        index_dir(dir.path(), false).unwrap();
        std::thread::sleep(Duration::from_millis(10));
        std::fs::write(dir.path().join("b.md"), b"# B\ny\n").unwrap(); // pending delta

        let _held = crate::index::lock::try_index_lock(dir.path()).expect("hold the lock");
        let start = std::time::Instant::now();
        // Lock is held by us on this thread; freshen must not hang — it returns by the deadline.
        freshen_blocking(dir.path(), Duration::from_millis(200)).unwrap();
        assert!(
            start.elapsed() >= Duration::from_millis(200),
            "waited for the deadline"
        );
        assert!(start.elapsed() < Duration::from_secs(2), "did not hang");
    }
}

#[cfg(test)]
mod store_root_tests {
    use super::*;

    #[test]
    fn doc_file_bare_relpath_is_backcompat_for_single_empty_label_root() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), "hi").unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        // bare relpath, byte-identical to today: root.join("a.md")
        let want = std::fs::canonicalize(dir.path().join("a.md")).unwrap();
        assert_eq!(std::fs::canonicalize(idx.doc_file("a.md")).unwrap(), want);
    }

    #[test]
    fn doc_file_resolves_label_to_the_right_root() {
        use crate::root::Root;
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(a.path().join("x.md"), "A").unwrap();
        std::fs::write(b.path().join("x.md"), "B").unwrap();
        let roots = vec![
            Root {
                label: "docs".into(),
                path: a.path().into(),
            },
            Root {
                label: "specs".into(),
                path: b.path().into(),
            },
        ];
        let idx = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert_eq!(
            std::fs::canonicalize(idx.doc_file("docs/x.md")).unwrap(),
            std::fs::canonicalize(a.path().join("x.md")).unwrap()
        );
        assert_eq!(
            std::fs::canonicalize(idx.doc_file("specs/x.md")).unwrap(),
            std::fs::canonicalize(b.path().join("x.md")).unwrap()
        );
    }

    #[test]
    fn doc_key_prefixes_label_unless_empty() {
        let a = tempfile::tempdir().unwrap();
        let root = abs_root(a.path());
        let abs = root.join("sub").join("f.md");
        assert_eq!(doc_key("", &root, &abs), "sub/f.md"); // back-compat bare
        assert_eq!(doc_key("docs", &root, &abs), "docs/sub/f.md"); // labeled
    }

    #[test]
    fn multi_root_indexes_both_and_keys_are_labeled_no_collision() {
        use crate::root::Root;
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(a.path().join("dup.md"), "alpha body unique_a").unwrap();
        std::fs::write(b.path().join("dup.md"), "beta body unique_b").unwrap();
        let roots = vec![
            Root {
                label: "A".into(),
                path: a.path().into(),
            },
            Root {
                label: "B".into(),
                path: b.path().into(),
            },
        ];
        let stats = index_dir_at(&roots, state.path(), true).unwrap();
        assert_eq!(
            stats.added, 2,
            "identical relpath in both roots → 2 docs, no collision"
        );
        // All artifacts under state, nothing written into the corpus roots.
        assert!(state.path().join(".glossa").join("index").is_dir());
        assert!(!a.path().join(".glossa").exists());
        assert!(!b.path().join(".glossa").exists());
        // Keys are labeled; doc_file round-trips each to its own root.
        let idx = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert!(idx.read_chunk_by_ord("A/dup.md", 1).unwrap().is_some());
        assert!(idx.read_chunk_by_ord("B/dup.md", 1).unwrap().is_some());
    }

    #[test]
    fn single_root_index_dir_wrapper_is_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), "hello single root").unwrap();
        let stats = index_dir(dir.path(), true).unwrap();
        assert_eq!(stats.added, 1);
        assert!(dir.path().join(".glossa").join("index").is_dir()); // co-located, as today
    }

    #[test]
    fn ensure_fresh_at_is_multi_root_and_does_not_destructively_wipe_a_separated_state_dir() {
        // Regression: `ensure_fresh` (the CLI search/grep/glob freshness pre-check) used to be a
        // single-root-only primitive. Calling it with `state_base` alone (as if it were the sole
        // corpus root) in SEPARATED state-dir mode makes its scan see NONE of the actual corpus
        // files (they live under the real roots, not state_base) and misreads every already-indexed
        // doc as deleted — the next reindex then DROPS it. `ensure_fresh_at` must walk every root.
        use crate::root::Root;
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap(); // state_base != any root: true separation
        std::fs::write(a.path().join("a.md"), "alpha content here").unwrap();
        std::fs::write(b.path().join("b.md"), "beta content here").unwrap();
        let roots = vec![
            Root {
                label: "docs".into(),
                path: a.path().into(),
            },
            Root {
                label: "specs".into(),
                path: b.path().into(),
            },
        ];
        index_dir_at(&roots, state.path(), true).unwrap();

        // Nothing changed on disk: ensure_fresh_at must be a no-op, NOT drop both docs.
        let s = ensure_fresh_at(&roots, state.path()).unwrap();
        assert_eq!(
            (s.added, s.removed, s.unchanged),
            (0, 0, 2),
            "unchanged multi-root corpus must not be reported as removed: {s:?}"
        );
        let idx = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert!(
            idx.read_chunk_by_ord("docs/a.md", 1).unwrap().is_some(),
            "docs/a.md must survive an ensure_fresh_at pass"
        );
        assert!(
            idx.read_chunk_by_ord("specs/b.md", 1).unwrap().is_some(),
            "specs/b.md must survive an ensure_fresh_at pass"
        );

        // A new file under the SECONDARY root is picked up.
        std::fs::write(b.path().join("c.md"), "gamma content here").unwrap();
        let s2 = ensure_fresh_at(&roots, state.path()).unwrap();
        assert_eq!(
            s2.added, 1,
            "new file under the secondary root is added: {s2:?}"
        );
        let idx2 = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert!(idx2.read_chunk_by_ord("specs/c.md", 1).unwrap().is_some());
    }

    #[test]
    fn index_file_into_persists_the_precomputed_labeled_key() {
        use crate::root::Root;
        // Multi-root: the persisted key carries the label the CALLER computed (never re-derived here).
        let labeled = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(labeled.path().join("f.md"), "labeled body zulu").unwrap();
        let roots = vec![Root {
            label: "docs".into(),
            path: labeled.path().into(),
        }];
        index_dir_at(&roots, state.path(), true).unwrap();
        let idx = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert!(
            idx.read_chunk_by_ord("docs/f.md", 1).unwrap().is_some(),
            "key is <label>/<relpath>"
        );
        assert!(
            idx.read_chunk_by_ord("f.md", 1).unwrap().is_none(),
            "label is not dropped"
        );

        // Single empty-label root: bare relpath, byte-identical to today.
        let bare = tempfile::tempdir().unwrap();
        let state2 = tempfile::tempdir().unwrap();
        std::fs::write(bare.path().join("f.md"), "bare body zulu").unwrap();
        index_dir_at(
            &[Root {
                label: String::new(),
                path: bare.path().into(),
            }],
            state2.path(),
            true,
        )
        .unwrap();
        let idx2 = DocIndex::open_or_create_at(
            &[Root {
                label: String::new(),
                path: bare.path().into(),
            }],
            state2.path(),
        )
        .unwrap();
        assert!(
            idx2.read_chunk_by_ord("f.md", 1).unwrap().is_some(),
            "empty label ⇒ bare relpath key"
        );
    }

    #[test]
    fn multi_root_freshen_picks_up_new_file_under_secondary_root() {
        use crate::root::Root;
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(a.path().join("seed.md"), "seed body in A").unwrap();
        let roots = vec![
            Root {
                label: "A".into(),
                path: a.path().into(),
            },
            Root {
                label: "B".into(),
                path: b.path().into(),
            },
        ];
        // Initial index: only root A has a file.
        let s0 = index_dir_at(&roots, state.path(), true).unwrap();
        assert_eq!(s0.added, 1);
        let dirsig0 = read_dirsig(state.path()).unwrap();

        // A NEW file lands under the SECONDARY root B. Before this fix, `reindex_dirs_locked`
        // only ever walked `idx.primary_root()`, so a live MCP server would never pick this up.
        std::fs::write(b.path().join("new.md"), "brand new body in B").unwrap();

        // The freshen gate: `dir_mtime_map_at` over ALL roots must differ from the persisted
        // dirsig (the write above bumped root B's OWN dir mtime, a key `dir_mtime_map` — the
        // single-root form — would never even observe).
        let cur = dir_mtime_map_at(&roots, state.path()).unwrap();
        assert_ne!(
            cur, dirsig0,
            "dirsig gate must detect the change under the secondary root"
        );
        let (changed, added, removed) = diff_corpus_dirs(&dirsig0, &cur);

        let stats = reindex_dirs_at_locked(
            &roots,
            state.path(),
            &cur,
            &changed,
            &added,
            &removed,
            false,
            None,
            read_retries(),
        )
        .unwrap();
        assert_eq!(
            stats.added, 1,
            "the new file under root B is picked up: {stats:?}"
        );

        let idx = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert!(
            idx.read_chunk_by_ord("B/new.md", 1).unwrap().is_some(),
            "new file is indexed under its LABELED key, not dropped or mis-keyed"
        );

        // `new.md` was written moments ago, so it falls inside the FRESH_WINDOW_SECS mid-copy
        // hold: root B's dirsig entry is intentionally NOT advanced to the fresh value yet (it
        // stays at its pre-write value, forcing a re-verify next freshen). That hold only
        // self-heals correctly if `parent_dir_key_at`'s key for root B agrees EXACTLY with
        // `dir_mtime_map_at`'s own key format (`"c:B:"`) — the reviewer's flagged concern. A
        // format mismatch would silently fail to hold it back (the key wouldn't be found in
        // `unsettled`), advancing the dirsig early and masking the re-verify.
        let dirsig1 = read_dirsig(state.path()).unwrap();
        assert_eq!(
            dirsig1.get("c:B:"),
            dirsig0.get("c:B:"),
            "root B's dir stays held back at its pre-write value (mid-copy hold), proving \
             parent_dir_key_at agrees with dir_mtime_map_at's key format: {dirsig1:?}"
        );
        // Root A (untouched by root B's write) settles normally to the current value.
        assert_eq!(dirsig1.get("c:A:"), cur.get("c:A:"));
    }

    #[test]
    fn split_doc_key_unmatched_label_falls_back_to_bare_relpath() {
        use crate::root::Root;
        let roots = vec![
            Root {
                label: "docs".into(),
                path: PathBuf::from("/tmp/docs"),
            },
            Root {
                label: "specs".into(),
                path: PathBuf::from("/tmp/specs"),
            },
        ];
        // A matching label splits normally.
        assert_eq!(split_doc_key("docs/a.md", &roots), ("docs", "a.md"));
        // A label that matches NO configured root (e.g. a since-removed root, or a link src that
        // predates a rename) falls back to treating the WHOLE key as a bare relpath under the
        // first root — the same defensive fallback `DocIndex::doc_file` uses.
        assert_eq!(split_doc_key("gone/old.md", &roots), ("", "gone/old.md"));
    }

    #[test]
    fn two_root_references_never_cross_roots() {
        use crate::root::Root;
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        // A SAME-NAMED target file in BOTH roots — a naive single-root-anchored resolver would
        // ambiguously canonicalize against whichever root it was given, potentially fabricating
        // (or missing) an edge.
        std::fs::write(a.path().join("src.md"), "see [it](target.md)").unwrap();
        std::fs::write(a.path().join("target.md"), "A's target").unwrap();
        std::fs::write(b.path().join("target.md"), "B's target").unwrap();
        let roots = vec![
            Root {
                label: "A".into(),
                path: a.path().into(),
            },
            Root {
                label: "B".into(),
                path: b.path().into(),
            },
        ];
        index_dir_at(&roots, state.path(), true).unwrap();
        let g = crate::graph::store::GraphStore::open(state.path()).unwrap();
        let neighbors = crate::graph::traverse::neighbors(&g, "A/src.md", None, 1).unwrap();
        assert!(
            neighbors.contains(&"A/target.md".to_string()),
            "src.md's relative link resolves to ITS OWN root's target: {neighbors:?}"
        );
        assert!(
            !neighbors.contains(&"B/target.md".to_string()),
            "must NOT fabricate a cross-root edge to the same-named file in root B: {neighbors:?}"
        );
    }

    #[test]
    fn resolve_dir_key_roundtrips_parent_dir_key_at() {
        use crate::root::Root;
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(a.path().join("sub")).unwrap();
        let roots = vec![
            Root {
                label: "docs".into(),
                path: a.path().into(),
            },
            Root {
                label: String::new(),
                path: b.path().into(),
            },
        ];

        // Labeled root, nested file.
        let labeled_key = doc_key(
            "docs",
            &abs_root(a.path()),
            &a.path().join("sub").join("f.md"),
        );
        let dir_key = parent_dir_key_at(&labeled_key, &roots);
        let (dir_abs, label) = resolve_dir_key(&dir_key, &roots).expect("labeled key resolves");
        assert_eq!(label, "docs");
        assert_eq!(dir_abs, abs_root(a.path()).join("sub"));

        // Unlabeled (back-compat) root, root-level file.
        let bare_key = doc_key("", &abs_root(b.path()), &b.path().join("g.md"));
        let dir_key2 = parent_dir_key_at(&bare_key, &roots);
        let (dir_abs2, label2) = resolve_dir_key(&dir_key2, &roots).expect("bare key resolves");
        assert_eq!(label2, "");
        assert_eq!(dir_abs2, abs_root(b.path()));
    }

    #[test]
    fn reindex_scoped_rescan_skips_unrelated_dir() {
        use crate::root::Root;
        let root = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("a")).unwrap();
        std::fs::create_dir_all(root.path().join("b")).unwrap();
        std::fs::write(root.path().join("a").join("f.md"), "alpha one\n").unwrap();
        std::fs::write(root.path().join("b").join("f.md"), "bravo one\n").unwrap();
        let roots = vec![Root {
            label: String::new(),
            path: root.path().into(),
        }];
        index_dir_at(&roots, state.path(), true).unwrap(); // seed manifest + dirsig
        rescan_probe::take(); // drop anything the seeding pass recorded

        // Content-only edit to a/f.md — a DIFFERENT length than the original so `FileSig` (size +
        // second-granularity mtime) registers a change even when both writes land in the same wall
        // clock second. This does NOT bump a/'s own directory mtime (a dir's mtime only tracks
        // add/remove/rename of its entries) — force that deterministically, the same technique
        // `freshen_blocking_scoped_picks_up_a_new_file` uses for a coarse-granularity FS clock.
        std::fs::write(root.path().join("a").join("f.md"), "alpha TWO revised\n").unwrap();
        filetime::set_file_mtime(
            root.path().join("a"),
            filetime::FileTime::from_system_time(
                std::time::SystemTime::now() + std::time::Duration::from_secs(30),
            ),
        )
        .unwrap();

        freshen_blocking_at(&roots, state.path(), std::time::Duration::from_secs(3)).unwrap();

        let walked = rescan_probe::take();
        let a_abs = abs_root(root.path()).join("a");
        let b_abs = abs_root(root.path()).join("b");
        assert!(
            walked.contains(&a_abs),
            "a's dir must be re-walked (it changed): {walked:?}"
        );
        assert!(
            !walked.contains(&b_abs),
            "b's dir must NOT be re-walked — the scoped best case: {walked:?}"
        );

        let idx = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert!(
            idx.search("TWO", 10)
                .unwrap()
                .iter()
                .any(|h| h.path == "a/f.md"),
            "a's updated content is searchable"
        );
        assert!(
            idx.search("bravo", 10)
                .unwrap()
                .iter()
                .any(|h| h.path == "b/f.md"),
            "b's untouched content is still searchable"
        );
    }

    #[test]
    fn reindex_scoped_rescan_still_drops_removed_dir_files() {
        use crate::root::Root;
        let root = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("c")).unwrap();
        std::fs::write(root.path().join("c").join("one.md"), "gamma one\n").unwrap();
        std::fs::write(root.path().join("c").join("two.md"), "gamma two\n").unwrap();
        // Task 8, fix round 1: a residual doc directly under the root keeps this a PARTIAL
        // deletion (some of the root's docs survive) rather than a complete wipe of a
        // previously-populated root. Under the empty-mount safety policy, emptying a root
        // ENTIRELY via the (force-less) freshen path now HOLDS by default — only a partial
        // removal still drops normally without `--force`, which is what this test guards.
        std::fs::write(root.path().join("keepalive.md"), "delta stays\n").unwrap();
        let roots = vec![Root {
            label: String::new(),
            path: root.path().into(),
        }];
        index_dir_at(&roots, state.path(), true).unwrap();

        std::fs::remove_dir_all(root.path().join("c")).unwrap();
        // Removing c/ bumps the ROOT dir's own mtime (an entry vanished from it) — force it
        // deterministically for a coarse-granularity FS clock, same technique as elsewhere here.
        filetime::set_file_mtime(
            root.path(),
            filetime::FileTime::from_system_time(
                std::time::SystemTime::now() + std::time::Duration::from_secs(30),
            ),
        )
        .unwrap();

        let stats =
            freshen_blocking_at(&roots, state.path(), std::time::Duration::from_secs(3)).unwrap();
        assert_eq!(
            stats.removed, 2,
            "both of c/'s files drop without a full-corpus walk: {stats:?}"
        );

        let idx = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert!(
            idx.search("gamma", 10).unwrap().is_empty(),
            "removed dir's content must no longer be searchable"
        );
        assert!(
            idx.search("delta", 10)
                .unwrap()
                .iter()
                .any(|h| h.path == "keepalive.md"),
            "the surviving doc outside the removed dir is untouched"
        );
        let manifest = Manifest::load(state.path());
        assert!(
            manifest.files.contains_key("keepalive.md") && manifest.files.len() == 1,
            "partial deletion: root not flagged as empty-mount (one doc still survives): {:?}",
            manifest.files
        );
    }

    /// Task 8, fix round 1: the empty-mount guard must also protect the SCOPED freshen path
    /// (`reindex_dirs_at_locked` via `scan_scoped_delta_at`), not just the CLI's always-full
    /// `scan_delta_at` walk. A share going empty while still mounted bumps only the ROOT's own
    /// dir key into `changed` (never all-empty), so `scan_scoped_delta_at` takes the SCOPED
    /// branch, not the full-walk fallback — this is exactly the gap a review caught: the
    /// per-dir scoped carry-forward would otherwise drop every doc under the root because BOTH
    /// its files live directly in the root (no surviving doc anywhere under the label).
    #[test]
    fn reindex_scoped_rescan_holds_root_on_empty_mount() {
        use crate::root::Root;
        let root = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("one.md"), "gamma one\n").unwrap();
        std::fs::write(root.path().join("two.md"), "gamma two\n").unwrap();
        let roots = vec![Root {
            label: String::new(),
            path: root.path().into(),
        }];
        index_dir_at(&roots, state.path(), true).unwrap(); // seed manifest with 2 entries

        // The share goes empty while still mounted: both files vanish, but the root directory
        // itself stays put — bumps ONLY the root's own dir mtime, the exact shape a transient
        // empty-mount presents as (never lands the root's own key in `removed`).
        std::fs::remove_file(root.path().join("one.md")).unwrap();
        std::fs::remove_file(root.path().join("two.md")).unwrap();
        filetime::set_file_mtime(
            root.path(),
            filetime::FileTime::from_system_time(
                std::time::SystemTime::now() + std::time::Duration::from_secs(30),
            ),
        )
        .unwrap();

        let stats =
            freshen_blocking_at(&roots, state.path(), std::time::Duration::from_secs(3)).unwrap();
        assert_eq!(
            stats.removed, 0,
            "held back, not mass-removed, via the SCOPED freshen path: {stats:?}"
        );

        let manifest = Manifest::load(state.path());
        assert_eq!(
            manifest.files.len(),
            2,
            "both stale docs survive the scoped rescan: {:?}",
            manifest.files
        );
        let idx = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert!(
            idx.search("gamma", 10)
                .unwrap()
                .iter()
                .any(|h| h.path == "one.md"),
            "held doc stays searchable"
        );

        // `--force` is the escape hatch for a genuinely emptied corpus root: it resets the
        // manifest before scanning, so `had_prior` is false and the empty state sticks.
        index_dir_at(&roots, state.path(), true).unwrap();
        let manifest = Manifest::load(state.path());
        assert!(
            manifest.files.is_empty(),
            "force makes the genuinely-empty state stick: {:?}",
            manifest.files
        );
    }

    #[test]
    fn reindex_scoped_rescan_falls_back_to_full_walk_on_unresolvable_key() {
        use crate::root::Root;
        let root = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("a.md"), "alpha content\n").unwrap();
        let roots = vec![Root {
            label: String::new(),
            path: root.path().into(),
        }];
        index_dir_at(&roots, state.path(), true).unwrap();

        // A new file that the (bogus) diff below never names — only a full-walk fallback picks it up.
        std::fs::write(root.path().join("b.md"), "bravo content\n").unwrap();
        let cur = dir_mtime_map_at(&roots, state.path()).unwrap();

        // Hand-crafted bogus `changed` key: a labeled form whose label matches no configured root.
        // `resolve_dir_key` must return `None` for it, and the caller must fall back to a full walk
        // rather than silently under-scan (and miss b.md).
        let bogus = vec!["c:no-such-label:sub".to_string()];
        let stats = reindex_dirs_at_locked(
            &roots,
            state.path(),
            &cur,
            &bogus,
            &[],
            &[],
            false,
            None,
            read_retries(),
        )
        .unwrap();
        assert_eq!(
            stats.added, 1,
            "fallback full walk still finds the new file: {stats:?}"
        );

        let idx = DocIndex::open_or_create_at(&roots, state.path()).unwrap();
        assert!(idx
            .search("bravo", 10)
            .unwrap()
            .iter()
            .any(|h| h.path == "b.md"));
    }
}
