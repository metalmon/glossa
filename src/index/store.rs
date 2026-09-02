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
    /// Absolute corpus root — the anchor that turns a stored relative doc key back into a real file
    /// path (`doc_file`). Set once at open; doc keys are always relative to it.
    pub root: PathBuf,
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

impl DocIndex {
    pub fn open_or_create(dir: &Path) -> anyhow::Result<DocIndex> {
        let schema = build_schema();
        let idx_path = index_dir_path(dir);
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
        Ok(DocIndex {
            index,
            fields,
            root: abs_root(dir),
            reader,
        })
    }

    /// Turn a stored (corpus-relative) doc key back into the real file path under the corpus root.
    /// The single place the system maps a key to a file — no path joining is scattered elsewhere.
    pub fn doc_file(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
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
}

pub fn file_sig(path: &Path) -> anyhow::Result<FileSig> {
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
}

/// Per-directory mtime map: `c:{rel}` for corpus dirs (gitignore-aware, skips `.glossa`), `n:{rel}`
/// for `.glossa/notes` dirs; values are nanosecond mtimes. Add/remove/rename of a file bumps its
/// parent dir's entry; a new/removed subdir adds/drops an entry. In-place content edits don't change it.
pub fn dir_mtime_map(dir: &Path) -> anyhow::Result<BTreeMap<String, u128>> {
    let root = abs_root(dir);
    let mut map: BTreeMap<String, u128> = BTreeMap::new();
    collect_dir_mtimes(&root, true, "c", &mut map);
    let notes_root = dir.join(".glossa").join("notes");
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
    let mut d = Delta::default();
    let root = abs_root(dir);
    crate::walk::walk_files(&root, None, true, &mut |path| {
        let p = rel_key(&root, path);
        let sig = match file_sig(path) {
            Ok(s) => s,
            Err(_) => return Ok(()),
        };
        if manifest.changed(&p, sig) {
            d.changed.push(p.clone());
        }
        d.next.files.insert(p, sig);
        Ok(())
    })?;
    d.removed = manifest
        .files
        .keys()
        .filter(|k| !d.next.files.contains_key(*k))
        .cloned()
        .collect();
    #[cfg(feature = "notebook")]
    scan_notes_delta(dir, manifest, &mut d)?;
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
    let manifest = Manifest::load(dir);
    let delta = scan_delta(dir, &manifest)?;
    let files = &delta.next.files;
    let notes_root = dir.join(".glossa").join("notes");
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
    let manifest = Manifest::load(dir);
    let delta = scan_delta(dir, &manifest)?;
    if delta.changed.is_empty()
        && delta.removed.is_empty()
        && delta.notes_changed.is_empty()
        && delta.notes_removed.is_empty()
    {
        return Ok(IndexStats {
            added: 0,
            removed: 0,
            unchanged: delta.next.files.len() + delta.next.notes.len(),
            ..Default::default()
        });
    }
    // Something changed → take the lock and index. index_dir recomputes the delta FRESH inside the
    // lock (this pre-scan is lock-free and only decides whether to bother): reusing this delta across
    // the lock boundary would be unsound — a concurrent indexer could land between, and our stale
    // file set would then delete what it just added.
    match index_dir(dir, false) {
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
    root: &Path,
    abs_path: &Path,
    links: &mut Vec<(String, String)>,
) -> anyhow::Result<Option<FileSig>> {
    let path_str = rel_key(root, abs_path);
    let sig = match file_sig(abs_path) {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };
    writer.delete_term(tantivy::Term::from_field_text(idx.fields.path, &path_str));
    graph.delete_auto_by_source(&path_str)?;
    let mut doc_written = false;
    let mut seq = 0u64;
    let mut prev_sec: Option<String> = None;
    let mut file_links: Vec<String> = Vec::new();
    let mut seen: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    // Index/graph write errors are intentionally not propagated here: one bad chunk must not
    // abort the whole run (matches the prior per-file behavior). The file is still recorded
    // in the manifest; a failed write is corrected on the next `reindex`.
    crate::extract::extract_file(abs_path, &mut |mut c: Chunk| {
        // Canonicalize the chunk to the corpus-relative key here, at the single indexing
        // boundary, so the index, the structural graph and section ids all speak ONE form.
        c.doc_path = PathBuf::from(&path_str);
        if !doc_written {
            let _ = crate::graph::build::build_document(graph, &path_str, sig);
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
        let _ = writer.add_document(doc!(
            idx.fields.body => c.text.clone(),
            idx.fields.body_trigrams => c.text.clone(),
            idx.fields.path => path_str.clone(),
            idx.fields.location => c.location.clone(),
            idx.fields.file_type => c.file_type.clone(),
            idx.fields.ord => ord,
        ));
        let _ = crate::graph::build::build_section(graph, &c, ord, sig);
        let cur_id = crate::graph::build::section_id(&path_str, &ord.to_string());
        if let Some(prev) = prev_sec.as_deref() {
            let _ = crate::graph::build::link_sequential(graph, prev, &cur_id, sig, &path_str);
        }
        if let Some(parent) = crate::graph::build::nearest_ancestor(&seen, &c.location) {
            let _ = crate::graph::build::link_parent(graph, &cur_id, &parent, sig, &path_str);
        }
        file_links.extend(crate::extract::links::extract_links(&c.text));
        seen.insert(c.location.clone(), cur_id.clone());
        prev_sec = Some(cur_id);
    })?;
    for t in file_links {
        links.push((path_str.clone(), t));
    }
    Ok(Some(sig))
}

/// Walk `dir`, (re)index changed files, drop removed files, update the manifest.
/// `force = true` ignores the manifest and rebuilds every file.
/// Streams each chunk directly into the tantivy writer + graph (constant memory).
pub fn index_dir(dir: &Path, force: bool) -> anyhow::Result<IndexStats> {
    // Cross-process guard: index_dir clears and rewrites `.glossa/index`, which collides with
    // another process doing the same on Windows ("Access is denied"). Hold `.glossa/index.lock`
    // for the whole rebuild; if another process already holds it, skip with a default (no-op)
    // stat rather than racing it — the index is cooperative, so whoever wins leaves it correct.
    // The guard drops at function exit, releasing the lock (RAII).
    let _lock = match crate::index::lock::try_index_lock(dir) {
        Some(guard) => guard,
        None => {
            eprintln!("index_dir: another process is indexing, skipping");
            return Ok(IndexStats::default());
        }
    };
    index_dir_locked(dir, force)
}

/// Resolve collected cross-document links against the current document set, creating `REFERENCES`
/// edges for those that resolve to a real Document node, and returning the `(src, raw_target)` pairs
/// that still don't resolve (target not an indexed doc). Mirrors the resolution `index_dir_locked`
/// did inline; shared with the scoped pass.
fn resolve_reference_links(
    graph: &crate::graph::store::GraphStore,
    root: &Path,
    files: &std::collections::BTreeMap<String, FileSig>,
    links: &[(String, String)],
) -> Vec<(String, String)> {
    // Nothing to resolve → skip touching the filesystem at all (the common case on a scoped
    // pass over a large corpus, where most dirs carry no cross-doc links).
    if links.is_empty() {
        return Vec::new();
    }
    // Canonicalize only the LINK TARGETS (O(links)), not every corpus document (O(files)) — a
    // scoped reindex pass over a handful of changed dirs must not pay an all-corpus filesystem
    // stat just because a link happened to be present. `canonicalize_stripped` mirrors `abs_root`'s
    // `\\?\`-stripping exactly (unlike `abs_root` itself, it returns `None` on failure rather than
    // falling back to the un-canonicalized path) so the result is in the SAME form as `root_abs`
    // and `rel_key`'s `strip_prefix` actually matches instead of silently falling through to the
    // absolute path.
    let root_abs = abs_root(root);
    let mut unresolved = Vec::new();
    for (src, raw_target) in links {
        let src_dir = Path::new(src).parent().unwrap_or_else(|| Path::new(""));
        let dst = canonicalize_stripped(&root.join(src_dir).join(raw_target))
            .map(|canon| rel_key(&root_abs, &canon))
            .filter(|rel| files.contains_key(rel));
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

/// Body of `index_dir` assuming `index.lock` is already held by the caller. Records `.glossa/dirsig`
/// = the current directory signature on every return, so a reader's `freshen_blocking` can tell the
/// index reflects the current tree even when the delta was empty (e.g. a temp file came and went).
pub fn index_dir_locked(dir: &Path, force: bool) -> anyhow::Result<IndexStats> {
    // force = true rebuilds the DOCUMENT-DERIVED layer from scratch: clear the tantivy index and
    // the auto-* graph (structural + lexical) so stale entries (deleted files, or docs previously
    // indexed under a different path form) cannot linger. The agent/curated reasoning graph in
    // graph.sqlite is PRESERVED — a reindex must not destroy hand/agent-built knowledge.
    let mut manifest = if force {
        Manifest::default()
    } else {
        Manifest::load(dir)
    };
    let schema_migrate = manifest.index_schema_version < INDEX_SCHEMA_VERSION;
    if force {
        let _ = std::fs::remove_dir_all(dir.join(".glossa").join("index"));
    } else if schema_migrate {
        let _ = std::fs::remove_dir_all(dir.join(".glossa").join("index"));
        manifest.files.clear();
        manifest.notes.clear();
    }
    let idx = DocIndex::open_or_create(dir)?;
    let graph = crate::graph::store::GraphStore::open(dir)?;
    if force {
        graph.delete_auto()?;
    }

    // The delta drives everything below in a single tree walk: the hot-path gate skips all work when
    // nothing changed, the walk-free indexing loop re-extracts just `delta.changed`, and the notes
    // pass reuses the same `notes_changed`/`notes_removed`. Under `force`/schema-migrate the manifest
    // was reset, so every file/note shows up as changed and is re-added.
    let mut delta = scan_delta(dir, &manifest)?;
    if !force
        && delta.changed.is_empty()
        && delta.removed.is_empty()
        && delta.notes_changed.is_empty()
        && delta.notes_removed.is_empty()
    {
        write_dirsig(dir, &dir_mtime_map(dir).unwrap_or_default());
        return Ok(IndexStats {
            added: 0,
            removed: 0,
            unchanged: delta.next.files.len() + delta.next.notes.len(),
            ..Default::default()
        });
    }

    let mut writer = with_writer_retry(|| idx.index.writer(50_000_000))?;
    let mut stats = IndexStats::default();
    let mut next = Manifest::default();

    let mut links: Vec<(String, String)> = Vec::new();
    // Files we actually (re)indexed this pass — re-stat'd at the end so a file that changed while
    // we were indexing it holds back its dir's dirsig entry (see `unsettled_dirs`).
    let mut indexed: Vec<(String, FileSig)> = Vec::new();
    eprintln!("indexing files under {}...", idx.root.display());
    // Drive indexing off the delta we already computed — do NOT walk the tree again just to re-stat
    // every file. `delta.next.files` is the full current file set (with signatures) that `scan_delta`
    // produced in a single walk, so it becomes `next.files` directly; `delta.changed` is exactly the
    // subset needing (re)extraction (under `force`/schema-migrate the manifest was reset, so every
    // file is "changed" and re-extracted — same as before). This removes one of the redundant full
    // walks per `kb index`/`ensure_fresh` pass (the "reindex is slower than early versions" report).
    next.files = std::mem::take(&mut delta.next.files);
    for path_str in &delta.changed {
        let Some(&sig) = next.files.get(path_str) else {
            continue;
        };
        eprintln!("  + {path_str}");
        let abs = idx.root.join(path_str);
        // A single unreadable/corrupt file (e.g. a .doc with a bad CFB header) must NOT abort the
        // whole index — log and skip it, exactly as the old walk_files-driven loop did (walk_files
        // caught the visit closure's error and printed "skip …"). The file stays recorded in
        // next.files, so a later pass treats it as unchanged and doesn't retry it every time.
        if let Err(e) = index_file_into(&idx, &graph, &writer, &idx.root, &abs, &mut links) {
            eprintln!("skip {}: {e}", abs.display());
            stats.errors.push((path_str.clone(), e.to_string()));
            continue;
        }
        indexed.push((path_str.clone(), sig));
        stats.added += 1;
    }
    stats.unchanged = next
        .files
        .len()
        .saturating_sub(stats.added + stats.errors.len());

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
    // Doc keys are corpus-relative, so resolve each to a real file through the corpus root anchor.
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
    next.unresolved_links = resolve_reference_links(&graph, &idx.root, &next.files, &links);
    // Notebook notes: index every changed note as a single `"note"` chunk (delete-by-path is
    // idempotent, so a note that was also written through in-process re-adds cleanly), and drop
    // removed notes. Notes never create graph nodes (§6 of the spec) — only search chunks.
    #[cfg(feature = "notebook")]
    {
        let notes_root = dir.join(".glossa").join("notes");
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
    next.save(dir)?;
    // Advance dirsig, holding back any dir whose file changed mid-pass so a full index can't poison
    // the dir-mtime gate either (mirrors reindex_dirs_locked).
    let cur = dir_mtime_map(dir).unwrap_or_default();
    let stored = read_dirsig(dir).unwrap_or_default();
    let unsettled = unsettled_dirs(&idx.root, &indexed);
    write_dirsig(dir, &settled_dirsig(&cur, &stored, &unsettled));
    Ok(stats)
}

/// Reindex ONE corpus document, assuming the caller already holds `index.lock`. Drops the file's old
/// chunks + auto-graph-by-source and rebuilds them, resolves its outgoing references against the
/// current document set, commits, and records the new signature in the manifest. `None` if the file
/// is gone/unreadable (manifest is left unchanged so a later pass can drop it).
pub fn index_one_file_locked(dir: &Path, rel: &str) -> anyhow::Result<Option<FileSig>> {
    let idx = DocIndex::open_or_create(dir)?;
    let graph = crate::graph::store::GraphStore::open(dir)?;
    let abs = idx.root.join(rel);
    let mut writer = with_writer_retry(|| idx.index.writer(50_000_000))?;
    let mut links: Vec<(String, String)> = Vec::new();
    let sig = match index_file_into(&idx, &graph, &writer, &idx.root, &abs, &mut links)? {
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
    let mut manifest = Manifest::load(dir);
    manifest.files.insert(rel.to_string(), sig);
    let _ = resolve_reference_links(&graph, &idx.root, &manifest.files, &links);
    // Bounded transient-FS retry around the `meta.json`-writing commit (see index_dir_locked).
    with_writer_retry(|| writer.commit())?;
    let mut m = Manifest::load(dir);
    m.files.insert(rel.to_string(), sig);
    m.save(dir)?;
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

/// Scoped reindex of only the corpus dirs the caller determined changed/added/removed (typically
/// from a `dir_mtime_map` diff), assuming `index.lock` is already held. Reindexes just the
/// affected dirs' files (via `index_file_into`), drops files/dirs that vanished (cleaning graph
/// edges in BOTH directions), runs the notes pass if `notes_touched`, re-resolves references
/// (including previously-persisted `unresolved_links`), updates the manifest partially, commits,
/// and writes `cur_map` to `.glossa/dirsig`. Mirrors `index_dir_locked`'s per-file logic, but the
/// walk (and thus the extraction cost) is scoped to `changed`/`added`, not the whole corpus.
pub fn reindex_dirs_locked(
    dir: &Path,
    cur_map: &BTreeMap<String, u128>,
    changed: &[String],
    added: &[String],
    removed: &[String],
    notes_touched: bool,
) -> anyhow::Result<IndexStats> {
    let idx = DocIndex::open_or_create(dir)?;
    let graph = crate::graph::store::GraphStore::open(dir)?;
    let mut writer = with_writer_retry(|| idx.index.writer(50_000_000))?;
    let mut manifest = Manifest::load(dir);
    let mut links: Vec<(String, String)> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut stats = IndexStats::default();
    // Files we actually (re)indexed this pass, with the signature we captured — re-stat'd at the
    // end so a file that changed WHILE we were indexing it holds back its dir's dirsig entry.
    let mut indexed: Vec<(String, FileSig)> = Vec::new();

    // 1. Rescan affected dirs: index new/changed IMMEDIATE files of each changed/added corpus dir
    // (same ignore semantics as the corpus walk, but depth-1 since only that dir's own files can
    // have changed — a changed grandchild dir shows up as its own separate key in the map diff).
    let rescan: Vec<String> = changed.iter().chain(added.iter()).cloned().collect();
    for key in &rescan {
        let d = key.strip_prefix("c:").unwrap_or("");
        let base = if d.is_empty() {
            idx.root.clone()
        } else {
            idx.root.join(d)
        };
        let mut wb = WalkBuilder::new(&base);
        wb.standard_filters(true);
        wb.require_git(false);
        wb.max_depth(Some(1));
        wb.filter_entry(|e| e.file_name() != ".glossa");
        for entry in wb.build().flatten() {
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let path = entry.path();
            let rel = rel_key(&idx.root, path);
            let sig = match file_sig(path) {
                Ok(s) => s,
                Err(_) => continue,
            };
            seen.insert(rel.clone());
            if manifest.files.get(&rel) != Some(&sig) {
                // Skip (don't abort the freshen on) a single corrupt/unreadable file — mirrors the
                // resilience of the `kb index` walk. On error the file is left unrecorded so a later
                // freshen can retry it once it changes.
                match index_file_into(&idx, &graph, &writer, &idx.root, path, &mut links) {
                    Ok(Some(_)) => {
                        manifest.files.insert(rel.clone(), sig);
                        indexed.push((rel.clone(), sig));
                        stats.added += 1;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        eprintln!("skip {}: {e}", path.display());
                        stats.errors.push((rel.clone(), e.to_string()));
                    }
                }
            }
        }
    }

    // 2. Drop removed files/dirs (clean edges BOTH directions): a manifest key K is removed iff
    // its parent dir is wholly gone, or K's parent dir was rescanned above and K didn't turn up.
    // Collect first, then remove, so we don't mutate `manifest.files` while iterating its keys.
    let rescan_set: std::collections::HashSet<&str> = rescan.iter().map(String::as_str).collect();
    let removed_set: std::collections::HashSet<&str> = removed.iter().map(String::as_str).collect();
    let all_keys: Vec<String> = manifest.files.keys().cloned().collect();
    let gone: Vec<String> = all_keys
        .into_iter()
        .filter(|k| {
            let pd = parent_dir_key(k);
            removed_set.contains(pd.as_str())
                || (rescan_set.contains(pd.as_str()) && !seen.contains(k.as_str()))
        })
        .collect();
    for k in &gone {
        writer.delete_term(tantivy::Term::from_field_text(idx.fields.path, k.as_str()));
        graph.delete_auto_by_source(k)?;
        graph.delete_auto_by_target(k)?;
        manifest.files.remove(k);
        stats.removed += 1;
    }

    // 3. Notes: run when a `n:` dir key moved OR a corpus document was removed this pass. Deleting
    // an owner document only bumps its `c:` parent dir (the notes tree under `.glossa/notes` is
    // untouched), so `notes_touched` alone would miss it and leave the orphaned note's chunk
    // searchable — unlike a full `index_dir`, which always runs the notes pass and lets
    // `owner_in` purge it. Reuse the exact notes-index/drop block `index_dir_locked` uses,
    // sourced from a fresh `scan_notes_delta` against the manifest as it stands after steps 1-2.
    // `owner_in` (inside `scan_notes_delta`) checks `d.next.files` for a note's owning document,
    // so it must be seeded with the current file map, not left empty.
    #[cfg(feature = "notebook")]
    if notes_touched || !gone.is_empty() {
        let mut d = Delta::default();
        d.next.files = manifest.files.clone();
        scan_notes_delta(dir, &manifest, &mut d)?;
        let notes_root = dir.join(".glossa").join("notes");
        for rel in &d.notes_changed {
            let body = match std::fs::read_to_string(notes_root.join(rel)) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("index notes: skip {rel}: {e}");
                    d.next.notes.remove(rel);
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
        for rel in &d.notes_removed {
            writer.delete_term(tantivy::Term::from_field_text(idx.fields.path, rel));
            stats.removed += 1;
        }
        manifest.notes = d.next.notes;
    }

    // 4. References: resolve freshly-collected links plus carried-over `unresolved_links` (dropping
    // any whose source no longer exists — its edges were already dropped in step 2) against the
    // now-current document set; store what's still unresolved back into the manifest.
    let fresh_srcs: std::collections::HashSet<String> =
        links.iter().map(|(src, _)| src.clone()).collect();
    let mut all = links;
    for (src, target) in manifest.unresolved_links.drain(..) {
        if manifest.files.contains_key(&src) && !fresh_srcs.contains(&src) {
            all.push((src, target));
        }
    }
    manifest.unresolved_links = resolve_reference_links(&graph, &idx.root, &manifest.files, &all);

    // Bounded transient-FS retry around the `meta.json`-writing commit (see index_dir_locked).
    with_writer_retry(|| writer.commit())?;
    manifest.save(dir)?;
    // Advance dirsig, but hold back any dir whose file changed under us (still being written): the
    // lock is held, so the on-disk dirsig is still the value we diffed against (`stored`).
    let stored = read_dirsig(dir).unwrap_or_default();
    let mut unsettled = unsettled_dirs(&idx.root, &indexed);
    // Also hold back a dir whose file we just indexed was written within the last few seconds: it
    // may still be mid-copy. A finalize write landing AFTER this pass does NOT re-bump the dir mtime,
    // so without this the dir-mtime gate would never re-open and the finalized content would stay
    // invisible to search/grep (regression: `freshen_gate_misses_content_finalized_after_a_settled_
    // index`). This is the freshen (MCP) path only — the CLI `scan_delta` path already re-stats every
    // file. The hold self-clears once the file ages past the window, so steady state stays fast.
    let now = now_secs();
    for (rel, sig) in &indexed {
        if now >= sig.mtime_secs && now - sig.mtime_secs < FRESH_WINDOW_SECS {
            unsettled.insert(parent_dir_key(rel));
        }
    }
    write_dirsig(dir, &settled_dirsig(cur_map, &stored, &unsettled));
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

/// Bring the on-disk index up to date with the current tree, synchronously, without ever hanging.
/// Fast path: if `.glossa/dirsig` already equals the current map, return immediately. Else
/// take `index.lock` and index; if another process holds it, poll until either the persisted
/// map reaches what we observed (the peer indexed it) or `timeout` elapses (serve current).
pub fn freshen_blocking(dir: &Path, timeout: std::time::Duration) -> anyhow::Result<IndexStats> {
    let cur = dir_mtime_map(dir)?;
    if read_dirsig(dir).as_ref() == Some(&cur) {
        return Ok(IndexStats::default());
    }
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if let Some(_guard) = crate::index::lock::try_index_lock(dir) {
            // We own the lock: compute the dir-map diff against the last-indexed state and
            // reindex only the changed/added/removed corpus dirs (scoped, sublinear pass).
            // On first run / migration `stored` is empty, so every dir is `added` and the
            // scoped pass reindexes everything -- equivalent to a full pass, self-healing.
            let stored = read_dirsig(dir).unwrap_or_default();
            let (changed, added, removed) = diff_corpus_dirs(&stored, &cur);
            let notes_touched = notes_submap_changed(&stored, &cur);
            return reindex_dirs_locked(dir, &cur, &changed, &added, &removed, notes_touched);
        }
        // Another process is indexing. If it already reached our observed state, we are fresh.
        if read_dirsig(dir).as_ref() == Some(&cur) {
            return Ok(IndexStats::default());
        }
        if std::time::Instant::now() >= deadline {
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
    fn note_is_dropped_when_owner_removed_and_restored_when_owner_returns() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("doc.md"), b"# Doc\nbody\n").unwrap();
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
