use crate::index::multilang::{default_detector, multilang_analyzer};
use crate::model::Chunk;
use anyhow::Context;
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, INDEXED, STORED, STRING};
use tantivy::schema::Value;
use tantivy::snippet::SnippetGenerator;
use tantivy::{doc, Index, IndexReader, TantivyDocument, TantivyError};

#[derive(Clone, Copy)]
pub struct Fields {
    pub body: Field,
    pub path: Field,
    pub location: Field,
    pub file_type: Field,
    pub ord: Field,
}

pub struct DocIndex {
    pub index: Index,
    pub fields: Fields,
    /// Long-lived reader, reused across every search/read_chunk. Building a reader reopens the
    /// segments, so doing it once per call (rather than per query) is what made repeated tool
    /// calls on a shared index pay an open cost each time. Refreshed after writes via reload().
    reader: IndexReader,
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
    let ord = sb.add_u64_field("ord", INDEXED | STORED);
    (sb.build(), Fields { body, path, location, file_type, ord })
}

fn index_dir_path(dir: &Path) -> std::path::PathBuf {
    dir.join(".glossa").join("index")
}

impl DocIndex {
    pub fn open_or_create(dir: &Path) -> anyhow::Result<DocIndex> {
        let (schema, fields) = build_schema();
        let idx_path = index_dir_path(dir);
        std::fs::create_dir_all(&idx_path).with_context(|| format!("create {idx_path:?}"))?;
        let index = match Index::create_in_dir(&idx_path, schema.clone()) {
            Ok(i) => i,
            Err(TantivyError::IndexAlreadyExists) => Index::open_in_dir(&idx_path)?,
            Err(e) => return Err(e.into()),
        };
        index
            .tokenizers()
            .register("multilang", multilang_analyzer(default_detector()));
        let reader = index.reader()?;
        Ok(DocIndex { index, fields, reader })
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
    /// One search-result line carrying exactly one number — the read key `[#ord]` — and a
    /// non-numeric label (the heading text, or the file type for paged formats whose location is
    /// itself a number) so nothing competes with the read key.
    pub fn display_line(&self) -> String {
        let label = if self.location.starts_with("p.") { self.file_type.as_str() } else { self.location.as_str() };
        format!("[#{}] {} · {} · {}", self.ord, self.path, label, self.snippet)
    }
}

impl DocIndex {
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
                d.get_first(f).and_then(|v| v.as_str()).unwrap_or("").to_string()
            };
            let snippet = snippet_gen.snippet_from_doc(&d).fragment().to_string();
            let ord = d.get_first(self.fields.ord).and_then(|v| v.as_u64()).unwrap_or(0);
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
        let pool = limit.saturating_mul(20).min(2000).max(limit);
        let hits = self.search(query, pool)?;
        let filtered: Vec<RankedHit> = hits
            .into_iter()
            .filter(|h| file_type.map_or(true, |ft| h.file_type == ft))
            .filter(|h| glob_re.as_ref().map_or(true, |re| re.is_match(&h.path)))
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
        let top = searcher.search(&BooleanQuery::new(clauses), &TopDocs::with_limit(1).order_by_score())?;
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

    /// Resolve chunk number `n` to the `location` string stored in the index for `path`.
    /// Mirrors `ord_body` but returns the location field instead of the body.
    /// Returns `None` when no chunk with that (path, ord) pair is indexed.
    pub fn location_for_ord(&self, path: &str, n: u64) -> anyhow::Result<Option<String>> {
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
                Ok(Some(d.get_first(self.fields.location).and_then(|v| v.as_str()).unwrap_or("").to_string()))
            }
            None => Ok(None),
        }
    }

    /// Resolve a `location` string to the chunk number (`ord`) stored in the index for `path`.
    /// Mirrors `read_chunk` (path+location BooleanQuery) but returns the `ord` field.
    /// Returns `None` when no chunk matches that (path, location) pair.
    pub fn ord_for_location(&self, path: &str, location: &str) -> anyhow::Result<Option<u64>> {
        use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
        let searcher = self.reader.searcher();
        let clauses: Vec<(Occur, Box<dyn Query>)> = vec![
            (Occur::Must, Box::new(TermQuery::new(
                tantivy::Term::from_field_text(self.fields.path, path), IndexRecordOption::Basic))),
            (Occur::Must, Box::new(TermQuery::new(
                tantivy::Term::from_field_text(self.fields.location, location), IndexRecordOption::Basic))),
        ];
        let top = searcher.search(&BooleanQuery::new(clauses), &TopDocs::with_limit(1).order_by_score())?;
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

    /// Resolve a possibly-mangled `input` path to the real indexed path by collapsing runs of
    /// whitespace (the model routinely turns a document's double space into a single one when
    /// copying a path). Returns the exact path only when exactly one indexed document matches the
    /// normalized form — never guesses between ambiguous candidates.
    pub fn resolve_path(&self, input: &str) -> anyhow::Result<Option<String>> {
        fn norm(s: &str) -> String {
            // Collapse runs of whitespace AND of path separators, and unify `/` and `\` to `\`, so
            // a model that double-escapes (`\\`) or swaps separators still resolves to the real path.
            let mut out = String::with_capacity(s.len());
            let mut prev_sep = false;
            let mut prev_ws = false;
            for c in s.chars() {
                if c == '/' || c == '\\' {
                    if !prev_sep {
                        out.push('\\');
                    }
                    prev_sep = true;
                    prev_ws = false;
                } else if c.is_whitespace() {
                    if !prev_ws && !out.is_empty() {
                        out.push(' ');
                    }
                    prev_ws = true;
                    prev_sep = false;
                } else {
                    out.push(c);
                    prev_sep = false;
                    prev_ws = false;
                }
            }
            out.trim().to_string()
        }
        let target = norm(input);
        let mut seen = std::collections::HashSet::new();
        let mut matches: Vec<String> = Vec::new();
        self.iter_chunks(|path, _ord, _ft, _body| {
            if seen.insert(path.to_string()) && norm(path) == target {
                matches.push(path.to_string());
            }
        })?;
        Ok(if matches.len() == 1 { matches.pop() } else { None })
    }
}

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

use crate::index::manifest::{FileSig, Manifest};

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
    /// Visit every stored chunk: `f(path, ord, file_type, body)`. Used by grep's full scan.
    pub fn iter_chunks(&self, mut f: impl FnMut(&str, u64, &str, &str)) -> anyhow::Result<()> {
        use tantivy::collector::DocSetCollector;
        use tantivy::query::AllQuery;
        let searcher = self.reader.searcher();
        let docs = searcher.search(&AllQuery, &DocSetCollector)?;
        for addr in docs {
            let d: TantivyDocument = searcher.doc(addr)?;
            let s = |fld| d.get_first(fld).and_then(|v| v.as_str()).unwrap_or("");
            let ord = d.get_first(self.fields.ord).and_then(|v| v.as_u64()).unwrap_or(0);
            f(s(self.fields.path), ord, s(self.fields.file_type), s(self.fields.body));
        }
        Ok(())
    }

    pub fn delete_path(&self, path: &str) -> anyhow::Result<()> {
        let mut writer = self.index.writer::<TantivyDocument>(50_000_000)?;
        writer.delete_term(tantivy::Term::from_field_text(self.fields.path, path));
        writer.commit()?;
        self.reader.reload()?;
        Ok(())
    }
}

/// Walk `dir`, (re)index changed files, drop removed files, update the manifest.
/// `force = true` ignores the manifest and rebuilds every file.
/// Streams each chunk directly into the tantivy writer + graph (constant memory).
pub fn index_dir(dir: &Path, force: bool) -> anyhow::Result<IndexStats> {
    // force = true is a true from-scratch rebuild: purge the whole index+graph first so stale
    // entries (deleted files, or docs previously indexed under a different path form) cannot
    // linger. Incremental delete-by-path alone can't catch those, since `force` ignores the
    // manifest that records the old paths.
    if force {
        let _ = std::fs::remove_dir_all(dir.join(".glossa"));
    }
    let idx = DocIndex::open_or_create(dir)?;
    let graph = crate::graph::store::GraphStore::open(dir)?;
    let manifest = if force { Manifest::default() } else { Manifest::load(dir) };

    let mut writer = idx.index.writer(50_000_000)?;
    let mut stats = IndexStats::default();
    let mut next = Manifest::default();

    let mut links: Vec<(String, String)> = Vec::new();
    eprintln!("indexing files under {}...", dir.display());
    crate::walk::walk_files(dir, None, true, &mut |path| {
        let path_str = path.to_string_lossy().to_string();
        let sig = match file_sig(path) {
            Ok(s) => s,
            Err(_) => return Ok(()),
        };
        next.files.insert(path_str.clone(), sig);
        if !force && !manifest.changed(&path_str, sig) {
            stats.unchanged += 1;
            return Ok(());
        }
        eprintln!("  + {path_str}");
        writer.delete_term(tantivy::Term::from_field_text(idx.fields.path, &path_str));
        graph.delete_by_source(&path_str)?;
        let mut doc_written = false;
        let mut seq = 0u64;
        let mut prev_sec: Option<String> = None;
        let mut file_links: Vec<String> = Vec::new();
        let mut seen: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        // Index/graph write errors are intentionally not propagated here: one bad chunk must not
        // abort the whole run (matches the prior per-file behavior). The file is still recorded
        // in the manifest; a failed write is corrected on the next `reindex`.
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
            let cur_id = crate::graph::build::section_id(&path_str, &c.location);
            if let Some(prev) = prev_sec.as_deref() {
                let _ = crate::graph::build::link_sequential(&graph, prev, &cur_id, sig, &path_str);
            }
            if let Some(parent) = crate::graph::build::nearest_ancestor(&seen, &c.location) {
                let _ = crate::graph::build::link_parent(&graph, &cur_id, &parent, sig, &path_str);
            }
            file_links.extend(crate::extract::links::extract_links(&c.text));
            seen.insert(c.location.clone(), cur_id.clone());
            prev_sec = Some(cur_id);
        })?;
        for t in file_links {
            links.push((path_str.clone(), t));
        }
        stats.added += 1;
        Ok(())
    })?;

    for old_path in manifest.files.keys() {
        if !next.files.contains_key(old_path) {
            writer.delete_term(tantivy::Term::from_field_text(idx.fields.path, old_path.as_str()));
            graph.delete_by_source(old_path)?;
            stats.removed += 1;
        }
    }
    // Cross-document REFERENCES: resolve collected link targets against indexed documents.
    let mut by_canon: std::collections::HashMap<std::path::PathBuf, String> = std::collections::HashMap::new();
    for p in next.files.keys() {
        if let Ok(c) = std::fs::canonicalize(p) {
            by_canon.insert(c, p.clone());
        }
    }
    for (src, target) in &links {
        let src_dir = std::path::Path::new(src).parent().unwrap_or_else(|| std::path::Path::new("."));
        if let Ok(canon) = std::fs::canonicalize(src_dir.join(target)) {
            if let Some(dst) = by_canon.get(&canon) {
                // Only link to a real Document node — a file with no extractable chunks is in
                // `next.files` but never got a node (build_document fires on the first chunk).
                if dst != src && matches!(graph.get_node(dst), Ok(Some(_))) {
                    let sig = next.files.get(src).copied().unwrap_or(FileSig { mtime_secs: 0, size: 0 });
                    let _ = crate::graph::build::link_reference(&graph, src, dst, sig);
                }
            }
        }
    }
    writer.commit()?;
    next.save(dir)?;
    Ok(stats)
}

#[cfg(test)]
mod incremental_tests {
    use super::*;
    use std::fs;

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
    fn index_dir_builds_sequential_and_hierarchy_edges() {
        use crate::graph::build::section_id;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nintro\n## B\nbody b\n## C\nbody c\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        let p = dir.path().join("a.md").to_string_lossy().to_string();
        let a = section_id(&p, "A");
        let ab = section_id(&p, "A > B");
        let ac = section_id(&p, "A > C");
        // sequential: A -> A>B -> A>C reachable from A's section via outgoing edges
        let na = crate::graph::traverse::neighbors(&g, &a, None, 1).unwrap();
        assert!(na.contains(&ab), "A neighbors include next/child A>B: {na:?}");
        // hierarchy: A>B's parent A is reachable
        let nab = crate::graph::traverse::neighbors(&g, &ab, None, 1).unwrap();
        assert!(nab.contains(&a), "A>B neighbors include parent A: {nab:?}");
        assert!(nab.contains(&ac), "A>B neighbors include next sibling A>C: {nab:?}");
    }

    #[test]
    fn index_dir_builds_cross_document_references() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nsee [the manual](b.md) and [ext](https://x.com)\n").unwrap();
        std::fs::write(dir.path().join("b.md"), b"# B\ncontent\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        let a = dir.path().join("a.md").to_string_lossy().to_string();
        let b = dir.path().join("b.md").to_string_lossy().to_string();
        let na = crate::graph::traverse::neighbors(&g, &a, None, 1).unwrap();
        assert!(na.contains(&b), "a.md REFERENCES b.md: {na:?}");
        assert!(!na.iter().any(|n| n.contains("x.com")), "external URL is not a REFERENCES edge: {na:?}");
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
        assert!(idx.search("alphaword", 10).unwrap().is_empty(), "removed file purged on force reindex");
        assert!(!idx.search("bravoword", 10).unwrap().is_empty(), "kept file still indexed");
    }

    #[test]
    fn reindex_picks_up_changes_and_skips_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.md"), "# T\nдоговоры поставка\n").unwrap();

        let s1 = index_dir(dir.path(), false).unwrap();
        assert_eq!(s1.added, 1);

        let s2 = index_dir(dir.path(), false).unwrap();
        assert_eq!(s2.unchanged, 1);
        assert_eq!(s2.added, 0);

        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let hits = idx.search("договор", 10).unwrap();
        assert!(hits.iter().any(|h| h.path.ends_with("a.md")));
    }

    #[test]
    fn index_dir_indexes_loose_images() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("Схемы")).unwrap();
        std::fs::write(dir.path().join("Схемы").join("profibus.png"), b"\x89PNG\r\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(idx.search("profibus", 10).unwrap().iter().any(|h| h.path.ends_with("profibus.png")),
            "loose image is searchable by name");
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
        idx.write_chunks(&[page("p.1", "first page body"), page("p.2", "second page body")])
            .unwrap();

        // Exact path+location returns that chunk's stored body — no file re-parse.
        assert_eq!(idx.read_chunk("doc.pdf", "p.2").unwrap().as_deref(), Some("second page body"));
        // Unknown location -> None, so the caller falls back to reading the file.
        assert_eq!(idx.read_chunk("doc.pdf", "p.99").unwrap(), None);
    }

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
        assert!(!md.display_line().contains("· md ·"), "file_type must not leak as label in non-paged line: {}", md.display_line());
    }

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

    #[test]
    fn iter_chunks_visits_every_stored_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            Chunk { doc_path: PathBuf::from("a.md"), location: "S1".into(), file_type: "md".into(), text: "alpha".into() },
            Chunk { doc_path: PathBuf::from("a.md"), location: "S2".into(), file_type: "md".into(), text: "beta".into() },
        ]).unwrap();
        let mut seen: Vec<(u64, String)> = Vec::new();
        idx.iter_chunks(|_path, ord, _ft, body| seen.push((ord, body.to_string()))).unwrap();
        seen.sort();
        assert_eq!(seen, vec![(1, "alpha".to_string()), (2, "beta".to_string())]);
    }

    #[test]
    fn location_ord_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let i = DocIndex::open_or_create(dir.path()).unwrap();
        i.write_chunks(&[
            crate::model::Chunk { doc_path: "d.md".into(), location: "A > B".into(), file_type: "md".into(), text: "x".into() },
        ]).unwrap();
        let n = i.ord_for_location("d.md", "A > B").unwrap().unwrap();
        assert_eq!(i.location_for_ord("d.md", n).unwrap().as_deref(), Some("A > B"));
        assert_eq!(i.ord_for_location("d.md", "missing").unwrap(), None);
    }

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
