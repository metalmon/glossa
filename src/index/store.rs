use crate::index::multilang::{default_detector, multilang_analyzer};
use crate::model::Chunk;
use anyhow::Context;
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, STORED, STRING};
use tantivy::schema::Value;
use tantivy::snippet::SnippetGenerator;
use tantivy::{doc, Index, TantivyDocument, TantivyError};

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
        Ok(DocIndex { index, fields })
    }
}

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
        let top = searcher.search(&parsed, &TopDocs::with_limit(limit).order_by_score())?;

        let snippet_gen = SnippetGenerator::create(&searcher, &*parsed, self.fields.body)?;

        let mut hits = Vec::with_capacity(top.len());
        for (score, addr) in top {
            let d: TantivyDocument = searcher.doc(addr)?;
            let get = |f: tantivy::schema::Field| -> String {
                d.get_first(f).and_then(|v| v.as_str()).unwrap_or("").to_string()
            };
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
        let mut writer = self.index.writer::<TantivyDocument>(50_000_000)?;
        writer.delete_term(tantivy::Term::from_field_text(self.fields.path, path));
        writer.commit()?;
        Ok(())
    }
}

/// Walk `dir`, (re)index changed files, drop removed files, update the manifest.
/// `force = true` ignores the manifest and rebuilds every file.
pub fn index_dir(dir: &Path, force: bool) -> anyhow::Result<IndexStats> {
    let idx = DocIndex::open_or_create(dir)?;
    let graph = crate::graph::store::GraphStore::open(dir)?;
    let manifest = if force { Manifest::default() } else { Manifest::load(dir) };
    let chunks = collect_chunks(dir, None, true)?;

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
        idx.delete_path(path)?;
        idx.write_chunks(file_chunks)?;
        graph.delete_by_source(path)?;
        crate::graph::build::build_structural(&graph, file_chunks, sig)?;
        stats.added += 1;
    }

    for old_path in manifest.files.keys() {
        if !next.files.contains_key(old_path) {
            idx.delete_path(old_path)?;
            graph.delete_by_source(old_path)?;
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
