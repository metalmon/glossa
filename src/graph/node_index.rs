//! A tantivy index over graph-node labels/aliases, so `glossary`/`resolve` can match a node from
//! a natural-language phrase by BM25 (morphology-aware, ranked) instead of a strict term-subset.
//! Reasoning nodes live in SQLite; this is the search view over their names. It is DERIVED — the
//! `GraphStore` rebuilds it whenever it falls out of sync with the node table, so it never needs a
//! migration and a missing/stale index self-heals.

use crate::index::multilang::{default_detector, multilang_analyzer};
use anyhow::Context;
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value, STORED, STRING};
use tantivy::{Index, IndexReader, TantivyDocument, TantivyError};

pub struct NodeIndex {
    index: Index,
    reader: IndexReader,
    id: Field,
    text: Field,
}

fn nodes_dir(dir: &Path) -> std::path::PathBuf {
    dir.join(".glossa").join("nodes")
}

impl NodeIndex {
    pub fn open_or_create(dir: &Path) -> anyhow::Result<NodeIndex> {
        let mut sb = Schema::builder();
        let id = sb.add_text_field("id", STRING | STORED);
        let text_opts = TextOptions::default().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("multilang")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        );
        let text = sb.add_text_field("text", text_opts);
        let schema = sb.build();
        let p = nodes_dir(dir);
        std::fs::create_dir_all(&p).with_context(|| format!("create {p:?}"))?;
        let index = match Index::create_in_dir(&p, schema.clone()) {
            Ok(i) => i,
            Err(TantivyError::IndexAlreadyExists) => Index::open_in_dir(&p)?,
            Err(e) => return Err(e.into()),
        };
        index
            .tokenizers()
            .register("multilang", multilang_analyzer(default_detector()));
        let reader = index.reader()?;
        Ok(NodeIndex { index, reader, id, text })
    }

    /// Number of indexed nodes (non-deleted). The `GraphStore` compares this to the node-table
    /// count to decide whether to rebuild.
    pub fn num_docs(&self) -> u64 {
        self.reader.searcher().num_docs()
    }

    /// Replace the whole index with `docs` — each `(node id, [label, alias, …])`. Each text is
    /// added as a SEPARATE value of the `text` field so the per-value language detector classifies
    /// label and aliases independently (a Russian alias on an English-leaning label must still be
    /// stemmed as Russian — otherwise a Russian query never matches it).
    pub fn rebuild(&self, docs: &[(String, Vec<String>)]) -> anyhow::Result<()> {
        let mut writer = self.index.writer(15_000_000)?;
        writer.delete_all_documents()?;
        for (id, texts) in docs {
            let mut d = TantivyDocument::default();
            d.add_text(self.id, id);
            for t in texts {
                d.add_text(self.text, t);
            }
            writer.add_document(d)?;
        }
        writer.commit()?;
        self.reader.reload()?;
        Ok(())
    }

    /// BM25 search over node text; returns node ids best-first (OR semantics — any shared term
    /// contributes, ranked by overlap and term rarity). A query that parses to nothing (only
    /// stopwords/punctuation) or fails to parse returns empty.
    pub fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<String>> {
        let searcher = self.reader.searcher();
        let parser = QueryParser::for_index(&self.index, vec![self.text]);
        let parsed = match parser.parse_query(query) {
            Ok(q) => q,
            Err(_) => return Ok(Vec::new()),
        };
        let top = searcher.search(&parsed, &TopDocs::with_limit(limit.max(1)).order_by_score())?;
        let mut ids = Vec::with_capacity(top.len());
        for (_score, addr) in top {
            let d: TantivyDocument = searcher.doc(addr)?;
            if let Some(v) = d.get_first(self.id).and_then(|v| v.as_str()) {
                ids.push(v.to_string());
            }
        }
        Ok(ids)
    }
}
