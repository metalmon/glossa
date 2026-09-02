//! A tantivy index over graph-node labels/aliases, so `glossary`/`resolve` can match a node from
//! a natural-language phrase by BM25 (morphology-aware, ranked) instead of a strict term-subset.
//! Reasoning nodes live in SQLite; this is the search view over their names. It is DERIVED — the
//! `GraphStore` rebuilds it whenever it falls out of sync with the node table, so it never needs a
//! migration and a missing/stale index self-heals.

use crate::index::multilang::{default_detector, multilang_analyzer};
use anyhow::Context;
use std::path::Path;
use tantivy::collector::{Count, TopDocs};
use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
use tantivy::schema::{
    Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value, STORED, STRING,
};
use tantivy::{Index, IndexReader, TantivyDocument, TantivyError, Term};

/// Default `df / total_nodes` ceiling for [`NodeIndex::is_salient`] — a term present in more than
/// this fraction of nodes reads as ordinary vocabulary ("member", "district"), not a discriminating
/// proper-noun-like mention worth trusting for a cross-document bridge. `GraphStore::term_is_salient`
/// uses this as its default.
pub const DEFAULT_SALIENCE_MAX_DF_RATIO: f64 = 0.1;

/// Absolute df ceiling for [`NodeIndex::is_salient`], on top of the ratio. A discriminating
/// proper-noun-like mention should resolve into a small handful of nodes; guards large corpora
/// where the ratio alone would still pass a term through with a big raw df (0.1 * 10,000 nodes =
/// 1,000 nodes is not "rare" by any reasonable reading). A term appearing in more nodes than this
/// is never treated as salient, no matter how small the ratio comes out.
pub const SALIENCE_MAX_ABS_DF: usize = 25;

pub struct NodeIndex {
    index: Index,
    reader: IndexReader,
    id: Field,
    text: Field,
    /// Sidecar holding the node-table content signature this index was last rebuilt from, so
    /// `resolve` can detect drift the raw doc count misses (see `GraphStore::resolve`).
    sig_path: std::path::PathBuf,
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
        Ok(NodeIndex {
            index,
            reader,
            id,
            text,
            sig_path: p.with_extension("sig"),
        })
    }

    /// The node-table content signature this index was last rebuilt from, if recorded.
    pub fn built_sig(&self) -> Option<u64> {
        std::fs::read_to_string(&self.sig_path)
            .ok()?
            .trim()
            .parse()
            .ok()
    }

    /// Persist the content signature the index was just rebuilt from, beside the segment dir.
    pub fn set_built_sig(&self, sig: u64) -> anyhow::Result<()> {
        std::fs::write(&self.sig_path, sig.to_string())
            .with_context(|| format!("write node-index sig {:?}", self.sig_path))?;
        Ok(())
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

    /// BM25 search over node text; returns node ids best-first. Returns EVERY node that shares at
    /// least one query term (recall first), ranked by BM25 — so a node matching several rare query
    /// terms outranks one matching a single generic word (a brand name present in most labels), and the caller
    /// sees the strongest matches at the top. A query that tokenizes to nothing returns empty.
    pub fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<String>> {
        // Tokenize the query with the index's own analyzer so query terms match indexed terms
        // (same morphology + language detection as index time).
        let terms = self.tokenize(query);
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let clauses: Vec<(Occur, Box<dyn Query>)> = terms
            .iter()
            .map(|t| {
                let q: Box<dyn Query> = Box::new(TermQuery::new(
                    Term::from_field_text(self.text, t),
                    IndexRecordOption::WithFreqs,
                ));
                (Occur::Should, q)
            })
            .collect();
        // OR semantics (match any term); BM25 does the ranking. No minimum-should-match floor —
        // surface everything that fits and let the ranking (and the reader) judge relevance.
        let bq = BooleanQuery::new(clauses);
        let searcher = self.reader.searcher();
        let top = searcher.search(&bq, &TopDocs::with_limit(limit.max(1)).order_by_score())?;
        let mut scored: Vec<(f32, String)> = Vec::with_capacity(top.len());
        for (score, addr) in top {
            let d: TantivyDocument = searcher.doc(addr)?;
            if let Some(v) = d.get_first(self.id).and_then(|v| v.as_str()) {
                scored.push((score, v.to_string()));
            }
        }
        // Deterministic tie-break: BM25 score descending, then node id ascending. tantivy breaks an
        // equal-score tie by internal DocId, which depends on index-build order and quantized
        // fieldnorms — two near-identical labels ("<field>" vs "<field> enum") that tie on score
        // would otherwise flip their order run to run (the Windows node-index flake). Ordering equal
        // scores by the stable node id makes the top hit deterministic regardless of tantivy internals.
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        Ok(scored.into_iter().map(|(_, id)| id).collect())
    }

    /// Tokenize `text` with the index's own analyzer (same morphology + language detection as
    /// index time), deduplicated, order preserved. Shared by `search` and `doc_freq` so a term
    /// always maps to the same indexed form.
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut terms: Vec<String> = Vec::new();
        if let Some(mut analyzer) = self.index.tokenizers().get("multilang") {
            let mut stream = analyzer.token_stream(text);
            while stream.advance() {
                let t = stream.token().text.clone();
                if !terms.contains(&t) {
                    terms.push(t);
                }
            }
        }
        terms
    }

    /// How many indexed nodes contain `term` (its document frequency) — the raw signal behind
    /// [`Self::is_salient`]. `term` is run through the same analyzer as `search`/index time, so
    /// morphology and language detection stay consistent; a term that tokenizes to nothing (empty
    /// or punctuation-only) has df 0. If `term` tokenizes to MULTIPLE sub-tokens — the common case,
    /// since the bridge caller's "term" is usually a multi-word node LABEL (e.g. "Meridian Falls
    /// District") — this counts nodes containing ALL sub-tokens (AND / co-occurrence), NOT nodes
    /// matching any single one. An OR would let one common word in the phrase ("District") swamp
    /// the count with unrelated nodes and push a genuinely distinctive multi-word mention over the
    /// salience ratio/cap, suppressing legitimate bridging — the opposite of what this gate is for.
    /// AND is an approximation (it ignores word order/adjacency; a `PhraseQuery` would be tighter)
    /// but is a correct upper bound on "nodes that could plausibly be about this phrase" and is
    /// cheap to reason about; a future revision can tighten it to a `PhraseQuery` if this proves
    /// too loose in practice.
    pub fn doc_freq(&self, term: &str) -> usize {
        let terms = self.tokenize(term);
        if terms.is_empty() {
            return 0;
        }
        let searcher = self.reader.searcher();
        if terms.len() == 1 {
            let t = Term::from_field_text(self.text, &terms[0]);
            return searcher.doc_freq(&t).unwrap_or(0) as usize;
        }
        let clauses: Vec<(Occur, Box<dyn Query>)> = terms
            .iter()
            .map(|t| {
                let q: Box<dyn Query> = Box::new(TermQuery::new(
                    Term::from_field_text(self.text, t),
                    IndexRecordOption::Basic,
                ));
                (Occur::Must, q)
            })
            .collect();
        let bq = BooleanQuery::new(clauses);
        searcher.search(&bq, &Count).unwrap_or(0)
    }

    /// True when `term` is DISCRIMINATING enough to trust for a cross-document bridge: it occurs
    /// in a small slice of the corpus (`df / total_nodes < max_df_ratio`), at least once, and under
    /// the absolute cap ([`SALIENCE_MAX_ABS_DF`]) regardless of how the ratio comes out. False for
    /// an empty index (nothing is discriminating against zero context) and for a term that doesn't
    /// tokenize to anything indexable.
    pub fn is_salient(&self, term: &str, max_df_ratio: f64) -> bool {
        let total = self.num_docs();
        if total == 0 {
            return false;
        }
        let df = self.doc_freq(term);
        if df == 0 || df > SALIENCE_MAX_ABS_DF {
            return false;
        }
        (df as f64) / (total as f64) < max_df_ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic fixture: an invented rare, proper-noun-like mention ("Zephyrine") appears in 2 of
    /// 10 nodes; an ordinary word ("council") appears in all 10 — the shape a real cross-document
    /// bridge gate must tell apart (bridge on the former, refuse the latter).
    fn rare_vs_common_fixture(dir: &Path) -> NodeIndex {
        let idx = NodeIndex::open_or_create(dir).unwrap();
        let mut docs: Vec<(String, Vec<String>)> = vec![
            ("n:1".into(), vec!["Zephyrine convened the council".into()]),
            ("n:2".into(), vec!["Zephyrine addressed the council".into()]),
        ];
        for i in 3..=10 {
            docs.push((format!("n:{i}"), vec!["The council met again".into()]));
        }
        idx.rebuild(&docs).unwrap();
        idx
    }

    #[test]
    fn rare_term_has_lower_df_than_common_term() {
        let dir = tempfile::tempdir().unwrap();
        let idx = rare_vs_common_fixture(dir.path());

        let rare_df = idx.doc_freq("Zephyrine");
        let common_df = idx.doc_freq("council");
        assert!(
            rare_df < common_df,
            "rare df ({rare_df}) must be lower than common df ({common_df})"
        );
        assert_eq!(rare_df, 2, "Zephyrine appears in exactly 2 of 10 nodes");
        assert_eq!(common_df, 10, "council appears in all 10 nodes");
    }

    #[test]
    fn is_salient_gates_rare_in_common_out() {
        let dir = tempfile::tempdir().unwrap();
        let idx = rare_vs_common_fixture(dir.path());

        let ratio = 0.3;
        assert!(
            idx.is_salient("Zephyrine", ratio),
            "a term in 2/10 nodes is discriminating at a 0.3 ratio"
        );
        assert!(
            !idx.is_salient("council", ratio),
            "a term in all 10/10 nodes is ordinary vocabulary, not a bridge-worthy mention"
        );
    }

    #[test]
    fn is_salient_multiword_phrase_not_swamped_by_common_subtoken() {
        // A distinctive multi-word LABEL ("Meridian Falls District") — exactly what `reach` bridges
        // on — contains a common sub-token ("District") that, alone, appears in many unrelated
        // nodes. df for the whole phrase must count nodes that mention the WHOLE phrase (word
        // co-occurrence), NOT any single sub-token — otherwise the common sub-token swamps the
        // count and a real distinctive mention gets misreported as non-salient.
        let dir = tempfile::tempdir().unwrap();
        let idx = NodeIndex::open_or_create(dir.path()).unwrap();
        let mut docs: Vec<(String, Vec<String>)> = vec![
            ("n:1".into(), vec!["Meridian Falls District office".into()]),
            (
                "n:2".into(),
                vec!["Meridian Falls District courthouse".into()],
            ),
        ];
        for i in 3..=20 {
            docs.push((format!("n:{i}"), vec!["The District held a meeting".into()]));
        }
        idx.rebuild(&docs).unwrap();

        // OR-over-subtokens would count all 20 nodes (every node contains "District"); the correct
        // whole-phrase (AND) count is the 2 nodes that mention all three words together.
        assert_eq!(
            idx.doc_freq("Meridian Falls District"),
            2,
            "df must reflect whole-phrase co-occurrence, not the common sub-token's df"
        );
        assert!(
            idx.is_salient("Meridian Falls District", 0.15),
            "a distinctive multi-word phrase confined to 2/20 nodes must read as salient"
        );
    }

    #[test]
    fn is_salient_false_for_unknown_or_unindexable_term() {
        let dir = tempfile::tempdir().unwrap();
        let idx = rare_vs_common_fixture(dir.path());

        assert_eq!(idx.doc_freq("Nonexistentia"), 0);
        assert!(!idx.is_salient("Nonexistentia", 0.9));
        // Punctuation-only query tokenizes to nothing.
        assert_eq!(idx.doc_freq("   "), 0);
        assert!(!idx.is_salient("   ", 0.9));
    }

    #[test]
    fn is_salient_respects_absolute_cap_even_under_ratio() {
        // A term present in most nodes of a LARGE graph can still clear a loose ratio while
        // clearly not being a rare, discriminating mention — the absolute cap must catch it.
        let dir = tempfile::tempdir().unwrap();
        let idx = NodeIndex::open_or_create(dir.path()).unwrap();
        let total = (SALIENCE_MAX_ABS_DF + 20) * 10; // df/total stays well under any loose ratio
        let common_df = SALIENCE_MAX_ABS_DF + 5; // over the absolute cap
        let mut docs: Vec<(String, Vec<String>)> = Vec::with_capacity(total);
        for i in 0..common_df {
            docs.push((format!("n:common:{i}"), vec!["Ubiquitine marker".into()]));
        }
        for i in 0..(total - common_df) {
            docs.push((
                format!("n:other:{i}"),
                vec!["Unrelated filler content".into()],
            ));
        }
        idx.rebuild(&docs).unwrap();

        let ratio = 0.9; // deliberately loose — only the absolute cap should block this
        assert!(
            (idx.doc_freq("Ubiquitine") as f64) / (idx.num_docs() as f64) < ratio,
            "fixture must clear the ratio on its own so only the absolute cap is under test"
        );
        assert!(!idx.is_salient("Ubiquitine", ratio));
    }
}
