//! #2 / #3 — Label and evidence similarity (embeddings-free).
//!
//! `label_jaccard` approximates BM25-over-labels with the Jaccard of stemmed term sets (same
//! morphology pipeline as search, via `TermAnalyzer`) — catches shared-word paraphrases. A real
//! tantivy-BM25 label index can replace it later behind the same output.
//! `shared_evidence` (#3) links nodes that MENTION the same chunk anchor — catches paraphrases with
//! NO shared words. Both emit candidate SIMILAR pairs for the merge / edge passes.

use crate::index::multilang::{default_detector, multilang_analyzer, TermAnalyzer};
use std::collections::BTreeSet;
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
use tantivy::schema::{IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value, STORED, STRING};
use tantivy::{doc, Index, TantivyDocument, Term};

/// #2 — Real BM25-over-labels via an in-RAM tantivy index using the SAME `multilang` analyzer as the
/// search index (morphology matches). For each label, BM25-search the others; a hit is kept when its
/// score is at least `min_ratio` of the query label's OWN self-score — a RELATIVE, corpus-independent
/// threshold (the absolute BM25 scale varies with corpus/IDF, so a relative cut is far more robust).
/// BM25's IDF down-weights generic shared words (unlike unweighted `label_jaccard`). The per-label
/// query is built from the analyzed (stemmed) terms — robust to punctuation (no query-parser).
/// Returns `(id_a, id_b, ratio)` (a<b, ratio in 0..1 = max over both query directions), score desc.
pub fn label_bm25(
    labels: &[(String, String)],
    min_ratio: f64,
    top_k: usize,
) -> anyhow::Result<Vec<(String, String, f64)>> {
    use std::collections::HashMap;
    if labels.len() < 2 {
        return Ok(Vec::new());
    }
    let mut sb = Schema::builder();
    let id_f = sb.add_text_field("id", STRING | STORED);
    let label_f = sb.add_text_field(
        "label",
        TextOptions::default().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("multilang")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        ),
    );
    let index = Index::create_in_ram(sb.build());
    index
        .tokenizers()
        .register("multilang", multilang_analyzer(default_detector()));
    {
        let mut w = index.writer_with_num_threads(1, 15_000_000)?;
        for (id, label) in labels {
            w.add_document(doc!(id_f => id.clone(), label_f => label.clone()))?;
        }
        w.commit()?;
    }
    let searcher = index.reader()?.searcher();
    let analyzer = TermAnalyzer::new();
    let mut best: HashMap<(String, String), f64> = HashMap::new();
    for (id, label) in labels {
        let mut terms = BTreeSet::new();
        analyzer.terms(label, &mut terms);
        if terms.is_empty() {
            continue;
        }
        let subs: Vec<(Occur, Box<dyn Query>)> = terms
            .iter()
            .map(|t| {
                (
                    Occur::Should,
                    Box::new(TermQuery::new(
                        Term::from_field_text(label_f, t),
                        IndexRecordOption::WithFreqs,
                    )) as Box<dyn Query>,
                )
            })
            .collect();
        let query = BooleanQuery::new(subs);
        let hits = searcher.search(&query, &TopDocs::with_limit(top_k + 1).order_by_score())?;
        // self matches all its own query terms → highest score = the self-score baseline
        let mut self_score = 0.0f64;
        let mut others: Vec<(String, f64)> = Vec::new();
        for (score, addr) in hits {
            let d: TantivyDocument = searcher.doc(addr)?;
            let other = d.get_first(id_f).and_then(|v| v.as_str()).unwrap_or_default().to_string();
            if other.is_empty() {
                continue;
            }
            if other == *id {
                self_score = self_score.max(score as f64);
            } else {
                others.push((other, score as f64));
            }
        }
        if self_score <= 0.0 {
            continue;
        }
        for (other, score) in others {
            let ratio = score / self_score;
            if ratio < min_ratio {
                continue;
            }
            let (a, b) = order(id, &other);
            best.entry((a, b))
                .and_modify(|s| {
                    if ratio > *s {
                        *s = ratio;
                    }
                })
                .or_insert(ratio);
        }
    }
    let mut out: Vec<(String, String, f64)> =
        best.into_iter().map(|((a, b), s)| (a, b, s)).collect();
    out.sort_by(|x, y| y.2.total_cmp(&x.2).then(x.0.cmp(&y.0)).then(x.1.cmp(&y.1)));
    Ok(out)
}

fn order(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

/// Pairwise label similarity by Jaccard of stemmed term sets. `labels`: `(id, label)`. Returns
/// `(id_a, id_b, score)` with id_a < id_b and `score >= threshold`, sorted by score desc.
pub fn label_jaccard(labels: &[(String, String)], threshold: f64) -> Vec<(String, String, f64)> {
    let analyzer = TermAnalyzer::new();
    let termsets: Vec<(&String, BTreeSet<String>)> = labels
        .iter()
        .map(|(id, label)| {
            let mut s = BTreeSet::new();
            analyzer.terms(label, &mut s);
            (id, s)
        })
        .collect();
    let mut out = Vec::new();
    for i in 0..termsets.len() {
        for j in (i + 1)..termsets.len() {
            let (ia, sa) = &termsets[i];
            let (ib, sb) = &termsets[j];
            if sa.is_empty() || sb.is_empty() {
                continue;
            }
            let inter = sa.intersection(sb).count();
            if inter == 0 {
                continue;
            }
            let union = sa.union(sb).count();
            let score = inter as f64 / union as f64;
            if score >= threshold {
                let (a, b) = order(ia, ib);
                out.push((a, b, score));
            }
        }
    }
    out.sort_by(|x, y| y.2.total_cmp(&x.2).then(x.0.cmp(&y.0)).then(x.1.cmp(&y.1)));
    out
}

/// Shared-evidence pairs: two nodes that reference the same chunk anchor are about the same thing.
/// `anchors`: `(node_id, anchor)` rows (a node may appear multiple times). Returns unique `(a < b)`
/// pairs that share ≥1 anchor, sorted.
pub fn shared_evidence(anchors: &[(String, String)]) -> Vec<(String, String)> {
    use std::collections::{HashMap, HashSet};
    let mut by_anchor: HashMap<&String, Vec<&String>> = HashMap::new();
    for (node, anchor) in anchors {
        by_anchor.entry(anchor).or_default().push(node);
    }
    let mut pairs: HashSet<(String, String)> = HashSet::new();
    for nodes in by_anchor.values() {
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                if nodes[i] != nodes[j] {
                    pairs.insert(order(nodes[i], nodes[j]));
                }
            }
        }
    }
    let mut v: Vec<(String, String)> = pairs.into_iter().collect();
    v.sort();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_jaccard_links_paraphrase_not_unrelated() {
        let labels = vec![
            ("s1".to_string(), "Потеря связи Modbus".to_string()),
            ("s2".to_string(), "Modbus потеря связи периодическая".to_string()),
            ("x".to_string(), "Замена предохранителя".to_string()),
        ];
        let pairs = label_jaccard(&labels, 0.4);
        assert!(
            pairs.iter().any(|(a, b, _)| a == "s1" && b == "s2"),
            "paraphrase pair expected, got {pairs:?}"
        );
        assert!(!pairs.iter().any(|(a, b, _)| a == "x" || b == "x"));
    }

    #[test]
    fn shared_evidence_links_nodes_on_same_chunk() {
        let anchors = vec![
            ("s1".to_string(), "doc#12".to_string()),
            ("s2".to_string(), "doc#12".to_string()),
            ("s3".to_string(), "doc#99".to_string()),
        ];
        assert_eq!(
            shared_evidence(&anchors),
            vec![("s1".to_string(), "s2".to_string())]
        );
    }

    #[test]
    fn label_bm25_links_paraphrase_not_unrelated() {
        let labels = vec![
            ("s1".to_string(), "Потеря связи Modbus".to_string()),
            ("s2".to_string(), "Modbus периодическая потеря связи".to_string()),
            ("x".to_string(), "Замена предохранителя питания".to_string()),
        ];
        // relative threshold: paraphrase scores a high fraction of self-score; unrelated never matches
        let pairs = label_bm25(&labels, 0.3, 5).unwrap();
        assert!(
            pairs.iter().any(|(a, b, r)| a == "s1" && b == "s2" && *r <= 1.0 && *r >= 0.3),
            "paraphrase pair expected with ratio, got {pairs:?}"
        );
        assert!(!pairs.iter().any(|(a, b, _)| a == "x" || b == "x"));
    }
}
