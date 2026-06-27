//! Integration pass — run the generalization techniques over a live `GraphStore` and persist
//! the results. Wired (later) into a `kb graph generalize` step after enrich/reindex.
//!
//! - Derived edges (closure #8, SIMILAR from link-prediction #4) are written with
//!   `origin = "auto-generalized"` and cleared/regenerated on each run (NOT the document `auto-*`
//!   layer, so reindex's structural rebuild and this pass don't fight).
//! - Communities (#6) + centrality (#7) go to the `node_meta` side table.
//! - Near-dup MERGE (#3 shared-evidence, Tier-1) is always COMPUTED, but only APPLIED when
//!   `opts.apply_merges` is set — it mutates/deletes agent nodes, so the safe default is report-only.

use super::{centrality, closure, community, linkpred, merge, similarity, Triple};
use crate::graph::ontology::Ontology;
use crate::graph::store::{Edge, GraphStore, NodeMeta, Provenance};
use std::collections::HashMap;

const SIMILAR: &str = "SIMILAR";
const DERIVED_ORIGIN: &str = "auto-generalized";

/// Tunables for the pass. `mentions_type` and the closure `rules` mirror the ontology's relation
/// names; defaults match the current reasoning ontology (TODO: source from `ontology.toml`).
pub struct Opts {
    pub apply_merges: bool,
    /// Jaccard threshold (0..1) for emitting a structural-similarity (link-prediction) SIMILAR edge.
    pub similar_threshold: f64,
    /// BM25-over-labels: RELATIVE threshold (fraction of a label's own self-score, 0..1) and top-k
    /// per label for a label-similarity SIMILAR edge. **Default `0.3`** — a label scoring ≥30% of
    /// another's self-score shares enough weighted (IDF-discounted) terms to be worth a soft SIMILAR
    /// link; low enough to surface partial-overlap paraphrases, high enough to drop one-word coincidences.
    pub bm25_min_ratio: f64,
    pub bm25_top_k: usize,
    /// BM25-over-labels MERGE threshold — a (much) higher RELATIVE ratio than `bm25_min_ratio`: two
    /// same-type nodes whose labels score this fraction of self-score are treated as near-duplicates
    /// (the 4B self-paraphrase pattern) and join the merge candidates. Higher = safer (destructive).
    /// **Default `0.7`** — near-identical labels (the 4B re-emits the same fact with a word reordered
    /// or added) score ≳70% of self; below that the risk of collapsing two genuinely distinct nodes
    /// outweighs the dedup gain. Both ratios are corpus-independent (unlike absolute BM25), so these
    /// defaults port across deployments — tune on real data, but `0.3 / 0.7` is the safe starting point.
    pub merge_bm25_min_ratio: f64,
    /// Edge type whose target chunk is a reasoning node's evidence anchor (shared-evidence merge).
    /// Sourced from the ontology's `[reasoning].mentions` (was a hardcoded "MENTIONS").
    pub mentions_type: String,
    /// Transitive-closure composition rules `(a, b, result)`, from `[reasoning].closure`.
    pub closure_rules: Vec<(String, String, String)>,
    /// Structural (never-reasoning) types excluded from merge/similar, from `[reasoning].structural`.
    pub structural: Vec<String>,
    pub now: u64,
}

impl Opts {
    /// Test/default tunables with NO domain rules — empty closure, default structural set,
    /// `"MENTIONS"`. Production builds via `from_ontology`.
    pub fn defaults(now: u64) -> Self {
        Opts {
            apply_merges: false,
            similar_threshold: 0.5,
            bm25_min_ratio: 0.3,
            bm25_top_k: 5,
            merge_bm25_min_ratio: 0.7,
            mentions_type: "MENTIONS".into(),
            closure_rules: vec![],
            structural: ["Document", "Section", "Term", "Topic"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            now,
        }
    }

    /// Source the domain rules (closure, mentions anchor, structural types) from the ontology's
    /// `[reasoning]` section; other tunables keep their defaults.
    pub fn from_ontology(ont: &Ontology, now: u64) -> Self {
        Opts {
            mentions_type: ont.mentions().to_string(),
            closure_rules: ont.closure_rules(),
            structural: ont.structural(),
            ..Opts::defaults(now)
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub inferred_edges: usize,
    pub similar_edges: usize,
    pub communities: usize,
    pub merge_candidates: usize,
    pub merged_nodes: usize,
}

fn derived_prov(now: u64, confidence: f32) -> Provenance {
    Provenance {
        source_path: DERIVED_ORIGIN.into(),
        range: None,
        file_sig: None,
        origin: DERIVED_ORIGIN.into(),
        confidence,
        created_at: now,
    }
}

/// Run the full generalization pass against `g`, persisting derived edges + `node_meta`, and merge
/// candidates (applied only when `opts.apply_merges`). Returns counts.
pub fn generalize(g: &GraphStore, opts: &Opts) -> anyhow::Result<Report> {
    let mut report = Report::default();

    // (Tier 1) near-dup MERGE from shared evidence: reasoning nodes that MENTION the same chunk.
    // Computed first (before deriving edges) on the original graph; applied only if requested.
    {
        let nodes = g.all_nodes()?;
        let type_of: HashMap<String, String> =
            nodes.iter().map(|n| (n.id.clone(), n.node_type.clone())).collect();
        let reasoning = |id: &str| {
            type_of
                .get(id)
                .map(|t| !opts.structural.iter().any(|s| s == t))
                .unwrap_or(false)
        };
        let anchors: Vec<(String, String)> = g
            .all_edges()?
            .into_iter()
            .filter(|e| e.edge_type == opts.mentions_type && reasoning(&e.from))
            .map(|e| (e.from, e.to))
            .collect();
        // Merge candidates from two strong near-dup signals, both restricted to SAME type (never
        // merge a Symptom into a Resolution): (a) shared chunk evidence (#3), and (b) very-high
        // BM25 label similarity (#2) — the 4B's self-paraphrased labels.
        let mut pairs: Vec<(String, String)> = similarity::shared_evidence(&anchors)
            .into_iter()
            .filter(|(a, b)| type_of.get(a) == type_of.get(b))
            .collect();
        let reasoning_labels: Vec<(String, String)> = nodes
            .iter()
            .filter(|n| reasoning(&n.id))
            .map(|n| (n.id.clone(), n.label.clone()))
            .collect();
        for (a, b, _s) in
            similarity::label_bm25(&reasoning_labels, opts.merge_bm25_min_ratio, opts.bm25_top_k)?
        {
            if type_of.get(&a) == type_of.get(&b) {
                pairs.push((a, b));
            }
        }
        let ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
        let groups = merge::merge_groups(&ids, &pairs);
        report.merge_candidates = groups.len();
        if opts.apply_merges {
            for group in &groups {
                // canonical = shortest label (closest superset of truncated paraphrases)
                let canonical = group
                    .iter()
                    .filter_map(|id| g.get_node(id).ok().flatten())
                    .min_by_key(|n| n.label.split_whitespace().count())
                    .map(|n| n.id);
                if let Some(canon) = canonical {
                    let dups: Vec<String> =
                        group.iter().filter(|id| **id != canon).cloned().collect();
                    report.merged_nodes += g.merge_nodes(&canon, &dups)?;
                }
            }
        }
    }

    // Reload after a possible merge, then derive edges + meta on the current graph.
    let nodes = g.all_nodes()?;
    let node_ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
    let edges: Vec<Triple> = g
        .all_edges()?
        .into_iter()
        .map(|e| (e.from, e.edge_type, e.to))
        .collect();

    // clear our own prior derived edges so a re-run doesn't accumulate duplicates
    g.delete_edges_by_origin(DERIVED_ORIGIN)?;

    // (#8) transitive closure → inferred edges (rules from the ontology's [reasoning].closure)
    let rules: Vec<closure::Rule> =
        opts.closure_rules.iter().map(|(a, b, r)| closure::Rule::new(a, b, r)).collect();
    for (f, t, to) in closure::transitive_closure(&edges, &rules) {
        g.put_edge(&Edge { from: f, edge_type: t, to, prov: derived_prov(opts.now, 1.0) })?;
        report.inferred_edges += 1;
    }

    // (#4 link-prediction + #2 BM25-over-labels) → SIMILAR edges, one per UNIQUE pair (keep max score)
    let mut similar: std::collections::BTreeMap<(String, String), f64> =
        std::collections::BTreeMap::new();
    let mut bump = |a: String, b: String, s: f64| {
        similar
            .entry((a, b))
            .and_modify(|v| {
                if s > *v {
                    *v = s;
                }
            })
            .or_insert(s);
    };
    for (a, b, s) in linkpred::jaccard_pairs(&edges, opts.similar_threshold) {
        bump(a, b, s);
    }
    let labels: Vec<(String, String)> = nodes
        .iter()
        .filter(|n| !opts.structural.iter().any(|s| s == &n.node_type))
        .map(|n| (n.id.clone(), n.label.clone()))
        .collect();
    for (a, b, s) in similarity::label_bm25(&labels, opts.bm25_min_ratio, opts.bm25_top_k)? {
        bump(a, b, s);
    }
    for ((a, b), score) in similar {
        g.put_edge(&Edge {
            from: a,
            edge_type: SIMILAR.into(),
            to: b,
            prov: derived_prov(opts.now, score as f32),
        })?;
        report.similar_edges += 1;
    }

    // (#6 + #7) communities + centrality → node_meta
    let comm = community::connected_components(&node_ids, &edges);
    let deg = centrality::degree(&node_ids, &edges);
    let pr = centrality::pagerank(&node_ids, &edges, 0.85, 30);
    let meta: Vec<(String, NodeMeta)> = node_ids
        .iter()
        .map(|id| {
            (
                id.clone(),
                NodeMeta {
                    community: comm.get(id).map(|&c| c as i64),
                    pagerank: pr.get(id).copied(),
                    degree: deg.get(id).map(|&d| d as i64),
                },
            )
        })
        .collect();
    report.communities = comm.values().copied().collect::<std::collections::BTreeSet<_>>().len();
    g.replace_node_meta(&meta)?;

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ontology::Ontology;
    use crate::graph::store::Node;

    const ONT: &str = r#"
[entities.Symptom]
props = ["name"]
[entities.Cause]
props = ["name"]
[entities.Resolution]
props = ["name"]
[entities.Section]
props = ["name"]
[relations.CAUSED_BY]
from = ["Symptom"]
to = ["Cause"]
[relations.RESOLVED_BY]
from = ["Cause", "Symptom"]
to = ["Resolution"]
[relations.MENTIONS]
from = ["Symptom"]
to = ["Section"]
[validation]
strict = false
[reasoning]
closure = [["CAUSED_BY", "RESOLVED_BY", "RESOLVED_BY"]]
"#;

    // Same ontology but WITHOUT a [reasoning] section — drives the back-compat (no-op closure) test.
    const ONT_NO_REASONING: &str = r#"
[entities.Symptom]
props = ["name"]
[entities.Cause]
props = ["name"]
[entities.Resolution]
props = ["name"]
[entities.Section]
props = ["name"]
[relations.CAUSED_BY]
from = ["Symptom"]
to = ["Cause"]
[relations.RESOLVED_BY]
from = ["Cause", "Symptom"]
to = ["Resolution"]
[relations.MENTIONS]
from = ["Symptom"]
to = ["Section"]
[validation]
strict = false
"#;

    fn prov() -> Provenance {
        Provenance {
            source_path: "case.md".into(),
            range: None,
            file_sig: None,
            origin: "agent".into(),
            confidence: 0.9,
            created_at: 1,
        }
    }
    fn node(id: &str, ty: &str, label: &str) -> Node {
        Node { id: id.into(), node_type: ty.into(), label: label.into(), aliases: vec![], prov: prov() }
    }
    fn edge(f: &str, t: &str, to: &str) -> Edge {
        Edge { from: f.into(), edge_type: t.into(), to: to.into(), prov: prov() }
    }

    #[test]
    fn closure_and_meta_applied_merge_reported_not_applied_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(ONT).unwrap();
        // S -CAUSED_BY-> C -RESOLVED_BY-> R  ⇒ closure infers S -RESOLVED_BY-> R
        // S and S2 both MENTION the same section #1 → merge candidate (same type Symptom)
        let nodes = vec![
            node("sym:s", "Symptom", "Потеря связи"),
            node("sym:s2", "Symptom", "Потеря связи периодическая"),
            node("cau:c", "Cause", "Малый maxTsdr"),
            node("res:r", "Resolution", "Поднять maxTsdr"),
            node("sec:1", "Section", "doc#1"),
        ];
        let edges = vec![
            edge("sym:s", "CAUSED_BY", "cau:c"),
            edge("cau:c", "RESOLVED_BY", "res:r"),
            edge("sym:s", "MENTIONS", "sec:1"),
            edge("sym:s2", "MENTIONS", "sec:1"),
        ];
        g.upsert(&ont, &nodes, &edges).unwrap();

        // closure rule now comes from the ontology's [reasoning].closure, not a hardcode
        let rep = generalize(&g, &Opts::from_ontology(&ont, 100)).unwrap();
        assert_eq!(rep.inferred_edges, 1, "closure should infer S RESOLVED_BY R");
        assert!(rep.merge_candidates >= 1, "S and S2 share a chunk → merge candidate");
        assert_eq!(rep.merged_nodes, 0, "merge must NOT apply by default");

        // inferred edge present with the derived origin
        let out = g.outgoing("sym:s").unwrap();
        assert!(out.iter().any(|e| e.edge_type == "RESOLVED_BY" && e.to == "res:r"
            && e.prov.origin == "auto-generalized"));
        // node_meta populated (community id at least for the connected nodes)
        assert!(g.node_meta("sym:s").unwrap().is_some());
        // both symptoms still exist (no merge)
        assert!(g.get_node("sym:s2").unwrap().is_some());
    }

    #[test]
    fn generalize_back_compat_no_reasoning() {
        // An ontology with no [reasoning] section → empty closure rules → no inferred edges,
        // proving back-compat: deployments that never declared reasoning rules are unaffected.
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(ONT_NO_REASONING).unwrap();
        let nodes = vec![
            node("sym:s", "Symptom", "Потеря связи"),
            node("cau:c", "Cause", "Малый maxTsdr"),
            node("res:r", "Resolution", "Поднять maxTsdr"),
        ];
        let edges = vec![
            edge("sym:s", "CAUSED_BY", "cau:c"),
            edge("cau:c", "RESOLVED_BY", "res:r"),
        ];
        g.upsert(&ont, &nodes, &edges).unwrap();

        let rep = generalize(&g, &Opts::from_ontology(&ont, 100)).unwrap();
        assert_eq!(rep.inferred_edges, 0, "no [reasoning].closure → no inferred edges");
        // the direct S->R shortcut must NOT have been created
        assert!(!g.outgoing("sym:s").unwrap().iter().any(|e| e.edge_type == "RESOLVED_BY"));
    }

    #[test]
    fn merge_applied_collapses_dups_and_reattaches_edges() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(ONT).unwrap();
        let nodes = vec![
            node("sym:s", "Symptom", "Потеря связи"),
            node("sym:s2", "Symptom", "Потеря связи периодическая повторяющаяся"),
            node("res:r", "Resolution", "Поднять maxTsdr"),
            node("sec:1", "Section", "doc#1"),
        ];
        let edges = vec![
            edge("sym:s", "MENTIONS", "sec:1"),
            edge("sym:s2", "MENTIONS", "sec:1"),
            edge("sym:s2", "RESOLVED_BY", "res:r"), // edge on the dup must survive on canonical
        ];
        g.upsert(&ont, &nodes, &edges).unwrap();

        let mut opts = Opts::defaults(100);
        opts.apply_merges = true;
        let rep = generalize(&g, &opts).unwrap();
        assert_eq!(rep.merged_nodes, 1, "one dup symptom collapsed");
        // canonical = shorter label "Потеря связи" (sym:s); the dup is gone, its alias folded
        assert!(g.get_node("sym:s2").unwrap().is_none());
        let canon = g.get_node("sym:s").unwrap().unwrap();
        assert!(canon.aliases.iter().any(|a| a.contains("повторяющаяся")));
        // the dup's RESOLVED_BY edge now hangs off the canonical
        assert!(g.outgoing("sym:s").unwrap().iter().any(|e| e.edge_type == "RESOLVED_BY" && e.to == "res:r"));
    }
}
