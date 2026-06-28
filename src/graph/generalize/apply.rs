//! Integration pass — run the generalization techniques over a live `GraphStore` and persist
//! the results. Wired (later) into a `kb graph generalize` step after enrich/reindex.
//!
//! - Derived edges (closure #8, SIMILAR from link-prediction #4) are written with
//!   `origin = "auto-generalized"` and cleared/regenerated on each run (NOT the document `auto-*`
//!   layer, so reindex's structural rebuild and this pass don't fight).
//! - Communities (#6) + centrality (#7) go to the `node_meta` side table.
//! - Near-dup MERGE (#3 shared-evidence, Tier-1) is always COMPUTED, but only APPLIED when
//!   `opts.apply_merges` is set — it mutates/deletes agent nodes, so the safe default is report-only.

use super::{centrality, closure, community, hygiene, linkpred, merge, similarity, Triple};
use crate::graph::ontology::{Ontology, Spine};
use crate::graph::store::{Edge, GraphStore, NodeMeta, Provenance};
use std::collections::{HashMap, HashSet};

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
    /// Transitive-closure composition rules `(a, b, result)`, from `[reasoning].closure`.
    pub closure_rules: Vec<(String, String, String)>,
    /// Structural (never-reasoning) types excluded from merge/similar, from `[reasoning].structural`.
    pub structural: Vec<String>,
    /// Hygiene: delete degenerate reasoning chains (report-only when false). CLI `--prune-incomplete`.
    pub prune_incomplete: bool,
    /// Hygiene: the ontology reasoning spines; empty → hygiene is a no-op. From `[reasoning].spines`.
    pub spines: Vec<Spine>,
    /// Hygiene: entity types that are endpoints of spine relations (to tell doomed from auxiliary).
    pub spine_types: HashSet<String>,
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
            closure_rules: vec![],
            structural: ["Document", "Section", "Term", "Topic"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            prune_incomplete: false,
            spines: vec![],
            spine_types: HashSet::new(),
            now,
        }
    }

    /// Source the domain rules (spine, closure, mentions anchor, structural types) from the
    /// ontology's `[reasoning]` section; other tunables keep their defaults. `prune_incomplete`
    /// stays false — callers opt in (CLI `--prune-incomplete`).
    pub fn from_ontology(ont: &Ontology, now: u64) -> Self {
        Opts {
            closure_rules: ont.closure_rules(),
            structural: ont.structural(),
            spines: ont.spines(),
            spine_types: ont.spine_types(),
            ..Opts::defaults(now)
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub prune_candidates: usize,
    pub pruned_nodes: usize,
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

    // (Hygiene) cull degenerate reasoning chains BEFORE anything is derived/merged over them.
    // Report-only unless opts.prune_incomplete; empty spine → no-op. Deleting here means the
    // merge/closure/meta passes below see the cleaned graph (they reload from `g`).
    {
        let id_types: Vec<(String, String)> =
            g.all_nodes()?.into_iter().map(|n| (n.id, n.node_type)).collect();
        let edges: Vec<Triple> =
            g.all_edges()?.into_iter().map(|e| (e.from, e.edge_type, e.to)).collect();
        let structural: HashSet<String> = opts.structural.iter().cloned().collect();
        let doomed = hygiene::incomplete_nodes(
            &id_types,
            &edges,
            &opts.spines,
            &opts.spine_types,
            &structural,
        );
        report.prune_candidates = doomed.len();
        if opts.prune_incomplete && !doomed.is_empty() {
            report.pruned_nodes = g.delete_nodes(&doomed)?;
        }
    }

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
            .filter(|e| e.edge_type == crate::graph::MENTIONS && reasoning(&e.from))
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

    // (#6 + #7) communities + centrality over the REASONING subgraph, on edges that INCLUDE the
    // derived closure + SIMILAR just written above (re-fetch — the `edges` snapshot predates them).
    // Over the FULL graph this made ~one community per Document (each doc + its sections is a
    // connected component via CONTAINS), and a much-MENTIONed section looked like a centrality hub.
    // Scoping to non-structural nodes + the edges among them keeps the meta about reasoning; the
    // shared-evidence relationship survives as the SIMILAR edges, which are reasoning↔reasoning.
    let structural: HashSet<String> = opts.structural.iter().cloned().collect();
    let type_of: HashMap<&str, &str> =
        nodes.iter().map(|n| (n.id.as_str(), n.node_type.as_str())).collect();
    let is_reasoning = |id: &str| type_of.get(id).is_some_and(|t| !structural.contains(*t));
    let r_ids: Vec<String> = node_ids.iter().filter(|id| is_reasoning(id)).cloned().collect();
    let r_edges: Vec<Triple> = g
        .all_edges()?
        .into_iter()
        .map(|e| (e.from, e.edge_type, e.to))
        .filter(|(f, _t, to)| is_reasoning(f) && is_reasoning(to))
        .collect();
    let comm = community::connected_components(&r_ids, &r_edges);
    let deg = centrality::degree(&r_ids, &r_edges);
    let pr = centrality::pagerank(&r_ids, &r_edges, 0.85, 30);
    let meta: Vec<(String, NodeMeta)> = r_ids
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
spines = [{ anchor = "Symptom", relations = ["CAUSED_BY", "RESOLVED_BY"] }]
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
    fn generalize_prune_reports_only_by_default() {
        // a degenerate Symptom→Resolution (no Cause): both are doomed, but the default is
        // report-only — nothing is deleted.
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(ONT).unwrap();
        let nodes = vec![node("sym:s", "Symptom", "S"), node("res:r", "Resolution", "R")];
        let edges = vec![edge("sym:s", "RESOLVED_BY", "res:r")];
        g.upsert(&ont, &nodes, &edges).unwrap();

        let rep = generalize(&g, &Opts::from_ontology(&ont, 100)).unwrap();
        assert_eq!(rep.prune_candidates, 2, "sym:s and res:r are both degenerate");
        assert_eq!(rep.pruned_nodes, 0, "report-only by default");
        assert!(g.get_node("sym:s").unwrap().is_some(), "nothing deleted without the flag");
        assert!(g.get_node("res:r").unwrap().is_some());
    }

    #[test]
    fn generalize_prune_applies_with_flag() {
        // complete S1→C1→R1 survives; degenerate S2→R2 (no Cause) is culled under the flag.
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(ONT).unwrap();
        let nodes = vec![
            node("sym:s1", "Symptom", "S1"),
            node("cau:c1", "Cause", "C1"),
            node("res:r1", "Resolution", "R1"),
            node("sym:s2", "Symptom", "S2"),
            node("res:r2", "Resolution", "R2"),
        ];
        let edges = vec![
            edge("sym:s1", "CAUSED_BY", "cau:c1"),
            edge("cau:c1", "RESOLVED_BY", "res:r1"),
            edge("sym:s2", "RESOLVED_BY", "res:r2"),
        ];
        g.upsert(&ont, &nodes, &edges).unwrap();

        let mut opts = Opts::from_ontology(&ont, 100);
        opts.prune_incomplete = true;
        let rep = generalize(&g, &opts).unwrap();

        assert_eq!(rep.prune_candidates, 2, "sym:s2 + res:r2 are degenerate");
        assert_eq!(rep.pruned_nodes, 2, "and they are deleted under the flag");
        // the complete chain survived
        assert!(g.get_node("sym:s1").unwrap().is_some());
        assert!(g.get_node("cau:c1").unwrap().is_some());
        assert!(g.get_node("res:r1").unwrap().is_some());
        // the degenerate pair is gone
        assert!(g.get_node("sym:s2").unwrap().is_none());
        assert!(g.get_node("res:r2").unwrap().is_none());
        // closure still inferred on the cleaned graph: S1 -RESOLVED_BY-> R1
        assert_eq!(rep.inferred_edges, 1);
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
