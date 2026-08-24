//! `kbx build` stage 2 — mechanical cross-document candidate grouping (Task 6).
//!
//! Purely mechanical: no model calls, no judgment. Groups non-structural graph nodes by shared
//! NORMALIZED ALIAS ONLY (per spec — alias-only is validated as sufficient on the pilot
//! reference; `apply_upsert` already globally dedups nodes by normalized (label, node_type) at
//! write time, so folding the label into this key would only additionally catch accidental
//! cross-node-TYPE label collisions, widening the candidate set beyond the validated scope) and
//! keeps only entities whose nodes are grounded (via their first `MENTIONS` edge) in ≥2 DISTINCT
//! documents — a same-doc match is not a cross-doc candidate. Each cross-doc node pair is
//! emitted once (deduped by unordered id pair, even when two nodes share more than one alias).
//! Judging which of these candidates are real cross-doc reasoning links is a LATER
//! (model-judged) stage; this one only proposes.

use anyhow::Result;
use glossa::graph::store::{normalize_label, GraphStore};
use glossa::graph::{MENTIONS, STRUCTURAL_NODES};
use std::collections::{BTreeMap, HashSet};

/// A mechanically-proposed cross-document candidate: two node ids that share a normalized
/// alias/label, each grounded in a different document. `entity` is the normalized shared key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidatePair {
    pub a: String,
    pub b: String,
    pub entity: String,
}

/// The document a node is grounded in: the path part (before `#`) of its first `MENTIONS` edge
/// target, in the order `GraphStore::outgoing` returns edges. Any live `MENTIONS` edge gives the
/// same doc for a given node in practice, so the "first" pick is just a cheap way to grab one —
/// it doesn't matter which. `None` when the node has no `MENTIONS` edge — an ungrounded node
/// can't anchor a cross-doc candidate.
fn grounding_doc(g: &GraphStore, node_id: &str) -> Result<Option<String>> {
    let edges = g.outgoing(node_id)?;
    Ok(edges
        .iter()
        .find(|e| e.edge_type == MENTIONS)
        .map(|e| e.to.split('#').next().unwrap_or(&e.to).to_string()))
}

/// Mechanically group non-structural nodes by shared normalized ALIAS (label is intentionally
/// excluded from the key — see module docs), keeping only entities whose nodes span ≥2 distinct
/// grounding documents, and emit each cross-doc node pair once (deduped by unordered id pair,
/// even when two nodes share more than one alias).
/// `max_alias_docs`: an alias grounded across MORE than this many distinct documents is too generic
/// to anchor a real cross-doc reasoning link (a common term like a product family or a unit name
/// recurs everywhere and would emit C(n,2) noise pairs the judge then burns a model call on each).
/// Such aliases are skipped. `0` disables the cap (every cross-doc alias anchors pairs — the old
/// behavior). A specific entity that genuinely bridges a few documents stays under the cap.
pub fn candidate_pairs(g: &GraphStore, max_alias_docs: usize) -> Result<Vec<CandidatePair>> {
    // key -> [(node_id, doc)], in a BTreeMap so iteration (and emission) order is deterministic
    // regardless of sqlite row order.
    let mut by_key: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();

    for node in g.all_nodes()? {
        if STRUCTURAL_NODES.contains(&node.node_type.as_str()) {
            continue;
        }
        let Some(doc) = grounding_doc(g, &node.id)? else {
            continue;
        };
        let mut keys: Vec<String> = node.aliases.iter().map(|a| normalize_label(a)).collect();
        keys.sort();
        keys.dedup();
        for key in keys {
            by_key
                .entry(key)
                .or_default()
                .push((node.id.clone(), doc.clone()));
        }
    }

    let mut seen_pairs: HashSet<(String, String)> = HashSet::new();
    let mut out = Vec::new();
    for (key, mut members) in by_key {
        members.sort();
        members.dedup();
        let distinct_docs: HashSet<&str> = members.iter().map(|(_, d)| d.as_str()).collect();
        if distinct_docs.len() < 2 {
            continue;
        }
        // Frequency prune: skip aliases spread across too many documents — generic terms whose
        // cross-doc pairs are almost all noise. `max_alias_docs == 0` disables the cap.
        if max_alias_docs != 0 && distinct_docs.len() > max_alias_docs {
            continue;
        }
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                let (id_a, doc_a) = &members[i];
                let (id_b, doc_b) = &members[j];
                if doc_a == doc_b {
                    continue;
                }
                let pair = if id_a < id_b {
                    (id_a.clone(), id_b.clone())
                } else {
                    (id_b.clone(), id_a.clone())
                };
                if seen_pairs.insert(pair.clone()) {
                    out.push(CandidatePair {
                        a: pair.0,
                        b: pair.1,
                        entity: key.clone(),
                    });
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glossa::graph::store::{Edge, Node, Provenance};

    /// A fresh in-memory-equivalent `GraphStore` over a leaked temp dir (test-only; the OS temp
    /// dir is never cleaned up, mirroring the throwaway dirs other graph unit tests use).
    fn mem_graph() -> GraphStore {
        let dir = tempfile::tempdir().unwrap().keep();
        GraphStore::open(&dir).unwrap()
    }

    /// Insert a non-structural (`Fact`) node with the given aliases, plus a `MENTIONS` edge
    /// grounding it at `mention` (a `path#pos` reference — the doc is the part before `#`).
    fn put_fact(
        g: &GraphStore,
        id: &str,
        label: &str,
        aliases: impl IntoIterator<Item = &'static str>,
        mention: &str,
    ) {
        let doc = mention.split('#').next().unwrap_or(mention);
        let prov = Provenance {
            source_path: doc.to_string(),
            range: None,
            file_sig: None,
            origin: "agent".into(),
            confidence: 0.9,
            created_at: 1,
        };
        g.put_node(&Node {
            id: id.into(),
            node_type: "Fact".into(),
            label: label.into(),
            aliases: aliases.into_iter().map(|a| a.to_string()).collect(),
            prov: prov.clone(),
        })
        .unwrap();
        g.put_edge(&Edge {
            from: id.into(),
            to: mention.into(),
            edge_type: MENTIONS.into(),
            prov,
        })
        .unwrap();
    }

    #[test]
    fn candidates_are_cross_doc_shared_alias() {
        let g = mem_graph();
        put_fact(&g, "f1", "n1", ["X"], "a.md#1");
        put_fact(&g, "f2", "n2", ["X"], "b.md#1");
        put_fact(&g, "f3", "n3", ["Y"], "b.md#1");
        let pairs = candidate_pairs(&g, 0).unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].entity, "x");
        let ids: HashSet<&str> = [pairs[0].a.as_str(), pairs[0].b.as_str()].into_iter().collect();
        assert_eq!(ids, ["f1", "f2"].into_iter().collect());
    }

    #[test]
    fn max_alias_docs_prunes_over_frequent_aliases() {
        // Alias "X" spans 3 docs (generic); alias "Y" spans exactly 2 (specific).
        let g = mem_graph();
        put_fact(&g, "f1", "n1", ["X", "Y"], "a.md#1");
        put_fact(&g, "f2", "n2", ["X", "Y"], "b.md#1");
        put_fact(&g, "f3", "n3", ["X"], "c.md#1");
        // Unlimited: X emits its cross-doc pairs, Y emits its pair.
        assert!(candidate_pairs(&g, 0).unwrap().iter().any(|p| p.entity == "x"));
        // Cap at 2 docs: X (3 docs) is pruned; Y (2 docs) survives.
        let capped = candidate_pairs(&g, 2).unwrap();
        assert!(capped.iter().all(|p| p.entity != "x"), "over-frequent alias must be pruned");
        assert!(capped.iter().any(|p| p.entity == "y"), "specific alias must survive");
    }

    #[test]
    fn same_doc_alias_match_is_not_a_candidate() {
        // Two nodes sharing an alias but grounded in the SAME doc must not pair.
        let g = mem_graph();
        put_fact(&g, "f1", "n1", ["X"], "a.md#1");
        put_fact(&g, "f2", "n2", ["X"], "a.md#2");
        let pairs = candidate_pairs(&g, 0).unwrap();
        assert!(pairs.is_empty());
    }

    #[test]
    fn shared_label_alone_does_not_group_without_a_shared_alias() {
        // Two cross-doc nodes with the SAME label but DISJOINT aliases must not pair — grouping
        // is alias-only, per spec (label is not folded into the key).
        let g = mem_graph();
        put_fact(&g, "f1", "shared", ["X"], "a.md#1");
        put_fact(&g, "f2", "shared", ["Y"], "b.md#1");
        let pairs = candidate_pairs(&g, 0).unwrap();
        assert!(pairs.is_empty());
    }

    #[test]
    fn structural_nodes_are_excluded() {
        // A structural Document node sharing a label with a cross-doc Fact must not seed a
        // candidate — structural nodes carry no reasoning content.
        let g = mem_graph();
        put_fact(&g, "f1", "shared", ["shared"], "a.md#1");
        put_fact(&g, "f2", "shared", ["shared"], "b.md#1");
        g.put_node(&Node {
            id: "doc:c".into(),
            node_type: "Document".into(),
            label: "shared".into(),
            aliases: vec![],
            prov: Provenance {
                source_path: "c.md".into(),
                range: None,
                file_sig: None,
                origin: "agent".into(),
                confidence: 0.9,
                created_at: 1,
            },
        })
        .unwrap();
        let pairs = candidate_pairs(&g, 0).unwrap();
        assert_eq!(pairs.len(), 1);
        assert!(!pairs.iter().any(|p| p.a == "doc:c" || p.b == "doc:c"));
    }
}
