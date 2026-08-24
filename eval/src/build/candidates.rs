//! `kbx build` stage 2 — mechanical cross-document candidate GROUPING (Task 6, entity-group
//! redesign).
//!
//! Purely mechanical: no model calls, no judgment. Groups non-structural graph nodes by shared
//! NORMALIZED ALIAS ONLY (per spec — alias-only is validated as sufficient on the pilot
//! reference; `apply_upsert` already globally dedups nodes by normalized (label, node_type) at
//! write time, so folding the label into this key would only additionally catch accidental
//! cross-node-TYPE label collisions, widening the candidate set beyond the validated scope) and
//! keeps only entities whose nodes are grounded (via their first `MENTIONS` edge) in ≥2 DISTINCT
//! documents — a same-doc match is not a cross-doc candidate.
//!
//! Unlike the retired pairwise design (which exploded each such entity into C(n,2) cross-doc node
//! PAIRS for a separate model call each), this stage emits ONE GROUP per cross-doc entity holding
//! every one of its member node ids. The judge (stage 3, `judge::judge_group`) then makes a single
//! model call per group, passing it every member fact at once and letting the model return an
//! arbitrary set of directed links among them — no frequency cap is needed here to bound
//! combinatorial cost, since there's no pairwise explosion left to bound; the model itself is
//! trusted to resolve a generic, many-document entity to "no links" (see `bridge.md`). A group's
//! own SIZE (fact count fed to one model call) is instead bounded downstream by
//! `--bridge-max-facts`, a prompt-fit guard, not a recall guard.

use anyhow::Result;
use glossa::graph::store::{normalize_label, GraphStore};
use glossa::graph::{MENTIONS, STRUCTURAL_NODES};
use std::collections::BTreeMap;

/// A mechanically-proposed cross-document candidate GROUP: every node id sharing a normalized
/// alias/label, grounded across ≥2 distinct documents. `entity` is the normalized shared key;
/// `members` is deduped and in deterministic (sorted) order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateGroup {
    pub entity: String,
    pub members: Vec<String>,
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
/// grounding documents, and emit one `CandidateGroup` per such entity holding every member node id
/// (deduped, deterministic order). No frequency cap: an entity spanning many documents still
/// yields exactly one group and one model call — the group judge decides what (if anything) links,
/// so there is no C(n,2) explosion left to bound here.
pub fn candidate_groups(g: &GraphStore) -> Result<Vec<CandidateGroup>> {
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

    let mut out = Vec::new();
    for (entity, mut members) in by_key {
        members.sort();
        members.dedup();
        let distinct_docs: std::collections::HashSet<&str> =
            members.iter().map(|(_, d)| d.as_str()).collect();
        if distinct_docs.len() < 2 {
            continue;
        }
        out.push(CandidateGroup {
            entity,
            members: members.into_iter().map(|(id, _)| id).collect(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glossa::graph::store::{Edge, Node, Provenance};
    use std::collections::HashSet;

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
    fn cross_doc_alias_yields_one_group_with_all_members() {
        let g = mem_graph();
        put_fact(&g, "f1", "n1", ["X"], "a.md#1");
        put_fact(&g, "f2", "n2", ["X"], "b.md#1");
        put_fact(&g, "f3", "n3", ["X"], "c.md#1");
        put_fact(&g, "f4", "n4", ["Y"], "b.md#1"); // unrelated single-doc-only alias so far
        let groups = candidate_groups(&g).unwrap();
        assert_eq!(groups.len(), 1, "only 'X' spans >=2 docs");
        assert_eq!(groups[0].entity, "x");
        let members: HashSet<&str> = groups[0].members.iter().map(|s| s.as_str()).collect();
        assert_eq!(members, ["f1", "f2", "f3"].into_iter().collect());
    }

    #[test]
    fn same_doc_only_alias_yields_no_group() {
        // Two nodes sharing an alias but grounded in the SAME doc must not group.
        let g = mem_graph();
        put_fact(&g, "f1", "n1", ["X"], "a.md#1");
        put_fact(&g, "f2", "n2", ["X"], "a.md#2");
        let groups = candidate_groups(&g).unwrap();
        assert!(groups.is_empty());
    }

    #[test]
    fn groups_and_members_are_in_deterministic_order() {
        let g = mem_graph();
        // Insert out of alphabetical order to prove the BTreeMap/sort gives stable output.
        put_fact(&g, "f_z", "z", ["Z"], "b.md#1");
        put_fact(&g, "f_z2", "z", ["Z"], "a.md#1");
        put_fact(&g, "f_a", "a", ["A"], "b.md#1");
        put_fact(&g, "f_a2", "a", ["A"], "a.md#1");
        let groups = candidate_groups(&g).unwrap();
        let entities: Vec<&str> = groups.iter().map(|gr| gr.entity.as_str()).collect();
        assert_eq!(entities, vec!["a", "z"], "groups sorted by entity key");
        let z_group = groups.iter().find(|gr| gr.entity == "z").unwrap();
        assert_eq!(z_group.members, vec!["f_z".to_string(), "f_z2".to_string()]);
    }

    #[test]
    fn shared_label_alone_does_not_group_without_a_shared_alias() {
        // Two cross-doc nodes with the SAME label but DISJOINT aliases must not group — grouping
        // is alias-only, per spec (label is not folded into the key).
        let g = mem_graph();
        put_fact(&g, "f1", "shared", ["X"], "a.md#1");
        put_fact(&g, "f2", "shared", ["Y"], "b.md#1");
        let groups = candidate_groups(&g).unwrap();
        assert!(groups.is_empty());
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
        let groups = candidate_groups(&g).unwrap();
        assert_eq!(groups.len(), 1);
        assert!(!groups[0].members.iter().any(|m| m == "doc:c"));
    }

    #[test]
    fn no_frequency_cap_a_many_doc_entity_still_yields_one_group() {
        // The retired pairwise design pruned aliases spanning "too many" docs to bound C(n,2)
        // noise. The group design has no such cap: a many-doc entity still yields exactly one
        // group (one model call) — the judge, not a mechanical cap, decides what links.
        let g = mem_graph();
        for (i, doc) in ["a.md", "b.md", "c.md", "d.md", "e.md", "f.md"]
            .iter()
            .enumerate()
        {
            put_fact(&g, &format!("f{i}"), "n", ["X"], &format!("{doc}#1"));
        }
        let groups = candidate_groups(&g).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].members.len(), 6);
    }
}
