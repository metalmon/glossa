//! `kbx build` — incremental delta detection (Task 11).
//!
//! Compares the indexed corpus against the reasoning graph already built over it, so a rebuild
//! only touches what actually changed: docs indexed but not yet extracted (`new`), docs whose
//! grounded reasoning nodes carry a stale `file_sig` (`changed`), and docs the graph still
//! references that vanished from the index (`gone`). Reuses glossa's own staleness detection
//! (`graph::generalize::hygiene::stale_nodes`) rather than recomputing signatures by hand, and
//! peels a node's grounding doc off its first-or-any `MENTIONS` edge the same way
//! `candidates::grounding_doc` / `chunks::chunk_text` do.

use anyhow::{Context, Result};
use glossa::graph::store::GraphStore;
use glossa::graph::{MENTIONS, STRUCTURAL_NODES};
use glossa::index::manifest::FileSig;
use glossa::index::store::DocIndex;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// New / changed / gone doc paths, relative to the corpus root — the unit `kbx build`'s
/// incremental mode re-runs extraction/candidates/judge over.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Delta {
    /// Indexed documents that own no non-structural (reasoning) node yet.
    pub new: Vec<String>,
    /// Documents whose current `file_sig` differs from the one stored on their grounded
    /// reasoning nodes.
    pub changed: Vec<String>,
    /// Documents referenced by a reasoning node's `MENTIONS` but absent from the current index.
    pub gone: Vec<String>,
}

/// Doc paths of every `MENTIONS` edge on `node_id` — the part before the `#ord` suffix, peeled
/// the same way `candidates::grounding_doc`/`chunks::chunk_text` do. Empty for an ungrounded
/// node. A node may carry more than one live `MENTIONS` (a cross-doc judged pair), so this
/// returns all of them rather than just the first.
fn grounding_docs(g: &GraphStore, node_id: &str) -> Result<Vec<String>> {
    let edges = g.outgoing(node_id)?;
    Ok(edges
        .iter()
        .filter(|e| e.edge_type == MENTIONS)
        .map(|e| {
            e.to.rsplit_once('#')
                .map(|(path, _)| path.to_string())
                .unwrap_or_else(|| e.to.clone())
        })
        .collect())
}

/// Distinct doc paths currently present in the tantivy index — one entry per chunk `path` field.
fn indexed_doc_set(idx: &DocIndex) -> Result<HashSet<String>> {
    let mut out = HashSet::new();
    idx.iter_chunks(|path, _ord, _file_type, _body| {
        out.insert(path.to_string());
    })
    .context("scanning index for doc paths")?;
    Ok(out)
}

/// Compute the new/changed/gone delta between the current index at `root` and the reasoning
/// nodes already in `g`. Never mutates either store.
pub fn compute_delta(root: &Path, idx: &DocIndex, g: &GraphStore) -> Result<Delta> {
    let indexed = indexed_doc_set(idx)?;
    let nodes = g.all_nodes().context("listing graph nodes")?;

    // Every doc a reasoning node is grounded to (owns it), and every doc any reasoning node's
    // MENTIONS references at all (for `gone`, which must catch a doc the graph still points at
    // even if it's the ONLY thing keeping that node's grounding doc referenced).
    let mut owned: HashSet<String> = HashSet::new();
    let mut referenced: HashSet<String> = HashSet::new();
    // Staleness input per glossa's own hygiene helper: (node id, source doc, stored file_sig).
    let mut stale_input: Vec<(String, String, Option<FileSig>)> = Vec::new();
    // node id -> its grounding doc(s), to map stale node ids back to doc paths.
    let mut node_docs: HashMap<String, Vec<String>> = HashMap::new();

    for n in &nodes {
        if STRUCTURAL_NODES.contains(&n.node_type.as_str()) {
            continue;
        }
        let docs = grounding_docs(g, &n.id)?;
        if docs.is_empty() {
            continue; // ungrounded — not this function's concern (doctor's `ungrounded` bucket)
        }
        for doc in &docs {
            owned.insert(doc.clone());
            referenced.insert(doc.clone());
            stale_input.push((n.id.clone(), doc.clone(), n.prov.file_sig));
        }
        node_docs.insert(n.id.clone(), docs);
    }

    // NEW: indexed docs that own no reasoning node at all.
    let mut new: Vec<String> = indexed
        .iter()
        .filter(|d| !owned.contains(*d))
        .cloned()
        .collect();
    new.sort();

    // CHANGED: distinct docs of nodes whose stored file_sig drifted from disk — reuse glossa's
    // own staleness detector rather than recomputing/comparing signatures by hand.
    let stale_ids = glossa::graph::generalize::hygiene::stale_nodes(root, &stale_input);
    let mut changed_set: HashSet<String> = HashSet::new();
    for id in &stale_ids {
        if let Some(docs) = node_docs.get(id) {
            changed_set.extend(docs.iter().cloned());
        }
    }
    let mut changed: Vec<String> = changed_set.into_iter().collect();
    changed.sort();

    // GONE: doc paths referenced by any reasoning node's MENTIONS but absent from the index now.
    let mut gone: Vec<String> = referenced
        .iter()
        .filter(|d| !indexed.contains(*d))
        .cloned()
        .collect();
    gone.sort();

    Ok(Delta { new, changed, gone })
}

/// Delete every non-structural (reasoning) node grounded ONLY to `doc` — every live `MENTIONS`
/// edge it carries targets `doc`, none targets a different doc. A node with a live `MENTIONS` to
/// another doc survives (it's still needed there); an ungrounded node is left alone (not this
/// function's concern). Returns the ids of the nodes actually removed — the caller (`run_build`)
/// needs them to invalidate any checkpoint mark (e.g. a `judge:{a}#{b}` pair) that referenced one
/// of them, since its cascade-deleted edges make the checkpoint the only stale record left.
pub fn drop_doc_nodes(g: &GraphStore, doc: &str) -> Result<Vec<String>> {
    let nodes = g.all_nodes().context("listing graph nodes")?;
    let mut removed = Vec::new();
    for n in &nodes {
        if STRUCTURAL_NODES.contains(&n.node_type.as_str()) {
            continue;
        }
        let docs = grounding_docs(g, &n.id)?;
        if docs.is_empty() || !docs.iter().all(|d| d == doc) {
            continue;
        }
        g.delete_node(&n.id)
            .with_context(|| format!("deleting node {} grounded only to {doc}", n.id))?;
        removed.push(n.id.clone());
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glossa::graph::store::{Edge, Node, Provenance};

    fn prov(source_path: &str, file_sig: Option<FileSig>) -> Provenance {
        prov_origin(source_path, file_sig, "agent")
    }

    fn prov_origin(source_path: &str, file_sig: Option<FileSig>, origin: &str) -> Provenance {
        Provenance {
            source_path: source_path.into(),
            range: None,
            file_sig,
            origin: origin.into(),
            confidence: 0.9,
            created_at: 1,
        }
    }

    fn put_fact(g: &GraphStore, id: &str, mentions: &[&str], file_sig: Option<FileSig>) {
        put_fact_origin(g, id, mentions, file_sig, "agent")
    }

    fn put_fact_origin(
        g: &GraphStore,
        id: &str,
        mentions: &[&str],
        file_sig: Option<FileSig>,
        origin: &str,
    ) {
        let source_path = mentions
            .first()
            .and_then(|m| m.rsplit_once('#'))
            .map(|(p, _)| p.to_string())
            .unwrap_or_default();
        g.put_node(&Node {
            id: id.into(),
            node_type: "Fact".into(),
            label: id.into(),
            aliases: vec![id.into()],
            prov: prov_origin(&source_path, file_sig, origin),
        })
        .unwrap();
        for m in mentions {
            let doc = m.rsplit_once('#').map(|(p, _)| p).unwrap_or(m);
            g.put_edge(&Edge {
                from: id.into(),
                to: (*m).into(),
                edge_type: MENTIONS.into(),
                prov: prov_origin(doc, file_sig, origin),
            })
            .unwrap();
        }
    }

    /// A corpus with three docs (a, b, c) indexed and structural nodes built, ready for reasoning
    /// nodes to be layered on top by each test.
    fn synthetic_corpus() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        std::fs::write(root.join("a.md"), "# A\n\nAlpha body.\n").unwrap();
        std::fs::write(root.join("b.md"), "# B\n\nBeta body.\n").unwrap();
        std::fs::write(root.join("c.md"), "# C\n\nGamma body.\n").unwrap();
        glossa::index::store::index_dir(&root, true).unwrap();
        (dir, root)
    }

    #[test]
    fn unchanged_new_changed_gone_delta() {
        let (_dir, root) = synthetic_corpus();
        let g = GraphStore::open(&root).unwrap();
        let idx = DocIndex::open_or_create(&root).unwrap();

        // a.md: owned by a reasoning node whose stored sig matches disk -> unchanged.
        let sig_a = glossa::index::store::file_sig(&root.join("a.md")).unwrap();
        put_fact(&g, "f_a", &["a.md#1"], Some(sig_a));

        // b.md: owned by a reasoning node whose stored sig does NOT match disk -> changed.
        let stale_sig = FileSig {
            mtime_secs: 1,
            size: 999_999,
        };
        put_fact(&g, "f_b", &["b.md#1"], Some(stale_sig));

        // c.md: indexed, but no reasoning node grounds it -> new.

        // doc_gone.md: referenced by a node's MENTIONS but never written/indexed -> gone.
        put_fact(&g, "f_gone", &["doc_gone.md#1"], None);

        let delta = compute_delta(&root, &idx, &g).unwrap();
        assert_eq!(delta.new, vec!["c.md".to_string()]);
        assert_eq!(delta.changed, vec!["b.md".to_string()]);
        assert_eq!(delta.gone, vec!["doc_gone.md".to_string()]);
    }

    #[test]
    fn drop_doc_nodes_removes_single_doc_node_keeps_multi_doc_node() {
        let (_dir, root) = synthetic_corpus();
        let g = GraphStore::open(&root).unwrap();

        // Grounded ONLY to a.md -> removed when a.md is dropped.
        put_fact(&g, "f_only_a", &["a.md#1"], None);
        // Grounded to BOTH a.md and b.md (a cross-doc judged node) -> survives.
        put_fact(&g, "f_multi", &["a.md#1", "b.md#1"], None);

        let removed = drop_doc_nodes(&g, "a.md").unwrap();
        assert_eq!(removed, vec!["f_only_a".to_string()]);
        assert!(g.get_node("f_only_a").unwrap().is_none());
        assert!(g.get_node("f_multi").unwrap().is_some());
    }

    /// Confirms `drop_doc_nodes` — the doc-CHANGE re-extract drop path — is origin-AGNOSTIC: it
    /// selects purely on grounding (non-structural, MENTIONS-only-to-`doc`), with no origin
    /// filter, so a `distil`-origin node (kbx distil densification writer) grounded only to the
    /// changed doc is dropped exactly like an `agent`-origin one, with no predicate to widen.
    #[test]
    fn drop_doc_nodes_drops_distil_origin_grounded_only_to_changed_doc() {
        let (_dir, root) = synthetic_corpus();
        let g = GraphStore::open(&root).unwrap();

        // distil-origin, grounded ONLY to a.md -> removed when a.md is dropped.
        put_fact_origin(&g, "f_distil_only_a", &["a.md#1"], None, "distil");
        // distil-origin, grounded to BOTH a.md and b.md -> survives.
        put_fact_origin(&g, "f_distil_multi", &["a.md#1", "b.md#1"], None, "distil");

        let removed = drop_doc_nodes(&g, "a.md").unwrap();
        assert_eq!(removed, vec!["f_distil_only_a".to_string()]);
        assert!(g.get_node("f_distil_only_a").unwrap().is_none());
        assert!(g.get_node("f_distil_multi").unwrap().is_some());
    }
}
