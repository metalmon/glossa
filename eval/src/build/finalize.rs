//! `kbx build` stage 4 — finalize (Task 9): after extraction/merge/judging land facts and
//! cross-doc edges in the graph, recompute the DERIVED layer (SIMILAR links + communities),
//! force a fresh `.glossa/nodes` BM25 rebuild so `resolve`/`glossary` see the final node set, run
//! `graph doctor` (hygiene: ungrounded/stale/incomplete), and return a human summary combining
//! node-type counts with the doctor report. Delegates to `glossa::graph::ops` so this stays a
//! thin driver, not a reimplementation.

use anyhow::{Context, Result};
use glossa::graph::ontology::Ontology;
use glossa::graph::ops::{graph_doctor, graph_generalize};
use glossa::graph::store::GraphStore;
use std::collections::BTreeMap;
use std::path::Path;

/// Run the finalize stage over the corpus at `root`: generalize, force a fresh node-search index,
/// doctor, and report. Returns a summary string — never mutates in a destructive way (generalize
/// is non-destructive per its own docs; doctor is report-only).
pub fn finalize(root: &Path) -> Result<String> {
    let ont = Ontology::load_or_default(root);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // 1. Generalize: recompute the derived layer (SIMILAR + communities) from what's currently
    // stored. Non-destructive.
    let g = GraphStore::open(root).context("open graph store for generalize")?;
    let generalize_summary = graph_generalize(&g, &ont, now);
    drop(g);

    // 2. Force a fresh `.glossa/nodes` BM25 rebuild. The index is normally rebuilt lazily (count/
    // content-signature gated, see `GraphStore::ensure_node_index_fresh`), but finalize wants a
    // guaranteed-fresh index reflecting everything the build stages just wrote. Wipe the on-disk
    // index (ignore NotFound) so the reopened `NodeIndex` starts empty — that guarantees the next
    // `resolve` call rebuilds it from `graph.sqlite` rather than possibly finding it already
    // "fresh" by count+signature.
    let nodes_dir = root.join(".glossa").join("nodes");
    match std::fs::remove_dir_all(&nodes_dir) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e).context("remove .glossa/nodes for forced rebuild"),
    }
    let g = GraphStore::open(root).context("reopen graph store after nodes-index wipe")?;
    // `resolve` rebuilds the node index before falling back to BM25 search whenever the query has
    // no exact normalized-label match — a query built to never hit a real label guarantees that
    // path runs and the rebuild happens now, not on the first real caller.
    let _ = g.resolve("__kbx_finalize_forced_nodes_index_rebuild__");

    // 3. Doctor: report-only hygiene pass (ungrounded/stale/incomplete).
    let doctor_summary = graph_doctor(&g, &ont, root);

    // Node-type counts for the summary.
    let nodes = g.all_nodes().context("list nodes for summary")?;
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for n in &nodes {
        *counts.entry(n.node_type.clone()).or_default() += 1;
    }

    let mut summary = String::new();
    summary.push_str("nodes by type:\n");
    for (node_type, count) in &counts {
        summary.push_str(&format!("  {node_type}: {count}\n"));
    }
    summary.push_str(&generalize_summary);
    summary.push('\n');
    summary.push_str(&doctor_summary);
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glossa::graph::store::{Edge, Node, Provenance};
    use glossa::graph::MENTIONS;

    #[test]
    fn finalize_summary_has_node_types_and_ungrounded() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("d.md"), "# Title\n\nGrounded chunk body.\n").unwrap();
        glossa::index::store::index_dir(root, true).unwrap();
        let g = GraphStore::open(root).unwrap();

        let prov = Provenance {
            source_path: "d.md".into(),
            range: None,
            file_sig: None,
            origin: "agent".into(),
            confidence: 0.9,
            created_at: 1,
        };
        g.put_node(&Node {
            id: "f1".into(),
            node_type: "Fact".into(),
            label: "n1".into(),
            aliases: vec!["n1".into()],
            prov: prov.clone(),
        })
        .unwrap();
        g.put_edge(&Edge {
            from: "f1".into(),
            to: "d.md#1".into(),
            edge_type: MENTIONS.into(),
            prov: prov.clone(),
        })
        .unwrap();
        g.put_node(&Node {
            id: "f2".into(),
            node_type: "Fact".into(),
            label: "n2".into(),
            aliases: vec!["n2".into()],
            prov,
        })
        .unwrap();
        drop(g);

        let summary = finalize(root).unwrap();
        assert!(
            summary.contains("Fact"),
            "summary should list the Fact node type: {summary}"
        );
        assert!(
            summary.contains("ungrounded"),
            "summary should include the doctor ungrounded count: {summary}"
        );
    }
}
