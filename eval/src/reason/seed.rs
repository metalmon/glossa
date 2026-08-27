//! Phase-2 (`kbx reason` seed-from-grounded) engine: read a grounded terminal's FULL source,
//! backward-synthesize the query-side reasoning layer (fan-out), and write it via the canonical
//! write triad. Sibling to `chain.rs` (gold-anchored, migrating to `kbx train`).

use crate::backend::glossa_tools;
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;
use glossa::trace::TraceLog;

/// The grounded source text for EVERY outgoing `MENTIONS` edge of `seed_id`, each target chunk
/// read and concatenated in sorted-ref order (deduped), separated by blank lines. Empty string if
/// the seed has no groundings or none resolve to readable chunks. Unlike
/// `distil::gen::seed_source_text` (first edge only), phase-2 needs the whole resolution — it may
/// span several chunks and the query-side (problem, aliases, task) must be inferred from all of them.
pub(crate) fn seed_source_text_all(
    root: &std::path::Path,
    idx: &DocIndex,
    g: &GraphStore,
    seed_id: &str,
) -> String {
    let mut targets: Vec<String> = g
        .outgoing(seed_id)
        .unwrap_or_default()
        .into_iter()
        .filter(|e| e.edge_type == glossa::graph::MENTIONS)
        .map(|e| e.to)
        .collect();
    targets.sort();
    targets.dedup();

    let trace = TraceLog::disabled();
    let mut out = String::new();
    for target in targets {
        let (path, n) = match target.rsplit_once('#') {
            Some((p, n)) => (p.to_string(), n.parse::<u64>().unwrap_or(0)),
            None => (target, 0),
        };
        let (text, _images) = glossa_tools::run_read(root, idx, None, &path, n, false, &trace);
        if !text.is_empty() {
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            out.push_str(&text);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use glossa::graph::store::{Edge, GraphStore, Node, Provenance};
    use glossa::index::store::DocIndex;

    fn prov() -> Provenance {
        Provenance { source_path: "d.md".into(), range: None, file_sig: None,
            origin: "test".into(), confidence: 0.9, created_at: 1 }
    }

    #[test]
    fn seed_source_text_all_concatenates_every_grounding_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            glossa::model::Chunk { doc_path: "d.md".into(), location: "S1".into(),
                file_type: "md".into(), text: "FIRST chunk body".into() },
            glossa::model::Chunk { doc_path: "d.md".into(), location: "S2".into(),
                file_type: "md".into(), text: "SECOND chunk body".into() },
        ]).unwrap();

        g.put_node(&Node { id: "res-1".into(), node_type: "Resolution".into(),
            label: "r".into(), aliases: vec![], prov: prov() }).unwrap();
        for n in [1u64, 2] {
            g.put_edge(&Edge { from: "res-1".into(), to: format!("d.md#{n}"),
                edge_type: glossa::graph::MENTIONS.to_string(), prov: prov() }).unwrap();
        }

        let text = seed_source_text_all(dir.path(), &idx, &g, "res-1");
        assert!(text.contains("FIRST chunk body"), "missing chunk 1: {text}");
        assert!(text.contains("SECOND chunk body"), "missing chunk 2: {text}");
    }

    #[test]
    fn seed_source_text_all_is_empty_when_ungrounded() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        g.put_node(&Node { id: "res-x".into(), node_type: "Resolution".into(),
            label: "r".into(), aliases: vec![], prov: prov() }).unwrap();
        assert_eq!(seed_source_text_all(dir.path(), &idx, &g, "res-x"), "");
    }
}
