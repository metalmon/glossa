//! `kbx build` — fetch a node's grounded chunk text (Task 7 of the `kbx build` pipeline).
//!
//! Feeds the judge (Task 8) the grounded SOURCE TEXT behind a graph node: the body of the chunk
//! its first `MENTIONS` edge points at. Per the brief, this is the parameter that moved judge
//! recall 0.20 → 1.00 in the earlier interview. Reuses glossa's existing chunk-read path
//! (`DocIndex::read_chunk_by_ord`, the same call the MCP `read` tool makes via
//! `crate::tools::read`) rather than reimplementing chunking.

use glossa::graph::store::GraphStore;
use glossa::graph::MENTIONS;
use glossa::index::store::DocIndex;
use std::path::Path;

/// Resolve `node_id`'s first `MENTIONS` edge target (a raw `<path>#n` reference), read that
/// chunk's body via the index, and return it. Empty `String` on ANY miss: no `MENTIONS` edge on
/// the node, an unparsable target, or a chunk the index doesn't have — the judge (Task 8) treats
/// an empty grounding as "nothing to show", not an error.
pub fn chunk_text(_root: &Path, idx: &DocIndex, node_id: &str, g: &GraphStore) -> String {
    let Ok(edges) = g.outgoing(node_id) else {
        return String::new();
    };
    let Some(target) = edges
        .iter()
        .find(|e| e.edge_type == MENTIONS)
        .map(|e| e.to.as_str())
    else {
        return String::new();
    };
    // MENTIONS targets are raw `<path>#n` strings — rsplit on `#` to split the ordinal off the
    // path (the path itself may legitimately contain no other `#`, so a plain rsplit_once is
    // exact and matches how candidates.rs's grounding_doc peels the same targets).
    let Some((path, n_str)) = target.rsplit_once('#') else {
        return String::new();
    };
    let Ok(n) = n_str.parse::<u64>() else {
        return String::new();
    };
    match idx.read_chunk_by_ord(path, n) {
        Ok(Some(chunk)) => chunk.body,
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glossa::graph::store::{Edge, Node, Provenance};

    #[test]
    fn chunk_text_reads_the_mentioned_chunk_body() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("d.md"), "# Title\n\nGrounded chunk body.\n").unwrap();
        glossa::index::store::index_dir(root, true).unwrap();
        let idx = DocIndex::open_or_create(root).unwrap();
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
            aliases: vec![],
            prov: prov.clone(),
        })
        .unwrap();
        g.put_edge(&Edge {
            from: "f1".into(),
            to: "d.md#1".into(),
            edge_type: MENTIONS.into(),
            prov,
        })
        .unwrap();

        let text = chunk_text(root, &idx, "f1", &g);
        assert!(
            text.contains("Grounded chunk body."),
            "expected the mentioned chunk's body, got: {text:?}"
        );
    }

    #[test]
    fn chunk_text_is_empty_when_node_has_no_mentions_edge() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("d.md"), "# Title\n\nGrounded chunk body.\n").unwrap();
        glossa::index::store::index_dir(root, true).unwrap();
        let idx = DocIndex::open_or_create(root).unwrap();
        let g = GraphStore::open(root).unwrap();

        g.put_node(&Node {
            id: "f1".into(),
            node_type: "Fact".into(),
            label: "n1".into(),
            aliases: vec![],
            prov: Provenance {
                source_path: "d.md".into(),
                range: None,
                file_sig: None,
                origin: "agent".into(),
                confidence: 0.9,
                created_at: 1,
            },
        })
        .unwrap();

        let text = chunk_text(root, &idx, "f1", &g);
        assert_eq!(text, "");
    }

    #[test]
    fn chunk_text_is_empty_for_an_unknown_node() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("d.md"), "# Title\n\nGrounded chunk body.\n").unwrap();
        glossa::index::store::index_dir(root, true).unwrap();
        let idx = DocIndex::open_or_create(root).unwrap();
        let g = GraphStore::open(root).unwrap();

        let text = chunk_text(root, &idx, "no-such-node", &g);
        assert_eq!(text, "");
    }
}
