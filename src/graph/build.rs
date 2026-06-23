use crate::graph::store::{Edge, GraphStore, Node, Provenance};
use crate::index::manifest::FileSig;
use crate::model::Chunk;

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Build deterministic structural graph (layer 1) for one file's chunks:
/// a Document node + a Section node per chunk + CONTAINS edges.
pub fn build_structural(g: &GraphStore, chunks: &[Chunk], sig: FileSig) -> anyhow::Result<()> {
    if chunks.is_empty() {
        return Ok(());
    }
    let path = chunks[0].doc_path.to_string_lossy().to_string();
    let created_at = now_secs();
    let prov = |range: Option<String>| Provenance {
        source_path: path.clone(),
        range,
        file_sig: Some(sig),
        origin: "auto-structural".into(),
        confidence: 1.0,
        created_at,
    };

    g.put_node(&Node {
        id: path.clone(),
        node_type: "Document".into(),
        label: path.clone(),
        aliases: vec![],
        prov: prov(None),
    })?;

    for c in chunks {
        let sec_id = format!("{path}#{}", c.location);
        g.put_node(&Node {
            id: sec_id.clone(),
            node_type: "Section".into(),
            label: c.location.clone(),
            aliases: vec![],
            prov: prov(Some(c.location.clone())),
        })?;
        g.put_edge(&Edge {
            from: path.clone(),
            to: sec_id,
            edge_type: "CONTAINS".into(),
            prov: prov(Some(c.location.clone())),
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn chunk(path: &str, loc: &str) -> Chunk {
        Chunk { doc_path: PathBuf::from(path), location: loc.into(), file_type: "md".into(), text: "t".into() }
    }

    #[test]
    fn builds_document_and_sections_then_deletes_by_source() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let sig = FileSig { mtime_secs: 1, size: 2 };
        build_structural(&g, &[chunk("a.md", "Intro"), chunk("a.md", "Body")], sig).unwrap();

        assert_eq!(g.node_count().unwrap(), 3); // 1 Document + 2 Section
        assert_eq!(g.edge_count().unwrap(), 2); // 2 CONTAINS
        assert!(g.get_node("a.md#Intro").unwrap().is_some());

        let removed = g.delete_by_source("a.md").unwrap();
        assert_eq!(removed, 5);
        assert_eq!(g.node_count().unwrap(), 0);
        assert_eq!(g.edge_count().unwrap(), 0);
    }
}
