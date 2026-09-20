use crate::graph::store::{Edge, GraphStore, Node, Provenance};
use crate::index::manifest::FileSig;
use crate::model::Chunk;

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Deterministic Section node id for a chunk location within a document.
pub fn section_id(path: &str, location: &str) -> String {
    format!("{path}#{location}")
}

fn structural_prov(src_path: &str, sig: FileSig) -> Provenance {
    Provenance {
        source_path: src_path.to_string(),
        range: None,
        file_sig: Some(sig),
        origin: "auto-structural".into(),
        confidence: 1.0,
        created_at: now_secs(),
    }
}

/// Put the Document node for `path` (idempotent).
pub fn build_document(g: &GraphStore, path: &str, sig: FileSig) -> anyhow::Result<()> {
    let created_at = now_secs();
    g.put_node(&Node {
        id: path.to_string(),
        node_type: "Document".into(),
        label: path.to_string(),
        aliases: vec![],
        prov: Provenance {
            source_path: path.to_string(),
            range: None,
            file_sig: Some(sig),
            origin: "auto-structural".into(),
            confidence: 1.0,
            created_at,
        },
    })
}

/// Put one Section node + CONTAINS edge for a chunk.
pub fn build_section(g: &GraphStore, chunk: &Chunk, ord: u64, sig: FileSig) -> anyhow::Result<()> {
    let path = chunk.doc_path.to_string_lossy().to_string();
    let created_at = now_secs();
    let prov = Provenance {
        source_path: path.clone(),
        range: Some(chunk.location.clone()),
        file_sig: Some(sig),
        origin: "auto-structural".into(),
        confidence: 1.0,
        created_at,
    };
    // Section id is the 1-based ordinal (`#n` — the same key read/grep/search show the
    // agent), so a numeric `path#n` from resolve_section_ref always matches. The heading
    // stays as the node label (and prov range) for display and hierarchy breadcrumbs.
    let sec_id = section_id(&path, &ord.to_string());
    g.put_node(&Node {
        id: sec_id.clone(),
        node_type: "Section".into(),
        label: chunk.location.clone(),
        aliases: vec![],
        prov: prov.clone(),
    })?;
    g.put_edge(&Edge {
        from: path,
        to: sec_id,
        edge_type: "CONTAINS".into(),
        prov,
    })
}

/// Link two consecutive sections in document order: prev →NEXT→ cur and cur →PREV→ prev.
pub fn link_sequential(
    g: &GraphStore,
    prev_id: &str,
    cur_id: &str,
    sig: FileSig,
    src_path: &str,
) -> anyhow::Result<()> {
    g.put_edge(&Edge {
        from: prev_id.to_string(),
        to: cur_id.to_string(),
        edge_type: "NEXT".into(),
        prov: structural_prov(src_path, sig),
    })?;
    g.put_edge(&Edge {
        from: cur_id.to_string(),
        to: prev_id.to_string(),
        edge_type: "PREV".into(),
        prov: structural_prov(src_path, sig),
    })?;
    Ok(())
}

/// Link a child section to its nearest ancestor: child →PARENT→ parent and parent →CHILD→ child.
pub fn link_parent(
    g: &GraphStore,
    child_id: &str,
    parent_id: &str,
    sig: FileSig,
    src_path: &str,
) -> anyhow::Result<()> {
    g.put_edge(&Edge {
        from: child_id.to_string(),
        to: parent_id.to_string(),
        edge_type: "PARENT".into(),
        prov: structural_prov(src_path, sig),
    })?;
    g.put_edge(&Edge {
        from: parent_id.to_string(),
        to: child_id.to_string(),
        edge_type: "CHILD".into(),
        prov: structural_prov(src_path, sig),
    })?;
    Ok(())
}

/// Link a document to another document it explicitly references.
pub fn link_reference(
    g: &GraphStore,
    src_doc: &str,
    dst_doc: &str,
    sig: FileSig,
) -> anyhow::Result<()> {
    g.put_edge(&Edge {
        from: src_doc.to_string(),
        to: dst_doc.to_string(),
        edge_type: "REFERENCES".into(),
        prov: structural_prov(src_doc, sig),
    })
}

/// Nearest existing ancestor of `location` among the per-file `seen` (location → sec_id) map:
/// the longest breadcrumb prefix (split on " > ") that has a Section node. None for top-level / PDF.
pub fn nearest_ancestor(
    seen: &std::collections::HashMap<String, String>,
    location: &str,
) -> Option<String> {
    let segs: Vec<&str> = location.split(" > ").collect();
    for k in (1..segs.len()).rev() {
        let prefix = segs[..k].join(" > ");
        if let Some(id) = seen.get(&prefix) {
            return Some(id.clone());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_ancestor_picks_longest_existing_prefix() {
        let mut seen = std::collections::HashMap::new();
        seen.insert("A".to_string(), "id:A".to_string());
        seen.insert("A > B".to_string(), "id:AB".to_string());
        // child "A > B > C": nearest is "A > B"
        assert_eq!(
            nearest_ancestor(&seen, "A > B > C").as_deref(),
            Some("id:AB")
        );
        // gap: "A" exists but "A > X" does not -> child "A > X > Y" falls back to "A"
        assert_eq!(
            nearest_ancestor(&seen, "A > X > Y").as_deref(),
            Some("id:A")
        );
        // top-level has no ancestor
        assert_eq!(nearest_ancestor(&seen, "A"), None);
        // PDF page (no " > ")
        assert_eq!(nearest_ancestor(&seen, "p.3"), None);
    }
}
