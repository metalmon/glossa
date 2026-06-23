use crate::index::manifest::FileSig;
use anyhow::Context;
use redb::{Database, ReadableDatabase, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};
use std::path::Path;

const NODES: TableDefinition<&str, &[u8]> = TableDefinition::new("nodes");
const EDGES: TableDefinition<&str, &[u8]> = TableDefinition::new("edges");

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Provenance {
    pub source_path: String,
    pub range: Option<String>,
    pub file_sig: Option<FileSig>,
    pub origin: String, // "auto-structural" | "auto-lexical" | "agent" | "curated"
    pub confidence: f32,
    pub created_at: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Node {
    pub id: String,
    pub node_type: String,
    pub label: String,
    pub aliases: Vec<String>,
    pub prov: Provenance,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub edge_type: String,
    pub prov: Provenance,
}

pub fn edge_key(from: &str, edge_type: &str, to: &str) -> String {
    format!("{from}\u{0}{edge_type}\u{0}{to}")
}

pub struct GraphStore {
    db: Database,
}

impl GraphStore {
    pub fn open(dir: &Path) -> anyhow::Result<GraphStore> {
        let gdir = dir.join(".glossa");
        std::fs::create_dir_all(&gdir).with_context(|| format!("create {gdir:?}"))?;
        let db = Database::create(gdir.join("graph.redb")).context("open redb")?;
        Ok(GraphStore { db })
    }

    pub fn put_node(&self, node: &Node) -> anyhow::Result<()> {
        let bytes = postcard::to_allocvec(node).context("ser node")?;
        let txn = self.db.begin_write()?;
        {
            let mut t = txn.open_table(NODES)?;
            t.insert(node.id.as_str(), bytes.as_slice())?;
        }
        txn.commit()?;
        Ok(())
    }

    pub fn get_node(&self, id: &str) -> anyhow::Result<Option<Node>> {
        let txn = self.db.begin_read()?;
        let t = txn.open_table(NODES)?;
        match t.get(id)? {
            Some(g) => Ok(Some(postcard::from_bytes(g.value()).context("de node")?)),
            None => Ok(None),
        }
    }

    pub fn put_edge(&self, edge: &Edge) -> anyhow::Result<()> {
        let key = edge_key(&edge.from, &edge.edge_type, &edge.to);
        let bytes = postcard::to_allocvec(edge).context("ser edge")?;
        let txn = self.db.begin_write()?;
        {
            let mut t = txn.open_table(EDGES)?;
            t.insert(key.as_str(), bytes.as_slice())?;
        }
        txn.commit()?;
        Ok(())
    }

    pub fn node_count(&self) -> anyhow::Result<u64> {
        let txn = self.db.begin_read()?;
        let t = txn.open_table(NODES)?;
        Ok(t.len()?)
    }

    pub fn edge_count(&self) -> anyhow::Result<u64> {
        let txn = self.db.begin_read()?;
        let t = txn.open_table(EDGES)?;
        Ok(t.len()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prov() -> Provenance {
        Provenance {
            source_path: "a.md".into(),
            range: Some("Intro".into()),
            file_sig: None,
            origin: "auto-structural".into(),
            confidence: 1.0,
            created_at: 0,
        }
    }

    #[test]
    fn node_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let n = Node {
            id: "a.md".into(),
            node_type: "Document".into(),
            label: "a.md".into(),
            aliases: vec![],
            prov: prov(),
        };
        g.put_node(&n).unwrap();
        assert_eq!(g.get_node("a.md").unwrap(), Some(n));
        assert_eq!(g.get_node("missing").unwrap(), None);
        assert_eq!(g.node_count().unwrap(), 1);
    }

    #[test]
    fn edge_persists_and_counts() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        g.put_edge(&Edge {
            from: "a.md".into(),
            to: "a.md#Intro".into(),
            edge_type: "CONTAINS".into(),
            prov: prov(),
        })
        .unwrap();
        assert_eq!(g.edge_count().unwrap(), 1);
    }
}
