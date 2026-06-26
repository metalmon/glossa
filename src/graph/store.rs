use crate::graph::ontology::Ontology;
use crate::index::manifest::FileSig;
use anyhow::Context;
use redb::{Database, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};
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
        let t = match txn.open_table(NODES) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => return Err(e.into()),
        };
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
        match txn.open_table(NODES) {
            Ok(t) => Ok(t.len()?),
            Err(redb::TableError::TableDoesNotExist(_)) => Ok(0),
            Err(e) => Err(e.into()),
        }
    }

    pub fn edge_count(&self) -> anyhow::Result<u64> {
        let txn = self.db.begin_read()?;
        match txn.open_table(EDGES) {
            Ok(t) => Ok(t.len()?),
            Err(redb::TableError::TableDoesNotExist(_)) => Ok(0),
            Err(e) => Err(e.into()),
        }
    }
}

impl GraphStore {
    pub fn delete_by_source(&self, source_path: &str) -> anyhow::Result<usize> {
        let (node_ids, edge_keys) = {
            let txn = self.db.begin_read()?;
            let mut nids = Vec::new();
            match txn.open_table(NODES) {
                Ok(t) => {
                    for entry in t.iter()? {
                        let (k, v) = entry?;
                        let n: Node = postcard::from_bytes(v.value())?;
                        if n.prov.source_path == source_path {
                            nids.push(k.value().to_string());
                        }
                    }
                }
                Err(redb::TableError::TableDoesNotExist(_)) => {}
                Err(e) => return Err(e.into()),
            }
            let mut eks = Vec::new();
            match txn.open_table(EDGES) {
                Ok(t) => {
                    for entry in t.iter()? {
                        let (k, v) = entry?;
                        let e: Edge = postcard::from_bytes(v.value())?;
                        if e.prov.source_path == source_path {
                            eks.push(k.value().to_string());
                        }
                    }
                }
                Err(redb::TableError::TableDoesNotExist(_)) => {}
                Err(e) => return Err(e.into()),
            }
            (nids, eks)
        };
        let removed = node_ids.len() + edge_keys.len();
        if removed == 0 {
            return Ok(0);
        }
        let txn = self.db.begin_write()?;
        {
            let mut nt = txn.open_table(NODES)?;
            for id in &node_ids {
                nt.remove(id.as_str())?;
            }
            let mut et = txn.open_table(EDGES)?;
            for k in &edge_keys {
                et.remove(k.as_str())?;
            }
        }
        txn.commit()?;
        Ok(removed)
    }

    pub fn outgoing(&self, from: &str) -> anyhow::Result<Vec<Edge>> {
        let start = format!("{from}\u{0}");
        let end = format!("{from}\u{1}");
        let txn = self.db.begin_read()?;
        let t = match txn.open_table(EDGES) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };
        let mut out = Vec::new();
        for entry in t.range(start.as_str()..end.as_str())? {
            let (_, v) = entry?;
            out.push(postcard::from_bytes(v.value())?);
        }
        Ok(out)
    }
}

impl GraphStore {
    pub fn upsert(&self, ont: &Ontology, nodes: &[Node], edges: &[Edge]) -> anyhow::Result<()> {
        // Validate everything BEFORE writing anything.
        for n in nodes {
            if n.prov.source_path.is_empty() {
                anyhow::bail!("node {:?} has empty provenance", n.id);
            }
            ont.validate_node(&n.node_type).map_err(|e| anyhow::anyhow!(e))?;
        }
        let type_of = |id: &str, batch: &[Node]| -> Option<String> {
            batch.iter().find(|n| n.id == id).map(|n| n.node_type.clone())
                .or_else(|| self.get_node(id).ok().flatten().map(|n| n.node_type))
        };
        for e in edges {
            if e.prov.source_path.is_empty() {
                anyhow::bail!("edge {}->{} has empty provenance", e.from, e.to);
            }
            let ft = type_of(&e.from, nodes).unwrap_or_default();
            let tt = type_of(&e.to, nodes).unwrap_or_default();
            ont.validate_edge(&e.edge_type, &ft, &tt).map_err(|e| anyhow::anyhow!(e))?;
        }
        for n in nodes {
            self.put_node(n)?;
        }
        for e in edges {
            self.put_edge(e)?;
        }
        Ok(())
    }

    fn all_nodes(&self) -> anyhow::Result<Vec<Node>> {
        let txn = self.db.begin_read()?;
        let t = match txn.open_table(NODES) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(vec![]),
            Err(e) => return Err(e.into()),
        };
        let mut out = Vec::new();
        for entry in t.iter()? {
            let (_, v) = entry?;
            out.push(postcard::from_bytes(v.value())?);
        }
        Ok(out)
    }

    /// Resolve a name/term to graph node ids. Matching is fuzzy: a node hits when the query's
    /// stemmed terms are all present in its `label` or one of its `aliases` (same morphology
    /// pipeline as search), so word order and inflection don't matter. Exact (case-insensitive)
    /// label/alias equality is always honored too. NOTE: this is morphology- + order-tolerant,
    /// NOT transliteration-aware — Cyrillic "модбас" still won't match Latin "Modbus".
    pub fn resolve(&self, name: &str) -> anyhow::Result<Vec<String>> {
        use std::collections::BTreeSet;
        let needle = name.to_lowercase();
        let query: BTreeSet<String> = crate::index::multilang::analyze_terms(name).into_iter().collect();
        let terms = |s: &str| -> BTreeSet<String> {
            crate::index::multilang::analyze_terms(s).into_iter().collect()
        };
        let mut ids = Vec::new();
        for n in self.all_nodes()? {
            let exact = n.label.to_lowercase() == needle
                || n.aliases.iter().any(|a| a.to_lowercase() == needle);
            let fuzzy = !query.is_empty()
                && (query.is_subset(&terms(&n.label))
                    || n.aliases.iter().any(|a| query.is_subset(&terms(a))));
            if exact || fuzzy {
                ids.push(n.id);
            }
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod upsert_tests {
    use super::*;
    use crate::graph::ontology::Ontology;

    fn agent_prov() -> Provenance {
        Provenance {
            source_path: "contract.docx".into(),
            range: None,
            file_sig: None,
            origin: "agent".into(),
            confidence: 0.9,
            created_at: 0,
        }
    }

    const ONT: &str = r#"
[entities.Organization]
props = ["name"]
[relations.PARTY_TO]
from = ["Organization"]
to = ["Document"]
[validation]
strict = true
"#;

    #[test]
    fn upsert_validates_and_resolve_finds_by_alias() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(ONT).unwrap();

        let org = Node {
            id: "org:acme".into(),
            node_type: "Organization".into(),
            label: "Acme Corp".into(),
            aliases: vec!["ООО Акме".into(), "ACME".into()],
            prov: agent_prov(),
        };
        let doc = Node {
            id: "contract.docx".into(),
            node_type: "Document".into(),
            label: "contract.docx".into(),
            aliases: vec![],
            prov: agent_prov(),
        };
        let edge = Edge {
            from: "org:acme".into(),
            to: "contract.docx".into(),
            edge_type: "PARTY_TO".into(),
            prov: agent_prov(),
        };
        g.upsert(&ont, &[org, doc], &[edge]).unwrap();

        assert_eq!(g.resolve("ооо акме").unwrap(), vec!["org:acme".to_string()]);
        assert_eq!(g.resolve("ACME").unwrap(), vec!["org:acme".to_string()]);
    }

    #[test]
    fn resolve_fuzzy_matches_reordered_and_inflected() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(ONT).unwrap();
        let n = Node {
            id: "n:sync".into(),
            node_type: "Organization".into(),
            label: "Синхронизация пространства параметров".into(),
            aliases: vec![],
            prov: agent_prov(),
        };
        g.upsert(&ont, &[n], &[]).unwrap();
        // Reordered + different inflection ("пространство" vs "пространства"): exact match
        // returns nothing, fuzzy (stemmed token subset) must find the node.
        assert_eq!(
            g.resolve("пространство синхронизация").unwrap(),
            vec!["n:sync".to_string()]
        );
    }

    #[test]
    fn resolve_fuzzy_does_not_match_unrelated_terms() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(ONT).unwrap();
        let n = Node {
            id: "n:sync".into(),
            node_type: "Organization".into(),
            label: "Синхронизация пространства параметров".into(),
            aliases: vec![],
            prov: agent_prov(),
        };
        g.upsert(&ont, &[n], &[]).unwrap();
        // A token NOT present in the label must not match (subset, not overlap).
        assert!(g.resolve("температура двигателя").unwrap().is_empty());
        // Empty / punctuation-only query must not match everything.
        assert!(g.resolve("   ").unwrap().is_empty());
    }

    #[test]
    fn upsert_rejects_undeclared_type_and_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(ONT).unwrap();
        let bad = Node {
            id: "x".into(),
            node_type: "Alien".into(),
            label: "x".into(),
            aliases: vec![],
            prov: agent_prov(),
        };
        assert!(g.upsert(&ont, &[bad], &[]).is_err());
        assert_eq!(g.node_count().unwrap(), 0); // nothing written
    }

    #[test]
    fn upsert_rejects_empty_provenance() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::default();
        let mut p = agent_prov();
        p.source_path = String::new();
        let n = Node { id: "x".into(), node_type: "Document".into(), label: "x".into(), aliases: vec![], prov: p };
        assert!(g.upsert(&ont, &[n], &[]).is_err());
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
