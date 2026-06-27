use crate::graph::ontology::Ontology;
use crate::index::manifest::FileSig;
use anyhow::Context;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Mutex;

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

/// Normalize a label for deduplication: lowercase, trim, collapse runs of whitespace to a single space.
pub fn normalize_label(s: &str) -> String {
    s.trim().to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ")
}

pub struct GraphStore {
    conn: Mutex<Connection>,
}

impl GraphStore {
    pub fn open(dir: &Path) -> anyhow::Result<GraphStore> {
        let gdir = dir.join(".glossa");
        std::fs::create_dir_all(&gdir).with_context(|| format!("create {gdir:?}"))?;
        let conn = Connection::open(gdir.join("graph.sqlite")).context("open sqlite")?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;
             CREATE TABLE IF NOT EXISTS nodes (
               id TEXT PRIMARY KEY, node_type TEXT NOT NULL, label TEXT NOT NULL,
               aliases TEXT NOT NULL, source_path TEXT NOT NULL, range TEXT,
               file_sig TEXT, origin TEXT NOT NULL, confidence REAL NOT NULL, created_at INTEGER NOT NULL,
               label_norm TEXT NOT NULL DEFAULT '');
             CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
             CREATE INDEX IF NOT EXISTS idx_nodes_label_norm ON nodes(label_norm);
             CREATE TABLE IF NOT EXISTS edges (
               efrom TEXT NOT NULL, eto TEXT NOT NULL, edge_type TEXT NOT NULL,
               source_path TEXT NOT NULL, range TEXT, file_sig TEXT, origin TEXT NOT NULL,
               confidence REAL NOT NULL, created_at INTEGER NOT NULL,
               PRIMARY KEY (efrom, edge_type, eto));
             CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(efrom);
             CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(eto);",
        )
        .context("init schema")?;
        Ok(GraphStore { conn: Mutex::new(conn) })
    }

    // ── private helpers: take &Connection (no Mutex locking) ─────────────────

    fn row_to_node(row: &rusqlite::Row<'_>) -> rusqlite::Result<Node> {
        let aliases_json: String = row.get(3)?;
        let file_sig_json: Option<String> = row.get(6)?;
        let confidence: f64 = row.get(8)?;
        let created_at: i64 = row.get(9)?;
        let aliases: Vec<String> = serde_json::from_str(&aliases_json).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, Box::new(e))
        })?;
        let file_sig: Option<FileSig> = file_sig_json
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(6, rusqlite::types::Type::Text, Box::new(e))
            })?;
        Ok(Node {
            id: row.get(0)?,
            node_type: row.get(1)?,
            label: row.get(2)?,
            aliases,
            prov: Provenance {
                source_path: row.get(4)?,
                range: row.get(5)?,
                file_sig,
                origin: row.get(7)?,
                confidence: confidence as f32,
                created_at: created_at as u64,
            },
        })
    }

    fn row_to_edge(row: &rusqlite::Row<'_>) -> rusqlite::Result<Edge> {
        let file_sig_json: Option<String> = row.get(5)?;
        let confidence: f64 = row.get(7)?;
        let created_at: i64 = row.get(8)?;
        let file_sig: Option<FileSig> = file_sig_json
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(5, rusqlite::types::Type::Text, Box::new(e))
            })?;
        Ok(Edge {
            from: row.get(0)?,
            to: row.get(1)?,
            edge_type: row.get(2)?,
            prov: Provenance {
                source_path: row.get(3)?,
                range: row.get(4)?,
                file_sig,
                origin: row.get(6)?,
                confidence: confidence as f32,
                created_at: created_at as u64,
            },
        })
    }

    fn put_node_c(c: &Connection, node: &Node) -> anyhow::Result<()> {
        let aliases_json = serde_json::to_string(&node.aliases).context("ser aliases")?;
        let file_sig_json = node
            .prov
            .file_sig
            .as_ref()
            .map(|fs| serde_json::to_string(fs))
            .transpose()
            .context("ser file_sig")?;
        c.execute(
            "INSERT OR REPLACE INTO nodes \
             (id, node_type, label, aliases, source_path, range, file_sig, origin, confidence, created_at, label_norm) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            rusqlite::params![
                node.id,
                node.node_type,
                node.label,
                aliases_json,
                node.prov.source_path,
                node.prov.range,
                file_sig_json,
                node.prov.origin,
                node.prov.confidence as f64,
                node.prov.created_at as i64,
                normalize_label(&node.label),
            ],
        )
        .context("insert node")?;
        Ok(())
    }

    fn get_node_c(c: &Connection, id: &str) -> anyhow::Result<Option<Node>> {
        let mut stmt = c
            .prepare(
                "SELECT id, node_type, label, aliases, source_path, range, file_sig, origin, \
                 confidence, created_at FROM nodes WHERE id = ?1",
            )
            .context("prepare get_node")?;
        match stmt.query_row(rusqlite::params![id], Self::row_to_node) {
            Ok(n) => Ok(Some(n)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn put_edge_c(c: &Connection, edge: &Edge) -> anyhow::Result<()> {
        let file_sig_json = edge
            .prov
            .file_sig
            .as_ref()
            .map(|fs| serde_json::to_string(fs))
            .transpose()
            .context("ser edge file_sig")?;
        c.execute(
            "INSERT OR REPLACE INTO edges \
             (efrom, eto, edge_type, source_path, range, file_sig, origin, confidence, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                edge.from,
                edge.to,
                edge.edge_type,
                edge.prov.source_path,
                edge.prov.range,
                file_sig_json,
                edge.prov.origin,
                edge.prov.confidence as f64,
                edge.prov.created_at as i64,
            ],
        )
        .context("insert edge")?;
        Ok(())
    }

    fn all_nodes_c(c: &Connection) -> anyhow::Result<Vec<Node>> {
        let mut stmt = c
            .prepare(
                "SELECT id, node_type, label, aliases, source_path, range, file_sig, origin, \
                 confidence, created_at FROM nodes",
            )
            .context("prepare all_nodes")?;
        let rows = stmt
            .query_map([], Self::row_to_node)
            .context("query all_nodes")?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.context("read node row")?);
        }
        Ok(out)
    }

    fn outgoing_c(c: &Connection, from: &str) -> anyhow::Result<Vec<Edge>> {
        let mut stmt = c
            .prepare(
                "SELECT efrom, eto, edge_type, source_path, range, file_sig, origin, \
                 confidence, created_at FROM edges WHERE efrom = ?1",
            )
            .context("prepare outgoing")?;
        let rows = stmt
            .query_map(rusqlite::params![from], Self::row_to_edge)
            .context("query outgoing")?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.context("read edge row")?);
        }
        Ok(out)
    }

    /// All node ids whose normalized label equals `norm` — served by `idx_nodes_label_norm`
    /// (indexed exact lookup, O(log N)), used by the exact paths of `resolve`/`find_by_label*`.
    fn ids_by_label_norm_c(c: &Connection, norm: &str) -> anyhow::Result<Vec<String>> {
        let mut stmt = c
            .prepare("SELECT id FROM nodes WHERE label_norm = ?1")
            .context("prepare ids_by_label_norm")?;
        let rows = stmt
            .query_map(rusqlite::params![norm], |r| r.get::<_, String>(0))
            .context("query ids_by_label_norm")?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.context("read id row")?);
        }
        Ok(out)
    }

    // ── public API ────────────────────────────────────────────────────────────

    pub fn put_node(&self, node: &Node) -> anyhow::Result<()> {
        let c = self.conn.lock().unwrap();
        Self::put_node_c(&c, node)
    }

    pub fn get_node(&self, id: &str) -> anyhow::Result<Option<Node>> {
        let c = self.conn.lock().unwrap();
        Self::get_node_c(&c, id)
    }

    pub fn put_edge(&self, edge: &Edge) -> anyhow::Result<()> {
        let c = self.conn.lock().unwrap();
        Self::put_edge_c(&c, edge)
    }

    pub fn node_count(&self) -> anyhow::Result<u64> {
        let c = self.conn.lock().unwrap();
        let n: i64 = c
            .query_row("SELECT COUNT(*) FROM nodes", [], |r| r.get(0))
            .context("node_count")?;
        Ok(n as u64)
    }

    pub fn edge_count(&self) -> anyhow::Result<u64> {
        let c = self.conn.lock().unwrap();
        let n: i64 = c
            .query_row("SELECT COUNT(*) FROM edges", [], |r| r.get(0))
            .context("edge_count")?;
        Ok(n as u64)
    }

    pub fn delete_by_source(&self, source_path: &str) -> anyhow::Result<usize> {
        let c = self.conn.lock().unwrap();
        // Cascade first: drop edges that REFERENCE nodes from this source (regardless of the edge's
        // own source_path) so none is left dangling at a deleted node — same as delete_by_type.
        c.execute(
            "DELETE FROM edges WHERE efrom IN (SELECT id FROM nodes WHERE source_path = ?1) \
             OR eto IN (SELECT id FROM nodes WHERE source_path = ?1)",
            rusqlite::params![source_path],
        )
        .context("delete edges referencing source nodes")?;
        let ref_edges = c.changes() as usize;
        c.execute(
            "DELETE FROM nodes WHERE source_path = ?1",
            rusqlite::params![source_path],
        )
        .context("delete nodes by source")?;
        let nodes_deleted = c.changes() as usize;
        c.execute(
            "DELETE FROM edges WHERE source_path = ?1",
            rusqlite::params![source_path],
        )
        .context("delete edges by source")?;
        let edges_deleted = c.changes() as usize;
        Ok(nodes_deleted + ref_edges + edges_deleted)
    }

    /// Delete only the DOCUMENT-DERIVED layer (origin `auto-*`: structural + lexical), preserving
    /// the agent/curated reasoning graph. Used by `index_dir(force=true)` so a reindex rebuilds the
    /// structure from documents without destroying hand/agent-built knowledge. Returns count removed.
    pub fn delete_auto(&self) -> anyhow::Result<usize> {
        let c = self.conn.lock().unwrap();
        c.execute("DELETE FROM nodes WHERE origin LIKE 'auto-%'", [])
            .context("delete auto nodes")?;
        let nodes_deleted = c.changes() as usize;
        c.execute("DELETE FROM edges WHERE origin LIKE 'auto-%'", [])
            .context("delete auto edges")?;
        let edges_deleted = c.changes() as usize;
        Ok(nodes_deleted + edges_deleted)
    }

    /// Like `delete_by_source`, but only the `auto-*` layer for that path — agent/curated nodes and
    /// edges referencing the same document are preserved. Used by incremental indexing of a
    /// changed/removed file.
    pub fn delete_auto_by_source(&self, source_path: &str) -> anyhow::Result<usize> {
        let c = self.conn.lock().unwrap();
        c.execute(
            "DELETE FROM nodes WHERE source_path = ?1 AND origin LIKE 'auto-%'",
            rusqlite::params![source_path],
        )
        .context("delete auto nodes by source")?;
        let nodes_deleted = c.changes() as usize;
        c.execute(
            "DELETE FROM edges WHERE source_path = ?1 AND origin LIKE 'auto-%'",
            rusqlite::params![source_path],
        )
        .context("delete auto edges by source")?;
        let edges_deleted = c.changes() as usize;
        Ok(nodes_deleted + edges_deleted)
    }

    /// Delete every node of `node_type` plus all edges touching those nodes. Returns count removed.
    pub fn delete_by_type(&self, node_type: &str) -> anyhow::Result<usize> {
        let c = self.conn.lock().unwrap();
        c.execute(
            "DELETE FROM edges WHERE efrom IN (SELECT id FROM nodes WHERE node_type = ?1) \
             OR eto IN (SELECT id FROM nodes WHERE node_type = ?1)",
            rusqlite::params![node_type],
        )
        .context("delete edges by type")?;
        let edges_deleted = c.changes() as usize;
        c.execute(
            "DELETE FROM nodes WHERE node_type = ?1",
            rusqlite::params![node_type],
        )
        .context("delete nodes by type")?;
        let nodes_deleted = c.changes() as usize;
        Ok(edges_deleted + nodes_deleted)
    }

    pub fn outgoing(&self, from: &str) -> anyhow::Result<Vec<Edge>> {
        let c = self.conn.lock().unwrap();
        Self::outgoing_c(&c, from)
    }

    /// Edges pointing INTO `to` (`eto = to`). Used by export to capture inbound edges too.
    pub fn incoming(&self, to: &str) -> anyhow::Result<Vec<Edge>> {
        let c = self.conn.lock().unwrap();
        let mut stmt = c
            .prepare(
                "SELECT efrom, eto, edge_type, source_path, range, file_sig, origin, confidence, \
                 created_at FROM edges WHERE eto = ?1",
            )
            .context("prepare incoming")?;
        let rows = stmt
            .query_map(rusqlite::params![to], Self::row_to_edge)
            .context("query incoming")?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn all_nodes(&self) -> anyhow::Result<Vec<Node>> {
        let c = self.conn.lock().unwrap();
        Self::all_nodes_c(&c)
    }

    /// Return the id of the first node (any type) whose normalized label matches.
    pub fn find_by_label(&self, label: &str) -> anyhow::Result<Option<String>> {
        let c = self.conn.lock().unwrap();
        Ok(Self::ids_by_label_norm_c(&c, &normalize_label(label))?
            .into_iter()
            .next())
    }

    /// All node ids whose normalized label equals `normalize_label(label)` — indexed exact lookup
    /// (O(log N)). Lets callers resolve a label to existing node(s) without loading the whole graph.
    pub fn ids_by_label_norm(&self, label: &str) -> anyhow::Result<Vec<String>> {
        let c = self.conn.lock().unwrap();
        Self::ids_by_label_norm_c(&c, &normalize_label(label))
    }

    /// Update the `label` and/or `node_type` of the node with the given id in place.
    /// Only fields that are `Some` are updated. Returns rows changed (0 or 1).
    /// If both args are `None`, does nothing and returns 0.
    pub fn update_node(&self, id: &str, new_label: Option<&str>, new_type: Option<&str>) -> anyhow::Result<usize> {
        if new_label.is_none() && new_type.is_none() {
            return Ok(0);
        }
        let c = self.conn.lock().unwrap();
        match (new_label, new_type) {
            (Some(lbl), Some(typ)) => {
                c.execute(
                    "UPDATE nodes SET label = ?1, label_norm = ?2, node_type = ?3 WHERE id = ?4",
                    rusqlite::params![lbl, normalize_label(lbl), typ, id],
                )
                .context("update_node label+type")?;
            }
            (Some(lbl), None) => {
                c.execute(
                    "UPDATE nodes SET label = ?1, label_norm = ?2 WHERE id = ?3",
                    rusqlite::params![lbl, normalize_label(lbl), id],
                )
                .context("update_node label")?;
            }
            (None, Some(typ)) => {
                c.execute(
                    "UPDATE nodes SET node_type = ?1 WHERE id = ?2",
                    rusqlite::params![typ, id],
                )
                .context("update_node type")?;
            }
            (None, None) => unreachable!(),
        }
        Ok(c.changes() as usize)
    }

    /// Return the id of the first node with matching node_type and normalized label.
    pub fn find_by_label_type(&self, label: &str, node_type: &str) -> anyhow::Result<Option<String>> {
        let c = self.conn.lock().unwrap();
        let mut stmt = c
            .prepare("SELECT id FROM nodes WHERE label_norm = ?1 AND node_type = ?2 LIMIT 1")
            .context("prepare find_by_label_type")?;
        match stmt.query_row(
            rusqlite::params![normalize_label(label), node_type],
            |r| r.get::<_, String>(0),
        ) {
            Ok(id) => Ok(Some(id)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Delete a node by id AND every edge that references it. Returns (#nodes + #edges) removed.
    pub fn delete_node(&self, id: &str) -> anyhow::Result<usize> {
        let c = self.conn.lock().unwrap();
        c.execute(
            "DELETE FROM edges WHERE efrom = ?1 OR eto = ?1",
            rusqlite::params![id],
        )
        .context("delete edges for node")?;
        let edges_deleted = c.changes() as usize;
        c.execute("DELETE FROM nodes WHERE id = ?1", rusqlite::params![id])
            .context("delete node")?;
        let nodes_deleted = c.changes() as usize;
        Ok(nodes_deleted + edges_deleted)
    }

    /// Delete the single edge matching (from, edge_type, to). Returns changes().
    pub fn delete_edge(&self, from: &str, edge_type: &str, to: &str) -> anyhow::Result<usize> {
        let c = self.conn.lock().unwrap();
        c.execute(
            "DELETE FROM edges WHERE efrom = ?1 AND edge_type = ?2 AND eto = ?3",
            rusqlite::params![from, edge_type, to],
        )
        .context("delete edge")?;
        Ok(c.changes() as usize)
    }

    pub fn upsert(&self, ont: &Ontology, nodes: &[Node], edges: &[Edge]) -> anyhow::Result<()> {
        // Lock ONCE for the entire operation — helpers use _c variants to avoid deadlock.
        let c = self.conn.lock().unwrap();

        // Validate everything BEFORE writing anything.
        for n in nodes {
            if n.prov.source_path.is_empty() {
                anyhow::bail!("node {:?} has empty provenance", n.id);
            }
            ont.validate_node(&n.node_type).map_err(|e| anyhow::anyhow!(e))?;
        }
        let type_of = |id: &str, batch: &[Node]| -> Option<String> {
            batch
                .iter()
                .find(|n| n.id == id)
                .map(|n| n.node_type.clone())
                .or_else(|| Self::get_node_c(&c, id).ok().flatten().map(|n| n.node_type))
        };
        for e in edges {
            if e.prov.source_path.is_empty() {
                anyhow::bail!("edge {}->{} has empty provenance", e.from, e.to);
            }
            let ft = type_of(&e.from, nodes).unwrap_or_default();
            let tt = type_of(&e.to, nodes).unwrap_or_default();
            ont.validate_edge(&e.edge_type, &ft, &tt)
                .map_err(|e| anyhow::anyhow!(e))?;
        }

        // Write atomically.
        let txn = c.unchecked_transaction().context("begin upsert txn")?;
        for n in nodes {
            Self::put_node_c(&txn, n)?;
        }
        for e in edges {
            Self::put_edge_c(&txn, e)?;
        }
        txn.commit().context("commit upsert")?;
        Ok(())
    }

    /// Resolve a name/term to graph node ids. Matching is fuzzy: a node hits when the query's
    /// stemmed terms are all present in its `label` or one of its `aliases` (same morphology
    /// pipeline as search), so word order and inflection don't matter. Exact (case-insensitive)
    /// label/alias equality is always honored too. NOTE: this is morphology- + order-tolerant,
    /// NOT transliteration-aware — Cyrillic "модбас" still won't match Latin "Modbus".
    pub fn resolve(&self, name: &str) -> anyhow::Result<Vec<String>> {
        use std::collections::BTreeSet;
        let c = self.conn.lock().unwrap();
        // Fast path: exact (normalized) label match via the label_norm index — the common case
        // during enrichment. Returns ALL nodes sharing that normalized label (near-dups expected).
        let exact = Self::ids_by_label_norm_c(&c, &normalize_label(name))?;
        if !exact.is_empty() {
            return Ok(exact);
        }
        // Fallback: morphology/order-tolerant fuzzy over all nodes. Also covers an exactly-matching
        // ALIAS (the query's terms are a subset of that alias's terms). Stemmers built ONCE per call
        // (TermAnalyzer, proven-equivalent to analyze_terms); scratch set reused across labels.
        let analyzer = crate::index::multilang::TermAnalyzer::new();
        let mut query: BTreeSet<String> = BTreeSet::new();
        analyzer.terms(name, &mut query);
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let mut cand: BTreeSet<String> = BTreeSet::new();
        let mut ids = Vec::new();
        for n in Self::all_nodes_c(&c)? {
            analyzer.terms(&n.label, &mut cand);
            let hit = query.is_subset(&cand)
                || n.aliases.iter().any(|a| {
                    analyzer.terms(a, &mut cand);
                    query.is_subset(&cand)
                });
            if hit {
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
    fn update_node_keeps_label_norm_index_consistent() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(ONT).unwrap();
        let n = Node {
            id: "org:x".into(),
            node_type: "Organization".into(),
            label: "Старое Имя".into(),
            aliases: vec![],
            prov: agent_prov(),
        };
        g.upsert(&ont, &[n], &[]).unwrap();
        // exact lookup served by the label_norm index
        assert_eq!(g.resolve("Старое Имя").unwrap(), vec!["org:x".to_string()]);
        assert_eq!(g.find_by_label("старое имя").unwrap(), Some("org:x".to_string()));

        // rename → label_norm must follow, otherwise the index goes stale
        g.update_node("org:x", Some("Новое Имя"), None).unwrap();
        assert_eq!(g.find_by_label("Новое Имя").unwrap(), Some("org:x".to_string()));
        assert_eq!(g.find_by_label("Старое Имя").unwrap(), None);
        assert_eq!(g.resolve("новое имя").unwrap(), vec!["org:x".to_string()]);
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
        let n = Node {
            id: "x".into(),
            node_type: "Document".into(),
            label: "x".into(),
            aliases: vec![],
            prov: p,
        };
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

    #[test]
    fn multiprocess_visibility() {
        // Two separate GraphStore handles on the same dir simulate two processes.
        // SQLite WAL must make a committed write visible to the second connection.
        let dir = tempfile::tempdir().unwrap();
        let writer = GraphStore::open(dir.path()).unwrap();
        let reader = GraphStore::open(dir.path()).unwrap();

        let n = Node {
            id: "vis:test".into(),
            node_type: "Document".into(),
            label: "visibility test".into(),
            aliases: vec![],
            prov: prov(),
        };
        writer.put_node(&n).unwrap();

        assert_eq!(reader.get_node("vis:test").unwrap(), Some(n));
        assert_eq!(reader.node_count().unwrap(), 1);
    }
}
