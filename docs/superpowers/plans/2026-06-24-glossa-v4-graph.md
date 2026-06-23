# glossa v4 — Milestone 4: Knowledge-graph substrate (redb) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pure-Rust knowledge-graph substrate — node/edge store with provenance, ontology validation, an agent-write API (`upsert`/`resolve`), bounded traversal (`neighbors`/`path`), and a deterministic auto layer-1 build during `kb index` — that Milestone 5 will expose over MCP for the connected agent to populate layers 3-4.

**Architecture:** Graph persists in `redb` (pure-Rust embedded store) at `<dir>/.glossa/graph.redb`. Nodes/edges are `postcard`-serialized structs carrying full provenance (source path, range, file signature, origin, confidence). The store is a disposable overlay over files (File-First): deleting it loses nothing. Code builds layers 1-2 deterministically; the agent (M5) writes layers 3-4 via the same validated `upsert`. Traversal is bounded BFS in code, not a query engine.

**Tech Stack:** Rust; `redb = "4.1"` (pure-Rust embedded store), `postcard = { version = "1", features = ["alloc"] }` (pure-Rust serde codec), `toml = "0.8"` (ontology schema), `serde` (already present). All pure Rust, offline, no `cc`/C.

## Global Constraints

- Pure Rust, single static binary, fully offline, **no C / no `cc`** (we removed `zstd-sys` in M3 and keep it out — `redb`/`postcard`/`toml` are all pure Rust; do NOT add `rusqlite`).
- Enforce the §12 File-First invariants: graph is derived & disposable; **every node/edge carries provenance** (`source_path`, `range`, `file_sig`, `origin`, `confidence`, `created_at`); file-level incremental via provenance ownership; staleness surfaced; bounded traversal only (no query engine); don't model the folder tree.
- Core node types `Document`/`Section`/`Term` and core edges `CONTAINS`/`MENTIONS`/`CO_OCCURS` are always valid (not subject to ontology `strict`). Entity/relation types from layers 3-4 are validated against `ontology.toml` (§11).
- Lexical/BM25 search (M1-M3) must keep working with NO graph — the graph is additive (graceful degradation).
- redb gotchas (verified): `Database::create(path)` is open-or-create; a `Table` borrows the write txn, so it must be **dropped (scope block) before `commit()`**; reads go through `redb::ReadableTable` (import it); `&str` keys are lexicographically ordered → prefix scan via `"p\x00".."p\x01"`.
- TDD: failing test first; frequent commits; DRY; YAGNI.

## Deferred to Milestone 5 (out of scope here)

- MCP server exposing `search`/`read`/`graph`/`resolve`/`neighbors` as agent tools.
- `read` (document region + image content) and `--expand` (glossary query expansion).
- Layer-2 co-occurrence `Term` extraction (only Document/Section auto-built here); agent-built layers 3-4 are exercised via `upsert` with synthetic data, not a live model.

---

### Task 1: redb graph store + model + provenance

**Files:**
- Modify: `Cargo.toml` (add `redb`, `postcard`)
- Create: `src/graph.rs` (module root: `pub mod store;`)
- Create: `src/graph/store.rs`
- Modify: `src/lib.rs` (add `pub mod graph;`)
- Test: `src/graph/store.rs` (inline, `tempfile`)

**Interfaces:**
- Consumes: `index::manifest::FileSig`, redb, postcard.
- Produces:
  - `graph::store::Provenance { source_path: String, range: Option<String>, file_sig: Option<FileSig>, origin: String, confidence: f32, created_at: u64 }`
  - `graph::store::Node { id: String, node_type: String, label: String, aliases: Vec<String>, prov: Provenance }`
  - `graph::store::Edge { from: String, to: String, edge_type: String, prov: Provenance }`
  - `graph::store::GraphStore` with `open(dir: &Path) -> anyhow::Result<GraphStore>`, `put_node(&Node)`, `get_node(&str) -> Option<Node>`, `put_edge(&Edge)`, `node_count()/edge_count()`.
  - edge key helper: `edge_key(from, edge_type, to) -> String` = `format!("{from}\x00{edge_type}\x00{to}")`.

- [ ] **Step 1: Write the failing test**

Create `src/graph/store.rs`:
```rust
use crate::index::manifest::FileSig;
use anyhow::Context;
use redb::{Database, ReadableTable, TableDefinition};
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
```

- [ ] **Step 2: Run it — verify it fails**

Run: `cargo test --lib graph::store`
Expected: FAIL — deps + module not present.

- [ ] **Step 3: Add deps + declare modules**

In `Cargo.toml` `[dependencies]`:
```toml
redb = "4.1"
postcard = { version = "1", features = ["alloc"] }
```
Create `src/graph.rs`:
```rust
pub mod store;
```
In `src/lib.rs`, add:
```rust
pub mod graph;
```

- [ ] **Step 4: Run it — verify it passes**

Run: `cargo test --lib graph::store` then `cargo test`
Expected: PASS. Verify NO new C dep: `cargo tree -i cc` → "did not match any packages".

Verification notes: `redb::ReadableTable` import is required for `get`/`len`/`range`; `t.len()` returns `Result<u64>`; `AccessGuard::value()` yields the bytes; if `Vec<u8>` vs `&[u8]` insert ergonomics differ, pass `bytes.as_slice()` (already done).

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock src/lib.rs src/graph.rs src/graph/store.rs
git commit -m "feat: redb graph store with provenance-bearing nodes/edges"
```

---

### Task 2: Ontology load + validation

**Files:**
- Modify: `Cargo.toml` (add `toml`)
- Create: `src/graph/ontology.rs`
- Modify: `src/graph.rs` (add `pub mod ontology;`)
- Test: `src/graph/ontology.rs` (inline)

**Interfaces:**
- Produces:
  - `graph::ontology::Ontology` with `parse(toml_str: &str) -> anyhow::Result<Ontology>`, `default() -> Ontology` (empty/permissive), and `validate_node(node_type: &str) -> Result<(), String>` / `validate_edge(edge_type, from_type, to_type) -> Result<(), String>`.
  - Core types are always allowed regardless of `strict`.

- [ ] **Step 1: Write the failing test**

Create `src/graph/ontology.rs`:
```rust
use serde::Deserialize;
use std::collections::BTreeMap;

const CORE_NODES: &[&str] = &["Document", "Section", "Term", "Topic"];
const CORE_EDGES: &[&str] = &["CONTAINS", "MENTIONS", "CO_OCCURS", "NEXT", "PREV"];

#[derive(Debug, Deserialize, Default)]
struct RawRelation {
    #[serde(default)]
    from: Vec<String>,
    #[serde(default)]
    to: Vec<String>,
}

#[derive(Debug, Deserialize, Default)]
struct RawValidation {
    #[serde(default)]
    strict: bool,
}

#[derive(Debug, Deserialize, Default)]
struct RawOntology {
    #[serde(default)]
    entities: BTreeMap<String, toml::Value>,
    #[serde(default)]
    relations: BTreeMap<String, RawRelation>,
    #[serde(default)]
    validation: RawValidation,
}

#[derive(Debug, Default)]
pub struct Ontology {
    entity_types: std::collections::BTreeSet<String>,
    relations: BTreeMap<String, RawRelation>,
    strict: bool,
}

impl Ontology {
    pub fn parse(toml_str: &str) -> anyhow::Result<Ontology> {
        let raw: RawOntology = toml::from_str(toml_str)?;
        Ok(Ontology {
            entity_types: raw.entities.keys().cloned().collect(),
            relations: raw.relations,
            strict: raw.validation.strict,
        })
    }

    pub fn validate_node(&self, node_type: &str) -> Result<(), String> {
        if CORE_NODES.contains(&node_type)
            || self.entity_types.contains(node_type)
            || !self.strict
        {
            Ok(())
        } else {
            Err(format!("unknown entity type '{node_type}' (strict)"))
        }
    }

    pub fn validate_edge(&self, edge_type: &str, from_type: &str, to_type: &str) -> Result<(), String> {
        if CORE_EDGES.contains(&edge_type) {
            return Ok(());
        }
        match self.relations.get(edge_type) {
            Some(r) => {
                let ok = |allowed: &Vec<String>, t: &str| {
                    allowed.is_empty() || allowed.iter().any(|a| a == "*" || a == t)
                };
                if ok(&r.from, from_type) && ok(&r.to, to_type) {
                    Ok(())
                } else {
                    Err(format!("relation '{edge_type}' endpoints {from_type}->{to_type} not allowed"))
                }
            }
            None if self.strict => Err(format!("unknown relation '{edge_type}' (strict)")),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOML: &str = r#"
[entities.Person]
props = ["full_name"]
[relations.AUTHORED_BY]
from = ["Document"]
to = ["Person"]
[validation]
strict = true
"#;

    #[test]
    fn core_types_always_allowed() {
        let o = Ontology::parse(TOML).unwrap();
        assert!(o.validate_node("Document").is_ok());
        assert!(o.validate_edge("CONTAINS", "Document", "Section").is_ok());
    }

    #[test]
    fn declared_types_pass_undeclared_fail_under_strict() {
        let o = Ontology::parse(TOML).unwrap();
        assert!(o.validate_node("Person").is_ok());
        assert!(o.validate_node("Alien").is_err());
        assert!(o.validate_edge("AUTHORED_BY", "Document", "Person").is_ok());
        assert!(o.validate_edge("AUTHORED_BY", "Document", "Alien").is_err());
        assert!(o.validate_edge("MADE_UP", "Person", "Person").is_err());
    }

    #[test]
    fn non_strict_allows_unknown() {
        let o = Ontology::default(); // strict = false
        assert!(o.validate_node("Anything").is_ok());
        assert!(o.validate_edge("WHATEVER", "A", "B").is_ok());
    }
}
```

- [ ] **Step 2: Run it — verify it fails, add dep + module**

Run: `cargo test --lib graph::ontology`
Expected: FAIL — `toml` dep + module missing.

In `Cargo.toml`: `toml = "0.8"`. In `src/graph.rs`: `pub mod ontology;`.
Re-run → PASS.

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml Cargo.lock src/graph.rs src/graph/ontology.rs
git commit -m "feat: ontology.toml load + node/edge validation (core types always allowed)"
```

---

### Task 3: `upsert` (validated, provenance-checked) + `resolve`

**Files:**
- Modify: `src/graph/store.rs` (add `upsert`, `resolve`, `all_nodes` helper)
- Test: `src/graph/store.rs` (inline)

**Interfaces:**
- Consumes: `Ontology`, `Node`, `Edge`.
- Produces:
  - `GraphStore::upsert(&self, ont: &Ontology, nodes: &[Node], edges: &[Edge]) -> anyhow::Result<()>` — validates every node/edge against the ontology (returns Err on first violation, writing nothing); requires each element's `prov.source_path` non-empty (invariant: no fact without provenance).
  - `GraphStore::resolve(&self, name: &str) -> anyhow::Result<Vec<String>>` — returns ids of nodes whose `label` or any `alias` case-insensitively equals `name` (entity-resolution helper for the agent).

- [ ] **Step 1: Write the failing test**

Append to `src/graph/store.rs` (add `use crate::graph::ontology::Ontology;` at top):
```rust
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
        let t = txn.open_table(NODES)?;
        let mut out = Vec::new();
        for entry in t.iter()? {
            let (_, v) = entry?;
            out.push(postcard::from_bytes(v.value())?);
        }
        Ok(out)
    }

    pub fn resolve(&self, name: &str) -> anyhow::Result<Vec<String>> {
        let needle = name.to_lowercase();
        let mut ids = Vec::new();
        for n in self.all_nodes()? {
            let hit = n.label.to_lowercase() == needle
                || n.aliases.iter().any(|a| a.to_lowercase() == needle);
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
```

- [ ] **Step 2: Run → fail → (impl is above) → pass**

Run: `cargo test --lib upsert_tests` (RED before pasting impl), then after impl `cargo test --lib`.
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/graph/store.rs
git commit -m "feat: validated graph upsert (provenance + ontology) and resolve()"
```

---

### Task 4: Bounded traversal — `neighbors` + `path`

**Files:**
- Create: `src/graph/traverse.rs`
- Modify: `src/graph.rs` (add `pub mod traverse;`)
- Modify: `src/graph/store.rs` (add `outgoing(&self, from: &str) -> Vec<Edge>` using the edge-key prefix scan)
- Test: `src/graph/traverse.rs` (inline)

**Interfaces:**
- Produces:
  - `GraphStore::outgoing(&self, from: &str) -> anyhow::Result<Vec<Edge>>` (prefix scan `"{from}\x00".."{from}\x01"`).
  - `graph::traverse::neighbors(g: &GraphStore, from: &str, edge_types: Option<&[String]>, depth: usize) -> anyhow::Result<Vec<String>>` (BFS to `depth`, returns reachable node ids excluding `from`, dedup).
  - `graph::traverse::path(g: &GraphStore, from: &str, to: &str, max_depth: usize) -> anyhow::Result<Option<Vec<String>>>` (BFS shortest path as id list, or None).

- [ ] **Step 1: Write the failing test**

Add to `src/graph/store.rs`:
```rust
impl GraphStore {
    pub fn outgoing(&self, from: &str) -> anyhow::Result<Vec<Edge>> {
        let start = format!("{from}\u{0}");
        let end = format!("{from}\u{1}");
        let txn = self.db.begin_read()?;
        let t = txn.open_table(EDGES)?;
        let mut out = Vec::new();
        for entry in t.range(start.as_str()..end.as_str())? {
            let (_, v) = entry?;
            out.push(postcard::from_bytes(v.value())?);
        }
        Ok(out)
    }
}
```

Create `src/graph/traverse.rs`:
```rust
use crate::graph::store::{Edge, GraphStore};
use std::collections::{HashSet, VecDeque};

fn type_match(e: &Edge, edge_types: Option<&[String]>) -> bool {
    match edge_types {
        None => true,
        Some(types) => types.iter().any(|t| t == &e.edge_type),
    }
}

pub fn neighbors(
    g: &GraphStore,
    from: &str,
    edge_types: Option<&[String]>,
    depth: usize,
) -> anyhow::Result<Vec<String>> {
    let mut visited: HashSet<String> = HashSet::from([from.to_string()]);
    let mut frontier: VecDeque<(String, usize)> = VecDeque::from([(from.to_string(), 0)]);
    let mut out = Vec::new();
    while let Some((node, d)) = frontier.pop_front() {
        if d >= depth {
            continue;
        }
        for e in g.outgoing(&node)? {
            if !type_match(&e, edge_types) {
                continue;
            }
            if visited.insert(e.to.clone()) {
                out.push(e.to.clone());
                frontier.push_back((e.to, d + 1));
            }
        }
    }
    Ok(out)
}

pub fn path(
    g: &GraphStore,
    from: &str,
    to: &str,
    max_depth: usize,
) -> anyhow::Result<Option<Vec<String>>> {
    if from == to {
        return Ok(Some(vec![from.to_string()]));
    }
    let mut visited: HashSet<String> = HashSet::from([from.to_string()]);
    // queue holds the full path so far
    let mut q: VecDeque<Vec<String>> = VecDeque::from([vec![from.to_string()]]);
    while let Some(p) = q.pop_front() {
        if p.len() > max_depth {
            continue;
        }
        let last = p.last().unwrap().clone();
        for e in g.outgoing(&last)? {
            if e.to == to {
                let mut found = p.clone();
                found.push(e.to);
                return Ok(Some(found));
            }
            if visited.insert(e.to.clone()) {
                let mut np = p.clone();
                np.push(e.to);
                q.push_back(np);
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::store::{Edge, GraphStore, Node, Provenance};

    fn prov() -> Provenance {
        Provenance { source_path: "s".into(), range: None, file_sig: None, origin: "agent".into(), confidence: 1.0, created_at: 0 }
    }
    fn node(g: &GraphStore, id: &str) {
        g.put_node(&Node { id: id.into(), node_type: "Entity".into(), label: id.into(), aliases: vec![], prov: prov() }).unwrap();
    }
    fn edge(g: &GraphStore, from: &str, to: &str, ty: &str) {
        g.put_edge(&Edge { from: from.into(), to: to.into(), edge_type: ty.into(), prov: prov() }).unwrap();
    }

    #[test]
    fn neighbors_respects_depth_and_type() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        for id in ["a", "b", "c", "d"] { node(&g, id); }
        edge(&g, "a", "b", "REL");
        edge(&g, "b", "c", "REL");
        edge(&g, "a", "d", "OTHER");

        let d1 = neighbors(&g, "a", None, 1).unwrap();
        assert!(d1.contains(&"b".to_string()) && d1.contains(&"d".to_string()) && !d1.contains(&"c".to_string()));

        let d2 = neighbors(&g, "a", None, 2).unwrap();
        assert!(d2.contains(&"c".to_string()));

        let only_rel = neighbors(&g, "a", Some(&["REL".to_string()]), 1).unwrap();
        assert_eq!(only_rel, vec!["b".to_string()]);
    }

    #[test]
    fn path_finds_chain() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        for id in ["a", "b", "c"] { node(&g, id); }
        edge(&g, "a", "b", "REL");
        edge(&g, "b", "c", "REL");
        assert_eq!(path(&g, "a", "c", 5).unwrap(), Some(vec!["a".into(), "b".into(), "c".into()]));
        assert_eq!(path(&g, "a", "z", 5).unwrap(), None);
    }
}
```

- [ ] **Step 2: Run → fail → declare module → pass**

Run `cargo test --lib graph::traverse` (RED), add `pub mod traverse;` to `src/graph.rs`, re-run → PASS, then `cargo test`.

- [ ] **Step 3: Commit**

```bash
git add src/graph.rs src/graph/store.rs src/graph/traverse.rs
git commit -m "feat: bounded graph traversal (neighbors + shortest path)"
```

---

### Task 5: Auto layer-1 build + provenance-owned incremental delete

**Files:**
- Modify: `src/graph/store.rs` (add `delete_by_source`)
- Create: `src/graph/build.rs` (`build_structural(g, chunks, sig_for)`)
- Modify: `src/graph.rs` (add `pub mod build;`)
- Modify: `src/index/store.rs` (`index_dir` also builds the structural graph per changed file)
- Test: `src/graph/build.rs` (inline) + an `index_dir` graph assertion

**Interfaces:**
- Produces:
  - `GraphStore::delete_by_source(&self, source_path: &str) -> anyhow::Result<usize>` — removes every node and edge whose `prov.source_path == source_path` (scan + remove; returns count). O(n); a secondary source index is a future optimization.
  - `graph::build::build_structural(g: &GraphStore, chunks: &[Chunk], sig: FileSig) -> anyhow::Result<()>` — for the chunks of ONE file, creates a `Document` node (id = path) + one `Section` node per chunk (id = `path#location`) + `CONTAINS` edges, all `origin = "auto-structural"`, provenance carrying `sig`.
  - `index_dir` calls `delete_by_source(path)` + `build_structural(...)` for each changed file (same loop that updates the tantivy index), so the graph mirrors the index incrementally.

- [ ] **Step 1: Write the failing test (delete_by_source + build_structural)**

Add to `src/graph/store.rs`:
```rust
impl GraphStore {
    pub fn delete_by_source(&self, source_path: &str) -> anyhow::Result<usize> {
        // Collect matching keys first (range/iter borrow conflicts with remove).
        let (node_ids, edge_keys) = {
            let txn = self.db.begin_read()?;
            let mut nids = Vec::new();
            {
                let t = txn.open_table(NODES)?;
                for entry in t.iter()? {
                    let (k, v) = entry?;
                    let n: Node = postcard::from_bytes(v.value())?;
                    if n.prov.source_path == source_path {
                        nids.push(k.value().to_string());
                    }
                }
            }
            let mut eks = Vec::new();
            {
                let t = txn.open_table(EDGES)?;
                for entry in t.iter()? {
                    let (k, v) = entry?;
                    let e: Edge = postcard::from_bytes(v.value())?;
                    if e.prov.source_path == source_path {
                        eks.push(k.value().to_string());
                    }
                }
            }
            (nids, eks)
        };
        let removed = node_ids.len() + edge_keys.len();
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
}
```

Create `src/graph/build.rs`:
```rust
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

        // 1 Document + 2 Section nodes, 2 CONTAINS edges.
        assert_eq!(g.node_count().unwrap(), 3);
        assert_eq!(g.edge_count().unwrap(), 2);
        assert!(g.get_node("a.md#Intro").unwrap().is_some());

        let removed = g.delete_by_source("a.md").unwrap();
        assert_eq!(removed, 5);
        assert_eq!(g.node_count().unwrap(), 0);
        assert_eq!(g.edge_count().unwrap(), 0);
    }
}
```

- [ ] **Step 2: Run → fail → declare module → pass**

Run `cargo test --lib graph::build` (RED), add `pub mod build;` to `src/graph.rs`, re-run → PASS.

- [ ] **Step 3: Wire into `index_dir`**

In `src/index/store.rs`, inside `index_dir`, after opening the tantivy index, open the graph and build structural graph for each (re)indexed file. Modify the per-file loop so the changed-file branch also updates the graph:
```rust
    let graph = crate::graph::store::GraphStore::open(dir)?;
```
In the loop, where a changed/new file is reindexed (the branch that calls `idx.delete_path(path)` + `idx.write_chunks(file_chunks)`), add:
```rust
        graph.delete_by_source(path)?;
        crate::graph::build::build_structural(&graph, file_chunks, sig)?;
```
And in the removed-files loop (where `idx.delete_path(old_path)` runs), add:
```rust
        graph.delete_by_source(old_path)?;
```

- [ ] **Step 4: Add an index_dir graph test**

Append to `src/index/store.rs` `incremental_tests`:
```rust
    #[test]
    fn index_dir_builds_structural_graph() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# Intro\nhello\n## Body\nworld\n").unwrap();
        index_dir(dir.path(), false).unwrap();
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        assert!(g.node_count().unwrap() >= 2); // Document + at least one Section
        let intro = g.resolve("Intro").unwrap();
        assert!(!intro.is_empty());
    }
```

- [ ] **Step 5: Run the full suite**

Run: `cargo test`
Expected: PASS (graph built during indexing; reindex drops+rebuilds per changed file).

- [ ] **Step 6: Commit**

```bash
git add src/graph.rs src/graph/store.rs src/graph/build.rs src/index/store.rs
git commit -m "feat: auto structural graph during index + provenance-owned incremental delete"
```

---

### Task 6: CLI — `kb graph stats` / `kb graph neighbors`

**Files:**
- Modify: `src/main.rs`
- Test: `tests/graph_it.rs`

**Interfaces:**
- Consumes: `glossa::graph::store::GraphStore`, `glossa::graph::traverse`.
- Produces: a `Graph` subcommand with `stats` and `neighbors` actions:
  - `kb graph stats [path]` → prints `nodes: N, edges: M`.
  - `kb graph neighbors <node-id> [path] [--depth N] [--type T]...` → prints reachable node ids.

- [ ] **Step 1: Write the failing test**

Create `tests/graph_it.rs`:
```rust
use assert_cmd::Command;
use predicates::str::contains;
use std::fs;

#[test]
fn graph_stats_and_neighbors_after_index() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), "# Intro\nhello\n## Body\nworld\n").unwrap();

    Command::cargo_bin("kb").unwrap()
        .args(["index", dir.path().to_str().unwrap()])
        .assert().success();

    Command::cargo_bin("kb").unwrap()
        .args(["graph", "stats", dir.path().to_str().unwrap()])
        .assert().success()
        .stdout(contains("nodes:"));

    // The Document node id is the file path; its CONTAINS neighbors are sections.
    let doc_id = dir.path().join("a.md").to_string_lossy().to_string();
    Command::cargo_bin("kb").unwrap()
        .args(["graph", "neighbors", &doc_id, dir.path().to_str().unwrap(), "--depth", "1"])
        .assert().success()
        .stdout(contains("Intro"));
}
```

- [ ] **Step 2: Run → fail**

Run: `cargo test --test graph_it`
Expected: FAIL — no `graph` subcommand.

- [ ] **Step 3: Add the CLI subcommand**

In `src/main.rs`, add a `Graph` variant with a nested action enum:
```rust
    /// Inspect the knowledge graph.
    Graph {
        #[command(subcommand)]
        action: GraphAction,
    },
```
```rust
#[derive(Subcommand)]
enum GraphAction {
    /// Print node/edge counts.
    Stats {
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Print nodes reachable from NODE_ID.
    Neighbors {
        node_id: String,
        #[arg(default_value = ".")]
        path: PathBuf,
        #[arg(long, default_value_t = 1)]
        depth: usize,
        #[arg(long = "type")]
        types: Vec<String>,
    },
}
```
In `main`, add the arm:
```rust
        Cmd::Graph { action } => match action {
            GraphAction::Stats { path } => {
                let g = glossa::graph::store::GraphStore::open(&path)?;
                println!("nodes: {}, edges: {}", g.node_count()?, g.edge_count()?);
                Ok(())
            }
            GraphAction::Neighbors { node_id, path, depth, types } => {
                let g = glossa::graph::store::GraphStore::open(&path)?;
                let filter = if types.is_empty() { None } else { Some(types.as_slice()) };
                for id in glossa::graph::traverse::neighbors(&g, &node_id, filter, depth)? {
                    println!("{id}");
                }
                Ok(())
            }
        },
```

- [ ] **Step 4: Run → pass + full suite**

Run: `cargo test --test graph_it` then `cargo test`.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/main.rs tests/graph_it.rs
git commit -m "feat: kb graph stats/neighbors CLI"
```

---

## Self-Review

**Spec coverage (Milestone 4 slice):**
- redb pure-Rust graph store + provenance-bearing nodes/edges → Task 1. ✓
- Ontology validation (core always allowed; strict for declared types) → Task 2. ✓
- Validated agent-write `upsert` + `resolve` → Task 3. ✓
- Bounded `neighbors`/`path` traversal (no query engine) → Task 4. ✓
- Auto layer-1 structural build + provenance-owned file-level incremental delete → Task 5. ✓
- CLI inspection surface → Task 6. ✓
- §12 invariants enforced: provenance required (Task 3 rejects empty), derived/disposable (store rebuilds from index), file-level incremental (Task 5), bounded traversal (Task 4). ✓
- Deferred (stated): MCP wiring, `read`/images, `--expand`, layer-2 term extraction, agent-built layers 3-4 with a live model (M5). ✓

**Placeholder scan:** none — every step has complete code + commands. Verification notes flag the few redb specifics to confirm on first compile (`ReadableTable` import, `len()`/`value()` ergonomics) — one-line confirms, not gaps.

**Type consistency:** `Node`/`Edge`/`Provenance` defined in Task 1, used unchanged in Tasks 3-6; `FileSig` reused from `index::manifest`; `GraphStore` methods (`open`/`put_*`/`get_node`/`*_count`/`outgoing`/`upsert`/`resolve`/`delete_by_source`) are defined once and consumed consistently; `Ontology::{parse,default,validate_node,validate_edge}` match between Tasks 2-3; traversal signatures match their CLI callers in Task 6.

**Dependency note:** new deps `redb 4.1`, `postcard 1`, `toml 0.8` — all pure Rust, offline, NO `cc`/C (verified: `redb` has no build-script/sys deps). Keeps the C-free build from M3. Do NOT introduce `rusqlite` (bundled SQLite would recompile C via cc).

**Performance note (acknowledged, not a gap):** `resolve` and `delete_by_source` scan all nodes/edges (O(n)). Fine for moderate KBs; a secondary `source_path → keys` index and a label index are future optimizations, logged here so they aren't mistaken for completeness.
