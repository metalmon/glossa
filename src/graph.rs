pub mod doctor;
pub mod node_index;
pub mod ontology;
pub mod ontology_export;
pub mod ppr;
pub mod store;

/// The structural anchor edge from a reasoning node to the section that is its evidence. A FIXED
/// system contract (one of `CORE_EDGES`), like `CONTAINS`/`NEXT` — NOT an ontology-configurable
/// domain relation. Structural tools (`read`, glossary anchors) reference it directly, so they
/// never depend on the ontology for it.
pub const MENTIONS: &str = "MENTIONS";

/// The base REASONING node type: an atomic fact/step in a reasoning chain. Always permitted
/// (like MENTIONS is always permitted for structural edges) even under a strict ontology that
/// declares only its own domain entity types — permitting it never forces its creation, and
/// never changes validation for any other declared type. NOT a `CORE_NODES`/`STRUCTURAL_NODES`
/// member: those are structural (indexer-built, id-as-path); `Fact` is a reasoning node like any
/// agent-authored entity, just one the engine accepts unconditionally.
pub const FACT: &str = "Fact";

/// The base CHAINING relation between reasoning nodes (e.g. `Fact -[LEADS_TO]-> Fact`). Always
/// permitted under a strict ontology, mirroring `FACT`. Deliberately NOT a `CORE_EDGES` member:
/// `CORE_EDGES` are forced to `RelationRole::Grounding` (see `relation_role`), whereas `LEADS_TO`
/// must read as `RelationRole::Chaining` — it is a reasoning hop, not a grounding anchor.
pub const LEADS_TO: &str = "LEADS_TO";

/// The FIXED structural node types the indexer builds from documents (their ids ARE paths, so a
/// `read` of one is a document read, not a reasoning-node read). Everything else is a reasoning
/// node. The ontology may add domain entity types, but these structural ones are a system contract.
pub const STRUCTURAL_NODES: &[&str] = &["Document", "Section", "Term", "Topic"];

/// True when `id` is itself grounded — a `Section`/`Document` node, or a reasoning node with a
/// live [`MENTIONS`] edge to one (mirrors `tools::owning_doc`'s grounding definition) — or reaches
/// such a grounded terminal by walking forward along non-grounding ("chaining") edges. Cycle-
/// guarded (visited set). A lightweight, live, per-node BFS over the store — NOT a whole-graph
/// scan — so it is cheap enough to call once per coverage-check candidate (see
/// `tools::abstention::covered`).
///
/// This helper takes no [`ontology::Ontology`], so it approximates `RelationRole::Chaining`
/// rather than reading it precisely: [`MENTIONS`] and the generalize-layer `SIMILAR` cross-link
/// are the two fixed Grounding-role edge types this codebase uses outside the ontology (mirroring
/// `Ontology::relation_role`'s built-in defaults for them); every other edge type is walked as a
/// chaining hop, matching the ontology's fail-open "unrecognized role reads as Chaining" default.
/// A corpus that declares an EXTRA Grounding-role relation beyond those two is only approximated
/// here (that edge is walked as if it were chaining) — acceptable for an abstention SIGNAL, not a
/// hard security boundary.
pub fn grounded_or_chains_to_grounded(g: &store::GraphStore, id: &str) -> anyhow::Result<bool> {
    fn is_grounded(g: &store::GraphStore, id: &str) -> anyhow::Result<bool> {
        let Some(node) = g.get_node(id)? else {
            return Ok(false);
        };
        if matches!(node.node_type.as_str(), "Section" | "Document") {
            return Ok(true);
        }
        for e in g.outgoing(id)? {
            if e.edge_type != MENTIONS {
                continue;
            }
            if let Some(t) = g.get_node(&e.to)? {
                if matches!(t.node_type.as_str(), "Section" | "Document") {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut queue: std::collections::VecDeque<String> = std::collections::VecDeque::new();
    visited.insert(id.to_string());
    queue.push_back(id.to_string());
    while let Some(cur) = queue.pop_front() {
        if is_grounded(g, &cur)? {
            return Ok(true);
        }
        for e in g.outgoing(&cur)? {
            if e.edge_type == MENTIONS || e.edge_type == "SIMILAR" {
                continue;
            }
            if visited.insert(e.to.clone()) {
                queue.push_back(e.to);
            }
        }
    }
    Ok(false)
}

#[cfg(test)]
mod grounded_or_chains_to_grounded_tests {
    use super::*;
    use crate::graph::store::{Edge, GraphStore, Node, Provenance};

    fn prov() -> Provenance {
        Provenance {
            source_path: "d.pdf".into(),
            range: None,
            file_sig: None,
            origin: "agent".into(),
            confidence: 0.8,
            created_at: 1,
        }
    }
    fn node(id: &str, ty: &str, label: &str) -> Node {
        Node {
            id: id.into(),
            node_type: ty.into(),
            label: label.into(),
            aliases: Vec::new(),
            prov: prov(),
        }
    }
    fn edge(from: &str, rel: &str, to: &str) -> Edge {
        Edge {
            from: from.into(),
            to: to.into(),
            edge_type: rel.into(),
            prov: prov(),
        }
    }

    #[test]
    fn node_directly_grounded_via_mentions() {
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        g.put_node(&node("sec:a", "Section", "Intro")).unwrap();
        g.put_node(&node("fact:a", "Fact", "Some fact")).unwrap();
        g.put_edge(&edge("fact:a", MENTIONS, "sec:a")).unwrap();
        assert!(grounded_or_chains_to_grounded(&g, "fact:a").unwrap());
    }

    #[test]
    fn node_reaches_grounded_terminal_via_chaining_edge() {
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        g.put_node(&node("sec:a", "Section", "Intro")).unwrap();
        g.put_node(&node("sym:a", "Symptom", "Bus dropout")).unwrap();
        g.put_node(&node("res:a", "Resolution", "Raise timeout"))
            .unwrap();
        g.put_edge(&edge("sym:a", "RESOLVED_BY", "res:a")).unwrap();
        g.put_edge(&edge("res:a", MENTIONS, "sec:a")).unwrap();
        assert!(
            grounded_or_chains_to_grounded(&g, "sym:a").unwrap(),
            "query-side node one chaining hop from a grounded terminal must count"
        );
    }

    #[test]
    fn node_with_no_grounded_terminal_is_false() {
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        g.put_node(&node("sym:orphan", "Symptom", "Unrelated symptom"))
            .unwrap();
        assert!(!grounded_or_chains_to_grounded(&g, "sym:orphan").unwrap());
    }

    #[test]
    fn unknown_id_is_false() {
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        assert!(!grounded_or_chains_to_grounded(&g, "no:such:id").unwrap());
    }
}

pub mod agent;
pub mod build;
pub mod compose;
pub mod csr;
pub mod generalize;
pub mod handle;
pub mod io;
pub mod lock;
pub mod ops;
pub mod query;
pub mod temporal;
pub mod traverse;

/// Test-only guard serializing tests that mutate the process-global `GLOSSA_PPR_*` env vars
/// (`GLOSSA_PPR_SIM_WEIGHT`, `GLOSSA_PPR_SPINE_WEIGHT`, `GLOSSA_PPR_BRIDGE`) read by
/// `ppr::sim_weight`/`ppr::spine_weight`/`ppr::bridge_mode`. Process env is shared across the
/// whole test binary (spans both `graph::ppr` and `graph::compose`), so concurrent tests
/// setting/clearing the same var race under parallel test execution. Every test that reads or
/// mutates one of these vars must hold this lock for its full body:
/// `let _guard = crate::graph::test_env_lock::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());`
/// (`unwrap_or_else` recovers from poisoning so one panicking test doesn't cascade-fail siblings).
#[cfg(test)]
pub(crate) mod test_env_lock {
    use std::sync::Mutex;
    pub(crate) static ENV_LOCK: Mutex<()> = Mutex::new(());
}
