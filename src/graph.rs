pub mod store;
pub mod node_index;
pub mod ontology;

/// The structural anchor edge from a reasoning node to the section that is its evidence. A FIXED
/// system contract (one of `CORE_EDGES`), like `CONTAINS`/`NEXT` — NOT an ontology-configurable
/// domain relation. Structural tools (`read`, glossary anchors) reference it directly, so they
/// never depend on the ontology for it.
pub const MENTIONS: &str = "MENTIONS";

/// The FIXED structural node types the indexer builds from documents (their ids ARE paths, so a
/// `read` of one is a document read, not a reasoning-node read). Everything else is a reasoning
/// node. The ontology may add domain entity types, but these structural ones are a system contract.
pub const STRUCTURAL_NODES: &[&str] = &["Document", "Section", "Term", "Topic"];
pub mod traverse;
pub mod build;
pub mod agent;
pub mod io;
pub mod ops;
pub mod generalize;
