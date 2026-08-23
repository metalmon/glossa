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
pub mod agent;
pub mod build;
pub mod generalize;
pub mod io;
pub mod lock;
pub mod ops;
pub mod query;
pub mod temporal;
pub mod compose;
pub mod traverse;
