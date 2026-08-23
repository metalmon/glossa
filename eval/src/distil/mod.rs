//! `kbx distil` scaffold: builds an ontology-TYPED reasoning layer, gold-anchored, alongside
//! the flat `Fact`/`LEADS_TO` base layer (see docs/superpowers/specs/2026-08-23-kbx-distil-design.md).
//!
//! This module currently holds only the schema-graph renderer: the piece that makes the
//! rest of distil ontology-general. It turns an `Ontology` (entity_types + relations, each
//! carrying `from`/`to` types and a `RelationRole`) into a compact, prompt-injectable text
//! block so the distil prompt is parameterized by the corpus's own ontology — no domain type
//! names (Symptom/Cause/...) are ever hardcoded in code or prompts.

use glossa::graph::ontology::Ontology;

mod chain;
mod run;
pub use chain::{chain_one_gold, DistilStats};
pub use run::{run_distil, DistilArgs};

/// Render the ontology's schema-graph (entity types + typed relations) as a compact text
/// block suitable for injecting into a prompt. Lists every `entity_type`, marking those that
/// `requires_grounding` and/or `requires_validity`, followed by every declared relation as
/// `from_type --RELATION--> to_type` with its `role` (Chaining/Grounding/Attribute).
///
/// Ontology-general by construction: everything is driven off `ont.entity_types()` and
/// `ont.raw_relations()` — nothing here is specific to any domain's type names.
pub fn schema_graph_block(ont: &Ontology) -> String {
    let mut out = String::new();

    out.push_str("Entity types:\n");
    for t in ont.entity_types() {
        let mut flags = Vec::new();
        if ont.requires_grounding(t) {
            flags.push("requires_grounding");
        }
        if ont.requires_validity(t) {
            flags.push("requires_validity");
        }
        if flags.is_empty() {
            out.push_str(&format!("  - {t}\n"));
        } else {
            out.push_str(&format!("  - {t} [{}]\n", flags.join(", ")));
        }
    }

    out.push_str("Relations:\n");
    for (name, rel) in ont.raw_relations() {
        let role = format!("{:?}", rel.role);
        if rel.from.is_empty() && rel.to.is_empty() {
            out.push_str(&format!("  - {name} ({role})\n"));
            continue;
        }
        for from_ty in &rel.from {
            if rel.to.is_empty() {
                out.push_str(&format!("  - {from_ty} --{name}--> ? ({role})\n"));
                continue;
            }
            for to_ty in &rel.to {
                out.push_str(&format!(
                    "  - {from_ty} --{name}--> {to_ty} ({role})\n"
                ));
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny 2-type/1-relation ontology, mirroring the construction pattern used by
    /// `src/graph/ontology.rs`'s own unit tests (`Ontology::parse` on an inline TOML string).
    const TOML: &str = r#"
[entities.A]
requires_grounding = true
[entities.B]

[relations.REL]
from = ["A"]
to = ["B"]
role = "chaining"
"#;

    #[test]
    fn schema_graph_block_renders_types_and_relation() {
        let ont = Ontology::parse(TOML).expect("tiny ontology parses");
        let block = schema_graph_block(&ont);

        // Both entity type names present.
        assert!(block.contains('A'), "block missing type A:\n{block}");
        assert!(block.contains('B'), "block missing type B:\n{block}");

        // The relation rendered as a typed schema-graph edge.
        assert!(
            block.contains("A --REL--> B"),
            "block missing typed edge A --REL--> B:\n{block}"
        );

        // The requires_grounding-marked type carries a visible marker.
        assert!(
            block.contains("A [requires_grounding]"),
            "block missing requires_grounding marker on A:\n{block}"
        );
    }
}
