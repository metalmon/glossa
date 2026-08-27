//! `kbx reason` scaffold: builds an ontology-TYPED reasoning layer, gold-anchored, alongside
//! the flat `Fact`/`LEADS_TO` base layer (see docs/superpowers/specs/2026-08-23-kbx-distil-design.md,
//! written before this module was renamed from `distil` to `reason`).
//!
//! This module currently holds only the schema-graph renderer: the piece that makes the
//! rest of reason ontology-general. It turns an `Ontology` (entity_types + relations, each
//! carrying `from`/`to` types and a `RelationRole`) into a compact, prompt-injectable text
//! block so the reason prompt is parameterized by the corpus's own ontology — no domain type
//! names (Symptom/Cause/...) are ever hardcoded in code or prompts.

use glossa::graph::ontology::Ontology;

mod run;
mod seed;
pub use run::{run_reason, ReasonArgs};
pub use seed::{chain_one_seed, ReasonStats};

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
        if let Some(d) = ont.description(t) {
            out.push_str(&format!("      {d}\n"));
        }
    }

    out.push_str("Relations:\n");
    for (name, rel) in ont.raw_relations() {
        let role = format!("{:?}", rel.role);
        if rel.from.is_empty() && rel.to.is_empty() {
            out.push_str(&format!("  - {name} ({role})\n"));
        } else {
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
        // Description printed ONCE per relation, after all its edge lines — not once per
        // from x to pair (a relation with multiple from/to types would otherwise repeat it).
        if let Some(d) = ont.description(name) {
            out.push_str(&format!("      {d}\n"));
        }
    }

    out
}

/// Compact block listing ONLY the ontology's requires_grounding entity types with their
/// descriptions — the node types the build harvest may create. No relations (harvest writes
/// nodes, not the reasoning spine).
pub fn grounding_schema_block(ont: &Ontology) -> String {
    let mut out = String::from("Node types you create (each grounded to the section it is read from):\n");
    for t in ont.entity_types() {
        if !ont.requires_grounding(t) { continue; }
        out.push_str(&format!("  - {t}\n"));
        if let Some(d) = ont.description(t) {
            out.push_str(&format!("      {d}\n"));
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

    #[test]
    fn schema_graph_block_prints_descriptions() {
        let ont = Ontology::parse(r#"
[entities.A]
requires_grounding = true
description = "a thing that prescribes an action"
[entities.B]
[relations.REL]
from = ["A"]
to = ["B"]
role = "chaining"
description = "links A to B"
"#).unwrap();
        let s = schema_graph_block(&ont);
        assert!(s.contains("a thing that prescribes an action"), "entity desc missing:\n{s}");
        assert!(s.contains("links A to B"), "relation desc missing:\n{s}");
    }

    #[test]
    fn schema_graph_block_relation_description_printed_once_per_relation() {
        // REL has TWO `to` types, so it renders two edge lines (A --REL--> B, A --REL--> C).
        // The description must appear exactly once for the relation, not once per edge line.
        let ont = Ontology::parse(r#"
[entities.A]
[entities.B]
[entities.C]
[relations.REL]
from = ["A"]
to = ["B", "C"]
role = "chaining"
description = "links A to B"
"#).unwrap();
        let s = schema_graph_block(&ont);
        assert!(s.contains("A --REL--> B"), "missing edge A->B:\n{s}");
        assert!(s.contains("A --REL--> C"), "missing edge A->C:\n{s}");
        assert_eq!(
            s.matches("links A to B").count(),
            1,
            "relation description should print exactly once, not once per edge line:\n{s}"
        );
    }

    #[test]
    fn grounding_schema_block_only_grounding_types() {
        let ont = Ontology::parse(r#"
[entities.Res]
requires_grounding = true
description = "prescribed action"
[entities.Sym]
description = "a reported problem"
[relations.R]
from=["Sym"]
to=["Res"]
role="chaining"
"#).unwrap();
        let s = grounding_schema_block(&ont);
        assert!(s.contains("Res"), "grounding type missing:\n{s}");
        assert!(s.contains("prescribed action"));
        assert!(!s.contains("Sym"), "non-grounding type leaked:\n{s}");
        assert!(!s.contains("--R-->"), "relations leaked into grounding block:\n{s}");
    }
}
