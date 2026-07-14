//! Resolve table-compiler graph wiring from an ontology overlay.

use crate::graph::ontology::Ontology;

/// Stable relation names the table compiler knows how to wire (capability contract).
pub const EDGE_CONSTRAINED_BY: &str = "CONSTRAINED_BY";
pub const EDGE_IF_FIELD: &str = "IF_FIELD";
pub const EDGE_IF_VALUE: &str = "IF_VALUE";
pub const EDGE_HAS_CONSTRAINT: &str = "HAS_CONSTRAINT";
pub const EDGE_HAS_EXPRESSION: &str = "HAS_EXPRESSION";
pub const EDGE_HAS_PATTERN: &str = "HAS_PATTERN";
pub const EDGE_HAS_MIN: &str = "HAS_MIN";
pub const EDGE_HAS_MAX: &str = "HAS_MAX";

pub const SUPPORTED_COMPILER_EDGES: &[&str] = &[
    EDGE_CONSTRAINED_BY,
    EDGE_IF_FIELD,
    EDGE_IF_VALUE,
    EDGE_HAS_CONSTRAINT,
    EDGE_HAS_EXPRESSION,
    EDGE_HAS_PATTERN,
    EDGE_HAS_MIN,
    EDGE_HAS_MAX,
];

/// Entity and constraint types resolved from `ontology.toml` relations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TablesCompileWiring {
    pub constrained_by: String,
    pub if_field: String,
    pub if_value: String,
    pub has_constraint: String,
    pub parameter_entity: String,
    pub literal_entity: String,
    pub enum_constraint: String,
    pub conditional_constraint: String,
}

impl TablesCompileWiring {
    /// Node types the compiler materializes and may replace per document on recompile
    /// (parameter entity + constraint payloads + literals).
    pub fn compile_layer_node_types(&self, ont: &Ontology) -> Vec<String> {
        use std::collections::BTreeSet;
        let mut types = BTreeSet::new();
        types.insert(self.parameter_entity.clone());
        types.insert(self.literal_entity.clone());
        types.insert(self.enum_constraint.clone());
        types.insert(self.conditional_constraint.clone());
        for name in ont.constraint_types().keys() {
            types.insert(name.clone());
        }
        types.into_iter().collect()
    }

    /// Relation names the compiler writes, intersected with what the ontology declares.
    pub fn compile_layer_edge_types(&self, ont: &Ontology) -> Vec<String> {
        SUPPORTED_COMPILER_EDGES
            .iter()
            .filter(|name| ont.raw_relations().contains_key(**name))
            .map(|s| (*s).to_string())
            .collect()
    }

    /// Resolve wiring from declared relations. Fails with an actionable message when the
    /// overlay is missing a required edge or endpoint type.
    pub fn resolve(ont: &Ontology) -> Result<Self, String> {
        let constrained_by = EDGE_CONSTRAINED_BY.to_string();
        let if_field = EDGE_IF_FIELD.to_string();
        let if_value = EDGE_IF_VALUE.to_string();
        let has_constraint = EDGE_HAS_CONSTRAINT.to_string();

        for name in SUPPORTED_COMPILER_EDGES {
            if !ont.raw_relations().contains_key(*name)
                && matches!(
                    *name,
                    EDGE_CONSTRAINED_BY | EDGE_IF_FIELD | EDGE_IF_VALUE | EDGE_HAS_CONSTRAINT
                )
            {
                return Err(format!(
                    "tables compile: ontology missing required relation [{relations}.{name}]",
                    relations = "relations",
                    name = name
                ));
            }
        }

        let (param_from, constrained_to) = ont.endpoint_types(&constrained_by);
        let parameter_entity = single_endpoint(&param_from, "CONSTRAINED_BY", "from")?;
        let conditional_constraint = single_type_in(
            &constrained_to,
            &["Conditional"],
            "CONSTRAINED_BY",
            "to (Conditional)",
        )?;
        let enum_constraint = pick_enum_type(ont, &constrained_to)?;

        let (cond_from, _) = ont.endpoint_types(&if_field);
        let (_, lit_to) = ont.endpoint_types(&if_field);
        let literal_entity = single_endpoint(&lit_to, "IF_FIELD", "to")?;
        let cond_from_if = single_endpoint(&cond_from, "IF_FIELD", "from")?;
        if cond_from_if != conditional_constraint {
            return Err(format!(
                "tables compile: IF_FIELD.from ({cond_from_if}) != Conditional in CONSTRAINED_BY.to ({conditional_constraint})"
            ));
        }
        let (cond_from2, _) = ont.endpoint_types(&if_value);
        let (_, lit_to2) = ont.endpoint_types(&if_value);
        if single_endpoint(&lit_to2, "IF_VALUE", "to")? != literal_entity {
            return Err(
                "tables compile: IF_FIELD.to and IF_VALUE.to must agree on Literal entity".into(),
            );
        }
        if single_endpoint(&cond_from2, "IF_VALUE", "from")? != conditional_constraint {
            return Err("tables compile: IF_VALUE.from must be Conditional".into());
        }
        let (hc_from, hc_to) = ont.endpoint_types(&has_constraint);
        if single_endpoint(&hc_from, "HAS_CONSTRAINT", "from")? != conditional_constraint {
            return Err("tables compile: HAS_CONSTRAINT.from must be Conditional".into());
        }
        if !hc_to.iter().any(|t| t == &enum_constraint || t == "*") {
            return Err(format!(
                "tables compile: HAS_CONSTRAINT.to must allow {enum_constraint} (got {hc_to:?})"
            ));
        }

        Ok(Self {
            constrained_by,
            if_field,
            if_value,
            has_constraint,
            parameter_entity,
            literal_entity,
            enum_constraint,
            conditional_constraint,
        })
    }
}

fn single_endpoint(types: &[String], rel: &str, role: &str) -> Result<String, String> {
    if types.len() == 1 {
        Ok(types[0].clone())
    } else {
        Err(format!(
            "tables compile: relations.{rel} {role} must declare exactly one type (got {types:?})"
        ))
    }
}

fn single_type_in(
    types: &[String],
    want: &[&str],
    rel: &str,
    role: &str,
) -> Result<String, String> {
    for w in want {
        if types.iter().any(|t| t == w) {
            return Ok((*w).to_string());
        }
    }
    Err(format!(
        "tables compile: relations.{rel} {role} must include one of {want:?} (got {types:?})"
    ))
}

fn pick_enum_type(ont: &Ontology, constrained_to: &[String]) -> Result<String, String> {
    if constrained_to.iter().any(|t| t == "Enum") && ont.has_constraint_type("Enum") {
        return Ok("Enum".into());
    }
    for t in constrained_to {
        if t != "Conditional" && ont.has_constraint_type(t) {
            return Ok(t.clone());
        }
    }
    Err(
        "tables compile: CONSTRAINED_BY.to must include Enum (or another discrete constraint type)"
            .into(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ontology::Ontology;

    fn eval_ontology() -> Ontology {
        let s = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/eval/ontology-constraint.toml"
        ));
        Ontology::parse(s).unwrap()
    }

    #[test]
    fn wiring_resolves_from_eval_ontology() {
        let w = TablesCompileWiring::resolve(&eval_ontology()).unwrap();
        assert_eq!(w.parameter_entity, "Field");
        assert_eq!(w.literal_entity, "Literal");
        assert_eq!(w.enum_constraint, "Enum");
        assert_eq!(w.conditional_constraint, "Conditional");
    }

    #[test]
    fn compile_layer_includes_parameter_and_compiler_edges_not_mentions() {
        let ont = eval_ontology();
        let w = TablesCompileWiring::resolve(&ont).unwrap();
        let nodes = w.compile_layer_node_types(&ont);
        assert!(
            nodes.iter().any(|t| t == "Field"),
            "compiler works with Field: {nodes:?}"
        );
        assert!(nodes.iter().any(|t| t == "Enum"));
        let edges = w.compile_layer_edge_types(&ont);
        assert!(edges.iter().any(|e| e == "CONSTRAINED_BY"));
        assert!(!edges.iter().any(|e| e == "MENTIONS"));
    }
}
