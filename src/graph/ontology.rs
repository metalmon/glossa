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
