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

/// One valid reasoning shape: an anchor node type plus the ordered relations leading from it
/// (e.g. anchor `Symptom`, relations `[CAUSED_BY, RESOLVED_BY]`). A node survives hygiene if it
/// lies on a COMPLETE instance of ANY declared spine — so distinct case shapes (causal
/// troubleshooting vs informational how-to) coexist without one pruning the other.
#[derive(Debug, Deserialize, Default, Clone)]
struct RawSpine {
    #[serde(default)]
    anchor: String,
    #[serde(default)]
    relations: Vec<String>,
}

/// Public form of a reasoning spine (see `RawSpine`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spine {
    pub anchor: String,
    pub relations: Vec<String>,
}

/// The `[reasoning]` overlay: domain-specific graph rules, kept OUT of the Rust code so the
/// engine stays domain-agnostic (support is one overlay among many). All keys optional.
#[derive(Debug, Deserialize, Default, Clone)]
struct RawReasoning {
    /// Valid reasoning shapes; a node survives hygiene if on a complete instance of any.
    #[serde(default)]
    spines: Vec<RawSpine>,
    /// Transitive-closure composition rules, each `[a, b, result]`.
    #[serde(default)]
    closure: Vec<Vec<String>>,
    /// Anchor edge from a reasoning node to the structural layer. Defaults to "MENTIONS".
    #[serde(default)]
    mentions: Option<String>,
    /// Override of the structural (never-reasoning) types. Defaults to the core nodes.
    #[serde(default)]
    structural: Vec<String>,
}

#[derive(Debug, Deserialize, Default)]
struct RawOntology {
    #[serde(default)]
    entities: BTreeMap<String, toml::Value>,
    #[serde(default)]
    relations: BTreeMap<String, RawRelation>,
    #[serde(default)]
    validation: RawValidation,
    #[serde(default)]
    reasoning: RawReasoning,
}

#[derive(Debug, Default)]
pub struct Ontology {
    entity_types: std::collections::BTreeSet<String>,
    relations: BTreeMap<String, RawRelation>,
    strict: bool,
    reasoning: RawReasoning,
}

impl Ontology {
    pub fn parse(toml_str: &str) -> anyhow::Result<Ontology> {
        let raw: RawOntology = toml::from_str(toml_str)?;
        Ok(Ontology {
            entity_types: raw.entities.keys().cloned().collect(),
            relations: raw.relations,
            strict: raw.validation.strict,
            reasoning: raw.reasoning,
        })
    }

    /// The reasoning spines — the valid shapes a node must lie on a complete instance of to
    /// survive the hygiene prune. Malformed entries (empty anchor or relations) are dropped.
    /// Empty when unset.
    pub fn spines(&self) -> Vec<Spine> {
        self.reasoning
            .spines
            .iter()
            .filter(|s| !s.anchor.is_empty() && !s.relations.is_empty())
            .map(|s| Spine { anchor: s.anchor.clone(), relations: s.relations.clone() })
            .collect()
    }

    /// Transitive-closure composition rules as `(a, b, result)` triples. Malformed inner vecs
    /// (length != 3) are skipped — that is a config error, not graph data.
    pub fn closure_rules(&self) -> Vec<(String, String, String)> {
        self.reasoning
            .closure
            .iter()
            .filter(|r| r.len() == 3)
            .map(|r| (r[0].clone(), r[1].clone(), r[2].clone()))
            .collect()
    }

    /// The anchor edge type from a reasoning node to the structural layer. "MENTIONS" when unset.
    pub fn mentions(&self) -> &str {
        self.reasoning.mentions.as_deref().unwrap_or("MENTIONS")
    }

    /// The structural (never-reasoning) types. Declared override, else the core nodes.
    pub fn structural(&self) -> Vec<String> {
        if self.reasoning.structural.is_empty() {
            CORE_NODES.iter().map(|s| s.to_string()).collect()
        } else {
            self.reasoning.structural.clone()
        }
    }

    /// Entity types that are endpoints (`from` or `to`) of any relation named in any spine.
    /// Used by the hygiene pass to tell a "doomed" reasoning node (a spine-type node not on a
    /// complete chain) from an auxiliary one. Empty when there are no spines.
    pub fn spine_types(&self) -> std::collections::HashSet<String> {
        let mut out = std::collections::HashSet::new();
        for sp in &self.reasoning.spines {
            for rel in &sp.relations {
                if let Some(r) = self.relations.get(rel) {
                    out.extend(r.from.iter().cloned());
                    out.extend(r.to.iter().cloned());
                }
            }
        }
        out
    }

    /// The distinct relation types that make up the reasoning spines, in first-seen order
    /// (e.g. `[CAUSED_BY, RESOLVED_BY]`). `glossary` walks these to render a reasoning node's
    /// full chain (Symptom→Cause→Resolution, Task→Resolution) in a single call. Empty when
    /// there are no spines — keeps the chain walk ontology-driven, never hard-coded.
    pub fn spine_relations(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for sp in &self.reasoning.spines {
            if sp.anchor.is_empty() || sp.relations.is_empty() {
                continue;
            }
            for rel in &sp.relations {
                if !out.iter().any(|r| r == rel) {
                    out.push(rel.clone());
                }
            }
        }
        out
    }

    pub fn load_or_default(root: &std::path::Path) -> Ontology {
        let p = root.join(".glossa").join("ontology.toml");
        match std::fs::read_to_string(&p) {
            Ok(s) => Ontology::parse(&s).unwrap_or_default(),
            Err(_) => Ontology::default(),
        }
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

    const REASONING_TOML: &str = r#"
[reasoning]
spines = [
  { anchor = "Symptom", relations = ["CAUSED_BY", "RESOLVED_BY"] },
  { anchor = "Task", relations = ["RESOLVED_BY"] },
]
mentions = "MENTIONS"
closure = [["CAUSED_BY", "RESOLVED_BY", "RESOLVED_BY"], ["A", "B"]]
structural = ["Document", "Section"]
"#;

    #[test]
    fn reasoning_section_parses() {
        let o = Ontology::parse(REASONING_TOML).unwrap();
        let spines = o.spines();
        assert_eq!(spines.len(), 2);
        assert_eq!(spines[0].anchor, "Symptom");
        assert_eq!(spines[0].relations, vec!["CAUSED_BY".to_string(), "RESOLVED_BY".to_string()]);
        assert_eq!(spines[1].anchor, "Task");
        assert_eq!(spines[1].relations, vec!["RESOLVED_BY".to_string()]);
        assert_eq!(o.mentions(), "MENTIONS");
        assert_eq!(o.structural(), vec!["Document".to_string(), "Section".to_string()]);
    }

    #[test]
    fn closure_rules_skip_malformed() {
        // the `["A","B"]` inner (len 2) must be dropped; the valid triple kept
        let o = Ontology::parse(REASONING_TOML).unwrap();
        assert_eq!(
            o.closure_rules(),
            vec![("CAUSED_BY".into(), "RESOLVED_BY".into(), "RESOLVED_BY".into())]
        );
    }

    #[test]
    fn spine_types_from_relations() {
        let toml = r#"
[relations.CAUSED_BY]
from = ["Symptom"]
to = ["Cause"]
[relations.RESOLVED_BY]
from = ["Symptom", "Cause"]
to = ["Resolution"]
[reasoning]
spines = [{ anchor = "Symptom", relations = ["CAUSED_BY", "RESOLVED_BY"] }]
"#;
        let o = Ontology::parse(toml).unwrap();
        let types = o.spine_types();
        assert_eq!(
            types,
            ["Symptom", "Cause", "Resolution"].into_iter().map(String::from).collect()
        );
        // no spine → no types
        assert!(Ontology::parse(TOML).unwrap().spine_types().is_empty());
    }

    #[test]
    fn spine_relations_deduped_in_order() {
        // union across both spines, first-seen order, RESOLVED_BY not repeated
        let o = Ontology::parse(REASONING_TOML).unwrap();
        assert_eq!(o.spine_relations(), vec!["CAUSED_BY".to_string(), "RESOLVED_BY".to_string()]);
        // no spine → no relations
        assert!(Ontology::parse(TOML).unwrap().spine_relations().is_empty());
    }

    #[test]
    fn reasoning_absent_yields_defaults() {
        let o = Ontology::parse(TOML).unwrap(); // TOML has no [reasoning]
        assert!(o.spines().is_empty());
        assert!(o.closure_rules().is_empty());
        assert_eq!(o.mentions(), "MENTIONS");
        assert_eq!(
            o.structural(),
            vec!["Document", "Section", "Term", "Topic"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn load_or_default_reads_file_else_default() {
        let dir = tempfile::tempdir().unwrap();
        // missing file → default (permissive)
        let o = Ontology::load_or_default(dir.path());
        assert!(o.validate_node("Anything").is_ok());
        // present file → parsed (strict)
        std::fs::create_dir_all(dir.path().join(".glossa")).unwrap();
        std::fs::write(dir.path().join(".glossa/ontology.toml"),
            "[entities.Person]\nprops=[]\n[validation]\nstrict=true\n").unwrap();
        let o2 = Ontology::load_or_default(dir.path());
        assert!(o2.validate_node("Person").is_ok());
        assert!(o2.validate_node("Alien").is_err());
    }
}
