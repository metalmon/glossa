use serde::Deserialize;
use std::collections::BTreeMap;

const CORE_NODES: &[&str] = crate::graph::STRUCTURAL_NODES;
const CORE_EDGES: &[&str] = &["CONTAINS", "MENTIONS", "CO_OCCURS", "NEXT", "PREV"];

#[derive(Debug, Deserialize, Default, Clone)]
pub struct RawRelation {
    #[serde(default)]
    pub from: Vec<String>,
    #[serde(default)]
    pub to: Vec<String>,
    /// Optional human/agent-facing note on what the relation means and how to use
    /// it — surfaced by `get_ontology` so the model builds the right shape.
    #[serde(default)]
    pub description: Option<String>,
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
    /// Override of the structural (never-reasoning) types. Defaults to the core nodes.
    #[serde(default)]
    structural: Vec<String>,
}

/// A type of constraint that can be applied to fields (e.g. Range, Regex, Enum).
/// Defines which edge types a constraint of this type expects as parameters.
#[derive(Debug, Deserialize, Default, Clone)]
struct RawConstraintType {
    #[serde(default)]
    params: Vec<String>,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug, Default)]
pub struct ConstraintType {
    pub params: Vec<String>,
}

/// A documented graph-building pattern. `example` is for humans reading the TOML;
/// agent-facing export omits it (models copy examples when uncertain).
#[derive(Debug, Deserialize, Default, Clone)]
struct RawPattern {
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    example: Option<String>,
}

/// Parsed `[meta]` overlay — domain label and optional prose for operators/agents.
#[derive(Debug, Deserialize, Default, Clone)]
struct RawMeta {
    #[serde(default)]
    domain: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    note: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Pattern {
    pub description: String,
    /// Human-only illustration; not exported via `get_ontology`.
    pub example: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Meta {
    pub domain: Option<String>,
    pub description: Option<String>,
    pub note: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct RawOntology {
    #[serde(default)]
    meta: RawMeta,
    #[serde(default)]
    entities: BTreeMap<String, toml::Value>,
    #[serde(default)]
    relations: BTreeMap<String, RawRelation>,
    #[serde(default)]
    validation: RawValidation,
    #[serde(default)]
    reasoning: RawReasoning,
    #[serde(default)]
    constraint_types: BTreeMap<String, RawConstraintType>,
    #[serde(default)]
    patterns: BTreeMap<String, RawPattern>,
}

#[derive(Debug, Default)]
pub struct Ontology {
    meta: Meta,
    entity_types: std::collections::BTreeSet<String>,
    /// Optional per-entity id prefix (e.g. Symptom → "sym"). Unset types fall back to lowercase type name.
    id_prefixes: BTreeMap<String, String>,
    relations: BTreeMap<String, RawRelation>,
    strict: bool,
    reasoning: RawReasoning,
    constraint_types: BTreeMap<String, ConstraintType>,
    patterns: BTreeMap<String, Pattern>,
    /// How-to notes keyed by entity / relation / constraint-type name, surfaced by
    /// `get_ontology`. Optional per type; empty when the ontology declares none.
    descriptions: BTreeMap<String, String>,
}

fn entity_id_prefix(v: &toml::Value) -> Option<String> {
    v.get("id_prefix").and_then(|p| p.as_str()).map(str::to_string)
}

impl Ontology {
    pub fn parse(toml_str: &str) -> anyhow::Result<Ontology> {
        let raw: RawOntology = toml::from_str(toml_str)?;
        let id_prefixes = raw
            .entities
            .iter()
            .filter_map(|(name, v)| entity_id_prefix(v).map(|p| (name.clone(), p)))
            .collect();
        // Collect how-to notes from entities, relations, and constraint types into
        // one map keyed by name (they share a type namespace in the graph).
        let mut descriptions: BTreeMap<String, String> = BTreeMap::new();
        for (name, v) in &raw.entities {
            if let Some(d) = v.get("description").and_then(|d| d.as_str()) {
                descriptions.insert(name.clone(), d.to_string());
            }
        }
        for (name, r) in &raw.relations {
            if let Some(d) = &r.description {
                descriptions.insert(name.clone(), d.clone());
            }
        }
        for (name, c) in &raw.constraint_types {
            if let Some(d) = &c.description {
                descriptions.insert(name.clone(), d.clone());
            }
        }
        let patterns = raw
            .patterns
            .into_iter()
            .filter_map(|(name, p)| {
                let description = p.description.filter(|d| !d.is_empty())?;
                Some((
                    name,
                    Pattern {
                        description,
                        example: p.example.filter(|e| !e.is_empty()),
                    },
                ))
            })
            .collect();
        Ok(Ontology {
            meta: Meta {
                domain: raw.meta.domain.filter(|d| !d.is_empty()),
                description: raw.meta.description.filter(|d| !d.is_empty()),
                note: raw.meta.note.filter(|d| !d.is_empty()),
            },
            entity_types: raw.entities.keys().cloned().collect(),
            id_prefixes,
            relations: raw.relations,
            strict: raw.validation.strict,
            reasoning: raw.reasoning,
            constraint_types: raw.constraint_types.into_iter().map(|(k, v)| {
                (k, ConstraintType { params: v.params })
            }).collect(),
            patterns,
            descriptions,
        })
    }

    /// How-to note for an entity / relation / constraint-type name, if the
    /// ontology declares one. Surfaced by `get_ontology`.
    pub fn description(&self, name: &str) -> Option<&str> {
        self.descriptions.get(name).map(|s| s.as_str())
    }

    pub fn meta(&self) -> &Meta {
        &self.meta
    }

    pub fn patterns(&self) -> &BTreeMap<String, Pattern> {
        &self.patterns
    }

    /// Prefix used when deriving a node id for `node_type` (`id_for` / label sanitize).
    pub fn id_abbrev(&self, node_type: &str) -> String {
        self.id_prefixes
            .get(node_type)
            .cloned()
            .unwrap_or_else(|| node_type.to_lowercase())
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

    /// The set of declared entity type names.
    pub fn entity_types(&self) -> &std::collections::BTreeSet<String> {
        &self.entity_types
    }

    /// The raw relations map (primarily for JSON export).
    pub fn raw_relations(&self) -> &BTreeMap<String, RawRelation> {
        &self.relations
    }

    /// Whether validation is strict (unknown entity types rejected).
    pub fn strict(&self) -> bool {
        self.strict
    }

    /// Get the map of defined constraint types (e.g. "Range" → {params: ["min", "max"]}).
    pub fn constraint_types(&self) -> &BTreeMap<String, ConstraintType> {
        &self.constraint_types
    }

    /// Check if a constraint type is defined in the ontology.
    pub fn has_constraint_type(&self, name: &str) -> bool {
        self.constraint_types.contains_key(name)
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
        // Constraint types (Range, Enum, Formula, …) are node types too: the agent
        // instantiates them and links Field → CONSTRAINED_BY → <constraint node>.
        if CORE_NODES.contains(&node_type)
            || self.entity_types.contains(node_type)
            || self.constraint_types.contains_key(node_type)
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
                    let hint = edge_validation_hint(edge_type, from_type, to_type);
                    Err(format!(
                        "relation '{edge_type}' endpoints {from_type}->{to_type} not allowed — \
                         check get_ontology → relations.{edge_type} for allowed from/to types{hint}"
                    ))
                }
            }
            None if self.strict => Err(format!("unknown relation '{edge_type}' (strict)")),
            None => Ok(()),
        }
    }

    /// Allowed (from-types, to-types) for a relation — empty vecs when the relation is
    /// unknown or leaves an end unconstrained. Used to disambiguate an edge endpoint when
    /// a Field and its Enum share a label (CONSTRAINED_BY `from` wants the Field, `to` the
    /// Enum), so the two do not collide onto the same node.
    pub fn endpoint_types(&self, edge_type: &str) -> (Vec<String>, Vec<String>) {
        self.relations
            .get(edge_type)
            .map(|r| (r.from.clone(), r.to.clone()))
            .unwrap_or_default()
    }
}

/// Extra guidance when a common mistake is detected (IF_FIELD/IF_VALUE → Field).
fn edge_validation_hint(edge_type: &str, from_type: &str, to_type: &str) -> &'static str {
    match (edge_type, from_type, to_type) {
        ("IF_FIELD", "Conditional", "Field") => {
            " — IF_FIELD must go to a Literal node (label = trigger field name as text), not a Field"
        }
        ("IF_VALUE", "Conditional", "Field") => {
            " — IF_VALUE must go to a Literal node (label = trigger value as text), not a Field"
        }
        ("HAS_MIN" | "HAS_MAX" | "HAS_PATTERN" | "HAS_EXPRESSION", _, "Field") => {
            " — parameter edges go to Literal nodes, not Field"
        }
        _ => "",
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
[constraint_types.Range]
params = ["min", "max"]
[constraint_types.Regex]
params = ["pattern"]
"#;

    #[test]
    fn meta_and_patterns_parsed() {
        let toml = r#"
[meta]
domain = "gost-regulatory"
description = "Test."

[patterns.foo]
description = "A pattern."
example = "human only"

[entities.X]
description = "entity"
"#;
        let o = Ontology::parse(toml).unwrap();
        assert_eq!(o.meta().domain.as_deref(), Some("gost-regulatory"));
        let p = o.patterns().get("foo").unwrap();
        assert_eq!(p.description, "A pattern.");
        assert_eq!(p.example.as_deref(), Some("human only"));
    }

    #[test]
    fn core_types_always_allowed() {
        let o = Ontology::parse(TOML).unwrap();
        assert!(o.validate_node("Document").is_ok());
        assert!(o.validate_edge("CONTAINS", "Document", "Section").is_ok());
    }

    #[test]
    fn constraint_types_parsed() {
        let o = Ontology::parse(TOML).unwrap();
        assert!(o.has_constraint_type("Range"));
        assert!(o.has_constraint_type("Regex"));
        assert!(!o.has_constraint_type("Enum"));
        let range = o.constraint_types().get("Range").unwrap();
        assert_eq!(range.params, vec!["min", "max"]);
    }

    #[test]
    fn constraint_types_absent_by_default() {
        let o = Ontology::default();
        assert!(o.constraint_types().is_empty());
    }

    #[test]
    fn constraint_types_are_valid_node_types_under_strict() {
        let o = Ontology::parse(TOML).unwrap();
        assert!(o.validate_node("Range").is_ok());
        assert!(o.validate_node("Regex").is_ok());
        assert!(o.validate_node("Enum").is_err()); // not declared in this ontology
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
        assert_eq!(
            o.structural(),
            vec!["Document", "Section", "Term", "Topic"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn id_prefix_parsed_and_fallback() {
        let toml = r#"
[entities.Symptom]
id_prefix = "sym"
props = []
[entities.Parameter]
props = []
"#;
        let o = Ontology::parse(toml).unwrap();
        assert_eq!(o.id_abbrev("Symptom"), "sym");
        assert_eq!(o.id_abbrev("Parameter"), "parameter");
        assert_eq!(o.id_abbrev("Unknown"), "unknown");
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
