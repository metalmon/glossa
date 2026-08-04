//! Baked task-ontology presets: the embedded template set + a registry over it.

include!(concat!(env!("OUT_DIR"), "/ontology_templates.rs"));

use crate::graph::ontology::Ontology;

/// Catalog entry for one preset, derived from its `[meta]`.
#[derive(Debug, Clone)]
pub struct TemplateInfo {
    pub name: String,
    pub family: Option<String>,
    pub tier: u8,
    pub description: Option<String>,
    pub aliases: Vec<String>,
}

/// Parse every embedded preset's `[meta]` into a catalog.
pub fn catalog() -> Vec<TemplateInfo> {
    TEMPLATES
        .iter()
        .map(|(name, toml)| {
            let o = Ontology::parse(toml).unwrap_or_default();
            let m = o.meta();
            TemplateInfo {
                name: (*name).to_string(),
                family: m.family.clone(),
                tier: m.tier,
                description: m.description.clone(),
                aliases: m.aliases.clone(),
            }
        })
        .collect()
}

/// Resolve a preset name OR alias to its canonical name. Trims whitespace.
pub fn resolve(query: &str) -> Option<String> {
    let q = query.trim();
    if raw(q).is_some() {
        return Some(q.to_string());
    }
    catalog()
        .into_iter()
        .find(|t| t.aliases.iter().any(|a| a == q))
        .map(|t| t.name)
}

/// Raw TOML for a preset by its exact name (filename stem), not an alias.
pub fn raw(name: &str) -> Option<&'static str> {
    TEMPLATES.iter().find(|(n, _)| *n == name).map(|(_, t)| *t)
}

/// All preset names in sorted order.
pub fn names() -> impl Iterator<Item = &'static str> {
    TEMPLATES.iter().map(|(n, _)| *n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ontology::Ontology;

    #[test]
    fn seeds_are_embedded_and_parse() {
        for seed in ["support", "compliance", "data-privacy"] {
            let toml = raw(seed).unwrap_or_else(|| panic!("missing preset {seed}"));
            Ontology::parse(toml).unwrap_or_else(|e| panic!("{seed} parse: {e}"));
        }
        assert!(names().count() >= 3);
    }

    #[test]
    fn resolve_by_name_and_alias() {
        assert_eq!(resolve("support").as_deref(), Some("support"));
        assert_eq!(resolve("troubleshooting").as_deref(), Some("support")); // alias
        assert_eq!(resolve("normocontrol").as_deref(), Some("compliance")); // alias
        assert_eq!(resolve("  compliance "), Some("compliance".to_string())); // trims
        assert!(resolve("nope-not-real").is_none());

        let cat = catalog();
        let support = cat.iter().find(|t| t.name == "support").unwrap();
        assert_eq!(support.tier, 2);
        assert_eq!(support.family.as_deref(), Some("causal"));
    }
}
