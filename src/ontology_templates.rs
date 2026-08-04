//! Baked task-ontology presets: the embedded template set + a registry over it.

include!(concat!(env!("OUT_DIR"), "/ontology_templates.rs"));

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
}
