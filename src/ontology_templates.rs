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

/// Levenshtein edit distance (small hand-rolled; no dep).
fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Preset names closest to `query` (over names AND aliases, mapped to the canonical name).
pub fn nearest(query: &str, k: usize) -> Vec<String> {
    let q = query.trim().to_lowercase();
    let mut scored: Vec<(usize, String)> = catalog()
        .into_iter()
        .map(|t| {
            let mut d = edit_distance(&q, &t.name.to_lowercase());
            for a in &t.aliases {
                d = d.min(edit_distance(&q, &a.to_lowercase()));
            }
            (d, t.name)
        })
        .collect();
    scored.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    scored.into_iter().take(k).map(|(_, n)| n).collect()
}

fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 2)
        .map(str::to_string)
        .collect()
}

/// Rank presets by token overlap of `query` against each preset's
/// description + family + aliases + name. Non-zero scores, descending.
pub fn suggest(query: &str, k: usize) -> Vec<(String, usize)> {
    let q: std::collections::HashSet<String> = tokenize(query).into_iter().collect();
    let mut scored: Vec<(String, usize)> = catalog()
        .into_iter()
        .map(|t| {
            let mut bag = String::new();
            bag.push_str(&t.name);
            bag.push(' ');
            if let Some(f) = &t.family {
                bag.push_str(f);
                bag.push(' ');
            }
            if let Some(d) = &t.description {
                bag.push_str(d);
                bag.push(' ');
            }
            bag.push_str(&t.aliases.join(" "));
            let bag: std::collections::HashSet<String> = tokenize(&bag).into_iter().collect();
            let score = q.intersection(&bag).count();
            (t.name, score)
        })
        .filter(|(_, s)| *s > 0)
        .collect();
    scored.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    scored.into_iter().take(k).collect()
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

    #[test]
    fn nearest_and_suggest_rank() {
        // typo of a real name surfaces it first
        let near = nearest("complaince", 3);
        assert_eq!(near.first().map(String::as_str), Some("compliance"));

        // free text about privacy ranks data-privacy above support
        let s = suggest("we keep a register of personal data and its retention period", 3);
        assert_eq!(s.first().map(|(n, _)| n.as_str()), Some("data-privacy"));
        assert!(s.iter().all(|(_, score)| *score > 0));
    }
}
