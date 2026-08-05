//! Baked task-ontology presets: the embedded template set + a registry over it.

include!(concat!(env!("OUT_DIR"), "/ontology_templates.rs"));

use crate::graph::ontology::Ontology;
use std::sync::LazyLock;

/// Catalog entry for one preset, derived from its `[meta]`.
#[derive(Debug, Clone)]
pub struct TemplateInfo {
    pub name: String,
    pub family: Option<String>,
    pub tier: u8,
    pub description: Option<String>,
    pub aliases: Vec<String>,
}

/// The catalog, parsed once from the embedded `[meta]` blocks. The templates are
/// baked at build time and never change at runtime, so a single parse suffices —
/// `resolve`/`nearest`/`suggest`/`catalog` all read this instead of re-parsing.
static CATALOG: LazyLock<Vec<TemplateInfo>> = LazyLock::new(|| {
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
});

/// The preset catalog (parsed once; see [`CATALOG`]).
pub fn catalog() -> Vec<TemplateInfo> {
    CATALOG.clone()
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
        .filter(|t| t.chars().count() > 2)
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

/// Outcome of a materialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Written {
    Created,
    Kept,
    Overwritten,
}

/// Write a preset to `<root>/.glossa/ontology.toml`. On an existing file: `Kept`
/// unless `force`. Unknown name → error listing the closest presets.
pub fn write_template(
    root: &std::path::Path,
    name_or_alias: &str,
    force: bool,
) -> anyhow::Result<Written> {
    let name = resolve(name_or_alias).ok_or_else(|| {
        let near = nearest(name_or_alias, 3).join(", ");
        anyhow::anyhow!("unknown ontology preset '{name_or_alias}' — did you mean: {near}? (kb ontology list)")
    })?;
    let toml = raw(&name).expect("resolved name is embedded");

    let glossa_dir = root.join(".glossa");
    let target = glossa_dir.join("ontology.toml");
    let existed = target.exists();
    if existed && !force {
        return Ok(Written::Kept);
    }
    std::fs::create_dir_all(&glossa_dir)?;
    std::fs::write(&target, toml)?;
    Ok(if existed { Written::Overwritten } else { Written::Created })
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

    #[test]
    fn write_template_create_keep_force() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let target = root.join(".glossa").join("ontology.toml");

        // create
        assert!(matches!(write_template(root, "support", false).unwrap(), Written::Created));
        assert!(target.exists());
        let written = std::fs::read_to_string(&target).unwrap();
        Ontology::parse(&written).unwrap(); // valid ontology on disk

        // keep (no force)
        assert!(matches!(write_template(root, "compliance", false).unwrap(), Written::Kept));
        assert!(std::fs::read_to_string(&target).unwrap().contains("Symptom")); // still support

        // force overwrite (via alias)
        assert!(matches!(write_template(root, "normocontrol", true).unwrap(), Written::Overwritten));
        assert!(std::fs::read_to_string(&target).unwrap().contains("NormativeRequirement"));

        // unknown → error whose message suggests the closest preset(s)
        let err = write_template(root, "compliancee", false).unwrap_err().to_string();
        assert!(err.contains("compliance"), "error should suggest a candidate: {err}");
        assert!(err.contains("kb ontology list"), "error should point at the list: {err}");
    }

    #[test]
    fn resolve_and_raw_stay_in_sync() {
        // Every catalog name and alias must resolve to a canonical name that `raw`
        // can materialize — this is the invariant `write_template`/`show` rely on
        // when they `expect`/`unwrap` after `resolve`.
        for t in catalog() {
            assert_eq!(resolve(&t.name).as_deref(), Some(t.name.as_str()));
            assert!(raw(&t.name).is_some(), "{}: raw() missing for a catalog name", t.name);
            for a in &t.aliases {
                assert_eq!(
                    resolve(a).as_deref(),
                    Some(t.name.as_str()),
                    "alias {a} should resolve to {}",
                    t.name
                );
                let canon = resolve(a).unwrap();
                assert!(raw(&canon).is_some(), "alias {a} resolves to a non-embedded name");
            }
        }
    }

    #[test]
    fn all_presets_well_formed() {
        // Core structural node types, always valid as a relation endpoint even though
        // they are never declared as `[entities.*]` in a preset. Mirrors
        // `crate::graph::STRUCTURAL_NODES` (kept as a literal here so this test doesn't
        // silently pass if that constant's contents ever drift).
        const CORE_NODES: &[&str] = &["Document", "Section", "Term", "Topic"];

        let mut seen_names = std::collections::HashSet::new();
        let mut seen_aliases = std::collections::HashSet::new();
        for (name, toml) in TEMPLATES {
            let o = Ontology::parse(toml)
                .unwrap_or_else(|e| panic!("{name} does not parse: {e}"));
            let m = o.meta();
            assert!(m.description.is_some(), "{name}: [meta].description required");
            assert!(m.family.is_some(), "{name}: [meta].family required");
            assert!(m.tier == 1 || m.tier == 2, "{name}: tier must be 1 or 2");
            assert!(seen_names.insert(name.to_string()), "duplicate preset name {name}");
            for a in &m.aliases {
                assert!(seen_aliases.insert(a.clone()), "{name}: duplicate alias {a}");
                assert!(raw(a).is_none(), "{name}: alias {a} collides with a preset name");
            }
            assert!(o.strict(), "{name}: presets must set strict = true");

            // Cross-reference integrity: `Ontology::parse` does NOT validate that a
            // relation endpoint names a declared type, or that a spine's relations
            // exist — those are only enforced at `graph_upsert` runtime. Assert both
            // here so a dangling reference in a preset fails the build, not a
            // production run.
            let entity_types = o.entity_types();
            let constraint_types = o.constraint_types();
            let relations = o.raw_relations();
            let endpoint_ok = |t: &str| {
                t == "*"
                    || CORE_NODES.contains(&t)
                    || entity_types.contains(t)
                    || constraint_types.contains_key(t)
            };
            for (rel_name, r) in relations {
                for t in &r.from {
                    assert!(
                        endpoint_ok(t),
                        "{name}: relations.{rel_name}.from references undeclared type '{t}'"
                    );
                }
                for t in &r.to {
                    assert!(
                        endpoint_ok(t),
                        "{name}: relations.{rel_name}.to references undeclared type '{t}'"
                    );
                }
            }
            for spine in o.spines() {
                for rel_name in &spine.relations {
                    assert!(
                        relations.contains_key(rel_name),
                        "{name}: reasoning spine (anchor {}) references undeclared relation '{rel_name}'",
                        spine.anchor
                    );
                }
            }
        }
        assert_eq!(TEMPLATES.len(), 26, "expected 26 presets");
    }

    #[test]
    fn grounding_coverage_marks_extracted_exempts_synthesized() {
        use crate::graph::ontology::Ontology;
        let g = |preset: &str, ty: &str| {
            Ontology::parse(raw(preset).unwrap()).unwrap().requires_grounding(ty)
        };
        // concrete extracted nouns are grounded
        assert!(g("qa-inspection", "Parameter"));
        assert!(g("compliance", "Field"));
        assert!(g("vendor", "Contract"));
        assert!(g("data-privacy", "DataAsset"));
        assert!(g("support", "Resolution"));
        // synthesized anchors + abstract carriers are NOT
        assert!(!g("support", "Symptom"));
        assert!(!g("support", "Task"));
        assert!(!g("compliance", "Literal"));
        assert!(!g("customer-journey", "PainPoint"));
        // grounding consolidated to one node per preset: siblings are ungrounded
        assert!(!g("vendor", "Service"));
        assert!(!g("data-privacy", "Processor"));
        // reg-change: the validity window lives on Requirement, not Standard
        assert!(!Ontology::parse(raw("reg-change").unwrap()).unwrap().requires_validity("Standard"));
        assert!(Ontology::parse(raw("reg-change").unwrap()).unwrap().requires_validity("Requirement"));
    }

    #[test]
    fn presets_are_thin_reasoning_skeletons() {
        use crate::graph::ontology::Ontology;
        let ground_count = |p: &str| {
            let o = Ontology::parse(raw(p).unwrap()).unwrap();
            o.entity_types().iter().filter(|t| o.requires_grounding(t)).count()
        };
        let has_entity = |p: &str, t: &str| {
            Ontology::parse(raw(p).unwrap()).unwrap().entity_types().contains(t)
        };
        // exactly one grounded node per preset (spot a representative set)
        for p in ["support","compliance","tender","certification","audit","fmea",
                  "risk-register","traceability","sop","policy","customer-journey",
                  "decision-log","faq","contract"] {
            assert_eq!(ground_count(p), 1, "{p}: exactly one requires_grounding");
        }
        // Evidence node is gone everywhere
        for p in ["compliance","tender","certification","audit"] {
            assert!(!has_entity(p, "Evidence"), "{p}: Evidence must be cut");
        }
        // context nouns cut
        assert!(!has_entity("support", "Component"));
        assert!(!has_entity("support", "Parameter"));
        assert!(!has_entity("sop", "Tool"));
        assert!(!has_entity("risk-register", "Owner"));
        // the grounded node is the right one
        assert!(Ontology::parse(raw("support").unwrap()).unwrap().requires_grounding("Resolution"));
        assert!(Ontology::parse(raw("audit").unwrap()).unwrap().requires_grounding("Control"));

        // constraint/flat presets: grounding consolidated to exactly one node each
        for p in ["qa-inspection","data-privacy","hr-compliance","access-governance",
                  "vendor","product-catalog","competency","org-roles","okr",
                  "project-schedule","timeline","reg-change"] {
            assert_eq!(ground_count(p), 1, "{p}: exactly one requires_grounding");
        }
        // the grounded node is the right one (spot check)
        assert!(Ontology::parse(raw("qa-inspection").unwrap()).unwrap().requires_grounding("Parameter"));
        assert!(Ontology::parse(raw("data-privacy").unwrap()).unwrap().requires_grounding("DataAsset"));
        assert!(Ontology::parse(raw("reg-change").unwrap()).unwrap().requires_grounding("Requirement"));
        // data-privacy is not over-cut: ROPA reasoning nodes remain (ungrounded)
        assert!(has_entity("data-privacy", "Purpose"));
        assert!(has_entity("data-privacy", "LegalBasis"));
        assert!(has_entity("data-privacy", "Processor"));
    }
}
