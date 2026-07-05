//! Serialize an [`Ontology`] into the agent-facing `get_ontology` JSON payload.
//! Generic — no domain hardcoding. Built-in engine edges (e.g. `MENTIONS`) are
//! injected here because they are not declared in overlay TOML.

use crate::graph::ontology::Ontology;
use serde::Serialize;
use serde_json::{json, Value};

const MENTIONS_DESCRIPTION: &str = "Cite where a fact was read: from a node to a document section \
written as path#n, where path is the indexed document and n is the chunk number from a \
search/grep/read result for it.";

#[derive(Serialize)]
struct RelationExport {
    from: Vec<String>,
    to: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    built_in: bool,
}

#[derive(Serialize)]
struct ParameterExport {
    #[serde(skip_serializing_if = "Option::is_none")]
    id_prefix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
}

#[derive(Serialize)]
struct ConstraintExport {
    params: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
}

#[derive(Serialize)]
struct PatternExport {
    name: String,
    description: String,
}

#[derive(Serialize)]
struct MetaExport {
    #[serde(skip_serializing_if = "Option::is_none")]
    domain: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    note: Option<String>,
}

/// Build the `get_ontology` JSON value from a parsed ontology overlay.
pub fn export_json(ont: &Ontology) -> Value {
    let meta = ont.meta();
    let meta_export = MetaExport {
        domain: meta.domain.clone(),
        description: meta.description.clone(),
        note: meta.note.clone(),
    };

    let mut parameters = serde_json::Map::new();
    for name in ont.entity_types() {
        parameters.insert(
            name.clone(),
            json!(ParameterExport {
                id_prefix: Some(ont.id_abbrev(name)),
                description: ont.description(name).map(str::to_string),
            }),
        );
    }

    let mut constraints = serde_json::Map::new();
    for (name, ct) in ont.constraint_types() {
        constraints.insert(
            name.clone(),
            json!(ConstraintExport {
                params: ct.params.clone(),
                description: ont.description(name).map(str::to_string),
            }),
        );
    }

    let mut relations = serde_json::Map::new();
    for (name, r) in ont.raw_relations() {
        relations.insert(
            name.clone(),
            json!(RelationExport {
                from: r.from.clone(),
                to: r.to.clone(),
                description: ont.description(name).map(str::to_string),
                built_in: false,
            }),
        );
    }
    relations.insert(
        "MENTIONS".into(),
        json!(RelationExport {
            from: vec!["any node".into()],
            to: vec!["document section path#n".into()],
            description: Some(MENTIONS_DESCRIPTION.into()),
            built_in: true,
        }),
    );

    let patterns: Vec<PatternExport> = ont
        .patterns()
        .iter()
        .map(|(name, p)| PatternExport {
            name: name.clone(),
            description: p.description.clone(),
        })
        .collect();

    json!({
        "meta": meta_export,
        "parameters": parameters,
        "constraints": constraints,
        "relations": relations,
        "patterns": patterns,
        "strict": ont.strict(),
    })
}

/// Pretty-printed JSON string for tool responses.
pub fn export_pretty(ont: &Ontology) -> String {
    serde_json::to_string_pretty(&export_json(ont)).unwrap_or_else(|_| "{}".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ontology::Ontology;

    const PATTERN_TOML: &str = r#"
[meta]
domain = "test-domain"
description = "Test overlay."

[entities.Field]
id_prefix = "fld"
description = "A parameter."

[patterns.independent_enum]
description = "Field to Enum with aliases."
example = "Field → Enum(aliases=[v1,v2])"

[relations.CONSTRAINED_BY]
from = ["Field"]
to = ["Enum"]
description = "Field to constraint."

[validation]
strict = true

[constraint_types.Enum]
params = ["values"]
description = "Allowed values in aliases."
"#;

    #[test]
    fn export_includes_meta_and_patterns_without_example() {
        let ont = Ontology::parse(PATTERN_TOML).unwrap();
        let j = export_json(&ont);
        assert_eq!(j["meta"]["domain"], "test-domain");
        let patterns = j["patterns"].as_array().unwrap();
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0]["name"], "independent_enum");
        assert_eq!(patterns[0]["description"], "Field to Enum with aliases.");
        assert!(patterns[0].get("example").is_none());
        // example is parsed into Ontology for humans
        assert_eq!(
            ont.patterns().get("independent_enum").unwrap().example.as_deref(),
            Some("Field → Enum(aliases=[v1,v2])")
        );
    }

    #[test]
    fn export_injects_builtin_mentions() {
        let ont = Ontology::parse(PATTERN_TOML).unwrap();
        let j = export_json(&ont);
        let mentions = &j["relations"]["MENTIONS"];
        assert_eq!(mentions["built_in"], true);
        assert!(mentions["description"].as_str().unwrap().contains("path#n"));
    }

    #[test]
    fn export_meta_note_from_toml() {
        let ont = Ontology::parse(
            r#"
[meta]
note = "Literal-only parameter edges."
"#,
        )
        .unwrap();
        let j = export_json(&ont);
        assert_eq!(j["meta"]["note"], "Literal-only parameter edges.");
        assert!(j.get("note").is_none(), "top-level note removed; use meta.note");
    }

    #[test]
    fn export_conditional_from_toml_params() {
        let ont = Ontology::parse(
            r#"
[entities.Literal]
id_prefix = "lit"
[relations.IF_FIELD]
from = ["Conditional"]
to = ["Literal"]
[constraint_types.Conditional]
params = [
  "IF_FIELD → Literal (trigger field name as text)",
  "IF_VALUE → Literal (trigger value as text)",
]
description = "Never wire IF_FIELD/IF_VALUE to a Field node."
"#,
        )
        .unwrap();
        let j = export_json(&ont);
        let c = &j["constraints"]["Conditional"];
        let params = c["params"].as_array().unwrap();
        assert!(params[0].as_str().unwrap().contains("Literal"));
        assert!(c["description"].as_str().unwrap().contains("Never wire"));
    }
}
