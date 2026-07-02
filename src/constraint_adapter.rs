//! Adapter: GraphStore → glossa_constraint::Problem
//!
//! Translates a constraint subgraph from the knowledge graph into the CSP solver's Problem.
//! Feature-gated behind `feature = "constraint"`.

use crate::graph::ontology::Ontology;
use crate::graph::store::GraphStore;
use glossa_constraint::{Constraint, FieldConstraints, Problem};

/// Edge types used as parameter edges from constraint nodes to literals.
/// Defined in ontology.toml under [relations].
#[allow(dead_code)]
const PARAM_EDGES: &[&str] = &["HAS_MIN", "HAS_MAX", "HAS_PATTERN", "HAS_DOMAIN", "HAS_EXPRESSION"];

/// Load a constraint problem from the graph for a given GOST document.
///
/// 1. Finds all Field nodes with the given source_path.
/// 2. For each Field, follows CONSTRAINED_BY → constraint nodes.
/// 3. For each constraint node, follows parameter edges → literals.
///
/// Returns None when the constraint feature is disabled (compile-time gate).
pub fn load_problem(g: &GraphStore, ont: &Ontology, source_path: &str) -> anyhow::Result<Problem> {
    let all_nodes = g.all_nodes()?;
    let all_edges = g.all_edges()?;

    // Index edges by source id for efficient lookup
    let mut outgoing: std::collections::HashMap<String, Vec<(String, String)>> =
        std::collections::HashMap::new();
    for e in &all_edges {
        outgoing
            .entry(e.from.clone())
            .or_default()
            .push((e.edge_type.clone(), e.to.clone()));
    }

    // Find all Field nodes for this source_path
    let fields: Vec<_> = all_nodes
        .iter()
        .filter(|n| n.node_type == "Field" && n.prov.source_path == source_path)
        .collect();

    let mut field_constraints = Vec::new();

    for field_node in &fields {
        let mut constraints: Vec<Constraint> = Vec::new();

        // Follow CONSTRAINED_BY edges from this field
        if let Some(edges) = outgoing.get(&field_node.id) {
            for (edge_type, constraint_id) in edges {
                if edge_type != "CONSTRAINED_BY" {
                    continue;
                }

                // Find the constraint node
                let cn = match all_nodes.iter().find(|n| n.id == *constraint_id) {
                    Some(n) => n,
                    None => continue,
                };

                // Build constraint from node type
                let constraint = build_constraint(&cn, &outgoing, &all_nodes, ont)?;
                constraints.extend(constraint);
            }
        }

        field_constraints.push(FieldConstraints {
            name: field_node.label.clone(),
            constraints,
        });
    }

    Ok(Problem {
        fields: field_constraints,
    })
}

/// Build Constraint variants from a constraint graph node.
fn build_constraint(
    cn: &crate::graph::store::Node,
    outgoing: &std::collections::HashMap<String, Vec<(String, String)>>,
    all_nodes: &[crate::graph::store::Node],
    _ont: &Ontology,
) -> anyhow::Result<Vec<Constraint>> {
    let constraint_type = cn.node_type.as_str();

    // Look up literals reachable from this constraint node
    let mut literals: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    if let Some(edges) = outgoing.get(&cn.id) {
        for (edge_type, to_id) in edges {
            if let Some(lit_node) = all_nodes.iter().find(|n| n.id == *to_id) {
                literals.insert(edge_type.clone(), lit_node.label.clone());
            }
        }
    }

    match constraint_type {
        "Range" => {
            let min_lit = literals.get("HAS_MIN").cloned().unwrap_or_default();
            let max_lit = literals.get("HAS_MAX").cloned().unwrap_or_default();
            let min: f64 = min_lit.parse().unwrap_or(f64::NEG_INFINITY);
            let max: f64 = max_lit.parse().unwrap_or(f64::INFINITY);
            Ok(vec![Constraint::Range { min, max }])
        }
        "Regex" => {
            let pattern = literals.get("HAS_PATTERN").cloned().unwrap_or_default();
            Ok(vec![Constraint::Regex { pattern }])
        }
        "Required" => Ok(vec![Constraint::Required]),
        "Forbidden" => Ok(vec![Constraint::Forbidden]),
        "Enum" => {
            // For enum, HAS_DOMAIN points to a Domain node whose HAS_LITERAL edges are the enum values
            let domain_id = outgoing.get(&cn.id).and_then(|edges| {
                edges
                    .iter()
                    .find(|(et, _)| et == "HAS_DOMAIN")
                    .map(|(_, to_id)| to_id.clone())
            });
            let values = match domain_id {
                Some(ref did) => outgoing
                    .get(did)
                    .map(|edges| {
                        edges
                            .iter()
                            .filter(|(et, _)| et == "HAS_LITERAL")
                            .filter_map(|(_, to_id)| {
                                all_nodes.iter().find(|n| n.id == *to_id).map(|n| n.label.clone())
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                None => vec![],
            };
            Ok(vec![Constraint::Enum { values }])
        }
        "Formula" => {
            let expression = literals.get("HAS_EXPRESSION").cloned().unwrap_or_default();
            Ok(vec![Constraint::Formula { expression }])
        }
        "Conditional" => {
            Ok(vec![]) // Conditional is reconstructed during agent graph building
        }
        _ => Ok(vec![]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::store::{GraphStore, Node, Provenance, Edge};

    fn make_adapter_problem(g: &GraphStore, ont: &Ontology, src: &str) -> Problem {
        load_problem(g, ont, src).unwrap()
    }

    fn insert_node(g: &GraphStore, id: &str, node_type: &str, label: &str, src: &str) {
        g.put_node(&Node {
            id: id.into(),
            node_type: node_type.into(),
            label: label.into(),
            aliases: vec![],
            prov: Provenance {
                source_path: src.into(),
                range: None,
                file_sig: None,
                origin: "agent".into(),
                confidence: 0.8,
                created_at: 1,
            },
        })
        .unwrap();
    }

    fn insert_edge(g: &GraphStore, from: &str, edge_type: &str, to: &str, src: &str) {
        g.put_edge(&Edge {
            from: from.into(),
            to: to.into(),
            edge_type: edge_type.into(),
            prov: Provenance {
                source_path: src.into(),
                range: None,
                file_sig: None,
                origin: "agent".into(),
                confidence: 0.8,
                created_at: 1,
            },
        })
        .unwrap();
    }

    fn make_ont() -> Ontology {
        Ontology::parse(
            r#"
[entities.Field]
id_prefix = "fld"
[entities.Literal]
id_prefix = "lit"
[entities.Domain]
id_prefix = "dom"
[relations.CONSTRAINED_BY]
from = ["Field", "Domain"]
to = ["*"]
[relations.HAS_MIN]
from = ["Range"]
to = ["Literal"]
[relations.HAS_MAX]
from = ["Range"]
to = ["Literal"]
[relations.HAS_PATTERN]
from = ["Regex"]
to = ["Literal"]
[relations.HAS_DOMAIN]
from = ["Enum"]
to = ["Domain"]
[relations.HAS_LITERAL]
from = ["Enum", "Domain"]
to = ["Literal"]
[relations.HAS_EXPRESSION]
from = ["Formula"]
to = ["Literal"]
[validation]
strict = false
[constraint_types.Range]
params = ["min", "max"]
[constraint_types.Regex]
params = ["pattern"]
[constraint_types.Enum]
params = ["values"]
[constraint_types.Formula]
params = ["expression"]
"#,
        )
        .unwrap()
    }

    #[test]
    fn load_single_range_constraint() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = make_ont();

        insert_node(&g, "fld:x", "Field", "X", "gost.pdf");
        insert_node(&g, "lit:0", "Literal", "0", "gost.pdf");
        insert_node(&g, "lit:100", "Literal", "100", "gost.pdf");
        insert_edge(&g, "fld:x", "HAS_LITERAL", "lit:0", "gost.pdf");
        insert_edge(&g, "fld:x", "HAS_LITERAL", "lit:100", "gost.pdf");
        // The model stores constraints via CONSTRAINED_BY + constraint node
        insert_node(&g, "fld:x-rng", "Range", "X range", "gost.pdf");
        insert_edge(&g, "fld:x", "CONSTRAINED_BY", "fld:x-rng", "gost.pdf");
        insert_edge(&g, "fld:x-rng", "HAS_MIN", "lit:0", "gost.pdf");
        insert_edge(&g, "fld:x-rng", "HAS_MAX", "lit:100", "gost.pdf");

        let problem = make_adapter_problem(&g, &ont, "gost.pdf");
        assert_eq!(problem.fields.len(), 1);
        assert_eq!(problem.fields[0].name, "X");
        assert_eq!(problem.fields[0].constraints.len(), 1);
        assert_eq!(
            problem.fields[0].constraints[0],
            Constraint::Range {
                min: 0.0,
                max: 100.0
            }
        );
    }

    #[test]
    fn load_enum_constraint_via_domain() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = make_ont();

        insert_node(&g, "fld:t", "Field", "Type", "gost.pdf");
        insert_node(&g, "dom:types", "Domain", "Valid types", "gost.pdf");
        insert_node(&g, "lit:a", "Literal", "A", "gost.pdf");
        insert_node(&g, "lit:b", "Literal", "B", "gost.pdf");

        insert_node(&g, "fld:t-enum", "Enum", "Type enum", "gost.pdf");
        insert_edge(&g, "fld:t", "CONSTRAINED_BY", "fld:t-enum", "gost.pdf");
        insert_edge(&g, "fld:t-enum", "HAS_DOMAIN", "dom:types", "gost.pdf");
        insert_edge(&g, "dom:types", "HAS_LITERAL", "lit:a", "gost.pdf");
        insert_edge(&g, "dom:types", "HAS_LITERAL", "lit:b", "gost.pdf");

        let problem = make_adapter_problem(&g, &ont, "gost.pdf");
        assert_eq!(problem.fields.len(), 1);
        assert_eq!(
            problem.fields[0].constraints[0],
            Constraint::Enum {
                values: vec!["A".into(), "B".into()]
            }
        );
    }

    #[test]
    fn load_multiple_gosts_isolated() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = make_ont();

        // GOST 1
        insert_node(&g, "fld:x1", "Field", "X", "gost1.pdf");
        insert_node(&g, "fld:x1-rng", "Range", "X range", "gost1.pdf");
        insert_node(&g, "lit:0", "Literal", "0", "gost1.pdf");
        insert_node(&g, "lit:10", "Literal", "10", "gost1.pdf");
        insert_edge(&g, "fld:x1", "CONSTRAINED_BY", "fld:x1-rng", "gost1.pdf");
        insert_edge(&g, "fld:x1-rng", "HAS_MIN", "lit:0", "gost1.pdf");
        insert_edge(&g, "fld:x1-rng", "HAS_MAX", "lit:10", "gost1.pdf");

        // GOST 2 — same field name, different range
        insert_node(&g, "fld:x2", "Field", "X", "gost2.pdf");
        insert_node(&g, "fld:x2-rng", "Range", "X range", "gost2.pdf");
        insert_node(&g, "lit:100", "Literal", "100", "gost2.pdf");
        insert_node(&g, "lit:200", "Literal", "200", "gost2.pdf");
        insert_edge(&g, "fld:x2", "CONSTRAINED_BY", "fld:x2-rng", "gost2.pdf");
        insert_edge(&g, "fld:x2-rng", "HAS_MIN", "lit:100", "gost2.pdf");
        insert_edge(&g, "fld:x2-rng", "HAS_MAX", "lit:200", "gost2.pdf");

        let p1 = make_adapter_problem(&g, &ont, "gost1.pdf");
        assert_eq!(p1.fields.len(), 1);
        if let Constraint::Range { min, max } = p1.fields[0].constraints[0] {
            assert_eq!(min, 0.0);
            assert_eq!(max, 10.0);
        } else {
            panic!("expected Range");
        }

        let p2 = make_adapter_problem(&g, &ont, "gost2.pdf");
        assert_eq!(p2.fields.len(), 1);
        if let Constraint::Range { min, max } = p2.fields[0].constraints[0] {
            assert_eq!(min, 100.0);
            assert_eq!(max, 200.0);
        } else {
            panic!("expected Range");
        }
    }

    #[test]
    fn load_required_and_forbidden() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = make_ont();

        insert_node(&g, "fld:a", "Field", "A", "gost.pdf");
        insert_node(&g, "fld:a-req", "Required", "A required", "gost.pdf");
        insert_edge(&g, "fld:a", "CONSTRAINED_BY", "fld:a-req", "gost.pdf");

        insert_node(&g, "fld:b", "Field", "B", "gost.pdf");
        insert_node(&g, "fld:b-forb", "Forbidden", "B forbidden", "gost.pdf");
        insert_edge(&g, "fld:b", "CONSTRAINED_BY", "fld:b-forb", "gost.pdf");

        let problem = make_adapter_problem(&g, &ont, "gost.pdf");
        assert_eq!(problem.fields.len(), 2);
        assert!(problem.fields.iter().any(|f| f.name == "A"
            && f.constraints.contains(&Constraint::Required)));
        assert!(problem.fields.iter().any(|f| f.name == "B"
            && f.constraints.contains(&Constraint::Forbidden)));
    }
}
