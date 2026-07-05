//! Adapter: GraphStore → glossa_constraint::Problem
//!
//! Translates a constraint subgraph from the knowledge graph into the CSP solver's Problem.
//! Feature-gated behind `feature = "constraint"`.

use crate::graph::ontology::Ontology;
use crate::graph::store::GraphStore;
use glossa_constraint::{Constraint, FieldConstraints, Problem};

/// Edge types used as parameter edges from constraint nodes to literals.
#[allow(dead_code)]
const PARAM_EDGES: &[&str] = &["HAS_MIN", "HAS_MAX", "HAS_PATTERN", "HAS_EXPRESSION"];

/// Load a constraint problem from the graph for a given GOST document.
///
/// 1. Finds all Field nodes with the given source_path.
/// 2. For each Field, follows CONSTRAINED_BY → constraint nodes.
/// 3. For each constraint node, follows parameter edges → literals.
///
/// Returns None when the constraint feature is disabled (compile-time gate).
/// Load a constraint problem from the graph.
///
/// `source_path`:
/// - `Some(path)` — only Field nodes citing that document (multi-tenant isolation:
///   several products/GOSTs share one base, keep them apart). The MCP tool uses this.
/// - `None` — every Field in the graph. A single product's constraints are now
///   assembled from several standards (the main GOST + the ones it references),
///   so the eval unions across all source documents.
pub fn load_problem(g: &GraphStore, ont: &Ontology, source_path: Option<&str>) -> anyhow::Result<Problem> {
    let all_nodes = g.all_nodes()?;
    let all_edges = g.all_edges()?;

    let outgoing = build_outgoing(&all_edges);

    // Field nodes in scope: all of them, or only those citing `source_path`.
    let fields: Vec<_> = all_nodes
        .iter()
        .filter(|n| n.node_type == "Field" && source_path.map_or(true, |src| n.prov.source_path == src))
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

/// Re-key an assignment onto the graph's actual Field labels using glossa's
/// morphology-aware resolver (`GraphStore::resolve`, the same one graph_upsert
/// uses for edge endpoints). A value keyed by a paraphrase of a parameter name
/// ("h, высота") still lands on the Field the agent created ("высота" / "Высота Т"),
/// so a faithful domain isn't silently skipped over a wording difference.
/// Keys that already match a Field, or resolve to no Field, pass through
/// unchanged (the latter then surface as NOT CHECKED in the feedback).
pub fn resolve_assignment_fields(
    g: &GraphStore,
    problem: &Problem,
    assignment: &[(String, serde_json::Value)],
) -> Vec<(String, serde_json::Value)> {
    let field_names: std::collections::HashSet<&str> =
        problem.fields.iter().map(|f| f.name.as_str()).collect();
    assignment
        .iter()
        .map(|(k, v)| {
            if field_names.contains(k.as_str()) {
                return (k.clone(), v.clone());
            }
            let resolved = g.resolve(k).ok().and_then(|ids| {
                ids.iter().find_map(|id| {
                    g.get_node(id)
                        .ok()
                        .flatten()
                        .filter(|n| n.node_type == "Field" && field_names.contains(n.label.as_str()))
                        .map(|n| n.label)
                })
            });
            (resolved.unwrap_or_else(|| k.clone()), v.clone())
        })
        .collect()
}

/// Render a solver outcome as actionable tool feedback for the agent.
///
/// A raw `SolveResult` JSON hides the two failure modes that matter most to a
/// model: an EMPTY problem (0 fields for the source → every mode is vacuously
/// "valid") and assignments whose field names matched nothing in the graph
/// (silently not checked). Both are called out explicitly here.
pub fn format_solve_feedback(
    problem: &Problem,
    result: &glossa_constraint::solver::SolveResult,
    assignment: &[(String, serde_json::Value)],
    source_path: &str,
) -> String {
    use std::fmt::Write;
    let mut out = String::new();

    if problem.fields.is_empty() {
        let _ = writeln!(
            out,
            "WARNING: 0 Field nodes cite source_path '{source_path}' — nothing was solved; any 'valid' verdict below is vacuous.\n\
             Build the constraint graph first: Field nodes must cite this document via source_path (see graph_stats)."
        );
    }

    let _ = writeln!(out, "mode={}  valid={}", result.mode, result.valid);
    let field_names: Vec<&str> = problem.fields.iter().map(|f| f.name.as_str()).collect();
    let n_constraints: usize = problem.fields.iter().map(|f| f.constraints.len()).sum();
    let _ = writeln!(
        out,
        "problem: {} fields, {} constraints from '{source_path}'",
        field_names.len(),
        n_constraints
    );
    if !field_names.is_empty() {
        let shown = field_names.iter().take(30).cloned().collect::<Vec<_>>().join(", ");
        let more = if field_names.len() > 30 { format!(" … +{} more", field_names.len() - 30) } else { String::new() };
        let _ = writeln!(out, "fields: {shown}{more}");
    }

    if !assignment.is_empty() {
        let known: std::collections::HashSet<&str> = field_names.iter().copied().collect();
        let unmatched: Vec<&str> = assignment
            .iter()
            .map(|(k, _)| k.as_str())
            .filter(|k| !known.contains(k))
            .collect();
        let _ = writeln!(
            out,
            "assignments: {} given, {} matched a field",
            assignment.len(),
            assignment.len() - unmatched.len()
        );
        if !unmatched.is_empty() {
            let _ = writeln!(
                out,
                "  NOT CHECKED (no Field with this label in the graph — assignment keys must match Field labels exactly): {}",
                unmatched.join(", ")
            );
        }
    }

    for v in &result.violations {
        let _ = writeln!(
            out,
            "violation: field '{}' [{}] {} (expected: {}; actual: {})",
            v.field, v.constraint, v.message, v.expected, v.actual
        );
    }
    for i in &result.issues {
        let _ = writeln!(out, "issue [{}]: field '{}': {}", i.severity, i.field, i.message);
    }
    if let Some(domains) = &result.domains {
        for d in domains {
            let _ = writeln!(out, "domain: {} = {}", d.field, format_domain(&d.domain));
        }
    }
    if result.valid && result.violations.is_empty() && result.issues.is_empty() && !problem.fields.is_empty() {
        let _ = writeln!(out, "no violations or issues.");
    }
    out.trim_end().to_string()
}

fn format_domain(d: &glossa_constraint::Domain) -> String {
    use glossa_constraint::Domain;
    match d {
        Domain::Interval { min, max } => format!("[{min}, {max}]"),
        Domain::Set { values } => {
            let shown = values.iter().take(40).cloned().collect::<Vec<_>>().join(", ");
            let more = if values.len() > 40 { format!(" … +{} more", values.len() - 40) } else { String::new() };
            format!("{{ {shown}{more} }}")
        }
        Domain::Any => "any (unconstrained)".into(),
        Domain::Empty => "EMPTY (unsatisfiable — constraints conflict)".into(),
        Domain::Regex { pattern } => format!("matching /{pattern}/"),
    }
}

/// Build Constraint variants from a constraint graph node.
pub fn constraints_from_node(
    cn: &crate::graph::store::Node,
    outgoing: &std::collections::HashMap<String, Vec<(String, String)>>,
    all_nodes: &[crate::graph::store::Node],
    ont: &Ontology,
) -> anyhow::Result<Vec<Constraint>> {
    build_constraint(cn, outgoing, all_nodes, ont)
}

/// Whether a reconstructed constraint carries actionable domain (not malformed/empty).
pub fn constraint_meaningful(c: &Constraint) -> bool {
    match c {
        Constraint::Enum { values } => !values.is_empty(),
        Constraint::Regex { pattern } => !pattern.is_empty(),
        Constraint::Formula { expression } => !expression.is_empty(),
        Constraint::Range { min, max } => min.is_finite() && max.is_finite() && min <= max,
        Constraint::Required | Constraint::Forbidden => true,
        Constraint::Conditional {
            condition_field,
            inner,
            ..
        } => !condition_field.is_empty() && constraint_meaningful(inner),
    }
}

fn build_outgoing(
    all_edges: &[crate::graph::store::Edge],
) -> std::collections::HashMap<String, Vec<(String, String)>> {
    let mut outgoing: std::collections::HashMap<String, Vec<(String, String)>> =
        std::collections::HashMap::new();
    for e in all_edges {
        outgoing
            .entry(e.from.clone())
            .or_default()
            .push((e.edge_type.clone(), e.to.clone()));
    }
    outgoing
}

/// All meaningful constraints on one Field (same extraction as `load_problem`).
pub fn field_constraints(
    g: &GraphStore,
    field_id: &str,
    ont: &Ontology,
) -> anyhow::Result<Vec<Constraint>> {
    let all_nodes = g.all_nodes()?;
    let outgoing = build_outgoing(&g.all_edges()?);
    let Some(field) = all_nodes.iter().find(|n| n.id == field_id) else {
        return Ok(vec![]);
    };
    if field.node_type != "Field" {
        return Ok(vec![]);
    }
    let mut constraints = Vec::new();
    if let Some(edges) = outgoing.get(field_id) {
        for (edge_type, constraint_id) in edges {
            if edge_type != "CONSTRAINED_BY" {
                continue;
            }
            let Some(cn) = all_nodes.iter().find(|n| n.id == *constraint_id) else {
                continue;
            };
            constraints.extend(constraints_from_node(cn, &outgoing, &all_nodes, ont)?);
        }
    }
    Ok(constraints
        .into_iter()
        .filter(|c| constraint_meaningful(c))
        .collect())
}

/// Field has at least one materialized constraint on any `CONSTRAINED_BY` target.
pub fn field_has_materialized_constraints(
    g: &GraphStore,
    field_id: &str,
    ont: &Ontology,
) -> anyhow::Result<bool> {
    Ok(!field_constraints(g, field_id, ont)?.is_empty())
}

/// Phase-A mapping score: fraction of MDM column values covered by the Field's
/// constraint union (Enum aliases across all Conditional branches; Range intervals).
pub fn field_reference_overlap(
    g: &GraphStore,
    field_id: &str,
    ont: &Ontology,
    mdm_valid: &std::collections::BTreeSet<String>,
) -> anyhow::Result<f64> {
    if mdm_valid.is_empty() {
        return Ok(0.0);
    }
    let constraints = field_constraints(g, field_id, ont)?;
    if constraints.is_empty() {
        return Ok(0.0);
    }
    let agent_dom = enum_reference_union(&constraints);
    let covered = mdm_valid
        .iter()
        .filter(|v| reference_value_covered(&constraints, &agent_dom, v))
        .count();
    Ok(covered as f64 / mdm_valid.len() as f64)
}

/// Union of normalized Enum reference strings (Conditional branches flattened).
fn enum_reference_union(constraints: &[Constraint]) -> std::collections::BTreeSet<String> {
    use glossa_constraint::solver::canon_scalar;
    let mut out = std::collections::BTreeSet::new();
    fn walk(c: &Constraint, out: &mut std::collections::BTreeSet<String>) {
        match c {
            Constraint::Enum { values } => {
                for v in values {
                    out.insert(canon_scalar(v));
                }
            }
            Constraint::Conditional { inner, .. } => walk(inner, out),
            _ => {}
        }
    }
    for c in constraints {
        walk(c, &mut out);
    }
    out
}

fn reference_value_covered(
    constraints: &[Constraint],
    agent_dom: &std::collections::BTreeSet<String>,
    ref_val: &str,
) -> bool {
    use glossa_constraint::solver::canon_scalar;
    let target = canon_scalar(ref_val);
    if agent_dom.contains(&target)
        || agent_dom
            .iter()
            .any(|a| glossa_constraint::enum_alias_matches(a, ref_val))
    {
        return true;
    }
    constraints
        .iter()
        .any(|c| constraint_covers_scalar_for_mapping(c, ref_val))
}

fn constraint_covers_scalar_for_mapping(c: &Constraint, ref_val: &str) -> bool {
    match c {
        Constraint::Enum { values } => {
            let target = glossa_constraint::solver::canon_scalar(ref_val);
            values.iter().any(|a| {
                glossa_constraint::solver::canon_scalar(a) == target
                    || glossa_constraint::enum_alias_matches(a, ref_val)
            })
        }
        Constraint::Conditional { inner, .. } => constraint_covers_scalar_for_mapping(inner, ref_val),
        Constraint::Range { min, max } => ref_val
            .parse::<f64>()
            .ok()
            .is_some_and(|n| n >= *min && n <= *max),
        Constraint::Regex { pattern } => glossa_constraint::regex_constraint_matches(pattern, ref_val),
        _ => false,
    }
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
            Ok(vec![Constraint::Enum {
                values: cn.aliases.clone(),
            }])
        }
        "Formula" => {
            let expression = literals.get("HAS_EXPRESSION").cloned().unwrap_or_default();
            Ok(vec![Constraint::Formula { expression }])
        }
        "Conditional" => {
            // Ontology shape: Conditional ──IF_FIELD──► Literal (the field name),
            // ──IF_VALUE──► Literal (the trigger value), ──HAS_CONSTRAINT──► the
            // constraint node(s) that apply while the condition holds. Each inner
            // constraint is wrapped; a Conditional missing its condition or inner
            // constraints contributes nothing (malformed).
            let condition_field = literals.get("IF_FIELD").cloned().unwrap_or_default();
            let condition_value = literals.get("IF_VALUE").cloned().unwrap_or_default();
            if condition_field.is_empty() {
                return Ok(vec![]);
            }
            let mut wrapped = Vec::new();
            if let Some(edges) = outgoing.get(&cn.id) {
                for (edge_type, to_id) in edges {
                    if edge_type != "HAS_CONSTRAINT" {
                        continue;
                    }
                    let Some(inner_node) = all_nodes.iter().find(|n| n.id == *to_id) else { continue };
                    if inner_node.node_type == "Conditional" {
                        continue; // ontology forbids nesting; also guards against cycles
                    }
                    for inner in build_constraint(inner_node, outgoing, all_nodes, _ont)? {
                        wrapped.push(Constraint::Conditional {
                            condition_field: condition_field.clone(),
                            condition_value: serde_json::Value::String(condition_value.clone()),
                            inner: Box::new(inner),
                        });
                    }
                }
            }
            Ok(wrapped)
        }
        _ => Ok(vec![]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::store::{GraphStore, Node, Provenance, Edge};

    fn make_adapter_problem(g: &GraphStore, ont: &Ontology, src: &str) -> Problem {
        load_problem(g, ont, Some(src)).unwrap()
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

    fn insert_node_aliases(g: &GraphStore, id: &str, node_type: &str, label: &str, aliases: &[&str], src: &str) {
        g.put_node(&Node {
            id: id.into(),
            node_type: node_type.into(),
            label: label.into(),
            aliases: aliases.iter().map(|s| s.to_string()).collect(),
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
[relations.CONSTRAINED_BY]
from = ["Field"]
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
[relations.HAS_EXPRESSION]
from = ["Formula"]
to = ["Literal"]
[relations.IF_FIELD]
from = ["Conditional"]
to = ["Literal"]
[relations.IF_VALUE]
from = ["Conditional"]
to = ["Literal"]
[relations.HAS_CONSTRAINT]
from = ["Conditional"]
to = ["*"]
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
[constraint_types.Required]
params = []
[constraint_types.Forbidden]
params = []
[constraint_types.Conditional]
params = ["condition_field", "condition_value"]
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
    fn load_enum_constraint_via_aliases() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = make_ont();

        insert_node(&g, "fld:t", "Field", "Type", "gost.pdf");
        insert_node_aliases(&g, "fld:t-enum", "Enum", "Type enum", &["A", "B"], "gost.pdf");
        insert_edge(&g, "fld:t", "CONSTRAINED_BY", "fld:t-enum", "gost.pdf");

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

    /// Compact Enum: allowed values live in the Enum node's aliases, no Domain/Literal
    /// nodes. One Field + one Enum + one edge is the whole constraint.
    #[test]
    fn load_enum_from_aliases_compact() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = make_ont();

        insert_node(&g, "fld:d", "Field", "Наружный диаметр", "gost.pdf");
        insert_node_aliases(&g, "enum:d", "Enum", "Наружный диаметр enum", &["125", "150", "115"], "gost.pdf");
        insert_edge(&g, "fld:d", "CONSTRAINED_BY", "enum:d", "gost.pdf");

        let problem = make_adapter_problem(&g, &ont, "gost.pdf");
        let f = problem.fields.iter().find(|f| f.name == "Наружный диаметр").unwrap();
        let values = f.constraints.iter().find_map(|c| match c {
            Constraint::Enum { values } => Some(values.clone()),
            _ => None,
        }).expect("Enum constraint from aliases");
        assert_eq!(values, vec!["125", "150", "115"]);

        // The solver honours it: 150 passes, 137 (in-between) is rejected.
        assert!(glossa_constraint::solver::validate(&problem,
            &[("Наружный диаметр".into(), serde_json::json!(150))]).is_empty());
        assert!(!glossa_constraint::solver::validate(&problem,
            &[("Наружный диаметр".into(), serde_json::json!(137))]).is_empty());
    }

    /// A value keyed by a paraphrase of the parameter name must still hit the
    /// Field the agent built, via the morphology resolver — otherwise a faithful
    /// domain is silently skipped and a bad value slips through as "valid".
    #[test]
    fn assignment_resolves_paraphrased_field_name() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = make_ont();

        // Agent named the field "Высота Т"; the assignment keys it "h, высота".
        insert_node(&g, "fld:h", "Field", "Высота Т", "gost.pdf");
        insert_node_aliases(&g, "enum:h", "Enum", "Высота Т enum", &["0,6", "0,8", "1,2"], "gost.pdf");
        insert_edge(&g, "fld:h", "CONSTRAINED_BY", "enum:h", "gost.pdf");

        let problem = make_adapter_problem(&g, &ont, "gost.pdf");
        let raw = vec![("h, высота".to_string(), serde_json::json!("99"))];

        // Without resolution the key misses the field and the bad value passes.
        assert!(glossa_constraint::solver::validate(&problem, &raw).is_empty());

        // With resolution it re-keys onto "Высота Т" and the enum rejects 99.
        let resolved = resolve_assignment_fields(&g, &problem, &raw);
        assert_eq!(resolved[0].0, "Высота Т");
        assert!(!glossa_constraint::solver::validate(&problem, &resolved).is_empty(),
            "99 ∉ {{0,6; 0,8; 1,2}} must be caught after resolution");
    }

    /// Conditional reconstructed from the graph: IF_FIELD/IF_VALUE literals give
    /// the condition, each HAS_CONSTRAINT target becomes a wrapped inner constraint.
    #[test]
    fn load_conditional_from_graph() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = make_ont();

        insert_node(&g, "fld:x", "Field", "X", "gost.pdf");
        insert_node(&g, "cond:x", "Conditional", "X when mode=41", "gost.pdf");
        insert_edge(&g, "fld:x", "CONSTRAINED_BY", "cond:x", "gost.pdf");

        insert_node(&g, "lit:field", "Literal", "mode", "gost.pdf");
        insert_node(&g, "lit:value", "Literal", "41", "gost.pdf");
        insert_edge(&g, "cond:x", "IF_FIELD", "lit:field", "gost.pdf");
        insert_edge(&g, "cond:x", "IF_VALUE", "lit:value", "gost.pdf");

        insert_node(&g, "req:x", "Required", "X required", "gost.pdf");
        insert_edge(&g, "cond:x", "HAS_CONSTRAINT", "req:x", "gost.pdf");

        let problem = make_adapter_problem(&g, &ont, "gost.pdf");
        let x = problem.fields.iter().find(|f| f.name == "X").expect("field X");
        let cond = x.constraints.iter().find_map(|c| match c {
            Constraint::Conditional { condition_field, condition_value, inner } =>
                Some((condition_field.clone(), condition_value.clone(), inner.clone())),
            _ => None,
        });
        let (cf, cv, inner) = cond.expect("Conditional must be reconstructed, not dropped");
        assert_eq!(cf, "mode");
        assert_eq!(cv, serde_json::Value::String("41".into()));
        assert_eq!(*inner, Constraint::Required);

        // The condition actually gates validation end-to-end.
        let v_active = glossa_constraint::solver::validate(
            &problem,
            &[("mode".to_string(), serde_json::json!(41))],
        );
        assert!(v_active.iter().any(|v| v.field == "X" && v.constraint == "Required"),
            "condition met + X missing → Required must fire: {v_active:?}");
        let v_inactive = glossa_constraint::solver::validate(
            &problem,
            &[("mode".to_string(), serde_json::json!("other"))],
        );
        assert!(v_inactive.is_empty(), "condition not met → no violations: {v_inactive:?}");
    }

    #[test]
    fn field_has_materialized_constraints_conditional_enum() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = make_ont();

        insert_node(&g, "fld:d", "Field", "D", "gost.pdf");
        insert_node(&g, "cond:41", "Conditional", "D when type=41", "gost.pdf");
        insert_edge(&g, "fld:d", "CONSTRAINED_BY", "cond:41", "gost.pdf");
        insert_node(&g, "lit:f", "Literal", "type", "gost.pdf");
        insert_node(&g, "lit:v", "Literal", "41", "gost.pdf");
        insert_edge(&g, "cond:41", "IF_FIELD", "lit:f", "gost.pdf");
        insert_edge(&g, "cond:41", "IF_VALUE", "lit:v", "gost.pdf");
        insert_node_aliases(&g, "enum:41", "Enum", "D41", &["50", "63"], "gost.pdf");
        insert_edge(&g, "cond:41", "HAS_CONSTRAINT", "enum:41", "gost.pdf");

        assert!(field_has_materialized_constraints(&g, "fld:d", &ont).unwrap());
    }

    #[test]
    fn field_has_materialized_constraints_empty_enum_unbuilt() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = make_ont();

        insert_node(&g, "fld:x", "Field", "X", "gost.pdf");
        insert_node_aliases(&g, "enum:x", "Enum", "X enum", &[], "gost.pdf");
        insert_edge(&g, "fld:x", "CONSTRAINED_BY", "enum:x", "gost.pdf");

        assert!(!field_has_materialized_constraints(&g, "fld:x", &ont).unwrap());
    }

    #[test]
    fn field_reference_overlap_unions_conditional_branches() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = make_ont();

        insert_node(&g, "fld:d", "Field", "D", "gost.pdf");
        for (suffix, vals) in [("41", &["50", "63"][..]), ("42", &["80", "100"][..])] {
            let cid = format!("cond:{suffix}");
            insert_node(&g, &cid, "Conditional", &format!("D type={suffix}"), "gost.pdf");
            insert_edge(&g, "fld:d", "CONSTRAINED_BY", &cid, "gost.pdf");
            insert_node(&g, &format!("lit:f:{suffix}"), "Literal", "type", "gost.pdf");
            insert_node(&g, &format!("lit:v:{suffix}"), "Literal", suffix, "gost.pdf");
            insert_edge(&g, &cid, "IF_FIELD", &format!("lit:f:{suffix}"), "gost.pdf");
            insert_edge(&g, &cid, "IF_VALUE", &format!("lit:v:{suffix}"), "gost.pdf");
            let eid = format!("enum:{suffix}");
            insert_node_aliases(&g, &eid, "Enum", "D enum", vals, "gost.pdf");
            insert_edge(&g, &cid, "HAS_CONSTRAINT", &eid, "gost.pdf");
        }

        let mdm: std::collections::BTreeSet<String> =
            ["50", "63", "80", "100"].iter().map(|s| s.to_string()).collect();
        let score = field_reference_overlap(&g, "fld:d", &ont, &mdm).unwrap();
        assert!((score - 1.0).abs() < 1e-9, "union of branches covers all MDM values: {score}");
    }
}
