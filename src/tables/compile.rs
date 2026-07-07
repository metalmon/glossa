//! Compile a materialized relation (`.csp` limit tables) into a constraint graph.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::graph::ontology::{OnUnsupported, Ontology};
use crate::graph::ops::{graph_upsert, UpsertEdge, UpsertNode};
use crate::graph::store::{normalize_label, GraphStore};
use crate::index::store::DocIndex;
use crate::tables::capabilities::{
    CapabilityId, CapabilityScan, CapabilityStatus, PATTERN_CONDITIONAL_ENUM,
    PATTERN_INDEPENDENT_ENUM,
};
use crate::tables::csp::{column_index, load_csp_dir, CspTable};
use crate::tables::wiring::TablesCompileWiring;

pub struct CompileReport {
    pub lines: Vec<String>,
}

/// Build constraint nodes from `.csp` limit tables and write via `graph_upsert`.
pub fn tables_to_graph(
    idx: &DocIndex,
    g: &GraphStore,
    ont: &Ontology,
    doc: &str,
    tables_dir: &Path,
) -> anyhow::Result<CompileReport> {
    let wiring = TablesCompileWiring::resolve(ont).map_err(|e| anyhow::anyhow!(e))?;
    let scan = CapabilityScan::scan(ont, &wiring);

    let cfg = ont.tables();
    let delimiter = cfg.delimiter.chars().next().unwrap_or(crate::tables::csp::CSP_DELIMITER);
    let rel = load_csp_dir(tables_dir, delimiter, Some(doc))?;
    let param_cols = parameter_columns(&rel, ont);
    if param_cols.is_empty() {
        anyhow::bail!("no parameter columns found in {}", tables_dir.display());
    }

    let mut nodes: Vec<UpsertNode> = Vec::new();
    let mut edges: Vec<UpsertEdge> = Vec::new();
    let mut lines: Vec<String> = Vec::new();
    lines.extend(scan.report_lines());
    lines.push(format!("doc={doc}"));
    lines.push(format!("tables-dir={}", tables_dir.display()));
    lines.push(format!(
        "relation: {} columns, {} rows",
        rel.headers.len(),
        rel.rows.len()
    ));

    for field_label in &param_cols {
        if let Some(hint) = cfg.compile_hints.get(field_label) {
            if hint.strategy.as_deref() == Some("skip") {
                lines.push(format!("Field «{field_label}» → skipped (compile hint)"));
                continue;
            }
            if matches!(
                hint.strategy.as_deref(),
                Some("formula" | "regex" | "range_if_numeric")
            ) {
                let cap = match hint.strategy.as_deref() {
                    Some("formula") => CapabilityId::Formula,
                    Some("regex") => CapabilityId::Regex,
                    _ => CapabilityId::ConditionalRange,
                };
                if let Err(e) = require_capability(&scan, cap, field_label, cfg.on_unsupported) {
                    match cfg.on_unsupported {
                        OnUnsupported::Error => return Err(e),
                        OnUnsupported::Skip | OnUnsupported::Warn => {
                            lines.push(format!("Field «{field_label}» → skipped ({e})"));
                            continue;
                        }
                    }
                }
            }
        }
        let y_idx = column_index(&rel.headers, field_label)
            .ok_or_else(|| anyhow::anyhow!("column not found for parameter {field_label}"))?;
        let triggers: Vec<String> = param_cols
            .iter()
            .filter(|c| *c != field_label)
            .filter(|c| column_index(&rel.headers, c).is_some())
            .cloned()
            .collect();

        nodes.push(UpsertNode {
            node_type: wiring.parameter_entity.clone(),
            label: field_label.clone(),
            source_path: doc.into(),
            aliases: vec![],
        });

        let shape = compile_field(
            &rel,
            y_idx,
            &triggers,
            field_label,
            doc,
            &wiring,
            &scan,
            &mut nodes,
            &mut edges,
        )?;
        lines.push(format!("Field «{field_label}» → {shape}"));
    }

    let layer_types = wiring.compile_layer_node_types(ont);
    let replaced = g
        .delete_agent_table_compile_layer(doc, &layer_types)
        .map_err(|e| anyhow::anyhow!("replace prior table-compile layer: {e}"))?;
    if replaced.nodes_removed > 0 || replaced.edges_removed > 0 {
        lines.push(format!(
            "replaced: {} nodes, {} edges removed",
            replaced.nodes_removed, replaced.edges_removed
        ));
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let out = graph_upsert(idx, g, ont, nodes, edges, now);
    if out.rejected {
        anyhow::bail!("graph_upsert rejected: {}", out.message);
    }
    lines.push(out.message);
    Ok(CompileReport { lines })
}

fn require_capability(
    scan: &CapabilityScan,
    id: CapabilityId,
    field: &str,
    _mode: OnUnsupported,
) -> Result<(), anyhow::Error> {
    match scan.status(id) {
        Some(CapabilityStatus::Ready) => Ok(()),
        Some(CapabilityStatus::OntologyGap(msg) | CapabilityStatus::CompilerGap(msg)) => {
            let e = anyhow::anyhow!(
                "Field \"{field}\" requires capability {} ({msg})",
                id.pattern_key()
            );
            Err(e)
        }
        None => Err(anyhow::anyhow!("unknown capability for Field \"{field}\"")),
    }
}

fn parameter_columns(rel: &CspTable, ont: &Ontology) -> Vec<String> {
    let cfg = ont.tables();
    let skip: BTreeSet<String> = cfg
        .skip_columns
        .iter()
        .map(|s| normalize_label(s))
        .collect();
    if !cfg.parameter_columns.is_empty() {
        return cfg
            .parameter_columns
            .iter()
            .filter(|c| column_index(&rel.headers, c).is_some())
            .cloned()
            .collect();
    }
    rel.headers
        .iter()
        .filter(|h| !h.is_empty() && !skip.contains(&normalize_label(h)))
        .cloned()
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn compile_field(
    rel: &CspTable,
    y_idx: usize,
    triggers: &[String],
    field_label: &str,
    doc: &str,
    wiring: &TablesCompileWiring,
    scan: &CapabilityScan,
    nodes: &mut Vec<UpsertNode>,
    edges: &mut Vec<UpsertEdge>,
) -> anyhow::Result<String> {
    let all_y: BTreeSet<String> = rel
        .rows
        .iter()
        .filter_map(|r| r.get(y_idx).filter(|v| !v.is_empty()).cloned())
        .collect();
    if all_y.is_empty() {
        return Ok("empty (no values)".into());
    }

    if let Some(trigger) = best_single_trigger(rel, y_idx, triggers) {
        let t_idx = column_index(&rel.headers, &trigger).unwrap();
        let groups = group_by_trigger(rel, y_idx, t_idx);
        if groups.len() > 1 {
            scan.require_ready(CapabilityId::ConditionalEnum)
                .map_err(|e| anyhow::anyhow!(e))?;
            let mut branches = 0usize;
            for (t_val, vals) in &groups {
                if vals.is_empty() {
                    continue;
                }
                let cond_label = format!("{field_label} when {trigger}={t_val}");
                let enum_label = format!("{cond_label} enum");
                let aliases: Vec<String> = vals.iter().cloned().collect();
                nodes.push(UpsertNode {
                    node_type: wiring.conditional_constraint.clone(),
                    label: cond_label.clone(),
                    source_path: doc.into(),
                    aliases: vec![],
                });
                nodes.push(UpsertNode {
                    node_type: wiring.literal_entity.clone(),
                    label: trigger.clone(),
                    source_path: doc.into(),
                    aliases: vec![],
                });
                nodes.push(UpsertNode {
                    node_type: wiring.literal_entity.clone(),
                    label: t_val.clone(),
                    source_path: doc.into(),
                    aliases: vec![],
                });
                nodes.push(UpsertNode {
                    node_type: wiring.enum_constraint.clone(),
                    label: enum_label.clone(),
                    source_path: doc.into(),
                    aliases,
                });
                edges.push(UpsertEdge {
                    from: field_label.into(),
                    edge_type: wiring.constrained_by.clone(),
                    to: cond_label.clone(),
                    source_path: doc.into(),
                });
                edges.push(UpsertEdge {
                    from: cond_label.clone(),
                    edge_type: wiring.if_field.clone(),
                    to: trigger.clone(),
                    source_path: doc.into(),
                });
                edges.push(UpsertEdge {
                    from: cond_label.clone(),
                    edge_type: wiring.if_value.clone(),
                    to: t_val.clone(),
                    source_path: doc.into(),
                });
                edges.push(UpsertEdge {
                    from: cond_label,
                    edge_type: wiring.has_constraint.clone(),
                    to: enum_label,
                    source_path: doc.into(),
                });
                branches += 1;
            }
            return Ok(format!(
                "{PATTERN_CONDITIONAL_ENUM} ({branches} branches on «{trigger}»)"
            ));
        }
    }

    scan.require_ready(CapabilityId::IndependentEnum)
        .map_err(|e| anyhow::anyhow!(e))?;
    let enum_label = format!("{field_label} enum");
    let aliases: Vec<String> = all_y.into_iter().collect();
    let n = aliases.len();
    nodes.push(UpsertNode {
        node_type: wiring.enum_constraint.clone(),
        label: enum_label.clone(),
        source_path: doc.into(),
        aliases,
    });
    edges.push(UpsertEdge {
        from: field_label.into(),
        edge_type: wiring.constrained_by.clone(),
        to: enum_label,
        source_path: doc.into(),
    });
    Ok(format!("{PATTERN_INDEPENDENT_ENUM} ({n} values)"))
}

/// Pick the trigger column with the fewest groups that still splits Y (>1 group).
fn best_single_trigger(rel: &CspTable, y_idx: usize, triggers: &[String]) -> Option<String> {
    let mut best: Option<(String, usize)> = None;
    for t in triggers {
        let t_idx = column_index(&rel.headers, t)?;
        let groups = group_by_trigger(rel, y_idx, t_idx);
        if groups.len() <= 1 {
            continue;
        }
        let score = groups.len();
        if best.as_ref().is_none_or(|(_, n)| score < *n) {
            best = Some((t.clone(), score));
        }
    }
    best.map(|(t, _)| t)
}

fn group_by_trigger(
    rel: &CspTable,
    y_idx: usize,
    t_idx: usize,
) -> BTreeMap<String, BTreeSet<String>> {
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for row in &rel.rows {
        let t_val = row.get(t_idx).cloned().unwrap_or_default();
        let y_val = row.get(y_idx).cloned().unwrap_or_default();
        if y_val.is_empty() {
            continue;
        }
        out.entry(t_val).or_default().insert(y_val);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ontology::Ontology;
    use crate::graph::store::GraphStore;
    use crate::index::store::DocIndex;
    use crate::model::Chunk;

    use crate::notebook::mirror_dir_for_doc;

    fn eval_ontology() -> Ontology {
        let s = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/eval/ontology-constraint.toml"
        ));
        Ontology::parse(s).unwrap()
    }

    fn fixture_store(dir: &Path, csv: &str, doc: &str) {
        let tables = crate::notebook::notes_root(dir).join(mirror_dir_for_doc(doc));
        std::fs::create_dir_all(&tables).unwrap();
        std::fs::write(tables.join("t.csp"), csv).unwrap();
        let idx = DocIndex::open_or_create(dir).unwrap();
        idx.write_chunks(&[Chunk {
            doc_path: doc.into(),
            location: String::new(),
            file_type: "pdf".into(),
            text: "stub".into(),
        }])
        .unwrap();
    }

    #[test]
    fn compiles_independent_enum() {
        let dir = tempfile::tempdir().unwrap();
        let doc = "test_doc.pdf";
        fixture_store(
            dir.path(),
            "Связка\tСопроводительный документ\nB\tGOST\nBF\tGOST\n",
            doc,
        );
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = eval_ontology();
        let report = tables_to_graph(
            &idx,
            &g,
            &ont,
            doc,
            &crate::notebook::notes_root(dir.path()).join(mirror_dir_for_doc(doc)),
        )
        .unwrap();
        assert!(report
            .lines
            .iter()
            .any(|l| l.contains(PATTERN_INDEPENDENT_ENUM)));
        let problem = crate::constraint_adapter::load_problem(&g, &ont, Some(doc)).unwrap();
        assert!(problem.fields.iter().any(|f| f.name == "Связка"));
    }

    #[test]
    fn compiles_conditional_when_type_splits_domain() {
        let dir = tempfile::tempdir().unwrap();
        let doc = "test_doc.pdf";
        fixture_store(
            dir.path(),
            "Обозначение типа\tНаружный диаметр\tСопроводительный документ\n41\t50\tG\n41\t63\tG\n42\t80\tG\n",
            doc,
        );
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = eval_ontology();
        tables_to_graph(
            &idx,
            &g,
            &ont,
            doc,
            &crate::notebook::notes_root(dir.path()).join(mirror_dir_for_doc(doc)),
        )
        .unwrap();
        let problem = crate::constraint_adapter::load_problem(&g, &ont, Some(doc)).unwrap();
        let d = problem
            .fields
            .iter()
            .find(|f| f.name == "Наружный диаметр")
            .unwrap();
        assert!(
            d.constraints
                .iter()
                .any(|c| matches!(c, glossa_constraint::Constraint::Conditional { .. })),
            "expected Conditional for D: {:?}",
            d.constraints
        );
    }

    #[test]
    fn recompile_replaces_stale_conditional_branches() {
        let dir = tempfile::tempdir().unwrap();
        let doc = "test_doc.pdf";
        let tables =
            crate::notebook::notes_root(dir.path()).join(mirror_dir_for_doc(doc));
        std::fs::create_dir_all(&tables).unwrap();
        let write_csp = |body: &str| {
            std::fs::write(tables.join("d.csp"), body).unwrap();
        };
        write_csp(
            "Обозначение типа\tНаружный диаметр\n41\t50\n41\t63\n42\t80\n",
        );
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[Chunk {
            doc_path: doc.into(),
            location: String::new(),
            file_type: "pdf".into(),
            text: "stub".into(),
        }])
        .unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = eval_ontology();
        tables_to_graph(&idx, &g, &ont, doc, &tables).unwrap();
        let before = g.all_nodes().unwrap().len();

        write_csp("Обозначение типа\tНаружный диаметр\n41\t50\n");
        let report = tables_to_graph(&idx, &g, &ont, doc, &tables).unwrap();
        assert!(
            report
                .lines
                .iter()
                .any(|l| l.contains("replaced:")),
            "expected replace line: {:?}",
            report.lines
        );
        let problem = crate::constraint_adapter::load_problem(&g, &ont, Some(doc)).unwrap();
        let d = problem
            .fields
            .iter()
            .find(|f| f.name == "Наружный диаметр")
            .unwrap();
        assert_eq!(
            d.constraints.len(),
            1,
            "stale conditional branch should be removed: {:?}",
            d.constraints
        );
        let after = g.all_nodes().unwrap().len();
        assert!(after <= before, "recompile should not accumulate orphan nodes");
    }
}
