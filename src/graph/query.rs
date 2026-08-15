//! Fuzzy read-only SQL over the reasoning graph (the `graph_query` tool).
//! Parse -> read-only gate -> locate fuzzy literals -> constrained resolution -> rewrite ->
//! execute -> chainable render. See docs/superpowers/specs/2026-08-15-graph-query-fuzzy-sql-design.md

use crate::graph::store::GraphStore;
use sqlparser::ast::{Query, SetExpr, Statement, TableFactor};
use sqlparser::dialect::SQLiteDialect;
use sqlparser::parser::Parser;

/// Tables/views the `graph_query` tool is allowed to read.
const WHITELIST: &[&str] = &["nodes", "edges", "node_validity", "edges_labeled"];

/// Parse `sql`, then gate it to a single read-only `SELECT` over whitelisted tables/views.
///
/// Rejects: writes/DDL (INSERT/UPDATE/DELETE/DROP/...), `ATTACH`, `PRAGMA`, multiple
/// statements, and any referenced table not in [`WHITELIST`] (including in subqueries and
/// set operations).
pub(crate) fn parse_readonly_select(sql: &str) -> Result<Query, String> {
    let stmts = Parser::parse_sql(&SQLiteDialect {}, sql).map_err(|e| format!("SQL parse error: {e}"))?;
    let [Statement::Query(q)] = stmts.as_slice() else {
        return Err("only a single read-only SELECT is allowed".into());
    };
    for name in referenced_tables(q) {
        if !WHITELIST.contains(&name.to_ascii_lowercase().as_str()) {
            return Err(format!(
                "table \"{name}\" is not queryable; allowed: {}",
                WHITELIST.join(", ")
            ));
        }
    }
    Ok((**q).clone())
}

/// Recursively collect every table name referenced by `q` — via `WITH`/CTE bodies, joins
/// (including parenthesized/nested joins), set operations, and derived-table subqueries — as
/// dotted identifier strings (e.g. `schema.table`).
fn referenced_tables(q: &Query) -> Vec<String> {
    let mut out = Vec::new();
    collect_from_query(q, &mut out);
    out
}

/// Walk a whole `Query`: its `WITH`/CTE clause (each CTE's own query, recursively — a CTE can
/// shadow a whitelisted name while its body reads a non-whitelisted table) and its body.
fn collect_from_query(q: &Query, out: &mut Vec<String>) {
    if let Some(with) = &q.with {
        for cte in &with.cte_tables {
            collect_from_query(&cte.query, out);
        }
    }
    collect_from_set_expr(&q.body, out);
}

fn collect_from_set_expr(expr: &SetExpr, out: &mut Vec<String>) {
    match expr {
        SetExpr::Select(select) => {
            for twj in &select.from {
                collect_from_table_factor(&twj.relation, out);
                for join in &twj.joins {
                    collect_from_table_factor(&join.relation, out);
                }
            }
        }
        SetExpr::Query(inner) => collect_from_query(inner, out),
        SetExpr::SetOperation { left, right, .. } => {
            collect_from_set_expr(left, out);
            collect_from_set_expr(right, out);
        }
        SetExpr::Values(_) | SetExpr::Insert(_) | SetExpr::Update(_) | SetExpr::Table(_) => {}
    }
}

fn collect_from_table_factor(tf: &TableFactor, out: &mut Vec<String>) {
    match tf {
        TableFactor::Table { name, .. } => {
            let dotted = name
                .0
                .iter()
                .map(|ident| ident.value.as_str())
                .collect::<Vec<_>>()
                .join(".");
            out.push(dotted);
        }
        TableFactor::Derived { subquery, .. } => collect_from_query(subquery, out),
        TableFactor::NestedJoin {
            table_with_joins, ..
        } => {
            collect_from_table_factor(&table_with_joins.relation, out);
            for join in &table_with_joins.joins {
                collect_from_table_factor(&join.relation, out);
            }
        }
        _ => {}
    }
}

/// Self-describing schema help for an empty/unclear `graph_query` call: the queryable
/// tables/views and their columns, the graph's *actual* `edge_type` and `node_type`
/// vocabularies (distinct values currently in use), and one example query.
pub(crate) fn schema_help(g: &GraphStore) -> String {
    let edge_types = g
        .run_select("SELECT DISTINCT edge_type FROM edges", 50)
        .map(|rows| rows.into_iter().filter_map(|r| r.into_iter().next()).collect::<Vec<_>>())
        .unwrap_or_default();
    let node_types = g
        .run_select("SELECT DISTINCT node_type FROM nodes", 50)
        .map(|rows| rows.into_iter().filter_map(|r| r.into_iter().next()).collect::<Vec<_>>())
        .unwrap_or_default();

    let mut out = String::new();
    out.push_str("graph_query: read-only SQL over the reasoning graph.\n\n");
    out.push_str("Queryable tables/views:\n");
    out.push_str("  nodes(id, node_type, label)\n");
    out.push_str("  edges(efrom, edge_type, eto)\n");
    out.push_str(
        "  node_validity(node_id, valid_from, valid_to, valid_from_raw, valid_to_raw)\n",
    );
    out.push_str("  edges_labeled(src_label, edge_type, dst_label, efrom, eto)\n\n");

    out.push_str("edge_type values currently in this graph:\n");
    if edge_types.is_empty() {
        out.push_str("  (none yet)\n");
    } else {
        out.push_str(&format!("  {}\n", edge_types.join(", ")));
    }

    out.push_str("\nnode_type values currently in this graph:\n");
    if node_types.is_empty() {
        out.push_str("  (none yet)\n");
    } else {
        out.push_str(&format!("  {}\n", node_types.join(", ")));
    }

    out.push_str("\nExample: SELECT label FROM nodes WHERE node_type='Fact' LIMIT 5\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::store::{Edge, Node, Provenance};

    fn prov() -> Provenance {
        Provenance {
            source_path: "a.md".into(),
            range: None,
            file_sig: None,
            origin: "auto-structural".into(),
            confidence: 1.0,
            created_at: 0,
        }
    }

    fn node_fact(id: &str, label: &str) -> Node {
        Node {
            id: id.into(),
            node_type: "Fact".into(),
            label: label.into(),
            aliases: vec![],
            prov: prov(),
        }
    }

    fn edge(from: &str, edge_type: &str, to: &str) -> Edge {
        Edge {
            from: from.into(),
            to: to.into(),
            edge_type: edge_type.into(),
            prov: prov(),
        }
    }

    #[test]
    fn schema_help_lists_tables_and_real_vocab() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        g.put_node(&node_fact("f:a", "x")).unwrap();
        g.put_node(&node_fact("f:b", "y")).unwrap();
        g.put_edge(&edge("f:a", "LEADS_TO", "f:b")).unwrap();
        let out = schema_help(&g);
        assert!(out.contains("nodes(") && out.contains("edges_labeled"));
        assert!(out.contains("LEADS_TO"), "shows real edge_type vocab: {out}");
        assert!(out.contains("Fact"), "shows real node_type vocab: {out}");
    }

    #[test]
    fn gate_accepts_select_over_whitelisted_tables() {
        assert!(parse_readonly_select("SELECT label FROM nodes WHERE node_type='Fact'").is_ok());
        assert!(parse_readonly_select("SELECT * FROM edges_labeled").is_ok());
    }

    #[test]
    fn gate_rejects_writes_ddl_and_unknown_tables() {
        for bad in [
            "DELETE FROM nodes",
            "UPDATE nodes SET label='x'",
            "INSERT INTO nodes VALUES ('a','Fact','b')",
            "DROP TABLE nodes",
            "ATTACH DATABASE 'x' AS y",
            "PRAGMA table_info(nodes)",
            "SELECT 1; SELECT 2",         // multi-statement
            "SELECT * FROM secret_table", // not whitelisted
        ] {
            assert!(parse_readonly_select(bad).is_err(), "must reject: {bad}");
        }
    }

    #[test]
    fn gate_rejects_cte_name_shadowing_and_nested_join_bypass() {
        for bad in [
            "WITH nodes AS (SELECT * FROM secret_table) SELECT * FROM nodes",
            "SELECT * FROM (nodes JOIN secret_table ON 1=1)",
        ] {
            assert!(parse_readonly_select(bad).is_err(), "must reject: {bad}");
        }
    }
}
