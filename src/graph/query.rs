//! Fuzzy read-only SQL over the reasoning graph (the `graph_query` tool).
//! Parse -> read-only gate -> locate fuzzy literals -> constrained resolution -> rewrite ->
//! execute -> chainable render. See docs/superpowers/specs/2026-08-15-graph-query-fuzzy-sql-design.md

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

/// Recursively collect every table name referenced by `q` (via joins, set operations, and
/// derived-table subqueries), as dotted identifier strings (e.g. `schema.table`).
fn referenced_tables(q: &Query) -> Vec<String> {
    let mut out = Vec::new();
    collect_from_set_expr(&q.body, &mut out);
    out
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
        SetExpr::Query(inner) => collect_from_set_expr(&inner.body, out),
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
        TableFactor::Derived { subquery, .. } => collect_from_set_expr(&subquery.body, out),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
