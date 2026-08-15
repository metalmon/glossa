//! Fuzzy read-only SQL over the reasoning graph (the `graph_query` tool).
//! Parse -> read-only gate -> locate fuzzy literals -> constrained resolution -> rewrite ->
//! execute -> chainable render. See docs/superpowers/specs/2026-08-15-graph-query-fuzzy-sql-design.md

use crate::graph::store::GraphStore;
use sqlparser::ast::{
    BinaryOperator, Expr, JoinConstraint, JoinOperator, Query, SetExpr, Statement, TableFactor,
    Value,
};
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

/// A `col = 'lit'` string literal found in a `WHERE`/`ON` clause, classified by the column
/// it's compared against so the resolver knows what kind of graph value to fuzzy-match it to.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Literal {
    pub value: String,
    pub kind: LitKind,
    pub loc: LitLoc,
}

/// What kind of graph value a located literal should be fuzzy-resolved against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LitKind {
    /// Compared against `edge_type` — resolve against the graph's relation vocabulary.
    Relation,
    /// Compared against `label`/`src_label`/`dst_label` — resolve against node labels.
    Entity,
}

/// Which top-level boolean expression a [`LitLoc`] path starts from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExprRoot {
    /// `Select.selection` — the `WHERE` clause.
    Where,
    /// The `ON` expr of `Select.from[table_idx].joins[join_idx]`.
    JoinOn { table_idx: usize, join_idx: usize },
}

/// Which operand of a binary `Expr` to descend into (while walking `AND`/`OR`) or which
/// operand held the literal (at the leaf `=` comparison).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Side {
    Left,
    Right,
}

/// Addresses one string-literal leaf inside a `Query`'s boolean expression tree, so the same
/// node can be re-found (and mutated) in a **structurally identical clone** of the `Query`
/// during the later rewrite pass.
///
/// Addressing scheme:
/// 1. Start at `root`: either `Select.selection` (`ExprRoot::Where`) or the `ON` expr of
///    `Select.from[table_idx].joins[join_idx]` (`ExprRoot::JoinOn`).
/// 2. Walk `path` in order. Each [`Side`] entry says: the current expr must be
///    `Expr::BinaryOp { left, op: And | Or, right }` — descend into `left` if the entry is
///    `Side::Left`, into `right` if `Side::Right`. Repeat until `path` is exhausted.
/// 3. The expr now reached must be the leaf `Expr::BinaryOp { left, op: Eq, right }` that
///    held the literal. `side` says which of `left`/`right` is the literal operand (the other
///    is the column operand, left untouched).
///
/// To rewrite: replay steps 1–2 with `&mut Expr` (matching `Expr::BinaryOp` and taking
/// `&mut *left`/`&mut *right` per `Side`, identical to how [`locate_literals`] reads it), then
/// replace the operand named by `side` in the leaf `BinaryOp`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LitLoc {
    pub root: ExprRoot,
    pub path: Vec<Side>,
    pub side: Side,
}

/// Find every `col = 'lit'` equality in `q`'s `WHERE` clause and JOIN `ON` clauses, and
/// classify each by the column it's compared against ([`LitKind::Relation`] for `edge_type`,
/// [`LitKind::Entity`] for `label`/`src_label`/`dst_label`, ignored otherwise). `col LIKE
/// 'lit'` is deliberately left alone — it's already fuzzy. Recurses through `AND`/`OR` to
/// reach leaf equalities; does not descend into subqueries.
pub(crate) fn locate_literals(q: &Query) -> Vec<Literal> {
    let mut out = Vec::new();
    let SetExpr::Select(select) = &*q.body else {
        return out;
    };
    if let Some(expr) = &select.selection {
        walk_bool_tree(expr, ExprRoot::Where, &mut Vec::new(), &mut out);
    }
    for (table_idx, twj) in select.from.iter().enumerate() {
        for (join_idx, join) in twj.joins.iter().enumerate() {
            if let Some(on_expr) = join_on_expr(&join.join_operator) {
                walk_bool_tree(
                    on_expr,
                    ExprRoot::JoinOn { table_idx, join_idx },
                    &mut Vec::new(),
                    &mut out,
                );
            }
        }
    }
    out
}

/// Pull the `ON` expr out of a join operator's constraint, if it has one (`USING`/`NATURAL`
/// joins and `CROSS`/`APPLY` joins have no `Expr` to search).
fn join_on_expr(op: &JoinOperator) -> Option<&Expr> {
    let constraint = match op {
        JoinOperator::Inner(c)
        | JoinOperator::LeftOuter(c)
        | JoinOperator::RightOuter(c)
        | JoinOperator::FullOuter(c)
        | JoinOperator::Semi(c)
        | JoinOperator::LeftSemi(c)
        | JoinOperator::RightSemi(c)
        | JoinOperator::Anti(c)
        | JoinOperator::LeftAnti(c)
        | JoinOperator::RightAnti(c) => Some(c),
        JoinOperator::AsOf { constraint, .. } => Some(constraint),
        JoinOperator::CrossJoin | JoinOperator::CrossApply | JoinOperator::OuterApply => None,
    };
    match constraint {
        Some(JoinConstraint::On(expr)) => Some(expr),
        _ => None,
    }
}

/// Recurse `expr` at `root`/`path`: descend `AND`/`OR` into both children (pushing the
/// matching [`Side`] onto `path` for each), and at a leaf `=` comparison, record a
/// [`Literal`] if one side is a column and the other a single-quoted string literal that
/// classifies via [`classify_column`].
fn walk_bool_tree(expr: &Expr, root: ExprRoot, path: &mut Vec<Side>, out: &mut Vec<Literal>) {
    let Expr::BinaryOp { left, op, right } = expr else {
        return;
    };
    match op {
        BinaryOperator::And | BinaryOperator::Or => {
            path.push(Side::Left);
            walk_bool_tree(left, root, path, out);
            path.pop();
            path.push(Side::Right);
            walk_bool_tree(right, root, path, out);
            path.pop();
        }
        BinaryOperator::Eq => {
            if let Some((value, side, column)) = eq_literal_and_column(left, right) {
                if let Some(kind) = classify_column(&column) {
                    out.push(Literal {
                        value,
                        kind,
                        loc: LitLoc { root, path: path.clone(), side },
                    });
                }
            }
        }
        _ => {}
    }
}

/// If exactly one of `left`/`right` is a column reference and the other a single-quoted
/// string literal, return `(literal text, which side held the literal, column name)`.
fn eq_literal_and_column(left: &Expr, right: &Expr) -> Option<(String, Side, String)> {
    if let (Some(col), Some(lit)) = (column_name(left), literal_string(right)) {
        return Some((lit, Side::Right, col));
    }
    if let (Some(col), Some(lit)) = (column_name(right), literal_string(left)) {
        return Some((lit, Side::Left, col));
    }
    None
}

/// The bare column name from `Expr::Identifier` (`col`) or `Expr::CompoundIdentifier`
/// (`table.col` — the last segment is the column).
fn column_name(e: &Expr) -> Option<String> {
    match e {
        Expr::Identifier(ident) => Some(ident.value.clone()),
        Expr::CompoundIdentifier(idents) => idents.last().map(|i| i.value.clone()),
        _ => None,
    }
}

/// The literal text from `Expr::Value(Value::SingleQuotedString(_))`.
fn literal_string(e: &Expr) -> Option<String> {
    match e {
        Expr::Value(Value::SingleQuotedString(s)) => Some(s.clone()),
        _ => None,
    }
}

/// Map a column name to what kind of graph value its literal should resolve against.
fn classify_column(col: &str) -> Option<LitKind> {
    match col.to_ascii_lowercase().as_str() {
        "edge_type" => Some(LitKind::Relation),
        "label" | "src_label" | "dst_label" => Some(LitKind::Entity),
        _ => None,
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

    #[test]
    fn locate_classifies_relation_vs_entity_equalities() {
        let q = parse_readonly_select(
            "SELECT dst_label FROM edges_labeled WHERE src_label = 'Senica' AND edge_type = 'located in'"
        ).unwrap();
        let lits = locate_literals(&q);
        let rel = lits.iter().find(|l| matches!(l.kind, LitKind::Relation)).unwrap();
        let ent = lits.iter().find(|l| matches!(l.kind, LitKind::Entity)).unwrap();
        assert_eq!(rel.value, "located in");
        assert_eq!(ent.value, "Senica");
        // LIKE stays fuzzy already -> not collected
        let q2 = parse_readonly_select("SELECT label FROM nodes WHERE label LIKE '%Kepler%'").unwrap();
        assert!(locate_literals(&q2).is_empty());
    }
}
