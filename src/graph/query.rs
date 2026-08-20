//! Fuzzy read-only SQL over the reasoning graph (the `sql` tool).
//! Parse -> read-only gate -> locate fuzzy literals -> constrained resolution -> rewrite ->
//! execute -> chainable render. See docs/superpowers/specs/2026-08-15-graph-query-fuzzy-sql-design.md

use crate::graph::store::GraphStore;
use crate::index::store::DocIndex;
use sqlparser::ast::{
    BinaryOperator, Expr, Function, FunctionArg, FunctionArgExpr, FunctionArgumentClause,
    FunctionArguments, GroupByExpr, HavingBound, JoinConstraint, JoinOperator, ListAggOnOverflow,
    Query, SelectItem, SetExpr, Statement, Subscript, TableFactor, Value,
};
use sqlparser::dialect::SQLiteDialect;
use sqlparser::parser::Parser;

/// Tables/views the `sql` tool is allowed to read.
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
            // Expression-level subqueries (projection / WHERE / HAVING / GROUP BY / ...) can
            // read tables that never appear in the FROM clause — walk every expression
            // position so a subquery can't smuggle a non-whitelisted table past the gate,
            // e.g. `SELECT (SELECT sql FROM sqlite_master) FROM nodes` or
            // `SELECT id FROM nodes WHERE label IN (SELECT community FROM node_meta)`.
            for item in &select.projection {
                match item {
                    SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. } => {
                        collect_from_expr(e, out)
                    }
                    SelectItem::QualifiedWildcard(..) | SelectItem::Wildcard(..) => {}
                }
            }
            if let Some(e) = &select.selection {
                collect_from_expr(e, out);
            }
            if let Some(e) = &select.prewhere {
                collect_from_expr(e, out);
            }
            if let Some(e) = &select.having {
                collect_from_expr(e, out);
            }
            if let Some(e) = &select.qualify {
                collect_from_expr(e, out);
            }
            match &select.group_by {
                GroupByExpr::All(_) => {}
                GroupByExpr::Expressions(exprs, _) => {
                    for e in exprs {
                        collect_from_expr(e, out);
                    }
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
        // Defense in depth: these factors WRAP an inner table (`PIVOT(...)`, `UNPIVOT(...)`,
        // `MATCH_RECOGNIZE(...)`) — fail closed by recursing into what they wrap instead of
        // silently letting a wrapped non-whitelisted table pass unchecked.
        TableFactor::Pivot { table, .. }
        | TableFactor::Unpivot { table, .. }
        | TableFactor::MatchRecognize { table, .. } => collect_from_table_factor(table, out),
        // Function/TableFunction/UNNEST/JsonTable/OpenJsonTable etc. carry no base-table read.
        _ => {}
    }
}

/// Recursively walk an expression tree for embedded subqueries (`Expr::Subquery`,
/// `Expr::InSubquery`, `Expr::Exists`, and bare-subquery function arguments) so their FROM
/// tables — and any of their own nested subqueries/CTEs — get whitelist-checked exactly like a
/// FROM-clause derived table. Descends through every expression shape that can nest another
/// `Expr` (binary/unary ops, function calls, CASE, casts, arrays, tuples, ...) so a subquery
/// can't hide a few levels deep and slip past the gate.
fn collect_from_expr(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::Subquery(q) | Expr::Exists { subquery: q, .. } => collect_from_query(q, out),
        Expr::InSubquery {
            expr: e, subquery, ..
        } => {
            collect_from_expr(e, out);
            collect_from_query(subquery, out);
        }
        Expr::IsFalse(e)
        | Expr::IsNotFalse(e)
        | Expr::IsTrue(e)
        | Expr::IsNotTrue(e)
        | Expr::IsNull(e)
        | Expr::IsNotNull(e)
        | Expr::IsUnknown(e)
        | Expr::IsNotUnknown(e)
        | Expr::Nested(e)
        | Expr::OuterJoin(e)
        | Expr::Prior(e)
        | Expr::UnaryOp { expr: e, .. }
        | Expr::CompositeAccess { expr: e, .. }
        | Expr::JsonAccess { value: e, .. }
        | Expr::Collate { expr: e, .. }
        | Expr::Named { expr: e, .. }
        | Expr::Cast { expr: e, .. }
        | Expr::Extract { expr: e, .. }
        | Expr::Ceil { expr: e, .. }
        | Expr::Floor { expr: e, .. } => collect_from_expr(e, out),
        Expr::IsDistinctFrom(a, b) | Expr::IsNotDistinctFrom(a, b) => {
            collect_from_expr(a, out);
            collect_from_expr(b, out);
        }
        Expr::InList { expr: e, list, .. } => {
            collect_from_expr(e, out);
            for it in list {
                collect_from_expr(it, out);
            }
        }
        Expr::InUnnest {
            expr: e, array_expr, ..
        } => {
            collect_from_expr(e, out);
            collect_from_expr(array_expr, out);
        }
        Expr::Between {
            expr: e, low, high, ..
        } => {
            collect_from_expr(e, out);
            collect_from_expr(low, out);
            collect_from_expr(high, out);
        }
        Expr::BinaryOp { left, right, .. }
        | Expr::AnyOp { left, right, .. }
        | Expr::AllOp { left, right, .. } => {
            collect_from_expr(left, out);
            collect_from_expr(right, out);
        }
        Expr::Like { expr: e, pattern, .. }
        | Expr::ILike { expr: e, pattern, .. }
        | Expr::SimilarTo { expr: e, pattern, .. }
        | Expr::RLike { expr: e, pattern, .. } => {
            collect_from_expr(e, out);
            collect_from_expr(pattern, out);
        }
        Expr::AtTimeZone {
            timestamp,
            time_zone,
        } => {
            collect_from_expr(timestamp, out);
            collect_from_expr(time_zone, out);
        }
        Expr::Convert { expr: e, styles, .. } => {
            collect_from_expr(e, out);
            for s in styles {
                collect_from_expr(s, out);
            }
        }
        Expr::Position { expr: e, r#in } => {
            collect_from_expr(e, out);
            collect_from_expr(r#in, out);
        }
        Expr::Substring {
            expr: e,
            substring_from,
            substring_for,
            ..
        } => {
            collect_from_expr(e, out);
            if let Some(f) = substring_from {
                collect_from_expr(f, out);
            }
            if let Some(f) = substring_for {
                collect_from_expr(f, out);
            }
        }
        Expr::Trim {
            expr: e,
            trim_what,
            trim_characters,
            ..
        } => {
            collect_from_expr(e, out);
            if let Some(w) = trim_what {
                collect_from_expr(w, out);
            }
            if let Some(cs) = trim_characters {
                for c in cs {
                    collect_from_expr(c, out);
                }
            }
        }
        Expr::Overlay {
            expr: e,
            overlay_what,
            overlay_from,
            overlay_for,
        } => {
            collect_from_expr(e, out);
            collect_from_expr(overlay_what, out);
            collect_from_expr(overlay_from, out);
            if let Some(f) = overlay_for {
                collect_from_expr(f, out);
            }
        }
        Expr::MapAccess { column, keys } => {
            collect_from_expr(column, out);
            for k in keys {
                collect_from_expr(&k.key, out);
            }
        }
        Expr::Function(func) => collect_from_function(func, out),
        Expr::Method(m) => {
            collect_from_expr(&m.expr, out);
            for f in &m.method_chain {
                collect_from_function(f, out);
            }
        }
        Expr::Case {
            operand,
            conditions,
            results,
            else_result,
        } => {
            if let Some(o) = operand {
                collect_from_expr(o, out);
            }
            for c in conditions {
                collect_from_expr(c, out);
            }
            for r in results {
                collect_from_expr(r, out);
            }
            if let Some(e) = else_result {
                collect_from_expr(e, out);
            }
        }
        Expr::GroupingSets(vv) | Expr::Cube(vv) | Expr::Rollup(vv) => {
            for v in vv {
                for e in v {
                    collect_from_expr(e, out);
                }
            }
        }
        Expr::Tuple(v) | Expr::Struct { values: v, .. } => {
            for e in v {
                collect_from_expr(e, out);
            }
        }
        Expr::Dictionary(fields) => {
            for f in fields {
                collect_from_expr(&f.value, out);
            }
        }
        Expr::Map(m) => {
            for entry in &m.entries {
                collect_from_expr(&entry.key, out);
                collect_from_expr(&entry.value, out);
            }
        }
        Expr::Subscript { expr: e, subscript } => {
            collect_from_expr(e, out);
            match &**subscript {
                Subscript::Index { index } => collect_from_expr(index, out),
                Subscript::Slice {
                    lower_bound,
                    upper_bound,
                    stride,
                } => {
                    if let Some(e) = lower_bound {
                        collect_from_expr(e, out);
                    }
                    if let Some(e) = upper_bound {
                        collect_from_expr(e, out);
                    }
                    if let Some(e) = stride {
                        collect_from_expr(e, out);
                    }
                }
            }
        }
        Expr::Array(arr) => {
            for e in &arr.elem {
                collect_from_expr(e, out);
            }
        }
        Expr::Interval(iv) => collect_from_expr(&iv.value, out),
        Expr::Lambda(l) => collect_from_expr(&l.body, out),
        // Leaves that cannot embed another expression or query.
        Expr::Identifier(_)
        | Expr::CompoundIdentifier(_)
        | Expr::Value(_)
        | Expr::IntroducedString { .. }
        | Expr::TypedString { .. }
        | Expr::MatchAgainst { .. }
        | Expr::Wildcard(_)
        | Expr::QualifiedWildcard(..) => {}
    }
}

fn collect_from_function(func: &Function, out: &mut Vec<String>) {
    collect_from_function_arguments(&func.parameters, out);
    collect_from_function_arguments(&func.args, out);
    if let Some(f) = &func.filter {
        collect_from_expr(f, out);
    }
    for ob in &func.within_group {
        collect_from_expr(&ob.expr, out);
    }
}

fn collect_from_function_arguments(args: &FunctionArguments, out: &mut Vec<String>) {
    match args {
        FunctionArguments::None => {}
        FunctionArguments::Subquery(q) => collect_from_query(q, out),
        FunctionArguments::List(list) => {
            for a in &list.args {
                collect_from_function_arg(a, out);
            }
            // Argument-list clauses (`ORDER BY`, `LIMIT`, `ON OVERFLOW`, `HAVING MIN/MAX`, ...)
            // sit alongside the positional args but are parsed independently, and several of
            // them carry their own `Expr` — e.g. `group_concat(id ORDER BY (SELECT ...))` or
            // `array_agg(x LIMIT (SELECT ...))`. Walk every clause that can hold an Expr.
            for clause in &list.clauses {
                collect_from_function_arg_clause(clause, out);
            }
        }
    }
}

fn collect_from_function_arg_clause(clause: &FunctionArgumentClause, out: &mut Vec<String>) {
    match clause {
        FunctionArgumentClause::OrderBy(order_exprs) => {
            for oe in order_exprs {
                collect_from_expr(&oe.expr, out);
            }
        }
        FunctionArgumentClause::Limit(e) => collect_from_expr(e, out),
        FunctionArgumentClause::OnOverflow(ListAggOnOverflow::Truncate { filler, .. }) => {
            if let Some(f) = filler {
                collect_from_expr(f, out);
            }
        }
        FunctionArgumentClause::OnOverflow(ListAggOnOverflow::Error) => {}
        FunctionArgumentClause::Having(HavingBound(_, e)) => collect_from_expr(e, out),
        // No embedded Expr: NULLS treatment flag, a bare literal Value, and a null-clause flag.
        FunctionArgumentClause::IgnoreOrRespectNulls(_)
        | FunctionArgumentClause::Separator(_)
        | FunctionArgumentClause::JsonNullClause(_) => {}
    }
}

fn collect_from_function_arg(a: &FunctionArg, out: &mut Vec<String>) {
    match a {
        FunctionArg::Named { arg, .. } | FunctionArg::Unnamed(arg) => {
            collect_from_function_arg_expr(arg, out)
        }
        FunctionArg::ExprNamed { name, arg, .. } => {
            collect_from_expr(name, out);
            collect_from_function_arg_expr(arg, out);
        }
    }
}

fn collect_from_function_arg_expr(a: &FunctionArgExpr, out: &mut Vec<String>) {
    if let FunctionArgExpr::Expr(e) = a {
        collect_from_expr(e, out);
    }
}

/// A `col = 'lit'` string literal found in a `WHERE`/`ON` clause, classified by the column
/// it's compared against so the resolver knows what kind of graph value to fuzzy-match it to.
#[derive(Debug, Clone, PartialEq, Eq)]
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
/// 0. Before matching at any step (start, or after any descent), transparently unwrap
///    `Expr::Nested(inner)` — recurse into `inner`, repeatedly if doubly-parenthesized.
///    Parens carry no addressing information: they consume no `path` entry, so a
///    parenthesized literal (e.g. `(edge_type = 'x')` or `(a = 'x') AND b = 'y'`) addresses to
///    exactly the same `path`/`side` it would without the parens.
/// 1. Start at `root`: either `Select.selection` (`ExprRoot::Where`) or the `ON` expr of
///    `Select.from[table_idx].joins[join_idx]` (`ExprRoot::JoinOn`).
/// 2. Walk `path` in order. Each [`Side`] entry says: the current expr (after unwrapping any
///    `Nested`, per step 0) must be `Expr::BinaryOp { left, op: And | Or, right }` — descend
///    into `left` if the entry is `Side::Left`, into `right` if `Side::Right`. Repeat until
///    `path` is exhausted.
/// 3. The expr now reached (after unwrapping any `Nested`, per step 0) must be the leaf
///    `Expr::BinaryOp { left, op: Eq, right }` that held the literal. `side` says which of
///    `left`/`right` is the literal operand (the other is the column operand, left untouched).
///
/// To rewrite: replay steps 0–2 with `&mut Expr` (matching `Expr::Nested`/`Expr::BinaryOp` and
/// taking `&mut *inner`/`&mut *left`/`&mut *right` per step, identical to how
/// [`locate_literals`] reads it via `walk_bool_tree`), then replace the operand named by
/// `side` in the leaf `BinaryOp`.
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// Fetch the `WHERE`/`ON` expr that a [`LitLoc::root`] points at.
fn root_expr(q: &Query, root: ExprRoot) -> Option<&Expr> {
    let SetExpr::Select(select) = &*q.body else {
        return None;
    };
    match root {
        ExprRoot::Where => select.selection.as_ref(),
        ExprRoot::JoinOn { table_idx, join_idx } => {
            let twj = select.from.get(table_idx)?;
            let join = twj.joins.get(join_idx)?;
            join_on_expr(&join.join_operator)
        }
    }
}

/// Unwrap `Expr::Nested` down to the first non-parenthesized expr, mirroring the step-0 rule
/// documented on [`LitLoc`].
fn unwrap_nested(mut e: &Expr) -> &Expr {
    while let Expr::Nested(inner) = e {
        e = inner;
    }
    e
}

/// Walk `path` from `root_expr` (per the addressing scheme documented on [`LitLoc`]) and return
/// the leaf `Expr::BinaryOp` reached.
fn navigate<'a>(root_expr: &'a Expr, path: &[Side]) -> Option<&'a Expr> {
    let mut cur = unwrap_nested(root_expr);
    for side in path {
        let Expr::BinaryOp { left, right, .. } = cur else {
            return None;
        };
        cur = unwrap_nested(match side {
            Side::Left => left,
            Side::Right => right,
        });
    }
    Some(cur)
}

/// Mutable twin of [`root_expr`] — fetch the `WHERE`/`ON` expr a [`LitLoc::root`] points at, for
/// in-place rewriting.
fn root_expr_mut(q: &mut Query, root: ExprRoot) -> Option<&mut Expr> {
    let SetExpr::Select(select) = &mut *q.body else {
        return None;
    };
    match root {
        ExprRoot::Where => select.selection.as_mut(),
        ExprRoot::JoinOn { table_idx, join_idx } => {
            let twj = select.from.get_mut(table_idx)?;
            let join = twj.joins.get_mut(join_idx)?;
            join_on_expr_mut(&mut join.join_operator)
        }
    }
}

/// Mutable twin of [`join_on_expr`].
fn join_on_expr_mut(op: &mut JoinOperator) -> Option<&mut Expr> {
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

/// Mutable twin of [`unwrap_nested`].
fn unwrap_nested_mut(e: &mut Expr) -> &mut Expr {
    match e {
        Expr::Nested(inner) => unwrap_nested_mut(inner),
        _ => e,
    }
}

/// Mutable twin of [`navigate`] — replay the addressing scheme documented on [`LitLoc`] over a
/// `&mut Expr` so the leaf `BinaryOp` can be rewritten in place.
fn navigate_mut<'a>(root_expr: &'a mut Expr, path: &[Side]) -> Option<&'a mut Expr> {
    let mut cur = unwrap_nested_mut(root_expr);
    for side in path {
        let Expr::BinaryOp { left, right, .. } = cur else {
            return None;
        };
        cur = unwrap_nested_mut(match side {
            Side::Left => left,
            Side::Right => right,
        });
    }
    Some(cur)
}

/// Substitute each located literal in a clone of `q` with its resolved real value and
/// re-serialize to SQL:
/// - [`LitKind::Relation`]: the operand is replaced verbatim with the resolved `edge_type`
///   string (e.g. `edge_type = 'LOCATED_IN'`).
/// - [`LitKind::Entity`]: the leaf equality is *relaxed* to a `LIKE` (`Expr::Like`, SQLite
///   renders it as the `LIKE` keyword) against `%<resolved>%`, so the label view still matches on
///   substring. [`resolve_assignment`] resolves entities to their original text today, so the
///   pattern carries the literal's text; the relaxation from `=` to `LIKE` is the semantic change.
///
/// Each literal is re-found in the clone via its recorded [`LitLoc`] (mirroring
/// [`locate_literals`]'s addressing exactly) before being mutated; since every edit replaces an
/// operand in place rather than adding/removing tree nodes, no path is invalidated by an earlier
/// edit and the order literals are processed in doesn't matter.
pub(crate) fn rewrite(q: &Query, chosen: &[(Literal, String, f32)]) -> String {
    let mut query = q.clone();
    for (lit, resolved, _score) in chosen {
        let Some(root) = root_expr_mut(&mut query, lit.loc.root) else {
            continue;
        };
        let Some(leaf) = navigate_mut(root, &lit.loc.path) else {
            continue;
        };
        let Expr::BinaryOp { left, right, .. } = leaf else {
            continue;
        };
        match lit.kind {
            LitKind::Relation => {
                let operand = match lit.loc.side {
                    Side::Left => left,
                    Side::Right => right,
                };
                **operand = Expr::Value(Value::SingleQuotedString(resolved.clone()));
            }
            LitKind::Entity => {
                // The non-literal operand (the column) survives untouched into the new `Like`.
                let column_expr = match lit.loc.side {
                    Side::Left => (**right).clone(),
                    Side::Right => (**left).clone(),
                };
                let token = resolved.as_str();
                *leaf = Expr::Like {
                    negated: false,
                    any: false,
                    expr: Box::new(column_expr),
                    pattern: Box::new(Expr::Value(Value::SingleQuotedString(format!("%{token}%")))),
                    escape_char: None,
                };
            }
        }
    }
    format!("{query}")
}

/// The column name a located `lit` was compared against (the operand of its leaf `=` that
/// isn't the literal itself), re-derived by replaying `lit.loc` over `q`.
fn literal_column_name(q: &Query, lit: &Literal) -> Option<String> {
    let root = root_expr(q, lit.loc.root)?;
    let leaf = navigate(root, &lit.loc.path)?;
    let Expr::BinaryOp { left, right, .. } = leaf else {
        return None;
    };
    // The literal held `side`; the column is the other operand.
    let column_expr = match lit.loc.side {
        Side::Left => right,
        Side::Right => left,
    };
    column_name(column_expr)
}

/// For each located `lit`, its chosen real graph value + score, jointly consistent with real
/// edges where a relation literal shares a `WHERE`/`ON` group with `src_label`/`dst_label`
/// entity literals.
///
/// Grouping: literals sharing the same [`ExprRoot`] (the same `WHERE` clause, or the same JOIN's
/// `ON` clause) form one group — good enough for the single-`edges_labeled`-reference case this
/// resolver targets; it does not attempt cross-group joint optimization.
///
/// Within a group that has a `Relation` literal alongside a `src_label`/`dst_label` `Entity`
/// literal, relation candidates (ranked by [`relation_candidates`]) are filtered by
/// [`GraphStore::edge_exists_like`] — only relations backed by a real edge between the (fuzzy,
/// `LIKE '%text%'`) entity endpoints survive — and the best surviving (i.e. first, since
/// candidates are already best-first) candidate is chosen; if none survive, falls back to the
/// top lexical candidate. Entity literals always resolve to their original text (fuzzy `LIKE`
/// matching is deferred to query rewrite). Literals outside any constrained group take their
/// top candidate unconstrained.
pub(crate) fn resolve_assignment(
    g: &GraphStore,
    q: &Query,
    lits: &[Literal],
) -> Vec<(Literal, String, f32)> {
    let edge_vocab = g.edge_type_vocab().unwrap_or_default();

    struct LitInfo<'a> {
        lit: &'a Literal,
        column: Option<String>,
        candidates: Vec<(String, f32)>,
    }
    let infos: Vec<LitInfo> = lits
        .iter()
        .map(|lit| {
            let column = literal_column_name(q, lit);
            let candidates = match lit.kind {
                LitKind::Relation => relation_candidates(&lit.value, &edge_vocab, 5),
                // Constraint checks (and the eventual rewrite) match entities via
                // `LIKE '%text%'` against the label view, so the candidate *is* the original
                // text — see the module doc on resolve_assignment. `entity_candidates` is still
                // consulted for its rank-based score (how confidently the graph resolves this
                // text to a real node), but its node-id values are not used as the assignment.
                LitKind::Entity => {
                    let score =
                        entity_candidates(g, &lit.value, 5).first().map(|(_, s)| *s).unwrap_or(1.0);
                    vec![(lit.value.clone(), score)]
                }
            };
            LitInfo { lit, column, candidates }
        })
        .collect();

    let mut out = Vec::with_capacity(infos.len());
    let mut handled = vec![false; infos.len()];

    for i in 0..infos.len() {
        if handled[i] {
            continue;
        }
        let group_idx: Vec<usize> =
            (0..infos.len()).filter(|&j| infos[j].lit.loc.root == infos[i].lit.loc.root).collect();
        for &j in &group_idx {
            handled[j] = true;
        }

        let is_col = |j: usize, name: &str| {
            matches!(infos[j].lit.kind, LitKind::Entity)
                && infos[j].column.as_deref().map(|c| c.eq_ignore_ascii_case(name)).unwrap_or(false)
        };
        let rel_idx =
            group_idx.iter().copied().find(|&j| matches!(infos[j].lit.kind, LitKind::Relation));
        let src_idx = group_idx.iter().copied().find(|&j| is_col(j, "src_label"));
        let dst_idx = group_idx.iter().copied().find(|&j| is_col(j, "dst_label"));

        if let Some(rel_j) = rel_idx {
            if src_idx.is_some() || dst_idx.is_some() {
                let src_like = src_idx.map(|si| format!("%{}%", infos[si].lit.value));
                let dst_like = dst_idx.map(|di| format!("%{}%", infos[di].lit.value));
                let survivor = infos[rel_j].candidates.iter().find(|(rel_val, _)| {
                    g.edge_exists_like(src_like.as_deref(), Some(rel_val), dst_like.as_deref())
                        .unwrap_or(false)
                });
                let (rel_value, rel_score) = survivor.cloned().unwrap_or_else(|| {
                    infos[rel_j]
                        .candidates
                        .first()
                        .cloned()
                        .unwrap_or_else(|| (infos[rel_j].lit.value.clone(), 0.0))
                });
                out.push((infos[rel_j].lit.clone(), rel_value, rel_score));

                for &j in &group_idx {
                    if j == rel_j {
                        continue;
                    }
                    let (val, score) = infos[j]
                        .candidates
                        .first()
                        .cloned()
                        .unwrap_or_else(|| (infos[j].lit.value.clone(), 0.0));
                    out.push((infos[j].lit.clone(), val, score));
                }
                continue;
            }
        }

        // Unlinked group (no relation+entity pairing to constrain): each literal takes its own
        // top candidate.
        for &j in &group_idx {
            let (val, score) = infos[j]
                .candidates
                .first()
                .cloned()
                .unwrap_or_else(|| (infos[j].lit.value.clone(), 0.0));
            out.push((infos[j].lit.clone(), val, score));
        }
    }

    out
}

/// Recurse `expr` at `root`/`path`: transparently unwrap `Expr::Nested` (parens carry no
/// addressing information — a parenthesized literal addresses to the same logical position it
/// would without the parens), descend `AND`/`OR` into both children (pushing the matching
/// [`Side`] onto `path` for each), and at a leaf `=` comparison, record a [`Literal`] if one
/// side is a column and the other a single-quoted string literal that classifies via
/// [`classify_column`].
fn walk_bool_tree(expr: &Expr, root: ExprRoot, path: &mut Vec<Side>, out: &mut Vec<Literal>) {
    if let Expr::Nested(inner) = expr {
        walk_bool_tree(inner, root, path, out);
        return;
    }
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

/// Normalize a relation/entity name for fuzzy comparison: lowercase, and split on `_`/space
/// into tokens (dropping empty tokens from runs of separators).
fn normalize_tokens(s: &str) -> Vec<String> {
    s.to_ascii_lowercase()
        .split(|c: char| c == '_' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

/// Levenshtein edit distance between two strings, counted in Unicode scalar values.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (alen, blen) = (a.len(), b.len());
    if alen == 0 {
        return blen;
    }
    if blen == 0 {
        return alen;
    }
    let mut prev: Vec<usize> = (0..=blen).collect();
    let mut cur: Vec<usize> = vec![0; blen + 1];
    for i in 1..=alen {
        cur[0] = i;
        for j in 1..=blen {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[blen]
}

/// Similarity score in `[0.0, 1.0]` between `query` and a `vocab` entry: the max of
/// exact/substring containment (1.0), token-Jaccard overlap of `_`/space-split tokens, and
/// `1 - levenshtein(a, b) / max(len(a), len(b))`. Comparison is case-insensitive throughout.
fn relation_similarity(query: &str, candidate: &str) -> f32 {
    let q_norm = query.to_ascii_lowercase();
    let c_norm = candidate.to_ascii_lowercase();
    if q_norm == c_norm || q_norm.contains(&c_norm) || c_norm.contains(&q_norm) {
        return 1.0;
    }

    let q_tokens = normalize_tokens(query);
    let c_tokens = normalize_tokens(candidate);
    let jaccard = if q_tokens.is_empty() || c_tokens.is_empty() {
        0.0
    } else {
        let q_set: std::collections::HashSet<&String> = q_tokens.iter().collect();
        let c_set: std::collections::HashSet<&String> = c_tokens.iter().collect();
        let inter = q_set.intersection(&c_set).count();
        let union = q_set.union(&c_set).count();
        if union == 0 { 0.0 } else { inter as f32 / union as f32 }
    };

    let max_len = q_norm.chars().count().max(c_norm.chars().count()).max(1);
    let edit_score = 1.0 - (levenshtein(&q_norm, &c_norm) as f32 / max_len as f32);

    jaccard.max(edit_score).max(0.0)
}

/// Top-`k` real relations from `vocab` ranked by similarity to `query` (case-insensitive;
/// exact/substring match scores 1.0, otherwise the max of token-Jaccard overlap and
/// `1 - levenshtein/maxlen`). Used to map an invented/mis-typed `edge_type` literal onto the
/// graph's actual relation vocabulary.
fn relation_candidates(query: &str, vocab: &[String], k: usize) -> Vec<(String, f32)> {
    let mut scored: Vec<(String, f32)> = vocab
        .iter()
        .map(|v| (v.clone(), relation_similarity(query, v)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

/// Top-`k` node ids resolving `name` via [`GraphStore::resolve`] (already rank-ordered, best
/// first), assigned descending scores by rank.
fn entity_candidates(g: &GraphStore, name: &str, k: usize) -> Vec<(String, f32)> {
    let ids = g.resolve(name).unwrap_or_default();
    ids.into_iter()
        .take(k)
        .enumerate()
        .map(|(i, id)| (id, (1.0 - i as f32 * 0.1).max(0.01)))
        .collect()
}

/// Self-describing schema help for an empty/unclear `sql` call: the queryable
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
    out.push_str("sql: read-only SQL over the reasoning graph.\n\n");
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

/// Column names, from `select_columns`, that carry a graph node id — the trigger for rendering
/// a `glossary`-style handle instead of the raw cell text in [`run`]'s chainable render.
const ID_COLUMNS: &[&str] = &["id", "efrom", "eto", "node_id"];

/// Run a fuzzy read-only `SELECT` end-to-end: gate (`parse_readonly_select`) -> locate fuzzy
/// literals -> resolve them against the real graph -> rewrite to concrete SQL -> execute
/// (`run_select`, capped at 50 rows) -> chainable render, with a transparency note prepended for
/// every literal substitution the resolver made. Empty/whitespace `sql` returns [`schema_help`]
/// instead of running anything; a query with no fuzzy literals runs verbatim (it's already
/// concrete). If the best assignment yields no rows, that's still rendered (as "no rows") rather
/// than treated as an error — the note above it explains what was substituted.
pub fn run(g: &GraphStore, idx: &DocIndex, sql: &str) -> String {
    if sql.trim().is_empty() {
        return schema_help(g);
    }
    let q = match parse_readonly_select(sql) {
        Ok(q) => q,
        Err(e) => return format!("{e}\n\n{}", schema_help(g)),
    };

    let lits = locate_literals(&q);
    let (real_sql, notes): (String, Vec<String>) = if lits.is_empty() {
        (sql.to_string(), Vec::new())
    } else {
        let chosen = resolve_assignment(g, &q, &lits);
        let real_sql = rewrite(&q, &chosen);
        let notes = chosen
            .iter()
            .filter_map(|(lit, resolved, score)| substitution_note(lit, resolved, *score))
            .collect();
        (real_sql, notes)
    };

    let rows = match g.run_select(&real_sql, 50) {
        Ok(rows) => rows,
        Err(e) => return format!("query error: {e}\n\n{}", schema_help(g)),
    };
    let cols = match g.select_columns(&real_sql) {
        Ok(cols) => cols,
        Err(e) => return format!("query error: {e}\n\n{}", schema_help(g)),
    };

    let body = render_rows(idx, g, &cols, &rows);
    if notes.is_empty() { body } else { format!("{}\n{body}", notes.join("\n")) }
}

/// One transparency-note line for a resolved literal, or `None` when nothing about the query's
/// meaning actually moved (an exact relation match). Entity literals always get a note: their
/// leaf `=` is always relaxed to a fuzzy `LIKE` by [`rewrite`], which changes matching semantics
/// even when the displayed text doesn't change — never a silent swap.
fn substitution_note(lit: &Literal, resolved: &str, score: f32) -> Option<String> {
    match lit.kind {
        LitKind::Relation => {
            if resolved == lit.value {
                None
            } else {
                Some(format!(
                    "note: edge_type \"{}\" -> \"{}\" (sim {score:.2})",
                    lit.value, resolved
                ))
            }
        }
        LitKind::Entity => {
            // Mirrors the `LIKE` pattern `rewrite` builds (the resolved text).
            Some(format!("note: \"{}\" -> LIKE \"%{resolved}%\"", lit.value))
        }
    }
}

/// Chainable render of `rows` (columns aligned to `cols`, from `select_columns`): a column whose
/// NAME is id-bearing (`id`/`efrom`/`eto`/`node_id`, per [`ID_COLUMNS`]) renders as a
/// `glossary`-style node handle — `id  [node_type]  label   — read <path>#n`; every other column
/// prints as its raw cell text (aggregates/labels have nothing to ground). One line per row.
fn render_rows(idx: &DocIndex, g: &GraphStore, cols: &[String], rows: &[Vec<String>]) -> String {
    if rows.is_empty() {
        return "(no rows)".to_string();
    }
    rows.iter()
        .map(|row| {
            // Render each cell: id-columns become glossary handles (which already embed the node
            // label); everything else prints raw.
            let cells: Vec<(bool, String)> = row
                .iter()
                .enumerate()
                .map(|(i, val)| {
                    let is_id_col = cols
                        .get(i)
                        .map(|c| ID_COLUMNS.iter().any(|n| c.eq_ignore_ascii_case(n)))
                        .unwrap_or(false);
                    if is_id_col && !val.is_empty() {
                        (true, render_handle(idx, g, val))
                    } else {
                        (false, val.clone())
                    }
                })
                .collect();
            // Drop a standalone label-column cell that a sibling handle on this row already spells
            // out, so `SELECT id, label` doesn't print the label twice.
            let handles: Vec<&str> =
                cells.iter().filter(|(h, _)| *h).map(|(_, t)| t.as_str()).collect();
            cells
                .iter()
                .enumerate()
                .filter(|(i, (is_handle, text))| {
                    if *is_handle || text.is_empty() {
                        return true;
                    }
                    let is_label_col = cols.get(*i).is_some_and(|c| {
                        matches!(
                            c.to_ascii_lowercase().as_str(),
                            "label" | "src_label" | "dst_label"
                        )
                    });
                    !(is_label_col && handles.iter().any(|h| h.contains(text.as_str())))
                })
                .map(|(_, (_, text))| text.clone())
                .collect::<Vec<_>>()
                .join("  ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Render a node id as its `glossary`-style handle: `id  [node_type]  label   — read <path>#n`.
/// Falls back to the bare id text when it doesn't resolve to a real node (e.g. a stray/NULL join
/// result) — there's nothing to ground.
fn render_handle(idx: &DocIndex, g: &GraphStore, id: &str) -> String {
    match g.get_node(id) {
        Ok(Some(node)) => {
            format!(
                "{}  [{}]  {}{}",
                node.id,
                node.node_type,
                node.label,
                crate::tools::read_anchor(idx, g, &node.id)
            )
        }
        _ => id.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::store::{Edge, Node, NodeValidity, Provenance};

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
    fn gate_rejects_expression_level_subquery_bypass() {
        for bad in [
            // Subquery in the projection reads a non-whitelisted table.
            "SELECT (SELECT sql FROM sqlite_master LIMIT 1) FROM nodes",
            // InSubquery in WHERE reads a non-whitelisted table.
            "SELECT id FROM nodes WHERE label IN (SELECT community FROM node_meta)",
        ] {
            assert!(parse_readonly_select(bad).is_err(), "must reject: {bad}");
        }
        // Positive control: a subquery over a WHITELISTED table must still be allowed.
        assert!(
            parse_readonly_select("SELECT id FROM nodes WHERE id IN (SELECT efrom FROM edges)")
                .is_ok()
        );
    }

    #[test]
    fn gate_rejects_subquery_in_function_arg_order_by_clause() {
        // `group_concat(... ORDER BY ...)` is a real SQLite aggregate-function clause; its
        // ORDER BY expression can itself hide a subquery over a non-whitelisted table.
        assert!(parse_readonly_select(
            "SELECT group_concat(id ORDER BY (SELECT sql FROM sqlite_master LIMIT 1)) FROM nodes"
        )
        .is_err());
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

    #[test]
    fn locate_unwraps_a_single_parenthesized_equality() {
        let q = parse_readonly_select(
            "SELECT dst_label FROM edges_labeled WHERE (edge_type = 'located in')",
        )
        .unwrap();
        let lits = locate_literals(&q);
        let rel = lits.iter().find(|l| matches!(l.kind, LitKind::Relation)).unwrap();
        assert_eq!(rel.value, "located in");
    }

    #[test]
    fn locate_finds_a_literal_in_a_join_on_clause() {
        // The equality lives in a JOIN ... ON, not WHERE — it must still be located, addressed
        // via ExprRoot::JoinOn (from[0].joins[0]) so `rewrite` can re-find it in the clone.
        let q = parse_readonly_select(
            "SELECT n.label FROM nodes n JOIN edges_labeled e ON e.edge_type = 'located in'",
        )
        .unwrap();
        let lits = locate_literals(&q);
        let rel = lits.iter().find(|l| matches!(l.kind, LitKind::Relation)).unwrap();
        assert_eq!(rel.value, "located in");
        assert_eq!(
            rel.loc,
            LitLoc {
                root: ExprRoot::JoinOn { table_idx: 0, join_idx: 0 },
                path: vec![],
                side: Side::Right,
            },
        );
    }

    #[test]
    fn relation_candidates_map_fuzzy_to_real_vocab() {
        let vocab = vec!["LEADS_TO".to_string(), "LOCATED_IN".to_string(), "CREATED_BY".to_string()];
        let top = relation_candidates("located in", &vocab, 2);
        assert_eq!(top[0].0, "LOCATED_IN", "best match is the real relation: {top:?}");
        // an invented name still lands on the closest real one, with a lower score
        let top2 = relation_candidates("PROPOSED_BY", &vocab, 1);
        assert_eq!(top2.len(), 1);
        assert!(top2[0].1 < top[0].1);
    }

    #[test]
    fn resolution_prefers_the_assignment_backed_by_a_real_edge() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        g.put_node(&node_fact("f:senica", "Senica")).unwrap();
        g.put_node(&node_fact("f:distr", "Senica District")).unwrap();
        g.put_edge(&edge("f:senica", "LOCATED_IN", "f:distr")).unwrap();
        // vocab also has a lexically-closer-but-edgeless relation
        g.put_node(&node_fact("f:x", "x")).unwrap();
        g.put_edge(&edge("f:x", "LOCATION_OF", "f:x")).unwrap();
        let q = parse_readonly_select(
            "SELECT dst_label FROM edges_labeled WHERE src_label='Senica' AND edge_type='located'",
        )
        .unwrap();
        let lits = locate_literals(&q);
        let chosen = resolve_assignment(&g, &q, &lits);
        let rel = chosen.iter().find(|(l, _, _)| matches!(l.kind, LitKind::Relation)).unwrap();
        assert_eq!(
            rel.1, "LOCATED_IN",
            "picks the relation that actually connects Senica, not the lexically-nearest edgeless one"
        );
    }

    #[test]
    fn resolution_edge_constraint_overrides_lexical_rank() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        // Real edge for Senica -> Senica District uses a relation name that is lexically FAR
        // from the query word 'located'.
        g.put_node(&node_fact("f:senica", "Senica")).unwrap();
        g.put_node(&node_fact("f:distr", "Senica District")).unwrap();
        g.put_edge(&edge("f:senica", "SITED_IN", "f:distr")).unwrap();
        // LOCATED_NEAR is in the vocab (so it's a relation_candidates hit) and is the lexical
        // top match for 'located' (substring), but it only connects unrelated nodes -- it is
        // edgeless between Senica and its district.
        g.put_node(&node_fact("f:p", "p")).unwrap();
        g.put_node(&node_fact("f:q", "q")).unwrap();
        g.put_edge(&edge("f:p", "LOCATED_NEAR", "f:q")).unwrap();

        let q = parse_readonly_select(
            "SELECT dst_label FROM edges_labeled WHERE src_label='Senica' AND edge_type='located'",
        )
        .unwrap();
        let lits = locate_literals(&q);

        // Sanity: LOCATED_NEAR really is the lexical top match, SITED_IN really is lower --
        // otherwise this test wouldn't be exercising the override at all.
        let vocab = g.edge_type_vocab().unwrap();
        let ranked = relation_candidates("located", &vocab, vocab.len());
        let top = &ranked[0];
        assert_eq!(top.0, "LOCATED_NEAR", "fixture invalid: LOCATED_NEAR must be the lexical top match: {ranked:?}");
        let sited_score = ranked.iter().find(|(r, _)| r == "SITED_IN").unwrap().1;
        assert!(sited_score < top.1, "fixture invalid: SITED_IN must rank lower than LOCATED_NEAR: {ranked:?}");

        let chosen = resolve_assignment(&g, &q, &lits);
        let rel = chosen.iter().find(|(l, _, _)| matches!(l.kind, LitKind::Relation)).unwrap();
        assert_eq!(
            rel.1, "SITED_IN",
            "edge-backed SITED_IN must win over lexically-top but edgeless LOCATED_NEAR: {chosen:?}"
        );
    }

    #[test]
    fn rewrite_substitutes_relation_and_relaxes_entity_to_like() {
        let q = parse_readonly_select(
            "SELECT dst_label FROM edges_labeled WHERE src_label='Senica' AND edge_type='located'",
        )
        .unwrap();
        // Drive the test through the real addressing produced by `locate_literals`, rather than
        // hand-building `LitLoc`, so the test proves the locate -> rewrite round-trip.
        let lits = locate_literals(&q);
        let rel = lits.iter().find(|l| matches!(l.kind, LitKind::Relation)).unwrap().clone();
        let ent = lits.iter().find(|l| matches!(l.kind, LitKind::Entity)).unwrap().clone();
        let chosen = vec![(rel, "LOCATED_IN".to_string(), 0.8f32), (ent, "Senica".to_string(), 0.9f32)];

        let sql = rewrite(&q, &chosen);
        assert!(sql.contains("edge_type = 'LOCATED_IN'"), "sql: {sql}");
        assert!(sql.to_lowercase().contains("src_label like '%senica%'"), "sql: {sql}");
    }

    #[test]
    fn locate_unwraps_a_parenthesized_operand_of_and() {
        let q = parse_readonly_select(
            "SELECT dst_label FROM edges_labeled WHERE (src_label = 'Senica') AND edge_type = 'x'",
        )
        .unwrap();
        let lits = locate_literals(&q);
        let ent = lits.iter().find(|l| matches!(l.kind, LitKind::Entity)).unwrap();
        let rel = lits.iter().find(|l| matches!(l.kind, LitKind::Relation)).unwrap();
        assert_eq!(ent.value, "Senica");
        assert_eq!(rel.value, "x");
    }

    #[test]
    fn run_answers_when_first_via_order_by_valid_from() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("g.md"), b"# H\nx\n").unwrap();
        crate::index::store::index_dir(dir.path(), true).unwrap();
        let idx = crate::index::store::DocIndex::open_or_create(dir.path()).unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        for (id, label, vf) in [
            ("f:early", "Heliocentrism was proposed as early as the 3rd century BC", "0001"),
            ("f:late", "A predictive heliocentric system was developed later", "1543"),
        ] {
            g.put_node(&node_fact(id, label)).unwrap();
            g.upsert_validity(
                id,
                &NodeValidity {
                    valid_from: Some(crate::graph::temporal::normalize_from(vf).unwrap()),
                    valid_to: None,
                    valid_from_raw: None,
                    valid_to_raw: None,
                },
            )
            .unwrap();
        }
        let out = crate::graph::query::run(
            &g,
            &idx,
            "SELECT n.id, n.label FROM nodes n JOIN node_validity v ON v.node_id=n.id \
             WHERE n.node_type='Fact' ORDER BY v.valid_from ASC LIMIT 1",
        );
        assert!(out.contains("3rd century BC"), "deterministic earliest, not the model's guess: {out}");
        assert!(
            crate::graph::query::run(&g, &idx, "   ").contains("nodes("),
            "empty -> schema"
        );
    }

    #[test]
    fn render_does_not_repeat_the_label_when_selecting_id_and_label() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("g.md"), b"# H\nx\n").unwrap();
        crate::index::store::index_dir(dir.path(), true).unwrap();
        let idx = crate::index::store::DocIndex::open_or_create(dir.path()).unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        g.put_node(&node_fact("f:k", "Kepler orbit law")).unwrap();
        // `id` renders as a glossary handle that already spells out the label; the standalone
        // `label` column must not print it a second time on the same line.
        let out = crate::graph::query::run(&g, &idx, "SELECT id, label FROM nodes");
        assert_eq!(
            out.matches("Kepler orbit law").count(),
            1,
            "label shown once, not duplicated: {out}"
        );
    }

    #[test]
    fn query_error_carries_the_schema_so_the_model_can_self_correct() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("g.md"), b"# H\nx\n").unwrap();
        crate::index::store::index_dir(dir.path(), true).unwrap();
        let idx = crate::index::store::DocIndex::open_or_create(dir.path()).unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        g.put_node(&node_fact("f:a", "x")).unwrap();
        // `src_label` lives on the edges_labeled VIEW, not on `edges` — the exact mistake the weak
        // reader made 12 times. The error must hand back the schema (which names edges_labeled) so
        // the next attempt can correct instead of blindly re-issuing the same broken query.
        let out = crate::graph::query::run(
            &g,
            &idx,
            "SELECT dst_label FROM edges WHERE src_label LIKE '%x%'",
        );
        assert!(out.contains("query error"), "surfaces the real error: {out}");
        assert!(out.contains("edges_labeled"), "appends the schema: {out}");
    }
}
