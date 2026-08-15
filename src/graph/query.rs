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
}
