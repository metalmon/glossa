//! Fuzzy read-only SQL over the reasoning graph (the `graph_query` tool).
//! Parse -> read-only gate -> locate fuzzy literals -> constrained resolution -> rewrite ->
//! execute -> chainable render. See docs/superpowers/specs/2026-08-15-graph-query-fuzzy-sql-design.md
