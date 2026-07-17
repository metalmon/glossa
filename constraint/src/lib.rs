pub mod expr;
pub mod model;
pub mod solver;

pub use model::{ConsistencyIssue, Constraint, Domain, FieldConstraints, Problem, Violation};
pub use solver::{
    canon_scalar, enum_alias_matches, is_enum_regex_alias, regex_constraint_matches, scalar_str,
    value_subseteq, values_cover, SolveMode,
};
