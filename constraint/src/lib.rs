pub mod model;
pub mod solver;
pub mod expr;

pub use model::{Constraint, FieldConstraints, Problem, Domain, Violation, ConsistencyIssue};
pub use solver::{SolveMode, canon_scalar, enum_alias_matches, is_enum_regex_alias, regex_constraint_matches, scalar_str};
