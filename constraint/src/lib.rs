pub mod model;
pub mod solver;
pub mod expr;

pub use model::{Constraint, FieldConstraints, Problem, Domain, Violation, ConsistencyIssue};
pub use solver::SolveMode;
