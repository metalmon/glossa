//! Vendored SOP core from zeroclaw (crates/zeroclaw-runtime/src/sop): the SOP.md /
//! SOP.toml format parser plus the routing state machine, copied verbatim so an SOP
//! authored here runs UNCHANGED under the zeroclaw engine in production.
//!
//! Deliberately NOT vendored (prod-runtime integration the eval doesn't need):
//! approval/checkpoint gates, cron dispatch, metrics, audit, the sqlite run-store,
//! calendar/security framing, and the 5k-line `engine.rs` orchestrator. In their
//! place the eval provides a thin [`driver`] that plugs a `run_episode` step
//! executor into [`route::resolve_next`].

pub mod condition;
pub mod driver;
pub mod parse;
pub mod prompt;
pub mod route;
pub mod rundata;
pub mod schema;
pub mod scope;
pub mod step_contract;
pub mod types;

pub use parse::{load_sop, parse_steps};
pub use route::{resolve_next, NextStep, RouteCtx};
pub use rundata::RunData;
pub use types::{Sop, SopExecutionMode, SopRun, SopStep, SopStepResult, SopStepStatus};
