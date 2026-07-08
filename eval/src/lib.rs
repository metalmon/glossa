//! Shared library for the dev tooling: consumed by both binaries — `kb-eval` (measure: run the agent
//! and score) and `kb-train` (build & learn: enrich the reasoning graph, optimize retrieval prompts).

pub mod backend;
pub mod constraint_gepa_sop;
pub mod constraint_score;
pub mod constraint_synthetic;
pub mod corpus;
pub mod dataset;
pub mod enrich;
pub mod export_tz;
pub mod export_tz_constraint;
pub mod gepa;
pub mod gepa_constraint;
pub mod prep;
pub mod run;
pub mod score;
pub mod sop;
pub mod trace_read;
pub mod tz;
