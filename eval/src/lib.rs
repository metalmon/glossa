//! Shared library for the dev tooling: consumed by both binaries — `kb-eval` (measure: run the agent
//! and score) and `kb-train` (build & learn: enrich the reasoning graph, optimize retrieval prompts).

pub mod backend;
pub mod corpus;
pub mod dataset;
pub mod enrich;
pub mod export_tz;
pub mod gepa;
pub mod prep;
pub mod run;
pub mod score;
pub mod trace_read;
pub mod tz;
