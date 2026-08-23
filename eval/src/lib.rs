//! Shared library for the dev tooling: consumed by both binaries — `kb-eval` (measure: run the agent
//! and score) and `kb-train` (build & learn: enrich the reasoning graph, optimize retrieval prompts).

pub mod backend;
pub mod bridge_probe;
pub mod build;
pub mod checkpoint;
pub mod constraint_gepa_sop;
pub mod constraint_score;
pub mod constraint_synthetic;
pub mod corpus;
pub mod dataset;
pub mod dataset_toml;
pub mod distil;
pub mod enrich;
pub mod export_tz;
pub mod export_tz_constraint;
pub mod gepa;
pub mod gepa_constraint;
pub mod gepa_graph;
pub mod judge;
pub mod lab;
pub mod prep;
pub mod report;
pub mod run;
pub mod scaffold;
pub mod score;
pub mod sop;
pub mod trace_read;
pub mod train;
pub mod tz;
pub mod workspace;
