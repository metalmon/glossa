//! Shared library for the dev tooling: consumed by both binaries — `kb-eval` (measure: run the agent
//! and score) and `kb-train` (build & learn: enrich the reasoning graph, optimize retrieval prompts).

pub mod backend;
pub mod bridge_probe;
pub mod build;
pub mod calibrate;
pub mod checkpoint;
pub mod connectivity;
pub mod dataset;
pub mod dataset_ops;
pub mod dataset_toml;
pub mod distil;
pub mod download;
pub mod episode;
pub mod export_tz;
pub mod finetune;
pub mod gepa;
pub mod gepa_checkpoint;
pub mod gepa_graph;
pub mod judge;
pub mod lab;
pub mod nli_check;
pub mod parallel;
pub mod reason;
pub mod report;
pub mod scaffold;
pub mod score;
pub mod train;
pub mod tz;
pub mod workspace;
