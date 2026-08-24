//! `kbx synth` — synthetic gold augmentation: the INVERSE of `distil::chain_one_gold`. Distil
//! walks BACKWARD from a known (question, answer) to a grounded chain; synth walks FORWARD from
//! an already-grounded seed node to INVENT a new (question, answer) pair whose path is that
//! chain, then verify-gates it (model self-gate + a cheap adversarial leak check) before keeping
//! it. Read-only on the graph — the only write anywhere in this module is the `--out` dataset
//! file `run_synth` produces, which `kbx distil --gold <out>` and `kbx eval --dataset <out>`
//! consume unchanged (same `dataset.toml` `[[case]]` shape).
//!
//! See docs/superpowers/specs/2026-08-24-kbx-synth-spec.md for the full design.

mod gen;
mod run;

pub use gen::{parse_propose_gold, DropReason, GenOutcome, GoldProposal, Seed};
pub use run::{eligible_seed_types, run_synth, seed_pool, SynthArgs};
