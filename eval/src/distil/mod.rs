//! `kbx distil` — graph densification (default mode) plus a retained synthetic-gold generator
//! (opt-in `--emit-golds`).
//!
//! **Default: densify.** A `[distil]`-configured STRONG model walks the whole corpus chunk by
//! chunk ([`densify_doc`]), sees what the weak `build`+`reason` layer already grounded to each
//! chunk, and writes via `graph_upsert` whatever grounded terminal or query-side reasoning is
//! genuinely missing — never touching what is already there. Every node/edge this pass writes is
//! stamped `origin = "distil"` (`ops::graph_upsert`'s origin param), separable from the weak
//! `"agent"` layer for ablation. Checkpointed/resumable per document
//! (`distil:{doc}` marks), like `build`'s extract stage.
//!
//! **`--emit-golds <file>`: the retained gold generator.** The former default `kbx distil`
//! behavior (genuine knowledge distillation, the INVERSE of `reason::chain_one_seed`): reason
//! walks BACKWARD from a grounded terminal to synthesize the query-side reasoning layer; this mode
//! walks FORWARD from an already-grounded seed node to INVENT a new (question, answer) pair whose
//! path is that chain, then verify-gates it (model self-gate + a cheap adversarial leak check)
//! before keeping it. Read-only on the graph — the only write this mode makes is the `--emit-golds`
//! dataset file, which `kbx eval --dataset <file>` consumes unchanged (same `dataset.toml`
//! `[[case]]` shape). Golds are still useful for `train`/`eval` datasets, so this mode is kept, not
//! deleted, just demoted from the default.
//!
//! `distil::run` dispatches between the two modes purely on whether `--emit-golds` was given (see
//! [`distil_mode`]); they share the `[distil]` strong-model endpoint but read their OWN system
//! prompt file — densify reads `distil.md`, the gold generator reads `distil_golds.md` — and
//! differ in what they write.
//!
//! See docs/superpowers/specs/2026-08-28-kbx-distil-densification-design.md for the densify design,
//! and docs/superpowers/specs/2026-08-24-kbx-synth-spec.md for the original gold-generator design
//! (written before this module was renamed from `synth` to `distil`).

pub(crate) mod aliases;
mod densify;
mod gen;
mod run;

pub use densify::{densify_doc, DensifyStats};
pub use gen::{parse_propose_gold, DropReason, GenOutcome, GoldProposal, Seed};
pub use run::{
    distil_mode, eligible_seed_types, run, run_densify, run_distil, seed_pool, DistilArgs, Mode,
};
