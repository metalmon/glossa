//! `kbx build` stage 1 — agentic STATED fact-extraction over a single document (Task 5 of the
//! `kbx build` pipeline). Later stages (cross-doc merge, generalization, ...) live alongside
//! this module as siblings once they land.

pub mod extract;

pub use extract::{extract_doc, parse_and_validate_upsert, ExtractStats};
