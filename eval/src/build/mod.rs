//! `kbx build` stage 1 — agentic STATED fact-extraction over a single document (Task 5 of the
//! `kbx build` pipeline). Later stages (cross-doc merge, generalization, ...) live alongside
//! this module as siblings once they land.

pub mod candidates;
pub mod chunks;
pub mod extract;

pub use candidates::{candidate_pairs, CandidatePair};
pub use chunks::chunk_text;
pub use extract::{extract_doc, parse_and_validate_upsert, ExtractStats};
