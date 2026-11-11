//! The row shapes of the GEPA supervision datasets (`search.jsonl`, `read.jsonl`, `grep.jsonl`,
//! `glob.jsonl`) and the two parsers `gepa` needs to fill them.
//!
//! These are what survived `export_tz.rs`, which exported the same rows from TensorZero episodes
//! in ClickHouse. That exporter's entry point (`kb-train export-tz`) was retired with the
//! `kb-train` binary and nothing called into it afterwards — `kbx export` builds its rows from
//! the run trajectories on disk instead (`load_trajectories_for_tag`, `export_sft`). The shapes
//! outlived the source they were first filled from, so they live here under their own name.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One row in `search.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchExample {
    pub episode_id: String,
    pub case_id: Option<String>,
    pub question: String,
    pub search_query: String,
    pub gold: Vec<String>,
    pub hit: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
}

/// One row in `read.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReadExample {
    pub episode_id: String,
    pub case_id: Option<String>,
    pub question: String,
    pub search_query: String,
    pub hits: Vec<Value>,
    pub gold: Vec<String>,
    pub model_read: Option<ReadPick>,
    pub hit: bool,
    /// `search` (default) or `grep` — which tool prefilled the read context.
    #[serde(default = "default_prefill_source")]
    pub prefill_source: String,
    /// Raw grep output when `prefill_source == "grep"`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub grep_result: String,
}

fn default_prefill_source() -> String {
    "search".into()
}

/// One row in `grep.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GrepExample {
    pub episode_id: String,
    pub case_id: Option<String>,
    pub question: String,
    pub grep_pattern: String,
    pub gold: Vec<String>,
    pub hit: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
    /// Oracle-derived row from registry gold (not from an eval episode).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub synthetic: bool,
}

/// One row in `glob.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GlobExample {
    pub episode_id: String,
    pub case_id: Option<String>,
    pub question: String,
    pub glob_pattern: String,
    pub gold: Vec<String>,
    pub hit: bool,
    /// Oracle-derived row from registry gold (not from an eval episode).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub synthetic: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReadPick {
    pub path: String,
    pub n: u64,
}

/// Normalize question text for registry join (collapse whitespace, trim).
pub fn normalize_question(q: &str) -> String {
    q.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Parse `path  (N chunks)` lines from a glob tool_result body.
pub fn parse_glob_paths(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('(') {
                return None;
            }
            let path = line.split("  (").next()?.trim();
            if path.is_empty() {
                None
            } else {
                Some(path.to_string())
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_glob_paths_reads_document_lines() {
        let paths = parse_glob_paths("kb-test/doc.pdf  (12 chunks)\nother.htm  (3 chunks)");
        assert_eq!(paths, vec!["kb-test/doc.pdf", "other.htm"]);
    }

    #[test]
    fn normalize_question_collapses_whitespace() {
        assert_eq!(normalize_question("  a\t b\n c  "), "a b c");
    }
}
