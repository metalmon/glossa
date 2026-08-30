pub mod audit;
pub mod cli_fmt;
pub mod convert;
pub mod default_ignore;
pub mod extract;
pub mod glob;
pub mod graph;
pub mod grep;
pub mod http_metrics;
pub mod index;
pub mod json_util;
pub mod mcp;
pub mod mcp_auth;
pub mod model;
pub mod ontology_templates;
pub mod prompts;
pub mod query;
pub mod read;
pub mod root;
pub mod search;
pub mod session_idle;
pub mod tables;
pub mod tools;
pub mod trace;
pub mod tz_export;
pub mod walk;

#[cfg(feature = "notebook")]
pub mod notebook;

#[cfg(feature = "constraint")]
pub mod constraint_adapter;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        assert!(!version().is_empty());
    }
}
