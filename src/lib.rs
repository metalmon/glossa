pub mod audit;
pub mod cli_fmt;
pub mod config;
pub mod conn_cap;
pub mod convert;
pub mod default_ignore;
pub mod extract;
pub mod fs_detect;
pub mod gate;
pub mod glob;
pub mod graph;
pub mod grep;
pub mod http_metrics;
pub mod index;
pub mod json_util;
pub mod logreload;
pub mod mcp;
pub mod mcp_auth;
pub mod model;
pub mod ontology_templates;
pub mod prompts;
pub mod query;
pub mod read;
pub mod root;
pub mod sdnotify;
pub mod search;
pub mod serve_guard;
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

#[cfg(feature = "tls")]
pub mod tls;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Serializes tests that mutate process-global environment variables (`std::env::set_var`/
/// `remove_var`). Rust runs tests concurrently in one process, so without this they race.
/// `into_inner()` recovers the guard if a holder panicked (a failed test must not poison it).
/// Every env-mutating test must hold this for its whole body:
/// `let _env = crate::TEST_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());`
#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        assert!(!version().is_empty());
    }
}
