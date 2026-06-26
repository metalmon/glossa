pub mod glob;
pub mod grep;
pub mod mcp;
pub mod tools;
pub mod model;
pub mod root;
pub mod extract;
pub mod graph;
pub mod index;
pub mod query;
pub mod search;
pub mod walk;
pub mod read;
pub mod trace;
pub mod cli_fmt;
pub mod tz_export;

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
