pub mod model;
pub mod extract;
pub mod graph;
pub mod index;
pub mod query;
pub mod search;
pub mod walk;
pub mod read;

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
