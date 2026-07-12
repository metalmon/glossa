//! Reader for `schema.toml` — the engine's parameter-order manifest. Thin form:
//! only the `[order]` section is consumed; unknown sections are ignored so the
//! format can grow (e.g. a future `[bindings]`) without breaking older readers.

use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize, Default)]
struct SchemaFile {
    #[serde(default)]
    order: Option<OrderSection>,
}

#[derive(Deserialize, Default)]
struct OrderSection {
    #[serde(default)]
    params: Vec<String>,
}

/// Ordered parameter names from a `schema.toml` `[order].params`. Empty on a
/// missing/unreadable file, invalid TOML, or a missing `[order]` section.
pub fn read_schema_order(path: &Path) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    match toml::from_str::<SchemaFile>(&text) {
        Ok(s) => s.order.unwrap_or_default().params,
        Err(_) => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_order_params_ignoring_unknown_sections() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("schema.toml");
        std::fs::write(
            &p,
            "[order]\nparams = [\"a\", \"b\", \"c\"]\n[bindings]\nx = 1\n",
        )
        .unwrap();
        assert_eq!(read_schema_order(&p), vec!["a".to_string(), "b".into(), "c".into()]);
    }

    #[test]
    fn missing_file_or_section_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_schema_order(&dir.path().join("nope.toml")).is_empty());
        let p = dir.path().join("s.toml");
        std::fs::write(&p, "doc = \"x\"\n").unwrap();
        assert!(read_schema_order(&p).is_empty());
    }
}
