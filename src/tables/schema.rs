//! Reader for `schema.toml` — the engine's marking-order manifest, unified
//! across the reference and the agent. The `[order]` section carries the
//! marking sequence expressed by field names (`params`) and/or `.csp` file
//! names (`files`), positionally aligned; either or both may be present and a
//! consumer takes whichever it needs. Unknown sections are ignored so the
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
    #[serde(default)]
    files: Vec<String>,
}

/// The `[order]` of a `schema.toml`: the marking sequence by field name
/// (`params`) and/or by `.csp` file name (`files`), positionally aligned.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct SchemaOrder {
    pub params: Vec<String>,
    pub files: Vec<String>,
}

/// Read the `[order]` section. Returns an empty `SchemaOrder` on a
/// missing/unreadable file, invalid TOML, or a missing `[order]` section.
pub fn read_schema_order(path: &Path) -> SchemaOrder {
    let Ok(text) = std::fs::read_to_string(path) else {
        return SchemaOrder::default();
    };
    match toml::from_str::<SchemaFile>(&text) {
        Ok(s) => {
            let o = s.order.unwrap_or_default();
            SchemaOrder { params: o.params, files: o.files }
        }
        Err(_) => SchemaOrder::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_params_and_files_ignoring_unknown_sections() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("schema.toml");
        std::fs::write(
            &p,
            "[order]\nparams = [\"a\", \"b\"]\nfiles = [\"a.csp\", \"b.csp\"]\n[bindings]\nx = 1\n",
        )
        .unwrap();
        let o = read_schema_order(&p);
        assert_eq!(o.params, vec!["a".to_string(), "b".into()]);
        assert_eq!(o.files, vec!["a.csp".to_string(), "b.csp".into()]);
    }

    #[test]
    fn params_only_leaves_files_empty() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("s.toml");
        std::fs::write(&p, "[order]\nparams = [\"x\"]\n").unwrap();
        let o = read_schema_order(&p);
        assert_eq!(o.params, vec!["x".to_string()]);
        assert!(o.files.is_empty());
    }

    #[test]
    fn missing_file_or_section_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_schema_order(&dir.path().join("nope.toml")), SchemaOrder::default());
        let p = dir.path().join("s.toml");
        std::fs::write(&p, "doc = \"x\"\n").unwrap();
        assert_eq!(read_schema_order(&p), SchemaOrder::default());
    }
}
