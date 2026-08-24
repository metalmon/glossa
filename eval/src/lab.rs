//! `lab.toml` — the file-first `kbx` eval toolkit's endpoint config (model/judge/reflect/distil
//! endpoints, default filenames, timeout). The corpus is NOT configured here: it comes from
//! kb-style PATH resolution (`glossa::root::resolve_root` via `kb_eval::workspace::resolve`),
//! exactly like `kb` itself.

use std::path::Path;

fn d120() -> u64 {
    120
}

#[derive(serde::Deserialize, Clone)]
pub struct Endpoint {
    pub endpoint: String,
    pub model: String,
    /// Inline literal API key. Prefer `api_key_env` to keep secrets out of the file.
    #[serde(default)]
    pub api_key: String,
    /// Name of an env var to read the API key from (optional).
    #[serde(default)]
    pub api_key_env: String,
    /// Per-endpoint request timeout, in seconds.
    #[serde(default = "d120")]
    pub timeout_secs: u64,
}

impl Endpoint {
    /// Resolve the API key for this endpoint: an inline `api_key` wins; otherwise
    /// fall back to reading `api_key_env` from the environment; otherwise `None`.
    pub fn resolve_key(&self) -> Option<String> {
        if !self.api_key.is_empty() {
            return Some(self.api_key.clone());
        }
        if !self.api_key_env.is_empty() {
            if let Ok(v) = std::env::var(&self.api_key_env) {
                return Some(v);
            }
        }
        None
    }
}

#[derive(serde::Deserialize, Clone)]
pub struct LabConfig {
    pub model: Endpoint,
    #[serde(default)]
    pub judge: Option<Endpoint>,
    /// Endpoint used to reflect on/rewrite prompts (`kbx train`, a later plan).
    #[serde(default)]
    pub reflect: Option<Endpoint>,
    /// Strong-model endpoint shared by `kbx reason` (grounded backward-chaining) and `kbx distil`
    /// (grounded synthetic-gold generation).
    #[serde(default)]
    pub distil: Option<Endpoint>,
}

impl LabConfig {
    /// Read and parse `<workspace>/lab.toml`.
    pub fn load(workspace: &Path) -> anyhow::Result<Self> {
        Self::load_at(&workspace.join("lab.toml"))
    }

    /// Read and parse an explicit `lab.toml` path.
    pub fn load_at(lab_path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(lab_path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", lab_path.display()))?;
        let config: LabConfig = toml::from_str(&text)
            .map_err(|e| anyhow::anyhow!("parsing {}: {e}", lab_path.display()))?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lab_config_parses_and_defaults() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("lab.toml"),
            "[model]\nendpoint=\"http://x\"\nmodel=\"m\"\n",
        )
        .unwrap();
        let c = LabConfig::load(dir.path()).unwrap();
        assert!(c.judge.is_none());
        assert!(c.reflect.is_none());
        assert!(c.distil.is_none());
        assert_eq!(c.model.timeout_secs, 120); // d120 default
    }

    #[test]
    fn lab_loads_without_corpus_and_reads_distil() {
        let toml = r#"
            [model]
            endpoint = "http://x"
            model = "m"
            [distil]
            endpoint = "http://y"
            model = "big"
        "#;
        let lab: LabConfig = toml::from_str(toml).unwrap();
        assert!(lab.distil.is_some());
        assert_eq!(lab.distil.as_ref().unwrap().model, "big");
    }

    #[test]
    fn load_at_reads_an_explicit_path() {
        let dir = tempfile::tempdir().unwrap();
        let lab_path = dir.path().join("custom-lab.toml");
        std::fs::write(&lab_path, "[model]\nendpoint=\"http://x\"\nmodel=\"m\"\n").unwrap();
        let c = LabConfig::load_at(&lab_path).unwrap();
        assert_eq!(c.model.model, "m");
    }
}
