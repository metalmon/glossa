//! `lab.toml` — the file-first `kbx` eval toolkit's endpoint config (model/judge/reflect/distil
//! endpoints, default filenames, timeout). The corpus is NOT configured here: it comes from
//! kb-style PATH resolution (`glossa::root::resolve_root` via `kb_eval::workspace::resolve`),
//! exactly like `kb` itself.

use std::path::{Path, PathBuf};

fn d120() -> u64 {
    120
}

fn default_prompt() -> String {
    "answer.md".to_string()
}

fn default_judge_prompt() -> String {
    "judge.md".to_string()
}

fn default_dataset() -> String {
    "dataset.toml".to_string()
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
    /// Strong-model endpoint for `kbx distil` (grounded synthesis, a later plan).
    #[serde(default)]
    pub distil: Option<Endpoint>,
    #[serde(default)]
    pub defaults: Defaults,
}

#[derive(serde::Deserialize, Clone)]
pub struct Defaults {
    #[serde(default = "default_prompt")]
    pub prompt: String,
    #[serde(default = "default_judge_prompt")]
    pub judge_prompt: String,
    #[serde(default = "default_dataset")]
    pub dataset: String,
}

impl Default for Defaults {
    fn default() -> Self {
        Defaults {
            prompt: default_prompt(),
            judge_prompt: default_judge_prompt(),
            dataset: default_dataset(),
        }
    }
}

/// Workspace-relative paths from `LabConfig`, resolved against the workspace dir.
/// Absolute paths in `lab.toml` are left as-is.
pub struct ResolvedPaths {
    pub prompt: PathBuf,
    pub judge_prompt: PathBuf,
    pub dataset: PathBuf,
}

fn resolve_one(workspace: &Path, value: &str) -> PathBuf {
    let p = Path::new(value);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        workspace.join(p)
    }
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

    /// Resolve the workspace-relative prompt/judge_prompt/dataset fields into
    /// absolute-or-workspace-joined paths.
    pub fn resolve(&self, workspace: &Path) -> ResolvedPaths {
        ResolvedPaths {
            prompt: resolve_one(workspace, &self.defaults.prompt),
            judge_prompt: resolve_one(workspace, &self.defaults.judge_prompt),
            dataset: resolve_one(workspace, &self.defaults.dataset),
        }
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
        assert_eq!(c.defaults.dataset, "dataset.toml"); // default
        assert!(c.judge.is_none());
        assert!(c.distil.is_none());
        assert_eq!(c.model.timeout_secs, 120);
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
