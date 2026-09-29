//! `inference-server serve` arguments. Pure clap struct (no ORT), so it compiles in every build.
use std::path::PathBuf;

use clap::Args;

/// Flags for `inference-server serve` (and, via the service runner, the installed service).
#[derive(Args, Debug, Clone)]
pub struct ServeArgs {
    /// Local NLI model dir (skips download). If absent and `--nli-repo` is set, the variant is
    /// fetched into the cache dir.
    #[arg(long = "nli-model-dir")]
    pub nli_model_dir: Option<PathBuf>,
    /// HF repo to auto-download the NLI model from when `--nli-model-dir` is absent.
    #[arg(long = "nli-repo")]
    pub nli_repo: Option<String>,
    /// NLI precision variant to fetch: fp32|fp16|int8 (default fp16, prod).
    #[arg(long = "nli-variant", default_value = "fp16")]
    pub nli_variant: String,
    /// Local reranker model dir (skips download).
    #[arg(long = "rerank-model-dir")]
    pub rerank_model_dir: Option<PathBuf>,
    /// HF repo to auto-download the reranker from when `--rerank-model-dir` is absent.
    #[arg(long = "rerank-repo")]
    pub rerank_repo: Option<String>,
    /// Reranker precision variant: fp32|fp16|int8 (default fp16).
    #[arg(long = "rerank-variant", default_value = "fp16")]
    pub rerank_variant: String,
    /// Cache dir for auto-downloaded models (default: <temp>/glossa-inference-cache).
    #[arg(long = "cache-dir")]
    pub cache_dir: Option<PathBuf>,
    /// Entailment softmax class index for the NLI model (from its id2label).
    #[arg(long = "nli-entail-index", default_value_t = 0)]
    pub entail_index: usize,
    /// Bind address. Default 127.0.0.1:8071 — NOT 8080 (that is the Glossa MCP's own default).
    #[arg(long, env = "GLOSSA_INFER_BIND", default_value = "127.0.0.1:8071")]
    pub bind: String,
    /// Device: `cpu` | `cuda` | `directml` | `rocm` (default: a GPU provider if compiled in, else
    /// `cpu`). CPU fallback is automatic.
    #[arg(long = "device")]
    pub device: Option<String>,
    /// GPU device id the provider binds to.
    #[arg(long = "gpu-id")]
    pub gpu_id: Option<i32>,
    /// GPU arena memory cap (MB) for the provider.
    #[arg(long = "gpu-mem-mb")]
    pub gpu_mem_mb: Option<usize>,
    /// Bearer api-key required on every request (except /health, /ready).
    #[arg(long = "api-key", env = "GLOSSA_INFER_API_KEY")]
    pub api_key: Option<String>,
    /// File whose first non-comment line is the Bearer api-key.
    #[arg(long = "api-key-file")]
    pub api_key_file: Option<PathBuf>,
    /// Allow a non-loopback bind without an api-key (otherwise refused).
    #[arg(long, env = "GLOSSA_INFER_INSECURE")]
    pub insecure: bool,
    /// Hard cap on concurrent in-flight requests; over it ⇒ HTTP 429. Unset ⇒ no cap (requests
    /// queue on the pool). (`--allowed-host`, CORS, and a request timeout are Phase-2, with TLS.)
    #[arg(long = "max-concurrency")]
    pub max_concurrency: Option<usize>,
    /// Max request body size (bytes).
    #[arg(long = "max-body-bytes", default_value_t = 4 * 1024 * 1024)]
    pub max_body_bytes: usize,
    /// Internal: launched under the Windows Service control manager.
    #[arg(long = "windows-service", hide = true)]
    pub windows_service: bool,
    /// Internal: the SCM service name.
    #[arg(long = "service-name", hide = true)]
    pub service_name: Option<String>,
}

impl ServeArgs {
    /// The download cache dir (explicit `--cache-dir`, else a temp-based default).
    pub fn cache_dir(&self) -> PathBuf {
        self.cache_dir
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("glossa-inference-cache"))
    }

    /// The RESOLVED Bearer key (see [`resolve_key`]). Use its `.is_some()` for the interlock's
    /// `has_auth` — a named-but-empty `--api-key-file` must NOT count as auth.
    pub fn effective_auth(&self) -> anyhow::Result<Option<String>> {
        resolve_key(self.api_key.as_deref(), self.api_key_file.as_deref())
    }
}

/// Resolve the Bearer key: inline `--api-key` wins; else the first non-blank, non-`#` line of
/// `--api-key-file`. A key-file that is named but resolves empty is a config ERROR (`Err`) — not a
/// silent "no auth", which would let the non-loopback interlock pass on a public bind.
pub fn resolve_key(
    api_key: Option<&str>,
    api_key_file: Option<&std::path::Path>,
) -> anyhow::Result<Option<String>> {
    if let Some(k) = api_key {
        return Ok(Some(k.to_string()));
    }
    let Some(f) = api_key_file else {
        return Ok(None);
    };
    let content = std::fs::read_to_string(f)?;
    match content
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
    {
        Some(k) => Ok(Some(k.to_string())),
        None => anyhow::bail!(
            "--api-key-file {} has no key (every line blank or a # comment)",
            f.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_key_bails_on_empty_or_comment_only_file() {
        let d = tempfile::tempdir().unwrap();
        let f = d.path().join("k.txt");
        std::fs::write(&f, "# just a comment\n\n   \n").unwrap();
        assert!(resolve_key(None, Some(&f)).is_err());
    }

    #[test]
    fn resolve_key_reads_first_real_line() {
        let d = tempfile::tempdir().unwrap();
        let f = d.path().join("k.txt");
        std::fs::write(&f, "# c\nsekret\nignored\n").unwrap();
        assert_eq!(
            resolve_key(None, Some(&f)).unwrap(),
            Some("sekret".to_string())
        );
    }

    #[test]
    fn resolve_key_inline_wins_and_none_when_absent() {
        assert_eq!(resolve_key(Some("k"), None).unwrap(), Some("k".to_string()));
        assert_eq!(resolve_key(None, None).unwrap(), None);
    }
}
