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
    /// Allowed Host header value(s); empty ⇒ no Host allowlist.
    #[arg(long = "allowed-host")]
    pub allowed_host: Vec<String>,
    /// NLI session-pool size (default 2).
    #[arg(long = "nli-workers", default_value_t = 2)]
    pub nli_workers: usize,
    /// Reranker session-pool size (default 2).
    #[arg(long = "rerank-workers", default_value_t = 2)]
    pub rerank_workers: usize,
    /// Shared execution-provider preference (comma-join or repeat), e.g. `--ep cuda,cpu`.
    #[arg(long = "ep", value_delimiter = ',')]
    pub ep: Vec<String>,
    /// NLI-only EP override (falls back to `--ep`).
    #[arg(long = "nli-ep", value_delimiter = ',')]
    pub nli_ep: Vec<String>,
    /// Reranker-only EP override (falls back to `--ep`).
    #[arg(long = "rerank-ep", value_delimiter = ',')]
    pub rerank_ep: Vec<String>,
    /// GPU device id the EP binds to.
    #[arg(long = "ep-device")]
    pub ep_device: Option<i32>,
    /// GPU arena memory cap (MB) for the EP.
    #[arg(long = "ep-mem-limit-mb")]
    pub ep_mem_limit_mb: Option<usize>,
    /// Bearer api-key required on every request (except /health, /ready).
    #[arg(long = "api-key", env = "GLOSSA_INFER_API_KEY")]
    pub api_key: Option<String>,
    /// File whose first non-comment line is the Bearer api-key.
    #[arg(long = "api-key-file")]
    pub api_key_file: Option<PathBuf>,
    /// Allow a non-loopback bind without an api-key (otherwise refused).
    #[arg(long, env = "GLOSSA_INFER_INSECURE")]
    pub insecure: bool,
    /// Hard cap on concurrent in-flight requests; over it ⇒ HTTP 429.
    #[arg(long = "max-concurrency")]
    pub max_concurrency: Option<usize>,
    /// Per-request read/write timeout (seconds).
    #[arg(long = "request-timeout-secs", default_value_t = 120)]
    pub request_timeout_secs: u64,
    /// Max request body size (bytes).
    #[arg(long = "max-body-bytes", default_value_t = 4 * 1024 * 1024)]
    pub max_body_bytes: usize,
    /// Enable `/metrics` (Prometheus).
    #[arg(long = "metrics")]
    pub metrics: bool,
    /// CORS allowed origin(s).
    #[arg(long = "cors-allow-origin")]
    pub cors_allow_origin: Vec<String>,
    /// Internal: launched under the Windows Service control manager.
    #[arg(long = "windows-service", hide = true)]
    pub windows_service: bool,
    /// Internal: the SCM service name.
    #[arg(long = "service-name", hide = true)]
    pub service_name: Option<String>,
}

impl ServeArgs {
    /// The Bearer key: `--api-key`, else the first non-blank, non-`#` line of `--api-key-file`.
    pub fn resolved_api_key(&self) -> anyhow::Result<Option<String>> {
        if let Some(k) = &self.api_key {
            return Ok(Some(k.clone()));
        }
        if let Some(f) = &self.api_key_file {
            let content = std::fs::read_to_string(f)?;
            return Ok(content
                .lines()
                .map(str::trim)
                .find(|l| !l.is_empty() && !l.starts_with('#'))
                .map(str::to_string));
        }
        Ok(None)
    }

    /// The download cache dir (explicit `--cache-dir`, else a temp-based default).
    pub fn cache_dir(&self) -> PathBuf {
        self.cache_dir
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("glossa-inference-cache"))
    }
}
