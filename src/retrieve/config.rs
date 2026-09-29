use std::path::{Path, PathBuf};

use crate::config_util::{env_bool, env_i32, env_string, env_usize};
use crate::graph::ontology::Ontology;

// Reranking only reorders the fetched pool, it never fetches deeper on its own. For rerank to
// pool deeper than the caller's search limit, operators must set `[rerank].pool_size` GREATER
// than that limit. E.g. the MCP `search` tool's default limit is 50, same as this default, so
// out of the box rerank reorders the same 50 hits rather than pooling deeper.
pub const DEFAULT_POOL_SIZE: usize = 50;

pub struct RerankConfig {
    pub enabled: bool,
    pub scorer: Option<String>,
    pub model_dir: Option<PathBuf>,
    pub pool_size: usize,
    pub execution_providers: Vec<String>,
    pub ep_device: Option<i32>,
    pub ep_mem_limit_mb: Option<usize>,
    /// Remote reranker (`scorer = "http"`) endpoint base URL; `None` ⇒ `resolve_reranker` cannot
    /// build an `HttpReranker`. Env `GLOSSA_RERANK_HTTP_ENDPOINT`.
    pub endpoint: Option<String>,
    /// Remote reranker per-call timeout in ms (default 5000). Env `GLOSSA_RERANK_HTTP_TIMEOUT_MS`.
    pub timeout_ms: u64,
    /// Optional Bearer api-key for the remote reranker. Env `GLOSSA_RERANK_HTTP_API_KEY`.
    pub api_key: Option<String>,
}

impl RerankConfig {
    /// `glossa_dir` is the corpus `.glossa` dir; ontology loads from its parent (mirrors
    /// `VerifyConfig::resolve`).
    pub fn resolve(glossa_dir: &Path) -> RerankConfig {
        let ont = glossa_dir.parent().map(Ontology::load_or_default);
        RerankConfig {
            enabled: env_bool("GLOSSA_RERANK_ENABLED")
                .or_else(|| ont.as_ref().and_then(|o| o.rerank_enabled()))
                .unwrap_or(false),
            scorer: env_string("GLOSSA_RERANK_SCORER").or_else(|| {
                ont.as_ref()
                    .and_then(|o| o.rerank_scorer())
                    .map(str::to_string)
            }),
            model_dir: env_string("GLOSSA_RERANK_MODEL_DIR")
                .or_else(|| {
                    ont.as_ref()
                        .and_then(|o| o.rerank_model_dir())
                        .map(str::to_string)
                })
                .map(PathBuf::from),
            pool_size: env_usize("GLOSSA_RERANK_POOL_SIZE")
                .or_else(|| ont.as_ref().and_then(|o| o.rerank_pool_size()))
                .unwrap_or(DEFAULT_POOL_SIZE),
            execution_providers: crate::config_util::expand_device(
                env_string("GLOSSA_RERANK_DEVICE")
                    .or_else(|| {
                        ont.as_ref()
                            .and_then(|o| o.rerank_device())
                            .map(str::to_string)
                    })
                    .as_deref(),
            ),
            ep_device: env_i32("GLOSSA_RERANK_GPU_ID")
                .or_else(|| ont.as_ref().and_then(|o| o.rerank_gpu_id())),
            ep_mem_limit_mb: env_usize("GLOSSA_RERANK_GPU_MEM_MB")
                .or_else(|| ont.as_ref().and_then(|o| o.rerank_gpu_mem_mb())),
            endpoint: env_string("GLOSSA_RERANK_HTTP_ENDPOINT").or_else(|| {
                ont.as_ref()
                    .and_then(|o| o.rerank_endpoint())
                    .map(str::to_string)
            }),
            timeout_ms: env_usize("GLOSSA_RERANK_HTTP_TIMEOUT_MS")
                .or_else(|| ont.as_ref().and_then(|o| o.rerank_timeout_ms()))
                .unwrap_or(5000) as u64,
            api_key: env_string("GLOSSA_RERANK_HTTP_API_KEY").or_else(|| {
                ont.as_ref()
                    .and_then(|o| o.rerank_api_key())
                    .map(str::to_string)
            }),
        }
    }

    /// Reranking runs only when explicitly enabled, an in-process scorer is selected, and a model
    /// dir is set. Anything else => today's plain-BM25 path.
    pub fn is_active(&self) -> bool {
        self.enabled && self.scorer.as_deref() == Some("in_process") && self.model_dir.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_off_when_no_ontology() {
        let _lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_RERANK_POOL_SIZE");
        let dir = tempfile::tempdir().unwrap();
        let c = RerankConfig::resolve(dir.path());
        assert!(!c.enabled);
        assert_eq!(c.scorer, None);
        assert_eq!(c.pool_size, DEFAULT_POOL_SIZE);
        assert!(!c.is_active());
    }

    #[test]
    fn active_requires_enabled_scorer_and_model_dir() {
        let _lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_RERANK_POOL_SIZE");
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "[rerank]\nenabled=true\nscorer=\"in_process\"\nmodel_dir=\"/m\"\npool_size=40\n\
             device=\"cuda\"\n",
        )
        .unwrap();
        let c = RerankConfig::resolve(&g);
        assert!(c.is_active());
        assert_eq!(c.pool_size, 40);
        assert_eq!(
            c.execution_providers,
            vec!["cuda".to_string(), "cpu".to_string()]
        );
    }

    #[test]
    fn rerank_http_config_round_trips() {
        let _lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_RERANK_HTTP_ENDPOINT");
        std::env::remove_var("GLOSSA_RERANK_HTTP_TIMEOUT_MS");
        std::env::remove_var("GLOSSA_RERANK_HTTP_API_KEY");
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "[rerank]\nenabled=true\nscorer=\"http\"\nendpoint=\"http://gpu:8080\"\n",
        )
        .unwrap();
        let c = RerankConfig::resolve(&g);
        assert_eq!(c.scorer.as_deref(), Some("http"));
        assert_eq!(c.endpoint.as_deref(), Some("http://gpu:8080"));
        assert_eq!(c.timeout_ms, 5000);
    }

    #[test]
    fn not_active_when_enabled_but_no_model_dir() {
        let _lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_RERANK_POOL_SIZE");
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "[rerank]\nenabled=true\nscorer=\"in_process\"\n",
        )
        .unwrap();
        assert!(!RerankConfig::resolve(&g).is_active());
    }

    #[test]
    fn env_overrides_pool_size() {
        let _lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::set_var("GLOSSA_RERANK_POOL_SIZE", "12");
        let dir = tempfile::tempdir().unwrap();
        let c = RerankConfig::resolve(dir.path());
        std::env::remove_var("GLOSSA_RERANK_POOL_SIZE");
        assert_eq!(c.pool_size, 12);
    }
}
