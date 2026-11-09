use std::path::{Path, PathBuf};

use crate::config_util::{env_bool, env_i32, env_string, env_usize};
use crate::graph::ontology::Ontology;

// Reranking only reorders the fetched pool, it never fetches deeper on its own. For rerank to
// pool deeper than the caller's search limit, operators must set `[rerank].pool_size` GREATER
// than that limit. E.g. the MCP `search` tool's default limit is 50, same as this default, so
// out of the box rerank reorders the same 50 hits rather than pooling deeper.
pub const DEFAULT_POOL_SIZE: usize = 50;

/// How many hits a `search` returns when the call names no `limit` and the corpus configures none
/// — the depth a reader actually sees in production. Named rather than repeated so a harness
/// measuring "what can production surface" reads that depth instead of picking its own. `kb search`
/// uses this same default: its own larger one widened the rerank pool via `pool_size.max(limit)`
/// and made the CLI and the agent tool rank different candidate sets.
pub const DEFAULT_SEARCH_LIMIT: usize = 50;

/// Per-corpus `[retrieval].search_limit` (env `GLOSSA_SEARCH_LIMIT`), or `None` when the corpus
/// sets none. Returned as an `Option` on purpose: the entry point supplies the fallback, so adding
/// this knob changes nothing for a corpus that does not set it. Both the agent tool and
/// `kb search` fall back to [`DEFAULT_SEARCH_LIMIT`] — they must agree, or rerank pools differ.
pub fn resolve_search_limit(glossa_dir: &Path) -> Option<usize> {
    env_usize("GLOSSA_SEARCH_LIMIT")
        .or_else(|| {
            glossa_dir
                .parent()
                .map(Ontology::load_or_default_shared)
                .and_then(|o| o.search_limit())
        })
        .filter(|n| *n > 0)
}

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
    /// Remote reranker backend (`scorer = "http"`): `tei` | `vllm` | `llamacpp` | `kbi` | `jina` |
    /// `cohere`. Names the operator's server; selects the wire shape (TEI vs Jina) so they never
    /// have to know which server speaks which protocol. Default `kbi` (our own inference server).
    /// Env `GLOSSA_RERANK_HTTP_BACKEND`.
    pub backend: String,
    /// Optional served-model name sent to Jina-family backends only (vLLM requires it; llama.cpp
    /// and kbi ignore it). Env `GLOSSA_RERANK_HTTP_MODEL`.
    pub model: Option<String>,
    /// Per-batch token budget for the in-process cross-encoder. `None` ⇒ the engine default.
    /// Env `GLOSSA_RERANK_BATCH_TOKENS`, then `[rerank].batch_tokens`.
    ///
    /// `in_process` only — batching belongs to the server for every `scorer = "http"` backend.
    /// And it is half of a pair: an ORT arena never returns a batch's peak, so this engine's
    /// budget and the gate's ADD UP on a shared device, and stay added up for the process's life.
    pub batch_tokens: Option<usize>,
    /// ONNX Runtime intra-op threads for the in-process session. `None` ⇒ the engine default.
    /// Env `GLOSSA_RERANK_INTRA_THREADS`, then `[rerank].intra_threads`. `in_process` only, and
    /// likewise half of a pair: two engines' thread counts oversubscribe the same cores additively.
    pub intra_threads: Option<usize>,
}

/// Whether a client-side compute knob is configured on a path where it cannot take effect.
///
/// Six remote reranker backends exist, and for every one of them batching and threading belong to
/// the server. A budget set under `scorer = "http"` does nothing; the caller uses this to SAY so
/// rather than drop it in silence, which is how the device keys on that path became a trap.
pub fn compute_knobs_inert(
    scorer: Option<&str>,
    batch_tokens: Option<usize>,
    intra_threads: Option<usize>,
) -> bool {
    scorer == Some("http") && any_compute_knob_set(batch_tokens, intra_threads)
}

/// Whether the operator configured either compute knob at all.
///
/// The remote path is not the only one that cannot honour them: the burn engines read neither yet
/// (they take no such parameter), so an `in_process` burn build must say so too. Silence there
/// would be the same trap in a second place.
pub fn any_compute_knob_set(batch_tokens: Option<usize>, intra_threads: Option<usize>) -> bool {
    batch_tokens.is_some() || intra_threads.is_some()
}

impl RerankConfig {
    /// `glossa_dir` is the corpus `.glossa` dir; ontology loads from its parent (mirrors
    /// `VerifyConfig::resolve`).
    pub fn resolve(glossa_dir: &Path) -> RerankConfig {
        let ont = glossa_dir.parent().map(Ontology::load_or_default_shared);
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
            backend: env_string("GLOSSA_RERANK_HTTP_BACKEND")
                .or_else(|| {
                    ont.as_ref()
                        .and_then(|o| o.rerank_backend())
                        .map(str::to_string)
                })
                .unwrap_or_else(|| "kbi".to_string()),
            model: env_string("GLOSSA_RERANK_HTTP_MODEL").or_else(|| {
                ont.as_ref()
                    .and_then(|o| o.rerank_model())
                    .map(str::to_string)
            }),
            // Per-engine env beats the ontology. `GLOSSA_NLI_BATCH_TOKENS` still exists and still
            // overrides BOTH engines at once, deeper in the harness — it is the blunt fallback,
            // this is the sharp one.
            batch_tokens: env_usize("GLOSSA_RERANK_BATCH_TOKENS")
                .or_else(|| ont.as_ref().and_then(|o| o.rerank_batch_tokens()))
                .filter(|n| *n > 0),
            intra_threads: env_usize("GLOSSA_RERANK_INTRA_THREADS")
                .or_else(|| ont.as_ref().and_then(|o| o.rerank_intra_threads()))
                .filter(|n| *n > 0),
        }
    }

    /// Reranking runs only when explicitly enabled, an in-process scorer is selected, and a model
    /// dir is set. Anything else => today's plain-BM25 path.
    pub fn is_active(&self) -> bool {
        self.enabled && self.scorer.as_deref() == Some("in_process") && self.model_dir.is_some()
    }
}

/// The search depth an entry point uses when its caller named no `limit`: the corpus's
/// `[retrieval].search_limit`, else [`DEFAULT_SEARCH_LIMIT`].
///
/// Every entry point MUST go through this rather than keeping a fallback of its own. The depth is
/// not merely how many rows come back: `retrieve_with` sizes the rerank pool as
/// `pool_size.max(limit)`, so an entry point with a larger fallback silently reranks a larger
/// candidate set and returns a different TOP ordering for the same query and index. That is exactly
/// what `kb search` (fallback 100) did against the agent's `search` tool (fallback 50) until both
/// were routed here.
pub fn search_limit_or_default(glossa_dir: &Path) -> usize {
    resolve_search_limit(glossa_dir).unwrap_or(DEFAULT_SEARCH_LIMIT)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The CLI and the agent tool must resolve the same depth from the same corpus, or rerank
    /// reorders different candidate pools and their top hits disagree.
    #[test]
    fn search_depth_falls_back_to_the_one_shared_default() {
        let _lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_SEARCH_LIMIT");
        let dir = tempfile::tempdir().unwrap();
        let glossa_dir = dir.path().join(".glossa");
        std::fs::create_dir_all(&glossa_dir).unwrap();
        assert_eq!(resolve_search_limit(&glossa_dir), None);
        assert_eq!(
            search_limit_or_default(&glossa_dir),
            DEFAULT_SEARCH_LIMIT,
            "an unconfigured corpus must give every entry point the same depth"
        );
    }

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
        std::env::remove_var("GLOSSA_RERANK_HTTP_BACKEND");
        std::env::remove_var("GLOSSA_RERANK_HTTP_MODEL");
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
        // Backend defaults to our own kbi; model unset.
        assert_eq!(c.backend, "kbi");
        assert_eq!(c.model, None);
    }

    #[test]
    fn rerank_backend_and_model_round_trip_from_ontology() {
        let _lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_RERANK_HTTP_BACKEND");
        std::env::remove_var("GLOSSA_RERANK_HTTP_MODEL");
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "[rerank]\nenabled=true\nscorer=\"http\"\nendpoint=\"http://vllm:8000\"\n\
             backend=\"vllm\"\nmodel=\"BAAI/bge-reranker-v2-m3\"\n",
        )
        .unwrap();
        let c = RerankConfig::resolve(&g);
        assert_eq!(c.backend, "vllm");
        assert_eq!(c.model.as_deref(), Some("BAAI/bge-reranker-v2-m3"));
    }

    /// Precedence, and the zero rule at the resolution layer — which is where a zero can actually
    /// arrive from, either file or environment.
    #[test]
    fn compute_knobs_resolve_env_over_ontology_and_treat_zero_as_unset() {
        let _lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GLOSSA_RERANK_BATCH_TOKENS");
        std::env::remove_var("GLOSSA_RERANK_INTRA_THREADS");
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "[rerank]\nbatch_tokens=1024\nintra_threads=2\n",
        )
        .unwrap();

        let c = RerankConfig::resolve(&g);
        assert_eq!(c.batch_tokens, Some(1024), "from the ontology");
        assert_eq!(c.intra_threads, Some(2));

        std::env::set_var("GLOSSA_RERANK_BATCH_TOKENS", "8192");
        assert_eq!(
            RerankConfig::resolve(&g).batch_tokens,
            Some(8192),
            "env wins"
        );

        std::env::set_var("GLOSSA_RERANK_BATCH_TOKENS", "0");
        assert_eq!(
            RerankConfig::resolve(&g).batch_tokens,
            None,
            "a 0 is 'I did not choose', not 'never batch'"
        );
        std::env::remove_var("GLOSSA_RERANK_BATCH_TOKENS");
    }

    /// Setting an in-process knob on a remote scorer is a configuration mistake the operator
    /// should hear about; these are the cases the caller warns on.
    #[test]
    fn compute_knobs_are_inert_only_on_a_remote_scorer_with_something_set() {
        assert!(compute_knobs_inert(Some("http"), Some(4096), None));
        assert!(compute_knobs_inert(Some("http"), None, Some(4)));
        assert!(
            !compute_knobs_inert(Some("http"), None, None),
            "nothing configured, nothing to warn about"
        );
        assert!(
            !compute_knobs_inert(Some("in_process"), Some(4096), None),
            "in_process is exactly where these work"
        );
        assert!(!compute_knobs_inert(None, Some(4096), None));
    }

    /// The burn arms need the engine-agnostic question — "did the operator set either of these" —
    /// because burn ignores them on the `in_process` path too, where `compute_knobs_inert` is
    /// deliberately false.
    #[test]
    fn any_compute_knob_set_is_independent_of_the_scorer() {
        assert!(any_compute_knob_set(Some(4096), None));
        assert!(any_compute_knob_set(None, Some(4)));
        assert!(any_compute_knob_set(Some(4096), Some(4)));
        assert!(!any_compute_knob_set(None, None));
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
