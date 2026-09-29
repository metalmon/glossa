//! Server state + startup: resolve each model dir (auto-download the variant when only a repo is
//! given), load N sessions per model into a pool. The pure `resolve_model_dir` is always compiled
//! and unit-tested; `ServerState`/`build_state` need an ORT engine and are feature-gated.
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

use crate::download::{download_variant, Variant};

/// Map a `--*-variant` flag to a [`Variant`].
pub fn parse_variant(s: &str) -> Result<Variant> {
    Ok(match s {
        "fp32" => Variant::Fp32,
        "fp16" => Variant::Fp16,
        "int8" => Variant::Int8,
        other => bail!("unknown variant {other:?} (expected fp32|fp16|int8)"),
    })
}

/// Decide which dir to load: an explicit `--*-model-dir` wins; else, given a `--*-repo`, fetch the
/// variant into `cache_dir/subdir` (skipping the download when `model.onnx` is already there); else
/// `None` (that model is not served).
pub fn resolve_model_dir(
    dir: Option<&Path>,
    repo: Option<&str>,
    variant: Variant,
    cache_dir: &Path,
    subdir: &str,
) -> Result<Option<PathBuf>> {
    if let Some(d) = dir {
        return Ok(Some(d.to_path_buf()));
    }
    let Some(repo) = repo else {
        return Ok(None);
    };
    let to = cache_dir.join(subdir);
    if !to.join("model.onnx").is_file() {
        download_variant(repo, "main", &to, variant)?;
    }
    Ok(Some(to))
}

// The running server holds loaded sessions, so it needs an ORT engine. NOTE: `InProcessNli::load`
// caches by (dir, providers, device, mem), so N loads share ONE session (one `Mutex<Session>`) —
// the pool is correct and keeps ONE VRAM copy, but `--*-workers N` is serialized until an uncached
// load lands (Phase 2). See the plan's Task 6 ruling.
#[cfg(any(
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
))]
pub use engine::{build_state, ServerState};

#[cfg(any(
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
))]
mod engine {
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::Arc;

    use anyhow::{bail, Result};
    use glossa_nli::{probe_gpu_ep, probe_rerank_ep, InProcessNli, InProcessReranker};

    use super::{parse_variant, resolve_model_dir};
    use crate::infer::cli::ServeArgs;
    use crate::infer::pool::Pool;

    /// Everything a handler needs: the per-model session pools, the probed EP names (for `/info`),
    /// the readiness flag, auth key, and the concurrency guard state.
    pub struct ServerState {
        pub nli: Option<Arc<Pool<InProcessNli>>>,
        pub rerank: Option<Arc<Pool<InProcessReranker>>>,
        pub nli_ep: Option<String>,
        pub rerank_ep: Option<String>,
        pub nli_variant: Option<String>,
        pub rerank_variant: Option<String>,
        pub ready: Arc<AtomicBool>,
        pub api_key: Option<String>,
        pub max_concurrency: Option<usize>,
        pub in_flight: Arc<AtomicUsize>,
    }

    pub fn build_state(args: &ServeArgs) -> Result<ServerState> {
        let cache = args.cache_dir();
        let nli_variant = parse_variant(&args.nli_variant)?;
        let rerank_variant = parse_variant(&args.rerank_variant)?;
        let nli_dir = resolve_model_dir(
            args.nli_model_dir.as_deref(),
            args.nli_repo.as_deref(),
            nli_variant,
            &cache,
            "nli",
        )?;
        let rerank_dir = resolve_model_dir(
            args.rerank_model_dir.as_deref(),
            args.rerank_repo.as_deref(),
            rerank_variant,
            &cache,
            "rerank",
        )?;
        if nli_dir.is_none() && rerank_dir.is_none() {
            bail!("no model given: pass --nli-model-dir/--nli-repo and/or --rerank-model-dir/--rerank-repo");
        }
        let nli_eps = if args.nli_ep.is_empty() {
            &args.ep
        } else {
            &args.nli_ep
        };
        let rerank_eps = if args.rerank_ep.is_empty() {
            &args.ep
        } else {
            &args.rerank_ep
        };

        let (nli, nli_ep) = match &nli_dir {
            Some(d) => {
                let ep = probe_gpu_ep(
                    d,
                    args.entail_index,
                    nli_eps,
                    args.ep_device,
                    args.ep_mem_limit_mb,
                )
                .ok()
                .flatten();
                let mut sessions = Vec::new();
                for _ in 0..args.nli_workers.max(1) {
                    sessions.push(InProcessNli::load(
                        d,
                        args.entail_index,
                        nli_eps,
                        args.ep_device,
                        args.ep_mem_limit_mb,
                    )?);
                }
                (Some(Arc::new(Pool::new(sessions))), ep)
            }
            None => (None, None),
        };
        let (rerank, rerank_ep) = match &rerank_dir {
            Some(d) => {
                let ep = probe_rerank_ep(d, rerank_eps, args.ep_device, args.ep_mem_limit_mb)
                    .ok()
                    .flatten();
                let mut sessions = Vec::new();
                for _ in 0..args.rerank_workers.max(1) {
                    sessions.push(InProcessReranker::load(
                        d,
                        rerank_eps,
                        args.ep_device,
                        args.ep_mem_limit_mb,
                    )?);
                }
                (Some(Arc::new(Pool::new(sessions))), ep)
            }
            None => (None, None),
        };
        Ok(ServerState {
            nli,
            rerank,
            nli_ep,
            rerank_ep,
            nli_variant: nli_dir.as_ref().map(|_| args.nli_variant.clone()),
            rerank_variant: rerank_dir.as_ref().map(|_| args.rerank_variant.clone()),
            ready: Arc::new(AtomicBool::new(true)),
            api_key: args.effective_auth()?,
            max_concurrency: args.max_concurrency,
            in_flight: Arc::new(AtomicUsize::new(0)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_model_dir_prefers_explicit_dir() {
        let d = tempfile::tempdir().unwrap();
        let got = resolve_model_dir(
            Some(d.path()),
            Some("some/repo"),
            crate::download::Variant::Fp16,
            d.path(),
            "nli",
        )
        .unwrap();
        assert_eq!(got.as_deref(), Some(d.path())); // explicit dir wins, no download
    }

    #[test]
    fn resolve_model_dir_none_when_neither_given() {
        let d = tempfile::tempdir().unwrap();
        let got =
            resolve_model_dir(None, None, crate::download::Variant::Fp16, d.path(), "nli").unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn parse_variant_maps_names() {
        assert_eq!(
            parse_variant("fp16").unwrap(),
            crate::download::Variant::Fp16
        );
        assert_eq!(
            parse_variant("int8").unwrap(),
            crate::download::Variant::Int8
        );
        assert_eq!(
            parse_variant("fp32").unwrap(),
            crate::download::Variant::Fp32
        );
        assert!(parse_variant("bogus").is_err());
    }
}
