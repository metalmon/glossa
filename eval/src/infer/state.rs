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
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use anyhow::{bail, Result};
    use glossa_nli::{probe_gpu_ep, probe_rerank_ep, InProcessNli, InProcessReranker};

    use super::{parse_variant, resolve_model_dir};
    use crate::infer::cli::ServeArgs;
    use crate::infer::pool::Pool;

    /// Handler-facing state. Pools start empty and are filled by [`ServerState::warm`] AFTER the
    /// socket binds, so `/health` reports 503 (loading) until the models are up (k8s-style
    /// readiness). Each pool sits behind a `Mutex<Option<..>>` set once at warm; reads clone the
    /// `Arc` under a brief, uncontended lock.
    pub struct ServerState {
        pub nli: Mutex<Option<Arc<Pool<InProcessNli>>>>,
        pub rerank: Mutex<Option<Arc<Pool<InProcessReranker>>>>,
        pub nli_ep: Mutex<Option<String>>,
        pub rerank_ep: Mutex<Option<String>>,
        pub nli_variant: Option<String>,
        pub rerank_variant: Option<String>,
        pub ready: AtomicBool,
        pub api_key: Option<String>,
        pub max_concurrency: Option<usize>,
        pub in_flight: AtomicUsize,
        // Resolved dirs + load params, consumed by warm().
        nli_dir: Option<PathBuf>,
        rerank_dir: Option<PathBuf>,
        entail_index: usize,
        nli_workers: usize,
        rerank_workers: usize,
        providers: Vec<String>,
        gpu_id: Option<i32>,
        gpu_mem_mb: Option<usize>,
    }

    /// Resolve model dirs (downloading a variant if only a repo is given) and return a
    /// NOT-YET-READY state (empty pools, `ready = false`). The caller binds the socket, then calls
    /// [`ServerState::warm`].
    pub fn build_state(args: &ServeArgs) -> Result<ServerState> {
        let cache = args.cache_dir();
        let nli_dir = resolve_model_dir(
            args.nli_model_dir.as_deref(),
            args.nli_repo.as_deref(),
            parse_variant(&args.nli_variant)?,
            &cache,
            "nli",
        )?;
        let rerank_dir = resolve_model_dir(
            args.rerank_model_dir.as_deref(),
            args.rerank_repo.as_deref(),
            parse_variant(&args.rerank_variant)?,
            &cache,
            "rerank",
        )?;
        if nli_dir.is_none() && rerank_dir.is_none() {
            bail!("no model given: pass --nli-model-dir/--nli-repo and/or --rerank-model-dir/--rerank-repo");
        }
        Ok(ServerState {
            nli: Mutex::new(None),
            rerank: Mutex::new(None),
            nli_ep: Mutex::new(None),
            rerank_ep: Mutex::new(None),
            nli_variant: nli_dir.as_ref().map(|_| args.nli_variant.clone()),
            rerank_variant: rerank_dir.as_ref().map(|_| args.rerank_variant.clone()),
            ready: AtomicBool::new(false),
            api_key: args.effective_auth()?,
            max_concurrency: args.max_concurrency,
            in_flight: AtomicUsize::new(0),
            nli_dir,
            rerank_dir,
            entail_index: args.entail_index,
            nli_workers: args.nli_workers.max(1),
            rerank_workers: args.rerank_workers.max(1),
            providers: glossa::config_util::expand_device(args.device.as_deref()),
            gpu_id: args.gpu_id,
            gpu_mem_mb: args.gpu_mem_mb,
        })
    }

    impl ServerState {
        /// Load the sessions (BLOCKING) and flip `ready`. Run on a blocking task AFTER the socket
        /// binds, so a probe during load sees `/health` 503. A load error propagates (the caller
        /// stops the server). NOTE (ruling T6c): `InProcessNli::load` caches by
        /// (dir,providers,dev,mem), so the N handles share ONE session — one VRAM copy, serialized
        /// until an uncached load lands (Phase 2).
        pub fn warm(&self) -> Result<()> {
            if let Some(d) = &self.nli_dir {
                let ep = probe_gpu_ep(
                    d,
                    self.entail_index,
                    &self.providers,
                    self.gpu_id,
                    self.gpu_mem_mb,
                )
                .ok()
                .flatten();
                let mut sessions = Vec::new();
                for _ in 0..self.nli_workers {
                    sessions.push(InProcessNli::load(
                        d,
                        self.entail_index,
                        &self.providers,
                        self.gpu_id,
                        self.gpu_mem_mb,
                    )?);
                }
                *self.nli.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some(Arc::new(Pool::new(sessions)));
                *self.nli_ep.lock().unwrap_or_else(|e| e.into_inner()) = ep;
            }
            if let Some(d) = &self.rerank_dir {
                let ep = probe_rerank_ep(d, &self.providers, self.gpu_id, self.gpu_mem_mb)
                    .ok()
                    .flatten();
                let mut sessions = Vec::new();
                for _ in 0..self.rerank_workers {
                    sessions.push(InProcessReranker::load(
                        d,
                        &self.providers,
                        self.gpu_id,
                        self.gpu_mem_mb,
                    )?);
                }
                *self.rerank.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some(Arc::new(Pool::new(sessions)));
                *self.rerank_ep.lock().unwrap_or_else(|e| e.into_inner()) = ep;
            }
            self.ready.store(true, Ordering::SeqCst);
            Ok(())
        }
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
