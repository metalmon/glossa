//! Server state + startup: resolve each model dir (auto-download the variant when only a repo is
//! given), load ONE session per model. The pure `resolve_model_dir` is always compiled and
//! unit-tested; `ServerState`/`build_state` need an ORT engine and are feature-gated.
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

// The running server holds one loaded session per model, so it needs an ORT engine.
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

    /// Handler-facing state. Each model holds exactly ONE session (its internal `Mutex<Session>`
    /// serializes inference — on a single GPU, request parallelism comes from batching, not from N
    /// sessions, which would only cost N× VRAM; the multi-GPU answer is multiple instances behind a
    /// load balancer). The session slots start empty and are filled by [`ServerState::warm`] AFTER
    /// the socket binds, so `/health` reports 503 (loading) until the models are up (k8s-style
    /// readiness). Each slot sits behind a `Mutex<Option<..>>` set once at warm; reads clone the
    /// `Arc` under a brief, uncontended lock.
    pub struct ServerState {
        pub nli: Mutex<Option<Arc<InProcessNli>>>,
        pub rerank: Mutex<Option<Arc<InProcessReranker>>>,
        pub nli_ep: Mutex<Option<String>>,
        pub rerank_ep: Mutex<Option<String>>,
        pub nli_variant: Option<String>,
        pub rerank_variant: Option<String>,
        pub ready: AtomicBool,
        pub api_key: Option<String>,
        pub allowed_host: Vec<String>,
        pub max_concurrency: Option<usize>,
        pub in_flight: AtomicUsize,
        /// The per-batch token budget a `--fit` run chose for each engine, set once at warm. `None`
        /// — the normal case — means requests run at the session's own configured budget, so the
        /// unfitted path is untouched.
        pub nli_fitted: Mutex<Option<usize>>,
        pub rerank_fitted: Mutex<Option<usize>>,
        // Resolved dirs + load params, consumed by warm().
        nli_dir: Option<PathBuf>,
        rerank_dir: Option<PathBuf>,
        entail_index: usize,
        providers: Vec<String>,
        gpu_id: Option<i32>,
        gpu_mem_mb: Option<usize>,
        nli_batch_tokens: Option<usize>,
        rerank_batch_tokens: Option<usize>,
        fit: bool,
        fit_max_rows: usize,
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
            allowed_host: args.allowed_host.clone(),
            max_concurrency: args.max_concurrency,
            in_flight: AtomicUsize::new(0),
            nli_fitted: Mutex::new(None),
            rerank_fitted: Mutex::new(None),
            nli_dir,
            rerank_dir,
            entail_index: args.entail_index,
            providers: glossa::config_util::expand_device(args.device.as_deref()),
            gpu_id: args.gpu_id,
            gpu_mem_mb: args.gpu_mem_mb,
            nli_batch_tokens: args.nli_batch_tokens,
            rerank_batch_tokens: args.rerank_batch_tokens,
            fit: args.fit,
            fit_max_rows: args.fit_max_rows,
        })
    }

    impl ServerState {
        /// Load the one session per configured model (BLOCKING) and flip `ready`. Run on a blocking
        /// task AFTER the socket binds, so a probe during load sees `/health` 503. A load error
        /// propagates (the caller stops the server). One session per model: the session's internal
        /// `Mutex<Session>` serializes inference, request parallelism comes from batching, and a
        /// single GPU gains nothing from a second in-process session (see [`ServerState`]).
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
                let session = InProcessNli::load(
                    d,
                    self.entail_index,
                    &self.providers,
                    self.gpu_id,
                    self.gpu_mem_mb,
                    self.nli_batch_tokens,
                    None,
                )?;
                *self.nli.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(session));
                *self.nli_ep.lock().unwrap_or_else(|e| e.into_inner()) = ep;
            }
            if let Some(d) = &self.rerank_dir {
                let ep = probe_rerank_ep(d, &self.providers, self.gpu_id, self.gpu_mem_mb)
                    .ok()
                    .flatten();
                let session = InProcessReranker::load(
                    d,
                    &self.providers,
                    self.gpu_id,
                    self.gpu_mem_mb,
                    self.rerank_batch_tokens,
                    None,
                )?;
                *self.rerank.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(session));
                *self.rerank_ep.lock().unwrap_or_else(|e| e.into_inner()) = ep;
            }
            // Both models are loaded by now, which is the only state worth measuring in: residency
            // is the hazard, so a size that fails here fails for the reason production would see.
            if self.fit {
                self.fit_in_place(&crate::fit::SweepOpts {
                    max_rows: self
                        .fit_max_rows
                        .clamp(1, glossa_nli::harness::NLI_BATCH_MAX_ROWS),
                    seq: crate::fit::effective_seq(None, glossa_nli::harness::DEFAULT_MAX_SEQ_LEN),
                    repeats: 3,
                    tolerance: crate::fit::DEFAULT_TOLERANCE,
                });
            }
            self.ready.store(true, Ordering::SeqCst);
            Ok(())
        }

        /// Measure the batch budget on this device and keep the answer IN MEMORY for this process.
        /// Writes nothing — a server that fits itself must not mutate anything on disk.
        ///
        /// A fit that cannot measure is reported and does not stop the server: the models loaded,
        /// so the service can serve at its configured budget. Taking a deployment down because a
        /// diagnostic came back empty would be the worse failure.
        pub fn fit_in_place(&self, opts: &crate::fit::SweepOpts) {
            use crate::fit::{
                budget_tokens, fit_refusal, fit_report, select, sweep, NliTarget, RerankTarget,
            };

            let seq = opts.seq;
            let rerank = self
                .rerank
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let nli = self.nli.lock().unwrap_or_else(|e| e.into_inner()).clone();
            // The same refusal the `kbx` side applies, from the strict probes `warm` already ran: a
            // configured provider that did not register leaves `*_ep` empty, and loading is
            // fail-open, so sweeping anyway would drive 512-token cross-encoder pools through a CPU
            // session and print the result as a device fit. On CPU the answer is known and the pool
            // is a thermal hazard, so this refuses in both cases — one line, and no sweep.
            let rerank_bound = self
                .rerank_ep
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_some();
            let nli_bound = self
                .nli_ep
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_some();

            // Report, then keep the number only when batching actually pays: a few percent for a
            // batch peak that never leaves the arena is the trade `PAYS_THRESHOLD` exists to refuse,
            // and asking for a measurement is not asking to spend VRAM on noise. Either way the
            // report says which happened, because "what the number is" and "whether it was taken"
            // are different facts and the operator needs both.
            let settle = |label: &str,
                          flag: &str,
                          outcome: Option<crate::fit::FitOutcome>,
                          neighbour: Option<(&str, Option<usize>)>,
                          slot: &Mutex<Option<usize>>| {
                let applied = outcome.as_ref().is_some_and(|o| o.pays);
                let hint = outcome.as_ref().map(|o| {
                    format!(
                        "kbi {flag} {} — serves at this size without re-measuring",
                        budget_tokens(o.chosen, seq)
                    )
                });
                eprint!(
                    "{}",
                    fit_report(
                        label,
                        seq,
                        outcome.as_ref(),
                        neighbour,
                        hint.as_deref(),
                        applied
                    )
                );
                if let Some(o) = outcome.filter(|o| o.pays) {
                    let tokens = budget_tokens(o.chosen, seq);
                    // Part 1 added this announcement for exactly this case: a budget larger than the
                    // planner can spend is silently equivalent to the largest it can, and a number
                    // nobody can act on should not be printed as a recommendation.
                    if let Some(effective) =
                        glossa_nli::harness::budget_beyond_planner(Some(tokens))
                    {
                        eprintln!(
                            "  note         {tokens} exceeds what the planner can spend; it behaves as {effective}"
                        );
                    }
                    *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(tokens);
                }
            };

            // Read AFTER the reranker sweep so the NLI report names the budget this run just chose,
            // not the configured one: printing a stale half of a sum is the exact failure the
            // two-number rule exists to prevent.
            let configured_or_fitted =
                |fitted: &Mutex<Option<usize>>, configured: Option<usize>| {
                    (*fitted.lock().unwrap_or_else(|e| e.into_inner())).or(configured)
                };

            if let Some(engine) = rerank.as_ref() {
                if let Some(why) = fit_refusal(Some("in_process"), None, rerank_bound) {
                    eprintln!("reranker fit skipped: {why}");
                } else {
                    let target = RerankTarget {
                        engine: engine.as_ref(),
                        seq,
                    };
                    match sweep(&target, opts) {
                        Ok(samples) => settle(
                            "reranker",
                            "--rerank-batch-tokens",
                            select(&samples, opts.tolerance),
                            nli.as_ref().map(|_| {
                                (
                                    "nli gate",
                                    configured_or_fitted(&self.nli_fitted, self.nli_batch_tokens),
                                )
                            }),
                            &self.rerank_fitted,
                        ),
                        Err(e) => {
                            eprintln!("reranker fit failed, serving at the configured budget: {e}")
                        }
                    }
                }
            }
            if let Some(engine) = nli.as_ref() {
                if let Some(why) = fit_refusal(Some("in_process"), None, nli_bound) {
                    eprintln!("nli fit skipped: {why}");
                } else {
                    let target = NliTarget {
                        engine: engine.as_ref(),
                        seq,
                    };
                    match sweep(&target, opts) {
                        Ok(samples) => settle(
                            "nli gate",
                            "--nli-batch-tokens",
                            select(&samples, opts.tolerance),
                            rerank.as_ref().map(|_| {
                                (
                                    "reranker",
                                    configured_or_fitted(
                                        &self.rerank_fitted,
                                        self.rerank_batch_tokens,
                                    ),
                                )
                            }),
                            &self.nli_fitted,
                        ),
                        Err(e) => {
                            eprintln!("nli fit failed, serving at the configured budget: {e}")
                        }
                    }
                }
            }
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
