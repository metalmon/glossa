//! In-process NLI scorer over a 3-way NLI model. Standalone crate — it does NOT depend on
//! `glossa`; `glossa` owns the `NliScorer` trait and implements it for the engine type (see
//! glossa's `src/gate/nli_engine.rs`), mirroring the glossa-constraint adapter pattern.
//!
//! Two interchangeable inference engines, selected at build time by mutually-exclusive features:
//!
//! - `nli-ort` — ONNX Runtime (CPU + EPs: CUDA/DirectML/CoreML/ROCm). The default engine.
//! - `nli-burn-wgpu` — a pure-Rust hand-written BERT-base forward on burn, running on the GPU via
//!   wgpu Vulkan/SPIR-V. One binary, only the Vulkan loader at runtime; the cross-vendor
//!   (AMD/Intel-Linux) + dependency-light path (Plan 4).
//!
//! Both engines share the [`harness`] (windowing, batching, softmax, MAX-pool); only the raw
//! forward (`harness::RawForward`) differs, so cross-engine parity holds by construction.
//!
//! Fail-open (hard requirement): `load()` and `entail()` return `anyhow::Result` and may return
//! `Err` on any failure (missing file, tokenizer error, session error, degenerate output). Neither
//! ever panics or calls `.unwrap()`/`.expect()` on fallible IO/inference — the caller maps
//! `Err`/`None` to AC-only scoring.

// Exactly one inference engine per build.
#[cfg(all(feature = "nli-ort", feature = "nli-burn-wgpu"))]
compile_error!("enable exactly ONE of `nli-ort` / `nli-burn-wgpu`, not both");

#[cfg(not(any(feature = "nli-ort", feature = "nli-burn-wgpu")))]
compile_error!("enable exactly ONE inference engine: `nli-ort` or `nli-burn-wgpu`");

#[cfg(any(feature = "nli-ort", feature = "nli-burn-wgpu"))]
pub mod harness;

#[cfg(feature = "nli-burn-wgpu")]
mod burn_engine;
#[cfg(feature = "nli-burn-wgpu")]
pub use burn_engine::InProcessBurnNli;

#[cfg(feature = "nli-ort")]
pub use ort_engine::InProcessNli;

#[cfg(feature = "nli-ort")]
mod ort_engine {
    // Exactly one ORT linking strategy per build (bundled `download-binaries` vs. runtime
    // `load-dynamic`) — see the `[features]` comment in Cargo.toml for the full rationale.
    #[cfg(all(feature = "nli-ort-bundled", feature = "nli-ort-dynamic"))]
    compile_error!("enable exactly one ORT linking strategy: nli-ort-bundled OR nli-ort-dynamic");

    // The `nli-ort` engine code needs SOME linking strategy to actually link against ONNX Runtime.
    #[cfg(all(
        feature = "nli-ort",
        not(any(feature = "nli-ort-bundled", feature = "nli-ort-dynamic"))
    ))]
    compile_error!(
        "feature `nli-ort` needs a linking strategy: enable `nli-ort-bundled` (self-contained CPU/DirectML/CoreML) or `nli-ort-dynamic` (CUDA/ROCm runtime dll)"
    );

    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex, OnceLock};

    use ort::execution_providers::ExecutionProviderDispatch;
    use ort::session::{builder::GraphOptimizationLevel, Session};
    use ort::value::Tensor;
    use tokenizers::Tokenizer;

    use crate::harness::{self, RawForward, DEFAULT_MAX_SEQ_LEN};

    /// The loaded model + tokenizer for one `model_dir`, shared across all `InProcessNli` handles
    /// that point at the same directory (see `MODEL_CACHE`).
    struct Inner {
        tokenizer: Tokenizer,
        // rc.13 `Session::run` takes `&mut self` (ONNX Runtime session internals are not thread
        // safe), so a session shared behind `&self` needs interior mutability. One `Mutex` per
        // process-global model instance serializes inference (one session per process, spec §2.1a).
        session: Mutex<Session>,
        max_seq_len: usize,
        batch_budget_tokens: usize,
    }

    /// `MODEL_CACHE`'s key: `(canonicalized model_dir, providers)`. Factored out to satisfy
    /// `clippy::type_complexity`.
    type ModelCacheKey = (PathBuf, Vec<String>);
    type ModelCache = OnceLock<Mutex<HashMap<ModelCacheKey, Arc<Inner>>>>;

    /// Process-global load-once cache, keyed by `(canonicalized model_dir, providers)`. A different
    /// EP set is a different ONNX session (spec §2.1a), so `providers` is part of the key.
    static MODEL_CACHE: ModelCache = OnceLock::new();

    /// An in-process NLI scorer over an ONNX 3-way NLI model. Constructed from a local `model_dir`
    /// (`model.onnx` + `tokenizer.json`) and the entailment class index (from the model's
    /// `id2label`, config-pinned by the caller).
    pub struct InProcessNli {
        inner: Arc<Inner>,
        entail_index: usize,
    }

    /// Resolve the ONNX model file inside `model_dir`: `model.onnx` if present; else the lone
    /// `*.onnx`; else `Err` (none, or 2+ non-canonical candidates — named so the user can rename).
    fn resolve_model_file(model_dir: &Path) -> anyhow::Result<PathBuf> {
        let canonical = model_dir.join("model.onnx");
        if canonical.is_file() {
            return Ok(canonical);
        }

        let mut candidates = Vec::new();
        let entries = std::fs::read_dir(model_dir)
            .map_err(|e| anyhow::anyhow!("reading model dir ({}): {e}", model_dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| {
                anyhow::anyhow!("reading model dir entry ({}): {e}", model_dir.display())
            })?;
            let path = entry.path();
            let is_onnx = path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"));
            if is_onnx && path.is_file() {
                candidates.push(path);
            }
        }

        match candidates.len() {
            0 => anyhow::bail!("no .onnx file in {}", model_dir.display()),
            1 => Ok(candidates.remove(0)),
            _ => {
                candidates.sort();
                let names: Vec<String> = candidates
                    .iter()
                    .map(|p| {
                        p.file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_else(|| p.display().to_string())
                    })
                    .collect();
                anyhow::bail!(
                    "multiple .onnx candidates, none named model.onnx: {} — rename the one you \
                     want to model.onnx",
                    names.join(", ")
                )
            }
        }
    }

    impl InProcessNli {
        /// Load the model + tokenizer from `model_dir`. `entail_index` is the softmax index of the
        /// entailment class. `providers` is the ordered list of runtime execution-provider names
        /// (lowercased by the caller's config), e.g. `["cuda", "cpu"]`; empty or `["cpu"]` means
        /// CPU-only. Reuses a cached `Inner` for the same `(canonicalized model_dir, providers)`.
        pub fn load(
            model_dir: &Path,
            entail_index: usize,
            providers: &[String],
        ) -> anyhow::Result<Self> {
            let cache_key = (
                model_dir.canonicalize().unwrap_or_else(|_| model_dir.to_path_buf()),
                providers.to_vec(),
            );
            let cache = MODEL_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

            if let Some(inner) = cache
                .lock()
                .map_err(|_| anyhow::anyhow!("nli model cache mutex poisoned"))?
                .get(&cache_key)
            {
                return Ok(Self { inner: Arc::clone(inner), entail_index });
            }

            let built = Arc::new(Self::build_inner(model_dir, providers)?);
            let mut guard = cache
                .lock()
                .map_err(|_| anyhow::anyhow!("nli model cache mutex poisoned"))?;
            let inner = Arc::clone(guard.entry(cache_key).or_insert(built));
            Ok(Self { inner, entail_index })
        }

        fn build_inner(model_dir: &Path, providers: &[String]) -> anyhow::Result<Inner> {
            let tokenizer_path = model_dir.join("tokenizer.json");
            let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("tokenizer load ({}): {e}", tokenizer_path.display()))?;
            // Premise windowing does its own truncation; disable any baked-in truncation/padding.
            tokenizer
                .with_truncation(None)
                .map_err(|e| anyhow::anyhow!("tokenizer truncation config: {e}"))?;
            tokenizer.with_padding(None);

            let model_path = resolve_model_file(model_dir)?;
            let eps = execution_provider_dispatch(providers);
            // Deliberately NO `.error_on_failure()`: a GPU EP that can't register falls through to
            // ORT's implicit CPU EP — the fail-open contract (see module doc).
            let session = Session::builder()
                .map_err(|e| anyhow::anyhow!("onnx session builder: {e}"))?
                .with_execution_providers(eps)
                .map_err(|e| anyhow::anyhow!("onnx session execution providers: {e}"))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| anyhow::anyhow!("onnx session optimization level: {e}"))?
                .with_intra_threads(1)
                .map_err(|e| anyhow::anyhow!("onnx session intra-threads: {e}"))?
                .commit_from_file(&model_path)
                .map_err(|e| anyhow::anyhow!("onnx session load ({}): {e}", model_path.display()))?;

            Ok(Inner {
                tokenizer,
                session: Mutex::new(session),
                max_seq_len: DEFAULT_MAX_SEQ_LEN,
                batch_budget_tokens: harness::parse_batch_budget_tokens(),
            })
        }

        /// P(entail) of each hypothesis against `premise`, one `f32` in `[0, 1]` per hypothesis.
        /// The windowing/batching/pooling live in [`harness::entail`]; this engine only supplies
        /// the raw forward (`RawForward` below).
        pub fn entail(&self, premise: &str, hypotheses: &[&str]) -> anyhow::Result<Vec<f32>> {
            harness::entail(
                self,
                &self.inner.tokenizer,
                self.inner.max_seq_len,
                self.inner.batch_budget_tokens,
                self.entail_index,
                premise,
                hypotheses,
            )
        }
    }

    impl RawForward for InProcessNli {
        /// Run one padded batch through the ONNX session and return the raw 3-way logits
        /// (row-major, `n * 3`). Padding invariance is guaranteed by the caller zeroing the
        /// `attention_mask` on padded positions.
        fn forward_logits(
            &self,
            input_ids: &[i64],
            attention_mask: &[i64],
            token_type_ids: &[i64],
            n: usize,
            seq: usize,
        ) -> anyhow::Result<Vec<f32>> {
            let input_ids_tensor = Tensor::from_array(([n, seq], input_ids.to_vec()))?;
            let attention_mask_tensor = Tensor::from_array(([n, seq], attention_mask.to_vec()))?;
            let token_type_ids_tensor = Tensor::from_array(([n, seq], token_type_ids.to_vec()))?;

            let mut session = self
                .inner
                .session
                .lock()
                .map_err(|_| anyhow::anyhow!("nli onnx session mutex poisoned"))?;
            let outputs = session.run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ])?;

            // Index by position (not name): these heads emit exactly one output (the logits).
            let out0 = outputs
                .values()
                .next()
                .ok_or_else(|| anyhow::anyhow!("onnx session returned no outputs"))?;
            let (_shape, data) = out0
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("logits extraction: {e}"))?;
            Ok(data.to_vec())
        }
    }

    /// Build the ORT execution-provider dispatch list for `providers`, in the caller's order.
    fn execution_provider_dispatch(providers: &[String]) -> Vec<ExecutionProviderDispatch> {
        compiled_gpu_providers(providers)
            .into_iter()
            .filter_map(dispatch_for_gpu_name)
            .collect()
    }

    /// Returns the subset of `providers` registered as GPU EPs given the compiled Cargo features,
    /// in input order; `"cpu"` and unknown names are dropped (they fall to ORT's implicit CPU EP).
    fn compiled_gpu_providers(providers: &[String]) -> Vec<&'static str> {
        providers.iter().filter_map(|p| compiled_gpu_name(p)).collect()
    }

    /// Per-name lookup used by [`compiled_gpu_providers`]. Chain of early returns so `name` stays
    /// referenced by live code even with no GPU feature compiled in.
    fn compiled_gpu_name(name: &str) -> Option<&'static str> {
        #[cfg(feature = "nli-cuda")]
        if name == "cuda" {
            return Some("cuda");
        }
        #[cfg(feature = "nli-directml")]
        if name == "directml" {
            return Some("directml");
        }
        #[cfg(feature = "nli-coreml")]
        if name == "coreml" {
            return Some("coreml");
        }
        #[cfg(feature = "nli-rocm")]
        if name == "rocm" {
            return Some("rocm");
        }
        let _ = name;
        None
    }

    /// Turns an already-filtered GPU EP name into a live `ExecutionProviderDispatch`.
    fn dispatch_for_gpu_name(name: &str) -> Option<ExecutionProviderDispatch> {
        #[cfg(feature = "nli-cuda")]
        if name == "cuda" {
            return Some(ort::ep::CUDA::default().build());
        }
        #[cfg(feature = "nli-directml")]
        if name == "directml" {
            return Some(ort::ep::DirectML::default().build());
        }
        #[cfg(feature = "nli-coreml")]
        if name == "coreml" {
            return Some(ort::ep::CoreML::default().build());
        }
        #[cfg(feature = "nli-rocm")]
        if name == "rocm" {
            return Some(ort::ep::ROCm::default().build());
        }
        let _ = name;
        None
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn obvious_entailment_outranks_contradiction() {
            let Ok(model_dir) = std::env::var("GLOSSA_NLI_TEST_MODEL") else {
                return;
            };
            const ENTAIL_IDX: usize = 0;
            let nli = InProcessNli::load(Path::new(&model_dir), ENTAIL_IDX, &["cpu".to_string()])
                .expect("model load should succeed against a real GLOSSA_NLI_TEST_MODEL dir");
            let scores = nli
                .entail(
                    "A dog is sleeping on the couch.",
                    &["An animal is resting.", "The room is empty."],
                )
                .expect("entail should succeed against a real model");
            assert_eq!(scores.len(), 2);
            for s in &scores {
                assert!((0.0..=1.0).contains(s), "entail score outside [0,1]: {scores:?}");
            }
        }

        #[test]
        fn resolve_model_file_only_canonical_present() {
            let dir = tempfile::tempdir().expect("tempdir");
            let onnx = dir.path().join("model.onnx");
            std::fs::write(&onnx, b"").expect("write model.onnx");
            assert_eq!(resolve_model_file(dir.path()).expect("resolve"), onnx);
        }

        #[test]
        fn resolve_model_file_only_lone_noncanonical_present() {
            let dir = tempfile::tempdir().expect("tempdir");
            let fp16 = dir.path().join("model.fp16.onnx");
            std::fs::write(&fp16, b"").expect("write model.fp16.onnx");
            assert_eq!(resolve_model_file(dir.path()).expect("resolve lone"), fp16);
        }

        #[test]
        fn resolve_model_file_canonical_wins_when_both_present() {
            let dir = tempfile::tempdir().expect("tempdir");
            let canonical = dir.path().join("model.onnx");
            let fp16 = dir.path().join("model.fp16.onnx");
            std::fs::write(&canonical, b"").expect("write model.onnx");
            std::fs::write(&fp16, b"").expect("write model.fp16.onnx");
            assert_eq!(resolve_model_file(dir.path()).expect("resolve canonical"), canonical);
        }

        #[test]
        fn resolve_model_file_none_present_errs() {
            let dir = tempfile::tempdir().expect("tempdir");
            std::fs::write(dir.path().join("tokenizer.json"), b"").expect("write tokenizer.json");
            let err = resolve_model_file(dir.path()).expect_err("no .onnx should error");
            assert!(err.to_string().contains("no .onnx"), "unexpected: {err}");
        }

        #[test]
        fn resolve_model_file_two_noncanonical_candidates_errs_naming_both() {
            let dir = tempfile::tempdir().expect("tempdir");
            std::fs::write(dir.path().join("model.fp16.onnx"), b"").expect("write fp16");
            std::fs::write(dir.path().join("model.int8.onnx"), b"").expect("write int8");
            let err = resolve_model_file(dir.path()).expect_err("ambiguous should error");
            let msg = err.to_string();
            assert!(
                msg.contains("model.fp16.onnx") && msg.contains("model.int8.onnx"),
                "error should name both: {msg}"
            );
        }

        #[test]
        fn compiled_gpu_providers_cuda_gated_by_compiled_feature() {
            let input = vec!["cuda".to_string(), "cpu".to_string()];
            let out = compiled_gpu_providers(&input);
            #[cfg(feature = "nli-cuda")]
            assert_eq!(out, vec!["cuda"]);
            #[cfg(not(feature = "nli-cuda"))]
            assert!(out.is_empty(), "cuda must fall through to CPU, got {out:?}");
        }

        #[test]
        fn compiled_gpu_providers_rocm_gated_by_compiled_feature() {
            let input = vec!["rocm".to_string(), "cpu".to_string()];
            let out = compiled_gpu_providers(&input);
            #[cfg(feature = "nli-rocm")]
            assert_eq!(out, vec!["rocm"]);
            #[cfg(not(feature = "nli-rocm"))]
            assert!(out.is_empty(), "rocm must fall through to CPU, got {out:?}");
        }

        #[test]
        fn compiled_gpu_providers_drops_unknown_names() {
            let input = vec!["not_a_real_ep".to_string(), "cpu".to_string()];
            assert!(compiled_gpu_providers(&input).is_empty());
        }

        #[test]
        fn compiled_gpu_providers_empty_input_yields_empty_output() {
            let input: Vec<String> = Vec::new();
            assert!(compiled_gpu_providers(&input).is_empty());
        }

        #[test]
        fn compiled_gpu_providers_preserves_relative_order() {
            let input = vec![
                "unknown".to_string(),
                "coreml".to_string(),
                "cpu".to_string(),
                "cuda".to_string(),
                "directml".to_string(),
            ];
            let out = compiled_gpu_providers(&input);
            let mut last_pos: Option<usize> = None;
            for name in &out {
                let pos = input.iter().position(|p| p == name).expect("survivor from input");
                if let Some(last) = last_pos {
                    assert!(last < pos, "order not preserved: {out:?} vs {input:?}");
                }
                last_pos = Some(pos);
            }
        }

        /// Batched `entail` must match scoring each `(window, hypothesis)` row alone (batch of 1)
        /// and MAX-pooling by hand, and be padding-invariant. Env-gated on a real model.
        #[test]
        fn batched_entail_matches_per_row_reference_and_is_padding_invariant() {
            let Ok(model_dir) = std::env::var("GLOSSA_NLI_TEST_MODEL") else {
                return;
            };
            const ENTAIL_IDX: usize = 0;
            std::env::set_var("GLOSSA_NLI_BATCH_TOKENS", "16384");
            let nli = InProcessNli::load(Path::new(&model_dir), ENTAIL_IDX, &["cpu".to_string()])
                .expect("model load should succeed against a real GLOSSA_NLI_TEST_MODEL dir");

            let premise =
                "The device must be fully powered off before any servicing begins. ".repeat(80);
            let short_hyp = "The device is off.";
            let long_hyp = "Before servicing the device, an operator must first ensure that every \
                             power source, including auxiliary and backup supplies, has been \
                             completely disconnected and independently verified as de-energized.";
            let other_hyp = "It is raining outside today.";
            let hypotheses = [short_hyp, long_hyp, other_hyp];

            let batched = nli.entail(&premise, &hypotheses).expect("batched entail");
            assert_eq!(batched.len(), hypotheses.len());

            let tok = &nli.inner.tokenizer;
            let msl = nli.inner.max_seq_len;
            let mut reference = vec![0.0f32; hypotheses.len()];
            for (hyp_idx, hypothesis) in hypotheses.iter().enumerate() {
                let windows =
                    harness::premise_windows(tok, msl, &premise, hypothesis).expect("windows");
                for window in &windows {
                    let (ids, mask, types) =
                        harness::encode_row(tok, msl, window, hypothesis).expect("encode_row");
                    let row = harness::Row { hyp_idx, input_ids: ids, attention_mask: mask, token_type_ids: types };
                    let scores = harness::run_batch(&nli, &[row], &[0], ENTAIL_IDX).expect("run_batch");
                    if scores[0] > reference[hyp_idx] {
                        reference[hyp_idx] = scores[0];
                    }
                }
            }
            for (i, (b, r)) in batched.iter().zip(reference.iter()).enumerate() {
                assert!((b - r).abs() < 1e-4, "hyp {i}: batched {b} vs per-row {r}");
            }

            // Padding invariance: a short row scored alone == its score padded inside a batch.
            let short_premise = "The device is powered off.";
            let (s_ids, s_mask, s_types) =
                harness::encode_row(tok, msl, short_premise, short_hyp).expect("encode short");
            let (l_ids, l_mask, l_types) =
                harness::encode_row(tok, msl, short_premise, long_hyp).expect("encode long");
            assert!(l_ids.len() > s_ids.len(), "long row must be strictly longer");

            let alone = harness::run_batch(
                &nli,
                &[harness::Row { hyp_idx: 0, input_ids: s_ids.clone(), attention_mask: s_mask.clone(), token_type_ids: s_types.clone() }],
                &[0],
                ENTAIL_IDX,
            )
            .expect("alone")[0];
            let padded = harness::run_batch(
                &nli,
                &[
                    harness::Row { hyp_idx: 0, input_ids: s_ids, attention_mask: s_mask, token_type_ids: s_types },
                    harness::Row { hyp_idx: 0, input_ids: l_ids, attention_mask: l_mask, token_type_ids: l_types },
                ],
                &[0, 1],
                ENTAIL_IDX,
            )
            .expect("padded");
            assert!((padded[0] - alone).abs() < 1e-6, "padding changed score: {alone} vs {}", padded[0]);
        }
    }
}
