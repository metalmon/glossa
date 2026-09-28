//! `kbx rerank check`: a readiness/sanity diagnostic for the in-process cross-encoder reranker,
//! mirroring `kbx nli check`'s ([`crate::nli_check`]) shape: [`rerank_check_ok`] is a PURE
//! function over two scores (unit-tested, no IO); [`rerank_check`] is the thin wrapper that probes
//! the execution provider, loads the model, and scores one fixed sanity pair, then prints the
//! report.
//!
//! Unlike `nli check`, this diagnostic does not resolve a corpus's `[verify.nli]`/`[rerank]`
//! config — the reranker has no corpus-side "is it wired for THIS corpus" question the way the
//! NLI grounding verifier does, so `--model-dir`/`--ep`/`--ep-device`/`--ep-mem-limit-mb` are
//! taken directly as flags (mirroring `kbx nli set`'s flag names/shapes).

use std::path::PathBuf;

use anyhow::Result;

/// Verdict for the reranker's inversion guard: does the `relevant` passage outscore the
/// `irrelevant` one? Pure — no IO, safe to call with any pair of scores.
pub fn rerank_check_ok(relevant: f32, irrelevant: f32) -> bool {
    relevant > irrelevant
}

/// `kbx rerank check --model-dir <dir> [--ep ...] [--ep-device N] [--ep-mem-limit-mb N]`: probe
/// the configured execution provider the same STRICT way `kbx nli check` does
/// (`glossa_nli::probe_rerank_ep`, which builds a session with `.error_on_failure()` so a GPU EP
/// that can't register is reported rather than silently falling back to CPU), then load
/// `InProcessReranker` and score one FIXED, language-neutral, generic (query, [relevant,
/// irrelevant]) pair — deliberately NOT drawn from any corpus/dataset/gold — asserting relevant >
/// irrelevant via [`rerank_check_ok`] (the inversion guard).
///
/// Only reachable when the real ORT reranker path is compiled — i.e. the features that pull in
/// the direct `glossa-nli` dep with its `nli-ort` engine (`nli-directml`/`nli-coreml`/`nli-cuda`/
/// `nli-rocm`; there is no burn/wgpu reranker backend, unlike the NLI verifier). Printing the
/// diagnosis IS the deliverable — a NOT READY verdict is not a process error, so this always
/// returns `Ok(())` (a load/inference error while gathering facts still propagates as `Err`,
/// matching `nli_check`'s IO-vs-diagnosis split).
pub fn rerank_check(
    model_dir: PathBuf,
    ep: Vec<String>,
    ep_device: Option<i32>,
    ep_mem_limit_mb: Option<usize>,
) -> Result<()> {
    println!("engine         = ort");
    println!("model_dir      = {}", model_dir.display());

    #[cfg(any(
        feature = "nli-directml",
        feature = "nli-coreml",
        feature = "nli-cuda",
        feature = "nli-rocm",
    ))]
    {
        // GPU execution-provider probe: `InProcessReranker::load` is fail-open (no
        // `.error_on_failure()`), so a GPU build that can't load its runtime silently runs on CPU
        // with no signal. This STRICT probe reports whether the configured GPU EP ACTUALLY
        // initialized — mirrors `nli_check`'s `probe_gpu_ep` block exactly.
        match glossa_nli::probe_rerank_ep(&model_dir, &ep, ep_device, ep_mem_limit_mb) {
            Ok(Some(name)) => println!("ep_active      = {name} (initialized)"),
            Ok(None) => println!("ep_active      = cpu (no GPU EP configured)"),
            Err(e) => {
                let requested = ep
                    .iter()
                    .find(|p| p.as_str() != "cpu")
                    .map(String::as_str)
                    .unwrap_or("gpu");
                println!(
                    "ep_active      = {requested} REQUESTED but FAILED to init -> running on \
                     CPU: {e}"
                );
            }
        }

        // Fixed, language-neutral, generic sanity pair — no corpus/gold values (see
        // [[no-corpus-values-in-sop]]): a capital-city fact (relevant) vs. an unrelated fruit fact
        // (irrelevant).
        let query = "What is the capital of France?";
        let relevant = "Paris is the capital and most populous city of France.";
        let irrelevant = "Bananas are a good source of potassium.";

        let reranker =
            glossa_nli::InProcessReranker::load(&model_dir, &ep, ep_device, ep_mem_limit_mb)?;
        let scores = reranker.rerank(query, &[relevant, irrelevant])?;
        match scores.as_slice() {
            [rel_score, irr_score, ..] => {
                println!("relevant score   = {rel_score}");
                println!("irrelevant score = {irr_score}");
                let ok = rerank_check_ok(*rel_score, *irr_score);
                println!(
                    "=> {}",
                    if ok {
                        "READY (relevant > irrelevant)"
                    } else {
                        "NOT READY: irrelevant scored >= relevant (inversion)"
                    }
                );
            }
            _ => println!(
                "=> not ready: rerank() returned {} score(s), expected 2",
                scores.len()
            ),
        }
    }

    #[cfg(not(any(
        feature = "nli-directml",
        feature = "nli-coreml",
        feature = "nli-cuda",
        feature = "nli-rocm",
    )))]
    {
        let _ = (ep, ep_device, ep_mem_limit_mb);
        println!(
            "=> not ready: reranker not available in this kbx build (the reranker itself is \
             CPU-capable in glossa, but this binary wasn't built with a feature that pulls it \
             in) - rebuild kbx with --features nli-directml (or nli-cuda/nli-coreml/nli-rocm) \
             to enable it"
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rerank_check_verdict_ok_when_relevant_wins() {
        assert!(rerank_check_ok(2.5, -1.0));
    }

    #[test]
    fn rerank_check_verdict_bad_when_inverted() {
        assert!(!rerank_check_ok(0.1, 0.9));
    }
}
