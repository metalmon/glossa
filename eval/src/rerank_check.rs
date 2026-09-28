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

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Verdict for the reranker's inversion guard: does the `relevant` passage outscore the
/// `irrelevant` one? Pure — no IO, safe to call with any pair of scores.
pub fn rerank_check_ok(relevant: f32, irrelevant: f32) -> bool {
    relevant > irrelevant
}

/// Human label for the compiled reranker engine, shown in the `check` report. Mirrors
/// `nli_check::engine_label`.
fn engine_label() -> &'static str {
    #[cfg(feature = "nli-burn")]
    {
        "burn/vulkan"
    }
    #[cfg(all(feature = "nli-burn-cpu", not(feature = "nli-burn")))]
    {
        "burn/cpu"
    }
    #[cfg(not(any(feature = "nli-burn", feature = "nli-burn-cpu")))]
    {
        "ort"
    }
}

/// `kbx rerank check --model-dir <dir> [--ep ...] [--ep-device N] [--ep-mem-limit-mb N]`: probe
/// the configured execution provider the same STRICT way `kbx nli check` does
/// (`glossa_nli::probe_rerank_ep`, which builds a session with `.error_on_failure()` so a GPU EP
/// that can't register is reported rather than silently falling back to CPU), then load
/// `InProcessReranker` and score one FIXED, language-neutral, generic (query, [relevant,
/// irrelevant]) pair — deliberately NOT drawn from any corpus/dataset/gold — asserting relevant >
/// irrelevant via [`rerank_check_ok`] (the inversion guard).
///
/// On a burn/wgpu build (`nli-burn`/`nli-burn-cpu`) the same sanity pair is scored via
/// `glossa_nli::InProcessBurnReranker` instead — the burn backend selects its own device, so there
/// is no EP to probe; the report prints `engine = burn/<backend>` in place of `ep_active`, mirroring
/// how `kbx nli check` handles the burn engine.
///
/// Printing the diagnosis IS the deliverable — a NOT READY verdict is not a process error, so this
/// always returns `Ok(())` (a load/inference error while gathering facts still propagates as
/// `Err`, matching `nli_check`'s IO-vs-diagnosis split).
pub fn rerank_check(
    model_dir: PathBuf,
    ep: Vec<String>,
    ep_device: Option<i32>,
    ep_mem_limit_mb: Option<usize>,
) -> Result<()> {
    println!("engine         = {}", engine_label());
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

    #[cfg(any(feature = "nli-burn", feature = "nli-burn-cpu"))]
    {
        // burn/wgpu engine: the backend selects its own device, so there is no EP to probe —
        // `engine_label()` above already reported which backend is compiled in. `ep`/`ep_device`/
        // `ep_mem_limit_mb` are ORT-only knobs and are ignored here.
        let _ = (&ep, ep_device, ep_mem_limit_mb);

        // Fixed, language-neutral, generic sanity pair — no corpus/gold values (see
        // [[no-corpus-values-in-sop]]): a capital-city fact (relevant) vs. an unrelated fruit fact
        // (irrelevant). Same pair as the ORT path above.
        let query = "What is the capital of France?";
        let relevant = "Paris is the capital and most populous city of France.";
        let irrelevant = "Bananas are a good source of potassium.";

        let reranker = glossa_nli::InProcessBurnReranker::load(&model_dir)?;
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
        feature = "nli-burn",
        feature = "nli-burn-cpu",
    )))]
    {
        let _ = (ep, ep_device, ep_mem_limit_mb);
        println!(
            "=> not ready: reranker not available in this kbx build (the reranker itself is \
             CPU-capable in glossa, but this binary wasn't built with a feature that pulls it \
             in) - rebuild kbx with --features nli-directml (or nli-cuda/nli-coreml/nli-rocm/ \
             nli-burn) to enable it"
        );
    }

    Ok(())
}

/// `kbx rerank set`: write `[rerank]` (model_dir + scorer, + pool_size / execution_providers /
/// ep_device / ep_mem_limit_mb when given, + `enabled = true`) into the corpus `ontology.toml` via
/// [`write_rerank_config`] and print what was written. Completes the `download` -> `set` -> `check`
/// workflow so a user never hand-edits TOML. Mirrors [`crate::nli_check::nli_set`].
#[allow(clippy::too_many_arguments)]
pub fn rerank_set(
    path: Option<PathBuf>,
    model_dir: PathBuf,
    scorer: String,
    pool_size: Option<usize>,
    execution_providers: Vec<String>,
    ep_device: Option<i32>,
    ep_mem_limit_mb: Option<usize>,
) -> Result<()> {
    let kbx_paths = crate::workspace::resolve(path);
    let glossa_dir = crate::workspace::glossa_dir(&kbx_paths.root);
    let eps = if execution_providers.is_empty() {
        None
    } else {
        Some(execution_providers.as_slice())
    };
    write_rerank_config(
        &glossa_dir,
        &model_dir,
        &scorer,
        pool_size,
        eps,
        ep_device,
        ep_mem_limit_mb,
    )?;

    let ontology_path = glossa_dir.join("ontology.toml");
    println!(
        "wrote [rerank] enabled = true, scorer = {scorer:?}, model_dir = {} -> {}",
        model_dir.display(),
        ontology_path.display()
    );
    if let Some(p) = pool_size {
        println!("wrote [rerank] pool_size = {p}");
    }
    if let Some(eps) = eps {
        println!("wrote [rerank] execution_providers = {eps:?}");
    }
    if let Some(id) = ep_device {
        println!("wrote [rerank] ep_device = {id}");
    }
    if let Some(mb) = ep_mem_limit_mb {
        println!("wrote [rerank] ep_mem_limit_mb = {mb}");
    }
    println!(
        "run `kbx rerank check --model-dir {}` to confirm readiness.",
        model_dir.display()
    );
    Ok(())
}

/// Write `[rerank].{enabled = true, scorer, model_dir}` (+ `pool_size` / `execution_providers` /
/// `ep_device` / `ep_mem_limit_mb` when given) into `<glossa_dir>/ontology.toml`, preserving every
/// other table/comment. Mirrors [`crate::nli_check::write_nli_config`]'s `toml_edit`
/// preserve-other-keys pattern: parse existing (or empty) into a `DocumentMut`, mutate only the
/// keys this function owns, write the whole document back.
#[allow(clippy::too_many_arguments)]
pub fn write_rerank_config(
    glossa_dir: &Path,
    model_dir: &Path,
    scorer: &str,
    pool_size: Option<usize>,
    execution_providers: Option<&[String]>,
    ep_device: Option<i32>,
    ep_mem_limit_mb: Option<usize>,
) -> Result<()> {
    use toml_edit::{value, Array, DocumentMut, Item, Table};

    let path = glossa_dir.join("ontology.toml");
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    let mut doc: DocumentMut = existing
        .parse()
        .with_context(|| format!("parsing {}", path.display()))?;

    if doc.get("rerank").is_none() {
        doc["rerank"] = Item::Table(Table::new());
    }
    let rr = doc["rerank"]
        .as_table_mut()
        .context("[rerank] is not a table")?;
    rr["enabled"] = value(true);
    rr["scorer"] = value(scorer);
    // toml_edit escapes Windows backslashes; round-trips back through RerankConfig::resolve unchanged.
    rr["model_dir"] = value(model_dir.display().to_string());
    if let Some(p) = pool_size {
        rr["pool_size"] = value(p as i64);
    }
    if let Some(eps) = execution_providers {
        let arr: Array = eps.iter().map(String::as_str).collect();
        rr["execution_providers"] = value(arr);
    }
    if let Some(id) = ep_device {
        rr["ep_device"] = value(id as i64);
    }
    if let Some(mb) = ep_mem_limit_mb {
        rr["ep_mem_limit_mb"] = value(mb as i64);
    }

    std::fs::create_dir_all(glossa_dir)?;
    std::fs::write(&path, doc.to_string())?;
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

    #[test]
    fn rerank_set_writes_table_preserving_siblings() {
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "# keep me\n[verify]\nenabled=true\n[retrieval]\nsim_weight=0.3\n",
        )
        .unwrap();
        write_rerank_config(
            &g,
            std::path::Path::new("/m"),
            "in_process",
            Some(40),
            Some(&["cuda".to_string(), "cpu".to_string()]),
            None,
            Some(1024),
        )
        .unwrap();
        let s = std::fs::read_to_string(g.join("ontology.toml")).unwrap();
        assert!(s.contains("# keep me") && s.contains("[verify]") && s.contains("[retrieval]"));
        assert!(s.contains("[rerank]"));
        assert!(s.contains("scorer = \"in_process\"") && s.contains("model_dir = \"/m\""));
        assert!(s.contains("pool_size = 40") && s.contains("ep_mem_limit_mb = 1024"));
        assert!(s.contains("enabled = true"));
        let doc: toml_edit::DocumentMut = s.parse().unwrap();
        assert_eq!(doc["rerank"]["pool_size"].as_integer(), Some(40));
        assert_eq!(
            doc["retrieval"]["sim_weight"].as_float(),
            Some(0.3),
            "sibling table must survive"
        );
    }
}
