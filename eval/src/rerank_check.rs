//! `kbx rerank check`: a readiness/sanity diagnostic for the in-process cross-encoder reranker,
//! mirroring `kbx nli check`'s ([`crate::nli_check`]) shape: [`rerank_check_ok`] is a PURE
//! function over two scores (unit-tested, no IO); [`rerank_check`] is the thin wrapper that probes
//! the execution provider, loads the model, and scores one fixed sanity pair, then prints the
//! report.
//!
//! Unlike `nli check`, this diagnostic does not resolve a corpus's `[verify.nli]`/`[rerank]`
//! config — the reranker has no corpus-side "is it wired for THIS corpus" question the way the
//! NLI grounding verifier does, so `--model-dir`/`--device`/`--gpu-id`/`--gpu-mem-mb` are
//! taken directly as flags (mirroring `kbx nli set`'s flag names/shapes).
//!
//! `--batch-tokens`/`--intra-threads` join them for the same reason they exist at all: the budget
//! is part of the SESSION's identity (it is in the model cache key) and a configured one is
//! validated at load. Probing without the value the corpus sets would report READY for a session
//! production never builds, and would skip the one check that names a budget the device cannot
//! take.

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

/// `kbx rerank check --model-dir <dir> [--device D] [--gpu-id N] [--gpu-mem-mb N]`: probe
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
    device: Option<String>,
    gpu_id: Option<i32>,
    gpu_mem_mb: Option<usize>,
    batch_tokens: Option<usize>,
    intra_threads: Option<usize>,
) -> Result<()> {
    println!("engine         = {}", engine_label());
    println!("model_dir      = {}", model_dir.display());
    // Printed because the budget is part of the session's identity (it is in the model cache key):
    // a READY verdict is only about the configuration named here, and an operator comparing this
    // against `[rerank]` needs to see which one was probed.
    println!(
        "batch_tokens   = {}",
        batch_tokens.map_or_else(|| "unset (engine default)".to_string(), |n| n.to_string())
    );
    println!(
        "intra_threads  = {}",
        intra_threads.map_or_else(|| "unset (engine default)".to_string(), |n| n.to_string())
    );
    // Discoverability instead of a bigger default: taking ~300 MB per engine unasked was reverted
    // once already, so the knob stays at one row per batch and the command people already run is
    // where they learn a bigger one is measurable.
    println!("               (`kbx rerank fit` measures what a larger batch_tokens buys here)");

    #[cfg(any(
        feature = "nli-directml",
        feature = "nli-coreml",
        feature = "nli-cuda",
        feature = "nli-rocm",
    ))]
    {
        // `--device` is a single value with auto-CPU fallback; expand to the provider list the
        // ORT probe/loader consume (e.g. `cuda` -> `["cuda", "cpu"]`, `None` -> `["cpu"]`).
        let ep = glossa::config_util::expand_device(device.as_deref());
        // GPU execution-provider probe: `InProcessReranker::load` is fail-open (no
        // `.error_on_failure()`), so a GPU build that can't load its runtime silently runs on CPU
        // with no signal. This STRICT probe reports whether the configured GPU EP ACTUALLY
        // initialized — mirrors `nli_check`'s `probe_gpu_ep` block exactly.
        match glossa_nli::probe_rerank_ep(&model_dir, &ep, gpu_id, gpu_mem_mb) {
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

        let reranker = glossa_nli::InProcessReranker::load(
            &model_dir,
            &ep,
            gpu_id,
            gpu_mem_mb,
            batch_tokens,
            intra_threads,
        )?;
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
        // `engine_label()` above already reported which backend is compiled in. `device`/`gpu_id`/
        // `gpu_mem_mb` are ORT-only knobs and are ignored here, as are the session budget and
        // thread count — the burn engines take neither parameter (see the burn arms of
        // `resolve_reranker` / `resolve_scorer`, which say so at load).
        let _ = (&device, gpu_id, gpu_mem_mb, batch_tokens, intra_threads);

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
        let _ = (device, gpu_id, gpu_mem_mb, batch_tokens, intra_threads);
        println!(
            "=> not ready: reranker not available in this kbx build (the reranker itself is \
             CPU-capable in glossa, but this binary wasn't built with a feature that pulls it \
             in) - rebuild kbx with --features nli-directml (or nli-cuda/nli-coreml/nli-rocm/ \
             nli-burn) to enable it"
        );
    }

    Ok(())
}

/// Options for [`rerank_fit`] — the sweep knobs plus the four `check` already takes. One struct
/// because a fit describes the deployment it will run under, and that is more than six parameters.
pub struct FitOpts {
    pub model_dir: PathBuf,
    pub device: Option<String>,
    pub gpu_id: Option<i32>,
    pub gpu_mem_mb: Option<usize>,
    pub max_rows: usize,
    pub seq: Option<usize>,
    pub repeats: usize,
    pub tolerance: f64,
    /// The scorer this corpus configures, so a remote one is refused by name rather than measured.
    pub scorer: Option<String>,
    /// The remote backend, when the scorer is `http`.
    pub backend: Option<String>,
    /// The neighbour engine's configured budget, printed beside the recommendation: two engines on
    /// one device cost the SUM of their batch peaks.
    pub neighbour: Option<(String, Option<usize>)>,
}

/// Measure what a larger batch buys on THIS device, and print the budget to configure.
///
/// Refuses rather than measures when the number would be about the wrong thing — a remote scorer, or
/// a GPU provider that did not actually bind (loading is fail-open, so that session is on CPU, where
/// a cross-encoder pool is a thermal hazard). Prints; writing is `rerank set --batch-tokens`'s job,
/// which keeps the `download → set → check` split intact.
pub fn rerank_fit(opts: &FitOpts) -> Result<()> {
    #[cfg(any(
        feature = "nli-directml",
        feature = "nli-coreml",
        feature = "nli-cuda",
        feature = "nli-rocm",
    ))]
    {
        use crate::fit::{fit_refusal, fit_report, select, sweep, RerankTarget, SweepOpts};

        let ep = glossa::config_util::expand_device(opts.device.as_deref());
        // The same STRICT probe `check` runs, and for the same reason: `load` is fail-open, so a
        // configured GPU that failed to register would otherwise be measured as if it were a GPU.
        let ep_bound = matches!(
            glossa_nli::probe_rerank_ep(&opts.model_dir, &ep, opts.gpu_id, opts.gpu_mem_mb),
            Ok(Some(_))
        );
        if let Some(why) = fit_refusal(opts.scorer.as_deref(), opts.backend.as_deref(), ep_bound) {
            anyhow::bail!("refusing to fit: {why}");
        }
        let seq = crate::fit::effective_seq(opts.seq, glossa_nli::harness::DEFAULT_MAX_SEQ_LEN);
        let tolerance = crate::fit::sanitize_tolerance(opts.tolerance);
        let max_rows = opts
            .max_rows
            .clamp(1, glossa_nli::harness::NLI_BATCH_MAX_ROWS);
        // Loaded at one row per batch: the sweep supplies each size explicitly, so the session's own
        // budget must not be the thing under measurement.
        let engine = glossa_nli::InProcessReranker::load(
            &opts.model_dir,
            &ep,
            opts.gpu_id,
            opts.gpu_mem_mb,
            None,
            None,
        )?;
        let samples = sweep(
            &RerankTarget {
                engine: &engine,
                seq,
            },
            &SweepOpts {
                max_rows,
                seq,
                repeats: opts.repeats,
                tolerance,
            },
        )?;
        let outcome = select(&samples, tolerance);
        let hint = outcome.as_ref().map(|o| {
            format!(
                "kbx rerank set --model-dir {} --batch-tokens {}",
                opts.model_dir.display(),
                crate::fit::budget_tokens(o.chosen, seq)
            )
        });
        print!(
            "{}",
            fit_report(
                "reranker",
                seq,
                outcome.as_ref(),
                opts.neighbour
                    .as_ref()
                    .map(|(name, tokens)| (name.as_str(), *tokens)),
                hint.as_deref(),
                // `kbx` never applies: it prints, and `set` writes.
                false,
            )
        );
        // Part 1 added this announcement for exactly this case: a budget above what the planner can
        // spend behaves as the largest it can, so a recommendation nobody can act on gets a note
        // rather than silence.
        if let Some(o) = outcome.as_ref() {
            let tokens = crate::fit::budget_tokens(o.chosen, seq);
            if let Some(effective) = glossa_nli::harness::budget_beyond_planner(Some(tokens)) {
                println!(
                    "  note         {tokens} exceeds what the planner can spend; it behaves as {effective}"
                );
            }
        }

        Ok(())
    }
    #[cfg(not(any(
        feature = "nli-directml",
        feature = "nli-coreml",
        feature = "nli-cuda",
        feature = "nli-rocm",
    )))]
    {
        let _ = opts;
        anyhow::bail!(
            "this build has no GPU execution provider compiled in, and a fit on CPU is refused \
             (known answer, and a thermal hazard): rebuild with --features nli-cuda / nli-directml"
        )
    }
}

/// `kbx rerank set`: write `[rerank]` (model_dir + scorer, + pool_size / device /
/// gpu_id / gpu_mem_mb when given, + `enabled = true`) into the corpus `ontology.toml` via
/// [`write_rerank_config`] and print what was written. Completes the `download` -> `set` -> `check`
/// workflow so a user never hand-edits TOML. Mirrors [`crate::nli_check::nli_set`].
#[allow(clippy::too_many_arguments)]
pub fn rerank_set(
    path: Option<PathBuf>,
    model_dir: PathBuf,
    scorer: String,
    pool_size: Option<usize>,
    device: Option<String>,
    gpu_id: Option<i32>,
    gpu_mem_mb: Option<usize>,
    batch_tokens: Option<usize>,
) -> Result<()> {
    let kbx_paths = crate::workspace::resolve(path);
    let glossa_dir = crate::workspace::glossa_dir(&kbx_paths.root);
    write_rerank_config(
        &glossa_dir,
        &model_dir,
        &scorer,
        pool_size,
        device.as_deref(),
        gpu_id,
        gpu_mem_mb,
        batch_tokens,
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
    if let Some(d) = &device {
        println!("wrote [rerank] device = {d:?}");
    }
    if let Some(id) = gpu_id {
        println!("wrote [rerank] gpu_id = {id}");
    }
    if let Some(mb) = gpu_mem_mb {
        println!("wrote [rerank] gpu_mem_mb = {mb}");
    }
    println!(
        "run `kbx rerank check --model-dir {}` to confirm readiness.",
        model_dir.display()
    );
    Ok(())
}

/// Write `[rerank].{enabled = true, scorer, model_dir}` (+ `pool_size` / `device` /
/// `gpu_id` / `gpu_mem_mb` when given) into `<glossa_dir>/ontology.toml`, preserving every
/// other table/comment. Mirrors [`crate::nli_check::write_nli_config`]'s `toml_edit`
/// preserve-other-keys pattern: parse existing (or empty) into a `DocumentMut`, mutate only the
/// keys this function owns, write the whole document back.
#[allow(clippy::too_many_arguments)]
pub fn write_rerank_config(
    glossa_dir: &Path,
    model_dir: &Path,
    scorer: &str,
    pool_size: Option<usize>,
    device: Option<&str>,
    gpu_id: Option<i32>,
    gpu_mem_mb: Option<usize>,
    batch_tokens: Option<usize>,
) -> Result<()> {
    use toml_edit::{value, DocumentMut, Item, Table};

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
    if let Some(d) = device {
        rr["device"] = value(d);
    }
    if let Some(id) = gpu_id {
        rr["gpu_id"] = value(id as i64);
    }
    if let Some(mb) = gpu_mem_mb {
        rr["gpu_mem_mb"] = value(mb as i64);
    }
    // Written only when given: an unset budget is one row per batch, and `rerank fit` is what
    // decides whether a bigger one is worth the resident memory on this device.
    if let Some(bt) = batch_tokens {
        rr["batch_tokens"] = value(bt as i64);
    }

    std::fs::create_dir_all(glossa_dir)?;
    std::fs::write(&path, doc.to_string())?;
    Ok(())
}

/// Probe a remote reranker endpoint (`kbx rerank check --endpoint`): score one generic
/// relevant-vs-irrelevant pair (NO corpus values, see [[no-corpus-values-in-sop]]) and report
/// reachability + the inversion guard. Fail-open — a transport error yields an UNREACHABLE line,
/// never a panic.
#[cfg(feature = "http-scorer")]
pub fn probe_remote_rerank(
    endpoint: &str,
    timeout_ms: u64,
    api_key: Option<String>,
    backend: &str,
    model: Option<String>,
) -> String {
    use glossa::retrieve::rerank::Reranker;
    let r = match glossa::http_scorer::client::new_ureq_reranker(
        endpoint.to_string(),
        timeout_ms,
        api_key,
        backend,
        model,
    ) {
        Ok(r) => r,
        Err(e) => return format!("remote = {backend} @ {endpoint} MISCONFIGURED: {e}"),
    };
    match r.rerank(
        "What is the capital of France?",
        &[
            "Paris is the capital of France.",
            "Bananas are a good source of potassium.",
        ],
    ) {
        Ok(s) if s.len() == 2 => format!(
            "remote = {backend} @ {endpoint} reachable (relevant {} irrelevant)",
            if rerank_check_ok(s[0], s[1]) {
                ">"
            } else {
                "<="
            }
        ),
        Ok(s) => format!(
            "remote = {backend} @ {endpoint} BAD RESPONSE ({} scores, expected 2)",
            s.len()
        ),
        Err(e) => format!("remote = {backend} @ {endpoint} UNREACHABLE: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "http-scorer")]
    #[test]
    fn remote_probe_line_reports_unreachable_on_transport_error() {
        let line = probe_remote_rerank("http://127.0.0.1:1", 200, None, "kbi", None);
        assert!(line.contains("UNREACHABLE") || line.contains("BAD RESPONSE"));
    }

    #[cfg(feature = "http-scorer")]
    #[test]
    fn remote_probe_line_reports_misconfigured_on_unknown_backend() {
        let line = probe_remote_rerank("http://127.0.0.1:1", 200, None, "vlm", None);
        assert!(line.contains("MISCONFIGURED"), "line was: {line}");
    }

    #[test]
    fn rerank_check_verdict_ok_when_relevant_wins() {
        assert!(rerank_check_ok(2.5, -1.0));
    }

    #[test]
    fn rerank_check_verdict_bad_when_inverted() {
        assert!(!rerank_check_ok(0.1, 0.9));
    }

    /// `rerank fit` prints a number and names the `set` that writes it; this is that write. The key
    /// round-trips through the resolver `resolve_reranker` reads, and an unset budget writes nothing
    /// — one row per batch stays the default nobody paid for.
    #[test]
    fn rerank_set_writes_the_batch_budget_only_when_given() {
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        write_rerank_config(
            &g,
            std::path::Path::new("/m"),
            "in_process",
            None,
            None,
            None,
            None,
            Some(4096),
        )
        .unwrap();
        let s = std::fs::read_to_string(g.join("ontology.toml")).unwrap();
        assert!(s.contains("batch_tokens = 4096"), "{s}");
        assert_eq!(
            glossa::retrieve::config::RerankConfig::resolve(&g).batch_tokens,
            Some(4096)
        );

        let bare = dir.path().join("bare");
        write_rerank_config(
            &bare,
            std::path::Path::new("/m"),
            "in_process",
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let s = std::fs::read_to_string(bare.join("ontology.toml")).unwrap();
        assert!(!s.contains("batch_tokens"), "unset writes no key: {s}");
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
            Some("cuda"),
            None,
            Some(1024),
            None,
        )
        .unwrap();
        let s = std::fs::read_to_string(g.join("ontology.toml")).unwrap();
        assert!(s.contains("# keep me") && s.contains("[verify]") && s.contains("[retrieval]"));
        assert!(s.contains("[rerank]"));
        assert!(s.contains("scorer = \"in_process\"") && s.contains("model_dir = \"/m\""));
        assert!(s.contains("device = \"cuda\""));
        assert!(s.contains("pool_size = 40") && s.contains("gpu_mem_mb = 1024"));
        assert!(s.contains("enabled = true"));
        let doc: toml_edit::DocumentMut = s.parse().unwrap();
        assert_eq!(doc["rerank"]["pool_size"].as_integer(), Some(40));
        assert_eq!(
            doc["retrieval"]["sim_weight"].as_float(),
            Some(0.3),
            "sibling table must survive"
        );

        // Substring/toml_edit checks don't prove the RUNTIME retrieval path can read the keys back.
        // Resolve through `RerankConfig::resolve` — the same read `resolve_reranker` uses — so this
        // test proves parity with the nli twin (`write_nli_config_roundtrips_into_ontology`), not
        // just that bytes landed in the file.
        let cfg = glossa::retrieve::config::RerankConfig::resolve(&g);
        assert!(cfg.is_active(), "written [rerank] must resolve as active");
        assert_eq!(cfg.scorer.as_deref(), Some("in_process"));
        assert_eq!(cfg.model_dir, Some(std::path::PathBuf::from("/m")));
        assert_eq!(cfg.pool_size, 40);
        assert_eq!(
            cfg.execution_providers,
            vec!["cuda".to_string(), "cpu".to_string()]
        );
        assert_eq!(cfg.ep_mem_limit_mb, Some(1024));
    }
}
