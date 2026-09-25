//! `kbx nli check`: a readiness doctor for the NLI answer-grounding verifier. `[verify.nli]`'s
//! design fails open (any missing piece silently falls back to AC-only, see
//! `glossa::gate::resolve_scorer`'s doc comment), which is exactly the problem — a deployment can
//! believe NLI is protecting it while it silently isn't. This module gathers the observable facts
//! and turns them into a single ordered verdict so that silence becomes a diagnosis.
//!
//! Split like `download.rs`: [`nli_verdict`] is a PURE function over [`NliFacts`] (unit-tested, no
//! IO); [`nli_check`] is the thin wrapper that gathers the facts (config resolution, filesystem
//! probes, an optional model load + sanity entailment) and prints the report.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use glossa::gate::config::VerifyConfig;

/// The observable facts about NLI readiness for one corpus, gathered by [`nli_check`] and judged
/// by [`nli_verdict`].
pub struct NliFacts {
    /// `[verify] mode`, normalized to `"ac"` | `"nli"` | `"combined"`.
    pub mode: String,
    /// Whether THIS kbx binary was built with the `nli` feature (`cfg!(feature = "nli")`) — not a
    /// corpus property, but a build-time fact that gates everything else.
    pub feature_built: bool,
    /// `[verify.nli] scorer` as configured (`Some("in_process")` is the only value that resolves
    /// today).
    pub scorer: Option<String>,
    /// `[verify.nli] model_dir` as configured.
    pub model_dir: Option<PathBuf>,
    /// `model_dir` exists AND contains both `model.onnx` and `tokenizer.json`.
    pub model_dir_exists: bool,
    /// `ORT_DYLIB_PATH` is set AND the path it names exists on disk.
    pub dylib_set: bool,
    /// `glossa::gate::resolve_scorer(&cfg).is_some()` — a scorer actually loaded.
    pub scorer_built: bool,
    /// Outcome of the language-agnostic lexical-identity sanity probe run against a loaded scorer
    /// (see [`Probe`]).
    pub probe: Probe,
}

/// Outcome of `nli_check`'s language-agnostic lexical-identity sanity probe: `entail(premise,
/// [identical, disjoint])` over one hard-coded pair whose "support" hypothesis is IDENTICAL to the
/// premise and whose "contra" hypothesis shares no tokens with it (NATO-phonetic tokens). `NotRun`
/// when no scorer resolved, so the probe never fired. `Errored` when the `entail()` call itself
/// returned `Err` (session, tokenizer, or output-shape failure) — this DOES block READY, since it
/// proves a loaded scorer can't actually run inference. `Ran { identity_ge_disjoint }` proves the
/// opposite: session, tokenizer, and 3-way output all work end to end. The bool inside `Ran` does
/// NOT gate READY (see [`nli_verdict`]): a well-trained model of any language should rank the
/// identical hypothesis above the disjoint one, so a `false` hints at a miswired scorer (e.g. a
/// wrong entail_index) rather than a language mismatch — it's surfaced as an advisory line only.
#[derive(Debug, Clone, PartialEq)]
pub enum Probe {
    NotRun,
    Errored(String),
    Ran { identity_ge_disjoint: bool },
}

/// Judge [`NliFacts`] into a `(ready, verdict_line)` pair. READY only when every gate passes, in
/// this order (the returned line names the FIRST one that fails): mode isn't `ac` -> feature `nli`
/// is built -> scorer is `"in_process"` -> `model_dir` is configured -> `model_dir` has both model
/// files -> a scorer actually loaded -> the sanity probe RAN without error. (`ORT_DYLIB_PATH` is
/// NOT a gate — see the note at the `scorer_built` check.) The probe's identity-vs-disjoint
/// comparison is advisory only and never blocks READY (see [`Probe`]'s doc comment for why: a `false`
/// points at a miswired scorer, not a language mismatch). Pure — no IO, no panics; safe to call with
/// any combination of facts (including ones
/// that couldn't co-occur in practice, e.g. `probe: Probe::Ran { .. }` with `scorer_built: false`).
pub fn nli_verdict(f: &NliFacts) -> (bool, String) {
    if f.mode == "ac" {
        return (
            false,
            "not ready: mode is ac (verifier runs AC-only; set [verify] mode = \"nli\" or \
             \"combined\" to use NLI)"
                .to_string(),
        );
    }
    if !f.feature_built {
        return (
            false,
            "not ready: feature nli not built (rebuild kbx with --features nli)".to_string(),
        );
    }
    if f.scorer.as_deref() != Some("in_process") {
        return (
            false,
            format!(
                "not ready: scorer is {} (expected \"in_process\")",
                f.scorer
                    .as_deref()
                    .map(|s| format!("{s:?}"))
                    .unwrap_or_else(|| "unset".to_string())
            ),
        );
    }
    if f.model_dir.is_none() {
        return (
            false,
            "not ready: no model_dir configured ([verify.nli] model_dir)".to_string(),
        );
    }
    if !f.model_dir_exists {
        return (
            false,
            "not ready: model_dir missing model weights and/or tokenizer.json (run `kbx nli \
             download`)"
                .to_string(),
        );
    }
    // NB: `ORT_DYLIB_PATH` is NOT gated here. A `download-binaries` build links the ONNX Runtime
    // into the binary, so the scorer loads with the var unset; and a `load-dynamic` build that
    // genuinely can't find the runtime simply fails `InProcessNli::load` -> `scorer_built == false`,
    // caught below. `dylib_set` stays an INFORMATIONAL line in the printed report, never a gate.
    if !f.scorer_built {
        return (
            false,
            "not ready: scorer failed to load (ORT runtime missing? see the load-failure log line \
             above; a load-dynamic build needs ORT_DYLIB_PATH)"
                .to_string(),
        );
    }
    match &f.probe {
        Probe::Ran { .. } => (true, "READY".to_string()),
        Probe::Errored(msg) => (
            false,
            format!("not ready: scorer loaded but inference failed: {msg}"),
        ),
        Probe::NotRun => (
            false,
            "not ready: sanity probe did not run (no scorer loaded)".to_string(),
        ),
    }
}

/// Whether `dir` holds the model weights THIS engine needs: the burn engine loads a
/// `.safetensors` file (Plan 4), the ORT engine loads `model.onnx`. `tokenizer.json` is checked
/// separately by the caller (both engines need it).
#[cfg(any(feature = "nli-burn", feature = "nli-burn-cpu"))]
fn model_weights_present(dir: &Path) -> bool {
    if dir.join("model.safetensors").is_file() {
        return true;
    }
    std::fs::read_dir(dir).is_ok_and(|rd| {
        rd.filter_map(|e| e.ok()).any(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                .is_some_and(|x| x.eq_ignore_ascii_case("safetensors"))
        })
    })
}

#[cfg(not(any(feature = "nli-burn", feature = "nli-burn-cpu")))]
fn model_weights_present(dir: &Path) -> bool {
    // Mirror `resolve_model_file`: `model.onnx`, else any lone `*.onnx` (e.g. `model.fp16.onnx`,
    // the recommended fp16 download). Checking only `model.onnx` false-negatived an fp16-only dir
    // even though the ORT engine loads it fine.
    if dir.join("model.onnx").is_file() {
        return true;
    }
    std::fs::read_dir(dir).is_ok_and(|rd| {
        rd.filter_map(|e| e.ok()).any(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                .is_some_and(|x| x.eq_ignore_ascii_case("onnx"))
        })
    })
}

/// Human label for the compiled inference engine, shown in the `check` report.
fn engine_label() -> &'static str {
    #[cfg(feature = "nli-burn")]
    {
        "burn-wgpu (vulkan)"
    }
    #[cfg(all(feature = "nli-burn-cpu", not(feature = "nli-burn")))]
    {
        "burn (ndarray/cpu)"
    }
    #[cfg(not(any(feature = "nli-burn", feature = "nli-burn-cpu")))]
    {
        "ort"
    }
}

/// `kbx nli check <path>`: resolve the corpus's `[verify.nli]` config the same way the runtime gate
/// does, gather [`NliFacts`], print a readable block, then the verdict from [`nli_verdict`].
/// Printing the diagnosis IS the deliverable — a non-ready verdict is not a process error, so this
/// always returns `Ok(())` (a resolution/IO failure while gathering facts still propagates).
pub fn nli_check(path: Option<PathBuf>) -> Result<()> {
    let kbx_paths = crate::workspace::resolve(path);
    let glossa_dir = crate::workspace::glossa_dir(&kbx_paths.root);
    let cfg = VerifyConfig::resolve(&glossa_dir);

    let mode = match cfg.mode {
        glossa::gate::VerifyMode::Ac => "ac",
        glossa::gate::VerifyMode::Nli => "nli",
        glossa::gate::VerifyMode::Combined => "combined",
    }
    .to_string();

    let model_dir_exists = cfg
        .model_dir
        .as_ref()
        .is_some_and(|d| model_weights_present(d) && d.join("tokenizer.json").is_file());

    let dylib_path = std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from);
    let dylib_set = dylib_path.as_ref().is_some_and(|p| p.exists());

    let scorer = glossa::gate::resolve_scorer(&cfg);
    let scorer_built = scorer.is_some();
    let probe = match &scorer {
        None => Probe::NotRun,
        // Language-agnostic lexical-identity smoke test: a premise entails an IDENTICAL hypothesis
        // (trivially, in any language) more than a token-disjoint one. `identity_ge_disjoint` should
        // hold for any working scorer regardless of the model's language; a `false` points at a
        // miswired scorer (e.g. a wrong entail_index that inverts the classes) or a degenerate model,
        // not a language mismatch. Uses NATO-phonetic tokens (an international, language-neutral set).
        Some(s) => match s.entail(
            "alpha bravo charlie delta",
            &["alpha bravo charlie delta", "november oscar papa quebec"],
        ) {
            Ok(scores) if scores.len() >= 2 => Probe::Ran {
                identity_ge_disjoint: scores[0] > scores[1],
            },
            // `entail()` returns one score per hypothesis (2 here); an `Ok` with fewer values
            // would mean the crate's own length contract broke — treat it like an error rather
            // than silently guess a comparison result.
            Ok(scores) => Probe::Errored(format!(
                "sanity probe returned {} score(s), expected 2",
                scores.len()
            )),
            Err(e) => Probe::Errored(e.to_string()),
        },
    };

    let facts = NliFacts {
        mode,
        // Any kb-eval feature that compiles the NLI engine counts — the GPU builds enable it via
        // `nli-cuda`/`nli-directml`/... (which route through glossa's `nli-dynamic`), NOT kb-eval's
        // own `nli`, so checking `nli` alone would misreport a GPU build as "feature not built".
        feature_built: cfg!(any(
            feature = "nli",
            feature = "nli-cuda",
            feature = "nli-directml",
            feature = "nli-coreml",
            feature = "nli-rocm",
            feature = "nli-burn",
            feature = "nli-burn-cpu",
        )),
        scorer: cfg.scorer.clone(),
        model_dir: cfg.model_dir.clone(),
        model_dir_exists,
        dylib_set,
        scorer_built,
        probe,
    };

    println!("mode           = {}", facts.mode);
    println!("engine         = {}", engine_label());
    println!(
        "feature nli    = {}",
        if facts.feature_built {
            "built"
        } else {
            "NOT built"
        }
    );
    println!(
        "scorer         = {}",
        facts.scorer.as_deref().unwrap_or("(unset)")
    );
    match &facts.model_dir {
        Some(d) => println!(
            "model_dir      = {} (exists: {})",
            d.display(),
            if facts.model_dir_exists {
                "weights + tokenizer.json present"
            } else {
                "NO — missing model weights and/or tokenizer.json"
            }
        ),
        None => println!("model_dir      = (unset)"),
    }
    if let Some(id) = cfg.execution_provider_device {
        println!("ep_device      = {id}");
    }
    if let Some(mb) = cfg.execution_provider_mem_limit_mb {
        println!("ep_mem_limit   = {mb} MB");
    }
    println!(
        "ORT_DYLIB_PATH = {}",
        match &dylib_path {
            Some(p) if facts.dylib_set => format!("{} (exists)", p.display()),
            Some(p) => format!("{} (MISSING)", p.display()),
            None => "(unset)".to_string(),
        }
    );
    println!(
        "scorer         = {}",
        if facts.scorer_built {
            "loaded"
        } else {
            "not loaded"
        }
    );
    match &facts.probe {
        Probe::Ran {
            identity_ge_disjoint,
        } => println!(
            "sanity         = ran (identity>=disjoint: {identity_ge_disjoint})  [language-agnostic \
             lexical check; false suggests a miswired scorer (e.g. wrong entail_index), not a \
             language mismatch]"
        ),
        Probe::Errored(msg) => println!("sanity         = ERRORED: {msg}"),
        Probe::NotRun => println!("sanity         = (skipped — no scorer loaded)"),
    }

    // GPU execution-provider probe: the serving scorer is fail-open (no `.error_on_failure()`), so a
    // CUDA/DirectML build that can't load its runtime silently runs on CPU with no signal. This
    // STRICT probe (`glossa_nli::probe_gpu_ep`) rebuilds a minimal session with `.error_on_failure()`
    // to report whether the configured GPU EP ACTUALLY initialized. Only reachable when the real ORT
    // scorer path is compiled — i.e. the features that pull in the direct `glossa-nli` dep (plain
    // `nli`/default builds don't, and are CPU-only anyway) — and only run when the model weights
    // exist to build a session from.
    #[cfg(any(
        feature = "nli-directml",
        feature = "nli-coreml",
        feature = "nli-cuda",
        feature = "nli-rocm",
        feature = "nli-burn",
        feature = "nli-burn-cpu",
    ))]
    if let Some(md) = cfg.model_dir.as_ref().filter(|d| model_weights_present(d)) {
        match glossa_nli::probe_gpu_ep(
            md,
            cfg.entail_index,
            &cfg.execution_providers,
            cfg.execution_provider_device,
            cfg.execution_provider_mem_limit_mb,
        ) {
            Ok(Some(ep)) => println!("ep_active      = {ep} (initialized)"),
            Ok(None) => println!("ep_active      = cpu (no GPU EP configured)"),
            Err(e) => {
                // On failure the probe can't return the EP name, so label it from the requested
                // provider list (first non-cpu entry) to match "cuda REQUESTED but FAILED".
                let requested = cfg
                    .execution_providers
                    .iter()
                    .find(|p| p.as_str() != "cpu")
                    .map(String::as_str)
                    .unwrap_or("gpu");
                println!(
                    "ep_active      = {requested} REQUESTED but FAILED to init -> running on CPU: {e}"
                );
            }
        }
    }

    let (_, verdict) = nli_verdict(&facts);
    println!("=> {verdict}");
    Ok(())
}

/// `kbx nli set <path> --model-dir <dir> [--scorer ...] [--entail-index N] [--mode ...] [--ep
/// ...] [--ep-device N] [--ep-mem-limit-mb N]`: resolve the glossa dir the same way [`nli_check`]
/// does, then write
/// `[verify.nli]` into the corpus `ontology.toml` via [`write_nli_config`] and print what was
/// written + a `kbx nli check` hint. Completes the `download` -> `set` -> `check` workflow so a user
/// never hand-edits TOML. `execution_providers` is written only when non-empty — an empty list
/// leaves the ontology key untouched (mirrors `entail_index`/`mode`/`ep_device`'s `Option` "only if
/// given" convention).
#[allow(clippy::too_many_arguments)]
pub fn nli_set(
    path: Option<PathBuf>,
    model_dir: PathBuf,
    scorer: String,
    entail_index: Option<usize>,
    mode: Option<String>,
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
    write_nli_config(
        &glossa_dir,
        &model_dir,
        &scorer,
        entail_index,
        mode.as_deref(),
        eps,
        ep_device,
        ep_mem_limit_mb,
    )?;

    let ontology_path = glossa_dir.join("ontology.toml");
    println!(
        "wrote [verify.nli] scorer = {scorer:?}, model_dir = {} -> {}",
        model_dir.display(),
        ontology_path.display()
    );
    if let Some(ei) = entail_index {
        println!("wrote [verify.nli] entail_index = {ei}");
    }
    if let Some(m) = &mode {
        println!("wrote [verify] mode = {m:?}");
    }
    if let Some(eps) = eps {
        println!("wrote [verify.nli] execution_providers = {eps:?}");
    }
    if let Some(id) = ep_device {
        println!("wrote [verify.nli] ep_device = {id}");
    }
    if let Some(mb) = ep_mem_limit_mb {
        println!("wrote [verify.nli] ep_mem_limit_mb = {mb}");
    }
    println!("run `kbx nli check` to confirm readiness.");
    Ok(())
}

/// Write `[verify.nli].{scorer,model_dir}` (+ `entail_index` when given, + `[verify].mode` when
/// given, + `execution_providers` when given, + `ep_device` when given, + `ep_mem_limit_mb` when
/// given) into `<glossa_dir>/ontology.toml`, preserving every
/// other table/comment. Mirrors `calibrate::write_threshold`'s established preserve-other-keys
/// pattern: parse the existing file (or start from an empty document when absent) into a
/// `toml_edit::DocumentMut`, mutate only the keys this function owns, then write the whole
/// document back. Does NOT touch `[verify.nli.threshold]` (calibration's own keys) or any other
/// table. `execution_providers` is written as a TOML array; `None` leaves the key untouched.
#[allow(clippy::too_many_arguments)]
pub fn write_nli_config(
    glossa_dir: &Path,
    model_dir: &Path,
    scorer: &str,
    entail_index: Option<usize>,
    mode: Option<&str>,
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

    if doc.get("verify").is_none() {
        doc["verify"] = Item::Table(Table::new());
    }
    let verify = doc["verify"]
        .as_table_mut()
        .context("[verify] is not a table")?;

    if let Some(m) = mode {
        verify["mode"] = value(m);
    }

    if verify.get("nli").is_none() {
        verify["nli"] = Item::Table(Table::new());
    }
    let nli = verify["nli"]
        .as_table_mut()
        .context("[verify.nli] is not a table")?;
    nli["scorer"] = value(scorer);
    // toml_edit escapes Windows backslashes in the emitted string; this round-trips back through
    // `VerifyConfig::resolve`'s `String` -> `PathBuf` unchanged.
    nli["model_dir"] = value(model_dir.display().to_string());
    if let Some(ei) = entail_index {
        nli["entail_index"] = value(ei as i64);
    }
    if let Some(eps) = execution_providers {
        let arr: Array = eps.iter().map(String::as_str).collect();
        nli["execution_providers"] = value(arr);
    }
    if let Some(id) = ep_device {
        nli["ep_device"] = value(id as i64);
    }
    if let Some(mb) = ep_mem_limit_mb {
        nli["ep_mem_limit_mb"] = value(mb as i64);
    }

    std::fs::create_dir_all(glossa_dir)?;
    std::fs::write(&path, doc.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ready_facts() -> NliFacts {
        NliFacts {
            mode: "nli".to_string(),
            feature_built: true,
            scorer: Some("in_process".to_string()),
            model_dir: Some(PathBuf::from("/models/rubert-nli")),
            model_dir_exists: true,
            dylib_set: true,
            scorer_built: true,
            probe: Probe::Ran {
                identity_ge_disjoint: true,
            },
        }
    }

    #[test]
    fn all_good_and_probe_ran_is_ready() {
        let (ready, line) = nli_verdict(&ready_facts());
        assert!(ready);
        assert_eq!(line, "READY");
    }

    #[test]
    fn mode_ac_blocks_first() {
        let mut f = ready_facts();
        f.mode = "ac".to_string();
        // Also break a later gate to prove mode is checked FIRST regardless.
        f.feature_built = false;
        let (ready, line) = nli_verdict(&f);
        assert!(!ready);
        assert!(line.contains("mode is ac"), "line was: {line}");
    }

    #[test]
    fn combined_mode_is_accepted_like_nli() {
        let mut f = ready_facts();
        f.mode = "combined".to_string();
        let (ready, _) = nli_verdict(&f);
        assert!(ready);
    }

    #[test]
    fn feature_not_built_blocks() {
        let mut f = ready_facts();
        f.feature_built = false;
        let (ready, line) = nli_verdict(&f);
        assert!(!ready);
        assert!(line.contains("feature nli not built"), "line was: {line}");
    }

    #[test]
    fn scorer_not_in_process_blocks() {
        let mut f = ready_facts();
        f.scorer = Some("http".to_string());
        let (ready, line) = nli_verdict(&f);
        assert!(!ready);
        assert!(line.contains("scorer is"), "line was: {line}");
    }

    #[test]
    fn no_model_dir_blocks() {
        let mut f = ready_facts();
        f.model_dir = None;
        let (ready, line) = nli_verdict(&f);
        assert!(!ready);
        assert!(line.contains("no model_dir"), "line was: {line}");
    }

    #[test]
    fn model_dir_missing_files_blocks() {
        let mut f = ready_facts();
        f.model_dir_exists = false;
        let (ready, line) = nli_verdict(&f);
        assert!(!ready);
        assert!(line.contains("model_dir missing"), "line was: {line}");
    }

    #[test]
    fn dylib_unset_does_not_block_when_scorer_loaded() {
        // A download-binaries build links the runtime in, so the scorer loads with ORT_DYLIB_PATH
        // unset. `check` must NOT report not-ready on the env var alone when the scorer actually
        // loaded and the probe ran — that was a stale load-dynamic-era assumption.
        let mut f = ready_facts();
        f.dylib_set = false;
        let (ready, line) = nli_verdict(&f);
        assert!(
            ready,
            "dylib_set=false must not block a loaded+running scorer; line: {line}"
        );
        assert_eq!(line, "READY");
    }

    #[test]
    fn scorer_failed_to_load_blocks() {
        let mut f = ready_facts();
        f.scorer_built = false;
        f.probe = Probe::NotRun;
        let (ready, line) = nli_verdict(&f);
        assert!(!ready);
        assert!(line.contains("scorer failed to load"), "line was: {line}");
    }

    /// `Probe::Ran { identity_ge_disjoint: false }` proves the scorer loaded and ran real inference
    /// end to end (session + tokenizer + 3-way output) — the comparison itself is advisory only
    /// (a `false` points at a miswired scorer, not a language mismatch), so it must NOT block READY.
    #[test]
    fn probe_ran_with_false_comparison_is_still_ready() {
        let mut f = ready_facts();
        f.probe = Probe::Ran {
            identity_ge_disjoint: false,
        };
        let (ready, line) = nli_verdict(&f);
        assert!(ready, "line was: {line}");
        assert_eq!(line, "READY");
    }

    #[test]
    fn probe_errored_blocks() {
        let mut f = ready_facts();
        f.probe = Probe::Errored("onnx session returned no outputs".to_string());
        let (ready, line) = nli_verdict(&f);
        assert!(!ready);
        assert!(line.contains("inference failed"), "line was: {line}");
        assert!(
            line.contains("onnx session returned no outputs"),
            "line was: {line}"
        );
    }

    #[test]
    fn probe_not_run_blocks_when_scorer_somehow_built() {
        // Synthetic combination (wouldn't occur via nli_check's own wiring, since the probe only
        // runs when a scorer loaded) — the pure function still handles it deterministically.
        let mut f = ready_facts();
        f.probe = Probe::NotRun;
        let (ready, line) = nli_verdict(&f);
        assert!(!ready);
        assert!(
            line.contains("sanity probe did not run"),
            "line was: {line}"
        );
    }

    /// `write_nli_config` round-trips model_dir/scorer/entail_index/mode into a fresh
    /// `ontology.toml` (mirrors `calibrate::write_threshold_roundtrips_into_ontology`) — and, since
    /// substring-checking the raw TOML doesn't prove the RUNTIME gate can read it back, this also
    /// resolves the written config the same way `nli_check`/the runtime gate do
    /// (`VerifyConfig::resolve`) and asserts the parsed fields match what was written.
    #[test]
    fn write_nli_config_roundtrips_into_ontology() {
        let dir = tempfile::tempdir().unwrap();
        let glossa = dir.path().join(".glossa"); // write_nli_config writes <glossa_dir>/ontology.toml
        write_nli_config(
            &glossa,
            Path::new("/models/rubert-nli"),
            "in_process",
            Some(2),
            Some("nli"),
            None,
            None,
            None,
        )
        .unwrap();
        let o = std::fs::read_to_string(glossa.join("ontology.toml")).unwrap();
        assert!(o.contains("[verify.nli]"));
        assert!(o.contains("scorer = \"in_process\""));
        assert!(o.contains("model_dir"));
        assert!(o.contains("rubert-nli"));
        assert!(o.contains("entail_index = 2"));
        assert!(o.contains("mode = \"nli\""));
        // Not touched.
        assert!(!o.contains("[verify.nli.threshold]"));

        // Real round-trip: resolve through the same path the runtime gate uses.
        let cfg = VerifyConfig::resolve(&glossa);
        assert_eq!(cfg.scorer.as_deref(), Some("in_process"));
        assert_eq!(cfg.model_dir, Some(PathBuf::from("/models/rubert-nli")));
        assert_eq!(cfg.entail_index, 2);
        assert_eq!(cfg.mode, glossa::gate::VerifyMode::Nli);
    }

    /// `model_dir` written as a literal Windows path (backslashes) round-trips unchanged through
    /// `toml_edit`'s string escaping and back out through `VerifyConfig::resolve`'s `PathBuf`
    /// parsing — this is a path-string test, not a locale/Cyrillic one (kept English-only).
    #[test]
    fn write_nli_config_roundtrips_windows_path() {
        let dir = tempfile::tempdir().unwrap();
        let glossa = dir.path().join(".glossa");
        let win_path = PathBuf::from(r"C:\models\rubert-nli");
        write_nli_config(
            &glossa,
            &win_path,
            "in_process",
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let cfg = VerifyConfig::resolve(&glossa);
        assert_eq!(cfg.model_dir, Some(win_path));
    }

    /// Omitting `entail_index`/`mode` writes neither key — `set` only wires what was given.
    #[test]
    fn write_nli_config_omits_unset_optional_fields() {
        let dir = tempfile::tempdir().unwrap();
        let glossa = dir.path().join(".glossa");
        write_nli_config(
            &glossa,
            Path::new("/models/x"),
            "in_process",
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let o = std::fs::read_to_string(glossa.join("ontology.toml")).unwrap();
        assert!(o.contains("scorer = \"in_process\""));
        assert!(!o.contains("entail_index"));
        assert!(!o.contains("ep_device"));
        assert!(!o.contains("mode ="));
        assert!(!o.contains("execution_providers"));
    }

    /// `write_nli_config` must preserve unrelated existing content in `ontology.toml`, not clobber
    /// it — same rationale as `calibrate::write_threshold_preserves_other_tables`.
    #[test]
    fn write_nli_config_preserves_other_tables() {
        let dir = tempfile::tempdir().unwrap();
        let glossa = dir.path().join(".glossa");
        std::fs::create_dir_all(&glossa).unwrap();
        std::fs::write(
            glossa.join("ontology.toml"),
            "# a hand-authored comment\n[types.symptom]\nlabel = \"Symptom\"\n\
             [verify.nli.threshold]\nsingle = 0.5\nmulti = 0.6\n",
        )
        .unwrap();
        write_nli_config(
            &glossa,
            Path::new("/models/rubert-nli"),
            "in_process",
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let o = std::fs::read_to_string(glossa.join("ontology.toml")).unwrap();
        assert!(o.contains("a hand-authored comment"));
        assert!(o.contains("[types.symptom]"));
        assert!(o.contains("label = \"Symptom\""));
        // Existing calibrated NLI thresholds must survive a `set` write untouched.
        assert!(o.contains("[verify.nli.threshold]"));
        assert!(o.contains("single = 0.5"));
        assert!(o.contains("multi = 0.6"));
        // And the new keys landed alongside them.
        assert!(o.contains("scorer = \"in_process\""));
        assert!(o.contains("rubert-nli"));
    }

    /// `write_nli_config`'s `execution_providers` param writes a TOML array under `[verify.nli]`
    /// and round-trips through `VerifyConfig::resolve` (the runtime gate's own read path) — extends
    /// `write_nli_config_roundtrips_into_ontology` for the new Plan 3 Task 1 key.
    #[test]
    fn write_nli_config_execution_providers_roundtrips_into_ontology() {
        let dir = tempfile::tempdir().unwrap();
        let glossa = dir.path().join(".glossa");
        write_nli_config(
            &glossa,
            Path::new("/models/rubert-nli"),
            "in_process",
            None,
            None,
            Some(&["cuda".to_string(), "cpu".to_string()]),
            None,
            None,
        )
        .unwrap();
        let o = std::fs::read_to_string(glossa.join("ontology.toml")).unwrap();
        assert!(o.contains("execution_providers"));
        assert!(o.contains("cuda"));

        let cfg = VerifyConfig::resolve(&glossa);
        assert_eq!(
            cfg.execution_providers,
            vec!["cuda".to_string(), "cpu".to_string()]
        );
    }

    /// `write_nli_config`'s `ep_device` param writes `[verify.nli].ep_device` and round-trips
    /// through `VerifyConfig::resolve` (the runtime gate's read path); `None` writes no key.
    #[test]
    fn write_nli_config_ep_device_roundtrips_into_ontology() {
        let dir = tempfile::tempdir().unwrap();
        let glossa = dir.path().join(".glossa");
        write_nli_config(
            &glossa,
            Path::new("/models/rubert-nli"),
            "in_process",
            None,
            None,
            None,
            Some(1),
            None,
        )
        .unwrap();
        let o = std::fs::read_to_string(glossa.join("ontology.toml")).unwrap();
        assert!(o.contains("ep_device = 1"));

        let cfg = VerifyConfig::resolve(&glossa);
        assert_eq!(cfg.execution_provider_device, Some(1));
    }

    /// `write_nli_config`'s `ep_mem_limit_mb` param writes `[verify.nli].ep_mem_limit_mb` and
    /// round-trips through `VerifyConfig::resolve`; `None` writes no key.
    #[test]
    fn write_nli_config_ep_mem_limit_mb_roundtrips_into_ontology() {
        let dir = tempfile::tempdir().unwrap();
        let glossa = dir.path().join(".glossa");
        write_nli_config(
            &glossa,
            Path::new("/models/rubert-nli"),
            "in_process",
            None,
            None,
            None,
            None,
            Some(512),
        )
        .unwrap();
        let o = std::fs::read_to_string(glossa.join("ontology.toml")).unwrap();
        assert!(o.contains("ep_mem_limit_mb = 512"));

        let cfg = VerifyConfig::resolve(&glossa);
        assert_eq!(cfg.execution_provider_mem_limit_mb, Some(512));
    }
}
