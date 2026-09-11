//! `kbx nli check`: a readiness doctor for the NLI answer-grounding verifier. `[verify.nli]`'s
//! design fails open (any missing piece silently falls back to AC-only, see
//! `glossa::gate::resolve_scorer`'s doc comment), which is exactly the problem — a deployment can
//! believe NLI is protecting it while it silently isn't. This module gathers the observable facts
//! and turns them into a single ordered verdict so that silence becomes a diagnosis.
//!
//! Split like `download.rs`: [`nli_verdict`] is a PURE function over [`NliFacts`] (unit-tested, no
//! IO); [`nli_check`] is the thin wrapper that gathers the facts (config resolution, filesystem
//! probes, an optional model load + sanity entailment) and prints the report.

use std::path::PathBuf;

use anyhow::Result;

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
    /// `Some(true)` when a loaded scorer's canned English sanity probe passed
    /// (`entail(support) > entail(contra)`), `Some(false)` when it ran and failed (or errored),
    /// `None` when no scorer loaded so the probe never ran.
    pub sanity_ok: Option<bool>,
}

/// Judge [`NliFacts`] into a `(ready, verdict_line)` pair. READY only when every gate passes, in
/// this order (the returned line names the FIRST one that fails): mode isn't `ac` -> feature `nli`
/// is built -> scorer is `"in_process"` -> `model_dir` is configured -> `model_dir` has both model
/// files -> `ORT_DYLIB_PATH` is set and exists -> a scorer actually loaded -> the sanity probe
/// passed. Pure — no IO, no panics; safe to call with any combination of facts (including ones that
/// couldn't co-occur in practice, e.g. `sanity_ok: Some(true)` with `scorer_built: false`).
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
            "not ready: model_dir missing model.onnx and/or tokenizer.json (run `kbx nli \
             download`)"
                .to_string(),
        );
    }
    if !f.dylib_set {
        return (
            false,
            "not ready: ORT_DYLIB_PATH not set (or the path it names doesn't exist)".to_string(),
        );
    }
    if !f.scorer_built {
        return (
            false,
            "not ready: scorer failed to load (see the load-failure log line above)".to_string(),
        );
    }
    match f.sanity_ok {
        Some(true) => (true, "READY".to_string()),
        Some(false) => (
            false,
            "not ready: sanity failed (entail(support) did not exceed entail(contra))".to_string(),
        ),
        None => (
            false,
            "not ready: sanity probe did not run (no scorer loaded)".to_string(),
        ),
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
        .is_some_and(|d| d.join("model.onnx").is_file() && d.join("tokenizer.json").is_file());

    let dylib_path = std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from);
    let dylib_set = dylib_path.as_ref().is_some_and(|p| p.exists());

    let scorer = glossa::gate::resolve_scorer(&cfg);
    let scorer_built = scorer.is_some();
    let sanity_ok = scorer.as_ref().map(|s| {
        match s.entail(
            "A dog is sleeping on the couch.",
            &["An animal is resting.", "The room is empty."],
        ) {
            Ok(scores) if scores.len() >= 2 => scores[0] > scores[1],
            _ => false,
        }
    });

    let facts = NliFacts {
        mode,
        feature_built: cfg!(feature = "nli"),
        scorer: cfg.scorer.clone(),
        model_dir: cfg.model_dir.clone(),
        model_dir_exists,
        dylib_set,
        scorer_built,
        sanity_ok,
    };

    println!("mode           = {}", facts.mode);
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
                "model.onnx, tokenizer.json"
            } else {
                "NO — missing model.onnx and/or tokenizer.json"
            }
        ),
        None => println!("model_dir      = (unset)"),
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
    match facts.sanity_ok {
        Some(true) => println!("sanity         = pass"),
        Some(false) => println!("sanity         = FAIL"),
        None => println!("sanity         = (skipped — no scorer loaded)"),
    }

    let (_, verdict) = nli_verdict(&facts);
    println!("=> {verdict}");
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
            sanity_ok: Some(true),
        }
    }

    #[test]
    fn all_good_and_sanity_true_is_ready() {
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
    fn dylib_unset_blocks() {
        let mut f = ready_facts();
        f.dylib_set = false;
        let (ready, line) = nli_verdict(&f);
        assert!(!ready);
        assert!(line.contains("ORT_DYLIB_PATH"), "line was: {line}");
    }

    #[test]
    fn scorer_failed_to_load_blocks() {
        let mut f = ready_facts();
        f.scorer_built = false;
        f.sanity_ok = None;
        let (ready, line) = nli_verdict(&f);
        assert!(!ready);
        assert!(line.contains("scorer failed to load"), "line was: {line}");
    }

    #[test]
    fn sanity_false_blocks() {
        let mut f = ready_facts();
        f.sanity_ok = Some(false);
        let (ready, line) = nli_verdict(&f);
        assert!(!ready);
        assert!(line.contains("sanity failed"), "line was: {line}");
    }

    #[test]
    fn sanity_none_blocks_when_scorer_somehow_not_built() {
        // Synthetic combination (wouldn't occur via nli_check's own wiring, since sanity is only
        // probed when a scorer loaded) — the pure function still handles it deterministically.
        let mut f = ready_facts();
        f.sanity_ok = None;
        let (ready, line) = nli_verdict(&f);
        assert!(!ready);
        assert!(
            line.contains("sanity probe did not run"),
            "line was: {line}"
        );
    }
}
