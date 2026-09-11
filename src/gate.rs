//! Answer-grounding gate (model-free). See docs/superpowers/specs/2026-09-06-answer-grounding-gate-design.md
pub mod config;
pub mod df;
pub mod df_cache;
pub mod nli;
#[cfg(feature = "nli")]
pub mod nli_engine;
pub mod score;
pub mod token;

pub use config::{VerifyConfig, VerifyMode};
pub use score::{decide, score, Bucket, Decision, GateOutcome, GateScore};

/// Resolve a `path#loc` citation to its chunk text, using the SAME extraction path the `read`
/// MCP tool uses (`crate::tools::read`, which `src/mcp.rs::read_common` calls). `glossa_dir` is the
/// corpus `.glossa` dir; the document index opens from its parent. The `#<n>` anchor is parsed by
/// `crate::tools::read` itself (via `parse_path_anchor`), so we hand it the whole `path#loc` token
/// as the path with a placeholder ordinal. Model-free, images off, tracing disabled — this is the
/// one resolver shared by the runtime gate (Task 6), the eval arm (Task 7), and calibration (Task 9),
/// so all three tokenize identical chunk text. When a path fails to resolve, `crate::tools::read`
/// returns its human-readable "not found / out of range" text rather than erroring; that text flows
/// through unchanged (mirroring exactly what a reader would have read).
pub fn read_chunk_text(glossa_dir: &std::path::Path, path_loc: &str) -> anyhow::Result<String> {
    let root = glossa_dir.parent().unwrap_or(glossa_dir);
    let idx = crate::index::store::DocIndex::open_or_create(root)?;
    let out = crate::tools::read(
        root,
        &idx,
        None,
        path_loc,
        1,
        false,
        &crate::trace::TraceLog::disabled(),
    );
    Ok(out.text)
}

/// Resolve `chunk_paths` (via [`read_chunk_text`]), score, and decide — the shared core between
/// the MCP `verify` handler (`src/mcp.rs`, Task 6) and the eval-side `verify` tool arm (Task 7).
/// The MCP handler further projects this outcome by serving profile (`project_verify`); the eval
/// side and calibration (Task 9) consume the outcome directly via [`verify_json`].
pub fn verify_outcome(
    glossa_dir: &std::path::Path,
    answer: &str,
    chunk_paths: &[String],
) -> anyhow::Result<(GateOutcome, usize)> {
    let cfg = VerifyConfig::resolve(glossa_dir);
    let scorer = resolve_scorer(&cfg);
    verify_outcome_with_scorer(glossa_dir, answer, chunk_paths, scorer.as_deref())
}

/// Build the runtime NLI scorer from `[verify.nli]`, or `None` (=> AC-only, fail-open — spec 4).
/// `scorer = "in_process"` + feature `nli` compiled in + `model_dir` set => a real `InProcessNli`;
/// any load error (bad path, corrupt export, missing ort runtime) is logged and downgraded to
/// `None` rather than propagated, so a broken model directory never takes the gate down — it just
/// falls back to AC-only. With the `nli` feature off this is the stub below, so `verify_outcome`'s
/// behaviour is byte-for-byte unchanged from before this scorer existed.
#[cfg(feature = "nli")]
pub fn resolve_scorer(cfg: &VerifyConfig) -> Option<Box<dyn nli::NliScorer>> {
    if cfg.scorer.as_deref() != Some("in_process") {
        return None;
    }
    let dir = cfg.model_dir.as_ref()?;
    match glossa_nli::InProcessNli::load(dir, cfg.entail_index) {
        Ok(s) => Some(Box::new(s)),
        Err(e) => {
            eprintln!("nli scorer load failed ({}): {e}", dir.display());
            None
        }
    }
}

/// `nli` feature off => no in-process scorer is even compiled; always `None` (AC-only).
#[cfg(not(feature = "nli"))]
pub fn resolve_scorer(_cfg: &VerifyConfig) -> Option<Box<dyn nli::NliScorer>> {
    None
}

/// Same as [`verify_outcome`], plus an optional NLI scorer. `None` ⇒ model-free (today's
/// behaviour). Plan 2 supplies a real `&dyn NliScorer`; the mode/short-circuit lives here (spec 4).
pub fn verify_outcome_with_scorer(
    glossa_dir: &std::path::Path,
    answer: &str,
    chunk_paths: &[String],
    scorer: Option<&dyn nli::NliScorer>,
) -> anyhow::Result<(GateOutcome, usize)> {
    let cfg = VerifyConfig::resolve(glossa_dir);
    let df = df_cache::cached_df(glossa_dir)?;
    let chunks: Vec<String> = chunk_paths
        .iter()
        .map(|p| read_chunk_text(glossa_dir, p))
        .collect::<anyhow::Result<_>>()?;
    let ac = score(answer, &chunks, &df, cfg.rare_df_frac);
    let answer_tokens = token::tokenize(answer).len();
    // Always compute NLI when the mode is ready for it — no AC-serve short-circuit. Under the
    // z-score consensus (spec §4 rev.5) a low-AC/high-NLI answer can still serve (z = zac + znli
    // > threshold), so skipping the scorer whenever AC alone doesn't serve would make that path
    // unreachable. Combined readiness is `is_combined_ready()` (calibrated combined stats), NOT
    // `is_nli_ready()` (nli thresholds) — the two modes are calibrated independently.
    let need_nli = scorer.is_some()
        && match cfg.mode {
            VerifyMode::Ac => false,
            VerifyMode::Nli => cfg.is_nli_ready(),
            VerifyMode::Combined => cfg.is_combined_ready(),
        };
    let nli_val = if need_nli {
        nli::nli_score(answer, &chunks, &df, &cfg, scorer.unwrap())
    } else {
        None
    };
    let outcome = score::decide_modes(ac, nli_val, &cfg, answer_tokens);
    Ok((outcome, chunk_paths.len()))
}

/// The full Editor/Full diagnostic JSON shape (8 fields: decision, score, bucket, threshold,
/// rare-token counts, the ungrounded tokens, chunk count). The SINGLE builder of this shape,
/// shared by [`verify_json`] (eval side + calibration) and the MCP handler's Editor/Full
/// projection (`src/mcp.rs::project_verify`), so the diagnostic shape can't drift between the two
/// surfaces. This carries the Editor-only internals (threshold/bucket/rare counts) and the full
/// `o.reason`-free diagnostic; the Reader profile gets a trimmed shape with a short static reason
/// (see `src/mcp.rs::project_verify`), not this full diagnostic.
pub fn full_diagnostic_json(o: &GateOutcome, n_chunks: usize) -> serde_json::Value {
    let decision = match o.decision {
        Decision::Serve => "serve",
        Decision::Abstain => "abstain",
    };
    serde_json::json!({
        "decision": decision,
        "score": o.score.grounding,
        "bucket": if matches!(o.score.bucket, Bucket::Single) { "single" } else { "multi" },
        "threshold": o.threshold,
        "rare_total": o.score.rare_total,
        "rare_ungrounded": o.score.rare_ungrounded,
        "ungrounded_tokens": o.score.ungrounded_tokens,
        "n_chunks": n_chunks,
    })
}

/// The Editor/Full diagnostic JSON for the eval-side `verify` executor and calibration (Task 9) —
/// neither has a serving-profile concept, so this always returns the full diagnostic (built by the
/// shared [`full_diagnostic_json`], mirroring `src/mcp.rs::project_verify`'s `Editor | Full` arm).
pub fn verify_json(
    glossa_dir: &std::path::Path,
    answer: &str,
    chunk_paths: &[String],
) -> anyhow::Result<serde_json::Value> {
    let (o, n_chunks) = verify_outcome(glossa_dir, answer, chunk_paths)?;
    Ok(full_diagnostic_json(&o, n_chunks))
}

/// The SINGLE builder of the Reader-profile verify JSON — shared by `src/mcp.rs::project_verify`'s
/// Reader arm and the eval executor (`eval/.../glossa_tools.rs`). Byte-identical to the shape the
/// live Reader client gets today (4 keys incl. `score`); do NOT add editor internals here.
pub fn reader_verify_json(o: &GateOutcome, _n_chunks: usize) -> serde_json::Value {
    let decision = match o.decision {
        Decision::Serve => "serve",
        Decision::Abstain => "abstain",
    };
    serde_json::json!({
        "decision": decision,
        "score": o.score.grounding,
        "reason_short": reader_reason_short(o),
        "ungrounded_tokens": o.score.ungrounded_tokens,
    })
}

/// A synthetic Reader response for "verify is not available here" (disabled/uncalibrated). The
/// answering prompt keys on `reason_short == "uncalibrated"` to treat verify as unavailable and fall
/// back to its own CHECK (NOT an abstain). Same 4-key shape as `reader_verify_json`.
pub fn reader_uncalibrated_json() -> serde_json::Value {
    serde_json::json!({
        "decision": "abstain",
        "score": 0.0,
        "reason_short": "uncalibrated",
        "ungrounded_tokens": [],
    })
}

/// A SHORT, static machine reason for the Reader profile (no ungrounded-token prose). Moved here
/// from `src/mcp.rs` so the eval side and the live server share one mapping.
pub(crate) fn reader_reason_short(o: &GateOutcome) -> &'static str {
    match o.decision {
        Decision::Serve => "grounded",
        Decision::Abstain => {
            if o.reason.starts_with("ungrounded specifics") {
                "ungrounded"
            } else if o.reason.starts_with("empty") {
                "empty answer"
            } else if o.reason == "uncalibrated" {
                "uncalibrated"
            } else if o.reason == "unentailed" {
                "unentailed"
            } else {
                "below threshold"
            }
        }
    }
}

#[cfg(test)]
mod resolve_scorer_tests {
    use super::*;

    fn base_cfg() -> VerifyConfig {
        VerifyConfig {
            enabled: true,
            rare_df_frac: 0.03,
            min_answer_tokens: 10,
            threshold_single: None,
            threshold_multi: None,
            mode: VerifyMode::Ac,
            nli_threshold_single: None,
            nli_threshold_multi: None,
            combined_single: None,
            combined_multi: None,
            scorer: None,
            model_dir: None,
            entail_index: 0,
        }
    }

    // Runs identically with the `nli` feature ON or OFF: with it off `resolve_scorer` is the
    // always-`None` stub; with it on, `cfg.scorer.is_none()` hits the function's own early return.
    // Either way this proves the fail-open default (no scorer configured => AC-only).
    #[test]
    fn none_scorer_yields_no_runtime_scorer() {
        let cfg = base_cfg();
        assert!(resolve_scorer(&cfg).is_none());
    }

    // Same reasoning as above, but for an unrecognised (or not-yet-built, e.g. "http") scorer
    // name: never a runtime scorer regardless of feature state.
    #[test]
    fn unrecognized_scorer_name_yields_no_runtime_scorer() {
        let mut cfg = base_cfg();
        cfg.scorer = Some("http".to_string());
        assert!(resolve_scorer(&cfg).is_none());
    }

    // With the `nli` feature ON, `scorer = "in_process"` but no `model_dir` must still fail open
    // rather than panic on the `?` — this is the "configured wrong" fail-open path (vs. simply
    // unconfigured above).
    #[cfg(feature = "nli")]
    #[test]
    fn in_process_without_model_dir_yields_no_runtime_scorer() {
        let mut cfg = base_cfg();
        cfg.scorer = Some("in_process".to_string());
        assert!(resolve_scorer(&cfg).is_none());
    }
}

#[cfg(test)]
mod reader_projection_tests {
    use super::*;
    use serde_json::json;

    // A minimal Serve outcome and an Abstain-uncalibrated outcome, built without touching decide().
    fn serve_outcome() -> GateOutcome {
        GateOutcome {
            decision: Decision::Serve,
            threshold: Some(0.8),
            reason: "grounded".into(),
            score: GateScore {
                // Exactly representable in both f32 and f64 — avoids a float-precision mismatch
                // between the f32 struct field and the f64 JSON literal in the assertion below
                // (0.9 is NOT exact in f32, so `json!(0.9)` would spuriously fail).
                grounding: 0.5,
                bucket: Bucket::Single,
                rare_total: 3,
                rare_ungrounded: 0,
                ungrounded_tokens: vec![],
            },
            nli: None,
        }
    }

    #[test]
    fn reader_reason_short_reports_unentailed() {
        let o = GateOutcome {
            decision: Decision::Abstain,
            threshold: Some(0.8),
            reason: "unentailed".into(),
            score: GateScore {
                grounding: 0.5,
                bucket: Bucket::Single,
                rare_total: 3,
                rare_ungrounded: 0,
                ungrounded_tokens: vec![],
            },
            nli: Some(0.1),
        };
        assert_eq!(reader_reason_short(&o), "unentailed");
    }

    #[test]
    fn reader_verify_json_has_exactly_four_keys_incl_score() {
        let v = reader_verify_json(&serve_outcome(), 1);
        let obj = v.as_object().unwrap();
        let mut keys: Vec<&str> = obj.keys().map(|s| s.as_str()).collect();
        keys.sort();
        assert_eq!(
            keys,
            ["decision", "reason_short", "score", "ungrounded_tokens"]
        );
        assert_eq!(v["decision"], "serve");
        assert_eq!(v["score"], json!(0.5));
        assert_eq!(v["reason_short"], "grounded");
        assert_eq!(v["ungrounded_tokens"], json!([]));
    }

    #[test]
    fn reader_uncalibrated_json_signals_uncalibrated() {
        let v = reader_uncalibrated_json();
        assert_eq!(v["reason_short"], "uncalibrated");
        // shape parity with reader_verify_json: same four keys
        let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(|s| s.as_str()).collect();
        keys.sort();
        assert_eq!(
            keys,
            ["decision", "reason_short", "score", "ungrounded_tokens"]
        );
    }
}

#[cfg(test)]
mod verify_json_tests {
    use super::*;

    fn write_calibrated_ontology(root: &std::path::Path) {
        let g = root.join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "[verify]\nenabled = true\n[verify.threshold]\nsingle = 0.8\nmulti = 0.9\n",
        )
        .unwrap();
    }

    /// `verify_json` needs a DF sidecar to score at all; without one it errors rather than
    /// panicking (mirrors the MCP handler's `df sidecar: {e}` failure mode).
    #[test]
    fn verify_json_errors_without_df_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        write_calibrated_ontology(dir.path());
        let glossa_dir = dir.path().join(".glossa");
        let out = verify_json(&glossa_dir, "some answer", &[]);
        assert!(out.is_err(), "no df sidecar written yet");
    }

    /// With a DF sidecar present and no chunk_paths, an answer with no rare tokens (per an empty
    /// DF table every token counts as rare — df==0) still yields a well-formed diagnostic JSON
    /// with the expected keys, uncalibrated-fail-closed decision aside.
    #[test]
    fn verify_json_shape_has_diagnostic_keys() {
        let dir = tempfile::tempdir().unwrap();
        write_calibrated_ontology(dir.path());
        let glossa_dir = dir.path().join(".glossa");
        df::DfTable::new()
            .save(&df::DfTable::sidecar_path(&glossa_dir))
            .unwrap();
        let out = verify_json(&glossa_dir, "", &[]).unwrap();
        for key in [
            "decision",
            "score",
            "bucket",
            "threshold",
            "rare_total",
            "rare_ungrounded",
            "ungrounded_tokens",
            "n_chunks",
        ] {
            assert!(out.get(key).is_some(), "missing key {key}: {out}");
        }
    }
}
