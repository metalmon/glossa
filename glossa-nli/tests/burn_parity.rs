//! Burn-engine parity: the integrated `InProcessBurnNli` raw forward must match the ORT reference
//! logits within threshold. The reference fixtures (`inputs.json` + `reference_logits.json`, tiny,
//! English-synthetic) are committed under `tests/fixtures/`; the ~700 MB model weights are NOT, so
//! the test is gated on a model dir supplied via env and skips (no-op) when it's absent:
//!
//! - `GLOSSA_NLI_BURN_TEST_MODEL` — dir with `model.safetensors` + `tokenizer.json` (REQUIRED to run)
//! - `GLOSSA_NLI_BURN_TEST_FIXTURES` — optional override for the fixtures dir (defaults to the
//!   committed `tests/fixtures/`)
//!
//! Compiles only under the burn engine.
#![cfg(feature = "nli-burn-wgpu")]

use std::path::{Path, PathBuf};

use glossa_nli::harness::RawForward;
use glossa_nli::InProcessBurnNli;

#[test]
fn burn_forward_matches_ort_reference_logits() {
    let Ok(model_dir) = std::env::var("GLOSSA_NLI_BURN_TEST_MODEL") else {
        return; // no model weights → skip (CI has no artifact)
    };
    let fixtures = std::env::var("GLOSSA_NLI_BURN_TEST_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures"));
    const ENTAIL_IDX: usize = 0;

    let nli = InProcessBurnNli::load(Path::new(&model_dir), ENTAIL_IDX)
        .expect("burn engine load should succeed against a real test model dir");

    let inputs: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(fixtures.join("inputs.json")).expect("open inputs.json"),
    )
    .expect("parse inputs.json");
    let refs: Vec<[f32; 3]> = serde_json::from_reader(
        std::fs::File::open(fixtures.join("reference_logits.json"))
            .expect("open reference_logits.json"),
    )
    .expect("parse reference_logits.json");

    let col = |item: &serde_json::Value, k: &str| -> Vec<i64> {
        item[k].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect()
    };

    let mut maxdiff = 0f32;
    for (i, item) in inputs.as_array().unwrap().iter().enumerate() {
        let ids = col(item, "ids");
        let mask = col(item, "mask");
        let types = col(item, "types");
        let seq = ids.len();
        let logits = nli
            .forward_logits(&ids, &mask, &types, 1, seq)
            .expect("burn forward should succeed");
        assert_eq!(logits.len(), 3, "expected 3-way logits");
        for k in 0..3 {
            maxdiff = maxdiff.max((logits[k] - refs[i][k]).abs());
        }
    }
    assert!(
        maxdiff < 1e-4,
        "burn forward diverged from ORT reference: max abs logit diff = {maxdiff:.3e} (want <1e-4)"
    );
    eprintln!("burn vs ORT reference: max abs logit diff = {maxdiff:.3e}");
}
