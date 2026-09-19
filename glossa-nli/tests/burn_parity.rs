//! Burn-engine parity: the integrated `InProcessBurnNli` raw forward must match the ORT reference
//! logits within threshold. Env-gated so CI stays green without the model artifact:
//!
//! - `GLOSSA_NLI_BURN_TEST_MODEL` — dir with `model.safetensors` + `tokenizer.json`
//! - `GLOSSA_NLI_BURN_TEST_FIXTURES` — dir with `inputs.json` + `reference_logits.json`
//!
//! Both are produced by the Plan 4 spike fixtures. Compiles only under the burn engine.
#![cfg(feature = "nli-burn-wgpu")]

use std::path::Path;

use glossa_nli::harness::RawForward;
use glossa_nli::InProcessBurnNli;

#[test]
fn burn_forward_matches_ort_reference_logits() {
    let (Ok(model_dir), Ok(fixtures)) = (
        std::env::var("GLOSSA_NLI_BURN_TEST_MODEL"),
        std::env::var("GLOSSA_NLI_BURN_TEST_FIXTURES"),
    ) else {
        return; // no artifact → skip (CI)
    };
    const ENTAIL_IDX: usize = 0;

    let nli = InProcessBurnNli::load(Path::new(&model_dir), ENTAIL_IDX)
        .expect("burn engine load should succeed against a real test model dir");

    let inputs: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(Path::new(&fixtures).join("inputs.json")).expect("open inputs.json"),
    )
    .expect("parse inputs.json");
    let refs: Vec<[f32; 3]> = serde_json::from_reader(
        std::fs::File::open(Path::new(&fixtures).join("reference_logits.json"))
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
