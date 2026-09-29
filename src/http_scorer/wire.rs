//! TEI/Jina scorer wire contract — the ONE source of truth, shared by the HTTP client
//! (`glossa`, `http-scorer` feature) and the `inference-server` (kb-eval crate). Pure serde +
//! pure functions; no network, no cargo feature.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub struct RerankRequest {
    pub query: String,
    pub texts: Vec<String>,
    pub raw_scores: bool,
}

#[derive(Serialize)]
pub struct PredictRequest {
    pub inputs: Vec<Vec<String>>,
}

#[derive(Deserialize)]
struct Rank {
    index: usize,
    score: Option<f32>,
}

#[derive(Deserialize)]
struct Prediction {
    label: String,
    score: f32,
}

pub fn build_rerank_request(query: &str, passages: &[&str], raw_scores: bool) -> RerankRequest {
    RerankRequest {
        query: query.to_string(),
        texts: passages.iter().map(|s| s.to_string()).collect(),
        raw_scores,
    }
}

pub fn build_predict_request(premise: &str, hypotheses: &[&str]) -> PredictRequest {
    PredictRequest {
        inputs: hypotheses
            .iter()
            .map(|h| vec![premise.to_string(), h.to_string()])
            .collect(),
    }
}

/// TEI bare-array rerank response `[{index,score}]` (sorted by score) → scores in INPUT order.
pub fn parse_rerank_bare(body: &str, n: usize) -> Result<Vec<f32>> {
    let ranks: Vec<Rank> = serde_json::from_str(body).context("parsing rerank response")?;
    if ranks.len() != n {
        bail!("rerank returned {} scores, expected {n}", ranks.len());
    }
    let mut out = vec![f32::NAN; n];
    for r in ranks {
        let s = r
            .score
            .filter(|v| v.is_finite())
            .with_context(|| format!("non-finite score at index {}", r.index))?;
        let slot = out
            .get_mut(r.index)
            .with_context(|| format!("index {} out of range 0..{n}", r.index))?;
        *slot = s;
    }
    if out.iter().any(|v| !v.is_finite()) {
        bail!("rerank response did not cover every input index");
    }
    Ok(out)
}

/// TEI `/predict` untagged union: single input ⇒ `Vec<Prediction>`, batch ⇒ `Vec<Vec<Prediction>>`.
/// Returns P(entail) at `entail_index` for each of the `n` hypotheses, in order.
pub fn parse_predict(body: &str, entail_index: usize, n: usize) -> Result<Vec<f32>> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum PredictResponse {
        Batch(Vec<Vec<Prediction>>),
        Single(Vec<Prediction>),
    }
    let rows: Vec<Vec<Prediction>> = match serde_json::from_str(body).context("parsing predict")? {
        PredictResponse::Batch(b) => b,
        PredictResponse::Single(s) => vec![s],
    };
    if rows.len() != n {
        bail!("predict returned {} rows, expected {n}", rows.len());
    }
    let mut out = Vec::with_capacity(n);
    for row in rows {
        let p = row
            .get(entail_index)
            .with_context(|| format!("entail_index {entail_index} >= {} classes", row.len()))?;
        if !p.score.is_finite() || !(0.0..=1.0).contains(&p.score) {
            bail!("predict score {} for label {:?} out of [0,1]", p.score, p.label);
        }
        out.push(p.score);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rerank_bare_remaps_index_to_input_order() {
        let body = r#"[{"index":2,"score":9.0},{"index":0,"score":1.0},{"index":1,"score":-3.0}]"#;
        let got = parse_rerank_bare(body, 3).unwrap();
        assert_eq!(got, vec![1.0, -3.0, 9.0]);
    }

    #[test]
    fn rerank_bare_wrong_length_is_err() {
        let body = r#"[{"index":0,"score":1.0}]"#;
        assert!(parse_rerank_bare(body, 2).is_err());
    }

    #[test]
    fn rerank_bare_non_finite_is_err() {
        let body = r#"[{"index":0,"score":null}]"#;
        assert!(parse_rerank_bare(body, 1).is_err());
    }

    #[test]
    fn predict_batch_union_takes_entail_index_in_order() {
        let body = r#"[[{"label":"entailment","score":0.9},{"label":"contradiction","score":0.1}],
                       [{"label":"entailment","score":0.2},{"label":"contradiction","score":0.8}]]"#;
        let got = parse_predict(body, 0, 2).unwrap();
        assert_eq!(got, vec![0.9, 0.2]);
    }

    #[test]
    fn predict_single_pair_union_is_accepted() {
        let body = r#"[{"label":"entailment","score":0.7},{"label":"contradiction","score":0.3}]"#;
        let got = parse_predict(body, 0, 1).unwrap();
        assert_eq!(got, vec![0.7]);
    }

    #[test]
    fn predict_entail_index_out_of_range_is_err() {
        let body = r#"[[{"label":"entailment","score":0.9}]]"#;
        assert!(parse_predict(body, 5, 1).is_err());
    }

    #[test]
    fn build_requests_shape() {
        let r = build_rerank_request("q", &["a", "b"], true);
        assert_eq!(r.texts, vec!["a", "b"]);
        assert!(r.raw_scores);
        let p = build_predict_request("prem", &["h1", "h2"]);
        assert_eq!(p.inputs, vec![vec!["prem", "h1"], vec!["prem", "h2"]]);
    }
}
