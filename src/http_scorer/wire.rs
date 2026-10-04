//! TEI/Jina scorer wire contract — the ONE source of truth, shared by the HTTP client
//! (`glossa`, `http-scorer` feature) and the `kbi` (kb-eval crate). Pure serde +
//! pure functions; no network, no cargo feature.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub struct RerankRequest {
    pub query: String,
    pub texts: Vec<String>,
    pub raw_scores: bool,
}

/// Jina/Cohere-family rerank request (`{query, documents}`), spoken by vLLM, llama.cpp, Cohere,
/// and our own `kbi`. `model` is serialized only when set — vLLM requires the served-model name,
/// llama.cpp and kbi ignore it. `top_n` is intentionally omitted so every input is scored.
#[derive(Serialize)]
pub struct JinaRerankRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub query: String,
    pub documents: Vec<String>,
    /// vLLM only: `false` asks for the raw logit instead of its default sigmoid, so the probability
    /// on our hits is computed by our own `sigmoid` like every other logit source. Absent for the
    /// other Jina-wire backends -- the hosted APIs reject unknown fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_activation: Option<bool>,
}

/// Which reranker wire shape a backend speaks. Operators name their SERVER (`backend = "vllm"`);
/// this maps that recognizable name to one of the two on-the-wire request/response shapes, so an
/// operator never has to know that vLLM and llama.cpp share the Jina protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RerankWire {
    /// TEI: request `{query, texts, raw_scores}` → bare `[{index, score}]`.
    Tei,
    /// Jina/Cohere: request `{query, documents}` → `{results:[{index, relevance_score}]}`.
    Jina,
}

impl RerankWire {
    /// Map an operator-facing backend name to its wire shape. Known names only — an unknown value
    /// is an error, so a typo (`"vlm"`) fails loudly instead of silently picking the wrong shape.
    pub fn from_backend(backend: &str) -> Result<RerankWire> {
        match backend.trim().to_ascii_lowercase().as_str() {
            // Our own kbi mirrors the request shape; its tested default path is `texts`+bare, so
            // `kbi` maps to TEI to keep the default a no-op against the historical client.
            "tei" | "kbi" => Ok(RerankWire::Tei),
            "vllm" | "llamacpp" | "jina" | "cohere" => Ok(RerankWire::Jina),
            other => bail!(
                "unknown rerank backend {other:?}; expected one of: \
                 tei, vllm, llamacpp, kbi, jina, cohere"
            ),
        }
    }
}

#[derive(Serialize)]
pub struct PredictRequest {
    pub inputs: Vec<Vec<String>>,
}

#[derive(Deserialize)]
struct Rank {
    index: usize,
    /// TEI names it `score`; the Jina family names it `relevance_score`. One field, both keys.
    #[serde(alias = "relevance_score")]
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

/// Serialize a rerank request in the shape the chosen backend speaks. `model` is used only by the
/// Jina family (and only when set); TEI ignores it and always sends `raw_scores: true`.
pub fn build_rerank_body(
    wire: RerankWire,
    query: &str,
    passages: &[&str],
    model: Option<&str>,
    ask_vllm_for_logits: bool,
) -> Result<String> {
    let body = match wire {
        RerankWire::Tei => serde_json::to_string(&build_rerank_request(query, passages, true)),
        RerankWire::Jina => serde_json::to_string(&JinaRerankRequest {
            model: model.map(str::to_string),
            query: query.to_string(),
            documents: passages.iter().map(|s| s.to_string()).collect(),
            use_activation: ask_vllm_for_logits.then_some(false),
        }),
    };
    body.context("serializing rerank request")
}

/// Whether a backend's rerank scores are raw logits. Each row is something we did, or something we
/// read, never something we assume: `tei`/`kbi` because our request sets `raw_scores: true`;
/// `vllm` because our request sets `use_activation: false`; `llamacpp` because its server writes
/// `res->score = embd[0]` straight from a classification head that has no sigmoid
/// (`tools/server/server-context.cpp`, `src/llama-graph.cpp`, read at 0be8468). `cohere` and `jina`
/// return a score they normalized and offer no raw option.
pub fn backend_emits_logits(backend: &str) -> bool {
    matches!(backend, "tei" | "kbi" | "vllm" | "llamacpp")
}

pub fn build_predict_request(premise: &str, hypotheses: &[&str]) -> PredictRequest {
    PredictRequest {
        inputs: hypotheses
            .iter()
            .map(|h| vec![premise.to_string(), h.to_string()])
            .collect(),
    }
}

/// Remap sorted-by-score `(index, score)` ranks back into INPUT order, validating full coverage
/// and finiteness. Shared by the TEI-only and tolerant parsers.
fn ranks_to_input_order(ranks: Vec<Rank>, n: usize) -> Result<Vec<f32>> {
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

/// TEI bare-array rerank response `[{index,score}]` (sorted by score) → scores in INPUT order.
/// Strict about the shape; used by the `kbi` server's round-trip test.
pub fn parse_rerank_bare(body: &str, n: usize) -> Result<Vec<f32>> {
    let ranks: Vec<Rank> = serde_json::from_str(body).context("parsing rerank response")?;
    ranks_to_input_order(ranks, n)
}

/// Tolerant rerank parser for the multi-backend client: accepts BOTH the TEI bare array
/// `[{index, score}]` and the Jina/Cohere envelope `{results:[{index, relevance_score}]}`, with
/// either score key (`score` or `relevance_score`). Remaps to INPUT order with the same validation.
pub fn parse_rerank(body: &str, n: usize) -> Result<Vec<f32>> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum RerankResponse {
        Wrapped { results: Vec<Rank> },
        Bare(Vec<Rank>),
    }
    let ranks =
        match serde_json::from_str::<RerankResponse>(body).context("parsing rerank response")? {
            RerankResponse::Wrapped { results } => results,
            RerankResponse::Bare(v) => v,
        };
    ranks_to_input_order(ranks, n)
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
            bail!(
                "predict score {} for label {:?} out of [0,1]",
                p.score,
                p.label
            );
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

    #[test]
    fn backend_names_map_to_wire_shapes() {
        // tei + our own kbi speak the TEI shape (kbi's tested default path); the rest are Jina.
        for tei in ["tei", "TEI", "kbi"] {
            assert_eq!(RerankWire::from_backend(tei).unwrap(), RerankWire::Tei);
        }
        for jina in ["vllm", "llamacpp", "jina", "cohere"] {
            assert_eq!(RerankWire::from_backend(jina).unwrap(), RerankWire::Jina);
        }
        assert!(RerankWire::from_backend("vlm").is_err());
    }

    #[test]
    fn tei_body_sends_texts_jina_body_sends_documents() {
        let tei = build_rerank_body(RerankWire::Tei, "q", &["a", "b"], None, false).unwrap();
        let v: serde_json::Value = serde_json::from_str(&tei).unwrap();
        assert_eq!(v["texts"], serde_json::json!(["a", "b"]));
        assert_eq!(v["raw_scores"], serde_json::json!(true));
        assert!(v.get("documents").is_none());

        let jina = build_rerank_body(RerankWire::Jina, "q", &["a", "b"], None, false).unwrap();
        let v: serde_json::Value = serde_json::from_str(&jina).unwrap();
        assert_eq!(v["documents"], serde_json::json!(["a", "b"]));
        assert!(v.get("texts").is_none());
        // model omitted when unset, so llama.cpp/kbi are not sent a bogus name.
        assert!(v.get("model").is_none());
    }

    #[test]
    fn jina_body_includes_model_when_set() {
        let jina =
            build_rerank_body(RerankWire::Jina, "q", &["a"], Some("bge-reranker"), false).unwrap();
        let v: serde_json::Value = serde_json::from_str(&jina).unwrap();
        assert_eq!(v["model"], serde_json::json!("bge-reranker"));
    }

    #[test]
    fn parse_rerank_accepts_jina_envelope_with_relevance_score() {
        let body = r#"{"results":[{"index":2,"relevance_score":9.0},
                                   {"index":0,"relevance_score":1.0},
                                   {"index":1,"relevance_score":-3.0}]}"#;
        let got = parse_rerank(body, 3).unwrap();
        assert_eq!(got, vec![1.0, -3.0, 9.0]);
    }

    #[test]
    fn parse_rerank_also_accepts_tei_bare_array() {
        let body = r#"[{"index":1,"score":5.0},{"index":0,"score":2.0}]"#;
        let got = parse_rerank(body, 2).unwrap();
        assert_eq!(got, vec![2.0, 5.0]);
    }

    #[test]
    fn parse_rerank_wrong_length_is_err() {
        let body = r#"{"results":[{"index":0,"relevance_score":1.0}]}"#;
        assert!(parse_rerank(body, 2).is_err());
    }

    /// vLLM applies sigmoid by default; we ask for the raw logit so our own sigmoid is the one
    /// scale. The flag is sent ONLY to vLLM: the hosted Jina/Cohere APIs reject unknown fields.
    #[test]
    fn vllm_request_asks_for_raw_logits_and_other_jina_backends_do_not() {
        let vllm = build_rerank_body(RerankWire::Jina, "q", &["a", "b"], Some("m"), true).unwrap();
        let v: serde_json::Value = serde_json::from_str(&vllm).unwrap();
        assert_eq!(v["use_activation"], serde_json::json!(false));

        let other = build_rerank_body(RerankWire::Jina, "q", &["a", "b"], None, false).unwrap();
        let o: serde_json::Value = serde_json::from_str(&other).unwrap();
        assert!(o.get("use_activation").is_none(), "{other}");

        let tei = build_rerank_body(RerankWire::Tei, "q", &["a"], None, false).unwrap();
        let t: serde_json::Value = serde_json::from_str(&tei).unwrap();
        assert_eq!(
            t["raw_scores"],
            serde_json::json!(true),
            "TEI path unchanged"
        );
    }

    /// Which backends hand us a raw logit -- by our own request (tei, kbi, vllm) or by their source
    /// (llama.cpp: `res->score = embd[0]`, head without sigmoid). Cohere and Jina return a
    /// normalized score we did not compute.
    #[test]
    fn logit_backends_are_the_ones_we_control_or_read() {
        for b in ["tei", "kbi", "vllm", "llamacpp"] {
            assert!(backend_emits_logits(b), "{b}");
        }
        for b in ["cohere", "jina"] {
            assert!(!backend_emits_logits(b), "{b}");
        }
    }
}
