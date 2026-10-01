//! HTTP handlers + router. The pure response encoders and the health code are always compiled and
//! unit-tested (they agree with the client's `glossa::http_scorer::wire` decode); the axum handlers
//! + router wire them to the session pools and need an ORT engine, so they are feature-gated.

/// 200 once models are loaded, 503 while still loading (k8s-style readiness).
pub fn health_code(ready: bool) -> u16 {
    if ready {
        200
    } else {
        503
    }
}

/// (index, score) pairs sorted by score descending, truncated to `top_n`.
fn sort_trunc(scores: &[f32], top_n: Option<usize>) -> Vec<(usize, f32)> {
    let mut v: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if let Some(n) = top_n {
        v.truncate(n);
    }
    v
}

/// TEI bare-array rerank response: `[{index, score}]`, sorted desc, truncated. Decoded by the
/// client's `wire::parse_rerank_bare`.
pub fn encode_rerank_bare(scores: &[f32], top_n: Option<usize>) -> String {
    let arr: Vec<serde_json::Value> = sort_trunc(scores, top_n)
        .into_iter()
        .map(|(i, s)| serde_json::json!({ "index": i, "score": s }))
        .collect();
    serde_json::to_string(&arr).unwrap_or_else(|_| "[]".to_string())
}

/// Jina/llama.cpp rerank response: `{model, object, usage, results:[{index, relevance_score}]}`.
pub fn encode_rerank_jina(scores: &[f32], top_n: Option<usize>, model: &str) -> String {
    let results: Vec<serde_json::Value> = sort_trunc(scores, top_n)
        .into_iter()
        .map(|(i, s)| serde_json::json!({ "index": i, "relevance_score": s }))
        .collect();
    serde_json::json!({
        "model": model,
        "object": "list",
        "usage": { "prompt_tokens": 0, "total_tokens": 0 },
        "results": results,
    })
    .to_string()
}

/// TEI `/predict` response for pre-resolved entailment scores: one row per hypothesis, each a
/// single-class `[{"label":"entailment","score":P}]`. The glossa server already applied the model's
/// entail_index server-side, so entailment sits at index 0 — the http client reads index 0 (see
/// `resolve_scorer`'s http arm). One hypothesis emits the union's single form (`Vec<Prediction>`),
/// many emit the batch form (`Vec<Vec<Prediction>>`); both decode via the client's `parse_predict`.
pub fn encode_predict(scores: &[f32]) -> String {
    let rows: Vec<serde_json::Value> = scores
        .iter()
        .map(|s| serde_json::json!([{ "label": "entailment", "score": s }]))
        .collect();
    if rows.len() == 1 {
        serde_json::to_string(&rows[0]).unwrap_or_else(|_| "[]".to_string())
    } else {
        serde_json::to_string(&rows).unwrap_or_else(|_| "[]".to_string())
    }
}

/// Passage strings from a `/rerank` `texts`/`documents` array — accepts bare strings AND Jina
/// `{ "text": "..." }` objects (external Jina clients send objects); entries with neither are
/// skipped.
pub fn extract_passages(arr: &[serde_json::Value]) -> Vec<String> {
    arr.iter()
        .filter_map(|v| {
            v.as_str()
                .map(String::from)
                .or_else(|| v.get("text").and_then(|t| t.as_str()).map(String::from))
        })
        .collect()
}

#[cfg(any(
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
))]
pub use server::router;

#[cfg(any(
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
))]
mod server {
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    use axum::extract::State;
    use axum::http::StatusCode;
    use axum::response::{IntoResponse, Response};
    use axum::routing::{get, post};
    use axum::{Json, Router};

    use super::{encode_rerank_bare, encode_rerank_jina, health_code};
    use crate::infer::state::ServerState;

    type Shared = Arc<ServerState>;

    pub fn router(state: Shared) -> Router {
        Router::new()
            .route("/health", get(health))
            .route("/ready", get(health))
            .route("/info", get(info))
            .route("/rerank", post(rerank))
            .route("/reranking", post(rerank))
            .route("/v1/rerank", post(rerank))
            .route("/predict", post(predict))
            .route("/metrics", get(metrics))
            .with_state(state)
    }

    async fn health(State(st): State<Shared>) -> Response {
        let code = StatusCode::from_u16(health_code(st.ready.load(Ordering::Relaxed))).unwrap();
        if code == StatusCode::OK {
            (code, Json(serde_json::json!({ "status": "ok" }))).into_response()
        } else {
            (code, Json(serde_json::json!({ "status": "loading" }))).into_response()
        }
    }

    async fn info(State(st): State<Shared>) -> Json<serde_json::Value> {
        let nli_loaded = st.nli.lock().unwrap_or_else(|e| e.into_inner()).is_some();
        let rerank_loaded = st
            .rerank
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some();
        let nli_ep = st.nli_ep.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let rerank_ep = st
            .rerank_ep
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        Json(serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "ready": st.ready.load(Ordering::Relaxed),
            "nli": st.nli_variant.as_ref().map(|v| serde_json::json!({
                "variant": v, "ep_active": nli_ep, "loaded": nli_loaded,
            })),
            "rerank": st.rerank_variant.as_ref().map(|v| serde_json::json!({
                "variant": v, "ep_active": rerank_ep, "loaded": rerank_loaded,
            })),
        }))
    }

    async fn metrics() -> Response {
        // Minimal Prometheus text; richer scorer series are a Phase-2 addition.
        (StatusCode::OK, "# glossa kbi\nglossa_infer_up 1\n").into_response()
    }

    fn overloaded() -> Response {
        (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({ "error": "overloaded" })),
        )
            .into_response()
    }

    /// RAII in-flight counter for the concurrency cap; holds the shared state to decrement on drop.
    struct InFlight(Shared);
    impl Drop for InFlight {
        fn drop(&mut self) {
            self.0.in_flight.fetch_sub(1, Ordering::SeqCst);
        }
    }
    fn admit(st: &Shared) -> Option<InFlight> {
        let n = st.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        let guard = InFlight(st.clone());
        match st.max_concurrency {
            Some(cap) if n > cap => None, // guard drops here → decrement
            _ => Some(guard),
        }
    }

    /// 503 while models are still loading, else 404 (this server serves no such model).
    fn not_available(st: &ServerState, what: &str) -> Response {
        if !st.ready.load(Ordering::Relaxed) {
            (StatusCode::SERVICE_UNAVAILABLE, "loading").into_response()
        } else {
            (StatusCode::NOT_FOUND, format!("{what} not configured")).into_response()
        }
    }

    async fn rerank(State(st): State<Shared>, Json(body): Json<serde_json::Value>) -> Response {
        let Some(_g) = admit(&st) else {
            return overloaded();
        };
        let scorer = match st.rerank.lock().unwrap_or_else(|e| e.into_inner()).clone() {
            Some(s) => s,
            None => return not_available(&st, "reranker"),
        };
        let query = body
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let (jina, passages) = match body.get("texts").and_then(|v| v.as_array()) {
            Some(a) => (false, super::extract_passages(a)),
            None => match body.get("documents").and_then(|v| v.as_array()) {
                Some(a) => (true, super::extract_passages(a)),
                None => {
                    return (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "missing `texts` or `documents`",
                    )
                        .into_response()
                }
            },
        };
        let top_n = body
            .get("top_n")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);
        // A `--fit` run chose a budget for this process and kept it in memory; without one the
        // session's own configured budget applies, so the unfitted path is byte-identical.
        let fitted = *st.rerank_fitted.lock().unwrap_or_else(|e| e.into_inner());
        let scored = tokio::task::spawn_blocking(move || {
            let refs: Vec<&str> = passages.iter().map(String::as_str).collect();
            match fitted {
                Some(b) => scorer.rerank_with_budget(&query, &refs, b),
                None => scorer.rerank(&query, &refs),
            }
        })
        .await;
        match scored {
            Ok(Ok(scores)) => {
                let body = if jina {
                    encode_rerank_jina(&scores, top_n, "glossa-reranker")
                } else {
                    encode_rerank_bare(&scores, top_n)
                };
                ([("content-type", "application/json")], body).into_response()
            }
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "rerank failed").into_response(),
        }
    }

    async fn predict(State(st): State<Shared>, Json(body): Json<serde_json::Value>) -> Response {
        let Some(_g) = admit(&st) else {
            return overloaded();
        };
        let scorer = match st.nli.lock().unwrap_or_else(|e| e.into_inner()).clone() {
            Some(s) => s,
            None => return not_available(&st, "nli"),
        };
        // `inputs` is a pair ["p","h"] or a batch [["p","h"], ...]; normalize to a batch of pairs.
        let inputs = body.get("inputs");
        let pairs: Vec<(String, String)> = match inputs.and_then(|v| v.as_array()) {
            Some(a) if a.iter().all(|e| e.is_string()) => {
                // single pair ["p","h"]
                let s: Vec<String> = a
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                if s.len() == 2 {
                    vec![(s[0].clone(), s[1].clone())]
                } else {
                    return (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "pair must be [premise, hypothesis]",
                    )
                        .into_response();
                }
            }
            Some(a) => {
                let mut out = Vec::new();
                for e in a {
                    let p: Vec<String> = e
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect();
                    if p.len() != 2 {
                        return (
                            StatusCode::UNPROCESSABLE_ENTITY,
                            "each pair must be [premise, hypothesis]",
                        )
                            .into_response();
                    }
                    out.push((p[0].clone(), p[1].clone()));
                }
                out
            }
            None => return (StatusCode::UNPROCESSABLE_ENTITY, "missing `inputs`").into_response(),
        };
        // Group by premise so one entail() call scores all its hypotheses (matches the client,
        // which sends one premise with many hypotheses).
        let premise = pairs.first().map(|(p, _)| p.clone()).unwrap_or_default();
        let same_premise = pairs.iter().all(|(p, _)| *p == premise);
        let fitted = *st.nli_fitted.lock().unwrap_or_else(|e| e.into_inner());
        let scored = tokio::task::spawn_blocking(move || {
            if same_premise {
                let hyps: Vec<&str> = pairs.iter().map(|(_, h)| h.as_str()).collect();
                match fitted {
                    Some(b) => scorer.entail_with_budget(&premise, &hyps, b),
                    None => scorer.entail(&premise, &hyps),
                }
            } else {
                // Mixed premises: score each pair individually, concatenate.
                let mut all = Vec::new();
                for (p, h) in &pairs {
                    let one = match fitted {
                        Some(b) => scorer.entail_with_budget(p, &[h.as_str()], b),
                        None => scorer.entail(p, &[h.as_str()]),
                    };
                    match one {
                        Ok(mut v) => all.append(&mut v),
                        Err(e) => return Err(e),
                    }
                }
                Ok(all)
            }
        })
        .await;
        match scored {
            Ok(Ok(scores)) => (
                [("content-type", "application/json")],
                super::encode_predict(&scores),
            )
                .into_response(),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "predict failed").into_response(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_is_503_until_ready_then_200() {
        assert_eq!(health_code(false), 503);
        assert_eq!(health_code(true), 200);
    }

    #[test]
    fn rerank_json_bare_is_sorted_desc_and_truncated_to_top_n() {
        let json = encode_rerank_bare(&[2.0, 9.0, -3.0], Some(2));
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v[0]["index"], 1);
        assert_eq!(v[1]["index"], 0);
        assert_eq!(v.as_array().unwrap().len(), 2);
    }

    #[test]
    fn rerank_jina_has_results_with_relevance_score() {
        let json = encode_rerank_jina(&[2.0, 9.0], None, "glossa-reranker");
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["results"][0]["index"], 1);
        assert!(v["results"][0]["relevance_score"].as_f64().is_some());
    }

    #[test]
    fn rerank_bare_roundtrips_through_client_wire_parse() {
        let json = encode_rerank_bare(&[2.0, 9.0, -3.0], None);
        let back = glossa::http_scorer::wire::parse_rerank_bare(&json, 3).unwrap();
        assert_eq!(back, vec![2.0, 9.0, -3.0]);
    }

    #[test]
    fn extract_passages_accepts_strings_and_jina_text_objects() {
        let v: serde_json::Value = serde_json::from_str(r#"["a", {"text":"b"}, "c"]"#).unwrap();
        assert_eq!(extract_passages(v.as_array().unwrap()), vec!["a", "b", "c"]);
    }

    #[test]
    fn predict_encode_roundtrips_through_client_wire_parse_at_index_0() {
        // The server pre-resolves entailment to index 0; the http client reads index 0. Pin it.
        let json = encode_predict(&[0.9, 0.2]);
        let back = glossa::http_scorer::wire::parse_predict(&json, 0, 2).unwrap();
        assert_eq!(back, vec![0.9, 0.2]);
        // single-hypothesis union form
        let one = encode_predict(&[0.7]);
        assert_eq!(
            glossa::http_scorer::wire::parse_predict(&one, 0, 1).unwrap(),
            vec![0.7]
        );
    }
}
