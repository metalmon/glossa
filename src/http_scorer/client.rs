//! Sync HTTP scorer clients (feature `http-scorer`): ureq + the shared wire contract, no ORT.
use anyhow::{Context, Result};

use crate::gate::nli::NliScorer;
use crate::http_scorer::wire;
use crate::retrieve::rerank::Reranker;

/// Injectable transport so the clients unit-test with a mock (no socket); the real impl is ureq.
pub trait HttpTransport {
    fn post_json(&self, url: &str, body: &str, api_key: Option<&str>) -> Result<String>;
}

pub struct UreqTransport {
    pub timeout_ms: u64,
}
impl HttpTransport for UreqTransport {
    fn post_json(&self, url: &str, body: &str, api_key: Option<&str>) -> Result<String> {
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_millis(self.timeout_ms))
            .build();
        let mut req = agent.post(url).set("content-type", "application/json");
        if let Some(k) = api_key {
            req = req.set("authorization", &format!("Bearer {k}"));
        }
        let resp = req
            .send_string(body)
            .context("http scorer request failed")?;
        resp.into_string().context("reading http scorer response")
    }
}

pub struct HttpNli {
    pub endpoint: String,
    pub entail_index: usize,
    pub transport: Box<dyn HttpTransport + Send + Sync>,
    pub api_key: Option<String>,
}
impl NliScorer for HttpNli {
    fn entail(&self, premise: &str, hypotheses: &[&str]) -> Result<Vec<f32>> {
        let body = serde_json::to_string(&wire::build_predict_request(premise, hypotheses))?;
        let url = join(&self.endpoint, "predict");
        let resp = self
            .transport
            .post_json(&url, &body, self.api_key.as_deref())?;
        wire::parse_predict(&resp, self.entail_index, hypotheses.len())
    }
}

pub struct HttpReranker {
    pub endpoint: String,
    pub transport: Box<dyn HttpTransport + Send + Sync>,
    pub api_key: Option<String>,
    /// Wire shape for the operator's backend (TEI vs Jina family).
    pub wire: wire::RerankWire,
    /// Served-model name, sent only by the Jina family and only when set (vLLM needs it).
    pub model: Option<String>,
    /// The operator's backend name, kept because two decisions hang on it beyond the wire shape:
    /// whether its scores are logits, and whether to ask it for raw scores (vLLM).
    pub backend: String,
}
impl Reranker for HttpReranker {
    fn rerank(&self, query: &str, passages: &[&str]) -> Result<Vec<f32>> {
        let body = wire::build_rerank_body(
            self.wire,
            query,
            passages,
            self.model.as_deref(),
            self.backend == "vllm",
        )?;
        let url = join(&self.endpoint, "rerank");
        let resp = self
            .transport
            .post_json(&url, &body, self.api_key.as_deref())?;
        wire::parse_rerank(&resp, passages.len())
    }
    fn emits_logits(&self) -> bool {
        wire::backend_emits_logits(&self.backend)
    }
}

fn join(base: &str, path: &str) -> String {
    format!("{}/{}", base.trim_end_matches('/'), path)
}

/// Real ureq-backed constructors used by `resolve_scorer`/`resolve_reranker`.
pub fn new_ureq_nli(
    endpoint: String,
    entail_index: usize,
    timeout_ms: u64,
    api_key: Option<String>,
) -> HttpNli {
    HttpNli {
        endpoint,
        entail_index,
        transport: Box::new(UreqTransport { timeout_ms }),
        api_key,
    }
}
/// Build a remote reranker for `backend` (`tei` | `vllm` | `llamacpp` | `kbi` | `jina` | `cohere`).
/// Errors on an unknown backend name so a misconfiguration fails loudly rather than silently
/// sending the wrong wire shape; callers fail-open to plain BM25 on that error.
pub fn new_ureq_reranker(
    endpoint: String,
    timeout_ms: u64,
    api_key: Option<String>,
    backend: &str,
    model: Option<String>,
) -> Result<HttpReranker> {
    let wire = wire::RerankWire::from_backend(backend)?;
    // Normalized HERE, once, because three decisions key off this string: the wire shape (which
    // `from_backend` normalizes internally), whether the backend emits logits, and whether to ask
    // vLLM for raw scores. Comparing the raw value in the last two would make `backend = "TEI"`
    // silently drop every `rel_rerank` while claiming on stderr that the server normalizes scores.
    let backend = backend.trim().to_ascii_lowercase();
    Ok(HttpReranker {
        endpoint,
        transport: Box::new(UreqTransport { timeout_ms }),
        api_key,
        wire,
        model,
        backend,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct Mock {
        reply: std::result::Result<String, ()>,
        /// Shared with the test so it can assert on the exact request body sent.
        seen: Arc<Mutex<Option<String>>>,
    }
    impl HttpTransport for Mock {
        fn post_json(&self, _url: &str, body: &str, _key: Option<&str>) -> anyhow::Result<String> {
            *self.seen.lock().unwrap() = Some(body.to_string());
            self.reply
                .clone()
                .map_err(|_| anyhow::anyhow!("transport down"))
        }
    }

    #[test]
    fn tei_reranker_sends_texts_and_maps_bare_array() {
        let seen = Arc::new(Mutex::new(None));
        let m = Mock {
            reply: Ok(r#"[{"index":1,"score":5.0},{"index":0,"score":2.0}]"#.into()),
            seen: Arc::clone(&seen),
        };
        let r = HttpReranker {
            endpoint: "http://x/rerank".into(),
            transport: Box::new(m),
            api_key: None,
            wire: wire::RerankWire::Tei,
            model: None,
            backend: "tei".into(),
        };
        let out = r.rerank("q", &["a", "b"]).unwrap();
        assert_eq!(out, vec![2.0, 5.0]);
        let sent: serde_json::Value =
            serde_json::from_str(seen.lock().unwrap().as_deref().unwrap()).unwrap();
        assert_eq!(sent["texts"], serde_json::json!(["a", "b"]));
    }

    #[test]
    fn jina_reranker_sends_documents_and_maps_envelope() {
        let seen = Arc::new(Mutex::new(None));
        let m = Mock {
            reply: Ok(
                r#"{"results":[{"index":1,"relevance_score":5.0},{"index":0,"relevance_score":2.0}]}"#
                    .into(),
            ),
            seen: Arc::clone(&seen),
        };
        let r = HttpReranker {
            endpoint: "http://x/rerank".into(),
            transport: Box::new(m),
            api_key: None,
            wire: wire::RerankWire::Jina,
            model: Some("bge-reranker".into()),
            backend: "jina".into(),
        };
        let out = r.rerank("q", &["a", "b"]).unwrap();
        assert_eq!(out, vec![2.0, 5.0]);
        let sent: serde_json::Value =
            serde_json::from_str(seen.lock().unwrap().as_deref().unwrap()).unwrap();
        assert_eq!(sent["documents"], serde_json::json!(["a", "b"]));
        assert_eq!(sent["model"], serde_json::json!("bge-reranker"));
    }

    #[test]
    fn unknown_backend_is_err_at_construction() {
        assert!(new_ureq_reranker("http://x".into(), 100, None, "vlm", None).is_err());
    }

    #[test]
    fn reranker_transport_error_is_err_fail_open() {
        let m = Mock {
            reply: Err(()),
            seen: Arc::new(Mutex::new(None)),
        };
        let r = HttpReranker {
            endpoint: "http://x/rerank".into(),
            transport: Box::new(m),
            api_key: None,
            wire: wire::RerankWire::Tei,
            model: None,
            backend: "tei".into(),
        };
        assert!(r.rerank("q", &["a"]).is_err());
    }

    #[test]
    fn nli_takes_entail_index() {
        let m = Mock {
            reply: Ok(r#"[[{"label":"e","score":0.9},{"label":"c","score":0.1}]]"#.into()),
            seen: Arc::new(Mutex::new(None)),
        };
        let n = HttpNli {
            endpoint: "http://x/predict".into(),
            entail_index: 0,
            transport: Box::new(m),
            api_key: None,
        };
        assert_eq!(n.entail("prem", &["h"]).unwrap(), vec![0.9]);
    }

    #[test]
    fn http_reranker_reports_logits_per_backend() {
        let mk = |b: &str| new_ureq_reranker("http://x".into(), 10, None, b, None).unwrap();
        assert!(mk("tei").emits_logits());
        assert!(mk("kbi").emits_logits());
        assert!(mk("vllm").emits_logits());
        assert!(mk("llamacpp").emits_logits());
        assert!(!mk("cohere").emits_logits());
        assert!(!mk("jina").emits_logits());
    }

    /// Three decisions key off the backend name, and `from_backend` normalizes only its own. A
    /// miscased or padded name used to pass the wire check, keep sending `raw_scores: true`, and
    /// then report the real logits as "not logits" -- dropping every rel_rerank and printing a
    /// stderr line that was false.
    #[test]
    fn backend_name_is_normalized_for_every_decision_that_reads_it() {
        let mk = |b: &str| new_ureq_reranker("http://x".into(), 10, None, b, None).unwrap();
        assert!(mk("TEI").emits_logits(), "case must not matter");
        assert!(mk(" vLLM ").emits_logits(), "padding must not matter");
        assert_eq!(mk(" vLLM ").backend, "vllm");

        // ... and the normalized name is what decides the vLLM-only request flag.
        let body = wire::build_rerank_body(
            wire::RerankWire::Jina,
            "q",
            &["a"],
            None,
            mk(" vLLM ").backend == "vllm",
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["use_activation"], serde_json::json!(false));
    }
}
