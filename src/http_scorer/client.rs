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
}
impl Reranker for HttpReranker {
    fn rerank(&self, query: &str, passages: &[&str]) -> Result<Vec<f32>> {
        let body = serde_json::to_string(&wire::build_rerank_request(query, passages, true))?;
        let url = join(&self.endpoint, "rerank");
        let resp = self
            .transport
            .post_json(&url, &body, self.api_key.as_deref())?;
        wire::parse_rerank_bare(&resp, passages.len())
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
pub fn new_ureq_reranker(
    endpoint: String,
    timeout_ms: u64,
    api_key: Option<String>,
) -> HttpReranker {
    HttpReranker {
        endpoint,
        transport: Box::new(UreqTransport { timeout_ms }),
        api_key,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct Mock {
        reply: std::result::Result<String, ()>,
        seen: Mutex<Option<String>>,
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
    fn reranker_maps_scores_to_input_order() {
        let m = Mock {
            reply: Ok(r#"[{"index":1,"score":5.0},{"index":0,"score":2.0}]"#.into()),
            seen: Mutex::new(None),
        };
        let r = HttpReranker {
            endpoint: "http://x/rerank".into(),
            transport: Box::new(m),
            api_key: None,
        };
        let out = r.rerank("q", &["a", "b"]).unwrap();
        assert_eq!(out, vec![2.0, 5.0]);
    }

    #[test]
    fn reranker_transport_error_is_err_fail_open() {
        let m = Mock {
            reply: Err(()),
            seen: Mutex::new(None),
        };
        let r = HttpReranker {
            endpoint: "http://x/rerank".into(),
            transport: Box::new(m),
            api_key: None,
        };
        assert!(r.rerank("q", &["a"]).is_err());
    }

    #[test]
    fn nli_takes_entail_index() {
        let m = Mock {
            reply: Ok(r#"[[{"label":"e","score":0.9},{"label":"c","score":0.1}]]"#.into()),
            seen: Mutex::new(None),
        };
        let n = HttpNli {
            endpoint: "http://x/predict".into(),
            entail_index: 0,
            transport: Box::new(m),
            api_key: None,
        };
        assert_eq!(n.entail("prem", &["h"]).unwrap(), vec![0.9]);
    }
}
