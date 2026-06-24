use super::{prompt, AgentBackend};
use crate::dataset::Question;
use anyhow::{anyhow, Context};
use std::path::Path;
use std::time::Duration;

/// Generic OpenAI-compatible chat backend (LM Studio, llama.cpp server, vLLM, OpenRouter, …).
/// The operator configures the server with the glossa MCP so the model calls search/read itself.
pub struct OpenAiBackend {
    pub endpoint: String, // base url, e.g. "http://localhost:1234"
    pub model: String,
    pub api_key: Option<String>,
    pub timeout: Duration,
}

impl AgentBackend for OpenAiBackend {
    fn needs_corpus(&self) -> bool {
        true
    }
    fn answer(&self, _work: &Path, q: &Question) -> anyhow::Result<String> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{ "role": "user", "content": prompt::build_prompt(q) }],
            "temperature": 0.0
        });
        let body_str = serde_json::to_string(&body)?;
        let url = format!("{}/v1/chat/completions", self.endpoint.trim_end_matches('/'));
        let mut req = ureq::post(&url).timeout(self.timeout).set("Content-Type", "application/json");
        if let Some(key) = &self.api_key {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let resp = req.send_string(&body_str).map_err(|e| anyhow!("endpoint request failed: {e}"))?;
        let text = resp.into_string().context("read endpoint response")?;
        let v: serde_json::Value = serde_json::from_str(&text).context("parse endpoint json")?;
        let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
        Ok(prompt::parse_answer(content))
    }
}
