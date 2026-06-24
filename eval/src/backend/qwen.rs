use super::{prompt, AgentBackend};
use crate::dataset::Question;
use anyhow::{anyhow, Context};
use std::path::Path;

/// Drives a local model via LM Studio's OpenAI-compatible chat API. The operator configures LM Studio
/// (once) with the glossa MCP server so the model itself calls search/read.
pub struct QwenBackend {
    pub url: String,
    pub model: String,
}

impl AgentBackend for QwenBackend {
    fn needs_corpus(&self) -> bool {
        true
    }
    fn answer(&self, _work: &Path, q: &Question) -> anyhow::Result<String> {
        // ureq is default-features=false so the `json` feature is absent;
        // we serialize/deserialize manually via serde_json.
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{ "role": "user", "content": prompt::build_prompt(q) }],
            "temperature": 0.0
        });
        let body_str = serde_json::to_string(&body)?;
        let resp = ureq::post(&format!("{}/v1/chat/completions", self.url))
            .set("Content-Type", "application/json")
            .send_string(&body_str)
            .map_err(|e| anyhow!("lmstudio request failed: {e}"))?;
        let text = resp.into_string().context("read lmstudio response")?;
        let v: serde_json::Value = serde_json::from_str(&text).context("parse lmstudio json")?;
        let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
        Ok(prompt::parse_answer(content))
    }
}
