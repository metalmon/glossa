//! Shared TensorZero gateway helpers (inference, feedback, episode ids).
//!
//! `/inference` requests follow the same shape and retry policy as `backend::tensorzero` and
//! `enrich`: `{ function_name, episode_id, input: { messages, system? }, tags?, variant_name? }`.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::time::Duration;

/// A UUIDv7 episode id whose embedded timestamp is `secs_back` seconds in the PAST, so the gateway
/// never rejects it as "in the future" under Docker/WSL host↔container clock skew.
pub fn backdated_episode_id(secs_back: u64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs().saturating_sub(secs_back);
    let ts = uuid::Timestamp::from_unix(uuid::NoContext, secs, now.subsec_nanos());
    uuid::Uuid::new_v7(ts).to_string()
}

/// Concatenate `text` blocks from a TensorZero `/inference` content array.
pub fn inference_text(content: &[Value]) -> String {
    content
        .iter()
        .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
        .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
        .collect::<Vec<_>>()
        .join("")
}

/// One `/inference` turn — same fields kb-eval reads from the gateway response.
pub struct InferenceTurn {
    pub content: Vec<Value>,
    pub episode_id: String,
}

impl InferenceTurn {
    pub fn text(&self) -> String {
        inference_text(&self.content)
    }
}

/// One `/inference` call (kb-eval pattern). Pass `system` to override the variant's system template
/// at runtime (GEPA select); omit it for functions whose prompt lives entirely in `tensorzero.toml`.
pub fn infer(
    gateway: &str,
    function: &str,
    episode_id: &str,
    messages: &[Value],
    tags: &Value,
    timeout: Duration,
    variant: Option<&str>,
    system: Option<&str>,
) -> Result<InferenceTurn> {
    let url = format!("{}/inference", gateway.trim_end_matches('/'));
    let mut input = json!({ "messages": messages });
    if let Some(system) = system {
        input["system"] = json!(system);
    }
    let mut body = json!({
        "function_name": function,
        "input": input,
        "episode_id": episode_id,
    });
    if let Some(variant) = variant {
        body["variant_name"] = json!(variant);
    }
    if tags.as_object().is_some_and(|o| !o.is_empty()) {
        body["tags"] = tags.clone();
    }
    let payload = serde_json::to_string(&body)?;
    let mut attempt = 0u32;
    let resp = loop {
        match ureq::post(&url)
            .timeout(timeout)
            .set("Content-Type", "application/json")
            .send_string(&payload)
        {
            Ok(r) => break r,
            Err(e) => {
                let retryable = match &e {
                    ureq::Error::Status(code, _) => *code >= 500,
                    ureq::Error::Transport(_) => true,
                };
                attempt += 1;
                if retryable && attempt <= 3 {
                    std::thread::sleep(Duration::from_millis(500 * u64::from(attempt)));
                    continue;
                }
                return Err(anyhow!("tensorzero /inference failed: {e}"));
            }
        }
    };
    let text = resp.into_string().context("read /inference response")?;
    let v: Value = serde_json::from_str(&text).context("parse /inference json")?;
    if let Some(err) = v.get("error") {
        bail!("tensorzero error: {err}");
    }
    let episode_id = v
        .get("episode_id")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let content = v
        .get("content")
        .and_then(|c| c.as_array())
        .cloned()
        .unwrap_or_default();
    Ok(InferenceTurn { content, episode_id })
}

/// POST episode-level feedback to the gateway (best-effort; never fails the caller).
pub fn post_feedback(gateway: &str, episode_id: &str, metric: &str, value: Value, tags: &Value) {
    let url = format!("{}/feedback", gateway.trim_end_matches('/'));
    let mut body = json!({ "episode_id": episode_id, "metric_name": metric, "value": value });
    if tags.as_object().is_some_and(|o| !o.is_empty()) {
        body["tags"] = tags.clone();
    }
    let _ = ureq::post(&url)
        .timeout(Duration::from_secs(30))
        .set("Content-Type", "application/json")
        .send_string(&serde_json::to_string(&body).unwrap_or_default());
}
