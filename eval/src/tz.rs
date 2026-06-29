//! Shared TensorZero gateway helpers (inference feedback, episode ids).

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

/// One `/inference` call; returns the assistant text blocks (empty on transport/parse failure).
pub fn infer_chat(
    gateway: &str,
    function: &str,
    variant: &str,
    episode_id: &str,
    messages: &[Value],
    tags: &Value,
) -> Option<String> {
    let url = format!("{}/inference", gateway.trim_end_matches('/'));
    let mut body = json!({
        "function_name": function,
        "variant_name": variant,
        "episode_id": episode_id,
        "input": { "messages": messages },
    });
    if tags.as_object().is_some_and(|o| !o.is_empty()) {
        body["tags"] = tags.clone();
    }
    let text = ureq::post(&url)
        .timeout(Duration::from_secs(120))
        .set("Content-Type", "application/json")
        .send_string(&body.to_string())
        .ok()?
        .into_string()
        .ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    if v.get("error").is_some() {
        return None;
    }
    let content = v.get("content").and_then(|c| c.as_array());
    Some(inference_text(content.map(|a| a.as_slice()).unwrap_or(&[])))
}
