//! Shared TensorZero gateway helpers (inference, feedback, episode ids).
//!
//! `/inference` requests follow the same shape and retry policy as `backend::tensorzero` and
//! `enrich`: `{ function_name, episode_id, input: { messages, system? }, tags?, variant_name? }`.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::time::Duration;

/// Docker on Windows publishes ports on IPv4 only. `localhost` often resolves to `::1` first;
/// Rust's HTTP client then hangs until timeout while `curl` may still work via IPv4.
pub fn gateway_base(gateway: &str) -> String {
    let base = gateway.trim_end_matches('/');
    if let Some(port) = base.strip_prefix("http://localhost:") {
        return format!("http://127.0.0.1:{port}");
    }
    if let Some(port) = base.strip_prefix("https://localhost:") {
        return format!("https://127.0.0.1:{port}");
    }
    if base == "http://localhost" {
        return "http://127.0.0.1".to_string();
    }
    if base == "https://localhost" {
        return "https://127.0.0.1".to_string();
    }
    base.to_string()
}

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
    /// OpenAI-style finish reason when the gateway returns it (`length` = output truncated).
    pub finish_reason: Option<String>,
}

impl InferenceTurn {
    pub fn text(&self) -> String {
        inference_text(&self.content)
    }
}

/// One `/inference` call (kb-eval pattern). Pass `system` to override the variant's system template
/// at runtime (GEPA search/read); omit it for functions whose prompt lives entirely in `tensorzero.toml`.
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
    let base = gateway_base(gateway);
    let url = format!("{base}/inference");
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
    let finish_reason = v
        .get("finish_reason")
        .and_then(|x| x.as_str())
        .map(str::to_string);
    Ok(InferenceTurn {
        content,
        episode_id,
        finish_reason,
    })
}

/// Fail fast when the gateway was not restarted after `tensorzero.toml` changes.
pub fn ensure_function(gateway: &str, function: &str, variant: Option<&str>) -> Result<()> {
    let base = gateway_base(gateway);
    let health_url = format!("{base}/health");
    let mut health_err = None;
    for attempt in 0..5 {
        match ureq::get(&health_url)
            .timeout(Duration::from_secs(10))
            .call()
        {
            Ok(_) => {
                health_err = None;
                break;
            }
            Err(e) => {
                health_err = Some(e);
                if attempt < 4 {
                    std::thread::sleep(Duration::from_secs(2));
                }
            }
        }
    }
    if let Some(e) = health_err {
        bail!(
            "gateway not ready at {health_url} ({e}) — is `tensorzero-gateway-1` up? \
             After `just gw-restart` wait ~10s; on Windows use `http://127.0.0.1:3000` not localhost"
        );
    }
    let url = format!("{base}/inference");
    let mut body = json!({
        "function_name": function,
        "episode_id": backdated_episode_id(30),
        "input": { "messages": [{"role": "user", "content": "ping"}] },
    });
    if let Some(variant) = variant {
        body["variant_name"] = json!(variant);
    }
    match ureq::post(&url)
        // LM Studio cold start + qwen tool-call often exceeds 15s; keep below infer() timeouts.
        .timeout(Duration::from_secs(90))
        .set("Content-Type", "application/json")
        .send_string(&serde_json::to_string(&body)?)
    {
        Ok(resp) => {
            if resp.status() == 404 {
                bail!(
                    "gateway unknown function '{function}' — restart gateway after config change: \
                     `just gw-restart` (or `cd eval/tensorzero && docker compose restart gateway`)"
                );
            }
            Ok(())
        }
        Err(ureq::Error::Status(404, _)) => {
            bail!(
                "gateway unknown function '{function}' — restart gateway after config change: \
                 `just gw-restart`"
            );
        }
        Err(e) => Err(anyhow!("gateway probe for '{function}' failed: {e}")),
    }
}

/// POST episode-level feedback to the gateway (best-effort; never fails the caller).
pub fn post_feedback(gateway: &str, episode_id: &str, metric: &str, value: Value, tags: &Value) {
    let url = format!("{}/feedback", gateway_base(gateway));
    let mut body = json!({ "episode_id": episode_id, "metric_name": metric, "value": value });
    if tags.as_object().is_some_and(|o| !o.is_empty()) {
        body["tags"] = tags.clone();
    }
    let _ = ureq::post(&url)
        .timeout(Duration::from_secs(30))
        .set("Content-Type", "application/json")
        .send_string(&serde_json::to_string(&body).unwrap_or_default());
}

#[cfg(test)]
mod tests {
    use super::gateway_base;

    #[test]
    fn gateway_base_replaces_localhost() {
        assert_eq!(gateway_base("http://localhost:3000"), "http://127.0.0.1:3000");
        assert_eq!(gateway_base("http://localhost:3000/"), "http://127.0.0.1:3000");
        assert_eq!(gateway_base("http://127.0.0.1:3000"), "http://127.0.0.1:3000");
        assert_eq!(gateway_base("https://localhost:3000"), "https://127.0.0.1:3000");
    }

    /// `cargo test -p kb-eval live_ensure_function -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_ensure_function_localhost() {
        super::ensure_function("http://localhost:3000", "search", Some("baseline")).unwrap();
    }
}
