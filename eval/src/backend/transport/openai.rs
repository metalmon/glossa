//! `OpenAiTransport`: the OpenAI-compatible Chat Completions (`/v1/chat/completions`) impl of
//! `ChatTransport`. MOVED here verbatim from `backend/openai.rs` behind the neutral trait seam
//! introduced in Task 1 — behavior-identical: same request shape, same retry/backoff, same raw
//! `Value` round-trip (so provider-specific fields like `reasoning_content` survive).
//!
//! This module also owns the OpenAI-compatible request-building the whole eval shares:
//! [`agent_chat_full`] (the agent-loop body + a single round-trip, returning the FULL response so
//! the resample layer can read `finish_reason`) and the `min_p`/`max_tokens` sampling extensions
//! folded into [`OpenAiTransport::call`]. The provider-neutral resample loop lives in
//! [`crate::backend::resample`]; `backend::openai::chat_once` is the one-shot (tools-free) judge path.

use super::{http_client, runtime, ChatTransport, ToolCall, TurnReply};
use crate::backend::resilience::RetryPolicy;
use crate::lab::Endpoint;
use anyhow::anyhow;
use serde_json::{json, Value};
use std::time::Duration;

/// Default `min_p` nucleus-sampling floor for an agent-loop chat call, overridable via
/// `KB_EVAL_MIN_P`. Trims the tail of low-probability tokens that otherwise widens at high
/// temperature, cutting down on degenerate generation loops without forcing low-temperature
/// (deterministic) sampling. `min_p` is an OpenAI-EXTENSION param (not core OpenAI), so it is sent
/// only by this OpenAI-compatible transport / [`agent_chat_full`] — never by [`crate::backend::
/// openai::chat_once`] (which targets a strict provider that may 400 on non-OpenAI fields) nor by
/// the Anthropic/Responses transports.
pub(crate) const DEFAULT_MIN_P: f64 = 0.1;

/// Cap on completion length per call. Bounds a runaway generation (a degenerate loop can otherwise
/// spew thousands of tokens before the server stops) while staying generous enough not to clip a
/// legitimate multi-node `graph_upsert` batch or a normal reader answer. Overridable via
/// `KB_EVAL_MAX_TOKENS`.
pub(crate) const DEFAULT_MAX_TOKENS: u64 = 16384;

/// Resolve the agent-loop `min_p` floor (env `KB_EVAL_MIN_P` > [`DEFAULT_MIN_P`]).
pub(crate) fn agent_min_p() -> f64 {
    std::env::var("KB_EVAL_MIN_P")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MIN_P)
}

/// Resolve the per-call completion cap (env `KB_EVAL_MAX_TOKENS` > [`DEFAULT_MAX_TOKENS`]).
pub(crate) fn agent_max_tokens() -> u64 {
    std::env::var("KB_EVAL_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_TOKENS)
}

pub struct OpenAiTransport;

impl ChatTransport for OpenAiTransport {
    fn tools_schema(&self, graph_on: bool) -> Value {
        tools_schema(graph_on)
    }

    fn call(
        &self,
        ep: &Endpoint,
        system: Option<&str>,
        messages: &[Value],
        tools: Option<&Value>,
        temperature: Option<f64>,
    ) -> anyhow::Result<TurnReply> {
        // System, when present, is folded in as a leading {role:"system"} message — the OpenAI
        // Chat Completions shape has no top-level `system` field (unlike Anthropic's `/v1/messages`).
        let mut full_messages = Vec::with_capacity(messages.len() + 1);
        if let Some(s) = system {
            full_messages.push(json!({ "role": "system", "content": s }));
        }
        full_messages.extend_from_slice(messages);

        // Anti-loop sampling extensions this OpenAI-compatible transport always sends (the reader,
        // build, distil and reason all drive their agent loop through here): `min_p` nucleus floor
        // + a `max_tokens` completion cap. Both are what `lmstudio_chat` used to send before it was
        // retired into the provider-neutral resample layer — keeping them HERE (OpenAI-compat
        // request-building) is what restores them for the reader and unifies every stage. `min_p` is
        // an OpenAI-extension param, so it is scoped to this transport (Anthropic/Responses omit it).
        let mut body = json!({
            "model": ep.model,
            "messages": full_messages,
            "min_p": agent_min_p(),
            "max_tokens": agent_max_tokens(),
        });
        // Include `temperature` only when set — `None` omits it so the provider default applies.
        if let Some(t) = temperature {
            body["temperature"] = json!(t);
        }
        if let Some(t) = tools {
            body["tools"] = t.clone();
        }

        // Retry/backoff comes from this endpoint's opt-in `rate_limit`; absent -> the historical
        // 4-attempt / 400ms*attempt defaults (see `RetryPolicy::from_rate_limit`).
        let retry = RetryPolicy::from_rate_limit(ep.rate_limit.as_ref());
        let full = chat_http_full(
            &ep.endpoint,
            ep.resolve_key().as_deref(),
            &body,
            Duration::from_secs(ep.timeout_secs),
            retry,
        )?;
        reply_from_response(&full)
    }

    fn push_assistant_turn(&self, messages: &mut Vec<Value>, reply: &TurnReply) {
        // Echo the provider's own message object verbatim (preserves e.g. `reasoning_content`
        // that reasoning models require echoed back on the next request).
        messages.push(reply.raw.clone());
    }

    fn push_tool_results(&self, messages: &mut Vec<Value>, results: &[(String, String)]) {
        for (id, body) in results {
            messages.push(json!({ "role": "tool", "tool_call_id": id, "content": body }));
        }
    }
}

/// Sync HTTP bridge: POST our JSON request `body` (the `{model, messages, tools, temperature}` shape
/// `agent_chat_full`/`chat_once`/`OpenAiTransport::call` build) to an OpenAI-compatible
/// `/v1/chat/completions` endpoint via `reqwest` and return the assistant `message` object
/// (`choices[0].message`) as a raw `Value`. Thin wrapper over [`chat_http_full`] for callers that
/// only want the message (`OpenAiTransport::call`); the token accounting + retry hardening all live
/// in `chat_http_full`, so both entry points record usage exactly once.
///
/// Deliberately RAW `Value` in and out — no typed OpenAI SDK structs — so provider-specific fields
/// survive the round-trip. In particular reasoning models (MiMo on OpenCode Zen) return
/// `reasoning_content` on assistant tool-call turns and require it echoed back on the next request;
/// typed message structs drop it and the provider then rejects the follow-up with HTTP 400.
// Retained as a thin message-extracting wrapper for the tests + external callers that don't carry a
// `RateLimit`; `OpenAiTransport::call` now goes through `chat_http_full` directly to pass a policy.
#[allow(dead_code)]
pub(crate) fn chat_http(
    endpoint: &str,
    api_key: Option<&str>,
    body: &Value,
    timeout: Duration,
) -> anyhow::Result<Value> {
    // Back-compat entry point: keeps the historical retry defaults (its callers don't carry an
    // `Endpoint`/`RateLimit`); the `Endpoint`-aware paths go through `chat_http_full` with a policy.
    let full = chat_http_full(endpoint, api_key, body, timeout, RetryPolicy::default())?;
    full.pointer("/choices/0/message")
        .cloned()
        .ok_or_else(|| anyhow!("chat response had no choices[0].message"))
}

/// Like [`chat_http`], but returns the WHOLE parsed response (validated to have
/// `choices[0].message`) rather than reducing it to just that field — so callers can also read
/// `choices[0].finish_reason` (e.g. the resample layer's length check, via `agent_chat_full`) or extract
/// the message themselves (`chat_once`). `finish_reason` must NEVER be injected into the message
/// object itself, since that object is echoed back into `messages` on the next request and a strict
/// endpoint may 400 on an unrecognized field.
///
/// Retries transient failures (transport drop, 429, 5xx, and — via
/// [`crate::backend::openai::is_transient_upstream`] — a 4xx / HTTP-200 body that is really an
/// UPSTREAM gateway failure, e.g. opencode zen's "Engine protocol predict request failed: fetch
/// failed") with `sleep(400ms * attempt)` backoff over `attempts 1..=4`. A genuine bad-request (bad
/// key, malformed payload, unknown model) matches none of the transient predicates and still fails
/// fast, so a real config bug isn't hidden behind seconds of backoff. On success, tallies the call's
/// usage into the process-global counters via [`crate::backend::openai::record_usage`] before
/// returning (the accounting subsystem lives in `backend::openai`; this bridge calls back into it).
///
/// `endpoint` is the FULL chat-completions URL (e.g. `http://localhost:1234/v1/chat/completions`),
/// POSTed verbatim — this function appends nothing. Callers configure the complete URL in
/// `lab.toml`'s `endpoint`, so the reader/judge/build/reflect paths all hit the URL as given with
/// no hidden path-rewriting (which previously double-appended `/v1` on the non-normalizing path).
pub(crate) fn chat_http_full(
    endpoint: &str,
    api_key: Option<&str>,
    body: &Value,
    timeout: Duration,
    retry: RetryPolicy,
) -> anyhow::Result<Value> {
    use crate::backend::openai::{is_transient_upstream, record_usage};
    let url = endpoint.to_string();

    // Trim trailing whitespace from every message: some strict providers (OpenCode Zen / mimo)
    // return HTTP 400 when a message ends in a newline. Harmless to content semantics.
    let mut body = body.clone();
    if let Some(msgs) = body.get_mut("messages").and_then(Value::as_array_mut) {
        for m in msgs {
            if let Some(s) = m.get("content").and_then(Value::as_str) {
                let trimmed = s.trim_end().to_string();
                m["content"] = Value::String(trimmed);
            }
        }
    }

    // RAW JSON round-trip via reqwest — deliberately NOT async-openai's typed message structs.
    // Reasoning models like MiMo (OpenCode Zen) return `reasoning_content` on assistant tool-call
    // turns and REQUIRE it echoed back on the next request; the SDK's typed structs silently drop
    // that field, causing HTTP 400 mid-loop. Passing raw `Value` messages preserves it end-to-end.
    let mut last_err = None;
    for attempt in 1..=retry.attempts {
        let (retryable, outcome): (bool, anyhow::Result<Value>) = runtime().block_on(async {
            let mut rb = http_client().post(&url).timeout(timeout).json(&body);
            if let Some(key) = api_key.filter(|k| !k.is_empty()) {
                rb = rb.bearer_auth(key);
            }
            let resp = match rb.send().await {
                Ok(r) => r,
                Err(e) => return (true, Err(anyhow!("send chat request: {e}"))),
            };
            let status = resp.status();
            let text = match resp.text().await {
                Ok(t) => t,
                Err(e) => return (true, Err(anyhow!("read chat response body: {e}"))),
            };
            if !status.is_success() {
                // 429 / 5xx are transient; so is a 4xx (often 400) whose BODY is an UPSTREAM
                // failure the gateway surfaced with a client-error status (e.g. opencode zen's
                // "Engine protocol predict request failed: fetch failed"). A genuine bad-request
                // (bad key, malformed payload, unknown model) does NOT match and still fails fast.
                let retryable = status.as_u16() == 429
                    || status.is_server_error()
                    || is_transient_upstream(&text);
                return (
                    retryable,
                    Err(anyhow!(
                        "chat endpoint returned {status}: {}",
                        text.chars().take(400).collect::<String>()
                    )),
                );
            }
            match serde_json::from_str::<Value>(&text) {
                Ok(v) => {
                    // Some OpenAI-compatible servers (LM Studio) return HTTP 200 with an
                    // `{"error": …}` body instead of a non-2xx status — surface it, don't discard
                    // it behind a vague "no choices" message.
                    if let Some(err) = v.get("error") {
                        // If that error is itself an upstream transient (same wording as the
                        // non-2xx path), retry it too.
                        let retryable = is_transient_upstream(&err.to_string());
                        (
                            retryable,
                            Err(anyhow!("chat endpoint returned an error: {err}")),
                        )
                    } else if v.pointer("/choices/0/message").is_some() {
                        // Tally usage into the process-global counters before handing the FULL
                        // response back (see `backend::openai::record_usage`).
                        record_usage(&v);
                        (false, Ok(v))
                    } else {
                        (
                            false,
                            Err(anyhow!("chat response had no choices[0].message")),
                        )
                    }
                }
                Err(e) => (false, Err(anyhow!("parse chat response json: {e}"))),
            }
        });
        match outcome {
            Ok(v) => return Ok(v),
            Err(e) => {
                last_err = Some(e);
                if retryable && attempt < retry.attempts {
                    std::thread::sleep(retry.backoff(attempt));
                } else {
                    break;
                }
            }
        }
    }
    Err(last_err.unwrap())
}

/// OpenAI function-tool schema for glossa's agent-facing tools, rendered from the single
/// shared registry (`glossa::tools::registry::registry()`) instead of a hand-written per-tool
/// JSON block — MCP and the eval agent can no longer drift apart on name/description/schema.
/// Graph-gated descriptors (glossary/reach/sql) are included only when `graph_on`; registry
/// order is preserved as-is (search/read/grep/glob first, then the graph tools), so ordering
/// here is a byproduct of the registry, not a curated hand-order.
pub(crate) fn tools_schema(graph_on: bool) -> Value {
    let tools: Vec<Value> = glossa::tools::registry::registry()
        .iter()
        .filter(|d| !d.graph_gated || graph_on)
        .map(|d| {
            json!({
                "type": "function",
                "function": {
                    "name": d.name,
                    "description": d.description,
                    "parameters": d.params_schema,
                }
            })
        })
        .collect();
    Value::Array(tools)
}

pub(crate) fn content_of(msg: &Value) -> String {
    msg.get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string()
}

/// Tool-call `function.arguments` is a JSON-encoded string per the OpenAI spec, but some servers
/// (incl. some LM Studio builds) return it as an already-parsed object. Accept both.
pub(crate) fn parse_tool_args(call: &Value) -> Value {
    match call.pointer("/function/arguments") {
        Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
        Some(v @ Value::Object(_)) => v.clone(),
        _ => json!({}),
    }
}

/// Normalize a WHOLE OpenAI-compatible chat response into a neutral [`TurnReply`]: extracts
/// `choices[0].message` (content + tool_calls) plus `choices[0].finish_reason`. The
/// `finish_reason` (`"length"`/`"stop"`/`"tool_calls"`/…) is carried on the reply — NOT injected
/// into `raw` (the message object, echoed back into the transcript next round; a strict endpoint
/// may 400 on an unrecognized field) — so the provider-neutral resample layer can read it. Shared
/// by [`OpenAiTransport::call`] and the `ClosureTransport` shim so both surface `finish_reason`
/// the same way.
pub(crate) fn reply_from_response(full: &Value) -> anyhow::Result<TurnReply> {
    let msg = full
        .pointer("/choices/0/message")
        .cloned()
        .ok_or_else(|| anyhow!("chat response had no choices[0].message"))?;
    let finish_reason = full
        .pointer("/choices/0/finish_reason")
        .and_then(Value::as_str)
        .map(str::to_string);
    let tool_calls: Vec<ToolCall> = msg
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .map(|call| ToolCall {
                    id: call
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    name: call
                        .pointer("/function/name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    args: parse_tool_args(call),
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(TurnReply {
        text: Some(content_of(&msg)),
        tool_calls,
        finish_reason,
        raw: msg,
    })
}

/// One OpenAI-compatible chat round-trip returning the WHOLE response `Value` (so the resample
/// layer, via the `ClosureTransport` shim, can read `choices[0].finish_reason`). This is the
/// body-building + single-call half of the retired `lmstudio_chat` — the resample loop itself now
/// lives provider-neutrally in [`crate::backend::resample::call_with_resample`], driven by the
/// agent loop for EVERY stage (reader included). Builds the agent-loop request body: `temperature`
/// (env `KB_EVAL_TEMP`, default `0.8` — this stage is stochastic and averaged over N runs), `min_p`
/// ([`agent_min_p`]), `max_tokens` ([`agent_max_tokens`]), and the passed `tools`. `url` is the
/// FULL chat-completions URL, POSTed verbatim.
///
/// Used by the closure-based callers still on the `ClosureTransport` shim (`build::extract`,
/// `distil::densify`/`gen`, `reason::seed`, `gepa_graph`) — they capture their own
/// endpoint/model/api_key/tools, exactly as they did when they called `lmstudio_chat`.
pub(crate) fn agent_chat_full(
    url: &str,
    model: &str,
    api_key: Option<&str>,
    tools: &Value,
    messages: &[Value],
    timeout: Duration,
) -> anyhow::Result<Value> {
    // Sampling temperature: overridable via KB_EVAL_TEMP for noise-sensitivity runs (default 0.8),
    // matching the historical `lmstudio_chat` default exactly.
    let temperature: f64 = std::env::var("KB_EVAL_TEMP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.8);
    let body = json!({
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": temperature,
        "min_p": agent_min_p(),
        "max_tokens": agent_max_tokens(),
    });
    // Diagnostics: KB_EVAL_DUMP_REQ=<path> writes the exact request body (incl. the `tools` array
    // with descriptions) sent to the endpoint, to prove what the model actually receives.
    if let Ok(p) = std::env::var("KB_EVAL_DUMP_REQ") {
        let _ = std::fs::write(&p, serde_json::to_string(&body)?);
    }
    chat_http_full(url, api_key, &body, timeout, RetryPolicy::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The configured `endpoint` is POSTed VERBATIM — nothing is appended (no `/v1/chat/completions`)
    /// and nothing is stripped. A distinctive path that any rewriting would corrupt proves it, and
    /// guards the old footgun where a `.../v1` endpoint got a second `/v1` on the chat_once path.
    #[test]
    fn chat_once_posts_endpoint_url_verbatim() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = [0u8; 2048];
            let n = sock.read(&mut buf).unwrap();
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let body = r#"{"choices":[{"message":{"role":"assistant","content":"ok"}}]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            sock.write_all(resp.as_bytes()).unwrap();
            req
        });

        // A path any append (/v1/chat/completions) or strip (/v1) would mangle.
        let endpoint = format!("http://127.0.0.1:{port}/custom/v1/chat/completions");
        let out = crate::backend::openai::chat_once(
            &endpoint,
            "m",
            &[json!({"role": "user", "content": "hi"})],
            None,
            5,
            None,
        )
        .unwrap();
        assert_eq!(out["content"], "ok");

        let req = server.join().unwrap();
        let request_line = req.lines().next().unwrap_or("");
        assert_eq!(
            request_line, "POST /custom/v1/chat/completions HTTP/1.1",
            "endpoint must be POSTed verbatim; got: {request_line}"
        );
    }

    #[test]
    fn parse_tool_args_handles_string_and_object() {
        let s = json!({ "function": { "arguments": "{\"query\":\"abc\"}" } });
        assert_eq!(parse_tool_args(&s)["query"], "abc");
        let o = json!({ "function": { "arguments": { "query": "abc" } } });
        assert_eq!(parse_tool_args(&o)["query"], "abc");
        let bad = json!({ "function": { "arguments": "not json" } });
        assert_eq!(parse_tool_args(&bad), json!({}));
    }

    /// New Task-2 test: `OpenAiTransport.tools_schema(true)` renders the OpenAI function-tool
    /// envelope and includes a graph-gated tool name (only advertised when `graph_on`).
    #[test]
    fn transport_tools_schema_graph_on_has_function_envelope_and_graph_tool() {
        let schema = OpenAiTransport.tools_schema(true);
        let s = serde_json::to_string(&schema).unwrap();
        assert!(
            s.contains("\"type\":\"function\""),
            "expected function envelope, got: {s}"
        );
        let names: Vec<&str> = schema
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t.pointer("/function/name").and_then(Value::as_str))
            .collect();
        assert!(
            names.contains(&"glossary"),
            "expected a graph-gated tool name (glossary) when graph_on; got {names:?}"
        );
    }

    /// New Task-2 test: `push_tool_results` appends one `{role:"tool", tool_call_id, content}`
    /// message per result.
    #[test]
    fn transport_push_tool_results_appends_tool_message() {
        let mut messages: Vec<Value> = vec![];
        OpenAiTransport
            .push_tool_results(&mut messages, &[("id1".to_string(), "body1".to_string())]);
        assert_eq!(
            messages,
            vec![json!({ "role": "tool", "tool_call_id": "id1", "content": "body1" })]
        );
    }
}
