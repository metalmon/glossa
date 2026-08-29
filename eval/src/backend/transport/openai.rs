//! `OpenAiTransport`: the OpenAI-compatible Chat Completions (`/v1/chat/completions`) impl of
//! `ChatTransport`. MOVED here verbatim from `backend/openai.rs` behind the neutral trait seam
//! introduced in Task 1 — behavior-identical: same request shape, same retry/backoff, same raw
//! `Value` round-trip (so provider-specific fields like `reasoning_content` survive).
//!
//! `chat_once`/`lmstudio_chat` in `backend/openai.rs` remain THIN back-compat wrappers over the
//! `chat_http` core moved here (their 9 existing callers aren't migrated in this task — see
//! Phase 2 Task 6 in the multi-api-transport plan).

use super::{http_client, runtime, ChatTransport, ToolCall, TurnReply};
use crate::lab::Endpoint;
use anyhow::anyhow;
use serde_json::{json, Value};
use std::time::Duration;

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
        temperature: f64,
    ) -> anyhow::Result<TurnReply> {
        // System, when present, is folded in as a leading {role:"system"} message — the OpenAI
        // Chat Completions shape has no top-level `system` field (unlike Anthropic's `/v1/messages`).
        let mut full_messages = Vec::with_capacity(messages.len() + 1);
        if let Some(s) = system {
            full_messages.push(json!({ "role": "system", "content": s }));
        }
        full_messages.extend_from_slice(messages);

        let mut body = json!({
            "model": ep.model,
            "messages": full_messages,
            "temperature": temperature,
        });
        if let Some(t) = tools {
            body["tools"] = t.clone();
        }

        let msg = chat_http(
            &ep.endpoint,
            ep.resolve_key().as_deref(),
            &body,
            Duration::from_secs(ep.timeout_secs),
        )?;

        let tool_calls: Vec<ToolCall> = msg
            .get("tool_calls")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .map(|call| ToolCall {
                        id: call.get("id").and_then(Value::as_str).unwrap_or("").to_string(),
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
            raw: msg,
        })
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
/// `lmstudio_chat`/`chat_once`/`OpenAiTransport::call` build) to an OpenAI-compatible
/// `/v1/chat/completions` endpoint via `reqwest` and return the assistant `message` object
/// (`choices[0].message`) as a raw `Value`. Thin wrapper over [`chat_http_full`] for callers that
/// only want the message (`OpenAiTransport::call`); the token accounting + retry hardening all live
/// in `chat_http_full`, so both entry points record usage exactly once.
///
/// Deliberately RAW `Value` in and out — no typed OpenAI SDK structs — so provider-specific fields
/// survive the round-trip. In particular reasoning models (MiMo on OpenCode Zen) return
/// `reasoning_content` on assistant tool-call turns and require it echoed back on the next request;
/// typed message structs drop it and the provider then rejects the follow-up with HTTP 400.
pub(crate) fn chat_http(
    endpoint: &str,
    api_key: Option<&str>,
    body: &Value,
    timeout: Duration,
) -> anyhow::Result<Value> {
    let full = chat_http_full(endpoint, api_key, body, timeout)?;
    full.pointer("/choices/0/message")
        .cloned()
        .ok_or_else(|| anyhow!("chat response had no choices[0].message"))
}

/// Like [`chat_http`], but returns the WHOLE parsed response (validated to have
/// `choices[0].message`) rather than reducing it to just that field — so callers can also read
/// `choices[0].finish_reason` (e.g. `backend::openai::lmstudio_chat`'s length-resample) or extract
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
    for attempt in 1..=4u32 {
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
                        (retryable, Err(anyhow!("chat endpoint returned an error: {err}")))
                    } else if v.pointer("/choices/0/message").is_some() {
                        // Tally usage into the process-global counters before handing the FULL
                        // response back (see `backend::openai::record_usage`).
                        record_usage(&v);
                        (false, Ok(v))
                    } else {
                        (false, Err(anyhow!("chat response had no choices[0].message")))
                    }
                }
                Err(e) => (false, Err(anyhow!("parse chat response json: {e}"))),
            }
        });
        match outcome {
            Ok(v) => return Ok(v),
            Err(e) => {
                last_err = Some(e);
                if retryable && attempt < 4 {
                    std::thread::sleep(Duration::from_millis(400 * attempt as u64));
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
        assert!(s.contains("\"type\":\"function\""), "expected function envelope, got: {s}");
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
        OpenAiTransport.push_tool_results(&mut messages, &[("id1".to_string(), "body1".to_string())]);
        assert_eq!(
            messages,
            vec![json!({ "role": "tool", "tool_call_id": "id1", "content": "body1" })]
        );
    }
}
