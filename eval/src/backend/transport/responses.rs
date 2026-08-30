//! `ResponsesTransport`: the OpenAI Responses API (`/v1/responses`) impl of `ChatTransport`. The
//! 3rd transport — the `qwen`/`luna` provider serves some models ONLY on this API, not Chat
//! Completions. Structurally mirrors `openai.rs`/`anthropic.rs` (same trait, same retry/backoff
//! shape, same raw-`Value` round-trip so provider-specific fields survive), but speaks the
//! Responses API's own wire vocabulary: a single flat `input` item array (STATELESS — the full
//! array is resent every call; `previous_response_id` is deliberately never used), a top-level
//! `instructions` string instead of a `{role:"system"}` message, and Responses' flat
//! `{type:"function",name,description,parameters}` tool-schema shape (no nested `function` object
//! like OpenAI Chat Completions).
//!
//! Wire format verified against the live endpoint (see the multi-api-transport plan, Task 5b).

use super::openai::parse_tool_args;
use super::{http_client, runtime, ChatTransport, ToolCall, TurnReply};
use crate::backend::resilience::RetryPolicy;
use crate::lab::Endpoint;
use anyhow::anyhow;
use serde_json::{json, Value};
use std::time::Duration;

/// Default `max_output_tokens` for a turn when the caller doesn't need a different cap —
/// Responses' analog of Chat Completions' `max_tokens` / Anthropic's `max_tokens`.
const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 4096;

pub struct ResponsesTransport;

impl ChatTransport for ResponsesTransport {
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
        // Responses has no `{role:"system"}` item in `input` — it's a dedicated top-level
        // `instructions` string, mirroring Anthropic's top-level `system` (unlike OpenAI Chat
        // Completions, which folds it into `messages`).
        let mut body = json!({
            "model": ep.model,
            "input": messages,
            "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        });
        // Include `temperature` only when set — `None` omits it so the provider default applies.
        if let Some(t) = temperature {
            body["temperature"] = json!(t);
        }
        if let Some(s) = system {
            body["instructions"] = json!(s);
        }
        if let Some(t) = tools {
            body["tools"] = t.clone();
        }

        let resp = responses_http(
            &ep.endpoint,
            ep.resolve_key().as_deref(),
            &body,
            Duration::from_secs(ep.timeout_secs),
            RetryPolicy::from_rate_limit(ep.rate_limit.as_ref()),
        )?;

        Ok(parse_response(resp))
    }

    fn push_assistant_turn(&self, messages: &mut Vec<Value>, reply: &TurnReply) {
        // Echo the provider's own `output` items verbatim — this preserves reasoning items
        // alongside `message`/`function_call` items, exactly as the Responses API returned them.
        let output = reply
            .raw
            .get("output")
            .and_then(Value::as_array)
            .filter(|items| !items.is_empty())
            .cloned()
            .unwrap_or_else(|| reconstruct_output_items(reply));
        messages.extend(output);
    }

    fn push_tool_results(&self, messages: &mut Vec<Value>, results: &[(String, String)]) {
        // Responses appends ONE `function_call_output` item per tool result directly to the
        // flat `input` array — unlike Anthropic's batched single user message.
        for (call_id, body) in results {
            messages.push(json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": body,
            }));
        }
    }
}

/// Rebuild minimal `output` items when `reply.raw` didn't carry a usable `output` array (e.g. a
/// hand-built `TurnReply` in a test) — a `message` item from `reply.text` (when non-empty)
/// followed by a `function_call` item per `reply.tool_calls`.
fn reconstruct_output_items(reply: &TurnReply) -> Vec<Value> {
    let mut items = Vec::new();
    if let Some(t) = reply.text.as_deref().filter(|t| !t.is_empty()) {
        items.push(json!({
            "type": "message",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": t }],
        }));
    }
    for call in &reply.tool_calls {
        items.push(json!({
            "type": "function_call",
            "call_id": call.id,
            "name": call.name,
            "arguments": serde_json::to_string(&call.args).unwrap_or_default(),
        }));
    }
    items
}

/// Responses API tool schema for glossa's agent-facing tools, rendered from the same shared
/// registry (`glossa::tools::registry::registry()`) the other transports use — same
/// name/description/params_schema per tool, but Responses' FLAT function shape
/// (`{type:"function",name,description,parameters}` at the top level, NOT OpenAI Chat's nested
/// `{type:"function",function:{...}}`). Graph-gated descriptors are included only when
/// `graph_on`.
pub(crate) fn tools_schema(graph_on: bool) -> Value {
    let tools: Vec<Value> = glossa::tools::registry::registry()
        .iter()
        .filter(|d| !d.graph_gated || graph_on)
        .map(|d| {
            json!({
                "type": "function",
                "name": d.name,
                "description": d.description,
                "parameters": d.params_schema,
            })
        })
        .collect();
    Value::Array(tools)
}

/// Parse a top-level Responses `/v1/responses` response (`{output:[...], ...}`) into the neutral
/// `TurnReply`: `text` is the concatenation of `output_text` content parts from `message` items,
/// `tool_calls` is one `ToolCall` per `function_call` item (its `call_id` becomes `ToolCall::id`,
/// and `arguments` — a JSON-encoded string per the Responses spec — is parsed via the same
/// string-or-object leniency `OpenAiTransport::parse_tool_args` uses for Chat Completions).
/// `raw` keeps the full response so `push_assistant_turn` can echo `output` items verbatim.
fn parse_response(resp: Value) -> TurnReply {
    let items = resp
        .get("output")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let text: String = items
        .iter()
        .filter(|it| it.get("type").and_then(Value::as_str) == Some("message"))
        .filter_map(|it| it.get("content").and_then(Value::as_array))
        .flat_map(|content| content.iter())
        .filter(|c| c.get("type").and_then(Value::as_str) == Some("output_text"))
        .filter_map(|c| c.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("");

    let tool_calls: Vec<ToolCall> = items
        .iter()
        .filter(|it| it.get("type").and_then(Value::as_str) == Some("function_call"))
        .map(|it| ToolCall {
            id: it
                .get("call_id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            name: it
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            args: parse_tool_args_responses(it),
        })
        .collect();

    // Normalize the Responses API's incompletion signal to the neutral `finish_reason` the resample
    // layer reads: an `incomplete` status whose `incomplete_details.reason` is `max_output_tokens`
    // (its length-cap signal) maps to `"length"`, so a truncated turn resamples like an OpenAI
    // `finish_reason == "length"` one. Other/absent statuses leave `finish_reason` as `None`.
    let finish_reason = if resp.get("status").and_then(Value::as_str) == Some("incomplete")
        && resp.pointer("/incomplete_details/reason").and_then(Value::as_str)
            == Some("max_output_tokens")
    {
        Some("length".to_string())
    } else {
        None
    };

    TurnReply {
        text: Some(text),
        tool_calls,
        finish_reason,
        raw: resp,
    }
}

/// `function_call` items' `arguments` field is a JSON-encoded string per the Responses spec, but
/// (mirroring `openai::parse_tool_args`'s leniency for Chat Completions) accept an
/// already-parsed object too. Reuses `openai::parse_tool_args`'s string-or-object logic by
/// wrapping the item in the `{"function":{"arguments":...}}` shape that helper expects.
fn parse_tool_args_responses(item: &Value) -> Value {
    parse_tool_args(&json!({ "function": { "arguments": item.get("arguments").cloned().unwrap_or(Value::Null) } }))
}

/// Sync HTTP bridge: POST our `{model, input, instructions?, tools?, max_output_tokens,
/// temperature}` body to a Responses-compatible `/v1/responses` endpoint via `reqwest`, and return
/// the full response `Value` (like Anthropic's `messages_http`, unlike OpenAI Chat's `chat_http` —
/// Responses' shape is already a single top-level object, so nothing to unwrap to `choices[0]`).
/// `endpoint` is POSTed verbatim.
///
/// Auth is standard `Authorization: Bearer <key>` (like Chat Completions, unlike Anthropic's
/// `x-api-key`). Retries transient failures (transport drop, 429, 5xx) with the same backoff shape
/// as the other two transports; a 4xx or a parse error is a hard error surfaced immediately.
fn responses_http(
    endpoint: &str,
    api_key: Option<&str>,
    body: &Value,
    timeout: Duration,
    retry: RetryPolicy,
) -> anyhow::Result<Value> {
    let url = endpoint.to_string();

    let mut last_err = None;
    for attempt in 1..=retry.attempts {
        let (retryable, outcome): (bool, anyhow::Result<Value>) = runtime().block_on(async {
            let mut rb = http_client().post(&url).timeout(timeout).json(body);
            if let Some(key) = api_key.filter(|k| !k.is_empty()) {
                rb = rb.bearer_auth(key);
            }
            let resp = match rb.send().await {
                Ok(r) => r,
                Err(e) => return (true, Err(anyhow!("send responses request: {e}"))),
            };
            let status = resp.status();
            let text = match resp.text().await {
                Ok(t) => t,
                Err(e) => return (true, Err(anyhow!("read responses response body: {e}"))),
            };
            if !status.is_success() {
                let retryable = status.as_u16() == 429 || status.is_server_error();
                return (
                    retryable,
                    Err(anyhow!(
                        "responses endpoint returned {status}: {}",
                        text.chars().take(400).collect::<String>()
                    )),
                );
            }
            match serde_json::from_str::<Value>(&text) {
                Ok(v) => {
                    // Some proxies return HTTP 200 with an `{"error": …}` body instead of a
                    // non-2xx status — surface it rather than silently returning empty content.
                    if let Some(err) = v.get("error") {
                        (
                            false,
                            Err(anyhow!("responses endpoint returned an error: {err}")),
                        )
                    } else {
                        (false, Ok(v))
                    }
                }
                Err(e) => (false, Err(anyhow!("parse responses response json: {e}"))),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_schema_is_flat_function_form() {
        let schema = ResponsesTransport.tools_schema(true);
        let s = serde_json::to_string(&schema).unwrap();
        assert!(
            !s.contains("\"function\":{"),
            "must NOT use OpenAI Chat's nested function envelope: {s}"
        );
        assert!(
            s.contains("\"type\":\"function\""),
            "expected the flat type:function marker: {s}"
        );
        let tools = schema.as_array().unwrap();
        assert!(
            tools
                .iter()
                .any(|t| t.get("name").and_then(Value::as_str) == Some("glossary")
                    && t.get("parameters").is_some_and(Value::is_object)
                    && t.get("description").is_some()),
            "expected flat name/description/parameters at top level, got {tools:?}"
        );
    }

    #[test]
    fn tools_schema_graph_off_omits_graph_tool() {
        let schema = ResponsesTransport.tools_schema(false);
        let names: Vec<&str> = schema
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t.get("name").and_then(Value::as_str))
            .collect();
        assert!(
            !names.contains(&"glossary"),
            "graph tool must be gated off when graph_on=false; got {names:?}"
        );
    }

    #[test]
    fn parse_response_extracts_text_and_tool_calls() {
        let resp = json!({
            "output": [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
                {"type": "function_call", "call_id": "c1", "name": "search", "arguments": "{\"q\":\"x\"}"}
            ]
        });
        let reply = parse_response(resp);
        assert_eq!(reply.text.as_deref(), Some("hi"));
        assert_eq!(reply.tool_calls.len(), 1);
        assert_eq!(reply.tool_calls[0].id, "c1");
        assert_eq!(reply.tool_calls[0].name, "search");
        assert_eq!(reply.tool_calls[0].args, json!({"q": "x"}));
    }

    #[test]
    fn push_tool_results_appends_function_call_output_items() {
        let mut messages: Vec<Value> = vec![];
        ResponsesTransport.push_tool_results(
            &mut messages,
            &[
                ("id1".to_string(), "body1".to_string()),
                ("id2".to_string(), "body2".to_string()),
            ],
        );
        assert_eq!(messages.len(), 2, "one item per result, appended directly to input");
        assert_eq!(messages[0]["type"], "function_call_output");
        assert_eq!(messages[0]["call_id"], "id1");
        assert_eq!(messages[0]["output"], "body1");
        assert_eq!(messages[1]["type"], "function_call_output");
        assert_eq!(messages[1]["call_id"], "id2");
        assert_eq!(messages[1]["output"], "body2");
    }

    #[test]
    fn push_assistant_turn_echoes_raw_output_items_verbatim() {
        let mut messages: Vec<Value> = vec![];
        let reply = TurnReply {
            text: Some("hi".to_string()),
            tool_calls: vec![ToolCall {
                id: "c1".to_string(),
                name: "search".to_string(),
                args: json!({"q": "x"}),
            }],
            finish_reason: None,
            raw: json!({
                "output": [
                    {"type": "reasoning", "summary": []},
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
                    {"type": "function_call", "call_id": "c1", "name": "search", "arguments": "{\"q\":\"x\"}"}
                ]
            }),
        };
        ResponsesTransport.push_assistant_turn(&mut messages, &reply);
        assert_eq!(messages.len(), 3, "reasoning item must be preserved verbatim");
        assert_eq!(messages[0]["type"], "reasoning");
        assert_eq!(messages[2]["call_id"], "c1");
    }

    #[test]
    fn push_assistant_turn_reconstructs_when_raw_has_no_output_array() {
        let mut messages: Vec<Value> = vec![];
        let reply = TurnReply {
            text: None,
            tool_calls: vec![ToolCall {
                id: "c1".to_string(),
                name: "search".to_string(),
                args: json!({"q": "x"}),
            }],
            finish_reason: None,
            raw: json!({ "no_output_here": true }),
        };
        ResponsesTransport.push_assistant_turn(&mut messages, &reply);
        assert_eq!(messages.len(), 1, "no message item when reply.text is None");
        assert_eq!(messages[0]["type"], "function_call");
        assert_eq!(messages[0]["call_id"], "c1");
        assert_eq!(messages[0]["name"], "search");
    }

    /// Local-socket integration test mirroring `anthropic.rs`'s `anthropic_socket_roundtrip_*`: a
    /// mock `/v1/responses` server checks the request shape (Bearer auth, top-level
    /// `instructions`, `input`, flat function `tools`), replies with a `function_call` turn then a
    /// final `message` turn, and asserts `run_agent_loop` drives the tool exactly once and returns
    /// the final answer.
    #[test]
    fn responses_socket_roundtrip_runs_tool_then_answers() {
        use crate::backend::agent_loop::run_agent_loop;
        use std::io::{Read, Write};
        use std::net::TcpListener;

        // Read one full HTTP request off `sock`: headers, then exactly `Content-Length` body
        // bytes. A single fixed-size `read()` can return a short read when the request body (here:
        // the full tools schema) spans more than one TCP segment — this loops until the declared
        // body length is actually in hand.
        fn read_full_http_request(sock: &mut std::net::TcpStream) -> String {
            let mut buf = Vec::new();
            let mut chunk = [0u8; 8192];
            loop {
                let n = sock.read(&mut chunk).unwrap();
                assert!(n > 0, "connection closed before a full request was read");
                buf.extend_from_slice(&chunk[..n]);
                let text = String::from_utf8_lossy(&buf);
                if let Some(header_end) = text.find("\r\n\r\n") {
                    let headers = &text[..header_end];
                    let content_length: usize = headers
                        .lines()
                        .find_map(|l| {
                            l.to_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().to_string())
                        })
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    let body_so_far = buf.len() - (header_end + 4);
                    if body_so_far >= content_length {
                        return text.into_owned();
                    }
                }
            }
        }

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        let server = std::thread::spawn(move || {
            let mut requests: Vec<String> = Vec::new();

            // Turn 1: the model asks for a tool.
            {
                let (mut sock, _) = listener.accept().unwrap();
                requests.push(read_full_http_request(&mut sock));
                let body = r#"{"output":[{"type":"function_call","call_id":"c1","name":"search","arguments":"{\"query\":\"x\"}"}]}"#;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                sock.write_all(resp.as_bytes()).unwrap();
            }
            // Turn 2: final text answer.
            {
                let (mut sock, _) = listener.accept().unwrap();
                requests.push(read_full_http_request(&mut sock));
                let body = r#"{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ANSWER: final"}]}]}"#;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                sock.write_all(resp.as_bytes()).unwrap();
            }
            requests
        });

        let ep = Endpoint {
            endpoint: format!("http://127.0.0.1:{port}/v1/responses"),
            model: "gpt-responses-test".to_string(),
            api_key: "sk-test".to_string(),
            api_key_env: String::new(),
            timeout_secs: 5,
            api: crate::lab::ApiKind::OpenAiResponses,
            temperature: None,
            rate_limit: None,
            fallback: Vec::new(),
        };

        let transport = ResponsesTransport;
        let tools = tools_schema(true);
        let exec = |name: &str, args: &Value| {
            assert_eq!(name, "search");
            assert_eq!(args["query"], "x");
            ("hit body".to_string(), vec!["hit-id".to_string()])
        };
        let out = run_agent_loop(
            &transport,
            &ep,
            Some("you are a test system"),
            vec![json!({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "question"}]
            })],
            Some(&tools),
            exec,
            |_, _| "dup".to_string(),
            4,
            None,
        )
        .unwrap();
        assert_eq!(out, "ANSWER: final");

        let requests = server.join().unwrap();
        assert_eq!(requests.len(), 2, "the tool must trigger exactly one extra turn");

        let req0 = &requests[0];
        assert!(
            req0.starts_with("POST /v1/responses HTTP/1.1"),
            "wrong request line: {req0}"
        );
        let req0_lc = req0.to_lowercase();
        assert!(
            req0_lc.contains("authorization: bearer sk-test"),
            "missing Bearer auth header: {req0}"
        );
        assert!(
            !req0_lc.contains("x-api-key"),
            "must NOT use Anthropic's x-api-key auth: {req0}"
        );

        let body0_str = req0.splitn(2, "\r\n\r\n").nth(1).unwrap_or("");
        let body0: Value = serde_json::from_str(body0_str).expect("request body must be JSON");
        assert_eq!(body0["instructions"], "you are a test system");
        assert_eq!(body0["model"], "gpt-responses-test");
        assert_eq!(body0["max_output_tokens"], json!(DEFAULT_MAX_OUTPUT_TOKENS));
        assert!(body0["input"].is_array(), "expected a flat input array");
        let req_tools = body0["tools"].as_array().expect("tools must be an array");
        assert!(
            req_tools.iter().any(|t| t["name"] == "search"
                && t["type"] == "function"
                && t["parameters"].is_object()
                && t.get("function").is_none()),
            "expected Responses-shaped flat function tools, got {req_tools:?}"
        );

        // The 2nd request's `input` array must carry the echoed function_call item AND its
        // function_call_output appended directly (no batching wrapper, unlike Anthropic).
        let body1_str = requests[1].splitn(2, "\r\n\r\n").nth(1).unwrap_or("");
        let body1: Value = serde_json::from_str(body1_str).expect("2nd request body must be JSON");
        let input = body1["input"].as_array().unwrap();
        assert!(
            input
                .iter()
                .any(|it| it["type"] == "function_call" && it["call_id"] == "c1"),
            "function_call item must be echoed back: {input:?}"
        );
        assert!(
            input
                .iter()
                .any(|it| it["type"] == "function_call_output" && it["call_id"] == "c1"),
            "function_call_output item must be appended: {input:?}"
        );
    }
}
