//! `AnthropicTransport`: the Anthropic Messages API (`/v1/messages`) impl of `ChatTransport`.
//! Structurally mirrors `backend::tensorzero.rs`'s block-based turn handling (assistant-echo with
//! content blocks, batched tool results), but speaks Anthropic's own native block vocabulary
//! (`text` / `thinking` / `tool_use` / `tool_result`) rather than TensorZero's neutral one.

use super::{http_client, runtime, ChatTransport, ToolCall, TurnReply};
use crate::lab::Endpoint;
use anyhow::anyhow;
use serde_json::{json, Value};
use std::time::Duration;

/// Anthropic Messages API version header — a fixed protocol version, not a model version.
/// Hardcoded for now; would become configurable if a future API revision needs opting into.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Default `max_tokens` for a turn when the caller doesn't need a different cap. Anthropic
/// REQUIRES this field (no server-side default), unlike OpenAI's Chat Completions.
const DEFAULT_MAX_TOKENS: u64 = 4096;

pub struct AnthropicTransport;

impl ChatTransport for AnthropicTransport {
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
        // Anthropic has no `{role:"system"}` message — it's a dedicated top-level field, unlike
        // OpenAI Chat Completions where `OpenAiTransport::call` folds it into `messages`.
        let mut body = json!({
            "model": ep.model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "messages": messages,
            "temperature": temperature,
        });
        if let Some(s) = system {
            body["system"] = json!(s);
        }
        if let Some(t) = tools {
            body["tools"] = t.clone();
        }

        let resp = messages_http(
            &ep.endpoint,
            ep.resolve_key().as_deref(),
            &body,
            Duration::from_secs(ep.timeout_secs),
        )?;

        Ok(parse_response(resp))
    }

    fn push_assistant_turn(&self, messages: &mut Vec<Value>, reply: &TurnReply) {
        // Echo the provider's own content blocks verbatim — this preserves `thinking` blocks
        // (extended-thinking models require them echoed back) and `tool_use` blocks alongside
        // any `text` block, exactly as Anthropic returned them.
        let content = reply
            .raw
            .get("content")
            .and_then(Value::as_array)
            .filter(|blocks| !blocks.is_empty())
            .cloned()
            .unwrap_or_else(|| reconstruct_assistant_content(reply));
        messages.push(json!({ "role": "assistant", "content": content }));
    }

    fn push_tool_results(&self, messages: &mut Vec<Value>, results: &[(String, String)]) {
        // Anthropic batches every tool_result for a round into ONE user message — unlike OpenAI's
        // one `{role:"tool"}` message per call.
        let blocks: Vec<Value> = results
            .iter()
            .map(|(id, body)| json!({ "type": "tool_result", "tool_use_id": id, "content": body }))
            .collect();
        messages.push(json!({ "role": "user", "content": blocks }));
    }
}

/// Rebuild a minimal assistant content array when `reply.raw` didn't carry a usable `content`
/// array (e.g. a hand-built `TurnReply` in a test, or a malformed upstream response) — a `text`
/// block from `reply.text` (when non-empty) followed by a `tool_use` block per `reply.tool_calls`.
fn reconstruct_assistant_content(reply: &TurnReply) -> Vec<Value> {
    let mut content = Vec::new();
    if let Some(t) = reply.text.as_deref().filter(|t| !t.is_empty()) {
        content.push(json!({ "type": "text", "text": t }));
    }
    for call in &reply.tool_calls {
        content.push(json!({
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": call.args,
        }));
    }
    content
}

/// Anthropic tool schema for glossa's agent-facing tools, rendered from the same shared registry
/// (`glossa::tools::registry::registry()`) `OpenAiTransport::tools_schema` uses — same
/// name/description/params_schema per tool, but the flat Anthropic envelope (`input_schema`, no
/// `type:"function"` wrapper). Graph-gated descriptors (glossary/reach/sql/…) are included only
/// when `graph_on`.
pub(crate) fn tools_schema(graph_on: bool) -> Value {
    let tools: Vec<Value> = glossa::tools::registry::registry()
        .iter()
        .filter(|d| !d.graph_gated || graph_on)
        .map(|d| {
            json!({
                "name": d.name,
                "description": d.description,
                "input_schema": d.params_schema,
            })
        })
        .collect();
    Value::Array(tools)
}

/// Parse a top-level Anthropic `/v1/messages` response (`{content:[...], stop_reason, ...}`) into
/// the neutral `TurnReply`: `text` is the concatenation of `type:"text"` blocks, `tool_calls` is
/// one `ToolCall` per `type:"tool_use"` block (its `input` object passed straight through as
/// `args` — Anthropic tool-call args are already a JSON object, never a string to re-parse, unlike
/// OpenAI's `function.arguments`). `raw` keeps the full response so `push_assistant_turn` can echo
/// `thinking`/other blocks verbatim.
fn parse_response(resp: Value) -> TurnReply {
    let blocks = resp
        .get("content")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let text: String = blocks
        .iter()
        .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|b| b.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("");

    let tool_calls: Vec<ToolCall> = blocks
        .iter()
        .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))
        .map(|b| ToolCall {
            id: b.get("id").and_then(Value::as_str).unwrap_or("").to_string(),
            name: b
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            args: b.get("input").cloned().unwrap_or_else(|| json!({})),
        })
        .collect();

    TurnReply {
        text: Some(text),
        tool_calls,
        raw: resp,
    }
}

/// Sync HTTP bridge: POST our `{model, max_tokens, system?, messages, tools?, temperature}` body
/// to an Anthropic-compatible `/v1/messages` endpoint via `reqwest`, and return the full response
/// `Value` (unlike OpenAI's `chat_http`, which unwraps to `choices[0].message` — Anthropic's shape
/// is already a single top-level object, so nothing to unwrap). `endpoint` is POSTed verbatim.
///
/// Auth is `x-api-key` (NOT `Authorization: Bearer`) plus the fixed `anthropic-version` header.
/// Retries transient failures (transport drop, 429, 5xx) with the same backoff shape as the
/// OpenAI transport's `chat_http`; a 4xx or a parse error is a hard error surfaced immediately.
fn messages_http(
    endpoint: &str,
    api_key: Option<&str>,
    body: &Value,
    timeout: Duration,
) -> anyhow::Result<Value> {
    let url = endpoint.to_string();

    let mut last_err = None;
    for attempt in 1..=4u32 {
        let (retryable, outcome): (bool, anyhow::Result<Value>) = runtime().block_on(async {
            let mut rb = http_client()
                .post(&url)
                .timeout(timeout)
                .header("anthropic-version", ANTHROPIC_VERSION);
            if let Some(key) = api_key.filter(|k| !k.is_empty()) {
                rb = rb.header("x-api-key", key);
            }
            let resp = match rb.json(body).send().await {
                Ok(r) => r,
                Err(e) => return (true, Err(anyhow!("send anthropic request: {e}"))),
            };
            let status = resp.status();
            let text = match resp.text().await {
                Ok(t) => t,
                Err(e) => return (true, Err(anyhow!("read anthropic response body: {e}"))),
            };
            if !status.is_success() {
                let retryable = status.as_u16() == 429 || status.is_server_error();
                return (
                    retryable,
                    Err(anyhow!(
                        "anthropic endpoint returned {status}: {}",
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
                            Err(anyhow!("anthropic endpoint returned an error: {err}")),
                        )
                    } else {
                        (false, Ok(v))
                    }
                }
                Err(e) => (false, Err(anyhow!("parse anthropic response json: {e}"))),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_schema_has_flat_envelope_and_graph_tool() {
        let schema = AnthropicTransport.tools_schema(true);
        let s = serde_json::to_string(&schema).unwrap();
        assert!(
            !s.contains("\"type\":\"function\""),
            "must NOT use OpenAI's function envelope: {s}"
        );
        assert!(
            s.contains("\"input_schema\""),
            "expected the Anthropic input_schema key: {s}"
        );
        let names: Vec<&str> = schema
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t.get("name").and_then(Value::as_str))
            .collect();
        assert!(
            names.contains(&"glossary"),
            "expected a graph-gated tool (glossary) when graph_on; got {names:?}"
        );
    }

    #[test]
    fn tools_schema_graph_off_omits_graph_tool() {
        let schema = AnthropicTransport.tools_schema(false);
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
    fn push_tool_results_batches_into_one_user_message() {
        let mut messages: Vec<Value> = vec![];
        AnthropicTransport.push_tool_results(
            &mut messages,
            &[
                ("id1".to_string(), "body1".to_string()),
                ("id2".to_string(), "body2".to_string()),
            ],
        );
        assert_eq!(messages.len(), 1, "must batch into exactly ONE user message");
        assert_eq!(messages[0]["role"], "user");
        let blocks = messages[0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[0]["tool_use_id"], "id1");
        assert_eq!(blocks[0]["content"], "body1");
        assert_eq!(blocks[1]["type"], "tool_result");
        assert_eq!(blocks[1]["tool_use_id"], "id2");
        assert_eq!(blocks[1]["content"], "body2");
    }

    #[test]
    fn parse_response_extracts_text_and_tool_calls() {
        let resp = json!({
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}}
            ]
        });
        let reply = parse_response(resp);
        assert_eq!(reply.text.as_deref(), Some("hi"));
        assert_eq!(reply.tool_calls.len(), 1);
        assert_eq!(reply.tool_calls[0].id, "t1");
        assert_eq!(reply.tool_calls[0].name, "search");
        assert_eq!(reply.tool_calls[0].args, json!({"q": "x"}));
    }

    #[test]
    fn push_assistant_turn_echoes_raw_content_blocks_verbatim() {
        let mut messages: Vec<Value> = vec![];
        let reply = TurnReply {
            text: Some("hi".to_string()),
            tool_calls: vec![ToolCall {
                id: "t1".to_string(),
                name: "search".to_string(),
                args: json!({"q": "x"}),
            }],
            raw: json!({
                "content": [
                    {"type": "thinking", "thinking": "reasoning..."},
                    {"type": "text", "text": "hi"},
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}}
                ]
            }),
        };
        AnthropicTransport.push_assistant_turn(&mut messages, &reply);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "assistant");
        let blocks = messages[0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 3, "thinking block must be preserved verbatim");
        assert_eq!(blocks[0]["type"], "thinking");
    }

    #[test]
    fn push_assistant_turn_reconstructs_when_raw_has_no_content_array() {
        let mut messages: Vec<Value> = vec![];
        let reply = TurnReply {
            text: None,
            tool_calls: vec![ToolCall {
                id: "t1".to_string(),
                name: "search".to_string(),
                args: json!({"q": "x"}),
            }],
            raw: json!({ "no_content_here": true }),
        };
        AnthropicTransport.push_assistant_turn(&mut messages, &reply);
        let blocks = messages[0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 1, "no text block when reply.text is None");
        assert_eq!(blocks[0]["type"], "tool_use");
        assert_eq!(blocks[0]["id"], "t1");
        assert_eq!(blocks[0]["name"], "search");
        assert_eq!(blocks[0]["input"], json!({"q": "x"}));
    }

    /// Local-socket integration test mirroring `openai.rs`'s
    /// `chat_once_posts_endpoint_url_verbatim`: a mock `/v1/messages` server checks the request
    /// shape (top-level `system`, `x-api-key` + `anthropic-version` headers, `max_tokens`,
    /// Anthropic-shaped `tools`), replies with a `tool_use` turn then a final `text` turn, and
    /// asserts `run_agent_loop` drives the tool exactly once and returns the final answer.
    #[test]
    fn anthropic_socket_roundtrip_runs_tool_then_answers() {
        use crate::backend::agent_loop::run_agent_loop;
        use std::io::{Read, Write};
        use std::net::TcpListener;

        // Read one full HTTP request off `sock`: headers, then exactly `Content-Length` body
        // bytes. A single fixed-size `read()` can return a short read when the request body
        // (here: the full tools schema) spans more than one TCP segment/packet — this loops
        // until the declared body length is actually in hand.
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
                        .find_map(|l| l.to_lowercase().strip_prefix("content-length:").map(|v| v.trim().to_string()))
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
                let body = r#"{"content":[{"type":"tool_use","id":"t1","name":"search","input":{"query":"x"}}],"stop_reason":"tool_use"}"#;
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
                let body = r#"{"content":[{"type":"text","text":"ANSWER: final"}],"stop_reason":"end_turn"}"#;
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
            endpoint: format!("http://127.0.0.1:{port}/v1/messages"),
            model: "claude-3-test".to_string(),
            api_key: "sk-test".to_string(),
            api_key_env: String::new(),
            timeout_secs: 5,
            api: crate::lab::ApiKind::Anthropic,
        };

        let transport = AnthropicTransport;
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
            vec![json!({"role": "user", "content": [{"type": "text", "text": "question"}]})],
            Some(&tools),
            exec,
            |_, _| "dup".to_string(),
            4,
        )
        .unwrap();
        assert_eq!(out, "ANSWER: final");

        let requests = server.join().unwrap();
        assert_eq!(requests.len(), 2, "the tool must trigger exactly one extra turn");

        let req0 = &requests[0];
        assert!(
            req0.starts_with("POST /v1/messages HTTP/1.1"),
            "wrong request line: {req0}"
        );
        let req0_lc = req0.to_lowercase();
        assert!(
            req0_lc.contains("x-api-key: sk-test"),
            "missing x-api-key header: {req0}"
        );
        assert!(
            req0.contains("anthropic-version: 2023-06-01"),
            "missing anthropic-version header: {req0}"
        );
        assert!(
            !req0_lc.contains("authorization:"),
            "must NOT use Bearer auth: {req0}"
        );

        let body0_str = req0.splitn(2, "\r\n\r\n").nth(1).unwrap_or("");
        let body0: Value = serde_json::from_str(body0_str).expect("request body must be JSON");
        assert_eq!(body0["system"], "you are a test system");
        assert_eq!(body0["max_tokens"], json!(DEFAULT_MAX_TOKENS));
        assert_eq!(body0["model"], "claude-3-test");
        let req_tools = body0["tools"].as_array().expect("tools must be an array");
        assert!(
            req_tools
                .iter()
                .any(|t| t["name"] == "search" && t["input_schema"].is_object()),
            "expected Anthropic-shaped tools with input_schema, got {req_tools:?}"
        );

        // The 2nd request's assistant turn (echoed by the agent loop) must carry the tool_use
        // block AND its tool_result must be batched into a single following user message.
        let body1_str = requests[1].splitn(2, "\r\n\r\n").nth(1).unwrap_or("");
        let body1: Value = serde_json::from_str(body1_str).expect("2nd request body must be JSON");
        let msgs = body1["messages"].as_array().unwrap();
        let assistant_turn = msgs
            .iter()
            .find(|m| m["role"] == "assistant")
            .expect("assistant turn must be echoed back");
        assert!(assistant_turn["content"]
            .as_array()
            .unwrap()
            .iter()
            .any(|b| b["type"] == "tool_use" && b["id"] == "t1"));
        let tool_result_msg = msgs
            .iter()
            .find(|m| {
                m["role"] == "user"
                    && m["content"]
                        .as_array()
                        .is_some_and(|a| a.iter().any(|b| b["type"] == "tool_result"))
            })
            .expect("a batched tool_result user message must be present");
        let tr_blocks = tool_result_msg["content"].as_array().unwrap();
        assert_eq!(tr_blocks.len(), 1);
        assert_eq!(tr_blocks[0]["tool_use_id"], "t1");
    }
}
