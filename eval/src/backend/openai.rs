use super::{prompt, AgentBackend};
use crate::dataset::Question;
use anyhow::{anyhow, bail, Context};
use glossa::trace::TraceLog;
use serde_json::{json, Value};
use std::path::Path;
use std::time::Duration;

/// Generic OpenAI-compatible chat backend (LM Studio, llama.cpp server, vLLM, OpenRouter, …).
///
/// The harness itself is the agent: it advertises glossa's `search`/`read` as OpenAI function
/// tools, runs the tool-call loop, and executes the tools IN-PROCESS against the corpus in `work`.
/// (We do NOT rely on the server's own MCP/tool execution — that is GUI-only in LM Studio and
/// makes retrieval unobservable.) Every tool call is logged to `work/.glossa/traces` in the same
/// JSONL format the MCP server uses, so `run::eval_one` measures retrieval-recall unchanged.
pub struct OpenAiBackend {
    pub endpoint: String, // base url, e.g. "http://localhost:1234"
    pub model: String,
    pub api_key: Option<String>,
    pub timeout: Duration,
}

const MAX_ROUNDS: usize = 8;

impl AgentBackend for OpenAiBackend {
    fn needs_corpus(&self) -> bool {
        true
    }

    fn answer(&self, work: &Path, q: &Question) -> anyhow::Result<String> {
        let url = format!("{}/v1/chat/completions", self.endpoint.trim_end_matches('/'));
        let tools = tools_schema();
        let chat = |messages: &[Value]| -> anyhow::Result<Value> {
            let body = json!({
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "temperature": 0.0
            });
            let body_str = serde_json::to_string(&body)?;
            let mut req = ureq::post(&url).timeout(self.timeout).set("Content-Type", "application/json");
            if let Some(key) = &self.api_key {
                req = req.set("Authorization", &format!("Bearer {key}"));
            }
            let resp = req.send_string(&body_str).map_err(|e| anyhow!("endpoint request failed: {e}"))?;
            let text = resp.into_string().context("read endpoint response")?;
            let v: Value = serde_json::from_str(&text).context("parse endpoint json")?;
            if let Some(err) = v.get("error") {
                bail!("endpoint returned error: {err}");
            }
            let msg = v["choices"][0]["message"].clone();
            if msg.is_null() {
                bail!("endpoint response missing choices[0].message: {text}");
            }
            Ok(msg)
        };

        let trace = TraceLog::to_dir(work);
        // Open the index once per question; the closure reuses it (cached reader) for every
        // search/read in the agent loop instead of reopening per tool call.
        let idx = glossa::index::store::DocIndex::open_or_create(work)?;
        let exec = |name: &str, args: &Value| execute_tool(name, args, &idx, &trace);

        let messages = vec![
            json!({ "role": "system", "content": prompt::system_prompt() }),
            json!({ "role": "user", "content": prompt::user_prompt(q) }),
        ];
        let raw = run_agent_loop(chat, messages, exec, MAX_ROUNDS)?;
        Ok(prompt::parse_answer(&raw))
    }
}

/// OpenAI function-tool schema for glossa's search/read.
fn tools_schema() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Full-text BM25 search over the knowledge base. Pass short KEYWORDS (morphology-aware), not a sentence. Returns ranked results as `path:location: snippet  [score]`.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "keywords to search for" },
                        "limit": { "type": "integer", "description": "max results (default 10)" }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a document's text. `path` is a path returned by search; `location` optionally narrows to a heading/sheet/page substring.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "document path from a search result" },
                        "location": { "type": "string", "description": "optional heading/page substring" }
                    },
                    "required": ["path"]
                }
            }
        }
    ])
}

/// Drive a tool-calling chat to a final textual answer.
///
/// `chat(messages)` returns the assistant `message` object (already extracted from
/// `choices[0].message`). When it carries `tool_calls`, each is dispatched through `exec(name,
/// args)` and the result fed back as a `role:"tool"` message, then the model is queried again —
/// up to `max_rounds`. The first message without tool calls yields the answer.
fn run_agent_loop<C, F>(
    mut chat: C,
    mut messages: Vec<Value>,
    mut exec: F,
    max_rounds: usize,
) -> anyhow::Result<String>
where
    C: FnMut(&[Value]) -> anyhow::Result<Value>,
    F: FnMut(&str, &Value) -> String,
{
    for _ in 0..max_rounds {
        let msg = chat(&messages)?;
        let calls: Vec<Value> = msg
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        if calls.is_empty() {
            return Ok(content_of(&msg));
        }
        messages.push(msg.clone()); // echo the assistant turn that requested the tools
        for call in &calls {
            let id = call.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let name = call.pointer("/function/name").and_then(|v| v.as_str()).unwrap_or("");
            let args = parse_tool_args(call);
            let result = exec(name, &args);
            messages.push(json!({ "role": "tool", "tool_call_id": id, "content": result }));
        }
    }
    // Out of rounds: nudge for a final answer (the model often keeps requesting tools otherwise)
    // and take whatever text it gives.
    messages.push(json!({
        "role": "user",
        "content": "Stop searching. Give your final answer now on a single line beginning with `ANSWER:`."
    }));
    let msg = chat(&messages)?;
    Ok(content_of(&msg))
}

fn content_of(msg: &Value) -> String {
    msg.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string()
}

/// Tool-call `function.arguments` is a JSON-encoded string per the OpenAI spec, but some servers
/// (incl. some LM Studio builds) return it as an already-parsed object. Accept both.
fn parse_tool_args(call: &Value) -> Value {
    match call.pointer("/function/arguments") {
        Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
        Some(v @ Value::Object(_)) => v.clone(),
        _ => json!({}),
    }
}

/// Execute one glossa tool in-process against the corpus in `work`, logging it to the trace
/// (same shape as the MCP server: search → array of {path,location,score}; read → {path}).
fn execute_tool(name: &str, args: &Value, idx: &glossa::index::store::DocIndex, trace: &TraceLog) -> String {
    crate::backend::glossa_tools::exec(name, args, idx, trace).0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn parse_tool_args_handles_string_and_object() {
        let s = json!({ "function": { "arguments": "{\"query\":\"abc\"}" } });
        assert_eq!(parse_tool_args(&s)["query"], "abc");
        let o = json!({ "function": { "arguments": { "query": "abc" } } });
        assert_eq!(parse_tool_args(&o)["query"], "abc");
        let bad = json!({ "function": { "arguments": "not json" } });
        assert_eq!(parse_tool_args(&bad), json!({}));
    }

    #[test]
    fn loop_returns_direct_answer_when_no_tool_calls() {
        let chat = |_: &[Value]| Ok(json!({ "role": "assistant", "content": "ANSWER: Bob" }));
        let exec = |_: &str, _: &Value| String::new();
        let out = run_agent_loop(chat, vec![], exec, 4).unwrap();
        assert_eq!(out, "ANSWER: Bob");
    }

    #[test]
    fn loop_dispatches_tool_then_answers() {
        let round = RefCell::new(0usize);
        let seen = RefCell::new(Vec::<(String, String)>::new());
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            if *r == 1 {
                // first turn requests a search
                Ok(json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "function": { "name": "search", "arguments": "{\"query\":\"corliss\"}" }
                    }]
                }))
            } else {
                // by now the tool result must be in the transcript
                let has_tool = msgs.iter().any(|m| m["role"] == "tool" && m["tool_call_id"] == "call_1");
                assert!(has_tool, "tool result not fed back: {msgs:?}");
                Ok(json!({ "role": "assistant", "content": "ANSWER: Chief of Protocol" }))
            }
        };
        let exec = |name: &str, args: &Value| {
            seen.borrow_mut().push((name.to_string(), args["query"].as_str().unwrap_or("").to_string()));
            "Meet_Corliss_Archer.md:p.1: ...  [9.0]".to_string()
        };
        let out = run_agent_loop(chat, vec![json!({"role":"user","content":"q"})], exec, 4).unwrap();
        assert_eq!(out, "ANSWER: Chief of Protocol");
        assert_eq!(seen.borrow().as_slice(), &[("search".to_string(), "corliss".to_string())]);
    }

    #[test]
    fn loop_stops_at_max_rounds() {
        // chat always asks for a tool; loop must terminate (max_rounds + 1 final call) not hang.
        let calls = RefCell::new(0usize);
        let chat = |_: &[Value]| {
            *calls.borrow_mut() += 1;
            Ok(json!({
                "role": "assistant", "content": "giving up",
                "tool_calls": [{ "id": "c", "function": { "name": "search", "arguments": "{\"query\":\"x\"}" } }]
            }))
        };
        let exec = |_: &str, _: &Value| "hit".to_string();
        let out = run_agent_loop(chat, vec![], exec, 3).unwrap();
        assert_eq!(out, "giving up");
        assert_eq!(*calls.borrow(), 4); // 3 rounds + 1 final
    }
}
