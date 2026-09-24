use serde_json::{json, Value};

/// Echo + exec args for one TZ tool_call block.
///
/// OpenRouter providers reject the *next* turn if any prior assistant
/// `function.arguments` is not valid JSON. When TZ hands back `arguments: null` +
/// broken `raw_arguments`, we heal via `jsonrepair` when possible; only
/// unrepairable junk collapses to `{}`.
struct NormalizedToolArgs {
    /// Value written into the echoed assistant `tool_call.arguments` (object or a
    /// string that itself parses as a JSON object).
    echo: Value,
    /// Object passed to `exec` (parsed from a JSON-string echo when needed).
    exec: Value,
}

fn normalize_tool_call_arguments(call: &Value) -> NormalizedToolArgs {
    let empty = || NormalizedToolArgs {
        echo: json!({}),
        exec: json!({}),
    };

    let from_object = |obj: Value| NormalizedToolArgs {
        echo: obj.clone(),
        exec: obj,
    };

    let from_string = |s: &str, original: &Value| match parse_tool_args_object(s) {
        Some(obj) => {
            // Prefer echoing the original string when it already parsed cleanly;
            // after healing, echo the repaired object (valid for providers).
            let echo = if serde_json::from_str::<Value>(s)
                .ok()
                .is_some_and(|v| v.is_object())
            {
                original.clone()
            } else {
                obj.clone()
            };
            NormalizedToolArgs { echo, exec: obj }
        }
        None => empty(),
    };

    // Prefer parsed `arguments` when TZ succeeded.
    if let Some(args) = call.get("arguments").filter(|v| !v.is_null()) {
        return match args {
            Value::Object(_) => from_object(args.clone()),
            Value::String(s) => from_string(s, args),
            _ => empty(),
        };
    }

    // `arguments: null` — fall back to raw_arguments (string or already-parsed).
    match call.get("raw_arguments") {
        None => empty(),
        Some(raw) => match raw {
            Value::Object(_) => from_object(raw.clone()),
            Value::String(s) => from_string(s, raw),
            _ => empty(),
        },
    }
}

/// Parse tool-call args as a JSON object, healing via [`jsonrepair`] when needed.
fn parse_tool_args_object(raw: &str) -> Option<Value> {
    let cleaned = preprocess_tool_args_json(raw);
    if let Ok(obj @ Value::Object(_)) = serde_json::from_str::<Value>(&cleaned) {
        return Some(obj);
    }
    match jsonrepair::loads(&cleaned, &jsonrepair::Options::default()) {
        Ok(obj @ Value::Object(_)) => Some(obj),
        _ => None,
    }
}

/// Strip non-JSON wrappers LLMs sometimes leave inside `function.arguments`.
fn preprocess_tool_args_json(s: &str) -> String {
    let mut t = s.trim().to_string();
    for tag in ["</tool_call>", "<tool_call>", "</function>", "<function>"] {
        t = t.replace(tag, "");
    }
    t = t.trim().to_string();
    if let Some(rest) = t.strip_prefix("```json") {
        t = rest.to_string();
    } else if let Some(rest) = t.strip_prefix("```") {
        t = rest.to_string();
    }
    if let Some(rest) = t.strip_suffix("```") {
        t = rest.to_string();
    }
    t.trim().to_string()
}

/// Normalize a TensorZero `/inference` content array into `ChatTransport`'s neutral `TurnReply`.
/// REUSES [`normalize_tool_call_arguments`] — the SAME healing is applied to every tool_call block
/// (repair-or-reject unparseable `arguments`/`raw_arguments`) — so `TzTransport::call` (the
/// native-transport seam) never echoes a poisoned `arguments: null` back into the transcript.
///
/// `raw` carries the NORMALIZED content array (healed tool_call `arguments`, every other block
/// untouched) under `{"content": [...]}`, so `TzTransport::push_assistant_turn` can echo it back
/// verbatim.
pub(crate) fn turn_reply_from_content(
    content: Vec<Value>,
    finish_reason: Option<String>,
) -> crate::backend::transport::TurnReply {
    use crate::backend::transport::ToolCall;

    let mut tool_calls = Vec::new();
    let mut normalized_blocks = Vec::with_capacity(content.len());
    for block in &content {
        let typ = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if typ == "tool_call" {
            let id = block
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let name = block
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let norm = normalize_tool_call_arguments(block);
            tool_calls.push(ToolCall {
                id: id.clone(),
                name: name.clone(),
                args: norm.exec,
            });
            normalized_blocks.push(json!({
                "type": "tool_call", "id": id, "name": name, "arguments": norm.echo
            }));
        } else {
            normalized_blocks.push(block.clone());
        }
    }
    let text: String = content
        .iter()
        .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
        .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
        .collect::<Vec<_>>()
        .join("");

    crate::backend::transport::TurnReply {
        text: Some(text),
        tool_calls,
        finish_reason,
        raw: json!({ "content": normalized_blocks }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_reply_from_content_extracts_text_and_tool_calls() {
        let content = vec![
            json!({ "type": "text", "text": "thinking..." }),
            json!({ "type": "tool_call", "id": "c1", "name": "search", "arguments": {"q": "x"} }),
        ];
        let reply = turn_reply_from_content(content, Some("stop".to_string()));
        assert_eq!(reply.text.as_deref(), Some("thinking..."));
        assert_eq!(reply.tool_calls.len(), 1);
        assert_eq!(reply.tool_calls[0].id, "c1");
        assert_eq!(reply.tool_calls[0].name, "search");
        assert_eq!(reply.tool_calls[0].args, json!({"q": "x"}));
        assert_eq!(reply.finish_reason.as_deref(), Some("stop"));
        assert_eq!(reply.raw["content"][1]["arguments"], json!({"q": "x"}));
    }

    /// A poisoned `arguments: null` + valid `raw_arguments` must be healed into the echoed
    /// `raw.content` block, through the `ChatTransport`-facing `turn_reply_from_content` seam.
    #[test]
    fn turn_reply_from_content_heals_null_arguments() {
        let content = vec![json!({
            "type": "tool_call", "id": "c1", "name": "read",
            "arguments": Value::Null,
            "raw_arguments": "{\"path\":\"a.md\"}"
        })];
        let reply = turn_reply_from_content(content, None);
        assert_eq!(reply.tool_calls[0].args, json!({"path": "a.md"}));
        assert_eq!(
            reply.raw["content"][0]["arguments"],
            json!("{\"path\":\"a.md\"}")
        );
    }

    #[test]
    fn normalize_accepts_object_and_valid_raw_string() {
        let obj = normalize_tool_call_arguments(&json!({
            "arguments": {"query": "x"}
        }));
        assert_eq!(obj.exec["query"], "x");

        let raw_ok = normalize_tool_call_arguments(&json!({
            "arguments": Value::Null,
            "raw_arguments": "{\"path\":\"a.md\",\"n\":1}"
        }));
        assert_eq!(raw_ok.echo, json!("{\"path\":\"a.md\",\"n\":1}"));
        assert_eq!(raw_ok.exec["path"], "a.md");

        // Truncated JSON is healed by jsonrepair — exec gets a closed object.
        let raw_trunc = normalize_tool_call_arguments(&json!({
            "arguments": Value::Null,
            "raw_arguments": "{\"path\":\"a.md\",\"n\":1"
        }));
        assert_eq!(raw_trunc.exec["path"], "a.md");
        assert!(raw_trunc.echo.is_object(), "healed args echoed as object");

        // Non-object JSON collapses to an empty object (tool args must be an object).
        let raw_bad = normalize_tool_call_arguments(&json!({
            "arguments": Value::Null,
            "raw_arguments": "[1, 2, 3]"
        }));
        assert_eq!(raw_bad.echo, json!({}));
        assert_eq!(raw_bad.exec, json!({}));
    }
}
