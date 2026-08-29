//! `ChatTransport`: a neutral seam between the agent loop and a concrete chat API
//! (OpenAI-compatible `/v1/chat/completions`, Anthropic `/v1/messages`, …).
//!
//! Phase 1 (this module) only introduces the trait, the neutral reply/tool-call types, and the
//! shared HTTP plumbing (`runtime`/`http_client`) moved here from `openai.rs` so both transport
//! impls can share one process-wide tokio runtime + reqwest client. No behavior changes yet —
//! `openai.rs` keeps its own request/response shapes; a later task wraps them in an
//! `OpenAiTransport` impl of this trait.

use std::sync::OnceLock;

pub mod anthropic;
pub mod openai;
pub mod responses;

/// One tool call the model asked to run, in a shape neutral to the wire format that produced it
/// (OpenAI's `tool_calls[].function`, Anthropic's `content[].type=="tool_use"`, …).
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}

/// One assistant turn, normalized away from the provider's wire shape: `text` is the answer (or
/// `None` when the turn is tool-calls-only), `tool_calls` are the neutral calls to dispatch, and
/// `raw` is the provider's own response payload verbatim — kept so a transport's
/// `push_assistant_turn` can echo back provider-specific fields (e.g. OpenAI's
/// `reasoning_content`, Anthropic's `thinking` blocks) that a typed round-trip would drop.
#[derive(Debug, Clone)]
pub struct TurnReply {
    pub text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub raw: serde_json::Value,
}

/// Abstracts one chat API's request/response shape behind the neutral types above, so the agent
/// loop (`run_agent_loop`) can drive OpenAI-compatible and Anthropic-compatible endpoints
/// identically. Object-safe by construction (no generic methods, no `Self` by value) — the design
/// is meant to be driven through `&dyn ChatTransport`.
pub trait ChatTransport {
    /// Render the tool registry into this provider's tool-schema shape (OpenAI's
    /// `{type:function,function:{...}}` envelope vs. Anthropic's flat `{name,input_schema}`).
    /// `graph_on` gates the graph-only tools (glossary/reach/sql/…) the same way both arms of the
    /// eval do today.
    fn tools_schema(&self, graph_on: bool) -> serde_json::Value;

    /// One request/response round-trip: POST `messages` (+ optional `system`, `tools`) to `ep` at
    /// the given sampling `temperature`, and return the normalized `TurnReply`.
    fn call(
        &self,
        ep: &crate::lab::Endpoint,
        system: Option<&str>,
        messages: &[serde_json::Value],
        tools: Option<&serde_json::Value>,
        temperature: f64,
    ) -> anyhow::Result<TurnReply>;

    /// Append the assistant's turn to the running transcript in this provider's own message
    /// shape (from `reply.raw`), preserving any provider-specific fields the neutral `TurnReply`
    /// doesn't carry.
    fn push_assistant_turn(&self, messages: &mut Vec<serde_json::Value>, reply: &TurnReply);

    /// Append the results of one or more executed tool calls, in this provider's own message
    /// shape (OpenAI: one `{role:"tool"}` message per result; Anthropic: one `{role:"user"}`
    /// message batching all `tool_result` blocks). `results` is `(tool_call_id, body)` pairs.
    fn push_tool_results(&self, messages: &mut Vec<serde_json::Value>, results: &[(String, String)]);
}

/// Shared tokio runtime backing the sync-over-async HTTP bridge: every transport's `call` is
/// driven from the (deliberately synchronous) agent loop, but `reqwest`'s client is async-only.
/// One process-wide multi-thread runtime lets every call just `block_on` instead of threading an
/// executor through the whole sync call stack. Moved here (was `openai.rs`) so both the OpenAI
/// and Anthropic transports share the same runtime/connection pool instead of each growing their
/// own.
pub(crate) fn runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("failed to build the tokio runtime backing the http chat bridge")
    })
}

/// Shared reqwest client (connection-pool + TLS-session reuse across chat calls). The per-request
/// timeout is set on the `RequestBuilder`, so one client serves every endpoint even when their
/// `timeout_secs` differ.
pub(crate) fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(reqwest::Client::new)
}

/// Select the `ChatTransport` for an [`crate::lab::ApiKind`]. All three kinds now have their own
/// transport (`OpenAiChat`, `Anthropic`, `OpenAiResponses`).
pub fn transport_for(api: crate::lab::ApiKind) -> Box<dyn ChatTransport> {
    use crate::lab::ApiKind;
    match api {
        ApiKind::OpenAiChat => Box::new(openai::OpenAiTransport),
        ApiKind::Anthropic => Box::new(anthropic::AnthropicTransport),
        ApiKind::OpenAiResponses => Box::new(responses::ResponsesTransport),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn turn_reply_field_access() {
        let reply = TurnReply {
            text: Some("x".into()),
            tool_calls: vec![ToolCall {
                id: "t".into(),
                name: "search".into(),
                args: json!({"q": "y"}),
            }],
            raw: json!({}),
        };
        assert_eq!(reply.text.as_deref(), Some("x"));
        assert_eq!(reply.tool_calls[0].name, "search");
        assert_eq!(reply.tool_calls[0].args, json!({"q": "y"}));
    }

    /// Object-safety check: a function taking `&dyn ChatTransport` must compile. If a method
    /// signature on the trait made it non-object-safe, this function would fail to compile.
    #[allow(dead_code)]
    fn _accepts(_: &dyn ChatTransport) {}

    /// The moved runtime/http_client helpers are reachable and usable from this module.
    #[test]
    fn runtime_and_http_client_are_reachable() {
        let _ = runtime();
        let _ = http_client();
    }

    /// Phase 1: `transport_for` must return a usable transport for every `ApiKind`, including
    /// the not-yet-built `Anthropic`/`OpenAiResponses` kinds (stubbed onto OpenAI chat rather
    /// than panicking/`todo!`-ing).
    #[test]
    fn transport_for_covers_every_api_kind() {
        for api in [
            crate::lab::ApiKind::OpenAiChat,
            crate::lab::ApiKind::Anthropic,
            crate::lab::ApiKind::OpenAiResponses,
        ] {
            let _: Box<dyn ChatTransport> = transport_for(api);
        }
    }
}
