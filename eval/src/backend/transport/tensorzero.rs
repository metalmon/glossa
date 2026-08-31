//! `TzTransport`: native TensorZero API (`/inference` + `/feedback`) implementation of
//! `ChatTransport` — NOT the OpenAI-compatible shim TensorZero also exposes, and NOT any
//! TensorZero SDK crate. Gives the `kbx eval` reader full TZ observability: every question's
//! reader loop groups into ONE episode (via the thread-local [`crate::episode`] module, set from
//! the first `/inference` response and reused on every subsequent turn), and the judge verdict
//! posts as `/feedback` on that episode (see `run_eval` in `bin/kbx.rs`).
//!
//! REUSES the native-API mechanics `backend::tensorzero.rs` (the closure-driven `run_episode`
//! engine) already established: block→`TurnReply` translation via
//! [`crate::backend::tensorzero::turn_reply_from_content`] (which in turn reuses
//! `normalize_tool_call_arguments`'s healing), and the shared gateway helpers in `tz.rs`
//! (`crate::tz::infer`, `crate::tz::post_feedback`, `crate::tz::gateway_base`).

use super::{ChatTransport, TurnReply};
use crate::episode;
use crate::lab::Endpoint;
use anyhow::anyhow;
use serde_json::{json, Value};
use std::time::Duration;

/// Holds just enough state, captured from the `Endpoint` at CONSTRUCTION time, to post
/// `/feedback` later without an `Endpoint` parameter (the `ChatTransport::post_feedback` default
/// signature carries none — see that trait method's doc). `call()` itself still receives a fresh
/// `Endpoint` per request (the `function_name` it reads comes from there), so only the gateway
/// base URL + timeout need to survive between `call()` and `post_feedback()`.
pub struct TzTransport {
    gateway: String,
    timeout: Duration,
}

impl TzTransport {
    pub fn new(ep: &Endpoint) -> Self {
        TzTransport {
            gateway: ep.endpoint.clone(),
            timeout: Duration::from_secs(ep.timeout_secs),
        }
    }
}

impl ChatTransport for TzTransport {
    fn tools_schema(&self, graph_on: bool) -> Value {
        // TensorZero's function config owns tool wiring server-side (a variant's `tools` list in
        // `tensorzero.toml`), so this schema is never sent over `/inference` — `call()` below
        // ignores its `tools` argument entirely. It still needs SOME shape to hand back to the
        // agent loop (which advertises it via `Some(&tools)` regardless of transport), so this
        // reuses the OpenAI-shape schema for consistency with the other transports rather than
        // inventing a TZ-specific — but unused — envelope.
        super::openai::tools_schema(graph_on)
    }

    fn call(
        &self,
        ep: &Endpoint,
        system: Option<&str>,
        messages: &[Value],
        _tools: Option<&Value>,
        _temperature: Option<f64>,
    ) -> anyhow::Result<TurnReply> {
        // Sampling + tools are owned by the TZ function/variant config, not per-call — mirrors
        // `backend::tensorzero::TensorZeroBackend::answer`, which never sends either either.
        let function = ep.function_name.as_deref().ok_or_else(|| {
            anyhow!(
                "endpoint api=\"tensorzero\" requires `function_name` in lab.toml \
                 (e.g. `function_name = \"answer_hotpot\"` under `[model]`)"
            )
        })?;
        // Reuse whatever episode this thread's case is already grouped under (set below by an
        // earlier turn); the FIRST turn of a case has none yet, so a fresh backdated id (see
        // `crate::tz::backdated_episode_id` — the SAME client-generation pattern `backend::
        // tensorzero::TensorZeroBackend::answer` already uses, immune to Docker/WSL clock skew)
        // is minted here rather than leaving the field to the gateway — every turn, including
        // the first, always sends a well-formed UUIDv7, never an empty string.
        let episode_id = episode::current().unwrap_or_else(|| crate::tz::backdated_episode_id(30));
        let turn = crate::tz::infer(
            &self.gateway,
            function,
            &episode_id,
            messages,
            &json!({}),
            self.timeout,
            None,
            system,
            None,
        )?;
        // Adopt whatever the gateway echoed back (normally the SAME id just sent); fall back to
        // our own generated id if the response omitted it, so the thread-local is never left
        // unset after a successful call — the next turn (or `run_eval`'s feedback post) always
        // has an episode id to reuse.
        episode::set(if turn.episode_id.is_empty() {
            episode_id
        } else {
            turn.episode_id
        });
        Ok(crate::backend::tensorzero::turn_reply_from_content(
            turn.content,
            turn.finish_reason,
        ))
    }

    fn push_assistant_turn(&self, messages: &mut Vec<Value>, reply: &TurnReply) {
        // `reply.raw` is `{"content": [...]}` — the NORMALIZED block array `turn_reply_from_content`
        // built (healed tool_call `arguments` already echoed, every other block untouched).
        // Echoing it verbatim mirrors `run_episode_gated`'s
        // `messages.push(json!({ "role": "assistant", "content": merged }))`.
        let content = reply
            .raw
            .get("content")
            .cloned()
            .unwrap_or_else(|| json!([]));
        messages.push(json!({ "role": "assistant", "content": content }));
    }

    fn push_tool_results(&self, messages: &mut Vec<Value>, results: &[(String, String)]) {
        // One `{role:"user"}` message per result, each carrying a single `tool_result` block —
        // mirrors `run_episode_gated`'s per-call push loop exactly. `ChatTransport::
        // push_tool_results` doesn't carry the tool NAME (only `(id, body)` pairs), unlike the
        // legacy closure loop's `tool_result` blocks (`{type, id, name, result}`); `id` alone is
        // what correlates a result to its `tool_call` block, so this omits `name`.
        for (id, body) in results {
            messages.push(json!({
                "role": "user",
                "content": [{ "type": "tool_result", "id": id, "result": body }]
            }));
        }
    }

    fn post_feedback(&self, episode_id: &str, metrics: &[(&str, Value)]) {
        // Best-effort: `crate::tz::post_feedback` already swallows its own errors (never fails
        // the caller) — a feedback post must never zero out an already-scored episode.
        for (metric, value) in metrics {
            crate::tz::post_feedback(&self.gateway, episode_id, metric, value.clone(), &json!({}));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn test_endpoint(gateway: &str, function: Option<&str>) -> Endpoint {
        Endpoint {
            endpoint: gateway.to_string(),
            model: "m".to_string(),
            api_key: String::new(),
            api_key_env: String::new(),
            timeout_secs: 5,
            api: crate::lab::ApiKind::Tensorzero,
            temperature: None,
            rate_limit: None,
            fallback: Vec::new(),
            function_name: function.map(str::to_string),
            feedback_score_metric: None,
            feedback_bool_metric: None,
        }
    }

    /// Reads one full HTTP request off `sock` (headers + declared `Content-Length` body), looping
    /// past short TCP reads — mirrors the identical helper in `transport::anthropic`'s tests.
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

    /// `call()` must error clearly when `function_name` is absent — no network attempt at all.
    #[test]
    fn call_without_function_name_errors_clearly() {
        let transport = TzTransport::new(&test_endpoint("http://127.0.0.1:1", None));
        let ep = test_endpoint("http://127.0.0.1:1", None);
        episode::reset();
        let err = transport
            .call(
                &ep,
                None,
                &[json!({"role": "user", "content": "hi"})],
                None,
                None,
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("function_name"),
            "error must name the missing field: {err}"
        );
    }

    /// `call()` builds the `/inference` body reusing `crate::tz::infer`: `function_name`,
    /// `input.messages`, `input.system`, and `episode_id` carrying the THREAD-LOCAL when one is
    /// already set (so every turn of a case sends the SAME id and groups into one TZ episode).
    /// After a successful call, the gateway's own `episode_id` becomes the new thread-local.
    #[test]
    fn call_builds_inference_body_with_episode_id_when_set() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let req = read_full_http_request(&mut sock);
            let body = r#"{"content":[{"type":"text","text":"ANSWER: ok"}],"episode_id":"ep-abc","finish_reason":"stop"}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(), body
            );
            sock.write_all(resp.as_bytes()).unwrap();
            req
        });

        episode::reset();
        episode::set("ep-existing".to_string());
        let ep = test_endpoint(&format!("http://127.0.0.1:{port}"), Some("answer_hotpot"));
        let transport = TzTransport::new(&ep);
        let reply = transport
            .call(
                &ep,
                Some("you are a test system"),
                &[json!({"role": "user", "content": [{"type": "text", "text": "question"}]})],
                None,
                None,
            )
            .unwrap();
        assert_eq!(reply.text.as_deref(), Some("ANSWER: ok"));
        assert_eq!(reply.finish_reason.as_deref(), Some("stop"));
        // The gateway's response episode_id must become the new thread-local current().
        assert_eq!(episode::current().as_deref(), Some("ep-abc"));

        let req = server.join().unwrap();
        let body_str = req.split_once("\r\n\r\n").map_or("", |x| x.1);
        let body: Value = serde_json::from_str(body_str).expect("request body must be JSON");
        assert_eq!(body["function_name"], "answer_hotpot");
        assert!(body["input"]["messages"].is_array());
        assert_eq!(body["input"]["system"], "you are a test system");
        assert_eq!(
            body["episode_id"], "ep-existing",
            "the PRE-EXISTING thread-local episode id must be sent, so every turn of the case groups together"
        );

        episode::reset();
    }

    /// First turn of a case: no thread-local episode id has been set yet, so `call()` mints a
    /// fresh backdated UUIDv7 (via `crate::tz::backdated_episode_id`, the same client-generation
    /// pattern the legacy `TensorZeroBackend` uses) rather than sending an empty/absent field —
    /// the request's `episode_id` must be a well-formed id, and a mock gateway that doesn't echo
    /// one back (empty `episode_id` in the response) must not leave the thread-local unset.
    #[test]
    fn call_first_turn_mints_a_fresh_episode_id_and_falls_back_on_an_empty_response() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let req = read_full_http_request(&mut sock);
            // Deliberately omits `episode_id` from the response, to exercise the fallback path.
            let body = r#"{"content":[{"type":"text","text":"ANSWER: first"}]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(), body
            );
            sock.write_all(resp.as_bytes()).unwrap();
            req
        });

        episode::reset();
        let ep = test_endpoint(&format!("http://127.0.0.1:{port}"), Some("f"));
        let transport = TzTransport::new(&ep);
        transport
            .call(
                &ep,
                None,
                &[json!({"role": "user", "content": "hi"})],
                None,
                None,
            )
            .unwrap();

        let req = server.join().unwrap();
        let body_str = req.split_once("\r\n\r\n").map_or("", |x| x.1);
        let body: Value = serde_json::from_str(body_str).unwrap();
        let sent_id = body["episode_id"]
            .as_str()
            .expect("episode_id must be a string")
            .to_string();
        assert!(
            !sent_id.is_empty(),
            "first turn must send a real (non-empty) episode id"
        );
        let uuid = uuid::Uuid::parse_str(&sent_id).expect("must be a valid UUID");
        assert_eq!(
            uuid.get_version_num(),
            7,
            "must be the same UUIDv7 backdated_episode_id mints"
        );

        // The mock response carried no episode_id -> the thread-local falls back to the id we
        // generated and sent, rather than being left unset.
        assert_eq!(episode::current().as_deref(), Some(sent_id.as_str()));

        episode::reset();
    }

    /// `post_feedback` POSTs one `/feedback` body per metric — `{episode_id, metric_name, value}`.
    #[test]
    fn post_feedback_builds_one_body_per_metric() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let mut reqs = Vec::new();
            for _ in 0..2 {
                let (mut sock, _) = listener.accept().unwrap();
                let req = read_full_http_request(&mut sock);
                let resp = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";
                sock.write_all(resp.as_bytes()).unwrap();
                reqs.push(req);
            }
            reqs
        });

        let ep = test_endpoint(&format!("http://127.0.0.1:{port}"), Some("f"));
        let transport = TzTransport::new(&ep);
        transport.post_feedback(
            "ep-xyz",
            &[("judge", json!(0.5)), ("correct", json!(false))],
        );

        let reqs = server.join().unwrap();
        assert_eq!(reqs.len(), 2);
        for req in &reqs {
            assert!(req.starts_with("POST /feedback"), "wrong path: {req}");
        }
        let body0: Value =
            serde_json::from_str(reqs[0].split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(body0["episode_id"], "ep-xyz");
        assert_eq!(body0["metric_name"], "judge");
        assert_eq!(body0["value"], 0.5);
        let body1: Value =
            serde_json::from_str(reqs[1].split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(body1["metric_name"], "correct");
        assert_eq!(body1["value"], false);
    }

    /// `tools_schema` still returns a usable (OpenAI-shaped) schema even though `call()` never
    /// sends it — the agent loop always asks for one regardless of transport.
    #[test]
    fn tools_schema_returns_a_nonempty_schema() {
        let ep = test_endpoint("http://x", Some("f"));
        let transport = TzTransport::new(&ep);
        let schema = transport.tools_schema(true);
        assert!(schema.as_array().is_some_and(|a| !a.is_empty()));
    }

    /// `push_assistant_turn` echoes the normalized `{"content": [...]}` block array verbatim.
    #[test]
    fn push_assistant_turn_echoes_normalized_content() {
        let ep = test_endpoint("http://x", Some("f"));
        let transport = TzTransport::new(&ep);
        let reply = crate::backend::tensorzero::turn_reply_from_content(
            vec![
                json!({ "type": "tool_call", "id": "c1", "name": "search", "arguments": {"q": "x"} }),
            ],
            None,
        );
        let mut messages: Vec<Value> = vec![];
        transport.push_assistant_turn(&mut messages, &reply);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["content"][0]["type"], "tool_call");
        assert_eq!(messages[0]["content"][0]["id"], "c1");
    }

    /// `push_tool_results` appends one `{role:"user"}` message PER result (not batched), each
    /// carrying a single `tool_result` block.
    #[test]
    fn push_tool_results_appends_one_message_per_result() {
        let ep = test_endpoint("http://x", Some("f"));
        let transport = TzTransport::new(&ep);
        let mut messages: Vec<Value> = vec![];
        transport.push_tool_results(
            &mut messages,
            &[
                ("c1".to_string(), "result-1".to_string()),
                ("c2".to_string(), "result-2".to_string()),
            ],
        );
        assert_eq!(messages.len(), 2, "one message per result, not batched");
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"][0]["type"], "tool_result");
        assert_eq!(messages[0]["content"][0]["id"], "c1");
        assert_eq!(messages[0]["content"][0]["result"], "result-1");
        assert_eq!(messages[1]["content"][0]["id"], "c2");
    }
}
