//! Generic tool-calling agent loop, driven through the neutral `ChatTransport` seam instead of a
//! raw-`Value`-shaped OpenAI closure. This is the SAME dedup/streak/NBA/round-cap substrate that
//! used to live inline in `backend::openai::run_agent_loop` (see that module's history) — moved
//! here VERBATIM (Task 3 of the multi-api-transport plan) so both `OpenAiTransport` and (later)
//! `AnthropicTransport` drive one implementation instead of each growing their own copy.
//!
//! `backend::openai::run_agent_loop` (the pre-existing `Fn(&[Value]) -> Result<Value>` calling
//! convention) is kept as a THIN backward-compatible shim over this function — its three
//! not-yet-migrated callers (`build::extract`, `distil::chain`, `gepa_graph`; migrating them to
//! `ChatTransport` directly is Phase 2 Task 6 of the plan) keep compiling unchanged. The dedup/
//! streak/NBA logic itself lives in exactly ONE place: here.

use crate::backend::transport::{transport_for, ChatTransport, TurnReply};
use crate::backend::user_sim::DialogueGate;
use crate::lab::Endpoint;
use serde_json::{json, Value};
use std::collections::HashSet;

/// One `transport.call`, wrapped in the resilience layer: throttle + fallback chain per the
/// endpoint's opt-in `rate_limit`/`fallback`. For the PRIMARY link the already-built `transport` is
/// reused verbatim (so the injected mock transport in tests, and today's real transport, drive the
/// primary exactly as before); only fallback links build their own transport via `transport_for`.
/// With `ep.fallback` empty and `ep.rate_limit` absent this is a single un-throttled `transport.call`
/// — byte-identical to the pre-Phase-3 code path.
fn resilient_call(
    transport: &dyn ChatTransport,
    ep: &Endpoint,
    system: Option<&str>,
    messages: &[Value],
    tools: Option<&Value>,
    temperature: Option<f64>,
) -> anyhow::Result<TurnReply> {
    crate::backend::resilience::call_resilient(ep, |link| {
        if std::ptr::eq(link, ep) {
            transport.call(link, system, messages, tools, temperature)
        } else {
            transport_for(link.api).call(link, system, messages, tools, temperature)
        }
    })
}

/// Unproductive-streak threshold: this many consecutive REAL (non-deduped) tool calls in a row
/// that each surface zero new identifiers trips the steer. Named so the TDD tests and the loop
/// agree on one number instead of a magic literal in two places. `pub(crate)` so callers outside
/// this module (e.g. `build::extract`'s regression test for the graph_upsert ids fix, and
/// `openai.rs`'s re-export for its own not-moved tests) can size their fixtures off the real
/// threshold instead of duplicating the literal.
pub(crate) const UNPRODUCTIVE_STREAK_K: usize = 3;

/// Drive a tool-calling chat to a final textual answer, generic over any `ChatTransport`.
///
/// `transport.call(ep, system, &messages, tools, temperature)` returns the normalized
/// `TurnReply`. When it carries `tool_calls`, each is dispatched through `exec(name, args)` and
/// the result fed back via `transport.push_tool_results`, then the model is queried again — up to
/// `max_rounds`. The first reply with no tool calls yields `reply.text` — UNLESS `user_sim` is
/// `Some`, in which case that text-only turn is first shown to a patient simulated-user gate (see
/// [`crate::backend::user_sim`]): a mere restated-question / thinking-out-loud turn is deflected
/// back into the loop as a `role:"user"` message and the loop continues, while a substantive answer
/// (or a gate error — fail-open) is returned. `None` reproduces the pre-gate behavior exactly.
///
/// Two independent stuck detectors sit on top of `exec` (ported verbatim from the old inline
/// loop, now operating on the neutral `ToolCall` shape instead of raw `Value` pointers):
/// - **Identical-repeat dedup**: the previous (tool, args) actually executed. When the model
///   re-issues the SAME call, it isn't re-run (identical result) — `on_repeat` (the next-best-action)
///   fires instead. This takes priority and doesn't touch the streak below (a dedup hit didn't
///   execute, so it can't be "unproductive").
/// - **Unproductive streak**: the model issues many DIFFERENT calls (varied tool/args — so they all
///   really execute) that each surface no NEW identifier — a search-flood that never progresses.
///   `exec` returns `(body, ids)`; `ids` are what a session-aware MCP server would track per call
///   (search hit locations, a read's path, …). Any id not already in `seen` resets the streak;
///   otherwise it grows. At `UNPRODUCTIVE_STREAK_K` the fed-back tool content becomes the steer
///   (`glossa_tools::unproductive_steer`) instead of the (already-seen) body, and the counter resets
///   so it fires once per streak, not on every call past the threshold.
///
/// Temperature: resolved ONCE at the top of the loop via [`Endpoint::resolve_temperature`]
/// (`KB_EVAL_TEMP` env override > `ep.temperature` > `None`) and passed to every `transport.call`.
/// `None` means the request OMITS `temperature` so the provider/model uses its own default — there
/// is no hardcoded fallback. `KB_EVAL_TEMP` is preserved as the override, so build/distil (which
/// set it before the worker pool starts) keep steering the reader's sampling exactly as before.
/// The `ChatTransport::call` signature takes `temperature` as an explicit argument rather than
/// reading the env itself, so the resolution happens at the call site — here, not in a transport.
pub fn run_agent_loop(
    transport: &dyn ChatTransport,
    ep: &Endpoint,
    system: Option<&str>,
    mut messages: Vec<Value>,
    tools: Option<&Value>,
    mut exec: impl FnMut(&str, &Value) -> (String, Vec<String>),
    on_repeat: impl Fn(&str, &Value) -> String,
    max_rounds: usize,
    user_sim: Option<&dyn DialogueGate>,
) -> anyhow::Result<String> {
    // This call is the start of a fresh CONVERSATION (one reason seed / one build doc / one eval
    // case / …), possibly on a dedicated worker thread under `kbx --jobs N`: reset THIS thread's
    // conversation-prefix tracker so the self-computed cache estimate (see
    // `backend::openai::usage_split_with_prefix`/`PREV_PROMPT_TOKENS`) never carries the PREVIOUS
    // conversation's tail `prompt_tokens` into this one's first request. Both entry points funnel
    // here (`OpenAiBackend::answer` drives this loop directly; the closure-based
    // `openai::run_agent_loop` shim delegates to it), so this one reset covers every caller.
    crate::backend::openai::reset_conversation_prefix();

    let temperature: Option<f64> = ep.resolve_temperature();

    // The original question, for the optional simulated-user gate (see the `user_sim` handling
    // below). Captured ONCE from the first `role:"user"` message in the seed transcript — both
    // entry points seed `[system, user]`, so this is the reader's question. Empty when there is no
    // user message (e.g. the shim's own unit tests, which never configure a gate anyway).
    let question: String = messages
        .iter()
        .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
        .and_then(|m| m.get("content").and_then(Value::as_str))
        .unwrap_or("")
        .to_string();

    // Stuck-detection substrate: the previous (tool, args) actually executed. When the model
    // re-issues the SAME call, we don't re-run it (identical result) — we hand off to `on_repeat`,
    // the next-best-action. Default callers pass `repeat_nudge`; the reader path passes a fan-out.
    let mut last_key: Option<String> = None;
    // Novelty tracking for the unproductive-streak detector (see the doc comment above).
    let mut seen: HashSet<String> = HashSet::new();
    let mut unproductive: usize = 0;

    for _ in 0..max_rounds {
        let reply: TurnReply = resilient_call(transport, ep, system, &messages, tools, temperature)?;
        if reply.tool_calls.is_empty() {
            let text = reply.text.clone().unwrap_or_default();
            match user_sim {
                // No gate configured -> today's behavior EXACTLY: the first text-only turn is the
                // final answer.
                None => return Ok(text),
                Some(gate) => match gate.judge(&question, &messages, &text) {
                    // Substantive answer (or the gate failed open) -> accept and return it.
                    Ok(None) => return Ok(text),
                    // The assistant only kept asking: echo its turn, append the in-character user
                    // deflection as a `role:"user"` message, and continue. Each deflection consumes
                    // a round, so this is naturally capped by `max_rounds`.
                    Ok(Some(deflection)) => {
                        transport.push_assistant_turn(&mut messages, &reply);
                        messages.push(json!({ "role": "user", "content": deflection }));
                        continue;
                    }
                    // Fail-open on a gate error: return the text rather than hang the run.
                    Err(_) => return Ok(text),
                },
            }
        }
        // Echo the assistant turn that requested the tools (mirrors the old `messages.push(msg.clone())`).
        transport.push_assistant_turn(&mut messages, &reply);

        let mut results: Vec<(String, String)> = Vec::with_capacity(reply.tool_calls.len());
        for call in &reply.tool_calls {
            let key = format!(
                "{}\u{1}{}",
                call.name,
                serde_json::to_string(&call.args).unwrap_or_default()
            );
            let result = if last_key.as_deref() == Some(key.as_str()) {
                on_repeat(&call.name, &call.args)
            } else {
                let (body, ids) = exec(&call.name, &call.args);
                last_key = Some(key);
                let has_new = ids.into_iter().fold(false, |acc, i| seen.insert(i) || acc);
                if has_new {
                    unproductive = 0;
                    body
                } else {
                    unproductive += 1;
                    if unproductive >= UNPRODUCTIVE_STREAK_K {
                        unproductive = 0;
                        crate::backend::glossa_tools::unproductive_steer(&call.name)
                    } else {
                        body
                    }
                }
            };
            results.push((call.id.clone(), result));
        }
        transport.push_tool_results(&mut messages, &results);
    }
    // Out of rounds: nudge for a final answer (the model often keeps requesting tools otherwise)
    // and take whatever text it gives.
    messages.push(json!({
        "role": "user",
        "content": "Stop searching. Give your final answer now on a single line beginning with `ANSWER:`."
    }));
    let reply = resilient_call(transport, ep, system, &messages, tools, temperature)?;
    Ok(reply.text.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::transport::ToolCall;
    use std::cell::RefCell;
    use std::collections::VecDeque;

    /// Default `on_repeat` for tests that don't exercise NBA: a static nudge.
    fn nudge(name: &str, _args: &Value) -> String {
        format!("(dup {name}) you already called this — try a different tool or change the query")
    }

    fn test_endpoint() -> Endpoint {
        Endpoint {
            endpoint: String::new(),
            model: String::new(),
            api_key: String::new(),
            api_key_env: String::new(),
            timeout_secs: 30,
            api: crate::lab::ApiKind::default(),
            temperature: None,
            rate_limit: None,
            fallback: Vec::new(),
        }
    }

    /// A scripted `ChatTransport`: replies are consumed in order, one per `call`. Records the
    /// transcript `call` saw at each round (a snapshot of `messages`) so tests can assert on fed-
    /// back tool content exactly like the old closure-based tests inspected `msgs` directly.
    /// `push_assistant_turn`/`push_tool_results` mirror `OpenAiTransport`'s shapes (the loop's
    /// dedup/streak logic is transport-agnostic, but the message shapes asserted on below are the
    /// OpenAI ones the old tests checked).
    struct MockTransport {
        replies: RefCell<VecDeque<TurnReply>>,
        calls: RefCell<Vec<Vec<Value>>>,
    }

    impl MockTransport {
        fn new(replies: Vec<TurnReply>) -> Self {
            Self {
                replies: RefCell::new(replies.into_iter().collect()),
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl ChatTransport for MockTransport {
        fn tools_schema(&self, _graph_on: bool) -> Value {
            json!([])
        }

        fn call(
            &self,
            _ep: &Endpoint,
            _system: Option<&str>,
            messages: &[Value],
            _tools: Option<&Value>,
            _temperature: Option<f64>,
        ) -> anyhow::Result<TurnReply> {
            self.calls.borrow_mut().push(messages.to_vec());
            self.replies
                .borrow_mut()
                .pop_front()
                .ok_or_else(|| anyhow::anyhow!("MockTransport: scripted replies exhausted"))
        }

        fn push_assistant_turn(&self, messages: &mut Vec<Value>, reply: &TurnReply) {
            messages.push(reply.raw.clone());
        }

        fn push_tool_results(&self, messages: &mut Vec<Value>, results: &[(String, String)]) {
            for (id, body) in results {
                messages.push(json!({ "role": "tool", "tool_call_id": id, "content": body }));
            }
        }
    }

    fn reply_text(s: &str) -> TurnReply {
        TurnReply {
            text: Some(s.to_string()),
            tool_calls: vec![],
            raw: json!({ "role": "assistant", "content": s }),
        }
    }

    fn reply_tool_call(id: &str, name: &str, args: Value) -> TurnReply {
        TurnReply {
            text: None,
            tool_calls: vec![ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                args,
            }],
            raw: json!({ "role": "assistant", "content": null }),
        }
    }

    /// Like `reply_tool_call`, but also carries `text` — mirrors the old mocked `chat` closures
    /// that returned a message with BOTH a `content` string AND `tool_calls` on every turn (some
    /// real models do this too); the loop must still dispatch the tool_calls and only surface
    /// `text` on the round where it terminates (an empty-tool_calls reply, or the trailing
    /// max-rounds "give up" call, which reads `reply.text` unconditionally).
    fn reply_tool_call_with_text(id: &str, name: &str, args: Value, text: &str) -> TurnReply {
        TurnReply {
            text: Some(text.to_string()),
            tool_calls: vec![ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                args,
            }],
            raw: json!({ "role": "assistant", "content": text }),
        }
    }

    #[test]
    fn loop_returns_direct_answer_when_no_tool_calls() {
        let ep = test_endpoint();
        let transport = MockTransport::new(vec![reply_text("ANSWER: Bob")]);
        let exec = |_: &str, _: &Value| (String::new(), Vec::new());
        let out = run_agent_loop(&transport, &ep, None, vec![], None, exec, nudge, 4, None).unwrap();
        assert_eq!(out, "ANSWER: Bob");
    }

    #[test]
    fn loop_dispatches_tool_then_answers() {
        let ep = test_endpoint();
        let transport = MockTransport::new(vec![
            reply_tool_call("call_1", "search", json!({"query": "corliss"})),
            reply_text("ANSWER: Chief of Protocol"),
        ]);
        let seen = RefCell::new(Vec::<(String, String)>::new());
        let exec = |name: &str, args: &Value| {
            seen.borrow_mut().push((
                name.to_string(),
                args["query"].as_str().unwrap_or("").to_string(),
            ));
            (
                "Meet_Corliss_Archer.md:p.1: ...  [9.0]".to_string(),
                vec!["Meet_Corliss_Archer.md:p.1".to_string()],
            )
        };
        let out = run_agent_loop(
            &transport,
            &ep,
            None,
            vec![json!({"role":"user","content":"q"})],
            None,
            exec,
            nudge,
            4,
            None,
        )
        .unwrap();
        assert_eq!(out, "ANSWER: Chief of Protocol");
        assert_eq!(
            seen.borrow().as_slice(),
            &[("search".to_string(), "corliss".to_string())]
        );
        // By the second call, the tool result from round 1 must be in the transcript.
        let second_call_msgs = &transport.calls.borrow()[1];
        let has_tool = second_call_msgs
            .iter()
            .any(|m| m["role"] == "tool" && m["tool_call_id"] == "call_1");
        assert!(has_tool, "tool result not fed back: {second_call_msgs:?}");
    }

    #[test]
    fn loop_dedupes_consecutive_identical_tool_calls() {
        // The model thrashes: same tool, same args, every round. The loop must execute it once,
        // then feed back a nudge (not another live result) so the model is pushed to switch.
        let ep = test_endpoint();
        // 5 rounds inside the loop (max_rounds=5) + 1 trailing "give up" call = 6 scripted replies.
        // Every reply carries BOTH `content:"looping"` and a tool_call (mirrors the old mocked
        // `chat` closure, which always returned both), so the loop dispatches tool_calls on every
        // round and the trailing "give up" call's `text` ("looping") is what's finally returned.
        let replies: Vec<TurnReply> = (0..6)
            .map(|_| reply_tool_call_with_text("c", "glossary", json!({"name": "X"}), "looping"))
            .collect();
        let transport = MockTransport::new(replies);
        let execs = RefCell::new(0usize);
        let exec = |_: &str, _: &Value| {
            *execs.borrow_mut() += 1;
            ("hit".to_string(), vec!["hit-id".to_string()])
        };
        let out = run_agent_loop(&transport, &ep, None, vec![], None, exec, nudge, 5, None).unwrap();
        assert_eq!(out, "looping");
        assert_eq!(*execs.borrow(), 1, "identical consecutive calls must execute only once");
        // From the 3rd call onward, the most recent tool result in the transcript must be a dedup
        // nudge (round 1 executes live; round 2's identical call is deduped and its nudge lands in
        // the transcript for round 3's call onward).
        for (i, msgs) in transport.calls.borrow().iter().enumerate() {
            if i < 2 {
                continue;
            }
            let last_tool = msgs.iter().rev().find(|m| m["role"] == "tool");
            let c = last_tool.and_then(|m| m["content"].as_str()).unwrap_or("");
            let lc = c.to_lowercase();
            assert!(
                lc.contains("different tool") || lc.contains("already"),
                "call {i} expected a dedup nudge, got: {c:?}"
            );
        }
    }

    #[test]
    fn loop_reexecutes_when_args_differ() {
        // A different tool OR different args is NOT a dedup — it must run.
        let ep = test_endpoint();
        // 4 rounds (max_rounds=4) + 1 trailing "give up" call = 5 scripted replies, alternating
        // tool name so no two consecutive calls share a dedup key.
        let replies: Vec<TurnReply> = (0..5)
            .map(|i| {
                let name = if i % 2 == 0 { "A" } else { "B" };
                reply_tool_call("c", name, json!({"name": "X"}))
            })
            .collect();
        let transport = MockTransport::new(replies);
        let execs = RefCell::new(0usize);
        // Novelty tracking isn't under test here; empty ids keep it a no-op for the streak detector.
        let exec = |_: &str, _: &Value| {
            *execs.borrow_mut() += 1;
            ("hit".to_string(), Vec::new())
        };
        let _ = run_agent_loop(&transport, &ep, None, vec![], None, exec, nudge, 4, None).unwrap();
        assert_eq!(*execs.borrow(), 4, "alternating tools must each execute");
    }

    #[test]
    fn loop_stops_at_max_rounds() {
        // chat always asks for a tool; loop must terminate (max_rounds + 1 final call) not hang.
        let ep = test_endpoint();
        // 3 rounds (max_rounds=3) + 1 trailing "give up" call = 4 scripted replies, each carrying
        // BOTH `content:"giving up"` and a tool_call (mirrors the old mocked `chat` closure).
        let replies: Vec<TurnReply> = (0..4)
            .map(|_| reply_tool_call_with_text("c", "search", json!({"query": "x"}), "giving up"))
            .collect();
        let transport = MockTransport::new(replies);
        let exec = |_: &str, _: &Value| ("hit".to_string(), Vec::new());
        let out = run_agent_loop(&transport, &ep, None, vec![], None, exec, nudge, 3, None).unwrap();
        // The trailing "give up" call's reply.text is what's returned regardless of it also
        // carrying tool_calls (the loop reads `text` unconditionally on the final call).
        assert_eq!(out, "giving up");
        assert_eq!(transport.calls.borrow().len(), 4); // 3 rounds + 1 final
    }

    #[test]
    fn loop_unproductive_streak_feeds_steer_after_k_plus_one_calls() {
        // K+1 tool calls with VARIED args (so none of them dedup — each really executes), but exec
        // keeps surfacing the SAME already-seen id: an over-search spiral (many different probes,
        // nothing new). By the (K+1)th call, the fed-back tool content must be the steer.
        let ep = test_endpoint();
        let mut replies: Vec<TurnReply> = (0..UNPRODUCTIVE_STREAK_K + 1)
            .map(|i| reply_tool_call(&format!("c{i}"), "search", json!({"query": format!("query-{i}")})))
            .collect();
        replies.push(reply_text("ANSWER: done"));
        let transport = MockTransport::new(replies);
        let execs = RefCell::new(0usize);
        let exec = |_: &str, _: &Value| {
            *execs.borrow_mut() += 1;
            // Always the same id, regardless of the (varied) query -> never novel after the first.
            ("same old snippet".to_string(), vec!["doc.md:p.1".to_string()])
        };
        let out = run_agent_loop(
            &transport,
            &ep,
            None,
            vec![],
            None,
            exec,
            nudge,
            UNPRODUCTIVE_STREAK_K + 3,
            None,
        )
        .unwrap();
        assert_eq!(out, "ANSWER: done");
        assert_eq!(
            *execs.borrow(),
            UNPRODUCTIVE_STREAK_K + 1,
            "each varied call must actually execute (not deduped)"
        );
        // The last call (the K+2th) must have seen the steer as the most recent tool result.
        let last_call_msgs = transport.calls.borrow().last().unwrap().clone();
        let last_tool = last_call_msgs.iter().rev().find(|m| m["role"] == "tool");
        let c = last_tool.and_then(|m| m["content"].as_str()).unwrap_or("");
        let lc = c.to_lowercase();
        assert!(
            lc.contains("no new information") && lc.contains("change approach"),
            "expected the unproductive-streak steer, got: {c:?}"
        );
    }

    // --- simulated-user gate wiring -------------------------------------------------------------

    /// A deterministic `DialogueGate` (no network, no rand): each `judge` call pops the next
    /// scripted verdict, so a test can drive the loop's deflect/accept/error paths exactly. `Ok(Some
    /// (..))` = deflect (feed back a user reply), `Ok(None)` = accept, `Err(..)` = gate error.
    struct MockGate {
        verdicts: RefCell<VecDeque<anyhow::Result<Option<String>>>>,
        calls: RefCell<usize>,
    }

    impl MockGate {
        fn new(verdicts: Vec<anyhow::Result<Option<String>>>) -> Self {
            Self {
                verdicts: RefCell::new(verdicts.into_iter().collect()),
                calls: RefCell::new(0),
            }
        }
    }

    impl DialogueGate for MockGate {
        fn judge(
            &self,
            _question: &str,
            _messages: &[Value],
            _proposed: &str,
        ) -> anyhow::Result<Option<String>> {
            *self.calls.borrow_mut() += 1;
            self.verdicts
                .borrow_mut()
                .pop_front()
                .unwrap_or(Ok(None)) // exhausted -> accept, so a test can't hang the loop
        }
    }

    /// A non-answer text-only turn: the gate deflects once, the loop appends a `role:"user"` message
    /// and CONTINUES (a second `transport.call`), then the next turn is accepted.
    #[test]
    fn user_sim_deflects_non_answer_then_continues() {
        let ep = test_endpoint();
        let transport = MockTransport::new(vec![
            reply_text("QUESTION: who is Bob?"), // restated question -> deflected
            reply_text("ANSWER: Bob Page"),      // real answer -> accepted
        ]);
        let gate = MockGate::new(vec![
            Ok(Some("I don't know, that's what I was hoping you'd tell me.".to_string())),
            Ok(None),
        ]);
        let exec = |_: &str, _: &Value| (String::new(), Vec::new());
        let out = run_agent_loop(
            &transport,
            &ep,
            None,
            vec![json!({"role":"user","content":"who is Bob?"})],
            None,
            exec,
            nudge,
            4,
            Some(&gate),
        )
        .unwrap();
        assert_eq!(out, "ANSWER: Bob Page");
        // The loop must have made a SECOND call (didn't return the first text-only turn early)...
        assert_eq!(transport.calls.borrow().len(), 2, "gate deflection must re-enter the loop");
        // ...and the deflection must be the last message the 2nd call saw, as a `role:"user"` turn.
        let second = &transport.calls.borrow()[1];
        let last_user = second.iter().rev().find(|m| m["role"] == "user");
        assert_eq!(
            last_user.and_then(|m| m["content"].as_str()),
            Some("I don't know, that's what I was hoping you'd tell me.")
        );
        assert_eq!(*gate.calls.borrow(), 2);
    }

    /// The gate signals DONE (`Ok(None)`) on the first text-only turn -> the loop returns that text,
    /// no extra round.
    #[test]
    fn user_sim_accepts_substantive_answer_immediately() {
        let ep = test_endpoint();
        let transport = MockTransport::new(vec![reply_text("ANSWER: Chief of Protocol")]);
        let gate = MockGate::new(vec![Ok(None)]);
        let exec = |_: &str, _: &Value| (String::new(), Vec::new());
        let out = run_agent_loop(
            &transport,
            &ep,
            None,
            vec![json!({"role":"user","content":"q"})],
            None,
            exec,
            nudge,
            4,
            Some(&gate),
        )
        .unwrap();
        assert_eq!(out, "ANSWER: Chief of Protocol");
        assert_eq!(transport.calls.borrow().len(), 1, "a DONE verdict must not re-enter the loop");
        assert_eq!(*gate.calls.borrow(), 1);
    }

    /// `user_sim = None` -> non-regression: the first text-only turn is returned verbatim and the
    /// gate is never consulted (there is none).
    #[test]
    fn no_user_sim_returns_first_text_turn_unchanged() {
        let ep = test_endpoint();
        let transport = MockTransport::new(vec![reply_text("QUESTION: who is Bob?")]);
        let exec = |_: &str, _: &Value| (String::new(), Vec::new());
        let out = run_agent_loop(
            &transport,
            &ep,
            None,
            vec![json!({"role":"user","content":"who is Bob?"})],
            None,
            exec,
            nudge,
            4,
            None,
        )
        .unwrap();
        assert_eq!(out, "QUESTION: who is Bob?");
        assert_eq!(transport.calls.borrow().len(), 1);
    }

    /// A gate error fails OPEN: the loop returns the assistant's text rather than hanging or erroring.
    #[test]
    fn user_sim_fails_open_on_gate_error() {
        let ep = test_endpoint();
        let transport = MockTransport::new(vec![reply_text("QUESTION: still thinking")]);
        let gate = MockGate::new(vec![Err(anyhow::anyhow!("sim endpoint down"))]);
        let exec = |_: &str, _: &Value| (String::new(), Vec::new());
        let out = run_agent_loop(
            &transport,
            &ep,
            None,
            vec![json!({"role":"user","content":"q"})],
            None,
            exec,
            nudge,
            4,
            Some(&gate),
        )
        .unwrap();
        assert_eq!(out, "QUESTION: still thinking");
        assert_eq!(transport.calls.borrow().len(), 1, "fail-open must not re-enter the loop");
        assert_eq!(*gate.calls.borrow(), 1);
    }
}
