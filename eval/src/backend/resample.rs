//! Provider-neutral resample layer: retries a chat turn whose OUTPUT is degenerate (hit the token
//! cap, spun into a repetition loop, or came back an empty no-tool turn) by sampling again, up to a
//! bounded budget. This is the "resample OUTER" half of the two-layer hardening the agent loop
//! applies to every `transport.call`:
//!
//! - [`crate::backend::resilience::call_resilient`] is the INNER layer — it handles a HARD failure
//!   of the request itself (transport drop, 429/5xx, an upstream-gateway error, a per-endpoint
//!   throttle, and the flat fallback chain). Its configs (`rate_limit`/`fallback`) are independent.
//! - [`call_with_resample`] is the OUTER layer here — the request SUCCEEDED (a `200` with a parsed
//!   reply), but the reply is a soft-degenerate OUTPUT worth resampling. Its budget
//!   ([`ResamplePolicy`]) is independent of the resilience config.
//!
//! So one logical model turn = `resample( resilient( transport.call ) )`: a network/rate-limit
//! failure is retried by resilience; a degenerate completion triggers a fresh sample. The predicate
//! and loop are transport-agnostic (they read the neutral [`TurnReply`]), so the reader
//! (`OpenAiTransport`), build/distil/reason (the closure shim), and any future Anthropic/Responses
//! stage all resample identically — this is what restored resampling to the reader after it was
//! lost driving the transport directly.
//!
//! DETERMINISM: the loop takes the transport as a `&dyn ChatTransport`, so the unit tests below
//! drive it through a scripted mock transport — no network, no `rand`, no real sleep.

use crate::backend::resilience::call_resilient;
use crate::backend::transport::{transport_for, ChatTransport, TurnReply};
use crate::lab::Endpoint;
use serde_json::Value;

/// How many times a single logical turn may resample a degenerate completion before giving up and
/// returning the last (best-effort) reply — sampling is stochastic, so a resample almost always
/// breaks a repetition loop / empty turn. The agent loop's own dedup/streak/max_rounds backstop
/// handles a model that keeps degenerating past this.
const GEN_LOOP_RETRIES: usize = 2;

/// Cap on how many of those resamples may be spent on `finish_reason == "length"` specifically,
/// before accepting the truncated best-effort completion. A chronically-verbose model that
/// overflows `max_tokens` every round would otherwise burn the full [`GEN_LOOP_RETRIES`] budget on
/// it (observed ~3x tokens wasted on an over-cap chat) — a length overrun is rarely fixed by one
/// stochastic resample, so it is capped tighter than the generic loop/empty bound. Overridable via
/// `KB_EVAL_MAX_LENGTH_RESAMPLE`.
const DEFAULT_MAX_LENGTH_RESAMPLE: usize = 1;

/// Independent budget for the OUTER resample layer (kept separate from the resilience layer's
/// `RetryPolicy`). `Default` reads `KB_EVAL_MAX_LENGTH_RESAMPLE` for the length sub-cap and uses
/// the historical constants otherwise, so the absent-config path reproduces the retired
/// `lmstudio_chat` resample behavior exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResamplePolicy {
    /// Total resamples allowed per turn (the loop runs `0..max_resamples`).
    pub max_resamples: usize,
    /// Of those, how many may be spent on a `finish_reason == "length"` turn.
    pub max_length_resamples: usize,
}

impl Default for ResamplePolicy {
    fn default() -> Self {
        Self {
            max_resamples: GEN_LOOP_RETRIES,
            max_length_resamples: std::env::var("KB_EVAL_MAX_LENGTH_RESAMPLE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_MAX_LENGTH_RESAMPLE),
        }
    }
}

/// Detect a degenerate generation loop: the tail of `text` is N identical consecutive blocks of some
/// short period (a phrase/line the model repeated until it ran out of tokens). Byte-level, so it also
/// catches whitespace-y repeats; returns false for normal prose. Pure — unit-tested. (Moved verbatim
/// from the retired `backend::openai::looks_looped`; the length/looped decision is preserved exactly.)
pub fn looks_looped(text: &str) -> bool {
    let t = text.trim_end();
    let b = t.as_bytes();
    let n = b.len();
    if n < 60 {
        return false;
    } // too short to judge
    let tail = 1200.min(n);
    const REPS: usize = 4; // need >=4 identical consecutive blocks
    for p in 4..=(tail / REPS) {
        // period (block length) in bytes
        let span = REPS * p;
        if span > tail {
            break;
        }
        let start = n - span;
        let unit = &b[n - p..];
        if (0..REPS).all(|k| &b[start + k * p..start + (k + 1) * p] == unit) {
            return true;
        }
    }
    false
}

/// Degeneracy predicate over a neutral [`TurnReply`] — true when the completion should be resampled
/// rather than accepted. Preserves the retired `should_resample`'s exact length/looped behavior and
/// adds the empty-turn case:
/// - `finish_reason == "length"` — a truncated verbose turn that likely never emitted a valid tool
///   call, OR
/// - the text looks like a degenerate repetition loop (see [`looks_looped`]), OR
/// - the reply has NO tool_calls AND no (non-whitespace) text — a "calls a tool then stalls / empty
///   turn" that carries nothing to act on.
///
/// CRITICAL: a turn that HAS tool_calls is NEVER flagged by the empty case — an empty `content` is
/// NORMAL on a tool-call turn (the content IS the tool call). Only a no-tool-AND-no-text turn is
/// degenerate on emptiness. Pure and separated from the loop so the decision is unit-testable.
pub fn is_degenerate(reply: &TurnReply) -> bool {
    let content = reply.text.as_deref().unwrap_or("");
    if reply.finish_reason.as_deref() == Some("length") || looks_looped(content) {
        return true;
    }
    // Empty degenerate turn: nothing to dispatch and nothing to say. Guarded on tool_calls being
    // empty so a normal tool-call turn (empty content, non-empty tool_calls) is never resampled.
    reply.tool_calls.is_empty() && content.trim().is_empty()
}

/// One logical model turn = resample(OUTER) over resilient(INNER) over `transport.call`.
///
/// Each attempt runs the request through [`call_resilient`] (retry/backoff/throttle + fallback per
/// `ep`'s opt-in `rate_limit`/`fallback`); for the PRIMARY link the already-built `transport` is
/// reused verbatim (so the injected mock transport in tests and today's real transport drive the
/// primary exactly as before), and only fallback links construct their own transport via
/// [`transport_for`]. On a SUCCESS whose OUTPUT is degenerate ([`is_degenerate`]) the turn is
/// resampled, up to `policy`'s bounds; if still degenerate after the budget, the last (best-effort)
/// reply is returned — the agent loop's dedup/streak/`max_rounds` backstop handles persistence.
/// Each resample bumps the process-global counter ([`crate::backend::openai::note_resample`]) so a
/// run's progress bar can surface it.
///
/// With `ep.fallback` empty, `ep.rate_limit` absent, and a non-degenerate first reply this is a
/// single un-throttled `transport.call` returned unchanged — byte-identical to the pre-resample
/// path for a normal completion.
pub fn call_with_resample(
    transport: &dyn ChatTransport,
    ep: &Endpoint,
    system: Option<&str>,
    messages: &[Value],
    tools: Option<&Value>,
    temperature: Option<f64>,
    policy: ResamplePolicy,
) -> anyhow::Result<TurnReply> {
    let attempt = || {
        call_resilient(ep, |link| {
            if std::ptr::eq(link, ep) {
                transport.call(link, system, messages, tools, temperature)
            } else {
                transport_for(link).call(link, system, messages, tools, temperature)
            }
        })
    };

    let mut reply = attempt()?;
    let mut length_resamples = 0usize;
    for _ in 0..policy.max_resamples {
        if !is_degenerate(&reply) {
            break;
        }
        if reply.finish_reason.as_deref() == Some("length") {
            // Length overruns get a tighter, separately-tracked cap: resampling rarely fixes a
            // chronically-verbose model, so don't spend the whole budget on it.
            if length_resamples >= policy.max_length_resamples {
                break;
            }
            length_resamples += 1;
        }
        crate::backend::openai::note_resample();
        reply = attempt()?;
    }
    Ok(reply)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::openai::{resamples, reset_resamples};
    use crate::backend::transport::ToolCall;
    use serde_json::json;
    use std::cell::RefCell;
    use std::collections::VecDeque;

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
            function_name: None,
            feedback_score_metric: None,
            feedback_bool_metric: None,
        }
    }

    fn reply(text: &str, finish: &str, tool_calls: Vec<ToolCall>) -> TurnReply {
        TurnReply {
            text: Some(text.to_string()),
            tool_calls,
            finish_reason: Some(finish.to_string()),
            raw: json!({}),
        }
    }

    /// A scripted `ChatTransport`: pops one reply per `call`, and counts how many calls it saw so a
    /// test can assert the exact number of (resilient) requests the resample loop issued. No
    /// network, no rand, no sleep — the whole layer is exercised deterministically.
    struct MockTransport {
        replies: RefCell<VecDeque<TurnReply>>,
        calls: RefCell<usize>,
    }
    impl MockTransport {
        fn new(replies: Vec<TurnReply>) -> Self {
            Self {
                replies: RefCell::new(replies.into_iter().collect()),
                calls: RefCell::new(0),
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
            _messages: &[Value],
            _tools: Option<&Value>,
            _temperature: Option<f64>,
        ) -> anyhow::Result<TurnReply> {
            *self.calls.borrow_mut() += 1;
            self.replies
                .borrow_mut()
                .pop_front()
                .ok_or_else(|| anyhow::anyhow!("MockTransport: scripted replies exhausted"))
        }
        fn push_assistant_turn(&self, _messages: &mut Vec<Value>, _reply: &TurnReply) {}
        fn push_tool_results(&self, _messages: &mut Vec<Value>, _results: &[(String, String)]) {}
    }

    /// Serializes every test that reads the process-global `RESAMPLES` counter: cargo runs a
    /// binary's tests in parallel by default, so without this one test's `reset_resamples()` /
    /// `note_resample()` could interleave with another's `resamples()` assertion.
    static RESAMPLE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Drive one `call_with_resample` through the mock transport, holding [`RESAMPLE_TEST_LOCK`]
    /// across the whole reset -> call -> read so the returned resample count is this run's own.
    /// Returns `(reply, transport_call_count, resample_count)`.
    fn run(replies: Vec<TurnReply>, policy: ResamplePolicy) -> (TurnReply, usize, u64) {
        let _guard = RESAMPLE_TEST_LOCK.lock().unwrap();
        reset_resamples();
        let ep = test_endpoint();
        let transport = MockTransport::new(replies);
        let out = call_with_resample(&transport, &ep, None, &[], None, None, policy)
            .expect("call_with_resample");
        let calls = *transport.calls.borrow();
        (out, calls, resamples())
    }

    // --- pure predicate --------------------------------------------------------------------------

    #[test]
    fn looks_looped_detects_and_passes() {
        assert!(looks_looped(
            &("prefix ".to_string() + &"ABABAB".repeat(20))
        ));
        assert!(!looks_looped("too short"));
        assert!(!looks_looped(
            "The quick brown fox jumps over the lazy dog. It was a bright cold day in April, and \
             the clocks were striking thirteen. A different sentence follows, varying its wording."
        ));
    }

    #[test]
    fn degenerate_on_length_looped_and_empty_no_tool_turn() {
        // finish_reason == length -> degenerate regardless of content.
        assert!(is_degenerate(&reply("anything", "length", vec![])));
        // A repetition loop even on a "stop" finish.
        let looped = "prefix ".to_string() + &"ABABAB".repeat(20);
        assert!(is_degenerate(&reply(&looped, "stop", vec![])));
        // Empty text AND no tool calls -> the stalled/empty turn.
        assert!(is_degenerate(&reply("   ", "stop", vec![])));
    }

    #[test]
    fn not_degenerate_on_good_answer_or_normal_tool_call_turn() {
        // A normal, complete answer.
        assert!(!is_degenerate(&reply(
            "ANSWER: a short normal answer",
            "stop",
            vec![]
        )));
        // A normal tool-call turn: empty content is NORMAL (the content IS the tool call) — and its
        // finish_reason is "tool_calls", not "length" — so it must NEVER be flagged degenerate.
        let tc = vec![ToolCall {
            id: "c1".into(),
            name: "search".into(),
            args: json!({"q": "x"}),
        }];
        assert!(!is_degenerate(&reply("", "tool_calls", tc)));
    }

    // --- loop behavior through the mock transport ------------------------------------------------

    #[test]
    fn resamples_once_on_length_then_returns_good_message() {
        let (out, calls, rs) = run(
            vec![
                reply("...truncated verbose reasoning", "length", vec![]),
                reply("ANSWER: ok", "stop", vec![]),
            ],
            ResamplePolicy::default(),
        );
        assert_eq!(
            calls, 2,
            "must resample exactly once after a finish_reason=length turn"
        );
        assert_eq!(
            out.text.as_deref(),
            Some("ANSWER: ok"),
            "returns the good (second) reply"
        );
        assert_eq!(rs, 1, "the counter must tally the one resample");
    }

    #[test]
    fn resamples_on_empty_no_tool_turn_then_returns_answer() {
        let (out, calls, rs) = run(
            vec![
                reply("   ", "stop", vec![]),
                reply("ANSWER: recovered", "stop", vec![]),
            ],
            ResamplePolicy::default(),
        );
        assert_eq!(calls, 2, "an empty no-tool turn must resample once");
        assert_eq!(out.text.as_deref(), Some("ANSWER: recovered"));
        assert_eq!(rs, 1);
    }

    #[test]
    fn normal_tool_call_turn_is_never_resampled() {
        // Empty content + tool_calls: the normal case. Only ONE call, no resample, returned as-is.
        let tc = vec![ToolCall {
            id: "c1".into(),
            name: "search".into(),
            args: json!({"q": "x"}),
        }];
        let (out, calls, rs) = run(vec![reply("", "tool_calls", tc)], ResamplePolicy::default());
        assert_eq!(calls, 1, "a normal tool-call turn must not be resampled");
        assert_eq!(rs, 0);
        assert!(
            !out.tool_calls.is_empty(),
            "the tool-call reply is returned unchanged"
        );
    }

    #[test]
    fn good_answer_returns_in_one_call() {
        let (out, calls, rs) = run(
            vec![reply("ANSWER: done", "stop", vec![])],
            ResamplePolicy::default(),
        );
        assert_eq!(calls, 1, "a good completion returns in one call");
        assert_eq!(rs, 0);
        assert_eq!(out.text.as_deref(), Some("ANSWER: done"));
    }

    #[test]
    fn length_resample_capped_accepts_second_truncated_completion() {
        // Chronically-verbose model: length on every turn. Default length-cap is 1, so exactly one
        // resample then it gives up and returns the second (still truncated) completion.
        let (out, calls, rs) = run(
            vec![
                reply("...truncated A", "length", vec![]),
                reply("...truncated B", "length", vec![]),
            ],
            ResamplePolicy::default(),
        );
        assert_eq!(
            calls, 2,
            "must resample once, not twice, before giving up on length"
        );
        assert_eq!(rs, 1);
        assert_eq!(
            out.text.as_deref(),
            Some("...truncated B"),
            "returns the last best-effort reply"
        );
    }

    #[test]
    fn loop_resample_path_still_honors_full_budget() {
        // A persistent repetition loop (finish "stop") is bounded by max_resamples, NOT the tighter
        // length sub-cap: the full GEN_LOOP_RETRIES (2) resamples fire, then it gives up.
        let looped = "prefix ".to_string() + &"ABABAB".repeat(20);
        let (out, calls, rs) = run(
            vec![
                reply(&looped, "stop", vec![]),
                reply(&looped, "stop", vec![]),
                reply(&looped, "stop", vec![]),
            ],
            ResamplePolicy::default(),
        );
        assert_eq!(
            calls, 3,
            "1 initial + 2 resamples for a persistent loop, unbounded by length cap"
        );
        assert_eq!(rs, 2);
        assert!(
            out.text.as_deref().unwrap().contains("ABABAB"),
            "returns last best-effort"
        );
    }

    #[test]
    fn zero_length_cap_never_resamples_on_length() {
        // max_length_resamples == 0 disables length-resampling: a single length turn is accepted.
        let policy = ResamplePolicy {
            max_resamples: 2,
            max_length_resamples: 0,
        };
        let (out, calls, rs) = run(vec![reply("...truncated", "length", vec![])], policy);
        assert_eq!(
            calls, 1,
            "length cap 0 accepts the first completion outright"
        );
        assert_eq!(rs, 0);
        assert_eq!(out.text.as_deref(), Some("...truncated"));
    }
}
