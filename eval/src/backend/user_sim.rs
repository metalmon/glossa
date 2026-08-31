//! The reader's "simulated user" dialogue gate.
//!
//! Fixes an EARLY-STOP failure: a weak reader sometimes ends a turn by merely restating the
//! question ("QUESTION: …") or thinking out loud, with NO tool call — which `run_agent_loop`
//! otherwise accepts as the final answer. The gate gives the model the DIALOGUE it expects: on a
//! text-only turn, a PATIENT simulated user (an LLM role-playing someone who wants help and does
//! NOT know the answer) decides whether the assistant actually answered. If it only kept asking,
//! the gate hands back a short, in-character USER reply that deflects (never revealing any fact)
//! and paces "step by step"; the loop appends that as a `role:"user"` message and continues. If
//! the assistant gave a SUBSTANTIVE answer, the gate signals DONE and the loop returns that answer
//! to the scoring judge.
//!
//! Design notes (from the co-design brief):
//! - LENIENT acceptance: the gate distinguishes "answered at all" from "still asking" — it does
//!   NOT demand completeness/perfection (a strict bar makes the weak reader spiral). Err toward
//!   DONE. Quality is measured later by the separate scoring judge, not here.
//! - No `final_answer` tool and no forcing: the only signal is the `[[DONE]]` sentinel.
//! - Opt-in: a gate is built only when `[user_sim]` is configured in `lab.toml`. Absent ⇒ the loop
//!   never consults a gate and today's behavior (text-only turn = final answer) holds EXACTLY.
//! - Bounded: each deflection consumes a loop round, so the gate is naturally capped by
//!   `max_rounds` (the existing out-of-rounds "ANSWER:" backstop still applies).
//! - Fail-open: any error talking to the gate endpoint accepts the answer (returns `Ok(None)`), so
//!   a flaky sim endpoint can never hang or fail a run.

use crate::lab::Endpoint;
use serde_json::{json, Value};

/// The exact sentinel the gate emits (alone, or embedded) when the assistant has given a
/// substantive answer. Kept as one constant so the persona template (`user_sim.md`) and the
/// classifier agree on the literal.
pub const DONE_SENTINEL: &str = "[[DONE]]";

/// The seam the agent loop consults on a text-only turn. `judge` returns:
/// - `Ok(None)` — accept the assistant's turn as the answer (it was substantive, OR the gate chose
///   to fail open); the loop returns the text.
/// - `Ok(Some(deflection))` — the assistant only kept asking; `deflection` is the in-character user
///   reply to append as a `role:"user"` message before continuing the loop.
/// - `Err(_)` — a hard error the caller treats as fail-open (returns the text). `UserSimGate` never
///   surfaces this (it folds endpoint errors into `Ok(None)` itself); the variant exists so a mock
///   can exercise the loop's fail-open path deterministically.
pub trait DialogueGate {
    fn judge(
        &self,
        question: &str,
        messages: &[Value],
        proposed: &str,
    ) -> anyhow::Result<Option<String>>;
}

/// Pure decision over the gate's raw reply: `None` = accept (the reply is the `[[DONE]]` sentinel,
/// or empty), `Some(reply)` = deflection to feed back. Factored out of the networked `judge` so the
/// accept-vs-deflect logic is unit-testable without a live endpoint.
pub(crate) fn classify_reply(reply: &str) -> Option<String> {
    let trimmed = reply.trim();
    if trimmed.is_empty() || trimmed.contains(DONE_SENTINEL) {
        None
    } else {
        Some(reply.to_string())
    }
}

/// The production gate: a one-shot chat to the `[user_sim]` endpoint with the persona prompt as
/// system and the original question + the assistant's latest text as the user message. Holds only
/// borrows (the endpoint + prompt live in the caller), so it's cheap to build per question/rollout.
pub struct UserSimGate<'a> {
    ep: &'a Endpoint,
    prompt: &'a str,
}

impl<'a> UserSimGate<'a> {
    pub fn new(ep: &'a Endpoint, prompt: &'a str) -> Self {
        Self { ep, prompt }
    }

    /// The user message handed to the sim: the original question plus the assistant's latest turn.
    /// No corpus/gold is ever in scope here (the loop carries none), so nothing to leak.
    fn build_user_message(question: &str, proposed: &str) -> String {
        format!(
            "The original question I asked you:\n{question}\n\nYour latest reply:\n{proposed}\n\n\
             Respond in character now."
        )
    }
}

impl DialogueGate for UserSimGate<'_> {
    fn judge(
        &self,
        question: &str,
        _messages: &[Value],
        proposed: &str,
    ) -> anyhow::Result<Option<String>> {
        let messages = vec![
            json!({ "role": "system", "content": self.prompt }),
            json!({ "role": "user", "content": Self::build_user_message(question, proposed) }),
        ];
        let key = self.ep.resolve_key();
        // Fail OPEN: any transport error accepts the assistant's turn rather than hang/fail the run.
        let msg = match crate::backend::openai::chat_once(
            &self.ep.endpoint,
            &self.ep.model,
            &messages,
            key.as_deref(),
            self.ep.timeout_secs,
            self.ep.resolve_temperature(),
        ) {
            Ok(m) => m,
            Err(_) => return Ok(None),
        };
        let reply = msg
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        Ok(classify_reply(&reply))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_accepts_on_bare_done_sentinel() {
        assert_eq!(classify_reply("[[DONE]]"), None);
        assert_eq!(classify_reply("  [[DONE]]  \n"), None);
    }

    #[test]
    fn classify_accepts_when_done_sentinel_is_embedded() {
        // Lenient: a substantive judgement that mentions DONE anywhere accepts.
        assert_eq!(classify_reply("Great, that answers it. [[DONE]]"), None);
    }

    #[test]
    fn classify_accepts_on_empty_reply() {
        // An empty gate reply is treated as accept, never as an infinite deflection.
        assert_eq!(classify_reply("   \n  "), None);
    }

    #[test]
    fn classify_deflects_on_substantive_user_reply() {
        let d = classify_reply(
            "I don't know, that's what I was hoping you'd tell me — take it step by step.",
        );
        assert_eq!(
            d.as_deref(),
            Some("I don't know, that's what I was hoping you'd tell me — take it step by step.")
        );
    }
}
