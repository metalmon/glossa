//! Per-thread capture of the reader's text-only `user_sim` dialogue.
//!
//! Split out of `backend::openai` so the reader-dialogue store lives on its own; the token
//! accounting subsystem (`backend::accounting`) clears it per conversation via [`reset`].

thread_local! {
    /// Text-only assistant↔user_sim dialogue captured during the CALLING THREAD's current reader
    /// conversation (no tool calls / tool results). Populated by `run_agent_loop_capturing` only
    /// when a `user_sim` gate is active AND it deflected at least once; drained by the eval closure
    /// via `take_reader_dialogue` on the SAME worker thread. Reset per conversation by
    /// `accounting::reset_conversation_prefix` (which calls [`reset`]), so a fresh case never
    /// inherits the previous case's dialogue.
    static READER_DIALOGUE: std::cell::RefCell<Vec<(String, String)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Append one `(role, text)` turn to the calling thread's reader-dialogue store (see
/// `READER_DIALOGUE`). Only the reader's agent loop calls this, and only under a `user_sim` gate.
pub fn push_reader_dialogue_turn(role: &str, text: &str) {
    READER_DIALOGUE.with(|d| d.borrow_mut().push((role.to_string(), text.to_string())));
}

/// Drain and return the calling thread's captured reader dialogue (leaves the store empty). Empty
/// when no `user_sim` dialogue occurred, so callers can treat empty as "grade the answer as before".
pub fn take_reader_dialogue() -> Vec<(String, String)> {
    READER_DIALOGUE.with(|d| std::mem::take(&mut *d.borrow_mut()))
}

/// Per-conversation outcome of the no-tool answer guard (see `agent_loop`): whether the reader
/// produced a substantive answer before any tool call (`fired`), and whether a later resample then
/// emitted a tool call that rescued it (`rescued`). Reset per conversation by [`reset`], drained by
/// [`take_no_tool_stats`] on the same worker thread — same handoff discipline as the dialogue store.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct NoToolStats {
    pub fired: bool,
    pub rescued: bool,
}

thread_local! {
    static NO_TOOL_STATS: std::cell::Cell<NoToolStats> =
        const { std::cell::Cell::new(NoToolStats { fired: false, rescued: false }) };
}

/// The reader answered without any tool call (from memory). Idempotent within a conversation.
pub fn mark_no_tool_gate_fired() {
    NO_TOOL_STATS.with(|s| {
        let mut v = s.get();
        v.fired = true;
        s.set(v);
    });
}

/// After the guard fired, a resample produced a tool call (the from-memory answer was rescued).
pub fn mark_no_tool_rescued() {
    NO_TOOL_STATS.with(|s| {
        let mut v = s.get();
        v.rescued = true;
        s.set(v);
    });
}

/// Drain this thread's stats (leaving them reset), for the case-result builder.
pub fn take_no_tool_stats() -> NoToolStats {
    NO_TOOL_STATS.with(|s| s.replace(NoToolStats::default()))
}

/// Clear the calling thread's reader-dialogue store. Called by
/// `accounting::reset_conversation_prefix` at the start of every conversation so a fresh
/// seed/doc/case never inherits the previous conversation's dialogue — keeping the store's
/// per-conversation reset behavior identical to when it lived alongside `PREV_PROMPT_TOKENS`.
pub(crate) fn reset() {
    READER_DIALOGUE.with(|d| d.borrow_mut().clear());
    NO_TOOL_STATS.with(|s| s.set(NoToolStats::default()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::accounting::reset_conversation_prefix;

    #[test]
    fn reader_dialogue_push_take_roundtrips_and_drains() {
        // Reset clears the per-thread dialogue store; push accumulates; take drains.
        reset_conversation_prefix();
        assert!(take_reader_dialogue().is_empty());
        push_reader_dialogue_turn("assistant", "hello");
        push_reader_dialogue_turn("user", "and?");
        let d = take_reader_dialogue();
        assert_eq!(
            d,
            vec![
                ("assistant".to_string(), "hello".to_string()),
                ("user".to_string(), "and?".to_string()),
            ]
        );
        // take drained it
        assert!(take_reader_dialogue().is_empty());
    }

    #[test]
    fn no_tool_stats_mark_take_roundtrip_and_reset() {
        // Reset (per-conversation) clears the stats; marks accumulate; take drains.
        reset_conversation_prefix();
        assert_eq!(take_no_tool_stats(), NoToolStats::default());
        mark_no_tool_gate_fired();
        mark_no_tool_rescued();
        let s = take_no_tool_stats();
        assert!(s.fired && s.rescued);
        // take drained it
        assert_eq!(take_no_tool_stats(), NoToolStats::default());
        // reset also clears without a take
        mark_no_tool_gate_fired();
        reset_conversation_prefix();
        assert_eq!(take_no_tool_stats(), NoToolStats::default());
    }
}
