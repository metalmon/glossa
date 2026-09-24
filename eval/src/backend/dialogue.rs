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

/// Clear the calling thread's reader-dialogue store. Called by
/// `accounting::reset_conversation_prefix` at the start of every conversation so a fresh
/// seed/doc/case never inherits the previous conversation's dialogue — keeping the store's
/// per-conversation reset behavior identical to when it lived alongside `PREV_PROMPT_TOKENS`.
pub(crate) fn reset() {
    READER_DIALOGUE.with(|d| d.borrow_mut().clear());
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
}
