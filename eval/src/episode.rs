//! Thread-local TensorZero episode id for the `kbx eval` reader loop — modeled directly on
//! `glossa::trace::last_trace_path` (same per-thread `RefCell` pattern, same guarantee: `kbx eval`
//! runs each case to completion on ONE worker thread via `run_units_parallel`, so a per-case
//! `reset()` is a per-episode reset even under `--jobs N` concurrency).
//!
//! [`crate::backend::transport::tensorzero::TzTransport::call`] sets the episode id from the
//! FIRST `/inference` response of a case and reuses it on every subsequent turn (passing it back
//! as the request's `episode_id`), grouping the whole reader loop into ONE TensorZero episode.
//! `run_eval` calls [`reset`] before running each case and reads [`current`] afterward to decide
//! whether to post feedback. Every non-TensorZero transport never touches this module, so
//! `current()` stays `None` for them and the feedback post in `run_eval` never fires.

use std::cell::RefCell;

thread_local! {
    static CURRENT_EPISODE: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Clear this thread's episode id. Call once per case, before the reader runs, so a stale episode
/// id from a PRIOR case (or a prior worker's reuse of this OS thread) can never leak into this
/// one's `/feedback` post.
pub fn reset() {
    CURRENT_EPISODE.with(|e| *e.borrow_mut() = None);
}

/// Record the episode id TensorZero returned on this thread's most recent `/inference` call.
pub fn set(id: String) {
    CURRENT_EPISODE.with(|e| *e.borrow_mut() = Some(id));
}

/// This thread's current episode id, if any `/inference` call has set one since the last
/// [`reset`]. Returns a clone so the caller owns it independently of the thread-local slot.
pub fn current() -> Option<String> {
    CURRENT_EPISODE.with(|e| e.borrow().clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_clears_to_none() {
        set("stale".to_string());
        reset();
        assert_eq!(current(), None);
    }

    #[test]
    fn set_then_current_returns_it() {
        reset();
        set("ep-123".to_string());
        assert_eq!(current().as_deref(), Some("ep-123"));
    }

    #[test]
    fn reset_after_set_then_set_again_reflects_latest() {
        reset();
        set("first".to_string());
        set("second".to_string());
        assert_eq!(current().as_deref(), Some("second"));
        reset();
        assert_eq!(current(), None);
    }

    /// Per-thread isolation: a fresh thread has never called `set`, so its slot starts `None`
    /// regardless of what the spawning thread's slot holds — the same guarantee `run_units_parallel`
    /// relies on for concurrent cases to never cross-contaminate episode ids.
    #[test]
    fn per_thread_independent() {
        reset();
        set("outer".to_string());
        let child_saw_none = std::thread::spawn(|| current().is_none()).join().unwrap();
        assert!(
            child_saw_none,
            "a thread that never called set() must see None"
        );
        assert_eq!(
            current().as_deref(),
            Some("outer"),
            "outer thread's slot must be untouched"
        );
    }
}
