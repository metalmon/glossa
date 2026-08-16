//! Gateway-enforced idle-session timeout for the streamable-http transport.
//!
//! rmcp 1.8 has no native "terminate a session after N minutes of inactivity" (only a completed-
//! channel cache TTL, and SSE keep-alive that keeps connections up). So we enforce it at our own
//! middleware layer: track the last time each `Mcp-Session-Id` made a request, and once a session
//! has been idle past the threshold, refuse its next request with 404 — the streamable-http signal
//! for an unknown/terminated session, which a spec-compliant client answers by re-initializing (a
//! cheap handshake; the KB holds no per-session state). OPT-IN: a zero threshold disables it, so it
//! never surprises a client that isn't ready for session expiry.

use std::collections::HashMap;
use std::sync::Mutex;

/// Per-session last-activity clock (session id → epoch-ms of last request). Shared behind an `Arc`
/// between the idle-check middleware and the periodic reaper.
#[derive(Default)]
pub struct SessionActivity {
    last: Mutex<HashMap<String, u64>>,
}

impl SessionActivity {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record activity for `session_id` at `now_ms`, returning whether it may proceed. Returns
    /// `false` (and forgets the session) only when `idle_ms > 0` and the session's previous activity
    /// is older than `idle_ms` — an idled-out session. A brand-new session id, or any session under
    /// the threshold, returns `true` and has its clock advanced. `idle_ms == 0` disables expiry.
    pub fn check_and_touch(&self, session_id: &str, idle_ms: u64, now_ms: u64) -> bool {
        let mut m = self.last.lock().unwrap_or_else(|e| e.into_inner());
        if idle_ms > 0 {
            if let Some(&prev) = m.get(session_id) {
                if now_ms.saturating_sub(prev) > idle_ms {
                    m.remove(session_id);
                    return false;
                }
            }
        }
        m.insert(session_id.to_string(), now_ms);
        true
    }

    /// Drop sessions idle longer than `idle_ms` (housekeeping so abandoned sessions don't accumulate
    /// in the map). No-op when disabled.
    pub fn reap(&self, idle_ms: u64, now_ms: u64) {
        if idle_ms == 0 {
            return;
        }
        let mut m = self.last.lock().unwrap_or_else(|e| e.into_inner());
        m.retain(|_, &mut t| now_ms.saturating_sub(t) <= idle_ms);
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.last.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_session_passes_and_is_tracked() {
        let a = SessionActivity::new();
        assert!(a.check_and_touch("s1", 1000, 10_000));
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn activity_under_threshold_keeps_session_alive() {
        let a = SessionActivity::new();
        assert!(a.check_and_touch("s1", 1000, 10_000));
        // 500ms later (< 1000ms idle) → still alive, clock advances.
        assert!(a.check_and_touch("s1", 1000, 10_500));
    }

    #[test]
    fn idle_past_threshold_expires_once_then_reinit_is_fresh() {
        let a = SessionActivity::new();
        assert!(a.check_and_touch("s1", 1000, 10_000));
        // 1500ms later (> 1000ms) → expired, and forgotten.
        assert!(!a.check_and_touch("s1", 1000, 11_500));
        assert_eq!(a.len(), 0, "expired session is dropped");
        // A re-initialized session (same or new id) starts fresh.
        assert!(a.check_and_touch("s1", 1000, 11_600));
    }

    #[test]
    fn zero_threshold_disables_expiry() {
        let a = SessionActivity::new();
        assert!(a.check_and_touch("s1", 0, 10_000));
        // Even a huge gap never expires when disabled.
        assert!(a.check_and_touch("s1", 0, 10_000_000));
    }

    #[test]
    fn reap_drops_only_stale_entries() {
        let a = SessionActivity::new();
        a.check_and_touch("fresh", 1000, 10_000);
        a.check_and_touch("stale", 1000, 5_000);
        a.reap(1000, 10_200); // fresh is 200ms old, stale is 5200ms old
        assert_eq!(a.len(), 1);
        assert!(a.check_and_touch("fresh", 1000, 10_300), "fresh survived");
    }
}
