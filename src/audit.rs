//! Security audit events.
//!
//! Dedicated, structured events on the `glossa::audit` tracing target so a SIEM / log pipeline can
//! select them independently of ordinary logs (and get one JSON object per event under
//! `GLOSSA_LOG_FORMAT=json`). The schema mirrors the fields a security event log is expected to
//! carry — type/category, time, source, result, subject/object — with `time` supplied by the
//! tracing timestamp. `subject` (the acting principal) stays coarse until OIDC/IdP integration lands
//! upstream: for now it is the client source (IP) for network events, or `-` where unknown.

/// The tracing target for audit events. Filter on it in a subscriber or downstream (e.g.
/// `RUST_LOG=glossa::audit=info`, or a SIEM rule on `"target":"glossa::audit"`).
pub const TARGET: &str = "glossa::audit";

/// Emit one security-audit event.
///
/// - `category`: the security class — `"auth"`, `"access"`, `"admin"`.
/// - `action`: the specific event — `"bearer_reject"`, `"tool_invoke"`, …
/// - `outcome`: `"ok"`, `"denied"`, `"error"`, `"invoked"`.
/// - `source`: where it came from — a client IP for network events, else `"-"`.
/// - `object`: the object acted on — a route (`"/mcp"`) or a tool name. (Named `object`, not
///   `target`, to avoid colliding with tracing's reserved `target` field in the JSON output.)
pub fn security_event(category: &str, action: &str, outcome: &str, source: &str, object: &str) {
    tracing::info!(
        target: TARGET,
        category,
        action,
        outcome,
        source,
        object,
        "security audit event"
    );
}
