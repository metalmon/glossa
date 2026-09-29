//! Auth + non-loopback interlock, reusing the MCP's own primitives (`glossa::serve_guard`,
//! `glossa::mcp_auth`) so the two servers behave identically. The decisions are pure and
//! unit-tested here; the axum layer that applies them is feature-gated with the rest of the server.

/// Refuse a non-loopback bind without auth. Reuses the MCP interlock (`interlock_refuses`). A TLS
/// bind counts as authentication-at-transport, so `tls_active` satisfies the interlock the same way
/// an api-key does.
pub fn interlock_ok(bind: &str, has_auth: bool, tls_active: bool, insecure: bool) -> bool {
    glossa::serve_guard::interlock_refuses(bind, has_auth, tls_active, insecure).is_none()
}

/// Host-header allowlist decision: an empty allowlist allows all (default); otherwise the request's
/// Host header (port ignored) must match one entry. A missing Host against a non-empty allowlist is
/// refused. Guards against DNS-rebinding / Host-confusion when the server is publicly reachable.
pub fn host_allowed(host: Option<&str>, allowed: &[String]) -> bool {
    if allowed.is_empty() {
        return true;
    }
    match host {
        Some(h) => {
            let bare = strip_host_port(h);
            allowed.iter().any(|a| a == bare)
        }
        None => false,
    }
}

/// Strip a trailing `:port` from a `Host` header value, returning the bare host. Handles bracketed
/// IPv6 literals (`[::1]:8071` / `[::1]` -> `::1`), where a naive `split(':')` would yield `"["`.
fn strip_host_port(h: &str) -> &str {
    if let Some(rest) = h.strip_prefix('[') {
        // Bracketed IPv6: the address is inside the brackets; ignore any `:port` after `]`.
        rest.split(']').next().unwrap_or(rest)
    } else {
        // Bare host or host:port. Strip only a single trailing `:port` — an unbracketed multi-colon
        // value isn't a valid Host header, so leave it untouched rather than truncate at the first `:`.
        match h.rsplit_once(':') {
            Some((host, _port)) if !host.contains(':') => host,
            _ => h,
        }
    }
}

/// Per-request auth decision: `/health` and `/ready` are exempt; with a key configured, require the
/// Bearer token (constant-time, reusing `glossa::mcp_auth::bearer_ok`); with no key, open (the
/// interlock already guards public binds).
pub fn auth_ok(path: &str, auth_header: Option<&str>, key: Option<&str>) -> bool {
    if path == "/health" || path == "/ready" {
        return true;
    }
    match key {
        Some(k) => glossa::mcp_auth::bearer_ok(auth_header, k),
        None => true,
    }
}

#[cfg(any(
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
))]
pub use layer::{auth_layer, host_layer};

#[cfg(any(
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
))]
mod layer {
    use std::sync::Arc;

    use axum::extract::{Request, State};
    use axum::http::StatusCode;
    use axum::middleware::Next;
    use axum::response::Response;

    use super::{auth_ok, host_allowed};
    use crate::infer::state::ServerState;

    /// Reject with 401 unless `auth_ok`. Mounted via `from_fn_with_state`.
    pub async fn auth_layer(
        State(st): State<Arc<ServerState>>,
        req: Request,
        next: Next,
    ) -> Result<Response, StatusCode> {
        let path = req.uri().path().to_string();
        let header = req
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(String::from);
        if auth_ok(&path, header.as_deref(), st.api_key.as_deref()) {
            Ok(next.run(req).await)
        } else {
            Err(StatusCode::UNAUTHORIZED)
        }
    }

    /// Reject with 403 unless the request's Host header passes the allowlist (`host_allowed`).
    /// Mounted via `from_fn_with_state`; a no-op when the allowlist is empty.
    pub async fn host_layer(
        State(st): State<Arc<ServerState>>,
        req: Request,
        next: Next,
    ) -> Result<Response, StatusCode> {
        let host = req
            .headers()
            .get("host")
            .and_then(|v| v.to_str().ok())
            .map(String::from);
        if host_allowed(host.as_deref(), &st.allowed_host) {
            Ok(next.run(req).await)
        } else {
            Err(StatusCode::FORBIDDEN)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interlock_refuses_public_bind_without_auth() {
        assert!(!interlock_ok("0.0.0.0:8071", false, false, false));
        assert!(interlock_ok("0.0.0.0:8071", true, false, false));
        assert!(interlock_ok("0.0.0.0:8071", false, false, true));
        assert!(interlock_ok("127.0.0.1:8071", false, false, false));
    }

    #[test]
    fn interlock_allows_public_bind_with_tls_and_no_key() {
        // TLS is auth-at-transport: a public bind with TLS but no api-key is allowed.
        assert!(interlock_ok("0.0.0.0:8071", false, true, false)); // tls only
        assert!(!interlock_ok("0.0.0.0:8071", false, false, false)); // neither -> refuse
        assert!(interlock_ok("0.0.0.0:8071", true, false, false)); // auth only
        assert!(interlock_ok("127.0.0.1:8071", false, false, false)); // loopback
    }

    #[test]
    fn host_allowed_matches_allowlist_or_all_when_empty() {
        assert!(host_allowed(Some("scorer.example.com"), &[])); // empty => all
        assert!(host_allowed(
            Some("scorer.example.com"),
            &["scorer.example.com".into()]
        ));
        assert!(host_allowed(
            Some("scorer.example.com:8071"),
            &["scorer.example.com".into()]
        )); // port ignored
        assert!(!host_allowed(
            Some("evil.example.com"),
            &["scorer.example.com".into()]
        ));
        assert!(!host_allowed(None, &["scorer.example.com".into()])); // missing Host, non-empty list
        // Bracketed IPv6 literals: the port after `]` is ignored, and the bare address matches.
        assert!(host_allowed(Some("[::1]:8071"), &["::1".into()]));
        assert!(host_allowed(Some("[::1]"), &["::1".into()]));
        assert!(!host_allowed(Some("[::1]:8071"), &["scorer.example.com".into()]));
    }

    #[test]
    fn auth_ok_exempts_health_requires_token_elsewhere() {
        assert!(auth_ok("/health", None, Some("sek")));
        assert!(auth_ok("/ready", None, Some("sek")));
        assert!(auth_ok("/rerank", Some("Bearer sek"), Some("sek")));
        assert!(!auth_ok("/rerank", Some("Bearer no"), Some("sek")));
        assert!(!auth_ok("/rerank", None, Some("sek")));
        assert!(auth_ok("/rerank", None, None));
    }
}
