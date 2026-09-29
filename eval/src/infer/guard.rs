//! Auth + non-loopback interlock, reusing the MCP's own primitives (`glossa::serve_guard`,
//! `glossa::mcp_auth`) so the two servers behave identically. The decisions are pure and
//! unit-tested here; the axum layer that applies them is feature-gated with the rest of the server.

/// Refuse a non-loopback bind without auth. Reuses the MCP interlock (`interlock_refuses`); Phase 1
/// has no TLS, so `tls_active = false`.
pub fn interlock_ok(bind: &str, has_auth: bool, insecure: bool) -> bool {
    glossa::serve_guard::interlock_refuses(bind, has_auth, false, insecure).is_none()
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
pub use layer::auth_layer;

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

    use super::auth_ok;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interlock_refuses_public_bind_without_auth() {
        assert!(!interlock_ok("0.0.0.0:8071", false, false));
        assert!(interlock_ok("0.0.0.0:8071", true, false));
        assert!(interlock_ok("0.0.0.0:8071", false, true));
        assert!(interlock_ok("127.0.0.1:8071", false, false));
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
