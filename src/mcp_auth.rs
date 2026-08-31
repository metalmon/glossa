//! Optional bearer-token guard for the Streamable-HTTP MCP endpoint.
//!
//! The MCP server listens on loopback by default and expects a TLS/auth gateway in front for
//! network exposure. As an interim access control (a shared "integration API key" before full
//! OIDC/IdP integration lands), an operator may set a token via `--auth-token` or the
//! `GLOSSA_MCP_TOKEN` env var. When set, every `/mcp` request must carry `Authorization: Bearer
//! <token>`; otherwise it is rejected with 401. When unset, the endpoint is unauthenticated (the
//! loopback default). Health/readiness/metrics endpoints are never guarded, so probes keep working.

/// Constant-time byte comparison — avoids leaking how many leading bytes of a candidate token match
/// via response timing. The length is allowed to leak (an unequal length short-circuits): token
/// length is not the secret. Not a general-purpose crypto primitive, just enough for this guard.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Decide whether a request's `Authorization` header carries the expected bearer token. `header` is
/// the raw header value (`None` if absent). Pure so the policy is unit-tested without an HTTP stack.
pub fn bearer_ok(header: Option<&str>, expected: &str) -> bool {
    match header.and_then(|h| h.strip_prefix("Bearer ")) {
        Some(token) => constant_time_eq(token.as_bytes(), expected.as_bytes()),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_time_eq_matches_std_equality() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreu"));
        assert!(!constant_time_eq(b"secret", b"secre")); // length differs
        assert!(!constant_time_eq(b"", b"x"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn bearer_ok_requires_exact_scheme_and_token() {
        assert!(bearer_ok(Some("Bearer s3cr3t"), "s3cr3t"));
        assert!(!bearer_ok(Some("Bearer wrong"), "s3cr3t"));
        assert!(
            !bearer_ok(Some("s3cr3t"), "s3cr3t"),
            "missing Bearer scheme"
        );
        assert!(
            !bearer_ok(Some("bearer s3cr3t"), "s3cr3t"),
            "scheme is case-sensitive"
        );
        assert!(!bearer_ok(Some("Bearer "), "s3cr3t"), "empty token");
        assert!(!bearer_ok(None, "s3cr3t"), "no header");
        // No trailing-substring or prefix match.
        assert!(!bearer_ok(Some("Bearer s3cr3t "), "s3cr3t"));
        assert!(!bearer_ok(Some("Bearer s3cr3tx"), "s3cr3t"));
    }
}
