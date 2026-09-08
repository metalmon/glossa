//! Startup safety interlock (§3c): refuse to serve a non-loopback bind with no authentication
//! (no bearer token AND no TLS/mTLS) unless the operator passes `--insecure`. A silent open MCP
//! endpoint on 0.0.0.0 with a forgotten token is the footgun this closes.
use std::net::ToSocketAddrs;

/// True when every resolved address of `bind` is loopback (127.0.0.0/8, ::1). A parse failure is
/// treated as NON-loopback (fail safe — an unparseable bind should not be assumed local).
pub fn is_loopback_bind(bind: &str) -> bool {
    match bind.to_socket_addrs() {
        Ok(mut it) => {
            let addrs: Vec<_> = it.by_ref().collect();
            !addrs.is_empty() && addrs.iter().all(|a| a.ip().is_loopback())
        }
        Err(_) => false,
    }
}

/// `Some(msg)` → refuse to start. Refuse only when: non-loopback AND no token AND no active TLS,
/// AND `--insecure` was not given.
pub fn interlock_refuses(
    bind: &str,
    has_token: bool,
    tls_active: bool,
    insecure: bool,
) -> Option<String> {
    if is_loopback_bind(bind) || has_token || tls_active || insecure {
        return None;
    }
    Some(format!(
        "refusing to serve MCP on non-loopback bind '{bind}' with no authentication \
         (no --auth-token/GLOSSA_MCP_TOKEN and no TLS). Set a token, enable TLS, or pass \
         --insecure to override (NOT recommended)."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loopback_no_token_ok() {
        assert!(interlock_refuses("127.0.0.1:8080", false, false, false).is_none());
    }
    #[test]
    fn nonloopback_no_auth_refused() {
        assert!(interlock_refuses("0.0.0.0:8080", false, false, false).is_some());
    }
    #[test]
    fn nonloopback_with_token_ok() {
        assert!(interlock_refuses("0.0.0.0:8080", true, false, false).is_none());
    }
    #[test]
    fn nonloopback_with_tls_ok() {
        assert!(interlock_refuses("0.0.0.0:8080", false, true, false).is_none());
    }
    #[test]
    fn nonloopback_insecure_override_ok() {
        assert!(interlock_refuses("0.0.0.0:8080", false, false, true).is_none());
    }
    #[test]
    fn unparseable_bind_treated_nonloopback() {
        assert!(!is_loopback_bind("not-an-addr"));
    }
}
