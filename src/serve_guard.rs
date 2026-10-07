//! Startup safety interlock (§3c): refuse to serve a non-loopback bind with no authentication
//! (no bearer token AND no TLS/mTLS) unless the operator passes `--insecure`. A silent open MCP
//! endpoint on 0.0.0.0 with a forgotten token is the footgun this closes.
use std::net::ToSocketAddrs;
use std::path::Path;

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

/// Read the bearer token from `path` (the `--auth-token-file` form).
///
/// Why a file at all: a service's command line is public — `ps` shows it to every local user and
/// a systemd unit is world-readable through `systemctl cat` — so `--auth-token <secret>` leaks the
/// moment it is baked into a service. The path leaks nothing; the file carries the secret and the
/// filesystem carries the permission.
///
/// The file is therefore held to that promise: on Unix it must not be readable by group or other
/// (mode & 0o077 == 0), or this refuses rather than serving with a secret anyone can read. The
/// content is trimmed (an editor's trailing newline is not part of the token) and must be
/// non-empty — an empty file means "no auth" by accident, which is exactly the footgun §3c closes.
pub fn read_token_file(path: &Path) -> anyhow::Result<String> {
    // One handle for both the checks and the read: a path checked and then re-opened can be
    // swapped in between (0644 at read time, 0600 at stat time). The regular-file check has to
    // come before the read for a second reason — `read_to_string` on a FIFO blocks forever, which
    // would hang the service at startup instead of failing it.
    use std::io::Read;
    let mut file = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("cannot open the auth token file {}: {e}", path.display()))?;
    let meta = file
        .metadata()
        .map_err(|e| anyhow::anyhow!("cannot stat the auth token file {}: {e}", path.display()))?;
    anyhow::ensure!(
        meta.is_file(),
        "the auth token file {} is not a regular file",
        path.display()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let mode = meta.permissions().mode();
        anyhow::ensure!(
            mode & 0o077 == 0,
            "the auth token file {} is readable by group or other (mode {:o}) — restrict it with \
             `chmod 600 {}`",
            path.display(),
            mode & 0o7777,
            path.display()
        );
        // SAFETY: `geteuid` is a parameter-free process query.
        let euid = unsafe { libc::geteuid() };
        anyhow::ensure!(
            meta.uid() == euid || meta.uid() == 0,
            "the auth token file {} is owned by uid {}, not this user or root — its owner could \
             read or replace the token",
            path.display(),
            meta.uid()
        );
    }
    let mut token = String::new();
    file.read_to_string(&mut token)
        .map_err(|e| anyhow::anyhow!("cannot read the auth token file {}: {e}", path.display()))?;
    // Only the trailing newline an editor leaves, and a BOM Notepad writes. Nothing else: the env
    // var is not trimmed either, and a token that differs by a space between the two paths would
    // be a silent 401 nobody could explain.
    let token = token
        .trim_start_matches('\u{feff}')
        .trim_end_matches(['\n', '\r'])
        .to_string();
    anyhow::ensure!(
        !token.is_empty(),
        "the auth token file {} is empty — write the token into it, or drop \
         `--auth-token-file` to serve without authentication",
        path.display()
    );
    Ok(token)
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

    /// Only the newline an editor leaves is stripped. A space is NOT: the env-var path does not
    /// trim either, and a token that differed by a space between the two paths would be a 401
    /// with no visible cause.
    #[test]
    fn only_the_trailing_newline_is_stripped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tok");
        std::fs::write(&path, "s3cret\r\n").unwrap();
        restrict(&path);

        assert_eq!(read_token_file(&path).unwrap(), "s3cret");

        std::fs::write(&path, " s3cret ").unwrap();
        restrict(&path);
        assert_eq!(read_token_file(&path).unwrap(), " s3cret ");
    }

    /// Notepad writes a \u{feff}; it is not whitespace, so without this it would become part of the
    /// token and every request would 401 with nothing to see.
    #[test]
    fn a_byte_order_mark_is_not_part_of_the_token() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tok");
        std::fs::write(&path, "\u{feff}s3cret\n").unwrap();
        restrict(&path);

        assert_eq!(read_token_file(&path).unwrap(), "s3cret");
    }

    /// A directory (or a FIFO) must be refused by the file-type check, not read.
    #[test]
    fn a_directory_is_refused_rather_than_read() {
        let dir = tempfile::tempdir().unwrap();

        let err = read_token_file(dir.path()).unwrap_err().to_string();
        assert!(
            err.contains("regular file") || err.contains("cannot"),
            "{err}"
        );
    }

    #[test]
    fn an_empty_token_file_is_refused_rather_than_serving_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tok");
        std::fs::write(&path, "\n").unwrap();
        restrict(&path);

        let err = read_token_file(&path).unwrap_err().to_string();
        assert!(err.contains("is empty"), "{err}");
    }

    #[test]
    fn a_missing_token_file_names_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent");

        let err = read_token_file(&path).unwrap_err().to_string();
        assert!(err.contains("absent"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn a_group_readable_token_file_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tok");
        std::fs::write(&path, "s3cret").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();

        let err = read_token_file(&path).unwrap_err().to_string();
        assert!(err.contains("group or other"), "{err}");
    }

    /// 0600 on Unix; a no-op elsewhere (the check is Unix-only).
    fn restrict(path: &std::path::Path) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        #[cfg(not(unix))]
        let _ = path;
    }
}
