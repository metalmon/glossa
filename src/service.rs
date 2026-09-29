//! Cross-platform service install/uninstall/start/stop/status for `kb` and `inference-server`.
//!
//! One shape for both binaries and both OSes: a [`ServiceSpec`] (name + program + args) is turned
//! into a Windows SCM service (via the `windows-service` crate's `ServiceManager`) or a Linux
//! systemd unit (rendered here, dropped into `/etc/systemd/system`, driven by `systemctl`). macOS
//! launchd is not supported yet. The two pure renderers — [`render_systemd_unit`] and
//! [`windows_bin_path`] — are unit-tested; the actual SCM/`systemctl` calls are `#[cfg]`-gated and
//! need elevation to run, so they are compile-verified and exercised operationally.

use std::path::{Path, PathBuf};

/// A service to register, in binary- and OS-neutral terms: a unique `name`, human labels, the
/// `program` to run, its `args`, and whether the program pings a systemd watchdog.
pub struct ServiceSpec {
    pub name: String,
    pub display_name: String,
    pub description: String,
    pub program: PathBuf,
    pub args: Vec<String>,
    pub watchdog: bool,
}

/// Render a systemd unit for `spec` (Linux). `Type=notify` (the binary calls `sd_notify` READY once
/// serving — see `sdnotify.rs`), `WatchdogSec=30` when the binary pings, `Restart=on-failure`.
pub fn render_systemd_unit(spec: &ServiceSpec) -> String {
    let exec = format!("{} {}", spec.program.display(), join_args_shell(&spec.args));
    let watchdog = if spec.watchdog { "WatchdogSec=30\n" } else { "" };
    format!(
        "[Unit]\nDescription={}\nAfter=network-online.target\nWants=network-online.target\n\n\
         [Service]\nType=notify\nExecStart={exec}\n{watchdog}Restart=on-failure\nRestartSec=2\n\n\
         [Install]\nWantedBy=multi-user.target\n",
        spec.description
    )
}

/// Shell-join args, double-quoting any that contain a space (systemd `ExecStart` parses this).
fn join_args_shell(args: &[String]) -> String {
    args.iter()
        .map(|a| {
            if a.contains(' ') {
                format!("\"{a}\"")
            } else {
                a.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Build the SCM `binPath`-style string (Windows): the program is always quoted, and every arg
/// containing a space is quoted, then space-joined — avoiding the classic `sc.exe binPath= ` footgun
/// where an unquoted space splits the path. (The programmatic `install` passes structured args to
/// the SCM instead; this pure helper mirrors the same quoting for diagnostics/round-tripping.)
pub fn windows_bin_path(program: &Path, args: &[String]) -> String {
    let mut parts = vec![format!("\"{}\"", program.display())];
    for a in args {
        if a.contains(' ') {
            parts.push(format!("\"{a}\""));
        } else {
            parts.push(a.clone());
        }
    }
    parts.join(" ")
}

// ---------------------------------------------------------------------------
// Windows: SCM via the `windows-service` crate's ServiceManager.
// ---------------------------------------------------------------------------
#[cfg(windows)]
mod os {
    use super::ServiceSpec;
    use std::ffi::OsString;
    use windows_service::service::{
        ServiceAccess, ServiceErrorControl, ServiceInfo, ServiceStartType, ServiceType,
    };
    use windows_service::service_manager::{ServiceManager, ServiceManagerAccess};

    /// The access rights `install` needs on the created/opened service.
    fn manage_access() -> ServiceAccess {
        ServiceAccess::CHANGE_CONFIG
            | ServiceAccess::START
            | ServiceAccess::STOP
            | ServiceAccess::DELETE
            | ServiceAccess::QUERY_STATUS
    }

    /// Map a `windows_service::Error` to a friendly one: access-denied → "run elevated".
    fn map_err(e: windows_service::Error) -> anyhow::Error {
        if let windows_service::Error::Winapi(io) = &e {
            if io.raw_os_error() == Some(5) {
                // ERROR_ACCESS_DENIED
                return anyhow::anyhow!(
                    "run elevated (Administrator) to manage Windows services"
                );
            }
        }
        anyhow::Error::new(e)
    }

    pub fn install(spec: &ServiceSpec) -> anyhow::Result<()> {
        let mgr =
            ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CREATE_SERVICE)
                .map_err(map_err)?;
        let info = ServiceInfo {
            name: OsString::from(&spec.name),
            display_name: OsString::from(&spec.display_name),
            service_type: ServiceType::OWN_PROCESS,
            start_type: ServiceStartType::AutoStart,
            error_control: ServiceErrorControl::Normal,
            executable_path: spec.program.clone(),
            launch_arguments: spec.args.iter().map(OsString::from).collect(),
            dependencies: vec![],
            account_name: None,
            account_password: None,
        };
        match mgr.create_service(&info, manage_access()) {
            Ok(svc) => {
                svc.set_description(&spec.description).map_err(map_err)?;
                Ok(())
            }
            // ERROR_SERVICE_EXISTS (1073): open it, refresh the description, report it clearly.
            Err(windows_service::Error::Winapi(io)) if io.raw_os_error() == Some(1073) => {
                let svc = mgr.open_service(&spec.name, manage_access()).map_err(map_err)?;
                svc.set_description(&spec.description).map_err(map_err)?;
                eprintln!(
                    "service {} already exists — left in place (uninstall first to recreate)",
                    spec.name
                );
                Ok(())
            }
            Err(e) => Err(map_err(e)),
        }
    }

    pub fn uninstall(name: &str) -> anyhow::Result<()> {
        let mgr = ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT)
            .map_err(map_err)?;
        let svc = mgr
            .open_service(name, ServiceAccess::STOP | ServiceAccess::DELETE)
            .map_err(map_err)?;
        let _ = svc.stop(); // best-effort: ignore "not running"
        svc.delete().map_err(map_err)?;
        Ok(())
    }

    pub fn start(name: &str) -> anyhow::Result<()> {
        use std::ffi::OsStr;
        let mgr = ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT)
            .map_err(map_err)?;
        let svc = mgr
            .open_service(name, ServiceAccess::START)
            .map_err(map_err)?;
        svc.start(&[] as &[&OsStr]).map_err(map_err)?;
        Ok(())
    }

    pub fn stop(name: &str) -> anyhow::Result<()> {
        let mgr = ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT)
            .map_err(map_err)?;
        let svc = mgr
            .open_service(name, ServiceAccess::STOP)
            .map_err(map_err)?;
        svc.stop().map_err(map_err)?;
        Ok(())
    }

    pub fn status(name: &str) -> anyhow::Result<String> {
        let mgr = ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT)
            .map_err(map_err)?;
        let svc = mgr
            .open_service(name, ServiceAccess::QUERY_STATUS)
            .map_err(map_err)?;
        let st = svc.query_status().map_err(map_err)?;
        Ok(format!("{:?}", st.current_state))
    }
}

// ---------------------------------------------------------------------------
// Linux: systemd unit file + systemctl.
// ---------------------------------------------------------------------------
#[cfg(target_os = "linux")]
mod os {
    use super::{render_systemd_unit, ServiceSpec};

    fn unit_path(name: &str) -> String {
        format!("/etc/systemd/system/{name}.service")
    }

    /// EACCES on the unit file → "run elevated".
    fn map_io_err(e: std::io::Error) -> anyhow::Error {
        if e.raw_os_error() == Some(13) {
            anyhow::anyhow!("run elevated (sudo) to manage systemd services")
        } else {
            anyhow::Error::new(e)
        }
    }

    fn systemctl(args: &[&str]) -> anyhow::Result<String> {
        let out = std::process::Command::new("systemctl")
            .args(args)
            .output()
            .map_err(map_io_err)?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            if stderr.contains("Access denied")
                || stderr.contains("Interactive authentication required")
                || out.status.code() == Some(4)
            {
                anyhow::bail!("run elevated (sudo) to manage systemd services: {}", stderr.trim());
            }
            anyhow::bail!("systemctl {:?} failed: {}", args, stderr.trim());
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    pub fn install(spec: &ServiceSpec) -> anyhow::Result<()> {
        std::fs::write(unit_path(&spec.name), render_systemd_unit(spec)).map_err(map_io_err)?;
        systemctl(&["daemon-reload"])?;
        systemctl(&["enable", &spec.name])?;
        Ok(())
    }

    pub fn uninstall(name: &str) -> anyhow::Result<()> {
        let _ = systemctl(&["stop", name]); // best-effort
        let _ = systemctl(&["disable", name]);
        std::fs::remove_file(unit_path(name)).map_err(map_io_err)?;
        systemctl(&["daemon-reload"])?;
        Ok(())
    }

    pub fn start(name: &str) -> anyhow::Result<()> {
        systemctl(&["start", name]).map(|_| ())
    }

    pub fn stop(name: &str) -> anyhow::Result<()> {
        systemctl(&["stop", name]).map(|_| ())
    }

    pub fn status(name: &str) -> anyhow::Result<String> {
        systemctl(&["is-active", name]).map(|s| s.trim().to_string())
    }
}

// ---------------------------------------------------------------------------
// Unsupported (macOS launchd deferred, and any other OS).
// ---------------------------------------------------------------------------
#[cfg(not(any(windows, target_os = "linux")))]
mod os {
    use super::ServiceSpec;

    const MSG: &str =
        "service management is not supported on this OS yet (Windows SCM + Linux systemd only)";

    pub fn install(_spec: &ServiceSpec) -> anyhow::Result<()> {
        anyhow::bail!(MSG)
    }
    pub fn uninstall(_name: &str) -> anyhow::Result<()> {
        anyhow::bail!(MSG)
    }
    pub fn start(_name: &str) -> anyhow::Result<()> {
        anyhow::bail!(MSG)
    }
    pub fn stop(_name: &str) -> anyhow::Result<()> {
        anyhow::bail!(MSG)
    }
    pub fn status(_name: &str) -> anyhow::Result<String> {
        anyhow::bail!(MSG)
    }
}

/// Register `spec` as a service (Windows SCM / Linux systemd; unsupported elsewhere).
pub fn install(spec: &ServiceSpec) -> anyhow::Result<()> {
    os::install(spec)
}
/// Remove the service `name`.
pub fn uninstall(name: &str) -> anyhow::Result<()> {
    os::uninstall(name)
}
/// Start the service `name`.
pub fn start(name: &str) -> anyhow::Result<()> {
    os::start(name)
}
/// Stop the service `name`.
pub fn stop(name: &str) -> anyhow::Result<()> {
    os::stop(name)
}
/// A one-line status string for the service `name`.
pub fn status(name: &str) -> anyhow::Result<String> {
    os::status(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn spec() -> ServiceSpec {
        ServiceSpec {
            name: "glossa-x".into(),
            display_name: "glossa x".into(),
            description: "test".into(),
            program: PathBuf::from("/opt/kb/kb"),
            args: vec![
                "mcp".into(),
                "/data/base one".into(),
                "--bind".into(),
                "127.0.0.1:8801".into(),
            ],
            watchdog: true,
        }
    }

    #[test]
    fn systemd_unit_has_execstart_notify_watchdog_and_restart() {
        let u = render_systemd_unit(&spec());
        assert!(u.contains("Type=notify"));
        assert!(u.contains("WatchdogSec="));
        assert!(u.contains("Restart=on-failure"));
        assert!(u.contains("WantedBy=multi-user.target"));
        // ExecStart quotes the arg with a space:
        assert!(u.contains("ExecStart=/opt/kb/kb mcp \"/data/base one\" --bind 127.0.0.1:8801"));
    }

    #[test]
    fn windows_bin_path_quotes_spaces() {
        let p = windows_bin_path(
            std::path::Path::new(r"C:\kb\kb.exe"),
            &[
                "mcp".into(),
                r"C:\kb\base one".into(),
                "--bind".into(),
                "127.0.0.1:8801".into(),
            ],
        );
        assert_eq!(
            p,
            r#""C:\kb\kb.exe" mcp "C:\kb\base one" --bind 127.0.0.1:8801"#
        );
    }
}
