//! Cross-platform service install/uninstall/start/stop/status for `kb` and `kbi`.
//!
//! One shape for both binaries and both OSes: a [`ServiceSpec`] (name + program + args) is turned
//! into a Windows SCM service (via the `windows-service` crate's `ServiceManager`) or a Linux
//! systemd unit (rendered here, dropped into `/etc/systemd/system`, driven by `systemctl`). macOS
//! launchd is not supported yet. The pure renderer — [`render_systemd_unit`] — is unit-tested;
//! the actual SCM/`systemctl` calls are `#[cfg]`-gated and need elevation to run, so they are
//! compile-verified and exercised operationally.

use std::path::{Path, PathBuf};

/// A service to register, in binary- and OS-neutral terms: a unique `name`, human labels, the
/// `program` to run, its `args`, and whether the program pings a systemd watchdog.
///
/// `args` are the plain serve args — exactly what an operator would type. The flags that make an
/// SCM-launched process hand off to the service dispatcher are **not** in here: they are
/// Windows-only and get added by [`ServiceSpec::scm_launch_args`] at install time. A systemd unit
/// must not carry them (see [`render_systemd_unit`]).
pub struct ServiceSpec {
    pub name: String,
    pub display_name: String,
    pub description: String,
    pub program: PathBuf,
    pub args: Vec<String>,
    pub watchdog: bool,
}

impl ServiceSpec {
    /// The args a Windows SCM launch needs: the serve args plus `--windows-service --service-name
    /// <name>`, which is how the SCM-launched process knows to route into the control-handler
    /// dispatcher instead of running in the foreground.
    ///
    /// Idempotent — a spec that already carries either flag is passed through unchanged, so a
    /// hand-built spec cannot end up with it twice.
    pub fn scm_launch_args(&self) -> Vec<String> {
        let mut args = self.args.clone();
        if !args.iter().any(|a| a == "--windows-service") {
            args.push("--windows-service".to_string());
        }
        if !args.iter().any(|a| a == "--service-name") {
            args.push("--service-name".to_string());
            args.push(self.name.clone());
        }
        args
    }
}

/// Render a systemd unit for `spec` (Linux). `Type=notify` (the binary calls `sd_notify` READY once
/// serving — see `sdnotify.rs`), `WatchdogSec=30` when the binary pings, `Restart=on-failure`.
///
/// `ExecStart` is the plain serve command line, with no `--windows-service`. That flag routes the
/// process into the service dispatcher, which off Windows is only a Ctrl-C/SIGTERM bridge — and
/// taking it means `run_serve` is told not to handle signals, which silently drops the SIGHUP
/// handler (log-level reload, TLS reload, forced freshen). The foreground path already does
/// everything a unit needs: READY=1 after bind, the watchdog ping, SIGTERM drain, and SIGHUP.
pub fn render_systemd_unit(spec: &ServiceSpec) -> String {
    let program = spec.program.display().to_string();
    let prog = if program.contains(' ') {
        format!("\"{program}\"")
    } else {
        program
    };
    let exec = format!("{prog} {}", join_args_shell(&spec.args));
    let watchdog = if spec.watchdog {
        "WatchdogSec=30\n"
    } else {
        ""
    };
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


// ---------------------------------------------------------------------------
// Generic service run-dispatcher: run an arbitrary serve closure under a platform's service
// supervisor (Windows SCM control handler; off Windows, a Ctrl-C bridge). Shared by `kb` and
// `kbi` — each passes its own serve closure.
// ---------------------------------------------------------------------------

/// The serve closure a service dispatcher drives: given a cancellation token (tripped on
/// Stop/Shutdown) and an `on_ready` callback (call once the listener is bound + warm, to flip the
/// service to "Running"), run the server until the token is cancelled.
pub type ServiceRun = Box<
    dyn FnOnce(tokio_util::sync::CancellationToken, Box<dyn FnOnce() + Send>) -> anyhow::Result<()>
        + Send,
>;

/// What an SCM control message maps to (Windows). Pure, so it is unit-testable without the SCM.
#[cfg(windows)]
#[derive(Debug, PartialEq, Eq)]
pub enum ControlOutcome {
    /// Stop / Shutdown → cancel the token (graceful shutdown).
    Cancel,
    /// Interrogate → acknowledge, no state change.
    NoOp,
    /// Anything else → report "not implemented" to the SCM.
    Unhandled,
}

/// Map an SCM control message to a [`ControlOutcome`] (pure).
#[cfg(windows)]
pub fn control_action(control: windows_service::service::ServiceControl) -> ControlOutcome {
    use windows_service::service::ServiceControl;
    match control {
        ServiceControl::Stop | ServiceControl::Shutdown => ControlOutcome::Cancel,
        ServiceControl::Interrogate => ControlOutcome::NoOp,
        _ => ControlOutcome::Unhandled,
    }
}

#[cfg(windows)]
pub use scm::run_as_service;

#[cfg(windows)]
mod scm {
    use super::{control_action, ControlOutcome, ServiceRun};
    use std::ffi::OsString;
    use std::sync::Mutex;
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;
    use windows_service::service::{
        ServiceControlAccept, ServiceExitCode, ServiceState, ServiceStatus, ServiceType,
    };
    use windows_service::service_control_handler::{self, ServiceControlHandlerResult};
    use windows_service::{define_windows_service, service_dispatcher};

    const SERVICE_TYPE: ServiceType = ServiceType::OWN_PROCESS;

    // The closure + name are stashed before the SCM dispatcher starts, then taken inside the
    // SCM-invoked `service_main`. `Mutex<Option<..>>` (const-init) keeps the statics `Sync`.
    static RUN: Mutex<Option<ServiceRun>> = Mutex::new(None);
    static NAME: Mutex<Option<String>> = Mutex::new(None);

    /// Run `run` as a Windows service: stash it + `service_name`, then hand control to the SCM
    /// dispatcher (blocks until the service stops). The control handler maps Stop/Shutdown → cancel;
    /// `run` receives the token + an `on_ready` that flips the service to `Running`.
    pub fn run_as_service(service_name: String, run: ServiceRun) -> anyhow::Result<()> {
        *NAME.lock().unwrap_or_else(|e| e.into_inner()) = Some(service_name.clone());
        *RUN.lock().unwrap_or_else(|e| e.into_inner()) = Some(run);
        service_dispatcher::start(service_name, ffi_service_main)
            .map_err(|e| anyhow::anyhow!("windows service dispatcher failed to start: {e}"))?;
        Ok(())
    }

    define_windows_service!(ffi_service_main, service_main);

    fn service_main(_args: Vec<OsString>) {
        if let Err(e) = run_service() {
            tracing::error!("windows service exited with error: {e}");
        }
    }

    fn run_service() -> anyhow::Result<()> {
        let cancel = CancellationToken::new();

        // Control handler: Stop / Shutdown → cancel (same graceful path as Ctrl-C).
        let handler_cancel = cancel.clone();
        let event_handler = move |control| -> ServiceControlHandlerResult {
            match control_action(control) {
                ControlOutcome::Cancel => {
                    handler_cancel.cancel();
                    ServiceControlHandlerResult::NoError
                }
                ControlOutcome::NoOp => ServiceControlHandlerResult::NoError,
                ControlOutcome::Unhandled => ServiceControlHandlerResult::NotImplemented,
            }
        };
        let name = NAME
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .unwrap_or_else(|| "glossa".to_string());
        let status_handle = service_control_handler::register(&name, event_handler)?;

        status_handle.set_service_status(ServiceStatus {
            service_type: SERVICE_TYPE,
            current_state: ServiceState::StartPending,
            controls_accepted: ServiceControlAccept::empty(),
            exit_code: ServiceExitCode::Win32(0),
            checkpoint: 0,
            wait_hint: Duration::from_secs(120),
            process_id: None,
        })?;

        // ServiceStatusHandle is Copy: one copy flips to Running via on_ready, the other reports
        // Stopped after the serve returns.
        let status_for_running = status_handle;
        let on_ready: Box<dyn FnOnce() + Send> = Box::new(move || {
            let _ = status_for_running.set_service_status(ServiceStatus {
                service_type: SERVICE_TYPE,
                current_state: ServiceState::Running,
                controls_accepted: ServiceControlAccept::STOP | ServiceControlAccept::SHUTDOWN,
                exit_code: ServiceExitCode::Win32(0),
                checkpoint: 0,
                wait_hint: Duration::default(),
                process_id: None,
            });
        });

        let run = RUN
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .expect("serve closure set before dispatcher start");
        let result = run(cancel, on_ready);

        status_handle.set_service_status(ServiceStatus {
            service_type: SERVICE_TYPE,
            current_state: ServiceState::Stopped,
            controls_accepted: ServiceControlAccept::empty(),
            exit_code: ServiceExitCode::Win32(if result.is_ok() { 0 } else { 1 }),
            checkpoint: 0,
            wait_hint: Duration::default(),
            process_id: None,
        })?;
        result
    }
}

/// Off Windows there is no SCM: run the closure directly, bridging Ctrl-C to the cancellation token,
/// with a no-op `on_ready` (systemd drives lifecycle via signals + `sd_notify`, not a control
/// handler). Kept so callers stay cross-platform; `--windows-service` is Windows-only in practice.
#[cfg(not(windows))]
pub fn run_as_service(_service_name: String, run: ServiceRun) -> anyhow::Result<()> {
    let cancel = tokio_util::sync::CancellationToken::new();
    let bridge = cancel.clone();
    std::thread::spawn(move || {
        if let Ok(rt) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            rt.block_on(async {
                // Ctrl-C (foreground) OR SIGTERM (what `systemctl stop` / container runtimes send)
                // → graceful shutdown. Without the SIGTERM arm a systemd service would be killed
                // abruptly instead of draining.
                let ctrl_c = async {
                    let _ = tokio::signal::ctrl_c().await;
                };
                #[cfg(unix)]
                {
                    let term = async {
                        match tokio::signal::unix::signal(
                            tokio::signal::unix::SignalKind::terminate(),
                        ) {
                            Ok(mut s) => {
                                s.recv().await;
                            }
                            Err(_) => std::future::pending::<()>().await,
                        }
                    };
                    tokio::select! { _ = ctrl_c => {}, _ = term => {} }
                }
                #[cfg(not(unix))]
                ctrl_c.await;
            });
            crate::sdnotify::stopping();
            bridge.cancel();
        }
    });
    run(cancel, Box::new(|| {}))
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

    /// Map a `windows_service::Error` to a friendly one: access-denied → "run elevated";
    /// service-not-found → a clear "no such service" instead of a raw WinAPI code.
    fn map_err(e: windows_service::Error) -> anyhow::Error {
        if let windows_service::Error::Winapi(io) = &e {
            match io.raw_os_error() {
                Some(5) => {
                    // ERROR_ACCESS_DENIED
                    return anyhow::anyhow!(
                        "run elevated (Administrator) to manage Windows services"
                    );
                }
                Some(1060) => {
                    // ERROR_SERVICE_DOES_NOT_EXIST
                    return anyhow::anyhow!("no such service (is it installed?)");
                }
                _ => {}
            }
        }
        anyhow::Error::new(e)
    }

    /// `start` is the one place where ERROR_ACCESS_DENIED does NOT mean "you are not elevated":
    /// the manager and the service handle are already open by then, so the denial comes from the
    /// SCM failing to LAUNCH the process — most often because the service's logon account cannot
    /// read or execute the image. Routing it through the generic map sent an operator who was
    /// already Administrator looking for a privilege they already had; observed against a service
    /// whose binary sat under another user's profile, where the account simply had no ACL on it.
    pub(super) fn map_start_err(name: &str, e: windows_service::Error) -> anyhow::Error {
        if let windows_service::Error::Winapi(io) = &e {
            match io.raw_os_error() {
                // ERROR_ACCESS_DENIED, from the launch rather than from the handle.
                Some(5) => {
                    return anyhow::anyhow!(
                    "access denied starting {name} — this is the service manager failing to launch \
                     it, not a missing privilege on your side. Check that the service's logon \
                     account can read and execute the image (`sc.exe qc {name}` prints \
                     BINARY_PATH_NAME; a binary under another user's profile is the usual cause), \
                     and that the account still holds `Log on as a service`."
                )
                }
                // ERROR_SERVICE_LOGON_FAILED
                Some(1069) => {
                    return anyhow::anyhow!(
                    "the logon account for {name} was rejected: wrong password (it is stored at \
                     install time and does not follow a password change), or the account lacks \
                     `Log on as a service` — secpol.msc, Local Policies, User Rights Assignment."
                )
                }
                // ERROR_SERVICE_REQUEST_TIMEOUT
                Some(1053) => {
                    return anyhow::anyhow!(
                    "{name} did not signal the service manager in time: the process launched but \
                     did not come up — look for its own startup failure (an unreadable --config \
                     path or corpus root will do it)."
                )
                }
                _ => {}
            }
        }
        map_err(e)
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
            launch_arguments: spec.scm_launch_args().iter().map(OsString::from).collect(),
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
                let svc = mgr
                    .open_service(&spec.name, manage_access())
                    .map_err(map_err)?;
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
        svc.start(&[] as &[&OsStr])
            .map_err(|e| map_start_err(name, e))?;
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

    #[cfg(test)]
    mod tests {
        use super::*;

        fn winapi(code: i32) -> windows_service::Error {
            windows_service::Error::Winapi(std::io::Error::from_raw_os_error(code))
        }

        /// A wrong diagnosis costs more than no diagnosis: "run elevated" sent an operator who was
        /// already Administrator hunting a privilege they had, while the actual cause — the service
        /// account having no ACL on the image — went unmentioned. Opening a handle and launching a
        /// process fail with the same code and mean different things.
        #[test]
        fn access_denied_on_start_does_not_blame_elevation() {
            let msg = map_start_err("glossa-x", winapi(5)).to_string();
            assert!(
                !msg.contains("run elevated"),
                "the handle is already open by then: {msg}"
            );
            assert!(
                msg.contains("read and execute the image") && msg.contains("glossa-x"),
                "it must name the likely cause and the service: {msg}"
            );
            // Opening a handle is the case where elevation IS the answer — unchanged.
            assert!(map_err(winapi(5)).to_string().contains("run elevated"));
        }

        #[test]
        fn logon_failure_and_timeout_are_named_rather_than_raw() {
            let msg = map_start_err("glossa-x", winapi(1069)).to_string();
            assert!(
                msg.contains("Log on as a service") && msg.contains("password"),
                "{msg}"
            );
            let msg = map_start_err("glossa-x", winapi(1053)).to_string();
            assert!(msg.contains("did not signal"), "{msg}");
            // Anything unrecognised still falls through to the shared mapping.
            assert!(map_start_err("glossa-x", winapi(1060))
                .to_string()
                .contains("no such service"));
        }
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
                anyhow::bail!(
                    "run elevated (sudo) to manage systemd services: {}",
                    stderr.trim()
                );
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

#[cfg(all(test, windows))]
mod control_tests {
    use super::*;
    use windows_service::service::ServiceControl;

    #[test]
    fn stop_and_shutdown_cancel_interrogate_noop() {
        assert!(matches!(
            control_action(ServiceControl::Stop),
            ControlOutcome::Cancel
        ));
        assert!(matches!(
            control_action(ServiceControl::Shutdown),
            ControlOutcome::Cancel
        ));
        assert!(matches!(
            control_action(ServiceControl::Interrogate),
            ControlOutcome::NoOp
        ));
        assert!(matches!(
            control_action(ServiceControl::Pause),
            ControlOutcome::Unhandled
        ));
    }
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
    fn systemd_unit_carries_no_scm_flags() {
        // Reported from a pilot kit: ExecStart arrived with `--windows-service` in it. Besides
        // reading as nonsense in a Linux unit, the flag routes the process into the service
        // dispatcher, which tells `run_serve` not to handle signals and so drops the SIGHUP
        // handler (log reload, TLS reload, forced freshen). `docs/deploy/mcp-server.md` documents
        // the flag as SCM-only.
        let u = render_systemd_unit(&spec());
        assert!(
            !u.contains("--windows-service"),
            "unit was:
{u}"
        );
        assert!(
            !u.contains("--service-name"),
            "unit was:
{u}"
        );
    }

    #[test]
    fn scm_launch_args_add_the_flags_once() {
        let s = spec();
        let a = s.scm_launch_args();
        assert!(a.starts_with(&s.args), "the serve args come first: {a:?}");
        assert!(a.windows(2).any(|w| w == ["--service-name", "glossa-x"]));
        assert_eq!(a.iter().filter(|x| *x == "--windows-service").count(), 1);
        // A spec that already carries them is passed through unchanged, so nothing can double up.
        let mut pre = spec();
        pre.args.push("--windows-service".into());
        pre.args.push("--service-name".into());
        pre.args.push("other".into());
        assert_eq!(pre.scm_launch_args(), pre.args);
    }

    #[test]
    fn systemd_unit_quotes_a_spaced_program_path() {
        let mut s = spec();
        s.program = PathBuf::from("/opt/my kb/kb");
        let u = render_systemd_unit(&s);
        assert!(
            u.contains("ExecStart=\"/opt/my kb/kb\" mcp \"/data/base one\""),
            "unit was:\n{u}"
        );
    }
}
