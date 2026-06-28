//! Native Windows Service integration: the SCM dispatcher + a control handler that maps Stop /
//! Shutdown to the shared cancellation token (graceful shutdown), reusing `crate::run_serve`.
//!
//! Install a service whose binPath carries the normal `mcp` args plus `--windows-service`, e.g.:
//!   sc.exe create glossa-base1-editor binPath= "C:\kb\kb.exe mcp C:\kb\base1 \
//!       --profile editor --transport streamable-http --bind 127.0.0.1:8801 \
//!       --allowed-host gw.internal --windows-service" start= auto
//! On `sc start`, the SCM launches the binary; clap routes `--windows-service` here.

use std::ffi::OsString;
use std::sync::OnceLock;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use windows_service::service::{
    ServiceControl, ServiceControlAccept, ServiceExitCode, ServiceState, ServiceStatus, ServiceType,
};
use windows_service::service_control_handler::{self, ServiceControlHandlerResult};
use windows_service::{define_windows_service, service_dispatcher};

const SERVICE_NAME: &str = "glossa";
const SERVICE_TYPE: ServiceType = ServiceType::OWN_PROCESS;

// Stashed before the dispatcher starts; read inside the SCM-invoked `service_main`.
static PARAMS: OnceLock<crate::ServeParams> = OnceLock::new();

/// Entry point from `main` when `--windows-service` is set: stash the serve config and hand control
/// to the SCM dispatcher. Blocks until the service stops.
pub fn run(params: crate::ServeParams) -> anyhow::Result<()> {
    let _ = PARAMS.set(params);
    service_dispatcher::start(SERVICE_NAME, ffi_service_main)
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

    // The SCM control handler: Stop / Shutdown → cancel (drives the same graceful path as Ctrl-C).
    let handler_cancel = cancel.clone();
    let event_handler = move |control: ServiceControl| -> ServiceControlHandlerResult {
        match control {
            ServiceControl::Stop | ServiceControl::Shutdown => {
                handler_cancel.cancel();
                ServiceControlHandlerResult::NoError
            }
            ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
            _ => ServiceControlHandlerResult::NotImplemented,
        }
    };
    let status_handle = service_control_handler::register(SERVICE_NAME, event_handler)?;

    let running = ServiceStatus {
        service_type: SERVICE_TYPE,
        current_state: ServiceState::Running,
        controls_accepted: ServiceControlAccept::STOP | ServiceControlAccept::SHUTDOWN,
        exit_code: ServiceExitCode::Win32(0),
        checkpoint: 0,
        wait_hint: Duration::default(),
        process_id: None,
    };
    status_handle.set_service_status(running)?;

    let params = PARAMS.get().expect("serve params set before dispatcher start").clone();
    let result = crate::run_serve(params, cancel, /* handle_signals = */ false);

    let stopped = ServiceStatus {
        service_type: SERVICE_TYPE,
        current_state: ServiceState::Stopped,
        controls_accepted: ServiceControlAccept::empty(),
        exit_code: ServiceExitCode::Win32(if result.is_ok() { 0 } else { 1 }),
        checkpoint: 0,
        wait_hint: Duration::default(),
        process_id: None,
    };
    status_handle.set_service_status(stopped)?;
    result
}
