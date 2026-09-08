//! Thin systemd `sd_notify` wrapper. Active only under systemd `Type=notify` (NOTIFY_SOCKET set);
//! a plain no-op everywhere else (stdio, dev, Windows) so nothing else in the server needs to care.
#[cfg(unix)]
mod imp {
    use sd_notify::NotifyState;
    pub fn ready() {
        let _ = sd_notify::notify(false, &[NotifyState::Ready]);
    }
    pub fn stopping() {
        let _ = sd_notify::notify(false, &[NotifyState::Stopping]);
    }
    /// Tell systemd a `Type=notify` service is mid-reload (SIGHUP log-level/freshen); it does not
    /// gate the watchdog or restart the process — `ready()` follows once the reload completes.
    pub fn reloading() {
        let _ = sd_notify::notify(false, &[NotifyState::Reloading]);
    }
    /// `Some(usec)` when systemd armed the watchdog for this process (`WatchdogSec` in the unit).
    pub fn watchdog_usec() -> Option<u64> {
        let mut usec: u64 = 0;
        // `true` = unset the env for children; we own the process, so clearing is fine.
        if sd_notify::watchdog_enabled(true, &mut usec) && usec > 0 {
            Some(usec)
        } else {
            None
        }
    }
    pub fn keepalive() {
        let _ = sd_notify::notify(false, &[NotifyState::Watchdog]);
    }
}
#[cfg(not(unix))]
mod imp {
    pub fn ready() {}
    pub fn stopping() {}
    pub fn reloading() {}
    pub fn watchdog_usec() -> Option<u64> {
        None
    }
    pub fn keepalive() {}
}
pub use imp::{keepalive, ready, reloading, stopping, watchdog_usec};

/// Spawn the watchdog pinger. Gated on ASYNC-RUNTIME LIVENESS (this task being scheduled at all),
/// NOT on `readiness()` (R-C1: a blocking open that a busy blocking-pool stalls → false restart).
/// A wedged runtime cannot run this task → pings stop → systemd restarts, which is correct.
pub fn spawn_watchdog(
    usec: u64,
    cancel: tokio_util::sync::CancellationToken,
) -> tokio::task::JoinHandle<()> {
    // Ping every usec/2 (systemd's recommended margin).
    let period = std::time::Duration::from_micros(usec / 2);
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(period);
        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = tick.tick() => keepalive(),
            }
        }
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn all_calls_are_inert_without_notify_socket() {
        // No NOTIFY_SOCKET in the test env → every call is a silent no-op.
        super::ready();
        super::stopping();
        super::reloading();
        super::keepalive();
        assert!(super::watchdog_usec().is_none());
    }

    #[tokio::test]
    async fn watchdog_task_exits_on_cancel() {
        let cancel = tokio_util::sync::CancellationToken::new();
        let h = super::spawn_watchdog(2_000 /*usec → 1ms period*/, cancel.clone());
        cancel.cancel();
        tokio::time::timeout(std::time::Duration::from_secs(1), h)
            .await
            .expect("watchdog task must exit promptly on cancel")
            .expect("join ok");
    }
}
