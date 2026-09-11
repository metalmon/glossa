//! Runtime log-level reload. The tracing filter is a `reload::Layer` whose handle is kept here;
//! the desired level is read from a control file (`<state-dir>/.glossa/loglevel`, one line: a bare
//! level like `debug` or a full RUST_LOG-style filter). An mtime-poll task applies it everywhere
//! (incl. Windows); SIGHUP applies it immediately on unix. Re-reading $RUST_LOG would be useless —
//! a running process's env is fixed — so the file is the source of truth.
use std::sync::OnceLock;
use tracing_subscriber::{filter::EnvFilter, prelude::*, reload, Registry};

pub static RELOAD_HANDLE: OnceLock<reload::Handle<EnvFilter, Registry>> = OnceLock::new();

/// Build and install the global subscriber with a reloadable EnvFilter layer. Best-effort: a second
/// install (tests) is ignored. `json` selects the JSON fmt layer (GLOSSA_LOG_FORMAT=json).
pub fn install(json: bool, default_filter: EnvFilter) {
    let (filter_layer, handle) = reload::Layer::new(default_filter);
    let _ = RELOAD_HANDLE.set(handle);
    let fmt = tracing_subscriber::fmt::layer().with_writer(std::io::stderr);
    let subscriber = Registry::default().with(filter_layer);
    // json vs human: both wrap the SAME reload filter layer over Registry, so RELOAD_HANDLE's type
    // (`Handle<EnvFilter, Registry>`) is identical in both arms.
    if json {
        let _ = subscriber.with(fmt.json().flatten_event(true)).try_init();
    } else {
        let _ = subscriber.with(fmt).try_init();
    }
}

/// Parse the control file and apply it via the reload handle. Returns the applied directive.
/// A missing/empty file or a parse error is a no-op (returns None) — never crash on operator typo.
pub fn apply_from_file(path: &std::path::Path) -> Option<String> {
    let raw = std::fs::read_to_string(path).ok()?;
    let directive = raw.trim();
    if directive.is_empty() {
        return None;
    }
    let filter = EnvFilter::try_new(directive).ok()?;
    let handle = RELOAD_HANDLE.get()?;
    handle.reload(filter).ok()?;
    Some(directive.to_string())
}

/// mtime-poll the control file (~5s) and apply on change. Cross-platform (the only reload trigger
/// on Windows). Stops with `cancel`.
pub fn spawn_poll(path: std::path::PathBuf, cancel: tokio_util::sync::CancellationToken) {
    tokio::spawn(async move {
        let mut last: Option<std::time::SystemTime> = None;
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(5));
        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = tick.tick() => {
                    let cur = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
                    if cur != last {
                        last = cur;
                        if let Some(d) = apply_from_file(&path) {
                            tracing::info!("log level reloaded from control file: {d}");
                        }
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn apply_from_file_parses_valid_and_rejects_garbage() {
        // Serialize on the crate-wide global-state lock: `install`/`reload` mutate the process-global
        // tracing subscriber + `RELOAD_HANDLE` (a `OnceLock`) shared by every test in this module.
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("loglevel");
        // Order-independent: both assertions hold whether or not a sibling test already installed the
        // process-global RELOAD_HANDLE. An empty file returns None before any parse; a directive
        // EnvFilter genuinely rejects returns None at the `try_new(..).ok()?` parse gate — BEFORE the
        // handle is ever consulted. NB: EnvFilter is lenient — "not a level" parses as a bare *target*
        // directive and is ACCEPTED (so it applies once a handle exists, which used to flake this
        // test); a bogus LEVEL like "foo=notalevel" is what `try_new` actually rejects.
        std::fs::write(&p, "").unwrap();
        assert_eq!(super::apply_from_file(&p), None, "empty file is a no-op");
        std::fs::write(&p, "foo=notalevel").unwrap();
        assert_eq!(
            super::apply_from_file(&p),
            None,
            "an unparseable directive (invalid level) is rejected at the parse gate, no crash"
        );
        std::fs::write(&p, "  debug\n").unwrap();
        // A syntactically valid directive parses; whether it applies depends on a handle existing,
        // which the end-to-end `apply_from_file_reloads_via_installed_handle` below covers.
    }

    #[test]
    fn apply_from_file_reloads_via_installed_handle() {
        // Serialize on the crate-wide global-state lock: this test reloads the process-global
        // subscriber's shared filter, which siblings (e.g. spawn_poll) also mutate — running them
        // concurrently races on that single handle.
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // `install` uses `try_init`, which is global-once-per-process; tolerate a prior init from
        // another test in this binary and only assert on the returned directive, never on global
        // filter state, so this stays isolated from other tests' subscriber.
        super::install(false, tracing_subscriber::EnvFilter::new("info"));
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("loglevel");
        std::fs::write(&p, "debug\n").unwrap();
        assert_eq!(
            super::apply_from_file(&p),
            Some("debug".to_string()),
            "a valid directive applies via the reload handle and returns the directive"
        );
    }

    #[test]
    fn missing_file_is_a_no_op() {
        // Held for parity with the other logreload tests: they all touch the process-global tracing
        // subscriber, so the whole module serializes on the shared lock.
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("does-not-exist");
        assert_eq!(super::apply_from_file(&p), None);
    }

    // Holds `TEST_ENV_LOCK` across `.await` points on purpose: the guard must serialize this test's
    // reload of the process-global subscriber against sibling logreload tests for the WHOLE body, and
    // `#[tokio::test]` runs on a current-thread runtime so the std guard never crosses threads.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn spawn_poll_detects_the_control_file_and_applies_it() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Ensure a handle exists (idempotent: OnceLock.set is a no-op if another test in this
        // binary already installed one — we only assert on the effect of OUR reload below).
        super::install(false, tracing_subscriber::EnvFilter::new("info"));
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("loglevel");
        std::fs::write(&p, "warn\n").unwrap();
        let cancel = tokio_util::sync::CancellationToken::new();
        super::spawn_poll(p.clone(), cancel.clone());
        // `tokio::time::interval` fires its first tick immediately, so the poll loop's first
        // iteration (last=None vs cur=Some(mtime) — a "change" from the unset baseline) runs
        // without waiting out the 5s period; give the spawned task a moment to be scheduled.
        for _ in 0..50 {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            if super::RELOAD_HANDLE
                .get()
                .and_then(|h| h.with_current(|f| f.to_string()).ok())
                .as_deref()
                == Some("warn")
            {
                break;
            }
        }
        cancel.cancel();
        let applied = super::RELOAD_HANDLE
            .get()
            .expect("install() above set the handle")
            .with_current(|f| f.to_string())
            .unwrap();
        assert_eq!(
            applied, "warn",
            "the poll task's first tick detected the control file and applied it via the shared handle"
        );
    }
}
