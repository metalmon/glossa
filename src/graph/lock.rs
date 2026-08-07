//! Cross-process advisory lock for agent graph-write operations (`graph_upsert` / `graph_update` /
//! `graph_delete`).
//!
//! SQLite's WAL journal + `busy_timeout` prevent physical corruption across processes, and the
//! in-process `Mutex<Connection>` serializes threads — but neither protects the read-modify-write an
//! agent write performs (e.g. `upsert` reads node types to auto-orient edges BEFORE its transaction),
//! so two processes can lose an update or orient an edge on a stale read. This advisory lock
//! serializes those whole operations across processes, the way `.glossa/index.lock` serializes
//! reindexes and `.glossa/generalize.lock` the derived pass.
//!
//! Semantics: bounded-wait. Contending writers queue up to a deadline, then bail with
//! [`LOCK_BUSY_MSG`] rather than silently dropping the write.

use std::fs::OpenOptions;
use std::path::Path;
use std::time::{Duration, Instant};

use fs4::FileExt;

pub const LOCK_BUSY_MSG: &str = "another process is writing the graph — retry";

/// Run `f` while holding `.glossa/graph.lock` (exclusive). Retries acquiring the advisory lock until
/// `timeout` elapses, then bails with [`LOCK_BUSY_MSG`]. Releases the lock when `f` returns.
pub fn with_graph_write_lock<R>(
    root: &Path,
    timeout: Duration,
    f: impl FnOnce() -> anyhow::Result<R>,
) -> anyhow::Result<R> {
    let lock_path = root.join(".glossa").join("graph.lock");
    if let Some(parent) = lock_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let lock_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)?;
    let deadline = Instant::now() + timeout;
    loop {
        match lock_file.try_lock() {
            Ok(()) => break,
            Err(std::fs::TryLockError::WouldBlock) => {
                if Instant::now() >= deadline {
                    anyhow::bail!(LOCK_BUSY_MSG);
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => return Err(e.into()),
        }
    }
    let result = f();
    let _ = FileExt::unlock(&lock_file);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_and_returns_value_when_free() {
        let dir = tempfile::tempdir().unwrap();
        let got = with_graph_write_lock(dir.path(), Duration::from_secs(1), || Ok(42)).unwrap();
        assert_eq!(got, 42);
    }

    #[test]
    fn releases_on_completion() {
        let dir = tempfile::tempdir().unwrap();
        with_graph_write_lock(dir.path(), Duration::from_secs(1), || Ok(())).unwrap();
        // A second call must still acquire it → the first released on return.
        with_graph_write_lock(dir.path(), Duration::from_secs(1), || Ok(())).unwrap();
    }

    #[test]
    fn bails_busy_when_held_by_another_process() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".glossa")).unwrap();
        let lock_path = dir.path().join(".glossa").join("graph.lock");
        let held = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .unwrap();
        held.try_lock().unwrap(); // simulate another process holding the lock

        let start = Instant::now();
        let res = with_graph_write_lock(dir.path(), Duration::from_millis(150), || Ok(()));
        assert!(res.is_err(), "must bail when the lock is held, not run f");
        assert!(res.unwrap_err().to_string().contains(LOCK_BUSY_MSG));
        assert!(start.elapsed() >= Duration::from_millis(150), "waited to the deadline");
        assert!(start.elapsed() < Duration::from_secs(2), "did not hang past the deadline");
        FileExt::unlock(&held).unwrap();
    }
}
