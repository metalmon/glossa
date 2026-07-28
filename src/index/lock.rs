//! Cross-process advisory lock for index (re)build operations.
//!
//! `index_dir` clears and rewrites `.glossa/index`. When two processes (editor instances, the
//! MCP server, a CLI `reindex`) do that on the same base at once, Windows file locking turns the
//! overlap into `Access is denied` — one process is removing the directory another is opening.
//! This lock serializes the whole rebuild: the first holder proceeds, everyone else skips (the
//! index is cooperative — whoever wins leaves it correct, so a skip is safe and idempotent).

use std::fs::{File, OpenOptions};
use std::path::Path;

use fs4::FileExt;

/// RAII holder of `.glossa/index.lock`. Releases the advisory lock on drop, so the lock lives
/// exactly as long as the `index_dir` call that took it.
pub struct IndexLockGuard {
    file: File,
}

impl Drop for IndexLockGuard {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}

/// Try to take the exclusive index lock for `root` without blocking. Returns `Some(guard)` when
/// this process now owns the lock, or `None` when another process already holds it (it is
/// indexing the same base) or the lock file could not be opened. Never blocks and never errors:
/// the caller treats `None` as "someone else is indexing, skip".
pub fn try_index_lock(root: &Path) -> Option<IndexLockGuard> {
    let lock_path = root.join(".glossa").join("index.lock");
    if let Some(parent) = lock_path.parent() {
        std::fs::create_dir_all(parent).ok()?;
    }
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .ok()?;
    match file.try_lock() {
        Ok(()) => Some(IndexLockGuard { file }),
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_is_exclusive_and_released_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        let first = try_index_lock(root);
        assert!(first.is_some(), "first acquire wins the lock");

        // While the first guard is alive, a second attempt (as another process would make) is
        // refused rather than blocking.
        assert!(
            try_index_lock(root).is_none(),
            "second acquire is refused while the lock is held"
        );

        drop(first);

        // After the holder drops, the lock is free again — a later `index_dir` can proceed.
        assert!(
            try_index_lock(root).is_some(),
            "lock is re-acquirable once released"
        );
    }
}
