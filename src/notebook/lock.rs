//! Cross-process advisory lock for notebook write operations.

use std::fs::OpenOptions;
use std::path::Path;

use fs4::FileExt;

pub const LOCK_BUSY_MSG: &str = "another process is editing the notebook — retry";

/// Run `f` while holding `.glossa/notebook.lock` (exclusive, non-blocking).
pub fn with_notebook_write_lock<R>(
    root: &Path,
    f: impl FnOnce() -> anyhow::Result<R>,
) -> anyhow::Result<R> {
    let lock_path = root.join(".glossa").join("notebook.lock");
    if let Some(parent) = lock_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let lock_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)?;
    match lock_file.try_lock() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => anyhow::bail!(LOCK_BUSY_MSG),
        Err(e) => return Err(e.into()),
    }
    let result = f();
    let _ = FileExt::unlock(&lock_file);
    result
}
