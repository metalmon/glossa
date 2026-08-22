//! Per-unit resume checkpoint substrate. Tracks done units across resumable operations.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::report::sanitize_id;

/// Persistent checkpoint storage for resumable operations.
/// Tracks which units have been processed via one-file-per-unit in `done/` subdirectory.
pub struct Checkpoint {
    dir: PathBuf,
}

impl Checkpoint {
    /// Open or create a checkpoint at the given run directory.
    /// Creates `run_dir/done/` subdirectory if it doesn't exist.
    pub fn open(run_dir: &Path) -> io::Result<Self> {
        let done_dir = run_dir.join("done");
        fs::create_dir_all(&done_dir)?;
        Ok(Checkpoint { dir: done_dir })
    }

    /// Check if a unit has been marked as done.
    pub fn is_done(&self, unit_id: &str) -> bool {
        let filename = sanitize_id(unit_id);
        self.dir.join(&filename).exists()
    }

    /// Mark a unit as done with a payload string.
    pub fn mark(&self, unit_id: &str, payload: &str) -> io::Result<()> {
        let filename = sanitize_id(unit_id);
        let path = self.dir.join(&filename);
        fs::write(path, payload)?;
        Ok(())
    }

    /// List all done unit IDs (sanitized filenames in the checkpoint directory).
    pub fn done_ids(&self) -> Vec<String> {
        match fs::read_dir(&self.dir) {
            Ok(entries) => entries
                .filter_map(|entry| {
                    entry.ok().and_then(|e| {
                        e.file_name()
                            .into_string()
                            .ok()
                            .filter(|name| !name.starts_with('.'))
                    })
                })
                .collect(),
            Err(_) => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_marks_and_reports_done() {
        let d = tempfile::tempdir().unwrap();
        let cp = Checkpoint::open(d.path()).unwrap();
        assert!(!cp.is_done("doc/a.md#judge:pair-1"));
        cp.mark("doc/a.md#judge:pair-1", "no-edge").unwrap();
        assert!(cp.is_done("doc/a.md#judge:pair-1"));
        // survives reopen
        let cp2 = Checkpoint::open(d.path()).unwrap();
        assert!(cp2.is_done("doc/a.md#judge:pair-1"));
    }
}
