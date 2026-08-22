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

    /// Remove a unit's mark (same id passed to `mark`/`is_done`) — e.g. when the work it recorded
    /// as done is invalidated (its underlying graph nodes were dropped) and must be redone on the
    /// next run instead of being skipped. A no-op if the mark doesn't exist.
    pub fn remove(&self, unit_id: &str) -> io::Result<()> {
        let filename = sanitize_id(unit_id);
        self.remove_raw(&filename)
    }

    /// Remove a mark by its exact on-disk filename, as returned by `done_ids()` — for batch
    /// invalidation by pattern (e.g. every `judge:*` mark referencing a node that was just
    /// dropped), where only the already-sanitized filename is known, not the original unit id
    /// (sanitizing folds distinct ids' punctuation to the same `_`, so it can't be un-sanitized).
    /// A no-op if the file doesn't exist.
    pub fn remove_raw(&self, filename: &str) -> io::Result<()> {
        let path = self.dir.join(filename);
        if path.exists() {
            fs::remove_file(path)?;
        }
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

    #[test]
    fn remove_clears_a_mark_and_is_a_noop_when_absent() {
        let d = tempfile::tempdir().unwrap();
        let cp = Checkpoint::open(d.path()).unwrap();
        cp.mark("extract:a.md", "done").unwrap();
        assert!(cp.is_done("extract:a.md"));

        cp.remove("extract:a.md").unwrap();
        assert!(!cp.is_done("extract:a.md"));

        // Removing again (already absent) must not error.
        cp.remove("extract:a.md").unwrap();
        assert!(!cp.is_done("extract:a.md"));
    }
}
