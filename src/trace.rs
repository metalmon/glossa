use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Process-global monotonic counter folded into every trace filename so two `to_dir` calls in the
/// same process — even at the same millisecond, on different threads — never collide on a name.
static TRACE_SEQ: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// The path `to_dir` most recently created ON THIS THREAD. Lets a concurrent caller (e.g. the
    /// eval worker pool) retrieve exactly the trace file its own `answer()` just wrote, without a
    /// racy before/after directory diff. Per-thread by construction, so parallel workers never read
    /// each other's path.
    static LAST_TRACE_PATH: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

/// The trace file created by the most recent `TraceLog::to_dir` call on the CALLING thread, if any.
/// Returns a clone so the caller owns it independently of the thread-local slot.
pub fn last_trace_path() -> Option<PathBuf> {
    LAST_TRACE_PATH.with(|p| p.borrow().clone())
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TraceEntry {
    pub ts_ms: u64,
    pub tool: String,
    pub args: serde_json::Value,
    pub result: serde_json::Value,
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Append-only JSONL tool-call log. `disabled()` is a no-op; `to_dir()` writes one line per call.
/// Cloning shares the same file lock, so concurrent tool calls on different clones of the same
/// log don't interleave mid-line.
#[derive(Clone)]
pub struct TraceLog {
    path: Option<PathBuf>,
    /// Process-wide advisory lock shared across all clones of the same TraceLog instance.
    lock: Arc<Mutex<()>>,
}

impl TraceLog {
    pub fn disabled() -> TraceLog {
        TraceLog {
            path: None,
            lock: Arc::new(Mutex::new(())),
        }
    }

    pub fn to_dir(root: &Path) -> TraceLog {
        let dir = root.join(".glossa").join("traces");
        let _ = std::fs::create_dir_all(&dir);
        let seq = TRACE_SEQ.fetch_add(1, Ordering::Relaxed);
        let file = dir.join(format!("{}-{}-{}.jsonl", now_ms(), std::process::id(), seq));
        // Record this thread's just-created path so a concurrent caller can retrieve its own
        // trace file directly (see `last_trace_path`) instead of diffing the directory.
        LAST_TRACE_PATH.with(|p| *p.borrow_mut() = Some(file.clone()));
        TraceLog {
            path: Some(file),
            lock: Arc::new(Mutex::new(())),
        }
    }

    pub fn log(&self, tool: &str, args: serde_json::Value, result: serde_json::Value) {
        let Some(p) = &self.path else { return };
        let entry = TraceEntry {
            ts_ms: now_ms(),
            tool: tool.to_string(),
            args,
            result,
        };
        if let Ok(line) = serde_json::to_string(&entry) {
            let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
            {
                let _ = writeln!(f, "{line}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        TraceLog::disabled().log(
            "search",
            serde_json::json!({"q":"x"}),
            serde_json::json!([]),
        );
        assert!(!dir.path().join(".glossa").join("traces").exists());
    }

    #[test]
    fn enabled_appends_parseable_lines() {
        let dir = tempfile::tempdir().unwrap();
        let log = TraceLog::to_dir(dir.path());
        log.log(
            "search",
            serde_json::json!({"query":"calibration"}),
            serde_json::json!([{"path":"a.md","location":"p.1","score":1.0}]),
        );
        log.log(
            "read",
            serde_json::json!({"path":"a.md"}),
            serde_json::json!({"path":"a.md","location":"p.1"}),
        );

        let tdir = dir.path().join(".glossa").join("traces");
        let file = std::fs::read_dir(&tdir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let body = std::fs::read_to_string(file).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2);
        let e0: TraceEntry = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(e0.tool, "search");
        assert_eq!(e0.args["query"], "calibration");
    }

    #[test]
    fn to_dir_sets_thread_local_last_trace_path() {
        let dir = tempfile::tempdir().unwrap();
        let log = TraceLog::to_dir(dir.path());
        let recorded = last_trace_path().expect("to_dir must record this thread's trace path");
        assert_eq!(
            Some(recorded),
            log.path,
            "last_trace_path must match the log's own path"
        );
    }

    #[test]
    fn two_to_dir_calls_produce_distinct_paths() {
        let dir = tempfile::tempdir().unwrap();
        let a = TraceLog::to_dir(dir.path());
        let b = TraceLog::to_dir(dir.path());
        assert_ne!(
            a.path, b.path,
            "the process-global counter must keep filenames unique"
        );
    }

    #[test]
    fn last_trace_path_is_per_thread() {
        let dir = tempfile::tempdir().unwrap();
        let outer = TraceLog::to_dir(dir.path());
        // A fresh thread has never called to_dir, so its thread-local starts empty.
        let child_saw_none = std::thread::spawn(|| last_trace_path().is_none())
            .join()
            .unwrap();
        assert!(
            child_saw_none,
            "a thread that never called to_dir must see None"
        );
        // The outer thread's slot is untouched by the child.
        assert_eq!(last_trace_path(), outer.path);
    }
}
