use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

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
#[derive(Clone)]
pub struct TraceLog {
    path: Option<PathBuf>,
}

impl TraceLog {
    pub fn disabled() -> TraceLog {
        TraceLog { path: None }
    }

    pub fn to_dir(root: &Path) -> TraceLog {
        let dir = root.join(".glossa").join("traces");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join(format!("{}-{}.jsonl", now_ms(), std::process::id()));
        TraceLog { path: Some(file) }
    }

    pub fn log(&self, tool: &str, args: serde_json::Value, result: serde_json::Value) {
        let Some(p) = &self.path else { return };
        let entry = TraceEntry { ts_ms: now_ms(), tool: tool.to_string(), args, result };
        if let Ok(line) = serde_json::to_string(&entry) {
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
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
        TraceLog::disabled().log("search", serde_json::json!({"q":"x"}), serde_json::json!([]));
        assert!(!dir.path().join(".glossa").join("traces").exists());
    }

    #[test]
    fn enabled_appends_parseable_lines() {
        let dir = tempfile::tempdir().unwrap();
        let log = TraceLog::to_dir(dir.path());
        log.log("search", serde_json::json!({"query":"поверка"}), serde_json::json!([{"path":"a.md","location":"p.1","score":1.0}]));
        log.log("read", serde_json::json!({"path":"a.md"}), serde_json::json!({"path":"a.md","location":"p.1"}));

        let tdir = dir.path().join(".glossa").join("traces");
        let file = std::fs::read_dir(&tdir).unwrap().next().unwrap().unwrap().path();
        let body = std::fs::read_to_string(file).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2);
        let e0: TraceEntry = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(e0.tool, "search");
        assert_eq!(e0.args["query"], "поверка");
    }
}
