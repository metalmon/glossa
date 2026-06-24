use glossa::trace::TraceEntry;
use std::path::Path;

pub fn read_window(traces_dir: &Path, t0_ms: u64, t1_ms: u64) -> anyhow::Result<Vec<TraceEntry>> {
    let mut out = Vec::new();
    let rd = match std::fs::read_dir(traces_dir) {
        Ok(rd) => rd,
        Err(_) => return Ok(out),
    };
    for ent in rd {
        let path = ent?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        for line in std::fs::read_to_string(&path)?.lines() {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(e) = serde_json::from_str::<TraceEntry>(line) {
                if e.ts_ms >= t0_ms && e.ts_ms <= t1_ms {
                    out.push(e);
                }
            }
        }
    }
    out.sort_by_key(|e| e.ts_ms);
    Ok(out)
}

/// Collect every `path` mentioned in search-result arrays and read results.
pub fn seen_files(entries: &[TraceEntry]) -> Vec<String> {
    let mut out = Vec::new();
    for e in entries {
        match &e.result {
            serde_json::Value::Array(arr) => {
                for v in arr {
                    if let Some(p) = v.get("path").and_then(|p| p.as_str()) {
                        out.push(p.to_string());
                    }
                }
            }
            serde_json::Value::Object(o) => {
                if let Some(p) = o.get("path").and_then(|p| p.as_str()) {
                    out.push(p.to_string());
                }
            }
            _ => {}
        }
    }
    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_by_time_and_extracts_paths() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.jsonl");
        std::fs::write(&p, concat!(
            r#"{"ts_ms":100,"tool":"search","args":{},"result":[{"path":"Bob_Page.md","location":"p.1"}]}"#, "\n",
            r#"{"ts_ms":500,"tool":"read","args":{},"result":{"path":"Alice.md"}}"#, "\n",
            r#"{"ts_ms":999,"tool":"search","args":{},"result":[{"path":"Late.md"}]}"#, "\n",
        )).unwrap();

        let win = read_window(dir.path(), 50, 600).unwrap();
        assert_eq!(win.len(), 2);
        let files = seen_files(&win);
        assert_eq!(files, vec!["Alice.md".to_string(), "Bob_Page.md".to_string()]);
    }

    #[test]
    fn missing_dir_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_window(&dir.path().join("nope"), 0, u64::MAX).unwrap().is_empty());
    }
}
