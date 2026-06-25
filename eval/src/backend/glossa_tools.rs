use glossa::index::store::DocIndex;
use glossa::trace::TraceLog;
use serde_json::{json, Value};

const READ_CHARS_CAP: usize = 4000;

/// Run a BM25 search against the corpus index; return (model-facing text, surfaced titles).
///
/// Takes a borrowed `DocIndex` so the caller opens it once per question and reuses it (with its
/// cached reader) across every search/read in the episode, instead of reopening per tool call.
pub fn run_search(idx: &DocIndex, query: &str, limit: usize, trace: &TraceLog) -> (String, Vec<String>) {
    match idx.search(query, limit.max(1)) {
        Ok(hits) => {
            let trace_hits: Vec<Value> = hits
                .iter()
                .map(|h| json!({ "path": h.path, "location": h.location, "score": h.score }))
                .collect();
            trace.log("search", json!({ "query": query }), json!(trace_hits));
            let titles: Vec<String> = hits.iter().map(|h| h.location.clone()).collect();
            if hits.is_empty() {
                return ("(no results)".to_string(), titles);
            }
            let body = hits.iter().map(|h| h.display_line()).collect::<Vec<_>>().join("\n");
            (body, titles)
        }
        Err(e) => (format!("search error: {e}"), Vec::new()),
    }
}

/// Parse the model's `n` argument: a JSON integer, or any string we strip to its digits
/// (e.g. "p.7" -> 7). None if no digits are present.
fn parse_n(v: &Value) -> Option<u64> {
    if let Some(n) = v.as_u64() { return Some(n); }
    let s: String = v.as_str()?.chars().filter(|c| c.is_ascii_digit()).collect();
    s.parse::<u64>().ok()
}

/// Read a chunk by (path, number n) from the index; truncated to fit small-model context.
pub fn run_read(idx: &DocIndex, path: &str, n: u64, trace: &TraceLog) -> String {
    match idx.read_chunk_by_ord(path, n) {
        Ok(Some(c)) => {
            trace.log("read", json!({ "path": path, "n": n }), json!({ "path": path }));
            let footer = match (c.prev, c.next) {
                (Some(p), Some(nx)) => format!("\n‹ prev #{p} · next #{nx} ›"),
                (None, Some(nx)) => format!("\n‹ start · next #{nx} ›"),
                (Some(p), None) => format!("\n‹ prev #{p} · end ›"),
                (None, None) => String::new(),
            };
            cap_read(c.body) + &footer
        }
        Ok(None) => format!("no chunk #{n} in {path}"),
        Err(e) => format!("read error: {e}"),
    }
}

/// Truncate read output to fit a small model's context window.
fn cap_read(text: String) -> String {
    if text.chars().count() > READ_CHARS_CAP {
        text.chars().take(READ_CHARS_CAP).collect::<String>() + "\n…(truncated)"
    } else {
        text
    }
}

/// Dispatch a tool by name. Returns (result string for the model, titles surfaced by a search).
pub fn exec(name: &str, args: &Value, idx: &DocIndex, trace: &TraceLog) -> (String, Vec<String>) {
    match name {
        "search" => {
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
            run_search(idx, query, limit, trace)
        }
        "read" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let n = args.get("n").and_then(parse_n).unwrap_or(0);
            (run_read(idx, path, n, trace), Vec::new())
        }
        other => (format!("unknown tool: {other}"), Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glossa::index::store::DocIndex;
    use glossa::model::Chunk;
    use glossa::trace::TraceLog;
    use std::path::PathBuf;

    #[test]
    fn read_accepts_integer_or_digit_string_and_returns_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            Chunk { doc_path: PathBuf::from("d.pdf"), location: "p.7".into(), file_type: "pdf".into(), text: "седьмая страница".into() },
        ]).unwrap();
        let trace = TraceLog::disabled();

        // integer n
        let out = exec("read", &json!({"path": "d.pdf", "n": 7}), &idx, &trace).0;
        assert!(out.contains("седьмая"), "got: {out}");
        // stray string "p.7" -> digit-strip fallback -> 7
        let out2 = exec("read", &json!({"path": "d.pdf", "n": "p.7"}), &idx, &trace).0;
        assert!(out2.contains("седьмая"), "digit-strip fallback: {out2}");
    }
}
