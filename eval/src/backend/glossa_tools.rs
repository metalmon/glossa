use glossa::trace::TraceLog;
use serde_json::{json, Value};
use std::path::Path;

const READ_CHARS_CAP: usize = 4000;

/// Run a BM25 search against the corpus index; return (model-facing text, surfaced titles).
pub fn run_search(work: &Path, query: &str, limit: usize, trace: &TraceLog) -> (String, Vec<String>) {
    let idx = match glossa::index::store::DocIndex::open_or_create(work) {
        Ok(i) => i,
        Err(e) => return (format!("search error: {e}"), Vec::new()),
    };
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
            let body = hits
                .iter()
                .map(|h| format!("{}:{}: {}  [{:.3}]", h.path, h.location, h.snippet, h.score))
                .collect::<Vec<_>>()
                .join("\n");
            (body, titles)
        }
        Err(e) => (format!("search error: {e}"), Vec::new()),
    }
}

/// Read a document (optionally a location); truncated to fit small-model context.
pub fn run_read(work: &Path, path: &str, location: Option<&str>, trace: &TraceLog) -> String {
    let _ = work; // path is absolute in search results
    match glossa::read::read_region(Path::new(path), location) {
        Ok(text) => {
            trace.log("read", json!({ "path": path, "location": location }), json!({ "path": path }));
            if text.chars().count() > READ_CHARS_CAP {
                text.chars().take(READ_CHARS_CAP).collect::<String>() + "\n…(truncated)"
            } else {
                text
            }
        }
        Err(e) => format!("read error: {e}"),
    }
}

/// Dispatch a tool by name. Returns (result string for the model, titles surfaced by a search).
pub fn exec(name: &str, args: &Value, work: &Path, trace: &TraceLog) -> (String, Vec<String>) {
    match name {
        "search" => {
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
            run_search(work, query, limit, trace)
        }
        "read" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let location = args.get("location").and_then(|v| v.as_str());
            (run_read(work, path, location, trace), Vec::new())
        }
        other => (format!("unknown tool: {other}"), Vec::new()),
    }
}
