use glossa::index::store::DocIndex;
use glossa::trace::TraceLog;
use serde_json::{json, Value};

/// Run a BM25 search (optionally scoped by path glob / file_type); model-facing numbered text + titles.
///
/// Takes a borrowed `DocIndex` so the caller opens it once per question and reuses it (with its
/// cached reader) across every search/read in the episode, instead of reopening per tool call.
pub fn run_search(idx: &DocIndex, query: &str, limit: usize, glob: Option<&str>, file_type: Option<&str>, trace: &TraceLog) -> (String, Vec<String>) {
    let (body, hits) = glossa::tools::search(idx, query, limit, glob, file_type, trace);
    (body, hits.iter().map(|h| h.location.clone()).collect())
}

/// List documents matching a shell glob; one `path  (N chunks)` per line.
pub fn run_glob(idx: &DocIndex, pattern: &str, trace: &TraceLog) -> (String, Vec<String>) {
    let body = glossa::tools::glob(idx, pattern, trace);
    (body, Vec::new())
}

/// Parse the model's `n` argument: a JSON integer, or any string we strip to its digits
/// (e.g. "p.7" -> 7). None if no digits are present.
fn parse_n(v: &Value) -> Option<u64> {
    if let Some(n) = v.as_u64() { return Some(n); }
    let s: String = v.as_str()?.chars().filter(|c| c.is_ascii_digit()).collect();
    s.parse::<u64>().ok()
}

/// Read a chunk: full text + the chunk's images (for the vision model, delivered by the backend).
pub fn run_read(idx: &DocIndex, path: &str, n: u64, trace: &TraceLog) -> (String, Vec<glossa::read::DocImage>) {
    let out = glossa::tools::read(idx, path, n, trace);
    (out.text, out.images)
}

/// Run a ripgrep-style exact/regex search over the extracted text; one line per match `path:#n: line`.
pub fn run_grep(idx: &DocIndex, pattern: &str, opts: glossa::grep::GrepOpts, trace: &TraceLog) -> (String, Vec<String>) {
    (glossa::tools::grep(idx, pattern, &opts, trace), Vec::new())
}

/// Dispatch a tool by name. Returns (result string for the model, titles surfaced by a search, images from read).
pub fn exec(name: &str, args: &Value, idx: &DocIndex, graph: Option<&glossa::graph::store::GraphStore>, trace: &TraceLog) -> (String, Vec<String>, Vec<glossa::read::DocImage>) {
    // The raw_arguments fallback (TZ hands back a JSON *string* when the model's args didn't match
    // the tool schema, e.g. a float where an int was required) would make field lookups see empty
    // values. Parse it back to an object so path/n/query/… resolve.
    let parsed;
    let args = if let Some(s) = args.as_str() {
        parsed = serde_json::from_str::<Value>(s).unwrap_or_else(|_| json!({}));
        &parsed
    } else {
        args
    };
    match name {
        "search" => {
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
            let glob = args.get("glob").and_then(|v| v.as_str());
            let file_type = args.get("file_type").and_then(|v| v.as_str());
            let (body, titles) = run_search(idx, query, limit, glob, file_type, trace);
            (body, titles, Vec::new())
        }
        "glob" => {
            let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
            let (body, titles) = run_glob(idx, pattern, trace);
            (body, titles, Vec::new())
        }
        "read" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let n = args.get("n").and_then(parse_n).unwrap_or(0);
            let (text, imgs) = run_read(idx, path, n, trace);
            (text, Vec::new(), imgs)
        }
        "grep" => {
            let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
            let opts = glossa::grep::GrepOpts {
                ignore_case: args.get("ignore_case").and_then(|v| v.as_bool()).unwrap_or(false),
                fixed: args.get("fixed").and_then(|v| v.as_bool()).unwrap_or(false),
                word: args.get("word").and_then(|v| v.as_bool()).unwrap_or(false),
                glob: args.get("glob").and_then(|v| v.as_str()).map(String::from),
                file_type: args.get("file_type").and_then(|v| v.as_str()).map(String::from),
            };
            let (body, titles) = run_grep(idx, pattern, opts, trace);
            (body, titles, Vec::new())
        }
        "glossary" => {
            let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let body = match graph { Some(g) => glossa::tools::glossary(g, name, trace), None => "(graph unavailable)".to_string() };
            (body, Vec::new(), Vec::new())
        }
        "neighbors" => {
            let node_id = args.get("node_id").and_then(|v| v.as_str()).unwrap_or("");
            let depth = args.get("depth").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
            let body = match graph { Some(g) => glossa::tools::neighbors(g, node_id, depth, trace), None => "(graph unavailable)".to_string() };
            (body, Vec::new(), Vec::new())
        }
        other => (format!("unknown tool: {other}"), Vec::new(), Vec::new()),
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
        let out = exec("read", &json!({"path": "d.pdf", "n": 7}), &idx, None, &trace).0;
        assert!(out.contains("седьмая"), "got: {out}");
        // stray string "p.7" -> digit-strip fallback -> 7
        let out2 = exec("read", &json!({"path": "d.pdf", "n": "p.7"}), &idx, None, &trace).0;
        assert!(out2.contains("седьмая"), "digit-strip fallback: {out2}");
    }

    #[test]
    fn grep_tool_finds_exact_token_via_exec() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            Chunk { doc_path: PathBuf::from("d.pdf"), location: "p.7".into(), file_type: "pdf".into(), text: "параметр maxTsdr равен 3000".into() },
        ]).unwrap();
        let trace = TraceLog::disabled();
        let out = exec("grep", &json!({"pattern": "maxTsdr"}), &idx, None, &trace).0;
        assert!(out.contains("maxTsdr"), "got: {out}");
        assert!(out.contains(":#7:"), "carries #n read key: {out}");
    }

    #[test]
    fn glob_and_scoped_search_via_exec() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            Chunk { doc_path: PathBuf::from("АБАК.pdf"), location: "p.1".into(), file_type: "pdf".into(), text: "горячая замена".into() },
            Chunk { doc_path: PathBuf::from("Other.pdf"), location: "p.1".into(), file_type: "pdf".into(), text: "горячая замена".into() },
        ]).unwrap();
        let trace = TraceLog::disabled();
        let g = exec("glob", &json!({"pattern": "*АБАК*"}), &idx, None, &trace).0;
        assert!(g.contains("АБАК") && !g.contains("Other"), "glob: {g}");
        let s = exec("search", &json!({"query": "замена", "glob": "*АБАК*"}), &idx, None, &trace).0;
        assert!(s.contains("АБАК") && !s.contains("Other"), "scoped search: {s}");
    }

    #[test]
    fn glossary_with_graph_and_without() {
        let dir = tempfile::tempdir().unwrap();
        // Write a small markdown file so the indexer has content to build a graph from.
        std::fs::write(dir.path().join("note.md"), "# Hello\n\nsome content\n").unwrap();
        glossa::index::store::index_dir(dir.path(), true).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = glossa::graph::store::GraphStore::open(dir.path()).unwrap();
        let trace = TraceLog::disabled();

        // Unknown name -> "(no matches)" when a real graph is present
        let result = exec("glossary", &json!({"name": "zzz-nomatch"}), &idx, Some(&g), &trace).0;
        assert_eq!(result, "(no matches)", "expected no matches, got: {result}");

        // graph = None -> "(graph unavailable)" regardless of args
        let result_no_graph = exec("glossary", &json!({"name": "zzz-nomatch"}), &idx, None, &trace).0;
        assert_eq!(result_no_graph, "(graph unavailable)");
    }
}
