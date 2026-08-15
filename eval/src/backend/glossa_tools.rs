use glossa::index::store::DocIndex;
use glossa::trace::TraceLog;
use serde_json::{json, Value};

/// Run a BM25 search (optionally scoped by path glob / file_type); model-facing numbered text + titles.
///
/// Takes a borrowed `DocIndex` so the caller opens it once per question and reuses it (with its
/// cached reader) across every search/read in the episode, instead of reopening per tool call.
pub fn run_search(
    idx: &DocIndex,
    query: &str,
    limit: usize,
    glob: Option<&str>,
    file_type: Option<&str>,
    trace: &TraceLog,
) -> (String, Vec<String>) {
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
    if let Some(n) = v.as_u64() {
        return Some(n);
    }
    let s: String = v.as_str()?.chars().filter(|c| c.is_ascii_digit()).collect();
    s.parse::<u64>().ok()
}

/// Read a chunk OR a reasoning node: full text + the chunk's images (for the vision model). `graph`
/// makes `read` omnivorous — a node id resolves to the node + its evidence chunks. `root` is the
/// corpus/notebook root (feature-gated: a doc-scoped notebook path is served from there).
/// When `page_image` is true (PDF only), returns a page raster instead of chunk text/embeds.
pub fn run_read(
    root: &std::path::Path,
    idx: &DocIndex,
    graph: Option<&glossa::graph::store::GraphStore>,
    path: &str,
    n: u64,
    page_image: bool,
    trace: &TraceLog,
) -> (String, Vec<glossa::read::DocImage>) {
    let out = glossa::tools::read(root, idx, graph, path, n, page_image, trace);
    (out.text, out.images)
}

/// Run a ripgrep-style exact/regex search over the extracted text; one line per match `path:#n: line`.
/// `root` is the corpus/notebook root (feature-gated: a doc-scoped notebook path is served from there).
pub fn run_grep(
    root: &std::path::Path,
    idx: &DocIndex,
    pattern: &str,
    opts: glossa::grep::GrepOpts,
    trace: &TraceLog,
) -> (String, Vec<String>) {
    (
        glossa::tools::grep(root, idx, pattern, &opts, trace),
        Vec::new(),
    )
}

/// Dispatch a tool by name. Returns (result string for the model, titles surfaced by a search, images from read).
/// `root` is the corpus/notebook root, threaded through to `read` for notebook-file serving.
pub fn exec(
    name: &str,
    args: &Value,
    root: &std::path::Path,
    idx: &DocIndex,
    graph: Option<&glossa::graph::store::GraphStore>,
    spec: &glossa::tools::ChainSpec,
    trace: &TraceLog,
) -> (String, Vec<String>, Vec<glossa::read::DocImage>) {
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
            let page_image = args
                .get("page_image")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let (text, imgs) = run_read(root, idx, graph, path, n, page_image, trace);
            (text, Vec::new(), imgs)
        }
        "grep" => {
            let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
            let path_arg = args.get("path").and_then(|v| v.as_str()).map(String::from);
            let usize_arg = |k: &str| args.get(k).and_then(|v| v.as_u64()).map(|n| n as usize);
            let bool_arg = |k: &str| args.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
            let context = usize_arg("context");
            let opts = glossa::grep::GrepOpts {
                ignore_case: bool_arg("ignore_case"),
                fixed: bool_arg("fixed"),
                word: bool_arg("word"),
                // Explicit `glob` wins; otherwise `path` scopes the search to that one document.
                glob: args
                    .get("glob")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .or_else(|| path_arg.as_deref().map(glossa::grep::path_to_glob)),
                file_type: args
                    .get("file_type")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                // -A/-B override the shared -C on their respective side.
                before: usize_arg("before").or(context).unwrap_or(0),
                after: usize_arg("after").or(context).unwrap_or(0),
                only_matching: bool_arg("only_matching"),
                line_number: bool_arg("line_number"),
                count: bool_arg("count"),
                max_count: usize_arg("max_count"),
                multiline: bool_arg("multiline"),
                line_cap: None,
                path: path_arg,
            };
            let (body, titles) = run_grep(root, idx, pattern, opts.with_default_context(), trace);
            (body, titles, Vec::new())
        }
        "glossary" => {
            let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("");
            // Loose like the MCP surface: a JSON string or a bare number both work.
            let as_of: Option<String> = args.get("as_of").and_then(|v| {
                glossa::json_util::deserialize_opt_string_loose(v).ok().flatten()
            });
            let body = match graph {
                Some(g) => {
                    glossa::tools::glossary(idx, g, name, spec, trace, as_of.as_deref(), None)
                }
                None => "(graph unavailable)".to_string(),
            };
            (body, Vec::new(), Vec::new())
        }
        "related" => {
            let node = args
                .get("node")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let path = args
                .get("path")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let n = args.get("n").and_then(parse_n);
            let as_of: Option<String> = args.get("as_of").and_then(|v| {
                glossa::json_util::deserialize_opt_string_loose(v).ok().flatten()
            });
            let body = match graph {
                Some(g) => {
                    glossa::tools::related(idx, g, node, path, n, trace, as_of.as_deref(), None)
                }
                None => "(graph unavailable)".to_string(),
            };
            (body, Vec::new(), Vec::new())
        }
        "neighbors" => {
            let node = args
                .get("node")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let path = args
                .get("path")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let n = args.get("n").and_then(parse_n);
            // Loose like the MCP surface: a JSON array, a single string, or a comma-separated
            // string all resolve to the same Vec<String> (some models send `edge_types:"REFERENCES"`).
            let edge_types: Option<Vec<String>> = args.get("edge_types").and_then(|v| {
                glossa::json_util::deserialize_opt_vec_string_loose(v).ok().flatten()
            });
            let direction = args.get("direction").and_then(|v| v.as_str()).unwrap_or("both");
            let as_of: Option<String> = args.get("as_of").and_then(|v| {
                glossa::json_util::deserialize_opt_string_loose(v).ok().flatten()
            });
            let body = match graph {
                Some(g) => glossa::tools::neighbors(
                    idx,
                    g,
                    node,
                    path,
                    n,
                    edge_types.as_deref(),
                    direction,
                    trace,
                    as_of.as_deref(),
                    None,
                ),
                None => "(graph unavailable)".to_string(),
            };
            (body, Vec::new(), Vec::new())
        }
        "path" => {
            let from = args
                .get("from")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let from_path = args
                .get("from_path")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let from_n = args.get("from_n").and_then(parse_n);
            let to = args
                .get("to")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let to_path = args
                .get("to_path")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let to_n = args.get("to_n").and_then(parse_n);
            // Loose like from_n/to_n above: accepts a numeric string ("9") as well as an integer.
            let max_depth = args
                .get("max_depth")
                .and_then(parse_n)
                .map(|n| n as usize)
                .unwrap_or(6);
            let body = match graph {
                Some(g) => glossa::tools::path_between(
                    idx, g, from, from_path, from_n, to, to_path, to_n, max_depth, trace,
                ),
                None => "(graph unavailable)".to_string(),
            };
            (body, Vec::new(), Vec::new())
        }
        "get_source_file" => {
            // Mirrors src/mcp.rs's `get_source_file` handler, minus the delivered file bytes:
            // eval has no ZeroClaw/ACP client downstream to receive `SourceFileOut::file`, so we
            // return only the model-facing provenance/error text (`SourceFileOut::text`) — that's
            // the same string the model would read off the real MCP tool result.
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let n = args.get("n").and_then(parse_n);
            let max_bytes = args
                .get("max_bytes")
                .and_then(parse_n)
                .unwrap_or(glossa::tools::DEFAULT_SOURCE_MAX_BYTES);
            let raw = args.get("raw").and_then(|v| v.as_bool()).unwrap_or(false);
            let out = glossa::tools::get_source_file(idx, graph, path, n, max_bytes, raw);
            (out.text, Vec::new(), Vec::new())
        }
        "graph_stats" => {
            let body = match graph {
                Some(g) => glossa::tools::graph_stats(g),
                None => "(graph unavailable)".to_string(),
            };
            (body, Vec::new(), Vec::new())
        }
        "graph_query" => {
            // `sql` is inherently a string (mirrors the real MCP `GraphQueryArgs`); empty/absent
            // returns the schema instead of running a query.
            let sql = args.get("sql").and_then(|v| v.as_str()).unwrap_or("");
            let body = match graph {
                Some(g) => glossa::tools::graph_query(idx, g, sql, trace),
                None => "(graph unavailable)".to_string(),
            };
            (body, Vec::new(), Vec::new())
        }
        "resolve" => {
            // entity resolution — a Reader tool, so present in EVERY profile; both answer_hotpot
            // and enrich can call it. Without this branch it fell through to "unknown tool".
            let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let body = match graph {
                Some(g) => match g.resolve(name) {
                    Ok(ids) => ids.join("\n"),
                    Err(e) => format!("resolve error: {e}"),
                },
                None => "(graph unavailable)".to_string(),
            };
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
        idx.write_chunks(&[Chunk {
            doc_path: PathBuf::from("d.pdf"),
            location: "p.7".into(),
            file_type: "pdf".into(),
            text: "seventh page".into(),
        }])
        .unwrap();
        let trace = TraceLog::disabled();

        // integer n
        let out = exec(
            "read",
            &json!({"path": "d.pdf", "n": 7}),
            dir.path(),
            &idx,
            None,
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert!(out.contains("seventh"), "got: {out}");
        // stray string "p.7" -> digit-strip fallback -> 7
        let out2 = exec(
            "read",
            &json!({"path": "d.pdf", "n": "p.7"}),
            dir.path(),
            &idx,
            None,
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert!(out2.contains("seventh"), "digit-strip fallback: {out2}");
    }

    #[test]
    fn grep_tool_finds_exact_token_via_exec() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[Chunk {
            doc_path: PathBuf::from("d.pdf"),
            location: "p.7".into(),
            file_type: "pdf".into(),
            text: "parameter maxTsdr equals 3000".into(),
        }])
        .unwrap();
        let trace = TraceLog::disabled();
        let out = exec(
            "grep",
            &json!({"pattern": "maxTsdr"}),
            dir.path(),
            &idx,
            None,
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert!(out.contains("maxTsdr"), "got: {out}");
        assert!(out.contains(":#7:"), "carries #n read key: {out}");
    }

    #[test]
    fn grep_path_arg_scopes_to_one_document() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            Chunk {
                doc_path: PathBuf::from("a.pdf"),
                location: "p.1".into(),
                file_type: "pdf".into(),
                text: "Density F24".into(),
            },
            Chunk {
                doc_path: PathBuf::from("b.pdf"),
                location: "p.1".into(),
                file_type: "pdf".into(),
                text: "Density F60".into(),
            },
        ])
        .unwrap();
        let trace = TraceLog::disabled();
        let out = exec(
            "grep",
            &json!({"pattern": "Density", "path": "a.pdf#1"}),
            dir.path(),
            &idx,
            None,
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert!(out.contains("a.pdf"), "scoped hit present: {out}");
        assert!(!out.contains("b.pdf"), "other document excluded: {out}");
    }

    #[test]
    fn glob_and_scoped_search_via_exec() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            Chunk {
                doc_path: PathBuf::from("TEMPLATE.pdf"),
                location: "p.1".into(),
                file_type: "pdf".into(),
                text: "hot swap".into(),
            },
            Chunk {
                doc_path: PathBuf::from("Other.pdf"),
                location: "p.1".into(),
                file_type: "pdf".into(),
                text: "hot swap".into(),
            },
        ])
        .unwrap();
        let trace = TraceLog::disabled();
        let g = exec(
            "glob",
            &json!({"pattern": "*TEMPLATE*"}),
            dir.path(),
            &idx,
            None,
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert!(g.contains("TEMPLATE") && !g.contains("Other"), "glob: {g}");
        let s = exec(
            "search",
            &json!({"query": "swap", "glob": "*TEMPLATE*"}),
            dir.path(),
            &idx,
            None,
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert!(
            s.contains("TEMPLATE") && !s.contains("Other"),
            "scoped search: {s}"
        );
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
        let result = exec(
            "glossary",
            &json!({"name": "zzz-nomatch"}),
            dir.path(),
            &idx,
            Some(&g),
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert_eq!(result, "(no matches)", "expected no matches, got: {result}");

        // graph = None -> "(graph unavailable)" regardless of args
        let result_no_graph = exec(
            "glossary",
            &json!({"name": "zzz-nomatch"}),
            dir.path(),
            &idx,
            None,
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert_eq!(result_no_graph, "(graph unavailable)");
    }

    /// Parity: MCP surface (tools::related directly) and eval surface (exec dispatch) must
    /// produce identical output for the same (idx, graph, path, n). Both call the shared fn.
    #[test]
    fn related_mcp_and_eval_parity() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("p.md"), "# Alpha\nintro\n## Beta\nbody\n").unwrap();
        glossa::index::store::index_dir(dir.path(), true).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = glossa::graph::store::GraphStore::open(dir.path()).unwrap();
        let trace = TraceLog::disabled();
        let path = "p.md".to_string(); // canonical key: corpus-root-relative

        // MCP path: call shared fn directly (same call as src/mcp.rs handler).
        let mcp_out =
            glossa::tools::related(&idx, &g, None, Some(&path), Some(1), &trace, None, None);
        // Eval path: dispatch through exec().
        let eval_out = exec(
            "related",
            &json!({"path": path, "n": 1}),
            dir.path(),
            &idx,
            Some(&g),
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;

        assert_eq!(
            mcp_out, eval_out,
            "MCP and eval surfaces must render identically"
        );
        // This plain corpus has no reasoning graph, so there are no SIMILAR cross-links.
        assert_eq!(
            mcp_out, "(no related cases)",
            "generalization neighbors on a graph-less corpus: {mcp_out}"
        );
    }

    /// `get_source_file` was advertised in the generated TZ tool lists but had no `exec` arm
    /// (fell through to "unknown tool"). Confirm it now resolves a real corpus file and mirrors
    /// the MCP handler's model-facing provenance text.
    #[test]
    fn get_source_file_via_exec_delivers_provenance_text() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("note.txt"), b"hello source file").unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[Chunk {
            doc_path: PathBuf::from("note.txt"),
            location: "1".into(),
            file_type: "txt".into(),
            text: "hello source file".into(),
        }])
        .unwrap();
        let trace = TraceLog::disabled();

        let out = exec(
            "get_source_file",
            &json!({"path": "note.txt"}),
            dir.path(),
            &idx,
            None,
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert!(
            out.contains("delivered whole file: note.txt"),
            "got: {out}"
        );

        // Unknown path -> the same not-found provenance text `get_source_file` returns directly
        // (not "unknown tool: get_source_file", which is what the missing arm used to produce).
        let missing = exec(
            "get_source_file",
            &json!({"path": "does-not-exist.pdf"}),
            dir.path(),
            &idx,
            None,
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert!(
            !missing.starts_with("unknown tool"),
            "must dispatch, not fall through: {missing}"
        );
    }

    fn prov() -> glossa::graph::store::Provenance {
        glossa::graph::store::Provenance {
            source_path: "p.md".into(),
            range: None,
            file_sig: None,
            origin: "agent".into(),
            confidence: 0.8,
            created_at: 1,
        }
    }
    fn gnode(id: &str) -> glossa::graph::store::Node {
        glossa::graph::store::Node {
            id: id.into(),
            node_type: "Entity".into(),
            label: id.into(),
            aliases: Vec::new(),
            prov: prov(),
        }
    }
    fn gedge(from: &str, rel: &str, to: &str) -> glossa::graph::store::Edge {
        glossa::graph::store::Edge {
            from: from.into(),
            to: to.into(),
            edge_type: rel.into(),
            prov: prov(),
        }
    }

    /// Finding 2: `neighbors`' `edge_types` must accept a single string or a comma-separated
    /// string, not just a JSON array — some models send `edge_types:"CONSTRAINED_BY"`. Before the
    /// fix, `.as_array()` silently dropped that filter (both edge types would show up).
    #[test]
    fn neighbors_edge_types_accepts_single_string_and_csv() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = glossa::graph::store::GraphStore::open(dir.path()).unwrap();
        for id in ["a", "b", "c"] {
            g.put_node(&gnode(id)).unwrap();
        }
        g.put_edge(&gedge("a", "REFERENCES", "b")).unwrap();
        g.put_edge(&gedge("c", "CONSTRAINED_BY", "a")).unwrap();
        let trace = TraceLog::disabled();

        let array_form = exec(
            "neighbors",
            &json!({"node": "a", "edge_types": ["CONSTRAINED_BY"]}),
            dir.path(),
            &idx,
            Some(&g),
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert!(
            array_form.contains("CONSTRAINED_BY") && !array_form.contains("REFERENCES"),
            "got: {array_form}"
        );

        let single_string = exec(
            "neighbors",
            &json!({"node": "a", "edge_types": "CONSTRAINED_BY"}),
            dir.path(),
            &idx,
            Some(&g),
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert_eq!(single_string, array_form, "single string must filter identically to array");

        let csv_string = exec(
            "neighbors",
            &json!({"node": "a", "edge_types": "CONSTRAINED_BY, NOPE"}),
            dir.path(),
            &idx,
            Some(&g),
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert_eq!(csv_string, array_form, "comma-separated string must filter identically");
    }

    /// `graph_query` dispatch: empty `sql` returns the schema (mirrors the real MCP tool's
    /// "empty query returns the schema" contract); no graph falls back to the same
    /// "(graph unavailable)" text every other graph tool arm returns.
    #[test]
    fn graph_query_dispatch_empty_sql_returns_schema() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = glossa::graph::store::GraphStore::open(dir.path()).unwrap();
        let trace = TraceLog::disabled();

        let with_graph = exec(
            "graph_query",
            &json!({"sql": ""}),
            dir.path(),
            &idx,
            Some(&g),
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert!(!with_graph.is_empty(), "empty sql should return the schema help text");
        assert!(with_graph.contains("graph_query"), "got: {with_graph}");

        let no_graph = exec(
            "graph_query",
            &json!({"sql": ""}),
            dir.path(),
            &idx,
            None,
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert_eq!(no_graph, "(graph unavailable)");
    }

    /// Finding 2: `path`'s `max_depth` must accept a numeric string ("9"), not just a JSON
    /// integer. Build a chain longer than the default depth (6) so the test only passes when the
    /// string is actually parsed rather than silently falling back to the default.
    #[test]
    fn path_max_depth_accepts_numeric_string() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = glossa::graph::store::GraphStore::open(dir.path()).unwrap();
        let ids = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7"]; // 7 hops, over the default cap of 6
        for id in ids {
            g.put_node(&gnode(id)).unwrap();
        }
        for w in ids.windows(2) {
            g.put_edge(&gedge(w[0], "REFERENCES", w[1])).unwrap();
        }
        let trace = TraceLog::disabled();

        // Default max_depth (6) is too shallow for a 7-hop chain.
        let default_depth = exec(
            "path",
            &json!({"from": "n0", "to": "n7"}),
            dir.path(),
            &idx,
            Some(&g),
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert!(
            default_depth.starts_with("no path"),
            "expected default depth to be too shallow: {default_depth}"
        );

        // A numeric-string max_depth must be parsed (not dropped to the default 6).
        let string_depth = exec(
            "path",
            &json!({"from": "n0", "to": "n7", "max_depth": "10"}),
            dir.path(),
            &idx,
            Some(&g),
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert!(
            string_depth.contains("--REFERENCES-->"),
            "numeric-string max_depth must extend the search: {string_depth}"
        );
    }
}
