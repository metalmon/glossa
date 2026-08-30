use glossa::index::store::DocIndex;
use glossa::trace::TraceLog;
use regex::Regex;
use serde_json::{json, Value};
use std::collections::{HashSet, VecDeque};
use std::sync::OnceLock;

/// Per-EPISODE retrieval-plateau tracker. A reader that spirals — issuing search after search that
/// each land on already-seen ground — never actually gains information, but the loop's
/// unproductive-streak guard (`unproductive_steer`) resets on ANY single new id, so a spiral of
/// varied slightly-productive probes evades it. This tracker measures a WINDOWED marginal-
/// information-gain PLATEAU instead: it records the number of NEW result-ids each retrieval call
/// surfaces into a sliding window of the last [`RetrievalProgress::W`] calls, and once enough calls
/// have run with a cumulative body of results, fires ONCE when that window's total new-id count
/// falls to (or below) [`RetrievalProgress::G`].
///
/// What it emits is a NEUTRAL OBSERVATION appended to the tool result (see [`RetrievalProgress::
/// observe`]) — a factual "gain has plateaued" marker with counts — NOT an instruction. The POLICY
/// (answer now / stop searching / change approach) is deliberately left to the reader PROMPT / GEPA:
/// baking a directive into the shared tool layer would break its reuse and pre-empt the very thing
/// the prompt is being optimized to decide.
///
/// One tracker lives per EPISODE (one eval case / one train rollout); callers that don't opt in
/// simply never construct one, so the tool result is byte-identical to today. Only RETRIEVAL tool
/// calls (the id-surfacing ones — see [`is_retrieval_tool`]) feed `observe`; write/non-retrieval
/// tools are skipped so their zero-id results don't spuriously drain the window.
#[derive(Debug, Default)]
pub struct RetrievalProgress {
    /// Every distinct result-id seen so far this episode (the cumulative retrieval footprint).
    seen: HashSet<String>,
    /// New-id count for each of the last `W` retrieval calls (front = oldest). A plateau is a
    /// window whose SUM has fallen to `<= G`.
    window: VecDeque<usize>,
    /// True once the plateau marker has been emitted for the CURRENT plateau; reset to false as
    /// soon as a later call surfaces genuinely new ids, so a second, distinct plateau can fire
    /// again rather than the signal being one-shot for the whole episode.
    fired: bool,
    /// Total retrieval calls observed this episode (the `>= M` gate below).
    calls: usize,
}

impl RetrievalProgress {
    /// Sliding-window size: judge the plateau over the last `W` retrieval calls.
    pub const W: usize = 3;
    /// Minimum retrieval calls before a plateau can fire — don't call it early on a reader that
    /// simply hasn't searched much yet.
    pub const M: usize = 4;
    /// Minimum cumulative distinct ids before a plateau can fire — a reader that has surfaced
    /// almost nothing isn't "plateaued", it just hasn't found the corpus yet.
    pub const E: usize = 3;
    /// New-id budget over the window: `sum(window) <= G` is the plateau condition. `0` = zero new
    /// ids across the last `W` calls.
    pub const G: usize = 0;

    pub fn new() -> Self {
        Self::default()
    }

    /// Record one RETRIEVAL call's surfaced `new_ids` and return `Some(marker)` exactly once when
    /// the window-of-last-`W` marginal gain has plateaued (`calls >= M` AND `seen >= E` AND a full
    /// window AND `sum(window) <= G`), else `None`. The marker is a neutral factual observation with
    /// counts — no imperative. Firing flips `fired` so it emits once per plateau; a later call that
    /// brings genuinely new ids clears `fired` so a subsequent plateau can fire again.
    pub fn observe(&mut self, new_ids: &[String]) -> Option<String> {
        self.calls += 1;
        let mut new_count = 0usize;
        for id in new_ids {
            if self.seen.insert(id.clone()) {
                new_count += 1;
            }
        }
        self.window.push_back(new_count);
        if self.window.len() > Self::W {
            self.window.pop_front();
        }
        // Genuinely new ground re-arms the signal for a future, distinct plateau.
        if new_count > 0 {
            self.fired = false;
        }
        let window_sum: usize = self.window.iter().sum();
        let plateaued = !self.fired
            && self.calls >= Self::M
            && self.seen.len() >= Self::E
            && self.window.len() >= Self::W
            && window_sum <= Self::G;
        if plateaued {
            self.fired = true;
            return Some(format!(
                "\n\n[retrieval: {} unique results gathered; {} new over the last {} searches — gain has plateaued]",
                self.seen.len(),
                window_sum,
                self.window.len(),
            ));
        }
        None
    }
}

/// Whether a tool NAME surfaces retrieval result-ids that should feed [`RetrievalProgress`] — the
/// id-producing reader tools (`search`/`glossary`/`related`/`neighbors`/`reach`/`sql`/`read`). Write
/// and non-id tools (`glob`/`grep`/`get_source_file`/`graph_stats`/`resolve`, and any unknown name)
/// are excluded so their zero-id results never spuriously count as "no new" and drain the plateau
/// window. `read` qualifies because its caller substitutes its `path` arg as the surfaced id (see
/// `openai::execute_tool` / `gepa_graph::rollout_one`), even though `exec` returns none for it.
pub fn is_retrieval_tool(name: &str) -> bool {
    matches!(
        name,
        "search" | "glossary" | "related" | "neighbors" | "reach" | "sql" | "read"
    )
}

/// Stable node identifiers surfaced by a graph tool's rendered body, for the unproductive-streak
/// novelty tracker in `openai::run_agent_loop`. Every graph-tool renderer in `glossa::tools`
/// (glossary's main hits AND its chain hops, related, neighbors, reach, and sql's
/// id-column handles) funnels a grounded node through `tools::node_ref`'s anchor in ONE of two
/// forms, both built from the SAME glued `<path>#<ord>` token (`tools.rs:532`):
///   - **entity/reasoning node**: `tools::read_anchor` wraps it as `— read <path>#<ord> · <label>`
///     (only emitted when the node has an outgoing MENTIONS edge to a section).
///   - **structural node (Section/Document)**: `tools::endpoint_ref`/`node_ref` print the anchor
///     BARE, with no "read" word at all — `<path>#<ord> · <label>` for a Section (glossary's
///     exact-title stub at `tools.rs:765`, `edge_line`/neighbors at `tools.rs:1018`,
///     `render_reach_chain` at `tools.rs:1190`), or `<path>  (document)` — no ord — for a Document.
/// The first fix here only matched the "— read" form, so a totally normal move — `neighbors` on a
/// Document returning its child Sections, or several glossary/reach calls landing on different
/// Section/Document nodes — surfaced ZERO ids per call and falsely tripped the streak on a reader
/// making real (structural) progress. The regex below matches the glued `<path>#<ord>` (or the
/// Document's `<path>  (document)`) regardless of whether "— read " precedes it, so both forms count.
/// It's still unambiguous against a glossary chain-hop line (`edge_type  [node_type]  label`, no
/// bare id at all — see `tools::chain_lines`): that line has no `#<ord>` or `(document)` token, so
/// it never matches — a generic `<token>  [Type]` scan would have misread the EDGE TYPE as a
/// stable node id instead, collapsing distinct endpoints reached via the same relation into one
/// false "already seen".
/// The id is `path#ord` (Section) or bare `path` (Document) — the same shape a `read` call's own
/// id takes for the no-`n` case. An ungrounded, non-structural node (no MENTIONS edge, so no
/// anchor at all) contributes nothing from that line; that's fine, the streak only needs SOME
/// calls in a burst to register progress, not every line.
fn extract_node_ids(body: &str) -> Vec<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    // The ordinal branch requires the GLUED `path#n` anchor (zero spaces) — `tools::node_ref`
    // emits exactly this tight form for every real anchor. Tolerating `\s*` here let a node's own
    // label text (e.g. "issue #42") spuriously match as an anchor, causing false novelty for the
    // loop detector; requiring glued-only fixes that. The bare `(document)` anchor is untouched
    // and still requires the wider `\s{2,}` gap so a path token's own text doesn't
    // false-positive into a document match.
    let re = RE.get_or_init(|| {
        Regex::new(r"(\S+?)(?:#(\d+)|\s{2,}\(document\))").expect("valid regex")
    });
    re.captures_iter(body)
        .map(|c| match c.get(2) {
            Some(ord) => format!("{}#{}", &c[1], ord.as_str()),
            None => c[1].to_string(),
        })
        .collect()
}

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
    scope: Option<&str>,
) -> (String, Vec<String>) {
    let (body, hits) = glossa::tools::search(idx, query, limit, glob, file_type, trace, scope);
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

/// Dispatch a tool by name. Returns (result string for the model, ids surfaced for the
/// unproductive-streak novelty tracker, images from read). The ids are search hit locations for
/// `search`, and — for the graph tools (glossary/related/neighbors/reach/sql) —
/// `path#ord` read-anchor ids scraped from the rendered body via [`extract_node_ids`]; `read`
/// itself returns none here (its caller in `openai::execute_tool` uses the `path` arg instead).
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
            let scope = args.get("scope").and_then(|v| v.as_str());
            let (body, titles) = run_search(idx, query, limit, glob, file_type, trace, scope);
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
                scope: args.get("scope").and_then(|v| v.as_str()).map(String::from),
            };
            let (body, titles) = run_grep(root, idx, pattern, opts.with_default_context(), trace);
            (body, titles, Vec::new())
        }
        "glossary" => {
            let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("");
            // Optional: the full question, so the composed neighbourhood ranks by its terms.
            let query = args.get("query").and_then(|v| v.as_str()).filter(|s| !s.is_empty());
            // Loose like the MCP surface: a JSON string or a bare number both work.
            let as_of: Option<String> = args.get("as_of").and_then(|v| {
                glossa::json_util::deserialize_opt_string_loose(v).ok().flatten()
            });
            let body = match graph {
                Some(g) => glossa::tools::glossary_with_query(
                    idx,
                    g,
                    name,
                    query,
                    spec,
                    trace,
                    as_of.as_deref(),
                    None,
                    None,
                ),
                None => "(graph unavailable)".to_string(),
            };
            let ids = extract_node_ids(&body);
            (body, ids, Vec::new())
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
                Some(g) => glossa::tools::related(
                    idx,
                    g,
                    node,
                    path,
                    n,
                    trace,
                    as_of.as_deref(),
                    None,
                    None,
                ),
                None => "(graph unavailable)".to_string(),
            };
            let ids = extract_node_ids(&body);
            (body, ids, Vec::new())
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
                    None,
                ),
                None => "(graph unavailable)".to_string(),
            };
            let ids = extract_node_ids(&body);
            (body, ids, Vec::new())
        }
        "reach" => {
            let from = args
                .get("from")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let from_path = args
                .get("from_path")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let from_n = args.get("from_n").and_then(parse_n);
            let relation = args
                .get("relation")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
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
            let bridge = args.get("bridge").and_then(|v| v.as_bool()).unwrap_or(true);
            let body = match graph {
                Some(g) => {
                    let ont = glossa::graph::ontology::Ontology::load_or_default(root);
                    glossa::tools::reach(
                        idx, g, &ont, from, from_path, from_n, relation, to, to_path, to_n,
                        max_depth, bridge, trace, None,
                    )
                }
                None => "(graph unavailable)".to_string(),
            };
            let ids = extract_node_ids(&body);
            (body, ids, Vec::new())
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
        "sql" => {
            // `sql` is inherently a string (mirrors the real MCP `GraphQueryArgs`); empty/absent
            // returns the schema instead of running a query.
            let sql = args.get("sql").and_then(|v| v.as_str()).unwrap_or("");
            let body = match graph {
                Some(g) => glossa::tools::sql(idx, g, sql, trace),
                None => "(graph unavailable)".to_string(),
            };
            let ids = extract_node_ids(&body);
            (body, ids, Vec::new())
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

/// The default next-best-action: a plain nudge. Used when no text intent can be fanned out — the
/// model is told the repeat is a dead end and to switch tool or query.
pub fn repeat_nudge(name: &str, _args: &Value) -> String {
    format!(
        "(skipped) You already called `{name}` with these exact arguments — its result is above and \
         rerunning returns the same thing. Try a DIFFERENT tool, or change the arguments/query."
    )
}

/// Steer fed back when the reader has strung together `UNPRODUCTIVE_STREAK_K` calls that each
/// surfaced nothing NEW — even though the calls themselves varied (different tool, different
/// args). This is the over-search-spiral case `repeat_nudge`/`next_best_action` don't catch:
/// there's no identical repeat to dedup, just many different probes landing on the same already-seen
/// ground, headed for the round cap and a garbage final answer. Fires once per streak (the caller
/// resets its counter after using this), not on every call past the threshold.
pub fn unproductive_steer(_name: &str) -> String {
    "(no new information) Your last few calls surfaced nothing new — more searching the same way \
     won't help. Commit your best SPECIFIC answer from what you've already found, or change \
     approach fundamentally (a different entity or relation) — do not just rephrase the same query."
        .to_string()
}

/// Next-best-action on a stuck (repeated) call: the model re-issued `name(args)`, so re-running is a
/// dead end. Take the text it fixated on and fan it across the COMPLEMENTARY tools (the ones it did
/// NOT just call), returning their non-empty results fused — concrete alternatives instead of the
/// same dead result. Bounded fan-out; no query reformulation yet (that is the systemic version).
/// Falls back to [`repeat_nudge`] when no text intent exists or nothing complementary comes back.
#[allow(clippy::too_many_arguments)]
pub fn next_best_action(
    name: &str,
    args: &Value,
    root: &std::path::Path,
    idx: &DocIndex,
    graph: Option<&glossa::graph::store::GraphStore>,
    spec: &glossa::tools::ChainSpec,
    trace: &TraceLog,
) -> String {
    let Some(term) = repeated_term(name, args) else {
        return repeat_nudge(name, args);
    };
    // Complementary tools to fan the fixated term across; skip the one just called. glossary +
    // search are the two text lookups; sql turns the term into a relation probe.
    let mut candidates: Vec<(&str, Value)> = Vec::new();
    if name != "search" {
        candidates.push(("search", json!({ "query": term })));
    }
    if name != "glossary" {
        candidates.push(("glossary", json!({ "name": term })));
    }
    if graph.is_some() && name != "sql" {
        let t = term.replace('\'', " ");
        candidates.push((
            "sql",
            json!({
                "sql": format!(
                    "SELECT src_label, edge_type, dst_label FROM edges_labeled \
                     WHERE src_label LIKE '%{t}%' OR dst_label LIKE '%{t}%' LIMIT 12"
                )
            }),
        ));
    }

    let mut out = format!(
        "(skipped) You already called `{name}` twice with the same arguments — that is a dead end. \
         Here is what the OTHER tools return for \"{term}\"; use one of these or change your query:\n"
    );
    let mut any = false;
    for (tool, a) in &candidates {
        let (body, _, _) = exec(tool, a, root, idx, graph, spec, trace);
        let body = body.trim();
        if body.is_empty() || looks_empty(body) {
            continue;
        }
        any = true;
        let snip: String = body.chars().take(600).collect();
        out.push_str(&format!("\n[{tool}]\n{snip}\n"));
    }
    if any {
        out
    } else {
        repeat_nudge(name, args)
    }
}

/// The free-text intent a repeated call fixated on. Ids/paths/SQL give no clean term to fan out, so
/// those fall back to the nudge.
fn repeated_term(name: &str, args: &Value) -> Option<String> {
    let key = match name {
        "glossary" => "name",
        "search" => "query",
        _ => return None,
    };
    let t = args.get(key)?.as_str()?.trim();
    (!t.is_empty()).then(|| t.to_string())
}

/// A tool body that carries no usable result — a miss. Cheap heuristic for v1 (real relevance
/// scoring is the systemic next-best-action's job).
fn looks_empty(body: &str) -> bool {
    let b = body.trim().to_lowercase();
    b.len() < 8
        || b.contains("no matches")
        || b.contains("not found")
        || b.contains("no results")
        || b.contains("(none")
        || b.contains("unavailable")
}

#[cfg(test)]
mod tests {
    use super::*;
    use glossa::index::store::DocIndex;
    use glossa::model::Chunk;
    use glossa::trace::TraceLog;
    use std::path::PathBuf;

    fn ids(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// A productive multi-hop — every call surfaces a genuinely NEW id — must NEVER fire: the
    /// window sum stays >= 1 > G on every call, so `observe` returns `None` throughout, even well
    /// past the M-call and E-id thresholds.
    #[test]
    fn plateau_never_fires_on_a_productive_multihop() {
        let mut p = RetrievalProgress::new();
        for i in 0..12 {
            let id = format!("doc.md#{i}");
            assert_eq!(
                p.observe(&[id.clone()]),
                None,
                "a call that adds a new id ({id}) must not plateau"
            );
        }
    }

    /// The plateau fires only once `calls >= M`, `seen >= E`, and the last-`W` window sum `<= G`
    /// all hold. Call 1 seeds E distinct ids; calls 2..=M add nothing, so by call M the window is
    /// all-zero and the marker fires — not before.
    #[test]
    fn plateau_fires_once_thresholds_met_not_before() {
        let mut p = RetrievalProgress::new();
        // Call 1: E distinct ids in one shot -> seen >= E, but calls < M so no fire yet.
        assert_eq!(p.observe(&ids(&["a", "b", "c"])), None, "call 1: below M");
        // Calls 2..M-1: nothing new, still below the M-call gate.
        for _ in 0..(RetrievalProgress::M - 2) {
            assert_eq!(p.observe(&[]), None, "no-new call below M must stay quiet");
        }
        // Call M: window is now all-zero over the last W calls, seen >= E, calls == M -> FIRE.
        let marker = p
            .observe(&[])
            .expect("plateau must fire once M/E/window-sum thresholds are all met");
        assert!(marker.contains("plateaued"), "marker: {marker}");
        assert!(marker.contains("3 unique results gathered"), "counts in marker: {marker}");
    }

    /// Fires ONCE per plateau: after firing it stays quiet on further no-new calls, then a call
    /// that brings genuinely new ids re-arms it so a SECOND, distinct plateau fires again.
    #[test]
    fn plateau_fires_once_then_rearms_on_new_ids() {
        let mut p = RetrievalProgress::new();
        p.observe(&ids(&["a", "b", "c"])); // seed E ids (call 1)
        for _ in 0..(RetrievalProgress::M - 2) {
            p.observe(&[]);
        }
        assert!(p.observe(&[]).is_some(), "first plateau fires at call M");
        // Still plateaued, but already fired -> silent.
        assert_eq!(p.observe(&[]), None, "second no-new call must not re-fire the same plateau");
        assert_eq!(p.observe(&[]), None, "still silent while plateaued");
        // A genuinely new id re-arms the signal and refills the window with a non-zero count.
        assert_eq!(p.observe(&ids(&["d"])), None, "new id: window sum > G, no fire, re-armed");
        // Drain the window back to all-zero over W calls -> a fresh plateau fires again.
        for _ in 0..(RetrievalProgress::W - 1) {
            assert_eq!(p.observe(&[]), None, "window not yet fully drained");
        }
        assert!(p.observe(&[]).is_some(), "a distinct later plateau must fire again");
    }

    /// The marker is a NEUTRAL observation: it states the counts and that gain has plateaued, and
    /// carries none of the imperative/policy words that belong in the prompt (that's GEPA's job).
    #[test]
    fn plateau_marker_is_a_neutral_observation_not_a_directive() {
        let mut p = RetrievalProgress::new();
        p.observe(&ids(&["a", "b", "c"]));
        for _ in 0..(RetrievalProgress::M - 2) {
            p.observe(&[]);
        }
        let marker = p.observe(&[]).expect("plateau fires").to_lowercase();
        for imperative in ["stop", "answer", "must", "commit", "should", "change approach", "give up"] {
            assert!(
                !marker.contains(imperative),
                "neutral marker must not contain the directive word {imperative:?}: {marker}"
            );
        }
    }

    /// `is_retrieval_tool` gates which tools feed the tracker: the id-surfacing reader tools do,
    /// write/non-id tools don't (so their zero-id results can't drain the plateau window).
    #[test]
    fn is_retrieval_tool_selects_id_surfacing_tools() {
        for t in ["search", "glossary", "related", "neighbors", "reach", "sql", "read"] {
            assert!(is_retrieval_tool(t), "{t} surfaces ids");
        }
        for t in ["glob", "grep", "get_source_file", "graph_stats", "resolve", "unknown"] {
            assert!(!is_retrieval_tool(t), "{t} must not feed the tracker");
        }
    }

    #[test]
    fn repeated_term_extracts_text_intent_only() {
        assert_eq!(
            repeated_term("glossary", &json!({"name":"Acme"})).as_deref(),
            Some("Acme")
        );
        assert_eq!(
            repeated_term("search", &json!({"query":"blue widget"})).as_deref(),
            Some("blue widget")
        );
        assert_eq!(repeated_term("read", &json!({"path":"x.md","n":1})), None);
        assert_eq!(repeated_term("sql", &json!({"sql":"SELECT 1"})), None);
        assert_eq!(repeated_term("glossary", &json!({"name":"   "})), None);
    }

    /// Regression guard for fix round 2: the entity-node "— read" anchor and the BARE structural
    /// (Section/Document) anchor must both be captured — the first version of this regex only
    /// matched the "— read" form, so a Section/Document endpoint (rendered by `endpoint_ref`/
    /// `node_ref` with no "read" word at all) surfaced zero ids.
    #[test]
    fn extract_node_ids_matches_entity_anchor_and_bare_structural_forms() {
        // Entity/reasoning node: `tools::read_anchor`'s "— read <path>#<ord> · <label>" (glued,
        // no space — the real form `tools::node_ref` emits).
        let entity_line = "n1  [Entity]  Some Fact   — read doc.md#3 · SecTitle";
        assert_eq!(extract_node_ids(entity_line), vec!["doc.md#3".to_string()]);

        // Bare structural Section endpoint (edge_line / render_reach_chain / glossary's
        // exact-title stub) — no "read" word, just the raw `node_ref` anchor.
        let section_line = "CONTAINS       ->  doc.md#2 · SecB";
        assert_eq!(extract_node_ids(section_line), vec!["doc.md#2".to_string()]);

        // Bare structural Document endpoint — no ord at all.
        let doc_line = "doc.md  (document)";
        assert_eq!(extract_node_ids(doc_line), vec!["doc.md".to_string()]);

        // A line with neither anchor form contributes nothing.
        assert!(extract_node_ids("REFERENCES      ->  fact-9  [Entity]  no anchor here").is_empty());

        // A single space before `#n` inside a node's own label text (NOT a real glued anchor)
        // must NOT spuriously match — this is the false-novelty bug the tightened regex fixes.
        assert!(
            extract_node_ids("n2  [Entity]  reported issue #42 with no read anchor").is_empty(),
            "a spaced '#n' inside label text must not be mistaken for a glued anchor"
        );

        // Multiple anchors on one body (e.g. several neighbors lines) all extract.
        let multi = format!("{section_line}\n{entity_line}\n{doc_line}");
        let mut ids = extract_node_ids(&multi);
        ids.sort();
        let mut want = vec!["doc.md".to_string(), "doc.md#2".to_string(), "doc.md#3".to_string()];
        want.sort();
        assert_eq!(ids, want);
    }

    #[test]
    fn looks_empty_flags_misses_not_hits() {
        assert!(looks_empty("no matches"));
        assert!(looks_empty("(graph unavailable)"));
        assert!(looks_empty(""));
        assert!(!looks_empty(
            "a fully grounded fact statement with real content here"
        ));
    }

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
        assert!(out.contains("d.pdf#7:"), "carries path#n read key: {out}");
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
            glossa::tools::related(&idx, &g, None, Some(&path), Some(1), &trace, None, None, None);
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

    /// `sql` dispatch: empty `sql` returns the schema (mirrors the real MCP tool's
    /// "empty query returns the schema" contract); no graph falls back to the same
    /// "(graph unavailable)" text every other graph tool arm returns.
    #[test]
    fn graph_query_dispatch_empty_sql_returns_schema() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = glossa::graph::store::GraphStore::open(dir.path()).unwrap();
        let trace = TraceLog::disabled();

        let with_graph = exec(
            "sql",
            &json!({"sql": ""}),
            dir.path(),
            &idx,
            Some(&g),
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert!(!with_graph.is_empty(), "empty sql should return the schema help text");
        assert!(with_graph.contains("sql"), "got: {with_graph}");

        let no_graph = exec(
            "sql",
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

    /// Finding 2: `reach`'s `max_depth` must accept a numeric string ("9"), not just a JSON
    /// integer. Build a chain longer than the default depth (6) so the test only passes when the
    /// string is actually parsed rather than silently falling back to the default.
    #[test]
    fn reach_max_depth_accepts_numeric_string() {
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
            "reach",
            &json!({"from": "n0", "to": "n7"}),
            dir.path(),
            &idx,
            Some(&g),
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert!(
            default_depth.starts_with("no grounded path"),
            "expected default depth to be too shallow: {default_depth}"
        );

        // A numeric-string max_depth must be parsed (not dropped to the default 6).
        let string_depth = exec(
            "reach",
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

    /// `tools_schema(true)`/`exec` parity guard (mirrors [[mcp-tool-add-rename-eval-sites]]):
    /// `reach` must dispatch to a real result, and the removed `path` name must fall through to
    /// "unknown tool" rather than silently misrouting.
    #[test]
    fn reach_dispatches_and_path_is_unknown() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = glossa::graph::store::GraphStore::open(dir.path()).unwrap();
        g.put_node(&gnode("n0")).unwrap();
        g.put_node(&gnode("n1")).unwrap();
        g.put_edge(&gedge("n0", "REFERENCES", "n1")).unwrap();
        let trace = TraceLog::disabled();

        let reach_out = exec(
            "reach",
            &json!({"from": "n0", "to": "n1"}),
            dir.path(),
            &idx,
            Some(&g),
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert!(reach_out.contains("--REFERENCES-->"), "got: {reach_out}");

        let path_out = exec(
            "path",
            &json!({"from": "n0", "to": "n1"}),
            dir.path(),
            &idx,
            Some(&g),
            &glossa::tools::ChainSpec::default(),
            &trace,
        )
        .0;
        assert!(path_out.starts_with("unknown tool"), "got: {path_out}");
    }
}
