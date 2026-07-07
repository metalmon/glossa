use crate::graph::ontology::Ontology;
use crate::graph::store::GraphStore;
use crate::index::store::index_dir;
use base64::Engine as _;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, ProtocolVersion, ServerCapabilities, ServerInfo};
use rmcp::{tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler};
use schemars::JsonSchema;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    Reader,
    Editor,
    Full,
}

impl Profile {
    pub fn parse(s: &str) -> Profile {
        match s {
            "editor" => Profile::Editor,
            "full" => Profile::Full,
            _ => Profile::Reader,
        }
    }
}

#[derive(Clone)]
pub struct GlossaServer {
    root: PathBuf,
    tool_router: ToolRouter<Self>,
    trace: crate::trace::TraceLog,
    /// Epoch-ms of the last file-first freshness scan, shared across cloned handlers. Throttles
    /// `ensure_fresh` so bursty tool calls don't each re-scan the corpus (see `kick_freshen`).
    last_fresh: Arc<AtomicU64>,
    /// Set when background freshen reindexed something — derived layer (closure/SIMILAR +
    /// community/centrality) is stale until debounced `generalize` runs (off the read hot path).
    dirty: Arc<AtomicBool>,
    /// Epoch-ms of the last indexing change — the debounce clock for the maintenance loop.
    last_change: Arc<AtomicU64>,
    /// True while a background `ensure_fresh` is running — concurrent `kick_freshen` calls bail out.
    indexing: Arc<AtomicBool>,
}

#[allow(dead_code)] // read tools stay enabled for Reader; listed for profile documentation
const NOTEBOOK_READ_TOOLS: &[&str] = &["ls", "cat"];
const NOTEBOOK_WRITE_TOOLS: &[&str] = &["note", "sed", "del"];
const EDITOR_TOOLS: &[&str] = &[
    "index",
    "reindex",
    "graph_upsert",
    "graph_delete",
    "graph_update",
    "graph_generalize",
    "graph_stats",
];
const FULL_TOOLS: &[&str] = &["purge"];
const GRAPH_TOOLS: &[&str] = &[
    "glossary",
    "neighbors",
    "graph_upsert",
    "graph_delete",
    "graph_update",
    "graph_generalize",
    "resolve",
    "index",
    "reindex",
    "purge",
];

impl GlossaServer {
    pub fn new(root: PathBuf, profile: Profile, trace: bool, no_graph: bool) -> Self {
        let mut router = Self::tool_router();
        if profile == Profile::Reader {
            for t in EDITOR_TOOLS
                .iter()
                .chain(FULL_TOOLS)
                .chain(NOTEBOOK_WRITE_TOOLS)
            {
                router.disable_route(*t);
            }
        } else if profile == Profile::Editor {
            for t in FULL_TOOLS {
                router.disable_route(*t);
            }
        }
        if no_graph {
            for t in GRAPH_TOOLS {
                router.disable_route(*t);
            }
        }
        #[cfg(not(feature = "notebook"))]
        {
            for t in NOTEBOOK_READ_TOOLS.iter().chain(NOTEBOOK_WRITE_TOOLS) {
                router.disable_route(*t);
            }
        }
        let trace = if trace {
            crate::trace::TraceLog::to_dir(&root)
        } else {
            crate::trace::TraceLog::disabled()
        };
        // Seed `last_fresh` to "now" so bursty first calls don't each scan — `run_serve` kicks one
        // background freshen after the transport is up. Keeps `new()` free of filesystem access.
        Self {
            root,
            tool_router: router,
            trace,
            last_fresh: Arc::new(AtomicU64::new(crate::trace::now_ms())),
            dirty: Arc::new(AtomicBool::new(false)),
            last_change: Arc::new(AtomicU64::new(0)),
            indexing: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Return the list of enabled tools (for config generation — not test-only).
    pub fn tool_specs(&self) -> Vec<rmcp::model::Tool> {
        self.tool_router.list_all()
    }

    /// File-first freshness off the read hot path: schedule `ensure_fresh` in the background when
    /// the throttle window allows. Read tools call this (no await) and serve the current on-disk
    /// index immediately. Best-effort: indexing errors never fail the tool.
    pub fn kick_freshen(&self) {
        const FRESH_WINDOW_MS: u64 = 1500;
        if self.indexing.load(Ordering::Acquire) {
            return;
        }
        let now = crate::trace::now_ms();
        let last = self.last_fresh.load(Ordering::Relaxed);
        if now.saturating_sub(last) < FRESH_WINDOW_MS {
            return;
        }
        if self
            .last_fresh
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return;
        }
        if self
            .indexing
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        let root = self.root.clone();
        let dirty = self.dirty.clone();
        let last_change = self.last_change.clone();
        let indexing = self.indexing.clone();
        tokio::spawn(async move {
            let result =
                tokio::task::spawn_blocking(move || crate::index::store::ensure_fresh(&root)).await;
            indexing.store(false, Ordering::Release);
            if let Ok(Ok(stats)) = result {
                if stats.added + stats.removed > 0 {
                    last_change.store(crate::trace::now_ms(), Ordering::Relaxed);
                    dirty.store(true, Ordering::Relaxed);
                }
            }
        });
    }

    /// Mark the derived graph layer stale (a freshen reindexed something) and stamp the change time —
    /// the debounce clock the maintenance loop waits on before running `generalize`.
    pub fn mark_dirty(&self) {
        self.last_change
            .store(crate::trace::now_ms(), Ordering::Relaxed);
        self.dirty.store(true, Ordering::Relaxed);
    }

    /// Debounce decision: run the generalize pass only when the graph is `dirty` AND no further
    /// indexing change has landed for at least `debounce_ms` (the corpus has settled). Pure so it is
    /// unit-testable without timers.
    fn maintenance_due(dirty: bool, last_change_ms: u64, now_ms: u64, debounce_ms: u64) -> bool {
        dirty && now_ms.saturating_sub(last_change_ms) >= debounce_ms
    }

    /// Recompute the DERIVED graph layer (closure/SIMILAR edges + community/centrality) from what is
    /// currently stored — the same non-destructive pass as the `graph_generalize` tool. Best-effort.
    /// Cross-process singleton: multiple editor instances are expected, so the pass is guarded by an
    /// advisory try-lock on `.glossa/generalize.lock`. If another editor holds it, this round is
    /// skipped (the holder refreshes the shared graph for everyone). The lock releases when `_lock`
    /// drops (function exit / process death).
    fn run_generalize(&self) {
        use fs4::fs_std::FileExt;
        let lock_path = self.root.join(".glossa").join("generalize.lock");
        let Ok(_lock) = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
        else {
            return;
        };
        match _lock.try_lock_exclusive() {
            Ok(true) => {} // acquired — we are the one editor running the pass this round
            _ => return,   // held by another editor (false) or lock error → skip
        }
        let Ok(g) = GraphStore::open(&self.root) else {
            return;
        };
        let ont = Ontology::load_or_default(&self.root);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let _ = crate::graph::ops::graph_generalize(&g, &ont, now);
        let _ = FileExt::unlock(&_lock);
    }

    /// Readiness probe: the index and graph at `root` can be opened (the server can serve). Backs the
    /// streamable-http `/ready` endpoint.
    pub fn readiness(&self) -> bool {
        crate::index::store::DocIndex::open_or_create(&self.root).is_ok()
            && GraphStore::open(&self.root).is_ok()
    }

    /// Prometheus text-exposition metrics for `/metrics`. Cheap, computed at scrape time: index size,
    /// graph size, and whether the derived layer is stale. (Request-rate/latency are left to the HTTP
    /// access log / gateway.)
    pub fn metrics_text(&self) -> String {
        let chunks = crate::index::store::DocIndex::open_or_create(&self.root)
            .ok()
            .and_then(|idx| idx.index.reader().ok().map(|r| r.searcher().num_docs()))
            .unwrap_or(0);
        let (nodes, edges) = match GraphStore::open(&self.root) {
            Ok(g) => (g.node_count().unwrap_or(0), g.edge_count().unwrap_or(0)),
            Err(_) => (0, 0),
        };
        let dirty = self.dirty.load(Ordering::Relaxed) as u8;
        let indexing = self.indexing.load(Ordering::Relaxed) as u8;
        format!(
            "# HELP glossa_up 1 if the server is running\n\
             # TYPE glossa_up gauge\nglossa_up 1\n\
             # HELP glossa_index_chunks Indexed chunks in the tantivy index\n\
             # TYPE glossa_index_chunks gauge\nglossa_index_chunks {chunks}\n\
             # HELP glossa_graph_nodes Knowledge-graph nodes\n\
             # TYPE glossa_graph_nodes gauge\nglossa_graph_nodes {nodes}\n\
             # HELP glossa_graph_edges Knowledge-graph edges\n\
             # TYPE glossa_graph_edges gauge\nglossa_graph_edges {edges}\n\
             # HELP glossa_graph_dirty Derived layer stale (1) or fresh (0)\n\
             # TYPE glossa_graph_dirty gauge\nglossa_graph_dirty {dirty}\n\
             # HELP glossa_indexing Background ensure_fresh in progress (1) or idle (0)\n\
             # TYPE glossa_indexing gauge\nglossa_indexing {indexing}\n"
        )
    }

    /// Background maintenance: after indexing changes settle (debounce), run ONE `generalize` pass to
    /// refresh the derived layer — off the read hot path, never per-file. Spawned once by `kb mcp`.
    /// Exits promptly when `cancel` fires (graceful shutdown), so the loop never outlives the server.
    pub async fn maintenance_loop(self, cancel: tokio_util::sync::CancellationToken) {
        const DEBOUNCE_MS: u64 = 5_000;
        const POLL_MS: u64 = 1_000;
        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = tokio::time::sleep(Duration::from_millis(POLL_MS)) => {}
            }
            if !Self::maintenance_due(
                self.dirty.load(Ordering::Relaxed),
                self.last_change.load(Ordering::Relaxed),
                crate::trace::now_ms(),
                DEBOUNCE_MS,
            ) {
                continue;
            }
            // Clear BEFORE running so a change landing during the pass re-arms it for the next round.
            self.dirty.store(false, Ordering::Relaxed);
            let me = self.clone();
            let _ = tokio::task::spawn_blocking(move || me.run_generalize()).await;
        }
    }

    #[cfg(test)]
    pub fn enabled_tools(&self) -> Vec<String> {
        self.tool_router
            .list_all()
            .iter()
            .map(|t| t.name.to_string())
            .collect()
    }
}

fn internal(e: anyhow::Error) -> McpError {
    McpError::internal_error(e.to_string(), None)
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchArgs {
    #[schemars(
        description = "natural-language keywords (morphology-aware, BM25-ranked) — NOT a regex"
    )]
    query: String,
    #[serde(default)]
    #[schemars(description = "max hits (default 50)")]
    limit: Option<usize>,
    #[serde(default)]
    #[schemars(
        description = "only documents whose path matches this ripgrep -g glob, e.g. *.pdf, **/*, *.{pdf,md}"
    )]
    glob: Option<String>,
    #[serde(default)]
    #[schemars(description = "only this file type, e.g. pdf (-t)")]
    file_type: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GlobArgs {
    #[schemars(
        description = "ripgrep -g glob over document paths, e.g. *, **/*, *.pdf, *.{pdf,htm}, *АБАК*"
    )]
    pattern: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ReadArgs {
    #[schemars(description = "document path, exactly as shown in a search result")]
    path: String,
    #[schemars(
        description = "chunk number to read, exactly as shown in `[#n]` in a search result (page number for PDFs)"
    )]
    n: u32,
    #[serde(default)]
    #[schemars(description = "include embedded images (default true)")]
    include_images: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct NeighborsArgs {
    #[serde(default)]
    #[schemars(
        description = "reasoning-node id from a `glossary` line (e.g. `sym:...`) — call after glossary to find alternate/similar cases"
    )]
    node: Option<String>,
    #[serde(default)]
    #[schemars(
        description = "document path, exactly as shown in a search result (use with `n` instead of `node`)"
    )]
    path: Option<String>,
    #[serde(default)]
    #[schemars(description = "chunk number, exactly as shown in `[#n]` in a search result")]
    n: Option<u64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct NameArg {
    name: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GraphStatsArgs {
    #[serde(default)]
    #[schemars(
        description = "document source_path: also report that document's per-Field coverage — which Fields still lack a source (MENTIONS edge) or values (CONSTRAINED_BY→Enum)"
    )]
    doc: Option<String>,
    #[serde(default)]
    #[schemars(
        description = "node id: instead of the summary, return EVERYTHING about that one node — id, type, label, aliases, and every outgoing/incoming edge"
    )]
    node: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Empty {}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct ConstraintSolveArgs {
    #[schemars(description = "GOST document source_path (which document's constraints to load)")]
    source_path: String,
    #[schemars(description = "mode: validate | infer | check")]
    mode: String,
    #[serde(default)]
    #[schemars(description = "field assignments for validation mode (JSON object of field→value)")]
    assignment: Option<std::collections::HashMap<String, serde_json::Value>>,
}

// No-arg tools take `Parameters<Empty>`. A derived empty-struct schema is a bare
// `{"type":"object"}` (schemars omits an empty `properties`), but LM Studio's OpenAI-tools
// validator REJECTS a tool whose `function.parameters` lacks a `properties` object (400 → the
// gateway 502s the whole inference). Emit an explicit empty `properties` so every consumer of
// the schema — the live MCP `list_tools`, `tool_specs`, and the TZ export — is valid.
impl JsonSchema for Empty {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Empty".into()
    }
    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "object",
            "properties": {}
        })
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct NoteArgs {
    #[schemars(description = "Indexed document path from grep/read (#n stripped server-side)")]
    doc: String,
    #[schemars(
        description = "Note filename with extension. `.csp` = a limit table (tab-separated rows, first line is ALWAYS the column headers — exact parameter names); any other extension is a free-form note"
    )]
    file: String,
    #[schemars(
        description = "Full file content (replaces the note if it exists, unless append=true). A `.csp` is validated on write: the reply echoes the parsed columns and row count, and a malformed table (empty header cell, ragged row) is rejected without writing"
    )]
    content: String,
    #[serde(default)]
    #[schemars(
        description = "Add to the existing file instead of replacing it. For a `.csp`, content is data rows placed under the existing header (repeating the header line is fine — it is deduplicated); for other files, content is appended as-is."
    )]
    append: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct NotebookPathArgs {
    #[schemars(
        description = "Notebook path from ls (<document>/<file>; document includes extension)"
    )]
    path: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct LsArgs {
    #[serde(default)]
    #[schemars(description = "Optional indexed document path to list notes for one standard only")]
    doc: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SedArgs {
    #[schemars(
        description = "Notebook path from ls (<document>/<file>; document path includes extension)"
    )]
    path: String,
    #[schemars(description = "Literal substring to find (not regex)")]
    old: String,
    #[schemars(description = "Replacement text")]
    new: String,
    #[serde(default)]
    #[schemars(description = "Replace all occurrences (default: first only)")]
    all: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GrepArgs {
    #[schemars(
        description = "The text to find. It is a regex by default, so `A|B` matches A or B, `[0-9]+` matches digits, `.` matches any character; plain text also works as-is. For a value list, grep one value or the parameter name with `context` to pull its table window."
    )]
    pattern: String,
    #[serde(default)]
    #[schemars(
        description = "Search only this document — the same path `glob`/`read`/`search` show (a trailing `#chunk` is ignored). Omit to search the whole base."
    )]
    path: Option<String>,
    #[serde(default)]
    #[schemars(description = "case-insensitive matching (-i)")]
    ignore_case: Option<bool>,
    #[serde(default)]
    #[schemars(
        description = "Match the pattern as literal characters, with no regex meaning (`|`, `.`, `*` match themselves). Usually leave this off — use it only to find text that itself contains regex symbols (-F)."
    )]
    fixed: Option<bool>,
    #[serde(default)]
    #[schemars(description = "match whole words only (-w)")]
    word: Option<bool>,
    #[serde(default)]
    #[schemars(
        description = "only files whose path matches this ripgrep -g glob, e.g. *.pdf, **/*"
    )]
    glob: Option<String>,
    #[serde(default)]
    #[schemars(description = "only this file type, e.g. pdf (-t)")]
    file_type: Option<String>,
    #[serde(default)]
    #[schemars(
        description = "Return N lines around each match — this turns a grep into a focused window read. To pull a value's whole table, grep one of its values (or the parameter name) with context ~20-40, instead of reading the whole document. Both sides; -A/-B override a side (-C)."
    )]
    context: Option<usize>,
    #[serde(default)]
    #[schemars(description = "emit N context lines before each match (-B)")]
    before: Option<usize>,
    #[serde(default)]
    #[schemars(description = "emit N context lines after each match (-A)")]
    after: Option<usize>,
    #[serde(default)]
    #[schemars(
        description = "print only the matched substring(s), one per line, not the whole line (-o)"
    )]
    only_matching: Option<bool>,
    #[serde(default)]
    #[schemars(
        description = "Show each match's line number within the chunk — the position of the hit, so you can point at where a value sits or read a window around that line (-n)."
    )]
    line_number: Option<bool>,
    #[serde(default)]
    #[schemars(
        description = "output only a count of matching lines per chunk, not the lines (-c)"
    )]
    count: Option<bool>,
    #[serde(default)]
    #[schemars(description = "stop after N matching lines per chunk (-m)")]
    max_count: Option<usize>,
    #[serde(default)]
    #[schemars(
        description = "let the pattern span lines: `.` matches newlines, matched against the whole chunk (-U)"
    )]
    multiline: Option<bool>,
}

#[derive(Debug, JsonSchema)]
struct GraphUpsertArgs {
    #[serde(default)]
    nodes: Vec<crate::graph::ops::UpsertNode>,
    #[serde(default)]
    edges: Vec<crate::graph::ops::UpsertEdge>,
    #[serde(skip, default)]
    parse_notes: Vec<String>,
}

impl<'de> serde::Deserialize<'de> for GraphUpsertArgs {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = serde_json::Value::deserialize(deserializer)?;
        let (nodes, edges, parse_notes) = crate::graph::ops::parse_upsert_payload(&v);
        Ok(Self {
            nodes,
            edges,
            parse_notes,
        })
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GraphDeleteEdge {
    #[schemars(description = "label of the source node")]
    from: String,
    #[schemars(description = "the edge type, e.g. RESOLVED_BY")]
    edge_type: String,
    #[schemars(description = "label of the target node")]
    to: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GraphDeleteArgs {
    #[serde(default)]
    #[schemars(description = "labels of reasoning nodes to remove")]
    nodes: Vec<String>,
    #[serde(default)]
    edges: Vec<GraphDeleteEdge>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GraphUpdateNode {
    #[schemars(description = "current label of the node to edit")]
    label: String,
    #[serde(default)]
    #[schemars(description = "new label, if renaming")]
    new_label: Option<String>,
    #[serde(default)]
    #[schemars(description = "new node type, if changing it")]
    new_type: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GraphUpdateArgs {
    #[serde(default)]
    nodes: Vec<GraphUpdateNode>,
    // The model commonly sends a SINGLE update FLAT — {label, new_label, new_type} — instead of
    // wrapping it in `nodes`. Accept that shape too rather than silently updating nothing.
    #[serde(default)]
    #[schemars(
        description = "single-update shortcut: current label of the node to edit (alternative to `nodes`)"
    )]
    label: Option<String>,
    #[serde(default)]
    new_label: Option<String>,
    #[serde(default)]
    new_type: Option<String>,
}

impl GraphUpdateArgs {
    /// Node updates from either accepted shape: the canonical `nodes: [...]`, or a single flat
    /// `{label, new_label?, new_type?}`. Empty only when neither was provided.
    fn into_updates(self) -> Vec<crate::graph::agent::NodeUpdate> {
        use crate::graph::agent::NodeUpdate;
        if !self.nodes.is_empty() {
            self.nodes
                .into_iter()
                .map(|n| NodeUpdate {
                    label: n.label,
                    new_label: n.new_label,
                    new_type: n.new_type,
                })
                .collect()
        } else if let Some(label) = self.label {
            vec![NodeUpdate {
                label,
                new_label: self.new_label,
                new_type: self.new_type,
            }]
        } else {
            vec![]
        }
    }
}

#[tool_router]
impl GlossaServer {
    #[tool(
        description = "Full-text search over the knowledge base — natural-language keywords (morphology-aware, BM25-ranked), NOT a regex. Returns ranked hits, one per line as `[#n] path · label · snippet`. Open a hit with `read(path, n)` using that `[#n]` number. Scope with optional glob/file_type filters; for an exact token or code use `grep` instead. Hits are ranked best-first — the top few usually contain the answer, so read those rather than running many searches."
    )]
    async fn search(
        &self,
        Parameters(a): Parameters<SearchArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.kick_freshen();
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let (body, _hits) = crate::tools::search(
            &idx,
            &a.query,
            a.limit.unwrap_or(50),
            a.glob.as_deref(),
            a.file_type.as_deref(),
            &self.trace,
        );
        Ok(CallToolResult::success(vec![Content::text(body)]))
    }

    #[tool(
        description = "Read material by reference. Usually a document chunk: pass the `path` and chunk number `n` (the `[#n]` from a search/grep result; for PDFs the page). It returns the chunk's WHOLE text — for a large chunk that is a lot, and a table in its middle is easy to under-read; when you only need a value or its table, `grep` that value with `context` and read just the window instead. Returns the full text plus prev/next chunk numbers; if `n` is out of range the reply states the valid range. You may ALSO pass a graph NODE id (e.g. a Resolution id from a `glossary` line) as `path` — then it returns that node plus every evidence chunk it and its 1-hop chain MENTION, each labelled with where it came from."
    )]
    async fn read(&self, Parameters(a): Parameters<ReadArgs>) -> Result<CallToolResult, McpError> {
        self.kick_freshen();
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let g = GraphStore::open(&self.root).ok();
        let out = crate::tools::read(&idx, g.as_ref(), &a.path, a.n as u64, &self.trace);
        let mut content = vec![Content::text(out.text)];
        if a.include_images.unwrap_or(true) {
            for img in out.images {
                let b64 = base64::engine::general_purpose::STANDARD.encode(&img.bytes);
                content.push(Content::image(b64, img.mime));
            }
        }
        Ok(CallToolResult::success(content))
    }

    #[tool(
        description = "Resolve a concept (a symptom, error, component or task in a few words) to graph nodes. A reasoning node prints its `id [type] label` followed by its full chain — cause → resolution — each with a `read path #n` anchor, so ONE call gives you the likely fix. The line may also show `· comm N · pr …` — the problem cluster id. After a hit, call `neighbors(<that node id>)` to list alternate and related cases before searching again. Structural Section/Document nodes show their `path #n` anchor. Empty result = nothing matches yet. Morphology-aware over labels/aliases. Also call it before creating a node, to REUSE an existing one."
    )]
    async fn glossary(
        &self,
        Parameters(a): Parameters<NameArg>,
    ) -> Result<CallToolResult, McpError> {
        self.kick_freshen();
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let spec = crate::tools::ChainSpec::from_ontology(&Ontology::load_or_default(&self.root));
        Ok(CallToolResult::success(vec![Content::text(
            crate::tools::glossary(&idx, &g, &a.name, &spec, &self.trace),
        )]))
    }

    #[tool(
        description = "Broaden a `glossary` hit — list OTHER solved cases linked to the same node. Call AFTER `glossary` when the cause→resolution chain is close but not quite right, you want alternates, or before running another search. Pass the reasoning-node `node` id copied from the glossary line (the token before `[Symptom]`/`[Cause]`/`[Resolution]`, e.g. `sym:...`), or a chunk `path` + `n`. Each line is prefixed and has a `read path #n` anchor: `SIMILAR` — paraphrase cases that share evidence; `COMMUNITY` — other nodes in the same problem cluster (same `comm N` as the glossary suffix), top by centrality. Empty → try another glossary term or fall back to search/grep. For the node's OWN chain, use `glossary` — not neighbors."
    )]
    async fn neighbors(
        &self,
        Parameters(a): Parameters<NeighborsArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.kick_freshen();
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let g = GraphStore::open(&self.root).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(
            crate::tools::neighbors(
                &idx,
                &g,
                a.node.as_deref(),
                a.path.as_deref(),
                a.n,
                &self.trace,
            ),
        )]))
    }

    #[tool(description = "Build/update the index + structural graph for the knowledge base.")]
    async fn index(&self, Parameters(_): Parameters<Empty>) -> Result<CallToolResult, McpError> {
        let s = index_dir(&self.root, false).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "indexed: {} added, {} removed, {} unchanged",
            s.added, s.removed, s.unchanged
        ))]))
    }

    #[tool(description = "Rebuild the index + graph from scratch.")]
    async fn reindex(&self, Parameters(_): Parameters<Empty>) -> Result<CallToolResult, McpError> {
        let s = index_dir(&self.root, true).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "reindexed: {} files",
            s.added
        ))]))
    }

    #[tool(description = "Resolve a name to existing graph node ids (entity resolution).")]
    async fn resolve(
        &self,
        Parameters(a): Parameters<NameArg>,
    ) -> Result<CallToolResult, McpError> {
        let g = GraphStore::open(&self.root).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(
            g.resolve(&a.name).map_err(internal)?.join("\n"),
        )]))
    }

    #[tool(
        description = "Return the knowledge-base ontology as JSON: parameters, constraints, relations, and graph-building patterns. Call first to learn valid node/edge shapes before graph_upsert."
    )]
    async fn get_ontology(
        &self,
        Parameters(_): Parameters<Empty>,
    ) -> Result<CallToolResult, McpError> {
        let ont = Ontology::load_or_default(&self.root);
        Ok(CallToolResult::success(vec![Content::text(
            crate::graph::ontology_export::export_pretty(&ont),
        )]))
    }

    #[cfg_attr(not(feature = "constraint"), allow(dead_code))]
    #[tool(
        description = "CSP solver for constraint graphs. Modes: validate, infer, check. Returns actionable feedback when the problem is empty or assignment keys miss Field labels."
    )]
    async fn constraint_solve(
        &self,
        Parameters(_a): Parameters<ConstraintSolveArgs>,
    ) -> Result<CallToolResult, McpError> {
        #[cfg(feature = "constraint")]
        {
            let a = _a;
            let g = GraphStore::open(&self.root).map_err(internal)?;
            let ont = Ontology::load_or_default(&self.root);
            let problem = crate::constraint_adapter::load_problem(&g, &ont, Some(&a.source_path))
                .map_err(|e| McpError::internal_error(e.to_string(), None))?;

            let mode = match a.mode.as_str() {
                "validate" => glossa_constraint::SolveMode::Validate,
                "infer" => glossa_constraint::SolveMode::Infer,
                "check" => glossa_constraint::SolveMode::Check,
                other => {
                    return Err(McpError::internal_error(
                        format!("unknown mode '{other}': use validate | infer | check"),
                        None,
                    ));
                }
            };

            let assignment: Vec<(String, serde_json::Value)> =
                a.assignment.unwrap_or_default().into_iter().collect();
            // Re-key onto the graph's Field labels via the morphology resolver, so a
            // value keyed by a paraphrase of a parameter name still hits its constraint.
            let assignment =
                crate::constraint_adapter::resolve_assignment_fields(&g, &problem, &assignment);

            let result = glossa_constraint::solver::solve(&problem, mode, &assignment);

            return Ok(CallToolResult::success(vec![Content::text(
                crate::constraint_adapter::format_solve_feedback(
                    &problem,
                    &result,
                    &assignment,
                    &a.source_path,
                ),
            )]));
        }

        #[cfg(not(feature = "constraint"))]
        {
            Err(McpError::internal_error(
                String::from(
                    "constraint feature is not enabled. Build glossa with --features constraint",
                ),
                None,
            ))
        }
    }

    #[tool(
        description = "Create/update reasoning nodes and directed edges. Each node needs a human-readable `label`, `node_type`, and indexed `source_path`. Reference endpoints in `edges` by label, or a document section as `<path>#<n>` where `<n>` is the INTEGER chunk number exactly as a search/grep/read shows it (the `[#n]` / `path:#n:`) — just that number, with nothing appended (no clause like `4.1`, no note in parentheses). The response lists written node ids and resolved edges. Send a node and edges that reference it in the same call."
    )]
    async fn graph_upsert(
        &self,
        Parameters(a): Parameters<GraphUpsertArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.kick_freshen();
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let ont = Ontology::load_or_default(&self.root);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let out = crate::graph::ops::graph_upsert(&idx, &g, &ont, a.nodes, a.edges, now);
        let message = if a.parse_notes.is_empty() {
            out.message
        } else {
            format!("{}\n{}", a.parse_notes.join("\n"), out.message)
        };
        Ok(CallToolResult::success(vec![Content::text(message)]))
    }

    #[tool(
        description = "Remove reasoning nodes/edges from the graph by label — use it to delete a node or relation you added by mistake or that is no longer valid. Deleting a node also removes edges touching it."
    )]
    async fn graph_delete(
        &self,
        Parameters(a): Parameters<GraphDeleteArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.kick_freshen();
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let refs: Vec<crate::graph::agent::EdgeRef> = a
            .edges
            .into_iter()
            .map(|e| crate::graph::agent::EdgeRef {
                from: e.from,
                edge_type: e.edge_type,
                to: e.to,
            })
            .collect();
        let msg = crate::graph::ops::graph_delete(&idx, &g, a.nodes, refs);
        Ok(CallToolResult::success(vec![Content::text(msg)]))
    }

    #[tool(
        description = "Edit an existing graph node in place — change its label or type while keeping its id and all its edges (delete-and-recreate would drop the edges). Identify the node by its label. To correct an edge, remove it with graph_delete and add the right one with graph_upsert."
    )]
    async fn graph_update(
        &self,
        Parameters(a): Parameters<GraphUpdateArgs>,
    ) -> Result<CallToolResult, McpError> {
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let msg = crate::graph::ops::graph_update(&g, a.into_updates());
        Ok(CallToolResult::success(vec![Content::text(msg)]))
    }

    #[tool(
        description = "Recompute the graph's DERIVED layer from what is currently stored: transitive-closure edges, SIMILAR links, communities and centrality (these surface in `glossary`/`neighbors`). Non-destructive — it never deletes or merges nodes. It also REPORTS how many degenerate reasoning chains exist as `prune_candidates` (a node off the reasoning spine) without removing them; actual pruning is a deliberate operator action. Run it after a batch of edits to refresh the derived view."
    )]
    async fn graph_generalize(
        &self,
        Parameters(_): Parameters<Empty>,
    ) -> Result<CallToolResult, McpError> {
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let ont = Ontology::load_or_default(&self.root);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        // Non-destructive (shared with the eval enricher → identical output).
        Ok(CallToolResult::success(vec![Content::text(
            crate::graph::ops::graph_generalize(&g, &ont, now),
        )]))
    }

    #[tool(
        description = "Universal graph statistics. Default (summary) mode: node counts by type and edge counts by relation, plus a per-community overview (each community's size and up to eight nodes ranked by centrality: `id [type] label`, PageRank). Pass `doc` (a document source_path) to also get that document's per-Field coverage: total Fields, how many have a SOURCE (an outgoing MENTIONS edge) vs still `to source`, and how many have a materialized constraint (per ontology) vs still `to value`. Pass `node` (a node id) to switch to node-inspection mode: everything about that one node — id, type, label, aliases, and every outgoing/incoming edge with the neighbour's label."
    )]
    async fn graph_stats(
        &self,
        Parameters(a): Parameters<GraphStatsArgs>,
    ) -> Result<CallToolResult, McpError> {
        let g = GraphStore::open(&self.root).map_err(internal)?;
        if let Some(id) = a.node.as_deref() {
            return Ok(CallToolResult::success(vec![Content::text(
                crate::tools::node_inspect(&g, id),
            )]));
        }
        let mut out = crate::tools::graph_stats(&g);
        if let Some(doc) = a.doc.as_deref() {
            let ont = Ontology::load_or_default(&self.root);
            out.push('\n');
            out.push_str(&crate::tools::checklist_coverage_report(&g, doc, &ont));
        }
        Ok(CallToolResult::success(vec![Content::text(out)]))
    }

    #[tool(
        description = "Find an exact string in the text — a code, identifier, parameter name, or a value (e.g. `maxTsdr`, `F24`, `400`). ripgrep regex supported; smart-case. Use it whenever you know a precise token to locate (beats keyword `search`; for fuzzy/conceptual lookup use `search`). TO READ A TABLE, grep one of its values with `context` set to ~20-40: the reply then carries that many lines around each hit — a focused window onto the table — so you get the whole column in one call without reading the entire chunk. Returns matching lines as `path:#n: line`; a context line uses `-` instead of `:`. Reach for `read(path, n)` only when you actually need a whole chunk, not to locate a value. Other flags mirror ripgrep: -i/-F/-w, -o only-matching, -n line-number, -c count, -m max-count, -U multiline."
    )]
    async fn grep(&self, Parameters(a): Parameters<GrepArgs>) -> Result<CallToolResult, McpError> {
        self.kick_freshen();
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let opts = crate::grep::GrepOpts {
            ignore_case: a.ignore_case.unwrap_or(false),
            fixed: a.fixed.unwrap_or(false),
            word: a.word.unwrap_or(false),
            // Explicit `glob` wins; otherwise `path` scopes the search to that one document.
            glob: a
                .glob
                .or_else(|| a.path.as_deref().map(crate::grep::path_to_glob)),
            file_type: a.file_type,
            // -A/-B take precedence over the shared -C on their respective side.
            before: a.before.or(a.context).unwrap_or(0),
            after: a.after.or(a.context).unwrap_or(0),
            only_matching: a.only_matching.unwrap_or(false),
            line_number: a.line_number.unwrap_or(false),
            count: a.count.unwrap_or(false),
            max_count: a.max_count,
            multiline: a.multiline.unwrap_or(false),
            line_cap: None,
        };
        Ok(CallToolResult::success(vec![Content::text(
            crate::tools::grep(&idx, &a.pattern, &opts.with_default_context(), &self.trace),
        )]))
    }

    #[tool(
        description = "List knowledge-base documents whose path matches a ripgrep `-g` glob (e.g. `*`, `**/*`, `*.pdf`, `*.{pdf,htm}`, `*АБАК*`). Returns one `path  (N chunks)` per line — use it to discover what documents exist or find a file by name, then `read(path, n)` or scope a `search`/`grep` to it. N is the document's last page/section number; for PDFs every page 1..N is addressable (blank pages return empty text)."
    )]
    async fn glob(&self, Parameters(a): Parameters<GlobArgs>) -> Result<CallToolResult, McpError> {
        self.kick_freshen();
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(
            crate::tools::glob(&idx, &a.pattern, &self.trace),
        )]))
    }

    #[cfg_attr(not(feature = "notebook"), allow(dead_code))]
    #[tool(
        description = "Create, fully replace, or (with append=true) extend a notebook note bound to an indexed document. Without append, an existing file is OVERWRITTEN — to add rows to a `.csp` table you already wrote, pass append=true. A `.csp` file is a limit table (tab-separated rows, first line = column headers) — the reply echoes the parsed columns and row count. Use ls/cat/sed/del with paths from ls afterward."
    )]
    async fn note(&self, Parameters(a): Parameters<NoteArgs>) -> Result<CallToolResult, McpError> {
        #[cfg(feature = "notebook")]
        {
            let idx =
                crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
            let msg = crate::tools::note(
                &self.root,
                &idx,
                &a.doc,
                &a.file,
                &a.content,
                a.append.unwrap_or(false),
            );
            return Ok(CallToolResult::success(vec![Content::text(msg)]));
        }
        #[cfg(not(feature = "notebook"))]
        {
            Err(McpError::internal_error(
                String::from(
                    "notebook feature is not enabled. Build glossa with --features notebook",
                ),
                None,
            ))
        }
    }

    #[cfg_attr(not(feature = "notebook"), allow(dead_code))]
    #[tool(
        description = "Read a notebook note by path (from ls). Not for indexed corpus documents — use read(path, n)."
    )]
    async fn cat(
        &self,
        Parameters(a): Parameters<NotebookPathArgs>,
    ) -> Result<CallToolResult, McpError> {
        #[cfg(feature = "notebook")]
        {
            let idx =
                crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
            let msg = crate::tools::cat_note(&self.root, &idx, &a.path);
            return Ok(CallToolResult::success(vec![Content::text(msg)]));
        }
        #[cfg(not(feature = "notebook"))]
        {
            Err(McpError::internal_error(
                String::from(
                    "notebook feature is not enabled. Build glossa with --features notebook",
                ),
                None,
            ))
        }
    }

    #[cfg_attr(not(feature = "notebook"), allow(dead_code))]
    #[tool(
        description = "List notebook notes. Optional doc filter. Paths in the output are arguments for cat/sed/del."
    )]
    async fn ls(&self, Parameters(a): Parameters<LsArgs>) -> Result<CallToolResult, McpError> {
        #[cfg(feature = "notebook")]
        {
            let idx =
                crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
            let msg = crate::tools::ls_notes(&self.root, &idx, a.doc.as_deref());
            return Ok(CallToolResult::success(vec![Content::text(msg)]));
        }
        #[cfg(not(feature = "notebook"))]
        {
            Err(McpError::internal_error(
                String::from(
                    "notebook feature is not enabled. Build glossa with --features notebook",
                ),
                None,
            ))
        }
    }

    #[cfg_attr(not(feature = "notebook"), allow(dead_code))]
    #[tool(description = "Delete a notebook note by path (from ls).")]
    async fn del(
        &self,
        Parameters(a): Parameters<NotebookPathArgs>,
    ) -> Result<CallToolResult, McpError> {
        #[cfg(feature = "notebook")]
        {
            let idx =
                crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
            let msg = crate::tools::del_note(&self.root, &idx, &a.path);
            return Ok(CallToolResult::success(vec![Content::text(msg)]));
        }
        #[cfg(not(feature = "notebook"))]
        {
            Err(McpError::internal_error(
                String::from(
                    "notebook feature is not enabled. Build glossa with --features notebook",
                ),
                None,
            ))
        }
    }

    #[cfg_attr(not(feature = "notebook"), allow(dead_code))]
    #[tool(
        description = "Patch a notebook note: literal substring replace. Use cat(path) first if unsure of the exact old text."
    )]
    async fn sed(&self, Parameters(a): Parameters<SedArgs>) -> Result<CallToolResult, McpError> {
        #[cfg(feature = "notebook")]
        {
            let idx =
                crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
            let msg = crate::tools::sed_note(
                &self.root,
                &idx,
                &a.path,
                &a.old,
                &a.new,
                a.all.unwrap_or(false),
            );
            return Ok(CallToolResult::success(vec![Content::text(msg)]));
        }
        #[cfg(not(feature = "notebook"))]
        {
            Err(McpError::internal_error(
                String::from(
                    "notebook feature is not enabled. Build glossa with --features notebook",
                ),
                None,
            ))
        }
    }

    #[tool(description = "Delete the index + graph for the knowledge base.")]
    async fn purge(&self, Parameters(_): Parameters<Empty>) -> Result<CallToolResult, McpError> {
        let g = self.root.join(".glossa");
        if g.exists() {
            std::fs::remove_dir_all(&g).map_err(|e| internal(e.into()))?;
        }
        Ok(CallToolResult::success(vec![Content::text(
            "purged .glossa",
        )]))
    }
}

#[tool_handler]
impl ServerHandler for GlossaServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::new(ServerCapabilities::builder().enable_tools().build());
        info.protocol_version = ProtocolVersion::V_2025_06_18;
        info.server_info.name = "glossa".into();
        info.server_info.version = env!("CARGO_PKG_VERSION").into();
        info.instructions = Some("glossa File-First knowledge-base search. `search` takes BM25 keywords (morphology-aware), returns numbered hits `[#n]`; `read` opens chunk number `n`.".into());
        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_param_schema_has_properties() {
        // no-arg tools (graph_generalize/index/reindex/purge) must expose an explicit
        // `properties: {}` — LM Studio's tools validator 400s when it is absent.
        let v = serde_json::to_value(schemars::schema_for!(Empty)).unwrap();
        assert!(
            v.get("properties").map(|p| p.is_object()).unwrap_or(false),
            "Empty schema must carry an object `properties`: {v}"
        );
    }

    #[test]
    fn graph_upsert_args_deserialize_from_json() {
        let json =
            r#"{"nodes":[{"node_type":"Document","label":"a","source_path":"a.md"}],"edges":[]}"#;
        let a: GraphUpsertArgs = serde_json::from_str(json).unwrap();
        assert_eq!(a.nodes.len(), 1);
        assert_eq!(a.nodes[0].label, "a");
        assert!(a.edges.is_empty());
    }

    #[test]
    fn graph_upsert_args_move_edge_from_nodes() {
        let json = r#"{"nodes":[{"node_type":"Enum","label":"T","source_path":"a.md","aliases":["1"]},{"from":"fld:t","edge_type":"CONSTRAINED_BY","to":"T","source_path":"a.md"}],"edges":[]}"#;
        let a: GraphUpsertArgs = serde_json::from_str(json).unwrap();
        assert_eq!(a.nodes.len(), 1);
        assert_eq!(a.edges.len(), 1);
        assert_eq!(a.edges[0].edge_type, "CONSTRAINED_BY");
        assert!(!a.parse_notes.is_empty());
    }

    #[test]
    fn graph_update_args_accept_nested_and_flat() {
        // canonical nested shape
        let nested: GraphUpdateArgs = serde_json::from_str(
            r#"{"nodes":[{"label":"old","new_label":"new","new_type":"Resolution"}]}"#,
        )
        .unwrap();
        let u = nested.into_updates();
        assert_eq!(u.len(), 1);
        assert_eq!(u[0].label, "old");
        assert_eq!(u[0].new_label.as_deref(), Some("new"));

        // FLAT shape the model commonly sends — must be accepted, not silently dropped
        let flat: GraphUpdateArgs =
            serde_json::from_str(r#"{"label":"old","new_label":"new","new_type":"Resolution"}"#)
                .unwrap();
        let u = flat.into_updates();
        assert_eq!(u.len(), 1, "a flat single update must yield one NodeUpdate");
        assert_eq!(u[0].label, "old");
        assert_eq!(u[0].new_type.as_deref(), Some("Resolution"));

        // genuinely empty → no updates (ops layer reports the clear message)
        let empty: GraphUpdateArgs = serde_json::from_str(r#"{}"#).unwrap();
        assert!(empty.into_updates().is_empty());
    }

    #[tokio::test]
    async fn read_by_number_returns_body_and_footer() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("d.md"), b"# A\nalpha\n# B\nbravo\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let srv = GlossaServer::new(dir.path().to_path_buf(), Profile::Editor, false, false);
        let path = "d.md".to_string(); // canonical key: corpus-root-relative

        let out = srv
            .read(Parameters(ReadArgs {
                path,
                n: 1,
                include_images: Some(false),
            }))
            .await
            .unwrap();
        let text = format!("{:?}", out);
        assert!(text.contains("alpha"), "body present: {text}");
        assert!(
            text.contains("#2") || text.contains("next"),
            "footer offers next: {text}"
        );
    }

    #[tokio::test]
    async fn grep_tool_finds_literal_across_chunks() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("d.md"), b"# A\nmaxTsdr 3000\n# B\nother\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let srv = GlossaServer::new(dir.path().to_path_buf(), Profile::Editor, false, false);
        let out = srv
            .grep(Parameters(GrepArgs {
                pattern: "maxTsdr".into(),
                path: None,
                ignore_case: None,
                fixed: None,
                word: None,
                glob: None,
                file_type: None,
                context: None,
                before: None,
                after: None,
                only_matching: None,
                line_number: None,
                count: None,
                max_count: None,
                multiline: None,
            }))
            .await
            .unwrap();
        assert!(format!("{:?}", out).contains("maxTsdr"));
        assert!(format!("{:?}", out).contains(":#")); // carries the #n read key
    }

    #[tokio::test]
    async fn glob_tool_lists_documents() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("АБАК.md"),
            "# A\nраз\n# B\nдва\n".as_bytes(),
        )
        .unwrap();
        std::fs::write(dir.path().join("Other.md"), "# A\nраз\n".as_bytes()).unwrap();
        index_dir(dir.path(), true).unwrap();
        let srv = GlossaServer::new(dir.path().to_path_buf(), Profile::Editor, false, false);
        let out = format!(
            "{:?}",
            srv.glob(Parameters(GlobArgs {
                pattern: "*АБАК*".into()
            }))
            .await
            .unwrap()
        );
        assert!(out.contains("АБАК"), "lists the matching doc: {out}");
        assert!(!out.contains("Other"), "excludes non-matching: {out}");
    }

    #[tokio::test]
    async fn glob_tool_recursive_lists_nested_docs() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("top.md"), "# A\none\n".as_bytes()).unwrap();
        std::fs::write(
            dir.path().join("sub").join("nested.md"),
            "# A\ntwo\n".as_bytes(),
        )
        .unwrap();
        index_dir(dir.path(), true).unwrap();
        let srv = GlossaServer::new(dir.path().to_path_buf(), Profile::Editor, false, false);
        let out = format!(
            "{:?}",
            srv.glob(Parameters(GlobArgs {
                pattern: "**/*".into()
            }))
            .await
            .unwrap()
        );
        assert!(out.contains("top.md"), "lists top-level: {out}");
        assert!(out.contains("nested"), "lists nested: {out}");
    }

    #[test]
    fn maintenance_due_only_when_dirty_and_quiet() {
        // never when clean
        assert!(!GlossaServer::maintenance_due(false, 0, 10_000, 5_000));
        // dirty but changes still arriving (within the debounce window) → wait
        assert!(!GlossaServer::maintenance_due(true, 8_000, 10_000, 5_000));
        // dirty and quiet for >= the debounce window → run
        assert!(GlossaServer::maintenance_due(true, 2_000, 10_000, 5_000));
    }

    #[tokio::test]
    async fn maintenance_loop_stops_on_cancel() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nx\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let srv = GlossaServer::new(dir.path().to_path_buf(), Profile::Editor, false, false);
        let cancel = tokio_util::sync::CancellationToken::new();
        cancel.cancel(); // pre-cancelled → the loop must return promptly, not hang
        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            srv.maintenance_loop(cancel),
        )
        .await
        .expect("maintenance_loop honored cancel");
    }

    #[test]
    fn run_generalize_populates_node_meta() {
        use crate::graph::store::{Node, Provenance};
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nintro\n## B\nbody b\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        // generalize scopes community/centrality to the REASONING subgraph (structural Document/
        // Section nodes are excluded), so seed a reasoning node for the pass to annotate.
        {
            let g = GraphStore::open(dir.path()).unwrap();
            g.put_node(&Node {
                id: "sym:x".into(),
                node_type: "Symptom".into(),
                label: "потеря связи".into(),
                aliases: vec![],
                prov: Provenance {
                    source_path: "a.md".into(),
                    range: None,
                    file_sig: None,
                    origin: "agent".into(),
                    confidence: 0.8,
                    created_at: 1,
                },
            })
            .unwrap();
        }
        let srv = GlossaServer::new(dir.path().to_path_buf(), Profile::Editor, false, false);
        srv.run_generalize();
        let g = GraphStore::open(dir.path()).unwrap();
        assert!(
            g.node_meta("sym:x").unwrap().is_some(),
            "generalize pass populated node_meta (community/centrality) for the reasoning node"
        );
    }

    #[test]
    fn readiness_true_and_metrics_render() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nhello world\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let srv = GlossaServer::new(dir.path().to_path_buf(), Profile::Editor, false, false);
        assert!(srv.readiness(), "index + graph open → ready");
        let m = srv.metrics_text();
        assert!(m.contains("glossa_up 1"), "metrics: {m}");
        assert!(m.contains("glossa_index_chunks"), "metrics: {m}");
        assert!(m.contains("glossa_graph_nodes"), "metrics: {m}");
        assert!(m.contains("glossa_graph_dirty"), "metrics: {m}");
    }

    #[test]
    fn run_generalize_skips_when_lock_held() {
        use crate::graph::store::{Node, Provenance};
        use fs4::fs_std::FileExt;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nintro\n## B\nbody b\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        {
            let g = GraphStore::open(dir.path()).unwrap();
            g.put_node(&Node {
                id: "sym:x".into(),
                node_type: "Symptom".into(),
                label: "потеря связи".into(),
                aliases: vec![],
                prov: Provenance {
                    source_path: "a.md".into(),
                    range: None,
                    file_sig: None,
                    origin: "agent".into(),
                    confidence: 0.8,
                    created_at: 1,
                },
            })
            .unwrap();
        }
        let srv = GlossaServer::new(dir.path().to_path_buf(), Profile::Editor, false, false);

        // Another editor holds the cross-process generalize lock.
        let lock_path = dir.path().join(".glossa").join("generalize.lock");
        let holder = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .unwrap();
        assert!(
            FileExt::try_lock_exclusive(&holder).unwrap(),
            "test acquires the lock"
        );

        // Lock held → this instance must SKIP the pass (no derived layer written).
        srv.run_generalize();
        assert!(
            GraphStore::open(dir.path())
                .unwrap()
                .node_meta("sym:x")
                .unwrap()
                .is_none(),
            "lock held → generalize skipped, no node_meta"
        );

        // Release → next run proceeds.
        FileExt::unlock(&holder).unwrap();
        srv.run_generalize();
        assert!(
            GraphStore::open(dir.path())
                .unwrap()
                .node_meta("sym:x")
                .unwrap()
                .is_some(),
            "lock free → generalize ran, node_meta written"
        );
    }

    #[test]
    fn profile_gates_tool_visibility() {
        let root = std::path::PathBuf::from(".");
        let reader = GlossaServer::new(root.clone(), Profile::Reader, false, false).enabled_tools();
        assert!(reader.contains(&"search".to_string()) && reader.contains(&"read".to_string()));
        assert!(reader.contains(&"ls".to_string()) && reader.contains(&"cat".to_string()));
        assert!(
            !reader.contains(&"note".to_string())
                && !reader.contains(&"sed".to_string())
                && !reader.contains(&"del".to_string())
        );
        assert!(
            !reader.contains(&"index".to_string())
                && !reader.contains(&"graph_upsert".to_string())
                && !reader.contains(&"purge".to_string())
        );
        assert!(!reader.contains(&"write".to_string()));

        let editor = GlossaServer::new(root.clone(), Profile::Editor, false, false).enabled_tools();
        assert!(
            editor.contains(&"note".to_string())
                && editor.contains(&"ls".to_string())
                && editor.contains(&"cat".to_string())
        );
        assert!(editor.contains(&"index".to_string()) && editor.contains(&"resolve".to_string()));
        assert!(
            editor.contains(&"graph_generalize".to_string()),
            "editor exposes the non-destructive generalize tool"
        );
        assert!(
            editor.contains(&"graph_stats".to_string()),
            "editor exposes graph stats"
        );
        assert!(!editor.contains(&"purge".to_string()));
        assert!(
            !reader.contains(&"graph_generalize".to_string()),
            "reader cannot generalize"
        );
        assert!(
            !reader.contains(&"graph_stats".to_string()),
            "reader cannot graph_stats"
        );

        let full = GlossaServer::new(root.clone(), Profile::Full, false, false).enabled_tools();
        assert!(full.contains(&"purge".to_string()));
        assert!(full.contains(&"note".to_string()) && full.contains(&"sed".to_string()));

        // resolve is a universally available tool — present in EVERY profile (not gated).
        for prof in [&reader, &editor, &full] {
            assert!(
                prof.contains(&"resolve".to_string()),
                "resolve must be in every profile"
            );
        }

        let ng = GlossaServer::new(root, Profile::Editor, false, true).enabled_tools();
        assert!(ng.contains(&"search".to_string()) && ng.contains(&"read".to_string()));
        assert!(
            !ng.contains(&"neighbors".to_string())
                && !ng.contains(&"graph_upsert".to_string())
                && !ng.contains(&"index".to_string())
        );
    }
}
