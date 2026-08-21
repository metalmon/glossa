use crate::graph::ontology::Ontology;
use crate::graph::store::GraphStore;
use crate::index::store::index_dir;
use base64::Engine as _;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolResult, ContentBlock as Content, ProtocolVersion, ResourceContents, ServerCapabilities,
    ServerInfo,
};
use rmcp::{tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler};
use schemars::JsonSchema;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
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
    /// Set when a freshen reindexed something — derived layer (closure/SIMILAR +
    /// community/centrality) is stale until debounced `generalize` runs (off the read hot path).
    dirty: Arc<AtomicBool>,
    /// Epoch-ms of the last indexing change — the debounce clock for the maintenance loop.
    last_change: Arc<AtomicU64>,
    /// Count of in-flight synchronous `freshen_now` calls (concurrent read tools + the startup
    /// warm-up). A counter, not a flag, so overlapping freshens don't clear each other early —
    /// surfaced as the `glossa_indexing` metrics gauge (1 when any freshen is running).
    indexing: Arc<AtomicUsize>,
    /// When true, `read` tool strips all image content from responses.
    no_image: bool,
    /// In-process cache of `manifest.json`, invalidated by the file's mtime.
    manifest_cache: Arc<Mutex<ManifestCache>>,
    /// HTTP request metrics (streamable-http): shared with the axum middleware, rendered by
    /// `metrics_text`. Inert under stdio (no middleware records into it).
    http: Arc<crate::http_metrics::HttpMetrics>,
}

#[derive(Default)]
struct ManifestCache {
    manifest: crate::index::manifest::Manifest,
    mtime_nanos: Option<u128>,
}

#[allow(dead_code)] // read tools stay enabled for Reader; listed for profile documentation
const NOTEBOOK_READ_TOOLS: &[&str] = &["ls"];
const NOTEBOOK_WRITE_TOOLS: &[&str] = &["note", "del"];
const EDITOR_TOOLS: &[&str] = &[
    "index",
    "graph_build",
    "graph_upsert",
    "graph_delete",
    "graph_update",
    "graph_generalize",
    "graph_stats",
    "graph_doctor",
    // Read-only, but withheld from Reader: low-level or rarely-reached navigation the weak reader
    // never calls in practice (measured over many runs: resolve 0%, constraint_solve 0%, neighbors
    // ~2%, related ~2-8% and correlating with wrong answers), so it is clutter that muddies tool
    // choice. `sql` is NOT here — it moved into the reader set.
    "resolve",
    "neighbors",
    "constraint_solve",
    "related",
];
const FULL_TOOLS: &[&str] = &["purge"];
const GRAPH_TOOLS: &[&str] = &[
    "glossary",
    "related",
    "neighbors",
    "reach",
    "graph_upsert",
    "graph_delete",
    "graph_update",
    "graph_generalize",
    "graph_doctor",
    "resolve",
    "index",
    "purge",
];

/// Launch-time tool-gating flags, bundled so `GlossaServer::new` doesn't grow a tail of positional
/// bools. All default off (every capability on); each `no_*` withholds one. Set from CLI args.
#[derive(Clone, Copy, Default)]
pub struct ServerFlags {
    /// Withhold the reasoning-graph tools (`--no-graph`): a plain file-search server.
    pub no_graph: bool,
    /// Withhold image delivery from `read` (images are opt-in via `--vision`).
    pub no_image: bool,
    /// Withhold `get_source_file` (original-file delivery is opt-in via `--source-file`).
    pub no_source_file: bool,
}

impl GlossaServer {
    pub fn new(root: PathBuf, profile: Profile, trace: bool, flags: ServerFlags) -> Self {
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
        if flags.no_graph {
            for t in GRAPH_TOOLS {
                router.disable_route(*t);
            }
        }
        if flags.no_source_file {
            router.disable_route("get_source_file");
        }
        if flags.no_image {
            if let Some(route) = router.map.get_mut("read") {
                let mut schema: serde_json::Value = serde_json::to_value(&*route.attr.input_schema)
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                if let Some(obj) = schema.as_object_mut() {
                    if let Some(props) = obj.get_mut("properties").and_then(|p| p.as_object_mut()) {
                        props.remove("page_image");
                        props.remove("include_images");
                    }
                    if let Some(req) = obj.get_mut("required").and_then(|r| r.as_array_mut()) {
                        req.retain(|v| {
                            v.as_str()
                                .map(|s| s != "page_image" && s != "include_images")
                                .unwrap_or(true)
                        });
                    }
                }
                let new_schema: rmcp::model::JsonObject =
                    serde_json::from_value(schema).unwrap_or_default();
                route.attr.input_schema = Arc::new(new_schema);
            }
        }
        #[cfg(not(feature = "notebook"))]
        {
            for t in NOTEBOOK_READ_TOOLS.iter().chain(NOTEBOOK_WRITE_TOOLS) {
                router.disable_route(*t);
            }
        }
        #[cfg(not(feature = "constraint"))]
        {
            router.disable_route("constraint_solve");
            router.disable_route("graph_build");
        }
        let trace = if trace {
            crate::trace::TraceLog::to_dir(&root)
        } else {
            crate::trace::TraceLog::disabled()
        };
        Self {
            root,
            tool_router: router,
            trace,
            dirty: Arc::new(AtomicBool::new(false)),
            last_change: Arc::new(AtomicU64::new(0)),
            indexing: Arc::new(AtomicUsize::new(0)),
            no_image: flags.no_image,
            manifest_cache: Arc::new(Mutex::new(ManifestCache::default())),
            http: Arc::new(crate::http_metrics::HttpMetrics::default()),
        }
    }

    /// Shared HTTP metrics handle — the streamable-http middleware records requests into it and
    /// `metrics_text` renders it.
    pub fn http_metrics(&self) -> Arc<crate::http_metrics::HttpMetrics> {
        self.http.clone()
    }

    /// Run `f` against the in-process manifest cache, reloading it when `manifest.json`'s mtime advanced
    /// (one `stat`; a full parse only when something reindexed). Backs both baseline lookups.
    fn with_manifest<T>(&self, f: impl FnOnce(&crate::index::manifest::Manifest) -> T) -> T {
        let p = self.root.join(".glossa").join("manifest.json");
        let cur = std::fs::metadata(&p)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos());
        let mut cache = self
            .manifest_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if cur != cache.mtime_nanos {
            cache.manifest = crate::index::manifest::Manifest::load(&self.root);
            cache.mtime_nanos = cur;
        }
        f(&cache.manifest)
    }

    /// Manifest signature for a corpus document `rel`, or `None`.
    fn baseline_sig(&self, rel: &str) -> Option<crate::index::manifest::FileSig> {
        self.with_manifest(|m| m.files.get(rel).copied())
    }

    /// Manifest signature for a notebook note `rel` (key relative to `.glossa/notes`), or `None`.
    fn baseline_note_sig(&self, rel: &str) -> Option<crate::index::manifest::FileSig> {
        self.with_manifest(|m| m.notes.get(rel).copied())
    }

    /// If `path` is an indexed corpus document whose on-disk signature differs from the manifest
    /// baseline, reindex just that file (best-effort: skip if another process holds the lock — the
    /// edit will be caught on a later read/explicit index). Cheap: resolves the path, one `stat`, one
    /// cache lookup; only touches the index when the file actually changed.
    fn lazy_reindex_if_changed(&self, path: &str) {
        let Ok(idx) = crate::index::store::DocIndex::open_or_create(&self.root) else {
            return;
        };
        let Some(rel) = idx.canonical_document_path(path) else {
            return; // not a corpus doc (note/graph id/garbage)
        };
        if idx.file_type_of(&rel).ok().flatten().as_deref() == Some("note") {
            // Notebook note: pick up an external in-place content edit (freshen's dir-mtime gate
            // misses it). Best-effort — skip if another process holds the lock.
            let abs = self.root.join(".glossa").join("notes").join(&rel);
            let Ok(cur) = crate::index::store::file_sig(&abs) else {
                return;
            };
            if self.baseline_note_sig(&rel) == Some(cur) {
                return; // unchanged
            }
            if let Some(_g) = crate::index::lock::try_index_lock(&self.root) {
                let _ = crate::index::store::reindex_note_locked(&self.root, &rel);
                self.mark_dirty();
            }
            return;
        }
        let abs = self.root.join(&rel);
        let Ok(cur) = crate::index::store::file_sig(&abs) else {
            return;
        };
        if self.baseline_sig(&rel) == Some(cur) {
            return; // unchanged
        }
        if let Some(_g) = crate::index::lock::try_index_lock(&self.root) {
            let _ = crate::index::store::index_one_file_locked(&self.root, &rel);
            self.mark_dirty();
        }
    }

    fn open_index_graph(
        &self,
    ) -> Result<(crate::index::store::DocIndex, Option<GraphStore>), McpError> {
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let g = GraphStore::open(&self.root).ok();
        Ok((idx, g))
    }

    /// Return the list of enabled tools (for config generation — not test-only).
    pub fn tool_specs(&self) -> Vec<rmcp::model::Tool> {
        self.tool_router.list_all()
    }

    /// Synchronous, sublinear freshness before serving a read: gate a full scan behind the cheap
    /// directory-mtime signature, index under the cross-process lock if the tree changed, then serve.
    /// Best-effort — indexing errors never fail the tool. Runs on the blocking pool so the async
    /// worker is not stalled, but the handler awaits it so the served index reflects the current tree.
    pub async fn freshen_now(&self) {
        self.indexing.fetch_add(1, Ordering::AcqRel);
        let root = self.root.clone();
        let res = tokio::task::spawn_blocking(move || {
            crate::index::store::freshen_blocking(&root, std::time::Duration::from_secs(3))
        })
        .await;
        self.indexing.fetch_sub(1, Ordering::AcqRel);
        if let Ok(Ok(stats)) = res {
            if stats.added + stats.removed > 0 {
                self.mark_dirty();
            }
        }
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
        use fs4::FileExt;
        let lock_path = self.root.join(".glossa").join("generalize.lock");
        let Ok(_lock) = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
        else {
            return;
        };
        match _lock.try_lock() {
            Ok(()) => {}      // acquired — we are the one editor running the pass this round
            Err(_) => return, // held or lock error → skip
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
        let indexing = (self.indexing.load(Ordering::Relaxed) > 0) as u8;
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
             # HELP glossa_indexing A freshen (freshen_now) is in progress (1) or idle (0)\n\
             # TYPE glossa_indexing gauge\nglossa_indexing {indexing}\n{http}",
            http = self.http.render(),
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
pub(crate) struct SearchArgs {
    #[schemars(
        description = "natural-language keywords (morphology-aware, BM25-ranked) — NOT a regex"
    )]
    query: String,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_usize_loose"
    )]
    #[schemars(description = "max hits (default 50)")]
    limit: Option<usize>,
    #[serde(default)]
    #[schemars(
        description = "only documents whose path matches this ripgrep -g glob, e.g. **/* (all) or *<name-fragment>*"
    )]
    glob: Option<String>,
    #[serde(default)]
    #[schemars(description = "restrict to a single file type (-t)")]
    file_type: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct GlobArgs {
    #[schemars(
        description = "ripgrep -g glob over document paths, e.g. * or **/* (all documents), or *<name-fragment>* to find a file by name"
    )]
    pattern: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct IndexArgs {
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_bool_loose"
    )]
    #[schemars(description = "rebuild the whole index from scratch (was the `reindex` tool)")]
    force: Option<bool>,
    #[serde(default)]
    #[schemars(
        description = "reindex just this one indexed document path (from grep/read), instead of the whole corpus"
    )]
    path: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct ReadArgs {
    #[schemars(description = "document path, exactly as shown in a search result")]
    path: String,
    #[serde(deserialize_with = "crate::json_util::deserialize_u32_loose")]
    #[schemars(
        description = "chunk number to read, exactly as shown in `[#n]` in a search result (page number for PDFs)"
    )]
    n: u32,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_bool_loose"
    )]
    #[schemars(description = "include embedded images (default true)")]
    include_images: Option<bool>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_bool_loose"
    )]
    #[schemars(
        description = "PDF only: return a raster of page `n` as JPEG (200 DPI) instead of text/embeds. Use when tables or layout are hard to read as text. Requires the server to be started with --vision."
    )]
    page_image: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SourceFileArgs {
    #[schemars(description = "document path, exactly as shown in a search/grep result")]
    path: String,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_u32_loose"
    )]
    #[schemars(
        description = "cited page number (PDF), as shown in `[#n]` — used for provenance and, if the file exceeds the cap, to deliver just that page"
    )]
    n: Option<u32>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_u64_loose"
    )]
    #[schemars(
        description = "maximum delivered size in bytes (default 10 MB, matching the ACP client cap)"
    )]
    max_bytes: Option<u64>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_bool_loose"
    )]
    #[schemars(
        description = "return the untouched original file instead of the default PDF conversion (currently only affects DOCX, delivered as PDF by default); default false"
    )]
    raw: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct RelatedArgs {
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
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_u64_loose"
    )]
    #[schemars(description = "chunk number, exactly as shown in `[#n]` in a search result")]
    n: Option<u64>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_string_loose"
    )]
    #[schemars(
        description = "show the graph as it was valid on this date (ISO-8601); a related node outside its validity interval is hidden. Timeless nodes are always shown. Filters the surrounding related nodes only, not the anchor `node` itself."
    )]
    as_of: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct NeighborsArgs {
    #[serde(default)]
    #[schemars(description = "graph node id (from a `glossary` line, e.g. `sym:...`)")]
    node: Option<String>,
    #[serde(default)]
    #[schemars(description = "document path from a search result (use with `n` instead of `node`)")]
    path: Option<String>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_u64_loose"
    )]
    #[schemars(description = "chunk number from a search/read (use with `path`)")]
    n: Option<u64>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_vec_string_loose"
    )]
    #[schemars(description = "keep only these edge/relation types (e.g. REFERENCES); omit for all")]
    edge_types: Option<Vec<String>>,
    #[serde(default)]
    #[schemars(description = "which edges: `out`, `in`, or `both` (default `both`)")]
    direction: Option<String>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_string_loose"
    )]
    #[schemars(
        description = "show the graph as it was valid on this date (ISO-8601); a neighbor outside its validity interval is hidden. Timeless nodes are always shown. Filters the surrounding neighbors only, not the anchor `node` itself."
    )]
    as_of: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct GlossaryArgs {
    #[schemars(description = "concept in your own words, e.g. \"connection loss\"")]
    name: String,
    #[schemars(
        description = "the full problem or question you are trying to solve, written out completely as a whole sentence — not a short phrase or the bare concept. It ranks the returned facts by what you actually need, so a complete, specific question ranks the right chain far higher than a terse one."
    )]
    query: String,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_string_loose"
    )]
    #[schemars(
        description = "show the graph as it was valid on this date (ISO-8601); a matched node outside its validity interval is hidden. Timeless nodes are always shown."
    )]
    as_of: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct ReachArgs {
    #[serde(default)]
    #[schemars(description = "start: graph node id (or use from_path+from_n)")]
    from: Option<String>,
    #[serde(default)]
    #[schemars(description = "start: document path (use with from_n instead of from)")]
    from_path: Option<String>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_u64_loose"
    )]
    #[schemars(description = "start: chunk number (use with from_path)")]
    from_n: Option<u64>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_string_loose"
    )]
    #[schemars(
        description = "relation to follow, fuzzy-matched to the ontology's real edge types (e.g. \"located\" matches LOCATED_IN); omit for ALL chaining relations (undirected — 'is this connected at all')"
    )]
    relation: Option<String>,
    #[serde(default)]
    #[schemars(
        description = "end: graph node id to VERIFY a connection to (or use to_path+to_n); omit for DISCOVERY — find every node `from` reaches along `relation`"
    )]
    to: Option<String>,
    #[serde(default)]
    #[schemars(description = "end: document path (use with to_n instead of to)")]
    to_path: Option<String>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_u64_loose"
    )]
    #[schemars(description = "end: chunk number (use with to_path)")]
    to_n: Option<u64>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_bool_loose"
    )]
    #[schemars(
        description = "cross-document bridge: when an in-graph chain dead-ends, resolve its mention to same-named nodes in OTHER documents and continue there (default true). false = graph-only, in-document connectivity — this reproduces the old `path` tool exactly when combined with to+no relation."
    )]
    bridge: Option<bool>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_usize_loose"
    )]
    #[schemars(description = "max edges to search (default 6, capped at 12)")]
    max_depth: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct NameArg {
    name: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GraphStatsArgs {
    #[serde(default)]
    #[schemars(
        description = "document source_path: list that document's owned non-structural nodes and all outgoing edges (doc is scope only; ontology-independent)"
    )]
    doc: Option<String>,
    #[serde(default)]
    #[schemars(
        description = "node id: instead of the summary, return EVERYTHING about that one node — id, type, label, aliases, and every outgoing/incoming edge"
    )]
    node: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct GraphQueryArgs {
    // `sql` is inherently a string — the loose json_util helpers exist to coerce numbers/bools
    // an LLM client sent as strings, which doesn't apply here. `default` so a missing/absent
    // `sql` deserializes to "" (empty → the tool returns the schema).
    #[serde(default)]
    #[schemars(
        description = "read-only SQL SELECT over the reasoning graph; empty (or omitted) returns the schema instead of running a query"
    )]
    sql: String,
}

#[derive(Debug, Deserialize)]
struct Empty {}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct ConstraintSolveArgs {
    #[schemars(description = "Source document source_path (which document's constraints to load)")]
    source_path: String,
    #[schemars(description = "mode: validate | infer | check")]
    mode: String,
    #[serde(default)]
    #[schemars(description = "field assignments for validation mode (JSON object of field→value)")]
    assignment: Option<std::collections::HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct GraphBuildArgs {
    #[schemars(
        description = "Document owner source_path — compiles every *.csp limit table under .glossa/notes/<doc>/ into the constraint graph (same as kb graph build)"
    )]
    doc: String,
    #[serde(default)]
    #[schemars(
        description = "Optional directory of .csp files; default is the document's notebook mirror"
    )]
    tables_dir: Option<String>,
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
        description = "Note filename with extension — free-form; choose any extension that fits the content (e.g. `.md` for a report or table, `.txt` for a jotting). Special case: `.csp` is a validated limit table (tab-separated rows, first line = column headers) used only by the constraint-graph workflow (graph_build) — do not reach for it for ordinary notes"
    )]
    file: String,
    #[schemars(
        description = "Full file content (replaces the note if it exists, unless append=true). Free-form content is stored as-is. Only a `.csp` note is validated on write: the reply echoes the parsed columns and row count, and a malformed table (empty header cell, ragged row) is rejected without writing"
    )]
    content: String,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_bool_loose"
    )]
    #[schemars(
        description = "Add to the existing file instead of replacing it. For a `.csp`, content is data rows placed under the existing header (repeating the header line is fine — it is deduplicated); for other files, content is appended as-is. Pass JSON boolean true (string \"true\" is also accepted)."
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
pub(crate) struct GrepArgs {
    #[schemars(
        description = "The text to find. It is a regex by default, so `A|B` matches A or B, `[0-9]+` matches digits, `.` matches any character; plain text also works as-is. For a value list, grep one value or the parameter name with `context` to pull its table window."
    )]
    pattern: String,
    #[schemars(
        description = "Search only this document — the same path `glob`/`read`/`search` show (a trailing `#chunk` is ignored). Omit to search the whole base."
    )]
    #[serde(default)]
    path: Option<String>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_bool_loose"
    )]
    #[schemars(description = "case-insensitive matching (-i)")]
    ignore_case: Option<bool>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_bool_loose"
    )]
    #[schemars(
        description = "Match the pattern as literal characters, with no regex meaning (`|`, `.`, `*` match themselves). Usually leave this off — use it only to find text that itself contains regex symbols (-F)."
    )]
    fixed: Option<bool>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_bool_loose"
    )]
    #[schemars(description = "match whole words only (-w)")]
    word: Option<bool>,
    #[serde(default)]
    #[schemars(
        description = "only files whose path matches this ripgrep -g glob, e.g. **/* (all) or *<name-fragment>*"
    )]
    glob: Option<String>,
    #[serde(default)]
    #[schemars(description = "restrict to a single file type (-t)")]
    file_type: Option<String>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_usize_loose"
    )]
    #[schemars(
        description = "Return N lines around each match — this turns a grep into a focused window read. To pull a value's whole table, grep one of its values (or the parameter name) with context ~20-40, instead of reading the whole document. Both sides; -A/-B override a side (-C)."
    )]
    context: Option<usize>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_usize_loose"
    )]
    #[schemars(description = "emit N context lines before each match (-B)")]
    before: Option<usize>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_usize_loose"
    )]
    #[schemars(description = "emit N context lines after each match (-A)")]
    after: Option<usize>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_bool_loose"
    )]
    #[schemars(
        description = "print only the matched substring(s), one per line, not the whole line (-o)"
    )]
    only_matching: Option<bool>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_bool_loose"
    )]
    #[schemars(
        description = "Show each match's line number within the chunk — the position of the hit, so you can point at where a value sits or read a window around that line (-n)."
    )]
    line_number: Option<bool>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_bool_loose"
    )]
    #[schemars(
        description = "output only a count of matching lines per chunk, not the lines (-c)"
    )]
    count: Option<bool>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_usize_loose"
    )]
    #[schemars(description = "stop after N matching lines per chunk (-m)")]
    max_count: Option<usize>,
    #[serde(
        default,
        deserialize_with = "crate::json_util::deserialize_opt_bool_loose"
    )]
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

#[allow(clippy::too_many_arguments)]
fn read_common(
    root: &std::path::Path,
    idx: &crate::index::store::DocIndex,
    g: Option<&GraphStore>,
    path: &str,
    n: u64,
    page_image: bool,
    include_images: bool,
    trace: &crate::trace::TraceLog,
) -> CallToolResult {
    let out = crate::tools::read(root, idx, g, path, n, page_image, trace);
    let mut content = Vec::new();
    // Images ride out as JPEG: base64-PNG is what overflows the stdio JSON-RPC frame on
    // figure-heavy pages, and JPEG is far smaller. `to_jpeg` passes real JPEGs through untouched.
    if page_image {
        if !out.text.is_empty() {
            content.push(Content::text(out.text));
        }
        for img in out.images {
            let img = crate::read::to_jpeg(img);
            let b64 = base64::engine::general_purpose::STANDARD.encode(&img.bytes);
            content.push(Content::image(b64, img.mime));
        }
    } else {
        content.push(Content::text(out.text));
        if include_images {
            for img in out.images {
                let img = crate::read::to_jpeg(img);
                let b64 = base64::engine::general_purpose::STANDARD.encode(&img.bytes);
                content.push(Content::image(b64, img.mime));
            }
        }
    }
    CallToolResult::success(content)
}

#[tool_router]
impl GlossaServer {
    // keep in sync with registry::DESC_SEARCH (rmcp's #[tool(description=…)] rejects a
    // non-literal path expr; the mcp_tool_list_matches_registry test enforces byte-equality).
    #[tool(
        description = "Full-text search over the knowledge base — natural-language keywords (morphology-aware, BM25-ranked), NOT a regex. Returns ranked hits, one per line as `[#n] path · label · snippet`. Open a hit with `read(path, n)` using that `[#n]` number. Scope with optional glob/file_type filters; for an exact token or code use `grep` instead. Hits are ranked best-first — the top few usually contain the answer, so read those rather than running many searches."
    )]
    async fn search(
        &self,
        Parameters(a): Parameters<SearchArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.freshen_now().await;
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

    // keep in sync with registry::DESC_READ (see search's comment above for why this is a literal).
    #[tool(
        description = "Read material by reference. Usually a document chunk: pass the `path` and chunk number `n` (the `[#n]` from a search/grep result; for PDFs the page). It returns the chunk's WHOLE text — for a large chunk that is a lot, and a table in its middle is easy to under-read; when you only need a value or its table, `grep` that value with `context` and read just the window instead. If a PDF table page is hard to read as text, call read again with `page_image: true` to return a 200 DPI JPEG instead (requires the server started with --vision). Returns the full text plus prev/next chunk numbers; if `n` is out of range the reply states the valid range. You may ALSO pass a graph NODE id (e.g. a Resolution id from a `glossary` line) as `path` — then it returns that node plus every evidence chunk it and its 1-hop chain MENTION, each labelled with where it came from."
    )]
    async fn read(&self, Parameters(a): Parameters<ReadArgs>) -> Result<CallToolResult, McpError> {
        self.freshen_now().await;
        self.lazy_reindex_if_changed(&a.path);
        let (idx, g) = self.open_index_graph()?;
        let page_image = !self.no_image && a.page_image.unwrap_or(false);
        let include_images = !self.no_image && a.include_images.unwrap_or(true);
        Ok(read_common(
            &self.root,
            &idx,
            g.as_ref(),
            &a.path,
            a.n as u64,
            page_image,
            include_images,
            &self.trace,
        ))
    }

    #[tool(
        description = "Deliver the ORIGINAL source file behind a citation to the user for source attribution — NOT for reading its text (use `read` for content). Pass the document `path` from a search/grep result and, for a PDF, the cited page `n`. Returns the file as an embedded resource the client can preview or download, plus a one-line note of what was delivered. A large PDF is delivered as just the cited page (still a real, text-bearing PDF); an oversize non-PDF, or an oversize ref with no page, returns guidance to cite a specific PDF page. Read-only; available in every profile. A DOCX is delivered as PDF by default (source format renders inconsistently across clients); pass `raw: true` to get the original .docx."
    )]
    async fn get_source_file(
        &self,
        Parameters(a): Parameters<SourceFileArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.freshen_now().await;
        self.lazy_reindex_if_changed(&a.path);
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let g = GraphStore::open(&self.root).ok();
        let max = a
            .max_bytes
            .unwrap_or(crate::tools::DEFAULT_SOURCE_MAX_BYTES);
        let out = crate::tools::get_source_file(
            &idx,
            g.as_ref(),
            &a.path,
            a.n.map(u64::from),
            max,
            a.raw.unwrap_or(false),
        );
        let mut content = vec![Content::text(out.text)];
        if let Some(f) = out.file {
            let b64 = base64::engine::general_purpose::STANDARD.encode(&f.bytes);
            content.push(Content::resource(
                ResourceContents::blob(b64, f.filename).with_mime_type(f.mime),
            ));
        }
        Ok(CallToolResult::success(content))
    }

    // keep in sync with registry::DESC_GLOSSARY (see search's comment above for why this is a literal).
    #[tool(
        description = "Resolve a concept (a symptom, error, component or task in a few words) to graph nodes. A reasoning node prints its `id [type] label` followed by its full chain — cause → resolution — each with a `read path #n` anchor, so ONE call gives you the likely fix. The line may also show `· comm N · pr …` — the problem cluster id. After a hit, call `related(<that node id>)` to list alternate and related cases before searching again. Structural Section/Document nodes show their `path #n` anchor. Empty result = nothing matches yet. Morphology-aware over labels/aliases. Also call it before creating a node, to REUSE an existing one."
    )]
    async fn glossary(
        &self,
        Parameters(a): Parameters<GlossaryArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.freshen_now().await;
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let spec = crate::tools::ChainSpec::from_ontology(&Ontology::load_or_default(&self.root));
        let stale = crate::tools::StaleChecker::new(self.root.clone());
        Ok(CallToolResult::success(vec![Content::text(
            crate::tools::glossary_with_query(
                &idx,
                &g,
                &a.name,
                Some(a.query.as_str()).filter(|s| !s.is_empty()),
                &spec,
                &self.trace,
                a.as_of.as_deref(),
                Some(&stale),
            ),
        )]))
    }

    #[tool(
        description = "Broaden a `glossary` hit — list OTHER solved cases linked to the same node. Call AFTER `glossary` when the cause→resolution chain is close but not quite right, you want alternates, or before running another search. Pass the reasoning-node `node` id copied from the glossary line (the token before `[Symptom]`/`[Cause]`/`[Resolution]`, e.g. `sym:...`), or a chunk `path` + `n`. Each line is prefixed and has a `read path #n` anchor: `SIMILAR` — paraphrase cases that share evidence; `COMMUNITY` — other nodes in the same problem cluster (same `comm N` as the glossary suffix), top by centrality. Empty → try another glossary term or fall back to search/grep. For the node's OWN chain, use `glossary` — not related."
    )]
    async fn related(
        &self,
        Parameters(a): Parameters<RelatedArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.freshen_now().await;
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let stale = crate::tools::StaleChecker::new(self.root.clone());
        Ok(CallToolResult::success(vec![Content::text(
            crate::tools::related(
                &idx,
                &g,
                a.node.as_deref(),
                a.path.as_deref(),
                a.n,
                &self.trace,
                a.as_of.as_deref(),
                Some(&stale),
            ),
        )]))
    }

    #[tool(
        description = "List a node's DIRECT structural edges (its actual typed relationships — e.g. what it REFERENCES, what CONSTRAINS it), one hop, each with the real edge direction (-> outgoing, <- incoming) and a `read path #n` anchor. Pass a `node` id (from `glossary`) or a chunk `path`+`n`. Filter with `edge_types` (relation names) and `direction` (out/in/both). This is FACTUAL graph structure — for fuzzy 'similar cases' use `related`; for how two nodes connect (possibly across documents) use `reach`. Empty => no such edges."
    )]
    async fn neighbors(
        &self,
        Parameters(a): Parameters<NeighborsArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.freshen_now().await;
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let direction = a.direction.as_deref().unwrap_or("both");
        let stale = crate::tools::StaleChecker::new(self.root.clone());
        Ok(CallToolResult::success(vec![Content::text(
            crate::tools::neighbors(
                &idx,
                &g,
                a.node.as_deref(),
                a.path.as_deref(),
                a.n,
                a.edge_types.as_deref(),
                direction,
                &self.trace,
                a.as_of.as_deref(),
                Some(&stale),
            ),
        )]))
    }

    // keep in sync with registry::DESC_REACH (see search's comment above for why this is a literal).
    #[tool(
        description = "Cross-document reasoning bridge — the ONE traversal tool, two directions. Omit `to` for DISCOVERY: walk `relation` forward from `from`, crossing document boundaries on shared mentions (the bridge, on by default), and return every node reached as a candidate answer — use this to resolve a relational multi-hop instead of inferring it from prose. Pass `to` for VERIFY: does a grounded path from `from` to that specific candidate exist (a self-check on an answer you already produced)? `relation` fuzzy-matches an ontology edge type (omit = all chaining relations, undirected). Each hop prints its real edge direction (--REL--> / <--REL--) with a `read path #n` anchor, or `↝ bridged on \"<term>\"` where the reasoning crossed a document — never a silent jump. Give `from`/`to` as node ids (from `glossary`) or as `from_path`+`from_n` / `to_path`+`to_n` chunk refs. `max_depth` defaults to 6 (max 12); `bridge` defaults to true (false = graph-only, in-document connectivity — this reproduces the old `path` tool). For a node's own direct edges use `neighbors`."
    )]
    async fn reach(
        &self,
        Parameters(a): Parameters<ReachArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.freshen_now().await;
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let ont = Ontology::load_or_default(&self.root);
        Ok(CallToolResult::success(vec![Content::text(
            crate::tools::reach(
                &idx,
                &g,
                &ont,
                a.from.as_deref(),
                a.from_path.as_deref(),
                a.from_n,
                a.relation.as_deref(),
                a.to.as_deref(),
                a.to_path.as_deref(),
                a.to_n,
                a.max_depth.unwrap_or(6),
                a.bridge.unwrap_or(true),
                &self.trace,
            ),
        )]))
    }

    #[tool(
        description = "Update the index + structural graph. No args: incremental over the whole knowledge base. force=true: full rebuild from scratch. path=<doc>: reindex just that one document (picks up an in-place edit)."
    )]
    async fn index(
        &self,
        Parameters(a): Parameters<IndexArgs>,
    ) -> Result<CallToolResult, McpError> {
        let started = std::time::Instant::now();
        if let Some(p) = a.path.as_deref() {
            let idx =
                crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
            let Some(rel) = idx.canonical_document_path(p) else {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "not an indexed document: {p}"
                ))]));
            };
            // Explicit call — bounded wait for the lock (mirror freshen_blocking's loop, ~3s).
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
            let mut done = false;
            loop {
                if let Some(_g) = crate::index::lock::try_index_lock(&self.root) {
                    crate::index::store::index_one_file_locked(&self.root, &rel)
                        .map_err(internal)?;
                    done = true;
                    break;
                }
                if std::time::Instant::now() >= deadline {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            if done {
                self.mark_dirty();
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "reindexed {rel} in {}",
                    crate::cli_fmt::format_elapsed(started.elapsed())
                ))]));
            }
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "index busy (another process is indexing), try again: {rel}"
            ))]));
        }
        let s = index_dir(&self.root, a.force.unwrap_or(false)).map_err(internal)?;
        self.mark_dirty();
        Ok(CallToolResult::success(vec![Content::text(format!(
            "indexed: {} added, {} removed, {} unchanged in {}",
            s.added,
            s.removed,
            s.unchanged,
            crate::cli_fmt::format_elapsed(started.elapsed())
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

    #[cfg_attr(not(feature = "constraint"), allow(dead_code))]
    #[cfg_attr(not(feature = "constraint"), allow(unused_variables))]
    #[tool(
        description = "Compile all .csp limit tables for doc into the constraint graph (kb graph build). On success: capability scan and per-Field shapes; on failure: compiler-style errors with file/row/field hints. Fix tables with read/note, then call again."
    )]
    async fn graph_build(
        &self,
        Parameters(args): Parameters<GraphBuildArgs>,
    ) -> Result<CallToolResult, McpError> {
        crate::audit::security_event("write", "tool_invoke", "invoked", "-", "graph_build");
        #[cfg(feature = "constraint")]
        {
            let idx =
                crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
            let g = GraphStore::open(&self.root).map_err(internal)?;
            let ont = Ontology::load_or_default(&self.root);
            let tables_dir = args.tables_dir.as_deref().map(std::path::Path::new);
            let msg = crate::tools::graph_build(&self.root, &idx, &g, &ont, &args.doc, tables_dir);
            return Ok(CallToolResult::success(vec![Content::text(msg)]));
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
        description = "Create/update reasoning nodes and directed edges. Each node needs a human-readable `label`, `node_type`, and indexed `source_path`. Reference endpoints in `edges` by label, or a document section as `<path>#<n>` where `<n>` is the INTEGER chunk number exactly as a search/grep/read shows it (the `[#n]` / `path:#n:`) — just that number, with nothing appended (no clause like `4.1`, no note in parentheses). When two nodes share a label but differ in type, a bare label is ambiguous — reference the endpoint by its node id (shown on every graph tool line) or qualify it as `Type:label` (e.g. `Symptom:cache`). The response lists written node ids and resolved edges. Send a node and edges that reference it in the same call."
    )]
    async fn graph_upsert(
        &self,
        Parameters(a): Parameters<GraphUpsertArgs>,
    ) -> Result<CallToolResult, McpError> {
        crate::audit::security_event("write", "tool_invoke", "invoked", "-", "graph_upsert");
        self.freshen_now().await;
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let ont = Ontology::load_or_default(&self.root);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let (nodes, edges, parse_notes) = (a.nodes, a.edges, a.parse_notes);
        let res = crate::graph::lock::with_graph_write_lock(
            &self.root,
            std::time::Duration::from_secs(5),
            || Ok(crate::graph::ops::graph_upsert(&idx, &g, &ont, nodes, edges, now)),
        );
        let message = match res {
            Ok(out) if parse_notes.is_empty() => out.message,
            Ok(out) => format!("{}\n{}", parse_notes.join("\n"), out.message),
            Err(e) => e.to_string(),
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
        crate::audit::security_event("write", "tool_invoke", "invoked", "-", "graph_delete");
        self.freshen_now().await;
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
        let nodes = a.nodes;
        let msg = match crate::graph::lock::with_graph_write_lock(
            &self.root,
            std::time::Duration::from_secs(5),
            || Ok(crate::graph::ops::graph_delete(&idx, &g, nodes, refs)),
        ) {
            Ok(m) => m,
            Err(e) => e.to_string(),
        };
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
        let updates = a.into_updates();
        let msg = match crate::graph::lock::with_graph_write_lock(
            &self.root,
            std::time::Duration::from_secs(5),
            || Ok(crate::graph::ops::graph_update(&g, updates)),
        ) {
            Ok(m) => m,
            Err(e) => e.to_string(),
        };
        Ok(CallToolResult::success(vec![Content::text(msg)]))
    }

    #[tool(
        description = "Recompute the graph's DERIVED layer from what is currently stored: transitive-closure edges, SIMILAR links, communities and centrality (these surface in `glossary`/`related`). Non-destructive — it never deletes or merges nodes. Run it after a batch of edits to refresh the derived view."
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
        description = "Diagnose graph health: the three doubts — ungrounded nodes (a requires-grounding node with no live MENTIONS), stale nodes (the source file backing a node has changed since it was recorded), and incomplete nodes (off-spine / degenerate, on no complete reasoning chain). Report-only — never mutates the graph. Use it to decide what to re-ground, re-verify, or clean up; pruning incomplete/ungrounded nodes is CLI-only (`kb graph doctor --prune-incomplete` / `--prune-ungrounded`)."
    )]
    async fn graph_doctor(
        &self,
        Parameters(_): Parameters<Empty>,
    ) -> Result<CallToolResult, McpError> {
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let ont = Ontology::load_or_default(&self.root);
        Ok(CallToolResult::success(vec![Content::text(
            crate::graph::ops::graph_doctor(&g, &ont, &self.root),
        )]))
    }

    #[tool(
        description = "Universal graph statistics. Default (summary) mode: node counts by type and edge counts by relation, plus a per-community overview (each community's size and up to eight nodes ranked by centrality: `id [type] label`, PageRank). Pass a document (its path — under `doc` OR `node`, resolved leniently) to list that document's owned non-structural nodes (`source_path` scope only) with all outgoing edges (type → target label or path#n). Ontology-independent. Pass `node` (a node id) to switch to node-inspection mode: everything about that one node — id, type, label, aliases, and every outgoing/incoming edge with the neighbour's label."
    )]
    async fn graph_stats(
        &self,
        Parameters(a): Parameters<GraphStatsArgs>,
    ) -> Result<CallToolResult, McpError> {
        let g = GraphStore::open(&self.root).map_err(internal)?;
        // Lenient doc resolution: an argument that names a DOCUMENT — whether under
        // `doc` or `node`, and even if slightly mistyped — routes to that document's
        // owned-node inventory (+ MENTIONS), so `graph_stats("<doc>")`
        // works regardless of the arg key. A `node` that is a real graph-node id still
        // gets node-inspection.
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let doc = a
            .doc
            .as_deref()
            .map(|d| {
                idx.canonical_document_path(d)
                    .unwrap_or_else(|| d.to_string())
            })
            .or_else(|| {
                a.node
                    .as_deref()
                    .and_then(|s| idx.canonical_document_path(s))
            });
        if let Some(doc) = doc {
            let mut out = crate::tools::graph_stats(&g);
            out.push('\n');
            out.push_str(&crate::tools::checklist_coverage_report(&g, &doc));
            return Ok(CallToolResult::success(vec![Content::text(out)]));
        }
        if let Some(id) = a.node.as_deref() {
            return Ok(CallToolResult::success(vec![Content::text(
                crate::tools::node_inspect(&g, id),
            )]));
        }
        Ok(CallToolResult::success(vec![Content::text(
            crate::tools::graph_stats(&g),
        )]))
    }

    // keep in sync with registry::DESC_SQL (see search's comment above for why this is a literal).
    #[tool(
        description = "Run a read-only SQL SELECT over the reasoning graph to compute/aggregate/rank/filter/traverse-by-join over facts and edges; an empty query returns the schema. Tables: nodes(id, node_type, label), edges(efrom, edge_type, eto), node_validity(node_id, valid_from, ...), edges_labeled(src_label, edge_type, dst_label, efrom, eto)."
    )]
    async fn sql(
        &self,
        Parameters(a): Parameters<GraphQueryArgs>,
    ) -> Result<CallToolResult, McpError> {
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(
            crate::tools::sql(&idx, &g, &a.sql, &self.trace),
        )]))
    }

    // keep in sync with registry::DESC_GREP (see search's comment above for why this is a literal).
    #[tool(
        description = "Find an exact string in the text — a code, identifier, parameter name, or a value (e.g. `maxTsdr`, `M6`, `250`). ripgrep regex supported; smart-case. Use it whenever you know a precise token to locate (beats keyword `search`; for fuzzy/conceptual lookup use `search`). TO READ A TABLE, grep one of its values with `context` set to ~20-40: the reply then carries that many lines around each hit — a focused window onto the table — so you get the whole column in one call without reading the entire chunk. Returns matching lines as `path:#n: line`; a context line uses `-` instead of `:`. Reach for `read(path, n)` only when you actually need a whole chunk, not to locate a value. Other flags mirror ripgrep: -i/-F/-w, -o only-matching, -n line-number, -c count, -m max-count, -U multiline."
    )]
    async fn grep(&self, Parameters(a): Parameters<GrepArgs>) -> Result<CallToolResult, McpError> {
        self.freshen_now().await;
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
            path: a.path,
        };
        Ok(CallToolResult::success(vec![Content::text(
            crate::tools::grep(
                &self.root,
                &idx,
                &a.pattern,
                &opts.with_default_context(),
                &self.trace,
            ),
        )]))
    }

    // keep in sync with registry::DESC_GLOB (see search's comment above for why this is a literal).
    #[tool(
        description = "List knowledge-base documents whose path matches a ripgrep `-g` glob (e.g. `*` or `**/*` for all documents, or `*<name-fragment>*` to find a file by name). Returns one `path  (N chunks)` per line — use it to discover what documents exist or find a file by name, then `read(path, n)` or scope a `search`/`grep` to it. N is the document's last page/section number; every page 1..N is addressable (blank pages return empty text)."
    )]
    async fn glob(&self, Parameters(a): Parameters<GlobArgs>) -> Result<CallToolResult, McpError> {
        self.freshen_now().await;
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(
            crate::tools::glob(&idx, &a.pattern, &self.trace),
        )]))
    }

    #[cfg_attr(not(feature = "notebook"), allow(dead_code))]
    #[tool(
        description = "Create, fully replace, or (with append=true) extend a notebook note bound to an indexed document. Notes are free-form: pick any extension (e.g. `.md`) that fits the content. Without append, an existing file is OVERWRITTEN; pass append=true to add to it. The `.csp` extension is a special validated limit-table format (tab-separated rows, first line = column headers) used only by the constraint-graph workflow — don't use it for ordinary notes. Use ls/read/del with paths from ls afterward."
    )]
    async fn note(&self, Parameters(a): Parameters<NoteArgs>) -> Result<CallToolResult, McpError> {
        crate::audit::security_event("write", "tool_invoke", "invoked", "-", "note");
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
        description = "List notebook notes. Optional doc filter. Paths in the output are arguments for read/del."
    )]
    async fn ls(&self, Parameters(a): Parameters<LsArgs>) -> Result<CallToolResult, McpError> {
        self.freshen_now().await;
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
        crate::audit::security_event("write", "tool_invoke", "invoked", "-", "del");
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

// Must use the instance router — bare `#[tool_handler]` defaults to
// `Self::tool_router()` (a fresh router), so profile `disable_route` never reaches tools/list.
#[tool_handler(router = self.tool_router)]
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
    fn arg_structs_accept_stringified_primitives() {
        // LLM/MCP clients sometimes JSON-encode primitives as strings; every arg field must coerce.
        let r: ReadArgs =
            serde_json::from_str(r#"{"path":"a.pdf","n":"8","page_image":"true"}"#).unwrap();
        assert_eq!(r.n, 8);
        assert_eq!(r.page_image, Some(true));

        let s: SourceFileArgs =
            serde_json::from_str(r#"{"path":"a.pdf","n":"3","max_bytes":"1024","raw":"true"}"#)
                .unwrap();
        assert_eq!(s.n, Some(3));
        assert_eq!(s.max_bytes, Some(1024));
        assert_eq!(s.raw, Some(true));

        let se: SearchArgs = serde_json::from_str(r#"{"query":"x","limit":"5"}"#).unwrap();
        assert_eq!(se.limit, Some(5));

        let ix: IndexArgs = serde_json::from_str(r#"{"force":"true","path":"a.md"}"#).unwrap();
        assert_eq!(ix.force, Some(true));
        assert_eq!(ix.path, Some("a.md".to_string()));

        let g: GrepArgs = serde_json::from_str(
            r#"{"pattern":"x","ignore_case":"true","context":"20","multiline":"1"}"#,
        )
        .unwrap();
        assert_eq!(g.ignore_case, Some(true));
        assert_eq!(g.context, Some(20));
        assert_eq!(g.multiline, Some(true));

        let ne: NeighborsArgs = serde_json::from_str(
            r#"{"node":"sym:x","n":"3","edge_types":"REFERENCES","direction":"out","as_of":"2022"}"#,
        )
        .unwrap();
        assert_eq!(ne.n, Some(3));
        assert_eq!(ne.edge_types, Some(vec!["REFERENCES".to_string()]));
        assert_eq!(ne.direction, Some("out".to_string()));
        assert_eq!(ne.as_of, Some("2022".to_string()));

        // as_of also accepts a bare JSON number (a model writing a year unquoted).
        let ne2: NeighborsArgs =
            serde_json::from_str(r#"{"node":"sym:x","as_of":2022}"#).unwrap();
        assert_eq!(ne2.as_of, Some("2022".to_string()));

        let gl: GlossaryArgs =
            serde_json::from_str(r#"{"name":"loss","query":"why is the connection lost","as_of":"2022-06-01"}"#).unwrap();
        assert_eq!(gl.as_of, Some("2022-06-01".to_string()));
        assert_eq!(gl.query, "why is the connection lost");

        let re: RelatedArgs =
            serde_json::from_str(r#"{"node":"sym:x","as_of":2022}"#).unwrap();
        assert_eq!(re.as_of, Some("2022".to_string()));

        let ra: ReachArgs = serde_json::from_str(
            r#"{"from":"sym:a","to":"sym:b","from_n":"1","to_n":"2","max_depth":"9","relation":"located","bridge":"false"}"#,
        )
        .unwrap();
        assert_eq!(ra.from_n, Some(1));
        assert_eq!(ra.to_n, Some(2));
        assert_eq!(ra.max_depth, Some(9));
        assert_eq!(ra.relation, Some("located".to_string()));
        assert_eq!(ra.bridge, Some(false));

        // relation also accepts a bare JSON number (mirrors as_of above); bridge accepts a bare bool.
        let ra2: ReachArgs =
            serde_json::from_str(r#"{"from":"sym:a","relation":2022,"bridge":true}"#).unwrap();
        assert_eq!(ra2.relation, Some("2022".to_string()));
        assert_eq!(ra2.bridge, Some(true));

        // Native JSON types still deserialize; absent optionals stay None.
        let r2: ReadArgs = serde_json::from_str(r#"{"path":"a.pdf","n":2}"#).unwrap();
        assert_eq!((r2.n, r2.page_image), (2, None));

        // `sql` is a plain required-with-default string: missing -> "", present -> the value.
        let gq: GraphQueryArgs = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(gq.sql, "");
        let gq2: GraphQueryArgs =
            serde_json::from_str(r#"{"sql":"select count(*) from nodes"}"#).unwrap();
        assert_eq!(gq2.sql, "select count(*) from nodes");
    }

    #[test]
    fn empty_param_schema_has_properties() {
        // no-arg tools (graph_generalize/purge) must expose an explicit
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
    fn baseline_sig_caches_until_manifest_mtime_moves() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nx\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let srv = GlossaServer::new(
            dir.path().to_path_buf(),
            Profile::Editor,
            false,
            ServerFlags::default(),
        );
        let s1 = srv.baseline_sig("a.md");
        assert!(s1.is_some(), "known file has a baseline sig");
        assert_eq!(srv.baseline_sig("a.md"), s1, "repeat lookup is stable");
        assert_eq!(srv.baseline_sig("missing.md"), None, "unknown file -> None");
        // After a real reindex the manifest mtime advances → cache picks up the new sig.
        std::thread::sleep(std::time::Duration::from_millis(10));
        std::fs::write(dir.path().join("a.md"), b"# A\nx changed bigger\n").unwrap();
        index_dir(dir.path(), false).unwrap();
        assert_eq!(
            srv.baseline_sig("a.md"),
            Some(crate::index::store::file_sig(&dir.path().join("a.md")).unwrap()),
            "cache reloaded after mtime bump"
        );
    }

    #[tokio::test]
    async fn graph_upsert_reports_busy_when_graph_lock_held() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nx\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let srv = GlossaServer::new(dir.path().to_path_buf(), Profile::Editor, false, ServerFlags::default());
        // Another process holds the agent-graph write lock.
        std::fs::create_dir_all(dir.path().join(".glossa")).unwrap();
        let held = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(dir.path().join(".glossa").join("graph.lock"))
            .unwrap();
        held.try_lock().unwrap();
        let out = srv
            .graph_upsert(Parameters(GraphUpsertArgs {
                nodes: vec![],
                edges: vec![],
                parse_notes: vec![],
            }))
            .await
            .unwrap();
        assert!(
            format!("{out:?}").contains("retry"),
            "graph_upsert must surface the busy message, not run under contention: {out:?}"
        );
        fs4::FileExt::unlock(&held).unwrap();
    }

    #[tokio::test]
    async fn read_picks_up_an_in_place_edit() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\noldbody\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let srv = GlossaServer::new(
            dir.path().to_path_buf(),
            Profile::Editor,
            false,
            ServerFlags::default(),
        );
        // In-place edit (no dir-mtime change → B1 freshen alone would miss it).
        std::thread::sleep(std::time::Duration::from_millis(10));
        std::fs::write(dir.path().join("a.md"), b"# A\nnewbody freshtoken\n").unwrap();
        // A read of the edited doc triggers the lazy per-file reindex.
        let _ = srv
            .read(Parameters(ReadArgs {
                path: "a.md".into(),
                n: 1,
                page_image: None,
                include_images: None,
            }))
            .await;
        // Now it is searchable.
        let out = srv
            .search(Parameters(SearchArgs {
                query: "freshtoken".into(),
                limit: None,
                glob: None,
                file_type: None,
            }))
            .await
            .unwrap();
        assert!(
            format!("{out:?}").contains("a.md"),
            "in-place edit picked up after reading the file"
        );
    }

    #[cfg(feature = "notebook")]
    #[tokio::test]
    async fn read_picks_up_an_external_note_edit() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("doc.md"), b"# Doc\nbody\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let srv = GlossaServer::new(
            dir.path().to_path_buf(),
            Profile::Editor,
            false,
            ServerFlags::default(),
        );
        // Create the note via the tool (write-through indexes it).
        srv.note(Parameters(NoteArgs {
            doc: "doc.md".into(),
            file: "n.csp".into(),
            content: "oldtoken\n".into(),
            append: None,
        }))
        .await
        .unwrap();
        // Edit the note file EXTERNALLY (bypassing the tool) — dir mtime of notes/doc.md changes only on
        // add/remove, not on a content edit, so freshen alone would miss it.
        std::thread::sleep(std::time::Duration::from_millis(10));
        std::fs::write(
            dir.path()
                .join(".glossa")
                .join("notes")
                .join("doc.md")
                .join("n.csp"),
            b"newtoken freshnote",
        )
        .unwrap();
        // A read of the note triggers the lazy note reindex.
        let _ = srv
            .read(Parameters(ReadArgs {
                path: "doc.md/n.csp".into(),
                n: 1,
                page_image: None,
                include_images: None,
            }))
            .await;
        let out = srv
            .search(Parameters(SearchArgs {
                query: "freshnote".into(),
                limit: None,
                glob: None,
                file_type: None,
            }))
            .await
            .unwrap();
        assert!(
            format!("{out:?}").contains("doc.md/n.csp"),
            "external note edit picked up after reading the note"
        );
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
        let srv = GlossaServer::new(
            dir.path().to_path_buf(),
            Profile::Editor,
            false,
            ServerFlags::default(),
        );
        let path = "d.md".to_string(); // canonical key: corpus-root-relative

        let out = srv
            .read(Parameters(ReadArgs {
                path,
                n: 1,
                include_images: Some(false),
                page_image: None,
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
        let srv = GlossaServer::new(
            dir.path().to_path_buf(),
            Profile::Editor,
            false,
            ServerFlags::default(),
        );
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
            dir.path().join("MODULE.md"),
            "# A\none\n# B\ntwo\n".as_bytes(),
        )
        .unwrap();
        std::fs::write(dir.path().join("Other.md"), "# A\none\n".as_bytes()).unwrap();
        index_dir(dir.path(), true).unwrap();
        let srv = GlossaServer::new(
            dir.path().to_path_buf(),
            Profile::Editor,
            false,
            ServerFlags::default(),
        );
        let out = format!(
            "{:?}",
            srv.glob(Parameters(GlobArgs {
                pattern: "*MODULE*".into()
            }))
            .await
            .unwrap()
        );
        assert!(out.contains("MODULE"), "lists the matching doc: {out}");
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
        let srv = GlossaServer::new(
            dir.path().to_path_buf(),
            Profile::Editor,
            false,
            ServerFlags::default(),
        );
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
        let srv = GlossaServer::new(
            dir.path().to_path_buf(),
            Profile::Editor,
            false,
            ServerFlags::default(),
        );
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
                label: "link loss".into(),
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
        let srv = GlossaServer::new(
            dir.path().to_path_buf(),
            Profile::Editor,
            false,
            ServerFlags::default(),
        );
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
        let srv = GlossaServer::new(
            dir.path().to_path_buf(),
            Profile::Editor,
            false,
            ServerFlags::default(),
        );
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
        use fs4::FileExt;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nintro\n## B\nbody b\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        {
            let g = GraphStore::open(dir.path()).unwrap();
            g.put_node(&Node {
                id: "sym:x".into(),
                node_type: "Symptom".into(),
                label: "link loss".into(),
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
        let srv = GlossaServer::new(
            dir.path().to_path_buf(),
            Profile::Editor,
            false,
            ServerFlags::default(),
        );

        // Another editor holds the cross-process generalize lock.
        let lock_path = dir.path().join(".glossa").join("generalize.lock");
        let holder = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .unwrap();
        FileExt::try_lock(&holder).unwrap();

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
        let reader =
            GlossaServer::new(root.clone(), Profile::Reader, false, ServerFlags::default()).enabled_tools();
        assert!(reader.contains(&"search".to_string()) && reader.contains(&"read".to_string()));
        #[cfg(feature = "notebook")]
        assert!(reader.contains(&"ls".to_string()));
        assert!(!reader.contains(&"note".to_string()) && !reader.contains(&"del".to_string()));
        assert!(
            !reader.contains(&"index".to_string())
                && !reader.contains(&"graph_upsert".to_string())
                && !reader.contains(&"purge".to_string())
        );
        assert!(!reader.contains(&"write".to_string()));

        let editor =
            GlossaServer::new(root.clone(), Profile::Editor, false, ServerFlags::default()).enabled_tools();
        #[cfg(feature = "notebook")]
        assert!(editor.contains(&"note".to_string()) && editor.contains(&"ls".to_string()));
        assert!(editor.contains(&"index".to_string()) && editor.contains(&"resolve".to_string()));
        assert!(
            editor.contains(&"graph_generalize".to_string()),
            "editor exposes the non-destructive generalize tool"
        );
        assert!(
            editor.contains(&"graph_stats".to_string()),
            "editor exposes graph stats"
        );
        assert!(editor.contains(&"sql".to_string()), "editor exposes sql");
        assert!(
            editor.contains(&"graph_doctor".to_string()),
            "editor exposes the report-only graph_doctor tool"
        );
        #[cfg(feature = "constraint")]
        assert!(
            editor.contains(&"graph_build".to_string()),
            "editor exposes graph_build with constraint feature"
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
        // sql (formerly graph_query) is read-only and moved INTO the reader set.
        assert!(reader.contains(&"sql".to_string()), "reader can run sql");
        // Low-level / rarely-reached read tools are withheld from Reader to cut tool-choice clutter.
        assert!(
            !reader.contains(&"resolve".to_string())
                && !reader.contains(&"neighbors".to_string())
                && !reader.contains(&"related".to_string()),
            "reader does not get the decluttered read tools"
        );
        assert!(
            !reader.contains(&"graph_doctor".to_string()),
            "reader cannot graph_doctor"
        );

        let full =
            GlossaServer::new(root.clone(), Profile::Full, false, ServerFlags::default()).enabled_tools();
        assert!(full.contains(&"purge".to_string()));
        #[cfg(feature = "notebook")]
        assert!(full.contains(&"note".to_string()) && full.contains(&"del".to_string()));

        // resolve is a low-level primitive: kept for editor/full, withheld from Reader (a tool the
        // reader never calls in practice — clutter that muddies tool choice).
        assert!(!reader.contains(&"resolve".to_string()), "reader does not get resolve");
        for prof in [&editor, &full] {
            assert!(
                prof.contains(&"resolve".to_string()),
                "editor/full keep resolve"
            );
        }

        let ng = GlossaServer::new(root, Profile::Editor, false, ServerFlags { no_graph: true, ..Default::default() }).enabled_tools();
        assert!(ng.contains(&"search".to_string()) && ng.contains(&"read".to_string()));
        assert!(
            !ng.contains(&"related".to_string())
                && !ng.contains(&"graph_upsert".to_string())
                && !ng.contains(&"index".to_string())
        );
    }

    #[test]
    fn source_file_flag_gates_get_source_file() {
        let root = std::path::PathBuf::from(".");
        // On by default (every profile) …
        let on = GlossaServer::new(root.clone(), Profile::Reader, false, ServerFlags::default())
            .enabled_tools();
        assert!(
            on.contains(&"get_source_file".to_string()),
            "get_source_file is available by default"
        );
        // … and withheld when the --source-file opt-in is off (no_source_file), any profile.
        let off = GlossaServer::new(
            root,
            Profile::Full,
            false,
            ServerFlags { no_source_file: true, ..Default::default() },
        )
        .enabled_tools();
        assert!(
            !off.contains(&"get_source_file".to_string()),
            "no_source_file withholds get_source_file"
        );
    }

    #[tokio::test]
    async fn search_sees_a_file_added_after_startup() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nalpha\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let srv = GlossaServer::new(
            dir.path().to_path_buf(),
            Profile::Editor,
            false,
            ServerFlags::default(),
        );

        // File added AFTER the server exists (as an external agent would).
        std::thread::sleep(std::time::Duration::from_millis(10));
        std::fs::write(dir.path().join("b.md"), b"# B\nbravo\n").unwrap();

        let out = srv
            .search(Parameters(SearchArgs {
                query: "bravo".into(),
                limit: None,
                glob: None,
                file_type: None,
            }))
            .await
            .unwrap();
        let text = format!("{out:?}");
        assert!(
            text.contains("b.md"),
            "search must reflect the newly added file: {text}"
        );
    }

    #[tokio::test]
    async fn index_tool_reindexes_a_single_edited_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\noriginaltoken\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let srv = GlossaServer::new(
            dir.path().to_path_buf(),
            Profile::Editor,
            false,
            ServerFlags::default(),
        );
        std::fs::write(dir.path().join("a.md"), b"# A\nrewritetoken\n").unwrap();
        srv.index(Parameters(IndexArgs {
            force: None,
            path: Some("a.md".into()),
        }))
        .await
        .unwrap();
        let out = srv
            .search(Parameters(SearchArgs {
                query: "rewritetoken".into(),
                limit: None,
                glob: None,
                file_type: None,
            }))
            .await
            .unwrap();
        assert!(
            format!("{out:?}").contains("a.md"),
            "single-file index made the edit searchable"
        );
    }

    #[test]
    fn path_tool_is_gone_reach_replaces_it() {
        let dir = tempfile::tempdir().unwrap();
        let srv = GlossaServer::new(
            dir.path().to_path_buf(),
            Profile::Editor,
            false,
            ServerFlags::default(),
        );
        let names: Vec<String> = srv
            .tool_specs()
            .into_iter()
            .map(|t| t.name.to_string())
            .collect();
        assert!(names.contains(&"reach".to_string()), "reach tool present");
        assert!(!names.contains(&"path".to_string()), "path tool removed");
    }

    #[test]
    fn mcp_tool_list_matches_registry() {
        // Names + descriptions the MCP server advertises for every agent-facing (registry) tool
        // must equal `crate::tools::registry::registry()` byte-for-byte — the single guard that
        // keeps mcp.rs and the registry from drifting apart. Profile::Full is the superset profile
        // (keeps resolve/related/neighbors that Reader withholds as clutter), so it is a strict
        // superset of the registry names; the extra admin/structural tools (index, purge, resolve,
        // graph_upsert, ...) are expected and NOT compared.
        let dir = tempfile::tempdir().unwrap();
        let srv = GlossaServer::new(
            dir.path().to_path_buf(),
            Profile::Full,
            false,
            ServerFlags::default(),
        );
        let mcp: std::collections::BTreeMap<String, String> = srv
            .tool_specs()
            .into_iter()
            .map(|t| (t.name.to_string(), t.description.unwrap_or_default().to_string()))
            .collect();
        let reg = crate::tools::registry::registry();
        for d in &reg {
            let mcp_desc = mcp
                .get(d.name)
                .unwrap_or_else(|| panic!("MCP does not advertise registry tool {}", d.name));
            assert_eq!(
                mcp_desc, d.description,
                "description drift between mcp.rs and registry for {}",
                d.name
            );
        }
        let reg_names: std::collections::BTreeSet<_> = reg.iter().map(|d| d.name).collect();
        let mcp_names: std::collections::BTreeSet<_> = mcp.keys().map(String::as_str).collect();
        assert!(
            reg_names.is_subset(&mcp_names),
            "registry tool set must be a subset of what MCP advertises: reg={reg_names:?} mcp={mcp_names:?}"
        );
    }

    #[test]
    fn reindex_tool_is_gone_index_remains() {
        let dir = tempfile::tempdir().unwrap();
        let srv = GlossaServer::new(
            dir.path().to_path_buf(),
            Profile::Editor,
            false,
            ServerFlags::default(),
        );
        let names: Vec<String> = srv
            .tool_specs()
            .into_iter()
            .map(|t| t.name.to_string())
            .collect();
        assert!(names.contains(&"index".to_string()), "index tool present");
        assert!(
            !names.contains(&"reindex".to_string()),
            "reindex tool removed"
        );
    }

    // Slow (~3s): holds the index lock for the handler's whole bounded-wait deadline to force
    // the timeout path. Verifies the message is honest — no fabricated "reindexed" success, and
    // no mark_dirty() side effect — when another process keeps the lock the entire time.
    #[tokio::test]
    async fn index_tool_path_reports_busy_when_lock_never_frees() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\noriginaltoken\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let srv = GlossaServer::new(
            dir.path().to_path_buf(),
            Profile::Editor,
            false,
            ServerFlags::default(),
        );
        // Another process holds the index lock for the whole handler call.
        let _holder = crate::index::lock::try_index_lock(dir.path()).unwrap();
        let out = srv
            .index(Parameters(IndexArgs {
                force: None,
                path: Some("a.md".into()),
            }))
            .await
            .unwrap();
        let text = format!("{out:?}");
        assert!(text.contains("busy"), "reports busy: {text}");
        assert!(
            !text.contains("reindexed"),
            "must not claim success while the lock stayed held: {text}"
        );
    }

    // Regression for the note-skip in `lazy_reindex_if_changed`: a notebook-note path handed to
    // `read` must be served as a note, not swept into a corpus reindex. Before the explicit
    // `file_type_of(&rel) == Some("note")` check, this only worked by accident (the mis-resolved
    // `root.join(rel)` for a note doesn't exist, so `file_sig` errored and the handler bailed).
    #[cfg(feature = "notebook")]
    #[tokio::test]
    async fn read_of_notebook_note_does_not_reindex_it_as_a_corpus_doc() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\noriginal\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let srv = GlossaServer::new(
            dir.path().to_path_buf(),
            Profile::Editor,
            false,
            ServerFlags::default(),
        );
        srv.note(Parameters(NoteArgs {
            doc: "a.md".into(),
            file: "summary.md".into(),
            content: "note-marker-content".into(),
            append: None,
        }))
        .await
        .unwrap();

        let idx = crate::index::store::DocIndex::open_or_create(dir.path()).unwrap();
        assert_eq!(
            idx.file_type_of("a.md/summary.md").unwrap().as_deref(),
            Some("note"),
            "note indexed as file_type=note before read"
        );

        let out = srv
            .read(Parameters(ReadArgs {
                path: "a.md/summary.md".into(),
                n: 1,
                include_images: None,
                page_image: None,
            }))
            .await
            .unwrap();
        let text = format!("{out:?}");
        assert!(
            text.contains("note-marker-content"),
            "read must serve the note content, not a corpus-doc miss: {text}"
        );

        // The lazy reindex-on-read must leave the note alone — still file_type=note, not
        // clobbered by a corpus reindex of "a.md/summary.md".
        assert_eq!(
            idx.file_type_of("a.md/summary.md").unwrap().as_deref(),
            Some("note"),
            "read of a note path must not turn it into a corpus-doc chunk"
        );
    }
}
