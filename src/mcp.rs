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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile { Reader, Editor, Full }

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
}

const EDITOR_TOOLS: &[&str] = &["index", "reindex", "graph_upsert", "graph_delete", "graph_update", "graph_generalize"];
const FULL_TOOLS: &[&str] = &["purge"];
const GRAPH_TOOLS: &[&str] = &["glossary", "neighbors", "graph_upsert", "graph_delete", "graph_update", "graph_generalize", "resolve", "index", "reindex", "purge"];

impl GlossaServer {
    pub fn new(root: PathBuf, profile: Profile, trace: bool, no_graph: bool) -> Self {
        let mut router = Self::tool_router();
        if profile == Profile::Reader {
            for t in EDITOR_TOOLS.iter().chain(FULL_TOOLS) {
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
        let trace = if trace { crate::trace::TraceLog::to_dir(&root) } else { crate::trace::TraceLog::disabled() };
        Self { root, tool_router: router, trace }
    }

    /// Return the list of enabled tools (for config generation — not test-only).
    pub fn tool_specs(&self) -> Vec<rmcp::model::Tool> {
        self.tool_router.list_all()
    }

    #[cfg(test)]
    pub fn enabled_tools(&self) -> Vec<String> {
        self.tool_router.list_all().iter().map(|t| t.name.to_string()).collect()
    }
}

fn internal(e: anyhow::Error) -> McpError {
    McpError::internal_error(e.to_string(), None)
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchArgs {
    #[schemars(description = "natural-language keywords (Russian or English; morphology-aware, BM25-ranked) — NOT a regex")]
    query: String,
    #[serde(default)]
    #[schemars(description = "max hits (default 50)")]
    limit: Option<usize>,
    #[serde(default)]
    #[schemars(description = "only documents whose path matches this glob, e.g. *.pdf or *АБАК* (-g)")]
    glob: Option<String>,
    #[serde(default)]
    #[schemars(description = "only this file type, e.g. pdf (-t)")]
    file_type: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GlobArgs {
    #[schemars(description = "shell glob over document paths, e.g. *.pdf or *Safety*")]
    pattern: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ReadArgs {
    #[schemars(description = "document path, exactly as shown in a search result")]
    path: String,
    #[schemars(description = "chunk number to read, exactly as shown in `[#n]` in a search result (page number for PDFs)")]
    n: u32,
    #[serde(default)]
    #[schemars(description = "include embedded images (default true)")]
    include_images: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct NeighborsArgs {
    #[schemars(description = "document path, exactly as shown in a search result")]
    path: String,
    #[schemars(description = "chunk number, exactly as shown in `[#n]` in a search result")]
    n: u64,
    #[serde(default)]
    depth: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct NameArg { name: String }

#[derive(Debug, Deserialize)]
struct Empty {}

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
struct GrepArgs {
    #[schemars(description = "ripgrep-style regex pattern (literal text also works)")]
    pattern: String,
    #[serde(default)]
    #[schemars(description = "case-insensitive matching (-i)")]
    ignore_case: Option<bool>,
    #[serde(default)]
    #[schemars(description = "treat the pattern as a fixed string, not a regex (-F)")]
    fixed: Option<bool>,
    #[serde(default)]
    #[schemars(description = "match whole words only (-w)")]
    word: Option<bool>,
    #[serde(default)]
    #[schemars(description = "only files whose path matches this glob, e.g. *.pdf (-g)")]
    glob: Option<String>,
    #[serde(default)]
    #[schemars(description = "only this file type, e.g. pdf (-t)")]
    file_type: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GraphUpsertArgs {
    #[serde(default)]
    nodes: Vec<crate::graph::ops::UpsertNode>,
    #[serde(default)]
    edges: Vec<crate::graph::ops::UpsertEdge>,
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
}

#[tool_router]
impl GlossaServer {
    #[tool(description = "Full-text search over the knowledge base — natural-language keywords (Russian or English; morphology-aware, BM25-ranked), NOT a regex. Returns ranked hits, one per line as `[#n] path · label · snippet`. Open a hit with `read(path, n)` using that `[#n]` number. Scope with optional glob/file_type filters; for an exact token or code use `grep` instead. Hits are ranked best-first — the top few usually contain the answer, so read those rather than running many searches.")]
    async fn search(&self, Parameters(a): Parameters<SearchArgs>) -> Result<CallToolResult, McpError> {
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let (body, _hits) = crate::tools::search(&idx, &a.query, a.limit.unwrap_or(50), a.glob.as_deref(), a.file_type.as_deref(), &self.trace);
        Ok(CallToolResult::success(vec![Content::text(body)]))
    }

    #[tool(description = "Read a document chunk by its number `n` — the `[#n]` from a search/grep result (for PDFs, the page number). Returns the chunk's full text plus the previous/next chunk numbers for context expansion. If `n` is out of range, the reply states the document's valid chunk range.")]
    async fn read(&self, Parameters(a): Parameters<ReadArgs>) -> Result<CallToolResult, McpError> {
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let out = crate::tools::read(&idx, &a.path, a.n as u64, &self.trace);
        let mut content = vec![Content::text(out.text)];
        if a.include_images.unwrap_or(true) {
            for img in out.images {
                let b64 = base64::engine::general_purpose::STANDARD.encode(&img.bytes);
                content.push(Content::image(b64, img.mime));
            }
        }
        Ok(CallToolResult::success(content))
    }

    #[tool(description = "List existing graph nodes whose label matches a concept, one per line as `id [type] label` — reasoning nodes (Symptom/Cause/Resolution/Task) plus structural Section/Document (those also show a `path #n` anchor). Call it BEFORE creating a node to find and REUSE an existing one instead of duplicating; an empty result means nothing matches yet. Matching is morphology-aware over labels/aliases.")]
    async fn glossary(&self, Parameters(a): Parameters<NameArg>) -> Result<CallToolResult, McpError> {
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let g = GraphStore::open(&self.root).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(crate::tools::glossary(&idx, &g, &a.name, &self.trace))]))
    }

    #[tool(description = "Graph neighbors of a chunk — pass the document `path` and chunk number `n` (the `[#n]` from a search/grep result). Returns linked sections/documents as `RELATION  path  #n · label`; read any with `read(path, n)`. Direct (1-hop) neighbors.")]
    async fn neighbors(&self, Parameters(a): Parameters<NeighborsArgs>) -> Result<CallToolResult, McpError> {
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let g = GraphStore::open(&self.root).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(crate::tools::neighbors(&idx, &g, &a.path, a.n, a.depth.unwrap_or(1), &self.trace))]))
    }

    #[tool(description = "Build/update the index + structural graph for the knowledge base.")]
    async fn index(&self, Parameters(_): Parameters<Empty>) -> Result<CallToolResult, McpError> {
        let s = index_dir(&self.root, false).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(format!("indexed: {} added, {} removed, {} unchanged", s.added, s.removed, s.unchanged))]))
    }

    #[tool(description = "Rebuild the index + graph from scratch.")]
    async fn reindex(&self, Parameters(_): Parameters<Empty>) -> Result<CallToolResult, McpError> {
        let s = index_dir(&self.root, true).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(format!("reindexed: {} files", s.added))]))
    }

    #[tool(description = "Resolve a name to existing graph node ids (entity resolution).")]
    async fn resolve(&self, Parameters(a): Parameters<NameArg>) -> Result<CallToolResult, McpError> {
        let g = GraphStore::open(&self.root).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(g.resolve(&a.name).map_err(internal)?.join("\n"))]))
    }

    #[tool(description = "Create/update reasoning nodes and directed edges. A node carries a `label` (NO id — the system derives the id and de-duplicates by label) plus `node_type` and `source_path`. Reference nodes in `edges` by their `label` (or a section as `<path>#<n>`). A Resolution label names the fix ACTION, never the literal value; write labels as a broad reusable class in the knowledge base's language. Send a node and the edges referencing it in the same call so both endpoints exist.")]
    async fn graph_upsert(&self, Parameters(a): Parameters<GraphUpsertArgs>) -> Result<CallToolResult, McpError> {
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let ont = Ontology::load_or_default(&self.root);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let out = crate::graph::ops::graph_upsert(&idx, &g, &ont, a.nodes, a.edges, now);
        Ok(CallToolResult::success(vec![Content::text(out.message)]))
    }

    #[tool(description = "Remove reasoning nodes/edges from the graph by label — use it to delete a node or relation you added by mistake or that is no longer valid. Deleting a node also removes edges touching it.")]
    async fn graph_delete(&self, Parameters(a): Parameters<GraphDeleteArgs>) -> Result<CallToolResult, McpError> {
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let refs: Vec<crate::graph::agent::EdgeRef> = a.edges
            .into_iter()
            .map(|e| crate::graph::agent::EdgeRef { from: e.from, edge_type: e.edge_type, to: e.to })
            .collect();
        let msg = crate::graph::ops::graph_delete(&idx, &g, a.nodes, refs);
        Ok(CallToolResult::success(vec![Content::text(msg)]))
    }

    #[tool(description = "Edit an existing graph node in place — change its label or type while keeping its id and all its edges (delete-and-recreate would drop the edges). Identify the node by its label. To correct an edge, remove it with graph_delete and add the right one with graph_upsert.")]
    async fn graph_update(&self, Parameters(a): Parameters<GraphUpdateArgs>) -> Result<CallToolResult, McpError> {
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let ups: Vec<crate::graph::agent::NodeUpdate> = a.nodes
            .into_iter()
            .map(|n| crate::graph::agent::NodeUpdate { label: n.label, new_label: n.new_label, new_type: n.new_type })
            .collect();
        let msg = crate::graph::ops::graph_update(&g, ups);
        Ok(CallToolResult::success(vec![Content::text(msg)]))
    }

    #[tool(description = "Recompute the graph's DERIVED layer from what is currently stored: transitive-closure edges, SIMILAR links, communities and centrality (these surface in `glossary`/`neighbors`). Non-destructive — it never deletes or merges nodes. It also REPORTS how many degenerate reasoning chains exist as `prune_candidates` (a node off the reasoning spine) without removing them; actual pruning is a deliberate operator action. Run it after a batch of edits to refresh the derived view.")]
    async fn graph_generalize(&self, Parameters(_): Parameters<Empty>) -> Result<CallToolResult, McpError> {
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let ont = Ontology::load_or_default(&self.root);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        // Non-destructive (shared with the eval enricher → identical output).
        Ok(CallToolResult::success(vec![Content::text(crate::graph::ops::graph_generalize(&g, &ont, now))]))
    }

    #[tool(description = "Find an exact string in the text — a code, version, identifier, parameter name, error message, or exact phrase (e.g. `maxTsdr`, `5.7.2`). ripgrep regex supported; smart-case. Use it whenever the question names a precise token to locate (codes/versions/part numbers beat keyword `search`). For fuzzy/conceptual lookup, use `search`. Returns matching lines as `path:#n: line`; read the full chunk with `read(path, n)`.")]
    async fn grep(&self, Parameters(a): Parameters<GrepArgs>) -> Result<CallToolResult, McpError> {
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let opts = crate::grep::GrepOpts { ignore_case: a.ignore_case.unwrap_or(false), fixed: a.fixed.unwrap_or(false), word: a.word.unwrap_or(false), glob: a.glob, file_type: a.file_type };
        Ok(CallToolResult::success(vec![Content::text(crate::tools::grep(&idx, &a.pattern, &opts, &self.trace))]))
    }

    #[tool(description = "List knowledge-base documents whose path matches a shell glob (e.g. `*.pdf`, `*Safety*`, `*АБАК*`). Returns one `path  (N chunks)` per line — use it to discover what documents exist or find a file by name, then `read(path, n)` or scope a `search`/`grep` to it. N is the document's last chunk number (page/section count).")]
    async fn glob(&self, Parameters(a): Parameters<GlobArgs>) -> Result<CallToolResult, McpError> {
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(crate::tools::glob(&idx, &a.pattern, &self.trace))]))
    }

    #[tool(description = "Delete the index + graph for the knowledge base.")]
    async fn purge(&self, Parameters(_): Parameters<Empty>) -> Result<CallToolResult, McpError> {
        let g = self.root.join(".glossa");
        if g.exists() {
            std::fs::remove_dir_all(&g).map_err(|e| internal(e.into()))?;
        }
        Ok(CallToolResult::success(vec![Content::text("purged .glossa")]))
    }
}

#[tool_handler]
impl ServerHandler for GlossaServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::new(ServerCapabilities::builder().enable_tools().build());
        info.protocol_version = ProtocolVersion::V_2025_06_18;
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
        let json = r#"{"nodes":[{"node_type":"Document","label":"a","source_path":"a.md"}],"edges":[]}"#;
        let a: GraphUpsertArgs = serde_json::from_str(json).unwrap();
        assert_eq!(a.nodes.len(), 1);
        assert_eq!(a.nodes[0].label, "a");
        assert!(a.edges.is_empty());
    }

    #[tokio::test]
    async fn read_by_number_returns_body_and_footer() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("d.md"), b"# A\nalpha\n# B\nbravo\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let srv = GlossaServer::new(dir.path().to_path_buf(), Profile::Editor, false, false);
        let path = dir.path().join("d.md").to_string_lossy().to_string();

        let out = srv.read(Parameters(ReadArgs { path, n: 1, include_images: Some(false) })).await.unwrap();
        let text = format!("{:?}", out);
        assert!(text.contains("alpha"), "body present: {text}");
        assert!(text.contains("#2") || text.contains("next"), "footer offers next: {text}");
    }

    #[tokio::test]
    async fn grep_tool_finds_literal_across_chunks() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("d.md"), b"# A\nmaxTsdr 3000\n# B\nother\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let srv = GlossaServer::new(dir.path().to_path_buf(), Profile::Editor, false, false);
        let out = srv.grep(Parameters(GrepArgs {
            pattern: "maxTsdr".into(), ignore_case: None, fixed: None, word: None, glob: None, file_type: None,
        })).await.unwrap();
        assert!(format!("{:?}", out).contains("maxTsdr"));
        assert!(format!("{:?}", out).contains(":#")); // carries the #n read key
    }

    #[tokio::test]
    async fn glob_tool_lists_documents() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("АБАК.md"), "# A\nраз\n# B\nдва\n".as_bytes()).unwrap();
        std::fs::write(dir.path().join("Other.md"), "# A\nраз\n".as_bytes()).unwrap();
        index_dir(dir.path(), true).unwrap();
        let srv = GlossaServer::new(dir.path().to_path_buf(), Profile::Editor, false, false);
        let out = format!("{:?}", srv.glob(Parameters(GlobArgs { pattern: "*АБАК*".into() })).await.unwrap());
        assert!(out.contains("АБАК"), "lists the matching doc: {out}");
        assert!(!out.contains("Other"), "excludes non-matching: {out}");
    }

    #[test]
    fn profile_gates_tool_visibility() {
        let root = std::path::PathBuf::from(".");
        let reader = GlossaServer::new(root.clone(), Profile::Reader, false, false).enabled_tools();
        assert!(reader.contains(&"search".to_string()) && reader.contains(&"read".to_string()));
        assert!(!reader.contains(&"index".to_string()) && !reader.contains(&"graph_upsert".to_string()) && !reader.contains(&"purge".to_string()));

        let editor = GlossaServer::new(root.clone(), Profile::Editor, false, false).enabled_tools();
        assert!(editor.contains(&"index".to_string()) && editor.contains(&"resolve".to_string()));
        assert!(editor.contains(&"graph_generalize".to_string()), "editor exposes the non-destructive generalize tool");
        assert!(!editor.contains(&"purge".to_string()));
        assert!(!reader.contains(&"graph_generalize".to_string()), "reader cannot generalize");

        let full = GlossaServer::new(root.clone(), Profile::Full, false, false).enabled_tools();
        assert!(full.contains(&"purge".to_string()));

        // resolve is a universally available tool — present in EVERY profile (not gated).
        for prof in [&reader, &editor, &full] {
            assert!(prof.contains(&"resolve".to_string()), "resolve must be in every profile");
        }

        let ng = GlossaServer::new(root, Profile::Editor, false, true).enabled_tools();
        assert!(ng.contains(&"search".to_string()) && ng.contains(&"read".to_string()));
        assert!(!ng.contains(&"neighbors".to_string()) && !ng.contains(&"graph_upsert".to_string()) && !ng.contains(&"index".to_string()));
    }
}
