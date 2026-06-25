use crate::graph::agent::apply_upsert;
use crate::graph::ontology::Ontology;
use crate::graph::store::GraphStore;
use crate::index::store::index_dir;
use crate::read::extract_images;
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

const EDITOR_TOOLS: &[&str] = &["index", "reindex", "graph_upsert", "resolve"];
const FULL_TOOLS: &[&str] = &["purge"];
const GRAPH_TOOLS: &[&str] = &["glossary", "neighbors", "graph_upsert", "resolve", "index", "reindex", "purge"];

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
    node_id: String,
    #[serde(default)]
    depth: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct NameArg { name: String }

#[derive(Debug, Deserialize, JsonSchema)]
struct Empty {}

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
    nodes: Vec<crate::graph::agent::NodeSpec>,
    #[serde(default)]
    edges: Vec<crate::graph::agent::EdgeSpec>,
}

#[tool_router]
impl GlossaServer {
    #[tool(description = "Full-text search over the knowledge base. Pass natural-language keywords (Russian or English; morphology-aware, BM25-ranked) — NOT a regex. Returns numbered hits in the form `[#n] path · label · snippet  [score]`; use `read(path, n)` to fetch the full text of chunk number `n`. If results are empty, run `index` on the base first. Scope with optional glob/file_type filters.")]
    async fn search(&self, Parameters(a): Parameters<SearchArgs>) -> Result<CallToolResult, McpError> {
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let (body, _hits) = crate::tools::search(&idx, &a.query, a.limit.unwrap_or(50), a.glob.as_deref(), a.file_type.as_deref(), &self.trace);
        Ok(CallToolResult::success(vec![Content::text(body)]))
    }

    #[tool(description = "Read a document chunk by its number `n` (the `[#n]` shown in search results; for PDFs this is the page number). Returns the chunk text plus the numbers of the previous/next chunks for context expansion.")]
    async fn read(&self, Parameters(a): Parameters<ReadArgs>) -> Result<CallToolResult, McpError> {
        let idx = crate::index::store::DocIndex::open_or_create(&self.root).map_err(internal)?;
        let path = std::path::PathBuf::from(&a.path);
        let chunk = idx.read_chunk_by_ord(&a.path, a.n as u64).map_err(internal)?
            .ok_or_else(|| McpError::invalid_params(format!("no chunk #{} in {}", a.n, a.path), None))?;
        self.trace.log("read", serde_json::json!({"path": a.path, "n": a.n}), serde_json::json!({"path": a.path}));
        let footer = match (chunk.prev, chunk.next) {
            (Some(p), Some(n)) => format!("\n\n‹ prev #{p} · next #{n} ›"),
            (None, Some(n)) => format!("\n\n‹ start of document · next #{n} ›"),
            (Some(p), None) => format!("\n\n‹ prev #{p} · end of document ›"),
            (None, None) => String::new(),
        };
        let mut content = vec![Content::text(format!("{}{}", chunk.body, footer))];
        if a.include_images.unwrap_or(true) {
            for img in extract_images(&path, 8).map_err(internal)? {
                let b64 = base64::engine::general_purpose::STANDARD.encode(&img.bytes);
                content.push(Content::image(b64, img.mime));
            }
        }
        Ok(CallToolResult::success(content))
    }

    #[tool(description = "List glossary node ids whose label/alias matches a name.")]
    async fn glossary(&self, Parameters(a): Parameters<NameArg>) -> Result<CallToolResult, McpError> {
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let ids = g.resolve(&a.name).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(ids.join("\n"))]))
    }

    #[tool(description = "Graph neighbors reachable from a node id.")]
    async fn neighbors(&self, Parameters(a): Parameters<NeighborsArgs>) -> Result<CallToolResult, McpError> {
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let ids = crate::graph::traverse::neighbors(&g, &a.node_id, None, a.depth.unwrap_or(1)).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(ids.join("\n"))]))
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

    #[tool(description = "Upsert agent-built graph nodes/edges (validated against ontology.toml). Each node/edge needs id/type/label and a source_path for provenance.")]
    async fn graph_upsert(&self, Parameters(a): Parameters<GraphUpsertArgs>) -> Result<CallToolResult, McpError> {
        let g = GraphStore::open(&self.root).map_err(internal)?;
        let ont = Ontology::load_or_default(&self.root);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let (n, e) = apply_upsert(&g, &ont, a.nodes, a.edges, now).map_err(internal)?;
        Ok(CallToolResult::success(vec![Content::text(format!("upserted {n} nodes, {e} edges"))]))
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
    fn graph_upsert_args_deserialize_from_json() {
        let json = r#"{"nodes":[{"id":"a","node_type":"Document","label":"a","source_path":"a.md"}],"edges":[]}"#;
        let a: GraphUpsertArgs = serde_json::from_str(json).unwrap();
        assert_eq!(a.nodes.len(), 1);
        assert_eq!(a.nodes[0].id, "a");
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
        assert!(!editor.contains(&"purge".to_string()));

        let full = GlossaServer::new(root.clone(), Profile::Full, false, false).enabled_tools();
        assert!(full.contains(&"purge".to_string()));

        let ng = GlossaServer::new(root, Profile::Editor, false, true).enabled_tools();
        assert!(ng.contains(&"search".to_string()) && ng.contains(&"read".to_string()));
        assert!(!ng.contains(&"neighbors".to_string()) && !ng.contains(&"graph_upsert".to_string()) && !ng.contains(&"index".to_string()));
    }
}
