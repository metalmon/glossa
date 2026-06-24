use crate::graph::agent::apply_upsert;
use crate::graph::ontology::Ontology;
use crate::graph::store::GraphStore;
use crate::index::store::index_dir;
use crate::query::{compile, QueryOpts};
use crate::read::{extract_images, read_region};
use crate::search::search_chunks;
use crate::walk::collect_chunks;
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
    #[schemars(description = "ripgrep-syntax query")]
    query: String,
    #[serde(default)]
    #[schemars(description = "max hits (default 50)")]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ReadArgs {
    #[schemars(description = "document path")]
    path: String,
    #[serde(default)]
    #[schemars(description = "optional location (heading/sheet/page) substring")]
    location: Option<String>,
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
struct GraphUpsertArgs {
    #[serde(default)]
    nodes: Vec<crate::graph::agent::NodeSpec>,
    #[serde(default)]
    edges: Vec<crate::graph::agent::EdgeSpec>,
}

#[tool_router]
impl GlossaServer {
    #[tool(description = "Search the knowledge base (ripgrep syntax). Returns path:location:line: snippet.")]
    async fn search(&self, Parameters(a): Parameters<SearchArgs>) -> Result<CallToolResult, McpError> {
        let opts = QueryOpts { smart_case: true, ..Default::default() };
        let re = compile(&a.query, &opts).map_err(|e| McpError::invalid_params(e.to_string(), None))?;
        let chunks = collect_chunks(&self.root, None, true).map_err(internal)?;
        let hits = search_chunks(&chunks, &re, a.limit.unwrap_or(50));
        let trace_hits: Vec<serde_json::Value> = hits.iter().map(|h| serde_json::json!({
            "path": h.doc_path.display().to_string(), "location": h.location, "line": h.line
        })).collect();
        self.trace.log("search", serde_json::json!({"query": a.query}), serde_json::json!(trace_hits));
        let body = hits.iter()
            .map(|h| format!("{}:{}:{}: {}", h.doc_path.display(), h.location, h.line, h.snippet))
            .collect::<Vec<_>>().join("\n");
        Ok(CallToolResult::success(vec![Content::text(body)]))
    }

    #[tool(description = "Read a document (optionally a location), with embedded images for the agent's vision.")]
    async fn read(&self, Parameters(a): Parameters<ReadArgs>) -> Result<CallToolResult, McpError> {
        let path = std::path::PathBuf::from(&a.path);
        let text = read_region(&path, a.location.as_deref()).map_err(internal)?;
        self.trace.log("read", serde_json::json!({"path": a.path, "location": a.location}), serde_json::json!({"path": a.path}));
        let mut content = vec![Content::text(text)];
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
        info.instructions = Some("glossa File-First knowledge-base search. Use ripgrep syntax for `search`.".into());
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
