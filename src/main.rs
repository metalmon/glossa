use clap::{Parser, Subcommand};
use glossa::query::{compile, QueryOpts};
use glossa::search::search_chunks;
use glossa::walk::collect_chunks;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "kb", about = "File-First knowledge-base search (ripgrep syntax)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Search files. PATTERN uses ripgrep syntax.
    Search {
        /// Search pattern (ripgrep regex syntax).
        pattern: String,
        /// Directory to search.
        path: Option<PathBuf>,
        /// Case-insensitive (rg -i).
        #[arg(short = 'i', long = "ignore-case")]
        ignore_case: bool,
        /// Match whole words (rg -w).
        #[arg(short = 'w', long = "word-regexp")]
        word: bool,
        /// Treat pattern as a literal string (rg -F).
        #[arg(short = 'F', long = "fixed-strings")]
        fixed: bool,
        /// Only search paths matching GLOB (rg -g).
        #[arg(short = 'g', long = "glob")]
        glob: Option<String>,
        /// Max number of hits.
        #[arg(long, default_value_t = 100)]
        limit: usize,
        /// Use the on-disk index for BM25-ranked, stemmed search (run `kb index` first).
        #[arg(long)]
        rank: bool,
        /// Disable .gitignore/.ignore/hidden filtering (index everything).
        #[arg(long = "no-ignore")]
        no_ignore: bool,
    },
    /// Build or update the on-disk index for ranked search.
    Index {
        path: Option<PathBuf>,
    },
    /// Rebuild the index from scratch.
    Reindex {
        path: Option<PathBuf>,
    },
    /// Inspect the knowledge graph.
    Graph {
        #[command(subcommand)]
        action: GraphAction,
    },
    /// Run the MCP server over stdio (for AI agents).
    Mcp {
        path: Option<PathBuf>,
        /// Tool profile: reader | editor | full.
        #[arg(long, default_value = "editor")]
        profile: String,
    },
}

#[derive(Subcommand)]
enum GraphAction {
    /// Print node/edge counts.
    Stats {
        path: Option<PathBuf>,
    },
    /// Print nodes reachable from NODE_ID.
    #[command(visible_alias = "neighbors")]
    Near {
        node_id: String,
        path: Option<PathBuf>,
        #[arg(long, default_value_t = 1)]
        depth: usize,
        #[arg(long = "type")]
        types: Vec<String>,
    },
    /// Show a node: type, label, provenance, and its outgoing edges.
    Node {
        node_id: String,
        path: Option<PathBuf>,
    },
    /// Show a path between two node ids (bounded).
    Path {
        from: String,
        to: String,
        path: Option<PathBuf>,
        #[arg(long, default_value_t = 6)]
        max_depth: usize,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Search { pattern, path, ignore_case, word, fixed, glob, limit, rank, no_ignore } => {
            let path = glossa::root::resolve_root(path);
            if rank {
                let idx = glossa::index::store::DocIndex::open_or_create(&path)?;
                for h in idx.search(&pattern, limit)? {
                    println!("{}:{}: {}  [{:.3}]", h.path, h.location, h.snippet, h.score);
                }
                return Ok(());
            }
            let opts = QueryOpts {
                ignore_case,
                smart_case: !ignore_case, // rg smart-case default
                word,
                fixed,
            };
            let re = compile(&pattern, &opts)?;
            let chunks = collect_chunks(&path, glob.as_deref(), !no_ignore)?;
            for h in search_chunks(&chunks, &re, limit) {
                println!("{}:{}:{}: {}", h.doc_path.display(), h.location, h.line, h.snippet);
            }
            Ok(())
        }
        Cmd::Index { path } => {
            let path = glossa::root::resolve_root(path);
            let stats = glossa::index::store::index_dir(&path, false)?;
            println!(
                "indexed: {} added, {} removed, {} unchanged",
                stats.added, stats.removed, stats.unchanged
            );
            Ok(())
        }
        Cmd::Reindex { path } => {
            let path = glossa::root::resolve_root(path);
            let stats = glossa::index::store::index_dir(&path, true)?;
            println!("reindexed: {} files", stats.added);
            Ok(())
        }
        Cmd::Mcp { path, profile } => {
            let path = glossa::root::resolve_root(path);
            use rmcp::{transport::stdio, ServiceExt};
            let server = glossa::mcp::GlossaServer::new(path, glossa::mcp::Profile::parse(&profile));
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async move {
                let service = server.serve(stdio()).await?;
                let _ = service.waiting().await;
                Ok::<(), anyhow::Error>(())
            })?;
            Ok(())
        }
        Cmd::Graph { action } => match action {
            GraphAction::Stats { path } => {
                let path = glossa::root::resolve_root(path);
                let g = glossa::graph::store::GraphStore::open(&path)?;
                println!("nodes: {}, edges: {}", g.node_count()?, g.edge_count()?);
                Ok(())
            }
            GraphAction::Near { node_id, path, depth, types } => {
                let path = glossa::root::resolve_root(path);
                let g = glossa::graph::store::GraphStore::open(&path)?;
                let filter = if types.is_empty() { None } else { Some(types.as_slice()) };
                for id in glossa::graph::traverse::neighbors(&g, &node_id, filter, depth)? {
                    println!("{id}");
                }
                Ok(())
            }
            GraphAction::Node { node_id, path } => {
                let path = glossa::root::resolve_root(path);
                let g = glossa::graph::store::GraphStore::open(&path)?;
                match g.get_node(&node_id)? {
                    Some(n) => {
                        let edges = g.outgoing(&node_id)?;
                        print!("{}", glossa::cli_fmt::render_node(&n, &edges));
                    }
                    None => println!("node not found: {node_id}"),
                }
                Ok(())
            }
            GraphAction::Path { from, to, path, max_depth } => {
                let path = glossa::root::resolve_root(path);
                let g = glossa::graph::store::GraphStore::open(&path)?;
                let found = glossa::graph::traverse::path(&g, &from, &to, max_depth)?;
                println!("{}", glossa::cli_fmt::render_path(found.as_ref(), &from, &to, max_depth));
                Ok(())
            }
        },
    }
}
