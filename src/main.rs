use clap::{Parser, Subcommand};
use glossa::query::{compile, QueryOpts};
use glossa::search::search_chunks;
use glossa::walk::collect_chunks;
use std::path::PathBuf;

#[derive(Clone, Copy, clap::ValueEnum)]
enum OutputFormat {
    /// pretty when stdout is a terminal, rg otherwise
    Auto,
    /// ripgrep-compatible: path:location[:line]: snippet
    Rg,
    /// numbered, aligned log-lines for humans
    Pretty,
}

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
        /// Output style: auto (pretty in a terminal, rg when piped), rg, or pretty.
        #[arg(long, value_enum, default_value = "auto")]
        format: OutputFormat,
    },
    /// Read a document's text. TARGET is a path, or a result number from the last search.
    Read {
        /// A file path, or a number referencing the last search's Nth result.
        target: String,
        /// Optional location (heading / "p.N") to narrow to.
        location: Option<String>,
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

fn print_read(path: &std::path::Path, location: Option<&str>) -> anyhow::Result<()> {
    let text = glossa::read::read_region(path, location)?;
    if glossa::cli_fmt::stdout_is_tty() {
        let head = match location {
            Some(l) => format!("── {} · {} ──", path.display(), l),
            None => format!("── {} ──", path.display()),
        };
        println!("{}", glossa::cli_fmt::dim(&head));
    }
    print!("{text}");
    if !text.ends_with('\n') {
        println!();
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Search { pattern, path, ignore_case, word, fixed, glob, limit, rank, no_ignore, format } => {
            let path = glossa::root::resolve_root(path);
            let pretty = match format {
                OutputFormat::Pretty => true,
                OutputFormat::Rg => false,
                OutputFormat::Auto => glossa::cli_fmt::stdout_is_tty(),
            };
            let mut rg_lines: Vec<String> = Vec::new();
            let mut display: Vec<glossa::cli_fmt::DisplayHit> = Vec::new();
            let mut records: Vec<(String, String)> = Vec::new();

            if rank {
                let idx = glossa::index::store::DocIndex::open_or_create(&path)?;
                for h in idx.search(&pattern, limit)? {
                    rg_lines.push(format!("{}:{}: {}  [{:.3}]", h.path, h.location, h.snippet, h.score));
                    display.push(glossa::cli_fmt::DisplayHit {
                        file: glossa::cli_fmt::rel_file(&path, &h.path),
                        location: h.location.clone(),
                        snippet: h.snippet.clone(),
                        score: Some(h.score),
                    });
                    records.push((h.path.clone(), h.location.clone()));
                }
            } else {
                let opts = QueryOpts { ignore_case, smart_case: !ignore_case, word, fixed };
                let re = compile(&pattern, &opts)?;
                let chunks = collect_chunks(&path, glob.as_deref(), !no_ignore)?;
                for h in search_chunks(&chunks, &re, limit) {
                    let p = h.doc_path.display().to_string();
                    rg_lines.push(format!("{}:{}:{}: {}", p, h.location, h.line, h.snippet));
                    display.push(glossa::cli_fmt::DisplayHit {
                        file: glossa::cli_fmt::rel_file(&path, &p),
                        location: h.location.clone(),
                        snippet: h.snippet.clone(),
                        score: None,
                    });
                    records.push((p, h.location.clone()));
                }
            }

            // Persist for `kb read <#>` (best-effort; ignore IO errors).
            let _ = glossa::cli_fmt::write_last_search(&path, &records);

            if pretty {
                print!("{}", glossa::cli_fmt::render_search_pretty(&display));
            } else {
                for l in &rg_lines {
                    println!("{l}");
                }
            }
            Ok(())
        }
        Cmd::Read { target, location } => {
            // Numeric target → resolve from the last search; otherwise treat as a path.
            if let Ok(n) = target.parse::<usize>() {
                let root = glossa::root::resolve_root(None);
                let rec = glossa::cli_fmt::read_last_search(&root)
                    .and_then(|c| glossa::cli_fmt::nth_record(&c, n));
                match rec {
                    Some((p, loc)) => {
                        let loc_opt = if loc.is_empty() || loc == "(no-text)" {
                            None
                        } else {
                            Some(loc)
                        };
                        print_read(std::path::Path::new(&p), loc_opt.as_deref())?;
                    }
                    None => println!("no result #{n} (run a search first)"),
                }
                return Ok(());
            }
            print_read(std::path::Path::new(&target), location.as_deref())?;
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
