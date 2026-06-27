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
        /// Only this file type, e.g. pdf (-t).
        #[arg(short = 't', long = "type")]
        file_type: Option<String>,
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
    /// Exact/regex (ripgrep-style) search over the extracted text.
    Grep {
        /// regex or literal pattern
        pattern: String,
        /// knowledge-base directory
        path: std::path::PathBuf,
        #[arg(short = 'i', long, help = "case-insensitive matching (-i)")] ignore_case: bool,
        #[arg(short = 'F', long)] fixed: bool,
        #[arg(short = 'w', long)] word: bool,
        #[arg(short = 'g', long)] glob: Option<String>,
        #[arg(short = 't', long = "type")] file_type: Option<String>,
    },
    /// List documents whose path matches a shell glob.
    Glob {
        /// glob pattern, e.g. *.pdf or *Safety*
        pattern: String,
        /// knowledge-base directory
        path: std::path::PathBuf,
    },
    /// Regenerate TensorZero tool config from MCP definitions (one source of truth).
    DumpTzTools {
        /// Directory containing tensorzero.toml and tools/.
        #[arg(long, default_value = "eval/tensorzero/config")]
        config_dir: PathBuf,
    },
    /// Run the MCP server over stdio (for AI agents).
    Mcp {
        path: Option<PathBuf>,
        /// Tool profile: reader | editor | full.
        #[arg(long, default_value = "editor")]
        profile: String,
        /// Log every tool call to <root>/.glossa/traces/*.jsonl (for the eval harness).
        #[arg(long)]
        trace: bool,
        /// Expose only search + read (graph/index/admin tools hidden) — eval control arm.
        #[arg(long = "no-graph")]
        no_graph: bool,
    },
}

#[derive(Subcommand)]
enum GraphAction {
    /// Print node/edge counts.
    Stats {
        path: Option<PathBuf>,
    },
    /// Run the deterministic generalization pass: transitive closure, SIMILAR links, communities
    /// and centrality (written as derived `auto-generalized` edges + `node_meta`). With `--merge`,
    /// also COLLAPSE near-duplicate nodes (mutates/deletes agent nodes); without it, report only.
    Generalize {
        path: Option<PathBuf>,
        #[arg(long, help = "also collapse near-duplicate nodes (destructive)")]
        merge: bool,
        #[arg(long, help = "delete degenerate reasoning chains off the ontology spine (destructive)")]
        prune_incomplete: bool,
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
    /// Dump all nodes (optionally filtered by type) with their outgoing edges.
    Dump {
        path: PathBuf,
        /// only show nodes of this type, e.g. Symptom or Resolution (omit for all)
        #[arg(long = "type")]
        node_type: Option<String>,
        /// output format: text (default), json, dot, graphml
        #[arg(long, default_value = "text")]
        format: String,
    },
    /// Import a graph file (JSON), replacing the semantic layer (file = source of truth).
    Import {
        file: PathBuf,
        path: PathBuf,
        #[arg(long)]
        format: Option<String>,
    },
    /// Delete all nodes of the given type (and edges touching them) — clean-slate a semantic layer.
    Prune {
        path: PathBuf,
        /// node type to delete, e.g. Symptom (repeatable)
        #[arg(long = "type", required = true)]
        node_type: Vec<String>,
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
        Cmd::Search { pattern, path, ignore_case, word, fixed, glob, file_type, limit, rank, no_ignore, format } => {
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
                for h in idx.search_filtered(&pattern, limit, glob.as_deref(), file_type.as_deref())? {
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
            // Don't clobber the previous search when this one returns no hits.
            if !records.is_empty() {
                let _ = glossa::cli_fmt::write_last_search(&path, &records);
            }

            if pretty {
                // For --rank, print worst→best so the most relevant sits next to the prompt.
                print!("{}", glossa::cli_fmt::render_search_pretty(&display, rank, &pattern));
            } else {
                for l in &rg_lines {
                    println!("{l}");
                }
            }
            Ok(())
        }
        Cmd::Read { target, location } => {
            // Precedence: existing path beats result-number beats fallback path open.
            // A real file named "3" should be opened directly, not treated as result #3.
            if std::path::Path::new(&target).exists() {
                // 1. Target is an existing path — open it directly.
                print_read(std::path::Path::new(&target), location.as_deref())?;
            } else if let Ok(n) = target.parse::<usize>() {
                // 2. Target is a number and no file by that name exists — resolve from last search.
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
            } else {
                // 3. Non-numeric, non-existing path — attempt open (will surface not-found error).
                print_read(std::path::Path::new(&target), location.as_deref())?;
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
            // Auto-run the generalization pass over the freshly rebuilt graph so derived edges
            // (closure + SIMILAR), communities and centrality stay in sync. Non-destructive:
            // merges are only reported, never applied here (use `kb graph generalize --merge`).
            let g = glossa::graph::store::GraphStore::open(&path)?;
            let ont = glossa::graph::ontology::Ontology::load_or_default(&path);
            let opts = glossa::graph::generalize::apply::Opts::from_ontology(&ont, glossa::trace::now_ms());
            let r = glossa::graph::generalize::apply::generalize(&g, &opts)?;
            println!(
                "generalized: inferred_edges={} similar_edges={} communities={} merge_candidates={}",
                r.inferred_edges, r.similar_edges, r.communities, r.merge_candidates
            );
            Ok(())
        }
        Cmd::Grep { pattern, path, ignore_case, fixed, word, glob, file_type } => {
            let idx = glossa::index::store::DocIndex::open_or_create(&path)?;
            let opts = glossa::grep::GrepOpts { ignore_case, fixed, word, glob, file_type };
            for h in glossa::grep::grep(&idx, &pattern, &opts)? {
                println!("{}", h.display_line());
            }
            Ok(())
        }
        Cmd::Glob { pattern, path } => {
            let idx = glossa::index::store::DocIndex::open_or_create(&path)?;
            for (p, n) in glossa::glob::glob_docs(&idx, &pattern)? {
                println!("{p}  ({n} chunks)");
            }
            Ok(())
        }
        Cmd::DumpTzTools { config_dir } => {
            let n = glossa::tz_export::dump(&config_dir)?;
            println!("dump-tz-tools: wrote {} tool schemas and updated tensorzero.toml", n);
            Ok(())
        }
        Cmd::Mcp { path, profile, trace, no_graph } => {
            let path = glossa::root::resolve_root(path);
            use rmcp::{transport::stdio, ServiceExt};
            let server = glossa::mcp::GlossaServer::new(path, glossa::mcp::Profile::parse(&profile), trace, no_graph);
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
            GraphAction::Generalize { path, merge, prune_incomplete } => {
                let path = glossa::root::resolve_root(path);
                let g = glossa::graph::store::GraphStore::open(&path)?;
                let ont = glossa::graph::ontology::Ontology::load_or_default(&path);
                let mut opts =
                    glossa::graph::generalize::apply::Opts::from_ontology(&ont, glossa::trace::now_ms());
                opts.apply_merges = merge;
                opts.prune_incomplete = prune_incomplete;
                let r = glossa::graph::generalize::apply::generalize(&g, &opts)?;
                println!(
                    "generalize: prune_candidates={} pruned_nodes={} inferred_edges={} \
                     similar_edges={} communities={} merge_candidates={} merged_nodes={}",
                    r.prune_candidates,
                    r.pruned_nodes,
                    r.inferred_edges,
                    r.similar_edges,
                    r.communities,
                    r.merge_candidates,
                    r.merged_nodes
                );
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
            GraphAction::Dump { path, node_type, format } => {
                let g = glossa::graph::store::GraphStore::open(&path)?;
                match format.as_str() {
                    "text" => {
                        let mut nodes = g.all_nodes()?;
                        nodes.sort_by(|a, b| a.node_type.cmp(&b.node_type).then(a.id.cmp(&b.id)));
                        for n in &nodes {
                            if node_type.as_deref().is_some_and(|t| t != n.node_type) {
                                continue;
                            }
                            let al = if n.aliases.is_empty() {
                                String::new()
                            } else {
                                format!("  ({})", n.aliases.join(", "))
                            };
                            println!("[{}] {}  {}{}", n.node_type, n.id, n.label, al);
                            for e in g.outgoing(&n.id)? {
                                println!("    -{}-> {}", e.edge_type, e.to);
                            }
                        }
                    }
                    "json" => {
                        use glossa::graph::io::{collect, to_json};
                        print!("{}", to_json(&collect(&g, node_type.as_deref())?)?);
                    }
                    "dot" => {
                        use glossa::graph::io::{collect, to_dot};
                        print!("{}", to_dot(&collect(&g, node_type.as_deref())?));
                    }
                    "graphml" => {
                        use glossa::graph::io::{collect, to_graphml};
                        print!("{}", to_graphml(&collect(&g, node_type.as_deref())?));
                    }
                    other => anyhow::bail!(
                        "unknown format {:?} — valid formats: text, json, dot, graphml",
                        other
                    ),
                }
                Ok(())
            }
            GraphAction::Import { file, path, format } => {
                let fmt = format.as_deref().map(|s| s.to_string()).unwrap_or_else(|| {
                    file.extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("json")
                        .to_string()
                });
                if fmt != "json" {
                    anyhow::bail!(
                        "import supports json only (graphml/dot are export-only)"
                    );
                }
                let contents = std::fs::read_to_string(&file)?;
                let export = glossa::graph::io::from_json(&contents)?;
                let ont = glossa::graph::ontology::Ontology::load_or_default(&path);
                let now = glossa::trace::now_ms();
                let g = glossa::graph::store::GraphStore::open(&path)?;
                let (pruned, n, ed) = glossa::graph::io::import_replace_layer(&g, &ont, export, now)?;
                println!("graph import: pruned {pruned}, +{n} nodes, +{ed} edges");
                Ok(())
            }
            GraphAction::Prune { path, node_type } => {
                let g = glossa::graph::store::GraphStore::open(&path)?;
                let mut total = 0;
                for t in &node_type {
                    let n = g.delete_by_type(t)?;
                    println!("graph prune: removed {n} entries of type {t}");
                    total += n;
                }
                println!("graph prune: {total} total entries removed");
                Ok(())
            }
        },
    }
}
