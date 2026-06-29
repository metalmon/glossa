use clap::{Parser, Subcommand};
use glossa::query::{compile, QueryOpts};
use glossa::search::search_chunks;
use glossa::walk::collect_chunks;
use std::path::PathBuf;

/// Native Windows Service integration (SCM dispatcher + control handler). Windows-only.
#[cfg(windows)]
mod winsvc;

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
    /// Search the knowledge base (BM25-ranked keywords over the index).
    Search {
        /// keywords (or a ripgrep regex with `--scan`)
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
        #[arg(short = 'l', long, default_value_t = 100)]
        limit: usize,
        /// literal ripgrep-regex scan of raw files instead of the BM25 index (slow, not stemmed)
        #[arg(short = 's', long)]
        scan: bool,
        /// Disable .gitignore/.ignore/hidden filtering (index everything).
        #[arg(short = 'u', long = "no-ignore")]
        no_ignore: bool,
        /// Output style: auto (pretty in a terminal, rg when piped), rg, or pretty.
        #[arg(short = 'f', long, value_enum, default_value = "auto")]
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
        /// knowledge-base directory (default: nearest indexed root / current dir)
        path: Option<PathBuf>,
        #[arg(short = 'i', long, help = "case-insensitive matching (-i)")] ignore_case: bool,
        #[arg(short = 'F', long)] fixed: bool,
        #[arg(short = 'w', long)] word: bool,
        #[arg(short = 'g', long)] glob: Option<String>,
        #[arg(short = 't', long = "type")] file_type: Option<String>,
    },
    /// List documents whose PATH matches a shell glob (matches file paths, NOT text inside them —
    /// for content use `search` or `grep`).
    Glob {
        /// glob over document PATHS, e.g. *.pdf or *Safety* (not a content search)
        pattern: String,
        /// knowledge-base directory (default: nearest indexed root / current dir)
        path: Option<PathBuf>,
    },
    /// Run the MCP server (stdio for a local subprocess, or streamable-http for the network), or an
    /// MCP-related subcommand.
    Mcp {
        #[command(subcommand)]
        action: Option<McpAction>,
        path: Option<PathBuf>,
        /// Tool profile: reader | editor | full.
        #[arg(short = 'p', long, default_value = "editor")]
        profile: String,
        /// Log every tool call to <root>/.glossa/traces/*.jsonl (for the eval harness).
        #[arg(short = 't', long)]
        trace: bool,
        /// Expose only search + read (graph/index/admin tools hidden) — eval control arm.
        #[arg(short = 'G', long = "no-graph")]
        no_graph: bool,
        /// Transport: stdio (local subprocess) or streamable-http (network endpoint at <bind>/mcp).
        #[arg(long, value_enum, default_value = "stdio", env = "GLOSSA_MCP_TRANSPORT")]
        transport: McpTransport,
        /// Bind address for --transport streamable-http.
        #[arg(long, default_value = "127.0.0.1:8080", env = "GLOSSA_MCP_BIND")]
        bind: String,
        /// Extra allowed `Host` header value(s) for streamable-http (DNS-rebind guard). Repeatable.
        /// Default permits loopback only — set your gateway/public host(s) for a prod deployment.
        #[arg(long = "allowed-host")]
        allowed_hosts: Vec<String>,
        /// Run under the Windows Service Control Manager (set by the service binPath; not for manual
        /// use). The SCM Stop/Shutdown control triggers the same graceful shutdown as Ctrl-C/SIGTERM.
        #[arg(long = "windows-service", hide = true)]
        windows_service: bool,
        /// SCM service name (Windows only; set by install scripts). Env: `GLOSSA_SERVICE_NAME`.
        #[arg(long = "service-name", hide = true, env = "GLOSSA_SERVICE_NAME")]
        service_name: Option<String>,
    },
}

/// MCP transport for `kb mcp`.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum McpTransport {
    /// Newline-delimited JSON-RPC over stdin/stdout (local subprocess clients).
    Stdio,
    /// MCP Streamable HTTP at `<bind>/mcp` (network clients; put a TLS/auth gateway in front).
    StreamableHttp,
}

#[derive(Subcommand)]
enum McpAction {
    /// Regenerate TensorZero tool config from the live MCP tool definitions (one source of truth).
    DumpTzTools {
        /// Directory containing tensorzero.toml and tools/.
        #[arg(short = 'd', long, default_value = "eval/tensorzero/config")]
        config_dir: PathBuf,
    },
}

#[derive(Subcommand)]
enum GraphAction {
    /// Print node/edge counts.
    Stats {
        path: Option<PathBuf>,
    },
    /// Find graph nodes by concept (the `glossary` tool) — prints `id [type] label` + edges.
    #[command(visible_aliases = ["search", "find"])]
    Glossary {
        /// concept in your own words, e.g. "потеря связи"
        query: String,
        path: Option<PathBuf>,
    },
    /// Browse graph nodes: a per-type count, or `--type T` to list that type.
    Ls {
        path: Option<PathBuf>,
        /// list nodes of this type, e.g. Symptom (omit for a per-type summary)
        #[arg(short = 't', long = "type")]
        node_type: Option<String>,
        #[arg(short = 'l', long, default_value_t = 50)]
        limit: usize,
    },
    /// Run the deterministic generalization pass: transitive closure, SIMILAR links, communities
    /// and centrality (written as derived `auto-generalized` edges + `node_meta`). With `--merge`,
    /// also COLLAPSE near-duplicate nodes (mutates/deletes agent nodes); without it, report only.
    Generalize {
        path: Option<PathBuf>,
        #[arg(short = 'm', long, help = "also collapse near-duplicate nodes (destructive)")]
        merge: bool,
        #[arg(short = 'p', long, help = "delete degenerate reasoning chains off the ontology spine (destructive)")]
        prune_incomplete: bool,
    },
    /// Print nodes reachable from NODE_ID.
    #[command(visible_alias = "neighbors")]
    Near {
        node_id: String,
        path: Option<PathBuf>,
        #[arg(short = 'd', long, default_value_t = 1)]
        depth: usize,
        #[arg(short = 't', long = "type")]
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
        #[arg(short = 'd', long, default_value_t = 6)]
        max_depth: usize,
    },
    /// Dump all nodes (optionally filtered by type) with their outgoing edges.
    Dump {
        path: PathBuf,
        /// only show nodes of this type, e.g. Symptom or Resolution (omit for all)
        #[arg(short = 't', long = "type")]
        node_type: Option<String>,
        /// output format: text (default), json, dot, graphml
        #[arg(short = 'f', long, default_value = "text")]
        format: String,
    },
    /// Import a graph file (JSON), replacing the semantic layer (file = source of truth).
    Import {
        file: PathBuf,
        path: PathBuf,
        #[arg(short = 'f', long)]
        format: Option<String>,
    },
    /// Delete all nodes of the given type (and edges touching them) — clean-slate a semantic layer.
    Prune {
        path: PathBuf,
        /// node type to delete, e.g. Symptom (repeatable)
        #[arg(short = 't', long = "type", required = true)]
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

/// Everything needed to start one MCP serve instance. Built once from the CLI; reused by both the
/// foreground path and the Windows Service path (which stashes it before the SCM dispatcher starts).
#[derive(Clone)]
pub(crate) struct ServeParams {
    pub path: PathBuf,
    pub profile: glossa::mcp::Profile,
    pub trace: bool,
    pub no_graph: bool,
    pub transport: McpTransport,
    pub bind: String,
    pub allowed_hosts: Vec<String>,
    pub service_name: Option<String>,
}

/// Run one MCP serve instance to completion. `cancel` drives graceful shutdown; when `handle_signals`
/// is set we install the OS-signal → cancel bridge (foreground path). The Windows Service path passes
/// `handle_signals = false` and cancels via the SCM control handler instead.
pub(crate) fn run_serve(
    p: ServeParams,
    cancel: tokio_util::sync::CancellationToken,
    handle_signals: bool,
    on_transport_ready: Option<Box<dyn FnOnce() + Send>>,
) -> anyhow::Result<()> {
    let server = glossa::mcp::GlossaServer::new(p.path, p.profile, p.trace, p.no_graph);
    // Freshness runs on EVERY instance (readers stay current). The heavy generalize loop runs ONLY on
    // the indexer (editor/full); among multiple editors it is further serialized by generalize.lock.
    let run_maintenance = p.profile != glossa::mcp::Profile::Reader;
    let transport = p.transport;
    let bind = p.bind;
    let allowed_hosts = p.allowed_hosts;
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        if handle_signals {
            let c = cancel.clone();
            tokio::spawn(async move {
                shutdown_signal().await;
                tracing::info!("shutdown signal received — draining");
                c.cancel();
            });
        }
        if run_maintenance {
            tokio::spawn(server.clone().maintenance_loop(cancel.clone()));
        }
        match transport {
            McpTransport::Stdio => {
                use rmcp::{transport::stdio, ServiceExt};
                let freshen_srv = server.clone();
                let service = server.serve(stdio()).await?;
                if let Some(f) = on_transport_ready {
                    f();
                }
                freshen_srv.kick_freshen();
                let _ = service.waiting().await; // client-driven: exit on stdin EOF
                cancel.cancel();
            }
            McpTransport::StreamableHttp => {
                serve_streamable_http(server, &bind, allowed_hosts, cancel, on_transport_ready).await?;
            }
        }
        Ok::<(), anyhow::Error>(())
    })?;
    Ok(())
}

/// Wait for an OS shutdown signal: Ctrl-C on every platform, plus SIGTERM on unix (what
/// `systemctl stop` / container runtimes send). Returns when either fires.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = term => {}
    }
}

/// Serve the MCP server over Streamable HTTP at `<bind>/mcp` (one shared `GlossaServer` across all
/// sessions). DNS-rebind protection allows loopback by default; pass `--allowed-host` for a gateway/
/// public host. TLS + auth are expected to be terminated by a reverse proxy in front. Ctrl-C
/// triggers a graceful shutdown: active sessions are terminated, the listener drains, and `cancel`
/// (shared with the maintenance loop) fires so the whole server stops together.
async fn serve_streamable_http(
    server: glossa::mcp::GlossaServer,
    bind: &str,
    allowed_hosts: Vec<String>,
    cancel: tokio_util::sync::CancellationToken,
    on_transport_ready: Option<Box<dyn FnOnce() + Send>>,
) -> anyhow::Result<()> {
    use rmcp::transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    };
    // Shutdown is driven by `cancel` (the caller wires the OS signal or the SCM control handler to it).
    let mut config = StreamableHttpServerConfig::default();
    config.cancellation_token = cancel.clone();
    if !allowed_hosts.is_empty() {
        config = config.with_allowed_hosts(allowed_hosts);
    }
    let ready_srv = server.clone();
    let metrics_srv = server.clone();
    let freshen_srv = server.clone();
    let service = StreamableHttpService::new(
        move || Ok(server.clone()),
        std::sync::Arc::new(LocalSessionManager::default()),
        config,
    );
    let app = axum::Router::new()
        // Liveness: the process is up.
        .route("/health", axum::routing::get(|| async { "ok" }))
        // Readiness: the index + graph are openable (the server can actually serve).
        .route(
            "/ready",
            axum::routing::get(move || {
                let s = ready_srv.clone();
                async move {
                    if s.readiness() {
                        (axum::http::StatusCode::OK, "ready")
                    } else {
                        (axum::http::StatusCode::SERVICE_UNAVAILABLE, "not ready")
                    }
                }
            }),
        )
        // Prometheus metrics (index/graph size, derived-layer staleness).
        .route(
            "/metrics",
            axum::routing::get(move || {
                let s = metrics_srv.clone();
                async move { s.metrics_text() }
            }),
        )
        .nest_service("/mcp", service)
        .layer(tower_http::trace::TraceLayer::new_for_http());
    let listener = tokio::net::TcpListener::bind(bind).await?;
    tracing::info!("glossa MCP (streamable-http) on http://{bind}/mcp  (+ /health /ready /metrics)");
    if let Some(f) = on_transport_ready {
        f();
    }
    freshen_srv.kick_freshen();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move { cancel.cancelled().await })
        .await?;
    tracing::info!("glossa MCP (streamable-http) stopped");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    // Structured logs go to STDERR — stdout is the stdio JSON-RPC channel and must never carry logs.
    // Level via RUST_LOG (default `info`). Best-effort init (a second init in tests is a no-op).
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                // Our logs at info; silence tantivy's per-reload INFO chatter. RUST_LOG overrides.
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,tantivy=warn")),
        )
        .with_writer(std::io::stderr)
        .try_init();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Search { pattern, path, ignore_case, word, fixed, glob, file_type, limit, scan, no_ignore, format } => {
            let path = glossa::root::resolve_root(path);
            let pretty = match format {
                OutputFormat::Pretty => true,
                OutputFormat::Rg => false,
                OutputFormat::Auto => glossa::cli_fmt::stdout_is_tty(),
            };
            let mut rg_lines: Vec<String> = Vec::new();
            let mut display: Vec<glossa::cli_fmt::DisplayHit> = Vec::new();
            let mut records: Vec<(String, String)> = Vec::new();

            if !scan {
                glossa::index::store::ensure_fresh(&path)?; // file-first: pick up new/changed docs
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
                // Index (default) results are ranked → print worst→best so the most relevant sits
                // next to the prompt; the literal `--scan` results are kept in file order.
                print!("{}", glossa::cli_fmt::render_search_pretty(&display, !scan, &pattern));
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
                            Some(loc.clone())
                        };
                        // The stored path is the INDEX key — it carries the corpus-root prefix from
                        // index time, so it does NOT resolve as a filesystem path from an arbitrary
                        // cwd (e.g. running `kb read 1` from inside the corpus dir → os error 3).
                        // Read the chunk straight from the index (cwd-independent, like MCP `read`);
                        // fall back to opening the file only when the chunk isn't indexed.
                        let from_index = loc_opt.as_deref().and_then(|l| {
                            glossa::index::store::DocIndex::open_or_create(&root)
                                .ok()
                                .and_then(|idx| idx.read_chunk(&p, l).ok().flatten())
                        });
                        match from_index {
                            Some(body) => {
                                if glossa::cli_fmt::stdout_is_tty() {
                                    println!("{}", glossa::cli_fmt::dim(&format!("── {p} · {loc} ──")));
                                }
                                print!("{body}");
                                if !body.ends_with('\n') {
                                    println!();
                                }
                            }
                            None => print_read(std::path::Path::new(&p), loc_opt.as_deref())?,
                        }
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
            let path = glossa::root::resolve_root(path);
            glossa::index::store::ensure_fresh(&path)?; // file-first: pick up new/changed docs
            let idx = glossa::index::store::DocIndex::open_or_create(&path)?;
            let opts = glossa::grep::GrepOpts { ignore_case, fixed, word, glob, file_type };
            for h in glossa::grep::grep(&idx, &pattern, &opts)? {
                println!("{}", h.display_line());
            }
            Ok(())
        }
        Cmd::Glob { pattern, path } => {
            let path = glossa::root::resolve_root(path);
            glossa::index::store::ensure_fresh(&path)?; // file-first: pick up new/changed docs
            let idx = glossa::index::store::DocIndex::open_or_create(&path)?;
            let docs = glossa::glob::glob_docs(&idx, &pattern)?;
            if docs.is_empty() {
                println!("(no documents match — ripgrep -g glob syntax: use * or **/* or *.{{pdf,md}}; matches PATHS not content; use `kb grep` or `kb search` for text)");
            } else {
                for (p, n) in docs {
                    println!("{p}  ({n} chunks)");
                }
            }
            Ok(())
        }
        Cmd::Mcp { action, path, profile, trace, no_graph, transport, bind, allowed_hosts, windows_service, service_name } => match action {
            Some(McpAction::DumpTzTools { config_dir }) => {
                let n = glossa::tz_export::dump(&config_dir)?;
                println!("dump-tz-tools: wrote {} tool schemas and updated tensorzero.toml", n);
                Ok(())
            }
            None => {
                let path = glossa::root::resolve_root(path);
                let params = ServeParams {
                    path,
                    profile: glossa::mcp::Profile::parse(&profile),
                    trace,
                    no_graph,
                    transport,
                    bind,
                    allowed_hosts,
                    service_name,
                };
                if windows_service {
                    // Launched by the SCM (binPath carries --windows-service): hand off to the
                    // service dispatcher, which runs run_serve under SCM control (Stop → cancel).
                    #[cfg(windows)]
                    {
                        return winsvc::run(params);
                    }
                    #[cfg(not(windows))]
                    {
                        anyhow::bail!("--windows-service is only supported on Windows");
                    }
                }
                // Foreground: OS signals (Ctrl-C / SIGTERM) drive graceful shutdown.
                run_serve(params, tokio_util::sync::CancellationToken::new(), true, None)?;
                Ok(())
            }
        },
        Cmd::Graph { action } => match action {
            GraphAction::Stats { path } => {
                let path = glossa::root::resolve_root(path);
                let g = glossa::graph::store::GraphStore::open(&path)?;
                println!("{}", glossa::tools::graph_stats(&g));
                Ok(())
            }
            GraphAction::Glossary { query, path } => {
                let path = glossa::root::resolve_root(path);
                glossa::index::store::ensure_fresh(&path)?; // file-first: pick up new/changed docs
                let idx = glossa::index::store::DocIndex::open_or_create(&path)?;
                let g = glossa::graph::store::GraphStore::open(&path)?;
                let trace = glossa::trace::TraceLog::disabled();
                let spec = glossa::tools::ChainSpec::from_ontology(&glossa::graph::ontology::Ontology::load_or_default(&path));
                println!("{}", glossa::tools::glossary(&idx, &g, &query, &spec, &trace));
                Ok(())
            }
            GraphAction::Ls { path, node_type, limit } => {
                let path = glossa::root::resolve_root(path);
                let g = glossa::graph::store::GraphStore::open(&path)?;
                let nodes = g.all_nodes()?;
                match node_type {
                    None => {
                        // per-type summary — the browse overview
                        let mut counts: std::collections::BTreeMap<String, usize> =
                            std::collections::BTreeMap::new();
                        for n in &nodes {
                            *counts.entry(n.node_type.clone()).or_default() += 1;
                        }
                        for (t, c) in &counts {
                            println!("{t}: {c}");
                        }
                        println!("\n(use --type <T> to list nodes, or `kb graph search <query>`)");
                    }
                    Some(t) => {
                        let matched: Vec<_> = nodes.iter().filter(|n| n.node_type == t).collect();
                        for n in matched.iter().take(limit) {
                            println!("{}  [{}]  {}", n.id, n.node_type, n.label);
                        }
                        if matched.len() > limit {
                            println!("… {} more (--limit to show more)", matched.len() - limit);
                        }
                    }
                }
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
