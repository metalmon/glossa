use clap::{Parser, Subcommand};
use glossa::query::{compile, QueryOpts};
use glossa::search::search_chunks;
use glossa::walk::collect_chunks;
use std::path::PathBuf;

/// Native Windows Service integration (SCM dispatcher + control handler). Windows-only.
#[cfg(windows)]
mod winsvc;

/// Resolve the KB root and report it — plus any ambiguity warnings — so the operator can SEE which
/// `.glossa` a command actually used. The nested-corpus and deleted-`.glossa`-walks-up traps are
/// otherwise silent (a deleted corpus `.glossa` makes `kb index` recreate the index in an ANCESTOR,
/// splitting CLI and MCP apart). `via_tracing` picks the channel: interactive CLI commands print
/// plain lines to stderr; the long-lived server routes them through `tracing` so they match its
/// other logs (and become JSON under `GLOSSA_LOG_FORMAT=json`).
fn resolve_root_reported(explicit: Option<PathBuf>, via_tracing: bool) -> PathBuf {
    let r = glossa::root::resolve_root_verbose(explicit);
    let shown = std::path::absolute(&r.root).unwrap_or_else(|_| r.root.clone());
    if via_tracing {
        tracing::info!(root = %shown.display(), "resolved kb root");
        for a in r.advisories() {
            tracing::warn!("{a}");
        }
    } else {
        eprintln!("root: {}", shown.display());
        for a in r.advisories() {
            eprintln!("warning: {a}");
        }
    }
    r.root
}

/// Root resolution for interactive CLI commands — reports to stderr as plain lines.
fn resolve_root_logged(explicit: Option<PathBuf>) -> PathBuf {
    resolve_root_reported(explicit, false)
}

/// Root resolution for the long-lived MCP server — reports through `tracing` (structured, JSON-able).
fn resolve_root_traced(explicit: Option<PathBuf>) -> PathBuf {
    resolve_root_reported(explicit, true)
}

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
#[command(
    name = "kb",
    version,
    about = "File-First knowledge-base search (ripgrep syntax)"
)]
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
        /// Restrict results to one document or path-glob (e.g. `manual.pdf` or `guides/**`);
        /// ANDed with --glob when both are set.
        #[arg(long)]
        scope: Option<String>,
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
    /// Print a file's full extracted text — a `cat` that understands Office and PDF. Reads the file
    /// directly: no index, no `.glossa`. Pipe it to your agent or grep it.
    Cat {
        /// Path to a document file (.pdf, .docx, .xlsx, .pptx, .md, …).
        target: PathBuf,
    },
    /// Update the index. No flags: incremental over the whole corpus. --force: full rebuild.
    /// --file <rel>: reindex just that one document (picks up an in-place edit).
    Index {
        path: Option<PathBuf>,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        file: Option<String>,
        /// Materialize a baked ontology preset before indexing (see `kb ontology list`).
        #[arg(long)]
        ontology: Option<String>,
    },
    /// Delete notebook notes whose owner document no longer exists in the corpus.
    #[cfg(feature = "notebook")]
    Prune {
        path: Option<PathBuf>,
        /// List what would be deleted without touching anything.
        #[arg(long)]
        dry_run: bool,
    },
    /// Inspect the knowledge graph.
    Graph {
        #[command(subcommand)]
        action: GraphAction,
    },
    /// Browse and apply baked ontology presets.
    Ontology {
        #[command(subcommand)]
        action: OntologyAction,
    },
    /// Exact/regex (ripgrep-style) search over the extracted text.
    Grep {
        /// regex or literal pattern
        pattern: String,
        /// knowledge-base directory (default: nearest indexed root / current dir)
        path: Option<PathBuf>,
        #[arg(short = 'i', long, help = "case-insensitive matching (-i)")]
        ignore_case: bool,
        #[arg(short = 'F', long)]
        fixed: bool,
        #[arg(short = 'w', long)]
        word: bool,
        #[arg(short = 'g', long)]
        glob: Option<String>,
        #[arg(short = 't', long = "type")]
        file_type: Option<String>,
        /// Restrict results to one document or path-glob (e.g. `manual.pdf` or `guides/**`);
        /// ANDed with --glob when both are set.
        #[arg(long)]
        scope: Option<String>,
        #[arg(short = 'A', long, help = "N context lines after each match (-A)")]
        after: Option<usize>,
        #[arg(short = 'B', long, help = "N context lines before each match (-B)")]
        before: Option<usize>,
        #[arg(
            short = 'C',
            long,
            help = "N context lines before AND after each match (-C)"
        )]
        context: Option<usize>,
        #[arg(
            short = 'o',
            long = "only-matching",
            help = "print only the matched substrings (-o)"
        )]
        only_matching: bool,
        #[arg(
            short = 'n',
            long = "line-number",
            help = "prefix each line with its chunk line number (-n)"
        )]
        line_number: bool,
        #[arg(
            short = 'c',
            long,
            help = "print only a count of matching lines per chunk (-c)"
        )]
        count: bool,
        #[arg(
            short = 'm',
            long = "max-count",
            help = "stop after N matching lines per chunk (-m)"
        )]
        max_count: Option<usize>,
        #[arg(short = 'U', long, help = "let the pattern span lines (-U)")]
        multiline: bool,
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
        /// DEPRECATED: images are off by default now — use `--vision` to enable them. Kept as an
        /// accepted no-op so existing launch commands don't break; it still forces images off.
        #[arg(short = 'N', long = "noimage", env = "GLOSSA_NO_IMAGE", hide = true)]
        no_image: bool,
        /// Enable image output in the `read` tool — embedded figures and `page_image`, served as
        /// JPEG. OFF by default: a figure-heavy page's base64 image payload can overflow the stdio
        /// JSON-RPC frame and drop the connection. Safe to enable on `--transport streamable-http`.
        #[arg(long = "vision", env = "GLOSSA_VISION")]
        vision: bool,
        /// Enable the `get_source_file` tool — delivers the ORIGINAL source file behind a citation
        /// (for attribution/download). OFF by default: many clients can't use the returned file
        /// resource, and it is dead weight where nothing consumes it. Opt in when the client does.
        #[arg(long = "source-file", env = "GLOSSA_SOURCE_FILE")]
        source_file: bool,
        /// Transport: stdio (local subprocess) or streamable-http (network endpoint at <bind>/mcp).
        #[arg(
            long,
            value_enum,
            default_value = "stdio",
            env = "GLOSSA_MCP_TRANSPORT"
        )]
        transport: McpTransport,
        /// Bind address for --transport streamable-http.
        #[arg(long, default_value = "127.0.0.1:8080", env = "GLOSSA_MCP_BIND")]
        bind: String,
        /// Extra allowed `Host` header value(s) for streamable-http (DNS-rebind guard). Repeatable.
        /// Default permits loopback only — set your gateway/public host(s) for a prod deployment.
        #[arg(long = "allowed-host")]
        allowed_hosts: Vec<String>,
        /// Optional bearer token guarding the streamable-http `/mcp` endpoint. If set (flag or
        /// `GLOSSA_MCP_TOKEN`), every `/mcp` request must send `Authorization: Bearer <token>` or is
        /// rejected with 401; `/health`, `/ready`, `/metrics` stay open for probes. Unset → no auth
        /// (the loopback default). Ignored for `--transport stdio` (a local subprocess).
        #[arg(long = "auth-token", env = "GLOSSA_MCP_TOKEN", hide_env_values = true)]
        auth_token: Option<String>,
        /// Idle-session timeout in seconds for the streamable-http transport: a session that makes no
        /// request for this long is refused with 404 on its next request, so the client
        /// re-initializes (a cheap handshake; the KB holds no per-session state). OPT-IN — `0`
        /// (default) disables it. Set e.g. `900` (15 min) for a corporate policy.
        #[arg(long = "session-idle-secs", env = "GLOSSA_MCP_SESSION_IDLE_SECS", default_value_t = 0)]
        session_idle_secs: u64,
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

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum ImportModeArg {
    /// Upsert into the existing graph, keeping everything already there (default).
    Merge,
    /// Prune the file's exported types first, then upsert (file = source of truth).
    Replace,
}

impl From<ImportModeArg> for glossa::graph::io::ImportMode {
    fn from(m: ImportModeArg) -> Self {
        match m {
            ImportModeArg::Merge => Self::Merge,
            ImportModeArg::Replace => Self::Replace,
        }
    }
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
    Stats { path: Option<PathBuf> },
    /// Find graph nodes by concept (the `glossary` tool) — prints `id [type] label` + edges.
    #[command(visible_aliases = ["search", "find"])]
    Glossary {
        /// concept in your own words, e.g. "connection loss"
        query: String,
        path: Option<PathBuf>,
        /// Show the graph as it was valid on this date (ISO-8601); a matched node outside its
        /// validity interval is hidden. Timeless nodes are always shown.
        #[arg(long = "as-of")]
        as_of: Option<String>,
        /// Restrict results to one document or path-glob (e.g. `manual.pdf` or `guides/**`).
        #[arg(long)]
        scope: Option<String>,
    },
    /// Run a read-only SQL SELECT over the graph (the `sql` tool). Empty SQL prints the schema.
    Query {
        /// a SELECT over nodes/edges/node_validity/edges_labeled; empty = show schema
        #[arg(default_value = "")]
        sql: String,
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
        /// Show the graph as it was valid on this date (ISO-8601); nodes outside their
        /// validity interval are hidden. Timeless nodes are always shown.
        #[arg(long = "as-of")]
        as_of: Option<String>,
        /// Reference instant for validity status (defaults to now). Deterministic in tests.
        #[arg(long)]
        now: Option<String>,
    },
    /// Run the deterministic generalization pass: transitive closure, SIMILAR links, communities
    /// and centrality (written as derived `auto-generalized` edges + `node_meta`). With `--merge`,
    /// also COLLAPSE near-duplicate nodes (mutates/deletes agent nodes); without it, report only.
    Generalize {
        path: Option<PathBuf>,
        #[arg(
            short = 'm',
            long,
            help = "also collapse near-duplicate nodes (destructive)"
        )]
        merge: bool,
    },
    /// Diagnose graph health: ungrounded / stale / incomplete nodes.
    Doctor {
        path: Option<PathBuf>,
        /// Delete off-spine (incomplete/degenerate) nodes.
        #[arg(long = "prune-incomplete")]
        prune_incomplete: bool,
        /// Delete ungrounded nodes (last resort; prefer re-grounding).
        #[arg(long = "prune-ungrounded")]
        prune_ungrounded: bool,
        /// Delete dangling nodes (last resort; prefer restoring the terminal).
        #[arg(long = "prune-dangling")]
        prune_dangling: bool,
        /// Override the mass-wipe guard and force the dangling prune even when it looks like an
        /// ontology mismatch (zero live terminals) or over half the reasoning layer. Human-only:
        /// not exposed over MCP.
        #[arg(long = "force")]
        force: bool,
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
        /// Show the graph as it was valid on this date (ISO-8601); nodes outside their
        /// validity interval are hidden. Timeless nodes are always shown.
        #[arg(long = "as-of")]
        as_of: Option<String>,
        /// Reference instant for validity status (defaults to now). Deterministic in tests.
        #[arg(long)]
        now: Option<String>,
        /// Restrict results to one document or path-glob (e.g. `manual.pdf` or `guides/**`).
        #[arg(long)]
        scope: Option<String>,
    },
    /// Show a node: type, label, provenance, and its outgoing edges.
    Node {
        node_id: String,
        path: Option<PathBuf>,
        /// Show the graph as it was valid on this date (ISO-8601); the node is treated as not
        /// found when outside its validity interval. Timeless nodes are always shown.
        #[arg(long = "as-of")]
        as_of: Option<String>,
        /// Reference instant for validity status (defaults to now). Deterministic in tests.
        #[arg(long)]
        now: Option<String>,
    },
    /// Cross-document reasoning bridge (the `reach` tool). Omit `--to` for DISCOVERY: walk
    /// `--relation` forward from `--from`, crossing document boundaries on shared mentions (the
    /// bridge, on by default), and print every node reached. Pass `--to` for VERIFY: does a
    /// grounded path from `--from` to that node exist? Replaces the old `path` command —
    /// `--to` + `--no-bridge` with no `--relation` reproduces a plain shortest-path lookup.
    Reach {
        /// start: node id (e.g. from `glossary`)
        #[arg(long)]
        from: String,
        /// relation to follow, fuzzy-matched to the ontology's real edge types; omit = all
        /// chaining relations (undirected)
        #[arg(short = 'r', long)]
        relation: Option<String>,
        /// end: node id to verify a connection to; omit for discovery
        #[arg(long)]
        to: Option<String>,
        path: Option<PathBuf>,
        /// Disable the cross-document bridge (graph-only, in-document connectivity only).
        #[arg(long = "no-bridge")]
        no_bridge: bool,
        #[arg(short = 'd', long, default_value_t = 6)]
        max_depth: usize,
        /// Restrict results to one document or path-glob (e.g. `manual.pdf` or `guides/**`).
        #[arg(long)]
        scope: Option<String>,
    },
    /// Dump all nodes (optionally filtered by type) with their outgoing edges.
    Dump {
        /// corpus directory (default: current directory)
        path: Option<PathBuf>,
        /// only show nodes of this type, e.g. Symptom or Resolution (omit for all)
        #[arg(short = 't', long = "type")]
        node_type: Option<String>,
        /// output format: text (default), json, dot, graphml, html
        /// (html = self-contained offline interactive viewer)
        #[arg(short = 'f', long, default_value = "text")]
        format: String,
        /// Show the graph as it was valid on this date (ISO-8601); nodes outside their
        /// validity interval are hidden. Timeless nodes are always shown.
        #[arg(long = "as-of")]
        as_of: Option<String>,
        /// Reference instant for validity status (defaults to now). Deterministic in tests.
        #[arg(long)]
        now: Option<String>,
    },
    /// Import a graph file (JSON). Default MERGES into the existing graph; `--mode replace` treats
    /// the file as source of truth for its types (prunes them first).
    Import {
        file: PathBuf,
        path: PathBuf,
        #[arg(short = 'f', long)]
        format: Option<String>,
        /// merge (default) = upsert into the existing graph; replace = prune the file's types first.
        #[arg(long, value_enum, default_value = "merge")]
        mode: ImportModeArg,
    },
    /// Delete all nodes of the given type (and edges touching them) — clean-slate a semantic layer.
    Prune {
        path: PathBuf,
        /// node type to delete, e.g. Symptom (repeatable)
        #[arg(short = 't', long = "type", required = true)]
        node_type: Vec<String>,
    },
    /// Compile a document's `.csp` limit tables (notebook notes) into the constraint graph.
    #[cfg(feature = "constraint")]
    Build {
        path: Option<PathBuf>,
        /// Owner document (`Field.source_path`), corpus-relative.
        #[arg(long)]
        doc: String,
        /// Directory with `*.csp` (default: the document's notes mirror under `.glossa/notes/`).
        #[arg(long)]
        tables_dir: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum OntologyAction {
    /// List the preset catalog (grouped by tier, then family).
    List {
        #[arg(long)]
        family: Option<String>,
        #[arg(long)]
        tier: Option<u8>,
    },
    /// Print a preset's TOML (accepts a name or alias).
    Show { name: String },
    /// Materialize a preset to <path>/.glossa/ontology.toml (no indexing).
    Init {
        path: Option<PathBuf>,
        #[arg(short = 't', long = "template")]
        template: String,
        #[arg(long)]
        force: bool,
    },
    /// Rank presets against a free-text description of your documents.
    Suggest {
        #[arg(trailing_var_arg = true, required = true)]
        text: Vec<String>,
    },
}

/// Unix seconds (UTC) -> the strict `YYYY-MM-DDThh:mm:ssZ` form `temporal::normalize_point`
/// accepts unchanged. Used as the `graph node` status reference when neither `--as-of` nor
/// `--now` is given (system time). Thin delegate — the actual (hand-rolled, no date crate) logic
/// lives in `graph::temporal` so the lib crate's `read` renderer can share it.
fn epoch_to_rfc3339(secs: i64) -> String {
    glossa::graph::temporal::epoch_to_rfc3339(secs)
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
    pub no_image: bool,
    pub no_source_file: bool,
    pub transport: McpTransport,
    pub bind: String,
    pub allowed_hosts: Vec<String>,
    /// Optional bearer token for the streamable-http `/mcp` endpoint (None → unauthenticated).
    pub auth_token: Option<String>,
    /// Idle-session timeout (seconds) for streamable-http; 0 disables (opt-in).
    pub session_idle_secs: u64,
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
    let server = glossa::mcp::GlossaServer::new(
        p.path,
        p.profile,
        p.trace,
        glossa::mcp::ServerFlags {
            no_graph: p.no_graph,
            no_image: p.no_image,
            no_source_file: p.no_source_file,
        },
    );
    // Freshness runs on EVERY instance (readers stay current). The heavy generalize loop runs ONLY on
    // the indexer (editor/full); among multiple editors it is further serialized by generalize.lock.
    let run_maintenance = p.profile != glossa::mcp::Profile::Reader;
    let transport = p.transport;
    let bind = p.bind;
    let allowed_hosts = p.allowed_hosts;
    let auth_token = p.auth_token;
    let session_idle_secs = p.session_idle_secs;
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
                tokio::spawn(async move { freshen_srv.freshen_now().await });
                let _ = service.waiting().await; // client-driven: exit on stdin EOF
                cancel.cancel();
            }
            McpTransport::StreamableHttp => {
                serve_streamable_http(
                    server,
                    &bind,
                    allowed_hosts,
                    auth_token,
                    session_idle_secs,
                    cancel,
                    on_transport_ready,
                )
                .await?;
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
    auth_token: Option<String>,
    session_idle_secs: u64,
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
    let http = server.http_metrics();
    let service = StreamableHttpService::new(
        move || {
            // A fresh session must NOT share the anti-loop tracker with any other session — the
            // rest of `server`'s Arc fields (index caches, http metrics, etc.) stay shared by
            // design, only `signals` gets swapped for a brand-new tracker per session.
            let mut s = server.clone();
            s.signals = std::sync::Arc::new(std::sync::Mutex::new(
                glossa::tools::retrieval_progress::ReaderSignals::new(),
            ));
            Ok(s)
        },
        std::sync::Arc::new(LocalSessionManager::default()),
        config,
    );
    // Guard ONLY the /mcp endpoint; /health, /ready, /metrics stay open so probes/monitoring work
    // without a token. Unset token → no auth (the loopback default).
    let mut mcp = axum::Router::new().nest_service("/mcp", service);
    // Idle-session timeout (opt-in), applied INNER so bearer auth (added after, thus outer) runs
    // first: an unauthenticated request gets 401 before we ever look at its session.
    let idle_ms = session_idle_secs.saturating_mul(1000);
    let activity = std::sync::Arc::new(glossa::session_idle::SessionActivity::new());
    if idle_ms > 0 {
        tracing::info!("MCP idle-session timeout: {session_idle_secs}s on /mcp (expired → 404, client re-inits)");
        mcp = mcp.layer(axum::middleware::from_fn_with_state(
            IdleState {
                activity: activity.clone(),
                idle_ms,
            },
            session_idle_layer,
        ));
    }
    match auth_token {
        Some(token) => {
            tracing::info!("MCP auth: bearer token required on /mcp (health endpoints stay open)");
            mcp = mcp.layer(axum::middleware::from_fn_with_state(
                AuthState {
                    token: std::sync::Arc::new(token),
                    metrics: http.clone(),
                },
                bearer_auth_layer,
            ));
        }
        None => {
            tracing::info!(
                "MCP auth: DISABLED (no --auth-token / GLOSSA_MCP_TOKEN) — serve on loopback or behind a TLS/auth gateway"
            );
        }
    }
    // Request metrics wrap /health, /ready and /mcp. /metrics is registered AFTER this `.layer`, so
    // scraping it is NOT counted as a served request (axum applies a layer only to routes added
    // before it) — the scrape must not measure itself.
    let observed = axum::Router::new()
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
        .merge(mcp)
        .layer(axum::middleware::from_fn_with_state(
            http.clone(),
            http_metrics_layer,
        ));
    let app = observed
        // Prometheus metrics (index/graph size, derived-layer staleness, HTTP request metrics).
        .route(
            "/metrics",
            axum::routing::get(move || {
                let s = metrics_srv.clone();
                async move { s.metrics_text() }
            }),
        )
        .layer(tower_http::trace::TraceLayer::new_for_http());
    let listener = tokio::net::TcpListener::bind(bind).await?;
    tracing::info!(
        "glossa MCP (streamable-http) on http://{bind}/mcp  (+ /health /ready /metrics)"
    );
    if let Some(f) = on_transport_ready {
        f();
    }
    tokio::spawn(async move { freshen_srv.freshen_now().await });
    if idle_ms > 0 {
        // Housekeeping: periodically drop sessions abandoned past the idle window so the activity
        // map can't grow unbounded. Stops with the server (shares `cancel`).
        let reaper = activity.clone();
        let rcancel = cancel.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = rcancel.cancelled() => break,
                    _ = tokio::time::sleep(std::time::Duration::from_secs(60)) => {
                        reaper.reap(idle_ms, glossa::trace::now_ms());
                    }
                }
            }
        });
    }
    axum::serve(listener, app)
        .with_graceful_shutdown(async move { cancel.cancelled().await })
        .await?;
    tracing::info!("glossa MCP (streamable-http) stopped");
    Ok(())
}

/// State for the bearer-auth middleware: the expected token plus the metrics handle (so a rejection
/// bumps `glossa_mcp_auth_rejected_total`).
#[derive(Clone)]
struct AuthState {
    token: std::sync::Arc<String>,
    metrics: std::sync::Arc<glossa::http_metrics::HttpMetrics>,
}

/// axum middleware: require `Authorization: Bearer <token>` on the guarded `/mcp` routes. A missing
/// or wrong token is rejected with 401, counted, and logged (a first audit signal for failed access
/// — the IB track wants auth events recorded). The token compare is constant-time (see `mcp_auth`).
async fn bearer_auth_layer(
    axum::extract::State(st): axum::extract::State<AuthState>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let ok = {
        let header = req
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok());
        glossa::mcp_auth::bearer_ok(header, &st.token)
    };
    if ok {
        next.run(req).await
    } else {
        st.metrics.inc_auth_rejected();
        let via = req
            .headers()
            .get("x-forwarded-for")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("-");
        glossa::audit::security_event("auth", "bearer_reject", "denied", via, "/mcp");
        (axum::http::StatusCode::UNAUTHORIZED, "unauthorized\n").into_response()
    }
}

/// axum middleware: time each served request and record it into the shared HTTP metrics (total,
/// status class, in-flight gauge, latency histogram). Applied to /health, /ready and /mcp — not to
/// /metrics itself.
async fn http_metrics_layer(
    axum::extract::State(m): axum::extract::State<std::sync::Arc<glossa::http_metrics::HttpMetrics>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    m.inc_in_flight();
    let start = std::time::Instant::now();
    let resp = next.run(req).await;
    m.dec_in_flight();
    m.record(resp.status().as_u16(), start.elapsed().as_secs_f64());
    resp
}

/// State for the idle-session middleware: the shared activity clock and the threshold (ms).
#[derive(Clone)]
struct IdleState {
    activity: std::sync::Arc<glossa::session_idle::SessionActivity>,
    idle_ms: u64,
}

/// axum middleware: enforce the idle-session timeout on `/mcp`. A request carrying an
/// `Mcp-Session-Id` that has been idle past the threshold is refused with 404 (the streamable-http
/// signal for a terminated session → the client re-initializes); the expiry is audited. Requests
/// without a session id (e.g. `initialize`) pass through untouched.
async fn session_idle_layer(
    axum::extract::State(st): axum::extract::State<IdleState>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let session_id = req
        .headers()
        .get("Mcp-Session-Id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    if let Some(sid) = session_id {
        if !st
            .activity
            .check_and_touch(&sid, st.idle_ms, glossa::trace::now_ms())
        {
            let source = req
                .headers()
                .get("x-forwarded-for")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("-");
            glossa::audit::security_event("session", "idle_expired", "denied", source, &sid);
            return (
                axum::http::StatusCode::NOT_FOUND,
                "session expired — reinitialize\n",
            )
                .into_response();
        }
    }
    next.run(req).await
}

fn main() -> anyhow::Result<()> {
    // Structured logs go to STDERR — stdout is the stdio JSON-RPC channel and must never carry logs.
    // Level via RUST_LOG (default `info`). `GLOSSA_LOG_FORMAT=json` emits one JSON object per line
    // (for a SIEM / log pipeline); anything else is the human-readable default. Best-effort init (a
    // second init in tests is a no-op). Read from the env directly — logging is set up before Cli
    // parsing so parse errors are still logged.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        // Our logs at info; silence noisy deps. RUST_LOG overrides.
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,tantivy=warn,pdf_oxide=error"));
    let json_logs = std::env::var("GLOSSA_LOG_FORMAT")
        .map(|v| v.eq_ignore_ascii_case("json"))
        .unwrap_or(false);
    if json_logs {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .json()
            .flatten_event(true)
            .try_init();
    } else {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .try_init();
    }
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Search {
            pattern,
            path,
            ignore_case,
            word,
            fixed,
            glob,
            file_type,
            scope,
            limit,
            scan,
            no_ignore,
            format,
        } => {
            let path = resolve_root_logged(path);
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
                for h in idx.search_filtered(
                    &pattern,
                    limit,
                    glob.as_deref(),
                    file_type.as_deref(),
                    scope.as_deref(),
                )? {
                    rg_lines.push(format!(
                        "{}:{}: {}  [{:.3}]",
                        h.path, h.location, h.snippet, h.score
                    ));
                    display.push(glossa::cli_fmt::DisplayHit {
                        file: glossa::cli_fmt::rel_file(&path, &h.path),
                        location: h.location.clone(),
                        snippet: h.snippet.clone(),
                        score: Some(h.score),
                    });
                    records.push((h.path.clone(), h.location.clone()));
                }
            } else {
                let opts = QueryOpts {
                    ignore_case,
                    smart_case: !ignore_case,
                    word,
                    fixed,
                };
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
                print!(
                    "{}",
                    glossa::cli_fmt::render_search_pretty(&display, !scan, &pattern)
                );
            } else {
                for l in &rg_lines {
                    println!("{l}");
                }
            }
            Ok(())
        }
        Cmd::Cat { target } => {
            // A `cat` for Office/PDF: extract the whole file's text straight from disk (no index).
            if !target.exists() {
                anyhow::bail!("no such file: {}", target.display());
            }
            print_read(&target, None)
        }
        Cmd::Read { target, location } => {
            // Precedence: existing path beats result-number beats fallback path open.
            // A real file named "3" should be opened directly, not treated as result #3.
            if std::path::Path::new(&target).exists() {
                // 1. Target is an existing path — open it directly.
                print_read(std::path::Path::new(&target), location.as_deref())?;
            } else if let Ok(n) = target.parse::<usize>() {
                // 2. Target is a number and no file by that name exists — resolve from last search.
                let root = resolve_root_logged(None);
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
                                    println!(
                                        "{}",
                                        glossa::cli_fmt::dim(&format!("── {p} · {loc} ──"))
                                    );
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
        Cmd::Index {
            path,
            force,
            file,
            ontology,
        } => {
            let root = resolve_root_logged(path);
            let started = std::time::Instant::now();
            if let Some(rel) = file {
                let idx = glossa::index::store::DocIndex::open_or_create(&root)?;
                let Some(rel) = idx.canonical_document_path(&rel) else {
                    anyhow::bail!("not an indexed document: {rel}");
                };
                let _lock = glossa::index::lock::try_index_lock(&root)
                    .ok_or_else(|| anyhow::anyhow!("another process is indexing; try again"))?;
                glossa::index::store::index_one_file_locked(&root, &rel)?;
                println!(
                    "reindexed {rel} in {}",
                    glossa::cli_fmt::format_elapsed(started.elapsed())
                );
                return Ok(());
            }
            if let Some(name) = ontology {
                match glossa::ontology_templates::write_template(&root, &name, false)? {
                    glossa::ontology_templates::Written::Created => {
                        println!("ontology: wrote '{name}' preset to .glossa/ontology.toml");
                    }
                    glossa::ontology_templates::Written::Kept => {
                        eprintln!(
                            "ontology.toml already exists; keeping it, ignoring --ontology {name} \
                             (use `kb ontology init --force` to replace)"
                        );
                    }
                    glossa::ontology_templates::Written::Overwritten => unreachable!("force=false"),
                }
            }
            // Seed a default whitelist `.ignore` on a corpus that has none, so a first index doesn't
            // slurp installers/archives/temp files as text. Never clobbers an existing ignore setup.
            if let Some(p) = glossa::default_ignore::seed_if_absent(&root) {
                eprintln!(
                    "wrote default {} (whitelist of supported types) — edit it to tune what's indexed",
                    p.display()
                );
            }
            let stats = glossa::index::store::index_dir(&root, force)?;
            let skipped = if stats.errors.is_empty() {
                String::new()
            } else {
                format!(", {} skipped(errors)", stats.errors.len())
            };
            println!(
                "indexed: {} added, {} removed, {} unchanged{} in {}",
                stats.added,
                stats.removed,
                stats.unchanged,
                skipped,
                glossa::cli_fmt::format_elapsed(started.elapsed())
            );
            if !stats.errors.is_empty() {
                eprintln!("errors ({}):", stats.errors.len());
                for (p, e) in &stats.errors {
                    eprintln!("  {p}: {e}");
                }
            }
            if force {
                // Auto-run the generalization pass over the freshly rebuilt graph so derived edges
                // (closure + SIMILAR), communities and centrality stay in sync. Non-destructive:
                // merges are only reported, never applied here (use `kb graph generalize --merge`).
                // This mirrors what the old `kb reindex` did — --force is its replacement.
                let g = glossa::graph::store::GraphStore::open(&root)?;
                let ont = glossa::graph::ontology::Ontology::load_or_default(&root);
                let opts = glossa::graph::generalize::apply::Opts::from_ontology(
                    &ont,
                    glossa::trace::now_ms(),
                );
                let r = glossa::graph::generalize::apply::generalize(&g, &opts)?;
                println!(
                    "generalized: inferred_edges={} similar_edges={} communities={} merge_candidates={}",
                    r.inferred_edges, r.similar_edges, r.communities, r.merge_candidates
                );
            }
            Ok(())
        }
        #[cfg(feature = "notebook")]
        Cmd::Prune { path, dry_run } => {
            let root = resolve_root_logged(path);
            let orphans = glossa::index::store::orphan_notes(&root)?;
            if orphans.is_empty() {
                println!("no orphaned notes");
                return Ok(());
            }
            if dry_run {
                println!("would prune {} orphaned note(s):", orphans.len());
                for o in &orphans {
                    println!("  {o}");
                }
            } else {
                let notes_root = root.join(".glossa").join("notes");
                let mut removed = 0usize;
                for o in &orphans {
                    match std::fs::remove_file(notes_root.join(o)) {
                        Ok(()) => removed += 1,
                        Err(e) => eprintln!("prune: failed to remove {o}: {e}"),
                    }
                }
                // Best-effort: remove now-empty mirror directories left behind by deleted notes,
                // walking up to (but not including) the notes root. `remove_dir` only succeeds on an
                // empty dir, so a still-populated mirror is left intact.
                for o in &orphans {
                    let mut dir = notes_root.join(o);
                    while let Some(parent) = dir.parent() {
                        if parent == notes_root.as_path() || std::fs::remove_dir(parent).is_err() {
                            break;
                        }
                        dir = parent.to_path_buf();
                    }
                }
                glossa::index::store::ensure_fresh(&root)?;
                println!("pruned {removed} orphaned note(s)");
            }
            Ok(())
        }
        Cmd::Grep {
            pattern,
            path,
            ignore_case,
            fixed,
            word,
            glob,
            file_type,
            scope,
            after,
            before,
            context,
            only_matching,
            line_number,
            count,
            max_count,
            multiline,
        } => {
            let path = resolve_root_logged(path);
            glossa::index::store::ensure_fresh(&path)?; // file-first: pick up new/changed docs
            let idx = glossa::index::store::DocIndex::open_or_create(&path)?;
            let opts = glossa::grep::GrepOpts {
                ignore_case,
                fixed,
                word,
                glob,
                file_type,
                // -A/-B override the shared -C on their respective side.
                before: before.or(context).unwrap_or(0),
                after: after.or(context).unwrap_or(0),
                only_matching,
                line_number,
                count,
                max_count,
                multiline,
                line_cap: None,
                path: None,
                scope,
            };
            for h in glossa::grep::grep(&idx, &pattern, &opts)? {
                println!("{}", h.display_line());
            }
            Ok(())
        }
        Cmd::Glob { pattern, path } => {
            let path = resolve_root_logged(path);
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
        Cmd::Mcp {
            action,
            path,
            profile,
            trace,
            no_graph,
            no_image,
            vision,
            source_file,
            transport,
            bind,
            allowed_hosts,
            auth_token,
            session_idle_secs,
            windows_service,
            service_name: _service_name,
        } => match action {
            Some(McpAction::DumpTzTools { config_dir }) => {
                let n = glossa::tz_export::dump(&config_dir)?;
                println!(
                    "dump-tz-tools: wrote {} tool schemas and updated tensorzero.toml",
                    n
                );
                Ok(())
            }
            None => {
                let path = resolve_root_traced(path);
                let params = ServeParams {
                    path,
                    profile: glossa::mcp::Profile::parse(&profile),
                    trace,
                    no_graph,
                    // Images are opt-in via --vision; the legacy --noimage still forces them off.
                    no_image: no_image || !vision,
                    // get_source_file is opt-in via --source-file (off by default).
                    no_source_file: !source_file,
                    transport,
                    bind,
                    allowed_hosts,
                    auth_token,
                    session_idle_secs,
                };
                if windows_service {
                    // Launched by the SCM (binPath carries --windows-service): hand off to the
                    // service dispatcher, which runs run_serve under SCM control (Stop → cancel).
                    #[cfg(windows)]
                    {
                        return winsvc::run(params, _service_name);
                    }
                    #[cfg(not(windows))]
                    {
                        anyhow::bail!("--windows-service is only supported on Windows");
                    }
                }
                // Foreground: OS signals (Ctrl-C / SIGTERM) drive graceful shutdown.
                run_serve(
                    params,
                    tokio_util::sync::CancellationToken::new(),
                    true,
                    None,
                )?;
                Ok(())
            }
        },
        Cmd::Graph { action } => match action {
            GraphAction::Stats { path } => {
                let path = resolve_root_logged(path);
                let g = glossa::graph::store::GraphStore::open(&path)?;
                println!("{}", glossa::tools::graph_stats(&g));
                Ok(())
            }
            GraphAction::Glossary { query, path, as_of, scope } => {
                let path = resolve_root_logged(path);
                glossa::index::store::ensure_fresh(&path)?; // file-first: pick up new/changed docs
                let idx = glossa::index::store::DocIndex::open_or_create(&path)?;
                let g = glossa::graph::store::GraphStore::open(&path)?;
                let trace = glossa::trace::TraceLog::disabled();
                let spec = glossa::tools::ChainSpec::from_ontology(
                    &glossa::graph::ontology::Ontology::load_or_default(&path),
                );
                let stale = glossa::tools::StaleChecker::new(path.clone());
                println!(
                    "{}",
                    glossa::tools::glossary(
                        &idx,
                        &g,
                        &query,
                        &spec,
                        &trace,
                        as_of.as_deref(),
                        Some(&stale),
                        scope.as_deref(),
                    )
                );
                Ok(())
            }
            GraphAction::Query { sql, path } => {
                let path = glossa::root::resolve_root(path);
                glossa::index::store::ensure_fresh(&path)?;
                let idx = glossa::index::store::DocIndex::open_or_create(&path)?;
                let g = glossa::graph::store::GraphStore::open(&path)?;
                let trace = glossa::trace::TraceLog::disabled();
                println!("{}", glossa::tools::sql(&idx, &g, &sql, &trace));
                Ok(())
            }
            GraphAction::Ls {
                path,
                node_type,
                limit,
                as_of,
                now: _now,
            } => {
                let path = resolve_root_logged(path);
                let g = glossa::graph::store::GraphStore::open(&path)?;
                let at = as_of
                    .as_deref()
                    .map(glossa::graph::temporal::normalize_point)
                    .transpose()?;
                let nodes = g.all_nodes()?;
                match node_type {
                    None => {
                        // per-type summary — the browse overview
                        let mut counts: std::collections::BTreeMap<String, usize> =
                            std::collections::BTreeMap::new();
                        for n in &nodes {
                            if let Some(a) = &at {
                                if !g.visible_at(&n.id, a)? {
                                    continue;
                                }
                            }
                            *counts.entry(n.node_type.clone()).or_default() += 1;
                        }
                        for (t, c) in &counts {
                            println!("{t}: {c}");
                        }
                        println!("\n(use --type <T> to list nodes, or `kb graph search <query>`)");
                    }
                    Some(t) => {
                        let mut matched = Vec::new();
                        for n in nodes.iter().filter(|n| n.node_type == t) {
                            if let Some(a) = &at {
                                if !g.visible_at(&n.id, a)? {
                                    continue;
                                }
                            }
                            matched.push(n);
                        }
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
            GraphAction::Generalize { path, merge } => {
                let path = resolve_root_logged(path);
                let g = glossa::graph::store::GraphStore::open(&path)?;
                let ont = glossa::graph::ontology::Ontology::load_or_default(&path);
                let mut opts = glossa::graph::generalize::apply::Opts::from_ontology(
                    &ont,
                    glossa::trace::now_ms(),
                );
                opts.apply_merges = merge;
                let r = glossa::graph::generalize::apply::generalize(&g, &opts)?;
                println!(
                    "generalize: inferred_edges={} similar_edges={} communities={} \
                     merge_candidates={} merged_nodes={}",
                    r.inferred_edges, r.similar_edges, r.communities, r.merge_candidates, r.merged_nodes,
                );
                Ok(())
            }
            GraphAction::Doctor {
                path,
                prune_incomplete,
                prune_ungrounded,
                mut prune_dangling,
                force,
            } => {
                let path = resolve_root_logged(path);
                let g = glossa::graph::store::GraphStore::open(&path)?;
                let ont = glossa::graph::ontology::Ontology::load_or_default(&path);
                let report = glossa::graph::doctor::doctor(&g, &ont, &path)?;
                print!("{}", glossa::graph::ops::fmt_doctor_report(&report));
                if prune_dangling && !force {
                    if let Some(reason) = glossa::graph::doctor::dangling_prune_risk(&report, &g, &ont)
                    {
                        prune_dangling = false;
                        println!(
                            "dangling prune REFUSED: {reason}\nre-run with --force to override."
                        );
                    }
                }
                if prune_incomplete || prune_ungrounded || prune_dangling {
                    let (inc, ung, dang) = glossa::graph::doctor::prune(
                        &g,
                        &report,
                        &glossa::graph::doctor::PruneOpts {
                            incomplete: prune_incomplete,
                            ungrounded: prune_ungrounded,
                            dangling: prune_dangling,
                        },
                    )?;
                    println!("pruned: incomplete={inc} ungrounded={ung} dangling={dang}");
                }
                Ok(())
            }
            GraphAction::Near {
                node_id,
                path,
                depth,
                types,
                as_of,
                now: _now,
                scope,
            } => {
                let path = resolve_root_logged(path);
                let g = glossa::graph::store::GraphStore::open(&path)?;
                let filter = if types.is_empty() {
                    None
                } else {
                    Some(types.as_slice())
                };
                let at = as_of
                    .as_deref()
                    .map(glossa::graph::temporal::normalize_point)
                    .transpose()?;
                // `Near` is a multi-hop BFS lister (`traverse::neighbors`), a different primitive
                // from the `neighbors` MCP tool (1-hop typed edges, `tools::neighbors`) — it has no
                // shared core fn to thread `scope` through, so filter its output ids directly here
                // with the same doc-attribution rule (`tools::owning_doc`/`in_scope`) the tools use.
                let scope_glob = glossa::tools::compile_scope(scope.as_deref())
                    .map_err(|e| anyhow::anyhow!(e))?;
                for id in glossa::graph::traverse::neighbors(&g, &node_id, filter, depth)? {
                    if let Some(a) = &at {
                        if !g.visible_at(&id, a)? {
                            continue;
                        }
                    }
                    if !glossa::tools::in_scope(
                        scope_glob.as_ref(),
                        glossa::tools::owning_doc(&g, &id).as_deref(),
                    ) {
                        continue;
                    }
                    // Section ids are opaque ordinals (`<path>#<n>`); print the node label
                    // (heading) alongside so the output stays human-readable.
                    match g.get_node(&id)? {
                        Some(n) if !n.label.is_empty() && n.label != id => {
                            println!("{id}  {}", n.label)
                        }
                        _ => println!("{id}"),
                    }
                }
                Ok(())
            }
            GraphAction::Node { node_id, path, as_of, now } => {
                let path = resolve_root_logged(path);
                let g = glossa::graph::store::GraphStore::open(&path)?;
                let at = as_of
                    .as_deref()
                    .map(glossa::graph::temporal::normalize_point)
                    .transpose()?;
                if let Some(a) = &at {
                    if !g.visible_at(&node_id, a)? {
                        println!("node not found: {node_id}");
                        return Ok(());
                    }
                }
                match g.get_node(&node_id)? {
                    Some(n) => {
                        let mut edges = g.outgoing(&node_id)?;
                        if let Some(a) = &at {
                            let mut kept = Vec::with_capacity(edges.len());
                            for e in edges {
                                if g.visible_at(&e.to, a)? {
                                    kept.push(e);
                                }
                            }
                            edges = kept;
                        }
                        print!("{}", glossa::cli_fmt::render_node(&n, &edges));
                        if let Some(v) = g.validity_for(&node_id)? {
                            let from_disp = v.valid_from_raw.as_deref().unwrap_or("(open)");
                            let to_disp = v.valid_to_raw.as_deref().unwrap_or("(open)");
                            println!("  valid:  {from_disp} .. {to_disp}");
                            // Reference instant: --as-of if given, else --now, else system time.
                            let reference = match (&at, &now) {
                                (Some(a), _) => a.clone(),
                                (None, Some(nw)) => glossa::graph::temporal::normalize_point(nw)?,
                                (None, None) => epoch_to_rfc3339((glossa::trace::now_ms() / 1000) as i64),
                            };
                            let superseded = g
                                .incoming(&node_id)?
                                .iter()
                                .any(|e| e.edge_type == "SUPERSEDES");
                            let status = glossa::graph::temporal::status(
                                v.valid_from.as_deref(),
                                v.valid_to.as_deref(),
                                superseded,
                                &reference,
                            );
                            let status_str = match status {
                                glossa::graph::temporal::Status::Future => "future",
                                glossa::graph::temporal::Status::Current => "current",
                                glossa::graph::temporal::Status::Expired => "expired",
                                glossa::graph::temporal::Status::Superseded => "superseded",
                            };
                            println!("  status: {status_str}");
                        }
                    }
                    None => println!("node not found: {node_id}"),
                }
                Ok(())
            }
            GraphAction::Reach {
                from,
                relation,
                to,
                path,
                no_bridge,
                max_depth,
                scope,
            } => {
                let path = glossa::root::resolve_root(path);
                glossa::index::store::ensure_fresh(&path)?;
                let idx = glossa::index::store::DocIndex::open_or_create(&path)?;
                let g = glossa::graph::store::GraphStore::open(&path)?;
                let ont = glossa::graph::ontology::Ontology::load_or_default(&path);
                let trace = glossa::trace::TraceLog::disabled();
                println!(
                    "{}",
                    glossa::tools::reach(
                        &idx,
                        &g,
                        &ont,
                        Some(&from),
                        None,
                        None,
                        relation.as_deref(),
                        to.as_deref(),
                        None,
                        None,
                        max_depth,
                        !no_bridge,
                        &trace,
                        scope.as_deref(),
                    )
                );
                Ok(())
            }
            GraphAction::Dump {
                path,
                node_type,
                format,
                as_of,
                now: _now,
            } => {
                let path = resolve_root_logged(path);
                let g = glossa::graph::store::GraphStore::open(&path)?;
                let at = as_of
                    .as_deref()
                    .map(glossa::graph::temporal::normalize_point)
                    .transpose()?;
                match format.as_str() {
                    "text" => {
                        let mut nodes = g.all_nodes()?;
                        nodes.sort_by(|a, b| a.node_type.cmp(&b.node_type).then(a.id.cmp(&b.id)));
                        for n in &nodes {
                            if node_type.as_deref().is_some_and(|t| t != n.node_type) {
                                continue;
                            }
                            if let Some(a) = &at {
                                if !g.visible_at(&n.id, a)? {
                                    continue;
                                }
                            }
                            let al = if n.aliases.is_empty() {
                                String::new()
                            } else {
                                format!("  ({})", n.aliases.join(", "))
                            };
                            println!("[{}] {}  {}{}", n.node_type, n.id, n.label, al);
                            for e in g.outgoing(&n.id)? {
                                if let Some(a) = &at {
                                    if !g.visible_at(&e.to, a)? {
                                        continue;
                                    }
                                }
                                println!("    -{}-> {}", e.edge_type, e.to);
                            }
                        }
                    }
                    "json" | "dot" | "graphml" | "html" => {
                        use glossa::graph::io::{collect, to_dot, to_graphml, to_html, to_json};
                        let mut export = collect(&g, node_type.as_deref())?;
                        if let Some(a) = &at {
                            let mut visible_ids = std::collections::HashSet::new();
                            for n in &export.nodes {
                                if g.visible_at(&n.id, a)? {
                                    visible_ids.insert(n.id.clone());
                                }
                            }
                            export.nodes.retain(|n| visible_ids.contains(&n.id));
                            let mut kept_edges = Vec::with_capacity(export.edges.len());
                            for e in export.edges {
                                // An endpoint outside the exported node set (e.g. a structural
                                // node) is checked directly — absent-from-export doesn't mean
                                // hidden-by-as-of.
                                let from_ok = visible_ids.contains(&e.from)
                                    || g.visible_at(&e.from, a)?;
                                let to_ok =
                                    visible_ids.contains(&e.to) || g.visible_at(&e.to, a)?;
                                if from_ok && to_ok {
                                    kept_edges.push(e);
                                }
                            }
                            export.edges = kept_edges;
                        }
                        match format.as_str() {
                            "json" => print!("{}", to_json(&export)?),
                            "dot" => print!("{}", to_dot(&export)),
                            "graphml" => print!("{}", to_graphml(&export)),
                            "html" => print!("{}", to_html(&g, &export, &path)),
                            _ => unreachable!(),
                        }
                    }
                    other => anyhow::bail!(
                        "unknown format {:?} — valid formats: text, json, dot, graphml, html",
                        other
                    ),
                }
                Ok(())
            }
            GraphAction::Import { file, path, format, mode } => {
                let fmt = format.as_deref().map(|s| s.to_string()).unwrap_or_else(|| {
                    file.extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("json")
                        .to_string()
                });
                if fmt != "json" {
                    anyhow::bail!("import supports json only (graphml/dot are export-only)");
                }
                let contents = std::fs::read_to_string(&file)?;
                let export = glossa::graph::io::from_json(&contents)?;
                let ont = glossa::graph::ontology::Ontology::load_or_default(&path);
                let now = glossa::trace::now_ms();
                let g = glossa::graph::store::GraphStore::open(&path)?;
                let (pruned, n, ed) =
                    glossa::graph::io::import_layer(&g, &ont, export, now, &path, mode.into())?;
                println!(
                    "graph import ({}): pruned {pruned}, +{n} nodes, +{ed} edges",
                    match mode {
                        ImportModeArg::Merge => "merge",
                        ImportModeArg::Replace => "replace",
                    }
                );
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
            #[cfg(feature = "constraint")]
            GraphAction::Build {
                path,
                doc,
                tables_dir,
            } => {
                let root = resolve_root_logged(path);
                glossa::index::store::ensure_fresh(&root)?;
                let tables = tables_dir.unwrap_or_else(|| {
                    glossa::notebook::notes_root(&root)
                        .join(glossa::notebook::mirror_dir_for_doc(&doc))
                });
                let idx = glossa::index::store::DocIndex::open_or_create(&root)?;
                let g = glossa::graph::store::GraphStore::open(&root)?;
                let ont = glossa::graph::ontology::Ontology::load_or_default(&root);
                let report = glossa::tables::tables_to_graph(&idx, &g, &ont, &doc, &tables)?;
                for line in &report.lines {
                    println!("{line}");
                }
                Ok(())
            }
        },
        Cmd::Ontology { action } => {
            use glossa::ontology_templates as ot;
            match action {
                OntologyAction::List { family, tier } => {
                    let mut cat = ot::catalog();
                    cat.sort_by(|a, b| a.tier.cmp(&b.tier)
                        .then(a.family.cmp(&b.family))
                        .then(a.name.cmp(&b.name)));
                    for t in cat {
                        if let Some(f) = &family {
                            if t.family.as_deref() != Some(f.as_str()) { continue; }
                        }
                        if let Some(n) = tier {
                            if t.tier != n { continue; }
                        }
                        let desc = t.description.as_deref().unwrap_or("");
                        let fam = t.family.as_deref().unwrap_or("-");
                        println!("[tier {}] {:<18} {:<12} {}", t.tier, t.name, fam, desc);
                    }
                    Ok(())
                }
                OntologyAction::Show { name } => {
                    let canon = ot::resolve(&name).ok_or_else(|| {
                        anyhow::anyhow!("unknown preset '{name}' — did you mean: {}? (kb ontology list)",
                            ot::nearest(&name, 3).join(", "))
                    })?;
                    print!("{}", ot::raw(&canon).unwrap());
                    Ok(())
                }
                OntologyAction::Init { path, template, force } => {
                    let root = resolve_root_logged(path);
                    match ot::write_template(&root, &template, force)? {
                        ot::Written::Created => println!("wrote '{template}' to .glossa/ontology.toml"),
                        ot::Written::Overwritten => println!("overwrote .glossa/ontology.toml with '{template}'"),
                        ot::Written::Kept => anyhow::bail!(
                            ".glossa/ontology.toml already exists — pass --force to replace it"),
                    }
                    Ok(())
                }
                OntologyAction::Suggest { text } => {
                    let q = text.join(" ");
                    let hits = ot::suggest(&q, 5);
                    if hits.is_empty() {
                        println!("no preset matched — try `kb ontology list`");
                    }
                    for (name, score) in hits {
                        println!("{name}\t(score {score})");
                    }
                    Ok(())
                }
            }
        }
    }
}
