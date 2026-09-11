use clap::{Parser, Subcommand};
use glossa::query::{compile, QueryOpts};
use glossa::search::search_chunks;
use glossa::walk::collect_chunks;
use std::path::PathBuf;

/// Native Windows Service integration (SCM dispatcher + control handler). Windows-only.
#[cfg(windows)]
mod winsvc;

/// Pure merge of `--root` flags / `GLOSSA_ROOTS` env / `--state-dir` into a `RootInputs`, factored
/// out of `resolve_inputs` so it is unit-testable without clap or real env vars. Precedence: an
/// explicit `--root` flag list wins OUTRIGHT over `GLOSSA_ROOTS` — a single `--root` present
/// suppresses the env entirely (never merged line-by-line). Only when NO `--root` flag is given do
/// we fall back to parsing `GLOSSA_ROOTS` (newline-separated `[LABEL=]PATH`, blank lines skipped).
fn build_root_inputs(
    positional: Option<PathBuf>,
    flags: &[String],
    env_roots: Option<String>,
    state_dir: Option<PathBuf>,
) -> anyhow::Result<glossa::root::RootInputs> {
    let roots = if !flags.is_empty() {
        flags
            .iter()
            .map(|s| glossa::root::parse_root_arg(s))
            .collect::<anyhow::Result<Vec<_>>>()?
    } else if let Some(env) = env_roots {
        env.lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(glossa::root::parse_root_arg)
            .collect::<anyhow::Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    Ok(glossa::root::RootInputs {
        positional,
        roots,
        state_dir,
    })
}

/// Scan raw argv for `--config <path>` / `--config=<path>` without invoking clap (clap parsing
/// happens later, in `Cli::parse()`; this only needs to run early enough to gate the logging-init
/// block). Global flags in this CLI can appear before or after the subcommand, so this scans the
/// whole argv, not just a fixed position.
fn peek_config_flag(args: &[String]) -> Option<PathBuf> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if let Some(v) = a.strip_prefix("--config=") {
            return Some(PathBuf::from(v));
        }
        if a == "--config" {
            return it.next().map(PathBuf::from);
        }
    }
    None
}

/// Resolve `roots`/`state_base` for one CLI invocation: build the inputs (flags + `GLOSSA_ROOTS`
/// env), resolve via the shared resolver (real cwd; the explicit-path guard lives there — no
/// silent CWD fallback once state-dir/multi-root is in play), then report the resolved root plus
/// advisories/warnings. `via_tracing` picks the channel: interactive CLI commands print plain
/// lines to stderr; the long-lived server routes them through `tracing` so they match its other
/// logs (and become JSON under `GLOSSA_LOG_FORMAT=json`).
fn resolve_inputs_reported(
    positional: Option<PathBuf>,
    root_flags: &[String],
    state_dir: Option<PathBuf>,
    via_tracing: bool,
) -> anyhow::Result<glossa::root::ResolvedRoot> {
    let env_roots = std::env::var("GLOSSA_ROOTS").ok();
    let inputs = build_root_inputs(positional, root_flags, env_roots, state_dir)?;
    let rr = glossa::root::resolve_roots_verbose(inputs)?;
    // Auto-create + verify `<state_base>/.glossa` is writable BEFORE anything tries to use it — a
    // clear startup error (read-only mount, stale network share, permissions) beats a confusing
    // failure deep inside the index/graph writer.
    glossa::root::ensure_state_writable(&rr.state_base)?;
    let warn = |msg: String| {
        if via_tracing {
            tracing::warn!("{msg}");
        } else {
            eprintln!("warning: {msg}");
        }
    };
    let shown = std::path::absolute(&rr.root).unwrap_or_else(|_| rr.root.clone());
    if via_tracing {
        tracing::info!(root = %shown.display(), "resolved kb root");
    } else {
        eprintln!("root: {}", shown.display());
    }
    for a in rr.advisories() {
        warn(a);
    }
    if let Some(w) = glossa::fs_detect::state_dir_network_warning(
        &glossa::fs_detect::SysFsDetector,
        &rr.state_base,
    ) {
        warn(w);
    }
    // Stale co-located `.glossa`: --state-dir moved state elsewhere, but a root still carries its
    // own populated `.glossa` (graph.sqlite present) — that layer is silently ignored now, which is
    // almost never intended (looks like data loss until you know to check).
    if rr.state_base != rr.root {
        for r in &rr.roots {
            let g = r.path.join(".glossa");
            if g.join("graph.sqlite").exists() {
                warn(format!(
                    "--state-dir set but a populated {} exists — its graph/edges are NOT used; \
                     move it into the state-dir or remove it",
                    g.display()
                ));
            }
        }
    }
    Ok(rr)
}

/// `resolve_inputs` for interactive CLI commands — reports to stderr as plain lines.
fn resolve_inputs(
    positional: Option<PathBuf>,
    root_flags: &[String],
    state_dir: Option<PathBuf>,
) -> anyhow::Result<glossa::root::ResolvedRoot> {
    resolve_inputs_reported(positional, root_flags, state_dir, false)
}

/// `resolve_inputs` for the long-lived MCP server — reports through `tracing`.
fn resolve_inputs_traced(
    positional: Option<PathBuf>,
    root_flags: &[String],
    state_dir: Option<PathBuf>,
) -> anyhow::Result<glossa::root::ResolvedRoot> {
    resolve_inputs_reported(positional, root_flags, state_dir, true)
}

/// Merge `--root`/`GLOSSA_ROOTS`/`--state-dir` with a deployment config's `[corpus]` section: the
/// flag/env pair outranks the file OUTRIGHT — checked here, since `resolve_inputs`'s own flag/env
/// precedence only ever sees whichever roots list it is handed — and the file is consulted only
/// when neither flag nor env supplied any roots. Shared by `Cmd::Mcp` (Task 6) and `Cmd::Index`
/// (Task 7) so this precedence lives in exactly one place.
fn merge_corpus(
    root_flags: &[String],
    state_dir: Option<PathBuf>,
    c: &glossa::config::DeploymentConfig,
) -> (Vec<String>, Option<PathBuf>) {
    let env_roots_present = std::env::var("GLOSSA_ROOTS").is_ok();
    let effective_roots: Vec<String> = if !root_flags.is_empty() || env_roots_present {
        root_flags.to_vec()
    } else {
        c.corpus.roots.clone()
    };
    let effective_state_dir = glossa::config::pick_opt(state_dir, c.corpus.state_dir.clone());
    (effective_roots, effective_state_dir)
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
    /// Corpus source folder(s): repeatable `--root [LABEL=]PATH`. A bare `--root PATH` auto-labels
    /// from the basename. Mutually usable with a single positional PATH (empty-label, back-compat).
    #[arg(long = "root", global = true, value_name = "[LABEL=]PATH")]
    root: Vec<String>,
    /// Local directory that holds `.glossa` state (index/graph/locks). Defaults to the (sole) corpus
    /// root — the co-located behavior. Point at LOCAL disk when the corpus is a network share.
    #[arg(long = "state-dir", global = true, env = "GLOSSA_STATE_DIR")]
    state_dir: Option<PathBuf>,
    /// Path to a TOML deployment config file (see docs/deploy/glossa.toml). Also `GLOSSA_CONFIG`.
    /// Settings here are the role's base; CLI flags and env vars override per setting. Consumed via a
    /// pre-parse argv peek (see `peek_config_flag`) so `[logging]` can be folded in before the tracing
    /// subscriber installs; this field exists so clap surfaces it in `--help` and validates its shape.
    #[arg(long, global = true, env = "GLOSSA_CONFIG")]
    config: Option<PathBuf>,
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
        /// Full rebuild from scratch. This is also the ONLY pass that (re)builds the answer-grounding
        /// DF sidecar (`.glossa/df`): incremental indexing never refreshes it, so run `--force` after
        /// large corpus changes to keep the `verify` gate's rarity counts accurate.
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
        /// `Option` with NO `default_value`: the built-in default now lives in `config::defaults`
        /// (Plan E merges CLI > env > config-file > default), resolved at the wiring layer.
        #[arg(long, value_enum, env = "GLOSSA_MCP_TRANSPORT")]
        transport: Option<McpTransport>,
        /// Bind address for --transport streamable-http. `Option` with NO `default_value` — see
        /// `transport` above; the default lives in `config::defaults::BIND`.
        #[arg(long, env = "GLOSSA_MCP_BIND")]
        bind: Option<String>,
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
        /// Override the non-loopback+no-auth startup refusal (§3c). Logs a loud warning + audit event.
        /// `Option<bool>` with NO `default_value`: unset means "defer to config" — the effective default
        /// `false` lives in `config::defaults` (Plan E merges CLI > env > config-file > default). Resolve
        /// to a bool at the wiring layer before passing to the interlock.
        #[arg(long = "insecure", env = "GLOSSA_MCP_INSECURE")]
        insecure: Option<bool>,
        /// PEM certificate chain for native TLS on --transport streamable-http (requires the `tls`
        /// build feature; the default build has no crypto surface). Set with --tls-key to serve
        /// HTTPS directly instead of terminating TLS at a reverse proxy.
        #[cfg(feature = "tls")]
        #[arg(long = "tls-cert", env = "GLOSSA_TLS_CERT")]
        tls_cert: Option<PathBuf>,
        /// PEM private key matching --tls-cert (requires the `tls` build feature).
        #[cfg(feature = "tls")]
        #[arg(long = "tls-key", env = "GLOSSA_TLS_KEY")]
        tls_key: Option<PathBuf>,
        /// PEM client-CA certificate(s) enabling mTLS: client certs are then REQUIRED and verified
        /// against this CA (requires --tls-cert/--tls-key and the `tls` build feature).
        #[cfg(feature = "tls")]
        #[arg(long = "tls-client-ca", env = "GLOSSA_TLS_CLIENT_CA")]
        tls_client_ca: Option<PathBuf>,
        /// Idle-session timeout in seconds for the streamable-http transport: a session that makes no
        /// request for this long is refused with 404 on its next request, so the client
        /// re-initializes (a cheap handshake; the KB holds no per-session state). OPT-IN — `0`
        /// (default) disables it. Set e.g. `900` (15 min) for a corporate policy.
        /// `Option` with NO `default_value_t` — see `transport` above; the default lives in
        /// `config::defaults::SESSION_IDLE_SECS`.
        #[arg(long = "session-idle-secs", env = "GLOSSA_MCP_SESSION_IDLE_SECS")]
        session_idle_secs: Option<u64>,
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
        /// Delete stale nodes (source drifted; last resort — prefer re-syncing / rebuilding).
        #[arg(long = "prune-stale")]
        prune_stale: bool,
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
        /// only delete nodes GROUNDED in a document whose path contains this substring (e.g. a
        /// generic vendor reference like "CODESYS Control V3"). Without it, the whole type is wiped.
        #[arg(long = "source")]
        source: Option<String>,
        /// list what would be deleted without touching the graph
        #[arg(long = "dry-run")]
        dry_run: bool,
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

/// The live `ReloadableTls`, registered by `serve_streamable_http` once it builds one (`tls`
/// feature + cert/key configured), so the SIGHUP handler above -- spawned earlier, before the
/// transport is known -- can trigger a reload without threading it through as a parameter. Set
/// once per process; a Windows-service restart that reuses the process would keep the previous
/// TLS config until that path is revisited (out of scope here, same limitation as other
/// process-lifetime statics in this file).
#[cfg(feature = "tls")]
static TLS_RELOADABLE: std::sync::OnceLock<std::sync::Arc<glossa::tls::ReloadableTls>> =
    std::sync::OnceLock::new();

/// Everything needed to start one MCP serve instance. Built once from the CLI; reused by both the
/// foreground path and the Windows Service path (which stashes it before the SCM dispatcher starts).
#[derive(Clone)]
pub(crate) struct ServeParams {
    pub roots: Vec<glossa::root::Root>,
    pub state_base: PathBuf,
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
    /// Override the non-loopback+no-auth startup refusal (§3c). `None` → resolved default `false`
    /// (Plan E's config-file merge slots in at this same wiring layer).
    pub insecure: Option<bool>,
    /// Idle-session timeout (seconds) for streamable-http; 0 disables (opt-in).
    pub session_idle_secs: u64,
    /// Native TLS cert/key/client-CA (§3b), only present in a `tls`-feature build.
    #[cfg(feature = "tls")]
    pub tls_cert: Option<PathBuf>,
    #[cfg(feature = "tls")]
    pub tls_key: Option<PathBuf>,
    #[cfg(feature = "tls")]
    pub tls_client_ca: Option<PathBuf>,
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
        p.roots,
        p.state_base,
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
    let insecure = p.insecure;
    let session_idle_secs = p.session_idle_secs;
    #[cfg(feature = "tls")]
    let tls_cert = p.tls_cert;
    #[cfg(feature = "tls")]
    let tls_key = p.tls_key;
    #[cfg(feature = "tls")]
    let tls_client_ca = p.tls_client_ca;
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        if handle_signals {
            let c = cancel.clone();
            tokio::spawn(async move {
                shutdown_signal().await;
                tracing::info!("shutdown signal received — draining");
                glossa::sdnotify::stopping();
                c.cancel();
            });
        }
        // SIGHUP: reload the log level from the control file and force a freshen, without a full
        // restart. Independent of the SIGTERM/Ctrl-C stream above — tokio allows multiple `signal()`
        // registrations for different signal kinds coexisting on the same runtime.
        #[cfg(unix)]
        if handle_signals {
            let hup_srv = server.clone();
            let hup_cancel = cancel.clone();
            let hup_path = server.state_dir().join(".glossa").join("loglevel");
            tokio::spawn(async move {
                let mut hup = match tokio::signal::unix::signal(
                    tokio::signal::unix::SignalKind::hangup(),
                ) {
                    Ok(s) => s,
                    Err(_) => return,
                };
                loop {
                    tokio::select! {
                        _ = hup_cancel.cancelled() => break,
                        got = hup.recv() => {
                            if got.is_none() { break; }
                            glossa::sdnotify::reloading(); // RELOADING=1 (no-op off systemd)
                            if let Some(d) = glossa::logreload::apply_from_file(&hup_path) {
                                tracing::info!("SIGHUP: log level reloaded: {d}");
                            }
                            tracing::warn!("SIGHUP does not reload bind/transport/state-dir/auth-token — restart for those");
                            #[cfg(feature = "tls")]
                            if let Some(r) = TLS_RELOADABLE.get() {
                                match r.reload() {
                                    Ok(()) => tracing::info!("SIGHUP: TLS cert/key reloaded"),
                                    Err(e) => tracing::warn!("SIGHUP: TLS reload failed, keeping the previous cert: {e:#}"),
                                }
                            }
                            tracing::info!("SIGHUP: forcing a freshen");
                            hup_srv.freshen_now().await;
                            glossa::sdnotify::ready(); // back to READY=1 after the reload
                        }
                    }
                }
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
                    insecure,
                    session_idle_secs,
                    cancel,
                    on_transport_ready,
                    #[cfg(feature = "tls")]
                    tls_cert,
                    #[cfg(feature = "tls")]
                    tls_key,
                    #[cfg(feature = "tls")]
                    tls_client_ca,
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
    insecure: Option<bool>,
    session_idle_secs: u64,
    cancel: tokio_util::sync::CancellationToken,
    on_transport_ready: Option<Box<dyn FnOnce() + Send>>,
    #[cfg(feature = "tls")] tls_cert: Option<PathBuf>,
    #[cfg(feature = "tls")] tls_key: Option<PathBuf>,
    #[cfg(feature = "tls")] tls_client_ca: Option<PathBuf>,
) -> anyhow::Result<()> {
    use rmcp::transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    };
    // Resolve Option<bool> → bool here; the `false` default is config::defaults' (Plan E), not clap's.
    let insecure = insecure.unwrap_or(false);
    // Real signal in a `tls` build: TLS is only "active" once both cert and key are configured
    // (matching the condition the serve step below uses to actually build the HTTPS listener).
    // Default (non-`tls`) build: always false, so the interlock's non-loopback+no-auth refusal
    // behaves exactly as before this task.
    #[cfg(feature = "tls")]
    let tls_active = tls_cert.is_some() && tls_key.is_some();
    #[cfg(not(feature = "tls"))]
    let tls_active = false;
    if let Some(msg) =
        glossa::serve_guard::interlock_refuses(bind, auth_token.is_some(), tls_active, insecure)
    {
        anyhow::bail!(msg);
    }
    if insecure && !glossa::serve_guard::is_loopback_bind(bind) && auth_token.is_none() {
        if tls_active {
            tracing::warn!(
                "--insecure: serving MCP on a non-loopback bind with no auth token (TLS is \
                 active, but token-less access is still weak unless a client certificate is \
                 required -- verify mTLS is enforced or set a token)"
            );
        } else {
            tracing::warn!("--insecure: serving MCP with NO authentication on a non-loopback bind");
        }
        glossa::audit::security_event("access", "insecure_serve", "override", "-", bind);
    }
    // Shutdown is driven by `cancel` (the caller wires the OS signal or the SCM control handler to it).
    let mut config = StreamableHttpServerConfig::default();
    config.cancellation_token = cancel.clone();
    if !allowed_hosts.is_empty() {
        config = config.with_allowed_hosts(allowed_hosts);
    }
    let ready_srv = server.clone();
    let metrics_srv = server.clone();
    let freshen_srv = server.clone();
    let loglevel_path = server.state_dir().join(".glossa").join("loglevel");
    let http = server.http_metrics();
    let service = StreamableHttpService::new(
        move || {
            // A fresh session must NOT share the anti-loop tracker with any other session — the
            // rest of `server`'s Arc fields (index caches, http metrics, etc.) stay shared by
            // design, only `signals` gets swapped for a brand-new tracker per session.
            let mut s = server.clone();
            s.signals = std::sync::Arc::new(parking_lot::Mutex::new(
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
    // Global request timeout + body-limit on /mcp only (/health, /ready, /metrics are registered
    // outside `mcp` and stay exempt). `req_timeout` MUST exceed the freshen serve-stale deadline
    // (Spec B D1) and legitimate slow-query time, or normal slow queries get cut mid-flight.
    let req_timeout = std::env::var("GLOSSA_MCP_REQUEST_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120u64);
    let max_body = std::env::var("GLOSSA_MCP_MAX_BODY_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4_000_000usize);
    mcp = mcp
        .layer(tower_http::timeout::TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            std::time::Duration::from_secs(req_timeout),
        ))
        .layer(tower_http::limit::RequestBodyLimitLayer::new(max_body));
    // Opt-in overload guards (§3c), all OFF unless their env var is set -- proxy deployments
    // (TLS/LB in front) delegate this to the proxy; these exist for native-TLS-without-proxy
    // deployments. The gating + layer construction lives in `apply_overload_guards` (pure,
    // env-free) so tests can exercise the REAL gating instead of a hand-rolled mirror of it.
    let max_concurrency = std::env::var("GLOSSA_MCP_MAX_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let rate_limit_per_sec = std::env::var("GLOSSA_MCP_RATE_LIMIT_PER_SEC")
        .ok()
        .and_then(|v| v.parse::<u32>().ok());
    if rate_limit_per_sec.is_some_and(|n| n > 0)
        && (std::env::var("GLOSSA_MCP_MAX_CONNECTIONS").is_ok() || tls_active)
    {
        tracing::warn!(
            "MCP overload guards: GLOSSA_MCP_RATE_LIMIT_PER_SEC is set together with a serving \
             path that can't supply axum's ConnectInfo -- GLOSSA_MCP_MAX_CONNECTIONS (the \
             connection-cap listener) and/or native TLS (serve_tls serves the plain `app` with \
             no `into_make_service_with_connect_info`) -- so per-IP rate-limiting degrades to a \
             single shared bucket for any /mcp request that doesn't carry a trusted \
             X-Forwarded-For/X-Real-Ip/Forwarded header (see conn_cap.rs and task-7-report.md). \
             Set a trusted proxy header if you need real per-IP limits in this configuration."
        );
    }
    mcp = apply_overload_guards(mcp, max_concurrency, rate_limit_per_sec);
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
    let scheme = if tls_active { "https" } else { "http" };
    tracing::info!(
        "glossa MCP (streamable-http) on {scheme}://{bind}/mcp  (+ /health /ready /metrics)"
    );
    glossa::sdnotify::ready(); // Type=notify: report READY after bind (R-C4), NOT after index warm-up
    if let Some(f) = on_transport_ready {
        f();
    }
    tokio::spawn(async move { freshen_srv.freshen_now().await });
    if let Some(usec) = glossa::sdnotify::watchdog_usec() {
        tracing::info!("systemd watchdog armed: pinging every {}us", usec / 2);
        glossa::sdnotify::spawn_watchdog(usec, cancel.clone());
    }
    glossa::logreload::spawn_poll(
        loglevel_path.clone(), // = <state-dir>/.glossa/loglevel, computed from `server` before it moves into the factory
        cancel.clone(),
    );
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
    // Listener-level connection cap (§3c, anti-slowloris), opt-in via GLOSSA_MCP_MAX_CONNECTIONS.
    // Parsed once here so both the native-TLS branch below (FU2: this cap now also applies over
    // TLS, nested inside `glossa::tls::serve_tls`) and the plaintext branch further down share the
    // SAME parsed knob.
    let max_connections = std::env::var("GLOSSA_MCP_MAX_CONNECTIONS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    // Native TLS (§3b, `tls` feature only): when cert+key are configured, terminate TLS in-process
    // instead of the plaintext/conn-cap paths below (a reverse-proxy in front stays the default
    // deployment for the plaintext build). Cert/key are re-read on SIGHUP (registered into
    // `TLS_RELOADABLE` for the HUP handler above) and on mtime change (`spawn_reload_poll`),
    // without dropping this listener -- see `glossa::tls` module docs.
    #[cfg(feature = "tls")]
    if tls_active {
        let files = glossa::tls::TlsFiles {
            cert: tls_cert.expect("tls_active implies tls_cert is set"),
            key: tls_key.expect("tls_active implies tls_key is set"),
            client_ca: tls_client_ca,
        };
        let mtls = files.client_ca.is_some();
        let reloadable = std::sync::Arc::new(glossa::tls::ReloadableTls::new(files)?);
        let _ = TLS_RELOADABLE.set(reloadable.clone());
        glossa::tls::spawn_reload_poll(reloadable.clone(), cancel.clone());
        let handshake_timeout = glossa::tls::handshake_timeout_from_env();
        let max_handshakes = glossa::tls::max_handshakes_from_env();
        tracing::info!(
            "MCP TLS: serving HTTPS{mtls_note} (reload on SIGHUP + cert-file mtime change, \
             {handshake_timeout:?} handshake timeout, max {max_handshakes} concurrent \
             handshakes{cap_note})",
            mtls_note = if mtls {
                " with client-certificate (mTLS) required"
            } else {
                ""
            },
            cap_note = match max_connections {
                Some(n) => format!(", max {n} concurrent established connections"),
                None => String::new(),
            }
        );
        glossa::tls::serve_tls(
            listener,
            app,
            reloadable,
            cancel,
            handshake_timeout,
            max_handshakes,
            max_connections,
        )
        .await?;
        tracing::info!("glossa MCP (streamable-http) stopped");
        return Ok(());
    }
    // Plaintext connection cap (§3c, anti-slowloris).
    //
    // Unset (the common case): serve exactly as before this task, PLUS peer-address extraction
    // (`into_make_service_with_connect_info`) so the per-IP rate-limit guard's
    // `SmartIpKeyExtractor` can fall back to the direct socket address when no
    // X-Forwarded-For/X-Real-Ip/Forwarded header is present. Harmless when that guard is also
    // off (GLOSSA_MCP_RATE_LIMIT_PER_SEC unset) -- nothing reads the extension.
    //
    // Set: serve through `CappedListener` instead. FLAGGED DEVIATION (see task-7-report.md and
    // the NOTE in conn_cap.rs): axum 0.8 only ships `Connected<IncomingStream<'_, L>>` for its
    // own listener types, and a bridge impl for a custom `Listener` wrapper isn't expressible
    // from outside axum's crate (orphan rule) -- so this branch does NOT wire connect-info. If
    // the rate-limit guard is ALSO enabled in this combination, its direct-socket fallback can't
    // fire (no ConnectInfo to read); a trusted proxy's forwarded-for/real-ip header still works
    // (`SmartIpKeyExtractor` checks those before ever falling back to the peer address), and
    // requests with neither degrade to `FailOpenIpKeyExtractor`'s global bucket (see its doc
    // comment) rather than being rejected outright -- the guard never fails closed.
    match max_connections {
        Some(max_conn) => {
            tracing::info!(
                "MCP overload guard: max concurrent connections = {max_conn} (excess waits at the listener)"
            );
            axum::serve(
                glossa::conn_cap::CappedListener::new(listener, max_conn),
                app,
            )
            .with_graceful_shutdown(async move { cancel.cancelled().await })
            .await?;
        }
        None => {
            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
            )
            .with_graceful_shutdown(async move { cancel.cancelled().await })
            .await?;
        }
    }
    tracing::info!("glossa MCP (streamable-http) stopped");
    Ok(())
}

/// Applies the opt-in overload guards (§3c) to `mcp`, given already-PARSED knob values -- no env
/// reads, no I/O. `serve_streamable_http` is the only real caller (it reads the env vars itself
/// and passes the parsed `Option`s in); tests call this directly with knob values of their
/// choosing, so they exercise the REAL `if let Some(..)` gating rather than a hand-rolled mirror
/// of it. Each guard's own comment (inline below) explains its layer placement/ordering.
fn apply_overload_guards(
    mut mcp: axum::Router,
    max_concurrency: Option<usize>,
    rate_limit_per_sec: Option<u32>,
) -> axum::Router {
    if let Some(n) = max_concurrency {
        tracing::info!("MCP overload guard: max in-flight /mcp = {n} (excess → 503)");
        // Built as ONE `tower::ServiceBuilder` stack and applied as a single `.layer()` call:
        // `Router::layer` type-checks each individual `.layer()` call's resulting error against
        // `Into<Infallible>` immediately, and `tower::load_shed`'s `Error` is a `BoxError` (not
        // `Infallible`) regardless of what it wraps -- so a bare `mcp.layer(LoadShed).layer(..)`
        // chain fails to compile one layer too early. Building the composite via `ServiceBuilder`
        // first, with `HandleErrorLayer` folding that `BoxError` back into a plain 503 response,
        // keeps the WHOLE unit's error type `Infallible` before it ever touches `mcp`.
        // `ServiceBuilder` order = declaration order = the order a request is seen (first added
        // = outermost): `HandleErrorLayer` outermost (catches errors from everything inside it),
        // then `LoadShedLayer` (watches `ConcurrencyLimit`'s readiness and, when at capacity,
        // sheds immediately instead of the caller queuing behind it -- queuing in front of a
        // heavy backend just deepens the latency tail), then `ConcurrencyLimitLayer` innermost.
        mcp = mcp.layer(
            tower::ServiceBuilder::new()
                .layer(axum::error_handling::HandleErrorLayer::new(
                    |_: tower::BoxError| async { axum::http::StatusCode::SERVICE_UNAVAILABLE },
                ))
                .layer(tower::load_shed::LoadShedLayer::new())
                .layer(tower::limit::ConcurrencyLimitLayer::new(n)),
        );
    }
    // Per-IP token-bucket rate-limit (opt-in, tower-governor): keeps one abusive/bursty client
    // from burning the concurrency-limit/load-shed budget that other clients need. Added AFTER
    // (thus outer to) the concurrency/load-shed guard above, so an over-quota IP gets 429 before
    // it ever touches the in-flight counter. `FailOpenIpKeyExtractor` (below) wraps
    // `SmartIpKeyExtractor`'s existing X-Forwarded-For/X-Real-Ip/Forwarded/ConnectInfo fallback
    // chain and only widens to a single global bucket as the very last resort -- see its doc
    // comment for why plain `SmartIpKeyExtractor` is unsafe to use directly here.
    if let Some(per_sec) = rate_limit_per_sec.filter(|n| *n > 0) {
        tracing::info!(
            "MCP overload guard: per-IP rate limit = {per_sec}/s on /mcp (excess → 429)"
        );
        let governor_conf = std::sync::Arc::new(
            tower_governor::governor::GovernorConfigBuilder::default()
                .key_extractor(FailOpenIpKeyExtractor)
                .per_nanosecond(1_000_000_000u64 / u64::from(per_sec))
                .burst_size(per_sec)
                .finish()
                .expect("non-zero per-second/burst-size always produce a GovernorConfig"),
        );
        mcp = mcp.layer(tower_governor::GovernorLayer::new(governor_conf));
    }
    mcp
}

/// Rate-limit bucket key: the caller's IP when derivable, else ONE shared global bucket.
///
/// `tower_governor`'s own extractors (`PeerIpKeyExtractor`, `SmartIpKeyExtractor`) return
/// `Err(GovernorError::UnableToExtractKey)` when nothing works, and `Governor` turns THAT into an
/// immediate rejection response for the request -- i.e. plugging one of them in directly does
/// NOT degrade gracefully when a key can't be derived, it takes `/mcp` down entirely for every
/// request in that state. That state is reachable in production: when the connection-cap guard
/// (`GLOSSA_MCP_MAX_CONNECTIONS`, `src/conn_cap.rs`) is also enabled, `serve_streamable_http`
/// serves through `CappedListener` without `into_make_service_with_connect_info` (axum's
/// `Connected` bridge for a custom `Listener` isn't expressible from outside axum's crate --
/// orphan rule), so `ConnectInfo` is never populated; a request with no trusted
/// X-Forwarded-For/X-Real-Ip/Forwarded header then has NO extractable key at all.
///
/// `FailOpenIpKeyExtractor` never fails closed: it delegates to `SmartIpKeyExtractor`'s full
/// fallback chain first (so the common cases -- a fronting proxy's header, or a direct socket
/// address when `ConnectInfo` IS available -- still get real per-IP limiting), and only widens
/// to `Global` when that returns `Err`. A `Global` bucket still bounds total `/mcp` load (just
/// not per-caller) -- it degrades the guard's precision, it never turns it into an outage.
#[derive(Clone, Hash, Eq, PartialEq, Debug)]
enum RateLimitKey {
    Ip(std::net::IpAddr),
    Global,
}

/// See [`RateLimitKey`] for why this exists instead of using `tower_governor`'s
/// `SmartIpKeyExtractor` directly.
#[derive(Clone, Copy, Debug)]
struct FailOpenIpKeyExtractor;

impl tower_governor::key_extractor::KeyExtractor for FailOpenIpKeyExtractor {
    type Key = RateLimitKey;

    fn extract<T>(
        &self,
        req: &axum::http::Request<T>,
    ) -> Result<Self::Key, tower_governor::errors::GovernorError> {
        // `KeyExtractor` (the trait being implemented here) is already in scope for calling its
        // own methods on other types within this `impl` block -- no extra `use` needed.
        match tower_governor::key_extractor::SmartIpKeyExtractor.extract(req) {
            Ok(ip) => Ok(RateLimitKey::Ip(ip)),
            Err(_) => Ok(RateLimitKey::Global),
        }
    }
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
    axum::extract::State(m): axum::extract::State<
        std::sync::Arc<glossa::http_metrics::HttpMetrics>,
    >,
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

/// env ?? file — the already-set process env (the operator's flag/env layer) wins; else the file value.
fn pick_env_file(env_key: &str, file: Option<String>) -> Option<String> {
    std::env::var(env_key).ok().or(file)
}

/// Export a merged value into the process env, only if something resolved (leaving the opt-in
/// overload knobs — `max_concurrency`/`rate_limit_per_sec`/`connection_cap` — genuinely unset, with
/// no spurious default, when neither env nor file supplies a value).
fn export_if_resolved(key: &str, merged: Option<String>) {
    if let Some(v) = merged {
        std::env::set_var(key, v);
    }
}

/// Fold the file's env-only `[retrieval]`/`[limits]` sections into the process env with correct
/// precedence (flag/env already-set wins; else the file value; else leave unset so each call site's
/// own built-in default applies). `[logging]` is handled separately, above, before the subscriber
/// installs.
fn export_env_overrides(c: &glossa::config::DeploymentConfig) {
    // [retrieval] — Spec B.
    export_if_resolved(
        "GLOSSA_READ_RETRIES",
        pick_env_file(
            "GLOSSA_READ_RETRIES",
            c.retrieval.read_retries.map(|n| n.to_string()),
        ),
    );
    export_if_resolved(
        "GLOSSA_READ_RETRY_BACKOFF_MS",
        pick_env_file(
            "GLOSSA_READ_RETRY_BACKOFF_MS",
            c.retrieval.read_retry_backoff_ms.map(|n| n.to_string()),
        ),
    );
    export_if_resolved(
        "GLOSSA_FRESHEN_DEADLINE_MS",
        pick_env_file(
            "GLOSSA_FRESHEN_DEADLINE_MS",
            c.retrieval.freshen_deadline_ms.map(|n| n.to_string()),
        ),
    );
    export_if_resolved(
        "GLOSSA_MIN_RESCAN_MS",
        pick_env_file(
            "GLOSSA_MIN_RESCAN_MS",
            c.retrieval.min_rescan_ms.map(|n| n.to_string()),
        ),
    );
    // [limits]/overload — Spec C (all env-only serve guards). Each is opt-in (no built-in default
    // here): leaving it unset when neither env nor file supplies a value is the correct behavior.
    export_if_resolved(
        "GLOSSA_MCP_REQUEST_TIMEOUT_SECS",
        pick_env_file(
            "GLOSSA_MCP_REQUEST_TIMEOUT_SECS",
            c.limits.request_timeout_secs.map(|n| n.to_string()),
        ),
    );
    export_if_resolved(
        "GLOSSA_MCP_MAX_BODY_BYTES",
        pick_env_file(
            "GLOSSA_MCP_MAX_BODY_BYTES",
            c.limits.max_body_bytes.map(|n| n.to_string()),
        ),
    );
    export_if_resolved(
        "GLOSSA_MCP_MAX_CONCURRENCY",
        pick_env_file(
            "GLOSSA_MCP_MAX_CONCURRENCY",
            c.limits.max_concurrency.map(|n| n.to_string()),
        ),
    );
    export_if_resolved(
        "GLOSSA_MCP_RATE_LIMIT_PER_SEC",
        pick_env_file(
            "GLOSSA_MCP_RATE_LIMIT_PER_SEC",
            c.limits.rate_limit_per_sec.map(|n| n.to_string()),
        ),
    );
    export_if_resolved(
        "GLOSSA_MCP_MAX_CONNECTIONS",
        pick_env_file(
            "GLOSSA_MCP_MAX_CONNECTIONS",
            c.limits.connection_cap.map(|n| n.to_string()),
        ),
    );
    // `max_handshakes` (round-1 fix C1, `tls` feature only) has a built-in default even when
    // unset everywhere, but it's still merged the same env-or-file way as its siblings above --
    // leaving it unset here simply means `glossa::tls::max_handshakes_from_env()` applies its own
    // default at the call site, same as any other unset env var.
    export_if_resolved(
        "GLOSSA_MCP_MAX_HANDSHAKES",
        pick_env_file(
            "GLOSSA_MCP_MAX_HANDSHAKES",
            c.limits.max_handshakes.map(|n| n.to_string()),
        ),
    );
}

fn main() -> anyhow::Result<()> {
    // A deployment config file (`--config` / `GLOSSA_CONFIG`) must be loaded before the tracing
    // subscriber installs below, because its `[logging]` section can affect that subscriber — but
    // `Cli::parse()` (which would normally give us the `--config` value) runs AFTER the subscriber
    // install (so parse errors are still logged). Resolve the flag by scanning raw argv instead.
    let argv: Vec<String> = std::env::args().collect();
    let deploy_cfg = match glossa::config::config_path(peek_config_flag(&argv)) {
        Some(p) => glossa::config::load(&p)?, // runs validate_static: secret rejection + tls gate
        None => glossa::config::DeploymentConfig::default(),
    };
    // Structured logs go to STDERR — stdout is the stdio JSON-RPC channel and must never carry logs.
    // Level via RUST_LOG (default `info`). `GLOSSA_LOG_FORMAT=json` emits one JSON object per line
    // (for a SIEM / log pipeline); anything else is the human-readable default. Best-effort init (a
    // second init in tests is a no-op). Read from the env directly — logging is set up before Cli
    // parsing so parse errors are still logged. [logging] is the one section that MUST resolve
    // before `Cli::parse()`; env still wins over the file (the same precedence as the rest of Plan
    // E), it's just inlined here because `pick`/`pick_opt` aren't reachable yet at this point.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        deploy_cfg
            .logging
            .level
            .as_deref()
            .and_then(|v| tracing_subscriber::EnvFilter::try_new(v).ok())
            .unwrap_or_else(|| {
                tracing_subscriber::EnvFilter::new("info,tantivy=warn,pdf_oxide=error")
            })
    });
    let json_logs = std::env::var("GLOSSA_LOG_FORMAT")
        .ok()
        .or_else(|| deploy_cfg.logging.format.clone())
        .map(|v| v.eq_ignore_ascii_case("json"))
        .unwrap_or(false);
    glossa::logreload::install(json_logs, filter);
    let Cli {
        root: root_flags,
        state_dir,
        config: _config, // already consumed via peek_config_flag above; kept for --help/validation
        cmd,
    } = Cli::parse();
    // [retrieval]/[limits] have NO parameter path — Spec B/C read them via std::env::var at their
    // call sites, all of which run later than this point, so exporting here (after Cli::parse,
    // unlike [logging]) is sufficient. Full precedence stays flag > env > file > default.
    export_env_overrides(&deploy_cfg);
    // Serving-only sections ([server]/[tls]/[limits]/[logging]) matter only to `kb mcp`. One role
    // config file must work for both provisioning (`kb index` and friends) and serving, so their
    // presence here is inert for every other subcommand — a debug note, never an error.
    if !matches!(cmd, Cmd::Mcp { .. }) && deploy_cfg.serving_sections_present() {
        tracing::debug!(
            "config: serving-only sections ([server]/[tls]/[limits]/[logging]) are ignored by this subcommand"
        );
    }
    match cmd {
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
            let rr = resolve_inputs(path, &root_flags, state_dir.clone())?;
            let pretty = match format {
                OutputFormat::Pretty => true,
                OutputFormat::Rg => false,
                OutputFormat::Auto => glossa::cli_fmt::stdout_is_tty(),
            };
            let mut rg_lines: Vec<String> = Vec::new();
            let mut display: Vec<glossa::cli_fmt::DisplayHit> = Vec::new();
            let mut records: Vec<(String, String)> = Vec::new();

            if !scan {
                glossa::index::store::ensure_fresh_at(&rr.roots, &rr.state_base)?; // file-first: pick up new/changed docs
                let idx =
                    glossa::index::store::DocIndex::open_or_create_at(&rr.roots, &rr.state_base)?;
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
                        file: glossa::cli_fmt::rel_file(&rr.root, &h.path),
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
                let chunks = collect_chunks(&rr.root, glob.as_deref(), !no_ignore)?;
                for h in search_chunks(&chunks, &re, limit) {
                    let p = h.doc_path.display().to_string();
                    rg_lines.push(format!("{}:{}:{}: {}", p, h.location, h.line, h.snippet));
                    display.push(glossa::cli_fmt::DisplayHit {
                        file: glossa::cli_fmt::rel_file(&rr.root, &p),
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
                let _ = glossa::cli_fmt::write_last_search(&rr.state_base, &records);
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
                let rr = resolve_inputs(None, &root_flags, state_dir.clone())?;
                let rec = glossa::cli_fmt::read_last_search(&rr.state_base)
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
                            glossa::index::store::DocIndex::open_or_create_at(
                                &rr.roots,
                                &rr.state_base,
                            )
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
            // Same 3-way corpus precedence as `Cmd::Mcp` (Task 6): --root/GLOSSA_ROOTS outrank the
            // file outright; the file's [corpus] is consulted only when neither supplied any roots.
            let (effective_roots, effective_state_dir) =
                merge_corpus(&root_flags, state_dir.clone(), &deploy_cfg);
            let rr = resolve_inputs(path, &effective_roots, effective_state_dir)?;
            let started = std::time::Instant::now();
            if let Some(rel) = file {
                let idx =
                    glossa::index::store::DocIndex::open_or_create_at(&rr.roots, &rr.state_base)?;
                let Some(rel) = idx.canonical_document_path(&rel) else {
                    anyhow::bail!("not an indexed document: {rel}");
                };
                let _lock = glossa::index::lock::try_index_lock(&rr.state_base)
                    .ok_or_else(|| anyhow::anyhow!("another process is indexing; try again"))?;
                glossa::index::store::index_one_file_locked_at(&rr.roots, &rr.state_base, &rel)?;
                println!(
                    "reindexed {rel} in {}",
                    glossa::cli_fmt::format_elapsed(started.elapsed())
                );
                return Ok(());
            }
            if let Some(name) = ontology {
                // The preset always materializes under the STATE base (`<state_base>/.glossa/ontology.toml`),
                // never a corpus root — state-dir separation means these can differ.
                match glossa::ontology_templates::write_template(&rr.state_base, &name, false)? {
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
            if let Some(p) = glossa::default_ignore::seed_if_absent(&rr.root) {
                eprintln!(
                    "wrote default {} (whitelist of supported types) — edit it to tune what's indexed",
                    p.display()
                );
            }
            let stats = glossa::index::store::index_dir_at(&rr.roots, &rr.state_base, force)?;
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
                let g = glossa::graph::store::GraphStore::open(&rr.state_base)?;
                let ont = glossa::graph::ontology::Ontology::load_or_default(&rr.state_base);
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
            let rr = resolve_inputs(path, &root_flags, state_dir.clone())?;
            let root = rr.state_base.clone();
            let orphans = glossa::index::store::orphan_notes_at(&rr.roots, &root)?;
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
                glossa::index::store::ensure_fresh_at(&rr.roots, &root)?;
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
            let rr = resolve_inputs(path, &root_flags, state_dir.clone())?;
            glossa::index::store::ensure_fresh_at(&rr.roots, &rr.state_base)?; // file-first: pick up new/changed docs
            let idx = glossa::index::store::DocIndex::open_or_create_at(&rr.roots, &rr.state_base)?;
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
            let rr = resolve_inputs(path, &root_flags, state_dir.clone())?;
            glossa::index::store::ensure_fresh_at(&rr.roots, &rr.state_base)?; // file-first: pick up new/changed docs
            let idx = glossa::index::store::DocIndex::open_or_create_at(&rr.roots, &rr.state_base)?;
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
            insecure,
            session_idle_secs,
            #[cfg(feature = "tls")]
            tls_cert,
            #[cfg(feature = "tls")]
            tls_key,
            #[cfg(feature = "tls")]
            tls_client_ca,
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
                let c = &deploy_cfg;
                // Corpus (Spec A): fall through to the file's roots/state_dir ONLY when neither
                // --root nor GLOSSA_ROOTS is present (see `merge_corpus`).
                let (effective_roots, effective_state_dir) =
                    merge_corpus(&root_flags, state_dir.clone(), c);
                let rr = resolve_inputs_traced(path, &effective_roots, effective_state_dir)?;
                // Server (Spec C): each setting is a clap Option<T> that already collapsed flag+env
                // via `env=`, so a plain two-tier pick/merge_list is correct here (no
                // GLOSSA_ROOTS-style 3-way trap).
                let bind = glossa::config::pick(
                    bind,
                    c.server.bind.clone(),
                    glossa::config::defaults::BIND.to_string(),
                );
                let transport = match glossa::config::pick_opt(
                    transport,
                    c.server
                        .transport
                        .as_deref()
                        .map(|s| <McpTransport as clap::ValueEnum>::from_str(s, true))
                        .transpose()
                        .map_err(anyhow::Error::msg)?,
                ) {
                    Some(t) => t,
                    None => McpTransport::Stdio, // = config::defaults::TRANSPORT
                };
                let session_idle_secs = glossa::config::pick(
                    session_idle_secs,
                    c.server.session_idle_secs,
                    glossa::config::defaults::SESSION_IDLE_SECS,
                );
                let allowed_hosts =
                    glossa::config::merge_list(allowed_hosts, c.server.allowed_hosts.clone());
                // stays Option<bool>; .unwrap_or(false) happens at the serve_streamable_http call
                // site (src/main.rs), unchanged.
                let insecure = glossa::config::pick_opt(insecure, c.server.insecure);
                #[cfg(feature = "tls")]
                let tls_cert =
                    glossa::config::pick_opt(tls_cert, c.tls.as_ref().and_then(|t| t.cert.clone()));
                #[cfg(feature = "tls")]
                let tls_key =
                    glossa::config::pick_opt(tls_key, c.tls.as_ref().and_then(|t| t.key.clone()));
                #[cfg(feature = "tls")]
                let tls_client_ca = glossa::config::pick_opt(
                    tls_client_ca,
                    c.tls.as_ref().and_then(|t| t.client_ca.clone()),
                );
                // auth_token is env/flag ONLY (never c.server.*) — validate_static already rejected
                // a token key in the file at load time, so there is nothing to merge here.
                let params = ServeParams {
                    roots: rr.roots,
                    state_base: rr.state_base,
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
                    insecure,
                    session_idle_secs,
                    #[cfg(feature = "tls")]
                    tls_cert,
                    #[cfg(feature = "tls")]
                    tls_key,
                    #[cfg(feature = "tls")]
                    tls_client_ca,
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
                let rr = resolve_inputs(path, &root_flags, state_dir.clone())?;
                let g = glossa::graph::store::GraphStore::open(&rr.state_base)?;
                println!("{}", glossa::tools::graph_stats(&g));
                Ok(())
            }
            GraphAction::Glossary {
                query,
                path,
                as_of,
                scope,
            } => {
                let rr = resolve_inputs(path, &root_flags, state_dir.clone())?;
                glossa::index::store::ensure_fresh_at(&rr.roots, &rr.state_base)?; // file-first: pick up new/changed docs
                let idx =
                    glossa::index::store::DocIndex::open_or_create_at(&rr.roots, &rr.state_base)?;
                let g = glossa::graph::store::GraphStore::open(&rr.state_base)?;
                let trace = glossa::trace::TraceLog::disabled();
                let spec = glossa::tools::ChainSpec::from_ontology(
                    &glossa::graph::ontology::Ontology::load_or_default(&rr.state_base),
                );
                let stale = glossa::tools::StaleChecker::new(rr.roots.clone());
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
                let rr = resolve_inputs(path, &root_flags, state_dir.clone())?;
                glossa::index::store::ensure_fresh_at(&rr.roots, &rr.state_base)?;
                let idx =
                    glossa::index::store::DocIndex::open_or_create_at(&rr.roots, &rr.state_base)?;
                let g = glossa::graph::store::GraphStore::open(&rr.state_base)?;
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
                let rr = resolve_inputs(path, &root_flags, state_dir.clone())?;
                let g = glossa::graph::store::GraphStore::open(&rr.state_base)?;
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
                let rr = resolve_inputs(path, &root_flags, state_dir.clone())?;
                let g = glossa::graph::store::GraphStore::open(&rr.state_base)?;
                let ont = glossa::graph::ontology::Ontology::load_or_default(&rr.state_base);
                let mut opts = glossa::graph::generalize::apply::Opts::from_ontology(
                    &ont,
                    glossa::trace::now_ms(),
                );
                opts.apply_merges = merge;
                let r = glossa::graph::generalize::apply::generalize(&g, &opts)?;
                println!(
                    "generalize: inferred_edges={} similar_edges={} communities={} \
                     merge_candidates={} merged_nodes={}",
                    r.inferred_edges,
                    r.similar_edges,
                    r.communities,
                    r.merge_candidates,
                    r.merged_nodes,
                );
                Ok(())
            }
            GraphAction::Doctor {
                path,
                prune_incomplete,
                prune_ungrounded,
                mut prune_dangling,
                prune_stale,
                force,
            } => {
                let rr = resolve_inputs(path, &root_flags, state_dir.clone())?;
                let g = glossa::graph::store::GraphStore::open(&rr.state_base)?;
                let ont = glossa::graph::ontology::Ontology::load_or_default(&rr.state_base);
                let report = glossa::graph::doctor::doctor(&g, &ont, &rr.roots)?;
                print!("{}", glossa::graph::ops::fmt_doctor_report(&report));
                if prune_dangling && !force {
                    if let Some(reason) =
                        glossa::graph::doctor::dangling_prune_risk(&report, &g, &ont)
                    {
                        prune_dangling = false;
                        println!(
                            "dangling prune REFUSED: {reason}\nre-run with --force to override."
                        );
                    }
                }
                if prune_incomplete || prune_ungrounded || prune_dangling || prune_stale {
                    let (inc, ung, dang, stale) = glossa::graph::doctor::prune(
                        &g,
                        &report,
                        &glossa::graph::doctor::PruneOpts {
                            incomplete: prune_incomplete,
                            ungrounded: prune_ungrounded,
                            dangling: prune_dangling,
                            stale: prune_stale,
                        },
                    )?;
                    println!(
                        "pruned: incomplete={inc} ungrounded={ung} dangling={dang} stale={stale}"
                    );
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
                let rr = resolve_inputs(path, &root_flags, state_dir.clone())?;
                let g = glossa::graph::store::GraphStore::open(&rr.state_base)?;
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
            GraphAction::Node {
                node_id,
                path,
                as_of,
                now,
            } => {
                let rr = resolve_inputs(path, &root_flags, state_dir.clone())?;
                let g = glossa::graph::store::GraphStore::open(&rr.state_base)?;
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
                                (None, None) => {
                                    epoch_to_rfc3339((glossa::trace::now_ms() / 1000) as i64)
                                }
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
                let rr = resolve_inputs(path, &root_flags, state_dir.clone())?;
                glossa::index::store::ensure_fresh_at(&rr.roots, &rr.state_base)?;
                let idx =
                    glossa::index::store::DocIndex::open_or_create_at(&rr.roots, &rr.state_base)?;
                let g = glossa::graph::store::GraphStore::open(&rr.state_base)?;
                let ont = glossa::graph::ontology::Ontology::load_or_default(&rr.state_base);
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
                let rr = resolve_inputs(path, &root_flags, state_dir.clone())?;
                let path = rr.state_base;
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
                                let from_ok =
                                    visible_ids.contains(&e.from) || g.visible_at(&e.from, a)?;
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
            GraphAction::Import {
                file,
                path,
                format,
                mode,
            } => {
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
            GraphAction::Prune {
                path,
                node_type,
                source,
                dry_run,
            } => {
                let g = glossa::graph::store::GraphStore::open(&path)?;
                let mut total = 0;
                for t in &node_type {
                    match &source {
                        // Source-scoped: delete only the nodes of this type grounded in a matching doc.
                        Some(src) => {
                            let ids = g.ids_of_type_grounded_in(t, src)?;
                            if dry_run {
                                println!(
                                    "graph prune (dry-run): {} {t} grounded in *{src}* would be removed:",
                                    ids.len()
                                );
                                for id in &ids {
                                    let label =
                                        g.get_node(id)?.map(|n| n.label).unwrap_or_default();
                                    println!("  {id}  {label}");
                                }
                            } else {
                                let n = g.delete_nodes(&ids)?;
                                println!(
                                    "graph prune: removed {n} entries ({} {t} nodes grounded in *{src}*)",
                                    ids.len()
                                );
                                total += n;
                            }
                        }
                        // Whole-type wipe (original behavior).
                        None => {
                            if dry_run {
                                let ids = g.ids_of_type(t)?;
                                println!(
                                    "graph prune (dry-run): all {} {t} nodes would be removed",
                                    ids.len()
                                );
                            } else {
                                let n = g.delete_by_type(t)?;
                                println!("graph prune: removed {n} entries of type {t}");
                                total += n;
                            }
                        }
                    }
                }
                if !dry_run {
                    println!("graph prune: {total} total entries removed");
                }
                Ok(())
            }
            #[cfg(feature = "constraint")]
            GraphAction::Build {
                path,
                doc,
                tables_dir,
            } => {
                let rr = resolve_inputs(path, &root_flags, state_dir.clone())?;
                glossa::index::store::ensure_fresh_at(&rr.roots, &rr.state_base)?;
                let tables = tables_dir.unwrap_or_else(|| {
                    glossa::notebook::notes_root(&rr.state_base)
                        .join(glossa::notebook::mirror_dir_for_doc(&doc))
                });
                let idx =
                    glossa::index::store::DocIndex::open_or_create_at(&rr.roots, &rr.state_base)?;
                let g = glossa::graph::store::GraphStore::open(&rr.state_base)?;
                let ont = glossa::graph::ontology::Ontology::load_or_default(&rr.state_base);
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
                    cat.sort_by(|a, b| {
                        a.tier
                            .cmp(&b.tier)
                            .then(a.family.cmp(&b.family))
                            .then(a.name.cmp(&b.name))
                    });
                    for t in cat {
                        if let Some(f) = &family {
                            if t.family.as_deref() != Some(f.as_str()) {
                                continue;
                            }
                        }
                        if let Some(n) = tier {
                            if t.tier != n {
                                continue;
                            }
                        }
                        let desc = t.description.as_deref().unwrap_or("");
                        let fam = t.family.as_deref().unwrap_or("-");
                        println!("[tier {}] {:<18} {:<12} {}", t.tier, t.name, fam, desc);
                    }
                    Ok(())
                }
                OntologyAction::Show { name } => {
                    let canon = ot::resolve(&name).ok_or_else(|| {
                        anyhow::anyhow!(
                            "unknown preset '{name}' — did you mean: {}? (kb ontology list)",
                            ot::nearest(&name, 3).join(", ")
                        )
                    })?;
                    print!("{}", ot::raw(&canon).unwrap());
                    Ok(())
                }
                OntologyAction::Init {
                    path,
                    template,
                    force,
                } => {
                    // The preset always materializes under the STATE base — see the matching note
                    // at `kb index --ontology` above.
                    let rr = resolve_inputs(path, &root_flags, state_dir.clone())?;
                    match ot::write_template(&rr.state_base, &template, force)? {
                        ot::Written::Created => {
                            println!("wrote '{template}' to .glossa/ontology.toml")
                        }
                        ot::Written::Overwritten => {
                            println!("overwrote .glossa/ontology.toml with '{template}'")
                        }
                        ot::Written::Kept => anyhow::bail!(
                            ".glossa/ontology.toml already exists — pass --force to replace it"
                        ),
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

#[cfg(test)]
mod cli_root_wiring_tests {
    use super::*;

    #[test]
    fn glossa_roots_env_and_flag_merge_flag_wins() {
        // flag > env: a --root flag present suppresses GLOSSA_ROOTS entirely (precedence contract).
        let inputs = build_root_inputs(
            None,
            &["docs=/mnt/a".to_string()],
            Some("specs=/mnt/b".to_string()),
            None,
        )
        .unwrap();
        assert_eq!(inputs.roots.len(), 1);
        assert_eq!(inputs.roots[0].label, "docs");
    }

    #[test]
    fn glossa_roots_env_used_when_no_flag_given() {
        let inputs = build_root_inputs(
            None,
            &[],
            Some("specs=/mnt/b\n\ndocs=/mnt/a".to_string()),
            None,
        )
        .unwrap();
        assert_eq!(inputs.roots.len(), 2, "blank env lines are skipped");
        assert_eq!(inputs.roots[0].label, "specs");
        assert_eq!(inputs.roots[1].label, "docs");
    }

    #[test]
    fn state_dir_without_path_errors_through_resolve_inputs() {
        let state = tempfile::tempdir().unwrap();
        let err = resolve_inputs(None, &[], Some(state.path().to_path_buf())).unwrap_err();
        assert!(err.to_string().contains("corpus path"), "got: {err}");
    }
}

/// Layer-wiring checks for the `/mcp` hardening layers (§3c): a minimal router built with the same
/// `tower_http` layers `serve_streamable_http` applies to its `mcp` sub-router, driven with
/// `tower::ServiceExt::oneshot` (no socket, no full server harness). `/health`/`/ready`/`/metrics`
/// are registered on `observed`/`app` OUTSIDE the layered `mcp` router in `serve_streamable_http`
/// (see the `.merge(mcp)` call), so they structurally never see these layers — that exemption is
/// enforced by router topology, not tested again here.
#[cfg(test)]
mod mcp_serve_layer_tests {
    use super::apply_overload_guards;
    use tower::{Service, ServiceExt};

    #[tokio::test]
    async fn mcp_body_limit_rejects_oversized_payload() {
        // `RequestBodyLimitLayer` rejects eagerly off the `Content-Length` header (see
        // tower_http::limit::request_body::RequestBodyLimit::call); a real HTTP/1 server fills
        // that header in from the wire, so it must be set explicitly when driving the service
        // in-process via `oneshot` (no socket, so nothing parses a wire request for us).
        let app = axum::Router::new()
            .route("/mcp", axum::routing::post(|| async { "ok" }))
            .layer(tower_http::limit::RequestBodyLimitLayer::new(8));
        let req = axum::http::Request::post("/mcp")
            .header(axum::http::header::CONTENT_LENGTH, "64")
            .body(axum::body::Body::from(vec![b'x'; 64]))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn mcp_body_limit_allows_payload_within_limit() {
        let app = axum::Router::new()
            .route("/mcp", axum::routing::post(|| async { "ok" }))
            .layer(tower_http::limit::RequestBodyLimitLayer::new(64));
        let req = axum::http::Request::post("/mcp")
            .body(axum::body::Body::from(vec![b'x'; 8]))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
    }

    #[tokio::test]
    async fn mcp_request_timeout_returns_408_for_slow_handler() {
        let app = axum::Router::new()
            .route(
                "/mcp",
                axum::routing::post(|| async {
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                    "ok"
                }),
            )
            .layer(tower_http::timeout::TimeoutLayer::with_status_code(
                axum::http::StatusCode::REQUEST_TIMEOUT,
                std::time::Duration::from_millis(20),
            ));
        let req = axum::http::Request::post("/mcp")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::REQUEST_TIMEOUT);
    }

    #[tokio::test]
    async fn mcp_request_timeout_allows_fast_handler() {
        let app = axum::Router::new()
            .route("/mcp", axum::routing::post(|| async { "ok" }))
            .layer(tower_http::timeout::TimeoutLayer::with_status_code(
                axum::http::StatusCode::REQUEST_TIMEOUT,
                std::time::Duration::from_secs(5),
            ));
        let req = axum::http::Request::post("/mcp")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
    }

    #[tokio::test]
    async fn mcp_concurrency_limit_and_load_shed_sheds_excess_request() {
        // Per the brief: an axum-Router `.oneshot()` per call can't reliably reproduce real
        // concurrent contention on shared middleware state (each in-process oneshot drives its
        // own call to completion independently) -- so this drives the SAME
        // `tower::ServiceBuilder` composite `serve_streamable_http` applies when
        // GLOSSA_MCP_MAX_CONCURRENCY is set directly via `tower::Service`, no axum Router/HTTP
        // involved. `ServiceBuilder`'s declaration order = request order (first added =
        // outermost): `LoadShedLayer` outer (watches `ConcurrencyLimitLayer`'s readiness and, at
        // capacity, sheds immediately instead of the caller queuing behind it), then
        // `ConcurrencyLimitLayer(1)` inner. Both layers implement `Clone` by sharing their inner
        // `Arc` state (the semaphore), so `svc.clone()` below is the SAME limiter, not a fresh
        // one -- exactly like two real concurrent requests hitting one running server.
        let slow = tower::service_fn(|_req: ()| async {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            Ok::<(), std::convert::Infallible>(())
        });
        let svc = tower::ServiceBuilder::new()
            .layer(tower::load_shed::LoadShedLayer::new())
            .layer(tower::limit::ConcurrencyLimitLayer::new(1))
            .service(slow);
        let mut svc1 = svc.clone();
        let mut svc2 = svc;
        let call1 = async {
            tower::ServiceExt::<()>::ready(&mut svc1).await.unwrap();
            svc1.call(()).await
        };
        let call2 = async {
            tower::ServiceExt::<()>::ready(&mut svc2).await.unwrap();
            svc2.call(()).await
        };
        let (r1, r2): (Result<(), tower::BoxError>, Result<(), tower::BoxError>) =
            tokio::join!(call1, call2);
        let results = [&r1, &r2];
        assert_eq!(
            results.iter().filter(|r| r.is_ok()).count(),
            1,
            "exactly one of the two concurrent calls should succeed, got {r1:?} / {r2:?}"
        );
        assert_eq!(
            results.iter().filter(|r| r.is_err()).count(),
            1,
            "exactly one of the two concurrent calls should be shed as Overloaded, got {r1:?} / {r2:?}"
        );
    }

    // The four tests below call `apply_overload_guards` directly -- the SAME function
    // `serve_streamable_http` calls with its parsed env values -- so they exercise the real
    // `if let Some(..)` gating for each knob, not a hand-rolled mirror of it (a prior version of
    // this test built an unrelated bare router by hand, which passed even if the gating in
    // `apply_overload_guards` were deleted entirely).
    //
    // These use SINGLE, sequential requests rather than concurrent ones on purpose: an
    // axum-Router `.oneshot()` clone driven concurrently via `tokio::join!` was already shown
    // (in the concurrency-limit test above, and in an earlier draft of these tests) to be an
    // unreliable way to reproduce real contention on shared middleware state through a full
    // Router -- exactly what the brief warns "oneshot is insufficient for concurrency" about.
    // `GLOSSA_MCP_MAX_CONCURRENCY(0)` sheds a SINGLE request deterministically (a limiter with
    // zero capacity is never ready, no contention needed), and the rate-limiter's admit/reject
    // decision is time-based, not concurrency-based, so two sequential calls suffice there too.
    fn mk_app() -> axum::Router {
        axum::Router::new().route("/mcp", axum::routing::post(|| async { "ok" }))
    }

    fn mk_req() -> axum::http::Request<axum::body::Body> {
        axum::http::Request::post("/mcp")
            .body(axum::body::Body::empty())
            .unwrap()
    }

    #[tokio::test]
    async fn apply_overload_guards_is_a_no_op_when_all_knobs_are_none() {
        let app = apply_overload_guards(mk_app(), None, None);
        let resp = app.oneshot(mk_req()).await.unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
    }

    #[tokio::test]
    async fn apply_overload_guards_adds_a_real_concurrency_shed_when_max_concurrency_is_set() {
        // A limiter built with capacity 0 is NEVER ready, so even a single, non-concurrent
        // request is shed -- this deterministically proves the `if let Some(n) =
        // max_concurrency` branch actually attached the ConcurrencyLimit+LoadShed layer (vs.
        // the `None` case above, which passes the very same request straight through as 200).
        let app = apply_overload_guards(mk_app(), Some(0), None);
        let resp = app.oneshot(mk_req()).await.unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn apply_overload_guards_adds_a_real_rate_limit_when_rate_limit_per_sec_is_set() {
        // burst_size == 1 means the SECOND request within the same ~second is over quota. Both
        // requests carry no forwarded-for header and no ConnectInfo extension (this test drives
        // the router directly with `.oneshot()`, not through a real listener), so both land on
        // `FailOpenIpKeyExtractor`'s `Global` bucket -- proving the `if let Some(per_sec) =
        // rate_limit_per_sec` branch attached a real, enforcing rate-limit layer (not just that
        // it compiles): first request admitted, second rejected, both against the SAME shared
        // limiter state via `app.clone()`.
        let app = apply_overload_guards(mk_app(), None, Some(1));
        let first = app.clone().oneshot(mk_req()).await.unwrap();
        let second = app.oneshot(mk_req()).await.unwrap();
        assert_eq!(first.status(), axum::http::StatusCode::OK);
        assert_eq!(second.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn rate_limit_guard_degrades_to_global_bucket_instead_of_failing_closed() {
        // Reproduces the exact "both knobs on" production scenario for the KEY-EXTRACTION path
        // specifically: `GLOSSA_MCP_MAX_CONNECTIONS` set means `serve_streamable_http` serves
        // WITHOUT `into_make_service_with_connect_info` (see conn_cap.rs's NOTE), so a request
        // with no trusted forwarded-for/real-ip/forwarded header has no `ConnectInfo` to fall
        // back to either -- exactly what this test recreates by never wiring ConnectInfo at all.
        // Before the fix, plugging `SmartIpKeyExtractor` straight into the rate-limit guard made
        // THIS request fail with `GovernorError::UnableToExtractKey` on the very first call --
        // i.e. every /mcp request in this combination, an outage, not a degraded guard.
        // `FailOpenIpKeyExtractor` must never do that: the first (and only) request here has to
        // succeed normally.
        let app = apply_overload_guards(mk_app(), None, Some(5));
        let resp = app.oneshot(mk_req()).await.unwrap();
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::OK,
            "a request with no forwarded header and no ConnectInfo must still be served (global \
             bucket), never rejected merely because a per-IP key couldn't be derived"
        );
    }
}
