//! Shared `service {install|uninstall|start|stop|status}` CLI shape, reused by `kb` and `kbi`.
//!
//! The install options mirror `kb mcp` one-for-one: a command line that works interactively turns
//! into a service command line unchanged. That is a deliberate reversal of the original design,
//! which curated the set down to what "can only describe a valid service" — curation kept being
//! indistinguishable from omission, and the omissions were silent. `--source-file` was missing, so
//! `get_source_file` could not be enabled for a service by any route at all (the tool-surface
//! switches have no deployment-config form to fall back on), and nothing said so; the operator just
//! got "unexpected argument".
//!
//! Parity is about the ANSWER, not about obeying everything. `--transport stdio` is a foot-gun and
//! is honoured loudly: it installs a service with no client and no stdin under the SCM/systemd, and
//! the only thing it costs is the operator's own time. `--auth-token` is parsed and then refused,
//! because honouring it means writing the secret into a command line every local account can read
//! (`ps`; `systemctl cat` renders the unit world-readable) for the life of the service — a cost
//! paid by everyone, not just whoever typed it. It stays in the parser so that converting a
//! `kb mcp` command line explains itself and names `--auth-token-file`, instead of failing with
//! "unexpected argument".

use crate::serve_guard::read_token_file as service_guard_read;
use crate::service::{self, ServiceSpec};
use clap::{Args, Subcommand};
use std::path::{Path, PathBuf};

/// Just a service name — the key for uninstall/start/stop/status. Reused across binaries.
#[derive(Args, Debug)]
pub struct NameArg {
    /// The service name (as given to `install --service-name`).
    pub service_name: String,
}

/// `kb service install` options: the same surface as `kb mcp`, so a working interactive command
/// line converts to a service unchanged. Defaults still describe a valid service — transport is
/// `streamable-http` unless named otherwise — but nothing is withheld; see the module docs for the
/// two options that state their cost before obeying.
#[derive(Args, Debug)]
pub struct InstallOpts {
    /// Unique service name (the SCM/systemd key). Repeat `install` with distinct names + ports to
    /// run several corpora from one `kb` executable.
    #[arg(long = "service-name")]
    pub service_name: String,
    /// Corpus directory to serve (single-root shorthand). Optional: use the global `--root
    /// label=path` (repeatable) instead for a multi-root service. Exactly one of the two is required.
    pub corpus: Option<PathBuf>,
    /// Tool profile: reader | editor | full.
    #[arg(short = 'p', long, default_value = "editor")]
    pub profile: String,
    /// Bind address for the streamable-http endpoint (each service needs a distinct port).
    #[arg(long)]
    pub bind: String,
    /// Allowed `Host` header value(s) (DNS-rebind guard); repeatable. Loopback is always allowed.
    #[arg(long = "allowed-host")]
    pub allowed_host: Vec<String>,
    /// Enable image output in the `read` tool (embedded figures + page images).
    #[arg(long)]
    pub vision: bool,
    /// Enable the retrieval anti-loop dedup (repeat/streak/plateau markers). Off by default; a guard
    /// for a weak reasoning reader — leave off for a general/display client.
    #[arg(long)]
    pub dedup: bool,
    /// Expose the `get_source_file` tool (hands a client the original file, not extracted text).
    /// Mirrors `kb mcp --source-file`.
    #[arg(long = "source-file")]
    pub source_file: bool,
    /// Serve without the graph tools. Mirrors `kb mcp --no-graph`.
    #[arg(long = "no-graph")]
    pub no_graph: bool,
    /// Log every tool call to `<root>/.glossa/traces/*.jsonl`. Mirrors `kb mcp --trace`.
    #[arg(short = 't', long)]
    pub trace: bool,
    /// Drop an MCP session after this many idle seconds; `0` disables the timeout. Mirrors
    /// `kb mcp --session-idle-secs` (and `[server].session_idle_secs`).
    #[arg(long = "session-idle-secs")]
    pub session_idle_secs: Option<u64>,
    /// Override the startup interlock that refuses a non-loopback bind with no authentication.
    /// Mirrors `kb mcp --insecure` (and `[server].insecure`): the server still warns loudly and
    /// emits an audit event — this only decides whether it refuses to start.
    #[arg(long = "insecure")]
    pub insecure: Option<bool>,
    /// Deprecated no-op, accepted so a working `kb mcp` command line can be turned into a service
    /// command line unchanged. Images are off unless `--vision` is given.
    #[arg(short = 'N', long = "noimage", hide = true)]
    pub no_image: bool,
    /// Transport for the service process: `streamable-http` (the default and the only one a
    /// service can actually run) or `stdio`.
    ///
    /// COST, stated plainly: a service has no client and no stdin under the SCM or systemd, so
    /// `stdio` installs a service that starts, finds nothing to talk to, and stops. The flag is
    /// here for command-line parity with `kb mcp`; choosing `stdio` is a deliberate act and
    /// `install` says so out loud rather than refusing.
    #[arg(long = "transport")]
    pub transport: Option<String>,
    /// Accepted for parity with `kb mcp`, and then REFUSED with an explanation: a service's
    /// command line is public (`ps`; `systemctl cat` renders the unit world-readable), so there is
    /// no way to honour a token VALUE here that is not a disclosure to every local account. Use
    /// `--auth-token-file`, which bakes only the path. The flag exists so that converting a working
    /// `kb mcp` command line says why, instead of "unexpected argument".
    #[arg(long = "auth-token")]
    pub auth_token: Option<String>,
    /// Path to a file holding the bearer token for this service's `/mcp` endpoint.
    ///
    /// The token is NOT taken as a value here on purpose: a service's command line is public (`ps`,
    /// and a systemd unit is world-readable via `systemctl cat`), so a secret baked into `ExecStart`
    /// leaks to every local user. Only this path is baked in; the file carries the secret, and
    /// `kb` refuses to start if it is readable by group or other. Mirrors `kb mcp --auth-token-file`.
    #[arg(long = "auth-token-file")]
    pub auth_token_file: Option<PathBuf>,
    /// PEM certificate chain: serve the streamable-http endpoint over TLS directly, instead of
    /// terminating it at a reverse proxy. Set together with `--tls-key`.
    ///
    /// Only the PATHS are baked into the service command line — the same reasoning as
    /// `--auth-token-file`, and the key never leaves its file.
    #[cfg(feature = "tls")]
    #[arg(long = "tls-cert", requires = "tls_key")]
    pub tls_cert: Option<PathBuf>,
    /// PEM private key matching `--tls-cert`.
    #[cfg(feature = "tls")]
    #[arg(long = "tls-key", requires = "tls_cert")]
    pub tls_key: Option<PathBuf>,
    /// PEM client-CA bundle: require and verify client certificates (mTLS). Needs `--tls-cert`.
    #[cfg(feature = "tls")]
    #[arg(long = "tls-client-ca", requires = "tls_cert")]
    pub tls_client_ca: Option<PathBuf>,
}

/// `kb service <action>`: install a streamable-http MCP service, or manage one by name.
// Parsed once per process and consumed immediately; `Install` is wide because it carries the whole
// service description, and boxing it would mean hand-writing what `Subcommand` derives.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand, Debug)]
pub enum ServiceAction {
    /// Install (register) a kb MCP service (streamable-http unless `--transport` says otherwise).
    Install(InstallOpts),
    /// Remove a service.
    Uninstall(NameArg),
    /// Start a service.
    Start(NameArg),
    /// Stop a service.
    Stop(NameArg),
    /// Print a service's status.
    Status(NameArg),
}

/// Build the [`ServiceSpec`] for a kb MCP service: transport is `streamable-http` unless the
/// operator named another, plus the
/// profile/bind/allowed-host/vision/dedup options and the `--windows-service`/`--service-name` flags
/// the SCM/systemd-launched process routes on. `program` is the `kb` executable path.
///
/// `roots`/`state_dir`/`config` are the operator's already-parsed GLOBAL `kb` flags (`--root` /
/// `--state-dir` / `--config`) — baked into the service's `kb mcp` command so a service is
/// configured exactly like a direct `kb mcp` invocation. The corpus is supplied by EITHER the
/// positional `opts.corpus` (single-root shorthand) OR one-or-more `roots` (`--root` outranks the
/// positional, matching `kb mcp`); exactly one is required.
pub fn kb_service_spec(
    name: &str,
    program: PathBuf,
    opts: &InstallOpts,
    roots: &[String],
    state_dir: Option<&Path>,
    config: Option<&Path>,
) -> anyhow::Result<ServiceSpec> {
    let mut args = vec!["mcp".to_string()];
    // Corpus: --root entries win over the positional shorthand (mirrors `kb mcp`'s merge_corpus).
    if !roots.is_empty() {
        for r in roots {
            args.push("--root".to_string());
            args.push(r.clone());
        }
    } else if let Some(c) = &opts.corpus {
        args.push(c.to_string_lossy().into_owned());
    } else {
        // A `--config` role file carries `[corpus].roots`, and the launched `kb mcp` resolves them
        // from it with the same flag > env > file precedence as an interactive run — so a config
        // alone IS a complete corpus specification and must not be rejected here. Requiring
        // `--root` on top of it split the one file that is supposed to drive both `kb index` and
        // `kb mcp`, which is how a service ends up addressing the corpus differently from the pass
        // that indexed it (different labels, hence different document keys).
        //
        // But accept it only after confirming the file actually names roots: with no corpus from
        // ANY source, `kb mcp` falls back to discovery — walking up from its working directory,
        // which for a service is whatever the SCM or systemd happened to start it in. That turns a
        // typo'd section name into a service quietly serving some unrelated `.glossa` it found, or
        // a dead unit explaining nothing. Fail here, where the operator is looking.
        let named_roots = match config {
            Some(cfg) => !crate::config::load(cfg)?.corpus.roots.is_empty(),
            None => false,
        };
        if !named_roots {
            anyhow::bail!(
                "provide a <corpus> positional, one or more `--root label=path`, or a `--config` \
                 role file whose [corpus] section names at least one root"
            );
        }
    }
    if let Some(sd) = state_dir {
        args.push("--state-dir".to_string());
        args.push(sd.to_string_lossy().into_owned());
    }
    if let Some(cfg) = config {
        // Absolute, for the same reason as the token file below: the rendered unit sets no
        // WorkingDirectory, so a relative path would resolve against whatever directory the SCM or
        // systemd starts the process in — and this file is what tells the service which corpus it
        // serves. Already-absolute paths are passed through untouched (no canonicalization: the
        // file is read at every start, and resolving symlinks now would pin today's target).
        let abs = if cfg.is_absolute() {
            cfg.to_path_buf()
        } else {
            std::path::absolute(cfg).map_err(|e| {
                anyhow::anyhow!("cannot resolve the config path {}: {e}", cfg.display())
            })?
        };
        args.push("--config".to_string());
        args.push(abs.to_string_lossy().into_owned());
    }
    // Transport defaults to the only one a service can serve. An operator who names `stdio`
    // anyway gets what they asked for and a plain statement of what it will do, because a service
    // that starts and immediately stops is worth one line at install time.
    let transport = match opts.transport.as_deref() {
        None | Some("streamable-http") => "streamable-http",
        Some("stdio") => {
            crate::cli_fmt::note(
                "--transport stdio: a service has no client and no stdin under the SCM/systemd, so \
                 this one will start, find nothing to talk to, and stop. Install it only if you \
                 know why you want it; `streamable-http` is what a service serves.",
            );
            "stdio"
        }
        Some(other) => anyhow::bail!(
            "unknown --transport {other}: expected `streamable-http` (what a service serves) or \
             `stdio`"
        ),
    };
    args.extend([
        "--transport".to_string(),
        transport.to_string(),
        "--profile".to_string(),
        opts.profile.clone(),
        "--bind".to_string(),
        opts.bind.clone(),
    ]);
    for h in &opts.allowed_host {
        args.push("--allowed-host".to_string());
        args.push(h.clone());
    }
    if opts.vision {
        args.push("--vision".to_string());
    }
    if opts.dedup {
        args.push("--dedup".to_string());
    }
    // The tool-surface switches have no route through the deployment config — unlike bind, hosts,
    // limits or TLS, there is no `[tools]` section — so whatever `install` does not carry, a service
    // cannot have. `--source-file` and `--no-graph` were missing, which left `get_source_file`
    // unreachable from a service by any means at all.
    if opts.source_file {
        args.push("--source-file".to_string());
    }
    if opts.no_graph {
        args.push("--no-graph".to_string());
    }
    if opts.trace {
        args.push("--trace".to_string());
    }
    if let Some(secs) = opts.session_idle_secs {
        args.push("--session-idle-secs".to_string());
        args.push(secs.to_string());
    }
    if let Some(v) = opts.insecure {
        args.push("--insecure".to_string());
        args.push(v.to_string());
    }
    // `--noimage` is deliberately NOT forwarded: it is a deprecated no-op, accepted here only so a
    // `kb mcp` command line copies over verbatim. Baking a dead flag into ExecStart would outlive
    // the flag itself.
    //
    // `--auth-token` is accepted by the parser and refused here, on purpose. Accepted, so that
    // converting a working `kb mcp` command line into a service gets an answer instead of clap's
    // "unexpected argument" — not knowing WHY a flag is missing was the original complaint. Refused,
    // because the only way to honour it is to write the secret into the service's command line,
    // which every local account can read (`ps`; `systemctl cat` renders the unit world-readable)
    // for the life of the service. There is no version of "install this token" that is not a
    // disclosure, so the answer is the file, not a warning.
    if opts.auth_token.is_some() {
        anyhow::bail!(
            "--auth-token cannot be installed: honouring it means writing the token into the \
             service's command line, which every local account can read (`ps`, and `systemctl cat` \
             renders the unit world-readable) for as long as the service exists. Put the token in a \
             file and pass `--auth-token-file <path>` — only the path is baked in, and `kb` refuses \
             to start if that file is readable by group or other."
        );
    }
    // The PATH goes into ExecStart, never the token itself — see `InstallOpts::auth_token_file`.
    // Absolute, because the rendered unit sets no WorkingDirectory: a relative path would resolve
    // against whatever the SCM/systemd happens to start the process in.
    if let Some(tf) = &opts.auth_token_file {
        let abs = std::fs::canonicalize(tf)
            .map_err(|e| anyhow::anyhow!("cannot use the auth token file {}: {e}", tf.display()))?;
        // Fail here, where the operator is looking, rather than at the service's first start where
        // it shows up as a dead unit and a line in the journal.
        service_guard_read(&abs)?;
        args.push("--auth-token-file".to_string());
        args.push(abs.to_string_lossy().into_owned());
    }
    // TLS paths, like the token file, are absolute: the rendered unit sets no WorkingDirectory.
    #[cfg(feature = "tls")]
    for (flag, path) in [
        ("--tls-cert", opts.tls_cert.as_ref()),
        ("--tls-key", opts.tls_key.as_ref()),
        ("--tls-client-ca", opts.tls_client_ca.as_ref()),
    ] {
        let Some(p) = path else { continue };
        let abs = std::fs::canonicalize(p)
            .map_err(|e| anyhow::anyhow!("cannot use {flag} {}: {e}", p.display()))?;
        args.push(flag.to_string());
        args.push(abs.to_string_lossy().into_owned());
    }
    args.push("--windows-service".to_string());
    args.push("--service-name".to_string());
    args.push(name.to_string());
    Ok(ServiceSpec {
        name: name.to_string(),
        display_name: format!("Glossa MCP ({name})"),
        description: format!("Glossa MCP server ({name})"),
        program,
        args,
        watchdog: true,
    })
}

/// Dispatch a `kb mcp service` action. `install` builds the spec from the current executable, baking
/// the operator's global `--root`/`--state-dir`/`--config` (already parsed, passed in); the rest act
/// on the named service. Prints a one-line outcome; errors carry the elevation hint.
pub fn run(
    action: ServiceAction,
    roots: &[String],
    state_dir: Option<&Path>,
    config: Option<&Path>,
) -> anyhow::Result<()> {
    match action {
        ServiceAction::Install(opts) => {
            let program = std::env::current_exe()
                .map_err(|e| anyhow::anyhow!("cannot resolve the kb executable path: {e}"))?;
            let spec =
                kb_service_spec(&opts.service_name, program, &opts, roots, state_dir, config)?;
            service::install(&spec)?;
            println!("installed service {}", opts.service_name);
        }
        ServiceAction::Uninstall(n) => {
            service::uninstall(&n.service_name)?;
            println!("uninstalled service {}", n.service_name);
        }
        ServiceAction::Start(n) => {
            service::start(&n.service_name)?;
            println!("started service {}", n.service_name);
        }
        ServiceAction::Stop(n) => {
            service::stop(&n.service_name)?;
            println!("stopped service {}", n.service_name);
        }
        ServiceAction::Status(n) => {
            println!("{}", service::status(&n.service_name)?);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(corpus: Option<&str>) -> InstallOpts {
        InstallOpts {
            service_name: "glossa-x".into(),
            corpus: corpus.map(PathBuf::from),
            profile: "editor".into(),
            bind: "127.0.0.1:8801".into(),
            allowed_host: vec!["gw.internal".into()],
            vision: true,
            dedup: true,
            source_file: false,
            no_graph: false,
            trace: false,
            session_idle_secs: None,
            insecure: None,
            no_image: false,
            transport: None,
            auth_token: None,
            auth_token_file: None,
            #[cfg(feature = "tls")]
            tls_cert: None,
            #[cfg(feature = "tls")]
            tls_key: None,
            #[cfg(feature = "tls")]
            tls_client_ca: None,
        }
    }

    /// A TLS service must carry the cert and key into `ExecStart`: the SCM/systemd process reads
    /// its own command line, so a path that stays in the installer serves plaintext in silence.
    #[cfg(feature = "tls")]
    #[test]
    fn kb_service_spec_bakes_absolute_tls_paths() {
        let dir = tempfile::tempdir().unwrap();
        let cert = dir.path().join("leaf.crt.pem");
        let key = dir.path().join("leaf.key.pem");
        std::fs::write(
            &cert,
            "-----BEGIN CERTIFICATE-----
",
        )
        .unwrap();
        std::fs::write(
            &key,
            "-----BEGIN PRIVATE KEY-----
",
        )
        .unwrap();
        let mut o = opts(Some("C:/kb/base"));
        o.tls_cert = Some(cert.clone());
        o.tls_key = Some(key.clone());
        let s = kb_service_spec("glossa-x", "C:/kb/kb.exe".into(), &o, &[], None, None).unwrap();
        let i = s
            .args
            .iter()
            .position(|a| a == "--tls-cert")
            .expect("--tls-cert baked in");
        assert!(std::path::Path::new(&s.args[i + 1]).is_absolute());
        assert!(s.args.iter().any(|a| a == "--tls-key"));
        // Не заданный клиентский корень не превращается в mTLS по недосмотру.
        assert!(!s.args.iter().any(|a| a == "--tls-client-ca"));
    }

    #[test]
    fn kb_service_spec_bakes_http_transport_and_service_flags() {
        // Positional corpus, no globals.
        let s = kb_service_spec(
            "glossa-x",
            "C:/kb/kb.exe".into(),
            &opts(Some("C:/kb/base")),
            &[],
            None,
            None,
        )
        .unwrap();
        assert_eq!(s.name, "glossa-x");
        assert!(s
            .args
            .starts_with(&["mcp".to_string(), "C:/kb/base".to_string()]));
        // transport is ALWAYS streamable-http — there is no way to ask for stdio:
        assert!(s
            .args
            .windows(2)
            .any(|w| w == ["--transport", "streamable-http"]));
        // --vision and --dedup are baked when requested.
        assert!(s.args.iter().any(|a| a == "--vision"));
        assert!(s.args.iter().any(|a| a == "--dedup"));
        assert!(s.args.iter().any(|a| a == "--windows-service"));
        assert!(s
            .args
            .windows(2)
            .any(|w| w == ["--service-name", "glossa-x"]));
    }

    #[test]
    fn kb_service_spec_bakes_root_state_dir_and_config_globals() {
        // No positional corpus — globals supply --root/--state-dir/--config (Change 3).
        // An ABSOLUTE config path (a temp dir gives one that is absolute on every platform — a
        // bare "/etc/..." is not, on Windows) so the spec bakes it through unchanged.
        let (_d, cfg) = role_file("[corpus]\nroots = [\"docs=/srv/a\"]\n");
        let s = kb_service_spec(
            "glossa-x",
            "C:/kb/kb.exe".into(),
            &opts(None),
            &["docs=/a".to_string(), "specs=/b".to_string()],
            Some(Path::new("/var/lib/glossa/x")),
            Some(&cfg),
        )
        .unwrap();
        assert!(s.args.windows(2).any(|w| w == ["--root", "docs=/a"]));
        assert!(s.args.windows(2).any(|w| w == ["--root", "specs=/b"]));
        assert!(s
            .args
            .windows(2)
            .any(|w| w == ["--state-dir", "/var/lib/glossa/x"]));
        let baked = cfg.to_string_lossy().into_owned();
        assert!(s.args.windows(2).any(|w| w == ["--config", baked.as_str()]));
        // --root supersedes the positional: no bare corpus token after `mcp`.
        assert_eq!(s.args.first().map(String::as_str), Some("mcp"));
        assert_eq!(s.args.get(1).map(String::as_str), Some("--root"));
    }

    /// The tool-surface switches have no `[tools]` section in the deployment config to fall back
    /// on, so a flag `install` does not carry is a capability a service cannot have at all. That
    /// is how `get_source_file` ended up unreachable from a service: `kb mcp` took
    /// `--source-file`, `install` did not.
    #[test]
    fn kb_service_spec_bakes_the_tool_surface_switches() {
        let mut o = opts(Some("/srv/corpus"));
        o.source_file = true;
        o.no_graph = true;
        o.trace = true;
        o.session_idle_secs = Some(900);
        o.insecure = Some(true);
        o.no_image = true;
        let s = kb_service_spec("glossa-x", "C:/kb/kb.exe".into(), &o, &[], None, None).unwrap();
        for flag in [
            "--vision",
            "--dedup",
            "--source-file",
            "--no-graph",
            "--trace",
        ] {
            assert!(
                s.args.iter().any(|a| a == flag),
                "{flag} must reach ExecStart: {:?}",
                s.args
            );
        }
        assert!(s
            .args
            .windows(2)
            .any(|w| w == ["--session-idle-secs", "900"]));
        assert!(s.args.windows(2).any(|w| w == ["--insecure", "true"]));
        // Accepted so a `kb mcp` command line copies over, but never baked: it is a dead flag.
        assert!(
            !s.args.iter().any(|a| a == "--noimage"),
            "a deprecated no-op must not be written into ExecStart: {:?}",
            s.args
        );
        // The token VALUE and a non-default transport are available but never implied: neither
        // appears unless the operator named it, which is what keeps the costly options costly only
        // for whoever chose to pay.
        assert!(
            !s.args.iter().any(|a| a == "--auth-token"),
            "the token value must not appear unasked: {:?}",
            s.args
        );
        assert_eq!(
            s.args
                .windows(2)
                .filter(|w| w[0] == "--transport")
                .map(|w| w[1].as_str())
                .collect::<Vec<_>>(),
            vec!["streamable-http"],
            "a service serves streamable-http unless told otherwise: {:?}",
            s.args
        );

        // ...and each one stays off unless asked for: a service must not quietly hand out the
        // original file bytes because the installer was generous.
        let mut o = opts(Some("/srv/corpus"));
        o.vision = false;
        o.dedup = false;
        let s = kb_service_spec("glossa-x", "C:/kb/kb.exe".into(), &o, &[], None, None).unwrap();
        for flag in [
            "--vision",
            "--dedup",
            "--source-file",
            "--no-graph",
            "--trace",
            "--session-idle-secs",
            "--insecure",
        ] {
            assert!(
                !s.args.iter().any(|a| a == flag),
                "{flag} must not appear unasked: {:?}",
                s.args
            );
        }
    }

    /// `--auth-token` is parsed and then refused: there is no way to install a token VALUE that is
    /// not a disclosure, so the flag exists only to say so. The secret must never reach the spec.
    #[test]
    fn kb_service_spec_refuses_a_token_value_and_never_leaks_it() {
        let mut o = opts(Some("/srv/corpus"));
        o.auth_token = Some("s3cret".into());
        let e = kb_service_spec("glossa-x", "C:/kb/kb.exe".into(), &o, &[], None, None)
            .expect_err("a token value must not be installable");
        let msg = e.to_string();
        assert!(
            msg.contains("--auth-token-file"),
            "the refusal must point at the way that works: {msg}"
        );
        assert!(
            !msg.contains("s3cret"),
            "and must not echo the secret it refused: {msg}"
        );
    }

    /// `--transport` is a foot-gun, not a hole: `stdio` is honoured, loudly, because a service
    /// that starts and stops costs the operator nothing but their own time.
    #[test]
    fn kb_service_spec_honours_a_named_transport() {
        let mut o = opts(Some("/srv/corpus"));
        o.transport = Some("stdio".into());
        let s = kb_service_spec("glossa-x", "C:/kb/kb.exe".into(), &o, &[], None, None).unwrap();
        assert!(s.args.windows(2).any(|w| w == ["--transport", "stdio"]));
        assert!(
            !s.args
                .windows(2)
                .any(|w| w[0] == "--transport" && w[1] == "streamable-http"),
            "the chosen transport replaces the default, it does not join it: {:?}",
            s.args
        );

        // A transport that is neither is a typo, not a choice.
        let mut o = opts(Some("/srv/corpus"));
        o.transport = Some("sse".into());
        let e = kb_service_spec("glossa-x", "C:/kb/kb.exe".into(), &o, &[], None, None);
        assert!(e.is_err(), "an unknown transport must not install");
    }

    /// A role file's `[corpus]` section is a complete corpus specification — `kb index` and
    /// `kb mcp` both resolve roots from it — so `service install --config role.toml` must not
    /// demand `--root` on top. Requiring it forced the operator to state the roots twice, in two
    /// places that can drift apart; and roots that drift re-key every document.
    /// Writes a role file and hands back its path (kept alive by the returned dir).
    fn role_file(body: &str) -> (tempfile::TempDir, PathBuf) {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("role.toml");
        std::fs::write(&p, body).unwrap();
        (d, p)
    }

    #[test]
    fn kb_service_spec_accepts_a_config_as_the_whole_corpus_spec() {
        let (_d, cfg) = role_file("[corpus]\nroots = [\"docs=/srv/a\"]\n");
        let s = kb_service_spec(
            "glossa-x",
            "C:/kb/kb.exe".into(),
            &opts(None),
            &[],
            None,
            Some(&cfg),
        )
        .expect("a --config role file naming roots must be accepted");
        let baked = cfg.to_string_lossy().into_owned();
        assert!(s.args.windows(2).any(|w| w == ["--config", baked.as_str()]));
        assert!(
            !s.args.iter().any(|a| a == "--root"),
            "nothing to bake: the roots live in the config: {:?}",
            s.args
        );
    }

    /// Fail CLOSED: a config with no roots is not a corpus specification. Accepting it would leave
    /// the service to fall back on discovery from whatever directory the SCM started it in.
    #[test]
    fn kb_service_spec_rejects_a_config_that_names_no_roots() {
        let (_d, cfg) = role_file("[server]\nbind = \"127.0.0.1:8080\"\n");
        let e = kb_service_spec(
            "glossa-x",
            "C:/kb/kb.exe".into(),
            &opts(None),
            &[],
            None,
            Some(&cfg),
        );
        assert!(e.is_err(), "a rootless config must not pass for a corpus");
    }

    #[test]
    fn kb_service_spec_requires_corpus_or_root() {
        let e = kb_service_spec(
            "glossa-x",
            "C:/kb/kb.exe".into(),
            &opts(None),
            &[],
            None,
            None,
        );
        assert!(e.is_err(), "neither corpus nor --root must error");
    }

    /// The point of `--auth-token-file`: a service's command line is public, so the PATH is what
    /// gets baked into ExecStart. A test that only checked the flag is present would pass for a
    /// version that also appended the secret, so assert the secret's absence too.
    #[test]
    fn the_spec_carries_the_token_file_path_and_never_a_token() {
        // A real file: install now validates it (and canonicalizes the path) instead of letting
        // the service fail at its first start.
        let dir = tempfile::tempdir().unwrap();
        let token_path = dir.path().join("kb.env");
        std::fs::write(&token_path, "s3cret").unwrap();
        restrict(&token_path);
        let mut opts = opts(Some("/srv/corpus"));
        opts.auth_token_file = Some(token_path.clone());

        let spec = kb_service_spec(
            "kb",
            PathBuf::from("/usr/local/bin/kb"),
            &opts,
            &[],
            None,
            None,
        )
        .expect("spec");

        let i = spec
            .args
            .iter()
            .position(|a| a == "--auth-token-file")
            .expect("the flag is passed to the served process");
        assert!(
            std::path::Path::new(&spec.args[i + 1]).is_absolute(),
            "the baked path must be absolute (the unit sets no WorkingDirectory): {}",
            spec.args[i + 1]
        );
        assert!(spec.args[i + 1].ends_with("kb.env"), "{}", spec.args[i + 1]);
        assert!(
            !spec
                .args
                .iter()
                .any(|a| a.starts_with("--auth-token") && a != "--auth-token-file"),
            "the token itself must never reach the command line: {:?}",
            spec.args
        );
    }

    #[test]
    fn without_the_flag_the_service_is_unauthenticated_as_before() {
        let spec = kb_service_spec(
            "kb",
            PathBuf::from("/usr/local/bin/kb"),
            &opts(Some("/srv/corpus")),
            &[],
            None,
            None,
        )
        .expect("spec");

        assert!(!spec.args.iter().any(|a| a.starts_with("--auth-token")));
    }

    /// 0600 on Unix; a no-op elsewhere (the mode/ownership check is Unix-only).
    fn restrict(path: &std::path::Path) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        #[cfg(not(unix))]
        let _ = path;
    }
}
