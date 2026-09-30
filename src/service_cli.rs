//! Shared `service {install|uninstall|start|stop|status}` CLI shape, reused by `kb` and
//! `kbi`. The install options are a CURATED set that can only describe a valid service:
//! `kb`'s transport is hardcoded to `streamable-http` (there is no `--transport` flag), so a stdio
//! service — which has no client/stdin under the SCM/systemd — is unrepresentable by construction.

use crate::service::{self, ServiceSpec};
use clap::{Args, Subcommand};
use std::path::{Path, PathBuf};

/// Just a service name — the key for uninstall/start/stop/status. Reused across binaries.
#[derive(Args, Debug)]
pub struct NameArg {
    /// The service name (as given to `install --service-name`).
    pub service_name: String,
}

/// Curated `kb service install` options. There is deliberately NO `--transport`: transport is
/// hardcoded to `streamable-http`, so a stdio service (which has no client/stdin under SCM/systemd)
/// cannot be created by construction.
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
}

/// `kb service <action>`: install a streamable-http MCP service, or manage one by name.
#[derive(Subcommand, Debug)]
pub enum ServiceAction {
    /// Install (register) a kb MCP service (always streamable-http).
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

/// Build the [`ServiceSpec`] for a kb MCP service: transport is ALWAYS `streamable-http`, plus the
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
        anyhow::bail!(
            "provide a <corpus> positional or one or more `--root label=path` for the service"
        );
    }
    if let Some(sd) = state_dir {
        args.push("--state-dir".to_string());
        args.push(sd.to_string_lossy().into_owned());
    }
    if let Some(cfg) = config {
        args.push("--config".to_string());
        args.push(cfg.to_string_lossy().into_owned());
    }
    args.extend([
        "--transport".to_string(),
        "streamable-http".to_string(),
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
        }
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
        let s = kb_service_spec(
            "glossa-x",
            "C:/kb/kb.exe".into(),
            &opts(None),
            &["docs=/a".to_string(), "specs=/b".to_string()],
            Some(Path::new("/var/lib/glossa/x")),
            Some(Path::new("/etc/glossa/role.toml")),
        )
        .unwrap();
        assert!(s.args.windows(2).any(|w| w == ["--root", "docs=/a"]));
        assert!(s.args.windows(2).any(|w| w == ["--root", "specs=/b"]));
        assert!(s
            .args
            .windows(2)
            .any(|w| w == ["--state-dir", "/var/lib/glossa/x"]));
        assert!(s
            .args
            .windows(2)
            .any(|w| w == ["--config", "/etc/glossa/role.toml"]));
        // --root supersedes the positional: no bare corpus token after `mcp`.
        assert_eq!(s.args.first().map(String::as_str), Some("mcp"));
        assert_eq!(s.args.get(1).map(String::as_str), Some("--root"));
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
}
