//! Declarative TOML deployment config (`--config` / `GLOSSA_CONFIG`) — Spec E.
//! Aggregates the Spec A/B/C settings surface for one MCP role. Adds no runtime
//! behavior: it is a new *source* of existing settings, layered under flags/env.
use anyhow::Context;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DeploymentConfig {
    #[serde(default)]
    pub corpus: CorpusSection,
    #[serde(default)]
    pub server: ServerSection,
    /// `Option` so an absent `[tls]` table is `None` and a present one (even empty) is `Some` —
    /// that presence flag drives the non-`tls`-build error in `validate_static`.
    #[serde(default)]
    pub tls: Option<TlsSection>,
    #[serde(default)]
    pub limits: LimitsSection,
    #[serde(default)]
    pub retrieval: RetrievalSection,
    #[serde(default)]
    pub logging: LoggingSection,
}

/// `[corpus]` — Spec A. `roots` are `"label=path"` entries (same grammar as `--root`).
#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CorpusSection {
    #[serde(default)]
    pub roots: Vec<String>,
    #[serde(default)]
    pub state_dir: Option<PathBuf>,
}

/// `[server]` — Spec C. The bearer token is intentionally NOT a usable field; `auth_token`/`token`
/// exist only so a secret in the file yields a targeted security error (not a generic unknown-field).
#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ServerSection {
    #[serde(default)]
    pub transport: Option<String>,
    #[serde(default)]
    pub bind: Option<String>,
    #[serde(default)]
    pub session_idle_secs: Option<u64>,
    #[serde(default)]
    pub allowed_hosts: Vec<String>,
    #[serde(default)]
    pub insecure: Option<bool>,
    #[serde(default)]
    pub auth_token: Option<toml::Value>,
    #[serde(default)]
    pub token: Option<toml::Value>,
}

/// `[tls]` — Spec C, only meaningful in a `--features tls` build.
#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TlsSection {
    #[serde(default)]
    pub cert: Option<PathBuf>,
    #[serde(default)]
    pub key: Option<PathBuf>,
    #[serde(default)]
    pub client_ca: Option<PathBuf>,
}

/// `[limits]` — Spec C. All opt-in; unset = off / built-in default.
#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LimitsSection {
    #[serde(default)]
    pub request_timeout_secs: Option<u64>,
    #[serde(default)]
    pub max_body_bytes: Option<usize>,
    #[serde(default)]
    pub max_concurrency: Option<usize>,
    #[serde(default)]
    pub rate_limit_per_sec: Option<u32>,
    #[serde(default)]
    pub connection_cap: Option<usize>,
    /// Pre-auth TLS handshake concurrency bound (`tls` feature only; round-1 fix C1) -- maps to
    /// `GLOSSA_MCP_MAX_HANDSHAKES`. Unlike the other knobs in this section it has a built-in
    /// default (`glossa::tls::DEFAULT_MAX_HANDSHAKES`) even when unset here and unset in the env.
    #[serde(default)]
    pub max_handshakes: Option<usize>,
}

/// `[retrieval]` — Spec B.
#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RetrievalSection {
    #[serde(default)]
    pub read_retries: Option<u32>,
    #[serde(default)]
    pub read_retry_backoff_ms: Option<u64>,
    #[serde(default)]
    pub freshen_deadline_ms: Option<u64>,
    /// Milliseconds — maps to `GLOSSA_MIN_RESCAN_MS` (default 2000).
    #[serde(default)]
    pub min_rescan_ms: Option<u64>,
}

/// `[logging]` — Spec C. `format` = `json|text` (→ `GLOSSA_LOG_FORMAT`); `level` = RUST_LOG-style.
#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LoggingSection {
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub level: Option<String>,
}

impl DeploymentConfig {
    /// Fail-closed checks that depend only on the file (no flag/env layer): secret rejection and
    /// TLS feature-gating (a populated `[tls]` section in a build without the `tls` cargo feature
    /// errors here rather than being silently ignored). Run at load time so every caller is
    /// protected identically.
    pub fn validate_static(&self) -> anyhow::Result<()> {
        if self.server.auth_token.is_some() || self.server.token.is_some() {
            anyhow::bail!(
                "the auth token must never live in the config file — set it via `--auth-token` \
                 or the `GLOSSA_MCP_TOKEN` env var instead. Remove the `token`/`auth_token` key \
                 from the [server] section."
            );
        }
        #[cfg(not(feature = "tls"))]
        if self.tls.is_some() {
            anyhow::bail!(
                "the [tls] section requires a binary built with the `tls` feature, but this build \
                 has it disabled. Rebuild with `--features tls`, or remove the [tls] section and \
                 terminate TLS at a reverse proxy."
            );
        }
        Ok(())
    }

    /// True when the file carries any serving-only setting ([server]/[tls]/[limits]/[logging]).
    /// Non-serving subcommands (`kb index`, `kb search`, `kb graph …`, …) use this to emit a
    /// `tracing::debug!` note (never an error) — one role config file must work for both
    /// provisioning (`kb index`) and serving (`kb mcp`), so a serving-only section present while
    /// running a non-serving subcommand is inert, not a failure.
    pub fn serving_sections_present(&self) -> bool {
        self.server != ServerSection::default()
            || self.tls.is_some()
            || self.limits != LimitsSection::default()
            || self.logging != LoggingSection::default()
    }
}

/// The config path to use: the explicit `--config` flag if given, else `GLOSSA_CONFIG`, else none.
pub fn config_path(flag: Option<PathBuf>) -> Option<PathBuf> {
    flag.or_else(|| std::env::var_os("GLOSSA_CONFIG").map(PathBuf::from))
}

/// Read and parse a deployment config file. Static validation (secrets, tls feature-gate) is applied
/// here via `validate_static` so every caller gets the same fail-closed guarantees.
pub fn load(path: &Path) -> anyhow::Result<DeploymentConfig> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading config file {}", path.display()))?;
    let cfg: DeploymentConfig = toml::from_str(&text)
        .with_context(|| format!("parsing config file {}", path.display()))?;
    cfg.validate_static()
        .with_context(|| format!("validating config file {}", path.display()))?;
    Ok(cfg)
}

/// Per-setting precedence with a built-in default: CLI-flag/env value beats the file value beats the
/// default. (clap's `env=` already collapses flag-then-env into `cli_or_env`.)
pub fn pick<T>(cli_or_env: Option<T>, file: Option<T>, default: T) -> T {
    cli_or_env.or(file).unwrap_or(default)
}

/// Precedence for a setting that has no built-in default (stays `None` when unset everywhere,
/// e.g. `state_dir`, tls paths, `max_concurrency`).
pub fn pick_opt<T>(cli_or_env: Option<T>, file: Option<T>) -> Option<T> {
    cli_or_env.or(file)
}

/// List precedence: the CLI/env list if the operator supplied any entries, else the file list
/// (for `roots` and `allowed_hosts`).
pub fn merge_list(cli_or_env: Vec<String>, file: Vec<String>) -> Vec<String> {
    if cli_or_env.is_empty() {
        file
    } else {
        cli_or_env
    }
}

/// Built-in defaults, kept here as the single source of truth so both clap and the merge agree.
/// Mirrors today's clap `default_value`s / env-or-default helpers (src/main.rs, src/index/store.rs);
/// settings with no built-in default (opt-in only, e.g. `max_concurrency`, `rate_limit_per_sec`,
/// `connection_cap`, `state_dir`, tls paths) are merged via `pick_opt` instead and have no const here.
pub mod defaults {
    /// src/main.rs:337 (`--bind`, env `GLOSSA_MCP_BIND`).
    pub const BIND: &str = "127.0.0.1:8080";
    /// src/main.rs:332 (`--transport`).
    pub const TRANSPORT: &str = "stdio";
    /// src/main.rs:377 (`--session-idle-secs`).
    pub const SESSION_IDLE_SECS: u64 = 0;
    /// src/main.rs:967 (`GLOSSA_MCP_REQUEST_TIMEOUT_SECS`).
    pub const REQUEST_TIMEOUT_SECS: u64 = 120;
    /// src/main.rs:971 (`GLOSSA_MCP_MAX_BODY_BYTES`).
    pub const MAX_BODY_BYTES: usize = 4_000_000;
    /// src/index/store.rs (`GLOSSA_READ_RETRIES`).
    pub const READ_RETRIES: u32 = 3;
    /// src/index/store.rs (`GLOSSA_READ_RETRY_BACKOFF_MS`).
    pub const READ_RETRY_BACKOFF_MS: u64 = 200;
    /// src/index/store.rs (`GLOSSA_FRESHEN_DEADLINE_MS`).
    pub const FRESHEN_DEADLINE_MS: u64 = 3000;
    /// src/index/store.rs (`GLOSSA_MIN_RESCAN_MS`).
    pub const MIN_RESCAN_MS: u64 = 2000;
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: &str = r#"
[corpus]
roots = ["docs=/mnt/store/a", "specs=/mnt/store/b"]
state_dir = "/var/lib/glossa/roleA"

[server]
transport = "streamable-http"
bind = "127.0.0.1:8080"
session_idle_secs = 900
allowed_hosts = ["kb.example.com"]

[tls]
cert = "/etc/glossa/roleA/tls/cert.pem"
key = "/etc/glossa/roleA/tls/key.pem"
client_ca = "/etc/glossa/roleA/tls/clients-ca.pem"

[limits]
request_timeout_secs = 120
max_body_bytes = 4000000
max_concurrency = 32

[retrieval]
read_retries = 3
read_retry_backoff_ms = 200
freshen_deadline_ms = 3000
min_rescan_ms = 300

[logging]
format = "json"
level = "info"
"#;

    #[test]
    fn parses_full_config() {
        let cfg: DeploymentConfig = toml::from_str(FULL).unwrap();
        assert_eq!(
            cfg.corpus.roots,
            vec!["docs=/mnt/store/a".to_string(), "specs=/mnt/store/b".to_string()]
        );
        assert_eq!(cfg.corpus.state_dir.as_deref(), Some(std::path::Path::new("/var/lib/glossa/roleA")));
        assert_eq!(cfg.server.transport.as_deref(), Some("streamable-http"));
        assert_eq!(cfg.server.session_idle_secs, Some(900));
        assert_eq!(cfg.limits.max_body_bytes, Some(4_000_000));
        assert_eq!(cfg.retrieval.read_retries, Some(3));
        assert_eq!(cfg.retrieval.min_rescan_ms, Some(300));
        assert_eq!(cfg.logging.format.as_deref(), Some("json"));
        let tls = cfg.tls.expect("[tls] present");
        assert_eq!(tls.cert.as_deref(), Some(std::path::Path::new("/etc/glossa/roleA/tls/cert.pem")));
    }

    #[test]
    fn empty_config_is_all_none() {
        let cfg: DeploymentConfig = toml::from_str("").unwrap();
        assert_eq!(cfg, DeploymentConfig::default());
        assert!(cfg.tls.is_none());
        assert!(cfg.corpus.roots.is_empty());
    }

    #[test]
    fn unknown_top_level_key_is_rejected() {
        let err = toml::from_str::<DeploymentConfig>("[bogus]\nx = 1\n").unwrap_err();
        assert!(err.to_string().contains("bogus") || err.to_string().contains("unknown"),
            "deny_unknown_fields must reject unknown section: {err}");
    }

    #[test]
    fn unknown_nested_key_is_rejected() {
        let err = toml::from_str::<DeploymentConfig>("[server]\nbnid = \"x\"\n").unwrap_err();
        assert!(err.to_string().contains("bnid") || err.to_string().contains("unknown"),
            "deny_unknown_fields must reject typo'd key: {err}");
    }

    #[test]
    fn config_path_prefers_flag_over_env() {
        // Flag wins even when the env is set.
        std::env::set_var("GLOSSA_CONFIG", "/from/env.toml");
        let got = config_path(Some(PathBuf::from("/from/flag.toml")));
        std::env::remove_var("GLOSSA_CONFIG");
        assert_eq!(got, Some(PathBuf::from("/from/flag.toml")));
    }

    #[test]
    fn config_path_none_when_unset() {
        std::env::remove_var("GLOSSA_CONFIG");
        assert_eq!(config_path(None), None);
    }

    #[test]
    fn config_path_falls_back_to_env_when_no_flag() {
        // No flag given: the env var must actually be read, not ignored.
        std::env::set_var("GLOSSA_CONFIG", "/from/env.toml");
        let got = config_path(None);
        std::env::remove_var("GLOSSA_CONFIG");
        assert_eq!(got, Some(PathBuf::from("/from/env.toml")));
    }

    #[test]
    fn load_reads_and_parses_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("role.toml");
        std::fs::write(&p, "[server]\nbind = \"0.0.0.0:9000\"\n").unwrap();
        let cfg = load(&p).unwrap();
        assert_eq!(cfg.server.bind.as_deref(), Some("0.0.0.0:9000"));
    }

    #[test]
    fn load_missing_file_errors_with_path() {
        let err = load(std::path::Path::new("/no/such/role.toml")).unwrap_err();
        assert!(err.to_string().contains("role.toml"), "error names the file: {err}");
    }

    #[test]
    fn load_bad_toml_errors_with_path() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("bad.toml");
        std::fs::write(&p, "this = = not toml").unwrap();
        let err = load(&p).unwrap_err();
        assert!(err.to_string().contains("bad.toml"), "parse error names the file: {err}");
    }

    #[test]
    fn token_key_in_file_is_rejected() {
        let cfg: DeploymentConfig =
            toml::from_str("[server]\nauth_token = \"deadbeef\"\n").unwrap();
        let err = cfg.validate_static().unwrap_err();
        let m = err.to_string();
        assert!(m.contains("token"), "secret error mentions token: {m}");
        assert!(
            m.contains("GLOSSA_MCP_TOKEN") || m.contains("--auth-token"),
            "secret error points at the env/flag path: {m}"
        );
    }

    #[test]
    fn bare_token_alias_in_file_is_rejected() {
        let cfg: DeploymentConfig = toml::from_str("[server]\ntoken = \"deadbeef\"\n").unwrap();
        assert!(cfg.validate_static().is_err());
    }

    #[test]
    fn no_token_key_validates_ok() {
        let cfg: DeploymentConfig =
            toml::from_str("[server]\nbind = \"127.0.0.1:8080\"\n").unwrap();
        assert!(cfg.validate_static().is_ok());
    }

    #[test]
    #[cfg(not(feature = "tls"))]
    fn tls_section_rejected_in_non_tls_build() {
        let cfg: DeploymentConfig =
            toml::from_str("[tls]\ncert = \"/etc/c.pem\"\nkey = \"/etc/k.pem\"\n").unwrap();
        let err = cfg.validate_static().unwrap_err();
        let m = err.to_string();
        assert!(m.contains("tls"), "error mentions the tls feature: {m}");
        assert!(m.contains("feature"), "error explains the feature gate: {m}");
    }

    #[test]
    #[cfg(not(feature = "tls"))]
    fn empty_tls_header_also_rejected_in_non_tls_build() {
        // Even a bare `[tls]` header signals intent that this build can't honor.
        let cfg: DeploymentConfig = toml::from_str("[tls]\n").unwrap();
        assert!(cfg.validate_static().is_err());
    }

    #[test]
    #[cfg(feature = "tls")]
    fn tls_section_accepted_in_tls_build() {
        let cfg: DeploymentConfig =
            toml::from_str("[tls]\ncert = \"/etc/c.pem\"\nkey = \"/etc/k.pem\"\n").unwrap();
        assert!(cfg.validate_static().is_ok());
    }

    #[test]
    fn pick_precedence_full_ladder() {
        // flag/env present → wins over file and default.
        assert_eq!(pick(Some(900u64), Some(120), 0), 900);
        // flag/env absent, file present → file wins over default.
        assert_eq!(pick(None, Some(120u64), 0), 120);
        // both absent → built-in default.
        assert_eq!(pick(None::<u64>, None, 0), 0);
    }

    #[test]
    fn pick_opt_keeps_none_when_all_unset() {
        assert_eq!(pick_opt(Some(1u8), Some(2)), Some(1));
        assert_eq!(pick_opt(None, Some(2u8)), Some(2));
        assert_eq!(pick_opt(None::<u8>, None), None);
    }

    #[test]
    fn merge_list_prefers_cli_env_then_file() {
        assert_eq!(
            merge_list(vec!["a".into()], vec!["b".into()]),
            vec!["a".to_string()]
        );
        assert_eq!(
            merge_list(vec![], vec!["b".into()]),
            vec!["b".to_string()]
        );
        assert_eq!(merge_list(Vec::new(), Vec::new()), Vec::<String>::new());
    }

    #[test]
    fn defaults_match_todays_clap_values() {
        assert_eq!(defaults::BIND, "127.0.0.1:8080");
        assert_eq!(defaults::TRANSPORT, "stdio");
        assert_eq!(defaults::SESSION_IDLE_SECS, 0);
    }

    #[test]
    fn serving_sections_present_detects_server_keys() {
        let cfg: DeploymentConfig = toml::from_str("[server]\nbind = \"127.0.0.1:8080\"\n").unwrap();
        assert!(cfg.serving_sections_present());
    }

    #[test]
    fn serving_sections_absent_for_corpus_only() {
        let cfg: DeploymentConfig =
            toml::from_str("[corpus]\nroots = [\"docs=/mnt/a\"]\n").unwrap();
        assert!(!cfg.serving_sections_present());
    }

    #[test]
    fn defaults_match_todays_env_or_default_helpers() {
        // src/main.rs + src/index/store.rs env-or-default helpers — kept in lockstep here so the
        // config-file merge path never silently diverges from the flag/env path's fallback value.
        assert_eq!(defaults::REQUEST_TIMEOUT_SECS, 120);
        assert_eq!(defaults::MAX_BODY_BYTES, 4_000_000);
        assert_eq!(defaults::READ_RETRIES, 3);
        assert_eq!(defaults::READ_RETRY_BACKOFF_MS, 200);
        assert_eq!(defaults::FRESHEN_DEADLINE_MS, 3000);
        assert_eq!(defaults::MIN_RESCAN_MS, 2000);
    }
}
