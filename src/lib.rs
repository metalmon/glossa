pub mod audit;
pub mod cli_fmt;
pub mod config;
pub mod config_util;
pub mod conn_cap;
pub mod convert;
pub mod default_ignore;
pub mod extract;
pub mod fs_detect;
pub mod gate;
pub mod glob;
pub mod graph;
pub mod grep;
pub mod http_metrics;
pub mod http_scorer;
pub mod index;
pub mod json_util;
pub mod logreload;
pub mod mcp;
pub mod mcp_auth;
pub mod model;
pub mod ontology_templates;
pub mod prompts;
pub mod query;
pub mod read;
pub mod retrieve;
pub mod root;
pub mod sdnotify;
pub mod search;
pub mod serve_guard;
pub mod service;
pub mod service_cli;
pub mod session_idle;
pub mod tables;
pub mod tools;
pub mod trace;
pub mod tz_export;
pub mod walk;

#[cfg(feature = "notebook")]
pub mod notebook;

#[cfg(feature = "constraint")]
pub mod constraint_adapter;

#[cfg(feature = "tls")]
pub mod tls;

/// Version string shown by `kb --version` / `kbx --version`, with the compiled NLI inference engine
/// AND its execution provider appended, so a binary self-reports exactly which build it is:
/// `ort-cuda`, `ort-directml`, `ort-coreml`, `ort` (CPU), `burn-wgpu`, or `none`.
///
/// The EP is part of this string on purpose. `release.yml` ships two different Windows prod
/// artifacts from the same platform — `engine: ort` (`--features nli-directml`) and
/// `engine: cuda13` (`--features nli-cuda`) — and both used to print the identical
/// `(nli engine: ort)`. The deploy procedure's "verify `--version` before the swap" step therefore
/// could not tell them apart, so a DirectML build could silently replace a CUDA one and downgrade
/// GPU inference with nothing in the output to show it. Runtime EP selection still belongs to
/// `kbx nli check`; this reports what was COMPILED IN.
pub fn version() -> &'static str {
    static V: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    V.get_or_init(|| format!("{} (nli engine: {})", env!("CARGO_PKG_VERSION"), engine()))
}

/// Ask the crate that OWNS the execution provider. glossa's own features cannot see the EP: a top
/// crate may enable it on `glossa-nli` while enabling only `nli-dynamic` here — which is exactly how
/// `kb` printed `ort-cuda` and `kbx` printed plain `ort` from the same cuda13 artifact.
///
/// Every EP feature on this crate implies one of these three (`nli-cuda`/`nli-rocm` => `nli-dynamic`,
/// `nli-directml`/`nli-coreml` => `nli`; see `[features]` in Cargo.toml), so this gate is exactly the
/// set of builds in which `glossa-nli` is linked at all.
#[cfg(any(feature = "nli", feature = "nli-dynamic", feature = "nli-burn"))]
fn engine() -> &'static str {
    glossa_nli::engine_name()
}
#[cfg(not(any(feature = "nli", feature = "nli-dynamic", feature = "nli-burn")))]
fn engine() -> &'static str {
    "none"
}

/// Serializes tests that mutate process-global environment variables (`std::env::set_var`/
/// `remove_var`). Rust runs tests concurrently in one process, so without this they race.
/// `into_inner()` recovers the guard if a holder panicked (a failed test must not poison it).
/// Every env-mutating test must hold this for its whole body:
/// `let _env = crate::TEST_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());`
#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        assert!(!version().is_empty());
    }

    /// Review Focus 5: with no NLI feature the glossa-nli dep is not linked at all, so `version()`
    /// must fall back rather than call into a crate that is absent.
    #[cfg(not(any(feature = "nli", feature = "nli-dynamic", feature = "nli-burn")))]
    #[test]
    fn version_engine_is_none_without_an_nli_feature() {
        assert!(version().ends_with("(nli engine: none)"), "{}", version());
    }

    /// The engine suffix must always be present and must name one of the known engines. A missing
    /// or mistyped `cfg` arm would otherwise surface only as a deploy-time surprise — which is the
    /// exact failure this string exists to prevent (two Windows prod artifacts that printed the
    /// same `ort` and could silently replace one another).
    #[test]
    fn version_names_a_known_engine() {
        let v = version();
        let (_, engine) = v
            .split_once("(nli engine: ")
            .unwrap_or_else(|| panic!("version must carry the engine suffix: {v}"));
        let engine = engine.trim_end_matches(')');
        assert!(
            matches!(
                engine,
                "ort-cuda"
                    | "ort-rocm"
                    | "ort-directml"
                    | "ort-coreml"
                    | "ort"
                    | "burn-wgpu"
                    | "none"
            ),
            "unknown engine token {engine:?} in version string {v:?}"
        );
    }
}
