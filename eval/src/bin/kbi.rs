//! `kbi` — the Glossa inference server: serve the NLI + reranker cross-encoders over HTTP. The real
//! server needs an ORT engine feature (it loads `InProcessNli`/`InProcessReranker`); a build without
//! one prints how to build it. Bare `kbi <flags>` runs the server (foreground, or under the Windows
//! SCM / Linux systemd via `--windows-service` / a systemd unit); `kbi service …` installs and
//! manages it as an OS service.

#[cfg(any(
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
))]
use kb_eval::infer::cli::ServeArgs;

/// `kbi [serve flags] | kbi service …`. Serve is the DEFAULT — bare `kbi --bind …` serves; the only
/// subcommand is `service`. ServeArgs has no positionals (all `--flags`), so `kbi service` is
/// unambiguously the subcommand.
#[cfg(any(
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
))]
#[derive(clap::Parser)]
#[command(name = "kbi", version = glossa::version())]
// The serve flags and the `service` subcommand are mutually exclusive: reject `kbi --bind X service …`
// (which would otherwise parse both and silently drop the pre-subcommand serve flags) rather than
// running the wrong thing.
#[command(args_conflicts_with_subcommands = true)]
struct Cli {
    /// `kbi service …` / `kbi fit …`. Absent ⇒ serve with the flags below.
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    serve: ServeArgs,
}

/// The two things `kbi` does besides serve.
#[cfg(any(
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
))]
// `Fit` flattens the whole serve-flag struct and `Service` holds a name or two, so the variants are
// far apart in size. Boxing the big one is what the lint asks for and what clap cannot do — a
// variant's payload has to implement `Args`, which `Box<FitOpts>` does not — and the cost it is
// warning about does not exist here: exactly one of these is built, once, at process start.
#[allow(clippy::large_enum_variant)]
#[derive(clap::Subcommand)]
enum Command {
    /// Install/manage `kbi` as an OS service.
    #[command(subcommand)]
    Service(InferServiceAction),
    /// Measure the batch budget this device actually pays for, and print it. Writes nothing — the
    /// number goes into `--rerank-batch-tokens` / `--nli-batch-tokens`, or into the ontology on the
    /// `kbx` side.
    Fit(FitOpts),
}

/// `kbi fit` — the same sweep `--fit` runs at startup, as a command that prints and exits.
#[cfg(any(
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
))]
#[derive(clap::Args)]
struct FitOpts {
    /// Model dirs/repos and device flags: the same ones serving takes, so a fit describes the
    /// deployment it will run under.
    #[command(flatten)]
    serve: ServeArgs,
    /// Ceiling on the sweep, in rows per batch (clamped to 64, the most the planner puts in one
    /// batch). Load-bearing: the sweep's own peak stays resident.
    #[arg(long = "max-rows", default_value_t = 32)]
    max_rows: usize,
    /// Row length to measure at, in tokens. Default AND ceiling: the length the engine runs (512)
    /// -- above it the tokenizer truncates, so a longer sweep would time 512-token rows and then
    /// recommend a budget for rows that do not exist. Shorter is honest, and fitting at the length a
    /// corpus actually produces is a different answer.
    #[arg(long = "seq")]
    seq: Option<usize>,
    /// Passes per size; the MIDDLE reading of each size is kept (one lucky pass must not
    /// take the recommendation).
    #[arg(long = "repeats", default_value_t = 3)]
    repeats: usize,
    /// How close to the best a smaller size must be to win it the recommendation.
    #[arg(long = "tolerance", default_value_t = kb_eval::fit::DEFAULT_TOLERANCE)]
    tolerance: f64,
}

#[cfg(any(
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
))]
#[derive(clap::Subcommand)]
enum InferServiceAction {
    /// Install (register) a service that runs `kbi` with the given serve flags.
    Install(InferInstallOpts),
    /// Remove a service.
    Uninstall(glossa::service_cli::NameArg),
    /// Start a service.
    Start(glossa::service_cli::NameArg),
    /// Stop a service.
    Stop(glossa::service_cli::NameArg),
    /// Print a service's status.
    Status(glossa::service_cli::NameArg),
}

#[cfg(any(
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
))]
#[derive(clap::Args)]
struct InferInstallOpts {
    /// Unique service name (the SCM/systemd key).
    #[arg(long = "service-name")]
    service_name: String,
    /// The `kbi` serve flags to bake into the service — pass them after `--`, e.g.
    /// `kbi service install --service-name s -- --rerank-repo <repo> --bind <addr>`.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    serve_args: Vec<String>,
}

#[cfg(any(
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
))]
fn main() -> anyhow::Result<()> {
    use clap::Parser;
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Service(action)) => run_service(action),
        Some(Command::Fit(o)) => run_fit(o),
        None => {
            let args = cli.serve;
            if args.windows_service {
                // Launched by the SCM (binPath carries --windows-service): drive serve under the
                // shared dispatcher (Stop/Shutdown → cancel; on_ready flips the service to Running).
                let name = args
                    .service_name
                    .clone()
                    .unwrap_or_else(|| "glossa-kbi".to_string());
                glossa::service::run_as_service(
                    name,
                    Box::new(move |cancel, on_ready| serve_blocking(args, cancel, on_ready)),
                )
            } else {
                // Foreground / systemd: serve_blocking installs its own Ctrl-C + SIGTERM bridges and
                // sends sd_notify readiness.
                serve_blocking(
                    args,
                    tokio_util::sync::CancellationToken::new(),
                    Box::new(|| {}),
                )
            }
        }
    }
}

/// Dispatch `kbi fit`: load every configured model — the neighbour included, because residency is
/// the hazard this measures under — sweep, and print. Binds no socket and writes nothing.
#[cfg(any(
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
))]
fn run_fit(o: FitOpts) -> anyhow::Result<()> {
    let mut args = o.serve.clone();
    // `warm()` would otherwise run the startup fit with the server's own knobs; here the knobs are
    // this command's, applied once the models are up.
    args.fit = false;
    let state = kb_eval::infer::state::build_state(&args)?;
    state.warm()?;
    state.fit_in_place(&kb_eval::fit::SweepOpts {
        // Clamped to what the planner can actually build a batch from, and to the row length the
        // engine will run: a sweep above either measures something the serving path cannot execute.
        max_rows: o.max_rows.clamp(1, glossa_nli::harness::NLI_BATCH_MAX_ROWS),
        seq: kb_eval::fit::effective_seq(o.seq, glossa_nli::harness::DEFAULT_MAX_SEQ_LEN),
        repeats: o.repeats,
        tolerance: kb_eval::fit::sanitize_tolerance(o.tolerance),
    });
    Ok(())
}

/// Dispatch `kbi service <action>`.
#[cfg(any(
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
))]
fn run_service(action: InferServiceAction) -> anyhow::Result<()> {
    use kb_eval::infer::cli::infer_service_spec;
    match action {
        InferServiceAction::Install(o) => {
            let program = std::env::current_exe()
                .map_err(|e| anyhow::anyhow!("cannot resolve the kbi executable path: {e}"))?;
            glossa::service::install(&infer_service_spec(&o.service_name, program, &o.serve_args))?;
            println!("installed service {}", o.service_name);
        }
        InferServiceAction::Uninstall(n) => {
            glossa::service::uninstall(&n.service_name)?;
            println!("uninstalled service {}", n.service_name);
        }
        InferServiceAction::Start(n) => {
            glossa::service::start(&n.service_name)?;
            println!("started service {}", n.service_name);
        }
        InferServiceAction::Stop(n) => {
            glossa::service::stop(&n.service_name)?;
            println!("stopped service {}", n.service_name);
        }
        InferServiceAction::Status(n) => {
            println!("{}", glossa::service::status(&n.service_name)?);
        }
    }
    Ok(())
}

/// Serve until `cancel` is tripped, calling `on_ready` once the socket is bound and the models are
/// warm. `cancel` is driven by the caller: the Windows SCM control handler under a service, or the
/// Ctrl-C / SIGTERM bridges this function installs for the foreground / systemd path.
#[cfg(any(
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
))]
fn serve_blocking(
    args: ServeArgs,
    cancel: tokio_util::sync::CancellationToken,
    on_ready: Box<dyn FnOnce() + Send>,
) -> anyhow::Result<()> {
    use kb_eval::infer::{guard, handlers, state};

    // Derive has_auth from the RESOLVED key (a named-but-empty --api-key-file must not count as
    // auth, and resolving it here also fails fast on an empty key-file).
    let has_auth = args.effective_auth()?.is_some();
    // TLS: Some only when both cert+key are given (bail on only one). A TLS bind is
    // auth-at-transport, so it satisfies the non-loopback interlock the same way an api-key does.
    let tls = args
        .tls_paths()?
        .map(|(cert, key, client_ca)| glossa::tls::TlsFiles {
            cert,
            key,
            client_ca,
        });
    anyhow::ensure!(
        guard::interlock_ok(&args.bind, has_auth, tls.is_some(), args.insecure),
        "refusing non-loopback bind {} without authentication or TLS (use --insecure to override)",
        args.bind
    );

    let bind = args.bind.clone();
    let scheme = if tls.is_some() { "https" } else { "http" };
    let max_body_bytes = args.max_body_bytes;
    let no_cap = args.max_concurrency.is_none();
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        // Resolve dirs (download if needed) and BIND before loading, so `/health` serves 503 while
        // the models load rather than refusing connections (k8s-style readiness).
        let st = std::sync::Arc::new(state::build_state(&args)?);
        let listener = tokio::net::TcpListener::bind(&bind).await?;
        println!("kbi binding {scheme}://{bind} — loading models (health = 503 until ready)…");
        if no_cap {
            eprintln!("note: no --max-concurrency cap; requests queue on the model session under load (no 429 shed).");
        }

        // Shutdown signals → cancel. Ctrl-C (foreground) always; SIGTERM (systemd stop) on unix.
        // Harmless under the Windows SCM, where the control handler already drives `cancel`.
        {
            let c = cancel.clone();
            tokio::spawn(async move {
                let _ = tokio::signal::ctrl_c().await;
                c.cancel();
            });
        }
        #[cfg(unix)]
        {
            let c = cancel.clone();
            tokio::spawn(async move {
                if let Ok(mut term) = tokio::signal::unix::signal(
                    tokio::signal::unix::SignalKind::terminate(),
                ) {
                    term.recv().await;
                    c.cancel();
                }
            });
        }

        // Report readiness NOW — right after bind, NOT after the (possibly long, download-included)
        // model warm-up: the process is up and accepting, and `/health` returns 503 until the models
        // load, so a supervisor's start doesn't hang on a cold download. Mirrors kb's post-bind READY
        // (main.rs `sdnotify::ready()` after bind). `on_ready` flips the Windows SCM to Running;
        // `sd_notify` READY satisfies a systemd Type=notify unit; the watchdog pings if configured.
        on_ready();
        glossa::sdnotify::ready();
        if let Some(usec) = glossa::sdnotify::watchdog_usec() {
            glossa::sdnotify::spawn_watchdog(usec, cancel.clone());
        }

        // Warm on a blocking task so the server is already accepting (and answering 503) during load.
        {
            let w = st.clone();
            let bind_msg = bind.clone();
            tokio::task::spawn_blocking(move || match w.warm() {
                Ok(()) => {
                    let nli_ep = w.nli_ep.lock().unwrap_or_else(|e| e.into_inner()).clone();
                    let rerank_ep = w.rerank_ep.lock().unwrap_or_else(|e| e.into_inner()).clone();
                    println!("kbi models loaded on {scheme}://{bind_msg}  (nli_ep={nli_ep:?} rerank_ep={rerank_ep:?})");
                    println!("  kbx nli set    --scorer http --endpoint {scheme}://{bind_msg}");
                    println!("  kbx rerank set --scorer http --endpoint {scheme}://{bind_msg}");
                }
                Err(e) => {
                    eprintln!("model load failed: {e}");
                    std::process::exit(1);
                }
            });
        }

        // CORS: permissive (Any origin, no credentials) when no origins are configured, else an
        // explicit allow-list.
        let cors = if args.cors_allow_origin.is_empty() {
            tower_http::cors::CorsLayer::permissive()
        } else {
            let origins: Vec<axum::http::HeaderValue> = args
                .cors_allow_origin
                .iter()
                .filter_map(|o| o.parse().ok())
                .collect();
            tower_http::cors::CorsLayer::new()
                .allow_origin(origins)
                .allow_methods(tower_http::cors::Any)
                .allow_headers(tower_http::cors::Any)
        };
        let timeout = tower_http::timeout::TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            std::time::Duration::from_secs(args.request_timeout_secs),
        );

        let app = handlers::router(st.clone())
            .layer(axum::middleware::from_fn_with_state(
                st.clone(),
                guard::auth_layer,
            ))
            .layer(axum::middleware::from_fn_with_state(
                st.clone(),
                guard::host_layer,
            ))
            .layer(cors)
            .layer(timeout)
            .layer(axum::extract::DefaultBodyLimit::max(max_body_bytes));

        match tls {
            Some(files) => {
                let reloadable = std::sync::Arc::new(glossa::tls::ReloadableTls::new(files)?);
                glossa::tls::spawn_reload_poll(reloadable.clone(), cancel.clone());
                glossa::tls::serve_tls(
                    listener,
                    app,
                    reloadable,
                    cancel,
                    glossa::tls::handshake_timeout_from_env(),
                    glossa::tls::max_handshakes_from_env(),
                    None,
                )
                .await?;
            }
            None => {
                axum::serve(listener, app)
                    .with_graceful_shutdown(async move { cancel.cancelled().await })
                    .await?;
            }
        }
        // Serve loop returned (cancel tripped): tell systemd we are stopping.
        glossa::sdnotify::stopping();
        anyhow::Ok(())
    })
}

#[cfg(not(any(
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
)))]
fn main() {
    eprintln!(
        "kbi needs an ORT engine feature — build with `--features nli-directml` \
         (self-contained) or `--features nli-cuda` (GPU)."
    );
    std::process::exit(2);
}

// Gated with the real CLI (compile-covered by CI's `--features "http-scorer nli-cuda" --all-targets`).
#[cfg(all(
    test,
    any(
        feature = "nli-directml",
        feature = "nli-coreml",
        feature = "nli-cuda",
        feature = "nli-rocm"
    )
))]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn bare_flags_serve_and_the_subcommands_are_service_and_fit() {
        // Bare flags => serve (no subcommand), proving serve is the default (decision B).
        let cli = Cli::try_parse_from(["kbi", "--bind", "0.0.0.0:9000"]).unwrap();
        assert!(cli.command.is_none());
        assert_eq!(cli.serve.bind, "0.0.0.0:9000");
        // `kbi service status <name>` => the service subcommand.
        let cli = Cli::try_parse_from(["kbi", "service", "status", "svc"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Service(InferServiceAction::Status(_)))
        ));
        // Mixing serve flags before the subcommand is REJECTED (args_conflicts_with_subcommands).
        assert!(
            Cli::try_parse_from(["kbi", "--bind", "0.0.0.0:9000", "service", "status", "svc"])
                .is_err()
        );
    }

    /// `fit` takes the serve flags AFTER the verb — before it they conflict with the subcommand —
    /// plus its own sweep knobs, and leaves the row length unset so it resolves to the model's max
    /// rather than to a guessed constant.
    #[test]
    fn fit_takes_the_serve_flags_after_the_verb_plus_its_own_knobs() {
        let cli = Cli::try_parse_from([
            "kbi",
            "fit",
            "--rerank-model-dir",
            "m",
            "--device",
            "cuda",
            "--max-rows",
            "16",
            "--repeats",
            "2",
        ])
        .unwrap();
        let Some(Command::Fit(o)) = cli.command else {
            panic!("`kbi fit` did not parse as the fit subcommand");
        };
        assert_eq!(o.max_rows, 16);
        assert_eq!(o.repeats, 2);
        assert!(o.seq.is_none(), "row length resolves at run time");
        assert_eq!(o.serve.device.as_deref(), Some("cuda"));
        assert_eq!(o.tolerance, kb_eval::fit::DEFAULT_TOLERANCE);
    }

    /// The startup form is a serve FLAG, not the verb: `kbi --fit` serves and fits on the way up.
    #[test]
    fn the_startup_fit_is_a_serve_flag() {
        let cli = Cli::try_parse_from([
            "kbi",
            "--rerank-model-dir",
            "m",
            "--fit",
            "--fit-max-rows",
            "8",
        ])
        .unwrap();
        assert!(cli.command.is_none());
        assert!(cli.serve.fit);
        assert_eq!(cli.serve.fit_max_rows, 8);
        assert!(
            !Cli::try_parse_from(["kbi", "--rerank-model-dir", "m"])
                .unwrap()
                .serve
                .fit,
            "fitting is opt-in: it costs the sweep's own resident peak"
        );
    }
}
