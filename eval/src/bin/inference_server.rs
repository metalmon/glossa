//! `inference-server` — serve the NLI + reranker cross-encoders over HTTP (Phase 1). The real
//! server needs an ORT engine feature (it loads `InProcessNli`/`InProcessReranker`); a build
//! without one prints how to build it. Runs in the foreground; Phase 2 adds `service install`.

#[cfg(any(
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
))]
fn main() -> anyhow::Result<()> {
    use clap::Parser;
    use kb_eval::infer::{cli::ServeArgs, guard, handlers, state};

    #[derive(Parser)]
    #[command(name = "inference-server", version = glossa::version())]
    struct Cli {
        #[command(flatten)]
        serve: ServeArgs,
    }

    let args = Cli::parse().serve;
    // Derive has_auth from the RESOLVED key (a named-but-empty --api-key-file must not count as
    // auth, and resolving it here also fails fast on an empty key-file).
    let has_auth = args.effective_auth()?.is_some();
    // TLS: Some only when both cert+key are given (bail on only one). A TLS bind is
    // auth-at-transport, so it satisfies the non-loopback interlock the same way an api-key does.
    let tls = args.tls_files()?;
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
        println!("inference-server binding {scheme}://{bind} — loading models (health = 503 until ready)…");
        if no_cap {
            eprintln!("note: no --max-concurrency cap; requests queue on the model session under load (no 429 shed).");
        }

        // Warm on a blocking task so the server is already accepting (and answering 503) during load.
        {
            let w = st.clone();
            let bind_msg = bind.clone();
            tokio::task::spawn_blocking(move || match w.warm() {
                Ok(()) => {
                    let nli_ep = w.nli_ep.lock().unwrap_or_else(|e| e.into_inner()).clone();
                    let rerank_ep = w.rerank_ep.lock().unwrap_or_else(|e| e.into_inner()).clone();
                    println!("inference-server READY on {scheme}://{bind_msg}  (nli_ep={nli_ep:?} rerank_ep={rerank_ep:?})");
                    println!("  kbx nli set    --scorer http --endpoint {scheme}://{bind_msg}");
                    println!("  kbx rerank set --scorer http --endpoint {scheme}://{bind_msg}");
                }
                Err(e) => {
                    eprintln!("model load failed: {e}");
                    std::process::exit(1);
                }
            });
        }

        // One cancellation token drives graceful shutdown for both the HTTP and HTTPS serve loops
        // (and the TLS cert-reload poller); ctrl_c cancels it.
        let cancel = tokio_util::sync::CancellationToken::new();
        {
            let c = cancel.clone();
            tokio::spawn(async move {
                let _ = tokio::signal::ctrl_c().await;
                c.cancel();
            });
        }

        let app = handlers::router(st.clone())
            .layer(axum::middleware::from_fn_with_state(
                st.clone(),
                guard::auth_layer,
            ))
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
        "inference-server needs an ORT engine feature — build with `--features nli-directml` \
         (self-contained) or `--features nli-cuda` (GPU)."
    );
    std::process::exit(2);
}
