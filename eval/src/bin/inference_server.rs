//! `inference-server` — serve the NLI + reranker cross-encoders over HTTP (Phase 1). The real
//! server needs an ORT engine feature (it loads `InProcessNli`/`InProcessReranker`); a build
//! without one prints how to build it. Runs in the foreground; Phase 2 adds `service install`.

#[cfg(any(
    feature = "nli",
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
    let has_auth = args.api_key.is_some() || args.api_key_file.is_some();
    anyhow::ensure!(
        guard::interlock_ok(&args.bind, has_auth, args.insecure),
        "refusing non-loopback bind {} without --api-key (use --insecure to override)",
        args.bind
    );

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let st = std::sync::Arc::new(state::build_state(&args)?);
        println!(
            "inference-server ready on http://{}  (nli_ep={:?} rerank_ep={:?})",
            args.bind, st.nli_ep, st.rerank_ep
        );
        println!("  kbx nli set    --scorer http --endpoint http://{}", args.bind);
        println!("  kbx rerank set --scorer http --endpoint http://{}", args.bind);
        let app = handlers::router(st.clone())
            .layer(axum::middleware::from_fn_with_state(
                st.clone(),
                guard::auth_layer,
            ))
            .layer(axum::extract::DefaultBodyLimit::max(args.max_body_bytes));
        let listener = tokio::net::TcpListener::bind(&args.bind).await?;
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await?;
        anyhow::Ok(())
    })
}

#[cfg(any(
    feature = "nli",
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
))]
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

#[cfg(not(any(
    feature = "nli",
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
