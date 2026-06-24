use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::time::Duration;

mod backend;
mod corpus;
mod dataset;
mod prep;
mod run;
mod score;
mod trace_read;

#[derive(Parser)]
#[command(name = "kb-eval", about = "glossa agent-eval harness")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run a benchmark and score it.
    Run {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long, value_enum)]
        backend: BackendKind,
        #[arg(long, default_value_t = 0)]
        limit: usize, // 0 = all
        #[arg(long, default_value = "kb")]
        kb_bin: String,
        #[arg(long, default_value = "eval-corpus")]
        work: PathBuf,
        /// Per-question timeout (seconds) for cli/openai backends.
        #[arg(long, default_value_t = 120)]
        timeout_secs: u64,
        /// MCP profile the agent's glossa server runs under.
        #[arg(long, default_value = "editor")]
        profile: String,
        /// Expose only search+read to the agent (graph/index hidden).
        #[arg(long = "no-graph")]
        no_graph: bool,
        // --- openai backend ---
        /// OpenAI-compatible base URL (LM Studio default).
        #[arg(long, default_value = "http://localhost:1234")]
        endpoint: String,
        /// Model id sent to the endpoint.
        #[arg(long, default_value = "local-model")]
        model: String,
        /// Env var to read the API key from (omit for keyless local servers).
        #[arg(long)]
        api_key_env: Option<String>,
        // --- cli backend ---
        /// CLI-agent command (must be an MCP client; default = claude).
        #[arg(long, default_value = "claude")]
        cli_cmd: String,
        /// CLI-agent arg template (repeatable; tokens {prompt}, {mcp_config}). Empty = claude preset.
        #[arg(long = "cli-arg")]
        cli_arg: Vec<String>,
    },
    /// Convert the HotpotQA abstracts tar.bz2 into a glossa-indexable markdown corpus.
    PrepFullwiki {
        /// Path to the `...-abstracts.tar.bz2` archive.
        #[arg(long)]
        archive: PathBuf,
        /// Output directory for the markdown shards.
        #[arg(long)]
        out: PathBuf,
        /// Only convert the first N shards (feasibility spike).
        #[arg(long)]
        max_shards: Option<usize>,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendKind { Mock, Cli, Openai }

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Run {
            dataset, backend, limit, kb_bin, work, timeout_secs, profile, no_graph,
            endpoint, model, api_key_env, cli_cmd, cli_arg,
        } => {
            use backend::AgentBackend;
            let timeout = Duration::from_secs(timeout_secs);
            let be: Box<dyn AgentBackend> = match backend {
                BackendKind::Mock => Box::new(backend::mock::MockBackend { canned: std::collections::HashMap::new() }),
                BackendKind::Openai => {
                    let api_key = api_key_env.and_then(|v| std::env::var(v).ok());
                    Box::new(backend::openai::OpenAiBackend { endpoint, model, api_key, timeout })
                }
                BackendKind::Cli => {
                    let args = if cli_arg.is_empty() { backend::cli::CliBackend::claude_preset() } else { cli_arg };
                    Box::new(backend::cli::CliBackend { command: cli_cmd, args, kb_bin: kb_bin.clone(), profile, no_graph, timeout })
                }
            };
            let name = format!("{backend:?}").to_lowercase();
            let report = run::run_eval(&dataset, be.as_ref(), &name, limit, &kb_bin, &work)?;
            let json_path = format!("eval-{}-{}.json", report.backend, glossa::trace::now_ms());
            std::fs::write(&json_path, serde_json::to_string_pretty(&report)?)?;
            println!(
                "backend={} questions={} EM={:.3} F1={:.3} retrieval_recall={:.3}\nwrote {}",
                report.backend, report.rows.len(), report.em_mean, report.f1_mean, report.recall_mean, json_path
            );
            Ok(())
        }
        Cmd::PrepFullwiki { archive, out, max_shards } => {
            let stats = prep::prep_fullwiki(&archive, &out, max_shards)?;
            println!("prep-fullwiki: {} shard(s), {} article(s) -> {}", stats.shards, stats.articles, out.display());
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_compiles() {
        assert!(true);
    }
}
