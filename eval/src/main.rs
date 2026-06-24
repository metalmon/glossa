use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

mod backend;
mod corpus;
mod dataset;
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
        #[arg(long, default_value = "http://localhost:1234")]
        lmstudio_url: String,
        #[arg(long, default_value = "kb")]
        kb_bin: String,
        #[arg(long, default_value = "eval-corpus")]
        work: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendKind { Mock, Qwen, Claude }

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Run { dataset, backend, limit, lmstudio_url, kb_bin, work } => {
            use backend::AgentBackend;
            let be: Box<dyn AgentBackend> = match backend {
                BackendKind::Mock => Box::new(backend::mock::MockBackend { canned: std::collections::HashMap::new() }),
                BackendKind::Qwen => Box::new(backend::qwen::QwenBackend { url: lmstudio_url, model: "local-model".to_string() }),
                BackendKind::Claude => Box::new(backend::claude::ClaudeBackend { kb_bin: kb_bin.clone(), profile: "editor".to_string(), no_graph: false }),
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
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_compiles() {
        assert!(true);
    }
}
