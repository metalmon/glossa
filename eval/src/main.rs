use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

mod backend;
mod corpus;
mod dataset;
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
        Cmd::Run { .. } => {
            println!("kb-eval: not yet implemented");
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
