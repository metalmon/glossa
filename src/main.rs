use clap::{Parser, Subcommand};
use glossa::query::{compile, QueryOpts};
use glossa::search::search_chunks;
use glossa::walk::collect_chunks;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "kb", about = "File-First knowledge-base search (ripgrep syntax)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Search files. PATTERN uses ripgrep syntax.
    Search {
        /// Search pattern (ripgrep regex syntax).
        pattern: String,
        /// Directory to search.
        #[arg(default_value = ".")]
        path: PathBuf,
        /// Case-insensitive (rg -i).
        #[arg(short = 'i', long = "ignore-case")]
        ignore_case: bool,
        /// Match whole words (rg -w).
        #[arg(short = 'w', long = "word-regexp")]
        word: bool,
        /// Treat pattern as a literal string (rg -F).
        #[arg(short = 'F', long = "fixed-strings")]
        fixed: bool,
        /// Only search paths matching GLOB (rg -g).
        #[arg(short = 'g', long = "glob")]
        glob: Option<String>,
        /// Max number of hits.
        #[arg(long, default_value_t = 100)]
        limit: usize,
        /// Use the on-disk index for BM25-ranked, stemmed search (run `kb index` first).
        #[arg(long)]
        rank: bool,
    },
    /// Build or update the on-disk index for ranked search.
    Index {
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Rebuild the index from scratch.
    Reindex {
        #[arg(default_value = ".")]
        path: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Search { pattern, path, ignore_case, word, fixed, glob, limit, rank } => {
            if rank {
                let idx = glossa::index::store::DocIndex::open_or_create(&path)?;
                for h in idx.search(&pattern, limit)? {
                    println!("{}:{}: {}  [{:.3}]", h.path, h.location, h.snippet, h.score);
                }
                return Ok(());
            }
            let opts = QueryOpts {
                ignore_case,
                smart_case: !ignore_case, // rg smart-case default
                word,
                fixed,
            };
            let re = compile(&pattern, &opts)?;
            let chunks = collect_chunks(&path, glob.as_deref())?;
            for h in search_chunks(&chunks, &re, limit) {
                println!("{}:{}:{}: {}", h.doc_path.display(), h.location, h.line, h.snippet);
            }
            Ok(())
        }
        Cmd::Index { path } => {
            let stats = glossa::index::store::index_dir(&path, false)?;
            println!(
                "indexed: {} added, {} removed, {} unchanged",
                stats.added, stats.removed, stats.unchanged
            );
            Ok(())
        }
        Cmd::Reindex { path } => {
            let stats = glossa::index::store::index_dir(&path, true)?;
            println!("reindexed: {} files", stats.added);
            Ok(())
        }
    }
}
