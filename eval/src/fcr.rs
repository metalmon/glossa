//! Run-free retrieval quality metric: **Full Chain Retrieval (FCR)**.
//!
//! For each answerable case with gold source chunks, retrieve the top-K and ask: is the WHOLE gold
//! set present (full chain), and what fraction of it was retrieved (partial recall)? No reader, no
//! judge, no GPU — just the corpus index + the dataset golds. Aggregated by `hop_type`, this exposes
//! the "high partial recall but broken chain" failure mode that multi-hop retrieval hides behind
//! plain Recall@k. See docs: run-free retrieval diagnostic alongside `kbx eval run`.

use anyhow::Context;
use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

/// Whether the WHOLE `gold` set is contained in `retrieved`, and the fraction of `gold` retrieved.
/// Empty gold is vacuously fully-covered (the runner skips empty-gold cases before calling this).
pub fn chain_recall(gold: &HashSet<String>, retrieved: &HashSet<String>) -> (bool, f32) {
    if gold.is_empty() {
        return (true, 1.0);
    }
    let hit = gold.iter().filter(|g| retrieved.contains(*g)).count();
    (hit == gold.len(), hit as f32 / gold.len() as f32)
}

/// Chunk refs (`path#N`) named by a glossary rendering's `— read <ref>` grounding anchors — the
/// graph-retrieval counterpart of a BM25 hit list. Tolerant by design: on each line it takes the
/// text after the LAST `read ` when that text ends in `#<digits>`, so a stray "read" in a label
/// never yields a false ref.
pub fn extract_graph_refs(body: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in body.lines() {
        let Some(pos) = line.rfind("read ") else {
            continue;
        };
        let cand = line[pos + "read ".len()..].trim();
        let Some(h) = cand.rfind('#') else {
            continue;
        };
        let ord = &cand[h + 1..];
        if !ord.is_empty()
            && ord.chars().all(|c| c.is_ascii_digit())
            && !out.iter().any(|r| r == cand)
        {
            out.push(cand.to_string());
        }
    }
    out
}

/// FCR tallies for one `hop_type` bucket.
#[derive(Default, Clone)]
pub struct BucketFcr {
    pub n: usize,
    /// Cases whose entire gold chain was retrieved.
    pub full: usize,
    /// Sum of per-case gold-retrieved fractions (mean = `partial()`).
    pub partial_sum: f32,
}

impl BucketFcr {
    pub fn add(&mut self, full: bool, partial: f32) {
        self.n += 1;
        if full {
            self.full += 1;
        }
        self.partial_sum += partial;
    }
    pub fn fcr(&self) -> f32 {
        if self.n == 0 {
            0.0
        } else {
            self.full as f32 / self.n as f32
        }
    }
    pub fn partial(&self) -> f32 {
        if self.n == 0 {
            0.0
        } else {
            self.partial_sum / self.n as f32
        }
    }
}

/// FCR aggregated by `hop_type`.
#[derive(Default)]
pub struct FcrReport {
    pub by_hop: BTreeMap<String, BucketFcr>,
    pub skipped_no_gold: usize,
}

impl FcrReport {
    pub fn add(&mut self, hop: &str, full: bool, partial: f32) {
        let key = if hop.is_empty() { "(untyped)" } else { hop };
        self.by_hop
            .entry(key.to_string())
            .or_default()
            .add(full, partial);
    }
    pub fn overall(&self) -> BucketFcr {
        let mut t = BucketFcr::default();
        for b in self.by_hop.values() {
            t.n += b.n;
            t.full += b.full;
            t.partial_sum += b.partial_sum;
        }
        t
    }
    /// Human table: one line per hop_type plus an ALL row.
    pub fn render(&self, k: usize) -> String {
        let mut s =
            format!("FCR@{k} (whole gold chain retrieved) + partial recall, by hop_type:\n");
        let line = |name: &str, b: &BucketFcr| {
            format!(
                "  {name:<12} n={:<4} FCR={:5.1}%  partial-recall={:5.1}%\n",
                b.n,
                100.0 * b.fcr(),
                100.0 * b.partial()
            )
        };
        for (hop, b) in &self.by_hop {
            s.push_str(&line(hop, b));
        }
        s.push_str(&line("ALL", &self.overall()));
        s
    }
}

/// `kbx eval fcr` arguments.
#[derive(clap::Args, Debug)]
pub struct FcrArgs {
    /// Corpus root (kb-style PATH resolution).
    pub path: Option<PathBuf>,
    /// Override the workspace's default `dataset.toml`.
    #[arg(long)]
    pub dataset: Option<PathBuf>,
    /// Retrieve this many hits per question before checking gold coverage.
    #[arg(long, default_value_t = 20)]
    pub k: usize,
    /// Only score cases whose `tags` include this value.
    #[arg(long = "tag-filter")]
    pub tag_filter: Option<String>,
    /// Retrieval method to score: `search` (BM25 over the index) or `graph` (glossary over the
    /// reasoning graph). `graph` ignores `--k` (glossary returns its own bounded neighbourhood).
    #[arg(long, value_enum, default_value_t = Via::Search)]
    pub via: Via,
}

/// Which retrieval path `kbx eval fcr` measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Via {
    /// BM25 full-text search over the index.
    Search,
    /// `glossary` retrieval over the reasoning graph (grounded chunks named by its `— read` anchors).
    Graph,
}

/// `kbx eval fcr`: run-free Full Chain Retrieval over the corpus index + dataset golds. Opens the
/// index (like `eval calibrate`), retrieves top-K per answerable question via BM25
/// (`search_filtered`), and reports FCR + partial recall by `hop_type`. No reader/judge/GPU.
pub fn run_fcr(args: FcrArgs) -> anyhow::Result<()> {
    let paths = crate::workspace::resolve(args.path.clone());
    let dataset_path = args
        .dataset
        .clone()
        .unwrap_or_else(|| paths.dataset.clone());
    let text = std::fs::read_to_string(&dataset_path)
        .with_context(|| format!("reading dataset {}", dataset_path.display()))?;
    let golds = crate::dataset_toml::parse_dataset_toml(&text)
        .with_context(|| format!("parsing dataset {}", dataset_path.display()))?;
    let idx = glossa::index::store::DocIndex::open_or_create(&paths.root)
        .with_context(|| format!("opening index at {}", paths.root.display()))?;
    // Graph retrieval path (only opened for `--via graph`): reuse the reader's `glossary` over the
    // reasoning graph, with a disabled trace and default chain spec.
    let graph = if args.via == Via::Graph {
        Some(
            glossa::graph::store::GraphStore::open(&paths.root)
                .with_context(|| format!("opening graph at {}", paths.root.display()))?,
        )
    } else {
        None
    };
    let spec = glossa::tools::ChainSpec::default();
    let trace = glossa::trace::TraceLog::disabled();

    let mut report = FcrReport::default();
    for q in &golds {
        if !q.answerable {
            continue;
        }
        if let Some(tf) = &args.tag_filter {
            if !q.tags.iter().any(|t| t == tf) {
                continue;
            }
        }
        let gold: HashSet<String> = q.source.iter().cloned().collect();
        if gold.is_empty() {
            report.skipped_no_gold += 1;
            continue;
        }
        let retrieved: HashSet<String> = match &graph {
            Some(g) => {
                let body = glossa::tools::glossary_with_query(
                    &idx,
                    g,
                    &q.question,
                    Some(&q.question),
                    &spec,
                    &trace,
                    None,
                    None,
                    None,
                );
                extract_graph_refs(&body).into_iter().collect()
            }
            None => idx
                .search_filtered(&q.question, args.k, None, None, None)
                .unwrap_or_default()
                .iter()
                .map(|h| format!("{}#{}", h.path, h.ord))
                .collect(),
        };
        let (full, partial) = chain_recall(&gold, &retrieved);
        report.add(&q.hop_type, full, partial);
    }

    println!(
        "retrieval: {}",
        match args.via {
            Via::Search => "search (BM25 over the index)",
            Via::Graph => "graph (glossary over the reasoning graph)",
        }
    );
    print!("{}", report.render(args.k));
    if report.skipped_no_gold > 0 {
        println!(
            "({} answerable case(s) skipped: no gold source chunks to score against)",
            report.skipped_no_gold
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(v: &[&str]) -> HashSet<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn chain_recall_full_partial_none_vacuous() {
        // whole chain present -> full, 1.0
        assert_eq!(
            chain_recall(&set(&["a", "b"]), &set(&["a", "b", "c"])),
            (true, 1.0)
        );
        // 1 of 3 present -> not full, 1/3
        let (full, part) = chain_recall(&set(&["a", "b", "c"]), &set(&["a", "x"]));
        assert!(!full);
        assert!((part - 1.0 / 3.0).abs() < 1e-6, "part={part}");
        // none present -> not full, 0.0
        assert_eq!(chain_recall(&set(&["a"]), &set(&["x"])), (false, 0.0));
        // empty gold -> vacuously full
        assert_eq!(chain_recall(&set(&[]), &set(&["x"])), (true, 1.0));
    }

    #[test]
    fn extract_graph_refs_reads_grounding_anchors_only() {
        let body = "res:abc  [Resolution]  Foo   — read Runtime/CODESYS Redundancy.pdf#18\n\
                    task:def  [Task]  how to read the manual\n\
                    cause:xyz  [Cause]  Bar   — read plk/setup guide.pdf#644\n";
        let refs = extract_graph_refs(body);
        assert!(
            refs.contains(&"Runtime/CODESYS Redundancy.pdf#18".to_string()),
            "{refs:?}"
        );
        assert!(
            refs.contains(&"plk/setup guide.pdf#644".to_string()),
            "{refs:?}"
        );
        // the "how to read the manual" line has no `#N` ref -> not extracted
        assert_eq!(refs.len(), 2, "{refs:?}");
    }

    #[test]
    fn report_aggregates_by_hop_and_overall() {
        let mut r = FcrReport::default();
        r.add("multihop", false, 0.5);
        r.add("multihop", true, 1.0);
        r.add("lexical", true, 1.0);
        let mh = &r.by_hop["multihop"];
        assert_eq!((mh.n, mh.full), (2, 1));
        assert!((mh.fcr() - 0.5).abs() < 1e-6);
        assert!((mh.partial() - 0.75).abs() < 1e-6);
        let o = r.overall();
        assert_eq!((o.n, o.full), (3, 2));
        let table = r.render(20);
        assert!(table.contains("multihop") && table.contains("ALL"));
    }
}
