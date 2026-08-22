//! `kbx build` stage 1 — agentic STATED fact-extraction over a single document (Task 5 of the
//! `kbx build` pipeline). Later stages (cross-doc merge, generalization, ...) live alongside
//! this module as siblings once they land.
//!
//! This module also wires all four stages into `run_build` (Task 10) — the orchestrator behind
//! the `kbx build` subcommand.

pub mod candidates;
pub mod chunks;
pub mod extract;
pub mod finalize;
pub mod incremental;
pub mod judge;

pub use candidates::{candidate_pairs, CandidatePair};
pub use chunks::chunk_text;
pub use extract::{extract_doc, parse_and_validate_upsert, ExtractStats};
pub use finalize::finalize;
pub use incremental::{compute_delta, drop_doc_nodes, Delta};
pub use judge::{judge_pair, run_judge, JudgeStats};

use crate::checkpoint::Checkpoint;
use crate::lab::LabConfig;
use crate::workspace::KbxPaths;
use anyhow::{Context, Result};
use clap::ValueEnum;
use glossa::graph::ontology::Ontology;
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::IsTerminal;

/// Which stage(s) of the build pipeline a `kbx build` invocation should run. `All` (the clap
/// default) runs every stage, in pipeline order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BuildStage {
    Extract,
    Candidates,
    Judge,
    Finalize,
    All,
}

/// CLI-level options for `kbx build`, folded from `Cmd::Build`'s clap fields.
#[derive(Debug, Clone)]
pub struct BuildOpts {
    pub stage: BuildStage,
    /// Restrict extraction to a single document (its structural-graph `Document` node id, i.e.
    /// its corpus-relative path).
    pub doc: Option<String>,
    /// Only extract the first N enumerated documents.
    pub limit: Option<usize>,
    /// Placeholder for Task 12's incremental rebuild. Currently a NO-OP: this task's build is
    /// always a full run regardless of `--force`.
    pub force: bool,
    /// Skip units (documents for extract, pairs for judge) already recorded done in the build
    /// checkpoint.
    pub resume: bool,
    /// Never draw the progress bar, even on a TTY.
    pub no_progress: bool,
}

/// indicatif progress bar over `len` units — hidden when `no_progress` is set or stdout/stderr
/// isn't a TTY (mirrors `kbx eval`'s `show_progress` gate in `src/bin/kbx.rs`).
fn progress_bar(len: usize, no_progress: bool) -> ProgressBar {
    let show =
        !no_progress && std::io::stdout().is_terminal() && std::io::stderr().is_terminal();
    if !show {
        return ProgressBar::hidden();
    }
    let pb = ProgressBar::new(len as u64);
    pb.set_style(
        ProgressStyle::with_template("{msg} [{pos}/{len}] {bar:40.cyan/blue} {elapsed_precise}")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    pb
}

/// Corpus-relative paths of every structural `Document` node in the graph, sorted for
/// deterministic enumeration order (sqlite row order isn't guaranteed).
fn enumerate_docs(g: &GraphStore) -> Result<Vec<String>> {
    let mut docs: Vec<String> = g
        .all_nodes()
        .context("listing nodes to enumerate documents")?
        .into_iter()
        .filter(|n| n.node_type == "Document")
        .map(|n| n.id)
        .collect();
    docs.sort();
    Ok(docs)
}

/// Orchestrate the `kbx build` pipeline over the corpus at `paths.root`: extract -> candidates ->
/// judge -> finalize, running only the stage(s) `opts.stage` selects (`All` runs all four in
/// order). Execution order: resolve the ontology (never fails — defaults when no
/// `ontology.toml`), ensure the corpus is indexed (structural `Document`/`Section` nodes + chunks
/// that every stage depends on), open the build checkpoint, then run each selected stage in turn.
/// `lab.toml` and the `builder.md`/`bridge.md` prompts are LAZY-loaded, one per stage, only when
/// that stage is actually selected: Extract needs `lab` + `builder.md`; Judge needs `lab` +
/// `bridge.md`; Candidates and Finalize need neither. This means `--stage finalize` or
/// `--stage candidates` runs on an indexed corpus with no `.glossa/kbx/` prompt files present at
/// all — only Extract/Judge (which call a model) require a scaffolded workspace.
pub fn run_build(paths: KbxPaths, opts: BuildOpts) -> Result<()> {
    let ontology = Ontology::load_or_default(&paths.root);

    // Incremental — a no-op if the corpus is already fully indexed. Guarantees the structural
    // Document/Section nodes and chunks that extraction/candidates/judge all depend on exist.
    glossa::index::store::index_dir(&paths.root, false).context("indexing corpus")?;

    // Build state lives under a single, stable `runs/build/` dir: unlike `kbx eval`'s per-tag
    // run history, one corpus has exactly one in-progress build to resume/checkpoint.
    let run_dir = paths.runs.join("build");
    let cp = Checkpoint::open(&run_dir).context("open build checkpoint")?;

    let run_extract = matches!(opts.stage, BuildStage::All | BuildStage::Extract);
    let run_candidates_stage = matches!(opts.stage, BuildStage::All | BuildStage::Candidates);
    let run_judge_stage = matches!(opts.stage, BuildStage::All | BuildStage::Judge);
    let run_finalize_stage = matches!(opts.stage, BuildStage::All | BuildStage::Finalize);

    if run_extract {
        let lab = LabConfig::load_at(&paths.lab)
            .with_context(|| format!("loading {}", paths.lab.display()))?;
        let builder_md = std::fs::read_to_string(&paths.builder)
            .with_context(|| format!("reading {}", paths.builder.display()))?;

        // A fresh, short-lived handle: enumerate then drop before extract_doc opens its own
        // per-document GraphStore connection, so nothing else holds `graph.sqlite` open across
        // the whole (possibly slow, model-calling) extraction loop.
        let mut docs = {
            let g = GraphStore::open(&paths.root).context("open graph store to enumerate docs")?;
            enumerate_docs(&g)?
        };
        if let Some(d) = &opts.doc {
            docs.retain(|p| p == d);
        }
        if let Some(n) = opts.limit {
            docs.truncate(n);
        }

        let pb = progress_bar(docs.len(), opts.no_progress);
        pb.set_message("extract");
        let mut total = ExtractStats::default();
        for doc in &docs {
            let unit_id = format!("extract:{doc}");
            if opts.resume && cp.is_done(&unit_id) {
                pb.inc(1);
                continue;
            }
            let stats = extract_doc(&paths.root, &lab, &builder_md, &ontology, doc)
                .with_context(|| format!("extracting {doc}"))?;
            total.nodes += stats.nodes;
            total.mentions += stats.mentions;
            cp.mark(&unit_id, "done")?;
            pb.inc(1);
        }
        pb.finish_and_clear();
        println!(
            "extract: {} doc(s), {} node(s), {} mention edge(s)",
            docs.len(),
            total.nodes,
            total.mentions
        );
    }

    if run_candidates_stage || run_judge_stage {
        // Re-open fresh: extraction (if it ran above) wrote through its own connections.
        let g = GraphStore::open(&paths.root).context("open graph store for candidates/judge")?;
        let pairs = candidate_pairs(&g).context("computing cross-doc candidates")?;
        if run_candidates_stage {
            println!("candidates: {} cross-doc pair(s)", pairs.len());
        }

        if run_judge_stage {
            let lab = LabConfig::load_at(&paths.lab)
                .with_context(|| format!("loading {}", paths.lab.display()))?;
            let bridge_md = std::fs::read_to_string(&paths.bridge)
                .with_context(|| format!("reading {}", paths.bridge.display()))?;

            let idx = DocIndex::open_or_create(&paths.root).context("open doc index")?;
            let pb = progress_bar(pairs.len(), opts.no_progress);
            pb.set_message("judge");
            let stats = run_judge(&paths.root, &lab, &bridge_md, &g, &idx, &pairs, &cp, &pb)
                .context("judging candidate pairs")?;
            pb.finish_and_clear();
            println!(
                "judge: {} judged, {} linked, {} skipped (ambiguous spine relation)",
                stats.judged, stats.linked, stats.skipped_ambiguous
            );
        }
    }

    if run_finalize_stage {
        let summary = finalize(&paths.root).context("finalizing build")?;
        println!("{summary}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glossa::graph::store::{Edge, Node, Provenance};
    use glossa::graph::MENTIONS;

    /// A tiny indexed corpus with one grounded `Fact` node — deliberately WITHOUT scaffolding
    /// `.glossa/kbx/` (no lab.toml, no builder/bridge prompts). Since the lazy-load fix,
    /// Finalize/Candidates need neither, so this is exactly the fixture that proves it: only
    /// Extract/Judge (out of scope here — both need a live model endpoint) would fail against it.
    fn synthetic_corpus() -> (tempfile::TempDir, KbxPaths) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("d.md"), "# Title\n\nGrounded chunk body.\n").unwrap();
        let paths = KbxPaths::for_root(root.to_path_buf());
        assert!(
            !paths.kbx_dir.exists(),
            "fixture must have no .glossa/kbx workspace scaffolded"
        );
        glossa::index::store::index_dir(root, true).unwrap();

        let g = GraphStore::open(root).unwrap();
        let prov = Provenance {
            source_path: "d.md".into(),
            range: None,
            file_sig: None,
            origin: "agent".into(),
            confidence: 0.9,
            created_at: 1,
        };
        g.put_node(&Node {
            id: "f1".into(),
            node_type: "Fact".into(),
            label: "n1".into(),
            aliases: vec!["n1".into()],
            prov: prov.clone(),
        })
        .unwrap();
        g.put_edge(&Edge {
            from: "f1".into(),
            to: "d.md#1".into(),
            edge_type: MENTIONS.into(),
            prov,
        })
        .unwrap();
        drop(g);

        (dir, paths)
    }

    #[test]
    fn run_build_finalize_stage_runs_without_error() {
        // No `.glossa/kbx/` at all (see `synthetic_corpus`) — Finalize must not touch lab.toml/
        // builder.md/bridge.md, so this run must succeed anyway.
        let (_dir, paths) = synthetic_corpus();
        let opts = BuildOpts {
            stage: BuildStage::Finalize,
            doc: None,
            limit: None,
            force: false,
            resume: false,
            no_progress: true,
        };
        run_build(paths, opts).unwrap();
    }

    #[test]
    fn run_build_candidates_stage_computes_cross_doc_pairs() {
        // No `.glossa/kbx/` here either — Candidates must not touch lab.toml/builder.md/
        // bridge.md, so this run must succeed with none of them present.
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("a.md"), "# A\n\nAlpha body.\n").unwrap();
        std::fs::write(root.join("b.md"), "# B\n\nBeta body.\n").unwrap();
        let paths = KbxPaths::for_root(root.to_path_buf());
        assert!(!paths.kbx_dir.exists());
        glossa::index::store::index_dir(root, true).unwrap();

        let g = GraphStore::open(root).unwrap();
        let prov = |src: &str| Provenance {
            source_path: src.into(),
            range: None,
            file_sig: None,
            origin: "agent".into(),
            confidence: 0.9,
            created_at: 1,
        };
        g.put_node(&Node {
            id: "f1".into(),
            node_type: "Fact".into(),
            label: "shared".into(),
            aliases: vec!["shared".into()],
            prov: prov("a.md"),
        })
        .unwrap();
        g.put_edge(&Edge {
            from: "f1".into(),
            to: "a.md#1".into(),
            edge_type: MENTIONS.into(),
            prov: prov("a.md"),
        })
        .unwrap();
        g.put_node(&Node {
            id: "f2".into(),
            node_type: "Fact".into(),
            label: "shared".into(),
            aliases: vec!["shared".into()],
            prov: prov("b.md"),
        })
        .unwrap();
        g.put_edge(&Edge {
            from: "f2".into(),
            to: "b.md#1".into(),
            edge_type: MENTIONS.into(),
            prov: prov("b.md"),
        })
        .unwrap();
        drop(g);

        let opts = BuildOpts {
            stage: BuildStage::Candidates,
            doc: None,
            limit: None,
            force: false,
            resume: false,
            no_progress: true,
        };
        run_build(paths, opts).unwrap();

        // The stage itself only prints; verify the underlying mechanical pairing it drives
        // directly, over the same fixture.
        let g = GraphStore::open(root).unwrap();
        let pairs = candidate_pairs(&g).unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].entity, "shared");
    }
}
