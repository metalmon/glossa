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

pub use candidates::{candidate_groups, CandidateGroup};
pub use chunks::chunk_text;
pub use extract::{extract_doc, ExtractStats};
pub use finalize::finalize;
pub use incremental::{compute_delta, drop_doc_nodes, Delta};
pub use judge::{judge_group, run_judge, JudgeStats};

use crate::checkpoint::Checkpoint;
use crate::lab::LabConfig;
use crate::workspace::KbxPaths;
use anyhow::{Context, Result};
use clap::ValueEnum;
use glossa::graph::ontology::Ontology;
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashSet;
use std::io::IsTerminal;
use std::path::Path;

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
    /// Bypass the incremental extract-list narrowing: extract EVERY enumerated document (not
    /// just NEW/CHANGED) and clear the run's judge checkpoint first, so every candidate pair is
    /// re-judged too — a true full rebuild. Gone-doc nodes are still dropped either way (see
    /// `run_build`'s extract stage) since a gone doc never comes back.
    pub force: bool,
    /// Skip units (documents for extract, pairs for judge) already recorded done in the build
    /// checkpoint.
    pub resume: bool,
    /// Never draw the progress bar, even on a TTY.
    pub no_progress: bool,
    /// Size guard (stage 3, prompt-fit not recall): cap the number of facts fed to a single
    /// group-judge model call. A group whose entity has more member facts than this is judged on
    /// only the first `bridge_max_facts` (deterministic order), with the truncation logged — see
    /// `judge::cap_facts`. `0` is treated as "no cap".
    pub bridge_max_facts: usize,
    /// Feed the images `read` returns (page rasters / embedded figures) to the extraction model
    /// as vision input, so scanned/image-only content can still yield grounded facts. OFF by
    /// default: the text-only extraction path stays byte-identical to today, and images are large
    /// on the wire. Threaded straight into `extract_doc`.
    pub vision: bool,
    /// Sampling temperature for the extract-stage model call. Higher values trade determinism for
    /// recall breadth over ambiguous/underspecified passages.
    pub build_temp: f64,
    /// Number of chunks folded into a single extract-stage model call (prompt-fit knob, mirrors
    /// `bridge_max_facts` for the judge stage).
    pub chunks_per_round: usize,
}

impl Default for BuildOpts {
    fn default() -> Self {
        BuildOpts {
            stage: BuildStage::All,
            doc: None,
            limit: None,
            force: false,
            resume: false,
            no_progress: false,
            bridge_max_facts: 0,
            vision: false,
            build_temp: 0.8,
            chunks_per_round: 3,
        }
    }
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

/// How much of a `run_build` pass actually did — the outcome of the incremental gate plus the
/// judge stage, inspectable by callers (tests, the CLI summary) instead of only printed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BuildReport {
    /// Corpus-relative paths of every document `extract_doc` actually ran over this pass, in
    /// enumeration order. Empty when the Extract stage didn't run at all (e.g. `--stage judge`).
    pub docs_extracted: Vec<String>,
    /// Candidate groups the Judge stage actually sent to the model this pass (mirrors
    /// `JudgeStats::judged`; a group already checkpointed from a prior run doesn't count). Zero
    /// when the Judge stage didn't run at all.
    pub groups_judged: usize,
}

/// Model-free selection of which docs `run_build`'s extract stage processes this pass:
/// `force` bypasses the delta entirely — every doc in `all_docs`; otherwise NEW ∪ CHANGED from
/// `delta`, restricted to (and order-preserving from) `all_docs` so a doc the delta names but
/// `all_docs` no longer enumerates can't sneak into the extract list. Pure and model-free by
/// design — the one piece of the incremental-gate logic this task can unit-test without a live
/// model (`--stage extract` itself cannot be).
pub fn plan_extract(delta: &Delta, force: bool, all_docs: &[String]) -> Vec<String> {
    if force {
        return all_docs.to_vec();
    }
    let wanted: HashSet<&String> = delta.new.iter().chain(delta.changed.iter()).collect();
    all_docs.iter().filter(|d| wanted.contains(d)).cloned().collect()
}

/// Which of `delta.changed`'s docs should have their stale reasoning nodes dropped this run:
/// only those actually present in `final_extract` (the fully-narrowed, post `--doc`/`--limit`
/// extract list). A changed doc NOT in `final_extract` keeps its nodes — dropping them without a
/// same-run re-extraction to replace them would silently orphan facts until some later,
/// unrestricted run picks the doc back up (it stays `changed` until then, never lost). Model-free
/// and pure, like `plan_extract`, so the drop-scoping decision is unit-testable on its own.
pub fn changed_docs_to_drop(delta: &Delta, final_extract: &[String]) -> Vec<String> {
    let extract_set: HashSet<&String> = final_extract.iter().collect();
    delta
        .changed
        .iter()
        .filter(|d| extract_set.contains(d))
        .cloned()
        .collect()
}

/// Fold every non-alphanumeric char to `_` — mirrors `sanitize_id`'s own cleaning step (minus its
/// trailing hash), so a raw node/doc id can be matched against an already-sanitized checkpoint
/// filename without needing to un-sanitize the filename (impossible in general: sanitizing is
/// lossy, several distinct raw ids can fold to the same cleaned string before the hash tells them
/// apart).
fn fold_for_match(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// Invalidate the checkpoint marks that a `doc`'s dropped reasoning nodes made stale: `doc`'s own
/// `extract:{doc}` mark (so a `--resume` run re-extracts it instead of silently skipping it and
/// leaving it with ZERO reasoning nodes), plus every stale `judge:*` mark. Group-judge checkpoint
/// marks are keyed by ENTITY (`judge:group:<entity>`, sanitized to `judge_group_...`), not by
/// member node id, so — unlike the retired pairwise marks (`judge:{a}#{b}`, whose filename embeds
/// both ids) — a dropped node id can't be matched back to the group mark that covered it. Rather
/// than track which entities a dropped node belonged to, every `judge_group_*` mark is invalidated
/// whenever ANY node was dropped for this doc: coarser than strictly necessary (some unaffected
/// groups get re-judged too), but safe — it never UNDER-invalidates, only over-invalidates, and a
/// re-judged group with no real change still ends up with the same "no links"/linked outcome.
/// (The `judge_` id-substring match below is kept for defense in depth in case any legacy pairwise
/// marks are still on disk from a build checkpoint that predates this redesign.) File-system-only,
/// no model/network — safe to call unconditionally, whether or not any mark actually existed.
fn invalidate_marks_for(cp: &Checkpoint, doc: &str, dropped_ids: &[String]) -> Result<()> {
    cp.remove(&format!("extract:{doc}"))
        .with_context(|| format!("removing extract:{doc} checkpoint mark"))?;
    if dropped_ids.is_empty() {
        return Ok(());
    }
    let folded: Vec<String> = dropped_ids.iter().map(|id| fold_for_match(id)).collect();
    for name in cp.done_ids() {
        let legacy_pair_match =
            name.starts_with("judge_") && folded.iter().any(|f| name.contains(f.as_str()));
        let group_match = name.starts_with("judge_group_");
        if legacy_pair_match || group_match {
            cp.remove_raw(&name)
                .with_context(|| format!("removing stale judge checkpoint mark {name}"))?;
        }
    }
    Ok(())
}

/// Clear a build run's persistent checkpoint (`run_dir/done/`) — `--force`'s "true full rebuild"
/// half: without this, the judge stage's `judge:{a}#{b}` marks from a prior run would keep every
/// already-judged pair skipped even though the caller asked to redo everything. A no-op when the
/// checkpoint dir doesn't exist yet (first-ever build).
fn clear_checkpoint(run_dir: &Path) -> Result<()> {
    let done_dir = run_dir.join("done");
    if done_dir.exists() {
        std::fs::remove_dir_all(&done_dir)
            .with_context(|| format!("clearing checkpoint {}", done_dir.display()))?;
    }
    Ok(())
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
pub fn run_build(paths: KbxPaths, opts: BuildOpts) -> Result<BuildReport> {
    let ontology = Ontology::load_or_default(&paths.root);

    // Incremental — a no-op if the corpus is already fully indexed. Guarantees the structural
    // Document/Section nodes and chunks that extraction/candidates/judge all depend on exist.
    glossa::index::store::index_dir(&paths.root, false).context("indexing corpus")?;

    // Build state lives under a single, stable `runs/build/` dir: unlike `kbx eval`'s per-tag
    // run history, one corpus has exactly one in-progress build to resume/checkpoint.
    let run_dir = paths.runs.join("build");
    // `--force` = a true full rebuild: clear the checkpoint BEFORE it's (re)opened below, so
    // every `judge:{a}#{b}` (and `extract:{doc}`) mark from a prior run is gone and nothing gets
    // skipped.
    if opts.force {
        clear_checkpoint(&run_dir).context("clearing build checkpoint for --force")?;
    }
    let cp = Checkpoint::open(&run_dir).context("open build checkpoint")?;
    let mut report = BuildReport::default();

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

        // Incremental gate: compute the new/changed/gone delta against the reasoning graph
        // already built, narrow `docs` to the FINAL extract list (NEW ∪ CHANGED, or every
        // enumerated doc under `--force`) via `plan_extract`, THEN apply `--doc`/`--limit` — in
        // that order, so the drop step below (which must match what's actually about to be
        // re-extracted) sees the fully-narrowed list, not the pre-`--doc`/`--limit` one.
        let idx = DocIndex::open_or_create(&paths.root).context("open doc index for delta")?;
        let g = GraphStore::open(&paths.root).context("open graph store for delta/drop")?;
        let delta = compute_delta(&paths.root, &idx, &g)?;

        docs = plan_extract(&delta, opts.force, &docs);
        if let Some(d) = &opts.doc {
            docs.retain(|p| p == d);
        }
        if let Some(n) = opts.limit {
            docs.truncate(n);
        }

        // GONE docs' nodes are dropped ALWAYS — unconditionally, regardless of `--force` and
        // regardless of `--doc`/`--limit` narrowing. A gone doc never comes back, so there is no
        // "wait for a same-run re-extraction" concern the way there is for CHANGED; leaving its
        // orphaned nodes in place would make `--force` a lie ("full rebuild" with stragglers).
        //
        // CHANGED docs' nodes are dropped ONLY for docs still in the final, narrowed `docs` list:
        // dropping a changed doc's nodes without re-extracting it in this SAME run would silently
        // orphan its facts (e.g. `--doc a.md` while b.md also changed must leave b.md's nodes
        // alone — b.md just stays CHANGED for the next, unrestricted run to pick up). Under
        // `--force`, `docs` is every enumerated doc, so every changed doc qualifies and gets
        // dropped+re-extracted, which is the correct full-rebuild behavior.
        //
        // Either way, dropping a doc's nodes invalidates any checkpoint mark that recorded work
        // now undone: the doc's own `extract:{doc}` mark (so `--resume` doesn't skip re-extracting
        // it and leave it with ZERO reasoning nodes) and every `judge:{a}#{b}` mark referencing a
        // dropped node id (its bridge edge was cascade-deleted with the node, so the checkpoint
        // mark is the only stale trace left — leaving it would suppress a needed re-judge).
        for doc in &delta.gone {
            let dropped_ids = drop_doc_nodes(&g, doc)
                .with_context(|| format!("dropping gone-doc reasoning nodes for {doc}"))?;
            invalidate_marks_for(&cp, doc, &dropped_ids)
                .with_context(|| format!("invalidating checkpoint marks for {doc}"))?;
        }
        for doc in changed_docs_to_drop(&delta, &docs) {
            let dropped_ids = drop_doc_nodes(&g, &doc)
                .with_context(|| format!("dropping stale reasoning nodes for {doc}"))?;
            invalidate_marks_for(&cp, &doc, &dropped_ids)
                .with_context(|| format!("invalidating checkpoint marks for {doc}"))?;
        }
        drop(g);

        let pb = progress_bar(docs.len(), opts.no_progress);
        pb.set_message("extract");
        let mut total = ExtractStats::default();
        for doc in &docs {
            let unit_id = format!("extract:{doc}");
            if opts.resume && cp.is_done(&unit_id) {
                pb.inc(1);
                continue;
            }
            let stats = extract_doc(
                &paths.root,
                &lab,
                &builder_md,
                &ontology,
                doc,
                opts.build_temp,
                opts.chunks_per_round,
                opts.vision,
            )
            .with_context(|| format!("extracting {doc}"))?;
            total.nodes += stats.nodes;
            total.mentions += stats.mentions;
            cp.mark(&unit_id, "done")?;
            report.docs_extracted.push(doc.clone());
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
        let groups = candidate_groups(&g).context("computing cross-doc candidate groups")?;
        if run_candidates_stage {
            let total_facts: usize = groups.iter().map(|gr| gr.members.len()).sum();
            println!(
                "candidates: {} cross-doc group(s), {} fact(s)",
                groups.len(),
                total_facts
            );
        }

        if run_judge_stage {
            let lab = LabConfig::load_at(&paths.lab)
                .with_context(|| format!("loading {}", paths.lab.display()))?;
            let bridge_md = std::fs::read_to_string(&paths.bridge)
                .with_context(|| format!("reading {}", paths.bridge.display()))?;

            let idx = DocIndex::open_or_create(&paths.root).context("open doc index")?;
            let pb = progress_bar(groups.len(), opts.no_progress);
            pb.set_message("judge");
            let stats = run_judge(
                &paths.root,
                &lab,
                &bridge_md,
                &g,
                &idx,
                &groups,
                &cp,
                &pb,
                opts.bridge_max_facts,
            )
            .context("judging candidate groups")?;
            pb.finish_and_clear();
            println!("judge: {} group(s) judged, {} link(s)", stats.judged, stats.linked);
            report.groups_judged = stats.judged;
        }
    }

    if run_finalize_stage {
        let summary = finalize(&paths.root).context("finalizing build")?;
        println!("{summary}");
    }

    Ok(report)
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
    fn build_opts_defaults() {
        let o = BuildOpts::default();
        assert_eq!(o.chunks_per_round, 3);
        assert!((o.build_temp - 0.8).abs() < 1e-9);
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
            bridge_max_facts: 40,
            vision: false,
            build_temp: 0.8,
            chunks_per_round: 3,
        };
        run_build(paths, opts).unwrap();
    }

    #[test]
    fn run_build_candidates_stage_computes_cross_doc_groups() {
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
            bridge_max_facts: 40,
            vision: false,
            build_temp: 0.8,
            chunks_per_round: 3,
        };
        run_build(paths, opts).unwrap();

        // The stage itself only prints; verify the underlying mechanical grouping it drives
        // directly, over the same fixture.
        let g = GraphStore::open(root).unwrap();
        let groups = candidate_groups(&g).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].entity, "shared");
    }

    /// `--stage extract` calls a live model, so `run_build` itself can't be unit-tested for the
    /// incremental gate. `plan_extract` is the pure, model-free decision it delegates to —
    /// exercised directly here per Task 12's test ruling. Case: a synthetic delta with one
    /// CHANGED doc selects only that doc, dropping the unchanged one.
    #[test]
    fn plan_extract_selects_changed_only() {
        let delta = Delta {
            new: vec![],
            changed: vec!["b.md".to_string()],
            gone: vec![],
        };
        let all_docs = vec!["a.md".to_string(), "b.md".to_string()];
        assert_eq!(plan_extract(&delta, false, &all_docs), vec!["b.md".to_string()]);
    }

    /// NEW docs (not just CHANGED) are included in the non-force selection.
    #[test]
    fn plan_extract_includes_new_docs() {
        let delta = Delta {
            new: vec!["c.md".to_string()],
            changed: vec!["b.md".to_string()],
            gone: vec![],
        };
        let all_docs = vec!["a.md".to_string(), "b.md".to_string(), "c.md".to_string()];
        assert_eq!(
            plan_extract(&delta, false, &all_docs),
            vec!["b.md".to_string(), "c.md".to_string()]
        );
    }

    /// `--force` bypasses the delta entirely: every enumerated doc, even ones the delta doesn't
    /// name as new/changed at all (a fully unchanged corpus still gets a full re-extract).
    #[test]
    fn plan_extract_force_selects_all() {
        let delta = Delta::default(); // nothing new/changed/gone
        let all_docs = vec!["a.md".to_string(), "b.md".to_string()];
        assert_eq!(plan_extract(&delta, true, &all_docs), all_docs);
    }

    /// `--force`'s other half: clearing the persistent checkpoint so a pair judged in a prior run
    /// isn't silently skipped. `clear_checkpoint` must make a previously-marked pair report
    /// `!is_done` again on the next `Checkpoint::open`.
    #[test]
    fn force_clears_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let run_dir = dir.path().join("runs").join("build");
        let cp = Checkpoint::open(&run_dir).unwrap();
        cp.mark("judge:a#b", "yes").unwrap();
        assert!(cp.is_done("judge:a#b"));

        clear_checkpoint(&run_dir).unwrap();

        let cp2 = Checkpoint::open(&run_dir).unwrap();
        assert!(!cp2.is_done("judge:a#b"));
    }

    /// `clear_checkpoint` must be a no-op (not an error) when the run has never checkpointed
    /// anything yet — the first-ever `--force` build on a fresh corpus.
    #[test]
    fn clear_checkpoint_noop_when_absent() {
        let dir = tempfile::tempdir().unwrap();
        let run_dir = dir.path().join("runs").join("build");
        assert!(!run_dir.exists());
        clear_checkpoint(&run_dir).unwrap();
    }

    // ---- Fix round 1 (drop-scoping bugs found in review) ----

    /// Pure unit test for `changed_docs_to_drop`: a doc excluded from `final_extract` (e.g. by
    /// `--doc`/`--limit`) must NOT be selected for dropping, even though it's in `delta.changed`
    /// — this is the CRITICAL bug's fix (dropping without a same-run re-extraction silently
    /// orphans facts).
    #[test]
    fn changed_docs_to_drop_excludes_doc_not_in_final_extract() {
        let delta = Delta {
            new: vec![],
            changed: vec!["a.md".to_string(), "b.md".to_string()],
            gone: vec![],
        };
        let final_extract = vec!["b.md".to_string()]; // a.md was excluded by --doc/--limit
        assert_eq!(
            changed_docs_to_drop(&delta, &final_extract),
            vec!["b.md".to_string()]
        );
    }

    /// Under `--force`, `final_extract` is every enumerated doc, so every changed doc qualifies
    /// for dropping (it's about to be re-extracted this same run).
    #[test]
    fn changed_docs_to_drop_includes_all_changed_when_final_extract_is_everything() {
        let delta = Delta {
            new: vec![],
            changed: vec!["a.md".to_string(), "b.md".to_string()],
            gone: vec![],
        };
        let final_extract = vec!["a.md".to_string(), "b.md".to_string(), "c.md".to_string()];
        assert_eq!(
            changed_docs_to_drop(&delta, &final_extract),
            vec!["a.md".to_string(), "b.md".to_string()]
        );
    }

    /// A minimal `.glossa/kbx/lab.toml` + `builder.md` scaffold: just enough for `run_build`'s
    /// Extract stage to load its lab config/prompt. No model call happens unless `extract_doc` is
    /// actually invoked, which the regression tests below avoid by narrowing `--doc` to a path
    /// that matches no enumerated document — the final extract list ends up empty, so the loop
    /// body (the only thing that would call a live model) never runs, while the delta/drop wiring
    /// this fix touches still executes for real inside `run_build` itself.
    fn scaffold_lab_and_builder(paths: &KbxPaths) {
        std::fs::create_dir_all(&paths.kbx_dir).unwrap();
        std::fs::write(
            &paths.lab,
            "[model]\nendpoint = \"http://unused\"\nmodel = \"unused\"\n",
        )
        .unwrap();
        std::fs::write(&paths.builder, "unused builder prompt").unwrap();
    }

    /// Regression fixture: a.md's reasoning node carries a current `file_sig` (unchanged), b.md's
    /// is stale (changed), and a third node is grounded only to a doc that was never indexed
    /// (gone) — mirrors `incremental.rs`'s own `unchanged_new_changed_gone_delta` fixture style.
    fn delta_regression_corpus() -> (tempfile::TempDir, KbxPaths) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("a.md"), "# A\n\nAlpha body.\n").unwrap();
        std::fs::write(root.join("b.md"), "# B\n\nBeta body.\n").unwrap();
        glossa::index::store::index_dir(root, true).unwrap();

        let paths = KbxPaths::for_root(root.to_path_buf());
        scaffold_lab_and_builder(&paths);

        let prov = |src: &str, file_sig: Option<glossa::index::manifest::FileSig>| Provenance {
            source_path: src.into(),
            range: None,
            file_sig,
            origin: "agent".into(),
            confidence: 0.9,
            created_at: 1,
        };

        let g = GraphStore::open(root).unwrap();

        let sig_a = glossa::index::store::file_sig(&root.join("a.md")).unwrap();
        g.put_node(&Node {
            id: "f_a".into(),
            node_type: "Fact".into(),
            label: "f_a".into(),
            aliases: vec!["f_a".into()],
            prov: prov("a.md", Some(sig_a)),
        })
        .unwrap();
        g.put_edge(&Edge {
            from: "f_a".into(),
            to: "a.md#1".into(),
            edge_type: MENTIONS.into(),
            prov: prov("a.md", Some(sig_a)),
        })
        .unwrap();

        let stale_sig = glossa::index::manifest::FileSig {
            mtime_secs: 1,
            size: 999_999,
        };
        g.put_node(&Node {
            id: "f_b".into(),
            node_type: "Fact".into(),
            label: "f_b".into(),
            aliases: vec!["f_b".into()],
            prov: prov("b.md", Some(stale_sig)),
        })
        .unwrap();
        g.put_edge(&Edge {
            from: "f_b".into(),
            to: "b.md#1".into(),
            edge_type: MENTIONS.into(),
            prov: prov("b.md", Some(stale_sig)),
        })
        .unwrap();

        g.put_node(&Node {
            id: "f_gone".into(),
            node_type: "Fact".into(),
            label: "f_gone".into(),
            aliases: vec!["f_gone".into()],
            prov: prov("doc_gone.md", None),
        })
        .unwrap();
        g.put_edge(&Edge {
            from: "f_gone".into(),
            to: "doc_gone.md#1".into(),
            edge_type: MENTIONS.into(),
            prov: prov("doc_gone.md", None),
        })
        .unwrap();
        drop(g);

        (dir, paths)
    }

    /// CRITICAL bug regression: `--doc` (or `--limit`) narrowing the extract list to exclude a
    /// CHANGED doc must NOT drop that doc's reasoning nodes this run.
    #[test]
    fn run_build_extract_keeps_changed_doc_nodes_when_doc_flag_excludes_it() {
        let (_dir, paths) = delta_regression_corpus();
        let root = paths.root.clone();

        let opts = BuildOpts {
            stage: BuildStage::Extract,
            doc: Some("nonexistent.md".to_string()),
            limit: None,
            force: false,
            resume: false,
            no_progress: true,
            bridge_max_facts: 40,
            vision: false,
            build_temp: 0.8,
            chunks_per_round: 3,
        };
        let report = run_build(paths, opts).unwrap();
        assert!(report.docs_extracted.is_empty());

        let g = GraphStore::open(&root).unwrap();
        assert!(
            g.get_node("f_b").unwrap().is_some(),
            "changed doc's node must survive when --doc excludes it from this run's extract list"
        );
        assert!(g.get_node("f_a").unwrap().is_some());
    }

    /// IMPORTANT bug regression: GONE docs' nodes are dropped unconditionally — even under
    /// `--force` (which used to skip the whole delta/drop step) and even when `--doc` narrows the
    /// extract list to nothing.
    #[test]
    fn run_build_force_still_drops_gone_doc_nodes_even_when_doc_flag_narrows_to_nothing() {
        let (_dir, paths) = delta_regression_corpus();
        let root = paths.root.clone();

        let opts = BuildOpts {
            stage: BuildStage::Extract,
            doc: Some("nonexistent.md".to_string()),
            limit: None,
            force: true,
            resume: false,
            no_progress: true,
            bridge_max_facts: 40,
            vision: false,
            build_temp: 0.8,
            chunks_per_round: 3,
        };
        let report = run_build(paths, opts).unwrap();
        assert!(report.docs_extracted.is_empty());

        let g = GraphStore::open(&root).unwrap();
        assert!(
            g.get_node("f_gone").unwrap().is_none(),
            "gone doc's node must be dropped even under --force + --doc narrowing to nothing"
        );
    }

    /// Same GONE-always guarantee without `--force`, confirming it isn't force-specific.
    #[test]
    fn run_build_drops_gone_doc_nodes_by_default_regardless_of_doc_flag() {
        let (_dir, paths) = delta_regression_corpus();
        let root = paths.root.clone();

        let opts = BuildOpts {
            stage: BuildStage::Extract,
            doc: Some("nonexistent.md".to_string()),
            limit: None,
            force: false,
            resume: false,
            no_progress: true,
            bridge_max_facts: 40,
            vision: false,
            build_temp: 0.8,
            chunks_per_round: 3,
        };
        run_build(paths, opts).unwrap();

        let g = GraphStore::open(&root).unwrap();
        assert!(g.get_node("f_gone").unwrap().is_none());
    }

    // ---- Fix round 2 (checkpoint marks left stale after a node drop) ----

    /// The exact scenario from review: `extract:a.md` and `judge:fA#fB` are marked done from a
    /// prior run; a.md's nodes are dropped this run (it owns `fA`). Both marks must be
    /// invalidated — `extract:a.md` so `--resume` re-extracts a.md instead of leaving it with
    /// zero reasoning nodes, `judge:fA#fB` so the pair (whose node `fA` no longer exists) gets
    /// re-judged rather than silently losing its bridge. An unrelated `judge:fX#fY` mark, which
    /// references neither dropped id, must survive untouched.
    #[test]
    fn invalidate_marks_for_removes_extract_and_referencing_judge_marks() {
        let dir = tempfile::tempdir().unwrap();
        let cp = Checkpoint::open(&dir.path().join("runs").join("build")).unwrap();
        cp.mark("extract:a.md", "done").unwrap();
        cp.mark("judge:fA#fB", "yes").unwrap();
        cp.mark("judge:fX#fY", "no").unwrap();

        invalidate_marks_for(&cp, "a.md", &["fA".to_string()]).unwrap();

        assert!(!cp.is_done("extract:a.md"));
        assert!(!cp.is_done("judge:fA#fB"));
        assert!(cp.is_done("judge:fX#fY"));
    }

    /// `invalidate_marks_for` must still remove the doc's own `extract:{doc}` mark even when no
    /// nodes were actually dropped for it (an empty `dropped_ids`) — e.g. a doc classified CHANGED
    /// whose reasoning nodes had already been removed by an earlier step in the same run.
    #[test]
    fn invalidate_marks_for_removes_extract_mark_even_with_no_dropped_ids() {
        let dir = tempfile::tempdir().unwrap();
        let cp = Checkpoint::open(&dir.path().join("runs").join("build")).unwrap();
        cp.mark("extract:a.md", "done").unwrap();

        invalidate_marks_for(&cp, "a.md", &[]).unwrap();

        assert!(!cp.is_done("extract:a.md"));
    }

    /// End-to-end through the real `run_build` wiring (not just the pure helper): a GONE doc's
    /// prior `extract:{doc}` mark and a `judge:*` mark referencing its dropped node must both be
    /// invalidated by an ordinary (non-`--force`) run. GONE docs are never in the final extract
    /// list, so this needs no live model — the extract loop body never runs for them.
    #[test]
    fn run_build_invalidates_checkpoint_marks_for_dropped_gone_doc_nodes() {
        let (_dir, paths) = delta_regression_corpus();
        let root = paths.root.clone();
        let run_dir = paths.runs.join("build");

        // Simulate marks left over from the run that originally extracted/judged doc_gone.md's
        // fact, before that file was deleted from the corpus.
        let cp = Checkpoint::open(&run_dir).unwrap();
        cp.mark("extract:doc_gone.md", "done").unwrap();
        cp.mark("judge:f_gone#f_a", "yes").unwrap();
        cp.mark("judge:unrelated_a#unrelated_b", "no").unwrap();
        drop(cp);

        let opts = BuildOpts {
            stage: BuildStage::Extract,
            doc: Some("nonexistent.md".to_string()),
            limit: None,
            force: false,
            resume: false,
            no_progress: true,
            bridge_max_facts: 40,
            vision: false,
            build_temp: 0.8,
            chunks_per_round: 3,
        };
        run_build(paths, opts).unwrap();

        let cp = Checkpoint::open(&run_dir).unwrap();
        assert!(
            !cp.is_done("extract:doc_gone.md"),
            "gone doc's stale extract mark must be invalidated"
        );
        assert!(
            !cp.is_done("judge:f_gone#f_a"),
            "judge mark referencing the dropped node must be invalidated"
        );
        assert!(
            cp.is_done("judge:unrelated_a#unrelated_b"),
            "an unrelated judge mark must survive"
        );

        let g = GraphStore::open(&root).unwrap();
        assert!(g.get_node("f_gone").unwrap().is_none());
    }
}
