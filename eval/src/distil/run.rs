//! `kbx distil` pipeline orchestrator: resolve the workspace, build the seed pool from grounded
//! non-structural graph nodes, run `gen::generate_one` (generate + verify-gate) once per attempt,
//! and write the kept synthetic golds as `[[case]]` rows to `--out` — the SAME `dataset.toml`
//! shape `dataset_toml::parse_dataset_toml` reads back, so `kbx reason --gold <out>` and
//! `kbx eval --dataset <out>` consume it unchanged.
//!
//! Read-only on the graph: this module never calls `graph_upsert`. The only write anywhere in
//! `kbx distil` is the `--out` dataset file.

use crate::checkpoint::Checkpoint;
use crate::lab::LabConfig;
use crate::distil::densify::{densify_doc, DensifyStats};
use crate::distil::gen::{generate_one, GenOutcome, Seed};
use crate::workspace::{self, KbxPaths};
use anyhow::{bail, Context, Result};
use glossa::graph::ontology::Ontology;
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::IsTerminal;
use std::path::PathBuf;

/// CLI-level options for `kbx distil`, folded from the `kbx` binary's clap fields (mirrors
/// `reason::ReasonArgs`'s shape).
#[derive(Debug, Clone)]
pub struct DistilArgs {
    /// Number of synthetic golds to ATTEMPT — the gate may drop some; kept/dropped are reported.
    /// `None` when `--target` is used instead. When both are `None`, defaults to 1 attempt.
    pub count: Option<usize>,
    /// Keep generating until this many golds are KEPT, bounded by `max_attempts`. Takes precedence
    /// over `count` when set.
    pub target: Option<usize>,
    /// Attempt ceiling when `target` is set (default: `target * 4`, floored at `target`).
    pub max_attempts: Option<usize>,
    /// Dataset TOML to write (default `<kbx>/dataset.synthetic.toml`). Always overwritten whole.
    pub out: Option<PathBuf>,
    /// Restrict seeds to this node_type (default: the ontology's grounding-required types, or
    /// every non-structural declared type when none are marked `requires_grounding`).
    pub seed_type: Option<String>,
    /// Never draw the progress bar, even on a TTY.
    pub no_progress: bool,
    // --- `kbx distil densify` orchestrator fields (Task 3 of the densification plan; `kbx`'s
    // CLI wiring for these lands in Task 4) ---
    /// Restrict `run_densify` to a single document (its structural-graph `Document` node id, i.e.
    /// its corpus-relative path) — mirrors `BuildOpts::doc`/`ReasonArgs`' single-unit narrowing.
    pub doc: Option<String>,
    /// Clear this run's `distil:{doc}` checkpoint first — a true full rebuild of the densify pass
    /// (mirrors `BuildOpts::force`/`ReasonArgs::force`).
    pub force: bool,
    /// Skip documents already recorded done in the densify checkpoint (mirrors
    /// `BuildOpts::resume`/`ReasonArgs::resume`).
    pub resume: bool,
    /// Number of chunks folded into a single densify round — threaded straight into
    /// `densify_doc` (mirrors `BuildOpts::chunks_per_round`). Default `3`, matching
    /// `BuildOpts::default()`.
    pub chunks_per_round: usize,
}

/// indicatif progress bar over `len` units — hidden when `no_progress` is set or stdout/stderr
/// isn't a TTY (mirrors `reason::progress_bar`).
fn progress_bar(len: usize, no_progress: bool) -> ProgressBar {
    let show = !no_progress && std::io::stdout().is_terminal() && std::io::stderr().is_terminal();
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

/// The node types eligible to seed from. An explicit `--seed-type` restricts to exactly that one
/// type (the caller's call — no further filtering). Otherwise: the ontology's types marked
/// `requires_grounding` (its "KNOWLEDGE" types, per the design doc), excluding anything the
/// ontology also declares structural; if none are marked, every declared type that isn't
/// structural. Structural (Document/Section) types are excluded either way as a safety net — the
/// seed pool itself does the real MENTIONS-groundedness filtering, but a structural node is never
/// eligible even if it happened to satisfy that.
pub fn eligible_seed_types(ont: &Ontology, seed_type: Option<&str>) -> BTreeSet<String> {
    if let Some(t) = seed_type {
        return std::iter::once(t.to_string()).collect();
    }
    let structural: HashSet<String> = ont.structural().into_iter().collect();
    let grounded: BTreeSet<String> = ont
        .entity_types()
        .iter()
        .filter(|t| ont.requires_grounding(t) && !structural.contains(*t))
        .cloned()
        .collect();
    if !grounded.is_empty() {
        return grounded;
    }
    ont.entity_types()
        .iter()
        .filter(|t| !structural.contains(*t))
        .cloned()
        .collect()
}

/// The seed pool: every node of an eligible type (see [`eligible_seed_types`]) carrying at least
/// one outgoing `MENTIONS` edge (grounded), sorted deterministically by id — so `--count` attempts
/// are reproducible run-to-run for the same graph, no RNG/wallclock involved.
pub fn seed_pool(g: &GraphStore, ont: &Ontology, seed_type: Option<&str>) -> Result<Vec<Seed>> {
    let types = eligible_seed_types(ont, seed_type);
    let mut seeds: Vec<Seed> = g
        .all_nodes()?
        .into_iter()
        .filter(|n| types.contains(&n.node_type))
        .filter(|n| {
            g.outgoing(&n.id)
                .map(|edges| edges.iter().any(|e| e.edge_type == glossa::graph::MENTIONS))
                .unwrap_or(false)
        })
        .map(|n| Seed {
            id: n.id,
            node_type: n.node_type,
            label: n.label,
        })
        .collect();
    seeds.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(seeds)
}

/// One kept synthetic gold, in the exact `[[case]]` shape `dataset_toml::parse_dataset_toml`
/// reads back (`id`/`question`/`answer`; `aliases`/`tags` are optional there and simply omitted
/// here — they default to empty on read-back).
#[derive(Debug, Serialize)]
struct OutCase {
    id: String,
    question: String,
    answer: String,
}

#[derive(Debug, Serialize)]
struct OutFile {
    case: Vec<OutCase>,
}

/// Serialize `kept` as `[[case]]` blocks and write them to `out_path` (created/truncated).
fn write_dataset_toml(out_path: &std::path::Path, kept: &[OutCase]) -> Result<()> {
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let file = OutFile {
        case: kept.iter().map(|c| OutCase {
            id: c.id.clone(),
            question: c.question.clone(),
            answer: c.answer.clone(),
        }).collect(),
    };
    let text = toml::to_string_pretty(&file).context("serializing synthetic dataset.toml")?;
    std::fs::write(out_path, text)
        .with_context(|| format!("writing {}", out_path.display()))?;
    Ok(())
}

/// Orchestrate `kbx distil` over the corpus at `path` (kb-style PATH resolution via
/// `workspace::resolve`): load `lab.toml` + ontology + `distil.md`, build the seed pool, attempt
/// `args.count` generate+gate passes (`gen::generate_one`), and write the kept golds to
/// `args.out` (default `<kbx>/dataset.synthetic.toml`).
pub fn run_distil(path: Option<PathBuf>, args: DistilArgs) -> Result<()> {
    let paths = workspace::resolve(path);
    run_distil_at(paths, args)
}

/// `run_distil`'s body, taking already-resolved `KbxPaths` — split out so tests can exercise it
/// without fighting `workspace::resolve`'s PATH-walking discovery (mirrors `reason::run_reason_at`).
fn run_distil_at(paths: KbxPaths, args: DistilArgs) -> Result<()> {
    let lab = LabConfig::load_at(&paths.lab)
        .with_context(|| format!("loading {}", paths.lab.display()))?;
    let ontology = Ontology::load_or_default(&paths.root);

    // Ensure the corpus is indexed — mirrors `run_distil`'s own first step; a no-op if already
    // indexed. Needed so `read` calls the generator makes resolve real chunks.
    glossa::index::store::index_dir(&paths.root, false).context("indexing corpus")?;

    let distil_md = std::fs::read_to_string(&paths.distil)
        .with_context(|| format!("reading {}", paths.distil.display()))?;

    let g = GraphStore::open(&paths.root)?;
    let seeds = seed_pool(&g, &ontology, args.seed_type.as_deref())?;
    if seeds.is_empty() {
        bail!(
            "kbx distil: no grounded seed nodes found (need a node of an eligible type carrying \
             an outgoing MENTIONS edge) — build the graph first (`kbx build`)"
        );
    }

    let out_path = args
        .out
        .clone()
        .unwrap_or_else(|| paths.kbx_dir.join("dataset.synthetic.toml"));

    // Attempt budget: `--target N` keeps generating until N are KEPT, bounded by `max_attempts`
    // (default N*4, floored at N) so a stubborn gate can't loop forever; otherwise a fixed `--count`
    // attempts (default 1). Both share the sorted-by-id pool cycled by attempt index — no
    // RNG/wallclock, so a given (graph, budget) is reproducible.
    let target = args.target;
    let cap = match target {
        Some(t) => args.max_attempts.unwrap_or(t.saturating_mul(4)).max(t),
        None => args.count.unwrap_or(1),
    };

    let pb = progress_bar(cap, args.no_progress);
    pb.set_message("distil");

    let mut kept: Vec<OutCase> = Vec::new();
    let mut n_dropped = 0usize;
    let mut attempts = 0usize;

    while attempts < cap {
        if target.is_some_and(|t| kept.len() >= t) {
            break;
        }
        let i = attempts;
        let seed = &seeds[i % seeds.len()];
        pb.set_message(format!("distil {i} (seed {})", seed.id));
        match generate_one(&paths, &ontology, &lab, &distil_md, seed)
            .with_context(|| format!("distil attempt {i} (seed {})", seed.id))?
        {
            GenOutcome::Kept(p) => {
                println!("distil {i}: kept \"{}\" (seed {})", p.question, seed.id);
                // Sequential id over KEPT golds (no gaps from dropped attempts).
                kept.push(OutCase {
                    id: format!("synth-{}", kept.len()),
                    question: p.question,
                    answer: p.answer,
                });
            }
            GenOutcome::Dropped(reason) => {
                println!(
                    "distil {i}: dropped ({}) (seed {})",
                    reason.describe(),
                    seed.id
                );
                n_dropped += 1;
            }
        }
        attempts += 1;
        pb.inc(1);
    }
    pb.finish_and_clear();

    write_dataset_toml(&out_path, &kept)?;

    println!(
        "distil: {} attempted, {} kept, {} dropped -> {}",
        attempts,
        kept.len(),
        n_dropped,
        out_path.display()
    );
    if let Some(t) = target {
        if kept.len() < t {
            println!(
                "distil: target {} NOT reached — kept {} in {} attempts (cap {}); raise --max-attempts or loosen seeds",
                t,
                kept.len(),
                attempts,
                cap
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// `kbx distil densify` — the whole-corpus, checkpointed/resumable orchestrator for `densify_doc`
// (Task 3 of the 2026-08-28 densification plan). Mirrors `run_build`'s extract stage
// (`crate::build::run_build`, the `if run_extract_stage { ... }` block) in shape: enumerate docs
// -> chunk-weighted progress bar (`crate::build::extract_total_chunks`/`extract_doc_weight`,
// widened to `pub(crate)` for this reuse) -> per-doc loop with a `distil:{doc}` checkpoint mark
// and per-round `on_progress` top-up -> `finish_and_clear` -> the SAME finalize
// (`crate::build::finalize`) `run_reason` runs at the end of `kbx reason`.
// ---------------------------------------------------------------------------------------------

/// The checkpoint unit id for one densify doc pass — `distil:{doc}`, matching `run_build`'s
/// `extract:{doc}` / `run_reason`'s `reason:{seed_id}` naming convention.
pub fn distil_unit_id(doc: &str) -> String {
    format!("distil:{doc}")
}

/// Narrow an enumerated doc list to exactly `doc` when set, preserving order — the `--doc`
/// selection `run_densify_at` applies right after enumeration (mirrors `BuildOpts::doc`'s
/// narrowing in `run_build`'s extract stage). Pure and model-free, extracted so `--doc` selection
/// is unit-testable without a live model.
pub fn select_docs(mut docs: Vec<String>, doc: Option<&str>) -> Vec<String> {
    if let Some(d) = doc {
        docs.retain(|p| p == d);
    }
    docs
}

/// A doc is skipped this densify pass iff `--resume` is set AND the checkpoint already recorded
/// its `distil:{doc}` mark done. Pure and model-free — mirrors `reason::should_skip` exactly, one
/// level down (doc id instead of seed id).
pub fn should_skip_densify(cp: &Checkpoint, doc: &str, resume: bool) -> bool {
    resume && cp.is_done(&distil_unit_id(doc))
}

/// Clear a densify run's persistent checkpoint (`run_dir/done/`) — `--force`'s full-rebuild half,
/// mirroring `build::clear_checkpoint`/`reason`'s own local copy of the same helper. A no-op when
/// the checkpoint dir doesn't exist yet (first-ever densify run).
fn clear_densify_checkpoint(run_dir: &std::path::Path) -> Result<()> {
    let done_dir = run_dir.join("done");
    if done_dir.exists() {
        std::fs::remove_dir_all(&done_dir)
            .with_context(|| format!("clearing checkpoint {}", done_dir.display()))?;
    }
    Ok(())
}

/// Orchestrate `kbx distil densify` over the corpus at `path` (kb-style PATH resolution via
/// `workspace::resolve`, mirroring `run_reason`/`run_distil`'s thin entry shape).
pub fn run_densify(path: Option<PathBuf>, args: &DistilArgs) -> Result<()> {
    let paths = workspace::resolve(path);
    run_densify_at(paths, args)
}

/// `run_densify`'s body, taking already-resolved `KbxPaths` — split out so tests can exercise it
/// without fighting `workspace::resolve`'s PATH-walking discovery (mirrors
/// `run_distil_at`/`run_reason_at`).
fn run_densify_at(paths: KbxPaths, args: &DistilArgs) -> Result<()> {
    let lab = LabConfig::load_at(&paths.lab)
        .with_context(|| format!("loading {}", paths.lab.display()))?;
    let ontology = Ontology::load_or_default(&paths.root);

    // Ensure the corpus is indexed (structural nodes + chunks) — no-op if already indexed, same
    // first step every `kbx` pipeline entry takes.
    glossa::index::store::index_dir(&paths.root, false).context("indexing corpus")?;

    let distil_md = std::fs::read_to_string(&paths.distil)
        .with_context(|| format!("reading {}", paths.distil.display()))?;

    // A fresh, short-lived handle: enumerate then drop before densify_doc opens its own
    // per-document GraphStore connection — same reasoning as `run_build`'s extract stage.
    let mut docs = {
        let g = GraphStore::open(&paths.root).context("open graph store to enumerate docs")?;
        crate::build::enumerate_docs(&g)?
    };
    docs = select_docs(docs, args.doc.as_deref());
    if docs.is_empty() {
        bail!(
            "kbx distil densify: no document matched (corpus empty, or --doc names a document \
             that isn't indexed)"
        );
    }

    // Weight the bar by chunk, not by document — identical rationale/mechanism to `run_build`'s
    // extract stage (see `extract_doc_weight`'s doc comment): a huge document is otherwise one
    // tick and the bar/ETA lie on a mixed-size corpus.
    let idx = DocIndex::open_or_create(&paths.root).context("open doc index for densify")?;
    let mut chunk_counts: HashMap<String, usize> = HashMap::new();
    idx.iter_chunks(|path, _ord, _kind, _text| {
        *chunk_counts.entry(path.to_string()).or_default() += 1;
    })
    .context("counting chunks per doc for densify-bar weights")?;
    let total_chunks = crate::build::extract_total_chunks(&docs, &chunk_counts);

    // Densify state lives under its own stable `runs/distil/` dir — one corpus has exactly one
    // in-progress densify pass to resume/checkpoint, same convention as `runs/build/`/`runs/reason/`.
    let run_dir = paths.runs.join("distil");
    if args.force {
        clear_densify_checkpoint(&run_dir).context("clearing distil checkpoint for --force")?;
    }
    let cp = Checkpoint::open(&run_dir).context("open distil checkpoint")?;

    let pb = progress_bar(total_chunks.max(1), args.no_progress);
    pb.set_message("distil (chunks)");
    let mut total = DensifyStats::default();
    let mut docs_done: Vec<String> = Vec::new();
    for doc in &docs {
        let w = crate::build::extract_doc_weight(doc, &chunk_counts) as u64;
        if should_skip_densify(&cp, doc, args.resume) {
            pb.inc(w);
            continue;
        }
        // Advance the bar WITHIN the doc via densify_doc's per-round callback (real chunks
        // covered), same as `run_build`'s extract stage: `covered` tracks what the callback
        // advanced so the per-doc total can be reconciled to this doc's weight below.
        let mut covered = 0u64;
        let stats = densify_doc(
            &paths,
            &ontology,
            &lab,
            &distil_md,
            doc,
            args.chunks_per_round,
            |n| {
                covered += n;
                pb.inc(n);
            },
        )
        .with_context(|| format!("densifying {doc}"))?;
        total.nodes += stats.nodes;
        total.edges += stats.edges;
        total.grounded += stats.grounded;
        cp.mark(&distil_unit_id(doc), "done")
            .with_context(|| format!("marking {doc} done"))?;
        docs_done.push(doc.clone());
        // A live indicatif bar owns the terminal: any per-doc message while it's live MUST go
        // through `pb.println`, never a bare `println!` (the bug this task's brief calls out by
        // name) — only the post-`finish_and_clear` summary below may use `println!` directly.
        pb.println(format!(
            "distil {doc}: {} node(s), {} edge(s), {} grounded",
            stats.nodes, stats.edges, stats.grounded
        ));
        // Top up any gap so the bar stays aligned to this doc's weight, same as `run_build`.
        if covered < w {
            pb.inc(w - covered);
        }
    }
    pb.finish_and_clear();
    println!(
        "distil: {} doc(s) densified, {} chunk(s), {} node(s), {} edge(s), {} grounded",
        docs_done.len(),
        total_chunks,
        total.nodes,
        total.edges,
        total.grounded
    );

    // Same hygiene/generalize/node-index finalize `run_reason` runs at the end of `kbx reason`,
    // so the derived layer + node index are refreshed after densify's writes too.
    let summary = crate::build::finalize(&paths.root).context("finalizing distil")?;
    println!("{summary}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glossa::graph::store::{Edge, Node, Provenance};

    fn prov() -> Provenance {
        Provenance {
            source_path: "doc.md".into(),
            range: None,
            file_sig: None,
            origin: "test".into(),
            confidence: 0.9,
            created_at: 1,
        }
    }

    const GROUNDING_ONT: &str = r#"
[entities.Fact]
requires_grounding = true
[entities.Document]
[entities.Section]

[relations.LEADS_TO]
from = ["Fact"]
to = ["Fact"]
"#;

    const UNGROUNDED_ONT: &str = r#"
[entities.Fact]
[entities.Document]
[entities.Section]

[relations.LEADS_TO]
from = ["Fact"]
to = ["Fact"]
"#;

    #[test]
    fn eligible_seed_types_prefers_grounding_required_types() {
        let ont = Ontology::parse(GROUNDING_ONT).unwrap();
        let types = eligible_seed_types(&ont, None);
        assert!(types.contains("Fact"));
        assert!(!types.contains("Document"), "structural type must never be eligible");
        assert!(!types.contains("Section"), "structural type must never be eligible");
    }

    #[test]
    fn eligible_seed_types_falls_back_to_all_non_structural_when_none_require_grounding() {
        let ont = Ontology::parse(UNGROUNDED_ONT).unwrap();
        let types = eligible_seed_types(&ont, None);
        assert!(types.contains("Fact"));
        assert!(!types.contains("Document"));
        assert!(!types.contains("Section"));
    }

    #[test]
    fn eligible_seed_types_explicit_seed_type_wins_outright() {
        let ont = Ontology::parse(GROUNDING_ONT).unwrap();
        let types = eligible_seed_types(&ont, Some("Section"));
        assert_eq!(types, std::iter::once("Section".to_string()).collect());
    }

    #[test]
    fn seed_pool_excludes_structural_and_requires_a_mentions_edge() {
        let dir = tempfile::tempdir().unwrap();
        let ont = Ontology::parse(GROUNDING_ONT).unwrap();
        let g = GraphStore::open(dir.path()).unwrap();

        // A grounded Fact: eligible.
        g.put_node(&Node {
            id: "fact-grounded".into(),
            node_type: "Fact".into(),
            label: "grounded fact".into(),
            aliases: Vec::new(),
            prov: prov(),
        })
        .unwrap();
        g.put_edge(&Edge {
            from: "fact-grounded".into(),
            to: "doc.md#1".into(),
            edge_type: glossa::graph::MENTIONS.to_string(),
            prov: prov(),
        })
        .unwrap();

        // An UNgrounded Fact (no MENTIONS edge): excluded.
        g.put_node(&Node {
            id: "fact-ungrounded".into(),
            node_type: "Fact".into(),
            label: "ungrounded fact".into(),
            aliases: Vec::new(),
            prov: prov(),
        })
        .unwrap();

        // A structural Document node, even if (hypothetically) grounded: excluded by type.
        g.put_node(&Node {
            id: "doc.md".into(),
            node_type: "Document".into(),
            label: "doc.md".into(),
            aliases: Vec::new(),
            prov: prov(),
        })
        .unwrap();
        g.put_edge(&Edge {
            from: "doc.md".into(),
            to: "doc.md#1".into(),
            edge_type: glossa::graph::MENTIONS.to_string(),
            prov: prov(),
        })
        .unwrap();

        let seeds = seed_pool(&g, &ont, None).unwrap();
        let ids: Vec<&str> = seeds.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["fact-grounded"], "seed pool must be exactly the grounded Fact: {ids:?}");
    }

    #[test]
    fn seed_pool_is_sorted_deterministically_by_id() {
        let dir = tempfile::tempdir().unwrap();
        let ont = Ontology::parse(GROUNDING_ONT).unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        for id in ["fact-c", "fact-a", "fact-b"] {
            g.put_node(&Node {
                id: id.into(),
                node_type: "Fact".into(),
                label: id.into(),
                aliases: Vec::new(),
                prov: prov(),
            })
            .unwrap();
            g.put_edge(&Edge {
                from: id.into(),
                to: "doc.md#1".into(),
                edge_type: glossa::graph::MENTIONS.to_string(),
                prov: prov(),
            })
            .unwrap();
        }
        let seeds = seed_pool(&g, &ont, None).unwrap();
        let ids: Vec<&str> = seeds.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["fact-a", "fact-b", "fact-c"]);
    }

    #[test]
    fn write_dataset_toml_round_trips_through_the_real_parser() {
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("dataset.synthetic.toml");
        let kept = vec![
            OutCase {
                id: "synth-0".into(),
                question: "what follows the seed?".into(),
                answer: "the terminal fact".into(),
            },
            OutCase {
                id: "synth-1".into(),
                question: "second question?".into(),
                answer: "second answer".into(),
            },
        ];
        write_dataset_toml(&out_path, &kept).unwrap();

        let text = std::fs::read_to_string(&out_path).unwrap();
        let parsed = crate::dataset_toml::parse_dataset_toml(&text).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].id, "synth-0");
        assert_eq!(parsed[0].question, "what follows the seed?");
        assert_eq!(parsed[0].answer, "the terminal fact");
        assert_eq!(parsed[1].id, "synth-1");
    }

    // ---- densify orchestrator (Task 3): doc selection + checkpoint, model-free ----------------

    #[test]
    fn distil_unit_id_formats_distil_prefix() {
        assert_eq!(distil_unit_id("a.md"), "distil:a.md");
    }

    /// `--doc` narrows an enumerated set to exactly that one doc; unset leaves it untouched.
    /// Mirrors `plan_extract`/`should_skip`'s pure, model-free unit-test style in `build`/`reason`.
    #[test]
    fn select_docs_filters_to_exactly_the_named_doc() {
        let docs = vec!["a.md".to_string(), "b.md".to_string(), "c.md".to_string()];

        assert_eq!(
            select_docs(docs.clone(), Some("b.md")),
            vec!["b.md".to_string()]
        );
        assert_eq!(select_docs(docs.clone(), None), docs);
    }

    /// `--doc` naming a document absent from the enumerated set yields an empty selection, not an
    /// error — `run_densify_at` turns that into a clear bail! rather than silently no-op-ing.
    #[test]
    fn select_docs_absent_doc_yields_empty() {
        let docs = vec!["a.md".to_string()];
        assert!(select_docs(docs, Some("nonexistent.md")).is_empty());
    }

    /// Step 1's TDD unit (per the brief, mirroring `reason::should_skip`'s own test): a doc id
    /// already recorded done in the checkpoint is skipped under `--resume`; a fresh id is not; and
    /// a done id is NOT skipped when `--resume` isn't set.
    #[test]
    fn should_skip_densify_marks_done_doc_under_resume_and_not_a_fresh_doc() {
        let dir = tempfile::tempdir().unwrap();
        let cp = Checkpoint::open(&dir.path().join("runs").join("distil")).unwrap();
        cp.mark(&distil_unit_id("a.md"), "done").unwrap();

        assert!(
            should_skip_densify(&cp, "a.md", true),
            "an already-done doc must be skipped under --resume"
        );
        assert!(
            !should_skip_densify(&cp, "b.md", true),
            "a fresh doc must not be skipped even under --resume"
        );
        assert!(
            !should_skip_densify(&cp, "a.md", false),
            "a done doc must NOT be skipped when --resume isn't set"
        );
    }

    /// `--force`'s checkpoint-clearing half: a doc marked done in a prior densify run must report
    /// `!is_done` again after `clear_densify_checkpoint`, mirroring `build::force_clears_checkpoint`.
    #[test]
    fn force_clears_densify_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let run_dir = dir.path().join("runs").join("distil");
        let cp = Checkpoint::open(&run_dir).unwrap();
        cp.mark(&distil_unit_id("a.md"), "done").unwrap();
        assert!(cp.is_done(&distil_unit_id("a.md")));

        clear_densify_checkpoint(&run_dir).unwrap();

        let cp2 = Checkpoint::open(&run_dir).unwrap();
        assert!(!cp2.is_done(&distil_unit_id("a.md")));
    }

    /// `clear_densify_checkpoint` must be a no-op (not an error) when the run has never
    /// checkpointed anything yet — the first-ever `--force` densify run on a fresh corpus.
    #[test]
    fn clear_densify_checkpoint_noop_when_absent() {
        let dir = tempfile::tempdir().unwrap();
        let run_dir = dir.path().join("runs").join("distil");
        assert!(!run_dir.exists());
        clear_densify_checkpoint(&run_dir).unwrap();
    }

    /// `DistilArgs`' new densify fields (Task 3) default to a sane full-corpus, non-resumed pass
    /// when constructed directly (as `kbx.rs`'s CLI wiring will do in Task 4).
    #[test]
    fn distil_args_densify_fields_are_plain_and_settable() {
        let args = DistilArgs {
            count: None,
            target: None,
            max_attempts: None,
            out: None,
            seed_type: None,
            no_progress: true,
            doc: Some("a.md".to_string()),
            force: true,
            resume: false,
            chunks_per_round: 3,
        };
        assert_eq!(args.doc.as_deref(), Some("a.md"));
        assert!(args.force);
        assert_eq!(args.chunks_per_round, 3);
    }
}
