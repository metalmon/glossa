//! `kbx synth` pipeline orchestrator: resolve the workspace, build the seed pool from grounded
//! non-structural graph nodes, run `gen::generate_one` (generate + verify-gate) once per attempt,
//! and write the kept synthetic golds as `[[case]]` rows to `--out` — the SAME `dataset.toml`
//! shape `dataset_toml::parse_dataset_toml` reads back, so `kbx distil --gold <out>` and
//! `kbx eval --dataset <out>` consume it unchanged.
//!
//! Read-only on the graph: this module never calls `graph_upsert`. The only write anywhere in
//! `kbx synth` is the `--out` dataset file.

use crate::lab::LabConfig;
use crate::synth::gen::{generate_one, GenOutcome, Seed};
use crate::workspace::{self, KbxPaths};
use anyhow::{bail, Context, Result};
use glossa::graph::ontology::Ontology;
use glossa::graph::store::GraphStore;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;
use std::collections::{BTreeSet, HashSet};
use std::io::IsTerminal;
use std::path::PathBuf;

/// CLI-level options for `kbx synth`, folded from the `kbx` binary's clap fields (mirrors
/// `distil::DistilArgs`'s shape).
#[derive(Debug, Clone)]
pub struct SynthArgs {
    /// Number of synthetic golds to ATTEMPT — the gate may drop some; kept/dropped are reported.
    pub count: usize,
    /// Dataset TOML to write (default `<kbx>/dataset.synthetic.toml`). Always overwritten whole.
    pub out: Option<PathBuf>,
    /// Restrict seeds to this node_type (default: the ontology's grounding-required types, or
    /// every non-structural declared type when none are marked `requires_grounding`).
    pub seed_type: Option<String>,
    /// Never draw the progress bar, even on a TTY.
    pub no_progress: bool,
}

/// indicatif progress bar over `len` units — hidden when `no_progress` is set or stdout/stderr
/// isn't a TTY (mirrors `distil::progress_bar`).
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

/// Orchestrate `kbx synth` over the corpus at `path` (kb-style PATH resolution via
/// `workspace::resolve`): load `lab.toml` + ontology + `synth.md`, build the seed pool, attempt
/// `args.count` generate+gate passes (`gen::generate_one`), and write the kept golds to
/// `args.out` (default `<kbx>/dataset.synthetic.toml`).
pub fn run_synth(path: Option<PathBuf>, args: SynthArgs) -> Result<()> {
    let paths = workspace::resolve(path);
    run_synth_at(paths, args)
}

/// `run_synth`'s body, taking already-resolved `KbxPaths` — split out so tests can exercise it
/// without fighting `workspace::resolve`'s PATH-walking discovery (mirrors `distil::run_distil_at`).
fn run_synth_at(paths: KbxPaths, args: SynthArgs) -> Result<()> {
    let lab = LabConfig::load_at(&paths.lab)
        .with_context(|| format!("loading {}", paths.lab.display()))?;
    let ontology = Ontology::load_or_default(&paths.root);

    // Ensure the corpus is indexed — mirrors `run_distil`'s own first step; a no-op if already
    // indexed. Needed so `read` calls the generator makes resolve real chunks.
    glossa::index::store::index_dir(&paths.root, false).context("indexing corpus")?;

    let synth_md = std::fs::read_to_string(&paths.synth)
        .with_context(|| format!("reading {}", paths.synth.display()))?;

    let g = GraphStore::open(&paths.root)?;
    let seeds = seed_pool(&g, &ontology, args.seed_type.as_deref())?;
    if seeds.is_empty() {
        bail!(
            "kbx synth: no grounded seed nodes found (need a node of an eligible type carrying \
             an outgoing MENTIONS edge) — build the graph first (`kbx build`)"
        );
    }

    let out_path = args
        .out
        .clone()
        .unwrap_or_else(|| paths.kbx_dir.join("dataset.synthetic.toml"));

    let pb = progress_bar(args.count, args.no_progress);
    pb.set_message("synth");

    let mut kept: Vec<OutCase> = Vec::new();
    let mut n_dropped = 0usize;

    for i in 0..args.count {
        // Deterministic-ish: sorted-by-id pool, varied by attempt index — no RNG/wallclock, and
        // an attempt count larger than the pool cycles back through it rather than erroring.
        let seed = &seeds[i % seeds.len()];
        pb.set_message(format!("synth {i} (seed {})", seed.id));
        match generate_one(&paths, &ontology, &lab, &synth_md, seed)
            .with_context(|| format!("synth attempt {i} (seed {})", seed.id))?
        {
            GenOutcome::Kept(p) => {
                println!("synth {i}: kept \"{}\" (seed {})", p.question, seed.id);
                kept.push(OutCase {
                    id: format!("synth-{i}"),
                    question: p.question,
                    answer: p.answer,
                });
            }
            GenOutcome::Dropped(reason) => {
                println!(
                    "synth {i}: dropped ({}) (seed {})",
                    reason.describe(),
                    seed.id
                );
                n_dropped += 1;
            }
        }
        pb.inc(1);
    }
    pb.finish_and_clear();

    write_dataset_toml(&out_path, &kept)?;

    println!(
        "synth: {} attempted, {} kept, {} dropped -> {}",
        args.count,
        kept.len(),
        n_dropped,
        out_path.display()
    );
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
}
