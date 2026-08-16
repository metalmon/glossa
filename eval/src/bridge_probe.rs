//! Bridge-probe: an adversarial fixture + scorer for the cross-document `reach` traversal
//! primitive (see `docs/superpowers/specs/2026-08-16-graph-cross-doc-bridge-design.md` §7.6).
//!
//! The mitigations that keep a corpus-resolve bridge from producing a confident wrong answer
//! (salience gate, relation-coherence prune, bridge budget) are untestable without cases
//! designed to trip them. This module loads a small **controlled synthetic** corpus
//! (`eval/tests/fixtures/bridge-probe/`) of trap cases tagged by [`TrapType`], and scores a
//! model's answers into **precision** (trap rows NOT lured into a false bridge) and **recall**
//! (true-positive rows where a legit bridge WAS followed) — reported overall and per trap type,
//! since a tool that never bridges wins precision but collapses recall, and vice versa.
//!
//! This is a mechanism unit-fixture, not the MuSiQue scientific benchmark
//! ([[per-query-graph-eval-plan]] rejects a homemade corpus for that purpose) — it exists solely
//! to TDD `reach`'s false-positive mitigations against.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Context;
use serde::Deserialize;

/// Why a bridge-probe case is adversarial (or not). `Homonym`, `CommonWord`, and
/// `TransitiveDrift` are traps: a naive bridge is lured into a wrong-but-specific answer.
/// `TruePositive` cases have a legit cross-document bridge that MUST connect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrapType {
    /// Two documents reuse the same pivot name/mention in different senses; the gold is the
    /// right-sense fact, a false bridge lands on the wrong-sense document's fact instead.
    Homonym,
    /// Two unrelated entities share a frequent role/relation word (e.g. "member", "founder");
    /// bridging on that common word alone would swap one entity's fact for the other's.
    CommonWord,
    /// The gold fact is directly stated (single-hop), but a spurious 2-bridge chain through
    /// unrelated intermediate mentions would drift onto a different, wrong document.
    TransitiveDrift,
    /// A genuine cross-document bridge: the answer is only reachable by following it.
    TruePositive,
}

/// One bridge-probe case: a question over a small synthetic corpus, with the trap-type
/// classification and the (relative) directory holding that case's `.md` documents.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeCase {
    pub id: String,
    pub question: String,
    pub answer: String,
    /// Alternative gold surface forms; empty when the primary answer is unambiguous.
    pub answer_aliases: Vec<String>,
    pub trap_type: TrapType,
    /// Directory name (relative to the fixture root, e.g. `eval/tests/fixtures/bridge-probe/`)
    /// holding this case's synthetic `.md` corpus.
    pub corpus_dir: String,
}

#[derive(Deserialize)]
struct RawProbeCase {
    id: String,
    question: String,
    answer: String,
    #[serde(default)]
    answer_aliases: Vec<String>,
    trap_type: TrapType,
    corpus_dir: String,
}

/// Parse the bridge-probe `.jsonl` text (one case per line; blank lines skipped).
pub fn parse_probe(jsonl: &str) -> anyhow::Result<Vec<ProbeCase>> {
    let mut out = Vec::new();
    for (i, line) in jsonl.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let r: RawProbeCase = serde_json::from_str(line)
            .with_context(|| format!("parse bridge-probe line {}", i + 1))?;
        out.push(ProbeCase {
            id: r.id,
            question: r.question,
            answer: r.answer,
            answer_aliases: r.answer_aliases,
            trap_type: r.trap_type,
            corpus_dir: r.corpus_dir,
        });
    }
    Ok(out)
}

/// Read and parse a bridge-probe `.jsonl` file from disk.
pub fn load_probe(path: impl AsRef<Path>) -> anyhow::Result<Vec<ProbeCase>> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read bridge-probe fixture {}", path.display()))?;
    parse_probe(&text)
}

/// Precision/recall report for a bridge-probe run, plus a per-trap-type breakdown.
///
/// `precision` = fraction of NON-`TruePositive` (trap) rows answered correctly — i.e. NOT lured
/// into a false bridge. `recall` = fraction of `TruePositive` rows answered correctly — i.e. the
/// legit cross-document bridge WAS followed. Reporting both matters: a tool that never bridges
/// scores perfect trap-precision but recall 0; the goal is high precision without collapsing
/// recall.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeReport {
    pub precision: f32,
    pub recall: f32,
    /// `(hits, total)` per trap type, in `TrapType` order.
    pub per_trap: BTreeMap<TrapType, (usize, usize)>,
}

/// Score model `answers` (parallel to `cases`, by index) against the bridge-probe gold. EM is
/// [`crate::score::relaxed_match_any`] over `[answer] + answer_aliases` (credits a correct
/// concise/verbose surface form while still rejecting an over-hop to an unrelated entity).
pub fn score_probe(cases: &[ProbeCase], answers: &[String]) -> ProbeReport {
    let mut per_trap: BTreeMap<TrapType, (usize, usize)> = BTreeMap::new();
    let mut trap_hits = 0usize;
    let mut trap_total = 0usize;
    let mut tp_hits = 0usize;
    let mut tp_total = 0usize;

    for (case, answer) in cases.iter().zip(answers.iter()) {
        let mut golds = vec![case.answer.clone()];
        golds.extend(case.answer_aliases.iter().cloned());
        let correct = crate::score::relaxed_match_any(answer, &golds);

        let entry = per_trap.entry(case.trap_type).or_insert((0, 0));
        entry.1 += 1;
        if correct {
            entry.0 += 1;
        }

        if case.trap_type == TrapType::TruePositive {
            tp_total += 1;
            if correct {
                tp_hits += 1;
            }
        } else {
            trap_total += 1;
            if correct {
                trap_hits += 1;
            }
        }
    }

    ProbeReport {
        precision: if trap_total == 0 {
            0.0
        } else {
            trap_hits as f32 / trap_total as f32
        },
        recall: if tp_total == 0 {
            0.0
        } else {
            tp_hits as f32 / tp_total as f32
        },
        per_trap,
    }
}

/// The real fixture's `.jsonl` path, relative to the crate root (`eval/`).
pub const FIXTURE_PATH: &str = "tests/fixtures/bridge-probe/bridge-probe.jsonl";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_probe_separates_precision_and_recall() {
        let cases = vec![
            ProbeCase {
                id: "h1".into(),
                question: "q".into(),
                answer: "RightSense".into(),
                answer_aliases: vec![],
                trap_type: TrapType::Homonym,
                corpus_dir: "h1".into(),
            },
            ProbeCase {
                id: "t1".into(),
                question: "q".into(),
                answer: "Target".into(),
                answer_aliases: vec![],
                trap_type: TrapType::TruePositive,
                corpus_dir: "t1".into(),
            },
        ];
        // model got the homonym trap right, missed the true-positive bridge
        let answers = vec!["RightSense".into(), "Wrong".into()];
        let r = score_probe(&cases, &answers);
        assert_eq!(r.precision, 1.0); // trap not lured
        assert_eq!(r.recall, 0.0); // legit bridge missed
    }

    #[test]
    fn score_probe_reports_per_trap_breakdown() {
        let cases = vec![
            ProbeCase {
                id: "cw1".into(),
                question: "q".into(),
                answer: "Right".into(),
                answer_aliases: vec![],
                trap_type: TrapType::CommonWord,
                corpus_dir: "cw1".into(),
            },
            ProbeCase {
                id: "cw2".into(),
                question: "q".into(),
                answer: "Right2".into(),
                answer_aliases: vec![],
                trap_type: TrapType::CommonWord,
                corpus_dir: "cw2".into(),
            },
            ProbeCase {
                id: "td1".into(),
                question: "q".into(),
                answer: "Drifted".into(),
                answer_aliases: vec![],
                trap_type: TrapType::TransitiveDrift,
                corpus_dir: "td1".into(),
            },
        ];
        // lured on cw2 (wrong answer) and td1 (wrong answer)
        let answers = vec!["Right".into(), "SomeoneElsesFact".into(), "1932".into()];
        let r = score_probe(&cases, &answers);
        assert_eq!(r.per_trap.get(&TrapType::CommonWord), Some(&(1, 2)));
        assert_eq!(r.per_trap.get(&TrapType::TransitiveDrift), Some(&(0, 1)));
        // overall trap precision: 1 correct out of 3 trap rows
        assert!((r.precision - (1.0 / 3.0)).abs() < 1e-6);
        // no TruePositive rows in this case set
        assert_eq!(r.recall, 0.0);
    }

    #[test]
    fn score_probe_matches_alias_and_relaxed_containment() {
        let cases = vec![ProbeCase {
            id: "tp1".into(),
            question: "q".into(),
            answer: "Vellmoor County".into(),
            answer_aliases: vec!["Vellmoor".into()],
            trap_type: TrapType::TruePositive,
            corpus_dir: "tp1".into(),
        }];
        let answers = vec!["Vellmoor".into()];
        let r = score_probe(&cases, &answers);
        assert_eq!(r.recall, 1.0);
    }

    #[test]
    fn load_probe_reads_the_real_fixture_with_at_least_three_per_trap_type() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_PATH);
        let cases = load_probe(&path).expect("load real bridge-probe fixture");
        assert!(cases.len() >= 12, "expected >=12 cases, got {}", cases.len());

        let mut counts: BTreeMap<TrapType, usize> = BTreeMap::new();
        for c in &cases {
            *counts.entry(c.trap_type).or_insert(0) += 1;
            assert!(!c.question.is_empty(), "case {} has an empty question", c.id);
            assert!(!c.answer.is_empty(), "case {} has an empty answer", c.id);
            assert!(
                !c.corpus_dir.is_empty(),
                "case {} has an empty corpus_dir",
                c.id
            );

            // Every case's corpus_dir must exist on disk and contain at least one .md doc —
            // the fixture is committed data, not just jsonl rows.
            let corpus_path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/bridge-probe")
                .join(&c.corpus_dir);
            assert!(
                corpus_path.is_dir(),
                "case {} corpus_dir {} does not exist",
                c.id,
                corpus_path.display()
            );
            let has_md = std::fs::read_dir(&corpus_path)
                .unwrap_or_else(|_| panic!("read corpus dir for case {}", c.id))
                .filter_map(|e| e.ok())
                .any(|e| e.path().extension().map(|ext| ext == "md").unwrap_or(false));
            assert!(has_md, "case {} corpus_dir has no .md docs", c.id);
        }

        for trap in [
            TrapType::Homonym,
            TrapType::CommonWord,
            TrapType::TransitiveDrift,
            TrapType::TruePositive,
        ] {
            let n = counts.get(&trap).copied().unwrap_or(0);
            assert!(n >= 3, "expected >=3 cases for {:?}, got {}", trap, n);
        }
    }
}
