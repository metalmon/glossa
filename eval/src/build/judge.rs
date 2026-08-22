//! `kbx build` stage 3 — pairwise, chunk-grounded bridge judge (Task 8 of the `kbx build`
//! pipeline).
//!
//! Consumes stage 2's mechanical candidates (`candidates::candidate_pairs`) and, for each pair
//! NOT already judged (per the `Checkpoint`), asks the model a single liberal A→B question: is B
//! a further fact about the entity A introduces? The prompt (`bridge.md`) is the SAME
//! entity-pivot criterion validated in the probe interview — grounded in each side's actual
//! chunk text (`chunks::chunk_text`), not just the bare label, which is what moved judge recall
//! 0.20 → 1.00 in that interview. A YES writes one cross-doc spine edge; a NO writes nothing to
//! the graph, so the checkpoint mark is the ONLY record that the pair was already judged — every
//! judged pair (YES and NO alike) is marked, or a `--resume` run would re-judge every rejected
//! pair forever.
//!
//! The spine edge's `edge_type` is never hardcoded: it's resolved from the ontology by role
//! (`RelationRole::Chaining`) and by the endpoint node types the declared relation's `from`/`to`
//! accept — same shape as `Ontology::validate_edge`'s own from/to check, just enumerating instead
//! of checking one name. `apply_upsert` still validates/auto-corrects the write against the
//! ontology; this only picks WHICH relation name to offer it.

use crate::backend::openai::chat_once;
use crate::build::chunks::chunk_text;
use crate::build::candidates::CandidatePair;
use crate::checkpoint::Checkpoint;
use crate::lab::{Endpoint, LabConfig};
use anyhow::Context;
use glossa::graph::agent::{apply_upsert, EdgeSpec};
use glossa::graph::ontology::{Ontology, RelationRole};
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;
use indicatif::ProgressBar;
use serde_json::json;
use std::path::Path;

/// How much of the candidate pair list a `run_judge` pass judged/linked.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct JudgeStats {
    pub judged: usize,
    pub linked: usize,
    /// Pairs the model said YES to but for which `spine_edge_type` could not resolve a single
    /// relation (zero or ambiguous Chaining candidates for the endpoint types). Distinct from
    /// `linked == 0` on its own: a run can legitimately end with no YES verdicts at all
    /// (`skipped_ambiguous == 0`), but a nonzero `skipped_ambiguous` means the model WANTED a
    /// link and the ontology couldn't supply an unambiguous relation for it — a config defect
    /// worth surfacing to the caller, not silence.
    pub skipped_ambiguous: usize,
}

/// Parse a `judge_pair` reply's verdict: the LAST `VERDICT:` line (case-insensitive), `yes` ->
/// `true`, anything else (including a missing `VERDICT:` line entirely) -> `false`. Mirrors
/// `judge::parse_verdict`'s "last line wins, unrecognized falls back" shape, just for a two-value
/// YES/NO vocabulary instead of correct/partial/wrong.
fn parse_yes_no(reply: &str) -> bool {
    let mut verdict = false;
    for line in reply.lines() {
        let trimmed = line.trim().to_lowercase();
        if let Some(rest) = trimmed.strip_prefix("verdict:") {
            verdict = rest.trim().starts_with("yes");
        }
    }
    verdict
}

/// One A->B bridge decision: system = `bridge_md` (the liberal entity-pivot criterion), user =
/// the two fact labels plus their grounded chunk text. Posts to `ep` via `chat_once` (temp 0, no
/// tools — same substrate `judge::judge` uses) and parses the reply with `parse_yes_no`.
pub fn judge_pair(
    ep: &Endpoint,
    bridge_md: &str,
    a_label: &str,
    a_chunk: &str,
    b_label: &str,
    b_chunk: &str,
) -> anyhow::Result<bool> {
    let api_key = ep.resolve_key();
    let user = format!(
        "A: {a_label}\nA's source text:\n{a_chunk}\n\n\
         B: {b_label}\nB's source text:\n{b_chunk}\n\n\
         Reply with `VERDICT: YES` or `VERDICT: NO`, then a short reason."
    );
    let messages = vec![
        json!({ "role": "system", "content": bridge_md }),
        json!({ "role": "user", "content": user }),
    ];
    let msg = chat_once(
        &ep.endpoint,
        &ep.model,
        &messages,
        api_key.as_deref(),
        ep.timeout_secs,
    )
    .context("bridge judge endpoint request failed")?;
    let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
    Ok(parse_yes_no(content))
}

/// The single declared `RelationRole::Chaining` relation whose `from`/`to` accept
/// (`a_type`, `b_type`) in that order — the ontology-general stand-in for a hardcoded spine edge
/// type. Mirrors `Ontology::validate_edge`'s own "empty or `*` or exact match" allowance rule.
///
/// `None` when zero or more than one relation fits. Either case means a would-be spine edge gets
/// dropped (a model YES with nowhere to land), which is an ontology/config defect, not a quiet
/// no-op — so both branches log via `eprintln!` (the crate doesn't pull in `tracing`), and are
/// worded distinctly: "no Chaining relation covers types" (zero matches) vs "ambiguous spine
/// relation" with the candidate names listed (more than one matches, e.g. two Chaining relations
/// like CAUSES/PRECEDES both declared Fact->Fact).
fn spine_edge_type(ont: &Ontology, a_type: &str, b_type: &str) -> Option<String> {
    let allows = |allowed: &[String], t: &str| {
        allowed.is_empty() || allowed.iter().any(|x| x == "*" || x == t)
    };
    let mut candidates: Vec<&String> = ont
        .raw_relations()
        .iter()
        .filter(|(name, r)| {
            matches!(ont.relation_role(name), RelationRole::Chaining)
                && allows(&r.from, a_type)
                && allows(&r.to, b_type)
        })
        .map(|(name, _)| name)
        .collect();
    candidates.sort();
    candidates.dedup();
    match candidates.as_slice() {
        [one] => Some((*one).clone()),
        [] => {
            eprintln!(
                "kbx build: no Chaining relation covers types {a_type}->{b_type} \
                 (would-be spine edge dropped)"
            );
            None
        }
        many => {
            let names: Vec<&str> = many.iter().map(|s| s.as_str()).collect();
            eprintln!(
                "kbx build: ambiguous spine relation for {a_type}->{b_type}: {} \
                 (would-be spine edge dropped)",
                names.join(", ")
            );
            None
        }
    }
}

/// Judge every candidate pair not already checkpointed: fetch both nodes' grounded chunk text,
/// ask `judge_pair`, and on YES write one spine edge (`apply_upsert`) using the ontology-resolved
/// `edge_type`. Marks the checkpoint for EVERY judged pair, YES or NO — a NO leaves no graph
/// trace, so the checkpoint is the only thing that stops a `--resume` run from re-judging it.
pub fn run_judge(
    root: &Path,
    lab: &LabConfig,
    bridge_md: &str,
    g: &GraphStore,
    idx: &DocIndex,
    pairs: &[CandidatePair],
    cp: &Checkpoint,
    pb: &ProgressBar,
) -> anyhow::Result<JudgeStats> {
    let ont = Ontology::load_or_default(root);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut stats = JudgeStats::default();

    for pair in pairs {
        let unit_id = format!("judge:{}#{}", pair.a, pair.b);
        if cp.is_done(&unit_id) {
            pb.inc(1);
            continue;
        }

        let (Some(node_a), Some(node_b)) = (g.get_node(&pair.a)?, g.get_node(&pair.b)?) else {
            // A candidate referencing a node that's since vanished (deleted/merged away) can't be
            // judged — mark it done so a resume doesn't retry it forever, but don't count it.
            cp.mark(&unit_id, "no-node")?;
            pb.inc(1);
            continue;
        };

        let a_chunk = chunk_text(root, idx, &pair.a, g);
        let b_chunk = chunk_text(root, idx, &pair.b, g);

        let yes = judge_pair(
            &lab.model,
            bridge_md,
            &node_a.label,
            &a_chunk,
            &node_b.label,
            &b_chunk,
        )?;
        stats.judged += 1;

        if yes {
            match spine_edge_type(&ont, &node_a.node_type, &node_b.node_type) {
                Some(edge_type) => {
                    let edge = EdgeSpec {
                        from: pair.a.clone(),
                        to: pair.b.clone(),
                        edge_type,
                        source_path: node_a.prov.source_path.clone(),
                        range: None,
                        confidence: None,
                    };
                    apply_upsert(g, &ont, Vec::new(), vec![edge], now, root)?;
                    stats.linked += 1;
                }
                None => stats.skipped_ambiguous += 1,
            }
        }

        cp.mark(&unit_id, if yes { "yes" } else { "no" })?;
        pb.inc(1);
    }

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yes_no_parsing() {
        assert!(parse_yes_no("some reason\nVERDICT: yes"));
        assert!(parse_yes_no("VERDICT: YES because they share the entity"));
        assert!(!parse_yes_no("VERDICT: NO because different subject"));
        assert!(!parse_yes_no("no verdict line here"));
        // last VERDICT wins
        assert!(!parse_yes_no("VERDICT: yes\nVERDICT: no"));
        assert!(parse_yes_no("VERDICT: no\nVERDICT: yes"));
    }

    /// Table-driven: `spine_edge_type` is pure (no model, no IO), so every case is built from an
    /// in-memory `Ontology::parse` TOML and checked directly, no live endpoint needed.
    #[test]
    fn spine_edge_type_resolution_table() {
        // 1. Single matching Chaining relation -> Some(that).
        let single = Ontology::parse(
            r#"
[entities.Fact]
[relations.LEADS_TO]
from = ["Fact"]
to = ["Fact"]
role = "chaining"
"#,
        )
        .unwrap();
        assert_eq!(
            spine_edge_type(&single, "Fact", "Fact"),
            Some("LEADS_TO".to_string())
        );

        // 2. Two Chaining relations both declared over the same endpoint types -> ambiguous ->
        // None (e.g. CAUSES/PRECEDES both Fact->Fact, per the review's motivating example).
        let ambiguous = Ontology::parse(
            r#"
[entities.Fact]
[relations.CAUSES]
from = ["Fact"]
to = ["Fact"]
role = "chaining"
[relations.PRECEDES]
from = ["Fact"]
to = ["Fact"]
role = "chaining"
"#,
        )
        .unwrap();
        assert_eq!(spine_edge_type(&ambiguous, "Fact", "Fact"), None);

        // 3. Zero matching relations (declared Chaining relation exists but endpoint types don't
        // fit) -> None.
        let zero = Ontology::parse(
            r#"
[entities.Fact]
[entities.Other]
[relations.LEADS_TO]
from = ["Other"]
to = ["Other"]
role = "chaining"
"#,
        )
        .unwrap();
        assert_eq!(spine_edge_type(&zero, "Fact", "Fact"), None);

        // 4. A Grounding-role relation with matching endpoint types must be excluded, not chosen
        // — only Chaining relations are eligible spine edges.
        let grounding_only = Ontology::parse(
            r#"
[entities.Fact]
[relations.MENTIONS_FACT]
from = ["Fact"]
to = ["Fact"]
role = "grounding"
"#,
        )
        .unwrap();
        assert_eq!(spine_edge_type(&grounding_only, "Fact", "Fact"), None);
    }
}
