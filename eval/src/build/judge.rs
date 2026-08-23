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
//! `kbx build` pairs are always Fact->Fact (Task 2 pins extraction to a single flat `Fact` node
//! type), so the spine edge's `edge_type` is hardcoded to `glossa::graph::LEADS_TO` via
//! `spine_edge_for_build` rather than resolved from the ontology by role/endpoint types — the
//! ontology-general resolution this module used to do (picking a declared `RelationRole::Chaining`
//! relation whose `from`/`to` fit the pair) is left to a later `distil` stage that re-types the
//! flat build graph against a real ontology. `apply_upsert` still validates the write; `LEADS_TO`
//! is always-permitted by `Ontology::validate_edge` regardless of what's declared (Task 1).

use crate::backend::openai::chat_once;
use crate::build::chunks::chunk_text;
use crate::build::candidates::CandidatePair;
use crate::checkpoint::Checkpoint;
use crate::lab::{Endpoint, LabConfig};
use anyhow::Context;
use glossa::graph::agent::{apply_upsert, EdgeSpec};
use glossa::graph::ontology::Ontology;
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

/// The edge type `kbx build`'s judge writes on a YES verdict. Build pairs are always Fact->Fact
/// (extraction is pinned to a single flat `Fact` node type — Task 2), so there's no per-pair
/// resolution to do: the spine edge is always `glossa::graph::LEADS_TO`. Factored to a function
/// (rather than inlined at the call site) purely as a test seam — asserting the constant directly
/// wouldn't prove the judge's write path actually uses it.
fn spine_edge_for_build() -> &'static str {
    glossa::graph::LEADS_TO
}

/// Judge every candidate pair not already checkpointed: fetch both nodes' grounded chunk text,
/// ask `judge_pair`, and on YES write one spine edge (`apply_upsert`) with `edge_type` fixed to
/// `spine_edge_for_build()`. Marks the checkpoint for EVERY judged pair, YES or NO — a NO leaves
/// no graph trace, so the checkpoint is the only thing that stops a `--resume` run from
/// re-judging it.
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
            let edge = EdgeSpec {
                from: pair.a.clone(),
                to: pair.b.clone(),
                edge_type: spine_edge_for_build().to_string(),
                source_path: node_a.prov.source_path.clone(),
                range: None,
                confidence: None,
            };
            apply_upsert(g, &ont, Vec::new(), vec![edge], now, root)?;
            stats.linked += 1;
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

    /// The pin's contract for the judge half: `kbx build` pairs are always Fact->Fact, so the
    /// YES-write's edge type is always `glossa::graph::LEADS_TO` — no ontology resolution, no
    /// live model needed to prove it.
    #[test]
    fn build_judge_writes_leads_to_for_fact_pairs() {
        assert_eq!(spine_edge_for_build(), glossa::graph::LEADS_TO);
    }
}
