//! `kbx build` stage 3 — entity-group, chunk-grounded bridge judge (Task 8 of the `kbx build`
//! pipeline; entity-group redesign, replacing the earlier pairwise judge).
//!
//! Consumes stage 2's mechanical candidate groups (`candidates::candidate_groups`) and, for each
//! group NOT already judged (per the `Checkpoint`), makes ONE model call passing EVERY member fact
//! (label + grounded chunk text, via `chunks::chunk_text`) at once, and asks the model to return an
//! arbitrary set of directed `LEADS_TO` links among them — a chain, a branch, a hub, several
//! disjoint links, or none are all valid replies (`bridge.md`). This replaces the old pair-at-a-time
//! design: instead of C(n,2) isolated yes/no calls per entity, the model sees the WHOLE entity in
//! one call and self-resolves generic/co-mention-only entities to "no links", so no mechanical
//! frequency cap is needed to bound cost or noise.
//!
//! `kbx build` groups are always Fact->Fact (Task 2 pins extraction to a single flat `Fact` node
//! type), so a written link's `edge_type` is hardcoded to `glossa::graph::LEADS_TO` via
//! `spine_edge_for_build` rather than resolved from the ontology by role/endpoint types — the
//! ontology-general resolution this module used to do (picking a declared `RelationRole::Chaining`
//! relation whose `from`/`to` fit the pair) is left to a later `reason` stage that re-types the
//! flat build graph against a real ontology. Every link is written through the SAME canonical
//! upsert path (`apply_upsert`) the rest of `kbx build` already uses — `apply_upsert` still
//! validates the write; `LEADS_TO` is always-permitted by `Ontology::validate_edge` regardless of
//! what's declared (Task 1).
//!
//! Checkpointing is per GROUP (`judge:group:<entity>`), not per link: "no links" is itself a
//! judged outcome and is marked done, so a `--resume` run doesn't re-ask the model about an entity
//! it already decided has no genuine cross-doc steps.

use crate::backend::openai::chat_once;
use crate::build::candidates::CandidateGroup;
use crate::build::chunks::chunk_text;
use crate::checkpoint::Checkpoint;
use crate::lab::{Endpoint, LabConfig};
use anyhow::Context;
use glossa::graph::agent::{apply_upsert, EdgeSpec};
use glossa::graph::ontology::Ontology;
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;
use indicatif::ProgressBar;
use serde::Deserialize;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// How much of the candidate group list a `run_judge` pass judged/linked. `judged` counts GROUPS
/// (one model call each), not individual links; `linked` counts the links actually written.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct JudgeStats {
    pub judged: usize,
    pub linked: usize,
}

/// One member fact rendered for the group judge prompt: its node id (the handle the model must
/// echo back in a link), label, grounding document, and grounded source chunk text.
#[derive(Debug, Clone)]
pub struct GroupFact {
    pub id: String,
    pub label: String,
    pub doc: String,
    pub chunk: String,
}

/// A single `{"from": ..., "to": ...}` link as the model returns it, before member/self-loop
/// validation.
#[derive(Debug, Deserialize)]
struct LinkRaw {
    from: String,
    to: String,
}

/// Extract the reply's link-set: find the first top-level JSON array in the reply text (tolerates
/// a fenced ```json block or trailing prose around it — the same leniency `parse_yes_no` used for
/// the old VERDICT line), parse it as `[{"from": "<id>", "to": "<id>"}, ...]`, then keep only links
/// whose BOTH ends are in `member_ids` and drop self-loops. Any parse failure, a missing `[`/`]`,
/// or an empty array all fall back to "no links" — the conservative default, matching this
/// module's bias toward thin/correct over dense.
fn parse_links(reply: &str, member_ids: &HashSet<&str>) -> Vec<(String, String)> {
    let Some(start) = reply.find('[') else {
        return Vec::new();
    };
    let Some(end) = reply.rfind(']') else {
        return Vec::new();
    };
    if end < start {
        return Vec::new();
    }
    let Ok(raw) = serde_json::from_str::<Vec<LinkRaw>>(&reply[start..=end]) else {
        return Vec::new();
    };

    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut out = Vec::new();
    for link in raw {
        if link.from == link.to {
            continue; // self-loop
        }
        if !member_ids.contains(link.from.as_str()) || !member_ids.contains(link.to.as_str()) {
            continue; // names a non-member id
        }
        if seen.insert((link.from.clone(), link.to.clone())) {
            out.push((link.from, link.to));
        }
    }
    out
}

/// Size guard (prompt-fit, not recall): cap `facts` to the first `max` entries (deterministic —
/// facts arrive in `CandidateGroup::members`' sorted order), returning whether truncation
/// happened so the caller can log it. `max == 0` is treated as "no cap" (defensive: a
/// misconfigured 0 must not silently judge zero facts).
fn cap_facts(mut facts: Vec<GroupFact>, max: usize) -> (Vec<GroupFact>, bool) {
    if max == 0 || facts.len() <= max {
        return (facts, false);
    }
    facts.truncate(max);
    (facts, true)
}

/// One entity-group bridge decision: system = `bridge_md` (the group link-set criterion), user =
/// every member fact's label + grounded chunk text. Posts to `ep` via `chat_once` (temp 0, no
/// tools — same substrate `judge::judge` and the old pairwise judge used) and parses the reply
/// with `parse_links`.
pub fn judge_group(
    ep: &Endpoint,
    bridge_md: &str,
    entity: &str,
    facts: &[GroupFact],
) -> anyhow::Result<Vec<(String, String)>> {
    let api_key = ep.resolve_key();

    let mut user = format!("Entity: {entity}\n\n");
    for (i, f) in facts.iter().enumerate() {
        user.push_str(&format!(
            "--- Fact {n} ---\nid: {id}\ndocument: {doc}\nlabel: {label}\nsource text:\n{chunk}\n\n",
            n = i + 1,
            id = f.id,
            doc = f.doc,
            label = f.label,
            chunk = f.chunk,
        ));
    }
    user.push_str(
        "Reply with a JSON array of the genuine links you found, each exactly \
         `{\"from\": \"<id>\", \"to\": \"<id>\"}` using the exact `id` values shown above. \
         If there are no genuine links, reply with an empty array `[]`.",
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
    .context("bridge group judge endpoint request failed")?;
    let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");

    let member_ids: HashSet<&str> = facts.iter().map(|f| f.id.as_str()).collect();
    Ok(parse_links(content, &member_ids))
}

/// The edge type `kbx build`'s judge writes for every link. Build groups are always Fact->Fact
/// (extraction is pinned to a single flat `Fact` node type — Task 2), so there's no per-link
/// resolution to do: every written link is always `glossa::graph::LEADS_TO`. Factored to a
/// function (rather than inlined at the call site) purely as a test seam — asserting the constant
/// directly wouldn't prove the judge's write path actually uses it.
fn spine_edge_for_build() -> &'static str {
    glossa::graph::LEADS_TO
}

/// Judge every candidate group not already checkpointed: fetch every surviving member's grounded
/// chunk text, cap to `max_facts` (logging any truncation), ask `judge_group` once, and write each
/// returned link as a spine edge (`apply_upsert`) with `edge_type` fixed to
/// `spine_edge_for_build()`. Marks the checkpoint for EVERY judged group, empty-link-set or not —
/// an empty result leaves no graph trace, so the checkpoint is the only thing that stops a
/// `--resume` run from re-judging it.
pub fn run_judge(
    root: &Path,
    lab: &LabConfig,
    bridge_md: &str,
    g: &GraphStore,
    idx: &DocIndex,
    groups: &[CandidateGroup],
    cp: &Checkpoint,
    pb: &ProgressBar,
    max_facts: usize,
) -> anyhow::Result<JudgeStats> {
    let ont = Ontology::load_or_default(root);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut stats = JudgeStats::default();

    for group in groups {
        let unit_id = format!("judge:group:{}", group.entity);
        if cp.is_done(&unit_id) {
            pb.inc(1);
            continue;
        }

        // Collect surviving members only — a candidate referencing a node that's since vanished
        // (deleted/merged away) is silently skipped rather than failing the whole group.
        let mut facts: Vec<GroupFact> = Vec::new();
        for id in &group.members {
            let Some(node) = g.get_node(id)? else {
                continue;
            };
            facts.push(GroupFact {
                id: id.clone(),
                label: node.label.clone(),
                doc: node.prov.source_path.clone(),
                chunk: chunk_text(root, idx, id, g),
            });
        }

        if facts.len() < 2 {
            // Fewer than two surviving members can't bridge anything — nothing to judge.
            cp.mark(&unit_id, "no-members")?;
            pb.inc(1);
            continue;
        }

        let (facts, truncated) = cap_facts(facts, max_facts);
        if truncated {
            println!(
                "judge: group '{}' has more than --bridge-max-facts {max_facts} fact(s); \
                 judging only the first {max_facts}",
                group.entity
            );
        }

        let doc_by_id: HashMap<&str, &str> =
            facts.iter().map(|f| (f.id.as_str(), f.doc.as_str())).collect();

        let links = judge_group(
            lab.bridge.as_ref().unwrap_or(&lab.model),
            bridge_md,
            &group.entity,
            &facts,
        )?;
        stats.judged += 1;

        for (from, to) in &links {
            let source_path = doc_by_id.get(from.as_str()).copied().unwrap_or_default();
            let edge = EdgeSpec {
                from: from.clone(),
                to: to.clone(),
                edge_type: spine_edge_for_build().to_string(),
                source_path: source_path.to_string(),
                range: None,
                confidence: None,
            };
            apply_upsert(g, &ont, Vec::new(), vec![edge], now, root, "agent")?;
            stats.linked += 1;
        }

        cp.mark(&unit_id, if links.is_empty() { "none" } else { "linked" })?;
        pb.inc(1);
    }

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fact(id: &str) -> GroupFact {
        GroupFact {
            id: id.to_string(),
            label: id.to_string(),
            doc: "d.md".to_string(),
            chunk: "text".to_string(),
        }
    }

    #[test]
    fn parse_links_keeps_only_member_to_member_links() {
        let facts = [fact("a"), fact("b"), fact("c")];
        let member_ids: HashSet<&str> = facts.iter().map(|f| f.id.as_str()).collect();
        let reply = r#"reasoning...
[{"from": "a", "to": "b"}, {"from": "b", "to": "ghost"}, {"from": "c", "to": "c"}]"#;
        let links = parse_links(reply, &member_ids);
        assert_eq!(links, vec![("a".to_string(), "b".to_string())]);
    }

    #[test]
    fn parse_links_empty_array_is_no_links() {
        let facts = [fact("a"), fact("b")];
        let member_ids: HashSet<&str> = facts.iter().map(|f| f.id.as_str()).collect();
        assert!(parse_links("no genuine links: []", &member_ids).is_empty());
    }

    #[test]
    fn parse_links_unparseable_reply_falls_back_to_no_links() {
        let facts = [fact("a"), fact("b")];
        let member_ids: HashSet<&str> = facts.iter().map(|f| f.id.as_str()).collect();
        assert!(parse_links("I don't know.", &member_ids).is_empty());
    }

    #[test]
    fn parse_links_tolerates_a_fenced_json_block() {
        let facts = [fact("a"), fact("b")];
        let member_ids: HashSet<&str> = facts.iter().map(|f| f.id.as_str()).collect();
        let reply = "Here is the link-set:\n```json\n[{\"from\": \"a\", \"to\": \"b\"}]\n```\n";
        let links = parse_links(reply, &member_ids);
        assert_eq!(links, vec![("a".to_string(), "b".to_string())]);
    }

    #[test]
    fn parse_links_dedups_repeated_links() {
        let facts = [fact("a"), fact("b")];
        let member_ids: HashSet<&str> = facts.iter().map(|f| f.id.as_str()).collect();
        let reply = r#"[{"from": "a", "to": "b"}, {"from": "a", "to": "b"}]"#;
        let links = parse_links(reply, &member_ids);
        assert_eq!(links, vec![("a".to_string(), "b".to_string())]);
    }

    #[test]
    fn cap_facts_truncates_deterministically_and_reports_it() {
        let facts: Vec<GroupFact> = (0..5).map(|i| fact(&format!("f{i}"))).collect();
        let (capped, truncated) = cap_facts(facts, 3);
        assert!(truncated);
        assert_eq!(
            capped.iter().map(|f| f.id.as_str()).collect::<Vec<_>>(),
            vec!["f0", "f1", "f2"],
            "must keep the first N in order, not an arbitrary subset"
        );
    }

    #[test]
    fn cap_facts_is_a_noop_under_the_limit() {
        let facts: Vec<GroupFact> = (0..2).map(|i| fact(&format!("f{i}"))).collect();
        let (capped, truncated) = cap_facts(facts, 40);
        assert!(!truncated);
        assert_eq!(capped.len(), 2);
    }

    #[test]
    fn cap_facts_zero_max_is_treated_as_no_cap() {
        let facts: Vec<GroupFact> = (0..3).map(|i| fact(&format!("f{i}"))).collect();
        let (capped, truncated) = cap_facts(facts, 0);
        assert!(!truncated);
        assert_eq!(capped.len(), 3);
    }

    /// The pin's contract for the judge half: `kbx build` groups are always Fact->Fact, so every
    /// written link's edge type is always `glossa::graph::LEADS_TO` — no ontology resolution, no
    /// live model needed to prove it.
    #[test]
    fn build_judge_writes_leads_to_for_fact_groups() {
        assert_eq!(spine_edge_for_build(), glossa::graph::LEADS_TO);
    }
}
