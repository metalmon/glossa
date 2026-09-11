//! Non-destructive relink: reconnect reasoning-node MENTIONS that point at a now-absent structural
//! target (`<dockey>#<n>` / `<dockey>`) to the LIVE structural node for the same file, matched by
//! filename + section suffix. This is `scratchpad/reprefix_graph.py` generalized from "add a label
//! prefix" to "match a relocated/relabelled doc against the graph's own current structural nodes".

use super::Triple;
use crate::graph::MENTIONS;
use std::collections::{HashMap, HashSet};

pub struct RelinkPlan {
    pub relinkable: Vec<(String, String, String)>,
    pub ambiguous: Vec<(String, String, Vec<String>)>,
    pub orphans: Vec<String>,
}

/// The match key for a structural target id: the segment after the last '/', or the WHOLE id when
/// there is no '/' (a bare/unlabeled key like `manual.pdf#12` — the discovery/empty-label case).
fn tail(id: &str) -> &str {
    id.rsplit('/').next().unwrap_or(id)
}

pub fn classify_relink(
    nodes: &[(String, String)],
    edges: &[Triple],
    ungrounded: &HashSet<String>,
) -> RelinkPlan {
    let existing: HashSet<&str> = nodes.iter().map(|(id, _)| id.as_str()).collect();

    // tail -> live structural node ids (doc-key-shaped ids that currently exist).
    let mut by_tail: HashMap<&str, Vec<&str>> = HashMap::new();
    for (id, _ty) in nodes {
        by_tail.entry(tail(id)).or_default().push(id.as_str());
    }

    let mut relinkable = Vec::new();
    let mut ambiguous = Vec::new();
    let mut recovered: HashSet<&str> = HashSet::new(); // reasoning ids that got at least one relink

    for (from, rel, to) in edges {
        if rel != MENTIONS || !ungrounded.contains(from.as_str()) {
            continue;
        }
        if existing.contains(to.as_str()) {
            continue; // live target — not our problem
        }
        let t = tail(to);
        let mut cands: Vec<&str> = by_tail
            .get(t)
            .map(|v| v.iter().copied().filter(|c| *c != to.as_str()).collect())
            .unwrap_or_default();
        cands.sort();
        cands.dedup();
        match cands.len() {
            0 => {} // no live doc with that filename+section → node stays orphan unless another edge recovers it
            1 => {
                relinkable.push((from.clone(), to.clone(), cands[0].to_string()));
                recovered.insert(from.as_str());
            }
            _ => ambiguous.push((from.clone(), to.clone(), cands.iter().map(|s| s.to_string()).collect())),
        }
    }

    // Orphans: ungrounded reasoning nodes that got neither a relink nor an ambiguous candidate.
    let ambiguous_from: HashSet<&str> = ambiguous.iter().map(|(f, _, _)| f.as_str()).collect();
    let mut orphans: Vec<String> = ungrounded
        .iter()
        .filter(|id| !recovered.contains(id.as_str()) && !ambiguous_from.contains(id.as_str()))
        .cloned()
        .collect();

    relinkable.sort();
    ambiguous.sort();
    orphans.sort();
    orphans.dedup();
    RelinkPlan { relinkable, ambiguous, orphans }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(p: &[(&str, &str)]) -> Vec<(String, String)> {
        p.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
    }
    fn e(t: &[(&str, &str, &str)]) -> Vec<Triple> {
        t.iter().map(|(a, b, c)| (a.to_string(), b.to_string(), c.to_string())).collect()
    }
    fn set(x: &[&str]) -> HashSet<String> {
        x.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn relinks_label_shift_by_filename_and_section() {
        // live section node is under the new "plc/" label; the reasoning MENTIONS points at the old bare key.
        let nodes = n(&[
            ("res:a", "Resolution"),
            ("plc/manual.pdf#12", "Section"), // live target (new key form)
        ]);
        let edges = e(&[("res:a", "MENTIONS", "manual.pdf#12")]); // dead: bare old key
        let plan = classify_relink(&nodes, &edges, &set(&["res:a"]));
        assert_eq!(plan.relinkable, vec![("res:a".into(), "manual.pdf#12".into(), "plc/manual.pdf#12".into())]);
        assert!(plan.ambiguous.is_empty() && plan.orphans.is_empty());
    }

    #[test]
    fn relinks_folder_move_same_filename() {
        let nodes = n(&[("res:a", "Resolution"), ("arch/2024/manual.pdf#3", "Section")]);
        let edges = e(&[("res:a", "MENTIONS", "plc/manual.pdf#3")]); // file moved plc/ -> arch/2024/
        let plan = classify_relink(&nodes, &edges, &set(&["res:a"]));
        assert_eq!(plan.relinkable, vec![("res:a".into(), "plc/manual.pdf#3".into(), "arch/2024/manual.pdf#3".into())]);
    }

    #[test]
    fn ambiguous_when_two_live_targets_share_filename_and_section() {
        let nodes = n(&[
            ("res:a", "Resolution"),
            ("plc/manual.pdf#1", "Section"),
            ("ivk/manual.pdf#1", "Section"),
        ]);
        let edges = e(&[("res:a", "MENTIONS", "manual.pdf#1")]);
        let plan = classify_relink(&nodes, &edges, &set(&["res:a"]));
        assert!(plan.relinkable.is_empty());
        assert_eq!(plan.ambiguous.len(), 1);
        assert_eq!(plan.ambiguous[0].0, "res:a");
        let mut cands = plan.ambiguous[0].2.clone();
        cands.sort();
        assert_eq!(cands, vec!["ivk/manual.pdf#1".to_string(), "plc/manual.pdf#1".to_string()]);
    }

    #[test]
    fn orphan_when_filename_absent_or_no_mentions() {
        let nodes = n(&[("res:gone", "Resolution"), ("res:none", "Resolution"), ("plc/other.pdf#1", "Section")]);
        let edges = e(&[("res:gone", "MENTIONS", "plc/deleted.pdf#1")]); // deleted.pdf not in index; res:none has no MENTIONS
        let plan = classify_relink(&nodes, &edges, &set(&["res:gone", "res:none"]));
        assert!(plan.relinkable.is_empty() && plan.ambiguous.is_empty());
        let mut o = plan.orphans.clone();
        o.sort();
        assert_eq!(o, vec!["res:gone".to_string(), "res:none".to_string()]);
    }

    #[test]
    fn multi_part_shift_classified_independently() {
        let nodes = n(&[
            ("res:p", "Resolution"),
            ("res:i", "Resolution"),
            ("plc/a.pdf#1", "Section"),
            ("ivk/b.pdf#2", "Section"),
        ]);
        let edges = e(&[("res:p", "MENTIONS", "a.pdf#1"), ("res:i", "MENTIONS", "b.pdf#2")]);
        let plan = classify_relink(&nodes, &edges, &set(&["res:p", "res:i"]));
        let mut r = plan.relinkable.clone();
        r.sort();
        assert_eq!(
            r,
            vec![
                ("res:i".into(), "b.pdf#2".into(), "ivk/b.pdf#2".into()),
                ("res:p".into(), "a.pdf#1".into(), "plc/a.pdf#1".into()),
            ]
        );
    }

    #[test]
    fn node_with_one_relinkable_and_one_orphan_target_is_not_orphan() {
        let nodes = n(&[("res:a", "Resolution"), ("plc/a.pdf#1", "Section")]);
        let edges = e(&[("res:a", "MENTIONS", "a.pdf#1"), ("res:a", "MENTIONS", "gone.pdf#9")]);
        let plan = classify_relink(&nodes, &edges, &set(&["res:a"]));
        assert_eq!(plan.relinkable, vec![("res:a".into(), "a.pdf#1".into(), "plc/a.pdf#1".into())]);
        assert!(plan.orphans.is_empty(), "a node recovered by one target is not an orphan");
    }
}
