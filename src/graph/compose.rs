//! Compositional narrowing: from a question's anchor entities, narrow the fact set to a small
//! grounded candidate list by pure graph math — df-gated alias-join, multi-anchor intersection,
//! IDF-ranked. A weak reader then answers by *reading* the few candidates instead of navigating.
//!
//! Universal + math-only: no learned model, no relation typing, no domain logic. The composition
//! the weak model fails at lives here; the model only supplies anchor strings + the query.

use crate::graph::store::GraphStore;
use std::collections::{HashMap, HashSet, VecDeque};

const STOP: &[&str] = &[
    "the", "a", "an", "of", "in", "on", "to", "is", "was", "were", "are", "for", "and", "or",
    "that", "which", "who", "what", "when", "where", "by", "with", "at", "as", "from", "into",
    "does", "did", "do", "part", "named", "after", "held", "over", "it", "its", "his", "her",
    "their", "he", "she",
];

fn normalize(s: &str) -> String {
    let mut out = String::new();
    let mut prev_sp = false;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            prev_sp = false;
        } else if !prev_sp {
            out.push(' ');
            prev_sp = true;
        }
    }
    out.trim().to_string()
}

fn tokens(s: &str) -> Vec<String> {
    normalize(s)
        .split(' ')
        .filter(|w| w.len() > 1 && !STOP.contains(w))
        .map(|w| w.to_string())
        .collect()
}

/// Alias-join index + document-frequency statistics, built once from the graph's Fact nodes.
pub struct AliasIndex {
    fact_aliases: HashMap<String, Vec<String>>, // fact id -> normalized aliases (df ≥ 3 chars)
    alias_facts: HashMap<String, Vec<String>>,  // normalized alias -> fact ids
    alias_df: HashMap<String, usize>,           // alias -> how many facts carry it
    fact_label: HashMap<String, String>,
    tok_df: HashMap<String, usize>, // label token -> fact count (for IDF)
    n_facts: usize,
}

pub fn build_alias_index(g: &GraphStore) -> anyhow::Result<AliasIndex> {
    let mut fact_aliases: HashMap<String, Vec<String>> = HashMap::new();
    let mut alias_facts: HashMap<String, Vec<String>> = HashMap::new();
    let mut fact_label: HashMap<String, String> = HashMap::new();
    let mut tok_df: HashMap<String, usize> = HashMap::new();
    let mut n_facts = 0usize;
    for node in g.all_nodes()? {
        if node.node_type != "Fact" {
            continue;
        }
        n_facts += 1;
        fact_label.insert(node.id.clone(), node.label.clone());
        for t in tokens(&node.label).into_iter().collect::<HashSet<_>>() {
            *tok_df.entry(t).or_default() += 1;
        }
        let mut als = Vec::new();
        let mut seen = HashSet::new();
        for a in &node.aliases {
            let na = normalize(a);
            if na.len() >= 3 && seen.insert(na.clone()) {
                alias_facts.entry(na.clone()).or_default().push(node.id.clone());
                als.push(na);
            }
        }
        fact_aliases.insert(node.id.clone(), als);
    }
    let alias_df = alias_facts.iter().map(|(a, v)| (a.clone(), v.len())).collect();
    Ok(AliasIndex { fact_aliases, alias_facts, alias_df, fact_label, tok_df, n_facts })
}

/// Facts reachable from `seeds` by joining on SPECIFIC aliases only (df ≤ `df_cap` — generic hub
/// entities are skipped, which is what keeps the intersection clean). Bounded by `max_hop` and
/// `visit_cap`.
pub fn reachable(
    idx: &AliasIndex,
    seeds: &HashSet<String>,
    df_cap: usize,
    max_hop: usize,
    visit_cap: usize,
) -> HashSet<String> {
    let mut seen: HashSet<String> = seeds.clone();
    let mut out: HashSet<String> = seeds.clone();
    let mut q: VecDeque<(String, usize)> = seeds.iter().map(|s| (s.clone(), 0)).collect();
    while let Some((f, h)) = q.pop_front() {
        if seen.len() >= visit_cap {
            break;
        }
        if h >= max_hop {
            continue;
        }
        if let Some(als) = idx.fact_aliases.get(&f) {
            for a in als {
                if idx.alias_df.get(a).copied().unwrap_or(0) > df_cap {
                    continue; // skip generic hub aliases
                }
                if let Some(fs) = idx.alias_facts.get(a) {
                    for nf in fs {
                        if seen.insert(nf.clone()) {
                            out.insert(nf.clone());
                            q.push_back((nf.clone(), h + 1));
                        }
                    }
                }
            }
        }
    }
    out
}

/// A narrowed, grounded candidate fact.
pub struct Candidate {
    pub id: String,
    pub label: String,
    pub score: f32,
}

/// Narrow to the top-`k` candidate facts: df-gated reachable set per anchor, INTERSECT across
/// anchors (union fallback if the intersection is empty), IDF-rank by the query's non-anchor tokens.
pub fn compose(
    g: &GraphStore,
    idx: &AliasIndex,
    anchors: &[&str],
    query: &str,
    k: usize,
    df_cap: usize,
) -> anyhow::Result<Vec<Candidate>> {
    let mut reach_sets: Vec<HashSet<String>> = Vec::new();
    let mut anchor_toks: HashSet<String> = HashSet::new();
    for anc in anchors {
        for t in tokens(anc) {
            anchor_toks.insert(t);
        }
        // Seed via the graph's own resolve (BM25/IDF over labels+aliases) — like the kb-test
        // symptom lookup, it handles a rich phrase by weighting the DISCRIMINATIVE tokens, so a
        // topic like "<entity> <what you want>" still lands on the entity's facts, not on generic
        // high-frequency words. Fall back to a tolerant alias-substring match only if resolve is dry.
        let mut seeds: HashSet<String> =
            g.resolve(anc)?.into_iter().filter(|id| idx.fact_label.contains_key(id)).collect();
        if seeds.is_empty() {
            let na = normalize(anc);
            for (alias, fs) in &idx.alias_facts {
                if (alias.contains(&na) || na.contains(alias.as_str()))
                    && (alias.len() as i64 - na.len() as i64).abs() < 12
                {
                    for f in fs {
                        seeds.insert(f.clone());
                    }
                }
            }
        }
        reach_sets.push(reachable(idx, &seeds, df_cap, 3, 4000));
    }
    let cand: HashSet<String> = if reach_sets.len() > 1 {
        let mut it = reach_sets.iter();
        let mut acc = it.next().cloned().unwrap_or_default();
        for s in it {
            acc = acc.intersection(s).cloned().collect();
        }
        if acc.is_empty() {
            reach_sets.iter().flatten().cloned().collect() // union fallback
        } else {
            acc
        }
    } else {
        reach_sets.into_iter().next().unwrap_or_default()
    };
    if cand.is_empty() {
        return Ok(vec![]);
    }
    let qtoks: HashSet<String> =
        tokens(query).into_iter().filter(|t| !anchor_toks.contains(t)).collect();
    let idf = |t: &str| (1.0 + idx.n_facts as f32 / (1.0 + *idx.tok_df.get(t).unwrap_or(&0) as f32)).ln();
    let mut scored: Vec<Candidate> = cand
        .into_iter()
        .map(|id| {
            let ft: HashSet<String> = idx
                .fact_label
                .get(&id)
                .map(|l| tokens(l).into_iter().collect())
                .unwrap_or_default();
            let score: f32 = qtoks.intersection(&ft).map(|t| idf(t)).sum();
            let label = idx.fact_label.get(&id).cloned().unwrap_or_default();
            Candidate { id, label, score }
        })
        .collect();
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    Ok(scored)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::store::{GraphStore, Node, Provenance};

    fn prov() -> Provenance {
        Provenance { source_path: "s.md".into(), range: None, file_sig: None, origin: "agent".into(), confidence: 1.0, created_at: 0 }
    }
    fn fact(g: &GraphStore, id: &str, label: &str, aliases: &[&str]) {
        g.put_node(&Node {
            id: id.into(),
            node_type: "Fact".into(),
            label: label.into(),
            aliases: aliases.iter().map(|s| s.to_string()).collect(),
            prov: prov(),
        })
        .unwrap();
    }

    #[test]
    fn alias_index_counts_df_and_maps_specific_aliases() {
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        fact(&g, "f1", "Ann studied at Uni", &["Ann", "Uni"]);
        fact(&g, "f2", "Uni ranked first", &["Uni", "ranking"]);
        fact(&g, "f3", "Bob studied at Uni", &["Bob", "Uni"]);
        let idx = build_alias_index(&g).unwrap();
        assert_eq!(idx.alias_df.get("uni").copied(), Some(3)); // generic hub
        assert_eq!(idx.alias_df.get("ann").copied(), Some(1)); // specific
        assert_eq!(idx.alias_facts.get("ranking").unwrap().len(), 1);
    }

    #[test]
    fn reachable_skips_generic_hub_aliases() {
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        // Ann -Uni(hub)- everyone; the specific bridge Ann->Prize is what we want, not the hub.
        fact(&g, "ann", "Ann studied", &["Ann", "Uni", "Prize2020"]);
        fact(&g, "prize", "Prize2020 went to the winner", &["Prize2020", "winner"]);
        for i in 0..20 {
            fact(&g, &format!("hub{i}"), "someone at Uni", &["Uni", &format!("p{i}")]);
        }
        let idx = build_alias_index(&g).unwrap();
        let seeds: HashSet<String> = ["ann".to_string()].into_iter().collect();
        // df_cap 5: 'uni' (df 21) is skipped, 'prize2020' (df 2) is followed.
        let r = reachable(&idx, &seeds, 5, 3, 1000);
        assert!(r.contains("prize"), "specific bridge followed");
        assert!(!r.contains("hub0"), "generic hub NOT followed");
    }

    #[test]
    fn compose_ranks_answer_fact_in_top_k() {
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        // 2-hop: Ann -> Redbrick (a specific, moderate-df bridge) -> its worldwide ranking.
        // Redbrick's other facts are noise on the same bridge; IDF on the query's rare tokens
        // ("worldwide", "ranking") must float the answer fact to the top.
        fact(&g, "a", "Ann studied at Redbrick", &["Ann", "Redbrick"]);
        fact(&g, "b", "Redbrick worldwide ranking is seventh", &["Redbrick"]);
        fact(&g, "c", "Redbrick was founded long ago", &["Redbrick"]);
        fact(&g, "e", "Redbrick has many students", &["Redbrick"]);
        // generic-hub noise elsewhere in the corpus (df high) — must not pull in via the gate.
        for i in 0..15 {
            fact(&g, &format!("d{i}"), "some city fact", &[&format!("City{i}"), "Common"]);
        }
        let idx = build_alias_index(&g).unwrap();
        let out = compose(&g, &idx, &["Ann"], "worldwide ranking of Ann university", 5, 10).unwrap();
        assert!(!out.is_empty());
        assert!(
            out[0].label.contains("worldwide ranking"),
            "answer fact ranked first: {:?}",
            out.iter().map(|c| &c.label).collect::<Vec<_>>()
        );
    }
}
