//! Compositional narrowing: from a question's anchor entities, narrow the fact set to a small
//! grounded candidate list by pure graph math — df-gated alias-join, multi-anchor intersection,
//! IDF-ranked. A weak reader then answers by *reading* the few candidates instead of navigating.
//!
//! Universal + math-only: no learned model, no relation typing, no domain logic. The composition
//! the weak model fails at lives here; the model only supplies anchor strings + the query.

use crate::graph::ppr;
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
        // Include every reasoning node, ontology-agnostic: skip only the FIXED structural node types
        // (a system contract, not an ontology choice) — never a hardcoded domain type like "Fact".
        if crate::graph::STRUCTURAL_NODES.contains(&node.node_type.as_str()) {
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
                alias_facts
                    .entry(na.clone())
                    .or_default()
                    .push(node.id.clone());
                als.push(na);
            }
        }
        fact_aliases.insert(node.id.clone(), als);
    }
    let alias_df = alias_facts
        .iter()
        .map(|(a, v)| (a.clone(), v.len()))
        .collect();
    Ok(AliasIndex {
        fact_aliases,
        alias_facts,
        alias_df,
        fact_label,
        tok_df,
        n_facts,
    })
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
        let mut seeds: HashSet<String> = g
            .resolve(anc)?
            .into_iter()
            .filter(|id| idx.fact_label.contains_key(id))
            .collect();
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
    let qtoks: HashSet<String> = tokens(query)
        .into_iter()
        .filter(|t| !anchor_toks.contains(t))
        .collect();
    let idf =
        |t: &str| (1.0 + idx.n_facts as f32 / (1.0 + *idx.tok_df.get(t).unwrap_or(&0) as f32)).ln();
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
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(k);
    Ok(scored)
}

/// PPR-ranked composed candidates — the connectivity-based successor to [`compose`]. Seed a random
/// walk with restart from the question's lexical hits (`resolve` over the node index) plus the
/// anchor entity, rank every node by connectivity, and surface the top-`k` non-structural
/// (reasoning/evidence) nodes. Ontology-blind: the walk reads no edge_type/node_type, and the only
/// type check — skipping STRUCTURAL_NODES on output — is a fixed system contract, not an ontology
/// choice. Structural nodes still carry mass *through* the walk, which is how it bridges documents.
pub fn compose_ppr(
    g: &GraphStore,
    name: &str,
    query: &str,
    k: usize,
) -> anyhow::Result<Vec<Candidate>> {
    let trans = g.ppr_transition()?;
    if trans.is_empty() {
        return Ok(vec![]);
    }
    // Seed the restart vector: whole-question lexical hits get mass decreasing by BM25 rank; the
    // anchor entity the reader looked up gets a strong boost. No NER, no relation names.
    let mut seeds: HashMap<String, f32> = HashMap::new();
    for (rank, id) in g.resolve(query)?.into_iter().take(20).enumerate() {
        *seeds.entry(id).or_default() += 1.0 / (1.0 + rank as f32);
    }
    for id in g.resolve(name)?.into_iter().take(5) {
        *seeds.entry(id).or_default() += 2.0;
    }
    let ranked = ppr::ppr(&trans, &seeds, 0.15, 30, 1e-6);
    // (`trans` is Arc<Transition>; &trans deref-coerces to &Transition)
    if ranked.is_empty() {
        return Ok(vec![]);
    }
    // id -> (node_type, label), one pass; avoids a DB hit per ranked node.
    let meta: HashMap<String, (String, String)> = g
        .all_nodes()?
        .into_iter()
        .map(|n| (n.id, (n.node_type, n.label)))
        .collect();
    let structural: HashSet<&str> = crate::graph::STRUCTURAL_NODES.iter().copied().collect();
    let seed_ids: HashSet<&String> = seeds.keys().collect();
    let mut out = Vec::new();
    for (id, score) in ranked {
        if out.len() >= k {
            break;
        }
        if seed_ids.contains(&id) {
            continue; // the seeds are what the reader already has; surface what they LEAD to
        }
        if let Some((nt, label)) = meta.get(&id) {
            if structural.contains(nt.as_str()) {
                continue;
            }
            out.push(Candidate {
                id,
                label: label.clone(),
                score,
            });
        }
    }
    Ok(out)
}

/// Reciprocal-rank fusion of two candidate rankings into one. Score-unit-agnostic — it reads the
/// two *rankings*, never their raw (incomparable) scores: `score = wc/(k+rank_compose) +
/// wp/(k+rank_ppr)`, multiplied by `cons` when a node appears in BOTH lists (the consensus reward).
/// Dedup by id; label from whichever list first carries it. Returns the top-`out_k` by fused score.
/// The three research arms are just parameter settings: pure RRF (wc=wp=1, cons=1), consensus-boost
/// (cons>1), weighted (wc≠wp).
pub fn fuse(
    compose: &[Candidate],
    ppr: &[Candidate],
    k: f32,
    wc: f32,
    wp: f32,
    cons: f32,
    out_k: usize,
) -> Vec<Candidate> {
    // id -> (fused score, label, in_compose, in_ppr)
    let mut acc: HashMap<String, (f32, String, bool, bool)> = HashMap::new();
    for (r, c) in compose.iter().enumerate() {
        let e = acc
            .entry(c.id.clone())
            .or_insert((0.0, c.label.clone(), false, false));
        e.0 += wc / (k + r as f32);
        e.2 = true;
    }
    for (r, c) in ppr.iter().enumerate() {
        let e = acc
            .entry(c.id.clone())
            .or_insert((0.0, c.label.clone(), false, false));
        e.0 += wp / (k + r as f32);
        e.3 = true;
    }
    let mut out: Vec<Candidate> = acc
        .into_iter()
        .map(|(id, (mut s, label, in_c, in_p))| {
            if in_c && in_p {
                s *= cons;
            }
            Candidate {
                id,
                label,
                score: s,
            }
        })
        .collect();
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(out_k);
    out
}

/// Rank-damping constant for the fusion. A convention (the value RRF papers use is ~60; a smaller
/// constant just lets the top ranks matter more on these short candidate lists), NOT a knob tuned to
/// any corpus — the fusion stays parameter-free.
const RRF_K: f32 = 10.0;

/// The composed section: PARAMETER-FREE reciprocal-rank fusion of the lexical (`compose`) and
/// connectivity (`compose_ppr`) rankings. Equal source weights, no consensus multiplier, a fixed
/// rank-damping constant — nothing is fit to a particular corpus, so there is nothing to overfit.
/// A node ranked well by EITHER signal surfaces; a node ranked well by BOTH surfaces highest.
pub fn compose_hybrid(
    g: &GraphStore,
    name: &str,
    query: &str,
    k: usize,
) -> anyhow::Result<Vec<Candidate>> {
    let ppr = compose_ppr(g, name, query, k * 2)?;
    let aidx = build_alias_index(g)?;
    let comp = compose(g, &aidx, &[name], query, k * 2, 8)?;
    Ok(fuse(&comp, &ppr, RRF_K, 1.0, 1.0, 1.0, k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::store::{Edge, GraphStore, Node, Provenance};

    fn cand(id: &str) -> Candidate {
        Candidate {
            id: id.into(),
            label: id.into(),
            score: 0.0,
        }
    }

    #[test]
    fn fuse_floats_the_consensus_node_above_single_source_ones() {
        // X tops BOTH lists; A and B each appear in only one. Fusion must rank X first (it is the
        // agreement), with A and B still present.
        let compose = [cand("X"), cand("A")];
        let ppr = [cand("X"), cand("B")];
        let out = fuse(&compose, &ppr, 10.0, 1.0, 1.0, 1.0, 5);
        assert_eq!(
            out[0].id,
            "X",
            "consensus node ranks first: {:?}",
            out.iter().map(|c| &c.id).collect::<Vec<_>>()
        );
        assert!(
            out.iter().any(|c| c.id == "A") && out.iter().any(|c| c.id == "B"),
            "both singles kept"
        );
    }

    #[test]
    fn consensus_multiplier_raises_a_both_node_score() {
        let compose = [cand("A"), cand("X")]; // X at rank 1 in compose
        let ppr = [cand("B"), cand("X")]; // X at rank 1 in ppr
        let plain = fuse(&compose, &ppr, 10.0, 1.0, 1.0, 1.0, 5);
        let boosted = fuse(&compose, &ppr, 10.0, 1.0, 1.0, 2.0, 5);
        let sx = |o: &[Candidate]| o.iter().find(|c| c.id == "X").unwrap().score;
        assert!(
            sx(&boosted) > sx(&plain),
            "cons>1 strictly raises the both-node score"
        );
    }

    #[test]
    fn weighted_fusion_favors_the_heavier_source() {
        // A leads compose, B leads ppr; nothing is shared. Weighting ppr heavier must put B on top.
        let compose = [cand("A")];
        let ppr = [cand("B")];
        let out = fuse(&compose, &ppr, 10.0, 1.0, 3.0, 1.0, 5);
        assert_eq!(
            out[0].id,
            "B",
            "heavier ppr weight wins: {:?}",
            out.iter().map(|c| (&c.id, c.score)).collect::<Vec<_>>()
        );
    }

    fn prov() -> Provenance {
        Provenance {
            source_path: "s.md".into(),
            range: None,
            file_sig: None,
            origin: "agent".into(),
            confidence: 1.0,
            created_at: 0,
        }
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
        fact(
            &g,
            "prize",
            "Prize2020 went to the winner",
            &["Prize2020", "winner"],
        );
        for i in 0..20 {
            fact(
                &g,
                &format!("hub{i}"),
                "someone at Uni",
                &["Uni", &format!("p{i}")],
            );
        }
        let idx = build_alias_index(&g).unwrap();
        let seeds: HashSet<String> = ["ann".to_string()].into_iter().collect();
        // df_cap 5: 'uni' (df 21) is skipped, 'prize2020' (df 2) is followed.
        let r = reachable(&idx, &seeds, 5, 3, 1000);
        assert!(r.contains("prize"), "specific bridge followed");
        assert!(!r.contains("hub0"), "generic hub NOT followed");
    }

    fn link(g: &GraphStore, a: &str, b: &str) {
        g.put_edge(&Edge {
            from: a.into(),
            to: b.into(),
            edge_type: "LEADS_TO".into(),
            prov: prov(),
        })
        .unwrap();
    }

    #[test]
    fn ppr_transition_persists_to_disk_and_survives_reopen() {
        let d = tempfile::tempdir().unwrap();
        {
            let g = GraphStore::open(d.path()).unwrap();
            fact(&g, "a", "A", &[]);
            fact(&g, "b", "B", &[]);
            link(&g, "a", "b");
            assert_eq!(g.ppr_transition().unwrap().len(), 2);
        }
        assert!(
            d.path().join(".glossa/ppr_transition.json").exists(),
            "transition cache persisted to disk"
        );
        // A fresh handle (new process, conceptually) loads it from disk — counts match, no rebuild.
        let g2 = GraphStore::open(d.path()).unwrap();
        assert_eq!(
            g2.ppr_transition().unwrap().len(),
            2,
            "loads across a reopen"
        );
    }

    #[test]
    fn compose_ppr_surfaces_the_connected_terminal_not_a_lexical_lookalike() {
        // Path-described question: the terminal answer shares NO tokens with the question, so the
        // old lexical rank cannot float it. PPR must, via connectivity: seed on the anchor, walk
        // Ann -> bridge -> terminal. A disconnected fact of the SAME shape ("first in the world")
        // must not surface — it has the words' shape but no path from the anchor.
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        fact(&g, "ann", "Ann", &["Ann"]);
        fact(
            &g,
            "bridge",
            "Ann studied at Redbrick University",
            &["Redbrick"],
        );
        fact(&g, "terminal", "ranked seventh in the world", &[]);
        fact(&g, "distractor", "ranked first in the world", &[]);
        link(&g, "ann", "bridge");
        link(&g, "bridge", "terminal");
        // distractor is intentionally isolated.
        // Seed only on the anchor (exact resolve, no BM25 contamination): terminal and distractor
        // draw NO direct seed mass, so the terminal can only arrive via connectivity — and the
        // disconnected lexical lookalike cannot arrive at all.
        let out = compose_ppr(&g, "Ann", "Ann", 5).unwrap();
        let labels: Vec<&str> = out.iter().map(|c| c.label.as_str()).collect();
        assert!(
            labels.iter().any(|l| l.contains("seventh in the world")),
            "connected terminal surfaces: {labels:?}"
        );
        assert!(
            !labels.iter().any(|l| l.contains("first in the world")),
            "disconnected lookalike does NOT surface: {labels:?}"
        );
    }

    #[test]
    fn compose_ranks_answer_fact_in_top_k() {
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        // 2-hop: Ann -> Redbrick (a specific, moderate-df bridge) -> its worldwide ranking.
        // Redbrick's other facts are noise on the same bridge; IDF on the query's rare tokens
        // ("worldwide", "ranking") must float the answer fact to the top.
        fact(&g, "a", "Ann studied at Redbrick", &["Ann", "Redbrick"]);
        fact(
            &g,
            "b",
            "Redbrick worldwide ranking is seventh",
            &["Redbrick"],
        );
        fact(&g, "c", "Redbrick was founded long ago", &["Redbrick"]);
        fact(&g, "e", "Redbrick has many students", &["Redbrick"]);
        // generic-hub noise elsewhere in the corpus (df high) — must not pull in via the gate.
        for i in 0..15 {
            fact(
                &g,
                &format!("d{i}"),
                "some city fact",
                &[&format!("City{i}"), "Common"],
            );
        }
        let idx = build_alias_index(&g).unwrap();
        let out = compose(
            &g,
            &idx,
            &["Ann"],
            "worldwide ranking of Ann university",
            5,
            10,
        )
        .unwrap();
        assert!(!out.is_empty());
        assert!(
            out[0].label.contains("worldwide ranking"),
            "answer fact ranked first: {:?}",
            out.iter().map(|c| &c.label).collect::<Vec<_>>()
        );
    }
}
