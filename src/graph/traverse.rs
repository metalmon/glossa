use crate::graph::ontology::{Ontology, RelationRole};
use crate::graph::store::{normalize_label, Edge, GraphStore};
use std::collections::{HashSet, VecDeque};

fn type_match(e: &Edge, edge_types: Option<&[String]>) -> bool {
    match edge_types {
        None => true,
        Some(types) => types.iter().any(|t| t == &e.edge_type),
    }
}

/// How a `reach` hop was reached from the previous node.
#[derive(Debug, Clone, PartialEq)]
pub enum HopVia {
    /// An in-graph edge. `forward=true` means the edge points prev->node (outgoing); `false`
    /// means node->prev (an incoming edge, walked backward for connectivity).
    Edge { edge_type: String, forward: bool },
    /// A cross-document corpus bridge: the previous node dead-ended in its document and we
    /// resolved its `term` to a same-mention node in another document. `confidence` (§7.5) is
    /// higher for an exact, unambiguous match and lower for a fuzzy / namesake-heavy one — the
    /// reader/verify step can distrust a weak bridge (never a silent jump).
    Bridge { term: String, confidence: f32 },
}

/// One step of a [`reach`] path.
#[derive(Debug, Clone, PartialEq)]
pub struct Hop {
    /// The node reached at this step.
    pub node: String,
    /// How we reached it from the previous hop. `None` only for the first hop (the `from` node).
    pub via: Option<HopVia>,
}

/// Result of a [`reach`] traversal. `paths` holds one hop-chain per discovered/verified answer
/// (each starts at `from` with `via=None`); `targets` lists the answer node ids (relation-valid
/// discovery targets, or the single verified `to`).
#[derive(Debug, Clone, PartialEq)]
pub struct ReachResult {
    pub paths: Vec<Vec<Hop>>,
    pub targets: Vec<String>,
}

/// Split a relation/edge-type name into lowercased alphanumeric tokens (on `_`, spaces, and other
/// non-alphanumeric separators): `LOCATED_IN` → `["located", "in"]`.
fn tokens(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// Fuzzy relation-name match against an edge type: case-insensitive, whole-name equality OR a
/// whole-token match (`located` ↔ `LOCATED_IN`). Deliberately NOT raw substring — that let
/// `REL` match `UNRELATED`/`CORRELATES`, which both over-walked edges and let a spurious bridge
/// pass the relation-coherence gate. An empty relation matches everything ("no filter").
fn relation_matches(edge_type: &str, relation: &str) -> bool {
    let a = edge_type.trim().to_lowercase();
    let b = relation.trim().to_lowercase();
    if b.is_empty() || a == b {
        return true;
    }
    let (ta, tb) = (tokens(edge_type), tokens(relation));
    // A whole token of one name equals the whole other name, or the two share a whole token.
    ta.iter().any(|t| t == &b) || tb.iter().any(|t| t == &a) || ta.iter().any(|t| tb.contains(t))
}

/// In-graph reasoning hops out of `node`. Grounding edges (e.g. `MENTIONS`) are never walked —
/// they'd flood — so only `role==Chaining` edges advance the chain (see [`Ontology::relation_role`],
/// which maps the built-in CORE_EDGES to Grounding). When `relation` is `None` we walk UNDIRECTED
/// (outgoing forward + incoming backward); when a relation is named we walk it FORWARD only and
/// require a token/whole-name match (spec §5, forward-first). Returns `(next_node, edge_type,
/// forward)`.
fn chaining_steps(
    g: &GraphStore,
    ont: &Ontology,
    node: &str,
    relation: Option<&str>,
) -> anyhow::Result<Vec<(String, String, bool)>> {
    let mut steps: Vec<(String, String, bool)> = Vec::new();
    for e in g.outgoing(node)? {
        if ont.relation_role(&e.edge_type) == RelationRole::Grounding {
            continue;
        }
        if let Some(rel) = relation {
            if !relation_matches(&e.edge_type, rel) {
                continue;
            }
        }
        steps.push((e.to, e.edge_type, true));
    }
    if relation.is_none() {
        for e in g.incoming(node)? {
            if ont.relation_role(&e.edge_type) == RelationRole::Grounding {
                continue;
            }
            steps.push((e.from, e.edge_type, false));
        }
    }
    Ok(steps)
}

/// Bridge confidence (§7.5): stronger for an exact normalized-label match, weaker for a fuzzy
/// (BM25) one, and penalized by ambiguity — more same-mention nodes across documents means more
/// namesake risk, so a lower score. v1 (df-based refinement deferred).
fn bridge_confidence(exact: bool, n_candidates: usize) -> f32 {
    let base = if exact { 0.9 } else { 0.5 };
    (base / n_candidates.max(1) as f32).clamp(0.05, 1.0)
}

/// Community-agreement factors folded into bridge confidence (homonym false-positive guard): the
/// graph's generalize pass groups reasoning nodes into topical communities, so a bridge candidate
/// in the SAME community as the dead-end mention is more likely the same sense/entity, and one in
/// a DIFFERENT community is more likely a namesake (wrong-sense homonym). Either side missing
/// community data (no generalize run yet) is neutral — never penalizes a pilot-style graph.
const BRIDGE_COMMUNITY_AGREE_BOOST: f32 = 1.25;
const BRIDGE_COMMUNITY_DISAGREE_PENALTY: f32 = 0.5;

/// Above this many same-mention candidates the base bridge signal is already ambiguous; combined
/// with a non-exact match (see [`bridge_gate_weak`]) that ambiguity, plus a community mismatch,
/// gates the candidate out. Kept as a separate named threshold from `bridge_confidence`'s own
/// ambiguity denominator so the gate's "many" can be tuned independently.
const BRIDGE_GATE_MANY_CANDIDATES: usize = 2;

/// `true` when a bridge candidate's OWN match signal is already weak: not an exact normalized-label
/// match, or drawn from a crowded (ambiguous) same-mention candidate set. Exact-match candidates
/// with few candidates are a STRONG bridge and are never considered weak here, regardless of
/// community — the conservative gate (§ below) only fires on weak signals.
fn bridge_gate_weak(exact: bool, n_candidates: usize) -> bool {
    !exact || n_candidates > BRIDGE_GATE_MANY_CANDIDATES
}

/// `true` when both sides have recorded communities and they differ. `None` on either side is
/// never a disagreement — it's neutral (no generalize data to compare).
fn community_disagrees(src: Option<i64>, cand: Option<i64>) -> bool {
    matches!((src, cand), (Some(a), Some(b)) if a != b)
}

/// Multiplicative community-agreement factor: boost on agreement, penalty on disagreement, neutral
/// (1.0) when either side has no recorded community.
fn community_factor(src: Option<i64>, cand: Option<i64>) -> f32 {
    match (src, cand) {
        (Some(a), Some(b)) if a == b => BRIDGE_COMMUNITY_AGREE_BOOST,
        (Some(a), Some(b)) if a != b => BRIDGE_COMMUNITY_DISAGREE_PENALTY,
        _ => 1.0,
    }
}

/// Forward-first, role-aware, corpus-bridging traversal (spec §5 + §7.5).
///
/// From `from`, walk `role==Chaining` edges matching `relation` (fuzzy; `None` ⇒ all chaining
/// relations, undirected — legacy connectivity). At an in-graph dead end, attempt a **salience-gated
/// cross-document bridge**: resolve the dead-end node's own label to same-mention nodes in OTHER
/// documents and continue there, but only if the bridge term is discriminating (`term_is_salient`)
/// and the landing document's chain continues along the requested `relation` (relation-coherence
/// prune). Bounded by `max_depth` (edge count) and `bridge_budget` (max bridges per path). Cycle
/// guard: a global visited-node set plus a per-term bridge-once guard so a mention can't re-fan.
///
/// - `to = None`  → **discovery:** collect nodes reached by forward-following `relation` as answers.
/// - `to = Some`  → **verify:** early-exit with the first path that reaches `to`.
/// Global work budget for a single `reach`. Even on a pathologically dense graph (an
/// over-connected LEADS_TO where nearly every node shares a bridgeable term), one reach must do
/// bounded work: an uncapped traversal did O(nodes × shared-candidates) index queries and appeared
/// to hang (minutes of low-CPU SQLite churn). These caps sit far above any thin reasoning-skeleton
/// traversal — a real multi-hop answer is a few nodes and a bridge or two away — so normal results
/// are unchanged; they only bound the fan-out on a dense graph, where an unbounded reach was
/// useless anyway.
pub(crate) const REACH_MAX_VISITED: usize = 512;
const REACH_MAX_BRIDGE_RESOLVES: usize = 48;

pub fn reach(
    g: &GraphStore,
    ont: &Ontology,
    from: &str,
    relation: Option<&str>,
    to: Option<&str>,
    max_depth: usize,
    bridge_budget: usize,
) -> anyhow::Result<ReachResult> {
    // Trivial: from IS the target.
    if to == Some(from) {
        let p = vec![Hop {
            node: from.to_string(),
            via: None,
        }];
        return Ok(ReachResult {
            paths: vec![p],
            targets: vec![from.to_string()],
        });
    }

    struct QItem {
        path: Vec<Hop>,
        bridges: usize,
    }

    let mut visited: HashSet<String> = HashSet::from([from.to_string()]);
    let mut bridged_terms: HashSet<String> = HashSet::new();
    let mut q: VecDeque<QItem> = VecDeque::from([QItem {
        path: vec![Hop {
            node: from.to_string(),
            via: None,
        }],
        bridges: 0,
    }]);
    let mut paths: Vec<Vec<Hop>> = Vec::new();
    let mut targets: Vec<String> = Vec::new();
    let mut targets_seen: HashSet<String> = HashSet::new();
    let mut bridge_resolves: usize = 0;

    while let Some(item) = q.pop_front() {
        // Global work budget: stop admitting more of a pathologically dense graph. Whatever has
        // been found so far is returned; for a verify (`to = Some`) this conservatively reports
        // "no path within budget" rather than churning the whole component.
        if visited.len() >= REACH_MAX_VISITED {
            break;
        }
        // path.len()-1 == edges walked so far.
        if item.path.len() > max_depth {
            continue;
        }
        let last = item.path.last().unwrap().node.clone();

        let steps = chaining_steps(g, ont, &last, relation)?;
        let dead_end = steps.is_empty();

        for (next, et, fwd) in steps {
            // Verify: return the first path that reaches `to` (checked before visited, like legacy).
            if to == Some(next.as_str()) {
                let mut p = item.path.clone();
                p.push(Hop {
                    node: next.clone(),
                    via: Some(HopVia::Edge {
                        edge_type: et,
                        forward: fwd,
                    }),
                });
                return Ok(ReachResult {
                    paths: vec![p],
                    targets: vec![next],
                });
            }
            if visited.insert(next.clone()) {
                let mut p = item.path.clone();
                p.push(Hop {
                    node: next.clone(),
                    via: Some(HopVia::Edge {
                        edge_type: et,
                        forward: fwd,
                    }),
                });
                // Discovery: a node reached by FORWARD-following the requested relation is a valid
                // target of that relation.
                if to.is_none() && fwd && relation.is_some() && targets_seen.insert(next.clone()) {
                    targets.push(next.clone());
                    paths.push(p.clone());
                }
                q.push_back(QItem {
                    path: p,
                    bridges: item.bridges,
                });
            }
        }

        // Cross-document bridge — only at an in-graph dead end, within the per-path budget, and
        // within the global bridge-resolve budget (each resolve is a corpus-wide index query; on a
        // dense graph an unbounded count of them is what made reach appear to hang).
        if dead_end && item.bridges < bridge_budget && bridge_resolves < REACH_MAX_BRIDGE_RESOLVES {
            if let Some(node) = g.get_node(&last)? {
                let term = node.label.clone();
                let norm_term = normalize_label(&term);
                let src_doc = node.prov.source_path.clone();
                let src_community = g.node_community(&last)?;
                // Per-term bridge-once (cycle guard) + salience/IDF gate (§7.5 MUST). Marking the
                // term consumed even when it fails the gate is fine — it won't be retried.
                if !norm_term.is_empty()
                    && bridged_terms.insert(norm_term.clone())
                    && g.term_is_salient(&term)?
                {
                    // Count the expensive resolve against the global budget (the salience gate has
                    // passed, so the corpus-wide resolve + per-candidate fan-out is about to run).
                    bridge_resolves += 1;
                    // Resolve the mention → same-mention nodes in OTHER documents only.
                    let mut others: Vec<(String, bool, Option<i64>)> = Vec::new();
                    for cand in g.resolve(&term)? {
                        if cand == last {
                            continue;
                        }
                        if let Some(cn) = g.get_node(&cand)? {
                            if cn.prov.source_path == src_doc {
                                continue;
                            }
                            let exact = normalize_label(&cn.label) == norm_term;
                            let cand_community = g.node_community(&cand)?;
                            others.push((cand, exact, cand_community));
                        }
                    }
                    let n_other = others.len();
                    for (cand, exact, cand_community) in others {
                        // Conservative community gate (homonym FP guard): skip ONLY when the
                        // candidate's own match signal is already weak AND its community
                        // disagrees with the source's. A strong (exact-match, few-candidates)
                        // bridge is never gated on community mismatch alone — legit cross-document
                        // bridges of the same entity can legitimately land in a different community.
                        if bridge_gate_weak(exact, n_other)
                            && community_disagrees(src_community, cand_community)
                        {
                            continue;
                        }
                        let conf = (bridge_confidence(exact, n_other)
                            * community_factor(src_community, cand_community))
                        .clamp(0.05, 1.0);
                        // Verify: a candidate that IS `to` is a hit even if it's a leaf with no
                        // onward chaining edge — check `to` BEFORE the coherence prune, else a
                        // legitimate bridge-landing target is wrongly pruned (false negative).
                        if to == Some(cand.as_str()) {
                            let mut p = item.path.clone();
                            p.push(Hop {
                                node: cand.clone(),
                                via: Some(HopVia::Bridge {
                                    term: term.clone(),
                                    confidence: conf,
                                }),
                            });
                            return Ok(ReachResult {
                                paths: vec![p],
                                targets: vec![cand],
                            });
                        }
                        // Relation-coherence prune (§7.5 MUST): keep the bridged branch alive only
                        // if the landing document's chain continues along the requested relation.
                        if chaining_steps(g, ont, &cand, relation)?.is_empty() {
                            continue;
                        }
                        if visited.insert(cand.clone()) {
                            let mut p = item.path.clone();
                            p.push(Hop {
                                node: cand.clone(),
                                via: Some(HopVia::Bridge {
                                    term: term.clone(),
                                    confidence: conf,
                                }),
                            });
                            q.push_back(QItem {
                                path: p,
                                bridges: item.bridges + 1,
                            });
                        }
                    }
                }
            }
        }
    }

    Ok(ReachResult { paths, targets })
}

pub fn neighbors(
    g: &GraphStore,
    from: &str,
    edge_types: Option<&[String]>,
    depth: usize,
) -> anyhow::Result<Vec<String>> {
    let mut visited: HashSet<String> = HashSet::from([from.to_string()]);
    let mut frontier: VecDeque<(String, usize)> = VecDeque::from([(from.to_string(), 0)]);
    let mut out = Vec::new();
    while let Some((node, d)) = frontier.pop_front() {
        if d >= depth {
            continue;
        }
        for e in g.outgoing(&node)? {
            if !type_match(&e, edge_types) {
                continue;
            }
            if visited.insert(e.to.clone()) {
                out.push(e.to.clone());
                frontier.push_back((e.to, d + 1));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::store::{Edge, GraphStore, Node, Provenance};

    fn prov() -> Provenance {
        Provenance {
            source_path: "s".into(),
            range: None,
            file_sig: None,
            origin: "agent".into(),
            confidence: 1.0,
            created_at: 0,
        }
    }
    fn node(g: &GraphStore, id: &str) {
        g.put_node(&Node {
            id: id.into(),
            node_type: "Entity".into(),
            label: id.into(),
            aliases: vec![],
            prov: prov(),
        })
        .unwrap();
    }
    fn edge(g: &GraphStore, from: &str, to: &str, ty: &str) {
        g.put_edge(&Edge {
            from: from.into(),
            to: to.into(),
            edge_type: ty.into(),
            prov: prov(),
        })
        .unwrap();
    }

    #[test]
    fn neighbors_respects_depth_and_type() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        for id in ["a", "b", "c", "d"] {
            node(&g, id);
        }
        edge(&g, "a", "b", "REL");
        edge(&g, "b", "c", "REL");
        edge(&g, "a", "d", "OTHER");

        let d1 = neighbors(&g, "a", None, 1).unwrap();
        assert!(
            d1.contains(&"b".to_string())
                && d1.contains(&"d".to_string())
                && !d1.contains(&"c".to_string())
        );

        let d2 = neighbors(&g, "a", None, 2).unwrap();
        assert!(d2.contains(&"c".to_string()));

        let only_rel = neighbors(&g, "a", Some(&["REL".to_string()]), 1).unwrap();
        assert_eq!(only_rel, vec!["b".to_string()]);
    }

    #[test]
    fn reach_is_undirected_and_records_direction_when_relation_omitted() {
        // `reach(relation=None, to=Some, bridge_budget=0)` is the old `path_between` behavior: an
        // undirected BFS over chaining edges that records each hop's real direction.
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        for id in ["a", "b", "c"] {
            node(&g, id);
        }
        // Only forward edges a->b->c exist.
        edge(&g, "a", "b", "REL");
        edge(&g, "b", "c", "REL");
        let ont = Ontology::default();

        // Forward reachable, hops carry forward=true.
        let fwd = reach(&g, &ont, "a", None, Some("c"), 5, 0).unwrap().paths[0].clone();
        let nodes: Vec<&str> = fwd.iter().map(|h| h.node.as_str()).collect();
        assert_eq!(nodes, vec!["a", "b", "c"]);
        assert_eq!(fwd[0].via, None);
        assert_eq!(
            fwd[1].via,
            Some(HopVia::Edge {
                edge_type: "REL".into(),
                forward: true
            })
        );
        assert_eq!(
            fwd[2].via,
            Some(HopVia::Edge {
                edge_type: "REL".into(),
                forward: true
            })
        );

        // Reverse direction is found too (undirected), with forward=false.
        let rev = reach(&g, &ont, "c", None, Some("a"), 5, 0).unwrap().paths[0].clone();
        let rnodes: Vec<&str> = rev.iter().map(|h| h.node.as_str()).collect();
        assert_eq!(rnodes, vec!["c", "b", "a"]);
        assert_eq!(
            rev[1].via,
            Some(HopVia::Edge {
                edge_type: "REL".into(),
                forward: false
            })
        );

        // Unreachable within depth.
        node(&g, "z");
        assert!(reach(&g, &ont, "a", None, Some("z"), 5, 0)
            .unwrap()
            .paths
            .is_empty());
        // Same node.
        let same = reach(&g, &ont, "a", None, Some("a"), 5, 0).unwrap().paths[0].clone();
        assert_eq!(same.len(), 1);
        assert_eq!(same[0].node, "a");
    }

    // ---- reach() tests ------------------------------------------------------------------------

    use crate::graph::ontology::Ontology;

    #[test]
    fn reach_discovery_is_bounded_by_global_visited_budget() {
        // A relation-chain longer than the global budget. Uncapped, discovery returns every node
        // on the chain; the budget must bound total work (and thus the target count) so a
        // pathologically dense/long graph can't make one reach churn the whole component — the
        // real-world stall was an over-connected LEADS_TO where reach did O(component) index work.
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let n = REACH_MAX_VISITED + 200;
        for i in 0..n {
            node(&g, &format!("n{i}"));
        }
        for i in 0..n - 1 {
            edge(&g, &format!("n{i}"), &format!("n{}", i + 1), "REL");
        }
        let ont = Ontology::default();
        // max_depth far exceeds the chain length, so the VISITED budget — not depth — is the limit.
        let res = reach(&g, &ont, "n0", Some("REL"), None, n + 10, 0).unwrap();
        assert!(
            res.targets.len() <= REACH_MAX_VISITED,
            "discovery targets {} must be bounded by the visited budget {REACH_MAX_VISITED}",
            res.targets.len()
        );
        assert!(
            res.targets.len() < n - 1,
            "the budget must actually bite on an over-long chain (got all {} targets)",
            res.targets.len()
        );
    }

    /// A node with an explicit label and source document (the doc identity the bridge crosses).
    fn dnode(g: &GraphStore, id: &str, label: &str, doc: &str) {
        g.put_node(&Node {
            id: id.into(),
            node_type: "Entity".into(),
            label: label.into(),
            aliases: vec![],
            prov: Provenance {
                source_path: doc.into(),
                ..prov()
            },
        })
        .unwrap();
    }

    /// Add `n` distinct filler nodes so the corpus is large enough that a df=2 mention clears the
    /// salience ratio gate (2/total < 0.1 needs total > 20).
    fn fillers(g: &GraphStore, n: usize) {
        for i in 0..n {
            dnode(
                g,
                &format!("filler{i}"),
                &format!("Fillerword{i}"),
                "filler.md",
            );
        }
    }

    #[test]
    fn reach_legacy_parity_no_bridge() {
        // relation=None, to=Some, bridge_budget=0 reproduces the pre-bridge `path` tool exactly —
        // the old undirected-connectivity behavior is a strict subset of `reach` (spec §6).
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        for id in ["a", "b", "c"] {
            node(&g, id);
        }
        edge(&g, "a", "b", "REL");
        edge(&g, "b", "c", "REL");
        let ont = Ontology::default();

        let res = reach(&g, &ont, "a", None, Some("c"), 5, 0).unwrap();
        assert_eq!(res.targets, vec!["c".to_string()]);
        let hops = &res.paths[0];
        let nodes: Vec<&str> = hops.iter().map(|h| h.node.as_str()).collect();
        assert_eq!(nodes, vec!["a", "b", "c"]);
        assert_eq!(hops[0].via, None);
        assert_eq!(
            hops[1].via,
            Some(HopVia::Edge {
                edge_type: "REL".into(),
                forward: true
            })
        );
        assert_eq!(
            hops[2].via,
            Some(HopVia::Edge {
                edge_type: "REL".into(),
                forward: true
            })
        );
    }

    #[test]
    fn reach_grounding_edges_are_not_walked() {
        // A MENTIONS (grounding) edge must not advance the chain — reach must not reach `ent`.
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        dnode(&g, "fact", "Fact one", "d.md");
        dnode(&g, "ent", "Some entity", "d.md");
        edge(&g, "fact", "ent", "MENTIONS");
        let ont = Ontology::default();
        // Verify: `ent` is unreachable over a grounding-only edge.
        let res = reach(&g, &ont, "fact", None, Some("ent"), 5, 0).unwrap();
        assert!(res.targets.is_empty());
        assert!(res.paths.is_empty());
    }

    #[test]
    fn reach_bridges_across_docs_on_salient_term() {
        // docA fact mentions a salient term "Xanthium" and dead-ends in-doc (only a grounding edge);
        // docB has a same-mention node with a REL chaining edge to the Target.
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        dnode(&g, "A_fact", "Xanthium", "docA.md");
        dnode(&g, "A_ent", "Local note", "docA.md");
        edge(&g, "A_fact", "A_ent", "MENTIONS"); // grounding, not a chain hop
        dnode(&g, "B_T", "Xanthium", "docB.md");
        dnode(&g, "Target", "TheAnswer", "docB.md");
        edge(&g, "B_T", "Target", "REL");
        fillers(&g, 25);
        let ont = Ontology::default();

        let res = reach(&g, &ont, "A_fact", Some("REL"), None, 6, 1).unwrap();
        assert!(res.targets.contains(&"Target".to_string()));
        let p = res
            .paths
            .iter()
            .find(|p| p.last().unwrap().node == "Target")
            .unwrap();
        assert!(p.iter().any(|h| matches!(
            &h.via,
            Some(HopVia::Bridge { term, confidence }) if term.eq_ignore_ascii_case("Xanthium") && *confidence > 0.0
        )));

        // No bridge budget ⇒ the chain stays inside docA and finds nothing.
        let none = reach(&g, &ont, "A_fact", Some("REL"), None, 6, 0).unwrap();
        assert!(none.targets.is_empty());
    }

    #[test]
    fn reach_salience_gate_blocks_common_word_bridge() {
        // The pivot mention is a common, high-df word → salience gate refuses the bridge.
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        dnode(&g, "A_fact", "Common", "docA.md");
        dnode(&g, "B_T", "Common", "docB.md");
        dnode(&g, "Target", "TheAnswer", "docB.md");
        edge(&g, "B_T", "Target", "REL");
        // Many other nodes also carry the token → df/total well over the ratio gate.
        for i in 0..10 {
            dnode(&g, &format!("common{i}"), "Common", "filler.md");
        }
        fillers(&g, 4);
        let ont = Ontology::default();

        let res = reach(&g, &ont, "A_fact", Some("REL"), None, 6, 1).unwrap();
        assert!(!res.targets.contains(&"Target".to_string()));
    }

    #[test]
    fn reach_relation_coherence_prunes_mismatched_bridge() {
        // The bridge lands in docB, but docB's chain continues along a DIFFERENT relation → pruned.
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        dnode(&g, "A_fact", "Xanthium", "docA.md");
        dnode(&g, "B_T", "Xanthium", "docB.md");
        dnode(&g, "Target", "TheAnswer", "docB.md");
        edge(&g, "B_T", "Target", "OTHER"); // not the requested REL
        fillers(&g, 25);
        let ont = Ontology::default();

        let res = reach(&g, &ont, "A_fact", Some("REL"), None, 6, 1).unwrap();
        assert!(!res.targets.contains(&"Target".to_string()));
    }

    #[test]
    fn reach_budget_blocks_two_bridge_drift() {
        // A(docA) ~Alpha~> B(docB) -REL-> Bmid(docB) ~Gamma~> C(docC) -REL-> Target(docC).
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        dnode(&g, "A", "Alpha", "docA.md");
        dnode(&g, "B", "Alpha", "docB.md");
        dnode(&g, "Bmid", "Gamma", "docB.md");
        dnode(&g, "C", "Gamma", "docC.md");
        dnode(&g, "Target", "TheAnswer", "docC.md");
        edge(&g, "B", "Bmid", "REL");
        edge(&g, "C", "Target", "REL");
        fillers(&g, 25);
        let ont = Ontology::default();

        // B=1: the second bridge (Gamma) is over budget → Target unreachable.
        let one = reach(&g, &ont, "A", Some("REL"), None, 8, 1).unwrap();
        assert!(!one.targets.contains(&"Target".to_string()));

        // B=2: the path exists and is found — proving only the budget blocked it above.
        let two = reach(&g, &ont, "A", Some("REL"), None, 8, 2).unwrap();
        assert!(two.targets.contains(&"Target".to_string()));
    }

    #[test]
    fn reach_does_not_walk_core_structural_edges() {
        // CONTAINS (doc→section) and NEXT (chunk sequence) are CORE_EDGES → Grounding role.
        // `reach` is role-filtered, so a chain that only connects through structural/grounding
        // edges (no chaining edge at all) must NOT be found — never flood the reasoning walk with
        // the structural backbone.
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        for id in ["a", "b", "c", "d"] {
            node(&g, id);
        }
        edge(&g, "a", "b", "CONTAINS");
        edge(&g, "b", "c", "NEXT");
        edge(&g, "c", "d", "MENTIONS");
        let ont = Ontology::default();
        let reasoning = reach(&g, &ont, "a", None, Some("d"), 5, 0).unwrap();
        assert!(reasoning.targets.is_empty());
    }

    #[test]
    fn relation_matches_is_token_exact_not_substring() {
        // Whole-name equality.
        assert!(relation_matches("LOCATED_IN", "LOCATED_IN"));
        // Whole-token match (case-insensitive).
        assert!(relation_matches("LOCATED_IN", "located"));
        assert!(relation_matches("REPORTS_TO", "reports"));
        // Substring must NOT match (the old bug): REL ⊄ UNRELATED/CORRELATES; ORG ⊄ CATEGORY.
        assert!(!relation_matches("UNRELATED", "REL"));
        assert!(!relation_matches("CORRELATES", "REL"));
        assert!(!relation_matches("CATEGORY", "ORG"));
        // Empty relation = no filter.
        assert!(relation_matches("ANYTHING", ""));
    }

    // ---- community-agreement bridge confidence (homonym FP guard) -----------------------------

    use crate::graph::store::NodeMeta;

    /// Set `node_meta.community` for a set of nodes in one `replace_node_meta` call (it replaces
    /// ALL rows, so every node whose community matters for the test must be listed together).
    fn set_communities(g: &GraphStore, rows: &[(&str, i64)]) {
        let rows: Vec<(String, NodeMeta)> = rows
            .iter()
            .map(|(id, c)| {
                (
                    (*id).to_string(),
                    NodeMeta {
                        community: Some(*c),
                        ..Default::default()
                    },
                )
            })
            .collect();
        g.replace_node_meta(&rows).unwrap();
    }

    #[test]
    fn reach_bridge_confidence_boosted_for_same_community_penalized_for_different() {
        // One dead-end term, two exact-match candidates in other docs — same match quality
        // (exact=true, n_candidates=2) for both, so any confidence gap comes purely from the
        // community factor. C1 shares the source's community; C2 doesn't.
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        dnode(&g, "Source", "Thornbill", "docS.md");
        dnode(&g, "C1", "Thornbill", "docC1.md");
        dnode(&g, "Target1", "AnswerOne", "docC1.md");
        edge(&g, "C1", "Target1", "REL");
        dnode(&g, "C2", "Thornbill", "docC2.md");
        dnode(&g, "Target2", "AnswerTwo", "docC2.md");
        edge(&g, "C2", "Target2", "REL");
        // 3 occurrences of "Thornbill" (Source + 2 candidates) need a bigger corpus than the
        // usual 2-occurrence fixture to clear the salience df/total ratio gate.
        fillers(&g, 40);
        set_communities(&g, &[("Source", 100), ("C1", 100), ("C2", 200)]);
        let ont = Ontology::default();

        let res = reach(&g, &ont, "Source", Some("REL"), None, 6, 2).unwrap();
        assert!(res.targets.contains(&"Target1".to_string()));
        assert!(res.targets.contains(&"Target2".to_string()));

        let conf_of = |target: &str| -> f32 {
            let p = res
                .paths
                .iter()
                .find(|p| p.last().unwrap().node == target)
                .unwrap();
            match &p.iter().rev().nth(1).unwrap().via {
                Some(HopVia::Bridge { confidence, .. }) => *confidence,
                other => panic!("expected a Bridge hop before {target}, got {other:?}"),
            }
        };
        let same = conf_of("Target1");
        let diff = conf_of("Target2");
        assert!(
            same > diff,
            "same-community bridge ({same}) must score higher than different-community ({diff})"
        );
    }

    #[test]
    fn reach_bridge_gate_skips_weak_mismatched_community_but_keeps_exact() {
        // Weak (fuzzy, non-exact) bridge + community disagreement → gated out entirely. A strong
        // (exact-match) bridge with the SAME community disagreement is conservatively kept — a
        // legit cross-document bridge of the same entity can land in a different community.
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();

        // Weak: fuzzy match only ("Zorblatt" vs "Zorblatt Prime" — no exact normalized-label hit).
        dnode(&g, "A1", "Zorblatt", "docA1.md");
        dnode(&g, "A2", "Zorblatt Prime", "docA2.md");
        dnode(&g, "TargetA", "AnswerA", "docA2.md");
        edge(&g, "A2", "TargetA", "REL");

        // Strong: exact same-label match in another doc.
        dnode(&g, "B1", "Milkwort", "docB1.md");
        dnode(&g, "B2", "Milkwort", "docB2.md");
        dnode(&g, "TargetB", "AnswerB", "docB2.md");
        edge(&g, "B2", "TargetB", "REL");

        fillers(&g, 25);
        set_communities(&g, &[("A1", 1), ("A2", 2), ("B1", 3), ("B2", 4)]);
        let ont = Ontology::default();

        let weak = reach(&g, &ont, "A1", Some("REL"), None, 6, 1).unwrap();
        assert!(
            !weak.targets.contains(&"TargetA".to_string()),
            "weak fuzzy bridge with community mismatch must be gated"
        );

        let strong = reach(&g, &ont, "B1", Some("REL"), None, 6, 1).unwrap();
        assert!(
            strong.targets.contains(&"TargetB".to_string()),
            "exact-match bridge must never be gated on community mismatch alone"
        );
    }

    #[test]
    fn reach_bridge_confidence_neutral_when_no_community_recorded() {
        // No node_meta/community written anywhere (the pilot-style, pre-generalize graph): the
        // community factor must be exactly neutral, reproducing the pre-community confidence value.
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        dnode(&g, "A_fact", "Xanthium", "docA.md");
        dnode(&g, "B_T", "Xanthium", "docB.md");
        dnode(&g, "Target", "TheAnswer", "docB.md");
        edge(&g, "B_T", "Target", "REL");
        fillers(&g, 25);
        let ont = Ontology::default();

        let res = reach(&g, &ont, "A_fact", Some("REL"), None, 6, 1).unwrap();
        assert!(res.targets.contains(&"Target".to_string()));
        let p = res
            .paths
            .iter()
            .find(|p| p.last().unwrap().node == "Target")
            .unwrap();
        let via = &p.iter().rev().nth(1).unwrap().via;
        let conf = match via {
            Some(HopVia::Bridge { confidence, .. }) => *confidence,
            other => panic!("expected a Bridge hop before Target, got {other:?}"),
        };
        assert_eq!(
            conf,
            bridge_confidence(true, 1),
            "no community data must be a no-op on confidence"
        );
    }

    #[test]
    fn reach_verify_accepts_leaf_bridge_landing() {
        // Verify mode: `to` IS the bridged node, a leaf with no onward chaining edge. It must be
        // returned (the coherence prune must not fire before the `to` check).
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        dnode(&g, "A_fact", "Xanthium", "docA.md");
        dnode(&g, "B_leaf", "Xanthium", "docB.md"); // no onward edge
        fillers(&g, 25);
        let ont = Ontology::default();
        let res = reach(&g, &ont, "A_fact", Some("REL"), Some("B_leaf"), 6, 1).unwrap();
        assert_eq!(res.targets, vec!["B_leaf".to_string()]);
        let hops = &res.paths[0];
        assert_eq!(hops.last().unwrap().node, "B_leaf");
        assert!(matches!(
            hops.last().unwrap().via,
            Some(HopVia::Bridge { .. })
        ));
    }
}
