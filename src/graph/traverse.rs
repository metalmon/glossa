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

/// Legacy hop shape kept for the `path_between` shim's public contract — `tools.rs`/mcp still
/// consume `via: Option<(edge_type, forward)>`. Superseded by [`Hop`]+[`HopVia`]; a later task
/// migrates those callers to `reach` and removes this shim. See [[graph-cross-doc-bridge]].
#[derive(Debug, Clone, PartialEq)]
pub struct LegacyHop {
    pub node: String,
    pub via: Option<(String, bool)>,
}

/// Fuzzy relation-name match against an edge type: case/whitespace-insensitive, substring either
/// way (mirrors how a small model may pass a slightly different spelling of a relation). An empty
/// relation matches everything (treated as "no filter").
fn relation_matches(edge_type: &str, relation: &str) -> bool {
    let a = edge_type.trim().to_lowercase();
    let b = relation.trim().to_lowercase();
    b.is_empty() || a == b || a.contains(&b) || b.contains(&a)
}

/// In-graph reasoning hops out of `node`. Grounding edges (e.g. `MENTIONS`) are never walked —
/// they'd flood — so only `role==Chaining` edges advance the chain (see [`Ontology::relation_role`],
/// which maps the built-in CORE_EDGES to Grounding). When `relation` is `None` we walk UNDIRECTED
/// (outgoing forward + incoming backward) to preserve legacy `path_between` connectivity; when a
/// relation is named we walk it FORWARD only and require a fuzzy edge-type match (spec §5,
/// forward-first). Returns `(next_node, edge_type, forward)`.
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
        let p = vec![Hop { node: from.to_string(), via: None }];
        return Ok(ReachResult { paths: vec![p], targets: vec![from.to_string()] });
    }

    struct QItem {
        path: Vec<Hop>,
        bridges: usize,
    }

    let mut visited: HashSet<String> = HashSet::from([from.to_string()]);
    let mut bridged_terms: HashSet<String> = HashSet::new();
    let mut q: VecDeque<QItem> = VecDeque::from([QItem {
        path: vec![Hop { node: from.to_string(), via: None }],
        bridges: 0,
    }]);
    let mut paths: Vec<Vec<Hop>> = Vec::new();
    let mut targets: Vec<String> = Vec::new();
    let mut targets_seen: HashSet<String> = HashSet::new();

    while let Some(item) = q.pop_front() {
        // path.len()-1 == edges walked so far; mirror legacy `path_between`'s depth guard exactly.
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
                p.push(Hop { node: next.clone(), via: Some(HopVia::Edge { edge_type: et, forward: fwd }) });
                return Ok(ReachResult { paths: vec![p], targets: vec![next] });
            }
            if visited.insert(next.clone()) {
                let mut p = item.path.clone();
                p.push(Hop {
                    node: next.clone(),
                    via: Some(HopVia::Edge { edge_type: et, forward: fwd }),
                });
                // Discovery: a node reached by FORWARD-following the requested relation is a valid
                // target of that relation.
                if to.is_none()
                    && fwd
                    && relation.is_some()
                    && targets_seen.insert(next.clone())
                {
                    targets.push(next.clone());
                    paths.push(p.clone());
                }
                q.push_back(QItem { path: p, bridges: item.bridges });
            }
        }

        // Cross-document bridge — only at an in-graph dead end and within the per-path budget.
        if dead_end && item.bridges < bridge_budget {
            if let Some(node) = g.get_node(&last)? {
                let term = node.label.clone();
                let norm_term = normalize_label(&term);
                let src_doc = node.prov.source_path.clone();
                // Per-term bridge-once (cycle guard) + salience/IDF gate (§7.5 MUST). Marking the
                // term consumed even when it fails the gate is fine — it won't be retried.
                if !norm_term.is_empty()
                    && bridged_terms.insert(norm_term.clone())
                    && g.term_is_salient(&term)?
                {
                    // Resolve the mention → same-mention nodes in OTHER documents only.
                    let mut others: Vec<(String, bool)> = Vec::new();
                    for cand in g.resolve(&term)? {
                        if cand == last {
                            continue;
                        }
                        if let Some(cn) = g.get_node(&cand)? {
                            if cn.prov.source_path == src_doc {
                                continue;
                            }
                            let exact = normalize_label(&cn.label) == norm_term;
                            others.push((cand, exact));
                        }
                    }
                    let n_other = others.len();
                    for (cand, exact) in others {
                        // Relation-coherence prune (§7.5 MUST): keep the bridged branch alive only
                        // if the landing document's chain continues along the requested relation.
                        if chaining_steps(g, ont, &cand, relation)?.is_empty() {
                            continue;
                        }
                        let conf = bridge_confidence(exact, n_other);
                        if to == Some(cand.as_str()) {
                            let mut p = item.path.clone();
                            p.push(Hop {
                                node: cand.clone(),
                                via: Some(HopVia::Bridge { term: term.clone(), confidence: conf }),
                            });
                            return Ok(ReachResult { paths: vec![p], targets: vec![cand] });
                        }
                        if visited.insert(cand.clone()) {
                            let mut p = item.path.clone();
                            p.push(Hop {
                                node: cand.clone(),
                                via: Some(HopVia::Bridge { term: term.clone(), confidence: conf }),
                            });
                            q.push_back(QItem { path: p, bridges: item.bridges + 1 });
                        }
                    }
                }
            }
        }
    }

    Ok(ReachResult { paths, targets })
}

/// Undirected shortest path from `from` to `to`, recording each edge's real direction. **Parity
/// shim** over [`reach`] (`relation=None`, `to=Some`, `bridge_budget=0`) — a strict subset that
/// reproduces the pre-bridge behavior exactly, so existing callers keep compiling. A later task
/// migrates them to `reach` and removes this. Returns the hop chain (including `from` with
/// `via=None`), or `None` if `to` is unreachable within `max_depth` edges.
pub fn path_between(
    g: &GraphStore,
    from: &str,
    to: &str,
    max_depth: usize,
) -> anyhow::Result<Option<Vec<LegacyHop>>> {
    let ont = Ontology::default();
    let res = reach(g, &ont, from, None, Some(to), max_depth, 0)?;
    Ok(res.paths.into_iter().next().map(|hops| {
        hops.into_iter()
            .map(|h| LegacyHop {
                node: h.node,
                via: match h.via {
                    None => None,
                    Some(HopVia::Edge { edge_type, forward }) => Some((edge_type, forward)),
                    // bridge_budget=0 ⇒ a bridge hop can never appear on a legacy path.
                    Some(HopVia::Bridge { term, .. }) => Some((term, true)),
                },
            })
            .collect()
    }))
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

pub fn path(
    g: &GraphStore,
    from: &str,
    to: &str,
    max_depth: usize,
) -> anyhow::Result<Option<Vec<String>>> {
    if from == to {
        return Ok(Some(vec![from.to_string()]));
    }
    let mut visited: HashSet<String> = HashSet::from([from.to_string()]);
    let mut q: VecDeque<Vec<String>> = VecDeque::from([vec![from.to_string()]]);
    while let Some(p) = q.pop_front() {
        if p.len() > max_depth {
            continue;
        }
        let last = p.last().unwrap().clone();
        for e in g.outgoing(&last)? {
            if e.to == to {
                let mut found = p.clone();
                found.push(e.to);
                return Ok(Some(found));
            }
            if visited.insert(e.to.clone()) {
                let mut np = p.clone();
                np.push(e.to);
                q.push_back(np);
            }
        }
    }
    Ok(None)
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
    fn path_finds_chain() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        for id in ["a", "b", "c"] {
            node(&g, id);
        }
        edge(&g, "a", "b", "REL");
        edge(&g, "b", "c", "REL");
        assert_eq!(
            path(&g, "a", "c", 5).unwrap(),
            Some(vec!["a".into(), "b".into(), "c".into()])
        );
        assert_eq!(path(&g, "a", "z", 5).unwrap(), None);
    }

    #[test]
    fn path_between_is_undirected_and_records_direction() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        for id in ["a", "b", "c"] {
            node(&g, id);
        }
        // Only forward edges a->b->c exist.
        edge(&g, "a", "b", "REL");
        edge(&g, "b", "c", "REL");

        // Forward reachable, hops carry forward=true.
        let fwd = path_between(&g, "a", "c", 5).unwrap().unwrap();
        let nodes: Vec<&str> = fwd.iter().map(|h| h.node.as_str()).collect();
        assert_eq!(nodes, vec!["a", "b", "c"]);
        assert_eq!(fwd[0].via, None);
        assert_eq!(fwd[1].via, Some(("REL".to_string(), true)));
        assert_eq!(fwd[2].via, Some(("REL".to_string(), true)));

        // Reverse direction is found too (undirected), with forward=false.
        let rev = path_between(&g, "c", "a", 5).unwrap().unwrap();
        let rnodes: Vec<&str> = rev.iter().map(|h| h.node.as_str()).collect();
        assert_eq!(rnodes, vec!["c", "b", "a"]);
        assert_eq!(rev[1].via, Some(("REL".to_string(), false)));

        // Unreachable within depth.
        node(&g, "z");
        assert_eq!(path_between(&g, "a", "z", 5).unwrap(), None);
        // Same node.
        let same = path_between(&g, "a", "a", 5).unwrap().unwrap();
        assert_eq!(same.len(), 1);
        assert_eq!(same[0].node, "a");
    }

    // ---- reach() tests ------------------------------------------------------------------------

    use crate::graph::ontology::Ontology;

    /// A node with an explicit label and source document (the doc identity the bridge crosses).
    fn dnode(g: &GraphStore, id: &str, label: &str, doc: &str) {
        g.put_node(&Node {
            id: id.into(),
            node_type: "Entity".into(),
            label: label.into(),
            aliases: vec![],
            prov: Provenance { source_path: doc.into(), ..prov() },
        })
        .unwrap();
    }

    /// Add `n` distinct filler nodes so the corpus is large enough that a df=2 mention clears the
    /// salience ratio gate (2/total < 0.1 needs total > 20).
    fn fillers(g: &GraphStore, n: usize) {
        for i in 0..n {
            dnode(g, &format!("filler{i}"), &format!("Fillerword{i}"), "filler.md");
        }
    }

    #[test]
    fn reach_legacy_parity_no_bridge() {
        // relation=None, to=Some, bridge_budget=0 reproduces the pre-bridge path_between exactly.
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
        assert_eq!(hops[1].via, Some(HopVia::Edge { edge_type: "REL".into(), forward: true }));
        assert_eq!(hops[2].via, Some(HopVia::Edge { edge_type: "REL".into(), forward: true }));

        // The shim maps that same result back to the old LegacyHop shape callers still consume.
        let legacy = path_between(&g, "a", "c", 5).unwrap().unwrap();
        let lnodes: Vec<&str> = legacy.iter().map(|h| h.node.as_str()).collect();
        assert_eq!(lnodes, vec!["a", "b", "c"]);
        assert_eq!(legacy[1].via, Some(("REL".to_string(), true)));
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
        let p = res.paths.iter().find(|p| p.last().unwrap().node == "Target").unwrap();
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
}
