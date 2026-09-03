//! Ontology-agnostic Personalized PageRank (random walk with restart) over the reasoning graph.
//!
//! The graph is treated as a generic weighted digraph: `node_type` and domain relations are NEVER
//! read. `edge_type` IS read, but only for a fixed system-level tier check (mechanical-similarity
//! edges like `SIMILAR`, derived by `generalize` from token/embedding overlap, are down-weighted
//! relative to authored/structural edges) — the ontology-blind contract holds at the node level and
//! for domain-specific relations, never at the SIMILAR-vs-everything-else tier.
//! Mass starts on the question's lexical seed nodes and flows across edges, ranking every node by
//! CONNECTIVITY to the question's neighborhood. This floats the terminal node of a multi-hop chain
//! even when it shares no tokens with the question — the thing lexical ranking cannot do — and it
//! disperses high-degree hubs for free (a PageRank property), so no df/hop caps are needed.

use crate::graph::store::GraphStore;
use std::collections::HashMap;
use std::path::Path;

/// Mechanical-similarity edge types: derived by `generalize` from token/embedding overlap, NOT
/// authored reasoning. A SYSTEM-level tier (like STRUCTURAL_NODES), not a domain-ontology choice.
/// NOTE: Keep in sync with the sibling definitions: `SOFT_EDGE_TYPES` in `src/graph/io.rs` (line 200)
/// and `SOFT_EDGES` in `src/graph/ontology.rs` (line 9). If a soft-edge type is added, update all three.
const SIMILARITY_EDGES: &[&str] = &["SIMILAR"];

/// `w_sim`: the transition weight of a mechanical-similarity edge relative to a reasoning edge (1.0).
///
/// Resolution precedence: the `GLOSSA_PPR_SIM_WEIGHT` env var (a sweep re-runs without recompiling
/// and without editing the corpus) > the per-corpus `[retrieval].sim_weight` in `ontology.toml` >
/// the engine default **0.1**. `gdir` is the corpus's `.glossa` directory; the ontology sits at
/// `gdir/ontology.toml`, so its root is `gdir.parent()`.
///
/// Why a knob and not a fixed default: the best value depends on the READER, not the graph. A weak
/// reader (e.g. a 4B) benefits from heavier SIMILAR mass (~0.3 won a kb-abac A/B on 4B); a strong
/// reader (e.g. a 35B) does better with the leaner 0.1. The graph can't see which reader consumes
/// it, so auto-deriving from graph structure would tune the wrong axis — hence a per-corpus config
/// value with an env override. 0.1 is the conservative general default (MuSiQue-validated). The
/// resolved weight is folded into the PPR transition cache signature (`cache_sig`), so changing the
/// env var OR the ontology value invalidates the matrix exactly like a graph edit does.
pub(crate) fn sim_weight(gdir: &Path) -> f32 {
    let valid = |w: f32| (w >= 0.0 && w.is_finite()).then_some(w);
    if let Some(w) = std::env::var("GLOSSA_PPR_SIM_WEIGHT")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .and_then(valid)
    {
        return w;
    }
    if let Some(w) = gdir
        .parent()
        .and_then(|root| crate::graph::ontology::Ontology::load_or_default(root).ppr_sim_weight())
        .and_then(valid)
    {
        return w;
    }
    0.1
}

/// `w_spine`: the transition weight of a reasoning-SPINE edge (a relation whose ontology
/// [`RelationRole`](crate::graph::ontology::RelationRole) is `Chaining` — the edges the traverse
/// layer walks) relative to a plain reasoning edge (1.0). `> 1.0` BOOSTS the spine so a node's one
/// load-bearing bridge edge isn't diluted by out-degree against its many Grounding/descriptive edges.
///
/// Resolution mirrors [`sim_weight`]: `GLOSSA_PPR_SPINE_WEIGHT` env > `[retrieval].spine_weight` in
/// `ontology.toml` > engine default **1.0 (a no-op)**. The 1.0 default keeps every existing graph
/// byte-identical until a corpus opts in — and makes an A/B a pure env flip. Like `w_sim`, the
/// resolved value is folded into the transition cache signature ([`cache_sig`]) so a change rebuilds
/// the matrix instead of silently no-op'ing. Reads the role from ontology DATA (never a hardcoded
/// relation name), so the engine stays ontology-blind.
pub(crate) fn spine_weight(gdir: &Path) -> f32 {
    let valid = |w: f32| (w >= 0.0 && w.is_finite()).then_some(w);
    if let Some(w) = std::env::var("GLOSSA_PPR_SPINE_WEIGHT")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .and_then(valid)
    {
        return w;
    }
    if let Some(w) = gdir
        .parent()
        .and_then(|root| crate::graph::ontology::Ontology::load_or_default(root).ppr_spine_weight())
        .and_then(valid)
    {
        return w;
    }
    1.0
}

/// Fold `w_sim` AND `w_spine` into the transition cache's content signature. The persisted transition
/// matrix bakes in both weights (edge weights = tier * confidence, tier scaled by whichever tier the
/// edge is in), so a cache built at one weight pair MUST NOT be reused at another — otherwise changing
/// `GLOSSA_PPR_SIM_WEIGHT`/`GLOSSA_PPR_SPINE_WEIGHT` silently has no effect while the graph is
/// unchanged. Mixing both weights' bits into the content signature makes a different pair a cache miss
/// (rebuild), exactly like a graph edit does.
pub(crate) fn cache_sig(content_sig: u64, w_sim: f32, w_spine: f32) -> u64 {
    content_sig
        ^ (w_sim.to_bits() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (w_spine.to_bits() as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
}

/// System-level tier weight for an edge in the PPR walk. `w_sim` is the weight for mechanical
/// similarity edges; `w_spine` the weight for reasoning-spine edges (`is_spine` — the caller resolves
/// it from the ontology's `RelationRole`, keeping this helper a pure lookup); plain reasoning edges
/// return 1.0. SIMILAR is checked FIRST, so a soft edge is never mistaken for spine even though a
/// missing role fails open to `Chaining`.
pub(crate) fn edge_tier_weight(edge_type: &str, w_sim: f32, w_spine: f32, is_spine: bool) -> f32 {
    if SIMILARITY_EDGES.contains(&edge_type) {
        w_sim
    } else if is_spine {
        w_spine
    } else {
        1.0
    }
}

/// On-disk shape version of `.glossa/ppr_transition.json`. Bump whenever the persisted `adj` shape
/// changes (e.g. unweighted `Vec<Vec<usize>>` -> weighted `Vec<Vec<(usize, f32)>>` at version 2) so a
/// stale cache from an older binary is invalidated & rebuilt instead of silently misparsed.
pub const TRANSITION_CACHE_VERSION: u32 = 2;

/// A symmetric transition structure built once from the graph's nodes + edges. Ontology-blind:
/// every stored edge becomes a bidirectional transition whose weight is its system-level tier (see
/// `edge_tier_weight`) MULTIPLIED by its stored `prov.confidence`. So mechanical-similarity edges
/// carry less mass than authored/structural ones (the tier), AND within a tier a low-confidence
/// edge carries less mass than a high-confidence one (the confidence factor). Legacy edges with
/// `confidence <= 0` default to 1.0, so pre-confidence graphs are unchanged.
pub struct Transition {
    ids: Vec<String>,            // idx -> node id
    idx: HashMap<String, usize>, // node id -> idx
    adj: Vec<Vec<(usize, f32)>>, // undirected weighted adjacency (neighbor idx, tier weight)
}

impl Transition {
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    fn index_of(&self, id: &str) -> Option<usize> {
        self.idx.get(id).copied()
    }
    /// Node ids in walk order (for persistence).
    pub fn ids(&self) -> &[String] {
        &self.ids
    }
    /// Undirected weighted adjacency in walk order (for persistence).
    pub fn adj(&self) -> &[Vec<(usize, f32)>] {
        &self.adj
    }
    /// Reassemble from persisted `(ids, adj)`; the id->index map is derived, not stored.
    pub fn from_parts(ids: Vec<String>, adj: Vec<Vec<(usize, f32)>>) -> Transition {
        let idx = ids
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect();
        Transition { ids, idx, adj }
    }
}

/// Load every node + edge into a symmetric adjacency. Edges whose endpoints aren't both present, and
/// self-loops, are dropped. No ontology is consulted.
pub fn build_transition(g: &GraphStore) -> anyhow::Result<Transition> {
    let nodes = g.all_nodes()?;
    let mut idx: HashMap<String, usize> = HashMap::with_capacity(nodes.len());
    let mut ids: Vec<String> = Vec::with_capacity(nodes.len());
    for n in &nodes {
        idx.entry(n.id.clone()).or_insert_with(|| {
            ids.push(n.id.clone());
            ids.len() - 1
        });
    }
    let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); ids.len()];
    let w_sim = sim_weight(g.gdir());
    let w_spine = spine_weight(g.gdir());
    // Ontology loaded ONCE for the per-edge spine (Chaining-role) lookup — reading role from ontology
    // DATA keeps the weighting ontology-blind (no hardcoded relation names). `load_or_default` never
    // fails, so a missing/absent ontology yields the default (every non-SIMILAR edge fails open to
    // Chaining → spine); with the 1.0 default `w_spine` that is still a no-op.
    let onto = g
        .gdir()
        .parent()
        .map(crate::graph::ontology::Ontology::load_or_default)
        .unwrap_or_default();
    for e in g.all_edges()? {
        if let (Some(&a), Some(&b)) = (idx.get(&e.from), idx.get(&e.to)) {
            if a != b {
                // Guard confidence: treat 0 or negative as 1.0 (default) so legacy graphs are unchanged.
                // Clamp to [0,1] — guards a future >1.0 edge from over-weighting past the reasoning tier.
                let confidence = if e.prov.confidence > 0.0 {
                    e.prov.confidence.min(1.0)
                } else {
                    1.0
                };
                let is_spine = onto.relation_role(&e.edge_type)
                    == crate::graph::ontology::RelationRole::Chaining;
                let w = edge_tier_weight(&e.edge_type, w_sim, w_spine, is_spine) * confidence;
                adj[a].push((b, w));
                adj[b].push((a, w)); // symmetric — a multi-hop answer may need to walk "backward"
            }
        }
    }
    Ok(Transition { ids, idx, adj })
}

/// Random walk with restart to a stationary distribution. `seeds` maps node id -> unnormalized mass
/// and is normalized to the restart vector `p`. `alpha` is the restart probability. Returns
/// (node id, stationary score) sorted descending, skipping zero-mass nodes. Dangling nodes (no
/// edges) teleport their mass back to `p` so total mass is conserved every iteration.
pub fn ppr(
    trans: &Transition,
    seeds: &HashMap<String, f32>,
    alpha: f32,
    max_iter: usize,
    eps: f32,
) -> Vec<(String, f32)> {
    let n = trans.len();
    if n == 0 {
        return Vec::new();
    }
    // Restart vector p: normalized seed mass.
    let mut p = vec![0.0f32; n];
    let mut sum = 0.0f32;
    for (id, &m) in seeds {
        if m > 0.0 {
            if let Some(i) = trans.index_of(id) {
                p[i] += m;
                sum += m;
            }
        }
    }
    if sum <= 0.0 {
        return Vec::new(); // no seed landed on a real node
    }
    for v in p.iter_mut() {
        *v /= sum;
    }

    let mut r = p.clone();
    for _ in 0..max_iter {
        let mut next = vec![0.0f32; n];
        let mut dangling = 0.0f32;
        for (i, &ri) in r.iter().enumerate() {
            let wdeg: f32 = trans.adj[i].iter().map(|(_, w)| *w).sum();
            if wdeg <= 0.0 {
                dangling += ri;
            } else {
                for &(j, w) in &trans.adj[i] {
                    next[j] += ri * w / wdeg;
                }
            }
        }
        // r' = (1-alpha)*(M r) + [alpha + (1-alpha)*dangling]*p  — teleport keeps sum(r)=1.
        let teleport = alpha + (1.0 - alpha) * dangling;
        let mut delta = 0.0f32;
        for i in 0..n {
            let v = (1.0 - alpha) * next[i] + teleport * p[i];
            delta += (v - r[i]).abs();
            next[i] = v;
        }
        r = next;
        if delta < eps {
            break;
        }
    }

    let mut out: Vec<(String, f32)> = r
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0.0)
        .map(|(i, &v)| (trans.ids[i].clone(), v))
        .collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::store::{Edge, GraphStore, Node, Provenance};

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
    fn node(g: &GraphStore, id: &str) {
        g.put_node(&Node {
            id: id.into(),
            node_type: "Fact".into(),
            label: id.into(),
            aliases: vec![],
            prov: prov(),
        })
        .unwrap();
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
    fn seed(ids: &[(&str, f32)]) -> HashMap<String, f32> {
        ids.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }
    fn rank_of(out: &[(String, f32)], id: &str) -> Option<usize> {
        out.iter().position(|(i, _)| i == id)
    }

    #[test]
    fn connected_terminal_outranks_a_disconnected_node() {
        // seed -> bridge -> terminal; `far` is an isolated node. The terminal shares no seed mass
        // directly, but connectivity carries mass to it; the disconnected node stays at zero.
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        for x in ["seed", "bridge", "terminal", "far"] {
            node(&g, x);
        }
        link(&g, "seed", "bridge");
        link(&g, "bridge", "terminal");
        let t = build_transition(&g).unwrap();
        let out = ppr(&t, &seed(&[("seed", 1.0)]), 0.15, 50, 1e-6);
        assert!(
            rank_of(&out, "terminal").is_some(),
            "terminal reached: {out:?}"
        );
        assert!(
            rank_of(&out, "far").is_none(),
            "disconnected node gets no mass: {out:?}"
        );
    }

    #[test]
    fn multi_path_answer_outranks_hub_leaf() {
        // `answer` is reachable from the seed two ways (direct and via the hub); `noise*` hang off
        // the hub only. Even though the hub has high degree, the answer's extra path floats it above
        // any single hub leaf — the property the df-cap hack approximated.
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        node(&g, "seed");
        node(&g, "hub");
        node(&g, "answer");
        for i in 0..10 {
            node(&g, &format!("noise{i}"));
            link(&g, "hub", &format!("noise{i}"));
        }
        link(&g, "seed", "hub");
        link(&g, "seed", "answer");
        link(&g, "hub", "answer");
        let t = build_transition(&g).unwrap();
        let out = ppr(&t, &seed(&[("seed", 1.0)]), 0.15, 50, 1e-6);
        let ra = rank_of(&out, "answer").unwrap();
        let rn = rank_of(&out, "noise0").unwrap();
        assert!(ra < rn, "answer outranks a hub leaf: {out:?}");
    }

    #[test]
    fn shared_neighbor_of_two_seeds_outranks_single_seed_neighbor() {
        // Cone-intersection intuition, soft: `both` neighbors both seeds; `one` neighbors only s1.
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        for x in ["s1", "s2", "both", "one"] {
            node(&g, x);
        }
        link(&g, "s1", "both");
        link(&g, "s2", "both");
        link(&g, "s1", "one");
        let t = build_transition(&g).unwrap();
        let out = ppr(&t, &seed(&[("s1", 1.0), ("s2", 1.0)]), 0.15, 50, 1e-6);
        assert!(
            rank_of(&out, "both").unwrap() < rank_of(&out, "one").unwrap(),
            "intersection node ranks above the single-seed neighbor: {out:?}"
        );
    }

    #[test]
    fn stationary_distribution_is_normalized() {
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        for x in ["a", "b", "c"] {
            node(&g, x);
        }
        link(&g, "a", "b");
        link(&g, "b", "c");
        let t = build_transition(&g).unwrap();
        let out = ppr(&t, &seed(&[("a", 1.0)]), 0.15, 100, 1e-9);
        let total: f32 = out.iter().map(|(_, v)| v).sum();
        assert!(
            (total - 1.0).abs() < 1e-3,
            "mass conserved to ~1.0, got {total}"
        );
    }

    #[test]
    fn empty_seed_yields_nothing() {
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        node(&g, "a");
        let t = build_transition(&g).unwrap();
        assert!(ppr(&t, &seed(&[("nonexistent", 1.0)]), 0.15, 50, 1e-6).is_empty());
    }

    #[test]
    fn three_tier_edge_weights() {
        // Tier semantics only (explicit weights; role resolution is the caller's job, tested
        // separately). w_sim = 0.1 (penalize SIMILAR), w_spine = 2.0 (boost the reasoning spine).
        let (w_sim, w_spine) = (0.1_f32, 2.0_f32);
        // Non-spine reasoning/structural edges (is_spine=false) carry full mass.
        assert_eq!(edge_tier_weight("MENTIONS", w_sim, w_spine, false), 1.0);
        assert_eq!(edge_tier_weight("CONTAINS", w_sim, w_spine, false), 1.0);
        assert_eq!(edge_tier_weight("NEXT", w_sim, w_spine, false), 1.0);
        // A spine (Chaining-role) edge is BOOSTED.
        assert_eq!(edge_tier_weight("LEADS_TO", w_sim, w_spine, true), 2.0);
        // SIMILAR is checked FIRST — even if a missing role failed open to spine (is_spine=true), a
        // soft edge still gets `w_sim`, never the boost.
        assert_eq!(edge_tier_weight("SIMILAR", w_sim, w_spine, true), 0.1);
        assert!(edge_tier_weight("SIMILAR", w_sim, w_spine, false) < 1.0);
        // Default w_spine = 1.0 makes the spine tier a no-op (backward compatible).
        assert_eq!(edge_tier_weight("LEADS_TO", w_sim, 1.0, true), 1.0);
    }

    #[test]
    fn sim_weight_resolves_default_then_ontology() {
        // Neutralize any ambient env override (a dev/CI shell may export GLOSSA_PPR_SIM_WEIGHT) so
        // this test exercises the config + default path deterministically.
        std::env::remove_var("GLOSSA_PPR_SIM_WEIGHT");
        let d = tempfile::tempdir().unwrap();
        let gdir = d.path().join(".glossa");
        std::fs::create_dir_all(&gdir).unwrap();

        // No ontology (or no [retrieval] key) → engine default 0.1.
        assert_eq!(sim_weight(&gdir), 0.1);

        // A per-corpus [retrieval].sim_weight is read.
        std::fs::write(
            gdir.join("ontology.toml"),
            "[retrieval]\nsim_weight = 0.42\n",
        )
        .unwrap();
        assert_eq!(sim_weight(&gdir), 0.42);

        // A malformed (negative) value is rejected → falls back to the default.
        std::fs::write(
            gdir.join("ontology.toml"),
            "[retrieval]\nsim_weight = -1.0\n",
        )
        .unwrap();
        assert_eq!(sim_weight(&gdir), 0.1);
    }

    #[test]
    fn cache_sig_differs_by_either_weight() {
        // Same content signature, different w_sim OR w_spine → different cache signature, so a cache
        // built at one weight pair is a miss at another (the whole point — a GLOSSA_PPR_*_WEIGHT flip
        // must take effect even when the graph is unchanged). Same pair → same sig (a hit).
        let content = 0xDEAD_BEEF_u64;
        assert_ne!(cache_sig(content, 0.1, 1.0), cache_sig(content, 0.3, 1.0)); // w_sim varies
        assert_ne!(cache_sig(content, 0.1, 1.0), cache_sig(content, 0.1, 2.0)); // w_spine varies
        assert_ne!(cache_sig(content, 0.1, 2.0), cache_sig(content, 0.2, 1.0)); // both vary
        assert_eq!(cache_sig(content, 0.3, 1.5), cache_sig(content, 0.3, 1.5)); // identical pair
    }

    #[test]
    fn spine_weight_resolves_default_then_ontology() {
        std::env::remove_var("GLOSSA_PPR_SPINE_WEIGHT");
        let d = tempfile::tempdir().unwrap();
        let gdir = d.path().join(".glossa");
        std::fs::create_dir_all(&gdir).unwrap();
        // No ontology (or no [retrieval] key) → engine default 1.0 (a no-op).
        assert_eq!(spine_weight(&gdir), 1.0);
        // A per-corpus [retrieval].spine_weight is read (a boost > 1 is allowed).
        std::fs::write(
            gdir.join("ontology.toml"),
            "[retrieval]\nspine_weight = 2.5\n",
        )
        .unwrap();
        assert_eq!(spine_weight(&gdir), 2.5);
        // A malformed (negative) value is rejected → falls back to the default.
        std::fs::write(
            gdir.join("ontology.toml"),
            "[retrieval]\nspine_weight = -1.0\n",
        )
        .unwrap();
        assert_eq!(spine_weight(&gdir), 1.0);
    }

    #[test]
    fn transition_roundtrips_weighted_adjacency() {
        let ids = vec!["a".to_string(), "b".to_string()];
        let adj = vec![vec![(1usize, 0.1f32)], vec![(0usize, 0.1f32)]];
        let t = Transition::from_parts(ids.clone(), adj.clone());
        assert_eq!(t.ids(), &ids[..]);
        assert_eq!(t.adj(), &adj[..]);
    }

    #[test]
    fn build_transition_weights_similar_below_reasoning() {
        // Neutralize any ambient env override so this asserts the engine default (0.1) on a corpus
        // with no `[retrieval].sim_weight`.
        std::env::remove_var("GLOSSA_PPR_SIM_WEIGHT");
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        for x in ["a", "b", "c"] {
            node(&g, x);
        }
        // a -LEADS_TO- b  (weight 1.0),  a -SIMILAR- c  (weight w_sim)
        g.put_edge(&Edge {
            from: "a".into(),
            to: "b".into(),
            edge_type: "LEADS_TO".into(),
            prov: prov(),
        })
        .unwrap();
        g.put_edge(&Edge {
            from: "a".into(),
            to: "c".into(),
            edge_type: "SIMILAR".into(),
            prov: prov(),
        })
        .unwrap();
        let t = build_transition(&g).unwrap();
        let ai = t.ids().iter().position(|s| s == "a").unwrap();
        let w: std::collections::HashMap<_, _> = t.adj()[ai]
            .iter()
            .map(|(j, w)| (t.ids()[*j].clone(), *w))
            .collect();
        assert_eq!(w["b"], 1.0);
        assert_eq!(w["c"], 0.1); // SIMILAR down-weighted to the default tier (0.1)
    }

    #[test]
    fn reasoning_terminal_outranks_similar_sibling() {
        // seed -LEADS_TO- terminal (the real bridge)
        // seed -SIMILAR- sibling, and give the sibling extra SIMILAR mass so at EQUAL weight it
        // would win.
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        for x in ["seed", "terminal", "sibling"] {
            node(&g, x);
        }
        g.put_edge(&Edge {
            from: "seed".into(),
            to: "terminal".into(),
            edge_type: "LEADS_TO".into(),
            prov: prov(),
        })
        .unwrap();
        for x in ["s1", "s2", "s3"] {
            node(&g, x);
            g.put_edge(&Edge {
                from: "sibling".into(),
                to: x.into(),
                edge_type: "SIMILAR".into(),
                prov: prov(),
            })
            .unwrap();
        }
        g.put_edge(&Edge {
            from: "seed".into(),
            to: "sibling".into(),
            edge_type: "SIMILAR".into(),
            prov: prov(),
        })
        .unwrap();
        let t = build_transition(&g).unwrap();
        let out = ppr(&t, &seed(&[("seed", 1.0)]), 0.15, 50, 1e-6);
        assert!(
            rank_of(&out, "terminal") < rank_of(&out, "sibling"),
            "reasoning terminal must outrank the SIMILAR-clustered sibling: {out:?}"
        );
    }

    #[test]
    fn low_confidence_edge_carries_less_mass() {
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        for x in ["a", "b"] {
            node(&g, x);
        }
        let mut p = prov();
        p.confidence = 0.5;
        g.put_edge(&Edge {
            from: "a".into(),
            to: "b".into(),
            edge_type: "LEADS_TO".into(),
            prov: p,
        })
        .unwrap();
        let t = build_transition(&g).unwrap();
        let ai = t.ids().iter().position(|s| s == "a").unwrap();
        assert_eq!(t.adj()[ai][0].1, 0.5); // 1.0 (tier) * 0.5 (confidence)
    }
}
