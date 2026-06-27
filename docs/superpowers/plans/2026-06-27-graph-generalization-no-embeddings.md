# Graph Generalization — No-Embeddings Algorithmic Layer

**Decision (2026-06-27):** generalize the reasoning graph with deterministic ALGORITHMS,
never embeddings. An embeddings endpoint is not available in every deployment; keeping
generalization embedding-free means ANY agent / any deployment can run it with zero extra
infrastructure (pure-Rust, offline, portable). Division of labour: the model **extracts**
(symptom → resolution from text); **algorithms generalize** (relate, cluster, rank). This
removes model variance from the generalization step and keeps the eval **honest** (it works
on graph structure + morphology, never on answer text).

## Techniques — all to be implemented

### Relating nodes (algorithmic edge generalization → `SIMILAR` edges)
1. **Morphology match — ALREADY in `resolve`/`glossary`.** `analyze_terms` stemming makes
   inflections and word-order match. Baseline; nothing to build.
2. **BM25-over-labels (tantivy).** Index node labels as documents; for each node BM25-search
   the others; high-overlap pairs → `SIMILAR` edge. Catches shared-word paraphrases
   ("X потеря связи" ≈ "потеря связи X периодическая"). No model.
3. **Shared-evidence linking (anchor-based).** Two reasoning nodes whose `MENTIONS` point to
   the SAME chunk are almost certainly about the same thing → `SIMILAR` (or merge). Catches
   paraphrases with NO shared words. Uses anchors we already have; no words, no model, no
   curation. Cheapest high-value lever specific to our graph.
4. **Structural link-prediction (Jaccard / Adamic-Adar).** Nodes sharing many graph neighbours
   (e.g. two Symptoms sharing a Resolution) → `SIMILAR`. Pure topology, no text.
5. **Synonym dictionary (tantivy synonym token-filter).** For TRUE synonyms that share no stem
   (разрыв ≈ потеря, сбой ≈ ошибка). Needs a domain synonym map (hand-curated or corpus-mined).
   Last resort before embeddings; only the synonym-expansion path requires curation.

**Near-duplicate MERGE (not just `SIMILAR`) — and it's needed IN-CASE, not only cross-case.**
Techniques #2–#4 can either add a `SIMILAR` edge OR *merge* two nodes into one (reattaching all
edges, keeping the others' labels as `aliases`). Merge is the stronger move and is required even
within a single case: observed in practice, the 4B enricher paraphrases its OWN labels — it creates
"Изменение maxTsdr и перезапуск службы", then "Изменение maxTsdr в конфигурации…", then references
the truncated "Изменение maxTsdr" — i.e. several near-duplicate nodes for one concept plus ambiguous
edge references. The shipped band-aid is edge-time fuzzy resolution (`resolve_endpoint_label` in
`ops.rs`: exact label → morphology `resolve` → shortest matching reasoning node). The real fix is a
MERGE pass in `kb graph generalize` that collapses near-dup labels (BM25/morphology overlap above a
threshold, or an identical shared-evidence anchor) into one canonical node, so routing and edge
references are unambiguous.

### Structure over the enriched graph
6. **Community detection (Louvain / label-propagation / connected-components).** Over the edge
   graph including the `SIMILAR` edges → problem FAMILIES. Store a community id on each node.
   `glossary`/`neighbors` then return a node's community siblings → broader routing
   (GraphRAG-style, but WITHOUT LLM community summaries).
7. **Centrality (PageRank / degree).** Hub Resolutions ("fixes many symptoms") / central Causes
   = "key knowledge"; prioritise them in retrieval / surface first.
8. **Transitive closure (ontology rules).** e.g. `A CAUSED_BY B` + `B RESOLVED_BY C` ⇒ ensure
   `A RESOLVED_BY C`. Deterministic edge inference per the ontology's relation composition.

## Architecture
- A deterministic POST-enrichment pass: `kb graph generalize` (CLI / build step), run after
  enrichment and after each reindex.
- Outputs: new edges (`SIMILAR`, inferred relations) stamped `origin = "auto-generalized"` — so
  `reindex`'s `delete_auto` (origin `auto-*`) clears and the pass regenerates them; they are
  derived, never hand-built, so they must NOT overwrite agent/curated edges. Plus node
  attributes (community id, centrality) on the node or a side table.
- Surfaced through the READ tools (`glossary`, `neighbors`) — NOT new agent tools. The agent
  just reads richer neighbourhoods; the generalization itself requires no model call.
- Pure-Rust, no embeddings endpoint, offline. The reasoning graph is small (hundreds of nodes)
  so even O(N²) similarity is fine; reuse tantivy (BM25, morphology, synonym filter) + the
  existing graph store; petgraph or simple hand-rolled algorithms for communities/centrality.
- Honest for eval: structural/morphological only, never answer text.

## Sequencing — measure each, do not optimise blind
1. Ablate the BASE reasoning graph first (graph vs `--no-graph` on the 14 held-out) — does the
   approach help at all?
2. Then layer each technique and ablate its DELTA, **in dependency order — refined for the 4B
   self-paraphrase pattern** (dominant defect: 3+ near-dup nodes per concept + shared chunk
   anchors; plus multi-hop fixes the 4B won't chain). **Dedup FIRST**: closure / communities /
   centrality / link-prediction are garbage-in-garbage-out over a graph full of near-dups —
   communities cluster the dups, centrality splits a hub's weight across copies, closure misses
   because edges are scattered. So order by dependency, NOT by cheapness:
   - **Tier 1 — kill the dominant defect (expect the biggest jump):** MERGE near-dups, signal =
     identical chunk anchor (#3) and/or BM25/morphology label overlap (#2/#1) above a threshold →
     collapse to one canonical node (aliases kept, edges reattached). This is the real fix that
     replaces the edge-time band-aid (`resolve_endpoint_label`). Shared-evidence linking (#3) is
     the same step (it both merges and adds `SIMILAR` for no-shared-word paraphrases). → moves
     `retrieved` / `recall@k` / `em` / `f1`.
   - **Tier 2 — completeness for the weak model:** transitive closure (#8) — PROMOTED above
     communities: on clean edges it makes `Symptom RESOLVED_BY Resolution` reachable in ONE hop,
     so the 4B doesn't have to chain. Then BM25-over-labels (#2) for shared-word paraphrases that
     sit on different chunks. → `em` / `f1` / `judge`.
   - **Tier 3 — structure, only AFTER the graph is deduped:** communities (#6) → centrality (#7).
   - **Tier 4 — residual:** structural link-prediction (#4) → synonym dictionary (#5).
3. Keep only techniques that move the metric.

## Implementation status (2026-06-27)
The deterministic pass is built and wired (`src/graph/generalize/*`, `apply::generalize`,
`kb graph generalize [--merge]`, auto-run at the end of `reindex`). 163 tests passing.

- [x] **#1 Morphology** — `TermAnalyzer` (cached stemmers), shared with `resolve`/`glossary`.
- [x] **#2 BM25-over-labels** — real in-RAM tantivy index (`similarity::label_bm25`), same
      `multilang` analyzer as search. **Relative threshold (fraction of self-score)**, not absolute:
      `bm25_min_ratio = 0.3` (SIMILAR edges), `merge_bm25_min_ratio = 0.7` (merge). Corpus/IDF-
      independent → defaults port across deployments; `0.3 / 0.7` is the safe starting point, tune on data.
- [x] **#3 Shared-evidence** — `similarity::shared_evidence` (same `MENTIONS` chunk anchor),
      feeds both merge candidates and SIMILAR.
- [x] **#4 Link-prediction** — `linkpred::jaccard_pairs` / `adamic_adar_pairs` → SIMILAR.
- [ ] **#5 Synonym dictionary** — deliberately deferred (needs curation; last resort before embeddings).
- [x] **#6 Communities** — `community::connected_components` (union-find). See library note below
      for the Louvain upgrade trigger.
- [x] **#7 Centrality** — `centrality::degree` + `pagerank`. Surfaced (with community) in
      `neighbors`/`glossary` via `node_meta` (`· comm N · pr x · deg N`); empty when not yet
      generalized, so non-generalized output is byte-identical (back-compat).
- [x] **#8 Transitive closure** — `closure::transitive_closure` with ontology-sourced rules
      (`[reasoning].closure`); derived edges stamped `origin = "auto-generalized"`.
- [x] **MERGE** near-dups — `merge::merge_groups` (union-find) + `GraphStore::merge_nodes`
      (canonical = shortest label, aliases folded, edges reattached). Report-only by default;
      applied only via `--merge`.

Still open: the **Sequencing** ablations (measure each technique's metric delta on the held-out set)
have NOT been run yet — the techniques exist; their individual value is not yet proven.

## Graph-library decision: hand-rolled vs petgraph / Louvain
**Decision (2026-06-27): stay hand-rolled for now; do NOT add petgraph.** Rationale: the graph
lives in SQLite (`GraphStore`); algorithms run over an edge list pulled into
`HashMap<String, BTreeSet<String>>`. petgraph works on integer `NodeIndex`, so adopting it means
building a petgraph graph + bidirectional `String ↔ NodeIndex` maps every pass. Of what we use, only
union-find is genuinely covered (`petgraph::unionfind`); PageRank, Jaccard/Adamic-Adar link
prediction, and rule-based closure are NOT in petgraph and stay custom regardless. Our functions are
small, dependency-free, deterministic, and tested — the right trade for the current scale.

**When to revisit (triggers, not now):**
- **Louvain/Leiden communities** — the most likely future need. Connected-components degenerates to
  ONE community once the graph densifies (SIMILAR + shared-evidence + closure cross-link everything).
  **Trigger:** `communities` in the `reindex`/`generalize` report stabilises at 1–2 on a large graph.
  Then implement a **deterministic Louvain WITHOUT petgraph** (over the existing `undirected_adjacency`;
  fix node-iteration order by sorted id for reproducibility — most crates randomise and break
  determinism), behind the SAME facade `community::connected_components(node_ids, edges) -> HashMap<id,
  community>`, so `apply.rs`/`node_meta` are untouched. Weighted variant wants edge weights (currently
  all 1.0); signature unchanged. Skip Leiden (refinement phase) until Louvain proves insufficient.
- **Betweenness/closeness centrality** — marginal over PageRank for "surface the central concept",
  and betweenness is O(V·E). Only if a concrete "find bridging concepts between topics" need appears.
- **Shortest paths / multi-hop** — plausible (the `neighbors depth` param is already stubbed), but
  unweighted BFS is ~20 lines; no library needed.
- **SCC on directed cycles** — niche; only as a data-quality lint (contradictory `CAUSED_BY` cycles).
- General rule: if graph analytics grows beyond this (Louvain + several of the above together), THEN
  reach for petgraph (+ a Louvain crate) rather than hand-rolling everything.

## Constraints
- **No embeddings endpoint dependency** — the whole point: portability to any deployment/agent.
- Pure-Rust; reuse tantivy (BM25 / morphology / synonym filter) and the existing graph store.
- `auto-generalized` origin so the layer is rebuilt on reindex and never clobbers agent/curated.
