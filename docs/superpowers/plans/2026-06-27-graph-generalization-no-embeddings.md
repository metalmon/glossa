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

## Constraints
- **No embeddings endpoint dependency** — the whole point: portability to any deployment/agent.
- Pure-Rust; reuse tantivy (BM25 / morphology / synonym filter) and the existing graph store.
- `auto-generalized` origin so the layer is rebuilt on reindex and never clobbers agent/curated.
