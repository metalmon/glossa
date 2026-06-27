# Graph Hygiene — Degenerate-Chain Prune — Design Spec

**Status:** approved (2026-06-27)
**Sequence:** Task 2 of 2. Builds on Task 1 (configurable ontology) — consumes `[reasoning].spine`.

## Goal

Before the generalization pass derives edges/communities/centrality, remove **degenerate
reasoning chains** — reasoning nodes that do NOT lie on a complete spine instance (for support:
`Symptom →CAUSED_BY→ Cause →RESOLVED_BY→ Resolution`). A Cause-less `Symptom→Resolution` adds
nothing over plain BM25 search (search already maps the query to that answer doc); the graph's
only added value is the Cause hop. Degenerate nodes also skew communities/centrality/link-pred.

Removal is **report-only by default**; it deletes only under an explicit `--prune-incomplete`
flag (mirrors `apply_merges`).

## Architecture

A new `generalize/hygiene.rs` with a PURE function over (id→type) nodes + `Triple` edges + the
ontology's `spine`/`structural`/`spine_types` — returns the ids to delete. `generalize()` runs
it first; when `opts.prune_incomplete` is set it calls a new `store::delete_nodes` (cascading
edges + node_meta), otherwise it only counts. Fully generic: zero domain literals (the spine and
types come from the ontology, per the Task-1 first principle).

## Tech Stack

Rust; existing `GraphStore` (rusqlite). No new dependencies.

## Global Constraints

- Pure-Rust, C-free; no new deps.
- **No domain literals** — spine relations, reasoning types, and structural types all come from
  the ontology (`spine()`, `spine_types()`, `structural()`).
- **Report-only default**; deletion only with `opts.prune_incomplete` / CLI `--prune-incomplete`.
  The `reindex` auto-run never prunes (stays non-destructive).
- **Empty spine → no-op** (an ontology without `[reasoning].spine` prunes nothing).
- Structural nodes (Document/Section/Term/Topic) are NEVER deleted (document layer).
- TDD; File-First.

---

## Section 1 — Completeness rule (the survivor set)

Let `spine = [r0, r1, …, r_{k-1}]` (ordered relation names from the ontology).

**Complete chain instance:** a node path `v0 -r0-> v1 -r1-> … -r_{k-1}-> vk`.
**Core survivors:** every node appearing at any position of any complete chain instance.

Computed without path enumeration:
- Backward `can_finish[i][n]`: `can_finish[k][*] = true`; `can_finish[i][n] = ∃ edge n -r_i-> m
  with can_finish[i+1][m]`.
- Forward `at[0] = { n : can_finish[0][n] }`; `at[i] = { m : can_finish[i][m] ∧ ∃ p∈at[i-1],
  p -r_{i-1}-> m }`. Core survivors = ⋃_i at[i]. O(k · |edges|).

## Section 2 — Cascade (auxiliary nodes) & the spine-types barrier

Node categories (all from the ontology, no literals):
- **structural** = `ontology.structural()` (default core four) — never pruned, never traversed.
- **spine types** = `ontology.spine_types()` = the union of `from`+`to` entity types of every
  relation named in `spine` (e.g. Symptom/Cause/Resolution). A spine-type node that is not a core
  survivor is **doomed**.
- **auxiliary** = every other non-structural type (Parameter/Product/Module/Medium/Standard/
  Version/Task).

**Keep set:** BFS seeded from core survivors, traversing edges whose other endpoint is
non-structural **and not a doomed spine-type node** (MENTIONS is excluded automatically — its
target Section is structural). This barrier is why we need `spine_types`: an auxiliary reachable
only *through* a doomed node (e.g. `Parameter ←SETS— deadResolution`) must NOT be rescued, since
that bridge node is about to be deleted. Auxiliaries chain (Parameter -OF-> Module) and survive
as long as the chain reaches a core survivor without crossing a doomed/structural node.

**Delete set** = all non-structural nodes not in the keep set = doomed spine-type nodes +
unreachable auxiliaries. Task nodes fall out for free (Task has no spine relation → never a core
survivor → doomed-or-unreachable).

## Section 3 — Pure function (src/graph/generalize/hygiene.rs)

```rust
/// Ids to delete: non-structural nodes not on a complete `spine` chain and not transitively
/// attached to one. `nodes` = (id, node_type); `edges` = (from, edge_type, to).
pub fn incomplete_nodes(
    nodes: &[(String, String)],
    edges: &[Triple],
    spine: &[String],
    spine_types: &std::collections::HashSet<String>,
    structural: &std::collections::HashSet<String>,
) -> Vec<String>   // sorted, deterministic
```
Empty `spine` → returns empty (no-op).

## Section 4 — Store (src/graph/store.rs)

```rust
/// Delete the given node ids and every incident edge + node_meta row, in one transaction.
/// Returns the number of node rows removed.
pub fn delete_nodes(&self, ids: &[String]) -> anyhow::Result<usize>
```

## Section 5 — Ontology accessor (src/graph/ontology.rs)

```rust
/// Entity types that are endpoints (from or to) of any relation named in `spine`.
pub fn spine_types(&self) -> std::collections::HashSet<String>
```

## Section 6 — Wiring (src/graph/generalize/apply.rs + main.rs)

- `Opts` gains `prune_incomplete: bool`, `spine: Vec<String>`, `spine_types: HashSet<String>`.
  `from_ontology` fills `spine`/`spine_types` (prune_incomplete stays false by default).
  `defaults` → empty spine (no-op).
- `Report` gains `prune_candidates: usize` and `pruned_nodes: usize`.
- `generalize()`: FIRST step — `let doomed = hygiene::incomplete_nodes(...)`;
  `report.prune_candidates = doomed.len()`; if `opts.prune_incomplete`
  `report.pruned_nodes = g.delete_nodes(&doomed)?` then reload nodes/edges; else proceed
  on the unpruned graph. Merge/closure/meta run after, on whatever graph remains.
- CLI `GraphAction::Generalize` gains `--prune-incomplete` → `opts.prune_incomplete`. The
  `reindex` auto-run leaves it false. Print line gains `prune_candidates`/`pruned_nodes`.

## Section 7 — Testing

Pure function (`hygiene.rs`):
- `complete_chain_survives`: S→C→R kept.
- `symptom_resolution_no_cause_pruned`: S→R (no Cause) → both pruned (the ticket-6 case).
- `symptom_cause_no_resolution_pruned`: S→C with no R → both pruned.
- `orphans_and_isolated_pruned`: lone Cause / lone Resolution / edge-less reasoning node pruned.
- `auxiliary_on_survivor_kept`: Resolution(survivor) -SETS-> Parameter kept; Parameter -OF-> Module kept.
- `auxiliary_behind_doomed_pruned`: Parameter attached only to a doomed Resolution → pruned (barrier).
- `structural_never_pruned`: a Section a doomed Symptom MENTIONS is untouched.
- `empty_spine_is_noop`: no spine → returns empty.

Store (`store.rs`):
- `delete_nodes_cascades_edges_and_meta`: deleting ids removes their nodes, incident edges, meta;
  unrelated nodes/edges survive.

Integration (`apply.rs`):
- `generalize_prune_reports_only_by_default`: degenerate graph → `prune_candidates>0`,
  `pruned_nodes==0`, nodes still present.
- `generalize_prune_applies_with_flag`: `opts.prune_incomplete=true` → degenerate nodes gone,
  closure/meta computed on the cleaned graph.
