# kbx distil — gold-anchored ontology-typed reasoning layer (design)

**Status:** design, not built. Frame agreed in-session 2026-08-23.

## Goal

A `kbx distil` verb that builds the ontology-TYPED reasoning layer **supervised from gold (question, answer) pairs**: it grounds each gold answer in the corpus, walks the reasoning chain that leads to it — grounding every node on the way and typing them per the corpus's `ontology.toml` — and reifies the question as the chain's entry node. Optionally it augments with **synthetic** grounded chains. The typed layer is added ALONGSIDE the flat `Fact`/`LEADS_TO` base layer in the same `graph.sqlite`.

This SUPERSEDES the earlier unsupervised full-corpus re-extraction design: distil does not type the whole corpus, it materializes only the **verified reasoning chains** (real + synthetic), which is thinner and directly useful.

## The ontology is a typed schema-graph (the universality key)

`ontology.toml` declares `entity_types` (node types) and `relations`, each relation carrying `from_type` / `to_type` / `role` (Chaining|Grounding). Read the Chaining relations as directed edges `from_type --relation--> to_type`: the ontology IS a directed graph over TYPES — the "schema-graph".

A reasoning chain for a question is a **path in this schema-graph**, instantiated with real, grounded entities. That is what makes the whole mechanic ontology-general — nothing is hardcoded to symptom/cause; those are just the entity_types/relations of the troubleshooting preset. A people/places ontology gives other path shapes over the same machinery.

## Gold anchors + per-node grounding

A gold `(Q, A)` gives TWO anchors on a schema-graph path:
- **A (answer) → a known-true terminal node.** The model, guided by the ontology, types A as the `entity_type` it fits (which must be a `to_type` of some relation) and grounds it to the chunk(s) that state it. Because A is gold, this grounding is unambiguous — the reliable anchor.
- **Q (question) → entry node.** The question is reified as a typed entry node (a Problem/query archetype). This is the QUERY SIDE and is **usually NOT grounded** — the corpus holds general knowledge, not this user's specific ticket, so the entry node has no `MENTIONS`. It hangs off the grounded knowledge chain (e.g. `Problem --HAS_SYMPTOM--> grounded Symptom`).

**Grounding is governed by the ontology's existing per-type `requires_grounding` flag — no new mechanism, no exemption.** `ontology.toml` sets `requires_grounding = true` on each `entity_type` that must carry a `MENTIONS`; this parses into `Ontology.grounding_types`, and the doctor's `ungrounded_nodes(nodes, edges, grounding_types)` hygiene check ONLY flags types that require it. So:
- Knowledge types (the corpus describes them — Symptom, Cause, Resolution, `Fact`) declare `requires_grounding = true`; distil grounds them and the doctor enforces it.
- Query/entry types (Problem/question archetypes) simply leave `requires_grounding` unset (default false); the doctor never flags them.

The entry node reified from the question falls out for free: it's a query type without `requires_grounding`, so being ungrounded is legal by config, not by a special case. A factual-entity entry that IS a corpus entity can still be grounded opportunistically. distil MUST honor `requires_grounding` — ground (and require grounding on) exactly the types the ontology marks, nothing hardcoded.

Retrieval implication: the entry nodes are the QUERY-FACING side of the typed layer — ungrounded Problem/question archetypes that a NEW user question matches against, then walks the grounded chain to reach the answer. The typed layer thus maps query-shapes to grounded answer-paths.

## Backward-anchored chain construction

From the grounded answer, distil discovers and grounds the intermediate typed nodes and the ontology-legal `relations` connecting them **back toward the question's entities**. Backward because the terminal is fixed and verified, so each step is a NARROW, checkable search — "what grounded thing, via which ontology relation (matching from/to types), leads to this node?" — instead of open-ended forward guessing. The ontology's relations constrain the legal steps; the corpus text supplies the actual links. Result: a grounded typed path `entry → … → terminal`.

Thin/correct discipline (unchanged): a typed node/edge is emitted iff sound; a hop that can't be grounded/justified is not invented. NO per-edge confidence field, NO runtime weight (see "Invariants").

## Synthetic augmentation (optional, count N)

Seeded from **any grounded typed node of any type T** (not only terminals): take a grounded node, traverse the ontology's relations in EITHER direction to build a new grounded chain, and synthesize its `(Q, A)`. Example (support): take a grounded Cause chunk → understand it → find its Symptom (backward via `CAUSED_BY`) and/or its Resolution (forward) → synthesize a Problem/question. Bounded by a requested count. This expands the typed layer and produces training data beyond the gold set. Universal: same "walk the schema-graph from a grounded seed" over any ontology.

## Two frames — BOTH supported (same engine, different gold scope + output intent)

- **(A) Honest graph-transfer measurement.** distil builds the typed layer from a TRAIN split of the gold only; eval runs on a HELD-OUT test split. The typed layer is then built blind to the test questions, so the "does the typed layer earn its keep" ablation stays honest — this respects [[per-query-graph-eval-plan]] / "strong builds blind to questions." Synthetic augmentation aids generalization to the test split.
- **(B) Domain-KB + training-data construction.** distil builds from ALL available solved cases (gold = closed tickets) + synthetic → a production domain knowledge base and training data for `kbx train`. Here question-contamination is a NON-issue because the goal is a working KB, not a blind benchmark — building a support KB from resolved cases is exactly correct.

Contamination guard: mode (A) requires a train/test split of the gold set and must never ingest a test question. Mode (B) has no such constraint. The mode is explicit (a CLI flag / gold-set scoping), never inferred.

## Architecture (additive, shared grounding)

Two layers in one `graph.sqlite`:

| | Base (`kbx build`) | Typed (`kbx distil`) |
|---|---|---|
| Nodes | `Fact` | ontology `entity_types` on verified chains |
| Edges | `LEADS_TO` (Chaining) | ontology `relations` (Chaining) |
| Grounding | `MENTIONS` → `<path>#n` | `MENTIONS` → same `<path>#n` (shared) |
| Source | weak model, unsupervised | strong model (`lab.[distil]`), gold-anchored |

Separated by `node_type`/`edge_type`; `Ontology::validate_node`/`validate_edge` (existing) enforce the schema — no new validation code. distil never modifies the flat layer. Typed nodes share the alias/resolve space with `Fact` so `glossary(entity)` surfaces both layers and they join.

## Temporality (config-driven, first-class)

The engine already carries valid-time (the `node_validity` side table: `valid_from`/`valid_to`(+`_raw`), plus as-of traversal — [[temporality-roadmap]] Phase 1), and the ontology declares `requires_validity = true` per `entity_type` (parsed into `Ontology.requires_validity(type)`, parallel to `requires_grounding`). distil **honors `requires_validity`**: for each typed node whose type requires it, distil sets `valid_from`/`valid_to` from the corpus text (a role held over a span, a state true in a window, a dated fact). This is what lets as-of / temporal questions resolve ("who was CEO in 2010", "the capital at that time") — the chain is walked as-of the question's time. Backward-chain and synthetic both preserve valid-time on validity-required nodes. Nothing hardcoded: which types need validity is the ontology author's declaration, same pattern as grounding.

Note — a matching gap exists in the shipped `kbx build`: its `builder.md` does not instruct `valid_from`/`valid_to` and the extract stage does not write them, so time-bound `Fact`s currently lose their valid-time even though the store supports it and the `Fact` type asks for it. That is a SEPARATE build retrofit (builder.md temporality instruction + extract valid-time write + a doctor `requires_validity` hygiene check), tracked outside this distil plan but sharing the same `node_validity`/`requires_validity` machinery.

## Invariants (carried from the agreed frame)

- **Ontology-general:** everything is defined over the ontology's schema-graph + gold anchors; no domain type names in code or prompts.
- **Thin, no confidence:** a typed link exists iff sound; no `confidence` field, no hedging.
- **No runtime weight:** typed relation wins by specificity where it exists; flat is the fallback; thin layers don't flood. The only measurement is the **layer-level ablation** flat-only vs flat+typed (arm-switchable read-time typed-hide filter), like graph-ON/OFF.

## Pipeline

1. **index** — ensure indexed (reuse).
2. **read-ontology** — load `Ontology`; build the typed schema-graph (Chaining relations as typed edges).
3. **per gold (Q, A)** — ground the answer (terminal); type+ground the question entities (entry); backward-chain grounding intermediates + ontology-legal relations to the entry; emit the thin typed chain (`validate_node`/`validate_edge` gate every write).
4. **synthetic (optional, N)** — seed from grounded nodes, traverse the schema-graph either direction, emit new grounded chains + synthesized `(Q,A)` (written to a dataset file for `kbx train`).
5. **finalize** — hygiene/doctor + node-index rebuild (reuse build's finalize).

Resumable per-unit checkpoint (per gold / per synthetic item), `--force`, `--limit` — like `kbx build`.

## Config / CLI

- `lab.[distil]` = strong-model endpoint (reserved).
- `distil.md` = the ontology-parameterized behavior-guide prompt (fed the ontology's types/relations at runtime; no type names baked in).
- `kbx distil [PATH] --gold <dataset> --mode {split,kb} --test-frac <f> --synthetic <N> --stage {chains,synthetic,finalize,all} --limit <n> --force --resume --no-progress`.

## Spike validation (2026-08-23, kb-abac, luna)

A throwaway spike ran the backward-chain on 3 kb-abac gold cases against the real support ontology. Result: **feasible.** Across all runs — ZERO fabricated groundings; luna reliably distinguished "I can quote this" from "I can't" and flagged ungroundable values honestly instead of inventing them; every emitted edge respected the ontology's from/to typing on the first try. One case produced a genuine multi-node chain (Symptom→Cause, Symptom→Resolution, Resolution→SETS→Parameter) with byte-verbatim groundings. So the design's thin / no-confidence / honest-gap stance is what a strong model does NATURALLY here — the dangerous failure (unsound or hallucinated chains) did not occur.

Two refinements the spike surfaced (fold into the plan):
1. **Prompt against under-construction (not a soundness fix).** When the terminal is hard to ground, the model tends to collapse to a bare one-hop stub instead of tracing the full reasoning path through the real, groundable intermediates that are present. `distil.md` must push "trace the whole reasoning path, grounding each genuine intermediate — do not stop at one hop just because the terminal is hard." Still thin (only real on-path steps), just not truncated.
2. **Groundable answer vs synthesized answer.** Some golds' answers are not a literal corpus span (a paraphrase / a synthesized "best answer"). Those legitimately yield a partial chain to the nearest groundable node, or are skipped — this is correct honesty, not failure (cf. the earlier "1572 not in corpus" gap). Frame (A) benchmark: such golds are simply hard/unanswerable; Frame (B) KB: prefer golds whose answer IS a groundable fact. The synthetic verify-gate is the same check in reverse.

## Open questions (resolve in the plan)

- **Answer/entry type inference:** how the model picks the terminal/entry `entity_type` for a given (Q,A) — pure model inference guided by the ontology, or an optional per-dataset hint (question-type → target-type)? Default: model inference constrained by the schema-graph (terminal must be a relation `to_type`).
- **Chain search when multiple schema-graph paths connect entry→terminal:** prefer the shortest grounded path? Let the model pick the one the corpus actually supports? Default: the grounded path the corpus supports; shortest as tie-break.
- **Synthetic quality gate:** synthesized `(Q,A)` must be answerable from the grounded chain alone (no leakage of unstated facts). Needs a verify step (answer the synthetic Q from the chain; keep only if it resolves).
- **Ablation baseline mechanics:** read-time typed-hide filter (recommended) vs pre-distil snapshot.
- **Cost/throughput:** strong-model, per-gold — resume + `--limit` essential.

## Non-goals

- No unsupervised full-corpus typing (superseded by gold-anchored chains + synthetic).
- No per-edge confidence, no runtime layer-weighting.
- distil never modifies/replaces the flat `Fact`/`LEADS_TO` layer; no new ontology schema mechanism.
