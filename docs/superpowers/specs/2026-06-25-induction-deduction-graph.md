# Design note: the agentic graph as an induction/deduction substrate

**Status:** design direction (operator-steered, 2026-06-25). Not a plan yet — captures the architecture;
near-term seed + roadmap split at the end. Motivated by the real-base (industrial-automation KB)
retrieval-path analysis: failures were vocabulary + implicit-environment gaps, both resolvable *from the
base if the agent reasons*, not missing content.

## Thesis: graph edges ARE induction/deduction relations
glossa's agentic graph is not just a structural index — its semantic edges materialize the two reasoning
modes:
- **Inductive edges** (specific → abstract): `symptom → subsystem`, `client phrasing → environment`,
  `observation → general rule`. *Building* these edges IS performing induction — generalizing scattered
  document facts into reusable abstractions.
- **Deductive edges** (abstract → specific): `environment/version → applicable procedure`,
  `general rule → concrete answer`, `category → resolving document`. *Traversing* these IS deduction —
  applying the abstraction to the specific question.

So the graph is a **reasoning substrate**: the agent induces it from the base, then deduces answers over it.
This reframes "determine the environment first" (STRATEGY.md): the environment is determined by traversing
inductive edges to the right abstraction, then deducing within it.

## Two prod tasks ⇒ two functions (corrects the earlier "one function" stance)
Both halves are REAL production operations (not an eval-only split), so they are genuinely separate TZ
functions, both deployed, same underlying model:
1. **`build_graph` (induction agent).** Multi-step research over the base → upserts typed edges
   (symptom↔term, product↔subsystem↔version, bridge links, heuristics). Periodic/offline; saturates graph
   layers 3–4 via `graph_upsert`. Feedback = graph quality vs a reference graph.
2. **`answer` (deduction agent).** Env-first answering that *traverses* the graph to determine context and
   resolve. Feedback = `em`/`f1`/`retrieved`.
(This matches TZ's own multi-hop benchmark, which trained two specialized policies — decomposition is right
when both are real tasks.)

## Optimization via a reference graph (distillation)
A strong model builds a **reference/gold graph** for the curated Q/A (the induction done well). Then TZ
optimizes Qwen's `build_graph` prompt to approach that reference (curated by edge-precision/recall vs gold),
and separately optimizes the `answer` prompt (`retrieved=true ∧ em=false` curation). Same single model in
prod, two optimized prompts. Seed for the reference graph already exists: the 54 edge candidates in
`kb-val/derived/REENGINEERING.md`.

## Ontology revision (the second operator point)
Current `ontology.toml` is structural only (Document/Section/CONTAINS). To carry reasoning semantics, extend it:
- **Node types (add):** `Environment`, `Product`, `Subsystem`, `Version`, `Standard`, `Symptom`,
  `Term`/`Alias`, `Concept`, `Heuristic` — alongside `Document`/`Section`.
- **Edge types, classed by reasoning mode:**
  - *Inductive:* `INDICATES` (symptom→cause), `SYMPTOM_OF`, `DISAMBIGUATES` (signal→environment),
    `GENERALIZES_TO`, `ALIAS_OF` (everyday term→technical token).
  - *Deductive:* `APPLIES_TO` (rule→case), `GOVERNS` (standard→procedure), `REQUIRES`, `RESOLVED_BY`
    (problem→document), `VALID_FOR` (procedure→version/environment).
  - *Structural (keep):* `CONTAINS`, `MENTIONS`.
- Edges should keep provenance (which doc induced them) — File-First: the graph is a disposable overlay
  rebuildable from the base.

## Near-term vs roadmap
- **Near-term (autonomous, low-risk):** formalize the REENGINEERING.md edge candidates into a typed
  **reference graph** (induction/deduction edge classes above) for the kb-val items — the gold target.
- **Roadmap (needs brainstorm→spec→plan):**
  1. Ontology revision (new node/edge types with reasoning semantics).
  2. `build_graph` induction agent (multi-step research → typed `graph_upsert`).
  3. Reference-graph distillation: optimize Qwen's `build_graph` prompt toward the gold graph (TZ recipe).
  4. Two-function (`build_graph` + `answer`) optimization, both deploying to the one prod model.
  5. `answer` agent traverses the semantic graph for env-determination (depends on 1–2).
