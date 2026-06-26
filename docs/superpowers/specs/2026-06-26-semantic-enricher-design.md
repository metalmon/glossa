# Semantic Reasoning-Graph Enricher — Design

**Date:** 2026-06-26
**Status:** Approved (direction confirmed) — SPIKE-gated
**Branch:** `feat/semantic-enricher`

## Goal

Let a qwen3.5-4b **enricher agent** learn the *tacit support reasoning* (symptom → cause → resolution → which section) by reverse-tracing solved cases, and write it into the graph as support-ontology nodes/edges. Then the answer agent, on **held-out** questions, reaches the right sections by traversing this graph via `neighbors`/`glossary` — closing the operator↔manual "среда" gap that BM25 can't.

## The honesty rule (non-negotiable)

The graph stores **routing/reasoning** (a broad Symptom pattern → the Sections that resolve it), **never the answer text**. So even when a held-out question matches a learned pattern, the model is only *routed to a section* — it still reads and synthesizes the answer itself (capability, not memorization). The graph is built ONLY from the **train split**; the **eval-heldout split never touches the graph**. Storing answer text, or building from the eval set, would make the metric dishonest.

## Inputs (done)

- **Support ontology** (`eval/ontology-support.toml`, deploy to `kb-test/ontology.toml`): entities `Symptom, Cause, Resolution, Task, Parameter, Product, Module, Medium, Standard, Version`; relations `CAUSED_BY, RESOLVED_BY, SETS, HANDLED_AS, FIXES, OF, OCCURS_IN`; `strict=true`. Core `MENTIONS` links any node → `Section(path,#n)`.
- **Split** (`kb-val/derived/train.json` = 24, `eval-heldout.json` = 14; 7 empty-gold dropped). Stratified by topic so each held-out class has train support; eval-heldout is the measurement set.

## Components

### 1. `graph_upsert` exposed to an enricher agent
The eval's `exec` (`eval/src/backend/glossa_tools.rs`) gains a `graph_upsert` arm (calls `glossa::graph::agent::apply_upsert`, validated against the ontology). The enricher's tool set = `search, read, grep, graph_upsert` (NOT the Reader profile — this agent MUTATES the graph). A new TZ function `enrich` declares these tools + the enricher prompt.

### 2. The enricher agent (per train case)
For each `(question, gold)` in train.json, the agent:
1. Reads Q + the **known** answer A.
2. Uses `search`/`read`/`grep` to locate the Section(s) the answer comes from.
3. Reverse-traces and **abstracts BROADLY**: a `Symptom` node whose label/aliases are the *broad* problem class (e.g. "PROFIBUS DP потеря связи / watchdog", not "Сгущение №3, ШВВП ×3"); a `Resolution` node (the fix pattern, e.g. "увеличить maxTsdr, пересчитать watchdog"); optional `Cause`.
4. `graph_upsert`s: `Symptom`, `Resolution` (+`Cause`), edges `Symptom →CAUSED_BY→ Cause`, `Symptom/Cause →RESOLVED_BY→ Resolution`, and `MENTIONS` from Symptom/Resolution → the supporting `Section(path,#n)` (the structural Section node already exists — link to it). NO answer text in any label.
   - First pass (spike + v1) uses ONLY the narrow core: `Symptom / Cause / Resolution` + `CAUSED_BY / RESOLVED_BY / MENTIONS` (+ `Task →MENTIONS→ Section` for how-to). Expand to Parameter/Medium/Version after the 4B proves reliable.

### 3. `neighbors`/`glossary` render entity nodes
The bridge renders `Section`/`Document` targets as `(path,#n)`. Extend the shared `node_ref` so a reasoning entity node (`Symptom`/`Resolution`/…) is rendered by **following its `MENTIONS` edges to the Section(s)** → `(path,#n)`. So `glossary("PROFIBUS потеря связи")` → the Symptom node → its Resolution + the `#n` sections; the agent reads them. The agent never sees raw node ids.

### 4. Measurement (ablation)
Run the answer eval on **eval-heldout (14)** twice: (a) graph populated from train, (b) graph WITHOUT the reasoning layer (structural only). Compare Judge. The delta on the reasoning-heavy held-out questions (ticket-11, abac-2, ticket-7, qa30-32, ticket-1) is the signal. Honest ceiling: measurable lift expected on ~5–7 of the 14 (the rest are grep-served lookups).

## Plan shape (SPIKE-FIRST)

- **Task 0 — SPIKE (gates everything):** minimal enricher — `graph_upsert` arm in exec + an `enrich` TZ function/prompt — run on **3–5 train cases** (pick reasoning-rich ones: abac-3, ticket-3, ticket-11, ticket-7). Dump the resulting nodes/edges and **eyeball whether qwen3.5-4b produced sensible, broad, answer-free reasoning**. GO/NO-GO. If the 4B produces garbage or leaks answer text, STOP and rethink (smaller ontology, few-shot prompt, or a stronger enricher model just for build-time).
- Task 1: full enricher over all 24 train cases.
- Task 2: `neighbors`/`glossary` entity-node rendering (MENTIONS → (path,#n)).
- Task 3: ablation measurement on eval-heldout.

## Constraints

- **Honesty:** routing not answers; build from train only; eval-heldout held out.
- The enricher is a **build-time agent** (ours to build) — distinct from the prod *answer* agent, so it does NOT violate the prompt/tools-only prod constraint (that binds the answer loop). In prod the customer runs an equivalent enricher fed their resolved tickets.
- **Tools-level** for the answer agent stays intact: at inference the answer agent only uses the Reader profile; the reasoning graph reaches it through `neighbors`/`glossary`, no harness change.
- `graph_upsert` validated against the strict support ontology — the 4B can't invent types.
- Broad patterns (transfer over precision).

## Risks

- **4B enricher quality** — the whole bet. The spike de-risks it before the full build.
- **Sparse graph** at 24 train cases — lift visible mainly where train/eval topics overlap; the number on 14 will be noisy. This is a proof-of-concept of the *mechanism*, not a production metric.
- **Leakage drift** — if the enricher copies answer text into a Resolution label, a matched held-out question could regurgitate. The spike inspects for this; the rule is "fix pattern, not answer".
