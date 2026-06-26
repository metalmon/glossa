# Semantic Reasoning-Graph Enricher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. **Task 0 (SPIKE) GATES the rest — do not start Tasks 1-3 until the spike is reviewed GO.**

**Goal:** a qwen3.5-4b enricher agent reverse-traces train cases into support-ontology reasoning edges; the answer agent reaches held-out answers by traversing them. See `docs/superpowers/specs/2026-06-26-semantic-enricher-design.md`.

## Global Constraints
- **Honesty:** graph stores routing/reasoning, NOT answer text; built from `kb-val/derived/train.json` ONLY; `eval-heldout.json` never touches the graph.
- Enricher is a **build-time agent** (its harness is ours — does NOT violate the answer-agent prompt/tools-only prod rule).
- `graph_upsert` validated against the strict support ontology (`kb-test/ontology.toml`, copied from `eval/ontology-support.toml`).
- Single source: the `graph_upsert` exec arm calls `glossa::graph::agent::apply_upsert` (no reimplementation).
- C-free binds glossa; kb-eval may use deps. STOP kb/kb-eval before any rebuild.

---

### Task 0 — SPIKE: minimal enricher on 4 cases, eyeball the graph (GO/NO-GO)

**Files:** `eval/src/backend/glossa_tools.rs` (graph_upsert arm), `eval/tensorzero/config/tensorzero.toml` + `tools/graph_upsert.json` (the `enrich` function + tool), `eval/src/main.rs` + a small `enrich` runner (`eval/src/enrich.rs`), `kb-test/ontology.toml` (deploy the overlay).

- [ ] **Step 1: Deploy the ontology** — copy `eval/ontology-support.toml` → `kb-test/ontology.toml`. (Confirm `Ontology::load_or_default(kb-test)` picks it up; `validate_node("Symptom")` ok, `validate_edge("RESOLVED_BY","Symptom","Resolution")` ok.)

- [ ] **Step 2: `graph_upsert` arm in `exec`** (`eval/src/backend/glossa_tools.rs`) — when `name == "graph_upsert"`, parse `args.nodes` / `args.edges` (serde into `Vec<glossa::graph::agent::NodeSpec>` / `Vec<EdgeSpec>`; the model supplies `id,node_type,label,aliases,source_path` per node and `from,to,edge_type,source_path` per edge), load `glossa::graph::ontology::Ontology::load_or_default(work_root)`, call `glossa::graph::agent::apply_upsert(graph, &ont, nodes, edges, now_secs)` (guard `graph` is `Some`), and return a text summary `"upserted N nodes, M edges"` (or the validation error string, so the model can self-correct). `exec` already receives `graph: Option<&GraphStore>` and a work root is available at the call site — thread the root in if needed.

- [ ] **Step 3: `enrich` TZ function + `graph_upsert` tool** — add `tools/graph_upsert.json` (schema: `{ nodes: array, edges: array }` with the NodeSpec/EdgeSpec fields) and `[tools.graph_upsert]` + `[functions.enrich]` (type chat, tools = `["search","read","grep","graph_upsert"]`, same `qwen` model, a variant with the SAME sampling preset). The `enrich` system prompt (new `eval/tensorzero/config/enrich/system.minijinja`):
  - "You build a reusable reasoning graph from a SOLVED support case. You are given a question and its KNOWN correct answer. Find, with search/read/grep, the document section(s) the answer comes from. Then record the GENERAL reasoning so a FUTURE similar question can be solved: upsert a broad `Symptom` node (the problem class — generalize, drop case-specific values), an optional `Cause`, a `Resolution` node (the fix PATTERN, NOT the verbatim answer), with edges `Symptom CAUSED_BY Cause`, `Symptom/Cause RESOLVED_BY Resolution`, and `MENTIONS` from them to each supporting Section. A Section node id is `\"<path>#<location-breadcrumb>\"`; get the path+location from your read/search results. Node ids: short stable slugs (e.g. `sym:profibus-link-loss`). source_path = the document path. NEVER put the literal answer text in a label — store the routing, not the answer. Allowed node types: Symptom, Cause, Resolution, Task. Allowed edges: CAUSED_BY, RESOLVED_BY, MENTIONS."
  - NOTE: `graph_upsert` is an MCP Editor tool; for the spike, hand-write `tools/graph_upsert.json` (mirror `mcp.rs` `GraphUpsertArgs`). Integrating it into `kb dump-tz-tools` is deferred to the full build.

- [ ] **Step 4: minimal `enrich` runner** (`eval/src/enrich.rs` + a `kb-eval enrich` subcommand): args `--train <json> --work <corpus> --limit <N> --tensorzero-endpoint … --tensorzero-function enrich`. For each of the first N train cases: open index+graph at `--work`, build the user prompt `"Question: {q}\nKnown correct answer: {gold}\nBuild the reasoning graph for this case."`, run the existing `run_episode` (tensorzero) with the `enrich` function and an `exec` that includes the graph_upsert arm. Reuse `MAX_ROUNDS`. Print per-case how many nodes/edges were upserted.

- [ ] **Step 5: run the spike on 4 reasoning-rich cases.** Build (`cargo build -p kb-eval --release`, kb-eval STOPPED first), restart gateway (`docker compose … restart gateway`) to load the `enrich` function, then run `kb-eval enrich --train kb-val/derived/train.json --work kb-test --limit 4 …` over cases like abac-3, ticket-3, ticket-11, ticket-7 (reorder train.json or filter so these are first). 

- [ ] **Step 6: dump + eyeball the graph.** Query `kb-test/.glossa` for the new nodes/edges (a tiny `kb` graph-dump or a ClickHouse-free read: use `kb.exe glossary`/`neighbors`, or add a one-off dump). Write the dump to `.superpowers/sdd/spike-graph.md` for the controller to inspect: are the Symptom/Resolution nodes BROAD and answer-free? do MENTIONS point at real sections? GO/NO-GO.

**This task ends with a written dump for human GO/NO-GO — do not proceed to Task 1.**

---

### Task 1 — full enricher over all 24 train cases (after GO)
Run the enricher over all of train.json; handle upsert errors per case; report graph size. (Detail after the spike confirms the prompt/types.)

### Task 2 — `neighbors`/`glossary` render entity nodes
Extend the shared `node_ref` so a `Symptom`/`Resolution`/… node renders by following its `MENTIONS` edges to `Section(path,#n)`. So `glossary(symptom-terms)` → the resolution + readable `#n` sections.

### Task 3 — ablation measurement on eval-heldout (14)
Run the answer eval on `eval-heldout.json` with the reasoning graph vs without (structural-only graph). Compare Judge; focus on the reasoning-heavy held-out cases.

## Self-Review
Spike is self-contained and gates the build. Honesty rule is in the enrich prompt (routing not answers) + the held-out split. apply_upsert is the single source for writes. The 4B-quality risk is exactly what the spike measures before investing in Tasks 1-3.
