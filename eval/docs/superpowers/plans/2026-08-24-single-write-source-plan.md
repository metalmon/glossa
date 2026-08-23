# Single write source — route build/distil through `ops::graph_upsert` — Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** ONE implementation of the graph-write tool. Today `graph_upsert` (write) is implemented twice — `glossa::graph::agent::apply_upsert` (id-based `NodeSpec`, used by eval build/distil) and `glossa::graph::ops::graph_upsert` (label-based `UpsertNode`, used by MCP) — and their behavior has repeatedly diverged (validity, grounding). Make `ops::graph_upsert` the single source of truth (exactly as the READER tools already share `glossa::tools`): route eval's build + distil writes through `ops::graph_upsert`, delete the duplicate `agent::apply_upsert`/`NodeSpec`/`EdgeSpec`, and fix grounding ONCE in `ops::graph_upsert` (auto-derive a MENTIONS edge from a node's `<path>#n` source_path; reject only when a requires_grounding node has neither a source_path chunk nor an explicit MENTIONS). This fixes the distil earns-keep grounding gap correctly, in one place, for every writer.

**Architecture:** `ops::graph_upsert(idx, g, ont, nodes: Vec<UpsertNode>, edges: Vec<UpsertEdge>, now) -> UpsertOutcome` is the canonical write (partial-apply: validate each item, write the good, drop the bad with an actionable reason). `ops::parse_upsert_payload(json)` parses a model `graph_upsert` call into `UpsertNode`/`UpsertEdge`. eval's `build::extract::extract_doc` and `distil::chain::chain_one_gold` currently parse via the eval-local `parse_and_validate_upsert` → `agent::apply_upsert` (all-or-nothing, id-based); they switch to `ops::parse_upsert_payload` → `ops::graph_upsert` (label-based). `UpsertNode` is label-based (id derived via `ops::id_for`), so the agent emits label-referenced upserts (what MCP already does).

**Spec:** none written; the diagnosis in-session is the spec: two write impls, behavior drift (validity hand-mirrored, grounding diverged); the fix is delegation to the one `ops` impl.

## Global Constraints

- English-only; no corpus/gold values in code/tests.
- MCP write behavior (`ops::graph_upsert`) must be UNCHANGED except for the new grounding auto-derive, which is additive (a requires_grounding node that already has an explicit MENTIONS is unaffected).
- Behavior-preserving for build/distil at the GRAPH level: after the reroute, `kbx build`/`kbx distil` produce a graph of the same shape (same node types/labels grounded to the same chunks) — validated by re-running on a small corpus. The `parse_and_validate_upsert` all-or-nothing→partial-apply change is an intentional robustness improvement (a bad node no longer discards a whole document's work); note it, keep the type-validation coverage.
- Build/test on C: via `CARGO_TARGET_DIR=C:/glossa-target` (glossa core touched → run `cargo test -p glossa` AND `-p kb-eval`). A parallel process may hold `C:/glossa-target/debug/kbx.exe`; if a build hits a file-lock, use `CARGO_TARGET_DIR=C:/glossa-target-sws`.
- Do NOT touch `glossa_tools.rs` or `tensorzero.rs`.

---

### Task 1: grounding auto-derive in `ops::graph_upsert` (the ONE place)

**Files:** Modify `src/graph/ops.rs` (the grounding step of `graph_upsert`); Test: `src/graph/ops.rs` `#[cfg(test)]`.

**Behavior:** in `graph_upsert`, when a node has a `<path>#n` chunk `source_path` and no explicit MENTIONS edge from it (in this batch or already in the graph), synthesize a MENTIONS edge `node -> <path>#n` from that provenance. Keep the existing requires_grounding enforcement as the fallback: a requires_grounding node that has NEITHER a chunk source_path NOR an explicit MENTIONS is still rejected/dropped with the existing actionable message. A node that already emits its own MENTIONS is unchanged (no duplicate).

- [ ] **Step 1: Failing tests** — (a) a node with a valid indexed `source_path=<doc>#n` and NO explicit MENTIONS edge results in a MENTIONS edge from it to `<doc>#n` after `graph_upsert`; (b) a node that DOES emit an explicit MENTIONS is not double-grounded; (c) a requires_grounding node with neither is still rejected (existing behavior preserved).
- [ ] **Step 2: run to verify (a)/(b) fail, (c) passes. Step 3: implement the auto-derive. Step 4: pass. Step 5: commit.**

### Task 2: route `build::extract::extract_doc` through `ops::graph_upsert`

**Files:** Modify `eval/src/build/extract.rs` (replace `parse_and_validate_upsert` + `agent::apply_upsert` with `ops::parse_upsert_payload` + `ops::graph_upsert`); keep the doc-scoped tool schema + the surfaced-id novelty tracking. Test: the existing extract tests migrate to the ops path.

**Interfaces:** the model's `graph_upsert` JSON → `ops::parse_upsert_payload` → `ops::graph_upsert(idx, g, ont, nodes, edges, now)`. Node references in edges become label-based (UpsertNode has no agent-assigned id). Surface upserted node ids from the `UpsertOutcome` for the unproductive-streak novelty tracker (the outcome reports written/merged ids).

- [ ] **Step 1: Failing test** — the extract-path unit that a `graph_upsert` with a valid typed/Fact node writes it (through ops), and an undeclared type is dropped with a reason (partial-apply), not the whole batch. **Step 2: fail. Step 3: reroute extract_doc + adapt the tests. Step 4: `cargo test -p kb-eval` green. Step 5: commit.**

### Task 3: route `distil::chain::chain_one_gold` through `ops::graph_upsert`

**Files:** Modify `eval/src/distil/chain.rs` (same reroute). Test: the distil non-Fact-type acceptance test migrates to the ops path.

- [ ] **Step 1: Failing test** — a declared non-`Fact` typed node with a `<doc>#n` source_path, upserted via the distil path, ends up GROUNDED (a MENTIONS edge exists) — proving the reroute fixes the distil grounding gap. **Step 2: fail. Step 3: reroute chain_one_gold. Step 4: green. Step 5: commit.**

### Task 4: delete the duplicate `agent::apply_upsert` / `NodeSpec` / `EdgeSpec`

**Files:** Modify `src/graph/agent.rs` (remove `apply_upsert`, `NodeSpec`, `EdgeSpec` and their tests, OR keep the CRUD helpers that are still used — verify no remaining non-test caller). Modify any remaining importers. Test: `cargo test -p glossa` + `-p kb-eval` green with the duplicate gone.

- [ ] **Step 1:** grep every caller of `agent::apply_upsert`/`NodeSpec`/`EdgeSpec`; confirm Tasks 2-3 removed the production ones (only tests remain). **Step 2:** delete them + their now-orphaned tests (the coverage moved to ops tests in T1-T3). **Step 3:** both suites green. **Step 4: commit.**

### Task 5: live validation (build + distil re-run, grounding + earns-keep)

- [ ] **Step 1:** Rebuild `kbx`. On a SCRATCH copy of a small domain corpus with a real ontology (the existing `E:\glossa\eval\.data\abac-ek` — customer data, scratch only, RU stays out of any committed file), re-run `kbx build --force` then `kbx distil --mode kb --limit 4`.
- [ ] **Step 2:** Confirm typed nodes are now GROUNDED: query the graph for MENTIONS edges from typed (non-Fact) nodes — must be > 0 (was 0 before). Confirm `kbx build`'s flat layer is unchanged in shape.
- [ ] **Step 3:** Re-run the earns-keep ablation (flat-only vs flat+typed, held-out) with the grounded typed layer + the universal `answer.md` reader (from `feat/distil-grounding`) and record the EM delta — does grounding move the typed layer from wash toward earning its keep? Report the number (no corpus values in the report).

## Self-Review

- Coverage: grounding-once (T1), build reroute (T2), distil reroute (T3), delete duplicate (T4), live validation (T5). Single-source achieved when T4 leaves exactly one write impl (`ops::graph_upsert`).
- Risk: the all-or-nothing→partial-apply change for build/distil (intentional; keep type-validation coverage). The id-based→label-based edge referencing (verify the agent's emitted edges resolve by label through ops's resolver).
- The universal `answer.md` reader lives on `feat/distil-grounding`; fold it in at merge time so the earns-keep re-run (T5) uses it.
