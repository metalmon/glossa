# kbx distil — Phase 1 (gold-anchored backward chains + earns-keep) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ship the CORE of `kbx distil`: for each gold (Q,A), a strong model builds the grounded, ontology-typed reasoning chain BACKWARD from the answer, added alongside the flat `Fact`/`LEADS_TO` layer. Prove it earns its keep on a domain corpus (kb-abac) via a flat-only vs flat+typed eval. Synthetic augmentation and train/test-split rigor are DEFERRED to later phases.

**Architecture:** A new `src/distil/` module reuses `kbx build`'s agentic substrate (`backend::openai::run_agent_loop` + a `graph_upsert` tool → `parse_and_validate_upsert` → `graph::agent::apply_upsert`, now temporality-aware) but (a) permits the ontology's own `entity_types`/`relations` (NOT pinned to `Fact`), (b) is driven per-GOLD not per-doc, (c) uses the `lab.[distil]` strong endpoint, (d) injects the ontology's schema-graph + the gold into the prompt. A `Cmd::Distil` mirrors `Cmd::Build`.

**Tech Stack:** Rust; reuse `eval/src/build/{extract,finalize}.rs`, `src/graph/agent.rs`, `src/graph/ontology.rs`, `eval/src/checkpoint.rs`, `eval/src/workspace.rs`, `eval/src/lab.rs`, `eval/src/dataset_toml.rs`, `eval/src/bin/kbx.rs`, `eval/src/scaffold.rs`, `templates/distil.md`.

**Spec:** `eval/docs/superpowers/specs/2026-08-23-kbx-distil-design.md` (read it — the frame's invariants bind every task).

## Global Constraints

- English-only public surface; NO corpus/gold values in any prompt, comment, or test (kb-abac is customer data — spike/validation outputs live in scratch only, never committed).
- `endpoint` = full chat-completions URL, verbatim (base_url fix already in master).
- Build/test on C: via `CARGO_TARGET_DIR=C:/glossa-target`; run `cargo test -p glossa` AND `-p kb-eval` where a task touches glossa core.
- Ontology-general: NO domain type names (Symptom/Cause/…) hardcoded in code or the committed prompt — types come from `ontology.toml` at runtime. `validate_node`/`validate_edge`/`requires_grounding`/`requires_validity` are the only schema authority.
- Thin, NO per-edge confidence field, NO runtime layer-weight (spec invariants).
- distil ADDS typed nodes/edges; it never modifies the flat `Fact`/`LEADS_TO` layer.

---

### Task 1: `src/distil/` module + ontology schema-graph renderer

**Files:**
- Create: `eval/src/distil/mod.rs`
- Modify: `eval/src/lib.rs` (`pub mod distil;`)
- Test: `eval/src/distil/mod.rs` `#[cfg(test)]`

**Interfaces:**
- Produces: `pub fn schema_graph_block(ont: &glossa::graph::ontology::Ontology) -> String` — renders the ontology's `entity_types` and `relations` (each `from_type --RELATION--> to_type`, plus its `role`, and `requires_grounding`/`requires_validity` flags per type) as a compact, prompt-injectable text block. This is what makes the prompt ontology-general.
- Consumes: `Ontology` accessors (`entity_types`, `relations`, `requires_grounding`, `requires_validity`, `relation_role`).

- [ ] **Step 1: Failing test** — `schema_graph_block` on a tiny 2-type/1-relation ontology contains both type names, the relation as `A --REL--> B`, and marks a `requires_grounding` type. Assert those substrings.
- [ ] **Step 2: Run to verify fail** — `CARGO_TARGET_DIR=C:/glossa-target cargo test -p kb-eval schema_graph_block`. Expected: FAIL (undefined).
- [ ] **Step 3: Implement** `schema_graph_block` iterating the ontology's types + relations into the text block. No hardcoded domain names.
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit.**

---

### Task 2: `templates/distil.md` — ontology-parameterized backward-chain prompt

**Files:**
- Modify: `eval/templates/distil.md` (currently a stub; `scaffold.rs` already `include_str!`s it)
- Test: `tests/kbx_cli.rs` or a scaffold test — assert `kbx init` writes a non-stub `distil.md` containing the backward-chain markers.

**Interfaces:** consumed at runtime by Task 3 (`chain_one_gold`), which prepends `schema_graph_block` + the gold to it.

- [ ] **Step 1: Failing test** — after `kbx init`, the workspace `distil.md` contains the anchor phrases `BACKWARD` and `=== CHAIN ===` (the output marker Task 3 parses). Assert present.
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Author `distil.md`** — a behavior guide (no "only/never/must" beyond the honesty rules): you are handed the ontology's typed schema-graph and a SOLVED case (question + verified answer); build the grounded typed reasoning chain BACKWARD from the answer — (1) ground the answer as the terminal typed node (a `to_type` of some relation), quoting its source; (2) trace the reasoning path back toward the question, grounding EACH genuine intermediate node — do not stop at one hop just because the terminal was easy; mine the source for the real intermediate facts on the path; (3) reify the question as the entry node (a query archetype — leave it ungrounded unless the corpus describes it); (4) connect with ontology-legal relations only (respect from/to). Ground only what you can quote; if a node is not in the source, say so rather than invent it. Set `valid_from`/`valid_to` where the type requires validity. Emit via `graph_upsert`. ABSTRACT — NO corpus/gold values, no real names/dates.
- [ ] **Step 4: Run to verify pass. Step 5: Commit.**

---

### Task 3: `chain_one_gold` — the backward-chain engine (permits ontology types, `lab.[distil]`)

**Files:**
- Create/extend: `eval/src/distil/chain.rs` (+ `mod chain;` in distil/mod.rs)
- Modify: `eval/src/build/extract.rs` — extract `build_tools_schema` and `parse_and_validate_upsert` reuse (they are already `pub`); if `build_tools_schema` hardcodes only `Fact`-friendly text, add a variant/param so distil's `graph_upsert` schema advertises the ontology's node types.
- Test: `eval/src/distil/chain.rs` `#[cfg(test)]` — `parse_and_validate_upsert` accepts a typed (non-`Fact`) node under a strict ontology that declares it (reuse the existing test's tempdir/ontology pattern).

**Interfaces:**
- Produces: `pub fn chain_one_gold(paths: &KbxPaths, ont: &Ontology, lab: &LabConfig, distil_md: &str, q: &str, a: &str) -> anyhow::Result<DistilStats>` where `DistilStats { nodes: usize, edges: usize, grounded: usize }`.
- Consumes: `backend::openai::{run_agent_loop, lmstudio_chat}`, `build::extract::{parse_and_validate_upsert, build_tools_schema}`, `graph::agent::apply_upsert`, `lab.distil` (error clearly if `lab.distil` is None: "kbx distil needs a [distil] endpoint in lab.toml").

- [ ] **Step 1: Failing test** — `parse_and_validate_upsert` accepts a node whose `node_type` is a declared ontology `entity_type` (not `Fact`) under a strict ontology, and REJECTS an undeclared type. (Guards that distil is not `Fact`-pinned.)
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement `chain_one_gold`** — build the system prompt as `schema_graph_block(ont)` + `distil_md`; the user turn is the gold `(q, a)`; drive `run_agent_loop` with `lab.distil` endpoint + a `graph_upsert` tool schema that advertises the ontology's node types; on each `graph_upsert` call, `parse_and_validate_upsert` (ontology-permitting) then `apply_upsert` (temporality-aware). Corpus tools (`search`/`read`/`grep`) doc-unscoped so the model can find groundings anywhere. Return `DistilStats`.
- [ ] **Step 4: Run to verify pass. Step 5: Commit.**

---

### Task 4: `run_distil` pipeline — gold loop, resume, finalize

**Files:**
- Create: `eval/src/distil/run.rs` (+ `mod run;`)
- Reuse: `eval/src/checkpoint.rs` (per-gold done-marks), `eval/src/build/finalize.rs::finalize`, `eval/src/dataset_toml.rs::parse_dataset_toml`.
- Test: `eval/src/distil/run.rs` — a `should_skip`/checkpoint unit (a gold id already marked done is skipped under `--resume`).

**Interfaces:**
- Produces: `pub struct DistilArgs { pub gold: Option<PathBuf>, pub mode: String, pub limit: Option<usize>, pub force: bool, pub resume: bool, pub no_progress: bool }` and `pub fn run_distil(path: Option<PathBuf>, args: DistilArgs) -> anyhow::Result<()>`.
- `--mode` accepts `split|kb`; Phase 1 wires the flag and, for `split`, holds out a deterministic test fraction it must NOT ingest (record which ids were held out to a run file); the held-out eval itself is Task 6. Default `kb`.

- [ ] **Step 1: Failing test** — checkpoint skip: a gold id recorded done is skipped on `--resume`; a fresh id is not. (Pure, using a tempdir Checkpoint.)
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement `run_distil`** — resolve workspace + `lab.distil` + ontology + gold dataset (default `paths.dataset`, override `--gold`); ensure indexed (reuse build's index call); for each gold (respecting `--limit`, `--mode split` hold-out, `--resume` checkpoint), `chain_one_gold`, mark done; then `finalize`. Progress bar gated on TTY + `--no-progress`.
- [ ] **Step 4: Run to verify pass. Step 5: Commit.**

---

### Task 5: `Cmd::Distil` CLI wiring

**Files:**
- Modify: `eval/src/bin/kbx.rs` (add `Distil` variant mirroring `Build`, dispatch to `distil::run_distil`)
- Test: `tests/kbx_cli.rs` — `kbx distil --help` lists `--gold`, `--mode`, `--resume`.

- [ ] **Step 1: Failing test** — `kbx distil --help` contains `--gold` and `--mode`.
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Add the `Distil` clap variant** (path, `--gold`, `--mode` default "kb", `--limit`, `--force`, `--resume`, `--no-progress`); build `DistilArgs`; call `run_distil`.
- [ ] **Step 4: Run to verify pass. Step 5: Commit.**

---

### Task 6: Live validation + earns-keep measurement (kb-abac; no code)

- [ ] **Step 1:** Rebuild `kbx`. Copy kb-abac to a SCRATCH corpus (never mutate the customer base). Ensure it has the flat `Fact`/`LEADS_TO` layer (run `kbx build` if needed) and its `ontology-support.toml`. Configure `lab.[distil]` = luna, `[model]` = 4b.
- [ ] **Step 2:** Snapshot the flat-only `graph.sqlite` (the ablation baseline). Run `kbx distil <scratch> --limit 10 --no-progress`; confirm typed nodes/edges added, doctor clean, entry nodes ungrounded-by-config are not flagged.
- [ ] **Step 3:** Earns-keep: run `kbx eval` on the flat-only snapshot vs the flat+typed graph over the SAME held-out question slice; compare EM. Record the delta (scratch only — no corpus values in any committed file).
- [ ] **Step 4:** If flat+typed ≥ flat-only, distil earns its keep → proceed to Phase 2 (synthetic). If it washes, capture the failure mode (under-construction? wrong groundings? ablation mechanics?) for a design revisit. Either way, write the verdict to scratch and report.

## Self-Review notes

- Coverage: schema renderer (T1), prompt (T2), engine (T3), pipeline+resume (T4), CLI (T5), live earns-keep (T6). Spec invariants (thin/no-confidence/no-weight/ontology-general/additive) bind T2-T4.
- Type consistency: `chain_one_gold(paths, ont, lab, distil_md, q, a) -> DistilStats` fixed in T3, consumed by T4. `DistilArgs`/`run_distil` fixed in T4, consumed by T5.
- Placeholder scan: `distil.md` and all tests must be free of corpus/gold values (abstract only).
- Deferred (NOT this plan): synthetic augmentation + verify-gate; train/test-split held-out eval rigor as a productized arm; read-time typed-hide ablation filter (Phase 1 uses a snapshot instead); retiring ops.rs's redundant validity-authoring block (tracked follow-up from the temporality retrofit).
