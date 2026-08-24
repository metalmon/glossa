# kbx synth — synthetic gold augmentation (MVP spec)

**Goal:** generate synthetic `(question, answer)` golds by re-packaging the EXISTING grounded
graph into new questions, so `kbx distil` can build a denser typed layer from more anchors. Inverse
of `chain_one_gold` (which is (Q,A)→chain; synth is graph→(Q,A)). Read-only on the graph; the only
output is a dataset file. Distil consumes it unchanged via `--gold`.

**Design lineage:** the agreed distil design (memory `kbx-distil-design`), Phase-1.5 "synthetic
augmentation (opt, count N)": seed from ANY grounded node of any type, traverse the schema-graph,
synthesize (Q, chain, A) + verify-gate (answerable from the chain alone, no leak).

## Command
`kbx synth [PATH] --count N [--out <file>] [--seed-type T] [--no-progress]`
- `PATH`: corpus root (kb-style resolution, same as other kbx verbs).
- `--count N`: number of synthetic golds to ATTEMPT (gate may drop some; report kept/dropped).
- `--out <file>`: dataset TOML to write (default `<workspace>/kbx/dataset.synthetic.toml`). Overwrite.
- `--seed-type T`: restrict seeds to node_type T (default: any grounded non-structural node —
  exclude Document/Section/Fact-only-structural; prefer the ontology's KNOWLEDGE types).

## Mechanic (per seed, one agentic pass — reuse the distil substrate)
1. **Seed pool:** query the graph for GROUNDED nodes (a node with an outgoing MENTIONS edge to a
   `<path>#n` chunk) of an ontology KNOWLEDGE type (types with `requires_grounding`, or all
   non-Document/Section when unset). Sample up to N (deterministic order by id; vary per index).
2. **Generate:** `run_agent_loop` with `lab.distil` (strong model), system = `schema_graph_block(ont)`
   + new `synth.md`. Give the model the seed node (id/type/label + its grounded source text) and the
   READ-ONLY reader tools from `glossa::tools` (glossary/reach/read/sql/search/grep — NO graph_upsert).
   Task: walk real ontology relations from the seed to form a chain, then emit ONE synthetic gold via
   a `propose_gold` tool: `{question, answer, chain_node_ids:[...], gate_ok:bool, gate_reason}`.
3. **Verify-gate (MVP = model self-gate + one leak check):**
   - self-gate: model sets `gate_ok=false` if the question is answerable without the chain, or the
     chain does not actually yield the answer.
   - leak check (cheap adversarial): one extra strong-model call with the QUESTION ALONE (no tools,
     no chain). If it returns the answer, mark leaked → DROP regardless of self-gate.
   - keep iff `gate_ok && !leaked`.
4. **Write:** kept golds appended as `[[case]]` (id=`synth-<i>`, question, answer) to `--out`,
   same schema `dataset.rs::Case` reads (id/question/answer). Report kept/dropped with reasons.

## Files
- Create `eval/src/synth/mod.rs` (+ `run.rs`, `gen.rs` if it grows) — the seed/generate/gate/write loop.
- Create `eval/templates/synth.md` — the generation prompt (behavior-guide style, NO corpus values).
- Modify `eval/src/bin/kbx.rs` — add `Cmd::Synth` + wire to `synth::run_synth`.
- Reuse: `backend::openai::{run_agent_loop, lmstudio_chat}`, `distil::schema_graph_block`,
  `glossa::tools` (reader registry), `dataset::Case` TOML shape, `GraphStore`, `Ontology`.

## Constraints
- English-only; NO corpus/gold values in code, tests, or `synth.md` (abstract placeholders only —
  memory `no-corpus-values-in-sop`).
- Read-only on the graph: synth NEVER calls graph_upsert. The only write is the `--out` dataset file.
- Deterministic-ish: no `Math.random`/wallclock in a way that breaks reproducibility; vary by seed index.
- Debug on the SANDBOX corpus `eval/.data/abac-ek-sws` only (kb-abac is mid-extract, do not touch).

## Test / debug plan (on abac-ek-sws)
- Unit: `propose_gold` parse → Case rows; leak-check drop path; seed pool excludes Document/Section.
- Live smoke: `kbx synth eval/.data/abac-ek-sws --count 3` → writes 1-3 gated golds; inspect that each
  question needs the chain (spot-check) and answers are grounded terminals. Then
  `kbx distil eval/.data/abac-ek-sws --gold .../dataset.synthetic.toml --mode kb` builds their chains.
