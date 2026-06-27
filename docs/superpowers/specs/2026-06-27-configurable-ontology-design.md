# Configurable Ontology — Design Spec

**Status:** approved (2026-06-27)
**Sequence:** Task 1 of 2 (foundation). Task 2 = graph-hygiene pass, builds on this.

## Goal

Make every domain-specific graph rule come from the ontology, not from Rust literals. The
technical-support ontology (Symptom/Cause/Resolution, CAUSED_BY/RESOLVED_BY, MENTIONS) is
one overlay among many; the code must contain **no domain entity/relation literals**. This
closes the existing hardcode debt in `generalize/apply.rs` and supplies the `spine` that the
Task-2 hygiene pass will consume.

## Architecture

A new `[reasoning]` section in `<corpus>/.glossa/ontology.toml` declares the reasoning rules.
The `Ontology` type parses it and exposes accessors; `generalize` sources its closure rules,
mentions-anchor edge, and structural-type set from there instead of `const`/`default_rules()`.
When the section is absent, every accessor returns empty/defaults → existing behaviour is
byte-identical (back-compat).

## Tech Stack

Rust, `toml` + `serde` (already deps). No new dependencies.

## Global Constraints

- Pure-Rust, C-free; **no new dependencies**.
- **No domain literals in code** — no `"Symptom"`, `"CAUSED_BY"`, `"MENTIONS"` etc. in `.rs`
  files (test fixtures excepted). All sourced from the ontology.
- `ontology.toml` lives at **`<corpus-root>/.glossa/ontology.toml`** (the loader already reads
  this path). It is managed config, not client data.
- **Back-compat:** an ontology with no `[reasoning]` section yields empty spine, empty closure
  rules, default structural set, and `mentions = "MENTIONS"` — generalize output unchanged.
- No legacy-location fallback in code (the kb-test file has been physically relocated; the
  product deploy doc instructs `.glossa/`).
- TDD; File-First.

---

## Section 1 — File location & deploy

- The live eval file has already been moved to `kb-test/.glossa/ontology.toml`.
- Update the deploy comment **inside** `ontology.toml` (currently "copy this to
  `<corpus-root>/ontology.toml`") to say `<corpus-root>/.glossa/ontology.toml`.
- `Ontology::load_or_default` already reads `root/.glossa/ontology.toml` — **no path change**.

## Section 2 — `[reasoning]` schema

```toml
[reasoning]
# Hygiene (Task 2): the ordered relation sequence a node must lie on a COMPLETE instance of
# to survive the prune. Empty/absent → hygiene is a no-op.
spine    = ["CAUSED_BY", "RESOLVED_BY"]

# The anchor edge from a reasoning node to the structural layer (Section). Default "MENTIONS".
mentions = "MENTIONS"

# Transitive-closure composition rules: [a, b, result] = if x-a->y and y-b->z infer x-result->z.
# Empty/absent → no inferred closure edges.
closure  = [["CAUSED_BY", "RESOLVED_BY", "RESOLVED_BY"]]

# Optional override of the structural (never-reasoning) types. Default: the core four below.
structural = ["Document", "Section", "Term", "Topic"]
```

All four keys are optional. `closure` is an **explicit** list (1:1 with `closure::Rule`),
not derived from `spine` — predictable and fully general.

## Section 3 — `Ontology` API (src/graph/ontology.rs)

Add a `RawReasoning` deserialization struct and store it on `Ontology`:

```rust
#[derive(Debug, Deserialize, Default)]
struct RawReasoning {
    #[serde(default)] spine: Vec<String>,
    #[serde(default)] closure: Vec<Vec<String>>,   // each inner = [a, b, result]
    #[serde(default)] mentions: Option<String>,
    #[serde(default)] structural: Vec<String>,
}
```

Accessors:
- `pub fn spine(&self) -> &[String]` — ordered spine relations (empty if unset).
- `pub fn closure_rules(&self) -> Vec<(String, String, String)>` — well-formed `[a,b,result]`
  triples only; a malformed inner vec (len != 3) is skipped (it is config error, not data).
- `pub fn mentions(&self) -> &str` — the mentions edge type; `"MENTIONS"` when unset.
- `pub fn structural(&self) -> Vec<String>` — declared set, or the core four
  `["Document","Section","Term","Topic"]` when unset.

The core-node/core-edge constants stay as validation defaults; `structural()` defaults to the
core-node four.

## Section 4 — Wire generalize to the ontology (src/graph/generalize/apply.rs)

- Delete `const STRUCTURAL` and `fn default_rules()`.
- `Opts` gains `pub closure_rules: Vec<(String,String,String)>` and `pub structural: Vec<String>`;
  `mentions_type` stays. Add `Opts::from_ontology(ont: &Ontology, now: u64) -> Opts` that fills
  `closure_rules`, `mentions_type`, `structural` from the accessors (other tunables = current
  defaults). Keep `Opts::defaults(now)` for tests (empty closure, default structural,
  `"MENTIONS"`).
- In `generalize()`: build `closure::Rule`s from `opts.closure_rules`; use `opts.structural`
  in place of the `STRUCTURAL` const; `opts.mentions_type` is already used.
- The CLI (`kb graph generalize`, src/main.rs) loads the ontology via
  `Ontology::load_or_default(work)` and builds `Opts::from_ontology`.

## Section 5 — Error handling & back-compat

- Missing `[reasoning]`: all accessors return empty/defaults; `generalize` infers no closure
  edges and uses the default structural set → output identical to today.
- Malformed `closure` inner vec (len ≠ 3): skipped silently in `closure_rules()` (still parses).
- A malformed ontology file still falls back to `Ontology::default()` (unchanged behaviour).

## Section 6 — Testing

- `reasoning_section_parses`: spine/mentions/closure/structural read back correctly.
- `reasoning_absent_yields_defaults`: no section → empty spine, empty closure, `"MENTIONS"`,
  core-four structural.
- `closure_rules_skip_malformed`: a `["A","B"]` inner (len 2) is dropped; valid ones kept.
- `opts_from_ontology_sources_rules`: `Opts::from_ontology` carries closure/mentions/structural.
- `generalize_back_compat_no_reasoning`: a graph + an ontology without `[reasoning]` produces
  the same report as `Opts::defaults` did before (no inferred edges).
- `generalize_closure_from_ontology`: with a `[reasoning].closure` rule, the expected closure
  edge is inferred (mirrors the existing apply.rs closure test, but rules come from ontology).
- Deploy comment in `ontology.toml` updated to `.glossa/` (doc check).
