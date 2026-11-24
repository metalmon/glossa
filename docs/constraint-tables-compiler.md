# Constraint tables compiler

> **Requires a `--features constraint` build.** `kb graph build` (and the mirrored MCP tool
> `graph_build`) only exist in a binary built with `cargo build --features constraint`; no release
> row enables that feature, so the pre-built `kb`/`kbi` binaries do not have it (`kb graph --help`
> on a default build lists no `build` subcommand). This is a deliberate, kept-but-off-by-default
> capability, not an oversight — build from source with the feature to follow this document.

`kb graph build` dematerializes agent `.csp` limit tables under `.glossa/notes/<document>/` into a constraint graph via `graph_upsert`.

The compiler is **ontology-driven**: it reads your corpus `ontology.toml` (entities, relations, constraint types, patterns, `[tables]`). It does **not** embed domain-specific type names. What it can emit is defined by a fixed **capability set** (this document) intersected with what your ontology declares.

Users assemble an ontology from the capabilities they need. If the ontology declares a shape the compiler cannot build from tables, or table data requires a missing capability, the compiler responds with an explicit, documented outcome — not silent degradation.

---

## Capabilities (what the compiler can emit)

Each capability maps to a `[patterns.*]` key in `ontology.toml`. The pattern documents intent for agents; the compiler checks that required relations exist before emitting.

| Capability | Pattern key | Required relations | Table signal | Status |
|------------|-------------|-------------------|--------------|--------|
| Independent enum | `independent_enum` | `CONSTRAINED_BY` (parameter → `Enum`) | Parameter column Y: values do not split by other parameter columns on the data | **v1 — implemented** |
| Conditional enum | `conditional_enum` | `CONSTRAINED_BY`, `IF_FIELD`, `IF_VALUE`, `HAS_CONSTRAINT` (→ `Enum`) | Y domain varies with trigger column X (FD on data) | **v1 — implemented** |
| Conditional range | `conditional_range` | above + inner `Range`, `HAS_MIN`, `HAS_MAX` | Per trigger group, Y is a numeric interval | **not implemented** (planned v2) |
| Formula | `formula_cross_field` | `CONSTRAINED_BY` (→ `Formula`), `HAS_EXPRESSION` (→ `Literal`) | Constant expression per field: `compile_hints` `strategy = "formula"` and/or dedicated expression column | **not implemented** (planned v2) |
| Regex | *(uses `Regex` constraint type)* | `CONSTRAINED_BY` (→ `Regex`), `HAS_PATTERN` (→ `Literal`) | `compile_hints` `strategy = "regex"` or pattern column | **not implemented** (planned v2) |
| Provenance | `provenance` | `MENTIONS` (core edge) | Optional provenance column (`[tables.provenance_column]` or per-hint) with section ref | **not implemented** (planned v2) |
| Combined constraints | `combined_constraints` | Multiple `CONSTRAINED_BY` on one parameter | Enum + Range on same field (intersection semantics in solver) | **not implemented** (planned v3) |

Only the two **v1** rows can actually emit a graph today (`check_independent`/`check_conditional_enum`
are the only functions in `src/tables/capabilities.rs` that can return `Ready`); every "not
implemented" row above unconditionally returns `compiler_gap`, even when its pattern, relations,
and constraint type are all correctly declared — the "v2"/"v3" label names a planned tier, not a
working-but-newer one. See the formula example below, which shows this directly.

Capabilities not listed here are **out of scope** for the table compiler (e.g. `Required` / `Forbidden` without tabular evidence). Phase-B agents may still add them via `graph_upsert`.

### Example: formula from a table (not implemented — shows the target shape)

`formula_cross_field` is **not implemented** (see Status above): even with every required
pattern/relation/constraint-type declared exactly as below, the compiler still reports
`compiler_gap (planned v2)` for the field and emits nothing. The example below is the shape it is
designed to emit once that lands, so an ontology author can wire the relations now and get a clean
`compiler_gap` (rather than an `ontology_gap`) confirming everything else is in place.

If the agent materializes:

```csv
Parameter	Expression
S	S = d * k
```

and the ontology declares `patterns.formula_cross_field`, `relations.CONSTRAINED_BY`, `relations.HAS_EXPRESSION`, and:

```toml
[tables.compile_hints."S"]
strategy = "formula"
expression_column = "Expression"
```

the compiler is designed to eventually emit `Field S ──CONSTRAINED_BY──► Formula ──HAS_EXPRESSION──► Literal("S = d * k")` — today it instead reports `capability formula_cross_field: compiler_gap (planned v2)` (see the scan output below) and emits no graph change for that field.

If `formula_cross_field` or `HAS_EXPRESSION` is missing from the ontology instead, the scan reports `ontology_gap` rather than `compiler_gap` — see [Reaction contract](#reaction-contract) below.

---

## Building an ontology from capabilities

1. Copy a base overlay (e.g. `eval/ontology-constraint.toml`).
2. Keep only `[patterns.*]` entries you need.
3. Declare every relation and entity type those patterns reference (`endpoint_types` must match).
4. Configure `[tables]`: delimiter, `skip_columns`, optional `parameter_columns`, `compile_hints`.
5. Run `kb graph build` on a smoke `.csp`; fix capability / ontology mismatches from the report.

The compiler resolves **types** from relations, not from hardcoded names:

- `parameter_entity` = `from` of `CONSTRAINED_BY`
- `literal_entity` = `to` of `IF_FIELD` / `HAS_EXPRESSION` / `HAS_PATTERN`
- `enum_constraint`, `conditional_constraint`, etc. = members of `CONSTRAINED_BY.to` ∩ `constraint_types`

Relation **names** (`CONSTRAINED_BY`, `IF_FIELD`, …) are the stable wire format between compiler, `constraint_adapter`, and overlays. Renaming requires a domain pack that keeps the same graph shape under different keys (future: explicit `[tables.compile.edge_map]` overrides).

---

## Reaction contract

Standard compiler behavior: **declare capabilities → validate ontology → compile or fail clearly**.

### 1. Capability scan (start of `tables_to_graph`)

For each capability, the compiler checks the ontology:

| Scan result | Meaning |
|-------------|---------|
| **ready** | Pattern declared (or implied by data/hint) and all required relations present with compatible endpoints |
| **ontology_gap** | Pattern declared in ontology but a required relation or constraint type is missing |
| **compiler_gap** | Data or hint requests a shape the compiler does not implement yet (a capability marked "not implemented" above) |

The scan is printed in `CompileReport` (first lines), e.g.:

```
capability independent_enum: ready
capability conditional_enum: ready
capability formula_cross_field: compiler_gap (planned v2)
capability provenance: ontology_gap (missing MENTIONS target types — N/A, MENTIONS is core)
```

### 2. Per-parameter column

| Outcome | When | Default behavior |
|---------|------|------------------|
| **compiled** | Shape inferred or hinted; capability ready | Nodes/edges emitted |
| **skipped** | `compile_hints.strategy = "skip"`, empty column, or `on_unsupported = "skip"` | Report line only; no graph change for that field |
| **error** | Hint or data requires unsupported capability, or `on_unsupported = "error"` | `tables_to_graph` returns `Err` with field name, capability, and fix hint |

### 3. `[tables.on_unsupported]` (ontology)

```toml
[tables]
on_unsupported = "error"   # default
# on_unsupported = "skip"  # eval-friendly: skip column, continue
# on_unsupported = "warn"  # emit report line, continue (no nodes for that field)
```

### 4. Error message shape

Errors are actionable:

```
tables compile: Field "S" requires capability formula_cross_field (strategy=formula)
  ontology: missing relation HAS_EXPRESSION (Formula → Literal)
  fix: add [relations.HAS_EXPRESSION] to ontology.toml or set strategy = "skip" for this column
```

### 5. `graph_upsert` rejection

If emitted graph violates strict ontology validation, the whole compile fails with the existing `graph_upsert` message (unchanged).

---

## `compile_hints` vocabulary

| `strategy` | Capability | Extra keys |
|------------|------------|------------|
| *(absent)* | Auto: `independent_enum` or `conditional_enum` from FD on data | `prefer_triggers = ["TriggerColumn"]` |
| `skip` | none | Leave field to phase-B agent |
| `enum` | Force `independent_enum` | — |
| `conditional` | Force `conditional_enum` | `prefer_triggers` |
| `range_if_numeric` | `conditional_range` when groups are intervals | v2 |
| `formula` | `formula_cross_field` | `expression_column = "…"` |
| `regex` | Regex constraint | `pattern_column = "…"` or inline pattern in hint (v2) |

---

## Multi-file `.csp` merge

`graph build` / `graph_build` reads **every** `*.csp` in the document notes dir. Two merge modes (same semantics as eval table coverage in `csp_column_values`):

| Mode | When | Result |
|------|------|--------|
| **Row shard** | All files share the same header row (normalized) | Data rows are **appended** — one relation table split across files (e.g. type 41 + type 42 size tables with columns `D`, `T`, `H`). |
| **Column union** | Headers differ | Schemas are **widened**; each file's rows map **sparsely** into the union columns. Row counts **need not match** across files. |

**Independent parameters** (D, grit, bond — flat enums): one narrow `.csp` per parameter is fine. The compiler collects allowed values **per column**; empty cells in other columns are ignored. A 31-row `D.csp` and a 25-row `F.csp` do not need aligned rows.

**Dependent parameters** (domain varies with a trigger): keep **multiple columns on the same rows** in one `.csp` (or shard files with **identical** headers). Conditional detection needs co-occurring values in one row.

Unit columns and metadata (e.g. `mm`) are not parameters unless listed in `[tables.parameter_columns]`; prefer omitting them from headers or add to `[tables.skip_columns]`.

---

## CLI

```bash
kb graph build <CORPUS> \
  --doc <document.pdf> \
  --tables-dir .glossa/notes/<document.pdf>
```

Default `--tables-dir` = the document's notes mirror (`<CORPUS>/.glossa/notes/<document>/`, full indexed path with extension); the compiler reads every `*.csp` in it.

See [agent-workspace-contract.md](agent-workspace-contract.md) for notebook tools (`note`, `ls`, `del`; note content is read with `read`).

Post-step in the constraint eval harness runs the same function after a phase-A episode.

MCP / eval agent tool: **`graph_build(doc)`** — same compile, compiler-style OK/FAILED output; use on SOP step 3 to fix `.csp` before `done`. Like the CLI, this tool is only advertised on a `--features constraint` build; a default build's `tools/list` does not include it.

---

## Implementation map (code)

| Component | Role |
|-----------|------|
| `src/tables/capabilities.rs` | Capability registry, ontology scan, required relations per pattern |
| `src/tables/wiring.rs` | Resolve entity/edge types from `Ontology::endpoint_types` |
| `src/tables/csp.rs` | Parse `.csp`, multi-file merge (row shard + column union) |
| `src/tables/compile.rs` | FD algorithm, emit via wiring |
| `src/tables/coverage.rs` | Table coverage accounting |
| `src/tables/report.rs` | `CompileReport` shape (per-capability scan lines, per-column outcomes) |
| `src/graph/ontology.rs` | `[tables]`, `compile_hints`, `on_unsupported` |
| This doc | User-facing contract; update when capabilities ship |

When adding a capability: extend the registry, implement emitter, add tests against `eval/ontology-constraint.toml`, update the Status column in the table above.
