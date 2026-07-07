# Constraint tables compiler

`kb graph build` dematerializes agent `.csp` limit tables under `.glossa/notes/<document>/` into a constraint graph via `graph_upsert`.

The compiler is **ontology-driven**: it reads your corpus `ontology.toml` (entities, relations, constraint types, patterns, `[tables]`). It does **not** embed domain-specific type names. What it can emit is defined by a fixed **capability set** (this document) intersected with what your ontology declares.

Users assemble an ontology from the capabilities they need. If the ontology declares a shape the compiler cannot build from tables, or table data requires a missing capability, the compiler responds with an explicit, documented outcome — not silent degradation.

---

## Capabilities (what the compiler can emit)

Each capability maps to a `[patterns.*]` key in `ontology.toml`. The pattern documents intent for agents; the compiler checks that required relations exist before emitting.

| Capability | Pattern key | Required relations | Table signal | Status |
|------------|-------------|-------------------|--------------|--------|
| Independent enum | `independent_enum` | `CONSTRAINED_BY` (parameter → `Enum`) | Parameter column Y: values do not split by other parameter columns on the data | **v1** |
| Conditional enum | `conditional_enum` | `CONSTRAINED_BY`, `IF_FIELD`, `IF_VALUE`, `HAS_CONSTRAINT` (→ `Enum`) | Y domain varies with trigger column X (FD on data) | **v1** |
| Conditional range | `conditional_range` | above + inner `Range`, `HAS_MIN`, `HAS_MAX` | Per trigger group, Y is a numeric interval | **v2** |
| Formula | `formula_cross_field` | `CONSTRAINED_BY` (→ `Formula`), `HAS_EXPRESSION` (→ `Literal`) | Constant expression per field: `compile_hints` `strategy = "formula"` and/or dedicated expression column | **v2** |
| Regex | *(uses `Regex` constraint type)* | `CONSTRAINED_BY` (→ `Regex`), `HAS_PATTERN` (→ `Literal`) | `compile_hints` `strategy = "regex"` or pattern column | **v2** |
| Provenance | `provenance` | `MENTIONS` (core edge) | Optional provenance column (`[tables.provenance_column]` or per-hint) with section ref | **v2** |
| Combined constraints | `combined_constraints` | Multiple `CONSTRAINED_BY` on one parameter | Enum + Range on same field (intersection semantics in solver) | **v3** |

Capabilities not listed here are **out of scope** for the table compiler (e.g. `Required` / `Forbidden` without tabular evidence). Phase-B agents may still add them via `graph_upsert`.

### Example: formula from a table

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

the compiler emits `Field S ──CONSTRAINED_BY──► Formula ──HAS_EXPRESSION──► Literal("S = d * k")`.

If `formula_cross_field` or `HAS_EXPRESSION` is missing from the ontology, see [Reaction contract](#reaction-contract) below.

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
| **compiler_gap** | Data or hint requests a shape the compiler does not implement yet (capability status not v1/v2) |

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

## CLI

```bash
kb graph build <CORPUS> \
  --doc <document.pdf> \
  --tables-dir .glossa/notes/<document.pdf>
```

Default `--tables-dir` = the document's notes mirror (`<CORPUS>/.glossa/notes/<document>/`, full indexed path with extension); the compiler reads every `*.csp` in it.

See [agent-workspace-contract.md](agent-workspace-contract.md) for notebook tools (`note`, `ls`, `cat`, `sed`, `del`).

Post-step in the constraint eval harness runs the same function after a phase-A episode.

---

## Implementation map (code)

| Component | Role |
|-----------|------|
| `src/tables/capabilities.rs` | Capability registry, ontology scan, required relations per pattern |
| `src/tables/wiring.rs` | Resolve entity/edge types from `Ontology::endpoint_types` |
| `src/tables/compile.rs` | FD algorithm, emit via wiring |
| `src/graph/ontology.rs` | `[tables]`, `compile_hints`, `on_unsupported` |
| This doc | User-facing contract; update when capabilities ship |

When adding a capability: extend the registry, implement emitter, add tests against `eval/ontology-constraint.toml`, update the Status column in the table above.
