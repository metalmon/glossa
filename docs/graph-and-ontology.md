# Graph and ontology

glossa separates **structural** graph data (auto-built from documents) from **reasoning** graph data (domain knowledge written by agents). The ontology file defines what reasoning types and relations are allowed.

## Layers

| Layer | Source | Examples |
|-------|--------|----------|
| Structural | Index pipeline | Document, Section, CONTAINS, MENTIONS |
| Reasoning | Agent `graph_upsert` | Symptom, Cause, Resolution, CAUSED_BY, RESOLVED_BY |
| Derived | `graph generalize` | SIMILAR, closure edges, community metadata |

Structural nodes bypass ontology validation. Reasoning nodes and edges are validated against `.glossa/ontology.toml`.

## Ontology file

The overlay lives at:

```
<corpus-root>/.glossa/ontology.toml
```

The quickest way to get one is a **preset** baked into the binary — pick it at
first index and glossa writes the file for you:

```bash
kb index ./my-corpus --ontology compliance   # see: kb ontology list
```

See [Ontology presets](ontology-presets.md) for the catalog and the
`kb ontology list/show/init/suggest` commands. To hand-write one instead, start
from a reference implementation:
- Technical-support knowledge base: [`eval/ontology-support.toml`](../eval/ontology-support.toml).
- Constraint/CSP validation (regulatory and standards documents): [`eval/ontology-constraint.toml`](../eval/ontology-constraint.toml).

### Entity types (support example)

| Type | `id_prefix` | Role |
|------|-------------|------|
| Symptom | `sym` | Observed problem (keep broad) |
| Cause | `cau` | Root cause |
| Resolution | `res` | Fix or procedure |
| Task | `tsk` | How-to intent |
| Parameter, Product, Module, … | (none) | Domain nouns |

### Relations (reasoning spine)

```
Symptom --CAUSED_BY--> Cause
Symptom/Cause/Task --RESOLVED_BY--> Resolution
Resolution --SETS--> Parameter
```

`CAUSED_BY` accepts **Symptom → Cause** only (not Task → Cause). Invalid endpoint pairs are rejected at upsert time.

### Strict validation

```toml
[validation]
strict = true
```

Unknown node or edge types are rejected — the enricher cannot invent types.

### Reasoning rules

The same file declares **spines** (valid chain shapes for hygiene) and **closure** rules (transitive edge composition). Consumed by `graph generalize` and prune logic — domain rules stay in TOML, not Rust.

### Grounding

Reasoning nodes should trace to a source document; the built-in `MENTIONS` edge grounds a node to a `Section`. Mark a node type as requiring grounding:

```toml
[entities.Resolution]
requires_grounding = true
```

Grounding is **transitive** — on a spine only the grounding node (e.g. `Resolution`) needs a direct `MENTIONS`; the others ground through it — so the flag goes on that one type. It is enforced two ways:

- **Write time** — `graph_upsert` rejects the whole call if a `requires_grounding` node has no `MENTIONS` (in the batch or already in the graph), naming the node and the edge to add. Document-agnostic: the `MENTIONS` may cite any indexed document, not only the node's `source_path`.
- **Standing** — `graph generalize` reports required nodes that lost a *live* `MENTIONS` (none, or one whose target section was removed) as `ungrounded=<n>` plus a re-ground list. This is a **separate bucket** from off-spine `prune_candidates`: re-ground them, or delete with `--prune-ungrounded` (destructive, CLI only).

### Valid-time

A reasoning node can carry a **validity interval** — the span of time the fact
it represents actually holds in the world. `graph_upsert` accepts two optional
fields per node:

```json
{"node_type": "Requirement", "label": "…", "source_path": "…",
 "valid_from": "2024-01-01", "valid_to": "2024-12-31"}
```

`valid_from`/`valid_to` accept any ISO-8601 granularity (a bare year, `YYYY-MM`,
`YYYY-MM-DD`, or a full RFC3339 instant); the raw string is kept alongside the
normalized bound so the original expression is never lost. `valid_to` is
optional — omitting it leaves the interval open-ended (still in effect).
Leaving both out on an update doesn't touch an existing interval; a node with
no validity at all is **timeless** and always considered current.

**`requires_validity`** is the exact analog of [`requires_grounding`](#grounding),
declared the same way in the ontology:

```toml
[entities.Requirement]
requires_validity = true
```

`graph_upsert` rejects a node of that type with no `valid_from` — supplied in
the same batch or already on record — naming the node and the fields to add.
Presets that model time-bound facts (`hr-compliance`, `data-privacy`,
`contract`, `reg-change`, `certification`) mark their timed type this way; see
[Ontology presets](ontology-presets.md).

**Reading as of a date.** `kb graph glossary/ls/near/node/dump` take `--as-of
<date>` on the CLI, and `glossary`/`neighbors`/`related` take an `as_of` arg
over MCP. A node outside its validity interval on that date is hidden from the
result; a timeless node is always shown. `neighbors`/`related` apply the filter
to the *surrounding* graph (edge endpoints, SIMILAR links, community
siblings) — the explicitly named anchor node is returned regardless of its own
window, so an expired node is still reachable by id, just with a filtered
neighbourhood. `kb graph node` / MCP `read` additionally report the node's own
**status** — current, future, expired, or superseded (via an incoming
`SUPERSEDES` edge) — so a caller can tell *why* it's outside the window.

**World time ≠ document time.** `valid_from`/`valid_to` describe when the fact
held *in the world*; provenance (`source_path`, the `MENTIONS` grounding,
`created_at`) describes when it was *recorded*. A requirement can be entered
into the graph today with a `valid_from` years in the past, or scheduled with a
`valid_from` in the future — the two clocks are independent, and only the
former is queried by `--as-of`.

## Operator workflow

### 1. Deploy ontology

Pick a [preset](ontology-presets.md) — it writes the overlay and indexes:

```bash
kb index ./my-corpus --ontology support   # kb ontology list for the catalog
```

Or hand-deploy a reference overlay:

```bash
mkdir -p ./my-corpus/.glossa
cp eval/ontology-support.toml ./my-corpus/.glossa/ontology.toml
kb index ./my-corpus
```

### 2. Enrich (batch)

The `kb-train enrich` command reverse-traces solved cases into reasoning edges. See [eval-and-training.md](eval-and-training.md).

### 3. Generalize

Recompute SIMILAR links, communities, and closure:

```bash
kb graph generalize ./my-corpus
```

Or via MCP: `graph_generalize`. Non-destructive by default. Destructive options on CLI only:

```bash
kb graph generalize ./my-corpus --merge              # collapse near-duplicates
kb graph generalize ./my-corpus --prune-incomplete   # remove off-spine nodes
kb graph generalize ./my-corpus --prune-ungrounded   # remove required nodes that lost MENTIONS grounding
```

The non-destructive run still **reports** `ungrounded=<n>` and lists those nodes (see [Grounding](#grounding)) so an agent can re-ground them.

### 4. Inspect

```bash
kb graph stats ./my-corpus
kb graph glossary "connection loss" ./my-corpus
kb graph near sym:abc123 ./my-corpus
kb graph ls -t Symptom ./my-corpus
kb graph node sym:abc123 ./my-corpus                  # one node: type, label, provenance, outgoing edges
kb graph path sym:abc123 res:def456 ./my-corpus       # bounded path between two node ids
```

MCP equivalents: `graph_stats`, `glossary`, `neighbors`, `read`.

### 5. Export, import, prune

```bash
kb graph dump ./my-corpus -f json          # dump all nodes + outgoing edges (text/json/dot/graphml/html)
kb graph dump ./my-corpus -f html > kb.html # self-contained offline graph explorer (see below)
kb graph import graph.json ./my-corpus     # replace the semantic layer from a graph file (file = source of truth)
kb graph prune ./my-corpus -t Symptom      # delete all nodes of a type (and edges touching them)
```

**`-f html` — offline interactive explorer.** Emits one self-contained HTML file
(the graph library and data are embedded; nothing is fetched at runtime, so it
works offline and can be handed to anyone). Open it in a browser: a glossary
search returns matching nodes, and selecting one shows a focused local view —
that node in the centre with its typed relations, similar nodes, and `MENTIONS`
sources — which you traverse by clicking. Node colours and the legend are
derived from the data, so it renders any graph. Light/dark theme follows the
system with a manual toggle; the UI is English, switching to Russian on a `ru`
locale. Mobile-friendly (touch pan/zoom, responsive layout).

## Constraint workflow (feature-gated)

Requires glossa built with `--features constraint`. Deploy the constraint ontology:

```bash
mkdir -p ./my-corpus/.glossa
cp eval/ontology-constraint.toml ./my-corpus/.glossa/ontology.toml
kb index ./my-corpus
```

Then model requirement constraints via `graph_upsert`:

1. **Create Field nodes** for each constrained parameter.
2. **Create constraint-type nodes** (`Range`, `Regex`, `Enum`, `Required`, `Forbidden`, `Formula`).
3. **Link** `Field --CONSTRAINED_BY--> constraint-node`.
4. **Attach parameters** via edges like `HAS_MIN`, `HAS_MAX`, `HAS_PATTERN`, `HAS_LITERAL` to `Literal` nodes.

Solve via the `constraint_solve` MCP tool in three modes:

| Mode | Purpose |
|------|---------|
| `validate` | Check concrete field values against constraints |
| `infer` | Compute allowed domains for each field |
| `check` | Detect inconsistencies in the constraint graph itself |

## MCP graph editing

Editor/full profiles expose:

- **`graph_upsert`** — batch create nodes and edges; see [mcp.md § graph_upsert](mcp.md#graph_upsert-response)
- **`graph_update`** — rename or retype without losing edges
- **`graph_delete`** — remove mistaken nodes or relations

Always call **`glossary`** before creating nodes to reuse existing ids.

## Id conventions

With `id_prefix` in ontology, stable ids are derived from type + normalized label (e.g. `sym:poterya-svyazi`). Reference nodes in edges by id or label; upsert resolves labels to ids.

## CLI ↔ MCP parity

| CLI | MCP tool |
|-----|----------|
| `kb graph glossary Q` | `glossary` |
| `kb graph near ID` | `neighbors` |
| `kb graph generalize` | `graph_generalize` |
| `kb graph stats` | `graph_stats` |
| `kb index` | `index` |

Shared implementation: [`src/graph/ops.rs`](../src/graph/ops.rs).

## Further reading

- [architecture.md](architecture.md) — derived layer algorithms
- [eval-and-training.md](eval-and-training.md) — enrich pipeline
- [mcp.md](mcp.md) — tool profiles and upsert responses
