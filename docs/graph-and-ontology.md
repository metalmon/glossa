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

Deploy a domain overlay at:

```
<corpus-root>/.glossa/ontology.toml
```

Reference implementations:
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

## Operator workflow

### 1. Deploy ontology

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
```

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
