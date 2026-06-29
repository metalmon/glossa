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

Reference implementation for technical support: [`eval/ontology-support.toml`](../eval/ontology-support.toml).

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
kb graph node sym:abc123 ./my-corpus
```

MCP equivalents: `graph_stats`, `glossary`, `neighbors`, `read`.

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
