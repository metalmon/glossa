# Build and maintain your knowledge graph

This is a task-oriented walkthrough of a knowledge graph's whole life: **create it**, **check its
health**, and **keep it correct as your documents change**. It covers both ways of driving glossa
— the **terminal** (`kb` / `kbx` commands) and an **agent** (the MCP tools) — and tells you which
surface to reach for.

For the data model and ontology reference, see [graph-and-ontology.md](graph-and-ontology.md);
for the full MCP tool surface, [mcp.md](mcp.md); for the reasoning-layer pipeline in depth,
[eval-and-training.md](eval-and-training.md).

> Sections marked **[Planned]** describe behaviour that is designed but not yet shipped. Every
> other command here is meant to run as written against the current binary.

---

## The graph has two layers

- **Structural layer** — `Document` and `Section` nodes that mirror your corpus. Built by
  **indexing**. Machine-derived and safe to rebuild from the files at any time.
- **Reasoning layer** — a thin skeleton on top: **grounded terminals** (the answers, tied to a
  source section) and **query-side** nodes (symptoms/tasks/causes that route a question to an
  answer). This layer is *authored* (by the builder or an agent), so glossa never deletes it
  silently — it only ever *flags* it when the source drifts.

You create the structural layer first, then the reasoning layer on top.

---

## Part 1 — Create the graph

### Step 1 — Index the corpus (structural layer)

Terminal:

```bash
kb index /path/to/corpus
```

Agent: the `index` tool does the same. This produces one `Document` node per file and one
`Section` node per chunk, plus the structural edges between them. Verify:

```bash
kb graph stats        # node/edge counts by type — you should see Document + Section
```

### Step 2 — Build the reasoning layer

The reasoning layer is built by the `kbx` toolkit. First scaffold a workspace and point it at a
model (edit `lab.toml` for your endpoint/model):

```bash
kbx init /path/to/corpus          # creates .glossa/kbx/ (lab.toml, prompts, dataset)
```

Then run the two phases:

```bash
kbx build   /path/to/corpus       # phase 1: harvest grounded terminals from each document
kbx reason  /path/to/corpus       # phase 2: synthesize the query-side reasoning layer
```

- `kbx build` walks each document and extracts the ontology's grounded terminal types (the types
  marked `requires_grounding`), grounding each to its source section.
- `kbx reason` seeds from every grounded terminal and works backward to build the query-side
  nodes and the typed edges that connect a question to that answer.

Verify the result:

```bash
kb graph stats                    # terminals + query-side nodes now present
kb graph doctor                   # should be clean on a freshly built graph
```

> Building the reasoning layer is a terminal (`kbx`) workflow. An agent cannot trigger the batch
> build over the MCP wire — but it can author the graph incrementally itself (next).

### Alternative — author the graph directly

Instead of (or alongside) the `kbx` pipeline you can write nodes and edges directly:

- **Agent:** the `graph_upsert` tool creates/updates nodes and edges. Ground a terminal by passing
  its source as a `path#n` token; leave the source empty for a query-side node.
- **Terminal:** `kb graph import <file.json> <path>` bulk-loads a graph export
  (`--mode replace` treats the file as the source of truth for the types it contains).

Use direct authoring for small, targeted additions; use `kbx build`/`reason` to (re)build a whole
corpus's reasoning layer.

---

## Part 2 — Check health with the doctor

```bash
kb graph doctor        # agent: the graph_doctor tool
```

The report lists four kinds of doubt:

| Doubt | What it means | What it asks of you |
|---|---|---|
| `ungrounded` | An answer node lost its link to a source section. | Re-ground it, or prune it if the source is gone. |
| `stale` | An answer's source **document was edited** since it was built. | **Re-ground** (re-build that doc) — never just delete. |
| `incomplete` | A node sits on no complete reasoning chain. | Finish the chain, or prune it. |
| `dangling` | A query-side node can no longer reach any live answer (its terminal went stale/ungrounded/deleted). | Fix/restore the terminal, or prune the orphaned branch. |

A clean graph reports `0` for all four. Read the doubts as a to-do list; the next part shows the
exact workflow for the common cause — your documents changing.

### Deep-clean a noisy reasoning layer (prune → merge)

A reasoning layer built by a small/local model accumulates noise: off-chain (`incomplete`) nodes,
orphaned (`dangling`) branches, and **near-duplicate** nodes — the model phrases the same cause or
resolution a little differently each pass. Before you rely on the graph (evaluation, serving),
deep-clean it in this order:

```bash
cp .glossa/graph.sqlite .glossa/graph.sqlite.bak        # 1. always back up first
kb graph doctor --prune-incomplete --prune-ungrounded --prune-dangling   # 2. prune the junk
kb graph generalize --merge                             # 3. collapse near-duplicates
kb graph doctor && kb graph stats                       # 4. verify: expect 0/0/0/0
```

**Prune before merge.** Pruning first means the dedup pass runs only over real, chain-complete
reasoning nodes, so it never risks merging a good node into one that was about to be pruned. (The
final graph is nearly identical either way — prune-first is simply the more principled default.)

The prune is **guarded**: if it would clear the whole reasoning layer or looks like an ontology
mismatch (zero live terminals), it refuses — pass `--force` to override (human-only, never over
MCP). So the ontology (`.glossa/ontology.toml`) **must be present**, or the doctor mis-classifies
every node as dangling. `generalize --merge` also recomputes the derived layer (similarity,
communities, centrality); those edges are read-excluded, so the recompute is harmless.

---

## Part 3 — When your documents change

This is the heart of maintenance. Three scenarios:

### You added a document

Index it, then extend the reasoning layer:

```bash
kb index /path/to/corpus          # picks up the new file
kbx build  /path/to/corpus        # harvests terminals from new/changed docs (incremental)
kbx reason /path/to/corpus        # synthesizes query-side for the new terminals
kb graph doctor                   # confirm clean
```

### You edited a document

Editing a source file makes the answers built from it **stale** — the extracted fact may no longer
match the text. The fix is to **re-ground**, not to delete (the document is still there, it just
changed).

```bash
kb index /path/to/corpus          # re-chunks the edited file, updates its signature
kb graph doctor                   # shows `stale` terminals (+ `dangling` chains to them)
```

Then re-ground the affected document:

```bash
kbx build  /path/to/corpus        # the incremental delta detects the changed doc,
                                  # drops its old reasoning nodes, re-extracts fresh ones
kbx reason /path/to/corpus        # re-synthesizes the query-side layer
kb graph doctor                   # back to clean
```

For a one-line edit you can instead fix the affected node directly with the agent's `graph_update`
(rename/retype) or `graph_upsert` (re-ground to the new section) — no full rebuild needed.

> Why not auto-fix on index? Because a re-index can be transient (a moved file, a quick edit you
> revert). glossa flags the drift and lets you decide, rather than silently discarding authored
> reasoning.

### You deleted a document

Removing a file orphans the answers built from it. Index to drop the structural layer, then clean
up the orphaned reasoning branch:

```bash
kb index /path/to/corpus          # drops the deleted file's Document/Section nodes
kb graph doctor                   # its terminals show `ungrounded`; their chains show `dangling`
kb graph doctor --prune-ungrounded   # removes the orphaned answer nodes
```

`--prune-ungrounded` clears the orphaned terminals; `--prune-dangling` then clears their now-
unreachable query-side chains in one pass. Combine the whole clean-up in one command:
`kb graph doctor --prune-ungrounded --prune-dangling`. Both prunes are guarded against a mass-wipe
(they refuse an ontology-mismatch or whole-layer wipe unless you pass `--force`).

---

## Part 4 — Edit the graph directly

| You want to… | Terminal | Agent (MCP) |
|---|---|---|
| Rename or retype one node | — | `graph_update` |
| Delete a specific node/edge | — | `graph_delete` |
| Wipe a whole node type (clean-slate a layer) | `kb graph prune -t <Type>` | — |
| Recompute the derived layer (closure, similarity, communities) | `kb graph generalize` | `graph_generalize` |
| Collapse near-duplicate nodes (destructive) | `kb graph generalize --merge` | — |

Two operations are deliberately **terminal-only** because they are high-impact:
`generalize --merge` (it *collapses* nodes and can merge ones that only look alike) and
`prune -t <Type>` (it wipes an entire layer). An agent has precise per-node tools instead
(`graph_update` / `graph_delete`).

> Clean-slate example: to rebuild just the reasoning layer, prune its authored types and re-run
> the pipeline — e.g. `kb graph prune -t <TerminalType> .` then `kbx build --force` then
> `kbx reason`. Keep `Document`/`Section` (the structural layer) intact.

---

## Part 5 — Terminal vs agent: who does what

| Task | Terminal (`kb`/`kbx`) | Agent (MCP) |
|---|---|---|
| Index / re-index | ✅ `kb index` | ✅ `index` |
| Build reasoning layer (batch) | ✅ `kbx build` + `kbx reason` | — (author incrementally via `graph_upsert`) |
| Author a node/edge | ✅ `kb graph import` (file) | ✅ `graph_upsert` |
| Diagnose health | ✅ `kb graph doctor` | ✅ `graph_doctor` |
| Prune doubtful nodes | ✅ doctor `--prune-*` | ✅ `graph_doctor` `prune_ungrounded`/`prune_incomplete`/`prune_dangling` |
| Clean-slate a type / merge dups | ✅ `prune -t` / `generalize --merge` | — (by design) |
| Inspect | ✅ `stats`/`ls`/`node`/`glossary`/`reach`/`sql` | ✅ `graph_stats`/`glossary`/`reach`/`sql`/… |

Rule of thumb: **an agent maintains its own authored nodes** (create, edit, delete, diagnose, and
— once shipped — prune what the doctor flags); **the blunt, corpus-wide operations stay in the
terminal** for a human to run deliberately.

---

## Roadmap (Planned)

These improvements are designed but not yet shipped; this manual will drop the **[Planned]** tags as
each ships:

- **Consistent MCP re-index** — the `index` tool's forced rebuild will re-run `generalize`
  automatically, matching `kb index --force`.

---

## See also

- [graph-and-ontology.md](graph-and-ontology.md) — data model, ontology file, doctor and operator
  reference.
- [mcp.md](mcp.md) — the full MCP tool surface and server roles.
- [eval-and-training.md](eval-and-training.md) — the `kbx` reasoning-layer pipeline in depth.
