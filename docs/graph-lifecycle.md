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
| `ungrounded` | An answer node lost its link to a source section. | Re-ground it, prune it if the source is gone, or — if the document just moved/relabeled — `kb graph doctor --relink` (see below). |
| `stale` | An answer's source **document was edited** since it was built. | **Re-ground** (re-build that doc); `--prune-stale` only as a last resort. |
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
match the text. Prefer to **re-ground** (the document is still there, it just changed); delete only
what genuinely no longer applies.

```bash
kb index /path/to/corpus          # re-chunks the edited file, updates its signature
kb graph doctor                   # shows `stale` terminals (+ `dangling` chains to them)
```

Then refresh the affected document:

```bash
kbx build  /path/to/corpus        # incremental: re-extracts fresh terminals from the changed doc
kbx reason /path/to/corpus        # re-synthesizes the query-side layer
```

`kbx build` is **construct-only** — it never deletes, so a re-extract leaves the **old** stale
terminals in place beside the fresh ones (unless a fresh one lands on the same label and re-grounds
it). Clear the drifted leftovers: re-ground them node-by-node (below), or — once you've confirmed the
old content is gone for good — prune the stale bucket:

```bash
kb graph doctor --prune-stale     # delete the terminals whose source drifted and weren't refreshed
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

### You relabeled the corpus, or moved a document between folders

A reasoning node is *grounded* while its `MENTIONS` edge points at a structural node
(`Document`/`Section`) that still exists. Two changes leave a document's **content** untouched but
break that link anyway, because the target's **key** changes even though nothing was deleted:

- **Relabeling** — the same corpus, addressed a different way. As covered in
  [configuration.md § Corpus roots and document keys](configuration.md#corpus-roots-and-document-keys),
  discovery yields label-free keys and an explicit path/`--root` yields basename-labeled ones.
  Index once by discovery and once with `kb index /data/plc`, and every key gains (or loses) a
  `plc/` prefix — `manual.pdf` ↔ `plc/manual.pdf`.
- **Moving a file between folders** inside the corpus — `plc/manual.pdf` becomes
  `arch/2024/manual.pdf`. The filename and section are unchanged; only the leading folder path is.

Either way, a fresh `kb index` updates the *structural* layer to the new keys immediately — but the
*reasoning* layer's `MENTIONS` edges still point at the old ones, so their terminals suddenly report
`ungrounded` even though the source document is right there, just addressed under a different key:

```
$ kb graph doctor
ungrounded: 517
   -> plc  (498)
  plc -> arch/2024  (14)
  (!) non-destructive fix:  kb graph doctor --relink   (full list: --verbose)
  ambiguous: 2   (same filename in several folders — resolve by hand)
  real orphans: 3
  res:gone  [Resolution]  power-cycle sequence  plc/deleted.pdf#1  ungrounded
  ...
```

The `ungrounded: 517` line is the raw total — it does not by itself claim what that total is made
of. The lines under it break it down honestly: that grouped block — `old_prefix -> new_prefix (N)`
— replaces the old flat per-node listing for the part that's recoverable: of the 517 nodes
reported, 498 lost their label prefix, 14 followed a folder move (512 relinkable in total), 2 are
`ambiguous` (the same `filename#section` now exists under more than one live document — the doctor
won't guess, resolve those by hand), and 3 are `real orphans` (no live document matches at all —
their source is genuinely gone). Relinkable + ambiguous + real orphans always add up to the
`ungrounded:` total.

The cure is `kb graph doctor --relink`: it finds each affected node's document at its **current**
key — matched by filename + section against the graph's own live structural nodes, not against a
separate index or manifest — and re-points the `MENTIONS` edge (and the node's provenance) at it.
It's non-destructive: nothing is deleted, and it backs up `.glossa/graph.sqlite` (plus its `-wal`/
`-shm` siblings, when present) before writing — each run's backup is timestamped
(`graph.sqlite.pre-relink-<epoch-seconds>`), so a second `--relink` run never overwrites the
previous run's backup. Run it, then confirm:

```
$ kb graph doctor --relink
relinked: 512 edges (0 duplicate edges dropped)
$ kb graph doctor
ungrounded: 5
  res:gone  [Resolution]  power-cycle sequence  plc/deleted.pdf#1  ungrounded
  ...
```

`relinked:` reports two counts, not one: `edges` repointed onto their live target, and any
duplicate edges dropped instead — the collision case where two dead `MENTIONS` rows under the same
node both resolve to the same live target (e.g. a document relabeled more than once); the second
one is a now-redundant duplicate, so it's deleted rather than repointed. Both counts together equal
the number of relinkable nodes fixed.

Once the 512 relinkable nodes are fixed, nothing is relinkable anymore, so the report falls back to
a plain `ungrounded: 5` listing (the 3 real orphans plus the 2 still-ambiguous nodes) — the grouped
shift summary and the `real orphans:` split only appear while the report still has a relinkable
group to summarize.

**`--prune-ungrounded` refuses while relinkable nodes exist.** Those 512 nodes are relocated
documents, not orphans, so pruning them would silently destroy recoverable reasoning — the command
exits with an error pointing at `kb graph doctor --relink` instead of deleting. Run `--relink`
first; `--force` overrides the refusal only if you genuinely intend to delete relocatable nodes
without recovering them.

**Current limit: relink follows a relabel or a folder move, not a rename.** The match key is
filename + section, so as long as `manual.pdf#12` keeps that name somewhere under the corpus,
`--relink` finds it regardless of which folder or label it's under. If the file itself is renamed
(`manual.pdf` → `manual-v2.pdf`), its key stops matching by filename and `--relink` can't follow it
automatically — those nodes remain genuine `ungrounded` doubts until re-grounded by hand (or the
old filename is restored).

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

## See also

- [graph-and-ontology.md](graph-and-ontology.md) — data model, ontology file, doctor and operator
  reference.
- [mcp.md](mcp.md) — the full MCP tool surface and server roles.
- [eval-and-training.md](eval-and-training.md) — the `kbx` reasoning-layer pipeline in depth.
