# Graph Address Bridge — Design

**Date:** 2026-06-26
**Status:** Approved (direction confirmed)
**Branch:** `feat/graph-address-bridge`

## Goal

Make the structural graph usable by the agent by putting the graph tools in the **same address space as search/read/grep** — `(path, #n)`. Today `neighbors` takes and returns opaque `node_id`s (`"path#location"`) that the agent can neither construct from a search hit nor feed back into `read`, so the graph (NEXT/PREV/PARENT/CHILD/REFERENCES) is dead weight. After this change, from any hit the agent can ask for its graph neighbors and get back readable `path  #n` references it can directly `read`.

## Background

- Search/grep return hits as `[#n] path …`; `read(path, n)` opens chunk `n`. One address space: `(path, #n)`.
- The graph addresses by `node_id`: a `Section` node id is `format!("{path}#{location}")` (location = the heading breadcrumb `"A > B"` or `"p.N"`); a `Document` node id is the path. `neighbors(node_id, depth)` returns connected `node_id`s.
- The agent gets `(path, #n)` from search but `neighbors` wants `node_id` and emits `node_id`s — a different space. No bridge ⇒ the agent can't go "found this section → show its neighbors → read them".
- `GraphStore` already provides: `get_node(id) -> Option<Node>` (Node has `node_type`, `label`, `prov.source_path`), `outgoing(id) -> Vec<Edge>` (Edge has `to`, `edge_type`). `crate::graph::build::section_id(path, location)` builds the id.
- The index stores `path`, `location`, and `ord` per chunk, so `location ↔ ord` for a path is recoverable.

## Design

### Index helpers (`src/index/store.rs`)
Two small tantivy lookups mirroring `ord_body`'s `(path, ord)` query:
- `location_for_ord(&self, path: &str, n: u64) -> anyhow::Result<Option<String>>` — the `location` of the chunk at `(path, ord=n)`.
- `ord_for_location(&self, path: &str, location: &str) -> anyhow::Result<Option<u64>>` — the `ord` of the chunk at `(path, location)`.

These bridge `(path, #n) ↔ node_id` without string-parsing the id.

### `glossa::tools::neighbors` — input + output in `(path, #n)`
New signature: `neighbors(idx: &DocIndex, g: &GraphStore, path: &str, n: u64, depth: usize, trace) -> String`.
1. `location = idx.location_for_ord(path, n)?` → if `None`, return `"no chunk #{n} in {path}"`.
2. `node_id = build::section_id(path, location)`.
3. Collect edges to render: the section's own `outgoing(node_id)` (NEXT/PREV/PARENT/CHILD) **plus** the parent `Document`'s `outgoing(path)` filtered to `REFERENCES` (so cross-document links are visible from any section).
4. For each edge, resolve the target via `g.get_node(edge.to)`:
   - `Section` → `path2 = node.prov.source_path`, `loc2 = node.label`, `ord2 = idx.ord_for_location(path2, loc2)?`; render `"{EDGE_TYPE}  {path2}  #{ord2} · {loc2}"`.
   - `Document` → render `"{EDGE_TYPE}  {path2}  (document)"` (path2 = `node.prov.source_path` / `node.label`).
   - target node missing or ord unresolved → skip (don't emit a dangling line).
5. Empty → `"(no linked sections)"`. The agent reads any line via `read(path2, ord2)`.

`depth > 1` may traverse further (best-effort); v1 can keep `depth=1` behavior and just render direct neighbors.

### `glossa::tools::glossary` — render resolved nodes as `(path, #n)` refs
`glossary(name)` still resolves `name → node_ids`, but renders each as a readable ref via `get_node` + the same Section/Document formatting (so when term/semantic nodes land later, the agent gets directly-readable targets). Empty → `"(no matches)"`.

### Surfaces — keep MCP ≡ eval (single source)
- `src/mcp.rs`: `NeighborsArgs { path: String, n: u32, depth: Option<usize> }` (was `node_id`); the `neighbors` tool opens the index + graph and calls `crate::tools::neighbors`. `glossary` unchanged signature, new rendering.
- `eval/src/backend/glossa_tools.rs`: the `neighbors` arm reads `path` + `n` (+ optional `depth`) and calls the shared fn (it already has `idx` and `graph`).
- `eval/tensorzero/config/tools/neighbors.json`: params `{ path (string, required), n (integer, required), depth (integer, optional) }`; description updated.
- Update the MCP + TZ tool descriptions to the new contract.

## Testing

- `location_for_ord` / `ord_for_location` round-trip on a small indexed doc.
- `neighbors(path, n)` on a nested-heading doc (built via `index_dir`) returns lines containing the parent/next section as `path  #ord` with the right EDGE_TYPE, and a cross-doc `REFERENCES` line when one doc links another; `read`-ing a returned `#ord` works.
- `neighbors` on an out-of-range `n` → `"no chunk #n"`; on an isolated section → `"(no linked sections)"`.
- Parity test (existing pattern): MCP-rendered == eval-rendered for the same `(path, n)`.
- Full `cargo test -p glossa`; C-free gate.

## Global constraints

- **Pure-Rust, C-free** (`cargo tree -p glossa -i cc` empty): no new deps.
- **Single source of truth**: MCP and eval call the SAME `glossa::tools::{neighbors, glossary}`; render identical bytes.
- **No harness changes** — this is a tool-level change only (per the prod constraint: prompt/tools only).
- TDD.

## Out of scope (the payoff that follows)

- **Semantic / logical edges** (term nodes, СУГ→нефтепродукты, symptom→cause) — the multi-hop content layer that rides on this bridge. Separate feature.
- Prompt guidance teaching the agent the `search → neighbors → read` flow (a small prompt follow-up, once the bridge works).

## Risks

- `ord_for_location` assumes `(path, location)` is unique — it is (one chunk per heading-path). A heading with no body produces no chunk/node, so its node_id never appears as a neighbor target (already handled by "nearest existing ancestor" at build time).
- Extra index lookups per neighbor line (one `ord_for_location` each) — fine; `neighbors` is not hot and returns a handful of lines.
