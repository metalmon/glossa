# Graph Address Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Put `neighbors`/`glossary` in the agent's `(path, #n)` address space so the structural graph is navigable from a search hit and the results are directly `read`-able.

**Architecture:** Two index lookups bridge `(path, #n) ↔ node_id`. `glossa::tools::neighbors` takes `(path, n)`, walks the section's edges (+ the parent doc's REFERENCES), and renders each neighbor as `EDGE_TYPE  path  #ord · label`. `glossary` renders resolved nodes the same way. MCP + eval call the shared functions (single source).

**Tech Stack:** Rust, `glossa::graph::store::GraphStore` (`get_node`, `outgoing`), `glossa::graph::build::section_id`, tantivy index lookups.

## Global Constraints

- **Pure-Rust, C-free** (`cargo tree -p glossa -i cc` empty): no new deps.
- **Single source of truth:** MCP (`src/mcp.rs`) and eval (`eval/src/backend/glossa_tools.rs`) call the SAME `glossa::tools::{neighbors, glossary}` and render identical bytes.
- **Tool-level only** (prod constraint: prompt/tools, never harness).
- TDD. Frequent commits.

---

### Task 1: index `location ↔ ord` helpers + `glossa::tools::neighbors`/`glossary` in `(path,#n)` space

**Files:**
- Modify: `src/index/store.rs` (two helpers + tests)
- Modify: `src/tools.rs` (`neighbors` new signature + `glossary` rendering + tests)

**Interfaces:**
- Produces: `DocIndex::location_for_ord(path, n) -> Result<Option<String>>`; `DocIndex::ord_for_location(path, location) -> Result<Option<u64>>`; `glossa::tools::neighbors(idx: &DocIndex, g: &GraphStore, path: &str, n: u64, depth: usize, trace: &TraceLog) -> String`; `glossary(g, name, trace)` unchanged signature, new rendering (now also needs `idx` — see Step 4).
- Consumes: `GraphStore::{get_node, outgoing}`, `Node{node_type,label,prov.source_path}`, `Edge{to,edge_type}`, `build::section_id(path, location)`.

- [ ] **Step 1: Add the two index helpers to `src/index/store.rs`** (mirror `ord_body`'s `(path, ord)` BooleanQuery, but return the other field). Implement `location_for_ord(path, n)` (query path+ord → return the `location` stored field) and `ord_for_location(path, location)` (query path+location → return the `ord` stored field). Use the existing `TopDocs::with_limit(1)` pattern; read the field via `d.get_first(self.fields.location)` / `self.fields.ord`.

- [ ] **Step 2: Test the helpers** in `src/index/store.rs`:
```rust
#[test]
fn location_ord_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let i = DocIndex::open_or_create(dir.path()).unwrap();
    i.write_chunks(&[
        crate::model::Chunk { doc_path: "d.md".into(), location: "A > B".into(), file_type: "md".into(), text: "x".into() },
    ]).unwrap();
    let n = i.ord_for_location("d.md", "A > B").unwrap().unwrap();
    assert_eq!(i.location_for_ord("d.md", n).unwrap().as_deref(), Some("A > B"));
    assert_eq!(i.ord_for_location("d.md", "missing").unwrap(), None);
}
```
Run `cargo test -p glossa index::store` → PASS.

- [ ] **Step 3: Rewrite `glossa::tools::neighbors`** to the `(path, n)` contract:
```rust
pub fn neighbors(idx: &DocIndex, g: &crate::graph::store::GraphStore, path: &str, n: u64, depth: usize, trace: &TraceLog) -> String {
    let _ = depth; // v1: direct neighbors only
    let location = match idx.location_for_ord(path, n) {
        Ok(Some(l)) => l,
        Ok(None) => return format!("no chunk #{n} in {path}"),
        Err(e) => return format!("neighbors error: {e}"),
    };
    let sec_id = crate::graph::build::section_id(path, &location);
    // The section's own edges, plus the parent document's REFERENCES (cross-doc links visible from any section).
    let mut edges = g.outgoing(&sec_id).unwrap_or_default();
    if let Ok(doc_edges) = g.outgoing(path) {
        edges.extend(doc_edges.into_iter().filter(|e| e.edge_type == "REFERENCES"));
    }
    let mut lines = Vec::new();
    for e in &edges {
        let Ok(Some(node)) = g.get_node(&e.to) else { continue };
        let tp = node.prov.source_path.as_str();
        if node.node_type == "Section" {
            if let Ok(Some(ord)) = idx.ord_for_location(tp, &node.label) {
                lines.push(format!("{}  {}  #{} · {}", e.edge_type, tp, ord, node.label));
            }
        } else {
            lines.push(format!("{}  {}  (document)", e.edge_type, tp));
        }
    }
    trace.log("neighbors", json!({"path": path, "n": n}), json!({"links": lines.len()}));
    if lines.is_empty() { "(no linked sections)".to_string() } else { lines.join("\n") }
}
```
(Confirm `Node.prov` is the field name for `Provenance` and `Provenance.source_path` exists — adjust if the accessor differs.)

- [ ] **Step 4: Update `glossary` to render resolved nodes as `(path,#n)` refs.** Change to `glossary(idx: &DocIndex, g: &GraphStore, name: &str, trace) -> String`: for each id from `g.resolve(name)`, `get_node` and render via the SAME Section/Document formatting as neighbors (Section → `path #ord · label`, Document → `path (document)`, unknown id → the raw id). Empty → `"(no matches)"`. Factor the per-node rendering into a shared private helper used by both `neighbors` and `glossary`.

- [ ] **Step 5: Tests in `src/tools.rs`** — build a graph via `index_dir` on a nested-heading doc + a cross-ref doc (reuse the patterns from the graph-enrichment tests), then assert `neighbors(&idx, &g, path, n, 1, &t)` returns lines with the parent/next section as `path  #ord` and the right EDGE_TYPE, and that an out-of-range `n` → `"no chunk #n"`, isolated → `"(no linked sections)"`. Verify a returned `#ord` actually `read`s.

- [ ] **Step 6: Build + C-free gate.** `cargo test -p glossa tools:: index::store` , `cargo tree -p glossa -i cc` (empty).

- [ ] **Step 7: Commit** `git commit -m "feat(tools): neighbors/glossary speak the (path,#n) address space"`

---

### Task 2: wire MCP + eval + TZ config to the new contract (parity)

**Files:**
- Modify: `src/mcp.rs` (`NeighborsArgs`, the `neighbors`/`glossary` tools, descriptions)
- Modify: `eval/src/backend/glossa_tools.rs` (`neighbors`/`glossary` arms)
- Modify: `eval/tensorzero/config/tools/neighbors.json` + `tensorzero.toml` `[tools.neighbors]`/`[tools.glossary]` descriptions

**Interfaces:**
- Consumes: `glossa::tools::{neighbors, glossary}` (Task 1).

- [ ] **Step 1: `src/mcp.rs`** — change `NeighborsArgs` to `{ path: String, n: u32, depth: Option<usize> }`; the `neighbors` tool opens the index + graph and calls `crate::tools::neighbors(&idx, &g, &a.path, a.n as u64, a.depth.unwrap_or(1), &self.trace)`. The `glossary` tool now passes the index too: `crate::tools::glossary(&idx, &g, &a.name, &self.trace)`. Update both `#[tool(description=...)]` strings: neighbors → "Graph neighbors of a chunk: pass the document path and the chunk number `n` (the `[#n]` from a search/grep result). Returns linked sections/documents as `RELATION  path  #n · label` — read any with `read(path, n)`."; glossary → "Resolve a term/name to indexed references (`path #n · label`)."

- [ ] **Step 2: `eval/src/backend/glossa_tools.rs`** — the `neighbors` arm reads `path` (string), `n` (via the existing `parse_n`), `depth` (optional u64→usize, default 1) and calls `glossa::tools::neighbors(idx, g, path, n, depth, trace)` (guard `graph` is `Some`, else `"(graph unavailable)"`). The `glossary` arm passes `idx` too.

- [ ] **Step 3: TZ config** — rewrite `eval/tensorzero/config/tools/neighbors.json` to `{ "properties": { "path": {"type":"string", …}, "n": {"type":"integer", …}, "depth": {"type":"integer", …} }, "required": ["path","n"] }`. Update `[tools.neighbors]` + `[tools.glossary]` descriptions in `tensorzero.toml` to match the MCP strings.

- [ ] **Step 4: Parity test** (existing pattern in `src/tools.rs` or mcp tests): for the same index+graph and `(path, n)`, the MCP-rendered neighbors text == the eval-exec neighbors text. (If a full parity harness is heavy, assert both call the shared fn and a representative render matches a golden string.)

- [ ] **Step 5: Build the eval (kb-eval STOPPED first!).** `cargo test -p kb-eval backend::glossa_tools` , `cargo build -p kb-eval --release`. Full `cargo test -p glossa`.

- [ ] **Step 6: Commit** `git commit -m "feat(neighbors/glossary): (path,#n) contract across MCP + eval + TZ config"`

## Self-Review

**Coverage:** index bridge helpers (T1), neighbors/glossary `(path,#n)` render (T1), MCP+eval+config wiring + parity (T2). **Types:** `neighbors(idx,g,path,n,depth,trace)->String`, `glossary(idx,g,name,trace)->String`, `location_for_ord`/`ord_for_location` consistent across tasks. **Placeholders:** `Node.prov.source_path` accessor flagged to confirm. **Single source:** both surfaces call the shared fns. **Prod constraint:** tool-level only, no harness change.
