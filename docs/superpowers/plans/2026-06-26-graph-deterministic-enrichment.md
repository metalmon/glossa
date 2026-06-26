# Deterministic Graph Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** At indexing time, deterministically add `NEXT`/`PREV` (sequential), `PARENT`/`CHILD` (heading hierarchy), and `REFERENCES` (cross-document links) edges so `neighbors`/`glossary` are useful — no model.

**Architecture:** New edge builders in `src/graph/build.rs`; a pure link parser `src/extract/links.rs`; `index_dir` (streaming) maintains per-file `prev_sec` + `seen` state for sequential/hierarchy edges and collects links for a post-walk `REFERENCES` resolution. All edges carry the file as provenance, so the existing `delete_by_source` cleans them on reindex. No ontology change.

**Tech Stack:** Rust, `glossa::graph::store::{GraphStore, Node, Edge, Provenance}`, `regex` (already a dep), `std::fs::canonicalize`.

## Global Constraints

- **Pure-Rust, C-free** (`cargo tree -p glossa -i cc` empty): NO new deps. `regex` is already a dep (`src/grep.rs`, `src/glob.rs`); `OnceLock` is std.
- Indexing must not abort on a bad file — keep the per-file error-tolerant streaming (`let _ = ...` on graph writes).
- Deterministic only; no model/network.
- Edge provenance `source_path = the file`; cleanup via existing `delete_by_source`.
- `Section` node id MUST be identical between `build_section` and the new edge wiring — both go through `section_id(path, location)`.
- TDD. Frequent commits.

---

### Task 1: Section↔Section edges — sequential (NEXT/PREV) + hierarchy (PARENT/CHILD)

**Files:**
- Modify: `src/graph/build.rs` (id helper, prov helper, edge builders, ancestor resolver + tests)
- Modify: `src/index/store.rs` (`index_dir` per-file state wiring + test)

**Interfaces:**
- Produces: `build::section_id(path: &str, location: &str) -> String`; `build::link_sequential(g, prev_id, cur_id, sig, src_path)`; `build::link_parent(g, child_id, parent_id, sig, src_path)`; `build::nearest_ancestor(seen: &HashMap<String,String>, location: &str) -> Option<String>`.
- Consumes: `GraphStore::put_edge`, `Edge`, `Provenance`, `FileSig`, `now_secs()`.

- [ ] **Step 1: Add helpers + edge builders to `src/graph/build.rs`**

Add near the top (after `now_secs`):
```rust
/// Deterministic Section node id for a chunk location within a document.
pub fn section_id(path: &str, location: &str) -> String {
    format!("{path}#{location}")
}

fn structural_prov(src_path: &str, sig: FileSig) -> Provenance {
    Provenance {
        source_path: src_path.to_string(),
        range: None,
        file_sig: Some(sig),
        origin: "auto-structural".into(),
        confidence: 1.0,
        created_at: now_secs(),
    }
}
```
Refactor `build_section` to use `section_id` (replace its `let sec_id = format!("{path}#{}", chunk.location);` with `let sec_id = section_id(&path, &chunk.location);`).

Append the edge builders + ancestor resolver:
```rust
/// Link two consecutive sections in document order: prev →NEXT→ cur and cur →PREV→ prev.
pub fn link_sequential(g: &GraphStore, prev_id: &str, cur_id: &str, sig: FileSig, src_path: &str) -> anyhow::Result<()> {
    g.put_edge(&Edge { from: prev_id.to_string(), to: cur_id.to_string(), edge_type: "NEXT".into(), prov: structural_prov(src_path, sig) })?;
    g.put_edge(&Edge { from: cur_id.to_string(), to: prev_id.to_string(), edge_type: "PREV".into(), prov: structural_prov(src_path, sig) })?;
    Ok(())
}

/// Link a child section to its nearest ancestor: child →PARENT→ parent and parent →CHILD→ child.
pub fn link_parent(g: &GraphStore, child_id: &str, parent_id: &str, sig: FileSig, src_path: &str) -> anyhow::Result<()> {
    g.put_edge(&Edge { from: child_id.to_string(), to: parent_id.to_string(), edge_type: "PARENT".into(), prov: structural_prov(src_path, sig) })?;
    g.put_edge(&Edge { from: parent_id.to_string(), to: child_id.to_string(), edge_type: "CHILD".into(), prov: structural_prov(src_path, sig) })?;
    Ok(())
}

/// Nearest existing ancestor of `location` among the per-file `seen` (location → sec_id) map:
/// the longest breadcrumb prefix (split on " > ") that has a Section node. None for top-level / PDF.
pub fn nearest_ancestor(seen: &std::collections::HashMap<String, String>, location: &str) -> Option<String> {
    let segs: Vec<&str> = location.split(" > ").collect();
    for k in (1..segs.len()).rev() {
        let prefix = segs[..k].join(" > ");
        if let Some(id) = seen.get(&prefix) {
            return Some(id.clone());
        }
    }
    None
}
```

- [ ] **Step 2: Unit-test `nearest_ancestor` in `src/graph/build.rs` tests**

```rust
#[test]
fn nearest_ancestor_picks_longest_existing_prefix() {
    let mut seen = std::collections::HashMap::new();
    seen.insert("A".to_string(), "id:A".to_string());
    seen.insert("A > B".to_string(), "id:AB".to_string());
    // child "A > B > C": nearest is "A > B"
    assert_eq!(nearest_ancestor(&seen, "A > B > C").as_deref(), Some("id:AB"));
    // gap: "A" exists but "A > X" does not -> child "A > X > Y" falls back to "A"
    assert_eq!(nearest_ancestor(&seen, "A > X > Y").as_deref(), Some("id:A"));
    // top-level has no ancestor
    assert_eq!(nearest_ancestor(&seen, "A"), None);
    // PDF page (no " > ")
    assert_eq!(nearest_ancestor(&seen, "p.3"), None);
}
```

- [ ] **Step 3: Run the build tests**

Run: `cargo test -p glossa graph::build`
Expected: `nearest_ancestor_picks_longest_existing_prefix` + the existing `builds_document_and_sections_then_deletes_by_source` PASS.

- [ ] **Step 4: Wire the per-file state into `index_dir`** (`src/index/store.rs`)

Inside the `walk::walk_files` callback, AFTER `let mut seq = 0u64;` add:
```rust
        let mut prev_sec: Option<String> = None;
        let mut seen: std::collections::HashMap<String, String> = std::collections::HashMap::new();
```
Inside the `extract_file` per-chunk closure, AFTER the existing `let _ = crate::graph::build::build_section(&graph, &c, sig);` add:
```rust
            let cur_id = crate::graph::build::section_id(&path_str, &c.location);
            if let Some(prev) = prev_sec.as_deref() {
                let _ = crate::graph::build::link_sequential(&graph, prev, &cur_id, sig, &path_str);
            }
            if let Some(parent) = crate::graph::build::nearest_ancestor(&seen, &c.location) {
                let _ = crate::graph::build::link_parent(&graph, &cur_id, &parent, sig, &path_str);
            }
            seen.insert(c.location.clone(), cur_id.clone());
            prev_sec = Some(cur_id);
```

- [ ] **Step 5: Add an `index_dir` integration test** (`src/index/store.rs` `incremental_tests`)

```rust
#[test]
fn index_dir_builds_sequential_and_hierarchy_edges() {
    use crate::graph::build::section_id;
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.md"), b"# A\nintro\n## B\nbody b\n## C\nbody c\n").unwrap();
    index_dir(dir.path(), true).unwrap();
    let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
    let p = dir.path().join("a.md").to_string_lossy().to_string();
    let a = section_id(&p, "A");
    let ab = section_id(&p, "A > B");
    let ac = section_id(&p, "A > C");
    // sequential: A -> A>B -> A>C reachable from A's section via outgoing edges
    let na = crate::graph::traverse::neighbors(&g, &a, None, 1).unwrap();
    assert!(na.contains(&ab), "A neighbors include next/child A>B: {na:?}");
    // hierarchy: A>B's parent A is reachable
    let nab = crate::graph::traverse::neighbors(&g, &ab, None, 1).unwrap();
    assert!(nab.contains(&a), "A>B neighbors include parent A: {nab:?}");
    assert!(nab.contains(&ac), "A>B neighbors include next sibling A>C: {nab:?}");
}
```

- [ ] **Step 6: Run tests + C-free gate**

Run: `cargo test -p glossa graph:: ` and `cargo test -p glossa index::store` and `cargo tree -p glossa -i cc` (empty).
Expected: all PASS, C-free holds.

- [ ] **Step 7: Commit**

```bash
git add src/graph/build.rs src/index/store.rs
git commit -m "feat(graph): sequential + hierarchy section edges at index time"
```

---

### Task 2: Cross-document `REFERENCES` from explicit links

**Files:**
- Create: `src/extract/links.rs` (pure link parser + tests)
- Modify: `src/extract/mod.rs` (add `pub mod links;`)
- Modify: `src/graph/build.rs` (add `link_reference`)
- Modify: `src/index/store.rs` (`index_dir`: collect links, post-walk resolution + test)

**Interfaces:**
- Produces: `extract::links::extract_links(text: &str) -> Vec<String>`; `build::link_reference(g, src_doc, dst_doc, sig)`.
- Consumes: `std::fs::canonicalize`, `next.files` (the indexed path → `FileSig` map).

- [ ] **Step 1: Create `src/extract/links.rs`**

```rust
//! Deterministic extraction of explicit document links (markdown + html), for building
//! cross-document REFERENCES graph edges. No model.

use regex::Regex;
use std::sync::OnceLock;

/// Markdown `[text](target)` and html `href="target"` targets, excluding external URLs
/// (http/https/mailto, protocol-relative `//`) and pure anchors (`#...`). A trailing
/// `#anchor` on an otherwise-local target is stripped.
pub fn extract_links(text: &str) -> Vec<String> {
    static MD: OnceLock<Regex> = OnceLock::new();
    static HTML: OnceLock<Regex> = OnceLock::new();
    let md = MD.get_or_init(|| Regex::new(r"\[[^\]]*\]\(([^)\s]+)\)").unwrap());
    let html = HTML.get_or_init(|| Regex::new(r#"(?i)href\s*=\s*["']([^"']+)["']"#).unwrap());
    let mut out = Vec::new();
    for caps in md.captures_iter(text) {
        if let Some(m) = caps.get(1) { push_if_local(m.as_str(), &mut out); }
    }
    for caps in html.captures_iter(text) {
        if let Some(m) = caps.get(1) { push_if_local(m.as_str(), &mut out); }
    }
    out
}

fn push_if_local(target: &str, out: &mut Vec<String>) {
    let t = target.trim();
    let lower = t.to_ascii_lowercase();
    if t.is_empty()
        || t.starts_with('#')
        || lower.starts_with("http://")
        || lower.starts_with("https://")
        || lower.starts_with("mailto:")
        || t.starts_with("//")
    {
        return;
    }
    let path = t.split('#').next().unwrap_or(t);
    if !path.is_empty() {
        out.push(path.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_markdown_and_html_local_links_only() {
        let text = "see [B](b.md) and <a href=\"sub/c.html\">c</a> and \
                    [ext](https://x.com) and [anchor](#sec) and [d](d.md#part)";
        let links = extract_links(text);
        assert!(links.contains(&"b.md".to_string()));
        assert!(links.contains(&"sub/c.html".to_string()));
        assert!(links.contains(&"d.md".to_string()), "trailing #anchor stripped");
        assert!(!links.iter().any(|l| l.contains("x.com")), "external excluded");
        assert!(!links.iter().any(|l| l.starts_with('#')), "anchor excluded");
    }
}
```

- [ ] **Step 2: Register the module** — add `pub mod links;` to `src/extract/mod.rs` (alongside the other `pub mod` extractor declarations).

- [ ] **Step 3: Add `link_reference` to `src/graph/build.rs`**

```rust
/// Link a document to another document it explicitly references.
pub fn link_reference(g: &GraphStore, src_doc: &str, dst_doc: &str, sig: FileSig) -> anyhow::Result<()> {
    g.put_edge(&Edge { from: src_doc.to_string(), to: dst_doc.to_string(), edge_type: "REFERENCES".into(), prov: structural_prov(src_doc, sig) })
}
```

- [ ] **Step 4: Collect links + resolve in `index_dir`** (`src/index/store.rs`)

Before `crate::walk::walk_files(...)`, declare:
```rust
    let mut links: Vec<(String, String)> = Vec::new();
```
Inside the walk callback, declare a per-file accumulator next to `prev_sec`:
```rust
        let mut file_links: Vec<String> = Vec::new();
```
Inside the per-chunk closure (after the section-edge wiring from Task 1):
```rust
            file_links.extend(crate::extract::links::extract_links(&c.text));
```
After the `extract_file(...)?;` call returns (still inside the walk callback), merge:
```rust
        for t in file_links {
            links.push((path_str.clone(), t));
        }
```
After the walk loop and the removed-files loop, BEFORE `writer.commit()?;`, resolve references:
```rust
    // Cross-document REFERENCES: resolve collected link targets against indexed documents.
    let mut by_canon: std::collections::HashMap<std::path::PathBuf, String> = std::collections::HashMap::new();
    for p in next.files.keys() {
        if let Ok(c) = std::fs::canonicalize(p) {
            by_canon.insert(c, p.clone());
        }
    }
    for (src, target) in &links {
        let src_dir = std::path::Path::new(src).parent().unwrap_or_else(|| std::path::Path::new("."));
        if let Ok(canon) = std::fs::canonicalize(src_dir.join(target)) {
            if let Some(dst) = by_canon.get(&canon) {
                if dst != src {
                    let sig = next.files.get(src).copied().unwrap_or(FileSig { mtime_secs: 0, size: 0 });
                    let _ = crate::graph::build::link_reference(&graph, src, dst, sig);
                }
            }
        }
    }
```
(Confirm the `FileSig` struct fields `{ mtime_secs, size }` against `src/index/manifest.rs`; adjust the fallback literal if the field names differ. `FileSig` must be `Copy` — it is used via `.copied()`; if not `Copy`, use `.cloned()`.)

- [ ] **Step 5: Add a references integration test** (`src/index/store.rs` `incremental_tests`)

```rust
#[test]
fn index_dir_builds_cross_document_references() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.md"), b"# A\nsee [the manual](b.md) and [ext](https://x.com)\n").unwrap();
    std::fs::write(dir.path().join("b.md"), b"# B\ncontent\n").unwrap();
    index_dir(dir.path(), true).unwrap();
    let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
    let a = dir.path().join("a.md").to_string_lossy().to_string();
    let b = dir.path().join("b.md").to_string_lossy().to_string();
    let na = crate::graph::traverse::neighbors(&g, &a, None, 1).unwrap();
    assert!(na.contains(&b), "a.md REFERENCES b.md: {na:?}");
}
```

- [ ] **Step 6: Run tests + C-free gate**

Run: `cargo test -p glossa extract::links` , `cargo test -p glossa index::store` , `cargo tree -p glossa -i cc` (empty).
Expected: all PASS, C-free holds.

- [ ] **Step 7: Run the full glossa suite**

Run: `cargo test -p glossa`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/extract/links.rs src/extract/mod.rs src/graph/build.rs src/index/store.rs
git commit -m "feat(graph): cross-document REFERENCES edges from explicit links"
```

## Self-Review

**Coverage:** sequential (T1), hierarchy (T1), references (T2), all via provenance-tagged edges cleaned by `delete_by_source`. **Types:** `section_id`/`structural_prov`/`link_sequential`/`link_parent`/`nearest_ancestor`/`link_reference` consistent across T1/T2; `extract_links` signature matches its call site. **Placeholders:** the only verify-before-write items are the `FileSig` field names / `Copy`-ness (bounded by "confirm against `src/index/manifest.rs`") and that `pub mod` style matches `src/extract/mod.rs`. **Reindex:** the eval run uses `reindex --force`, which rebuilds all edges cleanly.
