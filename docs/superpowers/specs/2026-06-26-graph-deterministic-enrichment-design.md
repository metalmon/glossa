# Deterministic Graph Enrichment — Design

**Date:** 2026-06-26
**Status:** Approved (design confirmed by user)
**Branch:** `feat/graph-deterministic-enrichment`

## Goal

Enrich the structural graph built at indexing time with everything derivable **deterministically, without a model**, so `neighbors`/`glossary` become useful: sequential section links, heading hierarchy, and cross-document references.

## Background

Today `src/graph/build.rs` builds only: a `Document` node per file, a `Section` node per chunk, and `CONTAINS` edges (Document→Section). Consequences: `neighbors(<section>)` returns nothing (Sections have no outgoing edges), `neighbors(<document>)` returns its sections, `glossary(name)` matches only an exact path/location string. The agent's graph tools are near-useless.

Indexing (`index_dir` in `src/index/store.rs`) **streams one chunk at a time** (`build_document` once + `build_section` per chunk), constant memory. The chunker (`chunk_markdown`) already encodes the full heading breadcrumb in `Chunk.location` (`"A > B > C"`), so heading depth/hierarchy is recoverable from the location string — **no `Chunk` change needed**.

## Design

All new edges are written via `GraphStore::put_edge` (the structural layer bypasses ontology validation), with `Provenance.source_path = the file`, so the existing `delete_by_source` cleans them on reindex. No ontology change.

### 1. Sequential `NEXT` / `PREV` (within a document)
In `index_dir`'s per-file streaming, keep `prev_sec: Option<String>`. For each chunk's section `cur`: if `prev` is `Some`, add `prev →NEXT→ cur` and `cur →PREV→ prev`; set `prev = cur`. This is document reading order (by `ord`), mirroring read's `‹ prev · next ›` footer. 100% reliable.

### 2. Heading hierarchy `PARENT` / `CHILD`
Derived from the `location` breadcrumb. Keep a per-file `seen: HashMap<location, sec_id>`. For a section with location `L` containing `" > "`, walk its prefixes longest→shortest (`"A > B"`, then `"A"`) and link to the **nearest existing ancestor** section (intermediate headings with empty bodies produce no node, hence "nearest"): add `child →PARENT→ parent` and `parent →CHILD→ child`. Then insert `L → sec_id` into `seen`. PDF chunks (`p.N`, no `" > "`) get no hierarchy — sequential only. No `Chunk` change.

### 3. Cross-document `REFERENCES` (explicit links only)
During the walk, collect explicit links from each chunk's (markdown) text — markdown `[text](target)` and html `href="target"` — skipping external URLs (`http://`, `https://`, `mailto:`) and pure anchors (`#…`). **After** the walk (all `Document` nodes known), resolve each `target` relative to its source file's directory and add `srcDoc →REFERENCES→ dstDoc` only when it resolves to an actually-indexed document. Resolution uses `std::fs::canonicalize` on both sides (handles `..`, separators, Windows case) against a `canonical → node_id` map of all indexed docs. Low false-positive: only real link syntax that resolves to a real indexed file.

### Edge types
`NEXT`, `PREV`, `PARENT`, `CHILD` (Section↔Section), `REFERENCES` (Document→Document) — in addition to the existing `CONTAINS`. `neighbors` (which returns all outgoing edge targets) needs no change: `neighbors(<section>)` → prev/next siblings + parent + children; `neighbors(<document>)` → sections + referenced documents.

## Components / files

- `src/graph/build.rs`: add `section_id(path, location) -> String` (shared id formatter), `link_sequential(g, prev_id, cur_id, sig, src)`, `link_parent(g, child_id, parent_id, sig, src)`, `nearest_ancestor(seen, location) -> Option<String>`. Refactor `build_section` to use `section_id`.
- `src/extract/links.rs` (new): `extract_links(text: &str) -> Vec<String>` — markdown + html link targets, external/anchor filtered. Pure + unit-tested.
- `src/index/store.rs` `index_dir`: per-file `prev_sec` + `seen`; collect `links: Vec<(src_path, target)>`; post-walk reference resolution.

## Reindex requirement

The graph changes → the corpus must be rebuilt with `index_dir(dir, force=true)` before the eval run. (The MCP server is stopped, so `kb.exe` is unlocked — normal `cargo build --release` + reindex.)

## Testing

- Sequential: index a multi-section markdown → consecutive sections have `NEXT`/`PREV`; `neighbors(section)` includes both adjacent sections.
- Hierarchy: nested headings (`# A` / `## B` / `### C` with bodies) → `B PARENT A`, `C PARENT B`, and `CHILD` inverse; nearest-ancestor when an intermediate heading has no body.
- References: two markdown docs where `a.md` links `[x](b.md)` → `a →REFERENCES→ b`; an `http://` link produces no edge; `neighbors(a)` includes `b`.
- `extract_links` unit tests: markdown, html, external-filtered, anchor-filtered.
- Existing graph/index tests stay green (`build_structural`, `index_dir_builds_structural_graph`).

## Global constraints

- **Pure-Rust, C-free** (`cargo tree -p glossa -i cc` empty): no new deps (use `regex` only if already present; prefer hand-rolled parsing to avoid adding deps — verify `regex` availability before using). Reuse `GraphStore`, `Provenance`, `canonicalize`.
- File-First; indexing must not abort on a bad file (keep the per-file error-tolerant streaming).
- Deterministic only — no model, no network.
- Reindex cleanup via existing `delete_by_source` provenance.
- TDD.

## Out of scope (follow-ups)

- Acronym aliases / `Term` nodes / a parsed glossary document (Layer-2, needs a term model and has medium reliability).
- Semantic/similarity edges (`MENTIONS`, `SIMILAR`, `CO_OCCURS`) — require a model or embeddings.
- Bare filename-mention cross-links (heuristic, lower reliability).

## Risks

- Incremental (non-force) reindex: a `REFERENCES` edge into a later-deleted document can dangle (target node gone, edge remains with the source's provenance). Minor; a full `reindex --force` is clean, which is what the eval run uses.
- `canonicalize` requires both files to exist (they do during indexing); links to not-yet-or-never-indexed targets simply produce no edge (correct).
