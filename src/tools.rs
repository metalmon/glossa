//! Single source of truth for the agent tools' model-facing output. Both the MCP server
//! (src/mcp.rs) and the kb-eval harness call these so prod and eval render identically.

use crate::grep::GrepOpts;
use crate::index::store::{DocIndex, RankedHit};
use crate::trace::TraceLog;
use serde_json::json;

/// BM25 search (optionally scoped). Returns (model text, hits for the caller's scoring).
pub fn search(idx: &DocIndex, query: &str, limit: usize, glob: Option<&str>, file_type: Option<&str>, trace: &TraceLog) -> (String, Vec<RankedHit>) {
    match idx.search_filtered(query, limit.max(1), glob, file_type) {
        Ok(hits) => {
            let th: Vec<_> = hits.iter().map(|h| json!({"path": h.path, "location": h.location, "score": h.score})).collect();
            trace.log("search", json!({"query": query}), json!(th));
            let body = if hits.is_empty() { "(no results)".to_string() }
                       else { hits.iter().map(|h| h.display_line()).collect::<Vec<_>>().join("\n") };
            (body, hits)
        }
        Err(e) => (format!("search error: {e}"), Vec::new()),
    }
}

/// ripgrep-style literal/regex search; model text only.
pub fn grep(idx: &DocIndex, pattern: &str, opts: &GrepOpts, trace: &TraceLog) -> String {
    match crate::grep::grep(idx, pattern, opts) {
        Ok(hits) => {
            trace.log("grep", json!({"pattern": pattern}), json!({"hits": hits.len()}));
            if hits.is_empty() { "(no matches)".to_string() }
            else { hits.iter().map(|h| h.display_line()).collect::<Vec<_>>().join("\n") }
        }
        Err(e) => format!("grep error: {e}"),
    }
}

/// A read result: the chunk text (with prev/next footer) plus the file's images for vision models.
pub struct ReadOut {
    pub text: String,
    pub images: Vec<crate::read::DocImage>,
}

/// Collect `(attribution, path, ord)` for every section node `nid` MENTIONS, de-duplicated via `seen`.
fn gather_mentions(idx: &DocIndex, g: &crate::graph::store::GraphStore, nid: &str, attr: &str, seen: &mut std::collections::HashSet<(String, u64)>, out: &mut Vec<(String, String, u64)>) {
    for e in g.outgoing(nid).unwrap_or_default() {
        if e.edge_type != crate::graph::MENTIONS {
            continue;
        }
        if let Some((p, loc)) = e.to.split_once('#') {
            if let Ok(Some(ord)) = idx.ord_for_location(p, loc) {
                if seen.insert((p.to_string(), ord)) {
                    out.push((attr.to_string(), p.to_string(), ord));
                }
            }
        }
    }
}

/// Omnivorous read of a graph NODE: the node's own line, plus every chunk it AND its 1-hop reasoning
/// neighbours MENTION — each labelled with where it came from. Reading a Resolution gives its fix
/// chunk; reading a Symptom also pulls the Cause/Resolution evidence one hop along the chain.
fn read_node(idx: &DocIndex, g: &crate::graph::store::GraphStore, node: crate::graph::store::Node) -> ReadOut {
    let mut seen = std::collections::HashSet::new();
    let mut chunks: Vec<(String, String, u64)> = Vec::new();
    gather_mentions(idx, g, &node.id, "MENTIONS", &mut seen, &mut chunks); // the node's own evidence
    let label_of = |nid: &str| g.get_node(nid).ok().flatten().map(|n| n.label).unwrap_or_else(|| nid.to_string());
    // 1-hop reasoning neighbours (outgoing →, incoming ←), skipping the MENTIONS-to-section links.
    for e in g.outgoing(&node.id).unwrap_or_default() {
        if e.edge_type != crate::graph::MENTIONS && !e.to.contains('#') {
            gather_mentions(idx, g, &e.to, &format!("via → {} {}", e.edge_type, label_of(&e.to)), &mut seen, &mut chunks);
        }
    }
    for e in g.incoming(&node.id).unwrap_or_default() {
        if e.edge_type != crate::graph::MENTIONS && !e.from.contains('#') {
            gather_mentions(idx, g, &e.from, &format!("via ← {} {}", e.edge_type, label_of(&e.from)), &mut seen, &mut chunks);
        }
    }
    let mut text = format!("{}  [{}]  {}", node.id, node.node_type, node.label);
    if chunks.is_empty() {
        text.push_str("\n(no source chunk linked to this node)");
    }
    let mut images = Vec::new();
    for (attr, p, ord) in &chunks {
        if let Ok(Some(c)) = idx.read_chunk_by_ord(p, *ord) {
            text.push_str(&format!("\n\n── {attr} · {p} #{ord} ──\n{}", c.body));
        }
        images.extend(crate::read::extract_images(&idx.doc_file(p), *ord, 4).unwrap_or_default());
    }
    ReadOut { text, images }
}

/// Read chunk `n` of `path`: full stored body + a unified prev/next footer, plus extracted images
/// (empty if the source file is absent — body still comes from the index). No truncation. Omnivorous:
/// if `path` is a graph NODE id, returns that node + its (1-hop) evidence chunks (see `read_node`).
pub fn read(idx: &DocIndex, graph: Option<&crate::graph::store::GraphStore>, path: &str, n: u64, trace: &TraceLog) -> ReadOut {
    // Omnivorous: a REASONING node id off a glossary line reads as the node + its evidence. A
    // structural node id IS a document path (e.g. a Document's id is its path), so it falls through
    // to the normal document read below.
    if let Some(g) = graph {
        if let Ok(Some(node)) = g.get_node(path) {
            if !crate::graph::STRUCTURAL_NODES.contains(&node.node_type.as_str()) {
                trace.log("read", json!({ "node": path }), json!({ "node": path }));
                return read_node(idx, g, node);
            }
        }
    }
    // Try the exact path first. If the document isn't found, the model may have mangled the path
    // (e.g. collapsed a double space when copying it) — resolve it tolerantly and retry once.
    let (path, chunk): (String, _) = match idx.read_chunk_by_ord(path, n) {
        Ok(Some(c)) => (path.to_string(), c),
        Ok(None) => match idx.last_chunk_ord(path) {
            // Document exists; the chunk number is just out of range → report the valid range.
            Ok(Some(max)) => return ReadOut {
                text: format!("no chunk #{n} in {path} — this document has {max} chunks (read #1..#{max})"),
                images: Vec::new(),
            },
            // No exact path match — try a whitespace-tolerant resolve, then retry once.
            Ok(None) => match idx.resolve_path(path).ok().flatten() {
                Some(real) => match idx.read_chunk_by_ord(&real, n) {
                    Ok(Some(c)) => (real, c),
                    _ => {
                        let text = match idx.last_chunk_ord(&real) {
                            Ok(Some(max)) => format!("no chunk #{n} in {real} — this document has {max} chunks (read #1..#{max})"),
                            _ => format!("no document indexed at {path} — check the path from a search/glob result"),
                        };
                        return ReadOut { text, images: Vec::new() };
                    }
                },
                None => return ReadOut {
                    text: format!("no document indexed at {path} — check the path from a search/glob result"),
                    images: Vec::new(),
                },
            },
            Err(e) => return ReadOut { text: format!("no chunk #{n} in {path} (range lookup failed: {e})"), images: Vec::new() },
        },
        Err(e) => return ReadOut { text: format!("read error: {e}"), images: Vec::new() },
    };
    let path = path.as_str();
    trace.log("read", json!({"path": path, "n": n}), json!({"path": path}));
    let footer = match (chunk.prev, chunk.next) {
        (Some(p), Some(nx)) => format!("\n\n‹ prev #{p} · next #{nx} ›"),
        (None, Some(nx)) => format!("\n\n‹ start of document · next #{nx} ›"),
        (Some(p), None) => format!("\n\n‹ prev #{p} · end of document ›"),
        (None, None) => String::new(),
    };
    let images = crate::read::extract_images(&idx.doc_file(path), n, 4).unwrap_or_default();
    ReadOut { text: format!("{}{}", chunk.body, footer), images }
}

/// List documents by path mask; model text only.
pub fn glob(idx: &DocIndex, pattern: &str, trace: &TraceLog) -> String {
    match crate::glob::glob_docs(idx, pattern) {
        Ok(docs) => {
            trace.log("glob", json!({"pattern": pattern}), json!({"docs": docs.len()}));
            if docs.is_empty() { "(no documents match — glob matches file PATHS, not text inside documents; use search or grep to find content)".to_string() }
            else { docs.iter().map(|(p, n)| format!("{p}  ({n} chunks)")).collect::<Vec<_>>().join("\n") }
        }
        Err(e) => format!("glob error: {e}"),
    }
}

/// Render a graph node as a `(path, #ord)` reference string.
/// Section  → `"path  #ord · label"` (None if `ord_for_location` can't resolve).
/// Document → `"path  (document)"`.
/// Other    → `None` (caller decides whether to skip or fall back to the raw id).
fn node_ref(idx: &DocIndex, node: &crate::graph::store::Node) -> Option<String> {
    let tp = node.prov.source_path.as_str();
    match node.node_type.as_str() {
        "Section" => idx
            .ord_for_location(tp, &node.label)
            .ok()
            .flatten()
            .map(|ord| format!("{}  #{} · {}", tp, ord, node.label)),
        "Document" => Some(format!("{}  (document)", tp)),
        _ => None,
    }
}

/// Compact community/centrality annotation drawn from `node_meta` (populated by
/// `kb graph generalize` / auto-run at the end of `reindex`). Returns "" when no meta exists yet,
/// so output is byte-identical on graphs that haven't been generalized. Format:
/// `"  · comm 3 · pr 0.142 · deg 5"` — only the parts that are present.
fn meta_suffix(g: &crate::graph::store::GraphStore, id: &str) -> String {
    let Ok(Some(m)) = g.node_meta(id) else { return String::new() };
    let mut parts = Vec::new();
    if let Some(c) = m.community {
        parts.push(format!("comm {c}"));
    }
    if let Some(pr) = m.pagerank {
        parts.push(format!("pr {pr:.3}"));
    }
    if let Some(d) = m.degree {
        parts.push(format!("deg {d}"));
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!("  · {}", parts.join(" · "))
    }
}

/// The ontology-derived knob the reasoning tools need: which relations form the spines, so
/// `glossary` can walk a node's full chain. Built once per call from `Ontology` by each surface
/// (MCP + eval) so both render identically. (The evidence anchor is the fixed `graph::MENTIONS`
/// contract — not configured here.)
pub struct ChainSpec {
    pub spine_rels: Vec<String>,
}

impl ChainSpec {
    pub fn from_ontology(ont: &crate::graph::ontology::Ontology) -> Self {
        ChainSpec { spine_rels: ont.spine_relations() }
    }
}

impl Default for ChainSpec {
    /// No spines — `glossary` prints just the node head. Used by callers/tests that don't drive
    /// the reasoning chain.
    fn default() -> Self {
        ChainSpec { spine_rels: Vec::new() }
    }
}

/// Render the far node of an edge: its `(path #ord)` anchor for a section/document, else
/// `id  [type]  label`. Used by `neighbors` to show related cases.
fn endpoint_ref(idx: &DocIndex, g: &crate::graph::store::GraphStore, nid: &str) -> String {
    match g.get_node(nid) {
        Ok(Some(node)) => match node_ref(idx, &node) {
            Some(r) => r,
            None => format!("{}  [{}]  {}", node.id, node.node_type, node.label),
        },
        _ => nid.to_string(),
    }
}

/// The read anchor for a reasoning node: follow its `MENTIONS` edge (the fixed evidence contract)
/// to a Section and render the section's `(path #ord · label)`, so the agent can `read(path, ord)`
/// for the detail behind the node. Empty string when the node mentions no indexed section.
fn read_anchor(idx: &DocIndex, g: &crate::graph::store::GraphStore, id: &str) -> String {
    for e in g.outgoing(id).unwrap_or_default() {
        if e.edge_type != crate::graph::MENTIONS {
            continue;
        }
        if let Ok(Some(sec)) = g.get_node(&e.to) {
            if let Some(r) = node_ref(idx, &sec) {
                return format!("   — read {r}");
            }
        }
    }
    String::new()
}

/// Walk a reasoning node's full spine chain (e.g. Symptom → Cause → Resolution, or Task →
/// Resolution) by following outgoing edges whose type is one of `spine_rels`, transitively. Each
/// hop is one indented line `→ REL  [Type] label  — read <anchor>`; deeper hops indent further.
/// `seen` breaks cycles. SIMILAR/structural edges are intentionally skipped — they belong to
/// `neighbors`, not the answer chain.
fn chain_lines(
    idx: &DocIndex,
    g: &crate::graph::store::GraphStore,
    id: &str,
    spec: &ChainSpec,
    depth: usize,
    seen: &mut std::collections::HashSet<String>,
    out: &mut Vec<String>,
) {
    for e in g.outgoing(id).unwrap_or_default() {
        if !spec.spine_rels.iter().any(|r| r == &e.edge_type) {
            continue;
        }
        if !seen.insert(e.to.clone()) {
            continue;
        }
        let Ok(Some(node)) = g.get_node(&e.to) else { continue };
        let indent = "    ".repeat(depth + 1);
        out.push(format!(
            "{indent}→ {}  [{}]  {}{}{}",
            e.edge_type,
            node.node_type,
            node.label,
            meta_suffix(g, &node.id),
            read_anchor(idx, g, &node.id),
        ));
        chain_lines(idx, g, &node.id, spec, depth + 1, seen, out);
    }
}

/// Resolve a name/term to graph nodes and render each with its full reasoning chain.
/// Structural nodes (Section/Document) show their `(path #ord)` anchor. A reasoning node
/// (Symptom/Cause/Task/…) shows its `id [type] label` then, walked from it along the ontology's
/// spine relations, the whole chain to the Resolution — so ONE `glossary` call surfaces the
/// cause and the fix, each with a `read` anchor. SIMILAR/community links live in `neighbors`.
/// Empty → `"(no matches)"`.
pub fn glossary(idx: &DocIndex, g: &crate::graph::store::GraphStore, name: &str, spec: &ChainSpec, trace: &TraceLog) -> String {
    match g.resolve(name) {
        Ok(ids) => {
            trace.log("glossary", json!({ "name": name }), json!({ "ids": ids.len() }));
            if ids.is_empty() {
                return "(no matches)".to_string();
            }
            let lines: Vec<String> = ids
                .iter()
                .map(|id| match g.get_node(id) {
                    Ok(Some(node)) => {
                        let base = format!("{}  [{}]  {}", node.id, node.node_type, node.label);
                        match node_ref(idx, &node) {
                            // structural node (Section/Document): show its anchor, no chain
                            Some(r) => format!("{base}  —  {r}{}", meta_suffix(g, &node.id)),
                            // reasoning node: append its spine chain (cause → resolution) inline
                            None => {
                                let head = format!(
                                    "{base}{}{}",
                                    meta_suffix(g, &node.id),
                                    read_anchor(idx, g, &node.id),
                                );
                                let mut seen = std::collections::HashSet::new();
                                seen.insert(node.id.clone());
                                let mut chain = Vec::new();
                                chain_lines(idx, g, &node.id, spec, 0, &mut seen, &mut chain);
                                if chain.is_empty() {
                                    head
                                } else {
                                    format!("{head}\n{}", chain.join("\n"))
                                }
                            }
                        }
                    }
                    _ => id.clone(),
                })
                .collect();
            lines.join("\n")
        }
        Err(e) => format!("glossary error: {e}"),
    }
}

/// The generalization layer around a node: its SIMILAR cross-links — the shared-evidence
/// "related cases" written by the generalize pass — in both directions, de-duplicated. The target
/// is either an explicit reasoning-node `id` (copied from a `glossary` line) or a chunk `(path, n)`
/// that resolves to its Section node. Each related node renders with a `read` anchor. No links →
/// `"(no related cases)"`. (Spine chains live in `glossary`; this is the cross-case view.)
pub fn neighbors(idx: &DocIndex, g: &crate::graph::store::GraphStore, node: Option<&str>, path: Option<&str>, n: Option<u64>, trace: &TraceLog) -> String {
    let id: String = if let Some(nid) = node.filter(|s| !s.trim().is_empty()) {
        nid.to_string()
    } else if let (Some(p), Some(nn)) = (path, n) {
        match idx.location_for_ord(p, nn) {
            Ok(Some(loc)) => crate::graph::build::section_id(p, &loc),
            Ok(None) => return format!("no chunk #{nn} in {p}"),
            Err(e) => return format!("neighbors error: {e}"),
        }
    } else {
        return "neighbors needs a node id (from glossary) or a (path, n) chunk".to_string();
    };
    let mut seen = std::collections::HashSet::new();
    let mut lines = Vec::new();
    for e in g.outgoing(&id).unwrap_or_default() {
        if e.edge_type == "SIMILAR" && seen.insert(e.to.clone()) {
            lines.push(format!("SIMILAR  {}{}{}", endpoint_ref(idx, g, &e.to), meta_suffix(g, &e.to), read_anchor(idx, g, &e.to)));
        }
    }
    for e in g.incoming(&id).unwrap_or_default() {
        if e.edge_type == "SIMILAR" && seen.insert(e.from.clone()) {
            lines.push(format!("SIMILAR  {}{}{}", endpoint_ref(idx, g, &e.from), meta_suffix(g, &e.from), read_anchor(idx, g, &e.from)));
        }
    }
    trace.log("neighbors", json!({"id": id}), json!({"related": lines.len()}));
    if lines.is_empty() { "(no related cases)".to_string() } else { lines.join("\n") }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::store::DocIndex;
    use crate::model::Chunk;
    use crate::trace::TraceLog;
    use std::path::PathBuf;

    fn idx() -> (tempfile::TempDir, DocIndex) {
        let d = tempfile::tempdir().unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        i.write_chunks(&[
            Chunk { doc_path: PathBuf::from("АБАК.pdf"), location: "p.7".into(), file_type: "pdf".into(), text: "параметр maxTsdr равен 3000".into() },
        ]).unwrap();
        (d, i)
    }

    fn prov() -> Provenance {
        Provenance { source_path: "d.pdf".into(), range: None, file_sig: None, origin: "agent".into(), confidence: 0.8, created_at: 1 }
    }
    fn node(id: &str, ty: &str, label: &str) -> Node {
        Node { id: id.into(), node_type: ty.into(), label: label.into(), aliases: Vec::new(), prov: prov() }
    }
    fn edge(from: &str, rel: &str, to: &str) -> Edge {
        Edge { from: from.into(), to: to.into(), edge_type: rel.into(), prov: prov() }
    }
    /// A graph with one full causal spine (Symptom→Cause→Resolution) plus two SIMILAR tasks
    /// hanging off the Symptom (the generalization layer).
    fn reasoning_graph() -> (tempfile::TempDir, GraphStore) {
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        for n in [
            node("sym:loss", "Symptom", "Профибус потеря связи"),
            node("cau:tsdr", "Cause", "Малый maxTsdr"),
            node("res:set", "Resolution", "Изменить maxTsdr в 3000"),
            node("tsk:upd", "Task", "Обновление ПО модулей"),
            node("tsk:prog", "Task", "Программирование АБАК"),
        ] {
            g.put_node(&n).unwrap();
        }
        for e in [
            edge("sym:loss", "CAUSED_BY", "cau:tsdr"),
            edge("cau:tsdr", "RESOLVED_BY", "res:set"),
            edge("sym:loss", "SIMILAR", "tsk:upd"),
            edge("sym:loss", "SIMILAR", "tsk:prog"),
        ] {
            g.put_edge(&e).unwrap();
        }
        (d, g)
    }
    fn spine_spec() -> ChainSpec {
        ChainSpec { spine_rels: vec!["CAUSED_BY".into(), "RESOLVED_BY".into()] }
    }

    #[test]
    fn glossary_renders_full_chain_in_one_call() {
        let (_d, i) = idx();
        let (_gd, g) = reasoning_graph();
        let t = TraceLog::disabled();
        let out = glossary(&i, &g, "Профибус потеря связи", &spine_spec(), &t);
        // entry node + the whole chain to the resolution, surfaced by a SINGLE call
        assert!(out.contains("[Symptom]") && out.contains("Профибус потеря связи"), "{out}");
        assert!(out.contains("→ CAUSED_BY") && out.contains("[Cause]") && out.contains("Малый maxTsdr"), "{out}");
        assert!(out.contains("→ RESOLVED_BY") && out.contains("[Resolution]") && out.contains("Изменить maxTsdr"), "{out}");
        // SIMILAR is no longer part of glossary — it moved to neighbors
        assert!(!out.contains("SIMILAR"), "glossary must not show SIMILAR: {out}");
    }

    #[test]
    fn glossary_chain_is_ontology_driven() {
        // No spine relations configured → glossary shows only the node head, no chain.
        let (_d, i) = idx();
        let (_gd, g) = reasoning_graph();
        let t = TraceLog::disabled();
        let empty = ChainSpec { spine_rels: Vec::new() };
        let out = glossary(&i, &g, "Профибус потеря связи", &empty, &t);
        assert!(out.contains("[Symptom]"), "{out}");
        assert!(!out.contains("CAUSED_BY") && !out.contains("RESOLVED_BY"), "no spines → no chain: {out}");
    }

    #[test]
    fn neighbors_shows_similar_generalization_for_a_node_id() {
        let (_d, i) = idx();
        let (_gd, g) = reasoning_graph();
        let t = TraceLog::disabled();
        let out = neighbors(&i, &g, Some("sym:loss"), None, None, &t);
        assert!(out.contains("SIMILAR") && out.contains("Обновление ПО модулей") && out.contains("Программирование АБАК"), "{out}");
        // it must NOT walk the causal spine — that is glossary's job
        assert!(!out.contains("CAUSED_BY") && !out.contains("[Resolution]"), "neighbors is generalization only: {out}");
    }

    #[test]
    fn read_is_omnivorous_over_reasoning_node_ids() {
        let (_d, i) = idx(); // АБАК.pdf #p.7 = "параметр maxTsdr равен 3000"
        let gd = tempfile::tempdir().unwrap();
        let g = GraphStore::open(gd.path()).unwrap();
        g.put_node(&node("res:fix", "Resolution", "Изменить maxTsdr в 3000")).unwrap();
        g.put_edge(&edge("res:fix", "MENTIONS", "АБАК.pdf#p.7")).unwrap();
        let t = TraceLog::disabled();
        // reading the NODE id returns the node line + the evidence chunk it MENTIONS, attributed.
        let out = read(&i, Some(&g), "res:fix", 1, &t).text;
        assert!(out.contains("[Resolution]") && out.contains("Изменить maxTsdr"), "node header: {out}");
        assert!(out.contains("── MENTIONS · АБАК.pdf #7 ──"), "attributed evidence header: {out}");
        assert!(out.contains("параметр maxTsdr равен 3000"), "evidence body: {out}");
        // a plain doc path still reads the chunk (not treated as a node).
        let doc = read(&i, Some(&g), "АБАК.pdf", 7, &t).text;
        assert!(doc.contains("параметр maxTsdr равен 3000") && !doc.contains("── MENTIONS"), "doc read unchanged: {doc}");
    }

    #[test]
    fn search_renders_numbered_or_empty() {
        let (_d, i) = idx();
        let t = TraceLog::disabled();
        let (body, hits) = search(&i, "maxTsdr", 10, None, None, &t);
        assert_eq!(hits.len(), 1);
        assert!(body.starts_with("[#7] ") && body.contains("maxTsdr"));
        let (empty, _) = search(&i, "nonexistentzzz", 10, None, None, &t);
        assert_eq!(empty, "(no results)");
    }

    #[test]
    fn read_returns_full_body_and_unified_footer() {
        let d = tempfile::tempdir().unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        let big = "Я".repeat(5000); // > old 4000-char cap
        i.write_chunks(&[
            Chunk { doc_path: PathBuf::from("d.md"), location: "S1".into(), file_type: "md".into(), text: big.clone() },
            Chunk { doc_path: PathBuf::from("d.md"), location: "S2".into(), file_type: "md".into(), text: "second".into() },
        ]).unwrap();
        let t = TraceLog::disabled();
        let out = read(&i, None, "d.md", 1, &t);
        assert!(out.text.contains(&big), "full body, no cap");                 // not truncated
        assert!(out.text.contains("next #2") && out.text.contains("end of document") == false);
        assert!(out.text.contains("‹ start of document · next #2 ›"));        // unified footer (MCP wording)
        // Out-of-range read reports the valid chunk range so the model can self-correct.
        let oor = read(&i, None, "d.md", 99, &t).text;
        assert!(oor.contains("no chunk #99 in d.md") && oor.contains("2 chunks") && oor.contains("#1..#2"), "range hint: {oor}");
        // A wrong path reports that the document isn't indexed.
        assert!(read(&i, None, "nope.md", 1, &t).text.contains("no document indexed at nope.md"));
    }

    #[test]
    fn read_tolerates_collapsed_whitespace_in_path() {
        let d = tempfile::tempdir().unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        // Real file name has a double space; the model copies it back with a single space.
        i.write_chunks(&[
            Chunk { doc_path: PathBuf::from("dir/Safety Manual -  1_0_3.pdf"), location: "p.1".into(), file_type: "pdf".into(), text: "safety body".into() },
        ]).unwrap();
        let t = TraceLog::disabled();
        let out = read(&i, None, "dir/Safety Manual - 1_0_3.pdf", 1, &t); // single space
        assert!(out.text.contains("safety body"), "resolved despite collapsed double space: {}", out.text);
        // Doubled / swapped separators (model over-escapes `\\` or uses `\`) also resolve.
        let out2 = read(&i, None, "dir\\\\Safety Manual - 1_0_3.pdf", 1, &t);
        assert!(out2.text.contains("safety body"), "resolved despite doubled backslash: {}", out2.text);
    }

    #[test]
    fn grep_and_glob_render() {
        let (_d, i) = idx();
        let t = TraceLog::disabled();
        assert!(grep(&i, "maxTsdr", &crate::grep::GrepOpts::default(), &t).contains(":#7:"));
        assert_eq!(grep(&i, "nomatchzzz", &crate::grep::GrepOpts::default(), &t), "(no matches)");
        assert!(glob(&i, "*АБАК*", &t).contains("АБАК.pdf  (7 chunks)"));
        assert!(glob(&i, "*nomatch*", &t).starts_with("(no documents match"));
    }

    // ── graph tool tests ────────────────────────────────────────────────────

    use crate::graph::store::{GraphStore, Node, Edge, Provenance};
    use crate::graph::ontology::Ontology;

    const GRAPH_ONT: &str = r#"
[entities.Organization]
props = ["name"]
[entities.Document]
props = []
[relations.PARTY_TO]
from = ["Organization"]
to = ["Document"]
[validation]
strict = true
"#;

    fn graph_prov() -> Provenance {
        Provenance {
            source_path: "contract.docx".into(),
            range: None,
            file_sig: None,
            origin: "agent".into(),
            confidence: 0.9,
            created_at: 0,
        }
    }

    fn graph_fixture() -> (tempfile::TempDir, GraphStore) {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(GRAPH_ONT).unwrap();
        let org = Node {
            id: "org:acme".into(),
            node_type: "Organization".into(),
            label: "Acme Corp".into(),
            aliases: vec!["ACME".into()],
            prov: graph_prov(),
        };
        let doc = Node {
            id: "contract.docx".into(),
            node_type: "Document".into(),
            label: "contract.docx".into(),
            aliases: vec![],
            prov: graph_prov(),
        };
        let edge = Edge {
            from: "org:acme".into(),
            to: "contract.docx".into(),
            edge_type: "PARTY_TO".into(),
            prov: graph_prov(),
        };
        g.upsert(&ont, &[org, doc], &[edge]).unwrap();
        (dir, g)
    }

    #[test]
    fn glossary_resolves_alias_and_returns_sentinel_on_miss() {
        let (_dir, g) = graph_fixture();
        let idir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(idir.path()).unwrap();
        let t = TraceLog::disabled();
        // "org:acme" is type Organization (unknown to node_ref) → falls back to raw id
        let result = glossary(&idx, &g, "ACME", &ChainSpec::default(), &t);
        assert!(result.contains("org:acme"), "expected node id in result, got: {result}");
        assert_eq!(glossary(&idx, &g, "nonesuch", &ChainSpec::default(), &t), "(no matches)");
    }

    #[test]
    fn glossary_surfaces_community_centrality_after_generalize() {
        let (_dir, g) = graph_fixture();
        let idir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(idir.path()).unwrap();
        let t = TraceLog::disabled();

        // Before the pass there is no node_meta → output is unannotated (back-compat).
        let before = glossary(&idx, &g, "ACME", &ChainSpec::default(), &t);
        assert!(!before.contains("comm "), "no meta annotation before generalize: {before}");

        // Run the generalization pass → community/centrality land in node_meta.
        let opts = crate::graph::generalize::apply::Opts::defaults(1);
        crate::graph::generalize::apply::generalize(&g, &opts).unwrap();

        let after = glossary(&idx, &g, "ACME", &ChainSpec::default(), &t);
        assert!(after.contains("org:acme"), "still shows the node id: {after}");
        assert!(after.contains("comm "), "glossary surfaces community after generalize: {after}");
    }

    #[test]
    fn glossary_shows_reasoning_node_edges() {
        use crate::graph::store::{Edge, GraphStore, Node, Provenance};
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(idir.path()).unwrap();
        let t = TraceLog::disabled();
        let prov = || Provenance {
            source_path: "c.md".into(),
            range: None,
            file_sig: None,
            origin: "agent".into(),
            confidence: 1.0,
            created_at: 1,
        };
        let node = |id: &str, ty: &str, label: &str| Node {
            id: id.into(),
            node_type: ty.into(),
            label: label.into(),
            aliases: vec![],
            prov: prov(),
        };
        g.put_node(&node("sym:loss", "Symptom", "Потеря связи")).unwrap();
        g.put_node(&node("cau:maxtsdr", "Cause", "Малый maxTsdr")).unwrap();
        g.put_edge(&Edge {
            from: "sym:loss".into(),
            to: "cau:maxtsdr".into(),
            edge_type: "CAUSED_BY".into(),
            prov: prov(),
        })
        .unwrap();

        // glossary on the Symptom walks the spine and surfaces the connected Cause inline,
        // so the agent sees the causal chain without a separate call.
        let out = glossary(&idx, &g, "Потеря связи", &spine_spec(), &t);
        assert!(out.contains("sym:loss"), "shows the matched node id: {out}");
        assert!(out.contains("→ CAUSED_BY") && out.contains("[Cause]"), "walks the CAUSED_BY hop: {out}");
        assert!(out.contains("Малый maxTsdr"), "shows the connected Cause label: {out}");
    }

    // Structural chunk-navigation tests (NEXT/CHILD/PARENT/PREV/REFERENCES, node_meta on a
    // section) were removed when `neighbors` became the generalization view (SIMILAR cross-links);
    // adjacent-chunk navigation now lives in `read`'s prev/next footer. The generalization
    // behavior is covered by `neighbors_shows_similar_generalization_for_a_node_id` and the
    // MCP≡eval parity test in eval/src/backend/glossa_tools.rs.

    #[test]
    fn glossary_renders_section_nodes_as_path_ord() {
        use crate::index::store::index_dir;

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("g.md"), b"# Intro\nhello\n## Details\nworld\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        let t = TraceLog::disabled();

        // "Intro" is a Section node; glossary should render it as "path  #1 · Intro".
        let result = glossary(&idx, &g, "Intro", &ChainSpec::default(), &t);
        assert!(result.contains("#1"), "section rendered with ord: {result}");
        assert!(result.contains("Intro"), "section label present: {result}");

        assert_eq!(glossary(&idx, &g, "nonexistentzzz", &ChainSpec::default(), &t), "(no matches)");
    }
}
