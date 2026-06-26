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

/// Read chunk `n` of `path`: full stored body + a unified prev/next footer, plus extracted images
/// (empty if the source file is absent — body still comes from the index). No truncation.
pub fn read(idx: &DocIndex, path: &str, n: u64, trace: &TraceLog) -> ReadOut {
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
    let images = crate::read::extract_images(std::path::Path::new(path), n, 4).unwrap_or_default();
    ReadOut { text: format!("{}{}", chunk.body, footer), images }
}

/// List documents by path mask; model text only.
pub fn glob(idx: &DocIndex, pattern: &str, trace: &TraceLog) -> String {
    match crate::glob::glob_docs(idx, pattern) {
        Ok(docs) => {
            trace.log("glob", json!({"pattern": pattern}), json!({"docs": docs.len()}));
            if docs.is_empty() { "(no documents match)".to_string() }
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

/// Resolve a name/term to graph node ids and render each as a `(path, #ord)` ref.
/// Section nodes render as `"path  #ord · label"`, Documents as `"path  (document)"`,
/// unresolved ids fall back to the raw id string. Empty → `"(no matches)"`.
pub fn glossary(idx: &DocIndex, g: &crate::graph::store::GraphStore, name: &str, trace: &TraceLog) -> String {
    match g.resolve(name) {
        Ok(ids) => {
            trace.log("glossary", json!({ "name": name }), json!({ "ids": ids.len() }));
            if ids.is_empty() {
                return "(no matches)".to_string();
            }
            let lines: Vec<String> = ids
                .iter()
                .map(|id| match g.get_node(id) {
                    Ok(Some(node)) => node_ref(idx, &node).unwrap_or_else(|| id.clone()),
                    _ => id.clone(),
                })
                .collect();
            lines.join("\n")
        }
        Err(e) => format!("glossary error: {e}"),
    }
}

/// Graph structural neighbors of chunk `n` in `path`, rendered in `(path, #ord)` address space.
/// Walks the section's own outgoing edges plus the parent document's REFERENCES edges.
/// Section targets render as `"EDGE_TYPE  path  #ord · label"`, Document targets as
/// `"EDGE_TYPE  path  (document)"`. Out-of-range `n` → `"no chunk #n in path"`.
/// No linked sections → `"(no linked sections)"`.
pub fn neighbors(idx: &DocIndex, g: &crate::graph::store::GraphStore, path: &str, n: u64, depth: usize, trace: &TraceLog) -> String {
    let _ = depth; // v1: direct neighbors only
    let location = match idx.location_for_ord(path, n) {
        Ok(Some(l)) => l,
        Ok(None) => return format!("no chunk #{n} in {path}"),
        Err(e) => return format!("neighbors error: {e}"),
    };
    let sec_id = crate::graph::build::section_id(path, &location);
    // Section's own edges, plus the parent document's REFERENCES (cross-doc links).
    let mut edges = g.outgoing(&sec_id).unwrap_or_default();
    if let Ok(doc_edges) = g.outgoing(path) {
        edges.extend(doc_edges.into_iter().filter(|e| e.edge_type == "REFERENCES"));
    }
    let mut lines = Vec::new();
    for e in &edges {
        let Ok(Some(node)) = g.get_node(&e.to) else { continue };
        let Some(ref_str) = node_ref(idx, &node) else { continue };
        lines.push(format!("{}  {}", e.edge_type, ref_str));
    }
    trace.log("neighbors", json!({"path": path, "n": n}), json!({"links": lines.len()}));
    if lines.is_empty() { "(no linked sections)".to_string() } else { lines.join("\n") }
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
        let out = read(&i, "d.md", 1, &t);
        assert!(out.text.contains(&big), "full body, no cap");                 // not truncated
        assert!(out.text.contains("next #2") && out.text.contains("end of document") == false);
        assert!(out.text.contains("‹ start of document · next #2 ›"));        // unified footer (MCP wording)
        // Out-of-range read reports the valid chunk range so the model can self-correct.
        let oor = read(&i, "d.md", 99, &t).text;
        assert!(oor.contains("no chunk #99 in d.md") && oor.contains("2 chunks") && oor.contains("#1..#2"), "range hint: {oor}");
        // A wrong path reports that the document isn't indexed.
        assert!(read(&i, "nope.md", 1, &t).text.contains("no document indexed at nope.md"));
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
        let out = read(&i, "dir/Safety Manual - 1_0_3.pdf", 1, &t); // single space
        assert!(out.text.contains("safety body"), "resolved despite collapsed double space: {}", out.text);
    }

    #[test]
    fn grep_and_glob_render() {
        let (_d, i) = idx();
        let t = TraceLog::disabled();
        assert!(grep(&i, "maxTsdr", &crate::grep::GrepOpts::default(), &t).contains(":#7:"));
        assert_eq!(grep(&i, "nomatchzzz", &crate::grep::GrepOpts::default(), &t), "(no matches)");
        assert!(glob(&i, "*АБАК*", &t).contains("АБАК.pdf  (7 chunks)"));
        assert_eq!(glob(&i, "*nomatch*", &t), "(no documents match)");
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
        let result = glossary(&idx, &g, "ACME", &t);
        assert!(result.contains("org:acme"), "expected node id in result, got: {result}");
        assert_eq!(glossary(&idx, &g, "nonesuch", &t), "(no matches)");
    }

    // ── structural (path, #n) address-space tests ───────────────────────────

    #[test]
    fn neighbors_speaks_path_ord_address_space() {
        use crate::index::store::index_dir;

        // Build a nested-heading document: chunk #1 = "A", #2 = "A > B", #3 = "A > C".
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nintro\n## B\nbody b\n## C\nbody c\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        let p = dir.path().join("a.md").to_string_lossy().to_string();
        let t = TraceLog::disabled();

        // Chunk #1 "A": outgoing edges include NEXT/CHILD to "A > B" which is #2.
        let n1 = neighbors(&idx, &g, &p, 1, 1, &t);
        assert!(
            n1.lines().any(|l| (l.contains("NEXT") || l.contains("CHILD")) && l.contains("#2")),
            "A's neighbors include a NEXT/CHILD edge to #2: {n1}"
        );

        // Chunk #2 "A > B": outgoing edges include PARENT/PREV back to "A" which is #1.
        let n2 = neighbors(&idx, &g, &p, 2, 1, &t);
        assert!(
            n2.lines().any(|l| (l.contains("PARENT") || l.contains("PREV")) && l.contains("#1")),
            "B's neighbors include a PARENT/PREV edge to #1: {n2}"
        );

        // A returned #ord from n2 should actually be readable.
        let parent_ord: u64 = n2
            .lines()
            .find(|l| l.contains("PARENT") || l.contains("PREV"))
            .and_then(|l| l.split('#').nth(1))
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse().ok())
            .expect("could not parse #ord from neighbors output");
        let read_out = read(&idx, &p, parent_ord, &t);
        assert!(
            read_out.text.contains("intro") || !read_out.text.starts_with("no chunk"),
            "reading #ord from neighbors output succeeds: {}",
            read_out.text
        );

        // Out-of-range n.
        let oor = neighbors(&idx, &g, &p, 999, 1, &t);
        assert!(oor.starts_with("no chunk #999"), "out-of-range: {oor}");

        // Isolated single-section document → no structural edges.
        let dir2 = tempfile::tempdir().unwrap();
        std::fs::write(dir2.path().join("lone.md"), b"# Solo\njust one section\n").unwrap();
        index_dir(dir2.path(), true).unwrap();
        let idx2 = DocIndex::open_or_create(dir2.path()).unwrap();
        let g2 = crate::graph::store::GraphStore::open(dir2.path()).unwrap();
        let p2 = dir2.path().join("lone.md").to_string_lossy().to_string();
        let lone = neighbors(&idx2, &g2, &p2, 1, 1, &t);
        assert_eq!(lone, "(no linked sections)", "isolated section: {lone}");
    }

    #[test]
    fn neighbors_includes_cross_doc_references() {
        use crate::index::store::index_dir;

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), b"# A\nsee [the manual](b.md)\n").unwrap();
        std::fs::write(dir.path().join("b.md"), b"# B\ncontent\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        let pa = dir.path().join("a.md").to_string_lossy().to_string();
        let pb = dir.path().join("b.md").to_string_lossy().to_string();
        let t = TraceLog::disabled();

        // Chunk #1 of a.md has a cross-doc REFERENCES edge to b.md (Document).
        let n1 = neighbors(&idx, &g, &pa, 1, 1, &t);
        assert!(
            n1.lines().any(|l| l.contains("REFERENCES") && l.contains(&pb)),
            "a.md #1 neighbors include REFERENCES to b.md: {n1}"
        );
    }

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
        let result = glossary(&idx, &g, "Intro", &t);
        assert!(result.contains("#1"), "section rendered with ord: {result}");
        assert!(result.contains("Intro"), "section label present: {result}");

        assert_eq!(glossary(&idx, &g, "nonexistentzzz", &t), "(no matches)");
    }
}
