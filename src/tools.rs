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
    let chunk = match idx.read_chunk_by_ord(path, n) {
        Ok(Some(c)) => c,
        Ok(None) => return ReadOut { text: format!("no chunk #{n} in {path}"), images: Vec::new() },
        Err(e) => return ReadOut { text: format!("read error: {e}"), images: Vec::new() },
    };
    trace.log("read", json!({"path": path, "n": n}), json!({"path": path}));
    let footer = match (chunk.prev, chunk.next) {
        (Some(p), Some(nx)) => format!("\n\n‹ prev #{p} · next #{nx} ›"),
        (None, Some(nx)) => format!("\n\n‹ start of document · next #{nx} ›"),
        (Some(p), None) => format!("\n\n‹ prev #{p} · end of document ›"),
        (None, None) => String::new(),
    };
    let images = crate::read::extract_images(std::path::Path::new(path), 8).unwrap_or_default();
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

/// Resolve a name/term to graph node ids (the "glossary" lookup). Model text only.
pub fn glossary(g: &crate::graph::store::GraphStore, name: &str, trace: &TraceLog) -> String {
    match g.resolve(name) {
        Ok(ids) => {
            trace.log("glossary", json!({ "name": name }), json!({ "ids": ids.len() }));
            if ids.is_empty() { "(no matches)".to_string() } else { ids.join("\n") }
        }
        Err(e) => format!("glossary error: {e}"),
    }
}

/// Graph neighbors reachable from a node id, up to `depth` hops. Model text only.
pub fn neighbors(g: &crate::graph::store::GraphStore, node_id: &str, depth: usize, trace: &TraceLog) -> String {
    match crate::graph::traverse::neighbors(g, node_id, None, depth) {
        Ok(ids) => {
            trace.log("neighbors", json!({ "node_id": node_id, "depth": depth }), json!({ "ids": ids.len() }));
            if ids.is_empty() { "(no neighbors)".to_string() } else { ids.join("\n") }
        }
        Err(e) => format!("neighbors error: {e}"),
    }
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
        assert_eq!(read(&i, "d.md", 99, &t).text, "no chunk #99 in d.md");
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
        let t = TraceLog::disabled();
        let result = glossary(&g, "ACME", &t);
        assert!(result.contains("org:acme"), "expected node id in result, got: {result}");
        assert_eq!(glossary(&g, "nonesuch", &t), "(no matches)");
    }

    #[test]
    fn neighbors_returns_connected_and_sentinel_on_isolated() {
        let (_dir, g) = graph_fixture();
        let t = TraceLog::disabled();
        let result = neighbors(&g, "org:acme", 1, &t);
        assert!(result.contains("contract.docx"), "expected neighbor in result, got: {result}");
        assert_eq!(neighbors(&g, "isolated", 1, &t), "(no neighbors)");
    }
}
