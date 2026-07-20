//! Single source of truth for the agent tools' model-facing output. Both the MCP server
//! (src/mcp.rs) and the kb-eval harness call these so prod and eval render identically.

use crate::grep::GrepOpts;
use crate::index::store::{DocIndex, RankedHit};
use crate::trace::TraceLog;
use serde_json::json;

/// Indexed document path suggestions for a failed path lookup (basename glob, up to 3).
pub fn document_path_hints(idx: &DocIndex, path: &str) -> String {
    let base = path.rsplit(['\\', '/']).next().unwrap_or(path);
    if base.len() < 4 {
        return String::new();
    }
    crate::glob::glob_docs(idx, &format!("*{base}*"))
        .ok()
        .map(|hits| {
            let hints: Vec<String> = hits.into_iter().take(3).map(|(p, _)| p).collect();
            crate::cli_fmt::format_did_you_mean(&hints)
        })
        .unwrap_or_default()
}

fn path_not_found(idx: &DocIndex, path: &str) -> String {
    format!(
        "no document indexed at {path} — check the path from a search/glob result{}",
        document_path_hints(idx, path)
    )
}

/// BM25 search (optionally scoped). Returns (model text, hits for the caller's scoring).
pub fn search(
    idx: &DocIndex,
    query: &str,
    limit: usize,
    glob: Option<&str>,
    file_type: Option<&str>,
    trace: &TraceLog,
) -> (String, Vec<RankedHit>) {
    match idx.search_filtered(query, limit.max(1), glob, file_type) {
        Ok(hits) => {
            let th: Vec<_> = hits
                .iter()
                .map(|h| json!({"path": h.path, "location": h.location, "score": h.score}))
                .collect();
            trace.log("search", json!({"query": query}), json!(th));
            let body = if hits.is_empty() {
                "(no results)".to_string()
            } else {
                hits.iter()
                    .map(|h| h.display_line())
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            (body, hits)
        }
        Err(e) => (format!("search error: {e}"), Vec::new()),
    }
}

/// A notebook file's line-filtered grep: iterate `body.lines()`, keeping lines that match
/// `pattern` under the same flag semantics (`fixed`/`ignore_case`/`word`) as the corpus grep —
/// reusing its `prepare`/matcher so a notebook and a corpus grep agree on what counts as a
/// match. Notebook files are short (no chunking), so every hit renders against a fixed `#1`,
/// mirroring the corpus format `path:#ord: line`.
#[cfg(feature = "notebook")]
fn grep_notebook_body(rel: &str, body: &str, pattern: &str, opts: &GrepOpts) -> String {
    if pattern.trim().is_empty() {
        return "(no matches)".to_string();
    }
    let (pattern, opts) = crate::grep::prepare(pattern, opts);
    let matcher = match crate::grep::build_matcher(&pattern, &opts) {
        Ok(m) => m,
        Err(e) => return format!("grep error: {e}"),
    };
    let hits: Vec<String> = body
        .lines()
        .filter(|l| matcher.is_match(l))
        .map(|l| format!("{rel}:#1: {}", l.trim()))
        .collect();
    if hits.is_empty() {
        "(no matches)".to_string()
    } else {
        hits.join("\n")
    }
}

/// ripgrep-style literal/regex search; model text only. Also serves notebook files
/// (feature-gated): a doc-scoped `opts.path` like `<doc>/<file>` is probed against the agent's
/// on-disk notebook before falling through to the corpus grep below.
pub fn grep(root: &std::path::Path, idx: &DocIndex, pattern: &str, opts: &GrepOpts, trace: &TraceLog) -> String {
    #[cfg(feature = "notebook")]
    if let Some(p) = opts.path.as_deref() {
        if looks_like_notebook_path(p) {
            if let Ok((rel, body)) = crate::notebook::read_note(root, idx, p) {
                trace.log("grep", json!({ "notebook": p, "pattern": pattern }), json!({ "notebook": rel }));
                return grep_notebook_body(&rel, &body, pattern, opts);
            }
        }
    }
    match crate::grep::grep(idx, pattern, opts) {
        Ok(hits) => {
            trace.log(
                "grep",
                json!({"pattern": pattern}),
                json!({"hits": hits.len()}),
            );
            if hits.is_empty() {
                "(no matches)".to_string()
            } else {
                hits.iter()
                    .map(|h| h.display_line())
                    .collect::<Vec<_>>()
                    .join("\n")
            }
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
fn gather_mentions(
    idx: &DocIndex,
    g: &crate::graph::store::GraphStore,
    nid: &str,
    attr: &str,
    seen: &mut std::collections::HashSet<(String, u64)>,
    out: &mut Vec<(String, String, u64)>,
) {
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
fn read_node(
    idx: &DocIndex,
    g: &crate::graph::store::GraphStore,
    node: crate::graph::store::Node,
) -> ReadOut {
    let mut seen = std::collections::HashSet::new();
    let mut chunks: Vec<(String, String, u64)> = Vec::new();
    gather_mentions(idx, g, &node.id, "MENTIONS", &mut seen, &mut chunks); // the node's own evidence
    let label_of = |nid: &str| {
        g.get_node(nid)
            .ok()
            .flatten()
            .map(|n| n.label)
            .unwrap_or_else(|| nid.to_string())
    };
    // 1-hop reasoning neighbours (outgoing →, incoming ←), skipping the MENTIONS-to-section links.
    for e in g.outgoing(&node.id).unwrap_or_default() {
        if e.edge_type != crate::graph::MENTIONS && !e.to.contains('#') {
            gather_mentions(
                idx,
                g,
                &e.to,
                &format!("via → {} {}", e.edge_type, label_of(&e.to)),
                &mut seen,
                &mut chunks,
            );
        }
    }
    for e in g.incoming(&node.id).unwrap_or_default() {
        if e.edge_type != crate::graph::MENTIONS && !e.from.contains('#') {
            gather_mentions(
                idx,
                g,
                &e.from,
                &format!("via ← {} {}", e.edge_type, label_of(&e.from)),
                &mut seen,
                &mut chunks,
            );
        }
    }
    let mut text = format!("{}  [{}]  {}", node.id, node.node_type, node.label);
    if chunks.is_empty() {
        text.push_str("\n(no source chunk linked to this node)");
    }
    let mut images = Vec::new();
    for (attr, p, ord) in &chunks {
        if let Ok(Some(c)) = idx.read_chunk_by_ord(p, *ord) {
            text.push_str(&format!(
                "\n\n── {attr} · {p} #{ord} ──\n{}",
                format_chunk_body(*ord, &c.body)
            ));
        }
        images.extend(crate::read::extract_images(&idx.doc_file(p), *ord, 4).unwrap_or_default());
    }
    ReadOut { text, images }
}

/// Human-readable body for a chunk whose stored text may be empty (blank PDF page).
pub fn format_chunk_body(ord: u64, body: &str) -> String {
    if body.trim().is_empty() {
        format!("(page #{ord} — no extractable text)")
    } else {
        body.to_string()
    }
}

/// A notebook file is addressed doc-scoped as `<doc>/<file>`: it has a path separator and a
/// note-file suffix. Corpus paths (a bare doc, a search result) don't match, so they skip the
/// notebook probe entirely.
#[cfg(feature = "notebook")]
pub fn looks_like_notebook_path(path: &str) -> bool {
    path.contains('/')
        && matches!(
            std::path::Path::new(path).extension().and_then(|e| e.to_str()),
            Some("csp" | "md" | "txt" | "json")
        )
}

/// Read chunk `n` of `path`: full stored body + a unified prev/next footer, plus extracted images
/// (empty if the source file is absent — body still comes from the index). No truncation. Omnivorous:
/// if `path` is a graph NODE id, returns that node + its (1-hop) evidence chunks (see `read_node`).
/// Also serves notebook files (feature-gated): a doc-scoped path like `<doc>/<file>` is probed
/// against the agent's on-disk notebook before falling through to the corpus read below.
pub fn read(
    root: &std::path::Path,
    idx: &DocIndex,
    graph: Option<&crate::graph::store::GraphStore>,
    path: &str,
    n: u64,
    page_image: bool,
    trace: &TraceLog,
) -> ReadOut {
    #[cfg(feature = "notebook")]
    if looks_like_notebook_path(path) {
        if page_image {
            return ReadOut {
                text: format!("page_image is only supported for PDF (got {path})"),
                images: Vec::new(),
            };
        }
        if let Ok((rel, body)) = crate::notebook::read_note(root, idx, path) {
            trace.log("read", json!({ "notebook": path }), json!({ "notebook": rel }));
            return ReadOut { text: format!("{rel}\n{body}"), images: Vec::new() };
        }
    }
    // Omnivorous: a REASONING node id off a glossary line reads as the node + its evidence. A
    // structural node id IS a document path (e.g. a Document's id is its path), so it falls through
    // to the normal document read below.
    if let Some(g) = graph {
        if let Ok(Some(node)) = g.get_node(path) {
            if !crate::graph::STRUCTURAL_NODES.contains(&node.node_type.as_str()) {
                if page_image {
                    return ReadOut {
                        text: format!("page_image is only supported for PDF (got {path})"),
                        images: Vec::new(),
                    };
                }
                trace.log("read", json!({ "node": path }), json!({ "node": path }));
                return read_node(idx, g, node);
            }
        }
    }
    // Try the exact path first. If the document isn't found, the model may have mangled the path
    // (e.g. collapsed a double space when copying it) — resolve it tolerantly and retry once.
    let (path, chunk, resolved_ord): (String, _, u64) = match idx.read_chunk_by_ord(path, n) {
        Ok(Some(c)) => (path.to_string(), c, n),
        Ok(None) => match idx.last_chunk_ord(path) {
            // Document exists; the chunk number is out of range. Rather than return an error a
            // small model loops on (it reuses a chunk number from another doc — a 1-chunk file
            // has no #3), CLAMP to the valid range and read that chunk. Its own header shows the
            // real ordinal, so the model sees what it actually got.
            Ok(Some(max)) => {
                let ord = n.clamp(1, max);
                match idx.read_chunk_by_ord(path, ord) {
                    Ok(Some(c)) => (path.to_string(), c, ord),
                    _ => return ReadOut {
                        text: format!("no chunk #{n} in {path} — this document has {max} chunks (read #1..#{max})"),
                        images: Vec::new(),
                    },
                }
            }
            // No exact path match — try a tolerant resolve, then retry once.
            Ok(None) => match idx.canonical_document_path(path) {
                Some(real) => match idx.read_chunk_by_ord(&real, n) {
                    Ok(Some(c)) => (real, c, n),
                    _ => {
                        let text = match idx.last_chunk_ord(&real) {
                            Ok(Some(max)) => format!("no chunk #{n} in {real} — this document has {max} chunks (read #1..#{max})"),
                            _ => path_not_found(idx, path),
                        };
                        return ReadOut { text, images: Vec::new() };
                    }
                },
                None => return ReadOut {
                    text: path_not_found(idx, path),
                    images: Vec::new(),
                },
            },
            Err(e) => return ReadOut { text: format!("no chunk #{n} in {path} (range lookup failed: {e})"), images: Vec::new() },
        },
        Err(e) => return ReadOut { text: format!("read error: {e}"), images: Vec::new() },
    };
    let path = path.as_str();
    trace.log("read", json!({"path": path, "n": n}), json!({"path": path}));
    let doc_path = idx.doc_file(path);
    if page_image {
        let ext = doc_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if ext != "pdf" {
            return ReadOut {
                text: format!("page_image is only supported for PDF (got {path})"),
                images: Vec::new(),
            };
        }
        let images = match crate::read::render_pdf_page(&doc_path, resolved_ord) {
            Ok(Some(img)) => vec![img],
            _ => Vec::new(),
        };
        let text = if images.is_empty() {
            format!("page_image: failed to render {path} #{resolved_ord}")
        } else {
            format!("page_image {path} #{resolved_ord}")
        };
        return ReadOut { text, images };
    }
    let footer = match (chunk.prev, chunk.next) {
        (Some(p), Some(nx)) => format!("\n\n‹ prev #{p} · next #{nx} ›"),
        (None, Some(nx)) => format!("\n\n‹ start of document · next #{nx} ›"),
        (Some(p), None) => format!("\n\n‹ prev #{p} · end of document ›"),
        (None, None) => String::new(),
    };
    let images = crate::read::extract_images(&doc_path, resolved_ord, 4).unwrap_or_default();
    let body = format_chunk_body(resolved_ord, &chunk.body);
    ReadOut {
        text: format!("{}{}", body, footer),
        images,
    }
}

/// List documents by path mask; model text only.
pub fn glob(idx: &DocIndex, pattern: &str, trace: &TraceLog) -> String {
    match crate::glob::glob_docs(idx, pattern) {
        Ok(docs) => {
            trace.log(
                "glob",
                json!({"pattern": pattern}),
                json!({"docs": docs.len()}),
            );
            if docs.is_empty() {
                "(no documents match — ripgrep -g glob syntax: use * or **/* or *.{pdf,md}; matches PATHS not content; use search or grep for text)".to_string()
            } else {
                docs.iter()
                    .map(|(p, n)| format!("{p}  ({n} chunks)"))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
        Err(e) => format!("glob error: {e}"),
    }
}

/// Notebook: create, replace, or (with `append`) extend a note bound to an indexed document.
#[cfg(feature = "notebook")]
pub fn note(
    root: &std::path::Path,
    idx: &crate::index::store::DocIndex,
    doc: &str,
    file: &str,
    content: &str,
    append: bool,
) -> String {
    crate::notebook::note(root, idx, doc, file, content, append)
}

#[cfg(feature = "notebook")]
pub fn ls_notes(
    root: &std::path::Path,
    idx: &crate::index::store::DocIndex,
    doc: Option<&str>,
) -> String {
    crate::notebook::ls(root, idx, doc)
}

#[cfg(feature = "notebook")]
pub fn del_note(root: &std::path::Path, idx: &crate::index::store::DocIndex, path: &str) -> String {
    crate::notebook::del(root, idx, path)
}

/// Compile agent `.csp` limit tables for `doc` into the constraint graph (`kb graph build`).
#[cfg(feature = "constraint")]
pub fn graph_build(
    root: &std::path::Path,
    idx: &crate::index::store::DocIndex,
    g: &crate::graph::store::GraphStore,
    ont: &crate::graph::ontology::Ontology,
    doc: &str,
    tables_dir: Option<&std::path::Path>,
) -> String {
    use crate::notebook::{mirror_dir_for_doc, notes_root};
    use crate::tables::{count_csp_files, format_graph_build_output, tables_to_graph};

    if count_csp_files(root, doc) == 0 {
        return "graph_build FAILED\n  no .csp files — materialize tables on step 2 first\n  fix: note(doc, file=\"….csp\", content=…) then graph_build again\n".into();
    }
    let tables = tables_dir
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| notes_root(root).join(mirror_dir_for_doc(doc)));
    format_graph_build_output(tables_to_graph(idx, g, ont, doc, &tables))
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
    let Ok(Some(m)) = g.node_meta(id) else {
        return String::new();
    };
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
        ChainSpec {
            spine_rels: ont.spine_relations(),
        }
    }
}

impl Default for ChainSpec {
    /// No spines — `glossary` prints just the node head. Used by callers/tests that don't drive
    /// the reasoning chain.
    fn default() -> Self {
        ChainSpec {
            spine_rels: Vec::new(),
        }
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
        let Ok(Some(node)) = g.get_node(&e.to) else {
            continue;
        };
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
pub fn glossary(
    idx: &DocIndex,
    g: &crate::graph::store::GraphStore,
    name: &str,
    spec: &ChainSpec,
    trace: &TraceLog,
) -> String {
    match g.resolve(name) {
        Ok(ids) => {
            trace.log(
                "glossary",
                json!({ "name": name }),
                json!({ "ids": ids.len() }),
            );
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
                                // Node ids are label-only, so the owner document is not in the id.
                                // Always show it (`@<doc>`) — the read-anchor doc is conditional on a
                                // MENTIONS edge and may point to a *different* doc (cross-standard
                                // reference), so the owner is not otherwise guaranteed in the head.
                                let owner = if node.prov.source_path.is_empty() {
                                    String::new()
                                } else {
                                    format!("  @{}", node.prov.source_path)
                                };
                                let head = format!(
                                    "{base}{owner}{}{}",
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
/// "related cases" written by the generalize pass — plus same-community siblings (other reasoning
/// nodes in the same connected component, top by PageRank). The target is either an explicit
/// reasoning-node `id` (copied from a `glossary` line) or a chunk `(path, n)` that resolves to
/// its Section node. Each related node renders with a `read` anchor. No links →
/// `"(no related cases)"`. (Spine chains live in `glossary`; this is the cross-case view.)
/// Top community siblings / stats examples per cluster (PageRank-ranked).
const COMMUNITY_TOP_LIMIT: usize = 8;

pub fn neighbors(
    idx: &DocIndex,
    g: &crate::graph::store::GraphStore,
    node: Option<&str>,
    path: Option<&str>,
    n: Option<u64>,
    trace: &TraceLog,
) -> String {
    let id: String = if let Some(nid) = node.filter(|s| !s.trim().is_empty()) {
        nid.to_string()
    } else if let (Some(p), Some(nn)) = (path, n) {
        // `#0` → first chunk: numbering is 1-based, but the model often writes 0 out of
        // programming habit. Tolerate it instead of erroring so it doesn't loop guessing.
        let nn = if nn == 0 { 1 } else { nn };
        match idx.location_for_ord(p, nn) {
            Ok(Some(loc)) => crate::graph::build::section_id(p, &loc),
            Ok(None) => {
                return format!(
                    "no chunk #{nn} in {p}; chunk numbers start at #1 — take it from a search/grep/read"
                )
            }
            Err(e) => return format!("neighbors error: {e}"),
        }
    } else {
        return "neighbors needs a node id (from glossary) or a (path, n) chunk".to_string();
    };
    let mut seen = std::collections::HashSet::new();
    let mut lines = Vec::new();
    for e in g.outgoing(&id).unwrap_or_default() {
        if e.edge_type == "SIMILAR" && seen.insert(e.to.clone()) {
            lines.push(format!(
                "SIMILAR  {}{}{}",
                endpoint_ref(idx, g, &e.to),
                meta_suffix(g, &e.to),
                read_anchor(idx, g, &e.to)
            ));
        }
    }
    for e in g.incoming(&id).unwrap_or_default() {
        if e.edge_type == "SIMILAR" && seen.insert(e.from.clone()) {
            lines.push(format!(
                "SIMILAR  {}{}{}",
                endpoint_ref(idx, g, &e.from),
                meta_suffix(g, &e.from),
                read_anchor(idx, g, &e.from)
            ));
        }
    }
    let similar_count = lines.len();
    if let Ok(Some(meta)) = g.node_meta(&id) {
        if let Some(comm) = meta.community {
            if let Ok(siblings) = g.community_siblings(comm, &id, COMMUNITY_TOP_LIMIT) {
                for (sib_id, _) in siblings {
                    if seen.insert(sib_id.clone()) {
                        lines.push(format!(
                            "COMMUNITY  {}{}{}",
                            endpoint_ref(idx, g, &sib_id),
                            meta_suffix(g, &sib_id),
                            read_anchor(idx, g, &sib_id),
                        ));
                    }
                }
            }
        }
    }
    trace.log(
        "neighbors",
        json!({"id": id}),
        json!({"similar": similar_count, "community": lines.len() - similar_count}),
    );
    if lines.is_empty() {
        "(no related cases)".to_string()
    } else {
        lines.join("\n")
    }
}

fn stats_example_line(
    g: &crate::graph::store::GraphStore,
    id: &str,
    meta: &crate::graph::store::NodeMeta,
) -> String {
    match g.get_node(id) {
        Ok(Some(n)) => {
            let pr = meta
                .pagerank
                .map(|p| format!("  · pr {p:.3}"))
                .unwrap_or_default();
            format!("  {}  [{}]  {}{}", n.id, n.node_type, n.label, pr)
        }
        _ => format!("  {id}"),
    }
}

/// Domain-independent breakdown of the graph: how many nodes of each `node_type`
/// and how many edges of each `edge_type` exist. Nothing here is hardcoded to a
/// domain — the type/relation names come from whatever ontology populated the
/// store (e.g. `nodes: Enum 8, Field 10, Section 40 | edges: CONSTRAINED_BY 8,
/// CONTAINS 40, MENTIONS 10`).
pub fn graph_type_counts(g: &crate::graph::store::GraphStore) -> String {
    use std::collections::BTreeMap;
    let mut node_types: BTreeMap<String, usize> = BTreeMap::new();
    let mut edge_types: BTreeMap<String, usize> = BTreeMap::new();
    for n in g.all_nodes().unwrap_or_default() {
        *node_types.entry(n.node_type).or_default() += 1;
    }
    for e in g.all_edges().unwrap_or_default() {
        *edge_types.entry(e.edge_type).or_default() += 1;
    }
    let join = |m: &BTreeMap<String, usize>| {
        if m.is_empty() {
            "(none)".to_string()
        } else {
            m.iter()
                .map(|(k, v)| format!("{k} {v}"))
                .collect::<Vec<_>>()
                .join(", ")
        }
    };
    format!(
        "by type — nodes: {} | edges: {}",
        join(&node_types),
        join(&edge_types)
    )
}

/// Universal "everything about one node" view: identity (`id`, `node_type`,
/// `label`, `aliases`) plus every OUTGOING and INCOMING edge with the other
/// endpoint's label. Fully domain-independent — used by `graph_stats(node=…)`.
pub fn node_inspect(g: &crate::graph::store::GraphStore, id: &str) -> String {
    let node = match g.get_node(id) {
        Ok(Some(n)) => n,
        Ok(None) => return format!("node {id}: not found"),
        Err(e) => return format!("node {id}: error: {e}"),
    };
    let label_of = |other: &str| match g.get_node(other) {
        Ok(Some(n)) => n.label,
        _ => "?".to_string(),
    };
    let aliases = if node.aliases.is_empty() {
        "—".to_string()
    } else {
        node.aliases.join(", ")
    };
    // Node ids are label-only, so always show the owner document — otherwise the
    // same field label across standards is indistinguishable in the head.
    let doc = if node.prov.source_path.is_empty() {
        "—"
    } else {
        node.prov.source_path.as_str()
    };
    let mut out = format!(
        "node {}\ntype: {}\ndoc: {}\nlabel: {}\naliases: {}",
        node.id, node.node_type, doc, node.label, aliases
    );
    let edges = g.all_edges().unwrap_or_default();
    let mut outgoing: Vec<String> = edges
        .iter()
        .filter(|e| e.from == node.id)
        .map(|e| format!("  -{}-> {} ({})", e.edge_type, e.to, label_of(&e.to)))
        .collect();
    let mut incoming: Vec<String> = edges
        .iter()
        .filter(|e| e.to == node.id)
        .map(|e| format!("  {} ({}) -{}->", e.from, label_of(&e.from), e.edge_type))
        .collect();
    outgoing.sort();
    incoming.sort();
    out.push_str(&format!("\noutgoing ({}):", outgoing.len()));
    out.push_str(&if outgoing.is_empty() {
        "\n  —".to_string()
    } else {
        format!("\n{}", outgoing.join("\n"))
    });
    out.push_str(&format!("\nincoming ({}):", incoming.len()));
    out.push_str(&if incoming.is_empty() {
        "\n  —".to_string()
    } else {
        format!("\n{}", incoming.join("\n"))
    });
    out
}

/// Graph node/edge counts and, when `node_meta` exists, a per-community overview with up to
/// [`COMMUNITY_TOP_LIMIT`] example nodes ranked by PageRank. The second line is a
/// domain-independent breakdown by node/edge type ([`graph_type_counts`]).
pub fn graph_stats(g: &crate::graph::store::GraphStore) -> String {
    let nodes = g.node_count().unwrap_or(0);
    let edges = g.edge_count().unwrap_or(0);
    let mut out = format!("nodes: {nodes}, edges: {edges}\n{}", graph_type_counts(g));
    if g.node_meta_count().unwrap_or(0) == 0 {
        return format!("{out}\ncommunities: (none)");
    }
    let sizes = g.community_sizes().unwrap_or_default();
    if sizes.is_empty() {
        return format!("{out}\ncommunities: (none)");
    }
    out.push_str(&format!("\ncommunities: {}", sizes.len()));
    for (comm, size) in sizes {
        out.push_str(&format!("\n\ncomm {comm}  ({size} nodes)"));
        for (id, meta) in g
            .community_top_nodes(comm, COMMUNITY_TOP_LIMIT)
            .unwrap_or_default()
        {
            out.push('\n');
            out.push_str(&stats_example_line(g, &id, &meta));
        }
    }
    out
}

/// Doc-scoped inventory for `graph_stats(doc=…)`: non-structural nodes owned by
/// `doc` (`source_path`) with all outgoing edges. Ontology-independent.
pub fn checklist_coverage_report(
    g: &crate::graph::store::GraphStore,
    doc: &str,
) -> String {
    match crate::graph::ops::doc_owned_inventory(g, doc) {
        Ok(Some(inv)) => {
            let mut lines = vec![format!("owned({doc}): {} nodes", inv.nodes.len())];
            for n in &inv.nodes {
                lines.push(format!("{}  [{}]  {}", n.id, n.node_type, n.label));
                for e in &n.outgoing {
                    lines.push(format!("  {} → {}", e.edge_type, e.to));
                }
            }
            lines.join("\n")
        }
        Ok(None) => format!("owned({doc}): (none)"),
        Err(e) => format!("owned({doc}): error: {e}"),
    }
}

/// Default per-file delivery cap for `get_source_file`, matching ZeroClaw's ACP `deliver_file`
/// limit (10 MB). Bytes above this degrade to a page slice or a structured error.
pub const DEFAULT_SOURCE_MAX_BYTES: u64 = 10 * 1024 * 1024;

/// The deliverable source artifact: original bytes (or an extracted PDF page) plus a display name.
/// Never carries provenance text — that rides on `SourceFileOut::text`.
pub struct SourceBlob {
    pub filename: String,
    pub mime: String,
    pub bytes: Vec<u8>,
}

/// Result of resolving a citation to its backing file. `text` is the model-facing provenance or
/// error line (never base64); `file` is the deliverable when resolution succeeded within the cap.
pub struct SourceFileOut {
    pub text: String,
    pub file: Option<SourceBlob>,
}

/// Document MIME by extension for source delivery. Unlike `read::mime_for` (image types only),
/// this covers the corpus's document formats; unknown → `application/octet-stream`.
fn source_mime(name: &str) -> &'static str {
    let ext = std::path::Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "pdf" => "application/pdf",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pptx" => "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "doc" => "application/msword",
        "xls" => "application/vnd.ms-excel",
        "ppt" => "application/vnd.ms-powerpoint",
        "rtf" => "application/rtf",
        "md" => "text/markdown",
        "txt" => "text/plain",
        "csv" => "text/csv",
        "json" => "application/json",
        "html" | "htm" => "text/html",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "tif" | "tiff" => "image/tiff",
        _ => "application/octet-stream",
    }
}

/// Resolve a citation reference to its backing corpus file, for delivery to the client as a
/// cited source (not for reading its text — that is `read`).
///
/// `path` is a document path (as from search/grep) or a structural node id; a *reasoning* node id
/// is rejected — it backs evidence in possibly several documents, not one file. `n` is the cited
/// page (1-based, PDF), used for provenance and for the over-cap page slice.
///
/// Delivery: the whole file when it is `<= max_bytes`; otherwise, for a page-scoped PDF, the cited
/// page extracted as its own PDF (text preserved, no rasterization); otherwise a structured error
/// asking for a specific PDF page. `text` always states what was delivered so the model cites honestly.
pub fn get_source_file(
    idx: &DocIndex,
    graph: Option<&crate::graph::store::GraphStore>,
    path: &str,
    n: Option<u64>,
    max_bytes: u64,
    raw: bool,
) -> SourceFileOut {
    let err = |text: String| SourceFileOut { text, file: None };

    // A reasoning node backs no single source file — steer to a document path/page instead.
    if let Some(g) = graph {
        if let Ok(Some(node)) = g.get_node(path) {
            if !crate::graph::STRUCTURAL_NODES.contains(&node.node_type.as_str()) {
                return err(format!(
                    "{path} is a reasoning node, not a source file — pass the document path (and page) from its `read` anchor instead"
                ));
            }
        }
    }

    // Resolve to a real indexed document, tolerant of a mangled path (same as `read`).
    let real = match idx.read_chunk_by_ord(path, 1) {
        Ok(Some(_)) => path.to_string(),
        _ => match idx.canonical_document_path(path) {
            Some(r) => r,
            None => return err(path_not_found(idx, path)),
        },
    };

    let doc_path = idx.doc_file(&real);
    let mut bytes = match std::fs::read(&doc_path) {
        Ok(b) => b,
        Err(e) => {
            return err(format!(
                "source file for {real} is not readable on disk ({e}) — the index has it but the file may have moved"
            ))
        }
    };
    let mut name = real.rsplit(['\\', '/']).next().unwrap_or(&real).to_string();
    let ext = std::path::Path::new(&real)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let page_note = n.map(|p| format!(" (cited page {p})")).unwrap_or_default();

    // DOCX is delivered as PDF by default — many clients render .docx inconsistently. `raw`
    // opts out. A conversion error is non-fatal: fall back to the original bytes with a note.
    let mut convert_note: Option<String> = None;
    if ext == "docx" && !raw {
        match crate::convert::docx_pdf::docx_to_pdf(&bytes) {
            Ok(pdf) => {
                let stem = name.strip_suffix(".docx").unwrap_or(&name).to_string();
                let pages = crate::convert::docx_pdf::pdf_page_count(&pdf)
                    .map(|p| format!("{p} pages"))
                    .unwrap_or_else(|| "the full document".to_string());
                convert_note = Some(format!(
                    "{real} was converted to PDF for delivery (source format renders inconsistently across clients). Returned: {stem}.pdf, {pages}. To get the untouched original file, call again with raw: true."
                ));
                bytes = pdf;
                name = format!("{stem}.pdf");
            }
            Err(e) => {
                convert_note = Some(format!(
                    "PDF conversion of {real} failed; returning the original .docx bytes unchanged. Reason: {e}."
                ));
            }
        }
    }
    let size = bytes.len() as u64;

    // Whole file when it fits. After a successful conversion `name` ends in .pdf, so
    // `source_mime` yields application/pdf; the delivered bytes are the generated PDF.
    if size <= max_bytes {
        return SourceFileOut {
            text: convert_note
                .unwrap_or_else(|| format!("delivered whole file: {real}{page_note}")),
            file: Some(SourceBlob { mime: source_mime(&name).to_string(), filename: name, bytes }),
        };
    }

    // Over cap: a page-scoped PDF degrades to the cited page as its own PDF (text preserved).
    if ext == "pdf" {
        let Some(page) = n else {
            return err(format!(
                "{real} is {size} bytes, over the {max_bytes}-byte cap — pass the cited page `n` to deliver just that page as a PDF"
            ));
        };
        return match slice_pdf_page(&doc_path, page) {
            Ok(sliced) if (sliced.len() as u64) <= max_bytes => {
                let stem = name.strip_suffix(".pdf").unwrap_or(&name);
                SourceFileOut {
                    text: format!(
                        "delivered page {page} of {real} as a PDF (original {size} bytes exceeds the {max_bytes}-byte cap)"
                    ),
                    file: Some(SourceBlob {
                        filename: format!("{stem}-p{page}.pdf"),
                        mime: "application/pdf".to_string(),
                        bytes: sliced,
                    }),
                }
            }
            Ok(sliced) => err(format!(
                "page {page} of {real} is still {} bytes, over the {max_bytes}-byte cap — cite a lighter page",
                sliced.len()
            )),
            Err(e) => err(format!("could not extract page {page} of {real}: {e}")),
        };
    }

    err(format!(
        "{real} is {size} bytes, over the {max_bytes}-byte cap, and is not a PDF — cite a specific PDF page for a lighter extract"
    ))
}

/// Extract 1-based PDF page `page` into its own single-page PDF (bytes). Text and vector content
/// are preserved (no rasterization). Errors if the page is out of range or the PDF can't be edited.
fn slice_pdf_page(doc_path: &std::path::Path, page: u64) -> anyhow::Result<Vec<u8>> {
    let mut ed = pdf_oxide::editor::DocumentEditor::open(doc_path)
        .map_err(|e| anyhow::anyhow!("open PDF: {e}"))?;
    let count = ed.current_page_count();
    let idx0 = page.saturating_sub(1) as usize;
    if idx0 >= count {
        anyhow::bail!("page {page} out of range (document has {count} pages)");
    }
    ed.extract_pages_to_bytes(&[idx0])
        .map_err(|e| anyhow::anyhow!("extract page: {e}"))
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
        i.write_chunks(&[Chunk {
            doc_path: PathBuf::from("МОДУЛЬ.pdf"),
            location: "p.7".into(),
            file_type: "pdf".into(),
            text: "параметр maxTsdr равен 3000".into(),
        }])
        .unwrap();
        (d, i)
    }

    fn pdf_corpus() -> (tempfile::TempDir, DocIndex) {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join("sample.pdf"),
            include_bytes!("../tests/fixtures/sample.pdf"),
        )
        .unwrap();
        crate::index::store::index_dir(d.path(), true).unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        (d, i)
    }

    fn docx_corpus() -> (tempfile::TempDir, DocIndex) {
        let d = tempfile::tempdir().unwrap();
        let mut doc = rdocx::Document::new();
        doc.add_paragraph("Integration fixture paragraph one.");
        doc.add_paragraph("Integration fixture paragraph two.");
        let bytes = doc.to_bytes().unwrap();
        std::fs::write(d.path().join("report.docx"), &bytes).unwrap();
        crate::index::store::index_dir(d.path(), true).unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        (d, i)
    }

    #[test]
    fn get_source_file_converts_docx_to_pdf_by_default() {
        let (_d, i) = docx_corpus();
        let out = get_source_file(&i, None, "report.docx", None, DEFAULT_SOURCE_MAX_BYTES, false);
        let f = out.file.expect("file delivered");
        assert_eq!(f.mime, "application/pdf");
        assert!(f.filename.ends_with(".pdf"));
        assert!(f.bytes.starts_with(b"%PDF"));
        assert!(out.text.contains("converted"));
        assert!(out.text.contains("raw: true"));
    }

    #[test]
    fn get_source_file_raw_returns_original_docx() {
        let (_d, i) = docx_corpus();
        let out = get_source_file(&i, None, "report.docx", None, DEFAULT_SOURCE_MAX_BYTES, true);
        let f = out.file.expect("file delivered");
        assert_eq!(f.filename, "report.docx");
        assert_eq!(
            f.mime,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        );
        assert!(out.text.contains("whole file"));
    }

    #[test]
    fn get_source_file_delivers_whole_pdf_under_cap() {
        let (_d, i) = pdf_corpus();
        let out = get_source_file(&i, None, "sample.pdf", Some(1), DEFAULT_SOURCE_MAX_BYTES, false);
        let f = out.file.expect("file delivered");
        assert_eq!(f.filename, "sample.pdf");
        assert_eq!(f.mime, "application/pdf");
        assert!(f.bytes.starts_with(b"%PDF"));
        assert!(out.text.contains("whole file"));
    }

    #[test]
    fn slice_pdf_page_yields_a_valid_pdf() {
        let d = tempfile::tempdir().unwrap();
        let pdf = d.path().join("sample.pdf");
        std::fs::write(&pdf, include_bytes!("../tests/fixtures/sample.pdf")).unwrap();
        // The core runtime check: pdf_oxide extracts page 1 into its own real PDF (text preserved).
        let bytes = slice_pdf_page(&pdf, 1).expect("extract page 1");
        assert!(bytes.starts_with(b"%PDF"));
        assert!(bytes.len() > 100);
        // Out-of-range page is an error, not a panic.
        assert!(slice_pdf_page(&pdf, 9999).is_err());
    }

    #[test]
    fn get_source_file_over_cap_pdf_delivers_page_slice() {
        let (d, i) = pdf_corpus();
        let pdf = d.path().join("sample.pdf");
        let whole = std::fs::read(&pdf).unwrap().len() as u64;
        let slice_len = slice_pdf_page(&pdf, 1).unwrap().len() as u64;
        // Only assert the slice-delivery branch when the fixture allows it (a single page smaller
        // than the whole file); the slicer itself is covered above regardless.
        if slice_len < whole {
            let out = get_source_file(&i, None, "sample.pdf", Some(1), slice_len, false);
            let f = out.file.expect("page slice delivered");
            assert_eq!(f.mime, "application/pdf");
            assert!(f.filename.ends_with("-p1.pdf"));
            assert!(f.bytes.starts_with(b"%PDF"));
            assert!(out.text.contains("page 1"));
        }
    }

    #[test]
    fn get_source_file_cap_below_page_size_errs_not_panics() {
        let (_d, i) = pdf_corpus();
        // A cap smaller than even one extracted page → structured error, no file (never a panic).
        let out = get_source_file(&i, None, "sample.pdf", Some(1), 10, false);
        assert!(out.file.is_none());
        assert!(out.text.contains("cap"));
    }

    #[test]
    fn get_source_file_over_cap_pdf_without_page_asks_for_one() {
        let (_d, i) = pdf_corpus();
        let out = get_source_file(&i, None, "sample.pdf", None, 10, false);
        assert!(out.file.is_none());
        assert!(out.text.contains("cited page"));
    }

    #[test]
    fn get_source_file_over_cap_non_pdf_errs() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("notes.md"), "# Title\n\nplenty of text here\n").unwrap();
        crate::index::store::index_dir(d.path(), true).unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        let out = get_source_file(&i, None, "notes.md", Some(1), 4, false);
        assert!(out.file.is_none());
        assert!(out.text.contains("not a PDF"));
    }

    #[test]
    fn get_source_file_unknown_path_errs() {
        let (_d, i) = pdf_corpus();
        let out = get_source_file(&i, None, "nope.pdf", Some(1), DEFAULT_SOURCE_MAX_BYTES, false);
        assert!(out.file.is_none());
        assert!(out.text.contains("no document indexed"));
    }

    #[test]
    fn read_page_image_returns_pdf_page_png_without_chunk_body() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join("sample.pdf"),
            include_bytes!("../tests/fixtures/sample.pdf"),
        )
        .unwrap();
        crate::index::store::index_dir(d.path(), true).unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();

        let out = read(
            d.path(),
            &i,
            None,
            "sample.pdf",
            1,
            true,
            &TraceLog::disabled(),
        );

        assert_eq!(out.text, "page_image sample.pdf #1");
        assert!(!out.text.contains("glossa sample"));
        assert_eq!(out.images.len(), 1);
        assert_eq!(out.images[0].mime, "image/png");
        assert!(out.images[0].bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
    }

    #[test]
    fn read_page_image_clamps_out_of_range_ord_to_last_page() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join("sample.pdf"),
            include_bytes!("../tests/fixtures/sample.pdf"),
        )
        .unwrap();
        crate::index::store::index_dir(d.path(), true).unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();

        let out = read(
            d.path(),
            &i,
            None,
            "sample.pdf",
            99,
            true,
            &TraceLog::disabled(),
        );

        assert_eq!(out.text, "page_image sample.pdf #1");
        assert_eq!(out.images.len(), 1);
        assert_eq!(out.images[0].mime, "image/png");
    }

    #[test]
    fn read_page_image_rejects_non_pdf_document() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("sample.md"), b"# Sample\nglossa sample\n").unwrap();
        crate::index::store::index_dir(d.path(), true).unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();

        let out = read(
            d.path(),
            &i,
            None,
            "sample.md",
            1,
            true,
            &TraceLog::disabled(),
        );

        assert_eq!(
            out.text,
            "page_image is only supported for PDF (got sample.md)"
        );
        assert!(out.images.is_empty());
    }

    #[test]
    fn read_empty_pdf_page_returns_success_with_label() {
        let d = tempfile::tempdir().unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        i.write_chunks(&[
            Chunk {
                doc_path: PathBuf::from("d.pdf"),
                location: "p.1".into(),
                file_type: "pdf".into(),
                text: "alpha".into(),
            },
            Chunk {
                doc_path: PathBuf::from("d.pdf"),
                location: "p.2".into(),
                file_type: "pdf".into(),
                text: String::new(),
            },
            Chunk {
                doc_path: PathBuf::from("d.pdf"),
                location: "p.3".into(),
                file_type: "pdf".into(),
                text: "charlie".into(),
            },
        ])
        .unwrap();
        let t = TraceLog::disabled();
        let out = read(d.path(), &i, None, "d.pdf", 2, false, &t);
        assert!(!out.text.contains("no chunk"), "must succeed: {}", out.text);
        assert!(out.text.contains("(page #2"), "blank label: {}", out.text);
        assert!(
            out.text.contains("prev #1") && out.text.contains("next #3"),
            "footer: {}",
            out.text
        );
    }

    #[test]
    fn read_serves_notebook_file_by_doc_scoped_path() {
        let dir = tempfile::tempdir().unwrap();
        let idx = crate::index::store::DocIndex::open_or_create(dir.path()).unwrap();
        // The notebook resolves a note's doc-scoped path against an INDEXED document (same
        // validation `note`/`ls` use), so register the doc before writing a note under it.
        let doc = "some_std.docx";
        idx.write_chunks(&[Chunk {
            doc_path: PathBuf::from(doc),
            location: "p.1".into(),
            file_type: "docx".into(),
            text: "placeholder".into(),
        }])
        .unwrap();
        // create a notebook file under the doc's mirror
        crate::notebook::note(dir.path(), &idx, doc, "sizes.csp", "D\n50\n63\n", false);
        let out = read(
            dir.path(),
            &idx,
            None,
            "some_std.docx/sizes.csp",
            1,
            false,
            &TraceLog::disabled(),
        );
        assert!(out.text.contains("50") && out.text.contains("63"), "got: {}", out.text);
        // a non-notebook, non-corpus path still errors as before
        let miss = read(
            dir.path(),
            &idx,
            None,
            "no/such/thing",
            1,
            false,
            &TraceLog::disabled(),
        );
        assert!(miss.text.contains("no document indexed") || miss.text.contains("not found"), "got: {}", miss.text);
    }

    fn prov() -> Provenance {
        Provenance {
            source_path: "d.pdf".into(),
            range: None,
            file_sig: None,
            origin: "agent".into(),
            confidence: 0.8,
            created_at: 1,
        }
    }
    fn node(id: &str, ty: &str, label: &str) -> Node {
        Node {
            id: id.into(),
            node_type: ty.into(),
            label: label.into(),
            aliases: Vec::new(),
            prov: prov(),
        }
    }
    fn edge(from: &str, rel: &str, to: &str) -> Edge {
        Edge {
            from: from.into(),
            to: to.into(),
            edge_type: rel.into(),
            prov: prov(),
        }
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
            node("tsk:prog", "Task", "Программирование контроллера"),
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
        ChainSpec {
            spine_rels: vec!["CAUSED_BY".into(), "RESOLVED_BY".into()],
        }
    }

    #[test]
    fn glossary_renders_full_chain_in_one_call() {
        let (_d, i) = idx();
        let (_gd, g) = reasoning_graph();
        let t = TraceLog::disabled();
        let out = glossary(&i, &g, "Профибус потеря связи", &spine_spec(), &t);
        // entry node + the whole chain to the resolution, surfaced by a SINGLE call
        assert!(
            out.contains("[Symptom]") && out.contains("Профибус потеря связи"),
            "{out}"
        );
        assert!(
            out.contains("→ CAUSED_BY") && out.contains("[Cause]") && out.contains("Малый maxTsdr"),
            "{out}"
        );
        assert!(
            out.contains("→ RESOLVED_BY")
                && out.contains("[Resolution]")
                && out.contains("Изменить maxTsdr"),
            "{out}"
        );
        // SIMILAR is no longer part of glossary — it moved to neighbors
        assert!(
            !out.contains("SIMILAR"),
            "glossary must not show SIMILAR: {out}"
        );
    }

    #[test]
    fn glossary_chain_is_ontology_driven() {
        // No spine relations configured → glossary shows only the node head, no chain.
        let (_d, i) = idx();
        let (_gd, g) = reasoning_graph();
        let t = TraceLog::disabled();
        let empty = ChainSpec {
            spine_rels: Vec::new(),
        };
        let out = glossary(&i, &g, "Профибус потеря связи", &empty, &t);
        assert!(out.contains("[Symptom]"), "{out}");
        assert!(
            !out.contains("CAUSED_BY") && !out.contains("RESOLVED_BY"),
            "no spines → no chain: {out}"
        );
    }

    #[test]
    fn neighbors_shows_similar_generalization_for_a_node_id() {
        let (_d, i) = idx();
        let (_gd, g) = reasoning_graph();
        let t = TraceLog::disabled();
        let out = neighbors(&i, &g, Some("sym:loss"), None, None, &t);
        assert!(
            out.contains("SIMILAR")
                && out.contains("Обновление ПО модулей")
                && out.contains("Программирование контроллера"),
            "{out}"
        );
        // it must NOT walk the causal spine — that is glossary's job
        assert!(
            !out.contains("CAUSED_BY") && !out.contains("[Resolution]"),
            "neighbors is generalization only: {out}"
        );
    }

    const REASONING_ONT: &str = r#"
[entities.Symptom]
props = ["name"]
[entities.Cause]
props = ["name"]
[entities.Resolution]
props = ["name"]
[entities.Section]
props = ["name"]
[relations.CAUSED_BY]
from = ["Symptom"]
to = ["Cause"]
[relations.RESOLVED_BY]
from = ["Cause", "Symptom"]
to = ["Resolution"]
[relations.MENTIONS]
from = ["Symptom"]
to = ["Section"]
[validation]
strict = false
[reasoning]
spines = [{ anchor = "Symptom", relations = ["CAUSED_BY", "RESOLVED_BY"] }]
closure = [["CAUSED_BY", "RESOLVED_BY", "RESOLVED_BY"]]
"#;

    fn two_component_reasoning_graph() -> (tempfile::TempDir, GraphStore) {
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        let ont = Ontology::parse(REASONING_ONT).unwrap();
        let nodes = [
            node("sym:a", "Symptom", "Профибус потеря связи"),
            node("cau:a", "Cause", "Малый maxTsdr"),
            node("res:a", "Resolution", "Поднять maxTsdr"),
            node("sym:b", "Symptom", "Modbus таймаут опроса"),
            node("cau:b", "Cause", "Неверный baud"),
            node("res:b", "Resolution", "Изменить скорость порта"),
        ];
        let edges = [
            edge("sym:a", "CAUSED_BY", "cau:a"),
            edge("cau:a", "RESOLVED_BY", "res:a"),
            edge("sym:b", "CAUSED_BY", "cau:b"),
            edge("cau:b", "RESOLVED_BY", "res:b"),
        ];
        g.upsert(&ont, &nodes, &edges).unwrap();
        let opts = crate::graph::generalize::apply::Opts::from_ontology(&ont, 100);
        crate::graph::generalize::apply::generalize(&g, &opts).unwrap();
        (d, g)
    }

    #[test]
    fn neighbors_shows_community_siblings_after_generalize() {
        let (_d, i) = idx();
        let (_gd, g) = two_component_reasoning_graph();
        let t = TraceLog::disabled();
        let out = neighbors(&i, &g, Some("sym:a"), None, None, &t);
        assert!(
            out.contains("COMMUNITY"),
            "expected community siblings: {out}"
        );
        assert!(
            out.contains("cau:a") || out.contains("res:a"),
            "same-component siblings: {out}"
        );
        assert!(
            !out.contains("sym:b") && !out.contains("cau:b") && !out.contains("res:b"),
            "other component excluded: {out}"
        );
        assert!(!out.contains("sym:a  [Symptom]"), "self excluded: {out}");
    }

    #[test]
    fn neighbors_dedups_similar_from_community_section() {
        let (_d, i) = idx();
        let (_gd, g) = reasoning_graph();
        let ont = Ontology::parse(REASONING_ONT).unwrap();
        let opts = crate::graph::generalize::apply::Opts::from_ontology(&ont, 100);
        crate::graph::generalize::apply::generalize(&g, &opts).unwrap();
        let t = TraceLog::disabled();
        let out = neighbors(&i, &g, Some("sym:loss"), None, None, &t);
        assert!(
            out.contains("SIMILAR") && out.contains("Обновление ПО модулей"),
            "{out}"
        );
        let community_lines: Vec<_> = out.lines().filter(|l| l.starts_with("COMMUNITY")).collect();
        assert!(
            !community_lines.iter().any(|l| l.contains("Обновление ПО модулей") || l.contains("Программирование контроллера")),
            "SIMILAR nodes must not repeat in COMMUNITY: {out}",
        );
    }

    #[test]
    fn neighbors_community_sorted_by_pagerank() {
        use crate::graph::store::NodeMeta;
        let (_d, i) = idx();
        let (_gd, g) = reasoning_graph();
        g.replace_node_meta(&[
            (
                "sym:loss".into(),
                NodeMeta {
                    community: Some(1),
                    pagerank: Some(0.9),
                    degree: Some(3),
                },
            ),
            (
                "cau:tsdr".into(),
                NodeMeta {
                    community: Some(1),
                    pagerank: Some(0.2),
                    degree: Some(1),
                },
            ),
            (
                "res:set".into(),
                NodeMeta {
                    community: Some(1),
                    pagerank: Some(0.7),
                    degree: Some(2),
                },
            ),
            (
                "tsk:upd".into(),
                NodeMeta {
                    community: Some(1),
                    pagerank: Some(0.5),
                    degree: Some(1),
                },
            ),
            (
                "tsk:prog".into(),
                NodeMeta {
                    community: Some(1),
                    pagerank: Some(0.1),
                    degree: Some(1),
                },
            ),
        ])
        .unwrap();
        let t = TraceLog::disabled();
        let out = neighbors(&i, &g, Some("sym:loss"), None, None, &t);
        let community_lines: Vec<_> = out.lines().filter(|l| l.starts_with("COMMUNITY")).collect();
        assert_eq!(
            community_lines.len(),
            2,
            "limit excludes SIMILAR dupes, top-2 by pr after SIMILAR taken: {out}"
        );
        assert!(
            community_lines[0].contains("res:set"),
            "highest pr sibling first: {out}"
        );
        assert!(
            community_lines[1].contains("cau:tsdr"),
            "second by pr: {out}"
        );
    }

    #[test]
    fn graph_stats_lists_communities_with_top_examples() {
        use crate::graph::store::NodeMeta;
        let gd = tempfile::tempdir().unwrap();
        let g = GraphStore::open(gd.path()).unwrap();
        for n in [
            node("sym:a", "Symptom", "Profibus loss"),
            node("sym:b", "Symptom", "Modbus timeout"),
        ] {
            g.put_node(&n).unwrap();
        }
        g.replace_node_meta(&[
            (
                "sym:a".into(),
                NodeMeta {
                    community: Some(0),
                    pagerank: Some(0.9),
                    degree: Some(2),
                },
            ),
            (
                "sym:b".into(),
                NodeMeta {
                    community: Some(1),
                    pagerank: Some(0.5),
                    degree: Some(1),
                },
            ),
        ])
        .unwrap();
        let out = graph_stats(&g);
        assert!(
            out.contains("nodes: 2") && out.contains("communities: 2"),
            "{out}"
        );
        assert!(
            out.contains("comm 0  (1 nodes)") && out.contains("Profibus loss"),
            "{out}"
        );
        assert!(
            out.contains("comm 1  (1 nodes)") && out.contains("Modbus timeout"),
            "{out}"
        );
    }

    #[test]
    fn graph_stats_without_meta_shows_none() {
        let gd = tempfile::tempdir().unwrap();
        let g = GraphStore::open(gd.path()).unwrap();
        g.put_node(&node("sym:x", "Symptom", "X")).unwrap();
        let out = graph_stats(&g);
        assert!(
            out.contains("nodes: 1") && out.contains("communities: (none)"),
            "{out}"
        );
    }

    #[test]
    fn read_is_omnivorous_over_reasoning_node_ids() {
        let (_d, i) = idx(); // МОДУЛЬ.pdf #p.7 = "параметр maxTsdr равен 3000"
        let gd = tempfile::tempdir().unwrap();
        let g = GraphStore::open(gd.path()).unwrap();
        g.put_node(&node("res:fix", "Resolution", "Изменить maxTsdr в 3000"))
            .unwrap();
        g.put_edge(&edge("res:fix", "MENTIONS", "МОДУЛЬ.pdf#p.7"))
            .unwrap();
        let t = TraceLog::disabled();
        // reading the NODE id returns the node line + the evidence chunk it MENTIONS, attributed.
        let out = read(_d.path(), &i, Some(&g), "res:fix", 1, false, &t).text;
        assert!(
            out.contains("[Resolution]") && out.contains("Изменить maxTsdr"),
            "node header: {out}"
        );
        assert!(
            out.contains("── MENTIONS · МОДУЛЬ.pdf #7 ──"),
            "attributed evidence header: {out}"
        );
        assert!(
            out.contains("параметр maxTsdr равен 3000"),
            "evidence body: {out}"
        );
        // a plain doc path still reads the chunk (not treated as a node).
        let doc = read(_d.path(), &i, Some(&g), "МОДУЛЬ.pdf", 7, false, &t).text;
        assert!(
            doc.contains("параметр maxTsdr равен 3000") && !doc.contains("── MENTIONS"),
            "doc read unchanged: {doc}"
        );
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
            Chunk {
                doc_path: PathBuf::from("d.md"),
                location: "S1".into(),
                file_type: "md".into(),
                text: big.clone(),
            },
            Chunk {
                doc_path: PathBuf::from("d.md"),
                location: "S2".into(),
                file_type: "md".into(),
                text: "second".into(),
            },
        ])
        .unwrap();
        let t = TraceLog::disabled();
        let out = read(d.path(), &i, None, "d.md", 1, false, &t);
        assert!(out.text.contains(&big), "full body, no cap"); // not truncated
        assert!(out.text.contains("next #2") && !out.text.contains("end of document"));
        assert!(out.text.contains("‹ start of document · next #2 ›")); // unified footer (MCP wording)
                                                                       // Out-of-range read CLAMPS to the valid range and returns that chunk (here the last,
                                                                       // #2 = "second") instead of an error the model loops on.
        let oor = read(d.path(), &i, None, "d.md", 99, false, &t).text;
        assert!(oor.contains("second"), "clamped to last chunk: {oor}");
        // A wrong path reports that the document isn't indexed.
        assert!(read(d.path(), &i, None, "nope.md", 1, false, &t)
            .text
            .contains("no document indexed at nope.md"));
    }

    #[test]
    fn read_tolerates_collapsed_whitespace_in_path() {
        let d = tempfile::tempdir().unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        // Real file name has a double space; the model copies it back with a single space.
        i.write_chunks(&[Chunk {
            doc_path: PathBuf::from("dir/Safety Manual -  1_0_3.pdf"),
            location: "p.1".into(),
            file_type: "pdf".into(),
            text: "safety body".into(),
        }])
        .unwrap();
        let t = TraceLog::disabled();
        let out = read(
            d.path(),
            &i,
            None,
            "dir/Safety Manual - 1_0_3.pdf",
            1,
            false,
            &t,
        ); // single space
        assert!(
            out.text.contains("safety body"),
            "resolved despite collapsed double space: {}",
            out.text
        );
        // Doubled / swapped separators (model over-escapes `\\` or uses `\`) also resolve.
        let out2 = read(
            d.path(),
            &i,
            None,
            "dir\\\\Safety Manual - 1_0_3.pdf",
            1,
            false,
            &t,
        );
        assert!(
            out2.text.contains("safety body"),
            "resolved despite doubled backslash: {}",
            out2.text
        );
    }

    #[test]
    fn read_tolerates_spurious_leading_prefix() {
        let d = tempfile::tempdir().unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        i.write_chunks(&[Chunk {
            doc_path: PathBuf::from("Архив БД\\doc.pdf"),
            location: "p.1".into(),
            file_type: "pdf".into(),
            text: "body text".into(),
        }])
        .unwrap();
        let t = TraceLog::disabled();
        let out = read(
            d.path(),
            &i,
            None,
            "kb-manual\\Архив БД\\doc.pdf",
            1,
            false,
            &t,
        );
        assert!(
            out.text.contains("body text"),
            "strip spurious prefix: {}",
            out.text
        );
    }

    #[test]
    fn read_suggests_path_on_total_miss() {
        let d = tempfile::tempdir().unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        i.write_chunks(&[Chunk {
            doc_path: PathBuf::from("Архив БД\\Методика поверки МодульПЛК 2025.pdf"),
            location: "p.1".into(),
            file_type: "pdf".into(),
            text: "x".into(),
        }])
        .unwrap();
        let t = TraceLog::disabled();
        let out = read(
            d.path(),
            &i,
            None,
            "kb-manual\\МодульПЛК 2025.pdf",
            1,
            false,
            &t,
        )
        .text;
        assert!(out.contains("did you mean"), "hint on miss: {out}");
        assert!(out.contains("Методика"), "suggest real path: {out}");
    }

    #[test]
    fn read_reasoning_node_id_unchanged() {
        let d = tempfile::tempdir().unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        g.put_node(&Node {
            id: "sym:test".into(),
            node_type: "Symptom".into(),
            label: "Test".into(),
            aliases: vec![],
            prov: prov(),
        })
        .unwrap();
        i.write_chunks(&[Chunk {
            doc_path: PathBuf::from("evidence.md"),
            location: "S1".into(),
            file_type: "md".into(),
            text: "evidence body".into(),
        }])
        .unwrap();
        g.put_edge(&Edge {
            from: "sym:test".into(),
            to: "evidence.md#S1".into(),
            edge_type: "MENTIONS".into(),
            prov: prov(),
        })
        .unwrap();
        let t = TraceLog::disabled();
        let out = read(d.path(), &i, Some(&g), "sym:test", 1, false, &t).text;
        assert!(out.contains("evidence body"), "reasoning node read: {out}");
    }

    #[test]
    fn read_page_image_rejects_reasoning_node_id() {
        let d = tempfile::tempdir().unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        g.put_node(&node("sym:test", "Symptom", "Test")).unwrap();

        let out = read(
            d.path(),
            &i,
            Some(&g),
            "sym:test",
            1,
            true,
            &TraceLog::disabled(),
        );

        assert_eq!(
            out.text,
            "page_image is only supported for PDF (got sym:test)"
        );
        assert!(out.images.is_empty());
    }

    #[test]
    fn grep_and_glob_render() {
        let (d, i) = idx();
        let t = TraceLog::disabled();
        assert!(grep(d.path(), &i, "maxTsdr", &crate::grep::GrepOpts::default(), &t).contains(":#7:"));
        assert_eq!(
            grep(d.path(), &i, "nomatchzzz", &crate::grep::GrepOpts::default(), &t),
            "(no matches)"
        );
        assert!(glob(&i, "*МОДУЛЬ*", &t).contains("МОДУЛЬ.pdf  (7 chunks)"));
        assert!(glob(&i, "*nomatch*", &t).starts_with("(no documents match"));
    }

    #[test]
    fn grep_matches_lines_in_a_notebook_file() {
        let dir = tempfile::tempdir().unwrap();
        let idx = crate::index::store::DocIndex::open_or_create(dir.path()).unwrap();
        // The notebook resolves a note's doc-scoped path against an INDEXED document (same
        // validation `note`/`ls` use), so register the doc before writing a note under it.
        let doc = "some_std.docx";
        idx.write_chunks(&[Chunk {
            doc_path: PathBuf::from(doc),
            location: "p.1".into(),
            file_type: "docx".into(),
            text: "placeholder".into(),
        }])
        .unwrap();
        crate::notebook::note(dir.path(), &idx, doc, "sizes.csp", "D\n50\n63\n80\n", false);
        let opts = crate::grep::GrepOpts {
            path: Some("some_std.docx/sizes.csp".into()),
            ..Default::default()
        };
        let out = grep(dir.path(), &idx, "63", &opts, &TraceLog::disabled());
        assert!(out.contains("63"), "notebook grep should find the line; got: {out}");
        assert!(!out.contains("no document indexed"), "must not error on a notebook path");
        // a non-matching pattern still renders "(no matches)", not an index error
        let none = grep(dir.path(), &idx, "nomatchzzz", &opts, &TraceLog::disabled());
        assert_eq!(none, "(no matches)");
    }

    // ── graph tool tests ────────────────────────────────────────────────────

    use crate::graph::ontology::Ontology;
    use crate::graph::store::{Edge, GraphStore, Node, Provenance};

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
        assert!(
            result.contains("org:acme"),
            "expected node id in result, got: {result}"
        );
        assert_eq!(
            glossary(&idx, &g, "nonesuch", &ChainSpec::default(), &t),
            "(no matches)"
        );
    }

    #[test]
    fn glossary_surfaces_community_centrality_after_generalize() {
        let (_dir, g) = graph_fixture();
        let idir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(idir.path()).unwrap();
        let t = TraceLog::disabled();

        // Before the pass there is no node_meta → output is unannotated (back-compat).
        let before = glossary(&idx, &g, "ACME", &ChainSpec::default(), &t);
        assert!(
            !before.contains("comm "),
            "no meta annotation before generalize: {before}"
        );

        // Run the generalization pass → community/centrality land in node_meta.
        let opts = crate::graph::generalize::apply::Opts::defaults(1);
        crate::graph::generalize::apply::generalize(&g, &opts).unwrap();

        let after = glossary(&idx, &g, "ACME", &ChainSpec::default(), &t);
        assert!(
            after.contains("org:acme"),
            "still shows the node id: {after}"
        );
        assert!(
            after.contains("comm "),
            "glossary surfaces community after generalize: {after}"
        );
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
        g.put_node(&node("sym:loss", "Symptom", "Потеря связи"))
            .unwrap();
        g.put_node(&node("cau:maxtsdr", "Cause", "Малый maxTsdr"))
            .unwrap();
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
        assert!(
            out.contains("→ CAUSED_BY") && out.contains("[Cause]"),
            "walks the CAUSED_BY hop: {out}"
        );
        assert!(
            out.contains("Малый maxTsdr"),
            "shows the connected Cause label: {out}"
        );
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
        std::fs::write(
            dir.path().join("g.md"),
            b"# Intro\nhello\n## Details\nworld\n",
        )
        .unwrap();
        index_dir(dir.path(), true).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        let t = TraceLog::disabled();

        // "Intro" is a Section node; glossary should render it as "path  #1 · Intro".
        let result = glossary(&idx, &g, "Intro", &ChainSpec::default(), &t);
        assert!(result.contains("#1"), "section rendered with ord: {result}");
        assert!(result.contains("Intro"), "section label present: {result}");

        assert_eq!(
            glossary(&idx, &g, "nonexistentzzz", &ChainSpec::default(), &t),
            "(no matches)"
        );
    }
}
