//! Single source of truth for the agent tools' model-facing output. Both the MCP server
//! (src/mcp.rs) and the kb-eval harness call these so prod and eval render identically.

use crate::grep::GrepOpts;
use crate::index::store::{DocIndex, RankedHit};
use crate::trace::TraceLog;
use serde_json::json;

/// Indexed document path suggestions for a failed path lookup. First a literal basename
/// substring glob (`*<base>*`); when that finds nothing — because the model swapped the
/// filename's spaces for underscores, or mangled a word — fall back to ranking every indexed
/// basename by how many word-tokens it shares with the failed basename, so a near-miss still
/// surfaces the real file. Up to 3 suggestions.
pub fn document_path_hints(idx: &DocIndex, path: &str) -> String {
    let base = path.rsplit(['\\', '/']).next().unwrap_or(path);
    if base.len() < 4 {
        return String::new();
    }
    let mut hints: Vec<String> = crate::glob::glob_docs(idx, &format!("*{base}*"))
        .map(|hits| hits.into_iter().take(3).map(|(p, _)| p).collect())
        .unwrap_or_default();
    if hints.is_empty() {
        hints = token_overlap_hints(idx, base);
    }
    crate::cli_fmt::format_did_you_mean(&hints)
}

/// Lowercased alphanumeric word-tokens of a filename, extension and 1-char tokens dropped —
/// the unit of comparison for a mangled-path suggestion, so a space/underscore swap or a
/// single wrong word still shares most tokens with the real name.
fn path_tokens(name: &str) -> std::collections::HashSet<String> {
    let stem = name.rsplit_once('.').map(|(s, _)| s).unwrap_or(name);
    stem.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.chars().count() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Rank indexed documents by word-token overlap with a failed basename `base`. Returns up to
/// 3 paths (best-first) whose basename shares the most tokens — at least 2, or all of them
/// when `base` has fewer than 2 tokens; empty when nothing meaningfully overlaps. Only a hint,
/// so a loose match is fine; the exact/tolerant lookup already ran and missed.
fn token_overlap_hints(idx: &DocIndex, base: &str) -> Vec<String> {
    let want = path_tokens(base);
    if want.is_empty() {
        return Vec::new();
    }
    let min_shared = want.len().clamp(1, 2);
    let all = match crate::glob::glob_docs(idx, "*") {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let mut scored: Vec<(usize, String)> = all
        .into_iter()
        .filter_map(|(p, _)| {
            let cand = p.rsplit(['\\', '/']).next().unwrap_or(&p);
            let shared = path_tokens(cand).intersection(&want).count();
            (shared >= min_shared).then_some((shared, p))
        })
        .collect();
    // Most overlap first; path order as a deterministic tiebreak.
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored.into_iter().take(3).map(|(_, p)| p).collect()
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
                .map(|h| json!({"path": h.path, "location": h.location, "score": h.score, "snippet": h.snippet}))
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
pub fn grep(
    root: &std::path::Path,
    idx: &DocIndex,
    pattern: &str,
    opts: &GrepOpts,
    trace: &TraceLog,
) -> String {
    #[cfg(feature = "notebook")]
    if let Some(p) = opts.path.as_deref() {
        if looks_like_notebook_path(p) {
            if let Ok((rel, body)) = crate::notebook::read_note(root, idx, p) {
                trace.log(
                    "grep",
                    json!({ "notebook": p, "pattern": pattern }),
                    json!({ "notebook": rel }),
                );
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
                // Feedback in the glossary/graph_query spirit: a weak model tends to over-build the
                // pattern (long `.*`/`|` regex) and match nothing. Steer it back to a simple literal
                // instead of leaving it to guess why zero came back.
                "(no matches) — the pattern matched no line. It is likely too specific: try ONE or \
                 two distinctive words, not a long regex with .* or | alternations. If you meant \
                 your text literally, set fixed:true. Fall back to search (morphology-aware BM25) \
                 when you don't know the exact string."
                    .to_string()
            } else {
                hits.iter()
                    .map(|h| h.display_line())
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
        Err(e) => format!(
            "grep error: {e}\nThat pattern failed as a regex. Set fixed:true to match it as a \
             literal string, or simplify it to one or two distinctive words."
        ),
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
    if let Ok(Some(v)) = g.validity_for(&node.id) {
        let from_disp = v.valid_from_raw.as_deref().unwrap_or("(open)");
        let to_disp = v.valid_to_raw.as_deref().unwrap_or("(open)");
        let now = crate::graph::temporal::epoch_to_rfc3339((crate::trace::now_ms() / 1000) as i64);
        let superseded = g
            .incoming(&node.id)
            .unwrap_or_default()
            .iter()
            .any(|e| e.edge_type == "SUPERSEDES");
        let status = crate::graph::temporal::status(
            v.valid_from.as_deref(),
            v.valid_to.as_deref(),
            superseded,
            &now,
        );
        let status_str = match status {
            crate::graph::temporal::Status::Future => "future",
            crate::graph::temporal::Status::Current => "current",
            crate::graph::temporal::Status::Expired => "expired",
            crate::graph::temporal::Status::Superseded => "superseded",
        };
        text.push_str(&format!("\nvalid:  {from_disp} .. {to_disp}\nstatus: {status_str}"));
    }
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
        images.extend(
            crate::read::extract_images(&idx.doc_file(p), *ord, usize::MAX).unwrap_or_default(),
        );
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
            std::path::Path::new(path)
                .extension()
                .and_then(|e| e.to_str()),
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
            trace.log(
                "read",
                json!({ "notebook": path }),
                json!({ "notebook": rel }),
            );
            return ReadOut {
                text: format!("{rel}\n{body}"),
                images: Vec::new(),
            };
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
                        return ReadOut {
                            text,
                            images: Vec::new(),
                        };
                    }
                },
                None => {
                    return ReadOut {
                        text: path_not_found(idx, path),
                        images: Vec::new(),
                    }
                }
            },
            Err(e) => {
                return ReadOut {
                    text: format!("no chunk #{n} in {path} (range lookup failed: {e})"),
                    images: Vec::new(),
                }
            }
        },
        Err(e) => {
            return ReadOut {
                text: format!("read error: {e}"),
                images: Vec::new(),
            }
        }
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
    let images =
        crate::read::extract_images(&doc_path, resolved_ord, usize::MAX).unwrap_or_default();
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

/// Advisory check for a grounded node's source having drifted since grounding: compares the
/// stored `Provenance.file_sig` against a fresh `crate::index::store::file_sig` re-stat of the
/// source file under `root`, re-stat results cached per `source_path` for the life of the
/// checker (one per tool call — many nodes typically share a doc). Same comparison as
/// `graph::generalize::hygiene::stale_nodes`, single-node form: a `None` stored sig, or a
/// missing/unreadable source, is never stale.
pub struct StaleChecker {
    root: std::path::PathBuf,
    cache: std::cell::RefCell<std::collections::HashMap<String, Option<crate::index::manifest::FileSig>>>,
}

impl StaleChecker {
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        Self {
            root: root.into(),
            cache: Default::default(),
        }
    }

    pub fn is_stale(&self, node: &crate::graph::store::Node) -> bool {
        let Some(stored) = node.prov.file_sig else {
            return false;
        };
        let cur = *self
            .cache
            .borrow_mut()
            .entry(node.prov.source_path.clone())
            .or_insert_with(|| {
                crate::index::store::file_sig(&self.root.join(&node.prov.source_path)).ok()
            });
        matches!(cur, Some(cur) if cur != stored)
    }
}

/// Compact community/centrality/staleness annotation. The community/centrality part is drawn
/// from `node_meta` (populated by `kb graph generalize` / auto-run at the end of `reindex`) and
/// is "" when no meta exists yet, so output is byte-identical on graphs that haven't been
/// generalized. When `stale` is `Some` and the node's source has drifted (`StaleChecker::is_stale`),
/// `⚠ stale` is appended regardless of `node_meta` state. Format:
/// `"  · comm 3 · pr 0.142 · deg 5 · ⚠ stale"` — only the parts that are present.
fn meta_suffix(
    g: &crate::graph::store::GraphStore,
    id: &str,
    stale: Option<&StaleChecker>,
) -> String {
    // community / pagerank / degree are SIMILAR-derived (generalize) — they belong to the `related`
    // cross-case view, not the reasoning surface. Dropped here so glossary/related edge lines stay
    // a thin grounded chain (they also bloated the reader's context). Staleness stays: it is a
    // grounding-integrity signal, not a similarity metric.
    let mut parts = Vec::new();
    if let Some(sc) = stale {
        if let Ok(Some(node)) = g.get_node(id) {
            if sc.is_stale(&node) {
                parts.push("⚠ stale".to_string());
            }
        }
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
/// `id  [type]  label`. Used by `related` to show related cases.
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
pub(crate) fn read_anchor(idx: &DocIndex, g: &crate::graph::store::GraphStore, id: &str) -> String {
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
/// `related`, not the answer chain.
#[allow(clippy::too_many_arguments)]
fn chain_lines(
    idx: &DocIndex,
    g: &crate::graph::store::GraphStore,
    id: &str,
    spec: &ChainSpec,
    depth: usize,
    seen: &mut std::collections::HashSet<String>,
    out: &mut Vec<String>,
    at: Option<&str>,
    stale: Option<&StaleChecker>,
) {
    // Bound the walk: without a cap, a densely-connected reasoning graph (many facts sharing
    // entities) lets a single glossary call render its whole connected component — hundreds of KB
    // that overflow the reader's context. A depth cap plus a total-line budget keep the chain
    // useful and small on any graph; sparse graphs (short chains) are unaffected.
    const MAX_CHAIN_DEPTH: usize = 5;
    const MAX_CHAIN_LINES: usize = 12;
    if depth >= MAX_CHAIN_DEPTH || out.len() >= MAX_CHAIN_LINES {
        return;
    }
    for e in g.outgoing(id).unwrap_or_default() {
        if out.len() >= MAX_CHAIN_LINES {
            return;
        }
        if !spec.spine_rels.iter().any(|r| r == &e.edge_type) {
            continue;
        }
        if !seen.insert(e.to.clone()) {
            continue;
        }
        if let Some(a) = at {
            if !g.visible_at(&e.to, a).unwrap_or(true) {
                continue;
            }
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
            meta_suffix(g, &node.id, stale),
            read_anchor(idx, g, &node.id),
        ));
        chain_lines(idx, g, &node.id, spec, depth + 1, seen, out, at, stale);
    }
}

/// Resolve a name/term to graph nodes and render each with its full reasoning chain.
/// Structural nodes (Section/Document) show their `(path #ord)` anchor. A reasoning node
/// (Symptom/Cause/Task/…) shows its `id [type] label` then, walked from it along the ontology's
/// spine relations, the whole chain to the Resolution — so ONE `glossary` call surfaces the
/// cause and the fix, each with a `read` anchor. SIMILAR/community links live in `related`.
/// Empty → `"(no matches)"`. `as_of` (when `Some`), normalized via `temporal::normalize_point`,
/// hides any resolved node (and any chain hop) outside its authored validity interval
/// (`GraphStore::visible_at`); a timeless node (no validity row) is always shown. `stale` (when
/// `Some`), flags any resolved node (or chain hop) whose source has drifted since grounding with
/// an inline `⚠ stale` marker — advisory only, never hides the node.
#[allow(clippy::too_many_arguments)]
/// Back-compat wrapper: `glossary` without a ranking query. Composed candidates fall back to
/// ranking by the entity name.
pub fn glossary(
    idx: &DocIndex,
    g: &crate::graph::store::GraphStore,
    name: &str,
    spec: &ChainSpec,
    trace: &TraceLog,
    as_of: Option<&str>,
    stale: Option<&StaleChecker>,
) -> String {
    glossary_with_query(idx, g, name, None, spec, trace, as_of, stale)
}

/// Look an entity up. `query` (when the caller has it — e.g. the full question) ranks the composed
/// neighbourhood by the question's own terms, not just the entity name; `None` ranks by the name.
pub fn glossary_with_query(
    idx: &DocIndex,
    g: &crate::graph::store::GraphStore,
    name: &str,
    query: Option<&str>,
    spec: &ChainSpec,
    trace: &TraceLog,
    as_of: Option<&str>,
    stale: Option<&StaleChecker>,
) -> String {
    let at = match as_of.map(crate::graph::temporal::normalize_point).transpose() {
        Ok(a) => a,
        Err(e) => return format!("as_of error: {e}"),
    };
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
            // A very common entity can be an alias on many facts; rendering each fact's full spine
            // chain floods the reader's context (a hub entity ballooned to hundreds of lines). Cap
            // the number of entry facts we expand — `resolve` returns them best-first, so the head
            // is the most relevant — and note the remainder so the reader knows to narrow the query.
            const GLOSSARY_MAX_ENTRIES: usize = 6;
            let visible: Vec<&String> = ids
                .iter()
                .filter(|id| {
                    at.as_deref()
                        .is_none_or(|a| g.visible_at(id, a).unwrap_or(true))
                })
                .collect();
            let truncated = visible.len().saturating_sub(GLOSSARY_MAX_ENTRIES);
            let mut lines: Vec<String> = visible
                .into_iter()
                .take(GLOSSARY_MAX_ENTRIES)
                .map(|id| match g.get_node(id) {
                    Ok(Some(node)) => {
                        let base = format!("{}  [{}]  {}", node.id, node.node_type, node.label);
                        match node_ref(idx, &node) {
                            // Structural node (Section/Document): show its anchor. A common entity
                            // whose name exactly matches a section title short-circuits `resolve` to
                            // this stub alone (the exact-label fast path never reaches BM25), so ALSO
                            // surface the reasoning nodes grounded (MENTIONS) to this chunk —
                            // otherwise a lookup of e.g. "Sun" yields a bare pointer and no facts.
                            Some(r) => {
                                let stub =
                                    format!("{base}  —  {r}{}", meta_suffix(g, &node.id, stale));
                                let grounded: Vec<String> = g
                                    .incoming(&node.id)
                                    .unwrap_or_default()
                                    .into_iter()
                                    .filter(|e| e.edge_type == crate::graph::MENTIONS)
                                    .filter_map(|e| g.get_node(&e.from).ok().flatten())
                                    .filter(|n| node_ref(idx, n).is_none()) // reasoning nodes only
                                    .take(8)
                                    .map(|n| {
                                        let line = format!(
                                            "{}  [{}]  {}{}",
                                            n.id,
                                            n.node_type,
                                            n.label,
                                            read_anchor(idx, g, &n.id)
                                        );
                                        // Expand this fact's reasoning chain inline (the same
                                        // rendering a direct node match gets), so the next fact in
                                        // a multi-hop path is visible in this one call — the reader
                                        // never has to pass a node id to neighbors/path.
                                        let mut seen = std::collections::HashSet::new();
                                        seen.insert(node.id.clone());
                                        seen.insert(n.id.clone());
                                        let mut chain = Vec::new();
                                        chain_lines(
                                            idx,
                                            g,
                                            &n.id,
                                            spec,
                                            0,
                                            &mut seen,
                                            &mut chain,
                                            at.as_deref(),
                                            stale,
                                        );
                                        if chain.is_empty() {
                                            line
                                        } else {
                                            format!("{line}\n{}", chain.join("\n"))
                                        }
                                    })
                                    .collect();
                                if grounded.is_empty() {
                                    stub
                                } else {
                                    format!("{stub}\n{}", grounded.join("\n"))
                                }
                            }
                            // reasoning node: append its spine chain (cause → resolution) inline
                            None => {
                                // The `— read` anchor (MENTIONS grounding) is the actionable source
                                // pointer. When it exists, drop the separate `@owner`: showing a
                                // second, possibly different document (the node's nominal owner vs
                                // the grounded doc — e.g. a component doc vs the standard it cites)
                                // made agents read the wrong one. Fall back to `@owner` only for an
                                // ungrounded node, where it is the only pointer to a document.
                                let anchor = read_anchor(idx, g, &node.id);
                                let owner = if node.prov.source_path.is_empty() || !anchor.is_empty() {
                                    String::new()
                                } else {
                                    format!("  @{}", node.prov.source_path)
                                };
                                let head =
                                    format!("{base}{owner}{}{anchor}", meta_suffix(g, &node.id, stale));
                                let mut seen = std::collections::HashSet::new();
                                seen.insert(node.id.clone());
                                let mut chain = Vec::new();
                                chain_lines(
                                    idx,
                                    g,
                                    &node.id,
                                    spec,
                                    0,
                                    &mut seen,
                                    &mut chain,
                                    at.as_deref(),
                                    stale,
                                );
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
            if lines.is_empty() {
                "(no matches)".to_string()
            } else {
                if truncated > 0 {
                    lines.push(format!(
                        "… {truncated} more fact(s) name this entity — narrow the query or use reach/graph_query to pick the one you need."
                    ));
                }
                // Composed neighborhood via Personalized PageRank: seed a random walk with restart
                // from the question's lexical hits (resolve over the node index) plus this entity,
                // and surface the top non-structural nodes it reaches by CONNECTIVITY — the multi-hop
                // target the reader would otherwise walk to by hand. Ontology-blind (the walk reads
                // no edge/node types), so a terminal answer that shares no words with the question
                // still floats up, and hubs disperse without a df cap.
                if let Ok(cands) =
                    crate::graph::compose::compose_ppr(g, name, query.unwrap_or(name), 20)
                {
                    let shown: std::collections::HashSet<&str> =
                        ids.iter().map(|s| s.as_str()).collect();
                    let extra: Vec<String> = cands
                        .iter()
                        .filter(|c| !shown.contains(c.id.as_str()))
                        .map(|c| format!("{}  [Fact]  {}{}", c.id, c.label, read_anchor(idx, g, &c.id)))
                        .collect();
                    if !extra.is_empty() {
                        lines.push(
                            "— composed (reachable from here via the reasoning graph) —".to_string(),
                        );
                        lines.extend(extra);
                    }
                }
                lines.join("\n")
            }
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

/// Resolve a tool's target node from either an explicit `node` id or a `(path, n)` chunk ref —
/// the shared entry-point for `related` / `neighbors` / `path`. `#0` is tolerated as `#1`
/// (models write 0 out of habit). Err holds a ready-to-return user message.
/// Resolve a tool's target node to `(id, note)`. An exact node id is used as-is (`note = None`).
/// Otherwise the string is treated as a name and resolved fuzzily via the same resolver `glossary`
/// uses — small models can't reliably reproduce a hashed id, so they may pass the label/entity
/// instead. Any such substitution returns a `note` so the caller can surface "searched X → used Y"
/// and it is never a silent swap.
fn resolve_node_ref(
    idx: &DocIndex,
    g: &crate::graph::store::GraphStore,
    node: Option<&str>,
    path: Option<&str>,
    n: Option<u64>,
) -> Result<(String, Option<String>), String> {
    if let Some(nid) = node.filter(|s| !s.trim().is_empty()) {
        let nid = nid.trim();
        // Exact node id → use it, no note.
        if matches!(g.get_node(nid), Ok(Some(_))) {
            return Ok((nid.to_string(), None));
        }
        // Not an id we hold: treat it as a name and resolve fuzzily. Report the substitution.
        if let Ok(hits) = g.resolve(nid) {
            if let Some(hit) = hits.into_iter().next() {
                let label = g
                    .get_node(&hit)
                    .ok()
                    .flatten()
                    .map(|x| x.label)
                    .unwrap_or_default();
                let note = format!("note: \"{nid}\" is not a node id — resolved it to {hit} ({label})");
                return Ok((hit, Some(note)));
            }
        }
        return Err(format!(
            "no node matches \"{nid}\" as an id or a name — copy an id or entity from a glossary/search line"
        ));
    }
    if let (Some(p), Some(nn)) = (path, n) {
        let nn = if nn == 0 { 1 } else { nn };
        return match idx.location_for_ord(p, nn) {
            Ok(Some(_)) => Ok((crate::graph::build::section_id(p, &nn.to_string()), None)),
            Ok(None) => Err(format!(
                "no chunk #{nn} in {p}; chunk numbers start at #1 — take it from a search/grep/read"
            )),
            Err(e) => Err(format!("resolve error: {e}")),
        };
    }
    Err("need a node id (from glossary) or a (path, n) chunk".to_string())
}

/// Prepend a resolution `note` (from [`resolve_node_ref`]) to a tool's body, so a fuzzy id
/// substitution is always visible to the caller.
fn prepend_note(note: Option<String>, body: String) -> String {
    match note {
        Some(n) => format!("{n}\n{body}"),
        None => body,
    }
}

/// `as_of` (when `Some`), normalized via `temporal::normalize_point`, drops any SIMILAR/COMMUNITY
/// endpoint outside its authored validity interval (`GraphStore::visible_at`); a timeless node
/// (no validity row) is always shown. This filters the *surrounding* graph only — the anchor node
/// itself (resolved from `node`/`path`+`n`) is always looked up and its related nodes computed
/// regardless of the anchor's own validity window. `stale` (when `Some`) flags any drifted-source
/// endpoint with an inline `⚠ stale` marker.
#[allow(clippy::too_many_arguments)]
pub fn related(
    idx: &DocIndex,
    g: &crate::graph::store::GraphStore,
    node: Option<&str>,
    path: Option<&str>,
    n: Option<u64>,
    trace: &TraceLog,
    as_of: Option<&str>,
    stale: Option<&StaleChecker>,
) -> String {
    let (id, note) = match resolve_node_ref(idx, g, node, path, n) {
        Ok(x) => x,
        Err(m) => return m,
    };
    let at = match as_of.map(crate::graph::temporal::normalize_point).transpose() {
        Ok(a) => a,
        Err(e) => return format!("as_of error: {e}"),
    };
    let visible =
        |nid: &str| at.as_deref().is_none_or(|a| g.visible_at(nid, a).unwrap_or(true));
    let mut seen = std::collections::HashSet::new();
    let mut lines = Vec::new();
    for e in g.outgoing(&id).unwrap_or_default() {
        if e.edge_type == "SIMILAR" && visible(&e.to) && seen.insert(e.to.clone()) {
            lines.push(format!(
                "SIMILAR  {}{}{}",
                endpoint_ref(idx, g, &e.to),
                meta_suffix(g, &e.to, stale),
                read_anchor(idx, g, &e.to)
            ));
        }
    }
    for e in g.incoming(&id).unwrap_or_default() {
        if e.edge_type == "SIMILAR" && visible(&e.from) && seen.insert(e.from.clone()) {
            lines.push(format!(
                "SIMILAR  {}{}{}",
                endpoint_ref(idx, g, &e.from),
                meta_suffix(g, &e.from, stale),
                read_anchor(idx, g, &e.from)
            ));
        }
    }
    let similar_count = lines.len();
    if let Ok(Some(meta)) = g.node_meta(&id) {
        if let Some(comm) = meta.community {
            if let Ok(siblings) = g.community_siblings(comm, &id, COMMUNITY_TOP_LIMIT) {
                for (sib_id, _) in siblings {
                    if visible(&sib_id) && seen.insert(sib_id.clone()) {
                        lines.push(format!(
                            "COMMUNITY  {}{}{}",
                            endpoint_ref(idx, g, &sib_id),
                            meta_suffix(g, &sib_id, stale),
                            read_anchor(idx, g, &sib_id),
                        ));
                    }
                }
            }
        }
    }
    trace.log(
        "related",
        json!({"id": id}),
        json!({"similar": similar_count, "community": lines.len() - similar_count}),
    );
    let body = if lines.is_empty() {
        "(no related cases)".to_string()
    } else {
        lines.join("\n")
    };
    prepend_note(note, body)
}

/// One structural-neighbor line: `<EDGE_TYPE> <arrow>  <endpoint><meta><read-anchor>`.
/// `forward=true` prints `->` (an outgoing edge), `false` prints `<-` (incoming).
#[allow(clippy::too_many_arguments)]
fn edge_line(
    idx: &DocIndex,
    g: &crate::graph::store::GraphStore,
    edge_type: &str,
    forward: bool,
    endpoint_id: &str,
    stale: Option<&StaleChecker>,
) -> String {
    let arrow = if forward { "->" } else { "<-" };
    format!(
        "{:<16}{}  {}{}{}",
        edge_type,
        arrow,
        endpoint_ref(idx, g, endpoint_id),
        meta_suffix(g, endpoint_id, stale),
        read_anchor(idx, g, endpoint_id),
    )
}

/// Structural 1-hop neighbors: the node's actual typed edges (e.g. REFERENCES, CONSTRAINED_BY),
/// each with its real direction. `direction` filters to outgoing (`"out"`), incoming (`"in"`) or
/// both; `edge_types` (when set) keeps only those relation names. `as_of` (when `Some`),
/// normalized via `temporal::normalize_point`, drops any endpoint outside its authored validity
/// interval (`GraphStore::visible_at`); a timeless endpoint (no validity row) is always shown.
/// This filters the *surrounding* graph only — the anchor node itself (resolved from
/// `node`/`path`+`n`) is always looked up and its edges enumerated regardless of the anchor's own
/// validity window. `stale` (when `Some`) flags any drifted-source endpoint with an inline
/// `⚠ stale` marker.
#[allow(clippy::too_many_arguments)]
pub fn neighbors(
    idx: &DocIndex,
    g: &crate::graph::store::GraphStore,
    node: Option<&str>,
    path: Option<&str>,
    n: Option<u64>,
    edge_types: Option<&[String]>,
    direction: &str,
    trace: &TraceLog,
    as_of: Option<&str>,
    stale: Option<&StaleChecker>,
) -> String {
    let (id, note) = match resolve_node_ref(idx, g, node, path, n) {
        Ok(x) => x,
        Err(m) => return m,
    };
    let at = match as_of.map(crate::graph::temporal::normalize_point).transpose() {
        Ok(a) => a,
        Err(e) => return format!("as_of error: {e}"),
    };
    let visible =
        |nid: &str| at.as_deref().is_none_or(|a| g.visible_at(nid, a).unwrap_or(true));
    let want = |et: &str| edge_types.is_none_or(|ts| ts.iter().any(|t| t == et));
    let mut lines = Vec::new();
    if direction != "in" {
        for e in g.outgoing(&id).unwrap_or_default() {
            if want(&e.edge_type) && visible(&e.to) {
                lines.push(edge_line(idx, g, &e.edge_type, true, &e.to, stale));
            }
        }
    }
    if direction != "out" {
        for e in g.incoming(&id).unwrap_or_default() {
            // A self-loop (from == id) is both outgoing and incoming; when the outgoing pass also
            // ran (direction "both") it was already printed, so don't emit it a second time.
            if e.from == id && direction != "in" {
                continue;
            }
            if want(&e.edge_type) && visible(&e.from) {
                lines.push(edge_line(idx, g, &e.edge_type, false, &e.from, stale));
            }
        }
    }
    trace.log(
        "neighbors",
        json!({"id": id, "direction": direction}),
        json!({"edges": lines.len()}),
    );
    let body = if lines.is_empty() {
        "(no structural edges)".to_string()
    } else {
        lines.join("\n")
    };
    prepend_note(note, body)
}

/// Default bridge budget when `bridge=true`: one cross-document hop covers almost all real
/// multi-hops; a bigger budget compounds fuzzy-match error, so it is not exposed as a caller knob
/// here — `bridge_budget >= 2` stays an explicit `traverse::reach` call, not the reader surface
/// (design spec §7.5).
const DEFAULT_REACH_BRIDGE_BUDGET: usize = 1;

/// The cross-document reasoning primitive (design spec §6): one traversal, two directions.
/// `to` omitted (all three `to_*` args empty) is **discovery/answering** — walk `relation` forward
/// from `from`, crossing document boundaries on shared salient mentions, and return every node it
/// reaches as a candidate answer; no answer is supplied by the caller. `to` given is
/// **verification** — does a grounded path from `from` to that specific candidate exist? Same BFS
/// engine (`crate::graph::traverse::reach`) both ways, `to` is simply optional.
///
/// `relation` fuzzy-matches an ontology edge type (`None` ⇒ all `chaining`-role relations,
/// undirected). `bridge` toggles the cross-document bridge: `false` ⇒ bridge_budget 0 (graph-only,
/// role-filtered, forward-first — no silent doc-crossing); `true` ⇒
/// [`DEFAULT_REACH_BRIDGE_BUDGET`]. `max_depth` is clamped to `[1, 12]`.
///
/// Render (spec §6/§7.5 transparency MUST): every discovered/verified path is a hop chain — the
/// `from` node's glossary-style handle, then one indented line per hop. An in-graph edge hop
/// renders with an arrow (`--REL-->` / `<--REL--`, the old `path` tool's convention); a
/// cross-document **bridge hop** renders as `↝ bridged on "<term>" (conf 0.NN)` so the reader
/// always sees WHERE and on what mention the reasoning crossed a document — never a silent jump.
/// Discovery with nothing found, or verify with no grounded path, returns a plain-English message
/// instead of an empty body.
#[allow(clippy::too_many_arguments)]
pub fn reach(
    idx: &DocIndex,
    g: &crate::graph::store::GraphStore,
    ont: &crate::graph::ontology::Ontology,
    from_node: Option<&str>,
    from_path: Option<&str>,
    from_n: Option<u64>,
    relation: Option<&str>,
    to_node: Option<&str>,
    to_path: Option<&str>,
    to_n: Option<u64>,
    max_depth: usize,
    bridge: bool,
    trace: &TraceLog,
) -> String {
    let (from, note_from) = match resolve_node_ref(idx, g, from_node, from_path, from_n) {
        Ok(x) => x,
        Err(m) => return format!("from: {m}"),
    };
    // `to` is optional: this is discovery mode only when the caller supplied no `to_*` ref at all.
    // Only attempt to resolve `to` when one was actually given, so a discovery call never sees a
    // spurious "to: ..." resolution error.
    let to_requested =
        to_node.is_some_and(|s| !s.trim().is_empty()) || (to_path.is_some() && to_n.is_some());
    let (to, note_to) = if to_requested {
        match resolve_node_ref(idx, g, to_node, to_path, to_n) {
            Ok((id, note)) => (Some(id), note),
            Err(m) => return format!("to: {m}"),
        }
    } else {
        (None, None)
    };
    let notes: Vec<String> = [note_from, note_to].into_iter().flatten().collect();
    let note = (!notes.is_empty()).then(|| notes.join("\n"));
    let depth = max_depth.clamp(1, 12);
    let budget = if bridge { DEFAULT_REACH_BRIDGE_BUDGET } else { 0 };

    let body = match crate::graph::traverse::reach(g, ont, &from, relation, to.as_deref(), depth, budget)
    {
        Ok(res) => {
            trace.log(
                "reach",
                json!({"from": from, "to": to, "relation": relation, "bridge": bridge}),
                json!({"targets": res.targets.len(), "paths": res.paths.len()}),
            );
            render_reach_result(idx, g, &from, to.as_deref(), relation, depth, &res)
        }
        Err(e) => format!("reach error: {e}"),
    };
    prepend_note(note, body)
}

/// One hop-chain of a [`reach`] result: the anchor node's handle, then one indented line per hop —
/// shared by discovery (one chain per discovered target) and verify (the single found path).
fn render_reach_chain(
    idx: &DocIndex,
    g: &crate::graph::store::GraphStore,
    hops: &[crate::graph::traverse::Hop],
) -> String {
    use crate::graph::traverse::HopVia;
    let mut out = Vec::with_capacity(hops.len());
    for (i, h) in hops.iter().enumerate() {
        if i == 0 {
            out.push(format!(
                "{}{}",
                endpoint_ref(idx, g, &h.node),
                read_anchor(idx, g, &h.node)
            ));
            continue;
        }
        let arrow = match h.via.as_ref().expect("non-first reach hop has via") {
            HopVia::Edge { edge_type, forward: true } => format!("--{edge_type}-->"),
            HopVia::Edge { edge_type, forward: false } => format!("<--{edge_type}--"),
            HopVia::Bridge { term, confidence } => {
                format!("↝ bridged on \"{term}\" (conf {confidence:.2})")
            }
        };
        out.push(format!(
            "  {}  {}{}",
            arrow,
            endpoint_ref(idx, g, &h.node),
            read_anchor(idx, g, &h.node)
        ));
    }
    out.join("\n")
}

/// Render a [`crate::graph::traverse::ReachResult`]: verify-mode (`to = Some`) shows the single
/// grounding path, or a "no grounded path" message when none closes; discovery mode (`to = None`)
/// shows one hop chain per discovered target (blank-line separated), or a "no reachable targets"
/// message when the walk finds nothing.
fn render_reach_result(
    idx: &DocIndex,
    g: &crate::graph::store::GraphStore,
    from: &str,
    to: Option<&str>,
    relation: Option<&str>,
    depth: usize,
    res: &crate::graph::traverse::ReachResult,
) -> String {
    match to {
        Some(target) => match res.paths.first() {
            Some(hops) => render_reach_chain(idx, g, hops),
            None => format!("no grounded path from {from} to {target} within depth {depth}"),
        },
        None => {
            if res.paths.is_empty() {
                format!(
                    "no reachable targets from {from} along {} within depth {depth}",
                    relation.unwrap_or("any chaining relation")
                )
            } else {
                res.paths
                    .iter()
                    .map(|hops| render_reach_chain(idx, g, hops))
                    .collect::<Vec<_>>()
                    .join("\n\n")
            }
        }
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
pub fn checklist_coverage_report(g: &crate::graph::store::GraphStore, doc: &str) -> String {
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
            Ok(pdf) if (pdf.len() as u64) <= max_bytes => {
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
            Ok(_) => {
                // The converted PDF is over the cap. Don't deliver an over-cap artifact; keep the
                // original .docx so the under-cap whole-file path can still deliver it (the file
                // was deliverable before this feature). raw: true always returns the original.
                convert_note = Some(format!(
                    "{real} converted to a PDF larger than the {max_bytes}-byte cap; delivering the original .docx instead. Pass raw: true to always get the original."
                ));
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
            file: Some(SourceBlob {
                mime: source_mime(&name).to_string(),
                filename: name,
                bytes,
            }),
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

/// Fuzzy read-only SQL query over the reasoning graph. Traces the call and delegates to
/// `crate::graph::query::run`, which handles empty queries (returns schema), fuzzy literal
/// resolution, and rendering.
pub fn graph_query(idx: &DocIndex, g: &crate::graph::store::GraphStore, sql: &str, trace: &TraceLog) -> String {
    let body = crate::graph::query::run(g, idx, sql);
    trace.log("graph_query", json!({"sql": sql}), json!({"result": body}));
    body
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
            doc_path: PathBuf::from("MODULE.pdf"),
            location: "p.7".into(),
            file_type: "pdf".into(),
            text: "response timeout param equals 3000".into(),
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
        let out = get_source_file(
            &i,
            None,
            "report.docx",
            None,
            DEFAULT_SOURCE_MAX_BYTES,
            false,
        );
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
        let out = get_source_file(
            &i,
            None,
            "report.docx",
            None,
            DEFAULT_SOURCE_MAX_BYTES,
            true,
        );
        let f = out.file.expect("file delivered");
        assert_eq!(f.filename, "report.docx");
        assert_eq!(
            f.mime,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        );
        assert!(out.text.contains("whole file"));
    }

    #[test]
    fn get_source_file_over_cap_conversion_falls_back_to_original_docx() {
        let (d, i) = docx_corpus();
        let docx = std::fs::read(d.path().join("report.docx")).unwrap();
        let pdf = crate::convert::docx_pdf::docx_to_pdf(&docx).unwrap();
        // Only meaningful when the converted PDF is heavier than the source docx.
        if pdf.len() as u64 > docx.len() as u64 {
            let max = docx.len() as u64; // PDF over cap, original docx fits (==)
            let out = get_source_file(&i, None, "report.docx", None, max, false);
            let f = out.file.expect("original docx delivered as fallback");
            assert_eq!(f.filename, "report.docx");
            assert_eq!(
                f.mime,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            );
            assert!(f.bytes.starts_with(b"PK"), "docx is a zip archive");
            assert!(out.text.contains("original .docx"));
        }
    }

    #[test]
    fn get_source_file_delivers_whole_pdf_under_cap() {
        let (_d, i) = pdf_corpus();
        let out = get_source_file(
            &i,
            None,
            "sample.pdf",
            Some(1),
            DEFAULT_SOURCE_MAX_BYTES,
            false,
        );
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
        std::fs::write(
            d.path().join("notes.md"),
            "# Title\n\nplenty of text here\n",
        )
        .unwrap();
        crate::index::store::index_dir(d.path(), true).unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        let out = get_source_file(&i, None, "notes.md", Some(1), 4, false);
        assert!(out.file.is_none());
        assert!(out.text.contains("not a PDF"));
    }

    #[test]
    fn get_source_file_unknown_path_errs() {
        let (_d, i) = pdf_corpus();
        let out = get_source_file(
            &i,
            None,
            "nope.pdf",
            Some(1),
            DEFAULT_SOURCE_MAX_BYTES,
            false,
        );
        assert!(out.file.is_none());
        assert!(out.text.contains("no document indexed"));
    }

    #[test]
    fn read_page_image_returns_pdf_page_jpeg_without_chunk_body() {
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
        assert_eq!(out.images[0].mime, "image/jpeg");
        assert!(out.images[0].bytes.starts_with(b"\xff\xd8"));
    }

    #[test]
    fn grep_no_match_steers_to_a_simpler_pattern() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("doc.md"), b"# Doc\nthe quick brown fox\n").unwrap();
        crate::index::store::index_dir(d.path(), true).unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        // An over-built regex (the 4b's failure mode) matches nothing — the return must coach it
        // back to a simple literal instead of a bare "(no matches)".
        let out = grep(
            d.path(),
            &i,
            "quick.*purple.*elephant|elephant.*quick",
            &GrepOpts::default(),
            &TraceLog::disabled(),
        );
        assert!(out.contains("no matches"), "still reports the miss: {out}");
        assert!(
            out.contains("fixed:true") && out.contains("distinctive"),
            "coaches toward a simpler literal: {out}"
        );
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
        assert_eq!(out.images[0].mime, "image/jpeg");
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
        assert!(
            out.text.contains("50") && out.text.contains("63"),
            "got: {}",
            out.text
        );
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
        assert!(
            miss.text.contains("no document indexed") || miss.text.contains("not found"),
            "got: {}",
            miss.text
        );
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
            node("sym:loss", "Symptom", "Bus link dropout"),
            node("cau:tsdr", "Cause", "Low respTimeout"),
            node("res:set", "Resolution", "Change respTimeout to 3000"),
            node("tsk:upd", "Task", "Update module firmware"),
            node("tsk:prog", "Task", "Program the controller"),
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
    fn glossary_as_of_omits_node_outside_validity_window() {
        let (_d, i) = idx();
        let (_gd, g) = reasoning_graph();
        // sym:loss is authored as valid only 2020-01-01 .. 2023-12-31.
        g.upsert_validity(
            "sym:loss",
            &crate::graph::store::NodeValidity {
                valid_from: Some(crate::graph::temporal::normalize_from("2020-01-01").unwrap()),
                valid_to: Some(crate::graph::temporal::normalize_to("2023-12-31").unwrap()),
                valid_from_raw: Some("2020-01-01".into()),
                valid_to_raw: Some("2023-12-31".into()),
            },
        )
        .unwrap();
        let t = TraceLog::disabled();
        // No as_of: node is visible as usual.
        let out = glossary(&i, &g, "Bus link dropout", &spine_spec(), &t, None, None);
        assert!(out.contains("[Symptom]"), "{out}");
        // as_of after the validity window: the node is filtered out entirely.
        let out = glossary(&i, &g, "Bus link dropout", &spine_spec(), &t, Some("2024-01-01"), None);
        assert_eq!(out, "(no matches)", "{out}");
        // as_of inside the window: still visible.
        let out = glossary(&i, &g, "Bus link dropout", &spine_spec(), &t, Some("2022-06-01"), None);
        assert!(out.contains("[Symptom]"), "{out}");
    }

    #[test]
    fn read_of_node_shows_validity_and_status() {
        let (_d, i) = idx();
        let (_gd, g) = reasoning_graph();
        g.upsert_validity(
            "sym:loss",
            &crate::graph::store::NodeValidity {
                valid_from: Some(crate::graph::temporal::normalize_from("2020-01-01").unwrap()),
                valid_to: Some(crate::graph::temporal::normalize_to("2023-12-31").unwrap()),
                valid_from_raw: Some("2020-01-01".into()),
                valid_to_raw: Some("2023-12-31".into()),
            },
        )
        .unwrap();
        let t = TraceLog::disabled();
        let out = read(_d.path(), &i, Some(&g), "sym:loss", 1, false, &t);
        assert!(
            out.text.contains("valid:  2020-01-01 .. 2023-12-31"),
            "{}",
            out.text
        );
        // "now" (real system time) is long past the 2023-12-31 valid_to.
        assert!(out.text.contains("status: expired"), "{}", out.text);
    }

    #[test]
    fn read_of_node_without_validity_row_omits_status() {
        let (_d, i) = idx();
        let (_gd, g) = reasoning_graph();
        let t = TraceLog::disabled();
        let out = read(_d.path(), &i, Some(&g), "sym:loss", 1, false, &t);
        assert!(!out.text.contains("valid:"), "{}", out.text);
        assert!(!out.text.contains("status:"), "{}", out.text);
    }

    #[test]
    fn glossary_renders_full_chain_in_one_call() {
        let (_d, i) = idx();
        let (_gd, g) = reasoning_graph();
        let t = TraceLog::disabled();
        let out = glossary(&i, &g, "Bus link dropout", &spine_spec(), &t, None, None);
        // entry node + the whole chain to the resolution, surfaced by a SINGLE call
        assert!(
            out.contains("[Symptom]") && out.contains("Bus link dropout"),
            "{out}"
        );
        assert!(
            out.contains("→ CAUSED_BY") && out.contains("[Cause]") && out.contains("Low respTimeout"),
            "{out}"
        );
        assert!(
            out.contains("→ RESOLVED_BY")
                && out.contains("[Resolution]")
                && out.contains("Change respTimeout"),
            "{out}"
        );
        // SIMILAR is no longer part of glossary — it moved to related
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
        let out = glossary(&i, &g, "Bus link dropout", &empty, &t, None, None);
        assert!(out.contains("[Symptom]"), "{out}");
        assert!(
            !out.contains("CAUSED_BY") && !out.contains("RESOLVED_BY"),
            "no spines → no chain: {out}"
        );
    }

    #[test]
    fn glossary_marks_stale_source() {
        // A grounded node whose stored file_sig no longer matches its source file (drifted
        // AFTER grounding) → glossary flags it inline with "⚠ stale". A node grounded against a
        // source that hasn't changed → no marker, even when the same StaleChecker is used.
        let (_d, i) = idx();
        let root = tempfile::tempdir().unwrap();
        let drifted_path = root.path().join("drifted.md");
        let fresh_path = root.path().join("fresh.md");
        std::fs::write(&drifted_path, b"v1").unwrap();
        std::fs::write(&fresh_path, b"unchanged").unwrap();
        let drifted_sig = crate::index::store::file_sig(&drifted_path).unwrap();
        let fresh_sig = crate::index::store::file_sig(&fresh_path).unwrap();

        let g = GraphStore::open(root.path()).unwrap();
        let mk_prov = |source_path: &str, sig| Provenance {
            source_path: source_path.into(),
            range: None,
            file_sig: Some(sig),
            origin: "agent".into(),
            confidence: 0.8,
            created_at: 1,
        };
        g.put_node(&Node {
            id: "sym:drift".into(),
            node_type: "Symptom".into(),
            label: "Drifted symptom".into(),
            aliases: Vec::new(),
            prov: mk_prov("drifted.md", drifted_sig),
        })
        .unwrap();
        g.put_node(&Node {
            id: "sym:fresh".into(),
            node_type: "Symptom".into(),
            label: "Fresh symptom".into(),
            aliases: Vec::new(),
            prov: mk_prov("fresh.md", fresh_sig),
        })
        .unwrap();

        // Drift drifted.md AFTER grounding; fresh.md is left untouched.
        std::fs::write(&drifted_path, b"v2-longer-content").unwrap();

        let t = TraceLog::disabled();
        let stale = StaleChecker::new(root.path());

        let out = glossary(&i, &g, "Drifted symptom", &ChainSpec::default(), &t, None, Some(&stale));
        assert!(out.contains("⚠ stale"), "expected stale marker: {out}");

        let out = glossary(&i, &g, "Fresh symptom", &ChainSpec::default(), &t, None, Some(&stale));
        assert!(!out.contains("⚠ stale"), "unexpected stale marker: {out}");

        // No checker at all (as production callers that lack a root may pass) → no marker either.
        let out = glossary(&i, &g, "Drifted symptom", &ChainSpec::default(), &t, None, None);
        assert!(!out.contains("⚠ stale"), "stale check must be opt-in: {out}");
    }

    #[test]
    fn related_shows_similar_generalization_for_a_node_id() {
        let (_d, i) = idx();
        let (_gd, g) = reasoning_graph();
        let t = TraceLog::disabled();
        let out = related(&i, &g, Some("sym:loss"), None, None, &t, None, None);
        assert!(
            out.contains("SIMILAR")
                && out.contains("Update module firmware")
                && out.contains("Program the controller"),
            "{out}"
        );
        // it must NOT walk the causal spine — that is glossary's job
        assert!(
            !out.contains("CAUSED_BY") && !out.contains("[Resolution]"),
            "related is generalization only: {out}"
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
            node("sym:a", "Symptom", "Bus link dropout"),
            node("cau:a", "Cause", "Low respTimeout"),
            node("res:a", "Resolution", "Raise respTimeout"),
            node("sym:b", "Symptom", "Modbus poll timeout"),
            node("cau:b", "Cause", "Wrong baud"),
            node("res:b", "Resolution", "Change port speed"),
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
    fn related_shows_community_siblings_after_generalize() {
        let (_d, i) = idx();
        let (_gd, g) = two_component_reasoning_graph();
        let t = TraceLog::disabled();
        let out = related(&i, &g, Some("sym:a"), None, None, &t, None, None);
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
    fn related_dedups_similar_from_community_section() {
        let (_d, i) = idx();
        let (_gd, g) = reasoning_graph();
        let ont = Ontology::parse(REASONING_ONT).unwrap();
        let opts = crate::graph::generalize::apply::Opts::from_ontology(&ont, 100);
        crate::graph::generalize::apply::generalize(&g, &opts).unwrap();
        let t = TraceLog::disabled();
        let out = related(&i, &g, Some("sym:loss"), None, None, &t, None, None);
        assert!(
            out.contains("SIMILAR") && out.contains("Update module firmware"),
            "{out}"
        );
        let community_lines: Vec<_> = out.lines().filter(|l| l.starts_with("COMMUNITY")).collect();
        assert!(
            !community_lines
                .iter()
                .any(|l| l.contains("Update module firmware")
                    || l.contains("Program the controller")),
            "SIMILAR nodes must not repeat in COMMUNITY: {out}",
        );
    }

    #[test]
    fn related_community_sorted_by_pagerank() {
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
        let out = related(&i, &g, Some("sym:loss"), None, None, &t, None, None);
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
    fn neighbors_lists_typed_edges_with_direction_and_filter() {
        let dir = tempfile::tempdir().unwrap();
        let idx = crate::index::store::DocIndex::open_or_create(dir.path()).unwrap();
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        for id in ["a", "b", "c"] {
            g.put_node(&node(id, "Entity", id)).unwrap();
        }
        g.put_edge(&edge("a", "REFERENCES", "b")).unwrap();
        g.put_edge(&edge("c", "CONSTRAINED_BY", "a")).unwrap();

        let both = neighbors(
            &idx,
            &g,
            Some("a"),
            None,
            None,
            None,
            "both",
            &TraceLog::disabled(),
            None,
            None,
        );
        assert!(both.contains("REFERENCES") && both.contains("->"));
        assert!(both.contains("CONSTRAINED_BY") && both.contains("<-"));

        let out = neighbors(
            &idx,
            &g,
            Some("a"),
            None,
            None,
            None,
            "out",
            &TraceLog::disabled(),
            None,
            None,
        );
        assert!(out.contains("REFERENCES") && !out.contains("CONSTRAINED_BY"));

        let filtered = neighbors(
            &idx,
            &g,
            Some("a"),
            None,
            None,
            Some(&["CONSTRAINED_BY".to_string()]),
            "both",
            &TraceLog::disabled(),
            None,
            None,
        );
        assert!(filtered.contains("CONSTRAINED_BY") && !filtered.contains("REFERENCES"));

        let empty = neighbors(
            &idx,
            &g,
            Some("b"),
            None,
            None,
            Some(&["NOPE".to_string()]),
            "both",
            &TraceLog::disabled(),
            None,
            None,
        );
        assert_eq!(empty, "(no structural edges)");
    }

    #[test]
    fn neighbors_both_dedups_self_loop() {
        let dir = tempfile::tempdir().unwrap();
        let idx = crate::index::store::DocIndex::open_or_create(dir.path()).unwrap();
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        g.put_node(&node("a", "Entity", "a")).unwrap();
        // A self-loop is both an outgoing and an incoming edge of `a`.
        g.put_edge(&edge("a", "LINKS", "a")).unwrap();

        // `both` must not print the self-loop twice.
        let both = neighbors(
            &idx,
            &g,
            Some("a"),
            None,
            None,
            None,
            "both",
            &TraceLog::disabled(),
            None,
            None,
        );
        assert_eq!(both.matches("LINKS").count(), 1, "self-loop rendered twice: {both}");

        // Direction-scoped views still show it once, with the matching arrow.
        let out = neighbors(
            &idx,
            &g,
            Some("a"),
            None,
            None,
            None,
            "out",
            &TraceLog::disabled(),
            None,
            None,
        );
        assert_eq!(out.matches("LINKS").count(), 1);
        assert!(out.contains("->"));
        let inc = neighbors(
            &idx,
            &g,
            Some("a"),
            None,
            None,
            None,
            "in",
            &TraceLog::disabled(),
            None,
            None,
        );
        assert_eq!(inc.matches("LINKS").count(), 1);
        assert!(inc.contains("<-"));
    }

    #[test]
    fn reach_tool_renders_chain_with_arrows_and_no_path_message() {
        // `reach(relation=None, bridge=false)` reproduces the old `path` tool's rendering exactly
        // (spec §6: `path` is a strict subset of `reach`).
        let dir = tempfile::tempdir().unwrap();
        let idx = crate::index::store::DocIndex::open_or_create(dir.path()).unwrap();
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        let ont = crate::graph::ontology::Ontology::default();
        for id in ["a", "b", "c", "d", "e"] {
            g.put_node(&node(id, "Entity", id)).unwrap();
        }
        // "e" is left isolated (no edges) — the "no path" case needs a real but unreachable node.
        g.put_edge(&edge("a", "REFERENCES", "b")).unwrap();
        g.put_edge(&edge("b", "REFERENCES", "c")).unwrap();
        // Edge points d->c; reaching d from c means traversing it against its stored
        // direction, so this hop must render as a reverse arrow (`<--...--`).
        g.put_edge(&edge("d", "CONSTRAINED_BY", "c")).unwrap();

        let out = reach(
            &idx,
            &g,
            &ont,
            Some("a"),
            None,
            None,
            None,
            Some("c"),
            None,
            None,
            6,
            false,
            &TraceLog::disabled(),
        );
        assert!(out.contains("--REFERENCES-->"), "got: {out}");
        // first line is the `from` node, no arrow prefix
        assert!(
            out.lines().next().unwrap().contains("a"),
            "got: {out}"
        );

        let reverse = reach(
            &idx,
            &g,
            &ont,
            Some("a"),
            None,
            None,
            None,
            Some("d"),
            None,
            None,
            6,
            false,
            &TraceLog::disabled(),
        );
        assert!(
            reverse.contains("<--CONSTRAINED_BY--"),
            "expected a reverse-direction hop: {reverse}"
        );

        let none = reach(
            &idx,
            &g,
            &ont,
            Some("a"),
            None,
            None,
            None,
            Some("e"),
            None,
            None,
            6,
            false,
            &TraceLog::disabled(),
        );
        assert!(none.starts_with("no grounded path"), "got: {none}");

        // max_depth is clamped to [1,12]; 999 must not panic and still finds it.
        let clamped = reach(
            &idx,
            &g,
            &ont,
            Some("a"),
            None,
            None,
            None,
            Some("c"),
            None,
            None,
            999,
            false,
            &TraceLog::disabled(),
        );
        assert!(clamped.contains("--REFERENCES-->"));
    }

    /// Add `n` distinct filler nodes so the corpus is large enough that a df=2 mention clears the
    /// `reach` salience/IDF gate (mirrors `traverse::tests::fillers`).
    fn reach_dnode(g: &GraphStore, id: &str, label: &str, doc: &str) {
        g.put_node(&Node {
            id: id.into(),
            node_type: "Entity".into(),
            label: label.into(),
            aliases: vec![],
            prov: Provenance {
                source_path: doc.into(),
                range: None,
                file_sig: None,
                origin: "agent".into(),
                confidence: 1.0,
                created_at: 0,
            },
        })
        .unwrap();
    }
    fn reach_fillers(g: &GraphStore, n: usize) {
        for i in 0..n {
            reach_dnode(g, &format!("rfiller{i}"), &format!("Fillerword{i}"), "filler.md");
        }
    }

    #[test]
    fn reach_tool_discovery_bridges_and_renders_target_handle_with_bridge_hop() {
        // docA fact mentions salient term "Xanthium" and dead-ends in-doc (only a grounding edge);
        // docB has a same-mention node with a chaining REL edge to the answer. `bridge=true` must
        // discover it across docs; the render must show the bridge hop with its term + confidence
        // AND the discovered target as a glossary-style handle. `bridge=false` must not cross.
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::default();
        reach_dnode(&g, "A_fact", "Xanthium", "docA.md");
        reach_dnode(&g, "A_ent", "Local note", "docA.md");
        g.put_edge(&edge("A_fact", "MENTIONS", "A_ent")).unwrap(); // grounding, not a chain hop
        reach_dnode(&g, "B_T", "Xanthium", "docB.md");
        reach_dnode(&g, "Target", "The Answer", "docB.md");
        g.put_edge(&edge("B_T", "REL", "Target")).unwrap();
        reach_fillers(&g, 25);

        let out = reach(
            &idx,
            &g,
            &ont,
            Some("A_fact"),
            None,
            None,
            Some("REL"),
            None,
            None,
            None,
            6,
            true,
            &TraceLog::disabled(),
        );
        assert!(out.contains("Target"), "expected the discovered target handle, got: {out}");
        assert!(out.contains("[Entity]"), "expected a glossary-style handle, got: {out}");
        assert!(
            out.contains("↝ bridged on \"Xanthium\""),
            "expected an explicit bridge hop naming its term, got: {out}"
        );
        assert!(out.contains("conf 0."), "expected a bridge confidence value, got: {out}");

        let no_bridge = reach(
            &idx,
            &g,
            &ont,
            Some("A_fact"),
            None,
            None,
            Some("REL"),
            None,
            None,
            None,
            6,
            false,
            &TraceLog::disabled(),
        );
        assert!(
            !no_bridge.contains("Target"),
            "bridge=false must not discover a cross-doc target, got: {no_bridge}"
        );
        assert!(
            no_bridge.starts_with("no reachable targets"),
            "expected a clear no-targets message, got: {no_bridge}"
        );
    }

    #[test]
    fn reach_tool_verify_mode_returns_path_or_no_grounded_path() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::default();
        reach_dnode(&g, "A_fact", "Xanthium", "docA.md");
        reach_dnode(&g, "B_T", "Xanthium", "docB.md");
        reach_dnode(&g, "Target", "The Answer", "docB.md");
        g.put_edge(&edge("B_T", "REL", "Target")).unwrap();
        // Only co-mentioned via a different, unrelated term — never chaining-connected.
        reach_dnode(&g, "Distractor", "Unrelated", "docC.md");
        reach_fillers(&g, 25);

        // A grounded candidate: reach discovers and returns the bridged path.
        let hit = reach(
            &idx,
            &g,
            &ont,
            Some("A_fact"),
            None,
            None,
            Some("REL"),
            Some("Target"),
            None,
            None,
            6,
            true,
            &TraceLog::disabled(),
        );
        assert!(hit.contains("Target"), "got: {hit}");
        assert!(
            hit.contains("↝ bridged on"),
            "verify path should show the bridge hop, got: {hit}"
        );

        // A candidate with no chaining path to it — no grounded path, not a silent empty body.
        let miss = reach(
            &idx,
            &g,
            &ont,
            Some("A_fact"),
            None,
            None,
            Some("REL"),
            Some("Distractor"),
            None,
            None,
            6,
            true,
            &TraceLog::disabled(),
        );
        assert!(
            miss.starts_with("no grounded path"),
            "expected a clear no-grounded-path message, got: {miss}"
        );
    }

    #[test]
    fn resolve_node_ref_prefers_explicit_node_id() {
        use crate::index::store::index_dir;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("g.md"), b"# Doc\nSome text.\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        g.put_node(&node("fact:x", "Fact", "Some fact")).unwrap();

        // An explicit, existing node id resolves to itself with no note.
        let (id, note) = resolve_node_ref(&idx, &g, Some("fact:x"), None, None).unwrap();
        assert_eq!(id, "fact:x");
        assert!(note.is_none());
        // Missing both node and (path, n) is an error.
        assert!(resolve_node_ref(&idx, &g, None, None, None).is_err());
    }

    #[test]
    fn resolve_node_ref_fuzzy_resolves_name_and_reports_it() {
        use crate::index::store::index_dir;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("g.md"), b"# Doc\nSome text.\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        g.put_node(&node(
            "fact:helio",
            "Fact",
            "Heliocentrism was proposed in the 3rd century BC",
        ))
        .unwrap();

        // A weak model can't reproduce a hashed id, so it passes the name instead. It must resolve
        // to the node AND report the substitution — never a silent swap.
        let (rid, rnote) =
            resolve_node_ref(&idx, &g, Some("Heliocentrism"), None, None).unwrap();
        assert_eq!(rid, "fact:helio", "resolved the name to the node");
        let rnote = rnote.expect("a fuzzy substitution must be reported");
        assert!(
            rnote.contains("Heliocentrism") && rnote.contains("fact:helio"),
            "note names both the query and the chosen node: {rnote}"
        );

        // A genuine miss is an error, not a wrong node silently returned.
        assert!(resolve_node_ref(&idx, &g, Some("zzz-nope-xyz"), None, None).is_err());
    }

    #[test]
    fn graph_stats_lists_communities_with_top_examples() {
        use crate::graph::store::NodeMeta;
        let gd = tempfile::tempdir().unwrap();
        let g = GraphStore::open(gd.path()).unwrap();
        for n in [
            node("sym:a", "Symptom", "Fieldbus loss"),
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
            out.contains("comm 0  (1 nodes)") && out.contains("Fieldbus loss"),
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
        let (_d, i) = idx(); // MODULE.pdf #p.7 = "response timeout param equals 3000"
        let gd = tempfile::tempdir().unwrap();
        let g = GraphStore::open(gd.path()).unwrap();
        g.put_node(&node("res:fix", "Resolution", "Change respTimeout to 3000"))
            .unwrap();
        g.put_edge(&edge("res:fix", "MENTIONS", "MODULE.pdf#p.7"))
            .unwrap();
        let t = TraceLog::disabled();
        // reading the NODE id returns the node line + the evidence chunk it MENTIONS, attributed.
        let out = read(_d.path(), &i, Some(&g), "res:fix", 1, false, &t).text;
        assert!(
            out.contains("[Resolution]") && out.contains("Change respTimeout"),
            "node header: {out}"
        );
        assert!(
            out.contains("── MENTIONS · MODULE.pdf #7 ──"),
            "attributed evidence header: {out}"
        );
        assert!(
            out.contains("response timeout param equals 3000"),
            "evidence body: {out}"
        );
        // a plain doc path still reads the chunk (not treated as a node).
        let doc = read(_d.path(), &i, Some(&g), "MODULE.pdf", 7, false, &t).text;
        assert!(
            doc.contains("response timeout param equals 3000") && !doc.contains("── MENTIONS"),
            "doc read unchanged: {doc}"
        );
    }

    #[test]
    fn search_renders_numbered_or_empty() {
        let (_d, i) = idx();
        let t = TraceLog::disabled();
        let (body, hits) = search(&i, "timeout", 10, None, None, &t);
        assert_eq!(hits.len(), 1);
        assert!(body.starts_with("[#7] ") && body.contains("timeout"));
        let (empty, _) = search(&i, "nonexistentzzz", 10, None, None, &t);
        assert_eq!(empty, "(no results)");
    }

    #[test]
    fn read_returns_full_body_and_unified_footer() {
        let d = tempfile::tempdir().unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        let big = "X".repeat(5000); // > old 4000-char cap
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
        // Doubled / swapped separators (model over-escapes `\\` or uses `\`) also resolve
        // via normalize_path() inside resolve_path.
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
    fn path_hints_recover_underscored_and_mangled_basename() {
        let d = tempfile::tempdir().unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        // The real (spaced) PDF, plus underscore-named siblings that mislead the model into
        // regularizing the spaced name and — separately — mistyping one word's ending.
        for p in [
            "Updated manuals/Guide for setup and programming ACME PLC_v_1_12.pdf",
            "v_3.0/1_1_overview.htm",
            "v_3.0/2_2_appearance.htm",
        ] {
            i.write_chunks(&[Chunk {
                doc_path: PathBuf::from(p),
                location: "p.1".into(),
                file_type: "pdf".into(),
                text: "x".into(),
            }])
            .unwrap();
        }
        // Model swapped spaces for underscores AND mistyped programming → programing;
        // the literal `*base*` glob finds nothing, so the token-overlap fallback must still name
        // the real file (7 of 8 tokens overlap).
        let hint = document_path_hints(
            &i,
            "Updated manuals/Guide_for_setup_and_programing_ACME_PLC_v_1_12.pdf",
        );
        assert!(
            hint.contains("Guide for setup and programming ACME PLC_v_1_12.pdf"),
            "token-overlap hint should surface the real spaced file: {hint}"
        );
        // A basename that shares no real tokens with any indexed doc yields no suggestion.
        assert_eq!(document_path_hints(&i, "totally_unrelated_xyzzy.txt"), "");
    }

    #[test]
    fn read_tolerates_spurious_leading_prefix() {
        let d = tempfile::tempdir().unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        i.write_chunks(&[Chunk {
            doc_path: PathBuf::from("Archive DB\\doc.pdf"),
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
            "kb-manual\\Archive DB\\doc.pdf",
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
            doc_path: PathBuf::from("Archive DB\\Calibration Guide PLCModule 2025.pdf"),
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
            "kb-manual\\PLCModule 2025.pdf",
            1,
            false,
            &t,
        )
        .text;
        assert!(out.contains("did you mean"), "hint on miss: {out}");
        assert!(out.contains("Calibration"), "suggest real path: {out}");
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
        assert!(grep(
            d.path(),
            &i,
            "timeout",
            &crate::grep::GrepOpts::default(),
            &t
        )
        .contains(":#7:"));
        let no = grep(d.path(), &i, "nomatchzzz", &crate::grep::GrepOpts::default(), &t);
        assert!(no.starts_with("(no matches)"), "still reports the miss: {no}");
        assert!(no.contains("fixed:true"), "coaches toward a simpler literal: {no}");
        assert!(glob(&i, "*MODULE*", &t).contains("MODULE.pdf  (7 chunks)"));
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
        assert!(
            out.contains("63"),
            "notebook grep should find the line; got: {out}"
        );
        assert!(
            !out.contains("no document indexed"),
            "must not error on a notebook path"
        );
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
        let result = glossary(&idx, &g, "ACME", &ChainSpec::default(), &t, None, None);
        assert!(
            result.contains("org:acme"),
            "expected node id in result, got: {result}"
        );
        assert_eq!(
            glossary(&idx, &g, "nonesuch", &ChainSpec::default(), &t, None, None),
            "(no matches)"
        );
    }

    #[test]
    fn glossary_omits_similar_derived_meta_even_after_generalize() {
        // community/pagerank/degree are SIMILAR-derived — they belong to `related`, not glossary's
        // reasoning surface. glossary must NOT surface them even once node_meta is populated.
        let (_dir, g) = graph_fixture();
        let idir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(idir.path()).unwrap();
        let t = TraceLog::disabled();

        let opts = crate::graph::generalize::apply::Opts::defaults(1);
        crate::graph::generalize::apply::generalize(&g, &opts).unwrap();

        let after = glossary(&idx, &g, "ACME", &ChainSpec::default(), &t, None, None);
        assert!(after.contains("org:acme"), "still shows the node id: {after}");
        assert!(
            !after.contains("comm ") && !after.contains(" pr ") && !after.contains(" deg "),
            "glossary must not surface SIMILAR-derived community/pagerank/degree: {after}"
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
        g.put_node(&node("sym:loss", "Symptom", "Link dropout"))
            .unwrap();
        g.put_node(&node("cau:maxtsdr", "Cause", "Low respTimeout"))
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
        let out = glossary(&idx, &g, "Link dropout", &spine_spec(), &t, None, None);
        assert!(out.contains("sym:loss"), "shows the matched node id: {out}");
        assert!(
            out.contains("→ CAUSED_BY") && out.contains("[Cause]"),
            "walks the CAUSED_BY hop: {out}"
        );
        assert!(
            out.contains("Low respTimeout"),
            "shows the connected Cause label: {out}"
        );
    }

    // Structural chunk-navigation tests (NEXT/CHILD/PARENT/PREV/REFERENCES, node_meta on a
    // section) were removed when `related` became the generalization view (SIMILAR cross-links);
    // adjacent-chunk navigation now lives in `read`'s prev/next footer. The generalization
    // behavior is covered by `related_shows_similar_generalization_for_a_node_id` and the
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
        let result = glossary(&idx, &g, "Intro", &ChainSpec::default(), &t, None, None);
        assert!(result.contains("#1"), "section rendered with ord: {result}");
        assert!(result.contains("Intro"), "section label present: {result}");

        assert_eq!(
            glossary(&idx, &g, "nonexistentzzz", &ChainSpec::default(), &t, None, None),
            "(no matches)"
        );
    }

    #[test]
    fn glossary_surfaces_facts_grounded_to_a_matched_section() {
        use crate::index::store::index_dir;

        // A common entity ("Sun") whose name exactly matches a section title. `resolve`'s
        // exact-label fast path returns ONLY that Section stub (never reaching BM25), so the
        // reasoning facts grounded to that chunk would be hidden. glossary must surface them.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("g.md"), b"# Sun\nThe Sun is a star.\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        let t = TraceLog::disabled();

        // The Section node id "Sun" resolves to (e.g. "g.md#1") — ground a reasoning fact to it.
        let sec_id = g.resolve("Sun").unwrap().into_iter().next().unwrap();
        g.put_node(&node(
            "fact:aristarchus",
            "Fact",
            "Heliocentrism was proposed as early as the 3rd century BC by Aristarchus",
        ))
        .unwrap();
        g.put_edge(&edge("fact:aristarchus", "MENTIONS", &sec_id)).unwrap();

        let out = glossary(&idx, &g, "Sun", &ChainSpec::default(), &t, None, None);
        assert!(
            out.contains("3rd century BC"),
            "fact grounded to the matched section must surface, not just the stub: {out}"
        );
    }

    #[test]
    fn glossary_expands_leads_to_of_section_grounded_facts() {
        use crate::index::store::index_dir;

        // A multi-hop answer lives one LEADS_TO away from an entity's own fact. When a common
        // entity name ("Sun") matches a Section, glossary surfaces the facts grounded to it — and
        // must ALSO expand their outgoing LEADS_TO one hop, so a weak reader sees the next fact in
        // its FIRST glossary call, without having to feed a node id to neighbors/path (which it
        // hallucinates). Same chain rendering the direct-node match already gets.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("g.md"), b"# Sun\nThe Sun is a star.\n").unwrap();
        index_dir(dir.path(), true).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();
        let t = TraceLog::disabled();

        let sec_id = g.resolve("Sun").unwrap().into_iter().next().unwrap();
        // A fact grounded to the "Sun" section ...
        g.put_node(&node(
            "fact:sun-center",
            "Fact",
            "The Sun is the star at the center of the Solar System",
        ))
        .unwrap();
        g.put_edge(&edge("fact:sun-center", "MENTIONS", &sec_id)).unwrap();
        // ... that LEADS_TO the answer fact (owned by another document).
        g.put_node(&node(
            "fact:origin",
            "Fact",
            "Heliocentrism was proposed as early as the 3rd century BC by Aristarchus",
        ))
        .unwrap();
        g.put_edge(&edge("fact:sun-center", "LEADS_TO", "fact:origin")).unwrap();

        // Real readers drive glossary with the ontology's spine (LEADS_TO here), not the empty
        // default — that is the spec under which the chain must expand.
        let spec = ChainSpec {
            spine_rels: vec!["LEADS_TO".to_string()],
        };
        let out = glossary(&idx, &g, "Sun", &spec, &t, None, None);
        assert!(
            out.contains("3rd century BC"),
            "glossary must expand the LEADS_TO neighbour of a section-grounded fact so the next \
             hop is visible in the first call, no node id needed: {out}"
        );
    }

    #[test]
    fn graph_query_wrapper_smoke_test() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = crate::graph::store::GraphStore::open(dir.path()).unwrap();

        // Add a couple of nodes to the graph
        g.put_node(&node("n1", "Entity", "First")).unwrap();
        g.put_node(&node("n2", "Entity", "Second")).unwrap();

        let t = TraceLog::disabled();

        // Empty call should return schema (contains "nodes(")
        let result = graph_query(&idx, &g, "", &t);
        assert!(
            result.contains("nodes("),
            "empty query should return schema: {result}"
        );

        // Real query should return something
        let query = "SELECT id, label FROM nodes LIMIT 1";
        let result = graph_query(&idx, &g, query, &t);
        assert!(
            !result.is_empty(),
            "query should return results: {result}"
        );
    }
}
