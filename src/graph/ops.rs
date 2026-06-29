//! Shared write operations for the reasoning graph.
//!
//! Both the MCP server (`src/mcp.rs`) and the kb-eval enricher (`eval/src/enrich.rs`)
//! funnel through these functions so validation and resolution behaviour is identical.

use crate::graph::agent::{apply_delete, apply_update, apply_upsert, EdgeRef, EdgeSpec, NodeSpec, NodeUpdate};
use crate::graph::ontology::Ontology;
use crate::graph::store::{GraphStore, normalize_label};
use crate::index::store::DocIndex;

// ── label-based input types ───────────────────────────────────────────────────

/// A reasoning node to create/update, identified by label only (no id).
/// The system derives the canonical id from `node_type` + `label` and deduplicates by label.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct UpsertNode {
    pub node_type: String,
    pub label: String,
    pub source_path: String,
    #[serde(default)]
    pub aliases: Vec<String>,
}

/// A directed edge identified by node labels (or section refs) at both endpoints.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct UpsertEdge {
    pub from: String,
    pub edge_type: String,
    pub to: String,
    pub source_path: String,
}

/// Derive the canonical node id from ontology prefix, type, and label.
/// The label is normalised (lowercase, collapsed whitespace) and spaces replaced with "-".
pub fn id_for(ont: &Ontology, node_type: &str, label: &str) -> String {
    let slug = normalize_label(label).replace(' ', "-");
    format!("{}:{}", ont.id_abbrev(node_type), slug)
}

/// Strip a mistaken type-prefix when the agent copied a graph id into the `label` field.
pub fn sanitize_label_for_upsert(ont: &Ontology, node_type: &str, label: &str) -> String {
    let prefix = format!("{}:", ont.id_abbrev(node_type));
    let trimmed = label.trim();
    if trimmed.len() > prefix.len() && trimmed[..prefix.len()].eq_ignore_ascii_case(&prefix) {
        trimmed[prefix.len()..].trim_start().to_string()
    } else {
        trimmed.to_string()
    }
}

// ── label helpers ─────────────────────────────────────────────────────────────

/// Resolve a section reference of the form `<path>#<n>` (a chunk number the agent has from a
/// read/search result) into the real Section node id `<path>#<location>`. Reasoning-node slugs
/// (`sym:…`) and already-resolved `<path>#<location>` refs (non-numeric suffix) pass through.
/// A numeric chunk ord that does NOT exist in the document is REJECTED with an actionable message
/// — so the model re-anchors to a real chunk instead of writing a dangling edge.
fn resolve_section_ref(idx: &DocIndex, s: &str) -> Result<String, String> {
    if let Some(pos) = s.rfind('#') {
        let suffix = &s[pos + 1..];
        if let Ok(n) = suffix.parse::<u64>() {
            let raw = &s[..pos];
            let path = idx
                .canonical_document_path(raw)
                .unwrap_or_else(|| raw.to_string());
            return match idx.location_for_ord(&path, n) {
                Ok(Some(loc)) => Ok(crate::graph::build::section_id(&path, &loc)),
                Ok(None) => Err(format!(
                    "chunk #{n} does not exist in {path}; take the chunk number from a search/grep/read on THIS document — never reuse a number from another file"
                )),
                Err(e) => Err(format!("could not resolve chunk #{n} in {path}: {e}")),
            };
        }
    }
    Ok(s.to_string())
}

/// Resolve an edge endpoint label to a node id: exact normalized-label match first, then a fuzzy
/// morphology match against existing reasoning nodes (the small model often paraphrases its own
/// label — a truncation or wording variant). Returns None when nothing matches.
fn resolve_endpoint_label(
    g: &GraphStore,
    label_to_id: &std::collections::HashMap<String, String>,
    label: &str,
) -> Option<String> {
    if g.get_node(label).ok().flatten().is_some() {
        return Some(label.to_string());
    }
    if let Some(id) = label_to_id.get(&normalize_label(label)) {
        return Some(id.clone());
    }
    // Exact match against EXISTING nodes via the label_norm index (replaces the old prebuilt
    // all_nodes map — same unfiltered "first exact" semantics, but O(log N) instead of O(N)).
    if let Some(id) = g.ids_by_label_norm(label).ok()?.into_iter().next() {
        return Some(id);
    }
    const STRUCTURAL: &[&str] = &["Document", "Section", "Term", "Topic"];
    let ids = g.resolve(label).ok()?;
    ids.into_iter()
        .filter_map(|id| g.get_node(&id).ok().flatten())
        .filter(|n| !STRUCTURAL.contains(&n.node_type.as_str()))
        // all morphology matches contain the query's terms; the SHORTEST label is the closest
        // superset of the model's (often truncated) reference.
        .min_by_key(|n| n.label.split_whitespace().count())
        .map(|n| n.id)
}

// ── public API ────────────────────────────────────────────────────────────────

/// Outcome of a `graph_upsert` operation.
pub struct UpsertOutcome {
    /// Formatted response for the model (via `format_upsert_response`).
    pub message: String,
    pub nodes: usize,
    pub edges: usize,
    /// True when the call was rejected (nothing was written).
    pub rejected: bool,
    /// Human-readable dump lines for logging:
    /// `"node <id> [<type>] <label>"` and `"edge <from> -<type>-> <to>"` (resolved).
    pub dump: Vec<String>,
    /// requested_id → canonical_id when dedup merged into an existing node
    pub merged: Vec<(String, String)>,
    /// Malformed items that were dropped (partial apply) or caused full rejection
    pub dropped: Vec<String>,
}

/// Build the human-readable tool response from an upsert outcome.
pub fn format_upsert_response(out: &UpsertOutcome) -> String {
    if out.rejected && out.nodes == 0 && out.edges == 0 {
        let mut s = String::from("REJECTED — nothing written");
        if !out.dropped.is_empty() {
            s.push_str("\n- ");
            s.push_str(&out.dropped.join("\n- "));
        }
        if !out.dump.is_empty() {
            // dump on full rejection holds the hint block appended by graph_upsert
            for line in &out.dump {
                if !line.is_empty() {
                    s.push('\n');
                    s.push_str(line);
                }
            }
        }
        return s;
    }

    let mut parts = vec![format!("upserted {} nodes, {} edges", out.nodes, out.edges)];
    if !out.dump.is_empty() {
        parts.push("Written:".into());
        for line in &out.dump {
            parts.push(format!("- {line}"));
        }
    }
    if !out.merged.is_empty() {
        parts.push("Merged (already existed, edges attached):".into());
        for (from, to) in &out.merged {
            parts.push(format!("- {from} → {to}"));
        }
    }
    if !out.dropped.is_empty() {
        parts.push(format!(
            "{} item(s) dropped (the rest WERE written) — fix and resend only these:",
            out.dropped.len()
        ));
        for e in &out.dropped {
            parts.push(format!("- {e}"));
        }
    }
    parts.join("\n")
}

/// Validate, resolve, and apply a graph upsert.
///
/// The caller provides node LABELS only (no ids). The system derives canonical ids via
/// `id_for(node_type, label)` and deduplicates by label. Edges reference nodes by their label
/// (or a section as `<path>#<n>`).
///
/// Steps:
/// 1. Validate each node's `source_path` is a real indexed document (reject hallucinated paths).
/// 2. Build `label_to_id` map: input nodes first (win), then existing graph nodes (`or_insert`).
/// 3. Build `Vec<NodeSpec>` using `id_for`.
/// 4. Resolve each `UpsertEdge` endpoint: section ref → resolved id; otherwise label → id via map.
/// 5. Build dump lines.
/// 6. If any errors: return rejection (nothing written), keeping the hint block.
/// 7. Otherwise call `apply_upsert` and return.
pub fn graph_upsert(
    idx: &DocIndex,
    g: &GraphStore,
    ont: &Ontology,
    nodes: Vec<UpsertNode>,
    edges: Vec<UpsertEdge>,
    now: u64,
) -> UpsertOutcome {
    // Partial apply: validate each item on its own, WRITE everything well-formed, and DROP only the
    // bad items with a clear, actionable reason — never discard valid work because a sibling item is
    // malformed (the old all-or-nothing footballed the model). Label length is NOT gated.
    let mut errs: Vec<String> = Vec::new();

    // (1) Nodes: keep those whose source_path is a real indexed document; drop the rest.
    struct SanitizedNode {
        node_type: String,
        label: String,
        aliases: Vec<String>,
    }
    let mut valid_nodes: Vec<(SanitizedNode, String)> = Vec::new();
    for nd in &nodes {
        match idx.canonical_document_path(&nd.source_path) {
            Some(canonical) => {
                let label = sanitize_label_for_upsert(ont, &nd.node_type, &nd.label);
                valid_nodes.push((
                    SanitizedNode {
                        node_type: nd.node_type.clone(),
                        label,
                        aliases: nd.aliases.clone(),
                    },
                    canonical,
                ));
            }
            None => errs.push(format!(
                "node \"{}\" dropped: source_path \"{}\" is not a document in the knowledge base — use a real path from a search/read result{}",
                nd.label, nd.source_path, crate::tools::document_path_hints(idx, &nd.source_path)
            )),
        }
    }

    // (2) label_to_id: ONLY the input (batch) nodes — they win over existing graph nodes.
    let mut label_to_id: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for (nd, _) in &valid_nodes {
        label_to_id.insert(
            normalize_label(&nd.label),
            id_for(ont, &nd.node_type, &nd.label),
        );
    }

    // (3) build NodeSpec list (valid nodes only)
    let nodespecs: Vec<NodeSpec> = valid_nodes
        .iter()
        .map(|(nd, canonical)| NodeSpec {
            id: id_for(ont, &nd.node_type, &nd.label),
            node_type: nd.node_type.clone(),
            label: nd.label.clone(),
            aliases: nd.aliases.clone(),
            source_path: canonical.clone(),
            range: None,
            confidence: None,
        })
        .collect();

    // batch ids — an edge may reference a node from this same call before it is committed
    let batch_ids: std::collections::HashSet<String> =
        nodespecs.iter().map(|n| n.id.clone()).collect();

    // (4) resolve edges
    let mut edgespecs: Vec<EdgeSpec> = Vec::new();
    for ue in &edges {
        let (of, ot, oet) = (ue.from.clone(), ue.to.clone(), ue.edge_type.clone());

        // missing endpoint — the most common malformed edge (e.g. MENTIONS with no `to`).
        if ue.from.trim().is_empty() || ue.to.trim().is_empty() {
            let which = if ue.from.trim().is_empty() { "from" } else { "to" };
            errs.push(format!(
                "edge -{oet}-> dropped: missing `{which}` — an edge needs BOTH a `from` and a `to` (a node label, or a section `<path>#<n>` for a MENTIONS target)"
            ));
            continue;
        }
        let edge_source_path = match idx.canonical_document_path(&ue.source_path) {
            Some(canonical) => canonical,
            None => {
                errs.push(format!(
                    "edge {of} -{oet}-> {ot} dropped: source_path \"{}\" is not a document — use a real path from a search/read result{}",
                    ue.source_path,
                    crate::tools::document_path_hints(idx, &ue.source_path)
                ));
                continue;
            }
        };

        let mut from_resolved: Option<String> = None;
        let mut to_resolved: Option<String> = None;
        let mut edge_ok = true;

        // resolve from endpoint
        match resolve_section_ref(idx, &ue.from) {
            Err(m) => {
                errs.push(format!("edge {of} -{oet}-> {ot} dropped: {m}"));
                edge_ok = false;
            }
            Ok(v) if v != ue.from => {
                // numeric section ref resolved to section id
                from_resolved = Some(v);
            }
            Ok(_) => {
                // treat as node label (exact, then fuzzy morphology fallback)
                match resolve_endpoint_label(g, &label_to_id, &ue.from) {
                    Some(id) => from_resolved = Some(id),
                    None => {
                        errs.push(format!(
                            "edge {of} -{oet}-> {ot} dropped: `from` label \"{of}\" matches no node — add a node with that label"
                        ));
                        edge_ok = false;
                    }
                }
            }
        }

        // resolve to endpoint
        match resolve_section_ref(idx, &ue.to) {
            Err(m) => {
                errs.push(format!("edge {of} -{oet}-> {ot} dropped: {m}"));
                edge_ok = false;
            }
            Ok(v) if v != ue.to => {
                // numeric section ref resolved to section id
                to_resolved = Some(v);
            }
            Ok(_) => {
                // treat as node label (exact, then fuzzy morphology fallback)
                match resolve_endpoint_label(g, &label_to_id, &ue.to) {
                    Some(id) => to_resolved = Some(id),
                    None => {
                        errs.push(format!(
                            "edge {of} -{oet}-> {ot} dropped: `to` label \"{ot}\" matches no node — add a node with that label"
                        ));
                        edge_ok = false;
                    }
                }
            }
        }

        if edge_ok {
            let from_id = from_resolved.unwrap();
            let to_id = to_resolved.unwrap();

            // post-resolution existence check
            let mut exists_ok = true;
            for (role, id) in [("from", from_id.clone()), ("to", to_id.clone())] {
                let exists =
                    batch_ids.contains(&id) || g.get_node(&id).ok().flatten().is_some();
                if !exists {
                    errs.push(format!(
                        "edge {of} -{oet}-> {ot} dropped: {role} endpoint '{id}' is not a known node — add it to nodes[] before referencing it"
                    ));
                    exists_ok = false;
                }
            }

            if exists_ok {
                edgespecs.push(EdgeSpec {
                    from: from_id,
                    to: to_id,
                    edge_type: ue.edge_type.clone(),
                    source_path: edge_source_path,
                    range: None,
                    confidence: None,
                });
            }
        }
    }

    // (5) build dump lines (resolved state, for the caller to log)
    let mut dump = Vec::new();
    for nd in &nodespecs {
        dump.push(format!("node {} [{}] {}", nd.id, nd.node_type, nd.label));
    }
    for e in &edgespecs {
        dump.push(format!("edge {} -{}-> {}", e.from, e.edge_type, e.to));
    }

    // (6) Nothing well-formed at all → full rejection with a label-matching hint.
    if nodespecs.is_empty() && edgespecs.is_empty() {
        const STRUCTURAL: &[&str] = &["Document", "Section", "Term", "Topic"];
        let existing: Vec<String> = g
            .all_nodes()
            .unwrap_or_default()
            .into_iter()
            .filter(|n| !STRUCTURAL.contains(&n.node_type.as_str()))
            .map(|n| format!("{} [{}] {}", n.id, n.node_type, n.label))
            .take(25)
            .collect();
        let mut hint_dump = Vec::new();
        if !existing.is_empty() {
            hint_dump.push(
                "Similar existing nodes (match by label in edges, do not put ids in label):".into(),
            );
            for line in existing {
                hint_dump.push(format!("- {line}"));
            }
        }
        let out = UpsertOutcome {
            message: String::new(),
            nodes: 0,
            edges: 0,
            rejected: true,
            dump: hint_dump,
            merged: vec![],
            dropped: errs,
        };
        return UpsertOutcome {
            message: format_upsert_response(&out),
            ..out
        };
    }

    // (7) Apply the well-formed items; report any dropped ones so the model resends JUST those.
    match apply_upsert(g, ont, nodespecs, edgespecs, now) {
        Ok(result) => {
            let out = UpsertOutcome {
                message: String::new(),
                nodes: result.nodes_written,
                edges: result.edges_written,
                rejected: false,
                dump,
                merged: result.merged,
                dropped: errs,
            };
            UpsertOutcome {
                message: format_upsert_response(&out),
                ..out
            }
        }
        Err(e) => {
            let out = UpsertOutcome {
                message: String::new(),
                nodes: 0,
                edges: 0,
                rejected: true,
                dump,
                merged: vec![],
                dropped: vec![format!("graph_upsert failed: {e}")],
            };
            UpsertOutcome {
                message: format!("graph_upsert failed: {e}"),
                ..out
            }
        }
    }
}

/// Delete reasoning nodes and/or edges by label.
/// Edge endpoints that look like `<path>#<n>` section refs are resolved to their canonical
/// Section node id (symmetry with `graph_upsert`). On resolution error the original string
/// is kept so a bad anchor simply doesn't match — deletion is best-effort.
/// Returns a human-readable result string.
pub fn graph_delete(idx: &DocIndex, g: &GraphStore, node_labels: Vec<String>, edges: Vec<EdgeRef>) -> String {
    let edges: Vec<EdgeRef> = edges.into_iter().map(|e| {
        let from_orig = e.from;
        let to_orig = e.to;
        let from = resolve_section_ref(idx, &from_orig).unwrap_or(from_orig);
        let to = resolve_section_ref(idx, &to_orig).unwrap_or(to_orig);
        EdgeRef { from, edge_type: e.edge_type, to }
    }).collect();
    match apply_delete(g, node_labels, edges) {
        Ok((n, notes)) => {
            let mut m = format!("deleted {n} graph entries");
            if !notes.is_empty() {
                m.push_str(&format!(
                    "\n{} reference(s) matched nothing (nothing deleted for these):\n- {}",
                    notes.len(),
                    notes.join("\n- ")
                ));
            }
            m
        }
        Err(e) => format!("graph_delete error: {e}"),
    }
}

/// Edit existing reasoning nodes in place — rename label and/or change type — by current label.
/// The node id and all its edges are preserved. Returns a human-readable result string.
pub fn graph_update(g: &GraphStore, nodes: Vec<NodeUpdate>) -> String {
    if nodes.is_empty() {
        return "updated 0 nodes — no update received. graph_update takes \
                {\"nodes\":[{\"label\":\"…\",\"new_label\":\"…\",\"new_type\":\"…\"}]} \
                (a single flat {\"label\":\"…\",\"new_label\":\"…\"} is also accepted). \
                Pass the node's current label and what to change."
            .to_string();
    }
    match apply_update(g, nodes) {
        Ok((n, notes)) => {
            let mut m = format!("updated {n} nodes");
            if !notes.is_empty() {
                m.push_str(&format!("\n{} skipped:\n- {}", notes.len(), notes.join("\n- ")));
            }
            m
        }
        Err(e) => format!("graph_update error: {e}"),
    }
}

/// Recompute the graph's DERIVED layer (transitive-closure edges, SIMILAR links, communities,
/// centrality) non-destructively and report the counts. Shared by the MCP `graph_generalize`
/// tool and the eval enricher so both emit identical output. `prune_incomplete`/`apply_merges`
/// stay off (from_ontology defaults), so it never deletes or merges — pruning is a CLI action.
pub fn graph_generalize(g: &GraphStore, ont: &Ontology, now: u64) -> String {
    let opts = crate::graph::generalize::apply::Opts::from_ontology(ont, now);
    match crate::graph::generalize::apply::generalize(g, &opts) {
        Ok(r) => format!(
            "generalized: prune_candidates={} inferred_edges={} similar_edges={} \
             communities={} merge_candidates={}",
            r.prune_candidates, r.inferred_edges, r.similar_edges, r.communities, r.merge_candidates
        ),
        Err(e) => format!("graph_generalize error: {e}"),
    }
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::agent::NodeUpdate;

    const DEDUP_ONT: &str = r#"
[entities.Symptom]
id_prefix = "sym"
props = []
[entities.Resolution]
id_prefix = "res"
props = []
[relations.RESOLVED_BY]
from = ["Symptom"]
to = ["Resolution"]
[validation]
strict = true
"#;

    fn unode(node_type: &str, label: &str, src: &str) -> UpsertNode {
        UpsertNode {
            node_type: node_type.into(),
            label: label.into(),
            source_path: src.into(),
            aliases: vec![],
        }
    }

    fn uedge(from: &str, edge_type: &str, to: &str, src: &str) -> UpsertEdge {
        UpsertEdge {
            from: from.into(),
            edge_type: edge_type.into(),
            to: to.into(),
            source_path: src.into(),
        }
    }

    /// Index a stub document so `has_document` accepts `path` as a valid source_path.
    fn write_doc(idx: &DocIndex, path: &str) {
        idx.write_chunks(&[crate::model::Chunk {
            doc_path: path.into(),
            location: "S1".into(),
            file_type: "md".into(),
            text: "stub content".into(),
        }])
        .unwrap();
    }

    /// Happy path: Symptom + Resolution + RESOLVED_BY in one batch — accepted and written.
    #[test]
    fn happy_path_symptom_resolution_resolved_by() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "case1.docx");

        let nodes = vec![
            unode("Symptom", "Потеря связи", "case1.docx"),
            unode("Resolution", "Перезагрузка модуля", "case1.docx"),
        ];
        let edges = vec![uedge("Потеря связи", "RESOLVED_BY", "Перезагрузка модуля", "case1.docx")];

        let out = graph_upsert(&idx, &g, &ont, nodes, edges, 1_000_000);
        assert!(!out.rejected, "should not be rejected: {}", out.message);
        assert_eq!(out.nodes, 2);
        assert_eq!(out.edges, 1);
        assert!(format_upsert_response(&out).contains("upserted 2 nodes, 1 edges"), "{}", out.message);
        assert!(out.message.contains("Written:"), "{}", out.message);
    }

    /// graph_update renames a node in place while preserving its id and all outgoing edges.
    #[test]
    fn update_renames_node_keeps_id_and_edges() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "case1.docx");

        // Upsert: Symptom + Resolution + RESOLVED_BY edge.
        let nodes = vec![
            unode("Symptom", "Потеря связи", "case1.docx"),
            unode("Resolution", "Перезагрузка модуля", "case1.docx"),
        ];
        let edges =
            vec![uedge("Потеря связи", "RESOLVED_BY", "Перезагрузка модуля", "case1.docx")];
        graph_upsert(&idx, &g, &ont, nodes, edges, 1);

        // Rename the Symptom node.
        let ups = vec![NodeUpdate {
            label: "Потеря связи".into(),
            new_label: Some("Нестабильная связь".into()),
            new_type: None,
        }];
        let result = graph_update(&g, ups);
        assert_eq!(result, "updated 1 nodes", "unexpected result: {result}");

        // (a) a node with the new label exists
        let id = g.find_by_label("Нестабильная связь").unwrap();
        assert!(id.is_some(), "new label not found");
        let id = id.unwrap();

        // (b) the id is the label-derived id for the original label
        let expected_id = id_for(&ont, "Symptom", "Потеря связи");
        assert_eq!(id, expected_id, "id should be the label-derived id");

        // (c) the RESOLVED_BY edge still exists
        let res_id = id_for(&ont, "Resolution", "Перезагрузка модуля");
        let out = g.outgoing(&id).unwrap();
        assert!(
            out.iter().any(|e| e.edge_type == "RESOLVED_BY" && e.to == res_id),
            "RESOLVED_BY edge lost after rename"
        );
    }

    /// Rejection: an edge whose `to` endpoint label is neither in the batch nor in the graph.
    #[test]
    fn rejects_edge_to_unknown_node() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "case1.docx");

        // Only the Symptom node — Resolution label is missing from batch and graph.
        let nodes = vec![unode("Symptom", "Потеря связи", "case1.docx")];
        let edges =
            vec![uedge("Потеря связи", "RESOLVED_BY", "Перезагрузка модуля", "case1.docx")];

        let out = graph_upsert(&idx, &g, &ont, nodes, edges, 1_000_000);
        // Partial apply: the valid Symptom IS written; only the edge to an unknown node is dropped.
        assert!(!out.rejected, "valid node written: {}", out.message);
        assert_eq!(out.nodes, 1, "the Symptom node is written");
        assert_eq!(out.edges, 0, "the edge to an unknown node is dropped");
        assert!(
            out.message.contains("dropped") && out.message.contains("matches no node"),
            "message should explain the dropped edge: {}",
            out.message
        );
    }

    /// Label-based upsert with no ids: ids are derived from labels, RESOLVED_BY edge is wired.
    #[test]
    fn label_based_upsert_no_ids() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "case1.docx");

        let nodes = vec![
            unode("Symptom", "Потеря связи", "case1.docx"),
            unode("Resolution", "Перезапуск", "case1.docx"),
        ];
        let edges = vec![uedge("Потеря связи", "RESOLVED_BY", "Перезапуск", "case1.docx")];

        let out = graph_upsert(&idx, &g, &ont, nodes, edges, 1_000_000);
        assert!(!out.rejected, "should not be rejected: {}", out.message);
        assert_eq!(out.nodes, 2);
        assert_eq!(out.edges, 1);

        // Assert the RESOLVED_BY edge connects the label-derived ids.
        let sym_id = id_for(&ont, "Symptom", "Потеря связи");
        let res_id = id_for(&ont, "Resolution", "Перезапуск");
        let outgoing = g.outgoing(&sym_id).unwrap();
        assert!(
            outgoing.iter().any(|e| e.edge_type == "RESOLVED_BY" && e.to == res_id),
            "RESOLVED_BY edge not found from {sym_id} to {res_id}: {outgoing:?}"
        );
    }

    /// Edge whose `to` label has no corresponding node is rejected; message names the label.
    #[test]
    fn edge_to_undefined_label_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "case1.docx");

        let nodes = vec![unode("Symptom", "Потеря связи", "case1.docx")];
        let edges = vec![uedge("Потеря связи", "RESOLVED_BY", "Неизвестный узел", "case1.docx")];

        let out = graph_upsert(&idx, &g, &ont, nodes, edges, 1_000_000);
        // Partial apply: the Symptom IS written; the edge to an undefined label is dropped (named).
        assert!(!out.rejected, "valid node written: {}", out.message);
        assert_eq!(out.nodes, 1);
        assert_eq!(out.edges, 0);
        assert!(
            out.message.contains("Неизвестный узел") && out.message.contains("dropped"),
            "message names the dropped edge's bad label: {}",
            out.message
        );
    }

    /// Partial apply: a malformed edge (MENTIONS with no `to`) is dropped with a clear message,
    /// but the valid nodes in the SAME call are still written — never football the whole upsert.
    #[test]
    fn partial_apply_keeps_valid_nodes_when_edge_missing_to() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "case1.docx");

        let nodes = vec![
            unode("Symptom", "Потеря связи", "case1.docx"),
            unode("Resolution", "Перезапуск", "case1.docx"),
        ];
        // a MENTIONS edge with no `to` target — malformed.
        let edges = vec![uedge("Потеря связи", "MENTIONS", "", "case1.docx")];

        let out = graph_upsert(&idx, &g, &ont, nodes, edges, 1_000_000);
        assert!(!out.rejected, "valid nodes are written: {}", out.message);
        assert_eq!(out.nodes, 2, "both valid nodes written");
        assert_eq!(out.edges, 0, "the edge with no `to` is dropped");
        assert!(
            out.message.contains("missing `to`"),
            "message clearly names the problem: {}",
            out.message
        );
    }

    #[test]
    fn graph_update_empty_input_explains_the_shape() {
        // The model sometimes sends a shape the handler parses to zero updates (the FLAT-vs-nodes
        // mismatch). Instead of a silent "updated 0 nodes", say what graph_update expects.
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let msg = graph_update(&g, vec![]);
        assert!(msg.contains("updated 0 nodes"), "{msg}");
        assert!(msg.contains("no update received"), "must explain the empty case: {msg}");
        assert!(msg.contains("\"nodes\""), "must show the expected shape: {msg}");
    }

    /// Clear feedback: delete/update report references that matched nothing instead of a silent
    /// "deleted 0" / "updated 0" that left the model guessing.
    #[test]
    fn delete_and_update_report_unmatched_refs() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "case1.docx");
        let _ = graph_upsert(&idx, &g, &ont, vec![unode("Symptom", "Потеря связи", "case1.docx")], vec![], 1);

        let del = graph_delete(&idx, &g, vec!["Несуществующий".into()], vec![]);
        assert!(del.contains("matched nothing"), "delete names the unmatched ref: {del}");

        let upd = graph_update(
            &g,
            vec![NodeUpdate { label: "Несуществующий".into(), new_label: Some("X".into()), new_type: None }],
        );
        assert!(
            upd.contains("skipped") && upd.contains("matched nothing"),
            "update reports the skipped ref: {upd}"
        );
    }

    /// The model paraphrases its own label across calls — a truncated reference must resolve
    /// fuzzily (morphology) to the existing node instead of being rejected.
    #[test]
    fn edge_label_resolves_fuzzily_to_paraphrase() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "c.docx");

        let out1 = graph_upsert(
            &idx, &g, &ont,
            vec![
                unode("Symptom", "Потеря связи Profibus", "c.docx"),
                unode("Resolution", "Изменение параметра maxTsdr и перезапуск службы", "c.docx"),
            ],
            vec![uedge("Потеря связи Profibus", "RESOLVED_BY", "Изменение параметра maxTsdr и перезапуск службы", "c.docx")],
            1,
        );
        assert!(!out1.rejected, "{}", out1.message);

        // Later edge references the Resolution by a TRUNCATED label.
        let out2 = graph_upsert(
            &idx, &g, &ont,
            vec![],
            vec![uedge("Потеря связи Profibus", "RESOLVED_BY", "Изменение параметра maxTsdr", "c.docx")],
            2,
        );
        assert!(!out2.rejected, "truncated label must resolve fuzzily: {}", out2.message);
    }

    /// Fix 1 — node with a source_path not in the index is rejected; a real path is accepted.
    #[test]
    fn rejects_node_with_fake_source_path() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();

        // Index only one real document.
        write_doc(&idx, "kb-test\\Доп.данные\\real.pdf");

        // Node with a hallucinated path — must be rejected.
        let out = graph_upsert(
            &idx, &g, &ont,
            vec![unode("Symptom", "Потеря связи", "case_support_001")],
            vec![],
            1,
        );
        assert!(out.rejected, "hallucinated source_path must be rejected: {}", out.message);
        assert!(
            out.message.contains("is not a document"),
            "message should say 'is not a document': {}",
            out.message
        );

        // Same node but with the real indexed path — must be accepted.
        let out_real = graph_upsert(
            &idx, &g, &ont,
            vec![unode("Symptom", "Потеря связи", "kb-test\\Доп.данные\\real.pdf")],
            vec![],
            2,
        );
        assert!(!out_real.rejected, "real source_path must be accepted: {}", out_real.message);
    }

    #[test]
    fn upsert_accepts_prefixed_source_path() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "kb-test\\Доп.данные\\real.pdf");

        let out = graph_upsert(
            &idx, &g, &ont,
            vec![unode("Symptom", "Потеря связи", "kb-manual\\kb-test\\Доп.данные\\real.pdf")],
            vec![],
            1,
        );
        assert!(!out.rejected, "prefixed source_path must resolve: {}", out.message);
        let sym_id = id_for(&ont, "Symptom", "Потеря связи");
        let node = g.get_node(&sym_id).unwrap().unwrap();
        assert_eq!(node.prov.source_path, "kb-test\\Доп.данные\\real.pdf");
    }

    #[test]
    fn upsert_rejects_bad_path_with_did_you_mean() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "kb-test\\Доп.данные\\real.pdf");

        let out = graph_upsert(
            &idx, &g, &ont,
            vec![unode("Symptom", "Потеря связи", "wrong\\real.pdf")],
            vec![],
            1,
        );
        assert!(out.rejected, "bad path must reject: {}", out.message);
        assert!(out.message.contains("did you mean"), "upsert hint: {}", out.message);
        assert!(out.message.contains("real.pdf"), "upsert suggests real path: {}", out.message);
    }

    #[test]
    fn resolve_section_ref_prefixed_path() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[crate::model::Chunk {
            doc_path: "docs\\real.md".into(),
            location: "Introduction".into(),
            file_type: "md".into(),
            text: "section content".into(),
        }]).unwrap();
        let resolved = resolve_section_ref(&idx, "kb-manual\\docs\\real.md#1").unwrap();
        assert_eq!(resolved, "docs\\real.md#Introduction");
    }

    /// Fix 2 — graph_delete resolves `<path>#<n>` section refs (symmetry with upsert).
    #[test]
    fn graph_delete_resolves_section_refs() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();

        // Index a real document with a known location so we can resolve chunk refs.
        idx.write_chunks(&[crate::model::Chunk {
            doc_path: "real.md".into(),
            location: "Introduction".into(),
            file_type: "md".into(),
            text: "section content".into(),
        }])
        .unwrap();

        // Upsert two reasoning nodes + RESOLVED_BY edge.
        let out = graph_upsert(
            &idx, &g, &ont,
            vec![
                unode("Symptom", "Test Symptom", "real.md"),
                unode("Resolution", "Test Fix", "real.md"),
            ],
            vec![uedge("Test Symptom", "RESOLVED_BY", "Test Fix", "real.md")],
            1,
        );
        assert!(!out.rejected, "{}", out.message);

        // Delete by label-based endpoints — basic case.
        let msg = graph_delete(
            &idx, &g,
            vec![],
            vec![EdgeRef {
                from: "Test Symptom".into(),
                edge_type: "RESOLVED_BY".into(),
                to: "Test Fix".into(),
            }],
        );
        assert!(msg.contains("deleted"), "basic label delete: {msg}");

        let sym_id = id_for(&ont, "Symptom", "Test Symptom");
        let outgoing = g.outgoing(&sym_id).unwrap();
        assert!(
            !outgoing.iter().any(|e| e.edge_type == "RESOLVED_BY"),
            "RESOLVED_BY edge should be deleted: {outgoing:?}"
        );

        // Section ref resolution: `real.md#1` resolves to `real.md#Introduction`.
        // There is no edge with that endpoint, so 0 entries deleted — but must NOT panic/error.
        let msg2 = graph_delete(
            &idx, &g,
            vec![],
            vec![EdgeRef {
                from: "Test Symptom".into(),
                edge_type: "RESOLVED_BY".into(),
                to: "real.md#1".into(),
            }],
        );
        assert!(!msg2.contains("error"), "section ref resolution must not produce error: {msg2}");

        // Non-existent chunk ref: resolve_section_ref errors, original kept, no panic.
        let msg3 = graph_delete(
            &idx, &g,
            vec![],
            vec![EdgeRef {
                from: "Test Symptom".into(),
                edge_type: "RESOLVED_BY".into(),
                to: "real.md#999".into(),
            }],
        );
        assert!(!msg3.contains("error"), "non-existent chunk ref must not produce error: {msg3}");
    }

    /// Agent copied sym: into label — sanitize prevents sym:sym: double prefix.
    #[test]
    fn sanitize_strips_type_prefix_from_label() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "case1.docx");

        let out = graph_upsert(
            &idx,
            &g,
            &ont,
            vec![unode("Symptom", "sym:Горячая замена CPU", "case1.docx")],
            vec![],
            1,
        );
        assert!(!out.rejected, "{}", out.message);
        let expected = id_for(&ont, "Symptom", "Горячая замена CPU");
        assert_eq!(expected, "sym:горячая-замена-cpu");
        assert!(g.get_node(&expected).unwrap().is_some());
        assert!(out.message.contains("Written:"), "{}", out.message);
        assert!(out.message.contains(&expected), "{}", out.message);
    }

    /// Edge endpoint may reference an existing node by id.
    #[test]
    fn edge_resolves_existing_node_by_id() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "case1.docx");

        graph_upsert(
            &idx,
            &g,
            &ont,
            vec![unode("Symptom", "Потеря связи", "case1.docx")],
            vec![],
            1,
        );
        let sym_id = id_for(&ont, "Symptom", "Потеря связи");
        let out = graph_upsert(
            &idx,
            &g,
            &ont,
            vec![unode("Resolution", "Перезапуск", "case1.docx")],
            vec![uedge(&sym_id, "RESOLVED_BY", "Перезапуск", "case1.docx")],
            2,
        );
        assert!(!out.rejected, "{}", out.message);
        assert_eq!(out.edges, 1);
    }

    /// Dedup reports merged ids when an existing node shares the label but has a different id.
    #[test]
    fn dedup_reports_merged_in_response() {
        use crate::graph::agent::{apply_upsert, NodeSpec};

        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "case1.docx");

        apply_upsert(
            &g,
            &ont,
            vec![NodeSpec {
                id: "sym:legacy-id".into(),
                node_type: "Symptom".into(),
                label: "Потеря связи".into(),
                aliases: vec![],
                source_path: "case1.docx".into(),
                range: None,
                confidence: None,
            }],
            vec![],
            1,
        )
        .unwrap();

        let out = graph_upsert(
            &idx,
            &g,
            &ont,
            vec![unode("Symptom", "Потеря связи", "case1.docx")],
            vec![],
            2,
        );
        assert!(!out.rejected, "{}", out.message);
        assert_eq!(out.nodes, 0, "deduped node not created");
        assert!(out.message.contains("Merged"), "{}", out.message);
        assert!(out.message.contains("sym:legacy-id"), "{}", out.message);
    }

    #[test]
    fn full_rejection_uses_rejected_header() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();

        let out = graph_upsert(
            &idx,
            &g,
            &ont,
            vec![unode("Symptom", "X", "nonexistent.docx")],
            vec![],
            1,
        );
        assert!(out.rejected, "{}", out.message);
        assert!(out.message.contains("REJECTED — nothing written"), "{}", out.message);
        assert!(out.message.contains("is not a document"), "{}", out.message);
    }

    #[test]
    fn full_rejection_hint_suggests_label_not_id_in_label_field() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "case1.docx");
        graph_upsert(
            &idx,
            &g,
            &ont,
            vec![unode("Symptom", "Existing symptom", "case1.docx")],
            vec![],
            1,
        );

        let out = graph_upsert(
            &idx,
            &g,
            &ont,
            vec![unode("Symptom", "Y", "bad.docx")],
            vec![],
            2,
        );
        assert!(out.rejected, "{}", out.message);
        assert!(
            out.message.contains("match by label in edges"),
            "hint wording: {}",
            out.message
        );
    }
}
