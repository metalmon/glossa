//! Shared write operations for the reasoning graph.
//!
//! Both the MCP server (`src/mcp.rs`) and the kb-eval enricher (`eval/src/enrich.rs`)
//! funnel through these functions so validation and resolution behaviour is identical.

use crate::graph::agent::{
    apply_delete, apply_update, apply_upsert, EdgeRef, EdgeSpec, NodeSpec, NodeUpdate,
};
use crate::graph::ontology::Ontology;
use crate::graph::store::{normalize_label, GraphStore, Node, NodeValidity};
use crate::graph::temporal;
use crate::index::store::DocIndex;
use serde_json::Value;

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
    /// Start of this node's authored validity interval, any ISO-8601 granularity
    /// (`"2020"`, `"2020-06"`, `"2020-06-15"`, or a full RFC3339 UTC instant).
    /// `None` when the caller doesn't touch validity (leaves any existing interval alone).
    #[serde(default, deserialize_with = "crate::json_util::deserialize_opt_string_loose")]
    pub valid_from: Option<String>,
    /// End of the authored validity interval, same granularity rules as `valid_from`.
    #[serde(default, deserialize_with = "crate::json_util::deserialize_opt_string_loose")]
    pub valid_to: Option<String>,
}

/// A directed edge identified by node labels (or section refs) at both endpoints.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct UpsertEdge {
    pub from: String,
    pub edge_type: String,
    pub to: String,
    pub source_path: String,
}

/// JSON item is an edge misplaced in `nodes[]` (has from/edge_type/to, no node_type).
fn upsert_item_looks_like_edge(v: &Value) -> bool {
    v.is_object()
        && v.get("node_type").is_none()
        && v.get("edge_type").and_then(|x| x.as_str()).is_some()
        && v.get("from").and_then(|x| x.as_str()).is_some()
        && v.get("to").and_then(|x| x.as_str()).is_some()
}

/// Parse a `graph_upsert` tool payload tolerantly: edge-shaped objects in `nodes[]`
/// are moved to `edges[]` (and node-shaped objects in `edges[]` to `nodes[]`).
pub fn parse_upsert_payload(v: &Value) -> (Vec<UpsertNode>, Vec<UpsertEdge>, Vec<String>) {
    let mut notes = Vec::new();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    if let Some(arr) = v.get("nodes").and_then(|a| a.as_array()) {
        for (i, item) in arr.iter().enumerate() {
            if upsert_item_looks_like_edge(item) {
                match serde_json::from_value::<UpsertEdge>(item.clone()) {
                    Ok(e) => {
                        notes.push(format!(
                            "nodes[{i}] is an edge (from/edge_type/to) — moved to edges[]; edges belong in edges[] only"
                        ));
                        edges.push(e);
                    }
                    Err(e) => {
                        notes.push(format!("nodes[{i}] dropped (edge-shaped but invalid): {e}"))
                    }
                }
            } else {
                match serde_json::from_value::<UpsertNode>(item.clone()) {
                    Ok(n) => nodes.push(n),
                    Err(e) => notes.push(format!("nodes[{i}] dropped: {e}")),
                }
            }
        }
    }

    if let Some(arr) = v.get("edges").and_then(|a| a.as_array()) {
        for (i, item) in arr.iter().enumerate() {
            if item.get("node_type").is_some() && !upsert_item_looks_like_edge(item) {
                match serde_json::from_value::<UpsertNode>(item.clone()) {
                    Ok(n) => {
                        notes.push(format!("edges[{i}] is a node — moved to nodes[]"));
                        nodes.push(n);
                    }
                    Err(e) => notes.push(format!("edges[{i}] dropped: {e}")),
                }
            } else {
                match serde_json::from_value::<UpsertEdge>(item.clone()) {
                    Ok(e) => edges.push(e),
                    Err(e) => notes.push(format!("edges[{i}] dropped: {e}")),
                }
            }
        }
    }

    if nodes.is_empty() && edges.is_empty() {
        if upsert_item_looks_like_edge(v) {
            if let Ok(e) = serde_json::from_value(v.clone()) {
                edges.push(e);
            }
        } else if let Ok(n) = serde_json::from_value(v.clone()) {
            nodes.push(n);
        }
    }

    (nodes, edges, notes)
}

/// FNV-1a (32-bit) as 8 lowercase hex — a fixed, build-stable hash (unlike std hashers) used to
/// keep short ids distinct.
fn fnv1a_hex(s: &str) -> String {
    let mut h: u32 = 0x811c_9dc5;
    for b in s.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    format!("{h:08x}")
}

/// A short, copyable, build-stable node id: `<abbrev>:<hash>` — a fixed hash of the normalized
/// label instead of the whole label as a dash-slug. Deterministic in the normalized label (dedup
/// relies on this: same label → same id); distinct labels get distinct hashes (the rare collision
/// is caught by [`disambiguate_id`] at write time). The human-readable text lives in the node's
/// `label`, shown next to the id everywhere it appears, so the id itself need not be readable.
pub fn id_for(ont: &Ontology, node_type: &str, label: &str) -> String {
    format!(
        "{}:{}",
        ont.id_abbrev(node_type),
        fnv1a_hex(&normalize_label(label))
    )
}

/// Resolve an id collision: if `base` is already taken (by a node with a DIFFERENT label), append
/// `-2`, `-3`, … until free. Guards against the rare hash collision silently overwriting a
/// distinct fact.
fn disambiguate_id(base: &str, taken: impl Fn(&str) -> bool) -> String {
    if !taken(base) {
        return base.to_string();
    }
    let mut n = 2u32;
    loop {
        let cand = format!("{base}-{n}");
        if !taken(&cand) {
            return cand;
        }
        n += 1;
    }
}

/// Strip a mistaken type-prefix when the agent copied a graph id into the `label` field.
pub fn sanitize_label_for_upsert(ont: &Ontology, node_type: &str, label: &str) -> String {
    let prefix = format!("{}:", ont.id_abbrev(node_type));
    let trimmed = label.trim();
    // `get` (not byte slicing): prefix.len() may fall inside a multi-byte char of a
    // non-ASCII label, and a slice there panics.
    match trimmed.get(..prefix.len()) {
        Some(head) if trimmed.len() > prefix.len() && head.eq_ignore_ascii_case(&prefix) => {
            trimmed[prefix.len()..].trim_start().to_string()
        }
        _ => trimmed.to_string(),
    }
}

// ── label helpers ─────────────────────────────────────────────────────────────

/// Resolve a section reference of the form `<path>#<n>` (a chunk number the agent has from a
/// read/search result) into the real Section node id `<path>#<location>`. Reasoning-node slugs
/// (`sym:…`) and already-resolved `<path>#<location>` refs (non-numeric suffix) pass through.
/// A numeric chunk ord that does NOT exist in the document is REJECTED with an actionable message
/// — so the model re-anchors to a real chunk instead of writing a dangling edge.
/// Returns `Ok(Some(id))` when `s` was a numeric section ref that resolved to a real
/// Section node id, `Ok(None)` when `s` is not a section ref (a label / reasoning slug
/// to resolve elsewhere), and `Err` when it was a section ref to a non-existent chunk.
/// This is a distinct signal — do NOT infer "was a section ref" from the string
/// changing, since a resolved id can equal the input (e.g. `<path>#1` for a heading-less
/// chunk whose location is its ordinal).
fn resolve_section_ref(idx: &DocIndex, s: &str) -> Result<Option<String>, String> {
    if let Some(pos) = s.rfind('#') {
        let suffix = &s[pos + 1..];
        if let Ok(n) = suffix.parse::<u64>() {
            let raw = &s[..pos];
            // A path that resolves to no indexed document is the real failure here —
            // say so plainly instead of blaming the chunk number, which sends the model
            // chasing a different number when it should fix the document path (e.g. it
            // wrote the name transliterated into another script instead of copying it).
            let Some(path) = idx.canonical_document_path(raw) else {
                return Err(format!(
                    "no document '{raw}' in the base — copy the document path exactly as grep/read prints it, character for character"
                ));
            };
            // Chunk numbering is 1-based (grep/read/search all show `#1` for the first
            // chunk). The model often reaches for 0-based indexing out of programming
            // habit and writes `#0`; treat that as the first chunk instead of failing, so
            // it doesn't loop guessing numbers. Higher out-of-range numbers still error —
            // those are a real mistake (a number copied from another file) — but the
            // message now states the 1-based convention.
            let n = if n == 0 { 1 } else { n };
            return match idx.location_for_ord(&path, n) {
                // Keep the chunk ORDINAL the agent gave (`path#3`) as the reference, not
                // the chunk's heading location — the agent works in chunk numbers (from
                // grep/read), so a heading-based id never matches what it wrote and reads
                // back as a "mangled" ref. We still resolve to confirm the chunk exists.
                Ok(Some(_)) => Ok(Some(crate::graph::build::section_id(&path, &n.to_string()))),
                Ok(None) => Err(format!(
                    "chunk #{n} does not exist in {path}; chunk numbers start at #1 — take the number from a search/grep/read on THIS document, never reuse one from another file"
                )),
                Err(e) => Err(format!("could not resolve chunk #{n} in {path}: {e}")),
            };
        }
    }
    Ok(None)
}

/// Shared resolution context for `EndpointResolver::resolve` — the index, graph, ontology,
/// and the batch-local lookup maps that every endpoint resolution needs, grouped so the
/// resolver takes two arguments (`label`, `prefer_types`) instead of eight.
struct EndpointResolver<'a> {
    idx: &'a DocIndex,
    g: &'a GraphStore,
    ont: &'a Ontology,
    label_to_id: &'a std::collections::HashMap<String, String>,
    label_type_to_id: &'a std::collections::HashMap<(String, String), String>,
    batch_ids: &'a std::collections::HashSet<String>,
}

impl EndpointResolver<'_> {
    /// Resolve an edge endpoint label to a node id: exact normalized-label match first, then a
    /// fuzzy morphology match against existing reasoning nodes (the small model often paraphrases
    /// its own label — a truncation or wording variant). As a last resort, strip a leading
    /// node-TYPE token the model prefixed onto its own label ("Standard SPEC-1234" → "SPEC-1234")
    /// — gated on the token naming a declared type, so ordinary multi-word labels are never
    /// mangled. Returns None when nothing matches.
    fn resolve(&self, label: &str, prefer_types: &[String]) -> Option<String> {
        let Self {
            idx,
            g,
            ont,
            label_to_id,
            label_type_to_id,
            batch_ids,
        } = self;
        if batch_ids.contains(label) {
            return Some(label.to_string());
        }
        if g.get_node(label).ok().flatten().is_some() {
            return Some(label.to_string());
        }
        // A Document endpoint (e.g. DEPENDS_ON doc->doc) arrives with the agent's path
        // separators ("docs/foo.docx"), but the stored Document node id uses the host
        // separator ("docs\foo.docx"). Resolve it through the SAME path normalization as
        // section refs (canonical_document_path) so it matches without the caller hand-
        // writing backslashes. A non-path label (e.g. a Reference "Some External Reference")
        // resolves to no indexed document and falls through to label matching below.
        if let Some(p) = idx.canonical_document_path(label) {
            if g.get_node(&p).ok().flatten().is_some() {
                return Some(p);
            }
        }
        // When the relation fixes this endpoint's type (CONSTRAINED_BY `from` is a Field, `to`
        // an Enum) and a Field and its Enum share this label, pick the existing node of the
        // wanted type so the two do not collide onto one node (the Enum->Enum rejection).
        if !prefer_types.is_empty() {
            let norm = normalize_label(label);
            for t in prefer_types {
                if let Some(id) = label_type_to_id.get(&(norm.clone(), t.clone())) {
                    return Some(id.clone());
                }
            }
            if let Ok(ids) = g.ids_by_label_norm(label) {
                for id in ids {
                    if let Ok(Some(n)) = g.get_node(&id) {
                        if prefer_types.iter().any(|t| t == &n.node_type) {
                            return Some(id);
                        }
                    }
                }
            }
        }
        if let Some(id) = label_to_id.get(&normalize_label(label)) {
            return Some(id.clone());
        }
        if let Some((first, rest)) = label.split_once(char::is_whitespace) {
            let rest = rest.trim();
            let is_type = !rest.is_empty()
                && (ont
                    .entity_types()
                    .iter()
                    .any(|t| t.eq_ignore_ascii_case(first))
                    || ["Document", "Section", "Term", "Topic"]
                        .iter()
                        .any(|t| t.eq_ignore_ascii_case(first)));
            if is_type {
                if let Some(id) = label_to_id.get(&normalize_label(rest)) {
                    return Some(id.clone());
                }
                if let Some(id) = g.ids_by_label_norm(rest).ok()?.into_iter().next() {
                    return Some(id);
                }
            }
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
        if out.dropped.iter().any(|e| e.contains("not allowed")) {
            parts.push(
                "Hint: call get_ontology — relations.<edge_type> lists allowed from/to node types."
                    .into(),
            );
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
    // A node also carrying valid_from/valid_to is normalized here — an unparseable date is a
    // client error, so (mirroring the hallucinated-source_path case) that ONE node is dropped
    // with an actionable message rather than failing the whole batch.
    struct SanitizedNode {
        node_type: String,
        label: String,
        aliases: Vec<String>,
        valid_from_raw: Option<String>,
        valid_from_norm: Option<String>,
        valid_to_raw: Option<String>,
        valid_to_norm: Option<String>,
    }
    let mut valid_nodes: Vec<(SanitizedNode, String)> = Vec::new();
    for nd in &nodes {
        match idx.canonical_document_path(&nd.source_path) {
            Some(canonical) => {
                let label = sanitize_label_for_upsert(ont, &nd.node_type, &nd.label);
                let valid_from_norm = match nd.valid_from.as_deref().map(temporal::normalize_from).transpose() {
                    Ok(v) => v,
                    Err(e) => {
                        errs.push(format!(
                            "node \"{}\" dropped: invalid valid_from \"{}\": {e}",
                            nd.label,
                            nd.valid_from.as_deref().unwrap_or_default()
                        ));
                        continue;
                    }
                };
                let valid_to_norm = match nd.valid_to.as_deref().map(temporal::normalize_to).transpose() {
                    Ok(v) => v,
                    Err(e) => {
                        errs.push(format!(
                            "node \"{}\" dropped: invalid valid_to \"{}\": {e}",
                            nd.label,
                            nd.valid_to.as_deref().unwrap_or_default()
                        ));
                        continue;
                    }
                };
                valid_nodes.push((
                    SanitizedNode {
                        node_type: nd.node_type.clone(),
                        label,
                        aliases: nd.aliases.clone(),
                        valid_from_raw: nd.valid_from.clone(),
                        valid_from_norm,
                        valid_to_raw: nd.valid_to.clone(),
                        valid_to_norm,
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

    // (2) Assign one canonical id per distinct (normalized) label, with a collision guard: the
    // short id embeds a hash of the full label, so distinct facts normally get distinct ids, but
    // on the rare hash collision `disambiguate_id` appends `-2/-3` rather than letting one fact
    // silently overwrite another. `norm_to_id` is the single source of truth; every later step
    // (nodespec, validity, edge endpoints) reads its id from here instead of recomputing.
    let mut norm_to_id: std::collections::HashMap<(String, String), String> =
        std::collections::HashMap::new(); // (norm, type) -> id
    let mut id_owner: std::collections::HashMap<String, (String, String)> =
        std::collections::HashMap::new(); // id -> owning (norm, type)
    let mut label_to_id: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    let mut label_type_to_id: std::collections::HashMap<(String, String), String> =
        std::collections::HashMap::new();
    for (nd, _) in &valid_nodes {
        let norm = normalize_label(&nd.label);
        let key = (norm.clone(), nd.node_type.clone());
        let id = norm_to_id
            .entry(key.clone())
            .or_insert_with(|| {
                let base = id_for(ont, &nd.node_type, &nd.label);
                let final_id = disambiguate_id(&base, |cand| {
                    if let Some(owner) = id_owner.get(cand) {
                        if owner != &key {
                            return true;
                        }
                    }
                    matches!(g.get_node(cand), Ok(Some(n))
                        if (normalize_label(&n.label), n.node_type.clone()) != key)
                });
                id_owner.insert(final_id.clone(), key.clone());
                final_id
            })
            .clone();
        label_type_to_id.insert((norm.clone(), nd.node_type.clone()), id.clone());
        label_to_id.insert(norm, id);
    }

    // (3) build NodeSpec list (valid nodes only)
    let nodespecs: Vec<NodeSpec> = valid_nodes
        .iter()
        .map(|(nd, canonical)| {
            let id = norm_to_id[&(normalize_label(&nd.label), nd.node_type.clone())].clone();
            // Preserve existing aliases when this upsert omits them. Re-sending a node
            // to touch its label/source (or to re-anchor a dropped edge) must NOT
            // silently wipe an Enum's values — a common small-model self-correction
            // that otherwise blanked the domain.
            let aliases = if nd.aliases.is_empty() {
                g.get_node(&id)
                    .ok()
                    .flatten()
                    .map(|n| n.aliases)
                    .unwrap_or_default()
            } else {
                nd.aliases.clone()
            };
            NodeSpec {
                id,
                node_type: nd.node_type.clone(),
                label: nd.label.clone(),
                aliases,
                source_path: canonical.clone(),
                range: None,
                confidence: None,
            }
        })
        .collect();

    // Validity supplied per node, keyed by the same id() nodespecs use — looked up both by
    // the (4c) requires_validity guarantee below and by the post-write authoring step (7).
    struct ValidityInput {
        valid_from_norm: Option<String>,
        valid_from_raw: Option<String>,
        valid_to_norm: Option<String>,
        valid_to_raw: Option<String>,
    }
    let mut validity_by_id: std::collections::HashMap<String, ValidityInput> = valid_nodes
        .iter()
        .map(|(nd, _)| {
            (
                norm_to_id[&(normalize_label(&nd.label), nd.node_type.clone())].clone(),
                ValidityInput {
                    valid_from_norm: nd.valid_from_norm.clone(),
                    valid_from_raw: nd.valid_from_raw.clone(),
                    valid_to_norm: nd.valid_to_norm.clone(),
                    valid_to_raw: nd.valid_to_raw.clone(),
                },
            )
        })
        .collect();

    // batch ids — an edge may reference a node from this same call before it is committed
    let batch_ids: std::collections::HashSet<String> =
        nodespecs.iter().map(|n| n.id.clone()).collect();

    // (4) resolve edges
    let resolver = EndpointResolver {
        idx,
        g,
        ont,
        label_to_id: &label_to_id,
        label_type_to_id: &label_type_to_id,
        batch_ids: &batch_ids,
    };
    let mut edgespecs: Vec<EdgeSpec> = Vec::new();
    for ue in &edges {
        let (of, ot, oet) = (ue.from.clone(), ue.to.clone(), ue.edge_type.clone());

        // missing endpoint — the most common malformed edge (e.g. MENTIONS with no `to`).
        if ue.from.trim().is_empty() || ue.to.trim().is_empty() {
            let which = if ue.from.trim().is_empty() {
                "from"
            } else {
                "to"
            };
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
        // An endpoint that resolved to a real indexed section (via resolve_section_ref)
        // is a valid document location even when that Section node is not materialised
        // in this graph — e.g. the eval keeps the corpus index and the agent's graph in
        // separate stores. Such endpoints skip the graph-node existence check below.
        let mut from_is_section = false;
        let mut to_is_section = false;
        let mut edge_ok = true;

        // A bare section anchor "#N" (no document) means "section N of this edge's
        // own document": qualify it with the edge's source_path so it resolves
        // instead of dropping. The model knows the document — it set source_path —
        // but writes the target as a short "#1".
        let qualify = |ep: &str| -> String {
            match ep.strip_prefix('#') {
                Some(n) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => {
                    format!("{edge_source_path}#{n}")
                }
                _ => ep.to_string(),
            }
        };
        let from_ep = qualify(&ue.from);
        let to_ep = qualify(&ue.to);
        let (mut from_types, to_types) = ont.endpoint_types(&ue.edge_type);
        // MENTIONS is a built-in edge (no TOML row), so endpoint_types is empty — still
        // prefer the Field when it shares a label with its Enum constraint node.
        if ue.edge_type == "MENTIONS" && from_types.is_empty() {
            from_types = vec!["Field".to_string()];
        }

        // resolve from endpoint
        if batch_ids.contains(&from_ep) {
            from_resolved = Some(from_ep.clone());
        } else {
            match resolve_section_ref(idx, &from_ep) {
                Err(m) => {
                    errs.push(format!("edge {of} -{oet}-> {ot} dropped: {m}"));
                    edge_ok = false;
                }
                Ok(Some(v)) => {
                    // numeric section ref resolved to section id
                    from_resolved = Some(v);
                    from_is_section = true;
                }
                Ok(None) => {
                    // treat as node label (exact, then fuzzy morphology fallback)
                    match resolver.resolve(&from_ep, &from_types) {
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
        }

        // resolve to endpoint
        if batch_ids.contains(&to_ep) {
            to_resolved = Some(to_ep.clone());
        } else {
            match resolve_section_ref(idx, &to_ep) {
                Err(m) => {
                    errs.push(format!("edge {of} -{oet}-> {ot} dropped: {m}"));
                    edge_ok = false;
                }
                Ok(Some(v)) => {
                    // numeric section ref resolved to section id
                    to_resolved = Some(v);
                    to_is_section = true;
                }
                Ok(None) => {
                    // treat as node label (exact, then fuzzy morphology fallback)
                    match resolver.resolve(&to_ep, &to_types) {
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
        }

        if edge_ok {
            let from_id = from_resolved.unwrap();
            let to_id = to_resolved.unwrap();

            // post-resolution existence check (a resolved section ref is validated by
            // the index, so it is exempt — its Section node need not live in this graph)
            let mut exists_ok = true;
            for (role, id, is_section) in [
                ("from", from_id.clone(), from_is_section),
                ("to", to_id.clone(), to_is_section),
            ] {
                if is_section {
                    continue;
                }
                let exists = batch_ids.contains(&id) || g.get_node(&id).ok().flatten().is_some();
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

    // (4b) Grounding guarantee. Node types the ontology marks `requires_grounding`
    // MUST have a MENTIONS edge to a source, in THIS batch or already in the graph.
    // A batch that would leave one ungrounded is rejected whole (nothing written).
    // Document-agnostic: the MENTIONS may cite any document.
    let grounding_errs: Vec<String> = nodespecs
        .iter()
        .filter(|n| ont.requires_grounding(&n.node_type))
        .filter(|n| {
            let in_batch = edgespecs
                .iter()
                .any(|e| e.edge_type == crate::graph::MENTIONS && e.from == n.id);
            let in_graph = g
                .outgoing(&n.id)
                .map(|es| es.iter().any(|e| e.edge_type == crate::graph::MENTIONS))
                .unwrap_or(false);
            !in_batch && !in_graph
        })
        .map(|n| {
            format!(
                "\"{}: {}\" requires grounding: add a MENTIONS edge from it to the document section it is grounded in — {{\"from\":\"{}\",\"edge_type\":\"MENTIONS\",\"to\":\"<doc>#<n>\",\"source_path\":\"<doc>\"}}",
                n.node_type, n.label, n.label
            )
        })
        .collect();
    if !grounding_errs.is_empty() {
        let mut dropped = grounding_errs;
        dropped.extend(errs);
        let out = UpsertOutcome {
            message: String::new(),
            nodes: 0,
            edges: 0,
            rejected: true,
            dump: vec![],
            merged: vec![],
            dropped,
        };
        return UpsertOutcome {
            message: format_upsert_response(&out),
            ..out
        };
    }

    // (4c) Validity guarantee. Node types the ontology marks `requires_validity` MUST carry a
    // valid_from, supplied in THIS batch or already recorded in the graph. Mirrors the grounding
    // guarantee above: a batch that would leave one untimed is rejected whole (nothing written).
    let validity_errs: Vec<String> = nodespecs
        .iter()
        .filter(|n| ont.requires_validity(&n.node_type))
        .filter(|n| {
            let supplied = validity_by_id
                .get(&n.id)
                .map(|v| v.valid_from_norm.is_some())
                .unwrap_or(false);
            let in_graph = g
                .validity_for(&n.id)
                .ok()
                .flatten()
                .map(|v| v.valid_from.is_some())
                .unwrap_or(false);
            !supplied && !in_graph
        })
        .map(|n| {
            format!(
                "\"{}: {}\" requires a validity interval — add valid_from (and optionally valid_to): {{\"type\":\"{}\",\"label\":\"{}\",\"valid_from\":\"<YYYY-MM-DD>\",\"valid_to\":\"<YYYY-MM-DD>\"}}",
                n.node_type, n.label, n.node_type, n.label
            )
        })
        .collect();
    if !validity_errs.is_empty() {
        let mut dropped = validity_errs;
        dropped.extend(errs);
        let out = UpsertOutcome {
            message: String::new(),
            nodes: 0,
            edges: 0,
            rejected: true,
            dump: vec![],
            merged: vec![],
            dropped,
        };
        return UpsertOutcome {
            message: format_upsert_response(&out),
            ..out
        };
    }

    // (5) build dump lines (resolved state, for the caller to log). Echo a node's
    // `aliases` when it has any — for an Enum these ARE the allowed values, so the
    // model sees exactly how many landed and catches a value it dropped while
    // writing the array (it may have named more in its reasoning than it wrote).
    let mut dump = Vec::new();
    for nd in &nodespecs {
        if nd.aliases.is_empty() {
            dump.push(format!("node {} [{}] {}", nd.id, nd.node_type, nd.label));
        } else {
            dump.push(format!(
                "node {} [{}] {} — {} value(s): [{}]",
                // Quote each value: many are decimals written with a comma ("22,23"),
                // so a bare comma-join reads ambiguously (is "22,23" one value or two?).
                nd.id,
                nd.node_type,
                nd.label,
                nd.aliases.len(),
                nd.aliases
                    .iter()
                    .map(|a| format!("\"{a}\""))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
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
            hint_dump.push("Similar existing nodes (use label OR node id in edge from/to):".into());
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

    // Validity to author once the write below succeeds — captured now (by id) since nodespecs
    // is moved into apply_upsert next. Only nodes that actually supplied a bound get a row;
    // a node that merely satisfied (4c) via an existing node_validity row is left untouched.
    let validity_writes: Vec<(String, ValidityInput)> = nodespecs
        .iter()
        .filter_map(|n| {
            let v = validity_by_id.remove(&n.id)?;
            (v.valid_from_norm.is_some() || v.valid_to_norm.is_some()).then_some((n.id.clone(), v))
        })
        .collect();

    // (7) Apply the well-formed items; report any dropped ones so the model resends JUST those.
    match apply_upsert(g, ont, nodespecs, edgespecs, now, &idx.root) {
        Ok(result) => {
            // Author node_validity for each node that supplied a bound. `result.merged` is
            // (requested_id → canonical_id) for every node that converged into an EXISTING
            // graph node under a different id (dedup by normalized label+type in apply_upsert).
            // A node in validity_writes still carries its pre-merge id (that's what nodespecs
            // held before the move above) — remap through `merged` so the row lands on the id
            // the node actually got stored under; otherwise it's authored under an id nothing
            // is stored at, and validity_for(canonical_id) silently misses it. A node that did
            // NOT merge keeps its own id (it IS the canonical id).
            let canonical_of: std::collections::HashMap<&str, &str> = result
                .merged
                .iter()
                .map(|(requested, canonical)| (requested.as_str(), canonical.as_str()))
                .collect();
            for (id, v) in &validity_writes {
                let target = canonical_of.get(id.as_str()).copied().unwrap_or(id.as_str());
                if let Err(e) = g.upsert_validity(
                    target,
                    &NodeValidity {
                        valid_from: v.valid_from_norm.clone(),
                        valid_to: v.valid_to_norm.clone(),
                        valid_from_raw: v.valid_from_raw.clone(),
                        valid_to_raw: v.valid_to_raw.clone(),
                    },
                ) {
                    errs.push(format!("node \"{target}\" validity not recorded: {e}"));
                }
            }
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
pub fn graph_delete(
    idx: &DocIndex,
    g: &GraphStore,
    node_labels: Vec<String>,
    edges: Vec<EdgeRef>,
) -> String {
    let edges: Vec<EdgeRef> = edges
        .into_iter()
        .map(|e| {
            let from_orig = e.from;
            let to_orig = e.to;
            let from = resolve_section_ref(idx, &from_orig)
                .ok()
                .flatten()
                .unwrap_or(from_orig);
            let to = resolve_section_ref(idx, &to_orig)
                .ok()
                .flatten()
                .unwrap_or(to_orig);
            EdgeRef {
                from,
                edge_type: e.edge_type,
                to,
            }
        })
        .collect();
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
                m.push_str(&format!(
                    "\n{} skipped:\n- {}",
                    notes.len(),
                    notes.join("\n- ")
                ));
            }
            m
        }
        Err(e) => format!("graph_update error: {e}"),
    }
}

/// Recompute the graph's DERIVED layer (transitive-closure edges, SIMILAR links, communities,
/// centrality) non-destructively and report the counts. Shared by the MCP `graph_generalize`
/// tool and the eval enricher so both emit identical output. `apply_merges` stays off
/// (from_ontology defaults), so it never deletes or merges nodes. Hygiene (ungrounded/stale/
/// incomplete detection + prune) is a separate concern — see `graph::doctor`.
pub fn graph_generalize(g: &GraphStore, ont: &Ontology, now: u64) -> String {
    let opts = crate::graph::generalize::apply::Opts::from_ontology(ont, now);
    match crate::graph::generalize::apply::generalize(g, &opts) {
        Ok(r) => format!(
            "generalized: inferred_edges={} similar_edges={} communities={} merge_candidates={}",
            r.inferred_edges, r.similar_edges, r.communities, r.merge_candidates
        ),
        Err(e) => format!("graph_generalize error: {e}"),
    }
}

fn fmt_sig(sig: Option<crate::index::manifest::FileSig>) -> String {
    match sig {
        Some(s) => format!("{}:{}", s.mtime_secs, s.size),
        None => "none".to_string(),
    }
}

fn fmt_doubtful_line(d: &crate::graph::doctor::DoubtfulNode) -> String {
    use crate::graph::doctor::Reason;
    let base = format!(
        "  {}  [{}]  {}  {}",
        d.id, d.node_type, d.label, d.source_path
    );
    match &d.reason {
        Reason::Ungrounded => format!("{base}  ungrounded"),
        Reason::Incomplete => format!("{base}  incomplete"),
        Reason::Stale { stored, current } => format!(
            "{base}  stale ({}→{})",
            fmt_sig(*stored),
            fmt_sig(*current)
        ),
    }
}

/// Bounded listing of a doctor bucket: at most `limit` lines, then a `… N more` summary.
fn fmt_bucket(name: &str, nodes: &[crate::graph::doctor::DoubtfulNode], limit: usize) -> String {
    let mut out = format!("{name}: {}\n", nodes.len());
    for d in nodes.iter().take(limit) {
        out.push_str(&fmt_doubtful_line(d));
        out.push('\n');
    }
    if nodes.len() > limit {
        out.push_str(&format!("  … {} more\n", nodes.len() - limit));
    }
    out
}

/// Render a `DoctorReport` as text — id/type/label/source_path/reason per doubtful node (stale
/// includes `stored→current` file signatures), bounded per bucket. Shared by the CLI
/// `graph doctor` command (which also needs the raw `DoctorReport` to drive `--prune-*`) and
/// `graph_doctor` below (report-only, for MCP).
pub fn fmt_doctor_report(rep: &crate::graph::doctor::DoctorReport) -> String {
    const LIMIT: usize = 50;
    let mut out = String::new();
    out.push_str(&fmt_bucket("ungrounded", &rep.ungrounded, LIMIT));
    out.push_str(&fmt_bucket("stale", &rep.stale, LIMIT));
    out.push_str(&fmt_bucket("incomplete", &rep.incomplete, LIMIT));
    out.push_str(&format!("unverifiable: {}\n", rep.unverifiable));
    out
}

/// Diagnose graph health (the three doubts: ungrounded / stale / incomplete) and render the
/// structured report as text so a healing agent (or a human running `kb graph doctor`) can act
/// on it. Report-only: never mutates the store. Shared by the CLI `graph doctor` command and
/// the MCP `graph_doctor` tool so both see identical output.
pub fn graph_doctor(g: &GraphStore, ont: &Ontology, root: &std::path::Path) -> String {
    match crate::graph::doctor::doctor(g, ont, root) {
        Ok(rep) => fmt_doctor_report(&rep),
        Err(e) => format!("graph_doctor error: {e}"),
    }
}

/// Doc-scoped inventory for `graph_stats(doc=…)`: every non-structural node owned by
/// `source_path == doc`, plus **all** outgoing edges. Ontology-independent — `doc` is only
/// a scope filter. Shared by MCP and constraint-eval so both see the same listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedEdgeView {
    pub edge_type: String,
    /// Neighbour label when the endpoint is a graph node; otherwise the raw id / `path#n`.
    pub to: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OwnedNodeView {
    pub id: String,
    pub node_type: String,
    pub label: String,
    pub outgoing: Vec<OwnedEdgeView>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DocOwnedInventory {
    pub nodes: Vec<OwnedNodeView>,
}

/// `None` when the document owns no non-structural nodes yet.
pub fn doc_owned_inventory(g: &GraphStore, doc: &str) -> anyhow::Result<Option<DocOwnedInventory>> {
    let nodes = g.all_nodes()?;
    let edges = g.all_edges()?;
    let structural: std::collections::HashSet<&str> =
        crate::graph::STRUCTURAL_NODES.iter().copied().collect();
    let label_of: std::collections::HashMap<&str, &str> = nodes
        .iter()
        .map(|n| (n.id.as_str(), n.label.as_str()))
        .collect();
    let mut owned: Vec<&Node> = nodes
        .iter()
        .filter(|n| n.prov.source_path == doc && !structural.contains(n.node_type.as_str()))
        .collect();
    if owned.is_empty() {
        return Ok(None);
    }
    owned.sort_by(|a, b| {
        a.node_type
            .cmp(&b.node_type)
            .then_with(|| a.label.cmp(&b.label))
            .then_with(|| a.id.cmp(&b.id))
    });
    let views = owned
        .into_iter()
        .map(|n| {
            let mut outgoing: Vec<OwnedEdgeView> = edges
                .iter()
                .filter(|e| e.from == n.id)
                .map(|e| {
                    let to = label_of
                        .get(e.to.as_str())
                        .map(|l| (*l).to_string())
                        .unwrap_or_else(|| e.to.clone());
                    OwnedEdgeView {
                        edge_type: e.edge_type.clone(),
                        to,
                    }
                })
                .collect();
            outgoing.sort_by(|a, b| a.edge_type.cmp(&b.edge_type).then_with(|| a.to.cmp(&b.to)));
            outgoing.dedup();
            OwnedNodeView {
                id: n.id.clone(),
                node_type: n.node_type.clone(),
                label: n.label.clone(),
                outgoing,
            }
        })
        .collect();
    Ok(Some(DocOwnedInventory { nodes: views }))
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::agent::NodeUpdate;
    use crate::graph::store::{Edge, Node, Provenance};

    fn cov_node(g: &GraphStore, id: &str, nt: &str, label: &str, aliases: &[&str], src: &str) {
        g.put_node(&Node {
            id: id.into(),
            node_type: nt.into(),
            label: label.into(),
            aliases: aliases.iter().map(|s| s.to_string()).collect(),
            prov: Provenance {
                source_path: src.into(),
                range: None,
                file_sig: None,
                origin: "agent".into(),
                confidence: 1.0,
                created_at: 0,
            },
        })
        .unwrap();
    }
    fn cov_edge(g: &GraphStore, from: &str, et: &str, to: &str) {
        g.put_edge(&Edge {
            from: from.into(),
            to: to.into(),
            edge_type: et.into(),
            prov: Provenance {
                source_path: "a.docx".into(),
                range: None,
                file_sig: None,
                origin: "agent".into(),
                confidence: 1.0,
                created_at: 0,
            },
        })
        .unwrap();
    }

    #[test]
    fn doc_owned_inventory_scopes_by_source_path_lists_all_outgoing() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        cov_node(&g, "fld:hgt-a", "Field", "height", &[], "a.docx");
        cov_node(
            &g,
            "enum:hgt-a",
            "Enum",
            "height enum",
            &["10", "20"],
            "a.docx",
        );
        cov_edge(&g, "fld:hgt-a", "CONSTRAINED_BY", "enum:hgt-a");
        cov_node(&g, "sec:a1", "Section", "section 1", &[], "a.docx");
        cov_edge(&g, "fld:hgt-a", "MENTIONS", "sec:a1");
        cov_node(&g, "prm:d", "Param", "D", &[], "a.docx");
        cov_node(&g, "prm:h", "Param", "H", &[], "a.docx");
        cov_edge(&g, "prm:h", "MENTIONS", "a.docx#2");
        cov_edge(&g, "prm:h", "DEPENDS_ON", "prm:d");
        cov_node(&g, "fld:wid-a", "Field", "width", &[], "a.docx");
        cov_node(&g, "fld:dia-b", "Field", "diameter", &[], "b.docx");

        let inv = doc_owned_inventory(&g, "a.docx")
            .unwrap()
            .expect("A owns nodes");
        let labels: Vec<_> = inv.nodes.iter().map(|n| n.label.as_str()).collect();
        assert!(labels.contains(&"height"));
        assert!(labels.contains(&"width"));
        assert!(labels.contains(&"H"), "Param must appear: {labels:?}");
        assert!(
            labels.contains(&"height enum"),
            "Enum owned by doc is listed: {labels:?}"
        );
        assert!(
            !labels.contains(&"section 1"),
            "Section is structural, excluded"
        );
        assert!(!labels.contains(&"diameter"));

        let hgt = inv
            .nodes
            .iter()
            .find(|n| n.id == "fld:hgt-a")
            .expect("height");
        assert!(
            hgt.outgoing
                .iter()
                .any(|e| e.edge_type == "MENTIONS" && e.to == "section 1"),
            "MENTIONS resolves Section to label: {:?}",
            hgt.outgoing
        );
        assert!(
            hgt.outgoing
                .iter()
                .any(|e| e.edge_type == "CONSTRAINED_BY" && e.to == "height enum"),
            "all outgoing types, not only MENTIONS: {:?}",
            hgt.outgoing
        );
        let prm = inv.nodes.iter().find(|n| n.id == "prm:h").expect("Param");
        assert!(prm
            .outgoing
            .iter()
            .any(|e| e.edge_type == "MENTIONS" && e.to == "a.docx#2"));
        assert!(
            prm.outgoing
                .iter()
                .any(|e| e.edge_type == "DEPENDS_ON" && e.to == "D"),
            "DEPENDS_ON must surface: {:?}",
            prm.outgoing
        );
        let wid = inv.nodes.iter().find(|n| n.label == "width").unwrap();
        assert!(wid.outgoing.is_empty());

        assert!(doc_owned_inventory(&g, "zzz.docx").unwrap().is_none());
    }

    #[test]
    fn doc_owned_inventory_ignores_ontology() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        cov_node(&g, "x:1", "Whatever", "X", &[], "doc.pdf");
        cov_edge(&g, "x:1", "MENTIONS", "doc.pdf#1");
        cov_edge(&g, "x:1", "CUSTOM_REL", "doc.pdf#9");
        let inv = doc_owned_inventory(&g, "doc.pdf").unwrap().unwrap();
        assert_eq!(inv.nodes.len(), 1);
        assert_eq!(inv.nodes[0].node_type, "Whatever");
        assert_eq!(inv.nodes[0].outgoing.len(), 2);
        assert!(inv.nodes[0]
            .outgoing
            .iter()
            .any(|e| e.edge_type == "CUSTOM_REL" && e.to == "doc.pdf#9"));
    }

    use crate::graph::ontology::Ontology;

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

    const GROUNDING_ONT: &str = r#"
[entities.Symptom]
id_prefix = "sym"
props = []
[entities.Resolution]
id_prefix = "res"
props = []
requires_grounding = true
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
            valid_from: None,
            valid_to: None,
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
            unode("Symptom", "Connection loss", "case1.docx"),
            unode("Resolution", "Module restart", "case1.docx"),
        ];
        let edges = vec![uedge(
            "Connection loss",
            "RESOLVED_BY",
            "Module restart",
            "case1.docx",
        )];

        let out = graph_upsert(&idx, &g, &ont, nodes, edges, 1_000_000);
        assert!(!out.rejected, "should not be rejected: {}", out.message);
        assert_eq!(out.nodes, 2);
        assert_eq!(out.edges, 1);
        assert!(
            format_upsert_response(&out).contains("upserted 2 nodes, 1 edges"),
            "{}",
            out.message
        );
        assert!(out.message.contains("Written:"), "{}", out.message);
    }

    /// Two nodes in one batch share a normalized label but declare DIFFERENT types.
    /// They are distinct facts and must be written as two distinct nodes — neither dropped,
    /// neither silently collapsed onto the other's id. (Currently the id is cached by label
    /// alone, so the second type conflates onto the first; this test pins the correct
    /// behavior and fails until that is fixed.)
    #[test]
    fn same_label_different_type_stay_distinct() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "case1.docx");

        // Same label "Cache", two distinct declared types, no edges.
        let nodes = vec![
            unode("Symptom", "Cache", "case1.docx"),
            unode("Resolution", "Cache", "case1.docx"),
        ];
        let out = graph_upsert(&idx, &g, &ont, nodes, vec![], 1_000_000);

        assert!(!out.rejected, "should not be rejected: {}", out.message);
        assert!(
            out.dropped.is_empty(),
            "nothing should be dropped: {:?}",
            out.dropped
        );

        // Two distinct nodes must land in the graph, one per type.
        assert_eq!(
            g.node_count().unwrap(),
            2,
            "same label + different type must stay distinct, not merge into one node"
        );
        let sym = g
            .find_by_label_type("Cache", "Symptom")
            .unwrap()
            .expect("Symptom 'Cache' node must exist");
        let res = g
            .find_by_label_type("Cache", "Resolution")
            .unwrap()
            .expect("Resolution 'Cache' node must exist");
        assert_ne!(sym, res, "the two nodes must have distinct ids");
        assert_eq!(out.nodes, 2, "both nodes reported as written");
    }

    /// Two nodes in one batch share the SAME normalized label AND the SAME type. They are the
    /// same fact and must collapse onto a single node — keying identity by (label, type) must
    /// not accidentally split idempotent re-sends.
    #[test]
    fn same_label_same_type_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "case1.docx");
        let out = graph_upsert(
            &idx,
            &g,
            &ont,
            vec![
                unode("Symptom", "Cache", "case1.docx"),
                unode("Symptom", "Cache", "case1.docx"),
            ],
            vec![],
            1_000_000,
        );
        assert!(!out.rejected, "{}", out.message);
        assert_eq!(g.node_count().unwrap(), 1, "same (label,type) is one node");
    }

    /// `<path>#0` resolves to the first chunk. The model often writes 0-based refs out of
    /// programming habit; the tool tolerates it (numbering is 1-based) instead of erroring,
    /// so it doesn't loop guessing chunk numbers. Out-of-range numbers still error, with a
    /// message stating the 1-based convention.
    #[test]
    fn section_ref_zero_maps_to_first_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        write_doc(&idx, "case1.docx"); // single chunk, ord 1

        let first = crate::graph::build::section_id("case1.docx", "1");
        assert_eq!(
            resolve_section_ref(&idx, "case1.docx#0").unwrap(),
            Some(first),
            "#0 should resolve to the first chunk (#1)"
        );

        // A genuinely out-of-range chunk still errors, and the message states the base.
        let err = resolve_section_ref(&idx, "case1.docx#5").unwrap_err();
        assert!(err.contains("start at #1"), "unexpected error: {err}");
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
            unode("Symptom", "Connection loss", "case1.docx"),
            unode("Resolution", "Module restart", "case1.docx"),
        ];
        let edges = vec![uedge(
            "Connection loss",
            "RESOLVED_BY",
            "Module restart",
            "case1.docx",
        )];
        graph_upsert(&idx, &g, &ont, nodes, edges, 1);

        // Rename the Symptom node.
        let ups = vec![NodeUpdate {
            label: "Connection loss".into(),
            new_label: Some("Unstable connection".into()),
            new_type: None,
        }];
        let result = graph_update(&g, ups);
        assert_eq!(result, "updated 1 nodes", "unexpected result: {result}");

        // (a) a node with the new label exists
        let id = g.find_by_label("Unstable connection").unwrap();
        assert!(id.is_some(), "new label not found");
        let id = id.unwrap();

        // (b) the id is the label-derived id for the original label
        let expected_id = id_for(&ont, "Symptom", "Connection loss");
        assert_eq!(id, expected_id, "id should be the label-derived id");

        // (c) the RESOLVED_BY edge still exists
        let res_id = id_for(&ont, "Resolution", "Module restart");
        let out = g.outgoing(&id).unwrap();
        assert!(
            out.iter()
                .any(|e| e.edge_type == "RESOLVED_BY" && e.to == res_id),
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
        let nodes = vec![unode("Symptom", "Connection loss", "case1.docx")];
        let edges = vec![uedge(
            "Connection loss",
            "RESOLVED_BY",
            "Module restart",
            "case1.docx",
        )];

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
            unode("Symptom", "Connection loss", "case1.docx"),
            unode("Resolution", "Restart", "case1.docx"),
        ];
        let edges = vec![uedge(
            "Connection loss",
            "RESOLVED_BY",
            "Restart",
            "case1.docx",
        )];

        let out = graph_upsert(&idx, &g, &ont, nodes, edges, 1_000_000);
        assert!(!out.rejected, "should not be rejected: {}", out.message);
        assert_eq!(out.nodes, 2);
        assert_eq!(out.edges, 1);

        // Assert the RESOLVED_BY edge connects the label-derived ids.
        let sym_id = id_for(&ont, "Symptom", "Connection loss");
        let res_id = id_for(&ont, "Resolution", "Restart");
        let outgoing = g.outgoing(&sym_id).unwrap();
        assert!(
            outgoing
                .iter()
                .any(|e| e.edge_type == "RESOLVED_BY" && e.to == res_id),
            "RESOLVED_BY edge not found from {sym_id} to {res_id}: {outgoing:?}"
        );
    }

    /// A Document endpoint (e.g. DEPENDS_ON doc->doc) written with the agent's forward
    /// slashes must resolve to the Document node stored with the host separator — without
    /// the caller hand-writing backslashes. Regression: `get_node` is an exact match, so a
    /// `docs/foo.docx` endpoint missed the stored `docs\foo.docx` id and fell through to the
    /// fuzzy label match, grabbing a token-overlapping reasoning node instead.
    #[test]
    fn edge_endpoint_document_path_normalizes_separators() {
        use crate::index::manifest::FileSig;
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        idx.write_chunks(&[crate::model::Chunk {
            doc_path: "docs\\case1.docx".into(),
            location: "1".into(),
            file_type: "docx".into(),
            text: "hello".into(),
        }])
        .unwrap();
        crate::graph::build::build_document(
            &g,
            "docs\\case1.docx",
            FileSig {
                mtime_secs: 0,
                size: 0,
            },
        )
        .unwrap();
        // A reasoning node whose label shares tokens with the path — the fuzzy fallback
        // could grab THIS instead of the Document without separator normalization.
        let out = graph_upsert(
            &idx,
            &g,
            &ont,
            vec![unode("Resolution", "case1 docx", "docs\\case1.docx")],
            vec![],
            1_000_000,
        );
        assert!(!out.rejected, "setup node written: {}", out.message);

        let resolver = EndpointResolver {
            idx: &idx,
            g: &g,
            ont: &ont,
            label_to_id: &Default::default(),
            label_type_to_id: &Default::default(),
            batch_ids: &std::collections::HashSet::new(),
        };
        let got = resolver.resolve("docs/case1.docx", &[]);
        assert_eq!(
            got,
            Some("docs\\case1.docx".to_string()),
            "forward-slash Document endpoint should resolve to the stored Document id"
        );
    }

    /// The model routinely prefixes an endpoint label with the node's TYPE
    /// ("Symptom Connection loss" for the node labelled "Connection loss"); the edge used
    /// to drop and the agent retried forever. Strip the type token and resolve.
    #[test]
    fn edge_endpoint_with_type_prefixed_label_resolves() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "case1.docx");

        let nodes = vec![
            unode("Symptom", "Connection loss", "case1.docx"),
            unode("Resolution", "Restart", "case1.docx"),
        ];
        // Both endpoints written type-prefixed, the way the model actually does it.
        let edges = vec![uedge(
            "Symptom Connection loss",
            "RESOLVED_BY",
            "Resolution Restart",
            "case1.docx",
        )];

        let out = graph_upsert(&idx, &g, &ont, nodes, edges, 1_000_000);
        assert_eq!(
            out.edges, 1,
            "type-prefixed endpoints should resolve: {}",
            out.message
        );
        let sym_id = id_for(&ont, "Symptom", "Connection loss");
        let res_id = id_for(&ont, "Resolution", "Restart");
        assert!(g
            .outgoing(&sym_id)
            .unwrap()
            .iter()
            .any(|e| e.edge_type == "RESOLVED_BY" && e.to == res_id));

        // An ordinary multi-word label whose first word is NOT a type is untouched:
        // "Connection loss" itself still resolves as before (no mangling).
        let resolver = EndpointResolver {
            idx: &idx,
            g: &g,
            ont: &ont,
            label_to_id: &Default::default(),
            label_type_to_id: &Default::default(),
            batch_ids: &std::collections::HashSet::new(),
        };
        let ok = resolver.resolve("Connection loss", &[]);
        assert_eq!(ok, Some(sym_id));
    }

    /// Edge whose `to` label has no corresponding node is rejected; message names the label.
    #[test]
    fn edge_to_undefined_label_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "case1.docx");

        let nodes = vec![unode("Symptom", "Connection loss", "case1.docx")];
        let edges = vec![uedge(
            "Connection loss",
            "RESOLVED_BY",
            "Unknown node label",
            "case1.docx",
        )];

        let out = graph_upsert(&idx, &g, &ont, nodes, edges, 1_000_000);
        // Partial apply: the Symptom IS written; the edge to an undefined label is dropped (named).
        assert!(!out.rejected, "valid node written: {}", out.message);
        assert_eq!(out.nodes, 1);
        assert_eq!(out.edges, 0);
        assert!(
            out.message.contains("Unknown node label") && out.message.contains("dropped"),
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
            unode("Symptom", "Connection loss", "case1.docx"),
            unode("Resolution", "Restart", "case1.docx"),
        ];
        // a MENTIONS edge with no `to` target — malformed.
        let edges = vec![uedge("Connection loss", "MENTIONS", "", "case1.docx")];

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
        assert!(
            msg.contains("no update received"),
            "must explain the empty case: {msg}"
        );
        assert!(
            msg.contains("\"nodes\""),
            "must show the expected shape: {msg}"
        );
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
        let _ = graph_upsert(
            &idx,
            &g,
            &ont,
            vec![unode("Symptom", "Connection loss", "case1.docx")],
            vec![],
            1,
        );

        let del = graph_delete(&idx, &g, vec!["Nonexistent".into()], vec![]);
        assert!(
            del.contains("matched nothing"),
            "delete names the unmatched ref: {del}"
        );

        let upd = graph_update(
            &g,
            vec![NodeUpdate {
                label: "Nonexistent".into(),
                new_label: Some("X".into()),
                new_type: None,
            }],
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
            &idx,
            &g,
            &ont,
            vec![
                unode("Symptom", "Fieldbus connection loss", "c.docx"),
                unode(
                    "Resolution",
                    "Change maxTsdr parameter and restart service",
                    "c.docx",
                ),
            ],
            vec![uedge(
                "Fieldbus connection loss",
                "RESOLVED_BY",
                "Change maxTsdr parameter and restart service",
                "c.docx",
            )],
            1,
        );
        assert!(!out1.rejected, "{}", out1.message);

        // Later edge references the Resolution by a TRUNCATED label.
        let out2 = graph_upsert(
            &idx,
            &g,
            &ont,
            vec![],
            vec![uedge(
                "Fieldbus connection loss",
                "RESOLVED_BY",
                "Change maxTsdr parameter",
                "c.docx",
            )],
            2,
        );
        assert!(
            !out2.rejected,
            "truncated label must resolve fuzzily: {}",
            out2.message
        );
    }

    /// Fix 1 — node with a source_path not in the index is rejected; a real path is accepted.
    #[test]
    fn rejects_node_with_fake_source_path() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();

        // Index only one real document.
        write_doc(&idx, "kb-test\\extra-data\\real.pdf");

        // Node with a hallucinated path — must be rejected.
        let out = graph_upsert(
            &idx,
            &g,
            &ont,
            vec![unode("Symptom", "Connection loss", "case_support_001")],
            vec![],
            1,
        );
        assert!(
            out.rejected,
            "hallucinated source_path must be rejected: {}",
            out.message
        );
        assert!(
            out.message.contains("is not a document"),
            "message should say 'is not a document': {}",
            out.message
        );

        // Same node but with the real indexed path — must be accepted.
        let out_real = graph_upsert(
            &idx,
            &g,
            &ont,
            vec![unode(
                "Symptom",
                "Connection loss",
                "kb-test\\extra-data\\real.pdf",
            )],
            vec![],
            2,
        );
        assert!(
            !out_real.rejected,
            "real source_path must be accepted: {}",
            out_real.message
        );
    }

    #[test]
    fn upsert_accepts_prefixed_source_path() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "kb-test\\extra-data\\real.pdf");

        let out = graph_upsert(
            &idx,
            &g,
            &ont,
            vec![unode(
                "Symptom",
                "Connection loss",
                "kb-manual\\kb-test\\extra-data\\real.pdf",
            )],
            vec![],
            1,
        );
        assert!(
            !out.rejected,
            "prefixed source_path must resolve: {}",
            out.message
        );
        let sym_id = id_for(&ont, "Symptom", "Connection loss");
        let node = g.get_node(&sym_id).unwrap().unwrap();
        assert_eq!(node.prov.source_path, "kb-test\\extra-data\\real.pdf");
    }

    #[test]
    fn upsert_rejects_bad_path_with_did_you_mean() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "kb-test\\extra-data\\real.pdf");

        let out = graph_upsert(
            &idx,
            &g,
            &ont,
            vec![unode("Symptom", "Connection loss", "wrong\\real.pdf")],
            vec![],
            1,
        );
        assert!(out.rejected, "bad path must reject: {}", out.message);
        assert!(
            out.message.contains("did you mean"),
            "upsert hint: {}",
            out.message
        );
        assert!(
            out.message.contains("real.pdf"),
            "upsert suggests real path: {}",
            out.message
        );
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
        }])
        .unwrap();
        // A numeric ref keeps its ordinal (`#1`), not the chunk's heading location.
        let resolved = resolve_section_ref(&idx, "kb-manual\\docs\\real.md#1").unwrap();
        assert_eq!(resolved, Some("docs\\real.md#1".to_string()));
    }

    #[test]
    fn resolve_section_ref_blank_pdf_page() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            crate::model::Chunk {
                doc_path: "spec.pdf".into(),
                location: "p.3".into(),
                file_type: "pdf".into(),
                text: "body".into(),
            },
            crate::model::Chunk {
                doc_path: "spec.pdf".into(),
                location: "p.4".into(),
                file_type: "pdf".into(),
                text: String::new(),
            },
        ])
        .unwrap();
        let resolved = resolve_section_ref(&idx, "spec.pdf#4").unwrap();
        assert_eq!(resolved, Some("spec.pdf#4".to_string()));
    }

    /// MENTIONS target that resolves to a real section is NOT a node in the agent graph.
    #[test]
    fn parse_upsert_moves_edge_shaped_object_from_nodes() {
        let v: Value = serde_json::from_str(
            r#"{
            "nodes": [
                {"node_type":"Enum","label":"Type","source_path":"spec.pdf","aliases":["41","42"]},
                {"edge_type":"CONSTRAINED_BY","from":"fld:type","to":"Type","source_path":"spec.pdf"}
            ],
            "edges": [
                {"edge_type":"MENTIONS","from":"fld:type","to":"spec.pdf#1","source_path":"spec.pdf"}
            ]
        }"#,
        )
        .unwrap();
        let (nodes, edges, notes) = parse_upsert_payload(&v);
        assert_eq!(nodes.len(), 1);
        assert_eq!(edges.len(), 2);
        assert!(notes.iter().any(|n| n.contains("moved to edges")));
    }

    #[test]
    fn mentions_from_label_shared_with_enum_targets_field() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(
            r#"
[entities.Field]
id_prefix = "fld"
[entities.Enum]
id_prefix = "enum"
[relations.CONSTRAINED_BY]
from = ["Field"]
to = ["Enum"]
"#,
        )
        .unwrap();
        idx.write_chunks(&[crate::model::Chunk {
            doc_path: "spec.pdf".into(),
            location: String::new(),
            file_type: "pdf".into(),
            text: "values table".into(),
        }])
        .unwrap();
        assert!(
            !graph_upsert(
                &idx,
                &g,
                &ont,
                vec![unode("Field", "Type", "spec.pdf")],
                vec![],
                1,
            )
            .rejected
        );
        let out = graph_upsert(
            &idx,
            &g,
            &ont,
            vec![unode("Enum", "Type", "spec.pdf")],
            vec![uedge("Type", "MENTIONS", "#1", "spec.pdf")],
            2,
        );
        assert!(!out.rejected, "{}", out.message);
        let fld = id_for(&ont, "Field", "Type");
        assert!(
            g.outgoing(&fld)
                .unwrap()
                .iter()
                .any(|e| e.edge_type == "MENTIONS" && e.to == "spec.pdf#1"),
            "MENTIONS must attach to the Field, not the Enum: {:?}",
            g.outgoing(&fld).unwrap()
        );
        let c = doc_owned_inventory(&g, "spec.pdf").unwrap().unwrap();
        assert!(
            c.nodes.iter().any(|n| {
                n.label == "Type"
                    && n.node_type == "Field"
                    && n.outgoing
                        .iter()
                        .any(|e| e.edge_type == crate::graph::MENTIONS)
            }),
            "Field Type should list MENTIONS: {:?}",
            c.nodes
        );
    }

    /// Eval scenario: the corpus index and the agent's graph are SEPARATE stores, so a
    /// The edge must still land — a bare "#1" is qualified by the edge's source_path, and
    /// a resolved section ref is exempt from the node-existence check. A single-chunk
    /// heading-less doc has no location, so its location falls back to the ordinal and
    /// its section id is "spec.docx#1".
    #[test]
    fn mentions_to_section_ref_survives_separate_agent_graph() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        idx.write_chunks(&[crate::model::Chunk {
            doc_path: "spec.docx".into(),
            location: String::new(),
            file_type: "docx".into(),
            text: "table of allowed values".into(),
        }])
        .unwrap();
        // Bare "#1" target (no path) — the model's short form.
        let out = graph_upsert(
            &idx,
            &g,
            &ont,
            vec![unode("Symptom", "Type", "spec.docx")],
            vec![uedge("Type", "MENTIONS", "#1", "spec.docx")],
            1,
        );
        assert!(!out.rejected, "upsert rejected: {}", out.message);
        let from_id = id_for(&ont, "Symptom", "Type");
        let outgoing = g.outgoing(&from_id).unwrap();
        assert!(
            outgoing
                .iter()
                .any(|e| e.edge_type == "MENTIONS" && e.to == "spec.docx#1"),
            "MENTIONS edge to a section ref must land; got: {:?}",
            outgoing
                .iter()
                .map(|e| (e.edge_type.clone(), e.to.clone()))
                .collect::<Vec<_>>()
        );
    }

    /// Delete a MENTIONS edge by label + `<path>#<n>` (the same shape the agent writes on upsert).
    /// Section targets are not reasoning nodes — delete must still remove the edge and not
    /// report a silent `deleted 0`.
    #[test]
    fn graph_delete_removes_mentions_edge_to_section() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        idx.write_chunks(&[crate::model::Chunk {
            doc_path: "spec.docx".into(),
            location: String::new(),
            file_type: "docx".into(),
            text: "table".into(),
        }])
        .unwrap();
        let out = graph_upsert(
            &idx,
            &g,
            &ont,
            vec![unode("Symptom", "Voltage", "spec.docx")],
            vec![uedge("Voltage", "MENTIONS", "spec.docx#1", "spec.docx")],
            1,
        );
        assert!(!out.rejected, "{}", out.message);

        let msg = graph_delete(
            &idx,
            &g,
            vec![],
            vec![EdgeRef {
                from: "Voltage".into(),
                edge_type: "MENTIONS".into(),
                to: "spec.docx#1".into(),
            }],
        );
        assert!(
            msg.contains("deleted 1") || msg.starts_with("deleted ") && !msg.contains("deleted 0"),
            "must delete the MENTIONS edge: {msg}"
        );
        assert!(
            !msg.contains("matched nothing") && !msg.contains("matched no"),
            "successful delete must not look like a miss: {msg}"
        );
        let from_id = id_for(&ont, "Symptom", "Voltage");
        let outgoing = g.outgoing(&from_id).unwrap();
        assert!(
            !outgoing.iter().any(|e| e.edge_type == "MENTIONS"),
            "MENTIONS edge must be gone: {outgoing:?}"
        );
    }

    /// Missing MENTIONS delete must explain why, not bare `deleted 0 graph entries`.
    #[test]
    fn graph_delete_missing_mentions_edge_explains() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        idx.write_chunks(&[crate::model::Chunk {
            doc_path: "spec.docx".into(),
            location: String::new(),
            file_type: "docx".into(),
            text: "table".into(),
        }])
        .unwrap();
        assert!(
            !graph_upsert(
                &idx,
                &g,
                &ont,
                vec![unode("Symptom", "Voltage", "spec.docx")],
                vec![],
                1,
            )
            .rejected
        );

        let msg = graph_delete(
            &idx,
            &g,
            vec![],
            vec![EdgeRef {
                from: "Voltage".into(),
                edge_type: "MENTIONS".into(),
                to: "spec.docx#1".into(),
            }],
        );
        assert!(msg.contains("deleted 0"), "{msg}");
        assert!(
            msg.contains("matched no") || msg.contains("matched nothing"),
            "must explain the no-op: {msg}"
        );
    }

    /// Re-upserting a node with NO aliases must not wipe the values it already has —
    /// the small model does this while fixing something else, and it blanked domains.
    #[test]
    fn reupsert_without_aliases_preserves_existing() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        idx.write_chunks(&[crate::model::Chunk {
            doc_path: "d.md".into(),
            location: "Intro".into(),
            file_type: "md".into(),
            text: "content".into(),
        }])
        .unwrap();
        let mut n = unode("Symptom", "Type", "d.md");
        n.aliases = vec!["41".into(), "42".into()];
        assert!(!graph_upsert(&idx, &g, &ont, vec![n], vec![], 1).rejected);
        // Re-send the same node with no aliases (e.g. re-anchoring a dropped edge).
        assert!(
            !graph_upsert(
                &idx,
                &g,
                &ont,
                vec![unode("Symptom", "Type", "d.md")],
                vec![],
                2
            )
            .rejected
        );
        let node = g
            .get_node(&id_for(&ont, "Symptom", "Type"))
            .unwrap()
            .unwrap();
        assert_eq!(
            node.aliases,
            vec!["41".to_string(), "42".to_string()],
            "aliases must survive a no-alias re-upsert"
        );
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
            &idx,
            &g,
            &ont,
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
            &idx,
            &g,
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
            &idx,
            &g,
            vec![],
            vec![EdgeRef {
                from: "Test Symptom".into(),
                edge_type: "RESOLVED_BY".into(),
                to: "real.md#1".into(),
            }],
        );
        assert!(
            !msg2.contains("error"),
            "section ref resolution must not produce error: {msg2}"
        );
        assert!(
            msg2.contains("matched no") || msg2.contains("matched nothing"),
            "absent edge must explain the no-op, not bare deleted 0: {msg2}"
        );

        // Non-existent chunk ref: resolve_section_ref errors, original kept, no panic.
        let msg3 = graph_delete(
            &idx,
            &g,
            vec![],
            vec![EdgeRef {
                from: "Test Symptom".into(),
                edge_type: "RESOLVED_BY".into(),
                to: "real.md#999".into(),
            }],
        );
        assert!(
            !msg3.contains("error"),
            "non-existent chunk ref must not produce error: {msg3}"
        );
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
            vec![unode("Symptom", "sym:CPU hot swap", "case1.docx")],
            vec![],
            1,
        );
        assert!(!out.rejected, "{}", out.message);
        let expected = id_for(&ont, "Symptom", "CPU hot swap");
        assert!(
            expected.starts_with("sym:") && expected.len() <= 16,
            "short hashed id: {expected}"
        );
        assert!(g.get_node(&expected).unwrap().is_some());
        assert!(out.message.contains("Written:"), "{}", out.message);
        assert!(out.message.contains(&expected), "{}", out.message);
    }

    #[test]
    fn id_for_is_short_stable_and_hashed() {
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        let long =
            "Heliocentrism was proposed as early as the 3rd century BC by Aristarchus of Samos";
        let id = id_for(&ont, "Symptom", long);
        assert!(id.starts_with("sym:"), "keeps the type abbrev prefix: {id}");
        assert!(
            id.chars().count() <= 32,
            "id must be short/copyable, got {} chars: {id}",
            id.chars().count()
        );
        // Deterministic + dedup-stable: same normalized label -> same id.
        assert_eq!(id, id_for(&ont, "Symptom", long), "deterministic");
        assert_eq!(
            id,
            id_for(
                &ont,
                "Symptom",
                "  heliocentrism WAS proposed as early as the 3rd century BC by aristarchus of samos "
            ),
            "normalized-label-stable (dedup relies on this)"
        );
        // Two different labels that share the readable prefix must still get different ids
        // (the hash disambiguates) — otherwise short ids would silently merge distinct facts.
        let other = "Heliocentrism was proposed by Copernicus in the 16th century instead";
        assert_ne!(
            id,
            id_for(&ont, "Symptom", other),
            "different labels -> different ids"
        );
    }

    #[test]
    fn disambiguate_id_appends_suffix_only_on_collision() {
        // free base id -> unchanged
        assert_eq!(disambiguate_id("sym:foo-ab12", |_| false), "sym:foo-ab12");
        // base taken by a different label -> next free numeric suffix
        let taken = |id: &str| id == "sym:foo-ab12" || id == "sym:foo-ab12-2";
        assert_eq!(disambiguate_id("sym:foo-ab12", taken), "sym:foo-ab12-3");
    }

    /// Cyrillic label where the type-prefix byte length lands mid-char — must not panic.
    #[test]
    fn sanitize_survives_multibyte_label() {
        const ONT: &str = r#"
[entities.EnumType]
id_prefix = "enum"
props = []
[validation]
strict = true
"#;
        let ont = Ontology::parse(ONT).unwrap();
        // prefix "enum:" is 5 bytes; byte 5 falls inside 'ü' (bytes 4..6) of "TypeüEnum"
        assert_eq!(
            sanitize_label_for_upsert(&ont, "EnumType", "TypeüEnum"),
            "TypeüEnum"
        );
        assert_eq!(
            sanitize_label_for_upsert(&ont, "EnumType", "enum: TypeüEnum"),
            "TypeüEnum"
        );
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
            vec![unode("Symptom", "Connection loss", "case1.docx")],
            vec![],
            1,
        );
        let sym_id = id_for(&ont, "Symptom", "Connection loss");
        let out = graph_upsert(
            &idx,
            &g,
            &ont,
            vec![unode("Resolution", "Restart", "case1.docx")],
            vec![uedge(&sym_id, "RESOLVED_BY", "Restart", "case1.docx")],
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
                label: "Connection loss".into(),
                aliases: vec![],
                source_path: "case1.docx".into(),
                range: None,
                confidence: None,
            }],
            vec![],
            1,
            dir.path(),
        )
        .unwrap();

        let out = graph_upsert(
            &idx,
            &g,
            &ont,
            vec![unode("Symptom", "Connection loss", "case1.docx")],
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
        assert!(
            out.message.contains("REJECTED — nothing written"),
            "{}",
            out.message
        );
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
            out.message
                .contains("Similar existing nodes (use label OR node id in edge from/to)"),
            "hint wording: {}",
            out.message
        );
        assert!(
            out.message.contains("Existing symptom"),
            "hint should list the existing node by its (usable) label: {}",
            out.message
        );
    }

    #[test]
    fn grounding_required_node_without_mentions_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(GROUNDING_ONT).unwrap();
        write_doc(&idx, "case1.docx");
        let out = graph_upsert(&idx, &g, &ont, vec![unode("Resolution", "Module restart", "case1.docx")], vec![], 1_000_000);
        assert!(out.rejected, "{}", out.message);
        assert_eq!(out.nodes, 0);
        assert_eq!(out.edges, 0);
        assert!(out.message.contains("requires grounding"), "{}", out.message);
        assert!(out.message.contains("Module restart"), "{}", out.message);
        assert!(g.all_nodes().unwrap().is_empty());
    }

    #[test]
    fn grounding_satisfied_by_mentions_in_same_batch() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(GROUNDING_ONT).unwrap();
        write_doc(&idx, "case1.docx");
        let out = graph_upsert(&idx, &g, &ont,
            vec![unode("Resolution", "Module restart", "case1.docx")],
            vec![uedge("Module restart", "MENTIONS", "case1.docx#1", "case1.docx")], 1_000_000);
        assert!(!out.rejected, "{}", out.message);
        assert_eq!(out.nodes, 1);
        assert_eq!(out.edges, 1);
    }

    #[test]
    fn grounding_mentions_to_other_document_is_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(GROUNDING_ONT).unwrap();
        write_doc(&idx, "case1.docx");
        write_doc(&idx, "other.docx");
        let out = graph_upsert(&idx, &g, &ont,
            vec![unode("Resolution", "Cross ref fix", "case1.docx")],
            vec![uedge("Cross ref fix", "MENTIONS", "other.docx#1", "other.docx")], 1_000_000);
        assert!(!out.rejected, "{}", out.message);
        assert_eq!(out.edges, 1);
    }

    #[test]
    fn grounding_satisfied_by_existing_edge_on_update() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(GROUNDING_ONT).unwrap();
        write_doc(&idx, "case1.docx");
        let out1 = graph_upsert(&idx, &g, &ont,
            vec![unode("Resolution", "Module restart", "case1.docx")],
            vec![uedge("Module restart", "MENTIONS", "case1.docx#1", "case1.docx")], 1);
        assert!(!out1.rejected, "{}", out1.message);
        let out2 = graph_upsert(&idx, &g, &ont,
            vec![unode("Resolution", "Module restart", "case1.docx")], vec![], 2);
        assert!(!out2.rejected, "existing grounding must satisfy: {}", out2.message);
    }

    #[test]
    fn non_required_type_without_grounding_unaffected() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(GROUNDING_ONT).unwrap();
        write_doc(&idx, "case1.docx");
        let out = graph_upsert(&idx, &g, &ont, vec![unode("Symptom", "Connection loss", "case1.docx")], vec![], 1_000_000);
        assert!(!out.rejected, "{}", out.message);
        assert_eq!(out.nodes, 1);
    }

    #[test]
    fn graph_generalize_text_is_derived_layer_only() {
        // generalize no longer runs/reports hygiene (ungrounded/incomplete detection moved to
        // `graph::doctor` — see doctor::tests::doctor_reports_three_buckets for that coverage).
        // An ungrounded node must NOT show up in the generalize summary.
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let prov = crate::graph::store::Provenance {
            source_path: "doc.md".into(), range: None, file_sig: None,
            origin: "agent".into(), confidence: 1.0, created_at: 1,
        };
        g.put_node(&crate::graph::store::Node {
            id: "res:b".into(), node_type: "Resolution".into(), label: "Ungrounded".into(),
            aliases: vec![], prov,
        }).unwrap();
        let ont = Ontology::parse(GROUNDING_ONT).unwrap();
        let text = graph_generalize(&g, &ont, 1);
        assert!(
            text.contains("inferred_edges=")
                && text.contains("similar_edges=")
                && text.contains("communities=")
                && text.contains("merge_candidates="),
            "{text}"
        );
        assert!(!text.contains("ungrounded"), "{text}");
        assert!(!text.contains("Ungrounded"), "{text}");
    }

    const VALIDITY_ONT: &str = r#"
[entities.Record]
id_prefix = "rec"
props = []
requires_validity = true
[validation]
strict = true
"#;

    /// `graph_upsert` authors a `node_validity` row (normalized period edges) for a node
    /// that supplied `valid_from`/`valid_to`, keeping the raw string alongside.
    #[test]
    fn upsert_writes_validity_and_normalizes() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(VALIDITY_ONT).unwrap();
        write_doc(&idx, "a.md");

        let node = UpsertNode {
            valid_from: Some("2020".into()),
            valid_to: Some("2023-06".into()),
            ..unode("Record", "clearance", "a.md")
        };
        let out = graph_upsert(&idx, &g, &ont, vec![node], vec![], 0);
        assert!(!out.rejected, "{}", out.message);

        let id = id_for(&ont, "Record", "clearance");
        let v = g.validity_for(&id).unwrap().unwrap();
        assert_eq!(v.valid_from.as_deref(), Some("2020-01-01T00:00:00Z"));
        assert_eq!(v.valid_to.as_deref(), Some("2023-06-30T23:59:59Z"));
        assert_eq!(v.valid_from_raw.as_deref(), Some("2020"));
    }

    /// A `requires_validity` node with no `valid_from` (neither in this batch nor already in
    /// the graph) is rejected whole — mirrors the grounding-guarantee reject semantics.
    #[test]
    fn requires_validity_rejects_untimed_node() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(VALIDITY_ONT).unwrap();
        write_doc(&idx, "a.md");

        // no valid_from → rejected, nothing written
        let untimed = unode("Record", "x", "a.md");
        let out = graph_upsert(&idx, &g, &ont, vec![untimed], vec![], 0);
        assert!(out.rejected, "{}", out.message);
        assert_eq!(out.nodes, 0);
        assert!(
            out.message.contains("requires a validity interval"),
            "{}",
            out.message
        );
        assert!(g.all_nodes().unwrap().is_empty());

        // with valid_from → accepted
        let timed = UpsertNode {
            valid_from: Some("2024-01-01".into()),
            ..unode("Record", "y", "a.md")
        };
        let out2 = graph_upsert(&idx, &g, &ont, vec![timed], vec![], 0);
        assert!(!out2.rejected, "{}", out2.message);
    }

    /// A node that DEDUP-MERGES into an existing graph node under a different id (same
    /// scenario as `dedup_reports_merged_in_response`) must have its validity authored under
    /// the id it actually landed on (the pre-existing "legacy" id), not the pre-merge
    /// `id_for()`-derived id — nothing is ever stored under that id, so a row written there
    /// would be a silent orphan invisible to `validity_for(canonical_id)`.
    #[test]
    fn validity_write_follows_merge_to_canonical_id() {
        use crate::graph::agent::{apply_upsert, NodeSpec};

        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        write_doc(&idx, "case1.docx");

        // Pre-existing node under a "legacy" id that does NOT match id_for(type, label).
        apply_upsert(
            &g,
            &ont,
            vec![NodeSpec {
                id: "sym:legacy-id".into(),
                node_type: "Symptom".into(),
                label: "Connection loss".into(),
                aliases: vec![],
                source_path: "case1.docx".into(),
                range: None,
                confidence: None,
            }],
            vec![],
            1,
            dir.path(),
        )
        .unwrap();

        // Upsert the SAME type+label with a validity bound — apply_upsert will dedup this
        // into "sym:legacy-id" (same as dedup_reports_merged_in_response).
        let node = UpsertNode {
            valid_from: Some("2021-01-01".into()),
            valid_to: Some("2022-01-01".into()),
            ..unode("Symptom", "Connection loss", "case1.docx")
        };
        let out = graph_upsert(&idx, &g, &ont, vec![node], vec![], 2);
        assert!(!out.rejected, "{}", out.message);
        assert_eq!(out.nodes, 0, "deduped node not created");
        assert!(out.message.contains("sym:legacy-id"), "{}", out.message);

        // The validity row must be on the canonical (stored) id...
        let v = g
            .validity_for("sym:legacy-id")
            .unwrap()
            .expect("validity authored under the canonical/stored id");
        assert_eq!(v.valid_from.as_deref(), Some("2021-01-01T00:00:00Z"));
        assert_eq!(v.valid_to.as_deref(), Some("2022-01-01T23:59:59Z"));

        // ...and NOT orphaned under the pre-merge id_for()-derived id (nothing is stored there).
        let pre_merge_id = id_for(&ont, "Symptom", "Connection loss");
        assert_ne!(pre_merge_id, "sym:legacy-id");
        assert!(
            g.validity_for(&pre_merge_id).unwrap().is_none(),
            "no orphan row under the pre-merge id"
        );
    }
}
