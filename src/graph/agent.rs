use crate::graph::ontology::Ontology;
use crate::graph::store::{normalize_label, Edge, GraphStore, Node, NodeValidity, Provenance};
use crate::graph::temporal;
use crate::index::manifest::FileSig;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct NodeSpec {
    pub id: String,
    pub node_type: String,
    pub label: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub source_path: String,
    #[serde(default)]
    pub range: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
    /// Start of this node's authored validity interval, any ISO-8601 granularity
    /// (`"2020"`, `"2020-06"`, `"2020-06-15"`, or a full RFC3339 UTC instant).
    /// `None` when the caller doesn't touch validity (mirrors `ops::UpsertNode`).
    #[serde(default, deserialize_with = "crate::json_util::deserialize_opt_string_loose")]
    pub valid_from: Option<String>,
    /// End of the authored validity interval, same granularity rules as `valid_from`.
    #[serde(default, deserialize_with = "crate::json_util::deserialize_opt_string_loose")]
    pub valid_to: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct EdgeSpec {
    pub from: String,
    pub to: String,
    pub edge_type: String,
    pub source_path: String,
    #[serde(default)]
    pub range: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
}

/// Resolve `source_path` against the corpus `root` — matching every staleness CONSUMER
/// (`hygiene::stale_nodes`, `doctor::doctor`, `tools::StaleChecker`), which all re-stat
/// `root.join(source_path)`. Stamping against the bare (CWD-relative) path here used to
/// silently defeat staleness whenever the process CWD differed from the corpus root (the
/// normal MCP-server deployment).
fn stat_sig(root: &std::path::Path, source_path: &str) -> Option<FileSig> {
    crate::index::store::file_sig(&root.join(source_path)).ok()
}

/// Result of applying an agent upsert batch (after dedup by label+type).
pub struct ApplyUpsertResult {
    pub nodes_written: usize,
    pub edges_written: usize,
    /// requested_id → canonical_id when the node merged into an existing one
    pub merged: Vec<(String, String)>,
}

/// Convert agent-supplied specs into provenance-stamped graph elements
/// (origin="agent", file_sig from the current source file, created_at=`now`),
/// validate against the ontology, and upsert. Deduplicates nodes by normalized
/// label+type so the graph converges across cases.
pub fn apply_upsert(
    g: &GraphStore,
    ont: &Ontology,
    nodes: Vec<NodeSpec>,
    edges: Vec<EdgeSpec>,
    now: u64,
    root: &std::path::Path,
) -> anyhow::Result<ApplyUpsertResult> {
    use std::collections::HashMap;

    let prov = |source_path: &str, range: Option<String>, confidence: Option<f32>| Provenance {
        source_path: source_path.to_string(),
        range,
        file_sig: stat_sig(root, source_path),
        origin: "agent".into(),
        confidence: confidence.unwrap_or(0.8),
        created_at: now,
    };

    // Build canonical-id map: nodes with same normalized label+type converge to one id.
    let mut canonical: HashMap<String, String> = HashMap::new();
    for n in &nodes {
        let mut canon = n.id.clone();
        // 1. Check earlier nodes in this batch first.
        'batch: for earlier in &nodes {
            if earlier.id == n.id {
                break 'batch;
            }
            if earlier.node_type == n.node_type
                && normalize_label(&earlier.label) == normalize_label(&n.label)
            {
                canon = canonical
                    .get(&earlier.id)
                    .cloned()
                    .unwrap_or_else(|| earlier.id.clone());
                break 'batch;
            }
        }
        // 2. If still unmerged, check the persistent graph.
        if canon == n.id {
            if let Some(existing_id) = g.find_by_label_type(&n.label, &n.node_type)? {
                if existing_id != n.id {
                    canon = existing_id;
                }
            }
        }
        canonical.insert(n.id.clone(), canon);
    }

    let merged: Vec<(String, String)> = nodes
        .iter()
        .filter_map(|n| {
            canonical
                .get(&n.id)
                .filter(|c| *c != &n.id)
                .map(|c| (n.id.clone(), c.clone()))
        })
        .collect();

    // Normalize any authored validity bounds up front (mirroring `ops::parse_and_validate_upsert`)
    // so a malformed date aborts the WHOLE batch before anything is written, same strictness as
    // the ontology-type check below. Only nodes that supplied at least one bound get an entry;
    // the row is keyed by the node's CANONICAL id (post label+type dedup) — validity attaches to
    // the merged node, not a pre-merge id that nothing ends up stored under.
    let mut validity_writes: Vec<(String, NodeValidity)> = Vec::new();
    for n in &nodes {
        if n.valid_from.is_none() && n.valid_to.is_none() {
            continue;
        }
        let valid_from = n
            .valid_from
            .as_deref()
            .map(temporal::normalize_from)
            .transpose()
            .map_err(|e| anyhow::anyhow!("node \"{}\": invalid valid_from: {e}", n.label))?;
        let valid_to = n
            .valid_to
            .as_deref()
            .map(temporal::normalize_to)
            .transpose()
            .map_err(|e| anyhow::anyhow!("node \"{}\": invalid valid_to: {e}", n.label))?;
        if valid_from.is_none() && valid_to.is_none() {
            continue;
        }
        let canon_id = canonical.get(&n.id).cloned().unwrap_or_else(|| n.id.clone());
        validity_writes.push((
            canon_id,
            NodeValidity {
                valid_from,
                valid_to,
                valid_from_raw: n.valid_from.clone(),
                valid_to_raw: n.valid_to.clone(),
            },
        ));
    }

    // Only create nodes whose canonical id is their own id (genuinely new ones).
    let model_nodes: Vec<Node> = nodes
        .iter()
        .filter(|n| canonical.get(&n.id).map(|c| c == &n.id).unwrap_or(true))
        .map(|n| {
            let p = prov(&n.source_path, n.range.clone(), n.confidence);
            Node {
                id: n.id.clone(),
                node_type: n.node_type.clone(),
                label: n.label.clone(),
                aliases: n.aliases.clone(),
                prov: p,
            }
        })
        .collect();

    // Rewrite edges: remap from/to through canonical map; unknown ids pass through unchanged.
    let model_edges: Vec<Edge> = edges
        .into_iter()
        .map(|e| {
            let from = canonical.get(&e.from).cloned().unwrap_or(e.from);
            let to = canonical.get(&e.to).cloned().unwrap_or(e.to);
            let p = prov(&e.source_path, e.range.clone(), e.confidence);
            Edge {
                from,
                to,
                edge_type: e.edge_type,
                prov: p,
            }
        })
        .collect();

    g.upsert(ont, &model_nodes, &model_edges)?;
    for (canon_id, v) in &validity_writes {
        g.upsert_validity(canon_id, v)?;
    }
    Ok(ApplyUpsertResult {
        nodes_written: model_nodes.len(),
        edges_written: model_edges.len(),
        merged,
    })
}

/// Label-based reference to an edge (from/to are human-readable labels, not internal ids).
pub struct EdgeRef {
    pub from: String,
    pub edge_type: String,
    pub to: String,
}

/// Delete nodes and/or edges by label. Node deletion also removes all attached edges.
/// Returns (total entries removed, notes about references that matched nothing — so the caller can
/// tell the model what was a no-op instead of a silent "deleted 0").
/// Resolve a node reference that may be either an id (as assigned in `graph_upsert`) or a label,
/// to the matching node ids. An exact id wins; otherwise every node whose normalized label matches.
/// Keeps the CRUD tools consistent — the agent can delete/update a node by the same id it used when
/// creating it (which is what it naturally does) OR by the human label.
fn resolve_node_ref(g: &GraphStore, reference: &str) -> anyhow::Result<Vec<String>> {
    if g.get_node(reference)?.is_some() {
        return Ok(vec![reference.to_string()]);
    }
    let norm = normalize_label(reference);
    Ok(g.all_nodes()?
        .into_iter()
        .filter(|n| normalize_label(&n.label) == norm)
        .map(|n| n.id)
        .collect())
}

/// On a no-match, suggest the closest existing node labels — fuzzy and inflection-tolerant via the
/// BM25 node index (`GraphStore::resolve`), so the caller can fix a typo or rephrase instead of
/// looping on a label that doesn't exist (e.g. deleting an inflected variant of a stored label →
/// "did you mean: «the stored label»").
/// Returns "" when nothing is close. The exact-normalized label is excluded (it would have matched).
fn did_you_mean(g: &GraphStore, reference: &str) -> String {
    let norm = crate::graph::store::normalize_label(reference);
    let mut labels: Vec<String> = Vec::new();
    for id in g.resolve(reference).unwrap_or_default() {
        if let Ok(Some(node)) = g.get_node(&id) {
            if crate::graph::store::normalize_label(&node.label) != norm
                && !labels.contains(&node.label)
            {
                labels.push(node.label);
            }
        }
        if labels.len() == 3 {
            break;
        }
    }
    if labels.is_empty() {
        String::new()
    } else {
        crate::cli_fmt::format_did_you_mean(&labels)
    }
}

/// Resolve an edge endpoint for delete: reasoning node id/label, or a section ref
/// (`path#n` / `path#location`) stored as a raw edge endpoint (MENTIONS targets).
fn resolve_delete_endpoint(g: &GraphStore, reference: &str) -> anyhow::Result<Option<String>> {
    let ids = resolve_node_ref(g, reference)?;
    if let Some(id) = ids.into_iter().next() {
        return Ok(Some(id));
    }
    if reference.contains('#') {
        return Ok(Some(reference.to_string()));
    }
    Ok(None)
}

pub fn apply_delete(
    g: &GraphStore,
    node_refs: Vec<String>,
    edges: Vec<EdgeRef>,
) -> anyhow::Result<(usize, Vec<String>)> {
    let mut total = 0;
    let mut notes: Vec<String> = Vec::new();

    // Delete every node matching each reference (id or label); note refs that matched nothing.
    for reference in &node_refs {
        let ids = resolve_node_ref(g, reference)?;
        if ids.is_empty() {
            notes.push(format!(
                "node \"{reference}\" matched nothing{}",
                did_you_mean(g, reference)
            ));
        }
        for id in ids {
            total += g.delete_node(&id)?;
        }
    }

    // Delete individual edges by (id-or-label / section-ref) endpoint pairs.
    for er in &edges {
        let from = resolve_delete_endpoint(g, &er.from)?;
        let to = resolve_delete_endpoint(g, &er.to)?;
        match (&from, &to) {
            (Some(f), Some(t)) => {
                let n = g.delete_edge(f, &er.edge_type, t)?;
                total += n;
                if n == 0 {
                    notes.push(format!(
                        "edge {} -{}-> {} matched no existing edge",
                        er.from, er.edge_type, er.to
                    ));
                }
            }
            _ => {
                let mut hint = String::new();
                if from.is_none() {
                    hint.push_str(&did_you_mean(g, &er.from));
                }
                if to.is_none() {
                    hint.push_str(&did_you_mean(g, &er.to));
                }
                notes.push(format!(
                    "edge {} -{}-> {}: an endpoint matched no node{hint}",
                    er.from, er.edge_type, er.to
                ));
            }
        }
    }

    Ok((total, notes))
}

/// Spec for an in-place node edit: change label and/or type while keeping the id and all edges.
pub struct NodeUpdate {
    pub label: String,
    pub new_label: Option<String>,
    pub new_type: Option<String>,
}

/// Rename and/or retype nodes in place, identified by id (as used in `graph_upsert`) or current
/// label. Skips references that resolve to nothing. Returns the total number of rows updated.
pub fn apply_update(
    g: &GraphStore,
    nodes: Vec<NodeUpdate>,
) -> anyhow::Result<(usize, Vec<String>)> {
    let mut total = 0;
    let mut notes: Vec<String> = Vec::new();
    for u in nodes {
        let ids = resolve_node_ref(g, &u.label)?;
        // Only rename when the reference is UNAMBIGUOUS. Renaming several distinct nodes that merely
        // share a label (e.g. same label, different node_type) would give them all the new label and
        // corrupt label identity — skip those (and SAY SO) rather than mangle the graph.
        match ids.len() {
            0 => notes.push(format!("node \"{}\" matched nothing — nothing renamed{}", u.label, did_you_mean(g, &u.label))),
            1 => total += g.update_node(&ids[0], u.new_label.as_deref(), u.new_type.as_deref())?,
            n => notes.push(format!(
                "node \"{}\" is ambiguous ({n} nodes share that label) — rename skipped to avoid corrupting identity",
                u.label
            )),
        }
    }
    Ok((total, notes))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONT: &str = r#"
[entities.Organization]
props = ["name"]
[relations.PARTY_TO]
from = ["Organization"]
to = ["Document"]
[validation]
strict = true
"#;

    const DEDUP_ONT: &str = r#"
[entities.Symptom]
props = []
[entities.Resolution]
props = []
[relations.RESOLVED_BY]
from = ["Symptom"]
to = ["Resolution"]
[validation]
strict = true
"#;

    fn node(id: &str, ty: &str, label: &str, src: &str) -> NodeSpec {
        NodeSpec {
            id: id.into(),
            node_type: ty.into(),
            label: label.into(),
            aliases: vec![],
            source_path: src.into(),
            range: None,
            confidence: None,
            valid_from: None,
            valid_to: None,
        }
    }

    fn node_with_validity(
        id: &str,
        ty: &str,
        label: &str,
        src: &str,
        valid_from: Option<&str>,
        valid_to: Option<&str>,
    ) -> NodeSpec {
        NodeSpec {
            valid_from: valid_from.map(String::from),
            valid_to: valid_to.map(String::from),
            ..node(id, ty, label, src)
        }
    }

    fn edge_spec(from: &str, to: &str, edge_type: &str, src: &str) -> EdgeSpec {
        EdgeSpec {
            from: from.into(),
            to: to.into(),
            edge_type: edge_type.into(),
            source_path: src.into(),
            range: None,
            confidence: None,
        }
    }

    #[test]
    fn applies_validated_agent_nodes_with_provenance() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(ONT).unwrap();
        let nodes = vec![
            node("org:acme", "Organization", "Acme", "contract.docx"),
            node(
                "contract.docx",
                "Document",
                "contract.docx",
                "contract.docx",
            ),
        ];
        let edges = vec![EdgeSpec {
            from: "org:acme".into(),
            to: "contract.docx".into(),
            edge_type: "PARTY_TO".into(),
            source_path: "contract.docx".into(),
            range: None,
            confidence: Some(0.9),
        }];
        let r = apply_upsert(&g, &ont, nodes, edges, 123, dir.path()).unwrap();
        assert_eq!((r.nodes_written, r.edges_written), (2, 1));
        assert_eq!(g.node_count().unwrap(), 2);
        let org = g.get_node("org:acme").unwrap().unwrap();
        assert_eq!(org.prov.origin, "agent");
        assert_eq!(org.prov.created_at, 123);
    }

    #[test]
    fn rejects_undeclared_type_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(ONT).unwrap();
        let nodes = vec![node("x", "Alien", "x", "d.docx")];
        assert!(apply_upsert(&g, &ont, nodes, vec![], 1, dir.path()).is_err());
        assert_eq!(g.node_count().unwrap(), 0);
    }

    #[test]
    fn dedup_merges_same_label_nodes() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();

        // First upsert: a Symptom + Resolution + RESOLVED_BY
        let nodes1 = vec![
            node(
                "sym:fibas-loss-1",
                "Symptom",
                "Fieldbus connection loss",
                "case1.docx",
            ),
            node(
                "res:restart-1",
                "Resolution",
                "Module restart",
                "case1.docx",
            ),
        ];
        let edges1 = vec![edge_spec(
            "sym:fibas-loss-1",
            "res:restart-1",
            "RESOLVED_BY",
            "case1.docx",
        )];
        apply_upsert(&g, &ont, nodes1, edges1, 1, dir.path()).unwrap();

        // Second upsert: DIFFERENT id but SAME label (different case + extra space) → dedup
        let nodes2 = vec![
            node(
                "sym:fibas-loss-2",
                "Symptom",
                "fieldbus  connection loss",
                "case2.docx",
            ),
            node(
                "res:check-cable-2",
                "Resolution",
                "Cable check",
                "case2.docx",
            ),
        ];
        let edges2 = vec![edge_spec(
            "sym:fibas-loss-2",
            "res:check-cable-2",
            "RESOLVED_BY",
            "case2.docx",
        )];
        apply_upsert(&g, &ont, nodes2, edges2, 2, dir.path()).unwrap();

        // Only 1 Symptom node (deduped — first id wins)
        let all = g.all_nodes().unwrap();
        let symptoms: Vec<_> = all.iter().filter(|n| n.node_type == "Symptom").collect();
        assert_eq!(symptoms.len(), 1, "expected exactly 1 Symptom after dedup");
        let symptom_id = &symptoms[0].id;
        assert_eq!(symptom_id, "sym:fibas-loss-1");

        // The second RESOLVED_BY edge must have been rewritten to originate from the first Symptom's id
        let out = g.outgoing(symptom_id).unwrap();
        assert_eq!(
            out.len(),
            2,
            "expected 2 outgoing edges from the deduplicated Symptom"
        );
        let has_check_cable = out
            .iter()
            .any(|e| e.to == "res:check-cable-2" && e.edge_type == "RESOLVED_BY");
        assert!(
            has_check_cable,
            "second RESOLVED_BY edge should point from first symptom id to res:check-cable-2"
        );
    }

    #[test]
    fn apply_delete_removes_node_and_edges() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();

        // Build a small Symptom→Resolution graph
        let nodes = vec![
            node("sym:test", "Symptom", "Test symptom", "test.docx"),
            node("res:test", "Resolution", "Test resolution", "test.docx"),
        ];
        let edges = vec![edge_spec(
            "sym:test",
            "res:test",
            "RESOLVED_BY",
            "test.docx",
        )];
        apply_upsert(&g, &ont, nodes, edges, 1, dir.path()).unwrap();

        assert_eq!(g.node_count().unwrap(), 2);
        assert_eq!(g.edge_count().unwrap(), 1);

        // Delete the Symptom by label
        apply_delete(&g, vec!["Test symptom".into()], vec![]).unwrap();

        // Symptom node is gone
        let all = g.all_nodes().unwrap();
        assert!(
            all.iter().all(|n| n.node_type != "Symptom"),
            "Symptom node should be deleted"
        );

        // Its RESOLVED_BY edge is also gone
        let out = g.outgoing("sym:test").unwrap();
        assert!(
            out.is_empty(),
            "Edges attached to deleted Symptom should be removed"
        );
    }

    #[test]
    fn apply_delete_accepts_id_not_just_label() {
        // The agent assigns ids in graph_upsert and naturally passes the ID to graph_delete.
        // This used to silently no-op (matched only by label) and the agent retried forever.
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        apply_upsert(
            &g,
            &ont,
            vec![node("sym:test", "Symptom", "Test symptom", "test.docx")],
            vec![],
            1,
            dir.path(),
        )
        .unwrap();

        let (removed, _) = apply_delete(&g, vec!["sym:test".into()], vec![]).unwrap();
        assert!(
            removed > 0,
            "delete-by-id must remove the node (the bug returned 0)"
        );
        assert!(
            g.get_node("sym:test").unwrap().is_none(),
            "node deleted by id"
        );
    }

    #[test]
    fn apply_delete_no_match_suggests_closest_label() {
        // The model deletes an inflected/typo'd label that doesn't exact-match the stored one
        // (the agent kept retrying an inflected form of the stored label). The note must point it at
        // the real label so it can fix the call instead of looping. Suggestion comes from the
        // inflection-tolerant BM25 resolve. Russian test data is intentional: it exercises the
        // Cyrillic stemming path (the inflections differ only in Russian endings).
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        apply_upsert(
            &g,
            &ont,
            vec![node(
                "sym:t",
                "Symptom",
                "Test pump symptom",
                "t.docx",
            )],
            vec![],
            1,
            dir.path(),
        )
        .unwrap();

        let (removed, notes) =
            apply_delete(&g, vec!["Test pump symptoms".into()], vec![]).unwrap();
        assert_eq!(removed, 0, "an inflected variant must not match exactly");
        assert_eq!(notes.len(), 1);
        assert!(notes[0].contains("matched nothing"), "note: {}", notes[0]);
        assert!(
            notes[0].contains("did you mean"),
            "a no-match must suggest a fix: {}",
            notes[0]
        );
        assert!(
            notes[0].contains("Test pump symptom"),
            "must name the close existing label: {}",
            notes[0]
        );
    }

    #[test]
    fn apply_upsert_authors_node_validity_for_bounded_node() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        let nodes = vec![node_with_validity(
            "sym:bounded",
            "Symptom",
            "Bounded symptom",
            "test.docx",
            Some("2020-01-01"),
            Some("2020-12-31"),
        )];
        apply_upsert(&g, &ont, nodes, vec![], 1, dir.path()).unwrap();

        let v = g
            .validity_for("sym:bounded")
            .unwrap()
            .expect("a node that supplied valid_from/valid_to gets a node_validity row");
        assert_eq!(v.valid_from_raw.as_deref(), Some("2020-01-01"));
        assert_eq!(v.valid_to_raw.as_deref(), Some("2020-12-31"));
        assert!(v.valid_from.is_some(), "normalized valid_from is stored");
        assert!(v.valid_to.is_some(), "normalized valid_to is stored");
    }

    #[test]
    fn apply_upsert_no_validity_row_when_node_has_no_bound() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        let nodes = vec![node("sym:unbounded", "Symptom", "Unbounded symptom", "test.docx")];
        apply_upsert(&g, &ont, nodes, vec![], 1, dir.path()).unwrap();

        assert!(
            g.validity_for("sym:unbounded").unwrap().is_none(),
            "a node with no valid_from/valid_to must get no node_validity row"
        );
    }

    #[test]
    fn apply_upsert_validity_attaches_to_canonical_id_after_merge() {
        // Two upserts, same label+type — dedup converges the SECOND node into the FIRST's id.
        // Its validity must land on that canonical id (the id it's actually stored under), not
        // its own pre-merge id (which nothing is stored at).
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();

        apply_upsert(
            &g,
            &ont,
            vec![node("sym:first", "Symptom", "Recurring symptom", "case1.docx")],
            vec![],
            1,
            dir.path(),
        )
        .unwrap();

        // Second upsert reuses a DIFFERENT id but the SAME normalized label → merges into sym:first.
        let nodes2 = vec![node_with_validity(
            "sym:second",
            "Symptom",
            "recurring  symptom",
            "case2.docx",
            Some("2021-06"),
            None,
        )];
        let r = apply_upsert(&g, &ont, nodes2, vec![], 2, dir.path()).unwrap();
        assert_eq!(r.merged, vec![("sym:second".to_string(), "sym:first".to_string())]);

        assert!(
            g.validity_for("sym:second").unwrap().is_none(),
            "no row under the pre-merge id"
        );
        let v = g
            .validity_for("sym:first")
            .unwrap()
            .expect("validity lands on the canonical (merged-into) id");
        assert_eq!(v.valid_from_raw.as_deref(), Some("2021-06"));
        assert!(v.valid_to.is_none());
    }

    #[test]
    fn apply_upsert_rejects_unparseable_validity_bound() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        let ont = Ontology::parse(DEDUP_ONT).unwrap();
        let nodes = vec![node_with_validity(
            "sym:bad",
            "Symptom",
            "Bad date symptom",
            "test.docx",
            Some("not-a-date"),
            None,
        )];
        assert!(apply_upsert(&g, &ont, nodes, vec![], 1, dir.path()).is_err());
        assert_eq!(g.node_count().unwrap(), 0, "the whole batch is dropped, nothing written");
    }
}
