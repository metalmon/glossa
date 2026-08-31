//! Integration test for Finding 2 of the graph-doctor/staleness review: a heal round-trip
//! through the REAL grounding path — `graph_upsert` → `apply_upsert` → `stat_sig` — instead of
//! a hand-set `file_sig` (which is what every other doctor/hygiene test uses and which hides a
//! producer/consumer path-base mismatch).
//!
//! Guards two things at once:
//! 1. Finding 1's fix — the PRODUCER (`agent::stat_sig`, called from `apply_upsert`'s `prov`
//!    closure) must resolve `source_path` against the corpus root, exactly like the CONSUMERS
//!    (`hygiene::stale_nodes`, `doctor::doctor`, `tools::StaleChecker`) already do. Before the
//!    fix, `stat_sig` stat'd the bare `source_path` against the process CWD, which is not the
//!    temp corpus dir in a test (or the corpus root in the normal MCP-server deployment) — so
//!    the stamped `file_sig` was `None` and staleness could never fire.
//! 2. The "re-grounding clears stale" invariant: re-`graph_upsert`-ing a drifted node's
//!    `MENTIONS` re-stamps `file_sig` against the now-current file, clearing the stale flag.

use glossa::graph::doctor;
use glossa::graph::ontology::Ontology;
use glossa::graph::ops::{graph_upsert, id_for, UpsertEdge, UpsertNode};
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;
use glossa::model::Chunk;

const ONT: &str = r#"
[entities.Resolution]
id_prefix = "res"
props = []
requires_grounding = true
[validation]
strict = true
"#;

fn unode() -> UpsertNode {
    UpsertNode {
        node_type: "Resolution".into(),
        label: "Module restart".into(),
        source_path: "case1.docx".into(),
        aliases: vec![],
        valid_from: None,
        valid_to: None,
    }
}

fn mentions_edge() -> UpsertEdge {
    UpsertEdge {
        from: "Module restart".into(),
        edge_type: "MENTIONS".into(),
        to: "case1.docx#1".into(),
        source_path: "case1.docx".into(),
    }
}

#[test]
fn heal_round_trip_through_real_grounding_path() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    // 1. A REAL source file under the temp corpus root, and enough indexing that graph_upsert
    // can ground a MENTIONS edge to it (canonical_document_path / location_for_ord).
    let doc_path = root.join("case1.docx");
    std::fs::write(&doc_path, b"original content").unwrap();

    let idx = DocIndex::open_or_create(root).unwrap();
    idx.write_chunks(&[Chunk {
        doc_path: "case1.docx".into(),
        location: "S1".into(),
        file_type: "docx".into(),
        text: "stub content".into(),
    }])
    .unwrap();

    let g = GraphStore::open(root).unwrap();
    let ont = Ontology::parse(ONT).unwrap();

    // 2. graph_upsert a grounded node with a MENTIONS to that source — this drives
    // apply_upsert's prov closure, which calls stat_sig(root, "case1.docx") for real (no
    // hand-set file_sig anywhere in this test).
    let out = graph_upsert(
        &idx,
        &g,
        &ont,
        vec![unode()],
        vec![mentions_edge()],
        1,
        "agent",
    );
    assert!(!out.rejected, "{}", out.message);
    assert_eq!(out.nodes, 1);
    assert_eq!(out.edges, 1);

    let node_id = id_for(&ont, "Resolution", "Module restart");
    let stored = g
        .get_node(&node_id)
        .unwrap()
        .expect("node written by graph_upsert");
    assert!(
        stored.prov.file_sig.is_some(),
        "producer must stamp a real file_sig against the corpus root, not the process CWD"
    );

    // Sanity: not stale immediately after grounding.
    let rep0 = doctor::doctor(&g, &ont, root).unwrap();
    assert!(
        !rep0.stale.iter().any(|d| d.id == node_id),
        "freshly grounded node must not be stale: {:?}",
        rep0.stale
    );

    // 3. Drift the source file (different length so the stat sig changes regardless of
    // filesystem mtime resolution).
    std::fs::write(&doc_path, b"drifted, much longer content now").unwrap();

    // 4. The node is now stale, per doctor::doctor's real-root comparison.
    let rep = doctor::doctor(&g, &ont, root).unwrap();
    assert!(
        rep.stale.iter().any(|d| d.id == node_id),
        "node should be stale after its source drifted: {:?}",
        rep.stale
    );

    // 5. Re-graph_upsert the same node's MENTIONS (re-grounds) — apply_upsert stamps a fresh
    // file_sig against the now-current (drifted) file, so it must no longer be stale.
    let out2 = graph_upsert(
        &idx,
        &g,
        &ont,
        vec![unode()],
        vec![mentions_edge()],
        2,
        "agent",
    );
    assert!(!out2.rejected, "{}", out2.message);

    let rep2 = doctor::doctor(&g, &ont, root).unwrap();
    assert!(
        !rep2.stale.iter().any(|d| d.id == node_id),
        "re-grounding must clear stale: {:?}",
        rep2.stale
    );
}
