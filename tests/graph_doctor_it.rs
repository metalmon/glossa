//! Integration tests for Task 4: `kb graph doctor` (report + `--prune-*`) and the moved
//! `generalize --prune-*` flags (removed in Task 3 — must now fail to parse).
//!
//! The fixture mirrors `graph::doctor`'s own unit test (`doctor_reports_three_buckets`):
//! a stale node (source drifted after upsert), an ungrounded node (no live MENTIONS), and
//! an incomplete node (on no complete spine) — built directly against `GraphStore`
//! (bypassing the ontology-validated `upsert`, like `graph_temporal_it.rs` does), with a
//! matching `.glossa/ontology.toml` so the CLI's `Ontology::load_or_default` sees the same
//! ontology the unit test constructs in-process.

use assert_cmd::Command;
use glossa::graph::store::{Edge, GraphStore, Node, Provenance};
use glossa::index::store::file_sig;
use predicates::prelude::PredicateBooleanExt;
use predicates::str::contains;
use std::path::Path;

const ONT: &str = r#"
[entities.Symptom]
props = ["name"]
[entities.Cause]
props = ["name"]
[entities.Resolution]
props = ["name"]
requires_grounding = true
[entities.Section]
props = ["name"]
[relations.CAUSED_BY]
from = ["Symptom"]
to = ["Cause"]
[relations.RESOLVED_BY]
from = ["Cause", "Symptom"]
to = ["Resolution"]
[relations.MENTIONS]
from = ["Symptom", "Resolution"]
to = ["Section"]
[validation]
strict = false
[reasoning]
spines = [{ anchor = "Symptom", relations = ["CAUSED_BY", "RESOLVED_BY"] }]
"#;

fn prov(source_path: &str, file_sig: Option<glossa::index::manifest::FileSig>) -> Provenance {
    Provenance {
        source_path: source_path.into(),
        range: None,
        file_sig,
        origin: "agent".into(),
        confidence: 0.9,
        created_at: 1,
    }
}

fn node(id: &str, ty: &str, label: &str, p: Provenance) -> Node {
    Node {
        id: id.into(),
        node_type: ty.into(),
        label: label.into(),
        aliases: vec![],
        prov: p,
    }
}

fn edge(f: &str, ty: &str, t: &str, p: Provenance) -> Edge {
    Edge {
        from: f.into(),
        to: t.into(),
        edge_type: ty.into(),
        prov: p,
    }
}

/// Builds: `res:a` grounded + on-spine but its source (`doc_a.md`) drifts after upsert → stale.
/// `res:b` on-spine but has no live `MENTIONS` → ungrounded. `cau:orphan` isolated, on no
/// complete spine → incomplete.
fn build_fixture(dir: &Path) {
    std::fs::create_dir_all(dir.join(".glossa")).unwrap();
    std::fs::write(dir.join(".glossa").join("ontology.toml"), ONT).unwrap();

    let g = GraphStore::open(dir).unwrap();

    let doc_a = dir.join("doc_a.md");
    std::fs::write(&doc_a, b"v1").unwrap();
    let sig0 = file_sig(&doc_a).unwrap();

    let nodes = vec![
        node("sym:a", "Symptom", "A", prov("doc_a.md", None)),
        node("cau:a", "Cause", "A cause", prov("doc_a.md", None)),
        node("res:a", "Resolution", "A res", prov("doc_a.md", Some(sig0))),
        node("sec:a", "Section", "A sec", prov("doc_a.md", None)),
        node("sym:b", "Symptom", "B", prov("doc_b.md", None)),
        node("cau:b", "Cause", "B cause", prov("doc_b.md", None)),
        node("res:b", "Resolution", "B res", prov("doc_b.md", None)),
        node("cau:orphan", "Cause", "Orphan", prov("doc_c.md", None)),
    ];
    for n in &nodes {
        g.put_node(n).unwrap();
    }
    let edges = vec![
        edge("sym:a", "CAUSED_BY", "cau:a", prov("doc_a.md", None)),
        edge("cau:a", "RESOLVED_BY", "res:a", prov("doc_a.md", None)),
        edge("res:a", "MENTIONS", "sec:a", prov("doc_a.md", None)),
        edge("sym:b", "CAUSED_BY", "cau:b", prov("doc_b.md", None)),
        edge("cau:b", "RESOLVED_BY", "res:b", prov("doc_b.md", None)),
    ];
    for e in &edges {
        g.put_edge(e).unwrap();
    }

    // Let doc_a drift so res:a's stored sig no longer matches disk → stale.
    std::fs::write(&doc_a, b"v2-longer").unwrap();
}

#[test]
fn doctor_reports_and_prunes() {
    let dir = tempfile::tempdir().unwrap();
    build_fixture(dir.path());

    // report shows all three doubts
    Command::cargo_bin("kb")
        .unwrap()
        .args(["graph", "doctor", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(
            contains("stale")
                .and(contains("ungrounded"))
                .and(contains("incomplete"))
                .and(contains("cau:orphan"))
                .and(contains("res:b"))
                .and(contains("res:a")),
        );

    // --prune-incomplete removes the incomplete node; stale survives
    Command::cargo_bin("kb")
        .unwrap()
        .args([
            "graph",
            "doctor",
            "--prune-incomplete",
            dir.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    let g = GraphStore::open(dir.path()).unwrap();
    assert!(
        g.get_node("cau:orphan").unwrap().is_none(),
        "incomplete node pruned"
    );
    assert!(
        g.get_node("res:a").unwrap().is_some(),
        "stale node must survive prune"
    );

    // a follow-up report no longer lists the pruned incomplete node
    Command::cargo_bin("kb")
        .unwrap()
        .args(["graph", "doctor", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(contains("cau:orphan").not());

    // generalize no longer accepts the moved flag (removed in Task 3)
    Command::cargo_bin("kb")
        .unwrap()
        .args([
            "graph",
            "generalize",
            "--prune-incomplete",
            dir.path().to_str().unwrap(),
        ])
        .assert()
        .failure();
}
