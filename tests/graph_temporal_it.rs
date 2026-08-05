//! Integration tests for Task 5: `--as-of` on the CLI graph reads + validity/status in `graph node`.
//!
//! The fixture is built directly against `GraphStore` (bypassing the ontology-validated `upsert`
//! via `put_node`/`put_edge`, like the store's own `upsert_tests`), since we only need a couple of
//! nodes with a validity row — no document indexing required for `graph ls`/`graph node`.

use assert_cmd::Command;
use glossa::graph::store::{Edge, GraphStore, Node, NodeValidity, Provenance};
use glossa::graph::temporal;
use predicates::prelude::PredicateBooleanExt;
use predicates::str::contains;

fn prov() -> Provenance {
    Provenance {
        source_path: "fixture.md".into(),
        range: None,
        file_sig: None,
        origin: "curated".into(),
        confidence: 1.0,
        created_at: 0,
    }
}

/// Builds a `.glossa` graph with one `Record` node valid 2020..2023 (id `res:the-id`) and one
/// timeless `Record` node (no validity row at all, id `res:timeless`) — the latter checks that
/// nodes without a validity row are always shown regardless of `--as-of`.
fn build_fixture(dir: &std::path::Path) {
    let g = GraphStore::open(dir).unwrap();
    g.put_node(&Node {
        id: "res:the-id".into(),
        node_type: "Record".into(),
        label: "the record".into(),
        aliases: vec![],
        prov: prov(),
    })
    .unwrap();
    g.upsert_validity(
        "res:the-id",
        &NodeValidity {
            valid_from: Some(temporal::normalize_from("2020").unwrap()),
            valid_to: Some(temporal::normalize_to("2023").unwrap()),
            valid_from_raw: Some("2020".into()),
            valid_to_raw: Some("2023".into()),
        },
    )
    .unwrap();
    g.put_node(&Node {
        id: "res:timeless".into(),
        node_type: "Record".into(),
        label: "timeless record".into(),
        aliases: vec![],
        prov: prov(),
    })
    .unwrap();
    // An edge from the timeless node into the temporal one, to exercise edge-endpoint filtering
    // in `graph dump` / `graph node`.
    g.put_edge(&Edge {
        from: "res:timeless".into(),
        to: "res:the-id".into(),
        edge_type: "RELATED".into(),
        prov: prov(),
    })
    .unwrap();
}

#[test]
fn graph_node_respects_as_of_and_shows_status() {
    let dir = tempfile::tempdir().unwrap();
    build_fixture(dir.path());

    // node hidden as-of 2024 (outside its 2020..2023 validity)
    Command::cargo_bin("kb")
        .unwrap()
        .args([
            "graph",
            "ls",
            "-t",
            "Record",
            "--as-of",
            "2024-01-01",
            dir.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(contains("res:the-id").not());

    // node shown as-of 2022 (inside its validity)
    Command::cargo_bin("kb")
        .unwrap()
        .args([
            "graph",
            "ls",
            "-t",
            "Record",
            "--as-of",
            "2022-01-01",
            dir.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(contains("res:the-id"));

    // the timeless node has no validity row — always shown, even far in the future.
    Command::cargo_bin("kb")
        .unwrap()
        .args([
            "graph",
            "ls",
            "-t",
            "Record",
            "--as-of",
            "2099-01-01",
            dir.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(contains("res:timeless"));

    // graph node shows interval + status against --now
    Command::cargo_bin("kb")
        .unwrap()
        .args([
            "graph",
            "node",
            "res:the-id",
            "--now",
            "2024-06-01",
            dir.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(contains("expired").and(contains("2020")).and(contains("2023")));

    // ...and "current" when the reference instant falls inside the interval.
    Command::cargo_bin("kb")
        .unwrap()
        .args([
            "graph",
            "node",
            "res:the-id",
            "--now",
            "2022-06-01",
            dir.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(contains("current"));

    // --as-of hides the node from `graph node` itself (treated as not found).
    Command::cargo_bin("kb")
        .unwrap()
        .args([
            "graph",
            "node",
            "res:the-id",
            "--as-of",
            "2024-01-01",
            dir.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(contains("node not found"));

    // default (no --as-of, no --now) behavior is unchanged: the node is shown, no filtering.
    Command::cargo_bin("kb")
        .unwrap()
        .args(["graph", "node", "res:the-id", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(contains("node not found").not());
}

#[test]
fn graph_dump_and_near_drop_edges_to_filtered_endpoints() {
    let dir = tempfile::tempdir().unwrap();
    build_fixture(dir.path());

    // dump --as-of 2024 (outside res:the-id's validity): the RELATED edge from res:timeless
    // into res:the-id must not be printed since its endpoint is filtered out.
    Command::cargo_bin("kb")
        .unwrap()
        .args([
            "graph",
            "dump",
            "--as-of",
            "2024-01-01",
            dir.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(contains("res:the-id").not())
        .stdout(contains("RELATED").not());

    // near from res:timeless --as-of 2024: res:the-id is unreachable (filtered).
    Command::cargo_bin("kb")
        .unwrap()
        .args([
            "graph",
            "near",
            "res:timeless",
            "--as-of",
            "2024-01-01",
            dir.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(contains("res:the-id").not());

    // near from res:timeless --as-of 2022: res:the-id is reachable.
    Command::cargo_bin("kb")
        .unwrap()
        .args([
            "graph",
            "near",
            "res:timeless",
            "--as-of",
            "2022-01-01",
            dir.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(contains("res:the-id"));
}

/// `graph dump -f json` (and by extension dot/graphml/html) shares the SAME node-set filter as
/// the text format — this covers the non-text path specifically, since `Dump`'s `--as-of` filters
/// `graph::io::collect()`'s `GraphExport` before formatting, not the rendered text.
#[test]
fn graph_dump_json_omits_out_of_window_node() {
    let dir = tempfile::tempdir().unwrap();
    build_fixture(dir.path());

    Command::cargo_bin("kb")
        .unwrap()
        .args([
            "graph",
            "dump",
            "-f",
            "json",
            "--as-of",
            "2024-01-01",
            dir.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(contains("res:the-id").not())
        .stdout(contains("res:timeless"));

    // ...and present as-of a date inside its validity window.
    Command::cargo_bin("kb")
        .unwrap()
        .args([
            "graph",
            "dump",
            "-f",
            "json",
            "--as-of",
            "2022-01-01",
            dir.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(contains("res:the-id"));
}
