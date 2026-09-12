use assert_cmd::Command;
use glossa::graph::store::{Edge, GraphStore, Node, Provenance};
use predicates::str::contains;
use std::fs;

fn prov(source_path: &str) -> Provenance {
    Provenance {
        source_path: source_path.into(),
        range: None,
        file_sig: None,
        origin: "agent".into(),
        confidence: 0.9,
        created_at: 1,
    }
}

#[test]
fn graph_stats_and_neighbors_after_index() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), "# Intro\nhello\n## Body\nworld\n").unwrap();

    Command::cargo_bin("kb")
        .unwrap()
        .args(["index", dir.path().to_str().unwrap()])
        .assert()
        .success();

    Command::cargo_bin("kb")
        .unwrap()
        .args(["graph", "stats", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(contains("nodes:"));

    // The Document node id is the corpus-relative doc key ("a.md"), not an absolute path — its
    // CONTAINS neighbors are sections. (Doc keys are corpus-root-relative since the path-canon work.)
    let doc_id = "a.md".to_string();
    Command::cargo_bin("kb")
        .unwrap()
        .args([
            "graph",
            "neighbors",
            &doc_id,
            dir.path().to_str().unwrap(),
            "--depth",
            "1",
        ])
        .assert()
        .success()
        .stdout(contains("Intro"));
}

/// `graph stats` output ends with the CLI output contract's summary block: type/community
/// detail first, then a final `nodes: N  edges: N  ...` line (via `cli_fmt::summary_string`).
#[test]
fn graph_stats_ends_with_summary_block() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), "# Intro\nhello\n## Body\nworld\n").unwrap();

    Command::cargo_bin("kb")
        .unwrap()
        .args(["index", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let out = Command::cargo_bin("kb")
        .unwrap()
        .args(["graph", "stats", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let s = String::from_utf8(out).unwrap();

    // Detail (the domain-independent type breakdown) prints before the trailing summary.
    let detail_pos = s.find("by type").expect("type-count detail line present");
    // The summary is the LAST non-empty content: a separator rule immediately above the final
    // `nodes: N  edges: N ...` pair line, and nothing meaningful follows it.
    let sep_pos = s.rfind('\u{2500}').expect("summary separator rule present");
    let nodes_pos = s.rfind("nodes:").expect("summary carries nodes:");
    assert!(
        detail_pos < sep_pos && sep_pos < nodes_pos,
        "detail must precede the trailing summary block:\n{s}"
    );
    assert!(s.contains("edges:"), "summary carries edges:\n{s}");
    assert!(s.trim_end().ends_with(|c: char| c.is_ascii_digit()));
}

/// `graph doctor` on a clean, freshly indexed corpus ends with the trailing summary block
/// (`ungrounded: 0  ... unverifiable: 0`), with the doubt-bucket detail above it.
#[test]
fn graph_doctor_ends_with_summary_block() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), "# Intro\nhello\n").unwrap();

    Command::cargo_bin("kb")
        .unwrap()
        .args(["index", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let out = Command::cargo_bin("kb")
        .unwrap()
        .args(["graph", "doctor", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let s = String::from_utf8(out).unwrap();

    let detail_pos = s.find("ungrounded:").expect("ungrounded bucket detail present");
    let sep_pos = s.rfind('\u{2500}').expect("summary separator rule present");
    let unverifiable_pos = s.rfind("unverifiable:").expect("summary carries unverifiable:");
    assert!(
        detail_pos < sep_pos && sep_pos < unverifiable_pos,
        "doubt-bucket detail must precede the trailing summary block:\n{s}"
    );
    assert!(s.contains("relinkable:"), "summary carries relinkable:\n{s}");
    assert!(s.contains("ambiguous:"), "summary carries ambiguous:\n{s}");
}

/// `graph glossary`'s trailing `matches:` count must reflect the SAME filtered set the body
/// renders — not an independent unfiltered re-query. Regression for a bug where `--scope`
/// narrowed what's printed but the summary still counted the unfiltered total. Fixture mirrors
/// the `tools::mod` unit test `glossary_scope_narrows_grounded_entries_to_owning_document`: two
/// `Fact` nodes sharing one label, each grounded (MENTIONS) to a different document's section, so
/// an unscoped lookup surfaces both and a `--scope` to one document surfaces only its own.
#[test]
fn graph_glossary_summary_matches_reflects_scope_filtered_count() {
    let dir = tempfile::tempdir().unwrap();
    let g = GraphStore::open(dir.path()).unwrap();
    g.put_node(&Node {
        id: "sec:docA".into(),
        node_type: "Section".into(),
        label: "Heading A".into(),
        aliases: vec![],
        prov: prov("docA.md"),
    })
    .unwrap();
    g.put_node(&Node {
        id: "sec:docB".into(),
        node_type: "Section".into(),
        label: "Heading B".into(),
        aliases: vec![],
        prov: prov("docB.md"),
    })
    .unwrap();
    g.put_node(&Node {
        id: "fact:a".into(),
        node_type: "Fact".into(),
        label: "Shared term".into(),
        aliases: vec![],
        prov: prov("docA.md"),
    })
    .unwrap();
    g.put_node(&Node {
        id: "fact:b".into(),
        node_type: "Fact".into(),
        label: "Shared term".into(),
        aliases: vec![],
        prov: prov("docB.md"),
    })
    .unwrap();
    g.put_edge(&Edge {
        from: "fact:a".into(),
        to: "sec:docA".into(),
        edge_type: "MENTIONS".into(),
        prov: prov("docA.md"),
    })
    .unwrap();
    g.put_edge(&Edge {
        from: "fact:b".into(),
        to: "sec:docB".into(),
        edge_type: "MENTIONS".into(),
        prov: prov("docB.md"),
    })
    .unwrap();

    // Unscoped: both facts are rendered — the summary's `matches:` must count both.
    let out = Command::cargo_bin("kb")
        .unwrap()
        .args(["graph", "glossary", "Shared term", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("fact:a") && s.contains("fact:b"), "{s}");
    assert!(s.contains("matches: 2"), "unscoped summary must count both shown matches:\n{s}");

    // `--scope docA.md`: only `fact:a` is rendered — the summary's `matches:` must drop to 1,
    // matching what's actually printed above it, not the unfiltered `resolve()` total of 2.
    let out = Command::cargo_bin("kb")
        .unwrap()
        .args([
            "graph",
            "glossary",
            "Shared term",
            dir.path().to_str().unwrap(),
            "--scope",
            "docA.md",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("fact:a") && !s.contains("fact:b"), "{s}");
    assert!(
        s.contains("matches: 1"),
        "scoped summary must match the filtered/shown count, not the unfiltered resolve():\n{s}"
    );
}
