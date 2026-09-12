use assert_cmd::Command;
use predicates::str::contains;
use std::fs;

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
