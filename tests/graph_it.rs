use assert_cmd::Command;
use predicates::str::contains;
use std::fs;

#[test]
fn graph_stats_and_neighbors_after_index() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), "# Intro\nhello\n## Body\nworld\n").unwrap();

    Command::cargo_bin("kb").unwrap()
        .args(["index", dir.path().to_str().unwrap()])
        .assert().success();

    Command::cargo_bin("kb").unwrap()
        .args(["graph", "stats", dir.path().to_str().unwrap()])
        .assert().success()
        .stdout(contains("nodes:"));

    // The Document node id is the corpus-relative doc key ("a.md"), not an absolute path — its
    // CONTAINS neighbors are sections. (Doc keys are corpus-root-relative since the path-canon work.)
    let doc_id = "a.md".to_string();
    Command::cargo_bin("kb").unwrap()
        .args(["graph", "neighbors", &doc_id, dir.path().to_str().unwrap(), "--depth", "1"])
        .assert().success()
        .stdout(contains("Intro"));
}
