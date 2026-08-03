use assert_cmd::Command;
use predicates::str::contains;
use std::fs;

#[test]
fn index_then_ranked_search_finds_inflection() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("a.md"),
        "# T\nSigned contracts for supply\n",
    )
    .unwrap();

    // Build the index.
    Command::cargo_bin("kb")
        .unwrap()
        .args(["index", dir.path().to_str().unwrap()])
        .assert()
        .success();

    // Ranked search with a different inflection than the document.
    // Index-ranked search is the default now (`--scan` is the literal-search opt-out).
    Command::cargo_bin("kb")
        .unwrap()
        .args(["search", "contract", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(contains("a.md"));
}
