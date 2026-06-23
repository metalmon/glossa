use assert_cmd::Command;
use predicates::str::contains;
use std::fs;

#[test]
fn index_then_ranked_search_finds_russian_inflection() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), "# T\nПодписаны договоры на поставку\n").unwrap();

    // Build the index.
    Command::cargo_bin("kb")
        .unwrap()
        .args(["index", dir.path().to_str().unwrap()])
        .assert()
        .success();

    // Ranked search with a different inflection than the document.
    Command::cargo_bin("kb")
        .unwrap()
        .args(["search", "--rank", "договор", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(contains("a.md"));
}
