use assert_cmd::Command;
use predicates::prelude::PredicateBooleanExt;
use predicates::str::contains;
use std::fs;

#[test]
fn kb_search_prints_matching_lines() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), b"# Intro\nthe cat sat\n").unwrap();

    let mut cmd = Command::cargo_bin("kb").unwrap();
    cmd.args(["search", "cat", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(contains("Intro").and(contains("the cat sat")));
}

#[test]
fn kb_search_word_flag_excludes_substring() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), b"# H\ncategory only\n").unwrap();

    let mut cmd = Command::cargo_bin("kb").unwrap();
    cmd.args(["search", "cat", "-w", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicates::str::is_empty());
}
