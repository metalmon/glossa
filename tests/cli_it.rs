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

#[test]
fn search_then_read_by_number() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("note.md"), b"# Title\nhello world here\n").unwrap();

    // pretty search numbers the hit
    Command::cargo_bin("kb").unwrap()
        .current_dir(dir.path())
        .args(["search", "hello", "--format", "pretty"])
        .assert()
        .success()
        .stdout(contains("[1]").and(contains("note.md")));

    // read by number resolves the recorded hit and prints its text
    Command::cargo_bin("kb").unwrap()
        .current_dir(dir.path())
        .args(["read", "1"])
        .assert()
        .success()
        .stdout(contains("hello world here"));
}

#[test]
fn zero_hit_search_preserves_last_search() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("note.md"), b"# Title\nhello world here\n").unwrap();
    // first search records a hit
    Command::cargo_bin("kb").unwrap().current_dir(dir.path())
        .args(["search", "hello"]).assert().success();
    // a search with no matches must NOT clobber the recorded hit
    Command::cargo_bin("kb").unwrap().current_dir(dir.path())
        .args(["search", "zzznomatchxyz"]).assert().success();
    // read 1 still resolves the earlier hit
    Command::cargo_bin("kb").unwrap().current_dir(dir.path())
        .args(["read", "1"]).assert().success()
        .stdout(predicates::str::contains("hello world here"));
}
