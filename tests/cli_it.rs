use assert_cmd::Command;
use predicates::prelude::PredicateBooleanExt;
use predicates::str::contains;
use std::fs;

#[test]
fn kb_search_prints_matching_lines() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), b"# Intro\nthe cat sat\n").unwrap();

    let mut cmd = Command::cargo_bin("kb").unwrap();
    cmd.args(["search", "--scan", "cat", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(contains("Intro").and(contains("the cat sat")));
}

#[test]
fn kb_search_word_flag_excludes_substring() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), b"# H\ncategory only\n").unwrap();

    let mut cmd = Command::cargo_bin("kb").unwrap();
    cmd.args([
        "search",
        "--scan",
        "cat",
        "-w",
        dir.path().to_str().unwrap(),
    ])
    .assert()
    .success()
    .stdout(predicates::str::is_empty());
}

#[test]
fn search_then_read_by_number() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("note.md"), b"# Title\nhello world here\n").unwrap();

    // pretty search numbers the hit
    Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir.path())
        .args(["search", "--scan", "hello", "--format", "pretty"])
        .assert()
        .success()
        .stdout(contains("#1").and(contains("note.md")));

    // read by number resolves the recorded hit and prints its text
    Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir.path())
        .args(["read", "1"])
        .assert()
        .success()
        .stdout(contains("hello world here"));
}

#[test]
fn kb_cat_dumps_full_file_text_without_index() {
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("doc.md");
    fs::write(&f, b"# Title\nfirst section\n\n# Two\nsecond section\n").unwrap();

    // `cat` prints the whole extracted text (every section), directly from the file.
    Command::cargo_bin("kb")
        .unwrap()
        .args(["cat", f.to_str().unwrap()])
        .assert()
        .success()
        .stdout(contains("first section").and(contains("second section")));
    // ...and it does not build an index (no `.glossa` litter) — a true one-shot read.
    assert!(
        !dir.path().join(".glossa").exists(),
        "cat must not create an index"
    );
}

#[test]
fn kb_cat_missing_file_errors() {
    Command::cargo_bin("kb")
        .unwrap()
        .args(["cat", "does-not-exist.pdf"])
        .assert()
        .failure();
}

#[test]
fn index_file_flag_reindexes_one_document() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), b"# Title\noldtok\n").unwrap();

    // Build the initial index.
    Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir.path())
        .args(["index"])
        .assert()
        .success();

    // Edit the file in place (same length, so size-based change detection alone can't catch it).
    fs::write(dir.path().join("a.md"), b"# Title\nnewtok\n").unwrap();

    // Reindex just that one file.
    Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir.path())
        .args(["index", "--file", "a.md"])
        .assert()
        .success();

    // The updated content is now searchable.
    Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir.path())
        .args(["search", "newtok"])
        .assert()
        .success()
        .stdout(contains("a.md"));
}

#[test]
fn zero_hit_search_preserves_last_search() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("note.md"), b"# Title\nhello world here\n").unwrap();
    // first search records a hit
    Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir.path())
        .args(["search", "--scan", "hello"])
        .assert()
        .success();
    // a search with no matches must NOT clobber the recorded hit
    Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir.path())
        .args(["search", "--scan", "zzznomatchxyz"])
        .assert()
        .success();
    // read 1 still resolves the earlier hit
    Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir.path())
        .args(["read", "1"])
        .assert()
        .success()
        .stdout(predicates::str::contains("hello world here"));
}
