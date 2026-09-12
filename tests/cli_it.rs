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

/// The last non-empty lines of `s`, in original order (helper for summary-block assertions).
fn last_nonempty_lines(s: &str, n: usize) -> Vec<&str> {
    let mut rev: Vec<&str> = s.lines().rev().filter(|l| !l.trim().is_empty()).take(n).collect();
    rev.reverse();
    rev
}

#[test]
fn search_pretty_ends_with_summary_block() {
    // CLI output contract: `search --format pretty` (the interactive/pretty path) must end its
    // stdout with the shared `cli_fmt::summary` block — a dim `─` rule, then `results: N`.
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), b"# Intro\nthe cat sat\n").unwrap();

    let assert = Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir.path())
        .args(["search", "--format", "pretty", "cat"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&assert.get_output().stdout).to_string();
    let tail = last_nonempty_lines(&stdout, 2);
    assert_eq!(tail.len(), 2, "expected separator + summary line, got:\n{stdout}");
    assert!(
        tail[0].chars().all(|c| c == '\u{2500}'),
        "line above summary must be the `\u{2500}` rule: {:?}",
        tail[0]
    );
    assert!(
        tail[1].contains("results:"),
        "last stdout line must be the summary: {:?}",
        tail[1]
    );
}

#[test]
fn search_rg_format_has_no_summary_block() {
    // The rg-style output mode is a grep-compatibility contract: piping `kb search --format rg`
    // must never gain a trailing summary line, or downstream `| wc -l` / `| awk` breaks.
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), b"# Intro\nthe cat sat\n").unwrap();

    let assert = Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir.path())
        .args(["search", "--format", "rg", "cat"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&assert.get_output().stdout).to_string();
    assert!(
        !stdout.contains("results:"),
        "rg-format stdout must stay grep-compatible (no summary block):\n{stdout}"
    );
}

#[test]
fn grep_stdout_stays_pipe_compatible_no_summary() {
    // `grep` has no `--format` flag; its pretty-vs-pipe gate is `stdout_is_tty()`, which is always
    // false under assert_cmd's captured pipe — exactly the case that must stay grep-compatible
    // (no trailing summary line to break `kb grep ... | ...` pipelines).
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), b"# H\nneedle here\n").unwrap();

    let assert = Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir.path())
        .args(["grep", "needle"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&assert.get_output().stdout).to_string();
    assert!(stdout.contains("needle"), "grep must still find the match:\n{stdout}");
    assert!(
        !stdout.contains("matches:") && !stdout.contains("files:"),
        "piped grep stdout must stay rg-compatible (no summary block):\n{stdout}"
    );
}

#[test]
fn glob_ends_with_summary_block() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), b"# H\nbody one\n").unwrap();
    fs::write(dir.path().join("b.md"), b"# H2\nbody two\n").unwrap();

    let assert = Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir.path())
        .args(["glob", "*.md"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&assert.get_output().stdout).to_string();
    let tail = last_nonempty_lines(&stdout, 2);
    assert_eq!(tail.len(), 2, "expected separator + summary line, got:\n{stdout}");
    assert!(tail[0].chars().all(|c| c == '\u{2500}'), "separator: {:?}", tail[0]);
    assert!(tail[1].contains("docs: 2"), "last stdout line: {:?}", tail[1]);
}

#[test]
fn index_ends_with_summary_block() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), b"# Title\nbody text\n").unwrap();

    let assert = Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir.path())
        .args(["index"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&assert.get_output().stdout).to_string();
    let tail = last_nonempty_lines(&stdout, 2);
    assert_eq!(tail.len(), 2, "expected separator + summary line, got:\n{stdout}");
    assert!(tail[0].chars().all(|c| c == '\u{2500}'), "separator: {:?}", tail[0]);
    assert!(
        tail[1].contains("added:") && tail[1].contains("elapsed:"),
        "last stdout line: {:?}",
        tail[1]
    );
}

#[test]
fn index_force_folds_generalize_counts_into_the_one_final_summary() {
    // Regression: `--force` used to print its own trailing "generalized: ..." line AFTER the
    // index summary, violating "exactly one summary block, last". Its counts must fold into the
    // SAME final summary instead.
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), b"# Title\nbody text\n").unwrap();

    let assert = Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir.path())
        .args(["index", "--force"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&assert.get_output().stdout).to_string();
    assert!(
        !stdout.contains("generalized:"),
        "the old separate generalize line must be folded into the summary:\n{stdout}"
    );
    let tail = last_nonempty_lines(&stdout, 2);
    assert_eq!(tail.len(), 2, "expected separator + summary line, got:\n{stdout}");
    assert!(
        tail[1].contains("added:") && tail[1].contains("elapsed:"),
        "last stdout line: {:?}",
        tail[1]
    );
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
