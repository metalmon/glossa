//! `kb read <n>` resolving a `--scan` hit must branch on the recorded resolver (`HitSource`), never
//! guess from the shape of the ordinal column — see `src/cli_fmt.rs` (`LastHit`/`nth_record`) and
//! the `kb read <n>` arm in `src/main.rs`.

use assert_cmd::Command;
use predicates::prelude::PredicateBooleanExt;

/// Both live bugs this phase closes, in one run, in a directory with NO index:
/// a scan hit whose location is all digits must read ITS OWN chunk (the old TSV read that
/// location as an ordinal and clamped), and a scan record must resolve with no index present.
#[test]
fn a_scan_hit_reads_its_own_chunk_without_an_index() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("doc.md"),
        "# 2024\nfirst section needle\n\n# Later\nsecond section\n",
    )
    .unwrap();

    Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir.path())
        .args(["search", "--scan", "needle"])
        .assert()
        .success();

    assert!(
        !dir.path().join(".glossa/index").exists(),
        "--scan must not build an index"
    );

    Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir.path())
        .args(["read", "1"])
        .assert()
        .success()
        .stdout(predicates::str::contains("first section needle"))
        .stdout(predicates::str::contains("second section").not());

    // The read itself must not have built an index either — this phase's headline guarantee is
    // that a scan hit resolves WITHOUT one, not just that the preceding search didn't build one.
    assert!(
        !dir.path().join(".glossa/index").exists(),
        "kb read <n> on a --scan hit must not build an index"
    );
}

/// Pins the two Review Focus cases the test above doesn't reach: a recorded scan hit whose
/// ordinal no longer exists in the file (reported, not silently clamped to a different chunk),
/// and an `n` beyond the recorded hits at all (a message, not a panic or an empty print).
#[test]
fn a_changed_file_and_an_out_of_range_n_both_say_so() {
    let dir = tempfile::tempdir().unwrap();
    // Two chunks, both matching "needle", so the scan records TWO hits (ords 1 and 2).
    std::fs::write(
        dir.path().join("doc.md"),
        "# 2024\nfirst section needle\n\n# Later\nsecond section needle\n",
    )
    .unwrap();

    Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir.path())
        .args(["search", "--scan", "needle"])
        .assert()
        .success();

    // The file changes underneath: now only one chunk exists, so the second recorded hit's
    // ordinal (2) no longer has a matching chunk.
    std::fs::write(dir.path().join("doc.md"), "# 2024\nfirst section needle\n").unwrap();

    Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir.path())
        .args(["read", "2"])
        .assert()
        .failure()
        .stderr(predicates::str::contains(
            "the file changed since the search",
        ));

    // n beyond the recorded hits is a message, not a panic or an empty print — and, unlike the
    // changed-file case above, it succeeds: there is no chunk to have changed underneath.
    Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir.path())
        .args(["read", "9"])
        .assert()
        .success()
        .stdout(predicates::str::contains("no result #9"));
}

/// `root.rs` never absolutizes a relative root, so a `--scan` record built against one must still
/// open from a DIFFERENT working directory later — `nth_chunk` opens the recorded path directly,
/// with no index to fall back on. Here the search runs with the relative root `corpus` (cwd =
/// `dir_a`), then `kb read` runs with cwd = an unrelated `dir_b`, pointed at the same corpus only
/// via an absolute `--root`. If the recorded path was kept relative (to `dir_a`), resolving it
/// against `dir_b`'s cwd opens nothing and the read fails.
#[test]
fn a_scan_hit_reads_back_from_a_different_working_directory() {
    let dir_a = tempfile::tempdir().unwrap();
    let corpus = dir_a.path().join("corpus");
    std::fs::create_dir_all(&corpus).unwrap();
    std::fs::write(corpus.join("doc.md"), "# 2024\nfirst section needle\n").unwrap();

    Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir_a.path())
        .args(["search", "--scan", "needle", "corpus"])
        .assert()
        .success();

    let dir_b = tempfile::tempdir().unwrap();
    Command::cargo_bin("kb")
        .unwrap()
        .current_dir(dir_b.path())
        .args(["read", "1", "--root"])
        .arg(&corpus)
        .assert()
        .success()
        .stdout(predicates::str::contains("first section needle"));
}
