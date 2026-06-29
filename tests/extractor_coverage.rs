use assert_cmd::Command;
use predicates::str::contains;
use std::fs;

/// End-to-end: the ripgrep-style scan (which routes through `extract_file`) must surface
/// unique needles from .txt / .json / .csv / .html files; binary .png must be silently skipped.
#[test]
fn extractor_coverage_text_formats() {
    let dir = tempfile::tempdir().unwrap();

    // Each fixture contains one needle that does not appear in any other file.
    fs::write(dir.path().join("notes.txt"), b"alpha apricot avocado").unwrap();
    fs::write(dir.path().join("data.json"), br#"{"fruit":"bananaberry"}"#).unwrap();
    fs::write(dir.path().join("table.csv"), b"name,score\ncranberry,9\n").unwrap();
    // damsonfruit is inside a tag — the HTML extractor must strip tags and expose it.
    fs::write(
        dir.path().join("page.html"),
        b"<h1>Heading</h1><p>damsonfruit</p>",
    )
    .unwrap();
    // Binary blob: must be silently skipped — indexing must not crash.
    fs::write(
        dir.path().join("blob.png"),
        [0x89u8, b'P', b'N', b'G', 0x00, 0x01, 0x02],
    )
    .unwrap();

    // `index` is optional for the default scan but exercises the index path end-to-end.
    Command::cargo_bin("kb")
        .unwrap()
        .args(["index", dir.path().to_str().unwrap()])
        .assert()
        .success();

    // Each needle is unique to exactly one file.  The scan output must contain it.
    for needle in ["apricot", "bananaberry", "cranberry", "damsonfruit"] {
        Command::cargo_bin("kb")
            .unwrap()
            .args(["search", needle, dir.path().to_str().unwrap()])
            .assert()
            .success()
            .stdout(contains(needle));
    }
}
