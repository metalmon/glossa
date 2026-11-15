use glossa::model::Location;
use glossa::walk::collect_chunks;
use std::fs;

/// The location these fixtures expect: one heading segment, as a PATH. Asserting the shape rather
/// than the rendered string is the point — this is the one test whose subject IS the location, so
/// it has to fail if a heading chain ever flattens into a `Label`.
fn heading(title: &str) -> Location {
    Location::Path(vec![title.to_string()])
}

#[test]
fn collects_chunks_from_markdown_files_in_tree() {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir(dir.path().join("sub")).unwrap();
    fs::write(dir.path().join("a.md"), b"# Title\nhello world\n").unwrap();
    fs::write(dir.path().join("sub/b.md"), b"# Other\nbye\n").unwrap();
    fs::write(dir.path().join("ignore.txt"), b"not indexed\n").unwrap();

    let chunks = collect_chunks(dir.path(), None, true).unwrap();
    // 2 markdown chunks + 1 text chunk (catch-all now indexes .txt files too)
    assert_eq!(chunks.len(), 3);
    assert!(chunks.iter().any(|c| c.location == heading("Title")));
    assert!(chunks.iter().any(|c| c.location == heading("Other")));
}

#[test]
fn glob_filters_paths() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("keep.md"), b"# K\nkeep\n").unwrap();
    fs::write(dir.path().join("skip.md"), b"# S\nskip\n").unwrap();

    let chunks = collect_chunks(dir.path(), Some("**/keep.md"), true).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].location, heading("K"));
}

#[test]
fn respects_gitignore_by_default() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join(".gitignore"), b"secret.md\n").unwrap();
    fs::write(dir.path().join("keep.md"), b"# K\nkeep\n").unwrap();
    fs::write(dir.path().join("secret.md"), b"# S\nsecret\n").unwrap();

    let respected = collect_chunks(dir.path(), None, true).unwrap();
    assert!(respected.iter().any(|c| c.location == heading("K")));
    assert!(
        !respected.iter().any(|c| c.location == heading("S")),
        "gitignored file must be skipped"
    );

    let all = collect_chunks(dir.path(), None, false).unwrap();
    assert!(
        all.iter().any(|c| c.location == heading("S")),
        "--no-ignore indexes everything"
    );
}
