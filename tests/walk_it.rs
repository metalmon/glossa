use glossa::walk::collect_chunks;
use std::fs;

#[test]
fn collects_chunks_from_markdown_files_in_tree() {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir(dir.path().join("sub")).unwrap();
    fs::write(dir.path().join("a.md"), b"# Title\nhello world\n").unwrap();
    fs::write(dir.path().join("sub/b.md"), b"# Other\nbye\n").unwrap();
    fs::write(dir.path().join("ignore.txt"), b"not indexed\n").unwrap();

    let chunks = collect_chunks(dir.path(), None).unwrap();
    assert_eq!(chunks.len(), 2);
    assert!(chunks.iter().any(|c| c.location == "Title"));
    assert!(chunks.iter().any(|c| c.location == "Other"));
}

#[test]
fn glob_filters_paths() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("keep.md"), b"# K\nkeep\n").unwrap();
    fs::write(dir.path().join("skip.md"), b"# S\nskip\n").unwrap();

    let chunks = collect_chunks(dir.path(), Some("**/keep.md")).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].location, "K");
}
