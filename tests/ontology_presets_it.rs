use assert_cmd::Command;
use predicates::str::contains;
use std::fs;

#[test]
fn index_ontology_materializes_then_indexes() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), b"# Doc\nhello\n").unwrap();

    Command::cargo_bin("kb")
        .unwrap()
        .args(["index", "--ontology", "support", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(contains("indexed"));

    let onto = fs::read_to_string(dir.path().join(".glossa").join("ontology.toml")).unwrap();
    assert!(onto.contains("Symptom"));
}

#[test]
fn index_ontology_keeps_existing_file() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), b"# Doc\nhello\n").unwrap();
    fs::create_dir_all(dir.path().join(".glossa")).unwrap();
    fs::write(dir.path().join(".glossa").join("ontology.toml"), b"# mine\n[entities.X]\nprops=[]\n").unwrap();

    Command::cargo_bin("kb")
        .unwrap()
        .args(["index", "--ontology", "compliance", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stderr(contains("keeping it"));

    let onto = fs::read_to_string(dir.path().join(".glossa").join("ontology.toml")).unwrap();
    assert!(onto.contains("# mine")); // untouched
}
