use assert_cmd::Command;
use predicates::prelude::PredicateBooleanExt;
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

#[test]
fn ontology_list_show_init_suggest() {
    // list
    Command::cargo_bin("kb").unwrap().args(["ontology", "list"])
        .assert().success()
        .stdout(contains("compliance").and(contains("support")));

    // show prints the TOML
    Command::cargo_bin("kb").unwrap().args(["ontology", "show", "support"])
        .assert().success()
        .stdout(contains("Symptom").and(contains("RESOLVED_BY")));

    // show resolves an alias
    Command::cargo_bin("kb").unwrap().args(["ontology", "show", "normocontrol"])
        .assert().success()
        .stdout(contains("NormativeRequirement"));

    // suggest ranks by free text
    Command::cargo_bin("kb").unwrap()
        .args(["ontology", "suggest", "register of personal data and retention"])
        .assert().success()
        .stdout(contains("data-privacy"));

    // init into a temp dir
    let dir = tempfile::tempdir().unwrap();
    Command::cargo_bin("kb").unwrap()
        .args(["ontology", "init", "--template", "compliance", dir.path().to_str().unwrap()])
        .assert().success();
    assert!(std::fs::read_to_string(dir.path().join(".glossa").join("ontology.toml")).unwrap()
        .contains("NormativeRequirement"));
}

#[test]
fn ontology_init_refuses_existing_without_force() {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir_all(dir.path().join(".glossa")).unwrap();
    fs::write(dir.path().join(".glossa").join("ontology.toml"), b"# mine\n").unwrap();

    // without --force → error, file untouched
    Command::cargo_bin("kb").unwrap()
        .args(["ontology", "init", "--template", "support", dir.path().to_str().unwrap()])
        .assert().failure()
        .stderr(contains("--force"));
    assert!(fs::read_to_string(dir.path().join(".glossa").join("ontology.toml")).unwrap().contains("# mine"));

    // with --force → replaced
    Command::cargo_bin("kb").unwrap()
        .args(["ontology", "init", "--template", "support", "--force", dir.path().to_str().unwrap()])
        .assert().success();
    assert!(fs::read_to_string(dir.path().join(".glossa").join("ontology.toml")).unwrap().contains("Symptom"));
}

#[test]
fn ontology_list_family_filter() {
    // --family narrows to that family only
    Command::cargo_bin("kb").unwrap()
        .args(["ontology", "list", "--family", "risk"])
        .assert().success()
        .stdout(contains("risk-register").and(contains("support").not()));
}

#[test]
fn ontology_suggest_empty_and_show_unknown() {
    // a query that overlaps nothing → explicit "no match" line, still exits 0
    Command::cargo_bin("kb").unwrap()
        .args(["ontology", "suggest", "zzzzqqq wwwwxyz"])
        .assert().success()
        .stdout(contains("no preset matched"));

    // show of an unknown name → error suggesting the closest preset
    Command::cargo_bin("kb").unwrap()
        .args(["ontology", "show", "complaince"])
        .assert().failure()
        .stderr(contains("compliance"));
}
