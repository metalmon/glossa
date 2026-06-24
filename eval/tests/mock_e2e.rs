use assert_cmd::Command;
use predicates::prelude::PredicateBooleanExt;
use predicates::str::contains;
use std::fs;

#[test]
fn mock_run_scores_and_reports() {
    let dir = tempfile::tempdir().unwrap();
    let ds = dir.path().join("ds.json");
    fs::write(&ds, r#"[
      {"_id":"q1","question":"Who?","answer":"Bob Page","context":[["Bob Page",["b1."]]],"supporting_facts":[["Bob Page",0]]},
      {"_id":"q2","question":"What?","answer":"42","context":[["N",["n1."]]],"supporting_facts":[["N",0]]}
    ]"#).unwrap();

    Command::cargo_bin("kb-eval").unwrap()
        .current_dir(dir.path())
        .args(["run", "--dataset", ds.to_str().unwrap(), "--backend", "mock"])
        .assert()
        .success()
        .stdout(contains("backend=mock").and(contains("questions=2")).and(contains("EM=")));

    let wrote = fs::read_dir(dir.path()).unwrap()
        .filter_map(|e| e.ok())
        .any(|e| e.file_name().to_string_lossy().starts_with("eval-mock-"));
    assert!(wrote, "a report JSON should be written");
}
