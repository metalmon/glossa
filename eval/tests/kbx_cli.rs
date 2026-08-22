//! CLI-level smoke test for the `kbx` binary's clap/CLI boundary — exercises the real compiled
//! binary via `assert_cmd` rather than calling the internal `scaffold_init` helper directly, so a
//! regression in argument parsing / wiring (not just the underlying logic) would be caught.
//! Hermetic: `kbx init` touches no network, no corpus, no LLM — just the filesystem.

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn kbx_init_scaffolds_workspace_and_refuses_to_overwrite_without_force() {
    let dir = tempfile::tempdir().expect("create tempdir");

    // First `kbx init <dir>`: should succeed and scaffold all workspace files.
    Command::cargo_bin("kbx")
        .expect("find kbx binary")
        .arg("init")
        .arg(dir.path())
        .assert()
        .success();

    for name in ["lab.toml", "answer.md", "judge.md", "dataset.toml"] {
        assert!(
            dir.path().join(name).is_file(),
            "expected {name} to be created by `kbx init`"
        );
    }
    assert!(
        dir.path().join("runs").is_dir(),
        "expected runs/ dir to be created by `kbx init`"
    );

    // Second `kbx init <dir>` without --force: should refuse (non-zero exit), since the
    // template files already exist.
    Command::cargo_bin("kbx")
        .expect("find kbx binary")
        .arg("init")
        .arg(dir.path())
        .assert()
        .failure()
        .stderr(predicate::str::is_empty().not());

    // Third `kbx init <dir> --force`: should succeed again, overwriting the existing files.
    Command::cargo_bin("kbx")
        .expect("find kbx binary")
        .arg("init")
        .arg(dir.path())
        .arg("--force")
        .assert()
        .success();
}
