//! CLI-level smoke test for the `kbx` binary's clap/CLI boundary — exercises the real compiled
//! binary via `assert_cmd` rather than calling the internal `scaffold_init` helper directly, so a
//! regression in argument parsing / wiring (not just the underlying logic) would be caught.
//! Hermetic: `kbx init` touches no network, no corpus, no LLM — just the filesystem.

use assert_cmd::Command;

#[test]
fn kbx_init_scaffolds_glossa_kbx_workspace_and_skips_existing_without_force() {
    let dir = tempfile::tempdir().expect("create tempdir");

    // First `kbx init <path>`: should succeed and scaffold `.glossa/kbx/` under the given root.
    Command::cargo_bin("kbx")
        .expect("find kbx binary")
        .arg("init")
        .arg(dir.path())
        .assert()
        .success();

    let kbx_dir = dir.path().join(".glossa").join("kbx");
    for name in [
        "lab.toml",
        "answer.md",
        "builder.md",
        "bridge.md",
        "judge.md",
        "reflect.md",
        "distil.md",
        "dataset.toml",
    ] {
        assert!(
            kbx_dir.join(name).is_file(),
            "expected .glossa/kbx/{name} to be created by `kbx init`"
        );
    }
    assert!(
        kbx_dir.join("runs").is_dir(),
        "expected .glossa/kbx/runs/ dir to be created by `kbx init`"
    );

    let lab_text = std::fs::read_to_string(kbx_dir.join("lab.toml")).unwrap();
    assert!(
        !lab_text.contains("corpus"),
        "lab.toml must not configure a corpus — it comes from kb-style PATH"
    );

    // Second `kbx init <path>` without --force: skips existing files rather than refusing.
    std::fs::write(kbx_dir.join("answer.md"), "custom edit").unwrap();
    Command::cargo_bin("kbx")
        .expect("find kbx binary")
        .arg("init")
        .arg(dir.path())
        .assert()
        .success();
    assert_eq!(
        std::fs::read_to_string(kbx_dir.join("answer.md")).unwrap(),
        "custom edit",
        "without --force, an existing file must be left untouched"
    );

    // Third `kbx init <path> --force`: overwrites the existing (edited) file.
    Command::cargo_bin("kbx")
        .expect("find kbx binary")
        .arg("init")
        .arg(dir.path())
        .arg("--force")
        .assert()
        .success();
    assert_ne!(
        std::fs::read_to_string(kbx_dir.join("answer.md")).unwrap(),
        "custom edit",
        "--force must overwrite existing files"
    );
}

/// `kbx train --help` should expose the GEPA budget knob and the apply-gate escape hatch — a
/// regression here means the `Train` clap variant lost a flag `run_train`'s `TrainArgs` needs.
#[test]
fn kbx_train_help_lists_budget_and_no_apply() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_kbx"))
        .args(["train", "--help"])
        .output()
        .unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("--budget") && s.contains("--no-apply"));
}

/// `kbx distil --help` should expose the gold-dataset override and the split/kb mode knob — a
/// regression here means the `Distil` clap variant lost a flag `run_distil`'s `DistilArgs` needs.
#[test]
fn kbx_distil_help_lists_gold_and_mode() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_kbx"))
        .args(["distil", "--help"])
        .output()
        .unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("--gold") && s.contains("--mode"));
}
