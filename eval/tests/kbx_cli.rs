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
        "reason.md",
        "distil.md",
        "distil_golds.md",
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

/// `kbx train --help` should expose the DSPy-style budget knobs and the apply-gate escape hatch —
/// a regression here means the `Train` clap variant lost a flag `run_train`'s `TrainArgs` needs.
#[test]
fn kbx_train_help_lists_budget_and_no_apply() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_kbx"))
        .args(["train", "--help"])
        .output()
        .unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(
        s.contains("--auto")
            && s.contains("--max-metric-calls")
            && s.contains("--max-full-evals")
            && s.contains("--no-apply")
    );
}

/// `kbx reason --help` should expose the seed-type restriction and the fan-out cap — a
/// regression here means the `Reason` clap variant lost a flag `run_reason`'s `ReasonArgs` needs.
#[test]
fn kbx_reason_help_lists_seed_type_and_fanout_max() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_kbx"))
        .args(["reason", "--help"])
        .output()
        .unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("--seed-type") && s.contains("--fanout-max"));
}

/// `kbx distil --help` should expose the attempt-count and seed-type knobs — a regression here
/// means the `Distil` clap variant lost a flag `run_distil`'s `DistilArgs` needs.
#[test]
fn kbx_distil_help_lists_count_and_seed_type() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_kbx"))
        .args(["distil", "--help"])
        .output()
        .unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("--count") && s.contains("--seed-type"));
}

/// `kbx distil --help` should also expose the densify-mode flags (Task 4: densify is now the
/// default, golds are opt-in via `--emit-golds`) — a regression here means the `Distil` clap
/// variant lost a flag `distil::run`'s dispatch needs.
#[test]
fn kbx_distil_help_lists_densify_flags_and_emit_golds() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_kbx"))
        .args(["distil", "--help"])
        .output()
        .unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("--emit-golds"));
    assert!(s.contains("--force"));
    assert!(s.contains("--resume"));
    assert!(s.contains("--doc"));
    assert!(s.contains("--chunks-per-round"));
    assert!(s.contains("--no-progress"));
}
