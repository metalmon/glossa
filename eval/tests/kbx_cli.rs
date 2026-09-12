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
        "golds.md",
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
        !lab_text.contains("corpus =") && !lab_text.contains("corpus="),
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

/// `kbx dataset stat` output contract: the per-case dump is DETAIL and must print before the
/// headline, and the headline (`cli_fmt::summary` block: separator rule + `label: value` pairs)
/// must be the LAST stdout content — a `kbx dataset stat file.toml | tail` should always land on
/// the tallies, not a random breakdown line. Hermetic: a standalone dataset.toml with no `.glossa`
/// ancestor never touches the graph, so this only exercises the pure-file stat path.
#[test]
fn kbx_dataset_stat_dumps_cases_before_the_headline_summary() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let file = dir.path().join("d.toml");
    std::fs::write(
        &file,
        r#"
[[case]]
id="q1"
question="What is A?"
answer="A is a thing."
hop_type="lexical"

[[case]]
id="q2"
question="What is B?"
answer="B is another thing."
hop_type="multihop"
needs_graph="yes"
"#,
    )
    .unwrap();

    let assert = Command::cargo_bin("kbx")
        .expect("find kbx binary")
        .arg("dataset")
        .arg("stat")
        .arg(&file)
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&assert.get_output().stdout).to_string();
    let lines: Vec<&str> = stdout.lines().collect();

    // Detail FIRST: the per-case dump (id [hop_type] question / -> answer preview).
    let q1_pos = lines
        .iter()
        .position(|l| l.starts_with("q1 ["))
        .expect("per-case dump for q1 present as detail");
    let hop_type_pos = lines
        .iter()
        .position(|l| l.starts_with("hop_type:"))
        .expect("hop_type breakdown present");
    assert!(
        q1_pos < hop_type_pos,
        "per-case dump must print BEFORE the numeric breakdown:\n{stdout}"
    );

    // Headline LAST: the shared `cli_fmt::summary` block (separator rule, then `label: value`
    // pairs) is the final stdout content — nothing prints after it.
    let sep_pos = lines
        .iter()
        .rposition(|l| !l.is_empty() && l.chars().all(|c| c == '\u{2500}'))
        .expect("cli_fmt summary separator rule present");
    assert!(
        hop_type_pos < sep_pos,
        "the breakdown detail must precede the final summary separator:\n{stdout}"
    );
    let tail = &lines[sep_pos..];
    assert!(
        tail.iter().any(|l| l.starts_with("cases: 2")),
        "summary block carries the cases tally:\n{stdout}"
    );
    assert_eq!(
        lines.last().copied(),
        Some("needs_graph: (unset)=1, yes=1"),
        "the summary block's last pair (needs_graph) is the LAST stdout line:\n{stdout}"
    );
    // `needs_graph` must appear exactly once — in the trailing summary — not also as a
    // byte-identical top-of-output body line above the per-case dump.
    assert_eq!(
        lines
            .iter()
            .filter(|l| l.starts_with("needs_graph:"))
            .count(),
        1,
        "needs_graph must appear exactly once (bottom summary only):\n{stdout}"
    );
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
