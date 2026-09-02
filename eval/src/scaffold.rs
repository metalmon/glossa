//! `kbx init` scaffolding: writes a fresh `<root>/.glossa/kbx/` eval workspace — `lab.toml`
//! (endpoints only, no corpus) + the editable `answer.md`/`builder.md`/`bridge.md`/`judge.md`
//! prompt files + `reflect.md`/`reason.md` stubs + the two `distil` prompts (`distil.md` for the
//! default densify pass, `golds.md` for the retained `--emit-golds` gold generator — each
//! mode reads its own file) + a starter `dataset.toml`, plus an empty `runs/` dir — from the
//! embedded templates in `eval/templates/`. Without `--force`, an existing file is left untouched
//! (skip-existing) so re-running `kbx init` on a live workspace never clobbers edits by accident.

use crate::workspace::KbxPaths;
use anyhow::Context;
use std::path::Path;

const LAB_TOML: &str = include_str!("../templates/lab.toml");
const ANSWER_MD: &str = include_str!("../templates/answer.md");
const BUILDER_MD: &str = include_str!("../templates/builder.md");
const BRIDGE_MD: &str = include_str!("../templates/bridge.md");
const JUDGE_MD: &str = include_str!("../templates/judge.md");
const REFLECT_MD: &str = include_str!("../templates/reflect.md");
const REASON_MD: &str = include_str!("../templates/reason.md");
const DISTIL_MD: &str = include_str!("../templates/distil.md");
const ALIASES_MD: &str = include_str!("../templates/aliases.md");
const GOLDS_MD: &str = include_str!("../templates/golds.md");
const USER_SIM_MD: &str = include_str!("../templates/user_sim.md");
const DATASET_TOML: &str = include_str!("../templates/dataset.toml");

/// Write a fresh `kbx` workspace at `<root>/.glossa/kbx/`: `lab.toml`, `answer.md`, `builder.md`,
/// `bridge.md`, `judge.md`, `reflect.md`, `reason.md`, `dataset.toml` (embedded templates) plus an
/// empty `runs/` dir. Without `force`, a file that already exists is left as-is (skip-existing);
/// with `force`, every template file is rewritten. Returns the `KbxPaths` for the scaffolded
/// workspace so callers don't have to re-derive them.
pub fn scaffold_init(root: &Path, force: bool) -> anyhow::Result<KbxPaths> {
    let paths = KbxPaths::for_root(root.to_path_buf());

    std::fs::create_dir_all(&paths.kbx_dir)
        .with_context(|| format!("create workspace dir {}", paths.kbx_dir.display()))?;
    std::fs::create_dir_all(&paths.runs)
        .with_context(|| format!("create {}", paths.runs.display()))?;

    let files: [(&Path, &str); 12] = [
        (&paths.lab, LAB_TOML),
        (&paths.answer, ANSWER_MD),
        (&paths.builder, BUILDER_MD),
        (&paths.bridge, BRIDGE_MD),
        (&paths.judge, JUDGE_MD),
        (&paths.reflect, REFLECT_MD),
        (&paths.reason, REASON_MD),
        (&paths.distil, DISTIL_MD),
        (&paths.aliases, ALIASES_MD),
        (&paths.golds, GOLDS_MD),
        (&paths.user_sim, USER_SIM_MD),
        (&paths.dataset, DATASET_TOML),
    ];

    for (p, content) in &files {
        if !force && p.exists() {
            continue;
        }
        std::fs::write(p, content).with_context(|| format!("write {}", p.display()))?;
    }

    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_creates_glossa_kbx_with_prompts() {
        let dir = tempfile::tempdir().unwrap();
        let p = scaffold_init(dir.path(), false).unwrap();
        for f in [
            &p.lab,
            &p.answer,
            &p.builder,
            &p.bridge,
            &p.judge,
            &p.reflect,
            &p.reason,
            &p.distil,
            &p.golds,
            &p.user_sim,
            &p.dataset,
        ] {
            assert!(f.exists(), "missing {}", f.display());
        }
        // prompts must NOT be indexable corpus content: they live under .glossa
        assert!(p.builder.starts_with(dir.path().join(".glossa")));
        // no corpus= leaked into lab.toml
        let lab = std::fs::read_to_string(&p.lab).unwrap();
        assert!(!lab.contains("corpus"));
    }

    /// A fresh `kbx init` workspace ships `[tuning]` fully populated at its documented defaults
    /// (3/30/3, plus the four `jobs_*` knobs at 3) — the user's explicit ask: a workspace should
    /// be ready to edit, not empty, so a user tuning a knob edits an existing line instead of
    /// having to invent the section from scratch. `LabConfig::load` (the same parser `kbx` itself
    /// uses) must read it back exactly.
    #[test]
    fn init_writes_lab_toml_with_tuning_section_at_documented_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let p = scaffold_init(dir.path(), false).unwrap();
        let lab_text = std::fs::read_to_string(&p.lab).unwrap();
        assert!(
            lab_text.contains("[tuning]"),
            "scaffolded lab.toml must have a [tuning] section"
        );

        let lab = kb_eval_lab_config(&p.lab);
        assert_eq!(lab.tuning.fanout_max, Some(3));
        assert_eq!(lab.tuning.max_rounds, Some(30));
        assert_eq!(lab.tuning.chunks_per_round, Some(3));
        assert_eq!(lab.tuning.jobs_build, Some(3));
        assert_eq!(lab.tuning.jobs_reason, Some(3));
        assert_eq!(lab.tuning.jobs_train, Some(3));
        assert_eq!(lab.tuning.jobs_distil, Some(3));
    }

    /// Thin helper so the scaffold test parses lab.toml through the SAME `LabConfig::load_at` path
    /// `kbx` itself uses, not a hand-rolled TOML check — proves the scaffolded file is not just
    /// text containing `[tuning]`, but a section the real config loader actually resolves.
    fn kb_eval_lab_config(lab_path: &std::path::Path) -> crate::lab::LabConfig {
        crate::lab::LabConfig::load_at(lab_path).expect("scaffolded lab.toml must parse")
    }

    #[test]
    fn init_writes_reason_md_with_backward_chain_markers() {
        let dir = tempfile::tempdir().unwrap();
        let p = scaffold_init(dir.path(), false).unwrap();
        let reason = std::fs::read_to_string(&p.reason).unwrap();
        assert!(
            reason.contains("query side"),
            "reason.md must frame the task as reconstructing the query side of the chain"
        );
        assert!(
            reason.contains("walk the schema-graph backward"),
            "reason.md must instruct walking the schema-graph backward from the grounded terminal"
        );
        assert!(
            reason.contains("entry node"),
            "reason.md must instruct reifying each path's far end as an entry node"
        );
    }

    #[test]
    fn init_writes_distil_md_with_densify_markers() {
        let dir = tempfile::tempdir().unwrap();
        let p = scaffold_init(dir.path(), false).unwrap();
        let distil = std::fs::read_to_string(&p.distil).unwrap();
        assert!(
            distil.contains("graph_upsert"),
            "distil.md must instruct the model to write via graph_upsert"
        );
        assert!(
            distil.contains("already holds") || distil.contains("already grounded"),
            "distil.md must frame the task around what the graph already holds for a section"
        );
        assert!(
            distil.contains("query-side"),
            "distil.md must cover query-side reasoning, not just grounded terminals"
        );
    }

    #[test]
    fn init_writes_golds_md_with_seed_generation_markers() {
        let dir = tempfile::tempdir().unwrap();
        let p = scaffold_init(dir.path(), false).unwrap();
        let golds = std::fs::read_to_string(&p.golds).unwrap();
        assert!(
            golds.contains("propose_gold"),
            "golds.md must instruct the model to call propose_gold"
        );
        assert!(
            golds.contains("gate_ok"),
            "golds.md must instruct the model to self-gate via gate_ok"
        );
    }

    #[test]
    fn init_creates_runs_dir() {
        let dir = tempfile::tempdir().unwrap();
        let p = scaffold_init(dir.path(), false).unwrap();
        assert!(p.runs.is_dir());
    }

    #[test]
    fn init_skips_existing_without_force_and_overwrites_with_force() {
        let dir = tempfile::tempdir().unwrap();
        let p = scaffold_init(dir.path(), false).unwrap();
        std::fs::write(&p.answer, "custom content").unwrap();

        // Re-running without --force must not clobber the edit.
        scaffold_init(dir.path(), false).unwrap();
        assert_eq!(
            std::fs::read_to_string(&p.answer).unwrap(),
            "custom content"
        );

        // --force overwrites.
        scaffold_init(dir.path(), true).unwrap();
        assert_ne!(
            std::fs::read_to_string(&p.answer).unwrap(),
            "custom content"
        );
    }
}
