//! `kbx init` scaffolding: writes a fresh `<root>/.glossa/kbx/` eval workspace — `lab.toml`
//! (endpoints only, no corpus) + the editable `answer.md`/`builder.md`/`bridge.md`/`judge.md`
//! prompt files + `reflect.md`/`reason.md` stubs + a starter `dataset.toml`, plus an empty
//! `runs/` dir — from the embedded templates in `eval/templates/`. Without `--force`, an
//! existing file is left untouched (skip-existing) so re-running `kbx init` on a live workspace
//! never clobbers edits by accident.

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

    let files: [(&Path, &str); 9] = [
        (&paths.lab, LAB_TOML),
        (&paths.answer, ANSWER_MD),
        (&paths.builder, BUILDER_MD),
        (&paths.bridge, BRIDGE_MD),
        (&paths.judge, JUDGE_MD),
        (&paths.reflect, REFLECT_MD),
        (&paths.reason, REASON_MD),
        (&paths.distil, DISTIL_MD),
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
        for f in [&p.lab, &p.answer, &p.builder, &p.bridge, &p.judge, &p.dataset] {
            assert!(f.exists(), "missing {}", f.display());
        }
        // prompts must NOT be indexable corpus content: they live under .glossa
        assert!(p.builder.starts_with(dir.path().join(".glossa")));
        // no corpus= leaked into lab.toml
        let lab = std::fs::read_to_string(&p.lab).unwrap();
        assert!(!lab.contains("corpus"));
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
    fn init_writes_distil_md_with_seed_generation_markers() {
        let dir = tempfile::tempdir().unwrap();
        let p = scaffold_init(dir.path(), false).unwrap();
        let distil = std::fs::read_to_string(&p.distil).unwrap();
        assert!(
            distil.contains("propose_gold"),
            "distil.md must instruct the model to call propose_gold"
        );
        assert!(
            distil.contains("gate_ok"),
            "distil.md must instruct the model to self-gate via gate_ok"
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
