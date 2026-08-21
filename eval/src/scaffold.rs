//! `kbx init` scaffolding: writes a fresh eval workspace — `lab.toml` + the editable
//! `answer.md`/`judge.md` prompt files + a starter `dataset.toml`, plus an empty `runs/` dir —
//! from the embedded templates in `eval/templates/`. Refuses to clobber an existing workspace
//! unless `--force` is passed, so re-running `kbx init` on a live workspace is a deliberate act.

use anyhow::Context;
use std::path::Path;

const LAB_TOML: &str = include_str!("../templates/lab.toml");
const ANSWER_MD: &str = include_str!("../templates/answer.md");
const JUDGE_MD: &str = include_str!("../templates/judge.md");
const DATASET_TOML: &str = include_str!("../templates/dataset.toml");

/// Write a fresh `kbx` workspace at `dir`: `lab.toml`, `answer.md`, `judge.md`, `dataset.toml`
/// (embedded templates) plus an empty `runs/` dir. Without `force`, refuses (errors) if ANY of
/// the four template files already exists — `runs/` itself is always fine to already be there.
pub fn scaffold_init(dir: &Path, force: bool) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create workspace dir {}", dir.display()))?;

    let files: [(&str, &str); 4] = [
        ("lab.toml", LAB_TOML),
        ("answer.md", ANSWER_MD),
        ("judge.md", JUDGE_MD),
        ("dataset.toml", DATASET_TOML),
    ];

    if !force {
        for (name, _) in &files {
            let p = dir.join(name);
            if p.exists() {
                anyhow::bail!(
                    "{} already exists — pass --force to overwrite",
                    p.display()
                );
            }
        }
    }

    for (name, content) in &files {
        let p = dir.join(name);
        std::fs::write(&p, content).with_context(|| format!("write {}", p.display()))?;
    }

    std::fs::create_dir_all(dir.join("runs"))
        .with_context(|| format!("create {}", dir.join("runs").display()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_writes_workspace_and_refuses_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        scaffold_init(dir.path(), false).unwrap();
        for f in ["lab.toml", "answer.md", "judge.md", "dataset.toml"] {
            assert!(dir.path().join(f).exists());
        }
        assert!(dir.path().join("runs").is_dir());
        assert!(
            scaffold_init(dir.path(), false).is_err(),
            "must refuse without --force"
        );
        scaffold_init(dir.path(), true).unwrap(); // --force ok
    }
}
