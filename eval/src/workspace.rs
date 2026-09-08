use std::path::{Path, PathBuf};

pub struct KbxPaths {
    pub root: PathBuf,
    /// Base directory for on-disk state (`.glossa/`, index, graph, `kbx/` config+runs). Defaults
    /// to `root` (today's co-located layout) via [`resolve`]/[`KbxPaths::for_root`]; diverges from
    /// `root` only when a caller goes through [`resolve_with`] with an explicit `--state-dir`.
    pub state_base: PathBuf,
    pub kbx_dir: PathBuf,
    pub lab: PathBuf,
    pub dataset: PathBuf,
    pub runs: PathBuf,
    pub answer: PathBuf,
    pub builder: PathBuf,
    pub bridge: PathBuf,
    pub judge: PathBuf,
    pub reflect: PathBuf,
    pub reason: PathBuf,
    pub distil: PathBuf,
    pub golds: PathBuf,
    /// `--aliases-only` enrichment prompt (`aliases.md`).
    pub aliases: PathBuf,
    /// Persona prompt for the reader's simulated-user dialogue gate (`backend::user_sim`). Loaded
    /// only when `[user_sim]` is configured in `lab.toml`; otherwise never read.
    pub user_sim: PathBuf,
}

impl KbxPaths {
    /// Co-located layout: `root` doubles as the state base (today's single-path default).
    pub fn for_root(root: PathBuf) -> Self {
        Self::for_root_and_state(root.clone(), root)
    }

    /// Split layout: corpus content under `root`, on-disk state (`.glossa/`, `kbx/` config+runs)
    /// under `state_base`. `for_root` is the co-located special case (`state_base == root`).
    pub fn for_root_and_state(root: PathBuf, state_base: PathBuf) -> Self {
        let kbx = glossa_dir(&state_base).join("kbx");
        let f = |n: &str| kbx.join(n);
        KbxPaths {
            lab: f("lab.toml"),
            dataset: f("dataset.toml"),
            runs: kbx.join("runs"),
            answer: f("answer.md"),
            builder: f("builder.md"),
            bridge: f("bridge.md"),
            judge: f("judge.md"),
            reflect: f("reflect.md"),
            reason: f("reason.md"),
            distil: f("distil.md"),
            golds: f("golds.md"),
            aliases: f("aliases.md"),
            user_sim: f("user_sim.md"),
            kbx_dir: kbx,
            root,
            state_base,
        }
    }
}

pub fn glossa_dir(root: &Path) -> PathBuf {
    root.join(".glossa")
}

/// Co-located resolution (unchanged default): a single PATH doubles as both corpus root and state
/// base, discovered via `glossa::root::resolve_root`'s walk-up/CWD-fallback rules.
pub fn resolve(explicit: Option<PathBuf>) -> KbxPaths {
    KbxPaths::for_root(glossa::root::resolve_root(explicit))
}

/// Split resolution: `inputs` carries the corpus path(s) and an optional `--state-dir` through
/// `glossa::root::resolve_roots_verbose`, which errors (never a silent CWD fallback) when
/// separated/multi-root mode is requested without a corpus path. Co-located when `inputs` carries
/// no `state_dir`/`roots` (mirrors `resolve`'s discovery rules via the same resolver).
pub fn resolve_with(inputs: glossa::root::RootInputs) -> anyhow::Result<KbxPaths> {
    let resolved = glossa::root::resolve_roots_verbose(inputs)?;
    Ok(KbxPaths::for_root_and_state(
        resolved.root,
        resolved.state_base,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layout_is_under_glossa_kbx() {
        let root = std::path::Path::new("/corp");
        let p = KbxPaths::for_root(root.to_path_buf());
        assert_eq!(p.kbx_dir, root.join(".glossa").join("kbx"));
        assert_eq!(p.lab, root.join(".glossa/kbx/lab.toml"));
        assert_eq!(p.dataset, root.join(".glossa/kbx/dataset.toml"));
        assert_eq!(p.runs, root.join(".glossa/kbx/runs"));
        assert_eq!(p.builder, root.join(".glossa/kbx/builder.md"));
        assert_eq!(p.bridge, root.join(".glossa/kbx/bridge.md"));
    }

    #[test]
    fn resolve_with_state_dir_splits_root_and_state_base() {
        let corpus = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        let inputs = glossa::root::RootInputs {
            positional: Some(corpus.path().to_path_buf()),
            state_dir: Some(state.path().to_path_buf()),
            ..Default::default()
        };
        let p = resolve_with(inputs).unwrap();
        assert_eq!(p.root, corpus.path());
        assert_eq!(p.state_base, state.path());
        assert_eq!(glossa_dir(&p.state_base), state.path().join(".glossa"));
    }
}
