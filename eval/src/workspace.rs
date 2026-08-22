use std::path::{Path, PathBuf};

pub struct KbxPaths {
    pub root: PathBuf,
    pub kbx_dir: PathBuf,
    pub lab: PathBuf,
    pub dataset: PathBuf,
    pub runs: PathBuf,
    pub answer: PathBuf,
    pub builder: PathBuf,
    pub bridge: PathBuf,
    pub judge: PathBuf,
    pub reflect: PathBuf,
    pub distil: PathBuf,
}

impl KbxPaths {
    pub fn for_root(root: PathBuf) -> Self {
        let kbx = root.join(".glossa").join("kbx");
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
            distil: f("distil.md"),
            kbx_dir: kbx,
            root,
        }
    }
}

pub fn glossa_dir(root: &Path) -> PathBuf {
    root.join(".glossa")
}

pub fn resolve(explicit: Option<PathBuf>) -> KbxPaths {
    KbxPaths::for_root(glossa::root::resolve_root(explicit))
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
}
