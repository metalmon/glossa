use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileSig {
    pub mtime_secs: u64,
    pub size: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Manifest {
    pub files: BTreeMap<String, FileSig>,
}

fn manifest_path(dir: &Path) -> std::path::PathBuf {
    dir.join(".glossa").join("manifest.json")
}

impl Manifest {
    pub fn load(dir: &Path) -> Manifest {
        let p = manifest_path(dir);
        match std::fs::read_to_string(&p) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
            Err(_) => Manifest::default(),
        }
    }

    pub fn save(&self, dir: &Path) -> anyhow::Result<()> {
        let p = manifest_path(dir);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let s = serde_json::to_string_pretty(self).context("serialize manifest")?;
        std::fs::write(&p, s).with_context(|| format!("write {p:?}"))?;
        Ok(())
    }

    /// True if the path is new or its signature differs from the recorded one.
    pub fn changed(&self, path: &str, sig: FileSig) -> bool {
        self.files.get(path) != Some(&sig)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_and_detects_change() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = Manifest::default();
        m.files.insert("a.md".into(), FileSig { mtime_secs: 10, size: 20 });
        m.save(dir.path()).unwrap();

        let loaded = Manifest::load(dir.path());
        assert_eq!(loaded.files.get("a.md"), Some(&FileSig { mtime_secs: 10, size: 20 }));
        assert!(!loaded.changed("a.md", FileSig { mtime_secs: 10, size: 20 }));
        assert!(loaded.changed("a.md", FileSig { mtime_secs: 11, size: 20 }));
        assert!(loaded.changed("new.md", FileSig { mtime_secs: 1, size: 1 }));
    }
}
