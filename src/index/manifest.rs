use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileSig {
    pub mtime_secs: u64,
    pub size: u64,
}

fn default_index_schema_version() -> u32 {
    1
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub files: BTreeMap<String, FileSig>,
    /// Notebook note files under `.glossa/notes`, keyed by path relative to the notes root
    /// (`doc.md/limits.csp`). `#[serde(default)]` so manifests written before this field existed
    /// still load. Lets `scan_delta` notice notes written outside `note()`.
    #[serde(default)]
    pub notes: BTreeMap<String, FileSig>,
    #[serde(default = "default_index_schema_version")]
    pub index_schema_version: u32,
}

impl Default for Manifest {
    fn default() -> Self {
        Self {
            files: BTreeMap::new(),
            notes: BTreeMap::new(),
            index_schema_version: default_index_schema_version(),
        }
    }
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
        // Atomic publish: write to a sibling temp file, then rename over the target, so a concurrent
        // `Manifest::load` never reads a half-written file (it would `unwrap_or_default()` to empty).
        let tmp = p.with_extension("json.tmp");
        std::fs::write(&tmp, s).with_context(|| format!("write {tmp:?}"))?;
        std::fs::rename(&tmp, &p).with_context(|| format!("rename {tmp:?} -> {p:?}"))?;
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
        m.files.insert(
            "a.md".into(),
            FileSig {
                mtime_secs: 10,
                size: 20,
            },
        );
        m.save(dir.path()).unwrap();

        let loaded = Manifest::load(dir.path());
        assert_eq!(
            loaded.files.get("a.md"),
            Some(&FileSig {
                mtime_secs: 10,
                size: 20
            })
        );
        assert!(!loaded.changed(
            "a.md",
            FileSig {
                mtime_secs: 10,
                size: 20
            }
        ));
        assert!(loaded.changed(
            "a.md",
            FileSig {
                mtime_secs: 11,
                size: 20
            }
        ));
        assert!(loaded.changed(
            "new.md",
            FileSig {
                mtime_secs: 1,
                size: 1
            }
        ));
    }

    #[test]
    fn save_is_atomic_temp_plus_rename() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = Manifest::default();
        m.files.insert(
            "a.md".into(),
            FileSig {
                mtime_secs: 3,
                size: 4,
            },
        );
        m.save(dir.path()).unwrap();
        // No leftover temp file, and the manifest round-trips.
        assert!(
            !dir.path()
                .join(".glossa")
                .join("manifest.json.tmp")
                .exists(),
            "temp file cleaned up"
        );
        let loaded = Manifest::load(dir.path());
        assert_eq!(
            loaded.files.get("a.md"),
            Some(&FileSig {
                mtime_secs: 3,
                size: 4
            })
        );
        // Saving again over an existing file still works (rename overwrites).
        m.files.insert(
            "b.md".into(),
            FileSig {
                mtime_secs: 5,
                size: 6,
            },
        );
        m.save(dir.path()).unwrap();
        assert_eq!(Manifest::load(dir.path()).files.len(), 2);
    }

    #[test]
    fn notes_roundtrip_and_default_to_empty() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = Manifest::default();
        m.notes.insert(
            "doc.md/limits.csp".into(),
            FileSig {
                mtime_secs: 7,
                size: 9,
            },
        );
        m.save(dir.path()).unwrap();
        let loaded = Manifest::load(dir.path());
        assert_eq!(loaded.notes.len(), 1);
        assert_eq!(
            loaded.notes.get("doc.md/limits.csp"),
            Some(&FileSig {
                mtime_secs: 7,
                size: 9
            })
        );
        // A manifest written before the field existed still parses (notes = empty).
        let old = r#"{"files":{"a.md":{"mtime_secs":1,"size":2}},"index_schema_version":2}"#;
        let parsed: Manifest = serde_json::from_str(old).unwrap();
        assert!(parsed.notes.is_empty());
    }
}
