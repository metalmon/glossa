use std::path::{Path, PathBuf};

/// Walk up from `start` to the first ancestor that contains a `.glossa/` directory.
pub fn discover_root_from(start: &Path) -> Option<PathBuf> {
    let mut dir = start;
    loop {
        if dir.join(".glossa").is_dir() {
            return Some(dir.to_path_buf());
        }
        dir = dir.parent()?;
    }
}

/// Resolve the knowledge-base root for a command:
/// - `Some(path)` → use it as-is (explicit);
/// - `None` → discover from the current dir (walk up to a `.glossa/`), else the current dir.
pub fn resolve_root(explicit: Option<PathBuf>) -> PathBuf {
    if let Some(p) = explicit {
        return p;
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    discover_root_from(&cwd).unwrap_or(cwd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_root_from_a_subfolder() {
        let base = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(base.path().join(".glossa")).unwrap();
        let sub = base.path().join("a").join("b");
        std::fs::create_dir_all(&sub).unwrap();
        // canonicalize to avoid Windows \\?\ / symlink mismatches in the assert
        let got = discover_root_from(&sub).unwrap();
        assert_eq!(
            std::fs::canonicalize(&got).unwrap(),
            std::fs::canonicalize(base.path()).unwrap()
        );
    }

    #[test]
    fn returns_none_when_no_glossa_anywhere() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("x");
        std::fs::create_dir_all(&sub).unwrap();
        assert!(discover_root_from(&sub).is_none());
    }

    #[test]
    fn resolve_root_keeps_explicit_path() {
        let p = PathBuf::from("some/explicit/path");
        assert_eq!(resolve_root(Some(p.clone())), p);
    }
}
