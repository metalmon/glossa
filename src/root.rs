use std::path::{Path, PathBuf};

/// Where the resolved root came from — surfaced so the CLI/MCP can warn when the choice is
/// implicit (walked up to an ancestor) instead of what the caller likely meant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootOrigin {
    /// An explicit `--path` was given; used verbatim.
    Explicit,
    /// The current directory itself held a `.glossa/`.
    Cwd,
    /// No `.glossa/` in the CWD — discovery walked UP to an ancestor's `.glossa/`.
    DiscoveredUp,
    /// No `.glossa/` anywhere up the chain; falling back to the CWD (a fresh index is created here).
    Fallback,
}

/// The outcome of root resolution: the chosen root, how it was chosen, and — for the nested-corpus
/// trap — the nearest `.glossa/` STRICTLY ABOVE the chosen root, if any. A second `.glossa` in the
/// ancestor chain means a server rooted higher would index this tree too (split-brain): worth a
/// warning, since it is almost never intended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRoot {
    pub root: PathBuf,
    pub origin: RootOrigin,
    pub nested_ancestor: Option<PathBuf>,
}

impl ResolvedRoot {
    /// Human-readable warnings to print (to stderr) for this resolution. Empty when the root is
    /// unambiguous. Pure so it can be unit-tested without touching stderr.
    pub fn advisories(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.origin == RootOrigin::DiscoveredUp {
            out.push(format!(
                "no .glossa in the current directory — using the one discovered above at {}. \
                 If you meant to index HERE, pass `--path .` (a deleted .glossa is silently \
                 recreated in an ancestor otherwise).",
                self.root.display()
            ));
        }
        if let Some(anc) = &self.nested_ancestor {
            out.push(format!(
                "another .glossa exists above the chosen root at {} — nested corpora. A server \
                 rooted there indexes this tree too, so the CLI and MCP can drift apart.",
                anc.display()
            ));
        }
        out
    }
}

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

/// The nearest `.glossa/` STRICTLY above `dir` (its parent chain), if any. Used to detect a nested
/// corpus — a `.glossa` above the one we chose.
fn ancestor_glossa_above(dir: &Path) -> Option<PathBuf> {
    let mut d = dir.parent()?;
    loop {
        if d.join(".glossa").is_dir() {
            return Some(d.to_path_buf());
        }
        d = d.parent()?;
    }
}

/// Pure core of root resolution (injectable `cwd`, so it is unit-testable without `current_dir`):
/// - explicit `Some(p)` → use it verbatim (`Explicit`), still flagging a nested ancestor;
/// - `None` → discover from `cwd` (walk up to a `.glossa/`), classifying `Cwd` vs `DiscoveredUp`,
///   or `Fallback` to `cwd` when there is no `.glossa/` anywhere.
pub fn resolve_root_from(explicit: Option<PathBuf>, cwd: &Path) -> ResolvedRoot {
    if let Some(p) = explicit {
        let nested_ancestor = ancestor_glossa_above(&p);
        return ResolvedRoot {
            root: p,
            origin: RootOrigin::Explicit,
            nested_ancestor,
        };
    }
    match discover_root_from(cwd) {
        Some(root) => {
            let origin = if root == cwd {
                RootOrigin::Cwd
            } else {
                RootOrigin::DiscoveredUp
            };
            let nested_ancestor = ancestor_glossa_above(&root);
            ResolvedRoot {
                root,
                origin,
                nested_ancestor,
            }
        }
        None => ResolvedRoot {
            root: cwd.to_path_buf(),
            origin: RootOrigin::Fallback,
            nested_ancestor: None,
        },
    }
}

/// Resolve the knowledge-base root for a command, reading the real `current_dir`. See
/// [`resolve_root_from`] for the resolution rules and [`ResolvedRoot::advisories`] for warnings.
pub fn resolve_root_verbose(explicit: Option<PathBuf>) -> ResolvedRoot {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    resolve_root_from(explicit, &cwd)
}

/// Resolve the knowledge-base root for a command:
/// - `Some(path)` → use it as-is (explicit);
/// - `None` → discover from the current dir (walk up to a `.glossa/`), else the current dir.
///
/// Thin wrapper over [`resolve_root_verbose`] that drops the origin/nesting metadata — for callers
/// that only need the path. Prefer [`resolve_root_verbose`] where warnings should be surfaced.
pub fn resolve_root(explicit: Option<PathBuf>) -> PathBuf {
    resolve_root_verbose(explicit).root
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_glossa(dir: &Path) {
        std::fs::create_dir_all(dir.join(".glossa")).unwrap();
    }

    #[test]
    fn discovers_root_from_a_subfolder() {
        let base = tempfile::tempdir().unwrap();
        mk_glossa(base.path());
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

    #[test]
    fn cwd_with_glossa_is_cwd_origin_no_nesting() {
        let base = tempfile::tempdir().unwrap();
        mk_glossa(base.path());
        let r = resolve_root_from(None, base.path());
        assert_eq!(r.root, base.path());
        assert_eq!(r.origin, RootOrigin::Cwd);
        assert_eq!(r.nested_ancestor, None);
        assert!(r.advisories().is_empty());
    }

    #[test]
    fn discovered_up_is_flagged_and_warns() {
        // corpus/.glossa exists; run from corpus/sub (no .glossa there) → walks up.
        let corpus = tempfile::tempdir().unwrap();
        mk_glossa(corpus.path());
        let sub = corpus.path().join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        let r = resolve_root_from(None, &sub);
        assert_eq!(r.root, corpus.path());
        assert_eq!(r.origin, RootOrigin::DiscoveredUp);
        assert_eq!(r.nested_ancestor, None);
        assert!(
            r.advisories()
                .iter()
                .any(|a| a.contains("discovered above")),
            "DiscoveredUp must warn: {:?}",
            r.advisories()
        );
    }

    #[test]
    fn deleted_corpus_glossa_walks_up_to_parent_and_warns() {
        // The reported trap: parent/.glossa exists, corpus/.glossa was DELETED. Running `kb index`
        // from the corpus (no --path) must NOT silently reindex into the parent without a warning.
        let parent = tempfile::tempdir().unwrap();
        mk_glossa(parent.path());
        let corpus = parent.path().join("corpus");
        std::fs::create_dir_all(&corpus).unwrap(); // corpus has NO .glossa (deleted)
        let r = resolve_root_from(None, &corpus);
        assert_eq!(r.root, parent.path(), "walks up to the parent's .glossa");
        assert_eq!(r.origin, RootOrigin::DiscoveredUp);
        assert!(!r.advisories().is_empty(), "the trap must be warned about");
    }

    #[test]
    fn explicit_root_flags_a_nested_ancestor() {
        // parent/.glossa AND parent/corpus/.glossa both exist; MCP started with --path corpus.
        let parent = tempfile::tempdir().unwrap();
        mk_glossa(parent.path());
        let corpus = parent.path().join("corpus");
        std::fs::create_dir_all(&corpus).unwrap();
        mk_glossa(&corpus);
        let r = resolve_root_from(Some(corpus.clone()), parent.path());
        assert_eq!(r.root, corpus);
        assert_eq!(r.origin, RootOrigin::Explicit);
        assert_eq!(
            r.nested_ancestor.as_deref(),
            Some(parent.path()),
            "the .glossa one level up must be reported as a nested ancestor"
        );
        assert!(
            r.advisories().iter().any(|a| a.contains("nested corpora")),
            "nested ancestor must warn: {:?}",
            r.advisories()
        );
    }

    #[test]
    fn fallback_when_no_glossa_uses_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("fresh");
        std::fs::create_dir_all(&sub).unwrap();
        let r = resolve_root_from(None, &sub);
        assert_eq!(r.root, sub);
        assert_eq!(r.origin, RootOrigin::Fallback);
        assert_eq!(r.nested_ancestor, None);
        assert!(r.advisories().is_empty());
    }
}
