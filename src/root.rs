use anyhow::Context;
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

/// One named corpus root: a stable label paired with a filesystem path. For the single positional
/// root, the label is ALWAYS empty — whether the corpus was reached by discovery (no `.glossa` in
/// cwd, walked up or fell back) or given explicitly as a positional `PATH`. A label only appears
/// for a corpus attached via `--root [LABEL=]PATH`. See [`parse_root_arg`] for that token format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Root {
    /// "" for discovery and positional `PATH`; set only for a `--root` entry — a stable label
    /// persisted in keys.
    pub label: String,
    pub path: PathBuf,
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
    /// The full set of corpus roots. Defaults to a single root at `root`, labeled per [`Root`]'s
    /// discovery-vs-explicit rule — back-compat for callers that only know about the single-root path.
    pub roots: Vec<Root>,
    /// Base directory for on-disk state (`.glossa/`, index, etc). Defaults to `root`.
    pub state_base: PathBuf,
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
        let roots = vec![Root {
            label: String::new(),
            path: p.clone(),
        }];
        let state_base = p.clone();
        return ResolvedRoot {
            root: p,
            origin: RootOrigin::Explicit,
            nested_ancestor,
            roots,
            state_base,
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
            let roots = vec![Root {
                label: String::new(),
                path: root.clone(),
            }];
            let state_base = root.clone();
            ResolvedRoot {
                root,
                origin,
                nested_ancestor,
                roots,
                state_base,
            }
        }
        None => ResolvedRoot {
            root: cwd.to_path_buf(),
            origin: RootOrigin::Fallback,
            nested_ancestor: None,
            roots: vec![Root {
                label: String::new(),
                path: cwd.to_path_buf(),
            }],
            state_base: cwd.to_path_buf(),
        },
    }
}

/// Parse one `--root` / `GLOSSA_ROOTS` token. `label=path` → explicit label (trimmed); bare `path`
/// → label auto-derived from the file name (basename). Errors on empty input or an empty label
/// before `=`. Never empty-labels a `Root` here; the empty label is reserved for the back-compat
/// single positional root.
pub fn parse_root_arg(s: &str) -> anyhow::Result<Root> {
    let s = s.trim();
    anyhow::ensure!(!s.is_empty(), "empty --root token: expected [LABEL=]PATH");
    match s.split_once('=') {
        Some((label, path)) => {
            let label = label.trim().to_string();
            anyhow::ensure!(
                !label.is_empty(),
                "empty root label before '=': use LABEL=PATH"
            );
            Ok(Root {
                label,
                path: PathBuf::from(path.trim()),
            })
        }
        None => {
            let p = PathBuf::from(s);
            let label = p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            anyhow::ensure!(
                !label.is_empty(),
                "cannot derive a label from '{s}': give an explicit LABEL=PATH"
            );
            Ok(Root { label, path: p })
        }
    }
}

/// The already-parsed multi-root inputs a caller assembles from flags/env/config, handed to the
/// shared resolver. `positional` is today's single `PATH` (empty-label root); `roots` are the
/// `--root`/`GLOSSA_ROOTS` entries (Task 1 parsed); `state_dir` is `--state-dir`/`GLOSSA_STATE_DIR`.
#[derive(Debug, Clone, Default)]
pub struct RootInputs {
    pub positional: Option<PathBuf>,
    pub roots: Vec<Root>,
    pub state_dir: Option<PathBuf>,
}

/// Pure core (injectable `cwd`). Errors — never a silent CWD fallback — when separated/multi-root
/// mode is requested (state_dir set, or ≥1 `--root`) but no corpus path is given, or when two
/// roots collide on label.
pub fn resolve_roots_from(inputs: RootInputs, cwd: &Path) -> anyhow::Result<ResolvedRoot> {
    let separated = inputs.state_dir.is_some() || !inputs.roots.is_empty();

    // Assemble the corpus roots. Precedence: explicit --root list wins; else the single positional
    // becomes the empty-label root; else (co-located, no separation) fall through to discovery.
    let roots: Vec<Root> = if !inputs.roots.is_empty() {
        inputs.roots.clone()
    } else if let Some(p) = &inputs.positional {
        vec![Root {
            label: String::new(),
            path: p.clone(),
        }]
    } else {
        Vec::new()
    };

    if separated {
        // Multi-root / state-dir is ALWAYS explicit — never a silent walk-up/CWD fallback.
        if roots.is_empty() {
            anyhow::bail!(
                "a corpus path is required in state-dir/multi-root mode: pass a positional PATH or --root [LABEL=]PATH \
                 (no current-directory fallback here, unlike the single co-located default)"
            );
        }
        let mut seen = std::collections::BTreeSet::new();
        for r in &roots {
            if !seen.insert(r.label.clone()) {
                anyhow::bail!(
                    "duplicate root label '{}' — give each --root an explicit LABEL=PATH so keys stay unique",
                    r.label
                );
            }
        }
        let primary = roots[0].path.clone();
        let state_base = inputs.state_dir.clone().unwrap_or_else(|| primary.clone());
        return Ok(ResolvedRoot {
            root: primary,
            origin: RootOrigin::Explicit,
            nested_ancestor: ancestor_glossa_above(&roots[0].path),
            roots,
            state_base,
        });
    }

    // Co-located default path: identical to today (discovery/walk-up), with the single-root defaults.
    Ok(resolve_root_from(inputs.positional, cwd))
}

/// Reads real `current_dir`. Thin wrapper over [`resolve_roots_from`].
pub fn resolve_roots_verbose(inputs: RootInputs) -> anyhow::Result<ResolvedRoot> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    resolve_roots_from(inputs, &cwd)
}

/// Resolve the knowledge-base root for a command, reading the real `current_dir`. See
/// [`resolve_root_from`] for the resolution rules and [`ResolvedRoot::advisories`] for warnings.
pub fn resolve_root_verbose(explicit: Option<PathBuf>) -> ResolvedRoot {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    resolve_root_from(explicit, &cwd)
}

/// `mkdir -p `<state_base>/.glossa`, then probe-write a throwaway file — a clear, early error when
/// the state dir isn't writable (read-only mount, permissions, a network share gone stale) instead
/// of a confusing failure deep inside the index/graph writer. Idempotent; safe to call on every
/// startup/resolution.
pub fn ensure_state_writable(state_base: &Path) -> anyhow::Result<()> {
    let g = state_base.join(".glossa");
    std::fs::create_dir_all(&g).with_context(|| format!("create state dir {}", g.display()))?;
    let probe = g.join(".write_probe");
    std::fs::write(&probe, b"ok").with_context(|| {
        format!(
            "state-dir {} is not writable — index/graph/locks cannot be persisted there",
            g.display()
        )
    })?;
    let _ = std::fs::remove_file(&probe);
    Ok(())
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

    #[test]
    fn resolved_root_defaults_are_single_empty_label_and_colocated_state() {
        let base = tempfile::tempdir().unwrap();
        mk_glossa(base.path());
        let r = resolve_root_from(None, base.path());
        assert_eq!(
            r.roots,
            vec![Root {
                label: String::new(),
                path: base.path().to_path_buf()
            }]
        );
        assert_eq!(r.state_base, base.path());
    }

    #[test]
    fn explicit_positional_stays_empty_label() {
        let base = tempfile::tempdir().unwrap();
        let corpus = base.path().join("plc");
        std::fs::create_dir_all(&corpus).unwrap();
        let r = resolve_root_from(Some(corpus.clone()), base.path());
        assert_eq!(r.origin, RootOrigin::Explicit);
        assert_eq!(
            r.roots,
            vec![Root {
                label: String::new(),
                path: corpus.clone()
            }]
        );
        assert_eq!(r.state_base, corpus);
    }

    #[test]
    fn discovery_keeps_empty_label() {
        let base = tempfile::tempdir().unwrap();
        mk_glossa(base.path());
        let r = resolve_root_from(None, base.path());
        assert_eq!(r.origin, RootOrigin::Cwd);
        assert_eq!(
            r.roots,
            vec![Root {
                label: String::new(),
                path: base.path().to_path_buf()
            }]
        );
    }

    #[test]
    fn positional_with_state_dir_stays_empty_label() {
        let corpus = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        let corpus_named = corpus.path().join("ivk");
        std::fs::create_dir_all(&corpus_named).unwrap();
        let inputs = RootInputs {
            positional: Some(corpus_named.clone()),
            state_dir: Some(state.path().to_path_buf()),
            ..Default::default()
        };
        let r = resolve_roots_from(inputs, corpus.path()).unwrap();
        assert_eq!(
            r.roots,
            vec![Root {
                label: String::new(),
                path: corpus_named
            }]
        );
        assert_eq!(r.state_base, state.path());
    }

    #[test]
    fn parse_root_arg_labeled_and_bare() {
        let labeled = parse_root_arg("docs=/mnt/a").unwrap();
        assert_eq!(
            labeled,
            Root {
                label: "docs".into(),
                path: PathBuf::from("/mnt/a")
            }
        );
        let bare = parse_root_arg("/mnt/store/specs").unwrap();
        assert_eq!(bare.label, "specs"); // basename auto-label
        assert_eq!(bare.path, PathBuf::from("/mnt/store/specs"));
        assert_eq!(parse_root_arg("  api = /mnt/b ").unwrap().label, "api"); // trims label
    }

    #[test]
    fn parse_root_arg_rejects_empty_input_and_empty_label() {
        assert!(parse_root_arg("   ")
            .unwrap_err()
            .to_string()
            .contains("empty"));
        assert!(parse_root_arg("=/mnt/a")
            .unwrap_err()
            .to_string()
            .contains("label"));
    }

    #[test]
    fn positional_root_via_root_inputs_stays_empty_label() {
        let base = tempfile::tempdir().unwrap();
        let corpus = base.path().join("plc");
        std::fs::create_dir_all(&corpus).unwrap();
        mk_glossa(&corpus);
        let inputs = RootInputs {
            positional: Some(corpus.clone()),
            ..Default::default()
        };
        let r = resolve_roots_from(inputs, base.path()).unwrap();
        assert_eq!(r.root, corpus);
        assert_eq!(
            r.roots,
            vec![Root {
                label: String::new(),
                path: corpus.clone()
            }]
        );
        assert_eq!(r.state_base, corpus);
        assert_eq!(r.origin, RootOrigin::Explicit);
    }

    #[test]
    fn state_dir_sets_state_base_and_keeps_roots() {
        // Named subdirectory (not the tempdir's own random name) so this doesn't read as
        // coincidentally passing for an empty-label assertion.
        let corpus_tmp = tempfile::tempdir().unwrap();
        let corpus = corpus_tmp.path().join("ivk");
        std::fs::create_dir_all(&corpus).unwrap();
        let state = tempfile::tempdir().unwrap();
        let inputs = RootInputs {
            positional: Some(corpus.clone()),
            state_dir: Some(state.path().to_path_buf()),
            ..Default::default()
        };
        let r = resolve_roots_from(inputs, corpus_tmp.path()).unwrap();
        assert_eq!(r.state_base, state.path());
        assert_eq!(
            r.roots,
            vec![Root {
                label: String::new(),
                path: corpus
            }]
        );
    }

    #[test]
    fn multi_root_unifies_and_derives_state_base_from_first_when_no_state_dir() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let inputs = RootInputs {
            roots: vec![
                Root {
                    label: "docs".into(),
                    path: a.path().into(),
                },
                Root {
                    label: "specs".into(),
                    path: b.path().into(),
                },
            ],
            ..Default::default()
        };
        let r = resolve_roots_from(inputs, a.path()).unwrap();
        assert_eq!(r.roots.len(), 2);
        assert_eq!(
            r.state_base,
            a.path(),
            "no --state-dir → first root is the state base"
        );
        assert_eq!(r.root, a.path());
    }

    #[test]
    fn state_dir_without_corpus_is_a_wiring_error_not_cwd_fallback() {
        let state = tempfile::tempdir().unwrap();
        let cwd = tempfile::tempdir().unwrap();
        let inputs = RootInputs {
            state_dir: Some(state.path().to_path_buf()),
            ..Default::default()
        };
        let err = resolve_roots_from(inputs, cwd.path()).unwrap_err();
        assert!(err.to_string().contains("corpus path"), "got: {err}");
    }

    #[test]
    fn ensure_state_writable_creates_dir_and_probes() {
        let state = tempfile::tempdir().unwrap();
        ensure_state_writable(state.path()).unwrap();
        assert!(state.path().join(".glossa").is_dir());
        // Idempotent — calling again on an already-writable dir still succeeds.
        ensure_state_writable(state.path()).unwrap();
    }

    #[test]
    fn duplicate_root_labels_error() {
        let a = tempfile::tempdir().unwrap();
        let inputs = RootInputs {
            roots: vec![
                Root {
                    label: "x".into(),
                    path: a.path().into(),
                },
                Root {
                    label: "x".into(),
                    path: a.path().into(),
                },
            ],
            ..Default::default()
        };
        assert!(resolve_roots_from(inputs, a.path())
            .unwrap_err()
            .to_string()
            .contains("label"));
    }
}
