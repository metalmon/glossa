//! Integration guarantee (spec §5): a full cycle against a separated corpus/state-dir never
//! writes into any corpus root — everything `.glossa`-shaped lands under the resolved
//! `state_base` instead. Read-only mounts aren't portable to Windows CI, so this proves the
//! guarantee structurally: snapshot each corpus root's file list + (len, mtime) before and after
//! a full cycle, and assert the delta is empty.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Recursive (path, len, mtime) snapshot of `root`, so a before/after diff catches ANY corpus
/// write — a new file, a modified one, or a touched mtime — not just `.glossa` creation.
fn snapshot(root: &Path) -> BTreeMap<PathBuf, (u64, std::time::SystemTime)> {
    let mut out = BTreeMap::new();
    for entry in ignore::WalkBuilder::new(root).build() {
        let entry = entry.expect("walk corpus root");
        if entry.file_type().is_some_and(|t| t.is_file()) {
            let meta = entry.metadata().expect("stat corpus file");
            out.insert(
                entry.path().to_path_buf(),
                (meta.len(), meta.modified().expect("mtime")),
            );
        }
    }
    out
}

/// Command-level proxy for the guarantee: the two tests above exercise the library write
/// PRIMITIVES directly (`index_dir_at`, `TraceLog::to_dir`, `with_notebook_write_lock`, ...).
/// This one instead runs the actual `kb` binary — `kb ontology init <corpus> --template faq
/// --state-dir <state>` — a real write-generating command whose materialize path was moved to
/// `state_base` in Task 8, so the guarantee is also proven at the CLI-command surface a user
/// actually invokes, not just at the internal API surface.
#[test]
fn kb_ontology_init_command_never_writes_into_corpus_root() {
    let corpus = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    std::fs::write(corpus.path().join("a.md"), "# Title\n\nunrelated corpus content").unwrap();
    let before = snapshot(corpus.path());

    assert_cmd::Command::cargo_bin("kb")
        .unwrap()
        .args([
            "ontology",
            "init",
            corpus.path().to_str().unwrap(),
            "--template",
            "faq",
            "--state-dir",
            state.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    let after = snapshot(corpus.path());
    assert_eq!(before, after, "kb ontology init must not write into the corpus root");
    assert!(!corpus.path().join(".glossa").exists());
    assert!(
        state.path().join(".glossa").join("ontology.toml").exists(),
        "ontology.toml must land under the state-dir, not the corpus"
    );
}

#[test]
fn full_cycle_never_writes_into_corpus_roots() {
    use glossa::root::Root;
    let corpus = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    std::fs::write(corpus.path().join("a.md"), "# Title\n\nbody about pumps and valves").unwrap();
    std::fs::write(corpus.path().join("b.md"), "see [a](a.md) for pumps").unwrap();
    let before = snapshot(corpus.path()); // (path, len, mtime) set, recursive
    let roots = [Root {
        label: String::new(),
        path: corpus.path().into(),
    }];
    glossa::index::store::index_dir_at(&roots, state.path(), true).unwrap();
    let h = glossa::graph::handle::GraphHandle::open_at(&roots, state.path()).unwrap();
    let _ = h.idx(); // search/read exercised via DocIndex
    glossa::cli_fmt::write_last_search(state.path(), &[("a.md".into(), "p.1".into())]).unwrap();
    // Traces and notebook notes must ALSO land under state_base, not the corpus.
    glossa::trace::TraceLog::to_dir(state.path()).log("read", serde_json::json!({}), serde_json::json!({}));
    // `notebook` is a cargo feature (default-on, off under --no-default-features); gate the
    // note-write step so this test also compiles in the lean release config CI.
    #[cfg(feature = "notebook")]
    glossa::notebook::with_notebook_write_lock(state.path(), || {
        let dir = glossa::notebook::notes_root(state.path()).join("a.md");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("n.md"), "note body").unwrap();
        Ok(())
    })
    .unwrap();
    let after = snapshot(corpus.path());
    assert_eq!(before, after, "no corpus file created or modified");
    assert!(!corpus.path().join(".glossa").exists());
    assert!(state.path().join(".glossa").join("graph.sqlite").exists());
    assert!(
        state.path().join(".glossa").join("traces").is_dir(),
        "traces under state_base"
    );
    #[cfg(feature = "notebook")]
    assert!(
        state
            .path()
            .join(".glossa")
            .join("notes")
            .join("a.md")
            .join("n.md")
            .exists(),
        "notes under state_base"
    );
}

/// Carry-forward from Task 7: the read-only guarantee must hold with MORE than one corpus root,
/// and a secondary root must actually work end-to-end — not just sit there unindexed. Covers both
/// halves: (a) a two-root index + a lazy reindex never write into EITHER corpus root, and (b) a
/// lazy reindex picks up an in-place edit made under the SECONDARY root.
#[test]
fn two_root_index_leaves_both_corpus_roots_unchanged() {
    use glossa::root::Root;
    let docs = tempfile::tempdir().unwrap();
    let specs = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    std::fs::write(
        docs.path().join("a.md"),
        "# Pumps\n\nprimary root content about pumps",
    )
    .unwrap();
    std::fs::write(
        specs.path().join("b.md"),
        "# Valves\n\nsecondary root content about valves",
    )
    .unwrap();

    let before_docs = snapshot(docs.path());

    let roots = vec![
        Root {
            label: "docs".into(),
            path: docs.path().into(),
        },
        Root {
            label: "specs".into(),
            path: specs.path().into(),
        },
    ];
    glossa::index::store::index_dir_at(&roots, state.path(), true).unwrap();

    // The secondary root is indexed and searchable end-to-end through the shared retrieval
    // handle — not just present in the walk, but actually retrievable.
    let h = glossa::graph::handle::GraphHandle::open_at(&roots, state.path()).unwrap();
    let hits = h.idx().search("valves", 10).unwrap();
    assert!(
        hits.iter().any(|r| r.path == "specs/b.md"),
        "secondary-root file must be indexed and searchable end-to-end: {hits:?}"
    );

    // An in-place edit under the SECONDARY root, picked up by a lazy reindex (the same primitive
    // `read`/freshen uses) — proves the secondary root isn't a write-once fluke.
    std::fs::write(
        specs.path().join("b.md"),
        "# Valves\n\nUPDATED secondary root content about actuators",
    )
    .unwrap();
    let before_specs_reindex_edit = snapshot(specs.path());
    glossa::index::store::ensure_fresh_at(&roots, state.path()).unwrap();
    let h2 = glossa::graph::handle::GraphHandle::open_at(&roots, state.path()).unwrap();
    let hits2 = h2.idx().search("actuators", 10).unwrap();
    assert!(
        hits2.iter().any(|r| r.path == "specs/b.md"),
        "in-place edit under the secondary root must be picked up by a lazy reindex: {hits2:?}"
    );

    // The lazy reindex itself wrote nothing else into the secondary root beyond our own
    // deliberate edit (already reflected in this snapshot, taken before `ensure_fresh_at` ran).
    let after_specs_reindex = snapshot(specs.path());
    assert_eq!(
        before_specs_reindex_edit, after_specs_reindex,
        "ensure_fresh_at (lazy reindex) must not write into the secondary corpus root"
    );

    // The PRIMARY root — untouched by any of this — stays byte-identical throughout.
    let after_docs = snapshot(docs.path());
    assert_eq!(
        before_docs, after_docs,
        "indexing/reindexing a sibling root must not touch the primary corpus root"
    );
    assert!(!docs.path().join(".glossa").exists());
    assert!(!specs.path().join(".glossa").exists());
    assert!(state.path().join(".glossa").join("graph.sqlite").exists());
}
