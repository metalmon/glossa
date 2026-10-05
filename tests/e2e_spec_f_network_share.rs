//! Spec F e2e: a corpus on a NETWORK SHARE, with `.glossa` state on local disk.
//!
//! This is the deployment the `--state-dir` flag exists for, and until now nothing exercised it.
//! It is also where Windows path handling bites: canonicalizing a UNC path yields
//! `\\?\UNC\server\share\dir`, and the engine used to strip only the four-character `\\?\` prefix,
//! leaving `UNC\server\share\dir` — a RELATIVE path whose first component is a directory named
//! "UNC". Every `root.join(rel)` built from it pointed nowhere, so a share-hosted corpus resolved
//! to nothing at all. `strip_verbatim_prefix` (src/index/store.rs) now handles both forms and has
//! string-level unit tests; this file is the end-to-end half.
//!
//! **Opt-in**, because CI runners have no share: set `GLOSSA_TEST_SHARE_ROOT` to a writable
//! directory reached through a share — a UNC path on Windows (`\\localhost\SomeShare\scratch`), or
//! an NFS/SMB mount point on Linux (`/mnt/kb-share/scratch`). The test creates its own corpus
//! subdirectory underneath, so it never touches anything already there, and removes it afterwards.
//! Without the variable it skips with a message rather than passing silently, because a test that
//! quietly does nothing is worse than no test.
#![cfg(feature = "e2e")]

#[path = "e2e/harness.rs"]
mod harness;

use std::path::{Path, PathBuf};

use harness::*;

/// A unique corpus directory under the operator-supplied share root. Unique so two runs, or a run
/// on two machines against one share, cannot collide.
fn share_corpus() -> Option<PathBuf> {
    let root = std::env::var("GLOSSA_TEST_SHARE_ROOT").ok()?;
    let root = root.trim();
    if root.is_empty() {
        return None;
    }
    let unique = format!(
        "glossa-share-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let dir = Path::new(root).join(unique);
    std::fs::create_dir_all(&dir).expect("creating the test corpus on the share");
    Some(dir)
}

/// Every entry under `dir`, relative and sorted — used to prove the corpus is untouched.
fn listing(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p.clone());
            }
            if let Ok(rel) = p.strip_prefix(dir) {
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    out.sort();
    out
}

#[test]
fn indexes_a_share_hosted_corpus_with_state_on_local_disk() {
    let Some(corpus) = share_corpus() else {
        eprintln!(
            "SKIPPED: set GLOSSA_TEST_SHARE_ROOT to a writable directory reached through a share \
             (UNC on Windows, a mount point on Linux) to run this test"
        );
        return;
    };

    // Two documents, each with a marker term that exists nowhere else, one of them in a
    // subdirectory so the relative-key arithmetic is exercised rather than assumed.
    std::fs::write(
        corpus.join("manual.md"),
        "# Pump manual\n\nThe sharealpha_marker term appears only in this file.\n",
    )
    .expect("writing the first document to the share");
    std::fs::create_dir_all(corpus.join("sub")).expect("creating a subdirectory on the share");
    std::fs::write(
        corpus.join("sub").join("notes.md"),
        "# Notes\n\nThe sharebeta_marker term appears only in the nested file.\n",
    )
    .expect("writing the nested document to the share");

    let before = listing(&corpus);
    let state = state_dir();

    let out = assert_cmd::Command::cargo_bin("kb")
        .unwrap()
        .arg("index")
        .arg("--root")
        .arg(format!("share={}", corpus.display()))
        .arg("--state-dir")
        .arg(state.path())
        .output()
        .expect("running kb index against the share");
    assert!(
        out.status.success(),
        "kb index failed on a share-hosted corpus:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // The state landed on local disk, and the ONLY thing `kb index` adds to the share is the
    // default `.ignore` it seeds on a corpus that has none. An earlier version of this test
    // asserted the share was untouched; that was simply false, and because the test is opt-in it
    // had never run to say so. The distinction matters to an operator: `--state-dir` keeps the
    // INDEX off the share, it does not make indexing read-only, and on a share that really is
    // read-only the seed fails (audibly, now) and the walk proceeds with no whitelist.
    assert!(
        state.path().join(".glossa").is_dir(),
        "no .glossa under --state-dir: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        !corpus.join(".glossa").exists(),
        "kb index wrote .glossa into the share-hosted corpus"
    );
    let added: Vec<String> = listing(&corpus)
        .into_iter()
        .filter(|p| !before.contains(p))
        .collect();
    assert_eq!(
        added,
        vec![".ignore".to_string()],
        "kb index added something other than the seeded .ignore to the share"
    );

    // Both documents are retrievable, which is what the UNC-prefix defect broke: the root resolved
    // to a relative path, so every document key pointed at a file that did not exist.
    for (term, want) in [
        ("sharealpha_marker", "share/manual.md"),
        ("sharebeta_marker", "share/sub/notes.md"),
    ] {
        let found = assert_cmd::Command::cargo_bin("kb")
            .unwrap()
            .arg("search")
            .arg(term)
            .arg("--state-dir")
            .arg(state.path())
            .arg("--root")
            .arg(format!("share={}", corpus.display()))
            .output()
            .expect("running kb search against the share");
        let stdout = String::from_utf8_lossy(&found.stdout).to_string();
        assert!(
            found.status.success(),
            "kb search failed for {term}:\n{}",
            String::from_utf8_lossy(&found.stderr)
        );
        assert!(
            stdout.contains(want),
            "search for {term} did not return {want}; got:\n{stdout}"
        );
    }

    // `read` resolves the stored key back to a file ON THE SHARE and returns its text — the step
    // that needs the root to still be an absolute path after canonicalization.
    let read = assert_cmd::Command::cargo_bin("kb")
        .unwrap()
        .arg("read")
        .arg("share/sub/notes.md#1")
        .arg("--state-dir")
        .arg(state.path())
        .arg("--root")
        .arg(format!("share={}", corpus.display()))
        .output()
        .expect("running kb read against the share");
    let body = String::from_utf8_lossy(&read.stdout).to_string();
    assert!(
        read.status.success() && body.contains("sharebeta_marker"),
        "kb read could not resolve a share-hosted chunk back to its file:\n{body}\n{}",
        String::from_utf8_lossy(&read.stderr)
    );

    let _ = std::fs::remove_dir_all(&corpus);
}
