//! Process-global cache of one long-lived `DocIndex` per corpus root, so a hot verify path
//! doesn't reopen the tantivy index (directory, `meta.json`, segment files) on every call — the
//! actually expensive part. One corpus per process is the norm (server / eval bind one
//! state_base), so the map holds a single entry; it is keyed defensively.
//!
//! This used to cache the PARSED `.glossa/df` SIDECAR, invalidated by that file's (mtime, len).
//! The sidecar was a once-built aggregate that could only grow, so it has since been removed
//! outright (writer and reader both — see `gate::df::DfTable`'s doc comment); `df`/`is_rare`
//! answer from the index itself instead.
//!
//! It also, briefly, cached the resulting `DfTable` itself, keyed by `Searcher::generation_id()`.
//! That was wrong and has been removed: `generation_id` is not derived from index state — it
//! comes from a plain `AtomicU64` counter that `IndexReader` creates fresh (`Default::default()`)
//! on every construction (tantivy `reader/mod.rs`), and `DocIndex::open_or_create` built a fresh
//! `IndexReader` on every call. So the id was 0 on every call, forever: the first `DfTable` ever
//! built got stuck in the cache permanently, which is worse than the sidecar this change replaces
//! (that one at least got rebuilt by `kb index --force`). The fix is to stop opening a fresh
//! index per call at all: hold ONE `DocIndex` (and its one `IndexReader`, whose counter now
//! actually advances across real reloads) per corpus, and force a reload before every lookup
//! instead of trusting the reader's own on-commit watch to have already fired.
//!
//! `IndexReader::reload()` is unconditional — it always builds a new `Searcher` and always bumps
//! the generation counter, whether or not anything actually changed on disk (same tantivy source)
//! — so there is no cheaper "has it changed" check than just reloading, and no reason to cache
//! the resulting `DfTable` across calls either (nothing in it is expensive to rebuild once the
//! per-token memo is gone — see `gate::df::DfTable`'s doc comment on why there isn't one). What
//! IS worth holding onto is the `DocIndex`/`IndexReader` itself, which is why that's what this
//! cache holds.
//!
//! This also fixes an accidental side effect of the old shape: a cached `DfTable` held a
//! `Searcher`, which pins the segment files it references (Windows will refuse to delete a file a
//! live `Searcher` still has open), and a cache that never invalidated therefore never released
//! them either. Reloading this same long-lived reader on every call replaces its ArcSwap'd
//! `Searcher` each time, so the PREVIOUS one (and the hold on its now-superseded segments) is
//! dropped as soon as nothing else references it — pinning is bounded to "whatever the last
//! reload loaded," the same as any other long-lived `DocIndex` in this codebase, not "forever."
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};

use crate::gate::df::DfTable;
use crate::index::store::DocIndex;

type IndexCache = HashMap<PathBuf, Arc<DocIndex>>;

fn cache() -> &'static RwLock<IndexCache> {
    static C: OnceLock<RwLock<IndexCache>> = OnceLock::new();
    C.get_or_init(|| RwLock::new(HashMap::new()))
}

/// The live `DfTable` for `glossa_dir`'s corpus: reuses (or lazily opens) this process's one
/// cached `DocIndex` for the corpus root, forces it to reload onto the latest commit, and builds
/// a fresh table from the resulting `Searcher`.
pub fn cached_df(glossa_dir: &Path) -> anyhow::Result<Arc<DfTable>> {
    let root = glossa_dir.parent().unwrap_or(glossa_dir);
    if !DocIndex::exists(root) {
        // Deliberate, not an oversight: a corpus that has never been `kb index`ed has nothing to
        // ground against yet. That is treated as an empty vocabulary — every answer token reads
        // as df==0 and the gate abstains the normal way (same as any other ungrounded answer) —
        // rather than as a distinct "no index" failure. If a future reader is tempted to turn
        // this into an error: don't, without also deciding what should surface it, since today
        // nothing upstream of `verify` distinguishes "never indexed" from "genuinely ungrounded"
        // and a silently-normal abstain is exactly the kind of defect this project keeps removing
        // — flagging it here is so the next person sees the tradeoff instead of assuming it's new.
        //
        // Returning here, before touching `cache` below, also means a status-only caller never
        // materializes an index just by asking (`DocIndex::exists` is read-only for this reason).
        return Ok(Arc::new(DfTable::new()));
    }
    let key = root.to_path_buf();
    // Read-then-maybe-write, as two separate statements rather than one `match`/`if let` whose
    // scrutinee keeps the read guard alive for the whole arm: a temporary from a match/if-let
    // scrutinee lives until the end of that expression, so taking the write lock inside an arm
    // of a match on the read lock's own result deadlocks (`RwLock` is not reentrant) — this did
    // exactly that on every first-ever call for a corpus until it was caught before landing.
    let existing = cache().read().unwrap().get(&key).cloned();
    let idx = match existing {
        Some(idx) => idx,
        None => {
            let idx = Arc::new(DocIndex::open_or_create(root)?);
            cache().write().unwrap().insert(key, Arc::clone(&idx));
            idx
        }
    };
    // Force the reload rather than trust the reader's own watch to have already fired — see the
    // module doc comment on why this is also the only "has anything changed" check worth having.
    idx.reload()?;
    Ok(Arc::new(DfTable::from_searcher(
        idx.searcher(),
        idx.fields.gate_terms,
    )))
}

#[cfg(test)]
mod tests {
    use super::cached_df;
    use crate::index::store::DocIndex;
    use crate::model::Chunk;
    use std::path::PathBuf;

    #[test]
    fn no_index_yet_answers_empty_without_creating_one() {
        let dir = tempfile::tempdir().unwrap();
        let glossa_dir = dir.path().join(".glossa");
        std::fs::create_dir_all(&glossa_dir).unwrap();

        let t = cached_df(&glossa_dir).unwrap();
        assert_eq!(t.n_chunks, 0);
        assert!(
            !DocIndex::exists(dir.path()),
            "a read-only df lookup on an unindexed corpus must not materialize an index"
        );
    }

    /// The regression this module exists to prevent: a `cached_df` call that reuses an
    /// already-cached `DocIndex` for this corpus must still see a write that landed AFTER that
    /// index was first cached, not serve whatever it saw on the first call forever. (The bug this
    /// pins was exactly that: a `Searcher::generation_id` that never left 0, so the fix had to
    /// stop reopening the index per call and start reloading a held-open one instead.)
    #[test]
    fn reused_cached_index_still_sees_a_later_write() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), "hello world").unwrap();
        crate::index::store::index_dir(dir.path(), false).unwrap();
        let glossa_dir = dir.path().join(".glossa");

        let before = cached_df(&glossa_dir).unwrap();
        assert_eq!(before.df("freshlyaddedterm"), 0);

        // Write through a SEPARATE `DocIndex` handle — standing in for another call path, or
        // another process — landing after this corpus's `DocIndex` is already cached above.
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[Chunk::labelled(
            PathBuf::from("b.md"),
            "md".into(),
            String::new(),
            "freshlyaddedterm appears".into(),
        )])
        .unwrap();

        let after = cached_df(&glossa_dir).unwrap();
        assert_eq!(
            after.df("freshlyaddedterm"),
            1,
            "the cached DocIndex must reload before answering, or this stays 0 forever"
        );
    }
}
