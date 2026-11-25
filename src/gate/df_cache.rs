//! Process-global cache of the live, index-backed `DfTable`, shared as `Arc` and invalidated by
//! the tantivy `Searcher`'s generation id — so a hot verify path on a huge corpus doesn't pay a
//! term-dictionary lookup per token on every call, while never serving numbers from before the
//! index last moved. One corpus per process is the norm (server / eval bind one state_base), so
//! the map holds a single entry; it is keyed defensively.
//!
//! This used to cache the PARSED `.glossa/df` SIDECAR, invalidated by that file's (mtime, len).
//! The sidecar is a once-built aggregate that can only grow (see `gate::df::DfTable`'s doc
//! comment), so it is no longer read here; `df`/`is_rare` now answer from the index itself, which
//! tantivy keeps correct incrementally. The cache key moved with it: a `Searcher::generation_id`
//! changes exactly when the reader has reloaded onto a new commit, which is the one signal that
//! actually means "the cached frequencies may now be wrong."
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};

use crate::gate::df::DfTable;
use crate::index::store::DocIndex;

type DfCache = HashMap<PathBuf, (u64, Arc<DfTable>)>;

fn cache() -> &'static RwLock<DfCache> {
    static C: OnceLock<RwLock<DfCache>> = OnceLock::new();
    C.get_or_init(|| RwLock::new(HashMap::new()))
}

/// The live `DfTable` for `glossa_dir`'s corpus, built from the tantivy index and reused until the
/// reader's generation advances (i.e. a write landed and reloaded). A corpus with no index yet
/// (nothing has ever been `kb index`ed) answers an empty table rather than erroring — unlike the
/// old sidecar-backed version, which errored on a missing file — and does so WITHOUT creating one:
/// `DocIndex::exists` is a read-only check for exactly this, so a verify call on an unindexed
/// corpus has no side effect of its own.
pub fn cached_df(glossa_dir: &Path) -> anyhow::Result<Arc<DfTable>> {
    let root = glossa_dir.parent().unwrap_or(glossa_dir);
    if !DocIndex::exists(root) {
        return Ok(Arc::new(DfTable::new()));
    }
    let idx = DocIndex::open_or_create(root)?;
    let searcher = idx.searcher();
    let gen = searcher.generation().generation_id();
    let key = glossa_dir.to_path_buf();
    if let Some((cached_gen, arc)) = cache().read().unwrap().get(&key) {
        if *cached_gen == gen {
            return Ok(Arc::clone(arc));
        }
    }
    let table = Arc::new(DfTable::from_searcher(searcher, idx.fields.gate_terms));
    cache()
        .write()
        .unwrap()
        .insert(key, (gen, Arc::clone(&table)));
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::cached_df;
    use crate::index::store::DocIndex;
    use crate::model::Chunk;
    use std::path::PathBuf;
    use std::sync::Arc;

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

    #[test]
    fn same_index_returns_same_arc_then_reloads_on_write() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), "hello world").unwrap();
        crate::index::store::index_dir(dir.path(), false).unwrap();
        let glossa_dir = dir.path().join(".glossa");

        let a = cached_df(&glossa_dir).unwrap();
        let b = cached_df(&glossa_dir).unwrap();
        assert!(
            Arc::ptr_eq(&a, &b),
            "cache hit returns the same Arc while the index hasn't moved on"
        );

        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[Chunk::labelled(
            PathBuf::from("b.md"),
            "md".into(),
            String::new(),
            "freshlyaddedterm appears".into(),
        )])
        .unwrap();

        let c = cached_df(&glossa_dir).unwrap();
        assert!(
            !Arc::ptr_eq(&a, &c),
            "a new commit advances the reader's generation; the stale cached table must not be reused"
        );
        assert_eq!(c.df("freshlyaddedterm"), 1);
    }
}
