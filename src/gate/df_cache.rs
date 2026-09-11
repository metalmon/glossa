//! Process-global cache of the parsed `DfTable`, shared as `Arc` and invalidated by the sidecar's
//! (mtime, len) signature — so a hot verify path on a huge corpus loads the whole-vocabulary df
//! sidecar once, not on every call. See spec 2.1b. One corpus per process is the norm (server /
//! eval bind one state_base), so the map holds a single entry; it is keyed defensively.
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::SystemTime;

use crate::gate::df::DfTable;

type Sig = (Option<SystemTime>, u64);
type DfCache = HashMap<PathBuf, (Sig, Arc<DfTable>)>;

fn cache() -> &'static RwLock<DfCache> {
    static C: OnceLock<RwLock<DfCache>> = OnceLock::new();
    C.get_or_init(|| RwLock::new(HashMap::new()))
}

fn sig_of(path: &Path) -> Sig {
    match std::fs::metadata(path) {
        Ok(m) => (m.modified().ok(), m.len()),
        Err(_) => (None, 0),
    }
}

/// The parsed `DfTable` for `glossa_dir`'s `df` sidecar, loaded once and reused until the sidecar's
/// (mtime, len) changes (i.e. `kb index` rewrote it). Errors exactly like `DfTable::load` when there
/// is no readable sidecar and nothing is cached yet.
pub fn cached_df(glossa_dir: &Path) -> anyhow::Result<Arc<DfTable>> {
    let path = DfTable::sidecar_path(glossa_dir);
    let sig = sig_of(&path);
    if let Some((cached_sig, arc)) = cache().read().unwrap().get(&path) {
        if *cached_sig == sig {
            return Ok(Arc::clone(arc));
        }
    }
    let table = Arc::new(DfTable::load(&path)?);
    cache()
        .write()
        .unwrap()
        .insert(path, (sig, Arc::clone(&table)));
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::cached_df;
    use crate::gate::df::DfTable;

    #[test]
    fn same_sidecar_returns_same_arc_then_reloads_on_change() {
        let dir = tempfile::tempdir().unwrap();
        let glossa_dir = dir.path().join(".glossa");
        std::fs::create_dir_all(&glossa_dir).unwrap();
        let sc = DfTable::sidecar_path(&glossa_dir);
        DfTable::new().save(&sc).unwrap();

        let a = cached_df(&glossa_dir).unwrap();
        let b = cached_df(&glossa_dir).unwrap();
        assert!(std::sync::Arc::ptr_eq(&a, &b), "cache hit returns the same Arc");

        // Change the sidecar: content differs -> (mtime,len) signature differs -> reload.
        let mut t = DfTable::new();
        t.add_chunk(&["alpha".into(), "beta".into()]);
        t.save(&sc).unwrap();
        let c = cached_df(&glossa_dir).unwrap();
        assert!(!std::sync::Arc::ptr_eq(&a, &c), "changed sidecar reloads");
        assert_eq!(c.n_chunks, 1);
    }
}
