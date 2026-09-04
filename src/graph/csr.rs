//! Out-of-core storage for the PPR transition graph: a binary CSR (compressed sparse row) matrix
//! plus an `fst`-backed id→index map and a string table, all memory-mapped read-only so process
//! memory stays independent of corpus size. This is the on-disk twin of `ppr::Transition`
//! (`ids: Vec<String>`, `adj: Vec<Vec<(usize, f32)>>`): `build` writes the mmap-friendly layout
//! from those in-memory parts, `open` maps the files back without loading the matrix onto the
//! heap.
//!
//! On-disk layout (all integers little-endian). All three files are **sig-scoped** — their names
//! embed the `content_sig` (`ppr_transition.<sig:016x>.<ext>`) so a rebuild for a changed graph
//! writes a NEW set and never overwrites the files a live reader is mmapping (the torn-read hazard).
//! - `ppr_transition.<sig>.csr`: header `{magic:u32, version:u32, sig:u64, n:u64, e2:u64}` followed
//!   by `offsets:[u64; n+1]`, `cols:[u32; e2]`, `weights:[f32; e2]` (`e2` = 2 × edge count, since the
//!   adjacency is undirected and stored both directions).
//! - `ppr_transition.<sig>.fst`: raw `fst::Map` bytes, id → row index.
//! - `ppr_transition.<sig>.idtab`: `[u64; n+1]` byte offsets into a trailing blob of concatenated id
//!   strings (row `i`'s id is the byte range `offsets[i]..offsets[i+1]`).
//!
//! `content_sig` is an opaque caller-supplied fingerprint of the source graph (e.g. a hash over
//! node/edge content); `open` returns `None` when the files for that sig are absent, which is the
//! cache-invalidation signal for callers to rebuild.
//!
//! **Atomic publish:** `build` writes each part to a per-process temp file, then renames it into
//! place — renaming the `.csr` LAST. Because `open` checks (and header-validates) the `.csr` first,
//! the appearance of the final `.csr` means all three parts are already published, so a concurrent
//! `open` sees either the complete previous set or the complete new one, never a mix. After
//! publishing, `build` best-effort deletes stale-sig and legacy fixed-name files.

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const MAGIC: u32 = 0x4353_5231; // "CSR1"
const VERSION: u32 = 1;

/// Shared basename of the three CSR part files; the full name is `<PREFIX>.<sig:016x>.<ext>`.
const FILE_PREFIX: &str = "ppr_transition";

/// Final path of a sig-scoped part file (`ext` ∈ {`csr`, `fst`, `idtab`}).
fn part_path(dir: &Path, sig: u64, ext: &str) -> PathBuf {
    dir.join(format!("{FILE_PREFIX}.{sig:016x}.{ext}"))
}

/// Per-process temp path a part is written to before its atomic rename into `part_path`. The pid
/// suffix keeps two processes' temps distinct even if the build lock is unavailable (degraded mode).
fn tmp_path(dir: &Path, sig: u64, ext: &str) -> PathBuf {
    dir.join(format!(
        "{FILE_PREFIX}.{sig:016x}.{ext}.{}.tmp",
        std::process::id()
    ))
}

/// Best-effort removal of every CSR part file that is NOT for `keep_sig` — stale sigs from earlier
/// builds and legacy fixed-name files (`ppr_transition.csr` etc.) from before sig-scoping. Ignores
/// errors: on Windows a file still mmapped by a live reader cannot be deleted, and that is fine —
/// the reader keeps a valid snapshot and the next build retries the cleanup. Never touches `.tmp`
/// files (a concurrent builder may be mid-write) or the `keep_sig` set.
fn cleanup_stale(dir: &Path, keep_sig: u64) {
    let keep: [String; 3] = ["csr", "fst", "idtab"].map(|e| {
        part_path(dir, keep_sig, e)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string()
    });
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with(FILE_PREFIX) {
            continue;
        }
        let is_part = name.ends_with(".csr") || name.ends_with(".fst") || name.ends_with(".idtab");
        if !is_part || keep.iter().any(|k| k == name) {
            continue;
        }
        let _ = std::fs::remove_file(entry.path());
    }
}

/// Header size in bytes: magic(4) + version(4) + sig(8) + n(8) + e2(8).
const HEADER_LEN: usize = 4 + 4 + 8 + 8 + 8;

/// A memory-mapped, read-only CSR transition graph: neighbor lists for `neighbors(i)` are sliced
/// directly out of the mmap, so no allocation grows with corpus size on the read path. The id fst
/// map and string table are mmapped alongside for `idx_of`/`id_of` lookups.
pub struct CsrTransition {
    csr: Mmap,
    idtab: Mmap,
    ids_fst: fst::Map<Mmap>,
    n: usize,
    /// Byte offset in `csr` where the `offsets:[u64; n+1]` array begins.
    offsets_pos: usize,
    /// Byte offset in `csr` where the `cols:[u32; e2]` array begins.
    cols_pos: usize,
    /// Byte offset in `csr` where the `weights:[f32; e2]` array begins.
    weights_pos: usize,
}

/// Iterator over one row's `(neighbor_idx, weight)` pairs, borrowed directly from the mmap.
pub struct NeighborIter<'a> {
    cols: &'a [u8],
    weights: &'a [u8],
    pos: usize,
    len: usize,
}

impl<'a> Iterator for NeighborIter<'a> {
    type Item = (u32, f32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.len {
            return None;
        }
        let byte = self.pos * 4;
        let col = read_u32_le(&self.cols[byte..byte + 4]);
        let weight = read_f32_le(&self.weights[byte..byte + 4]);
        self.pos += 1;
        Some((col, weight))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.pos;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for NeighborIter<'_> {}

fn read_u32_le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("4-byte slice"))
}

fn read_u64_le(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("8-byte slice"))
}

fn read_f32_le(bytes: &[u8]) -> f32 {
    f32::from_le_bytes(bytes.try_into().expect("4-byte slice"))
}

impl CsrTransition {
    /// Build the on-disk CSR + fst + idtab files from an in-memory adjacency list, the same shape
    /// as `ppr::Transition` (`ids`, `adj`). `content_sig` is stamped into the header for
    /// `open`'s staleness check.
    pub fn build(
        ids: &[String],
        adj: &[Vec<(usize, f32)>],
        dir: &Path,
        content_sig: u64,
    ) -> Result<()> {
        if ids.len() != adj.len() {
            bail!(
                "csr build: ids.len()={} != adj.len()={}",
                ids.len(),
                adj.len()
            );
        }
        let n = ids.len();
        let e2: usize = adj.iter().map(|row| row.len()).sum();

        // --- .csr: header + offsets + cols + weights ---
        let mut offsets: Vec<u64> = Vec::with_capacity(n + 1);
        let mut cols: Vec<u32> = Vec::with_capacity(e2);
        let mut weights: Vec<f32> = Vec::with_capacity(e2);
        let mut running: u64 = 0;
        offsets.push(0);
        for row in adj {
            for &(neighbor, w) in row {
                cols.push(neighbor as u32);
                weights.push(w);
            }
            running += row.len() as u64;
            offsets.push(running);
        }

        // Each part is written to a temp file whose handle is dropped (closed) BEFORE the rename —
        // renaming a still-open file trips a Windows sharing violation.
        let csr_tmp = tmp_path(dir, content_sig, "csr");
        {
            let file = File::create(&csr_tmp)
                .with_context(|| format!("creating {}", csr_tmp.display()))?;
            let mut w = BufWriter::new(file);
            w.write_all(&MAGIC.to_le_bytes())?;
            w.write_all(&VERSION.to_le_bytes())?;
            w.write_all(&content_sig.to_le_bytes())?;
            w.write_all(&(n as u64).to_le_bytes())?;
            w.write_all(&(e2 as u64).to_le_bytes())?;
            for off in &offsets {
                w.write_all(&off.to_le_bytes())?;
            }
            for c in &cols {
                w.write_all(&c.to_le_bytes())?;
            }
            for wt in &weights {
                w.write_all(&wt.to_le_bytes())?;
            }
            w.flush()?;
        }

        // --- .fst: id -> idx, keys must be inserted in sorted byte order ---
        let mut sorted: Vec<(&str, u32)> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i as u32))
            .collect();
        sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));

        let fst_tmp = tmp_path(dir, content_sig, "fst");
        {
            let fst_file = File::create(&fst_tmp)
                .with_context(|| format!("creating {}", fst_tmp.display()))?;
            let fst_writer = BufWriter::new(fst_file);
            let mut builder = fst::MapBuilder::new(fst_writer)?;
            for (id, idx) in &sorted {
                builder.insert(id.as_bytes(), *idx as u64)?;
            }
            builder.finish()?; // flushes + closes the underlying writer
        }

        // --- .idtab: [u64; n+1] byte offsets + concatenated id bytes, in original row order ---
        let mut id_offsets: Vec<u64> = Vec::with_capacity(n + 1);
        let mut running: u64 = 0;
        id_offsets.push(0);
        for id in ids {
            running += id.len() as u64;
            id_offsets.push(running);
        }

        let idtab_tmp = tmp_path(dir, content_sig, "idtab");
        {
            let idtab_file = File::create(&idtab_tmp)
                .with_context(|| format!("creating {}", idtab_tmp.display()))?;
            let mut w = BufWriter::new(idtab_file);
            for off in &id_offsets {
                w.write_all(&off.to_le_bytes())?;
            }
            for id in ids {
                w.write_all(id.as_bytes())?;
            }
            w.flush()?;
        }

        // Publish atomically: rename `.fst` and `.idtab` FIRST, then `.csr` LAST. `open` gates on the
        // `.csr` (existence + header) before touching the others, so a `.csr` present ⟺ all three
        // published — a concurrent reader never sees a mixed/partial set.
        std::fs::rename(&fst_tmp, part_path(dir, content_sig, "fst"))
            .with_context(|| format!("publishing {}", fst_tmp.display()))?;
        std::fs::rename(&idtab_tmp, part_path(dir, content_sig, "idtab"))
            .with_context(|| format!("publishing {}", idtab_tmp.display()))?;
        std::fs::rename(&csr_tmp, part_path(dir, content_sig, "csr"))
            .with_context(|| format!("publishing {}", csr_tmp.display()))?;

        // Reclaim disk from superseded sigs / legacy fixed-name files (best-effort; see fn docs).
        cleanup_stale(dir, content_sig);
        Ok(())
    }

    /// Memory-map the CSR + fst + idtab files and validate the header. Returns `Ok(None)` when
    /// the files are missing or the stored `content_sig`/version doesn't match — the caller's
    /// signal to rebuild via `build` rather than trust a stale cache.
    pub fn open(dir: &Path, content_sig: u64) -> Result<Option<CsrTransition>> {
        // Sig-scoped names: absence just means "not built for this sig" → rebuild signal.
        let csr_path = part_path(dir, content_sig, "csr");
        let fst_path = part_path(dir, content_sig, "fst");
        let idtab_path = part_path(dir, content_sig, "idtab");
        if !csr_path.exists() || !fst_path.exists() || !idtab_path.exists() {
            return Ok(None);
        }

        let csr_file =
            File::open(&csr_path).with_context(|| format!("opening {}", csr_path.display()))?;
        // SAFETY: read-only, file-backed mapping (not anonymous, not MmapMut) so the OS pages it
        // in lazily from disk and it is not charged against the Windows commit limit. The mapped
        // file is not mutated by any other handle for the lifetime of this map in normal use.
        let csr = unsafe { Mmap::map(&csr_file)? };
        if csr.len() < HEADER_LEN {
            return Ok(None);
        }

        let magic = read_u32_le(&csr[0..4]);
        let version = read_u32_le(&csr[4..8]);
        let sig = read_u64_le(&csr[8..16]);
        let n = read_u64_le(&csr[16..24]) as usize;
        let e2 = read_u64_le(&csr[24..32]) as usize;

        if magic != MAGIC || version != VERSION || sig != content_sig {
            return Ok(None);
        }

        let offsets_pos = HEADER_LEN;
        let cols_pos = offsets_pos + (n + 1) * 8;
        let weights_pos = cols_pos + e2 * 4;
        let expected_len = weights_pos + e2 * 4;
        if csr.len() < expected_len {
            bail!(
                "csr file truncated: have {} bytes, expected at least {}",
                csr.len(),
                expected_len
            );
        }

        let idtab_file =
            File::open(&idtab_path).with_context(|| format!("opening {}", idtab_path.display()))?;
        // SAFETY: same read-only file-backed contract as `csr` above.
        let idtab = unsafe { Mmap::map(&idtab_file)? };

        let fst_file =
            File::open(&fst_path).with_context(|| format!("opening {}", fst_path.display()))?;
        // SAFETY: same read-only file-backed contract as `csr` above.
        let fst_mmap = unsafe { Mmap::map(&fst_file)? };
        let ids_fst = fst::Map::new(fst_mmap)
            .with_context(|| format!("parsing fst map at {}", fst_path.display()))?;

        Ok(Some(CsrTransition {
            csr,
            idtab,
            ids_fst,
            n,
            offsets_pos,
            cols_pos,
            weights_pos,
        }))
    }

    /// Number of nodes (rows) in the graph.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    fn offset(&self, i: usize) -> u64 {
        let byte = self.offsets_pos + i * 8;
        read_u64_le(&self.csr[byte..byte + 8])
    }

    /// Neighbors of row `i` as `(idx, weight)` pairs, sliced directly from the mmap (no
    /// allocation of the full matrix). Returns an empty iterator for an out-of-range `i`.
    pub fn neighbors(&self, i: u32) -> NeighborIter<'_> {
        let i = i as usize;
        if i >= self.n {
            return NeighborIter {
                cols: &[],
                weights: &[],
                pos: 0,
                len: 0,
            };
        }
        let start = self.offset(i) as usize;
        let end = self.offset(i + 1) as usize;
        let len = end - start;
        let cols_start = self.cols_pos + start * 4;
        let cols_end = self.cols_pos + end * 4;
        let weights_start = self.weights_pos + start * 4;
        let weights_end = self.weights_pos + end * 4;
        NeighborIter {
            cols: &self.csr[cols_start..cols_end],
            weights: &self.csr[weights_start..weights_end],
            pos: 0,
            len,
        }
    }

    /// Row index for a node id, via the mmapped fst map (no full id-table scan).
    pub fn idx_of(&self, id: &str) -> Option<u32> {
        self.ids_fst.get(id.as_bytes()).map(|v| v as u32)
    }

    /// Node id for a row index, sliced out of the mmapped id table.
    pub fn id_of(&self, i: u32) -> Option<&str> {
        let i = i as usize;
        if i >= self.n {
            return None;
        }
        let offsets_len = (self.n + 1) * 8;
        let start = read_u64_le(&self.idtab[i * 8..i * 8 + 8]) as usize;
        let end = read_u64_le(&self.idtab[(i + 1) * 8..(i + 1) * 8 + 8]) as usize;
        let blob = &self.idtab[offsets_len..];
        std::str::from_utf8(&blob[start..end]).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_roundtrips_neighbors_and_id_maps() {
        let dir = tempfile::tempdir().unwrap();
        let ids = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        // undirected adjacency: a-b (0.5), b-c (1.0)
        let adj = vec![
            vec![(1usize, 0.5f32)],
            vec![(0usize, 0.5f32), (2usize, 1.0f32)],
            vec![(1usize, 1.0f32)],
        ];
        CsrTransition::build(&ids, &adj, dir.path(), 42).unwrap();
        let csr = CsrTransition::open(dir.path(), 42)
            .unwrap()
            .expect("fresh sig opens");
        assert_eq!(csr.len(), 3);
        assert_eq!(csr.idx_of("b"), Some(1));
        assert_eq!(csr.id_of(1), Some("b"));
        let mut nb: Vec<(u32, f32)> = csr.neighbors(1).collect();
        nb.sort_by_key(|x| x.0);
        assert_eq!(nb, vec![(0, 0.5), (2, 1.0)]);
        // stale sig → None (cache invalidation)
        assert!(CsrTransition::open(dir.path(), 43).unwrap().is_none());
    }

    /// Windows-safety: `open` must produce read-only, file-backed mmaps (not anonymous, not
    /// writable) so the mapping isn't charged against the Windows commit limit. This test just
    /// exercises build -> open -> use -> drop on that path without panicking; the real guarantee
    /// is the `Mmap::map` (not `MmapMut`, not `map_anon`) call in `open` above.
    #[test]
    fn csr_open_is_read_only_mmap_backed() {
        let dir = tempfile::tempdir().unwrap();
        let ids = vec!["x".to_string(), "y".to_string()];
        let adj = vec![vec![(1usize, 1.0f32)], vec![(0usize, 1.0f32)]];
        CsrTransition::build(&ids, &adj, dir.path(), 7).unwrap();

        let csr = CsrTransition::open(dir.path(), 7).unwrap().expect("opens");
        assert_eq!(csr.len(), 2);
        // Touch every mapped region to force the OS to actually page it in read-only.
        let _ = csr.idx_of("x");
        let _ = csr.id_of(0);
        let _: Vec<(u32, f32)> = csr.neighbors(0).collect();
        drop(csr);

        // The underlying files must remain intact and reopenable after the mmap is dropped.
        assert!(CsrTransition::open(dir.path(), 7).unwrap().is_some());
    }

    #[test]
    fn csr_open_returns_none_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        assert!(CsrTransition::open(dir.path(), 1).unwrap().is_none());
    }

    #[test]
    fn rebuild_for_new_sig_keeps_old_reader_valid_and_publishes_new() {
        let dir = tempfile::tempdir().unwrap();
        let ids = vec!["a".to_string(), "b".to_string()];
        let adj = vec![vec![(1usize, 1.0f32)], vec![(0usize, 1.0f32)]];
        CsrTransition::build(&ids, &adj, dir.path(), 100).unwrap();
        let old = CsrTransition::open(dir.path(), 100)
            .unwrap()
            .expect("sig 100 opens");
        // A "graph changed" rebuild under a NEW sig writes a fresh file set and must NOT disturb the
        // live `old` reader (the sig-scoped names + RCU guarantee that underpins the shared handle).
        CsrTransition::build(&ids, &adj, dir.path(), 200).unwrap();
        assert_eq!(old.idx_of("a"), Some(0));
        assert_eq!(old.id_of(1), Some("b"));
        assert_eq!(old.neighbors(0).count(), 1);
        assert!(
            CsrTransition::open(dir.path(), 200).unwrap().is_some(),
            "the new sig is published and openable"
        );
    }

    #[test]
    fn build_cleans_stale_sigs_and_legacy_and_leaves_no_temp() {
        let dir = tempfile::tempdir().unwrap();
        let ids = vec!["x".to_string()];
        let adj: Vec<Vec<(usize, f32)>> = vec![vec![]];
        // A legacy fixed-name file from before sig-scoping must be reclaimed on the next build.
        std::fs::write(dir.path().join("ppr_transition.csr"), b"legacy").unwrap();

        CsrTransition::build(&ids, &adj, dir.path(), 1).unwrap();
        // No reader holds sig 1, so the next build's cleanup can delete it cross-platform.
        CsrTransition::build(&ids, &adj, dir.path(), 2).unwrap();

        assert!(
            CsrTransition::open(dir.path(), 1).unwrap().is_none(),
            "superseded sig 1 was cleaned up"
        );
        assert!(
            CsrTransition::open(dir.path(), 2).unwrap().is_some(),
            "current sig 2 is present"
        );
        assert!(
            !dir.path().join("ppr_transition.csr").exists(),
            "legacy fixed-name file was cleaned up"
        );
        let temps: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(
            temps.is_empty(),
            "no temp files left after publish: {temps:?}"
        );
    }
}
