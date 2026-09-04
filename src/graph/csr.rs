//! Out-of-core storage for the PPR transition graph: a binary CSR (compressed sparse row) matrix
//! plus an `fst`-backed id→index map and a string table, all memory-mapped read-only so process
//! memory stays independent of corpus size. This is the on-disk twin of `ppr::Transition`
//! (`ids: Vec<String>`, `adj: Vec<Vec<(usize, f32)>>`): `build` writes the mmap-friendly layout
//! from those in-memory parts, `open` maps the files back without loading the matrix onto the
//! heap.
//!
//! On-disk layout (all integers little-endian):
//! - `ppr_transition.csr`: header `{magic:u32, version:u32, sig:u64, n:u64, e2:u64}` followed by
//!   `offsets:[u64; n+1]`, `cols:[u32; e2]`, `weights:[f32; e2]` (`e2` = 2 × edge count, since the
//!   adjacency is undirected and stored both directions).
//! - `ppr_transition.fst`: raw `fst::Map` bytes, id → row index.
//! - `ppr_transition.idtab`: `[u64; n+1]` byte offsets into a trailing blob of concatenated id
//!   strings (row `i`'s id is the byte range `offsets[i]..offsets[i+1]`).
//!
//! `content_sig` is an opaque caller-supplied fingerprint of the source graph (e.g. a hash over
//! node/edge content); `open` returns `None` when the stored sig doesn't match, which is the
//! cache-invalidation signal for callers to rebuild.

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const MAGIC: u32 = 0x4353_5231; // "CSR1"
const VERSION: u32 = 1;

const CSR_FILE: &str = "ppr_transition.csr";
const FST_FILE: &str = "ppr_transition.fst";
const IDTAB_FILE: &str = "ppr_transition.idtab";

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

        let csr_path = dir.join(CSR_FILE);
        let file =
            File::create(&csr_path).with_context(|| format!("creating {}", csr_path.display()))?;
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

        // --- .fst: id -> idx, keys must be inserted in sorted byte order ---
        let mut sorted: Vec<(&str, u32)> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i as u32))
            .collect();
        sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));

        let fst_path = dir.join(FST_FILE);
        let fst_file =
            File::create(&fst_path).with_context(|| format!("creating {}", fst_path.display()))?;
        let fst_writer = BufWriter::new(fst_file);
        let mut builder = fst::MapBuilder::new(fst_writer)?;
        for (id, idx) in &sorted {
            builder.insert(id.as_bytes(), *idx as u64)?;
        }
        builder.finish()?;

        // --- .idtab: [u64; n+1] byte offsets + concatenated id bytes, in original row order ---
        let mut id_offsets: Vec<u64> = Vec::with_capacity(n + 1);
        let mut running: u64 = 0;
        id_offsets.push(0);
        for id in ids {
            running += id.len() as u64;
            id_offsets.push(running);
        }

        let idtab_path = dir.join(IDTAB_FILE);
        let idtab_file = File::create(&idtab_path)
            .with_context(|| format!("creating {}", idtab_path.display()))?;
        let mut w = BufWriter::new(idtab_file);
        for off in &id_offsets {
            w.write_all(&off.to_le_bytes())?;
        }
        for id in ids {
            w.write_all(id.as_bytes())?;
        }
        w.flush()?;

        Ok(())
    }

    /// Memory-map the CSR + fst + idtab files and validate the header. Returns `Ok(None)` when
    /// the files are missing or the stored `content_sig`/version doesn't match — the caller's
    /// signal to rebuild via `build` rather than trust a stale cache.
    pub fn open(dir: &Path, content_sig: u64) -> Result<Option<CsrTransition>> {
        let csr_path = dir.join(CSR_FILE);
        let fst_path = dir.join(FST_FILE);
        let idtab_path = dir.join(IDTAB_FILE);
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
}
