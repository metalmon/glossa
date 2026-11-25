use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tantivy::schema::Field;
use tantivy::{Searcher, Term};

/// Where a `DfTable` gets its numbers from.
///
/// `Map` is the legacy in-memory aggregate: built by `add_chunk` during a full/`--force` index
/// pass (the `.glossa/df` sidecar writer — `index::store`), and by every test in this module and
/// in `score.rs`/`nli.rs` that wants a table with no real index behind it at all (`DfTable::new`).
/// `save`/`load` still read and write it; the sidecar itself is deliberately kept
/// written-but-unread by production code (cheap rollback for one release) rather than removed in
/// this change.
///
/// `Index` is the live path: `df`/`is_rare` answer from tantivy's own term dictionary via
/// `Searcher::doc_freq`, which is always consistent with whatever the index currently holds —
/// incrementally maintained, correctly, by tantivy itself, so it never goes stale the way the
/// once-built sidecar aggregate did.
///
/// No per-token memo here (an earlier version of this change had one, keyed by token string):
/// `score`'s answer tokenization and `nli`'s per-claim `mean_filt` are each at most a few hundred
/// `doc_freq` lookups per `verify` call — microseconds against a memory-mapped term dictionary,
/// not a cost worth caching. A cache would also be unbounded: the answer text is model-generated,
/// so a long-lived server would grow a `HashMap<String, u32>` by every distinct token any answer
/// ever used, with no eviction — a real memory-growth shape for no measured benefit.
enum Source {
    Map(HashMap<String, u32>),
    Index { searcher: Searcher, field: Field },
}

/// token → number of corpus chunks containing it, plus the chunk count. See [`Source`] for the
/// two ways this answers a lookup.
pub struct DfTable {
    pub n_chunks: u32,
    source: Source,
}

impl Default for DfTable {
    fn default() -> Self {
        Self::new()
    }
}

impl DfTable {
    /// An empty, in-memory table. Used by the sidecar writer (via `add_chunk`) and by every test
    /// that exercises gate scoring without a real tantivy index behind it.
    pub fn new() -> Self {
        DfTable {
            n_chunks: 0,
            source: Source::Map(HashMap::new()),
        }
    }

    /// The live, index-backed table: `df`/`is_rare` read `searcher`'s term dictionary for `field`
    /// (the gate's own `gate_terms` field — see `index::store::build_schema`).
    ///
    /// `n_chunks` is the sum of `max_doc()` over every segment, deliberately NOT
    /// `Searcher::num_docs()`: `num_docs` counts only ALIVE documents, while `doc_freq` reads the
    /// term dictionary as written at segment-build time and never consults the delete bitset —
    /// pairing the two would inflate `df / n_chunks` from both ends as a corpus accumulates edits.
    /// `max_doc` puts both numbers on the same (upper-bound) basis.
    pub fn from_searcher(searcher: Searcher, field: Field) -> Self {
        let n_chunks: u32 = searcher.segment_readers().iter().map(|r| r.max_doc()).sum();
        DfTable {
            n_chunks,
            source: Source::Index { searcher, field },
        }
    }

    /// Only meaningful on the `Map` source (the sidecar writer's full/`--force` pass). A no-op on
    /// `Index` — nothing calls this on a live table, since tantivy maintains its own postings.
    pub fn add_chunk(&mut self, tokens: &[String]) {
        if let Source::Map(map) = &mut self.source {
            self.n_chunks += 1;
            let mut seen = std::collections::HashSet::new();
            for tok in tokens {
                if seen.insert(tok.as_str()) {
                    *map.entry(tok.clone()).or_insert(0) += 1;
                }
            }
        }
    }

    pub fn df(&self, token: &str) -> u32 {
        match &self.source {
            Source::Map(map) => map.get(token).copied().unwrap_or(0),
            Source::Index { searcher, field } => {
                // Measured, not assumed (an earlier version of this comment claimed the opposite
                // and was wrong — see `deleting_a_segments_only_live_chunk_drops_its_terms_to_zero`
                // for what actually happens). `InvertedIndexReader::doc_freq` does read the term
                // dictionary as written at segment-build time, ignoring the delete bitset — true
                // in isolation — but that is NOT the whole story: a segment whose live-doc count
                // reaches exactly 0 is dropped from the committed segment list on ITS OWN next
                // commit (`SegmentManager::remove_empty_segments`, called from `save_metas` on
                // EVERY commit, not only a merge). So a term's count is not pinned until a merge —
                // it can drop the moment every segment holding it empties out. What stays stale
                // until an actual merge rewrites it is narrower: a document deleted from a segment
                // that still has OTHER live documents, where the segment itself isn't pruned and
                // its dictionary entry is never adjusted. Either way, a returned 0 is trustworthy
                // for the presence question — nothing here can manufacture a zero for a term that
                // is genuinely live in the index; only a positive count can still be stale-high.
                let t = Term::from_field_text(*field, token);
                searcher.doc_freq(&t).unwrap_or(0) as u32
            }
        }
    }

    /// Vocabulary size — only meaningful on `Map` (the sidecar writer's progress message reads
    /// it). `Index` has no cheap equivalent (it would mean walking the whole term dictionary) and
    /// nothing needs it there, so it reads 0/empty.
    pub fn len(&self) -> usize {
        match &self.source {
            Source::Map(map) => map.len(),
            Source::Index { .. } => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        match &self.source {
            Source::Map(map) => map.is_empty(),
            Source::Index { .. } => true,
        }
    }

    pub fn is_rare(&self, token: &str, rare_df_frac: f32) -> bool {
        let d = self.df(token);
        d == 0 || (d as f32) <= rare_df_frac * (self.n_chunks as f32)
    }

    pub fn sidecar_path(glossa_dir: &Path) -> PathBuf {
        glossa_dir.join("df")
    }

    /// Only ever called on a `Map`-sourced table (the sidecar writer). An `Index`-sourced table
    /// has nothing of its own to persist — erroring here is simpler and safer than silently
    /// writing an empty/misleading sidecar.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let Source::Map(map) = &self.source else {
            return Err(std::io::Error::other(
                "DfTable::save: an index-backed table has nothing to persist",
            ));
        };
        // line format: "<n_chunks>\n" then "<df>\t<token>\n" per entry
        let mut s = format!("{}\n", self.n_chunks);
        for (tok, d) in map {
            s.push_str(&format!("{d}\t{tok}\n"));
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        crate::index::manifest::atomic_write(path, s.as_bytes())
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let mut lines = raw.lines();
        let n_chunks = lines
            .next()
            .and_then(|l| l.trim().parse().ok())
            .unwrap_or(0);
        let mut map = HashMap::new();
        for l in lines {
            if let Some((d, tok)) = l.split_once('\t') {
                if let Ok(d) = d.parse::<u32>() {
                    map.insert(tok.to_string(), d);
                }
            }
        }
        Ok(DfTable {
            n_chunks,
            source: Source::Map(map),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::DfTable;
    use crate::gate::token::tokenize;
    use crate::index::store::DocIndex;

    #[test]
    fn rarity_and_roundtrip() {
        let mut t = DfTable::new();
        for page in [
            "network modbus setup",
            "modbus and modbus again",
            "channel calibration pp.19.00.00.00",
        ] {
            t.add_chunk(&tokenize(page));
        }
        assert_eq!(t.n_chunks, 3);
        assert_eq!(t.df("modbus"), 2); // on 2 of 3 chunks
        assert_eq!(t.df("pp.19.00.00.00"), 1); // rare code
        assert!(t.is_rare("pp.19.00.00.00", 0.5)); // 1 <= 0.5*3
        assert!(!t.is_rare("modbus", 0.5)); // 2 > 1.5
        assert!(t.is_rare("neverseen", 0.5)); // df==0 ⇒ rare
        let dir = tempfile::tempdir().unwrap();
        let p = DfTable::sidecar_path(dir.path());
        t.save(&p).unwrap();
        let back = DfTable::load(&p).unwrap();
        assert_eq!(back.n_chunks, 3);
        assert_eq!(back.df("modbus"), 2);
    }

    /// THE defect this change fixes, pinned directly: under the old once-built sidecar, a term
    /// that only ever appears in a document added AFTER the last full/`--force` pass read as
    /// df == 0 (frequency zero) while genuinely present in the corpus, because an aggregate sum
    /// has no per-document decrement and incremental passes therefore left it untouched. Reading
    /// from tantivy's own (incrementally correct) index instead means an incrementally-added
    /// document's terms are visible immediately, with no `--force` required.
    #[test]
    fn incrementally_added_term_is_visible_with_no_force() {
        let dir = tempfile::tempdir().unwrap();
        // First pass: brand-new corpus, so this is schema-migrate-equivalent (no manifest yet) —
        // irrelevant to what's under test, which is the SECOND, genuinely incremental pass below.
        std::fs::write(dir.path().join("a.md"), "the first document only").unwrap();
        crate::index::store::index_dir(dir.path(), false).unwrap();

        // Second pass: force = false, and the manifest is already at the current schema, so this
        // is a real incremental pass — `delta.changed` is just the new file, and the old sidecar
        // code path (`df = None`) would never have touched `.glossa/df` here.
        std::fs::write(
            dir.path().join("b.md"),
            "a brandnewincrementalterm appears only here",
        )
        .unwrap();
        crate::index::store::index_dir(dir.path(), false).unwrap();

        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let df = DfTable::from_searcher(idx.searcher(), idx.fields.gate_terms);
        assert_eq!(
            df.df("brandnewincrementalterm"),
            1,
            "a term added by an incremental (non-force) pass must be visible immediately, not \
             read back as the old sidecar's frozen df==0"
        );
    }

    /// What deleting a chunk actually does — measured, after an earlier version of this test
    /// asserted the opposite (expected 1, got 0) on a claimed mechanism ("stays until a merge")
    /// that turned out to be wrong. Two things ARE true in isolation: `doc_freq` itself never
    /// consults the delete bitset (`InvertedIndexReader::doc_freq` reads only the term dictionary
    /// as written at segment-build time), and tantivy's merge policy never merges on a deletion
    /// alone (its delete-ratio trigger needs ratio > 1.0, which a ratio bounded by 1 can never
    /// reach). But neither of those is what governs this test: a SEPARATE, unconditional step
    /// runs on every commit, not only a merge — `SegmentUpdater::save_metas` calls
    /// `SegmentManager::committed_segment_metas`, which calls `remove_empty_segments` FIRST,
    /// dropping any committed segment whose `num_docs()` is now exactly 0 before meta.json is
    /// even written. This chunk is the ONLY document in its (freshly created) segment, so
    /// deleting it drops that segment's live-doc count to 0, and the delete's OWN commit prunes
    /// the segment on the spot — no merge, no delay. `doc_freq` then sums over zero segments for
    /// this term, because the segment holding its (otherwise untouched) postings is no longer
    /// part of the searchable set at all.
    ///
    /// This is deterministic for the case pinned here — a document that is the LAST live
    /// document in its segment — but NOT a general "every delete zeroes its terms immediately"
    /// claim: a chunk deleted out of a segment that still has OTHER live documents leaves that
    /// segment's `num_docs()` > 0, so it is not pruned, and its term dictionary (never touched by
    /// the delete bitset) keeps reporting whatever it did at build time. That residual-staleness
    /// case is layout-dependent and is NOT pinned here.
    #[test]
    fn deleting_a_segments_only_live_chunk_drops_its_terms_to_zero() {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[crate::model::Chunk::labelled(
            std::path::PathBuf::from("doomed.md"),
            "md".into(),
            String::new(),
            "a onlyheredeletedterm appears".into(),
        )])
        .unwrap();

        let before = DfTable::from_searcher(idx.searcher(), idx.fields.gate_terms);
        assert_eq!(before.df("onlyheredeletedterm"), 1);

        idx.delete_path("doomed.md").unwrap();

        let after = DfTable::from_searcher(idx.searcher(), idx.fields.gate_terms);
        assert_eq!(
            after.df("onlyheredeletedterm"),
            0,
            "the chunk was the only live document in its segment; deleting it drops that \
             segment's num_docs() to 0, which save_metas prunes from meta.json on this same \
             commit — no merge needed"
        );
    }
}
