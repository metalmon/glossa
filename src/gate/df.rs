use tantivy::schema::Field;
use tantivy::{Searcher, Term};

/// Where a `DfTable` gets its numbers from.
///
/// `Empty` is the zero-vocabulary table: every `df` lookup reads 0, so `is_rare` treats every
/// token as rare. Used by every test in this module and in `score.rs`/`nli.rs` that wants a table
/// with no real index behind it at all (`DfTable::new`), and by `gate::df_cache::cached_df` for a
/// corpus that has never been `kb index`ed (see that module's doc comment). This used to be a real
/// in-memory aggregate (`Map(HashMap<String, u32>)`, built by an `add_chunk` that the old
/// `.glossa/df` sidecar writer called during a full/`--force` index pass, with `save`/`load` to
/// persist it) — removed once nothing read it: production switched to `Index` below, and this
/// type's only remaining non-index use is as an empty placeholder, which needs no map at all.
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
    Empty,
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
    /// The empty, zero-vocabulary table. Used as a placeholder by tests that exercise gate scoring
    /// without a real tantivy index behind it, and by `df_cache::cached_df` for a never-indexed
    /// corpus.
    pub fn new() -> Self {
        DfTable {
            n_chunks: 0,
            source: Source::Empty,
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

    pub fn df(&self, token: &str) -> u32 {
        match &self.source {
            Source::Empty => 0,
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

    /// Vocabulary size. Only ever 0: the `Empty` source has nothing in it, and `Index` has no
    /// cheap equivalent (it would mean walking the whole term dictionary) and nothing needs it
    /// there either.
    pub fn len(&self) -> usize {
        0
    }

    pub fn is_empty(&self) -> bool {
        true
    }

    pub fn is_rare(&self, token: &str, rare_df_frac: f32) -> bool {
        let d = self.df(token);
        d == 0 || (d as f32) <= rare_df_frac * (self.n_chunks as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::DfTable;
    use crate::index::store::DocIndex;

    #[test]
    fn rarity_on_an_empty_table() {
        let t = DfTable::new();
        assert_eq!(t.n_chunks, 0);
        assert_eq!(t.df("modbus"), 0);
        assert!(t.is_rare("modbus", 0.5)); // df==0 ⇒ rare, regardless of the token
        assert!(t.is_empty());
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
