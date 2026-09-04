//! Flat-RSS scale gate for the out-of-core graph engine.
//!
//! The hard requirement: a retrieval request's working set must scale with the LOCAL neighborhood it
//! touches, not with the corpus size. Forward-push local PPR ([`ppr_push`]) expands only nodes whose
//! residual exceeds `eps * weighted_degree`, so its work (node expansions = "touched") is bounded by
//! locality. This test proves it directly: a 20× larger corpus must not blow up the touched count.
//!
//! Touched-count is the honest, deterministic proxy for RSS here — under the memory-mapped CSR the OS
//! pages in only the rows the push reads, so peak resident set tracks the touched set, and asserting
//! the touched set is flat across N asserts RSS is flat across N without a flaky cross-platform RSS
//! probe.

use glossa::graph::csr::CsrTransition;
use glossa::graph::ppr::ppr_push_touched;
use std::collections::HashMap;

/// A synthetic ring-lattice graph of `n` nodes, each undirected-connected to its ±1 and ±2 neighbors
/// (fixed avg degree 4, independent of `n`). Built to a temp dir as a memory-mapped CSR. Returns the
/// owned `TempDir` (its files back the mmap — it MUST outlive the returned `CsrTransition`) and the
/// opened CSR. Node ids are `n{i}`.
fn build_synthetic(n: usize) -> (tempfile::TempDir, CsrTransition) {
    let ids: Vec<String> = (0..n).map(|i| format!("n{i}")).collect();
    let adj: Vec<Vec<(usize, f32)>> = (0..n)
        .map(|i| {
            vec![
                ((i + 1) % n, 1.0),
                ((i + 2) % n, 1.0),
                ((i + n - 1) % n, 1.0),
                ((i + n - 2) % n, 1.0),
            ]
        })
        .collect();
    let dir = tempfile::tempdir().unwrap();
    // `content_sig` is opaque here; a fixed value is fine (one graph per temp dir).
    CsrTransition::build(&ids, &adj, dir.path(), 0xF1A7_5CA1E).unwrap();
    let csr = CsrTransition::open(dir.path(), 0xF1A7_5CA1E)
        .unwrap()
        .expect("just-built CSR opens");
    (dir, csr)
}

#[test]
fn forward_push_working_set_is_flat_across_corpus_size() {
    let (_d_small, small) = build_synthetic(1_000);
    let (_d_big, big) = build_synthetic(20_000);
    assert_eq!(small.len(), 1_000);
    assert_eq!(big.len(), 20_000);

    let seeds = HashMap::from([("n0".to_string(), 1.0f32)]);
    let touched_small = ppr_push_touched(&small, &seeds, 0.15, 1e-4);
    let touched_big = ppr_push_touched(&big, &seeds, 0.15, 1e-4);

    // The push must actually run (a locality bug that touched nothing would trivially "pass" a ≤).
    assert!(touched_small > 0, "small push touched nothing");
    assert!(touched_big > 0, "big push touched nothing");
    // Working set depends on locality, not N: 20× the corpus must not 20× the touch count. For a
    // ring lattice the local neighborhood is identical at both sizes, so the counts should match
    // closely; the ≤ 2× band is generous headroom against boundary effects.
    assert!(
        touched_big <= touched_small * 2,
        "working set grew with corpus size: touched_big={touched_big} vs touched_small={touched_small}"
    );
}
