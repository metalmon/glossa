//! Spec B e2e: network-read / freshness BEHAVIOR against the real `kb` binary.
//!
//! HONESTY NOTE: the fault-injection seam (`read_fault`) that Spec B's unit tests use to force
//! transient corpus-read failures is `#[cfg(test)]`, in-process only, and is therefore UNREACHABLE
//! from an externally-spawned `kb` binary. This file consequently covers only externally-observable
//! behavior: (a) an on-read freshen picks up a newly-added corpus file, and (b) the retry-tuning
//! knobs are accepted and the server still serves. It does NOT (and cannot) assert fault-injected
//! retry/serve-stale behavior over the socket.
//!
//! REGRESSION (the search-reader staleness fix): the server serves search from a shared handle whose
//! search index is behind an `ArcSwap`; a freshen that indexes a change swaps in a freshly-opened
//! reader (see `GraphHandle::refresh_idx`). So a reader ALREADY BUILT by an earlier query must, after
//! a subsequent freshen, serve a newly-added file too — not keep its stale snapshot. This test proves
//! exactly that over the socket: it does a first search (which builds the reader), THEN adds a file,
//! THEN asserts a later search through that same live server surfaces it (previously it could not, and
//! the test had to fall back to `/metrics`, which opens a fresh reader per scrape).
#![cfg(feature = "e2e")]

#[path = "e2e/harness.rs"]
mod harness;

use harness::*;
use std::time::{Duration, Instant};

#[test]
fn on_read_freshen_picks_up_a_newly_added_file() {
    // One file at start; a small min-rescan so the on-read freshen isn't throttled during the test.
    let c = Corpus::with_files(&[("a.md", "# First\n\nThis document mentions term_one only.\n")]);
    let state = state_dir();
    let server = ServerBuilder::new()
        .root(c.root_arg("docs"))
        .state_dir(state.path())
        .env("GLOSSA_MIN_RESCAN_MS", "50")
        .start();

    // BUILD the shared search reader NOW, before the change: a first search that finds the initial
    // file. This is the pre-existing, long-lived reader a daemon caches for its lifetime — the exact
    // thing that used to go stale. (A brief retry only absorbs startup-freshen/commit latency.)
    let mcp = McpClient::connect(server.base());
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if mcp.search_text("term_one").contains("term_one") {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the initial file never became searchable.\nstderr:\n{}",
            server.stderr()
        );
        std::thread::sleep(Duration::from_millis(200));
    }

    // Add a second file carrying term_two, then push the dir mtime strictly forward so the freshen
    // dir-mtime gate reliably re-scans (defeating the coarse Windows FS-clock race).
    c.add_file("b.md", "# Second\n\nThis document introduces term_two.\n");
    bump_dir_mtime(c.path());

    // The SAME already-built reader must now surface the new file: the on-read freshen indexes it and
    // swaps a fresh reader into the shared handle. (Retries absorb the min-rescan gate + commit, NOT
    // reader staleness — pre-fix this search would keep serving the stale snapshot indefinitely.)
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut found = false;
    loop {
        if mcp.search_text("term_two").contains("term_two") {
            found = true;
            break;
        }
        if Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(300));
    }
    assert!(
        found,
        "the already-built search reader did not surface the newly-added file after a freshen.\nstderr:\n{}",
        server.stderr()
    );

    // And the original file is still retrievable (the swapped-in reader carries both).
    assert!(
        mcp.search_text("term_one").contains("term_one"),
        "original file should remain retrievable after the freshen"
    );
}

#[test]
fn retry_tuning_knobs_are_accepted_and_server_serves() {
    // Setting the retry knobs must not break startup: the server still becomes ready and serves.
    let c = Corpus::with_files(&[("a.md", "# Doc\n\nsearchable smoke_term content.\n")]);
    let state = state_dir();

    // Index the corpus first, targeting the SAME labeled root + state-dir the server below will
    // serve (the root label is persisted in index keys, so it must match the server's `--root`
    // exactly), with the same retry-tuning knobs set (mirrors e2e_spec_a_multiroot's index-then-serve
    // pattern), so the server's first search is against an already-built index instead of racing
    // startup indexing.
    assert_cmd::Command::cargo_bin("kb")
        .unwrap()
        .arg("index")
        .arg("--root")
        .arg(c.root_arg("docs"))
        .arg("--state-dir")
        .arg(state.path())
        .env("GLOSSA_READ_RETRIES", "5")
        .env("GLOSSA_READ_RETRY_BACKOFF_MS", "50")
        .assert()
        .success();

    let server = ServerBuilder::new()
        .root(c.root_arg("docs"))
        .state_dir(state.path())
        .env("GLOSSA_READ_RETRIES", "5")
        .env("GLOSSA_READ_RETRY_BACKOFF_MS", "50")
        .start();

    // Readiness is already gated by start(); confirm it directly and run one search.
    assert_eq!(http_get(server.base(), "/ready", &[]).status, 200);
    let mcp = McpClient::connect(server.base());
    assert!(
        mcp.search_text("smoke_term").contains("smoke_term"),
        "server with retry knobs set should still serve a search.\nstderr:\n{}",
        server.stderr()
    );
}
