//! Spec B e2e: network-read / freshness BEHAVIOR against the real `kb` binary.
//!
//! HONESTY NOTE: the fault-injection seam (`read_fault`) that Spec B's unit tests use to force
//! transient corpus-read failures is `#[cfg(test)]`, in-process only, and is therefore UNREACHABLE
//! from an externally-spawned `kb` binary. This file consequently covers only externally-observable
//! behavior: (a) an on-read freshen picks up a newly-added corpus file, and (b) the retry-tuning
//! knobs are accepted and the server still serves. It does NOT (and cannot) assert fault-injected
//! retry/serve-stale behavior over the socket.
//!
//! OBSERVED (documented for the freshen test's shape): the on-read freshen DOES index a newly-added
//! file server-side (the `/metrics` `glossa_index_chunks` gauge climbs), and the FIRST `search`/`read`
//! after the file appears observes it. A search reader that was already built by an EARLIER query,
//! however, keeps serving its snapshot for a just-added file (an externally-committed segment is not
//! surfaced to the pre-existing reader) — so this test drives the negative "before" state via
//! `/metrics` (which does not build the search reader) rather than a search, then asserts the first
//! post-add search surfaces the new file.
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

    // Establish the "before" state WITHOUT building the search reader: the /metrics chunk gauge
    // reflects only the single initial file. (Poll briefly for the startup freshen to land it.)
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let m = http_get(server.base(), "/metrics", &[]).body;
        if m.contains("glossa_index_chunks 1") {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "startup freshen never indexed the initial file (chunks stayed 0).\nmetrics:\n{m}"
        );
        std::thread::sleep(Duration::from_millis(200));
    }

    // Add a second file carrying term_two, then push the dir mtime strictly forward so the freshen
    // dir-mtime gate reliably re-scans (defeating the coarse Windows FS-clock race).
    c.add_file("b.md", "# Second\n\nThis document introduces term_two.\n");
    bump_dir_mtime(c.path());

    // The FIRST search drives the on-read freshen AND is the first reader build, so it observes the
    // newly-added file. (A few retries only absorb commit/gate latency, not reader staleness.)
    let mcp = McpClient::connect(server.base());
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
        "on-read freshen did not surface the newly-added file to the first search.\nstderr:\n{}",
        server.stderr()
    );

    // The original file is still retrievable too (the reader now carries both).
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
