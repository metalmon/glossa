//! Spec A e2e: multiple labeled corpus roots + a separated `.glossa` state directory, proven
//! end-to-end against the real `kb` binary — index writes land only under `--state-dir` (never in a
//! corpus root), and both roots are retrievable over the live MCP endpoint.
#![cfg(feature = "e2e")]

#[path = "e2e/harness.rs"]
mod harness;

use harness::*;

#[test]
fn index_writes_only_to_state_dir_then_serves_both_roots() {
    // Two labeled corpora, each with a unique marker term, plus a separate state dir.
    let docs = Corpus::with_files(&[(
        "a.md",
        "# Root A\n\nThe alpha_marker term appears only in the first corpus.\n",
    )]);
    let specs = Corpus::with_files(&[(
        "b.md",
        "# Root B\n\nThe beta_marker term appears only in the second corpus.\n",
    )]);
    let state = state_dir();

    // Index both roots into the state dir.
    assert_cmd::Command::cargo_bin("kb")
        .unwrap()
        .arg("index")
        .arg("--root")
        .arg(docs.root_arg("docs"))
        .arg("--root")
        .arg(specs.root_arg("specs"))
        .arg("--state-dir")
        .arg(state.path())
        .assert()
        .success();

    // State lives under the state dir...
    assert!(
        state.path().join(".glossa").join("index").is_dir(),
        "expected <state>/.glossa/index to exist after indexing"
    );
    // ...and NOT co-located in either corpus root.
    assert!(
        !docs.path().join(".glossa").exists(),
        "corpus root A must not carry a .glossa"
    );
    assert!(
        !specs.path().join(".glossa").exists(),
        "corpus root B must not carry a .glossa"
    );

    // Serve both roots off the shared state dir and prove multi-root retrieval end-to-end.
    let server = ServerBuilder::new()
        .root(docs.root_arg("docs"))
        .root(specs.root_arg("specs"))
        .state_dir(state.path())
        .start();

    let mcp = McpClient::connect(server.base());
    let a = mcp.search_text("alpha_marker");
    assert!(
        a.contains("alpha_marker"),
        "root A term not retrievable over MCP: {a:?}\nstderr:\n{}",
        server.stderr()
    );
    let b = mcp.search_text("beta_marker");
    assert!(
        b.contains("beta_marker"),
        "root B term not retrievable over MCP: {b:?}\nstderr:\n{}",
        server.stderr()
    );
}
