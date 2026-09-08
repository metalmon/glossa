use assert_cmd::Command;
use predicates::prelude::PredicateBooleanExt;
use std::io::Write;

/// `kb index` must honor `[corpus]` roots from the config file (shared resolver), building the
/// index for the file's root without a positional path.
#[test]
fn kb_index_honors_corpus_from_config() {
    let dir = tempfile::tempdir().unwrap();
    let corpus = dir.path().join("corpus");
    std::fs::create_dir_all(&corpus).unwrap();
    std::fs::write(corpus.join("note.md"), "hello networked world").unwrap();
    let state = dir.path().join("state");
    let cfg = dir.path().join("role.toml");
    std::fs::write(
        &cfg,
        format!(
            "[corpus]\nroots = [\"docs={}\"]\nstate_dir = \"{}\"\n",
            corpus.display().to_string().replace('\\', "/"),
            state.display().to_string().replace('\\', "/"),
        ),
    )
    .unwrap();

    Command::cargo_bin("kb")
        .unwrap()
        .args(["index", "--config"])
        .arg(&cfg)
        .assert()
        .success()
        .stdout(predicates::str::contains("indexed"));
    // Artifacts landed under the file's state_dir, proving [corpus] was consumed by kb index.
    assert!(state.join(".glossa").join("index").exists());
}

/// A serving-only section (`[server]`) present while running `kb index` is INERT — a debug note, not
/// an error. The command still succeeds.
#[test]
fn kb_index_tolerates_serving_only_section() {
    let dir = tempfile::tempdir().unwrap();
    let corpus = dir.path().join("corpus");
    std::fs::create_dir_all(&corpus).unwrap();
    std::fs::write(corpus.join("note.md"), "x").unwrap();
    let state = dir.path().join("state");
    let cfg = dir.path().join("role.toml");
    std::fs::write(
        &cfg,
        format!(
            "[corpus]\nroots = [\"docs={}\"]\nstate_dir = \"{}\"\n\
             [server]\nbind = \"0.0.0.0:9000\"\n",
            corpus.display().to_string().replace('\\', "/"),
            state.display().to_string().replace('\\', "/"),
        ),
    )
    .unwrap();

    Command::cargo_bin("kb")
        .unwrap()
        .args(["index", "--config"])
        .arg(&cfg)
        .assert()
        .success(); // serving-only [server] does NOT make kb index fail
}

/// A non-loopback bind supplied by the FILE, with no token anywhere, must trip Spec C's safety
/// interlock at startup — proving the interlock sees the merged (file-derived) bind, not just flags.
#[test]
fn interlock_fires_on_bind_from_config_file() {
    let dir = tempfile::tempdir().unwrap();
    let corpus = dir.path().join("corpus");
    std::fs::create_dir_all(&corpus).unwrap();
    let cfg = dir.path().join("role.toml");
    let mut f = std::fs::File::create(&cfg).unwrap();
    write!(
        f,
        "[corpus]\nroots = [\"docs={}\"]\nstate_dir = \"{}\"\n\
         [server]\ntransport = \"streamable-http\"\nbind = \"0.0.0.0:8137\"\n",
        corpus.display().to_string().replace('\\', "/"),
        dir.path().join("state").display().to_string().replace('\\', "/"),
    )
    .unwrap();

    // No --auth-token / GLOSSA_MCP_TOKEN, no --insecure → must refuse to start.
    Command::cargo_bin("kb")
        .unwrap()
        .args(["mcp", "--config"])
        .arg(&cfg)
        .env_remove("GLOSSA_MCP_TOKEN")
        .timeout(std::time::Duration::from_secs(20))
        .assert()
        .failure()
        .stderr(predicates::str::contains("insecure").or(predicates::str::contains("auth")));
}

/// The same shape of file as above declares `transport = "streamable-http"` + a non-loopback
/// `bind`, which alone would hit the interlock path (see the test above). Passing `--transport
/// stdio` on the CLI must override the file's transport — proving flag > file precedence — so the
/// process takes the stdio code path (never touches `bind`/the interlock at all) instead. With
/// stdin closed immediately, the stdio transport reports a connection-closed handshake error
/// rather than the interlock's "insecure"/"auth" refusal — that distinction is what proves which
/// code path actually ran.
#[test]
fn flag_transport_overrides_config_file_transport() {
    let dir = tempfile::tempdir().unwrap();
    let corpus = dir.path().join("corpus");
    std::fs::create_dir_all(&corpus).unwrap();
    let cfg = dir.path().join("role.toml");
    let mut f = std::fs::File::create(&cfg).unwrap();
    write!(
        f,
        "[corpus]\nroots = [\"docs={}\"]\nstate_dir = \"{}\"\n\
         [server]\ntransport = \"streamable-http\"\nbind = \"0.0.0.0:8138\"\n",
        corpus.display().to_string().replace('\\', "/"),
        dir.path().join("state").display().to_string().replace('\\', "/"),
    )
    .unwrap();

    Command::cargo_bin("kb")
        .unwrap()
        .args(["mcp", "--config"])
        .arg(&cfg)
        .args(["--transport", "stdio"])
        .write_stdin(&b""[..]) // writes nothing, then closes stdin -> immediate EOF
        .timeout(std::time::Duration::from_secs(20))
        .assert()
        .failure()
        .stderr(
            predicates::str::contains("connection closed")
                .and(predicates::str::contains("insecure").not())
                .and(predicates::str::contains("refusing to serve").not()),
        );
}
