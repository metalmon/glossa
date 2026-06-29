# Contributing to glossa

Thank you for your interest in contributing. glossa is a pure-Rust, offline knowledge-base engine with an MCP server for LLM agents.

## Prerequisites

- Rust stable (2021 edition)
- [just](https://github.com/casey/just) (optional but recommended for the dev pipeline)

## Build and test

```bash
# Release build (recommended)
just build
# or
cargo build --workspace --release

# Run tests (matches CI)
cargo test -p glossa --release --locked
cargo test -p kb-eval --release --locked
```

The workspace also includes `kb-eval` and `kb-train`. Full workspace tests can take longer; avoid running `cargo test --workspace` while a long `kb-train enrich` process holds a binary lock on Windows.

### CI and releases

- **CI** (`.github/workflows/ci.yml`): push/PR → tests on Ubuntu + Windows, `cargo check` on Ubuntu.
- **Releases** (`.github/workflows/release.yml`): push a tag `v0.1.0` → GitHub Release with `kb` for Linux, Windows, macOS (arm64 + x64).

```bash
git tag v0.1.0
git push origin v0.1.0
```

Release artifacts ship **`kb` only** (the operator binary). `kb-eval` / `kb-train` are built from source for benchmark/enrich workflows.

## MCP tool schema sync

After changing tool definitions in `src/mcp.rs`, regenerate TensorZero tool schemas:

```bash
just tools
just gw-restart   # if the TensorZero gateway is running
```

## Pull requests

- Keep diffs focused; match existing code style and naming.
- Add or update tests for behavior changes in `src/`.
- Do not commit secrets, local corpora (`kb-test/`, `kb-val/`), or generated artifacts (`.glossa/`, `eval-*.json`, `gepa-out/`).

## Questions

Open a GitHub issue when the repository is published. For architecture and usage, start with [README.md](README.md) and [docs/README.md](docs/README.md).
