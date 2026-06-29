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

# Run tests
cargo test -p glossa --release
```

The workspace also includes `kb-eval` and `kb-train`. Full workspace tests can take longer; avoid running `cargo test --workspace` while a long `kb-train enrich` process holds a binary lock on Windows.

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
