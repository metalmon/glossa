# Changelog

All notable changes to glossa are documented here. Release tags ship the **`kb`** binary only; `kb-eval` / `kb-train` are built from source.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Quad GEPA**: optimize prod `answer_hotpot` prompt against search, grep, glob, and read micro-tasks via TensorZero (`functions.search`, `grep`, `glob`, `read`, `gepa_reflect`).
- **export-tz**: emit `grep.jsonl` and `glob.jsonl`; synthetic grep/glob rows when episodes lack those tool calls; improved gold path canonicalization.
- **Pareto GEPA**: `pareto_size`, frequency-weighted parent selection, full-val final candidate pick (canonical acceptance after minibatch improve).
- **justfile**: Windows `.exe` binaries for `kb-eval` / `kb-train`; `gepa-apply` fix; default judge in `just eval`; `gepa-reset` / `eval-reset` recipes.
- TZ micro-task templates: `grep/system.minijinja`, `glob/system.minijinja`.

### Changed

- **Eval harness**: TensorZero backend skips per-question corpus wipe/reindex; glossa-train JSON accepts missing `context` / `supporting_facts`.
- **Prod agent prompt**: glob-first retrieval protocol in `answer_hotpot/system.minijinja`.
- **Docs**: rewritten [eval-and-training.md](docs/eval-and-training.md) for quad GEPA and current just recipes.

### Fixed

- GEPA scoring via TZ gateway (IPv4 localhost, episode id skew, tool-call parsing).
- Stale GNU `kb-eval` / `kb-train` artifacts on Windows bypassing fresh builds.
- `gepa-reset` now clears `grep` and `glob` inference history.

## [1.1.0] — 2025-06-XX

- TensorZero eval integration, kb-train enrich, initial GEPA search/read path, justfile dev pipeline.

## [1.0.0]

- Initial public release: file-first index, graph, MCP server, BM25 search, grep, glob, read.

[Unreleased]: https://github.com/metalmon/glossa/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/metalmon/glossa/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/metalmon/glossa/releases/tag/v1.0.0
