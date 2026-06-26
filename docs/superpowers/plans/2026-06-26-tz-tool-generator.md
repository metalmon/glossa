# TZ Tool-Config Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Generate the eval's TensorZero tool config (`tools/*.json` schemas + `[tools.*]` descriptions + the `answer_hotpot` tool list) FROM the canonical MCP tool definitions, so tool schemas/descriptions have ONE source (`src/mcp.rs`) and stop drifting from the gateway config.

**Architecture:** A `kb dump-tz-tools <config-dir>` subcommand builds a Reader-profile `GlossaServer`, calls `tool_router.list_all()` (rmcp `Tool { name, description, input_schema }`), writes each tool's `input_schema` to `<dir>/tools/<name>.json`, and marker-splices the `[tools.*]` blocks + the `answer_hotpot` `tools = [...]` list into `<dir>/tensorzero.toml` (the gateway reads that single file).

**Tech Stack:** Rust, rmcp 1.8.0 (`ToolRouter::list_all`), serde_json, the existing `GlossaServer`.

## Global Constraints
- **Pure-Rust, C-free** (`cargo tree -p glossa -i cc` empty): no new deps.
- Only `tools/*.json` and the MARKED regions of `tensorzero.toml` are touched — the rest of the toml (functions, variants, sampling, metrics) is preserved byte-for-byte.
- The generated tool set = the **Reader profile** (search/read/grep/glob/glossary/neighbors) — the answering agent's tools.
- TDD.

---

### Task 1: `kb dump-tz-tools` generator

**Files:**
- Create: `src/tz_export.rs`
- Modify: `src/mcp.rs` (a public accessor for the tool list), `src/lib.rs` (`pub mod tz_export;` if needed), `src/main.rs` (the subcommand)
- Modify (one-time, by hand): `eval/tensorzero/config/tensorzero.toml` (add the marker comments — see Step 4)

**Interfaces:**
- Produces: `glossa::tz_export::dump(config_dir: &Path) -> anyhow::Result<()>`; a `kb dump-tz-tools [--config-dir <dir>]` CLI command.
- Consumes: `GlossaServer` + a new `pub fn tool_specs(&self) -> Vec<rmcp::model::Tool>` accessor (wraps `self.tool_router.list_all()`).

- [ ] **Step 1: Expose the tool list from `GlossaServer`** — in `src/mcp.rs` add (non-test) `pub fn tool_specs(&self) -> Vec<rmcp::model::Tool> { self.tool_router.list_all() }`.

- [ ] **Step 2: Write `src/tz_export.rs`** with `dump(config_dir)`:
  1. `let srv = crate::mcp::GlossaServer::new(std::path::PathBuf::from("."), crate::mcp::Profile::Reader, false, false);` then `let tools = srv.tool_specs();` (Reader profile → the 6 query tools).
  2. For each `t` in tools: write `config_dir/tools/{t.name}.json` = `serde_json::to_string_pretty(&*t.input_schema)?` (+ trailing newline). Create the `tools/` dir if missing.
  3. Build the `[tools.*]` toml text: for each tool, in a STABLE order (sort by name),
     ```
     [tools.<name>]
     description = <toml-escaped description>
     parameters = "tools/<name>.json"
     ```
     Use a real TOML string escape (or serialize via `toml` only if it's already a dep — otherwise escape `"` and `\` and wrap; descriptions have no newlines). Confirm `toml` crate availability; if absent, hand-escape (descriptions are single-line).
  4. Build the tool-list line: `tools = [<comma-separated "name">]` sorted.
  5. Splice into `config_dir/tensorzero.toml`: replace the text between `# >>> GENERATED TOOLS …` and `# <<< GENERATED TOOLS` with the `[tools.*]` blocks, and between `# >>> GENERATED TOOL LIST` and `# <<< GENERATED TOOL LIST` with the `tools = [...]` line. If a marker pair is missing, return an error telling the user to add the markers once.

- [ ] **Step 3: Add the `kb` subcommand** in `src/main.rs`: `DumpTzTools { #[arg(long, default_value = "eval/tensorzero/config")] config_dir: PathBuf }` → calls `glossa::tz_export::dump(&config_dir)?` and prints a one-line summary (N tools written).

- [ ] **Step 4: Add the markers to `eval/tensorzero/config/tensorzero.toml`** (one-time, by hand in this task): wrap the existing `[tools.search]…[tools.neighbors]` blocks with
  ```
  # >>> GENERATED TOOLS (kb dump-tz-tools) — do not edit by hand
  …existing [tools.*]…
  # <<< GENERATED TOOLS
  ```
  and wrap the `answer_hotpot` `tools = […]` line with
  ```
  # >>> GENERATED TOOL LIST
  tools = ["search", "read", "grep", "glob", "glossary", "neighbors"]
  # <<< GENERATED TOOL LIST
  ```

- [ ] **Step 5: Test** `src/tz_export.rs`: in a tempdir, write a minimal `tensorzero.toml` containing the four markers + some surrounding `[functions.x]` text and a `tools/` dir; run `dump(tmp)`; assert (a) `tools/search.json` exists and parses as JSON with a `properties` object, (b) the toml's GENERATED-TOOLS region now contains `[tools.search]` + `parameters = "tools/search.json"`, (c) the surrounding non-marked text is unchanged, (d) the tool-list region contains `"neighbors"`. Run `cargo test -p glossa tz_export`.

- [ ] **Step 6: Run the generator for real + verify the JSON is valid** — `cargo run --release -p glossa --bin kb -- dump-tz-tools` (or build then run), then `git diff --stat eval/tensorzero/config` to confirm only `tools/*.json` + the marked toml regions changed. Confirm each `tools/*.json` is valid JSON.

- [ ] **Step 7: C-free gate + commit** `cargo tree -p glossa -i cc` (empty); `git add -A && git commit -m "feat(kb): dump-tz-tools generates TZ tool config from MCP definitions (one source)"`

## Self-Review
**Coverage:** generator (Steps 1-3), markers (4), test (5), real run + validation (6). **Types:** `tool_specs() -> Vec<rmcp::model::Tool>`, `dump(&Path) -> Result<()>`. **Risk:** rmcp `input_schema` (schemars draft-07) may carry `$schema`/`title` the gateway tolerates — verify the gateway loads the regenerated config (controller will smoke-test post-merge). **Placeholder:** TOML escaping approach flagged (use `toml` crate if present, else hand-escape single-line descriptions).
