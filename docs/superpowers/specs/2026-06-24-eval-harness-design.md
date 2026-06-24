# glossa agent-eval harness — design

Status: design approved (brainstorming), 2026-06-24. Next: implementation plan (writing-plans).

## Goal

A model-agnostic harness that drives an LLM to answer benchmark questions **by using glossa's real MCP
tools**, captures the retrieval trace, and scores the answers. First use (Track A): position glossa's
engine on a public multi-hop benchmark (HotpotQA-distractor) with the official answer metric, for both
a strong model (Claude) and a weak local model (Qwen3.5-4B). It is the shared foundation later reused
by Track B (domain Q/A, gold spans, heatmap) — but Track B is out of scope here.

## Context & constraints

- glossa is a pure-Rust, offline, File-First KB tool. Its MCP server (`kb mcp --profile …`) exposes
  `search`, `read`, `glossary`, `neighbors`, `index`, `reindex`, `resolve`, `graph_upsert`, `purge`.
- The **product** (`kb` binary) must stay pure-Rust/offline/single-binary. The **harness is a dev tool**,
  not shipped to clients — its dependencies (HTTP client, later a plotter) must NOT leak into `kb`.
- The eval drives the **real shipped integration**: the model is a **direct MCP client** of glossa
  (Claude via `claude -p` with glossa configured; Qwen via LM Studio's MCP support). The harness does
  NOT mediate the tool loop. (Decided over a TensorZero gateway, which would force harness-owned tool
  loops + ClickHouse and not capture the retrieval trace.)
- The **retrieval trace** (what we score on) is produced by the **glossa MCP server**, written to a
  local JSONL file. No external trace service.

## Architecture

Convert the repo into a cargo **workspace**:
- `glossa/` — product crate: lib + `kb` binary (unchanged ethos: pure Rust, offline).
- `eval/` — dev crate: `kb-eval` binary. Depends on the `glossa` lib only for shared trace types;
  drives the `kb` binary and `claude -p` as **subprocesses** and LM Studio over HTTP. May use a pure-Rust
  HTTP client (`ureq` + rustls, C-free) for LM Studio. No `kb` product dep gains anything from this.

### Product-side additions (small, in `glossa`)

1. **MCP trace logging.** When `kb mcp --trace` is set (or env `GLOSSA_TRACE=1`), every tool invocation
   appends one JSON line to `<root>/.glossa/traces/<unix_ts_ms>-<pid>.jsonl`:
   `{ "ts_ms": u64, "tool": "search"|"read"|…, "args": {…}, "result": {…summary…} }`.
   For `search`, `result` carries the hits as `[{path, location, score}]`; for `read`, the resolved
   `{path, location}`. Off by default (no behaviour change for normal use). This is the retrieval trace.
2. **`kb mcp --no-graph`.** Registers only `search` + `read` (graph/index/admin tools hidden) — the
   control arm for a future graph-off/on A/B. Reuses the existing `ToolRouter::disable_route` mechanism.

### `kb-eval` components (each a focused module)

1. **`dataset`** — parse a HotpotQA-distractor JSON file into
   `Question { id, question, answer, paragraphs: Vec<Paragraph{title, sentences: Vec<String>}>, supporting_titles: Vec<String> }`.
   (`supporting_facts` in HotpotQA are `[title, sent_idx]`; we keep the distinct gold paragraph titles.)
2. **`corpus`** — a single **fixed working dir** `<work>/` (e.g. `eval-corpus/`), rebuilt per question:
   clear its doc files + its `.glossa/`, write each paragraph of the current `Question` to
   `<work>/<sanitized title>.md` (heading = title, body = sentences), then run `kb index <work>`
   (subprocess). A fixed dir is required because LM Studio's MCP config points at one static path; both
   backends' glossa MCP therefore target `<work>/`, and we swap the corpus by reindexing between
   questions (the MCP tools open the index/graph per call, so a persistent server sees the new content).
3. **`backend`** — trait `AgentBackend { fn answer(&self, corpus_dir, question) -> Result<String> }`
   with three impls:
   - `claude`: spawn `claude -p "<prompt>" --mcp-config <generated>` (glossa MCP at `corpus_dir`),
     capture stdout, parse the answer after the `ANSWER:` marker.
   - `qwen`: POST to LM Studio's OpenAI-compatible `/v1/chat/completions` (LM Studio configured with the
     glossa MCP server at `corpus_dir`); LM Studio runs the tool loop; capture the final message, parse `ANSWER:`.
   - `mock`: returns a canned answer (for deterministic tests/CI; no live model).
   The prompt instructs: answer the question using the glossa tools over the indexed corpus, then output
   the final answer on a line beginning `ANSWER:`.
4. **`trace`** — read `<corpus_dir>/.glossa/traces/*.jsonl`; select entries in the time window
   `[t_send, t_recv]` recorded around the backend call (sequential runs → unambiguous). Yields the
   question's tool calls + returned hits.
5. **`score`** — pure functions:
   - `normalize(s)` (HotpotQA: lowercase, strip articles/punctuation, collapse whitespace).
   - `exact_match(pred, gold) -> bool`, `token_f1(pred, gold) -> f32`.
   - `retrieval_recall(trace, supporting_titles) -> f32`: fraction of gold paragraph titles whose
     file/`path` appeared in any `search`/`read` result in the trace.
6. **`run`** — iterate questions (configurable `--limit N`), per question: build+index corpus, start
   `kb mcp --trace` on it, call the backend, read+window the trace, score; aggregate; write
   `eval-<backend>-<ts>.json` (per-question rows) + a text summary (mean EM, F1, retrieval-recall, count).

### Data flow (per question, sequential)

`load Q` → `corpus::rebuild <work> + kb index` → (`kb mcp --trace` on `<work>`: fresh per `claude -p`
call, or the persistent LM Studio-owned server) → record `t_send` → `backend.answer()` (model uses MCP
tools) → record `t_recv` → `trace::window` over `<work>/.glossa/traces/*.jsonl` → `score` → row.

## v1 scope

- Backends: `qwen` (LM Studio), `claude` (`claude -p`), `mock` (tests).
- Benchmark: **HotpotQA-distractor**, local JSON, `--limit N`.
- Graph available (default profile); `--no-graph` exists but the rigorous graph-off/on A/B is **deferred**
  to a large-corpus setting.
- Output: per-run JSON + text aggregate (EM / token-F1 / retrieval-recall, per backend).
- CLI sketch: `kb-eval run --dataset <path> --backend qwen|claude|mock [--limit N] [--lmstudio-url URL]`.

### Honest caveat — the graph A/B

On a 10-paragraph distractor corpus the knowledge graph rarely helps (plain `search` over 10 paragraphs
suffices); the graph's multi-hop benefit appears on **large** corpora (HotpotQA-fullwiki or the operator's
real domain base). So the graph-off vs graph-on A/B is deferred to a large-corpus run. `--no-graph` is
built now so the A/B is possible later. v1's headline is **engine positioning** (answer EM/F1) +
retrieval-recall, comparing Claude vs Qwen.

## Error handling

- A question whose corpus fails to index, whose backend errors/times out, or whose answer can't be parsed
  is recorded as a `failed` row (with reason) and scored 0 — the run never aborts on one bad question.
- The trace file may be empty (model answered without tools) → retrieval-recall 0, answer still scored.
- Backends have a per-question timeout; on timeout the row is `failed: timeout`.

## Testing

- **Unit (pure, deterministic):** `normalize`/`exact_match`/`token_f1` against the HotpotQA reference
  examples; HotpotQA JSON parsing into `Question`; trace JSONL parsing + time-window selection;
  `retrieval_recall` over a synthetic trace + supporting titles.
- **Integration:** the `mock` backend end-to-end on a tiny 2-question synthetic dataset — build corpus,
  (mock) answer, score, aggregate — proving the pipeline without a live model or network. Runs in CI.
- Live backends (`qwen`, `claude`) are exercised manually by the operator (need LM Studio / Claude Code).

## Non-goals (deferred)

- PNG heatmap (per-question × per-step visualization).
- LLM-as-judge answer scoring (Track B free-form answers).
- Track B: domain Q/A curation, gold-span bootstrap, groundedness/citation metric.
- The large-corpus graph-off/on A/B and HotpotQA-fullwiki.
- Auto-download of datasets (operator provides the local JSON).
- TensorZero / external inference observability.

## Notes

- Pure Rust throughout the harness too (no C): `ureq`+rustls for HTTP; subprocess for `kb`/`claude`.
  (The harness is dev-only, so this is a preference, not a hard product constraint.)
- Sequential execution is intentional: it makes time-windowed trace correlation unambiguous without
  threading run-ids through LM Studio's persistent MCP connection.
