# Multi-API transport (OpenAI + Anthropic) + resilience — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Abstract the `kb-eval` chat layer behind a `ChatTransport` trait so both OpenAI-compatible and Anthropic-compatible (`/v1/messages`) APIs work for EVERY call path (agentic reader/build/distil AND one-shot judge/reflect), selected by an `api` config field per endpoint; then add per-endpoint rate-limiting and a fallback chain.

**Architecture:** A neutral `ChatTransport` trait (`call → TurnReply{text,tool_calls,raw}`, `push_assistant_turn`, `push_tool_results`) with two impls (OpenAI wraps today's code behavior-identically; Anthropic is new). `run_agent_loop` becomes generic over `&dyn ChatTransport`, keeping its dedup/streak/NBA logic verbatim. A `resilience` layer wraps transport calls with per-endpoint rate-limit + retry + fallback-chain. Config: `Endpoint` gains `api: ApiKind`, `rate_limit`, `fallback: Vec<Endpoint>` (all serde-default → existing `lab.toml` unchanged).

**Tech Stack:** Rust, `reqwest` (existing sync-over-async bridge), `serde_json` raw `Value` round-trip (preserve provider fields), existing `glossa_tools::exec`.

**Spec:** none written (design agreed in-session from the code-explorer blueprint). The blueprint's seam analysis binds Phase 1.

## Global Constraints

- English-only public surface; no corpus/gold values in code/tests.
- `endpoint` = the FULL request URL, POSTed verbatim (no path append/strip) — for OpenAI `.../v1/chat/completions`, for Anthropic `.../v1/messages`.
- BACKWARD COMPAT: existing `lab.toml`/CLI with no `api`/`rate_limit`/`fallback` must behave EXACTLY as today (OpenAI, no throttle, no fallback). Every new config field is `#[serde(default)]`.
- BEHAVIOR-IDENTICAL refactor in Phase 1: all existing `openai.rs` tests (URL-verbatim, agent-loop dedup/streak/NBA, parse_tool_args, injected-system-prompt) pass unchanged or move verbatim; `cargo test -p kb-eval` stays green throughout.
- `glossa_tools::exec` (tool execution) and the `AgentBackend` trait are UNTOUCHED — the abstraction sits above tool execution and below the backends.
- Build/test on C: via `CARGO_TARGET_DIR=C:/glossa-target`.
- Do NOT change `tensorzero.rs` (it's the structural template to imitate, not to edit).

---

## PHASE 1 — Transport abstraction + OpenAI impl (behavior-identical)

### Task 1: `ChatTransport` trait + neutral types + shared plumbing

**Files:**
- Create: `eval/src/backend/transport/mod.rs`
- Modify: `eval/src/backend/mod.rs` (`pub mod transport;`)
- Test: `transport/mod.rs` `#[cfg(test)]`

**Interfaces (Produces):**
- `pub struct ToolCall { pub id: String, pub name: String, pub args: serde_json::Value }`
- `pub struct TurnReply { pub text: Option<String>, pub tool_calls: Vec<ToolCall>, pub raw: serde_json::Value }`
- `pub trait ChatTransport { fn tools_schema(&self, graph_on: bool) -> Value; fn call(&self, ep: &crate::lab::Endpoint, system: Option<&str>, messages: &[Value], tools: Option<&Value>, temperature: f64) -> anyhow::Result<TurnReply>; fn push_assistant_turn(&self, messages: &mut Vec<Value>, reply: &TurnReply); fn push_tool_results(&self, messages: &mut Vec<Value>, results: &[(String, String)]); }`
- Move `runtime()` and `http_client()` here as `pub(crate)` (shared by both transports).

- [ ] **Step 1: Write failing test** — a trivial doc-level test that constructs a `TurnReply { text: Some("x"), tool_calls: vec![], raw: json!({}) }` and asserts field access; and that the trait object type `&dyn ChatTransport` names-resolve (compile check). (The real behavior tests live in Tasks 2-3.)
- [ ] **Step 2: Run to verify fail** (compile error — types undefined).
- [ ] **Step 3: Implement** the types + trait + move `runtime()`/`http_client()` (delete from openai.rs, re-export or reference from transport::mod).
- [ ] **Step 4: Run to verify pass. Step 5: Commit.**

### Task 2: OpenAI transport impl (wraps today's code, behavior-identical)

**Files:**
- Create: `eval/src/backend/transport/openai.rs`
- Modify: `eval/src/backend/openai.rs` (remove the transport-specific fns being moved; keep `OpenAiBackend` + `AgentBackend` impl for now)
- Test: move the relevant existing tests (`chat_once_posts_endpoint_url_verbatim`, `parse_tool_args_handles_string_and_object`) into `transport/openai.rs` and keep them green.

**Interfaces (Produces):**
- `pub struct OpenAiTransport;` implementing `ChatTransport`:
  - `tools_schema` = today's `tools_schema` (`{type:function,function:{...}}`).
  - `call` builds `{model, messages(+system as a message), tools?, temperature}` (OpenAI shape; system folded into messages as `{role:system}`), POSTs verbatim to `ep.endpoint` with Bearer auth via the moved `chat_http` core, extracts `choices[0].message`, returns `TurnReply` (text = `content_of`; tool_calls parsed from `message.tool_calls` via `parse_tool_args`; raw = the message).
  - `push_assistant_turn` = `messages.push(reply.raw.clone())` (preserves `reasoning_content`).
  - `push_tool_results` = push one `{role:"tool", tool_call_id:id, content:body}` per result.
- Keep `chat_once`/`lmstudio_chat` as THIN back-compat wrappers that delegate to `OpenAiTransport` (so external callers compile) OR migrate their callers in Task 3 — decide in the task and record the ruling.

- [ ] **Step 1: Write failing test** — construct `OpenAiTransport`, call `tools_schema(true)`, assert it contains `"type":"function"` and a graph tool name; assert `push_tool_results` appends a `{role:"tool"}` message with the given id+body. (Network `call` is covered by the moved verbatim-URL test against a local socket.)
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** `OpenAiTransport` by MOVING `chat_http`, `tools_schema`, `parse_tool_args`, `content_of` bodies here (behavior identical); wire `call`/`push_*` to them.
- [ ] **Step 4: Run to verify pass** — moved tests + suite green. **Step 5: Commit.**

### Task 3: Generic `run_agent_loop` over `ChatTransport` + `OpenAiBackend` rewire

**Files:**
- Create: `eval/src/backend/agent_loop.rs`
- Modify: `eval/src/backend/openai.rs` (`OpenAiBackend::answer` drives the generic loop with `OpenAiTransport`), `eval/src/backend/mod.rs`
- Test: MOVE the existing agent-loop tests (`loop_returns_direct_answer_*`, `loop_dispatches_tool_then_answers`, `loop_dedupes_*`, `loop_reexecutes_*`, `loop_stops_at_max_rounds`, `loop_unproductive_streak_*`, `agent_uses_injected_system_prompt`) to exercise the generic loop with a mock `ChatTransport`.

**Interfaces (Produces):**
- `pub fn run_agent_loop(transport: &dyn ChatTransport, ep: &Endpoint, system: Option<&str>, seed: Vec<Value>, tools: Option<&Value>, exec: impl FnMut(&str,&Value)->(String,Vec<String>), nba: impl Fn(&str,&Value)->String, max_rounds: usize) -> anyhow::Result<String>` — the loop calls `transport.call(...) -> TurnReply`, uses `reply.tool_calls` (neutral) for dedup/streak/NBA, `transport.push_assistant_turn` + `transport.push_tool_results` to advance messages; terminates on empty tool_calls (returns `reply.text`) or max_rounds. All dedup/streak/NBA/final-nudge logic is ported VERBATIM, now operating on the neutral `ToolCall` shape instead of raw `Value` pointers.
- The mock transport for tests: a struct wrapping a `Vec<TurnReply>` (scripted replies) recording pushed messages.

- [ ] **Step 1: Write the failing test** — port `loop_dispatches_tool_then_answers` to drive `run_agent_loop` with a scripted mock transport returning [reply-with-one-tool_call, reply-with-text]; assert the tool executed once and the final text returned. (This is the key behavior-preservation test.)
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** `run_agent_loop` generic; rewrite `OpenAiBackend::answer` to build `OpenAiTransport` + call it. Migrate the rest of the moved loop tests to the mock transport.
- [ ] **Step 4: Run the WHOLE suite** — `cargo test -p kb-eval` green (every ported loop test passes → behavior preserved). **Step 5: Commit.**

### Task 4: `ApiKind` config field (default OpenAi) + transport selector

**Files:**
- Modify: `eval/src/lab.rs` (`Endpoint` gains `#[serde(default)] pub api: ApiKind`), `eval/src/backend/transport/mod.rs` (a `for_api(api: ApiKind) -> Box<dyn ChatTransport>` selector)
- Test: `lab.rs` — an `Endpoint` toml WITHOUT `api` parses to `ApiKind::OpenAi`; WITH `api = "anthropic"` parses to `ApiKind::Anthropic`.

**Interfaces (Produces):**
- `pub enum ApiKind { #[default] OpenAi, Anthropic }` with serde rename `openai`/`anthropic`.
- `pub fn transport_for(api: ApiKind) -> Box<dyn ChatTransport>` (Phase 1: returns OpenAiTransport for both; Anthropic filled in Phase 2 — a `todo!`/error is NOT acceptable, so return OpenAi for Anthropic with a `log::warn` "anthropic transport not yet built, using openai" until Phase 2 Task 5 replaces it). RULING to record: temporary OpenAi-for-Anthropic stub is acceptable ONLY within Phase 1; Phase 2 Task 5 must replace it before any Anthropic endpoint is used.

- [ ] **Step 1: Failing test** — the two toml-parse assertions above. **Step 2: fail. Step 3: implement** `ApiKind` + `transport_for`. **Step 4: pass. Step 5: commit.**

**Phase 1 exit gate:** `cargo test -p kb-eval` fully green; `kbx build`/`eval`/`distil`/`train` behavior byte-identical (spot-check one live `kbx eval` on pilot-12 flat if a model is available). No Anthropic behavior yet.

---

## PHASE 2 — Anthropic transport (full agentic)

### Task 5: `AnthropicTransport` impl

**Files:**
- Create: `eval/src/backend/transport/anthropic.rs`
- Modify: `transport/mod.rs` (`transport_for` returns `AnthropicTransport` for `ApiKind::Anthropic`)
- Test: `transport/anthropic.rs` `#[cfg(test)]` (pure shape tests, no network) + a local-socket integration test.

**Interfaces:** `pub struct AnthropicTransport;` implementing `ChatTransport`:
- `tools_schema` → `[{"name","description","input_schema": <the registry's params_schema>}]` (drop OpenAI's `type`/`function` envelope; reuse `glossa::tools::registry()` schema payload).
- `call` → builds `{model, max_tokens: <configurable, default e.g. 4096>, system: <the system string, top-level>, messages, tools?, temperature}`; headers `x-api-key: <key>` + `anthropic-version: 2023-06-01` (constant, configurable later); POSTs verbatim to `ep.endpoint`; parses the top-level `content` block array → `TurnReply{ text = concatenated text blocks, tool_calls = [{id,name,args=input} for each type:"tool_use" block], raw = full response }`. Reuse the moved retry/backoff + raw-Value round-trip.
- `push_assistant_turn` → push `{role:"assistant", content: <reply.raw's content blocks>}` (preserve `thinking`/`tool_use` blocks verbatim from `reply.raw`).
- `push_tool_results` → push ONE `{role:"user", content: [{type:"tool_result", tool_use_id:id, content:body} for each result]}` (batched per Anthropic spec).

- [ ] **Step 1: Failing tests** — (a) `tools_schema` output has `input_schema` and NO `type:"function"`; (b) `push_tool_results` with two results appends exactly ONE user message whose content is two `tool_result` blocks with the right ids; (c) a parse helper turns a canned Anthropic response (`content:[{type:text,text:"hi"},{type:tool_use,id:"t1",name:"search",input:{"q":"x"}}]`) into `TurnReply{ text:Some("hi"), tool_calls:[{id:"t1",name:"search",args:{"q":"x"}}] }`.
- [ ] **Step 2: fail. Step 3: implement** `AnthropicTransport` (mirror `tensorzero.rs`'s block handling for structure). **Step 4: pass.**
- [ ] **Step 5: Integration test** — a `std::net::TcpListener` mock serving `/v1/messages`: assert the request has top-level `system`, `x-api-key` + `anthropic-version` headers, `max_tokens`, and Anthropic-shaped `tools`; return a `tool_use` response then a text response, and assert `run_agent_loop(&AnthropicTransport, ...)` runs the tool once and returns the final text. **Step 6: commit.**

### Task 6: Wire api-type selection into every call site

**Files:**
- Modify: `eval/src/judge.rs`, `eval/src/build/judge.rs`, `eval/src/train.rs`, `eval/src/build/extract.rs`, `eval/src/distil/chain.rs`, `eval/src/gepa_graph.rs` (+ its `GepaGraphConfig` gains `api: ApiKind`), `eval/src/backend/openai.rs` (`OpenAiBackend` gains `api: ApiKind`, uses `transport_for`), `eval/src/bin/kbx.rs`, `eval/src/main.rs` (add `--api openai|anthropic` flag)
- Test: an `OpenAiBackend`/`chat_once`-path unit that with `api=Anthropic` selects the Anthropic transport (assert via a shape check or a mock).

**Interfaces (Consumes):** `transport_for(ep.api)` at each site; replace direct `lmstudio_chat`/`chat_once` construction with `transport_for(ep.api).call(...)` / `run_agent_loop(&*transport_for(ep.api), ...)`.

- [ ] **Step 1: Failing test** — construct the eval reader `OpenAiBackend` with `api: Anthropic` and assert it drives the Anthropic transport (via a local-socket `/v1/messages` mock hit, or a transport-kind getter).
- [ ] **Step 2: fail. Step 3: thread `api` through all 9 sites** (each `Endpoint` consumer picks `transport_for(ep.api)`; `GepaGraphConfig`/`OpenAiBackend` gain the field; `main.rs` gets `--api`). **Step 4: whole suite green. Step 5: commit.**

**Phase 2 exit gate:** an Anthropic endpoint (e.g. OpenCode qwen3.7-plus) drives the agentic reader/build/distil AND one-shot judge/reflect. Live smoke against the real Anthropic endpoint when available.

---

## PHASE 3 — Resilience: per-endpoint rate-limit + fallback chain

### Task 7: `RateLimit` config + throttle + configurable retry/backoff

**Files:**
- Modify: `eval/src/lab.rs` (`Endpoint` gains `#[serde(default)] pub rate_limit: Option<RateLimit>`; `RateLimit { rpm: Option<u32>, max_inflight: Option<u32>, retry: Option<u32>, backoff_ms: Option<u64> }`)
- Create: `eval/src/backend/resilience.rs` (a per-endpoint token-bucket/min-interval throttle keyed by endpoint URL, + the retry/backoff policy currently hardcoded in `chat_http`, made configurable)
- Test: `resilience.rs` — a throttle permits N then blocks/sleeps deterministically (inject a fake clock via a passed-in `now`/sleep fn to keep it deterministic — do NOT call real `Date::now`); retry policy computes the right backoff sequence for a given config.

**Interfaces (Produces):** `pub fn throttle(ep_key: &str, rl: &RateLimit)` (acquire a slot, blocking to respect rpm/max_inflight); the existing retry loop in `chat_http`/the transport `call` reads `retry`/`backoff_ms` from `RateLimit` instead of the hardcoded `1..=4`.

- [ ] **Step 1: Failing test** — throttle with `rpm=60` (min-interval 1s) called twice with an injected clock sleeps ~1s between; retry config `retry=3, backoff_ms=500` yields backoff steps [500,1000,1500] (or the chosen policy). **Step 2: fail. Step 3: implement. Step 4: pass. Step 5: commit.**

### Task 8: Fallback chain (`Endpoint.fallback`)

**Files:**
- Modify: `eval/src/lab.rs` (`Endpoint` gains `#[serde(default)] pub fallback: Vec<Endpoint>`)
- Modify: `eval/src/backend/resilience.rs` (a `call_resilient(chain: &Endpoint, transport_for, make_call) -> Result<TurnReply>` that tries the primary, and on a hard failure — exhausted retries / 5xx / timeout / connect error — advances to each `fallback` Endpoint in order, each with its OWN `api`/`rate_limit`)
- Modify: the call sites to route transport calls through `call_resilient(ep, ...)` instead of a single `transport.call`
- Test: `resilience.rs` — `call_resilient` with a primary whose `make_call` always errors and a fallback that succeeds returns the fallback's reply; a chain where all fail returns the LAST error; a primary that succeeds never touches the fallback.

**Interfaces (Produces):** `pub fn call_resilient(chain: &Endpoint, system, messages, tools, temperature) -> anyhow::Result<TurnReply>` — resolves `transport_for(link.api)` per link, applies `throttle`+retry per link, falls through the `fallback` vec on hard failure. Recursion note: only the top-level `Endpoint`'s `fallback` is honored (fallbacks' own `fallback` fields are ignored — flat chain, documented) to avoid unbounded recursion.

- [ ] **Step 1: Failing test** — the three `call_resilient` cases above with mock `make_call`s (no network). **Step 2: fail. Step 3: implement** the flat-chain fallback. **Step 4: pass.**
- [ ] **Step 5: Route the 9 call sites** through `call_resilient`; whole suite green. **Step 6: commit.**

### Task 9: Live validation + docs

- [ ] **Step 1:** Rebuild `kbx`. Update `templates/lab.toml` comments to document `api`, `rate_limit`, and `[[<stage>.fallback]]` (abstract example, no secrets).
- [ ] **Step 2:** Live smoke: a `lab.toml` with `[model]` = a flaky/primary endpoint + a `[[model.fallback]]` = local 4b; run `kbx eval` on pilot-12 and confirm it FALLS BACK when the primary errors (simulate by pointing primary at a dead URL). Confirm rate-limit throttles (rpm=low) without errors. Scratch only.
- [ ] **Step 3:** Commit doc/template updates.

## Self-Review notes

- Coverage: trait+types (T1), OpenAI impl behavior-identical (T2), generic loop (T3), config api (T4) [Phase 1]; Anthropic transport (T5) + call-site wiring (T6) [Phase 2]; rate-limit (T7) + fallback (T8) + live (T9) [Phase 3].
- Type consistency: `ChatTransport`/`TurnReply`/`ToolCall` fixed in T1, consumed unchanged T2-T8. `ApiKind` T4 → used T5/T6. `RateLimit`/`fallback` T7/T8.
- Behavior-preservation is the #1 risk (T2-T3 refactor a core tested module): every existing openai.rs test MOVES and stays green; no test is deleted, only relocated to exercise the new seam.
- Placeholder scan: the Phase-1 `transport_for` OpenAi-for-Anthropic stub is a RULING-gated temporary (replaced in T5), not a silent placeholder.
- Determinism: resilience tests inject clock/sleep (no real `Date::now`/`Math::random`), per the codebase's determinism rule.
- Do NOT touch `tensorzero.rs` (template only) or `glossa_tools::exec` (transport-agnostic).
