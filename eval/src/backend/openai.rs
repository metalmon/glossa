use super::{prompt, AgentBackend};
use crate::dataset::Question;
use anyhow::anyhow;
use glossa::read::DocImage;
use glossa::trace::TraceLog;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

/// Process-global running total of tokens consumed across every chat call routed through
/// `chat_http` — every one of reason/build/eval/distil goes through that single function, so this
/// one counter tallies all of them without each call site needing to thread usage back up itself.
static TOKENS_USED: AtomicU64 = AtomicU64::new(0);

/// Current value of the running token counter (see `TOKENS_USED`).
pub fn tokens_used() -> u64 {
    TOKENS_USED.load(Ordering::Relaxed)
}

/// Zero the running token counter. Call at the start of a run loop that wants its own per-run
/// total (reason/build-extract/build-judge/eval each call this before their loop starts), so one
/// stage's bar reflects only tokens spent in that stage, not a prior one's leftover total.
pub fn reset_tokens() {
    TOKENS_USED.store(0, Ordering::Relaxed);
}

/// Process-global running count of resamples performed by `lmstudio_chat`'s resample loop (both the
/// length-cap path and the degenerate-loop path) across every chat call — mirrors `TOKENS_USED` so a
/// run loop can surface `resamples()` on the same progress-bar message instead of the resample
/// diagnostic colliding with the bar via a raw `eprintln!`.
static RESAMPLES: AtomicU64 = AtomicU64::new(0);

/// Current value of the running resample counter (see `RESAMPLES`).
pub fn resamples() -> u64 {
    RESAMPLES.load(Ordering::Relaxed)
}

/// Zero the running resample counter. Call at the start of a run loop, alongside `reset_tokens()`,
/// so one stage's bar reflects only resamples spent in that stage.
pub fn reset_resamples() {
    RESAMPLES.store(0, Ordering::Relaxed);
}

/// Extract total token usage from a parsed chat-completions response `resp`: `usage.total_tokens`
/// when present, else `usage.prompt_tokens + usage.completion_tokens`, else `0` when `usage` is
/// absent entirely (some OpenAI-compatible servers omit it). Factored out of `chat_http` as a pure
/// function purely so this extraction logic is unit-testable without a live server.
pub fn usage_tokens(resp: &Value) -> u64 {
    let usage = resp.get("usage");
    if let Some(t) = usage.and_then(|u| u.get("total_tokens")).and_then(Value::as_u64) {
        return t;
    }
    let prompt = usage
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let completion = usage
        .and_then(|u| u.get("completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    prompt + completion
}

/// Format a token count compactly for a progress-bar message: `999` -> `"999"`, `1500` ->
/// `"1.5k"`, `2_000_000` -> `"2.0M"`. Pure — unit-tested.
pub fn human_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Generic OpenAI-compatible chat backend (LM Studio, llama.cpp server, vLLM, OpenRouter, …).
///
/// The harness itself is the agent: it advertises glossa's `search`/`read` as OpenAI function
/// tools, runs the tool-call loop, and executes the tools IN-PROCESS against the corpus in `work`.
/// (We do NOT rely on the server's own MCP/tool execution — that is GUI-only in LM Studio and
/// makes retrieval unobservable.) Every tool call is logged to `work/.glossa/traces` in the same
/// JSONL format the MCP server uses, so `run::eval_one` measures retrieval-recall unchanged.
pub struct OpenAiBackend {
    pub endpoint: String, // full chat-completions URL, e.g. "http://localhost:1234/v1/chat/completions"
    pub model: String,
    pub api_key: Option<String>,
    pub timeout: Duration,
    /// graph-ON arm when true (opens the graph and advertises the graph tools); graph-OFF
    /// baseline when false (flat search/read only). The A/B knob for the graph-transfer eval.
    pub use_graph: bool,
    /// Runtime-injected system prompt (e.g. loaded from an editable `.md` file at launch), used
    /// VERBATIM as the system message when `Some`. `None` preserves today's behavior exactly:
    /// the compiled `prompt::system_prompt(self.use_graph)`. This is what lets the reader's
    /// prompt be edited without a rebuild.
    pub system_prompt: Option<String>,
}

const MAX_ROUNDS: usize = 50;

/// Default `min_p` nucleus-sampling floor for the agent-loop chat call (`lmstudio_chat`),
/// overridable via `KB_EVAL_MIN_P`. Trims the tail of low-probability tokens that otherwise widen at
/// high temperature, cutting down on degenerate generation loops without forcing low-temperature
/// (deterministic) sampling. NOT sent by `chat_once` — that path is already greedy (temperature 0)
/// and targets a strict provider that may reject non-OpenAI fields.
const DEFAULT_MIN_P: f64 = 0.1;
/// Cap on completion length per call. Bounds a runaway generation (a degenerate loop can otherwise
/// spew thousands of tokens before the server stops) while staying generous enough not to clip a
/// legitimate multi-node `graph_upsert` batch or a normal reader answer. Overridable via
/// `KB_EVAL_MAX_TOKENS`.
const DEFAULT_MAX_TOKENS: u64 = 16384;

/// How many times to resample a completion whose content is a degenerate repetition loop before
/// giving up and returning the last (best-effort) result.
const GEN_LOOP_RETRIES: usize = 2;

/// Cap on how many times a single call resamples on `finish_reason == "length"` alone before giving
/// up and accepting the truncated best-effort completion. A chronically-verbose model that overflows
/// `max_tokens` every round would otherwise burn up to `GEN_LOOP_RETRIES` full generations per call
/// (observed: ~3x tokens wasted on an over-cap chat) — length overruns are rarely fixed by a single
/// stochastic resample, so this is capped tighter than the generic loop-resample bound. Overridable
/// via `KB_EVAL_MAX_LENGTH_RESAMPLE`.
const DEFAULT_MAX_LENGTH_RESAMPLE: usize = 1;

impl AgentBackend for OpenAiBackend {
    fn needs_corpus(&self) -> bool {
        true
    }

    fn answer(&self, work: &Path, q: &Question) -> anyhow::Result<String> {
        // The endpoint is the full chat-completions URL, used verbatim (no path is appended).
        let url = self.endpoint.clone();
        let graph = if self.use_graph {
            glossa::graph::store::GraphStore::open(work).ok()
        } else {
            None
        };
        let tools = tools_schema(graph.is_some());
        let chat = |messages: &[Value]| {
            lmstudio_chat(
                &url,
                &self.model,
                self.api_key.as_deref(),
                &tools,
                messages,
                self.timeout,
            )
        };

        let trace = TraceLog::to_dir(work);
        // Open the index once per question; the closure reuses it (cached reader) for every
        // search/read in the agent loop instead of reopening per tool call.
        let idx = glossa::index::store::DocIndex::open_or_create(work)?;
        // Ontology-driven chain spec so glossary/related render identically to the MCP surface.
        let spec = glossa::tools::ChainSpec::from_ontology(
            &glossa::graph::ontology::Ontology::load_or_default(work),
        );
        let exec = |name: &str, args: &Value| {
            let (body, ids) = execute_tool(name, args, work, &idx, graph.as_ref(), &spec, &trace);
            // Diagnostics: KB_EVAL_DUMP_TOOLS=1 prints each tool call + a truncated body to
            // stderr, so a smoke run doubles as an episode transcript (why the reader searches).
            if std::env::var("KB_EVAL_DUMP_TOOLS").is_ok() {
                let snippet: String = body.chars().take(500).collect();
                eprintln!("\n[TOOL] {name} {args}\n[BODY] {snippet}\n[--- {} chars ---]", body.len());
            }
            // The reader path never feeds vision input — `execute_tool` already discards
            // whatever images `glossa_tools::exec` surfaced (e.g. from `read`). Only
            // `build::extract::extract_doc`'s `--vision` path populates this slot.
            (body, ids, Vec::<DocImage>::new())
        };

        let messages = self.build_messages(q);
        // Next-best-action on a stuck (repeated) call: fan the fixated query across the
        // complementary tools instead of re-running the dead one.
        let nba = |name: &str, args: &Value| {
            crate::backend::glossa_tools::next_best_action(
                name, args, work, &idx, graph.as_ref(), &spec, &trace,
            )
        };
        let raw = run_agent_loop(chat, messages, exec, nba, MAX_ROUNDS)?;
        Ok(prompt::parse_answer(&raw))
    }
}

impl OpenAiBackend {
    /// Assemble the two seed messages (system + user) for one question. When `self.system_prompt`
    /// is `Some`, its content becomes the system message VERBATIM (a runtime `.md` override —
    /// no rebuild needed to edit the reader's prompt); `None` preserves today's behavior exactly:
    /// the compiled `prompt::system_prompt(self.use_graph)`. (`GraphStore::open` in `answer()`
    /// creates the store on demand, so `graph.is_some()` there and `self.use_graph` here agree in
    /// practice — the config flag is what actually decides which prompt variant is compiled.)
    fn build_messages(&self, q: &Question) -> Vec<Value> {
        let system = match &self.system_prompt {
            Some(s) => s.clone(),
            None => prompt::system_prompt(self.use_graph).to_string(),
        };
        vec![
            json!({ "role": "system", "content": system }),
            json!({ "role": "user", "content": prompt::user_prompt(q) }),
        ]
    }

    /// Test-only constructor: minimal backend with an injected system prompt, for exercising
    /// `build_messages` without a live endpoint or corpus.
    #[cfg(test)]
    fn for_test_with_prompt(s: &str) -> Self {
        OpenAiBackend {
            endpoint: String::new(),
            model: String::new(),
            api_key: None,
            timeout: Duration::from_secs(1),
            use_graph: false,
            system_prompt: Some(s.to_string()),
        }
    }
}

/// Minimal one-shot OpenAI-compatible chat call: a plain completion (e.g. the file-prompt judge in
/// `judge.rs`) instead of the full tool-calling agent loop. Builds a greedy (temperature 0), tools
/// free request body and drives it through `chat_http` — the same transport the agent loop uses.
pub(crate) fn chat_once(
    endpoint: &str,
    model: &str,
    messages: &[Value],
    api_key: Option<&str>,
    timeout_secs: u64,
) -> anyhow::Result<Value> {
    // One-shot (the file-prompt judge): greedy sampling so grading is reproducible run-to-run
    // (NOT the reader's stochastic KB_EVAL_TEMP), and NO `tools` field at all — a strict provider
    // (the MiMo/OpenCode Zen endpoint this backend targets) rejects an empty `tools: []` array.
    // NOTE: no `min_p` here either — it's a non-OpenAI extension a strict provider may 400 on, and
    // it would be inert anyway since this path is greedy (temperature 0).
    let max_tokens: u64 = std::env::var("KB_EVAL_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_TOKENS);
    let body = json!({
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    });
    chat_http(endpoint, api_key, &body, Duration::from_secs(timeout_secs))
        .map(|v| v.pointer("/choices/0/message").cloned().unwrap_or_else(|| json!({})))
}

/// Shared tokio runtime backing the sync bridge below: the agent loop (`run_agent_loop` and every
/// caller of `lmstudio_chat`/`chat_once`) is deliberately synchronous, but `reqwest`'s client is
/// async-only. One process-wide multi-thread runtime lets every call just `block_on` instead of
/// threading an executor through the whole sync call stack.
fn runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("failed to build the tokio runtime backing the http chat bridge")
    })
}

/// Shared reqwest client (connection-pool + TLS-session reuse across chat calls). The per-request
/// timeout is set on the `RequestBuilder`, so one client serves the reader and judge endpoints even
/// when their `timeout_secs` differ.
fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(reqwest::Client::new)
}

/// True when an error body reflects a transient UPSTREAM failure of the gateway's own backend (its
/// fetch/predict to the model dropped, timed out, or was overloaded) rather than a fault in OUR
/// request. Some OpenAI-compatible gateways (e.g. opencode zen) surface these with a client-error
/// status (400) or an HTTP-200 `{"error"}` body, so status code alone misclassifies them as fatal.
/// A genuine bad-request — bad key, malformed payload, unknown model — matches none of these and
/// still fails fast, so a real config bug isn't hidden behind seconds of backoff.
fn is_transient_upstream(body: &str) -> bool {
    let b = body.to_ascii_lowercase();
    [
        "fetch failed",
        "predict request failed",
        "engine protocol",
        "upstream",
        "timed out",
        "timeout",
        "temporarily",
        "overloaded",
        "connection reset",
        "bad gateway",
        "service unavailable",
        "try again",
    ]
    .iter()
    .any(|needle| b.contains(needle))
}

/// Sync HTTP bridge: POST our JSON request `body` (the `{model, messages, tools, temperature}` shape
/// `lmstudio_chat`/`chat_once` build) to an OpenAI-compatible `/v1/chat/completions` endpoint via
/// `reqwest` and return the FULL parsed response body as a raw `Value` (validated to have
/// `choices[0].message`, but NOT reduced to just that field — see below).
///
/// Deliberately RAW `Value` in and out — no typed OpenAI SDK structs — so provider-specific fields
/// survive the round-trip. In particular reasoning models (MiMo on OpenCode Zen) return
/// `reasoning_content` on assistant tool-call turns and require it echoed back on the next request;
/// typed message structs drop it and the provider then rejects the follow-up with HTTP 400.
///
/// Returns the whole response (not just `choices[0].message`) so callers can also read
/// `choices[0].finish_reason` (e.g. `lmstudio_chat`'s length-resample). Callers that only want the
/// assistant message extract it themselves — `finish_reason` must NEVER be injected into the
/// message object itself, since that object is echoed back into `messages` on the next request and
/// a strict endpoint may 400 on an unrecognized field.
///
/// `endpoint` is the FULL chat-completions URL (e.g. `http://localhost:1234/v1/chat/completions`),
/// POSTed verbatim — this function appends nothing. Callers configure the complete URL in
/// `lab.toml`'s `endpoint`, so the reader/judge/build/reflect paths all hit the URL as given with
/// no hidden path-rewriting (which previously double-appended `/v1` on the non-normalizing path).
fn chat_http(
    endpoint: &str,
    api_key: Option<&str>,
    body: &Value,
    timeout: Duration,
) -> anyhow::Result<Value> {
    let url = endpoint.to_string();

    // Trim trailing whitespace from every message: some strict providers (OpenCode Zen / mimo)
    // return HTTP 400 when a message ends in a newline. Harmless to content semantics.
    let mut body = body.clone();
    if let Some(msgs) = body.get_mut("messages").and_then(Value::as_array_mut) {
        for m in msgs {
            if let Some(s) = m.get("content").and_then(Value::as_str) {
                let trimmed = s.trim_end().to_string();
                m["content"] = Value::String(trimmed);
            }
        }
    }

    // RAW JSON round-trip via reqwest — deliberately NOT async-openai's typed message structs.
    // Reasoning models like MiMo (OpenCode Zen) return `reasoning_content` on assistant tool-call
    // turns and REQUIRE it echoed back on the next request; the SDK's typed structs silently drop
    // that field, causing HTTP 400 mid-loop. Passing raw `Value` messages preserves it end-to-end.
    // reqwest still gives robust transport/TLS; we retry ONLY transient failures (transport drop,
    // 429 rate-limit, 5xx) — a 4xx (bad key / malformed request) or a parse error is a hard error
    // surfaced immediately, so a real config bug isn't hidden behind seconds of backoff.
    let mut last_err = None;
    for attempt in 1..=4u32 {
        let (retryable, outcome): (bool, anyhow::Result<Value>) = runtime().block_on(async {
            let mut rb = http_client().post(&url).timeout(timeout).json(&body);
            if let Some(key) = api_key.filter(|k| !k.is_empty()) {
                rb = rb.bearer_auth(key);
            }
            let resp = match rb.send().await {
                Ok(r) => r,
                Err(e) => return (true, Err(anyhow!("send chat request: {e}"))),
            };
            let status = resp.status();
            let text = match resp.text().await {
                Ok(t) => t,
                Err(e) => return (true, Err(anyhow!("read chat response body: {e}"))),
            };
            if !status.is_success() {
                // 429 / 5xx are transient; so is a 4xx (often 400) whose BODY is an UPSTREAM
                // failure the gateway surfaced with a client-error status (e.g. opencode zen's
                // "Engine protocol predict request failed: fetch failed"). A genuine bad-request
                // (bad key, malformed payload, unknown model) does NOT match and still fails fast.
                let retryable = status.as_u16() == 429
                    || status.is_server_error()
                    || is_transient_upstream(&text);
                return (
                    retryable,
                    Err(anyhow!(
                        "chat endpoint returned {status}: {}",
                        text.chars().take(400).collect::<String>()
                    )),
                );
            }
            match serde_json::from_str::<Value>(&text) {
                Ok(v) => {
                    // Some OpenAI-compatible servers (LM Studio) return HTTP 200 with an
                    // `{"error": …}` body instead of a non-2xx status — surface it, don't discard
                    // it behind a vague "no choices" message.
                    if let Some(err) = v.get("error") {
                        // Some gateways return HTTP 200 with an `{"error": …}` body; if that error
                        // is an upstream transient (same wording as the non-2xx path), retry it too.
                        let retryable = is_transient_upstream(&err.to_string());
                        (retryable, Err(anyhow!("chat endpoint returned an error: {err}")))
                    } else if v.pointer("/choices/0/message").is_some() {
                        // Tally this call's usage into the process-global running counter (see
                        // `usage_tokens`/`TOKENS_USED`) before handing the response back.
                        TOKENS_USED.fetch_add(usage_tokens(&v), Ordering::Relaxed);
                        // Return the FULL response — see the doc comment above for why.
                        (false, Ok(v))
                    } else {
                        (false, Err(anyhow!("chat response had no choices[0].message")))
                    }
                }
                Err(e) => (false, Err(anyhow!("parse chat response json: {e}"))),
            }
        });
        match outcome {
            Ok(msg) => return Ok(msg),
            Err(e) => {
                last_err = Some(e);
                if retryable && attempt < 4 {
                    std::thread::sleep(Duration::from_millis(400 * attempt as u64));
                } else {
                    break;
                }
            }
        }
    }
    Err(last_err.unwrap())
}

/// One OpenAI-compatible `/v1/chat/completions` call to an LM Studio-style endpoint, driven through
/// the raw `reqwest` bridge (see `chat_http`). Returns the assistant `message` object (already extracted
/// from `choices[0].message`). Samples at `temperature: 0.8` — temp 0 is not a reliable greedy mode
/// on this reasoning model/backend (unstable outputs), so runs are stochastic and must be averaged
/// over N. Also sends `min_p` (default `DEFAULT_MIN_P`, overridable via `KB_EVAL_MIN_P`) to trim the
/// low-probability tail at that temperature. Shared by the eval backend and the graph GEPA optimizer
/// so both drive the same server the same way.
///
/// Resamples up to `GEN_LOOP_RETRIES` times before giving up and returning the last (best-effort)
/// result — sampling is stochastic, so a resample almost always breaks the problem — when either:
/// - the completion looks like a degenerate generation loop (see `looks_looped`), or
/// - the completion hit the token cap (`finish_reason == "length"`): the model burned its budget on
///   verbose content-reasoning and either never emitted the tool call or emitted a truncated one
///   whose JSON args don't parse. A resample at the same (stochastic) temperature is often terser.
/// See `should_resample` for the (unit-tested) decision predicate.
///
/// `url` is the FULL chat-completions URL and is passed through to `chat_http` verbatim.
pub(crate) fn lmstudio_chat(
    url: &str,
    model: &str,
    api_key: Option<&str>,
    tools: &Value,
    messages: &[Value],
    timeout: Duration,
) -> anyhow::Result<Value> {
    // Sampling temperature: overridable via KB_EVAL_TEMP for noise-sensitivity runs (default 0.8).
    let temperature: f64 = std::env::var("KB_EVAL_TEMP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.8);
    let min_p: f64 = std::env::var("KB_EVAL_MIN_P")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MIN_P);
    let max_tokens: u64 = std::env::var("KB_EVAL_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_TOKENS);
    let body = json!({
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": temperature,
        "min_p": min_p,
        "max_tokens": max_tokens
    });
    // Diagnostics: KB_EVAL_DUMP_REQ=<path> writes the exact request body (incl. the `tools` array
    // with descriptions) sent to the endpoint, to prove what the model actually receives.
    if let Ok(p) = std::env::var("KB_EVAL_DUMP_REQ") {
        let _ = std::fs::write(&p, serde_json::to_string(&body)?);
    }
    let max_length_resample: usize = std::env::var("KB_EVAL_MAX_LENGTH_RESAMPLE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_LENGTH_RESAMPLE);
    let mut length_resamples = 0usize;
    let mut full = chat_http(url, api_key, &body, timeout)?;
    for _ in 0..GEN_LOOP_RETRIES {
        let content = full
            .pointer("/choices/0/message/content")
            .and_then(|c| c.as_str())
            .unwrap_or("");
        let finish = full
            .pointer("/choices/0/finish_reason")
            .and_then(|f| f.as_str())
            .unwrap_or("");
        if !should_resample(finish, content) {
            break;
        }
        if finish == "length" {
            // Length overruns get a tighter, separately-tracked cap: resampling rarely fixes a
            // chronically-verbose model, so don't spend the full GEN_LOOP_RETRIES budget on it.
            if length_resamples >= max_length_resample {
                break;
            }
            length_resamples += 1;
        }
        RESAMPLES.fetch_add(1, Ordering::Relaxed);
        full = chat_http(url, api_key, &body, timeout)?;
    }
    full.pointer("/choices/0/message")
        .cloned()
        .ok_or_else(|| anyhow!("chat response had no choices[0].message"))
}

/// Decision predicate for `lmstudio_chat`'s resample loop: true when the completion should be
/// resampled rather than accepted — either it hit the token cap (`finish_reason == "length"`, a
/// truncated verbose turn that likely never emitted a valid tool call) or its content looks like a
/// degenerate repetition loop (see `looks_looped`). Pure and separated from the loop so the decision
/// itself is unit-testable without a mock HTTP server.
fn should_resample(finish: &str, content: &str) -> bool {
    finish == "length" || looks_looped(content)
}

/// Detect a degenerate generation loop: the tail of `text` is N identical consecutive blocks of some
/// short period (a phrase/line the model repeated until it ran out of tokens). Byte-level, so it also
/// catches whitespace-y repeats; returns false for normal prose. Pure — unit-tested.
fn looks_looped(text: &str) -> bool {
    let t = text.trim_end();
    let b = t.as_bytes();
    let n = b.len();
    if n < 60 {
        return false;
    } // too short to judge
    let tail = 1200.min(n);
    const REPS: usize = 4; // need >=4 identical consecutive blocks
    for p in 4..=(tail / REPS) {
        // period (block length) in bytes
        let span = REPS * p;
        if span > tail {
            break;
        }
        let start = n - span;
        let unit = &b[n - p..];
        if (0..REPS).all(|k| &b[start + k * p..start + (k + 1) * p] == unit) {
            return true;
        }
    }
    false
}

/// OpenAI function-tool schema for glossa's agent-facing tools, rendered from the single
/// shared registry (`glossa::tools::registry::registry()`) instead of a hand-written per-tool
/// JSON block — MCP and the eval agent can no longer drift apart on name/description/schema.
/// Graph-gated descriptors (glossary/reach/sql) are included only when `graph_on`; registry
/// order is preserved as-is (search/read/grep/glob first, then the graph tools), so ordering
/// here is a byproduct of the registry, not a curated hand-order.
pub(crate) fn tools_schema(graph_on: bool) -> Value {
    let tools: Vec<Value> = glossa::tools::registry::registry()
        .iter()
        .filter(|d| !d.graph_gated || graph_on)
        .map(|d| {
            json!({
                "type": "function",
                "function": {
                    "name": d.name,
                    "description": d.description,
                    "parameters": d.params_schema,
                }
            })
        })
        .collect();
    Value::Array(tools)
}

/// Unproductive-streak threshold: this many consecutive REAL (non-deduped) tool calls in a row that
/// each surface zero new identifiers trips the steer. Named so the TDD tests and the loop agree on
/// one number instead of a magic literal in two places. `pub(crate)` so callers outside this module
/// (e.g. `build::extract`'s regression test for the graph_upsert ids fix) can size their fixtures
/// off the real threshold instead of duplicating the literal.
pub(crate) const UNPRODUCTIVE_STREAK_K: usize = 3;

/// Cap on how many images one `exec` call's tool result feeds the model in a single vision
/// user-message, per Task-spec guard against a context flood (a figure-heavy scanned page can
/// return many images; `to_jpeg` bounds each one's SIZE but not the COUNT). Extras are dropped
/// with a logged note — never a silent truncation.
pub(crate) const MAX_IMAGES_PER_TURN: usize = 4;

/// Build the vision-input user message for a set of images returned by a tool call this round, or
/// `None` when there are none. `--vision`-only mechanism (see `run_agent_loop`): the OpenAI-
/// compatible `/v1/chat/completions` shape has no image slot on a `role:"tool"` message, so images
/// ride in a FOLLOW-UP `role:"user"` message whose `content` is an array — one leading text part
/// plus one `image_url` part per image, each a `data:image/jpeg;base64,<payload>` URI.
///
/// Every image is normalized through [`glossa::read::to_jpeg`] first (JPEG passes through
/// untouched; anything else is decoded and re-encoded) and base64-encoded with the STANDARD
/// (padded, unwrapped) alphabet — canonical, no embedded whitespace/newlines, since a malformed
/// `image_url.url` 400s on our opencode-zen endpoint.
///
/// Caps the images fed to [`MAX_IMAGES_PER_TURN`]; when the tool call returned more, the extras
/// are dropped and a note is printed to stderr (no silent truncation — see the constraint above).
fn vision_user_message(images: &[DocImage]) -> Option<Value> {
    if images.is_empty() {
        return None;
    }
    use base64::Engine as _;
    let total = images.len();
    let capped: Vec<&DocImage> = images.iter().take(MAX_IMAGES_PER_TURN).collect();
    if total > MAX_IMAGES_PER_TURN {
        eprintln!(
            "[vision] {total} image(s) returned this turn; feeding only the first \
             {MAX_IMAGES_PER_TURN} (dropping {})",
            total - MAX_IMAGES_PER_TURN
        );
    }
    let mut content = vec![json!({
        "type": "text",
        "text": format!("Images from that read ({}):", capped.len())
    })];
    for img in capped {
        let jpeg = glossa::read::to_jpeg(img.clone());
        let payload = base64::engine::general_purpose::STANDARD.encode(&jpeg.bytes);
        content.push(json!({
            "type": "image_url",
            "image_url": { "url": format!("data:image/jpeg;base64,{payload}") }
        }));
    }
    Some(json!({ "role": "user", "content": Value::Array(content) }))
}

/// Drive a tool-calling chat to a final textual answer.
///
/// `chat(messages)` returns the assistant `message` object (already extracted from
/// `choices[0].message`). When it carries `tool_calls`, each is dispatched through `exec(name,
/// args)` and the result fed back as a `role:"tool"` message, then the model is queried again —
/// up to `max_rounds`. The first message without tool calls yields the answer.
///
/// Two independent stuck detectors sit on top of `exec`:
/// - **Identical-repeat dedup**: the previous (tool, args) actually executed. When the model
///   re-issues the SAME call, it isn't re-run (identical result) — `on_repeat` (the next-best-action)
///   fires instead. This takes priority and doesn't touch the streak below (a dedup hit didn't
///   execute, so it can't be "unproductive").
/// - **Unproductive streak**: the model issues many DIFFERENT calls (varied tool/args — so they all
///   really execute) that each surface no NEW identifier — a search-flood that never progresses.
///   `exec` returns `(body, ids)`; `ids` are what a session-aware MCP server would track per call
///   (search hit locations, a read's path, …). Any id not already in `seen` resets the streak;
///   otherwise it grows. At `UNPRODUCTIVE_STREAK_K` the fed-back tool content becomes the steer
///   (`glossa_tools::unproductive_steer`) instead of the (already-seen) body, and the counter resets
///   so it fires once per streak, not on every call past the threshold.
pub(crate) fn run_agent_loop<C, F, N>(
    mut chat: C,
    mut messages: Vec<Value>,
    mut exec: F,
    mut on_repeat: N,
    max_rounds: usize,
) -> anyhow::Result<String>
where
    C: FnMut(&[Value]) -> anyhow::Result<Value>,
    F: FnMut(&str, &Value) -> (String, Vec<String>, Vec<DocImage>),
    N: FnMut(&str, &Value) -> String,
{
    // Stuck-detection substrate: the previous (tool, args) actually executed. When the model
    // re-issues the SAME call, we don't re-run it (identical result) — we hand off to `on_repeat`,
    // the next-best-action. Default callers pass `repeat_nudge`; the reader path passes a fan-out.
    let mut last_key: Option<String> = None;
    // Novelty tracking for the unproductive-streak detector (see the doc comment above).
    let mut seen: HashSet<String> = HashSet::new();
    let mut unproductive: usize = 0;
    for _ in 0..max_rounds {
        let msg = chat(&messages)?;
        let calls: Vec<Value> = msg
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        if calls.is_empty() {
            return Ok(content_of(&msg));
        }
        messages.push(msg.clone()); // echo the assistant turn that requested the tools
        for call in &calls {
            let id = call.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let name = call
                .pointer("/function/name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let args = parse_tool_args(call);
            let key = format!("{name}\u{1}{}", serde_json::to_string(&args).unwrap_or_default());
            let (result, images) = if last_key.as_deref() == Some(key.as_str()) {
                (on_repeat(name, &args), Vec::new())
            } else {
                let (body, ids, images) = exec(name, &args);
                last_key = Some(key);
                let has_new = ids.into_iter().fold(false, |acc, i| seen.insert(i) || acc);
                let text = if has_new {
                    unproductive = 0;
                    body
                } else {
                    unproductive += 1;
                    if unproductive >= UNPRODUCTIVE_STREAK_K {
                        unproductive = 0;
                        crate::backend::glossa_tools::unproductive_steer(name)
                    } else {
                        body
                    }
                };
                (text, images)
            };
            messages.push(json!({ "role": "tool", "tool_call_id": id, "content": result }));
            // Vision (build --vision only; every other caller's exec always returns empty here):
            // a tool-role message can't carry image content on this endpoint, so any images the
            // call surfaced ride in ONE follow-up user message right after it.
            if let Some(img_msg) = vision_user_message(&images) {
                messages.push(img_msg);
            }
        }
    }
    // Out of rounds: nudge for a final answer (the model often keeps requesting tools otherwise)
    // and take whatever text it gives.
    messages.push(json!({
        "role": "user",
        "content": "Stop searching. Give your final answer now on a single line beginning with `ANSWER:`."
    }));
    let msg = chat(&messages)?;
    Ok(content_of(&msg))
}

fn content_of(msg: &Value) -> String {
    msg.get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string()
}

/// Tool-call `function.arguments` is a JSON-encoded string per the OpenAI spec, but some servers
/// (incl. some LM Studio builds) return it as an already-parsed object. Accept both.
fn parse_tool_args(call: &Value) -> Value {
    match call.pointer("/function/arguments") {
        Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
        Some(v @ Value::Object(_)) => v.clone(),
        _ => json!({}),
    }
}

/// Execute one glossa tool in-process against the corpus in `work`, logging it to the trace
/// (same shape as the MCP server: search → array of {path,location,score}; read → {path}).
///
/// Returns `(body, ids)` — `ids` are the identifiers this call surfaced (what a session-aware MCP
/// server would track for novelty), from `glossa_tools::exec`'s second return value: `search`'s hit
/// locations, and the graph tools' (glossary/related/neighbors/reach/sql) `path#ord`
/// read-anchor ids scraped from their rendered bodies. `read` itself surfaces no ids there, so it's
/// special-cased here to the `path` argument instead. `run_agent_loop` uses these to detect an
/// unproductive streak — many varied calls (including varied graph navigation) that surface
/// nothing new — without falsely tripping on a reader that IS making real graph progress.
fn execute_tool(
    name: &str,
    args: &Value,
    root: &Path,
    idx: &glossa::index::store::DocIndex,
    graph: Option<&glossa::graph::store::GraphStore>,
    spec: &glossa::tools::ChainSpec,
    trace: &TraceLog,
) -> (String, Vec<String>) {
    // No registry-membership pre-check here: `registry()` is the ADVERTISING source (what
    // `tools_schema` puts in front of the model); execution dispatches whatever
    // `glossa_tools::exec` supports, which is a superset (it also serves non-agent-facing
    // callers, e.g. related/neighbors for MCP's Editor/Full profiles). `exec` already returns
    // its own "unknown tool" body for names it genuinely doesn't handle, so it is the sole gate.
    let (body, ids, _images) =
        crate::backend::glossa_tools::exec(name, args, root, idx, graph, spec, trace);
    let ids = if name == "read" {
        // Mirror glossa_tools::exec's own raw_arguments fallback so a stringified args object
        // still yields the path.
        let parsed;
        let a = if let Some(s) = args.as_str() {
            parsed = serde_json::from_str::<Value>(s).unwrap_or_else(|_| json!({}));
            &parsed
        } else {
            args
        };
        a.get("path")
            .and_then(|v| v.as_str())
            .filter(|p| !p.is_empty())
            .map(|p| vec![p.to_string()])
            .unwrap_or_default()
    } else {
        ids
    };
    (body, ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn transient_upstream_retries_but_real_bad_request_fails_fast() {
        // Upstream failures a gateway surfaces with a 400 / 200-error body — retry these.
        for transient in [
            r#"{"error":"Engine protocol predict request failed: fetch failed"}"#,
            "upstream connect error",
            "The service is temporarily overloaded, try again",
            "502 Bad Gateway",
            "read timed out",
        ] {
            assert!(is_transient_upstream(transient), "should retry: {transient}");
        }
        // Genuine client faults — must fail fast, never retry.
        for fatal in [
            r#"{"error":{"message":"Invalid API key provided"}}"#,
            r#"{"error":{"message":"model 'foo' does not exist"}}"#,
            r#"{"error":{"message":"invalid 'messages': malformed request"}}"#,
        ] {
            assert!(!is_transient_upstream(fatal), "should fail fast: {fatal}");
        }
    }

    #[test]
    fn usage_tokens_prefers_total_falls_back_to_sum_then_zero() {
        // total_tokens present -> used directly, even if it disagrees with the sum (trust the
        // server's own total over reconstructing it).
        assert_eq!(
            usage_tokens(&json!({"usage": {"total_tokens": 42, "prompt_tokens": 1, "completion_tokens": 1}})),
            42
        );
        // No total_tokens -> fall back to prompt_tokens + completion_tokens.
        assert_eq!(
            usage_tokens(&json!({"usage": {"prompt_tokens": 10, "completion_tokens": 5}})),
            15
        );
        // No usage object at all -> 0, never a panic.
        assert_eq!(usage_tokens(&json!({"choices": []})), 0);
    }

    #[test]
    fn human_tokens_formats_compactly() {
        assert_eq!(human_tokens(999), "999");
        assert_eq!(human_tokens(1500), "1.5k");
        assert_eq!(human_tokens(2_000_000), "2.0M");
    }

    #[test]
    fn tokens_used_resets_and_accumulates() {
        // Process-global counter — reset first so this test isn't order-dependent on whatever
        // other tests in this file (or run concurrently) touched it.
        reset_tokens();
        assert_eq!(tokens_used(), 0);
        TOKENS_USED.fetch_add(usage_tokens(&json!({"usage": {"total_tokens": 7}})), Ordering::Relaxed);
        TOKENS_USED.fetch_add(usage_tokens(&json!({"usage": {"total_tokens": 3}})), Ordering::Relaxed);
        assert_eq!(tokens_used(), 10);
        reset_tokens();
        assert_eq!(tokens_used(), 0);
    }

    /// Default `on_repeat` for tests that don't exercise NBA: a static nudge.
    fn nudge(name: &str, _args: &Value) -> String {
        format!("(dup {name}) you already called this — try a different tool or change the query")
    }

    #[test]
    fn agent_uses_injected_system_prompt() {
        let b = OpenAiBackend::for_test_with_prompt("SYS-MARKER-123");
        let msgs = b.build_messages(&Question {
            question: "hi".into(),
            ..Default::default()
        });
        assert_eq!(msgs[0]["role"], "system");
        assert!(msgs[0]["content"].as_str().unwrap().contains("SYS-MARKER-123"));
        assert_eq!(msgs[1]["role"], "user");
        assert!(msgs[1]["content"].as_str().unwrap().contains("hi"));
    }

    /// The configured `endpoint` is POSTed VERBATIM — nothing is appended (no `/v1/chat/completions`)
    /// and nothing is stripped. A distinctive path that any rewriting would corrupt proves it, and
    /// guards the old footgun where a `.../v1` endpoint got a second `/v1` on the chat_once path.
    #[test]
    fn chat_once_posts_endpoint_url_verbatim() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = [0u8; 2048];
            let n = sock.read(&mut buf).unwrap();
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let body = r#"{"choices":[{"message":{"role":"assistant","content":"ok"}}]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            sock.write_all(resp.as_bytes()).unwrap();
            req
        });

        // A path any append (/v1/chat/completions) or strip (/v1) would mangle.
        let endpoint = format!("http://127.0.0.1:{port}/custom/v1/chat/completions");
        let out = chat_once(&endpoint, "m", &[json!({"role": "user", "content": "hi"})], None, 5).unwrap();
        assert_eq!(out["content"], "ok");

        let req = server.join().unwrap();
        let request_line = req.lines().next().unwrap_or("");
        assert_eq!(
            request_line, "POST /custom/v1/chat/completions HTTP/1.1",
            "endpoint must be POSTed verbatim; got: {request_line}"
        );
    }

    #[test]
    fn parse_tool_args_handles_string_and_object() {
        let s = json!({ "function": { "arguments": "{\"query\":\"abc\"}" } });
        assert_eq!(parse_tool_args(&s)["query"], "abc");
        let o = json!({ "function": { "arguments": { "query": "abc" } } });
        assert_eq!(parse_tool_args(&o)["query"], "abc");
        let bad = json!({ "function": { "arguments": "not json" } });
        assert_eq!(parse_tool_args(&bad), json!({}));
    }

    #[test]
    fn loop_returns_direct_answer_when_no_tool_calls() {
        let chat = |_: &[Value]| Ok(json!({ "role": "assistant", "content": "ANSWER: Bob" }));
        let exec = |_: &str, _: &Value| (String::new(), Vec::new(), Vec::new());
        let out = run_agent_loop(chat, vec![], exec, nudge, 4).unwrap();
        assert_eq!(out, "ANSWER: Bob");
    }

    #[test]
    fn loop_dispatches_tool_then_answers() {
        let round = RefCell::new(0usize);
        let seen = RefCell::new(Vec::<(String, String)>::new());
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            if *r == 1 {
                // first turn requests a search
                Ok(json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "function": { "name": "search", "arguments": "{\"query\":\"corliss\"}" }
                    }]
                }))
            } else {
                // by now the tool result must be in the transcript
                let has_tool = msgs
                    .iter()
                    .any(|m| m["role"] == "tool" && m["tool_call_id"] == "call_1");
                assert!(has_tool, "tool result not fed back: {msgs:?}");
                Ok(json!({ "role": "assistant", "content": "ANSWER: Chief of Protocol" }))
            }
        };
        let exec = |name: &str, args: &Value| {
            seen.borrow_mut().push((
                name.to_string(),
                args["query"].as_str().unwrap_or("").to_string(),
            ));
            (
                "Meet_Corliss_Archer.md:p.1: ...  [9.0]".to_string(),
                vec!["Meet_Corliss_Archer.md:p.1".to_string()],
                Vec::new(),
            )
        };
        let out =
            run_agent_loop(chat, vec![json!({"role":"user","content":"q"})], exec, nudge, 4).unwrap();
        assert_eq!(out, "ANSWER: Chief of Protocol");
        assert_eq!(
            seen.borrow().as_slice(),
            &[("search".to_string(), "corliss".to_string())]
        );
    }

    #[test]
    fn loop_dedupes_consecutive_identical_tool_calls() {
        // The model thrashes: same tool, same args, every round. The loop must execute it once,
        // then feed back a nudge (not another live result) so the model is pushed to switch.
        let execs = RefCell::new(0usize);
        let round = RefCell::new(0usize);
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            // Round 1 executes; round 2's identical call is deduped and its nudge lands in the
            // transcript for round 3's chat onward (round 2 still sees round 1's live result).
            if *r >= 3 {
                let last_tool = msgs.iter().rev().find(|m| m["role"] == "tool");
                let c = last_tool.and_then(|m| m["content"].as_str()).unwrap_or("");
                let lc = c.to_lowercase();
                assert!(
                    lc.contains("different tool") || lc.contains("already"),
                    "round {} expected a dedup nudge, got: {c:?}",
                    *r
                );
            }
            Ok(json!({
                "role": "assistant", "content": "looping",
                "tool_calls": [{ "id": "c", "function": { "name": "glossary", "arguments": "{\"name\":\"X\"}" } }]
            }))
        };
        let exec = |_: &str, _: &Value| {
            *execs.borrow_mut() += 1;
            ("hit".to_string(), vec!["hit-id".to_string()], Vec::new())
        };
        let out = run_agent_loop(chat, vec![], exec, nudge, 5).unwrap();
        assert_eq!(out, "looping");
        assert_eq!(*execs.borrow(), 1, "identical consecutive calls must execute only once");
    }

    #[test]
    fn loop_reexecutes_when_args_differ() {
        // A different tool OR different args is NOT a dedup — it must run.
        let execs = RefCell::new(0usize);
        let round = RefCell::new(0usize);
        let chat = |_: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            let name = if *r % 2 == 1 { "A" } else { "B" };
            Ok(json!({
                "role": "assistant", "content": "alternating",
                "tool_calls": [{ "id": "c", "function": { "name": name, "arguments": "{\"name\":\"X\"}" } }]
            }))
        };
        // Novelty tracking isn't under test here; empty ids keep it a no-op for the streak detector.
        let exec = |_: &str, _: &Value| {
            *execs.borrow_mut() += 1;
            ("hit".to_string(), Vec::new(), Vec::new())
        };
        let _ = run_agent_loop(chat, vec![], exec, nudge, 4).unwrap();
        assert_eq!(*execs.borrow(), 4, "alternating tools must each execute");
    }

    #[test]
    fn loop_stops_at_max_rounds() {
        // chat always asks for a tool; loop must terminate (max_rounds + 1 final call) not hang.
        let calls = RefCell::new(0usize);
        let chat = |_: &[Value]| {
            *calls.borrow_mut() += 1;
            Ok(json!({
                "role": "assistant", "content": "giving up",
                "tool_calls": [{ "id": "c", "function": { "name": "search", "arguments": "{\"query\":\"x\"}" } }]
            }))
        };
        let exec = |_: &str, _: &Value| ("hit".to_string(), Vec::new(), Vec::new());
        let out = run_agent_loop(chat, vec![], exec, nudge, 3).unwrap();
        assert_eq!(out, "giving up");
        assert_eq!(*calls.borrow(), 4); // 3 rounds + 1 final
    }

    #[test]
    fn loop_unproductive_streak_feeds_steer_after_k_plus_one_calls() {
        // K+1 tool calls with VARIED args (so none of them dedup — each really executes), but exec
        // keeps surfacing the SAME already-seen id: an over-search spiral (many different probes,
        // nothing new). By the (K+1)th call, the fed-back tool content must be the steer.
        let round = RefCell::new(0usize);
        let execs = RefCell::new(0usize);
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            if *r == UNPRODUCTIVE_STREAK_K + 2 {
                // By now K+1 tool calls have executed; the most recent tool result must be the steer.
                let last_tool = msgs.iter().rev().find(|m| m["role"] == "tool");
                let c = last_tool.and_then(|m| m["content"].as_str()).unwrap_or("");
                let lc = c.to_lowercase();
                assert!(
                    lc.contains("no new information") && lc.contains("change approach"),
                    "round {} expected the unproductive-streak steer, got: {c:?}",
                    *r
                );
                return Ok(json!({ "role": "assistant", "content": "ANSWER: done" }));
            }
            let query = format!("query-{}", *r); // distinct args every round -> never dedups
            Ok(json!({
                "role": "assistant", "content": "searching",
                "tool_calls": [{
                    "id": format!("c{}", *r),
                    "function": { "name": "search", "arguments": json!({"query": query}).to_string() }
                }]
            }))
        };
        let exec = |_: &str, _: &Value| {
            *execs.borrow_mut() += 1;
            // Always the same id, regardless of the (varied) query -> never novel after the first.
            ("same old snippet".to_string(), vec!["doc.md:p.1".to_string()], Vec::new())
        };
        let out = run_agent_loop(chat, vec![], exec, nudge, UNPRODUCTIVE_STREAK_K + 3).unwrap();
        assert_eq!(out, "ANSWER: done");
        assert_eq!(
            *execs.borrow(),
            UNPRODUCTIVE_STREAK_K + 1,
            "each varied call must actually execute (not deduped)"
        );
    }

    #[test]
    fn loop_unproductive_streak_never_fires_when_calls_are_productive() {
        // Every call surfaces a brand-new id, so the streak resets each time — the steer must never
        // fire even past K calls.
        let round = RefCell::new(0usize);
        let rounds_to_run = UNPRODUCTIVE_STREAK_K + 4;
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            if let Some(last_tool) = msgs.iter().rev().find(|m| m["role"] == "tool") {
                let c = last_tool["content"].as_str().unwrap_or("");
                assert!(
                    !c.to_lowercase().contains("no new information"),
                    "steer must not fire on productive calls, got: {c:?}"
                );
            }
            if *r > rounds_to_run {
                return Ok(json!({ "role": "assistant", "content": "ANSWER: done" }));
            }
            Ok(json!({
                "role": "assistant", "content": "searching",
                "tool_calls": [{
                    "id": format!("c{}", *r),
                    "function": { "name": "search", "arguments": json!({"query": format!("q{}", *r)}).to_string() }
                }]
            }))
        };
        let counter = RefCell::new(0usize);
        let exec = |_: &str, _: &Value| {
            let mut c = counter.borrow_mut();
            *c += 1;
            (format!("hit {}", *c), vec![format!("doc-{}.md", *c)], Vec::new()) // new id every call
        };
        let out = run_agent_loop(chat, vec![], exec, nudge, rounds_to_run + 2).unwrap();
        assert_eq!(out, "ANSWER: done");
    }

    // --- graph-tool id extraction: regression guard for the misfire the coordinator flagged -----
    //
    // The two mock-exec tests above prove the streak MECHANISM. These two prove the id SOURCE for
    // the graph tools is correct: they drive `run_agent_loop` through the REAL `execute_tool` ->
    // `glossa_tools::exec` -> `extract_node_ids` path (no mock exec), against a real indexed corpus
    // + graph, so a genuine glossary/reach/neighbors-style reader can't be falsely steered mid
    // navigation just because graph tools used to surface no ids at all.

    use glossa::graph::store::{Edge, GraphStore, Node, Provenance};
    use glossa::index::store::DocIndex;

    fn fixture_prov() -> Provenance {
        Provenance {
            source_path: "doc.md".into(),
            range: None,
            file_sig: None,
            origin: "test".into(),
            confidence: 0.9,
            created_at: 1,
        }
    }

    /// Build a small real corpus + graph: an indexed markdown doc (so its Section nodes carry
    /// working `read` anchors via the real `index_dir` path — the same machinery a real corpus
    /// uses) plus a `hub` Entity node connected to `n` `fact_i` Entity nodes, each `MENTIONS`-
    /// grounded to one of the doc's real sections. A `neighbors` call on `hub` therefore renders a
    /// real `— read doc.md  #ord · …` anchor per fact, exactly like a genuine graph-reader's
    /// glossary/reach/neighbors traversal would.
    fn build_hub_fixture(n: usize) -> (tempfile::TempDir, DocIndex, GraphStore) {
        let dir = tempfile::tempdir().unwrap();
        let mut md = String::from("# Root\nintro\n");
        for i in 0..n {
            md.push_str(&format!("\n## Sec{i}\nbody {i}\n"));
        }
        std::fs::write(dir.path().join("doc.md"), md).unwrap();
        glossa::index::store::index_dir(dir.path(), true).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = GraphStore::open(dir.path()).unwrap();

        let sec_ids: Vec<String> = g
            .outgoing("doc.md")
            .unwrap()
            .into_iter()
            .filter(|e| e.edge_type == "CONTAINS")
            .map(|e| e.to)
            .collect();
        assert!(
            sec_ids.len() >= n,
            "expected >= {n} indexed sections, got {}: {sec_ids:?}",
            sec_ids.len()
        );

        g.put_node(&Node {
            id: "hub".into(),
            node_type: "Entity".into(),
            label: "hub".into(),
            aliases: Vec::new(),
            prov: fixture_prov(),
        })
        .unwrap();
        for (i, sec_id) in sec_ids.iter().take(n).enumerate() {
            let fact_id = format!("fact-{i}");
            g.put_node(&Node {
                id: fact_id.clone(),
                node_type: "Entity".into(),
                label: format!("Fact {i}"),
                aliases: Vec::new(),
                prov: fixture_prov(),
            })
            .unwrap();
            // Ground the fact in a real section -> gives it a working `read` anchor.
            g.put_edge(&Edge {
                from: fact_id.clone(),
                to: sec_id.clone(),
                edge_type: glossa::graph::MENTIONS.to_string(),
                prov: fixture_prov(),
            })
            .unwrap();
            // A distinct edge type per fact, so `neighbors(hub, edge_types=[REL_i])` surfaces
            // exactly ONE fact per call — mirrors a reader stepping to one node at a time.
            g.put_edge(&Edge {
                from: "hub".into(),
                to: fact_id,
                edge_type: format!("REL_{i}"),
                prov: fixture_prov(),
            })
            .unwrap();
        }
        (dir, idx, g)
    }

    #[test]
    fn loop_real_graph_navigation_to_distinct_nodes_never_falsely_steers() {
        // K+1 REAL `neighbors` calls on the actual glossa_tools dispatch, each stepping to a
        // DIFFERENT fact (distinct edge_types filter -> distinct MENTIONS-grounded target) — this is
        // what a graph reader walking glossary -> neighbors -> neighbors -> ... looks like. Before
        // the fix, graph tools surfaced zero ids, so this would misfire the steer at call K+1 even
        // though every call reached genuinely new ground.
        let n = UNPRODUCTIVE_STREAK_K + 1;
        let (dir, idx, g) = build_hub_fixture(n);
        let spec = glossa::tools::ChainSpec::default();
        let trace = TraceLog::disabled();
        let exec = |name: &str, args: &Value| {
            let (body, ids) = execute_tool(name, args, dir.path(), &idx, Some(&g), &spec, &trace);
            (body, ids, Vec::new())
        };

        let round = RefCell::new(0usize);
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            if let Some(last_tool) = msgs.iter().rev().find(|m| m["role"] == "tool") {
                let c = last_tool["content"].as_str().unwrap_or("");
                assert!(
                    !c.to_lowercase().contains("no new information"),
                    "round {}: steer must not fire while reaching NEW graph nodes, got: {c:?}",
                    *r
                );
            }
            if *r > n {
                return Ok(json!({ "role": "assistant", "content": "ANSWER: done" }));
            }
            let i = *r - 1;
            Ok(json!({
                "role": "assistant", "content": "walking the graph",
                "tool_calls": [{
                    "id": format!("c{}", *r),
                    "function": {
                        "name": "neighbors",
                        "arguments": json!({"node": "hub", "edge_types": [format!("REL_{i}")]}).to_string()
                    }
                }]
            }))
        };
        let out = run_agent_loop(chat, vec![], exec, nudge, n + 2).unwrap();
        assert_eq!(out, "ANSWER: done");
    }

    #[test]
    fn loop_real_graph_navigation_stuck_on_one_node_does_trigger_streak() {
        // K+1 REAL `neighbors` calls with VARIED args (different direction/edge_types combinations,
        // so none dedup) that all resolve to the SAME single grounded fact — a graph reader stuck
        // re-probing one node from different angles. By the (K+1)th call the fed-back tool content
        // must be the steer, proving re-surfaced (not just varied) graph-tool calls DO count as
        // unproductive.
        let (dir, idx, g) = build_hub_fixture(1);
        let spec = glossa::tools::ChainSpec::default();
        let trace = TraceLog::disabled();
        let exec = |name: &str, args: &Value| {
            let (body, ids) = execute_tool(name, args, dir.path(), &idx, Some(&g), &spec, &trace);
            (body, ids, Vec::new())
        };

        // Distinct argument objects that all resolve to the SAME single hub->fact-0 edge.
        let variants = [
            json!({"node": "hub"}),
            json!({"node": "hub", "direction": "out"}),
            json!({"node": "hub", "edge_types": ["REL_0"]}),
            json!({"node": "hub", "edge_types": ["REL_0"], "direction": "out"}),
        ];
        assert!(
            variants.len() >= UNPRODUCTIVE_STREAK_K + 1,
            "need at least K+1 distinct variants"
        );

        let round = RefCell::new(0usize);
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            if *r == UNPRODUCTIVE_STREAK_K + 2 {
                let last_tool = msgs.iter().rev().find(|m| m["role"] == "tool");
                let c = last_tool.and_then(|m| m["content"].as_str()).unwrap_or("");
                let lc = c.to_lowercase();
                assert!(
                    lc.contains("no new information") && lc.contains("change approach"),
                    "round {}: expected the unproductive-streak steer on a real re-surfaced graph \
                     node, got: {c:?}",
                    *r
                );
                return Ok(json!({ "role": "assistant", "content": "ANSWER: done" }));
            }
            let args = variants[(*r - 1) % variants.len()].clone();
            Ok(json!({
                "role": "assistant", "content": "re-probing",
                "tool_calls": [{
                    "id": format!("c{}", *r),
                    "function": { "name": "neighbors", "arguments": args.to_string() }
                }]
            }))
        };
        let out = run_agent_loop(chat, vec![], exec, nudge, UNPRODUCTIVE_STREAK_K + 3).unwrap();
        assert_eq!(out, "ANSWER: done");
    }

    // --- structural (Section/Document) node ids: regression guard for fix round 2 ----------------
    //
    // `extract_node_ids`'s first version only matched the entity-node "— read <path> #<ord>" form.
    // Section/Document endpoints render the SAME `<path>  #<ord>` anchor but BARE — no "read" word
    // (`tools::endpoint_ref`/`node_ref`) — so a reader stepping between Sections (e.g. `neighbors`
    // on a Document, or glossary/reach landing on distinct Sections) surfaced zero ids per call and
    // got falsely steered off a genuinely productive path. `build_section_hub_fixture` connects
    // `hub` DIRECTLY to Section nodes (skipping the MENTIONS-grounded entity layer the fixture above
    // uses), so `neighbors` renders exactly the bare structural form under test.

    /// Like `build_hub_fixture`, but `hub`'s edges point straight at the doc's real Section nodes —
    /// no intermediate MENTIONS-grounded entity. A `neighbors(hub, edge_types=[REL_i])` call then
    /// renders the endpoint via the BARE structural anchor (`tools::node_ref`'s Section arm), with
    /// no "— read" prefix — the exact form the first `extract_node_ids` missed.
    fn build_section_hub_fixture(n: usize) -> (tempfile::TempDir, DocIndex, GraphStore) {
        let dir = tempfile::tempdir().unwrap();
        let mut md = String::from("# Root\nintro\n");
        for i in 0..n {
            md.push_str(&format!("\n## Sec{i}\nbody {i}\n"));
        }
        std::fs::write(dir.path().join("doc.md"), md).unwrap();
        glossa::index::store::index_dir(dir.path(), true).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = GraphStore::open(dir.path()).unwrap();

        let sec_ids: Vec<String> = g
            .outgoing("doc.md")
            .unwrap()
            .into_iter()
            .filter(|e| e.edge_type == "CONTAINS")
            .map(|e| e.to)
            .collect();
        assert!(
            sec_ids.len() >= n,
            "expected >= {n} indexed sections, got {}: {sec_ids:?}",
            sec_ids.len()
        );

        g.put_node(&Node {
            id: "hub".into(),
            node_type: "Entity".into(),
            label: "hub".into(),
            aliases: Vec::new(),
            prov: fixture_prov(),
        })
        .unwrap();
        for (i, sec_id) in sec_ids.iter().take(n).enumerate() {
            // Distinct edge type per section, direct hub -> Section (no entity/MENTIONS layer) —
            // so the rendered endpoint is a BARE structural anchor, not a "— read" one.
            g.put_edge(&Edge {
                from: "hub".into(),
                to: sec_id.clone(),
                edge_type: format!("REL_{i}"),
                prov: fixture_prov(),
            })
            .unwrap();
        }
        (dir, idx, g)
    }

    #[test]
    fn loop_real_structural_navigation_to_distinct_sections_never_falsely_steers() {
        // K+1 REAL `neighbors` calls, each stepping to a DIFFERENT Section endpoint rendered with
        // the BARE structural anchor (no "— read" prefix). Before fix round 2, these surfaced zero
        // ids, so this exact case — a reader walking distinct sections of a document — would
        // misfire the steer at call K+1 despite reaching genuinely new ground every time.
        let n = UNPRODUCTIVE_STREAK_K + 1;
        let (dir, idx, g) = build_section_hub_fixture(n);
        let spec = glossa::tools::ChainSpec::default();
        let trace = TraceLog::disabled();
        let exec = |name: &str, args: &Value| {
            let (body, ids) = execute_tool(name, args, dir.path(), &idx, Some(&g), &spec, &trace);
            (body, ids, Vec::new())
        };

        let round = RefCell::new(0usize);
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            if let Some(last_tool) = msgs.iter().rev().find(|m| m["role"] == "tool") {
                let c = last_tool["content"].as_str().unwrap_or("");
                assert!(
                    !c.to_lowercase().contains("no new information"),
                    "round {}: steer must not fire while reaching NEW structural (Section) nodes, \
                     got: {c:?}",
                    *r
                );
            }
            if *r > n {
                return Ok(json!({ "role": "assistant", "content": "ANSWER: done" }));
            }
            let i = *r - 1;
            Ok(json!({
                "role": "assistant", "content": "walking the document's sections",
                "tool_calls": [{
                    "id": format!("c{}", *r),
                    "function": {
                        "name": "neighbors",
                        "arguments": json!({"node": "hub", "edge_types": [format!("REL_{i}")]}).to_string()
                    }
                }]
            }))
        };
        let out = run_agent_loop(chat, vec![], exec, nudge, n + 2).unwrap();
        assert_eq!(out, "ANSWER: done");
    }

    #[test]
    fn loop_real_structural_navigation_stuck_on_one_section_does_trigger_streak() {
        // Symmetric check: varied `neighbors` calls that all resolve to the SAME single Section
        // endpoint must still trip the streak by the (K+1)th call — proves the relaxed regex isn't
        // so permissive it stops detecting genuine repetition on structural nodes too.
        let (dir, idx, g) = build_section_hub_fixture(1);
        let spec = glossa::tools::ChainSpec::default();
        let trace = TraceLog::disabled();
        let exec = |name: &str, args: &Value| {
            let (body, ids) = execute_tool(name, args, dir.path(), &idx, Some(&g), &spec, &trace);
            (body, ids, Vec::new())
        };

        let variants = [
            json!({"node": "hub"}),
            json!({"node": "hub", "direction": "out"}),
            json!({"node": "hub", "edge_types": ["REL_0"]}),
            json!({"node": "hub", "edge_types": ["REL_0"], "direction": "out"}),
        ];
        assert!(
            variants.len() >= UNPRODUCTIVE_STREAK_K + 1,
            "need at least K+1 distinct variants"
        );

        let round = RefCell::new(0usize);
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            if *r == UNPRODUCTIVE_STREAK_K + 2 {
                let last_tool = msgs.iter().rev().find(|m| m["role"] == "tool");
                let c = last_tool.and_then(|m| m["content"].as_str()).unwrap_or("");
                let lc = c.to_lowercase();
                assert!(
                    lc.contains("no new information") && lc.contains("change approach"),
                    "round {}: expected the unproductive-streak steer on a re-surfaced structural \
                     node, got: {c:?}",
                    *r
                );
                return Ok(json!({ "role": "assistant", "content": "ANSWER: done" }));
            }
            let args = variants[(*r - 1) % variants.len()].clone();
            Ok(json!({
                "role": "assistant", "content": "re-probing",
                "tool_calls": [{
                    "id": format!("c{}", *r),
                    "function": { "name": "neighbors", "arguments": args.to_string() }
                }]
            }))
        };
        let out = run_agent_loop(chat, vec![], exec, nudge, UNPRODUCTIVE_STREAK_K + 3).unwrap();
        assert_eq!(out, "ANSWER: done");
    }

    // --- vision: `--vision`-only image threading (Task: kbx build --vision) ------------------
    //
    // `vision_user_message` is the pure builder (images -> the OpenAI-compatible content-array
    // user message); these tests exercise it directly (on/off/cap) plus `run_agent_loop` end to
    // end, proving the loop actually appends the message right after the tool result when `exec`
    // surfaces images, and appends NOTHING when it doesn't — which is what keeps every non-vision
    // caller (the reader, `kbx reason`, `kbx distil`, the GEPA rollout) byte-identical to today.

    fn stub_image(tag: u8) -> DocImage {
        // mime "image/jpeg" short-circuits `to_jpeg` (returns the bytes unchanged, no real JPEG
        // decode needed) — exactly what a stubbed unit test wants: no fixture image file, no
        // `image` crate round-trip, just distinct bytes per stub so multiple images are
        // distinguishable in the encoded output.
        DocImage {
            mime: "image/jpeg".to_string(),
            bytes: vec![0xFF, 0xD8, 0xFF, tag], // fake JPEG-ish bytes, tag makes each stub unique
        }
    }

    #[test]
    fn vision_message_none_when_no_images() {
        assert!(vision_user_message(&[]).is_none());
    }

    #[test]
    fn vision_message_builds_canonical_data_uri_content_array() {
        let img = stub_image(1);
        let msg = vision_user_message(&[img.clone()]).expect("one image -> Some(message)");
        assert_eq!(msg["role"], "user");
        let content = msg["content"].as_array().expect("content must be an array");
        assert_eq!(content.len(), 2, "one text part + one image_url part: {content:?}");
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
        let url = content[1]["image_url"]["url"].as_str().expect("image_url.url string");
        assert!(
            url.starts_with("data:image/jpeg;base64,"),
            "must be a JPEG data URI, got: {url}"
        );
        let payload = url.strip_prefix("data:image/jpeg;base64,").unwrap();
        assert!(
            !payload.contains('\n') && !payload.contains(' '),
            "base64 payload must be canonical (no embedded whitespace/newlines): {payload:?}"
        );
        use base64::Engine as _;
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(payload)
            .expect("payload must be valid standard base64");
        assert_eq!(decoded, img.bytes, "decoded payload must round-trip the original JPEG bytes");
    }

    #[test]
    fn vision_message_caps_at_max_images_per_turn() {
        let images: Vec<DocImage> = (0..(MAX_IMAGES_PER_TURN as u8 + 2)).map(stub_image).collect();
        assert!(images.len() > MAX_IMAGES_PER_TURN, "test must exceed the cap");
        let msg = vision_user_message(&images).expect("non-empty -> Some(message)");
        let content = msg["content"].as_array().unwrap();
        let image_parts = content.iter().filter(|p| p["type"] == "image_url").count();
        assert_eq!(
            image_parts, MAX_IMAGES_PER_TURN,
            "extras must be dropped, not fed: {content:?}"
        );
    }

    #[test]
    fn loop_vision_on_appends_image_user_message_after_tool_result() {
        // exec surfaces one image on its (only) call; the loop must push a role:"user"
        // content-array message with a data:image/jpeg;base64,... image_url part right after the
        // role:"tool" message, before the model is asked again.
        let round = RefCell::new(0usize);
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            if *r == 1 {
                return Ok(json!({
                    "role": "assistant", "content": null,
                    "tool_calls": [{
                        "id": "c1",
                        "function": { "name": "read", "arguments": "{\"path\":\"scan.pdf\"}" }
                    }]
                }));
            }
            // Round 2: the transcript from round 1's tool call must already carry the image
            // message, positioned right after the tool result.
            let tool_pos = msgs.iter().position(|m| m["role"] == "tool");
            let img_pos = msgs.iter().position(|m| {
                m["role"] == "user"
                    && m["content"]
                        .as_array()
                        .is_some_and(|c| c.iter().any(|p| p["type"] == "image_url"))
            });
            assert!(tool_pos.is_some(), "tool result missing: {msgs:?}");
            assert_eq!(
                img_pos,
                tool_pos.map(|p| p + 1),
                "image message must come right after the tool result: {msgs:?}"
            );
            let url = msgs[img_pos.unwrap()]["content"][1]["image_url"]["url"]
                .as_str()
                .unwrap_or("");
            assert!(url.starts_with("data:image/jpeg;base64,"), "got: {url}");
            Ok(json!({ "role": "assistant", "content": "ANSWER: done" }))
        };
        let exec = |_: &str, _: &Value| {
            ("(scanned page text)".to_string(), vec!["scan.pdf".to_string()], vec![stub_image(9)])
        };
        let out = run_agent_loop(chat, vec![], exec, nudge, 3).unwrap();
        assert_eq!(out, "ANSWER: done");
    }

    #[test]
    fn loop_vision_off_appends_no_image_message() {
        // Mirrors every non-vision caller (reader, reason, distil, GEPA): exec always surfaces an
        // empty image vec, so the loop must push ONLY the tool message — byte-identical to the
        // pre-vision transcript shape.
        let round = RefCell::new(0usize);
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            if *r == 1 {
                return Ok(json!({
                    "role": "assistant", "content": null,
                    "tool_calls": [{
                        "id": "c1",
                        "function": { "name": "read", "arguments": "{\"path\":\"scan.pdf\"}" }
                    }]
                }));
            }
            let has_image_msg = msgs.iter().any(|m| {
                m["role"] == "user"
                    && m["content"]
                        .as_array()
                        .is_some_and(|c| c.iter().any(|p| p["type"] == "image_url"))
            });
            assert!(!has_image_msg, "vision-off must never append an image message: {msgs:?}");
            Ok(json!({ "role": "assistant", "content": "ANSWER: done" }))
        };
        let exec = |_: &str, _: &Value| {
            ("(scanned page text)".to_string(), vec!["scan.pdf".to_string()], Vec::new())
        };
        let out = run_agent_loop(chat, vec![], exec, nudge, 3).unwrap();
        assert_eq!(out, "ANSWER: done");
    }
}

#[cfg(test)]
mod schema_tests {
    use super::*;

    fn tool_names(v: &Value) -> Vec<String> {
        v.as_array()
            .unwrap()
            .iter()
            .filter_map(|t| {
                t.pointer("/function/name")
                    .and_then(|n| n.as_str())
                    .map(String::from)
            })
            .collect()
    }

    #[test]
    fn grep_is_advertised_in_both_arms() {
        // grep is ungated in the registry, so it must appear in both graph-OFF and graph-ON.
        assert!(tool_names(&tools_schema(false)).contains(&"grep".into()), "graph-OFF must advertise grep");
        assert!(tool_names(&tools_schema(true)).contains(&"grep".into()), "graph-ON must advertise grep");
    }

    #[test]
    fn openai_tools_match_registry_graph_on() {
        // graph-ON advertises the FULL registry, in registry order — no hand-curated subset or
        // reordering; MCP and the eval agent render from the same source of truth.
        let names = tool_names(&tools_schema(true));
        let reg: Vec<_> = glossa::tools::registry::registry()
            .iter()
            .map(|d| d.name.to_string())
            .collect();
        assert_eq!(names, reg, "graph-ON tool set must equal registry order");
    }

    #[test]
    fn openai_tools_hide_graph_gated_when_off() {
        let names = tool_names(&tools_schema(false));
        // related/neighbors aren't in the registry at all (withheld from the Reader profile as
        // measured clutter) — only glossary/reach/sql are graph-gated now.
        for gated in ["glossary", "reach", "sql"] {
            assert!(
                !names.contains(&gated.to_string()),
                "graph-OFF must NOT advertise graph-gated tool {gated}; got {names:?}"
            );
        }
        for ungated in ["search", "read", "grep", "glob"] {
            assert!(
                names.contains(&ungated.to_string()),
                "graph-OFF must advertise ungated tool {ungated}; got {names:?}"
            );
        }
    }

    #[test]
    fn looks_looped_detects_repeated_tail() {
        let text = "Some normal preamble. ".to_string() + &"the same phrase over and over. ".repeat(10);
        assert!(looks_looped(&text));
    }

    #[test]
    fn looks_looped_detects_short_cycle() {
        let text = "prefix ".to_string() + &"ABABAB".repeat(20);
        assert!(looks_looped(&text));
    }

    #[test]
    fn looks_looped_passes_normal_prose() {
        let text = "The quick brown fox jumps over the lazy dog. It was a bright cold day in April, \
                     and the clocks were striking thirteen. Meanwhile, a different sentence follows, \
                     varying its wording and structure so no short block repeats consecutively.";
        assert!(!looks_looped(text));
    }

    #[test]
    fn looks_looped_short_text_is_false() {
        assert!(!looks_looped("too short"));
    }

    #[test]
    fn should_resample_on_length_or_loop_but_not_normal() {
        // Hit the token cap: resample regardless of content.
        assert!(should_resample("length", "any content, even short"));
        // Looked like a degenerate repetition loop, even though it finished "normally".
        let looped = "prefix ".to_string() + &"ABABAB".repeat(20);
        assert!(should_resample("stop", &looped));
        // Normal, complete, non-looping content: accept it.
        assert!(!should_resample("stop", "ANSWER: a short normal answer"));
    }

    /// Serializes every end-to-end `lmstudio_chat` resample test in this module: they all share the
    /// global `RESAMPLES` counter and some of them also mutate the `KB_EVAL_MAX_LENGTH_RESAMPLE` env
    /// var, so running them concurrently (the default for `cargo test`) would let one test's counter
    /// deltas or env var leak into another's assertions.
    static RESAMPLE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// RAII guard that restores `KB_EVAL_MAX_LENGTH_RESAMPLE` to whatever it was before `set()` (or
    /// removes it if it was unset), on drop — including an early return via a failed `assert!` —
    /// so a test that overrides the cap can never leak the override into a later test in the same
    /// process.
    struct MaxLengthResampleEnvGuard {
        prev: Option<String>,
    }
    impl MaxLengthResampleEnvGuard {
        fn set(val: &str) -> Self {
            let prev = std::env::var("KB_EVAL_MAX_LENGTH_RESAMPLE").ok();
            std::env::set_var("KB_EVAL_MAX_LENGTH_RESAMPLE", val);
            Self { prev }
        }
    }
    impl Drop for MaxLengthResampleEnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var("KB_EVAL_MAX_LENGTH_RESAMPLE", v),
                None => std::env::remove_var("KB_EVAL_MAX_LENGTH_RESAMPLE"),
            }
        }
    }

    /// Spins up a one-shot mock HTTP server that answers `bodies.len()` sequential POSTs, in order,
    /// with each of `bodies` as the raw JSON response — shared by every resample-loop end-to-end test
    /// below so the socket/response plumbing (mirroring `chat_once_posts_endpoint_url_verbatim`'s
    /// pattern) is written once. Returns the endpoint URL, a live request counter, and the server's
    /// join handle (call `.join()` after the `lmstudio_chat` call returns).
    fn mock_chat_server(
        bodies: Vec<&'static str>,
    ) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>, std::thread::JoinHandle<()>) {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let requests = Arc::new(AtomicUsize::new(0));
        let requests_srv = requests.clone();
        let server = std::thread::spawn(move || {
            for body in bodies {
                let (mut sock, _) = listener.accept().unwrap();
                let mut buf = [0u8; 4096];
                let _ = sock.read(&mut buf).unwrap();
                requests_srv.fetch_add(1, Ordering::SeqCst);
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                sock.write_all(resp.as_bytes()).unwrap();
            }
        });
        (format!("http://127.0.0.1:{port}/v1/chat/completions"), requests, server)
    }

    /// Mirrors `chat_once_posts_endpoint_url_verbatim`'s mock-server pattern, but drives
    /// `lmstudio_chat`'s resample loop: the first response hits the token cap
    /// (`finish_reason:"length"`), so it must resample once; the second response is a normal
    /// completion, so the loop stops there and returns it. Proves the length branch actually fires
    /// end-to-end (not just the pure predicate), that the returned message is the GOOD (second) one
    /// (not the truncated first one), and that the process-global `resamples()` counter tallies it.
    #[test]
    fn lmstudio_chat_resamples_once_on_length_then_returns_good_message() {
        use std::sync::atomic::Ordering;
        let _guard = RESAMPLE_TEST_LOCK.lock().unwrap();
        reset_resamples();

        let (endpoint, requests, server) = mock_chat_server(vec![
            // Truncated at the cap: no tool call, no finish content that matters — only
            // finish_reason drives the resample.
            r#"{"choices":[{"message":{"role":"assistant","content":"...truncated verbose reasoning"},"finish_reason":"length"}]}"#,
            r#"{"choices":[{"message":{"role":"assistant","content":"ANSWER: ok"},"finish_reason":"stop"}]}"#,
        ]);
        let msg = lmstudio_chat(
            &endpoint,
            "m",
            None,
            &json!([]),
            &[json!({"role": "user", "content": "hi"})],
            Duration::from_secs(5),
        )
        .unwrap();

        server.join().unwrap();
        assert_eq!(
            requests.load(Ordering::SeqCst),
            2,
            "must resample exactly once after a finish_reason=length turn"
        );
        assert_eq!(msg["content"], "ANSWER: ok", "must return the second (good) message, not the truncated first one");
        assert_eq!(resamples(), 1, "the counter must tally the one resample this call made");
    }

    /// A chronically-verbose model hits the token cap on EVERY turn. With the default
    /// `DEFAULT_MAX_LENGTH_RESAMPLE == 1`, the loop must resample only once on `finish_reason:
    /// "length"`, then give up and accept the second (still-truncated) completion rather than
    /// burning a third generation — this is the fix for the token-thrashing bug (a chronically
    /// verbose model no longer costs up to `GEN_LOOP_RETRIES` full generations per call).
    #[test]
    fn length_resample_capped_at_default_accepts_second_truncated_completion() {
        use std::sync::atomic::Ordering;
        let _guard = RESAMPLE_TEST_LOCK.lock().unwrap();
        reset_resamples();

        let (endpoint, requests, server) = mock_chat_server(vec![
            r#"{"choices":[{"message":{"role":"assistant","content":"...truncated A"},"finish_reason":"length"}]}"#,
            r#"{"choices":[{"message":{"role":"assistant","content":"...truncated B"},"finish_reason":"length"}]}"#,
        ]);
        let msg = lmstudio_chat(
            &endpoint,
            "m",
            None,
            &json!([]),
            &[json!({"role": "user", "content": "hi"})],
            Duration::from_secs(5),
        )
        .unwrap();

        server.join().unwrap();
        assert_eq!(requests.load(Ordering::SeqCst), 2, "must resample exactly once, not twice, before giving up");
        assert_eq!(resamples(), 1, "the counter must reflect exactly one resample");
        assert_eq!(
            msg["content"], "...truncated B",
            "must give up and return the second (still-truncated) completion, not keep resampling"
        );
    }

    /// A normal, complete first response (`finish_reason: "stop"`, no repetition loop) must never
    /// trigger a resample at all.
    #[test]
    fn normal_completion_never_resamples() {
        use std::sync::atomic::Ordering;
        let _guard = RESAMPLE_TEST_LOCK.lock().unwrap();
        reset_resamples();

        let (endpoint, requests, server) = mock_chat_server(vec![
            r#"{"choices":[{"message":{"role":"assistant","content":"ANSWER: ok"},"finish_reason":"stop"}]}"#,
        ]);
        let msg = lmstudio_chat(
            &endpoint,
            "m",
            None,
            &json!([]),
            &[json!({"role": "user", "content": "hi"})],
            Duration::from_secs(5),
        )
        .unwrap();

        server.join().unwrap();
        assert_eq!(requests.load(Ordering::SeqCst), 1, "a normal completion must not be resampled");
        assert_eq!(resamples(), 0);
        assert_eq!(msg["content"], "ANSWER: ok");
    }

    /// The length-resample cap must not shrink the unrelated degenerate-loop resample path: a
    /// completion that finishes normally (`finish_reason: "stop"`) but whose content looks like a
    /// repetition loop must still resample up to the full `GEN_LOOP_RETRIES` (2) times before giving
    /// up, exactly as before this change.
    #[test]
    fn loop_resample_path_still_honors_gen_loop_retries() {
        use std::sync::atomic::Ordering;
        let _guard = RESAMPLE_TEST_LOCK.lock().unwrap();
        reset_resamples();

        let looped = format!(
            r#"{{"choices":[{{"message":{{"role":"assistant","content":"prefix {}"}},"finish_reason":"stop"}}]}}"#,
            "ABABAB".repeat(20)
        );
        let looped: &'static str = Box::leak(looped.into_boxed_str());
        let (endpoint, requests, server) = mock_chat_server(vec![looped, looped, looped]);
        let msg = lmstudio_chat(
            &endpoint,
            "m",
            None,
            &json!([]),
            &[json!({"role": "user", "content": "hi"})],
            Duration::from_secs(5),
        )
        .unwrap();

        server.join().unwrap();
        assert_eq!(
            requests.load(Ordering::SeqCst),
            3,
            "must resample the full GEN_LOOP_RETRIES (2) times for a persistent repetition loop, unbounded by the length cap"
        );
        assert_eq!(resamples(), 2);
        assert!(msg["content"].as_str().unwrap().contains("ABABAB"), "gives up and returns the last best-effort completion");
    }

    /// `KB_EVAL_MAX_LENGTH_RESAMPLE=0` must disable length-resampling entirely: even a single
    /// `finish_reason: "length"` turn is accepted as-is, with zero resamples.
    #[test]
    fn max_length_resample_zero_never_resamples_on_length() {
        use std::sync::atomic::Ordering;
        let _guard = RESAMPLE_TEST_LOCK.lock().unwrap();
        reset_resamples();
        let _env = MaxLengthResampleEnvGuard::set("0");

        let (endpoint, requests, server) = mock_chat_server(vec![
            r#"{"choices":[{"message":{"role":"assistant","content":"...truncated"},"finish_reason":"length"}]}"#,
        ]);
        let msg = lmstudio_chat(
            &endpoint,
            "m",
            None,
            &json!([]),
            &[json!({"role": "user", "content": "hi"})],
            Duration::from_secs(5),
        )
        .unwrap();

        server.join().unwrap();
        assert_eq!(requests.load(Ordering::SeqCst), 1, "KB_EVAL_MAX_LENGTH_RESAMPLE=0 must accept the first completion outright");
        assert_eq!(resamples(), 0);
        assert_eq!(msg["content"], "...truncated");
    }
}
