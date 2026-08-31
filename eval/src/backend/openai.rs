use super::{prompt, AgentBackend};
use crate::backend::transport::{ChatTransport, ToolCall, TurnReply};
use crate::dataset::Question;
use glossa::read::DocImage;
use glossa::trace::TraceLog;
use indicatif::ProgressBar;
use serde_json::{json, Value};
use std::cell::Cell;
use std::path::Path;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

/// Process-global running total of NEWLY-processed tokens (freshly-processed prompt + all
/// completion) consumed across every chat call routed through `chat_http` — every one of
/// reason/build/eval/distil goes through that single function, so this one counter tallies all of
/// them without each call site needing to thread usage back up itself. Split from the CACHED
/// counter below because with prompt caching most of a naive running total is cheap re-sent
/// prompt, not new work — see `usage_split`.
static NEW_TOKENS: AtomicU64 = AtomicU64::new(0);

/// Process-global running total of CACHED prompt tokens (either server-reported
/// `usage.prompt_tokens_details.cached_tokens`, or — when the server omits that field — a
/// SELF-COMPUTED estimate; see `usage_split_with_prefix`) across every chat call — mirrors
/// `NEW_TOKENS`, tallied by the same call in `chat_http`.
static CACHED_TOKENS: AtomicU64 = AtomicU64::new(0);

/// The PREVIOUS chat request's `prompt_tokens` within the current conversation (one agent-loop
/// run: one reason seed, one build doc, one eval case, …). Each round of `run_agent_loop`
/// re-sends the whole prior transcript as a prefix and appends to it, so on a server that omits
/// `prompt_tokens_details.cached_tokens` (e.g. LM Studio), THIS round's re-sent prefix is exactly
/// last round's `prompt_tokens` — see `usage_split_with_prefix`. Reset to 0 at the start of every
/// conversation by `reset_conversation_prefix` so one seed's tail doesn't leak into the next
/// seed's estimate.
///
/// **THREAD-LOCAL, not a process-global.** Under parallel workers (`kbx --jobs N`) each worker
/// thread drives its own conversation concurrently with the others; a shared global here would let
/// worker A's stored `prompt_tokens` leak into worker B's next estimate (interleaved conversations
/// corrupting each other's prefix) — see the parallel-jobs design doc's "Caching under
/// parallelism" section. Each thread gets its own independent cell, so the SEQUENTIAL-within-one-
/// conversation assumption `usage_split_with_prefix` relies on holds per-thread even though many
/// threads run concurrently. The cross-worker aggregates (`NEW_TOKENS`/`CACHED_TOKENS`/
/// `CACHE_ESTIMATED`/`RESAMPLES`) stay process-global atomics — their SUM across workers is still
/// the correct total, unlike this per-conversation prefix.
thread_local! {
    static PREV_PROMPT_TOKENS: Cell<u64> = const { Cell::new(0) };
}

/// Set true whenever the MOST RECENT `chat_http` call had to self-estimate its cached-token split
/// (the server omitted `prompt_tokens_details`) rather than use a server-reported figure. Drives
/// the `~` (estimated) marker in `status_message`/`token_summary`. Reset to `false` in
/// `reset_tokens`. Sticky for the run rather than per-call: once a run has needed even one
/// estimate, the whole run's cache figure is an estimate-tainted mix and should read as such.
static CACHE_ESTIMATED: AtomicBool = AtomicBool::new(false);

/// Current value of the running new-token counter (see `NEW_TOKENS`).
pub fn new_tokens() -> u64 {
    NEW_TOKENS.load(Ordering::Relaxed)
}

/// Current value of the running cached-token counter (see `CACHED_TOKENS`).
pub fn cached_tokens() -> u64 {
    CACHED_TOKENS.load(Ordering::Relaxed)
}

/// Reset the CALLING THREAD's conversation-prefix tracker (see `PREV_PROMPT_TOKENS`) to 0 — call
/// at the START of every conversation (`run_agent_loop`'s entry, before its first `chat` call) so
/// a fresh seed/doc/case never estimates its first request's cache off the PREVIOUS conversation's
/// tail `prompt_tokens`. Also called by `reset_tokens` (a run boundary is a conversation boundary
/// too). Under parallel workers each worker thread has its own thread-local cell, so this only
/// ever clears the calling thread's own prefix — never another worker's in-flight conversation.
pub fn reset_conversation_prefix() {
    PREV_PROMPT_TOKENS.with(|prev| prev.set(0));
}

/// Zero both running token counters (plus the conversation-prefix tracker and the estimated-flag)
/// . Call at the start of a run loop that wants its own per-run total (reason/build-extract/
/// build-judge/eval each call this before their loop starts), so one stage's bar reflects only
/// tokens spent in that stage, not a prior one's leftover total.
pub fn reset_tokens() {
    NEW_TOKENS.store(0, Ordering::Relaxed);
    CACHED_TOKENS.store(0, Ordering::Relaxed);
    CACHE_ESTIMATED.store(false, Ordering::Relaxed);
    reset_conversation_prefix();
}

/// Process-global running count of resamples performed by the provider-neutral resample layer
/// ([`crate::backend::resample::call_with_resample`], driven by the agent loop for every stage) —
/// across the length-cap path, the degenerate-loop path, and the empty-no-tool-turn path. Mirrors
/// `NEW_TOKENS` so a run loop can surface `resamples()` on the same progress-bar message instead of
/// the resample diagnostic colliding with the bar via a raw `eprintln!`.
static RESAMPLES: AtomicU64 = AtomicU64::new(0);

/// Current value of the running resample counter (see `RESAMPLES`).
pub fn resamples() -> u64 {
    RESAMPLES.load(Ordering::Relaxed)
}

/// Tally one resample into the process-global counter. Called by
/// [`crate::backend::resample::call_with_resample`] each time it resamples a degenerate completion
/// — the counter subsystem (like the token counters) lives HERE, and the resample layer calls in.
pub(crate) fn note_resample() {
    RESAMPLES.fetch_add(1, Ordering::Relaxed);
}

/// Zero the running resample counter. Call at the start of a run loop, alongside `reset_tokens()`,
/// so one stage's bar reflects only resamples spent in that stage.
pub fn reset_resamples() {
    RESAMPLES.store(0, Ordering::Relaxed);
}

/// Read `usage.prompt_tokens` out of a parsed chat-completions response, `0` when absent. Used
/// both by `usage_split_with_prefix` and by `chat_http` to update `PREV_PROMPT_TOKENS` after a
/// call, so both read the exact same field the exact same way.
fn prompt_tokens(resp: &Value) -> u64 {
    resp.pointer("/usage/prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

/// Split a parsed chat-completions response `resp`'s token usage into `(new, cached, estimated)`,
/// given `prev_prompt` — the PREVIOUS chat request's `prompt_tokens` in the same conversation (see
/// `PREV_PROMPT_TOKENS`).
///
/// - No `usage` at all -> `(0, 0, false)`.
/// - Server reports `prompt_tokens_details.cached_tokens` -> the REAL split: `cached` = the
///   reported figure, `new` = `(prompt_tokens - cached) + completion_tokens`, `estimated = false`.
/// - Server omits that field (e.g. LM Studio, verified absent) -> a SELF-COMPUTED estimate. Within
///   one agent-loop conversation, each round re-sends the ENTIRE previous prompt as a prefix and
///   only appends to it — messages never shrink or get rewritten. On a cloud API with prompt
///   caching that re-sent prefix would be served from cache, so `prev_prompt` (last round's whole
///   prompt) is exactly this round's estimated cache hit: `cached_est = prev_prompt` when
///   `prompt_tokens >= prev_prompt && prev_prompt > 0`, else `0`. The `>=` guard handles a SMALLER
///   prompt than last round: that can only mean a new/different conversation reusing the same
///   global counter (e.g. a single-shot `chat_once` call that bypasses `run_agent_loop`'s
///   per-conversation reset), not a shrinking prefix — so no cached prefix is assumed.
///   `new = (prompt_tokens - cached_est) + completion_tokens`, `estimated = true`.
///
/// This is a CONSERVATIVE estimate: it only credits the within-conversation re-sent prefix, never
/// cross-conversation system-prompt caching a real provider might also apply. Pure — factored out
/// of `chat_http` so it's unit-testable without a live server.
pub fn usage_split_with_prefix(resp: &Value, prev_prompt: u64) -> (u64, u64, bool) {
    let Some(usage) = resp.get("usage") else {
        return (0, 0, false);
    };
    let prompt = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let completion = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    match usage
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_u64)
    {
        Some(cached) => {
            let new = prompt.saturating_sub(cached) + completion;
            (new, cached, false)
        }
        None => {
            let cached_est = if prompt >= prev_prompt && prev_prompt > 0 {
                prev_prompt
            } else {
                0
            };
            let new = prompt.saturating_sub(cached_est) + completion;
            (new, cached_est, true)
        }
    }
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

/// Render the cache segment's label — `"{cached} cache~"` when this run's cache figure includes a
/// self-computed estimate (`CACHE_ESTIMATED`, see `usage_split_with_prefix`), else the plain
/// `"{cached} cache"` for a server-reported split. The trailing `~` is the sole marker distinguishing
/// an estimated figure from a real one, reused identically by `status_message` (live bar) and
/// `token_summary` (final line).
fn cache_segment() -> String {
    let n = human_tokens(cached_tokens());
    if CACHE_ESTIMATED.load(Ordering::Relaxed) {
        format!("{n} cache~")
    } else {
        format!("{n} cache")
    }
}

/// Compose the live after-time status-bar segment: `" · {N new · M cache[~]}{ · N resampled}"`. The
/// front-of-bar `{prefix}` carries a single STATIC stage word (`reasoning`/`building`/… set once by
/// each run loop), so this message carries only the running token counters and the resample count.
/// New and cache are ALWAYS shown as two labelled segments, so the split is legible even on a
/// server that reports no cached tokens (a local LM Studio run reads `... · N new · M cache~`, the
/// `~` making visible that the cache figure is a self-computed estimate, not server-reported).
/// Pure aside from reading the shared atomics, so `StatusTicker` and any direct caller compose
/// exactly the same text as each other.
fn status_message() -> String {
    let mut parts: Vec<String> = Vec::new();
    parts.push(format!("{} new", human_tokens(new_tokens())));
    parts.push(cache_segment());
    let n = resamples();
    if n > 0 {
        parts.push(format!("{n} resampled"));
    }
    format!(" · {}", parts.join(" · "))
}

/// Final one-line token summary for a finished run loop: `"{new} new · {cached} cache[~]"`, with
/// `~` carrying the same estimated-cache meaning as `status_message`'s live segment (see
/// `cache_segment`). Each run loop (reason/build-extract/build-judge/eval/distil-gen/distil-
/// densify) prints this once via the bar's `pb.println`/a plain `println!` right after its loop
/// ends, so a LOCAL run (no server-reported cached tokens) still lets the user project cloud cost
/// from the new-vs-cache split. When the figure is estimated, the caller should also append the
/// `" (cache estimated from prompt re-send)"` footnote once — this fn returns only the compact
/// counts line so callers can decide whether/how to append that footnote.
pub fn token_summary() -> String {
    format!("{} new · {}", human_tokens(new_tokens()), cache_segment())
}

/// True when this run's cache figure includes at least one self-computed estimate (server omitted
/// `prompt_tokens_details`) rather than being entirely server-reported — lets a caller decide
/// whether to append the `" (cache estimated from prompt re-send)"` footnote after `token_summary`.
pub fn cache_is_estimated() -> bool {
    CACHE_ESTIMATED.load(Ordering::Relaxed)
}

/// Compute a stable ETA (in whole seconds remaining) from a bar's own `elapsed_secs`/`pos`/`len`,
/// or `None` when there's no basis for an estimate yet (`pos == 0`, or `pos > len` — the latter
/// shouldn't happen but is guarded rather than underflowing). Plain linear-rate arithmetic:
/// `elapsed * (len - pos) / pos`, computed FRESH from the bar's current position/length/elapsed
/// every time it's called — unlike indicatif's own `{eta_precise}`, which maintains a smoothed
/// rate estimator fed by every redraw. `enable_steady_tick` (used here to animate the spinner)
/// redraws every ~90ms regardless of whether real progress (`pb.inc`) happened in between, so
/// indicatif's estimator gets flooded with near-zero-delta-position samples between real steps;
/// its smoothed rate collapses toward zero and the resulting ETA blows up to absurd values (e.g.
/// "448d"). Recomputing from raw elapsed/pos/len on each tick has no history to corrupt. Pure —
/// unit-tested.
fn eta_secs(elapsed_secs: u64, pos: u64, len: u64) -> Option<u64> {
    if pos == 0 || pos > len {
        return None;
    }
    let remaining = len - pos;
    Some(elapsed_secs.saturating_mul(remaining) / pos)
}

/// Render an `eta_secs` result the same way indicatif renders `{elapsed_precise}`/`{eta_precise}`
/// (`HH:MM:SS`, or `Nd HH:MM:SS` past a day) by reusing indicatif's own `FormattedDuration` — so
/// the stable ETA looks visually consistent with the elapsed time it sits next to on the bar.
/// `None` (no progress yet) renders as a placeholder instead of a misleading `00:00:00`.
fn format_eta(secs: Option<u64>) -> String {
    match secs {
        Some(s) => indicatif::FormattedDuration(Duration::from_secs(s)).to_string(),
        None => "--:--:--".to_string(),
    }
}

/// Background thread that keeps a progress bar's after-time message (`{msg}`) reflecting live
/// in-loop progress, redrawing every ~90ms. Each tick it sets the message to the ETA plus the
/// running token/resample counters. It does NOT touch the bar's `{prefix}`: that carries a single
/// STATIC stage word (`reasoning`/`building`/…) set once by the owning run loop before the loop
/// starts and never changed mid-run — the spinner + these counters + the ETA supply the "alive"
/// feel without a flickering activity word. A run loop's own per-seed `pb.set_message` only fires
/// once per seed/case, so it can't keep the ETA/counters current WITHIN one seed — this ticker is
/// what actually keeps the bar looking alive while a seed is in flight.
///
/// Also computes and prepends a self-computed, stable ETA (see `eta_secs`) as `<{eta}` — the
/// bar's TEMPLATE now ends in plain `{elapsed_precise}{msg}` (no `{eta_precise}` of its own; see
/// the 4 `ProgressStyle::with_template` call sites), so this ticker is the sole source of the
/// `<eta` half of the "elapsed<eta" pairing the bar displays.
///
/// `ProgressBar` is `Clone + Send + Sync` (cloning shares the same underlying draw target), so the
/// ticker owns its own clone and never needs to borrow the caller's `pb`. On a hidden (non-TTY)
/// bar, `set_message` is a harmless no-op, so starting a ticker unconditionally is safe.
pub struct StatusTicker {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl StatusTicker {
    /// Spawn the ticker thread against a clone of `pb`. Runs until this `StatusTicker` is dropped.
    pub fn start(pb: &ProgressBar) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_bg = Arc::clone(&stop);
        let pb = pb.clone();
        let handle = std::thread::spawn(move || {
            while !stop_bg.load(Ordering::Relaxed) {
                let len = pb.length().unwrap_or(0);
                let eta = eta_secs(pb.elapsed().as_secs(), pb.position(), len);
                // After-time message only: ETA + tokens/resamples. The static `{prefix}` word is
                // owned by the run loop and left untouched here.
                pb.set_message(format!("<{}{}", format_eta(eta), status_message()));
                std::thread::sleep(Duration::from_millis(90));
            }
        });
        StatusTicker {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for StatusTicker {
    /// Stop the ticker thread and join it, so a `StatusTicker` can never outlive the loop that
    /// started it — no leaked thread left writing to a bar the caller has already cleared/dropped.
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
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
    /// Per-endpoint sampling temperature carried from the reader's `[model]` endpoint. Folded into
    /// the `Endpoint` handed to the agent loop, where `Endpoint::resolve_temperature` still lets
    /// `KB_EVAL_TEMP` override it and `None` omits the field so the provider default applies.
    pub temperature: Option<f64>,
    /// Optional `[user_sim]` endpoint for the simulated-user dialogue gate. `Some` builds a
    /// [`crate::backend::user_sim::UserSimGate`] in `answer` (paired with `user_sim_prompt`) so a
    /// text-only turn is dialogue-gated instead of accepted outright; `None` reproduces today's
    /// behavior exactly (no gate). See `backend::user_sim`.
    pub user_sim: Option<crate::lab::Endpoint>,
    /// The simulated-user persona prompt (`user_sim.md`), used VERBATIM as the gate's system
    /// message. Only consulted when `user_sim` is also `Some`; `None` disables the gate.
    pub user_sim_prompt: Option<String>,
    /// Opt-in rate-limit/retry policy carried from the reader's `[model]` endpoint, folded back into
    /// the `Endpoint` handed to the agent loop so `backend::resilience` throttles + retry-tunes this
    /// endpoint. `None` (the default) reproduces today's behavior exactly.
    pub rate_limit: Option<crate::lab::RateLimit>,
    /// Opt-in fallback chain carried from the reader's `[model]` endpoint. On a hard failure the
    /// agent loop advances through these via `resilience::call_resilient`. Empty (the default)
    /// reproduces today's behavior exactly (no fallback).
    pub fallback: Vec<crate::lab::Endpoint>,
    /// Which `ChatTransport` this backend's `[model]` endpoint speaks (`backend::transport::
    /// transport_for`). Defaults to `OpenAiChat` (today's hardcoded behavior) so every existing
    /// caller that doesn't set this field keeps driving `OpenAiTransport` unchanged; set to
    /// `Tensorzero` to route the reader through `TzTransport`'s native `/inference` + `/feedback`.
    pub api: crate::lab::ApiKind,
    /// TensorZero function name (only consulted when `api == Tensorzero`; see
    /// `crate::lab::Endpoint::function_name`).
    pub function_name: Option<String>,
    /// TensorZero feedback metric name for the graded judge score (only consulted when
    /// `api == Tensorzero`; see `crate::lab::Endpoint::feedback_score_metric`).
    pub feedback_score_metric: Option<String>,
    /// TensorZero feedback metric name for the boolean correctness flag (only consulted when
    /// `api == Tensorzero`; see `crate::lab::Endpoint::feedback_bool_metric`).
    pub feedback_bool_metric: Option<String>,
}

const MAX_ROUNDS: usize = 50;

impl AgentBackend for OpenAiBackend {
    fn needs_corpus(&self) -> bool {
        true
    }

    fn answer(&self, work: &Path, q: &Question) -> anyhow::Result<String> {
        self.answer_capturing(work, q, None)
    }
}

impl OpenAiBackend {
    /// Capture-aware reader. Identical to [`AgentBackend::answer`], but when `capture` is `Some`
    /// it records the full chat trajectory (the seed system+user, every assistant/tool round, and
    /// the final answer turn) into the sink for fine-tuning dataset collection (`kbx eval
    /// --capture`). `capture: None` reproduces `answer` exactly — the trait method just delegates
    /// here with `None`, so the non-capturing path is byte-identical to before.
    pub fn answer_capturing(
        &self,
        work: &Path,
        q: &Question,
        capture: Option<&mut crate::backend::agent_loop::CapturedEpisode>,
    ) -> anyhow::Result<String> {
        // The endpoint is the full chat-completions URL, used verbatim (no path is appended).
        // `Endpoint` here is a plain data carrier for `OpenAiTransport::call` — same fields
        // (`endpoint`/`model`/`api_key`/`timeout_secs`) `lmstudio_chat` used to take as loose
        // arguments. `resolve_key()` with an empty `api_key_env` reduces to "use `api_key` if
        // non-empty, else None" — the same behavior `self.api_key.as_deref()` had (an empty-string
        // key is filtered out later by `chat_http` regardless, in both the old and new paths).
        let ep = self.endpoint_config();
        let graph = if self.use_graph {
            glossa::graph::store::GraphStore::open(work).ok()
        } else {
            None
        };
        // Selected by `self.api` (default `OpenAiChat`, unchanged for every existing caller) —
        // `transport_for` builds `TzTransport` when `api = "tensorzero"`, so the eval reader gets
        // native TZ episode grouping + feedback without a separate code path.
        let transport = crate::backend::transport::transport_for(&ep);
        let tools = transport.tools_schema(graph.is_some());

        let trace = TraceLog::to_dir(work);
        // Open the index once per question; the closure reuses it (cached reader) for every
        // search/read in the agent loop instead of reopening per tool call.
        let idx = glossa::index::store::DocIndex::open_or_create(work)?;
        // Ontology-driven chain spec so glossary/related render identically to the MCP surface.
        let spec = glossa::tools::ChainSpec::from_ontology(
            &glossa::graph::ontology::Ontology::load_or_default(work),
        );
        // Per-episode reader-signal tracker (see `glossa_tools::ReaderSignals`): this wrapper only
        // acts on its PLATEAU kind — a NEUTRAL "gain has plateaued" observation applied via the
        // signal's own render (drop the redundant body on a drained plateau, or append the marker
        // when the call still turned up a little new ground). Repeat/Streak are left to the agent
        // loop's own pre-exec dedup / unproductive-streak guard (`agent_loop.rs`) — acting on them
        // here too would double up. Owned here so it resets per question; the POLICY (what to do
        // about a plateau) stays in the reader prompt / GEPA, not in the tool layer.
        let mut signals = crate::backend::glossa_tools::ReaderSignals::new();
        let exec = |name: &str, args: &Value| {
            let (mut body, ids) =
                execute_tool(name, args, work, &idx, graph.as_ref(), &spec, &trace);
            // Diagnostics: KB_EVAL_DUMP_TOOLS=1 prints each tool call + a truncated body to
            // stderr, so a smoke run doubles as an episode transcript (why the reader searches).
            if std::env::var("KB_EVAL_DUMP_TOOLS").is_ok() {
                let snippet: String = body.chars().take(500).collect();
                eprintln!(
                    "\n[TOOL] {name} {args}\n[BODY] {snippet}\n[--- {} chars ---]",
                    body.len()
                );
            }
            // Only id-surfacing RETRIEVAL calls feed the tracker; only its PLATEAU kind is acted on
            // here — Repeat/Streak are the loop's job (see the comment above).
            if crate::backend::glossa_tools::is_retrieval_tool(name) {
                let key = format!("{name}:{args}");
                body = crate::backend::glossa_tools::apply_plateau_render(
                    &mut signals,
                    name,
                    &key,
                    &ids,
                    body,
                );
            }
            // The reader path never feeds vision input — `execute_tool` already discards whatever
            // images `glossa_tools::exec` surfaced (e.g. from `read`); only `build::extract::
            // extract_doc`'s `--vision` path threads images (through the closure-based shim, whose
            // `exec` is 3-tuple). This `answer` drives the generic loop DIRECTLY, whose `exec` is
            // 2-tuple, so no image slot is needed here.
            (body, ids)
        };

        // `build_messages` already folds the system prompt in as the leading message (as the old
        // path's raw `messages` array did), so `system` is passed as `None` here — `OpenAiTransport
        // ::call` would otherwise prepend a SECOND system message.
        let seed_messages = self.build_messages(q);
        // Next-best-action on a stuck (repeated) call: fan the fixated query across the
        // complementary tools instead of re-running the dead one.
        let nba = |name: &str, args: &Value| {
            crate::backend::glossa_tools::next_best_action(
                name,
                args,
                work,
                &idx,
                graph.as_ref(),
                &spec,
                &trace,
            )
        };
        // Simulated-user dialogue gate: built only when BOTH the `[user_sim]` endpoint and the
        // persona prompt are present. Absent -> `None` -> the loop keeps today's behavior exactly
        // (a text-only turn is the final answer).
        let gate = match (&self.user_sim, &self.user_sim_prompt) {
            (Some(ep), Some(prompt)) => {
                Some(crate::backend::user_sim::UserSimGate::new(ep, prompt))
            }
            _ => None,
        };
        let user_sim = gate
            .as_ref()
            .map(|g| g as &dyn crate::backend::user_sim::DialogueGate);
        let raw = crate::backend::agent_loop::run_agent_loop_capturing(
            transport.as_ref(),
            &ep,
            None,
            seed_messages,
            Some(&tools),
            exec,
            nba,
            MAX_ROUNDS,
            user_sim,
            capture,
        )?;
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
            temperature: None,
            user_sim: None,
            user_sim_prompt: None,
            rate_limit: None,
            fallback: Vec::new(),
            api: crate::lab::ApiKind::default(),
            function_name: None,
            feedback_score_metric: None,
            feedback_bool_metric: None,
        }
    }

    /// Build the `Endpoint` this backend's `[model]` config resolves to — the SAME construction
    /// `answer_capturing` uses for `transport_for`, factored out so [`post_feedback`] can hand it
    /// to a freshly-built transport without re-running the reader.
    fn endpoint_config(&self) -> crate::lab::Endpoint {
        crate::lab::Endpoint {
            endpoint: self.endpoint.clone(),
            model: self.model.clone(),
            api_key: self.api_key.clone().unwrap_or_default(),
            api_key_env: String::new(),
            timeout_secs: self.timeout.as_secs(),
            api: self.api,
            temperature: self.temperature,
            rate_limit: self.rate_limit.clone(),
            fallback: self.fallback.clone(),
            function_name: self.function_name.clone(),
            feedback_score_metric: self.feedback_score_metric.clone(),
            feedback_bool_metric: self.feedback_bool_metric.clone(),
        }
    }

    /// Post episode-level feedback through this backend's own transport (selected by `self.api`).
    /// A no-op for every non-TensorZero transport (the default `ChatTransport::post_feedback`) —
    /// only `TzTransport` overrides it, and `run_eval` only calls this when `episode::current()`
    /// was `Some` after the reader ran, so off-TZ this is never reached in the first place.
    pub fn post_feedback(&self, episode_id: &str, metrics: &[(&str, serde_json::Value)]) {
        let ep = self.endpoint_config();
        crate::backend::transport::transport_for(&ep).post_feedback(episode_id, metrics);
    }
}

/// Minimal one-shot OpenAI-compatible chat call: a plain completion (e.g. the file-prompt judge in
/// `judge.rs`) instead of the full tool-calling agent loop. Builds a tools-free request body and
/// drives it through `chat_http` — the same transport the agent loop uses. `temperature` follows
/// the same uniform rule as every other call site: `Some(t)` samples at `t`, `None` OMITS the
/// `temperature` field so the provider/model applies its own default (callers resolve it via
/// `Endpoint::resolve_temperature`, i.e. `KB_EVAL_TEMP` env > `ep.temperature` > `None`).
pub(crate) fn chat_once(
    endpoint: &str,
    model: &str,
    messages: &[Value],
    api_key: Option<&str>,
    timeout_secs: u64,
    temperature: Option<f64>,
) -> anyhow::Result<Value> {
    // One-shot (e.g. the file-prompt judge): NO `tools` field at all — a strict provider (the
    // MiMo/OpenCode Zen endpoint this backend targets) rejects an empty `tools: []` array.
    // NOTE: no `min_p` here either — it's a non-OpenAI extension a strict provider may 400 on.
    let max_tokens: u64 = crate::backend::transport::openai::agent_max_tokens();
    let mut body = json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    });
    // Include `temperature` only when set — `None` omits it so the provider default applies.
    if let Some(t) = temperature {
        body["temperature"] = json!(t);
    }
    chat_http_full(
        endpoint,
        api_key,
        &body,
        Duration::from_secs(timeout_secs),
        crate::backend::resilience::RetryPolicy::default(),
    )
    .map(|v| {
        v.pointer("/choices/0/message")
            .cloned()
            .unwrap_or_else(|| json!({}))
    })
}

/// True when an error body reflects a transient UPSTREAM failure of the gateway's own backend (its
/// fetch/predict to the model dropped, timed out, or was overloaded) rather than a fault in OUR
/// request. Some OpenAI-compatible gateways (e.g. opencode zen) surface these with a client-error
/// status (400) or an HTTP-200 `{"error"}` body, so status code alone misclassifies them as fatal.
/// A genuine bad-request — bad key, malformed payload, unknown model — matches none of these and
/// still fails fast, so a real config bug isn't hidden behind seconds of backoff.
///
/// `pub(crate)` so the moved HTTP bridge in `transport::openai::chat_http_full` can share this one
/// predicate (the accounting statics + this classifier stay HOME here; the transport calls back in)
/// instead of duplicating the needle list.
pub(crate) fn is_transient_upstream(body: &str) -> bool {
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

/// Tally one successful chat response's usage into the process-global running counters (see
/// `usage_split_with_prefix`/`NEW_TOKENS`/`CACHED_TOKENS`/`CACHE_ESTIMATED`) and advance this
/// thread's conversation-prefix tracker (`PREV_PROMPT_TOKENS`). Called once per successful HTTP
/// round-trip from `transport::openai::chat_http_full` — the accounting subsystem lives HERE (one
/// definition of every static), and the moved HTTP bridge calls back into it. `prev` is read from
/// the CALLING THREAD's own cell BEFORE this call's own `prompt_tokens` overwrites it (see
/// `PREV_PROMPT_TOKENS`'s doc comment for the per-thread sequential-calls assumption).
pub(crate) fn record_usage(resp: &Value) {
    let prev = PREV_PROMPT_TOKENS.with(|prev| prev.get());
    let (new, cached, estimated) = usage_split_with_prefix(resp, prev);
    NEW_TOKENS.fetch_add(new, Ordering::Relaxed);
    CACHED_TOKENS.fetch_add(cached, Ordering::Relaxed);
    if estimated {
        CACHE_ESTIMATED.store(true, Ordering::Relaxed);
    }
    PREV_PROMPT_TOKENS.with(|prev| prev.set(prompt_tokens(resp)));
}

/// The sync HTTP bridge (raw JSON round-trip + retry/backoff), `tools_schema`, `content_of`, and
/// `parse_tool_args` live in `transport::openai` as the shared core behind `OpenAiTransport`.
/// Re-imported here so `chat_once` and the schema tests keep working unchanged. `chat_http` returns
/// the assistant message (`choices[0].message`); `chat_http_full` returns the WHOLE response so
/// `chat_once` can extract the message itself. Both record token usage exactly once via
/// `record_usage`. (The retired `lmstudio_chat` is gone — its body-building lives in
/// `transport::openai::agent_chat_full` and its resample loop in `backend::resample`.)
pub(crate) use crate::backend::transport::openai::{
    chat_http, chat_http_full, content_of, parse_tool_args, tools_schema,
};

/// Unproductive-streak threshold. MOVED to `backend::agent_loop` (Task 3 of the multi-api-
/// transport plan) along with the loop's dedup/streak/NBA logic; re-exported here so this
/// module's own not-yet-moved tests (and `build::extract`'s regression test) keep compiling
/// against `crate::backend::openai::UNPRODUCTIVE_STREAK_K` unchanged.
pub(crate) use crate::backend::agent_loop::UNPRODUCTIVE_STREAK_K;

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

/// Thin backward-compatible shim over `agent_loop::run_agent_loop`: adapts the pre-transport
/// calling convention (`chat: Fn(&[Value]) -> Result<Value>`, returning the assistant `message`
/// object already extracted from `choices[0].message`) onto the generic, transport-driven loop.
///
/// All dedup/streak/NBA logic now lives in exactly ONE place (`agent_loop::run_agent_loop`) — this
/// function does not reimplement any of it, it only translates the closure into a one-off
/// `ChatTransport` (`ClosureTransport`, below) so `run_agent_loop`'s three callers not yet migrated
/// to `ChatTransport` directly (`build::extract`, `distil::chain`, `gepa_graph` — Phase 2 Task 6 of
/// the multi-api-transport plan) keep compiling and behaving identically. `OpenAiBackend::answer`
/// itself no longer goes through this shim — it drives `agent_loop::run_agent_loop` directly with
/// a real `OpenAiTransport`.
pub(crate) fn run_agent_loop<C, F, N>(
    chat: C,
    messages: Vec<Value>,
    exec: F,
    on_repeat: N,
    max_rounds: usize,
    user_sim: Option<&dyn crate::backend::user_sim::DialogueGate>,
) -> anyhow::Result<String>
where
    C: FnMut(&[Value]) -> anyhow::Result<Value>,
    F: FnMut(&str, &Value) -> (String, Vec<String>, Vec<DocImage>),
    N: Fn(&str, &Value) -> String,
{
    // Images surfaced by `exec` during the current round ride here until the transport's
    // `push_tool_results` drains them into ONE follow-up vision user message (build --vision only;
    // every other caller's `exec` returns an empty image vec, so nothing is ever appended and the
    // transcript stays byte-identical to the non-vision path). Shared via `Rc<RefCell<…>>` so the
    // 2-tuple `exec` adapter below and the `ClosureTransport` both reach one buffer without either
    // borrowing the other — the generic `agent_loop::run_agent_loop` seam is deliberately image-
    // agnostic (its `exec` is `(String, Vec<String>)`), so vision threading lives HERE in the shim,
    // keeping that one loop implementation free of any image concern. The per-conversation prefix
    // reset (`reset_conversation_prefix`) now happens inside `agent_loop::run_agent_loop` itself, so
    // this shim and `OpenAiBackend::answer` (which drives that loop directly) both get it.
    let pending_images: Rc<std::cell::RefCell<Vec<DocImage>>> =
        Rc::new(std::cell::RefCell::new(Vec::new()));
    let sink = Rc::clone(&pending_images);
    let mut exec = exec;
    let exec2 = move |name: &str, args: &Value| {
        let (body, ids, images) = exec(name, args);
        if !images.is_empty() {
            sink.borrow_mut().extend(images);
        }
        (body, ids)
    };
    let transport = ClosureTransport::new(chat, pending_images);
    // A blank `Endpoint`: `ClosureTransport::call` ignores it entirely (the wrapped closure
    // already captures its own endpoint/model/api_key/tools, exactly as the old direct callers
    // of `lmstudio_chat` did).
    let ep = crate::lab::Endpoint {
        endpoint: String::new(),
        model: String::new(),
        api_key: String::new(),
        api_key_env: String::new(),
        timeout_secs: 120,
        api: crate::lab::ApiKind::default(),
        temperature: None,
        // Blank endpoint -> no throttle, no fallback (the shim's `ClosureTransport` captures the
        // real endpoint). `call_resilient` therefore makes exactly one primary call, unchanged.
        rate_limit: None,
        fallback: Vec::new(),
        function_name: None,
        feedback_score_metric: None,
        feedback_bool_metric: None,
    };
    crate::backend::agent_loop::run_agent_loop(
        &transport, &ep, None, messages, None, exec2, on_repeat, max_rounds, user_sim,
    )
}

/// Adapts a legacy `FnMut(&[Value]) -> Result<Value>` chat closure into a one-off `ChatTransport`,
/// so `run_agent_loop`'s shim above can drive the generic loop. `push_assistant_turn`/
/// `push_tool_results`/tool-call parsing mirror `OpenAiTransport`'s shapes exactly (same raw-message
/// echo, same per-call `{role:"tool"}` push). `pending_images` is the shim's shared vision buffer:
/// `push_tool_results` drains it into a follow-up `role:"user"` image message right after the tool
/// results (build --vision).
///
/// The closure returns EITHER the FULL chat response (the real callers, via
/// `transport::openai::agent_chat_full` — so `choices[0].finish_reason` is available to the
/// resample layer) OR a bare assistant message object (this module's shim unit tests). `call`
/// accepts both: it uses the full-response shape when `choices[0].message` is present (also lifting
/// `finish_reason`), and otherwise treats the returned `Value` as the message itself
/// (`finish_reason` then `None`).
struct ClosureTransport<C> {
    chat: std::cell::RefCell<C>,
    pending_images: Rc<std::cell::RefCell<Vec<DocImage>>>,
}

impl<C> ClosureTransport<C> {
    fn new(chat: C, pending_images: Rc<std::cell::RefCell<Vec<DocImage>>>) -> Self {
        Self {
            chat: std::cell::RefCell::new(chat),
            pending_images,
        }
    }
}

impl<C> ChatTransport for ClosureTransport<C>
where
    C: FnMut(&[Value]) -> anyhow::Result<Value>,
{
    fn tools_schema(&self, graph_on: bool) -> Value {
        tools_schema(graph_on)
    }

    fn call(
        &self,
        _ep: &crate::lab::Endpoint,
        _system: Option<&str>,
        messages: &[Value],
        _tools: Option<&Value>,
        _temperature: Option<f64>,
    ) -> anyhow::Result<TurnReply> {
        let v = (self.chat.borrow_mut())(messages)?;
        // Full response (real callers) vs bare message (shim tests): a full response carries
        // `choices[0].message` + `choices[0].finish_reason`; a bare message IS the message.
        let has_choices = v.pointer("/choices/0/message").is_some();
        let finish_reason = v
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
            .map(str::to_string);
        let msg = if has_choices {
            v.pointer("/choices/0/message").cloned().unwrap_or(v)
        } else {
            v
        };
        let tool_calls: Vec<ToolCall> = msg
            .get("tool_calls")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .map(|call| ToolCall {
                        id: call
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        name: call
                            .pointer("/function/name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        args: parse_tool_args(call),
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(TurnReply {
            text: Some(content_of(&msg)),
            tool_calls,
            finish_reason,
            raw: msg,
        })
    }

    fn push_assistant_turn(&self, messages: &mut Vec<Value>, reply: &TurnReply) {
        messages.push(reply.raw.clone());
    }

    fn push_tool_results(&self, messages: &mut Vec<Value>, results: &[(String, String)]) {
        for (id, body) in results {
            messages.push(json!({ "role": "tool", "tool_call_id": id, "content": body }));
        }
        // Vision (build --vision only): a `role:"tool"` message can't carry image content on this
        // endpoint, so any images this round's `exec` surfaced (buffered by the shim's 2-tuple
        // adapter) ride in ONE follow-up `role:"user"` message right after the tool results. Empty
        // for every non-vision caller, so `vision_user_message` returns `None` and nothing is added.
        let images: Vec<DocImage> = self.pending_images.borrow_mut().drain(..).collect();
        if let Some(img_msg) = vision_user_message(&images) {
            messages.push(img_msg);
        }
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
            assert!(
                is_transient_upstream(transient),
                "should retry: {transient}"
            );
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
    fn usage_split_reports_real_cache_when_server_provides_it() {
        // prompt_tokens_details.cached_tokens present -> the REAL split: cached tallied
        // separately, new is the freshly-processed prompt remainder plus all of completion,
        // estimated is false regardless of prev_prompt.
        assert_eq!(
            usage_split_with_prefix(
                &json!({
                    "usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 200,
                        "prompt_tokens_details": {"cached_tokens": 800}
                    }
                }),
                0
            ),
            (400, 800, false)
        );
    }

    #[test]
    fn usage_split_no_usage_object_is_all_zero_not_estimated() {
        // No usage object at all -> (0, 0, false), never a panic.
        assert_eq!(
            usage_split_with_prefix(&json!({"choices": []}), 500),
            (0, 0, false)
        );
    }

    #[test]
    fn usage_split_estimates_cache_from_prev_prompt_when_server_omits_it() {
        // No prompt_tokens_details at all (e.g. LM Studio, verified absent) and this round's
        // prompt GREW over the previous round's -> the grown prefix is estimated cached: this is
        // the worked example from the task brief (prev=1000, prompt=1400, completion=200 ->
        // new=600, cached=1000, estimated=true).
        assert_eq!(
            usage_split_with_prefix(
                &json!({"usage": {"prompt_tokens": 1400, "completion_tokens": 200}}),
                1000
            ),
            (600, 1000, true)
        );
    }

    #[test]
    fn usage_split_shrink_guard_assumes_no_cached_prefix() {
        // This round's prompt is SMALLER than prev_prompt -> can't be a re-sent-prefix growth
        // (e.g. a fresh/different conversation reusing the same global counter) -> cached_est is
        // 0, not a negative/garbage figure, and new = prompt + completion in full.
        assert_eq!(
            usage_split_with_prefix(
                &json!({"usage": {"prompt_tokens": 10, "completion_tokens": 5}}),
                500
            ),
            (15, 0, true)
        );
    }

    #[test]
    fn usage_split_zero_prev_prompt_never_estimates_a_cache() {
        // prev_prompt == 0 means there IS no previous request in this conversation yet (the
        // first call after reset_conversation_prefix) -> nothing to credit as a re-sent prefix,
        // even though 0 technically satisfies "prompt >= prev".
        assert_eq!(
            usage_split_with_prefix(
                &json!({"usage": {"prompt_tokens": 10, "completion_tokens": 5}}),
                0
            ),
            (15, 0, true)
        );
    }

    #[test]
    fn token_summary_carries_tilde_only_when_estimated() {
        reset_tokens();
        NEW_TOKENS.store(600, Ordering::Relaxed);
        CACHED_TOKENS.store(1000, Ordering::Relaxed);
        assert_eq!(token_summary(), "600 new · 1.0k cache");
        assert!(!cache_is_estimated());

        CACHE_ESTIMATED.store(true, Ordering::Relaxed);
        assert_eq!(token_summary(), "600 new · 1.0k cache~");
        assert!(cache_is_estimated());
        reset_tokens();
    }

    /// Integration test for the real `chat_http` wiring (not just the pure `usage_split_with_prefix`
    /// fn): two sequential calls against a mock server, mirroring one agent-loop conversation where
    /// round 2 re-sends round 1's whole prompt as a prefix and appends to it. The server (like LM
    /// Studio) never reports `prompt_tokens_details`, so both calls fall to the self-computed
    /// estimate path — this proves `chat_http` actually reads/writes `PREV_PROMPT_TOKENS` and flips
    /// `CACHE_ESTIMATED` end to end, not just that the pure fn is correct in isolation.
    #[test]
    fn chat_http_estimates_cache_across_two_sequential_calls_in_one_conversation() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let responses = [
            r#"{"choices":[{"message":{"role":"assistant","content":"r1"}}],"usage":{"prompt_tokens":1000,"completion_tokens":50}}"#,
            r#"{"choices":[{"message":{"role":"assistant","content":"r2"}}],"usage":{"prompt_tokens":1400,"completion_tokens":200}}"#,
        ];
        let server = std::thread::spawn(move || {
            for body in responses {
                let (mut sock, _) = listener.accept().unwrap();
                let mut buf = [0u8; 4096];
                let _ = sock.read(&mut buf).unwrap();
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                sock.write_all(resp.as_bytes()).unwrap();
            }
        });

        reset_tokens(); // fresh conversation: PREV_PROMPT_TOKENS=0, CACHE_ESTIMATED=false
        let endpoint = format!("http://127.0.0.1:{port}/v1/chat/completions");
        let body = json!({"model": "m", "messages": []});

        chat_http(&endpoint, None, &body, Duration::from_secs(5)).unwrap();
        // Round 1: prev_prompt was 0 -> no prefix to credit -> new = 1000+50, cached = 0.
        assert_eq!(new_tokens(), 1050);
        assert_eq!(cached_tokens(), 0);
        assert!(
            cache_is_estimated(),
            "server omits prompt_tokens_details -> estimate path"
        );

        chat_http(&endpoint, None, &body, Duration::from_secs(5)).unwrap();
        // Round 2: prev_prompt is now round 1's 1000 (the re-sent prefix) -> cached_est=1000,
        // new = (1400-1000)+200 = 600. Running totals accumulate across both calls.
        assert_eq!(new_tokens(), 1050 + 600);
        assert_eq!(cached_tokens(), 1000);

        server.join().unwrap();
        reset_tokens();
    }

    #[test]
    fn reset_conversation_prefix_stops_the_next_call_crediting_a_stale_prefix() {
        // A new conversation must not inherit the previous conversation's tail prompt_tokens as a
        // false "cached prefix" — `run_agent_loop` calls this at the top of every conversation for
        // exactly this reason.
        reset_tokens();
        PREV_PROMPT_TOKENS.with(|prev| prev.set(1000));
        reset_conversation_prefix();
        let (new, cached, estimated) = usage_split_with_prefix(
            &json!({"usage": {"prompt_tokens": 10, "completion_tokens": 5}}),
            PREV_PROMPT_TOKENS.with(|prev| prev.get()),
        );
        assert_eq!((new, cached, estimated), (15, 0, true));
    }

    /// Task 2 (parallel-jobs): `PREV_PROMPT_TOKENS` is thread-local, so two "workers" running
    /// concurrent conversations on separate threads each track their OWN previous `prompt_tokens`
    /// independently — worker A's stored prefix must never leak into worker B's estimate (the bug
    /// a process-global `AtomicU64` would have under `kbx --jobs N`). The aggregate `NEW_TOKENS`/
    /// `CACHED_TOKENS` are still process-global atomics, so their sum across both threads must
    /// equal the sum of what each thread computed on its own.
    #[test]
    fn prev_prompt_tokens_is_thread_local_across_parallel_workers() {
        reset_tokens();

        // Worker A: a 2-round conversation with one prefix size; worker B: a DIFFERENT 2-round
        // conversation with a different prefix size, running concurrently on another thread. If
        // the tracker were a shared global, whichever thread's `store` landed last would corrupt
        // the other's next `load` and the per-round splits below would not match.
        let worker_a = std::thread::spawn(|| {
            reset_conversation_prefix(); // each worker resets its own thread-local at conversation start
            let prev1 = PREV_PROMPT_TOKENS.with(|c| c.get());
            let (n1, c1, _) = usage_split_with_prefix(
                &json!({"usage": {"prompt_tokens": 1000, "completion_tokens": 50}}),
                prev1,
            );
            PREV_PROMPT_TOKENS.with(|c| c.set(1000));
            NEW_TOKENS.fetch_add(n1, Ordering::Relaxed);
            CACHED_TOKENS.fetch_add(c1, Ordering::Relaxed);

            let prev2 = PREV_PROMPT_TOKENS.with(|c| c.get());
            let (n2, c2, _) = usage_split_with_prefix(
                &json!({"usage": {"prompt_tokens": 1400, "completion_tokens": 200}}),
                prev2,
            );
            NEW_TOKENS.fetch_add(n2, Ordering::Relaxed);
            CACHED_TOKENS.fetch_add(c2, Ordering::Relaxed);
            (n1, c1, n2, c2)
        });

        let worker_b = std::thread::spawn(|| {
            reset_conversation_prefix();
            let prev1 = PREV_PROMPT_TOKENS.with(|c| c.get());
            let (n1, c1, _) = usage_split_with_prefix(
                &json!({"usage": {"prompt_tokens": 300, "completion_tokens": 20}}),
                prev1,
            );
            PREV_PROMPT_TOKENS.with(|c| c.set(300));
            NEW_TOKENS.fetch_add(n1, Ordering::Relaxed);
            CACHED_TOKENS.fetch_add(c1, Ordering::Relaxed);

            let prev2 = PREV_PROMPT_TOKENS.with(|c| c.get());
            let (n2, c2, _) = usage_split_with_prefix(
                &json!({"usage": {"prompt_tokens": 500, "completion_tokens": 40}}),
                prev2,
            );
            NEW_TOKENS.fetch_add(n2, Ordering::Relaxed);
            CACHED_TOKENS.fetch_add(c2, Ordering::Relaxed);
            (n1, c1, n2, c2)
        });

        let (a_n1, a_c1, a_n2, a_c2) = worker_a.join().unwrap();
        let (b_n1, b_c1, b_n2, b_c2) = worker_b.join().unwrap();

        // Worker A: round 1 has no prefix (fresh thread-local) -> new=1050, cached=0. Round 2
        // re-sends round 1's 1000 as prefix -> new=(1400-1000)+200=600, cached=1000.
        assert_eq!((a_n1, a_c1), (1050, 0));
        assert_eq!((a_n2, a_c2), (600, 1000));
        // Worker B: independently, round 1 new=320, cached=0. Round 2 new=(500-300)+40=240,
        // cached=300. If A's and B's thread-locals had collided, at least one of these would be
        // wrong (e.g. B's round 1 would spuriously credit A's leftover prefix).
        assert_eq!((b_n1, b_c1), (320, 0));
        assert_eq!((b_n2, b_c2), (240, 300));

        // The process-global aggregates still sum correctly across both worker threads.
        assert_eq!(new_tokens(), a_n1 + a_n2 + b_n1 + b_n2);
        assert_eq!(cached_tokens(), a_c1 + a_c2 + b_c1 + b_c2);

        reset_tokens();
    }

    #[test]
    fn human_tokens_formats_compactly() {
        assert_eq!(human_tokens(999), "999");
        assert_eq!(human_tokens(1500), "1.5k");
        assert_eq!(human_tokens(2_000_000), "2.0M");
    }

    #[test]
    fn eta_secs_none_at_zero_progress_and_when_pos_exceeds_len() {
        assert_eq!(eta_secs(100, 0, 1871), None); // no progress yet -> no basis for an estimate
        assert_eq!(eta_secs(100, 5, 3), None); // guarded, shouldn't happen in practice
    }

    #[test]
    fn eta_secs_linear_rate_after_one_step() {
        // 1 of 1871 steps took 100s -> remaining 1870 steps at the same rate.
        assert_eq!(eta_secs(100, 1, 1871), Some(100 * 1870));
    }

    #[test]
    fn eta_secs_zero_when_almost_done() {
        assert_eq!(eta_secs(100, 10, 10), Some(0));
    }

    #[test]
    fn format_eta_matches_elapsed_precise_style() {
        assert_eq!(format_eta(None), "--:--:--");
        assert_eq!(format_eta(Some(0)), "00:00:00");
        assert_eq!(format_eta(Some(65)), "00:01:05");
        // Past a day -> "Nd HH:MM:SS", matching indicatif's own FormattedDuration exactly (this
        // is the stable replacement for the corrupted "{eta_precise}" that used to show "448d").
        assert_eq!(format_eta(Some(3 * 86400 + 5 * 3600)), "3d 05:00:00");
    }

    #[test]
    fn tokens_used_resets_and_accumulates() {
        // Process-global counters — reset first so this test isn't order-dependent on whatever
        // other tests in this file (or run concurrently) touched them.
        reset_tokens();
        assert_eq!(new_tokens(), 0);
        assert_eq!(cached_tokens(), 0);
        assert!(!cache_is_estimated());
        let (n1, c1, e1) = usage_split_with_prefix(
            &json!({"usage": {"prompt_tokens": 5, "completion_tokens": 2}}),
            0,
        );
        NEW_TOKENS.fetch_add(n1, Ordering::Relaxed);
        CACHED_TOKENS.fetch_add(c1, Ordering::Relaxed);
        if e1 {
            CACHE_ESTIMATED.store(true, Ordering::Relaxed);
        }
        let (n2, c2, e2) = usage_split_with_prefix(
            &json!({
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": 6}
                }
            }),
            5,
        );
        NEW_TOKENS.fetch_add(n2, Ordering::Relaxed);
        CACHED_TOKENS.fetch_add(c2, Ordering::Relaxed);
        if e2 {
            CACHE_ESTIMATED.store(true, Ordering::Relaxed);
        }
        assert_eq!(new_tokens(), 12); // (5+2) + (10-6+1)
        assert_eq!(cached_tokens(), 6);
        // First call had no usage details -> reported cached path never fires -> not estimated
        // either (prev_prompt was 0, so the shrink-guard/zero-prev rule kept cached_est at 0, but
        // e1 IS true since prompt_tokens_details was absent); second call reported real cache ->
        // e2 is false. Overall the run is estimate-tainted because of the first call.
        assert!(e1);
        assert!(!e2);
        assert!(cache_is_estimated());
        reset_tokens();
        assert_eq!(new_tokens(), 0);
        assert_eq!(cached_tokens(), 0);
        assert!(!cache_is_estimated());
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
        assert!(msgs[0]["content"]
            .as_str()
            .unwrap()
            .contains("SYS-MARKER-123"));
        assert_eq!(msgs[1]["role"], "user");
        assert!(msgs[1]["content"].as_str().unwrap().contains("hi"));
    }

    // `chat_once_posts_endpoint_url_verbatim` and `parse_tool_args_handles_string_and_object`
    // MOVED to `transport::openai`'s test module (Task 2 of the multi-api-transport plan) — they
    // now exercise the moved `chat_http`/`parse_tool_args` core directly at its new home.

    // `loop_returns_direct_answer_when_no_tool_calls`, `loop_dispatches_tool_then_answers`,
    // `loop_dedupes_consecutive_identical_tool_calls`, `loop_reexecutes_when_args_differ`,
    // `loop_stops_at_max_rounds`, and `loop_unproductive_streak_feeds_steer_after_k_plus_one_calls`
    // MOVED to `backend::agent_loop`'s test module (Task 3 of the multi-api-transport plan) — they
    // now drive the generic `agent_loop::run_agent_loop` through a mock `ChatTransport` instead of
    // this module's closure-based shim. `loop_unproductive_streak_never_fires_when_calls_are_productive`
    // and the graph-tool id-extraction regression tests below stay here: they exercise this
    // module's `run_agent_loop` shim (still the closure-based entry point `build::extract`/
    // `distil::chain`/`gepa_graph` use) and/or this module's own `execute_tool`.

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
            (
                format!("hit {}", *c),
                vec![format!("doc-{}.md", *c)],
                Vec::new(),
            ) // new id every call
        };
        let out = run_agent_loop(chat, vec![], exec, nudge, rounds_to_run + 2, None).unwrap();
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
        let out = run_agent_loop(chat, vec![], exec, nudge, n + 2, None).unwrap();
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
        let out =
            run_agent_loop(chat, vec![], exec, nudge, UNPRODUCTIVE_STREAK_K + 3, None).unwrap();
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
        let out = run_agent_loop(chat, vec![], exec, nudge, n + 2, None).unwrap();
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
        let out =
            run_agent_loop(chat, vec![], exec, nudge, UNPRODUCTIVE_STREAK_K + 3, None).unwrap();
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
        assert_eq!(
            content.len(),
            2,
            "one text part + one image_url part: {content:?}"
        );
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
        let url = content[1]["image_url"]["url"]
            .as_str()
            .expect("image_url.url string");
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
        assert_eq!(
            decoded, img.bytes,
            "decoded payload must round-trip the original JPEG bytes"
        );
    }

    #[test]
    fn vision_message_caps_at_max_images_per_turn() {
        let images: Vec<DocImage> = (0..(MAX_IMAGES_PER_TURN as u8 + 2))
            .map(stub_image)
            .collect();
        assert!(
            images.len() > MAX_IMAGES_PER_TURN,
            "test must exceed the cap"
        );
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
            (
                "(scanned page text)".to_string(),
                vec!["scan.pdf".to_string()],
                vec![stub_image(9)],
            )
        };
        let out = run_agent_loop(chat, vec![], exec, nudge, 3, None).unwrap();
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
            assert!(
                !has_image_msg,
                "vision-off must never append an image message: {msgs:?}"
            );
            Ok(json!({ "role": "assistant", "content": "ANSWER: done" }))
        };
        let exec = |_: &str, _: &Value| {
            (
                "(scanned page text)".to_string(),
                vec!["scan.pdf".to_string()],
                Vec::new(),
            )
        };
        let out = run_agent_loop(chat, vec![], exec, nudge, 3, None).unwrap();
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
        assert!(
            tool_names(&tools_schema(false)).contains(&"grep".into()),
            "graph-OFF must advertise grep"
        );
        assert!(
            tool_names(&tools_schema(true)).contains(&"grep".into()),
            "graph-ON must advertise grep"
        );
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
}
