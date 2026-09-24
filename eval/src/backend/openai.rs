use super::{prompt, AgentBackend};
use crate::backend::loop_compat::execute_tool;
use crate::backend::transport::ChatTransport;
use crate::backend::vision::VisionTransport;
use crate::dataset::Question;
use glossa::read::DocImage;
use glossa::trace::TraceLog;
use serde_json::{json, Value};
use std::path::Path;
use std::rc::Rc;
use std::time::Duration;

/// Token/resample accounting (`NEW_TOKENS`/`CACHED_TOKENS`/…) moved to `backend::accounting`, the
/// progress bar's `StatusTicker` to `backend::progress`, the reader-dialogue store to
/// `backend::dialogue`, the `--vision` image threading to `backend::vision`, and the legacy
/// agent-loop shim (`run_agent_loop`/`ClosureTransport`/`execute_tool`) to `backend::loop_compat`.
/// The sync HTTP bridge, `tools_schema`, `content_of`, and `parse_tool_args` live in
/// `transport::openai` behind `OpenAiTransport`. This module keeps only the OpenAI-specific
/// remnant: `OpenAiBackend`, the one-shot `chat_once`/`chat_once_resampled` helpers, the
/// `is_transient_upstream` classifier, and the `answer_tool_context` surface.
///
/// `record_usage` lives in `backend::accounting` now; it is re-exported here under the historical
/// `crate::backend::openai::record_usage` path because `transport::openai`'s HTTP bridge imports it
/// via that path and that file is not modified as part of this split.
pub(crate) use crate::backend::accounting::record_usage;

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
    /// Enable vision: advertise `read(page_image)` in the tool schema AND feed the images a tool
    /// call returns back to the model — mirrors an MCP server started with `--vision`. Plumbed from
    /// the `kbx eval --vision` CLI flag (`false`/off by default, matching today's behavior exactly)
    /// into `answer_tool_context`'s `no_image: !vision`. When on (and the reader speaks the
    /// OpenAI-compatible chat API), `answer_capturing` wraps the transport in `VisionTransport`, so
    /// page images a `read` surfaces ride to the model in a follow-up `role:"user"` image message
    /// right after the tool result (via `vision_user_message`) — the same mechanism `build
    /// --vision`'s extract path uses. Off, the answer path is byte-identical to before (no image
    /// message emitted).
    pub vision: bool,
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
    /// Run-wide shared retrieval snapshot. When `Some`, `answer_capturing` reuses this handle's
    /// graph + index (the CSR/PPR matrix is built ONCE for the whole run) instead of opening per
    /// question — the eval counterpart to the MCP server's shared handle, and the fix for the
    /// per-question `GraphStore::open` that made RSS scale with the worker count. `None` reproduces
    /// today's per-question open exactly. `run_eval` opens one and clones it into every worker.
    pub shared: Option<std::sync::Arc<glossa::graph::handle::GraphHandle>>,
    /// Agent-loop round cap for THIS reader. `run_eval` sets it from `lab.toml`'s `[tuning]
    /// max_rounds` (so an operator's config actually bounds the eval reader, not just reason/build);
    /// callers without a lab config use [`DEFAULT_MAX_ROUNDS`].
    pub max_rounds: usize,
    /// Opt-in extra request headers carried from the reader's `[model]` endpoint, folded back into
    /// the `Endpoint` handed to the agent loop (see `crate::lab::Endpoint::headers`). Empty (the
    /// default) reproduces today's behavior exactly (no extra headers).
    pub headers: std::collections::BTreeMap<String, String>,
    /// Gates the per-episode `ReaderSignals` PLATEAU tracker (spec: dedup unification §4.4) —
    /// `true` builds a live `ReaderSignals::new()` in `answer_capturing`; `false` (the default,
    /// mirroring `config::defaults::DEDUP`) builds `ReaderSignals::disabled()`, a stateless
    /// passthrough, so an eval run left at the default sees the SAME (no-marker) retrieval bodies
    /// prod's MCP server serves by default. `kbx eval --dedup` overrides; the legacy `kb-eval run`
    /// CLI has no flag and always resolves to the const default.
    pub dedup: bool,
}

/// Fallback agent-loop round cap for the eval reader when `lab.toml`'s `[tuning] max_rounds` is
/// unset. `run_eval` passes the configured value through `OpenAiBackend.max_rounds`; this is only the
/// default for callers without a lab config (the legacy CLI and tests).
pub const DEFAULT_MAX_ROUNDS: usize = 50;

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
        // Retrieval state: reuse the run-wide shared `GraphHandle` when present (opened ONCE in
        // `run_eval` and cloned into every worker — the graph/CSR matrix is built once, not per
        // question), else open per question (today's behavior). `graph` stays `None` in the
        // graph-OFF baseline arm regardless, so the A/B knob is unchanged.
        let local_handle;
        // The shared handle's search index is behind an `ArcSwap`; hold its current snapshot in an
        // outer binding so the `&DocIndex` borrow below outlives the match.
        let shared_idx;
        let (idx, graph): (
            &glossa::index::store::DocIndex,
            Option<&glossa::graph::store::GraphStore>,
        ) = match &self.shared {
            Some(h) => {
                shared_idx = h.idx();
                (
                    &shared_idx,
                    if self.use_graph { Some(&h.graph) } else { None },
                )
            }
            None => {
                local_handle = (
                    glossa::index::store::DocIndex::open_or_create(work)?,
                    if self.use_graph {
                        glossa::graph::store::GraphStore::open(work).ok()
                    } else {
                        None
                    },
                );
                (&local_handle.0, local_handle.1.as_ref())
            }
        };
        // Selected by `self.api` (default `OpenAiChat`, unchanged for every existing caller) —
        // `transport_for` builds `TzTransport` when `api = "tensorzero"`, so the eval reader gets
        // native TZ episode grouping + feedback without a separate code path.
        let transport = crate::backend::transport::transport_for(&ep);
        // Serving parity: withhold `verify` from the advertised schema when the answer-grounding
        // gate is disabled/uncalibrated for this corpus — mirrors the live MCP server's fail-closed
        // advertisement (the `exec` arm already withholds the diagnostic; this also stops the model
        // from being offered a tool call that can't do anything).
        // Advertise `read(page_image)` only for the api kinds that can actually FEED images back to
        // the model — i.e. the same gate the `VisionTransport` install uses below (OpenAiChat).
        // Non-OpenAiChat transports have no image side channel, so advertising the capability would
        // offer a field they silently ignore; fold the api gate into the advertised surface here so
        // advertise and feed stay in lockstep.
        let effective_vision = self.vision && matches!(self.api, crate::lab::ApiKind::OpenAiChat);
        let ctx = answer_tool_context(work, graph.is_some(), effective_vision);
        let tools = transport.tools_schema(&ctx);

        let trace = TraceLog::to_dir(work);
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
        let mut signals = if self.dedup {
            crate::backend::glossa_tools::ReaderSignals::new()
        } else {
            crate::backend::glossa_tools::ReaderSignals::disabled()
        };
        // Under `--vision`, the images each `read`/tool call surfaces are buffered here and drained
        // by `VisionTransport::push_tool_results` into a follow-up `role:"user"` image message right
        // after the round's tool results — the same seam the build/extract path uses via
        // `ClosureTransport`. The generic agent loop's `exec` is `(String, Vec<String>)` (image-
        // agnostic on purpose), so the images ride this side channel instead of the loop's return.
        // Off `--vision` the buffer is never drained (the wrapper isn't installed), so the transcript
        // is byte-identical to before.
        let pending_images: Rc<std::cell::RefCell<Vec<DocImage>>> =
            Rc::new(std::cell::RefCell::new(Vec::new()));
        let image_sink = Rc::clone(&pending_images);
        let exec = |name: &str, args: &Value| {
            let (mut body, ids, images) = execute_tool(name, args, work, idx, graph, &spec, &trace);
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
            // Buffer any images this call surfaced for the vision side channel (drained by
            // `VisionTransport` under `--vision`; left untouched otherwise). Empty for every
            // non-image call, so nothing accumulates on the text-only reader path.
            if !images.is_empty() {
                image_sink.borrow_mut().extend(images);
            }
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
                name, args, work, idx, graph, &spec, &trace,
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
        // Under `--vision`, wrap the reader's real transport so images the round's `exec` surfaced
        // (buffered in `pending_images`) ride to the model in a follow-up `role:"user"` image
        // message right after the tool results — matching a vision-enabled MCP server. Gated to the
        // OpenAI-compatible transport, since `vision_user_message` emits the OpenAI `image_url`
        // data-URI shape; every other api kind (and every non-vision run) drives the inner transport
        // directly, byte-identical to before.
        let vision_wrap;
        let loop_transport: &dyn ChatTransport =
            if self.vision && matches!(self.api, crate::lab::ApiKind::OpenAiChat) {
                vision_wrap = VisionTransport {
                    inner: transport.as_ref(),
                    pending_images: Rc::clone(&pending_images),
                };
                &vision_wrap
            } else {
                transport.as_ref()
            };
        let raw = crate::backend::agent_loop::run_agent_loop_capturing(
            loop_transport,
            &ep,
            None,
            seed_messages,
            Some(&tools),
            exec,
            nba,
            self.max_rounds,
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
            vision: false,
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
            shared: None,
            max_rounds: DEFAULT_MAX_ROUNDS,
            headers: std::collections::BTreeMap::new(),
            dedup: false,
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
            headers: self.headers.clone(),
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
///
/// Retained under `cfg(test)` only: every production one-off call now goes through
/// [`chat_once_resampled`] (raw `chat_once` had transport-level resilience but no degenerate
/// resample). This is kept solely to exercise the tools-free body-building / verbatim-endpoint-URL
/// path in the transport tests (`chat_once_posts_endpoint_url_verbatim`).
#[cfg(test)]
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
    let max_tokens: u64 = crate::backend::transport::agent_max_tokens();
    let mut body = json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    });
    // Include `temperature` only when set — `None` omits it so the provider default applies.
    if let Some(t) = temperature {
        body["temperature"] = json!(t);
    }
    crate::backend::transport::openai::chat_http_full(
        endpoint,
        api_key,
        &body,
        Duration::from_secs(timeout_secs),
        crate::backend::resilience::RetryPolicy::default(),
        // Test-only strict-provider one-shot helper: no `Endpoint` here, so no headers to resolve.
        &[],
    )
    .map(|v| {
        v.pointer("/choices/0/message")
            .cloned()
            .unwrap_or_else(|| json!({}))
    })
}

/// Single-shot model call with the SAME provider-neutral degenerate-resample as the reader's agent
/// loop (length cap / repetition loop / empty turn — see [`crate::backend::resample`]). Use instead
/// of raw [`chat_once`] for every one-off call (judge, reflect, user-sim, …) so a degenerate
/// completion is retried uniformly and WORD-AGNOSTICALLY (never keyed on the reply's content).
/// Returns the assistant message shaped like `chat_once` (a `{"role","content"}` object) so callers
/// reading `.get("content")` are unchanged. `system` travels inside `messages` (as messages[0]),
/// exactly as the raw `chat_once` sites already pass it.
pub(crate) fn chat_once_resampled(
    ep: &crate::lab::Endpoint,
    messages: &[Value],
) -> anyhow::Result<Value> {
    let transport = crate::backend::transport::transport_for(ep);
    let turn = crate::backend::resample::call_with_resample(
        transport.as_ref(),
        ep,
        None,
        messages,
        None,
        ep.resolve_temperature(),
        crate::backend::resample::ResamplePolicy::default(),
    )?;
    Ok(serde_json::json!({
        "role": "assistant",
        "content": turn.text.unwrap_or_default(),
    }))
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

/// Build the `ToolContext` an eval answering run advertises tools from — the prod Reader
/// deployment surface. `verify_available` is resolved from the corpus's `.glossa` gate config
/// (`enabled && is_calibrated`), matching the live MCP server's fail-closed advertisement;
/// `no_source_file` is always false (eval delivers provenance, never withholds the tool);
/// `no_image` is `!vision`; notebook/constraint features are off in the answering eval. Shared by
/// the reader (`backend::openai`) and GEPA graph reflection (`gepa_graph`) so both describe the
/// SAME tool set the reader will actually be offered.
pub(crate) fn answer_tool_context(
    work: &std::path::Path,
    graph_on: bool,
    vision: bool,
) -> glossa::tools::registry::ToolContext {
    use glossa::tools::registry::{FeatureSet, Tier, ToolContext};
    let glossa_dir = work.join(".glossa");
    let cfg = glossa::gate::VerifyConfig::resolve(&glossa_dir);
    ToolContext {
        profile: Tier::Reader,
        graph_on,
        verify_available: cfg.enabled && cfg.is_calibrated(),
        no_source_file: false,
        no_image: !vision,
        features: FeatureSet::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn shared_handle_is_reused_not_reopened() {
        let dir = tempfile::tempdir().unwrap();
        // One run-wide handle, as `run_eval` opens it before the worker pool.
        let h = std::sync::Arc::new(glossa::graph::handle::GraphHandle::open(dir.path()).unwrap());
        // Two per-case backends built from the SAME Arc (as the worker closure does per question).
        let a = OpenAiBackend {
            shared: Some(h.clone()),
            ..OpenAiBackend::for_test_with_prompt("x")
        };
        let b = OpenAiBackend {
            shared: Some(h.clone()),
            ..OpenAiBackend::for_test_with_prompt("x")
        };
        // Both reference the SAME handle instance — the graph/index/CSR is opened once for the whole
        // run, not re-opened per backend/question.
        assert!(std::sync::Arc::ptr_eq(
            a.shared.as_ref().unwrap(),
            b.shared.as_ref().unwrap()
        ));
        // With no shared handle the backend falls back to per-question open (today's behavior).
        assert!(OpenAiBackend::for_test_with_prompt("x").shared.is_none());
    }

    #[test]
    fn answer_tool_context_maps_vision_to_no_image() {
        let dir = tempfile::tempdir().unwrap();
        let ctx_on = answer_tool_context(dir.path(), true, true);
        let ctx_off = answer_tool_context(dir.path(), true, false);
        assert!(!ctx_on.no_image);
        assert!(ctx_off.no_image);
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
    // live in `transport::openai`'s test module — they exercise the moved `chat_http`/
    // `parse_tool_args` core directly at its new home.
    //
    // The agent-loop tests (`loop_*`) and the tool-schema tests (`grep_is_advertised_in_both_arms`,
    // `openai_tools_*`) moved with the shim to `backend::loop_compat`; the token-accounting tests to
    // `backend::accounting`; the ETA/format tests to `backend::progress`; the reader-dialogue test
    // to `backend::dialogue`; the vision-message tests to `backend::vision`.
}
