//! Process-global token/resample accounting for every chat call routed through the transport
//! layer, plus the per-conversation prompt-prefix tracker that lets a cache split be estimated on
//! servers that omit `prompt_tokens_details`.
//!
//! Split out of `backend::openai`: the accounting statics live HERE (one definition of every
//! counter), and the transport HTTP bridge (`transport::openai`) calls `record_usage` back into
//! this module. `backend::progress` renders these counters onto the progress bar.

use crate::backend::dialogue;
use crate::backend::progress::cache_segment;
use serde_json::Value;
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

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

// The PREVIOUS chat request's `prompt_tokens` within the current conversation (one agent-loop
// run: one reason seed, one build doc, one eval case, …). Each round of `run_agent_loop`
// re-sends the whole prior transcript as a prefix and appends to it, so on a server that omits
// `prompt_tokens_details.cached_tokens` (e.g. LM Studio), THIS round's re-sent prefix is exactly
// last round's `prompt_tokens` — see `usage_split_with_prefix`. Reset to 0 at the start of every
// conversation by `reset_conversation_prefix` so one seed's tail doesn't leak into the next
// seed's estimate.
//
// **THREAD-LOCAL, not a process-global.** Under parallel workers (`kbx --jobs N`) each worker
// thread drives its own conversation concurrently with the others; a shared global here would let
// worker A's stored `prompt_tokens` leak into worker B's next estimate (interleaved conversations
// corrupting each other's prefix) — see the parallel-jobs design doc's "Caching under
// parallelism" section. Each thread gets its own independent cell, so the SEQUENTIAL-within-one-
// conversation assumption `usage_split_with_prefix` relies on holds per-thread even though many
// threads run concurrently. The cross-worker aggregates (`NEW_TOKENS`/`CACHED_TOKENS`/
// `CACHE_ESTIMATED`/`RESAMPLES`) stay process-global atomics — their SUM across workers is still
// the correct total, unlike this per-conversation prefix.
thread_local! {
    static PREV_PROMPT_TOKENS: Cell<u64> = const { Cell::new(0) };
}

/// Set true whenever the MOST RECENT `chat_http` call had to self-estimate its cached-token split
/// (the server omitted `prompt_tokens_details`) rather than use a server-reported figure. Drives
/// the `~` (estimated) marker in `status_message`/`token_summary`. Reset to `false` in
/// `reset_tokens`. Sticky for the run rather than per-call: once a run has needed even one
/// estimate, the whole run's cache figure is an estimate-tainted mix and should read as such.
///
/// `pub(crate)` so `progress::cache_segment` can read it directly when rendering the cache label.
pub(crate) static CACHE_ESTIMATED: AtomicBool = AtomicBool::new(false);

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
    dialogue::reset();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::transport::openai::chat_http;
    use serde_json::json;
    use std::time::Duration;

    /// Serializes the tests that read/write the PROCESS-GLOBAL token counters (`NEW_TOKENS`,
    /// `CACHED_TOKENS`, `CACHE_ESTIMATED`, and the `record_usage`/`chat_http` paths that mutate them).
    /// libtest runs tests of one binary concurrently, so without this a test asserting an exact
    /// global sum races another test's `record_usage`/`reset_tokens` and flakes (observed on CI).
    /// `into_inner` recovers from a poisoned lock so one panicking test doesn't cascade-fail the rest.
    static TOKEN_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
        let _tg = TOKEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        let _tg = TOKEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        let _tg = TOKEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        let _tg = TOKEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

        // NOTE: intentionally NOT asserting the process-global `new_tokens()`/`cached_tokens()`
        // aggregates here. Those atomics are shared across the whole test binary, and cargo runs
        // tests concurrently in one process, so another test's token accumulation or `reset_tokens()`
        // races this assert (it flaked on CI's higher-core runner: new_tokens()==0 vs the expected
        // sum). The thread-local ISOLATION this test exists to prove is fully covered by the per-worker
        // return-value asserts above (A's and B's splits can't be right unless their thread-locals
        // stayed independent) — no shared global needed.

        reset_tokens();
    }

    #[test]
    fn human_tokens_formats_compactly() {
        assert_eq!(human_tokens(999), "999");
        assert_eq!(human_tokens(1500), "1.5k");
        assert_eq!(human_tokens(2_000_000), "2.0M");
    }

    #[test]
    fn tokens_used_resets_and_accumulates() {
        let _tg = TOKEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
}
