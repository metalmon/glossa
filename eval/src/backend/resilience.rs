//! Rate-limit resilience layer (Phase 3 of the multi-api-transport plan): per-endpoint configurable
//! retry/backoff, a per-endpoint throttle (requests-per-minute min-interval + in-flight cap), and a
//! flat fallback chain. Every piece is OPT-IN: with `rate_limit`/`fallback` ABSENT from an
//! [`Endpoint`], behavior is byte-identical to before this module existed —
//! [`RetryPolicy::from_rate_limit(None)`] yields the historical `4 attempts / 400ms*attempt` linear
//! backoff, [`call_resilient`] never touches the throttle (it only throttles a link whose
//! `rate_limit` is `Some`) and, with an empty `fallback`, makes exactly one call through the primary.
//!
//! DETERMINISM: the throttle's timing is driven through an injectable [`Clock`] so the backoff /
//! min-interval math is unit-tested with a fake clock+sleep — the tests here NEVER call real
//! `Instant::now`/`std::thread::sleep`/`rand`. The process-wide [`throttle`] entry point binds the
//! real clock; tests exercise [`MinInterval`]/[`RetryPolicy`] directly.

use crate::backend::transport::TurnReply;
use crate::lab::{Endpoint, RateLimit};
use std::collections::HashMap;
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Configurable retry/backoff policy, derived from an [`Endpoint`]'s [`RateLimit`]. The `Default`
/// (and [`from_rate_limit(None)`](RetryPolicy::from_rate_limit)) reproduces the historical hardcoded
/// loop EXACTLY: `attempts = 4`, `backoff_ms = 400`, so attempt `n` (1-based) sleeps `400 * n` ms and
/// the loop runs attempts `1..=4`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Total attempts (the transport loops `1..=attempts`). Historical default: 4.
    pub attempts: u32,
    /// Linear backoff base in ms: after a failed attempt `n` the loop sleeps `backoff_ms * n`.
    /// Historical default: 400.
    pub backoff_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        // The exact constants the transports hardcoded before Phase 3 (`for attempt in 1..=4`,
        // `sleep(400 * attempt)`). Preserving them here is what makes the absent-config path
        // byte-identical to today.
        Self {
            attempts: 4,
            backoff_ms: 400,
        }
    }
}

impl RetryPolicy {
    /// Build the policy for an endpoint's optional [`RateLimit`]: unset knobs fall back to the
    /// historical defaults, so `from_rate_limit(None)` == [`RetryPolicy::default`]. `attempts` is
    /// clamped to at least 1 (a `retry = 0` in the config would otherwise make the loop a no-op).
    pub fn from_rate_limit(rl: Option<&RateLimit>) -> Self {
        let d = Self::default();
        match rl {
            None => d,
            Some(r) => Self {
                attempts: r.retry.unwrap_or(d.attempts).max(1),
                backoff_ms: r.backoff_ms.unwrap_or(d.backoff_ms),
            },
        }
    }

    /// Backoff sleep after a failed attempt `n` (1-based): `backoff_ms * n` — linear, matching the
    /// historical `400 * attempt`.
    pub fn backoff(&self, attempt: u32) -> Duration {
        Duration::from_millis(self.backoff_ms.saturating_mul(u64::from(attempt)))
    }
}

/// Injectable time source, so throttle timing is deterministic in tests (a fake clock advances only
/// when `sleep` is called). `now_ms` is monotonic milliseconds from an arbitrary epoch — only
/// differences matter.
pub trait Clock {
    fn now_ms(&self) -> u64;
    fn sleep(&self, dur: Duration);
}

/// Real clock backing the process-wide [`throttle`]: monotonic `Instant` since first use + real
/// `thread::sleep`. Never used by the deterministic tests.
struct RealClock;

impl Clock for RealClock {
    fn now_ms(&self) -> u64 {
        static BASE: OnceLock<Instant> = OnceLock::new();
        BASE.get_or_init(Instant::now).elapsed().as_millis() as u64
    }
    fn sleep(&self, dur: Duration) {
        std::thread::sleep(dur);
    }
}

/// Min-interval rate limiter: enforces at least `min_interval_ms` between successive request STARTS
/// for one endpoint (a requests-per-minute cap of `60_000 / rpm`). Pure timing state so it is
/// unit-tested against a fake [`Clock`]; the process-wide [`throttle`] holds one behind a mutex per
/// endpoint URL.
#[derive(Debug)]
pub struct MinInterval {
    min_interval_ms: u64,
    /// Earliest `now_ms` at which the next request may start (0 until the first acquire).
    next_allowed_ms: u64,
}

impl MinInterval {
    /// `rpm` requests per minute -> `60_000 / rpm` ms between starts. `rpm == 0` disables the gate.
    pub fn from_rpm(rpm: u32) -> Self {
        let min_interval_ms = if rpm == 0 { 0 } else { 60_000 / u64::from(rpm) };
        Self {
            min_interval_ms,
            next_allowed_ms: 0,
        }
    }

    /// Acquire the next slot: if the clock is earlier than `next_allowed_ms`, sleep the difference;
    /// then reserve the following interval. Returns the ms actually slept (0 when no wait was
    /// needed) — handy for assertions.
    pub fn acquire(&mut self, clock: &dyn Clock) -> u64 {
        if self.min_interval_ms == 0 {
            return 0;
        }
        let now = clock.now_ms();
        let start = now.max(self.next_allowed_ms);
        let wait = start - now;
        if wait > 0 {
            clock.sleep(Duration::from_millis(wait));
        }
        self.next_allowed_ms = start + self.min_interval_ms;
        wait
    }
}

/// Blocking in-flight semaphore for one endpoint: `acquire` waits until fewer than `max` requests
/// are outstanding, then returns an RAII [`InflightGuard`] that decrements on drop.
#[derive(Debug)]
struct Inflight {
    max: u32,
    state: Mutex<u32>,
    cv: Condvar,
}

impl Inflight {
    fn new(max: u32) -> Self {
        Self {
            max,
            state: Mutex::new(0),
            cv: Condvar::new(),
        }
    }
    fn acquire<'a>(&'a self) -> InflightGuard<'a> {
        let mut n = self.state.lock().unwrap();
        while *n >= self.max {
            n = self.cv.wait(n).unwrap();
        }
        *n += 1;
        InflightGuard { owner: self }
    }
}

/// RAII slot from [`Inflight::acquire`] / [`throttle`]: releases the in-flight count on drop.
pub struct InflightGuard<'a> {
    owner: &'a Inflight,
}

impl Drop for InflightGuard<'_> {
    fn drop(&mut self) {
        let mut n = self.owner.state.lock().unwrap();
        *n = n.saturating_sub(1);
        self.owner.cv.notify_one();
    }
}

/// One endpoint's throttle state (min-interval + in-flight cap), owned by the process-wide registry.
struct EndpointThrottle {
    interval: Mutex<MinInterval>,
    inflight: Option<Inflight>,
}

/// The throttle guard [`throttle`] returns: holds the in-flight slot (if any) for the duration of
/// the caller's request. The min-interval wait has already completed by the time this is returned.
/// `'static` because the registry entries live for the process lifetime.
pub struct ThrottleGuard {
    _inflight: Option<InflightGuard<'static>>,
}

fn registry() -> &'static Mutex<HashMap<String, &'static EndpointThrottle>> {
    static REG: OnceLock<Mutex<HashMap<String, &'static EndpointThrottle>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Acquire a throttle slot for `ep_key` (the endpoint URL) under its [`RateLimit`]: blocks to
/// respect `rpm` (min-interval between starts) and `max_inflight` (concurrent cap), then returns a
/// guard that releases the in-flight slot on drop. Only ever called for a link whose `rate_limit`
/// is `Some`, so the absent-config path never reaches here. Uses the real clock; the deterministic
/// timing tests drive [`MinInterval`] directly with a fake clock.
pub fn throttle(ep_key: &str, rl: &RateLimit) -> ThrottleGuard {
    // Per-endpoint state is created once and leaked into a `'static` so guards can borrow it without
    // lifetime plumbing through the call graph. There are only a handful of distinct endpoint URLs
    // in a run, so this is a bounded one-time leak, not a per-call allocation.
    let entry: &'static EndpointThrottle = {
        let mut reg = registry().lock().unwrap();
        if let Some(e) = reg.get(ep_key) {
            e
        } else {
            let leaked: &'static EndpointThrottle = Box::leak(Box::new(EndpointThrottle {
                interval: Mutex::new(MinInterval::from_rpm(rl.rpm.unwrap_or(0))),
                inflight: rl.max_inflight.filter(|m| *m > 0).map(Inflight::new),
            }));
            reg.insert(ep_key.to_string(), leaked);
            leaked
        }
    };

    // Respect the min-interval first (may sleep), then take an in-flight slot held across the call.
    {
        let mut mi = entry.interval.lock().unwrap();
        mi.acquire(&RealClock);
    }
    ThrottleGuard {
        _inflight: entry.inflight.as_ref().map(Inflight::acquire),
    }
}

/// Try `make_call` against the primary `chain` endpoint and, on a HARD failure (any `Err` — the
/// transport has already exhausted its own retries / classified 5xx / timeout / connect as fatal),
/// advance through `chain.fallback` in order, each link with its OWN api/rate_limit. Returns the
/// first success; if every link fails, returns the LAST error. A primary success never touches the
/// fallback. Each link whose `rate_limit` is `Some` is throttled (held across its call) — a link
/// with no `rate_limit` is called directly, so the absent-config path is a single un-throttled call.
///
/// FLAT chain: only `chain`'s own `fallback` is honored — a fallback's own `fallback` is IGNORED
/// (never recursed) to bound the chain.
///
/// `make_call` receives the exact `&Endpoint` for the link (the same reference as `chain` for the
/// primary, so a caller may `std::ptr::eq(link, chain)` to reuse a pre-built primary transport and
/// only construct a transport for fallback links).
pub fn call_resilient<F>(chain: &Endpoint, mut make_call: F) -> anyhow::Result<TurnReply>
where
    F: FnMut(&Endpoint) -> anyhow::Result<TurnReply>,
{
    let mut last_err: Option<anyhow::Error> = None;
    for link in std::iter::once(chain).chain(chain.fallback.iter()) {
        // Throttle only a link that opted in; the guard (if any) is held across the call and drops
        // right after, releasing the in-flight slot.
        let _guard = link
            .rate_limit
            .as_ref()
            .map(|rl| throttle(&link.endpoint, rl));
        match make_call(link) {
            Ok(reply) => return Ok(reply),
            Err(e) => last_err = Some(e),
        }
    }
    // With an empty fallback this is just the primary's own error, preserving today's surface.
    Err(last_err.expect("call_resilient always attempts at least the primary link"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::cell::{Cell, RefCell};

    fn ep(endpoint: &str) -> Endpoint {
        Endpoint {
            endpoint: endpoint.to_string(),
            model: "m".into(),
            api_key: String::new(),
            api_key_env: String::new(),
            timeout_secs: 1,
            api: crate::lab::ApiKind::default(),
            temperature: None,
            rate_limit: None,
            fallback: Vec::new(),
            function_name: None,
            feedback_score_metric: None,
            feedback_bool_metric: None,
        }
    }

    fn reply(text: &str) -> TurnReply {
        TurnReply {
            text: Some(text.to_string()),
            tool_calls: vec![],
            finish_reason: None,
            raw: json!({}),
        }
    }

    /// Fake clock: `now` only advances when `sleep` is called, and every slept duration is recorded.
    /// This is what keeps the throttle-timing tests deterministic (no real `Instant`/`thread::sleep`).
    struct FakeClock {
        now: Cell<u64>,
        slept: RefCell<Vec<u64>>,
    }
    impl FakeClock {
        fn new() -> Self {
            Self {
                now: Cell::new(0),
                slept: RefCell::new(vec![]),
            }
        }
    }
    impl Clock for FakeClock {
        fn now_ms(&self) -> u64 {
            self.now.get()
        }
        fn sleep(&self, dur: Duration) {
            let ms = dur.as_millis() as u64;
            self.slept.borrow_mut().push(ms);
            self.now.set(self.now.get() + ms);
        }
    }

    #[test]
    fn retry_policy_default_is_todays_constants() {
        let d = RetryPolicy::default();
        assert_eq!(d.attempts, 4);
        assert_eq!(d.backoff_ms, 400);
        // Absent rate_limit == the historical default, exactly.
        assert_eq!(RetryPolicy::from_rate_limit(None), d);
        // The historical 400 * attempt sequence over attempts 1..=4.
        let steps: Vec<u64> = (1..=4).map(|n| d.backoff(n).as_millis() as u64).collect();
        assert_eq!(steps, vec![400, 800, 1200, 1600]);
    }

    #[test]
    fn retry_policy_reads_config_and_clamps_zero() {
        let rl = RateLimit {
            rpm: None,
            max_inflight: None,
            retry: Some(3),
            backoff_ms: Some(500),
        };
        let p = RetryPolicy::from_rate_limit(Some(&rl));
        assert_eq!(p.attempts, 3);
        let steps: Vec<u64> = (1..=3).map(|n| p.backoff(n).as_millis() as u64).collect();
        assert_eq!(steps, vec![500, 1000, 1500]);
        // retry = 0 is clamped to at least one attempt so the loop is never a no-op.
        let z = RateLimit {
            rpm: None,
            max_inflight: None,
            retry: Some(0),
            backoff_ms: None,
        };
        assert_eq!(RetryPolicy::from_rate_limit(Some(&z)).attempts, 1);
        // Unset knobs keep historical defaults even when the table is present.
        assert_eq!(RetryPolicy::from_rate_limit(Some(&z)).backoff_ms, 400);
    }

    #[test]
    fn min_interval_sleeps_one_period_between_two_calls() {
        // rpm = 60 -> 1000ms min interval. Two back-to-back acquires at t=0: the first waits 0, the
        // second sleeps exactly one interval — asserted deterministically via the fake clock.
        let clock = FakeClock::new();
        let mut mi = MinInterval::from_rpm(60);
        let w1 = mi.acquire(&clock);
        let w2 = mi.acquire(&clock);
        assert_eq!(w1, 0, "first request should not wait");
        assert_eq!(w2, 1000, "second request should wait one 1000ms interval");
        assert_eq!(
            *clock.slept.borrow(),
            vec![1000],
            "only the second call sleeps"
        );
    }

    #[test]
    fn min_interval_zero_rpm_never_waits() {
        let clock = FakeClock::new();
        let mut mi = MinInterval::from_rpm(0);
        assert_eq!(mi.acquire(&clock), 0);
        assert_eq!(mi.acquire(&clock), 0);
        assert!(clock.slept.borrow().is_empty());
    }

    #[test]
    fn call_resilient_primary_success_never_touches_fallback() {
        let mut primary = ep("primary");
        primary.fallback = vec![ep("fb")];
        let calls = RefCell::new(vec![]);
        let out = call_resilient(&primary, |link| {
            calls.borrow_mut().push(link.endpoint.clone());
            Ok(reply("ok"))
        })
        .unwrap();
        assert_eq!(out.text.as_deref(), Some("ok"));
        assert_eq!(
            *calls.borrow(),
            vec!["primary".to_string()],
            "fallback must not be called"
        );
    }

    #[test]
    fn call_resilient_falls_through_to_first_working_fallback() {
        let mut primary = ep("primary");
        primary.fallback = vec![ep("fb1"), ep("fb2")];
        let calls = RefCell::new(vec![]);
        let out = call_resilient(&primary, |link| {
            calls.borrow_mut().push(link.endpoint.clone());
            if link.endpoint == "fb1" {
                Ok(reply("from-fb1"))
            } else {
                Err(anyhow::anyhow!("hard fail on {}", link.endpoint))
            }
        })
        .unwrap();
        assert_eq!(out.text.as_deref(), Some("from-fb1"));
        // Tried primary then fb1, stopped before fb2.
        assert_eq!(
            *calls.borrow(),
            vec!["primary".to_string(), "fb1".to_string()]
        );
    }

    #[test]
    fn call_resilient_all_fail_returns_last_error() {
        let mut primary = ep("primary");
        primary.fallback = vec![ep("fb1"), ep("fb2")];
        let err = call_resilient(&primary, |link| {
            Err::<TurnReply, _>(anyhow::anyhow!("fail-{}", link.endpoint))
        })
        .unwrap_err();
        assert!(
            err.to_string().contains("fb2"),
            "should surface the LAST link's error: {err}"
        );
    }

    #[test]
    fn call_resilient_no_fallback_is_single_primary_call() {
        // The absent-config shape: empty fallback, no rate_limit -> exactly one make_call, the
        // primary's own error surfaced verbatim (byte-identical to a bare transport call).
        let primary = ep("primary");
        let n = Cell::new(0);
        let err = call_resilient(&primary, |_| {
            n.set(n.get() + 1);
            Err::<TurnReply, _>(anyhow::anyhow!("boom"))
        })
        .unwrap_err();
        assert_eq!(n.get(), 1);
        assert_eq!(err.to_string(), "boom");
    }

    #[test]
    fn throttle_inflight_cap_blocks_until_slot_frees() {
        // Light functional check of the in-flight semaphore (the timing-free part): with max=1, a
        // second acquire from another thread only proceeds after the first guard drops.
        let rl = RateLimit {
            rpm: None,
            max_inflight: Some(1),
            retry: None,
            backoff_ms: None,
        };
        let key = "test://inflight-cap";
        let g1 = throttle(key, &rl);
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let done2 = std::sync::Arc::clone(&done);
        let t = std::thread::spawn(move || {
            let _g2 = throttle(
                "test://inflight-cap",
                &RateLimit {
                    rpm: None,
                    max_inflight: Some(1),
                    retry: None,
                    backoff_ms: None,
                },
            );
            done2.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        // The spawned acquire must be blocked while g1 is held.
        std::thread::yield_now();
        assert!(!done.load(std::sync::atomic::Ordering::SeqCst));
        drop(g1);
        t.join().unwrap();
        assert!(done.load(std::sync::atomic::Ordering::SeqCst));
    }
}
