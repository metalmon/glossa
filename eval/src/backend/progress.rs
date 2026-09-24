//! Progress-bar helpers: the background [`StatusTicker`] and the pure ETA / status-message
//! formatters it renders. Split out of `backend::openai`; these read the token/resample counters
//! from `backend::accounting`.

use crate::backend::accounting::{
    cached_tokens, human_tokens, new_tokens, resamples, CACHE_ESTIMATED,
};
use indicatif::ProgressBar;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

/// Render the cache segment's label — `"{cached} cache~"` when this run's cache figure includes a
/// self-computed estimate (`CACHE_ESTIMATED`, see `usage_split_with_prefix`), else the plain
/// `"{cached} cache"` for a server-reported split. The trailing `~` is the sole marker distinguishing
/// an estimated figure from a real one, reused identically by `status_message` (live bar) and
/// `token_summary` (final line).
pub(crate) fn cache_segment() -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
