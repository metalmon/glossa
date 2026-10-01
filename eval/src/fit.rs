//! The batch-size decision, as pure functions.
//!
//! Layer 2 of the compute budget (spec: `2026-09-30-compute-budget-design.md`): settle, sweep,
//! select. This module holds the part that needs no device — the rule that turns a set of
//! `(size, rows/s)` measurements into a configured budget, the doubling walk that produces the
//! sizes, and the criterion that decides when a device has stopped moving enough to time it.
//! Execution against a real engine lives beside it; the decision lives here so CI can check it.

/// One measured point: a batch size and the throughput observed at it.
pub struct Sample {
    pub rows: usize,
    pub rows_per_sec: f64,
}

/// What the fit decided, and the evidence for it.
pub struct FitOutcome {
    /// The size to configure: the SMALLEST within `tolerance` of the best.
    pub chosen: usize,
    pub best: Sample,
    /// Throughput at `best` over throughput at one row — how much batching is worth here at all.
    pub gain_over_one: f64,
    /// Whether that gain is worth permanent VRAM. See [`PAYS_THRESHOLD`].
    pub pays: bool,
}

/// How close to the best a smaller size must be to win. 5%: wide enough to swallow the run-to-run
/// spread measured with the sweep protocol (0.8-6%), narrow enough that a real step is not thrown
/// away.
pub const DEFAULT_TOLERANCE: f64 = 0.05;

/// Below this total gain, batching is reported as not worth its memory. 1.25x: CUDA measured 1.22x
/// and costs ~300 MB per engine permanently, which is the case this threshold exists to reject;
/// DirectML measured 7.8x.
pub const PAYS_THRESHOLD: f64 = 1.25;

/// The size to configure, given the complete sample: the smallest size within `tolerance` of the
/// best throughput. `None` only for an empty sample.
pub fn select(samples: &[Sample], tolerance: f64) -> Option<FitOutcome> {
    let best = samples
        .iter()
        .max_by(|a, b| a.rows_per_sec.total_cmp(&b.rows_per_sec))?;
    let floor = best.rows_per_sec * (1.0 - tolerance);
    // Smallest size that clears the floor. `min_by_key` over rows, not a scan in sample order:
    // the caller may hand these over in any order, and a shuffled sweep does.
    let chosen = samples
        .iter()
        .filter(|s| s.rows_per_sec >= floor)
        .min_by_key(|s| s.rows)?;
    let at_one = samples
        .iter()
        .find(|s| s.rows == 1)
        .map(|s| s.rows_per_sec)
        .unwrap_or(best.rows_per_sec);
    let gain = if at_one > 0.0 {
        best.rows_per_sec / at_one
    } else {
        1.0
    };
    Some(FitOutcome {
        chosen: chosen.rows,
        best: Sample {
            rows: best.rows,
            rows_per_sec: best.rows_per_sec,
        },
        gain_over_one: gain,
        pays: gain >= PAYS_THRESHOLD,
    })
}

/// Doubling, capped. `None` ends the sweep.
pub fn next_size(current: usize, max_rows: usize) -> Option<usize> {
    let next = current.saturating_mul(2);
    (current < max_rows).then_some(next.min(max_rows))
}

/// Whether a reference-size series has stopped moving: the last `need` readings within `tolerance`
/// of their own median.
///
/// The device is slow cold and fast under sustained load (measured: 9.1 rows/s cold, 32.8 after
/// 45 s of load, 19.7 after 60 s idle, 33.2 under load again), and it recovers after idling — so
/// this is clock/power state, not allocator fragmentation. Timing before it settles is what made
/// an earlier ascending sweep produce a clean and entirely spurious gradient.
pub fn settled(series: &[f64], tolerance: f64, need: usize) -> bool {
    if series.len() < need || need == 0 {
        return false;
    }
    let tail = &series[series.len() - need..];
    let mut sorted: Vec<f64> = tail.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    median > 0.0
        && tail
            .iter()
            .all(|v| (v - median).abs() / median <= tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(rows: usize, rps: f64) -> Sample {
        Sample {
            rows,
            rows_per_sec: rps,
        }
    }

    /// The rule: the SMALLEST size within `tolerance` of the best, not the fastest size. Memory is
    /// permanent (an ORT arena never returns a batch's peak), so a size that is 3% faster for
    /// twice the resident VRAM is a bad trade made silently.
    #[test]
    fn picks_the_cheapest_size_within_tolerance_of_the_best() {
        // Measured on DirectML, settled: the knee is 8 and nothing smaller is close.
        let dml = [
            s(1, 4.2),
            s(2, 8.5),
            s(4, 14.6),
            s(8, 32.9),
            s(16, 25.9),
            s(32, 30.0),
        ];
        let out = select(&dml, DEFAULT_TOLERANCE).unwrap();
        assert_eq!(out.chosen, 8);
        assert_eq!(out.best.rows, 8);
        assert!(out.pays, "7.8x must read as paying");

        // Measured on CUDA, settled: the best is 16, but 4 is within 5% of it and costs far less.
        let cuda = [
            s(1, 42.3),
            s(2, 45.5),
            s(4, 49.4),
            s(8, 51.2),
            s(16, 51.8),
            s(32, 50.5),
        ];
        let out = select(&cuda, DEFAULT_TOLERANCE).unwrap();
        assert_eq!(out.chosen, 4, "cheapest near-equal size, not the fastest");
        assert!(!out.pays, "1.22x must NOT read as paying");
    }

    /// Both measured curves are non-monotonic. A greedy walk would stop at the first plateau and
    /// answer differently, which is why selection is a rule over the COMPLETE sample.
    #[test]
    fn survives_a_non_monotonic_curve() {
        // Measured: the NLI gate on CUDA is slower at 16 than at 8, then faster again at 32.
        let nli = [s(1, 123.0), s(8, 134.9), s(16, 129.2), s(32, 152.9)];
        let out = select(&nli, DEFAULT_TOLERANCE).unwrap();
        assert_eq!(
            out.chosen, 32,
            "the best is past a dip; the rule must still find it"
        );
    }

    #[test]
    fn a_single_sample_yields_itself_and_an_empty_one_yields_nothing() {
        assert_eq!(select(&[s(4, 10.0)], DEFAULT_TOLERANCE).unwrap().chosen, 4);
        assert!(select(&[], DEFAULT_TOLERANCE).is_none());
    }

    #[test]
    fn ties_go_to_the_smaller_size() {
        let out = select(&[s(4, 50.0), s(8, 50.0)], DEFAULT_TOLERANCE).unwrap();
        assert_eq!(out.chosen, 4);
    }

    #[test]
    fn doubling_stops_at_the_cap() {
        assert_eq!(next_size(1, 64), Some(2));
        assert_eq!(next_size(32, 64), Some(64));
        assert_eq!(next_size(64, 64), None, "at the cap the sweep is over");
        assert_eq!(next_size(48, 64), Some(64), "never overshoot the cap");
    }

    /// The whole reason a startup fit is allowed at all: it waits for the device to stop moving.
    #[test]
    fn settling_needs_a_quiet_tail_not_just_length() {
        assert!(!settled(&[9.1, 20.0, 33.0], 0.05, 3), "still ramping");
        assert!(
            settled(&[9.1, 20.0, 32.8, 33.0, 32.9], 0.05, 3),
            "the tail is quiet"
        );
        assert!(!settled(&[33.0], 0.05, 3), "too few readings to tell");
        assert!(!settled(&[], 0.05, 3));
    }
}
