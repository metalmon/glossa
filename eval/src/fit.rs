//! The batch-size decision, as pure functions.
//!
//! Layer 2 of the compute budget (spec: `2026-09-30-compute-budget-design.md`): settle, sweep,
//! select. This module holds the part that needs no device — the rule that turns a set of
//! `(size, rows/s)` measurements into a configured budget, the doubling walk that produces the
//! sizes, and the criterion that decides when a device has stopped moving enough to time it.
//! Execution against a real engine lives beside it; the decision lives here so CI can check it.

/// One measured point: a batch size and the throughput observed at it.
#[derive(Debug, Clone)]
pub struct Sample {
    pub rows: usize,
    pub rows_per_sec: f64,
}

/// What the fit decided, and the evidence for it.
#[derive(Debug, Clone)]
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

/// One thing the fit needs of an engine: score a pool at an exact batch size, with the same
/// planning and padding the serving path uses. Deliberately not "give me a number" — timing
/// belongs to the sweep, so both engines and a fake share one measurement loop.
pub trait FitTarget {
    fn score_pool(&self, rows: usize) -> anyhow::Result<()>;
}

pub struct SweepOpts {
    /// Ceiling on the sweep. Load-bearing, not cosmetic: the sweep's own peak stays RESIDENT, so
    /// this is the operator's cap on what the measurement itself costs.
    pub max_rows: usize,
    pub seq: usize,
    pub repeats: usize,
    pub tolerance: f64,
}

/// Whether an engine error is the device declining to allocate, rather than a fault in the model or
/// the call. Matched on the message because neither ORT nor burn gives allocation failures a type
/// of their own, and the strings differ per provider: CUDA reports `CUDA failure 2: out of
/// memory`, ORT's own arena reports a failed allocation, DirectML surfaces `E_OUTOFMEMORY`.
///
/// Lives here rather than in `glossa-nli` so the sweep — and its tests — build with no engine
/// feature at all: `glossa-nli` is an optional dependency of this crate.
fn is_allocation_failure(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    [
        "out of memory",
        "outofmemory",
        "failed to allocate",
        "allocation failed",
        "memoryallocation",
        "not enough memory",
    ]
    .iter()
    .any(|needle| m.contains(needle))
}

/// Settle, then walk doubling sizes, returning one [`Sample`] per size the device accepted.
///
/// A size the device refuses to allocate ends the sweep with what already worked — an engine that
/// cannot batch still has to serve. Any other error is a real fault and propagates: retrying it at
/// a smaller size would turn a broken model into a slow one.
pub fn sweep(target: &dyn FitTarget, opts: &SweepOpts) -> anyhow::Result<Vec<Sample>> {
    // Settle first. Without this the sweep measures the device ramping up and reports a gradient
    // that is an artefact of its own ordering.
    let mut series = Vec::new();
    for _ in 0..12 {
        let t0 = std::time::Instant::now();
        target.score_pool(1)?;
        series.push(1.0 / t0.elapsed().as_secs_f64().max(f64::EPSILON));
        if settled(&series, opts.tolerance, 3) {
            break;
        }
    }

    let mut out = Vec::new();
    let mut rows = 1usize;
    loop {
        let mut best_for_size = 0.0f64;
        let mut refused = false;
        for _ in 0..opts.repeats.max(1) {
            let t0 = std::time::Instant::now();
            match target.score_pool(rows) {
                Ok(()) => {
                    let rps = rows as f64 / t0.elapsed().as_secs_f64().max(f64::EPSILON);
                    // Best of the repeats, not the mean: a slow reading is contamination (another
                    // process took the device), a fast one cannot be.
                    best_for_size = best_for_size.max(rps);
                }
                Err(e) if is_allocation_failure(&e.to_string()) => {
                    refused = true;
                    break;
                }
                Err(e) => return Err(e),
            }
        }
        if refused {
            break;
        }
        out.push(Sample {
            rows,
            rows_per_sec: best_for_size,
        });
        match next_size(rows, opts.max_rows) {
            Some(n) => rows = n,
            None => break,
        }
    }
    Ok(out)
}

/// The reranker as a fit target: a synthetic pool of `rows` passages at `seq` tokens, scored
/// through `rerank_with_budget` so the measured path is the serving path.
///
/// Gated on the execution-provider features, which are what make `glossa-nli` a dependency of this
/// crate at all (see `eval/Cargo.toml`).
#[cfg(any(
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
))]
pub struct RerankTarget<'a> {
    pub engine: &'a glossa_nli::InProcessReranker,
    pub seq: usize,
}

#[cfg(any(
    feature = "nli-directml",
    feature = "nli-coreml",
    feature = "nli-cuda",
    feature = "nli-rocm"
))]
impl FitTarget for RerankTarget<'_> {
    fn score_pool(&self, rows: usize) -> anyhow::Result<()> {
        // A filler word repeated to length: the fit measures shapes, not relevance, and generic
        // filler keeps corpus text out of a diagnostic.
        let passage = "text ".repeat(self.seq);
        let refs: Vec<&str> = (0..rows).map(|_| passage.as_str()).collect();
        self.engine
            .rerank_with_budget("query", &refs, rows * self.seq)
            .map(|_| ())
    }
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

    struct RefusesAbove {
        limit: usize,
        calls: std::cell::RefCell<Vec<usize>>,
    }
    impl FitTarget for RefusesAbove {
        fn score_pool(&self, rows: usize) -> anyhow::Result<()> {
            self.calls.borrow_mut().push(rows);
            if rows > self.limit {
                anyhow::bail!("CUDA failure 2: out of memory");
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
            Ok(())
        }
    }

    /// A device that cannot take a larger batch must end the sweep with what worked, not fail the
    /// command: an unprobe-able device still has to serve, at one row per batch.
    #[test]
    fn a_refused_size_ends_the_sweep_without_failing() {
        let t = RefusesAbove {
            limit: 4,
            calls: Default::default(),
        };
        let out = sweep(
            &t,
            &SweepOpts {
                max_rows: 64,
                seq: 512,
                repeats: 1,
                tolerance: 0.05,
            },
        )
        .expect("a refusal is not a command failure");
        assert!(
            out.iter().all(|s| s.rows <= 4),
            "no sample above the refusal: {out:?}"
        );
        assert!(
            out.iter().any(|s| s.rows == 4),
            "the last size that worked is kept"
        );
        assert!(
            t.calls.borrow().iter().any(|&r| r == 8),
            "the sweep has to TRY the next size to learn it is refused"
        );
    }

    /// An error that is NOT an allocation failure is a real fault and must propagate — retrying it
    /// at a smaller size would turn a broken model into a slow one.
    #[test]
    fn a_non_allocation_error_propagates() {
        struct Broken;
        impl FitTarget for Broken {
            fn score_pool(&self, _: usize) -> anyhow::Result<()> {
                anyhow::bail!("logits extraction: shape mismatch")
            }
        }
        assert!(sweep(
            &Broken,
            &SweepOpts {
                max_rows: 8,
                seq: 512,
                repeats: 1,
                tolerance: 0.05
            }
        )
        .is_err());
    }

    /// The refusal predicate is the whole difference between those two behaviours, and it reads
    /// provider prose. Strings are the real ones each provider emits.
    #[test]
    fn allocation_failures_are_told_apart_from_faults() {
        assert!(is_allocation_failure("CUDA failure 2: out of memory"));
        assert!(is_allocation_failure(
            "Failed to allocate memory for requested buffer of size 1342177280"
        ));
        assert!(is_allocation_failure(
            "DML allocator: E_OUTOFMEMORY (0x8007000E)"
        ));
        assert!(!is_allocation_failure("logits extraction: shape mismatch"));
        assert!(
            !is_allocation_failure("no room left in the output tensor"),
            "a substring match on `oom` would fire here; it must not"
        );
    }
}
