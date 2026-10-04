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

/// Why a fit must not run here, if it must not. `None` ⇒ measure.
///
/// Two refusals, both of them cases where measuring would produce a true number about the wrong
/// thing: a remote scorer (the batching is the server's, and timing round-trips would describe the
/// network), and a device that did not actually bind (loading is fail-open, so a configured `cuda`
/// that fails to register runs on CPU — and a cross-encoder pool on CPU is a thermal hazard as well
/// as a wrong answer).
pub fn fit_refusal(scorer: Option<&str>, backend: Option<&str>, ep_bound: bool) -> Option<String> {
    if scorer == Some("http") {
        let backend = backend.unwrap_or("remote");
        return Some(format!(
            "scorer = \"http\" (backend {backend}): batching belongs to that server, so there is \
             nothing measurable from this side — a client-side batch_tokens is inert here. Tune it \
             where it lives (TEI: --max-batch-tokens; vLLM: --max-num-batched-tokens; llama.cpp: \
             -ub/-b)."
        ));
    }
    if !ep_bound {
        return Some(
            "no GPU execution provider is bound: the configured device did not register, and \
             loading is fail-open, so this session is on CPU. A fit here would time a CPU session \
             and report it as a GPU one — and a cross-encoder pool on CPU is a thermal hazard. Run \
             `check` to see which provider actually bound."
                .to_string(),
        );
    }
    None
}

/// The row length a fit may actually measure at: the request, capped by what the engine will put
/// through a session (`engine_max`). Above that the tokenizer truncates, so a sweep "at 1024" would
/// time 512-token rows and then recommend a budget twice what those rows need — honest about
/// nothing. `None` ⇒ the engine's maximum.
pub fn effective_seq(requested: Option<usize>, engine_max: usize) -> usize {
    requested.unwrap_or(engine_max).clamp(1, engine_max.max(1))
}

/// A usable tolerance. A negative one puts the floor ABOVE the best sample, leaves the filter empty
/// and makes `select` return `None` — which the report states as "the device refused every batch
/// size", a false claim about the hardware produced by a typo in a flag.
pub fn sanitize_tolerance(requested: f64) -> f64 {
    if requested.is_finite() {
        requested.clamp(0.0, 1.0)
    } else {
        DEFAULT_TOLERANCE
    }
}

/// The token budget that configures a chosen row count: the sweep decides in ROWS, while the knob an
/// operator writes (`batch_tokens`) is in tokens, and the bridge between them is the row length the
/// fit measured at. Stated in one place so the two units cannot drift apart.
pub fn budget_tokens(chosen_rows: usize, seq: usize) -> usize {
    chosen_rows.saturating_mul(seq).max(1)
}

/// The operator-facing report. Prints the recommendation for the engine that was fitted AND the
/// neighbour's configured budget, because two engines on one card cost the SUM of their high-water
/// marks and nobody should carry away one half of a sum.
///
/// `neighbour` is the other engine's name and its currently configured `batch_tokens` (`None` when
/// it is configured but at the default); `None` for the pair means no neighbour on this device, which
/// the report says out loud — a single-engine measurement is a different claim.
pub fn fit_report(
    engine: &str,
    seq: usize,
    outcome: Option<&FitOutcome>,
    neighbour: Option<(&str, Option<usize>)>,
    apply_hint: Option<&str>,
    applied: bool,
) -> String {
    let mut out = format!("{engine} fit, {seq}-token rows:\n");
    match outcome {
        None => {
            out.push_str(
                "  nothing measured — the device refused every batch size, down to a single row\n",
            );
            return out;
        }
        Some(o) => {
            let tokens = budget_tokens(o.chosen, seq);
            out.push_str(&format!(
                "  best         {} rows, {:.1} rows/s\n",
                o.best.rows, o.best.rows_per_sec
            ));
            out.push_str(&format!(
                "  recommended  {} rows => batch_tokens = {tokens}\n",
                o.chosen
            ));
            if o.pays {
                out.push_str(&format!(
                    "  gain         {:.2}x over one row per batch — batching pays here\n",
                    o.gain_over_one
                ));
            } else {
                out.push_str(&format!(
                    "  gain         {:.2}x over one row per batch — batching does NOT pay here, \
                     and the batch's peak stays resident for the life of the process\n",
                    o.gain_over_one
                ));
            }
        }
    }
    match neighbour {
        Some((name, Some(tokens))) => out.push_str(&format!(
            "  neighbour    {name}: batch_tokens = {tokens} — both are resident, so the device \
             pays the SUM\n"
        )),
        Some((name, None)) => out.push_str(&format!(
            "  neighbour    {name}: batch_tokens unset (one row per batch) — both are resident, so \
             the device pays the SUM\n"
        )),
        None => out.push_str(
            "  neighbour    none configured on this device — this is a single-engine measurement, \
             and a second engine would add its own resident peak\n",
        ),
    }
    // Whether the number was TAKEN is a different fact from what the number is, and the operator
    // needs both: a recommendation discarded as not worth the memory must not be followed by a line
    // implying it is now in force.
    if applied {
        out.push_str(
            "  applied      yes, for THIS process only — nothing written to disk
",
        );
    } else if let Some(hint) = apply_hint {
        out.push_str(&format!("  apply        {hint}\n"));
    }
    out
}

/// One thing the fit needs of an engine: score a pool of `pool_rows` rows in batches of
/// `batch_rows`, with the same planning and padding the serving path uses. Deliberately not "give me
/// a number" — timing belongs to the sweep, so both engines and a fake share one measurement loop.
///
/// Both counts are passed because the pool is held FIXED across the sweep while the batch size
/// varies: that is what makes the arms equal work and lets tokenization cancel out of the ratio.
pub trait FitTarget {
    fn score_pool(&self, batch_rows: usize, pool_rows: usize) -> anyhow::Result<()>;
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
    let lowered = msg.to_ascii_lowercase();
    // Word sequences, not substrings: providers spell these with spaces, underscores and case
    // (`CUDA_ERROR_OUT_OF_MEMORY`), and a plain `contains("out of memory")` also fires on innocent
    // prose like "the layout of memory", which would end a sweep that should have propagated.
    let words: Vec<&str> = lowered
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    const PHRASES: [&[&str]; 4] = [
        &["out", "of", "memory"],
        &["failed", "to", "allocate"],
        &["allocation", "failed"],
        &["not", "enough", "memory"],
    ];
    if PHRASES
        .iter()
        .any(|p| words.windows(p.len()).any(|w| w == *p))
    {
        return true;
    }
    // Single tokens: the HRESULT by name or by code, and the CUDA enums spelled as one word.
    words.iter().any(|w| {
        matches!(
            *w,
            "outofmemory" | "cudaerroroutofmemory" | "cudaerrormemoryallocation"
        ) || w.contains("8007000e")
    })
}

/// The middle reading of a set, which is what the protocol specifies per size. Not the best:
/// run-to-run spread is 0.8-5.1%, the same magnitude as [`DEFAULT_TOLERANCE`], so one lucky reading
/// at a small size would clear the floor and take the recommendation. Not the mean either — one
/// contaminated reading (another process took the device) should not move the answer.
fn median(readings: &mut [f64]) -> f64 {
    if readings.is_empty() {
        return 0.0;
    }
    readings.sort_by(f64::total_cmp);
    readings[readings.len() / 2]
}

/// Settle, then walk doubling batch sizes, returning one [`Sample`] per size the device accepted.
///
/// Every arm scores the SAME pool — `opts.max_rows` rows — and only the batch size changes, so each
/// arm does equal total work. Otherwise a larger batch would generate more load during its own
/// measurement and raise the device's clock for itself, and the per-arm tokenization cost (which
/// scales with the pool, not the batch) would not cancel in the 1-row ratio that decides whether
/// batching pays at all.
///
/// A size the device refuses to allocate ends the sweep with what already worked — an engine that
/// cannot batch still has to serve, including one that cannot manage a single row. Any other error
/// is a real fault and propagates: retrying it at a smaller size would turn a broken model into a
/// slow one.
pub fn sweep(target: &dyn FitTarget, opts: &SweepOpts) -> anyhow::Result<Vec<Sample>> {
    let pool = opts.max_rows.max(1);
    // Settle first. Without this the sweep measures the device ramping up and reports a gradient
    // that is an artefact of its own ordering.
    let mut series = Vec::new();
    for _ in 0..12 {
        let t0 = std::time::Instant::now();
        match target.score_pool(1, pool) {
            Ok(()) => series.push(pool as f64 / t0.elapsed().as_secs_f64().max(f64::EPSILON)),
            // A device that will not take one row at a time has nothing to measure, but it still has
            // to serve: end the sweep with no samples and let the report say so in words.
            Err(e) if is_allocation_failure(&e.to_string()) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        }
        if settled(&series, opts.tolerance, 3) {
            break;
        }
    }

    let mut out = Vec::new();
    let mut rows = 1usize;
    loop {
        let mut readings = Vec::with_capacity(opts.repeats.max(1));
        let mut refused = false;
        for _ in 0..opts.repeats.max(1) {
            let t0 = std::time::Instant::now();
            match target.score_pool(rows, pool) {
                Ok(()) => {
                    readings.push(pool as f64 / t0.elapsed().as_secs_f64().max(f64::EPSILON));
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
            rows_per_sec: median(&mut readings),
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
#[cfg(feature = "engine")]
pub struct RerankTarget<'a> {
    pub engine: &'a glossa_nli::Reranker,
    pub seq: usize,
}

/// The NLI gate as a fit target: one synthetic premise scored against `rows` hypotheses through
/// `entail_with_budget`, so the measured path is again the serving path.
///
/// The premise is sized just under `seq` so the harness keeps it in one window; if a tokenizer
/// stretches it past the window budget anyway, the harness splits it and every size in the sweep
/// pays the same multiple, so the shape of the curve — which is all selection reads — is unchanged.
/// What the number then means is hypotheses per second rather than rows per second.
#[cfg(feature = "engine")]
pub struct NliTarget<'a> {
    pub engine: &'a glossa_nli::Nli,
    pub seq: usize,
}

#[cfg(feature = "engine")]
impl FitTarget for NliTarget<'_> {
    fn score_pool(&self, batch_rows: usize, pool_rows: usize) -> anyhow::Result<()> {
        // Sized UNDER `seq` so the pair tokenizes to at most `seq` ids and the harness keeps the
        // premise in one window: a premise that overflows the window is split, and the batch would
        // then hold a different number of rows than the one being measured.
        let premise = "text ".repeat(self.seq.saturating_sub(24).max(1));
        let hypothesis = "text text text text";
        let hyps: Vec<&str> = (0..pool_rows.max(1)).map(|_| hypothesis).collect();
        self.engine
            .entail_with_budget(&premise, &hyps, budget_tokens(batch_rows, self.seq))
            .map(|_| ())
    }
}

#[cfg(feature = "engine")]
impl FitTarget for RerankTarget<'_> {
    fn score_pool(&self, batch_rows: usize, pool_rows: usize) -> anyhow::Result<()> {
        // A filler word repeated to length: the fit measures shapes, not relevance, and generic
        // filler keeps corpus text out of a diagnostic. Sized UNDER `seq` (query + pair overhead
        // included) so a row never exceeds the length the budget is computed from -- a longer row
        // silently fits fewer of them per batch than the size being measured.
        let passage = "text ".repeat(self.seq.saturating_sub(16).max(1));
        let refs: Vec<&str> = (0..pool_rows.max(1)).map(|_| passage.as_str()).collect();
        self.engine
            .rerank_with_budget("query", &refs, budget_tokens(batch_rows, self.seq))
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
        calls: std::cell::RefCell<Vec<(usize, usize)>>,
    }
    impl FitTarget for RefusesAbove {
        fn score_pool(&self, batch_rows: usize, pool_rows: usize) -> anyhow::Result<()> {
            self.calls.borrow_mut().push((batch_rows, pool_rows));
            if batch_rows > self.limit {
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
            t.calls.borrow().contains(&(8, 64)),
            "the sweep has to TRY the next size to learn it is refused"
        );
        assert!(
            t.calls.borrow().iter().all(|&(_, pool)| pool == 64),
            "every arm scores the SAME pool, so the arms are equal work"
        );
    }

    /// An error that is NOT an allocation failure is a real fault and must propagate — retrying it
    /// at a smaller size would turn a broken model into a slow one.
    #[test]
    fn a_non_allocation_error_propagates() {
        struct Broken;
        impl FitTarget for Broken {
            fn score_pool(&self, _: usize, _: usize) -> anyhow::Result<()> {
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

    /// Batching belongs to the server for every remote backend, so measuring round-trips from the
    /// client would be a true number about the wrong thing.
    #[test]
    fn fit_refuses_a_remote_scorer_naming_the_backend() {
        let why = fit_refusal(Some("http"), Some("vllm"), /*ep_bound=*/ true).unwrap();
        assert!(why.contains("vllm"), "{why}");
        assert!(why.contains("http"), "{why}");
    }

    /// Loading is fail-open: a configured `cuda` that cannot register runs on CPU. Measuring that
    /// would time a CPU session while reporting a GPU one — and on this machine a cross-encoder
    /// pool on CPU is a thermal hazard, not just a wrong number.
    #[test]
    fn fit_refuses_when_the_provider_did_not_bind() {
        assert!(fit_refusal(Some("in_process"), None, false).is_some());
        assert!(
            fit_refusal(Some("in_process"), None, true).is_none(),
            "bound GPU: measure"
        );
    }

    /// Two numbers, never one: the operator must not take away half of a sum.
    #[test]
    fn a_paying_fit_reports_the_recommendation_and_the_neighbour() {
        let dml = [s(1, 4.2), s(4, 14.6), s(8, 32.9), s(16, 25.9)];
        let out = select(&dml, DEFAULT_TOLERANCE).unwrap();
        let text = fit_report(
            "reranker",
            512,
            Some(&out),
            Some(("nli gate", Some(1024))),
            Some("kbx rerank set --batch-tokens 4096"),
            false,
        );
        assert!(
            text.contains("recommended  8 rows => batch_tokens = 4096"),
            "{text}"
        );
        assert!(text.contains("batching pays here"), "{text}");
        assert!(text.contains("nli gate: batch_tokens = 1024"), "{text}");
        assert!(
            text.contains("kbx rerank set --batch-tokens 4096"),
            "{text}"
        );
    }

    /// The honest negative outcome: a number an operator can act on either way.
    #[test]
    fn a_marginal_gain_is_reported_as_not_paying() {
        let cuda = [s(1, 42.3), s(4, 49.4), s(16, 51.8)];
        let out = select(&cuda, DEFAULT_TOLERANCE).unwrap();
        let text = fit_report("reranker", 512, Some(&out), None, None, false);
        assert!(text.contains("does NOT pay here"), "{text}");
        assert!(
            text.contains("single-engine measurement"),
            "no neighbour has to be said out loud: {text}"
        );
    }

    /// A device that refuses even one row still has to leave the operator with a sentence.
    #[test]
    fn nothing_measured_is_said_in_words() {
        let text = fit_report("nli gate", 512, None, Some(("reranker", None)), None, false);
        assert!(text.contains("nothing measured"), "{text}");
    }

    #[test]
    fn rows_become_tokens_through_the_measured_row_length() {
        assert_eq!(budget_tokens(8, 512), 4096);
        assert_eq!(budget_tokens(0, 512), 1, "never configure a zero budget");
    }

    /// A device that will not take even ONE row must still leave a serving process behind: the sweep
    /// ends with no samples and the report says so, instead of the command failing. Before this the
    /// settle loop propagated the refusal, and the "nothing measured" line was all but unreachable.
    #[test]
    fn a_device_that_refuses_a_single_row_yields_no_samples_not_an_error() {
        let t = RefusesAbove {
            limit: 0,
            calls: Default::default(),
        };
        let out = sweep(
            &t,
            &SweepOpts {
                max_rows: 8,
                seq: 512,
                repeats: 1,
                tolerance: 0.05,
            },
        )
        .expect("a device that cannot batch at all is not a command failure");
        assert!(out.is_empty(), "{out:?}");
        assert!(
            select(&out, DEFAULT_TOLERANCE).is_none(),
            "and selection has nothing to choose from"
        );
    }

    /// Per size the protocol takes the MIDDLE reading. Best-of-repeats let one lucky reading at a
    /// small size clear the tolerance floor and take the recommendation, and the run-to-run spread
    /// (0.8-5.1%) is the same magnitude as the tolerance itself.
    #[test]
    fn a_size_is_scored_by_its_median_reading_not_its_best() {
        assert_eq!(median(&mut [10.0, 50.0, 12.0]), 12.0);
        assert_eq!(median(&mut [10.0]), 10.0);
        assert_eq!(median(&mut []), 0.0, "no readings is not a panic");
    }

    /// Measuring above the length the engine will actually run is measuring something else: the
    /// tokenizer truncates, and the recommended budget would then be a multiple of what rows need.
    #[test]
    fn the_measured_row_length_is_capped_by_the_engine() {
        assert_eq!(effective_seq(None, 512), 512, "default is the engine max");
        assert_eq!(effective_seq(Some(256), 512), 256, "shorter rows are legal");
        assert_eq!(effective_seq(Some(1024), 512), 512, "longer ones are not");
        assert_eq!(effective_seq(Some(0), 512), 1);
    }

    /// A typo in `--tolerance` must not become a false claim about the hardware: a negative
    /// tolerance puts the floor above every sample, which reads as "the device refused everything".
    #[test]
    fn a_nonsense_tolerance_cannot_become_a_claim_about_the_device() {
        assert_eq!(sanitize_tolerance(-0.1), 0.0);
        assert_eq!(sanitize_tolerance(2.0), 1.0);
        assert_eq!(sanitize_tolerance(f64::NAN), DEFAULT_TOLERANCE);
        assert_eq!(sanitize_tolerance(0.05), 0.05);
        assert!(
            select(&[s(1, 10.0), s(2, 20.0)], sanitize_tolerance(-0.1)).is_some(),
            "a sanitised tolerance always selects something from a non-empty sample"
        );
    }

    /// The report has to say whether the number was TAKEN, not only what it is: a recommendation
    /// discarded as not worth the memory must not be followed by a line implying otherwise.
    #[test]
    fn the_report_says_whether_the_budget_was_applied() {
        let out = select(&[s(1, 4.2), s(8, 32.9)], DEFAULT_TOLERANCE).unwrap();
        let applied = fit_report(
            "reranker",
            512,
            Some(&out),
            None,
            Some("kbi --rerank-batch-tokens 4096"),
            true,
        );
        assert!(applied.contains("applied      yes"), "{applied}");
        assert!(
            !applied.contains("apply        kbi"),
            "an applied budget must not also be advertised as a next step: {applied}"
        );

        let printed_only = fit_report(
            "reranker",
            512,
            Some(&out),
            None,
            Some("kbx rerank set --batch-tokens 4096"),
            false,
        );
        assert!(
            printed_only.contains("apply        kbx rerank set"),
            "{printed_only}"
        );
        assert!(!printed_only.contains("applied      yes"), "{printed_only}");
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
        assert!(
            is_allocation_failure("CUDA_ERROR_OUT_OF_MEMORY"),
            "the driver API spells it with underscores"
        );
        assert!(
            is_allocation_failure("HRESULT 0x8007000E"),
            "DirectML sometimes surfaces only the code"
        );
        assert!(!is_allocation_failure("logits extraction: shape mismatch"));
        assert!(
            !is_allocation_failure("the layout of memory for this tensor is unsupported"),
            "a substring match on `out of memory` fires here; word matching must not"
        );
        assert!(
            !is_allocation_failure("no room left in the output tensor"),
            "a substring match on `oom` would fire here; it must not"
        );
    }
}
