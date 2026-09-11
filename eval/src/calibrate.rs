//! `kbx eval calibrate` — sweep the verify-gate threshold (`glossa::gate`) over a past `kbx eval
//! run`'s graded cases and set the corpus's `ontology.toml` `[verify.threshold]` from real data
//! instead of a guess. It loads the run's cases, scores each with the model-free grounding gate,
//! sweeps the serve/abstain threshold per bucket (single vs multi cited chunks) under an optional
//! error cap, writes `calibration.svg` / `calibration.json`, prints a short operating-point
//! summary, and — with `--write` — persists the chosen thresholds to `ontology.toml`.

use crate::judge::Verdict;
use crate::report::CaseResult;
use anyhow::Context;
use glossa::gate::{score, Bucket, VerifyConfig};
use indicatif::{ProgressBar, ProgressStyle};
use std::io::IsTerminal;
use std::time::Duration;

/// A TTY-gated progress bar over the per-case scoring pass; hidden when stderr is not a terminal
/// (CI, redirected output) so logs stay clean. Same style as the `build` / `reason` bars.
fn mk_bar(len: u64) -> ProgressBar {
    if !std::io::stderr().is_terminal() {
        return ProgressBar::hidden();
    }
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.white} {prefix} [{pos}/{len}] {wide_bar:.white} {elapsed_precise}{msg}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    pb.enable_steady_tick(Duration::from_millis(90));
    pb
}

/// Cross-validation fold-threshold spread, appended to a bucket summary as an honesty note: a wide
/// spread means the single written threshold is unstable at this sample size. Empty when fewer than
/// two folds produced a threshold to compare.
fn cv_spread(folds: &[Option<f32>]) -> String {
    let vals: Vec<f32> = folds.iter().filter_map(|t| *t).collect();
    if vals.len() < 2 {
        return String::new();
    }
    let lo = vals.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    format!("   [CV folds: {lo:.2}–{hi:.2}]")
}

/// `kbx eval calibrate` flags. No `path`: the corpus + `runs/` dir come from kb-style PATH
/// resolution off the current directory (see `crate::workspace::resolve`), matching every other
/// `kbx` subcommand's default.
#[derive(clap::Args, Debug)]
pub struct CalibrateArgs {
    /// Run tag to calibrate on (a directory under `runs/`). Omit to use the most recent run.
    #[arg(long)]
    pub run: Option<String>,
    /// Which cited-chunk bucket(s) to sweep: `single`, `multi`, or `both`.
    #[arg(long, default_value = "both")]
    pub bucket: String,
    /// Error cap: pick the highest-coverage threshold whose served answers are at most this
    /// fraction wrong (e.g. `0.2` = at most 20% wrong). Omit for no cap (maximise coverage).
    #[arg(long)]
    pub max_error: Option<f32>,
    /// Cross-validation folds used to report threshold stability (a diagnostic spread only; the
    /// written threshold is fit on all cases).
    #[arg(long, default_value_t = 5)]
    pub folds: usize,
    /// Persist the chosen threshold(s) to the corpus `ontology.toml`. Without it the run is a
    /// dry-run — it still writes calibration.svg / calibration.json but changes no config.
    #[arg(long)]
    pub write: bool,
    /// Directory for calibration.svg / calibration.json. Defaults to the run dir
    /// (`.glossa/kbx/runs/<tag>/`); never the indexed corpus.
    #[arg(long)]
    pub out: Option<std::path::PathBuf>,
}

/// A run's confusion cell for calibration purposes — reuses `report.rs`'s `(answerable, verdict)`
/// classification (see `classify`) rather than inventing a parallel one. `Skip` drops a case out of
/// the calibration binary entirely (a safe-miss Partial under `credit_abstention`, or Unscored).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Cell {
    ShouldServe,
    ShouldAbstain,
    Skip,
}

/// One graded, chunk-resolved case ready for the threshold sweep. `Copy` so `run` can filter the
/// whole-pool `Vec<Case>` into a per-bucket `Vec<Case>` (needed by [`sweep`]/[`sweep_cv`], which
/// take an owned slice) without a manual field-by-field rebuild.
#[derive(Clone, Copy)]
pub struct Case {
    pub grounding: f32,
    pub bucket: Bucket,
    pub cell: Cell,
    /// NLI entailment shadow score, when a scorer ran. Plan 1 has no NLI scorer wired into
    /// [`load_cases`] — it always sets `None` here; this is the slot Plan 2 fills. Unit tests
    /// construct `Case`s with `Some(_)` directly to exercise [`select_mode`]'s `nli` branch.
    pub nli: Option<f32>,
}

/// Reuse `report.rs`'s `confusion_text` cell classification under abstention_policy
/// `credit_abstention`. `classify` is only ever called on a case that already has a non-empty
/// `final_answer` (the reader served SOMETHING) — so unlike `confusion_text`, an unanswerable
/// question lands `ShouldAbstain` regardless of verdict (including `Unscored`, no judge
/// configured): the reader did not decline, and it should have.
fn classify(c: &CaseResult, credit_abstention: bool) -> Cell {
    use Verdict::*;
    match (c.answerable, &c.verdict) {
        (true, Correct) => Cell::ShouldServe,
        (true, Wrong) => Cell::ShouldAbstain,
        (true, Partial) if credit_abstention => Cell::Skip, // safe miss, not in the binary
        (true, Partial) => Cell::ShouldAbstain,
        (false, _) => Cell::ShouldAbstain, // any answer to an unanswerable Q
        (_, Unscored) => Cell::Skip,
    }
}

/// Load every graded, chunk-resolvable case from `runs/<tag>/cases/*.json` under `run_dir`.
/// `corpus_glossa` is the corpus `.glossa` dir the run was produced against (resolved by the
/// caller — `load_cases` stays pure so it's directly unit-testable against a fixture). Skips
/// errored cases and cases with no served answer (nothing to check groundedness of), and drops
/// `Cell::Skip` cases (safe misses / unscored) out of the calibration pool.
pub fn load_cases(
    run_dir: &std::path::Path,
    corpus_glossa: &std::path::Path,
    credit_abstention: bool,
) -> anyhow::Result<Vec<Case>> {
    let df =
        glossa::gate::df::DfTable::load(&glossa::gate::df::DfTable::sidecar_path(corpus_glossa))
            .with_context(|| format!("loading df sidecar under {}", corpus_glossa.display()))?;
    let rare_df_frac = VerifyConfig::resolve(corpus_glossa).rare_df_frac;
    let raw = crate::report::load_cases(&run_dir.join("cases"))?;
    let pb = mk_bar(raw.len() as u64);
    pb.set_prefix("scoring cases");
    let mut out = Vec::new();
    for c in raw {
        pb.inc(1);
        if c.errored || c.final_answer.trim().is_empty() {
            continue; // no served candidate to check
        }
        let cell = classify(&c, credit_abstention);
        if cell == Cell::Skip {
            continue;
        }
        let chunks: Vec<String> = c
            .chunk_paths
            .iter()
            .filter_map(|p| glossa::gate::read_chunk_text(corpus_glossa, p).ok())
            .collect();
        let s = score(&c.final_answer, &chunks, &df, rare_df_frac);
        out.push(Case {
            grounding: s.grounding,
            bucket: s.bucket,
            cell,
            nli: None, // Plan 1 has no NLI scorer wired in yet; Plan 2 fills this shadow slot.
        });
    }
    pb.finish_and_clear();
    Ok(out)
}

/// One point on the threshold-sweep curve: at `threshold`, cases with `grounding > threshold` are
/// served. `answered_pct` and `error_pct` are both over the served/universe split described in
/// spec §10 — "answers X%" means ALL served cases (correct + wrong), not just the correct ones.
#[derive(serde::Serialize)]
pub struct Point {
    pub threshold: f32,
    pub answered_pct: f32,
    pub error_pct: f32,
    pub served: usize,
}

/// The full sweep over one bucket's cases (caller filters by `Bucket` before calling `sweep` —
/// `sweep` itself is bucket-agnostic). `n` is the universe size the sweep was computed over.
pub struct BucketReport {
    pub points: Vec<Point>,
    pub n: usize,
}

impl BucketReport {
    /// Lowest threshold whose `error_pct` is within `budget` — i.e. the maximum-coverage
    /// threshold that still meets the tolerated false-serve rate. `None` when no threshold on the
    /// curve satisfies `budget` (e.g. the single highest threshold, which serves nothing and has
    /// `error_pct == 0.0` by the empty-served convention below, always satisfies `budget >= 0.0`,
    /// so this is only `None` for a degenerate empty sweep).
    pub fn recommended_for(&self, budget: f32) -> Option<f32> {
        self.points
            .iter()
            .filter(|p| p.error_pct <= budget)
            .min_by(|a, b| a.threshold.partial_cmp(&b.threshold).unwrap())
            .map(|p| p.threshold)
    }
}

/// Shared sweep logic over any (cloneable) iterator of case references — factored out so
/// [`sweep`] and [`sweep_cv`] (which sweeps a filtered subset) don't duplicate the threshold-curve
/// math. `n` is the universe size for `answered_pct` (the caller's fold-subset size for
/// `sweep_cv`, or `cases.len()` for the whole-set `sweep`).
///
/// Candidate thresholds are the MIDPOINTS between consecutive distinct grounding scores, plus a
/// sentinel below the minimum (serves everyone) and one above the maximum (serves no one) —
/// never a score's own value. `grounding > t` at `t == some_case.grounding` already excludes that
/// exact case, but calibrating to that knife-edge overfits a single training example: a held-out
/// case scoring a hair above it would be served anyway. The midpoint carries a genuine margin.
fn sweep_over<'a>(cases: impl Iterator<Item = &'a Case> + Clone, n: usize) -> BucketReport {
    let mut scores: Vec<f32> = cases.clone().map(|c| c.grounding).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    scores.dedup();
    let mut thresholds: Vec<f32> = Vec::with_capacity(scores.len() + 1);
    if let Some(&min) = scores.first() {
        thresholds.push(min - 0.001); // below every score: serves everyone
    }
    for w in scores.windows(2) {
        thresholds.push((w[0] + w[1]) / 2.0);
    }
    if let Some(&max) = scores.last() {
        thresholds.push(max + 0.001); // above every score: serves no one
    }
    let points = thresholds
        .into_iter()
        .map(|t| {
            let served: Vec<&Case> = cases.clone().filter(|c| c.grounding > t).collect();
            let wrong = served
                .iter()
                .filter(|c| c.cell == Cell::ShouldAbstain)
                .count();
            // spec §10: "answers X%" = served / universe (all served, correct+wrong); error =
            // wrong / served.
            Point {
                threshold: t,
                served: served.len(),
                answered_pct: if n == 0 {
                    0.0
                } else {
                    served.len() as f32 / n as f32
                },
                error_pct: if served.is_empty() {
                    0.0
                } else {
                    wrong as f32 / served.len() as f32
                },
            }
        })
        .collect();
    BucketReport { points, n }
}

/// Sweep candidate thresholds (see [`sweep_over`]) derived from `cases`' grounding scores and
/// report the resulting `(answered_pct, error_pct)` curve over the whole set. `_folds` is
/// accepted (not used here) so callers can pass `CalibrateArgs::folds` uniformly with
/// [`sweep_cv`]; the honesty spread across folds comes from `sweep_cv`, not from this whole-set
/// curve.
pub fn sweep(cases: &[Case], _folds: usize) -> BucketReport {
    sweep_over(cases.iter(), cases.len())
}

/// The calibrated verify combination policy for one corpus: which signal decides, plus the AC and
/// (when chosen) NLI per-bucket thresholds. Plan 1 selects between `ac` and `nli`; `combined` is
/// deferred to Plan 2 (needs real labeled nli).
pub struct ModeSelection {
    pub mode: glossa::gate::config::VerifyMode,
    pub ac_single: f32,
    pub ac_multi: f32,
    pub nli_single: Option<f32>,
    pub nli_multi: Option<f32>,
}

/// The recommended threshold for one bucket via the existing sweep, or None if that bucket is empty.
fn bucket_threshold(cases: &[Case], bucket: Bucket, folds: usize, budget: f32) -> Option<f32> {
    let bucket_cases: Vec<Case> = cases
        .iter()
        .copied()
        .filter(|c| c.bucket == bucket)
        .collect();
    if bucket_cases.is_empty() {
        return None;
    }
    sweep(&bucket_cases, folds).recommended_for(budget)
}

/// Fraction of the whole pool that would be SERVED at the given per-bucket thresholds using `score`
/// (a case is served when its score > its bucket threshold). Used to compare ac vs nli coverage.
fn coverage_at(
    cases: &[Case],
    score: impl Fn(&Case) -> f32,
    single_thr: f32,
    multi_thr: f32,
) -> f32 {
    if cases.is_empty() {
        return 0.0;
    }
    let served = cases
        .iter()
        .filter(|c| {
            let thr = if matches!(c.bucket, Bucket::Single) {
                single_thr
            } else {
                multi_thr
            };
            score(c) > thr
        })
        .count();
    served as f32 / cases.len() as f32
}

/// Choose `ac` vs `nli` from the labeled pool by max served-coverage subject to `error <= budget`,
/// tie-break `ac > nli` (a model-free gate beats a model gate that only ties it). `nli` is a
/// candidate only when enough cases carry a score (`nli.is_some()`); otherwise `ac` wins by default.
/// `combined` is intentionally not considered here (Plan 1 scope ruling).
///
/// `ac_single`/`ac_multi` are the FINAL, already-decided AC thresholds — the caller (`run`'s
/// per-bucket sweep loop) has already recomputed them for whichever bucket(s) `--bucket` asked for
/// and preserved the existing ontology value for the bucket it didn't touch. `select_mode` must NOT
/// recompute AC itself (that would silently override the preserved bucket from the full case pool,
/// defeating `--bucket single`/`multi`); it only decides the mode and, when wanted, an NLI
/// candidate. `want_single`/`want_multi` gate the NLI sweep the same way: an unswept bucket's NLI
/// threshold stays `None`, which (since `Nli` requires BOTH `nli_single` and `nli_multi` to be
/// `Some`) naturally falls back to `Ac` rather than inventing an NLI threshold for a bucket the
/// caller didn't ask to calibrate.
pub fn select_mode(
    cases: &[Case],
    budget: f32,
    folds: usize,
    ac_single: f32,
    ac_multi: f32,
    want_single: bool,
    want_multi: bool,
) -> ModeSelection {
    let ac_cov = coverage_at(cases, |c| c.grounding, ac_single, ac_multi);

    // NLI operating point: sweep on the nli value, over cases that HAVE one. If none carry nli, NLI
    // is unavailable -> ac wins.
    let nli_cases: Vec<Case> = cases
        .iter()
        .filter(|c| c.nli.is_some())
        .map(|c| Case {
            grounding: c.nli.unwrap(),
            bucket: c.bucket,
            cell: c.cell,
            nli: c.nli,
        })
        .collect();
    let (mode, nli_single, nli_multi) = if nli_cases.is_empty() {
        (glossa::gate::config::VerifyMode::Ac, None, None)
    } else {
        // Only sweep a bucket's NLI threshold when the caller asked to calibrate it; an unwanted
        // bucket stays `None` (see the doc comment above — this is what makes `--bucket single`
        // leave the multi NLI threshold untouched too).
        let ns = want_single
            .then(|| bucket_threshold(&nli_cases, Bucket::Single, folds, budget))
            .flatten();
        let nm = want_multi
            .then(|| bucket_threshold(&nli_cases, Bucket::Multi, folds, budget))
            .flatten();
        match (ns, nm) {
            (Some(s), Some(m)) => {
                let nli_cov = coverage_at(cases, |c| c.nli.unwrap_or(f32::NEG_INFINITY), s, m);
                // tie-break ac > nli: nli wins only if strictly higher coverage at the budget.
                if nli_cov > ac_cov {
                    (glossa::gate::config::VerifyMode::Nli, Some(s), Some(m))
                } else {
                    (glossa::gate::config::VerifyMode::Ac, Some(s), Some(m))
                }
            }
            _ => (glossa::gate::config::VerifyMode::Ac, ns, nm),
        }
    };
    ModeSelection {
        mode,
        ac_single,
        ac_multi,
        nli_single,
        nli_multi,
    }
}

/// Deterministic FNV-1a hash of a case index, used only to assign k-fold membership. Never
/// `rand`/wall-clock: the sweep must be reproducible run-to-run over the same graded pool.
fn fold_of(index: usize, folds: usize) -> usize {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in index.to_le_bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (h % folds as u64) as usize
}

/// Per-fold cross-validated threshold recommendation — the "honesty spread" companion to
/// [`sweep`]'s whole-set curve. For each of `folds` deterministic (hashed-index) partitions, the
/// threshold is picked with [`BucketReport::recommended_for`] on the OTHER `folds - 1` partitions
/// (train), never on the held-out fold itself, so the spread across `fold_thresholds` reflects
/// genuine sample variance rather than the whole-set sweep overfitting itself.
pub struct CvReport {
    pub fold_thresholds: Vec<Option<f32>>,
}

pub fn sweep_cv(cases: &[Case], folds: usize, budget: f32) -> CvReport {
    let folds = folds.max(1);
    let fold_thresholds = (0..folds)
        .map(|held_out| {
            let train: Vec<&Case> = cases
                .iter()
                .enumerate()
                .filter(|(i, _)| fold_of(*i, folds) != held_out)
                .map(|(_, c)| c)
                .collect();
            let report = sweep_over(train.iter().copied(), train.len());
            report.recommended_for(budget)
        })
        .collect();
    CvReport { fold_thresholds }
}

/// The most recent run directory under `runs/` by directory mtime — used when `--run` is omitted.
/// Only directories that actually hold a `cases/` subdir count as runs (skips stray dirs).
fn latest_run(runs_dir: &std::path::Path) -> anyhow::Result<String> {
    std::fs::read_dir(runs_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("cases").is_dir())
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .ok_or_else(|| anyhow::anyhow!("no runs found under {}", runs_dir.display()))
}

/// Render the ASCII threshold-curve report for one bucket: a headline followed by one line per
/// swept threshold. Axes are labelled explicitly ("% answered" / "% wrong") so the report is
/// self-explanatory without cross-referencing the JSON schema.
pub fn render_ascii(bucket: &str, rep: &BucketReport) -> String {
    let mut s = format!("[{bucket}] answer-rate vs error-rate  (x = % answered, y = % wrong)\n");
    for p in &rep.points {
        s.push_str(&format!(
            "  thr {:.2} -> answers {:>4.0}%  (of those wrong {:>4.1}%)  to operator {:>4.0}%\n",
            p.threshold,
            p.answered_pct * 100.0,
            p.error_pct * 100.0,
            (1.0 - p.answered_pct) * 100.0
        ));
    }
    s
}

/// Render a self-contained SVG line chart (no external refs — no `xlink:href`, no `http(s)://` in
/// any attribute value; the `xmlns` declaration is a namespace, not a network fetch) with two
/// polylines: answered% (green) and error% (red) against threshold, left-to-right.
pub fn render_svg(bucket: &str, rep: &BucketReport) -> String {
    let mut pts_a = String::new();
    let mut pts_e = String::new();
    for (i, p) in rep.points.iter().enumerate() {
        let x = 40.0 + i as f32 / rep.points.len().max(1) as f32 * 320.0;
        pts_a.push_str(&format!("{x:.0},{:.0} ", 180.0 - p.answered_pct * 160.0));
        pts_e.push_str(&format!("{x:.0},{:.0} ", 180.0 - p.error_pct * 160.0));
    }
    format!(
        "<svg xmlns='http://www.w3.org/2000/svg' width='400' height='220'><title>{bucket}</title>\
        <polyline fill='none' stroke='#2a7' points='{pts_a}'/>\
        <polyline fill='none' stroke='#c33' points='{pts_e}'/></svg>"
    )
}

/// Set `[verify].mode`, `[verify.ac.threshold]` (canonical), `[verify.threshold]` (legacy sync —
/// keeps the currently-deployed binary working for one release), `[verify.nli.threshold]` (only
/// when `sel` carries both nli thresholds), and `[verify.calibration]` in
/// `<glossa_dir>/ontology.toml`, preserving every other table/comment via `toml_edit`. `toml_edit`
/// does not auto-vivify intermediate tables on assignment, so each level is created explicitly
/// when absent before its keys are set.
pub fn write_threshold(
    glossa_dir: &std::path::Path,
    sel: &ModeSelection,
    run: &str,
    max_error: Option<f32>,
    answered: f32,
    error: f32,
    folds: usize,
) -> anyhow::Result<()> {
    use toml_edit::{value, DocumentMut, Item, Table};
    // `f32 as f64` preserves the f32's own rounding noise (e.g. `0.42f32 as f64` is NOT the
    // nearest f64 to 0.42), so `toml_edit`'s float formatter would print a long, ugly decimal
    // instead of "0.42". Round-tripping through the f32's own shortest `Display` string first
    // recovers the clean decimal before widening to the f64 `value()` wants.
    let f = |v: f32| -> f64 {
        v.to_string()
            .parse()
            .expect("f32 Display always parses as f64")
    };
    let path = glossa_dir.join("ontology.toml");
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    let mut doc: DocumentMut = existing
        .parse()
        .with_context(|| format!("parsing {}", path.display()))?;

    if doc.get("verify").is_none() {
        doc["verify"] = Item::Table(Table::new());
    }
    let verify = doc["verify"]
        .as_table_mut()
        .context("[verify] is not a table")?;
    if verify.get("threshold").is_none() {
        verify["threshold"] = Item::Table(Table::new());
    }
    if verify.get("calibration").is_none() {
        verify["calibration"] = Item::Table(Table::new());
    }
    if verify.get("ac").is_none() {
        verify["ac"] = Item::Table(Table::new());
    }

    let mode_str = match sel.mode {
        glossa::gate::config::VerifyMode::Ac => "ac",
        glossa::gate::config::VerifyMode::Nli => "nli",
        glossa::gate::config::VerifyMode::Combined => "combined",
    };
    verify["mode"] = value(mode_str);

    if verify["ac"]
        .as_table_mut()
        .context("[verify.ac] is not a table")?
        .get("threshold")
        .is_none()
    {
        verify["ac"]["threshold"] = Item::Table(Table::new());
    }
    verify["ac"]["threshold"]["single"] = value(f(sel.ac_single));
    verify["ac"]["threshold"]["multi"] = value(f(sel.ac_multi));

    // Legacy sync (spec 2.5): keep the currently-deployed binary (which only reads
    // `[verify.threshold]`) working for one release after `[verify.ac.threshold]` becomes
    // canonical.
    verify["threshold"]["single"] = value(f(sel.ac_single));
    verify["threshold"]["multi"] = value(f(sel.ac_multi));

    if let (Some(s), Some(m)) = (sel.nli_single, sel.nli_multi) {
        if verify.get("nli").is_none() {
            verify["nli"] = Item::Table(Table::new());
        }
        if verify["nli"]
            .as_table_mut()
            .context("[verify.nli] is not a table")?
            .get("threshold")
            .is_none()
        {
            verify["nli"]["threshold"] = Item::Table(Table::new());
        }
        verify["nli"]["threshold"]["single"] = value(f(s));
        verify["nli"]["threshold"]["multi"] = value(f(m));
    }

    verify["calibration"]["run"] = value(run);
    if let Some(me) = max_error {
        verify["calibration"]["max_error"] = value(f(me));
    }
    verify["calibration"]["answered_pct"] = value(f(answered));
    verify["calibration"]["error_pct"] = value(f(error));
    verify["calibration"]["folds"] = value(folds as i64);

    std::fs::create_dir_all(glossa_dir)?;
    std::fs::write(&path, doc.to_string())?;
    Ok(())
}

/// `kbx eval calibrate`: resolve the corpus + run dir, load the graded case pool via
/// [`load_cases`], sweep the verify-gate threshold per bucket (`sweep`/`sweep_cv`), print the
/// ASCII curve, and write `calibration.svg`/`calibration.json` to the OUTPUT dir (`--out` if
/// given, else `runs/<tag>/` under the kbx workspace — never the indexed corpus). Under `--write`
/// the recommended threshold(s) are persisted to the corpus's `ontology.toml`
/// ([`write_threshold`]); otherwise this is a dry run and nothing under the corpus is touched.
pub fn run(args: CalibrateArgs) -> anyhow::Result<()> {
    let kbx_paths = crate::workspace::resolve(None);
    let lab = crate::lab::LabConfig::load_at(&kbx_paths.lab)
        .with_context(|| format!("loading {}", kbx_paths.lab.display()))?;
    let credit_abstention =
        crate::lab::AbstentionPolicy::from_opt(lab.tuning.abstention_policy.as_deref())
            .credit_abstention();

    let tag = match &args.run {
        Some(t) => t.clone(),
        None => latest_run(&kbx_paths.runs)?,
    };
    let run_dir = kbx_paths.runs.join(&tag);
    let corpus_glossa = crate::workspace::glossa_dir(&kbx_paths.root);

    let cases = load_cases(&run_dir, &corpus_glossa, credit_abstention)?;

    // Output dir: `--out` if given, else the run dir under the kbx workspace (`runs_dir` is
    // already `.glossa/kbx/runs`, excluded from indexing). NEVER the indexed corpus content root.
    let out_dir = args
        .out
        .clone()
        .unwrap_or_else(|| kbx_paths.runs.join(&tag));
    std::fs::create_dir_all(&out_dir)?;

    // Keep whichever bucket's threshold this sweep didn't touch as it already is in
    // `ontology.toml` (or fail-closed at 1.0 — grounding is capped at 1.0 and `decide` uses
    // strict `>`, so a threshold of exactly 1.0 never serves) rather than inventing a value for a
    // bucket the caller didn't ask to calibrate.
    let existing_cfg = VerifyConfig::resolve(&corpus_glossa);
    let mut single_threshold = existing_cfg.threshold_single.unwrap_or(1.0);
    let mut multi_threshold = existing_cfg.threshold_multi.unwrap_or(1.0);

    // `recommended_for` wants a budget, not an `Option`; an omitted `--max-error` means "no error
    // cap", i.e. accept the maximum-coverage threshold regardless of error rate (budget 1.0).
    let budget = args.max_error.unwrap_or(1.0);
    let want_single = args.bucket == "single" || args.bucket == "both";
    let want_multi = args.bucket == "multi" || args.bucket == "both";

    let mut svg_report = String::new();
    let mut json_buckets = serde_json::Map::new();
    // Weighted (by bucket universe size `n`) average of the recommended operating point across
    // every bucket actually swept, reported in `[verify.calibration]` as the single summary
    // answered/error pair `write_threshold` takes.
    let mut weighted_answered = 0.0f64;
    let mut weighted_error = 0.0f64;
    let mut weighted_n = 0usize;
    let mut summaries: Vec<String> = Vec::new();

    for (name, bucket, wanted) in [
        ("single", Bucket::Single, want_single),
        ("multi", Bucket::Multi, want_multi),
    ] {
        if !wanted {
            continue;
        }
        let bucket_cases: Vec<Case> = cases
            .iter()
            .copied()
            .filter(|c| c.bucket == bucket)
            .collect();
        let rep = sweep(&bucket_cases, args.folds);
        let cv = sweep_cv(&bucket_cases, args.folds, budget);
        svg_report.push_str(&render_svg(name, &rep));

        let recommended = rep.recommended_for(budget);
        if let Some(t) = recommended {
            match bucket {
                Bucket::Single => single_threshold = t,
                Bucket::Multi => multi_threshold = t,
            }
            if let Some(p) = rep.points.iter().find(|p| p.threshold == t) {
                weighted_answered += p.answered_pct as f64 * rep.n as f64;
                weighted_error += p.error_pct as f64 * rep.n as f64;
                weighted_n += rep.n;
                summaries.push(format!(
                    "  {name:<6} (N={:<4}) threshold {t:.2}  ->  answers {:.0}% · of those {:.0}% wrong · {:.0}% declined{}",
                    rep.n,
                    p.answered_pct * 100.0,
                    p.error_pct * 100.0,
                    (1.0 - p.answered_pct) * 100.0,
                    cv_spread(&cv.fold_thresholds),
                ));
            }
        }

        json_buckets.insert(
            name.to_string(),
            serde_json::json!({
                "n": rep.n,
                "points": rep.points,
                "recommended_threshold": recommended,
                "cv_fold_thresholds": cv.fold_thresholds,
            }),
        );
    }

    // Human summary to stdout (the full per-threshold curve lives in calibration.svg / .json).
    let cap = match args.max_error {
        Some(e) => format!("target error ≤ {:.0}%", e * 100.0),
        None => "no error cap".to_string(),
    };
    println!("\nCalibration — run \"{tag}\"   ({cap})\n");
    if summaries.is_empty() {
        println!("  (no eligible cases to calibrate on)");
    } else {
        for line in &summaries {
            println!("{line}");
        }
    }
    println!();

    std::fs::write(out_dir.join("calibration.svg"), &svg_report)
        .with_context(|| format!("writing {}", out_dir.join("calibration.svg").display()))?;
    let json_doc = serde_json::json!({
        "run": tag,
        "bucket": args.bucket,
        "folds": args.folds,
        "max_error": args.max_error,
        "buckets": json_buckets,
    });
    std::fs::write(
        out_dir.join("calibration.json"),
        serde_json::to_string_pretty(&json_doc)?,
    )
    .with_context(|| format!("writing {}", out_dir.join("calibration.json").display()))?;

    if args.write {
        let answered_pct = if weighted_n > 0 {
            (weighted_answered / weighted_n as f64) as f32
        } else {
            0.0
        };
        let error_pct = if weighted_n > 0 {
            (weighted_error / weighted_n as f64) as f32
        } else {
            0.0
        };
        let sel = select_mode(
            &cases,
            budget,
            args.folds,
            single_threshold,
            multi_threshold,
            want_single,
            want_multi,
        );
        write_threshold(
            &corpus_glossa,
            &sel,
            &tag,
            args.max_error,
            answered_pct,
            error_pct,
            args.folds,
        )?;
        let mode_str = match sel.mode {
            glossa::gate::config::VerifyMode::Ac => "ac",
            glossa::gate::config::VerifyMode::Nli => "nli",
            glossa::gate::config::VerifyMode::Combined => "combined",
        };
        println!(
            "written to ontology.toml: mode={mode_str} single={:.2} multi={:.2}   (weighted: answers {:.0}% at {:.0}% error)",
            sel.ac_single,
            sel.ac_multi,
            answered_pct * 100.0,
            error_pct * 100.0,
        );
    } else {
        println!("dry-run — not written (use --write to persist to ontology.toml)");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{load_cases, Cell};
    use crate::judge::Verdict;
    use crate::report::{write_case, CaseResult};

    /// Writes a minimal corpus (indexed, so `.glossa/df` exists) + one answerable-correct
    /// `CaseResult` under `runs/t1/cases/`. Returns `(run_dir, corpus_glossa, _tempdir_guard)` —
    /// the guard must stay bound in the caller so the tempdir isn't dropped mid-test.
    fn fixture_run() -> (std::path::PathBuf, std::path::PathBuf, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("corpus");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("a.md"),
            "the calibration procedure uses code pp.19.00.00.00 for the sensor",
        )
        .unwrap();
        glossa::index::store::index_dir(&root, true).unwrap();
        let corpus_glossa = root.join(".glossa");

        let run_dir = dir.path().join("runs").join("t1");
        let cases_dir = run_dir.join("cases");
        let case = CaseResult {
            id: "q1".into(),
            verdict: Verdict::Correct,
            reason: "ok".into(),
            f1: 1.0,
            em: 1.0,
            tools: vec!["read".into()],
            answer: "pp.19.00.00.00".into(),
            transcript: "T".into(),
            judge_raw: String::new(),
            hop_type: String::new(),
            needs_graph: String::new(),
            errored: false,
            answerable: true,
            final_answer: "the code is pp.19.00.00.00".into(),
            chunk_paths: vec!["a.md#1".into()],
            ranked_sources: vec![],
        };
        write_case(&cases_dir, &case).unwrap();
        (run_dir, corpus_glossa, dir)
    }

    #[test]
    fn classifies_and_scores_from_case() {
        let (run_dir, corpus_glossa, _guard) = fixture_run();
        let cases = load_cases(&run_dir, &corpus_glossa, false).unwrap();
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].cell, Cell::ShouldServe);
    }

    /// An errored case (transport/endpoint failure, never produced an answer) is dropped before
    /// classification even runs — it's not a wrong answer, just no candidate to check.
    #[test]
    fn errored_case_is_skipped() {
        let (run_dir, corpus_glossa, _guard) = fixture_run();
        let cases_dir = run_dir.join("cases");
        let mut errored = CaseResult {
            id: "q2".into(),
            verdict: Verdict::Unscored,
            reason: "transport error".into(),
            f1: 0.0,
            em: 0.0,
            tools: vec![],
            answer: String::new(),
            transcript: String::new(),
            judge_raw: String::new(),
            hop_type: String::new(),
            needs_graph: String::new(),
            errored: true,
            answerable: true,
            final_answer: String::new(),
            chunk_paths: vec![],
            ranked_sources: vec![],
        };
        write_case(&cases_dir, &errored).unwrap();
        errored.id = "q3".into(); // empty final_answer, not errored — also skipped
        errored.errored = false;
        write_case(&cases_dir, &errored).unwrap();

        let cases = load_cases(&run_dir, &corpus_glossa, false).unwrap();
        assert_eq!(cases.len(), 1, "only the fixture's q1 should remain");
    }

    /// An unanswerable question that still produced a non-empty answer (didn't abstain) is a
    /// ShouldAbstain miss regardless of verdict — including Unscored (no judge configured).
    #[test]
    fn unanswerable_served_answer_is_should_abstain_even_unscored() {
        let (run_dir, corpus_glossa, _guard) = fixture_run();
        let cases_dir = run_dir.join("cases");
        let hallucinated = CaseResult {
            id: "u1".into(),
            verdict: Verdict::Unscored,
            reason: String::new(),
            f1: 0.0,
            em: 0.0,
            tools: vec![],
            answer: "some answer".into(),
            transcript: String::new(),
            judge_raw: String::new(),
            hop_type: String::new(),
            needs_graph: String::new(),
            errored: false,
            answerable: false,
            final_answer: "some answer".into(),
            chunk_paths: vec![],
            ranked_sources: vec![],
        };
        write_case(&cases_dir, &hallucinated).unwrap();

        let cases = load_cases(&run_dir, &corpus_glossa, false).unwrap();
        assert_eq!(cases.len(), 2);
        assert!(cases.iter().any(|c| c.cell == Cell::ShouldAbstain));
    }

    #[test]
    fn sweep_is_monotone_and_picks_budget() {
        use super::{sweep, Case, Cell};
        use glossa::gate::Bucket;
        let mk = |g: f32, cell| Case {
            grounding: g,
            bucket: Bucket::Single,
            cell,
            nli: None,
        };
        let cases = vec![
            mk(0.9, Cell::ShouldServe),
            mk(0.8, Cell::ShouldServe),
            mk(0.4, Cell::ShouldAbstain),
            mk(0.85, Cell::ShouldAbstain), // one high-scoring wrong
        ];
        let rep = sweep(&cases, 1); // folds=1 = whole-set
                                    // raising threshold cannot increase error_pct at the recommended point
        let rec = rep.recommended_for(0.0).unwrap(); // 0 error ⇒ threshold above 0.85
        assert!(rec > 0.85);
    }

    /// `sweep_cv`'s fold assignment must be deterministic (no rand, no wall-clock): calling it
    /// twice on the same input yields identical per-fold recommendations.
    #[test]
    fn sweep_cv_is_deterministic() {
        use super::{sweep_cv, Case, Cell};
        use glossa::gate::Bucket;
        let mk = |g: f32, cell| Case {
            grounding: g,
            bucket: Bucket::Single,
            cell,
            nli: None,
        };
        let cases = vec![
            mk(0.95, Cell::ShouldServe),
            mk(0.9, Cell::ShouldServe),
            mk(0.7, Cell::ShouldServe),
            mk(0.6, Cell::ShouldAbstain),
            mk(0.5, Cell::ShouldServe),
            mk(0.3, Cell::ShouldAbstain),
        ];
        let a = sweep_cv(&cases, 3, 0.0);
        let b = sweep_cv(&cases, 3, 0.0);
        assert_eq!(a.fold_thresholds, b.fold_thresholds);
        assert_eq!(a.fold_thresholds.len(), 3);
    }

    #[test]
    fn ascii_curve_has_axes_labels() {
        let rep = super::BucketReport {
            n: 4,
            points: vec![
                super::Point {
                    threshold: 0.4,
                    answered_pct: 0.5,
                    error_pct: 0.2,
                    served: 2,
                },
                super::Point {
                    threshold: 0.9,
                    answered_pct: 0.25,
                    error_pct: 0.0,
                    served: 1,
                },
            ],
        };
        let s = super::render_ascii("single", &rep);
        assert!(s.contains("% answered") && s.contains("% wrong"));
    }

    #[test]
    fn svg_is_self_contained() {
        let rep = super::BucketReport {
            n: 1,
            points: vec![super::Point {
                threshold: 0.4,
                answered_pct: 0.5,
                error_pct: 0.1,
                served: 1,
            }],
        };
        let svg = super::render_svg("single", &rep);
        assert!(svg.starts_with("<svg") && svg.contains("</svg>"));
        // "no external refs" means no xlink:href / http(s):// URL fetched by the renderer — the
        // mandatory `xmlns='http://www.w3.org/2000/svg'` namespace declaration is not a network
        // reference, so it's excluded before checking for one.
        assert!(!svg.contains("xlink:href"));
        let without_namespace = svg.replacen("http://www.w3.org/2000/svg", "", 1);
        assert!(!without_namespace.contains("http://") && !without_namespace.contains("https://"));
    }

    /// A `ModeSelection` fixture for `write_threshold` tests: `ac` mode, no nli thresholds (the
    /// Plan 1 runtime default — `select_mode` always picks `ac` with `Case.nli == None`).
    fn ac_sel(ac_single: f32, ac_multi: f32) -> super::ModeSelection {
        super::ModeSelection {
            mode: glossa::gate::config::VerifyMode::Ac,
            ac_single,
            ac_multi,
            nli_single: None,
            nli_multi: None,
        }
    }

    #[test]
    fn write_threshold_roundtrips_into_ontology() {
        let dir = tempfile::tempdir().unwrap();
        let glossa = dir.path().join(".glossa"); // write_threshold writes <glossa_dir>/ontology.toml
        let sel = ac_sel(0.42, 0.55);
        super::write_threshold(&glossa, &sel, "runX", Some(0.02), 0.18, 0.015, 5).unwrap();
        let o = std::fs::read_to_string(glossa.join("ontology.toml")).unwrap();
        assert!(o.contains("[verify.threshold]") && o.contains("single = 0.42"));
        assert!(o.contains("[verify.ac.threshold]") && o.contains("single = 0.42"));
        assert!(o.contains("mode = \"ac\""));
        // No nli data in this fixture -> no [verify.nli.threshold] block written.
        assert!(!o.contains("[verify.nli.threshold]"));
    }

    /// `write_threshold` must preserve unrelated existing content in `ontology.toml`, not clobber
    /// it — it uses `toml_edit` specifically so a hand-authored preset survives the write.
    #[test]
    fn write_threshold_preserves_other_tables() {
        let dir = tempfile::tempdir().unwrap();
        let glossa = dir.path().join(".glossa");
        std::fs::create_dir_all(&glossa).unwrap();
        std::fs::write(
            glossa.join("ontology.toml"),
            "# a hand-authored comment\n[types.symptom]\nlabel = \"Symptom\"\n",
        )
        .unwrap();
        let sel = ac_sel(0.3, 0.6);
        super::write_threshold(&glossa, &sel, "runY", None, 0.5, 0.01, 3).unwrap();
        let o = std::fs::read_to_string(glossa.join("ontology.toml")).unwrap();
        assert!(o.contains("a hand-authored comment"));
        assert!(o.contains("[types.symptom]"));
        assert!(o.contains("label = \"Symptom\""));
        assert!(o.contains("[verify.calibration]") && o.contains("run = \"runY\""));
    }

    /// `latest_run` picks the newer-mtime `runs/<tag>` dir (deterministic `filetime`, not
    /// wall-clock sleeps) and skips dirs with no `cases/` subdir.
    #[test]
    fn latest_run_picks_newest_mtime() {
        let dir = tempfile::tempdir().unwrap();
        let runs = dir.path().join("runs");
        let old = runs.join("old");
        let new = runs.join("new");
        let stray = runs.join("not_a_run"); // no cases/ subdir — must be skipped
        std::fs::create_dir_all(old.join("cases")).unwrap();
        std::fs::create_dir_all(new.join("cases")).unwrap();
        std::fs::create_dir_all(&stray).unwrap();
        filetime::set_file_mtime(&old, filetime::FileTime::from_unix_time(1_000_000, 0)).unwrap();
        filetime::set_file_mtime(&new, filetime::FileTime::from_unix_time(2_000_000, 0)).unwrap();
        filetime::set_file_mtime(&stray, filetime::FileTime::from_unix_time(3_000_000, 0)).unwrap();
        assert_eq!(super::latest_run(&runs).unwrap(), "new");
    }

    /// Build a synthetic `Case` for `select_mode` tests directly (all fields `pub`).
    fn case(grounding: f32, bucket: super::Bucket, nli: Option<f32>, cell: Cell) -> super::Case {
        super::Case {
            grounding,
            bucket,
            cell,
            nli,
        }
    }

    #[test]
    fn select_mode_picks_ac_when_no_nli_data() {
        use super::{select_mode, Bucket};
        // All nli None -> ac chosen regardless of grounding distribution. The passed ac thresholds
        // are irrelevant to the outcome here (an empty nli pool short-circuits to `Ac` before the
        // coverage comparison), so any fixed values exercise the guard.
        let cases = vec![
            case(0.9, Bucket::Single, None, Cell::ShouldServe),
            case(0.1, Bucket::Single, None, Cell::ShouldAbstain),
        ];
        let sel = select_mode(&cases, 0.5, 5, 1.0, 1.0, true, true);
        assert!(matches!(sel.mode, glossa::gate::config::VerifyMode::Ac));
        assert_eq!(sel.nli_single, None);
    }

    /// Proves "nli strictly beats ac at the budget -> nli chosen", not just "nli data present".
    ///
    /// `select_mode` no longer recomputes AC itself (that's the caller's job — see
    /// `select_mode_preserves_unswept_bucket_threshold` below); the passed `ac_single`/`ac_multi`
    /// here are the same 0.501 the OLD in-fn recompute would have produced for this pool, traced
    /// against `sweep_over`/`recommended_for`: all 4 `grounding` values tie at 0.5 (two
    /// ShouldServe, two ShouldAbstain) per bucket, so the only candidate thresholds are "serve
    /// everyone" (0.499, error 50%) and "serve no one" (0.501, error 0%); at budget 0.0 only the
    /// latter qualifies. With `ac_single = ac_multi = 0.501`, AC's pool-wide coverage is exactly 0
    /// (0.5 is never `> 0.501`).
    ///
    /// The same cases carry `nli` scores that perfectly separate ShouldServe (0.9) from
    /// ShouldAbstain (0.1) in each bucket, and both buckets are wanted (`want_single`/`want_multi`
    /// both `true`), so `select_mode` sweeps NLI for both. Midpoint thresholds are 0.099 (serves
    /// everyone, error 50%), 0.5 (serves only the two 0.9 cases, error 0%), and 0.901 (serves no
    /// one, error 0%); at budget 0.0 the LOWEST qualifying threshold is 0.5, so
    /// `nli_single = nli_multi = 0.5` and nli's pool-wide coverage is 2/4 = 0.5 — strictly greater
    /// than AC's 0, so `select_mode` must pick `Nli`.
    #[test]
    fn select_mode_picks_nli_when_it_covers_strictly_more() {
        use super::{select_mode, Bucket};
        let cases = vec![
            case(0.5, Bucket::Single, Some(0.9), Cell::ShouldServe),
            case(0.5, Bucket::Single, Some(0.1), Cell::ShouldAbstain),
            case(0.5, Bucket::Multi, Some(0.9), Cell::ShouldServe),
            case(0.5, Bucket::Multi, Some(0.1), Cell::ShouldAbstain),
        ];
        let sel = select_mode(&cases, 0.0, 5, 0.501, 0.501, true, true);
        assert!(matches!(sel.mode, glossa::gate::config::VerifyMode::Nli));
        assert!(sel.nli_single.is_some());
        assert!(sel.nli_multi.is_some());
    }

    /// The CRITICAL regression this fixes: `--bucket single` (i.e. `want_multi = false`) must not
    /// let `select_mode` recompute — and thereby silently overwrite — the multi bucket's AC
    /// threshold from whatever multi-bucket cases happen to be in the loaded pool. The pool below
    /// has both buckets (so a full-pool recompute WOULD have produced a different multi threshold
    /// than 0.42, proving this isn't vacuous), but with `want_multi = false` the passed `ac_multi`
    /// must come back verbatim, and the multi NLI sweep must not run either (so `Nli` can't be
    /// selected — its selection needs both `nli_single` and `nli_multi`, and the pool here has no
    /// nli data anyway, so `Ac` is expected regardless).
    #[test]
    fn select_mode_preserves_unswept_bucket_threshold() {
        // Pool has BOTH buckets, but the caller asked to sweep only single (want_multi=false).
        // The passed ac_multi must be preserved verbatim, NOT recomputed from the multi cases.
        use super::{select_mode, Bucket};
        let cases = vec![
            case(0.9, Bucket::Single, None, Cell::ShouldServe),
            case(0.1, Bucket::Single, None, Cell::ShouldAbstain),
            case(0.9, Bucket::Multi, None, Cell::ShouldServe),
            case(0.1, Bucket::Multi, None, Cell::ShouldAbstain),
        ];
        let sel = select_mode(
            &cases, 0.5, 5, /*ac_single*/ 0.30, /*ac_multi*/ 0.42, true,
            /*want_multi*/ false,
        );
        assert_eq!(sel.ac_multi, 0.42); // preserved, not recomputed
        assert!(matches!(sel.mode, glossa::gate::config::VerifyMode::Ac));
    }
}
