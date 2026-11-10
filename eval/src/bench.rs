//! Measurement harness for the extraction layer: `kbx bench extract`.
//!
//! Phase 1 of `docs/superpowers/specs/2026-11-10-omnivorous-extraction-design.md` is "measure
//! first, no code change before a baseline exists". This module is that baseline, and it is built
//! to keep the rule literally true: it drives an **already-built** `kb` binary as a child process
//! rather than calling the library, so the thing measured is the thing shipped, and nothing in the
//! extraction path is instrumented to make the measurement possible.
//!
//! Per-file durations therefore come from timestamping the `  + <doc_key>` progress lines
//! `kb index` already writes to stderr (`src/index/store.rs:2187`). Post-run figures — chunk count,
//! DF sidecar, manifest schema version — are read from what the run leaves behind.

use anyhow::{Context, Result};
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------------
// Task 1: a deterministic corpus
// ---------------------------------------------------------------------------------------------

/// What corpus to generate. Two runs with the same spec must produce byte-identical trees, or a
/// phase-to-phase delta measures the corpus instead of the code.
#[derive(Debug, Clone)]
pub struct CorpusSpec {
    pub files: usize,
    pub seed: u64,
    /// Sizes, in MiB, of extra single large files to append under `ladder/`. This is what exposes
    /// the per-file-size ceiling the spec asks for — and on Windows, where peak RSS is not
    /// measured, it is the memory evidence. Empty means no ladder. Sizes need not double: to
    /// bisect a step that one doubling jumped over, pass the intermediate sizes.
    pub ladder_mib: Vec<u64>,
    /// How many distinct words the prose draws from, Zipf-weighted (rank 1 is ~`n` times as likely
    /// as rank `n`), which is roughly how a real corpus's terms are distributed. At
    /// [`DEFAULT_VOCAB_WORDS`] the corpus has 32 terms and near-duplicate text; a realistic figure
    /// is tens of thousands. Same byte count and chunk count either way, so a row at one
    /// vocabulary and a row at another differ in exactly this.
    pub vocab_words: usize,
}

/// Distinct words the default generator draws from — the `WORDS` list. A corpus this narrow is a
/// pathological shape for an inverted index (few terms, enormous posting lists), which is why
/// [`CorpusSpec::vocab_words`] exists: the first tail measurements were all taken at this
/// vocabulary and could not be told apart from a corpus-size effect.
pub const DEFAULT_VOCAB_WORDS: usize = WORDS.len();

/// The doubling ladder the first baseline used, kept as a name so a row saying "the default
/// ladder" means one thing.
pub const DEFAULT_LADDER_MIB: [u64; 5] = [1, 2, 4, 8, 16];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusStats {
    pub files: usize,
    pub bytes: u64,
    pub largest: u64,
}

/// Marker tokens planted in every generated corpus, in known proportions, so a DF ratio can be
/// sampled on the same tokens across runs (see [`index_figures`]).
pub const PLANTED_TOKENS: [&str; 4] = ["zylathon", "qorvex", "brimtalve", "dunmorrow"];

/// Write a generated Markdown corpus at `dest`.
///
/// Markdown on purpose: it is the cheapest extractor in the tree, so the numbers reflect dispatch,
/// chunking and indexing rather than one format's parser. A `.ignore` is seeded from the real
/// `DEFAULT_IGNORE` so the walk behaves as it does on a customer corpus.
pub fn generate_corpus(dest: &Path, spec: &CorpusSpec) -> Result<CorpusStats> {
    std::fs::create_dir_all(dest).with_context(|| format!("create {}", dest.display()))?;
    std::fs::write(dest.join(".ignore"), glossa::default_ignore::DEFAULT_IGNORE)
        .context("seed .ignore")?;

    // ChaCha8, not StdRng (which this repo uses elsewhere): StdRng's algorithm may change between
    // `rand` major versions, and this corpus must stay byte-identical across phases that may
    // outlive a `rand` bump.
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(spec.seed);
    let vocab = vocabulary(spec.vocab_words, spec.seed);

    let mut bytes = 0u64;
    let mut largest = 0u64;
    let mut written = 0usize;

    for i in 0..spec.files {
        // A few nested directories, because the walk's per-directory work and the graph's
        // heading-path ancestry both scale with structure, not just with file count.
        let dir = dest.join(format!("d{:02}", i % 16));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("doc-{i:05}.md"));
        let body = synth_markdown(&mut rng, i, 40, &vocab);
        std::fs::write(&path, body.as_bytes())?;
        let len = body.len() as u64;
        bytes += len;
        largest = largest.max(len);
        written += 1;
    }

    if !spec.ladder_mib.is_empty() {
        let dir = dest.join("ladder");
        std::fs::create_dir_all(&dir)?;
        for (step, mib) in spec.ladder_mib.iter().enumerate() {
            let target = mib * (1u64 << 20);
            let path = dir.join(format!("big-{step:02}-{mib}MiB.md"));
            let len = write_sized_markdown(&path, &mut rng, target, &vocab)?;
            bytes += len;
            largest = largest.max(len);
            written += 1;
        }
        // A ladder exists to price its largest file, and the last file walked gets no per-file
        // sample at all (see `per_file_durations`) — so the ladder ends with a tiny file that sorts
        // after every other path in the corpus. The big file gets a closing tick and a real
        // duration; this one absorbs the finalize and is reported as the row's tail.
        bytes += write_tail_sentinel(dest)?;
        written += 1;
    }

    Ok(CorpusStats {
        files: written,
        bytes,
        largest,
    })
}

/// Write the tail sentinel: a tiny document under `zz-tail-sentinel/`, which sorts after every
/// other path a corpus is likely to hold. Returns its size in bytes.
///
/// Its only job is to be walked last. The last file walked gets no per-file sample (a progress line
/// says when a file *started*, so it has no closing tick), and whatever is left — the tantivy
/// commit, the manifest, the DF sidecar — lands in the row's tail. With a sentinel, every file the
/// corpus actually holds gets a real duration and the tail is the finalize alone.
///
/// **Only ever write this into a corpus that exists to be measured** — a generated one, or a copy
/// (`just bench-copy`). It adds a document and a chunk, so a corpus anyone reads from must not get
/// one (`never-write-into-indexed-corpus`).
pub fn write_tail_sentinel(dest: &Path) -> Result<u64> {
    let dir = dest.join("zz-tail-sentinel");
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let body = synth_markdown(&mut rng, 0, 1, &vocabulary(DEFAULT_VOCAB_WORDS, 0));
    std::fs::write(dir.join("tiny.md"), body.as_bytes()).context("write tail sentinel")?;
    Ok(body.len() as u64)
}

/// The vocabulary a corpus draws from: either the fixed `WORDS` list, or `n` generated pseudo-words.
///
/// Generated rather than taken from a word list so the vocabulary is a knob with no dependency and
/// no language: syllables, deterministic in the corpus seed, and long enough (5-9 chars) that the
/// index's term dictionary carries a realistic amount of bytes per term.
fn vocabulary(n: usize, seed: u64) -> Vec<String> {
    if n <= WORDS.len() {
        return WORDS.iter().take(n.max(1)).map(|w| w.to_string()).collect();
    }
    const ONSET: [&str; 16] = [
        "b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "tr",
    ];
    const NUCLEUS: [&str; 8] = ["a", "e", "i", "o", "u", "ai", "ou", "ea"];
    const CODA: [&str; 8] = ["", "n", "r", "s", "t", "l", "m", "k"];
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed ^ 0x5f3d_1a7c);
    let mut out = Vec::with_capacity(n);
    let mut seen = std::collections::HashSet::with_capacity(n);
    while out.len() < n {
        let syllables = 2 + (out.len() % 2);
        let mut w = String::with_capacity(9);
        for _ in 0..syllables {
            w.push_str(ONSET[rng.gen_range(0..ONSET.len())]);
            w.push_str(NUCLEUS[rng.gen_range(0..NUCLEUS.len())]);
            w.push_str(CODA[rng.gen_range(0..CODA.len())]);
        }
        if seen.insert(w.clone()) {
            out.push(w);
        }
    }
    out
}

/// Draw a word index Zipf-like (s = 1) by inverse CDF: `rank = n^u` for uniform `u`, so rank 1 is
/// about `n` times as likely as rank `n`. Cheap enough to call per word, which matters at 60 MiB.
fn zipf_index(rng: &mut impl Rng, n: usize) -> usize {
    let u: f64 = rng.gen_range(0.0..1.0);
    let rank = (n as f64).powf(u) as usize;
    rank.min(n.saturating_sub(1))
}

/// One synthetic document: a heading path a few levels deep, prose, and the planted markers.
fn synth_markdown(rng: &mut impl Rng, index: usize, paragraphs: usize, vocab: &[String]) -> String {
    let mut out = String::with_capacity(paragraphs * 220);
    out.push_str(&format!("# Document {index}\n\n"));
    for p in 0..paragraphs {
        if p.is_multiple_of(8) {
            out.push_str(&format!("## Section {}\n\n", p / 8));
        }
        if p.is_multiple_of(24) {
            out.push_str(&format!("### Subsection {}\n\n", p / 24));
        }
        for _ in 0..24 {
            let w = &vocab[zipf_index(rng, vocab.len())];
            out.push_str(w);
            out.push(' ');
        }
        // Plant one marker per document in a stable position, and a rarer one every 10th doc, so
        // the sampled DF ratios cover both a common and a rare token.
        if p == 0 {
            out.push_str(PLANTED_TOKENS[index % 2]);
            out.push(' ');
        }
        if p == 1 && index.is_multiple_of(10) {
            out.push_str(PLANTED_TOKENS[2 + (index / 10) % 2]);
            out.push(' ');
        }
        out.push_str("\n\n");
    }
    out
}

/// Grow one document to `target` bytes, returning what was written.
fn write_sized_markdown(
    path: &Path,
    rng: &mut impl Rng,
    target: u64,
    vocab: &[String],
) -> Result<u64> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut written = 0u64;
    f.write_all(b"# Large document\n\n")?;
    written += 18;
    while written < target {
        let chunk = synth_markdown(rng, 0, 40, vocab);
        f.write_all(chunk.as_bytes())?;
        written += chunk.len() as u64;
    }
    f.flush()?;
    Ok(written)
}

const WORDS: [&str; 32] = [
    "access",
    "control",
    "policy",
    "subject",
    "object",
    "attribute",
    "rule",
    "decision",
    "request",
    "resource",
    "environment",
    "obligation",
    "advice",
    "target",
    "effect",
    "permit",
    "deny",
    "indeterminate",
    "scope",
    "binding",
    "profile",
    "schema",
    "element",
    "namespace",
    "identifier",
    "reference",
    "constraint",
    "condition",
    "function",
    "designator",
    "selector",
    "combiner",
];

// ---------------------------------------------------------------------------------------------
// Task 2: progress lines and percentiles
// ---------------------------------------------------------------------------------------------

/// One `  + <doc_key>` line, stamped with the elapsed time at which it was read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileTick {
    pub doc_key: String,
    pub at: Duration,
}

/// The exact prefix `index_dir_at_locked` writes before indexing each file.
const PROGRESS_PREFIX: &str = "  + ";

/// Recognise a progress line. The rest of the line is the doc key **verbatim** — real keys carry
/// spaces and non-ASCII, so nothing here splits on whitespace.
pub fn parse_progress(line: &str, at: Duration) -> Option<FileTick> {
    let key = line.strip_prefix(PROGRESS_PREFIX)?;
    if key.is_empty() {
        return None;
    }
    Some(FileTick {
        doc_key: key.to_string(),
        at,
    })
}

/// Per-file durations: the gap from each tick to the next.
///
/// **The last file gets no sample**, and that is the whole point. A tick says when a file *started*,
/// so the last file's interval has no closing tick — running it to the end of the process instead
/// hands it everything the run does after its last file: the tantivy commit, the graph commit, the
/// DF sidecar, the manifest. That is how this harness manufactured a "16 MiB cliff": in every
/// corpus measured, `ladder/big-04-16MiB.md` sorted last, so it was charged the entire finalize.
/// Measured alone, 15 MiB costs 27.3 s and 16 MiB 30.1 s — no cliff at all, and sixteen 1 MiB
/// files cost 21.5 s against one 16 MiB file's 30.1 s for the same chunks.
///
/// What the finalize costs is now reported by [`tail`] under its own name.
///
/// Empty in, empty out — an all-unchanged pass emits no ticks at all, and reporting a 0 ms sample
/// there would be a claim rather than an absence. One tick in is also empty out: with a single
/// file, nothing separates that file from the finalize.
pub fn per_file_durations(ticks: &[FileTick]) -> Vec<Duration> {
    ticks
        .windows(2)
        .map(|w| w[1].at.saturating_sub(w[0].at))
        .collect()
}

/// Everything after the last file started: that file's own work plus the run's finalize, which no
/// progress line separates. `None` for a run that ticked no files.
///
/// This grows with the size of the index being committed, not with the last file, so a row's tail
/// belongs next to its chunk count. Watch it: at ~10k chunks it is ~15 s, at ~37k chunks ~237 s.
pub fn tail(ticks: &[FileTick], total: Duration) -> Option<Duration> {
    let last = ticks.last()?;
    Some(total.saturating_sub(last.at))
}

/// The `n` costliest files, slowest first, as `(doc_key, millis)`.
///
/// A p95 without the filename behind it is half a finding: the first baseline put `kb-abac`'s p95 at
/// 108x its p50, and nothing in the row said which document did it. This is what makes that
/// actionable.
pub fn slowest_files(ticks: &[FileTick], n: usize) -> Vec<(String, u64)> {
    let durations = per_file_durations(ticks);
    let mut pairs: Vec<(String, u64)> = ticks
        .iter()
        .zip(durations.iter())
        .map(|(t, d)| (t.doc_key.clone(), d.as_millis() as u64))
        .collect();
    // Slowest first; ties broken by key so two runs of one corpus list them in the same order.
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    pairs.truncate(n);
    pairs
}

/// Nearest-rank percentile over an already-sorted slice. `None` for no samples.
pub fn percentile(sorted: &[Duration], q: f64) -> Option<Duration> {
    if sorted.is_empty() {
        return None;
    }
    let rank = (q * sorted.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(sorted.len() - 1);
    Some(sorted[idx])
}

// ---------------------------------------------------------------------------------------------
// Task 3: the run driver
// ---------------------------------------------------------------------------------------------

/// Everything one `kb index` run yielded, including a failed one.
#[derive(Debug, Clone)]
pub struct RunOutcome {
    pub wall: Duration,
    pub ticks: Vec<FileTick>,
    pub peak_rss_bytes: Option<u64>,
    /// How busy the machine was while this ran — without it a row cannot be judged.
    pub contention: Contention,
    /// The canonical `added/removed/unchanged[/skipped]` block `kb index` prints last on stdout.
    pub summary: BTreeMap<String, String>,
    pub exit_ok: bool,
    /// Last lines of stderr, so a failure is diagnosable from the report alone.
    pub stderr_tail: String,
    /// How long the DF sidecar write took, when the run did one (a `--force` pass or a schema
    /// migration). `kb index` brackets it with its own stderr lines for exactly this reason, and it
    /// is the one named phase inside the run's tail — on a real corpus's vocabulary it can dominate
    /// it, while the generated corpus's 32-word vocabulary makes it free.
    pub df_write: Option<Duration>,
}

/// The stderr lines `kb index` brackets its DF sidecar write with (`src/index/store.rs`). Matched
/// as prefixes: both carry counts after them.
const DF_WRITE_BEGIN: &str = "writing DF sidecar:";
const DF_WRITE_END: &str = "DF sidecar written";

/// Parse the canonical summary block (`key: value` pairs, whitespace-separated).
pub fn parse_summary(line: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut it = line.split_whitespace().peekable();
    while let Some(tok) = it.next() {
        if let Some(key) = tok.strip_suffix(':') {
            if let Some(val) = it.next() {
                out.insert(key.to_string(), val.to_string());
            }
        }
    }
    out
}

/// Run `kb index [--force] <corpus>` and collect what it reports.
///
/// A missing or non-executable binary is an error: a benchmarks row must never claim a number it
/// did not take. A binary that runs and exits non-zero is NOT an error — the numbers collected so
/// far are returned with `exit_ok = false`, because a partially-indexed corpus is itself a result.
pub fn run_index(bin: &Path, corpus: &Path, force: bool) -> Result<RunOutcome> {
    let mut cmd = std::process::Command::new(bin);
    cmd.arg("index").arg(corpus);
    if force {
        cmd.arg("--force");
    }
    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    let started = Instant::now();
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn {} index", bin.display()))?;

    let pid = child.id();
    let rss = PeakRss::start(pid);
    let load = ContentionProbe::start();

    // Drain stdout on its own thread. Reading stderr to EOF first would deadlock if the child
    // filled the stdout pipe buffer meanwhile — and `--force` keeps writing after the summary
    // (it runs the generalize pass), so "stdout is small" is not a safe assumption.
    let stdout = child.stdout.take().expect("stdout was piped");
    let stdout_reader = std::thread::spawn(move || {
        let mut lines = Vec::new();
        for line in BufReader::new(stdout).lines() {
            match line {
                Ok(l) => lines.push(l),
                Err(_) => break,
            }
        }
        lines
    });

    let stderr = child.stderr.take().expect("stderr was piped");
    let mut ticks = Vec::new();
    let mut df_started: Option<Duration> = None;
    let mut df_write: Option<Duration> = None;
    let mut tail: std::collections::VecDeque<String> = std::collections::VecDeque::new();
    for line in BufReader::new(stderr).lines() {
        let Ok(line) = line else { break };
        if let Some(t) = parse_progress(&line, started.elapsed()) {
            ticks.push(t);
        }
        if line.starts_with(DF_WRITE_BEGIN) {
            df_started = Some(started.elapsed());
        } else if line.starts_with(DF_WRITE_END) {
            df_write = df_started.map(|b| started.elapsed().saturating_sub(b));
        }
        tail.push_back(line);
        if tail.len() > 20 {
            tail.pop_front();
        }
    }

    let stdout_lines: Vec<String> = stdout_reader.join().unwrap_or_default();
    let status = child.wait().context("wait for kb index")?;
    let wall = started.elapsed();
    let peak_rss_bytes = rss.finish();
    let contention = load.finish();

    // The summary is the LAST line carrying `added:` — the CLI contract puts exactly one canonical
    // block on stdout, last.
    let summary = stdout_lines
        .iter()
        .rev()
        .find(|l| l.contains("added:"))
        .map(|l| parse_summary(l))
        .unwrap_or_default();

    Ok(RunOutcome {
        wall,
        ticks,
        peak_rss_bytes,
        contention,
        summary,
        exit_ok: status.success(),
        stderr_tail: tail.into_iter().collect::<Vec<_>>().join("\n"),
        df_write,
    })
}

// ---------------------------------------------------------------------------------------------
// Machine contention
// ---------------------------------------------------------------------------------------------

/// How busy the machine was, as a factor against its own state at the start of the run.
///
/// This exists because the first baseline was taken on a developer workstation with other work on
/// it, which makes every row un-judgeable: a 165-second file could be superlinear behaviour or a
/// load spike, and nothing in the row could tell them apart. A row that cannot be judged is not
/// evidence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Contention {
    /// The probe's time with the machine as it was when the run started, in microseconds.
    pub baseline_us: u64,
    /// Median probe time during the run.
    pub p50_us: u64,
    /// Worst probe time during the run — what a spike looks like.
    pub max_us: u64,
    pub samples: usize,
}

impl Contention {
    /// Median probe time over its baseline. 1.0 means as quiet as at the start; 3.0 means the probe
    /// took three times as long, so something else was taking the CPU.
    pub fn factor(&self) -> f64 {
        if self.baseline_us == 0 {
            return 1.0;
        }
        self.p50_us as f64 / self.baseline_us as f64
    }

    /// Worst-case factor — a short spike shows here and not in `factor`.
    pub fn peak_factor(&self) -> f64 {
        if self.baseline_us == 0 {
            return 1.0;
        }
        self.max_us as f64 / self.baseline_us as f64
    }
}

/// Fixed, deterministic CPU work. The accumulator is returned so the optimiser cannot drop it.
fn probe_work() -> u64 {
    let mut acc: u64 = 0x9E37_79B9_7F4A_7C15;
    for i in 0..3_000_000u64 {
        acc = acc.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(i);
        acc ^= acc >> 29;
    }
    acc
}

/// Time one probe.
fn probe_once() -> Duration {
    let t = Instant::now();
    let acc = probe_work();
    let d = t.elapsed();
    // Consume the accumulator so the loop cannot be optimised away.
    if acc == 0 {
        eprintln!("probe produced zero, which is possible but vanishingly unlikely");
    }
    d
}

/// Samples [`probe_once`] on its own thread for the length of a run.
///
/// It costs one core. That is the point: it measures how much of a core the rest of the machine is
/// willing to leave alone, which is the quantity a reader of the row needs.
struct ContentionProbe {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    samples: std::sync::Arc<std::sync::Mutex<Vec<u64>>>,
    baseline_us: u64,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl ContentionProbe {
    fn start() -> Self {
        // Two probes, take the faster: the first can pay for cold caches and branch predictors.
        let baseline_us = probe_once().min(probe_once()).as_micros() as u64;
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let samples = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let (s, v) = (stop.clone(), samples.clone());
        let handle = Some(std::thread::spawn(move || {
            while !s.load(std::sync::atomic::Ordering::Relaxed) {
                let us = probe_once().as_micros() as u64;
                if let Ok(mut g) = v.lock() {
                    g.push(us);
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        }));
        Self {
            stop,
            samples,
            baseline_us,
            handle,
        }
    }

    fn finish(self) -> Contention {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(h) = self.handle {
            let _ = h.join();
        }
        let mut v = self.samples.lock().map(|g| g.clone()).unwrap_or_default();
        v.sort_unstable();
        let p50 = if v.is_empty() {
            self.baseline_us
        } else {
            v[v.len() / 2]
        };
        let max = v.last().copied().unwrap_or(self.baseline_us);
        Contention {
            baseline_us: self.baseline_us,
            p50_us: p50,
            max_us: max,
            samples: v.len(),
        }
    }
}

/// Peak resident set of a child, where the platform gives it away cheaply.
///
/// Linux only, by `VmHWM` in `/proc/<pid>/status`. On other platforms this reports `None` rather
/// than a guess: the Win32 route needs `windows-sys` as a direct dependency, and the
/// dependency-free alternative (spawning a shell every few hundred milliseconds for a multi-minute
/// run) would steal CPU from the process being measured. The report names the platform so a `None`
/// reads as "not measured here", and the size ladder carries the memory evidence instead.
struct PeakRss {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    peak: std::sync::Arc<std::sync::atomic::AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl PeakRss {
    fn start(pid: u32) -> Self {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let peak = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let handle = if cfg!(target_os = "linux") {
            let (s, p) = (stop.clone(), peak.clone());
            Some(std::thread::spawn(move || {
                while !s.load(std::sync::atomic::Ordering::Relaxed) {
                    if let Some(kb) = read_vm_hwm_kb(pid) {
                        p.fetch_max(kb * 1024, std::sync::atomic::Ordering::Relaxed);
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            }))
        } else {
            None
        };
        Self { stop, peak, handle }
    }

    fn finish(self) -> Option<u64> {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(h) = self.handle {
            let _ = h.join();
        }
        let v = self.peak.load(std::sync::atomic::Ordering::Relaxed);
        if v == 0 {
            None
        } else {
            Some(v)
        }
    }
}

/// `VmHWM` (peak resident set, in kB) for a pid, or `None` where it cannot be read.
fn read_vm_hwm_kb(pid: u32) -> Option<u64> {
    let s = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            return rest
                .split_whitespace()
                .next()
                .and_then(|n| n.parse::<u64>().ok());
        }
    }
    None
}

// ---------------------------------------------------------------------------------------------
// Task 4: what the run left behind
// ---------------------------------------------------------------------------------------------

/// Figures read from the index, the DF sidecar and the manifest after a run.
#[derive(Debug, Clone)]
pub struct IndexFigures {
    pub chunks: Option<u64>,
    pub df_n_chunks: Option<u32>,
    pub df_tokens: Option<usize>,
    /// `df(token) / n_chunks` for each requested token — the quantity `is_rare` compares against
    /// (`src/gate/df.rs:47-49`). Counts alone would not show the denominator moving.
    pub df_ratio_sample: Vec<(String, f32)>,
    pub schema_version: u32,
    pub files_in_manifest: usize,
}

/// Read post-run figures from `state_base`.
///
/// `df_tokens` are **supplied, not discovered**: `DfTable`'s map is private, and a fixed token list
/// is the more honest measurement anyway — "the most frequent tokens of this run" samples different
/// tokens once the corpus changes, so a phase-to-phase comparison would compare different things.
/// A missing sidecar reports `None`, never 0: zero would read as "no chunks", a different claim.
pub fn index_figures(state_base: &Path, df_tokens: &[String]) -> Result<IndexFigures> {
    let chunks = glossa::index::store::DocIndex::open_or_create(state_base)
        .ok()
        .and_then(|idx| idx.index.reader().ok().map(|r| r.searcher().num_docs()));

    let glossa_dir = state_base.join(".glossa");
    let df_path = glossa::gate::df::DfTable::sidecar_path(&glossa_dir);
    let df = glossa::gate::df::DfTable::load(&df_path).ok();

    let (df_n_chunks, df_tokens_len, df_ratio_sample) = match &df {
        Some(t) => {
            let n = t.n_chunks;
            let sample = df_tokens
                .iter()
                .map(|tok| {
                    let ratio = if n == 0 {
                        0.0
                    } else {
                        t.df(tok) as f32 / n as f32
                    };
                    (tok.clone(), ratio)
                })
                .collect();
            (Some(n), Some(t.len()), sample)
        }
        None => (None, None, Vec::new()),
    };

    let manifest = glossa::index::manifest::Manifest::load(state_base);

    Ok(IndexFigures {
        chunks,
        df_n_chunks,
        df_tokens: df_tokens_len,
        df_ratio_sample,
        schema_version: manifest.index_schema_version,
        files_in_manifest: manifest.files.len(),
    })
}

// ---------------------------------------------------------------------------------------------
// Task 5: the read path
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadStats {
    pub calls: usize,
    pub p50: Option<Duration>,
    pub p95: Option<Duration>,
}

/// Time `kb read <path#n>` once per reference.
///
/// The spec requires the read path, not only indexing, because whole-file reads sit on it:
/// `get_source_file` reads the whole document before consulting its size cap, and
/// `read::extract_images` runs on every MCP read. These are CLI invocations, so they carry process
/// startup — the report says so rather than passing them off as in-process timings.
pub fn run_reads(bin: &Path, corpus: &Path, refs: &[String]) -> Result<ReadStats> {
    let mut samples = Vec::with_capacity(refs.len());
    for r in refs {
        let started = Instant::now();
        let out = std::process::Command::new(bin)
            .arg("read")
            .arg(r)
            .arg(corpus)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        if out.is_err() {
            continue;
        }
        samples.push(started.elapsed());
    }
    samples.sort_unstable();
    Ok(ReadStats {
        calls: samples.len(),
        p50: percentile(&samples, 0.50),
        p95: percentile(&samples, 0.95),
    })
}

// ---------------------------------------------------------------------------------------------
// Task 6: the report
// ---------------------------------------------------------------------------------------------

/// One measured run, as it goes into `docs/benchmarks.md` and a JSON sidecar.
///
/// Durations are stored as milliseconds rather than `Duration` so the JSON stays readable to a
/// person comparing two runs by eye, which is what the benchmarks log is for.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchReport {
    pub corpus: String,
    pub kb_version: String,
    pub platform: String,
    pub files_walked: usize,
    pub wall_ms: u64,
    /// Over every file but the last — see [`per_file_durations`] for why the last one is excluded.
    pub per_file_p50_ms: Option<u64>,
    pub per_file_p95_ms: Option<u64>,
    /// The last file's own work plus the run's finalize, which no progress line separates. Read it
    /// against `chunks`, not against the last file's size.
    pub tail_ms: Option<u64>,
    /// The DF sidecar write, the one named phase inside `tail_ms`. `None` when the run wrote none.
    pub df_write_ms: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    /// Machine contention during the run. A row taken at factor 1.0 is comparable with another at
    /// 1.0; one taken at 3.0 is a different experiment.
    pub contention: Contention,
    pub chunks: Option<u64>,
    pub chunks_per_sec: Option<f64>,
    pub largest_file_bytes: Option<u64>,
    pub df_n_chunks: Option<u32>,
    pub df_ratio_sample: Vec<(String, f32)>,
    /// The costliest files, slowest first — what makes a long p95 tail actionable rather than just
    /// alarming.
    pub slowest: Vec<(String, u64)>,
    /// The manifest's schema version before and after the run. A row that does not say which side
    /// of a migration it measured describes two different corpora — `kb-abac` sits at 3 against a
    /// current 5, so this is not hypothetical.
    pub schema_before: u32,
    pub schema_after: u32,
    pub summary: BTreeMap<String, String>,
    pub exit_ok: bool,
    pub read_calls: usize,
    pub read_p50_ms: Option<u64>,
    pub read_p95_ms: Option<u64>,
    /// Non-empty when the run failed — the last stderr lines, so the row is diagnosable alone.
    pub stderr_tail: String,
}

impl BenchReport {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }

    /// One `docs/benchmarks.md` table row.
    pub fn to_markdown_row(&self) -> String {
        let ms = |v: Option<u64>| v.map(|x| format!("{x}")).unwrap_or_else(|| "—".into());
        let rss = self
            .peak_rss_bytes
            .map(|b| format!("{:.0} MiB", b as f64 / (1024.0 * 1024.0)))
            .unwrap_or_else(|| "not measured".into());
        let cps = self
            .chunks_per_sec
            .map(|v| format!("{v:.0}"))
            .unwrap_or_else(|| "—".into());
        let largest = self
            .largest_file_bytes
            .map(|b| format!("{:.1} MiB", b as f64 / (1024.0 * 1024.0)))
            .unwrap_or_else(|| "—".into());
        // The tail is a lump of several phases; name the one the run timed, rather than leave a
        // reader of a long tail to guess which phase it was.
        let tail_cell = match (self.tail_ms, self.df_write_ms) {
            (Some(t), Some(df)) => format!("{t} ms (df {df})"),
            (Some(t), None) => format!("{t} ms"),
            (None, _) => "—".to_string(),
        };
        format!(
            "| {} | {} | {} | {} | {} ms | {} / {} ms | {} | {} | {} | {} | {} | schema {}→{} | {} / {} ms | x{:.2} / x{:.2} | {} |",
            self.corpus,
            self.kb_version,
            self.platform,
            self.files_walked,
            self.wall_ms,
            ms(self.per_file_p50_ms),
            ms(self.per_file_p95_ms),
            tail_cell,
            rss,
            self.chunks.map(|c| c.to_string()).unwrap_or_else(|| "—".into()),
            cps,
            largest,
            self.schema_before,
            self.schema_after,
            ms(self.read_p50_ms),
            ms(self.read_p95_ms),
            self.contention.factor(),
            self.contention.peak_factor(),
            if self.exit_ok { "ok" } else { "**FAILED**" },
        )
    }

    /// The header the rows sit under, so a fresh `docs/benchmarks.md` section is self-describing.
    pub fn markdown_header() -> String {
        [
            "| Corpus | kb | Platform | Files | Wall | per-file p50/p95 | Tail | Peak RSS | Chunks | Chunks/s | Largest file | Migration | read p50/p95 | Load med/peak | Run |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        .join("\n")
    }
}

/// What to measure.
#[derive(Debug, Clone)]
pub struct BenchOpts {
    pub bin: PathBuf,
    pub corpus: PathBuf,
    pub df_tokens: Vec<String>,
    pub reads: usize,
    pub force: bool,
}

/// Measure one corpus end to end: schema before, index run, figures after, read path.
pub fn measure(opts: &BenchOpts) -> Result<BenchReport> {
    let schema_before = glossa::index::manifest::Manifest::load(&opts.corpus).index_schema_version;

    let kb_version = std::process::Command::new(&opts.bin)
        .arg("--version")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into());

    let run = run_index(&opts.bin, &opts.corpus, opts.force)?;

    let mut durations = per_file_durations(&run.ticks);
    durations.sort_unstable();
    let figures = index_figures(&opts.corpus, &opts.df_tokens)?;

    // Read references straight out of the index, so the read timings hit real chunks rather than
    // addresses we invented.
    let refs = read_refs(&opts.corpus, opts.reads);
    let reads = run_reads(&opts.bin, &opts.corpus, &refs)?;

    let wall_ms = run.wall.as_millis() as u64;
    let chunks_per_sec = match (figures.chunks, wall_ms) {
        (Some(c), w) if w > 0 => Some(c as f64 * 1000.0 / w as f64),
        _ => None,
    };

    Ok(BenchReport {
        corpus: opts.corpus.display().to_string(),
        kb_version,
        platform: std::env::consts::OS.to_string(),
        files_walked: run.ticks.len(),
        wall_ms,
        per_file_p50_ms: percentile(&durations, 0.50).map(|d| d.as_millis() as u64),
        per_file_p95_ms: percentile(&durations, 0.95).map(|d| d.as_millis() as u64),
        tail_ms: tail(&run.ticks, run.wall).map(|d| d.as_millis() as u64),
        df_write_ms: run.df_write.map(|d| d.as_millis() as u64),
        peak_rss_bytes: run.peak_rss_bytes,
        contention: run.contention,
        chunks: figures.chunks,
        chunks_per_sec,
        largest_file_bytes: largest_file(&opts.corpus),
        df_n_chunks: figures.df_n_chunks,
        df_ratio_sample: figures.df_ratio_sample,
        slowest: slowest_files(&run.ticks, 5),
        schema_before,
        schema_after: figures.schema_version,
        summary: run.summary,
        exit_ok: run.exit_ok,
        read_calls: reads.calls,
        read_p50_ms: reads.p50.map(|d| d.as_millis() as u64),
        read_p95_ms: reads.p95.map(|d| d.as_millis() as u64),
        stderr_tail: if run.exit_ok {
            String::new()
        } else {
            run.stderr_tail
        },
    })
}

/// Up to `n` `path#1` references drawn from the manifest, in manifest order so two runs read the
/// same documents.
fn read_refs(corpus: &Path, n: usize) -> Vec<String> {
    glossa::index::manifest::Manifest::load(corpus)
        .files
        .keys()
        .take(n)
        .map(|k| format!("{k}#1"))
        .collect()
}

/// Largest walked file, which is what the size ladder exists to surface.
fn largest_file(corpus: &Path) -> Option<u64> {
    let mut largest = 0u64;
    let _ = glossa::walk::walk_files(corpus, None, true, &mut |p| {
        if let Ok(m) = std::fs::metadata(p) {
            largest = largest.max(m.len());
        }
        Ok(())
    });
    if largest == 0 {
        None
    } else {
        Some(largest)
    }
}

// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Sorted relative paths plus each file's bytes, hashed into one string — enough to prove two
    /// generated trees are identical without pulling in a crypto dependency.
    fn dir_digest(root: &Path) -> String {
        fn walk(dir: &Path, base: &Path, out: &mut Vec<(String, Vec<u8>)>) {
            let mut entries: Vec<_> = std::fs::read_dir(dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .collect();
            entries.sort_by_key(|e| e.file_name());
            for e in entries {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, base, out);
                } else {
                    let rel = p
                        .strip_prefix(base)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/");
                    out.push((rel, std::fs::read(&p).unwrap()));
                }
            }
        }
        let mut files = Vec::new();
        walk(root, root, &mut files);
        let mut s = String::new();
        for (rel, bytes) in files {
            s.push_str(&format!("{rel}:{}:{:x}\n", bytes.len(), fnv(&bytes)));
        }
        s
    }

    fn fnv(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }

    #[test]
    fn generate_corpus_is_byte_identical_for_one_seed() {
        // A phase-to-phase delta must measure the code, not the corpus.
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let spec = CorpusSpec {
            files: 50,
            seed: 7,
            ladder_mib: Vec::new(),
            vocab_words: DEFAULT_VOCAB_WORDS,
        };
        generate_corpus(a.path(), &spec).unwrap();
        generate_corpus(b.path(), &spec).unwrap();
        assert_eq!(dir_digest(a.path()), dir_digest(b.path()));
    }

    #[test]
    fn a_different_seed_gives_a_different_corpus() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        generate_corpus(
            a.path(),
            &CorpusSpec {
                files: 20,
                seed: 1,
                ladder_mib: Vec::new(),
                vocab_words: DEFAULT_VOCAB_WORDS,
            },
        )
        .unwrap();
        generate_corpus(
            b.path(),
            &CorpusSpec {
                files: 20,
                seed: 2,
                ladder_mib: Vec::new(),
                vocab_words: DEFAULT_VOCAB_WORDS,
            },
        )
        .unwrap();
        assert_ne!(dir_digest(a.path()), dir_digest(b.path()));
    }

    #[test]
    fn generate_corpus_reports_largest_file_and_seeds_an_ignore() {
        let d = tempfile::tempdir().unwrap();
        let st = generate_corpus(
            d.path(),
            &CorpusSpec {
                files: 10,
                seed: 1,
                ladder_mib: DEFAULT_LADDER_MIB.to_vec(),
                vocab_words: DEFAULT_VOCAB_WORDS,
            },
        )
        .unwrap();
        assert_eq!(
            st.files, 16,
            "10 documents, the 5-step ladder, and the tail sentinel"
        );
        assert!(
            st.largest >= 16 * 1024 * 1024,
            "size ladder must reach 16 MiB, got {}",
            st.largest
        );
        assert!(d.path().join(".ignore").exists());
    }

    #[test]
    fn a_ladder_can_bisect_a_step_the_doubling_jumped_over() {
        // The whole point of the field being a list: 8 -> 16 MiB is one doubling, and the step
        // inside it is what needs sizes the doubling never visits.
        let d = tempfile::tempdir().unwrap();
        let st = generate_corpus(
            d.path(),
            &CorpusSpec {
                files: 0,
                seed: 1,
                ladder_mib: vec![10, 12],
                vocab_words: DEFAULT_VOCAB_WORDS,
            },
        )
        .unwrap();
        assert_eq!(st.files, 3, "two ladder files plus the tail sentinel");
        assert!(d.path().join("ladder/big-00-10MiB.md").exists());
        assert!(d.path().join("ladder/big-01-12MiB.md").exists());
        assert!(st.largest >= 12 * 1024 * 1024 && st.largest < 13 * 1024 * 1024);
    }

    #[test]
    fn a_wide_vocabulary_is_wide_deterministic_and_zipf_weighted() {
        // The knob exists to tell a corpus-size effect apart from a vocabulary effect, so what it
        // must guarantee is: the words really are distinct, the corpus is still reproducible from
        // its seed, and the distribution is skewed rather than uniform (a real corpus's is).
        let v = vocabulary(5_000, 7);
        assert_eq!(v.len(), 5_000);
        assert_eq!(
            v.iter().collect::<std::collections::HashSet<_>>().len(),
            5_000
        );
        assert_eq!(v, vocabulary(5_000, 7), "same seed, same vocabulary");
        assert_ne!(v, vocabulary(5_000, 8));
        // 2-3 syllables, each 2-5 chars: an onset, a nucleus, and a coda that may be empty.
        assert!(
            v.iter().all(|w| w.len() >= 4 && w.len() <= 15),
            "{:?}",
            &v[..4]
        );

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let mut hits = vec![0u32; 5_000];
        for _ in 0..50_000 {
            hits[zipf_index(&mut rng, 5_000)] += 1;
        }
        // Under this law P(rank <= k) = ln k / ln n, so the first ten of 5000 ranks hold ~27% of
        // the draws where a uniform pick would give 0.2%, and the commonest term is drawn orders of
        // magnitude more often than the mean. (What it does NOT do is leave the tail unseen: at
        // ~5867/r expected draws every rank is still drawn, which is why that is not asserted.)
        let head: u32 = hits[..10].iter().sum();
        assert!(head > 10_000, "head too light to be Zipf: {head}");
        let mean = 50_000.0 / 5_000.0;
        let top = *hits.iter().max().unwrap() as f64;
        assert!(top > 50.0 * mean, "flattest rank {top} vs mean {mean}");
    }

    #[test]
    fn a_narrow_vocabulary_is_the_fixed_list_so_old_rows_stay_comparable() {
        assert_eq!(vocabulary(DEFAULT_VOCAB_WORDS, 3), WORDS.to_vec());
    }

    #[test]
    fn the_tail_sentinel_sorts_after_every_other_path() {
        // Its whole job is to be walked last, so that the ladder's largest file gets a closing
        // tick. If some directory ever sorts after it, the ladder is measuring the finalize again.
        let d = tempfile::tempdir().unwrap();
        generate_corpus(
            d.path(),
            &CorpusSpec {
                files: 40,
                seed: 1,
                ladder_mib: vec![1],
                vocab_words: DEFAULT_VOCAB_WORDS,
            },
        )
        .unwrap();
        let mut dirs: Vec<String> = std::fs::read_dir(d.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .filter(|n| !n.starts_with('.'))
            .collect();
        dirs.sort();
        assert_eq!(dirs.last().unwrap(), "zz-tail-sentinel");
    }

    #[test]
    fn planted_tokens_are_present_so_a_df_ratio_can_be_sampled() {
        let d = tempfile::tempdir().unwrap();
        generate_corpus(
            d.path(),
            &CorpusSpec {
                files: 20,
                seed: 3,
                ladder_mib: Vec::new(),
                vocab_words: DEFAULT_VOCAB_WORDS,
            },
        )
        .unwrap();
        let mut joined = String::new();
        for e in std::fs::read_dir(d.path().join("d00")).unwrap() {
            joined.push_str(&std::fs::read_to_string(e.unwrap().path()).unwrap());
        }
        assert!(
            PLANTED_TOKENS.iter().any(|t| joined.contains(t)),
            "no planted marker in the corpus"
        );
    }

    #[test]
    fn parses_a_progress_line_and_ignores_everything_else() {
        assert_eq!(
            parse_progress("  + a/b.md", Duration::from_millis(5))
                .unwrap()
                .doc_key,
            "a/b.md"
        );
        assert!(parse_progress("added: 3  unchanged: 1", Duration::ZERO).is_none());
        assert!(parse_progress("skip (walk error): boom", Duration::ZERO).is_none());
        assert!(parse_progress("  + ", Duration::ZERO).is_none());
    }

    #[test]
    fn keeps_doc_keys_with_spaces_and_non_ascii() {
        // Real corpus keys look like this; nothing may split on whitespace.
        let line = "  + plant/certificates/refusal letter TRTS 004.PDF";
        let t = parse_progress(line, Duration::ZERO).unwrap();
        assert_eq!(t.doc_key, "plant/certificates/refusal letter TRTS 004.PDF");
    }

    #[test]
    fn percentiles_of_no_samples_are_absent_not_zero() {
        // An all-unchanged pass emits no ticks at all.
        assert_eq!(per_file_durations(&[]), Vec::<Duration>::new());
        assert_eq!(percentile(&[], 0.5), None);
    }

    #[test]
    fn the_last_file_gets_no_per_file_sample() {
        // It cannot have one: a tick says when a file started, and after the last file the run
        // commits the index. Charging that to the file is how the "16 MiB cliff" was invented.
        let ticks = vec![
            FileTick {
                doc_key: "a".into(),
                at: Duration::from_millis(0),
            },
            FileTick {
                doc_key: "b".into(),
                at: Duration::from_millis(100),
            },
        ];
        assert_eq!(per_file_durations(&ticks), vec![Duration::from_millis(100)]);
        assert_eq!(
            tail(&ticks, Duration::from_millis(250)),
            Some(Duration::from_millis(150))
        );
    }

    #[test]
    fn one_file_is_all_tail_and_no_per_file_sample() {
        // A single-file run cannot separate the file from the finalize, so it claims neither.
        let ticks = vec![FileTick {
            doc_key: "only.md".into(),
            at: Duration::from_millis(5),
        }];
        assert!(per_file_durations(&ticks).is_empty());
        assert_eq!(
            tail(&ticks, Duration::from_millis(30_005)),
            Some(Duration::from_millis(30_000))
        );
    }

    #[test]
    fn a_run_that_ticked_nothing_has_no_tail() {
        assert_eq!(tail(&[], Duration::from_secs(9)), None);
    }

    #[test]
    fn slowest_files_names_the_costly_file_and_is_stable() {
        let ticks = vec![
            FileTick {
                doc_key: "cheap.md".into(),
                at: Duration::from_millis(0),
            },
            FileTick {
                doc_key: "dear.pdf".into(),
                at: Duration::from_millis(10),
            },
            FileTick {
                doc_key: "mid.md".into(),
                at: Duration::from_millis(5010),
            },
        ];
        let top = slowest_files(&ticks, 2);
        assert_eq!(top[0], ("dear.pdf".to_string(), 5000));
        // "mid.md" is last, so it has no measured duration and cannot appear.
        assert_eq!(top[1], ("cheap.md".to_string(), 10));
        // Same input, same order — a row compared across phases must list the same files.
        assert_eq!(top, slowest_files(&ticks, 2));
    }

    #[test]
    fn slowest_files_of_no_ticks_is_empty() {
        assert!(slowest_files(&[], 5).is_empty());
    }

    #[test]
    fn percentile_picks_by_nearest_rank() {
        let s: Vec<Duration> = (1..=10).map(Duration::from_millis).collect();
        assert_eq!(percentile(&s, 0.5), Some(Duration::from_millis(5)));
        assert_eq!(percentile(&s, 0.95), Some(Duration::from_millis(10)));
        assert_eq!(percentile(&s, 1.0), Some(Duration::from_millis(10)));
    }

    #[test]
    fn a_failing_child_is_reported_not_swallowed() {
        // No benchmarks row may claim success for a run that never happened.
        let out = run_index(
            Path::new("definitely-no-such-kb-binary"),
            Path::new("."),
            true,
        );
        assert!(
            out.is_err(),
            "a missing binary must be an error, not an empty RunOutcome"
        );
    }

    #[test]
    fn parses_the_canonical_summary_block() {
        let s = parse_summary("added: 38  removed: 0  unchanged: 0  skipped: 2");
        assert_eq!(s.get("added").map(String::as_str), Some("38"));
        assert_eq!(s.get("removed").map(String::as_str), Some("0"));
        assert_eq!(s.get("skipped").map(String::as_str), Some("2"));
        assert!(parse_summary("nothing to see here").is_empty());
    }

    #[test]
    fn a_missing_df_sidecar_reads_as_absent_not_zero() {
        // 0 would read as "no chunks", which is a different claim from "not measured".
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(d.path().join(".glossa")).unwrap();
        let f = index_figures(d.path(), &["zylathon".to_string()]).unwrap();
        assert_eq!(f.df_n_chunks, None);
        assert_eq!(f.df_tokens, None);
        assert!(f.df_ratio_sample.is_empty());
    }

    #[test]
    fn reports_the_schema_version_the_run_left_behind() {
        // An absent manifest defaults to version 1 (`default_index_schema_version`).
        let d = tempfile::tempdir().unwrap();
        let f = index_figures(d.path(), &[]).unwrap();
        assert_eq!(f.schema_version, 1);
        assert_eq!(f.files_in_manifest, 0);
    }

    #[test]
    fn read_stats_over_no_refs_are_absent_not_zero() {
        let s = run_reads(Path::new("kb"), Path::new("."), &[]).unwrap();
        assert_eq!(s.calls, 0);
        assert_eq!(s.p50, None);
        assert_eq!(s.p95, None);
    }

    fn sample_report() -> BenchReport {
        BenchReport {
            corpus: "kb-abac-copy".into(),
            kb_version: "kb 0.5.7".into(),
            platform: std::env::consts::OS.to_string(),
            files_walked: 38,
            wall_ms: 12_000,
            per_file_p50_ms: Some(210),
            per_file_p95_ms: Some(900),
            tail_ms: Some(15632),
            df_write_ms: Some(11),
            peak_rss_bytes: None,
            contention: Contention {
                baseline_us: 10_000,
                p50_us: 12_000,
                max_us: 30_000,
                samples: 24,
            },
            chunks: Some(4_200),
            chunks_per_sec: Some(350.0),
            largest_file_bytes: Some(16 * 1024 * 1024),
            df_n_chunks: Some(4_200),
            df_ratio_sample: vec![("zylathon".into(), 0.5)],
            slowest: vec![("d00/doc-00001.md".into(), 5270)],
            schema_before: 3,
            schema_after: 5,
            summary: BTreeMap::new(),
            exit_ok: true,
            read_calls: 20,
            read_p50_ms: Some(40),
            read_p95_ms: Some(85),
            stderr_tail: String::new(),
        }
    }

    #[test]
    fn markdown_row_states_the_schema_versions_and_the_platform() {
        let row = sample_report().to_markdown_row();
        assert!(
            row.contains("schema 3→5"),
            "a row must say which side of a migration it measured: {row}"
        );
        assert!(row.contains(std::env::consts::OS), "{row}");
    }

    #[test]
    fn an_unmeasured_peak_rss_says_so_rather_than_showing_zero() {
        // On Windows there is no peak RSS; a row showing "0 MiB" would be a false claim.
        // The cell is matched whole — a bare `contains("0 MiB")` also matches the largest-file
        // column's "16.0 MiB", which is how the first version of this assertion failed.
        let row = sample_report().to_markdown_row();
        assert!(row.contains("| not measured |"), "{row}");
        assert!(!row.contains("| 0 MiB |"), "{row}");
    }

    #[test]
    fn a_failed_run_is_marked_in_its_row() {
        let mut r = sample_report();
        r.exit_ok = false;
        assert!(r.to_markdown_row().contains("**FAILED**"));
    }

    #[test]
    fn report_json_round_trips() {
        let r = sample_report();
        let back: BenchReport = serde_json::from_str(&r.to_json()).unwrap();
        assert_eq!(back.schema_before, 3);
        assert_eq!(back.schema_after, 5);
        assert_eq!(back.files_walked, 38);
    }

    #[test]
    fn contention_factor_is_one_when_the_machine_was_as_quiet_as_at_the_start() {
        let c = Contention {
            baseline_us: 10_000,
            p50_us: 10_000,
            max_us: 10_000,
            samples: 5,
        };
        assert!((c.factor() - 1.0).abs() < 1e-9);
        assert!((c.peak_factor() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn contention_reports_a_spike_in_peak_not_in_median() {
        let c = Contention {
            baseline_us: 10_000,
            p50_us: 11_000,
            max_us: 40_000,
            samples: 20,
        };
        assert!((c.factor() - 1.1).abs() < 1e-9, "median barely moved");
        assert!(
            (c.peak_factor() - 4.0).abs() < 1e-9,
            "the spike shows in peak"
        );
    }

    #[test]
    fn contention_with_no_baseline_does_not_divide_by_zero() {
        let c = Contention {
            baseline_us: 0,
            p50_us: 5,
            max_us: 9,
            samples: 1,
        };
        assert_eq!(c.factor(), 1.0);
        assert_eq!(c.peak_factor(), 1.0);
    }

    #[test]
    fn the_row_names_the_df_phase_inside_the_tail() {
        let row = sample_report().to_markdown_row();
        assert!(row.contains("15632 ms (df 11)"), "{row}");
        // A run that wrote no sidecar says nothing about one, rather than claiming 0 ms.
        let mut r = sample_report();
        r.df_write_ms = None;
        assert!(r.to_markdown_row().contains("| 15632 ms |"));
    }

    #[test]
    fn the_row_states_the_machine_load() {
        let row = sample_report().to_markdown_row();
        assert!(row.contains("x1.20"), "median load factor missing: {row}");
        assert!(row.contains("x3.00"), "peak load factor missing: {row}");
    }

    #[test]
    fn the_header_columns_match_a_row() {
        let header = BenchReport::markdown_header();
        let cols = |s: &str| s.matches('|').count();
        let head_line = header.lines().next().unwrap();
        assert_eq!(
            cols(head_line),
            cols(&sample_report().to_markdown_row()),
            "header and row must have the same number of columns"
        );
    }
}
