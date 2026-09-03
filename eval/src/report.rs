//! File-based run report: `write_run` renders `runs/<tag>/report.md` (graded judge quality +
//! counts, a by-question-type breakdown, then a demoted lexical EM/F1 section, then a per-case
//! table) plus one `<id>.trace.md` per case (full transcript, answer, raw judge reply).
//! Deterministic — the timestamp is a caller-supplied string, never read from the clock inside
//! this module.

use crate::judge::Verdict;
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// One judged case, ready to render into the report table + its own trace file. Also the unit of
/// incremental persistence (`write_case`/`load_cases`) that backs `--resume`.
#[derive(Serialize, Deserialize)]
pub struct CaseResult {
    pub id: String,
    pub verdict: Verdict,
    pub reason: String,
    pub f1: f32,
    pub em: f32,
    pub tools: Vec<String>,
    pub answer: String,
    pub transcript: String,
    pub judge_raw: String,
    /// Reasoning shape the case exercises (`lexical|multihop|mixed`), carried from `Question` for
    /// the "By question type" report breakdown. `#[serde(default)]` so cases persisted by an
    /// older binary (before this field existed) still deserialize under `--resume`.
    #[serde(default)]
    pub hop_type: String,
    /// Whether answering the case requires graph-based reasoning (`yes|no|maybe`), same role as
    /// `hop_type` on a separate axis.
    #[serde(default)]
    pub needs_graph: String,
    /// True when the reader's ENDPOINT/TRANSPORT failed (e.g. a 500, or a context-length overflow
    /// on a server without middle-out truncation) — the reader never produced an answer, so this is
    /// NOT a wrong answer. Such a case is EXCLUDED from the graded-quality denominator (it would
    /// unfairly deflate the score) and surfaced as a separate count instead of being scored 0.0. A
    /// case where the reader returned wrong/empty TEXT is `errored: false` and scored normally.
    /// `#[serde(default)]` keeps run reports persisted before this field existed loadable.
    #[serde(default)]
    pub errored: bool,
    /// Whether the case is answerable from the corpus. `false` = an abstention test (out of scope);
    /// the correct behavior is to decline. Carried so the confusion matrix can split answerable vs
    /// abstention cells from `(answerable, verdict)` alone — no phrase matching, ontology-agnostic.
    /// `#[serde(default = "default_true")]` so pre-existing persisted cases load as answerable.
    #[serde(default = "default_true")]
    pub answerable: bool,
}

fn default_true() -> bool {
    true
}

/// Run-level metadata carried in the report header. `timestamp` is passed in by the caller (never
/// computed here) so `write_run`'s output is deterministic and testable.
pub struct RunMeta {
    pub model: String,
    pub judge: String,
    pub corpus: String,
    pub n: usize,
    pub timestamp: String,
}

impl RunMeta {
    /// Fixed metadata for tests — avoids every test inventing its own placeholder values.
    pub fn test() -> Self {
        RunMeta {
            model: "test-model".to_string(),
            judge: "test-judge".to_string(),
            corpus: "test-corpus".to_string(),
            n: 0,
            timestamp: "1970-01-01T00:00:00Z".to_string(),
        }
    }
}

fn verdict_label(v: Verdict) -> &'static str {
    match v {
        Verdict::Correct => "correct",
        Verdict::Partial => "partial",
        Verdict::Wrong => "wrong",
        Verdict::Unscored => "unscored",
    }
}

/// Verdict tallies shared by `quality`, `summary_text`, and `lexical_text` so the three never
/// disagree on how a run's cases are counted.
struct Tally {
    /// All cases, including endpoint-errored ones (used only for the `total` line + verdict %).
    total: usize,
    correct: usize,
    partial: usize,
    wrong: usize,
    unscored: usize,
    /// Reader-endpoint-errored cases (`CaseResult.errored`). Counted but EXCLUDED from every scored
    /// denominator (graded quality, EM/F1) so an endpoint failure never deflates the reported score.
    errored: usize,
    /// Non-errored case count — the denominator for the lexical EM/F1 means (errored cases never
    /// contribute an EM/F1 sample).
    scored: usize,
    f1_sum: f32,
    em_sum: f32,
}

impl Tally {
    /// Graded-quality denominator: cases with a REAL verdict (Correct/Partial/Wrong). Excludes both
    /// endpoint-errored cases and judge-`Unscored` cases, so neither an endpoint failure nor a judge
    /// error deflates the headline. `0` in the no-judge path (every case Unscored) — callers guard
    /// against dividing by it.
    fn graded(&self) -> usize {
        self.correct + self.partial + self.wrong
    }
}

fn tally(results: &[CaseResult]) -> Tally {
    let mut t = Tally {
        total: results.len(),
        correct: 0,
        partial: 0,
        wrong: 0,
        unscored: 0,
        errored: 0,
        scored: 0,
        f1_sum: 0.0,
        em_sum: 0.0,
    };
    for r in results {
        // An endpoint-errored case never produced an answer: exclude it from every scored tally
        // (verdict counts, EM/F1 sums) and count it separately.
        if r.errored {
            t.errored += 1;
            continue;
        }
        match r.verdict {
            Verdict::Correct => t.correct += 1,
            Verdict::Partial => t.partial += 1,
            Verdict::Wrong => t.wrong += 1,
            Verdict::Unscored => t.unscored += 1,
        }
        t.f1_sum += r.f1;
        t.em_sum += r.em;
        t.scored += 1;
    }
    t
}

/// Shared graded-quality arithmetic: `(correct*1.0 + partial*0.5) / graded`, i.e. a correct case
/// scores 1.0, a partial case scores 0.5, wrong scores 0.0. `graded` is the count of cases with a
/// REAL verdict (Correct/Partial/Wrong) — endpoint-errored and judge-`Unscored` cases are excluded
/// from it by the callers, so neither deflates the headline. Returns 0.0 for `graded == 0` (the
/// no-judge path, where every case is Unscored) rather than dividing by zero. Factored out of
/// `quality` so the overall headline and the by-question-type breakdown never disagree on the
/// formula.
fn quality_score(correct: usize, partial: usize, graded: usize) -> f32 {
    if graded == 0 {
        return 0.0;
    }
    (correct as f32 + 0.5 * partial as f32) / graded as f32
}

/// Primary headline metric for eval report golds: long free-text paragraph answers make
/// exact-match (EM) essentially always 0 and token-F1 only a weak lexical proxy, so the graded
/// LLM-judge verdict is the real signal. See `quality_score` for the formula.
pub fn quality(results: &[CaseResult]) -> f32 {
    let t = tally(results);
    quality_score(t.correct, t.partial, t.graded())
}

/// Confusion view over `(answerable, verdict)` — the FP-vs-FN picture, ontology-agnostic (no phrase
/// matching). Derived purely from each case's `answerable` flag and graded verdict; endpoint-errored
/// and `Unscored` cases are excluded (never graded). Empty string when nothing was graded (e.g. a
/// `--no-judge`/`--no-gold` run) so callers can print it unconditionally.
///
/// Cells (non-errored, real verdict):
/// - answerable: Correct = answered right (coverage) · Wrong = FP (wrong assertion) · Partial = a
///   partial/declined answer (safe-ish miss)
/// - abstention (`answerable=false`): Correct = correctly declined · Wrong = FP (hallucination) ·
///   Partial = declined but added unsupported claims
///
/// Rates: coverage = answerable-correct / answerable-graded; hallucination = abstention-wrong /
/// abstention-graded; false-positive = all-wrong / all-graded (under safety_first a decline is
/// `partial`, so Wrong isolates fabrication and this IS the FP rate the apply-gate caps).
pub fn confusion_text(results: &[CaseResult]) -> String {
    let (mut a_c, mut a_p, mut a_w) = (0usize, 0usize, 0usize);
    let (mut u_c, mut u_p, mut u_w) = (0usize, 0usize, 0usize);
    for r in results {
        if r.errored {
            continue;
        }
        match (r.answerable, r.verdict) {
            (true, Verdict::Correct) => a_c += 1,
            (true, Verdict::Partial) => a_p += 1,
            (true, Verdict::Wrong) => a_w += 1,
            (false, Verdict::Correct) => u_c += 1,
            (false, Verdict::Partial) => u_p += 1,
            (false, Verdict::Wrong) => u_w += 1,
            (_, Verdict::Unscored) => {}
        }
    }
    let a_graded = a_c + a_p + a_w;
    let u_graded = u_c + u_p + u_w;
    if a_graded + u_graded == 0 {
        return String::new();
    }
    let rate = |num: usize, den: usize| -> f32 {
        if den == 0 {
            0.0
        } else {
            num as f32 / den as f32
        }
    };
    let mut s = String::from("confusion (FP vs FN):\n");
    if a_graded > 0 {
        s.push_str(&format!(
            "answerable  n={a_graded}: correct {a_c} | partial {a_p} | wrong/FP {a_w}   -> coverage {:.3}\n",
            rate(a_c, a_graded)
        ));
    }
    if u_graded > 0 {
        s.push_str(&format!(
            "abstention  n={u_graded}: declined(correct) {u_c} | wrong/hallucination {u_w} | partial {u_p}   -> abstention-acc {:.3}, hallucination {:.3}\n",
            rate(u_c, u_graded),
            rate(u_w, u_graded)
        ));
    }
    s.push_str(&format!(
        "false-positive rate (wrong assertions / graded): {:.3}",
        rate(a_w + u_w, a_graded + u_graded)
    ));
    s
}

/// Headline stats — graded judge quality (primary) plus counts/percentages per verdict —
/// rendered as plain text. Shared by `write_run` (the report.md primary section) and the CLI's
/// post-run stdout print, so the two never drift apart. EM/F1 are NOT included here; see
/// `lexical_text` for those (demoted to a secondary, clearly-labelled block).
pub fn summary_text(results: &[CaseResult]) -> String {
    let t = tally(results);
    let pct = |n: usize| -> f32 {
        if t.total == 0 {
            0.0
        } else {
            100.0 * n as f32 / t.total as f32
        }
    };
    let q = quality(results);

    let mut s = format!(
        "judge quality (graded): {q:.3}  (correct + 0.5*partial) / answered\n\ncorrect  {} ({:.1}%)\npartial  {} ({:.1}%)\nwrong {} ({:.1}%)\nunscored {} ({:.1}%)\ntotal {}\n",
        t.correct, pct(t.correct), t.partial, pct(t.partial), t.wrong, pct(t.wrong), t.unscored, pct(t.unscored), t.total
    );
    // Surface endpoint-errored cases as their own line (never silently dropped): they are excluded
    // from the graded-quality denominator above, so this makes the exclusion visible.
    if t.errored > 0 {
        s.push_str(&format!("errored (endpoint, excluded): {}\n", t.errored));
    }
    // Prominent per-hop-type breakdown right under the headline, so the multihop-vs-lexical gap is
    // visible at a glance (the full table still lives in the "By question type" section below).
    let hop = hop_summary_line(results);
    if !hop.is_empty() {
        s.push_str(&hop);
    }
    s
}

/// Compact headline breakdown line, e.g. `by hop_type: lexical 0.545 (22) · multihop 0.400 (5)` —
/// graded `quality_score` and case count per `hop_type`, alphabetical (deterministic). Empty when
/// there are no results.
fn hop_summary_line(results: &[CaseResult]) -> String {
    let mut g: std::collections::BTreeMap<&str, (usize, usize, usize, usize)> =
        std::collections::BTreeMap::new(); // value -> (correct, partial, wrong, answered_n)
    for r in results {
        // Endpoint-errored cases never produced an answer — keep them out of the per-hop breakdown
        // and its denominator (which divides by answered cases, matching the headline).
        if r.errored {
            continue;
        }
        let k = if r.hop_type.is_empty() {
            "(untyped)"
        } else {
            r.hop_type.as_str()
        };
        let e = g.entry(k).or_default();
        e.3 += 1;
        match r.verdict {
            Verdict::Correct => e.0 += 1,
            Verdict::Partial => e.1 += 1,
            Verdict::Wrong => e.2 += 1,
            Verdict::Unscored => {}
        }
    }
    if g.is_empty() {
        return String::new();
    }
    // Graded denominator per group is `correct+partial+wrong` (Unscored excluded); the shown `(n)`
    // is the answered count.
    let parts: Vec<String> = g
        .iter()
        .map(|(k, (c, p, w, n))| format!("{k} {:.3} ({n})", quality_score(*c, *p, *c + *p + *w)))
        .collect();
    format!("by hop_type: {}\n", parts.join(" · "))
}

/// Secondary, lexical-only stats (EM mean / F1 mean). Kept for sanity-checking but demoted out of
/// the primary summary because both metrics are unreliable for the long free-text paragraph
/// answers these evals grade — EM is ~always 0 and F1 only weakly correlates with judge quality.
pub fn lexical_text(results: &[CaseResult]) -> String {
    let t = tally(results);
    // Divide by the non-errored (`scored`) count — an endpoint-errored case contributed no EM/F1
    // sample, so including it in the denominator would deflate both means.
    let f1_mean = if t.scored == 0 {
        0.0
    } else {
        t.f1_sum / t.scored as f32
    };
    let em_mean = if t.scored == 0 {
        0.0
    } else {
        t.em_sum / t.scored as f32
    };
    format!("EM mean: {em_mean:.3}\nF1 mean: {f1_mean:.3}\n")
}

/// One compact table grouping `results` by `key`, sorted alphabetically by group value (so the
/// section is deterministic regardless of the run's case order) via `BTreeMap`. Cases with an
/// empty group value are folded into `"(untyped)"`. Each row is the group's graded `quality`
/// (`quality_score`, matching the overall headline) plus its correct/partial/wrong tally.
fn by_type_table(
    dimension: &str,
    results: &[CaseResult],
    key: impl Fn(&CaseResult) -> &str,
) -> String {
    let mut groups: std::collections::BTreeMap<&str, Vec<&CaseResult>> =
        std::collections::BTreeMap::new();
    for r in results {
        // Endpoint-errored cases are excluded from the per-type breakdown (they never produced an
        // answer), keeping each row's denominator on answered cases like the headline.
        if r.errored {
            continue;
        }
        let k = key(r);
        let k = if k.is_empty() { "(untyped)" } else { k };
        groups.entry(k).or_default().push(r);
    }

    let mut out = format!("### {dimension}\n\n");
    out.push_str("| value | n | quality | correct | partial | wrong |\n");
    out.push_str("|---|---|---|---|---|---|\n");
    for (name, rs) in groups {
        let correct = rs.iter().filter(|r| r.verdict == Verdict::Correct).count();
        let partial = rs.iter().filter(|r| r.verdict == Verdict::Partial).count();
        let wrong = rs.iter().filter(|r| r.verdict == Verdict::Wrong).count();
        let n = rs.len();
        // Graded denominator excludes Unscored (a judge error must not deflate the row); `n` remains
        // the answered group size for context.
        let q = quality_score(correct, partial, correct + partial + wrong);
        out.push_str(&format!(
            "| {name} | {n} | {q:.3} | {correct} | {partial} | {wrong} |\n"
        ));
    }
    out
}

/// "By question type" section: one subsection per grouping dimension carried on `CaseResult`
/// (`hop_type`, `needs_graph`), each a compact per-value quality table. Answers "does the reader
/// handle lexical (search-answerable) vs multihop (graph-needed) questions" at a glance.
pub fn by_type_text(results: &[CaseResult]) -> String {
    let mut out = String::new();
    out.push_str(&by_type_table("hop_type", results, |r| r.hop_type.as_str()));
    out.push('\n');
    out.push_str(&by_type_table("needs_graph", results, |r| {
        r.needs_graph.as_str()
    }));
    out
}

/// Sanitize a case id into a filesystem-safe filename stem: path separators and any non
/// alphanumeric byte become `_`. Used by `write_case` so ids containing `/` (or other punctuation)
/// don't escape `cases_dir` or collide with reserved characters.
pub fn sanitize_id(id: &str) -> String {
    use std::hash::{Hash, Hasher};
    let cleaned: String = id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    // Append a deterministic short hash of the RAW id. Cleaning folds every non-alnum char to `_`,
    // so distinct ids like "a-b" and "a.b" would otherwise share a filename and one `write_case`
    // would silently overwrite the other's persisted JSON — corrupting `--resume`. The hash keeps
    // filenames unique; `DefaultHasher::new()` has a fixed seed, so it is stable across runs (a
    // resume in a later process finds the same file).
    let mut h = std::collections::hash_map::DefaultHasher::new();
    id.hash(&mut h);
    format!("{cleaned}-{:08x}", h.finish() as u32)
}

/// Persist a single case as `<cases_dir>/<sanitized-id>.json` (pretty JSON). Creates `cases_dir`
/// if needed. Called after each case finishes so a mid-run crash keeps every completed case on
/// disk — the foundation for `--resume`.
pub fn write_case(cases_dir: &Path, r: &CaseResult) -> anyhow::Result<()> {
    fs::create_dir_all(cases_dir)
        .with_context(|| format!("create cases dir {}", cases_dir.display()))?;
    let path = cases_dir.join(format!("{}.json", sanitize_id(&r.id)));
    let json = serde_json::to_string_pretty(r).context("serialize CaseResult")?;
    fs::write(&path, json).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Load every `<cases_dir>/*.json` back into `CaseResult`s. Returns an empty vec if `cases_dir`
/// doesn't exist (a fresh, non-resumed run). Order is whatever `read_dir` yields — callers that
/// care about order (report tables) should sort/dedup as needed.
pub fn load_cases(cases_dir: &Path) -> anyhow::Result<Vec<CaseResult>> {
    let mut out = Vec::new();
    let rd = match fs::read_dir(cases_dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(e).with_context(|| format!("read cases dir {}", cases_dir.display())),
    };
    for ent in rd {
        let ent = ent.with_context(|| format!("read entry in {}", cases_dir.display()))?;
        let path = ent.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let case: CaseResult =
            serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        out.push(case);
    }
    Ok(out)
}

/// Write `runs/<tag>/report.md` + one `<id>.trace.md` per case. Returns the `report.md` path.
pub fn write_run(
    runs_dir: &Path,
    tag: &str,
    meta: &RunMeta,
    results: &[CaseResult],
) -> anyhow::Result<PathBuf> {
    let dir = runs_dir.join(tag);
    fs::create_dir_all(&dir).with_context(|| format!("create run dir {}", dir.display()))?;

    let mut out = String::new();
    out.push_str(&format!("# Run `{tag}`\n\n"));
    out.push_str(&format!("- model: {}\n", meta.model));
    out.push_str(&format!("- judge: {}\n", meta.judge));
    out.push_str(&format!("- corpus: {}\n", meta.corpus));
    out.push_str(&format!("- n: {}\n", meta.n));
    out.push_str(&format!("- timestamp: {}\n\n", meta.timestamp));

    out.push_str("## Judge quality (primary)\n\n");
    out.push_str(&summary_text(results));
    out.push('\n');

    let confusion = confusion_text(results);
    if !confusion.is_empty() {
        out.push_str("\n## Abstention / FP-vs-FN\n\n");
        out.push_str(&confusion);
        out.push('\n');
    }

    out.push_str("## By question type\n\n");
    out.push_str(&by_type_text(results));
    out.push('\n');

    out.push_str("## Lexical (secondary — unreliable for free-text answers)\n\n");
    out.push_str(&lexical_text(results));
    out.push('\n');

    out.push_str("## Cases\n\n");
    out.push_str("| id | verdict | f1 | tools | reason |\n");
    out.push_str("|---|---|---|---|---|\n");
    for r in results {
        out.push_str(&format!(
            "| {} | {} | {:.2} | {} | {} |\n",
            r.id,
            verdict_label(r.verdict),
            r.f1,
            r.tools.join(", "),
            r.reason.replace('\n', " "),
        ));
    }

    let report_path = dir.join("report.md");
    fs::write(&report_path, out).with_context(|| format!("write {}", report_path.display()))?;

    for r in results {
        let trace = format!(
            "# Case `{}`\n\n## Transcript\n\n{}\n\n## Answer\n\n{}\n\n## Judge raw\n\n{}\n",
            r.id, r.transcript, r.answer, r.judge_raw
        );
        let trace_path = dir.join(format!("{}.trace.md", r.id));
        fs::write(&trace_path, trace).with_context(|| format!("write {}", trace_path.display()))?;
    }

    Ok(report_path)
}

/// One row of the customer-facing question -> answer deliverable written by `write_answers_csv`.
/// `gold`/`verdict` are only rendered when the run had golds (`with_gold`); for a `--no-gold`
/// predict-only run they are left empty and their columns are omitted entirely.
pub struct AnswerRow {
    pub id: String,
    pub question: String,
    pub answer: String,
    pub gold: String,
    pub verdict: String,
}

/// Field separator for the answers CSV: a SEMICOLON, not a comma. Excel uses the locale list
/// separator when opening a `.csv` by double-click — `;` on many non-US locales (e.g. RU/EU) — so a
/// comma-delimited file lands entirely in column A there. `;` is what those Excels expect, and the
/// free-text fields (which do contain commas) stay quoted regardless.
const CSV_SEP: &str = ";";

/// Quote one CSV field: collapse ALL internal whitespace (newlines/tabs/repeated spaces) to single
/// spaces so a field never spans multiple spreadsheet rows, then wrap in double quotes and double any
/// embedded quote (RFC 4180). A multi-line quoted field is valid CSV but Excel spills it across rows
/// when it doesn't recognize the delimiter — collapsing removes that failure mode entirely.
fn csv_field(s: &str) -> String {
    let flat = s.split_whitespace().collect::<Vec<_>>().join(" ");
    format!("\"{}\"", flat.replace('"', "\"\""))
}

/// Write the run as a flat question->answer table for handing to the customer to grade. UTF-8 with a
/// leading BOM so Excel opens Cyrillic correctly on double-click; `;`-separated (see `CSV_SEP`);
/// CRLF line endings; every field collapsed to a single line. A trailing empty `quality` column is
/// always present for the customer to fill in. Columns:
///   with_gold=true  -> id;question;answer;gold;verdict;quality
///   with_gold=false -> id;question;answer;quality
/// Returns the path written (creating parent dirs as needed).
pub fn write_answers_csv(
    path: &Path,
    rows: &[AnswerRow],
    with_gold: bool,
) -> anyhow::Result<PathBuf> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create answers dir {}", parent.display()))?;
        }
    }
    let header: &[&str] = if with_gold {
        &["id", "question", "answer", "gold", "verdict", "quality"]
    } else {
        &["id", "question", "answer", "quality"]
    };
    let mut out = String::from("\u{FEFF}"); // BOM
    out.push_str(
        &header
            .iter()
            .map(|h| csv_field(h))
            .collect::<Vec<_>>()
            .join(CSV_SEP),
    );
    out.push_str("\r\n");
    for r in rows {
        let fields: Vec<String> = if with_gold {
            vec![
                csv_field(&r.id),
                csv_field(&r.question),
                csv_field(&r.answer),
                csv_field(&r.gold),
                csv_field(&r.verdict),
                csv_field(""), // quality — blank for the customer
            ]
        } else {
            vec![
                csv_field(&r.id),
                csv_field(&r.question),
                csv_field(&r.answer),
                csv_field(""), // quality — blank for the customer
            ]
        };
        out.push_str(&fields.join(CSV_SEP));
        out.push_str("\r\n");
    }
    fs::write(path, out).with_context(|| format!("write answers csv {}", path.display()))?;
    Ok(path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answers_csv_quotes_columns_and_bom_adapts_to_gold() {
        let dir = tempfile::tempdir().unwrap();

        // with_gold=true: 6 columns, embedded comma/quote/newline survive quoting.
        let rows = vec![AnswerRow {
            id: "q1".into(),
            question: "Line one,\nwith a \"quote\"".into(),
            answer: "Because A leads to B.".into(),
            gold: "A -> B".into(),
            verdict: "Correct".into(),
        }];
        let p = dir.path().join("with_gold.csv");
        write_answers_csv(&p, &rows, true).unwrap();
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(text.starts_with('\u{FEFF}'), "must start with a BOM");
        let header = text.lines().next().unwrap();
        assert_eq!(
            header.trim_start_matches('\u{FEFF}'),
            "\"id\";\"question\";\"answer\";\"gold\";\"verdict\";\"quality\""
        );
        // Embedded newline is COLLAPSED to a space (no multi-row spill); embedded quote is doubled
        // and the field stays wrapped in quotes.
        assert!(text.contains("\"Line one, with a \"\"quote\"\"\""));
        assert!(text.contains("\"Correct\""));
        // Every record is exactly one physical line: header + 1 row = 2 lines.
        assert_eq!(text.lines().count(), 2, "one physical line per record");
        // Trailing quality column present and blank.
        assert!(text.trim_end().ends_with(";\"\""));

        // with_gold=false: 4 columns, no gold/verdict.
        let p2 = dir.path().join("no_gold.csv");
        write_answers_csv(&p2, &rows, false).unwrap();
        let text2 = std::fs::read_to_string(&p2).unwrap();
        let header2 = text2.lines().next().unwrap();
        assert_eq!(
            header2.trim_start_matches('\u{FEFF}'),
            "\"id\";\"question\";\"answer\";\"quality\""
        );
        assert!(
            !text2.contains("\"gold\""),
            "no-gold run omits the gold column"
        );
    }

    #[test]
    fn report_has_counts_table_and_traces() {
        let dir = tempfile::tempdir().unwrap();
        let rs = vec![
            CaseResult {
                id: "q1".into(),
                verdict: Verdict::Correct,
                f1: 0.5,
                tools: vec!["glossary".into()],
                answer: "A".into(),
                transcript: "T".into(),
                reason: "ok".into(),
                em: 0.0,
                judge_raw: "VERDICT: correct".into(),
                hop_type: "lexical".into(),
                needs_graph: "no".into(),
                errored: false,
                answerable: true,
            },
            CaseResult {
                id: "q2".into(),
                verdict: Verdict::Wrong,
                f1: 0.0,
                tools: vec![],
                answer: "B".into(),
                transcript: "T2".into(),
                reason: "no".into(),
                em: 0.0,
                judge_raw: "VERDICT: wrong".into(),
                hop_type: "multihop".into(),
                needs_graph: "yes".into(),
                errored: false,
                answerable: true,
            },
        ];
        let p = write_run(dir.path(), "t1", &RunMeta::test(), &rs).unwrap();
        let report = std::fs::read_to_string(&p).unwrap();
        assert!(report.contains("judge quality (graded): 0.500"));
        assert!(report.contains("## Judge quality (primary)"));
        assert!(report.contains("correct  1") && report.contains("wrong 1"));
        assert!(report.contains("## By question type"));
        assert!(report.contains("### hop_type") && report.contains("### needs_graph"));
        assert!(report.contains("| lexical | 1 | 1.000 | 1 | 0 | 0 |"));
        assert!(report.contains("| multihop | 1 | 0.000 | 0 | 0 | 1 |"));
        assert!(report.contains("## Lexical (secondary"));
        assert!(report.contains("EM mean:") && report.contains("F1 mean:"));
        assert!(report.contains("| q1 |") && report.contains("| q2 |"));
        assert!(dir.path().join("t1/q1.trace.md").exists());
    }

    fn case(id: &str, verdict: Verdict) -> CaseResult {
        CaseResult {
            id: id.into(),
            verdict,
            f1: 0.0,
            tools: vec![],
            answer: "A".into(),
            transcript: "T".into(),
            reason: "r".into(),
            em: 0.0,
            judge_raw: String::new(),
            hop_type: String::new(),
            needs_graph: String::new(),
            errored: false,
            answerable: true,
        }
    }

    /// Like `case`, but also stamps `hop_type`/`needs_graph` for the by-question-type grouping
    /// tests below.
    fn case_typed(id: &str, verdict: Verdict, hop_type: &str, needs_graph: &str) -> CaseResult {
        CaseResult {
            hop_type: hop_type.into(),
            needs_graph: needs_graph.into(),
            ..case(id, verdict)
        }
    }

    #[test]
    fn confusion_text_splits_answerable_and_abstention_cells() {
        let a1 = case("a1", Verdict::Correct); // answerable, answered right
        let a2 = case("a2", Verdict::Wrong); // answerable, wrong assertion (FP)
        let mut u1 = case("u1", Verdict::Correct); // abstention, correctly declined
        u1.answerable = false;
        let mut u2 = case("u2", Verdict::Wrong); // abstention, hallucinated (FP)
        u2.answerable = false;
        let c = confusion_text(&[a1, a2, u1, u2]);
        assert!(c.contains("coverage 0.500"), "1 correct / 2 answerable");
        assert!(c.contains("hallucination 0.500"), "1 wrong / 2 abstention");
        assert!(
            c.contains("false-positive rate (wrong assertions / graded): 0.500"),
            "2 wrong / 4 graded"
        );
        // Nothing graded (all Unscored) -> empty string, so callers can print unconditionally.
        assert_eq!(confusion_text(&[case("x", Verdict::Unscored)]).len(), 0);
    }

    #[test]
    fn write_case_and_load_cases_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let cases_dir = dir.path().join("cases");
        let rs = vec![case("q1", Verdict::Correct), case("q2", Verdict::Wrong)];
        for r in &rs {
            write_case(&cases_dir, r).unwrap();
        }

        let loaded = load_cases(&cases_dir).unwrap();
        assert_eq!(loaded.len(), 2);
        let mut ids: Vec<&str> = loaded.iter().map(|c| c.id.as_str()).collect();
        ids.sort();
        assert_eq!(ids, vec!["q1", "q2"]);
    }

    #[test]
    fn load_cases_empty_when_dir_missing() {
        let dir = tempfile::tempdir().unwrap();
        let loaded = load_cases(&dir.path().join("nonexistent")).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn write_case_sanitizes_id_with_path_separators() {
        let dir = tempfile::tempdir().unwrap();
        let cases_dir = dir.path().join("cases");
        write_case(&cases_dir, &case("a/b:c", Verdict::Partial)).unwrap();
        let loaded = load_cases(&cases_dir).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "a/b:c");
    }

    #[test]
    fn write_case_ids_that_clean_alike_do_not_collide() {
        // "a-b" and "a.b" both fold to "a_b"; without the hash suffix they'd overwrite each other.
        let dir = tempfile::tempdir().unwrap();
        let cases_dir = dir.path().join("cases");
        write_case(&cases_dir, &case("a-b", Verdict::Correct)).unwrap();
        write_case(&cases_dir, &case("a.b", Verdict::Wrong)).unwrap();
        let mut ids: Vec<String> = load_cases(&cases_dir)
            .unwrap()
            .into_iter()
            .map(|c| c.id)
            .collect();
        ids.sort();
        assert_eq!(ids, vec!["a-b".to_string(), "a.b".to_string()]);
    }

    fn quality_cases() -> Vec<CaseResult> {
        vec![
            {
                let mut c = case("q1", Verdict::Correct);
                c.f1 = 1.0;
                c.em = 1.0;
                c
            },
            {
                let mut c = case("q2", Verdict::Wrong);
                c.f1 = 0.0;
                c.em = 0.0;
                c
            },
        ]
    }

    #[test]
    fn summary_text_leads_with_graded_quality_and_reports_counts() {
        let s = summary_text(&quality_cases());
        assert!(s.contains("judge quality (graded): 0.500"));
        // Headline quality line comes before the counts block.
        assert!(s.find("judge quality").unwrap() < s.find("correct  1").unwrap());
        assert!(s.contains("correct  1 (50.0%)"));
        assert!(s.contains("wrong 1 (50.0%)"));
        assert!(s.contains("total 2"));
        // EM/F1 are demoted out of the primary summary entirely.
        assert!(!s.contains("EM mean"));
        assert!(!s.contains("F1 mean"));
    }

    #[test]
    fn lexical_text_reports_em_f1_means() {
        let s = lexical_text(&quality_cases());
        assert!(s.contains("EM mean: 0.500"));
        assert!(s.contains("F1 mean: 0.500"));
    }

    #[test]
    fn quality_scores_correct_full_partial_half_wrong_and_excludes_unscored() {
        let rs = vec![
            case("q1", Verdict::Correct),
            case("q2", Verdict::Partial),
            case("q3", Verdict::Wrong),
            case("q4", Verdict::Unscored),
        ];
        // Graded denominator counts only real verdicts (Correct/Partial/Wrong); the Unscored case
        // (e.g. a judge error) is excluded so it can't deflate the headline:
        // (1.0 + 0.5 + 0.0) / 3 = 0.5
        assert!((quality(&rs) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn quality_is_zero_for_empty_results() {
        assert_eq!(quality(&[]), 0.0);
    }

    #[test]
    fn quality_is_zero_when_all_unscored_no_judge_path() {
        // No-judge path: every case is Unscored, so the graded denominator is 0 — must print 0.0,
        // not divide by zero.
        let rs = vec![case("q1", Verdict::Unscored), case("q2", Verdict::Unscored)];
        assert_eq!(quality(&rs), 0.0);
    }

    #[test]
    fn errored_case_excluded_from_graded_quality_and_surfaced() {
        // One Correct, one Wrong, one endpoint-errored: the errored case leaves the graded
        // denominator, so quality = (1 + 0) / 2 = 0.5, and the errored count is surfaced.
        let mut errored = case("q3", Verdict::Unscored);
        errored.errored = true;
        let rs = vec![
            case("q1", Verdict::Correct),
            case("q2", Verdict::Wrong),
            errored,
        ];
        assert!((quality(&rs) - 0.5).abs() < 1e-6);
        let s = summary_text(&rs);
        assert!(
            s.contains("errored (endpoint, excluded): 1"),
            "errored count must be visible in the summary: {s}"
        );
        // total still reports every case (errored included).
        assert!(s.contains("total 3"));
    }

    #[test]
    fn unscored_judge_error_excluded_from_graded_denominator() {
        // A judge-errored (Unscored, NOT reader-errored) case must not deflate graded quality:
        // (1) / 1 = 1.0, and no endpoint-error line appears.
        let rs = vec![case("q1", Verdict::Correct), case("q2", Verdict::Unscored)];
        assert!((quality(&rs) - 1.0).abs() < 1e-6);
        assert!(!summary_text(&rs).contains("errored (endpoint"));
    }

    fn by_type_cases() -> Vec<CaseResult> {
        vec![
            case_typed("q1", Verdict::Correct, "lexical", "no"),
            case_typed("q2", Verdict::Correct, "lexical", "no"),
            case_typed("q3", Verdict::Partial, "multihop", "yes"),
            case_typed("q4", Verdict::Wrong, "multihop", "yes"),
            // No hop_type/needs_graph at all -- groups under "(untyped)".
            case("q5", Verdict::Correct),
        ]
    }

    #[test]
    fn by_type_text_computes_per_group_quality_and_counts() {
        let s = by_type_text(&by_type_cases());
        assert!(s.contains("### hop_type"));
        assert!(s.contains("### needs_graph"));
        // lexical: 2 correct / 2 total -> quality 1.0
        assert!(s.contains("| lexical | 2 | 1.000 | 2 | 0 | 0 |"));
        // multihop: 1 partial + 1 wrong / 2 total -> (0 + 0.5) / 2 = 0.250
        assert!(s.contains("| multihop | 2 | 0.250 | 0 | 1 | 1 |"));
        // untyped case (q5) groups separately from lexical/multihop.
        assert!(s.contains("| (untyped) | 1 | 1.000 | 1 | 0 | 0 |"));
        // needs_graph axis: yes = the two multihop cases, no = the two lexical cases.
        assert!(s.contains("| yes | 2 | 0.250 | 0 | 1 | 1 |"));
        assert!(s.contains("| no | 2 | 1.000 | 2 | 0 | 0 |"));
    }

    #[test]
    fn by_type_text_groups_empty_hop_type_as_untyped() {
        let rs = vec![case("q1", Verdict::Correct), case("q2", Verdict::Wrong)];
        let s = by_type_text(&rs);
        // Both cases share the same (empty) hop_type/needs_graph -- one "(untyped)" row per
        // dimension, not two.
        assert_eq!(s.matches("(untyped)").count(), 2);
        assert!(s.contains("| (untyped) | 2 | 0.500 | 1 | 0 | 1 |"));
    }

    #[test]
    fn write_run_includes_by_question_type_section() {
        let dir = tempfile::tempdir().unwrap();
        let rs = by_type_cases();
        let p = write_run(dir.path(), "t2", &RunMeta::test(), &rs).unwrap();
        let report = std::fs::read_to_string(&p).unwrap();
        assert!(report.contains("## By question type"));
        assert!(report.contains("| lexical | 2 | 1.000 | 2 | 0 | 0 |"));
        // The section comes before the demoted lexical EM/F1 section.
        assert!(
            report.find("## By question type").unwrap()
                < report.find("## Lexical (secondary").unwrap()
        );
    }
}
