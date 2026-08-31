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
    total: usize,
    correct: usize,
    partial: usize,
    wrong: usize,
    unscored: usize,
    f1_sum: f32,
    em_sum: f32,
}

fn tally(results: &[CaseResult]) -> Tally {
    let mut t = Tally {
        total: results.len(),
        correct: 0,
        partial: 0,
        wrong: 0,
        unscored: 0,
        f1_sum: 0.0,
        em_sum: 0.0,
    };
    for r in results {
        match r.verdict {
            Verdict::Correct => t.correct += 1,
            Verdict::Partial => t.partial += 1,
            Verdict::Wrong => t.wrong += 1,
            Verdict::Unscored => t.unscored += 1,
        }
        t.f1_sum += r.f1;
        t.em_sum += r.em;
    }
    t
}

/// Shared graded-quality arithmetic: `(correct*1.0 + partial*0.5) / total`, i.e. a correct case
/// scores 1.0, a partial case scores 0.5, wrong/unscored score 0.0. Returns 0.0 for `total == 0`
/// rather than dividing by zero. Factored out of `quality` so the overall headline and the
/// by-question-type breakdown (`by_type_text`) never disagree on the formula.
fn quality_score(correct: usize, partial: usize, total: usize) -> f32 {
    if total == 0 {
        return 0.0;
    }
    (correct as f32 + 0.5 * partial as f32) / total as f32
}

/// Primary headline metric for eval report golds: long free-text paragraph answers make
/// exact-match (EM) essentially always 0 and token-F1 only a weak lexical proxy, so the graded
/// LLM-judge verdict is the real signal. See `quality_score` for the formula.
pub fn quality(results: &[CaseResult]) -> f32 {
    let t = tally(results);
    quality_score(t.correct, t.partial, t.total)
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
        "judge quality (graded): {q:.3}  (correct + 0.5*partial) / total\n\ncorrect  {} ({:.1}%)\npartial  {} ({:.1}%)\nwrong {} ({:.1}%)\nunscored {} ({:.1}%)\ntotal {}\n",
        t.correct, pct(t.correct), t.partial, pct(t.partial), t.wrong, pct(t.wrong), t.unscored, pct(t.unscored), t.total
    );
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
    let mut g: std::collections::BTreeMap<&str, (usize, usize, usize)> =
        std::collections::BTreeMap::new(); // value -> (correct, partial, n)
    for r in results {
        let k = if r.hop_type.is_empty() {
            "(untyped)"
        } else {
            r.hop_type.as_str()
        };
        let e = g.entry(k).or_default();
        e.2 += 1;
        match r.verdict {
            Verdict::Correct => e.0 += 1,
            Verdict::Partial => e.1 += 1,
            _ => {}
        }
    }
    if g.is_empty() {
        return String::new();
    }
    let parts: Vec<String> = g
        .iter()
        .map(|(k, (c, p, n))| format!("{k} {:.3} ({n})", quality_score(*c, *p, *n)))
        .collect();
    format!("by hop_type: {}\n", parts.join(" · "))
}

/// Secondary, lexical-only stats (EM mean / F1 mean). Kept for sanity-checking but demoted out of
/// the primary summary because both metrics are unreliable for the long free-text paragraph
/// answers these evals grade — EM is ~always 0 and F1 only weakly correlates with judge quality.
pub fn lexical_text(results: &[CaseResult]) -> String {
    let t = tally(results);
    let f1_mean = if t.total == 0 {
        0.0
    } else {
        t.f1_sum / t.total as f32
    };
    let em_mean = if t.total == 0 {
        0.0
    } else {
        t.em_sum / t.total as f32
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
        let q = quality_score(correct, partial, n);
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn quality_scores_correct_full_partial_half_wrong_unscored_zero() {
        let rs = vec![
            case("q1", Verdict::Correct),
            case("q2", Verdict::Partial),
            case("q3", Verdict::Wrong),
            case("q4", Verdict::Unscored),
        ];
        // (1.0 + 0.5 + 0.0 + 0.0) / 4 = 0.375
        assert!((quality(&rs) - 0.375).abs() < 1e-6);
    }

    #[test]
    fn quality_is_zero_for_empty_results() {
        assert_eq!(quality(&[]), 0.0);
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
