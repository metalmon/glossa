//! File-based run report: `write_run` renders `runs/<tag>/report.md` (counts + a per-case table)
//! plus one `<id>.trace.md` per case (full transcript, answer, raw judge reply). Deterministic —
//! the timestamp is a caller-supplied string, never read from the clock inside this module.

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

/// Aggregate headline stats — counts/percentages per verdict + EM/F1 means — rendered as plain
/// text. Shared by `write_run` (the report.md `## Counts` section) and the CLI's post-run stdout
/// print, so the two never drift apart.
pub fn summary_text(results: &[CaseResult]) -> String {
    let total = results.len();
    let mut correct = 0usize;
    let mut partial = 0usize;
    let mut wrong = 0usize;
    let mut unscored = 0usize;
    let mut f1_sum = 0f32;
    let mut em_sum = 0f32;
    for r in results {
        match r.verdict {
            Verdict::Correct => correct += 1,
            Verdict::Partial => partial += 1,
            Verdict::Wrong => wrong += 1,
            Verdict::Unscored => unscored += 1,
        }
        f1_sum += r.f1;
        em_sum += r.em;
    }
    let pct = |n: usize| -> f32 {
        if total == 0 {
            0.0
        } else {
            100.0 * n as f32 / total as f32
        }
    };
    let f1_mean = if total == 0 { 0.0 } else { f1_sum / total as f32 };
    let em_mean = if total == 0 { 0.0 } else { em_sum / total as f32 };

    format!(
        "correct  {correct} ({:.1}%)\npartial  {partial} ({:.1}%)\nwrong {wrong} ({:.1}%)\nunscored {unscored} ({:.1}%)\ntotal {total}\nEM mean: {em_mean:.3}\nF1 mean: {f1_mean:.3}\n",
        pct(correct), pct(partial), pct(wrong), pct(unscored)
    )
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
        Err(e) => {
            return Err(e).with_context(|| format!("read cases dir {}", cases_dir.display()))
        }
    };
    for ent in rd {
        let ent = ent.with_context(|| format!("read entry in {}", cases_dir.display()))?;
        let path = ent.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let text =
            fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let case: CaseResult = serde_json::from_str(&text)
            .with_context(|| format!("parse {}", path.display()))?;
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

    out.push_str("## Counts\n\n");
    out.push_str(&summary_text(results));
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
        fs::write(&trace_path, trace)
            .with_context(|| format!("write {}", trace_path.display()))?;
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
            },
        ];
        let p = write_run(dir.path(), "t1", &RunMeta::test(), &rs).unwrap();
        let report = std::fs::read_to_string(&p).unwrap();
        assert!(report.contains("correct  1") && report.contains("wrong 1"));
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
        let mut ids: Vec<String> = load_cases(&cases_dir).unwrap().into_iter().map(|c| c.id).collect();
        ids.sort();
        assert_eq!(ids, vec!["a-b".to_string(), "a.b".to_string()]);
    }

    #[test]
    fn summary_text_reports_counts_and_means() {
        let rs = vec![
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
        ];
        let s = summary_text(&rs);
        assert!(s.contains("correct  1 (50.0%)"));
        assert!(s.contains("wrong 1 (50.0%)"));
        assert!(s.contains("total 2"));
        assert!(s.contains("EM mean: 0.500"));
        assert!(s.contains("F1 mean: 0.500"));
    }
}
