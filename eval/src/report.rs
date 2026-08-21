//! File-based run report: `write_run` renders `runs/<tag>/report.md` (counts + a per-case table)
//! plus one `<id>.trace.md` per case (full transcript, answer, raw judge reply). Deterministic —
//! the timestamp is a caller-supplied string, never read from the clock inside this module.

use crate::judge::Verdict;
use anyhow::Context;
use std::fs;
use std::path::{Path, PathBuf};

/// One judged case, ready to render into the report table + its own trace file.
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

/// Write `runs/<tag>/report.md` + one `<id>.trace.md` per case. Returns the `report.md` path.
pub fn write_run(
    runs_dir: &Path,
    tag: &str,
    meta: &RunMeta,
    results: &[CaseResult],
) -> anyhow::Result<PathBuf> {
    let dir = runs_dir.join(tag);
    fs::create_dir_all(&dir).with_context(|| format!("create run dir {}", dir.display()))?;

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

    let mut out = String::new();
    out.push_str(&format!("# Run `{tag}`\n\n"));
    out.push_str(&format!("- model: {}\n", meta.model));
    out.push_str(&format!("- judge: {}\n", meta.judge));
    out.push_str(&format!("- corpus: {}\n", meta.corpus));
    out.push_str(&format!("- n: {}\n", meta.n));
    out.push_str(&format!("- timestamp: {}\n\n", meta.timestamp));

    out.push_str("## Counts\n\n");
    out.push_str(&format!(
        "correct  {correct} ({:.1}%)\npartial  {partial} ({:.1}%)\nwrong {wrong} ({:.1}%)\nunscored {unscored} ({:.1}%)\ntotal {total}\n\n",
        pct(correct), pct(partial), pct(wrong), pct(unscored)
    ));
    out.push_str(&format!("EM mean: {em_mean:.3}\nF1 mean: {f1_mean:.3}\n\n"));

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
}
