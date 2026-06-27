use crate::backend::AgentBackend;
use crate::{corpus, dataset, score, trace_read};
use glossa::trace::now_ms;
use serde::Serialize;
use std::path::Path;

#[derive(Serialize)]
pub struct Row {
    pub id: String,
    pub question: String,
    pub gold: String,
    pub pred: String,
    pub em: bool,
    pub f1: f32,
    pub retrieval_recall: f32,
    pub failed: Option<String>,
    /// Per-question glossa tool-call trace (search queries + ranked results + reads). Captured for
    /// failure analysis; omitted from the JSON when empty (e.g. mock backend).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transcript: Vec<glossa::trace::TraceEntry>,
    pub recall_at_5: f32,
    pub recall_at_10: f32,
    pub recall_at_20: f32,
    pub mrr: f32,
}

#[derive(Serialize)]
pub struct Report {
    pub backend: String,
    pub rows: Vec<Row>,
    pub em_mean: f32,
    pub f1_mean: f32,
    pub recall_mean: f32,
    pub recall_at_5_mean: f32,
    pub recall_at_10_mean: f32,
    pub recall_at_20_mean: f32,
    pub mrr_mean: f32,
}

pub fn run_eval(
    dataset_path: &Path,
    backend: &dyn AgentBackend,
    backend_name: &str,
    limit: usize,
    kb_bin: &str,
    work: &Path,
    fullwiki: Option<&Path>,
) -> anyhow::Result<Report> {
    let json = std::fs::read_to_string(dataset_path)?;
    let mut questions = dataset::parse_hotpot(&json)?;
    if limit > 0 && questions.len() > limit {
        questions.truncate(limit);
    }
    let eff_work = fullwiki.unwrap_or(work);
    let rows: Vec<Row> = questions
        .iter()
        .map(|q| eval_one(backend, q, kb_bin, eff_work, fullwiki.is_some()))
        .collect();
    let n = rows.len().max(1) as f32;
    let em_mean = rows.iter().filter(|r| r.em).count() as f32 / n;
    let f1_mean = rows.iter().map(|r| r.f1).sum::<f32>() / n;
    let recall_mean = rows.iter().map(|r| r.retrieval_recall).sum::<f32>() / n;
    let recall_at_5_mean = rows.iter().map(|r| r.recall_at_5).sum::<f32>() / n;
    let recall_at_10_mean = rows.iter().map(|r| r.recall_at_10).sum::<f32>() / n;
    let recall_at_20_mean = rows.iter().map(|r| r.recall_at_20).sum::<f32>() / n;
    let mrr_mean = rows.iter().map(|r| r.mrr).sum::<f32>() / n;
    Ok(Report {
        backend: backend_name.to_string(), rows, em_mean, f1_mean, recall_mean,
        recall_at_5_mean, recall_at_10_mean, recall_at_20_mean, mrr_mean,
    })
}

fn eval_one(backend: &dyn AgentBackend, q: &dataset::Question, kb_bin: &str, work: &Path, fullwiki: bool) -> Row {
    let base = Row {
        id: q.id.clone(), question: q.question.clone(), gold: q.answer.clone(),
        pred: String::new(), em: false, f1: 0.0, retrieval_recall: 0.0, failed: None,
        transcript: Vec::new(),
        recall_at_5: 0.0, recall_at_10: 0.0, recall_at_20: 0.0, mrr: 0.0,
    };
    // In fullwiki mode the shared corpus is pre-built; do NOT write/index/clear per question.
    if backend.needs_corpus() && !fullwiki {
        if let Err(e) = corpus::write_corpus(work, q).and_then(|_| corpus::index(work, kb_bin)) {
            return Row { failed: Some(format!("corpus: {e}")), ..base };
        }
    }
    let t0 = now_ms();
    let pred = match backend.answer(work, q) {
        Ok(p) => p,
        Err(e) => return Row { failed: Some(format!("backend: {e}")), ..base },
    };
    let t1 = now_ms();
    let entries = if backend.needs_corpus() {
        let dir = work.join(".glossa").join("traces");
        trace_read::read_window(&dir, t0, t1).unwrap_or_default()
    } else {
        Vec::new()
    };
    let recall = if backend.needs_corpus() {
        score::retrieval_recall(
            &trace_read::seen_files(&entries),
            &trace_read::seen_locations(&entries),
            &q.supporting_titles,
        )
    } else {
        0.0
    };
    let titles = score::ranked_titles(&entries);
    Row {
        em: score::exact_match(&pred, &q.answer),
        f1: score::token_f1(&pred, &q.answer),
        retrieval_recall: recall,
        recall_at_5: score::recall_at_k(&titles, &q.supporting_titles, 5),
        recall_at_10: score::recall_at_k(&titles, &q.supporting_titles, 10),
        recall_at_20: score::recall_at_k(&titles, &q.supporting_titles, 20),
        mrr: score::mrr(&titles, &q.supporting_titles),
        pred,
        transcript: entries,
        ..base
    }
}

#[cfg(test)]
mod fullwiki_tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use std::collections::HashMap;

    #[test]
    fn fullwiki_mode_does_not_build_per_question_corpus() {
        let dir = tempfile::tempdir().unwrap();
        let corpus = dir.path().join("wiki");
        std::fs::create_dir_all(&corpus).unwrap();
        // a sentinel a per-question write_corpus clear would remove
        std::fs::write(corpus.join("Sentinel.md"), b"# Sentinel\nkeep me\n").unwrap();

        let dataset = dir.path().join("d.json");
        std::fs::write(&dataset, br#"[{"_id":"q1","question":"Who?","answer":"Bob",
            "context":[["Bob Page",["b."]]],"supporting_facts":[["Bob Page",0]]}]"#).unwrap();

        let be = MockBackend { canned: HashMap::new() };
        let report = run_eval(&dataset, &be, "mock", 0, "kb", dir.path(), Some(corpus.as_path())).unwrap();

        assert_eq!(report.rows.len(), 1);
        assert!(corpus.join("Sentinel.md").exists(), "fullwiki must NOT clear the shared corpus");
        assert_eq!(report.recall_at_5_mean, 0.0); // empty mock transcript
    }
}
