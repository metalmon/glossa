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
}

#[derive(Serialize)]
pub struct Report {
    pub backend: String,
    pub rows: Vec<Row>,
    pub em_mean: f32,
    pub f1_mean: f32,
    pub recall_mean: f32,
}

pub fn run_eval(
    dataset_path: &Path,
    backend: &dyn AgentBackend,
    backend_name: &str,
    limit: usize,
    kb_bin: &str,
    work: &Path,
) -> anyhow::Result<Report> {
    let json = std::fs::read_to_string(dataset_path)?;
    let mut questions = dataset::parse_hotpot(&json)?;
    if limit > 0 && questions.len() > limit {
        questions.truncate(limit);
    }
    let rows: Vec<Row> = questions.iter().map(|q| eval_one(backend, q, kb_bin, work)).collect();
    let n = rows.len().max(1) as f32;
    let em_mean = rows.iter().filter(|r| r.em).count() as f32 / n;
    let f1_mean = rows.iter().map(|r| r.f1).sum::<f32>() / n;
    let recall_mean = rows.iter().map(|r| r.retrieval_recall).sum::<f32>() / n;
    Ok(Report { backend: backend_name.to_string(), rows, em_mean, f1_mean, recall_mean })
}

fn eval_one(backend: &dyn AgentBackend, q: &dataset::Question, kb_bin: &str, work: &Path) -> Row {
    let base = Row {
        id: q.id.clone(), question: q.question.clone(), gold: q.answer.clone(),
        pred: String::new(), em: false, f1: 0.0, retrieval_recall: 0.0, failed: None,
    };
    if backend.needs_corpus() {
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
    let recall = if backend.needs_corpus() {
        let dir = work.join(".glossa").join("traces");
        let entries = trace_read::read_window(&dir, t0, t1).unwrap_or_default();
        score::retrieval_recall(&trace_read::seen_files(&entries), &q.supporting_titles)
    } else {
        0.0
    };
    Row {
        em: score::exact_match(&pred, &q.answer),
        f1: score::token_f1(&pred, &q.answer),
        retrieval_recall: recall,
        pred,
        ..base
    }
}
