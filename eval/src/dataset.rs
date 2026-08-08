use anyhow::Context;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq)]
pub struct Paragraph {
    pub title: String,
    pub sentences: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub id: String,
    pub question: String,
    pub answer: String,
    /// Alternative gold surface forms (MuSiQue ships `answer_aliases`); empty for datasets
    /// without them. Used so EM does not penalize a correct-but-differently-spelled answer.
    pub answer_aliases: Vec<String>,
    pub paragraphs: Vec<Paragraph>,
    pub supporting_titles: Vec<String>,
}

#[derive(Deserialize)]
struct RawItem {
    #[serde(rename = "_id")]
    id: String,
    question: String,
    answer: String,
    #[serde(default)]
    context: Vec<(String, Vec<String>)>,
    #[serde(default)]
    supporting_facts: Vec<(String, i64)>,
}

pub fn sanitize_title(title: &str) -> String {
    let mut s: String = title
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    s = s.trim().replace(' ', "_");
    if s.is_empty() {
        s.push_str("untitled");
    }
    s
}

pub fn parse_hotpot(json: &str) -> anyhow::Result<Vec<Question>> {
    let raw: Vec<RawItem> = serde_json::from_str(json).context("parse hotpot json")?;
    Ok(raw
        .into_iter()
        .map(|r| {
            let mut titles: Vec<String> = r.supporting_facts.into_iter().map(|(t, _)| t).collect();
            titles.sort();
            titles.dedup();
            Question {
                id: r.id,
                question: r.question,
                answer: r.answer,
                answer_aliases: Vec::new(),
                paragraphs: r
                    .context
                    .into_iter()
                    .map(|(title, sentences)| Paragraph { title, sentences })
                    .collect(),
                supporting_titles: titles,
            }
        })
        .collect())
}

#[derive(Deserialize)]
struct RawMusique {
    id: String,
    question: String,
    answer: String,
    #[serde(default)]
    answer_aliases: Vec<String>,
    #[serde(default)]
    paragraphs: Vec<RawMusiqueParagraph>,
}

#[derive(Deserialize)]
struct RawMusiqueParagraph {
    #[serde(default)]
    title: String,
    #[serde(default)]
    paragraph_text: String,
    #[serde(default)]
    is_supporting: bool,
}

/// Parse MuSiQue (`.jsonl`, one record per line) into the shared `Question` shape. Each MuSiQue
/// paragraph is one text block, so it becomes a single-sentence `Paragraph`; `supporting_titles`
/// are the titles of the `is_supporting` paragraphs (sorted + deduped, mirroring `parse_hotpot`).
/// `answer_aliases` are carried for fair EM. Blank lines are skipped.
pub fn parse_musique(jsonl: &str) -> anyhow::Result<Vec<Question>> {
    let mut out = Vec::new();
    for (i, line) in jsonl.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let r: RawMusique =
            serde_json::from_str(line).with_context(|| format!("parse musique line {}", i + 1))?;
        let mut titles: Vec<String> = r
            .paragraphs
            .iter()
            .filter(|p| p.is_supporting)
            .map(|p| p.title.clone())
            .collect();
        titles.sort();
        titles.dedup();
        out.push(Question {
            id: r.id,
            question: r.question,
            answer: r.answer,
            answer_aliases: r.answer_aliases,
            paragraphs: r
                .paragraphs
                .into_iter()
                .map(|p| Paragraph {
                    title: p.title,
                    sentences: vec![p.paragraph_text],
                })
                .collect(),
            supporting_titles: titles,
        });
    }
    Ok(out)
}

/// Which on-disk dataset shape `run_eval` should parse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DatasetFormat {
    /// HotpotQA JSON array (may carry inline `context` → per-question corpus).
    Hotpot,
    /// Raw MuSiQue `.jsonl` (records carry `paragraphs`; supporting titles derived from them).
    Musique,
    /// Flat questions-file `.jsonl` for fixed-corpus runs (no paragraphs; titles given directly).
    Questions,
}

/// Dispatch to the loader for `format`.
pub fn parse_dataset(format: DatasetFormat, text: &str) -> anyhow::Result<Vec<Question>> {
    match format {
        DatasetFormat::Hotpot => parse_hotpot(text),
        DatasetFormat::Musique => parse_musique(text),
        DatasetFormat::Questions => parse_questions(text),
    }
}

#[derive(Deserialize)]
struct RawQuestions {
    id: String,
    question: String,
    answer: String,
    #[serde(default)]
    answer_aliases: Vec<String>,
    #[serde(default)]
    supporting_titles: Vec<String>,
}

/// Parse a flat questions-file (`.jsonl`, one record per line) for **fixed-corpus** runs: each
/// record is `{id, question, answer, answer_aliases?, supporting_titles?}` and is answered against
/// a pre-built shared corpus (`--fullwiki`), so it carries NO paragraphs — the runner never builds
/// a per-question corpus. `supporting_titles` are taken directly (used for retrieval recall);
/// `answer_aliases` are carried for alias-aware EM/F1. Blank lines are skipped.
pub fn parse_questions(jsonl: &str) -> anyhow::Result<Vec<Question>> {
    let mut out = Vec::new();
    for (i, line) in jsonl.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let r: RawQuestions = serde_json::from_str(line)
            .with_context(|| format!("parse questions line {}", i + 1))?;
        out.push(Question {
            id: r.id,
            question: r.question,
            answer: r.answer,
            answer_aliases: r.answer_aliases,
            paragraphs: Vec::new(),
            supporting_titles: r.supporting_titles,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"[
      {"_id":"q1","question":"Who?","answer":"Bob",
       "context":[["Alice",["s1.","s2."]],["Bob Page",["b1."]]],
       "supporting_facts":[["Bob Page",0],["Bob Page",0],["Alice",1]]}
    ]"#;

    #[test]
    fn parses_glossa_train_without_hotpot_context() {
        let json = r#"[
          {"_id":"syn-1","question":"Q?","answer":"A","source":"doc.htm:#1"}
        ]"#;
        let qs = parse_hotpot(json).unwrap();
        assert_eq!(qs.len(), 1);
        assert_eq!(qs[0].id, "syn-1");
        assert!(qs[0].paragraphs.is_empty());
        assert!(qs[0].supporting_titles.is_empty());
    }

    #[test]
    fn parses_questions_and_dedups_supporting_titles() {
        let qs = parse_hotpot(SAMPLE).unwrap();
        assert_eq!(qs.len(), 1);
        assert_eq!(qs[0].answer, "Bob");
        assert_eq!(qs[0].paragraphs.len(), 2);
        assert_eq!(
            qs[0].supporting_titles,
            vec!["Alice".to_string(), "Bob Page".to_string()]
        );
    }

    #[test]
    fn sanitize_title_is_fs_safe() {
        assert_eq!(sanitize_title("Bob Page"), "Bob_Page");
        assert_eq!(sanitize_title("A/B: C?"), "A_B__C_");
    }

    #[test]
    fn parses_musique_jsonl_with_aliases_and_supporting() {
        let jsonl = concat!(
            r#"{"id":"2hop__1_2","question":"Q hop?","answer":"Final","answer_aliases":["Final Ans","F."],"paragraphs":[{"idx":0,"title":"Doc A","paragraph_text":"A text.","is_supporting":true},{"idx":1,"title":"Doc B","paragraph_text":"B text.","is_supporting":false},{"idx":2,"title":"Doc C","paragraph_text":"C text.","is_supporting":true}]}"#,
            "\n",
            r#"{"id":"2hop__3_4","question":"Q2?","answer":"Ans2","paragraphs":[{"idx":0,"title":"Doc D","paragraph_text":"D text.","is_supporting":true}]}"#,
            "\n"
        );
        let qs = parse_musique(jsonl).unwrap();
        assert_eq!(qs.len(), 2);
        assert_eq!(qs[0].id, "2hop__1_2");
        assert_eq!(qs[0].answer, "Final");
        assert_eq!(
            qs[0].answer_aliases,
            vec!["Final Ans".to_string(), "F.".to_string()]
        );
        assert_eq!(qs[0].paragraphs.len(), 3);
        assert_eq!(qs[0].paragraphs[0].title, "Doc A");
        assert_eq!(qs[0].paragraphs[0].sentences, vec!["A text.".to_string()]);
        // Only the is_supporting paragraphs, sorted + deduped.
        assert_eq!(
            qs[0].supporting_titles,
            vec!["Doc A".to_string(), "Doc C".to_string()]
        );
        // Second record: no aliases; single supporting doc.
        assert!(qs[1].answer_aliases.is_empty());
        assert_eq!(qs[1].supporting_titles, vec!["Doc D".to_string()]);
    }

    #[test]
    fn parses_flat_questions_file_for_fixed_corpus_mode() {
        // The questions-file shape used against a pre-built shared corpus: supporting_titles are
        // given DIRECTLY (not derived from inline paragraphs), and there are NO paragraphs — so
        // the runner never builds a per-question corpus. Aliases are carried for fair EM.
        let jsonl = concat!(
            r#"{"id":"2hop__1_2","question":"Q hop?","answer":"Final","answer_aliases":["Final Ans","F."],"supporting_titles":["Doc A","Doc C"]}"#,
            "\n\n", // blank line must be skipped
            r#"{"id":"2hop__3_4","question":"Q2?","answer":"Ans2"}"#,
            "\n"
        );
        let qs = parse_questions(jsonl).unwrap();
        assert_eq!(qs.len(), 2);
        assert_eq!(qs[0].id, "2hop__1_2");
        assert_eq!(qs[0].answer, "Final");
        assert_eq!(
            qs[0].answer_aliases,
            vec!["Final Ans".to_string(), "F.".to_string()]
        );
        // supporting_titles come straight from the field...
        assert_eq!(
            qs[0].supporting_titles,
            vec!["Doc A".to_string(), "Doc C".to_string()]
        );
        // ...and NO per-question corpus is implied.
        assert!(qs[0].paragraphs.is_empty());
        // Second record: missing optional fields default to empty.
        assert!(qs[1].answer_aliases.is_empty());
        assert!(qs[1].supporting_titles.is_empty());
        assert!(qs[1].paragraphs.is_empty());
    }
}
