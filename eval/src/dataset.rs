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
    pub paragraphs: Vec<Paragraph>,
    pub supporting_titles: Vec<String>,
}

#[derive(Deserialize)]
struct RawItem {
    #[serde(rename = "_id")]
    id: String,
    question: String,
    answer: String,
    context: Vec<(String, Vec<String>)>,
    supporting_facts: Vec<(String, i64)>,
}

pub fn sanitize_title(title: &str) -> String {
    let mut s: String = title
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' { c } else { '_' })
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
                paragraphs: r.context.into_iter().map(|(title, sentences)| Paragraph { title, sentences }).collect(),
                supporting_titles: titles,
            }
        })
        .collect())
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
    fn parses_questions_and_dedups_supporting_titles() {
        let qs = parse_hotpot(SAMPLE).unwrap();
        assert_eq!(qs.len(), 1);
        assert_eq!(qs[0].answer, "Bob");
        assert_eq!(qs[0].paragraphs.len(), 2);
        assert_eq!(qs[0].supporting_titles, vec!["Alice".to_string(), "Bob Page".to_string()]);
    }

    #[test]
    fn sanitize_title_is_fs_safe() {
        assert_eq!(sanitize_title("Bob Page"), "Bob_Page");
        assert_eq!(sanitize_title("A/B: C?"), "A_B__C_");
    }
}
