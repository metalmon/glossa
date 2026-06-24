use crate::dataset::Question;

pub fn build_prompt(q: &Question) -> String {
    format!(
        "You are answering a question using a document search tool (glossa MCP: `search`, `read`).\n\
         Search the indexed corpus, read what you need, then answer.\n\
         Output ONLY your final answer on a single line beginning with `ANSWER:`.\n\n\
         Question: {}",
        q.question
    )
}

/// Extract the answer after the last `ANSWER:` marker; if absent, the trimmed whole output.
pub fn parse_answer(model_output: &str) -> String {
    if let Some(idx) = model_output.rfind("ANSWER:") {
        model_output[idx + "ANSWER:".len()..].trim().lines().next().unwrap_or("").trim().to_string()
    } else {
        model_output.trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Question;

    #[test]
    fn parse_answer_takes_after_marker() {
        assert_eq!(parse_answer("thinking...\nANSWER: Bob Page\n"), "Bob Page");
        assert_eq!(parse_answer("ANSWER:  42 "), "42");
        assert_eq!(parse_answer("no marker here"), "no marker here");
    }

    #[test]
    fn build_prompt_includes_question_and_marker() {
        let q = Question { id: "x".into(), question: "Who?".into(), answer: "".into(), paragraphs: vec![], supporting_titles: vec![] };
        let p = build_prompt(&q);
        assert!(p.contains("Who?") && p.contains("ANSWER:"));
    }
}
