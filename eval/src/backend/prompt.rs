use crate::dataset::Question;

/// Role + search strategy + output contract. Sent as the `system` message for chat backends
/// (this also overrides any system prompt configured on the model server), and folded into the
/// combined single-string prompt for line-oriented CLI agents.
pub fn system_prompt() -> &'static str {
    "You answer a question using glossa, a document-search tool over MCP with two tools:\n\
     - search(query): full-text BM25 search. Pass short KEYWORDS, not a sentence. Morphology-aware.\n\
     - read(path, location): open a result to read its full text.\n\
     Strategy (follow it):\n\
     1. Break the question into key entities/terms.\n\
     2. Call search SEVERAL times with different formulations (synonyms, broader/narrower). One query is rarely enough.\n\
     3. For multi-hop questions, find the bridge entity first, then search again using what you found.\n\
     4. Open the most relevant results with read before answering; ground every claim in the text.\n\
     5. If nothing is found after several different queries, give your best answer anyway.\n\
     Output ONLY your final answer on a single line beginning with `ANSWER:` — keep it as short as possible (a name, entity, number, or yes/no)."
}

/// The per-question user turn.
pub fn user_prompt(q: &Question) -> String {
    format!("Question: {}", q.question)
}

/// Combined single-string prompt for CLI agents that take one prompt argument.
pub fn build_prompt(q: &Question) -> String {
    format!("{}\n\n{}", system_prompt(), user_prompt(q))
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

    fn q() -> Question {
        Question { id: "x".into(), question: "Who?".into(), answer: "".into(), paragraphs: vec![], supporting_titles: vec![] }
    }

    #[test]
    fn parse_answer_takes_after_marker() {
        assert_eq!(parse_answer("thinking...\nANSWER: Bob Page\n"), "Bob Page");
        assert_eq!(parse_answer("ANSWER:  42 "), "42");
        assert_eq!(parse_answer("no marker here"), "no marker here");
    }

    #[test]
    fn system_prompt_states_tools_and_marker() {
        let s = system_prompt();
        assert!(s.contains("search") && s.contains("read") && s.contains("ANSWER:"));
    }

    #[test]
    fn user_prompt_is_just_the_question() {
        assert_eq!(user_prompt(&q()), "Question: Who?");
    }

    #[test]
    fn build_prompt_includes_question_and_marker() {
        let p = build_prompt(&q());
        assert!(p.contains("Who?") && p.contains("ANSWER:"));
    }
}
