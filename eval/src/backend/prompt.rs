use crate::dataset::Question;

/// Role + search strategy + output contract. Sent as the `system` message for chat backends
/// (this also overrides any system prompt configured on the model server), and folded into the
/// combined single-string prompt for line-oriented CLI agents.
///
/// `graph_on` selects the strategy: the flat baseline (search/read only) or the graph-first
/// variant that teaches the model to FOLLOW the pre-built reasoning graph. The graph-first text is
/// part of the graph-ON *treatment* (per the eval spec: the prompt that teaches graph use is the
/// real variable) — the OFF arm never sees it, keeping the ablation clean.
pub fn system_prompt(graph_on: bool) -> &'static str {
    if graph_on {
        GRAPH_SYSTEM_PROMPT
    } else {
        FLAT_SYSTEM_PROMPT
    }
}

const FLAT_SYSTEM_PROMPT: &str =
    "You answer a question using glossa, a document-search tool over MCP with two tools:\n\
     - search(query): full-text BM25 search. Pass short KEYWORDS, not a sentence. Morphology-aware.\n\
     - read(path, location): open a result to read its full text.\n\
     Strategy (follow it):\n\
     1. Break the question into key entities/terms.\n\
     2. Call search SEVERAL times with different formulations (synonyms, broader/narrower). One query is rarely enough.\n\
     3. For multi-hop questions, find the bridge entity first, then search again using what you found.\n\
     4. Open the most relevant results with read before answering; ground every claim in the text.\n\
     5. If nothing is found after several different queries, give your best answer anyway.\n\
     ANSWER FORMAT (strict — graded by exact match):\n\
     - Output ONLY one final line beginning with `ANSWER:` and nothing after it.\n\
     - The answer must be the SHORTEST exact span that answers the question — usually 1-4 words (a name, place, date, number).\n\
     - No explanation, no full sentence, no extra context, do not restate the question, no trailing punctuation.\n\
     - For yes/no questions answer exactly `yes` or `no`.\n\
     - Examples: `ANSWER: Chief of Protocol` · `ANSWER: yes` · `ANSWER: Animorphs` · `ANSWER: 1972`.";

const GRAPH_SYSTEM_PROMPT: &str =
    "You answer a question using a corpus that has a PRE-BUILT REASONING GRAPH: entities grounded to \
     source text and linked by typed relations (located-in, part-of, created-by, family-of, \
     member-of). A strong model already wired the multi-hop connections — follow them to the answer.\n\
     Path to the answer:\n\
     1. Take the entity in the question and call glossary on it. It prints that entity with its relations as a chain, e.g. `Senica -> LOCATED_IN -> Senica District`.\n\
     2. The answer is usually a neighbour in that chain. Match the question to the relation and read that node: located-in / part-of -> the place it sits in; created-by -> whoever made / voiced / founded it; family-of -> the relative. Answer at the level asked — the district when it asks for a district.\n\
     3. For an answer one more hop away, call neighbors(node id) or path(from id, to id) and follow the typed edges.\n\
     4. Confirm on the source with read(path, n), then answer.\n\
     5. When glossary shows only `[Section]` / `[Document]` lines (no typed relations), the graph is thin here — reach the answer with search(keywords) + read instead. Answer your best even if the corpus is quiet.\n\
     Reply with one line: `ANSWER: <shortest exact span>` — a name, place, date, or number, usually 1-4 words (or `yes` / `no`). Examples: `ANSWER: Chief of Protocol` · `ANSWER: 1972`.";

/// The per-question user turn.
pub fn user_prompt(q: &Question) -> String {
    format!("Question: {}", q.question)
}

/// Combined single-string prompt for CLI agents that take one prompt argument.
pub fn build_prompt(q: &Question, graph_on: bool) -> String {
    format!("{}\n\n{}", system_prompt(graph_on), user_prompt(q))
}

/// Extract the answer after the last `ANSWER:` marker; if absent, the trimmed whole output.
pub fn parse_answer(model_output: &str) -> String {
    if let Some(idx) = model_output.rfind("ANSWER:") {
        model_output[idx + "ANSWER:".len()..]
            .trim()
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .to_string()
    } else {
        model_output.trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Question;

    fn q() -> Question {
        Question {
            id: "x".into(),
            question: "Who?".into(),
            answer: "".into(),
            paragraphs: vec![],
            answer_aliases: vec![],
            supporting_titles: vec![],
        }
    }

    #[test]
    fn parse_answer_takes_after_marker() {
        assert_eq!(parse_answer("thinking...\nANSWER: Bob Page\n"), "Bob Page");
        assert_eq!(parse_answer("ANSWER:  42 "), "42");
        assert_eq!(parse_answer("no marker here"), "no marker here");
    }

    #[test]
    fn system_prompt_states_tools_and_marker() {
        let s = system_prompt(false);
        assert!(s.contains("search") && s.contains("read") && s.contains("ANSWER:"));
    }

    #[test]
    fn graph_prompt_teaches_graph_first_and_flat_does_not() {
        // The graph-first strategy is part of the graph-ON treatment; the OFF arm must never see
        // it (else the arms differ by prompt, not just graph presence — the ablation is confounded).
        let on = system_prompt(true);
        assert!(
            on.contains("glossary") && on.to_lowercase().contains("graph"),
            "graph-ON prompt must teach the graph tools"
        );
        let off = system_prompt(false);
        assert!(
            !off.contains("glossary") && !off.to_lowercase().contains("graph"),
            "graph-OFF prompt must stay flat (no graph leakage)"
        );
        // Both still enforce the same answer contract.
        assert!(on.contains("ANSWER:") && off.contains("ANSWER:"));
    }

    #[test]
    fn user_prompt_is_just_the_question() {
        assert_eq!(user_prompt(&q()), "Question: Who?");
    }

    #[test]
    fn build_prompt_includes_question_and_marker() {
        let p = build_prompt(&q(), false);
        assert!(p.contains("Who?") && p.contains("ANSWER:"));
    }
}
