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
     - grep(pattern): exact/regex line search — use it when you know the precise string (a name, title, phrase); returns every literal match as `path:#n: line`, catching what fuzzy search buries.\n\
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
    "You answer questions over a corpus that has a PRE-BUILT REASONING GRAPH: short fact-statements, \
     each grounded to its source text, linked into reasoning chains. Here is the terrain and the \
     tools — you pick the path.\n\
     \n\
     The graph, honestly:\n\
     - A chain links reasoning WITHIN a document: from a fact to the facts that follow from it, each \
     with a `read` pointer to its source.\n\
     - Entities recur ACROSS documents. For a multi-hop question the piece you need often lives in a \
     DIFFERENT document than the one that names the question's entity — the chain in front of you may \
     not reach it on its own.\n\
     \n\
     What each tool is for (each tool's own description says when to reach for it):\n\
     - glossary(entity): look an entity up — its grounded fact plus the chain it sits in.\n\
     - reach(entity, relation): follow that relation from the entity to what it points to, crossing \
     into other documents when the link leaves this one — this is how you close a multi-hop the \
     visible chain doesn't. Pass a candidate answer as a third argument to check it is really \
     connected, not just co-mentioned.\n\
     - sql(sql): when the answer is a ranking or an extreme (which / earliest / largest / \
     first) among candidates, let SQL over the graph decide rather than guessing from prose.\n\
     - read(path, n): open a fact's source to read its exact wording.\n\
     - search(keywords): fall back to full-text when the graph is quiet.\n\
     - grep(pattern): exact/regex line search when you know the precise string — the specific name \
     or title BM25 buries under broad topic matches; returns every literal match as `path:#n: line`.\n\
     \n\
     The answer is the entity the question's OWN relation lands on directly — its immediate target, \
     one step away. Not a broader entity that merely contains that target, not the far end of a \
     DIFFERENT relation, not a name that only sits near the entity in prose. A multi-hop passes \
     THROUGH an intermediate entity, but the answer is what the OUTER relation lands on when applied \
     to that intermediate — not the intermediate itself. That outer relation also fixes what KIND of \
     thing the answer is: if your candidate is a different kind than the question asked for, you have \
     stopped at the intermediate — apply the relation to it and go on. Pin the exact relation the \
     question asks and take its direct target — reach(entity, that-relation) returns exactly that. \
     Ground the answer in the graph or its source.\n\
     Where your answer came from decides whether it is settled: an answer that came out of a graph \
     traversal (reach or sql returned it) is already grounded. But an answer you inferred by \
     READING PROSE — a name that merely appears near the entity in some text — is not settled until \
     reach confirms the connection: call reach(entity, relation, that answer); if no path comes back, \
     the two were only co-mentioned, not actually connected, so reconsider.\n\
     When several searches keep returning more background but not the answer, that is the sign you are \
     circling — widening the search finds context, not answers. The answer comes from following the \
     exact relation the question names (reach / sql), not from more search. So either name that \
     relation and traverse it, or — if the connection truly isn't there — commit your single best \
     specific answer (a name, date, place, or number); never stall on a hedge like \"cannot be determined\".\n\
     \n\
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
