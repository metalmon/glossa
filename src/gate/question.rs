//! Question-term validation for the `verify` tool's chunk-free mode.
//!
//! When `verify` is called with a `text` but NO `chunk_paths`, we don't ground an answer against
//! evidence — instead we surface the query terms that are NOT in the KB vocabulary (node labels +
//! aliases), so the agent stops building a search on jargon/wrong words verbatim and reformulates.
//! A *negative* signal: only ungrounded terms are reported. For a term that is merely misspelled we
//! add near (Levenshtein) "did you mean" candidates. Model-free, deterministic.
//! Design: docs/superpowers/specs/2026-09-27-verify-question-terms.md

use std::collections::HashSet;

/// One reported (ungrounded) term and, when it looks like a typo, near vocabulary candidates.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TermCheck {
    pub term: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<String>,
}

/// The chunk-free `verify` report: the query terms NOT found in the KB vocabulary. An empty list
/// means every content term is known. A term WITH suggestions is a probable typo; a term with none
/// is jargon / genuinely out-of-vocabulary — the agent should find a synonym, not search it verbatim.
#[derive(Debug, Clone, serde::Serialize)]
pub struct QuestionReport {
    pub terms: Vec<TermCheck>,
}

/// Max "did you mean" candidates offered per ungrounded term.
const MAX_SUGGESTIONS: usize = 3;

/// Vocabulary terms within a small Levenshtein edit distance of `term` (typo / near-spelling only),
/// ranked by distance, capped at [`MAX_SUGGESTIONS`]. PURE edit distance — deliberately NOT
/// substring or token-overlap similarity, which would offer a short vocab fragment as a "match" for
/// a longer query word (the nonsense this module's redesign removes). `max_dist = clamp(len/3, 1, 2)`
/// so a 3–4 char term needs an exact-ish match while a longer one tolerates two edits; the cheap
/// length pre-filter skips candidates that cannot possibly be within `max_dist`.
fn typo_suggestions(term: &str, vocab: &[String]) -> Vec<String> {
    let tlen = term.chars().count();
    let max_dist = (tlen / 3).clamp(1, 2);
    let mut scored: Vec<(usize, &str)> = vocab
        .iter()
        .filter(|c| c.as_str() != term)
        .filter(|c| c.chars().count().abs_diff(tlen) <= max_dist)
        .filter_map(|c| {
            let d = crate::graph::query::levenshtein(term, c);
            (d <= max_dist).then_some((d, c.as_str()))
        })
        .collect();
    scored.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for (_, c) in scored {
        if seen.insert(c) {
            out.push(c.to_string());
            if out.len() >= MAX_SUGGESTIONS {
                break;
            }
        }
    }
    out
}

/// Pure classifier: split `text` into content terms (rare-code-preserving tokenizer, `< 3` chars
/// dropped) and report ONLY those NOT present in `vocab_tokens` (the KB's label/alias words, already
/// lowercased), each with near typo candidates. Grounded terms are omitted. No stopword list, no
/// salience filter, no semantic mapping — "is this word in the KB, considering aliases?" and nothing
/// more. No graph access — the caller supplies the vocabulary, so this is unit-testable in isolation.
pub fn classify_terms(text: &str, vocab_tokens: &[String]) -> QuestionReport {
    let vocab_set: HashSet<&str> = vocab_tokens.iter().map(String::as_str).collect();
    let mut seen: HashSet<String> = HashSet::new();
    let mut terms = Vec::new();
    for tok in crate::gate::token::tokenize(text) {
        if !seen.insert(tok.clone()) {
            continue;
        }
        if vocab_set.contains(tok.as_str()) {
            continue; // grounded ⇒ not news, stay silent
        }
        let suggestions = typo_suggestions(&tok, vocab_tokens);
        terms.push(TermCheck {
            term: tok,
            suggestions,
        });
    }
    QuestionReport { terms }
}

/// Wiring: build the KB vocabulary (unique tokens of every node label + alias) from an already-open
/// graph and classify `text`'s terms against it. Aliases are first-class — a jargon word registered
/// as a node alias is grounded and stays silent. Reuses the caller's graph handle (no fresh open).
/// Empty graph ⇒ every content term reported (nothing to ground against).
pub fn check_question(
    graph: &crate::graph::store::GraphStore,
    text: &str,
) -> anyhow::Result<QuestionReport> {
    let mut vocab_tokens: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for n in graph.all_nodes()? {
        for label in std::iter::once(n.label).chain(n.aliases) {
            for tok in crate::gate::token::tokenize(&label) {
                if seen.insert(tok.clone()) {
                    vocab_tokens.push(tok);
                }
            }
        }
    }
    Ok(classify_terms(text, &vocab_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn vocab(words: &[&str]) -> Vec<String> {
        words.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn known_terms_are_not_reported() {
        let r = classify_terms("setpoint modbus", &vocab(&["setpoint", "modbus"]));
        assert!(r.terms.is_empty(), "all grounded ⇒ empty report, got {:?}", r.terms);
    }

    #[test]
    fn typo_reported_with_near_suggestion() {
        let r = classify_terms("setpiont modbus", &vocab(&["setpoint", "modbus"]));
        let by: HashMap<&str, &TermCheck> = r.terms.iter().map(|t| (t.term.as_str(), t)).collect();
        assert!(!by.contains_key("modbus"), "grounded term must not be reported");
        let sp = by.get("setpiont").expect("typo must be reported");
        assert!(
            sp.suggestions.contains(&"setpoint".to_string()),
            "typo 'setpiont' should suggest 'setpoint', got {:?}",
            sp.suggestions
        );
    }

    #[test]
    fn absent_term_flagged_without_suggestion() {
        // Nothing edit-near in the vocab ⇒ flagged as absent with no did-you-mean (find a synonym).
        let r = classify_terms("jargonword modbus", &vocab(&["setpoint", "modbus"]));
        let by: HashMap<&str, &TermCheck> = r.terms.iter().map(|t| (t.term.as_str(), t)).collect();
        let j = by.get("jargonword").expect("absent term must be reported");
        assert!(j.suggestions.is_empty(), "no near match ⇒ empty, got {:?}", j.suggestions);
    }

    #[test]
    fn short_vocab_token_not_offered_as_substring() {
        // Regression: the old substring/Jaccard ranker offered a short vocab fragment ("set") as a
        // "match" for a longer query word. Pure Levenshtein must suggest only the genuine near term.
        let r = classify_terms("setpointer", &vocab(&["set", "setpoint"]));
        let t = &r.terms[0];
        assert_eq!(t.term, "setpointer");
        assert!(t.suggestions.contains(&"setpoint".to_string()), "got {:?}", t.suggestions);
        assert!(!t.suggestions.contains(&"set".to_string()), "must NOT offer the short fragment 'set'");
    }
}
