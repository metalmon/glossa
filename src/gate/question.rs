//! Question-term validation for the `verify` tool's chunk-free mode.
//!
//! When `verify` is called with a `text` but NO `chunk_paths`, we don't ground an answer against
//! evidence — instead we surface the query terms whose Snowball STEM is not in the KB vocabulary
//! (node labels + aliases), so the agent stops building a search on jargon/wrong words verbatim and
//! reformulates. Stem-based so an inflected query word matches the corpus's form of the same word
//! (crucial for Russian's rich inflection) instead of being flagged as absent. A *negative* signal:
//! only ungrounded terms are reported. For a term that is merely misspelled we add near (Levenshtein)
//! "did you mean" candidates. Model-free, deterministic.
//! Design: docs/superpowers/specs/2026-09-27-verify-question-terms.md

use rust_stemmers::{Algorithm, Stemmer as RsStemmer};
use std::collections::HashSet;

/// Snowball stemmers for stem-based grounding: a query term is present when its stem matches a
/// vocabulary token's stem, so inflected forms (plurals, cases) are not flagged as missing. The
/// algorithm is chosen per token by script — a Cyrillic character ⇒ Russian, else English — the same
/// rule the search index uses (`crate::index::multilang::script_detector`). Snowball is a stemmer,
/// not a full morphological analyzer, so it is deliberately biased toward grounding: it may collapse
/// a derivation as well as an inflection, which for this negative signal means one fewer false "not
/// in the corpus" flag — the right direction for a tool whose job is to surface genuine jargon/typos.
struct Stemmers {
    ru: RsStemmer,
    en: RsStemmer,
}

impl Stemmers {
    fn new() -> Self {
        Self {
            ru: RsStemmer::create(Algorithm::Russian),
            en: RsStemmer::create(Algorithm::English),
        }
    }

    fn stem(&self, tok: &str) -> String {
        let cyrillic = tok.chars().any(|c| ('\u{0400}'..='\u{04FF}').contains(&c));
        if cyrillic {
            self.ru.stem(tok)
        } else {
            self.en.stem(tok)
        }
        .into_owned()
    }
}

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
/// dropped) and report ONLY those whose Snowball STEM is not among the vocabulary's stems (the KB's
/// label/alias words, already lowercased), each with near typo candidates. Stem comparison so an
/// inflected query word (a plural or a non-nominative case of the corpus's form) is grounded, not
/// flagged.
/// Grounded terms are omitted. No stopword list, no salience filter, no semantic mapping — "is this
/// word (in any inflection) in the KB, considering aliases?" and nothing more. No graph access — the
/// caller supplies the vocabulary, so this is unit-testable in isolation.
pub fn classify_terms(text: &str, vocab_tokens: &[String]) -> QuestionReport {
    let stemmers = Stemmers::new();
    let vocab_stems: HashSet<String> = vocab_tokens.iter().map(|t| stemmers.stem(t)).collect();
    let mut seen: HashSet<String> = HashSet::new();
    let mut terms = Vec::new();
    for tok in crate::gate::token::tokenize(text) {
        if !seen.insert(tok.clone()) {
            continue;
        }
        if vocab_stems.contains(&stemmers.stem(&tok)) {
            continue; // grounded by stem ⇒ not news, stay silent
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
        assert!(
            r.terms.is_empty(),
            "all grounded ⇒ empty report, got {:?}",
            r.terms
        );
    }

    #[test]
    fn inflected_forms_are_grounded_by_stem() {
        // A query word in a different inflection than the vocabulary form is the SAME word and must
        // stay silent — grounding compares Snowball stems, not exact tokens. Tested with English
        // morphology (the repo is English-only); the production win is Russian, where the per-token
        // Cyrillic->Russian algorithm collapses a noun's case/number forms to one stem the same way.
        let r = classify_terms("running dogs", &vocab(&["run", "dog"]));
        assert!(
            r.terms.is_empty(),
            "inflected forms share a stem with the vocab ⇒ grounded, got {:?}",
            r.terms
        );
    }

    #[test]
    fn typo_reported_with_near_suggestion() {
        let r = classify_terms("setpiont modbus", &vocab(&["setpoint", "modbus"]));
        let by: HashMap<&str, &TermCheck> = r.terms.iter().map(|t| (t.term.as_str(), t)).collect();
        assert!(
            !by.contains_key("modbus"),
            "grounded term must not be reported"
        );
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
        assert!(
            j.suggestions.is_empty(),
            "no near match ⇒ empty, got {:?}",
            j.suggestions
        );
    }

    #[test]
    fn derivation_grounds_by_stem_and_short_fragment_never_suggested() {
        // A derivation that shares a vocab word's stem is grounded — Snowball is intentionally
        // stem-biased, so "setpointer" (stem "setpoint") stays silent rather than being flagged.
        let grounded = classify_terms("setpointer", &vocab(&["set", "setpoint"]));
        assert!(
            grounded.terms.is_empty(),
            "a derivation sharing a vocab stem grounds, got {:?}",
            grounded.terms
        );
        // A genuine typo (stem differs) is still reported, and suggests the real near term via pure
        // Levenshtein — never the short fragment "set" (the nonsense the old substring ranker gave).
        let typo = classify_terms("setpiont", &vocab(&["set", "setpoint"]));
        let t = &typo.terms[0];
        assert_eq!(t.term, "setpiont");
        assert!(
            t.suggestions.contains(&"setpoint".to_string()),
            "got {:?}",
            t.suggestions
        );
        assert!(
            !t.suggestions.contains(&"set".to_string()),
            "must NOT offer the short fragment 'set'"
        );
    }
}
