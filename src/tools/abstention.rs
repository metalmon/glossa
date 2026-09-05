use std::collections::BTreeSet;

pub struct AbsentTerm { pub term: String }

const STOP: &[&str] = &["который","которая","которые","нужно","можно","будет","этот","этих",
    "такой","такие","через","после","перед","около","между","например","вопрос","когда","чтобы"];

pub fn coverage_uncovered(question: &str, covered: &dyn Fn(&str) -> bool, _k: usize) -> Vec<AbsentTerm> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for tok in question.split(|c: char| !c.is_alphabetic()) {
        if tok.chars().count() < 5 { continue; }
        let low = tok.to_lowercase();
        if STOP.contains(&low.as_str()) || !seen.insert(low.clone()) { continue; }
        if !covered(tok) { out.push(AbsentTerm { term: tok.to_string() }); }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_flags_only_uncovered_distinctive_terms() {
        // "covered" = present in this fake corpus set (case-insensitive substring of any entry)
        let corpus = ["настройка profibus maxtsdr", "модули ивк"];
        let covered = |t: &str| corpus.iter().any(|c| c.contains(&t.to_lowercase()));
        let absent = coverage_uncovered("как настроить maxTsdr для неведомыйтермин", &covered, 1);
        let terms: Vec<&str> = absent.iter().map(|a| a.term.as_str()).collect();
        assert!(terms.iter().any(|t| t.eq_ignore_ascii_case("неведомыйтермин")));
        assert!(!terms.iter().any(|t| t.eq_ignore_ascii_case("maxTsdr")), "covered term not flagged");
        assert!(!terms.iter().any(|t| t.chars().count() < 5), "short/stopwords excluded");
    }
}
