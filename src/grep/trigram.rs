//! Regex/literal → trigram boolean query for grep prefiltering (Cox-style, char ngrams).

use super::GrepOpts;
use regex_syntax::hir::{Hir, HirKind};

const MAX_SET_SIZE: usize = 8;

/// Trigram prefilter plan. `Any` means fall back to a full chunk scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrigramQuery {
    Any,
    And(Vec<String>),
    Or(Vec<TrigramQuery>),
}

impl TrigramQuery {
    pub fn is_any(&self) -> bool {
        matches!(self, TrigramQuery::Any)
    }
}

/// Build a trigram prefilter plan for `grep`. Always sound: confirmation regex is authoritative.
pub fn trigram_plan(pattern: &str, opts: &GrepOpts) -> TrigramQuery {
    if opts.fixed {
        return literal_to_and(pattern, case_insensitive(opts, pattern));
    }
    // Mixed-case + explicit -i: case folding for grams is unsound (same rule as removed BM25 prefilter).
    if opts.ignore_case && pattern.chars().any(|c| c.is_uppercase()) {
        return TrigramQuery::Any;
    }
    let pat = pattern.to_string();
    let Ok(hir) = regex_syntax::parse(&pat) else {
        return TrigramQuery::Any;
    };
    hir_to_query(&hir, case_insensitive(opts, pattern))
}

fn case_insensitive(opts: &GrepOpts, pattern: &str) -> bool {
    opts.ignore_case || !pattern.chars().any(|c| c.is_uppercase())
}

/// All overlapping 3-character windows on `s` (always lowercased — matches `body_trigrams` indexing).
pub fn literal_trigrams(literal: &str, _case_insensitive: bool) -> Option<Vec<String>> {
    char_trigrams(&literal.to_lowercase())
}

fn char_trigrams(s: &str) -> Option<Vec<String>> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 3 {
        return None;
    }
    Some((0..=chars.len() - 3).map(|i| chars[i..i + 3].iter().collect()).collect())
}

fn literal_to_and(literal: &str, case_insensitive: bool) -> TrigramQuery {
    match literal_trigrams(literal, case_insensitive) {
        Some(grams) => TrigramQuery::And(grams),
        None => TrigramQuery::Any,
    }
}

fn merge_and(mut parts: Vec<TrigramQuery>) -> TrigramQuery {
    parts.retain(|p| !p.is_any());
    if parts.is_empty() {
        return TrigramQuery::Any;
    }
    if parts.len() == 1 {
        return parts.pop().unwrap();
    }
    let mut grams: Vec<String> = Vec::new();
    let mut nested_or = Vec::new();
    for p in parts {
        match p {
            TrigramQuery::And(mut g) => grams.append(&mut g),
            TrigramQuery::Or(o) => nested_or.push(TrigramQuery::Or(o)),
            TrigramQuery::Any => {}
        }
    }
    if !nested_or.is_empty() {
        return TrigramQuery::Or(nested_or);
    }
    grams.sort();
    grams.dedup();
    if grams.is_empty() {
        TrigramQuery::Any
    } else {
        TrigramQuery::And(grams)
    }
}

fn hir_to_query(hir: &Hir, case_insensitive: bool) -> TrigramQuery {
    match hir.kind() {
        HirKind::Empty => TrigramQuery::Any,
        HirKind::Literal(bytes) => {
            let lit = String::from_utf8_lossy(&bytes.0);
            literal_to_and(&lit, case_insensitive)
        }
        HirKind::Capture(cap) => hir_to_query(&cap.sub, case_insensitive),
        HirKind::Class(_) => TrigramQuery::Any,
        HirKind::Look(_) => TrigramQuery::Any,
        HirKind::Repetition(rep) => {
            if rep.min == 0 {
                TrigramQuery::Any
            } else {
                hir_to_query(&rep.sub, case_insensitive)
            }
        }
        HirKind::Concat(subs) => {
            if subs.len() > MAX_SET_SIZE {
                return TrigramQuery::Any;
            }
            let parts: Vec<TrigramQuery> = subs.iter().map(|h| hir_to_query(h, case_insensitive)).collect();
            merge_and(parts)
        }
        HirKind::Alternation(alts) => {
            if alts.len() > MAX_SET_SIZE {
                return TrigramQuery::Any;
            }
            let branches: Vec<TrigramQuery> = alts
                .iter()
                .map(|h| hir_to_query(h, case_insensitive))
                .filter(|q| !q.is_any())
                .collect();
            if branches.is_empty() {
                TrigramQuery::Any
            } else if branches.len() == 1 {
                branches.into_iter().next().unwrap()
            } else {
                TrigramQuery::Or(branches)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_trigrams_windows() {
        assert_eq!(
            literal_trigrams("maxTsdr", true),
            Some(vec!["max".into(), "axt".into(), "xts".into(), "tsd".into(), "sdr".into()])
        );
        assert!(literal_trigrams("ab", true).is_none());
    }

    #[test]
    fn fixed_pattern_is_and() {
        match trigram_plan("maxTsdr", &GrepOpts { fixed: true, ..Default::default() }) {
            TrigramQuery::And(g) => assert!(g.contains(&"max".to_string())),
            _ => panic!("expected And"),
        }
    }

    #[test]
    fn alternation_is_or() {
        match trigram_plan("регистрация|компонент", &GrepOpts { ignore_case: true, ..Default::default() }) {
            TrigramQuery::Or(branches) => assert_eq!(branches.len(), 2),
            q => panic!("expected Or, got {q:?}"),
        }
    }

    #[test]
    fn short_pattern_is_any() {
        assert!(trigram_plan("ab", &GrepOpts { fixed: true, ..Default::default() }).is_any());
    }
}
