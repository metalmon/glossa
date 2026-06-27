use crate::dataset::sanitize_title;

/// HotpotQA answer normalization: lowercase, drop articles, drop punctuation, collapse whitespace.
pub fn normalize(s: &str) -> String {
    let lower = s.to_lowercase();
    let no_punct: String = lower.chars().map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' }).collect();
    no_punct
        .split_whitespace()
        .filter(|w| !matches!(*w, "a" | "an" | "the"))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn exact_match(pred: &str, gold: &str) -> bool {
    normalize(pred) == normalize(gold)
}

pub fn token_f1(pred: &str, gold: &str) -> f32 {
    let p: Vec<String> = normalize(pred).split_whitespace().map(|s| s.to_string()).collect();
    let g: Vec<String> = normalize(gold).split_whitespace().map(|s| s.to_string()).collect();
    if p.is_empty() || g.is_empty() {
        return if p.is_empty() && g.is_empty() { 1.0 } else { 0.0 };
    }
    let mut shared = 0usize;
    let mut gleft = g.clone();
    for tok in &p {
        if let Some(pos) = gleft.iter().position(|x| x == tok) {
            shared += 1;
            gleft.remove(pos);
        }
    }
    if shared == 0 {
        return 0.0;
    }
    let precision = shared as f32 / p.len() as f32;
    let recall = shared as f32 / g.len() as f32;
    2.0 * precision * recall / (precision + recall)
}

/// Fraction of gold supporting paragraphs whose file appeared in the trace's seen files or
/// seen locations, matched by sanitized-title substring.
///
/// `seen_files` contains the `path` field from trace results (correct for normal mode).
/// `seen_locations` contains the `location` field from search results (correct for fullwiki
/// mode, where `path` is a shard file like `wiki/AA_wiki_00.md` but `location` carries the
/// article title).  A title is counted as recalled if its sanitized stem appears in either.
pub fn retrieval_recall(
    seen_files: &[String],
    seen_locations: &[String],
    supporting_titles: &[String],
) -> f32 {
    if supporting_titles.is_empty() {
        return 1.0;
    }
    let hit = supporting_titles
        .iter()
        .filter(|t| {
            let stem = sanitize_title(t);
            seen_files.iter().any(|f| f.contains(&stem))
                || seen_locations.iter().any(|l| sanitize_title(l).contains(&stem))
        })
        .count();
    hit as f32 / supporting_titles.len() as f32
}

/// Distinct article titles a question's searches surfaced, best-rank-first. Search-result hits carry
/// the article title in their `location` field; ties are broken by first occurrence across searches.
pub fn ranked_titles(transcript: &[glossa::trace::TraceEntry]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for e in transcript {
        if e.tool != "search" {
            continue;
        }
        if let Some(arr) = e.result.as_array() {
            for hit in arr {
                if let Some(loc) = hit.get("location").and_then(|v| v.as_str()) {
                    if seen.insert(normalize(loc)) {
                        out.push(loc.to_string());
                    }
                }
            }
        }
    }
    out
}

/// Fraction of gold titles found within the top-k of the merged ranked list.
pub fn recall_at_k(ranked: &[String], gold: &[String], k: usize) -> f32 {
    if gold.is_empty() {
        return 1.0;
    }
    let top: Vec<String> = ranked.iter().take(k).map(|t| normalize(t)).collect();
    let hit = gold.iter().filter(|g| top.contains(&normalize(g))).count();
    hit as f32 / gold.len() as f32
}

/// Reciprocal rank of the first gold title in the merged ranked list (0 if none).
pub fn mrr(ranked: &[String], gold: &[String]) -> f32 {
    if gold.is_empty() {
        return 0.0;
    }
    let goldn: Vec<String> = gold.iter().map(|g| normalize(g)).collect();
    for (i, t) in ranked.iter().enumerate() {
        if goldn.contains(&normalize(t)) {
            return 1.0 / (i as f32 + 1.0);
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_drops_articles_punct_case() {
        assert_eq!(normalize("The Big-Apple!"), "big apple");
    }

    #[test]
    fn em_and_f1() {
        assert!(exact_match("the cat", "Cat"));
        assert!(!exact_match("cat", "dog"));
        // pred "quick brown" vs gold "the quick brown fox": shared=2, P=2/2=1.0, R=2/3=0.667, F1=0.8
        assert!((token_f1("quick brown", "the quick brown fox") - 0.8).abs() < 1e-3);
        assert_eq!(token_f1("cat", "dog"), 0.0);
    }

    #[test]
    fn retrieval_recall_matches_by_sanitized_title() {
        let seen = vec!["eval-corpus/Bob_Page.md".to_string()];
        assert!((retrieval_recall(&seen, &[], &["Bob Page".into(), "Alice".into()]) - 0.5).abs() < 1e-6);
        assert_eq!(retrieval_recall(&seen, &[], &["Bob Page".into()]), 1.0);
        assert_eq!(retrieval_recall(&[], &[], &["Bob Page".into()]), 0.0);
    }

    #[test]
    fn retrieval_recall_fullwiki_via_location() {
        // In fullwiki mode the path is a shard file — the article title is in location.
        let shard_files = vec!["wiki/AA_wiki_00.md".to_string()];
        let locations = vec!["Bob Page".to_string(), "Other Article".to_string()];
        // "Bob Page" not found in shard path, but found via location → recall 1.0
        assert_eq!(retrieval_recall(&shard_files, &locations, &["Bob Page".into()]), 1.0);
        // "Alice" not found anywhere → recall 0.0
        assert_eq!(retrieval_recall(&shard_files, &locations, &["Alice".into()]), 0.0);
        // Both: Bob found via location, Alice not → 0.5
        assert!((retrieval_recall(&shard_files, &locations, &["Bob Page".into(), "Alice".into()]) - 0.5).abs() < 1e-6);
    }
}

#[cfg(test)]
mod retrieval_at_k_tests {
    use super::*;
    use glossa::trace::TraceEntry;

    #[test]
    fn ranked_titles_dedups_across_searches() {
        let mk = |hits: serde_json::Value| TraceEntry {
            ts_ms: 0, tool: "search".into(), args: serde_json::json!({}), result: hits,
        };
        let tr = vec![
            mk(serde_json::json!([{"location":"A","score":2.0},{"location":"B","score":1.0}])),
            mk(serde_json::json!([{"location":"B","score":3.0},{"location":"C","score":1.0}])),
        ];
        assert_eq!(ranked_titles(&tr), vec!["A".to_string(), "B".to_string(), "C".to_string()]);
    }

    #[test]
    fn recall_and_mrr() {
        let ranked = vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()];
        let gold = vec!["C".to_string(), "E".to_string()];
        assert!((recall_at_k(&ranked, &gold, 2) - 0.0).abs() < 1e-6); // C is rank 3
        assert!((recall_at_k(&ranked, &gold, 3) - 0.5).abs() < 1e-6);
        assert!((mrr(&ranked, &gold) - (1.0 / 3.0)).abs() < 1e-4);
    }

    #[test]
    fn matching_is_normalized() {
        let ranked = vec!["The Beatles".to_string()];
        assert_eq!(recall_at_k(&ranked, &["the beatles".to_string()], 1), 1.0);
    }
}
