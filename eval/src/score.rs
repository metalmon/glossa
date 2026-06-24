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

/// Fraction of gold supporting paragraphs whose file appeared in the trace's seen files,
/// matched by sanitized-title filename substring.
pub fn retrieval_recall(seen_files: &[String], supporting_titles: &[String]) -> f32 {
    if supporting_titles.is_empty() {
        return 1.0;
    }
    let hit = supporting_titles
        .iter()
        .filter(|t| {
            let stem = sanitize_title(t);
            seen_files.iter().any(|f| f.contains(&stem))
        })
        .count();
    hit as f32 / supporting_titles.len() as f32
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
        assert!((retrieval_recall(&seen, &["Bob Page".into(), "Alice".into()]) - 0.5).abs() < 1e-6);
        assert_eq!(retrieval_recall(&seen, &["Bob Page".into()]), 1.0);
        assert_eq!(retrieval_recall(&[], &["Bob Page".into()]), 0.0);
    }
}
