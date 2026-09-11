//! Rare-code-preserving tokenizer: a token starts with a unicode word char and may contain
//! internal `. / _ -` so part numbers / extensions / protocol strings stay whole.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let is_word = |c: char| c.is_alphanumeric();
    let is_inner = |c: char| matches!(c, '.' | '/' | '_' | '-');
    for c in text.chars() {
        if is_word(c) || (is_inner(c) && !cur.is_empty()) {
            cur.push(c);
        } else if !cur.is_empty() {
            push_token(&mut out, &cur);
            cur.clear();
        }
    }
    if !cur.is_empty() {
        push_token(&mut out, &cur);
    }
    out
}

fn push_token(out: &mut Vec<String>, raw: &str) {
    let t = raw
        .trim_matches(|c| matches!(c, '.' | '/' | '_' | '-'))
        .to_lowercase();
    if t.chars().count() >= 3 {
        out.push(t);
    }
}

#[cfg(test)]
mod tests {
    use super::tokenize;
    #[test]
    fn keeps_codes_and_lowercases() {
        let t = tokenize("Install GSD-driver PP.19.00.00.00 the file config.mimg Café");
        assert!(
            t.contains(&"pp.19.00.00.00".to_string()),
            "dotted code kept whole: {t:?}"
        );
        assert!(
            t.contains(&"config.mimg".to_string()),
            "extension kept: {t:?}"
        );
        assert!(
            t.contains(&"café".to_string()),
            "non-ascii lowercased by char count: {t:?}"
        );
        assert!(
            !t.iter().any(|w| w.chars().count() < 3),
            "sub-3-char tokens dropped: {t:?}"
        );
    }
}
