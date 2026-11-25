//! Rare-code-preserving tokenizer: a token starts with a unicode word char and may contain
//! internal `. / _ -` so part numbers / extensions / protocol strings stay whole.
use tantivy::tokenizer::{TextAnalyzer, Token, TokenStream, Tokenizer};

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

/// A tantivy `Tokenizer` whose output is BYTE-FOR-BYTE [`tokenize`]'s output — not a
/// reimplementation via tantivy's split/filter primitives, which is how `body`'s `multilang`
/// tokenizer and the gate's own tokenizer would quietly drift apart. Delegating means the two
/// can never disagree; `gate_term_analyzer_matches_tokenize` below pins that this delegation
/// is actually wired (the thing worth protecting against is someone later inlining a "faster"
/// reimplementation here that stops calling `tokenize`).
///
/// Byte offsets are left at their default (0, 0): this tokenizer only ever backs an indexed,
/// NOT-stored field (the gate wants frequencies, never the text back), so nothing reads them —
/// there is no snippet/highlight path over gate terms to get an offset wrong for.
#[derive(Clone, Default)]
pub struct GateTermTokenizer;

pub struct GateTermTokenStream {
    tokens: std::vec::IntoIter<String>,
    token: Token,
}

impl Tokenizer for GateTermTokenizer {
    type TokenStream<'a> = GateTermTokenStream;
    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        GateTermTokenStream {
            tokens: tokenize(text).into_iter(),
            token: Token::default(),
        }
    }
}

impl TokenStream for GateTermTokenStream {
    fn advance(&mut self) -> bool {
        match self.tokens.next() {
            Some(t) => {
                self.token.text = t;
                self.token.position = self.token.position.wrapping_add(1);
                true
            }
            None => false,
        }
    }
    fn token(&self) -> &Token {
        &self.token
    }
    fn token_mut(&mut self) -> &mut Token {
        &mut self.token
    }
}

/// The analyzer registered with tantivy under the `"gate_term"` name (see
/// `index::store::register_tokenizers`) — the one place the gate's own vocabulary field gets its
/// postings from.
pub fn gate_term_analyzer() -> TextAnalyzer {
    TextAnalyzer::builder(GateTermTokenizer).build()
}

#[cfg(test)]
mod tests {
    use super::{gate_term_analyzer, tokenize};

    /// Pins the tantivy-side `gate_term` analyzer to `tokenize` — the safety rail the brief for
    /// this change calls "the single most important fact": the gate has its OWN tokenizer,
    /// deliberately different from `body`'s `multilang` one (no stemming, dotted codes kept
    /// whole), so the field that backs `DfTable::df` must tokenize EXACTLY like the gate does, or
    /// every multi-part code and every inflected word reads back as df==0 and the gate's rare set
    /// explodes into mass abstention. Not a hand-picked sample — explicit edge cases plus a
    /// deterministic fuzz over a mixed RU/EN/digit/punctuation/code/unicode-edge palette, so any
    /// divergence between the two surfaces, the way `term_analyzer_matches_index_pipeline` does
    /// for the `multilang` pair.
    #[test]
    fn gate_term_analyzer_matches_tokenize() {
        let mut analyzer = gate_term_analyzer();
        let mut check = |s: &str| {
            let mut ts = analyzer.token_stream(s);
            let mut got = Vec::new();
            while ts.advance() {
                got.push(ts.token().text.clone());
            }
            assert_eq!(
                got,
                tokenize(s),
                "gate_term analyzer must match tokenize for {s:?}"
            );
        };

        for s in [
            "Install GSD-driver PP.19.00.00.00 the file config.mimg Café",
            "modbus and modbus again",
            "running RUNS runner",
            "p.7 (\u{0441}\u{0442}\u{0440}.) \u{0434}\u{043E}\u{0433}\u{043E}\u{0432}\u{043E}\u{0440}",
            "maxTsdr=3000, version 5.7.2",
            "---...___///",
            "a.b/c_d-e",
            "   ",
            "",
        ] {
            check(s);
        }

        // Deterministic fuzz (no rand crate): LCG over a varied char palette, same style as
        // `term_analyzer_matches_index_pipeline`.
        let palette: Vec<char> = "abzqAZ059 \t\n.,-=/_()#—\u{0431}\u{044A}\u{0439}ßİ²"
            .chars()
            .collect();
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let next = |st: &mut u64| -> u64 {
            *st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *st >> 33
        };
        for _ in 0..5000 {
            let len = (next(&mut state) % 20) as usize;
            let s: String = (0..len)
                .map(|_| palette[(next(&mut state) as usize) % palette.len()])
                .collect();
            check(&s);
        }
    }

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
