use rust_stemmers::{Algorithm, Stemmer as RsStemmer};
use std::sync::Arc;
use tantivy::tokenizer::{
    LowerCaser, SimpleTokenizer, TextAnalyzer, Token, TokenFilter, TokenStream, Tokenizer,
};

/// Maps a text to the stemming algorithm to use for it.
/// Cloneable, thread-safe — tantivy tokenizers must be `Clone + Send + Sync + 'static`.
pub type DetectFn = Arc<dyn Fn(&str) -> Algorithm + Send + Sync>;

/// Default, zero-dependency detector. RU and EN live in different scripts, so a
/// single Cyrillic char is a reliable, free signal for Russian; everything else → English.
pub fn script_detector() -> DetectFn {
    Arc::new(|text: &str| {
        if text.chars().any(|c| ('\u{0400}'..='\u{04FF}').contains(&c)) {
            Algorithm::Russian
        } else {
            Algorithm::English
        }
    })
}

/// The detector used by default (script-based; embeds no models, pulls no extra deps).
pub fn default_detector() -> DetectFn {
    script_detector()
}

/// Optional `lingua`-backed detector, for distinguishing same-script languages.
/// Build with `--features lingua`.
#[cfg(feature = "lingua")]
pub fn lingua_detector() -> DetectFn {
    use lingua::{Language, LanguageDetectorBuilder};
    let detector =
        LanguageDetectorBuilder::from_languages(&[Language::English, Language::Russian]).build();
    Arc::new(move |text: &str| match detector.detect_language_of(text) {
        Some(Language::Russian) => Algorithm::Russian,
        _ => Algorithm::English,
    })
}

/// A tantivy TokenFilter that picks a stemming algorithm per text (via `detect`),
/// then stems every token. Tokens must already be lower-cased (chain LowerCaser first).
#[derive(Clone)]
pub struct MultiLangStemmer {
    detect: DetectFn,
}

impl MultiLangStemmer {
    pub fn new(detect: DetectFn) -> Self {
        Self { detect }
    }
}

#[derive(Clone)]
pub struct MultiLangTokenizer<T> {
    inner: T,
    detect: DetectFn,
}

pub struct MultiLangStream<T> {
    inner: T,
    stemmer: RsStemmer,
    buf: String,
}

impl TokenFilter for MultiLangStemmer {
    type Tokenizer<T: Tokenizer> = MultiLangTokenizer<T>;
    fn transform<T: Tokenizer>(self, inner: T) -> MultiLangTokenizer<T> {
        MultiLangTokenizer {
            inner,
            detect: self.detect,
        }
    }
}

impl<T: Tokenizer> Tokenizer for MultiLangTokenizer<T> {
    type TokenStream<'a> = MultiLangStream<T::TokenStream<'a>>;
    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        let algo = (self.detect)(text);
        MultiLangStream {
            inner: self.inner.token_stream(text),
            stemmer: RsStemmer::create(algo),
            buf: String::new(),
        }
    }
}

impl<T: TokenStream> TokenStream for MultiLangStream<T> {
    fn advance(&mut self) -> bool {
        if !self.inner.advance() {
            return false;
        }
        let token = self.inner.token_mut();
        let stemmed = self.stemmer.stem(&token.text);
        if stemmed != token.text.as_str() {
            self.buf.clear();
            self.buf.push_str(&stemmed);
            token.text.clear();
            token.text.push_str(&self.buf);
        }
        true
    }
    fn token(&self) -> &Token {
        self.inner.token()
    }
    fn token_mut(&mut self) -> &mut Token {
        self.inner.token_mut()
    }
}

/// Compose the full analyzer: split → lowercase → multilingual stem.
pub fn multilang_analyzer(detect: DetectFn) -> TextAnalyzer {
    TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(LowerCaser)
        .filter(MultiLangStemmer::new(detect))
        .build()
}

/// Analyze a string into its stemmed, lower-cased terms using the SAME pipeline as the
/// search index (split → lowercase → multilingual stem). Lets non-index callers (e.g.
/// graph entity resolution) match query/label terms with search-consistent morphology.
pub fn analyze_terms(text: &str) -> Vec<String> {
    let mut analyzer = multilang_analyzer(default_detector());
    let mut ts = analyzer.token_stream(text);
    let mut out = Vec::new();
    while ts.advance() {
        out.push(ts.token().text.clone());
    }
    out
}

/// Reusable analyzer for the SAME term pipeline as the search index (split on non-alphanumeric →
/// lowercase → per-text RU/EN stem), but with the two stemmers built ONCE and reused across many
/// strings. `analyze_terms`/`multilang_analyzer` rebuild a tokenizer+stemmer per string, which is
/// far too costly in hot loops — graph `resolve` stems every node label/alias on every call, and
/// `graph_upsert` calls `resolve` for every paraphrased edge endpoint during enrichment.
/// Equivalence to `analyze_terms` is enforced by `term_analyzer_matches_index_pipeline`.
pub struct TermAnalyzer {
    ru: RsStemmer,
    en: RsStemmer,
}

impl Default for TermAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl TermAnalyzer {
    pub fn new() -> Self {
        TermAnalyzer {
            ru: RsStemmer::create(Algorithm::Russian),
            en: RsStemmer::create(Algorithm::English),
        }
    }

    /// Fill `out` (cleared first) with the stemmed, lower-cased terms of `text`.
    pub fn terms(&self, text: &str, out: &mut std::collections::BTreeSet<String>) {
        out.clear();
        // Same per-text script detection as `script_detector`: any Cyrillic char → Russian.
        let cyrillic = text.chars().any(|c| ('\u{0400}'..='\u{04FF}').contains(&c));
        let stemmer = if cyrillic { &self.ru } else { &self.en };
        for tok in text.split(|c: char| !c.is_alphanumeric()) {
            if tok.is_empty() {
                continue;
            }
            out.insert(stemmer.stem(&tok.to_lowercase()).to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(analyzer: &mut TextAnalyzer, text: &str) -> Vec<String> {
        let mut ts = analyzer.token_stream(text);
        let mut out = Vec::new();
        while ts.advance() {
            out.push(ts.token().text.clone());
        }
        out
    }

    #[test]
    fn russian_inflections_stem_to_same_root() {
        let mut a = multilang_analyzer(default_detector());
        let one = tokens(&mut a, "договор");
        let many = tokens(&mut a, "договоры договоров договорам");
        let root = &one[0];
        assert!(
            many.iter().all(|t| t == root),
            "expected all forms to stem to {root:?}, got {many:?}"
        );
    }

    #[test]
    fn english_inflections_stem_to_same_root() {
        let mut a = multilang_analyzer(default_detector());
        let toks = tokens(&mut a, "running runs runner");
        assert_eq!(toks[0], "run");
        assert_eq!(toks[1], "run");
    }

    #[test]
    fn term_analyzer_matches_index_pipeline() {
        // TermAnalyzer (cached stemmers) MUST produce the same term set as analyze_terms (the
        // tantivy SimpleTokenizer + LowerCaser + per-text stem pipeline) — NOT a hand-picked
        // sample: explicit edge cases + a deterministic fuzz over a mixed RU/EN/digit/punctuation/
        // unicode-edge palette, so any tokenization/lowercase/stem divergence surfaces.
        let a = TermAnalyzer::new();
        let mut buf = std::collections::BTreeSet::new();
        let mut check = |s: &str| {
            a.terms(s, &mut buf);
            let expected: std::collections::BTreeSet<String> =
                analyze_terms(s).into_iter().collect();
            assert_eq!(buf, expected, "TermAnalyzer must match analyze_terms for {s:?}");
        };

        for s in [
            "договоры договоров договорам", "running RUNS runner",
            "Периодическая потеря связи Modbus", "maxTsdr=3000, версия 5.7.2",
            "p.7 (стр.)", "ёлка ЁЖ-1", "ßİ²—", "   ", "",
        ] {
            check(s);
        }

        // Deterministic fuzz (no rand crate): LCG over a varied char palette.
        let palette: Vec<char> = "abzqAZабяАЯёЁъй059 \t\n.,-=/()#—ßİ²".chars().collect();
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let next = |st: &mut u64| -> u64 {
            *st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *st >> 33
        };
        for _ in 0..5000 {
            let len = (next(&mut state) % 16) as usize;
            let s: String = (0..len)
                .map(|_| palette[(next(&mut state) as usize) % palette.len()])
                .collect();
            check(&s);
        }
    }
}
