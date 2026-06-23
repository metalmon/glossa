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
}
