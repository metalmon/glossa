//! Ripgrep-style literal/regex search over the extracted text stored in the index (File-First:
//! the index is a disposable accelerator; canonical content is the file). v1 always full-scans
//! all stored chunk bodies and confirms with the real `regex` engine. The trigram accelerator
//! is added in v2.

use crate::glob::{compile_glob, path_matches};
use crate::index::store::DocIndex;
use regex::RegexBuilder;

#[derive(Debug, Default, Clone)]
pub struct GrepOpts {
    /// Force case-insensitive matching (-i). Matching is smart-case by default (case-insensitive
    /// unless the pattern has an uppercase letter), so -i is only needed to force folding.
    pub ignore_case: bool,
    pub fixed: bool,
    pub word: bool,
    pub glob: Option<String>,
    pub file_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GrepHit {
    pub path: String,
    pub ord: u64,
    pub line: String,
}

impl GrepHit {
    /// One grep result line, addressing the chunk by its read key `#ord`.
    pub fn display_line(&self) -> String {
        format!("{}:#{}: {}", self.path, self.ord, self.line)
    }
}

/// Build the line matcher from the pattern + flags. `-F` escapes the whole pattern, `-w` wraps it
/// in word boundaries. Case folding is **smart-case** by default — case-insensitive unless the
/// pattern contains an uppercase letter — so a lowercase query matches mixed-case text; `-i`
/// forces folding unconditionally.
fn build_matcher(pattern: &str, opts: &GrepOpts) -> anyhow::Result<regex::Regex> {
    let mut body = if opts.fixed { regex::escape(pattern) } else { pattern.to_string() };
    if opts.word {
        body = format!(r"\b(?:{body})\b");
    }
    let smart_case = !pattern.chars().any(|c| c.is_uppercase());
    let re = RegexBuilder::new(&body).case_insensitive(opts.ignore_case || smart_case).build()?;
    Ok(re)
}

const GREP_MAX_HITS: usize = 1000;

pub fn grep(idx: &DocIndex, pattern: &str, opts: &GrepOpts) -> anyhow::Result<Vec<GrepHit>> {
    if pattern.trim().is_empty() {
        return Ok(Vec::new());
    }
    let matcher = build_matcher(pattern, opts)?;
    let glob_m = match &opts.glob { Some(g) => Some(compile_glob(g)?), None => None };
    let mut hits = Vec::new();
    let mut visit = |path: &str, ord: u64, file_type: &str, body: &str| {
        if hits.len() >= GREP_MAX_HITS { return; }
        if let Some(ft) = &opts.file_type { if file_type != ft { return; } }
        if let Some(m) = &glob_m { if !path_matches(m, path) { return; } }
        for line in body.lines() {
            if matcher.is_match(line) {
                hits.push(GrepHit { path: path.to_string(), ord, line: line.trim().to_string() });
                if hits.len() >= GREP_MAX_HITS { return; }
            }
        }
    };
    idx.iter_chunks(&mut visit)?;
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Chunk;
    use std::path::PathBuf;

    fn idx_with(chunks: &[(&str, &str, &str, &str)]) -> (tempfile::TempDir, DocIndex) {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let cs: Vec<Chunk> = chunks.iter().map(|(p, loc, ft, t)| Chunk {
            doc_path: PathBuf::from(p), location: (*loc).into(), file_type: (*ft).into(), text: (*t).into(),
        }).collect();
        idx.write_chunks(&cs).unwrap();
        (dir, idx)
    }

    #[test]
    fn grep_finds_exact_cyrillic_code_token() {
        // BM25 tokenization/stemming can blur exact codes; grep must find the literal.
        let (_d, idx) = idx_with(&[
            ("d.pdf", "p.7", "pdf", "Установите параметр maxTsdr равным 3000 tbit."),
            ("d.pdf", "p.8", "pdf", "Прочая страница без кода."),
        ]);
        let hits = grep(&idx, "maxTsdr", &GrepOpts::default()).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].ord, 7);
        assert!(hits[0].line.contains("maxTsdr"));
        assert_eq!(hits[0].display_line(), "d.pdf:#7: Установите параметр maxTsdr равным 3000 tbit.");
    }

    #[test]
    fn grep_regex_and_flags() {
        let (_d, idx) = idx_with(&[
            ("a.md", "S1", "md", "Договор №42 подписан"),
            ("b.md", "S1", "md", "договоры разные"),
            ("c.pdf", "p.1", "pdf", "ДОГОВОР заглавными"),
        ]);
        // -i smart-case (no uppercase in pattern -> case-insensitive)
        let hits = grep(&idx, "договор", &GrepOpts { ignore_case: true, ..Default::default() }).unwrap();
        assert_eq!(hits.len(), 3);
        // -F fixed string: the literal "№42" is found, regex metachars not interpreted
        let f = grep(&idx, "№42", &GrepOpts { fixed: true, ..Default::default() }).unwrap();
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].path, "a.md");
        // -t file_type filter
        let t = grep(&idx, "договор", &GrepOpts { ignore_case: true, file_type: Some("pdf".into()), ..Default::default() }).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].path, "c.pdf");
        // -g glob on path
        let g = grep(&idx, "договор", &GrepOpts { ignore_case: true, glob: Some("*.md".into()), ..Default::default() }).unwrap();
        assert_eq!(g.len(), 2);
        // -w word boundary: "договор" as a whole word does NOT match inside "договоры"
        let w = grep(&idx, "договор", &GrepOpts { ignore_case: true, word: true, ..Default::default() }).unwrap();
        assert_eq!(w.iter().filter(|h| h.path == "b.md").count(), 0);
    }

    #[test]
    fn grep_glob_recursive_on_windows_paths() {
        let (_d, idx) = idx_with(&[
            ("dir\\nested.md", "S1", "md", "договор подписан"),
            ("top.md", "S1", "md", "договор другой"),
            ("dir\\nested.pdf", "p.1", "pdf", "договор pdf"),
        ]);
        let g = grep(
            &idx,
            "договор",
            &GrepOpts { ignore_case: true, glob: Some("**/*.md".into()), ..Default::default() },
        )
        .unwrap();
        assert_eq!(g.len(), 2);
        let paths: std::collections::BTreeSet<_> = g.iter().map(|h| h.path.as_str()).collect();
        assert_eq!(paths, ["dir\\nested.md", "top.md"].into_iter().collect());
    }

    #[test]
    fn grep_prefilter_matches_fullscan_results() {
        // A regex with a long literal token uses the BM25 prefilter; results must equal the
        // full-scan results (the prefilter is an optimization, never changes correctness).
        let (_d, idx) = idx_with(&[
            ("a.md", "S1", "md", "регистрация устройства завершена"),
            ("b.md", "S1", "md", "несвязанный текст без слова"),
            ("c.md", "S2", "md", "повторная регистрация устройства"),
        ]);
        let hits = grep(&idx, "регистрация", &GrepOpts { ignore_case: true, ..Default::default() }).unwrap();
        let paths: std::collections::BTreeSet<_> = hits.iter().map(|h| h.path.clone()).collect();
        assert_eq!(paths, ["a.md".to_string(), "c.md".to_string()].into_iter().collect());
    }

    #[test]
    fn grep_substring_inside_token_is_found() {
        // ripgrep matches substrings; "Tsdr" lives INSIDE the token "maxTsdr" and must be found
        // (the removed BM25 token-prefilter would have missed this).
        let (_d, idx) = idx_with(&[
            ("d.pdf", "p.5", "pdf", "Параметр maxTsdr настраивается."),
        ]);
        let hits = grep(&idx, "Tsdr", &GrepOpts::default()).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].ord, 5);
        assert!(hits[0].line.contains("maxTsdr"));
    }

    #[test]
    fn grep_alternation_prefilter_is_sound() {
        let (_d, idx) = idx_with(&[
            ("a.md", "S1", "md", "выполнена регистрация устройства"),
            ("b.md", "S1", "md", "сторонний компонент"),
            ("c.md", "S1", "md", "обновлённый компонент системы"),
        ]);
        // Alternation: must find BOTH the "регистрация" chunk and a "компонент" chunk — the
        // prefilter must NOT pick one literal and drop the other's matches.
        let hits = grep(&idx, "регистрация|компонент", &GrepOpts { ignore_case: true, ..Default::default() }).unwrap();
        let paths: std::collections::BTreeSet<_> = hits.iter().map(|h| h.path.clone()).collect();
        assert_eq!(paths, ["a.md".to_string(), "b.md".to_string(), "c.md".to_string()].into_iter().collect());
    }

    #[test]
    fn grep_smart_case_default_folds_lowercase_only() {
        let (_d, idx) = idx_with(&[("d.pdf", "p.1", "pdf", "Контроллер АБАК подключён")]);
        // lowercase pattern, no -i: smart-case folds case -> matches the uppercase "АБАК"
        assert_eq!(grep(&idx, "абак", &GrepOpts::default()).unwrap().len(), 1);
        // pattern WITH an uppercase letter, no -i: smart-case stays case-sensitive -> "Абак" != "АБАК"
        assert_eq!(grep(&idx, "Абак", &GrepOpts::default()).unwrap().len(), 0);
    }

    #[test]
    fn grep_blank_pattern_returns_empty() {
        let (_d, idx) = idx_with(&[
            ("a.md", "S1", "md", "some content here"),
            ("b.md", "S1", "md", "more content there"),
        ]);
        // blank, whitespace-only, and empty patterns must all return empty — not the whole index.
        assert!(grep(&idx, "", &GrepOpts::default()).unwrap().is_empty());
        assert!(grep(&idx, "   ", &GrepOpts::default()).unwrap().is_empty());
        assert!(grep(&idx, "\t", &GrepOpts::default()).unwrap().is_empty());
    }
}
