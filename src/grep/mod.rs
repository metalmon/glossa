//! Ripgrep-style literal/regex search over extracted text in the index (File-First).
//! Uses a char-trigram prefilter when selective; always confirms with the real `regex` engine.

mod trigram;

use crate::glob::{compile_glob, path_matches};
use crate::index::store::DocIndex;
use regex::RegexBuilder;
pub use trigram::{literal_trigrams, TrigramQuery};

#[derive(Debug, Default, Clone)]
pub struct GrepOpts {
    /// Force case-insensitive matching (-i). Matching is smart-case by default (case-insensitive
    /// unless the pattern has an uppercase letter), so -i is only needed to force folding.
    pub ignore_case: bool,
    pub fixed: bool,
    pub word: bool,
    pub glob: Option<String>,
    pub file_type: Option<String>,
    /// -B N: emit N context lines BEFORE each match (from the same chunk).
    pub before: usize,
    /// -A N: emit N context lines AFTER each match (from the same chunk). `-C n` sets both.
    pub after: usize,
    /// -o: emit only the matched substring(s) of each line, one per line, not the whole line.
    pub only_matching: bool,
    /// -n: include the 1-based line number within the chunk in each emitted line.
    pub line_number: bool,
    /// -c: output only a count of matching lines per chunk, not the lines themselves.
    pub count: bool,
    /// -m N: stop after N matching lines per chunk.
    pub max_count: Option<usize>,
    /// -U: let the pattern span lines (`.` matches `\n`); match the whole chunk body at once.
    pub multiline: bool,
    /// Cap each emitted line to this many characters. PDF extraction often stores a whole
    /// page as ONE line, so a context grep would hand the model the entire page per hit.
    /// A match line keeps a window around its first match; a context line keeps its head.
    /// `None` = verbatim (core default).
    pub line_cap: Option<usize>,
    /// The grep target path as the model passed it, before any `path_to_glob` conversion
    /// (feature-gated notebook dispatch only — `grep()` in this module never reads it).
    /// `crate::tools::grep` probes this for a doc-scoped notebook path (`<doc>/<file>`)
    /// before falling through to the corpus grep below.
    pub path: Option<String>,
}

impl GrepOpts {
    /// Tool-level default: a plain grep (no context / only-matching / count set) returns a
    /// small WINDOW around each match. A small model greps heavily but never sets `context`,
    /// so a bare matching line is useless for reading a value's table; a default window gives
    /// it the surrounding rows without having to know the flag. The core `grep()` does NOT
    /// apply this — it honours opts verbatim (its tests stay literal) — so only the tool
    /// callers (MCP + eval), which share this method, window by default and stay identical.
    /// The same tool boundary also caps emitted line length (giant one-line PDF pages).
    pub fn with_default_context(mut self) -> Self {
        const DEFAULT: usize = 8;
        const LINE_CAP: usize = 500;
        if self.before == 0 && self.after == 0 && !self.only_matching && !self.count {
            self.before = DEFAULT;
            self.after = DEFAULT;
        }
        if self.line_cap.is_none() {
            self.line_cap = Some(LINE_CAP);
        }
        self
    }
}

/// Map a document path (as `read`/`glob`/`search` show it, optionally with a `#chunk`
/// suffix) to a glob that scopes a grep to that one document. A plain path gets a `**/`
/// prefix so both bare and nested stored paths match; a value already carrying glob
/// metacharacters passes through unchanged.
pub fn path_to_glob(path: &str) -> String {
    let p = match path.rfind('#') {
        Some(i) if i + 1 < path.len() && path[i + 1..].chars().all(|c| c.is_ascii_digit()) => {
            &path[..i]
        }
        _ => path,
    };
    let p = p.trim();
    if p.chars().any(|c| "*?[{".contains(c)) {
        p.to_string()
    } else {
        format!("**/{p}")
    }
}

/// What a [`GrepHit`] represents, so `display_line` can render it distinctly (ripgrep uses `:` for
/// a match and `-` for a context line).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HitKind {
    /// A line (or, with `-o`, a substring) that matched the pattern.
    Match,
    /// A neighbouring line pulled in by `-A`/`-B`/`-C` context, not itself a match.
    Context,
    /// A per-chunk match count (`-c`); the count is carried in `line`.
    Count,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GrepHit {
    pub path: String,
    pub ord: u64,
    /// 1-based line number within the chunk. `0` means "do not show" (line numbering off).
    pub line_no: usize,
    pub line: String,
    pub kind: HitKind,
}

impl GrepHit {
    /// One grep result line, addressing the chunk by its read key `#ord`. A match uses `:` as the
    /// separator, a context line uses `-` (mirroring ripgrep), and the 1-based `line_no` is inserted
    /// only when line numbering is on (`line_no > 0`). Default output (no new flags) is byte-identical
    /// to the historical `"{path}:#{ord}: {line}"`.
    pub fn display_line(&self) -> String {
        let sep = if self.kind == HitKind::Context {
            '-'
        } else {
            ':'
        };
        if self.line_no > 0 {
            format!(
                "{}{sep}#{}{sep}{}{sep} {}",
                self.path, self.ord, self.line_no, self.line
            )
        } else {
            format!("{}{sep}#{}{sep} {}", self.path, self.ord, self.line)
        }
    }
}

/// Build the line matcher from the pattern + flags. `-F` escapes the whole pattern, `-w` wraps it
/// in word boundaries. Case folding is **smart-case** by default — case-insensitive unless the
/// pattern contains an uppercase letter — so a lowercase query matches mixed-case text; `-i`
/// forces folding unconditionally.
pub(crate) fn build_matcher(pattern: &str, opts: &GrepOpts) -> anyhow::Result<regex::Regex> {
    let mut body = if opts.fixed {
        regex::escape(pattern)
    } else {
        pattern.to_string()
    };
    if opts.word {
        body = format!(r"\b(?:{body})\b");
    }
    let smart_case = !pattern.chars().any(|c| c.is_uppercase());
    let re = RegexBuilder::new(&body)
        .case_insensitive(opts.ignore_case || smart_case)
        // -U: let `.` cross line boundaries so the pattern can span lines when matched against the
        // whole chunk body. Off by default → line-by-line matching is unchanged.
        .dot_matches_new_line(opts.multiline)
        .build()?;
    Ok(re)
}

/// 0-based line index containing byte offset `off` within `body` (for multiline match anchoring).
fn line_of_offset(body: &str, off: usize) -> usize {
    body.as_bytes()[..off]
        .iter()
        .filter(|&&b| b == b'\n')
        .count()
}

/// Rewrite runs of literal spaces in the pattern to `\s+` (outside `[...]` classes), so a
/// single-spaced query matches the irregular whitespace of extracted PDF text ("Пример
/// условного" finds "Пример  условного"). Returns `None` when the pattern has no space.
/// The caller must treat the rewritten pattern as a regex (clear `fixed`); `fixed` input
/// is escaped first so its other characters stay literal.
fn normalize_pattern_spaces(pattern: &str, fixed: bool) -> Option<String> {
    if !pattern.contains(' ') {
        return None;
    }
    let base = if fixed {
        regex::escape(pattern)
    } else {
        pattern.to_string()
    };
    let mut out = String::with_capacity(base.len() + 8);
    let mut chars = base.chars().peekable();
    let mut in_class = false;
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                out.push(c);
                if let Some(n) = chars.next() {
                    out.push(n);
                }
            }
            '[' if !in_class => {
                in_class = true;
                out.push(c);
            }
            ']' if in_class => {
                in_class = false;
                out.push(c);
            }
            ' ' if !in_class => {
                while chars.peek() == Some(&' ') {
                    chars.next();
                }
                out.push_str(r"\s+");
            }
            _ => out.push(c),
        }
    }
    Some(out)
}

/// Entry-point pattern/flag preparation shared by `grep` and `grep_fullscan`: honour regex
/// intent under `fixed`, then make literal spaces whitespace-tolerant. Both the matcher and
/// the trigram prefilter see the SAME prepared pattern, so they stay consistent (the `\s+`
/// links become `Any` in the plan and the word literals still prefilter).
pub(crate) fn prepare(pattern: &str, opts: &GrepOpts) -> (String, GrepOpts) {
    let mut o = honor_regex_intent(pattern, opts);
    match normalize_pattern_spaces(pattern, o.fixed) {
        Some(p) => {
            o.fixed = false;
            (p, o)
        }
        None => (pattern.to_string(), o),
    }
}

/// Shorten an emitted line to `cap` characters. A match line keeps a window around its
/// first match (so the hit stays visible); a context line (or a line the matcher cannot
/// locate a match in, e.g. multiline spans) keeps its head. Dropped characters are
/// summarized so the model knows the line continues.
fn cap_line(line: &str, cap: usize, matcher: Option<&regex::Regex>) -> String {
    let chars: Vec<char> = line.chars().collect();
    if chars.len() <= cap {
        return line.to_string();
    }
    let start = match matcher.and_then(|m| m.find(line)) {
        Some(m) => {
            let m_start = line[..m.start()].chars().count();
            let m_len = line[m.start()..m.end()].chars().count();
            // Center the window on the match (clamped to the line).
            m_start
                .saturating_sub(cap.saturating_sub(m_len) / 2)
                .min(chars.len() - cap)
        }
        None => 0,
    };
    let window: String = chars[start..start + cap].iter().collect();
    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    out.push_str(window.trim());
    out.push_str(&format!(" … [+{} chars]", chars.len() - start - cap));
    out
}

/// Compute all [`GrepHit`]s for a single chunk and append them to `out`. This is the one place the
/// flag semantics live, shared by the trigram `grep` and the `grep_fullscan` equivalence path so
/// both render identically. `matcher` is already built from the pattern + flags.
fn chunk_hits(
    matcher: &regex::Regex,
    opts: &GrepOpts,
    path: &str,
    ord: u64,
    body: &str,
    out: &mut Vec<GrepHit>,
) {
    let lines: Vec<&str> = body.lines().collect();

    // Matching line indices (0-based). In multiline mode we match the whole body and anchor each
    // match to the line of its start; otherwise we test line by line (historical behavior).
    let mut match_lines: Vec<usize> = if opts.multiline {
        let mut v = Vec::new();
        for m in matcher.find_iter(body) {
            let ln = line_of_offset(body, m.start());
            if v.last() != Some(&ln) {
                v.push(ln);
            }
        }
        v
    } else {
        lines
            .iter()
            .enumerate()
            .filter(|(_, l)| matcher.is_match(l))
            .map(|(i, _)| i)
            .collect()
    };

    // -m N: cap matching lines per chunk.
    if let Some(mc) = opts.max_count {
        match_lines.truncate(mc);
    }

    let line_no = |i: usize| if opts.line_number { i + 1 } else { 0 };

    // -c: just the count of matching lines per chunk.
    if opts.count {
        out.push(GrepHit {
            path: path.to_string(),
            ord,
            line_no: 0,
            line: match_lines.len().to_string(),
            kind: HitKind::Count,
        });
        return;
    }

    // -o: only the matched substring(s), one hit per occurrence (context is not applied, as in rg).
    if opts.only_matching {
        let render = |s: &str| match opts.line_cap {
            Some(cap) => cap_line(s.trim(), cap, None),
            None => s.trim().to_string(),
        };
        if opts.multiline {
            let allowed: std::collections::HashSet<usize> = match_lines.iter().copied().collect();
            for m in matcher.find_iter(body) {
                let ln = line_of_offset(body, m.start());
                if allowed.contains(&ln) {
                    out.push(GrepHit {
                        path: path.to_string(),
                        ord,
                        line_no: line_no(ln),
                        line: render(m.as_str()),
                        kind: HitKind::Match,
                    });
                }
            }
        } else {
            for &i in &match_lines {
                for m in matcher.find_iter(lines[i]) {
                    out.push(GrepHit {
                        path: path.to_string(),
                        ord,
                        line_no: line_no(i),
                        line: render(m.as_str()),
                        kind: HitKind::Match,
                    });
                }
            }
        }
        return;
    }

    // Whole-line matches, plus -A/-B/-C context. Build the set of line indices to emit from the
    // retained matches (merging overlapping windows via the ordered set so lines never repeat), and
    // tag each emitted line as a match or as context.
    let mut is_match_line = vec![false; lines.len()];
    for &i in &match_lines {
        is_match_line[i] = true;
    }
    let mut emit: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    let last = lines.len().saturating_sub(1);
    for &i in &match_lines {
        let lo = i.saturating_sub(opts.before);
        let hi = (i + opts.after).min(last);
        for j in lo..=hi {
            emit.insert(j);
        }
    }
    for j in emit {
        let kind = if is_match_line[j] {
            HitKind::Match
        } else {
            HitKind::Context
        };
        let trimmed = lines[j].trim();
        let line = match opts.line_cap {
            // Only a match line gets the match-centered window; context keeps its head.
            Some(cap) => cap_line(trimmed, cap, (kind == HitKind::Match).then_some(matcher)),
            None => trimmed.to_string(),
        };
        out.push(GrepHit {
            path: path.to_string(),
            ord,
            line_no: line_no(j),
            line,
            kind,
        });
    }
}

const GREP_MAX_HITS: usize = 1000;

/// A small model routinely sets `fixed` yet writes a regex (`A|B`, `x.*y`). When the
/// pattern carries regex metacharacters, honour that intent by treating it as a regex —
/// clearing `fixed` here, at the entry, so BOTH the trigram prefilter and the matcher
/// agree (clearing it only in the matcher left the prefilter hunting for a literal pipe
/// and finding no candidate chunks). A bare `.` is not treated as meta, so a decimal
/// like "3.0" stays literal under `fixed`.
fn honor_regex_intent(pattern: &str, opts: &GrepOpts) -> GrepOpts {
    let mut o = opts.clone();
    if o.fixed && pattern.chars().any(|c| "|*+?[](){}\\^$".contains(c)) {
        o.fixed = false;
    }
    o
}

pub fn grep(idx: &DocIndex, pattern: &str, opts: &GrepOpts) -> anyhow::Result<Vec<GrepHit>> {
    if pattern.trim().is_empty() {
        return Ok(Vec::new());
    }
    let (pattern, opts) = prepare(pattern, opts);
    let (pattern, opts) = (pattern.as_str(), &opts);
    let matcher = build_matcher(pattern, opts)?;
    let glob_m = match &opts.glob {
        Some(g) => Some(compile_glob(g)?),
        None => None,
    };
    let plan = trigram::trigram_plan(pattern, opts);
    let mut hits = Vec::new();
    // The trigram plan can visit the same chunk via several OR branches; a chunk's hits are fully
    // determined by its body, so process each `(path, ord)` once to match the single-pass fullscan.
    let mut seen = std::collections::HashSet::new();
    let mut visit = |path: &str, ord: u64, file_type: &str, body: &str| {
        if hits.len() >= GREP_MAX_HITS {
            return;
        }
        if let Some(ft) = &opts.file_type {
            if file_type != ft {
                return;
            }
        }
        if let Some(m) = &glob_m {
            if !path_matches(m, path) {
                return;
            }
        }
        if !seen.insert((path.to_string(), ord)) {
            return;
        }
        chunk_hits(&matcher, opts, path, ord, body, &mut hits);
    };
    run_plan(idx, &plan, &mut visit)?;
    Ok(hits)
}

fn run_plan(
    idx: &DocIndex,
    plan: &TrigramQuery,
    visit: &mut impl FnMut(&str, u64, &str, &str),
) -> anyhow::Result<()> {
    match plan {
        TrigramQuery::Any => idx.iter_chunks(visit),
        TrigramQuery::And(grams) => idx.iter_chunks_trigram_candidates(grams, visit),
        TrigramQuery::Or(branches) => {
            for branch in branches {
                run_plan(idx, branch, visit)?;
            }
            Ok(())
        }
    }
}

/// Full-scan grep (for equivalence tests).
pub fn grep_fullscan(
    idx: &DocIndex,
    pattern: &str,
    opts: &GrepOpts,
) -> anyhow::Result<Vec<GrepHit>> {
    if pattern.trim().is_empty() {
        return Ok(Vec::new());
    }
    let (pattern, opts) = prepare(pattern, opts);
    let (pattern, opts) = (pattern.as_str(), &opts);
    let matcher = build_matcher(pattern, opts)?;
    let glob_m = match &opts.glob {
        Some(g) => Some(compile_glob(g)?),
        None => None,
    };
    let mut hits = Vec::new();
    let mut visit = |path: &str, ord: u64, file_type: &str, body: &str| {
        if hits.len() >= GREP_MAX_HITS {
            return;
        }
        if let Some(ft) = &opts.file_type {
            if file_type != ft {
                return;
            }
        }
        if let Some(m) = &glob_m {
            if !path_matches(m, path) {
                return;
            }
        }
        chunk_hits(&matcher, opts, path, ord, body, &mut hits);
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
        let cs: Vec<Chunk> = chunks
            .iter()
            .map(|(p, loc, ft, t)| Chunk {
                doc_path: PathBuf::from(p),
                location: (*loc).into(),
                file_type: (*ft).into(),
                text: (*t).into(),
            })
            .collect();
        idx.write_chunks(&cs).unwrap();
        (dir, idx)
    }

    fn assert_same_hits(pattern: &str, opts: &GrepOpts, idx: &DocIndex) {
        let pre = grep(idx, pattern, opts).unwrap();
        let full = grep_fullscan(idx, pattern, opts).unwrap();
        let mut a: Vec<_> = pre
            .iter()
            .map(|h| (h.path.as_str(), h.ord, h.line.as_str()))
            .collect();
        let mut b: Vec<_> = full
            .iter()
            .map(|h| (h.path.as_str(), h.ord, h.line.as_str()))
            .collect();
        a.sort();
        b.sort();
        assert_eq!(a, b, "prefilter != fullscan for {pattern:?}");
    }

    #[test]
    fn grep_finds_exact_cyrillic_code_token() {
        let (_d, idx) = idx_with(&[
            (
                "d.pdf",
                "p.7",
                "pdf",
                "Установите параметр maxTsdr равным 3000 tbit.",
            ),
            ("d.pdf", "p.8", "pdf", "Прочая страница без кода."),
        ]);
        let hits = grep(&idx, "maxTsdr", &GrepOpts::default()).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].ord, 7);
        assert!(hits[0].line.contains("maxTsdr"));
        assert_eq!(
            hits[0].display_line(),
            "d.pdf:#7: Установите параметр maxTsdr равным 3000 tbit."
        );
    }

    #[test]
    fn grep_regex_and_flags() {
        let (_d, idx) = idx_with(&[
            ("a.md", "S1", "md", "Договор №42 подписан"),
            ("b.md", "S1", "md", "договоры разные"),
            ("c.pdf", "p.1", "pdf", "ДОГОВОР заглавными"),
        ]);
        let hits = grep(
            &idx,
            "договор",
            &GrepOpts {
                ignore_case: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(hits.len(), 3);
        let f = grep(
            &idx,
            "№42",
            &GrepOpts {
                fixed: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].path, "a.md");
        let t = grep(
            &idx,
            "договор",
            &GrepOpts {
                ignore_case: true,
                file_type: Some("pdf".into()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].path, "c.pdf");
        let g = grep(
            &idx,
            "договор",
            &GrepOpts {
                ignore_case: true,
                glob: Some("*.md".into()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(g.len(), 2);
        let w = grep(
            &idx,
            "договор",
            &GrepOpts {
                ignore_case: true,
                word: true,
                ..Default::default()
            },
        )
        .unwrap();
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
            &GrepOpts {
                ignore_case: true,
                glob: Some("**/*.md".into()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(g.len(), 2);
        let paths: std::collections::BTreeSet<_> = g.iter().map(|h| h.path.as_str()).collect();
        assert_eq!(paths, ["dir\\nested.md", "top.md"].into_iter().collect());
    }

    #[test]
    fn grep_prefilter_matches_fullscan_results() {
        let (_d, idx) = idx_with(&[
            ("a.md", "S1", "md", "регистрация устройства завершена"),
            ("b.md", "S1", "md", "несвязанный текст без слова"),
            ("c.md", "S2", "md", "повторная регистрация устройства"),
        ]);
        assert_same_hits(
            "регистрация",
            &GrepOpts {
                ignore_case: true,
                ..Default::default()
            },
            &idx,
        );
    }

    #[test]
    fn grep_substring_inside_token_is_found() {
        let (_d, idx) = idx_with(&[("d.pdf", "p.5", "pdf", "Параметр maxTsdr настраивается.")]);
        assert_same_hits("Tsdr", &GrepOpts::default(), &idx);
    }

    #[test]
    fn grep_alternation_prefilter_is_sound() {
        let (_d, idx) = idx_with(&[
            ("a.md", "S1", "md", "выполнена регистрация устройства"),
            ("b.md", "S1", "md", "сторонний компонент"),
            ("c.md", "S1", "md", "обновлённый компонент системы"),
        ]);
        assert_same_hits(
            "регистрация|компонент",
            &GrepOpts {
                ignore_case: true,
                ..Default::default()
            },
            &idx,
        );
    }

    #[test]
    fn grep_trigram_matches_fullscan_fixed() {
        let (_d, idx) = idx_with(&[
            (
                "d.pdf",
                "p.7",
                "pdf",
                "Установите параметр maxTsdr равным 3000 tbit.",
            ),
            ("d.pdf", "p.8", "pdf", "Прочая страница без кода."),
        ]);
        assert_same_hits(
            "maxTsdr",
            &GrepOpts {
                fixed: true,
                ..Default::default()
            },
            &idx,
        );
    }

    #[test]
    fn grep_short_literal_full_scans() {
        let (_d, idx) = idx_with(&[("a.md", "S1", "md", "xx ab yy")]);
        assert_same_hits(
            "ab",
            &GrepOpts {
                fixed: true,
                ..Default::default()
            },
            &idx,
        );
    }

    #[test]
    fn grep_smart_case_default_folds_lowercase_only() {
        let (_d, idx) = idx_with(&[("d.pdf", "p.1", "pdf", "Контроллер МОДУЛЬ подключён")]);
        assert_eq!(grep(&idx, "модуль", &GrepOpts::default()).unwrap().len(), 1);
        assert_eq!(grep(&idx, "Модуль", &GrepOpts::default()).unwrap().len(), 0);
    }

    #[test]
    fn grep_blank_pattern_returns_empty() {
        let (_d, idx) = idx_with(&[
            ("a.md", "S1", "md", "some content here"),
            ("b.md", "S1", "md", "more content there"),
        ]);
        assert!(grep(&idx, "", &GrepOpts::default()).unwrap().is_empty());
        assert!(grep(&idx, "   ", &GrepOpts::default()).unwrap().is_empty());
        assert!(grep(&idx, "\t", &GrepOpts::default()).unwrap().is_empty());
    }

    // ── new ripgrep-parity flags ────────────────────────────────────────────

    /// Default output (no new flags) is byte-identical to the historical format.
    #[test]
    fn grep_default_display_is_unchanged() {
        let (_d, idx) = idx_with(&[("d.md", "S1", "md", "alpha\nbeta match here\ngamma")]);
        let h = grep(&idx, "match", &GrepOpts::default()).unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].kind, HitKind::Match);
        assert_eq!(h[0].line_no, 0);
        assert_eq!(h[0].display_line(), "d.md:#1: beta match here");
    }

    #[test]
    fn grep_context_before_after_and_overlap_merge() {
        let (_d, idx) = idx_with(&[("d.md", "S1", "md", "l1\nl2\nHIT a\nl4\nl5\nHIT b\nl7")]);
        // -A1 -B1 around two matches that do NOT overlap → each match yields 3 lines.
        let h = grep(
            &idx,
            "HIT",
            &GrepOpts {
                before: 1,
                after: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let lines: Vec<_> = h.iter().map(|x| (x.line.as_str(), x.kind)).collect();
        assert_eq!(
            lines,
            vec![
                ("l2", HitKind::Context),
                ("HIT a", HitKind::Match),
                ("l4", HitKind::Context),
                ("l5", HitKind::Context),
                ("HIT b", HitKind::Match),
                ("l7", HitKind::Context),
            ]
        );
        // Wider context makes the two windows overlap; the shared middle lines must NOT repeat.
        let h = grep(
            &idx,
            "HIT",
            &GrepOpts {
                before: 2,
                after: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let seen: Vec<_> = h.iter().map(|x| x.line.clone()).collect();
        let uniq: std::collections::BTreeSet<_> = seen.iter().cloned().collect();
        assert_eq!(
            seen.len(),
            uniq.len(),
            "overlapping windows must de-dup: {seen:?}"
        );
        // covers the whole 7-line chunk exactly once
        assert_eq!(h.len(), 7);
        // context lines render with '-' separators
        let ctx = h.iter().find(|x| x.kind == HitKind::Context).unwrap();
        assert!(
            ctx.display_line().contains("-#1-"),
            "{}",
            ctx.display_line()
        );
    }

    #[test]
    fn grep_only_matching_emits_substrings() {
        let (_d, idx) = idx_with(&[("d.md", "S1", "md", "code A12 and A34 here\nnope")]);
        let h = grep(
            &idx,
            "A[0-9]+",
            &GrepOpts {
                only_matching: true,
                ..Default::default()
            },
        )
        .unwrap();
        let got: Vec<_> = h.iter().map(|x| x.line.as_str()).collect();
        assert_eq!(got, vec!["A12", "A34"]);
        assert!(h.iter().all(|x| x.kind == HitKind::Match));
    }

    #[test]
    fn fixed_with_regex_metacharacters_honours_regex_intent() {
        // A small model routinely sets fixed=true yet writes a regex (`A|B`, `x.*y`).
        // The intent must be honoured through BOTH the trigram prefilter and the
        // matcher — clearing fixed only in the matcher left the prefilter hunting a
        // literal pipe and finding zero candidate chunks (the real bug).
        let (_d, idx) = idx_with(&[(
            "d.md",
            "S1",
            "md",
            "row Таблица one\nrow Тип two\njust D three\nsize 3.0 mm\nsize 3X0 mm",
        )]);
        let alt: Vec<_> = grep(
            &idx,
            "Таблица|Тип",
            &GrepOpts {
                fixed: true,
                ..Default::default()
            },
        )
        .unwrap()
        .iter()
        .map(|h| h.line.clone())
        .collect();
        assert_eq!(alt, vec!["row Таблица one", "row Тип two"]);
        // No metacharacters → a true literal; a bare `.` stays literal, so "3.0" ≠ "3X0".
        let dec: Vec<_> = grep(
            &idx,
            "3.0",
            &GrepOpts {
                fixed: true,
                ..Default::default()
            },
        )
        .unwrap()
        .iter()
        .map(|h| h.line.clone())
        .collect();
        assert_eq!(dec, vec!["size 3.0 mm"]);
    }

    #[test]
    fn grep_line_number_threads_position() {
        let (_d, idx) = idx_with(&[("d.md", "S1", "md", "one\ntwo target\nthree")]);
        let h = grep(
            &idx,
            "target",
            &GrepOpts {
                line_number: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].line_no, 2);
        assert_eq!(h[0].display_line(), "d.md:#1:2: two target");
    }

    #[test]
    fn grep_count_reports_per_chunk_total() {
        let (_d, idx) = idx_with(&[
            ("a.md", "S1", "md", "hit\nmiss\nhit\nhit"),
            ("b.md", "S1", "md", "miss only"),
        ]);
        let h = grep(
            &idx,
            "hit",
            &GrepOpts {
                count: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(h.len(), 1, "only the chunk with matches reports a count");
        assert_eq!(h[0].kind, HitKind::Count);
        assert_eq!(h[0].line, "3");
        assert_eq!(h[0].display_line(), "a.md:#1: 3");
    }

    #[test]
    fn grep_max_count_stops_per_chunk() {
        let (_d, idx) = idx_with(&[("d.md", "S1", "md", "m\nm\nm\nm\nm")]);
        let h = grep(
            &idx,
            "m",
            &GrepOpts {
                max_count: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(h.len(), 2);
        // combined with -c the count reflects the cap
        let c = grep(
            &idx,
            "m",
            &GrepOpts {
                max_count: Some(2),
                count: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(c[0].line, "2");
    }

    #[test]
    fn grep_multiline_spans_lines() {
        let (_d, idx) = idx_with(&[("d.md", "S1", "md", "start\nBEGIN foo\nbar END\ntail")]);
        // `.` crosses the newline only with -U; the match starts on line 2.
        let h = grep(
            &idx,
            "BEGIN.*END",
            &GrepOpts {
                multiline: true,
                line_number: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].line_no, 2, "anchored to the line of the match start");
        // without -U the same pattern cannot match across the newline
        let none = grep(&idx, "BEGIN.*END", &GrepOpts::default()).unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn grep_multiline_only_matching_returns_spanned_text() {
        let (_d, idx) = idx_with(&[("d.md", "S1", "md", "x\nBEGIN a\nb END\ny")]);
        let h = grep(
            &idx,
            "BEGIN.*END",
            &GrepOpts {
                multiline: true,
                only_matching: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(h.len(), 1);
        assert!(
            h[0].line.contains("BEGIN") && h[0].line.contains("END"),
            "{}",
            h[0].line
        );
    }

    #[test]
    fn path_to_glob_maps_document_paths() {
        // plain path → recursive glob so bare and nested stored paths both match
        assert_eq!(path_to_glob("d.pdf"), "**/d.pdf");
        // a #chunk suffix (as read keys carry) is dropped
        assert_eq!(path_to_glob("d.pdf#12"), "**/d.pdf");
        // a non-numeric tail keeps the '#' (it is part of the name)
        assert_eq!(path_to_glob("weird#name.md"), "**/weird#name.md");
        // an explicit glob passes through unchanged
        assert_eq!(path_to_glob("**/*.pdf"), "**/*.pdf");
        assert_eq!(path_to_glob("*.md"), "*.md");
    }

    #[test]
    fn grep_path_scopes_to_one_document() {
        let (_d, idx) = idx_with(&[
            ("a.pdf", "p.1", "pdf", "параметр Скорость V10"),
            ("b.pdf", "p.1", "pdf", "параметр Скорость V20"),
        ]);
        let opts = GrepOpts {
            glob: Some(path_to_glob("a.pdf#1")),
            ..Default::default()
        };
        let h = grep(&idx, "Скорость", &opts).unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].path, "a.pdf");
    }

    #[test]
    fn grep_single_spaces_match_irregular_whitespace() {
        // Extracted PDF text often carries double spaces; a single-spaced query must match.
        let (_d, idx) = idx_with(&[(
            "d.pdf",
            "p.3",
            "pdf",
            "Пример  условного   обозначения изделия",
        )]);
        let h = grep(&idx, "Пример условного обозначения", &GrepOpts::default()).unwrap();
        assert_eq!(h.len(), 1, "single-spaced pattern must match double spaces");
        assert_same_hits("Пример условного обозначения", &GrepOpts::default(), &idx);
        // `fixed` with a space normalizes too, keeping its other characters literal
        let f = grep(
            &idx,
            "условного   обозначения",
            &GrepOpts {
                fixed: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(f.len(), 1, "fixed pattern with spaces must normalize");
    }

    #[test]
    fn grep_space_normalization_skips_char_classes() {
        let (_d, idx) = idx_with(&[("d.md", "S1", "md", "a-b next\na b other")]);
        // The space inside [...] must stay literal; only the outer text is affected.
        let h = grep(&idx, "a[- ]b", &GrepOpts::default()).unwrap();
        assert_eq!(h.len(), 2, "class space stays a literal alternative");
    }

    #[test]
    fn grep_line_cap_windows_giant_lines() {
        // A one-line "page" (no \n) of ~3000 chars: with a cap, the emitted match line
        // must stay within bounds and keep the hit visible.
        let giant = format!("{} maxTsdr {}", "x".repeat(1500), "y".repeat(1500));
        let (_d, idx) = idx_with(&[("d.pdf", "p.1", "pdf", giant.as_str())]);
        let opts = GrepOpts::default().with_default_context();
        assert_eq!(opts.line_cap, Some(500));
        let h = grep(&idx, "maxTsdr", &opts).unwrap();
        assert_eq!(h.len(), 1);
        assert!(
            h[0].line.chars().count() < 600,
            "capped: {} chars",
            h[0].line.chars().count()
        );
        assert!(h[0].line.contains("maxTsdr"), "match stays visible");
        assert!(h[0].line.contains("chars]"), "truncation is marked");
        // without a cap the core stays verbatim
        let full = grep(&idx, "maxTsdr", &GrepOpts::default()).unwrap();
        assert!(full[0].line.chars().count() > 2500);
    }

    #[test]
    fn grep_line_cap_truncates_context_tail() {
        let long_ctx = "c".repeat(1200);
        let body = format!("{long_ctx}\nHIT here\ntail");
        let (_d, idx) = idx_with(&[("d.md", "S1", "md", body.as_str())]);
        let opts = GrepOpts {
            before: 1,
            after: 1,
            line_cap: Some(100),
            ..Default::default()
        };
        let h = grep(&idx, "HIT", &opts).unwrap();
        let ctx = h.iter().find(|x| x.kind == HitKind::Context).unwrap();
        assert!(
            ctx.line.chars().count() < 130,
            "{}",
            ctx.line.chars().count()
        );
        assert!(ctx.line.contains("chars]"));
        assert!(ctx.line.starts_with('c'), "context keeps its head");
    }
}
