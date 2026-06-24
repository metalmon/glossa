use crate::graph::store::{Edge, Node};
use std::io::IsTerminal;
use std::path::Path;

/// True when stdout is an interactive terminal and color isn't disabled via NO_COLOR.
pub fn use_color() -> bool {
    std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none()
}

fn paint(code: &str, s: &str) -> String {
    if use_color() {
        format!("\x1b[{code}m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}
pub fn dim(s: &str) -> String { paint("2", s) }
pub fn cyan(s: &str) -> String { paint("36", s) }
pub fn bold(s: &str) -> String { paint("1", s) }
pub fn magenta(s: &str) -> String { paint("35", s) }
pub fn green(s: &str) -> String { paint("32", s) }

/// Query tokens to highlight in snippets: alphanumeric runs of length ≥ 2.
fn query_terms(q: &str) -> Vec<String> {
    q.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.chars().count() >= 2)
        .map(|t| t.to_string())
        .collect()
}

/// Wrap every case-insensitive occurrence of any `term` in `text` using `wrap`. Pure (no TTY),
/// so it is unit-testable; the color wrapper is applied by `highlight`.
fn highlight_with(text: &str, terms: &[String], wrap: impl Fn(&str) -> String) -> String {
    // Case-insensitive matching needs byte offsets that line up between `text` and its lowercase
    // form; bail (no highlight) if lowercasing changes the byte length (rare: ß, İ, …).
    let lower = text.to_lowercase();
    if terms.is_empty() || lower.len() != text.len() {
        return text.to_string();
    }
    let mut marked = vec![false; text.len()];
    for t in terms {
        let tl = t.to_lowercase();
        if tl.len() < 2 {
            continue;
        }
        let mut start = 0;
        while let Some(pos) = lower[start..].find(&tl) {
            let s = start + pos;
            let e = s + tl.len();
            for b in marked.iter_mut().take(e).skip(s) {
                *b = true;
            }
            start = e;
        }
    }
    let mut out = String::new();
    let mut i = 0;
    while i < text.len() {
        let s = i;
        let on = marked[i];
        while i < text.len() && marked[i] == on {
            i += 1;
        }
        let run = &text[s..i]; // run boundaries are term-aligned → valid char boundaries
        if on {
            out.push_str(&wrap(run));
        } else {
            out.push_str(run);
        }
    }
    out
}

/// Highlight query terms in a snippet (bold red), only when color is enabled.
fn highlight(text: &str, terms: &[String]) -> String {
    if use_color() {
        highlight_with(text, terms, |m| paint("1;31", m))
    } else {
        text.to_string()
    }
}

/// True when stdout is an interactive terminal (independent of NO_COLOR — used for format choice).
pub fn stdout_is_tty() -> bool {
    std::io::stdout().is_terminal()
}

/// One search hit prepared for the pretty (log-line) view.
pub struct DisplayHit {
    pub file: String,      // path shown to the human (relative to root if possible)
    pub location: String,  // "p.142" | heading | "(no-text)"
    pub snippet: String,
    pub score: Option<f32>, // Some for --rank
}

/// Render results as two-line entries — `#N  score  path  · loc`, then the snippet indented below —
/// so long paths never collide with the snippet. `score` is shown first (for `--rank`). The `#N` is
/// the result number for `kb read <#>` and always counts from the most relevant hit (`#1`).
///
/// When `reverse` is true (used for `--rank`), entries print worst-first so the **most relevant hit
/// sits last, right above the prompt** — no scrolling up — and the footer moves to the top. `#1` still
/// labels the best hit. Empty → "no results".
pub fn render_search_pretty(hits: &[DisplayHit], reverse: bool, query: &str) -> String {
    if hits.is_empty() {
        return "no results".to_string();
    }
    let total = hits.len();
    let idx_w = total.to_string().len();
    let terms = query_terms(query);
    let footer = dim(&format!("{total} results · kb read <#> to open"));

    let entry = |i: usize, h: &DisplayHit| -> String {
        // Line 1: #N  [score]  path   (path kept alone so a long path can't orphan the location)
        let mut l1 = dim(&format!("#{:<w$}", i + 1, w = idx_w));
        l1.push_str("  ");
        if let Some(s) = h.score {
            l1.push_str(&cyan(&format!("{s:.3}")));
            l1.push_str("  ");
        }
        l1.push_str(&magenta(&h.file));
        // Line 2: indented  location  snippet (with the matched term highlighted).
        let mut l2 = String::from("    ");
        if !h.location.is_empty() {
            l2.push_str(&green(&h.location));
            l2.push_str("  ");
        }
        l2.push_str(&highlight(h.snippet.trim(), &terms));
        format!("{l1}\n{l2}\n\n")
    };

    let mut out = String::new();
    if reverse {
        out.push_str(&footer);
        out.push_str("\n\n");
        for (i, h) in hits.iter().enumerate().rev() {
            out.push_str(&entry(i, h));
        }
        // Drop the trailing blank line so the best hit sits right above the prompt.
        while out.ends_with('\n') {
            out.pop();
        }
        out.push('\n');
    } else {
        for (i, h) in hits.iter().enumerate() {
            out.push_str(&entry(i, h));
        }
        out.push_str(&footer);
        out.push('\n');
    }
    out
}

/// Display name for a hit: path relative to `root` if possible, else the file name.
pub fn rel_file(root: &Path, p: &str) -> String {
    let pp = Path::new(p);
    match pp.strip_prefix(root) {
        Ok(r) => r.display().to_string(),
        Err(_) => pp
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| p.to_string()),
    }
}

/// Persist the ordered hits (path<TAB>location) so `kb read <#>` can resolve them later.
pub fn write_last_search(root: &Path, records: &[(String, String)]) -> std::io::Result<()> {
    let dir = root.join(".glossa");
    std::fs::create_dir_all(&dir)?;
    let mut s = String::new();
    for (p, loc) in records {
        s.push_str(p);
        s.push('\t');
        s.push_str(loc);
        s.push('\n');
    }
    std::fs::write(dir.join("last_search.tsv"), s)
}

/// Read the raw last-search TSV (None if absent).
pub fn read_last_search(root: &Path) -> Option<String> {
    std::fs::read_to_string(root.join(".glossa").join("last_search.tsv")).ok()
}

/// 1-based lookup of a `path<TAB>location` record in last-search TSV content.
pub fn nth_record(tsv: &str, n: usize) -> Option<(String, String)> {
    if n == 0 {
        return None;
    }
    let line = tsv.lines().nth(n - 1)?;
    let mut it = line.splitn(2, '\t');
    let path = it.next()?.to_string();
    let loc = it.next().unwrap_or("").to_string();
    Some((path, loc))
}

/// Human-readable rendering of a graph node with its outgoing edges.
pub fn render_node(n: &Node, edges: &[Edge]) -> String {
    let mut out = String::new();
    out.push_str(&format!("{} {}\n", dim("node:"), bold(&n.id)));
    out.push_str(&format!("  type:   {}\n", n.node_type));
    out.push_str(&format!("  label:  {}\n", n.label));
    if !n.aliases.is_empty() {
        out.push_str(&format!("  aliases: {}\n", n.aliases.join(", ")));
    }
    let p = &n.prov;
    out.push_str(&format!(
        "  origin: {}   confidence: {:.2}   created_at: {}\n",
        p.origin, p.confidence, p.created_at
    ));
    match &p.range {
        Some(r) => out.push_str(&format!("  source: {} @ {}\n", p.source_path, r)),
        None => out.push_str(&format!("  source: {}\n", p.source_path)),
    }
    if edges.is_empty() {
        out.push_str(&format!("  {}\n", dim("edges: none")));
    } else {
        out.push_str(&format!("  edges ({}):\n", edges.len()));
        for e in edges {
            out.push_str(&format!("    --{}--> {}\n", e.edge_type, e.to));
        }
    }
    out
}

/// Human-readable rendering of a path query result.
pub fn render_path(found: Option<&Vec<String>>, from: &str, to: &str, max_depth: usize) -> String {
    match found {
        Some(p) => {
            let hops = p.len().saturating_sub(1);
            format!("path ({hops} hops): {}", p.join(" → "))
        }
        None => format!("no path from {from} to {to} within depth {max_depth}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::store::{Edge, Node, Provenance};

    fn prov() -> Provenance {
        Provenance {
            source_path: "a.md".into(), range: Some("Intro".into()), file_sig: None,
            origin: "auto-structural".into(), confidence: 1.0, created_at: 0,
        }
    }

    #[test]
    fn render_node_shows_fields_and_edges() {
        let n = Node {
            id: "a.md".into(), node_type: "Document".into(), label: "a.md".into(),
            aliases: vec!["alpha".into()], prov: prov(),
        };
        let edges = vec![Edge {
            from: "a.md".into(), to: "a.md#Intro".into(), edge_type: "CONTAINS".into(), prov: prov(),
        }];
        let s = render_node(&n, &edges);
        assert!(s.contains("node: a.md"));
        assert!(s.contains("type:   Document"));
        assert!(s.contains("aliases: alpha"));
        assert!(s.contains("source: a.md @ Intro"));
        assert!(s.contains("--CONTAINS--> a.md#Intro"));
    }

    #[test]
    fn render_node_handles_no_edges() {
        let n = Node {
            id: "x".into(), node_type: "Term".into(), label: "x".into(),
            aliases: vec![], prov: prov(),
        };
        assert!(render_node(&n, &[]).contains("edges: none"));
    }

    #[test]
    fn render_path_found_and_missing() {
        let p = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(render_path(Some(&p), "a", "c", 6), "path (2 hops): a → b → c");
        assert_eq!(render_path(None, "a", "z", 6), "no path from a to z within depth 6");
    }

    #[test]
    fn render_search_pretty_two_line_score_first() {
        let hits = vec![
            DisplayHit { file: "a.md".into(), location: "p.1".into(), snippet: " hello ".into(), score: None },
            DisplayHit { file: "deep\\longname.pdf".into(), location: "Sec".into(), snippet: "world".into(), score: Some(1.234) },
        ];
        let s = render_search_pretty(&hits, false, "hello");
        // Line 1: number, then (for ranked) score BEFORE the path.
        assert!(s.contains("#1  a.md"));
        assert!(s.contains("#2  1.234  deep\\longname.pdf"), "score precedes the path");
        // Line 2: location leads the indented snippet line (page can't orphan off a long path).
        assert!(s.contains("\n    p.1  hello\n"));
        assert!(s.contains("2 results · kb read <#> to open"));
        // Forward order: #1 printed before #2; footer last.
        assert!(s.find("#1").unwrap() < s.find("#2").unwrap());
        assert!(s.find("#2").unwrap() < s.find("2 results").unwrap());
    }

    #[test]
    fn render_search_pretty_reverse_puts_best_last() {
        let hits = vec![
            DisplayHit { file: "best.md".into(), location: "p.1".into(), snippet: "a".into(), score: Some(9.0) },
            DisplayHit { file: "worst.md".into(), location: "p.2".into(), snippet: "b".into(), score: Some(1.0) },
        ];
        let s = render_search_pretty(&hits, true, "");
        // Footer at the top; worst (#2) above best (#1), so #1 is nearest the prompt.
        let f = s.find("2 results").unwrap();
        let p1 = s.find("#1").unwrap();
        let p2 = s.find("#2").unwrap();
        assert!(f < p2 && p2 < p1, "footer top, then #2 (worst), then #1 (best) last");
        // #1 still labels the most relevant hit (best.md).
        assert!(s.contains("#1  9.000  best.md"));
    }

    #[test]
    fn highlight_with_wraps_case_insensitive_cyrillic() {
        let out = highlight_with(
            "Проведение Поверки и поверка",
            &["поверк".to_string()],
            |m| format!("[{m}]"),
        );
        assert_eq!(out, "Проведение [Поверк]и и [поверк]а");
    }

    #[test]
    fn query_terms_keeps_words_of_len_2_plus() {
        assert_eq!(query_terms("АБАК ПЛК"), vec!["АБАК".to_string(), "ПЛК".to_string()]);
        assert!(query_terms("a .").is_empty());
    }

    #[test]
    fn render_search_pretty_empty_is_no_results() {
        assert_eq!(render_search_pretty(&[], false, ""), "no results");
    }

    #[test]
    fn nth_record_is_1based_and_bounded() {
        let tsv = "a.md\tp.1\nb.pdf\t(no-text)\n";
        assert_eq!(nth_record(tsv, 1), Some(("a.md".into(), "p.1".into())));
        assert_eq!(nth_record(tsv, 2), Some(("b.pdf".into(), "(no-text)".into())));
        assert_eq!(nth_record(tsv, 3), None);
        assert_eq!(nth_record(tsv, 0), None);
    }
}
