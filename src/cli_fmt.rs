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

fn pad(s: &str, w: usize) -> String {
    let len = s.chars().count();
    if len >= w { s.to_string() } else { format!("{s}{}", " ".repeat(w - len)) }
}

/// Render numbered, aligned, colored log-lines + a footer. Empty → "no results".
pub fn render_search_pretty(hits: &[DisplayHit]) -> String {
    if hits.is_empty() {
        return "no results".to_string();
    }
    let total = hits.len();
    let idx_w = total.to_string().len();
    let file_w = hits.iter().map(|h| h.file.chars().count()).max().unwrap_or(0);
    let loc_w = hits.iter().map(|h| h.location.chars().count()).max().unwrap_or(0);
    let mut out = String::new();
    for (i, h) in hits.iter().enumerate() {
        let idx = dim(&format!("[{:>w$}]", i + 1, w = idx_w));
        let file = pad(&h.file, file_w);          // default color
        let loc = cyan(&pad(&h.location, loc_w)); // color applied AFTER padding (keeps alignment)
        let mut line = format!("{idx} {file}  {loc}  {}", h.snippet.trim());
        if let Some(s) = h.score {
            line.push_str(&dim(&format!("  [{s:.3}]")));
        }
        out.push_str(&line);
        out.push('\n');
    }
    out.push_str(&dim(&format!("{total} results · kb read <#> to open")));
    out.push('\n');
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
    fn render_search_pretty_numbers_aligns_scores() {
        let hits = vec![
            DisplayHit { file: "a.md".into(), location: "p.1".into(), snippet: " hello ".into(), score: None },
            DisplayHit { file: "longname.pdf".into(), location: "Sec".into(), snippet: "world".into(), score: Some(1.234) },
        ];
        let s = render_search_pretty(&hits);
        assert!(s.contains("[1] a.md"));
        assert!(s.contains("[2] longname.pdf"));
        assert!(s.contains("hello"));              // snippet trimmed
        assert!(s.contains("[1.234]"));            // score shown for ranked hit
        assert!(s.contains("2 results · kb read <#> to open"));
    }

    #[test]
    fn render_search_pretty_empty_is_no_results() {
        assert_eq!(render_search_pretty(&[]), "no results");
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
