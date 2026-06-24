use crate::graph::store::{Edge, Node};
use std::io::IsTerminal;

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
}
