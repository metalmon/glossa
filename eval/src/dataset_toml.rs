//! `dataset.toml` parser — the file-first `kbx` eval toolkit's own case format
//! (`[[case]] id/question/answer/aliases/tags/hop_type/needs_graph`), distinct from the
//! JSON/JSONL loaders in `dataset.rs` (hotpot/musique/questions). Produces the same shared
//! `Question` shape so the rest of the toolkit (scoring, reporting) doesn't need to know which
//! format a run used.

use crate::dataset::Question;
use anyhow::Context;
use serde::Deserialize;

#[derive(Deserialize)]
struct RawCase {
    id: String,
    question: String,
    answer: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    tags: Vec<String>,
    /// Reasoning shape the case exercises (`lexical|multihop|mixed`); eval metadata used to slice
    /// the report by question type. Optional -- defaults to empty ("(untyped)" in the report).
    #[serde(default)]
    hop_type: String,
    /// Whether answering the case requires graph-based reasoning (`yes|no|maybe`); same reporting
    /// role as `hop_type`, on a separate axis.
    #[serde(default)]
    needs_graph: String,
    /// Source chunk refs (`"<path>#<location>"`) behind this gold, e.g.
    /// `source = ["a.pdf#p.1", "b.pdf#p.2"]`. Optional — defaults to empty, in which case the
    /// judge grades gold-only (unchanged behavior). When present, the judge loads these chunks as
    /// EVIDENCE and credits a correct answer that exceeds the terse gold.
    #[serde(default)]
    source: Vec<String>,
}

#[derive(Deserialize)]
struct File {
    #[serde(default)]
    case: Vec<RawCase>,
}

/// Parse a `dataset.toml` (`[[case]] id/question/answer/aliases/tags/hop_type/needs_graph`) into
/// the shared `Question` shape. `answer` supports TOML's native multi-line `"""` strings.
/// `aliases` maps to `Question::answer_aliases`; `tags`, `hop_type`, `needs_graph` are carried
/// as-is. All are optional and default to empty (`hop_type`/`needs_graph` render as
/// `"(untyped)"` in the by-question-type report section).
pub fn parse_dataset_toml(text: &str) -> anyhow::Result<Vec<Question>> {
    let file: File = toml::from_str(text).context("parse dataset.toml")?;
    Ok(file
        .case
        .into_iter()
        .map(|c| Question {
            id: c.id,
            question: c.question,
            answer: c.answer,
            answer_aliases: c.aliases,
            paragraphs: Vec::new(),
            supporting_titles: Vec::new(),
            tags: c.tags,
            hop_type: c.hop_type,
            needs_graph: c.needs_graph,
            source: c.source,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cases_with_multiline_and_optional_fields() {
        let t = r#"
[[case]]
id="q1"
question="Q one?"
answer="""
line A
line B
"""
aliases=["alt"]
tags=["net"]
hop_type="multihop"
needs_graph="yes"

[[case]]
id="q2"
question="Q two?"
answer="short"
"#;
        let cs = parse_dataset_toml(t).unwrap();
        assert_eq!(cs.len(), 2);
        assert!(cs[0].answer.contains("line A") && cs[0].answer.contains("line B"));
        assert_eq!(cs[0].answer_aliases, vec!["alt"]);
        assert_eq!(cs[0].tags, vec!["net"]);
        assert_eq!(cs[0].hop_type, "multihop");
        assert_eq!(cs[0].needs_graph, "yes");
        assert!(cs[1].answer_aliases.is_empty() && cs[1].tags.is_empty());
        // hop_type/needs_graph are optional and default to empty when the case omits them.
        assert!(cs[1].hop_type.is_empty());
        assert!(cs[1].needs_graph.is_empty());
    }

    #[test]
    fn parses_source_refs_and_defaults_empty() {
        let t = r#"
[[case]]
id="q1"
question="Q one?"
answer="short"
source=["a.pdf#p.1", "b.pdf#p.2"]

[[case]]
id="q2"
question="Q two?"
answer="short"
"#;
        let cs = parse_dataset_toml(t).unwrap();
        assert_eq!(cs.len(), 2);
        // A case with `source` parses to its list of refs, in order.
        assert_eq!(
            cs[0].source,
            vec!["a.pdf#p.1".to_string(), "b.pdf#p.2".to_string()]
        );
        // Absent `source` defaults to an empty vec (gold-only judging).
        assert!(cs[1].source.is_empty());
    }
}
