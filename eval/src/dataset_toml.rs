//! `dataset.toml` parser — the file-first `kbx` eval toolkit's own case format
//! (`[[case]] id/question/answer/aliases/tags`), distinct from the JSON/JSONL loaders in
//! `dataset.rs` (hotpot/musique/questions). Produces the same shared `Question` shape so the
//! rest of the toolkit (scoring, reporting) doesn't need to know which format a run used.

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
}

#[derive(Deserialize)]
struct File {
    #[serde(default)]
    case: Vec<RawCase>,
}

/// Parse a `dataset.toml` (`[[case]] id/question/answer/aliases/tags`) into the shared
/// `Question` shape. `answer` supports TOML's native multi-line `"""` strings. `aliases` maps to
/// `Question::answer_aliases`; `tags` is carried as-is. Both are optional and default to empty.
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
        assert!(cs[1].answer_aliases.is_empty() && cs[1].tags.is_empty());
    }
}
