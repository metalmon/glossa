use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub doc_path: PathBuf,
    pub location: String,
    pub file_type: String,
    pub text: String,
}

/// The separator the extractor used to join heading segments before `Location` existed.
/// `join_path` is the only place it is used to build a string back up.
pub const LOCATION_SEP: &str = " > ";

pub fn join_path(path: &[String]) -> String {
    path.join(LOCATION_SEP)
}

/// Where a chunk sits in its document: a path of heading segments for a nested chunk, or a
/// flat label (a row range, `(image)`, or empty) for one that isn't nested in anything.
/// `untagged` so a path serializes as a JSON array and a label as a JSON string, with no
/// tag field to carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Location {
    Path(Vec<String>),
    Label(String),
}

impl Location {
    /// Joins a `Path`'s segments with `LOCATION_SEP`, or returns a `Label` as-is. Deliberately
    /// not reversible: a segment may itself legally contain `LOCATION_SEP` (a markdown heading
    /// can contain " > "), so the joined string cannot be split back into the original segments.
    pub fn display(&self) -> String {
        match self {
            Location::Path(segments) => join_path(segments),
            Location::Label(label) => label.clone(),
        }
    }

    pub fn is_nested(&self) -> bool {
        matches!(self, Location::Path(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_constructs_and_compares() {
        let a = Chunk {
            doc_path: PathBuf::from("a.md"),
            location: "Intro".into(),
            file_type: "md".into(),
            text: "hello".into(),
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn display_joins_segments_and_returns_a_label_verbatim() {
        assert_eq!(
            Location::Path(vec!["Chapter 2".into(), "Limits".into()]).display(),
            "Chapter 2 > Limits"
        );
        assert_eq!(Location::Label("rows 12-40".into()).display(), "rows 12-40");
        assert_eq!(Location::Label(String::new()).display(), "");
    }

    /// A heading may legally contain the separator. The segments hold it verbatim; only `display()`
    /// ever joins, and the joined string is deliberately NOT re-splittable back into them.
    #[test]
    fn a_segment_may_contain_the_separator() {
        let loc = Location::Path(vec!["A > B".into(), "C".into()]);
        assert_eq!(loc.display(), "A > B > C");
        assert_eq!(loc, Location::Path(vec!["A > B".into(), "C".into()]));
    }

    #[test]
    fn is_nested_distinguishes_the_two_shapes() {
        assert!(Location::Path(vec!["S".into()]).is_nested());
        assert!(!Location::Label("S".into()).is_nested());
    }
}
