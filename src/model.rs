use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub doc_path: PathBuf,
    pub location: Location,
    pub file_type: String,
    pub text: String,
}

impl Chunk {
    /// A chunk that sits inside containers the document itself names: a heading chain, a sheet, a
    /// slide. Segments that are empty or whitespace-only address nothing and would render as a
    /// doubled separator, so they are dropped. If nothing survives the chunk is flat rather than a
    /// `Path(vec![])` — an empty list is the normal case at the sites that call this (a heading-less
    /// markdown chunk, a docx with neither heading nor container title), not a programming error.
    pub fn nested(doc_path: PathBuf, file_type: String, path: Vec<String>, text: String) -> Chunk {
        let path: Vec<String> = path.into_iter().filter(|s| !s.trim().is_empty()).collect();
        let location = if path.is_empty() {
            Location::Label(String::new())
        } else {
            Location::Path(path)
        };
        Chunk {
            doc_path,
            location,
            file_type,
            text,
        }
    }

    /// A chunk whose location names a slice, an attachment or a condition rather than a container
    /// the author created: `rows 12-40`, `Chart 1`, `(image)`, `note`, `(no-text)`, or nothing at
    /// all. The label is kept verbatim — it is not a path and is never split or trimmed.
    pub fn labelled(doc_path: PathBuf, file_type: String, label: String, text: String) -> Chunk {
        Chunk {
            doc_path,
            location: Location::Label(label),
            file_type,
            text,
        }
    }
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
        let a = Chunk::nested(
            PathBuf::from("a.md"),
            "md".into(),
            vec!["Intro".into()],
            "hello".into(),
        );
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn nested_normalises_emptiness_only() {
        // An empty or whitespace segment has no address and would render as a double separator.
        let c = Chunk::nested(
            "d.md".into(),
            "md".into(),
            vec!["A".into(), "  ".into(), "B".into()],
            "t".into(),
        );
        assert_eq!(c.location, Location::Path(vec!["A".into(), "B".into()]));
        // An author's text is never rewritten beyond that.
        let c = Chunk::nested("d.md".into(), "md".into(), vec!["A > B".into()], "t".into());
        assert_eq!(c.location, Location::Path(vec!["A > B".into()]));
    }

    /// `Path(vec![])` is a bug rather than a flat chunk, and an empty list is the NORMAL case at two
    /// live sites (a heading-less markdown chunk; a docx with neither heading nor container title).
    #[test]
    fn nested_with_no_segments_is_an_empty_label() {
        let c = Chunk::nested("d.md".into(), "md".into(), vec![], "t".into());
        assert_eq!(c.location, Location::Label(String::new()));
        let c = Chunk::nested(
            "d.md".into(),
            "md".into(),
            vec!["".into(), " ".into()],
            "t".into(),
        );
        assert_eq!(c.location, Location::Label(String::new()));
    }

    #[test]
    fn labelled_keeps_its_string_verbatim() {
        let c = Chunk::labelled("d.pdf".into(), "pdf".into(), "(no-text)".into(), "t".into());
        assert_eq!(c.location, Location::Label("(no-text)".into()));
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
        match loc {
            Location::Path(segments) => {
                assert_eq!(segments, vec!["A > B".to_string(), "C".to_string()])
            }
            Location::Label(_) => panic!("expected a Path"),
        }
    }

    #[test]
    fn is_nested_distinguishes_the_two_shapes() {
        assert!(Location::Path(vec!["S".into()]).is_nested());
        assert!(!Location::Label("S".into()).is_nested());
    }
}
