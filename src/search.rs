use crate::model::Chunk;
use regex::Regex;
use std::path::PathBuf;

#[derive(Debug, PartialEq, Eq)]
pub struct Hit {
    pub doc_path: PathBuf,
    pub location: String,
    pub line: usize,
    pub snippet: String,
}

pub fn search_chunks(chunks: &[Chunk], re: &Regex, limit: usize) -> Vec<Hit> {
    let mut hits = Vec::new();
    for c in chunks {
        for (i, line) in c.text.lines().enumerate() {
            if re.is_match(line) {
                hits.push(Hit {
                    doc_path: c.doc_path.clone(),
                    location: c.location.clone(),
                    line: i + 1,
                    snippet: line.trim().to_string(),
                });
                if hits.len() >= limit {
                    return hits;
                }
            }
        }
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(text: &str) -> Chunk {
        Chunk {
            doc_path: PathBuf::from("d.md"),
            location: "S".into(),
            file_type: "md".into(),
            text: text.into(),
        }
    }

    #[test]
    fn finds_matching_line_with_number_and_snippet() {
        let chunks = vec![chunk("alpha\nbeta cat\ngamma\n")];
        let re = Regex::new("cat").unwrap();
        let hits = search_chunks(&chunks, &re, 100);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].line, 2);
        assert_eq!(hits[0].snippet, "beta cat");
        assert_eq!(hits[0].location, "S");
    }

    #[test]
    fn respects_limit() {
        let chunks = vec![chunk("cat\ncat\ncat\n")];
        let re = Regex::new("cat").unwrap();
        let hits = search_chunks(&chunks, &re, 2);
        assert_eq!(hits.len(), 2);
    }
}
