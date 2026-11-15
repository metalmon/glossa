use crate::model::Chunk;
use regex::Regex;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, PartialEq, Eq)]
pub struct Hit {
    pub doc_path: PathBuf,
    pub location: String,
    pub ord: u64,
    pub line: usize,
    pub snippet: String,
}

pub fn search_chunks(chunks: &[Chunk], re: &Regex, limit: usize) -> Vec<Hit> {
    if limit == 0 {
        return Vec::new();
    }
    let mut hits = Vec::new();
    // The ordinal addresses the CHUNK, not the match, so it advances once per
    // chunk in this outer loop regardless of how many lines inside match (or
    // none). Keyed by doc_path rather than a single running counter, so it is
    // correct even if a document's chunks are not contiguous in `chunks`.
    let mut next_ord: HashMap<PathBuf, u64> = HashMap::new();
    for c in chunks {
        let counter = next_ord.entry(c.doc_path.clone()).or_insert(0);
        *counter += 1;
        let ord = *counter;
        for (i, line) in c.text.lines().enumerate() {
            if re.is_match(line) {
                hits.push(Hit {
                    doc_path: c.doc_path.clone(),
                    location: c.location.clone(),
                    ord,
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

    fn chunk_in(doc_path: &str, text: &str) -> Chunk {
        Chunk {
            doc_path: PathBuf::from(doc_path),
            location: "S".into(),
            file_type: "md".into(),
            text: text.into(),
        }
    }

    fn chunk(text: &str) -> Chunk {
        chunk_in("d.md", text)
    }

    #[test]
    fn ordinals_are_per_document_and_one_based() {
        // Chunks interleave A, B, A, B so a single running counter (not keyed
        // by doc_path) would number hits 1, 2, 3, 4 instead of the correct
        // 1, 1, 2, 2 — the per-document count must restart for B.
        let chunks = vec![
            chunk_in("a.md", "alpha needle"),
            chunk_in("b.md", "delta needle"),
            chunk_in("a.md", "beta needle"),
            chunk_in("b.md", "epsilon needle"),
        ];
        let hits = search_chunks(&chunks, &Regex::new("needle").unwrap(), 10);
        let got: Vec<(String, u64)> = hits
            .iter()
            .map(|h| (h.doc_path.display().to_string(), h.ord))
            .collect();
        assert_eq!(
            got,
            vec![
                ("a.md".into(), 1),
                ("b.md".into(), 1),
                ("a.md".into(), 2),
                ("b.md".into(), 2),
            ]
        );
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
    fn respects_limit_and_shares_ordinal_across_matches_in_one_chunk() {
        let chunks = vec![chunk("cat\ncat\ncat\n")];
        let re = Regex::new("cat").unwrap();
        let hits = search_chunks(&chunks, &re, 2);
        assert_eq!(hits.len(), 2);
        // Both hits come from the same chunk, so the ordinal is computed
        // once for the chunk and must be identical on every hit it produces
        // — a regression that recomputed `ord` per matching line (instead of
        // once per chunk, before the inner loop) would still pass every
        // other assertion here but diverge on this one.
        assert_eq!(hits[0].ord, hits[1].ord);
    }

    #[test]
    fn limit_zero_returns_no_hits() {
        let chunks = vec![chunk("cat\ncat\n")];
        let re = Regex::new("cat").unwrap();
        let hits = search_chunks(&chunks, &re, 0);
        assert_eq!(hits.len(), 0);
    }

    #[test]
    fn non_matching_chunk_still_consumes_an_ordinal() {
        // Chunk 2 has no matching line but must still advance the counter,
        // so chunk 3's hit reports ord 3, not 2. A regression that gated the
        // counter on `re.is_match` (incrementing only when a chunk produces
        // a hit) would instead report ord 2 here.
        let chunks = vec![chunk("cat\n"), chunk("dog\n"), chunk("cat\n")];
        let re = Regex::new("cat").unwrap();
        let hits = search_chunks(&chunks, &re, 10);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].ord, 1);
        assert_eq!(hits[1].ord, 3);
    }
}
