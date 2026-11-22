//! Placing a chart chunk where the chart occurs, rather than after all the body text.
//!
//! Every format that can recover an embedded object's position calls `splice_charts_at_anchors`.
//! The chart chunks themselves are built exactly as they were — same title, same rendered text,
//! same `Location`; this module decides position and nothing else.

use crate::extract::office_chunk::ChunkSpan;
use crate::model::Chunk;

/// Where an extractor saw an embedded object: `(section index, element index)` into the
/// `DocumentIR` the body chunks were built from.
///
/// Both coordinates are needed. `chunk_ir` pushes every section's chunks into one shared vector,
/// so element indices restart per section while chunk indices do not: element 2 of sheet 1 and
/// element 2 of sheet 2 are the same number.
pub type ChartAnchor = (usize, usize);

/// Place each chart chunk immediately after the body chunk whose span contains its anchor.
///
/// A chart whose anchor resolves to no chunk — it fell in a gap between spans, past the last
/// element, or into a section that emitted nothing — is **appended at the end**, which is where
/// every chart used to go. Never the nearest chunk: a chart at a plausible-but-wrong position
/// yields a citation that reads as precise and is false.
///
/// Charts anchored into the same chunk keep the order they were given in. With no charts the
/// vector is returned untouched.
pub fn splice_charts_at_anchors(
    chunks: Vec<Chunk>,
    spans: &[ChunkSpan],
    charts: Vec<(ChartAnchor, Chunk)>,
) -> Vec<Chunk> {
    if charts.is_empty() {
        return chunks;
    }
    // Resolve EVERY anchor before moving a single chunk. Each insertion shifts the index of every
    // chunk after it, so an implementation that splices as it resolves places its second chart one
    // position too early — `two_charts_in_one_document_both_land_correctly` is that defect.
    let mut placed: Vec<(usize, Chunk)> = Vec::with_capacity(charts.len());
    let mut appended: Vec<Chunk> = Vec::new();
    for (anchor, chart) in charts {
        match containing_chunk(spans, anchor) {
            Some(i) => placed.push((i, chart)),
            None => appended.push(chart),
        }
    }
    // Stable, so same-chunk charts stay in their given order.
    placed.sort_by_key(|(at, _)| *at);

    let mut out = Vec::with_capacity(chunks.len() + placed.len() + appended.len());
    let mut placed = placed.into_iter().peekable();
    for (i, chunk) in chunks.into_iter().enumerate() {
        out.push(chunk);
        while let Some((_, chart)) = placed.next_if(|(at, _)| *at == i) {
            out.push(chart);
        }
    }
    // A resolved index past the last chunk can only mean the caller paired a chunk vector with
    // someone else's spans. Append rather than drop: losing a chart is the worse failure.
    out.extend(placed.map(|(_, chart)| chart));
    out.extend(appended);
    out
}

/// The index of the chunk whose span contains `anchor`, or `None` when none does.
///
/// `pub(crate)` so a caller can ask the same question this function answers without splicing —
/// `office.rs` counts the anchors that resolve to nothing so it can say so once per file. A
/// second answer for the same anchor is one binary search, and the chart count is already bounded
/// per file by whatever recovered the anchors.
///
/// Binary search, not a scan: both the number of charts and the number of chunks come out of the
/// document, so a scan per anchor would let a crafted file buy their product. This needs no
/// skip-and-warn budget instead, because spans arrive sorted — `chunk_ir` walks sections in order
/// and flushes within a section in element order, so spans ascend by `(section, elements.start)`
/// and never overlap. The containment test below is still exact, so were that invariant ever
/// broken the worst outcome is a missed anchor, which appends: honest, not wrong.
pub(crate) fn containing_chunk(
    spans: &[ChunkSpan],
    (section, element): ChartAnchor,
) -> Option<usize> {
    let i = spans
        .partition_point(|s| (s.section, s.elements.start) <= (section, element))
        .checked_sub(1)?;
    let span = &spans[i];
    (span.section == section && span.elements.contains(&element)).then_some(i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn body(text: &str) -> Chunk {
        Chunk::nested(
            PathBuf::from("doc.ods"),
            "ods".to_string(),
            vec![],
            text.to_string(),
        )
    }

    fn chart(title: &str) -> Chunk {
        Chunk::labelled(
            PathBuf::from("doc.ods"),
            "ods".to_string(),
            title.to_string(),
            format!("Chart: {title}"),
        )
    }

    fn span(section: usize, elements: std::ops::Range<usize>) -> ChunkSpan {
        ChunkSpan { section, elements }
    }

    /// The text of every chunk, in order — the only thing these tests are about.
    fn texts(chunks: &[Chunk]) -> Vec<&str> {
        chunks.iter().map(|c| c.text.as_str()).collect()
    }

    #[test]
    fn charts_are_spliced_after_the_chunk_containing_their_anchor() {
        let chunks = vec![body("first"), body("second"), body("third")];
        let spans = vec![span(0, 0..2), span(0, 2..4), span(0, 4..6)];
        let out = splice_charts_at_anchors(chunks, &spans, vec![((0, 3), chart("Sales"))]);
        assert_eq!(
            texts(&out),
            vec!["first", "second", "Chart: Sales", "third"]
        );
    }

    #[test]
    fn two_charts_in_one_document_both_land_correctly() {
        let chunks = vec![body("first"), body("second"), body("third")];
        let spans = vec![span(0, 0..2), span(0, 2..4), span(0, 4..6)];
        // Anchors into the first and the last chunk. An implementation that splices the first
        // chart before it has resolved the second places the second one chunk too early, because
        // the insertion shifted every later chunk index by one.
        let out = splice_charts_at_anchors(
            chunks,
            &spans,
            vec![((0, 1), chart("Sales")), ((0, 5), chart("Costs"))],
        );
        assert_eq!(
            texts(&out),
            vec!["first", "Chart: Sales", "second", "third", "Chart: Costs",]
        );
    }

    #[test]
    fn two_charts_in_the_same_chunk_keep_their_given_order() {
        let chunks = vec![body("first"), body("second")];
        let spans = vec![span(0, 0..2), span(0, 2..4)];
        let out = splice_charts_at_anchors(
            chunks,
            &spans,
            vec![((0, 0), chart("Sales")), ((0, 1), chart("Costs"))],
        );
        assert_eq!(
            texts(&out),
            vec!["first", "Chart: Sales", "Chart: Costs", "second"]
        );
    }

    #[test]
    fn an_anchor_with_no_containing_chunk_appends_at_the_end() {
        let chunks = vec![body("first"), body("second")];
        // Elements 2 and 3 render to nothing, so they belong to no chunk: a gap, by design.
        let spans = vec![span(0, 0..2), span(0, 4..6)];
        let out = splice_charts_at_anchors(
            chunks,
            &spans,
            vec![
                ((0, 3), chart("InGap")),
                ((0, 99), chart("PastTheEnd")),
                ((7, 0), chart("NoSuchSection")),
            ],
        );
        assert_eq!(
            texts(&out),
            vec![
                "first",
                "second",
                "Chart: InGap",
                "Chart: PastTheEnd",
                "Chart: NoSuchSection",
            ]
        );
    }

    #[test]
    fn no_charts_means_the_chunk_vector_is_returned_unchanged() {
        let chunks = vec![body("first"), body("second")];
        let spans = vec![span(0, 0..1), span(0, 1..2)];
        let before = chunks.clone();
        assert_eq!(splice_charts_at_anchors(chunks, &spans, vec![]), before);
    }

    #[test]
    fn anchors_in_two_sections_sharing_an_element_index_land_in_different_chunks() {
        // The multi-sheet case the section index exists for: element 0 of sheet 1 and element 0 of
        // sheet 2 are the same number, and a span lacking the section cannot tell them apart.
        let chunks = vec![body("sheet one"), body("sheet two")];
        let spans = vec![span(0, 0..1), span(1, 0..1)];
        let out = splice_charts_at_anchors(
            chunks,
            &spans,
            vec![((1, 0), chart("Second")), ((0, 0), chart("First"))],
        );
        assert_eq!(
            texts(&out),
            vec!["sheet one", "Chart: First", "sheet two", "Chart: Second",]
        );
    }
}
