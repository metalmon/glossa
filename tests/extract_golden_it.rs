//! A golden pin on what the document extractors actually emit.
//!
//! It covers every document extractor's committed fixtures — office (docx, xlsx, xlsb, legacy doc
//! and xls), rtf and odf — pinning each one's exact chunk count, exact chunk text and exact order.
//!
//! Every other extraction test asserts a *property* — "a chart chunk exists", "the text contains
//! the marker", "the stored location equals the emitted one". None of them can witness an
//! extractor upgrade changing the words we index, or a change in the order we emit them in: a
//! library can start or stop emitting a paragraph, a table row, a sheet name or a chart's text, a
//! chart chunk can move from the end of a document to its anchor, and every one of those
//! assertions still holds. This file is the tripwire that does not.
//!
//! **A red here is not a regression by default.** It means the extracted output moved — almost
//! always because an extractor library (or our own chunker, or a chart's placement) was changed on
//! purpose. The right response is to read the diff the failure prints, judge whether the new output
//! is an improvement, and then update the expectation below. Do not "fix" an extractor to satisfy
//! this test, and do not update the expectation without reading what changed: the diff is the only
//! review this path gets. These values are a tripwire, not a specification — they record what we
//! emit today, not what we promise to emit. An `INDEX_SCHEMA_VERSION` bump usually belongs with the
//! change that moves these values, because a corpus indexed with the old output has to be rebuilt
//! to get the new one.
//!
//! The fixtures are the committed synthetic ones (`tests/fixtures/README.md`), so these values are
//! stable across machines and do not depend on any corpus.

use glossa::extract::extract_file;
use std::path::Path;

/// `fixture file name` -> the exact text of every chunk it yields, in order.
///
/// The chunk *count* is pinned by the length of each slice, and the text by its contents. Both
/// matter: a split that moves changes the count without changing the concatenated text.
/// Harvested from `office_oxide 0.1.17` on 2026-11-17. There is no `.pptx`/`.ppt` fixture in the
/// repo and no plain `sample.xlsx` or `sample.rtf`; add their rows here if any is ever committed.
///
/// Every `.rtf`, `.odt`, `.ods` and `.odp` fixture in `tests/fixtures/` is listed. If one is added
/// there, add its row here too — an unlisted fixture is an unpinned fixture.
const EXPECTED: &[(&str, &[&str])] = &[
    (
        "sample.docx",
        &["Sample\n\nglossa sample document used as a neutral test fixture."],
    ),
    (
        "sample_table.docx",
        &["| H1 | H2 |\n| --- | --- |\n| R1 | R2 |"],
    ),
    (
        // Two chunks: the body, byte-identical to `sample.docx`'s, and our own chart chunk. The
        // body is the control — this fixture's `word/document.xml` has no `w:drawing`, so the
        // chart text must reach the index ONLY through the `Chart:` chunk. If the first string
        // ever grows the chart's numbers, `office_oxide` has started putting chart text in the
        // body flow for this file and it is being indexed twice.
        "sample_chart.docx",
        &[
            "Sample\n\nglossa sample document used as a neutral test fixture.",
            "Chart: Sales by quarter (barChart)\n\n|  | Series 1 | Series 2 |\n| --- | --- | --- |\n| Q1 | 4.3 | 2.4 |\n| Q2 | 2.5 | 4.4 |\n| Q3 | 3.5 | 3.5 |",
        ],
    ),
    (
        // The anchored counterpart: its chart part IS referenced from the body
        // (`w:drawing` -> `a:graphicData` -> `c:chart r:id`), so `office_oxide` hoists the
        // chart's text into the body flow and `drop_hoisted_docx_chart_text` takes that copy
        // back out. If the body string below ever grows "Title: Sales by quarter" or a bare
        // "Series 2: ..." line, the de-duplication stopped firing and the chart is indexed
        // twice. The SmartArt lines ("Diagram node ...") belong in the body and must stay:
        // nothing of ours re-emits them.
        "sample_chart_anchored.docx",
        &[
            "Sample\n\nglossa sample document used as a neutral test fixture.\n\nBody text that follows the chart.\n\nDiagram node alpha\n\nDiagram node beta\n\nBody text that follows the diagram.",
            "Chart: Sales by quarter (barChart)\n\n|  | Series 1 | Series 2 |\n| --- | --- | --- |\n| Q1 | 4.3 | 2.4 |\n| Q2 | 2.5 | 4.4 |\n| Q3 | 3.5 | 3.5 |",
        ],
    ),
    (
        // The sheet, then the chart — exactly two. `office_oxide` also builds its own synthetic
        // "Charts" section for xlsx; a third chunk here means the pass that drops it stopped
        // recognising its shape and the chart's numbers are in the index twice.
        "sample_chart.xlsx",
        &[
            "| glossa sample | Series 1 |\n| --- | --- |\n| Q1 | 4.3 |\n| Q2 | 2.5 |\n| Q3 | 3.5 |",
            "Chart: Sales by quarter (barChart)\n\n|  | Series 1 | Series 2 |\n| --- | --- | --- |\n| Q1 | 4.3 | 2.4 |\n| Q2 | 2.5 | 4.4 |\n| Q3 | 3.5 | 3.5 |",
        ],
    ),
    (
        "sample.xlsb",
        &["| glossa sample xlsb |  |\n| --- | --- |\n| Sensor | Limit |\n| inlet pressure | 42 |\n| outlet pressure | 97 |"],
    ),
    ("sample_legacy.doc", &["# glossa sample legacy doc"]),
    (
        "sample_legacy.xls",
        &["| glossa sample legacy xls |\n| --- |"],
    ),
    (
        // POST-CHANGE, not an original baseline: Task 5 of the chart-ordinals plan (`e93f211`)
        // gave an rtf legacy chart its own ordinal position, so the chart chunk's place in this
        // sequence is already the new interleaved behaviour, not what the extractor emitted
        // before that commit.
        // Caveat on what this row can witness: **this fixture cannot demonstrate chart
        // placement at all, and a reader must not take these two chunks as evidence either
        // way.** It carries no heading and sits under the 4000-character threshold, so
        // `chunk_ir` coalesces the whole body into ONE chunk — which makes "after the chunk
        // containing the anchor" and "appended at the end" the same sequence. The test that
        // actually demonstrates the ordinal placement is
        // `src/extract/rtf.rs::a_legacy_chart_lands_after_the_paragraph_that_contains_it`,
        // which injects a heading into this fixture first so the body splits.
        "sample_chart_legacy.rtf",
        &[
            "glossa sample\n\nQuarterly fruit sales, charted below.\n\nEnd of sample.",
            "Chart: Chart 1\n\n|  | Apples | Pears | Plums |\n| --- | --- | --- | --- |\n| Qtr1 | 20.4 | 30.6 | 45.9 |\n| Qtr2 | 27.4 | 38.6 | 46.9 |\n| Qtr3 | 90 | 34.6 | 45 |\n| Qtr4 | 20.4 | 31.6 | 43.9 |",
        ],
    ),
    // The odf rows below are PRE-CHANGE: odf charts still append at the end of the document.
    // Tasks 6 and 7 of the chart-ordinals plan move them to their anchors, which will red these
    // rows on purpose — update them there deliberately.
    (
        // The chartless control for odt: `sample_chart.odt`'s first three chunks must stay
        // byte-identical to these.
        "sample.odt",
        &[
            "glossa sample document.",
            "# Alpha Section\n\nAlpha body paragraph one.\n\nAlpha body paragraph two.",
            "# Beta Section\n\nBeta body paragraph.\n\n| h1 | h2 |\n| --- | --- |\n| wide | wide |",
        ],
    ),
    (
        // Caveat on what this row can witness: **it cannot witness chart placement, and the
        // trailing chart chunk here is not evidence that placement is broken.** This fixture's
        // `content.xml` contains no `<draw:object>` at all — the `Object 1/content.xml` chart
        // part is in the zip but nothing in the body references it — so the chart has no anchor
        // to recover and takes the honest end-of-document fallback. The tests that demonstrate
        // the odt placement are in `src/extract/odf.rs`
        // (`an_odt_chart_lands_after_the_paragraph_that_contains_it` and
        // `a_page_anchored_odt_chart_is_appended_not_placed`), on bodies that carry one.
        "sample_chart.odt",
        &[
            "glossa sample document.",
            "# Alpha Section\n\nAlpha body paragraph one.\n\nAlpha body paragraph two.",
            "# Beta Section\n\nBeta body paragraph.\n\n| h1 | h2 |\n| --- | --- |\n| wide | wide |",
            "Chart: Sales by quarter (bar)\n\n|  | Series 1 |\n| --- | --- |\n| Q1 | 4.3 |\n| Q2 | 2.5 |",
        ],
    ),
    (
        // Two sheets, one chunk each: `Sheet1` then `Data`.
        "sample.ods",
        &[
            "| glossa sample |  |\n| --- | --- |\n| a | b |\n| 1 | 2 |",
            "| x | dup | dup | dup | y |\n| --- | --- | --- | --- | --- |",
        ],
    ),
    (
        "sample_chart.ods",
        &[
            "| glossa sample |  |\n| --- | --- |\n| a | b |\n| 1 | 2 |",
            "| x | dup | dup | dup | y |\n| --- | --- | --- | --- | --- |",
            "Chart: Sales by quarter (bar)\n\n|  | Series 1 | Series 2 |\n| --- | --- | --- |\n| Q1 | 4.3 | 2.4 |\n| Q2 | 2.5 | 4.4 |\n| Q3 | 3.5 | 3.5 |",
        ],
    ),
    (
        // Same lineage as `sample_chart.ods`, and `Sheet1` differs by one empty row: the
        // `<draw:frame>` that anchors the chart sits in its own `<table:table-row>`, so the sheet
        // renders a blank `|  |  |` line the unanchored fixture does not have. That is today's
        // behaviour, pinned here rather than endorsed.
        "sample_chart_anchored.ods",
        &[
            "| glossa sample |  |\n| --- | --- |\n|  |  |\n| a | b |\n| 1 | 2 |",
            "| x | dup | dup | dup | y |\n| --- | --- | --- | --- | --- |",
            "Chart: Sales by quarter (bar)\n\n|  | Series 1 | Series 2 |\n| --- | --- | --- |\n| Q1 | 4.3 | 2.4 |\n| Q2 | 2.5 | 4.4 |\n| Q3 | 3.5 | 3.5 |",
        ],
    ),
    (
        // The ref-only chart: its values came from resolving the cell ranges against `Sheet1`,
        // which is why the chart table repeats the sheet's own numbers.
        //
        // Same caveat as the rtf row: this fixture has a single sheet, so its body is ONE chunk
        // and "after the chunk containing the anchor" coincides with "appended at the end".
        // When the ods anchoring lands, this row will not witness it — a multi-sheet fixture
        // (`sample.ods`'s two sheets, or `sample_chart_anchored.ods`) is what can.
        "sample_chart_ref.ods",
        &[
            "| Cat | Series 1 |\n| --- | --- |\n| Q1 | 4.3 |\n| Q2 | 2.5 |\n| Q3 | 3.5 |",
            "Chart: Sales by quarter (bar)\n\n|  | Series 1 |\n| --- | --- |\n| Q1 | 4.3 |\n| Q2 | 2.5 |\n| Q3 | 3.5 |",
        ],
    ),
    (
        "sample.odp",
        &[
            "# glossa sample slide\n\nFirst slide body text.",
            "# Second Heading\n\nSecond slide body text.",
        ],
    ),
    (
        // Same caveat as `sample_chart.odt`, for the same reason: no `<draw:object>` anywhere in
        // this fixture's `content.xml`, so its chart is an orphan and appends. The test that
        // demonstrates the odp placement is
        // `src/extract/odf.rs::an_odp_chart_lands_in_its_own_slide_after_the_containing_block`.
        "sample_chart.odp",
        &[
            "# glossa sample slide\n\nFirst slide body text.",
            "# Second Heading\n\nSecond slide body text.",
            "Chart: Sales by quarter (bar)\n\n|  | Series 1 |\n| --- | --- |\n| Q1 | 4.3 |\n| Q2 | 2.5 |",
        ],
    ),
];

#[test]
fn document_fixtures_chunk_exactly_as_pinned() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut moved: Vec<&str> = Vec::new();
    // Built for every fixture, not just the ones that moved, so one failing run hands over a
    // complete replacement for EXPECTED rather than a fragment that has to be merged by hand.
    let mut actual_source = String::new();

    for (name, expected) in EXPECTED {
        let path = fixtures.join(name);
        let mut chunks = Vec::new();
        extract_file(&path, &mut |c| chunks.push(c))
            .unwrap_or_else(|e| panic!("{name}: extraction failed: {e}"));
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();

        actual_source.push_str(&format!("    ({name:?}, &[\n"));
        for t in &texts {
            actual_source.push_str(&format!("        {t:?},\n"));
        }
        actual_source.push_str("    ]),\n");

        if texts.as_slice() != *expected {
            moved.push(name);
        }
    }

    assert!(
        moved.is_empty(),
        "the extracted output moved for: {moved:?}\n\n\
         Read the change before accepting it (see this file's module comment), then replace \
         EXPECTED with:\n\n{actual_source}"
    );
}
