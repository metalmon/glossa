//! A golden pin on what the office extractors actually emit.
//!
//! Every other office test asserts a *property* — "a chart chunk exists", "the text contains the
//! marker", "the stored location equals the emitted one". None of them can witness an extractor
//! upgrade changing the words we index: `office_oxide` can start or stop emitting a paragraph, a
//! table row, a sheet name or a chart's text, and every one of those assertions still holds. This
//! file is the tripwire that does not.
//!
//! **A red here is not a regression by default.** It means the extracted output moved — almost
//! always because `office_oxide` (or our own chunker) was upgraded on purpose. The right response
//! is to read the diff the failure prints, decide whether the new output is an improvement, and
//! then update the expectation below. Do not "fix" the extractor to satisfy this test, and do not
//! update the expectation without reading what changed: the diff is the only review this upgrade
//! path gets. An `INDEX_SCHEMA_VERSION` bump usually belongs with the change that moves these
//! values, because a corpus indexed with the old output has to be rebuilt to get the new one.
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
/// repo and no plain `sample.xlsx`; add their rows here if either is ever committed.
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
];

#[test]
fn office_fixtures_chunk_exactly_as_pinned() {
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
