use glossa::walk::collect_chunks;
use std::path::Path;

#[test]
fn collects_chunks_across_office_and_pdf_fixtures() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let chunks = collect_chunks(&fixtures, None, true).unwrap();
    assert!(
        chunks.iter().any(|c| c.file_type == "docx"),
        "no docx chunks"
    );
    assert!(chunks.iter().any(|c| c.file_type == "pdf"), "no pdf chunks");
    assert!(
        chunks.iter().any(|c| c.text.contains("glossa sample")),
        "marker text not found in any chunk"
    );
}

#[test]
fn collects_ooxml_chart_chunks() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let chunks = collect_chunks(&fixtures, None, true).unwrap();
    // a chart chunk from one of the sample_chart.* fixtures
    assert!(
        chunks
            .iter()
            .any(|c| c.text.starts_with("Chart:") && c.text.contains("---")),
        "no OOXML chart data-table chunk collected"
    );

    // Only that the docx path produces one at all. A count here would be a census of how many
    // chart-bearing docx fixtures happen to be committed, not a statement about the extractor —
    // what each fixture chunks to is pinned by `extract_golden_it.rs` instead.
    assert!(
        chunks
            .iter()
            .any(|c| c.file_type == "docx" && c.text.starts_with("Chart:")),
        "no docx chart chunk collected"
    );
}

#[test]
fn collects_odf_chunks() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let chunks = collect_chunks(&fixtures, None, true).unwrap();
    assert!(chunks.iter().any(|c| c.file_type == "odt"), "no odt chunks");
    assert!(
        chunks
            .iter()
            .any(|c| c.file_type == "ods" && c.text.contains("---")),
        "no ods GFM table"
    );
}

#[test]
fn collects_odf_chart_chunks() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let chunks = collect_chunks(&fixtures, None, true).unwrap();
    // The kinds of ods chart this collects: a local-table chart with nothing in the body
    // referencing it (sample_chart.ods, an orphan), a ref-only chart resolved against the sheet's
    // own cells (sample_chart_ref.ods), and an anchored one whose in-body <draw:object> gives the
    // chart chunk the sheet as its parent instead of a bare label (sample_chart_anchored.ods,
    // hand-built FROM sample_chart.ods with only the anchor added, so its chart chunk's TEXT is
    // byte-identical and the two are distinguishable only by `location`).
    //
    // Deliberately NOT a count. A total over the whole fixture tree is a census of how many
    // chart-bearing fixtures happen to be committed, not a statement about the extractor, and it
    // reds every time anyone adds one (it did, for `sample_chart_multi_sheet.ods`). Each
    // fixture's exact chunk count, text and order is pinned by `extract_golden_it.rs`, which is
    // the real check; this test is about which SHAPES of ods chart reach the collector at all.
    // Its ooxml twin above lost the same assertion for the same reason.
    let ods_chart_chunks: Vec<_> = chunks
        .iter()
        .filter(|c| c.file_type == "ods" && c.text.starts_with("Chart:"))
        .collect();
    assert!(
        !ods_chart_chunks.is_empty(),
        "no ods chart chunks collected"
    );

    let anchored_chart = ods_chart_chunks
        .iter()
        .find(|c| c.location.is_nested())
        .unwrap_or_else(|| panic!("no nested (anchored) chart chunk: {ods_chart_chunks:?}"));
    assert_eq!(
        anchored_chart.location.display(),
        "Sheet1 > Sales by quarter",
        "anchored chart's rendered location mismatch"
    );

    // The ref-only chart's text is distinct from the two local-table charts
    // (no "Series 2", different category/value cells), so it's still
    // findable by text alone.
    let ref_chart = ods_chart_chunks
        .iter()
        .find(|c| {
            c.text.starts_with("Chart: Sales by quarter (bar)") && !c.text.contains("Series 2")
        })
        .unwrap_or_else(|| panic!("no resolved ref-only chart chunk: {ods_chart_chunks:?}"));
    assert!(ref_chart.text.contains("Series 1"), "{}", ref_chart.text);
    assert!(
        ref_chart.text.contains("Q1")
            && ref_chart.text.contains("Q2")
            && ref_chart.text.contains("Q3"),
        "{}",
        ref_chart.text
    );
    assert!(
        ref_chart.text.contains("4.3")
            && ref_chart.text.contains("2.5")
            && ref_chart.text.contains("3.5")
    );

    // The two local-table charts (sample_chart.ods, sample_chart_anchored.ods) share identical
    // text ("Series 2", "4.4"); split them apart by location instead.
    let local_table_charts: Vec<_> = ods_chart_chunks
        .iter()
        .filter(|c| c.text.contains("Series 2"))
        .collect();
    assert_eq!(
        local_table_charts.len(),
        2,
        "expected two local-table chart chunks (orphan + anchored): {local_table_charts:?}"
    );
    for c in &local_table_charts {
        assert!(c.text.contains("4.4"), "{}", c.text);
    }
    assert!(
        local_table_charts.iter().any(|c| !c.location.is_nested()),
        "expected the orphan (sample_chart.ods) local-table chart to stay a Label: {local_table_charts:?}"
    );
    assert!(
        local_table_charts.iter().any(|c| c.location.is_nested()),
        "expected the anchored (sample_chart_anchored.ods) local-table chart to be a Path: {local_table_charts:?}"
    );

    // FIX 4: chart extraction is container-agnostic — sample_chart.odt and
    // sample_chart.odp carry the same embedded local-table chart and must
    // each yield a chart chunk too, not just the .ods fixtures.
    let odt_chart = chunks
        .iter()
        .find(|c| c.file_type == "odt" && c.text.starts_with("Chart:"))
        .unwrap_or_else(|| panic!("expected an odt chart chunk (sample_chart.odt)"));
    assert!(odt_chart.text.contains("Series 1"), "{}", odt_chart.text);
    assert!(
        odt_chart.text.contains('|') && odt_chart.text.contains("---"),
        "{}",
        odt_chart.text
    );

    let odp_chart = chunks
        .iter()
        .find(|c| c.file_type == "odp" && c.text.starts_with("Chart:"))
        .unwrap_or_else(|| panic!("expected an odp chart chunk (sample_chart.odp)"));
    assert!(odp_chart.text.contains("Series 1"), "{}", odp_chart.text);
    assert!(
        odp_chart.text.contains('|') && odp_chart.text.contains("---"),
        "{}",
        odp_chart.text
    );
}
