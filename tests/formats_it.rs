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

    let docx_chart_chunks = chunks
        .iter()
        .filter(|c| c.file_type == "docx" && c.text.starts_with("Chart:"))
        .count();
    // sample_chart.docx has exactly one chart; sample.docx has none
    assert_eq!(docx_chart_chunks, 1, "expected exactly one docx chart chunk");
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
    // one chart chunk from sample_chart.ods (Object 1/content.xml, local
    // table) and one from sample_chart_ref.ods (ref-only chart, resolved
    // against sheet cells).
    let ods_chart_chunks: Vec<_> = chunks
        .iter()
        .filter(|c| c.file_type == "ods" && c.text.starts_with("Chart:"))
        .collect();
    assert_eq!(ods_chart_chunks.len(), 2, "expected two ods chart chunks: {ods_chart_chunks:?}");

    // Both fixtures happen to share the title/kind ("Sales by quarter (bar)")
    // and the Series 1 values, so disambiguate the ref-only chart by the
    // absence of "Series 2" — only sample_chart.ods's local table has it.
    let ref_chart = ods_chart_chunks
        .iter()
        .find(|c| c.text.starts_with("Chart: Sales by quarter (bar)") && !c.text.contains("Series 2"))
        .unwrap_or_else(|| panic!("no resolved ref-only chart chunk: {ods_chart_chunks:?}"));
    assert!(ref_chart.text.contains("Series 1"), "{}", ref_chart.text);
    assert!(
        ref_chart.text.contains("Q1") && ref_chart.text.contains("Q2") && ref_chart.text.contains("Q3"),
        "{}",
        ref_chart.text
    );
    assert!(ref_chart.text.contains("4.3") && ref_chart.text.contains("2.5") && ref_chart.text.contains("3.5"));

    let local_table_chart = ods_chart_chunks
        .iter()
        .find(|c| c.text.contains("Series 2"))
        .unwrap_or_else(|| panic!("no local-table chart chunk: {ods_chart_chunks:?}"));
    assert!(local_table_chart.text.contains("4.4"), "{}", local_table_chart.text);
}
