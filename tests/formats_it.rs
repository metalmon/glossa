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
    // a chart chunk from sample_chart.ods (Object 1/content.xml, local table)
    let ods_chart_chunks = chunks
        .iter()
        .filter(|c| c.file_type == "ods" && c.text.starts_with("Chart:"))
        .count();
    assert_eq!(ods_chart_chunks, 1, "expected exactly one ods chart chunk");
}
