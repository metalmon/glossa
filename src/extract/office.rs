use crate::extract::office_chunk::chunk_ir;
use crate::extract::office_table::expand_merged_tables;
use crate::extract::Extractor;
use crate::model::Chunk;
use anyhow::anyhow;
use office_oxide::{Document, DocumentFormat};
use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::path::Path;

pub struct OfficeExtractor;

fn format_for(ext: &str) -> Option<DocumentFormat> {
    match ext {
        "docx" => Some(DocumentFormat::Docx),
        "doc" => Some(DocumentFormat::Doc),
        // BIFF12 is a different encoding of the same model, so it rides the Xlsx format — which is
        // how office_oxide's own `from_extension` maps it.
        "xlsx" | "xlsb" => Some(DocumentFormat::Xlsx),
        "xls" => Some(DocumentFormat::Xls),
        "pptx" => Some(DocumentFormat::Pptx),
        "ppt" => Some(DocumentFormat::Ppt),
        _ => None,
    }
}

/// Does this IR's last section match the synthetic chart section `office_oxide` appends for
/// xlsx (`convert_xlsx.rs`, around line 380) whenever its own `doc.chart_text` is non-empty?
/// Detected structurally, not by title alone — a user may legitimately name a real worksheet
/// "Charts" — so this requires ALL of: it is the last section, its title is exactly "Charts",
/// and its elements are the strict alternation `Heading(level 3, "Chart 1")`, `Paragraph`,
/// `Heading(level 3, "Chart 2")`, `Paragraph`, ... starting at 1, with nothing else mixed in.
/// Returns the chart count when it matches (used only to describe the section in a warning —
/// the actual drop/keep decision compares chart-PART NAMES, see `xlsx_chart_part_names` and its
/// caller below, because an equal count does not mean the same charts were found), `None` when
/// the section does not match.
fn synthetic_xlsx_chart_section_chart_count(ir: &office_oxide::ir::DocumentIR) -> Option<usize> {
    use office_oxide::ir::{Element, InlineContent};

    let section = ir.sections.last()?;
    if section.title.as_deref() != Some("Charts") {
        return None;
    }
    let elements = &section.elements;
    if elements.is_empty() || elements.len() % 2 != 0 {
        return None;
    }
    let mut count = 0usize;
    for pair in elements.chunks(2) {
        count += 1;
        let Element::Heading(h) = &pair[0] else {
            return None;
        };
        if h.level != 3 {
            return None;
        }
        let heading_text: String = h
            .content
            .iter()
            .filter_map(|c| match c {
                InlineContent::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect();
        if heading_text != format!("Chart {count}") {
            return None;
        }
        if !matches!(&pair[1], Element::Paragraph(_)) {
            return None;
        }
    }
    Some(count)
}

/// The xlsx chart-part names `office_oxide`'s own xlsx reader would see (`xlsx/mod.rs`:
/// `n.starts_with("xl/charts/chart") && n.ends_with(".xml")`, scanning every zip entry). We
/// cannot read which of those it actually kept — its accept rule lives inside
/// `extract_chart_text` and is internal — so this stands in for "every chart part present in
/// the package" and is deliberately broader than our own `ooxml_chart` glob (which also
/// excludes `colors`/`style` parts): the synthetic section is only dropped when our own pass
/// covers this whole set, never when we merely match its size.
fn xlsx_chart_part_names(bytes: &[u8]) -> Vec<String> {
    let Ok(zip) = zip::ZipArchive::new(Cursor::new(bytes)) else {
        return Vec::new();
    };
    zip.file_names()
        .filter(|n| n.starts_with("xl/charts/chart") && n.ends_with(".xml"))
        .map(|s| s.to_string())
        .collect()
}

/// The OOXML relationship type of a chart part.
const CHART_REL_TYPE: &str =
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart";

/// REJECTION bound on how much of `word/_rels/document.xml.rels` is read. A zip entry's
/// decompressed size is a claim the file makes, so the read is capped instead of believed. Real
/// relationship parts are kilobytes; 4 MiB is far past any document a reader cites.
const MAX_DOCUMENT_RELS_BYTES: u64 = 4 * 1024 * 1024;

/// Every chart relationship declared by `word/_rels/document.xml.rels`, mapped from its id to the
/// package part it resolves to — the same resolution `office_oxide`'s own reader does when it goes
/// looking for a `<c:chart r:id="…"/>`'s text. Empty (so the whole docx pass is skipped) for a
/// document that references no chart at all, which is the common case and the cheap one: it costs
/// one small zip entry and no second parse of the document.
fn docx_chart_parts_by_rel_id(bytes: &[u8]) -> HashMap<String, String> {
    use quick_xml::events::Event;

    use std::io::Read as _;

    let mut out = HashMap::new();
    let Ok(mut zip) = zip::ZipArchive::new(Cursor::new(bytes)) else {
        return out;
    };
    let Ok(rels) = zip.by_name("word/_rels/document.xml.rels") else {
        return out;
    };
    let mut xml = String::new();
    // The decompressed size of a zip entry is whatever the file says it is, so this read is
    // bounded rather than trusted. A relationship part past this size is not one a reader cites;
    // the truncation leaves the map short or empty, and a short map means "keep both copies".
    if rels
        .take(MAX_DOCUMENT_RELS_BYTES)
        .read_to_string(&mut xml)
        .is_err()
    {
        return out;
    }

    let mut reader = quick_xml::Reader::from_str(&xml);
    reader.config_mut().check_end_names = false;
    loop {
        let e = match reader.read_event() {
            Ok(Event::Empty(e)) | Ok(Event::Start(e)) => e,
            Ok(Event::Eof) | Err(_) => break,
            _ => continue,
        };
        if e.local_name().as_ref() != b"Relationship" {
            continue;
        }
        let mut id = None;
        let mut ty = None;
        let mut target = None;
        // `TargetMode="External"` names a part outside the package, which neither pass can read;
        // such a relationship is left out and so simply never matches a drawing.
        let mut external = false;
        for attr in e.attributes().flatten() {
            let value = String::from_utf8_lossy(&attr.value).into_owned();
            let name = attr.key.local_name();
            let name = name.as_ref();
            if name == b"Id" {
                id = Some(value);
            } else if name == b"Type" {
                ty = Some(value);
            } else if name == b"Target" {
                target = Some(value);
            } else if name == b"TargetMode" {
                external = value == "External";
            }
        }
        if external || ty.as_deref() != Some(CHART_REL_TYPE) {
            continue;
        }
        if let (Some(id), Some(part)) =
            (id, target.as_deref().and_then(resolve_document_rel_target))
        {
            out.insert(id, part);
        }
        // The caller rejects a document with more chart drawings than this anyway, so a table
        // bigger than the bound can only be a file inflating the work. Stopping here leaves the
        // table short, and a drawing whose relationship is not in it keeps both copies.
        if out.len() >= MAX_DEDUPED_CHART_DRAWINGS_PER_FILE {
            break;
        }
    }
    out
}

/// A `word/_rels/document.xml.rels` target as a package part name. Targets are relative to the
/// part that owns the relationship (`word/document.xml`), except a leading `/` which makes them
/// package-absolute. Anything that walks upwards returns `None` rather than a guess — an
/// unresolved chart keeps both copies, which is the safe side.
fn resolve_document_rel_target(target: &str) -> Option<String> {
    if let Some(abs) = target.strip_prefix('/') {
        return Some(abs.to_string());
    }
    let t = target.strip_prefix("./").unwrap_or(target);
    if t.is_empty() || t.contains("..") {
        return None;
    }
    Some(format!("word/{t}"))
}

/// REJECTION bound on how many chart-bearing drawings one docx may have de-duplicated, and on how
/// many chart relationships its `document.xml.rels` may declare. Both counts come straight out of
/// the file — a small zip can declare a great many of either — and each one costs a walk of the
/// body flow looking for its text, so without a per-file bound a crafted document buys quadratic
/// work for a few hundred bytes. Past the bound the de-duplication is skipped entirely and both
/// copies of every chart's text are kept, which is the same answer this pass already gives
/// whenever identification fails. A document with more charts than this is not one a reader
/// cites; the same number as `rtf_chart`'s `MAX_CHARTS_PER_FILE`, for the same reason.
const MAX_DEDUPED_CHART_DRAWINGS_PER_FILE: usize = 64;

/// Every drawing in the body that references a chart part whose text `office_oxide` recovered, in
/// document order, as `(relationship id, the lines it recovered)`, plus the total number found —
/// which keeps counting after `out` stops growing at `MAX_DEDUPED_CHART_DRAWINGS_PER_FILE`, so the
/// caller can say how far over the bound a document is without holding a line of it. Those exact
/// lines are what `convert_docx.rs`'s `collect_paragraph_chart_text` pushes into the IR as plain
/// paragraphs, one per line, at the chart's slot in the body flow.
///
/// `dgm_text` — a SmartArt diagram's text, which travels the very same path — is deliberately NOT
/// collected: nothing of ours re-emits it, so it must stay in the body flow or it is lost.
fn collect_docx_chart_drawings<'a>(
    blocks: &'a [office_oxide::docx::BlockElement],
    out: &mut Vec<(&'a str, std::sync::Arc<[String]>)>,
    total: &mut usize,
) {
    use office_oxide::docx::{BlockElement, ParagraphContent, Run, RunContent};

    for block in blocks {
        match block {
            BlockElement::Paragraph(p) => {
                for pc in &p.content {
                    let runs: &[Run] = match pc {
                        ParagraphContent::Run(r) => std::slice::from_ref(r),
                        ParagraphContent::Hyperlink(hl) => &hl.runs,
                    };
                    for run in runs {
                        for rc in &run.content {
                            match rc {
                                RunContent::Drawing(d) => {
                                    if let Some(rid) = d.chart_rel_id.as_deref() {
                                        if !d.chart_text.is_empty() {
                                            *total += 1;
                                            if out.len() < MAX_DEDUPED_CHART_DRAWINGS_PER_FILE {
                                                out.push((
                                                    rid,
                                                    std::sync::Arc::clone(&d.chart_text),
                                                ));
                                            }
                                        }
                                    }
                                }
                                // A text box's body is ordinary block content and can hold a
                                // chart of its own; `convert_docx` hoists from there too.
                                RunContent::TextBox(inner) => {
                                    collect_docx_chart_drawings(inner, out, total)
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            BlockElement::Table(t) => {
                for row in &t.rows {
                    for cell in &row.cells {
                        collect_docx_chart_drawings(&cell.content, out, total);
                    }
                }
            }
        }
    }
}

/// The text of an IR element that is exactly the shape `collect_paragraph_chart_text` emits: a
/// paragraph holding one plain text span and nothing else. Anything richer — a styled run, a
/// heading, two spans — is a body paragraph and is never a candidate.
fn hoisted_line_text(e: &office_oxide::ir::Element) -> Option<&str> {
    use office_oxide::ir::{Element, InlineContent};

    let Element::Paragraph(p) = e else {
        return None;
    };
    match p.content.as_slice() {
        [InlineContent::Text(t)] => Some(t.text.as_str()),
        _ => None,
    }
}

/// Where in the IR a chart's hoisted lines sit: the first contiguous run of plain paragraphs, at
/// or after `from`, whose texts equal `lines` in order. Searching forward from the previous
/// chart's run keeps two charts with identical text from both resolving to the first one.
fn find_hoisted_chart_run(
    ir: &office_oxide::ir::DocumentIR,
    from: (usize, usize),
    lines: &[String],
) -> Option<(usize, usize)> {
    if lines.is_empty() {
        return None;
    }
    for (s, section) in ir.sections.iter().enumerate().skip(from.0) {
        let els = &section.elements;
        if els.len() < lines.len() {
            continue;
        }
        let start = if s == from.0 { from.1 } else { 0 };
        for (at, window) in els.windows(lines.len()).enumerate().skip(start) {
            if window
                .iter()
                .zip(lines)
                .all(|(e, line)| hoisted_line_text(e) == Some(line.as_str()))
            {
                return Some((s, at));
            }
        }
    }
    None
}

/// Drop the copy of a chart's text that `office_oxide` 0.1.17 hoists into the docx body flow —
/// but only for a chart part our own `ooxml_chart` pass covered, so the words it removes are
/// words that reach the index anyway, in a better shape (`Chart: <title>` plus a markdown table).
///
/// This is the xlsx de-duplication (`synthetic_xlsx_chart_section_chart_count` above) in the
/// harder form the docx reader presents: there is no synthetic section and no heading pattern to
/// recognise, because the lines are unmarked paragraphs sitting in the middle of the body. The
/// identification is therefore made with `office_oxide`'s own public docx parser rather than by a
/// text heuristic: `DrawingInfo::chart_rel_id` says which relationship a drawing points at and
/// `DrawingInfo::chart_text` gives the exact lines the converter will emit for it, so the IR
/// paragraphs are matched by that value, in document order, and nothing else is ever a candidate.
///
/// When identification fails — the relationship does not resolve, our pass did not cover the
/// part (a chart with no series, say), or the lines are not where they should be — BOTH copies
/// are kept and a warning names the file and the chart. Losing a chart to tidy the index is the
/// worse failure, and it is doubly so for SmartArt, whose text rides the same converter path with
/// no chart chunk of ours to replace it; `collect_docx_chart_drawings` never collects it.
fn drop_hoisted_docx_chart_text(
    path: &Path,
    bytes: &[u8],
    ir: &mut office_oxide::ir::DocumentIR,
    covered_chart_parts: &[String],
) {
    let parts_by_rel_id = docx_chart_parts_by_rel_id(bytes);
    if parts_by_rel_id.is_empty() {
        return;
    }
    let doc = match office_oxide::docx::DocxDocument::from_reader(Cursor::new(bytes.to_vec())) {
        Ok(doc) => doc,
        Err(e) => {
            tracing::warn!(
                "docx {}: could not re-read the package to tell chart text from body text ({e}) — keeping both copies of every chart's text",
                path.display()
            );
            return;
        }
    };
    let mut drawings = Vec::new();
    let mut total_drawings = 0usize;
    collect_docx_chart_drawings(&doc.body.elements, &mut drawings, &mut total_drawings);
    if drawings.is_empty() {
        return;
    }
    if total_drawings > MAX_DEDUPED_CHART_DRAWINGS_PER_FILE {
        tracing::warn!(
            "docx {}: {total_drawings} chart-bearing drawings is past the per-file bound of {MAX_DEDUPED_CHART_DRAWINGS_PER_FILE} — skipping the chart de-duplication entirely and keeping both copies of every chart's text; {} drawing(s) went unprocessed",
            path.display(),
            total_drawings - MAX_DEDUPED_CHART_DRAWINGS_PER_FILE
        );
        return;
    }
    let covered: HashSet<&str> = covered_chart_parts.iter().map(String::as_str).collect();
    // Resolve every chart to its range FIRST, then rebuild each section in one pass. Removing as
    // we went would shift every later element once per chart, and the cursor only ever moves
    // forward, so one chart's run is never claimed by another and the whole scan stays linear in
    // the number of elements however many charts there are.
    let mut to_remove: Vec<(usize, usize, usize)> = Vec::new();
    let mut cursor = (0usize, 0usize);
    for (i, (rel_id, lines)) in drawings.iter().enumerate() {
        let part = parts_by_rel_id.get(*rel_id);
        let Some((section, at)) = find_hoisted_chart_run(ir, cursor, lines) else {
            // Our model of where the converter puts chart text no longer matches this document,
            // and the charts after this one are found by walking on from here — so stop rather
            // than rescan the same tail for each of them, and leave the rest of the flow alone.
            tracing::warn!(
                "docx {}: chart relationship {rel_id} named {} line(s) of text that are not in the document flow where the converter puts them — keeping both copies from here on, {} chart(s) left untouched",
                path.display(),
                lines.len(),
                drawings.len() - i
            );
            break;
        };
        cursor = (section, at + lines.len());
        match part {
            Some(part) if covered.contains(part.as_str()) => {
                to_remove.push((section, at, lines.len()));
            }
            _ => {
                tracing::warn!(
                    "docx {}: chart relationship {rel_id} resolves to {}, which our chart pass did not cover — keeping office_oxide's copy of its text rather than risk losing the chart",
                    path.display(),
                    part.map(String::as_str).unwrap_or("no readable chart part")
                );
            }
        }
    }
    let mut by_section: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (section, at, len) in to_remove {
        by_section.entry(section).or_default().push((at, len));
    }
    for (section, ranges) in by_section {
        let elements = &mut ir.sections[section].elements;
        let mut drop_it = vec![false; elements.len()];
        for (at, len) in ranges {
            for slot in drop_it.iter_mut().skip(at).take(len) {
                *slot = true;
            }
        }
        let mut i = 0;
        elements.retain(|_| {
            let keep = !drop_it[i];
            i += 1;
            keep
        });
    }
}

impl Extractor for OfficeExtractor {
    fn file_types(&self) -> &'static [&'static str] {
        &["docx", "doc", "xlsx", "xlsb", "xls", "pptx", "ppt"]
    }

    fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        let fmt = format_for(&ext).ok_or_else(|| anyhow!("unsupported office extension: {ext}"))?;
        let doc = Document::from_reader(Cursor::new(bytes.to_vec()), fmt)
            .map_err(|e| anyhow!("office parse failed for {}: {e}", path.display()))?;
        let mut ir = doc.to_ir();
        expand_merged_tables(&mut ir);
        let (chart_chunks, covered_chart_parts) =
            crate::extract::ooxml_chart::extract_charts_with_coverage(path, bytes, &ext);
        if ext == "xlsx" {
            if let Some(synthetic_chart_count) = synthetic_xlsx_chart_section_chart_count(&ir) {
                let present_parts = xlsx_chart_part_names(bytes);
                let covered: HashSet<&str> =
                    covered_chart_parts.iter().map(String::as_str).collect();
                let missing: Vec<&String> = present_parts
                    .iter()
                    .filter(|p| !covered.contains(p.as_str()))
                    .collect();
                // Vacuously true on an empty `present_parts` is the wrong default here: that
                // would mean our own zip scan found nothing even though `office_oxide` clearly
                // built a chart section from *something*, so treat "found zero present parts"
                // as "not covered" rather than as trivially satisfied.
                if !present_parts.is_empty() && missing.is_empty() {
                    ir.sections.pop();
                } else {
                    tracing::warn!(
                        "xlsx {}: office_oxide's synthetic chart section names {synthetic_chart_count} chart(s), but our chart pass covered only {}/{} chart part(s) in the package (missing: {missing:?}) — keeping both sections rather than risk losing chart text",
                        path.display(),
                        present_parts.len() - missing.len(),
                        present_parts.len()
                    );
                }
            }
        }
        if ext == "docx" {
            drop_hoisted_docx_chart_text(path, bytes, &mut ir, &covered_chart_parts);
        }
        let mut chunks = chunk_ir(path, &ir, &ext);
        chunks.extend(chart_chunks);
        if chunks.is_empty() {
            // "Parsed fine, produced nothing" is the failure that hides. An error is listed in the
            // run summary and retried on the next pass; `Ok(vec![])` enters the manifest as
            // indexed-and-empty WITH its signature recorded, so the document is never looked at
            // again — and nothing in the output distinguishes it from a document that genuinely
            // has no text. `rtf.rs` and `odf.rs` have refused this since they were written; this
            // reader is the one that did not, and it is the one that has silently under-extracted
            // before (the `w:pBdr` scanner desync that ran the reader to EOF).
            //
            // The cost, taken knowingly: an error is a permanent skip, and `index_file` buffers
            // before it deletes, so a document EDITED down to no text keeps its last indexed text
            // under its new signature until a `--force`. That is the better of the two wrongs —
            // zero text from an office reader is far more often the reader's failure than the
            // author's intent — and it is what `odf.rs` and `rtf.rs` have always done. The real
            // repair is a verdict the caller can tell apart ("parsed, genuinely empty" vs "could
            // not read"), which belongs with the extractor registry, not in this branch.
            return Err(anyhow!(
                "office produced no chunks for {} — the file parsed but yielded no text",
                path.display()
            ));
        }
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_office_reader_declines_rtf_bytes_rather_than_half_reading_them() {
        // `extract_file`'s mislabeled-extension rescue rests on this: a `.doc` that is really RTF
        // must either fail or produce nothing, so the rescue can see it declined. The dangerous
        // third outcome is a reader that "succeeds" with garbage chunks — then the file is indexed
        // as mangled text and nothing downstream can tell. That is the docx under-extraction class
        // we have already been bitten by, so it is pinned here rather than assumed.
        let r = OfficeExtractor.extract(
            Path::new("mislabelled.doc"),
            br"{\rtf1\ansi\pard Real text lives here.\par}",
        );
        match r {
            Err(_) => {}
            Ok(cs) => assert!(
                cs.is_empty(),
                "the office reader claimed to read RTF, so the rescue will never fire: {:?}",
                cs.iter().map(|c| &c.text).collect::<Vec<_>>()
            ),
        }
    }

    /// A docx that opens cleanly but carries no text must be a verdict, not `Ok(vec![])`.
    /// Built by repacking the real fixture with an empty body, so everything else about the
    /// package stays a genuine docx and the only thing under test is the empty result.
    #[test]
    fn a_document_that_opens_but_yields_no_text_is_an_error_not_an_empty_success() {
        use std::io::{Read as _, Write as _};
        const EMPTY_BODY: &[u8] = br#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:sectPr/></w:body></w:document>"#;
        let mut zr = zip::ZipArchive::new(Cursor::new(
            include_bytes!("../../tests/fixtures/sample.docx").to_vec(),
        ))
        .unwrap();
        let names: Vec<String> = zr
            .file_names()
            .filter(|n| !n.ends_with('/'))
            .map(|s| s.to_string())
            .collect();
        let mut buf = Vec::new();
        {
            let mut zw = zip::ZipWriter::new(Cursor::new(&mut buf));
            let opts = zip::write::SimpleFileOptions::default();
            for name in names {
                let mut bytes = Vec::new();
                zr.by_name(&name).unwrap().read_to_end(&mut bytes).unwrap();
                zw.start_file(&name, opts).unwrap();
                zw.write_all(if name == "word/document.xml" {
                    EMPTY_BODY
                } else {
                    bytes.as_slice()
                })
                .unwrap();
            }
            zw.finish().unwrap();
        }
        let e = OfficeExtractor
            .extract(Path::new("blank.docx"), &buf)
            .unwrap_err()
            .to_string();
        assert!(e.contains("no chunks"), "{e}");
    }

    #[test]
    fn extracts_text_from_docx_fixture() {
        let bytes = include_bytes!("../../tests/fixtures/sample.docx");
        let chunks = OfficeExtractor
            .extract(Path::new("sample.docx"), bytes)
            .unwrap();
        let joined = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            joined.contains("glossa sample"),
            "expected fixture marker text, got: {joined}"
        );
        assert!(chunks.iter().all(|c| c.file_type == "docx"));
    }

    #[test]
    fn extracts_text_from_xlsb_fixture() {
        // `.xlsb` is a BIFF12 binary workbook — a different encoding of the same model as `.xlsx`,
        // which `office_oxide` reads since 0.1.12. Before that it was not in `file_types()`, so a
        // `.xlsb` in a corpus produced NOTHING, silently: no extractor claimed it and no error said
        // so. The fixture is synthetic and English-only (see tests/fixtures/README if adding more).
        let bytes = include_bytes!("../../tests/fixtures/sample.xlsb");
        let chunks = OfficeExtractor
            .extract(Path::new("sample.xlsb"), bytes)
            .unwrap();
        let joined = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            joined.contains("glossa sample"),
            "expected fixture marker text, got: {joined}"
        );
        assert!(
            joined.contains("inlet pressure") && joined.contains("97"),
            "expected the sheet's cells, got: {joined}"
        );
        assert!(chunks.iter().all(|c| c.file_type == "xlsb"));
    }

    #[test]
    fn claims_xlsb_as_a_supported_type() {
        // `file_types()` is the gate: a type absent here is never routed to this extractor, so the
        // mapping below and this list have to agree.
        assert!(OfficeExtractor.file_types().contains(&"xlsb"));
        assert_eq!(format_for("xlsb"), Some(DocumentFormat::Xlsx));
    }

    #[test]
    fn unsupported_extension_errors() {
        let err = OfficeExtractor
            .extract(Path::new("x.rtf"), b"junk")
            .unwrap_err();
        assert!(err.to_string().contains("unsupported office extension"));
    }

    #[test]
    fn extracts_table_as_markdown() {
        let bytes = include_bytes!("../../tests/fixtures/sample_table.docx");
        let chunks = OfficeExtractor
            .extract(Path::new("sample_table.docx"), bytes)
            .unwrap();
        let joined = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            joined.contains('|') && joined.contains("---"),
            "expected a GFM pipe table from the docx table, got:\n{joined}"
        );
    }

    /// End-to-end through OfficeExtractor: office_oxide must open the
    /// synthetic fixture (an injected, undeclared chart part must be inert
    /// to the text path) AND the wiring in `extract` must append the chart
    /// chunk after the doc's text chunks.
    #[test]
    fn office_extractor_appends_chart_chunk() {
        let bytes = include_bytes!("../../tests/fixtures/sample_chart.docx");
        let chunks = OfficeExtractor
            .extract(Path::new("sample_chart.docx"), bytes)
            .unwrap();
        assert!(
            chunks.iter().any(|c| !c.text.starts_with("Chart:")),
            "expected at least one non-chart (text) chunk, got: {chunks:?}"
        );
        assert!(
            chunks.iter().any(|c| c.text.starts_with("Chart:")),
            "expected a chart chunk appended, got: {chunks:?}"
        );
    }

    /// Repack `sample_chart_anchored.docx` with one part swapped. Used to build the
    /// identification-failure cases without committing a second near-identical fixture.
    fn anchored_fixture_with(part: &str, replacement: &[u8]) -> Vec<u8> {
        use std::io::{Read as _, Write as _};
        let mut zr = zip::ZipArchive::new(Cursor::new(
            include_bytes!("../../tests/fixtures/sample_chart_anchored.docx").to_vec(),
        ))
        .unwrap();
        let names: Vec<String> = zr
            .file_names()
            .filter(|n| !n.ends_with('/'))
            .map(|s| s.to_string())
            .collect();
        let mut buf = Vec::new();
        {
            let mut zw = zip::ZipWriter::new(Cursor::new(&mut buf));
            let opts = zip::write::SimpleFileOptions::default();
            for name in names {
                let mut bytes = Vec::new();
                zr.by_name(&name).unwrap().read_to_end(&mut bytes).unwrap();
                zw.start_file(&name, opts).unwrap();
                zw.write_all(if name == part {
                    replacement
                } else {
                    bytes.as_slice()
                })
                .unwrap();
            }
            zw.finish().unwrap();
        }
        buf
    }

    fn joined(chunks: &[Chunk]) -> String {
        chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// `office_oxide` 0.1.17 hoists a native chart's text into the docx body flow as plain
    /// paragraphs at the chart's own slot, while our `ooxml_chart` pass builds a well-formed
    /// `Chart: …` chunk from the very same part — so without the de-duplication the chart's
    /// words reach the index twice, in two shapes. `sample_chart_anchored.docx` is the fixture
    /// that actually exercises this: unlike `sample_chart.docx`, its chart part is referenced
    /// through a real `w:drawing` → `a:graphicData` → `c:chart r:id`.
    #[test]
    fn docx_chart_text_is_not_duplicated() {
        let bytes = include_bytes!("../../tests/fixtures/sample_chart_anchored.docx");
        let chunks = OfficeExtractor
            .extract(Path::new("sample_chart_anchored.docx"), bytes)
            .unwrap();
        let text = joined(&chunks);

        // "4.4" is a cached value of the chart's second series and appears nowhere in the body —
        // the only ways it reaches the index are the two chart paths, so it is the clean signal.
        assert_eq!(
            text.matches("4.4").count(),
            1,
            "chart value must be indexed exactly once, got:\n{text}"
        );
        // And the copy that survived is ours, not the flat hoisted one.
        assert!(
            chunks
                .iter()
                .any(|c| c.text.starts_with("Chart: Sales by quarter")),
            "expected the structured chart chunk to survive, got: {chunks:?}"
        );
        assert!(
            !text.contains("Series 2: 2.4, 4.4, 3.5"),
            "office_oxide's flat hoisted line is still in the index:\n{text}"
        );
    }

    /// The guard that matters more than the de-duplication: the paragraphs around the chart are
    /// ordinary body text and must come through untouched.
    #[test]
    fn docx_body_text_survives_the_chart_dedup() {
        let bytes = include_bytes!("../../tests/fixtures/sample_chart_anchored.docx");
        let chunks = OfficeExtractor
            .extract(Path::new("sample_chart_anchored.docx"), bytes)
            .unwrap();
        let text = joined(&chunks);
        for expected in [
            "glossa sample document used as a neutral test fixture.",
            "Body text that follows the chart.",
            "Body text that follows the diagram.",
        ] {
            assert!(
                text.contains(expected),
                "the chart de-duplication ate body text {expected:?}, got:\n{text}"
            );
        }
    }

    /// A document that references no chart never reaches the matching code at all, so its chunks
    /// are whatever they were before. `sample_chart.docx` is the sharper of the two cases: it
    /// HAS a chart part, so our chart pass runs, but nothing in the body points at it.
    #[test]
    fn a_docx_with_no_chart_is_untouched_by_the_dedup() {
        let plain = OfficeExtractor
            .extract(
                Path::new("sample.docx"),
                include_bytes!("../../tests/fixtures/sample.docx"),
            )
            .unwrap();
        assert_eq!(
            plain.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(),
            vec!["Sample\n\nglossa sample document used as a neutral test fixture."],
        );

        let unreferenced = OfficeExtractor
            .extract(
                Path::new("sample_chart.docx"),
                include_bytes!("../../tests/fixtures/sample_chart.docx"),
            )
            .unwrap();
        assert_eq!(
            unreferenced
                .iter()
                .map(|c| c.text.as_str())
                .collect::<Vec<_>>(),
            vec![
                "Sample\n\nglossa sample document used as a neutral test fixture.",
                "Chart: Sales by quarter (barChart)\n\n|  | Series 1 | Series 2 |\n| --- | --- | --- |\n| Q1 | 4.3 | 2.4 |\n| Q2 | 2.5 | 4.4 |\n| Q3 | 3.5 | 3.5 |",
            ],
        );
    }

    /// The worse failure mode, pinned. 0.1.17 routes a SmartArt diagram's text through the same
    /// converter path as a chart's — plain paragraphs hoisted into the body flow — but nothing
    /// of ours re-emits it, so removing it would simply lose it. The anchored fixture carries a
    /// real `dgm:relIds` diagram next to its chart for exactly this test.
    #[test]
    fn smartart_text_is_not_eaten_by_the_chart_dedup() {
        let bytes = include_bytes!("../../tests/fixtures/sample_chart_anchored.docx");
        let chunks = OfficeExtractor
            .extract(Path::new("sample_chart_anchored.docx"), bytes)
            .unwrap();
        let text = joined(&chunks);
        for node in ["Diagram node alpha", "Diagram node beta"] {
            assert_eq!(
                text.matches(node).count(),
                1,
                "SmartArt text {node:?} must reach the index exactly once (it has no chart chunk \
                 to replace it), got:\n{text}"
            );
        }
    }

    /// The denial-of-service guard. Both the number of chart relationships a `.docx` declares and
    /// the number of drawings that point at them come straight out of the file, and each drawing
    /// costs a walk of the body flow — so a few hundred bytes of zip could buy quadratic work.
    /// Past `MAX_DEDUPED_CHART_DRAWINGS_PER_FILE` the de-duplication is skipped and both copies
    /// are kept, so the document still extracts, body text and all. Built by repeating the
    /// fixture's own chart drawing rather than by committing a second fixture.
    #[test]
    fn a_docx_declaring_absurdly_many_chart_drawings_still_extracts() {
        let drawing = {
            let mut zr = zip::ZipArchive::new(Cursor::new(
                include_bytes!("../../tests/fixtures/sample_chart_anchored.docx").to_vec(),
            ))
            .unwrap();
            use std::io::Read as _;
            let mut doc = String::new();
            zr.by_name("word/document.xml")
                .unwrap()
                .read_to_string(&mut doc)
                .unwrap();
            let start = doc.find("<w:drawing>").unwrap();
            let end = doc.find("</w:drawing>").unwrap() + "</w:drawing>".len();
            doc[start..end].to_string()
        };
        let mut body = String::from(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:dgm="http://schemas.openxmlformats.org/drawingml/2006/diagram"><w:body><w:p><w:r><w:t>glossa sample document used as a neutral test fixture.</w:t></w:r></w:p>"#,
        );
        for _ in 0..(MAX_DEDUPED_CHART_DRAWINGS_PER_FILE * 8) {
            body.push_str("<w:p><w:r>");
            body.push_str(&drawing);
            body.push_str("</w:r></w:p>");
        }
        body.push_str("<w:sectPr/></w:body></w:document>");

        let buf = anchored_fixture_with("word/document.xml", body.as_bytes());
        let chunks = OfficeExtractor
            .extract(Path::new("many_charts.docx"), &buf)
            .unwrap();
        let text = joined(&chunks);
        assert!(
            text.contains("glossa sample document used as a neutral test fixture."),
            "the bound must skip the de-duplication, not the extraction, got:\n{text}"
        );
    }

    /// The data-loss guard: when our own pass does not cover the chart part a drawing points at
    /// — here a chart with a title and no `<c:ser>` at all, which `ooxml_chart` skips — the
    /// hoisted copy is the ONLY place that chart's text reaches the index, so both are kept and
    /// a warning says so.
    #[test]
    fn docx_keeps_the_hoisted_chart_text_when_our_pass_misses_the_part() {
        const TITLE_ONLY_CHART: &[u8] = br#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<c:chart><c:title><c:tx><c:rich><a:p><a:r><a:t>Sales by quarter</a:t></a:r></a:p></c:rich></c:tx></c:title>
<c:plotArea><c:barChart></c:barChart></c:plotArea></c:chart></c:chartSpace>"#;

        let buf = anchored_fixture_with("word/charts/chart1.xml", TITLE_ONLY_CHART);
        let chunks = OfficeExtractor
            .extract(Path::new("sample_chart_title_only.docx"), &buf)
            .unwrap();
        let text = joined(&chunks);

        assert!(
            !chunks.iter().any(|c| c.text.starts_with("Chart:")),
            "our chart pass should not have covered a title-only chart, got: {chunks:?}"
        );
        assert!(
            text.contains("Title: Sales by quarter"),
            "the hoisted copy is the only place this chart's text could reach the index, so it \
             must survive, got:\n{text}"
        );
        assert!(text.contains("Body text that follows the chart."), "{text}");
    }

    /// For xlsx, `office_oxide` appends its own synthetic "Charts" section (a flat blob per
    /// chart) on top of our well-formed `ooxml_chart::extract_charts` chunk — the same numbers
    /// reaching the index twice. When our pass covers every chart part present in the package
    /// (as it does here — one chart, with series data our pass reads fine), the fix drops the
    /// synthetic section, leaving the chart's numbers indexed exactly once, in our chunk, with
    /// the sheet's own text untouched.
    #[test]
    fn xlsx_chart_text_is_not_duplicated() {
        let bytes = include_bytes!("../../tests/fixtures/sample_chart.xlsx");
        let chunks = OfficeExtractor
            .extract(Path::new("sample_chart.xlsx"), bytes)
            .unwrap();
        let joined = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        // "4.4" is a cached value from the chart's second series and appears nowhere on the
        // sheet itself — the only way it reaches the index is through chart extraction, so it
        // is the clean signal for "did the same chart data land twice".
        assert_eq!(
            joined.matches("4.4").count(),
            1,
            "chart value must be indexed exactly once, got:\n{joined}"
        );

        // Our well-formed chunk is the one that survived.
        assert!(
            chunks
                .iter()
                .any(|c| c.text.starts_with("Chart: Sales by quarter")),
            "expected the structured chart chunk to survive, got: {chunks:?}"
        );

        // The fix did not eat a real section: the sheet's own cell text is still there.
        assert!(
            joined.contains("glossa sample"),
            "expected the sheet's own marker text, got:\n{joined}"
        );
    }

    /// The data-loss guard: when `office_oxide`'s synthetic section names a chart part our own
    /// pass did not cover — here, a chart with only a `<c:title>` and no `<c:ser>` at all, so
    /// our pass finds no series and skips it — the fix must keep BOTH sections rather than drop
    /// the one `office_oxide` built from it, or that chart's title would reach the index
    /// nowhere at all. Built by repacking the validated `sample_chart.xlsx` fixture with its
    /// chart part swapped for a title-only one, so everything else (the sheet, the rels chain)
    /// stays the already-exercised fixture.
    #[test]
    fn xlsx_keeps_both_sections_when_our_pass_misses_a_present_chart_part() {
        use std::io::{Read as _, Write as _};
        const TITLE_ONLY_CHART: &[u8] = br#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<c:chart><c:title><c:tx><c:rich><a:p><a:r><a:t>Sales by quarter</a:t></a:r></a:p></c:rich></c:tx></c:title>
<c:plotArea><c:barChart></c:barChart></c:plotArea></c:chart></c:chartSpace>"#;

        let mut zr = zip::ZipArchive::new(Cursor::new(
            include_bytes!("../../tests/fixtures/sample_chart.xlsx").to_vec(),
        ))
        .unwrap();
        let names: Vec<String> = zr
            .file_names()
            .filter(|n| !n.ends_with('/'))
            .map(|s| s.to_string())
            .collect();
        let mut buf = Vec::new();
        {
            let mut zw = zip::ZipWriter::new(Cursor::new(&mut buf));
            let opts = zip::write::SimpleFileOptions::default();
            for name in names {
                let mut bytes = Vec::new();
                zr.by_name(&name).unwrap().read_to_end(&mut bytes).unwrap();
                zw.start_file(&name, opts).unwrap();
                zw.write_all(if name == "xl/charts/chart1.xml" {
                    TITLE_ONLY_CHART
                } else {
                    bytes.as_slice()
                })
                .unwrap();
            }
            zw.finish().unwrap();
        }

        let chunks = OfficeExtractor
            .extract(Path::new("sample_chart_title_only.xlsx"), &buf)
            .unwrap();
        let joined = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        // Our own pass found no series in this chart, so it produced no "Chart: ..." chunk —
        // the synthetic section is the only place this chart's title could reach the index.
        assert!(
            !chunks.iter().any(|c| c.text.starts_with("Chart:")),
            "our chart pass should not have covered a title-only chart, got: {chunks:?}"
        );
        assert!(
            joined.contains("Title: Sales by quarter"),
            "expected office_oxide's synthetic section to survive (our pass did not cover its \
             only chart part), got:\n{joined}"
        );

        // The fix did not touch the sheet's own text.
        assert!(joined.contains("glossa sample"), "got:\n{joined}");
    }

    /// The synthetic-section predicate is structural, not title-based: a real worksheet that a
    /// user happens to name "Charts" — holding an ordinary table, not the heading/paragraph
    /// alternation `office_oxide` emits — must not be mistaken for the synthetic section.
    #[test]
    fn a_real_sheet_named_charts_is_not_mistaken_for_the_synthetic_section() {
        use office_oxide::ir::{
            DocumentIR, Element, Paragraph, Section, Table, TableCell, TableRow,
        };

        let ir = DocumentIR {
            sections: vec![Section {
                title: Some("Charts".to_string()),
                elements: vec![Element::Table(Table {
                    rows: vec![TableRow {
                        cells: vec![TableCell {
                            content: vec![Element::Paragraph(Paragraph::default())],
                            ..Default::default()
                        }],
                        ..Default::default()
                    }],
                    ..Default::default()
                })],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(synthetic_xlsx_chart_section_chart_count(&ir), None);
    }

    /// The positive case: office_oxide's actual shape (two charts) is recognized and counted.
    #[test]
    fn the_synthetic_charts_section_shape_is_recognized_and_counted() {
        use office_oxide::ir::{
            DocumentIR, Element, Heading, InlineContent, Paragraph, Section, TextSpan,
        };

        let chart_heading = |n: usize| {
            Element::Heading(Heading {
                level: 3,
                content: vec![InlineContent::Text(TextSpan {
                    text: format!("Chart {n}"),
                    ..Default::default()
                })],
                ..Default::default()
            })
        };
        let ir = DocumentIR {
            sections: vec![Section {
                title: Some("Charts".to_string()),
                elements: vec![
                    chart_heading(1),
                    Element::Paragraph(Paragraph::default()),
                    chart_heading(2),
                    Element::Paragraph(Paragraph::default()),
                ],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(synthetic_xlsx_chart_section_chart_count(&ir), Some(2));
    }
}
