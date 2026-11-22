use crate::extract::chart_splice::{containing_chunk, splice_charts_at_anchors, ChartAnchor};
use crate::extract::office_chunk::chunk_ir_with_spans;
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
fn docx_chart_parts_by_rel_id(path: &Path, bytes: &[u8]) -> HashMap<String, String> {
    use quick_xml::events::Event;

    use std::io::Read as _;

    let mut out = HashMap::new();
    let Ok(mut zip) = zip::ZipArchive::new(Cursor::new(bytes)) else {
        tracing::warn!(
            "docx {}: the package would not open as a zip for the relationship read — keeping both copies of every chart's text",
            path.display()
        );
        return out;
    };
    // A document part with no relationships at all is normal (it is what `sample.docx` looks
    // like) and is not worth a warning: it simply means there is no chart to de-duplicate.
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
        tracing::warn!(
            "docx {}: word/_rels/document.xml.rels could not be read to its end (unreadable, or past the {MAX_DOCUMENT_RELS_BYTES}-byte bound) — keeping both copies of every chart's text",
            path.display()
        );
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
        // Top-level paragraphs ONLY. A `BlockElement::Table` is skipped, and `RunContent::TextBox`
        // below is not descended into either — DELIBERATELY, do not "fix" this back. A chart
        // inside a table cell or a text box is real, but `convert_docx` hoists its lines into
        // THAT container's own content (`TableCell.content`, `Element::TextBox.content`), never
        // into the section's top-level `elements` that the matcher scans. Collecting such a
        // drawing could therefore only ever produce a miss, and a miss costs a warning and keeps
        // both copies — so the honest thing is not to claim it as a candidate at all. Nested
        // charts simply keep both copies, silently and by construction.
        let BlockElement::Paragraph(p) = block else {
            continue;
        };
        for pc in &p.content {
            let runs: &[Run] = match pc {
                ParagraphContent::Run(r) => std::slice::from_ref(r),
                ParagraphContent::Hyperlink(hl) => &hl.runs,
            };
            for run in runs {
                for rc in &run.content {
                    let RunContent::Drawing(d) = rc else {
                        continue;
                    };
                    if let Some(rid) = d.chart_rel_id.as_deref() {
                        if !d.chart_text.is_empty() {
                            *total += 1;
                            if out.len() < MAX_DEDUPED_CHART_DRAWINGS_PER_FILE {
                                out.push((rid, std::sync::Arc::clone(&d.chart_text)));
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The text of an IR element shaped the way `collect_paragraph_chart_text` emits one: a paragraph
/// holding a single plain text span. This is a weak filter, and deliberately named as one — an
/// ordinary one-run body paragraph has exactly this shape too, so the ONLY thing separating a
/// hoisted chart line from body prose is the text itself. That is why the matcher below never
/// trusts a single text match: see `hoisted_chart_run_occurrences`.
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

/// REJECTION bound on how large a body this pass will match a chart against. It also bounds the
/// comparison budget below, which is derived from the element count — without it an enormous body
/// would buy an enormous budget. A document with this many elements is not one anyone is citing a
/// chart's position in.
const MAX_DEDUP_SECTION_ELEMENTS: usize = 100_000;

/// How many element comparisons the whole de-duplication may spend, per element in the document.
/// Honest input needs far less than one pass: a chart's lines are found where the converter put
/// them, and a miss fails on the first element of the window. Bounding each factor separately —
/// the line count, the drawing count, a cheap pre-check — does NOT bound their product, and a
/// crafted document picks whichever factor is left; one budget spent across every group does,
/// whatever shape the amplification takes.
const CHART_MATCH_COMPARISONS_PER_ELEMENT: usize = 8;

/// Every place in the IR where this chart's hoisted lines appear as a contiguous run of plain
/// paragraphs, in document order and never overlapping, stopping once more than `want` have been
/// found (the caller only needs to know whether the count is exactly `want`). `None` means the
/// shared `budget` ran out mid-scan and the answer is unknown — the caller then keeps both copies
/// for everything, which is what every other failure here does.
///
/// Occurrences rather than "the first match": the predicate is text equality and nothing else
/// (see `hoisted_line_text`), so a body paragraph that happens to repeat a chart's lines AHEAD of
/// the chart would otherwise be the paragraph removed — its words would still reach the index
/// through our `Chart:` chunk, but its position and heading context would not. Counting instead
/// of taking the first match makes that case visible: when the number of runs does not equal the
/// number of drawings carrying these exact lines, the caller cannot say which run belongs to
/// which drawing, and keeps both copies. When the counts DO agree, the k-th run is the k-th such
/// drawing's, because the converter emits them in the same document order this walk reads them.
fn hoisted_chart_run_occurrences(
    ir: &office_oxide::ir::DocumentIR,
    lines: &[String],
    want: usize,
    budget: &mut usize,
) -> Option<Vec<(usize, usize)>> {
    let mut found = Vec::new();
    if lines.is_empty() {
        return Some(found);
    }
    for (s, section) in ir.sections.iter().enumerate() {
        let els = &section.elements;
        if els.len() < lines.len() {
            continue;
        }
        let mut at = 0usize;
        while at + lines.len() <= els.len() {
            let mut matched = 0usize;
            while matched < lines.len() {
                // Charged per comparison actually performed, so no shape of input — however
                // many lines, however deep the partial matches run — can spend more than the
                // one budget the caller set from the document's own size.
                if *budget == 0 {
                    return None;
                }
                *budget -= 1;
                if hoisted_line_text(&els[at + matched]) != Some(lines[matched].as_str()) {
                    break;
                }
                matched += 1;
            }
            if matched == lines.len() {
                found.push((s, at));
                if found.len() > want {
                    return Some(found);
                }
                at += lines.len();
            } else {
                at += 1;
            }
        }
    }
    Some(found)
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
/// `DrawingInfo::chart_text` gives the exact lines the converter will emit for it. Matching those
/// lines in the IR is still text equality, so it is only trusted when it is unambiguous: the runs
/// of each line sequence are COUNTED, and a sequence is resolved only when the document flow
/// holds exactly as many runs as there are drawings carrying it (see
/// `hoisted_chart_run_occurrences`). Nested charts — inside a table cell or a text box — are not
/// candidates at all; `collect_docx_chart_drawings` says why.
///
/// When identification fails — the relationship does not resolve, our pass did not cover the
/// part (a chart with no series, say), or the run count is ambiguous — BOTH copies
/// are kept and a warning names the file and the chart. Losing a chart to tidy the index is the
/// worse failure, and it is doubly so for SmartArt, whose text rides the same converter path with
/// no chart chunk of ours to replace it; `collect_docx_chart_drawings` never collects it.
///
/// Returns WHICH of those endings happened, and fills `anchors` with the position each removed
/// chart's chunk should take — see the comment on the anchor pass below. The outcome is what makes
/// each bound testable on its own: without it a test can only observe "both copies were kept",
/// which every guard here produces, so a test meant to pin the comparison budget would go on
/// passing if the budget were deleted and some other guard took over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DedupOutcome {
    /// `document.xml.rels` declares no chart relationship we could resolve.
    NoChartRelationships,
    /// The package would not re-open through the docx parser.
    Unreadable,
    /// No drawing in the body flow references a chart part with text.
    NoChartDrawings,
    /// Past `MAX_DEDUPED_CHART_DRAWINGS_PER_FILE`.
    TooManyDrawings,
    /// Past `MAX_DEDUP_SECTION_ELEMENTS`.
    TooManyElements,
    /// The shared comparison budget ran out mid-scan.
    BudgetExhausted,
    /// At least one group's run count did not equal its drawing count.
    Ambiguous,
    /// Two charts' runs overlapped, so the file's de-duplication was abandoned.
    OverlappingRuns,
    /// Every group resolved; whatever was covered was removed.
    Resolved,
}

fn drop_hoisted_docx_chart_text(
    path: &Path,
    bytes: &[u8],
    ir: &mut office_oxide::ir::DocumentIR,
    covered_chart_parts: &[String],
    anchors: &mut HashMap<String, ChartAnchor>,
) -> DedupOutcome {
    let parts_by_rel_id = docx_chart_parts_by_rel_id(path, bytes);
    if parts_by_rel_id.is_empty() {
        return DedupOutcome::NoChartRelationships;
    }
    // The price of identifying charts without a library change, paid only by documents that
    // actually declare one: `bytes.to_vec()` is a second full copy of the package, this is a
    // second parse of the document part, and `docx_chart_parts_by_rel_id` already opened a
    // `ZipArchive` over the same bytes — a third. Worth revisiting if `office_oxide` ever marks
    // the hoisted paragraphs in the IR, which would make all of it unnecessary.
    let doc = match office_oxide::docx::DocxDocument::from_reader(Cursor::new(bytes.to_vec())) {
        Ok(doc) => doc,
        Err(e) => {
            tracing::warn!(
                "docx {}: could not re-read the package to tell chart text from body text ({e}) — keeping both copies of every chart's text",
                path.display()
            );
            return DedupOutcome::Unreadable;
        }
    };
    let mut drawings = Vec::new();
    let mut total_drawings = 0usize;
    collect_docx_chart_drawings(&doc.body.elements, &mut drawings, &mut total_drawings);
    if drawings.is_empty() {
        return DedupOutcome::NoChartDrawings;
    }
    if total_drawings > MAX_DEDUPED_CHART_DRAWINGS_PER_FILE {
        tracing::warn!(
            "docx {}: {total_drawings} chart-bearing drawings is past the per-file bound of {MAX_DEDUPED_CHART_DRAWINGS_PER_FILE} — skipping the chart de-duplication entirely and keeping both copies of every chart's text; {} drawing(s) went unprocessed",
            path.display(),
            total_drawings - MAX_DEDUPED_CHART_DRAWINGS_PER_FILE
        );
        return DedupOutcome::TooManyDrawings;
    }
    let covered: HashSet<&str> = covered_chart_parts.iter().map(String::as_str).collect();

    // Drawings that recovered the SAME lines are resolved together: the body flow cannot say
    // which run belongs to which of them, only how many runs there are, so the question is
    // answerable for the group or for none of it. Linear grouping — there are at most
    // `MAX_DEDUPED_CHART_DRAWINGS_PER_FILE` of them.
    let mut groups: Vec<(&[String], Vec<&str>)> = Vec::new();
    for (rel_id, lines) in &drawings {
        match groups.iter_mut().find(|(l, _)| *l == &lines[..]) {
            Some((_, ids)) => ids.push(*rel_id),
            None => groups.push((&lines[..], vec![*rel_id])),
        }
    }

    let elements: usize = ir.sections.iter().map(|s| s.elements.len()).sum();
    if elements > MAX_DEDUP_SECTION_ELEMENTS {
        tracing::warn!(
            "docx {}: {elements} body elements is past the bound of {MAX_DEDUP_SECTION_ELEMENTS} — skipping the chart de-duplication and keeping both copies of every chart's text",
            path.display()
        );
        return DedupOutcome::TooManyElements;
    }
    // ONE budget for the whole call, spent across every group. Bounding the line count, the
    // drawing count, or the cost of any single window each bound a factor; only this bounds
    // their product, which is what a crafted document actually inflates.
    let total_budget = elements * CHART_MATCH_COMPARISONS_PER_ELEMENT + 1024;
    let mut budget = total_budget;

    // Resolve every chart to its range FIRST, then rebuild each section in one pass. Removing as
    // we go would shift every later element once per chart. A group that cannot be resolved only
    // costs itself: the others are found independently, so one unmatchable chart never disables
    // the de-duplication for the rest of the document.
    let mut to_remove: Vec<(usize, usize, usize, String)> = Vec::new();
    let mut any_ambiguous = false;
    for (i, (lines, rel_ids)) in groups.iter().enumerate() {
        let Some(occurrences) =
            hoisted_chart_run_occurrences(ir, lines, rel_ids.len(), &mut budget)
        else {
            tracing::warn!(
                "docx {}: matching chart text against the body flow spent its whole comparison budget ({total_budget} for {elements} element(s)) — keeping both copies of every chart's text; {} group(s) of drawings went unprocessed",
                path.display(),
                groups.len() - i
            );
            return DedupOutcome::BudgetExhausted;
        };
        if occurrences.len() != rel_ids.len() {
            any_ambiguous = true;
            tracing::warn!(
                "docx {}: {} drawing(s) ({}) recovered {} line(s) of chart text, but the document flow holds {} run(s) of exactly those lines — cannot say which run is the hoisted copy, so keeping both copies for all of them",
                path.display(),
                rel_ids.len(),
                rel_ids.join(", "),
                lines.len(),
                occurrences.len()
            );
            continue;
        }
        for (rel_id, (section, at)) in rel_ids.iter().zip(&occurrences) {
            let part = parts_by_rel_id.get(*rel_id);
            match part {
                Some(part) if covered.contains(part.as_str()) => {
                    to_remove.push((*section, *at, lines.len(), part.clone()));
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
    }

    // Runs found for two different line sequences can in principle overlap (one sequence ending
    // where another begins is enough). Removing an overlapping pair would take out more than
    // either chart's text, so the whole file falls back to keeping both copies instead.
    to_remove.sort_unstable_by_key(|&(section, at, ..)| (section, at));
    if to_remove
        .windows(2)
        .any(|w| w[0].0 == w[1].0 && w[0].1 + w[0].2 > w[1].1)
    {
        tracing::warn!(
            "docx {}: two charts' hoisted text resolved to overlapping runs in the document flow — keeping both copies of every chart's text rather than remove the wrong paragraphs",
            path.display()
        );
        return DedupOutcome::OverlappingRuns;
    }

    // Where each removed chart's own chunk belongs, for the caller to splice it at: the element
    // immediately BEFORE the run of hoisted lines. `collect_paragraph_chart_text` runs last of
    // the collectors for a paragraph, so that element came out of the SAME source paragraph as
    // the chart — but it is not always the paragraph itself. In `convert_docx.rs` (0.1.17) the
    // paragraph's own slot may be an `Element::Paragraph` (empty, for a chart-only paragraph), a
    // `ThematicBreak` (an empty paragraph with a bottom border), or an `Element::Heading`; and
    // three collectors run in between, so a paragraph that also carries a float, an inline image
    // or a text box puts THAT hoisted element last before the run instead.
    //
    // Any of them is a correct anchor, because they are siblings in one element flow and
    // `chunk_ir` puts siblings in one chunk — with one case worth naming: a heading OPENS a chunk
    // (`office_chunk.rs`), so a chart anchored in a heading paragraph follows the whole chunk that
    // heading starts rather than sitting at its head. That is still "after the chunk containing
    // the paragraph", which is all this feature promises; the spec's granularity caveat says the
    // same about short flat documents.
    //
    // Indices are into the IR the body chunks are built from, i.e. AFTER the removals below, so
    // each run's start is shifted back by everything already removed from its section. The runs
    // are sorted and non-overlapping, so that sum never exceeds the run's own start.
    //
    // Only charts identified HERE get an anchor; everything else — an unresolved relationship, a
    // part our pass did not cover, an ambiguous run count, any of the bounds above — reaches the
    // caller with none and appends at the end, as it did before interleaving existed. Two drawings
    // naming one part also yield no anchor: one chunk cannot occupy two positions, and choosing
    // one of them is the plausible-but-wrong placement this design refuses. So does a run with
    // nothing in front of it, whose containing paragraph is not in the IR to point at.
    let mut candidates: HashMap<String, Option<ChartAnchor>> = HashMap::new();
    let mut run_section = usize::MAX;
    let mut removed_before = 0usize;
    for (section, at, len, part) in &to_remove {
        let (section, at, len) = (*section, *at, *len);
        if section != run_section {
            run_section = section;
            removed_before = 0;
        }
        let anchor = (at - removed_before).checked_sub(1).map(|el| (section, el));
        candidates
            .entry(part.clone())
            .and_modify(|a| *a = None)
            .or_insert(anchor);
        removed_before += len;
    }
    anchors.extend(
        candidates
            .into_iter()
            .filter_map(|(part, anchor)| anchor.map(|a| (part, a))),
    );

    let mut by_section: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (section, at, len, _) in to_remove {
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
    if any_ambiguous {
        DedupOutcome::Ambiguous
    } else {
        DedupOutcome::Resolved
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
        // Where each chart part's own chunk belongs in the body flow. Filled for docx only: it is
        // the one format whose charts we can locate (see `drop_hoisted_docx_chart_text`), and an
        // empty map means every chart appends, which is what xlsx and pptx keep doing.
        let mut chart_anchors: HashMap<String, ChartAnchor> = HashMap::new();
        if ext == "docx" {
            // The outcome is for the tests, which need to tell the guards apart; the warnings
            // inside already say everything a reader of the log needs.
            drop_hoisted_docx_chart_text(
                path,
                bytes,
                &mut ir,
                &covered_chart_parts,
                &mut chart_anchors,
            );
        }
        let (chunks, spans) = chunk_ir_with_spans(path, &ir, &ext);
        // The chart chunks themselves are built exactly as before — same title, same text, same
        // `Location`. Only where they land is new, and only for a chart whose paragraphs the
        // de-duplication positively identified.
        //
        // Entry i of `chart_chunks` and entry i of `covered_chart_parts` describe the same chart.
        // What makes that true is the shape of `extract_charts_with_coverage`: both vectors are
        // built in ONE iteration of ONE loop over one `parts` vector, and both of that loop's
        // `continue`s sit ahead of either push, so neither vector can gain an entry the other
        // skipped. **That invariant is load-bearing here** — anyone splitting those pushes apart,
        // or adding a path that pushes one without the other, silently moves charts to other
        // charts' positions, and a chart at the wrong position reads as a precise citation and is
        // false. The length check below cannot catch that in general (it is unreachable today,
        // precisely because the invariant holds); it is kept only as a cheap backstop, and pairing
        // nothing is the honest answer when it ever trips.
        let parts: &[String] = if chart_chunks.len() == covered_chart_parts.len() {
            &covered_chart_parts
        } else {
            &[]
        };
        let mut anchored: Vec<(ChartAnchor, Chunk)> = Vec::new();
        let mut unanchored: Vec<Chunk> = Vec::new();
        for (i, chart) in chart_chunks.into_iter().enumerate() {
            match parts.get(i).and_then(|part| chart_anchors.get(part)) {
                Some(&anchor) => anchored.push((anchor, chart)),
                None => unanchored.push(chart),
            }
        }
        // An anchor can be recovered and still resolve to no chunk: `chunk_ir`'s mapping has gaps
        // — an empty paragraph past the 4000-character threshold flushes and enters no buffer, and
        // a run of elements that renders to nothing emits no chunk at all. A chart whose paragraph
        // lands in one appends, which is the honest answer, but on a long document it happens with
        // nothing in the log to say the position was lost.
        //
        // ONE line per file, carrying the count — never one per chart. Both the chart count and
        // the chunk count come out of the document, so a line each would let a crafted file write
        // the log; `odf.rs`'s Z-order summary is the same shape for the same reason.
        let unplaceable = anchored
            .iter()
            .filter(|(anchor, _)| containing_chunk(&spans, *anchor).is_none())
            .count();
        if unplaceable > 0 {
            tracing::warn!(
                "docx {}: {unplaceable} of {} chart(s) with a recovered anchor have no containing chunk (their paragraph fell on a chunk boundary, or its run produced no chunk) — those append at the end of the document instead of taking their own position",
                path.display(),
                anchored.len()
            );
        }
        let mut chunks = splice_charts_at_anchors(chunks, &spans, anchored);
        chunks.extend(unanchored);
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
        repack_with(
            include_bytes!("../../tests/fixtures/sample_chart_anchored.docx"),
            part,
            replacement,
        )
    }

    /// Rewrite one part of a zip package, leaving every other entry byte-identical.
    fn repack_with(package: &[u8], part: &str, replacement: &[u8]) -> Vec<u8> {
        repack_parts(package, &[(part, replacement)])
    }

    /// Rewrite — or add — several parts of a zip package at once, leaving every other entry
    /// byte-identical. Adding is what the two-chart test needs: a second chart part and a second
    /// relationship pointing at it, over the committed one-chart fixture.
    fn repack_parts(package: &[u8], parts: &[(&str, &[u8])]) -> Vec<u8> {
        use std::io::{Read as _, Write as _};
        let mut zr = zip::ZipArchive::new(Cursor::new(package.to_vec())).unwrap();
        let names: Vec<String> = zr
            .file_names()
            .filter(|n| !n.ends_with('/'))
            .map(|s| s.to_string())
            .collect();
        let mut buf = Vec::new();
        {
            let mut zw = zip::ZipWriter::new(Cursor::new(&mut buf));
            let opts = zip::write::SimpleFileOptions::default();
            for name in &names {
                let mut bytes = Vec::new();
                zr.by_name(name).unwrap().read_to_end(&mut bytes).unwrap();
                zw.start_file(name, opts).unwrap();
                let replacement = parts
                    .iter()
                    .find(|(p, _)| *p == name.as_str())
                    .map(|(_, b)| *b);
                zw.write_all(replacement.unwrap_or(bytes.as_slice()))
                    .unwrap();
            }
            for (part, bytes) in parts {
                if !names.iter().any(|n| n.as_str() == *part) {
                    zw.start_file(*part, opts).unwrap();
                    zw.write_all(bytes).unwrap();
                }
            }
            zw.finish().unwrap();
        }
        buf
    }

    /// The anchored fixture's own `<w:drawing>` for the chart (the first one in its body), and
    /// the `<w:document>` opening tag that declares every namespace it uses. Together they let a
    /// test rebuild `word/document.xml` around the real chart reference instead of hand-copying
    /// the drawing XML into each test.
    fn anchored_chart_drawing() -> (String, String) {
        use std::io::Read as _;
        let mut zr = zip::ZipArchive::new(Cursor::new(
            include_bytes!("../../tests/fixtures/sample_chart_anchored.docx").to_vec(),
        ))
        .unwrap();
        let mut doc = String::new();
        zr.by_name("word/document.xml")
            .unwrap()
            .read_to_string(&mut doc)
            .unwrap();
        let open = doc[doc.find("<w:document").unwrap()..doc.find("<w:body>").unwrap()].to_string();
        let start = doc.find("<w:drawing>").unwrap();
        let end = doc.find("</w:drawing>").unwrap() + "</w:drawing>".len();
        (open, doc[start..end].to_string())
    }

    /// Which ending the de-duplication took for this package, by running exactly what `extract`
    /// runs. Asserting on this rather than on "both copies were kept" is what separates the
    /// guards: every one of them keeps both copies, so only the outcome says which one acted.
    fn dedup_outcome_for(bytes: &[u8]) -> DedupOutcome {
        let path = Path::new("outcome.docx");
        let doc = Document::from_reader(Cursor::new(bytes.to_vec()), DocumentFormat::Docx).unwrap();
        let mut ir = doc.to_ir();
        expand_merged_tables(&mut ir);
        let (_, covered) =
            crate::extract::ooxml_chart::extract_charts_with_coverage(path, bytes, "docx");
        drop_hoisted_docx_chart_text(path, bytes, &mut ir, &covered, &mut HashMap::new())
    }

    /// A chart part whose recovered text is exactly one `Series: <name>` line per name, in order:
    /// no `<c:title>` and no values, which is the shape `core::chart::chart_text_lines` turns
    /// into `Series: …` lines and nothing else. Deterministic, so a test can write the very same
    /// strings into the body.
    fn series_only_chart(names: &[&str]) -> String {
        let mut chart = String::from(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><c:chart><c:plotArea><c:barChart>"#,
        );
        for name in names {
            chart.push_str(&format!("<c:ser><c:tx><c:v>{name}</c:v></c:tx></c:ser>"));
        }
        chart.push_str("</c:barChart></c:plotArea></c:chart></c:chartSpace>");
        chart
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

    /// A level-1 heading paragraph, which is what forces `chunk_ir` to start a new chunk. The
    /// outline level is given directly rather than through a `Heading1` style, because
    /// `office_oxide` resolves a style id against `word/styles.xml` and these test packages have
    /// none — a bare `w:pStyle` would silently produce an ordinary paragraph and a one-chunk
    /// document, which is exactly the test that cannot fail.
    fn heading(text: &str) -> String {
        format!(
            "<w:p><w:pPr><w:pStyle w:val=\"Heading1\"/><w:outlineLvl w:val=\"0\"/></w:pPr>\
             <w:r><w:t>{text}</w:t></w:r></w:p>"
        )
    }

    /// The index of the one chunk whose text contains `needle`.
    fn chunk_holding(chunks: &[Chunk], needle: &str) -> usize {
        let hits: Vec<usize> = chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.text.contains(needle))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            hits.len(),
            1,
            "expected exactly one chunk containing {needle:?}, got {hits:?} in {:?}",
            chunks.iter().map(|c| &c.text).collect::<Vec<_>>()
        );
        hits[0]
    }

    /// A chart chunk takes its own ordinal position, after the chunk holding the paragraph the
    /// chart is anchored in, instead of being appended after all the body text.
    ///
    /// **Why this test rebuilds the fixture's body.** Chunk boundaries come only from a heading or
    /// an empty paragraph past the 4000-character threshold, so `sample_chart_anchored.docx` as
    /// committed is ONE chunk — and in a one-chunk document "after the containing chunk" and "at
    /// the end" are the same position, which makes such a test green whether this feature works or
    /// not. The heading below is what gives the document a chunk AFTER the chart, and the two
    /// assertions pin both bounds: the chart follows the body chunk, and the tail chunk follows the
    /// chart. Revert the splice and the chart lands last, which fails the first assertion.
    #[test]
    fn a_docx_chart_lands_after_the_paragraph_that_contains_it() {
        let (open, drawing) = anchored_chart_drawing();
        let body = format!(
            "{open}<w:body>\
             <w:p><w:r><w:t>glossa sample document used as a neutral test fixture.</w:t></w:r></w:p>\
             <w:p><w:r>{drawing}</w:r></w:p>\
             <w:p><w:r><w:t>Body text that follows the chart.</w:t></w:r></w:p>\
             {}\
             <w:p><w:r><w:t>Tail text after the heading.</w:t></w:r></w:p>\
             <w:sectPr/></w:body></w:document>",
            heading("Later section")
        );

        let buf = anchored_fixture_with("word/document.xml", body.as_bytes());
        let chunks = OfficeExtractor
            .extract(Path::new("anchored.docx"), &buf)
            .unwrap();
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(
            chunks.len(),
            3,
            "expected body, chart, tail — got {texts:?}"
        );

        let body_at = chunk_holding(&chunks, "Body text that follows the chart.");
        let chart_at = chunk_holding(&chunks, "Chart: Sales by quarter");
        let tail_at = chunk_holding(&chunks, "Tail text after the heading.");
        assert_eq!(body_at, 0, "{texts:?}");
        assert_eq!(
            chart_at,
            body_at + 1,
            "the chart must follow its own body chunk, not the document: {texts:?}"
        );
        assert_eq!(
            tail_at,
            chart_at + 1,
            "the chart must come BEFORE the later chunk: {texts:?}"
        );
    }

    /// A second chart in a second chunk, which is where a splice that resolves as it goes fails:
    /// inserting the first chart shifts every later chunk index by one, so the second lands a
    /// chunk early. Built over the committed fixture with a second chart part and a second
    /// relationship added, rather than as a second near-identical fixture.
    ///
    /// Same revert bar as the test above, twice over: each chart is pinned both to the chunk it
    /// follows and to the chunk that follows it, so appending either one fails.
    #[test]
    fn two_charts_in_one_docx_both_land_correctly() {
        const COSTS_CHART: &[u8] = br#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<c:chart><c:title><c:tx><c:rich><a:p><a:r><a:t>Costs by quarter</a:t></a:r></a:p></c:rich></c:tx></c:title>
<c:plotArea><c:lineChart>
<c:ser><c:tx><c:v>Series 9</c:v></c:tx>
<c:cat><c:strRef><c:strCache><c:pt idx="0"><c:v>Q1</c:v></c:pt></c:strCache></c:strRef></c:cat>
<c:val><c:numRef><c:numCache><c:pt idx="0"><c:v>7.7</c:v></c:pt></c:numCache></c:numRef></c:val>
</c:ser></c:lineChart></c:plotArea></c:chart></c:chartSpace>"#;
        const RELS: &[u8] = br#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId10" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart" Target="charts/chart1.xml"/><Relationship Id="rId12" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart" Target="charts/chart2.xml"/></Relationships>"#;
        // The second chart part has to be declared too, or the package is not a valid docx.
        const TYPES: &[u8] = br#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/charts/chart1.xml" ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/><Override PartName="/word/charts/chart2.xml" ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/></Types>"#;

        let (open, drawing) = anchored_chart_drawing();
        let second = drawing.replace("rId10", "rId12");
        let body = format!(
            "{open}<w:body>\
             <w:p><w:r><w:t>glossa sample document used as a neutral test fixture.</w:t></w:r></w:p>\
             <w:p><w:r>{drawing}</w:r></w:p>\
             {}\
             <w:p><w:r><w:t>Middle body text.</w:t></w:r></w:p>\
             <w:p><w:r>{second}</w:r></w:p>\
             {}\
             <w:p><w:r><w:t>Tail body text.</w:t></w:r></w:p>\
             <w:sectPr/></w:body></w:document>",
            heading("Second section"),
            heading("Third section")
        );

        let buf = repack_parts(
            include_bytes!("../../tests/fixtures/sample_chart_anchored.docx"),
            &[
                ("word/document.xml", body.as_bytes()),
                ("word/_rels/document.xml.rels", RELS),
                ("word/charts/chart2.xml", COSTS_CHART),
                ("[Content_Types].xml", TYPES),
            ],
        );
        let chunks = OfficeExtractor
            .extract(Path::new("two_charts.docx"), &buf)
            .unwrap();
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(
            chunks.len(),
            5,
            "expected three body chunks and two charts — got {texts:?}"
        );

        let first_body = chunk_holding(&chunks, "glossa sample document");
        let sales = chunk_holding(&chunks, "Chart: Sales by quarter");
        let middle = chunk_holding(&chunks, "Middle body text.");
        let costs = chunk_holding(&chunks, "Costs by quarter");
        let tail = chunk_holding(&chunks, "Tail body text.");
        assert_eq!(sales, first_body + 1, "{texts:?}");
        assert_eq!(middle, sales + 1, "{texts:?}");
        assert_eq!(costs, middle + 1, "{texts:?}");
        assert_eq!(tail, costs + 1, "{texts:?}");
    }

    /// The guarantee for the overwhelming majority of files: a document with no chart anchor gets
    /// the chunk sequence it got before interleaving existed, byte for byte. `sample_chart.docx`
    /// is the sharper of the three — it HAS a chart part, so a chart chunk is built, but nothing
    /// in its body points at one, so there is no anchor and the chart appends exactly as before.
    #[test]
    fn a_chartless_docx_chunks_exactly_as_before() {
        for (name, bytes, expected) in [
            (
                "sample.docx",
                include_bytes!("../../tests/fixtures/sample.docx").as_slice(),
                vec!["Sample\n\nglossa sample document used as a neutral test fixture."],
            ),
            (
                "sample_table.docx",
                include_bytes!("../../tests/fixtures/sample_table.docx").as_slice(),
                vec!["| H1 | H2 |\n| --- | --- |\n| R1 | R2 |"],
            ),
            (
                "sample_chart.docx",
                include_bytes!("../../tests/fixtures/sample_chart.docx").as_slice(),
                vec![
                    "Sample\n\nglossa sample document used as a neutral test fixture.",
                    "Chart: Sales by quarter (barChart)\n\n|  | Series 1 | Series 2 |\n| --- | --- | --- |\n| Q1 | 4.3 | 2.4 |\n| Q2 | 2.5 | 4.4 |\n| Q3 | 3.5 | 3.5 |",
                ],
            ),
        ] {
            let chunks = OfficeExtractor.extract(Path::new(name), bytes).unwrap();
            assert_eq!(
                chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(),
                expected,
                "{name} chunked differently than before interleaving"
            );
        }
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
        let (open, drawing) = anchored_chart_drawing();
        let mut body = format!(
            "{open}<w:body><w:p><w:r><w:t>glossa sample document used as a neutral test \
             fixture.</w:t></w:r></w:p>"
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

    /// The complexity guard, pinned on input built to defeat the cheap ones. Bounding the line
    /// count, the drawing count or the cost of one window each bound a FACTOR; this document
    /// inflates their product instead: a chart whose recovered lines are all identical but the
    /// last, against a body of paragraphs equal to that line, so every window matches deeply and
    /// fails only at the end. The shared comparison budget stops it, and stopping means keeping
    /// both copies — the hoisted lines are still in the flow afterwards.
    #[test]
    fn a_docx_built_to_defeat_the_cheap_match_guards_still_terminates() {
        // 19 identical series names, then one that differs.
        let mut names = vec!["S"; 19];
        names.push("Z");
        let chart = series_only_chart(&names);

        let (open, drawing) = anchored_chart_drawing();
        let mut body = format!(
            "{open}<w:body><w:p><w:r><w:t>glossa sample document used as a neutral test \
             fixture.</w:t></w:r></w:p><w:p><w:r>{drawing}</w:r></w:p>"
        );
        for _ in 0..400 {
            body.push_str("<w:p><w:r><w:t>Series: S</w:t></w:r></w:p>");
        }
        body.push_str("<w:sectPr/></w:body></w:document>");

        let buf = anchored_fixture_with("word/document.xml", body.as_bytes());
        let buf = repack_with(&buf, "word/charts/chart1.xml", chart.as_bytes());

        let chunks = OfficeExtractor
            .extract(Path::new("adversarial_chart.docx"), &buf)
            .unwrap();
        let text = joined(&chunks);

        // THE point of the test: the comparison budget is what stopped this, not some other
        // guard. The body holds only "Series: S" paragraphs, so the full sequence occurs exactly
        // once — delete the budget and this document resolves cleanly rather than turning
        // ambiguous, which is why this assertion fails if the budget is ever removed.
        assert_eq!(dedup_outcome_for(&buf), DedupOutcome::BudgetExhausted);

        // And the consequences: terminated, both copies kept, body text intact.
        assert!(
            text.contains("Series: Z"),
            "the hoisted chart text should have been kept when the budget ran out, got:\n{text}"
        );
        assert!(text.contains("glossa sample"), "{text}");
    }

    /// The other half of the pair. This document is tiny, so the comparison budget cannot be the
    /// thing that stops it — what stops it is the run count: the body carries a verbatim second
    /// copy of the chart's two lines, so two runs exist for one drawing and the pass cannot say
    /// which is the hoisted one. Delete the count rule and "first match wins" would resolve this
    /// document, so the assertion fails; exactly as the budget test fails if the budget goes.
    #[test]
    fn a_docx_whose_body_repeats_a_chart_line_run_is_ambiguous_not_budget_bound() {
        let chart = series_only_chart(&["A", "B"]);
        let (open, drawing) = anchored_chart_drawing();
        let body = format!(
            "{open}<w:body>\
             <w:p><w:r><w:t>glossa sample document used as a neutral test fixture.</w:t></w:r></w:p>\
             <w:p><w:r>{drawing}</w:r></w:p>\
             <w:p><w:r><w:t>Series: A</w:t></w:r></w:p>\
             <w:p><w:r><w:t>Series: B</w:t></w:r></w:p>\
             <w:sectPr/></w:body></w:document>"
        );

        let buf = anchored_fixture_with("word/document.xml", body.as_bytes());
        let buf = repack_with(&buf, "word/charts/chart1.xml", chart.as_bytes());

        assert_eq!(dedup_outcome_for(&buf), DedupOutcome::Ambiguous);

        // Both copies kept, so the duplicated body run is still there — twice.
        let chunks = OfficeExtractor
            .extract(Path::new("ambiguous_chart.docx"), &buf)
            .unwrap();
        let text = joined(&chunks);
        assert_eq!(
            text.matches("Series: B").count(),
            2,
            "neither run may be removed when the pass cannot tell them apart, got:\n{text}"
        );
        assert!(text.contains("glossa sample"), "{text}");
    }

    /// A chart inside a table cell is not a candidate: `convert_docx` hoists its lines into the
    /// cell's own content, not into the section's top-level elements, so it can never be matched
    /// there. What this pins is that the nested chart costs nothing to the ones that CAN be
    /// matched — the body chart after it still de-duplicates. Before the matcher was made
    /// group-wise, the nested chart's miss stopped the pass and silently disabled de-duplication
    /// for every later chart in the document.
    #[test]
    fn a_chart_in_a_table_does_not_disable_the_dedup_for_later_body_charts() {
        let (open, drawing) = anchored_chart_drawing();
        let body = format!(
            "{open}<w:body>\
             <w:p><w:r><w:t>glossa sample document used as a neutral test fixture.</w:t></w:r></w:p>\
             <w:tbl><w:tblGrid><w:gridCol w:w=\"5000\"/></w:tblGrid>\
             <w:tr><w:tc><w:p><w:r>{drawing}</w:r></w:p></w:tc></w:tr></w:tbl>\
             <w:p><w:r>{drawing}</w:r></w:p>\
             <w:p><w:r><w:t>Body text that follows the chart.</w:t></w:r></w:p>\
             <w:sectPr/></w:body></w:document>"
        );

        let buf = anchored_fixture_with("word/document.xml", body.as_bytes());
        let chunks = OfficeExtractor
            .extract(Path::new("chart_in_table.docx"), &buf)
            .unwrap();
        let text = joined(&chunks);

        assert!(
            chunks
                .iter()
                .any(|c| c.text.starts_with("Chart: Sales by quarter")),
            "expected our structured chart chunk, got: {chunks:?}"
        );
        // Two drawings hoist the same flat line: one into the table cell (kept, by design) and
        // one into the body flow (removed). At most one may remain. Before the matcher was made
        // group-wise this was two, because the nested chart's miss stopped the pass.
        assert!(
            text.matches("Series 2: 2.4, 4.4, 3.5").count() <= 1,
            "the body chart's hoisted copy should have been removed, got:\n{text}"
        );
        assert!(
            text.contains("Body text that follows the chart."),
            "the body text did not survive, got:\n{text}"
        );
        assert!(text.contains("glossa sample"), "{text}");
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
