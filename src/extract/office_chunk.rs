use crate::extract::office_table::table_to_markdown;
use crate::model::Chunk;
use office_oxide::ir::{DocumentIR, Element, InlineContent, List};
use std::path::Path;

pub const CHUNK_CHAR_THRESHOLD: usize = 4000;

/// Where one emitted chunk came from: the index of the section it was built in, and the half-open
/// range of that section's element indices it covers.
///
/// Reported parallel to the chunk vector — entry `i` describes chunk `i`. Element indices restart in
/// every section while chunk indices do not, so the section is part of the answer, not context.
///
/// The mapping is not total the other way round: a run of elements that renders to nothing emits no
/// chunk (`flush_buf` returns early), and an empty paragraph past the threshold is dropped without
/// ever entering a buffer. Those element indices are covered by no span at all. A caller resolving
/// an anchor treats "no span contains it" as a real answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkSpan {
    pub section: usize,
    pub elements: std::ops::Range<usize>,
}

pub fn chunk_ir(path: &Path, ir: &DocumentIR, file_type: &str) -> Vec<Chunk> {
    chunk_ir_with_threshold(path, ir, file_type, CHUNK_CHAR_THRESHOLD)
}

/// `chunk_ir` plus the per-chunk report of the elements each chunk covers.
pub fn chunk_ir_with_spans(
    path: &Path,
    ir: &DocumentIR,
    file_type: &str,
) -> (Vec<Chunk>, Vec<ChunkSpan>) {
    chunk_ir_inner(path, ir, file_type, CHUNK_CHAR_THRESHOLD)
}

pub fn chunk_ir_with_threshold(
    path: &Path,
    ir: &DocumentIR,
    file_type: &str,
    threshold: usize,
) -> Vec<Chunk> {
    chunk_ir_inner(path, ir, file_type, threshold).0
}

fn chunk_ir_inner(
    path: &Path,
    ir: &DocumentIR,
    file_type: &str,
    threshold: usize,
) -> (Vec<Chunk>, Vec<ChunkSpan>) {
    let mut out = Vec::new();
    let mut spans: Vec<ChunkSpan> = Vec::new();

    for (section_index, section) in ir.sections.iter().enumerate() {
        let section_title = section.title.as_deref();
        let mut heading_path: Vec<String> = Vec::new();
        let mut buf: Vec<Element> = Vec::new();
        let mut buf_start: Option<usize> = None;
        let mut glued_note_after_table = false;

        // Flush, and record the range only if a chunk actually came out. `flush_buf` emits at most
        // one chunk and sometimes none, so the growth of `out` is what decides — the bookkeeping
        // never decides anything the walk had not already decided.
        macro_rules! flush_recording {
            ($end:expr) => {{
                let start = buf_start.take();
                let before = out.len();
                flush_buf(
                    path,
                    file_type,
                    &mut buf,
                    &heading_path,
                    section_title,
                    &mut out,
                );
                if out.len() > before {
                    spans.push(ChunkSpan {
                        section: section_index,
                        elements: start.unwrap_or($end)..$end,
                    });
                }
            }};
        }

        for (element_index, el) in section.elements.iter().enumerate() {
            if let Element::Heading(h) = el {
                flush_recording!(element_index);
                glued_note_after_table = false;
                let title = inline_md(&h.content);
                let level = h.level as usize;
                heading_path.truncate(level.saturating_sub(1));
                buf_start.get_or_insert(element_index);
                heading_path.push(title);
                buf.push(el.clone());
                continue;
            }

            let rendered_len = render_elements(&buf).chars().count();
            if is_empty_paragraph(el) {
                if rendered_len >= threshold {
                    flush_recording!(element_index);
                    glued_note_after_table = false;
                    // This element enters no buffer, so no span covers it: a gap, by design.
                    continue;
                }
                buf_start.get_or_insert(element_index);
                buf.push(el.clone());
                glued_note_after_table = false;
                continue;
            }

            // Soft-split at threshold: glue a caption paragraph onto the following table,
            // and at most one note paragraph after a table (`glued_note_after_table`).
            if rendered_len >= threshold {
                let glue_caption_to_table =
                    buf.last().is_some_and(is_nonempty_text_block) && is_table(el);
                let glue_note_to_table = buf.last().is_some_and(is_table)
                    && matches!(el, Element::Paragraph(_))
                    && is_nonempty_text_block(el)
                    && !glued_note_after_table;

                if glue_caption_to_table {
                    buf_start.get_or_insert(element_index);
                    buf.push(el.clone());
                    glued_note_after_table = false;
                    continue;
                }
                if glue_note_to_table {
                    buf_start.get_or_insert(element_index);
                    buf.push(el.clone());
                    glued_note_after_table = true;
                    continue;
                }

                flush_recording!(element_index);
            }

            buf_start.get_or_insert(element_index);
            buf.push(el.clone());
            glued_note_after_table = false;
        }
        flush_recording!(section.elements.len());
    }
    (out, spans)
}

fn is_empty_paragraph(el: &Element) -> bool {
    match el {
        Element::Paragraph(p) => inline_md(&p.content).trim().is_empty(),
        _ => false,
    }
}

fn is_table(el: &Element) -> bool {
    matches!(el, Element::Table(_))
}

fn is_nonempty_text_block(el: &Element) -> bool {
    match el {
        Element::Paragraph(p) => !inline_md(&p.content).trim().is_empty(),
        Element::Heading(_) => true,
        _ => false,
    }
}

/// Does this format's `Section::title` name a container, or guess a title?
///
/// A spreadsheet's section is a sheet and a presentation's is a slide: the name is the author's,
/// it addresses the content, and the headings inside it are its children. For `docx`/`doc` there
/// is no such container — `office_oxide` fills `Section::title` with the text of the section's
/// *first heading* (`convert_docx.rs:74`, `convert_doc.rs:32`), which is already in
/// `heading_path`. Prepending that would either duplicate the heading or, worse, hang every later
/// heading in the document under the first one. `odt` sections carry no title at all.
fn section_is_container(file_type: &str) -> bool {
    matches!(file_type, "xlsx" | "xls" | "ods" | "pptx" | "ppt" | "odp")
}

fn flush_buf(
    path: &Path,
    file_type: &str,
    buf: &mut Vec<Element>,
    heading_path: &[String],
    section_title: Option<&str>,
    out: &mut Vec<Chunk>,
) {
    if buf.is_empty() {
        return;
    }
    let text = render_elements(buf);
    buf.clear();
    if text.trim().is_empty() {
        return;
    }
    // `location` is a path, and the sheet or slide a heading sits in is one of its segments.
    // Dropping the container the moment a heading appeared was the bug: two slides that both open
    // with "Intro" produced the same `location`, and the graph keys its per-file `seen` map on that
    // string (`index/store.rs`) while `nearest_ancestor` splits it on `LOCATION_SEP` — so the second
    // container overwrote the first, and any deeper chunk in either one parented to whichever wrote
    // last. Not every `Section::title` is a container, though — see `section_is_container`.
    // The emptiness test trims, so this filter and `Chunk::nested`'s own cannot drift apart: a
    // whitespace-only sheet or slide title addresses nothing either way and is not a segment. It
    // used to survive the `is_empty()` test here and store a lone `" "`; such a chunk now stores
    // `""`. That is the same declared change as the empty docx heading, reached from the container
    // side. The title that DOES survive is pushed verbatim — trimming decides whether a segment
    // exists, never what an author's text says.
    let mut segs: Vec<String> = Vec::new();
    if let Some(t) = section_title.filter(|t| !t.trim().is_empty()) {
        let container_of_these_headings =
            section_is_container(file_type) && heading_path.first().map(String::as_str) != Some(t);
        if heading_path.is_empty() || container_of_these_headings {
            segs.push(t.to_string());
        }
    }
    segs.extend(heading_path.iter().cloned());
    out.push(Chunk::nested(
        path.to_path_buf(),
        file_type.to_string(),
        segs,
        text,
    ));
}

pub(crate) fn render_elements(elements: &[Element]) -> String {
    let mut parts = Vec::new();
    for el in elements {
        let s = match el {
            Element::Heading(h) => {
                let level = (h.level as usize).clamp(1, 6);
                format!("{} {}", "#".repeat(level), inline_md(&h.content))
            }
            Element::Paragraph(p) => inline_md(&p.content),
            Element::Table(t) => table_to_markdown(t),
            Element::List(list) => render_list(list, 0),
            Element::TextBox(text_box) => render_elements(&text_box.content),
            Element::ThematicBreak => "---".to_string(),
            Element::CodeBlock(cb) => {
                let lang = cb.language.as_deref().unwrap_or("");
                format!("```{lang}\n{}\n```", cb.content)
            }
            _ => String::new(),
        };
        if !s.trim().is_empty() {
            parts.push(s);
        }
    }
    parts.join("\n\n")
}

fn inline_md(content: &[InlineContent]) -> String {
    let mut out = String::new();
    for c in content {
        match c {
            InlineContent::Text(t) => {
                let mut s = t.text.clone();
                if t.bold && t.italic {
                    s = format!("***{s}***");
                } else if t.bold {
                    s = format!("**{s}**");
                } else if t.italic {
                    s = format!("*{s}*");
                }
                if let Some(url) = &t.hyperlink {
                    s = format!("[{s}]({url})");
                }
                out.push_str(&s);
            }
            InlineContent::LineBreak => out.push('\n'),
            _ => {}
        }
    }
    out
}

fn render_list(list: &List, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    let mut lines = Vec::new();
    for (i, item) in list.items.iter().enumerate() {
        let body = item
            .content
            .iter()
            .map(|e| match e {
                Element::Paragraph(p) => inline_md(&p.content),
                other => render_elements(std::slice::from_ref(other)),
            })
            .collect::<Vec<_>>()
            .join(" ");
        let marker = if list.ordered {
            format!("{}. ", i + 1)
        } else {
            "- ".into()
        };
        lines.push(format!("{pad}{marker}{body}"));
        if let Some(nested) = &item.nested {
            lines.push(render_list(nested, indent + 1));
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use office_oxide::ir::{Heading, ListItem, Paragraph, Section, TextBox, TextSpan};

    fn text_para(s: &str) -> Element {
        Element::Paragraph(Paragraph {
            content: vec![InlineContent::Text(TextSpan {
                text: s.to_string(),
                ..Default::default()
            })],
            ..Default::default()
        })
    }

    /// The report is parallel to the chunk vector and carries the section index, because element
    /// indices restart in every section while chunk indices do not.
    #[test]
    fn chunk_ir_reports_a_section_and_element_range_per_chunk() {
        let heading = |text: &str| {
            Element::Heading(Heading {
                level: 1,
                content: vec![InlineContent::Text(TextSpan {
                    text: text.into(),
                    ..Default::default()
                })],
                ..Default::default()
            })
        };
        let ir = DocumentIR {
            sections: vec![
                Section {
                    elements: vec![heading("H1"), text_para("a"), text_para("b")],
                    ..Default::default()
                },
                Section {
                    elements: vec![text_para("c")],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let (chunks, spans) = chunk_ir_with_spans(Path::new("a.docx"), &ir, "docx");
        assert_eq!(chunks.len(), spans.len(), "{spans:?}");
        assert_eq!(
            spans,
            vec![
                ChunkSpan {
                    section: 0,
                    elements: 0..3,
                },
                ChunkSpan {
                    section: 1,
                    elements: 0..1,
                },
            ],
            "{spans:?}"
        );
        assert!(chunks[0].text.contains("H1") && chunks[0].text.contains('b'));
        assert!(chunks[1].text.contains('c'));
    }

    /// The guard that the bookkeeping is additive: the chunks are byte-for-byte what they were
    /// before the report existed, and the reporting entry point agrees with the plain one.
    #[test]
    fn chunk_ir_output_is_unchanged_by_the_reporting() {
        let ir = DocumentIR {
            sections: vec![Section {
                elements: vec![
                    Element::Heading(Heading {
                        level: 1,
                        content: vec![InlineContent::Text(TextSpan {
                            text: "Alpha".into(),
                            ..Default::default()
                        })],
                        ..Default::default()
                    }),
                    text_para("one"),
                    text_para(""),
                    Element::Heading(Heading {
                        level: 2,
                        content: vec![InlineContent::Text(TextSpan {
                            text: "Beta".into(),
                            ..Default::default()
                        })],
                        ..Default::default()
                    }),
                    text_para("two"),
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let chunks = chunk_ir(Path::new("a.docx"), &ir, "docx");
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let locs: Vec<String> = chunks.iter().map(|c| c.location.display()).collect();
        assert_eq!(texts, ["# Alpha\n\none", "## Beta\n\ntwo"], "{texts:?}");
        assert_eq!(locs, ["Alpha", "Alpha > Beta"], "{locs:?}");

        let (reported, _) = chunk_ir_with_spans(Path::new("a.docx"), &ir, "docx");
        assert_eq!(reported.len(), chunks.len());
        for (a, b) in reported.iter().zip(&chunks) {
            assert_eq!(a.text, b.text);
            assert_eq!(a.location.display(), b.location.display());
            assert_eq!(a.file_type, b.file_type);
            assert_eq!(a.doc_path, b.doc_path);
        }
    }

    #[test]
    fn chunk_splits_on_heading() {
        let ir = DocumentIR {
            sections: vec![Section {
                elements: vec![
                    text_para("intro"),
                    Element::Heading(Heading {
                        level: 1,
                        content: vec![InlineContent::Text(TextSpan {
                            text: "H1".into(),
                            ..Default::default()
                        })],
                        ..Default::default()
                    }),
                    text_para("under"),
                ],
                ..Default::default()
            }],
            ..Default::default()
        };
        let chunks = chunk_ir(Path::new("a.docx"), &ir, "docx");
        assert!(chunks.len() >= 2, "{:?}", chunks.len());
        assert!(chunks[0].text.contains("intro"));
        assert_eq!(chunks.last().unwrap().location.display(), "H1");
        assert!(chunks.last().unwrap().text.contains("under"));
    }

    #[test]
    fn chunk_splits_on_section_boundary() {
        let ir = DocumentIR {
            sections: vec![
                Section {
                    title: Some("Sheet1".into()),
                    elements: vec![text_para("a")],
                    ..Default::default()
                },
                Section {
                    title: Some("Sheet2".into()),
                    elements: vec![text_para("b")],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let chunks = chunk_ir(Path::new("a.xlsx"), &ir, "xlsx");
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].location.display(), "Sheet1");
        assert_eq!(chunks[1].location.display(), "Sheet2");
        // A sheet is a container the document names, so a chunk whose whole location is its name
        // is nested in it rather than flat.
        assert!(chunks[0].location.is_nested());
    }

    /// Two slides opening with the same heading used to produce the SAME `location`, because a
    /// heading replaced the slide name instead of hanging under it. The graph keys its per-file
    /// `seen` map on that string, so one slide's entry overwrote the other's and a deeper chunk in
    /// either slide parented across the boundary.
    #[test]
    fn a_slide_name_is_the_ancestor_of_the_headings_inside_it() {
        let slide = |name: &str| Section {
            title: Some(name.into()),
            elements: vec![
                Element::Heading(Heading {
                    level: 1,
                    content: vec![InlineContent::Text(TextSpan {
                        text: "Intro".into(),
                        ..Default::default()
                    })],
                    ..Default::default()
                }),
                text_para("body"),
            ],
            ..Default::default()
        };
        let ir = DocumentIR {
            sections: vec![slide("Slide1"), slide("Slide2")],
            ..Default::default()
        };
        let locs: Vec<String> = chunk_ir(Path::new("a.odp"), &ir, "odp")
            .iter()
            .map(|c| c.location.display())
            .collect();
        assert_eq!(locs, ["Slide1 > Intro", "Slide2 > Intro"], "{locs:?}");
    }

    /// The counter-case that keeps the fix honest: `office_oxide` fills a docx section's title with
    /// the text of its first heading, so prepending it there would read "H1 > H1" — and would hang
    /// every later heading in the document under the first one.
    #[test]
    fn a_guessed_docx_section_title_is_not_an_ancestor() {
        let ir = DocumentIR {
            sections: vec![Section {
                title: Some("First".into()),
                elements: vec![
                    Element::Heading(Heading {
                        level: 1,
                        content: vec![InlineContent::Text(TextSpan {
                            text: "First".into(),
                            ..Default::default()
                        })],
                        ..Default::default()
                    }),
                    text_para("a"),
                    Element::Heading(Heading {
                        level: 1,
                        content: vec![InlineContent::Text(TextSpan {
                            text: "Second".into(),
                            ..Default::default()
                        })],
                        ..Default::default()
                    }),
                    text_para("b"),
                ],
                ..Default::default()
            }],
            ..Default::default()
        };
        let locs: Vec<String> = chunk_ir(Path::new("a.docx"), &ir, "docx")
            .iter()
            .map(|c| c.location.display())
            .collect();
        assert_eq!(locs, ["First", "Second"], "{locs:?}");
    }

    /// A heading with empty content used to render a dangling separator. It now renders the
    /// surviving segment alone — the only display change in this plan, asserted as the new string
    /// so it is a decision rather than a diff nobody looked at.
    ///
    /// The plan placed this test in `src/extract/markdown.rs`, but markdown cannot reach the
    /// behaviour: `parse_atx_heading` rejects an empty title, so a markdown heading chain never has
    /// a hole in it. `convert_docx.rs` is the one feeder that pushes a `Heading` without guarding
    /// its text, and `heading_path` does not filter it, so a docx IR is where the case lives.
    #[test]
    fn an_empty_heading_segment_no_longer_renders_a_dangling_separator() {
        let ir = DocumentIR {
            sections: vec![Section {
                elements: vec![
                    // An empty heading-styled paragraph: `convert_docx.rs` pushes it unguarded.
                    Element::Heading(Heading {
                        level: 1,
                        content: vec![InlineContent::Text(TextSpan {
                            text: String::new(),
                            ..Default::default()
                        })],
                        ..Default::default()
                    }),
                    text_para("a"),
                    Element::Heading(Heading {
                        level: 2,
                        content: vec![InlineContent::Text(TextSpan {
                            text: "B".into(),
                            ..Default::default()
                        })],
                        ..Default::default()
                    }),
                    text_para("b"),
                ],
                ..Default::default()
            }],
            ..Default::default()
        };
        let chunks = chunk_ir(Path::new("a.docx"), &ir, "docx");
        assert_eq!(chunks.len(), 2, "{chunks:?}");
        assert_eq!(chunks[1].location.display(), "B");
    }

    #[test]
    fn soft_split_on_empty_para_only_after_threshold() {
        let ir = DocumentIR {
            sections: vec![Section {
                elements: vec![
                    text_para(&"x".repeat(50)),
                    text_para(""),
                    text_para("still-same"),
                ],
                ..Default::default()
            }],
            ..Default::default()
        };
        let chunks = chunk_ir_with_threshold(Path::new("a.docx"), &ir, "docx", 10_000);
        assert_eq!(chunks.len(), 1);

        let ir = DocumentIR {
            sections: vec![Section {
                elements: vec![
                    text_para(&"a".repeat(30)),
                    text_para(""),
                    text_para(&"b".repeat(5)),
                ],
                ..Default::default()
            }],
            ..Default::default()
        };
        let chunks = chunk_ir_with_threshold(Path::new("a.docx"), &ir, "docx", 20);
        assert!(chunks.len() >= 2, "got {}", chunks.len());
        for chunk in &chunks {
            assert!(
                !chunk.text.contains("\n\n\n"),
                "kept empty para must not add extra blank lines: {:?}",
                chunk.text
            );
        }
    }

    #[test]
    fn glue_caption_and_note_around_table() {
        use office_oxide::ir::{Table, TableCell, TableRow};

        let table = Element::Table(Table {
            rows: vec![TableRow {
                cells: vec![TableCell {
                    content: vec![text_para("1")],
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        });
        let ir = DocumentIR {
            sections: vec![Section {
                elements: vec![
                    text_para(&"p".repeat(40)),
                    text_para("Table 1 — caption"),
                    table,
                    text_para("Note under"),
                    text_para("second after note"),
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let chunks = chunk_ir_with_threshold(Path::new("a.docx"), &ir, "docx", 30);
        assert!(
            chunks.iter().any(|chunk| {
                chunk.text.contains("Table 1 — caption")
                    && chunk.text.contains("| 1 |")
                    && chunk.text.contains("Note under")
            }),
            "expected glued caption+table+note chunk, got {:?}",
            chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>()
        );
        assert!(
            !chunks.iter().any(|chunk| {
                chunk.text.contains("Note under") && chunk.text.contains("second after note")
            }),
            "only one note may glue to the table; second para should split out"
        );
        if chunks.len() >= 2 {
            assert!(
                chunks.iter().any(|c| c.text.contains("second after note")),
                "later chunk should carry the post-note paragraph"
            );
        }
    }

    #[test]
    fn render_heading_and_paragraph() {
        let els = vec![
            Element::Heading(Heading {
                level: 1,
                content: vec![InlineContent::Text(TextSpan {
                    text: "Title".into(),
                    ..Default::default()
                })],
                ..Default::default()
            }),
            text_para("Body"),
        ];
        let md = render_elements(&els);
        assert!(md.contains("# Title"), "{md}");
        assert!(md.contains("Body"), "{md}");
    }

    #[test]
    fn render_unordered_list() {
        let els = vec![Element::List(List {
            ordered: false,
            items: vec![
                ListItem {
                    content: vec![text_para("First")],
                    nested: None,
                },
                ListItem {
                    content: vec![text_para("Second")],
                    nested: None,
                },
            ],
            ..Default::default()
        })];
        let md = render_elements(&els);
        assert!(md.contains("- First"), "{md}");
        assert!(md.contains("- Second"), "{md}");
    }

    #[test]
    fn renders_text_box_content_in_elements_and_chunks() {
        let text_box = Element::TextBox(TextBox {
            content: vec![text_para("slide body")],
            ..Default::default()
        });
        assert!(render_elements(std::slice::from_ref(&text_box)).contains("slide body"));

        let ir = DocumentIR {
            sections: vec![Section {
                elements: vec![text_box],
                ..Default::default()
            }],
            ..Default::default()
        };
        let chunks = chunk_ir(Path::new("slides.pptx"), &ir, "pptx");
        assert_eq!(chunks.len(), 1);
        assert!(
            chunks[0].text.contains("slide body"),
            "{:?}",
            chunks[0].text
        );
    }

    #[test]
    fn renders_hyperlinked_text_after_emphasis() {
        let content = vec![InlineContent::Text(TextSpan {
            text: "linked".into(),
            bold: true,
            hyperlink: Some("https://example.com".into()),
            ..Default::default()
        })];

        assert_eq!(inline_md(&content), "[**linked**](https://example.com)");
    }
}
