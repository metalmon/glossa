use crate::extract::office_table::table_to_markdown;
use crate::model::Chunk;
use office_oxide::ir::{DocumentIR, Element, InlineContent, List};
use std::path::Path;

pub const CHUNK_CHAR_THRESHOLD: usize = 4000;

pub fn chunk_ir(path: &Path, ir: &DocumentIR, file_type: &str) -> Vec<Chunk> {
    chunk_ir_with_threshold(path, ir, file_type, CHUNK_CHAR_THRESHOLD)
}

pub fn chunk_ir_with_threshold(
    path: &Path,
    ir: &DocumentIR,
    file_type: &str,
    threshold: usize,
) -> Vec<Chunk> {
    let mut out = Vec::new();

    for section in &ir.sections {
        let section_title = section.title.as_deref();
        let mut heading_path: Vec<String> = Vec::new();
        let mut buf: Vec<Element> = Vec::new();
        let mut glued_note_after_table = false;

        for el in &section.elements {
            if let Element::Heading(h) = el {
                flush_buf(
                    path,
                    file_type,
                    &mut buf,
                    &heading_path,
                    section_title,
                    &mut out,
                );
                glued_note_after_table = false;
                let title = inline_md(&h.content);
                let level = h.level as usize;
                heading_path.truncate(level.saturating_sub(1));
                heading_path.push(title);
                buf.push(el.clone());
                continue;
            }

            let rendered_len = render_elements(&buf).chars().count();
            if is_empty_paragraph(el) {
                if rendered_len >= threshold {
                    flush_buf(
                        path,
                        file_type,
                        &mut buf,
                        &heading_path,
                        section_title,
                        &mut out,
                    );
                    glued_note_after_table = false;
                    continue;
                }
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
                    buf.push(el.clone());
                    glued_note_after_table = false;
                    continue;
                }
                if glue_note_to_table {
                    buf.push(el.clone());
                    glued_note_after_table = true;
                    continue;
                }

                flush_buf(
                    path,
                    file_type,
                    &mut buf,
                    &heading_path,
                    section_title,
                    &mut out,
                );
            }

            buf.push(el.clone());
            glued_note_after_table = false;
        }
        flush_buf(
            path,
            file_type,
            &mut buf,
            &heading_path,
            section_title,
            &mut out,
        );
    }
    out
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
    let location = if !heading_path.is_empty() {
        heading_path.join(" > ")
    } else {
        section_title.unwrap_or("").to_string()
    };
    out.push(Chunk {
        doc_path: path.to_path_buf(),
        location,
        file_type: file_type.to_string(),
        text,
    });
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
        assert_eq!(chunks.last().unwrap().location, "H1");
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
        assert_eq!(chunks[0].location, "Sheet1");
        assert_eq!(chunks[1].location, "Sheet2");
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
