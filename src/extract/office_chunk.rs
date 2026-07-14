use crate::extract::office_table::table_to_markdown;
use crate::model::Chunk;
use office_oxide::ir::{
    DocumentIR, Element, Heading, InlineContent, List, ListItem, Paragraph, TextSpan,
};
use std::path::Path;

pub const CHUNK_CHAR_THRESHOLD: usize = 4000;

pub fn chunk_ir(path: &Path, ir: &DocumentIR, file_type: &str) -> Vec<Chunk> {
    let _ = (path, ir, file_type);
    todo!("chunk_ir")
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

fn text_para(s: &str) -> Element {
    Element::Paragraph(Paragraph {
        content: vec![InlineContent::Text(TextSpan {
            text: s.to_string(),
            ..Default::default()
        })],
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
