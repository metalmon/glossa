use office_oxide::ir::{
    DocumentIR, Element, InlineContent, Paragraph, Table, TableCell, TableRow, TextSpan,
};

/// Expand every table in the IR so merged spans become a dense grid
/// with the origin value repeated in each covered cell.
pub fn expand_merged_tables(ir: &mut DocumentIR) {
    for section in &mut ir.sections {
        for el in &mut section.elements {
            expand_element(el);
        }
    }
}

fn expand_element(el: &mut Element) {
    match el {
        Element::Table(t) => {
            for row in &mut t.rows {
                for cell in &mut row.cells {
                    for child in &mut cell.content {
                        expand_element(child);
                    }
                }
            }
            *t = expand_table(t);
        }
        Element::List(list) => {
            for item in &mut list.items {
                for child in &mut item.content {
                    expand_element(child);
                }
                if let Some(nested) = &mut item.nested {
                    for nested_item in &mut nested.items {
                        for child in &mut nested_item.content {
                            expand_element(child);
                        }
                    }
                }
            }
        }
        Element::TextBox(text_box) => {
            for child in &mut text_box.content {
                expand_element(child);
            }
        }
        _ => {}
    }
}

pub fn expand_table(table: &Table) -> Table {
    let n_rows = table.rows.len();
    if n_rows == 0 {
        return Table::default();
    }

    let mut occupancy: Vec<u32> = Vec::new();
    let mut max_cols = 0usize;
    let mut placements: Vec<(usize, usize, usize, usize)> = Vec::new();

    for (r, row) in table.rows.iter().enumerate() {
        for occ in &mut occupancy {
            if *occ > 0 {
                *occ -= 1;
            }
        }

        let mut col = 0usize;
        for cell in &row.cells {
            while col < occupancy.len() && occupancy[col] > 0 {
                col += 1;
            }

            let col_span = cell.col_span.max(1) as usize;
            let row_span = cell.row_span.max(1) as usize;

            placements.push((r, col, col_span, row_span));

            for c in col..col + col_span {
                while occupancy.len() <= c {
                    occupancy.push(0);
                }
                occupancy[c] = occupancy[c].max(row_span as u32);
            }

            col += col_span;
            max_cols = max_cols.max(col);
        }
    }

    if max_cols == 0 {
        return Table::default();
    }

    let mut grid: Vec<Vec<Option<TableCell>>> = (0..n_rows)
        .map(|_| (0..max_cols).map(|_| None).collect())
        .collect();

    let mut placement_idx = 0usize;
    let mut occupancy: Vec<u32> = Vec::new();

    for (r, row) in table.rows.iter().enumerate() {
        for occ in &mut occupancy {
            if *occ > 0 {
                *occ -= 1;
            }
        }

        let mut col = 0usize;
        for cell in &row.cells {
            while col < occupancy.len() && occupancy[col] > 0 {
                col += 1;
            }

            let (pr, pc, col_span, row_span) = placements[placement_idx];
            debug_assert_eq!(pr, r);
            debug_assert_eq!(pc, col);

            // Clamp the vertical span to the rows actually available. A
            // malformed/adversarial `row_span` (or a well-formed one whose
            // covered continuation rows got dropped elsewhere, e.g. ODF's
            // trailing-empty-row clamp) can otherwise point past the last row
            // and panic on `grid[r + dr]` (index out of bounds). Horizontal
            // col_span is safe because `max_cols` grows to fit it; vertical
            // is not, since `n_rows` is fixed by `table.rows.len()`. In-range
            // spans (the overwhelming common case) are completely unaffected.
            let clamped_row_span = row_span.min(n_rows - r);
            let densified = densified_cell(cell);
            for dr in 0..clamped_row_span {
                for dc in 0..col_span {
                    grid[r + dr][col + dc] = Some(densified.clone());
                }
            }

            for c in col..col + col_span {
                while occupancy.len() <= c {
                    occupancy.push(0);
                }
                occupancy[c] = occupancy[c].max(row_span as u32);
            }

            col += col_span;
            placement_idx += 1;
        }
    }

    let empty = text_cell("");
    let mut out_rows = Vec::with_capacity(n_rows);
    for (r, row) in table.rows.iter().enumerate() {
        let cells: Vec<TableCell> = grid[r]
            .iter()
            .map(|slot| slot.clone().unwrap_or_else(|| empty.clone()))
            .collect();
        out_rows.push(TableRow {
            cells,
            is_header: row.is_header,
            height_twips: row.height_twips,
            allow_break: row.allow_break,
            repeat_as_header: row.repeat_as_header,
        });
    }

    Table {
        rows: out_rows,
        column_widths_twips: table.column_widths_twips.clone(),
        border: table.border.clone(),
        alignment: table.alignment.clone(),
        cell_padding_twips: table.cell_padding_twips,
        caption: table.caption.clone(),
        width_twips: table.width_twips,
        indent_left_twips: table.indent_left_twips,
    }
}

fn densified_cell(cell: &TableCell) -> TableCell {
    let mut c = cell.clone();
    c.col_span = 1;
    c.row_span = 1;
    c
}

/// GFM pipe table from an already-expanded (or span=1) table.
pub fn table_to_markdown(table: &Table) -> String {
    if table.rows.is_empty() {
        return String::new();
    }

    let col_count = table.rows.iter().map(|r| r.cells.len()).max().unwrap_or(0);
    if col_count == 0 {
        return String::new();
    }

    let mut result = String::new();

    let first_row = &table.rows[0];
    result.push('|');
    for i in 0..col_count {
        let text = first_row
            .cells
            .get(i)
            .map(markdown_cell_text)
            .unwrap_or_default();
        result.push(' ');
        result.push_str(&text);
        result.push_str(" |");
    }
    result.push('\n');

    result.push('|');
    for _ in 0..col_count {
        result.push_str(" --- |");
    }
    result.push('\n');

    for row in table.rows.iter().skip(1) {
        result.push('|');
        for i in 0..col_count {
            let text = row.cells.get(i).map(markdown_cell_text).unwrap_or_default();
            result.push(' ');
            result.push_str(&text);
            result.push_str(" |");
        }
        result.push('\n');
    }

    if result.ends_with('\n') {
        result.pop();
    }

    result
}

fn markdown_cell_text(cell: &TableCell) -> String {
    cell_plain(cell).replace('|', "\\|")
}

fn cell_plain(cell: &TableCell) -> String {
    cell.content
        .iter()
        .map(element_plain)
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .replace('\n', " ")
}

fn element_plain(element: &Element) -> String {
    match element {
        Element::Paragraph(p) => inline_plain(&p.content),
        Element::Heading(h) => inline_plain(&h.content),
        Element::List(list) => list
            .items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let marker = if list.ordered {
                    format!("{}. ", index + 1)
                } else {
                    "- ".to_string()
                };
                let mut text = item
                    .content
                    .iter()
                    .map(element_plain)
                    .filter(|text| !text.trim().is_empty())
                    .collect::<Vec<_>>()
                    .join(" ");
                if let Some(nested) = &item.nested {
                    let nested_text = element_plain(&Element::List(nested.clone()));
                    if !nested_text.is_empty() {
                        if !text.is_empty() {
                            text.push(' ');
                        }
                        text.push_str(&nested_text);
                    }
                }
                format!("{marker}{text}")
            })
            .collect::<Vec<_>>()
            .join(" "),
        Element::Table(table) => table_to_markdown(table),
        Element::TextBox(text_box) => text_box
            .content
            .iter()
            .map(element_plain)
            .filter(|text| !text.trim().is_empty())
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

fn inline_plain(content: &[InlineContent]) -> String {
    let mut s = String::new();
    for c in content {
        match c {
            InlineContent::Text(t) => s.push_str(&t.text),
            InlineContent::LineBreak => s.push(' '),
            _ => {}
        }
    }
    s
}

fn text_cell(s: &str) -> TableCell {
    TableCell {
        content: vec![Element::Paragraph(Paragraph {
            content: vec![InlineContent::Text(TextSpan {
                text: s.to_string(),
                ..Default::default()
            })],
            ..Default::default()
        })],
        col_span: 1,
        row_span: 1,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use office_oxide::ir::{List, ListItem, Section, TextBox};

    #[test]
    fn expand_merged_tables_on_document_ir() {
        let table = Table {
            rows: vec![TableRow {
                cells: vec![
                    TableCell {
                        content: text_cell("A").content,
                        col_span: 2,
                        row_span: 1,
                        ..Default::default()
                    },
                    text_cell("B"),
                ],
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut ir = DocumentIR {
            sections: vec![Section {
                elements: vec![Element::Table(table)],
                ..Default::default()
            }],
            ..Default::default()
        };
        expand_merged_tables(&mut ir);
        let Element::Table(expanded) = &ir.sections[0].elements[0] else {
            panic!("expected table element");
        };
        assert_eq!(expanded.rows[0].cells.len(), 3);
        assert_eq!(cell_plain(&expanded.rows[0].cells[0]), "A");
        assert_eq!(cell_plain(&expanded.rows[0].cells[1]), "A");
        assert_eq!(cell_plain(&expanded.rows[0].cells[2]), "B");
        let md = table_to_markdown(expanded);
        assert!(md.contains("| A | A |"), "got:\n{md}");
    }

    #[test]
    fn expand_horizontal_span_repeats_value() {
        // | A (colspan 2) |   | B |
        let table = Table {
            rows: vec![TableRow {
                cells: vec![
                    TableCell {
                        content: text_cell("A").content,
                        col_span: 2,
                        row_span: 1,
                        ..Default::default()
                    },
                    text_cell("B"),
                ],
                ..Default::default()
            }],
            ..Default::default()
        };
        let expanded = expand_table(&table);
        assert_eq!(expanded.rows.len(), 1);
        assert_eq!(expanded.rows[0].cells.len(), 3);
        assert_eq!(cell_plain(&expanded.rows[0].cells[0]), "A");
        assert_eq!(cell_plain(&expanded.rows[0].cells[1]), "A");
        assert_eq!(cell_plain(&expanded.rows[0].cells[2]), "B");
        assert!(expanded.rows[0]
            .cells
            .iter()
            .all(|c| c.col_span == 1 && c.row_span == 1));
    }

    #[test]
    fn expand_vertical_span_repeats_value() {
        // col0: X rowspan 2; col1: Y / Z
        let table = Table {
            rows: vec![
                TableRow {
                    cells: vec![
                        TableCell {
                            content: text_cell("X").content,
                            col_span: 1,
                            row_span: 2,
                            ..Default::default()
                        },
                        text_cell("Y"),
                    ],
                    ..Default::default()
                },
                TableRow {
                    // office_oxide omits vMerge continue — only Z
                    cells: vec![text_cell("Z")],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let expanded = expand_table(&table);
        assert_eq!(expanded.rows.len(), 2);
        assert_eq!(expanded.rows[0].cells.len(), 2);
        assert_eq!(expanded.rows[1].cells.len(), 2);
        assert_eq!(cell_plain(&expanded.rows[0].cells[0]), "X");
        assert_eq!(cell_plain(&expanded.rows[1].cells[0]), "X");
        assert_eq!(cell_plain(&expanded.rows[0].cells[1]), "Y");
        assert_eq!(cell_plain(&expanded.rows[1].cells[1]), "Z");
    }

    #[test]
    fn expand_table_clamps_row_span_exceeding_available_rows() {
        // Regression (CRITICAL panic): a malformed/adversarial row_span
        // pointing past the table's last row must be clamped, not panic
        // `grid[r + dr]` with an out-of-bounds index. This can happen from a
        // parser bug/malformed file (row_span=2 on a 1-row table) or from a
        // well-formed source whose covered continuation rows got dropped
        // elsewhere (e.g. ODF's trailing-empty-row clamp) before reaching
        // expand_table.
        let table = Table {
            rows: vec![TableRow {
                cells: vec![TableCell {
                    content: text_cell("only").content,
                    col_span: 1,
                    row_span: 2, // exceeds n_rows = 1
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let expanded = expand_table(&table); // must not panic
        assert_eq!(
            expanded.rows.len(),
            1,
            "grid must stay 1 row (n_rows), not grow to row_span"
        );
        assert_eq!(expanded.rows[0].cells.len(), 1);
        assert_eq!(cell_plain(&expanded.rows[0].cells[0]), "only");
    }

    #[test]
    fn expand_2x2_block_repeats_value() {
        let table = Table {
            rows: vec![
                TableRow {
                    cells: vec![
                        TableCell {
                            content: text_cell("M").content,
                            col_span: 2,
                            row_span: 2,
                            ..Default::default()
                        },
                        text_cell("R1"),
                    ],
                    ..Default::default()
                },
                TableRow {
                    cells: vec![text_cell("R2")],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        // Grid width: colspan2 + R1 = 3 cols; row2 has continue omitted for M + R2
        let expanded = expand_table(&table);
        assert_eq!(expanded.rows[0].cells.len(), 3);
        assert_eq!(expanded.rows[1].cells.len(), 3);
        assert_eq!(cell_plain(&expanded.rows[0].cells[0]), "M");
        assert_eq!(cell_plain(&expanded.rows[0].cells[1]), "M");
        assert_eq!(cell_plain(&expanded.rows[1].cells[0]), "M");
        assert_eq!(cell_plain(&expanded.rows[1].cells[1]), "M");
        assert_eq!(cell_plain(&expanded.rows[0].cells[2]), "R1");
        assert_eq!(cell_plain(&expanded.rows[1].cells[2]), "R2");
    }

    #[test]
    fn table_to_markdown_escapes_pipes_in_cell_text() {
        let table = Table {
            rows: vec![TableRow {
                cells: vec![text_cell("a|b"), text_cell("c")],
                ..Default::default()
            }],
            ..Default::default()
        };
        let md = table_to_markdown(&table);
        assert!(md.contains("| a\\|b | c |"), "got:\n{md}");
    }

    #[test]
    fn table_to_markdown_pipes_expanded_grid() {
        let table = expand_table(&Table {
            rows: vec![TableRow {
                cells: vec![TableCell {
                    content: text_cell("A").content,
                    col_span: 2,
                    row_span: 1,
                    ..Default::default()
                }],
                is_header: true,
                ..Default::default()
            }],
            ..Default::default()
        });
        let md = table_to_markdown(&table);
        assert!(md.contains("| A | A |"), "got:\n{md}");
        assert!(md.contains("---"), "got:\n{md}");
    }

    #[test]
    fn table_to_markdown_keeps_list_and_nested_table_cell_content() {
        let nested_table = Table {
            rows: vec![TableRow {
                cells: vec![text_cell("nested value")],
                ..Default::default()
            }],
            ..Default::default()
        };
        let table = Table {
            rows: vec![TableRow {
                cells: vec![
                    TableCell {
                        content: vec![Element::List(List {
                            ordered: false,
                            items: vec![ListItem {
                                content: vec![Element::Paragraph(Paragraph {
                                    content: vec![InlineContent::Text(TextSpan {
                                        text: "bullet text".into(),
                                        ..Default::default()
                                    })],
                                    ..Default::default()
                                })],
                                nested: None,
                            }],
                            ..Default::default()
                        })],
                        ..Default::default()
                    },
                    TableCell {
                        content: vec![Element::Table(nested_table)],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let md = table_to_markdown(&table);
        assert!(md.contains("- bullet text"), "got:\n{md}");
        assert!(md.contains("nested value"), "got:\n{md}");
    }

    #[test]
    fn expands_tables_nested_in_text_boxes() {
        let nested_table = Table {
            rows: vec![TableRow {
                cells: vec![TableCell {
                    content: text_cell("A").content,
                    col_span: 2,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut ir = DocumentIR {
            sections: vec![Section {
                elements: vec![Element::TextBox(TextBox {
                    content: vec![Element::Table(nested_table)],
                    ..Default::default()
                })],
                ..Default::default()
            }],
            ..Default::default()
        };

        expand_merged_tables(&mut ir);
        let Element::TextBox(text_box) = &ir.sections[0].elements[0] else {
            panic!("expected text box");
        };
        let Element::Table(table) = &text_box.content[0] else {
            panic!("expected nested table");
        };
        assert_eq!(table.rows[0].cells.len(), 2);
        assert!(table.rows[0].cells.iter().all(|cell| cell.col_span == 1));
    }
}
