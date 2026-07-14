use pdf_oxide::structure::{Table, TableCell, TableRow};

/// Expand merged cells into a dense grid with origin text repeated in every covered slot.
pub fn expand_table(table: &Table) -> Table {
    let n_rows = table.rows.len();
    if n_rows == 0 {
        return Table::default();
    }

    let mut occupancy: Vec<u32> = Vec::new();
    let mut max_cols = table.col_count;
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

            let col_span = cell.colspan.max(1) as usize;
            let row_span = cell.rowspan.max(1) as usize;

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

            let densified = densified_cell(cell);
            for dr in 0..row_span {
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

    let empty = empty_cell();
    let mut out_rows = Vec::with_capacity(n_rows);
    for (r, row) in table.rows.iter().enumerate() {
        let cells: Vec<TableCell> = grid[r]
            .iter()
            .map(|slot| slot.clone().unwrap_or_else(|| empty.clone()))
            .collect();
        out_rows.push(TableRow {
            cells,
            is_header: row.is_header,
        });
    }

    Table {
        rows: out_rows,
        has_header: table.has_header,
        col_count: max_cols,
        bbox: table.bbox,
    }
}

fn densified_cell(cell: &TableCell) -> TableCell {
    TableCell::new(cell.text.clone(), cell.is_header)
}

fn empty_cell() -> TableCell {
    TableCell::new(String::new(), false)
}

/// GFM pipe table from an already-expanded (or span=1) table.
pub fn table_to_markdown(table: &Table) -> String {
    if table.rows.is_empty() {
        return String::new();
    }

    let col_count = table
        .rows
        .iter()
        .map(|r| r.cells.len())
        .max()
        .unwrap_or(0);
    if col_count == 0 {
        return String::new();
    }

    let has_header = table.has_header
        || table
            .rows
            .first()
            .is_some_and(|row| row.is_header);

    let mut result = String::new();

    if has_header {
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
            append_markdown_row(&mut result, row, col_count);
        }
    } else {
        for row in &table.rows {
            append_markdown_row(&mut result, row, col_count);
        }
    }

    if result.ends_with('\n') {
        result.pop();
    }

    result
}

fn append_markdown_row(result: &mut String, row: &TableRow, col_count: usize) {
    result.push('|');
    for i in 0..col_count {
        let text = row.cells.get(i).map(markdown_cell_text).unwrap_or_default();
        result.push(' ');
        result.push_str(&text);
        result.push_str(" |");
    }
    result.push('\n');
}

fn markdown_cell_text(cell: &TableCell) -> String {
    cell.text.replace('\n', " ").replace('|', "\\|")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_cell(text: &str, is_header: bool) -> TableCell {
        TableCell::new(text.to_string(), is_header)
    }

    fn span_cell(text: &str, colspan: u32, rowspan: u32) -> TableCell {
        TableCell::new(text.to_string(), false)
            .with_colspan(colspan)
            .with_rowspan(rowspan)
    }

    fn text_row(cells: Vec<TableCell>, is_header: bool) -> TableRow {
        TableRow { cells, is_header }
    }

    #[test]
    fn expand_horizontal_span_repeats_value() {
        // | A (colspan 2) | B |
        let table = Table {
            rows: vec![text_row(
                vec![span_cell("A", 2, 1), text_cell("B", false)],
                false,
            )],
            ..Default::default()
        };
        let expanded = expand_table(&table);
        assert_eq!(expanded.rows.len(), 1);
        assert_eq!(expanded.rows[0].cells.len(), 3);
        assert_eq!(expanded.rows[0].cells[0].text, "A");
        assert_eq!(expanded.rows[0].cells[1].text, "A");
        assert_eq!(expanded.rows[0].cells[2].text, "B");
        assert!(expanded.rows[0]
            .cells
            .iter()
            .all(|c| c.colspan == 1 && c.rowspan == 1));
    }

    #[test]
    fn expand_vertical_span_repeats_value() {
        // col0: X rowspan 2; col1: Y / Z
        let table = Table {
            rows: vec![
                text_row(
                    vec![span_cell("X", 1, 2), text_cell("Y", false)],
                    false,
                ),
                text_row(vec![text_cell("Z", false)], false),
            ],
            ..Default::default()
        };
        let expanded = expand_table(&table);
        assert_eq!(expanded.rows.len(), 2);
        assert_eq!(expanded.rows[0].cells.len(), 2);
        assert_eq!(expanded.rows[1].cells.len(), 2);
        assert_eq!(expanded.rows[0].cells[0].text, "X");
        assert_eq!(expanded.rows[1].cells[0].text, "X");
        assert_eq!(expanded.rows[0].cells[1].text, "Y");
        assert_eq!(expanded.rows[1].cells[1].text, "Z");
    }

    #[test]
    fn expand_2x2_block_repeats_value() {
        let table = Table {
            rows: vec![
                text_row(
                    vec![span_cell("M", 2, 2), text_cell("R1", false)],
                    false,
                ),
                text_row(vec![text_cell("R2", false)], false),
            ],
            ..Default::default()
        };
        let expanded = expand_table(&table);
        assert_eq!(expanded.rows[0].cells.len(), 3);
        assert_eq!(expanded.rows[1].cells.len(), 3);
        assert_eq!(expanded.rows[0].cells[0].text, "M");
        assert_eq!(expanded.rows[0].cells[1].text, "M");
        assert_eq!(expanded.rows[1].cells[0].text, "M");
        assert_eq!(expanded.rows[1].cells[1].text, "M");
        assert_eq!(expanded.rows[0].cells[2].text, "R1");
        assert_eq!(expanded.rows[1].cells[2].text, "R2");
    }

    #[test]
    fn table_to_markdown_escapes_pipes_in_cell_text() {
        let table = Table {
            rows: vec![text_row(
                vec![text_cell("a|b", false), text_cell("c", false)],
                false,
            )],
            ..Default::default()
        };
        let md = table_to_markdown(&table);
        assert!(md.contains("| a\\|b | c |"), "got:\n{md}");
    }

    #[test]
    fn table_to_markdown_omits_separator_without_header_flags() {
        let table = Table {
            has_header: false,
            rows: vec![
                text_row(vec![text_cell("a", false), text_cell("b", false)], false),
                text_row(vec![text_cell("c", false), text_cell("d", false)], false),
            ],
            ..Default::default()
        };
        let md = table_to_markdown(&table);
        assert!(!md.contains("---"), "got:\n{md}");
        assert!(md.contains("| a | b |"), "got:\n{md}");
        assert!(md.contains("| c | d |"), "got:\n{md}");
    }

    #[test]
    fn table_to_markdown_includes_separator_when_first_row_is_header() {
        let table = Table {
            has_header: false,
            rows: vec![
                text_row(vec![text_cell("h1", true), text_cell("h2", true)], true),
                text_row(vec![text_cell("d1", false), text_cell("d2", false)], false),
            ],
            ..Default::default()
        };
        let md = table_to_markdown(&table);
        assert!(md.contains("---"), "got:\n{md}");
    }
}
