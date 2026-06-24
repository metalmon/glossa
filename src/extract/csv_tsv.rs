use crate::extract::text::detect;
use crate::model::Chunk;
use encoding_rs::Encoding;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

const ROWS_PER_CHUNK: usize = 100;

/// Reads raw bytes line-by-line and decodes each line with a fixed encoding.
struct DecodingLines {
    inner: BufReader<File>,
    enc: &'static Encoding,
}

impl DecodingLines {
    fn next_line(&mut self) -> std::io::Result<Option<String>> {
        let mut raw = Vec::new();
        let n = self.inner.read_until(b'\n', &mut raw)?;
        if n == 0 {
            return Ok(None);
        }
        while raw.last() == Some(&b'\n') || raw.last() == Some(&b'\r') {
            raw.pop();
        }
        let (text, _, _) = self.enc.decode(&raw);
        Ok(Some(text.into_owned()))
    }
}

/// Stream a CSV/TSV file: first line is the header, repeated at the top of each chunk; subsequent
/// rows are grouped 100-per-chunk with location `rows A-B` (1-based data-row range).
pub fn stream(path: &Path, file_type: &str, sink: &mut dyn FnMut(Chunk)) -> anyhow::Result<()> {
    let mut head = vec![0u8; 64 * 1024];
    {
        let mut f = File::open(path)?;
        let n = f.read(&mut head)?;
        head.truncate(n);
    }
    let Some(enc) = detect(&head) else { return Ok(()) };

    let mut lines = DecodingLines { inner: BufReader::new(File::open(path)?), enc };
    let header = match lines.next_line()? {
        Some(h) => h,
        None => return Ok(()),
    };

    let mut buf = String::new();
    let mut count = 0usize;       // rows in the current chunk
    let mut data_row = 0usize;    // 1-based index of the last data row read
    let mut start = 1usize;       // first data row in the current chunk
    let emit = |buf: &mut String, start: usize, end: usize, sink: &mut dyn FnMut(Chunk)| {
        if buf.trim().is_empty() {
            return;
        }
        sink(Chunk {
            doc_path: path.to_path_buf(),
            location: format!("rows {start}-{end}"),
            file_type: file_type.to_string(),
            text: std::mem::take(buf),
        });
    };

    while let Some(line) = lines.next_line()? {
        data_row += 1;
        if count == 0 {
            buf.push_str(&header);
            buf.push('\n');
            start = data_row;
        }
        buf.push_str(&line);
        buf.push('\n');
        count += 1;
        if count >= ROWS_PER_CHUNK {
            emit(&mut buf, start, data_row, sink);
            count = 0;
        }
    }
    if count > 0 {
        emit(&mut buf, start, data_row, sink);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(out: &[Chunk]) -> usize {
        // data rows across all chunks = total lines minus one header line per chunk
        out.iter().map(|c| c.text.lines().count() - 1).sum()
    }

    #[test]
    fn groups_rows_with_header_repeated() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.csv");
        let mut body = String::from("name,age\n");
        for i in 0..150 {
            body.push_str(&format!("p{i},{i}\n"));
        }
        std::fs::write(&p, body).unwrap();
        let mut out = Vec::new();
        stream(&p, "csv", &mut |c| out.push(c)).unwrap();
        assert_eq!(out.len(), 2); // 100 + 50
        assert!(out[0].text.starts_with("name,age"));
        assert!(out[1].text.starts_with("name,age"));
        assert_eq!(out[0].location, "rows 1-100");
        assert_eq!(out[1].location, "rows 101-150");
        assert_eq!(rows(&out), 150);
    }

    #[test]
    fn tsv_is_handled() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.tsv");
        std::fs::write(&p, b"a\tb\n1\t2\n").unwrap();
        let mut out = Vec::new();
        stream(&p, "tsv", &mut |c| out.push(c)).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].text.contains("a\tb") && out[0].text.contains("1\t2"));
    }
}
