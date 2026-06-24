# Extractor Coverage Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop silently dropping readable files: add a catch-all text extractor with encoding detection, plus CSV/TSV and HTML extractors, on a constant-memory streaming extract→index pipeline (no size limit).

**Architecture:** `walk::walk_files` enumerates files and calls a per-file closure; `extract::extract_file(path, sink)` streams one file's chunks into a `&mut dyn FnMut(Chunk)` sink (binary doc formats — md/office/pdf — load whole via the existing `Extractor` trait; csv/tsv/html/unknown stream from the path). `index_dir` holds one tantivy writer + graph and indexes each chunk as it arrives, so memory is O(one chunk) regardless of file size. `collect_chunks` becomes a thin Vec-collecting wrapper for `read` and tests.

**Tech Stack:** Rust, tantivy (existing), `chardetng` (charset detection) + `encoding_rs` (decoding) — both pure-Rust.

## Global Constraints

- Pure Rust, offline, single binary. **C-free invariant:** `cargo tree -p glossa -i cc` must print nothing. The only new deps are `chardetng` and `encoding_rs` (both pure-Rust).
- File-First: **never drop a readable file**; only genuinely binary files (NUL byte or >10% control bytes in the 64 KiB prefix) are skipped.
- **No size limit:** the pipeline streams; never materialize all chunks of the tree, and never require a whole huge file's chunks at once during indexing.
- Follow the existing `Extractor` trait pattern for whole-file formats; new streaming extractors read from the path directly.
- TDD: failing test first, minimal code, pass, commit. Run tests with `cargo test -p glossa`.
- Chunk windowing: a new chunk every **100 lines OR 4000 chars, whichever first**; `location` = `""` for a single-window file, else `part.1`, `part.2`, … (1-based).

---

### Task 1: Add deps + encoding detection/decode (`extract::text` core)

**Files:**
- Modify: `Cargo.toml:14-35` (add two deps)
- Create: `src/extract/text.rs`
- Modify: `src/extract.rs:11-14` (add `pub mod text;`)

**Interfaces:**
- Produces: `text::detect(prefix: &[u8]) -> Option<&'static encoding_rs::Encoding>` (None = binary). `text::decode_all(bytes: &[u8]) -> Option<String>` (whole-buffer decode; None = binary).

- [ ] **Step 1: Add dependencies**

In `Cargo.toml` under `[dependencies]` (after the `base64`/`tokio` lines), add:

```toml
chardetng = "0.1"
encoding_rs = "0.8"
```

- [ ] **Step 2: Declare the module**

In `src/extract.rs`, add to the module list (currently `pub mod chunk; pub mod markdown; pub mod office; pub mod pdf;`):

```rust
pub mod text;
```

- [ ] **Step 3: Write failing tests**

Create `src/extract/text.rs`:

```rust
use encoding_rs::{Encoding, UTF_16BE, UTF_16LE, UTF_8};

/// Detect the charset of a text file from a prefix (first ~64 KiB). Returns None if the bytes look
/// binary (a NUL byte, or >10% C0 control bytes other than tab/newline/carriage-return).
pub fn detect(prefix: &[u8]) -> Option<&'static Encoding> {
    if prefix.is_empty() {
        return Some(UTF_8);
    }
    // BOM sniffing.
    if prefix.starts_with(&[0xEF, 0xBB, 0xBF]) {
        return Some(UTF_8);
    }
    if prefix.starts_with(&[0xFF, 0xFE]) {
        return Some(UTF_16LE);
    }
    if prefix.starts_with(&[0xFE, 0xFF]) {
        return Some(UTF_16BE);
    }
    // Binary sniff (only meaningful for non-UTF-16 content).
    let mut control = 0usize;
    for &b in prefix {
        if b == 0 {
            return None;
        }
        if b < 0x20 && b != b'\t' && b != b'\n' && b != b'\r' {
            control += 1;
        }
    }
    if control * 10 > prefix.len() {
        return None;
    }
    // Strict UTF-8 over the prefix wins; else guess (cp1251 / koi8-r / latin-*).
    if std::str::from_utf8(prefix).is_ok() {
        return Some(UTF_8);
    }
    let mut det = chardetng::EncodingDetector::new();
    det.feed(prefix, true);
    Some(det.guess(None, true))
}

/// Decode a whole buffer to UTF-8 text, or None if it looks binary.
pub fn decode_all(bytes: &[u8]) -> Option<String> {
    let enc = detect(bytes)?;
    let (text, _, _) = enc.decode(bytes);
    Some(text.into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_utf8_and_bom() {
        assert_eq!(decode_all("héllo".as_bytes()).unwrap(), "héllo");
        let mut bom = vec![0xEF, 0xBB, 0xBF];
        bom.extend_from_slice("hi".as_bytes());
        assert_eq!(decode_all(&bom).unwrap(), "hi");
    }

    #[test]
    fn decodes_utf16_le_and_be() {
        let le = [0xFF, 0xFE, b'h', 0x00, b'i', 0x00];
        assert_eq!(decode_all(&le).unwrap(), "hi");
        let be = [0xFE, 0xFF, 0x00, b'h', 0x00, b'i'];
        assert_eq!(decode_all(&be).unwrap(), "hi");
    }

    #[test]
    fn decodes_windows_1251_russian() {
        // "Привет" in Windows-1251.
        let cp1251 = [0xCF, 0xF0, 0xE8, 0xE2, 0xE5, 0xF2];
        assert_eq!(decode_all(&cp1251).unwrap(), "Привет");
    }

    #[test]
    fn binary_is_none() {
        assert!(decode_all(&[0x89, b'P', b'N', b'G', 0x00, 0x1A]).is_none()); // NUL present
        let mut ctrl = vec![0x01u8; 100];
        ctrl[50] = b'x';
        assert!(decode_all(&ctrl).is_none()); // mostly control
    }
}
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cargo test -p glossa --lib extract::text`
Expected: FAIL — until Step 1–3 land the crate won't compile (`chardetng`/`encoding_rs` unknown). After adding deps + module + this file, they PASS.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p glossa --lib extract::text`
Expected: PASS (4 tests).

- [ ] **Step 6: Verify C-free invariant**

Run: `cargo tree -p glossa -i cc`
Expected: prints nothing (no crate named `cc` in the dependency graph).

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock src/extract.rs src/extract/text.rs
git commit -m "feat(extract): encoding detection + decode (chardetng + encoding_rs)"
```

---

### Task 2: Text windowing (`text::Windower`)

**Files:**
- Modify: `src/extract/text.rs`

**Interfaces:**
- Consumes: `model::Chunk`.
- Produces: `text::Windower` with `fn new(path: &Path, file_type: &str) -> Self`, `fn push_line(&mut self, line: &str, sink: &mut dyn FnMut(Chunk))`, `fn finish(self, sink: &mut dyn FnMut(Chunk))`. Emits a chunk every 100 lines or 4000 chars; `location == ""` if the file produced exactly one window, else `part.1`, `part.2`, …

- [ ] **Step 1: Write failing tests**

Add to `src/extract/text.rs` (top, after imports add `use crate::model::Chunk; use std::path::{Path, PathBuf};`):

```rust
const MAX_LINES: usize = 100;
const MAX_CHARS: usize = 4000;

/// Accumulates lines into windowed chunks. Holds at most one finished window so it can label a
/// single-window file with an empty location and multi-window files with `part.N`.
pub struct Windower {
    path: PathBuf,
    file_type: String,
    buf: String,
    lines: usize,
    pending: Option<String>, // a completed window not yet emitted (awaiting "is there another?")
    emitted: usize,
}

impl Windower {
    pub fn new(path: &Path, file_type: &str) -> Self {
        Windower { path: path.to_path_buf(), file_type: file_type.to_string(), buf: String::new(), lines: 0, pending: None, emitted: 0 }
    }

    fn flush_pending(&mut self, sink: &mut dyn FnMut(Chunk)) {
        if let Some(text) = self.pending.take() {
            self.emitted += 1;
            sink(Chunk {
                doc_path: self.path.clone(),
                location: format!("part.{}", self.emitted),
                file_type: self.file_type.clone(),
                text,
            });
        }
    }

    fn close_window(&mut self, sink: &mut dyn FnMut(Chunk)) {
        if self.buf.trim().is_empty() {
            self.buf.clear();
            self.lines = 0;
            return;
        }
        self.flush_pending(sink); // a previous window exists -> we are multi-window
        self.pending = Some(std::mem::take(&mut self.buf));
        self.lines = 0;
    }

    pub fn push_line(&mut self, line: &str, sink: &mut dyn FnMut(Chunk)) {
        self.buf.push_str(line);
        self.buf.push('\n');
        self.lines += 1;
        if self.lines >= MAX_LINES || self.buf.chars().count() >= MAX_CHARS {
            self.close_window(sink);
        }
    }

    pub fn finish(mut self, sink: &mut dyn FnMut(Chunk)) {
        if !self.buf.trim().is_empty() {
            self.close_window(sink);
        }
        // Now emit the last pending window: location "" if it is the only one, else part.N.
        if let Some(text) = self.pending.take() {
            if self.emitted == 0 {
                sink(Chunk { doc_path: self.path.clone(), location: String::new(), file_type: self.file_type.clone(), text });
            } else {
                self.emitted += 1;
                sink(Chunk { doc_path: self.path, location: format!("part.{}", self.emitted), file_type: self.file_type, text });
            }
        }
    }
}

#[cfg(test)]
mod window_tests {
    use super::*;

    fn run(lines: &[&str]) -> Vec<Chunk> {
        let mut out = Vec::new();
        let mut w = Windower::new(Path::new("d.txt"), "txt");
        {
            let mut sink = |c: Chunk| out.push(c);
            for l in lines {
                w.push_line(l, &mut sink);
            }
            w.finish(&mut sink);
        }
        out
    }

    #[test]
    fn single_window_has_empty_location() {
        let out = run(&["alpha", "beta"]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].location, "");
        assert!(out[0].text.contains("alpha") && out[0].text.contains("beta"));
        assert_eq!(out[0].file_type, "txt");
    }

    #[test]
    fn splits_into_part_n_windows() {
        let many: Vec<&str> = (0..250).map(|_| "x").collect();
        let out = run(&many);
        assert_eq!(out.len(), 3); // 100 + 100 + 50 lines
        assert_eq!(out[0].location, "part.1");
        assert_eq!(out[1].location, "part.2");
        assert_eq!(out[2].location, "part.3");
        let total_lines: usize = out.iter().map(|c| c.text.lines().count()).sum();
        assert_eq!(total_lines, 250); // every line preserved
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p glossa --lib extract::text`
Expected: FAIL — `Windower` not yet defined (compile error) before adding the struct; PASS once the struct above is present.

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test -p glossa --lib extract::text`
Expected: PASS (now 6 tests in the file).

- [ ] **Step 4: Commit**

```bash
git add src/extract/text.rs
git commit -m "feat(extract): line windowing into part.N chunks"
```

---

### Task 3: Streaming text reader (`text::stream_text`)

**Files:**
- Modify: `src/extract/text.rs`

**Interfaces:**
- Consumes: `detect`, `Windower`.
- Produces: `text::stream_text(path: &Path, file_type: &str, sink: &mut dyn FnMut(Chunk)) -> anyhow::Result<()>` — streams the file, skipping it silently if binary.

- [ ] **Step 1: Write failing tests**

Add to `src/extract/text.rs` (add imports `use std::fs::File; use std::io::Read;`):

```rust
const PREFIX_BYTES: usize = 64 * 1024;
const READ_BLOCK: usize = 64 * 1024;

/// Stream-decode a text file (any detected encoding) into windowed chunks. Binary files are skipped.
pub fn stream_text(path: &Path, file_type: &str, sink: &mut dyn FnMut(Chunk)) -> anyhow::Result<()> {
    let mut file = File::open(path)?;
    let mut prefix = vec![0u8; PREFIX_BYTES];
    let n = read_fill(&mut file, &mut prefix)?;
    prefix.truncate(n);
    let Some(enc) = detect(&prefix) else { return Ok(()) }; // binary -> skip

    let mut decoder = enc.new_decoder();
    let mut win = Windower::new(path, file_type);
    let mut pending = String::new(); // partial trailing line across decode blocks
    let mut block = vec![0u8; READ_BLOCK];

    // Feed the already-read prefix, then the rest, then a final empty flush.
    let mut chunk_bytes: Vec<u8> = prefix;
    loop {
        let last = chunk_bytes.is_empty();
        let mut decoded = String::with_capacity(chunk_bytes.len() + 16);
        let _ = decoder.decode_to_string(&chunk_bytes, &mut decoded, last);
        pending.push_str(&decoded);
        // Drain complete lines.
        while let Some(nl) = pending.find('\n') {
            let line: String = pending.drain(..=nl).collect();
            win.push_line(line.trim_end_matches(['\n', '\r']), sink);
        }
        if last {
            break;
        }
        let m = read_fill(&mut file, &mut block)?;
        if m == 0 {
            chunk_bytes = Vec::new(); // triggers a final last=true pass
        } else {
            chunk_bytes = block[..m].to_vec();
        }
    }
    if !pending.is_empty() {
        win.push_line(pending.trim_end_matches(['\n', '\r']), sink);
    }
    win.finish(sink);
    Ok(())
}

/// Read repeatedly until `buf` is full or EOF; returns bytes read.
fn read_fill(file: &mut File, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut total = 0;
    while total < buf.len() {
        let n = file.read(&mut buf[total..])?;
        if n == 0 {
            break;
        }
        total += n;
    }
    Ok(total)
}

#[cfg(test)]
mod stream_tests {
    use super::*;

    #[test]
    fn streams_utf8_file_single_window() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.txt");
        std::fs::write(&p, b"line one\nline two\n").unwrap();
        let mut out = Vec::new();
        stream_text(&p, "txt", &mut |c| out.push(c)).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].text.contains("line one") && out[0].text.contains("line two"));
        assert_eq!(out[0].location, "");
    }

    #[test]
    fn streams_windows_1251_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("ru.txt");
        std::fs::write(&p, [0xCF, 0xF0, 0xE8, 0xE2, 0xE5, 0xF2, b'\n']).unwrap(); // "Привет\n"
        let mut out = Vec::new();
        stream_text(&p, "txt", &mut |c| out.push(c)).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].text.contains("Привет"));
    }

    #[test]
    fn binary_file_yields_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("b.bin");
        std::fs::write(&p, [0x00u8, 0x01, 0x02, 0x03]).unwrap();
        let mut out = Vec::new();
        stream_text(&p, "bin", &mut |c| out.push(c)).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn large_file_streams_all_lines_across_windows() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("big.txt");
        let body: String = (0..250).map(|i| format!("row {i}\n")).collect();
        std::fs::write(&p, body).unwrap();
        let mut out = Vec::new();
        stream_text(&p, "txt", &mut |c| out.push(c)).unwrap();
        assert!(out.len() >= 3);
        let total: usize = out.iter().map(|c| c.text.lines().count()).sum();
        assert_eq!(total, 250);
        assert_eq!(out[0].location, "part.1");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail then pass**

Run: `cargo test -p glossa --lib extract::text`
Expected: FAIL before the `stream_text` code is present (compile error), PASS after (now 10 tests).

- [ ] **Step 3: Commit**

```bash
git add src/extract/text.rs
git commit -m "feat(extract): streaming text reader with encoding detection"
```

---

### Task 4: CSV/TSV streaming extractor

**Files:**
- Create: `src/extract/csv_tsv.rs`
- Modify: `src/extract.rs` (add `pub mod csv_tsv;`)

**Interfaces:**
- Produces: `csv_tsv::stream(path: &Path, file_type: &str, sink: &mut dyn FnMut(Chunk)) -> anyhow::Result<()>` — header repeated atop each chunk; 100 data rows per chunk; `location = "rows A-B"`.

- [ ] **Step 1: Declare the module**

In `src/extract.rs` add: `pub mod csv_tsv;`

- [ ] **Step 2: Write the implementation + tests**

Create `src/extract/csv_tsv.rs`. We read raw bytes line-by-line (via `read_until(b'\n')`) and decode
each line with the detected encoding — no `encoding_rs_io` dependency:

```rust
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
    let mut emit = |buf: &mut String, start: usize, end: usize, sink: &mut dyn FnMut(Chunk)| {
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
```

Tests appended to the file:

```rust
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
```

- [ ] **Step 3: Run tests to verify they fail then pass**

Run: `cargo test -p glossa --lib extract::csv_tsv`
Expected: FAIL until the manual `DecodingLines` implementation replaces the stub, then PASS (2 tests).

- [ ] **Step 4: Verify C-free + no stray deps**

Run: `cargo tree -p glossa -i cc` (expected: empty) and confirm `encoding_rs_io` is NOT in `Cargo.toml`.

- [ ] **Step 5: Commit**

```bash
git add src/extract.rs src/extract/csv_tsv.rs
git commit -m "feat(extract): streaming CSV/TSV extractor"
```

---

### Task 5: HTML stripper + streaming extractor

**Files:**
- Create: `src/extract/html.rs`
- Modify: `src/extract.rs` (add `pub mod html;`)

**Interfaces:**
- Produces: `html::strip_html(input: &str) -> String` (pure); `html::stream(path, file_type, sink) -> anyhow::Result<()>`.

- [ ] **Step 1: Declare the module**

In `src/extract.rs` add: `pub mod html;`

- [ ] **Step 2: Write failing tests + implementation**

Create `src/extract/html.rs`:

```rust
use crate::extract::text::{decode_all, Windower};
use crate::model::Chunk;
use std::path::Path;

/// Strip tags from HTML: drop <script>/<style> bodies, remove all tags, decode common entities,
/// collapse runs of blank lines.
pub fn strip_html(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let lower = input.to_lowercase();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            // Skip <script>...</script> and <style>...</style> bodies entirely.
            for (tag, end) in [("<script", "</script>"), ("<style", "</style>")] {
                if lower[i..].starts_with(tag) {
                    if let Some(rel) = lower[i..].find(end) {
                        i += rel + end.len();
                    } else {
                        i = bytes.len();
                    }
                    out.push(' ');
                    break;
                }
            }
            // Skip a normal tag <...>.
            if i < bytes.len() && bytes[i] == b'<' {
                if let Some(rel) = input[i..].find('>') {
                    i += rel + 1;
                    out.push(' ');
                    continue;
                } else {
                    break;
                }
            }
            continue;
        }
        out.push(input[i..].chars().next().unwrap());
        i += input[i..].chars().next().unwrap().len_utf8();
    }
    let decoded = out
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ");
    // Collapse blank-line runs.
    let mut result = String::with_capacity(decoded.len());
    let mut blanks = 0;
    for line in decoded.lines() {
        if line.trim().is_empty() {
            blanks += 1;
            if blanks > 1 {
                continue;
            }
            result.push('\n');
        } else {
            blanks = 0;
            result.push_str(line.trim());
            result.push('\n');
        }
    }
    result
}

/// Read an HTML file, strip it, and window the text into chunks.
pub fn stream(path: &Path, file_type: &str, sink: &mut dyn FnMut(Chunk)) -> anyhow::Result<()> {
    let bytes = std::fs::read(path)?;
    let Some(text) = decode_all(&bytes) else { return Ok(()) };
    let stripped = strip_html(&text);
    let mut win = Windower::new(path, file_type);
    for line in stripped.lines() {
        win.push_line(line, sink);
    }
    win.finish(sink);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_tags_scripts_styles_and_entities() {
        let html = "<html><head><style>p{color:red}</style></head>\
                    <body><script>alert(1)</script><h1>Title</h1><p>Hello &amp; bye</p></body></html>";
        let s = strip_html(html);
        assert!(s.contains("Title"));
        assert!(s.contains("Hello & bye"));
        assert!(!s.contains("color:red"));
        assert!(!s.contains("alert"));
        assert!(!s.contains('<'));
    }

    #[test]
    fn stream_produces_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("d.html");
        std::fs::write(&p, b"<h1>Hi</h1><p>body text here</p>").unwrap();
        let mut out = Vec::new();
        stream(&p, "html", &mut |c| out.push(c)).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].text.contains("body text here"));
    }
}
```

- [ ] **Step 3: Run tests to verify they fail then pass**

Run: `cargo test -p glossa --lib extract::html`
Expected: FAIL before the file exists, PASS after (2 tests).

- [ ] **Step 4: Commit**

```bash
git add src/extract.rs src/extract/html.rs
git commit -m "feat(extract): HTML tag-stripping streaming extractor"
```

---

### Task 6: Markdown decode fix

**Files:**
- Modify: `src/extract/markdown.rs:13-16`

**Interfaces:**
- Consumes: `text::decode_all`.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `src/extract/markdown.rs`:

```rust
#[test]
fn decodes_non_utf8_markdown() {
    // "# Привет" header in Windows-1251 (0x23 0x20 then cp1251 bytes).
    let mut bytes = vec![b'#', b' '];
    bytes.extend_from_slice(&[0xCF, 0xF0, 0xE8, 0xE2, 0xE5, 0xF2]); // Привет
    bytes.push(b'\n');
    let chunks = MarkdownExtractor.extract(Path::new("ru.md"), &bytes).unwrap();
    assert_eq!(chunks.len(), 0); // header-only file: heading recorded, no body chunk
    // Now with a body line.
    let mut b2 = bytes.clone();
    b2.extend_from_slice("body\n".as_bytes());
    let c2 = MarkdownExtractor.extract(Path::new("ru.md"), &b2).unwrap();
    assert_eq!(c2.len(), 1);
    assert_eq!(c2[0].location, "Привет");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p glossa --lib extract::markdown::tests::decodes_non_utf8_markdown`
Expected: FAIL — `from_utf8_lossy` turns cp1251 bytes into replacement chars, so `location != "Привет"`.

- [ ] **Step 3: Implement**

Replace the body of `MarkdownExtractor::extract` (`src/extract/markdown.rs:13-16`):

```rust
    fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>> {
        let text = crate::extract::text::decode_all(bytes).unwrap_or_default();
        Ok(chunk_markdown(path, &text, "md"))
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p glossa --lib extract::markdown`
Expected: PASS (all markdown tests including the new one).

- [ ] **Step 5: Commit**

```bash
git add src/extract/markdown.rs
git commit -m "fix(extract): decode markdown with charset detection (was from_utf8_lossy)"
```

---

### Task 7: Per-file dispatch + walk enumeration

**Files:**
- Modify: `src/extract.rs` (add `extract_file`)
- Modify: `src/walk.rs` (add `walk_files`; rewrite `collect_chunks` as a wrapper; keep `extractors()` for the whole-file formats)

**Interfaces:**
- Produces: `extract::extract_file(path: &Path, sink: &mut dyn FnMut(Chunk)) -> anyhow::Result<()>`; `walk::walk_files(root, glob, respect_ignore, visit: &mut dyn FnMut(&Path) -> anyhow::Result<()>) -> anyhow::Result<()>`.
- Consumes: `MarkdownExtractor`, `OfficeExtractor`, `PdfExtractor` (whole-file `Extractor`s); `text::stream_text`, `csv_tsv::stream`, `html::stream`.

- [ ] **Step 1: Add `extract_file` to `src/extract.rs`**

Append to `src/extract.rs`:

```rust
use crate::model::Chunk;
use std::path::Path;

/// Extract one file's chunks into `sink`. Whole-file binary/doc formats (md/office/pdf) are read
/// fully; csv/tsv/html and any other readable file stream from the path (constant memory).
pub fn extract_file(path: &Path, sink: &mut dyn FnMut(Chunk)) -> anyhow::Result<()> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    for ex in crate::walk::extractors() {
        if ex.file_types().contains(&ext.as_str()) {
            let bytes = std::fs::read(path)?;
            for c in ex.extract(path, &bytes)? {
                sink(c);
            }
            return Ok(());
        }
    }
    match ext.as_str() {
        "csv" | "tsv" => csv_tsv::stream(path, &ext, sink),
        "html" | "htm" => html::stream(path, &ext, sink),
        other => {
            let ft = if other.is_empty() { "txt" } else { other };
            text::stream_text(path, ft, sink)
        }
    }
}
```

- [ ] **Step 2: Add `walk_files` and rewrite `collect_chunks` in `src/walk.rs`**

Replace the body of `collect_chunks` and add `walk_files`. `extractors()` stays unchanged (md/office/pdf only). New code:

```rust
/// Enumerate indexable files under `root` (gitignore-aware, skipping `.glossa`), calling `visit`
/// for each file path.
pub fn walk_files(
    root: &Path,
    glob: Option<&str>,
    respect_ignore: bool,
    visit: &mut dyn FnMut(&Path) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let matcher = match glob {
        Some(g) => Some(Glob::new(g)?.compile_matcher()),
        None => None,
    };
    let mut wb = WalkBuilder::new(root);
    wb.standard_filters(respect_ignore);
    wb.require_git(!respect_ignore);
    wb.filter_entry(|e| e.file_name() != ".glossa");
    for result in wb.build() {
        let entry = match result {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skip (walk error): {e}");
                continue;
            }
        };
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.path();
        if let Some(m) = &matcher {
            if !m.is_match(path) {
                continue;
            }
        }
        if let Err(e) = visit(path) {
            eprintln!("skip {}: {}", path.display(), e);
        }
    }
    Ok(())
}

/// Collect all chunks under `root` into a Vec (thin wrapper over the streaming pipeline; for `read`
/// and tests — `index_dir` streams instead).
pub fn collect_chunks(root: &Path, glob: Option<&str>, respect_ignore: bool) -> anyhow::Result<Vec<Chunk>> {
    let mut all = Vec::new();
    walk_files(root, glob, respect_ignore, &mut |path| {
        crate::extract::extract_file(path, &mut |c| all.push(c))
    })?;
    Ok(all)
}
```

Remove now-unused imports from `walk.rs` if the compiler flags them (the per-extractor loop moved into `extract_file`). Keep `use globset::Glob; use ignore::WalkBuilder;`.

- [ ] **Step 3: Write a test for catch-all coverage**

Add to `src/walk.rs` a `#[cfg(test)] mod cover_tests`:

```rust
#[cfg(test)]
mod cover_tests {
    use super::*;

    #[test]
    fn collect_indexes_text_json_code_skips_binary() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"plain text alpha").unwrap();
        std::fs::write(dir.path().join("b.json"), br#"{"key":"jsonvalue"}"#).unwrap();
        std::fs::write(dir.path().join("c.rs"), b"fn beta() {}").unwrap();
        std::fs::write(dir.path().join("d.png"), [0x89, b'P', 0x00, 0x01]).unwrap();
        let chunks = collect_chunks(dir.path(), None, false).unwrap();
        let joined: String = chunks.iter().map(|c| c.text.clone()).collect::<Vec<_>>().join("\n");
        assert!(joined.contains("alpha"));
        assert!(joined.contains("jsonvalue"));
        assert!(joined.contains("beta"));
        // the .png is binary -> no chunk from it
        assert!(chunks.iter().all(|c| c.file_type != "png"));
    }
}
```

- [ ] **Step 4: Run tests to verify they fail then pass**

Run: `cargo test -p glossa --lib walk`
Expected: PASS after `extract_file`/`walk_files` are in place. Also run the whole lib: `cargo test -p glossa` (existing `collect_chunks` callers still compile).

- [ ] **Step 5: Commit**

```bash
git add src/extract.rs src/walk.rs
git commit -m "feat(extract): per-file dispatch + walk_files; catch-all fallback"
```

---

### Task 8: Streaming `index_dir` (constant memory)

**Files:**
- Modify: `src/index/store.rs:143-191` (`index_dir`); add a streaming graph helper call.
- Modify: `src/graph/build.rs` (add `build_structural_doc` + `build_structural_section` OR reuse — see below).

**Interfaces:**
- Consumes: `walk::walk_files`, `extract::extract_file`, `DocIndex`, `GraphStore`, `Manifest`.
- Produces: `index_dir(dir, force)` streaming each chunk into a single writer + graph; same `IndexStats` semantics.

- [ ] **Step 1: Add per-chunk graph helpers to `src/graph/build.rs`**

Add (keep the existing `build_structural` for its test/back-compat):

```rust
/// Put the Document node for `path` (idempotent).
pub fn build_document(g: &GraphStore, path: &str, sig: FileSig) -> anyhow::Result<()> {
    let created_at = now_secs();
    g.put_node(&Node {
        id: path.to_string(),
        node_type: "Document".into(),
        label: path.to_string(),
        aliases: vec![],
        prov: Provenance { source_path: path.to_string(), range: None, file_sig: Some(sig), origin: "auto-structural".into(), confidence: 1.0, created_at },
    })
}

/// Put one Section node + CONTAINS edge for a chunk.
pub fn build_section(g: &GraphStore, chunk: &Chunk, sig: FileSig) -> anyhow::Result<()> {
    let path = chunk.doc_path.to_string_lossy().to_string();
    let created_at = now_secs();
    let prov = Provenance { source_path: path.clone(), range: Some(chunk.location.clone()), file_sig: Some(sig), origin: "auto-structural".into(), confidence: 1.0, created_at };
    let sec_id = format!("{path}#{}", chunk.location);
    g.put_node(&Node { id: sec_id.clone(), node_type: "Section".into(), label: chunk.location.clone(), aliases: vec![], prov: prov.clone() })?;
    g.put_edge(&Edge { from: path, to: sec_id, edge_type: "CONTAINS".into(), prov })
}
```

(`Provenance` must derive `Clone`; if it does not, construct it twice instead of cloning.)

- [ ] **Step 2: Rewrite `index_dir` to stream**

Replace `index_dir` (`src/index/store.rs:143-191`) with:

```rust
pub fn index_dir(dir: &Path, force: bool) -> anyhow::Result<IndexStats> {
    let idx = DocIndex::open_or_create(dir)?;
    let graph = crate::graph::store::GraphStore::open(dir)?;
    let manifest = if force { Manifest::default() } else { Manifest::load(dir) };

    let mut writer = idx.index.writer(50_000_000)?;
    let mut stats = IndexStats::default();
    let mut next = Manifest::default();

    eprintln!("indexing files under {}...", dir.display());
    crate::walk::walk_files(dir, None, true, &mut |path| {
        let path_str = path.to_string_lossy().to_string();
        let sig = match file_sig(path) {
            Ok(s) => s,
            Err(_) => return Ok(()),
        };
        next.files.insert(path_str.clone(), sig);
        if !force && !manifest.changed(&path_str, sig) {
            stats.unchanged += 1;
            return Ok(());
        }
        eprintln!("  + {path_str}");
        writer.delete_term(tantivy::Term::from_field_text(idx.fields.path, &path_str));
        graph.delete_by_source(&path_str)?;
        let mut doc_written = false;
        crate::extract::extract_file(path, &mut |c| {
            if !doc_written {
                let _ = crate::graph::build::build_document(&graph, &path_str, sig);
                doc_written = true;
            }
            let _ = writer.add_document(doc!(
                idx.fields.body => c.text.clone(),
                idx.fields.path => path_str.clone(),
                idx.fields.location => c.location.clone(),
                idx.fields.file_type => c.file_type.clone(),
            ));
            let _ = crate::graph::build::build_section(&graph, &c, sig);
        })?;
        stats.added += 1;
        Ok(())
    })?;

    for old_path in manifest.files.keys() {
        if !next.files.contains_key(old_path) {
            writer.delete_term(tantivy::Term::from_field_text(idx.fields.path, old_path.as_str()));
            graph.delete_by_source(old_path)?;
            stats.removed += 1;
        }
    }
    writer.commit()?;
    next.save(dir)?;
    Ok(stats)
}
```

Add `use tantivy::doc;` if not already imported at the top (it is, via line 10 `use tantivy::{doc, ...}`). Remove the now-unused `use crate::walk::collect_chunks;` import (line 113) and the `BTreeMap` grouping.

- [ ] **Step 3: Run the existing index tests**

Run: `cargo test -p glossa --lib index::store`
Expected: PASS — `index_dir_builds_structural_graph`, `index_dir_skips_malformed_pdf_and_continues`, `reindex_picks_up_changes_and_skips_unchanged` still hold (same observable behavior, now streamed).

- [ ] **Step 4: Run the whole lib test suite**

Run: `cargo test -p glossa`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/index/store.rs src/graph/build.rs
git commit -m "feat(index): streaming index_dir, constant memory per chunk (no size limit)"
```

---

### Task 9: `read_region` via `extract_file`

**Files:**
- Modify: `src/read.rs:1-48`

**Interfaces:**
- Consumes: `extract::extract_file`.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `src/read.rs`:

```rust
#[test]
fn reads_plain_txt_via_fallback() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("notes.txt");
    std::fs::write(&p, b"first line\nsecond line\n").unwrap();
    let out = read_region(&p, None).unwrap();
    assert!(out.contains("first line") && out.contains("second line"));
}

#[test]
fn reads_csv_via_fallback() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("t.csv");
    std::fs::write(&p, b"name,age\nbob,5\n").unwrap();
    let out = read_region(&p, None).unwrap();
    assert!(out.contains("name,age") && out.contains("bob,5"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p glossa --lib read::tests::reads_plain_txt_via_fallback`
Expected: FAIL — current `read_region` only consults `extractors()` (md/office/pdf), so a `.txt` yields empty chunks → empty string → assertion fails.

- [ ] **Step 3: Rewrite the chunk-collection part of `read_region`**

Replace lines `src/read.rs:13-20` (the `let bytes = ...` read + the `for ex in extractors()` loop) with:

```rust
    let mut chunks = Vec::new();
    crate::extract::extract_file(path, &mut |c| chunks.push(c))?;
```

Remove the now-unused `use crate::walk::extractors;` (line 1) and the `ext`/`bytes` locals if they become unused (the `ext` variable and the `std::fs::read` are no longer needed here; `extract_file` does its own read). Keep `use anyhow::Context;` only if still used — if not, drop it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p glossa --lib read`
Expected: PASS (existing read tests + the two new ones).

- [ ] **Step 5: Commit**

```bash
git add src/read.rs
git commit -m "feat(read): read any readable file via extract_file fallback"
```

---

### Task 10: End-to-end integration + docs

**Files:**
- Create/Modify: a CLI integration test (follow the existing `tests/` pattern, e.g. `tests/extractor_coverage.rs`)
- Modify: `docs/ROADMAP.md` (move the file-type item from backlog to "What works today")

**Interfaces:**
- Consumes: the `kb` binary (`assert_cmd`).

- [ ] **Step 1: Write the integration test**

Create `tests/extractor_coverage.rs`:

```rust
use assert_cmd::Command;
use predicates::str::contains;

#[test]
fn indexes_text_formats_and_searches_them() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("notes.txt"), b"alpha apricot avocado").unwrap();
    std::fs::write(dir.path().join("data.json"), br#"{"fruit":"bananaberry"}"#).unwrap();
    std::fs::write(dir.path().join("table.csv"), b"name,score\ncranberry,9\n").unwrap();
    std::fs::write(dir.path().join("page.html"), b"<h1>Heading</h1><p>damsonfruit</p>").unwrap();
    std::fs::write(dir.path().join("blob.png"), [0x89u8, b'P', b'N', b'G', 0x00, 0x01, 0x02]).unwrap();

    Command::cargo_bin("kb").unwrap().arg("index").arg(dir.path()).assert().success();

    for needle in ["apricot", "bananaberry", "cranberry", "damsonfruit"] {
        Command::cargo_bin("kb").unwrap()
            .arg("search").arg(needle).arg(dir.path())
            .assert().success().stdout(contains(needle).or(contains("notes").or(contains("data")).or(contains("table")).or(contains("page"))));
    }
}
```

(If the `kb search` CLI signature differs — check `src/main.rs` for the exact arg order — adjust the args to match; the assertion only needs the command to succeed and surface a matching document.)

- [ ] **Step 2: Run the integration test**

Run: `cargo test -p glossa --test extractor_coverage`
Expected: PASS — all four text formats are indexed and searchable; the binary `.png` is silently skipped.

- [ ] **Step 3: Verify C-free one more time**

Run: `cargo tree -p glossa -i cc`
Expected: empty.

- [ ] **Step 4: Update ROADMAP**

In `docs/ROADMAP.md`, under "What works today / Extraction", append: `txt/json/yaml/xml/toml/log/source (catch-all text, charset-detected via chardetng+encoding_rs), csv/tsv, html`. Remove (or mark DONE) the "File-type coverage (File-First gap)" backlog bullet, leaving its sub-items (structured JSON, heading-aware HTML, csv-crate, rtf/epub/eml) under a smaller "extractor backlog" note.

- [ ] **Step 5: Run the full workspace suite**

Run: `cargo test`
Expected: PASS (glossa lib + bin + eval).

- [ ] **Step 6: Commit**

```bash
git add tests/extractor_coverage.rs docs/ROADMAP.md
git commit -m "test(extract): e2e coverage for text/json/csv/html; docs"
```

---

## Notes for the implementer

- After all tasks, the session owner will stop the LM Studio MCP server (which holds `target/release/kb.exe`) and run `cargo build --release` to ship the new binary. Dev/test cycles use `target/debug` and do not conflict, so do not attempt the release build yourself.
- If `Provenance` does not derive `Clone`, construct it inline twice in `build_section` rather than cloning.
- Keep the binary doc-format path (md/office/pdf) on the existing `Extractor` trait; only csv/tsv/html/unknown stream from the path.
