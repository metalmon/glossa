use encoding_rs::{Encoding, UTF_16BE, UTF_16LE, UTF_8};
use crate::model::Chunk;
use std::path::{Path, PathBuf};

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
