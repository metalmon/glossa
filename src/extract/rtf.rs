//! RTF → `office_oxide::ir`, so an RTF file is chunked by the same chunker as docx and ODF
//! (`office_chunk::chunk_ir`) and gets the same heading path, table expansion and chunk
//! boundaries. `odf.rs` takes exactly this route; nothing here is RTF-specific downstream.
//!
//! Written rather than taken from a crate because the two things a crate would have been chosen
//! for are already here or absent from the crates: code-page decoding is `encoding_rs`, a direct
//! dependency with a working precedent in `extract/text.rs`, and the structure this module exists
//! to recover — a heading level per paragraph and `\trowd`/`\cell`/`\row` tables — is what the
//! surveyed crates flatten into text.
//!
//! **Two passes.** The first reads only the header groups: the document code page (`\ansicpg`),
//! the font table's per-font charsets (`\fcharset`), and the stylesheet's style→heading mapping.
//! The second walks the body with those in hand. Two simple scans beat one parser with modes, and
//! each is testable alone.
//!
//! **Degradation is deliberate and silent in one direction only.** An unknown control word is
//! ignored, an ignorable destination (`{\*\…}`) and the known noise destinations are skipped whole,
//! `\bin` skips its byte count, and an unbalanced brace ends the document rather than failing it —
//! a truncated file gives up the text it has. What is NOT silent is producing nothing: an RTF file
//! that yields no chunks is an error, same as the other extractors, because "indexed, empty" is the
//! failure mode that hides.

use crate::extract::office_chunk::chunk_ir;
use crate::extract::{Chunk, Extractor};
use anyhow::{anyhow, Result};
use encoding_rs::{Encoding, WINDOWS_1252};
use office_oxide::ir::{
    DocumentIR, Element, Heading, InlineContent, Paragraph, Section, Table, TableCell, TableRow,
    TextSpan,
};
use std::collections::HashMap;
use std::path::Path;

pub struct RtfExtractor;

/// Total table cells one document may contribute, across every table in it. Same budget and the
/// same reasoning as `odf.rs`'s `MAX_TABLE_CELLS`: `\cell` and `\row` cost a heap allocation each
/// and nothing in the format bounds how many a file may emit, so a malformed or hostile document
/// could otherwise amplify a few kilobytes of control words into unbounded memory — and
/// `office_table::expand_merged_tables` walks whatever comes out of here afterwards.
const MAX_TABLE_CELLS: usize = 1_000_000;

/// RTF's signature: `{\rtf` and a version digit. Used by this extractor and by the fallback that
/// rescues a file whose extension lies (`extract.rs`).
pub fn looks_like_rtf(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(512)];
    // A producer may emit a UTF-8 BOM or leading whitespace before the opening brace.
    let start = head
        .iter()
        .position(|b| !b.is_ascii_whitespace() && !matches!(b, 0xEF | 0xBB | 0xBF))
        .unwrap_or(head.len());
    head[start..].starts_with(b"{\\rtf")
}

// ---------------------------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
enum Tok<'a> {
    Open,
    Close,
    /// `\word` with an optional signed parameter. The word is ASCII by the format's definition.
    Control(&'a str, Option<i32>),
    /// `\X` for a non-alphabetic X — `\\`, `\{`, `\~`, `\*` and friends.
    Symbol(u8),
    /// `\'hh` — one byte in the current code page.
    Byte(u8),
    /// A run of plain bytes, still in the current code page.
    Text(&'a [u8]),
}

struct Lexer<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Lexer<'a> {
    fn new(b: &'a [u8]) -> Self {
        Lexer { b, i: 0 }
    }

    /// Raw byte skip for `\bin N`, which carries binary that is not RTF at all.
    fn skip_bytes(&mut self, n: usize) {
        self.i = self.i.saturating_add(n).min(self.b.len());
    }

    fn next_tok(&mut self) -> Option<Tok<'a>> {
        loop {
            let c = *self.b.get(self.i)?;
            match c {
                b'{' => {
                    self.i += 1;
                    return Some(Tok::Open);
                }
                b'}' => {
                    self.i += 1;
                    return Some(Tok::Close);
                }
                // Raw line breaks carry no meaning in RTF; the paragraph break is `\par`.
                b'\r' | b'\n' => {
                    self.i += 1;
                    continue;
                }
                b'\\' => {
                    self.i += 1;
                    // A trailing backslash at EOF: nothing follows, so nothing to emit.
                    let n = *self.b.get(self.i)?;
                    if n.is_ascii_alphabetic() {
                        let start = self.i;
                        while self.b.get(self.i).is_some_and(|x| x.is_ascii_alphabetic()) {
                            self.i += 1;
                        }
                        let word = std::str::from_utf8(&self.b[start..self.i]).unwrap_or("");
                        let mut param: Option<i32> = None;
                        let neg = self.b.get(self.i) == Some(&b'-');
                        if neg {
                            self.i += 1;
                        }
                        if self.b.get(self.i).is_some_and(|x| x.is_ascii_digit()) {
                            let ds = self.i;
                            while self.b.get(self.i).is_some_and(|x| x.is_ascii_digit()) {
                                self.i += 1;
                            }
                            // A parameter longer than i32 is malformed; treat it as absent rather
                            // than panicking on the parse.
                            let v: i64 = std::str::from_utf8(&self.b[ds..self.i])
                                .unwrap_or("")
                                .parse()
                                .unwrap_or(0);
                            let v = if neg { -v } else { v };
                            param = Some(v.clamp(i32::MIN as i64, i32::MAX as i64) as i32);
                        }
                        // One space after a control word is its delimiter and not text.
                        if self.b.get(self.i) == Some(&b' ') {
                            self.i += 1;
                        }
                        return Some(Tok::Control(word, param));
                    }
                    if n == b'\'' {
                        self.i += 1;
                        let hex = self.b.get(self.i..self.i + 2);
                        if let Some(h) = hex {
                            let s = std::str::from_utf8(h).unwrap_or("");
                            if let Ok(v) = u8::from_str_radix(s, 16) {
                                self.i += 2;
                                return Some(Tok::Byte(v));
                            }
                        }
                        // Malformed escape: consume the quote and carry on.
                        return Some(Tok::Symbol(b'\''));
                    }
                    self.i += 1;
                    return Some(Tok::Symbol(n));
                }
                _ => {
                    let start = self.i;
                    while self
                        .b
                        .get(self.i)
                        .is_some_and(|x| !matches!(x, b'{' | b'}' | b'\\' | b'\r' | b'\n'))
                    {
                        self.i += 1;
                    }
                    return Some(Tok::Text(&self.b[start..self.i]));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Pass 1: the header groups
// ---------------------------------------------------------------------------------------------

/// What the body walk needs to know before it starts: which code page bytes are in, and which
/// styles mean "heading".
#[derive(Debug, Default)]
struct Header {
    /// `\ansicpg`. `None` → `WINDOWS_1252`, which is what `\ansi` alone means.
    doc: Option<&'static Encoding>,
    /// Font index → code page, from the font table's `\fcharset`.
    fonts: HashMap<i32, &'static Encoding>,
    /// Style index → heading level (1-9).
    headings: HashMap<i32, u8>,
}

impl Header {
    fn doc_encoding(&self) -> &'static Encoding {
        self.doc.unwrap_or(WINDOWS_1252)
    }
}

/// `\ansicpgN` → an `encoding_rs` encoding. The label route rather than a hand-written table: the
/// numbers ARE the Windows code pages, and `encoding_rs` already knows them by that name.
fn encoding_for_codepage(cp: i32) -> Option<&'static Encoding> {
    match cp {
        // Not a Windows code page label, and the two RTF files that declare it mean UTF-8.
        65001 => Some(encoding_rs::UTF_8),
        10000 => Encoding::for_label(b"macintosh"),
        n if n > 0 => Encoding::for_label(format!("windows-{n}").as_bytes())
            .or_else(|| Encoding::for_label(format!("cp{n}").as_bytes())),
        _ => None,
    }
}

/// `\fcharsetN` → a code page. This mapping is the format's, not ours: the charset numbers are
/// Windows script ids and each names exactly one code page.
fn encoding_for_charset(charset: i32) -> Option<&'static Encoding> {
    let cp = match charset {
        0 => 1252,        // ANSI
        1 => return None, // "default" — the document's own page, decided by the caller
        2 => return None, // Symbol — not text in any page
        77 => 10000,      // Mac Roman
        128 => 932,       // Shift-JIS
        129 => 949,
        130 => 1361,
        134 => 936,
        136 => 950,
        161 => 1253,
        162 => 1254,
        163 => 1258,
        177 => 1255,
        178..=180 => 1256,
        181 | 182 => 1255,
        186 => 1257,
        204 => 1251,
        222 => 874,
        238 => 1250,
        254 => 437,
        255 => 850,
        _ => return None,
    };
    encoding_for_codepage(cp)
}

/// A style entry is a heading when it says so in a language-independent way: `\outlinelevelN`
/// (0-based, so level 0 is heading 1). The English name is accepted as a fallback, and nothing
/// else is — a producer that localises style names and omits the outline level yields no headings
/// rather than a guess, which is the honest failure here.
fn heading_level(outline: Option<i32>, name: &str) -> Option<u8> {
    if let Some(o) = outline {
        if (0..9).contains(&o) {
            return Some(o as u8 + 1);
        }
    }
    let lower = name.to_lowercase();
    let rest = lower
        .strip_prefix("heading")?
        .trim_start_matches([' ', '-']);
    let n: u8 = rest.trim().parse().ok()?;
    (1..=9).contains(&n).then_some(n)
}

/// Scan the header groups. Stops at the first body token, so a large document is not walked twice.
fn scan_header(bytes: &[u8]) -> Header {
    let mut h = Header::default();
    let mut lex = Lexer::new(bytes);
    // Depth of the group we are inside, and which header group it is.
    let mut depth: i32 = 0;
    let mut in_fonttbl: Option<i32> = None;
    let mut in_stylesheet: Option<i32> = None;
    // The entry being accumulated inside a font table or stylesheet group.
    let mut cur_index: Option<i32> = None;
    let mut cur_charset: Option<i32> = None;
    let mut cur_outline: Option<i32> = None;
    let mut cur_name = String::new();

    while let Some(t) = lex.next_tok() {
        match t {
            Tok::Open => depth += 1,
            Tok::Close => {
                // An entry ends with its group, for producers that omit the `;`.
                if in_fonttbl.is_some() {
                    if let (Some(i), Some(cs)) = (cur_index, cur_charset) {
                        if let Some(e) = encoding_for_charset(cs) {
                            h.fonts.insert(i, e);
                        }
                    }
                } else if in_stylesheet.is_some() {
                    if let Some(i) = cur_index {
                        if let Some(l) = heading_level(cur_outline, cur_name.trim()) {
                            h.headings.insert(i, l);
                        }
                    }
                }
                cur_index = None;
                cur_charset = None;
                cur_outline = None;
                cur_name.clear();
                if in_fonttbl == Some(depth) {
                    in_fonttbl = None;
                }
                if in_stylesheet == Some(depth) {
                    in_stylesheet = None;
                }
                depth -= 1;
                if depth <= 0 {
                    break;
                }
            }
            Tok::Control(w, p) => match w {
                "ansicpg" => h.doc = p.and_then(encoding_for_codepage),
                "fonttbl" => in_fonttbl = Some(depth),
                "stylesheet" => in_stylesheet = Some(depth),
                "f" if in_fonttbl.is_some() => cur_index = p,
                "fcharset" if in_fonttbl.is_some() => cur_charset = p,
                "s" if in_stylesheet.is_some() => cur_index = p,
                // `\sN` is absent on the stylesheet's Normal entry, which is style 0.
                "snext" | "sbasedon" if in_stylesheet.is_some() => {}
                "outlinelevel" if in_stylesheet.is_some() => cur_outline = p,
                // The body has started; the header groups are all behind us.
                "pard" | "par" | "sect" | "plain" => break,
                _ => {}
            },
            Tok::Text(t) if in_stylesheet.is_some() => {
                // A style entry's name is plain text terminated by `;`.
                let s = String::from_utf8_lossy(t);
                if let Some((before, _)) = s.split_once(';') {
                    cur_name.push_str(before);
                    if let Some(i) = cur_index {
                        if let Some(l) = heading_level(cur_outline, cur_name.trim()) {
                            h.headings.insert(i, l);
                        }
                    }
                    cur_index = None;
                    cur_outline = None;
                    cur_name.clear();
                } else {
                    cur_name.push_str(&s);
                }
            }
            _ => {}
        }
    }
    h
}

// ---------------------------------------------------------------------------------------------
// Pass 2: the body
// ---------------------------------------------------------------------------------------------

/// Group-scoped state. RTF's braces are the scope: a group inherits its parent's state and its
/// changes die with its closing brace, which is why this is a stack of copies.
#[derive(Clone)]
struct State {
    enc: &'static Encoding,
    /// `\sN` of the paragraph being built.
    style: Option<i32>,
    /// `\ucN` — how many character-equivalents follow a `\uN` as its substitute, to be dropped.
    uc: usize,
    /// Inside a destination whose text must not reach the body.
    skip: bool,
}

/// Destinations whose content is not document text. `\*` covers the ignorable ones a reader may
/// not understand; these are the named ones a reader DOES understand and still must not index.
fn is_noise_destination(w: &str) -> bool {
    matches!(
        w,
        "fonttbl"
            | "colortbl"
            | "stylesheet"
            | "info"
            | "pict"
            | "object"
            | "footnote"
            | "header"
            | "headerl"
            | "headerr"
            | "headerf"
            | "footer"
            | "footerl"
            | "footerr"
            | "footerf"
            | "filetbl"
            | "listtable"
            | "listoverridetable"
            | "revtbl"
            | "rsidtbl"
            | "generator"
            | "themedata"
            | "colorschememapping"
            | "latentstyles"
            | "datastore"
            | "xmlnstbl"
            | "pgdsctbl"
            | "xe"
            | "tc"
            | "bkmkstart"
            | "bkmkend"
            | "nonshppict"
    )
}

/// A control word that stands for one character of text.
fn literal_for(w: &str) -> Option<&'static str> {
    Some(match w {
        "tab" => "\t",
        "emdash" => "—",
        "endash" => "–",
        "emspace" | "enspace" | "qmspace" => " ",
        "lquote" => "\u{2018}",
        "rquote" => "\u{2019}",
        "ldblquote" => "\u{201c}",
        "rdblquote" => "\u{201d}",
        "bullet" => "•",
        "line" | "softline" => " ",
        _ => return None,
    })
}

/// The body walk: tokens in, `DocumentIR` out.
fn parse_to_ir(bytes: &[u8], header: &Header) -> DocumentIR {
    let mut lex = Lexer::new(bytes);
    let mut stack: Vec<State> = vec![State {
        enc: header.doc_encoding(),
        style: None,
        uc: 1,
        skip: false,
    }];
    let mut elements: Vec<Element> = Vec::new();

    // The paragraph under construction: decoded text plus the undecoded byte run feeding it.
    let mut text = String::new();
    let mut pending: Vec<u8> = Vec::new();
    let mut pending_enc: &'static Encoding = header.doc_encoding();
    // Character-equivalents still to drop after a `\uN`.
    let mut skip_chars: usize = 0;
    // Table accumulation.
    let mut in_table = false;
    let mut rows: Vec<TableRow> = Vec::new();
    let mut cells: Vec<TableCell> = Vec::new();
    // Cells accumulated across every table in the document, against `MAX_TABLE_CELLS`.
    let mut table_cells: usize = 0;
    let mut budget_warned = false;

    macro_rules! flush_bytes {
        () => {
            if !pending.is_empty() {
                let (s, _, _) = pending_enc.decode(&pending);
                text.push_str(&s);
                pending.clear();
            }
        };
    }

    macro_rules! end_paragraph {
        ($st:expr) => {{
            flush_bytes!();
            let body = std::mem::take(&mut text);
            let trimmed = body.trim();
            if !trimmed.is_empty() {
                let el = match $st.style.and_then(|s| header.headings.get(&s)).copied() {
                    Some(level) => Element::Heading(Heading {
                        level,
                        content: vec![InlineContent::Text(TextSpan {
                            text: trimmed.to_string(),
                            ..Default::default()
                        })],
                        ..Default::default()
                    }),
                    None => Element::Paragraph(Paragraph {
                        content: vec![InlineContent::Text(TextSpan {
                            text: trimmed.to_string(),
                            ..Default::default()
                        })],
                        ..Default::default()
                    }),
                };
                elements.push(el);
            }
        }};
    }

    macro_rules! close_table {
        () => {
            if !cells.is_empty() {
                rows.push(TableRow {
                    cells: std::mem::take(&mut cells),
                    ..Default::default()
                });
            }
            if !rows.is_empty() {
                elements.push(Element::Table(Table {
                    rows: std::mem::take(&mut rows),
                    ..Default::default()
                }));
            }
        };
    }

    while let Some(t) = lex.next_tok() {
        let cur = stack.last().cloned().unwrap_or(State {
            enc: header.doc_encoding(),
            style: None,
            uc: 1,
            skip: false,
        });
        match t {
            Tok::Open => stack.push(cur),
            Tok::Close => {
                flush_bytes!();
                if stack.len() > 1 {
                    stack.pop();
                }
            }
            Tok::Symbol(s) => {
                if skip_chars > 0 {
                    skip_chars -= 1;
                    continue;
                }
                match s {
                    b'*' => {
                        if let Some(st) = stack.last_mut() {
                            st.skip = true;
                        }
                    }
                    b'\\' | b'{' | b'}' if !cur.skip => {
                        flush_bytes!();
                        text.push(s as char);
                    }
                    b'~' if !cur.skip => {
                        flush_bytes!();
                        text.push('\u{a0}');
                    }
                    b'_' if !cur.skip => {
                        flush_bytes!();
                        text.push('-');
                    }
                    // `\` before a line break is a paragraph break.
                    b'\r' | b'\n' if !cur.skip => {
                        end_paragraph!(cur);
                    }
                    _ => {}
                }
            }
            Tok::Byte(b) => {
                if skip_chars > 0 {
                    skip_chars -= 1;
                    continue;
                }
                if !cur.skip {
                    // Buffered rather than decoded one byte at a time: a double-byte page needs
                    // both halves to produce one character.
                    if pending_enc != cur.enc {
                        flush_bytes!();
                        pending_enc = cur.enc;
                    }
                    pending.push(b);
                }
            }
            Tok::Text(run) => {
                let mut run = run;
                if skip_chars > 0 {
                    let drop = skip_chars.min(run.len());
                    skip_chars -= drop;
                    run = &run[drop..];
                }
                if !cur.skip && !run.is_empty() {
                    if pending_enc != cur.enc {
                        flush_bytes!();
                        pending_enc = cur.enc;
                    }
                    pending.extend_from_slice(run);
                }
            }
            Tok::Control(w, p) => {
                if skip_chars > 0 {
                    // A control word counts as one of the substitute characters to drop.
                    skip_chars -= 1;
                    if w == "bin" {
                        lex.skip_bytes(p.unwrap_or(0).max(0) as usize);
                    }
                    continue;
                }
                match w {
                    "u" => {
                        if !cur.skip {
                            flush_bytes!();
                            // The parameter is signed: a code point above 32767 arrives negative.
                            let raw = p.unwrap_or(0);
                            let cp = if raw < 0 { raw + 65536 } else { raw };
                            if let Some(c) = u32::try_from(cp).ok().and_then(char::from_u32) {
                                text.push(c);
                            }
                        }
                        skip_chars = cur.uc;
                    }
                    "uc" => {
                        if let Some(st) = stack.last_mut() {
                            st.uc = p.unwrap_or(1).max(0) as usize;
                        }
                    }
                    "ansicpg" => {
                        // Honoured here too: some producers emit it after the first group.
                        if let Some(e) = p.and_then(encoding_for_codepage) {
                            if let Some(st) = stack.last_mut() {
                                st.enc = e;
                            }
                        }
                    }
                    "f" => {
                        if let Some(e) = p.and_then(|i| header.fonts.get(&i)).copied() {
                            if let Some(st) = stack.last_mut() {
                                st.enc = e;
                            }
                        }
                    }
                    "s" => {
                        if let Some(st) = stack.last_mut() {
                            st.style = p;
                        }
                    }
                    "pard" => {
                        if let Some(st) = stack.last_mut() {
                            st.style = None;
                        }
                        // `\pard` resets paragraph properties, and being inside a table is one of
                        // them: it is how a producer says the table is over. A cell's own
                        // paragraphs re-assert it with `\pard\intbl`.
                        in_table = false;
                    }
                    "plain" => {}
                    "par" | "sect" => {
                        if !cur.skip {
                            if in_table {
                                // A paragraph break inside a cell is a line inside that cell.
                                flush_bytes!();
                                text.push(' ');
                            } else {
                                // A paragraph outside a table ends any table still open, so the
                                // table lands before it rather than swallowing it.
                                close_table!();
                                end_paragraph!(cur);
                            }
                        }
                    }
                    "intbl" | "trowd" => {
                        if !cur.skip {
                            if !in_table {
                                // Whatever preceded the table is its own paragraph.
                                end_paragraph!(cur);
                            }
                            in_table = true;
                        }
                    }
                    "cell" | "nestcell" => {
                        if !cur.skip && table_cells < MAX_TABLE_CELLS {
                            flush_bytes!();
                            table_cells += 1;
                            let body = std::mem::take(&mut text);
                            cells.push(TableCell {
                                content: vec![Element::Paragraph(Paragraph {
                                    content: vec![InlineContent::Text(TextSpan {
                                        text: body.trim().to_string(),
                                        ..Default::default()
                                    })],
                                    ..Default::default()
                                })],
                                ..Default::default()
                            });
                            in_table = true;
                        } else if !cur.skip && !budget_warned {
                            tracing::warn!(
                                "rtf: table exceeds {MAX_TABLE_CELLS}-cell budget, dropping further cells"
                            );
                            budget_warned = true;
                            // The text still has to go somewhere, or a dropped cell's content
                            // silently joins the next one.
                            flush_bytes!();
                            text.clear();
                        }
                    }
                    "row" | "nestrow" => {
                        if !cur.skip && table_cells < MAX_TABLE_CELLS {
                            flush_bytes!();
                            if !text.trim().is_empty() {
                                // A row whose last cell had no `\cell`: keep the text.
                                let body = std::mem::take(&mut text);
                                cells.push(TableCell {
                                    content: vec![Element::Paragraph(Paragraph {
                                        content: vec![InlineContent::Text(TextSpan {
                                            text: body.trim().to_string(),
                                            ..Default::default()
                                        })],
                                        ..Default::default()
                                    })],
                                    ..Default::default()
                                });
                                table_cells += 1;
                            }
                            text.clear();
                            // An empty row is not pushed: a file can emit `\row` forever without
                            // ever emitting a cell, and a million empty rows are still a million
                            // allocations that `expand_merged_tables` then walks.
                            if !cells.is_empty() {
                                rows.push(TableRow {
                                    cells: std::mem::take(&mut cells),
                                    ..Default::default()
                                });
                            }
                        }
                    }
                    "bin" => lex.skip_bytes(p.unwrap_or(0).max(0) as usize),
                    _ => {
                        if is_noise_destination(w) {
                            if let Some(st) = stack.last_mut() {
                                st.skip = true;
                            }
                        } else if !cur.skip {
                            if let Some(lit) = literal_for(w) {
                                flush_bytes!();
                                text.push_str(lit);
                            }
                        }
                    }
                }
            }
        }
    }

    // EOF: a truncated document gives up what it has rather than nothing.
    let last = stack.last().cloned().unwrap_or(State {
        enc: header.doc_encoding(),
        style: None,
        uc: 1,
        skip: false,
    });
    end_paragraph!(last);
    close_table!();

    DocumentIR {
        sections: vec![Section {
            elements,
            ..Default::default()
        }],
        ..Default::default()
    }
}

/// Parse `bytes` into the IR. Public so the mislabeled-extension fallback can reuse it without
/// going through `Extractor::extract`'s path handling.
pub fn rtf_to_ir(bytes: &[u8]) -> DocumentIR {
    let header = scan_header(bytes);
    parse_to_ir(bytes, &header)
}

impl Extractor for RtfExtractor {
    fn file_types(&self) -> &'static [&'static str] {
        &["rtf"]
    }

    fn extract(&self, path: &Path, bytes: &[u8]) -> Result<Vec<Chunk>> {
        if !looks_like_rtf(bytes) {
            return Err(anyhow!(
                "{} is named .rtf but does not start with {{\\rtf",
                path.display()
            ));
        }
        let mut ir = rtf_to_ir(bytes);
        crate::extract::office_table::expand_merged_tables(&mut ir);
        let chunks = chunk_ir(path, &ir, "rtf");
        if chunks.is_empty() {
            return Err(anyhow!("rtf produced no chunks for {}", path.display()));
        }
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_of(ir: &DocumentIR) -> Vec<String> {
        let mut out = Vec::new();
        for s in &ir.sections {
            for e in &s.elements {
                match e {
                    Element::Heading(h) => out.push(format!(
                        "h{}: {}",
                        h.level,
                        h.content
                            .iter()
                            .map(|c| match c {
                                InlineContent::Text(t) => t.text.clone(),
                                _ => String::new(),
                            })
                            .collect::<String>()
                    )),
                    Element::Paragraph(p) => out.push(
                        p.content
                            .iter()
                            .map(|c| match c {
                                InlineContent::Text(t) => t.text.clone(),
                                _ => String::new(),
                            })
                            .collect::<String>(),
                    ),
                    Element::Table(t) => out.push(format!(
                        "table {}x{}",
                        t.rows.len(),
                        t.rows.first().map(|r| r.cells.len()).unwrap_or(0)
                    )),
                    _ => {}
                }
            }
        }
        out
    }

    #[test]
    fn the_signature_is_recognised_past_a_bom_and_whitespace() {
        assert!(looks_like_rtf(b"{\\rtf1\\ansi test}"));
        assert!(looks_like_rtf(b"\xEF\xBB\xBF{\\rtf1"));
        assert!(looks_like_rtf(b"\r\n  {\\rtf1"));
        assert!(!looks_like_rtf(b"PK\x03\x04"));
        assert!(!looks_like_rtf(b""));
        // A `.doc` that really is a binary Word file must NOT be claimed as RTF.
        assert!(!looks_like_rtf(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"));
    }

    #[test]
    fn plain_paragraphs_come_out_as_paragraphs() {
        let rtf = b"{\\rtf1\\ansi\\pard First paragraph.\\par Second paragraph.\\par}";
        assert_eq!(
            text_of(&rtf_to_ir(rtf)),
            vec!["First paragraph.", "Second paragraph."]
        );
    }

    #[test]
    fn a_style_with_an_outline_level_becomes_a_heading() {
        // The stylesheet is where the level lives; the body only names the style.
        let rtf = b"{\\rtf1\\ansi{\\stylesheet{\\s1\\outlinelevel0 heading 1;}{\\s2\\outlinelevel1 heading 2;}}\
\\pard\\s1 Chapter\\par\\pard\\s2 Section\\par\\pard Body text\\par}";
        assert_eq!(
            text_of(&rtf_to_ir(rtf)),
            vec!["h1: Chapter", "h2: Section", "Body text"]
        );
    }

    #[test]
    fn an_english_style_name_is_the_fallback_when_no_outline_level_is_given() {
        let rtf = b"{\\rtf1\\ansi{\\stylesheet{\\s3 heading 3;}}\\pard\\s3 Titled\\par}";
        assert_eq!(text_of(&rtf_to_ir(rtf)), vec!["h3: Titled"]);
    }

    #[test]
    fn a_localised_style_name_without_an_outline_level_is_not_guessed_at() {
        // Degradation, stated: the paragraph is kept as a paragraph rather than promoted on a
        // hunch about what the name means.
        let rtf = "{\\rtf1\\ansi{\\stylesheet{\\s1 Kopfzeile 1;}}\\pard\\s1 Kapitel\\par}";
        assert_eq!(text_of(&rtf_to_ir(rtf.as_bytes())), vec!["Kapitel"]);
    }

    #[test]
    fn hex_escapes_decode_under_the_declared_code_page() {
        // cp1251 bytes for a short Russian word: the format carries the page, so nothing is
        // guessed. Decoded under the WRONG page this is mojibake, which is the failure this
        // extractor exists to avoid.
        let rtf = b"{\\rtf1\\ansi\\ansicpg1251\\pard \\'d2\\'e5\\'f1\\'f2\\par}";
        assert_eq!(text_of(&rtf_to_ir(rtf)), vec!["Тест"]);
    }

    #[test]
    fn a_font_charset_overrides_the_document_page() {
        // The document says 1252, the font says 204 (Cyrillic) — the font wins for its own runs.
        let rtf = b"{\\rtf1\\ansi\\ansicpg1252{\\fonttbl{\\f0\\fcharset0 Arial;}{\\f1\\fcharset204 Arial Cyr;}}\
\\pard\\f1 \\'d2\\'e5\\'f1\\'f2\\par}";
        assert_eq!(text_of(&rtf_to_ir(rtf)), vec!["Тест"]);
    }

    #[test]
    fn a_unicode_escape_wins_and_its_substitute_is_dropped() {
        // `ၘ` with `\uc1` is followed by one substitute byte that must not reach the text.
        let rtf = b"{\\rtf1\\ansi\\uc1\\pard \\u1058 ?\\u1077 ?\\u1089 ?\\u1090 ?\\par}";
        assert_eq!(text_of(&rtf_to_ir(rtf)), vec!["Тест"]);
    }

    #[test]
    fn a_negative_unicode_parameter_is_the_high_half_of_the_bmp() {
        // 0xFFFD arrives as -3 from producers that write the parameter signed.
        let rtf = b"{\\rtf1\\ansi\\uc0\\pard \\u-3 \\par}";
        assert_eq!(text_of(&rtf_to_ir(rtf)), vec!["\u{fffd}"]);
    }

    #[test]
    fn noise_destinations_and_ignorable_groups_contribute_no_text() {
        let rtf =
            b"{\\rtf1\\ansi{\\fonttbl{\\f0\\fcharset0 Arial;}}{\\colortbl;\\red0\\green0\\blue0;}\
{\\info{\\title Secret title}{\\author Someone}}{\\*\\generator Writer 1.0;}\
{\\header Page header}\\pard Body only\\par}";
        assert_eq!(text_of(&rtf_to_ir(rtf)), vec!["Body only"]);
    }

    #[test]
    fn a_table_becomes_a_table_with_its_cells() {
        let rtf = b"{\\rtf1\\ansi\\pard Before\\par\
\\trowd\\intbl A1\\cell A2\\cell\\row\
\\trowd\\intbl B1\\cell B2\\cell\\row\
\\pard After\\par}";
        assert_eq!(
            text_of(&rtf_to_ir(rtf)),
            vec!["Before", "table 2x2", "After"]
        );
    }

    #[test]
    fn a_paragraph_after_a_table_closes_it_instead_of_joining_it() {
        // `\pard` is the producer's way of saying the table is over. Without honouring it the
        // table stayed open to the end of the document and every later paragraph became cell text
        // — which is what the "value never read" lint on `in_table` was really pointing at.
        let rtf =
            br"{\rtf1\ansi\pard Before\par\trowd\intbl A1\cell\row\pard After\par\pard Last\par}";
        assert_eq!(
            text_of(&rtf_to_ir(rtf)),
            vec!["Before", "table 1x1", "After", "Last"]
        );
    }

    #[test]
    fn a_table_is_capped_and_empty_rows_cost_nothing() {
        // Nothing in RTF bounds how many `\cell`/`\row` a file may emit, and each costs a heap
        // allocation that `expand_merged_tables` then walks. A document of nothing but row markers
        // must not turn a few kilobytes into a table at all.
        let mut rtf = br"{\rtf1\ansi\trowd\intbl".to_vec();
        for _ in 0..50_000 {
            rtf.extend_from_slice(br"\row");
        }
        rtf.extend_from_slice(b"}");
        let ir = rtf_to_ir(&rtf);
        let cells: usize = ir
            .sections
            .iter()
            .flat_map(|s| &s.elements)
            .map(|e| match e {
                Element::Table(t) => t.rows.iter().map(|r| r.cells.len()).sum(),
                _ => 0usize,
            })
            .sum();
        assert_eq!(cells, 0, "empty rows must not become cells");
        assert!(
            text_of(&ir).iter().all(|s| !s.starts_with("table")),
            "a table of empty rows should not be emitted: {:?}",
            text_of(&ir)
        );
    }

    #[test]
    fn a_truncated_document_keeps_the_text_it_has() {
        // No closing brace, no final `\par`: the paragraph still arrives.
        let rtf = b"{\\rtf1\\ansi\\pard Half a sentence";
        assert_eq!(text_of(&rtf_to_ir(rtf)), vec!["Half a sentence"]);
    }

    #[test]
    fn bin_data_is_skipped_by_its_byte_count() {
        // The three bytes after `\bin3` are binary and must not be read as RTF — including the
        // backslash, which would otherwise start a control word.
        let rtf = b"{\\rtf1\\ansi\\pard Text\\bin3 \\{}more\\par}";
        assert_eq!(text_of(&rtf_to_ir(rtf)), vec!["Textmore"]);
    }

    #[test]
    fn escaped_braces_and_backslashes_are_literal_text() {
        let rtf = b"{\\rtf1\\ansi\\pard a \\{b\\} \\\\c\\par}";
        assert_eq!(text_of(&rtf_to_ir(rtf)), vec!["a {b} \\c"]);
    }

    #[test]
    fn control_word_literals_become_their_characters() {
        let rtf = b"{\\rtf1\\ansi\\pard a\\tab b\\emdash c\\lquote d\\par}";
        assert_eq!(text_of(&rtf_to_ir(rtf)), vec!["a\tb—c\u{2018}d"]);
    }

    #[test]
    fn the_extractor_refuses_a_file_that_only_has_the_extension() {
        let e = RtfExtractor
            .extract(Path::new("x.rtf"), b"not rtf at all")
            .unwrap_err()
            .to_string();
        assert!(e.contains("does not start with"), "{e}");
    }

    #[test]
    fn the_extractor_refuses_a_document_with_no_text() {
        let e = RtfExtractor
            .extract(
                Path::new("x.rtf"),
                b"{\\rtf1\\ansi{\\fonttbl{\\f0 Arial;}}}",
            )
            .unwrap_err()
            .to_string();
        assert!(e.contains("no chunks"), "{e}");
    }

    #[test]
    fn the_extractor_chunks_through_the_office_chunker() {
        // The point of building IR rather than text: `location` carries the heading path, exactly
        // as it does for docx and ODF.
        let rtf = b"{\\rtf1\\ansi{\\stylesheet{\\s1\\outlinelevel0 heading 1;}}\
\\pard\\s1 Chapter One\\par\\pard Body of the chapter.\\par}";
        let chunks = RtfExtractor.extract(Path::new("doc.rtf"), rtf).unwrap();
        assert!(!chunks.is_empty());
        assert!(
            chunks.iter().any(|c| c.location.contains("Chapter One")),
            "no heading path in {:?}",
            chunks.iter().map(|c| &c.location).collect::<Vec<_>>()
        );
        assert_eq!(chunks[0].file_type, "rtf");
    }
}
