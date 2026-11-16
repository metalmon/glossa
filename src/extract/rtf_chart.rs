//! Recover the data grid of a legacy MSGraph chart embedded in an RTF file.
//!
//! Many old `.doc` files are in fact RTF, and the charts in them are OLE1-embedded
//! `MSGraph.Chart.8` objects rather than OOXML chart parts. `rtf.rs`'s body walk skips the whole
//! `{\object …}` group as noise — correctly, since its `{\result …}` fallback is a picture — so
//! the numbers never reach the index. This module picks them back up by scanning the raw bytes
//! for `{\*\objdata …}`, which is independent of the body walk and cannot disturb it.
//!
//! The chain, each link proven against real Word bytes (see
//! `docs/superpowers/research/2026-11-16-msgraph-chart-data.md`):
//!
//! 1. `{\*\objdata <hex>}` decodes to an OLE1 native stream.
//! 2. Its header is `version(4) format(4)`, three length-prefixed null-terminated strings
//!    (class, topic, item), `native length(4)`, then the payload. The class gates us.
//! 3. The payload is a compound file holding a small `Workbook` stream — small enough to live in
//!    the mini stream, so a linear scan of the container cannot find it. `office_oxide`'s
//!    `CfbReader` already follows the mini-FAT, so that is borrowed rather than written.
//! 4. The grid is in that stream as `0x0004` LABEL and `0x0003` NUMBER records in the short,
//!    BIFF2-shaped layout — *not* the BIFF8 `0x0203`/`0x0204` that `office_oxide`'s own xls path
//!    looks for, which is why routing through that path yields an empty sheet and no error.
//!
//! `SERIES` and `BRAI` carry only counts and empty formulas; the values are in the cell records.
//! The BOF substream type is deliberately never consulted: a real Word file writes an
//! undocumented `dt` of `0x8000` where the documented value is `0x0020`, so every record from the
//! start of the stream is walked and cells are taken wherever they appear.

use crate::extract::office_table::table_to_markdown;
use crate::extract::ooxml_chart::{chart_to_table, ChartData, Series};
use crate::extract::{MAX_GRID_CELLS, MAX_REPEAT};
use crate::model::Chunk;
use office_oxide::cfb::{CfbReader, CFB_SIGNATURE};
use std::collections::HashMap;
use std::io::Cursor;
use std::path::Path;

/// The one embedded class this reader claims to understand.
const MSGRAPH_CLASS: &str = "MSGraph.Chart.8";

/// Cap on the `Workbook` stream a compound file may declare. The real Word file's is 3 KB and
/// the whole embedded object is 17 KB; this leaves four orders of magnitude of headroom while
/// still refusing a directory entry that claims gigabytes.
const MAX_WORKBOOK_BYTES: u64 = 32 * 1024 * 1024;

/// Every `{\*\objdata …}` payload in the file, hex-decoded.
///
/// Only the hex form is handled. An `\objdata` whose payload is a `\bin` binary run (legal RTF,
/// not what Word writes here) contains a backslash, which ends collection for that object and
/// leaves it out — the document still extracts, it just contributes no chart.
fn objdata_blobs(bytes: &[u8]) -> Vec<Vec<u8>> {
    const TAG: &[u8] = br"{\*\objdata";
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let Some(p) = find(&bytes[i..], TAG) else {
            break;
        };
        let start = i + p + TAG.len();
        let mut hex: Vec<u8> = Vec::new();
        let mut j = start;
        let mut hex_only = true;
        while j < bytes.len() {
            let b = bytes[j];
            if b == b'}' {
                break;
            } else if b.is_ascii_hexdigit() {
                hex.push(b);
            } else if !b.is_ascii_whitespace() {
                hex_only = false;
                break;
            }
            j += 1;
        }
        if hex_only && hex.len() >= 2 {
            out.push(decode_hex(&hex));
        }
        i = j.saturating_add(1);
    }
    out
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn decode_hex(h: &[u8]) -> Vec<u8> {
    fn nib(b: u8) -> u8 {
        match b {
            b'0'..=b'9' => b - b'0',
            b'a'..=b'f' => b - b'a' + 10,
            _ => b - b'A' + 10,
        }
    }
    h.as_chunks::<2>()
        .0
        .iter()
        .map(|p| (nib(p[0]) << 4) | nib(p[1]))
        .collect()
}

/// Split an OLE1 native stream into its class name and its native payload.
fn ole1_native(data: &[u8]) -> Option<(String, &[u8])> {
    let u32at = |p: usize| -> Option<usize> {
        let b: [u8; 4] = data.get(p..p.checked_add(4)?)?.try_into().ok()?;
        Some(u32::from_le_bytes(b) as usize)
    };
    // version(4) format(4)
    let mut p = 8usize;
    let mut class = String::new();
    for which in 0..3 {
        let len = u32at(p)?;
        p = p.checked_add(4)?;
        let raw = data.get(p..p.checked_add(len)?)?;
        p = p.checked_add(len)?;
        if which == 0 {
            // length-prefixed AND null-terminated: keep what precedes the terminator
            let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
            class = String::from_utf8_lossy(&raw[..end]).into_owned();
        }
    }
    let n = u32at(p)?;
    p = p.checked_add(4)?;
    Some((class, data.get(p..p.checked_add(n)?)?))
}

/// Read the `Workbook` stream out of an embedded compound file, or `None` for anything that is
/// not one.
///
/// Guarded by `catch_unwind` for the same reason as `read.rs`'s `legacy_ole_images_guarded`:
/// `CfbReader` runs a fragile container parser over arbitrary user bytes, and a panic in it
/// would otherwise abort the whole extraction rather than cost one chart. The declared stream
/// size is checked against `MAX_WORKBOOK_BYTES` *before* reading, so a header promising a huge
/// stream cannot make us materialize it.
fn workbook_stream(native: &[u8]) -> Option<Vec<u8>> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut cfb = CfbReader::new(Cursor::new(native)).ok()?;
        let idx = cfb.find_entry("Workbook")?;
        if cfb.entries().get(idx)?.stream_size > MAX_WORKBOOK_BYTES {
            return None;
        }
        cfb.read_stream_by_index(idx).ok()
    }))
    .ok()
    .flatten()
}

/// Walk `type(2) len(2) data` and read the datasheet cells, as `(row, col, text)`.
fn cells(stream: &[u8]) -> Vec<(u16, u16, String)> {
    let mut out = Vec::new();
    let mut p = 0usize;
    while p + 4 <= stream.len() {
        let code = u16::from_le_bytes([stream[p], stream[p + 1]]);
        let len = u16::from_le_bytes([stream[p + 2], stream[p + 3]]) as usize;
        let Some(d) = stream.get(p + 4..p + 4 + len) else {
            break;
        };
        p += 4 + len;
        // rw(2) col(2) attr(3) — the attribute block is 3 bytes here, not a 2-byte ixfe
        if d.len() < 7 {
            continue;
        }
        let rw = u16::from_le_bytes([d[0], d[1]]);
        let col = u16::from_le_bytes([d[2], d[3]]);
        match code {
            // LABEL: cch(1) grbit(1) then UTF-16LE when grbit bit 0 is set, else 8-bit
            0x0004 if d.len() >= 9 => {
                let cch = d[7] as usize;
                let body = &d[9..];
                let s = if (d[8] & 1) == 1 {
                    let n = cch.min(body.len() / 2);
                    let u: Vec<u16> = (0..n)
                        .map(|i| u16::from_le_bytes([body[2 * i], body[2 * i + 1]]))
                        .collect();
                    String::from_utf16_lossy(&u)
                } else {
                    String::from_utf8_lossy(&body[..cch.min(body.len())]).into_owned()
                };
                out.push((rw, col, s));
            }
            // NUMBER: an 8-byte little-endian double, record length 15
            0x0003 if d.len() >= 15 => {
                let v = f64::from_le_bytes([d[7], d[8], d[9], d[10], d[11], d[12], d[13], d[14]]);
                // Display on f64 drops a zero fraction, so 90.0 renders as "90", as the
                // datasheet shows it.
                out.push((rw, col, v.to_string()));
            }
            _ => {}
        }
    }
    out
}

/// Pivot the datasheet into the shape `ooxml_chart` already renders: row 0 is the category
/// header, column 0 the series names, the interior the values.
///
/// Bounded against the same hazard as `odf_chart::grid_range_values`, with the same two caps:
/// `rw` and `col` are `u16` fields read straight out of an untrusted file, so the grid's extents
/// are attacker-chosen. One 15-byte NUMBER record declaring `rw = 65535, col = 65535` would
/// otherwise pivot into ~4.3 billion strings — a stalled indexer from a few hundred bytes.
///
/// Where `odf_chart` truncates to the budget, this skips the chart outright, because the two
/// cases differ: an ODF range that overruns still addresses a real sheet, whereas here the
/// extents are *derived from the cells themselves*, so a span past the cap means a corrupt or
/// hostile record rather than a big datasheet — truncating would materialize a six-figure table
/// of empty strings around the handful of genuine cells. The document still extracts its text;
/// one warning names the file and the measured extents, so this is never silent.
///
/// Lookup is a map rather than a scan over `grid`: with a scan, every cell visited would cost
/// another pass over every record, so a clamp alone would still leave a quadratic.
fn grid_to_chart(grid: &[(u16, u16, String)], file_ctx: &Path) -> Option<ChartData> {
    let max_row = usize::from(grid.iter().map(|x| x.0).max()?);
    let max_col = usize::from(grid.iter().map(|x| x.1).max()?);
    let total = max_row.saturating_mul(max_col);
    if max_row > MAX_REPEAT || max_col > MAX_REPEAT || total > MAX_GRID_CELLS {
        tracing::warn!(
            "rtf {}: embedded chart datasheet spans rows 0..{max_row}, cols 0..{max_col} \
             ({total} cells, span cap {MAX_REPEAT}, cell cap {MAX_GRID_CELLS}); skipping it",
            file_ctx.display()
        );
        return None;
    }
    let cells: HashMap<(u16, u16), &str> = grid
        .iter()
        .map(|(r, c, v)| ((*r, *c), v.as_str()))
        .collect();
    // in range by construction: both bounds passed the MAX_REPEAT check, well under u16::MAX
    let at = |r: usize, c: usize| cells.get(&(r as u16, c as u16)).map(|s| s.to_string());
    let cats: Vec<String> = (1..=max_col)
        .map(|c| at(0, c).unwrap_or_default())
        .collect();
    let mut series = Vec::new();
    for r in 1..=max_row {
        let vals: Vec<String> = (1..=max_col)
            .map(|c| at(r, c).unwrap_or_default())
            .collect();
        // A row of nothing but gaps is a hole in a sparse datasheet, not a series.
        if vals.iter().all(String::is_empty) {
            continue;
        }
        series.push(Series {
            name: at(r, 0),
            cats: cats.clone(),
            vals,
        });
    }
    Some(ChartData {
        title: None,
        kind: String::new(),
        series,
    })
}

/// Render every legacy MSGraph chart embedded in `bytes` as a searchable chunk (header + GFM
/// data table). Infallible by design, exactly like `ooxml_chart::extract_charts`: anything
/// unrecognised — a non-hex `\objdata`, another embedded class, a truncated blob, a payload that
/// is not a compound file — skips that object and leaves the document's own text untouched.
pub fn extract_charts(path: &Path, bytes: &[u8]) -> Vec<Chunk> {
    let mut out = Vec::new();
    for blob in objdata_blobs(bytes) {
        let Some((class, native)) = ole1_native(&blob) else {
            continue;
        };
        if class != MSGRAPH_CLASS || !native.starts_with(&CFB_SIGNATURE) {
            continue;
        }
        let Some(stream) = workbook_stream(native) else {
            tracing::warn!(
                "embedded chart in {} has no readable Workbook stream",
                path.display()
            );
            continue;
        };
        let Some(cd) = grid_to_chart(&cells(&stream), path).filter(|cd| !cd.series.is_empty())
        else {
            tracing::warn!("embedded chart in {} had no datasheet", path.display());
            continue;
        };
        // No parent: chart parenting and chart ordinals are an open design question, and
        // `ooxml_chart` is flat for the same reason. `Chunk::labelled` until that is settled.
        let title = format!("Chart {}", out.len() + 1);
        let text = format!(
            "Chart: {}\n\n{}",
            title,
            table_to_markdown(&chart_to_table(&cd))
        );
        out.push(Chunk::labelled(
            path.to_path_buf(),
            "rtf".to_string(),
            title,
            text,
        ));
    }
    out
}
