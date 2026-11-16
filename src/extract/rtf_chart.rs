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
use crate::model::Chunk;
use office_oxide::cfb::{CfbReader, CFB_SIGNATURE};
use std::collections::HashMap;
use std::io::Cursor;
use std::path::Path;

/// The one embedded class this reader claims to understand.
const MSGRAPH_CLASS: &str = "MSGraph.Chart.8";

/// REJECTION bound on either extent of a legacy datasheet, in rows or columns.
///
/// Not a rendering limit and not a truncation budget: `rw`/`col` are `u16` fields read straight
/// out of an untrusted file, so a span past this is a corrupt or hostile record, and the chart
/// is dropped rather than clamped. **Do not raise it to let a bigger chart render** — that is
/// what `odf_chart`'s separate `MAX_GRID_CELLS` budget is for, and widening this one widens a
/// denial-of-service guard.
const MAX_CHART_GRID_SPAN: usize = 4096;

/// REJECTION bound on the cells one FILE may spend across every chart in it. Same standing as
/// `MAX_CHART_GRID_SPAN`: a denial-of-service guard, not a rendering knob, and **it must not be
/// raised to accommodate a chart someone wants to see**.
///
/// Per file, not per chart, and that distinction is the whole point: a per-item limit multiplies
/// straight through when a document carries many `\object` groups, so N charts each sitting just
/// under a per-chart cap would cost N times it. `extract_charts` spends this down as it goes and
/// stops when it runs out.
///
/// A format-legal `MSGraph.Chart.8` can declare 255 series of 4000 points — about 1.02 million
/// cells — and this drops it whole. That is the intended ruling, not an oversight: a datasheet
/// that size is not chart data any reader cites, and the alternative is emitting a
/// multi-megabyte table chunk into the index. The warning names the file and the measured
/// extents so a real occurrence is diagnosable rather than mysterious.
const MAX_CHART_GRID_CELLS: usize = 200_000;

/// REJECTION bound on how many charts one file may contribute, the count half of the per-file
/// budget (`MAX_CHART_GRID_CELLS` is the size half). A document with more embedded charts than
/// this is not something a reader cites, and without it a file could multiply the cell budget by
/// however many `\object` groups it declares.
pub(crate) const MAX_CHARTS_PER_FILE: usize = 64;

/// REJECTION bound on how empty a datasheet's declared extent may be, as a ratio of extent to
/// records actually present.
///
/// The shape it rejects: a handful of genuine cells plus one record with a far-away `rw`/`col`,
/// which inflates the extent without carrying data — a fabricated extent rather than a big
/// chart. The real fixtures are ~19 records in a 4x5 extent, a ratio near 1, so even a sparse
/// genuine datasheet clears this by a wide margin.
const MAX_GRID_SPARSITY: usize = 64;

/// Cap on how many `{\*\objdata …}` groups one file is scanned for.
///
/// The decoded bytes were already bounded by the file — the hex runs are disjoint, so their sum
/// cannot exceed it — but the *count* was not, and every blob costs a `Vec` allocation plus a
/// pass through the pipeline's gates before any chart budget can apply. A real document carries
/// a handful.
const MAX_OBJDATA_BLOBS: usize = 1024;

/// Cap on the `Workbook` stream a compound file may declare, checked before the stream is read.
///
/// What it really bounds is the *record walk's* transient allocation, not disk: `cells` runs
/// before any extent check and holds one `(u16, u16, String)` per cell-shaped record it finds.
/// The shortest such record is 13 bytes (a 4-byte header plus a LABEL's 9-byte minimum), so
/// this admits at most ~161k entries — below `MAX_CHART_GRID_CELLS`, which is the point: the two
/// bounds agree instead of the stream cap quietly admitting an order of magnitude more than the
/// grid bound allows. A real MSGraph datasheet stream is about 3 KB, so this is still ~700x
/// headroom.
const MAX_WORKBOOK_BYTES: u64 = 2 * 1024 * 1024;

/// Every `{\*\objdata …}` payload in the file, hex-decoded.
///
/// Only the hex form is handled. An `\objdata` whose payload is a `\bin` binary run (legal RTF,
/// not what Word writes here) contains a backslash, which ends collection for that object and
/// leaves it out — the document still extracts, it just contributes no chart.
///
/// Stops after `MAX_OBJDATA_BLOBS`, so a file declaring a million tiny groups cannot make the
/// caller walk a million pipeline entries before any chart budget gets a say.
fn objdata_blobs(bytes: &[u8]) -> Vec<Vec<u8>> {
    const TAG: &[u8] = br"{\*\objdata";
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() && out.len() < MAX_OBJDATA_BLOBS {
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
/// Bounded against the hazard `odf_chart::grid_range_values` also guards: `rw` and `col` are
/// `u16` fields read straight out of an untrusted file, so the grid's extents are
/// attacker-chosen. One 15-byte NUMBER record declaring `rw = 65535, col = 65535` would
/// otherwise pivot into ~4.3 billion strings — a stalled indexer from a few hundred bytes.
///
/// Where `odf_chart` truncates to a budget, this skips the chart outright, which is why the two
/// modules keep separate constants: an ODF range that overruns still addresses a real sheet,
/// whereas here the extents are *derived from the cells themselves*, so a span past the cap
/// means a corrupt or hostile record rather than a big datasheet — truncating would materialize
/// a six-figure table of empty strings around the handful of genuine cells. The document still
/// extracts its text; one warning names the file and the measured extents, so this is never
/// silent.
///
/// Lookup is a map rather than a scan over `grid`: with a scan, every cell visited would cost
/// another pass over every record, so a clamp alone would still leave a quadratic.
///
/// `cell_budget` is what is left of the FILE's budget, not a fresh per-chart allowance, and the
/// extent this chart costs comes back with it so the caller can spend it down.
fn grid_to_chart(
    grid: &[(u16, u16, String)],
    file_ctx: &Path,
    cell_budget: usize,
) -> Option<(ChartData, usize)> {
    let max_row = usize::from(grid.iter().map(|x| x.0).max()?);
    let max_col = usize::from(grid.iter().map(|x| x.1).max()?);
    let extent = (max_row + 1).saturating_mul(max_col + 1);
    let refused = if max_row > MAX_CHART_GRID_SPAN || max_col > MAX_CHART_GRID_SPAN {
        Some("an extent past the span bound")
    } else if extent > grid.len().saturating_mul(MAX_GRID_SPARSITY) {
        // A fabricated extent: a far-away `rw`/`col` inflating the grid around a few real cells.
        Some("an extent that is almost entirely holes")
    } else if extent > cell_budget {
        Some("no cell budget left in this file")
    } else {
        None
    };
    if let Some(why) = refused {
        tracing::warn!(
            "rtf {}: skipping an embedded chart with {why} — rows 0..{max_row}, cols 0..{max_col} \
             ({extent} cells) from {} records, span bound {MAX_CHART_GRID_SPAN}, sparsity bound \
             {MAX_GRID_SPARSITY}, {cell_budget} cells of budget left; the document still extracts",
            file_ctx.display(),
            grid.len()
        );
        return None;
    }
    let cells: HashMap<(u16, u16), &str> = grid
        .iter()
        .map(|(r, c, v)| ((*r, *c), v.as_str()))
        .collect();
    // in range by construction: both extents passed the span bound, well under u16::MAX
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
    Some((
        ChartData {
            title: None,
            kind: String::new(),
            series,
        },
        extent,
    ))
}

/// Render every legacy MSGraph chart embedded in `bytes` as a searchable chunk (header + GFM
/// data table). Infallible by design, exactly like `ooxml_chart::extract_charts`: anything
/// unrecognised — a non-hex `\objdata`, another embedded class, a truncated blob, a payload that
/// is not a compound file — skips that object and leaves the document's own text untouched.
///
/// The budgets are per FILE, spent down across the loop, because the per-chart bounds alone are
/// not bounds at all: a document may carry any number of `\object` groups, so N charts each
/// sitting just under a per-chart cap would cost N times it. Running out stops the loop — it is
/// never an error, and the chunks already built, plus the document's own text, still come back.
pub fn extract_charts(path: &Path, bytes: &[u8]) -> Vec<Chunk> {
    let mut out = Vec::new();
    let mut cell_budget = MAX_CHART_GRID_CELLS;
    let mut exhausted = None;
    for blob in objdata_blobs(bytes) {
        if out.len() >= MAX_CHARTS_PER_FILE {
            exhausted = Some("the per-file chart count");
            break;
        }
        if cell_budget == 0 {
            exhausted = Some("the per-file cell budget");
            break;
        }
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
        let Some((cd, extent)) = grid_to_chart(&cells(&stream), path, cell_budget)
            .filter(|(cd, _)| !cd.series.is_empty())
        else {
            tracing::warn!("embedded chart in {} had no datasheet", path.display());
            continue;
        };
        cell_budget = cell_budget.saturating_sub(extent);
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
    if let Some(which) = exhausted {
        tracing::warn!(
            "rtf {}: stopped extracting embedded charts — {which} ran out after {} chart(s) \
             (count bound {MAX_CHARTS_PER_FILE}, cell bound {MAX_CHART_GRID_CELLS}); the \
             document still extracts",
            path.display(),
            out.len()
        );
    }
    out
}
