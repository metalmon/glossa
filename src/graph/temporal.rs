//! Valid-time date handling: normalize ISO-8601 (date granularities + strict
//! RFC3339 UTC) to a fixed-width, lexicographically-comparable string, and derive
//! per-node temporal status. Hand-rolled — no date crate. EDTF (`~`/`?`/unspecified)
//! is a Phase-4 concern and is rejected here.

use anyhow::{bail, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Future,
    Current,
    Expired,
    Superseded,
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => if is_leap(y) { 29 } else { 28 },
        _ => 0,
    }
}

/// Parse the leading Y / Y-M / Y-M-D; returns (year, month, day-or-None) or None
/// for a full datetime (handled separately).
fn parse_date(s: &str) -> Option<(i64, u32, Option<u32>)> {
    let parts: Vec<&str> = s.split('-').collect();
    let y: i64 = parts.first()?.parse().ok()?;
    if parts.len() == 1 {
        return Some((y, 0, None));
    }
    let m: u32 = parts.get(1)?.parse().ok()?;
    if !(1..=12).contains(&m) {
        return None;
    }
    if parts.len() == 2 {
        return Some((y, m, None));
    }
    let d: u32 = parts.get(2)?.parse().ok()?;
    if d < 1 || d > days_in_month(y, m) {
        return None;
    }
    Some((y, m, Some(d)))
}

fn edtf_marker(s: &str) -> bool {
    s.contains('~') || s.contains('?') || s.contains('X') || s.contains('x') || s.contains("..")
}

/// A strict `YYYY-MM-DDThh:mm:ssZ` datetime passes through unchanged if valid.
fn passthrough_datetime(s: &str) -> Option<String> {
    // exactly 20 chars: 2024-03-01T09:30:00Z
    if s.len() != 20 || !s.ends_with('Z') || s.as_bytes().get(10) != Some(&b'T') {
        return None;
    }
    let (date, rest) = s.split_at(10);
    parse_date(date)?; // validates the date part
    let hms = &rest[1..rest.len() - 1]; // strip 'T' and 'Z'
    let t: Vec<&str> = hms.split(':').collect();
    let (h, mi, se): (u32, u32, u32) = (t.first()?.parse().ok()?, t.get(1)?.parse().ok()?, t.get(2)?.parse().ok()?);
    if h > 23 || mi > 59 || se > 59 {
        return None;
    }
    Some(s.to_string())
}

fn normalize(s: &str, end: bool) -> Result<String> {
    let s = s.trim();
    if s.is_empty() {
        bail!("empty date");
    }
    if edtf_marker(s) {
        bail!("uncertain/EDTF dates ('{s}') are not supported yet (Phase 4)");
    }
    if let Some(dt) = passthrough_datetime(s) {
        return Ok(dt);
    }
    let (y, m, d) = parse_date(s).ok_or_else(|| anyhow::anyhow!("unparseable date: '{s}'"))?;
    let (mm, dd, time) = if m == 0 {
        (if end { 12 } else { 1 }, if end { 31 } else { 1 }, if end { "23:59:59" } else { "00:00:00" })
    } else if d.is_none() {
        (m, if end { days_in_month(y, m) } else { 1 }, if end { "23:59:59" } else { "00:00:00" })
    } else {
        (m, d.unwrap(), if end { "23:59:59" } else { "00:00:00" })
    };
    Ok(format!("{y:04}-{mm:02}-{dd:02}T{time}Z"))
}

pub fn normalize_from(s: &str) -> Result<String> {
    normalize(s, false)
}
pub fn normalize_to(s: &str) -> Result<String> {
    normalize(s, true)
}
pub fn normalize_point(s: &str) -> Result<String> {
    normalize(s, false)
}

/// Status of a node's interval against instant `at` (all normalized).
pub fn status(from: Option<&str>, to: Option<&str>, superseded: bool, at: &str) -> Status {
    if superseded {
        return Status::Superseded;
    }
    if let Some(f) = from {
        if f > at {
            return Status::Future;
        }
    }
    if let Some(t) = to {
        if t < at {
            return Status::Expired;
        }
    }
    Status::Current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_granularities_to_period_edges() {
        assert_eq!(normalize_from("2024").unwrap(), "2024-01-01T00:00:00Z");
        assert_eq!(normalize_to("2024").unwrap(), "2024-12-31T23:59:59Z");
        assert_eq!(normalize_from("2024-02").unwrap(), "2024-02-01T00:00:00Z");
        assert_eq!(normalize_to("2024-02").unwrap(), "2024-02-29T23:59:59Z"); // leap
        assert_eq!(normalize_to("2023-02").unwrap(), "2023-02-28T23:59:59Z");
        assert_eq!(normalize_from("2024-03-01").unwrap(), "2024-03-01T00:00:00Z");
        assert_eq!(normalize_to("2024-03-01").unwrap(), "2024-03-01T23:59:59Z");
        assert_eq!(normalize_from("2024-03-01T09:30:00Z").unwrap(), "2024-03-01T09:30:00Z");
        // ordering holds lexicographically
        assert!(normalize_from("2024-01-01").unwrap() < normalize_to("2024-12-31").unwrap());
    }

    #[test]
    fn rejects_edtf_and_garbage() {
        assert!(normalize_from("~2019").is_err());
        assert!(normalize_from("2019?").is_err());
        assert!(normalize_from("not-a-date").is_err());
        assert!(normalize_from("2024-13").is_err()); // bad month
    }

    #[test]
    fn status_derives() {
        let at = "2024-06-01T00:00:00Z";
        assert_eq!(status(Some("2025-01-01T00:00:00Z"), None, false, at), Status::Future);
        assert_eq!(status(Some("2024-01-01T00:00:00Z"), Some("2024-03-01T23:59:59Z"), false, at), Status::Expired);
        assert_eq!(status(Some("2024-01-01T00:00:00Z"), None, false, at), Status::Current);
        assert_eq!(status(Some("2024-01-01T00:00:00Z"), None, true, at), Status::Superseded);
    }
}
