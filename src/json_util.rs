//! Loose coercions for LLM tool-call JSON (bools as strings, etc.).

use serde::de::{self, Deserializer, Visitor};
use serde::Deserialize;
use serde_json::Value;
use std::fmt;

/// Accept JSON `true`/`false`, or common model slips: `"true"`/`"false"`, `1`/`0`.
pub fn as_bool(v: &Value) -> Option<bool> {
    match v {
        Value::Bool(b) => Some(*b),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Some(true),
            "false" | "0" | "no" | "off" | "" => Some(false),
            _ => None,
        },
        Value::Number(n) => n
            .as_i64()
            .map(|i| i != 0)
            .or_else(|| n.as_u64().map(|u| u != 0))
            .or_else(|| n.as_f64().map(|f| f != 0.0)),
        _ => None,
    }
}

/// Serde helper for `Option<bool>` that also accepts string/number forms.
/// Use with `#[serde(default, deserialize_with = "json_util::deserialize_opt_bool_loose")]`.
pub fn deserialize_opt_bool_loose<'de, D>(deserializer: D) -> Result<Option<bool>, D::Error>
where
    D: Deserializer<'de>,
{
    struct OptBoolVisitor;

    impl<'de> Visitor<'de> for OptBoolVisitor {
        type Value = Option<bool>;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a boolean, or \"true\"/\"false\"/0/1")
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_bool<E: de::Error>(self, v: bool) -> Result<Self::Value, E> {
            Ok(Some(v))
        }

        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
            Ok(Some(v != 0))
        }

        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
            Ok(Some(v != 0))
        }

        fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
            Ok(Some(v != 0.0))
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            as_bool(&Value::String(v.to_string()))
                .ok_or_else(|| de::Error::custom(format!("invalid bool string {v:?}")))
                .map(Some)
        }
    }

    deserializer.deserialize_any(OptBoolVisitor)
}

/// Accept a JSON integer, or a numeric string (`"8"` → 8), or a whole-number float. Non-negative.
pub fn as_u64(v: &Value) -> Option<u64> {
    match v {
        Value::Number(n) => n.as_u64().or_else(|| {
            n.as_f64()
                .filter(|f| *f >= 0.0 && f.fract() == 0.0)
                .map(|f| f as u64)
        }),
        Value::String(s) => {
            let t = s.trim();
            if t.is_empty() {
                None
            } else {
                t.parse::<u64>().ok()
            }
        }
        _ => None,
    }
}

/// Serde helper for `Option<u64>` that also accepts numeric strings (`"8"` → 8).
/// Use with `#[serde(default, deserialize_with = "crate::json_util::deserialize_opt_u64_loose")]`.
pub fn deserialize_opt_u64_loose<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    struct OptU64Visitor;

    impl<'de> Visitor<'de> for OptU64Visitor {
        type Value = Option<u64>;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("an integer, or a numeric string like \"8\"")
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
            Ok(Some(v))
        }

        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
            u64::try_from(v)
                .map(Some)
                .map_err(|_| de::Error::custom(format!("expected a non-negative integer, got {v}")))
        }

        fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
            if v >= 0.0 && v.fract() == 0.0 {
                Ok(Some(v as u64))
            } else {
                Err(de::Error::custom(format!("expected a whole non-negative number, got {v}")))
            }
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            if v.trim().is_empty() {
                return Ok(None);
            }
            as_u64(&Value::String(v.to_string()))
                .ok_or_else(|| de::Error::custom(format!("invalid integer string {v:?}")))
                .map(Some)
        }
    }

    deserializer.deserialize_any(OptU64Visitor)
}

/// `Option<u32>` variant of [`deserialize_opt_u64_loose`] (range-checked).
pub fn deserialize_opt_u32_loose<'de, D>(deserializer: D) -> Result<Option<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    match deserialize_opt_u64_loose(deserializer)? {
        Some(v) => u32::try_from(v)
            .map(Some)
            .map_err(|_| de::Error::custom(format!("{v} is out of range for u32"))),
        None => Ok(None),
    }
}

/// `Option<usize>` variant of [`deserialize_opt_u64_loose`] (range-checked).
pub fn deserialize_opt_usize_loose<'de, D>(deserializer: D) -> Result<Option<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    match deserialize_opt_u64_loose(deserializer)? {
        Some(v) => usize::try_from(v)
            .map(Some)
            .map_err(|_| de::Error::custom(format!("{v} is out of range for usize"))),
        None => Ok(None),
    }
}

/// Normalize a path string by collapsing double backslashes (`\\` → `\`).
///
/// Handles MCP clients that double-escape backslashes in JSON: the JSON `\\\\`
/// (four chars) deserializes to `\\` (two chars), which this function collapses
/// back to `\` (one char) — the correct Windows path separator.
fn normalize_path_escapes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' && chars.peek() == Some(&'\\') {
            out.push('\\');
            chars.next(); // skip the second backslash
        } else {
            out.push(c);
        }
    }
    out
}

/// Serde helper for `String` path fields that normalizes double-escaped backslashes.
/// Use with `#[serde(deserialize_with = "crate::json_util::deserialize_str_loose")]`.
pub fn deserialize_str_loose<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    struct StrVisitor;

    impl<'de> Visitor<'de> for StrVisitor {
        type Value = String;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a string")
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(normalize_path_escapes(v))
        }

        fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
            Ok(normalize_path_escapes(&v))
        }
    }

    deserializer.deserialize_any(StrVisitor)
}

/// `Option<String>` variant — normalizes double-escaped backslashes; absent/empty → `None`.
pub fn deserialize_opt_str_loose<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    match Option::<String>::deserialize(deserializer)? {
        Some(s) if !s.is_empty() => Ok(Some(normalize_path_escapes(&s))),
        _ => Ok(None),
    }
}

/// Required-`u32` variant: accepts a JSON integer or numeric string; rejects null/empty.
/// Use with `#[serde(deserialize_with = "crate::json_util::deserialize_u32_loose")]` on a
/// non-optional field (absent → serde's normal "missing field" error).
pub fn deserialize_u32_loose<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: Deserializer<'de>,
{
    match deserialize_opt_u64_loose(deserializer)? {
        Some(v) => u32::try_from(v)
            .map_err(|_| de::Error::custom(format!("{v} is out of range for u32"))),
        None => Err(de::Error::custom("expected an integer")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serde_json::json;

    #[test]
    fn as_bool_accepts_string_and_number() {
        assert_eq!(as_bool(&json!(true)), Some(true));
        assert_eq!(as_bool(&json!("true")), Some(true));
        assert_eq!(as_bool(&json!("True")), Some(true));
        assert_eq!(as_bool(&json!("false")), Some(false));
        assert_eq!(as_bool(&json!(1)), Some(true));
        assert_eq!(as_bool(&json!(0)), Some(false));
        assert_eq!(as_bool(&json!("maybe")), None);
    }

    #[test]
    fn opt_bool_serde_from_string() {
        #[derive(Deserialize)]
        struct T {
            #[serde(default, deserialize_with = "deserialize_opt_bool_loose")]
            append: Option<bool>,
        }
        let t: T = serde_json::from_str(r#"{"append":"true"}"#).unwrap();
        assert_eq!(t.append, Some(true));
        let t: T = serde_json::from_str(r#"{"append":true}"#).unwrap();
        assert_eq!(t.append, Some(true));
        let t: T = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(t.append, None);
    }

    #[test]
    fn as_u64_accepts_string_and_number() {
        assert_eq!(as_u64(&json!(8)), Some(8));
        assert_eq!(as_u64(&json!("8")), Some(8));
        assert_eq!(as_u64(&json!(" 8 ")), Some(8));
        assert_eq!(as_u64(&json!(8.0)), Some(8));
        assert_eq!(as_u64(&json!(8.5)), None);
        assert_eq!(as_u64(&json!(-1)), None);
        assert_eq!(as_u64(&json!("nope")), None);
    }

    #[test]
    fn opt_and_required_int_serde_from_string() {
        #[derive(Deserialize)]
        struct T {
            #[serde(deserialize_with = "deserialize_u32_loose")]
            n: u32,
            #[serde(default, deserialize_with = "deserialize_opt_u64_loose")]
            max_bytes: Option<u64>,
            #[serde(default, deserialize_with = "deserialize_opt_usize_loose")]
            limit: Option<usize>,
        }
        // String-encoded primitives (the LLM/client slip this fix targets).
        let t: T = serde_json::from_str(r#"{"n":"8","max_bytes":"1024","limit":"5"}"#).unwrap();
        assert_eq!((t.n, t.max_bytes, t.limit), (8, Some(1024), Some(5)));
        // Native types still work; optionals absent → None.
        let t: T = serde_json::from_str(r#"{"n":3}"#).unwrap();
        assert_eq!((t.n, t.max_bytes, t.limit), (3, None, None));
        // Out-of-range u32 and a missing required field are rejected.
        assert!(serde_json::from_str::<T>(r#"{"n":"4294967296"}"#).is_err());
        assert!(serde_json::from_str::<T>(r#"{}"#).is_err());
    }

    #[test]
    fn str_loose_normalizes_double_backslashes() {
        #[derive(Deserialize)]
        struct T {
            #[serde(deserialize_with = "deserialize_str_loose")]
            path: String,
            #[serde(default, deserialize_with = "deserialize_opt_str_loose")]
            opt_path: Option<String>,
        }
        // Double-escaped backslashes → single
        let t: T =
            serde_json::from_str(r#"{"path":"a\\\\b\\\\c.pdf"}"#).unwrap();
        assert_eq!(t.path, r"a\b\c.pdf");
        // Single backslash passes through unchanged
        let t: T =
            serde_json::from_str(r#"{"path":"a\\b.pdf"}"#).unwrap();
        assert_eq!(t.path, r"a\b.pdf");
        // No backslashes at all
        let t: T =
            serde_json::from_str(r#"{"path":"hello.pdf"}"#).unwrap();
        assert_eq!(t.path, "hello.pdf");
        // Cyrillic + backslash
        let t: T = serde_json::from_str(
            r#"{"path":"Доп.данные\\Обновлённые руководства\\guide.pdf"}"#,
        )
        .unwrap();
        assert_eq!(t.path, "Доп.данные\\Обновлённые руководства\\guide.pdf");
        // Optional: present with double escapes
        let t: T =
            serde_json::from_str(r#"{"path":"x","opt_path":"a\\\\b"}"#).unwrap();
        assert_eq!(t.opt_path, Some(r"a\b".to_string()));
        // Optional: absent → None
        let t: T = serde_json::from_str(r#"{"path":"x"}"#).unwrap();
        assert_eq!(t.opt_path, None);
        // Optional: empty string → None
        let t: T =
            serde_json::from_str(r#"{"path":"x","opt_path":""}"#).unwrap();
        assert_eq!(t.opt_path, None);
    }
}
