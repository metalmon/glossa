//! Loose coercions for LLM tool-call JSON (bools as strings, etc.).

use serde::de::{self, Deserializer, Visitor};
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
}
