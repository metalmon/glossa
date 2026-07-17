use serde::{Deserialize, Serialize};

/// A single constraint on a field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Constraint {
    Range {
        min: f64,
        max: f64,
    },
    Regex {
        pattern: String,
    },
    Required,
    Forbidden,
    Enum {
        values: Vec<String>,
    },
    Formula {
        expression: String,
    },
    Conditional {
        condition_field: String,
        condition_value: serde_json::Value,
        inner: Box<Constraint>,
    },
}

/// All constraints for one named field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FieldConstraints {
    pub name: String,
    pub constraints: Vec<Constraint>,
}

/// A CSP problem: set of fields with their constraints.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Problem {
    pub fields: Vec<FieldConstraints>,
}

/// The inferred domain for a field after constraint propagation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "value")]
pub enum Domain {
    #[serde(rename = "interval")]
    Interval { min: f64, max: f64 },
    #[serde(rename = "set")]
    Set { values: Vec<String> },
    #[serde(rename = "any")]
    Any,
    #[serde(rename = "empty")]
    Empty,
    #[serde(rename = "regex")]
    Regex { pattern: String },
}

impl Domain {
    pub fn contains_string(&self, v: &str) -> bool {
        match self {
            Domain::Interval { min, max } => v.parse::<f64>().is_ok_and(|n| n >= *min && n <= *max),
            Domain::Set { values } => values.iter().any(|x| x == v),
            Domain::Any => true,
            Domain::Empty => false,
            Domain::Regex { pattern } => {
                regex_lite::Regex::new(pattern).is_ok_and(|re| re.is_match(v))
            }
        }
    }
}

/// A violation of one constraint by a field assignment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Violation {
    pub field: String,
    pub constraint: String,
    pub message: String,
    pub expected: String,
    pub actual: String,
}

/// A consistency issue in the constraint graph itself.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConsistencyIssue {
    pub field: String,
    pub severity: String,
    pub message: String,
}

/// Error type for the constraint solver.
#[derive(Debug, thiserror::Error)]
pub enum SolverError {
    #[error("field '{0}' not found in problem")]
    FieldNotFound(String),
    #[error("invalid literal value for {1}: {0}")]
    InvalidLiteral(String, String),
    #[error("regex error: {0}")]
    RegexError(String),
    #[error("expression error: {0}")]
    ExpressionError(String),
    #[error("{0}")]
    General(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_interval_contains_number() {
        let d = Domain::Interval {
            min: 0.0,
            max: 999.0,
        };
        assert!(d.contains_string("500"));
        assert!(!d.contains_string("-1"));
        assert!(!d.contains_string("1000"));
    }

    #[test]
    fn domain_set_contains_string() {
        let d = Domain::Set {
            values: vec!["A".into(), "B".into(), "C".into()],
        };
        assert!(d.contains_string("A"));
        assert!(d.contains_string("B"));
        assert!(!d.contains_string("D"));
    }

    #[test]
    fn domain_any_contains_everything() {
        let d = Domain::Any;
        assert!(d.contains_string("anything"));
    }

    #[test]
    fn domain_empty_contains_nothing() {
        let d = Domain::Empty;
        assert!(!d.contains_string("anything"));
    }

    #[test]
    fn domain_regex_matches() {
        let d = Domain::Regex {
            pattern: r"^RU-\d{6}$".into(),
        };
        assert!(d.contains_string("RU-123456"));
        assert!(!d.contains_string("RU-12345"));
        assert!(!d.contains_string("R2-123456"));
    }

    #[test]
    fn constraint_serialization_roundtrip() {
        let c = Constraint::Range {
            min: 0.0,
            max: 999.0,
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: Constraint = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }
}
