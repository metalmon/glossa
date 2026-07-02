use crate::model::*;

/// The operation mode for the CSP solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SolveMode {
    /// Check field assignments against constraints.
    Validate,
    /// Infer each field's valid domain from constraints.
    Infer,
    /// Check internal consistency of the constraint set.
    Check,
}

/// Result of a constraint solve operation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct SolveResult {
    pub mode: String,
    pub valid: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub violations: Vec<Violation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub issues: Vec<ConsistencyIssue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domains: Option<Vec<FieldDomain>>,
}

/// Inferred domain for one field.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct FieldDomain {
    pub field: String,
    pub domain: Domain,
}

/// Value equality tolerant of the String/Number split: the graph stores literals
/// as strings while assignments may carry JSON numbers ("41" ≡ 41).
fn values_eq(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    use serde_json::Value;
    if a == b {
        return true;
    }
    let as_f64 = |v: &Value| -> Option<f64> {
        match v {
            Value::Number(n) => n.as_f64(),
            Value::String(s) => s.trim().parse().ok(),
            _ => None,
        }
    };
    match (as_f64(a), as_f64(b)) {
        (Some(x), Some(y)) => (x - y).abs() < f64::EPSILON,
        _ => match (a, b) {
            (Value::String(x), Value::String(y)) => x.trim() == y.trim(),
            _ => false,
        },
    }
}

/// Validate field assignments against constraints.
pub fn validate(problem: &Problem, assignment: &[(String, serde_json::Value)]) -> Vec<Violation> {
    let mut violations = Vec::new();
    let assign_map: std::collections::HashMap<&str, &serde_json::Value> =
        assignment.iter().map(|(k, v)| (k.as_str(), v)).collect();

    for fc in &problem.fields {
        for c in &fc.constraints {
            let applicable = match c {
                Constraint::Conditional { condition_field, condition_value, .. } => {
                    assign_map
                        .get(condition_field.as_str())
                        .is_some_and(|v| values_eq(v, condition_value))
                }
                _ => true,
            };
            if !applicable {
                continue;
            }

            let inner = match c {
                Constraint::Conditional { inner, .. } => inner,
                other => other,
            };

            match inner {
                Constraint::Required => {
                    if let Some(val) = assign_map.get(fc.name.as_str()) {
                        if val.is_null() || val.as_str().is_some_and(|s| s.is_empty()) {
                            violations.push(Violation {
                                field: fc.name.clone(),
                                constraint: "Required".into(),
                                message: format!("field '{}' is required but empty", fc.name),
                                expected: "non-empty value".into(),
                                actual: format!("{val}"),
                            });
                        }
                    } else {
                        violations.push(Violation {
                            field: fc.name.clone(),
                            constraint: "Required".into(),
                            message: format!("field '{}' is required but missing", fc.name),
                            expected: "value present".into(),
                            actual: "missing".into(),
                        });
                    }
                }
                Constraint::Forbidden => {
                    if let Some(val) = assign_map.get(fc.name.as_str()) {
                        if !val.is_null() {
                            violations.push(Violation {
                                field: fc.name.clone(),
                                constraint: "Forbidden".into(),
                                message: format!("field '{}' must be empty", fc.name),
                                expected: "empty".into(),
                                actual: format!("{val}"),
                            });
                        }
                    }
                }
                Constraint::Range { min, max } => {
                    if let Some(val) = assign_map.get(fc.name.as_str()) {
                        let n: f64 = match val {
                            serde_json::Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
                            serde_json::Value::String(s) => s.parse().unwrap_or(f64::NAN),
                            _ => f64::NAN,
                        };
                        if n.is_nan() || n < *min || n > *max {
                            violations.push(Violation {
                                field: fc.name.clone(),
                                constraint: "Range".into(),
                                message: format!(
                                    "field '{}' = {n} outside range [{}, {}]",
                                    fc.name, min, max
                                ),
                                expected: format!("{}..={}", min, max),
                                actual: format!("{n}"),
                            });
                        }
                    }
                }
                Constraint::Regex { pattern } => {
                    if let Some(val) = assign_map.get(fc.name.as_str()) {
                        let s = match val {
                            serde_json::Value::String(s) => s.clone(),
                            _ => format!("{val}"),
                        };
                        if regex_lite::Regex::new(pattern).is_ok_and(|re| !re.is_match(&s)) {
                            violations.push(Violation {
                                field: fc.name.clone(),
                                constraint: "Regex".into(),
                                message: format!(
                                    "field '{}' = '{s}' does not match pattern '{pattern}'",
                                    fc.name
                                ),
                                expected: format!("match '{pattern}'"),
                                actual: s,
                            });
                        }
                    }
                }
                Constraint::Enum { values } => {
                    if let Some(val) = assign_map.get(fc.name.as_str()) {
                        let matches = |v: &str| -> bool {
                            match val {
                                serde_json::Value::String(s) => v == s,
                                serde_json::Value::Number(n) => n
                                    .as_f64()
                                    .and_then(|a| v.parse::<f64>().ok().map(|b| (a, b)))
                                    .is_some_and(|(a, b)| (a - b).abs() < f64::EPSILON),
                                _ => false,
                            }
                        };
                        let s = format!("{val}");
                        if !values.iter().any(|v| matches(v)) {
                            violations.push(Violation {
                                field: fc.name.clone(),
                                constraint: "Enum".into(),
                                message: format!(
                                    "field '{}' = '{s}' not in {{ {} }}",
                                    fc.name,
                                    values.join(", ")
                                ),
                                expected: format!("one of: {}", values.join(", ")),
                                actual: s,
                            });
                        }
                    }
                }
                Constraint::Formula { expression } => {
                    if let Some(_val) = assign_map.get(fc.name.as_str()) {
                        let mut vars: std::collections::HashMap<String, f64> =
                            std::collections::HashMap::new();
                        for (k, v) in &assign_map {
                            if let Some(n) = v.as_f64() {
                                vars.insert(k.to_string(), n);
                            } else if let Some(s) = v.as_str() {
                                if let Ok(n) = s.parse::<f64>() {
                                    vars.insert(k.to_string(), n);
                                }
                            }
                        }
                        match crate::expr::eval_bool(expression, &vars) {
                            Ok(true) => {}
                            Ok(false) => {
                                violations.push(Violation {
                                    field: fc.name.clone(),
                                    constraint: "Formula".into(),
                                    message: format!(
                                        "formula '{}' evaluated to false",
                                        expression
                                    ),
                                    expected: "true".into(),
                                    actual: "false".into(),
                                });
                            }
                            Err(e) => {
                                violations.push(Violation {
                                    field: fc.name.clone(),
                                    constraint: "Formula".into(),
                                    message: format!("formula error: {e}"),
                                    expected: "valid expression".into(),
                                    actual: e,
                                });
                            }
                        }
                    }
                }
                _ => {} // Conditional handled above
            }
        }
    }

    violations
}

/// Infer the valid domain for each field from constraints, narrowed by any
/// partial `assignment`:
/// - a `Conditional` unfolds into its inner constraint when the condition field
///   is assigned and matches (unassigned/mismatching conditions contribute nothing);
/// - a `Formula` of the form `<field> = <expr>` pins the field to the computed
///   value once every variable on the right-hand side is assigned.
pub fn infer_domains(problem: &Problem, assignment: &[(String, serde_json::Value)]) -> Vec<FieldDomain> {
    let assign_map: std::collections::HashMap<&str, &serde_json::Value> =
        assignment.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let mut vars: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for (k, v) in assignment {
        let n = match v {
            serde_json::Value::Number(n) => n.as_f64(),
            serde_json::Value::String(s) => s.trim().parse().ok(),
            _ => None,
        };
        if let Some(n) = n {
            vars.insert(k.clone(), n);
        }
    }

    problem
        .fields
        .iter()
        .map(|fc| {
            // Unfold conditionals whose condition currently holds.
            let effective: Vec<&Constraint> = fc
                .constraints
                .iter()
                .filter_map(|c| match c {
                    Constraint::Conditional { condition_field, condition_value, inner } => assign_map
                        .get(condition_field.as_str())
                        .is_some_and(|v| values_eq(v, condition_value))
                        .then(|| inner.as_ref()),
                    other => Some(other),
                })
                .collect();

            let mut domain = intersect_domains(&effective);

            // A resolvable formula pins the field to a single value.
            for c in &effective {
                if let Constraint::Formula { expression } = c {
                    if let Some(v) = formula_pinned_value(&fc.name, expression, &vars) {
                        domain = intersect_domain(domain, Domain::Set { values: vec![fmt_num(v)] });
                    }
                }
            }

            FieldDomain {
                field: fc.name.clone(),
                domain,
            }
        })
        .collect()
}

/// For a formula shaped `<field> = <expr>` (a single `=` that is not part of a
/// comparison operator), evaluate `<expr>` when all its variables are assigned.
/// `expr::eval` is this crate's own arithmetic parser over f64 variables — no
/// code execution.
fn formula_pinned_value(
    field: &str,
    expression: &str,
    vars: &std::collections::HashMap<String, f64>,
) -> Option<f64> {
    let bytes = expression.as_bytes();
    let mut eq_pos: Option<usize> = None;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'=' {
            let prev = i.checked_sub(1).map(|p| bytes[p]);
            let next = bytes.get(i + 1);
            if matches!(prev, Some(b'<') | Some(b'>') | Some(b'!') | Some(b'=')) || next == Some(&b'=') {
                continue; // part of ==, <=, >=, !=
            }
            if eq_pos.is_some() {
                return None; // more than one bare '=' — not a simple definition
            }
            eq_pos = Some(i);
        }
    }
    let pos = eq_pos?;
    if expression[..pos].trim() != field {
        return None;
    }
    crate::expr::eval(expression[pos + 1..].trim(), vars).ok()
}

fn fmt_num(n: f64) -> String {
    if n.fract() == 0.0 && n.is_finite() {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

fn intersect_domains(constraints: &[&Constraint]) -> Domain {
    let mut domains: Vec<Domain> = Vec::new();
    for c in constraints {
        match c {
            Constraint::Range { min, max } => {
                domains.push(Domain::Interval { min: *min, max: *max });
            }
            Constraint::Regex { pattern } => {
                domains.push(Domain::Regex {
                    pattern: pattern.clone(),
                });
            }
            Constraint::Enum { values } => {
                domains.push(Domain::Set {
                    values: values.clone(),
                });
            }
            Constraint::Required => {} // doesn't constrain domain
            Constraint::Forbidden => {
                domains.push(Domain::Empty);
            }
            // Formula and Conditional are handled by infer_domains: conditionals
            // unfold before this call, resolvable formulas pin the value after it.
            _ => {}
        }
    }

    if domains.is_empty() {
        return Domain::Any;
    }

    let mut result = domains.remove(0);
    for d in domains {
        result = intersect_domain(result, d);
        if result == Domain::Empty {
            break;
        }
    }
    result
}

fn intersect_domain(a: Domain, b: Domain) -> Domain {
    use Domain::*;
    match (&a, &b) {
        (Any, _) => b,
        (_, Any) => a,
        (Empty, _) | (_, Empty) => Empty,
        (Interval { min: min1, max: max1 }, Interval { min: min2, max: max2 }) => {
            let min = min1.max(*min2);
            let max = max1.min(*max2);
            if min > max {
                Empty
            } else {
                Interval { min, max }
            }
        }
        (Set { values: v1 }, Set { values: v2 }) => {
            let intersection: Vec<String> =
                v1.iter().filter(|x| v2.contains(x)).cloned().collect();
            if intersection.is_empty() {
                Empty
            } else {
                Set { values: intersection }
            }
        }
        (Set { values }, Interval { min, max }) => {
            let filtered: Vec<String> = values
                .iter()
                .filter(|v| v.parse::<f64>().is_ok_and(|n| n >= *min && n <= *max))
                .cloned()
                .collect();
            if filtered.is_empty() {
                Empty
            } else {
                Set { values: filtered }
            }
        }
        (Interval { min, max }, Set { values }) => {
            let filtered: Vec<String> = values
                .iter()
                .filter(|v| v.parse::<f64>().is_ok_and(|n| n >= *min && n <= *max))
                .cloned()
                .collect();
            if filtered.is_empty() {
                Empty
            } else {
                Set { values: filtered }
            }
        }
        _ => a,
    }
}

/// Check internal consistency of the constraint graph.
pub fn check_consistency(problem: &Problem) -> Vec<ConsistencyIssue> {
    let mut issues = Vec::new();
    for fc in &problem.fields {
        for c in &fc.constraints {
            match c {
                Constraint::Range { min, max } => {
                    if min > max {
                        issues.push(ConsistencyIssue {
                            field: fc.name.clone(),
                            severity: "error".into(),
                            message: format!(
                                "Range min ({}) > max ({}) — no possible value",
                                min, max
                            ),
                        });
                    }
                }
                Constraint::Regex { pattern } => {
                    if regex_lite::Regex::new(pattern).is_err() {
                        issues.push(ConsistencyIssue {
                            field: fc.name.clone(),
                            severity: "error".into(),
                            message: format!("Invalid regex pattern: '{}'", pattern),
                        });
                    }
                }
                Constraint::Enum { values } => {
                    if values.is_empty() {
                        issues.push(ConsistencyIssue {
                            field: fc.name.clone(),
                            severity: "error".into(),
                            message: "Enum has no values — no possible value".into(),
                        });
                    }
                }
                _ => {}
            }
        }
        // Check conflicting constraints
        let has_forbidden = fc.constraints.iter().any(|c| matches!(c, Constraint::Forbidden));
        let has_required = fc.constraints.iter().any(|c| matches!(c, Constraint::Required));
        if has_forbidden && has_required {
            issues.push(ConsistencyIssue {
                field: fc.name.clone(),
                severity: "error".into(),
                message: "Conflicting constraints: Required + Forbidden on the same field".into(),
            });
        }
    }
    issues
}

/// Main entry point: dispatch based on mode.
pub fn solve(problem: &Problem, mode: SolveMode, assignment: &[(String, serde_json::Value)]) -> SolveResult {
    match mode {
        SolveMode::Validate => {
            let violations = validate(problem, assignment);
            SolveResult {
                mode: "validate".into(),
                valid: violations.is_empty(),
                violations,
                issues: vec![],
                domains: None,
            }
        }
        SolveMode::Infer => {
            let domains = infer_domains(problem, assignment);
            SolveResult {
                mode: "infer".into(),
                valid: true,
                violations: vec![],
                issues: vec![],
                domains: Some(domains),
            }
        }
        SolveMode::Check => {
            let issues = check_consistency(problem);
            SolveResult {
                mode: "check".into(),
                valid: issues.is_empty(),
                violations: vec![],
                issues,
                domains: None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_range(min: f64, max: f64) -> Constraint {
        Constraint::Range { min, max }
    }

    fn make_enum(values: &[&str]) -> Constraint {
        Constraint::Enum {
            values: values.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn make_assignment(field: &str, val: serde_json::Value) -> (String, serde_json::Value) {
        (field.to_string(), val)
    }

    fn make_problem(fields: Vec<(&str, Vec<Constraint>)>) -> Problem {
        Problem {
            fields: fields
                .into_iter()
                .map(|(name, constraints)| FieldConstraints {
                    name: name.to_string(),
                    constraints,
                })
                .collect(),
        }
    }

    #[test]
    fn validate_range_happy_path() {
        let problem = make_problem(vec![("x", vec![make_range(0.0, 100.0)])]);
        let assignment = vec![make_assignment("x", serde_json::json!(50))];
        let violations = validate(&problem, &assignment);
        assert!(violations.is_empty());
    }

    #[test]
    fn validate_range_outside() {
        let problem = make_problem(vec![("x", vec![make_range(0.0, 100.0)])]);
        let assignment = vec![make_assignment("x", serde_json::json!(200))];
        let violations = validate(&problem, &assignment);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].constraint, "Range");
    }

    #[test]
    fn validate_range_string_value() {
        let problem = make_problem(vec![("x", vec![make_range(0.0, 100.0)])]);
        let assignment = vec![make_assignment("x", serde_json::json!("50"))];
        let violations = validate(&problem, &assignment);
        assert!(violations.is_empty());
    }

    #[test]
    fn validate_required_field_present() {
        let problem = make_problem(vec![("x", vec![Constraint::Required])]);
        let assignment = vec![make_assignment("x", serde_json::json!("hello"))];
        let violations = validate(&problem, &assignment);
        assert!(violations.is_empty());
    }

    #[test]
    fn validate_required_field_missing() {
        let problem = make_problem(vec![("x", vec![Constraint::Required])]);
        let assignment: Vec<(String, serde_json::Value)> = vec![];
        let violations = validate(&problem, &assignment);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].message.contains("missing"));
    }

    #[test]
    fn validate_enum() {
        let problem = make_problem(vec![("x", vec![make_enum(&["A", "B", "C"])])]);
        let assignment = vec![make_assignment("x", serde_json::json!("A"))];
        assert!(validate(&problem, &assignment).is_empty());
        let assignment2 = vec![make_assignment("x", serde_json::json!("D"))];
        assert!(!validate(&problem, &assignment2).is_empty());
    }

    #[test]
    fn validate_conditional_only_applies_when_condition_met() {
        let inner = Constraint::Range { min: 10.0, max: 20.0 };
        let cond = Constraint::Conditional {
            condition_field: "type".into(),
            condition_value: serde_json::json!("special"),
            inner: Box::new(inner),
        };
        let problem = make_problem(vec![("x", vec![cond])]);
        // condition not met → no violation
        let assignment = vec![
            make_assignment("type", serde_json::json!("normal")),
            make_assignment("x", serde_json::json!(99)),
        ];
        assert!(validate(&problem, &assignment).is_empty());
        // condition met → violation
        let assignment2 = vec![
            make_assignment("type", serde_json::json!("special")),
            make_assignment("x", serde_json::json!(99)),
        ];
        assert!(!validate(&problem, &assignment2).is_empty());
    }

    #[test]
    fn infer_range_domain() {
        let problem = make_problem(vec![("x", vec![make_range(0.0, 100.0)])]);
        let domains = infer_domains(&problem, &[]);
        assert_eq!(domains.len(), 1);
        assert_eq!(
            domains[0].domain,
            Domain::Interval {
                min: 0.0,
                max: 100.0
            }
        );
    }

    #[test]
    fn infer_enum_domain() {
        let problem = make_problem(vec![("x", vec![make_enum(&["A", "B"])])]);
        let domains = infer_domains(&problem, &[]);
        assert_eq!(
            domains[0].domain,
            Domain::Set {
                values: vec!["A".into(), "B".into()]
            }
        );
    }

    #[test]
    fn check_consistency_min_gt_max() {
        let problem = make_problem(vec![("x", vec![make_range(100.0, 0.0)])]);
        let issues = check_consistency(&problem);
        assert_eq!(issues.len(), 1);
        assert!(issues[0].message.contains("min"));
    }

    #[test]
    fn check_consistency_enum_empty() {
        let problem = make_problem(vec![("x", vec![make_enum(&[])])]);
        let issues = check_consistency(&problem);
        assert_eq!(issues.len(), 1);
    }

    #[test]
    fn check_consistency_required_and_forbidden_conflict() {
        let problem = make_problem(vec![("x", vec![Constraint::Required, Constraint::Forbidden])]);
        let issues = check_consistency(&problem);
        assert_eq!(issues.len(), 1);
        assert!(issues[0].message.contains("Required + Forbidden"));
    }

    #[test]
    fn solve_mode_validate_passes() {
        let problem = make_problem(vec![("x", vec![make_range(0.0, 10.0)])]);
        let assignment = vec![make_assignment("x", serde_json::json!(5))];
        let result = solve(&problem, SolveMode::Validate, &assignment);
        assert!(result.valid);
        assert_eq!(result.violations.len(), 0);
    }

    #[test]
    fn solve_mode_infer_returns_domains() {
        let problem = make_problem(vec![("x", vec![make_range(0.0, 10.0)])]);
        let result = solve(&problem, SolveMode::Infer, &[]);
        assert!(result.valid);
        assert!(result.domains.is_some());
    }

    #[test]
    fn solve_mode_check_returns_issues() {
        let problem = make_problem(vec![("x", vec![make_range(10.0, 0.0)])]);
        let result = solve(&problem, SolveMode::Check, &[]);
        assert!(!result.valid);
        assert_eq!(result.issues.len(), 1);
    }

    #[test]
    fn domain_intersection_interval_overlap() {
        let a = Domain::Interval { min: 0.0, max: 100.0 };
        let b = Domain::Interval { min: 50.0, max: 150.0 };
        let c = intersect_domain(a, b);
        assert_eq!(c, Domain::Interval { min: 50.0, max: 100.0 });
    }

    #[test]
    fn domain_intersection_interval_disjoint() {
        let a = Domain::Interval { min: 0.0, max: 10.0 };
        let b = Domain::Interval { min: 20.0, max: 30.0 };
        let c = intersect_domain(a, b);
        assert_eq!(c, Domain::Empty);
    }

    #[test]
    fn domain_intersection_set_with_interval() {
        let set = Domain::Set {
            values: vec!["1".into(), "50".into(), "100".into()],
        };
        let interval = Domain::Interval { min: 10.0, max: 60.0 };
        let c = intersect_domain(set, interval);
        assert_eq!(
            c,
            Domain::Set {
                values: vec!["50".into()]
            }
        );
    }

    /// A `S = d * 1.7` formula pins S once d is assigned; without d it pins nothing.
    #[test]
    fn infer_formula_pins_dependent_field() {
        let problem = make_problem(vec![
            ("d", vec![make_range(2.0, 36.0)]),
            ("S", vec![Constraint::Formula { expression: "S = d * 1.7".into() }]),
        ]);
        let domains = infer_domains(&problem, &[make_assignment("d", serde_json::json!(10))]);
        let s = domains.iter().find(|d| d.field == "S").unwrap();
        assert_eq!(s.domain, Domain::Set { values: vec!["17".into()] });

        let domains = infer_domains(&problem, &[]);
        let s = domains.iter().find(|d| d.field == "S").unwrap();
        assert_eq!(s.domain, Domain::Any);
    }

    /// A Conditional unfolds into its inner domain only when the condition holds;
    /// the condition matches across the String/Number split ("41" ≡ 41).
    #[test]
    fn infer_conditional_unfolds_when_condition_met() {
        let cond = Constraint::Conditional {
            condition_field: "mode".into(),
            condition_value: serde_json::json!("41"),
            inner: Box::new(make_enum(&["a", "b"])),
        };
        let problem = make_problem(vec![("x", vec![cond])]);

        let met = infer_domains(&problem, &[make_assignment("mode", serde_json::json!(41))]);
        assert_eq!(met[0].domain, Domain::Set { values: vec!["a".into(), "b".into()] });

        let unmet = infer_domains(&problem, &[make_assignment("mode", serde_json::json!("42"))]);
        assert_eq!(unmet[0].domain, Domain::Any);

        let unassigned = infer_domains(&problem, &[]);
        assert_eq!(unassigned[0].domain, Domain::Any);
    }

    /// validate applies a Conditional whose condition value type differs from the
    /// assignment's (graph literal "41" vs JSON number 41).
    #[test]
    fn validate_conditional_condition_matches_across_value_types() {
        let cond = Constraint::Conditional {
            condition_field: "mode".into(),
            condition_value: serde_json::json!("41"),
            inner: Box::new(make_enum(&["a"])),
        };
        let problem = make_problem(vec![("x", vec![cond])]);
        let violations = validate(
            &problem,
            &[
                make_assignment("mode", serde_json::json!(41)),
                make_assignment("x", serde_json::json!("z")),
            ],
        );
        assert_eq!(violations.len(), 1, "condition met — inner enum must fire");
    }
}
