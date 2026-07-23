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

/// The raw scalar text of a JSON value — no surrounding quotes (unlike `Display`
/// on a `Value::String`).
pub fn scalar_str(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Fold the Cyrillic letters that are visual homoglyphs of a Latin letter to
/// that Latin letter (input already lowercased). Real-world technical data mixes
/// scripts freely in codes — a grade "14А" typed with Cyrillic А and "14a" typed
/// with Latin a are the SAME value, but different Unicode. Folding to one script
/// makes them compare equal. Only the ~12 look-alikes are touched; any other
/// Cyrillic letter (a real word) is left as-is.
fn fold_homoglyph(c: char) -> char {
    match c {
        'а' => 'a',
        'в' => 'b',
        'е' => 'e',
        'к' => 'k',
        'м' => 'm',
        'н' => 'h',
        'о' => 'o',
        'р' => 'p',
        'с' => 'c',
        'т' => 't',
        'у' => 'y',
        'х' => 'x',
        other => other,
    }
}

/// Unify the dash variants (em/en/minus/hyphen) to `-` and drop spaces around it,
/// so a range label reads the same however it was typed: "41—43" == "41 - 43" ==
/// "41-43". Non-dash text is untouched.
fn normalize_dashes(s: &str) -> String {
    let unified: String = s
        .chars()
        .map(|c| {
            if matches!(c, '—' | '–' | '−' | '‐' | '-') {
                '-'
            } else {
                c
            }
        })
        .collect();
    if unified.contains('-') {
        unified
            .split('-')
            .map(str::trim)
            .collect::<Vec<_>>()
            .join("-")
    } else {
        unified
    }
}

/// Parse a range label `a-b` (after canonicalisation) into `(lo, hi)`. `None` when
/// it is not a two-number range (a plain value, a code, a negative number).
pub fn parse_range(v: &str) -> Option<(f64, f64)> {
    let c = canon_scalar(v);
    let (a, b) = c.split_once('-')?;
    let (a, b): (f64, f64) = (a.trim().parse().ok()?, b.trim().parse().ok()?);
    Some(if a <= b { (a, b) } else { (b, a) })
}

/// Integer closed interval from a range label (`25-49`) or a plain integer atom (`27`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IntSpan {
    lo: i64,
    hi: i64,
}

fn near_int(x: f64) -> Option<i64> {
    if !x.is_finite() || (x - x.round()).abs() > 1e-9 {
        return None;
    }
    Some(x.round() as i64)
}

fn parse_int_span(v: &str) -> Option<IntSpan> {
    let c = canon_scalar(v);
    if let Some((a, b)) = parse_range(&c) {
        return Some(IntSpan {
            lo: near_int(a)?,
            hi: near_int(b)?,
        });
    }
    let n: f64 = c.replace(',', ".").parse().ok()?;
    let x = near_int(n)?;
    Some(IntSpan { lo: x, hi: x })
}

fn span_subseteq(piece: IntSpan, gold: IntSpan) -> bool {
    piece.lo >= gold.lo && piece.hi <= gold.hi
}

fn lattice_step(endpoints: &[i64]) -> i64 {
    if endpoints.is_empty() {
        return 1;
    }
    let all_odd = endpoints.iter().all(|e| e.rem_euclid(2) == 1);
    let all_even = endpoints.iter().all(|e| e.rem_euclid(2) == 0);
    if all_odd || all_even {
        2
    } else {
        1
    }
}

/// Whether agent values collectively cover a gold scalar under range equivalence.
///
/// After canon (`25—49` ≡ `25-49`):
/// - exact string match → covered;
/// - gold atom → covered if some agent span contains it;
/// - gold range → agent spans that lie inside the gold range must have matching
///   hull (`lo`/`hi`), and every lattice point in the gold range must fall in at
///   least one such span. Lattice step is 2 when all endpoints among gold and
///   agent numeric spans are odd (or all even); otherwise step 1 (every integer).
pub fn values_cover<'a, I>(agent_vals: I, gold: &str) -> bool
where
    I: IntoIterator<Item = &'a str>,
{
    let gold_c = canon_scalar(gold);
    let agents: Vec<String> = agent_vals
        .into_iter()
        .map(canon_scalar)
        .collect();
    if agents.iter().any(|a| a == &gold_c) {
        return true;
    }
    let Some(gold_span) = parse_int_span(&gold_c) else {
        return false;
    };
    let pieces: Vec<IntSpan> = agents.iter().filter_map(|a| parse_int_span(a)).collect();
    if gold_span.lo == gold_span.hi {
        let x = gold_span.lo;
        return pieces.iter().any(|p| p.lo <= x && x <= p.hi);
    }
    let contributing: Vec<IntSpan> = pieces
        .iter()
        .copied()
        .filter(|p| span_subseteq(*p, gold_span))
        .collect();
    if contributing.is_empty() {
        return false;
    }
    let hull_lo = contributing.iter().map(|p| p.lo).min().unwrap();
    let hull_hi = contributing.iter().map(|p| p.hi).max().unwrap();
    if hull_lo != gold_span.lo || hull_hi != gold_span.hi {
        return false;
    }
    let mut ends = vec![gold_span.lo, gold_span.hi];
    for p in &pieces {
        ends.push(p.lo);
        ends.push(p.hi);
    }
    let step = lattice_step(&ends);
    let mut x = gold_span.lo;
    while x <= gold_span.hi {
        if !contributing.iter().any(|p| p.lo <= x && x <= p.hi) {
            return false;
        }
        x += step;
    }
    true
}

/// Whether one agent value is allowed by a gold value (exact, or agent span ⊆ gold span).
pub fn value_subseteq(agent: &str, gold: &str) -> bool {
    let a = canon_scalar(agent);
    let g = canon_scalar(gold);
    if a == g {
        return true;
    }
    match (parse_int_span(&a), parse_int_span(&g)) {
        (Some(ap), Some(gp)) => span_subseteq(ap, gp),
        _ => false,
    }
}

/// Canonical form of a scalar for comparison. A decimal written with either a
/// comma (comma-decimal locales: "0,6", "1,0") or a dot collapses to one numeric
/// form, so "1,0" == "1.0" == "1" and "152,4" == "152.4". Non-numeric values
/// (e.g. "F46", "23-25", "12a") are trimmed, lowercased, have Cyrillic/Latin
/// homoglyphs folded to one script ("14А" == "14a", "В" == "b"), and dash variants
/// unified ("41—43" == "41-43"). Symmetric, so it is safe to apply on both sides
/// of both validation and metric comparisons.
pub fn canon_scalar(s: &str) -> String {
    let t = s.trim();
    if let Ok(n) = t.replace(',', ".").parse::<f64>() {
        if n.is_finite() && n.fract() == 0.0 {
            return format!("{}", n as i64);
        }
        return format!("{n}"); // f64 Display drops trailing zeros
    }
    let folded: String = t.to_lowercase().chars().map(fold_homoglyph).collect();
    normalize_dashes(&folded)
}

/// True when an Enum alias carries regex metacharacters and should be compiled,
/// not compared as a literal token.
pub fn is_enum_regex_alias(alias: &str) -> bool {
    alias.chars().any(|c| "[](){}+*?\\^$|".contains(c))
}

/// Whether a marking satisfies an Enum alias written as a regex pattern.
/// Both sides are canon-scalar'd before matching so a weak pattern like `\d+[A]`
/// accepts Cyrillic А in the marking — the same rule as eval `domain_covers`.
pub fn enum_alias_matches(alias: &str, marking: &str) -> bool {
    if !is_enum_regex_alias(alias) {
        return false;
    }
    let marking = canon_scalar(marking);
    let pattern = canon_scalar(alias);
    regex_lite::Regex::new(&format!("^(?:{pattern})$")).is_ok_and(|re| re.is_match(&marking))
}

/// Whether a marking satisfies a standalone Regex constraint pattern.
pub fn regex_constraint_matches(pattern: &str, marking: &str) -> bool {
    let marking = canon_scalar(marking);
    let pattern = canon_scalar(pattern);
    regex_lite::Regex::new(&pattern).is_ok_and(|re| re.is_match(&marking))
}

/// Value equality tolerant of the String/Number split and of decimal notation:
/// the graph stores literals as strings (possibly comma-decimals) while
/// assignments may carry JSON numbers ("41" ≡ 41, "1,0" ≡ 1).
fn values_eq(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    a == b || canon_scalar(&scalar_str(a)) == canon_scalar(&scalar_str(b))
}

/// Validate field assignments against constraints.
pub fn validate(problem: &Problem, assignment: &[(String, serde_json::Value)]) -> Vec<Violation> {
    let mut violations = Vec::new();
    let assign_map: std::collections::HashMap<&str, &serde_json::Value> =
        assignment.iter().map(|(k, v)| (k.as_str(), v)).collect();

    for fc in &problem.fields {
        for c in &fc.constraints {
            let applicable = match c {
                Constraint::Conditional {
                    condition_field,
                    condition_value,
                    ..
                } => assign_map
                    .get(condition_field.as_str())
                    .is_some_and(|v| values_eq(v, condition_value)),
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
                        let s = scalar_str(val);
                        if !regex_constraint_matches(pattern, &s) {
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
                        // Compare in canonical form so "1,0" (comma-decimal literal)
                        // matches the number 1, and "125" matches "125".
                        let s = scalar_str(val);
                        let target = canon_scalar(&s);
                        let exact = values.iter().any(|v| canon_scalar(v) == target);
                        // Range containment: a numeric marking falls inside an allowed
                        // range LABEL — "42" is valid against the allowed value "41-43".
                        let in_range = target.parse::<f64>().ok().is_some_and(|n| {
                            values
                                .iter()
                                .any(|v| parse_range(v).is_some_and(|(a, b)| a <= n && n <= b))
                        });
                        // Pattern membership: regex alias in the Enum set (e.g. `\d+[A]`
                        // for a whole family of grades). Canon on both sides so Cyrillic
                        // А in a marking matches Latin A in the pattern.
                        let matches_pattern = values.iter().any(|v| enum_alias_matches(v, &s));
                        if !exact && !in_range && !matches_pattern {
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
                                    message: format!("formula '{}' evaluated to false", expression),
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
pub fn infer_domains(
    problem: &Problem,
    assignment: &[(String, serde_json::Value)],
) -> Vec<FieldDomain> {
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
                    Constraint::Conditional {
                        condition_field,
                        condition_value,
                        inner,
                    } => assign_map
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
                        domain = intersect_domain(
                            domain,
                            Domain::Set {
                                values: vec![fmt_num(v)],
                            },
                        );
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
            if matches!(prev, Some(b'<') | Some(b'>') | Some(b'!') | Some(b'='))
                || next == Some(&b'=')
            {
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
                domains.push(Domain::Interval {
                    min: *min,
                    max: *max,
                });
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
        (
            Interval {
                min: min1,
                max: max1,
            },
            Interval {
                min: min2,
                max: max2,
            },
        ) => {
            let min = min1.max(*min2);
            let max = max1.min(*max2);
            if min > max {
                Empty
            } else {
                Interval { min, max }
            }
        }
        (Set { values: v1 }, Set { values: v2 }) => {
            let intersection: Vec<String> = v1.iter().filter(|x| v2.contains(x)).cloned().collect();
            if intersection.is_empty() {
                Empty
            } else {
                Set {
                    values: intersection,
                }
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
                Constraint::Range { min, max } if min > max => {
                    issues.push(ConsistencyIssue {
                        field: fc.name.clone(),
                        severity: "error".into(),
                        message: format!("Range min ({}) > max ({}) — no possible value", min, max),
                    });
                }
                Constraint::Regex { pattern } if regex_lite::Regex::new(pattern).is_err() => {
                    issues.push(ConsistencyIssue {
                        field: fc.name.clone(),
                        severity: "error".into(),
                        message: format!("Invalid regex pattern: '{}'", pattern),
                    });
                }
                Constraint::Enum { values } if values.is_empty() => {
                    issues.push(ConsistencyIssue {
                        field: fc.name.clone(),
                        severity: "error".into(),
                        message: "Enum has no values — no possible value".into(),
                    });
                }
                _ => {}
            }
        }
        // Check conflicting constraints
        let has_forbidden = fc
            .constraints
            .iter()
            .any(|c| matches!(c, Constraint::Forbidden));
        let has_required = fc
            .constraints
            .iter()
            .any(|c| matches!(c, Constraint::Required));
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
pub fn solve(
    problem: &Problem,
    mode: SolveMode,
    assignment: &[(String, serde_json::Value)],
) -> SolveResult {
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
    fn range_label_normalization_and_parse() {
        assert_eq!(canon_scalar("41—43"), "41-43"); // em-dash
        assert_eq!(canon_scalar("41 - 43"), "41-43"); // spaces around hyphen
        assert_eq!(canon_scalar("23–25"), "23-25"); // en-dash
        assert_eq!(parse_range("23-25"), Some((23.0, 25.0)));
        assert_eq!(parse_range("41—43"), Some((41.0, 43.0)));
        assert_eq!(parse_range("F46"), None);
        assert_eq!(parse_range("125"), None);
    }

    #[test]
    fn values_cover_odd_bands_match_wide_gold_range() {
        // Narrow adjacent bands vs one wide gold range — same odd lattice, full hull.
        let bands = [
            "25-27", "29-31", "33-35", "37-39", "41-43", "45-49",
        ];
        assert!(values_cover(bands.iter().copied(), "25—49"));
        assert!(values_cover(["25-49"].iter().copied(), "25-49"));
        // Missing an interior odd band → gap on step-2 lattice.
        let gap = ["25-27", "29-31", "37-39", "41-43", "45-49"];
        assert!(!values_cover(gap.iter().copied(), "25-49"));
        // Hull too short.
        assert!(!values_cover(["25-27", "29-31"].iter().copied(), "25-49"));
        // Piece outside gold → not contributing; hull fails if only outsider remains.
        assert!(!values_cover(["25-51"].iter().copied(), "25-49"));
    }

    #[test]
    fn values_cover_mixed_parity_uses_integer_lattice() {
        // Mixed endpoints → every integer in [lo,hi] must be covered.
        assert!(values_cover(["1-5"].iter().copied(), "1-5"));
        assert!(values_cover(["1-2", "3-5"].iter().copied(), "1-5"));
        assert!(!values_cover(["1-2", "4-5"].iter().copied(), "1-5")); // missing 3
    }

    #[test]
    fn values_cover_atom_inside_agent_range() {
        assert!(values_cover(["25-49"].iter().copied(), "27"));
        assert!(values_cover(["25-27", "29-31"].iter().copied(), "27"));
        assert!(!values_cover(["25-27"].iter().copied(), "33"));
        assert!(!values_cover(["F46"].iter().copied(), "27"));
    }

    #[test]
    fn value_subseteq_band_inside_gold_range() {
        assert!(value_subseteq("25-27", "25-49"));
        assert!(value_subseteq("27", "25-49"));
        assert!(value_subseteq("25—49", "25-49"));
        assert!(!value_subseteq("25-51", "25-49"));
        assert!(!value_subseteq("B", "25-49"));
    }

    #[test]
    fn validate_enum_range_label_membership_and_containment() {
        // A parameter whose allowed values are RANGE LABELS.
        let problem = make_problem(vec![("si", vec![make_enum(&["23-25", "41-43"])])]);
        let ok = |v: serde_json::Value| validate(&problem, &[make_assignment("si", v)]).is_empty();
        assert!(
            ok(serde_json::json!("41—43")),
            "exact label match, dash variant"
        );
        assert!(ok(serde_json::json!(42)), "number inside an allowed range");
        assert!(
            ok(serde_json::json!("24")),
            "number-as-string inside a range"
        );
        assert!(!ok(serde_json::json!(30)), "number outside every range");
        assert!(!ok(serde_json::json!("50-52")), "label not in the set");
    }

    #[test]
    fn validate_conditional_only_applies_when_condition_met() {
        let inner = Constraint::Range {
            min: 10.0,
            max: 20.0,
        };
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
        let problem = make_problem(vec![(
            "x",
            vec![Constraint::Required, Constraint::Forbidden],
        )]);
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
        let a = Domain::Interval {
            min: 0.0,
            max: 100.0,
        };
        let b = Domain::Interval {
            min: 50.0,
            max: 150.0,
        };
        let c = intersect_domain(a, b);
        assert_eq!(
            c,
            Domain::Interval {
                min: 50.0,
                max: 100.0
            }
        );
    }

    #[test]
    fn domain_intersection_interval_disjoint() {
        let a = Domain::Interval {
            min: 0.0,
            max: 10.0,
        };
        let b = Domain::Interval {
            min: 20.0,
            max: 30.0,
        };
        let c = intersect_domain(a, b);
        assert_eq!(c, Domain::Empty);
    }

    #[test]
    fn domain_intersection_set_with_interval() {
        let set = Domain::Set {
            values: vec!["1".into(), "50".into(), "100".into()],
        };
        let interval = Domain::Interval {
            min: 10.0,
            max: 60.0,
        };
        let c = intersect_domain(set, interval);
        assert_eq!(
            c,
            Domain::Set {
                values: vec!["50".into()]
            }
        );
    }

    #[test]
    fn canon_scalar_collapses_decimal_notation() {
        assert_eq!(canon_scalar("1,0"), "1");
        assert_eq!(canon_scalar("1.0"), "1");
        assert_eq!(canon_scalar("1"), "1");
        assert_eq!(canon_scalar("10,0"), "10");
        assert_eq!(canon_scalar("152,4"), "152.4");
        assert_eq!(canon_scalar("0,6"), "0.6");
        // non-numeric passes through, lowercased
        assert_eq!(canon_scalar("23-25"), "23-25");
        assert_eq!(canon_scalar("F46"), "f46");
    }

    #[test]
    fn canon_scalar_folds_cyrillic_latin_homoglyphs() {
        // Grade "14А" (Cyrillic А) and "14a" (Latin a) are the same value.
        assert_eq!(canon_scalar("14А"), canon_scalar("14a"));
        assert_eq!(canon_scalar("14А"), "14a");
        // Code letters: "В"(Cyrillic) == "b", "Р"(Cyrillic) folds to "p" while the
        // non-homoglyph "д" is left as-is (so only real look-alikes are touched).
        assert_eq!(canon_scalar("В"), "b");
        assert_eq!(canon_scalar("Рд"), "pд");
        // Code with Cyrillic М: "М50" == "m50".
        assert_eq!(canon_scalar("М50"), "m50");
        // A Latin-only value is unchanged (only case folds).
        assert_eq!(canon_scalar("BF"), "bf");
        // A non-homoglyph Cyrillic letter is left alone (not mangled): б stays б.
        assert_eq!(canon_scalar("аб"), "aб"); // а→a (homoglyph); б has no Latin look-alike
    }

    /// A comma-decimal enum literal ("1,0") matches the assigned number 1 and its
    /// dot/plain spellings; a value outside the set is still rejected.
    #[test]
    fn validate_enum_matches_comma_decimals() {
        let problem = make_problem(vec![("h", vec![make_enum(&["0,6", "1,0", "152,4"])])]);
        for ok in [
            serde_json::json!(1),
            serde_json::json!("1.0"),
            serde_json::json!("1,0"),
            serde_json::json!(152.4),
        ] {
            assert!(
                validate(&problem, &[make_assignment("h", ok.clone())]).is_empty(),
                "{ok} should pass"
            );
        }
        assert!(
            !validate(&problem, &[make_assignment("h", serde_json::json!(2))]).is_empty(),
            "2 ∉ set"
        );
    }

    /// An allowed value written as a regex ("[0-9]+А") is a category pattern: a specific
    /// grade "14А" matches, "14Х"/"А14"/"99" do not, while a plain literal ("Z") in the
    /// same set stays literal.
    #[test]
    fn validate_enum_matches_category_pattern() {
        let problem = make_problem(vec![("m", vec![make_enum(&["[0-9]+А", "Z", "[0-9]+С"])])]);
        for ok in ["14А", "25А", "Z", "63С"] {
            assert!(
                validate(&problem, &[make_assignment("m", serde_json::json!(ok))]).is_empty(),
                "{ok} should pass"
            );
        }
        for bad in ["14Х", "А14", "99"] {
            assert!(
                !validate(&problem, &[make_assignment("m", serde_json::json!(bad))]).is_empty(),
                "{bad} should fail"
            );
        }
    }

    #[test]
    fn validate_enum_matches_backslash_d_pattern() {
        let problem = make_problem(vec![("m", vec![make_enum(&[r"\d+[A]", r"\d+[C]", "Z"])])]);
        for ok in ["14A", "14А", "12A", "53C", "63С", "Z"] {
            assert!(
                validate(&problem, &[make_assignment("m", serde_json::json!(ok))]).is_empty(),
                "{ok} should pass"
            );
        }
        for bad in ["A", "14", "A14", "14X"] {
            assert!(
                !validate(&problem, &[make_assignment("m", serde_json::json!(bad))]).is_empty(),
                "{bad} should fail"
            );
        }
    }

    /// A `S = d * 1.7` formula pins S once d is assigned; without d it pins nothing.
    #[test]
    fn infer_formula_pins_dependent_field() {
        let problem = make_problem(vec![
            ("d", vec![make_range(2.0, 36.0)]),
            (
                "S",
                vec![Constraint::Formula {
                    expression: "S = d * 1.7".into(),
                }],
            ),
        ]);
        let domains = infer_domains(&problem, &[make_assignment("d", serde_json::json!(10))]);
        let s = domains.iter().find(|d| d.field == "S").unwrap();
        assert_eq!(
            s.domain,
            Domain::Set {
                values: vec!["17".into()]
            }
        );

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
        assert_eq!(
            met[0].domain,
            Domain::Set {
                values: vec!["a".into(), "b".into()]
            }
        );

        let unmet = infer_domains(
            &problem,
            &[make_assignment("mode", serde_json::json!("42"))],
        );
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
