//! #8 — Transitive closure over ontology relation-composition rules.
//! e.g. `A -CAUSED_BY-> B` + `B -RESOLVED_BY-> C`  ⇒  `A -RESOLVED_BY-> C`.

use super::Triple;
use std::collections::HashSet;

/// A composition rule: if `x -a-> y` and `y -b-> z` then infer `x -result-> z`.
pub struct Rule {
    pub a: String,
    pub b: String,
    pub result: String,
}

impl Rule {
    pub fn new(a: &str, b: &str, result: &str) -> Self {
        Rule { a: a.into(), b: b.into(), result: result.into() }
    }
}

/// New edges inferred by applying each rule once (single-hop composition), excluding any edge
/// already present in `edges` and any self-edge. Deterministic (sorted, deduplicated).
pub fn transitive_closure(edges: &[Triple], rules: &[Rule]) -> Vec<Triple> {
    let existing: HashSet<Triple> = edges.iter().cloned().collect();
    let mut out: HashSet<Triple> = HashSet::new();
    for rule in rules {
        for (xf, xt, xto) in edges {
            if *xt != rule.a {
                continue;
            }
            for (yf, yt, yto) in edges {
                if *yt != rule.b || yf != xto {
                    continue;
                }
                if xf == yto {
                    continue;
                }
                let inferred = (xf.clone(), rule.result.clone(), yto.clone());
                if !existing.contains(&inferred) {
                    out.insert(inferred);
                }
            }
        }
    }
    let mut v: Vec<Triple> = out.into_iter().collect();
    v.sort();
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t(a: &str, b: &str, c: &str) -> Triple {
        (a.into(), b.into(), c.into())
    }

    #[test]
    fn infers_symptom_resolved_by_via_cause() {
        let edges = vec![t("S", "CAUSED_BY", "C"), t("C", "RESOLVED_BY", "R")];
        let rules = vec![Rule::new("CAUSED_BY", "RESOLVED_BY", "RESOLVED_BY")];
        assert_eq!(transitive_closure(&edges, &rules), vec![t("S", "RESOLVED_BY", "R")]);
    }

    #[test]
    fn does_not_duplicate_existing_or_self() {
        let edges = vec![
            t("S", "CAUSED_BY", "C"),
            t("C", "RESOLVED_BY", "R"),
            t("S", "RESOLVED_BY", "R"), // already present
        ];
        let rules = vec![Rule::new("CAUSED_BY", "RESOLVED_BY", "RESOLVED_BY")];
        assert!(transitive_closure(&edges, &rules).is_empty());
    }

    #[test]
    fn no_rule_match_yields_nothing() {
        let edges = vec![t("A", "REL", "B"), t("B", "REL2", "C")];
        let rules = vec![Rule::new("CAUSED_BY", "RESOLVED_BY", "RESOLVED_BY")];
        assert!(transitive_closure(&edges, &rules).is_empty());
    }
}
