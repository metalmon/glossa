//! Table compiler capabilities: what shapes can be emitted and whether the ontology supports them.

use crate::graph::ontology::Ontology;
use crate::tables::wiring::{
    TablesCompileWiring, EDGE_CONSTRAINED_BY, EDGE_HAS_CONSTRAINT, EDGE_HAS_EXPRESSION,
    EDGE_HAS_MAX, EDGE_HAS_MIN, EDGE_HAS_PATTERN, EDGE_IF_FIELD, EDGE_IF_VALUE,
};

pub const PATTERN_INDEPENDENT_ENUM: &str = "independent_enum";
pub const PATTERN_CONDITIONAL_ENUM: &str = "conditional_enum";
pub const PATTERN_CONDITIONAL_RANGE: &str = "conditional_range";
pub const PATTERN_FORMULA: &str = "formula_cross_field";
pub const PATTERN_PROVENANCE: &str = "provenance";
pub const PATTERN_COMBINED: &str = "combined_constraints";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityId {
    IndependentEnum,
    ConditionalEnum,
    ConditionalRange,
    Formula,
    Regex,
    Provenance,
    Combined,
}

impl CapabilityId {
    pub fn pattern_key(self) -> &'static str {
        match self {
            Self::IndependentEnum => PATTERN_INDEPENDENT_ENUM,
            Self::ConditionalEnum => PATTERN_CONDITIONAL_ENUM,
            Self::ConditionalRange => PATTERN_CONDITIONAL_RANGE,
            Self::Formula => PATTERN_FORMULA,
            Self::Regex => "regex",
            Self::Provenance => PATTERN_PROVENANCE,
            Self::Combined => PATTERN_COMBINED,
        }
    }

    pub fn implemented(self) -> bool {
        matches!(self, Self::IndependentEnum | Self::ConditionalEnum)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityStatus {
    Ready,
    OntologyGap(String),
    CompilerGap(String),
}

pub struct CapabilityScan {
    pub entries: Vec<(CapabilityId, CapabilityStatus)>,
}

impl CapabilityScan {
    pub fn scan(ont: &Ontology, wiring: &TablesCompileWiring) -> Self {
        let entries = vec![
            (
                CapabilityId::IndependentEnum,
                check_independent(ont, wiring),
            ),
            (
                CapabilityId::ConditionalEnum,
                check_conditional_enum(ont, wiring),
            ),
            (CapabilityId::ConditionalRange, check_conditional_range(ont)),
            (CapabilityId::Formula, check_formula(ont, wiring)),
            (CapabilityId::Regex, check_regex(ont, wiring)),
            (CapabilityId::Provenance, check_provenance(ont)),
            (CapabilityId::Combined, check_combined(ont)),
        ];
        Self { entries }
    }

    pub fn report_lines(&self) -> Vec<String> {
        self.entries
            .iter()
            .map(|(id, st)| {
                let key = id.pattern_key();
                match st {
                    CapabilityStatus::Ready => format!("capability {key}: ready"),
                    CapabilityStatus::OntologyGap(msg) => {
                        format!("capability {key}: ontology_gap ({msg})")
                    }
                    CapabilityStatus::CompilerGap(msg) => {
                        format!("capability {key}: compiler_gap ({msg})")
                    }
                }
            })
            .collect()
    }

    pub fn status(&self, id: CapabilityId) -> Option<&CapabilityStatus> {
        self.entries.iter().find(|(i, _)| *i == id).map(|(_, s)| s)
    }

    pub fn require_ready(&self, id: CapabilityId) -> Result<(), String> {
        match self.status(id) {
            Some(CapabilityStatus::Ready) => Ok(()),
            Some(CapabilityStatus::OntologyGap(msg)) => Err(format!(
                "capability {} not available: {msg}",
                id.pattern_key()
            )),
            Some(CapabilityStatus::CompilerGap(msg)) => Err(format!(
                "capability {} not implemented: {msg}",
                id.pattern_key()
            )),
            None => Err(format!("unknown capability {}", id.pattern_key())),
        }
    }
}

fn pattern_declared(ont: &Ontology, key: &str) -> bool {
    ont.patterns().contains_key(key)
}

fn relation_ok(ont: &Ontology, name: &str) -> bool {
    ont.raw_relations().contains_key(name)
}

fn check_independent(ont: &Ontology, wiring: &TablesCompileWiring) -> CapabilityStatus {
    if !pattern_declared(ont, PATTERN_INDEPENDENT_ENUM) {
        return CapabilityStatus::OntologyGap(format!("add [patterns.{PATTERN_INDEPENDENT_ENUM}]"));
    }
    if !relation_ok(ont, EDGE_CONSTRAINED_BY) {
        return CapabilityStatus::OntologyGap(format!("missing relation {EDGE_CONSTRAINED_BY}"));
    }
    if !ont.has_constraint_type(&wiring.enum_constraint) {
        return CapabilityStatus::OntologyGap(format!(
            "missing constraint_types.{}",
            wiring.enum_constraint
        ));
    }
    if !CapabilityId::IndependentEnum.implemented() {
        return CapabilityStatus::CompilerGap("planned v1".into());
    }
    CapabilityStatus::Ready
}

fn check_conditional_enum(ont: &Ontology, wiring: &TablesCompileWiring) -> CapabilityStatus {
    if !pattern_declared(ont, PATTERN_CONDITIONAL_ENUM) {
        return CapabilityStatus::OntologyGap(format!("add [patterns.{PATTERN_CONDITIONAL_ENUM}]"));
    }
    for e in [
        EDGE_CONSTRAINED_BY,
        EDGE_IF_FIELD,
        EDGE_IF_VALUE,
        EDGE_HAS_CONSTRAINT,
    ] {
        if !relation_ok(ont, e) {
            return CapabilityStatus::OntologyGap(format!("missing relation {e}"));
        }
    }
    if !ont.has_constraint_type(&wiring.conditional_constraint) {
        return CapabilityStatus::OntologyGap(format!(
            "missing constraint_types.{}",
            wiring.conditional_constraint
        ));
    }
    if !CapabilityId::ConditionalEnum.implemented() {
        return CapabilityStatus::CompilerGap("planned v1".into());
    }
    CapabilityStatus::Ready
}

fn check_conditional_range(ont: &Ontology) -> CapabilityStatus {
    if !pattern_declared(ont, PATTERN_CONDITIONAL_RANGE) {
        return CapabilityStatus::CompilerGap("pattern not declared (optional)".into());
    }
    for e in [EDGE_HAS_MIN, EDGE_HAS_MAX] {
        if !relation_ok(ont, e) {
            return CapabilityStatus::OntologyGap(format!("missing relation {e}"));
        }
    }
    if !ont.has_constraint_type("Range") {
        return CapabilityStatus::OntologyGap("missing constraint_types.Range".into());
    }
    CapabilityStatus::CompilerGap("planned v2".into())
}

fn check_formula(ont: &Ontology, wiring: &TablesCompileWiring) -> CapabilityStatus {
    if !pattern_declared(ont, PATTERN_FORMULA) {
        return CapabilityStatus::CompilerGap("pattern not declared (optional)".into());
    }
    if !relation_ok(ont, EDGE_HAS_EXPRESSION) {
        return CapabilityStatus::OntologyGap(format!("missing relation {EDGE_HAS_EXPRESSION}"));
    }
    if !ont.has_constraint_type("Formula") {
        return CapabilityStatus::OntologyGap("missing constraint_types.Formula".into());
    }
    let _ = wiring;
    CapabilityStatus::CompilerGap("planned v2".into())
}

fn check_regex(ont: &Ontology, _wiring: &TablesCompileWiring) -> CapabilityStatus {
    if !relation_ok(ont, EDGE_HAS_PATTERN) {
        return CapabilityStatus::OntologyGap(format!("missing relation {EDGE_HAS_PATTERN}"));
    }
    if !ont.has_constraint_type("Regex") {
        return CapabilityStatus::OntologyGap("missing constraint_types.Regex".into());
    }
    CapabilityStatus::CompilerGap("planned v2".into())
}

fn check_provenance(_ont: &Ontology) -> CapabilityStatus {
    // MENTIONS is a core edge; provenance emitter is v2.
    CapabilityStatus::CompilerGap("planned v2".into())
}

fn check_combined(ont: &Ontology) -> CapabilityStatus {
    if !pattern_declared(ont, PATTERN_COMBINED) {
        return CapabilityStatus::CompilerGap("pattern not declared (optional)".into());
    }
    CapabilityStatus::CompilerGap("planned v3".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ontology::Ontology;
    use crate::tables::wiring::TablesCompileWiring;

    fn eval_ontology() -> Ontology {
        let s = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/eval/ontology-constraint.toml"
        ));
        Ontology::parse(s).unwrap()
    }

    #[test]
    fn eval_ontology_v1_capabilities_ready() {
        let ont = eval_ontology();
        let wiring = TablesCompileWiring::resolve(&ont).unwrap();
        let scan = CapabilityScan::scan(&ont, &wiring);
        assert!(matches!(
            scan.status(CapabilityId::IndependentEnum),
            Some(CapabilityStatus::Ready)
        ));
        assert!(matches!(
            scan.status(CapabilityId::ConditionalEnum),
            Some(CapabilityStatus::Ready)
        ));
        assert!(matches!(
            scan.status(CapabilityId::Formula),
            Some(CapabilityStatus::CompilerGap(_))
        ));
    }
}
