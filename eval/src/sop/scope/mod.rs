//! Per-step allow/deny tool scope (vendored from zeroclaw SOP). Enforcement is a
//! prod-runtime concern (zeroclaw's `resolve_excluded` + SecurityPolicy), so only
//! the authoring type is kept here; the eval driver parses it but does not enforce.

use serde::{Deserialize, Serialize};

/// Per-step allow/deny tool scope. Enforcement is opt-in through SOP config.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepToolScope {
    /// Only these names or groups are allowed when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow: Option<Vec<String>>,
    /// These names or groups are always subtracted.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deny: Vec<String>,
}
