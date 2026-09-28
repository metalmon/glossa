//! Shared config-resolution helpers: env-var readers and execution-provider normalization.
//! Extracted from `gate::config` so other config resolvers (e.g. a future `RerankConfig`) can
//! reuse them without duplicating the parsing/fail-open behavior.

/// Execution providers the runtime actually knows how to register — exactly the EPs `glossa-nli`
/// has a Cargo feature + dispatch arm for (`nli-cuda`, `nli-directml`, `nli-coreml`, `nli-rocm`)
/// plus the implicit `"cpu"` fallback. Anything else is dropped with a warning rather than
/// erroring — fail-open, since a bad/typo'd EP name should never block config resolution.
pub const KNOWN_EXECUTION_PROVIDERS: &[&str] = &["cpu", "cuda", "directml", "coreml", "rocm"];

/// Lowercase + trim each entry, drop anything outside [`KNOWN_EXECUTION_PROVIDERS`] (warning, not
/// error), and default to `["cpu"]` when the result is empty — whether because the input was empty
/// or because every entry was unknown.
pub fn normalize_eps(raw: impl Iterator<Item = String>) -> Vec<String> {
    let normalized: Vec<String> = raw
        .map(|s| s.trim().to_lowercase())
        .filter(|s| {
            let known = KNOWN_EXECUTION_PROVIDERS.contains(&s.as_str());
            if !known {
                tracing::warn!(execution_provider = %s, "unknown execution_providers entry dropped");
            }
            known
        })
        .collect();
    if normalized.is_empty() {
        vec!["cpu".to_string()]
    } else {
        normalized
    }
}

pub fn env_f32(key: &str) -> Option<f32> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}
pub fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}
pub fn env_bool(key: &str) -> Option<bool> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}
pub fn env_string(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}
/// Parse an env var as `i32`. Absent/empty ⇒ `None`; a set-but-non-integer value is ignored with a
/// warning (fail-open — a typo'd device id should never take config resolution down, it just falls
/// back to the default device).
pub fn env_i32(key: &str) -> Option<i32> {
    let raw = std::env::var(key).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.parse::<i32>() {
        Ok(n) => Some(n),
        Err(_) => {
            tracing::warn!(key, value = %raw, "non-integer NLI EP device id ignored");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_eps_drops_unknown_and_lowercases() {
        let out = normalize_eps(["CUDA".to_string(), "bogus".to_string()].into_iter());
        assert_eq!(out, vec!["cuda".to_string()]);
    }

    #[test]
    fn normalize_eps_empty_falls_back_to_cpu() {
        let out = normalize_eps(std::iter::empty());
        assert_eq!(out, vec!["cpu".to_string()]);
    }
}
