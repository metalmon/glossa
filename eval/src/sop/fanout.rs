//! Split a fan-out SOP step body into role sections and compose per-role prompts.

use crate::constraint_gepa_sop::strip_gepa_anchor_lines;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct RoleSections {
    pub shared: String,
    pub orchestrator: String,
    pub worker: String,
}

const M_SHARED: &str = "{# SHARED #}";
const M_ORCH: &str = "{# ORCHESTRATOR #}";
const M_WORKER: &str = "{# WORKER #}";

/// Split a step body on whole-line role markers. Text before the first marker
/// (and any `{# SHARED #}` block) goes to `shared`; the rest to the named role.
pub fn split_role_sections(body: &str) -> RoleSections {
    let mut out = RoleSections::default();
    // `shared` starts as the current bucket so pre-marker text is shared.
    let mut cur = 0u8; // 0 shared, 1 orchestrator, 2 worker
    for line in body.lines() {
        match line.trim() {
            M_SHARED => cur = 0,
            M_ORCH => cur = 1,
            M_WORKER => cur = 2,
            _ => {
                let bucket = match cur {
                    1 => &mut out.orchestrator,
                    2 => &mut out.worker,
                    _ => &mut out.shared,
                };
                bucket.push_str(line);
                bucket.push('\n');
            }
        }
    }
    out.shared = out.shared.trim().to_string();
    out.orchestrator = out.orchestrator.trim().to_string();
    out.worker = out.worker.trim().to_string();
    out
}

pub fn format_orchestrator_prompt(step_body: &str) -> String {
    let s = split_role_sections(&strip_gepa_anchor_lines(step_body));
    format!("{}\n\n{}\n", s.shared, s.orchestrator)
}

pub fn format_worker_prompt(step_body: &str, task: &str) -> String {
    let s = split_role_sections(&strip_gepa_anchor_lines(step_body));
    format!("{}\n\n{}\n\nТвоя задача:\n{}\n", s.shared, s.worker, task)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_three_role_sections() {
        let body = "\
intro line
{# SHARED #}
shared A
shared B
{# ORCHESTRATOR #}
orch A
{# WORKER #}
worker A
worker B";
        let s = split_role_sections(body);
        assert!(s.shared.contains("intro line"));
        assert!(s.shared.contains("shared A") && s.shared.contains("shared B"));
        assert_eq!(s.orchestrator.trim(), "orch A");
        assert!(s.worker.contains("worker A") && s.worker.contains("worker B"));
        assert!(!s.shared.contains("orch A"));
    }

    #[test]
    fn missing_markers_yield_empty() {
        let s = split_role_sections("just shared text");
        assert_eq!(s.shared.trim(), "just shared text");
        assert!(s.orchestrator.is_empty() && s.worker.is_empty());
    }

    #[test]
    fn builds_role_prompts() {
        let body = "{# SHARED #}\nBoth read this.\n{# ORCHESTRATOR #}\nDecompose and spawn.\n{# WORKER #}\nBuild one table.";
        let orch = format_orchestrator_prompt(body);
        assert!(orch.contains("Both read this."));
        assert!(orch.contains("Decompose and spawn."));
        assert!(!orch.contains("Build one table."));

        let w = format_worker_prompt(body, "Собери поле «высота»");
        assert!(w.contains("Both read this."));
        assert!(w.contains("Build one table."));
        assert!(w.contains("Собери поле «высота»"));
        assert!(!w.contains("Decompose and spawn."));
    }
}
