//! Eval SOP helpers: zeroclaw-compatible step context + file-backed eval tools.
//!
//! Step framing matches `zeroclaw-runtime` `format_step_context()` — body comes
//! straight from `SOP.md`. Eval-only tool schemas: `get_task.json`, `sop_advance.json`.

use anyhow::{Context, Result};
use serde_json::Value;
use std::fmt::Write as _;
use std::path::Path;

use super::types::{Sop, SopRun, SopStep};

const SOP_ADVANCE_FILE: &str = "sop_advance.json";
const GET_TASK_FILE: &str = "get_task.json";
const DELEGATE_FILE: &str = "delegate.json";

/// Replace `{key}` placeholders (legacy; prefer trigger payload / get_task).
pub fn substitute_placeholders(text: &str, vars: &[(&str, &str)]) -> String {
    let mut out = text.to_string();
    for (key, val) in vars {
        out = out.replace(&format!("{{{key}}}"), val);
    }
    out
}

/// Same shape as zeroclaw `engine.rs::format_step_context` (without security framing).
pub fn format_step_context(sop: &Sop, run: &SopRun, step: &SopStep) -> String {
    let mut ctx = format!(
        "[SOP: {} (run {}) — Step {} of {}]\n\n",
        sop.name, run.run_id, step.number, run.total_steps
    );

    if let Some(prev) = run.step_results.last() {
        let _ = writeln!(
            ctx,
            "Previous: Step {} {} — {}",
            prev.step_number, prev.status, prev.output
        );
    }

    let body = crate::constraint_gepa_sop::strip_gepa_anchor_lines(&step.body);
    let _ = write!(ctx, "\nCurrent step: **{}**\n{}\n", step.title, body);

    if !step.suggested_tools.is_empty() {
        let _ = write!(
            ctx,
            "\nSuggested tools: {}\n",
            step.suggested_tools.join(", ")
        );
    }

    ctx
}

/// Tool result prefix matching zeroclaw `sop_advance.rs`.
pub fn format_sop_advance_reply(sop: &Sop, run: &SopRun, step: &SopStep) -> String {
    format!(
        "Step recorded. Next step for run {}:\n\n{}",
        run.run_id,
        format_step_context(sop, run, step)
    )
}

pub fn load_tool_json(sop_dir: &Path, filename: &str) -> Result<Value> {
    let raw = std::fs::read_to_string(sop_dir.join(filename))
        .with_context(|| format!("read tool schema {}", sop_dir.join(filename).display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parse {filename}"))
}

pub fn load_sop_advance_tool(sop_dir: &Path) -> Result<Value> {
    load_tool_json(sop_dir, SOP_ADVANCE_FILE)
}

pub fn load_get_task_tool(sop_dir: &Path) -> Result<Value> {
    load_tool_json(sop_dir, GET_TASK_FILE)
}

pub fn load_delegate_tool(sop_dir: &Path) -> Result<Value> {
    load_tool_json(sop_dir, DELEGATE_FILE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sop::driver::minimal_run;
    use crate::sop::types::{SopExecutionMode, SopStepResult, SopStepStatus};

    fn minimal_sop() -> Sop {
        Sop {
            name: "test-sop".into(),
            description: String::new(),
            version: String::new(),
            priority: Default::default(),
            execution_mode: SopExecutionMode::Auto,
            triggers: Vec::new(),
            steps: vec![SopStep {
                number: 1,
                title: "Seed".into(),
                body: "Do the thing.".into(),
                suggested_tools: vec!["grep".into()],
                requires_confirmation: false,
                kind: Default::default(),
                schema: None,
                scope: None,
                routing: Default::default(),
                on_failure: Default::default(),
                mode: None,
            }],
            cooldown_secs: 0,
            max_concurrent: 1,
            location: None,
            deterministic: false,
        }
    }

    #[test]
    fn format_step_context_strips_gepa_anchors_from_body() {
        let sop = minimal_sop();
        let run = minimal_run(&sop);
        let mut step = sop.steps[0].clone();
        step.body = "{# GEPA:DISCOVER_START #}\nDo work.\n{# GEPA:DISCOVER_END #}".into();
        let out = format_step_context(&sop, &run, &step);
        assert!(!out.contains("GEPA:DISCOVER"));
        assert!(out.contains("Do work."));
    }

    #[test]
    fn loads_delegate_tool_schema() {
        // The shipped example SOP is single-agent (no delegate.json), so exercise
        // the loader against a representative schema written to a temp dir.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("delegate.json"),
            r#"{
              "name": "delegate",
              "description": "Hand a task to a named subagent.",
              "parameters": {
                "type": "object",
                "required": ["agent", "task"],
                "properties": {
                  "agent": { "type": "string", "enum": ["researcher", "worker"] },
                  "task": { "type": "string", "description": "What the subagent must produce." }
                }
              }
            }"#,
        )
        .unwrap();
        let t = load_delegate_tool(dir.path()).unwrap();
        assert_eq!(t["name"], "delegate");
        let required = t["parameters"]["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "agent"));
        assert!(required.iter().any(|v| v == "task"));
        // Both subagents are named in the schema so the orchestrator knows its options.
        let agents = t["parameters"]["properties"]["agent"]["enum"]
            .as_array()
            .unwrap();
        assert!(agents.iter().any(|v| v == "researcher"));
        assert!(agents.iter().any(|v| v == "worker"));
        let task_desc = t["parameters"]["properties"]["task"]["description"]
            .as_str()
            .unwrap();
        assert!(
            !task_desc.trim().is_empty(),
            "task description must guide the subagent: {task_desc}"
        );
    }

    #[test]
    fn format_step_context_matches_zeroclaw_shape() {
        let sop = minimal_sop();
        let mut run = minimal_run(&sop);
        run.step_results.push(SopStepResult {
            step_number: 1,
            status: SopStepStatus::Completed,
            output: r#"{"remaining": 2}"#.into(),
            started_at: String::new(),
            completed_at: None,
        });
        let out = format_step_context(&sop, &run, &sop.steps[0]);
        assert!(out.starts_with("[SOP: test-sop (run "));
        assert!(out.contains("Previous: Step 1 completed"));
        assert!(out.contains("Current step: **Seed**"));
        assert!(out.contains("Suggested tools: grep"));
        assert!(
            !out.contains("When this step is complete, call `sop_advance`"),
            "step body owns advance/finish; no generic footer"
        );
    }
}
