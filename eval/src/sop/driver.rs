//! Thin SOP driver — the piece the eval provides in place of zeroclaw's `engine.rs`.
//!
//! The vendored `route::resolve_next` IS the state machine; this driver just turns
//! the crank: find the current step, hand it to a caller-supplied executor (which
//! runs a `run_episode`), record the result, and route to the next step until the
//! run completes. Step outputs accumulate in `RunData`, so `when:`/`$.path`
//! conditions in the SOP.md drive loops and branches exactly as they would under
//! the production engine.

use crate::sop::route::{resolve_next, NextStep, RouteCtx};
use crate::sop::rundata::RunData;
use crate::sop::route::guard::step_visit_count;
use crate::sop::types::{
    Sop, SopEvent, SopRun, SopRunStatus, SopStep, SopStepResult, SopStepStatus, SopTriggerSource,
};

/// What a step executor returns after running one step's work.
pub struct StepOutcome {
    pub status: SopStepStatus,
    /// The step's output as a JSON string (parsed into `RunData` for `$.path`
    /// conditions). Use `"{}"` when the step produces no structured output.
    pub output: String,
}

impl StepOutcome {
    pub fn completed(output: impl Into<String>) -> Self {
        Self { status: SopStepStatus::Completed, output: output.into() }
    }
    pub fn failed(output: impl Into<String>) -> Self {
        Self { status: SopStepStatus::Failed, output: output.into() }
    }
}

/// Loop bound per step (zeroclaw's default is 256): a self-routing step may run at
/// most this many times before the run fails, guarding against a runaway loop.
pub struct DriverConfig {
    pub max_step_visits: u32,
}
impl Default for DriverConfig {
    fn default() -> Self {
        Self { max_step_visits: 256 }
    }
}

/// The terminal state of a driven run.
pub struct RunReport {
    pub status: SopRunStatus,
    pub run_data: RunData,
    pub steps_run: usize,
    pub fail_reason: Option<String>,
}

pub fn minimal_run(sop: &Sop) -> SopRun {
    SopRun {
        run_id: format!("eval-{}", sop.name),
        sop_name: sop.name.clone(),
        trigger_event: SopEvent {
            source: SopTriggerSource::Manual,
            topic: None,
            payload: None,
            timestamp: String::new(),
        },
        frame_marker_id: String::new(),
        status: SopRunStatus::Running,
        current_step: sop.steps.first().map(|s| s.number).unwrap_or(1),
        total_steps: sop.steps.len() as u32,
        started_at: String::new(),
        completed_at: None,
        step_results: Vec::new(),
        waiting_since: None,
        llm_calls_saved: 0,
    }
}

/// Drive `sop` to a terminal state. `exec(step, run_data, visit)` runs one step's
/// work — `run_data` is the accumulated outputs so far, `visit` is how many times
/// this step has already run (0 on the first visit, for per-iteration state).
pub fn run_sop<F>(sop: &Sop, cfg: &DriverConfig, mut exec: F) -> RunReport
where
    F: FnMut(&SopStep, &RunData, u32) -> StepOutcome,
{
    let mut run = minimal_run(sop);
    let mut steps_run = 0usize;

    loop {
        let Some(step) = sop.steps.iter().find(|s| s.number == run.current_step).cloned() else {
            run.status = SopRunStatus::Completed;
            break;
        };
        let visit = step_visit_count(&run, run.current_step);
        let run_data = RunData::from_step_results(&run.step_results);

        let outcome = exec(&step, &run_data, visit);
        steps_run += 1;
        let last_status = outcome.status;
        run.step_results.push(SopStepResult {
            step_number: run.current_step,
            status: outcome.status,
            output: outcome.output,
            started_at: String::new(),
            completed_at: None,
        });

        let run_data = RunData::from_step_results(&run.step_results);
        let ctx = RouteCtx {
            sop,
            run: &run,
            run_data: &run_data,
            last_status,
            max_step_visits: cfg.max_step_visits,
        };
        match resolve_next(&ctx) {
            NextStep::Step(n) | NextStep::Wait(n) => run.current_step = n,
            NextStep::Retry => { /* same step number → re-run */ }
            NextStep::Complete => {
                run.status = SopRunStatus::Completed;
                break;
            }
            NextStep::Fail(reason) => {
                run.status = SopRunStatus::Failed;
                return RunReport {
                    status: SopRunStatus::Failed,
                    run_data: RunData::from_step_results(&run.step_results),
                    steps_run,
                    fail_reason: Some(reason),
                };
            }
        }
    }

    RunReport {
        status: run.status,
        run_data: RunData::from_step_results(&run.step_results),
        steps_run,
        fail_reason: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sop::parse::parse_steps;
    use crate::sop::types::{Sop, SopExecutionMode};

    fn sop_from_md(md: &str) -> Sop {
        Sop {
            name: "t".into(),
            description: String::new(),
            version: String::new(),
            priority: Default::default(),
            execution_mode: SopExecutionMode::Auto,
            triggers: Vec::new(),
            steps: parse_steps(md),
            cooldown_secs: 0,
            max_concurrent: 1,
            location: None,
            deterministic: false,
        }
    }

    /// Step 2 self-routes (`next: 2`) while `when: $.remaining > 0` holds, so the
    /// driver runs it once per remaining item — the per-list-element loop.
    #[test]
    fn driver_loops_step_over_a_list() {
        // Condition paths address a step's output as `$.steps.<n>.<field>`
        // (RunData::to_payload wraps outputs under a `steps` object).
        let md = "## Steps\n\n\
                  1. **Seed** — start.\n\n\
                  2. **Process one** — handle an item.\n   - when: $.steps.2.remaining > 0\n   - next: 2\n";
        let sop = sop_from_md(md);
        let total = 3;
        let mut processed = 0;
        let report = run_sop(&sop, &DriverConfig::default(), |step, _rd, _visit| {
            if step.number == 1 {
                StepOutcome::completed(format!("{{\"remaining\": {total}}}"))
            } else {
                processed += 1;
                let remaining = total - processed;
                StepOutcome::completed(format!("{{\"remaining\": {remaining}}}"))
            }
        });
        assert_eq!(report.status, SopRunStatus::Completed);
        assert_eq!(processed, total, "step 2 ran once per item");
    }
}
