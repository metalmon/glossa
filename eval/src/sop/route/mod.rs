pub mod failure;
pub mod guard;

use super::condition::evaluate_condition;
use super::rundata::RunData;
use super::types::{Sop, SopRun, SopStep, SopStepStatus};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NextStep {
    Step(u32),
    Retry,
    Complete,
    Fail(String),
    Wait(u32),
}

pub struct RouteCtx<'a> {
    pub sop: &'a Sop,
    pub run: &'a SopRun,
    pub run_data: &'a RunData,
    pub last_status: SopStepStatus,
    pub max_step_visits: u32,
}

/// Pick the next step, preserving linear behavior when no routing is declared.
pub fn resolve_next(ctx: &RouteCtx<'_>) -> NextStep {
    if ctx.last_status == SopStepStatus::Failed {
        return NextStep::Fail("step failed".into());
    }

    let Some(current) = ctx
        .sop
        .steps
        .iter()
        .find(|step| step.number == ctx.run.current_step)
    else {
        return NextStep::Complete;
    };

    // `when` gates the loop back-edge, not the whole run: while the condition holds we
    // follow the explicit `next` (usually a self-loop); once it clears, the loop is done
    // and control falls through to the linear next step (current + 1), completing only if
    // that runs off the end. This lets a value-loop be followed by a review step. NOTE:
    // upstream zeroclaw ends the run on a false `when` — this is the local behaviour
    // proposed in zeroclaw-labs/zeroclaw#8719 and is a deliberate, backward-compatible
    // divergence (a tail-loop whose loop step is last still completes here, as current+1
    // runs off the end).
    let when_holds = current
        .routing
        .when
        .as_deref()
        .map(|w| evaluate_condition(w, Some(&ctx.run_data.to_payload().to_string())));
    let (next_step, followed_explicit) = if when_holds == Some(false) {
        (ctx.run.current_step.saturating_add(1), false)
    } else {
        let explicit_next = current.routing.next;
        (
            explicit_next.unwrap_or_else(|| ctx.run.current_step.saturating_add(1)),
            explicit_next.is_some(),
        )
    };
    let Some(step) = ctx.sop.steps.iter().find(|step| step.number == next_step) else {
        return if !followed_explicit && next_step > ctx.run.total_steps {
            NextStep::Complete
        } else {
            NextStep::Fail(format!("step {next_step} does not exist"))
        };
    };
    if !guard::within_visit_bound(ctx.run, next_step, ctx.max_step_visits) {
        return NextStep::Fail(format!("step {next_step} visit limit reached"));
    }

    if eligible(step, ctx.run_data) {
        NextStep::Step(next_step)
    } else {
        NextStep::Wait(next_step)
    }
}

/// A step is eligible when all declared dependencies have produced outputs.
pub fn eligible(step: &SopStep, run_data: &RunData) -> bool {
    step.routing
        .depends_on
        .iter()
        .all(|dependency| run_data.outputs.contains_key(dependency))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sop::step_contract::StepRouting;
    use crate::sop::types::{
        SopEvent, SopExecutionMode, SopPriority, SopRunStatus, SopStepKind, SopTriggerSource,
    };

    fn step(number: u32) -> SopStep {
        SopStep {
            number,
            title: format!("Step {number}"),
            body: String::new(),
            suggested_tools: Vec::new(),
            requires_confirmation: false,
            kind: SopStepKind::Execute,
            schema: None,
            scope: None,
            routing: StepRouting::default(),
            on_failure: Default::default(),
            mode: None,
        }
    }

    fn sop() -> Sop {
        Sop {
            name: "test".into(),
            description: "test".into(),
            version: "0.1.0".into(),
            priority: SopPriority::Normal,
            execution_mode: SopExecutionMode::Auto,
            triggers: Vec::new(),
            steps: vec![step(1), step(2)],
            cooldown_secs: 0,
            max_concurrent: 1,
            location: None,
            deterministic: false,
        }
    }

    fn run() -> SopRun {
        SopRun {
            run_id: "run".into(),
            sop_name: "test".into(),
            trigger_event: SopEvent {
                source: SopTriggerSource::Manual,
                topic: None,
                payload: None,
                timestamp: "now".into(),
            },
            frame_marker_id: "marker-run".into(),
            status: SopRunStatus::Running,
            current_step: 1,
            total_steps: 2,
            started_at: "now".into(),
            completed_at: None,
            step_results: Vec::new(),
            waiting_since: None,
            llm_calls_saved: 0,
        }
    }

    #[test]
    fn linear_default_routes_to_next_step() {
        let sop = sop();
        let run = run();
        let run_data = RunData::default();
        let ctx = RouteCtx {
            sop: &sop,
            run: &run,
            run_data: &run_data,
            last_status: SopStepStatus::Completed,
            max_step_visits: 256,
        };

        assert_eq!(resolve_next(&ctx), NextStep::Step(2));
    }

    #[test]
    fn dependency_without_output_waits() {
        let mut sop = sop();
        sop.steps[1].routing.depends_on = vec![1];
        let mut run = run();
        run.current_step = 1;
        let run_data = RunData::default();
        let ctx = RouteCtx {
            sop: &sop,
            run: &run,
            run_data: &run_data,
            last_status: SopStepStatus::Completed,
            max_step_visits: 256,
        };

        assert_eq!(resolve_next(&ctx), NextStep::Wait(2));
    }

    #[test]
    fn materialize_loop_exits_to_compile_when_remaining_zero() {
        use crate::sop::parse::parse_steps;
        use crate::sop::rundata::RunData;
        use crate::sop::types::SopStepResult;

        // Three plain steps with no routing metadata, like the fan-out SOP shape.
        let md = "\
## Steps

1. **Collect tables** — delegate the collection.
   - tools: delegate, sop_advance
2. **Schema** — write the manifest.
   - tools: note, sop_advance
3. **Compile** — build the graph.
   - tools: graph_build
";
        let mut sop = sop();
        sop.steps = parse_steps(md);
        sop.steps.truncate(3); // only need steps 1-3 for routing test

        let mut run = run();
        run.total_steps = 3;
        run.current_step = 2;
        run.step_results = vec![SopStepResult {
            step_number: 2,
            status: SopStepStatus::Completed,
            output: r#"{"remaining": 0}"#.into(),
            started_at: String::new(),
            completed_at: None,
        }];
        let run_data = RunData::from_step_results(&run.step_results);
        let ctx = RouteCtx {
            sop: &sop,
            run: &run,
            run_data: &run_data,
            last_status: SopStepStatus::Completed,
            max_step_visits: 256,
        };

        assert_eq!(
            resolve_next(&ctx),
            NextStep::Step(3),
            "remaining=0 must fall through to step 3 Compile, not Complete"
        );
    }
}
