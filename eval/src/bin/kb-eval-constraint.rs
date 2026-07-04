use anyhow::{Context, Result};
use clap::Parser;
use glossa::graph::ontology::Ontology;
use glossa::graph::agent::NodeUpdate;
use glossa::graph::store::{Edge, GraphStore, Node, Provenance};
use glossa::index::store::DocIndex;
use glossa::trace::TraceLog;
use glossa_constraint::solver::{SolveMode, SolveResult};
use kb_eval::backend::tensorzero::{run_episode, run_episode_gated, EpisodePolicy, TzTurn};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Duration;

/// List the knowledge base's documents and pick a primary. A product's
/// constraints are now assembled from several standards (the main GOST + the ones
/// it references), so solving unions across ALL of them (`load_problem(None)`);
/// the primary is only a representative label for reference provenance and the
/// prompt. `--doc` pins the primary to a specific indexed document.
fn resolve_source_doc(idx: &DocIndex, requested: Option<&str>) -> Result<(String, Vec<String>)> {
    let mut set = BTreeSet::new();
    idx.iter_chunks(|path, _, _, _| { set.insert(path.to_string()); })?;
    let docs: Vec<String> = set.into_iter().collect();
    if docs.is_empty() {
        anyhow::bail!("knowledge base has no indexed documents");
    }
    let primary = match requested {
        Some(doc) => idx.canonical_document_path(doc)
            .ok_or_else(|| anyhow::anyhow!("--doc {doc:?} is not an indexed document in the knowledge base"))?,
        None => docs[0].clone(),
    };
    Ok((primary, docs))
}

// ── CLI ──

#[derive(Parser)]
#[command(name = "kb-eval-constraint")]
struct Cli {
    #[arg(long, default_value = "http://localhost:3000")]
    gateway: String,
    #[arg(long, default_value = "eval/ontology-constraint.toml")]
    ontology: PathBuf,
    #[arg(long, default_value = "kb-gost")]
    kb: PathBuf,
    /// Directory with reference validation tables (xlsx-converted JSON).
    #[arg(long, default_value = "kb-val-gost")]
    val_dir: PathBuf,
    /// Source document (indexed path) constraints are extracted from.
    /// Defaults to the single document in the KB; required when the KB has several.
    #[arg(long)]
    doc: Option<String>,
    #[arg(long, default_value_t = 60)]
    timeout_secs: u64,
    #[arg(long = "tag", value_name = "KEY=VALUE")]
    tag: Vec<String>,
    #[arg(long)]
    csp_only: bool,
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// Enable the SOP-style completion gate: the episode is structured as
    /// numbered steps and `done` is rejected until every Field has an Enum with
    /// values and constraint_solve(check) is clean. Off = the plain agentic loop.
    #[arg(long)]
    sop: bool,
    /// Directory holding a zeroclaw-format SOP (SOP.toml + SOP.md). When set,
    /// discovery is driven step-by-step through the vendored SOP engine (one
    /// focused episode per step; step 2 loops per checklist parameter) instead of
    /// one all-at-once episode. Portable: the same directory runs under zeroclaw.
    #[arg(long)]
    sop_dir: Option<PathBuf>,
    /// Pin the TensorZero variant for `constraint_validate` (e.g. `qwen4b` for the
    /// local 4B, `qwen35` for the 35B). Pinned = that variant ONLY, no silent
    /// fallback to another on failure — so a run measures exactly one model.
    /// Omit to let the gateway pick (fallback enabled).
    #[arg(long)]
    variant: Option<String>,
}

#[derive(Clone)]
struct ColInfo {
    /// Human-readable parameter name (from the sheet header row / attribute
    /// dictionary) — this is what the agent can plausibly derive from the GOST.
    /// MDM GUIDs are translation keys only and never leave the loader.
    name: String,
    /// Unit of measure, split off by convert-xlsx ("Наружный диаметр [мм]" → "мм").
    unit: Option<String>,
    valid: Vec<String>,
}

struct Case {
    name: String,
    mode: SolveMode,
    assignments: Vec<(String, Value)>,
}

fn prov(src: &str) -> Provenance {
    Provenance { source_path: src.into(), range: None, file_sig: None, origin: "agent".into(), confidence: 1.0, created_at: 0 }
}

// ── Load validation tables ──

/// Stringify a table cell: numbers and bools become their text form (the MDM
/// export mixes "41" and 41 for the same parameter), empty/null cells are None.
fn cell_to_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Load the reference tables. The JSON is produced by `convert-xlsx`, which cuts
/// MDM UID columns and keys every column by its human-readable name — so this is
/// a plain read: rows are data, columns merge by name across files.
/// Files whose stem starts with `_` are metadata and skipped.
fn load_validation_data(val_dir: &std::path::Path) -> Result<(Vec<ColInfo>, Vec<BTreeMap<String, String>>)> {
    let mut col_map: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut unit_map: BTreeMap<String, String> = BTreeMap::new();
    let mut all_rows: Vec<BTreeMap<String, String>> = Vec::new();

    for entry in std::fs::read_dir(val_dir)? {
        let path = entry?.path();
        let stem = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        if path.extension().map_or(true, |e| e != "json") || stem.starts_with('_') {
            continue;
        }
        let data: Value = serde_json::from_reader(std::fs::File::open(&path)?)?;
        let tables = data["tables"].as_array().context("no tables array")?;

        for tbl in tables {
            if let Some(units) = tbl["units"].as_object() {
                for (name, u) in units {
                    if let Some(u) = u.as_str() {
                        unit_map.entry(name.clone()).or_insert_with(|| u.to_string());
                    }
                }
            }
            let rows = tbl["rows"].as_array().context("no rows array")?;
            for row in rows {
                let row = row.as_object().context("bad row")?;
                let mut clean = BTreeMap::new();
                for (name, v) in row {
                    let Some(val) = cell_to_string(v) else { continue };
                    clean.insert(name.clone(), val.clone());
                    col_map.entry(name.clone()).or_default().insert(val);
                }
                if !clean.is_empty() { all_rows.push(clean); }
            }
        }
    }

    let cols: Vec<ColInfo> = col_map.into_iter()
        .map(|(name, vals)| {
            let unit = unit_map.get(&name).cloned();
            ColInfo { name, unit, valid: vals.into_iter().collect() }
        })
        .collect();
    Ok((cols, all_rows))
}

// ── Reference graph ──

/// Compact constraint graph: one Field and one Enum per parameter, allowed
/// values carried as the Enum node's `aliases`. ~2 nodes + 1 edge per field
/// instead of a Literal node per value — the representation the agent targets.
fn build_reference_graph(g: &GraphStore, cols: &[ColInfo], src: &str) {
    for (ci, col) in cols.iter().enumerate() {
        if col.valid.len() <= 1 { continue; }
        let fld_id = format!("fld:{ci}");
        let enum_id = format!("enum:{ci}");
        g.put_node(&Node { id: fld_id.clone(), node_type: "Field".into(), label: col.name.clone(), aliases: vec![], prov: prov(src) }).unwrap();
        g.put_node(&Node { id: enum_id.clone(), node_type: "Enum".into(), label: format!("{} enum", col.name), aliases: col.valid.clone(), prov: prov(src) }).unwrap();
        g.put_edge(&Edge { from: fld_id, edge_type: "CONSTRAINED_BY".into(), to: enum_id, prov: prov(src) }).unwrap();
    }
}

fn export_graph(g: &GraphStore) -> Value {
    let nodes: Vec<Value> = g.all_nodes().unwrap_or_default().iter()
        .map(|n| json!({"id": n.id, "type": n.node_type, "label": n.label, "aliases": n.aliases}))
        .collect();
    let edges: Vec<Value> = g.all_edges().unwrap_or_default().iter()
        .map(|e| json!({"from": e.from, "edge_type": e.edge_type, "to": e.to}))
        .collect();
    json!({"nodes": nodes, "edges": edges})
}

// ── Test cases from table rows ──

fn generate_cases(cols: &[ColInfo], rows: &[BTreeMap<String, String>], limit: usize) -> Vec<Case> {
    let mut cases = Vec::new();
    let col_by_key: BTreeMap<&str, &ColInfo> = cols.iter().map(|c| (c.name.as_str(), c)).collect();

    let pick_invalid = |key: &str| -> Option<String> {
        col_by_key.get(key).and_then(|ci| {
            for test in &["INVALID", "999999", "ZZZZZ", "none", "0", "-1"] {
                if !ci.valid.iter().any(|v| v == test) { return Some(test.to_string()); }
            }
            None
        })
    };

    for (ri, row) in rows.iter().enumerate() {
        if limit > 0 && ri >= limit { break; }
        let name = format!("row{ri}");
        let assign: Vec<(String, Value)> = row.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect();
        cases.push(Case { name: format!("{name}_valid"), mode: SolveMode::Validate, assignments: assign.clone() });

        for k in row.keys().filter(|k| col_by_key.get(k.as_str()).map(|c| c.valid.len() > 1).unwrap_or(false)).take(3) {
            if let Some(bad) = pick_invalid(k) {
                let mut bad_assign: Vec<(String, Value)> = row.iter().filter(|(kk, _)| *kk != k).map(|(kk, vv)| (kk.clone(), Value::String(vv.clone()))).collect();
                bad_assign.push((k.clone(), Value::String(bad)));
                cases.push(Case { name: format!("{name}_invalid_{k}"), mode: SolveMode::Validate, assignments: bad_assign });
            }
        }
    }
    cases
}

// ── CSP solver ──

/// SOP-style completion gate (mirrors the zeroclaw engine's `validate_step_output`
/// + `finish_run`): the graph is "step-complete" only when every Field has a
/// CONSTRAINED_BY edge to an Enum with values and constraint_solve(check) reports
/// no issues. Returns `Some(feedback)` naming what is missing, or `None` to accept
/// `done`. Universal — reads only the agent's own graph, never the reference.
fn sop_gate_check(dir: &std::path::Path) -> Option<String> {
    let g = match GraphStore::open(dir) {
        Ok(g) => g,
        Err(_) => return Some("the constraint graph could not be opened.".into()),
    };
    let nodes = g.all_nodes().unwrap_or_default();
    let edges = g.all_edges().unwrap_or_default();

    // A Field is "complete" when it has a CONSTRAINED_BY edge to an Enum with values.
    let field_complete = |f: &Node| -> bool {
        edges.iter().any(|e| {
            e.edge_type == "CONSTRAINED_BY"
                && e.from == f.id
                && nodes.iter().any(|n| n.id == e.to && n.node_type == "Enum" && !n.aliases.is_empty())
        })
    };

    // Step 1: the Checklist node commits to the full parameter set.
    let checklist: Vec<&String> = nodes.iter()
        .filter(|n| n.node_type == "Checklist")
        .flat_map(|n| n.aliases.iter())
        .collect();
    if checklist.is_empty() {
        return Some("Step 1 is unmet: create ONE Checklist node whose aliases list EVERY parameter in the document's table.".into());
    }

    // Step 2: every checklist parameter must resolve to a complete Field.
    let uncovered: Vec<&str> = checklist.iter()
        .filter(|param| {
            let hit = g.resolve(param).ok().and_then(|ids| {
                ids.iter().find_map(|id| nodes.iter().find(|n| &n.id == id && n.node_type == "Field"))
                    .map(|f| field_complete(f))
            });
            !hit.unwrap_or(false)
        })
        .map(|s| s.as_str())
        .collect();
    if !uncovered.is_empty() {
        return Some(format!(
            "these checklist parameters still have no Field with a non-empty Enum (Step 2): {}.",
            uncovered.join(", ")
        ));
    }

    // Also reject any half-built Field that isn't on the checklist.
    let fields: Vec<&Node> = nodes.iter().filter(|n| n.node_type == "Field").collect();
    let incomplete: Vec<&str> = fields.iter()
        .filter(|f| !field_complete(f))
        .map(|f| f.label.as_str())
        .collect();
    if !incomplete.is_empty() {
        return Some(format!(
            "these Field(s) still have no Enum with allowed values (Step 2): {}.",
            incomplete.join(", ")
        ));
    }
    let ont = Ontology::load_or_default(dir);
    if let Ok(problem) = glossa::constraint_adapter::load_problem(&g, &ont, None) {
        let check = glossa_constraint::solver::solve(&problem, SolveMode::Check, &[]);
        if !check.issues.is_empty() {
            let first = check.issues.iter().take(3)
                .map(|i| format!("{}: {}", i.field, i.message))
                .collect::<Vec<_>>().join("; ");
            return Some(format!("constraint_solve(check) still reports issues (Step 3): {first}."));
        }
    }
    None
}

// ── Agent step execution (shared by the single-episode and SOP-driven paths) ──

/// The tool executor for an agent episode: search/read/grep/glob + the graph and
/// constraint tools, all bound to one agent graph dir and KB. A fresh one is made
/// per episode (the SOP driver runs one episode per step).
fn make_exec(
    agent_g_dir: std::path::PathBuf,
    kb: std::path::PathBuf,
    src_doc: String,
) -> impl Fn(&str, &Value) -> (String, Vec<String>, Vec<glossa::read::DocImage>) + Sync {
    let idx_kb = DocIndex::open_or_create(&kb).expect("open kb index");
    let spec_kb = glossa::tools::ChainSpec::from_ontology(&Ontology::load_or_default(&kb));
    let trace_kb = TraceLog::disabled();
    move |name: &str, args: &Value| {
        let g = GraphStore::open(&agent_g_dir).unwrap();
        let ont = Ontology::load_or_default(&agent_g_dir);
        match name {
            "search" | "read" | "grep" | "glob" => {
                kb_eval::backend::glossa_tools::exec(name, args, &idx_kb, Some(&g), &spec_kb, &trace_kb)
            }
            "graph_upsert" => (exec_graph_upsert(&idx_kb, &g, &ont, args), vec![], vec![]),
            "graph_delete" => (exec_graph_delete(&idx_kb, &g, args), vec![], vec![]),
            "graph_update" => (exec_graph_update(&g, args), vec![], vec![]),
            "graph_stats" => {
                // Same shared op as the MCP server: `doc` adds the checklist-coverage block.
                let mut out = glossa::tools::graph_stats(&g);
                if let Some(doc) = args.get("doc").and_then(|v| v.as_str()) {
                    out.push('\n');
                    out.push_str(&glossa::tools::checklist_coverage_report(&g, doc));
                }
                (out, vec![], vec![])
            }
            "graph_generalize" => {
                let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
                (glossa::graph::ops::graph_generalize(&g, &ont, now), vec![], vec![])
            }
            "constraint_solve" => {
                let sm = match args.get("mode").and_then(|v| v.as_str()).unwrap_or("validate") {
                    "infer" => SolveMode::Infer,
                    "check" => SolveMode::Check,
                    _ => SolveMode::Validate,
                };
                let assignments: Vec<(String, Value)> = args.get("field_assignments").and_then(|v| v.as_object())
                    .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();
                let problem = glossa::constraint_adapter::load_problem(&g, &ont, None).unwrap();
                let result = glossa_constraint::solver::solve(&problem, sm, &assignments);
                (glossa::constraint_adapter::format_solve_feedback(&problem, &result, &assignments, &src_doc), vec![], vec![])
            }
            "get_ontology" => {
                let relations: serde_json::Map<String, Value> = ont.raw_relations().iter()
                    .map(|(name, r)| (name.clone(), json!({"from": r.from, "to": r.to})))
                    .collect();
                let constraint_types: serde_json::Map<String, Value> = ont.constraint_types().iter()
                    .map(|(name, ct)| (name.clone(), json!({"params": ct.params})))
                    .collect();
                let node_types: Vec<&str> = ont.entity_types().iter().map(|s| s.as_str())
                    .chain(ont.constraint_types().keys().map(|s| s.as_str()))
                    .collect();
                (serde_json::to_string(&json!({
                    "node_types": node_types, "relations": relations,
                    "constraint_types": constraint_types, "strict": ont.strict(),
                })).unwrap_or_default(), vec![], vec![])
            }
            "done" => (json!({"status": "done"}).to_string(), vec![], vec![]),
            other => (format!("unknown tool: {other}"), vec![], vec![]),
        }
    }
}

/// A TensorZero chat closure for one episode against `constraint_validate`.
fn make_chat(
    url: String,
    fn_name: String,
    tags: Value,
    timeout: Duration,
    eid: String,
    variant: Option<String>,
) -> impl FnMut(&[Value], Option<&str>) -> Result<TzTurn> {
    move |messages: &[Value], ep: Option<&str>| {
        let e = ep.unwrap_or(&eid);
        let turn = kb_eval::tz::infer(&url, &fn_name, e, messages, &tags, timeout, variant.as_deref(), None, None)?;
        Ok(TzTurn { content: turn.content, episode_id: turn.episode_id })
    }
}

/// (first uncovered checklist parameter, count of uncovered) — a checklist param
/// is "covered" when it resolves to a Field with a non-empty Enum. Drives the SOP
/// build loop's `$.steps.N.remaining` condition.
/// The shared coverage op (`glossa::graph::ops::checklist_coverage`), scoped to
/// `doc`'s checklist + citation set — the SAME numbers `graph_stats(doc=…)`
/// reports to the agent, so the SOP gate and the model see one truth.
/// None until step 1 has created the checklist.
fn coverage(agent_g_dir: &std::path::Path, doc: &str) -> Option<glossa::graph::ops::ChecklistCoverage> {
    let g = GraphStore::open(agent_g_dir).ok()?;
    glossa::graph::ops::checklist_coverage(&g, doc).ok().flatten()
}

/// Parameters still needing work — unbuilt OR unmapped, in checklist order.
/// This is the build loop's gate metric: the loop keeps going until every
/// parameter has both its constraint (Field→Enum) and its source (DEFINED_IN).
fn pending_params(cov: &Option<glossa::graph::ops::ChecklistCoverage>) -> Vec<String> {
    match cov {
        Some(c) => c.params.iter()
            .filter(|p| c.unbuilt.contains(p) || c.unmapped.contains(p))
            .cloned().collect(),
        None => Vec::new(),
    }
}

/// The agent's step-output contract (the zeroclaw shape: the engine reads the
/// payload the AGENT produced — the driver adds no numbers of its own): the
/// episode's final answer must contain `"remaining": N`. The LAST occurrence
/// wins (the final self-check overrides earlier narration). None when the agent
/// failed to report — the SOP condition then fail-closes and the loop ends.
fn parse_reported_remaining(answer: &str) -> Option<usize> {
    let pos = answer.rfind("remaining")?;
    let digits: String = answer[pos..]
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// SOP-driven discovery, run exactly as it will run under zeroclaw: ONE
/// continuous agent conversation. The agent does a step's work with the tools,
/// then calls `sop_advance(status, output)` to finish it and receive the next
/// step's context — the engine routes on the agent's `output` payload
/// (`{"remaining": N}`), never on driver-computed numbers. `sop_advance` is
/// provided as a dynamic tool via `additional_tools`, mirroring the tool
/// zeroclaw's runtime injects; the same SOP.md therefore deploys unchanged.
///
/// The driver adds NO steering: it does not pick the target parameter, inject
/// hints, or compute the gate. It only routes the agent's report through the
/// vendored engine and surfaces the next step. Graph truth is logged next to the
/// agent's report purely as an observability signal (a gap is a quality metric).
#[allow(clippy::too_many_arguments)]
fn run_sop_conversation(
    sop_dir: &std::path::Path,
    agent_g_dir: &std::path::Path,
    kb: &std::path::Path,
    src_doc: &str,
    kb_docs_list: &str,
    gateway: &str,
    tags: &Value,
    timeout: Duration,
    variant: Option<&str>,
) -> Result<usize> {
    use kb_eval::sop;
    use sop::route::{resolve_next, NextStep, RouteCtx};
    use sop::types::{SopRunStatus, SopStep, SopStepResult, SopStepStatus};
    let sop_def = sop::load_sop(sop_dir, sop::types::SopExecutionMode::Auto)
        .with_context(|| format!("load SOP from {}", sop_dir.display()))?;
    let n_steps = sop_def.steps.len();
    eprintln!("[sop] loaded '{}' ({} steps, continuous conversation) from {}", sop_def.name, n_steps, sop_dir.display());
    let max_visits = sop::driver::DriverConfig::default().max_step_visits;

    // Dynamic tool injected via additional_tools — the same `sop_advance` the
    // zeroclaw runtime provides, so the SOP.md is portable as-is.
    let tools = [json!({
        "name": "sop_advance",
        "description": "Finish the CURRENT SOP step and receive the next one. Call it once the step's \
            work is done. `output` is a JSON-object string carrying the step's result; for these steps \
            it is {\"remaining\": N} where N is how many checklist parameters graph_stats(doc=…) still \
            lists as unbuilt or unmapped.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "completed | failed"},
                "output": {"type": "string", "description": "JSON result string, e.g. {\"remaining\": 3}"}
            },
            "required": ["status", "output"]
        }
    })];

    let step_ctx = |step: &SopStep| -> String {
        let tools_line = if step.suggested_tools.is_empty() { String::new() }
            else { format!("\nTools: {}.", step.suggested_tools.join(", ")) };
        format!(
            "── SOP step {} of {}: {} ──\n{}{}\n\n\
             When this step is done, call `sop_advance` with status=\"completed\" and \
             output=\"{{\\\"remaining\\\": N}}\".",
            step.number, n_steps, step.title, step.body, tools_line)
    };

    let intro = format!(
        "You are building the constraint set for regulatory document '{src_doc}'.\n\
         Knowledge base standards: {kb_docs_list}.\n\n\
         You are running an SOP as ONE continuous session. Do the current step's work with the tools, \
         then call `sop_advance` to finish it and receive the next step. When `sop_advance` replies \
         that the SOP is complete, call `done`.\n\n"
    );
    let first_prompt = format!("{intro}{}", step_ctx(&sop_def.steps[0]));

    // Shared SOP run state, mutated by the sop_advance handler inside the (Sync) exec.
    let run = std::sync::Mutex::new(sop::driver::minimal_run(&sop_def));
    let normal_exec = make_exec(agent_g_dir.to_path_buf(), kb.to_path_buf(), src_doc.to_string());

    let exec = |name: &str, args: &Value| -> (String, Vec<String>, Vec<glossa::read::DocImage>) {
        if name != "sop_advance" {
            return normal_exec(name, args);
        }
        let output = args.get("output").and_then(|v| v.as_str()).unwrap_or("{}").to_string();
        let status = match args.get("status").and_then(|v| v.as_str()) {
            Some("failed") => SopStepStatus::Failed,
            Some("skipped") => SopStepStatus::Skipped,
            _ => SopStepStatus::Completed,
        };
        let mut run = run.lock().unwrap();
        let cur = run.current_step;
        run.step_results.push(SopStepResult {
            step_number: cur, status, output: output.clone(),
            started_at: String::new(), completed_at: None,
        });
        let reported = parse_reported_remaining(&output).map(|n| n.to_string()).unwrap_or_else(|| "?".into());
        let truth = pending_params(&coverage(agent_g_dir, src_doc)).len();
        eprintln!("  [sop] step {cur} advanced → agent reports {reported} remaining (graph: {truth} pending)");
        // Route on the agent's payload — identical engine to zeroclaw.
        let run_data = sop::rundata::RunData::from_step_results(&run.step_results);
        let next = {
            let ctx = RouteCtx { sop: &sop_def, run: &run, run_data: &run_data, last_status: status, max_step_visits: max_visits };
            resolve_next(&ctx)
        };
        let msg = match next {
            NextStep::Step(n) | NextStep::Wait(n) => {
                run.current_step = n;
                match sop_def.steps.iter().find(|s| s.number == n) {
                    Some(s) => step_ctx(s),
                    None => { run.status = SopRunStatus::Completed; "SOP complete. Call `done` to finish.".into() }
                }
            }
            NextStep::Retry => sop_def.steps.iter().find(|s| s.number == cur)
                .map(step_ctx).unwrap_or_else(|| "SOP complete. Call `done`.".into()),
            NextStep::Complete => { run.status = SopRunStatus::Completed; "SOP complete — every step is done. Call `done` to finish.".into() }
            NextStep::Fail(r) => { run.status = SopRunStatus::Failed; format!("SOP failed: {r}. Call `done` to finish.") }
        };
        (msg, vec![], vec![])
    };

    let eid = kb_eval::tz::backdated_episode_id(30);
    let chat = move |messages: &[Value], ep: Option<&str>| -> Result<TzTurn> {
        let e = ep.unwrap_or(&eid);
        let turn = kb_eval::tz::infer(gateway, "constraint_validate", e, messages, tags, timeout, variant, None, Some(&tools))?;
        Ok(TzTurn { content: turn.content, episode_id: turn.episode_id })
    };

    // One continuous episode over the whole SOP (round budget covers all steps).
    let _ = run_episode(chat, &first_prompt, exec, 250, EpisodePolicy::enrich());
    let run = run.into_inner().unwrap();
    eprintln!("[sop] run {:?} after {} step-transitions", run.status, run.step_results.len());
    Ok(run.step_results.len())
}

fn solve_csp(dir: &std::path::Path, g: &GraphStore, mode: SolveMode, assignments: &[(String, Value)]) -> SolveResult {
    let ont = Ontology::load_or_default(dir);
    let problem = glossa::constraint_adapter::load_problem(g, &ont, None).unwrap();
    // Re-key onto the graph's Field labels so a paraphrased parameter name still
    // hits its constraint (matches the MCP tool's behaviour).
    let assignments = glossa::constraint_adapter::resolve_assignment_fields(g, &problem, assignments);
    glossa_constraint::solver::solve(&problem, mode, &assignments)
}

fn setup(dir: &std::path::Path, ontology_toml: &str, cols: &[ColInfo], src: &str) -> GraphStore {
    let glossa_dir = dir.join(".glossa");
    std::fs::create_dir_all(&glossa_dir).unwrap();
    std::fs::write(glossa_dir.join("ontology.toml"), ontology_toml).unwrap();
    let g = GraphStore::open(dir).unwrap();
    build_reference_graph(&g, cols, src);
    g
}

// ── Agent graph tools ──

/// Funnel through the same shared op as the MCP server (`glossa::graph::ops::graph_upsert`):
/// label-based nodes (canonical id derived from node_type+label), per-item validation with
/// actionable feedback instead of silently dropping malformed items.
fn exec_graph_upsert(idx: &DocIndex, g: &GraphStore, ont: &Ontology, args: &Value) -> String {
    fn parse_items<T: serde::de::DeserializeOwned>(args: &Value, key: &str, errs: &mut Vec<String>) -> Vec<T> {
        args.get(key).and_then(|v| v.as_array()).map(|arr| {
            arr.iter().enumerate().filter_map(|(i, item)| {
                match serde_json::from_value::<T>(item.clone()) {
                    Ok(v) => Some(v),
                    Err(e) => { errs.push(format!("{key}[{i}] dropped: {e}")); None }
                }
            }).collect()
        }).unwrap_or_default()
    }

    let mut errs: Vec<String> = Vec::new();
    let mut nodes: Vec<glossa::graph::ops::UpsertNode> = parse_items(args, "nodes", &mut errs);
    let edges: Vec<glossa::graph::ops::UpsertEdge> = parse_items(args, "edges", &mut errs);
    // The model sometimes sends ONE node as a flat object instead of {"nodes":[…]}
    // (same tolerance graph_update already has for its flat form). Without this the
    // call wrote nothing and the model misread the outcome as "this node_type is
    // invalid", derailing the whole episode.
    if nodes.is_empty() && edges.is_empty() {
        if let Ok(n) = serde_json::from_value::<glossa::graph::ops::UpsertNode>(args.clone()) {
            nodes.push(n);
        } else if errs.is_empty() {
            errs.push("nothing to write — graph_upsert takes {\"nodes\":[{node_type,label,source_path,…}], \"edges\":[{from,edge_type,to,source_path}]}".into());
        }
    }
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    let out = glossa::graph::ops::graph_upsert(idx, g, ont, nodes, edges, now);
    if errs.is_empty() {
        out.message
    } else {
        format!("{}\n{}", errs.join("\n"), out.message)
    }
}

fn exec_graph_delete(idx: &DocIndex, g: &GraphStore, args: &Value) -> String {
    let nodes: Vec<String> = args.get("nodes").and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|n| n.as_str().map(String::from)).collect())
        .unwrap_or_default();
    glossa::graph::ops::graph_delete(idx, g, nodes, vec![])
}

fn exec_graph_update(g: &GraphStore, args: &Value) -> String {
    if let (Some(label), Some(new_label)) = (args.get("label").and_then(|v| v.as_str()), args.get("new_label").and_then(|v| v.as_str())) {
        let new_type = args.get("new_type").and_then(|v| v.as_str());
        let nodes = vec![NodeUpdate { label: label.to_string(), new_label: Some(new_label.to_string()), new_type: new_type.map(String::from) }];
        glossa::graph::ops::graph_update(g, nodes)
    } else {
        "no valid update params".to_string()
    }
}

// ── Compare agent graph vs reference ──

/// Metric-side label normalisation: glossa's normalize_label plus stripping a
/// trailing bracketed unit — the MDM export writes "Наружный диаметр [мм]" and
/// "125 [мм]" where the GOST (and hence the agent) has "Наружный диаметр" / "125".
fn norm_metric(s: &str) -> String {
    let s = glossa::graph::store::normalize_label(s);
    match s.rfind('[') {
        Some(p) if s.ends_with(']') => s[..p].trim_end().to_string(),
        _ => s,
    }
}

/// Value normalisation for domain coverage: strip a trailing unit, then collapse
/// decimal notation via the solver's canonical form ("1,0"/"1.0"/"1" → "1"), so
/// the metric matches the same way the solver compares.
fn norm_value(s: &str) -> String {
    let stripped = match s.rfind('[') {
        Some(p) if s.trim_end().ends_with(']') => s[..p].trim_end().to_string(),
        _ => s.to_string(),
    };
    glossa_constraint::solver::canon_scalar(&stripped)
}

fn compare_graphs(agent_g: &GraphStore, ref_json: &Value) -> (f64, f64, f64) {
    let ref_nodes = ref_json["nodes"].as_array().map(|a| a.as_slice()).unwrap_or_default();
    let ref_edges = ref_json["edges"].as_array().map(|a| a.as_slice()).unwrap_or_default();

    let agent_nodes = agent_g.all_nodes().unwrap_or_default();
    let agent_edges = agent_g.all_edges().unwrap_or_default();

    // Field coverage BY DOMAIN, not by name: the GOST and the MDM reference name
    // the same parameter differently ("предельная рабочая скорость" vs "максимальная
    // скорость вращения"), so a name match under-counts. Instead a reference
    // parameter counts as covered when some agent Enum reproduces the majority of
    // its allowed-value set — the domain identifies the parameter, the label doesn't.
    let ref_id_enum: std::collections::HashMap<&str, BTreeSet<String>> = ref_nodes.iter()
        .filter(|n| n.get("type").and_then(|t| t.as_str()) == Some("Enum"))
        .filter_map(|n| {
            let id = n.get("id")?.as_str()?;
            let vals = n.get("aliases")?.as_array()?.iter().filter_map(|v| v.as_str()).map(norm_value).collect();
            Some((id, vals))
        })
        .collect();
    // (reference parameter label, its allowed-value set)
    let ref_params: Vec<(String, BTreeSet<String>)> = ref_nodes.iter()
        .filter(|n| n.get("type").and_then(|t| t.as_str()) == Some("Field"))
        .filter_map(|f| {
            let fid = f.get("id")?.as_str()?;
            let label = norm_metric(f.get("label")?.as_str()?);
            let enum_id = ref_edges.iter().find(|e| {
                e.get("edge_type").and_then(|t| t.as_str()) == Some("CONSTRAINED_BY")
                    && e.get("from").and_then(|x| x.as_str()) == Some(fid)
            }).and_then(|e| e.get("to").and_then(|x| x.as_str()))?;
            Some((label, ref_id_enum.get(enum_id).cloned().unwrap_or_default()))
        })
        .collect();
    let agent_domains: Vec<BTreeSet<String>> = agent_nodes.iter()
        .filter(|n| n.node_type == "Enum" && !n.aliases.is_empty())
        .map(|n| n.aliases.iter().map(|s| norm_value(s)).collect())
        .collect();
    // covered = some agent Enum holds ≥ half of this parameter's reference values.
    let covered = |dom: &BTreeSet<String>| -> bool {
        !dom.is_empty() && agent_domains.iter().any(|ad| {
            dom.iter().filter(|v| ad.contains(*v)).count() as f64 / dom.len() as f64 >= 0.5
        })
    };
    let field_cov = if ref_params.is_empty() { 1.0 }
        else { ref_params.iter().filter(|(_, d)| covered(d)).count() as f64 / ref_params.len() as f64 };

    // Value coverage: allowed values are the Enum nodes' aliases (compact form).
    // We also union any legacy Literal-node labels the agent may still emit, so a
    // graph in either representation is scored fairly.
    let ref_lits: BTreeSet<String> = ref_nodes.iter()
        .filter(|n| n.get("type").and_then(|t| t.as_str()) == Some("Enum"))
        .filter_map(|n| n.get("aliases").and_then(|a| a.as_array()))
        .flatten()
        .filter_map(|v| v.as_str())
        .map(norm_value)
        .collect();
    let agent_lits: BTreeSet<String> = agent_nodes.iter()
        .flat_map(|n| {
            let enum_vals = if n.node_type == "Enum" { n.aliases.clone() } else { vec![] };
            let legacy = if n.node_type == "Literal" { vec![n.label.clone()] } else { vec![] };
            enum_vals.into_iter().chain(legacy)
        })
        .map(|s| norm_value(&s))
        .collect();
    let literal_cov = if ref_lits.is_empty() { 1.0 }
        else { ref_lits.iter().filter(|l| agent_lits.contains(*l)).count() as f64 / ref_lits.len() as f64 };

    if std::env::var_os("KB_TRACE").is_some() {
        use std::collections::BTreeMap;
        let mut hist: BTreeMap<&str, usize> = BTreeMap::new();
        for n in &agent_nodes { *hist.entry(n.node_type.as_str()).or_default() += 1; }
        eprintln!("  [CMP] agent node types: {hist:?}");
        let a_sample: Vec<&String> = agent_lits.iter().take(12).collect();
        let r_sample: Vec<&String> = ref_lits.iter().take(12).collect();
        eprintln!("  [CMP] agent enum values (norm, {}): {a_sample:?}", agent_lits.len());
        eprintln!("  [CMP] ref   enum values (norm, {}): {r_sample:?}", ref_lits.len());
        let missing: Vec<&String> = ref_params.iter()
            .filter(|(_, d)| !covered(d))
            .map(|(label, _)| label)
            .collect();
        eprintln!("  [CMP] missing fields by domain ({}/{}): {missing:?}", missing.len(), ref_params.len());
    }

    // Edge coverage: shared edges (edge_type + from_label + to_label)
    // (approximate: count edge type matches)
    let ref_edge_types: BTreeSet<&str> = ref_edges.iter()
        .filter_map(|e| e.get("edge_type").and_then(|t| t.as_str()))
        .collect();
    let agent_edge_types: BTreeSet<&str> = agent_edges.iter()
        .map(|e| e.edge_type.as_str())
        .collect();
    let constraint_cov = if ref_edge_types.is_empty() { 1.0 }
        else { ref_edge_types.iter().filter(|t| agent_edge_types.contains(*t)).count() as f64 / ref_edge_types.len() as f64 };

    (field_cov, constraint_cov, literal_cov)
}

// ── Main ──

fn main() -> Result<()> {
    let cli = Cli::parse();
    let timeout = Duration::from_secs(cli.timeout_secs);
    let tags: Value = {
        let mut m = serde_json::Map::new();
        for t in &cli.tag {
            if let Some((k, v)) = t.split_once('=') {
                m.insert(k.trim().to_string(), json!(v.trim()));
            }
        }
        Value::Object(m)
    };

    let ontology_toml = std::fs::read_to_string(&cli.ontology).context("read ontology")?;
    Ontology::parse(&ontology_toml).context("ontology parse")?;

    let idx_kb = DocIndex::open_or_create(&cli.kb).context("open kb index")?;
    let (src_doc, kb_docs) = resolve_source_doc(&idx_kb, cli.doc.as_deref())?;
    drop(idx_kb);
    let kb_docs_list = kb_docs.join(", ");

    let (cols, rows) = load_validation_data(&cli.val_dir).context("load validation tables")?;
    eprintln!("Loaded {}: {} columns, {} valid rows", cli.val_dir.display(), cols.len(), rows.len());
    for col in &cols {
        eprintln!("  col {}: {} valid values", col.name, col.valid.len());
    }

    let cases = generate_cases(&cols, &rows, cli.limit);
    eprintln!("Test cases: {} ({} valid + {} invalid)", cases.len(),
        cases.iter().filter(|c| c.name.contains("valid")).count(),
        cases.iter().filter(|c| c.name.contains("invalid")).count());

    println!("constraint-eval  ontology={}  doc={src_doc}", cli.ontology.display());
    println!("  val_dir={}  cols={}  cases={}  csp_only={}  variant={}",
        cli.val_dir.display(), cols.len(), cases.len(), cli.csp_only,
        cli.variant.as_deref().unwrap_or("(gateway default — fallback ON)"));
    println!();

    // Build reference graph once (for CSP-only and for comparison)
    let ref_dir = tempfile::tempdir().context("ref tempdir")?;
    let ref_g = setup(ref_dir.path(), &ontology_toml, &cols, &src_doc);
    let ref_graph_json = export_graph(&ref_g);

    for (i, case) in cases.iter().enumerate() {
        if cli.limit > 0 && i >= cli.limit { break; }
        let mode_label = match case.mode {
            SolveMode::Validate => "validate",
            SolveMode::Infer => "infer",
            SolveMode::Check => "check",
        };

        // ── CSP-only: just solve against reference ──
        if cli.csp_only {
            let dir = tempfile::tempdir().context("csp tempdir")?;
            let g = setup(dir.path(), &ontology_toml, &cols, &src_doc);
            let t0 = std::time::Instant::now();
            let csp = solve_csp(dir.path(), &g, case.mode, &case.assignments);
            let csp_ms = t0.elapsed().as_millis();

            let expected_valid = !case.name.contains("invalid");
            let csp_ok = csp.valid == expected_valid;
            let marker = if csp_ok { "✓" } else { "✗" };

            print!("{marker}[{}/{}] {} {}  CSP: valid={} violations={} domains={} issues={} ({}ms)",
                i + 1, cases.len(), case.name, mode_label,
                csp.valid, csp.violations.len(),
                csp.domains.as_ref().map(|d| d.len()).unwrap_or(0),
                csp.issues.len(), csp_ms);
            if !csp_ok {
                let first = csp.violations.first().map(|v| format!(" {}={} ∉ {}", v.field, v.actual, v.expected)).unwrap_or_default();
                print!("  actual={} expected_valid={csp_ok}{first}", !csp.valid);
            }
            println!();
            continue;
        }

        // ── DISCOVERY (once): the agent builds the constraint graph; every row is
        //    then validated against it below. Runs on the first non-csp_only case,
        //    then breaks — the graph does not depend on the row.
        eprintln!("[discovery] building the constraint graph from the GOST...");
        let eid = kb_eval::tz::backdated_episode_id(30);
        let eid_chat = eid.clone();
        let url = cli.gateway.clone();
        let fn_name = "constraint_validate".to_string();
        let tz_tags = tags.clone();
        let ref_json_clone = ref_graph_json.clone();
        let tim = timeout;

        let variant_chat = cli.variant.clone();
        let chat = move |messages: &[Value], ep: Option<&str>| -> Result<TzTurn> {
            let eid = ep.unwrap_or(&eid_chat);
            let turn = kb_eval::tz::infer(&url, &fn_name, eid, messages, &tz_tags, tim, variant_chat.as_deref(), None, None)?;
            Ok(TzTurn { content: turn.content, episode_id: turn.episode_id })
        };

        let agent_dir = tempfile::tempdir().context("agent tempdir")?;
        let agent_g_dir = agent_dir.path().to_path_buf();
        let ont_path = agent_g_dir.join(".glossa").join("ontology.toml");
        std::fs::create_dir_all(agent_g_dir.join(".glossa")).unwrap();
        std::fs::write(&ont_path, &ontology_toml).unwrap();
        let _agent_init = GraphStore::open(&agent_g_dir).unwrap();

        let prompt = format!(
            "You are building the constraint set for a product specified by GOST '{src_doc}'.\n\n\
             === PHASE 1: Build constraint graph ===\n\
             The knowledge base holds these standards: {kb_docs_list}. A parameter's allowed\n\
             values may live in the main GOST or in a standard it references (e.g. grit, hardness,\n\
             marking are defined in referenced GOSTs) — use search/read across ALL of them.\n\
             To get the COMPLETE parameter set, find the product's designation example\n\
             ('условное обозначение' / 'Пример') — that one string enumerates every parameter in\n\
             order; enumerating from scattered value tables misses parameters.\n\
             Call get_ontology first to see the legal node types and relation signatures.\n\
             For EACH parameter create exactly two nodes: a Field node and an Enum node, linked\n\
             Field --CONSTRAINED_BY--> Enum. Put ALL of the parameter's allowed values into the\n\
             Enum node's `aliases` list — do NOT create a separate node per value.\n\
             A numeric parameter whose table lists specific allowed values (e.g. diameters\n\
             125, 150, ...) is an Enum of exactly those values, NOT a Range: a Range would admit\n\
             in-between values the standard forbids.\n\
             Batch: one graph_upsert call carries every Field/Enum node and edge at once.\n\n\
             === PHASE 2: Self-check ===\n\
             Call constraint_solve(mode=\"check\") to verify graph consistency.\n\n\
             === DONE ===\n\
             Call done(note=\"summary\") ONLY when: every parameter from the document is a Field,\n\
             each Field's allowed values are in its Enum node's aliases (an Enum with empty aliases\n\
             is a useless constraint), and constraint_solve(mode=\"check\") reports no issues."
        );
        // SOP mode: tell the model completion is gated, so it can't stop early.
        let prompt = if cli.sop {
            format!(
                "{prompt}\n\n\
                 === SOP ENFORCEMENT ===\n\
                 This task runs as an SOP with a gated completion:\n\
                 Step 1 (COMPLETE parameter set): find the product's designation / marking example\n\
                 in the GOST — search for 'условное обозначение' / 'Пример'. That single designation\n\
                 string enumerates EVERY parameter of the product in order, and the sentence right\n\
                 before it names each one. Read that sentence and that string, and take the full\n\
                 parameter list from them — it is authoritative and complete. Do NOT assemble the\n\
                 set by scanning value tables (you will miss parameters).\n\
                 Create ONE Checklist node whose `aliases` are the parameter names you read there.\n\
                 Step 2: for each checklist parameter create a Field and an Enum\n\
                 (Field --CONSTRAINED_BY--> Enum) with all allowed values in the Enum's aliases —\n\
                 the values may be in the main GOST or a referenced standard.\n\
                 Step 3: constraint_solve(check) must be clean.\n\
                 `done` is REJECTED — with the list of what is still missing — until every checklist\n\
                 parameter has a Field with a non-empty Enum and the check passes. Do not stop early."
            )
        } else {
            prompt
        };

        let agent_g_dir_clone = agent_g_dir.clone();
        let src_doc_exec = src_doc.clone();
        // Move kb resources into the exec closure
        let idx_kb = DocIndex::open_or_create(&cli.kb).context("open kb index")?;
        let spec_kb = glossa::tools::ChainSpec::from_ontology(&Ontology::load_or_default(&cli.kb));
        let trace_kb = TraceLog::disabled();

        let exec = move |name: &str, args: &Value| -> (String, Vec<String>, Vec<glossa::read::DocImage>) {
            let g = GraphStore::open(&agent_g_dir_clone).unwrap();
            let ont = Ontology::load_or_default(&agent_g_dir_clone);

            match name {
                "search" | "read" | "grep" | "glob" => {
                    kb_eval::backend::glossa_tools::exec(name, args, &idx_kb, Some(&g), &spec_kb, &trace_kb)
                }
                "graph_upsert" => (exec_graph_upsert(&idx_kb, &g, &ont, args), vec![], vec![]),
                "graph_delete" => (exec_graph_delete(&idx_kb, &g, args), vec![], vec![]),
                "graph_update" => (exec_graph_update(&g, args), vec![], vec![]),
                "graph_stats" => {
                // Same shared op as the MCP server: `doc` adds the checklist-coverage block.
                let mut out = glossa::tools::graph_stats(&g);
                if let Some(doc) = args.get("doc").and_then(|v| v.as_str()) {
                    out.push('\n');
                    out.push_str(&glossa::tools::checklist_coverage_report(&g, doc));
                }
                (out, vec![], vec![])
            }
                "graph_generalize" => {
                    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
                    (glossa::graph::ops::graph_generalize(&g, &ont, now), vec![], vec![])
                }
                "constraint_solve" => {
                    let sm = match args.get("mode").and_then(|v| v.as_str()).unwrap_or("validate") {
                        "infer" => SolveMode::Infer,
                        "check" => SolveMode::Check,
                        _ => SolveMode::Validate,
                    };
                    let assignments: Vec<(String, Value)> = args.get("field_assignments").and_then(|v| v.as_object())
                        .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                        .unwrap_or_default();
                    // Same feedback the MCP server gives: empty problems and
                    // unmatched assignment keys are called out, not hidden in JSON.
                    let problem = glossa::constraint_adapter::load_problem(&g, &ont, None).unwrap();
                    let result = glossa_constraint::solver::solve(&problem, sm, &assignments);
                    (glossa::constraint_adapter::format_solve_feedback(&problem, &result, &assignments, &src_doc_exec), vec![], vec![])
                }
                "get_ontology" => {
                    // Full machine-usable ontology: node types (constraint types ARE
                    // node types) and relation endpoint signatures — without the
                    // signatures the model has to guess legal edges by trial and error.
                    let relations: serde_json::Map<String, Value> = ont.raw_relations().iter()
                        .map(|(name, r)| (name.clone(), json!({"from": r.from, "to": r.to})))
                        .collect();
                    let constraint_types: serde_json::Map<String, Value> = ont.constraint_types().iter()
                        .map(|(name, ct)| (name.clone(), json!({"params": ct.params})))
                        .collect();
                    let node_types: Vec<&str> = ont.entity_types().iter().map(|s| s.as_str())
                        .chain(ont.constraint_types().keys().map(|s| s.as_str()))
                        .collect();
                    let info = json!({
                        "node_types": node_types,
                        "relations": relations,
                        "constraint_types": constraint_types,
                        "strict": ont.strict(),
                    });
                    (serde_json::to_string(&info).unwrap_or_default(), vec![], vec![])
                }
                // `done` is a control signal — run_episode intercepts it and never
                // reaches exec; this arm only guards against policy changes.
                "done" => (json!({"status": "done"}).to_string(), vec![], vec![]),
                other => (format!("unknown tool: {other}"), vec![], vec![]),
            }
        };

        let t1 = std::time::Instant::now();
        // 12 rounds starved the agent: reading the GOST + batched upserts for ~10
        // fields with hundreds of literals + solve + done needs room to work.
        let outcome = if let Some(sop_dir) = cli.sop_dir.clone() {
            // SOP-driven: the vendored engine drives one focused episode per step
            // (step 2 loops per checklist parameter). The single prompt/chat/exec
            // built above are unused in this path — the driver builds its own.
            let steps = run_sop_conversation(&sop_dir, &agent_g_dir, &cli.kb, &src_doc,
                &kb_docs_list, &cli.gateway, &tags, timeout, cli.variant.as_deref()).unwrap_or(0);
            Ok(kb_eval::backend::tensorzero::EpisodeOutcome {
                answer: String::new(), episode_id: None, surfaced_titles: vec![],
                done: true, rounds: steps,
            })
        } else if cli.sop {
            let gate_dir = agent_g_dir.clone();
            run_episode_gated(chat, &prompt, exec, 50, EpisodePolicy::enrich(),
                move || sop_gate_check(&gate_dir))
        } else {
            run_episode(chat, &prompt, exec, 50, EpisodePolicy::enrich())
        };
        let tz_ms = t1.elapsed().as_millis();

        // ── Post-episode: compare agent graph with reference ──
        let agent_g = GraphStore::open(&agent_g_dir).unwrap();
        let (field_cov, constraint_cov, literal_cov) = compare_graphs(&agent_g, &ref_json_clone);
        // Exclude the reference map (Standard nodes, DEFINED_IN edges) from the
        // counts — those measure the constraint graph, not the SOP scaffolding.
        let agent_nodes = agent_g.all_nodes().map(|v| v.iter().filter(|n| n.node_type != "Standard").count()).unwrap_or(0);
        let agent_edges = agent_g.all_edges().map(|v| v.iter().filter(|e| e.edge_type != "DEFINED_IN").count()).unwrap_or(0);
        let was_done = outcome.as_ref().map(|o| o.done).unwrap_or(false);
        let rounds = outcome.as_ref().map(|o| o.rounds).unwrap_or(0);

        // ── DISCOVERY coverage feedback (one build) ──
        let cli_gateway = &cli.gateway;
        kb_eval::tz::post_feedback(cli_gateway, &eid, "field_coverage", json!(field_cov), &tags);
        kb_eval::tz::post_feedback(cli_gateway, &eid, "constraint_coverage", json!(constraint_cov), &tags);
        kb_eval::tz::post_feedback(cli_gateway, &eid, "literal_coverage", json!(literal_cov), &tags);
        kb_eval::tz::post_feedback(cli_gateway, &eid, "agent_graph_node_count", json!(agent_nodes as f64), &tags);
        kb_eval::tz::post_feedback(cli_gateway, &eid, "agent_graph_edge_count", json!(agent_edges as f64), &tags);
        kb_eval::tz::post_feedback(cli_gateway, &eid, "tools_used", json!(rounds as f64), &tags);
        if let Err(e) = &outcome { println!("EPISODE ERROR: {e}"); }
        println!("DISCOVERY  done={was_done} rounds={rounds} tz={tz_ms}ms  graph: {agent_nodes} nodes/{agent_edges} edges  cov: f={field_cov:.2} c={constraint_cov:.2} l={literal_cov:.2}");

        // ── VALIDATION: sweep EVERY row through the AGENT graph (algorithmic, no LLM) ──
        // The GOST (agent) and MDM (rows) use different names for a parameter, so map
        // each MDM column to the agent field whose Enum domain matches it, then re-key
        // the row before solving.
        let agent_nodes_v = agent_g.all_nodes().unwrap_or_default();
        let agent_edges_v = agent_g.all_edges().unwrap_or_default();
        let agent_field_domains: Vec<(String, BTreeSet<String>)> = agent_nodes_v.iter()
            .filter(|n| n.node_type == "Field")
            .filter_map(|f| {
                let enum_id = agent_edges_v.iter()
                    .find(|e| e.edge_type == "CONSTRAINED_BY" && e.from == f.id)
                    .map(|e| e.to.clone())?;
                let dom: BTreeSet<String> = agent_nodes_v.iter()
                    .find(|n| n.id == enum_id && n.node_type == "Enum")
                    .map(|n| n.aliases.iter().map(|s| norm_value(s)).collect())?;
                Some((f.label.clone(), dom))
            })
            .collect();
        let mdm_to_agent: BTreeMap<String, String> = cols.iter().filter_map(|c| {
            let dom: BTreeSet<String> = c.valid.iter().map(|s| norm_value(s)).collect();
            if dom.is_empty() { return None; }
            agent_field_domains.iter()
                .find(|(_, ad)| dom.iter().filter(|v| ad.contains(*v)).count() as f64 / dom.len() as f64 >= 0.5)
                .map(|(label, _)| (c.name.clone(), label.clone()))
        }).collect();
        let agent_constrains = agent_edges_v.iter().any(|e| e.edge_type == "CONSTRAINED_BY");

        let (mut vpass, mut vtot, mut icatch, mut itot) = (0usize, 0usize, 0usize, 0usize);
        for c in &cases {
            let assign: Vec<(String, Value)> = c.assignments.iter()
                .map(|(k, v)| (mdm_to_agent.get(k).cloned().unwrap_or_else(|| k.clone()), v.clone()))
                .collect();
            let csp = solve_csp(&agent_g_dir, &agent_g, c.mode, &assign);
            let expected_valid = !c.name.contains("invalid");
            let ok = csp.valid == expected_valid;
            if expected_valid { vtot += 1; if ok { vpass += 1; } }
            else { itot += 1; if ok { icatch += 1; } }
        }
        let val_acc = if cases.is_empty() { 0.0 } else { (vpass + icatch) as f64 / cases.len() as f64 };
        let valid_pass_rate = if vtot == 0 { 0.0 } else { vpass as f64 / vtot as f64 };
        let invalid_catch_rate = if itot == 0 { 0.0 } else { icatch as f64 / itot as f64 };
        kb_eval::tz::post_feedback(cli_gateway, &eid, "validation_accuracy", json!(val_acc), &tags);
        kb_eval::tz::post_feedback(cli_gateway, &eid, "valid_pass_rate", json!(valid_pass_rate), &tags);
        kb_eval::tz::post_feedback(cli_gateway, &eid, "invalid_catch_rate", json!(invalid_catch_rate), &tags);
        println!("VALIDATION over {} rows (mapped {}/{} params)  acc={val_acc:.3}  valid_pass={vpass}/{vtot}  invalid_catch={icatch}/{itot}{}",
            cases.len(), mdm_to_agent.len(), cols.len(),
            if agent_constrains { "" } else { "  [WARN: agent graph has no constraints]" });
        break;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_reported_remaining_contract() {
        assert_eq!(parse_reported_remaining(r#"done. {"remaining": 7}"#), Some(7));
        assert_eq!(parse_reported_remaining("remaining: 0"), Some(0));
        // The LAST report wins — the final self-check overrides earlier narration.
        assert_eq!(parse_reported_remaining(r#"{"remaining": 9} … after the upsert: {"remaining": 3}"#), Some(3));
        assert_eq!(parse_reported_remaining("no report at all"), None);
        assert_eq!(parse_reported_remaining("remaining params: none"), None);
        assert_eq!(parse_reported_remaining(""), None);
    }

    fn put(g: &GraphStore, id: &str, nt: &str, label: &str, aliases: &[&str]) {
        g.put_node(&Node {
            id: id.into(),
            node_type: nt.into(),
            label: label.into(),
            aliases: aliases.iter().map(|s| s.to_string()).collect(),
            prov: prov("gost.docx"),
        }).unwrap();
    }
    fn edge(g: &GraphStore, from: &str, to: &str) {
        g.put_edge(&Edge { from: from.into(), to: to.into(), edge_type: "CONSTRAINED_BY".into(), prov: prov("gost.docx") }).unwrap();
    }

    /// The SOP gate blocks `done` until every checklist parameter has a complete
    /// Field, then clears — the completeness guarantee against the declared set.
    #[test]
    fn sop_gate_enforces_checklist_coverage() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();

        // No checklist → Step 1 unmet.
        assert!(sop_gate_check(dir.path()).unwrap().contains("Checklist"));

        // Checklist commits to two parameters; none built yet.
        put(&g, "chk:1", "Checklist", "parameters", &["высота", "диаметр"]);
        assert!(sop_gate_check(dir.path()).unwrap().contains("диаметр"));

        // Build "высота" fully; "диаметр" still uncovered.
        put(&g, "fld:h", "Field", "высота", &[]);
        put(&g, "enum:h", "Enum", "высота enum", &["1", "2"]);
        edge(&g, "fld:h", "enum:h");
        let r = sop_gate_check(dir.path()).unwrap();
        assert!(r.contains("диаметр") && !r.contains("высота"), "{r}");

        // Build "диаметр" fully → gate clears.
        put(&g, "fld:d", "Field", "диаметр", &[]);
        put(&g, "enum:d", "Enum", "диаметр enum", &["10", "20"]);
        edge(&g, "fld:d", "enum:d");
        assert!(sop_gate_check(dir.path()).is_none(), "all covered → done allowed");
    }
}
