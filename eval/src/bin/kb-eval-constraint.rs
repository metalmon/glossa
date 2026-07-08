use anyhow::{Context, Result};
use clap::Parser;
use glossa::graph::agent::NodeUpdate;
use glossa::graph::ontology::Ontology;
use glossa::graph::store::{Edge, GraphStore, Node, Provenance};
use glossa::index::store::DocIndex;
use glossa::trace::TraceLog;
use glossa_constraint::solver::{SolveMode, SolveResult};
use kb_eval::backend::tensorzero::{run_episode, EpisodeOutcome, EpisodePolicy, TzTurn};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// List the knowledge base's documents and pick a primary. A product's
/// constraints are now assembled from several standards (the main GOST + the ones
/// it references), so solving unions across ALL of them (`load_problem(None)`);
/// the primary is only a representative label for reference provenance and the
/// prompt. `--doc` pins the primary to a specific indexed document.
fn resolve_source_doc(idx: &DocIndex, requested: Option<&str>) -> Result<(String, Vec<String>)> {
    let mut set = BTreeSet::new();
    idx.iter_chunks(|path, _, _, _| {
        set.insert(path.to_string());
    })?;
    let docs: Vec<String> = set.into_iter().collect();
    if docs.is_empty() {
        anyhow::bail!("knowledge base has no indexed documents");
    }
    let primary = match requested {
        Some(doc) => idx.canonical_document_path(doc).ok_or_else(|| {
            anyhow::anyhow!("--doc {doc:?} is not an indexed document in the knowledge base")
        })?,
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
    #[arg(long, default_value = "kb-test")]
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
    /// Directory holding a zeroclaw-format SOP (SOP.toml + SOP.md). When set,
    /// discovery is driven step-by-step through the vendored SOP engine (one
    /// focused episode per step; step 2 loops per parameter) instead of
    /// one all-at-once episode. Portable: the same directory runs under zeroclaw.
    #[arg(long)]
    sop_dir: Option<PathBuf>,
    /// Pin the TensorZero variant for `constraint_validate` (e.g. `qwen4b` for the
    /// local 4B, `qwen35` for the 35B). Pinned = that variant ONLY, no silent
    /// fallback to another on failure — so a run measures exactly one model.
    /// Omit to let the gateway pick (fallback enabled).
    #[arg(long)]
    variant: Option<String>,
    /// Keep the agent workspace directory after the run (for inspecting notebook files).
    #[arg(long)]
    keep_agent_dir: Option<PathBuf>,
    /// Export agent notebook files after the episode (also auto-enabled when --tag run=… is set).
    #[arg(long)]
    export_notes: bool,
    /// Override export destination root (default: eval/results/<run>/agent or eval/results/agent/<episode-id>).
    #[arg(long)]
    export_notes_dir: Option<PathBuf>,
    /// Phase A: table coverage metrics only; skip tables-to-graph and CSP validation.
    #[arg(long, default_value_t = true)]
    tables_only: bool,
    /// Full pipeline: compile tables to graph and run CSP validation (overrides tables-only).
    #[arg(long)]
    full_pipeline: bool,
}

#[derive(Clone)]
struct ColInfo {
    /// Human-readable parameter name (from the sheet header row / attribute
    /// dictionary) — this is what the agent can plausibly derive from the GOST.
    /// MDM GUIDs are translation keys only and never leave the loader.
    name: String,
    valid: Vec<String>,
}

struct Case {
    name: String,
    mode: SolveMode,
    assignments: Vec<(String, Value)>,
}

fn prov(src: &str) -> Provenance {
    Provenance {
        source_path: src.into(),
        range: None,
        file_sig: None,
        origin: "agent".into(),
        confidence: 1.0,
        created_at: 0,
    }
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

type ValidationData = (Vec<ColInfo>, Vec<BTreeMap<String, String>>);

/// Load the reference tables. The JSON is produced by `convert-xlsx`, which cuts
/// MDM UID columns and keys every column by its human-readable name — so this is
/// a plain read: rows are data, columns merge by name across files.
/// Files whose stem starts with `_` are metadata and skipped.
fn load_validation_data(val_dir: &std::path::Path) -> Result<ValidationData> {
    let mut col_map: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut all_rows: Vec<BTreeMap<String, String>> = Vec::new();

    for entry in std::fs::read_dir(val_dir)? {
        let path = entry?.path();
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        if path.extension().is_none_or(|e| e != "json") || stem.starts_with('_') {
            continue;
        }
        let data: Value = serde_json::from_reader(std::fs::File::open(&path)?)?;
        let tables = data["tables"].as_array().context("no tables array")?;

        for tbl in tables {
            let rows = tbl["rows"].as_array().context("no rows array")?;
            for row in rows {
                let row = row.as_object().context("bad row")?;
                let mut clean = BTreeMap::new();
                for (name, v) in row {
                    let Some(val) = cell_to_string(v) else {
                        continue;
                    };
                    clean.insert(name.clone(), val.clone());
                    col_map.entry(name.clone()).or_default().insert(val);
                }
                if !clean.is_empty() {
                    all_rows.push(clean);
                }
            }
        }
    }

    let cols: Vec<ColInfo> = col_map
        .into_iter()
        // A column with a single value is document metadata (product name, the
        // accompanying-document reference, an abrasive flag), not a constrained
        // parameter — a one-value "domain" is nothing to model or measure. Keep only
        // columns whose values actually form a set the agent must reproduce.
        .filter(|(_, vals)| vals.len() >= 2)
        .map(|(name, vals)| ColInfo {
            name,
            valid: vals.into_iter().collect(),
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
        if col.valid.len() <= 1 {
            continue;
        }
        let fld_id = format!("fld:{ci}");
        let enum_id = format!("enum:{ci}");
        g.put_node(&Node {
            id: fld_id.clone(),
            node_type: "Field".into(),
            label: col.name.clone(),
            aliases: vec![],
            prov: prov(src),
        })
        .unwrap();
        g.put_node(&Node {
            id: enum_id.clone(),
            node_type: "Enum".into(),
            label: format!("{} enum", col.name),
            aliases: col.valid.clone(),
            prov: prov(src),
        })
        .unwrap();
        g.put_edge(&Edge {
            from: fld_id,
            edge_type: "CONSTRAINED_BY".into(),
            to: enum_id,
            prov: prov(src),
        })
        .unwrap();
    }
}

fn export_graph(g: &GraphStore) -> Value {
    let nodes: Vec<Value> = g
        .all_nodes()
        .unwrap_or_default()
        .iter()
        .map(|n| json!({"id": n.id, "type": n.node_type, "label": n.label, "aliases": n.aliases}))
        .collect();
    let edges: Vec<Value> = g
        .all_edges()
        .unwrap_or_default()
        .iter()
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
                if !ci.valid.iter().any(|v| v == test) {
                    return Some(test.to_string());
                }
            }
            None
        })
    };

    for (ri, row) in rows.iter().enumerate() {
        if limit > 0 && ri >= limit {
            break;
        }
        let name = format!("row{ri}");
        let assign: Vec<(String, Value)> = row
            .iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect();
        cases.push(Case {
            name: format!("{name}_valid"),
            mode: SolveMode::Validate,
            assignments: assign.clone(),
        });

        for k in row
            .keys()
            .filter(|k| {
                col_by_key
                    .get(k.as_str())
                    .map(|c| c.valid.len() > 1)
                    .unwrap_or(false)
            })
            .take(3)
        {
            if let Some(bad) = pick_invalid(k) {
                let mut bad_assign: Vec<(String, Value)> = row
                    .iter()
                    .filter(|(kk, _)| *kk != k)
                    .map(|(kk, vv)| (kk.clone(), Value::String(vv.clone())))
                    .collect();
                bad_assign.push((k.clone(), Value::String(bad)));
                cases.push(Case {
                    name: format!("{name}_invalid_{k}"),
                    mode: SolveMode::Validate,
                    assignments: bad_assign,
                });
            }
        }
    }
    cases
}

// ── CSP solver ──

// ── Agent step execution (shared by the single-episode and SOP-driven paths) ──

/// Execute notebook tools against the agent store (notes under `.glossa/notes/`).
fn exec_notebook(
    agent_g_dir: &std::path::Path,
    idx: &DocIndex,
    name: &str,
    args: &Value,
) -> String {
    match name {
        "note" => glossa::tools::note(
            agent_g_dir,
            idx,
            args.get("doc").and_then(|v| v.as_str()).unwrap_or(""),
            args.get("file").and_then(|v| v.as_str()).unwrap_or(""),
            args.get("content").and_then(|v| v.as_str()).unwrap_or(""),
            args.get("append")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        ),
        "cat" => glossa::tools::cat_note(
            agent_g_dir,
            idx,
            args.get("path").and_then(|v| v.as_str()).unwrap_or(""),
        ),
        "ls" => glossa::tools::ls_notes(agent_g_dir, idx, args.get("doc").and_then(|v| v.as_str())),
        "del" => glossa::tools::del_note(
            agent_g_dir,
            idx,
            args.get("path").and_then(|v| v.as_str()).unwrap_or(""),
        ),
        "sed" => glossa::tools::sed_note(
            agent_g_dir,
            idx,
            args.get("path").and_then(|v| v.as_str()).unwrap_or(""),
            args.get("old").and_then(|v| v.as_str()).unwrap_or(""),
            args.get("new").and_then(|v| v.as_str()).unwrap_or(""),
            args.get("all").and_then(|v| v.as_bool()).unwrap_or(false),
        ),
        other => format!("unknown notebook tool: {other}"),
    }
}

/// Remove a previous agent workspace overlay so stale notebook files and graph
/// state from an earlier run (or `--keep-agent-dir` reuse) cannot leak into scoring.
fn wipe_agent_glossa(agent_g_dir: &std::path::Path) -> std::io::Result<()> {
    let glossa = agent_g_dir.join(".glossa");
    if glossa.exists() {
        std::fs::remove_dir_all(&glossa)?;
    }
    Ok(())
}

static INTERRUPT_TEMP: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();

fn register_interrupt_temp_cleanup(path: PathBuf) {
    let slot = INTERRUPT_TEMP.get_or_init(|| Mutex::new(None));
    *slot.lock().expect("interrupt temp lock") = Some(path);
    static HANDLER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if HANDLER
        .compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::SeqCst,
            std::sync::atomic::Ordering::SeqCst,
        )
        .is_ok()
    {
        let _ = ctrlc::set_handler(|| {
            if let Some(slot) = INTERRUPT_TEMP.get() {
                if let Ok(mut guard) = slot.lock() {
                    if let Some(path) = guard.take() {
                        eprintln!("\n[agent] interrupted — removing {}", path.display());
                        let _ = std::fs::remove_dir_all(&path);
                    }
                }
            }
            std::process::exit(130);
        });
    }
}

fn unregister_interrupt_temp_cleanup() {
    if let Some(slot) = INTERRUPT_TEMP.get() {
        *slot.lock().expect("interrupt temp lock") = None;
    }
}

fn should_export_notes(cli: &Cli, tags: &Value) -> bool {
    cli.export_notes || tags.get("run").is_some()
}

fn resolve_export_notes_root(cli: &Cli, tags: &Value, episode_id: &str) -> Option<PathBuf> {
    if !should_export_notes(cli, tags) {
        return None;
    }
    if let Some(dir) = &cli.export_notes_dir {
        return Some(dir.clone());
    }
    if let Some(run) = tags.get("run").and_then(|v| v.as_str()) {
        return Some(PathBuf::from("eval/results").join(run).join("agent"));
    }
    Some(PathBuf::from("eval/results/agent").join(episode_id))
}

fn export_agent_notes(agent_g_dir: &Path, src_doc: &str, dst_root: &Path) -> Result<PathBuf> {
    let src = glossa::notebook::notes_root(agent_g_dir)
        .join(glossa::notebook::mirror_dir_for_doc(src_doc));
    if !src.exists() {
        anyhow::bail!("no notebook files at {}", src.display());
    }
    let dst = dst_root.join(glossa::notebook::mirror_dir_for_doc(src_doc));
    if dst.exists() {
        std::fs::remove_dir_all(&dst)
            .with_context(|| format!("clear export dir {}", dst.display()))?;
    }
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create export parent {}", parent.display()))?;
    }
    copy_dir_all(&src, &dst).with_context(|| format!("export notebook to {}", dst.display()))?;
    Ok(dst)
}

fn finalize_agent_workspace(
    agent_g_dir: &Path,
    src_doc: &str,
    export_dst: Option<&Path>,
    mut agent_temp: Option<tempfile::TempDir>,
    keep_agent_dir: bool,
) {
    if let Some(dst_root) = export_dst {
        match export_agent_notes(agent_g_dir, src_doc, dst_root) {
            Ok(dst) => println!("[agent] exported notebook → {}", dst.display()),
            Err(e) => eprintln!("[agent] export notes failed: {e:#}"),
        }
    }
    unregister_interrupt_temp_cleanup();
    if keep_agent_dir {
        return;
    }
    if let Some(temp) = agent_temp.take() {
        match temp.close() {
            Ok(()) => eprintln!("[agent] removed temp workspace"),
            Err(e) => eprintln!(
                "[agent] WARN: failed to remove temp workspace {}: {e}",
                agent_g_dir.display()
            ),
        }
    }
}

/// The tool executor for an agent episode.
fn copy_dir_all(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_all(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Eval-only: returns the document path pinned by `--doc` / CLI (not in prod MCP).
fn exec_get_task(doc: &str) -> String {
    json!({ "doc": doc }).to_string()
}

fn default_eval_sop_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("sops/gost-constraints")
}

/// The tool executor for an agent episode. Index AND graph open the SAME store
/// (`agent_g_dir`, a per-run copy of the indexed KB) — as in prod and the glossa
/// kb-train eval — so the document's Section nodes are present and a MENTIONS edge
/// resolves to a real section, not a fabricated node.
fn make_exec(
    agent_g_dir: std::path::PathBuf,
    src_doc: String,
) -> impl Fn(&str, &Value) -> (String, Vec<String>, Vec<glossa::read::DocImage>) + Sync {
    let idx_kb = DocIndex::open_or_create(&agent_g_dir).expect("open agent store index");
    let spec_kb = glossa::tools::ChainSpec::from_ontology(&Ontology::load_or_default(&agent_g_dir));
    let trace_kb = TraceLog::disabled();
    move |name: &str, args: &Value| {
        let g = GraphStore::open(&agent_g_dir).unwrap();
        let ont = Ontology::load_or_default(&agent_g_dir);
        match name {
            "get_task" => (exec_get_task(&src_doc), vec![], vec![]),
            "search" | "read" | "grep" | "glob" | "glossary" | "neighbors" | "resolve" => {
                kb_eval::backend::glossa_tools::exec(
                    name,
                    args,
                    &idx_kb,
                    Some(&g),
                    &spec_kb,
                    &trace_kb,
                )
            }
            "note" | "cat" | "ls" | "del" | "sed" => (
                exec_notebook(&agent_g_dir, &idx_kb, name, args),
                vec![],
                vec![],
            ),
            "index" | "reindex" => (
                match glossa::index::store::index_dir(&agent_g_dir, name == "reindex") {
                    Ok(s) => format!(
                        "indexed: {} added, {} removed, {} unchanged",
                        s.added, s.removed, s.unchanged
                    ),
                    Err(e) => format!("index error: {e}"),
                },
                vec![],
                vec![],
            ),
            "graph_upsert" => (exec_graph_upsert(&idx_kb, &g, &ont, args), vec![], vec![]),
            "graph_build" => {
                let doc = args.get("doc").and_then(|v| v.as_str()).unwrap_or(&src_doc);
                let tables_dir = args.get("tables_dir").and_then(|v| v.as_str());
                (
                    glossa::tools::graph_build(
                        &agent_g_dir,
                        &idx_kb,
                        &g,
                        &ont,
                        doc,
                        tables_dir.map(std::path::Path::new),
                    ),
                    vec![],
                    vec![],
                )
            }
            "graph_delete" => (exec_graph_delete(&idx_kb, &g, args), vec![], vec![]),
            "graph_update" => (exec_graph_update(&g, args), vec![], vec![]),
            "graph_stats" => {
                // Same contract as prod MCP: node mode inspects one node; doc mode adds
                // the per-Field graph coverage; otherwise the plain summary. No table
                // overlay here — table progress reaches the agent via note()'s .csp echo
                // and the sop_advance remaining-count.
                if let Some(node) = args.get("node").and_then(|v| v.as_str()) {
                    (glossa::tools::node_inspect(&g, node), vec![], vec![])
                } else {
                    let mut out = glossa::tools::graph_stats(&g);
                    if let Some(doc) = args.get("doc").and_then(|v| v.as_str()) {
                        out.push('\n');
                        out.push_str(&glossa::tools::checklist_coverage_report(&g, doc, &ont));
                    }
                    (out, vec![], vec![])
                }
            }
            "graph_generalize" => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                (
                    glossa::graph::ops::graph_generalize(&g, &ont, now),
                    vec![],
                    vec![],
                )
            }
            "constraint_solve" => {
                let sm = match args
                    .get("mode")
                    .and_then(|v| v.as_str())
                    .unwrap_or("validate")
                {
                    "infer" => SolveMode::Infer,
                    "check" => SolveMode::Check,
                    _ => SolveMode::Validate,
                };
                let assignments: Vec<(String, Value)> = args
                    .get("field_assignments")
                    .and_then(|v| v.as_object())
                    .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();
                let problem = glossa::constraint_adapter::load_problem(&g, &ont, None).unwrap();
                let result = glossa_constraint::solver::solve(&problem, sm, &assignments);
                (
                    glossa::constraint_adapter::format_solve_feedback(
                        &problem,
                        &result,
                        &assignments,
                        &src_doc,
                    ),
                    vec![],
                    vec![],
                )
            }
            "get_ontology" => (
                glossa::graph::ontology_export::export_pretty(&ont),
                vec![],
                vec![],
            ),
            "done" => (json!({"status": "done"}).to_string(), vec![], vec![]),
            other => (format!("unknown tool: {other}"), vec![], vec![]),
        }
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

/// Eval harness cap: step 1 may loop while workbook gaps remain; step 2 loops
/// once per parameter (~10) plus a small margin.
/// Production zeroclaw keeps `DriverConfig::default().max_step_visits` (256).
const SOP_EVAL_MAX_STEP_VISITS: u32 = 20;
/// Abort when the agent burns this many LLM turns on one SOP step without calling
/// `sop_advance` (typical stall: re-read / re-grep with no graph progress).
const SOP_MAX_LLM_ROUNDS_PER_STEP: usize = 75;

/// TensorZero may assign a different episode id on the first `/inference` response than
/// the client-generated UUID we send; `run_episode` tracks the gateway's id.
fn episode_id_for_report(fallback: &str, outcome: Option<&EpisodeOutcome>) -> String {
    outcome
        .and_then(|o| o.episode_id.as_deref())
        .filter(|id| !id.is_empty())
        .unwrap_or(fallback)
        .to_string()
}

/// Step 2 advanced 3+ times reporting the same positive `remaining` — no progress.
fn step2_advance_stuck(results: &[kb_eval::sop::types::SopStepResult]) -> Option<usize> {
    let reps: Vec<Option<usize>> = results
        .iter()
        .filter(|r| r.step_number == 2)
        .map(|r| parse_reported_remaining(&r.output))
        .collect();
    if reps.len() < 3 {
        return None;
    }
    let last = reps[reps.len() - 1]?;
    if last == 0 {
        return None;
    }
    if reps[reps.len() - 3..].iter().all(|r| *r == Some(last)) {
        Some(last)
    } else {
        None
    }
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
///
/// Eval-only guardrails (not in prod zeroclaw): per-step LLM round cap, progress
/// logging, step-2 stuck detection (3× same `remaining`), and a lower visit limit.
#[allow(clippy::too_many_arguments)]
fn run_sop_conversation(
    sop_dir: &std::path::Path,
    agent_g_dir: &std::path::Path,
    src_doc: &str,
    _kb_docs_list: &str,
    gateway: &str,
    tags: &Value,
    timeout: Duration,
    variant: Option<&str>,
) -> Result<EpisodeOutcome> {
    use kb_eval::sop;
    use sop::route::{resolve_next, NextStep, RouteCtx};
    use sop::types::{SopRunStatus, SopStepResult, SopStepStatus};
    let sop_def = sop::load_sop(sop_dir, sop::types::SopExecutionMode::Auto)
        .with_context(|| format!("load SOP from {}", sop_dir.display()))?;
    let n_steps = sop_def.steps.len();
    eprintln!(
        "[sop] loaded '{}' ({} steps, continuous conversation) from {}",
        sop_def.name,
        n_steps,
        sop_dir.display()
    );
    let max_visits = SOP_EVAL_MAX_STEP_VISITS;

    let get_task_tool = sop::prompt::load_get_task_tool(sop_dir)?;
    let sop_advance_tool = sop::prompt::load_sop_advance_tool(sop_dir)?;
    let eval_tools = [get_task_tool, sop_advance_tool];

    let llm_rounds_step = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let llm_rounds_total = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // Shared SOP run state, mutated by the sop_advance handler inside the (Sync) exec.
    let run = std::sync::Arc::new(std::sync::Mutex::new(sop::driver::minimal_run(&sop_def)));
    let first_prompt = {
        let run = run.lock().unwrap();
        sop::prompt::format_step_context(&sop_def, &run, &sop_def.steps[0])
    };

    let normal_exec = make_exec(agent_g_dir.to_path_buf(), src_doc.to_string());
    let llm_rounds_step_exec = std::sync::Arc::clone(&llm_rounds_step);

    let exec = {
        let run = std::sync::Arc::clone(&run);
        let sop_def = sop_def.clone();
        move |name: &str, args: &Value| -> (String, Vec<String>, Vec<glossa::read::DocImage>) {
            if name != "sop_advance" {
                return normal_exec(name, args);
            }
            llm_rounds_step_exec.store(0, std::sync::atomic::Ordering::Relaxed);
            let output = args
                .get("output")
                .and_then(|v| v.as_str())
                .unwrap_or("{}")
                .to_string();
            let status = match args.get("status").and_then(|v| v.as_str()) {
                Some("failed") => SopStepStatus::Failed,
                Some("skipped") => SopStepStatus::Skipped,
                _ => SopStepStatus::Completed,
            };
            let mut run = run.lock().unwrap();
            let cur = run.current_step;
            run.step_results.push(SopStepResult {
                step_number: cur,
                status,
                output: output.clone(),
                started_at: String::new(),
                completed_at: None,
            });
            let reported = parse_reported_remaining(&output)
                .map(|n| n.to_string())
                .unwrap_or_else(|| "?".into());
            eprintln!("  [sop] step {cur} advanced → agent reports {reported} remaining");
            if let Some(stuck_at) = step2_advance_stuck(&run.step_results) {
                eprintln!("  [sop] step 2 stuck at remaining={stuck_at} (3× sop_advance, no progress) — forcing step 3");
                run.current_step = 3;
                let msg = sop_def
                    .steps
                    .iter()
                    .find(|s| s.number == 3)
                    .map(|s| sop::prompt::format_sop_advance_reply(&sop_def, &run, s))
                    .unwrap_or_else(|| "SOP complete. Call `done` to finish.".into());
                return (msg, vec![], vec![]);
            }
            // Route on the agent's payload — identical engine to zeroclaw.
            let run_data = sop::rundata::RunData::from_step_results(&run.step_results);
            let next = {
                let ctx = RouteCtx {
                    sop: &sop_def,
                    run: &run,
                    run_data: &run_data,
                    last_status: status,
                    max_step_visits: max_visits,
                };
                resolve_next(&ctx)
            };
            let msg = match next {
                NextStep::Step(n) | NextStep::Wait(n) => {
                    run.current_step = n;
                    match sop_def.steps.iter().find(|s| s.number == n) {
                        Some(s) => sop::prompt::format_sop_advance_reply(&sop_def, &run, s),
                        None => {
                            run.status = SopRunStatus::Completed;
                            "SOP complete. Call `done` to finish.".into()
                        }
                    }
                }
                NextStep::Retry => sop_def
                    .steps
                    .iter()
                    .find(|s| s.number == cur)
                    .map(|s| sop::prompt::format_sop_advance_reply(&sop_def, &run, s))
                    .unwrap_or_else(|| "SOP complete. Call `done`.".into()),
                NextStep::Complete => {
                    run.status = SopRunStatus::Completed;
                    "SOP complete — every step is done. Call `done` to finish.".into()
                }
                NextStep::Fail(r) => {
                    run.status = SopRunStatus::Failed;
                    format!("SOP failed: {r}. Call `done` to finish.")
                }
            };
            (msg, vec![], vec![])
        }
    };

    let eid = kb_eval::tz::backdated_episode_id(30);
    let run_log = std::sync::Arc::clone(&run);
    let llm_rounds_step_chat = std::sync::Arc::clone(&llm_rounds_step);
    let llm_rounds_total_chat = std::sync::Arc::clone(&llm_rounds_total);
    let chat = move |messages: &[Value], ep: Option<&str>| -> Result<TzTurn> {
        let since_advance =
            llm_rounds_step_chat.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let total = llm_rounds_total_chat.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if total.is_multiple_of(10) {
            let step = run_log.lock().unwrap().current_step;
            eprintln!("  [sop] LLM round {total} step {step} ({since_advance} since sop_advance)");
        }
        if since_advance > SOP_MAX_LLM_ROUNDS_PER_STEP {
            let step = run_log.lock().unwrap().current_step;
            anyhow::bail!(
                "SOP step {step} exceeded {SOP_MAX_LLM_ROUNDS_PER_STEP} LLM rounds without sop_advance — aborting (partial graph kept for validation)"
            );
        }
        let e = ep.unwrap_or(&eid);
        let turn = kb_eval::tz::infer(
            gateway,
            "constraint_validate",
            e,
            messages,
            tags,
            timeout,
            variant,
            None,
            Some(&eval_tools),
        )?;
        Ok(TzTurn {
            content: turn.content,
            episode_id: turn.episode_id,
        })
    };

    // One continuous episode over the whole SOP (round budget covers all steps).
    // Dedup is temporarily OFF: it caches notebook-dependent reads (e.g. graph_stats,
    // cat) with no invalidation on note/sed/del, replaying stale output mid-run.
    // Re-enable once dedup is notebook-aware.
    let policy = EpisodePolicy {
        stop_on_done: true,
        dedup_readonly: false,
    };
    let outcome = run_episode(chat, &first_prompt, exec, 250, policy)?;
    let run = run.lock().unwrap();
    eprintln!(
        "[sop] run {:?} after {} step-transitions (episode done={} llm_rounds={})",
        run.status,
        run.step_results.len(),
        outcome.done,
        outcome.rounds
    );
    Ok(outcome)
}

fn solve_csp(
    dir: &std::path::Path,
    g: &GraphStore,
    mode: SolveMode,
    assignments: &[(String, Value)],
) -> SolveResult {
    let ont = Ontology::load_or_default(dir);
    let problem = glossa::constraint_adapter::load_problem(g, &ont, None).unwrap();
    // Re-key onto the graph's Field labels so a paraphrased parameter name still
    // hits its constraint (matches the MCP tool's behaviour).
    let assignments =
        glossa::constraint_adapter::resolve_assignment_fields(g, &problem, assignments);
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
    let (nodes, edges, notes) = glossa::graph::ops::parse_upsert_payload(args);
    let mut errs = notes;
    if nodes.is_empty() && edges.is_empty() && errs.is_empty() {
        errs.push("nothing to write — graph_upsert takes {\"nodes\":[{node_type,label,source_path,…}], \"edges\":[{from,edge_type,to,source_path}]}".into());
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let out = glossa::graph::ops::graph_upsert(idx, g, ont, nodes, edges, now);
    if errs.is_empty() {
        out.message
    } else {
        format!("{}\n{}", errs.join("\n"), out.message)
    }
}

fn exec_graph_delete(idx: &DocIndex, g: &GraphStore, args: &Value) -> String {
    let nodes: Vec<String> = args
        .get("nodes")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|n| n.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    glossa::graph::ops::graph_delete(idx, g, nodes, vec![])
}

fn exec_graph_update(g: &GraphStore, args: &Value) -> String {
    if let (Some(label), Some(new_label)) = (
        args.get("label").and_then(|v| v.as_str()),
        args.get("new_label").and_then(|v| v.as_str()),
    ) {
        let new_type = args.get("new_type").and_then(|v| v.as_str());
        let nodes = vec![NodeUpdate {
            label: label.to_string(),
            new_label: Some(new_label.to_string()),
            new_type: new_type.map(String::from),
        }];
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

/// A reference value is covered by an agent domain by exact membership, or because the
/// domain holds a regex PATTERN that the value matches (canon on both sides).
fn domain_covers(agent_dom: &BTreeSet<String>, ref_val: &str) -> bool {
    agent_dom.contains(ref_val)
        || agent_dom
            .iter()
            .any(|a| glossa_constraint::enum_alias_matches(a, ref_val))
}

fn compare_graphs(agent_g: &GraphStore, ref_json: &Value) -> (f64, f64, f64) {
    let ref_nodes = ref_json["nodes"]
        .as_array()
        .map(|a| a.as_slice())
        .unwrap_or_default();
    let ref_edges = ref_json["edges"]
        .as_array()
        .map(|a| a.as_slice())
        .unwrap_or_default();

    let agent_nodes = agent_g.all_nodes().unwrap_or_default();
    let agent_edges = agent_g.all_edges().unwrap_or_default();

    // Field coverage BY DOMAIN, not by name: the GOST and the MDM reference name
    // the same parameter differently ("предельная рабочая скорость" vs "максимальная
    // скорость вращения"), so a name match under-counts. Instead a reference
    // parameter counts as covered when some agent Enum reproduces the majority of
    // its allowed-value set — the domain identifies the parameter, the label doesn't.
    let ref_id_enum: std::collections::HashMap<&str, BTreeSet<String>> = ref_nodes
        .iter()
        .filter(|n| n.get("type").and_then(|t| t.as_str()) == Some("Enum"))
        .filter_map(|n| {
            let id = n.get("id")?.as_str()?;
            let vals = n
                .get("aliases")?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_str())
                .map(norm_value)
                .collect();
            Some((id, vals))
        })
        .collect();
    // (reference parameter label, its allowed-value set)
    let ref_params: Vec<(String, BTreeSet<String>)> = ref_nodes
        .iter()
        .filter(|n| n.get("type").and_then(|t| t.as_str()) == Some("Field"))
        .filter_map(|f| {
            let fid = f.get("id")?.as_str()?;
            let label = norm_metric(f.get("label")?.as_str()?);
            let enum_id = ref_edges
                .iter()
                .find(|e| {
                    e.get("edge_type").and_then(|t| t.as_str()) == Some("CONSTRAINED_BY")
                        && e.get("from").and_then(|x| x.as_str()) == Some(fid)
                })
                .and_then(|e| e.get("to").and_then(|x| x.as_str()))?;
            Some((label, ref_id_enum.get(enum_id).cloned().unwrap_or_default()))
        })
        .collect();
    let agent_domains: Vec<BTreeSet<String>> = agent_nodes
        .iter()
        .filter(|n| n.node_type == "Enum" && !n.aliases.is_empty())
        .map(|n| n.aliases.iter().map(|s| norm_value(s)).collect())
        .collect();
    // covered = some agent Enum holds ≥ half of this parameter's reference values.
    let covered = |dom: &BTreeSet<String>| -> bool {
        !dom.is_empty()
            && agent_domains.iter().any(|ad| {
                dom.iter().filter(|v| domain_covers(ad, v)).count() as f64 / dom.len() as f64 >= 0.5
            })
    };
    let field_cov = if ref_params.is_empty() {
        1.0
    } else {
        ref_params.iter().filter(|(_, d)| covered(d)).count() as f64 / ref_params.len() as f64
    };

    // Value coverage: allowed values are the Enum nodes' aliases (compact form).
    // We also union any legacy Literal-node labels the agent may still emit, so a
    // graph in either representation is scored fairly.
    let ref_lits: BTreeSet<String> = ref_nodes
        .iter()
        .filter(|n| n.get("type").and_then(|t| t.as_str()) == Some("Enum"))
        .filter_map(|n| n.get("aliases").and_then(|a| a.as_array()))
        .flatten()
        .filter_map(|v| v.as_str())
        .map(norm_value)
        .collect();
    let agent_lits: BTreeSet<String> = agent_nodes
        .iter()
        .flat_map(|n| {
            let enum_vals = if n.node_type == "Enum" {
                n.aliases.clone()
            } else {
                vec![]
            };
            let legacy = if n.node_type == "Literal" {
                vec![n.label.clone()]
            } else {
                vec![]
            };
            enum_vals.into_iter().chain(legacy)
        })
        .map(|s| norm_value(&s))
        .collect();
    let literal_cov = if ref_lits.is_empty() {
        1.0
    } else {
        ref_lits
            .iter()
            .filter(|l| domain_covers(&agent_lits, l))
            .count() as f64
            / ref_lits.len() as f64
    };

    if std::env::var_os("KB_TRACE").is_some() {
        use std::collections::BTreeMap;
        let mut hist: BTreeMap<&str, usize> = BTreeMap::new();
        for n in &agent_nodes {
            *hist.entry(n.node_type.as_str()).or_default() += 1;
        }
        eprintln!("  [CMP] agent node types: {hist:?}");
        let a_sample: Vec<&String> = agent_lits.iter().take(12).collect();
        let r_sample: Vec<&String> = ref_lits.iter().take(12).collect();
        eprintln!(
            "  [CMP] agent enum values (norm, {}): {a_sample:?}",
            agent_lits.len()
        );
        eprintln!(
            "  [CMP] ref   enum values (norm, {}): {r_sample:?}",
            ref_lits.len()
        );
        let missing: Vec<&String> = ref_params
            .iter()
            .filter(|(_, d)| !covered(d))
            .map(|(label, _)| label)
            .collect();
        eprintln!(
            "  [CMP] missing fields by domain ({}/{}): {missing:?}",
            missing.len(),
            ref_params.len()
        );
    }

    // Edge coverage: shared edges (edge_type + from_label + to_label)
    // (approximate: count edge type matches)
    let ref_edge_types: BTreeSet<&str> = ref_edges
        .iter()
        .filter_map(|e| e.get("edge_type").and_then(|t| t.as_str()))
        .collect();
    let agent_edge_types: BTreeSet<&str> =
        agent_edges.iter().map(|e| e.edge_type.as_str()).collect();
    let constraint_cov = if ref_edge_types.is_empty() {
        1.0
    } else {
        ref_edge_types
            .iter()
            .filter(|t| agent_edge_types.contains(*t))
            .count() as f64
            / ref_edge_types.len() as f64
    };

    (field_cov, constraint_cov, literal_cov)
}

/// Tables-only comparison, by domain: a reference parameter is identified among
/// the agent's `.csp` columns by its VALUE SET, never by name (same semantics
/// as `compare_graphs`' field/literal coverage — synonyms don't matter).
/// Returns ((params_covered, params_total), (values_covered, values_total)).
fn compare_tables_by_domain(
    agent_g_dir: &std::path::Path,
    src_doc: &str,
    cols: &[ColInfo],
) -> ((usize, usize), (usize, usize)) {
    let col_values = glossa::tables::csp_column_values(agent_g_dir, src_doc).unwrap_or_else(|e| {
        eprintln!("[tables] csp scan error: {e:#}");
        Default::default()
    });
    let agent_domains: Vec<BTreeSet<String>> = col_values
        .values()
        .map(|vals| vals.iter().map(|v| norm_value(v)).collect())
        .collect();
    // A parameter is covered when some single .csp column reproduces ≥ half of
    // its allowed-value set.
    let params_covered = cols
        .iter()
        .filter(|c| {
            let dom: BTreeSet<String> = c.valid.iter().map(|v| norm_value(v)).collect();
            !dom.is_empty()
                && agent_domains.iter().any(|ad| {
                    dom.iter().filter(|v| domain_covers(ad, v)).count() as f64 / dom.len() as f64
                        >= 0.5
                })
        })
        .count();
    // Value coverage: union of reference values vs union of all agent cells.
    let agent_union: BTreeSet<String> = agent_domains.into_iter().flatten().collect();
    let ref_union: BTreeSet<String> = cols
        .iter()
        .flat_map(|c| c.valid.iter())
        .map(|v| norm_value(v))
        .collect();
    let values_covered = ref_union
        .iter()
        .filter(|v| domain_covers(&agent_union, v))
        .count();
    (
        (params_covered, cols.len()),
        (values_covered, ref_union.len()),
    )
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
    eprintln!(
        "Loaded {}: {} columns, {} valid rows",
        cli.val_dir.display(),
        cols.len(),
        rows.len()
    );
    for col in &cols {
        eprintln!("  col {}: {} valid values", col.name, col.valid.len());
    }

    let cases = generate_cases(&cols, &rows, cli.limit);
    eprintln!(
        "Test cases: {} ({} valid + {} invalid)",
        cases.len(),
        cases.iter().filter(|c| c.name.contains("valid")).count(),
        cases.iter().filter(|c| c.name.contains("invalid")).count()
    );

    let tables_only = cli.tables_only && !cli.full_pipeline;
    let ref_value_total: usize = cols
        .iter()
        .flat_map(|c| c.valid.iter())
        .map(|v| norm_value(v))
        .collect::<BTreeSet<_>>()
        .len();
    println!(
        "TABLES ref  params={}  values={}  ({})",
        cols.len(),
        ref_value_total,
        cli.val_dir.display()
    );

    println!(
        "constraint-eval  ontology={}  doc={src_doc}",
        cli.ontology.display()
    );
    println!(
        "  val_dir={}  cols={}  cases={}  csp_only={}  tables_only={}  variant={}",
        cli.val_dir.display(),
        cols.len(),
        cases.len(),
        cli.csp_only,
        tables_only,
        cli.variant
            .as_deref()
            .unwrap_or("(gateway default — fallback ON)")
    );
    println!();

    // Build reference graph once (for CSP-only and for comparison)
    let ref_dir = tempfile::tempdir().context("ref tempdir")?;
    let ref_g = setup(ref_dir.path(), &ontology_toml, &cols, &src_doc);
    let ref_graph_json = export_graph(&ref_g);

    for (i, case) in cases.iter().enumerate() {
        if cli.limit > 0 && i >= cli.limit {
            break;
        }
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

            print!(
                "{marker}[{}/{}] {} {}  CSP: valid={} violations={} domains={} issues={} ({}ms)",
                i + 1,
                cases.len(),
                case.name,
                mode_label,
                csp.valid,
                csp.violations.len(),
                csp.domains.as_ref().map(|d| d.len()).unwrap_or(0),
                csp.issues.len(),
                csp_ms
            );
            if !csp_ok {
                let first = csp
                    .violations
                    .first()
                    .map(|v| format!(" {}={} ∉ {}", v.field, v.actual, v.expected))
                    .unwrap_or_default();
                print!("  actual={} expected_valid={csp_ok}{first}", !csp.valid);
            }
            println!();
            continue;
        }

        // ── DISCOVERY (once): the agent builds the constraint graph; every row is
        //    then validated against it below. Runs on the first non-csp_only case,
        //    then breaks — the graph does not depend on the row.
        eprintln!("[discovery] building the constraint graph...");
        let eid = kb_eval::tz::backdated_episode_id(30);
        let eid_chat = eid.clone();
        let url = cli.gateway.clone();
        let fn_name = "constraint_validate".to_string();
        let tz_tags = tags.clone();
        let ref_json_clone = ref_graph_json.clone();
        let tim = timeout;

        let eval_sop_dir = default_eval_sop_dir();
        let get_task_tool = kb_eval::sop::prompt::load_get_task_tool(&eval_sop_dir)
            .context("load get_task.json")?;
        let eval_tools = [get_task_tool.clone()];

        let variant_chat = cli.variant.clone();
        let chat = move |messages: &[Value], ep: Option<&str>| -> Result<TzTurn> {
            let eid = ep.unwrap_or(&eid_chat);
            let turn = kb_eval::tz::infer(
                &url,
                &fn_name,
                eid,
                messages,
                &tz_tags,
                tim,
                variant_chat.as_deref(),
                None,
                Some(&eval_tools),
            )?;
            Ok(TzTurn {
                content: turn.content,
                episode_id: turn.episode_id,
            })
        };

        let agent_temp = if cli.keep_agent_dir.is_some() {
            if let Some(ref keep) = cli.keep_agent_dir {
                std::fs::create_dir_all(keep).context("create keep-agent-dir")?;
            }
            None
        } else {
            Some(tempfile::tempdir().context("agent tempdir")?)
        };
        let agent_g_dir = cli
            .keep_agent_dir
            .clone()
            .unwrap_or_else(|| agent_temp.as_ref().unwrap().path().to_path_buf());
        if cli.keep_agent_dir.is_none() {
            register_interrupt_temp_cleanup(agent_g_dir.clone());
        }
        let export_notes_root = resolve_export_notes_root(&cli, &tags, &eid);
        let ont_path = agent_g_dir.join(".glossa").join("ontology.toml");
        // Seed the agent's store with a per-run copy of the indexed KB (index + graph,
        // including the Document/Section nodes) so index and graph share ONE store, as
        // in prod and the glossa kb-train eval. Runs stay isolated and the shared KB
        // is never mutated; wipe any prior `.glossa` first (keep-agent-dir reuse).
        wipe_agent_glossa(&agent_g_dir).context("wipe prior agent store")?;
        copy_dir_all(&cli.kb.join(".glossa"), &agent_g_dir.join(".glossa"))
            .context("seed agent store from KB")?;
        std::fs::write(&ont_path, &ontology_toml).unwrap();
        let _agent_init = GraphStore::open(&agent_g_dir).unwrap();

        let eval_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let prompt = std::fs::read_to_string(eval_dir.join("prompts/constraint-phase-a.md"))
            .context("read eval/prompts/constraint-phase-a.md")?;

        let agent_g_dir_clone = agent_g_dir.clone();
        let src_doc_exec = src_doc.clone();
        // Move kb resources into the exec closure
        let idx_kb = DocIndex::open_or_create(&cli.kb).context("open kb index")?;
        let spec_kb = glossa::tools::ChainSpec::from_ontology(&Ontology::load_or_default(&cli.kb));
        let trace_kb = TraceLog::disabled();

        let exec = move |name: &str,
                         args: &Value|
              -> (String, Vec<String>, Vec<glossa::read::DocImage>) {
            let g = GraphStore::open(&agent_g_dir_clone).unwrap();
            let ont = Ontology::load_or_default(&agent_g_dir_clone);

            match name {
                "get_task" => (exec_get_task(&src_doc_exec), vec![], vec![]),
                "search" | "read" | "grep" | "glob" | "glossary" | "neighbors" | "resolve" => {
                    kb_eval::backend::glossa_tools::exec(
                        name,
                        args,
                        &idx_kb,
                        Some(&g),
                        &spec_kb,
                        &trace_kb,
                    )
                }
                "note" | "cat" | "ls" | "del" | "sed" => {
                    let idx_agent = DocIndex::open_or_create(&agent_g_dir_clone).unwrap();
                    (
                        exec_notebook(&agent_g_dir_clone, &idx_agent, name, args),
                        vec![],
                        vec![],
                    )
                }
                "index" | "reindex" => (
                    match glossa::index::store::index_dir(&agent_g_dir_clone, name == "reindex") {
                        Ok(s) => format!(
                            "indexed: {} added, {} removed, {} unchanged",
                            s.added, s.removed, s.unchanged
                        ),
                        Err(e) => format!("index error: {e}"),
                    },
                    vec![],
                    vec![],
                ),
                "graph_upsert" => (exec_graph_upsert(&idx_kb, &g, &ont, args), vec![], vec![]),
                "graph_build" => {
                    let doc = args
                        .get("doc")
                        .and_then(|v| v.as_str())
                        .unwrap_or(&src_doc_exec);
                    let tables_dir = args.get("tables_dir").and_then(|v| v.as_str());
                    (
                        glossa::tools::graph_build(
                            &agent_g_dir_clone,
                            &idx_kb,
                            &g,
                            &ont,
                            doc,
                            tables_dir.map(std::path::Path::new),
                        ),
                        vec![],
                        vec![],
                    )
                }
                "graph_delete" => (exec_graph_delete(&idx_kb, &g, args), vec![], vec![]),
                "graph_update" => (exec_graph_update(&g, args), vec![], vec![]),
                "graph_stats" => {
                    // Same contract as prod MCP: pure graph statistics (no table overlay).
                    if let Some(node) = args.get("node").and_then(|v| v.as_str()) {
                        (glossa::tools::node_inspect(&g, node), vec![], vec![])
                    } else {
                        let mut out = glossa::tools::graph_stats(&g);
                        if let Some(doc) = args.get("doc").and_then(|v| v.as_str()) {
                            out.push('\n');
                            out.push_str(&glossa::tools::checklist_coverage_report(&g, doc, &ont));
                        }
                        (out, vec![], vec![])
                    }
                }
                "graph_generalize" => {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    (
                        glossa::graph::ops::graph_generalize(&g, &ont, now),
                        vec![],
                        vec![],
                    )
                }
                "constraint_solve" => {
                    let sm = match args
                        .get("mode")
                        .and_then(|v| v.as_str())
                        .unwrap_or("validate")
                    {
                        "infer" => SolveMode::Infer,
                        "check" => SolveMode::Check,
                        _ => SolveMode::Validate,
                    };
                    let assignments: Vec<(String, Value)> = args
                        .get("field_assignments")
                        .and_then(|v| v.as_object())
                        .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                        .unwrap_or_default();
                    // Same feedback the MCP server gives: empty problems and
                    // unmatched assignment keys are called out, not hidden in JSON.
                    let problem = glossa::constraint_adapter::load_problem(&g, &ont, None).unwrap();
                    let result = glossa_constraint::solver::solve(&problem, sm, &assignments);
                    (
                        glossa::constraint_adapter::format_solve_feedback(
                            &problem,
                            &result,
                            &assignments,
                            &src_doc_exec,
                        ),
                        vec![],
                        vec![],
                    )
                }
                "get_ontology" => (
                    glossa::graph::ontology_export::export_pretty(&ont),
                    vec![],
                    vec![],
                ),
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
            run_sop_conversation(
                &sop_dir,
                &agent_g_dir,
                &src_doc,
                &kb_docs_list,
                &cli.gateway,
                &tags,
                timeout,
                cli.variant.as_deref(),
            )
            .map_err(|e| {
                eprintln!("[sop] aborted: {e:#}");
                e
            })
        } else {
            // Dedup temporarily OFF (see the SOP path) — notebook writes don't
            // invalidate cached reads yet.
            run_episode(
                chat,
                &prompt,
                exec,
                50,
                EpisodePolicy {
                    stop_on_done: true,
                    dedup_readonly: false,
                },
            )
        };
        let tz_ms = t1.elapsed().as_millis();

        let agent_g = GraphStore::open(&agent_g_dir).unwrap();
        let was_done = outcome.as_ref().map(|o| o.done).unwrap_or(false);
        let rounds = outcome.as_ref().map(|o| o.rounds).unwrap_or(0);
        let reported_eid = episode_id_for_report(&eid, outcome.as_ref().ok());
        let cli_gateway = &cli.gateway;

        if tables_only {
            let ((pc, pt), (vc, vt)) = compare_tables_by_domain(&agent_g_dir, &src_doc, &cols);
            let csp_count = glossa::tables::count_csp_files(&agent_g_dir, &src_doc);
            let frac = |n: usize, d: usize| if d == 0 { 1.0 } else { n as f64 / d as f64 };
            kb_eval::tz::post_feedback(
                cli_gateway,
                &reported_eid,
                "ref_param_coverage",
                json!(frac(pc, pt)),
                &tags,
            );
            kb_eval::tz::post_feedback(
                cli_gateway,
                &reported_eid,
                "ref_value_coverage",
                json!(frac(vc, vt)),
                &tags,
            );
            kb_eval::tz::post_feedback(
                cli_gateway,
                &reported_eid,
                "table_csp_count",
                json!(csp_count as f64),
                &tags,
            );
            kb_eval::tz::post_feedback(
                cli_gateway,
                &reported_eid,
                "tools_used",
                json!(rounds as f64),
                &tags,
            );
            if let Err(e) = &outcome {
                println!("EPISODE ERROR: {e}");
            }
            println!("TABLES agent  params={pc}/{pt}  values={vc}/{vt}  csp={csp_count}");
            println!(
                "TABLES episode={reported_eid}  done={was_done} rounds={rounds} tz={tz_ms}ms  agent_dir={}",
                agent_g_dir.display()
            );
            drop(agent_g);
            finalize_agent_workspace(
                &agent_g_dir,
                &src_doc,
                export_notes_root.as_deref(),
                agent_temp,
                cli.keep_agent_dir.is_some(),
            );
            break;
        }

        // ── Compile agent .csp tables → constraint graph (if present) ──
        {
            let idx_agent = DocIndex::open_or_create(&agent_g_dir).unwrap();
            let ont = Ontology::load_or_default(&agent_g_dir);
            let tables_path = glossa::notebook::notes_root(&agent_g_dir)
                .join(glossa::notebook::mirror_dir_for_doc(&src_doc));
            // The mirror also holds workbook.md and free-form notes — only
            // compile when the agent actually wrote at least one .csp table.
            if glossa::tables::count_csp_files(&agent_g_dir, &src_doc) > 0 {
                eprintln!("[tables-to-graph] compiling {} …", tables_path.display());
                match glossa::tables::tables_to_graph(
                    &idx_agent,
                    &agent_g,
                    &ont,
                    &src_doc,
                    &tables_path,
                ) {
                    Ok(report) => {
                        for line in &report.lines {
                            eprintln!("  {line}");
                        }
                    }
                    Err(e) => eprintln!("[tables-to-graph] failed: {e:#}"),
                }
            }
        }

        // ── Post-episode: compare agent graph with reference ──
        let (field_cov, constraint_cov, literal_cov) = compare_graphs(&agent_g, &ref_json_clone);
        // Exclude the reference map (Standard nodes, DEFINED_IN edges) from the
        // counts — those measure the constraint graph, not the SOP scaffolding.
        let agent_nodes = agent_g
            .all_nodes()
            .map(|v| v.iter().filter(|n| n.node_type != "Standard").count())
            .unwrap_or(0);
        let agent_edges = agent_g
            .all_edges()
            .map(|v| v.iter().filter(|e| e.edge_type != "DEFINED_IN").count())
            .unwrap_or(0);

        // ── DISCOVERY coverage feedback (one build) ──
        kb_eval::tz::post_feedback(
            cli_gateway,
            &reported_eid,
            "field_coverage",
            json!(field_cov),
            &tags,
        );
        kb_eval::tz::post_feedback(
            cli_gateway,
            &reported_eid,
            "constraint_coverage",
            json!(constraint_cov),
            &tags,
        );
        kb_eval::tz::post_feedback(
            cli_gateway,
            &reported_eid,
            "literal_coverage",
            json!(literal_cov),
            &tags,
        );
        kb_eval::tz::post_feedback(
            cli_gateway,
            &reported_eid,
            "agent_graph_node_count",
            json!(agent_nodes as f64),
            &tags,
        );
        kb_eval::tz::post_feedback(
            cli_gateway,
            &reported_eid,
            "agent_graph_edge_count",
            json!(agent_edges as f64),
            &tags,
        );
        kb_eval::tz::post_feedback(
            cli_gateway,
            &reported_eid,
            "tools_used",
            json!(rounds as f64),
            &tags,
        );
        if let Err(e) = &outcome {
            println!("EPISODE ERROR: {e}");
        }
        println!(
            "DISCOVERY  episode={reported_eid}  done={was_done} rounds={rounds} tz={tz_ms}ms  agent_dir={}  graph: {agent_nodes} nodes/{agent_edges} edges  cov: f={field_cov:.2} c={constraint_cov:.2} l={literal_cov:.2}",
            agent_g_dir.display()
        );

        // ── VALIDATION: sweep EVERY row through the AGENT graph (algorithmic, no LLM) ──
        // The GOST (agent) and MDM (rows) use different names for a parameter, so map
        // each MDM column to the agent field whose constraint domain matches it, then re-key
        // the row before solving.
        let ont_v = Ontology::load_or_default(&agent_g_dir);
        let agent_nodes_v = agent_g.all_nodes().unwrap_or_default();
        let agent_edges_v = agent_g.all_edges().unwrap_or_default();
        let mut mdm_to_agent: BTreeMap<String, String> = BTreeMap::new();
        let mut mapping_parts: Vec<String> = Vec::new();
        let mut candidates: Vec<(String, String, f64)> = Vec::new();
        for c in &cols {
            let dom: BTreeSet<String> = c.valid.iter().map(|s| norm_value(s)).collect();
            if dom.is_empty() {
                continue;
            }
            for f in agent_nodes_v.iter().filter(|n| n.node_type == "Field") {
                let Ok(score) = glossa::constraint_adapter::field_reference_overlap(
                    &agent_g, &f.id, &ont_v, &dom,
                ) else {
                    continue;
                };
                if score >= 0.5 {
                    candidates.push((c.name.clone(), f.label.clone(), score));
                }
            }
        }
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        let mut used_cols: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut used_fields: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for (col, field, score) in candidates {
            if !used_cols.insert(col.clone()) || !used_fields.insert(field.clone()) {
                continue;
            }
            mdm_to_agent.insert(col.clone(), field.clone());
            mapping_parts.push(format!("{col} → «{field}» ({score:.2})"));
        }
        let unmapped: Vec<&str> = cols
            .iter()
            .filter(|c| !mdm_to_agent.contains_key(&c.name))
            .map(|c| c.name.as_str())
            .collect();
        println!(
            "VALIDATION mapping: {}; unmapped: {}",
            if mapping_parts.is_empty() {
                "—".into()
            } else {
                mapping_parts.join("; ")
            },
            if unmapped.is_empty() {
                "—".into()
            } else {
                unmapped.join(", ")
            }
        );
        let agent_constrains = agent_edges_v
            .iter()
            .any(|e| e.edge_type == "CONSTRAINED_BY");

        let (mut vpass, mut vtot, mut icatch, mut itot) = (0usize, 0usize, 0usize, 0usize);
        for c in &cases {
            let assign: Vec<(String, Value)> = c
                .assignments
                .iter()
                .map(|(k, v)| {
                    (
                        mdm_to_agent.get(k).cloned().unwrap_or_else(|| k.clone()),
                        v.clone(),
                    )
                })
                .collect();
            let csp = solve_csp(&agent_g_dir, &agent_g, c.mode, &assign);
            let expected_valid = !c.name.contains("invalid");
            let ok = csp.valid == expected_valid;
            if expected_valid {
                vtot += 1;
                if ok {
                    vpass += 1;
                }
            } else {
                itot += 1;
                if ok {
                    icatch += 1;
                }
            }
        }
        let val_acc = if cases.is_empty() {
            0.0
        } else {
            (vpass + icatch) as f64 / cases.len() as f64
        };
        let valid_pass_rate = if vtot == 0 {
            0.0
        } else {
            vpass as f64 / vtot as f64
        };
        let invalid_catch_rate = if itot == 0 {
            0.0
        } else {
            icatch as f64 / itot as f64
        };
        kb_eval::tz::post_feedback(
            cli_gateway,
            &reported_eid,
            "validation_accuracy",
            json!(val_acc),
            &tags,
        );
        kb_eval::tz::post_feedback(
            cli_gateway,
            &reported_eid,
            "valid_pass_rate",
            json!(valid_pass_rate),
            &tags,
        );
        kb_eval::tz::post_feedback(
            cli_gateway,
            &reported_eid,
            "invalid_catch_rate",
            json!(invalid_catch_rate),
            &tags,
        );
        println!("VALIDATION over {} rows (mapped {}/{} params)  acc={val_acc:.3}  valid_pass={vpass}/{vtot}  invalid_catch={icatch}/{itot}{}",
            cases.len(), mdm_to_agent.len(), cols.len(),
            if agent_constrains { "" } else { "  [WARN: agent graph has no constraints]" });
        drop(agent_g);
        finalize_agent_workspace(
            &agent_g_dir,
            &src_doc,
            export_notes_root.as_deref(),
            agent_temp,
            cli.keep_agent_dir.is_some(),
        );
        break;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step2_advance_stuck_detects_flat_remaining() {
        use kb_eval::sop::types::{SopStepResult, SopStepStatus};
        let mk = |n: usize| SopStepResult {
            step_number: 2,
            status: SopStepStatus::Completed,
            output: format!(r#"{{"remaining": {n}}}"#),
            started_at: String::new(),
            completed_at: None,
        };
        assert_eq!(step2_advance_stuck(&[mk(7), mk(7)]), None);
        assert_eq!(step2_advance_stuck(&[mk(7), mk(7), mk(7)]), Some(7));
        assert_eq!(step2_advance_stuck(&[mk(7), mk(7), mk(6)]), None);
        assert_eq!(step2_advance_stuck(&[mk(0), mk(0), mk(0)]), None);
    }

    #[test]
    fn episode_id_for_report_prefers_gateway_id() {
        let fallback = "019f3e27-c4e8-79a4-b4e1-479b31901782";
        let outcome = kb_eval::backend::tensorzero::EpisodeOutcome {
            answer: String::new(),
            episode_id: Some("019f3e27-c60d-7b4f-a135-74871c652565".into()),
            surfaced_titles: vec![],
            done: true,
            rounds: 49,
        };
        assert_eq!(
            episode_id_for_report(fallback, Some(&outcome)),
            "019f3e27-c60d-7b4f-a135-74871c652565"
        );
        assert_eq!(episode_id_for_report(fallback, None), fallback);
    }

    #[test]
    fn parse_reported_remaining_contract() {
        assert_eq!(
            parse_reported_remaining(r#"done. {"remaining": 7}"#),
            Some(7)
        );
        assert_eq!(parse_reported_remaining("remaining: 0"), Some(0));
        // The LAST report wins — the final self-check overrides earlier narration.
        assert_eq!(
            parse_reported_remaining(r#"{"remaining": 9} … after the upsert: {"remaining": 3}"#),
            Some(3)
        );
        assert_eq!(parse_reported_remaining("no report at all"), None);
        assert_eq!(parse_reported_remaining("remaining params: none"), None);
        assert_eq!(parse_reported_remaining(""), None);
    }

    #[test]
    fn export_agent_notes_copies_mirror() {
        let dir = tempfile::tempdir().unwrap();
        let doc = "kb-gost/test.pdf";
        let mirror = glossa::notebook::notes_root(dir.path())
            .join(glossa::notebook::mirror_dir_for_doc(doc));
        std::fs::create_dir_all(&mirror).unwrap();
        std::fs::write(mirror.join("t.csp"), "X\n1\n").unwrap();
        let out = tempfile::tempdir().unwrap();
        let dst = export_agent_notes(dir.path(), doc, out.path()).unwrap();
        assert!(dst.join("t.csp").exists());
    }

    #[test]
    fn resolve_export_notes_root_uses_run_tag() {
        let cli = Cli {
            gateway: String::new(),
            ontology: PathBuf::new(),
            kb: PathBuf::new(),
            val_dir: PathBuf::new(),
            doc: None,
            timeout_secs: 60,
            tag: vec!["run=deploy-test".into()],
            csp_only: false,
            limit: 0,
            sop_dir: None,
            variant: None,
            keep_agent_dir: None,
            export_notes: false,
            export_notes_dir: None,
            tables_only: true,
            full_pipeline: false,
        };
        let tags: Value = json!({"run": "deploy-test"});
        let root = resolve_export_notes_root(&cli, &tags, "ep1").unwrap();
        assert_eq!(root, PathBuf::from("eval/results/deploy-test/agent"));
    }

    #[test]
    fn wipe_agent_glossa_removes_notes_and_graph() {
        let dir = tempfile::tempdir().unwrap();
        let doc = "kb-gost/test.pdf";
        let mirror = glossa::notebook::notes_root(dir.path())
            .join(glossa::notebook::mirror_dir_for_doc(doc));
        std::fs::create_dir_all(&mirror).unwrap();
        std::fs::write(mirror.join("stale.csp"), "X\n1\n").unwrap();
        std::fs::create_dir_all(dir.path().join(".glossa")).unwrap();
        std::fs::write(dir.path().join(".glossa/graph.sqlite"), b"").unwrap();

        wipe_agent_glossa(dir.path()).unwrap();
        assert!(!dir.path().join(".glossa").exists());
    }

    #[test]
    fn tables_domain_compare_matches_by_values_not_names() {
        let dir = tempfile::tempdir().unwrap();
        let doc = "kb-gost/test.pdf";
        let mirror = glossa::notebook::notes_root(dir.path())
            .join(glossa::notebook::mirror_dir_for_doc(doc));
        std::fs::create_dir_all(&mirror).unwrap();
        // Column named differently from the reference parameter, but its value
        // set reproduces the reference domain → covered.
        std::fs::write(
            mirror.join("t.csp"),
            "Диаметр круга|Марка\n125|14A\n150|25A\n",
        )
        .unwrap();
        let cols = vec![
            ColInfo {
                name: "Наружный диаметр".into(),
                valid: vec!["125".into(), "150".into()],
            },
            ColInfo {
                name: "Высота".into(),
                valid: vec!["20".into(), "32".into()],
            },
        ];
        let ((pc, pt), (vc, vt)) = compare_tables_by_domain(dir.path(), doc, &cols);
        assert_eq!((pc, pt), (1, 2));
        assert_eq!((vc, vt), (2, 4));
    }

    #[test]
    fn tables_domain_compare_regex_alias_covers_values() {
        let dir = tempfile::tempdir().unwrap();
        let doc = "kb-gost/test.pdf";
        let mirror = glossa::notebook::notes_root(dir.path())
            .join(glossa::notebook::mirror_dir_for_doc(doc));
        std::fs::create_dir_all(&mirror).unwrap();
        // Agent wrote a regex PATTERN instead of enumerating marks — it must
        // cover concrete reference values like "14A" and "25A".
        std::fs::write(mirror.join("t.csp"), "Марка\n\\d+A\n").unwrap();
        let cols = vec![ColInfo {
            name: "Марка материала".into(),
            valid: vec!["14A".into(), "25A".into()],
        }];
        let ((pc, pt), (vc, vt)) = compare_tables_by_domain(dir.path(), doc, &cols);
        assert_eq!((pc, pt), (1, 1));
        assert_eq!((vc, vt), (2, 2));
    }

    #[test]
    fn tables_domain_compare_empty_when_no_csp() {
        let dir = tempfile::tempdir().unwrap();
        let cols = vec![ColInfo {
            name: "X".into(),
            valid: vec!["1".into()],
        }];
        let ((pc, _), (vc, _)) = compare_tables_by_domain(dir.path(), "kb-gost/none.pdf", &cols);
        assert_eq!(pc, 0);
        assert_eq!(vc, 0);
    }
}
