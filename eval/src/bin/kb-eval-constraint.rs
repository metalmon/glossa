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
    /// Re-score an existing --keep-agent-dir (its `.csp` notes) WITHOUT running the
    /// agent — fast metric-only iteration after a norm/metric change. Requires
    /// --keep-agent-dir to already exist with a `.glossa` store; skips the wipe/reseed
    /// so the prior run's notes are preserved.
    #[arg(long)]
    score_only: bool,
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

#[derive(Clone, Debug)]
struct RefTable {
    /// Gold file stem (for reporting).
    name: String,
    /// Columns that vary in this file (≥2 distinct values), name-sorted.
    params: Vec<String>,
    /// Deduped rows projected onto `params` (same order).
    rows: Vec<Vec<String>>,
}

type ValidationData = (Vec<ColInfo>, Vec<BTreeMap<String, String>>, Vec<RefTable>, Vec<String>);

/// Load the reference tables. The JSON is produced by `convert-xlsx`, which cuts
/// MDM UID columns and keys every column by its human-readable name — so this is
/// a plain read: rows are data, columns merge by name across files.
/// Files whose stem starts with `_` are metadata and skipped.
fn load_validation_data(val_dir: &std::path::Path) -> Result<ValidationData> {
    let mut col_map: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut all_rows: Vec<BTreeMap<String, String>> = Vec::new();
    let mut ref_tables: Vec<RefTable> = Vec::new();

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

        // Per-file accumulation so we can recover relational (multi-column) tables.
        let mut file_cols: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut file_rows: Vec<BTreeMap<String, String>> = Vec::new();

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
                    col_map.entry(name.clone()).or_default().insert(val.clone());
                    file_cols.entry(name.clone()).or_default().insert(val);
                }
                if !clean.is_empty() {
                    all_rows.push(clean.clone());
                    file_rows.push(clean);
                }
            }
        }

        // A dependent table = ≥2 columns that each vary (≥2 distinct values).
        let params: Vec<String> = file_cols
            .iter()
            .filter(|(_, vals)| vals.len() >= 2)
            .map(|(name, _)| name.clone())
            .collect();
        if params.len() >= 2 {
            let mut seen: BTreeSet<Vec<String>> = BTreeSet::new();
            let mut rows: Vec<Vec<String>> = Vec::new();
            for r in &file_rows {
                let tuple: Vec<String> = params
                    .iter()
                    .map(|p| r.get(p).cloned().unwrap_or_default())
                    .collect();
                if tuple.iter().all(|c| !c.is_empty()) && seen.insert(tuple.clone()) {
                    rows.push(tuple);
                }
            }
            // Skip tables where every row is missing at least one projected cell
            // (rows ends up empty) — an empty table would otherwise render as
            // spuriously "covered" downstream.
            if !rows.is_empty() {
                ref_tables.push(RefTable { name: stem, params, rows });
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
    let canon_order = glossa::tables::read_schema_order(&val_dir.join("schema.toml")).params;
    Ok((cols, all_rows, ref_tables, canon_order))
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
        "ls" => glossa::tools::ls_notes(agent_g_dir, idx, args.get("doc").and_then(|v| v.as_str())),
        "del" => glossa::tools::del_note(
            agent_g_dir,
            idx,
            args.get("path").and_then(|v| v.as_str()).unwrap_or(""),
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
                    &agent_g_dir,
                    &idx_kb,
                    Some(&g),
                    &spec_kb,
                    &trace_kb,
                )
            }
            "note" | "ls" | "del" => (
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
                // Same contract as prod MCP (see mcp.rs): an arg naming a DOCUMENT —
                // under `doc` OR `node`, resolved leniently — routes to that document's
                // owned-node inventory (+ MENTIONS targets to `read`). A real
                // node id gets node-inspection; nothing → the plain summary.
                let doc = args
                    .get("doc")
                    .and_then(|v| v.as_str())
                    .map(|d| idx_kb.canonical_document_path(d).unwrap_or_else(|| d.to_string()))
                    .or_else(|| {
                        args.get("node")
                            .and_then(|v| v.as_str())
                            .and_then(|s| idx_kb.canonical_document_path(s))
                    });
                if let Some(doc) = doc {
                    let mut out = glossa::tools::graph_stats(&g);
                    out.push('\n');
                    out.push_str(&glossa::tools::checklist_coverage_report(&g, &doc));
                    (out, vec![], vec![])
                } else if let Some(node) = args.get("node").and_then(|v| v.as_str()) {
                    (glossa::tools::node_inspect(&g, node), vec![], vec![])
                } else {
                    (glossa::tools::graph_stats(&g), vec![], vec![])
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

/// Per-step subagent round budget: each SOP step runs at most this many LLM turns before its
/// episode is cut and the driver advances to the next step.
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
    use sop::types::{SopRunStatus, SopStepResult, SopStepStatus};
    let sop_def = sop::load_sop(sop_dir, sop::types::SopExecutionMode::Auto)
        .with_context(|| format!("load SOP from {}", sop_dir.display()))?;
    let n_steps = sop_def.steps.len();
    eprintln!(
        "[sop] loaded '{}' ({} steps, per-step subagents) from {}",
        sop_def.name,
        n_steps,
        sop_dir.display()
    );

    let get_task_tool = sop::prompt::load_get_task_tool(sop_dir)?;
    let sop_advance_tool = sop::prompt::load_sop_advance_tool(sop_dir)?;
    let delegate_tool = sop::prompt::load_delegate_tool(sop_dir)?;
    // `get_task_tool`/`delegate_tool` stay in scope for the fan-out orchestrator's tool set below.
    let eval_tools = [get_task_tool.clone(), sop_advance_tool];

    let llm_rounds_step = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let llm_rounds_total = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // Shared SOP run state, mutated by the sop_advance handler inside the (Sync) exec.
    let run = std::sync::Arc::new(std::sync::Mutex::new(sop::driver::minimal_run(&sop_def)));

    let normal_exec = make_exec(agent_g_dir.to_path_buf(), src_doc.to_string());
    let llm_rounds_step_exec = std::sync::Arc::clone(&llm_rounds_step);

    let exec = {
        let run = std::sync::Arc::clone(&run);
        move |name: &str, args: &Value| -> (String, Vec<String>, Vec<glossa::read::DocImage>) {
            if name != "sop_advance" {
                return normal_exec(name, args);
            }
            // Intra-step progress signal (e.g. Materialize's table-by-table loop). It does NOT
            // change the step — step transitions happen on `done` (driver loop below). Reset the
            // per-step round counter so a step that keeps making progress isn't cut by the cap.
            llm_rounds_step_exec.store(0, std::sync::atomic::Ordering::Relaxed);
            let output = args
                .get("output")
                .and_then(|v| v.as_str())
                .unwrap_or("{}")
                .to_string();
            let reported = parse_reported_remaining(&output)
                .map(|n| n.to_string())
                .unwrap_or_else(|| "?".into());
            let cur = run.lock().unwrap().current_step;
            eprintln!("  [sop] step {cur} progress → agent reports {reported} remaining");
            let msg = format!(
                "Progress recorded ({reported} remaining for this step). Keep working on this \
                 step; when it is fully complete, call `done` to finish the step and move on."
            );
            (msg, vec![], vec![])
        }
    };

    let eid = kb_eval::tz::backdated_episode_id(30);
    let eid_ret = eid.clone();
    let run_log = std::sync::Arc::clone(&run);
    let llm_rounds_step_chat = std::sync::Arc::clone(&llm_rounds_step);
    let llm_rounds_total_chat = std::sync::Arc::clone(&llm_rounds_total);
    let mut chat = move |messages: &[Value], ep: Option<&str>| -> Result<TzTurn> {
        let since_step =
            llm_rounds_step_chat.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let total = llm_rounds_total_chat.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if total.is_multiple_of(10) {
            let step = run_log.lock().unwrap().current_step;
            eprintln!("  [sop] LLM round {total} step {step} ({since_step} in step)");
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

    // Each SOP step runs as its OWN subagent (a fresh conversation): the step does its work
    // and calls `done`, which finishes that subagent and advances the driver to the next step —
    // a brand-new conversation seeded only with that step's context (its SOP body). This mirrors
    // how the SOP deploys in prod (one agent per step) and keeps a step from being polluted by
    // the whole run's accumulated history. `sop_advance` is an intra-step progress signal
    // (Materialize's loop), NOT a transition. One episode_id throughout so feedback/scoring stay
    // unified. Dedup OFF: the current dedup key is not path-aware across the unified
    // read/grep-over-notebook surface, so it would wrongly collapse distinct reads/greps
    // against different notebook paths. Re-enable once dedup is made path-aware.
    let policy = EpisodePolicy {
        stop_on_done: true,
        dedup_readonly: false,
    };
    let mut total_rounds = 0usize;
    let mut total_deduped = 0usize;
    let mut episode_id = Some(eid_ret);
    let mut sop_done = false;
    // Linear advance increments the step each iteration; the bound is a backstop only.
    for _ in 0..(sop_def.steps.len() + 2) {
        let cur = run.lock().unwrap().current_step;
        let step = match sop_def.steps.iter().find(|s| s.number == cur) {
            Some(s) => s.clone(),
            None => {
                sop_done = true;
                break;
            }
        };
        llm_rounds_step.store(0, std::sync::atomic::Ordering::Relaxed);
        // Fan-out step: an orchestrator agent decomposes the step and `spawn`s worker
        // sub-episodes (one focused table each). Detected by the ORCHESTRATOR role marker.
        let outcome = if step.body.contains("{# ORCHESTRATOR #}") {
            use std::sync::atomic::Ordering::Relaxed;
            let orch_prompt = sop::fanout::format_orchestrator_prompt(&step.body);
            // Orchestrator tool set: get_task (read the source) + delegate (run subagents).
            let orch_tools = [get_task_tool.clone(), delegate_tool.clone()];
            let orch_eid = kb_eval::tz::backdated_episode_id(30);
            let orch_chat = {
                let gw = gateway.to_string();
                let tags = tags.clone();
                let variant_s = variant.map(|s| s.to_string());
                let rounds_step = std::sync::Arc::clone(&llm_rounds_step);
                let rounds_total = std::sync::Arc::clone(&llm_rounds_total);
                let run_log = std::sync::Arc::clone(&run);
                move |messages: &[Value], ep: Option<&str>| -> Result<TzTurn> {
                    let since_step = rounds_step.fetch_add(1, Relaxed) + 1;
                    let total = rounds_total.fetch_add(1, Relaxed) + 1;
                    if total.is_multiple_of(10) {
                        let s = run_log.lock().unwrap().current_step;
                        eprintln!("  [sop] LLM round {total} step {s} ({since_step} in step, orch)");
                    }
                    let e = ep.unwrap_or(&orch_eid);
                    let turn = kb_eval::tz::infer(
                        &gw,
                        "constraint_validate",
                        e,
                        messages,
                        &tags,
                        timeout,
                        variant_s.as_deref(),
                        None,
                        Some(&orch_tools),
                    )?;
                    Ok(TzTurn {
                        content: turn.content,
                        episode_id: turn.episode_id,
                    })
                }
            };
            // Delegate-intercepting exec: `delegate` runs a nested subagent episode (via
            // run_subagent) and returns ITS OWN answer; every other tool falls through to the
            // normal dispatch. A per-step counter (Arc<AtomicUsize>, since exec is Fn + Sync)
            // backstops runaway fan-out at WORKER_CAP_PER_STEP.
            let spawn_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let orch_exec = {
                let sopd = sop_dir.to_path_buf();
                let agent_dir = agent_g_dir.to_path_buf();
                let doc = src_doc.to_string();
                let gw = gateway.to_string();
                let tags = tags.clone();
                let variant_s = variant.map(|s| s.to_string());
                let step_body = step.body.clone();
                let normal = make_exec(agent_dir.clone(), doc.clone());
                let count = std::sync::Arc::clone(&spawn_count);
                move |name: &str, args: &Value| -> (String, Vec<String>, Vec<glossa::read::DocImage>) {
                    if name == "delegate" {
                        let n = count.fetch_add(1, Relaxed) + 1;
                        if let Some(msg) = spawn_capped(n, WORKER_CAP_PER_STEP) {
                            return (msg.to_string(), vec![], vec![]);
                        }
                        // The orchestrator names the subagent (`researcher`/`worker`); we
                        // run that role's prompt and return ITS OWN final answer — no
                        // harness receipt (mirrors prod delegation).
                        let agent = args.get("agent").and_then(|v| v.as_str()).unwrap_or("worker");
                        let task = args.get("task").and_then(|v| v.as_str()).unwrap_or("");
                        let prompt = sop::fanout::format_agent_prompt(&step_body, agent, task);
                        let answer = run_subagent(
                            &sopd,
                            &agent_dir,
                            &doc,
                            &gw,
                            &tags,
                            variant_s.as_deref(),
                            timeout,
                            &prompt,
                        );
                        return (answer, vec![], vec![]);
                    }
                    normal(name, args)
                }
            };
            run_episode(
                orch_chat,
                &orch_prompt,
                &orch_exec,
                SOP_MAX_LLM_ROUNDS_PER_STEP,
                policy,
            )?
        } else {
            let prompt = {
                let run = run.lock().unwrap();
                sop::prompt::format_step_context(&sop_def, &run, &step)
            };
            run_episode(&mut chat, &prompt, &exec, SOP_MAX_LLM_ROUNDS_PER_STEP, policy)?
        };
        total_rounds += outcome.rounds;
        total_deduped += outcome.deduped;
        if outcome.episode_id.is_some() {
            episode_id = outcome.episode_id.clone();
        }
        {
            let mut run = run.lock().unwrap();
            run.step_results.push(SopStepResult {
                step_number: cur,
                status: SopStepStatus::Completed,
                output: outcome.answer.clone(),
                started_at: String::new(),
                completed_at: None,
            });
        }
        eprintln!(
            "  [sop] step {cur} finished (subagent done={} rounds={} deduped={}) → next step",
            outcome.done, outcome.rounds, outcome.deduped
        );
        let next = cur + 1;
        if sop_def.steps.iter().any(|s| s.number == next) {
            run.lock().unwrap().current_step = next;
        } else {
            run.lock().unwrap().status = SopRunStatus::Completed;
            sop_done = true;
            break;
        }
    }
    let run = run.lock().unwrap();
    eprintln!(
        "[sop] run {:?} — {} step-subagents executed (llm_rounds={}, deduped={} tool calls skipped)",
        run.status,
        run.step_results.len(),
        total_rounds,
        total_deduped
    );
    Ok(EpisodeOutcome {
        answer: String::new(),
        episode_id,
        surfaced_titles: vec![],
        done: sop_done,
        rounds: total_rounds,
        deduped: total_deduped,
    })
}

const WORKER_MAX_ROUNDS: usize = 80;

/// Per-step worker backstop: beyond this many `spawn`s in a single fan-out step,
/// `spawn` refuses and tells the orchestrator to stop. Not a target — a runaway guard.
const WORKER_CAP_PER_STEP: usize = 30;

/// Worker-cap decision for the fan-out `spawn` arm. `count` is the post-increment
/// spawn count for this step. Returns `None` to allow the spawn, or `Some(message)`
/// once the count exceeds `cap` — the message tells the orchestrator to stop and `done`.
fn spawn_capped(count: usize, cap: usize) -> Option<&'static str> {
    if count > cap {
        Some(
            "Worker limit reached for this step. Do not spawn more workers; \
             call `done` to finish the step.",
        )
    } else {
        None
    }
}

/// One-line receipt: files whose row count grew during the worker episode.

/// Run ONE delegated subagent episode and return its OWN final answer.
///
/// A subagent neither delegates nor advances SOP steps: its tool set is the plain
/// dispatch (`make_exec`) plus `get_task` ONLY (no `delegate`, no `sop_advance`).
/// It does its work on disk and calls `done`; we return its final message (the
/// `done` note / last content) — mirroring prod delegation, where the harness
/// computes no receipt. Infer errors are non-fatal — a dead episode yields a
/// short placeholder answer so the orchestrator always gets a reply.
#[allow(clippy::too_many_arguments)]
fn run_subagent(
    _sop_dir: &std::path::Path,
    agent_g_dir: &std::path::Path,
    src_doc: &str,
    gateway: &str,
    tags: &Value,
    variant: Option<&str>,
    timeout: Duration,
    worker_prompt: &str,
) -> String {
    // Subagents get NO `get_task` — it's an orchestrator-only stand-in for the
    // user-provided task; the orchestrator names the document in the `delegate`
    // task prose (as in prod). A subagent's advertised tools are the
    // constraint_validate function set (search/read/grep/note/graph_stats/…).

    // Plain tool dispatch — already handles search/read/grep/note/ls/del/done.
    let worker_exec = make_exec(agent_g_dir.to_path_buf(), src_doc.to_string());

    let eid = kb_eval::tz::backdated_episode_id(30);
    let worker_chat = move |messages: &[Value], ep: Option<&str>| -> Result<TzTurn> {
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
            None,
        )?;
        Ok(TzTurn {
            content: turn.content,
            episode_id: turn.episode_id,
        })
    };

    let outcome = run_episode(
        worker_chat,
        worker_prompt,
        &worker_exec,
        WORKER_MAX_ROUNDS,
        EpisodePolicy {
            stop_on_done: true,
            dedup_readonly: false,
        },
    );
    // The subagent returns its OWN answer (its `done` note / last message) — as in
    // prod delegation. No harness-computed receipt.
    match outcome {
        Ok(o) if !o.answer.trim().is_empty() => o.answer,
        Ok(o) => format!("(субагент завершил без текста, done={})", o.done),
        Err(e) => format!("(субагент прервался: {e})"),
    }
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
    // Defensive: drop trailing list-separator punctuation an agent may copy from a
    // source that lists values inline ("50; 63; 80"). A value never legitimately
    // ends in ';' or ','; the decimal comma is always internal ("0,6"). Gold values
    // carry no trailing separators, so this only pulls a dirty agent value toward
    // its clean match — recall is monotonic (never reduced), and it cannot merge two
    // genuinely distinct values (those differ before the trailing separator).
    let stripped = stripped.trim_end_matches([';', ',', ' ']).to_string();
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

const PARAM_COVERED_THRESHOLD: f64 = 0.8;

#[derive(Debug)]
struct ParamCoverage {
    ref_name: String,
    agent_col: Option<String>,
    recall_hit: usize,
    recall_total: usize,
    missing: Vec<String>,
}

#[derive(Debug)]
struct TablesReport {
    params: Vec<ParamCoverage>,
    values_covered: usize,
    values_total: usize,
}

impl TablesReport {
    fn params_covered(&self, threshold: f64) -> usize {
        self.params
            .iter()
            .filter(|p| {
                p.recall_total > 0
                    && p.recall_hit as f64 / p.recall_total as f64 >= threshold
            })
            .count()
    }
    fn params_total(&self) -> usize {
        self.params.len()
    }
}

fn format_param_table(report: &TablesReport, threshold: f64) -> String {
    let mut out = String::new();
    for p in &report.params {
        let recall = if p.recall_total == 0 {
            1.0
        } else {
            p.recall_hit as f64 / p.recall_total as f64
        };
        let mark = if recall >= threshold { "  " } else { "XX" };
        let col = p.agent_col.as_deref().unwrap_or("—");
        let missing = if p.missing.is_empty() {
            "—".to_string()
        } else {
            let shown: Vec<&str> = p.missing.iter().take(8).map(String::as_str).collect();
            let extra = p.missing.len().saturating_sub(8);
            if extra > 0 {
                format!("{}…+{}", shown.join(","), extra)
            } else {
                shown.join(",")
            }
        };
        out.push_str(&format!(
            "  {mark} {:<34} {:<14} {}/{}  missing: {}\n",
            p.ref_name, col, p.recall_hit, p.recall_total, missing
        ));
    }
    out
}

fn format_relation_report(report: &RelationReport, threshold: f64) -> String {
    let f1 = |p: f64, r: f64| if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) };
    let mut out = String::new();
    let (rh, rt_, ah, at_) = (
        report.rows_hit(),
        report.rows_total(),
        report.agent_rows_hit(),
        report.agent_rows_total(),
    );
    let recall = if rt_ == 0 { 1.0 } else { rh as f64 / rt_ as f64 };
    let precision = if at_ == 0 { 0.0 } else { ah as f64 / at_ as f64 };
    out.push_str(&format!(
        "  relations: {}/{} tables | recall {rh}/{rt_} ({recall:.2}) | precision {ah}/{at_} ({precision:.2}) | F1 {:.2}\n",
        report.tables_covered(threshold),
        report.tables.len(),
        f1(precision, recall),
    ));
    for t in &report.tables {
        if t.agent_file.is_none() {
            out.push_str(&format!("     {:<28} [{}] —\n", t.ref_name, t.params.join("×")));
            continue;
        }
        let file = t.agent_file.as_deref().unwrap_or("—");
        let recall = if t.rows_total == 0 { 1.0 } else { t.rows_hit as f64 / t.rows_total as f64 };
        let precision = if t.agent_rows_total == 0 {
            0.0
        } else {
            t.agent_rows_hit as f64 / t.agent_rows_total as f64
        };
        let mark = if recall >= threshold { "  " } else { "XX" };
        out.push_str(&format!(
            "  {mark} {:<28} [{}] recall {}/{} ({:.2})  precision {}/{} ({:.2})  F1 {:.2}\n",
            t.ref_name,
            file,
            t.rows_hit,
            t.rows_total,
            recall,
            t.agent_rows_hit,
            t.agent_rows_total,
            precision,
            f1(precision, recall),
        ));
    }
    out
}

#[derive(Debug)]
struct RelationCoverage {
    ref_name: String,
    params: Vec<String>,
    agent_file: Option<String>,
    rows_hit: usize,
    rows_total: usize,
    agent_rows_hit: usize,
    agent_rows_total: usize,
}

#[derive(Debug)]
struct RelationReport {
    tables: Vec<RelationCoverage>,
}

impl RelationReport {
    fn rows_hit(&self) -> usize {
        self.tables.iter().map(|t| t.rows_hit).sum()
    }
    fn rows_total(&self) -> usize {
        self.tables.iter().map(|t| t.rows_total).sum()
    }
    fn agent_rows_hit(&self) -> usize {
        self.tables.iter().map(|t| t.agent_rows_hit).sum()
    }
    fn agent_rows_total(&self) -> usize {
        self.tables.iter().map(|t| t.agent_rows_total).sum()
    }
    fn tables_covered(&self, threshold: f64) -> usize {
        self.tables
            .iter()
            .filter(|t| t.rows_total > 0 && t.rows_hit as f64 / t.rows_total as f64 >= threshold)
            .count()
    }
}

/// Kuhn's augmenting-path step: try to find an augmenting path from left-node
/// `u`, reassigning existing matches along the way. `match_right[v]` holds the
/// left-node currently matched to right-node `v` (`usize::MAX` if free).
fn try_kuhn(u: usize, adj: &[Vec<usize>], visited: &mut [bool], match_right: &mut [usize]) -> bool {
    for &v in &adj[u] {
        if visited[v] {
            continue;
        }
        visited[v] = true;
        if match_right[v] == usize::MAX || try_kuhn(match_right[v], adj, visited, match_right) {
            match_right[v] = u;
            return true;
        }
    }
    false
}

/// Maximum bipartite matching (Kuhn's algorithm) between `n_left` left nodes
/// and `n_right` right nodes. `adj[u]` lists the right-node indices `u` may
/// match to. Returns `mapping[u]` = its matched right-node index, or
/// `usize::MAX` if `u` has no match in the maximum matching found.
fn max_bipartite_matching(n_left: usize, n_right: usize, adj: &[Vec<usize>]) -> Vec<usize> {
    let mut match_right: Vec<usize> = vec![usize::MAX; n_right];
    for u in 0..n_left {
        let mut visited = vec![false; n_right];
        try_kuhn(u, adj, &mut visited, &mut match_right);
    }
    let mut mapping = vec![usize::MAX; n_left];
    for (v, &u) in match_right.iter().enumerate() {
        if u != usize::MAX {
            mapping[u] = v;
        }
    }
    mapping
}

struct AgentTbl {
    file: String,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    col_vals: Vec<BTreeSet<String>>,
}

/// Load each agent `.csp` for `src_doc` as an `AgentTbl`, with a normalized
/// value-set per column.
fn load_agent_tables(agent_g_dir: &std::path::Path, src_doc: &str) -> Vec<AgentTbl> {
    let agent_raw =
        glossa::tables::csp_tables_per_file(agent_g_dir, src_doc).unwrap_or_default();
    agent_raw
        .into_iter()
        .map(|(file, t)| {
            let headers = t.headers.clone();
            let col_vals: Vec<BTreeSet<String>> = (0..t.headers.len())
                .map(|i| {
                    t.rows
                        .iter()
                        .filter_map(|r| r.get(i))
                        .filter(|c| !c.is_empty())
                        .map(|c| norm_value(c))
                        .collect()
                })
                .collect();
            AgentTbl { file, headers, rows: t.rows, col_vals }
        })
        .collect()
}

/// Normalize a parameter/header name for matching: lowercase, trim, and drop a
/// trailing short code token (e.g. "высота T" -> "высота", "диаметр D" -> "диаметр").
fn norm_name(s: &str) -> String {
    let s = s.trim().to_lowercase();
    match s.rsplit_once(' ') {
        Some((head, tail)) if tail.chars().count() <= 2 && !head.is_empty() => head.trim().to_string(),
        _ => s,
    }
}

/// Pure scorer for relational coverage. For each reference table, match its
/// **discriminating** params (value-set ≥ 2) to distinct agent columns via
/// maximum bipartite matching — adjacency is bound by manifest POSITION
/// (`canon_order`/`agent_order`) with a value-overlap fallback — then measure:
///   - recall: reference tuples reproduced as agent rows;
///   - precision: agent tuples (deduped, projected onto the matched columns)
///     that are genuinely in the reference.
/// Constant params (a single value, e.g. a fixed marking context) are ignored —
/// they carry no combinatorial information and the compiler drops them too, so
/// a correct 2-column table matches a 3-param reference table. If a reference
/// table has no discriminating param, fall back to all params (so an
/// all-constant table cannot match trivially).
fn score_relations(
    ref_tables: &[RefTable],
    agent: &[AgentTbl],
    canon_order: &[String],
    agent_params: &[String],
    agent_files: &[String],
) -> RelationReport {
    let mut out = Vec::new();
    for rt in ref_tables {
        let ref_col_vals: Vec<BTreeSet<String>> = (0..rt.params.len())
            .map(|pi| rt.rows.iter().filter_map(|r| r.get(pi)).map(|c| norm_value(c)).collect())
            .collect();

        let discriminating: Vec<usize> =
            (0..rt.params.len()).filter(|&pi| ref_col_vals[pi].len() >= 2).collect();
        let required: Vec<usize> =
            if discriminating.is_empty() { (0..rt.params.len()).collect() } else { discriminating };

        // Positional binding via the two schema.toml manifests. A gold param maps
        // to its position in the reference order (`canon_order`); the agent field
        // at that SAME position (`agent_order`) names the agent column. Position
        // bridges reference↔agent — each side uses only its own naming, the agent
        // never has to know reference names. When no manifest position resolves a
        // param (missing manifest, out-of-range, or the agent's own header does
        // not match its own manifest field), fall back to value overlap. `score`
        // (match size) is the primary ranking key so recall/precision semantics
        // are unchanged; `pos_score` (params resolved positionally) breaks ties in
        // favor of the positionally-bound table.
        let mut best: Option<(usize, Vec<usize>, usize, usize)> = None;
        for (ai, at) in agent.iter().enumerate() {
            let mut pos_score = 0usize;
            let adj: Vec<Vec<usize>> = required
                .iter()
                .map(|&pi| {
                    let by_pos: Vec<usize> = canon_order
                        .iter()
                        .position(|n| n == &rt.params[pi])
                        .map(|cpos| {
                            // 1) File path (preferred): the agent field at this
                            // position names a `.csp` file; if it is THIS file,
                            // match the param to its value-overlapping columns
                            // within — no header name needed (fixes code≠header).
                            if agent_files.get(cpos).is_some_and(|f| f == &at.file) {
                                let cols: Vec<usize> = at
                                    .col_vals
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, cv)| ref_col_vals[pi].iter().any(|v| cv.contains(v)))
                                    .map(|(ci, _)| ci)
                                    .collect();
                                if !cols.is_empty() {
                                    return cols;
                                }
                            }
                            // 2) Field-name path: the agent field name at this
                            // position matches an agent header (agent-internal).
                            if let Some(afield) = agent_params.get(cpos) {
                                let an = norm_name(afield);
                                let cols: Vec<usize> = at
                                    .headers
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, h)| {
                                        let hn = norm_name(h);
                                        hn == an
                                            || (an.chars().count() > 1 && hn.contains(&an))
                                            || (hn.chars().count() > 1 && an.contains(&hn))
                                    })
                                    .map(|(ci, _)| ci)
                                    .collect();
                                if !cols.is_empty() {
                                    return cols;
                                }
                            }
                            Vec::new()
                        })
                        .unwrap_or_default();
                    if !by_pos.is_empty() {
                        pos_score += 1;
                        by_pos
                    } else {
                        at.col_vals
                            .iter()
                            .enumerate()
                            .filter(|(_, cv)| ref_col_vals[pi].iter().any(|v| cv.contains(v)))
                            .map(|(ci, _)| ci)
                            .collect()
                    }
                })
                .collect();
            let mapping = max_bipartite_matching(required.len(), at.col_vals.len(), &adj);
            let score = mapping.iter().filter(|&&ci| ci != usize::MAX).count();
            if best.as_ref().is_none_or(|(_, _, s, ps)| (score, pos_score) > (*s, *ps)) {
                best = Some((ai, mapping, score, pos_score));
            }
        }

        let cov = match best {
            Some((ai, mapping, score, _)) if score == required.len() => {
                let at = &agent[ai];
                let project_ref = |r: &[String]| -> Vec<String> {
                    required
                        .iter()
                        .map(|&pi| r.get(pi).map(|c| norm_value(c)).unwrap_or_default())
                        .collect()
                };
                let agent_tuples: BTreeSet<Vec<String>> = at
                    .rows
                    .iter()
                    .map(|r| {
                        mapping
                            .iter()
                            .map(|&ci| r.get(ci).map(|c| norm_value(c)).unwrap_or_default())
                            .collect()
                    })
                    .collect();
                let rows_hit =
                    rt.rows.iter().filter(|r| agent_tuples.contains(&project_ref(r))).count();
                let gold_proj: BTreeSet<Vec<String>> =
                    rt.rows.iter().map(|r| project_ref(r)).collect();
                let agent_rows_total = agent_tuples.len();
                let agent_rows_hit =
                    agent_tuples.iter().filter(|t| gold_proj.contains(*t)).count();
                RelationCoverage {
                    ref_name: rt.name.clone(),
                    params: rt.params.clone(),
                    agent_file: Some(at.file.clone()),
                    rows_hit,
                    rows_total: rt.rows.len(),
                    agent_rows_hit,
                    agent_rows_total,
                }
            }
            _ => RelationCoverage {
                ref_name: rt.name.clone(),
                params: rt.params.clone(),
                agent_file: None,
                rows_hit: 0,
                rows_total: rt.rows.len(),
                agent_rows_hit: 0,
                agent_rows_total: 0,
            },
        };
        out.push(cov);
    }
    RelationReport { tables: out }
}

/// Relational (combination-row) coverage: load the agent `.csp` tables and score
/// them against the reference tables (see `score_relations`). Complements
/// `compare_tables_by_domain`, which only scores per-parameter domains.
fn compare_relations(
    agent_g_dir: &std::path::Path,
    src_doc: &str,
    ref_tables: &[RefTable],
    canon_order: &[String],
) -> RelationReport {
    let agent = load_agent_tables(agent_g_dir, src_doc);
    let agent_order = glossa::tables::read_schema_order(
        &agent_g_dir.join(".glossa/notes").join(src_doc).join("schema.toml"),
    );
    score_relations(ref_tables, &agent, canon_order, &agent_order.params, &agent_order.files)
}

/// Tables-only comparison, by domain: a reference parameter is identified among
/// the agent's `.csp` columns by its VALUE SET, never by name (same semantics
/// as `compare_graphs`' field/literal coverage — synonyms don't matter).
/// Returns TablesReport: per-parameter assignments (greedy by recall) with recall
/// metrics, plus union value-coverage.
fn compare_tables_by_domain(
    agent_g_dir: &std::path::Path,
    src_doc: &str,
    cols: &[ColInfo],
) -> TablesReport {
    let col_values = glossa::tables::csp_column_values(agent_g_dir, src_doc).unwrap_or_else(|e| {
        eprintln!("[tables] csp scan error: {e:#}");
        Default::default()
    });
    // Agent columns as an ordered list (BTreeMap is name-sorted → stable indices).
    let agent_cols: Vec<(String, BTreeSet<String>)> = col_values
        .iter()
        .map(|(name, vals)| (name.clone(), vals.iter().map(|v| norm_value(v)).collect()))
        .collect();
    // Reference domains (canon), preserving `cols` order.
    let ref_doms: Vec<BTreeSet<String>> = cols
        .iter()
        .map(|c| c.valid.iter().map(|v| norm_value(v)).collect())
        .collect();

    // Candidate (ref, agent) pairs with a non-zero overlap.
    struct Cand {
        ri: usize,
        ai: usize,
        hit: usize,
        recall: f64,
    }
    let mut cands: Vec<Cand> = Vec::new();
    for (ri, dom) in ref_doms.iter().enumerate() {
        if dom.is_empty() {
            continue;
        }
        for (ai, (_, ad)) in agent_cols.iter().enumerate() {
            let hit = dom.iter().filter(|v| domain_covers(ad, v)).count();
            if hit > 0 {
                cands.push(Cand {
                    ri,
                    ai,
                    hit,
                    recall: hit as f64 / dom.len() as f64,
                });
            }
        }
    }
    // Greedy one-to-one: best recall first; deterministic tie-breaks.
    cands.sort_by(|a, b| {
        b.recall
            .partial_cmp(&a.recall)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.hit.cmp(&a.hit))
            .then(cols[a.ri].name.cmp(&cols[b.ri].name))
            .then(agent_cols[a.ai].0.cmp(&agent_cols[b.ai].0))
    });
    let mut assigned: Vec<Option<usize>> = vec![None; cols.len()];
    let mut agent_taken = vec![false; agent_cols.len()];
    for c in &cands {
        if assigned[c.ri].is_none() && !agent_taken[c.ai] {
            assigned[c.ri] = Some(c.ai);
            agent_taken[c.ai] = true;
        }
    }

    let params: Vec<ParamCoverage> = cols
        .iter()
        .enumerate()
        .map(|(ri, c)| {
            let dom = &ref_doms[ri];
            match assigned[ri] {
                Some(ai) => {
                    let ad = &agent_cols[ai].1;
                    let missing: Vec<String> =
                        dom.iter().filter(|v| !domain_covers(ad, v)).cloned().collect();
                    ParamCoverage {
                        ref_name: c.name.clone(),
                        agent_col: Some(agent_cols[ai].0.clone()),
                        recall_hit: dom.len() - missing.len(),
                        recall_total: dom.len(),
                        missing,
                    }
                }
                None => ParamCoverage {
                    ref_name: c.name.clone(),
                    agent_col: None,
                    recall_hit: 0,
                    recall_total: dom.len(),
                    missing: dom.iter().cloned().collect(),
                },
            }
        })
        .collect();

    // Value coverage (union) — unchanged definition.
    let agent_union: BTreeSet<String> =
        agent_cols.iter().flat_map(|(_, s)| s.iter().cloned()).collect();
    let ref_union: BTreeSet<String> = ref_doms.iter().flatten().cloned().collect();
    let values_covered = ref_union
        .iter()
        .filter(|v| domain_covers(&agent_union, v))
        .count();

    TablesReport {
        params,
        values_covered,
        values_total: ref_union.len(),
    }
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

    let (cols, rows, ref_tables, canon_order) =
        load_validation_data(&cli.val_dir).context("load validation tables")?;
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
        if cli.score_only {
            anyhow::ensure!(
                cli.keep_agent_dir.is_some() && agent_g_dir.join(".glossa").is_dir(),
                "--score-only requires an existing --keep-agent-dir with a .glossa store to score"
            );
        } else {
            // Seed a fresh per-run store from the KB. Skipped for --score-only so the
            // prior run's notes (.csp) survive to be re-scored.
            wipe_agent_glossa(&agent_g_dir).context("wipe prior agent store")?;
            copy_dir_all(&cli.kb.join(".glossa"), &agent_g_dir.join(".glossa"))
                .context("seed agent store from KB")?;
            std::fs::write(&ont_path, &ontology_toml).unwrap();
        }
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
                        &agent_g_dir_clone,
                        &idx_kb,
                        Some(&g),
                        &spec_kb,
                        &trace_kb,
                    )
                }
                "note" | "ls" | "del" => {
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
                            out.push_str(&glossa::tools::checklist_coverage_report(&g, doc));
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
        let outcome = if cli.score_only {
            // No run — score the existing notes. Zero outcome; downstream reads
            // done/rounds via unwrap_or so this flows straight into the report.
            Ok(EpisodeOutcome {
                answer: String::new(),
                episode_id: None,
                surfaced_titles: vec![],
                done: false,
                rounds: 0,
                deduped: 0,
            })
        } else if let Some(sop_dir) = cli.sop_dir.clone() {
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
            let report = compare_tables_by_domain(&agent_g_dir, &src_doc, &cols);
            let pc = report.params_covered(PARAM_COVERED_THRESHOLD);
            let pt = report.params_total();
            let vc = report.values_covered;
            let vt = report.values_total;
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
            let rel_report = compare_relations(&agent_g_dir, &src_doc, &ref_tables, &canon_order);
            print!("{}", format_relation_report(&rel_report, PARAM_COVERED_THRESHOLD));
            print!("{}", format_param_table(&report, PARAM_COVERED_THRESHOLD));
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
    fn spawn_capped_boundary() {
        // At/under the cap: allowed (None). The Nth allowed spawn has count == cap.
        assert!(spawn_capped(1, 30).is_none());
        assert!(spawn_capped(30, 30).is_none());
        // First spawn past the cap is refused with a stop-and-done message.
        let over = spawn_capped(31, 30).expect("31 > 30 must be capped");
        assert!(over.to_lowercase().contains("done"), "{over}");
        assert!(spawn_capped(1000, 30).is_some());
        // Degenerate cap of 0: even the first spawn is refused.
        assert!(spawn_capped(1, 0).is_some());
    }

    #[test]
    fn norm_value_strips_trailing_separators() {
        // A trailing list-separator the agent copied from an inline source list
        // must normalize to the same value as the clean form.
        assert_eq!(norm_value("50;"), norm_value("50"));
        assert_eq!(norm_value("0,6;"), norm_value("0,6"));
        assert_eq!(norm_value("F24,"), norm_value("F24"));
        assert_eq!(norm_value("80 "), norm_value("80"));
        // The internal decimal comma is preserved (not a trailing separator).
        assert_ne!(norm_value("0,6"), norm_value("06"));
        // Distinct values stay distinct — they differ before any trailing char.
        assert_ne!(norm_value("50;"), norm_value("60;"));
    }

    #[test]
    fn load_validation_extracts_relational_tables() {
        let dir = tempfile::tempdir().unwrap();
        // Dependent table: h varies with D (2 varying columns) + a constant metadata column.
        std::fs::write(
            dir.path().join("Height.json"),
            r#"{"tables":[{"rows":[
                {"h":"0,6","D":125,"Doc":"G"},
                {"h":"0,6","D":150,"Doc":"G"},
                {"h":"0,8","D":125,"Doc":"G"}
            ]}]}"#,
        ).unwrap();
        // Flat table: one varying column.
        std::fs::write(
            dir.path().join("Grit.json"),
            r#"{"tables":[{"rows":[{"Grit":"F16"},{"Grit":"F20"}]}]}"#,
        ).unwrap();
        // Metadata file (underscore) is ignored.
        std::fs::write(dir.path().join("_meta.json"), r#"{"tables":[{"rows":[{"x":"1"}]}]}"#).unwrap();

        let (_cols, _rows, refs, _canon) = load_validation_data(dir.path()).unwrap();
        assert_eq!(refs.len(), 1, "only the dependent table is relational");
        let h = &refs[0];
        assert_eq!(h.name, "Height");
        assert_eq!(h.params, vec!["D".to_string(), "h".to_string()]); // name-sorted
        // rows projected to (D, h), deduped: (125,0,6) (150,0,6) (125,0,8)
        assert_eq!(h.rows.len(), 3);
        assert!(h.rows.contains(&vec!["125".to_string(), "0,6".to_string()]));
        assert!(h.rows.contains(&vec!["150".to_string(), "0,6".to_string()]));
    }

    #[test]
    fn load_validation_reads_canon_order() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("Тип.json"),
            r#"{"tables":[{"rows":[{"D":"1","h":"2"}]}]}"#,
        )
        .unwrap();
        std::fs::write(dir.path().join("schema.toml"), "[order]\nparams = [\"Тип\", \"Высота\"]\n").unwrap();
        let (_cols, _rows, _refs, canon) = load_validation_data(dir.path()).unwrap();
        assert_eq!(canon, vec!["Тип".to_string(), "Высота".into()]);
    }

    #[test]
    fn relation_report_formats_summary() {
        let rep = RelationReport {
            tables: vec![
                RelationCoverage {
                    ref_name: "Height".into(),
                    params: vec!["D".into(), "h".into()],
                    agent_file: Some("height.csp".into()),
                    rows_hit: 140,
                    rows_total: 161,
                    agent_rows_hit: 140,
                    agent_rows_total: 150,
                },
                RelationCoverage {
                    ref_name: "SoundIndex".into(),
                    params: vec!["ЗИ".into(), "Связка".into()],
                    agent_file: None,
                    rows_hit: 0,
                    rows_total: 19,
                    agent_rows_hit: 0,
                    agent_rows_total: 0,
                },
            ],
        };
        let out = format_relation_report(&rep, PARAM_COVERED_THRESHOLD);
        assert!(out.contains("recall 140/180"), "{out}");
        assert!(out.contains("precision 140/150"), "{out}");
        assert!(out.contains("Height"), "{out}");
        assert!(out.contains("—"), "{out}"); // missing agent table shown as —
    }

    #[test]
    fn constant_gold_param_is_dropped_from_match() {
        // Gold table (D, h, тип) where тип is constant → agent (D, h) must still match.
        let rt = RefTable {
            name: "Height".into(),
            params: vec!["D".into(), "h".into(), "тип".into()],
            rows: vec![
                vec!["125".into(), "0,6".into(), "41".into()],
                vec!["150".into(), "0,6".into(), "41".into()],
                vec!["125".into(), "0,8".into(), "41".into()],
            ],
        };
        // Build col_vals via norm_value, exactly as load_agent_tables does, so the
        // agent side is normalized the same way as the reference side.
        let rows = vec![
            vec!["125".to_string(), "0,6".to_string()],
            vec!["150".to_string(), "0,6".to_string()],
            vec!["125".to_string(), "0,8".to_string()],
        ];
        let col_vals: Vec<BTreeSet<String>> = (0..2)
            .map(|i| rows.iter().filter_map(|r| r.get(i)).map(|c| norm_value(c)).collect())
            .collect();
        let agent = vec![AgentTbl {
            file: "D_h.csp".into(),
            headers: vec!["col1".into(), "col2".into()],
            rows,
            col_vals,
        }];
        // Empty manifests → value-overlap fallback (the constant-param path).
        let rep = score_relations(&[rt], &agent, &[], &[], &[]);
        let t = &rep.tables[0];
        assert_eq!(t.agent_file.as_deref(), Some("D_h.csp"), "matched despite constant тип");
        assert_eq!(t.rows_hit, 3, "all 3 gold rows covered on (D,h)");
        assert_eq!(t.rows_total, 3);
    }

    #[test]
    fn precision_counts_spurious_agent_rows() {
        // Agent reproduces both gold pairs plus 1 spurious pair → recall 1.0, precision 2/3.
        let rt = RefTable {
            name: "Grit".into(),
            params: vec!["F".into(), "K".into()],
            rows: vec![vec!["F30".into(), "1".into()], vec!["F30".into(), "2".into()]],
        };
        let agent = vec![AgentTbl {
            file: "F_K.csp".into(),
            headers: vec!["col1".into(), "col2".into()],
            rows: vec![
                vec!["F30".into(), "1".into()],
                vec!["F30".into(), "2".into()],
                vec!["F30".into(), "9".into()], // spurious
            ],
            col_vals: vec![
                ["F30"].iter().map(|s| s.to_string()).collect(),
                ["1", "2", "9"].iter().map(|s| s.to_string()).collect(),
            ],
        }];
        let t = &score_relations(&[rt], &agent, &[], &[], &[]).tables[0];
        assert_eq!(t.rows_hit, 2);
        assert_eq!(t.rows_total, 2);
        assert_eq!(t.agent_rows_hit, 2, "2 agent pairs are in gold");
        assert_eq!(t.agent_rows_total, 3, "3 distinct agent pairs");
    }

    // Height and bore share values (10, 13); binding must send Высота to the D_T
    // (height) table, not the value-overlapping D_H one. Two positional paths:
    // field name and file name.
    fn hbore_setup() -> (RefTable, Vec<AgentTbl>, Vec<String>) {
        let rt = RefTable {
            name: "Высота".into(),
            params: vec!["Наружный диаметр".into(), "Высота".into()],
            rows: vec![vec!["125".into(), "10".into()], vec!["150".into(), "13".into()]],
        };
        let mk = |file: &str, headers: [&str; 2]| {
            let rows: Vec<Vec<String>> =
                vec![vec!["125".into(), "10".into()], vec!["150".into(), "13".into()]];
            let col_vals: Vec<BTreeSet<String>> = (0..2)
                .map(|i| rows.iter().filter_map(|r| r.get(i)).map(|c| norm_value(c)).collect())
                .collect();
            AgentTbl {
                file: file.into(),
                headers: vec![headers[0].into(), headers[1].into()],
                rows,
                col_vals,
            }
        };
        let canon = vec!["Наружный диаметр".to_string(), "Высота".into()];
        (
            rt,
            vec![mk("D_H.csp", ["D", "H"]), mk("D_T.csp", ["D", "T"])],
            canon,
        )
    }

    #[test]
    fn binds_by_field_name_position() {
        // Agent codes T/H match its own headers; canon position → agent field name.
        let (rt, agent, canon) = hbore_setup();
        let agent_params = vec!["D".to_string(), "T".into()];
        let t = &score_relations(&[rt], &agent, &canon, &agent_params, &[]).tables[0];
        assert_eq!(t.agent_file.as_deref(), Some("D_T.csp"), "bound by field-name position");
    }

    #[test]
    fn binds_by_file_name_position() {
        // Headers are useless ("c1"/"c2"); the FILE at each canon position binds.
        let (rt, mut agent, canon) = hbore_setup();
        for at in &mut agent {
            at.headers = vec!["c1".into(), "c2".into()];
        }
        let agent_files = vec!["D_T.csp".to_string(), "D_T.csp".into()];
        let t = &score_relations(&[rt], &agent, &canon, &[], &agent_files).tables[0];
        assert_eq!(t.agent_file.as_deref(), Some("D_T.csp"), "bound by file-name position");
    }

    #[test]
    fn compile_step_number_is_three() {
        use kb_eval::sop::{load_sop, types::SopExecutionMode};
        let sop = load_sop(&default_eval_sop_dir(), SopExecutionMode::Auto).expect("load sop");
        let compile = sop
            .steps
            .iter()
            .find(|s| s.title.contains("Compile"))
            .expect("Compile step");
        // Fan-out Build is step 1, Schema step 2, so Compile is step 3.
        assert_eq!(compile.number, 3);
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
            deduped: 0,
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
    fn relations_score_row_recall() {
        use glossa::notebook::{mirror_dir_for_doc, notes_root};
        let dir = tempfile::tempdir().unwrap();
        let doc = "kb-gost/test.pdf";
        let mirror = notes_root(dir.path()).join(mirror_dir_for_doc(doc));
        std::fs::create_dir_all(&mirror).unwrap();
        // Agent multi-column table with 2 of the 3 reference combinations.
        std::fs::write(mirror.join("height.csp"), "h\tD\n0,6\t125\n0,6\t150\n").unwrap();

        let refs = vec![RefTable {
            name: "Height".into(),
            params: vec!["D".into(), "h".into()],
            rows: vec![
                vec!["125".into(), "0,6".into()],
                vec!["150".into(), "0,6".into()],
                vec!["125".into(), "0,8".into()],
            ],
        }];
        let rep = compare_relations(dir.path(), doc, &refs, &[]);
        assert_eq!(rep.tables.len(), 1);
        assert_eq!(rep.tables[0].agent_file.as_deref(), Some("height.csp"));
        assert_eq!((rep.tables[0].rows_hit, rep.tables[0].rows_total), (2, 3));
        assert_eq!(rep.tables_covered(0.8), 0); // 2/3 < 0.8
    }

    /// A perfect one-to-one assignment exists ("AllVals" -> B, "Narrow" -> A),
    /// but greedy per-param assignment (processing A before B, in name-sorted
    /// order) claims the value-superset column "AllVals" for A first — because
    /// it ties "Narrow" on overlap count and is encountered first — leaving B
    /// unmatched (only "AllVals" overlaps B, and it's already used). Maximum
    /// bipartite matching finds the reassignment that satisfies both params.
    #[test]
    fn relations_max_matching_beats_greedy() {
        use glossa::notebook::{mirror_dir_for_doc, notes_root};
        let dir = tempfile::tempdir().unwrap();
        let doc = "kb-gost/test.pdf";
        let mirror = notes_root(dir.path()).join(mirror_dir_for_doc(doc));
        std::fs::create_dir_all(&mirror).unwrap();
        // Column 0 ("AllVals") is a value superset {1,2,3,4}; column 1 ("Narrow")
        // is {1,2} only. Row 3 (AllVals=3, Narrow=1) reproduces the ref tuple
        // ("1","3") under the correct mapping A->Narrow, B->AllVals.
        std::fs::write(
            mirror.join("pair.csp"),
            "AllVals\tNarrow\n1\t1\n2\t2\n3\t1\n4\t2\n",
        )
        .unwrap();

        let refs = vec![RefTable {
            name: "AB".into(),
            params: vec!["A".into(), "B".into()],
            rows: vec![vec!["1".into(), "3".into()]],
        }];
        let rep = compare_relations(dir.path(), doc, &refs, &[]);
        assert_eq!(rep.tables.len(), 1);
        assert_eq!(
            rep.tables[0].agent_file.as_deref(),
            Some("pair.csp"),
            "a valid one-to-one column assignment exists and should be found"
        );
        assert_eq!(rep.tables[0].rows_hit, 1);
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
            score_only: false,
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
        let report = compare_tables_by_domain(dir.path(), doc, &cols);
        assert_eq!(
            (report.params_covered(PARAM_COVERED_THRESHOLD), report.params_total()),
            (1, 2)
        );
        assert_eq!((report.values_covered, report.values_total), (2, 4));
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
        let report = compare_tables_by_domain(dir.path(), doc, &cols);
        assert_eq!(
            (report.params_covered(PARAM_COVERED_THRESHOLD), report.params_total()),
            (1, 1)
        );
        assert_eq!((report.values_covered, report.values_total), (2, 2));
    }

    #[test]
    fn tables_domain_compare_empty_when_no_csp() {
        let dir = tempfile::tempdir().unwrap();
        let cols = vec![ColInfo {
            name: "X".into(),
            valid: vec!["1".into()],
        }];
        let report = compare_tables_by_domain(dir.path(), "kb-gost/none.pdf", &cols);
        assert_eq!(report.params_covered(PARAM_COVERED_THRESHOLD), 0);
        assert_eq!(report.values_covered, 0);
    }

    #[test]
    fn tables_domain_compare_exclusive_assignment_no_parasite() {
        let dir = tempfile::tempdir().unwrap();
        let doc = "kb-gost/test.pdf";
        let mirror = glossa::notebook::notes_root(dir.path())
            .join(glossa::notebook::mirror_dir_for_doc(doc));
        std::fs::create_dir_all(&mirror).unwrap();
        // Diameter column fully covers its own ref AND coincidentally most speeds.
        std::fs::write(mirror.join("diameter.csp"), "D\n50\n63\n80\n100\n125\n").unwrap();
        // The agent's own speed column is wrong: only 3 of 6 gold speeds.
        std::fs::write(mirror.join("speed.csp"), "speed\n63\n80\n100\n").unwrap();
        let cols = vec![
            ColInfo {
                name: "Наружный диаметр".into(),
                valid: vec!["50".into(), "63".into(), "80".into(), "100".into(), "125".into()],
            },
            ColInfo {
                name: "Скорость".into(),
                valid: vec![
                    "32".into(), "50".into(), "63".into(), "80".into(), "100".into(), "125".into(),
                ],
            },
        ];
        let report = compare_tables_by_domain(dir.path(), doc, &cols);
        // Diameter takes the diameter column; speed can only be served by speed.csp.
        let speed = report.params.iter().find(|p| p.ref_name == "Скорость").unwrap();
        assert_eq!(speed.agent_col.as_deref(), Some("speed"));
        assert_eq!((speed.recall_hit, speed.recall_total), (3, 6));
        // At the 0.8 bar only Diameter (1.0) counts; speed (0.5) does not.
        assert_eq!(report.params_covered(PARAM_COVERED_THRESHOLD), 1);
        assert_eq!(report.params_total(), 2);
    }

    #[test]
    fn tables_domain_compare_threshold_bite_at_half() {
        let dir = tempfile::tempdir().unwrap();
        let doc = "kb-gost/test.pdf";
        let mirror = glossa::notebook::notes_root(dir.path())
            .join(glossa::notebook::mirror_dir_for_doc(doc));
        std::fs::create_dir_all(&mirror).unwrap();
        // Exactly half the reference domain present.
        std::fs::write(mirror.join("t.csp"), "V\n1\n2\n").unwrap();
        let cols = vec![ColInfo {
            name: "P".into(),
            valid: vec!["1".into(), "2".into(), "3".into(), "4".into()],
        }];
        let report = compare_tables_by_domain(dir.path(), doc, &cols);
        assert_eq!(report.params_covered(0.5), 1); // old bar would pass
        assert_eq!(report.params_covered(PARAM_COVERED_THRESHOLD), 0); // 0.8 fails it
        let p = &report.params[0];
        assert_eq!((p.recall_hit, p.recall_total), (2, 4));
        assert_eq!(p.missing, vec!["3".to_string(), "4".to_string()]);
    }

    #[test]
    fn format_param_table_marks_and_missing() {
        let report = TablesReport {
            params: vec![
                ParamCoverage {
                    ref_name: "Наружный диаметр".into(),
                    agent_col: Some("diameter".into()),
                    recall_hit: 5,
                    recall_total: 5,
                    missing: vec![],
                },
                ParamCoverage {
                    ref_name: "Скорость".into(),
                    agent_col: Some("speed".into()),
                    recall_hit: 3,
                    recall_total: 6,
                    missing: vec!["100".into(), "32".into(), "50".into()],
                },
                ParamCoverage {
                    ref_name: "Звуковой индекс".into(),
                    agent_col: None,
                    recall_hit: 0,
                    recall_total: 8,
                    missing: vec!["23-25".into()],
                },
            ],
            values_covered: 97,
            values_total: 105,
        };
        let out = format_param_table(&report, PARAM_COVERED_THRESHOLD);
        // Covered param has no XX flag; its line names its column.
        assert!(out.contains("diameter"));
        // Under-covered param is flagged and shows its recall + missing values.
        assert!(out.contains("XX"));
        assert!(out.contains("speed"));
        assert!(out.contains("3/6"));
        assert!(out.contains("missing: 100,32,50"));
        // Unassigned param shows a dash for the column.
        assert!(out.contains("Звуковой индекс"));
        assert!(out.contains("—"));
    }
}
