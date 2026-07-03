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

/// Resolve the source document the constraints are extracted from: `--doc` when
/// given (canonicalized against the index), otherwise auto-detect — a KB with a
/// single indexed document needs no configuration.
/// This must be the CANONICAL indexed path: `load_problem` isolates fields by
/// exact `prov.source_path` match, and agent nodes get the canonical path from
/// `ops::graph_upsert`.
fn resolve_source_doc(idx: &DocIndex, requested: Option<&str>) -> Result<String> {
    if let Some(doc) = requested {
        return idx.canonical_document_path(doc)
            .ok_or_else(|| anyhow::anyhow!("--doc {doc:?} is not an indexed document in the knowledge base"));
    }
    let mut docs = BTreeSet::new();
    idx.iter_chunks(|path, _, _, _| { docs.insert(path.to_string()); })?;
    match docs.len() {
        0 => anyhow::bail!("knowledge base has no indexed documents"),
        1 => Ok(docs.into_iter().next().unwrap()),
        n => anyhow::bail!("knowledge base has {n} documents — pass --doc <path> to pick the source: {docs:?}"),
    }
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
fn sop_gate_check(dir: &std::path::Path, src: &str) -> Option<String> {
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
    if let Ok(problem) = glossa::constraint_adapter::load_problem(&g, &ont, src) {
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

fn solve_csp(dir: &std::path::Path, g: &GraphStore, mode: SolveMode, assignments: &[(String, Value)], src: &str) -> SolveResult {
    let ont = Ontology::load_or_default(dir);
    let problem = glossa::constraint_adapter::load_problem(g, &ont, src).unwrap();
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
    let nodes: Vec<glossa::graph::ops::UpsertNode> = parse_items(args, "nodes", &mut errs);
    let edges: Vec<glossa::graph::ops::UpsertEdge> = parse_items(args, "edges", &mut errs);
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

/// Field-name match: normalized equality, or containment for wording variants
/// ("h, высота" vs "высота"). Guarded by length so short names don't match everything.
fn field_labels_match(a: &str, b: &str) -> bool {
    a == b || (a.len() >= 4 && b.len() >= 4 && (a.contains(b) || b.contains(a)))
}

fn compare_graphs(agent_g: &GraphStore, ref_json: &Value) -> (f64, f64, f64) {
    let ref_nodes = ref_json["nodes"].as_array().map(|a| a.as_slice()).unwrap_or_default();
    let ref_edges = ref_json["edges"].as_array().map(|a| a.as_slice()).unwrap_or_default();

    let agent_nodes = agent_g.all_nodes().unwrap_or_default();
    let agent_edges = agent_g.all_edges().unwrap_or_default();

    // Field coverage: which reference fields are present in agent graph
    let ref_field_labels: BTreeSet<String> = ref_nodes.iter()
        .filter(|n| n.get("type").and_then(|t| t.as_str()) == Some("Field"))
        .filter_map(|n| n.get("label").and_then(|l| l.as_str()))
        .map(norm_metric)
        .collect();
    let agent_field_labels: BTreeSet<String> = agent_nodes.iter()
        .filter(|n| n.node_type == "Field")
        .flat_map(|n| std::iter::once(n.label.as_str()).chain(n.aliases.iter().map(|a| a.as_str())))
        .map(norm_metric)
        .collect();

    let field_cov = if ref_field_labels.is_empty() { 1.0 }
        else {
            ref_field_labels.iter()
                .filter(|r| agent_field_labels.iter().any(|a| field_labels_match(r, a)))
                .count() as f64 / ref_field_labels.len() as f64
        };

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
        let missing: Vec<&String> = ref_field_labels.iter()
            .filter(|r| !agent_field_labels.iter().any(|a| field_labels_match(r, a)))
            .collect();
        eprintln!("  [CMP] missing fields ({}/{}): {missing:?}", missing.len(), ref_field_labels.len());
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
    let src_doc = resolve_source_doc(&idx_kb, cli.doc.as_deref())?;
    drop(idx_kb);

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
    println!("  val_dir={}  cols={}  cases={}  csp_only={}", cli.val_dir.display(), cols.len(), cases.len(), cli.csp_only);
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
            let csp = solve_csp(dir.path(), &g, case.mode, &case.assignments, &src_doc);
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

        // ── Full agentic eval ──
        eprintln!("[{}/{}] {} {}  agentic...", i + 1, cases.len(), case.name, mode_label);
        let eid = kb_eval::tz::backdated_episode_id(30);
        let eid_chat = eid.clone();
        let url = cli.gateway.clone();
        let fn_name = "constraint_validate".to_string();
        let tz_tags = tags.clone();
        let ref_json_clone = ref_graph_json.clone();
        let tim = timeout;

        let chat = move |messages: &[Value], ep: Option<&str>| -> Result<TzTurn> {
            let eid = ep.unwrap_or(&eid_chat);
            let turn = kb_eval::tz::infer(&url, &fn_name, eid, messages, &tz_tags, tim, None, None)?;
            Ok(TzTurn { content: turn.content, episode_id: turn.episode_id })
        };

        let agent_dir = tempfile::tempdir().context("agent tempdir")?;
        let agent_g_dir = agent_dir.path().to_path_buf();
        let ont_path = agent_g_dir.join(".glossa").join("ontology.toml");
        std::fs::create_dir_all(agent_g_dir.join(".glossa")).unwrap();
        std::fs::write(&ont_path, &ontology_toml).unwrap();
        let _agent_init = GraphStore::open(&agent_g_dir).unwrap();

        // Units travel separately from values: "Наружный диаметр=125 (unit: мм)".
        let unit_by_field: BTreeMap<&str, &str> = cols.iter()
            .filter_map(|c| c.unit.as_deref().map(|u| (c.name.as_str(), u)))
            .collect();
        let assign_desc: String = case.assignments.iter()
            .map(|(k, v)| {
                let u = unit_by_field.get(k.as_str()).map(|u| format!(" (unit: {u})")).unwrap_or_default();
                format!("{k}={v}{u}")
            })
            .collect::<Vec<_>>().join(", ");
        let prompt = format!(
            "You are validating field assignments against the regulatory document '{src_doc}'.\n\n\
             === PHASE 1: Build constraint graph ===\n\
             The document is in the knowledge base. Use search/read to find its parameter tables.\n\
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
             === PHASE 3: Validate ===\n\
             Check these field assignments:\n\
             Mode: {mode_label}\n\
             {assign_desc}\n\n\
             Call constraint_solve(mode=\"{mode_label}\", field_assignments={{...}}) to check.\n\
             Compare solver result with your own analysis.\n\n\
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
                 Step 1: read the document's parameter table and create ONE Checklist node whose\n\
                 `aliases` list the name of EVERY parameter in it (this is your commitment to the\n\
                 full set).\n\
                 Step 2: for each parameter on the checklist create a Field and an Enum\n\
                 (Field --CONSTRAINED_BY--> Enum) with all allowed values in the Enum's aliases.\n\
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
                "graph_stats" => (glossa::tools::graph_stats(&g), vec![], vec![]),
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
                    let problem = glossa::constraint_adapter::load_problem(&g, &ont, &src_doc_exec).unwrap();
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
        let outcome = if cli.sop {
            let gate_dir = agent_g_dir.clone();
            let gate_src = src_doc.clone();
            run_episode_gated(chat, &prompt, exec, 50, EpisodePolicy::enrich(),
                move || sop_gate_check(&gate_dir, &gate_src))
        } else {
            run_episode(chat, &prompt, exec, 50, EpisodePolicy::enrich())
        };
        let tz_ms = t1.elapsed().as_millis();

        // ── Post-episode: compare agent graph with reference ──
        let agent_g = GraphStore::open(&agent_g_dir).unwrap();
        let (field_cov, constraint_cov, literal_cov) = compare_graphs(&agent_g, &ref_json_clone);
        let agent_nodes = agent_g.all_nodes().map(|v| v.len()).unwrap_or(0);
        let agent_edges = agent_g.all_edges().map(|v| v.len()).unwrap_or(0);
        let was_done = outcome.as_ref().map(|o| o.done).unwrap_or(false);
        let rounds = outcome.as_ref().map(|o| o.rounds).unwrap_or(0);

        // csp_agreement: does the agent's graph give the same verdict as the reference
        // graph for this case's assignments? This is the end-to-end usefulness metric.
        // Guard against a degenerate win: an empty/constraint-less graph calls every
        // valid row "valid" vacuously, so require the agent to actually constrain
        // something (≥1 CONSTRAINED_BY link) for the agreement to count.
        let csp_ref = solve_csp(ref_dir.path(), &ref_g, case.mode, &case.assignments, &src_doc);
        let csp_agent = solve_csp(&agent_g_dir, &agent_g, case.mode, &case.assignments, &src_doc);
        let agent_constrains = agent_g.all_edges().unwrap_or_default().iter()
            .any(|e| e.edge_type == "CONSTRAINED_BY");
        let csp_agreement = csp_agent.valid == csp_ref.valid && agent_constrains;

        // Post feedback
        let cli_gateway = &cli.gateway;
        kb_eval::tz::post_feedback(cli_gateway, &eid, "field_coverage", json!(field_cov), &tags);
        kb_eval::tz::post_feedback(cli_gateway, &eid, "constraint_coverage", json!(constraint_cov), &tags);
        kb_eval::tz::post_feedback(cli_gateway, &eid, "literal_coverage", json!(literal_cov), &tags);
        kb_eval::tz::post_feedback(cli_gateway, &eid, "agent_graph_node_count", json!(agent_nodes as f64), &tags);
        kb_eval::tz::post_feedback(cli_gateway, &eid, "agent_graph_edge_count", json!(agent_edges as f64), &tags);
        kb_eval::tz::post_feedback(cli_gateway, &eid, "csp_agreement", json!(csp_agreement), &tags);
        kb_eval::tz::post_feedback(cli_gateway, &eid, "tools_used", json!(rounds as f64), &tags);

        print!("[{}/{}] {} {}  ", i + 1, cases.len(), case.name, mode_label);
        match &outcome {
            Ok(out) => {
                print!("done={was_done} rounds={rounds} tz={tz_ms}ms  graph: {agent_nodes} nodes/{agent_edges} edges  cov: f={field_cov:.2} c={constraint_cov:.2} l={literal_cov:.2}  csp_agree={csp_agreement}");
                if !csp_agreement {
                    let vs: Vec<String> = csp_agent.violations.iter().take(3)
                        .map(|v| format!("{} [{}] expected {}, actual {}", v.field, v.constraint, v.expected, v.actual))
                        .collect();
                    print!("\n    agent_csp: valid={} (ref valid={}) {}", csp_agent.valid, csp_ref.valid, vs.join("; "));
                }
                let llm_text = &out.answer;
                let csp_has_violations = !csp_ref.violations.is_empty();
                let llm_correct = if csp_has_violations {
                    llm_text.to_lowercase().contains("violation")
                } else {
                    llm_text.to_lowercase().contains("valid") || llm_text.to_lowercase().contains("no violation")
                };
                kb_eval::tz::post_feedback(cli_gateway, &eid, "llm_correct", json!(llm_correct), &tags);
                print!("  llm_correct={llm_correct}");
                if !llm_correct {
                    print!("  answer: {llm_text:.100}");
                }
                println!();
            }
            Err(e) => println!("EPISODE ERROR: {e}"),
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(sop_gate_check(dir.path(), "gost.docx").unwrap().contains("Checklist"));

        // Checklist commits to two parameters; none built yet.
        put(&g, "chk:1", "Checklist", "parameters", &["высота", "диаметр"]);
        assert!(sop_gate_check(dir.path(), "gost.docx").unwrap().contains("диаметр"));

        // Build "высота" fully; "диаметр" still uncovered.
        put(&g, "fld:h", "Field", "высота", &[]);
        put(&g, "enum:h", "Enum", "высота enum", &["1", "2"]);
        edge(&g, "fld:h", "enum:h");
        let r = sop_gate_check(dir.path(), "gost.docx").unwrap();
        assert!(r.contains("диаметр") && !r.contains("высота"), "{r}");

        // Build "диаметр" fully → gate clears.
        put(&g, "fld:d", "Field", "диаметр", &[]);
        put(&g, "enum:d", "Enum", "диаметр enum", &["10", "20"]);
        edge(&g, "fld:d", "enum:d");
        assert!(sop_gate_check(dir.path(), "gost.docx").is_none(), "all covered → done allowed");
    }
}
