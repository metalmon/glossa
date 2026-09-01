//! `kbx distil --aliases-only`: CHAIN-driven alias enrichment.
//!
//! Enrich alias-POOR reasoning nodes so `glossary`/`resolve` match how users actually phrase
//! questions — including the ungrounded query-side entries (Symptom/Task). The unit of work is a
//! whole reasoning CHAIN (a grounded terminal plus the non-structural nodes reachable from it over
//! Chaining-relation edges), processed in ONE agent pass so the case's shared source is read once
//! rather than per node.
//!
//! The HARNESS — not the model — decides which nodes are poor (alias count `< min_aliases`,
//! restricted to `--seed-type`/`--doc`) AND pre-reads the chain's grounded terminal source once as
//! context; the model only chooses which aliases to add. Its ONLY tool is `graph_update`: with no
//! `graph_upsert` and no read/reach/search affordances it can neither create nodes/edges nor spin
//! on exploration, so a chain is one bounded pass (see `ALIAS_MAX_ROUNDS`). The strong model comes
//! from the `[distil]` endpoint, like the other `kbx distil` modes. Mirrors
//! `reason::run::run_reason_at` (worker pool + shared `GraphWriter`/`DocIndex` + progress bar) and
//! `reason::seed::chain_one_seed` (agent loop + `exec` closure) in shape.

use crate::backend::glossa_tools;
use crate::backend::openai::{
    cache_is_estimated, reset_resamples, reset_tokens, run_agent_loop, token_summary, StatusTicker,
};
use crate::backend::transport::openai::agent_chat_full;
use crate::distil::run::DistilArgs;
use crate::distil::seed_pool;
use crate::lab::{Endpoint, LabConfig};
use crate::parallel::{run_units_parallel, GraphWriter};
use crate::reason::schema_graph_block;
use crate::workspace::KbxPaths;
use anyhow::{Context, Result};
use glossa::graph::lock::with_graph_write_lock;
use glossa::graph::ontology::{Ontology, RelationRole};
use glossa::graph::store::{GraphStore, Node};
use glossa::index::store::DocIndex;
use glossa::read::DocImage;
use glossa::tools::ChainSpec;
use glossa::trace::TraceLog;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::IsTerminal;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// Fallback worker-pool size when neither `--jobs` nor `lab.toml`'s `[tuning] jobs_distil`
/// overrides it — same source-of-truth rationale as densify's own `DEFAULT_JOBS`.
const DEFAULT_JOBS: usize = 3;

/// Safety cap on how many nodes one chain's BFS may gather — a runaway densely-connected graph
/// must not put an unbounded node list into a single agent prompt.
const CHAIN_CAP: usize = 32;

/// Round cap for one chain's enrichment loop. The model has only `graph_update` (no read/reach), so
/// it adds aliases and stops in a round or two; this is just a ceiling against a spinning model.
const ALIAS_MAX_ROUNDS: usize = 4;

/// The enricher's system prompt — FILE-FIRST: the editable `aliases.md` in the workspace overrides
/// it; this embedded copy (the shipped template) is only the fallback when that file is absent.
/// Loaded once in `enrich_aliases_at` and threaded into every chain, so the behaviour is edited in
/// the `.md`, never here. Per-chain data (context + poor-node list) lives in the user message.
const DEFAULT_ALIASES_MD: &str = include_str!("../../templates/aliases.md");

/// One chain's worth of enrichment work: the poor nodes to enrich plus a grounded `anchor`
/// terminal the model can `reach(...)` to read the case's source once.
#[derive(Debug, Clone)]
struct AliasChain {
    /// Deterministic sort key AND the label used in the progress line — the terminal id for a
    /// terminal-seeded chain, or the node's own id for a singleton-sweep chain.
    key: String,
    /// A grounded terminal id the model can `reach(...)` to land on the case's source section.
    anchor: String,
    /// The harness-selected alias-poor nodes (the model does NOT choose these).
    poor: Vec<Node>,
}

/// indicatif progress bar over `len` units — hidden when `no_progress` or not a TTY (mirrors
/// `reason::progress_bar`/`distil::run::progress_bar`).
fn progress_bar(len: usize, no_progress: bool) -> ProgressBar {
    let show = !no_progress && std::io::stdout().is_terminal() && std::io::stderr().is_terminal();
    if !show {
        return ProgressBar::hidden();
    }
    let pb = ProgressBar::new(len as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.white} {prefix} [{pos}/{len}] {wide_bar:.white} {elapsed_precise}{msg}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    pb.enable_steady_tick(Duration::from_millis(90));
    pb
}

/// A `Chaining`-role edge per the ontology (both `LEADS_TO`-style declared relations and the
/// default for undeclared non-core relations); NOT grounding (`MENTIONS`) or attribute edges.
/// Exactly the test reason uses to decide "chaining" (see `reason::run::run_reason_at`).
fn is_chaining(ont: &Ontology, edge_type: &str) -> bool {
    ont.relation_role(edge_type) == RelationRole::Chaining
}

/// Pure: which nodes in a chain are "alias-poor" work — alias count strictly below `min_aliases`,
/// further restricted to `seed_type` when set. Returns their ids in input order. Model-free, so it
/// is unit-tested against a tiny in-memory node list (like `reason::chainless_seeds`'s tests).
pub(crate) fn poor_node_ids(
    chain: &[Node],
    min_aliases: usize,
    seed_type: Option<&str>,
) -> Vec<String> {
    chain
        .iter()
        .filter(|n| seed_type.is_none_or(|t| n.node_type == t))
        .filter(|n| n.aliases.len() < min_aliases)
        .map(|n| n.id.clone())
        .collect()
}

/// True if `id` carries an outgoing `MENTIONS` grounding into document `doc` (target `doc` itself
/// or a `doc#<n>` section). Used only for the `--doc` restriction; an ungrounded node has no such
/// edge and is therefore skipped under `--doc`.
fn grounded_to_doc(g: &GraphStore, id: &str, doc: &str) -> bool {
    let prefix = format!("{doc}#");
    g.outgoing(id).unwrap_or_default().into_iter().any(|e| {
        e.edge_type == glossa::graph::MENTIONS && (e.to == doc || e.to.starts_with(&prefix))
    })
}

/// BFS the connected set of NON-structural reasoning nodes reachable from `start` over
/// Chaining-relation edges (both directions), bounded at [`CHAIN_CAP`]. Nodes already claimed by
/// an earlier chain (`seen`) are not crossed into, so chains stay disjoint. Grounding (`MENTIONS`)
/// targets are `path#n` section refs, not nodes — `is_chaining` filters them out, so they never
/// enter the set.
fn gather_chain(
    g: &GraphStore,
    ont: &Ontology,
    structural: &HashSet<String>,
    by_id: &HashMap<String, Node>,
    start: &str,
    seen: &HashSet<String>,
) -> Result<Vec<String>> {
    let mut chain: Vec<String> = Vec::new();
    let mut local: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<String> = VecDeque::new();
    queue.push_back(start.to_string());
    local.insert(start.to_string());

    while let Some(id) = queue.pop_front() {
        match by_id.get(&id) {
            Some(node) if !structural.contains(&node.node_type) => {}
            _ => continue, // unknown id or a structural node: never part of a reasoning chain
        }
        chain.push(id.clone());
        if chain.len() >= CHAIN_CAP {
            break;
        }
        let mut neighbors: Vec<String> = Vec::new();
        for e in g.outgoing(&id)? {
            if is_chaining(ont, &e.edge_type) {
                neighbors.push(e.to);
            }
        }
        for e in g.incoming(&id)? {
            if is_chaining(ont, &e.edge_type) {
                neighbors.push(e.from);
            }
        }
        for nid in neighbors {
            if local.contains(&nid) || seen.contains(&nid) || !by_id.contains_key(&nid) {
                continue;
            }
            local.insert(nid.clone());
            queue.push_back(nid);
        }
    }
    Ok(chain)
}

/// The chunk a grounded node was harvested from: the `path#ord` target of its first `MENTIONS`
/// edge (the grounding), parsed into `(path, ord)`. The node's own `source_path` is document-level
/// provenance with NO chunk ordinal (`range` is unset on terminals), so the ordinal lives only on
/// the grounding edge's `to` handle — that is what we resolve here. `None` for an ungrounded node
/// (a singleton query-side entry with no `MENTIONS`).
fn grounding_chunk(g: &GraphStore, anchor: &str) -> Option<(String, u64)> {
    let edges = g.outgoing(anchor).ok()?;
    let to = &edges.iter().find(|e| e.edge_type == "MENTIONS")?.to;
    let (path, ord) = to.rsplit_once('#')?;
    Some((path.to_string(), ord.parse::<u64>().ok()?))
}

/// The grounded source text of a chain's `anchor` terminal, truncated — fed to the model as the
/// case's context so it needs no `read`/`reach` tools. Empty when the anchor is ungrounded (a
/// singleton query-side node) or its chunk can't be read; the poor-node labels still carry meaning.
fn alias_context(
    g: &GraphStore,
    root: &Path,
    idx: &DocIndex,
    anchor: &str,
    trace: &TraceLog,
) -> String {
    let Some((path, ord)) = grounding_chunk(g, anchor) else {
        return String::new();
    };
    let (text, _imgs) = glossa_tools::run_read(root, idx, Some(g), &path, ord, false, trace);
    text.chars().take(2000).collect()
}

fn build_alias_user_message(ont: &Ontology, context: &str, poor: &[Node]) -> String {
    // Legend of ONLY the node types present in this case, with their ontology descriptions, so the
    // model aliases each node in the right register (a Symptom wants generous plain-user phrasings;
    // a Resolution wants the procedure's short names). Deduped, first-seen order — not per node.
    let mut legend = String::new();
    let mut seen = HashSet::new();
    for n in poor {
        if seen.insert(n.node_type.clone()) {
            match ont.description(&n.node_type) {
                Some(d) => legend.push_str(&format!("- {} — {}\n", n.node_type, d)),
                None => legend.push_str(&format!("- {}\n", n.node_type)),
            }
        }
    }

    let mut list = String::new();
    for n in poor {
        let aliases = if n.aliases.is_empty() {
            "(none)".to_string()
        } else {
            n.aliases.join(", ")
        };
        list.push_str(&format!(
            "- {} [{}] «{}» — current aliases: {}\n",
            n.id, n.node_type, n.label, aliases
        ));
    }
    let ctx = if context.trim().is_empty() {
        "(no source text available)".to_string()
    } else {
        context.trim().to_string()
    };
    // Data only — the how (user-phrasings, graph_update, no new nodes) lives in `aliases.md`.
    format!(
        "Case source (context):\n---\n{ctx}\n---\n\nNode types in this case:\n{legend}\nEnrich the search aliases of these nodes:\n{list}"
    )
}

/// The alias-mode tool schema: `graph_update` is the ONLY tool (adding aliases). No `graph_upsert`
/// (can't create nodes/edges) and no read/reach/search (the harness pre-feeds the case source), so
/// a chain is one bounded write pass. That the tool set is graph_update-only IS the enforcement —
/// not a prompt rule.
fn alias_tools_schema() -> Value {
    // ONLY graph_update. The harness pre-feeds the case's grounded source text in the prompt, so
    // the model needs no read/reach/search exploration — it just adds aliases and calls
    // graph_update. Dropping the agentic round-trips is what turns a ~day-long full pass into hours.
    Value::Array(vec![graph_update_tool_value()])
}

/// The `graph_update` function-schema for alias mode — deliberately alias-only: each entry names an
/// existing node (by id or current label) and the aliases to ADD. No node/edge creation surface.
fn graph_update_tool_value() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "graph_update",
            "description": "Add search aliases to EXISTING reasoning nodes in place. Never creates nodes or edges. One entry per node.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": { "type": "string", "description": "the id OR current label of an existing node to enrich" },
                                "add_aliases": { "type": "array", "items": { "type": "string" }, "description": "extra search phrasings a user might type for this node" }
                            },
                            "required": ["label", "add_aliases"]
                        }
                    }
                },
                "required": ["nodes"]
            }
        }
    })
}

/// Parse the model's `graph_update` args into `(reference, add_aliases)` pairs — the canonical
/// `nodes: [...]` shape, or a single flat `{label|id, add_aliases}`. `new_label`/`new_type` are not
/// offered in alias mode, so only the alias fields are read. `id` is accepted as an alias for
/// `label` (either resolves the node in `apply_update`).
fn parse_alias_updates(args: &Value) -> Vec<glossa::graph::agent::NodeUpdate> {
    #[derive(serde::Deserialize)]
    struct Arg {
        #[serde(alias = "id")]
        label: String,
        #[serde(
            default,
            deserialize_with = "glossa::json_util::deserialize_opt_vec_string_loose"
        )]
        add_aliases: Option<Vec<String>>,
    }
    let to_update = |a: Arg| glossa::graph::agent::NodeUpdate {
        label: a.label,
        new_label: None,
        new_type: None,
        add_aliases: a.add_aliases.unwrap_or_default(),
    };
    if let Some(arr) = args.get("nodes").and_then(|v| v.as_array()) {
        arr.iter()
            .filter_map(|n| serde_json::from_value::<Arg>(n.clone()).ok())
            .map(to_update)
            .collect()
    } else if args.get("label").is_some() || args.get("id").is_some() {
        serde_json::from_value::<Arg>(args.clone())
            .ok()
            .map(to_update)
            .into_iter()
            .collect()
    } else {
        Vec::new()
    }
}

/// Resolve a node reference (id or label) to its current alias count, 0 if it resolves to nothing
/// — used to diff alias counts around one `graph_update` call for the progress stats.
fn alias_count(g: &GraphStore, reference: &str) -> usize {
    let id = if g.get_node(reference).ok().flatten().is_some() {
        Some(reference.to_string())
    } else {
        g.find_by_label(reference).ok().flatten()
    };
    id.and_then(|i| g.get_node(&i).ok().flatten())
        .map(|n| n.aliases.len())
        .unwrap_or(0)
}

/// Run one chain's agent pass: read the case once, then `graph_update` the listed poor nodes with
/// user-phrasing aliases. Returns `(aliases_added, nodes_changed)` for the progress line.
#[allow(clippy::too_many_arguments)]
fn enrich_one_chain(
    root: &Path,
    ont: &Ontology,
    ep: &Endpoint,
    tools: &Value,
    spec: &ChainSpec,
    alias_md: &str,
    unit: &AliasChain,
    writer: &GraphWriter,
    idx: &DocIndex,
    max_rounds: usize,
) -> Result<(usize, usize)> {
    let g = writer.store();
    let trace = TraceLog::disabled();

    let system = format!("{}\n\n{alias_md}", schema_graph_block(ont));
    // Harness reads the case's grounded source ONCE and feeds it as context — the model gets no
    // read/reach tools, so it can't spend rounds exploring; it just generates aliases.
    let context = alias_context(g, root, idx, &unit.anchor, &trace);
    let user = build_alias_user_message(ont, &context, &unit.poor);
    let messages = vec![
        json!({ "role": "system", "content": system }),
        json!({ "role": "user", "content": user }),
    ];

    let endpoint = ep.endpoint.clone();
    let model = ep.model.clone();
    let api_key = ep.resolve_key();
    let timeout = Duration::from_secs(ep.timeout_secs);

    let chat = |messages: &[Value]| {
        agent_chat_full(
            &endpoint,
            &model,
            api_key.as_deref(),
            tools,
            messages,
            timeout,
        )
    };

    let mut aliases_added = 0usize;
    let mut nodes_changed = 0usize;

    let exec = |name: &str, args: &Value| -> (String, Vec<String>, Vec<DocImage>) {
        if name == "graph_update" {
            let ups = parse_alias_updates(args);
            if ups.is_empty() {
                return (
                    "graph_update: no node to enrich — pass nodes:[{label|id, add_aliases:[…]}]"
                        .to_string(),
                    Vec::new(),
                    Vec::new(),
                );
            }
            let refs: Vec<String> = ups.iter().map(|u| u.label.clone()).collect();
            // The whole read-modify-write runs under the on-disk write lock (coordinating with a
            // concurrent MCP writer), mirroring how the MCP `graph_update` tool calls it. Chains
            // are node-disjoint, so two eval workers never touch the same node.
            let outcome = with_graph_write_lock(root, Duration::from_secs(120), || {
                let before: Vec<usize> = refs.iter().map(|r| alias_count(g, r)).collect();
                let msg = glossa::graph::ops::graph_update(g, ups);
                let after: Vec<usize> = refs.iter().map(|r| alias_count(g, r)).collect();
                let mut added = 0usize;
                let mut changed = 0usize;
                for (b, a) in before.iter().zip(&after) {
                    if a > b {
                        added += a - b;
                        changed += 1;
                    }
                }
                Ok((msg, added, changed))
            });
            match outcome {
                Ok((msg, added, changed)) => {
                    aliases_added += added;
                    nodes_changed += changed;
                    (msg, Vec::new(), Vec::new())
                }
                Err(e) => (format!("graph_update failed: {e}"), Vec::new(), Vec::new()),
            }
        } else {
            let (body, ids, _images) =
                glossa_tools::exec(name, args, root, idx, Some(g), spec, &trace);
            (body, ids, Vec::new())
        }
    };

    let on_repeat = |name: &str, _args: &Value| {
        format!(
            "(dup {name}) you already called this — try a different query, or call graph_update"
        )
    };

    run_agent_loop(chat, messages, exec, on_repeat, max_rounds, None)?;
    Ok((aliases_added, nodes_changed))
}

/// Build the deterministic list of chains to enrich: terminal-seeded chains first (BFS over
/// chaining edges), then a singleton sweep of any remaining alias-poor non-structural node so
/// global coverage holds. Returns the work units sorted by key. `--doc` (when set) keeps only poor
/// nodes grounded to that doc; ungrounded nodes are skipped under `--doc`.
fn build_alias_chains(
    g: &GraphStore,
    ont: &Ontology,
    args: &DistilArgs,
) -> Result<Vec<AliasChain>> {
    let seed_type = args.seed_type.as_deref();
    let doc = args.doc.as_deref();
    let all_nodes = g.all_nodes()?;
    let by_id: HashMap<String, Node> = all_nodes
        .iter()
        .map(|n| (n.id.clone(), n.clone()))
        .collect();
    let structural: HashSet<String> = ont.structural().into_iter().collect();

    // Keep only poor nodes that also satisfy the `--doc` grounding restriction, preserving order.
    let qualifying = |chain: &[Node]| -> Vec<Node> {
        let poor_ids: HashSet<String> = poor_node_ids(chain, args.min_aliases, seed_type)
            .into_iter()
            .collect();
        chain
            .iter()
            .filter(|n| poor_ids.contains(&n.id))
            .filter(|n| doc.is_none_or(|d| grounded_to_doc(g, &n.id, d)))
            .cloned()
            .collect()
    };

    let mut seen: HashSet<String> = HashSet::new();
    let mut units: Vec<AliasChain> = Vec::new();

    // 1) Terminal-seeded chains — same grounded seed pool `kbx reason` uses.
    let seeds = seed_pool(g, ont, seed_type)?;
    for seed in &seeds {
        if seen.contains(&seed.id) {
            continue;
        }
        let chain_ids = gather_chain(g, ont, &structural, &by_id, &seed.id, &seen)?;
        for id in &chain_ids {
            seen.insert(id.clone());
        }
        let chain_nodes: Vec<Node> = chain_ids
            .iter()
            .filter_map(|id| by_id.get(id).cloned())
            .collect();
        let poor = qualifying(&chain_nodes);
        if poor.is_empty() {
            continue;
        }
        units.push(AliasChain {
            key: seed.id.clone(),
            anchor: seed.id.clone(),
            poor,
        });
    }

    // 2) Singleton sweep — any remaining alias-poor non-structural node not yet claimed.
    for n in &all_nodes {
        if seen.contains(&n.id) || structural.contains(&n.node_type) {
            continue;
        }
        let one = std::slice::from_ref(n);
        let poor = qualifying(one);
        if poor.is_empty() {
            continue;
        }
        seen.insert(n.id.clone());
        units.push(AliasChain {
            key: n.id.clone(),
            anchor: n.id.clone(),
            poor,
        });
    }

    units.sort_by(|a, b| a.key.cmp(&b.key));
    Ok(units)
}

/// Orchestrate `kbx distil --aliases-only` over the corpus at already-resolved `paths`: build the
/// CHAIN work units (harness-selected poor nodes), then run one alias-enrichment agent pass per
/// chain through the shared worker pool.
pub fn enrich_aliases_at(paths: KbxPaths, args: &DistilArgs) -> Result<()> {
    let lab = LabConfig::load_at(&paths.lab)
        .with_context(|| format!("loading {}", paths.lab.display()))?;
    let ont = Ontology::load_or_default(&paths.root);

    // Ensure the corpus is indexed (chunks) so the model's read/reach resolve real sections.
    glossa::index::store::index_dir(&paths.root, false).context("indexing corpus")?;

    // Use the [distil] endpoint like the other `kbx distil` modes (densify, --emit-golds), so the
    // subcommand honours one configured role. Point [distil] at a strong model for good aliases.
    let ep = lab.distil.clone().ok_or_else(|| {
        anyhow::anyhow!("kbx distil --aliases-only needs a [distil] endpoint in lab.toml")
    })?;

    let g = Arc::new(GraphStore::open(&paths.root)?);
    let units = build_alias_chains(&g, &ont, args)?;
    if units.is_empty() {
        println!(
            "aliases: no alias-poor nodes to enrich (all nodes already have >= {} aliases, or none \
             matched --seed-type/--doc)",
            args.min_aliases
        );
        return Ok(());
    }

    let idx = DocIndex::open_or_create(&paths.root).context("opening doc index")?;
    let writer = GraphWriter::new(Arc::clone(&g), paths.root.clone());
    let spec = ChainSpec::from_ontology(&ont);
    let tools = alias_tools_schema();
    // File-first prompt: the editable `aliases.md` overrides the embedded default when present.
    let alias_md =
        std::fs::read_to_string(&paths.aliases).unwrap_or_else(|_| DEFAULT_ALIASES_MD.to_string());

    let jobs = crate::lab::resolve(args.jobs, lab.tuning.jobs_distil, DEFAULT_JOBS).max(1);
    // Small cap: with only graph_update (no exploration tools) the model adds aliases and stops in
    // a round or two — this is just a ceiling so a misbehaving model can't spin.
    let max_rounds = crate::lab::resolve(args.max_rounds, lab.tuning.max_rounds, ALIAS_MAX_ROUNDS);

    reset_tokens();
    reset_resamples();

    let pb = progress_bar(units.len(), args.no_progress);
    pb.set_prefix("aliasing");
    let ticker = StatusTicker::start(&pb);

    let root = paths.root.as_path();
    let results = run_units_parallel(
        units,
        jobs,
        &pb,
        |_unit| 1,
        |unit: &AliasChain| -> Result<(usize, usize)> {
            let (added, changed) = enrich_one_chain(
                root, &ont, &ep, &tools, &spec, &alias_md, unit, &writer, &idx, max_rounds,
            )
            .with_context(|| format!("enriching aliases for chain {}", unit.key))?;
            pb.println(format!(
                "aliases {}: +{added} on {changed} node(s)",
                unit.key
            ));
            Ok((added, changed))
        },
    )?;

    let chains = results.len();
    let (mut total_aliases, mut total_nodes) = (0usize, 0usize);
    for (added, changed) in &results {
        total_aliases += added;
        total_nodes += changed;
    }

    drop(ticker);
    pb.finish_and_clear();
    println!(
        "aliases: {chains} chain(s) processed, {total_nodes} node(s) enriched, {total_aliases} \
         alias(es) added"
    );
    let footnote = if cache_is_estimated() {
        " (cache estimated from prompt re-send)"
    } else {
        ""
    };
    println!("tokens: {}{footnote}", token_summary());

    // Refresh the node search index (glossary/resolve BM25) so the new aliases resolve — the same
    // finalize `kbx reason`/`kbx distil densify` run at the end of their passes.
    let summary = crate::build::finalize(&paths.root).context("finalizing aliases")?;
    println!("{summary}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glossa::graph::store::Provenance;

    fn prov() -> Provenance {
        Provenance {
            source_path: "d.md".into(),
            range: None,
            file_sig: None,
            origin: "test".into(),
            confidence: 0.9,
            created_at: 1,
        }
    }

    fn node(id: &str, ty: &str, aliases: &[&str]) -> Node {
        Node {
            id: id.into(),
            node_type: ty.into(),
            label: id.into(),
            aliases: aliases.iter().map(|s| s.to_string()).collect(),
            prov: prov(),
        }
    }

    /// The harness (not the model) selects poor nodes: alias count strictly below `min_aliases`,
    /// and — when `--seed-type` is set — only that type. Pure, tiny in-memory fixture (mirrors
    /// `reason::chainless_seeds`'s test style).
    #[test]
    fn poor_node_ids_filters_by_alias_count_and_seed_type() {
        let chain = vec![
            node("sym:a", "Symptom", &[]),              // 0 aliases -> poor
            node("sym:b", "Symptom", &["x", "y", "z"]), // 3 aliases -> rich (>= 3)
            node("task:c", "Task", &["one"]),           // 1 alias   -> poor
            node("res:d", "Resolution", &["a", "b"]),   // 2 aliases -> poor
        ];

        // No seed_type: every node with < 3 aliases is poor.
        let ids = poor_node_ids(&chain, 3, None);
        assert_eq!(ids, vec!["sym:a", "task:c", "res:d"]);

        // seed_type restricts to that type BEFORE the alias-count gate.
        let syms = poor_node_ids(&chain, 3, Some("Symptom"));
        assert_eq!(syms, vec!["sym:a"], "only the poor Symptom, not task/res");

        // A lower threshold makes fewer nodes poor.
        let strict = poor_node_ids(&chain, 1, None);
        assert_eq!(strict, vec!["sym:a"], "only 0-alias nodes are below 1");
    }

    /// The case-source chunk is resolved from the node's `MENTIONS` grounding edge (`to = path#ord`),
    /// NOT from its own `source_path`. Regression: the ordinal was parsed off `source_path`, but
    /// terminals store a bare path there (the ordinal lives only on the grounding edge), so every
    /// chain got an empty context and the model enriched blind.
    #[test]
    fn grounding_chunk_reads_ordinal_from_mentions_edge_not_source_path() {
        let dir = tempfile::tempdir().unwrap();
        let g = GraphStore::open(dir.path()).unwrap();

        // Grounded terminal: bare-path `source_path` (no `#ord`), grounded via a MENTIONS edge
        // whose target handle carries the chunk ordinal.
        g.put_node(&node("res:x", "Resolution", &[])).unwrap();
        g.put_edge(&glossa::graph::store::Edge {
            from: "res:x".into(),
            to: "docs/manual.pdf#7".into(),
            edge_type: "MENTIONS".into(),
            prov: prov(),
        })
        .unwrap();
        assert_eq!(
            grounding_chunk(&g, "res:x"),
            Some(("docs/manual.pdf".to_string(), 7))
        );

        // Ungrounded singleton (no MENTIONS): resolves to nothing — must NOT fall back to the
        // node's own `source_path`.
        g.put_node(&node("task:y", "Task", &[])).unwrap();
        assert_eq!(grounding_chunk(&g, "task:y"), None);
    }

    /// `graph_update` args parse from both the canonical `nodes:[...]` and a flat single entry,
    /// accepting `id` as an alias for `label` and a loose comma-string for `add_aliases`.
    #[test]
    fn parse_alias_updates_accepts_nested_flat_and_id() {
        let nested = parse_alias_updates(&json!({
            "nodes": [{"label": "sym:a", "add_aliases": ["pump broken", "no flow"]}]
        }));
        assert_eq!(nested.len(), 1);
        assert_eq!(nested[0].label, "sym:a");
        assert_eq!(nested[0].add_aliases, vec!["pump broken", "no flow"]);
        assert!(nested[0].new_label.is_none() && nested[0].new_type.is_none());

        // flat, using `id`, with a single comma-joined string
        let flat = parse_alias_updates(&json!({"id": "task:c", "add_aliases": "reset, reboot"}));
        assert_eq!(flat.len(), 1);
        assert_eq!(flat[0].label, "task:c");
        assert_eq!(flat[0].add_aliases, vec!["reset", "reboot"]);

        // nothing to update
        assert!(parse_alias_updates(&json!({})).is_empty());
    }

    /// The alias tool schema exposes read/search/grep/reach + graph_update, and must NOT expose
    /// graph_upsert — that removed affordance is the enforcement that no node/edge is created.
    #[test]
    fn alias_tools_schema_is_graph_update_only() {
        let schema = alias_tools_schema();
        let names: Vec<String> = schema
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| {
                t.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .map(String::from)
            })
            .collect();
        // ONLY graph_update — the harness pre-feeds the case source, so no read/reach/search
        // exploration and no graph_upsert (the model can neither create nodes/edges nor wander).
        assert_eq!(
            names,
            vec!["graph_update".to_string()],
            "alias mode must expose only graph_update: {names:?}"
        );
    }
}
