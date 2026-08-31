//! `kbx distil`'s per-seed generator: an agentic pass over ONE grounded seed node, reusing the
//! SAME agent-loop substrate as `reason::chain_one_seed` (`run_agent_loop` + `lmstudio_chat` +
//! `schema_graph_block(ont)`) but READ-ONLY — the model gets the reader tools
//! (`search`/`read`/`grep`/`glob`/`glossary`/`reach`/`sql`) and a `propose_gold` tool, and NEVER
//! `graph_upsert`. The only thing this pass produces is an in-memory `GoldProposal`; nothing is
//! written to the graph.
//!
//! Mirrors `chain_one_seed`'s shape closely (same `paths`/`ont`/`lab` inputs, same
//! open-store-per-call pattern) so the two stay recognizably siblings — `chain_one_seed` walks
//! backward from a grounded terminal; `generate_one` walks FORWARD from a seed to invent a new
//! (question, answer) pair instead.

use crate::backend::glossa_tools;
use crate::backend::openai::{chat_once, run_agent_loop};
use crate::backend::transport::openai::agent_chat_full;
use crate::lab::LabConfig;
use crate::reason::schema_graph_block;
use crate::score::contains_match;
use crate::workspace::KbxPaths;
use anyhow::anyhow;
use glossa::graph::ontology::Ontology;
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;
use glossa::trace::TraceLog;
use serde_json::{json, Value};
use std::cell::RefCell;
use std::time::Duration;

/// High cap on tool-call rounds for one seed's generate pass — mirrors `chain_one_seed`'s
/// `MAX_ROUNDS`, trimmed a little since this pass has no writes to make, only exploration plus
/// one `propose_gold` call.
const MAX_ROUNDS: usize = 20;

/// One grounded seed node candidate to generate a synthetic gold from — a node's id/type/label,
/// enough to introduce it to the model without re-deriving it from the graph mid-prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Seed {
    pub id: String,
    pub node_type: String,
    pub label: String,
}

/// A parsed `propose_gold` tool-call payload.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GoldProposal {
    pub question: String,
    pub answer: String,
    pub chain_node_ids: Vec<String>,
    pub gate_ok: bool,
    pub gate_reason: String,
}

/// Why an attempted synthetic gold was NOT kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DropReason {
    /// The model never called `propose_gold` (or its final call was unparseable).
    NoProposal,
    /// The model's own `gate_ok` came back false — it judged the question answerable without
    /// the chain, or the chain didn't actually reach the answer.
    SelfGateFailed(String),
    /// The adversarial leak check answered the question correctly with no tools and no chain.
    Leaked,
}

impl DropReason {
    /// Short human-readable reason, for run-summary logging.
    pub fn describe(&self) -> String {
        match self {
            DropReason::NoProposal => "no propose_gold call".to_string(),
            DropReason::SelfGateFailed(why) => format!("self-gate failed: {why}"),
            DropReason::Leaked => "leaked (answerable question-alone)".to_string(),
        }
    }
}

/// The outcome of one `generate_one` attempt: either a kept, gated proposal, or why it was dropped.
#[derive(Debug, Clone, PartialEq)]
pub enum GenOutcome {
    Kept(GoldProposal),
    Dropped(DropReason),
}

/// Parse a `propose_gold` tool call's JSON arguments into a [`GoldProposal`]. `question`/`answer`
/// must both be present and non-blank after trimming — nothing sane to keep without them, so a
/// missing/empty one makes the whole call unparseable (`None`). `chain_node_ids`/`gate_reason` are
/// tolerant (default to empty); `gate_ok` defaults to `false` when absent — an unmarked proposal
/// is treated as not self-gated, never as an accidental pass.
pub fn parse_propose_gold(args: &Value) -> Option<GoldProposal> {
    let question = args.get("question")?.as_str()?.trim().to_string();
    let answer = args.get("answer")?.as_str()?.trim().to_string();
    if question.is_empty() || answer.is_empty() {
        return None;
    }
    let chain_node_ids = args
        .get("chain_node_ids")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let gate_ok = args
        .get("gate_ok")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let gate_reason = args
        .get("gate_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    Some(GoldProposal {
        question,
        answer,
        chain_node_ids,
        gate_ok,
        gate_reason,
    })
}

/// OpenAI-function tool schema for the distil generator: the full read-only reader registry
/// (`search`/`read`/`grep`/`glob`/`glossary`/`reach`/`sql` — glossa's registry order, unfiltered;
/// distil always has a graph open, so nothing is gated here) plus `propose_gold`. Deliberately NO
/// `graph_upsert` — this pass never writes to the graph.
fn distil_tools_schema() -> Value {
    let mut tools: Vec<Value> = glossa::tools::registry::registry()
        .iter()
        .map(|d| {
            json!({
                "type": "function",
                "function": {
                    "name": d.name,
                    "description": d.description,
                    "parameters": d.params_schema,
                }
            })
        })
        .collect();
    tools.push(json!({
        "type": "function",
        "function": {
            "name": "propose_gold",
            "description": "Emit ONE synthetic (question, answer) gold once you've traced a real \
                chain from the seed node along the ontology's relations to a grounded terminal \
                fact. `chain_node_ids` lists the ids you walked, seed through terminal, in order. \
                Set `gate_ok` to false (with a short `gate_reason`) if the question is answerable \
                without the chain, or if the chain you traced doesn't actually reach the answer — \
                an honest false is useful, a dishonest true is not.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": { "type": "string" },
                    "answer": { "type": "string" },
                    "chain_node_ids": { "type": "array", "items": { "type": "string" } },
                    "gate_ok": { "type": "boolean" },
                    "gate_reason": { "type": "string" }
                },
                "required": ["question", "answer", "gate_ok"]
            }
        }
    }));
    Value::Array(tools)
}

/// The grounded source text for `seed_id`'s first outgoing MENTIONS edge (empty string if it has
/// none, or if the target doesn't resolve to a readable chunk) — gives the model real corpus text
/// to start from instead of just a bare id/label.
fn seed_source_text(
    root: &std::path::Path,
    idx: &DocIndex,
    g: &GraphStore,
    seed_id: &str,
) -> String {
    let target = g
        .outgoing(seed_id)
        .unwrap_or_default()
        .into_iter()
        .find(|e| e.edge_type == glossa::graph::MENTIONS)
        .map(|e| e.to);
    let Some(target) = target else {
        return String::new();
    };
    let (path, n) = match target.rsplit_once('#') {
        Some((p, n)) => (p.to_string(), n.parse::<u64>().unwrap_or(0)),
        None => (target, 0),
    };
    let trace = TraceLog::disabled();
    let (text, _images) = glossa_tools::run_read(root, idx, None, &path, n, false, &trace);
    text
}

/// Run one seed's generate + verify-gate pass: a strong model (`lab.distil`) explores the corpus
/// via the read-only reader tools from `seed`, then either proposes a gated synthetic gold via
/// `propose_gold` or leaves nothing to keep. Verify-gate is MVP-level: the model's own `gate_ok`,
/// plus one extra adversarial call — the question ALONE, no tools, no chain — that drops the
/// proposal if it answers correctly anyway (a leak). NEVER calls `graph_upsert`; opens its own
/// `GraphStore`/`DocIndex` per call, mirroring `chain_one_seed`.
pub fn generate_one(
    paths: &KbxPaths,
    ont: &Ontology,
    lab: &LabConfig,
    distil_md: &str,
    seed: &Seed,
) -> anyhow::Result<GenOutcome> {
    let distil_ep = lab
        .distil
        .as_ref()
        .ok_or_else(|| anyhow!("kbx distil needs a [distil] endpoint in lab.toml"))?;

    let root = paths.root.as_path();
    let g = GraphStore::open(root)?;
    let idx = DocIndex::open_or_create(root)?;
    let trace = TraceLog::disabled();
    let spec = glossa::tools::ChainSpec::from_ontology(ont);

    let source_text = seed_source_text(root, &idx, &g, &seed.id);
    let system = format!("{}\n\n{distil_md}", schema_graph_block(ont));
    let user = format!(
        "Seed node: {} [{}] \"{}\"\nGrounded source text:\n{}\n\nExplore from this seed, trace a \
         real chain along the ontology's relations, and call `propose_gold` with one question \
         answerable only by following that chain, its grounded terminal answer, and your honest \
         self-gate.",
        seed.id,
        seed.node_type,
        seed.label,
        if source_text.is_empty() {
            "(no grounded source text found for this seed)"
        } else {
            source_text.as_str()
        }
    );
    let messages = vec![
        json!({ "role": "system", "content": system }),
        json!({ "role": "user", "content": user }),
    ];

    let endpoint = distil_ep.endpoint.clone();
    let model = distil_ep.model.clone();
    let api_key = distil_ep.resolve_key();
    let timeout = Duration::from_secs(distil_ep.timeout_secs);
    let tools = distil_tools_schema();

    // Full-response one-shot; resampling is applied provider-neutrally by the agent loop
    // (`backend::resample::call_with_resample`).
    let chat = |messages: &[Value]| {
        agent_chat_full(
            &endpoint,
            &model,
            api_key.as_deref(),
            &tools,
            messages,
            timeout,
        )
    };

    let proposal: RefCell<Option<GoldProposal>> = RefCell::new(None);
    let exec = |name: &str, args: &Value| -> (String, Vec<String>, Vec<glossa::read::DocImage>) {
        if name == "propose_gold" {
            match parse_propose_gold(args) {
                Some(p) => {
                    let ids = p.chain_node_ids.clone();
                    let msg = format!(
                        "recorded proposal (gate_ok={}): \"{}\" -> \"{}\"",
                        p.gate_ok, p.question, p.answer
                    );
                    *proposal.borrow_mut() = Some(p);
                    (msg, ids, Vec::new())
                }
                None => (
                    "propose_gold rejected: `question` and `answer` are required and must be \
                     non-empty"
                        .to_string(),
                    Vec::new(),
                    Vec::new(),
                ),
            }
        } else {
            // Distil's exploratory read never feeds vision input — discard, mirroring the reader
            // and reason paths (only `kbx build --vision` populates this).
            let (body, ids, _images) =
                glossa_tools::exec(name, args, root, &idx, Some(&g), &spec, &trace);
            (body, ids, Vec::new())
        }
    };

    let on_repeat = |name: &str, _args: &Value| {
        format!(
            "(dup {name}) you already called this — try a different tool, a different query, \
             or move on to propose_gold"
        )
    };

    run_agent_loop(chat, messages, exec, on_repeat, MAX_ROUNDS, None)?;

    let Some(proposal) = proposal.into_inner() else {
        return Ok(GenOutcome::Dropped(DropReason::NoProposal));
    };
    if !proposal.gate_ok {
        return Ok(GenOutcome::Dropped(DropReason::SelfGateFailed(
            proposal.gate_reason.clone(),
        )));
    }

    // Leak check (cheap adversarial): one extra strong-model call with the QUESTION ALONE — no
    // tools, no chain. If it answers correctly anyway, the question tests nothing and is dropped
    // regardless of the model's own gate_ok.
    let leak_messages = vec![json!({ "role": "user", "content": proposal.question.clone() })];
    let leak_resp = chat_once(
        &endpoint,
        &model,
        &leak_messages,
        api_key.as_deref(),
        distil_ep.timeout_secs,
        distil_ep.resolve_temperature(),
    )?;
    let leak_text = leak_resp
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    if contains_match(leak_text, &proposal.answer) {
        return Ok(GenOutcome::Dropped(DropReason::Leaked));
    }

    Ok(GenOutcome::Kept(proposal))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_propose_gold_reads_full_payload() {
        let args = json!({
            "question": " what step follows the seed? ",
            "answer": " the terminal fact ",
            "chain_node_ids": ["seed-1", "mid-1", "term-1"],
            "gate_ok": true,
            "gate_reason": "chain checks out"
        });
        let p = parse_propose_gold(&args).expect("well-formed payload parses");
        assert_eq!(p.question, "what step follows the seed?");
        assert_eq!(p.answer, "the terminal fact");
        assert_eq!(p.chain_node_ids, vec!["seed-1", "mid-1", "term-1"]);
        assert!(p.gate_ok);
        assert_eq!(p.gate_reason, "chain checks out");
    }

    #[test]
    fn parse_propose_gold_defaults_optional_fields() {
        let args = json!({ "question": "q?", "answer": "a" });
        let p = parse_propose_gold(&args).expect("minimal payload parses");
        assert!(p.chain_node_ids.is_empty());
        assert!(
            !p.gate_ok,
            "gate_ok must default to false, never an accidental pass"
        );
        assert!(p.gate_reason.is_empty());
    }

    #[test]
    fn parse_propose_gold_rejects_missing_or_blank_question_or_answer() {
        assert!(parse_propose_gold(&json!({ "answer": "a" })).is_none());
        assert!(parse_propose_gold(&json!({ "question": "q?" })).is_none());
        assert!(parse_propose_gold(&json!({ "question": "  ", "answer": "a" })).is_none());
        assert!(parse_propose_gold(&json!({ "question": "q?", "answer": "  " })).is_none());
    }

    #[test]
    fn distil_tools_schema_advertises_reader_tools_and_propose_gold_but_never_graph_upsert() {
        let tools = distil_tools_schema();
        let names: Vec<String> = tools
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t.pointer("/function/name").and_then(|n| n.as_str()))
            .map(String::from)
            .collect();
        for t in [
            "search",
            "read",
            "grep",
            "glob",
            "glossary",
            "reach",
            "sql",
            "propose_gold",
        ] {
            assert!(
                names.contains(&t.to_string()),
                "missing tool {t}: {names:?}"
            );
        }
        assert!(
            !names.contains(&"graph_upsert".to_string()),
            "distil must NEVER advertise graph_upsert (read-only): {names:?}"
        );
    }
}
