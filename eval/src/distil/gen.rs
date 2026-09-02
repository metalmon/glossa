//! `kbx distil`'s per-seed generator: an agentic pass over ONE grounded seed node, reusing the
//! SAME agent-loop substrate as `reason::chain_one_seed` (`run_agent_loop` + `lmstudio_chat` +
//! `schema_graph_block(ont)`) but READ-ONLY — the model gets the reader tools
//! (`search`/`read`/`grep`/`glob`/`glossary`/`reach`/`sql`) and a `propose_gold` tool, and NEVER
//! `graph_upsert`. The only thing this pass produces is an in-memory `GoldProposal`; nothing is
//! written to the graph.
//!
//! Mirrors `chain_one_seed`'s shape closely (same `paths`/`ont`/`lab` inputs, same
//! open-store-per-call pattern) so the two stay recognizably siblings — BOTH now walk BACKWARD
//! from a grounded terminal. `chain_one_seed` synthesizes the query-side reasoning layer that
//! leads to the terminal; `generate_one` instead invents a NEW (question, answer) pair whose
//! fixed ANSWER is that terminal fact and whose new entry angle requires reasoning to reach it.

use crate::backend::glossa_tools;
use crate::backend::openai::run_agent_loop;
use crate::backend::transport::openai::agent_chat_full;
use crate::lab::LabConfig;
use crate::reason::schema_graph_block;
use crate::score::contains_match;
use crate::workspace::KbxPaths;
use anyhow::anyhow;
use glossa::graph::ontology::Ontology;
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;
use glossa::tools::ChainSpec;
use glossa::trace::TraceLog;
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::HashSet;
use std::path::Path;
use std::time::Duration;

/// Depth/breadth caps on the backward incoming-chain gather ([`gather_incoming_chain`]) so a
/// densely-connected hub terminal can't explode the prompt: at most this many predecessor hops
/// back, and at most [`GATHER_MAX_LINKS`] links total.
const GATHER_MAX_DEPTH: usize = 3;
const GATHER_MAX_LINKS: usize = 6;

/// Per-link source-text cap (chars) when rendering the existing chain into the prompt — one chunk
/// can be large, and this block is only context ("what already exists"), not the working set.
const GATHER_SRC_MAX: usize = 400;

/// Top-K depth for the objective (code-B) `hop_type` retrieval probe (see [`code_b_hop_type`]).
const HOP_TYPE_TOPK: usize = 5;

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
    /// The MODEL's own hop-type self-report (`"lexical"`/`"multihop"`/`""` when absent). Advisory
    /// only — the emitted gold's `hop_type` is the OBJECTIVE code-B retrieval verdict
    /// ([`code_b_hop_type`]); this is kept solely to log model/retrieval disagreement.
    pub hop_type: String,
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
    let hop_type = args
        .get("hop_type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    Some(GoldProposal {
        question,
        answer,
        chain_node_ids,
        gate_ok,
        gate_reason,
        hop_type,
    })
}

/// OpenAI-function tool schema for the distil generator: ONLY `propose_gold`. The reader tools
/// (`search`/`read`/`grep`/`glob`/`glossary`/`reach`/`sql`) are deliberately OMITTED — the generator
/// is handed everything it needs up front (the terminal, its full grounded source text, and the
/// existing incoming chain), so it constructs the (question, answer) in ONE shot instead of
/// exploring. This removes the two failure modes of a tool-driven pass on a weak model: it can no
/// longer wander for `MAX_ROUNDS` and never call `propose_gold`, and it can no longer balloon the
/// trajectory with accumulated tool output into a context-length overflow. NO `graph_upsert` either
/// — this pass never writes to the graph.
fn distil_tools_schema() -> Value {
    let mut tools: Vec<Value> = Vec::new();
    tools.push(json!({
        "type": "function",
        "function": {
            "name": "propose_gold",
            "description": "Emit ONE synthetic (question, answer) gold: a NEW question whose fixed \
                ANSWER is the given grounded terminal, reachable only by reasoning through the \
                corpus to it. `chain_node_ids` lists the ids on the path you expect a reader to \
                walk, entry through terminal, in order. Set `gate_ok` to false (with a short \
                `gate_reason`) if the question is answerable without that reasoning, or if the \
                path doesn't actually reach the terminal answer — an honest false is useful, a \
                dishonest true is not. `hop_type` is your own read of the question's shape.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": { "type": "string" },
                    "answer": { "type": "string" },
                    "chain_node_ids": { "type": "array", "items": { "type": "string" } },
                    "gate_ok": { "type": "boolean" },
                    "gate_reason": { "type": "string" },
                    "hop_type": { "type": "string", "enum": ["lexical", "multihop"] }
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

/// One predecessor on an existing incoming reasoning chain: its node id/type/label plus the
/// grounded source text it was read from (empty when ungrounded). Ontology-general — carries only
/// whatever `node_type` the ontology assigned, never a hardcoded domain type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainLink {
    pub id: String,
    pub node_type: String,
    pub label: String,
    pub source_text: String,
}

/// Walk BACKWARD from a grounded terminal along CHAINING-role edges (the ontology's spine
/// relations, from [`ChainSpec::spine_rels`] — the SAME set `reason`/`glossary` chain on) via
/// `g.incoming`, collecting the existing predecessor links that already lead to this terminal. This
/// is the "what already exists" context shown to the model so it can invent a DIFFERENT entry angle
/// rather than reproduce a covered question. Bounded by [`GATHER_MAX_DEPTH`]/[`GATHER_MAX_LINKS`]
/// so a hub terminal can't explode the prompt. Ontology-general: the chaining edge set comes from
/// the ontology, never a named relation. Empty when the terminal has no incoming chaining edges.
pub(crate) fn gather_incoming_chain(
    root: &Path,
    idx: &DocIndex,
    g: &GraphStore,
    spec: &ChainSpec,
    terminal_id: &str,
) -> Vec<ChainLink> {
    let mut out: Vec<ChainLink> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    seen.insert(terminal_id.to_string());
    let mut frontier = vec![terminal_id.to_string()];
    let mut depth = 0;
    while depth < GATHER_MAX_DEPTH && out.len() < GATHER_MAX_LINKS && !frontier.is_empty() {
        let mut next: Vec<String> = Vec::new();
        for node in &frontier {
            for e in g.incoming(node).unwrap_or_default() {
                if !spec.spine_rels.iter().any(|r| r == &e.edge_type) {
                    continue;
                }
                if !seen.insert(e.from.clone()) {
                    continue;
                }
                if let Ok(Some(pred)) = g.get_node(&e.from) {
                    let source_text = seed_source_text(root, idx, g, &pred.id);
                    out.push(ChainLink {
                        id: pred.id.clone(),
                        node_type: pred.node_type,
                        label: pred.label,
                        source_text,
                    });
                    next.push(pred.id);
                    if out.len() >= GATHER_MAX_LINKS {
                        break;
                    }
                }
            }
            if out.len() >= GATHER_MAX_LINKS {
                break;
            }
        }
        frontier = next;
        depth += 1;
    }
    out
}

/// Count the DIRECT incoming chaining-role edges into `terminal_id` — the cheap per-terminal
/// "how many existing chains enter here" metric behind `--max-chains`. Each direct incoming
/// chaining edge is one existing chain arriving at the terminal; a terminal already fed by many
/// is well-covered and worth skipping so generation spreads to under-covered terminals.
/// Ontology-general (chaining set from [`ChainSpec`]); never walks deeper than one hop, by design.
pub(crate) fn incoming_chain_count(g: &GraphStore, spec: &ChainSpec, terminal_id: &str) -> usize {
    g.incoming(terminal_id)
        .unwrap_or_default()
        .iter()
        .filter(|e| spec.spine_rels.iter().any(|r| r == &e.edge_type))
        .count()
}

/// Pure code-B classifier: `"lexical"` when the terminal's grounding chunk id appears among the
/// question's top-K lexical (BM25) hit chunk ids — the answer is lexically reachable from the
/// question — else `"multihop"` (the question's own tokens don't retrieve the terminal, so a
/// reader must traverse to it). Both sides are `<path>#<ord>` chunk ids. No graph/glossary/PPR
/// involved — that is the whole point (a graph-aware probe would surface the terminal THROUGH the
/// reasoning chain and falsely read "lexical").
pub(crate) fn classify_hop_type(terminal_chunk: &str, topk_chunk_ids: &[String]) -> &'static str {
    if topk_chunk_ids.iter().any(|c| c == terminal_chunk) {
        "lexical"
    } else {
        "multihop"
    }
}

/// The terminal seed's grounding chunk id (`<path>#<ord>`): its first outgoing `MENTIONS` target.
/// `None` when ungrounded.
fn seed_grounding_chunk(g: &GraphStore, seed_id: &str) -> Option<String> {
    g.outgoing(seed_id)
        .unwrap_or_default()
        .into_iter()
        .find(|e| e.edge_type == glossa::graph::MENTIONS)
        .map(|e| e.to)
}

/// Objective (code-B) `hop_type` for a kept proposal, computed with NO model call: run the
/// question through the PURE LEXICAL document index (`DocIndex::search`, BM25 over the doc body —
/// graph-INDEPENDENT) and check whether the terminal's grounding chunk lands in the top-K
/// ([`HOP_TYPE_TOPK`]) hits via [`classify_hop_type`]. This is the label emitted for the gold.
/// Falls back to `"multihop"` when the terminal has no grounding chunk (nothing lexical to hit).
pub(crate) fn code_b_hop_type(
    idx: &DocIndex,
    g: &GraphStore,
    seed_id: &str,
    question: &str,
) -> String {
    let Some(chunk) = seed_grounding_chunk(g, seed_id) else {
        return "multihop".to_string();
    };
    let hits = idx.search(question, HOP_TYPE_TOPK).unwrap_or_default();
    let topk: Vec<String> = hits
        .iter()
        .map(|h| format!("{}#{}", h.path, h.ord))
        .collect();
    classify_hop_type(&chunk, &topk).to_string()
}

/// Truncate `s` to at most `max` chars (char-boundary safe), appending an ellipsis when cut — keeps
/// one large chunk of existing-chain context from dominating the prompt.
fn truncate_for_prompt(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max).collect();
        format!("{t}…")
    }
}

/// Render the existing incoming chain as an indented, bounded block for the prompt. Empty input
/// renders a clear "nothing yet" line so the model never sees a blank section.
fn format_existing_chain(links: &[ChainLink]) -> String {
    if links.is_empty() {
        return "(no existing chain currently leads to this terminal)".to_string();
    }
    links
        .iter()
        .map(|l| {
            let src = if l.source_text.is_empty() {
                String::new()
            } else {
                let t = truncate_for_prompt(&l.source_text, GATHER_SRC_MAX);
                format!("\n    {}", t.replace('\n', "\n    "))
            };
            format!("  - {} [{}] \"{}\"{}", l.id, l.node_type, l.label, src)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build the per-seed user message for the BACKWARD, terminal-anchored gold generator: the seed is
/// the fixed grounded terminal (the ANSWER), its source text grounds what the terminal is about,
/// and the existing incoming chain is shown ONLY as covered context to invent a different entry
/// angle from. Pure (no I/O) so it is unit-testable; ontology-general (names no concrete types).
pub(crate) fn build_distil_user_message(
    seed: &Seed,
    source_text: &str,
    existing: &[ChainLink],
) -> String {
    let body = if source_text.is_empty() {
        "(no grounded source text found for this terminal)"
    } else {
        source_text
    };
    format!(
        "Grounded TERMINAL node — this is the fixed ANSWER: {} [{}] \"{}\"\nIts grounded source \
         text:\n{}\n\nA chain that ALREADY leads to this terminal (shown ONLY so you can see what \
         is already covered and understand what the terminal is about — do NOT reproduce it or its \
         question):\n{}\n\nInvent ONE NEW question, from a DIFFERENT entry angle, whose answer is \
         this same terminal fact and which a reader can reach only by reasoning through the corpus \
         to it. End with exactly one `propose_gold` call.",
        seed.id,
        seed.node_type,
        seed.label,
        body,
        format_existing_chain(existing),
    )
}

/// Forces the generator to end with a `propose_gold` TOOL CALL, not a plain text turn. The shared
/// agent loop treats the first tool-free text turn as a final answer and returns — correct for the
/// reader, wrong here: a distil attempt is only recorded through `propose_gold`, so a model that
/// "thinks out loud" instead of calling the tool would silently waste the whole attempt (observed:
/// weak models emit a prose answer or an empty turn and never call the tool). This reuses the loop's
/// existing `DialogueGate` seam (the same one `user_sim` uses): on a tool-free turn it deflects with
/// a nudge to call `propose_gold` and lets the loop continue, UNTIL a proposal has actually been
/// captured — then it accepts, so the loop exits cleanly right after the tool call. `resample` still
/// retries a genuinely degenerate (empty/looped/length) turn underneath this.
struct ProposeGoldGate<'a> {
    proposal: &'a RefCell<Option<GoldProposal>>,
}

impl crate::backend::user_sim::DialogueGate for ProposeGoldGate<'_> {
    fn judge(
        &self,
        _q: &str,
        _messages: &[Value],
        _proposed: &str,
    ) -> anyhow::Result<Option<String>> {
        if self.proposal.borrow().is_some() {
            // The tool call already landed — accept this closing text turn and let the loop end.
            Ok(None)
        } else {
            // No proposal yet: push the model back to the tool instead of accepting a text answer.
            Ok(Some(
                "You have not called `propose_gold` yet. A plain text reply is NOT recorded — your \
                 work counts ONLY through the `propose_gold` tool call. Call `propose_gold` now with \
                 question, answer, chain_node_ids, gate_ok, gate_reason, and hop_type (use \
                 gate_ok=false if you could not build a usable question)."
                    .to_string(),
            ))
        }
    }
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
    let spec = ChainSpec::from_ontology(ont);

    let source_text = seed_source_text(root, &idx, &g, &seed.id);
    // The existing incoming chain(s) that already lead to this terminal — ontology-general backward
    // walk along chaining-role edges, shown to the model only as covered context (invent a new
    // entry angle, don't reproduce it).
    let existing = gather_incoming_chain(root, &idx, &g, &spec, &seed.id);
    let system = format!("{}\n\n{distil_md}", schema_graph_block(ont));
    let user = build_distil_user_message(seed, &source_text, &existing);
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

    // Gate the loop on an actual `propose_gold` tool call (see `ProposeGoldGate`): deflect tool-free
    // text turns back to the tool instead of accepting them as a final answer, until a proposal is
    // captured. Extract via `borrow_mut().take()` (not `into_inner`, which would move `proposal`
    // while the gate still borrows it).
    let gate = ProposeGoldGate {
        proposal: &proposal,
    };
    run_agent_loop(chat, messages, exec, on_repeat, MAX_ROUNDS, Some(&gate))?;

    let Some(proposal) = proposal.borrow_mut().take() else {
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
    let leak_resp = crate::backend::openai::chat_once_resampled(distil_ep, &leak_messages)?;
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
            "gate_reason": "chain checks out",
            "hop_type": " multihop "
        });
        let p = parse_propose_gold(&args).expect("well-formed payload parses");
        assert_eq!(p.question, "what step follows the seed?");
        assert_eq!(p.answer, "the terminal fact");
        assert_eq!(p.chain_node_ids, vec!["seed-1", "mid-1", "term-1"]);
        assert!(p.gate_ok);
        assert_eq!(p.gate_reason, "chain checks out");
        assert_eq!(p.hop_type, "multihop", "hop_type parsed and trimmed");
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
        assert!(
            p.hop_type.is_empty(),
            "hop_type defaults to empty when absent"
        );
    }

    #[test]
    fn classify_hop_type_lexical_when_terminal_in_topk_else_multihop() {
        let topk = vec![
            "a.md#1".to_string(),
            "b.md#3".to_string(),
            "c.md#2".to_string(),
        ];
        assert_eq!(classify_hop_type("b.md#3", &topk), "lexical");
        assert_eq!(classify_hop_type("z.md#9", &topk), "multihop");
        assert_eq!(
            classify_hop_type("a.md#1", &[]),
            "multihop",
            "empty hit list is never lexical"
        );
    }

    // ---- backward incoming-chain gatherer (ontology-general, chaining-role edges only) ----------

    const CHAIN_ONT: &str = r#"
[entities.Fact]
requires_grounding = true
[entities.Document]
[entities.Section]

[relations.LEADS_TO]
from = ["Fact"]
to = ["Fact"]
role = "chaining"

[reasoning]
[[reasoning.spines]]
anchor = "Fact"
relations = ["LEADS_TO"]
"#;

    fn prov() -> glossa::graph::store::Provenance {
        glossa::graph::store::Provenance {
            source_path: "d.md".into(),
            range: None,
            file_sig: None,
            origin: "test".into(),
            confidence: 0.9,
            created_at: 1,
        }
    }

    /// A terminal with two grounded predecessors reachable via chaining-role incoming edges is
    /// gathered (each predecessor's grounded chunk text included); a terminal with no incoming
    /// chaining edges gathers nothing. Ontology-general: the chaining set comes from the ontology.
    #[test]
    fn gather_incoming_chain_collects_grounded_predecessors_and_is_empty_without_them() {
        use glossa::graph::store::{Edge, Node};
        let dir = tempfile::tempdir().unwrap();
        let ont = Ontology::parse(CHAIN_ONT).unwrap();
        let spec = ChainSpec::from_ontology(&ont);
        let g = GraphStore::open(dir.path()).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[
            glossa::model::Chunk {
                doc_path: "d.md".into(),
                location: "S1".into(),
                file_type: "md".into(),
                text: "PRED ONE body".into(),
            },
            glossa::model::Chunk {
                doc_path: "d.md".into(),
                location: "S2".into(),
                file_type: "md".into(),
                text: "PRED TWO body".into(),
            },
        ])
        .unwrap();

        for (id, label, chunk) in [
            ("term", "terminal", 3u64),
            ("pred-a", "pred a", 1),
            ("pred-b", "pred b", 2),
        ] {
            g.put_node(&Node {
                id: id.into(),
                node_type: "Fact".into(),
                label: label.into(),
                aliases: vec![],
                prov: prov(),
            })
            .unwrap();
            g.put_edge(&Edge {
                from: id.into(),
                to: format!("d.md#{chunk}"),
                edge_type: glossa::graph::MENTIONS.to_string(),
                prov: prov(),
            })
            .unwrap();
        }
        // pred-a --LEADS_TO--> term, pred-b --LEADS_TO--> term (chaining, incoming to term).
        for from in ["pred-a", "pred-b"] {
            g.put_edge(&Edge {
                from: from.into(),
                to: "term".into(),
                edge_type: "LEADS_TO".into(),
                prov: prov(),
            })
            .unwrap();
        }

        let links = gather_incoming_chain(dir.path(), &idx, &g, &spec, "term");
        let ids: HashSet<&str> = links.iter().map(|l| l.id.as_str()).collect();
        assert!(
            ids.contains("pred-a") && ids.contains("pred-b"),
            "both predecessors: {ids:?}"
        );
        let texts: String = links.iter().map(|l| l.source_text.clone()).collect();
        assert!(
            texts.contains("PRED ONE body") && texts.contains("PRED TWO body"),
            "grounded text: {texts}"
        );

        // A terminal with no incoming chaining edges gathers nothing.
        assert!(
            gather_incoming_chain(dir.path(), &idx, &g, &spec, "pred-a").is_empty(),
            "no incoming chaining edges -> empty"
        );
    }

    /// `incoming_chain_count` counts direct incoming chaining-role edges (and ignores non-chaining
    /// incoming edges like MENTIONS), driving the `--max-chains` skip metric.
    #[test]
    fn incoming_chain_count_counts_direct_chaining_edges_only() {
        use glossa::graph::store::{Edge, Node};
        let dir = tempfile::tempdir().unwrap();
        let ont = Ontology::parse(CHAIN_ONT).unwrap();
        let spec = ChainSpec::from_ontology(&ont);
        let g = GraphStore::open(dir.path()).unwrap();
        for id in ["term", "pred-a", "pred-b"] {
            g.put_node(&Node {
                id: id.into(),
                node_type: "Fact".into(),
                label: id.into(),
                aliases: vec![],
                prov: prov(),
            })
            .unwrap();
        }
        for from in ["pred-a", "pred-b"] {
            g.put_edge(&Edge {
                from: from.into(),
                to: "term".into(),
                edge_type: "LEADS_TO".into(),
                prov: prov(),
            })
            .unwrap();
        }
        assert_eq!(incoming_chain_count(&g, &spec, "term"), 2);
        assert_eq!(incoming_chain_count(&g, &spec, "pred-a"), 0);
    }

    #[test]
    fn parse_propose_gold_rejects_missing_or_blank_question_or_answer() {
        assert!(parse_propose_gold(&json!({ "answer": "a" })).is_none());
        assert!(parse_propose_gold(&json!({ "question": "q?" })).is_none());
        assert!(parse_propose_gold(&json!({ "question": "  ", "answer": "a" })).is_none());
        assert!(parse_propose_gold(&json!({ "question": "q?", "answer": "  " })).is_none());
    }

    #[test]
    fn distil_tools_schema_is_propose_gold_only_no_reader_tools_no_graph_upsert() {
        // One-shot generation: the model gets all context up front (terminal + source + incoming
        // chain), so the ONLY tool is propose_gold — no reader tools (no wandering / no overflow),
        // and never graph_upsert (read-only pass).
        let tools = distil_tools_schema();
        let names: Vec<String> = tools
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t.pointer("/function/name").and_then(|n| n.as_str()))
            .map(String::from)
            .collect();
        assert_eq!(
            names,
            vec!["propose_gold".to_string()],
            "distil generator must advertise ONLY propose_gold: {names:?}"
        );
        for forbidden in ["search", "read", "grep", "glossary", "reach", "sql", "graph_upsert"] {
            assert!(
                !names.contains(&forbidden.to_string()),
                "distil must not advertise {forbidden}: {names:?}"
            );
        }
    }
}
