//! Backward-compatible agent-loop shim over the generic `agent_loop::run_agent_loop`.
//!
//! Adapts the pre-transport calling convention (`chat: Fn(&[Value]) -> Result<Value>`) onto the
//! generic transport-driven loop, so the callers not yet migrated to `ChatTransport` directly
//! (`build::extract`, `distil::chain`, `gepa_graph`) keep compiling and behaving identically.
//! Split out of `backend::openai`; kept SEPARATE from the generic `backend::agent_loop` (which
//! holds the loop itself).

use crate::backend::transport::openai::{content_of, parse_tool_args, tools_schema_from_ctx};
use crate::backend::transport::{ChatTransport, ToolCall, TurnReply};
use crate::backend::vision::vision_user_message;
use glossa::read::DocImage;
use glossa::trace::TraceLog;
use serde_json::{json, Value};
use std::path::Path;
use std::rc::Rc;

/// Thin backward-compatible shim over `agent_loop::run_agent_loop`: adapts the pre-transport
/// calling convention (`chat: Fn(&[Value]) -> Result<Value>`, returning the assistant `message`
/// object already extracted from `choices[0].message`) onto the generic, transport-driven loop.
///
/// All dedup/streak/NBA logic now lives in exactly ONE place (`agent_loop::run_agent_loop`) — this
/// function does not reimplement any of it, it only translates the closure into a one-off
/// `ChatTransport` (`ClosureTransport`, below) so `run_agent_loop`'s three callers not yet migrated
/// to `ChatTransport` directly (`build::extract`, `distil::chain`, `gepa_graph` — Phase 2 Task 6 of
/// the multi-api-transport plan) keep compiling and behaving identically. `OpenAiBackend::answer`
/// itself no longer goes through this shim — it drives `agent_loop::run_agent_loop` directly with
/// a real `OpenAiTransport`.
pub(crate) fn run_agent_loop<C, F, N>(
    chat: C,
    messages: Vec<Value>,
    exec: F,
    on_repeat: N,
    max_rounds: usize,
    user_sim: Option<&dyn crate::backend::user_sim::DialogueGate>,
) -> anyhow::Result<String>
where
    C: FnMut(&[Value]) -> anyhow::Result<Value>,
    F: FnMut(&str, &Value) -> (String, Vec<String>, Vec<DocImage>),
    N: Fn(&str, &Value) -> String,
{
    // Images surfaced by `exec` during the current round ride here until the transport's
    // `push_tool_results` drains them into ONE follow-up vision user message (build --vision only;
    // every other caller's `exec` returns an empty image vec, so nothing is ever appended and the
    // transcript stays byte-identical to the non-vision path). Shared via `Rc<RefCell<…>>` so the
    // 2-tuple `exec` adapter below and the `ClosureTransport` both reach one buffer without either
    // borrowing the other — the generic `agent_loop::run_agent_loop` seam is deliberately image-
    // agnostic (its `exec` is `(String, Vec<String>)`), so vision threading lives HERE in the shim,
    // keeping that one loop implementation free of any image concern. The per-conversation prefix
    // reset (`reset_conversation_prefix`) now happens inside `agent_loop::run_agent_loop` itself, so
    // this shim and `OpenAiBackend::answer` (which drives that loop directly) both get it.
    let pending_images: Rc<std::cell::RefCell<Vec<DocImage>>> =
        Rc::new(std::cell::RefCell::new(Vec::new()));
    let sink = Rc::clone(&pending_images);
    let mut exec = exec;
    let exec2 = move |name: &str, args: &Value| {
        let (body, ids, images) = exec(name, args);
        if !images.is_empty() {
            sink.borrow_mut().extend(images);
        }
        (body, ids)
    };
    let transport = ClosureTransport::new(chat, pending_images);
    // A blank `Endpoint`: `ClosureTransport::call` ignores it entirely (the wrapped closure
    // already captures its own endpoint/model/api_key/tools, exactly as the old direct callers
    // of `lmstudio_chat` did).
    let ep = crate::lab::Endpoint {
        endpoint: String::new(),
        model: String::new(),
        api_key: String::new(),
        api_key_env: String::new(),
        timeout_secs: 120,
        api: crate::lab::ApiKind::default(),
        temperature: None,
        // Blank endpoint -> no throttle, no fallback (the shim's `ClosureTransport` captures the
        // real endpoint). `call_resilient` therefore makes exactly one primary call, unchanged.
        rate_limit: None,
        fallback: Vec::new(),
        function_name: None,
        feedback_score_metric: None,
        feedback_bool_metric: None,
        headers: std::collections::BTreeMap::new(),
    };
    crate::backend::agent_loop::run_agent_loop(
        &transport, &ep, None, messages, None, exec2, on_repeat, max_rounds, user_sim,
    )
}

/// Adapts a legacy `FnMut(&[Value]) -> Result<Value>` chat closure into a one-off `ChatTransport`,
/// so `run_agent_loop`'s shim above can drive the generic loop. `push_assistant_turn`/
/// `push_tool_results`/tool-call parsing mirror `OpenAiTransport`'s shapes exactly (same raw-message
/// echo, same per-call `{role:"tool"}` push). `pending_images` is the shim's shared vision buffer:
/// `push_tool_results` drains it into a follow-up `role:"user"` image message right after the tool
/// results (build --vision).
///
/// The closure returns EITHER the FULL chat response (the real callers, via
/// `transport::openai::agent_chat_full` — so `choices[0].finish_reason` is available to the
/// resample layer) OR a bare assistant message object (this module's shim unit tests). `call`
/// accepts both: it uses the full-response shape when `choices[0].message` is present (also lifting
/// `finish_reason`), and otherwise treats the returned `Value` as the message itself
/// (`finish_reason` then `None`).
struct ClosureTransport<C> {
    chat: std::cell::RefCell<C>,
    pending_images: Rc<std::cell::RefCell<Vec<DocImage>>>,
}

impl<C> ClosureTransport<C> {
    fn new(chat: C, pending_images: Rc<std::cell::RefCell<Vec<DocImage>>>) -> Self {
        Self {
            chat: std::cell::RefCell::new(chat),
            pending_images,
        }
    }
}

impl<C> ChatTransport for ClosureTransport<C>
where
    C: FnMut(&[Value]) -> anyhow::Result<Value>,
{
    fn tools_schema(&self, ctx: &glossa::tools::registry::ToolContext) -> Value {
        tools_schema_from_ctx(ctx)
    }

    fn call(
        &self,
        _ep: &crate::lab::Endpoint,
        _system: Option<&str>,
        messages: &[Value],
        _tools: Option<&Value>,
        _temperature: Option<f64>,
    ) -> anyhow::Result<TurnReply> {
        let v = (self.chat.borrow_mut())(messages)?;
        // Full response (real callers) vs bare message (shim tests): a full response carries
        // `choices[0].message` + `choices[0].finish_reason`; a bare message IS the message.
        let has_choices = v.pointer("/choices/0/message").is_some();
        let finish_reason = v
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
            .map(str::to_string);
        let msg = if has_choices {
            v.pointer("/choices/0/message").cloned().unwrap_or(v)
        } else {
            v
        };
        let tool_calls: Vec<ToolCall> = msg
            .get("tool_calls")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .map(|call| ToolCall {
                        id: call
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        name: call
                            .pointer("/function/name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        args: parse_tool_args(call),
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(TurnReply {
            text: Some(content_of(&msg)),
            tool_calls,
            finish_reason,
            raw: msg,
        })
    }

    fn push_assistant_turn(&self, messages: &mut Vec<Value>, reply: &TurnReply) {
        messages.push(reply.raw.clone());
    }

    fn push_tool_results(&self, messages: &mut Vec<Value>, results: &[(String, String)]) {
        for (id, body) in results {
            messages.push(json!({ "role": "tool", "tool_call_id": id, "content": body }));
        }
        // Vision (build --vision only): a `role:"tool"` message can't carry image content on this
        // endpoint, so any images this round's `exec` surfaced (buffered by the shim's 2-tuple
        // adapter) ride in ONE follow-up `role:"user"` message right after the tool results. Empty
        // for every non-vision caller, so `vision_user_message` returns `None` and nothing is added.
        let images: Vec<DocImage> = self.pending_images.borrow_mut().drain(..).collect();
        if let Some(img_msg) = vision_user_message(&images) {
            messages.push(img_msg);
        }
    }
}

/// Execute one glossa tool in-process against the corpus in `work`, logging it to the trace
/// (same shape as the MCP server: search → array of {path,location,score}; read → {path}).
///
/// Returns `(body, ids, images)` — `ids` are the identifiers this call surfaced (what a
/// session-aware MCP server would track for novelty), from `glossa_tools::exec`'s second return
/// value: `search`'s hit locations, and the graph tools' (glossary/related/neighbors/reach/sql)
/// `path#ord` read-anchor ids scraped from their rendered bodies. `read` itself surfaces no ids
/// there, so it's special-cased here to the `path` argument instead. `run_agent_loop` uses these to
/// detect an unproductive streak — many varied calls (including varied graph navigation) that
/// surface nothing new — without falsely tripping on a reader that IS making real graph progress.
///
/// `images` are the page images `glossa_tools::exec` surfaced (e.g. a `read(page_image)` on a
/// scanned page), forwarded verbatim — never dropped here. The caller decides whether to feed them:
/// under `--vision` the answer path rides them to the model in a follow-up `role:"user"` image
/// message (see `vision_user_message` / `VisionTransport`); a non-vision caller simply ignores this
/// element, so the transcript is byte-identical to before.
pub(crate) fn execute_tool(
    name: &str,
    args: &Value,
    root: &Path,
    idx: &glossa::index::store::DocIndex,
    graph: Option<&glossa::graph::store::GraphStore>,
    spec: &glossa::tools::ChainSpec,
    trace: &TraceLog,
) -> (String, Vec<String>, Vec<DocImage>) {
    // No registry-membership pre-check here: `registry()` is the ADVERTISING source (what
    // `tools_schema` puts in front of the model); execution dispatches whatever
    // `glossa_tools::exec` supports, which is a superset (it also serves non-agent-facing
    // callers, e.g. related/neighbors for MCP's Editor/Full profiles). `exec` already returns
    // its own "unknown tool" body for names it genuinely doesn't handle, so it is the sole gate.
    let (body, ids, images) =
        crate::backend::glossa_tools::exec(name, args, root, idx, graph, spec, trace);
    let ids = if name == "read" {
        // Mirror glossa_tools::exec's own raw_arguments fallback so a stringified args object
        // still yields the path.
        let parsed;
        let a = if let Some(s) = args.as_str() {
            parsed = serde_json::from_str::<Value>(s).unwrap_or_else(|_| json!({}));
            &parsed
        } else {
            args
        };
        a.get("path")
            .and_then(|v| v.as_str())
            .filter(|p| !p.is_empty())
            .map(|p| vec![p.to_string()])
            .unwrap_or_default()
    } else {
        ids
    };
    (body, ids, images)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::agent_loop::UNPRODUCTIVE_STREAK_K;
    use std::cell::RefCell;

    /// Default `on_repeat` for tests that don't exercise NBA: a static nudge.
    fn nudge(name: &str, _args: &Value) -> String {
        format!("(dup {name}) you already called this — try a different tool or change the query")
    }

    fn stub_image(tag: u8) -> DocImage {
        DocImage {
            mime: "image/jpeg".to_string(),
            bytes: vec![0xFF, 0xD8, 0xFF, tag],
        }
    }

    #[test]
    fn loop_unproductive_streak_never_fires_when_calls_are_productive() {
        // Every call surfaces a brand-new id, so the streak resets each time — the steer must never
        // fire even past K calls.
        let round = RefCell::new(0usize);
        let rounds_to_run = UNPRODUCTIVE_STREAK_K + 4;
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            if let Some(last_tool) = msgs.iter().rev().find(|m| m["role"] == "tool") {
                let c = last_tool["content"].as_str().unwrap_or("");
                assert!(
                    !c.to_lowercase().contains("no new information"),
                    "steer must not fire on productive calls, got: {c:?}"
                );
            }
            if *r > rounds_to_run {
                return Ok(json!({ "role": "assistant", "content": "ANSWER: done" }));
            }
            Ok(json!({
                "role": "assistant", "content": "searching",
                "tool_calls": [{
                    "id": format!("c{}", *r),
                    "function": { "name": "search", "arguments": json!({"query": format!("q{}", *r)}).to_string() }
                }]
            }))
        };
        let counter = RefCell::new(0usize);
        let exec = |_: &str, _: &Value| {
            let mut c = counter.borrow_mut();
            *c += 1;
            (
                format!("hit {}", *c),
                vec![format!("doc-{}.md", *c)],
                Vec::new(),
            ) // new id every call
        };
        let out = run_agent_loop(chat, vec![], exec, nudge, rounds_to_run + 2, None).unwrap();
        assert_eq!(out, "ANSWER: done");
    }

    // --- graph-tool id extraction: regression guard for the misfire the coordinator flagged -----
    //
    // The two mock-exec tests above prove the streak MECHANISM. These two prove the id SOURCE for
    // the graph tools is correct: they drive `run_agent_loop` through the REAL `execute_tool` ->
    // `glossa_tools::exec` -> `extract_node_ids` path (no mock exec), against a real indexed corpus
    // + graph, so a genuine glossary/reach/neighbors-style reader can't be falsely steered mid
    // navigation just because graph tools used to surface no ids at all.

    use glossa::graph::store::{Edge, GraphStore, Node, Provenance};
    use glossa::index::store::DocIndex;

    fn fixture_prov() -> Provenance {
        Provenance {
            source_path: "doc.md".into(),
            range: None,
            file_sig: None,
            origin: "test".into(),
            confidence: 0.9,
            created_at: 1,
        }
    }

    /// Build a small real corpus + graph: an indexed markdown doc (so its Section nodes carry
    /// working `read` anchors via the real `index_dir` path — the same machinery a real corpus
    /// uses) plus a `hub` Entity node connected to `n` `fact_i` Entity nodes, each `MENTIONS`-
    /// grounded to one of the doc's real sections. A `neighbors` call on `hub` therefore renders a
    /// real `— read doc.md  #ord · …` anchor per fact, exactly like a genuine graph-reader's
    /// glossary/reach/neighbors traversal would.
    fn build_hub_fixture(n: usize) -> (tempfile::TempDir, DocIndex, GraphStore) {
        let dir = tempfile::tempdir().unwrap();
        let mut md = String::from("# Root\nintro\n");
        for i in 0..n {
            md.push_str(&format!("\n## Sec{i}\nbody {i}\n"));
        }
        std::fs::write(dir.path().join("doc.md"), md).unwrap();
        glossa::index::store::index_dir(dir.path(), true).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = GraphStore::open(dir.path()).unwrap();

        let sec_ids: Vec<String> = g
            .outgoing("doc.md")
            .unwrap()
            .into_iter()
            .filter(|e| e.edge_type == "CONTAINS")
            .map(|e| e.to)
            .collect();
        assert!(
            sec_ids.len() >= n,
            "expected >= {n} indexed sections, got {}: {sec_ids:?}",
            sec_ids.len()
        );

        g.put_node(&Node {
            id: "hub".into(),
            node_type: "Entity".into(),
            label: "hub".into(),
            aliases: Vec::new(),
            prov: fixture_prov(),
        })
        .unwrap();
        for (i, sec_id) in sec_ids.iter().take(n).enumerate() {
            let fact_id = format!("fact-{i}");
            g.put_node(&Node {
                id: fact_id.clone(),
                node_type: "Entity".into(),
                label: format!("Fact {i}"),
                aliases: Vec::new(),
                prov: fixture_prov(),
            })
            .unwrap();
            // Ground the fact in a real section -> gives it a working `read` anchor.
            g.put_edge(&Edge {
                from: fact_id.clone(),
                to: sec_id.clone(),
                edge_type: glossa::graph::MENTIONS.to_string(),
                prov: fixture_prov(),
            })
            .unwrap();
            // A distinct edge type per fact, so `neighbors(hub, edge_types=[REL_i])` surfaces
            // exactly ONE fact per call — mirrors a reader stepping to one node at a time.
            g.put_edge(&Edge {
                from: "hub".into(),
                to: fact_id,
                edge_type: format!("REL_{i}"),
                prov: fixture_prov(),
            })
            .unwrap();
        }
        (dir, idx, g)
    }

    #[test]
    fn loop_real_graph_navigation_to_distinct_nodes_never_falsely_steers() {
        // K+1 REAL `neighbors` calls on the actual glossa_tools dispatch, each stepping to a
        // DIFFERENT fact (distinct edge_types filter -> distinct MENTIONS-grounded target) — this is
        // what a graph reader walking glossary -> neighbors -> neighbors -> ... looks like. Before
        // the fix, graph tools surfaced zero ids, so this would misfire the steer at call K+1 even
        // though every call reached genuinely new ground.
        let n = UNPRODUCTIVE_STREAK_K + 1;
        let (dir, idx, g) = build_hub_fixture(n);
        let spec = glossa::tools::ChainSpec::default();
        let trace = TraceLog::disabled();
        let exec = |name: &str, args: &Value| {
            let (body, ids, _images) =
                execute_tool(name, args, dir.path(), &idx, Some(&g), &spec, &trace);
            (body, ids, Vec::new())
        };

        let round = RefCell::new(0usize);
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            if let Some(last_tool) = msgs.iter().rev().find(|m| m["role"] == "tool") {
                let c = last_tool["content"].as_str().unwrap_or("");
                assert!(
                    !c.to_lowercase().contains("no new information"),
                    "round {}: steer must not fire while reaching NEW graph nodes, got: {c:?}",
                    *r
                );
            }
            if *r > n {
                return Ok(json!({ "role": "assistant", "content": "ANSWER: done" }));
            }
            let i = *r - 1;
            Ok(json!({
                "role": "assistant", "content": "walking the graph",
                "tool_calls": [{
                    "id": format!("c{}", *r),
                    "function": {
                        "name": "neighbors",
                        "arguments": json!({"node": "hub", "edge_types": [format!("REL_{i}")]}).to_string()
                    }
                }]
            }))
        };
        let out = run_agent_loop(chat, vec![], exec, nudge, n + 2, None).unwrap();
        assert_eq!(out, "ANSWER: done");
    }

    #[test]
    fn loop_real_graph_navigation_stuck_on_one_node_does_trigger_streak() {
        // K+1 REAL `neighbors` calls with VARIED args (different direction/edge_types combinations,
        // so none dedup) that all resolve to the SAME single grounded fact — a graph reader stuck
        // re-probing one node from different angles. By the (K+1)th call the fed-back tool content
        // must be the steer, proving re-surfaced (not just varied) graph-tool calls DO count as
        // unproductive.
        let (dir, idx, g) = build_hub_fixture(1);
        let spec = glossa::tools::ChainSpec::default();
        let trace = TraceLog::disabled();
        let exec = |name: &str, args: &Value| {
            let (body, ids, _images) =
                execute_tool(name, args, dir.path(), &idx, Some(&g), &spec, &trace);
            (body, ids, Vec::new())
        };

        // Distinct argument objects that all resolve to the SAME single hub->fact-0 edge.
        let variants = [
            json!({"node": "hub"}),
            json!({"node": "hub", "direction": "out"}),
            json!({"node": "hub", "edge_types": ["REL_0"]}),
            json!({"node": "hub", "edge_types": ["REL_0"], "direction": "out"}),
        ];
        assert!(
            variants.len() > UNPRODUCTIVE_STREAK_K,
            "need at least K+1 distinct variants"
        );

        let round = RefCell::new(0usize);
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            if *r == UNPRODUCTIVE_STREAK_K + 2 {
                let last_tool = msgs.iter().rev().find(|m| m["role"] == "tool");
                let c = last_tool.and_then(|m| m["content"].as_str()).unwrap_or("");
                let lc = c.to_lowercase();
                assert!(
                    lc.contains("no new information") && lc.contains("change approach"),
                    "round {}: expected the unproductive-streak steer on a real re-surfaced graph \
                     node, got: {c:?}",
                    *r
                );
                return Ok(json!({ "role": "assistant", "content": "ANSWER: done" }));
            }
            let args = variants[(*r - 1) % variants.len()].clone();
            Ok(json!({
                "role": "assistant", "content": "re-probing",
                "tool_calls": [{
                    "id": format!("c{}", *r),
                    "function": { "name": "neighbors", "arguments": args.to_string() }
                }]
            }))
        };
        let out =
            run_agent_loop(chat, vec![], exec, nudge, UNPRODUCTIVE_STREAK_K + 3, None).unwrap();
        assert_eq!(out, "ANSWER: done");
    }

    // --- structural (Section/Document) node ids: regression guard for fix round 2 ----------------
    //
    // `extract_node_ids`'s first version only matched the entity-node "— read <path> #<ord>" form.
    // Section/Document endpoints render the SAME `<path>  #<ord>` anchor but BARE — no "read" word
    // (`tools::endpoint_ref`/`node_ref`) — so a reader stepping between Sections (e.g. `neighbors`
    // on a Document, or glossary/reach landing on distinct Sections) surfaced zero ids per call and
    // got falsely steered off a genuinely productive path. `build_section_hub_fixture` connects
    // `hub` DIRECTLY to Section nodes (skipping the MENTIONS-grounded entity layer the fixture above
    // uses), so `neighbors` renders exactly the bare structural form under test.

    /// Like `build_hub_fixture`, but `hub`'s edges point straight at the doc's real Section nodes —
    /// no intermediate MENTIONS-grounded entity. A `neighbors(hub, edge_types=[REL_i])` call then
    /// renders the endpoint via the BARE structural anchor (`tools::node_ref`'s Section arm), with
    /// no "— read" prefix — the exact form the first `extract_node_ids` missed.
    fn build_section_hub_fixture(n: usize) -> (tempfile::TempDir, DocIndex, GraphStore) {
        let dir = tempfile::tempdir().unwrap();
        let mut md = String::from("# Root\nintro\n");
        for i in 0..n {
            md.push_str(&format!("\n## Sec{i}\nbody {i}\n"));
        }
        std::fs::write(dir.path().join("doc.md"), md).unwrap();
        glossa::index::store::index_dir(dir.path(), true).unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let g = GraphStore::open(dir.path()).unwrap();

        let sec_ids: Vec<String> = g
            .outgoing("doc.md")
            .unwrap()
            .into_iter()
            .filter(|e| e.edge_type == "CONTAINS")
            .map(|e| e.to)
            .collect();
        assert!(
            sec_ids.len() >= n,
            "expected >= {n} indexed sections, got {}: {sec_ids:?}",
            sec_ids.len()
        );

        g.put_node(&Node {
            id: "hub".into(),
            node_type: "Entity".into(),
            label: "hub".into(),
            aliases: Vec::new(),
            prov: fixture_prov(),
        })
        .unwrap();
        for (i, sec_id) in sec_ids.iter().take(n).enumerate() {
            // Distinct edge type per section, direct hub -> Section (no entity/MENTIONS layer) —
            // so the rendered endpoint is a BARE structural anchor, not a "— read" one.
            g.put_edge(&Edge {
                from: "hub".into(),
                to: sec_id.clone(),
                edge_type: format!("REL_{i}"),
                prov: fixture_prov(),
            })
            .unwrap();
        }
        (dir, idx, g)
    }

    #[test]
    fn loop_real_structural_navigation_to_distinct_sections_never_falsely_steers() {
        // K+1 REAL `neighbors` calls, each stepping to a DIFFERENT Section endpoint rendered with
        // the BARE structural anchor (no "— read" prefix). Before fix round 2, these surfaced zero
        // ids, so this exact case — a reader walking distinct sections of a document — would
        // misfire the steer at call K+1 despite reaching genuinely new ground every time.
        let n = UNPRODUCTIVE_STREAK_K + 1;
        let (dir, idx, g) = build_section_hub_fixture(n);
        let spec = glossa::tools::ChainSpec::default();
        let trace = TraceLog::disabled();
        let exec = |name: &str, args: &Value| {
            let (body, ids, _images) =
                execute_tool(name, args, dir.path(), &idx, Some(&g), &spec, &trace);
            (body, ids, Vec::new())
        };

        let round = RefCell::new(0usize);
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            if let Some(last_tool) = msgs.iter().rev().find(|m| m["role"] == "tool") {
                let c = last_tool["content"].as_str().unwrap_or("");
                assert!(
                    !c.to_lowercase().contains("no new information"),
                    "round {}: steer must not fire while reaching NEW structural (Section) nodes, \
                     got: {c:?}",
                    *r
                );
            }
            if *r > n {
                return Ok(json!({ "role": "assistant", "content": "ANSWER: done" }));
            }
            let i = *r - 1;
            Ok(json!({
                "role": "assistant", "content": "walking the document's sections",
                "tool_calls": [{
                    "id": format!("c{}", *r),
                    "function": {
                        "name": "neighbors",
                        "arguments": json!({"node": "hub", "edge_types": [format!("REL_{i}")]}).to_string()
                    }
                }]
            }))
        };
        let out = run_agent_loop(chat, vec![], exec, nudge, n + 2, None).unwrap();
        assert_eq!(out, "ANSWER: done");
    }

    #[test]
    fn loop_real_structural_navigation_stuck_on_one_section_does_trigger_streak() {
        // Symmetric check: varied `neighbors` calls that all resolve to the SAME single Section
        // endpoint must still trip the streak by the (K+1)th call — proves the relaxed regex isn't
        // so permissive it stops detecting genuine repetition on structural nodes too.
        let (dir, idx, g) = build_section_hub_fixture(1);
        let spec = glossa::tools::ChainSpec::default();
        let trace = TraceLog::disabled();
        let exec = |name: &str, args: &Value| {
            let (body, ids, _images) =
                execute_tool(name, args, dir.path(), &idx, Some(&g), &spec, &trace);
            (body, ids, Vec::new())
        };

        let variants = [
            json!({"node": "hub"}),
            json!({"node": "hub", "direction": "out"}),
            json!({"node": "hub", "edge_types": ["REL_0"]}),
            json!({"node": "hub", "edge_types": ["REL_0"], "direction": "out"}),
        ];
        assert!(
            variants.len() > UNPRODUCTIVE_STREAK_K,
            "need at least K+1 distinct variants"
        );

        let round = RefCell::new(0usize);
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            if *r == UNPRODUCTIVE_STREAK_K + 2 {
                let last_tool = msgs.iter().rev().find(|m| m["role"] == "tool");
                let c = last_tool.and_then(|m| m["content"].as_str()).unwrap_or("");
                let lc = c.to_lowercase();
                assert!(
                    lc.contains("no new information") && lc.contains("change approach"),
                    "round {}: expected the unproductive-streak steer on a re-surfaced structural \
                     node, got: {c:?}",
                    *r
                );
                return Ok(json!({ "role": "assistant", "content": "ANSWER: done" }));
            }
            let args = variants[(*r - 1) % variants.len()].clone();
            Ok(json!({
                "role": "assistant", "content": "re-probing",
                "tool_calls": [{
                    "id": format!("c{}", *r),
                    "function": { "name": "neighbors", "arguments": args.to_string() }
                }]
            }))
        };
        let out =
            run_agent_loop(chat, vec![], exec, nudge, UNPRODUCTIVE_STREAK_K + 3, None).unwrap();
        assert_eq!(out, "ANSWER: done");
    }

    // --- vision: `--vision`-only image threading (Task: kbx build --vision) ------------------
    //
    // These drive `run_agent_loop` end to end, proving the loop actually appends the image message
    // right after the tool result when `exec` surfaces images, and appends NOTHING when it doesn't
    // — which is what keeps every non-vision caller (the reader, `kbx reason`, `kbx distil`, the
    // GEPA rollout) byte-identical to today. (The pure `vision_user_message` builder is unit-tested
    // in `backend::vision`.)

    #[test]
    fn loop_vision_on_appends_image_user_message_after_tool_result() {
        // exec surfaces one image on its (only) call; the loop must push a role:"user"
        // content-array message with a data:image/jpeg;base64,... image_url part right after the
        // role:"tool" message, before the model is asked again.
        let round = RefCell::new(0usize);
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            if *r == 1 {
                return Ok(json!({
                    "role": "assistant", "content": null,
                    "tool_calls": [{
                        "id": "c1",
                        "function": { "name": "read", "arguments": "{\"path\":\"scan.pdf\"}" }
                    }]
                }));
            }
            // Round 2: the transcript from round 1's tool call must already carry the image
            // message, positioned right after the tool result.
            let tool_pos = msgs.iter().position(|m| m["role"] == "tool");
            let img_pos = msgs.iter().position(|m| {
                m["role"] == "user"
                    && m["content"]
                        .as_array()
                        .is_some_and(|c| c.iter().any(|p| p["type"] == "image_url"))
            });
            assert!(tool_pos.is_some(), "tool result missing: {msgs:?}");
            assert_eq!(
                img_pos,
                tool_pos.map(|p| p + 1),
                "image message must come right after the tool result: {msgs:?}"
            );
            let url = msgs[img_pos.unwrap()]["content"][1]["image_url"]["url"]
                .as_str()
                .unwrap_or("");
            assert!(url.starts_with("data:image/jpeg;base64,"), "got: {url}");
            Ok(json!({ "role": "assistant", "content": "ANSWER: done" }))
        };
        let exec = |_: &str, _: &Value| {
            (
                "(scanned page text)".to_string(),
                vec!["scan.pdf".to_string()],
                vec![stub_image(9)],
            )
        };
        let out = run_agent_loop(chat, vec![], exec, nudge, 3, None).unwrap();
        assert_eq!(out, "ANSWER: done");
    }

    #[test]
    fn loop_vision_off_appends_no_image_message() {
        // Mirrors every non-vision caller (reader, reason, distil, GEPA): exec always surfaces an
        // empty image vec, so the loop must push ONLY the tool message — byte-identical to the
        // pre-vision transcript shape.
        let round = RefCell::new(0usize);
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            if *r == 1 {
                return Ok(json!({
                    "role": "assistant", "content": null,
                    "tool_calls": [{
                        "id": "c1",
                        "function": { "name": "read", "arguments": "{\"path\":\"scan.pdf\"}" }
                    }]
                }));
            }
            let has_image_msg = msgs.iter().any(|m| {
                m["role"] == "user"
                    && m["content"]
                        .as_array()
                        .is_some_and(|c| c.iter().any(|p| p["type"] == "image_url"))
            });
            assert!(
                !has_image_msg,
                "vision-off must never append an image message: {msgs:?}"
            );
            Ok(json!({ "role": "assistant", "content": "ANSWER: done" }))
        };
        let exec = |_: &str, _: &Value| {
            (
                "(scanned page text)".to_string(),
                vec!["scan.pdf".to_string()],
                Vec::new(),
            )
        };
        let out = run_agent_loop(chat, vec![], exec, nudge, 3, None).unwrap();
        assert_eq!(out, "ANSWER: done");
    }
}

#[cfg(test)]
mod schema_tests {
    use crate::backend::transport::openai::tools_schema_from_ctx;
    use serde_json::Value;

    fn tool_names(v: &Value) -> Vec<String> {
        v.as_array()
            .unwrap()
            .iter()
            .filter_map(|t| {
                t.pointer("/function/name")
                    .and_then(|n| n.as_str())
                    .map(String::from)
            })
            .collect()
    }

    /// Test-only `ToolContext` builder mirroring `transport::openai::tests::ctx`.
    fn schema_ctx(graph_on: bool, verify_available: bool) -> glossa::tools::registry::ToolContext {
        use glossa::tools::registry::{FeatureSet, Tier, ToolContext};
        ToolContext {
            profile: Tier::Reader,
            graph_on,
            verify_available,
            no_source_file: false,
            no_image: false,
            features: FeatureSet::default(),
        }
    }

    #[test]
    fn grep_is_advertised_in_both_arms() {
        // grep is ungated in the catalog, so it must appear in both graph-OFF and graph-ON.
        assert!(
            tool_names(&tools_schema_from_ctx(&schema_ctx(false, true))).contains(&"grep".into()),
            "graph-OFF must advertise grep"
        );
        assert!(
            tool_names(&tools_schema_from_ctx(&schema_ctx(true, true))).contains(&"grep".into()),
            "graph-ON must advertise grep"
        );
    }

    /// `tools_schema`/`exec` parity guard (mirrors [[mcp-tool-add-rename-eval-sites]]): when BOTH
    /// gates are open (graph-ON, verify available) the advertised schema must equal the FULL
    /// resolved Reader catalog, in catalog order — no hand-curated subset or reordering; MCP and the
    /// eval agent render from the same source of truth (`resolve_tools`), and `verify`'s exec arm
    /// (`glossa_tools::exec`) is only exercised for a name the model was actually offered.
    #[test]
    fn openai_tools_match_registry_graph_on() {
        let ctx = schema_ctx(true, true);
        let names = tool_names(&tools_schema_from_ctx(&ctx));
        let reg: Vec<_> = glossa::tools::registry::resolve_tools(&ctx)
            .iter()
            .map(|t| t.name.to_string())
            .collect();
        assert_eq!(names, reg, "graph-ON tool set must equal resolver order");
    }

    #[test]
    fn openai_tools_hide_graph_gated_when_off() {
        let names = tool_names(&tools_schema_from_ctx(&schema_ctx(false, true)));
        // related/neighbors aren't in the registry at all (withheld from the Reader profile as
        // measured clutter) — only glossary/reach/sql are graph-gated now.
        for gated in ["glossary", "reach", "sql"] {
            assert!(
                !names.contains(&gated.to_string()),
                "graph-OFF must NOT advertise graph-gated tool {gated}; got {names:?}"
            );
        }
        for ungated in ["search", "read", "grep", "glob"] {
            assert!(
                names.contains(&ungated.to_string()),
                "graph-OFF must advertise ungated tool {ungated}; got {names:?}"
            );
        }
    }

    /// Task-3 serving parity: `verify` is withheld from the advertised schema when
    /// `verify_available` is false, and present (graph tools notwithstanding) when true.
    #[test]
    fn openai_tools_hide_verify_when_unavailable() {
        let names = tool_names(&tools_schema_from_ctx(&schema_ctx(true, false)));
        assert!(
            !names.contains(&"verify".to_string()),
            "verify must be withheld when unavailable; got {names:?}"
        );
        let names_on = tool_names(&tools_schema_from_ctx(&schema_ctx(true, true)));
        assert!(
            names_on.contains(&"verify".to_string()),
            "verify must be advertised when available; got {names_on:?}"
        );
    }
}
