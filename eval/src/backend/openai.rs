use super::{prompt, AgentBackend};
use crate::dataset::Question;
use anyhow::{anyhow, bail, Context};
use glossa::trace::TraceLog;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::Path;
use std::time::Duration;

/// Generic OpenAI-compatible chat backend (LM Studio, llama.cpp server, vLLM, OpenRouter, …).
///
/// The harness itself is the agent: it advertises glossa's `search`/`read` as OpenAI function
/// tools, runs the tool-call loop, and executes the tools IN-PROCESS against the corpus in `work`.
/// (We do NOT rely on the server's own MCP/tool execution — that is GUI-only in LM Studio and
/// makes retrieval unobservable.) Every tool call is logged to `work/.glossa/traces` in the same
/// JSONL format the MCP server uses, so `run::eval_one` measures retrieval-recall unchanged.
pub struct OpenAiBackend {
    pub endpoint: String, // base url, e.g. "http://localhost:1234"
    pub model: String,
    pub api_key: Option<String>,
    pub timeout: Duration,
    /// graph-ON arm when true (opens the graph and advertises the graph tools); graph-OFF
    /// baseline when false (flat search/read only). The A/B knob for the graph-transfer eval.
    pub use_graph: bool,
}

const MAX_ROUNDS: usize = 50;

impl AgentBackend for OpenAiBackend {
    fn needs_corpus(&self) -> bool {
        true
    }

    fn answer(&self, work: &Path, q: &Question) -> anyhow::Result<String> {
        let url = format!(
            "{}/v1/chat/completions",
            self.endpoint.trim_end_matches('/')
        );
        let graph = if self.use_graph {
            glossa::graph::store::GraphStore::open(work).ok()
        } else {
            None
        };
        let tools = tools_schema(graph.is_some());
        let chat = |messages: &[Value]| {
            lmstudio_chat(
                &url,
                &self.model,
                self.api_key.as_deref(),
                &tools,
                messages,
                self.timeout,
            )
        };

        let trace = TraceLog::to_dir(work);
        // Open the index once per question; the closure reuses it (cached reader) for every
        // search/read in the agent loop instead of reopening per tool call.
        let idx = glossa::index::store::DocIndex::open_or_create(work)?;
        // Ontology-driven chain spec so glossary/related render identically to the MCP surface.
        let spec = glossa::tools::ChainSpec::from_ontology(
            &glossa::graph::ontology::Ontology::load_or_default(work),
        );
        let exec = |name: &str, args: &Value| {
            let (body, ids) = execute_tool(name, args, work, &idx, graph.as_ref(), &spec, &trace);
            // Diagnostics: KB_EVAL_DUMP_TOOLS=1 prints each tool call + a truncated body to
            // stderr, so a smoke run doubles as an episode transcript (why the reader searches).
            if std::env::var("KB_EVAL_DUMP_TOOLS").is_ok() {
                let snippet: String = body.chars().take(500).collect();
                eprintln!("\n[TOOL] {name} {args}\n[BODY] {snippet}\n[--- {} chars ---]", body.len());
            }
            (body, ids)
        };

        let messages = vec![
            json!({ "role": "system", "content": prompt::system_prompt(graph.is_some()) }),
            json!({ "role": "user", "content": prompt::user_prompt(q) }),
        ];
        // Next-best-action on a stuck (repeated) call: fan the fixated query across the
        // complementary tools instead of re-running the dead one.
        let nba = |name: &str, args: &Value| {
            crate::backend::glossa_tools::next_best_action(
                name, args, work, &idx, graph.as_ref(), &spec, &trace,
            )
        };
        let raw = run_agent_loop(chat, messages, exec, nba, MAX_ROUNDS)?;
        Ok(prompt::parse_answer(&raw))
    }
}

/// One OpenAI-compatible `/v1/chat/completions` POST to an LM Studio-style endpoint. Returns the
/// assistant `message` object (already extracted from `choices[0].message`). Samples at
/// `temperature: 0.8` — temp 0 is not a reliable greedy mode on this reasoning model/backend
/// (unstable outputs), so runs are stochastic and must be averaged over N. Shared by the eval
/// backend and the graph GEPA optimizer so both drive the same server the same way.
pub(crate) fn lmstudio_chat(
    url: &str,
    model: &str,
    api_key: Option<&str>,
    tools: &Value,
    messages: &[Value],
    timeout: Duration,
) -> anyhow::Result<Value> {
    // Sampling temperature: overridable via KB_EVAL_TEMP for noise-sensitivity runs (default 0.8).
    let temperature: f64 = std::env::var("KB_EVAL_TEMP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.8);
    let body = json!({
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": temperature
    });
    let body_str = serde_json::to_string(&body)?;
    // Diagnostics: KB_EVAL_DUMP_REQ=<path> writes the exact request body (incl. the `tools` array
    // with descriptions) sent to the endpoint, to prove what the model actually receives.
    if let Ok(p) = std::env::var("KB_EVAL_DUMP_REQ") {
        let _ = std::fs::write(&p, &body_str);
    }
    // Retry transient network drops: LM Studio occasionally closes a pooled keep-alive connection
    // that ureq then reuses → "established connection was aborted" (os error 10060). A fresh request
    // (new connection) + short backoff recovers it. Non-transport errors (HTTP status) surface at once.
    let mut attempt = 0u32;
    let resp = loop {
        let mut req = ureq::post(url)
            .timeout(timeout)
            .set("Content-Type", "application/json");
        if let Some(key) = api_key {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        match req.send_string(&body_str) {
            Ok(r) => break r,
            Err(ureq::Error::Transport(t)) if attempt < 3 => {
                attempt += 1;
                std::thread::sleep(Duration::from_millis(400 * attempt as u64));
                continue;
            }
            Err(e) => return Err(anyhow!("endpoint request failed: {e}")),
        }
    };
    let text = resp.into_string().context("read endpoint response")?;
    let v: Value = serde_json::from_str(&text).context("parse endpoint json")?;
    if let Some(err) = v.get("error") {
        bail!("endpoint returned error: {err}");
    }
    let msg = v["choices"][0]["message"].clone();
    if msg.is_null() {
        bail!("endpoint response missing choices[0].message: {text}");
    }
    Ok(msg)
}

/// OpenAI function-tool schema for glossa's search/read.
pub(crate) fn tools_schema(graph_on: bool) -> Value {
    let search = json!({
        "type": "function",
        "function": {
            "name": "search",
            "description": "Full-text BM25 search over the knowledge base. Pass short KEYWORDS (morphology-aware), not a sentence. Returns ranked results as `path:location: snippet  [score]`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "keywords to search for" },
                    "limit": { "type": "integer", "description": "max results (default 10)" }
                },
                "required": ["query"]
            }
        }
    });
    let read = json!({
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a chunk's full text. `path` is a document path from a search/glossary result; `n` is the chunk number shown as `[#n]`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "document path, exactly as shown in a search result" },
                    "n": { "type": "integer", "description": "chunk number to read, exactly as shown in `[#n]` (page number for PDFs)" }
                },
                "required": ["path", "n"]
            }
        }
    });
    let grep = json!({
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Exact / regex line search over the raw corpus text — the precise complement to `search`. Where `search` ranks fuzzily by topic, `grep` returns EVERY line that literally matches, as `path:#n: line`. Reach for it the moment you know the exact string you want (a name, title, phrase): it finds the specific entity that BM25 buries under broad topic matches, and the `#n` doubles as the `read` key. Keep the pattern SIMPLE — one or two distinctive words. A long regex with `.*` or `|` alternations usually matches nothing; when you mean your text literally, set `fixed:true`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "text or regex to match literally against corpus lines" },
                    "path": { "type": "string", "description": "optional: scope the search to one document path (as shown in a result)" },
                    "ignore_case": { "type": "boolean", "description": "case-insensitive match (default false)" },
                    "fixed": { "type": "boolean", "description": "treat pattern as a literal string, not a regex (default false)" },
                    "word": { "type": "boolean", "description": "match whole words only (default false)" }
                },
                "required": ["pattern"]
            }
        }
    });
    if !graph_on {
        return Value::Array(vec![search, grep, read]);
    }
    // graph-ON arm: lead with the reasoning-graph tools so the model FOLLOWS the pre-built
    // multi-hop chain instead of re-deriving it with many flat searches. Ordering matters — a weak
    // model reaches for the first-listed tool — so `glossary` (the entry point) comes before
    // `search`, which stays as a fallback.
    let glossary = json!({
        "type": "function",
        "function": {
            "name": "glossary",
            "description": "Look a named entity up in the pre-built reasoning graph: returns its grounded node plus its typed relations (part-of, created-by, family-of, located-in, member-of, …), each anchored to a source chunk. One call surfaces the chain that would otherwise take several searches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "the term or entity to look up" },
                    "query": { "type": "string", "description": "the full question you are trying to answer — this ranks the composed facts by what you actually need, so pass it whenever you have one" }
                },
                "required": ["name"]
            }
        }
    });
    let neighbors = json!({
        "type": "function",
        "function": {
            "name": "neighbors",
            "description": "Follow ONE hop: the typed 1-hop edges from a graph node — the direct relations (e.g. created-by, part-of, family-of) stored in the graph. Use it to step from an entity to the bridge entity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node": { "type": "string", "description": "graph node id, exactly as shown on a `glossary` line (e.g. `person:clancy-brown`) — the id, NOT the label" },
                    "path": { "type": "string", "description": "document path (use with `n` instead of `node`)" },
                    "n": { "type": "integer", "description": "chunk number (use with `path`)" },
                    "direction": { "type": "string", "description": "out | in | both (default both)" }
                }
            }
        }
    });
    let reach = json!({
        "type": "function",
        "function": {
            "name": "reach",
            "description": "The reasoning-chain tool — one call, two uses. ANSWER a relational multi-hop: give `from` and `relation`, omit `to`, and it discovers every node `from` reaches along that relation, crossing document boundaries on shared mentions when the in-graph chain dead-ends — use this instead of inferring the answer from prose. VERIFY a candidate you already formed: add `to` and it returns the grounded connecting chain, or nothing if the candidate doesn't actually connect. `relation` is fuzzy-matched to the graph's real edge types (omit for any connection at all). Subsumes a plain 'are these two connected' lookup: give both `from` and `to`, omit `relation`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from": { "type": "string", "description": "start: graph node id from a `glossary` line (e.g. `person:clancy-brown`)" },
                    "relation": { "type": "string", "description": "relation to follow, fuzzy-matched to the graph's real edge types (e.g. \"located\"); omit for any connection" },
                    "to": { "type": "string", "description": "end: graph node id from a `glossary` line — VERIFY mode; omit for DISCOVERY (find what `from` reaches)" }
                },
                "required": ["from"]
            }
        }
    });
    let related = json!({
        "type": "function",
        "function": {
            "name": "related",
            "description": "Nodes related to a graph node — similar / same-community / transitive-closure links. Use to widen from a hit to its neighborhood when a direct hop isn't enough.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node": { "type": "string", "description": "graph node id from a `glossary` line (e.g. `person:clancy-brown`)" },
                    "path": { "type": "string", "description": "document path (use with `n` instead of `node`)" }
                }
            }
        }
    });
    let graph_query = json!({
        "type": "function",
        "function": {
            "name": "graph_query",
            "description": "Read-only SQL SELECT over the reasoning graph. Reach for it when the answer is a SPECIFIC related entity, a ranking, or an extreme ('which place/which year', 'the earliest/largest/first'): let the query carry the judgment instead of inferring from prose. Match the source entity and the relation, order or filter, take the target — it hands back the exact related entity AT THE EDGE'S LEVEL, the immediate one the question points at, not a broader parent. A fuzzy relation name is fine — `edge_type LIKE '%…%'` — the engine resolves it to the graph's real relation names. Main view for traversal (a join is one hop): edges_labeled(src_label, edge_type, dst_label, efrom, eto). Call it once with an EMPTY query first to see the schema and this graph's real edge_type/node_type vocabulary. Also: nodes(id, node_type, label), edges(efrom, edge_type, eto), node_validity(node_id, valid_from, ...).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": { "type": "string", "description": "read-only SQL SELECT over the reasoning graph; empty (or omitted) returns the schema with the real edge_type/node_type vocabulary instead of running a query" }
                }
            }
        }
    });
    Value::Array(vec![glossary, neighbors, reach, related, graph_query, search, grep, read])
}

/// Unproductive-streak threshold: this many consecutive REAL (non-deduped) tool calls in a row that
/// each surface zero new identifiers trips the steer. Named so the TDD tests and the loop agree on
/// one number instead of a magic literal in two places.
const UNPRODUCTIVE_STREAK_K: usize = 3;

/// Drive a tool-calling chat to a final textual answer.
///
/// `chat(messages)` returns the assistant `message` object (already extracted from
/// `choices[0].message`). When it carries `tool_calls`, each is dispatched through `exec(name,
/// args)` and the result fed back as a `role:"tool"` message, then the model is queried again —
/// up to `max_rounds`. The first message without tool calls yields the answer.
///
/// Two independent stuck detectors sit on top of `exec`:
/// - **Identical-repeat dedup**: the previous (tool, args) actually executed. When the model
///   re-issues the SAME call, it isn't re-run (identical result) — `on_repeat` (the next-best-action)
///   fires instead. This takes priority and doesn't touch the streak below (a dedup hit didn't
///   execute, so it can't be "unproductive").
/// - **Unproductive streak**: the model issues many DIFFERENT calls (varied tool/args — so they all
///   really execute) that each surface no NEW identifier — a search-flood that never progresses.
///   `exec` returns `(body, ids)`; `ids` are what a session-aware MCP server would track per call
///   (search hit locations, a read's path, …). Any id not already in `seen` resets the streak;
///   otherwise it grows. At `UNPRODUCTIVE_STREAK_K` the fed-back tool content becomes the steer
///   (`glossa_tools::unproductive_steer`) instead of the (already-seen) body, and the counter resets
///   so it fires once per streak, not on every call past the threshold.
pub(crate) fn run_agent_loop<C, F, N>(
    mut chat: C,
    mut messages: Vec<Value>,
    mut exec: F,
    mut on_repeat: N,
    max_rounds: usize,
) -> anyhow::Result<String>
where
    C: FnMut(&[Value]) -> anyhow::Result<Value>,
    F: FnMut(&str, &Value) -> (String, Vec<String>),
    N: FnMut(&str, &Value) -> String,
{
    // Stuck-detection substrate: the previous (tool, args) actually executed. When the model
    // re-issues the SAME call, we don't re-run it (identical result) — we hand off to `on_repeat`,
    // the next-best-action. Default callers pass `repeat_nudge`; the reader path passes a fan-out.
    let mut last_key: Option<String> = None;
    // Novelty tracking for the unproductive-streak detector (see the doc comment above).
    let mut seen: HashSet<String> = HashSet::new();
    let mut unproductive: usize = 0;
    for _ in 0..max_rounds {
        let msg = chat(&messages)?;
        let calls: Vec<Value> = msg
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        if calls.is_empty() {
            return Ok(content_of(&msg));
        }
        messages.push(msg.clone()); // echo the assistant turn that requested the tools
        for call in &calls {
            let id = call.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let name = call
                .pointer("/function/name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let args = parse_tool_args(call);
            let key = format!("{name}\u{1}{}", serde_json::to_string(&args).unwrap_or_default());
            let result = if last_key.as_deref() == Some(key.as_str()) {
                on_repeat(name, &args)
            } else {
                let (body, ids) = exec(name, &args);
                last_key = Some(key);
                let has_new = ids.into_iter().fold(false, |acc, i| seen.insert(i) || acc);
                if has_new {
                    unproductive = 0;
                    body
                } else {
                    unproductive += 1;
                    if unproductive >= UNPRODUCTIVE_STREAK_K {
                        unproductive = 0;
                        crate::backend::glossa_tools::unproductive_steer(name)
                    } else {
                        body
                    }
                }
            };
            messages.push(json!({ "role": "tool", "tool_call_id": id, "content": result }));
        }
    }
    // Out of rounds: nudge for a final answer (the model often keeps requesting tools otherwise)
    // and take whatever text it gives.
    messages.push(json!({
        "role": "user",
        "content": "Stop searching. Give your final answer now on a single line beginning with `ANSWER:`."
    }));
    let msg = chat(&messages)?;
    Ok(content_of(&msg))
}

fn content_of(msg: &Value) -> String {
    msg.get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string()
}

/// Tool-call `function.arguments` is a JSON-encoded string per the OpenAI spec, but some servers
/// (incl. some LM Studio builds) return it as an already-parsed object. Accept both.
fn parse_tool_args(call: &Value) -> Value {
    match call.pointer("/function/arguments") {
        Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
        Some(v @ Value::Object(_)) => v.clone(),
        _ => json!({}),
    }
}

/// Execute one glossa tool in-process against the corpus in `work`, logging it to the trace
/// (same shape as the MCP server: search → array of {path,location,score}; read → {path}).
///
/// Returns `(body, ids)` — `ids` are the identifiers this call surfaced (what a session-aware MCP
/// server would track for novelty), from `glossa_tools::exec`'s second return value: `search`'s hit
/// locations, and the graph tools' (glossary/related/neighbors/reach/graph_query) `path#ord`
/// read-anchor ids scraped from their rendered bodies. `read` itself surfaces no ids there, so it's
/// special-cased here to the `path` argument instead. `run_agent_loop` uses these to detect an
/// unproductive streak — many varied calls (including varied graph navigation) that surface
/// nothing new — without falsely tripping on a reader that IS making real graph progress.
fn execute_tool(
    name: &str,
    args: &Value,
    root: &Path,
    idx: &glossa::index::store::DocIndex,
    graph: Option<&glossa::graph::store::GraphStore>,
    spec: &glossa::tools::ChainSpec,
    trace: &TraceLog,
) -> (String, Vec<String>) {
    let (body, ids, _images) =
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
    (body, ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Default `on_repeat` for tests that don't exercise NBA: a static nudge.
    fn nudge(name: &str, _args: &Value) -> String {
        format!("(dup {name}) you already called this — try a different tool or change the query")
    }

    #[test]
    fn parse_tool_args_handles_string_and_object() {
        let s = json!({ "function": { "arguments": "{\"query\":\"abc\"}" } });
        assert_eq!(parse_tool_args(&s)["query"], "abc");
        let o = json!({ "function": { "arguments": { "query": "abc" } } });
        assert_eq!(parse_tool_args(&o)["query"], "abc");
        let bad = json!({ "function": { "arguments": "not json" } });
        assert_eq!(parse_tool_args(&bad), json!({}));
    }

    #[test]
    fn loop_returns_direct_answer_when_no_tool_calls() {
        let chat = |_: &[Value]| Ok(json!({ "role": "assistant", "content": "ANSWER: Bob" }));
        let exec = |_: &str, _: &Value| (String::new(), Vec::new());
        let out = run_agent_loop(chat, vec![], exec, nudge, 4).unwrap();
        assert_eq!(out, "ANSWER: Bob");
    }

    #[test]
    fn loop_dispatches_tool_then_answers() {
        let round = RefCell::new(0usize);
        let seen = RefCell::new(Vec::<(String, String)>::new());
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            if *r == 1 {
                // first turn requests a search
                Ok(json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "function": { "name": "search", "arguments": "{\"query\":\"corliss\"}" }
                    }]
                }))
            } else {
                // by now the tool result must be in the transcript
                let has_tool = msgs
                    .iter()
                    .any(|m| m["role"] == "tool" && m["tool_call_id"] == "call_1");
                assert!(has_tool, "tool result not fed back: {msgs:?}");
                Ok(json!({ "role": "assistant", "content": "ANSWER: Chief of Protocol" }))
            }
        };
        let exec = |name: &str, args: &Value| {
            seen.borrow_mut().push((
                name.to_string(),
                args["query"].as_str().unwrap_or("").to_string(),
            ));
            (
                "Meet_Corliss_Archer.md:p.1: ...  [9.0]".to_string(),
                vec!["Meet_Corliss_Archer.md:p.1".to_string()],
            )
        };
        let out =
            run_agent_loop(chat, vec![json!({"role":"user","content":"q"})], exec, nudge, 4).unwrap();
        assert_eq!(out, "ANSWER: Chief of Protocol");
        assert_eq!(
            seen.borrow().as_slice(),
            &[("search".to_string(), "corliss".to_string())]
        );
    }

    #[test]
    fn loop_dedupes_consecutive_identical_tool_calls() {
        // The model thrashes: same tool, same args, every round. The loop must execute it once,
        // then feed back a nudge (not another live result) so the model is pushed to switch.
        let execs = RefCell::new(0usize);
        let round = RefCell::new(0usize);
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            // Round 1 executes; round 2's identical call is deduped and its nudge lands in the
            // transcript for round 3's chat onward (round 2 still sees round 1's live result).
            if *r >= 3 {
                let last_tool = msgs.iter().rev().find(|m| m["role"] == "tool");
                let c = last_tool.and_then(|m| m["content"].as_str()).unwrap_or("");
                let lc = c.to_lowercase();
                assert!(
                    lc.contains("different tool") || lc.contains("already"),
                    "round {} expected a dedup nudge, got: {c:?}",
                    *r
                );
            }
            Ok(json!({
                "role": "assistant", "content": "looping",
                "tool_calls": [{ "id": "c", "function": { "name": "glossary", "arguments": "{\"name\":\"X\"}" } }]
            }))
        };
        let exec = |_: &str, _: &Value| {
            *execs.borrow_mut() += 1;
            ("hit".to_string(), vec!["hit-id".to_string()])
        };
        let out = run_agent_loop(chat, vec![], exec, nudge, 5).unwrap();
        assert_eq!(out, "looping");
        assert_eq!(*execs.borrow(), 1, "identical consecutive calls must execute only once");
    }

    #[test]
    fn loop_reexecutes_when_args_differ() {
        // A different tool OR different args is NOT a dedup — it must run.
        let execs = RefCell::new(0usize);
        let round = RefCell::new(0usize);
        let chat = |_: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            let name = if *r % 2 == 1 { "A" } else { "B" };
            Ok(json!({
                "role": "assistant", "content": "alternating",
                "tool_calls": [{ "id": "c", "function": { "name": name, "arguments": "{\"name\":\"X\"}" } }]
            }))
        };
        // Novelty tracking isn't under test here; empty ids keep it a no-op for the streak detector.
        let exec = |_: &str, _: &Value| {
            *execs.borrow_mut() += 1;
            ("hit".to_string(), Vec::new())
        };
        let _ = run_agent_loop(chat, vec![], exec, nudge, 4).unwrap();
        assert_eq!(*execs.borrow(), 4, "alternating tools must each execute");
    }

    #[test]
    fn loop_stops_at_max_rounds() {
        // chat always asks for a tool; loop must terminate (max_rounds + 1 final call) not hang.
        let calls = RefCell::new(0usize);
        let chat = |_: &[Value]| {
            *calls.borrow_mut() += 1;
            Ok(json!({
                "role": "assistant", "content": "giving up",
                "tool_calls": [{ "id": "c", "function": { "name": "search", "arguments": "{\"query\":\"x\"}" } }]
            }))
        };
        let exec = |_: &str, _: &Value| ("hit".to_string(), Vec::new());
        let out = run_agent_loop(chat, vec![], exec, nudge, 3).unwrap();
        assert_eq!(out, "giving up");
        assert_eq!(*calls.borrow(), 4); // 3 rounds + 1 final
    }

    #[test]
    fn loop_unproductive_streak_feeds_steer_after_k_plus_one_calls() {
        // K+1 tool calls with VARIED args (so none of them dedup — each really executes), but exec
        // keeps surfacing the SAME already-seen id: an over-search spiral (many different probes,
        // nothing new). By the (K+1)th call, the fed-back tool content must be the steer.
        let round = RefCell::new(0usize);
        let execs = RefCell::new(0usize);
        let chat = |msgs: &[Value]| {
            let mut r = round.borrow_mut();
            *r += 1;
            if *r == UNPRODUCTIVE_STREAK_K + 2 {
                // By now K+1 tool calls have executed; the most recent tool result must be the steer.
                let last_tool = msgs.iter().rev().find(|m| m["role"] == "tool");
                let c = last_tool.and_then(|m| m["content"].as_str()).unwrap_or("");
                let lc = c.to_lowercase();
                assert!(
                    lc.contains("no new information") && lc.contains("change approach"),
                    "round {} expected the unproductive-streak steer, got: {c:?}",
                    *r
                );
                return Ok(json!({ "role": "assistant", "content": "ANSWER: done" }));
            }
            let query = format!("query-{}", *r); // distinct args every round -> never dedups
            Ok(json!({
                "role": "assistant", "content": "searching",
                "tool_calls": [{
                    "id": format!("c{}", *r),
                    "function": { "name": "search", "arguments": json!({"query": query}).to_string() }
                }]
            }))
        };
        let exec = |_: &str, _: &Value| {
            *execs.borrow_mut() += 1;
            // Always the same id, regardless of the (varied) query -> never novel after the first.
            ("same old snippet".to_string(), vec!["doc.md:p.1".to_string()])
        };
        let out = run_agent_loop(chat, vec![], exec, nudge, UNPRODUCTIVE_STREAK_K + 3).unwrap();
        assert_eq!(out, "ANSWER: done");
        assert_eq!(
            *execs.borrow(),
            UNPRODUCTIVE_STREAK_K + 1,
            "each varied call must actually execute (not deduped)"
        );
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
            (format!("hit {}", *c), vec![format!("doc-{}.md", *c)]) // new id every call
        };
        let out = run_agent_loop(chat, vec![], exec, nudge, rounds_to_run + 2).unwrap();
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
            execute_tool(name, args, dir.path(), &idx, Some(&g), &spec, &trace)
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
        let out = run_agent_loop(chat, vec![], exec, nudge, n + 2).unwrap();
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
            execute_tool(name, args, dir.path(), &idx, Some(&g), &spec, &trace)
        };

        // Distinct argument objects that all resolve to the SAME single hub->fact-0 edge.
        let variants = [
            json!({"node": "hub"}),
            json!({"node": "hub", "direction": "out"}),
            json!({"node": "hub", "edge_types": ["REL_0"]}),
            json!({"node": "hub", "edge_types": ["REL_0"], "direction": "out"}),
        ];
        assert!(
            variants.len() >= UNPRODUCTIVE_STREAK_K + 1,
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
        let out = run_agent_loop(chat, vec![], exec, nudge, UNPRODUCTIVE_STREAK_K + 3).unwrap();
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
            execute_tool(name, args, dir.path(), &idx, Some(&g), &spec, &trace)
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
        let out = run_agent_loop(chat, vec![], exec, nudge, n + 2).unwrap();
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
            execute_tool(name, args, dir.path(), &idx, Some(&g), &spec, &trace)
        };

        let variants = [
            json!({"node": "hub"}),
            json!({"node": "hub", "direction": "out"}),
            json!({"node": "hub", "edge_types": ["REL_0"]}),
            json!({"node": "hub", "edge_types": ["REL_0"], "direction": "out"}),
        ];
        assert!(
            variants.len() >= UNPRODUCTIVE_STREAK_K + 1,
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
        let out = run_agent_loop(chat, vec![], exec, nudge, UNPRODUCTIVE_STREAK_K + 3).unwrap();
        assert_eq!(out, "ANSWER: done");
    }
}

#[cfg(test)]
mod schema_tests {
    use super::*;

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

    #[test]
    fn graph_tools_lead_in_graph_on() {
        // Ordering nudges tool choice: a weak model reaches for the first-listed tool. In graph-ON
        // the graph entry point (glossary) must precede the flat `search`, so the model probes the
        // pre-built chain before falling back to keyword search.
        let names = tool_names(&tools_schema(true));
        let gi = names.iter().position(|n| n == "glossary").expect("glossary present");
        let si = names.iter().position(|n| n == "search").expect("search present");
        assert!(gi < si, "graph tools must lead search in graph-ON; got {names:?}");
    }

    #[test]
    fn grep_is_advertised_in_both_arms() {
        // grep was mirrored into the tensorzero surface but never into this backend's hardcoded
        // schema, so the model was never offered the precise line-lookup that complements fuzzy
        // `search`. It must appear in both graph-OFF and graph-ON.
        assert!(tool_names(&tools_schema(false)).contains(&"grep".into()), "graph-OFF must advertise grep");
        assert!(tool_names(&tools_schema(true)).contains(&"grep".into()), "graph-ON must advertise grep");
    }

    #[test]
    fn tools_schema_gates_graph_tools_on_graph_on() {
        let off = tool_names(&tools_schema(false));
        assert!(off.contains(&"search".into()) && off.contains(&"read".into()));
        assert!(
            !off.contains(&"glossary".into()),
            "graph-OFF must NOT advertise graph tools"
        );
        let on = tool_names(&tools_schema(true));
        for t in ["glossary", "related", "neighbors", "reach", "graph_query"] {
            assert!(on.contains(&t.to_string()), "graph-ON must advertise {t}");
        }
        assert!(
            !on.contains(&"path".into()),
            "reach replaces path — graph-ON must NOT advertise the removed path tool"
        );
        assert!(
            !off.contains(&"graph_query".into()),
            "graph-OFF must NOT advertise graph_query"
        );
        assert!(
            !off.contains(&"reach".into()),
            "graph-OFF must NOT advertise reach"
        );
        assert!(
            on.contains(&"search".into()) && on.contains(&"read".into()),
            "flat tools remain in graph-ON"
        );
    }
}
