use serde_json::{json, Value};
use std::collections::HashSet;

/// Per-turn wall-clock profiling, silent unless `KB_PROF` is set. Used to separate inference vs
/// tool latency (e.g. it surfaced that proxied inference dominated, then large-PDF `read`).
macro_rules! prof {
    ($($a:tt)*) => { if std::env::var_os("KB_PROF").is_some() { eprintln!($($a)*); } };
}

pub struct EpisodeOutcome {
    pub answer: String,
    pub episode_id: Option<String>,
    pub surfaced_titles: Vec<String>,
}

/// A single TensorZero `/inference` turn result.
pub struct TzTurn {
    pub content: Vec<Value>, // TZ content blocks: {type:"tool_call"|"text"|"thought", ...}
    pub episode_id: String,
}

/// How an episode terminates and whether read-only tool calls are de-duplicated.
#[derive(Clone, Copy, Default)]
pub struct EpisodePolicy {
    /// Terminate ONLY on an explicit `done` tool call (enrich): a text-only turn is no longer an
    /// ending — the model is nudged to act or to call `done`, so "narrate-then-stop" can't end the
    /// episode prematurely. When false (answer), a text-only turn IS the final answer, as before.
    pub stop_on_done: bool,
    /// Short-circuit a read-only tool call whose (name, args) already ran this episode, returning a
    /// hint instead of re-executing — breaks the re-read-the-same-page thrash loop.
    pub dedup_readonly: bool,
}

impl EpisodePolicy {
    /// Answer mode: a text-only turn ends the episode; no dedup (keeps eval ≡ prod for answering).
    pub fn answer() -> Self { Self::default() }
    /// Enrich mode: an explicit `done` ends the episode; read-only calls are de-duplicated.
    pub fn enrich() -> Self { Self { stop_on_done: true, dedup_readonly: true } }
}

/// What a tool call reads or changes — drives dedup invalidation. An identical call is "stale" (must
/// re-run) once the state it depends on changed: a graph read after any graph mutation, a corpus read
/// after an index rebuild. Corpus reads over the static KB never go stale, so they dedup once forever.
#[derive(PartialEq, Clone, Copy)]
enum ToolKind {
    Corpus,       // search/read/grep/glob — static KB; dedup once, never invalidate
    GraphRead,    // glossary/neighbors/resolve — invalidated by any graph mutation
    GraphMutate,  // graph_upsert/delete/update/generalize — invalidates graph reads + graph mutates
    CorpusMutate, // index/reindex/purge — invalidates everything
    Control,      // done — never deduped
}

fn tool_kind(name: &str) -> ToolKind {
    match name {
        "done" => ToolKind::Control,
        "graph_upsert" | "graph_delete" | "graph_update" | "graph_generalize" => ToolKind::GraphMutate,
        "index" | "reindex" | "purge" => ToolKind::CorpusMutate,
        "glossary" | "neighbors" | "resolve" | "graph_stats" => ToolKind::GraphRead,
        _ => ToolKind::Corpus,
    }
}

/// Drive a TensorZero episode to a final outcome.
///
/// `chat(messages, episode_id)` performs one `/inference` call (episode_id is None on the first turn,
/// then the id returned by the first turn). `tool_call` blocks are executed via `exec(name, args)`
/// and fed back as `tool_result` blocks. Termination + dedup follow `policy` (see `EpisodePolicy`).
pub fn run_episode<C, X>(
    mut chat: C,
    user_question: &str,
    exec: X,
    max_rounds: usize,
    policy: EpisodePolicy,
) -> anyhow::Result<EpisodeOutcome>
where
    C: FnMut(&[Value], Option<&str>) -> anyhow::Result<TzTurn>,
    X: Fn(&str, &Value) -> (String, Vec<String>, Vec<glossa::read::DocImage>) + Sync,
{
    let mut messages: Vec<Value> = vec![json!({ "role": "user", "content": user_question })];
    let mut episode_id: Option<String> = None;
    let mut surfaced_titles: Vec<String> = Vec::new();
    // (name, args) of read-only calls already executed this episode — backs the dedup guard.
    let mut seen: HashSet<(String, String)> = HashSet::new();

    for _ in 0..max_rounds {
        let turn = chat(&messages, episode_id.as_deref())?;
        if !turn.episode_id.is_empty() {
            episode_id = Some(turn.episode_id.clone());
        }

        let tool_calls: Vec<&Value> = turn
            .content
            .iter()
            .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_call"))
            .collect();

        if tool_calls.is_empty() {
            let answer: String = turn
                .content
                .iter()
                .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("");
            if !policy.stop_on_done {
                return Ok(EpisodeOutcome { answer, episode_id, surfaced_titles });
            }
            // Narrate-then-stop: the model described its next action but emitted no tool call. Don't
            // end the episode — record what it said and nudge it to act or to signal completion.
            messages.push(json!({ "role": "assistant", "content": [{ "type": "text", "text": answer }] }));
            messages.push(json!({ "role": "user", "content": [{ "type": "text", "text":
                "You ended a turn without calling a tool. If the reasoning graph for this case is complete (or already present), call `done`. Otherwise issue the tool call you described." }] }));
            continue;
        }

        // Explicit completion signal (enrich): the model called `done` → the episode is finished.
        if policy.stop_on_done {
            if let Some(done) = tool_calls.iter().find(|c| c.get("name").and_then(|n| n.as_str()) == Some("done")) {
                let note = done
                    .get("arguments")
                    .and_then(|a| a.get("note"))
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string();
                return Ok(EpisodeOutcome { answer: note, episode_id, surfaced_titles });
            }
        }

        // Build owned (id, name, args) triples with the raw_arguments fallback applied
        // BEFORE spawning, so each thread owns its data.
        // TZ returns `arguments: null` (with the raw string under `raw_arguments`)
        // when the model emitted unparseable JSON args. Echoing `null` back is
        // rejected by the input schema ("did not match any variant of
        // ToolCallWrapper"), so fall back to `raw_arguments` (a string, which TZ
        // accepts) and finally to `{}`.
        let calls: Vec<(String, String, Value)> = tool_calls
            .iter()
            .map(|call| {
                let id = call.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let name = call.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let args = call
                    .get("arguments")
                    .filter(|v| !v.is_null())
                    .cloned()
                    .or_else(|| call.get("raw_arguments").cloned())
                    .unwrap_or_else(|| json!({}));
                (id, name, args)
            })
            .collect();

        // ONE assistant message whose content array contains ALL tool_call blocks
        // for this round (correctly represents parallel tool calls as a single turn).
        let tool_call_blocks: Vec<Value> = calls
            .iter()
            .map(|(id, name, args)| {
                json!({ "type": "tool_call", "id": id, "name": name, "arguments": args })
            })
            .collect();
        messages.push(json!({ "role": "assistant", "content": tool_call_blocks }));

        // Dedup: skip a call whose (name,args) is still "current" in `seen`; `done`/control is never
        // deduped. Then invalidate `seen` for the next turn — a graph mutation makes prior graph
        // reads + graph mutates stale (corpus reads survive); an index rebuild makes everything stale.
        let run_flags: Vec<bool> = calls
            .iter()
            .map(|(_, name, args)| {
                !(policy.dedup_readonly
                    && tool_kind(name) != ToolKind::Control
                    && seen.contains(&(name.clone(), args.to_string())))
            })
            .collect();
        if policy.dedup_readonly {
            for ((_, name, args), &ran) in calls.iter().zip(&run_flags) {
                if !ran {
                    continue; // deduped → didn't execute → state unchanged
                }
                match tool_kind(name) {
                    ToolKind::CorpusMutate => seen.clear(),
                    ToolKind::GraphMutate => seen.retain(|(n, _)| tool_kind(n) == ToolKind::Corpus),
                    _ => {}
                }
                seen.insert((name.clone(), args.to_string()));
            }
        }

        // Execute the to-run calls concurrently; substitute a hint for deduped ones. Order kept.
        let results: Vec<(String, String, String, Vec<String>, Vec<glossa::read::DocImage>)> =
            std::thread::scope(|s| {
                let handles: Vec<Option<_>> = calls
                    .iter()
                    .zip(&run_flags)
                    .map(|((id, name, args), &run)| {
                        if run {
                            Some(s.spawn(|| {
                                let (result, titles, images) = exec(name, args);
                                (id.clone(), name.clone(), result, titles, images)
                            }))
                        } else {
                            None
                        }
                    })
                    .collect();
                handles
                    .into_iter()
                    .zip(calls.iter())
                    .map(|(h, (id, name, _args))| match h {
                        Some(h) => h.join().unwrap_or_else(|_| {
                            (id.clone(), name.clone(), "tool panicked".to_string(), Vec::new(), Vec::new())
                        }),
                        None => (
                            id.clone(),
                            name.clone(),
                            format!("(skipped) You already called `{name}` with these exact arguments and nothing has changed since. Use what you have, do something different, or call `done`."),
                            Vec::new(),
                            Vec::new(),
                        ),
                    })
                    .collect()
            });

        // Push one tool_result user message per call (in original call order),
        // followed by any image blocks. Result MUST be a string.
        for (id, name, result, titles, images) in results {
            surfaced_titles.extend(titles);
            messages.push(json!({
                "role": "user",
                "content": [{ "type": "tool_result", "id": id, "name": name, "result": result }]
            }));
            if let Some(img_msg) = image_user_message(&images) {
                messages.push(img_msg);
            }
        }
    }
    // Out of rounds: best-effort empty answer (the report still scores it).
    Ok(EpisodeOutcome { answer: String::new(), episode_id, surfaced_titles })
}

/// Build a user message carrying the read's images as TZ image content blocks (vision input),
/// or None when there are no images. Uses the content-block shape verified by the Task-1 spike.
fn image_user_message(images: &[glossa::read::DocImage]) -> Option<Value> {
    if images.is_empty() {
        return None;
    }
    use base64::Engine as _;
    let mut content = vec![json!({"type": "text", "text": "(images from the chunk you just read)"})];
    for img in images {
        let b64 = base64::engine::general_purpose::STANDARD.encode(&img.bytes);
        content.push(json!({"type": "image", "mime_type": img.mime, "data": b64}));
    }
    Some(json!({"role": "user", "content": content}))
}

use super::{prompt, AgentBackend};
use crate::dataset::Question;
use anyhow::{anyhow, bail, Context};
use glossa::trace::TraceLog;
use std::path::Path;
use std::time::Duration;

// High cap: effectively "no limit" for real episodes, but still bounded to prevent a runaway
// tool-call loop from spinning forever.
const MAX_ROUNDS: usize = 50;

/// Drives Qwen through the TensorZero gateway (function `answer_hotpot`), executing glossa tools
/// in-process, then posts em/f1/retrieved feedback for the episode. The prompt lives in TZ config.
pub struct TensorZeroBackend {
    pub endpoint: String, // e.g. http://localhost:3000
    pub function: String, // e.g. answer_hotpot
    pub timeout: Duration,
    pub tags: serde_json::Value, // flat {key: value} object attached to /inference + /feedback ({} = none)
    pub judge_endpoint: Option<String>, // OpenAI-compatible endpoint for the LLM-judge (None = disabled)
    pub judge_model: String,
    pub judge_api_key: Option<String>,
}

impl AgentBackend for TensorZeroBackend {
    fn needs_corpus(&self) -> bool {
        true
    }

    fn rebuild_corpus_each_question(&self) -> bool {
        false
    }

    fn answer(&self, work: &Path, q: &Question) -> anyhow::Result<String> {
        let url = format!("{}/inference", crate::tz::gateway_base(&self.endpoint));
        let function = self.function.clone();
        let timeout = self.timeout;
        let mut tag_map = self
            .tags
            .as_object()
            .cloned()
            .unwrap_or_default();
        tag_map.insert("case_id".to_string(), json!(q.id));
        let tags = Value::Object(tag_map);
        // Client-generated episode_id, back-dated 30s so its UUIDv7 timestamp is always in the PAST
        // relative to the gateway's clock — immune to Docker/WSL host↔container clock skew. The same id
        // is sent on every turn, so all inferences group into one episode (telemetry + feedback).
        let eid = crate::tz::backdated_episode_id(30);
        let chat = |messages: &[Value], _episode_id: Option<&str>| -> anyhow::Result<TzTurn> {
            let t0 = std::time::Instant::now();
            let mut body = json!({ "function_name": function, "input": { "messages": messages }, "episode_id": eid });
            if tags.as_object().is_some_and(|o| !o.is_empty()) {
                body["tags"] = tags.clone();
            }
            let payload = serde_json::to_string(&body)?;
            // Retry transient gateway failures (5xx, timeouts, dropped connections) — the local LM
            // provider occasionally hiccups and would otherwise zero out the whole question.
            let mut attempt = 0u32;
            let resp = loop {
                match ureq::post(&url)
                    .timeout(timeout)
                    .set("Content-Type", "application/json")
                    .send_string(&payload)
                {
                    Ok(r) => break r,
                    Err(e) => {
                        let retryable = match &e {
                            ureq::Error::Status(code, _) => *code >= 500,
                            ureq::Error::Transport(_) => true,
                        };
                        attempt += 1;
                        if retryable && attempt <= 3 {
                            std::thread::sleep(std::time::Duration::from_millis(500 * u64::from(attempt)));
                            continue;
                        }
                        return Err(anyhow!("tensorzero /inference failed: {e}"));
                    }
                }
            };
            let text = resp.into_string().context("read /inference response")?;
            let v: Value = serde_json::from_str(&text).context("parse /inference json")?;
            if let Some(err) = v.get("error") {
                bail!("tensorzero error: {err}");
            }
            let episode_id = v.get("episode_id").and_then(|x| x.as_str()).unwrap_or("").to_string();
            let content = v.get("content").and_then(|c| c.as_array()).cloned().unwrap_or_default();
            prof!("[prof] infer {}ms  ctx_msgs={}", t0.elapsed().as_millis(), messages.len());
            Ok(TzTurn { content, episode_id })
        };

        let trace = TraceLog::to_dir(work);
        // Open the index once per question; the closure reuses it (cached reader) for every
        // search/read round instead of reopening per tool call.
        let idx = glossa::index::store::DocIndex::open_or_create(work)?;
        let graph = glossa::graph::store::GraphStore::open(work).ok();
        // Ontology-driven chain spec (spine relations + MENTIONS) so glossary/neighbors render
        // the reasoning chain identically to the MCP surface.
        let spec = glossa::tools::ChainSpec::from_ontology(&glossa::graph::ontology::Ontology::load_or_default(work));
        let exec = |name: &str, args: &Value| {
            let t = std::time::Instant::now();
            let r = crate::backend::glossa_tools::exec(name, args, &idx, graph.as_ref(), &spec, &trace);
            prof!("[prof] tool {name} {}ms", t.elapsed().as_millis());
            r
        };

        let user = prompt::user_prompt(q);
        let q0 = std::time::Instant::now();
        let outcome = run_episode(chat, &user, exec, MAX_ROUNDS, EpisodePolicy::answer())?;
        prof!("[prof] --- episode loop {}ms ---", q0.elapsed().as_millis());
        let pred = prompt::parse_answer(&outcome.answer);

        // Post Track-B feedback for this episode (best-effort; never fail the run on feedback error).
        {
            // dedup surfaced titles by normalized form, first-occurrence order = the merged ranking
            let mut seen = std::collections::HashSet::new();
            let ranked: Vec<String> = outcome
                .surfaced_titles
                .iter()
                .filter(|t| seen.insert(crate::score::normalize(t)))
                .cloned()
                .collect();
            let em = crate::score::exact_match(&pred, &q.answer);
            let f1 = crate::score::token_f1(&pred, &q.answer);
            let retrieved = retrieved_any(&outcome.surfaced_titles, &q.supporting_titles);
            self.feedback(&eid, "em", json!(em));
            self.feedback(&eid, "f1", json!(f1));
            self.feedback(&eid, "retrieved", json!(retrieved));
            self.feedback(&eid, "recall_at_5", json!(crate::score::recall_at_k(&ranked, &q.supporting_titles, 5)));
            self.feedback(&eid, "recall_at_10", json!(crate::score::recall_at_k(&ranked, &q.supporting_titles, 10)));
            self.feedback(&eid, "recall_at_20", json!(crate::score::recall_at_k(&ranked, &q.supporting_titles, 20)));
            self.feedback(&eid, "mrr", json!(crate::score::mrr(&ranked, &q.supporting_titles)));
            let tj = std::time::Instant::now();
            let j = self.judge_score(&q.question, &q.answer, &pred);
            prof!("[prof] judge {}ms", tj.elapsed().as_millis());
            if let Some(j) = j {
                self.feedback(&eid, "judge", json!(j));
            }
        }
        Ok(pred)
    }
}

impl TensorZeroBackend {
    fn feedback(&self, episode_id: &str, metric: &str, value: Value) {
        let url = format!("{}/feedback", crate::tz::gateway_base(&self.endpoint));
        let mut body = json!({ "episode_id": episode_id, "metric_name": metric, "value": value });
        if self.tags.as_object().is_some_and(|o| !o.is_empty()) {
            body["tags"] = self.tags.clone();
        }
        let _ = ureq::post(&url)
            .timeout(self.timeout)
            .set("Content-Type", "application/json")
            .send_string(&serde_json::to_string(&body).unwrap_or_default());
    }

    /// LLM-judge: rate the candidate answer against the gold reference (0.0–1.0). None if the judge is
    /// disabled or the call/parse fails. The right correctness metric for free-form answers (token-F1
    /// is a poor fit for multi-sentence prose).
    fn judge_score(&self, question: &str, gold: &str, answer: &str) -> Option<f32> {
        let endpoint = self.judge_endpoint.as_ref()?;
        let url = format!("{}/v1/chat/completions", endpoint.trim_end_matches('/'));
        let prompt = format!(
            "You grade a candidate answer against a reference answer for a technical support question \
             (industrial automation, АБАК PLC).\nQuestion: {question}\nReference (correct) answer: {gold}\n\
             Candidate answer: {answer}\nHow correct is the candidate versus the reference? Reply with ONLY \
             a number from 0.0 (wrong/contradictory) to 1.0 (fully correct/equivalent); partial credit \
             allowed. Number only."
        );
        let body = json!({ "model": self.judge_model, "temperature": 0.0,
            "messages": [{ "role": "user", "content": prompt }] });
        let mut req = ureq::post(&url).timeout(self.timeout).set("Content-Type", "application/json");
        if let Some(k) = &self.judge_api_key {
            req = req.set("Authorization", &format!("Bearer {k}"));
        }
        let text = req.send_string(&serde_json::to_string(&body).ok()?).ok()?.into_string().ok()?;
        let v: Value = serde_json::from_str(&text).ok()?;
        let content = v["choices"][0]["message"]["content"].as_str()?;
        parse_first_float(content).map(|f| f.clamp(0.0, 1.0))
    }
}

/// First non-negative float found in a string (e.g. `0.8` from a judge reply). None if absent.
fn parse_first_float(s: &str) -> Option<f32> {
    let mut num = String::new();
    for ch in s.chars() {
        if ch.is_ascii_digit() || (ch == '.' && !num.contains('.')) {
            num.push(ch);
        } else if !num.is_empty() {
            break;
        }
    }
    num.parse::<f32>().ok()
}

/// True if any gold supporting title appears among the titles the agent's searches surfaced.
/// Empty gold is trivially satisfied; an empty `surfaced` with real gold is NOT (it retrieved nothing).
fn retrieved_any(surfaced: &[String], gold: &[String]) -> bool {
    if gold.is_empty() {
        return true;
    }
    let surf: Vec<String> = surfaced.iter().map(|s| crate::score::normalize(s)).collect();
    gold.iter().any(|g| surf.contains(&crate::score::normalize(g)))
}

#[cfg(test)]
mod retrieved_tests {
    use super::*;

    #[test]
    fn backdated_episode_id_is_valid_past_v7() {
        let id = crate::tz::backdated_episode_id(30);
        let u = uuid::Uuid::parse_str(&id).unwrap();
        assert_eq!(u.get_version_num(), 7, "must be UUIDv7");
        let (secs, _) = u.get_timestamp().unwrap().to_unix();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        assert!(secs <= now && secs + 120 >= now, "timestamp must be a few seconds in the PAST");
    }

    #[test]
    fn parse_first_float_extracts_judge_score() {
        assert_eq!(parse_first_float("0.8"), Some(0.8));
        assert_eq!(parse_first_float("Score: 1.0 (fully correct)"), Some(1.0));
        assert_eq!(parse_first_float("the answer is wrong"), None);
    }

    #[test]
    fn parse_first_float_trailing_period_is_not_decimal() {
        // A number ending a sentence (e.g. "0.8.") must still parse correctly.
        assert_eq!(parse_first_float("0.8. correct"), Some(0.8));
        assert_eq!(parse_first_float("1.0."), Some(1.0));
        assert_eq!(parse_first_float("score: 3"), Some(3.0));
    }

    #[test]
    fn retrieved_any_semantics() {
        assert!(retrieved_any(&["The Beatles".into()], &["the beatles".into()])); // normalized match
        assert!(!retrieved_any(&["X".into()], &["Y".into()]));                     // surfaced, but no match
        assert!(!retrieved_any(&[], &["anything".into()]));                        // NO search -> not retrieved
        assert!(retrieved_any(&["whatever".into()], &[]));                         // empty gold -> trivially true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn episode_executes_tool_then_answers() {
        let round = RefCell::new(0usize);
        let chat = |_msgs: &[Value], _eid: Option<&str>| {
            let mut r = round.borrow_mut();
            *r += 1;
            if *r == 1 {
                Ok(TzTurn {
                    content: vec![json!({ "type": "tool_call", "id": "c1", "name": "search", "arguments": { "query": "corliss" } })],
                    episode_id: "ep1".into(),
                })
            } else {
                Ok(TzTurn {
                    content: vec![json!({ "type": "text", "text": "ANSWER: Chief of Protocol" })],
                    episode_id: "ep1".into(),
                })
            }
        };
        let exec = |name: &str, args: &Value| {
            assert_eq!(name, "search");
            assert_eq!(args["query"], "corliss");
            ("Meet_Corliss_Archer.md:Meet Corliss Archer: …  [9.0]".to_string(),
             vec!["Meet Corliss Archer".to_string()],
             Vec::new())
        };
        let out = run_episode(chat, "Question: ...", exec, 8, EpisodePolicy::answer()).unwrap();
        assert_eq!(out.answer, "ANSWER: Chief of Protocol");
        assert_eq!(out.episode_id.as_deref(), Some("ep1"));
        assert_eq!(out.surfaced_titles, vec!["Meet Corliss Archer".to_string()]);
    }

    #[test]
    fn episode_returns_direct_answer() {
        let chat = |_: &[Value], _: Option<&str>| Ok(TzTurn {
            content: vec![json!({ "type": "text", "text": "ANSWER: yes" })],
            episode_id: "e".into(),
        });
        let exec = |_: &str, _: &Value| (String::new(), Vec::new(), Vec::new());
        let out = run_episode(chat, "q", exec, 4, EpisodePolicy::answer()).unwrap();
        assert_eq!(out.answer, "ANSWER: yes");
    }

    #[test]
    fn empty_episode_id_is_treated_as_none() {
        let chat = |_: &[Value], _: Option<&str>| Ok(TzTurn {
            content: vec![json!({ "type": "text", "text": "ANSWER: x" })],
            episode_id: "".into(),
        });
        let exec = |_: &str, _: &Value| (String::new(), Vec::new(), Vec::new());
        let out = run_episode(chat, "q", exec, 4, EpisodePolicy::answer()).unwrap();
        assert_eq!(out.episode_id, None, "an empty episode_id must not become Some(\"\")");
    }

    #[test]
    fn image_user_message_uses_working_tz_shape() {
        let imgs = vec![glossa::read::DocImage { mime: "image/png".into(), bytes: vec![1, 2, 3] }];
        let m = image_user_message(&imgs).unwrap();
        assert_eq!(m["role"], "user");
        let blocks = m["content"].as_array().unwrap();
        // exactly one image block, in the shape Task 1's spike proved works:
        assert!(blocks.iter().any(|b| b["type"] == "image"
            && b["mime_type"].is_string()
            && b["data"].is_string()),
            "image block must use Format A: {{type:image, mime_type, data}}");
        assert!(image_user_message(&[]).is_none());
    }

    /// Two tool_call blocks in a single round must produce ONE assistant message with
    /// both blocks, TWO separate tool_result user messages, and both execs must fire.
    #[test]
    fn parallel_tool_calls_produce_one_assistant_message() {
        use std::cell::RefCell;
        use std::sync::{Arc, Mutex};

        let round = RefCell::new(0usize);
        let captured_msgs: RefCell<Vec<Value>> = RefCell::new(Vec::new());

        let chat = |msgs: &[Value], _eid: Option<&str>| {
            let mut r = round.borrow_mut();
            *r += 1;
            if *r == 1 {
                Ok(TzTurn {
                    content: vec![
                        json!({ "type": "tool_call", "id": "c1", "name": "search", "arguments": { "query": "alpha" } }),
                        json!({ "type": "tool_call", "id": "c2", "name": "read",   "arguments": { "path": "b.md" } }),
                    ],
                    episode_id: "ep2".into(),
                })
            } else {
                *captured_msgs.borrow_mut() = msgs.to_vec();
                Ok(TzTurn {
                    content: vec![json!({ "type": "text", "text": "ANSWER: done" })],
                    episode_id: "ep2".into(),
                })
            }
        };

        let called_names: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let names_clone = Arc::clone(&called_names);
        let exec = move |name: &str, _args: &Value| {
            names_clone.lock().unwrap().push(name.to_string());
            ("result".to_string(), vec![format!("title-{name}")], Vec::new())
        };

        let out = run_episode(chat, "q", exec, 4, EpisodePolicy::answer()).unwrap();
        assert_eq!(out.answer, "ANSWER: done");

        // Both execs must have fired.
        let mut names = called_names.lock().unwrap().clone();
        names.sort();
        assert_eq!(names, vec!["read".to_string(), "search".to_string()]);

        // surfaced_titles collected from both calls.
        let mut titles = out.surfaced_titles.clone();
        titles.sort();
        assert_eq!(titles, vec!["title-read".to_string(), "title-search".to_string()]);

        // Message structure after round 1:
        //   [0] user question
        //   [1] assistant { content: [tool_call c1, tool_call c2] }   ← ONE message, TWO blocks
        //   [2] user { tool_result c1 }
        //   [3] user { tool_result c2 }
        let msgs = captured_msgs.borrow();
        assert_eq!(msgs.len(), 4, "expected [user_q, assistant, tool_result_c1, tool_result_c2]");

        let asst = &msgs[1];
        assert_eq!(asst["role"], "assistant");
        let blocks = asst["content"].as_array().expect("content must be array");
        assert_eq!(blocks.len(), 2, "ONE assistant message must contain BOTH tool_call blocks");
        assert!(blocks.iter().any(|b| b["type"] == "tool_call" && b["id"] == "c1"), "c1 block missing");
        assert!(blocks.iter().any(|b| b["type"] == "tool_call" && b["id"] == "c2"), "c2 block missing");

        // Two separate tool_result messages in call order.
        assert_eq!(msgs[2]["role"], "user");
        assert_eq!(msgs[2]["content"][0]["type"], "tool_result");
        assert_eq!(msgs[2]["content"][0]["id"], "c1");
        assert_eq!(msgs[3]["role"], "user");
        assert_eq!(msgs[3]["content"][0]["type"], "tool_result");
        assert_eq!(msgs[3]["content"][0]["id"], "c2");
    }

    #[test]
    fn echoes_raw_arguments_when_arguments_is_null() {
        // When the model emits unparseable JSON args, TZ returns `arguments: null`
        // plus the original string under `raw_arguments`. The echoed assistant
        // tool_call must carry the raw string (TZ rejects a null `arguments`).
        let round = RefCell::new(0usize);
        let seen_round2: RefCell<Vec<Value>> = RefCell::new(Vec::new());
        let chat = |msgs: &[Value], _eid: Option<&str>| {
            let mut r = round.borrow_mut();
            *r += 1;
            if *r == 1 {
                Ok(TzTurn {
                    content: vec![json!({
                        "type": "tool_call", "id": "c1", "name": "read",
                        "arguments": Value::Null,
                        "raw_arguments": "{\"path\":\"a.md\",\"n\":8}"
                    })],
                    episode_id: "ep".into(),
                })
            } else {
                *seen_round2.borrow_mut() = msgs.to_vec();
                Ok(TzTurn {
                    content: vec![json!({ "type": "text", "text": "ANSWER: ok" })],
                    episode_id: "ep".into(),
                })
            }
        };
        let exec = |_: &str, _: &Value| ("result".to_string(), Vec::new(), Vec::new());
        let out = run_episode(chat, "q", exec, 4, EpisodePolicy::answer()).unwrap();
        assert_eq!(out.answer, "ANSWER: ok");
        let msgs = seen_round2.borrow();
        // [user question, assistant tool_call, user tool_result]
        let echoed = &msgs[1]["content"][0];
        assert_eq!(echoed["type"], "tool_call");
        assert_eq!(
            echoed["arguments"],
            json!("{\"path\":\"a.md\",\"n\":8}"),
            "null `arguments` must be replaced by the raw_arguments string"
        );
    }

    #[test]
    fn enrich_stops_on_done_and_never_executes_it() {
        let chat = |_: &[Value], _: Option<&str>| Ok(TzTurn {
            content: vec![json!({ "type": "tool_call", "id": "d1", "name": "done", "arguments": { "note": "chain written" } })],
            episode_id: "e".into(),
        });
        let exec = |name: &str, _: &Value| {
            assert_ne!(name, "done", "`done` is a control signal — it must be intercepted, never executed");
            (String::new(), Vec::new(), Vec::new())
        };
        let out = run_episode(chat, "q", exec, 4, EpisodePolicy::enrich()).unwrap();
        assert_eq!(out.answer, "chain written", "done's note becomes the outcome");
    }

    #[test]
    fn enrich_dedups_identical_readonly_call() {
        use std::cell::RefCell;
        use std::sync::{Arc, Mutex};
        let round = RefCell::new(0usize);
        let last: RefCell<Vec<Value>> = RefCell::new(Vec::new());
        // Rounds 1 & 2 issue the SAME read(a.md, 1); round 3 finishes with `done`.
        let chat = |msgs: &[Value], _: Option<&str>| {
            let mut r = round.borrow_mut();
            *r += 1;
            match *r {
                1 | 2 => Ok(TzTurn {
                    content: vec![json!({ "type": "tool_call", "id": "c", "name": "read", "arguments": { "path": "a.md", "n": 1 } })],
                    episode_id: "e".into(),
                }),
                _ => {
                    *last.borrow_mut() = msgs.to_vec();
                    Ok(TzTurn { content: vec![json!({ "type": "tool_call", "id": "d", "name": "done", "arguments": {} })], episode_id: "e".into() })
                }
            }
        };
        let reads = Arc::new(Mutex::new(0usize));
        let rc = Arc::clone(&reads);
        let exec = move |name: &str, _: &Value| {
            if name == "read" { *rc.lock().unwrap() += 1; }
            ("page text".to_string(), Vec::new(), Vec::new())
        };
        run_episode(chat, "q", exec, 8, EpisodePolicy::enrich()).unwrap();
        assert_eq!(*reads.lock().unwrap(), 1, "an identical read must execute only once");
        // the 2nd identical read fed back the dedup hint, not a fresh execution.
        let msgs = last.borrow();
        let hinted = msgs.iter().any(|m| m["content"][0]["result"].as_str().map(|s| s.contains("(skipped)")).unwrap_or(false));
        assert!(hinted, "the deduped call must return the (skipped) hint");
    }

    #[test]
    fn enrich_dedups_repeated_mutator_until_state_changes() {
        use std::cell::RefCell;
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex};
        let round = RefCell::new(0usize);
        // generalize, generalize(dup), upsert(changes graph), generalize(now fresh again), done.
        let script = ["graph_generalize", "graph_generalize", "graph_upsert", "graph_generalize"];
        let chat = |_: &[Value], _: Option<&str>| {
            let mut r = round.borrow_mut();
            let i = *r;
            *r += 1;
            if i < script.len() {
                Ok(TzTurn { content: vec![json!({ "type": "tool_call", "id": format!("c{i}"), "name": script[i], "arguments": {} })], episode_id: "e".into() })
            } else {
                Ok(TzTurn { content: vec![json!({ "type": "tool_call", "id": "d", "name": "done", "arguments": {} })], episode_id: "e".into() })
            }
        };
        let counts: Arc<Mutex<HashMap<String, usize>>> = Arc::new(Mutex::new(HashMap::new()));
        let cc = Arc::clone(&counts);
        let exec = move |name: &str, _: &Value| {
            *cc.lock().unwrap().entry(name.to_string()).or_default() += 1;
            (String::new(), Vec::new(), Vec::new())
        };
        run_episode(chat, "q", exec, 10, EpisodePolicy::enrich()).unwrap();
        let c = counts.lock().unwrap();
        assert_eq!(c.get("graph_generalize").copied().unwrap_or(0), 2, "2nd generalize deduped; 3rd re-runs because the upsert changed the graph");
        assert_eq!(c.get("graph_upsert").copied().unwrap_or(0), 1);
    }

    #[test]
    fn enrich_graph_mutation_invalidates_repeated_graph_read() {
        use std::cell::RefCell;
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex};
        let round = RefCell::new(0usize);
        // glossary, glossary(dup), upsert, glossary(now stale → re-runs), done.
        let chat = |_: &[Value], _: Option<&str>| {
            let mut r = round.borrow_mut();
            let i = *r;
            *r += 1;
            let block = match i {
                0 | 1 | 3 => json!({ "type": "tool_call", "id": format!("c{i}"), "name": "glossary", "arguments": { "concept": "насос" } }),
                2 => json!({ "type": "tool_call", "id": "u", "name": "graph_upsert", "arguments": {} }),
                _ => json!({ "type": "tool_call", "id": "d", "name": "done", "arguments": {} }),
            };
            Ok(TzTurn { content: vec![block], episode_id: "e".into() })
        };
        let counts: Arc<Mutex<HashMap<String, usize>>> = Arc::new(Mutex::new(HashMap::new()));
        let cc = Arc::clone(&counts);
        let exec = move |name: &str, _: &Value| {
            *cc.lock().unwrap().entry(name.to_string()).or_default() += 1;
            (String::new(), Vec::new(), Vec::new())
        };
        run_episode(chat, "q", exec, 10, EpisodePolicy::enrich()).unwrap();
        let c = counts.lock().unwrap();
        assert_eq!(c.get("glossary").copied().unwrap_or(0), 2, "repeat glossary deduped, but re-runs after the upsert (graph changed)");
        assert_eq!(c.get("graph_upsert").copied().unwrap_or(0), 1);
    }

    #[test]
    fn enrich_text_only_turn_does_not_end_episode() {
        use std::cell::RefCell;
        let round = RefCell::new(0usize);
        let last: RefCell<Vec<Value>> = RefCell::new(Vec::new());
        let chat = |msgs: &[Value], _: Option<&str>| {
            let mut r = round.borrow_mut();
            *r += 1;
            if *r == 1 {
                // narrate-then-stop: prose, no tool call.
                Ok(TzTurn { content: vec![json!({ "type": "text", "text": "Let me read the chunk first." })], episode_id: "e".into() })
            } else {
                *last.borrow_mut() = msgs.to_vec();
                Ok(TzTurn { content: vec![json!({ "type": "tool_call", "id": "d", "name": "done", "arguments": { "note": "ok" } })], episode_id: "e".into() })
            }
        };
        let exec = |_: &str, _: &Value| (String::new(), Vec::new(), Vec::new());
        let out = run_episode(chat, "q", exec, 4, EpisodePolicy::enrich()).unwrap();
        assert_eq!(out.answer, "ok");
        assert!(*round.borrow() >= 2, "a text-only turn must NOT end an enrich episode");
        let msgs = last.borrow();
        let nudged = msgs.iter().any(|m| m["role"] == "user"
            && m["content"][0]["text"].as_str().map(|s| s.contains("without calling a tool")).unwrap_or(false));
        assert!(nudged, "a nudge must follow a text-only enrich turn");
    }
}
