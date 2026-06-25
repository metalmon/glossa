use serde_json::{json, Value};

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

/// Drive a TensorZero episode to a final text answer.
///
/// `chat(messages, episode_id)` performs one `/inference` call (episode_id is None on the first turn,
/// then the id returned by the first turn). When the returned content has `tool_call` blocks, each is
/// executed via `exec(name, args)` and fed back as a `tool_result` content block; otherwise the
/// concatenated `text` blocks are the answer.
pub fn run_episode<C, X>(
    mut chat: C,
    user_question: &str,
    mut exec: X,
    max_rounds: usize,
) -> anyhow::Result<EpisodeOutcome>
where
    C: FnMut(&[Value], Option<&str>) -> anyhow::Result<TzTurn>,
    X: FnMut(&str, &Value) -> (String, Vec<String>),
{
    let mut messages: Vec<Value> = vec![json!({ "role": "user", "content": user_question })];
    let mut episode_id: Option<String> = None;
    let mut surfaced_titles: Vec<String> = Vec::new();

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
            return Ok(EpisodeOutcome { answer, episode_id, surfaced_titles });
        }

        for call in &tool_calls {
            let id = call.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let name = call.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let args = call.get("arguments").cloned().unwrap_or_else(|| json!({}));
            // echo the assistant tool_call, then the tool_result (result MUST be a string)
            messages.push(json!({
                "role": "assistant",
                "content": [{ "type": "tool_call", "id": id, "name": name, "arguments": args }]
            }));
            let (result, titles) = exec(name, &args);
            surfaced_titles.extend(titles);
            messages.push(json!({
                "role": "user",
                "content": [{ "type": "tool_result", "id": id, "name": name, "result": result }]
            }));
        }
    }
    // Out of rounds: best-effort empty answer (the report still scores it).
    Ok(EpisodeOutcome { answer: String::new(), episode_id, surfaced_titles })
}

use super::{prompt, AgentBackend};
use crate::dataset::Question;
use anyhow::{anyhow, bail, Context};
use glossa::trace::TraceLog;
use std::path::Path;
use std::time::Duration;

const MAX_ROUNDS: usize = 8;

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

    fn answer(&self, work: &Path, q: &Question) -> anyhow::Result<String> {
        let url = format!("{}/inference", self.endpoint.trim_end_matches('/'));
        let function = self.function.clone();
        let timeout = self.timeout;
        let tags = self.tags.clone();
        // Client-generated episode_id, back-dated 30s so its UUIDv7 timestamp is always in the PAST
        // relative to the gateway's clock — immune to Docker/WSL host↔container clock skew. The same id
        // is sent on every turn, so all inferences group into one episode (telemetry + feedback).
        let eid = backdated_episode_id(30);
        let chat = |messages: &[Value], _episode_id: Option<&str>| -> anyhow::Result<TzTurn> {
            let mut body = json!({ "function_name": function, "input": { "messages": messages }, "episode_id": eid });
            if tags.as_object().is_some_and(|o| !o.is_empty()) {
                body["tags"] = tags.clone();
            }
            let resp = ureq::post(&url)
                .timeout(timeout)
                .set("Content-Type", "application/json")
                .send_string(&serde_json::to_string(&body)?)
                .map_err(|e| anyhow!("tensorzero /inference failed: {e}"))?;
            let text = resp.into_string().context("read /inference response")?;
            let v: Value = serde_json::from_str(&text).context("parse /inference json")?;
            if let Some(err) = v.get("error") {
                bail!("tensorzero error: {err}");
            }
            let episode_id = v.get("episode_id").and_then(|x| x.as_str()).unwrap_or("").to_string();
            let content = v.get("content").and_then(|c| c.as_array()).cloned().unwrap_or_default();
            Ok(TzTurn { content, episode_id })
        };

        let trace = TraceLog::to_dir(work);
        let exec = |name: &str, args: &Value| crate::backend::glossa_tools::exec(name, args, work, &trace);

        let user = prompt::user_prompt(q);
        let outcome = run_episode(chat, &user, exec, MAX_ROUNDS)?;
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
            if let Some(j) = self.judge_score(&q.question, &q.answer, &pred) {
                self.feedback(&eid, "judge", json!(j));
            }
        }
        Ok(pred)
    }
}

impl TensorZeroBackend {
    fn feedback(&self, episode_id: &str, metric: &str, value: Value) {
        let url = format!("{}/feedback", self.endpoint.trim_end_matches('/'));
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
        if ch.is_ascii_digit() || ch == '.' {
            num.push(ch);
        } else if !num.is_empty() {
            break;
        }
    }
    num.parse::<f32>().ok()
}

/// A UUIDv7 episode id whose embedded timestamp is `secs_back` seconds in the PAST, so the gateway
/// never rejects it as "in the future" under Docker/WSL host↔container clock skew. Reused across an
/// episode's turns to keep all inferences grouped.
fn backdated_episode_id(secs_back: u64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs().saturating_sub(secs_back);
    let ts = uuid::Timestamp::from_unix(uuid::NoContext, secs, now.subsec_nanos());
    uuid::Uuid::new_v7(ts).to_string()
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
        let id = backdated_episode_id(30);
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
             vec!["Meet Corliss Archer".to_string()])
        };
        let out = run_episode(chat, "Question: ...", exec, 8).unwrap();
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
        let exec = |_: &str, _: &Value| (String::new(), Vec::new());
        let out = run_episode(chat, "q", exec, 4).unwrap();
        assert_eq!(out.answer, "ANSWER: yes");
    }

    #[test]
    fn empty_episode_id_is_treated_as_none() {
        let chat = |_: &[Value], _: Option<&str>| Ok(TzTurn {
            content: vec![json!({ "type": "text", "text": "ANSWER: x" })],
            episode_id: "".into(),
        });
        let exec = |_: &str, _: &Value| (String::new(), Vec::new());
        let out = run_episode(chat, "q", exec, 4).unwrap();
        assert_eq!(out.episode_id, None, "an empty episode_id must not become Some(\"\")");
    }
}
