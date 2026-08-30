//! Fine-tuning dataset collection — the provider-agnostic capture + export layer.
//!
//! Phase 1 (capture) records a reader episode's full chat trajectory via
//! [`crate::backend::agent_loop::CapturedEpisode`]; `kbx eval --capture` joins the graded judge
//! [`Verdict`] as the reward and writes one [`TrajectoryRecord`] per sampled episode to
//! `runs/<tag>/trajectories.jsonl`.
//!
//! Phase 2 (export) is this module's pure, deterministic, network-free post-process:
//! [`export_sft`] turns the Correct trajectories into Unsloth SFT lines (default `messages` shape,
//! optional `sharegpt` `conversations` shape) and [`export_dpo`] pairs a Correct against a Wrong
//! trajectory per question into TRL/Unsloth `DPOTrainer` `{prompt, chosen, rejected}` rows. The
//! reward is the evidence-grounded judge verdict already computed at eval time — nothing here calls
//! a model.

use crate::judge::Verdict;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// One captured reader episode as persisted to `runs/<tag>/trajectories.jsonl` (one JSON object per
/// line). `messages` is the complete OpenAI/ChatML trajectory (system, user, every assistant turn
/// with its `tool_calls`, every `tool` result, and the final assistant answer); `verdict` is the
/// joined reward. `--samples N` accumulates several rows sharing an `id`/`question` with varied
/// outcomes — the raw material DPO needs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryRecord {
    pub id: String,
    pub question: String,
    pub model: String,
    /// Tools schema advertised to the model this episode (`null` when tools were disabled).
    #[serde(default)]
    pub tools: Value,
    /// The full chat trajectory, ending in the final assistant answer turn.
    pub messages: Vec<Value>,
    /// The parsed final answer (post `parse_answer`), for reference/debugging.
    #[serde(default)]
    pub answer: String,
    /// The graded judge verdict — the reward that drives SFT keep / DPO chosen-vs-rejected.
    pub verdict: Verdict,
    #[serde(default)]
    pub hop_type: String,
}

/// Output shape for SFT export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SftShape {
    /// `{"messages":[{"role":..,"content":..}, ...]}` for Unsloth `apply_chat_template` (default).
    Messages,
    /// `{"conversations":[{"from":"system|human|gpt","value":..}]}` for `standardize_sharegpt`.
    Sharegpt,
}

/// Append every record in `records` as one JSON line to `<dir>/trajectories.jsonl`, creating `dir`
/// if needed, and return the file path. Deterministic (records are written in the given order).
pub fn write_trajectories(dir: &Path, records: &[TrajectoryRecord]) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating trajectories dir {}", dir.display()))?;
    let path = dir.join("trajectories.jsonl");
    let mut out = String::new();
    for r in records {
        out.push_str(&serde_json::to_string(r)?);
        out.push('\n');
    }
    std::fs::write(&path, out).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// Load `trajectories.jsonl` for one run tag: read `<runs>/<tag>/trajectories.jsonl` and parse each
/// non-blank line into a [`TrajectoryRecord`]. A missing file yields an empty vec (a tag that was
/// never captured contributes nothing), so `export --from a,b` over mixed tags never
/// panics.
pub fn load_trajectories_for_tag(runs_dir: &Path, tag: &str) -> Result<Vec<TrajectoryRecord>> {
    let path = runs_dir.join(tag).join("trajectories.jsonl");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    parse_trajectories(&text)
}

/// Parse JSONL trajectory text into records (blank lines skipped). Empty input → empty vec.
pub fn parse_trajectories(text: &str) -> Result<Vec<TrajectoryRecord>> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let r: TrajectoryRecord = serde_json::from_str(line)
            .with_context(|| format!("parsing trajectory line {}", i + 1))?;
        out.push(r);
    }
    Ok(out)
}

/// Whether a verdict qualifies as an SFT demonstration: `Correct` always; `Partial` only when
/// `include_partial`. `Wrong`/`Unscored` never.
fn keep_for_sft(v: Verdict, include_partial: bool) -> bool {
    matches!(v, Verdict::Correct) || (include_partial && matches!(v, Verdict::Partial))
}

/// Build the SFT lines from captured trajectories. One JSON object per kept trajectory (Correct,
/// plus Partial when `include_partial`), in input order. `Messages` passes the captured `messages`
/// through as `{"messages":[...]}`; `Sharegpt` maps them to `{"conversations":[{"from","value"}]}`.
pub fn export_sft(
    records: &[TrajectoryRecord],
    shape: SftShape,
    include_partial: bool,
) -> Vec<Value> {
    records
        .iter()
        .filter(|r| keep_for_sft(r.verdict, include_partial))
        .map(|r| match shape {
            SftShape::Messages => json!({ "messages": r.messages }),
            SftShape::Sharegpt => json!({ "conversations": to_conversations(&r.messages) }),
        })
        .collect()
}

/// Map an OpenAI/ChatML `messages` array to ShareGPT `conversations`: system→system, user→human,
/// tool→human (tool results are observations fed back into the human side of the flow),
/// assistant→gpt. Each `value` is the message rendered to a string (see [`message_text`]).
fn to_conversations(messages: &[Value]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("");
            let from = match role {
                "system" => "system",
                "assistant" => "gpt",
                // user + tool both fold into the human side of the dialogue.
                _ => "human",
            };
            json!({ "from": from, "value": message_text(m) })
        })
        .collect()
}

/// Render one chat message to a plain string for ShareGPT / DPO completions. Prefers a string
/// `content`; joins the `text` parts of an array (vision) content; else falls back to a compact
/// JSON of `tool_calls` (an assistant tool-request turn with null content), else empty.
fn message_text(m: &Value) -> String {
    match m.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => m
            .get("tool_calls")
            .map(|tc| serde_json::to_string(tc).unwrap_or_default())
            .unwrap_or_default(),
    }
}

/// The prompt portion of a trajectory: every leading message up to (but not including) the first
/// assistant turn — i.e. the system + user seed. Used as the DPO `prompt` (a messages array).
fn prompt_messages(messages: &[Value]) -> Vec<Value> {
    messages
        .iter()
        .take_while(|m| m.get("role").and_then(Value::as_str) != Some("assistant"))
        .cloned()
        .collect()
}

/// The final assistant answer of a trajectory, as a string (the DPO chosen/rejected completion).
/// Reads the last `assistant` message's rendered text; falls back to the record's parsed `answer`.
fn final_completion(r: &TrajectoryRecord) -> String {
    r.messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
        .map(message_text)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| r.answer.clone())
}

/// Outcome of a DPO export: the pair rows plus how many questions were skipped for lacking both a
/// Correct and a Wrong trajectory.
pub struct DpoExport {
    pub pairs: Vec<Value>,
    pub questions_skipped: usize,
}

/// Build DPO `{prompt, chosen, rejected}` rows. Group the trajectories by question `id`; for each
/// group that has BOTH a Correct and a Wrong trajectory, emit up to `max_pairs` rows pairing
/// Correct[i] (chosen) with Wrong[i] (rejected). `prompt` is the shared system+user seed as a
/// messages array. Groups lacking either class are skipped and counted. Deterministic: groups are
/// emitted in first-seen id order, trajectories in input order.
pub fn export_dpo(records: &[TrajectoryRecord], max_pairs: usize) -> DpoExport {
    // Preserve first-seen id order for deterministic output.
    let mut order: Vec<String> = Vec::new();
    let mut groups: std::collections::HashMap<String, Vec<&TrajectoryRecord>> =
        std::collections::HashMap::new();
    for r in records {
        groups.entry(r.id.clone()).or_insert_with(|| {
            order.push(r.id.clone());
            Vec::new()
        });
        groups.get_mut(&r.id).unwrap().push(r);
    }

    let mut pairs = Vec::new();
    let mut questions_skipped = 0usize;
    for id in &order {
        let group = &groups[id];
        let correct: Vec<&&TrajectoryRecord> = group
            .iter()
            .filter(|r| matches!(r.verdict, Verdict::Correct))
            .collect();
        let wrong: Vec<&&TrajectoryRecord> = group
            .iter()
            .filter(|r| matches!(r.verdict, Verdict::Wrong))
            .collect();
        if correct.is_empty() || wrong.is_empty() {
            questions_skipped += 1;
            continue;
        }
        let n = max_pairs.min(correct.len()).min(wrong.len());
        for i in 0..n {
            let chosen = correct[i];
            let rejected = wrong[i];
            // TRL/Unsloth conversational DPO: `prompt` is the [system,user] seed; `chosen`/
            // `rejected` are message-lists holding the FINAL assistant answer turn (not raw
            // strings), so the preference signal is correct-answer vs wrong-answer given the prompt.
            pairs.push(json!({
                "prompt": prompt_messages(&chosen.messages),
                "chosen": [{ "role": "assistant", "content": final_completion(chosen) }],
                "rejected": [{ "role": "assistant", "content": final_completion(rejected) }],
            }));
        }
    }
    DpoExport { pairs, questions_skipped }
}

/// Serialize export rows to JSONL text (one compact object per line, trailing newline per row).
pub fn to_jsonl(rows: &[Value]) -> String {
    let mut out = String::new();
    for r in rows {
        out.push_str(&serde_json::to_string(r).unwrap_or_default());
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic trajectory record (no corpus/gold values — pure arithmetic placeholders).
    fn rec(id: &str, verdict: Verdict, user: &str, answer: &str) -> TrajectoryRecord {
        TrajectoryRecord {
            id: id.to_string(),
            question: user.to_string(),
            model: "test-model".to_string(),
            tools: json!([{"type": "function", "function": {"name": "search"}}]),
            messages: vec![
                json!({"role": "system", "content": "You answer questions."}),
                json!({"role": "user", "content": user}),
                json!({"role": "assistant", "content": null, "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "search", "arguments": "{\"q\":\"n\"}"}}
                ]}),
                json!({"role": "tool", "tool_call_id": "c1", "content": "result body"}),
                json!({"role": "assistant", "content": answer}),
            ],
            answer: answer.to_string(),
            verdict,
            hop_type: String::new(),
        }
    }

    #[test]
    fn sft_messages_keeps_only_correct_by_default() {
        let recs = vec![
            rec("q1", Verdict::Correct, "What is 2+2?", "ANSWER: 4"),
            rec("q2", Verdict::Wrong, "What is 3+3?", "ANSWER: 5"),
            rec("q3", Verdict::Partial, "What is 5+5?", "ANSWER: about 10"),
        ];
        let lines = export_sft(&recs, SftShape::Messages, false);
        assert_eq!(lines.len(), 1, "only the Correct trajectory is a demonstration");
        // Line is `{"messages":[...]}` passed through verbatim, ending in the final answer turn.
        let msgs = lines[0].get("messages").and_then(Value::as_array).unwrap();
        assert_eq!(msgs.len(), 5);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs.last().unwrap()["content"], "ANSWER: 4");
    }

    #[test]
    fn sft_include_partial_adds_partial() {
        let recs = vec![
            rec("q1", Verdict::Correct, "What is 2+2?", "ANSWER: 4"),
            rec("q3", Verdict::Partial, "What is 5+5?", "ANSWER: about 10"),
        ];
        assert_eq!(export_sft(&recs, SftShape::Messages, false).len(), 1);
        assert_eq!(export_sft(&recs, SftShape::Messages, true).len(), 2);
    }

    #[test]
    fn sft_sharegpt_maps_roles() {
        let recs = vec![rec("q1", Verdict::Correct, "What is 2+2?", "ANSWER: 4")];
        let lines = export_sft(&recs, SftShape::Sharegpt, false);
        let conv = lines[0].get("conversations").and_then(Value::as_array).unwrap();
        assert_eq!(conv[0]["from"], "system");
        assert_eq!(conv[1]["from"], "human"); // user -> human
        assert_eq!(conv[1]["value"], "What is 2+2?");
        assert_eq!(conv[2]["from"], "gpt"); // assistant tool-call turn -> gpt
        assert_eq!(conv[3]["from"], "human"); // tool result -> human side
        assert_eq!(conv[4]["from"], "gpt");
        assert_eq!(conv[4]["value"], "ANSWER: 4");
    }

    #[test]
    fn dpo_pairs_correct_with_wrong_and_skips_single_class() {
        let recs = vec![
            // q1 has both classes -> one pair.
            rec("q1", Verdict::Correct, "What is 2+2?", "ANSWER: 4"),
            rec("q1", Verdict::Wrong, "What is 2+2?", "ANSWER: 5"),
            // q2 only Correct -> skipped.
            rec("q2", Verdict::Correct, "What is 3+3?", "ANSWER: 6"),
            // q3 only Wrong -> skipped.
            rec("q3", Verdict::Wrong, "What is 9+1?", "ANSWER: 11"),
        ];
        let out = export_dpo(&recs, 1);
        assert_eq!(out.pairs.len(), 1);
        assert_eq!(out.questions_skipped, 2);
        let p = &out.pairs[0];
        // prompt is the system+user seed (no assistant turns).
        let prompt = p.get("prompt").and_then(Value::as_array).unwrap();
        assert_eq!(prompt.len(), 2);
        assert_eq!(prompt[0]["role"], "system");
        assert_eq!(prompt[1]["role"], "user");
        // chosen/rejected are message-lists holding the final assistant answer turn.
        assert_eq!(p["chosen"][0]["role"], "assistant");
        assert_eq!(p["chosen"][0]["content"], "ANSWER: 4");
        assert_eq!(p["rejected"][0]["content"], "ANSWER: 5");
    }

    #[test]
    fn dpo_max_pairs_caps_per_question() {
        let recs = vec![
            rec("q1", Verdict::Correct, "What is 2+2?", "ANSWER: 4"),
            rec("q1", Verdict::Correct, "What is 2+2?", "ANSWER: four"),
            rec("q1", Verdict::Wrong, "What is 2+2?", "ANSWER: 5"),
            rec("q1", Verdict::Wrong, "What is 2+2?", "ANSWER: 3"),
        ];
        assert_eq!(export_dpo(&recs, 1).pairs.len(), 1, "default caps at 1 pair/question");
        assert_eq!(export_dpo(&recs, 2).pairs.len(), 2, "max_pairs=2 yields two pairs");
        // Can't exceed the smaller class size.
        assert_eq!(export_dpo(&recs, 9).pairs.len(), 2);
    }

    #[test]
    fn empty_input_yields_empty_output_no_panic() {
        let recs: Vec<TrajectoryRecord> = Vec::new();
        assert!(export_sft(&recs, SftShape::Messages, true).is_empty());
        let out = export_dpo(&recs, 1);
        assert!(out.pairs.is_empty());
        assert_eq!(out.questions_skipped, 0);
        assert_eq!(parse_trajectories("").unwrap().len(), 0);
        assert_eq!(parse_trajectories("\n  \n").unwrap().len(), 0);
    }

    #[test]
    fn write_then_load_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let runs = dir.path().join("runs");
        let tag_dir = runs.join("tagA");
        let recs = vec![
            rec("q1", Verdict::Correct, "What is 2+2?", "ANSWER: 4"),
            rec("q1", Verdict::Wrong, "What is 2+2?", "ANSWER: 5"),
        ];
        let path = write_trajectories(&tag_dir, &recs).unwrap();
        assert!(path.ends_with("trajectories.jsonl"));
        let loaded = load_trajectories_for_tag(&runs, "tagA").unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].id, "q1");
        assert!(matches!(loaded[1].verdict, Verdict::Wrong));
        // A tag that was never captured contributes nothing (no panic).
        assert!(load_trajectories_for_tag(&runs, "missing").unwrap().is_empty());
    }

    #[test]
    fn to_jsonl_is_one_object_per_line() {
        let rows = vec![json!({"a": 1}), json!({"b": 2})];
        let text = to_jsonl(&rows);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "{\"a\":1}");
    }
}
