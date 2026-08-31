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
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
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

/// The stable substring the eval reader's plateau render (`glossa_tools::apply_plateau_render` /
/// `retrieval_progress::plateau_marker`) always includes — "N unique results gathered; M new over
/// the last W searches — **gain has plateaued**". Matching on JUST this tail deliberately excludes
/// the repeat marker ("identical query already run this session — result unchanged") and the streak
/// marker ("last K searches surfaced no new information"): this export feature is plateau-specific,
/// not a general retrieval-signal detector.
const PLATEAU_MARKER_SUBSTR: &str = "gain has plateaued";

/// Whether a captured chat message is a plateau-signal TOOL result — a `role:"tool"` message whose
/// rendered text contains [`PLATEAU_MARKER_SUBSTR`].
fn is_plateau_tool_message(m: &Value) -> bool {
    m.get("role").and_then(Value::as_str) == Some("tool")
        && message_text(m).contains(PLATEAU_MARKER_SUBSTR)
}

/// Index (into `messages`) of the LAST plateau-signal tool message, if the trajectory has one. A
/// trajectory can plateau more than once (`ReaderSignals` re-arms after new ground); only the LAST
/// occurrence matters for "did the reader keep searching AFTER being told gain had plateaued".
fn last_plateau_turn_index(messages: &[Value]) -> Option<usize> {
    messages
        .iter()
        .enumerate()
        .filter(|(_, m)| is_plateau_tool_message(m))
        .map(|(i, _)| i)
        .next_back()
}

/// Whether a trajectory contains at least one plateau-signal tool message anywhere.
fn has_plateau_turn(messages: &[Value]) -> bool {
    last_plateau_turn_index(messages).is_some()
}

/// Whether an assistant message issued at least one tool call (a non-empty `tool_calls` array) —
/// i.e. it kept searching rather than answering in plain text.
fn is_tool_calling_assistant(m: &Value) -> bool {
    m.get("role").and_then(Value::as_str) == Some("assistant")
        && m.get("tool_calls")
            .and_then(Value::as_array)
            .map(|a| !a.is_empty())
            .unwrap_or(false)
}

/// A trajectory "spiraled past" its plateau: it has a plateau-signal tool message AND at least one
/// assistant message AFTER that last plateau turn still issued a tool call — i.e. the reader kept
/// searching once told gain had flatlined, instead of answering. A trajectory with no plateau turn
/// never spirals (there was nothing to spiral past). NOTE: this is purely a TRAJECTORY-SHAPE
/// predicate — it says nothing about whether the answer ended up right or wrong; the judge verdict
/// (not this predicate) is what decides chosen-vs-rejected eligibility (see [`export_dpo`]).
fn spiraled_past_plateau(messages: &[Value]) -> bool {
    match last_plateau_turn_index(messages) {
        Some(idx) => messages[idx + 1..].iter().any(is_tool_calling_assistant),
        None => false,
    }
}

/// Outcome of an SFT export: the kept lines plus how many are "signal-reaction" demonstrations (a
/// kept trajectory that contains a plateau turn — the reader saw the signal and, per its Correct/
/// kept verdict, still landed a good answer).
pub struct SftExport {
    pub rows: Vec<Value>,
    pub signal_reaction: usize,
}

/// Build the SFT lines from captured trajectories. One JSON object per kept trajectory (Correct,
/// plus Partial when `include_partial`). `Messages` passes the captured `messages` through as
/// `{"messages":[...]}`; `Sharegpt` maps them to `{"conversations":[{"from","value"}]}`.
///
/// `prefer_signal`: when set, stable-partitions the kept trajectories so ones containing a plateau
/// turn come first (relative order preserved within each half) — a mild up-weight of "reacted to
/// the signal and still answered correctly" demonstrations, with no other reordering machinery.
/// Default (`false`) keeps today's input order exactly.
pub fn export_sft(
    records: &[TrajectoryRecord],
    shape: SftShape,
    include_partial: bool,
    prefer_signal: bool,
) -> SftExport {
    let kept: Vec<&TrajectoryRecord> = records
        .iter()
        .filter(|r| keep_for_sft(r.verdict, include_partial))
        .collect();
    let signal_reaction = kept
        .iter()
        .filter(|r| has_plateau_turn(&r.messages))
        .count();
    let ordered: Vec<&TrajectoryRecord> = if prefer_signal {
        let (with_signal, without_signal): (Vec<_>, Vec<_>) = kept
            .into_iter()
            .partition(|r| has_plateau_turn(&r.messages));
        with_signal.into_iter().chain(without_signal).collect()
    } else {
        kept
    };
    let rows = ordered
        .into_iter()
        .map(|r| match shape {
            SftShape::Messages => json!({ "messages": r.messages }),
            SftShape::Sharegpt => json!({ "conversations": to_conversations(&r.messages) }),
        })
        .collect();
    SftExport {
        rows,
        signal_reaction,
    }
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

/// Outcome of a DPO export: the pair rows, how many questions were skipped (lacking both a Correct
/// and a Wrong trajectory, or — under `focus_plateau` — lacking a Wrong that spiraled past a
/// plateau), and how many emitted pairs are plateau-contrastive (rejected spiraled past a plateau).
pub struct DpoExport {
    pub pairs: Vec<Value>,
    pub questions_skipped: usize,
    pub plateau_contrastive: usize,
}

/// Build DPO `{prompt, chosen, rejected}` rows. Group the trajectories by question `id`; for each
/// group that has BOTH a Correct and a Wrong trajectory, emit up to `max_pairs` rows pairing a
/// chosen (Correct) with a rejected (Wrong) trajectory. `prompt` is the shared system+user seed as a
/// messages array. Groups lacking either class are skipped and counted. Deterministic: groups are
/// emitted in first-seen id order, trajectories in input order (within a class).
///
/// Plateau-aware pairing (both driven purely by the JUDGE VERDICT — never by trajectory shape):
/// - **chosen** candidates are ordered so a Correct trajectory that ALSO contains a plateau turn
///   (it answered despite / after the signal) comes first; every OTHER Correct trajectory — INCLUDING
///   one that kept searching past its own plateau turn and still landed Correct — remains fully
///   eligible as chosen, just not prioritized to the front. CRITICAL: because chosen is drawn only
///   from `Verdict::Correct`, a Correct trajectory can never end up rejected here, no matter how it
///   reacted to a plateau — that's what prevents training premature give-up.
/// - **rejected** candidates are ordered so a Wrong trajectory that [`spiraled_past_plateau`] (kept
///   searching after the signal instead of answering) comes first; other Wrong trajectories follow.
/// - `focus_plateau`: when set, the rejected pool is restricted to ONLY spiraled-Wrong trajectories
///   — a question whose Wrong trajectories never spiraled is skipped (and counted). When unset, the
///   full Wrong pool is used (spiraled ones biased to the front), so the feature degrades to today's
///   any-Correct-vs-any-Wrong behavior when no trajectory plateaued at all.
pub fn export_dpo(
    records: &[TrajectoryRecord],
    max_pairs: usize,
    focus_plateau: bool,
) -> DpoExport {
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
    let mut plateau_contrastive = 0usize;
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

        // Chosen pool: Correct trajectories, plateau-turn ones biased to the front. Stable
        // partition — every Correct trajectory stays eligible regardless of how it reacted.
        let (chosen_signal, chosen_rest): (Vec<_>, Vec<_>) = correct
            .into_iter()
            .partition(|r| has_plateau_turn(&r.messages));
        let chosen_pool: Vec<&&TrajectoryRecord> =
            chosen_signal.into_iter().chain(chosen_rest).collect();

        // Rejected pool: Wrong trajectories that spiraled past a plateau, biased to the front (or
        // exclusively, under `focus_plateau`).
        let (rejected_spiraled, rejected_rest): (Vec<_>, Vec<_>) = wrong
            .into_iter()
            .partition(|r| spiraled_past_plateau(&r.messages));
        if focus_plateau && rejected_spiraled.is_empty() {
            questions_skipped += 1;
            continue;
        }
        let rejected_pool: Vec<&&TrajectoryRecord> = if focus_plateau {
            rejected_spiraled
        } else {
            rejected_spiraled.into_iter().chain(rejected_rest).collect()
        };

        let n = max_pairs.min(chosen_pool.len()).min(rejected_pool.len());
        for i in 0..n {
            let chosen = chosen_pool[i];
            let rejected = rejected_pool[i];
            if spiraled_past_plateau(&rejected.messages) {
                plateau_contrastive += 1;
            }
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
    DpoExport {
        pairs,
        questions_skipped,
        plateau_contrastive,
    }
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
        let out = export_sft(&recs, SftShape::Messages, false, false);
        assert_eq!(
            out.rows.len(),
            1,
            "only the Correct trajectory is a demonstration"
        );
        assert_eq!(
            out.signal_reaction, 0,
            "no trajectory here has a plateau turn"
        );
        // Line is `{"messages":[...]}` passed through verbatim, ending in the final answer turn.
        let msgs = out.rows[0]
            .get("messages")
            .and_then(Value::as_array)
            .unwrap();
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
        assert_eq!(
            export_sft(&recs, SftShape::Messages, false, false)
                .rows
                .len(),
            1
        );
        assert_eq!(
            export_sft(&recs, SftShape::Messages, true, false)
                .rows
                .len(),
            2
        );
    }

    #[test]
    fn sft_sharegpt_maps_roles() {
        let recs = vec![rec("q1", Verdict::Correct, "What is 2+2?", "ANSWER: 4")];
        let out = export_sft(&recs, SftShape::Sharegpt, false, false);
        let conv = out.rows[0]
            .get("conversations")
            .and_then(Value::as_array)
            .unwrap();
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
        let out = export_dpo(&recs, 1, false);
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
        assert_eq!(
            export_dpo(&recs, 1, false).pairs.len(),
            1,
            "default caps at 1 pair/question"
        );
        assert_eq!(
            export_dpo(&recs, 2, false).pairs.len(),
            2,
            "max_pairs=2 yields two pairs"
        );
        // Can't exceed the smaller class size.
        assert_eq!(export_dpo(&recs, 9, false).pairs.len(), 2);
    }

    #[test]
    fn empty_input_yields_empty_output_no_panic() {
        let recs: Vec<TrajectoryRecord> = Vec::new();
        assert!(export_sft(&recs, SftShape::Messages, true, false)
            .rows
            .is_empty());
        let out = export_dpo(&recs, 1, false);
        assert!(out.pairs.is_empty());
        assert_eq!(out.questions_skipped, 0);
        assert_eq!(out.plateau_contrastive, 0);
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
        assert!(load_trajectories_for_tag(&runs, "missing")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn to_jsonl_is_one_object_per_line() {
        let rows = vec![json!({"a": 1}), json!({"b": 2})];
        let text = to_jsonl(&rows);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "{\"a\":1}");
    }

    // ---- Part B: plateau-turn detection + plateau-focused DPO pairing -----------------------

    /// A synthetic tool-call assistant turn (arithmetic placeholders only, no corpus values).
    fn tool_call_msg(call_id: &str) -> Value {
        json!({"role": "assistant", "content": null, "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "search", "arguments": "{\"q\":\"n\"}"}}
        ]})
    }

    /// A plain (non-plateau) tool result.
    fn tool_result_msg(call_id: &str) -> Value {
        json!({"role": "tool", "tool_call_id": call_id, "content": "some result body"})
    }

    /// A tool result carrying the stable plateau-marker substring (see
    /// `retrieval_progress::plateau_marker` / `PLATEAU_MARKER_SUBSTR`) — this is what
    /// `apply_plateau_render` (Part A) feeds back into the trajectory.
    fn plateau_tool_msg(call_id: &str) -> Value {
        json!({"role": "tool", "tool_call_id": call_id,
               "content": "[retrieval: 3 unique results gathered; 0 new over the last 3 searches — gain has plateaued]"})
    }

    fn answer_msg(answer: &str) -> Value {
        json!({"role": "assistant", "content": answer})
    }

    /// Build a synthetic trajectory record from an explicit `messages` sequence (a variant of
    /// `rec` for tests that need to control plateau/spiral shape precisely).
    fn rec_msgs(
        id: &str,
        verdict: Verdict,
        user: &str,
        answer: &str,
        messages: Vec<Value>,
    ) -> TrajectoryRecord {
        TrajectoryRecord {
            id: id.to_string(),
            question: user.to_string(),
            model: "test-model".to_string(),
            tools: json!([{"type": "function", "function": {"name": "search"}}]),
            messages,
            answer: answer.to_string(),
            verdict,
            hop_type: String::new(),
        }
    }

    fn seed(user: &str) -> Vec<Value> {
        vec![
            json!({"role": "system", "content": "You answer questions."}),
            json!({"role": "user", "content": user}),
        ]
    }

    #[test]
    fn plateau_turn_detection_and_spiral_predicate() {
        // No plateau turn at all -> has_plateau_turn false, spiraled false.
        let mut m = seed("q");
        m.push(tool_call_msg("c1"));
        m.push(tool_result_msg("c1"));
        m.push(answer_msg("a"));
        assert!(!has_plateau_turn(&m));
        assert!(!spiraled_past_plateau(&m));

        // Plateau turn, but the reader answered right after (no assistant tool_calls after it) ->
        // has_plateau_turn true, spiraled false.
        let mut m = seed("q");
        m.push(tool_call_msg("c1"));
        m.push(plateau_tool_msg("c1"));
        m.push(answer_msg("a"));
        assert!(has_plateau_turn(&m));
        assert!(
            !spiraled_past_plateau(&m),
            "answering right after the signal is not a spiral"
        );

        // Plateau turn, THEN another tool call -> spiraled true (kept searching past the signal).
        let mut m = seed("q");
        m.push(tool_call_msg("c1"));
        m.push(plateau_tool_msg("c1"));
        m.push(tool_call_msg("c2"));
        m.push(tool_result_msg("c2"));
        m.push(answer_msg("a"));
        assert!(has_plateau_turn(&m));
        assert!(spiraled_past_plateau(&m));
    }

    #[test]
    fn dpo_plateau_focus_prefers_spiraled_rejected_and_never_uses_correct_as_rejected() {
        // (i) Correct with a plateau turn, answered right after (not spiraled).
        let mut i_msgs = seed("Q1");
        i_msgs.push(tool_call_msg("c1"));
        i_msgs.push(plateau_tool_msg("c1"));
        i_msgs.push(answer_msg("ANSWER: correct-plain"));
        let i = rec_msgs(
            "q1",
            Verdict::Correct,
            "Q1",
            "ANSWER: correct-plain",
            i_msgs,
        );

        // (ii) Wrong that spiraled past a plateau.
        let mut ii_msgs = seed("Q1");
        ii_msgs.push(tool_call_msg("c1"));
        ii_msgs.push(plateau_tool_msg("c1"));
        ii_msgs.push(tool_call_msg("c2"));
        ii_msgs.push(tool_result_msg("c2"));
        ii_msgs.push(answer_msg("ANSWER: wrong-spiraled"));
        let ii = rec_msgs(
            "q1",
            Verdict::Wrong,
            "Q1",
            "ANSWER: wrong-spiraled",
            ii_msgs,
        );

        // (iii) Wrong with no plateau turn at all.
        let mut iii_msgs = seed("Q1");
        iii_msgs.push(tool_call_msg("c1"));
        iii_msgs.push(tool_result_msg("c1"));
        iii_msgs.push(answer_msg("ANSWER: wrong-no-plateau"));
        let iii = rec_msgs(
            "q1",
            Verdict::Wrong,
            "Q1",
            "ANSWER: wrong-no-plateau",
            iii_msgs,
        );

        // (iv) Correct that ALSO searched past a plateau turn — CRITICAL: must remain eligible as
        // chosen and must NEVER be used as rejected, because the judge verdict is Correct.
        let mut iv_msgs = seed("Q1");
        iv_msgs.push(tool_call_msg("c1"));
        iv_msgs.push(plateau_tool_msg("c1"));
        iv_msgs.push(tool_call_msg("c2"));
        iv_msgs.push(tool_result_msg("c2"));
        iv_msgs.push(answer_msg("ANSWER: correct-spiraled"));
        let iv = rec_msgs(
            "q1",
            Verdict::Correct,
            "Q1",
            "ANSWER: correct-spiraled",
            iv_msgs,
        );

        // A second question with only a non-spiraled Wrong -> under --dpo-focus plateau it must be
        // skipped for lacking a spiraled rejected (even though it has a plain Correct+Wrong pair).
        let q2_correct = rec("q2", Verdict::Correct, "Q2", "ANSWER: 2-correct");
        let mut q2_wrong_msgs = seed("Q2");
        q2_wrong_msgs.push(tool_call_msg("c1"));
        q2_wrong_msgs.push(tool_result_msg("c1"));
        q2_wrong_msgs.push(answer_msg("ANSWER: 2-wrong"));
        let q2_wrong = rec_msgs("q2", Verdict::Wrong, "Q2", "ANSWER: 2-wrong", q2_wrong_msgs);

        let recs = vec![
            i.clone(),
            ii.clone(),
            iii.clone(),
            iv.clone(),
            q2_correct,
            q2_wrong,
        ];

        // Default (no focus): rejected pool biases the spiraled Wrong (ii) to the front over the
        // non-spiraled Wrong (iii); q2 still pairs (falls back to its only Wrong).
        let out = export_dpo(&recs, 1, false);
        assert_eq!(out.pairs.len(), 2, "both q1 and q2 pair without focus");
        assert_eq!(out.questions_skipped, 0);
        let q1_pair = out
            .pairs
            .iter()
            .find(|p| {
                p["chosen"][0]["content"] == "ANSWER: correct-plain"
                    || p["chosen"][0]["content"] == "ANSWER: correct-spiraled"
            })
            .expect("q1 pair present");
        assert_eq!(
            q1_pair["rejected"][0]["content"], "ANSWER: wrong-spiraled",
            "the spiraled Wrong is biased to the front even without --dpo-focus"
        );
        assert!(
            out.pairs
                .iter()
                .all(|p| p["rejected"][0]["content"] != "ANSWER: correct-plain"
                    && p["rejected"][0]["content"] != "ANSWER: correct-spiraled"),
            "a Correct trajectory (i/iv) must never appear as rejected, spiraled or not"
        );
        assert_eq!(
            out.plateau_contrastive, 1,
            "only q1's pair is plateau-contrastive"
        );

        // --dpo-focus plateau: q1 pairs using the spiraled Wrong; q2 (no spiraled Wrong) is skipped.
        let focused = export_dpo(&recs, 1, true);
        assert_eq!(
            focused.pairs.len(),
            1,
            "only q1 has a spiraled-Wrong rejected candidate"
        );
        assert_eq!(
            focused.questions_skipped, 1,
            "q2 is skipped for lacking a spiraled rejected"
        );
        assert_eq!(focused.plateau_contrastive, 1);
        assert_eq!(
            focused.pairs[0]["rejected"][0]["content"],
            "ANSWER: wrong-spiraled"
        );
        assert!(
            focused.pairs[0]["chosen"][0]["content"] == "ANSWER: correct-plain"
                || focused.pairs[0]["chosen"][0]["content"] == "ANSWER: correct-spiraled",
            "chosen must be one of q1's two Correct trajectories"
        );
    }

    #[test]
    fn sft_prefer_signal_stable_partitions_signal_reaction_first() {
        // First in input order: a Correct trajectory with NO plateau turn.
        let plain = rec("q1", Verdict::Correct, "Q1", "ANSWER: plain");
        // Second: a Correct trajectory WITH a plateau turn (the "reacted to signal" demonstration).
        let mut signal_msgs = seed("Q2");
        signal_msgs.push(tool_call_msg("c1"));
        signal_msgs.push(plateau_tool_msg("c1"));
        signal_msgs.push(answer_msg("ANSWER: signal"));
        let signal = rec_msgs("q2", Verdict::Correct, "Q2", "ANSWER: signal", signal_msgs);

        let recs = vec![plain, signal];

        // Default: today's input order (plain first).
        let out = export_sft(&recs, SftShape::Messages, false, false);
        assert_eq!(out.signal_reaction, 1);
        assert_eq!(
            out.rows[0]["messages"].as_array().unwrap().last().unwrap()["content"],
            "ANSWER: plain"
        );
        assert_eq!(
            out.rows[1]["messages"].as_array().unwrap().last().unwrap()["content"],
            "ANSWER: signal"
        );

        // --sft-prefer-signal: the plateau-turn trajectory is partitioned to the front.
        let out2 = export_sft(&recs, SftShape::Messages, false, true);
        assert_eq!(out2.signal_reaction, 1);
        assert_eq!(
            out2.rows[0]["messages"].as_array().unwrap().last().unwrap()["content"],
            "ANSWER: signal"
        );
        assert_eq!(
            out2.rows[1]["messages"].as_array().unwrap().last().unwrap()["content"],
            "ANSWER: plain"
        );
    }
}
